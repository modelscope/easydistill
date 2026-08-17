# Copyright 2026 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""T2V/I2V (text/image-to-video) distillation pipeline."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.backends.t2v_base import T2VBackend
from easydistill.eval import T2VVideoEvaluator
from easydistill.operators.t2v import T2VGenerationOperator, T2VPromptOptimizer
from easydistill.operators.t2v.resume import (
    eval_row_complete,
    generate_row_complete,
    load_completed_rows,
    merge_resumed,
    optimize_row_complete,
    split_pending,
)

from .base import BaseDistillationPipeline
from .common import (
    run_build_t2v_sft_stage,
    run_quality_filter_stage,
    run_t2v_eval_stage,
)

logger = logging.getLogger(__name__)

# Default metrics used by quality_filter's top-k/ratio averaging; must be a
# subset of configs/eval/t2v/vlm_dimensions.yaml.  first_frame_consistency
# is excluded because it only applies to I2V rows.
_DEFAULT_EVAL_METRICS = [
    "prompt_consistency",
    "visual_quality",
    "subject_consistency",
]

# Stages that support `resume: true` in their stage config, mapped to the
# predicate deciding whether a previously saved row is complete.  The cheap
# pure-CPU stages (quality_filter / build_t2v_sft) re-run in full.
_RESUME_PREDICATES = {
    "prompt_optimize": optimize_row_complete,
    "t2v_generate": generate_row_complete,
    "t2v_eval": eval_row_complete,
}


def _run_prompt_optimize_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Optimize seed prompts into detailed video-level T2V/I2V prompts."""
    operator = T2VPromptOptimizer(backend=backend, config=stage_config)
    return operator.run(data)


def _run_t2v_generate_stage(
    t2v_backend: T2VBackend,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate videos from optimized prompts using the T2V backend."""
    operator = T2VGenerationOperator(backend=t2v_backend, config=stage_config)
    return operator.run(data)


class T2VDistillationPipeline(BaseDistillationPipeline):
    """End-to-end T2V/I2V video distillation pipeline.

    Supports both plain T2V rows and I2V rows carrying a
    ``first_frame_image`` field; the two kinds may be mixed in one batch.
    The recommended flow is:
      1. (optional) prompt_optimize — enhance seed prompts via LLM/VLM
         (I2V rows are grounded in their first frame)
      2. t2v_generate — generate videos from prompts via T2V backend
      3. (optional) t2v_eval — precheck + frame-based VLM-as-judge scoring
      4. (optional) quality_filter — keep only the best data
      5. build_t2v_sft — final multi-modal SFT dataset

    The last stage must be ``build_t2v_sft``.

    Three backends are used:
      - ``backend`` (ModelBackend): for prompt optimization (text/VLM model).
      - ``eval_backend`` (ModelBackend / VLM): for video evaluation.
        If not provided, falls back to ``backend``.
      - ``t2v_backend`` (T2VBackend): for video generation.
    """

    _last_stage = "build_t2v_sft"
    _default_eval_metrics = _DEFAULT_EVAL_METRICS

    def __init__(
        self,
        backend: Any,
        t2v_backend: T2VBackend,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        eval_backend: Optional[Any] = None,
    ):
        super().__init__(
            backend=backend,
            pipeline_config=pipeline_config,
            dataset_config=dataset_config,
            generation_config=generation_config,
            sft_config=sft_config,
            eval_config=eval_config,
        )
        self.t2v_backend = t2v_backend
        self.eval_backend = eval_backend or backend

    def _stage_output_path(self, stage_name: str) -> Optional[str]:
        """Return the configured output_path of the named stage, if any."""
        for stage in self.pipeline_config:
            if stage.get("stage") == stage_name:
                return stage.get("output_path")
        return None

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        resume = bool(stage_config.get("resume"))
        predicate = _RESUME_PREDICATES.get(stage_name)
        if not (resume and predicate):
            return self._run_stage(stage_name, stage_config, data, eval_metrics)

        output_path = self._stage_output_path(stage_name)
        if not output_path:
            logger.warning(
                "Stage %s has resume enabled but no output_path; running in full.",
                stage_name,
            )
            return self._run_stage(stage_name, stage_config, data, eval_metrics)

        completed = load_completed_rows(output_path, predicate)
        done, pending = split_pending(data, completed)
        if done:
            logger.info(
                "Stage %s resume: reusing %d completed rows from %s; %d pending.",
                stage_name,
                len(done),
                output_path,
                len(pending),
            )
        if not pending:
            return merge_resumed(data, completed, [])
        new_rows = self._run_stage(stage_name, stage_config, pending, eval_metrics)
        return merge_resumed(data, completed, new_rows)

    def _run_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == "prompt_optimize":
            data = _run_prompt_optimize_stage(self.backend, stage_config, data)
        elif stage_name == "t2v_generate":
            gen_cfg = {**self.generation_config, **stage_config}
            if gen_cfg.get("resume"):
                # Row-level checkpointing into the stage output file enables
                # mid-stage crash recovery on the next resumed run.
                checkpoint = self._stage_output_path(stage_name)
                if checkpoint:
                    gen_cfg["checkpoint_path"] = checkpoint
            data = _run_t2v_generate_stage(self.t2v_backend, gen_cfg, data)
        elif stage_name == "t2v_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            eval_cfg.pop("resume", None)
            data = run_t2v_eval_stage(
                self.eval_backend,
                eval_cfg,
                data,
                T2VVideoEvaluator,
            )
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_t2v_sft":
            data = run_build_t2v_sft_stage(data, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
