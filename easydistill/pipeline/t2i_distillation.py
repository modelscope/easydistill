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

"""T2I (text-to-image) distillation pipeline."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.backends.t2i_base import T2IBackend
from easydistill.eval import T2IImageEvaluator
from easydistill.operators.t2i import T2IGenerationOperator, T2IPromptOptimizer

from .base import BaseDistillationPipeline
from .common import (
    run_build_t2i_sft_stage,
    run_quality_filter_stage,
    run_t2i_eval_stage,
)

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_METRICS = [
    "prompt_consistency",
    "aesthetic_quality",
    "detail_richness",
    "artifact_absence",
]


def _run_prompt_optimize_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Optimize seed prompts into detailed T2I prompts."""
    operator = T2IPromptOptimizer(backend=backend, config=stage_config)
    return operator.run(data)


def _run_t2i_generate_stage(
    t2i_backend: T2IBackend,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate images from optimized prompts using the T2I backend."""
    operator = T2IGenerationOperator(backend=t2i_backend, config=stage_config)
    return operator.run(data)


class T2IDistillationPipeline(BaseDistillationPipeline):
    """End-to-end T2I text-to-image distillation pipeline.

    The recommended flow is:
      1. (optional) prompt_optimize — enhance seed prompts via LLM
      2. t2i_generate — generate images from prompts via T2I backend
      3. (optional) t2i_eval — VLM-as-judge quality scoring
      4. (optional) quality_filter — keep only the best data
      5. build_t2i_sft — final multi-modal SFT dataset

    The last stage must be ``build_t2i_sft``.

    Three backends are used:
      - ``backend`` (ModelBackend): for prompt optimization (text model).
      - ``eval_backend`` (ModelBackend / VLM): for image evaluation.
        If not provided, falls back to ``backend``.
      - ``t2i_backend`` (T2IBackend): for image generation.
    """

    _last_stage = "build_t2i_sft"
    _default_eval_metrics = _DEFAULT_EVAL_METRICS

    def __init__(
        self,
        backend: Any,
        t2i_backend: T2IBackend,
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
        self.t2i_backend = t2i_backend
        self.eval_backend = eval_backend or backend

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == "prompt_optimize":
            data = _run_prompt_optimize_stage(self.backend, stage_config, data)
        elif stage_name == "t2i_generate":
            gen_cfg = {**self.generation_config, **stage_config}
            data = _run_t2i_generate_stage(self.t2i_backend, gen_cfg, data)
        elif stage_name == "t2i_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            data = run_t2i_eval_stage(
                self.eval_backend,
                eval_cfg,
                data,
                T2IImageEvaluator,
            )
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_t2i_sft":
            data = run_build_t2i_sft_stage(data, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
