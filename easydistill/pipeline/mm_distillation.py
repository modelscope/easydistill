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

"""Best-practice multi-modal black-box KD pipeline."""

from typing import Any, Dict, List

from easydistill.eval import MMInstructionFollowingEvaluator
from easydistill.operators.mm import MMGenerationOperator

from .base import BaseDistillationPipeline
from .common import run_build_sft_stage, run_eval_stage, run_quality_filter_stage

_DEFAULT_EVAL_METRICS = [
    "informativeness",
    "helpfulness",
    "generalization",
    "correctness",
]


def _run_mm_generation_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate teacher responses for multi-modal instruction rows."""
    operator = MMGenerationOperator(backend=backend, config=stage_config)
    return operator.run(data)


class MMDistillationPipeline(BaseDistillationPipeline):
    """End-to-end advanced pipeline for multi-modal black-box KD.

    The recommended flow is:
      1. mm_instruct_distill (teacher response generation)
      2. mm_instruct_eval (LLM-as-judge quality scoring)
      3. quality_filter (keep only the best data)
      4. build_sft (final SFT dataset with multi-modal messages)

    The last stage must be `build_sft`.
    """

    _last_stage = "build_sft"
    _default_eval_metrics = _DEFAULT_EVAL_METRICS

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == "mm_instruct_distill":
            data = _run_mm_generation_stage(self.backend, stage_config, data)
        elif stage_name == "mm_instruct_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            data = run_eval_stage(
                self.backend,
                eval_cfg,
                data,
                MMInstructionFollowingEvaluator,
                image_key="images",
            )
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            data = run_build_sft_stage(data, self.generation_config, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
