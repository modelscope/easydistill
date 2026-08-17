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

"""Best-practice multi-modal CoT distillation pipeline."""

from typing import Any, Dict, List, Type, cast

from easydistill.eval import MMCoTEvaluator
from easydistill.operators.mm import MMCoTGenerationOperator
from easydistill.rewrite import (
    MMCoTLong2ShortOperator,
    MMCoTShort2LongOperator,
)

from .base import BaseDistillationPipeline
from .common import run_build_sft_stage, run_eval_stage, run_quality_filter_stage

_MMCOT_OPERATORS: Dict[str, Type[Any]] = {
    "mm_cot_distill": MMCoTGenerationOperator,
    "mm_cot_long2short": MMCoTLong2ShortOperator,
    "mm_cot_short2long": MMCoTShort2LongOperator,
}

_DEFAULT_EVAL_METRICS = [
    "reasoning_verbosity",
    "cognitive_difficulty",
    "logical_correctness",
]


def _run_mmcot_stage(
    backend: Any,
    stage_name: str,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run an MMCoT rewrite/generation stage."""
    op_cls = _MMCOT_OPERATORS[stage_name]
    operator = op_cls(backend=backend, config=stage_config)
    return cast(List[Dict[str, Any]], operator.run(data))


class MMCoTDistillationPipeline(BaseDistillationPipeline):
    """End-to-end advanced pipeline for multi-modal CoT distillation.

    The recommended flow is:
      1. mm_cot_distill (teacher MMCoT generation)
      2. (optional) mm_cot_long2short or mm_cot_short2long
      3. mm_cot_eval (LLM-as-judge quality scoring)
      4. quality_filter (keep only the best data)
      5. build_sft (final SFT dataset with multi-modal messages)

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
        if stage_name in _MMCOT_OPERATORS:
            data = _run_mmcot_stage(self.backend, stage_name, stage_config, data)
        elif stage_name == "mm_cot_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            data = run_eval_stage(
                self.backend,
                eval_cfg,
                data,
                MMCoTEvaluator,
                image_key="images",
            )
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            data = run_build_sft_stage(data, self.generation_config, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
