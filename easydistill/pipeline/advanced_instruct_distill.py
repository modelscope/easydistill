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

"""Advanced instruction distillation pipeline."""

from typing import Any, Dict, List

from easydistill.data.models import GenerationRequest
from easydistill.eval import InstructionFollowingEvaluator
from easydistill.operators import TextGenerationOperator

from .base import BaseDistillationPipeline
from .common import (
    _extract_strings,
    _format_synthesis_outputs,
    _run_instruction_balance_stage,
    _run_synthesis_stage,
    run_build_sft_stage,
    run_eval_stage,
    run_quality_filter_stage,
)

_DEFAULT_EVAL_METRICS = [
    "informativeness",
    "helpfulness",
    "generalization",
    "correctness",
]


def _run_generation_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    global_generation_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate teacher responses for instruction rows."""
    gen_cfg = {**global_generation_config, **stage_config}
    generator = TextGenerationOperator(backend=backend, config=gen_cfg)

    requests = []
    for idx, row in enumerate(data):
        instruction = row.get("instruction")
        if not instruction:
            continue
        requests.append(
            GenerationRequest(
                id=str(row.get("id", idx)),
                instruction=instruction if isinstance(instruction, list) else str(instruction),
                system_prompt=row.get("system") or gen_cfg.get("system_prompt"),
                metadata={
                    "row_index": idx,
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in {"instruction", "system", "id", "response"}
                    },
                },
            )
        )

    results = generator.run(requests)
    result_by_index = {
        result.request.metadata["row_index"]: result
        for result in results
        if result is not None and result.request.metadata.get("row_index") is not None
    }
    output_rows = []
    for idx, row in enumerate(data):
        result = result_by_index.get(idx)
        if result is None:
            continue
        new_row = dict(row)
        new_row["response"] = result.response
        new_row["id"] = result.request.id
        output_rows.append(new_row)
    return output_rows


class AdvancedInstructDistillPipeline(BaseDistillationPipeline):
    """End-to-end advanced pipeline for instruction distillation.

    The recommended flow is:
      1. (optional) instruction_expansion or instruction_response_extraction
      2. (optional) instruction_refinement
      3. generate (teacher responses)
      4. instruct_eval (LLM-as-judge quality scoring)
      5. quality_filter (keep only the best data)
      6. build_sft (final SFT dataset)

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
        synthesis_stages = {
            "instruction_expansion",
            "instruction_refinement",
            "instruction_response_extraction",
        }
        if stage_name in synthesis_stages:
            input_key = "text" if stage_name == "instruction_response_extraction" else "instruction"
            inputs = _extract_strings(data, input_key)
            outputs = _run_synthesis_stage(self.backend, stage_name, stage_config, inputs)
            data = _format_synthesis_outputs(outputs, stage_name)
        elif stage_name == "instruction_balance":
            data = _run_instruction_balance_stage(self.backend, stage_config, data)
        elif stage_name == "generate":
            data = _run_generation_stage(
                self.backend,
                stage_config,
                data,
                self.generation_config,
            )
        elif stage_name == "instruct_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            data = run_eval_stage(
                self.backend, eval_cfg, data, InstructionFollowingEvaluator
            )
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            data = run_build_sft_stage(data, self.generation_config, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
