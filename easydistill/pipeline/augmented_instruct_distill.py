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

"""Augmented instruction distillation: seed prompts -> expansion -> refinement -> SFT."""

from typing import Any, Dict, List

from .base import BaseDistillationPipeline
from .common import (
    _extract_strings,
    _format_synthesis_outputs,
    _run_distill_stage,
    _run_instruction_balance_stage,
    _run_synthesis_stage,
)


class AugmentedInstructDistillPipeline(BaseDistillationPipeline):
    """Chain instruction synthesis stages and a final distillation stage into one run.

    Example pipeline config:
      pipeline:
        - stage: instruction_expansion
          config:
            num_in_context_samples: 2
            num_output_samples: 5
          output_path: outputs/stage1.jsonl
        - stage: instruction_refinement
          config:
            max_workers: 3
          output_path: outputs/stage2.jsonl
        - stage: instruct_distill
          config:
            temperature: 0.7
            max_tokens: 512
            max_workers: 5
    """

    _last_stage = "instruct_distill"

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
        elif stage_name == "instruct_distill":
            data = _run_distill_stage(
                self.backend,
                stage_config,
                data,
                self.generation_config,
                self.sft_config,
            )
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
