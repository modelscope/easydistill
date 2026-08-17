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

"""Instruction expansion operator."""

from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.synthesis.base import PromptSynthesisOperator
from easydistill.operators.synthesis.utils import (
    extract_tagged_answer,
    format_in_context_examples,
    sample_in_context_examples,
)
from easydistill.prompts import resolve_prompt


class InstructionExpansionOperator(PromptSynthesisOperator[str, str]):
    """Generate new instructions from seed instructions via in-context learning.

    Configurable fields:
      - prompt_template: prompt template with {examples} placeholder.
      - prompt_template_file: path to a text file containing the prompt template.
      - num_in_context_samples: number of seed examples to include per prompt.
      - num_output_samples: number of new instructions to generate in total.
      - seed: random seed for sampling in-context examples.
    """

    name = "instruction_expansion"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/expansion_prompt.txt"
        )
        self.num_in_context_samples = int(self.config.get("num_in_context_samples", 3))
        self.num_output_samples = int(self.config.get("num_output_samples", 10))
        seed = self.config.get("seed")
        self.seed = int(seed) if seed is not None else None

    def _build_requests(self, inputs: List[str]) -> List[GenerationRequest]:
        if len(inputs) < self.num_in_context_samples:
            raise ValueError(
                f"Need at least {self.num_in_context_samples} seed instructions, got {len(inputs)}"
            )
        requests = []
        for idx in range(self.num_output_samples):
            examples = sample_in_context_examples(
                inputs,
                self.num_in_context_samples,
                seed=(self.seed + idx) if self.seed is not None else None,
            )
            prompt = self.prompt_template.format(examples=format_in_context_examples(examples))
            requests.append(
                GenerationRequest(
                    id=f"expand_{idx}",
                    instruction=prompt,
                    metadata={"task": "instruction_expansion"},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[str]:
        answer = extract_tagged_answer(result.response, "answer")
        if answer:
            return answer
        # Fallback: treat the whole response as the instruction if no tag.
        text = result.response.strip()
        return text if text else None
