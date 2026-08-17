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

"""Instruction-response extraction operator."""

from typing import Any, Dict, List, Optional, Tuple

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.synthesis.base import PromptSynthesisOperator
from easydistill.operators.synthesis.utils import extract_instruction_response
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely


def _filter_pair(pair: Tuple[Optional[str], Optional[str]]) -> Optional[Tuple[str, str]]:
    """Return the pair only if both instruction and response are non-empty."""
    instruction, response = pair
    if instruction and response:
        return (instruction, response)
    return None


class InstructionResponseExtractionOperator(PromptSynthesisOperator[str, Tuple[str, str]]):
    """Extract (instruction, response) pairs from raw text.

    Configurable fields:
      - prompt_template: prompt template with {text} placeholder.
      - prompt_template_file: path to a text file containing the prompt template.
      - use_llm: if False, only do regex extraction without calling LLM (default True).
    """

    name = "instruction_response_extraction"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/extraction_prompt.txt"
        )
        self.use_llm = bool(self.config.get("use_llm", True))

    def _build_requests(self, inputs: List[str]) -> List[GenerationRequest]:
        if not self.use_llm:
            return []
        requests = []
        for idx, text in enumerate(inputs):
            prompt = format_prompt_safely(self.prompt_template, text=text)
            requests.append(
                GenerationRequest(
                    id=f"extract_{idx}",
                    instruction=prompt,
                    metadata={"task": "instruction_response_extraction", "text": text},
                )
            )
        return requests

    def run(self, inputs: List[str]) -> List[Tuple[str, str]]:
        """Extract pairs. If use_llm is False, parse inputs directly."""
        if not self.use_llm:
            outputs: List[Tuple[str, str]] = []
            for text in inputs:
                pair = _filter_pair(extract_instruction_response(text))
                if pair is not None:
                    outputs.append(pair)
            return outputs
        return super().run(inputs)

    def _parse_result(self, result: GenerationResult) -> Optional[Tuple[str, str]]:
        # First try to parse the LLM output.
        pair = _filter_pair(extract_instruction_response(result.response))
        if pair is not None:
            return pair
        # Fallback: parse the original input text if the LLM failed to wrap tags.
        original_text = result.request.metadata.get("text", "")
        if original_text:
            pair = _filter_pair(extract_instruction_response(original_text))
            if pair is not None:
                return pair
        return None
