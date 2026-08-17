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

"""Instruction refinement operator."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.synthesis.base import PromptSynthesisOperator
from easydistill.operators.synthesis.utils import extract_tagged_answer
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely
from easydistill.utils.image import content_has_images

logger = logging.getLogger(__name__)


class InstructionRefinementOperator(PromptSynthesisOperator[str, str]):
    """Rewrite / optimize existing instructions.

    Configurable fields:
      - prompt_template: prompt template with {instruction} placeholder.
      - prompt_template_file: path to a text file containing the prompt template.
    """

    name = "instruction_refinement"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/refinement_prompt.txt"
        )

    def _build_requests(self, inputs: List[str]) -> List[GenerationRequest]:
        requests = []
        for idx, instruction in enumerate(inputs):
            if content_has_images(instruction):
                logger.warning(
                    "Dropping image content from instruction %d in text-only refinement.",
                    idx,
                )
            prompt = format_prompt_safely(self.prompt_template, instruction=instruction)
            requests.append(
                GenerationRequest(
                    id=f"refine_{idx}",
                    instruction=prompt,
                    metadata={"task": "instruction_refinement", "original": instruction},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[str]:
        answer = extract_tagged_answer(result.response, "answer")
        if answer:
            return answer
        text = result.response.strip()
        return text if text else None
