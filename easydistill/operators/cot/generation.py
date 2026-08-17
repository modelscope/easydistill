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

"""CoT generation operator."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely
from easydistill.utils.image import content_has_images

from .base import PromptCoTOperator
from .utils import extract_cot_sections

logger = logging.getLogger(__name__)


class CoTGenerationOperator(PromptCoTOperator[str, Dict[str, str]]):
    """Generate a chain-of-thought reasoning and final solution for a problem.

    Configurable fields:
      - prompt_template: prompt template with {problem} placeholder.
      - prompt_template_file: path to a text file containing the prompt template.
      - temperature, max_tokens, max_workers, show_progress.

    Output dict:
      - instruction: the original problem.
      - response: the raw model output.
      - thought: extracted thought section.
      - solution: extracted solution section.
    """

    name = "cot_distill"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_generation_prompt.txt"
        )

    def _build_requests(self, inputs: List[str]) -> List[GenerationRequest]:
        requests = []
        for idx, problem in enumerate(inputs):
            if content_has_images(problem):
                logger.warning(
                    "Dropping image content from problem %d in text-only CoT generation.",
                    idx,
                )
            prompt = format_prompt_safely(self.prompt_template, problem=problem)
            requests.append(
                GenerationRequest(
                    id=f"cot_gen_{idx}",
                    instruction=prompt,
                    metadata={"problem": problem, "task": "cot_distill"},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, str]]:
        problem = result.request.metadata.get("problem", "")
        raw_response = result.response or ""
        thought, solution = extract_cot_sections(raw_response)
        if not raw_response.strip():
            return None
        return {
            "instruction": problem,
            "response": raw_response,
            "thought": thought or "",
            "solution": solution or "",
        }
