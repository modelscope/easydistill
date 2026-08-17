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

"""Multi-modal CoT generation operator."""

from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import build_multimodal_user_content, format_prompt_safely

from ...cot.utils import extract_cot_sections
from .base import PromptMMCoTOperator


class MMCoTGenerationOperator(PromptMMCoTOperator):
    """Generate a chain-of-thought reasoning and solution for a multi-modal problem.

    Configurable fields:
      - prompt_template: prompt template with {problem} placeholder.
      - prompt_template_file: path to a text file containing the prompt template.
      - system_prompt: optional system message.
      - temperature, max_tokens, max_workers, show_progress.

    Input: list of dicts with keys:
      - instruction: text problem/prompt.
      - images: list of image references.

    Output dict:
      - instruction: the original text problem.
      - images: the original image references.
      - response: the raw model output.
      - thought: extracted thought section.
      - solution: extracted solution section.
    """

    name = "mm_cot_distill"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_generation_prompt.txt"
        )
        self.system_prompt = self.config.get("system_prompt")

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            problem = row.get("instruction") or row.get("problem") or ""
            images = row.get("images") or []
            prompt = format_prompt_safely(self.prompt_template, problem=problem)
            content = build_multimodal_user_content(prompt, images)
            metadata = {
                k: v for k, v in row.items() if k not in {"instruction", "problem", "images"}
            }
            metadata["instruction"] = problem
            metadata["images"] = images
            requests.append(
                GenerationRequest(
                    id=f"mm_cot_gen_{idx}",
                    instruction=content,
                    system_prompt=self.system_prompt,
                    metadata=metadata,
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        problem = result.request.metadata.get("instruction", "")
        images = result.request.metadata.get("images", [])
        raw_response = result.response or ""
        if not raw_response.strip():
            return None
        thought, solution = extract_cot_sections(raw_response)
        output = dict(result.request.metadata)
        output.update(
            {
                "instruction": problem,
                "images": images,
                "response": raw_response,
                "thought": thought or "",
                "solution": solution or "",
            }
        )
        return output
