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

"""CoT long-to-short simplification operator."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval._common import approx_token_count
from easydistill.operators.cot.base import PromptCoTOperator
from easydistill.operators.cot.utils import extract_between_tags
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely
from easydistill.utils.image import content_has_images

logger = logging.getLogger(__name__)


class CoTLong2ShortOperator(PromptCoTOperator[Tuple[str, str], Dict[str, str]]):
    """Simplify an existing chain-of-thought reasoning process.

    Configurable fields:
      - prompt_template: prompt template with {problem} and {answer} placeholders.
      - prompt_template_file: path to a text file containing the prompt template.
      - max_length: optional maximum character length for the simplified output.
      - target_compression_ratio: optional target ratio of simplified / original
        length (e.g., 0.5 means half the length). Reported in metadata; the
        operator does not retry if the target is missed.
      - verify_solution_tags: if True, drop simplified outputs that do not
        preserve the text between <|begin_of_solution|> and <|end_of_solution|>
        from the original answer.
      - temperature, max_tokens, max_workers, show_progress.

    Input: list of (problem, answer) tuples. `answer` is the full CoT output
    (reasoning + solution) that should be simplified.

    Output dict:
      - instruction: the original problem.
      - response: the simplified reasoning process.
      - original_response: the original answer passed in.
      - original_tokens: approximate token count of the original answer.
      - simplified_tokens: approximate token count of the simplified answer.
      - compression_ratio: simplified_tokens / original_tokens.
    """

    name = "cot_long2short"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_long2short_prompt.txt"
        )
        self.max_length = self.config.get("max_length")
        self.target_compression_ratio = self.config.get("target_compression_ratio")
        self.verify_solution_tags = self.config.get("verify_solution_tags", False)

    def _build_requests(self, inputs: List[Tuple[str, str]]) -> List[GenerationRequest]:
        requests = []
        for idx, (problem, answer) in enumerate(inputs):
            if content_has_images(problem):
                logger.warning(
                    "Dropping image content from problem %d in text-only long2short.",
                    idx,
                )
            if content_has_images(answer):
                logger.warning(
                    "Dropping image content from answer %d in text-only long2short.",
                    idx,
                )
            prompt = format_prompt_safely(self.prompt_template, problem=problem, answer=answer)
            requests.append(
                GenerationRequest(
                    id=f"cot_l2s_{idx}",
                    instruction=prompt,
                    metadata={
                        "problem": problem,
                        "answer": answer,
                        "task": "cot_long2short",
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, str]]:
        problem = result.request.metadata.get("problem", "")
        original_answer = result.request.metadata.get("answer", "")
        simplified = (result.response or "").strip()
        if not simplified:
            return None

        if self.max_length is not None and len(simplified) > self.max_length:
            # Hard truncate to max_length; this is a safety guard, not ideal but
            # prevents obviously over-long outputs from polluting the dataset.
            simplified = simplified[: self.max_length].rsplit(" ", 1)[0].strip()
            if not simplified:
                return None

        original_tokens = approx_token_count(original_answer)
        simplified_tokens = approx_token_count(simplified)
        compression_ratio = round(simplified_tokens / original_tokens, 4)

        if self.verify_solution_tags:
            original_solution = extract_between_tags(
                original_answer, "<|begin_of_solution|>", "<|end_of_solution|>"
            )
            if original_solution and original_solution not in simplified:
                return None

        output: Dict[str, Any] = {
            "instruction": problem,
            "response": simplified,
            "original_response": original_answer,
            "original_tokens": original_tokens,
            "simplified_tokens": simplified_tokens,
            "compression_ratio": compression_ratio,
        }
        if self.target_compression_ratio is not None:
            output["target_compression_ratio"] = self.target_compression_ratio
        return output
