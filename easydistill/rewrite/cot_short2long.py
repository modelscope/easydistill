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

"""CoT short-to-long extension operator."""

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


class CoTShort2LongOperator(PromptCoTOperator[Tuple[str, str], Dict[str, str]]):
    """Extend an existing chain-of-thought reasoning process.

    Configurable fields:
      - prompt_template: prompt template with {problem} and {answer} placeholders.
      - prompt_template_file: path to a text file containing the prompt template.
      - min_steps: optional minimum number of reasoning steps expected in the
        extended output (reported in metadata, not enforced by retry).
      - target_expansion_ratio: optional target ratio of extended / original
        length (e.g., 2.0 means twice as long). Reported in metadata.
      - verify_solution_tags: if True, drop extended outputs that do not
        preserve the text between <|begin_of_solution|> and <|end_of_solution|>
        from the original answer.
      - temperature, max_tokens, max_workers, show_progress.

    Input: list of (problem, answer) tuples. `answer` is the full CoT output
    that should be extended with more details.

    Output dict:
      - instruction: the original problem.
      - response: the extended reasoning process.
      - original_response: the original answer passed in.
      - original_tokens: approximate token count of the original answer.
      - extended_tokens: approximate token count of the extended answer.
      - expansion_ratio: extended_tokens / original_tokens.
      - step_count: approximate number of reasoning steps (split by '\n\n').
    """

    name = "cot_short2long"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_short2long_prompt.txt"
        )
        self.min_steps = self.config.get("min_steps")
        self.target_expansion_ratio = self.config.get("target_expansion_ratio")
        self.verify_solution_tags = self.config.get("verify_solution_tags", False)

    def _count_steps(self, text: str) -> int:
        """Count reasoning steps separated by blank lines."""
        if not text:
            return 0
        return len([s for s in text.split("\n\n") if s.strip()])

    def _build_requests(self, inputs: List[Tuple[str, str]]) -> List[GenerationRequest]:
        requests = []
        for idx, (problem, answer) in enumerate(inputs):
            if content_has_images(problem):
                logger.warning(
                    "Dropping image content from problem %d in text-only short2long.",
                    idx,
                )
            if content_has_images(answer):
                logger.warning(
                    "Dropping image content from answer %d in text-only short2long.",
                    idx,
                )
            prompt = format_prompt_safely(self.prompt_template, problem=problem, answer=answer)
            requests.append(
                GenerationRequest(
                    id=f"cot_s2l_{idx}",
                    instruction=prompt,
                    metadata={
                        "problem": problem,
                        "answer": answer,
                        "task": "cot_short2long",
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, str]]:
        problem = result.request.metadata.get("problem", "")
        original_answer = result.request.metadata.get("answer", "")
        extended = (result.response or "").strip()
        if not extended:
            return None

        original_tokens = approx_token_count(original_answer)
        extended_tokens = approx_token_count(extended)
        expansion_ratio = round(extended_tokens / original_tokens, 4)

        if self.verify_solution_tags:
            original_solution = extract_between_tags(
                original_answer, "<|begin_of_solution|>", "<|end_of_solution|>"
            )
            if original_solution and original_solution not in extended:
                return None

        output: Dict[str, Any] = {
            "instruction": problem,
            "response": extended,
            "original_response": original_answer,
            "original_tokens": original_tokens,
            "extended_tokens": extended_tokens,
            "expansion_ratio": expansion_ratio,
            "step_count": self._count_steps(extended),
        }
        if self.min_steps is not None:
            output["min_steps"] = self.min_steps
        if self.target_expansion_ratio is not None:
            output["target_expansion_ratio"] = self.target_expansion_ratio
        return output
