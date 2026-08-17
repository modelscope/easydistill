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

"""Multi-modal CoT short-to-long extension operator."""

from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval._common import approx_token_count
from easydistill.operators.cot.utils import extract_between_tags
from easydistill.operators.mm.cot.base import PromptMMCoTOperator
from easydistill.prompts import resolve_prompt
from easydistill.utils import build_multimodal_user_content, format_prompt_safely


class MMCoTShort2LongOperator(PromptMMCoTOperator):
    """Extend a multi-modal chain-of-thought reasoning process.

    Configurable fields mirror the text CoT short2long operator.

    Input: list of dicts with keys:
      - instruction: text problem/prompt.
      - images: list of image references.
      - response: the CoT output that should be extended.

    Output dict:
      - instruction: the original text problem.
      - images: the original image references.
      - response: the extended reasoning process.
      - original_response: the original answer passed in.
      - expansion_ratio: extended length / original length.
      - step_count: approximate number of reasoning steps.
    """

    name = "mm_cot_short2long"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_short2long_prompt.txt"
        )
        self.min_steps = self.config.get("min_steps")
        self.target_expansion_ratio = self.config.get("target_expansion_ratio")
        self.verify_solution_tags = self.config.get("verify_solution_tags", False)
        self.system_prompt = self.config.get("system_prompt")

    def _count_steps(self, text: str) -> int:
        if not text:
            return 0
        return len([s for s in text.split("\n\n") if s.strip()])

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            problem = row.get("instruction") or row.get("problem") or ""
            answer = row.get("response") or row.get("answer") or ""
            images = row.get("images") or []
            prompt = format_prompt_safely(self.prompt_template, problem=problem, answer=answer)
            content = build_multimodal_user_content(prompt, images)
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {"instruction", "problem", "answer", "response", "images"}
            }
            metadata.update({"problem": problem, "answer": answer, "images": images})
            requests.append(
                GenerationRequest(
                    id=f"mm_cot_s2l_{idx}",
                    instruction=content,
                    system_prompt=self.system_prompt,
                    metadata=metadata,
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        problem = result.request.metadata.get("problem", "")
        original_answer = result.request.metadata.get("answer", "")
        images = result.request.metadata.get("images", [])
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

        output = dict(result.request.metadata)
        output.update(
            {
                "instruction": problem,
                "images": images,
                "response": extended,
                "original_response": original_answer,
                "original_tokens": original_tokens,
                "extended_tokens": extended_tokens,
                "expansion_ratio": expansion_ratio,
                "step_count": self._count_steps(extended),
            }
        )
        if self.min_steps is not None:
            output["min_steps"] = self.min_steps
        if self.target_expansion_ratio is not None:
            output["target_expansion_ratio"] = self.target_expansion_ratio
        return output
