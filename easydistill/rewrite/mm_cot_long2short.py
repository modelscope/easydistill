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

"""Multi-modal CoT long-to-short simplification operator."""

from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval._common import approx_token_count
from easydistill.operators.cot.utils import extract_between_tags
from easydistill.operators.mm.cot.base import PromptMMCoTOperator
from easydistill.prompts import resolve_prompt
from easydistill.utils import build_multimodal_user_content, format_prompt_safely


class MMCoTLong2ShortOperator(PromptMMCoTOperator):
    """Simplify a multi-modal chain-of-thought reasoning process.

    Configurable fields mirror the text CoT long2short operator.

    Input: list of dicts with keys:
      - instruction: text problem/prompt.
      - images: list of image references.
      - response: the full CoT output (reasoning + solution).

    Output dict:
      - instruction: the original text problem.
      - images: the original image references.
      - response: the simplified reasoning process.
      - original_response: the original answer passed in.
      - compression_ratio: simplified length / original length.
    """

    name = "mm_cot_long2short"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config, default_file="configs/prompts/cot_long2short_prompt.txt"
        )
        self.max_length = self.config.get("max_length")
        self.target_compression_ratio = self.config.get("target_compression_ratio")
        self.verify_solution_tags = self.config.get("verify_solution_tags", False)
        self.system_prompt = self.config.get("system_prompt")

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
                    id=f"mm_cot_l2s_{idx}",
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
        simplified = (result.response or "").strip()
        if not simplified:
            return None

        if self.max_length is not None and len(simplified) > self.max_length:
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

        output = dict(result.request.metadata)
        output.update(
            {
                "instruction": problem,
                "images": images,
                "response": simplified,
                "original_response": original_answer,
                "original_tokens": original_tokens,
                "simplified_tokens": simplified_tokens,
                "compression_ratio": compression_ratio,
            }
        )
        if self.target_compression_ratio is not None:
            output["target_compression_ratio"] = self.target_compression_ratio
        return output
