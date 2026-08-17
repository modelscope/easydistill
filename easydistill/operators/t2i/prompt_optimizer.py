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

"""T2I prompt optimization operator."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.operators.synthesis.utils import extract_tagged_answer
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely

logger = logging.getLogger(__name__)


class T2IPromptOptimizer(PromptGenerationOperator[Dict[str, Any], Dict[str, Any]]):
    """Optimize seed T2I prompts into detailed, high-quality prompts.

    Uses an LLM (typically a VLM or strong text model) to rewrite simple seed
    prompts into rich, descriptive prompts suitable for text-to-image models.

    Configurable fields:
      - prompt_template: template with {prompt} placeholder.
      - prompt_template_file: path to a file containing the template.
      - system_prompt: optional system message.
      - model_id, temperature, max_tokens, max_workers, show_progress.

    Input: list of dicts with key ``prompt`` (and optional ``id``).
    Output: list of dicts with keys ``id``, ``raw_prompt``, ``optimized_prompt``.
    """

    name = "t2i_prompt_optimize"
    default_max_tokens = 1024

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            default_file="configs/prompts/t2i_prompt_optimize_prompt.txt",
        )
        self.system_prompt = self.config.get("system_prompt")

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            prompt = row.get("prompt") or row.get("optimized_prompt") or ""
            if not prompt:
                logger.warning("Row %d has empty prompt, skipping.", idx)
                continue
            formatted = format_prompt_safely(self.prompt_template, prompt=prompt)
            requests.append(
                GenerationRequest(
                    id=str(row.get("id", idx)),
                    instruction=formatted,
                    system_prompt=self.system_prompt,
                    metadata={
                        "raw_prompt": prompt,
                        "row_index": idx,
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in {"prompt", "optimized_prompt", "id"}
                        },
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        answer = extract_tagged_answer(result.response, "answer")
        if not answer:
            # Fallback: use the stripped response.
            answer = result.response.strip()
        if not answer:
            return None
        meta = result.request.metadata
        output: Dict[str, Any] = {
            "id": result.request.id,
            "raw_prompt": meta.get("raw_prompt", ""),
            "optimized_prompt": answer,
        }
        # Preserve any extra metadata from the original row.
        for k, v in meta.items():
            if k not in {"raw_prompt", "row_index"}:
                output.setdefault(k, v)
        return output
