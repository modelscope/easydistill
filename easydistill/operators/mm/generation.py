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

"""Multi-modal black-box KD generation operator."""

import logging
from typing import Any, Dict, List, Optional, Union

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import build_multimodal_user_content, format_prompt_safely
from easydistill.utils.image import content_has_images

from ..prompt_base import PromptGenerationOperator

logger = logging.getLogger(__name__)


class MMGenerationOperator(PromptGenerationOperator[Dict[str, Any], Dict[str, Any]]):
    """Generate teacher responses for (image, text) instruction pairs.

    Configurable fields:
      - prompt_template: optional prompt template with {instruction} placeholder.
        If omitted, the instruction is sent directly as the user message.
      - prompt_template_file: path to a text file containing the prompt template.
      - system_prompt: optional system message.
      - model_id: model identifier passed to the backend.
      - temperature, max_tokens, max_workers, show_progress.

    Input: list of dicts with keys:
      - instruction: text prompt.
      - images: list of image references (paths, URLs, or base64 data URLs).
      - Any additional keys are preserved in the output metadata.

    Output dict:
      - instruction: the original text prompt.
      - images: the original image references.
      - response: the generated teacher response.
      - Any additional input metadata.
    """

    name = "mm_instruct_distill"
    default_max_tokens = 2048

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(self.config)
        self.system_prompt = resolve_prompt(
            self.config,
            template_key="system_prompt",
            file_key="system_prompt_file",
            default_file="configs/prompts/mm_generation_prompt.txt",
        )

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            instruction = row.get("instruction", "")
            images = row.get("images") or []
            user_content: Union[str, List[Dict[str, Any]]]
            if isinstance(instruction, list):
                # Preserve pre-built multi-modal content lists as the user message.
                if self.prompt_template:
                    logger.warning(
                        "prompt_template ignored for row %d because instruction is a "
                        "pre-built multi-modal content list.",
                        idx,
                    )
                if images and content_has_images(instruction):
                    logger.warning(
                        "Ignoring separate images for row %d because the instruction "
                        "already contains image items.",
                        idx,
                    )
                    images = []
                user_content = instruction
            else:
                user_content = (
                    format_prompt_safely(self.prompt_template, instruction=instruction)
                    if self.prompt_template
                    else instruction
                )
            content = build_multimodal_user_content(user_content, images)
            metadata = {k: v for k, v in row.items() if k not in {"instruction", "images"}}
            metadata["instruction"] = instruction
            metadata["images"] = images
            requests.append(
                GenerationRequest(
                    id=f"mm_gen_{idx}",
                    instruction=content,
                    system_prompt=self.system_prompt,
                    metadata=metadata,
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        response = (result.response or "").strip()
        if not response:
            return None
        output = dict(result.request.metadata)
        output["response"] = response
        return output
