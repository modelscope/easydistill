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

"""LLM-as-judge evaluation for T2I (text-to-image) datasets."""

from typing import Any, Dict, List, Tuple

from .base import LLMJudgeEvaluator


class T2IImageEvaluator(LLMJudgeEvaluator):
    """Evaluate T2I-generated images with a VLM (vision-language model) judge.

    The evaluator receives rows containing a text prompt and one or more
    generated image URLs.  For each row, it sends the prompt text and the
    image to the VLM judge, which returns a score per configured metric.

    Supported metrics (all 0-9 scale):
      - ``prompt_consistency``: how faithfully the image depicts the prompt.
      - ``aesthetic_quality``: visual appeal, composition, color, lighting.
      - ``detail_richness``: level of fine detail and texture.
      - ``artifact_absence``: freedom from generation artifacts.

    Row fields used:
      - ``optimized_prompt`` (or ``prompt``): the text prompt.
      - ``image_urls``: list of image URLs; the first URL is evaluated.
    """

    name = "t2i_image_evaluator"
    DEFAULT_PROMPTS_FILE = "configs/prompts/t2i_eval_prompts.yaml"
    BOOL_METRICS: set = set()  # All T2I metrics are 0-9 integer scores.

    def _extract_sample(self, sample: Dict[str, Any]) -> Tuple[str, str, str]:
        sample_id = str(sample.get("id", sample.get("index", 0)))
        instruction = (
            sample.get("optimized_prompt")
            or sample.get("prompt")
            or sample.get("instruction")
            or ""
        )
        # ``output`` is used as a placeholder in the eval prompt template.
        # The actual image is attached via _extract_images and sent as
        # multi-modal content to the VLM judge.
        image_urls = sample.get("image_urls") or []
        output = image_urls[0] if image_urls else ""
        return sample_id, instruction, output

    def _extract_images(self, sample: Dict[str, Any]) -> List[str]:
        image_urls = sample.get("image_urls") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        # Evaluate only the first image per row.
        return image_urls[:1] if image_urls else []
