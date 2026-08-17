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

"""T2V SFT builder: prompt (+first frame) + video rows -> SFT training samples."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import SFTSample
from easydistill.operators.base import Operator

logger = logging.getLogger(__name__)

# Evaluation-score keys preserved into SFT sample metadata; must match the
# metric columns produced by the t2v_eval checkers (vlm dimensions plus the
# vbench/omni columns configured by default).
_EVAL_SCORE_KEYS = (
    "prompt_consistency",
    "visual_quality",
    "subject_consistency",
    "first_frame_consistency",
    "motion_quality",
    "temporal_execution",
    "camera_accuracy",
)


class T2VSFTBuilder(Operator[List[Dict[str, Any]], List[SFTSample]]):
    """Convert T2V/I2V data rows into multi-modal SFT samples.

    Each row should contain at least ``optimized_prompt`` (or ``prompt``) and
    ``video_urls`` (a list of video URLs).  The builder creates an SFT sample
    where the user message is the prompt (plus the conditioning first-frame
    image for I2V rows) and the assistant message is a multi-modal content
    list containing the video.

    Configurable fields:
      - skip_empty: skip rows with no videos or empty prompts (default True).
      - min_prompt_length: minimum prompt length in characters (default 0).
      - max_videos_per_prompt: max videos to include per sample (default 1;
        only the first video is used for SFT).
      - system_prompt: optional system prompt for all samples.
    """

    name = "t2v_sft_builder"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        skip_empty = self.config.get("skip_empty")
        self.skip_empty = bool(skip_empty) if skip_empty is not None else True
        self.min_prompt_length = int(self.config.get("min_prompt_length", 0))
        self.max_videos_per_prompt = int(self.config.get("max_videos_per_prompt", 1))
        self.system_prompt = self.config.get("system_prompt")

    def _is_valid(self, row: Dict[str, Any]) -> bool:
        prompt = row.get("optimized_prompt") or row.get("prompt") or ""
        video_urls = row.get("video_urls") or []
        if self.skip_empty and (not prompt or not video_urls):
            return False
        return len(prompt) >= self.min_prompt_length

    def run(self, data: List[Dict[str, Any]]) -> List[SFTSample]:
        samples = []
        for idx, row in enumerate(data):
            if not self._is_valid(row):
                logger.info("Skipping invalid T2V row for id: %s", row.get("id", idx))
                continue
            prompt = row.get("optimized_prompt") or row.get("prompt", "")
            video_urls = row.get("video_urls") or []
            # Use only the first video for SFT (or up to max_videos_per_prompt).
            selected = video_urls[: self.max_videos_per_prompt]
            if not selected:
                continue

            metadata = {
                "source": "t2v_distillation",
                "t2v_model": row.get("t2v_model"),
                "t2v_mode": row.get("t2v_mode", "t2v"),
                "raw_prompt": row.get("raw_prompt", ""),
                "request_id": str(row.get("id", idx)),
            }
            # Preserve evaluation scores if present.
            for key in _EVAL_SCORE_KEYS:
                if key in row:
                    metadata[key] = row[key]

            sample = SFTSample.from_prompt_video(
                prompt=prompt,
                video_url=selected[0],
                first_frame_image=row.get("first_frame_image"),
                system=self.system_prompt,
                metadata=metadata,
            )
            samples.append(sample)
        logger.info("Built %d T2V SFT samples from %d rows.", len(samples), len(data))
        return samples
