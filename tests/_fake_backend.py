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

"""Minimal fake backends for unit tests. Not part of the public package."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult


class FakeBackend(ModelBackend):
    """Returns deterministic templated responses for tests."""

    def __init__(self, response_template: str = "{}"):
        self.response_template = response_template

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        request = build_generation_request(messages)
        # For multi-modal requests the instruction may be a content list.
        instruction_text = request.instruction
        if isinstance(instruction_text, list):
            text_parts = [
                item.get("text", "")
                for item in instruction_text
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            instruction_text = " ".join(text_parts) or "[multimodal]"
        return GenerationResult(
            request=request,
            response=self.response_template.format(instruction_text),
            model=model_id or "fake",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    def health_check(self) -> bool:
        return True


class FakeVideoJudgeBackend(ModelBackend):
    """Answers VLM video-judge prompts with scripted dimension judgments.

    ``scores`` maps dimension name -> (score, confidence).  Requested
    dimensions are parsed from the prompt's dimension block (lines starting
    with ``- <name>``); dimensions marked "(NOT applicable" in the prompt
    are answered with ``applicable: false``.
    """

    def __init__(
        self,
        scores: Optional[Dict[str, Tuple[int, float]]] = None,
        default_score: int = 3,
        default_confidence: float = 0.9,
    ):
        self.scores = scores or {}
        self.default_score = default_score
        self.default_confidence = default_confidence
        self.call_count = 0
        self.last_prompt: str = ""
        self.last_image_count: int = 0
        self.last_video_count: int = 0
        self.last_video_url: str = ""

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        self.call_count += 1
        request = build_generation_request(messages)
        content = messages[-1]["content"]
        if isinstance(content, list):
            prompt = " \n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            self.last_image_count = sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "image_url"
            )
            video_items = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") == "video_url"
            ]
            self.last_video_count = len(video_items)
            self.last_video_url = (
                video_items[0].get("video_url", {}).get("url", "") if video_items else ""
            )
        else:
            prompt = str(content)
            self.last_image_count = 0
            self.last_video_count = 0
            self.last_video_url = ""
        self.last_prompt = prompt

        judgments = []
        for match in re.finditer(r"^- ([a-z_]+)(\(NOT applicable[^)]*\))?", prompt, re.M):
            name = match.group(1)
            not_applicable = "(NOT applicable" in (
                prompt.split(f"- {name}", 1)[1].split("\n", 1)[0]
            )
            if not_applicable:
                judgments.append(
                    {
                        "dimension": name,
                        "applicable": False,
                        "score": None,
                        "confidence": None,
                        "evidence_frames": [],
                        "reason": "not applicable",
                    }
                )
                continue
            score, confidence = self.scores.get(
                name, (self.default_score, self.default_confidence)
            )
            judgments.append(
                {
                    "dimension": name,
                    "applicable": True,
                    "score": score,
                    "confidence": confidence,
                    "evidence_frames": [1, 2],
                    "reason": f"reason for {name}",
                }
            )
        return GenerationResult(
            request=request,
            response=json.dumps({"dimension_judgments": judgments}),
            model=model_id or "fake-judge",
        )

    def health_check(self) -> bool:
        return True
