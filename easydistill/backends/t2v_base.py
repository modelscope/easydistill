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

"""Abstract T2V backend for text-to-video and image-to-video generation."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from easydistill.data.models import VideoGenerationResult


class T2VBackend(ABC):
    """Base class for all T2V/I2V (text-to-video / image-to-video) backends.

    Video-generation APIs use a submit-poll protocol distinct from both chat
    completions and image generation, so they live in a separate abstraction
    parallel to :class:`ModelBackend` and :class:`T2IBackend`.  This keeps the
    operator code backend-agnostic: the same T2V operator works with Wanx
    video, PAI-deployed video models, or any future video-generation endpoint.

    A single backend serves both modes: passing ``first_frame_image`` switches
    the call to I2V (image-to-video), while omitting it runs plain T2V.  This
    mirrors the upstream provider APIs, where T2V and I2V share one protocol
    and differ only in the conditioning image and model identifier.
    """

    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: Optional[str] = None,
        duration: Optional[float] = None,
        first_frame_image: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """Generate a video from a text prompt (and optional first frame).

        Args:
            prompt: Text prompt describing the desired video.
            model_id: Model identifier. If None, the backend picks a default
                appropriate for the mode (T2V vs I2V).
            size: Legacy video resolution string (e.g. ``"1280*720"``);
                newer APIs use ``resolution`` / ``ratio`` passed via kwargs.
            duration: Video duration in seconds (backend-specific default).
            first_frame_image: Conditioning first-frame image URL or local
                path.  When provided, the backend runs in I2V mode.
            **kwargs: Backend-specific generation arguments (e.g. fps,
                negative_prompt, seed).

        Returns:
            A VideoGenerationResult containing the generated video URLs and
            metadata.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        return True

    def close(self) -> None:
        """Release any resources held by the backend."""
        return None

    def __enter__(self) -> "T2VBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
