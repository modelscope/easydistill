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

"""Abstract T2I backend for text-to-image generation."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from easydistill.data.models import ImageGenerationResult


class T2IBackend(ABC):
    """Base class for all T2I (text-to-image / text-to-video) backends.

    T2I/T2V model APIs use a different protocol from chat completions, so they
    live in a separate abstraction parallel to :class:`ModelBackend`.  This keeps
    the operator code backend-agnostic: the same T2I operator works with Wanx,
    PAI-Diffusion, or any future image-generation endpoint.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate one or more images from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            model_id: Model identifier. If None, backend picks a default.
            size: Image size string (format depends on the backend).
            n: Number of images to generate.
            **kwargs: Backend-specific generation arguments.

        Returns:
            An ImageGenerationResult containing the generated image URLs and
            metadata.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        return True

    def close(self) -> None:
        """Release any resources held by the backend."""
        return None

    def __enter__(self) -> "T2IBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
