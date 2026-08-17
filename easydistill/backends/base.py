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

"""Abstract model backend."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationResult


class ModelBackend(ABC):
    """Base class for all model backends.

    All teacher/student model calls go through this abstraction so that the same
    operator code works with PAI-Token, PAI-EAS, OpenAI-compatible endpoints, or
    local inference engines.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a chat completion from the model.

        Args:
            messages: OpenAI-compatible message list. Content may be a string or
                a multi-modal content list for vision-language models.
            model_id: Model identifier. If None, backend picks a default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Backend-specific generation arguments.

        Returns:
            A GenerationResult containing the generated response and metadata.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        return True

    def close(self) -> None:
        """Release any resources held by the backend."""
        return None

    def __enter__(self) -> "ModelBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
