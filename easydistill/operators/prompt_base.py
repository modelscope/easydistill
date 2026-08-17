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

"""Base class for prompt-based generation operators."""

import logging
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult, SFTSample
from easydistill.utils import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
    DEFAULT_TEMPERATURE,
)

from .generation import TextGenerationOperator
from .sft_builder import SFTDatasetBuilder

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class PromptGenerationOperator(Generic[T, U]):
    """Base operator that generates or rewrites data via prompt + LLM.

    Subclasses implement:
      - _build_requests(inputs) -> List[GenerationRequest]
      - _parse_result(result) -> Optional[U]

    Configurable fields:
      - system_prompt: optional system message.
      - model_id: model identifier passed to the backend.
      - temperature: sampling temperature.
      - max_tokens: max tokens per response.
      - show_progress: whether to show tqdm progress bar.
      - max_workers: number of concurrent workers.
      - retry_attempts: number of retries per request on transient failures.
      - retry_backoff_base: base delay (seconds) for exponential backoff.
      - retry_max_wait: max wait (seconds) between retries.
      - raise_on_error: if True, raise on final generation failure.
    """

    name = "prompt_generation"
    default_max_tokens: int = 2048

    def __init__(
        self,
        backend: ModelBackend,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        max_tokens = self.config.get("max_tokens") or self.default_max_tokens
        show_progress = self.config.get("show_progress")
        self.generator = TextGenerationOperator(
            backend=backend,
            config={
                "system_prompt": self.config.get("system_prompt"),
                "model_id": self.config.get("model_id"),
                "temperature": (
                    float(self.config["temperature"])
                    if self.config.get("temperature") is not None
                    else DEFAULT_TEMPERATURE
                ),
                "max_tokens": int(max_tokens),
                "show_progress": bool(show_progress) if show_progress is not None else True,
                "max_workers": int(self.config.get("max_workers") or DEFAULT_MAX_WORKERS),
                "raise_on_error": bool(self.config.get("raise_on_error") or False),
                "retry_attempts": int(
                    self.config.get("retry_attempts") or DEFAULT_RETRY_ATTEMPTS
                ),
                "retry_backoff_base": float(
                    self.config.get("retry_backoff_base") or DEFAULT_RETRY_BACKOFF_BASE
                ),
                "retry_max_wait": float(
                    self.config.get("retry_max_wait") or DEFAULT_RETRY_MAX_WAIT
                ),
            },
        )

    @abstractmethod
    def _build_requests(self, inputs: List[T]) -> List[GenerationRequest]:
        """Build generation requests from raw inputs."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, result: GenerationResult) -> Optional[U]:
        """Parse a generation result into an output object."""
        raise NotImplementedError

    def run(self, inputs: List[T]) -> List[U]:
        """Run prompt-based generation over a list of inputs."""
        if not inputs:
            return []
        requests = self._build_requests(inputs)
        results = self.generator.run(requests)
        outputs = []
        for result in results:
            parsed = self._parse_result(result)
            if parsed is not None:
                outputs.append(parsed)
            else:
                logger.warning(
                    "Failed to parse %s result for request %s",
                    self.name,
                    result.request.id,
                )
        logger.info(
            "%s produced %d valid outputs from %d inputs.",
            self.name,
            len(outputs),
            len(inputs),
        )
        return outputs

    def run_to_sft(
        self,
        inputs: List[T],
        sft_config: Optional[Dict[str, Any]] = None,
    ) -> List[SFTSample]:
        """Run generation and return SFT samples instead of parsed outputs.

        This is the standard output path for basic distillation features that
        produce training data for LLaMA-Factory / ms-swift.
        """
        if not inputs:
            return []
        requests = self._build_requests(inputs)
        results = self.generator.run(requests)
        builder = SFTDatasetBuilder(config=sft_config)
        return builder.run(results)
