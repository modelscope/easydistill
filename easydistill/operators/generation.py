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

"""Text generation operator: seed instruction -> teacher response."""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
    DEFAULT_TEMPERATURE,
    format_prompt_safely,
    progress,
)

from .base import Operator

logger = logging.getLogger(__name__)


class TextGenerationOperator(Operator[List[GenerationRequest], List[GenerationResult]]):
    """Generate teacher responses for a list of seed instructions.

    Supports both sequential and concurrent API calling. Use `max_workers` to
    control parallelism for real API backends.

    Configurable fields:
      - system_prompt: optional system message.
      - model_id: model identifier passed to the backend.
      - temperature: sampling temperature.
      - max_tokens: max tokens per response.
      - prompt_template: optional template with {instruction} placeholder applied
        to each request's instruction before sending to the backend.
      - prompt_template_file: path to a text file containing the prompt template.
      - show_progress: whether to show tqdm progress bar.
      - max_workers: number of concurrent workers (default 1, sequential).
      - retry_attempts: number of retries per request on transient failures (default 3).
      - retry_backoff_base: base delay (seconds) for exponential backoff (default 1.0).
      - retry_max_wait: max wait (seconds) between retries (default 30.0).
      - raise_on_error: if True, raise on final generation failure.
    """

    name = "text_generation"

    def __init__(
        self,
        backend: ModelBackend,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.backend = backend
        self.system_prompt = self.config.get("system_prompt")
        self.model_id = self.config.get("model_id")
        self.prompt_template = resolve_prompt(self.config)
        temperature = self.config.get("temperature")
        self.temperature = (
            float(temperature) if temperature is not None else DEFAULT_TEMPERATURE
        )
        max_tokens = int(self.config.get("max_tokens") or DEFAULT_MAX_TOKENS)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        self.max_tokens = max_tokens
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        max_workers = int(self.config.get("max_workers") or DEFAULT_MAX_WORKERS)
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        self.max_workers = max_workers
        self.raise_on_error = bool(self.config.get("raise_on_error") or False)
        retry_attempts = self.config.get("retry_attempts")
        self.retry_attempts = (
            DEFAULT_RETRY_ATTEMPTS if retry_attempts is None else int(retry_attempts)
        )
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be a non-negative integer.")
        self.retry_backoff_base = float(
            self.config.get("retry_backoff_base") or DEFAULT_RETRY_BACKOFF_BASE
        )
        self.retry_max_wait = float(
            self.config.get("retry_max_wait") or DEFAULT_RETRY_MAX_WAIT
        )

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        messages = []
        system = request.system_prompt or self.system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        instruction = request.instruction
        if self.prompt_template:
            instruction = format_prompt_safely(self.prompt_template, instruction=instruction)
        messages.append({"role": "user", "content": instruction})
        return messages

    def _prepare_request(self, request: GenerationRequest) -> None:
        """Record the effective system prompt on the request for observability."""
        if request.system_prompt is None and self.system_prompt is not None:
            request.system_prompt = self.system_prompt

    def _is_retryable(self, exc: Exception) -> bool:
        """Return True if the exception is likely transient and worth retrying."""
        # Always retry obvious network / timeout errors.
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        # If httpx is available, retry its network and timeout errors.
        exc_module = type(exc).__module__
        exc_name = type(exc).__name__
        if exc_module == "httpx" and exc_name in (
            "ConnectError",
            "ReadError",
            "WriteError",
            "TimeoutException",
            "NetworkError",
        ):
            return True
        # If the backend is the OpenAI client, retry rate-limit and server errors.
        return exc_module == "openai" and exc_name in (
            "RateLimitError",
            "InternalServerError",
            "APITimeoutError",
            "APIConnectionError",
        )

    def _generate_one(self, request: GenerationRequest) -> Optional[GenerationResult]:
        """Generate a single response, retrying transient failures.

        Makes one initial attempt plus up to ``retry_attempts`` retries with
        jittered exponential backoff. Only retryable exceptions trigger a retry;
        client-side errors fail fast. The final failure is caught unless
        ``raise_on_error`` is enabled.
        """
        last_exc: Optional[Exception] = None
        total_attempts = self.retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            try:
                messages = self._build_messages(request)
                self._prepare_request(request)
                result = self.backend.generate(
                    messages=messages,
                    model_id=self.model_id or request.metadata.get("model_id"),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result.request = request
                result.metadata.update(request.metadata)
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == total_attempts or not self._is_retryable(exc):
                    break
                wait = min(
                    self.retry_backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random()),
                    self.retry_max_wait,
                )
                logger.warning(
                    "Generation failed for instruction %s (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    request.id,
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "Failed to generate for instruction %s after %d attempts: %s",
            request.id,
            total_attempts,
            last_exc,
        )
        if self.raise_on_error and last_exc is not None:
            raise last_exc
        return None

    def run(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if self.max_workers <= 1:
            return self._run_sequential(requests)
        return self._run_concurrent(requests)

    def _run_sequential(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        results = []
        for request in progress(
            requests,
            enabled=self.show_progress,
            total=len(requests),
            desc="Generating responses",
        ):
            result = self._generate_one(request)
            if result is not None:
                results.append(result)
        return results

    def _run_concurrent(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        results: List[Optional[GenerationResult]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_one, request): idx
                for idx, request in enumerate(requests)
            }
            futures_iter = progress(
                futures.items(),
                enabled=self.show_progress,
                total=len(futures),
                desc="Generating responses",
            )

            for future, idx in futures_iter:
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Generation task %s raised: %s",
                        requests[idx].id,
                        exc,
                    )
                    result = None
                if result is not None:
                    results[idx] = result

        return [r for r in results if r is not None]
