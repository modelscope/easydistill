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

"""T2I generation operator: optimized prompt -> teacher images."""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from easydistill.backends.t2i_base import T2IBackend
from easydistill.utils import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
    progress,
)

logger = logging.getLogger(__name__)


class T2IGenerationOperator:
    """Generate images from prompts using a T2I backend.

    Unlike text-generation operators, this does not inherit from
    :class:`PromptGenerationOperator` because the T2I backend uses a different
    API protocol (image generation, not chat completions).  Concurrency and
    retry logic are self-contained, mirroring :class:`TextGenerationOperator`.

    Configurable fields:
      - model_id: model identifier passed to the T2I backend.
      - size: image size string (e.g. "1024*1024").
      - n: number of images to generate per prompt.
      - max_workers: number of concurrent workers (default 1).
      - retry_attempts: retries on transient failures (default 3).
      - retry_backoff_base: base delay for exponential backoff (default 1.0).
      - retry_max_wait: max wait between retries (default 30.0).
      - show_progress: whether to show tqdm progress bar.
      - raise_on_error: if True, raise on final generation failure.
      - prompt_key: row key to read the prompt from (default "optimized_prompt").
    """

    name = "t2i_generate"

    def __init__(self, backend: T2IBackend, config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.config = config or {}
        self.model_id = self.config.get("model_id")
        self.size = self.config.get("size", "1024*1024")
        self.n = int(self.config.get("n", 1))
        self.max_workers = int(self.config.get("max_workers", DEFAULT_MAX_WORKERS))
        self.retry_attempts = int(self.config.get("retry_attempts", DEFAULT_RETRY_ATTEMPTS))
        self.retry_backoff_base = float(
            self.config.get("retry_backoff_base", DEFAULT_RETRY_BACKOFF_BASE)
        )
        self.retry_max_wait = float(self.config.get("retry_max_wait", DEFAULT_RETRY_MAX_WAIT))
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self.raise_on_error = bool(self.config.get("raise_on_error") or False)
        self.prompt_key = self.config.get("prompt_key", "optimized_prompt")
        # Pass through any extra kwargs to the backend.
        self._extra_kwargs: Dict[str, Any] = {
            k: v
            for k, v in self.config.items()
            if k
            not in {
                "model_id",
                "size",
                "n",
                "max_workers",
                "retry_attempts",
                "retry_backoff_base",
                "retry_max_wait",
                "show_progress",
                "raise_on_error",
                "prompt_key",
            }
        }

    def _generate_one(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate images for a single row, retrying transient failures."""
        prompt = row.get(self.prompt_key) or row.get("prompt") or ""
        if not prompt:
            logger.warning("Row %s has empty prompt, skipping.", row.get("id"))
            return None

        last_exc: Optional[Exception] = None
        total_attempts = self.retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            try:
                result = self.backend.generate_image(
                    prompt=prompt,
                    model_id=self.model_id,
                    size=self.size,
                    n=self.n,
                    **self._extra_kwargs,
                )
                if not result.image_urls:
                    logger.warning(
                        "T2I generation for row %s returned no images.", row.get("id")
                    )
                    return None
                new_row = dict(row)
                new_row["image_urls"] = result.image_urls
                new_row["t2i_model"] = result.model
                if result.usage:
                    new_row["t2i_usage"] = result.usage
                return new_row
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == total_attempts or not self._is_retryable(exc):
                    break
                wait = min(
                    self.retry_backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random()),
                    self.retry_max_wait,
                )
                logger.warning(
                    "T2I generation failed for row %s (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    row.get("id"),
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "Failed to generate images for row %s after %d attempts: %s",
            row.get("id"),
            total_attempts,
            last_exc,
        )
        if self.raise_on_error and last_exc is not None:
            raise last_exc
        return None

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the exception is likely transient."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        exc_name = type(exc).__name__
        # Retry on common transient HTTP / API errors.
        return exc_name in (
            "ConnectError",
            "ReadError",
            "WriteError",
            "TimeoutException",
            "NetworkError",
            "RateLimitError",
            "InternalServerError",
            "APITimeoutError",
            "APIConnectionError",
        )

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            return []
        if self.max_workers <= 1:
            return self._run_sequential(data)
        return self._run_concurrent(data)

    def _run_sequential(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for row in progress(
            data,
            enabled=self.show_progress,
            total=len(data),
            desc="Generating images",
        ):
            result = self._generate_one(row)
            if result is not None:
                results.append(result)
        return results

    def _run_concurrent(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(data)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_one, row): idx
                for idx, row in enumerate(data)
            }
            futures_iter = progress(
                futures.items(),
                enabled=self.show_progress,
                total=len(futures),
                desc="Generating images",
            )
            for future, idx in futures_iter:
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "T2I generation task at index %d raised: %s",
                        idx,
                        exc,
                    )
                    result = None
                if result is not None:
                    results[idx] = result
        return [r for r in results if r is not None]
