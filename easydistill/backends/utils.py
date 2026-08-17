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

"""Shared helpers for model backends."""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from easydistill.data.models import GenerationRequest
from easydistill.utils.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def build_generation_request(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    **extra: Any,
) -> GenerationRequest:
    """Build a GenerationRequest from an OpenAI-style message list.

    The instruction is taken from the last message. The system prompt is taken
    from the first message only when its role is ``system``. Any extra keyword
    arguments are stored in the request metadata.
    """
    instruction = messages[-1].get("content", "") if messages else ""
    system_prompt = None
    if messages and messages[0].get("role") == "system":
        system_content = messages[0].get("content")
        # System prompts must be plain text; ignore multi-modal system content.
        system_prompt = system_content if isinstance(system_content, str) else None
    metadata: Dict[str, Any] = dict(extra)
    if model is not None:
        metadata["model"] = model
    return GenerationRequest(
        instruction=instruction,
        system_prompt=system_prompt,
        metadata=metadata,
    )


def retry_with_backoff(
    func: Callable[[], _T],
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS + 1,
    backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
    max_wait: float = DEFAULT_RETRY_MAX_WAIT,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
    ),
    description: str = "operation",
) -> _T:
    """Call ``func`` repeatedly with exponential backoff on transient failures.

    Args:
        func: Zero-argument callable to retry.
        max_attempts: Maximum total attempts (attempts = retries + 1).
        backoff_base: Base delay for exponential backoff.
        max_wait: Cap on the wait time between attempts.
        retryable_exceptions: Exception classes that should trigger a retry.
        description: Human-readable name for log messages.

    Returns:
        The value returned by ``func``.

    Raises:
        The last exception raised by ``func`` if all attempts are exhausted, or
        any non-retryable exception on the first occurrence.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except KeyboardInterrupt:
            logger.warning("%s interrupted by user on attempt %d.", description, attempt)
            raise
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            wait = min(
                backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random()),
                max_wait,
            )
            logger.warning(
                "%s failed on attempt %d/%d: %s. Retrying in %.1fs.",
                description,
                attempt,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{description} failed after {max_attempts} attempts")
