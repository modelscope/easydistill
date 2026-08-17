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

"""Wanx (Tongyi Wanxiang) text-to-image backend via the dashscope SDK."""

import logging
import time
from typing import Any, Dict, List, Optional

from easydistill.data.models import ImageGenerationResult

from .t2i_base import T2IBackend
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)

try:
    import dashscope

    _HAS_DASHSCOPE = True
except ImportError:
    _HAS_DASHSCOPE = False
    dashscope = None  # type: ignore

_DEFAULT_WANX_MODEL = "wanx2.1-t2i-turbo"
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_MAX_POLL_WAIT = 300.0
_DEFAULT_PROGRESS_LOG_INTERVAL = 30.0


class WanxBackend(T2IBackend):
    """Text-to-image backend using Alibaba Tongyi Wanxiang (Wanx) via dashscope.

    Wanx is an asynchronous image-synthesis service: ``call`` submits a task
    and returns a task id; ``fetch`` polls until the task is SUCCEEDED.  This
    backend encapsulates the submit-poll loop so callers see a synchronous
    ``generate_image`` interface.
    """

    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        max_poll_wait: float = _DEFAULT_MAX_POLL_WAIT,
        timeout: float = 120.0,
        retry_attempts: int = 3,
        retry_backoff_base: float = 1.0,
        retry_max_wait: float = 30.0,
    ):
        if not _HAS_DASHSCOPE:
            raise ImportError(
                "The 'dashscope' package is required for WanxBackend. "
                "Install it with: pip install dashscope"
            )
        self.api_key = api_key
        self.model_id = model_id or _DEFAULT_WANX_MODEL
        self.poll_interval = poll_interval
        self.max_poll_wait = max_poll_wait
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff_base = retry_backoff_base
        self.retry_max_wait = retry_max_wait

    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        model = model_id or self.model_id
        logger.info("Wanx: submitting T2I task (model=%s, size=%s, n=%d).", model, size, n)

        # Submit the asynchronous task.  ``request_timeout`` is the dashscope
        # SDK keyword for the per-request HTTP timeout; the overall polling
        # duration is bounded separately by ``max_poll_wait``.
        submit_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "request_timeout": self.timeout,
            "api_key": self.api_key,
        }
        submit_kwargs.update(kwargs)

        def _submit() -> Any:
            return dashscope.ImageSynthesis.call(**submit_kwargs)

        rsp = retry_with_backoff(
            _submit,
            max_attempts=self.retry_attempts + 1,
            backoff_base=self.retry_backoff_base,
            max_wait=self.retry_max_wait,
            description="Wanx submit",
        )

        if rsp.status_code != 200:
            raise RuntimeError(
                f"Wanx submit failed (status={rsp.status_code}): "
                f"{getattr(rsp, 'message', '')}"
            )

        # If the call is synchronous (already succeeded), return immediately.
        if rsp.output and rsp.output.get("results"):
            return self._build_result(prompt, model, rsp)

        task_id = rsp.output.get("task_id") if rsp.output else None
        if not task_id:
            raise RuntimeError(f"Wanx submit returned no task_id: {rsp}")

        # Poll until completion.
        return self._poll_task(task_id, prompt, model)

    def _poll_task(self, task_id: str, prompt: str, model: str) -> ImageGenerationResult:
        deadline = time.monotonic() + self.max_poll_wait
        start_time = time.monotonic()
        last_log_time = start_time
        poll_count = 0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            poll_count += 1

            def _fetch() -> Any:
                return dashscope.ImageSynthesis.fetch(task=task_id, api_key=self.api_key)

            try:
                rsp = retry_with_backoff(
                    _fetch,
                    max_attempts=self.retry_attempts + 1,
                    backoff_base=self.retry_backoff_base,
                    max_wait=self.retry_max_wait,
                    description=f"Wanx fetch task {task_id}",
                )
            except KeyboardInterrupt:
                logger.warning(
                    "Wanx polling interrupted for task %s. The remote task may still be running.",
                    task_id,
                )
                raise

            task_status = rsp.output.get("task_status") if rsp.output else None
            logger.debug("Wanx task %s status: %s", task_id, task_status)

            now = time.monotonic()
            if now - last_log_time >= _DEFAULT_PROGRESS_LOG_INTERVAL:
                logger.info(
                    "Wanx task %s still %s after %.0fs (%d polls).",
                    task_id,
                    task_status,
                    now - start_time,
                    poll_count,
                )
                last_log_time = now

            if task_status == "SUCCEEDED":
                logger.info(
                    "Wanx task %s completed after %.0fs (%d polls).",
                    task_id,
                    now - start_time,
                    poll_count,
                )
                return self._build_result(prompt, model, rsp)
            if task_status == "FAILED":
                raise RuntimeError(
                    f"Wanx task {task_id} failed: {getattr(rsp, 'message', '')}"
                )
            # PENDING / RUNNING -> keep polling.
        raise TimeoutError(
            f"Wanx task {task_id} did not complete within {self.max_poll_wait}s."
        )

    @staticmethod
    def _build_result(prompt: str, model: str, rsp: Any) -> ImageGenerationResult:
        output = rsp.output or {}
        results: List[Any] = output.get("results") or []
        image_urls = [item.get("url") for item in results if item.get("url")]
        usage = rsp.usage if hasattr(rsp, "usage") and rsp.usage else None
        return ImageGenerationResult(
            prompt=prompt,
            image_urls=image_urls,
            model=model,
            usage=usage,
            metadata={"task_id": output.get("task_id")},
        )

    def health_check(self) -> bool:
        try:
            # A lightweight check: verify the API key is set.
            return bool(self.api_key)
        except Exception as exc:
            logger.warning("Wanx health check failed: %s", exc)
            return False
