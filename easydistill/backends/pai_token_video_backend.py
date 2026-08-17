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

"""PAI-Token video backend for T2V/I2V models behind the PAI-Token gateway.

The gateway proxies DashScope-style video synthesis: an async submit-poll
HTTP protocol (no SDK dependency):

1. **Submit**: ``POST {base_url}{submit_path}`` with header
   ``X-DashScope-Async: enable`` and body
   ``{"model", "input": {"prompt", "media"?}, "parameters": {...}}``
   returns ``{"output": {"task_id": ...}}`` (``media`` carries the I2V
   first frame; legacy gateways may use ``input.img_url`` instead).
2. **Poll**: ``GET {base_url}/tasks/{task_id}`` until
   ``output.task_status`` is ``SUCCEEDED``; the video URL is read from
   ``output.video_url`` (or ``output.results[].url``).

NOTE(slot-5): the default ``submit_path`` / task path follow the DashScope
convention and are configurable pending confirmation of the gateway routes.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from easydistill.data.models import VideoGenerationResult
from easydistill.utils import normalize_image_reference, safe_filename_stem

from .t2v_base import T2VBackend

logger = logging.getLogger(__name__)

_DEFAULT_T2V_MODEL = "happyhorse-1.1-t2v"
_DEFAULT_I2V_MODEL = "happyhorse-1.1-i2v"
_DEFAULT_SUBMIT_PATH = "/services/aigc/video-generation/video-synthesis"
_DEFAULT_PROGRESS_LOG_INTERVAL = 30.0


class PaiTokenVideoBackend(T2VBackend):
    """Text/image-to-video backend via the PAI-Token gateway.

    One backend serves both modes: when ``first_frame_image`` is provided
    the call runs I2V (the I2V default model is used) and otherwise plain
    T2V.  Local first-frame paths are converted to base64 data URLs
    automatically.

    Args:
        api_key: PAI-Token API key (Bearer auth).
        base_url: Gateway base URL (e.g.
            ``https://dashscope.aliyuncs.com/api/v1``).
        model_id: Default T2V model identifier.
        i2v_model_id: Default I2V model identifier.
        submit_path: Video synthesis submit route under ``base_url``.
        i2v_image_field: How the conditioning frame is carried in the submit
            payload.  ``"media"`` (default) uses the current DashScope
            protocol ``input.media = [{"type": "first_frame", "url": ...}]``
            (happyhorse / wan2.7 series); ``"img_url"`` uses the legacy
            ``input.img_url`` field of older wan models.
        timeout: Per-request HTTP timeout in seconds.
        poll_interval: Seconds between task status polls.
        max_poll_wait: Maximum seconds to wait for a task to complete.
        output_dir: Directory to download finished videos into.  When set,
            result ``video_urls`` carry local file paths (the remote URLs
            are kept in ``metadata.remote_urls``), so downstream consumers
            (frame sampling, VBench) always see local files.  When ``None``,
            the remote URLs are returned as-is.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: Optional[str] = None,
        i2v_model_id: Optional[str] = None,
        submit_path: str = _DEFAULT_SUBMIT_PATH,
        i2v_image_field: str = "media",
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        max_poll_wait: float = 1800.0,
        output_dir: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id or _DEFAULT_T2V_MODEL
        self.i2v_model_id = i2v_model_id or _DEFAULT_I2V_MODEL
        self.submit_path = submit_path
        if i2v_image_field not in ("media", "img_url"):
            raise ValueError(
                f"i2v_image_field must be 'media' or 'img_url', "
                f"got {i2v_image_field!r}."
            )
        self.i2v_image_field = i2v_image_field
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll_wait = max_poll_wait
        self.output_dir = output_dir

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: Optional[str] = None,
        duration: Optional[float] = None,
        first_frame_image: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        is_i2v = bool(first_frame_image)
        model = model_id or (self.i2v_model_id if is_i2v else self.model_id)

        input_payload: Dict[str, Any] = {"prompt": prompt}
        if first_frame_image:
            # Local paths / file:// refs become base64 data URLs; http(s)
            # URLs pass through unchanged.
            image_ref = normalize_image_reference(first_frame_image)
            if self.i2v_image_field == "media":
                # Current DashScope protocol (happyhorse / wan2.7 series).
                input_payload["media"] = [
                    {"type": "first_frame", "url": image_ref}
                ]
            else:
                input_payload["img_url"] = image_ref

        parameters: Dict[str, Any] = {}
        # Legacy `size` only applies when the new-style resolution/ratio
        # knobs are not configured (wan2.7 / happyhorse use those instead).
        if size and "resolution" not in kwargs and "ratio" not in kwargs:
            parameters["size"] = size.replace("x", "*")
        if duration is not None:
            # DashScope video models require an integer number of seconds.
            duration_value = float(duration)
            parameters["duration"] = (
                int(duration_value)
                if duration_value.is_integer()
                else duration_value
            )
        parameters.update(kwargs)

        payload = {"model": model, "input": input_payload, "parameters": parameters}
        submit_url = f"{self.base_url}{self.submit_path}"
        logger.info(
            "PAI-Token-Video: submitting %s task (model=%s, size=%s, duration=%s).",
            "I2V" if is_i2v else "T2V",
            model,
            size,
            duration,
        )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                submit_url,
                headers={**self._headers, "X-DashScope-Async": "enable"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            output = data.get("output") or {}

            # Some deployments answer synchronously.
            if output.get("video_url") or output.get("results"):
                result = self._build_result(prompt, model, first_frame_image, output)
                return self._localize_result(result)

            task_id = output.get("task_id")
            if not task_id:
                raise RuntimeError(
                    f"PAI-Token-Video submit returned no task_id: {data}"
                )
            result = self._poll_task(client, task_id, prompt, model, first_frame_image)
            return self._localize_result(result)

    def _poll_task(
        self,
        client: httpx.Client,
        task_id: str,
        prompt: str,
        model: str,
        first_frame_image: Optional[str],
    ) -> VideoGenerationResult:
        task_url = f"{self.base_url}/tasks/{task_id}"
        deadline = time.monotonic() + self.max_poll_wait
        start_time = time.monotonic()
        last_log_time = start_time
        poll_count = 0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            poll_count += 1

            resp = client.get(task_url, headers=self._headers)
            resp.raise_for_status()
            output = (resp.json() or {}).get("output") or {}
            task_status = output.get("task_status")
            logger.debug("PAI-Token-Video task %s status: %s", task_id, task_status)

            now = time.monotonic()
            if now - last_log_time >= _DEFAULT_PROGRESS_LOG_INTERVAL:
                logger.info(
                    "PAI-Token-Video task %s still %s after %.0fs (%d polls).",
                    task_id,
                    task_status,
                    now - start_time,
                    poll_count,
                )
                last_log_time = now

            if task_status == "SUCCEEDED":
                logger.info(
                    "PAI-Token-Video task %s completed after %.0fs (%d polls).",
                    task_id,
                    now - start_time,
                    poll_count,
                )
                return self._build_result(prompt, model, first_frame_image, output)
            if task_status in ("FAILED", "CANCELED"):
                raise RuntimeError(
                    f"PAI-Token-Video task {task_id} {task_status}: "
                    f"{output.get('message', '')}"
                )
            # PENDING / RUNNING -> keep polling.
        raise TimeoutError(
            f"PAI-Token-Video task {task_id} did not complete within "
            f"{self.max_poll_wait}s."
        )

    @staticmethod
    def _build_result(
        prompt: str,
        model: str,
        first_frame_image: Optional[str],
        output: Dict[str, Any],
    ) -> VideoGenerationResult:
        video_urls: List[str] = []
        if output.get("video_url"):
            video_urls.append(output["video_url"])
        for item in output.get("results") or []:
            url = item.get("video_url") or item.get("url")
            if url:
                video_urls.append(url)
        return VideoGenerationResult(
            prompt=prompt,
            video_urls=video_urls,
            first_frame_image=first_frame_image,
            model=model,
            usage=output.get("usage"),
            metadata={"task_id": output.get("task_id")},
        )

    def _localize_result(self, result: VideoGenerationResult) -> VideoGenerationResult:
        """Download remote videos to ``output_dir``, keeping remote URLs in metadata."""
        if not self.output_dir or not result.video_urls:
            return result
        os.makedirs(self.output_dir, exist_ok=True)
        task_id = safe_filename_stem(
            result.metadata.get("task_id") or uuid.uuid4().hex[:12]
        )
        local_paths: List[str] = []
        remote_urls: List[str] = []
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for idx, url in enumerate(result.video_urls):
                if not url.startswith(("http://", "https://")):
                    local_paths.append(url)  # already local
                    continue
                suffix = f"_{idx}" if len(result.video_urls) > 1 else ""
                file_path = os.path.join(self.output_dir, f"{task_id}{suffix}.mp4")
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in resp.iter_bytes():
                            f.write(chunk)
                local_paths.append(file_path)
                remote_urls.append(url)
                logger.info("PAI-Token-Video: video saved to %s.", file_path)
        result.video_urls = local_paths
        if remote_urls:
            result.metadata["remote_urls"] = remote_urls
        return result

    def health_check(self) -> bool:
        try:
            # A lightweight check: verify the API key and base URL are set.
            return bool(self.api_key and self.base_url)
        except Exception as exc:
            logger.warning("PAI-Token-Video health check failed: %s", exc)
            return False

    def close(self) -> None:
        """Nothing to clean up (httpx clients are created per-request)."""
        pass
