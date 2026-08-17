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

"""PAI video backend for T2V/I2V models deployed on PAI-EAS.

Supports three protocols:

1. **sglang mode** (``protocol: sglang``, the real-world deployment):
   ``POST /v1/videos`` with ``{"prompt", "task", "target": {...}}`` returns
   a video object with an ``id``; the client polls
   ``GET /v1/videos/{id}`` until ``status`` is ``completed`` and downloads
   the mp4 from ``GET /v1/videos/{id}/content``.  This is the OpenAI-style
   video API served by sglang-deployed video models (e.g. MiniMax-H3 with
   tasks ``t2va`` / ``fl2va`` / ``ref2va``).

2. **Sync mode**: ``POST /videos/generations`` returns
   ``{data: [{url}]}`` immediately.  Used by video models deployed with
   OpenAI-compatible servers.

3. **Async mode**: ``POST /videos/generations`` returns
   ``{task_id: "..."}``.  The client then polls ``GET /tasks/{task_id}/status``
   until the status is ``completed``, and downloads the video from
   ``GET /tasks/{task_id}/video``.

For the legacy protocol the mode is auto-detected from the response: if
``task_id`` is present, async mode is used; otherwise sync mode is used.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from easydistill.data.models import VideoGenerationResult
from easydistill.utils import normalize_image_reference, safe_filename_stem

from .t2v_base import T2VBackend

logger = logging.getLogger(__name__)

_DEFAULT_PAI_VIDEO_MODEL = "wan-video"


class PAIVideoBackend(T2VBackend):
    """Text/image-to-video backend for PAI-EAS deployed video models.

    Works with any endpoint that implements an OpenAI-style
    ``/videos/generations`` schema (sync mode) **or** the async task-based
    protocol used by PAI-EAS deployments.

    Args:
        endpoint_url: Full URL ending with ``/v1`` (e.g.
            ``http://xxx.pai-eas.aliyuncs.com/v1``).
        token: Authentication token.
        model_id: Default model identifier.
        timeout: HTTP timeout in seconds.
        auth_prefix: Prefix for the Authorization header.  Defaults to
            ``"Bearer "``.  Use ``""`` for EAS endpoints that expect a raw
            token without the ``Bearer`` prefix.
        output_dir: Directory to save downloaded videos.  Required for
            async/sglang modes that return binary video data.
        poll_interval: Seconds between status polls in async mode.
        max_poll_wait: Maximum seconds to wait for an async task to complete.
        protocol: ``"legacy"`` (default; ``/videos/generations`` sync/async
            auto-detect) or ``"sglang"`` (OpenAI-style ``/v1/videos`` API
            used by sglang-deployed video models).
        t2v_task / i2v_task: sglang task names for T2V / I2V rows
            (defaults ``t2va`` / ``fl2va``).
        sglang_short_edge / sglang_aspect_ratio / sglang_duration_seconds:
            sglang ``target`` defaults (``768`` / ``"16:9"`` / ``5``);
            ``duration`` on the call and ``target`` in kwargs override them.
    """

    def __init__(
        self,
        endpoint_url: str,
        token: str,
        model_id: Optional[str] = None,
        timeout: float = 300.0,
        auth_prefix: str = "Bearer ",
        output_dir: Optional[str] = None,
        poll_interval: float = 10.0,
        max_poll_wait: float = 1800.0,
        protocol: str = "legacy",
        t2v_task: str = "t2va",
        i2v_task: str = "fl2va",
        sglang_short_edge: int = 768,
        sglang_aspect_ratio: str = "16:9",
        sglang_duration_seconds: float = 5.0,
    ):
        if protocol not in ("legacy", "sglang"):
            raise ValueError(
                f"PAIVideoBackend protocol must be 'legacy' or 'sglang', "
                f"got {protocol!r}."
            )
        self.endpoint_url = endpoint_url.rstrip("/")
        self.token = token
        self.model_id = model_id or _DEFAULT_PAI_VIDEO_MODEL
        self.timeout = timeout
        self._auth_prefix = auth_prefix
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.max_poll_wait = max_poll_wait
        self.protocol = protocol
        self.t2v_task = t2v_task
        self.i2v_task = i2v_task
        self.sglang_short_edge = int(sglang_short_edge)
        self.sglang_aspect_ratio = str(sglang_aspect_ratio)
        self.sglang_duration_seconds = float(sglang_duration_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _base_url(self) -> str:
        """Base URL without ``/v1`` suffix (for task polling endpoints)."""
        url = self.endpoint_url
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"{self._auth_prefix}{self.token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # T2VBackend interface
    # ------------------------------------------------------------------

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

        Automatically detects sync vs async mode from the response (legacy
        protocol).  Extra kwargs (e.g. ``fps``, ``seed``, ``negative_prompt``)
        are passed through to the API for backends that support them.
        """
        if self.protocol == "sglang":
            return self._generate_sglang(
                prompt, model_id, duration, first_frame_image, kwargs
            )

        model = model_id or self.model_id
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        if size:
            payload["size"] = size.replace("*", "x")
        if duration is not None:
            payload["duration"] = duration
        if first_frame_image:
            # Local paths / file:// refs become base64 data URLs; http(s)
            # URLs pass through unchanged.
            payload["image_url"] = normalize_image_reference(first_frame_image)
        payload.update(kwargs)

        url = f"{self.endpoint_url}/videos/generations"
        logger.info(
            "PAI-Video: calling %s (model=%s, size=%s, mode=%s).",
            url,
            model,
            size,
            "I2V" if first_frame_image else "T2V",
        )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Auto-detect: async task-based response
            if "task_id" in data:
                return self._handle_async_response(
                    data, prompt, model, first_frame_image, client
                )

            # Sync OpenAI-style response
            return self._handle_sync_response(data, prompt, model, first_frame_image)

    # ------------------------------------------------------------------
    # sglang /v1/videos protocol
    # ------------------------------------------------------------------

    def _build_sglang_target(
        self, duration: Optional[float], extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the sglang ``target`` spec for one generation call."""
        target = extra.pop("target", None)
        if isinstance(target, dict) and target:
            return dict(target)
        seconds = duration if duration is not None else self.sglang_duration_seconds
        return {
            "short_edge": self.sglang_short_edge,
            "aspect_ratio": self.sglang_aspect_ratio,
            "duration_seconds": seconds,
        }

    def _generate_sglang(
        self,
        prompt: str,
        model_id: Optional[str],
        duration: Optional[float],
        first_frame_image: Optional[str],
        extra: Dict[str, Any],
    ) -> VideoGenerationResult:
        """Submit-poll-download via the sglang OpenAI-style /v1/videos API."""
        model = model_id or self.model_id
        task = self.i2v_task if first_frame_image else self.t2v_task
        target = self._build_sglang_target(duration, extra)
        submit_url = f"{self._base_url}/v1/videos"

        payload: Dict[str, Any] = {"prompt": prompt, "task": task, "target": target}
        reference_url: Optional[str] = None
        if first_frame_image:
            ref = normalize_image_reference(first_frame_image)
            if ref.startswith(("http://", "https://")):
                reference_url = ref
                payload["reference_url"] = ref
            else:
                raise ValueError(
                    "PAI-Video (sglang) I2V requires an http(s) first-frame "
                    "URL (the service fetches it via 'reference_url'); got a "
                    "local/base64 reference. Upload the frame first or use a "
                    "DashScope-style backend for local images."
                )
        payload.update(extra)
        logger.info(
            "PAI-Video (sglang): submitting %s to %s (target=%s).",
            task,
            submit_url,
            target,
        )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(submit_url, headers=self._headers, json=payload)
            resp.raise_for_status()
            video = resp.json()
            video_id = video.get("id")
            if not video_id:
                raise RuntimeError(
                    f"PAI-Video (sglang) submit returned no video id: {video}"
                )

            status_url = f"{self._base_url}/v1/videos/{video_id}"
            deadline = time.monotonic() + self.max_poll_wait
            start_time = time.monotonic()
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"PAI-Video (sglang) video {video_id} did not complete "
                        f"within {self.max_poll_wait}s."
                    )
                time.sleep(self.poll_interval)
                status_resp = client.get(status_url, headers=self._headers)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data.get("status")
                logger.debug(
                    "PAI-Video (sglang) video %s status: %s (%s%%).",
                    video_id,
                    status,
                    status_data.get("progress"),
                )
                if status == "completed":
                    break
                if status in ("failed", "cancelled", "canceled"):
                    raise RuntimeError(
                        f"PAI-Video (sglang) video {video_id} {status}: "
                        f"{status_data.get('error')}"
                    )

            logger.info(
                "PAI-Video (sglang): video %s completed in %.0fs.",
                video_id,
                time.monotonic() - start_time,
            )

            # Prefer a served URL; otherwise download the binary content.
            remote_url = status_data.get("url")
            if remote_url:
                remote_resp = client.get(remote_url, follow_redirects=True)
                remote_resp.raise_for_status()
                video_bytes = remote_resp.content
            else:
                content_resp = client.get(
                    f"{self._base_url}/v1/videos/{video_id}/content",
                    headers=self._headers,
                )
                content_resp.raise_for_status()
                video_bytes = content_resp.content

        if not self.output_dir:
            raise RuntimeError(
                "PAI-Video (sglang) returns binary video data; set "
                "t2v_backend.output_dir to save it."
            )
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"{safe_filename_stem(str(video_id))}.mp4")
        with open(file_path, "wb") as f:
            f.write(video_bytes)
        logger.info("PAI-Video (sglang): video saved to %s.", file_path)

        metadata: Dict[str, Any] = {
            "endpoint": self.endpoint_url,
            "mode": "sglang",
            "task": task,
            "video_id": video_id,
        }
        if reference_url:
            metadata["reference_url"] = reference_url
        result = VideoGenerationResult(
            prompt=prompt,
            video_urls=[file_path],
            first_frame_image=first_frame_image,
            model=model,
            usage=None,
            metadata=metadata,
        )
        if remote_url:
            result.metadata["remote_urls"] = [remote_url]
        return result

    def _handle_sync_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        model: str,
        first_frame_image: Optional[str],
    ) -> VideoGenerationResult:
        """Handle synchronous ``{data: [...]}`` response."""
        items: List[Any] = data.get("data") or []
        video_urls: List[str] = []
        for item in items:
            url = item.get("url") or item.get("video_url")
            if url:
                video_urls.append(url)

        return VideoGenerationResult(
            prompt=prompt,
            video_urls=video_urls,
            first_frame_image=first_frame_image,
            model=model,
            usage=data.get("usage"),
            metadata={"endpoint": self.endpoint_url, "mode": "sync"},
        )

    def _handle_async_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        model: str,
        first_frame_image: Optional[str],
        client: httpx.Client,
    ) -> VideoGenerationResult:
        """Handle async task-based response (video models on PAI-EAS).

        Flow: submit → poll ``/tasks/{task_id}/status`` → download
        ``/tasks/{task_id}/video``.
        """
        task_id = data["task_id"]
        base_url = self._base_url
        logger.info("PAI-Video: async task %s submitted, polling...", task_id)

        # Poll for completion
        deadline = time.monotonic() + self.max_poll_wait
        while True:
            status_resp = client.get(
                f"{base_url}/tasks/{task_id}/status",
                headers=self._headers,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")

            if status == "completed":
                logger.info("PAI-Video: task %s completed.", task_id)
                break
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"Video generation task {task_id} failed: {error}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Video generation task {task_id} timed out after "
                    f"{self.max_poll_wait}s."
                )

            time.sleep(self.poll_interval)

        # The completed status may carry the result URL directly.
        video_urls: List[str] = []
        result_url = status_data.get("video_url") or status_data.get("url")
        if result_url:
            video_urls.append(result_url)
        else:
            # Otherwise download the video binary.
            video_resp = client.get(
                f"{base_url}/tasks/{task_id}/video",
                headers=self._headers,
            )
            video_resp.raise_for_status()
            video_bytes = video_resp.content

            content_type = video_resp.headers.get("content-type", "video/mp4")
            ext = "webm" if "webm" in content_type else "mp4"

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                file_path = os.path.join(self.output_dir, f"{task_id}.{ext}")
                with open(file_path, "wb") as f:
                    f.write(video_bytes)
                video_urls.append(file_path)
                logger.info("PAI-Video: video saved to %s.", file_path)
            else:
                raise RuntimeError(
                    "PAI-Video async response returned binary video data but "
                    "no 'output_dir' is configured to save it. Set "
                    "t2v_backend.output_dir in your config."
                )

        return VideoGenerationResult(
            prompt=prompt,
            video_urls=video_urls,
            first_frame_image=first_frame_image,
            model=model,
            usage=None,
            metadata={
                "endpoint": self.endpoint_url,
                "mode": "async",
                "task_id": task_id,
            },
        )

    def health_check(self) -> bool:
        """Check if the endpoint is reachable.

        Tries ``GET /models`` first (OpenAI-compatible endpoints).  If that
        returns non-200, falls back to checking the base URL (some EAS
        deployments don't have ``/models``).
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.endpoint_url}/models",
                    headers=self._headers,
                )
                if resp.status_code == 200:
                    return True
                # Some EAS deployments don't have /models endpoint.
                # Any non-5xx response from the base URL means the server is up.
                resp = client.get(self._base_url, headers=self._headers)
                return resp.status_code < 500
        except Exception as exc:
            logger.warning("PAI-Video health check failed: %s", exc)
            return False

    def close(self) -> None:
        """Nothing to clean up (httpx clients are created per-request)."""
        pass
