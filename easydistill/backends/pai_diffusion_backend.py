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

"""PAI-Diffusion backend for T2I models deployed on PAI-EAS.

Supports two protocols:

1. **Sync mode** (OpenAI-compatible): ``POST /images/generations`` returns
   ``{data: [{url|b64_json}]}`` immediately.  Used by SD/Flux models deployed
   with vLLM or similar OpenAI-compatible servers.

2. **Async mode** (Qwen-Image on EAS): ``POST /images/generations`` returns
   ``{task_id: "..."}``.  The client then polls ``GET /tasks/{task_id}/status``
   until the status is ``completed``, and downloads the image from
   ``GET /tasks/{task_id}/image``.

The mode is auto-detected from the response: if ``task_id`` is present, async
mode is used; otherwise sync mode is used.
"""

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from easydistill.data.models import ImageGenerationResult

from .t2i_base import T2IBackend

logger = logging.getLogger(__name__)

_DEFAULT_PAIDIFF_MODEL = "stable-diffusion-xl"


class PAIDiffusionBackend(T2IBackend):
    """Text-to-image backend for PAI-EAS deployed diffusion models.

    Works with any endpoint that implements the OpenAI ``/images/generations``
    schema (sync mode) **or** the Qwen-Image async task-based protocol.

    Args:
        endpoint_url: Full URL ending with ``/v1`` (e.g.
            ``http://xxx.pai-eas.aliyuncs.com/v1``).
        token: Authentication token.
        model_id: Default model identifier.
        timeout: HTTP timeout in seconds.
        auth_prefix: Prefix for the Authorization header.  Defaults to
            ``"Bearer "``.  Use ``""`` for EAS endpoints that expect a raw
            token without the ``Bearer`` prefix.
        output_dir: Directory to save downloaded images (async mode only).
            If ``None``, images are returned as base64 data URLs.
        poll_interval: Seconds between status polls in async mode.
        max_poll_wait: Maximum seconds to wait for an async task to complete.
    """

    def __init__(
        self,
        endpoint_url: str,
        token: str,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        auth_prefix: str = "Bearer ",
        output_dir: Optional[str] = None,
        poll_interval: float = 5.0,
        max_poll_wait: float = 300.0,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.token = token
        self.model_id = model_id or _DEFAULT_PAIDIFF_MODEL
        self.timeout = timeout
        self._auth_prefix = auth_prefix
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.max_poll_wait = max_poll_wait

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

    @staticmethod
    def _parse_size_to_wh(size: str) -> Tuple[int, int]:
        """Parse ``"1024*1024"`` or ``"1024x1024"`` to ``(1024, 1024)``."""
        parts = size.replace("*", "x").split("x")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 1024, 1024

    # ------------------------------------------------------------------
    # T2IBackend interface
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate an image from a text prompt.

        Automatically detects sync vs async mode from the response.
        Extra kwargs (e.g. ``seed``, ``negative_prompt``, ``infer_steps``,
        ``cfg_scale``) are passed through to the API for backends that
        support them (notably Qwen-Image on EAS).
        """
        model = model_id or self.model_id
        api_size = size.replace("*", "x")
        width, height = self._parse_size_to_wh(size)

        # Build payload with both sync and async fields for maximum
        # compatibility.  Sync endpoints ignore width/height; async endpoints
        # ignore model/size/n.
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": api_size,
            "n": n,
            "width": width,
            "height": height,
        }
        payload.update(kwargs)

        url = f"{self.endpoint_url}/images/generations"
        logger.info(
            "PAI-Diffusion: calling %s (model=%s, size=%s, n=%d).",
            url,
            model,
            api_size,
            n,
        )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Auto-detect: async task-based response (Qwen-Image on EAS)
            if "task_id" in data:
                return self._handle_async_response(data, prompt, model, client)

            # Sync OpenAI-compatible response
            return self._handle_sync_response(data, prompt, model)

    def _handle_sync_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        model: str,
    ) -> ImageGenerationResult:
        """Handle synchronous OpenAI-compatible ``{data: [...]}`` response."""
        images: List[Any] = data.get("data") or []
        image_urls: List[str] = []
        for item in images:
            url = item.get("url")
            if url:
                image_urls.append(url)
            else:
                b64 = item.get("b64_json")
                if b64:
                    image_urls.append(f"data:image/png;base64,{b64}")

        return ImageGenerationResult(
            prompt=prompt,
            image_urls=image_urls,
            model=model,
            usage=data.get("usage"),
            metadata={"endpoint": self.endpoint_url, "mode": "sync"},
        )

    def _handle_async_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        model: str,
        client: httpx.Client,
    ) -> ImageGenerationResult:
        """Handle async task-based response (Qwen-Image on PAI-EAS).

        Flow: submit → poll ``/tasks/{task_id}/status`` → download
        ``/tasks/{task_id}/image``.
        """
        task_id = data["task_id"]
        base_url = self._base_url
        logger.info("PAI-Diffusion: async task %s submitted, polling...", task_id)

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
                logger.info("PAI-Diffusion: task %s completed.", task_id)
                break
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"Image generation task {task_id} failed: {error}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Image generation task {task_id} timed out after "
                    f"{self.max_poll_wait}s."
                )

            time.sleep(self.poll_interval)

        # Download the image
        image_resp = client.get(
            f"{base_url}/tasks/{task_id}/image",
            headers=self._headers,
        )
        image_resp.raise_for_status()
        image_bytes = image_resp.content

        # Determine content type and extension
        content_type = image_resp.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            ext = "png"
        elif "webp" in content_type:
            ext = "webp"
        else:
            ext = "jpg"

        image_urls: List[str] = []
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir, f"{task_id}.{ext}")
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            image_urls.append(file_path)
            logger.info("PAI-Diffusion: image saved to %s.", file_path)
        else:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            image_urls.append(f"data:image/{ext};base64,{b64}")

        return ImageGenerationResult(
            prompt=prompt,
            image_urls=image_urls,
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
            logger.warning("PAI-Diffusion health check failed: %s", exc)
            return False

    def close(self) -> None:
        """Nothing to clean up (httpx clients are created per-request)."""
        pass
