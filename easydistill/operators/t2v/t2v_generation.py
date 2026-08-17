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

"""T2V generation operator: optimized prompt (+optional first frame) -> teacher videos."""

import base64
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from easydistill.backends.t2v_base import T2VBackend
from easydistill.utils import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
    progress,
)

from .resume import RowCheckpointWriter

logger = logging.getLogger(__name__)

# Video generation is slow and expensive; default to sequential execution.
_DEFAULT_T2V_MAX_WORKERS = 1


def _image_dimensions(image_ref: str) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` of a local or data-URL image, else None.

    http(s) references are left untouched (None) so remote frames always
    pass the size check; decoding them here would block generation.
    """
    if not isinstance(image_ref, str) or not image_ref:
        return None
    try:
        import cv2  # noqa: PLC0415 - optional dependency, imported lazily
        import numpy as np  # noqa: PLC0415

        if image_ref.startswith(("http://", "https://")):
            return None
        if image_ref.startswith("data:"):
            header, _, payload = image_ref.partition(",")
            if ";base64" not in header or not payload:
                return None
            buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(re.sub(r"^file://", "", image_ref), cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read image dimensions from %r: %s", image_ref, exc)
        return None
    if img is None:
        return None
    height, width = img.shape[:2]
    return width, height


class T2VGenerationOperator:
    """Generate videos from prompts using a T2V backend.

    Mirrors :class:`T2IGenerationOperator`: concurrency and retry logic are
    self-contained because the T2V backend uses a submit-poll protocol rather
    than chat completions.  Rows carrying a ``first_frame_image`` field run in
    I2V mode; rows without it run plain T2V.  Both kinds may be mixed in one
    batch.

    Configurable fields:
      - model_id: model identifier passed to the T2V backend.
      - size: video resolution string (e.g. "1280*720").
      - duration: video duration in seconds (backend default when omitted).
      - max_workers: number of concurrent workers (default 1).
      - retry_attempts: retries on transient failures (default 3).
      - retry_backoff_base: base delay for exponential backoff (default 1.0).
      - retry_max_wait: max wait between retries (default 30.0).
      - show_progress: whether to show tqdm progress bar.
      - raise_on_error: if True, raise on final generation failure.
      - prompt_key: row key to read the prompt from (default "optimized_prompt").
      - first_frame_key: row key to read the conditioning first-frame image
        from (default "first_frame_image").
      - checkpoint_path: when set, each completed row is appended (and fsynced)
        to this JSONL file immediately, so a crashed run can resume without
        re-generating finished videos.  Normally injected by the pipeline's
        ``resume: true`` stage option.

    Per-row resolution control (T2V rows only):
      - row ``resolution``: overrides the configured resolution tier for that
        row; the special value ``auto`` keeps the configured tier and lets the
        prompt-optimize stage pick a content-driven ``ratio``.
      - row ``ratio``: overrides the configured aspect ratio for that row.
      - I2V rows follow their first frame: resolution / ratio / size knobs
        are never sent for them (a per-row value logs a warning).
      - i2v_frame_check: "off" (default) | "warn" | "skip" | "raise" —
        validate the first frame's dimensions before generating.
      - i2v_frame_min_edge: minimum shorter-edge pixels (default 256).
      - i2v_frame_max_aspect: maximum aspect ratio, long/short (default 3.0).
    """

    name = "t2v_generate"

    def __init__(self, backend: T2VBackend, config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.config = config or {}
        self.model_id = self.config.get("model_id")
        # Resolution control is backend-specific: legacy APIs take `size`,
        # newer DashScope/EAS APIs take `resolution` / `ratio` / `target`
        # knobs passed through as extra kwargs.  No default is imposed here.
        self.size = self.config.get("size")
        duration = self.config.get("duration")
        self.duration = float(duration) if duration is not None else None
        self.max_workers = int(self.config.get("max_workers", _DEFAULT_T2V_MAX_WORKERS))
        self.retry_attempts = int(self.config.get("retry_attempts", DEFAULT_RETRY_ATTEMPTS))
        self.retry_backoff_base = float(
            self.config.get("retry_backoff_base", DEFAULT_RETRY_BACKOFF_BASE)
        )
        self.retry_max_wait = float(self.config.get("retry_max_wait", DEFAULT_RETRY_MAX_WAIT))
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self.raise_on_error = bool(self.config.get("raise_on_error") or False)
        self.prompt_key = self.config.get("prompt_key", "optimized_prompt")
        self.first_frame_key = self.config.get("first_frame_key", "first_frame_image")
        checkpoint_path = self.config.get("checkpoint_path")
        self._checkpoint = RowCheckpointWriter(checkpoint_path) if checkpoint_path else None
        self.i2v_frame_check = str(self.config.get("i2v_frame_check") or "off").lower()
        if self.i2v_frame_check not in ("off", "warn", "skip", "raise"):
            raise ValueError(
                f"Invalid i2v_frame_check {self.i2v_frame_check!r}; "
                "expected off, warn, skip or raise."
            )
        self.i2v_frame_min_edge = int(self.config.get("i2v_frame_min_edge", 256))
        self.i2v_frame_max_aspect = float(self.config.get("i2v_frame_max_aspect", 3.0))
        # Pass through any extra kwargs to the backend.
        self._extra_kwargs: Dict[str, Any] = {
            k: v
            for k, v in self.config.items()
            if k
            not in {
                "model_id",
                "size",
                "duration",
                "max_workers",
                "retry_attempts",
                "retry_backoff_base",
                "retry_max_wait",
                "show_progress",
                "raise_on_error",
                "prompt_key",
                "first_frame_key",
                "checkpoint_path",
                "resume",
                "i2v_frame_check",
                "i2v_frame_min_edge",
                "i2v_frame_max_aspect",
            }
        }

    def _generate_one(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate videos for a single row, retrying transient failures."""
        prompt = row.get(self.prompt_key) or row.get("prompt") or ""
        if not prompt:
            logger.warning("Row %s has empty prompt, skipping.", row.get("id"))
            return None
        first_frame_image = row.get(self.first_frame_key) or None

        size = self.size
        extra_kwargs = dict(self._extra_kwargs)
        if first_frame_image:
            # I2V follows the conditioning frame: framing knobs are never sent.
            dropped = [k for k in ("resolution", "ratio") if extra_kwargs.pop(k, None)]
            if size is not None:
                dropped.append("size")
                size = None
            if row.get("resolution") or row.get("ratio"):
                logger.warning(
                    "Row %s is I2V; its resolution/ratio fields are ignored "
                    "(output follows the first frame).",
                    row.get("id"),
                )
            elif dropped:
                logger.debug(
                    "Row %s is I2V; dropping configured %s.",
                    row.get("id"),
                    ", ".join(dropped),
                )
            if self.i2v_frame_check != "off" and not self._first_frame_ok(row, first_frame_image):
                return None
        else:
            row_resolution = row.get("resolution")
            if row_resolution and str(row_resolution).lower() != "auto":
                extra_kwargs["resolution"] = str(row_resolution)
            row_ratio = row.get("ratio")
            if row_ratio:
                extra_kwargs["ratio"] = str(row_ratio)

        last_exc: Optional[Exception] = None
        total_attempts = self.retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            try:
                result = self.backend.generate_video(
                    prompt=prompt,
                    model_id=self.model_id,
                    size=size,
                    duration=self.duration,
                    first_frame_image=first_frame_image,
                    **extra_kwargs,
                )
                if not result.video_urls:
                    logger.warning(
                        "T2V generation for row %s returned no videos.", row.get("id")
                    )
                    return None
                new_row = dict(row)
                new_row["video_urls"] = result.video_urls
                new_row["t2v_model"] = result.model
                new_row["t2v_mode"] = "i2v" if first_frame_image else "t2v"
                if result.usage:
                    new_row["t2v_usage"] = result.usage
                # When the backend localized the videos, keep the remote URLs
                # so URL-transport consumers (e.g. the omni checker) can use them.
                remote_urls = (result.metadata or {}).get("remote_urls")
                if remote_urls:
                    new_row["video_remote_urls"] = remote_urls
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
                    "T2V generation failed for row %s (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    row.get("id"),
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "Failed to generate videos for row %s after %d attempts: %s",
            row.get("id"),
            attempt,
            last_exc,
        )
        if self.raise_on_error and last_exc is not None:
            raise last_exc
        return None

    def _first_frame_ok(self, row: Dict[str, Any], first_frame_image: str) -> bool:
        """Validate the I2V first frame's dimensions per ``i2v_frame_check``.

        Returns False when the row should be skipped.  Frames whose size
        cannot be determined locally (e.g. plain http(s) URLs) pass through
        untouched.
        """
        dims = _image_dimensions(first_frame_image)
        if dims is None:
            return True
        width, height = dims
        short_edge, long_edge = min(width, height), max(width, height)
        issues = []
        if short_edge < self.i2v_frame_min_edge:
            issues.append(
                f"short edge {short_edge}px < i2v_frame_min_edge {self.i2v_frame_min_edge}px"
            )
        if short_edge and long_edge / short_edge > self.i2v_frame_max_aspect:
            issues.append(
                f"aspect ratio {long_edge / short_edge:.2f} > "
                f"i2v_frame_max_aspect {self.i2v_frame_max_aspect}"
            )
        if not issues:
            return True
        message = (
            f"Row {row.get('id')} first frame ({width}x{height}) failed the "
            f"resolution check: {'; '.join(issues)}."
        )
        if self.i2v_frame_check == "raise":
            raise ValueError(message)
        logger.warning("%s%s", message, " Skipping row." if self.i2v_frame_check == "skip" else "")
        return self.i2v_frame_check != "skip"

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
            desc="Generating videos",
        ):
            result = self._generate_one(row)
            if result is not None:
                if self._checkpoint is not None:
                    self._checkpoint.append(result)
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
                desc="Generating videos",
            )
            for future, idx in futures_iter:
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "T2V generation task at index %d raised: %s",
                        idx,
                        exc,
                    )
                    result = None
                if result is not None:
                    if self._checkpoint is not None:
                        self._checkpoint.append(result)
                    results[idx] = result
        return [r for r in results if r is not None]
