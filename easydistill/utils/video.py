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

"""Video helpers: frame sampling for VLM-based evaluation.

Frames are sent to VLM judges as base64 JPEG data URIs in temporal order
(the multi-image payload pattern: vision models interpret an ordered image
sequence as a video).  Decoding uses OpenCV, which is an optional
dependency — install with ``pip install easydistill[t2v]``.
"""

import base64
import logging
import os
import re
from typing import List, NamedTuple, Optional

logger = logging.getLogger(__name__)

DEFAULT_FRAME_COUNT = 8
DEFAULT_FRAME_MAX_SIZE = 768
DEFAULT_JPEG_QUALITY = 85


class VideoFrame(NamedTuple):
    """One sampled frame: its timestamp (seconds, None if unknown) and data URI."""

    timestamp: Optional[float]
    data_url: str


_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


def load_video_to_data_url(video_path: str, max_bytes: Optional[int] = None) -> str:
    """Load a local video file and return it as a base64 data URL.

    Args:
        video_path: Local video file path (``file://`` prefix accepted).
        max_bytes: Optional size cap; raises ValueError when exceeded so
            callers can fall back to URL transport instead of building a
            payload the endpoint would reject.
    """
    path = re.sub(r"^file://", "", video_path)
    if not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")
    size = os.path.getsize(path)
    if max_bytes is not None and size > max_bytes:
        raise ValueError(
            f"Video {path} is {size} bytes, exceeding the {max_bytes}-byte "
            "cap for base64 transport."
        )
    mime = _VIDEO_MIME_TYPES.get(os.path.splitext(path)[1].lower(), "video/mp4")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video frame sampling. "
            "Install it with: pip install easydistill[t2v]"
        ) from exc
    return cv2


def sample_video_frames(
    video_path: str,
    count: int = DEFAULT_FRAME_COUNT,
    max_size: int = DEFAULT_FRAME_MAX_SIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> List[VideoFrame]:
    """Uniformly sample ``count`` frames from a local video file.

    Frames are resized so their long side is at most ``max_size`` (to bound
    VLM token cost) and encoded as JPEG base64 data URIs.

    Args:
        video_path: Local video file path (``file://`` prefix accepted).
        count: Number of frames to sample (uniform over the full duration,
            always including the first and last decodable frame).
        max_size: Maximum long-side resolution of the encoded frames.
        jpeg_quality: JPEG encoding quality (1-100).

    Returns:
        List of :class:`VideoFrame` in temporal order.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the video cannot be opened or no frame decodes.
    """
    cv2 = _require_cv2()
    path = re.sub(r"^file://", "", video_path)
    if not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            raise ValueError(f"Video has no decodable frames: {path}")

        if total <= count:
            indices = list(range(total))
        else:
            step = (total - 1) / (count - 1)
            indices = sorted({round(i * step) for i in range(count)})

        frames: List[VideoFrame] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to decode frame %d of %s.", idx, path)
                continue
            height, width = frame.shape[:2]
            long_side = max(height, width)
            if long_side > max_size:
                scale = max_size / long_side
                frame = cv2.resize(
                    frame, (int(width * scale), int(height * scale))
                )
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not ok:
                logger.warning("Failed to encode frame %d of %s.", idx, path)
                continue
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            timestamp = (idx / fps) if fps > 0 else None
            frames.append(
                VideoFrame(
                    timestamp=timestamp,
                    data_url=f"data:image/jpeg;base64,{encoded}",
                )
            )
        if not frames:
            raise ValueError(f"No frame could be decoded from video: {path}")
        return frames
    finally:
        cap.release()
