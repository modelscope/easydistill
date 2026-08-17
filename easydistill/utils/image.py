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

"""Image helpers for multi-modal inputs."""

import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_image_url(value: str) -> bool:
    """Return True if value looks like a URL or a base64 data URL."""
    if not isinstance(value, str):
        return False
    return value.startswith(("http://", "https://", "data:image"))


def _guess_mime_type(path: str) -> str:
    """Guess MIME type from file extension, defaulting to image/png."""
    ext = os.path.splitext(path)[1].lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mapping.get(ext, "image/png")


def load_image_to_data_url(path: str) -> str:
    """Load a local image file and return it as a base64 data URL."""
    path = re.sub(r"^file://", "", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type = _guess_mime_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_image_reference(ref: str) -> str:
    """Normalize a single image reference to a URL usable by vision APIs.

    - http(s) URLs are returned as-is.
    - base64 data URLs are returned as-is.
    - local paths (with or without file://) are converted to base64 data URLs.
    """
    if not isinstance(ref, str):
        raise TypeError(f"Image reference must be a string, got {type(ref).__name__}")
    if is_image_url(ref):
        return ref
    return load_image_to_data_url(ref)


def normalize_image_references(refs: Optional[List[str]]) -> List[str]:
    """Normalize a list of image references."""
    if not refs:
        return []
    return [normalize_image_reference(ref) for ref in refs]


def _extract_text_from_content(content: Any) -> str:
    """Extract text from a pre-built multi-modal content list.

    Falls back to ``str(content)`` if no text items are found.
    """
    if not isinstance(content, list):
        return str(content)
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return " ".join(texts) if texts else str(content)


def content_has_images(content: Any) -> bool:
    """Return True if ``content`` is a list containing image_url items."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == "image_url" for item in content
    )


def format_prompt_safely(template: Optional[str], **kwargs: Any) -> str:
    """Format a prompt template, stringifying any multi-modal content lists.

    This prevents ``TypeError`` when a value is a pre-built OpenAI content list
    instead of a plain string. Text items are extracted from content lists;
    image items are ignored because images are passed separately to vision APIs.
    Missing placeholders are left unchanged and a warning is logged.
    """
    if template is None:
        return ""
    string_kwargs = {
        k: v if isinstance(v, str) else _extract_text_from_content(v) for k, v in kwargs.items()
    }
    try:
        return template.format(**string_kwargs)
    except KeyError as exc:
        logger.warning(
            "Prompt template is missing placeholder '%s'; returning template unchanged.",
            exc,
        )
        return template


def build_multimodal_user_content(
    instruction: Any,
    images: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build an OpenAI-compatible user content list from text + images.

    If instruction is already a list, it is treated as a pre-built content list.
    Otherwise it is wrapped as a text item. Images are inserted before the text
    item, which is the convention used by most vision APIs.
    """
    content: List[Dict[str, Any]] = []
    if images:
        for image_url in normalize_image_references(images):
            content.append({"type": "image_url", "image_url": {"url": image_url}})

    if isinstance(instruction, list):
        content.extend(instruction)
    else:
        text = str(instruction)
        if text:
            content.append({"type": "text", "text": text})
    return content
