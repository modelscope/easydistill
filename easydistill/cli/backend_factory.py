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

"""Backend factory for CLI runners."""

import logging
import os
from typing import Any, Dict, Optional

from easydistill.backends import (
    EASBackend,
    OpenAIBackend,
    PAIDiffusionBackend,
    PaiTokenBackend,
    PaiTokenVideoBackend,
    PAIVideoBackend,
    QwenImageBackend,
    T2IBackend,
    T2VBackend,
    WanxBackend,
)
from easydistill.backends.base import ModelBackend
from easydistill.utils import expand_env_vars

logger = logging.getLogger(__name__)


def _resolve_backend_value(config: Dict[str, Any], key: str, env_var: str) -> Any:
    """Return config[key] if set, otherwise the value of env_var."""
    value = config.get(key)
    return value if value is not None else os.getenv(env_var)


def _parse_numeric(
    value: Any,
    name: str,
    type_: type,
    min_value: Optional[float] = None,
) -> Any:
    """Parse a numeric config value and enforce an optional minimum bound.

    Raises ValueError with a helpful message if parsing fails or the value is
    below the minimum.
    """
    if value is None:
        return None
    try:
        parsed = type_(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config field '{name}' must be a valid {type_.__name__}, got {value!r}."
        ) from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(
            f"Config field '{name}' must be >= {min_value}, got {parsed}."
        )
    return parsed


def build_backend(config: Dict[str, Any]) -> ModelBackend:
    """Build a ModelBackend from a config dict."""
    config = expand_env_vars(config)
    backend_type = config.get("type", "openai").lower()

    timeout = _parse_numeric(config.get("timeout"), "timeout", float, 0.0)
    timeout = 120.0 if timeout is None else timeout
    max_retries = _parse_numeric(config.get("max_retries"), "max_retries", int, 0.0)
    # Default to 0 so operators own retry/backoff by default. Users can opt in
    # to client-level retries by setting backend.max_retries explicitly.
    max_retries = 0 if max_retries is None else max_retries

    if backend_type == "openai":
        api_key = _resolve_backend_value(config, "api_key", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI backend requires 'api_key' or OPENAI_API_KEY env var.")
        base_url = _resolve_backend_value(config, "base_url", "OPENAI_BASE_URL")
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return OpenAIBackend(
            api_key=api_key,
            base_url=base_url,
            model_id=config.get("model_id"),
            timeout=timeout,
            max_retries=max_retries,
        )
    if backend_type == "pai_token":
        api_key = _resolve_backend_value(config, "api_key", "PAI_TOKEN_API_KEY")
        if not api_key:
            raise ValueError(
                "PAI-Token backend requires 'api_key' or PAI_TOKEN_API_KEY env var."
            )
        base_url = _resolve_backend_value(config, "base_url", "PAI_TOKEN_BASE_URL")
        if not base_url:
            base_url = "https://cn-beijing.pai-token.aliyuncs.com/v1"
        return PaiTokenBackend(
            api_key=api_key,
            base_url=base_url,
            model_id=config.get("model_id"),
            timeout=timeout,
            max_retries=max_retries,
        )
    if backend_type == "pai_eas":
        endpoint_url = _resolve_backend_value(config, "endpoint_url", "EAS_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError(
                "PAI-EAS backend requires 'endpoint_url' or EAS_ENDPOINT_URL env var."
            )
        token = _resolve_backend_value(config, "token", "EAS_TOKEN")
        if not token:
            raise ValueError("PAI-EAS backend requires 'token' or EAS_TOKEN env var.")
        return EASBackend(
            endpoint_url=endpoint_url,
            token=token,
            model_id=config.get("model_id"),
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported backend type: {backend_type}")


def build_t2i_backend(config: Dict[str, Any]) -> T2IBackend:
    """Build a T2IBackend from a config dict.

    Supports ``type: wanx`` (Tongyi Wanxiang via dashscope),
    ``type: qwen_image`` (Qwen-Image via dashscope), and
    ``type: pai_diffusion`` (PAI-EAS deployed SD/Flux/Qwen-Image via
    OpenAI-compatible or async task-based images API).
    """
    config = expand_env_vars(config)
    backend_type = config.get("type", "").lower()

    timeout = _parse_numeric(config.get("timeout"), "timeout", float, 0.0)
    timeout = 120.0 if timeout is None else timeout

    if backend_type == "wanx":
        api_key = _resolve_backend_value(config, "api_key", "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "Wanx backend requires 'api_key' or DASHSCOPE_API_KEY env var."
            )
        poll_interval = _parse_numeric(
            config.get("poll_interval"), "poll_interval", float, 0.0
        )
        max_poll_wait = _parse_numeric(
            config.get("max_poll_wait"), "max_poll_wait", float, 0.0
        )
        return WanxBackend(
            api_key=api_key,
            model_id=config.get("model_id"),
            timeout=timeout,
            poll_interval=2.0 if poll_interval is None else poll_interval,
            max_poll_wait=300.0 if max_poll_wait is None else max_poll_wait,
        )

    if backend_type == "qwen_image":
        api_key = _resolve_backend_value(config, "api_key", "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "Qwen-Image backend requires 'api_key' or DASHSCOPE_API_KEY env var."
            )
        poll_interval = _parse_numeric(
            config.get("poll_interval"), "poll_interval", float, 0.0
        )
        max_poll_wait = _parse_numeric(
            config.get("max_poll_wait"), "max_poll_wait", float, 0.0
        )
        return QwenImageBackend(
            api_key=api_key,
            model_id=config.get("model_id"),
            timeout=timeout,
            poll_interval=2.0 if poll_interval is None else poll_interval,
            max_poll_wait=300.0 if max_poll_wait is None else max_poll_wait,
        )

    if backend_type == "pai_diffusion":
        endpoint_url = _resolve_backend_value(config, "endpoint_url", "EAS_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError(
                "PAI-Diffusion backend requires 'endpoint_url' or EAS_ENDPOINT_URL env var."
            )
        token = _resolve_backend_value(config, "token", "EAS_TOKEN")
        if not token:
            raise ValueError(
                "PAI-Diffusion backend requires 'token' or EAS_TOKEN env var."
            )
        poll_interval = _parse_numeric(
            config.get("poll_interval"), "poll_interval", float, 0.0
        )
        max_poll_wait = _parse_numeric(
            config.get("max_poll_wait"), "max_poll_wait", float, 0.0
        )
        return PAIDiffusionBackend(
            endpoint_url=endpoint_url,
            token=token,
            model_id=config.get("model_id"),
            timeout=timeout,
            auth_prefix=config.get("auth_prefix", "Bearer "),
            output_dir=config.get("output_dir"),
            poll_interval=5.0 if poll_interval is None else poll_interval,
            max_poll_wait=300.0 if max_poll_wait is None else max_poll_wait,
        )

    raise ValueError(f"Unsupported T2I backend type: {backend_type}")


def build_t2v_backend(config: Dict[str, Any]) -> T2VBackend:
    """Build a T2VBackend from a config dict.

    Supports ``type: pai_token_video`` (T2V/I2V models behind the PAI-Token
    gateway, DashScope-style async protocol) and ``type: pai_video``
    (PAI-EAS deployed video models via sync or async task-based videos API).
    """
    config = expand_env_vars(config)
    backend_type = config.get("type", "").lower()

    timeout = _parse_numeric(config.get("timeout"), "timeout", float, 0.0)
    timeout = 300.0 if timeout is None else timeout
    poll_interval = _parse_numeric(
        config.get("poll_interval"), "poll_interval", float, 0.0
    )
    max_poll_wait = _parse_numeric(
        config.get("max_poll_wait"), "max_poll_wait", float, 0.0
    )

    if backend_type == "pai_token_video":
        api_key = _resolve_backend_value(config, "api_key", "PAI_TOKEN_API_KEY")
        if not api_key:
            raise ValueError(
                "PAI-Token-Video backend requires 'api_key' or "
                "PAI_TOKEN_API_KEY env var."
            )
        base_url = _resolve_backend_value(config, "base_url", "PAI_TOKEN_BASE_URL")
        if not base_url:
            raise ValueError(
                "PAI-Token-Video backend requires 'base_url' or "
                "PAI_TOKEN_BASE_URL env var."
            )
        return PaiTokenVideoBackend(
            api_key=api_key,
            base_url=base_url,
            model_id=config.get("model_id"),
            i2v_model_id=config.get("i2v_model_id"),
            i2v_image_field=config.get("i2v_image_field", "media"),
            submit_path=config.get(
                "submit_path", "/services/aigc/video-generation/video-synthesis"
            ),
            timeout=timeout,
            poll_interval=5.0 if poll_interval is None else poll_interval,
            max_poll_wait=1800.0 if max_poll_wait is None else max_poll_wait,
            output_dir=config.get("output_dir"),
        )

    if backend_type == "pai_video":
        endpoint_url = _resolve_backend_value(config, "endpoint_url", "EAS_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError(
                "PAI-Video backend requires 'endpoint_url' or EAS_ENDPOINT_URL env var."
            )
        token = _resolve_backend_value(config, "token", "EAS_TOKEN")
        if not token:
            raise ValueError(
                "PAI-Video backend requires 'token' or EAS_TOKEN env var."
            )
        return PAIVideoBackend(
            endpoint_url=endpoint_url,
            token=token,
            model_id=config.get("model_id"),
            timeout=timeout,
            auth_prefix=config.get("auth_prefix", "Bearer "),
            output_dir=config.get("output_dir"),
            poll_interval=10.0 if poll_interval is None else poll_interval,
            max_poll_wait=1800.0 if max_poll_wait is None else max_poll_wait,
            protocol=config.get("protocol", "legacy"),
            t2v_task=config.get("t2v_task", "t2va"),
            i2v_task=config.get("i2v_task", "fl2va"),
            sglang_short_edge=config.get("sglang_short_edge", 768),
            sglang_aspect_ratio=config.get("sglang_aspect_ratio", "16:9"),
            sglang_duration_seconds=config.get("sglang_duration_seconds", 5.0),
        )

    raise ValueError(f"Unsupported T2V backend type: {backend_type}")


def check_backend_health(backend) -> None:
    """Verify the backend is healthy, raising if the check fails."""
    if not backend.health_check():
        raise RuntimeError(
            "Backend health check failed. Check your backend config and connectivity."
        )


def close_backends(*backends: Any) -> bool:
    """Close all supplied backends, logging but ignoring individual errors.

    Accepts both :class:`~easydistill.backends.base.ModelBackend` and
    :class:`~easydistill.backends.t2i_base.T2IBackend` instances (or any
    object implementing ``close()``). ``None`` entries are skipped.

    Returns:
        True if every backend closed without error, False otherwise.
    """
    all_ok = True
    for backend in backends:
        if backend is None:
            continue
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            logger.error("Failed to close backend %r: %s", backend, exc)
    return all_ok
