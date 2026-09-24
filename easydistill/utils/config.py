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

"""Config loading and env-var expansion utilities."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Match, Optional, TypeVar, Union, cast

import yaml

from .schemas import validate_config

logger = logging.getLogger(__name__)

# Match ${VAR_NAME} or $VAR_NAME in config strings.
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} / $VAR placeholders in config values."""
    if isinstance(value, str):

        def replacer(match: Match) -> str:
            var_name = match.group(1) or match.group(2)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Config references unset environment variable '{var_name}'. "
                    f"Set the variable or remove the placeholder."
                )
            return env_value

        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    with path_obj.open(encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)  # type: ignore[no-any-return]
        return json.load(f)  # type: ignore[no-any-return]


def validate_config_paths(config: Dict[str, Any]) -> None:
    """Validate that configured input paths exist and outputs are writable.

    Raises:
        ValueError: If an input path is missing or an output path is invalid.
    """
    dataset = config.get("dataset", {})
    input_path = dataset.get("input_path")
    if input_path and not Path(input_path).exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    output_path = dataset.get("output_path")
    if output_path:
        output_obj = Path(output_path)
        if output_obj.exists() and output_obj.is_dir():
            raise ValueError(f"Output path is a directory: {output_path}")

    for stage in config.get("pipeline", []):
        stage_output = stage.get("output_path")
        if stage_output:
            stage_obj = Path(stage_output)
            if stage_obj.exists() and stage_obj.is_dir():
                raise ValueError(f"Stage output path is a directory: {stage_output}")


def coerce_int(value: Any) -> Optional[int]:
    """Return *value* as an int, or ``None`` when it is not usable as a count."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10)
        except ValueError:
            pass
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def coerce_float(value: Any) -> Optional[float]:
    """Return *value* as a float, or ``None`` when it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


_Num = TypeVar("_Num", int, float)


def config_number(cfg: Any, key: str, default: _Num, *, where: str) -> _Num:
    """Read a numeric config field without letting a typo kill the caller."""
    raw = cfg.get(key, default) if isinstance(cfg, dict) else default
    value: Optional[Union[int, float]] = (
        coerce_int(raw) if isinstance(default, int) else coerce_float(raw)
    )
    if value is None:
        logger.warning(
            "%s.%s = %r is not a number; using %r instead.",
            where, key, raw, default,
        )
        return default
    return cast(_Num, value)


def config_list(config: Any, key: str, *, where: str) -> List[Any]:
    """Return ``config[key]`` as a list, tolerating the shapes humans write."""
    if not isinstance(config, dict):
        return []
    value = config.get(key)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, int, float, bool)):
        logger.warning(
            "%s.%s = %r is a single value, not a list; reading it as one entry.",
            where, key, value,
        )
        return [value]
    logger.warning("%s.%s = %r cannot be read as a list; ignoring it.", where, key, value)
    return []


def config_section(config: Any, key: str) -> Dict[str, Any]:
    """Return ``config[key]`` as a dict, falling back to ``{}`` for anything else.

    YAML maps a key written with no value to ``None``, not ``{}``; this helper
    keeps the usual ``config.get(key, {}).get(...)`` chain safe against that.
    """
    if not isinstance(config, dict):
        return {}
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def load_expanded_config(path: str) -> Dict[str, Any]:
    """Load a config file, expand environment variables, and validate it."""
    cfg = validate_config(expand_env_vars(load_config(path)))
    validate_config_paths(cfg)
    return cfg
