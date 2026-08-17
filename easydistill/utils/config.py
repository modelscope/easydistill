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
from typing import Any, Dict, Match

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


def load_expanded_config(path: str) -> Dict[str, Any]:
    """Load a config file, expand environment variables, and validate it."""
    cfg = validate_config(expand_env_vars(load_config(path)))
    validate_config_paths(cfg)
    return cfg
