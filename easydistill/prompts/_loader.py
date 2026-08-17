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

"""Helpers for loading and selecting prompt templates.

The resolution helpers enforce a single priority across the codebase:

1. File path provided in config (`prompt_template_file` / `prompts_file`).
2. Inline value provided in config (`prompt_template` / `prompts`).
3. Package default constant passed by the caller.
"""

from pathlib import Path
from typing import Any, Dict, Optional, overload

from easydistill.utils import load_config


def load_prompt_template_from_file(path: str) -> str:
    """Load a single prompt template from a text file."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_prompts_from_file(path: str) -> Dict[str, Any]:
    """Load a dict of prompt templates from a YAML/JSON file."""
    data = load_config(path)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a mapping of metric/prompt.")
    return data


def _strip_prompt_values(mapping: Dict[str, Any]) -> Dict[str, str]:
    """Remove a single trailing newline from string prompt values."""
    return {
        key: (value.rstrip("\n") if isinstance(value, str) else value)
        for key, value in mapping.items()
    }


@overload
def resolve_prompt(
    config: Dict[str, Any],
    *,
    template_key: str = ...,
    file_key: str = ...,
    default: str,
    default_file: Optional[str] = ...,
) -> str: ...


@overload
def resolve_prompt(
    config: Dict[str, Any],
    *,
    template_key: str = ...,
    file_key: str = ...,
    default: Optional[str] = ...,
    default_file: str,
) -> str: ...


@overload
def resolve_prompt(
    config: Dict[str, Any],
    *,
    template_key: str = ...,
    file_key: str = ...,
    default: None = ...,
    default_file: None = ...,
) -> Optional[str]: ...


def resolve_prompt(
    config: Dict[str, Any],
    *,
    template_key: str = "prompt_template",
    file_key: str = "prompt_template_file",
    default: Optional[str] = None,
    default_file: Optional[str] = None,
) -> Optional[str]:
    """Select a single prompt template from config or a fallback default.

    Resolution order:
      1. ``config[file_key]`` — load template from the referenced text file.
      2. ``config[template_key]`` — use the inline template string.
      3. ``default`` — caller-provided fallback template string.
      4. ``default_file`` — caller-provided fallback template file path.

    Raises:
        FileNotFoundError: If a configured or fallback prompt template file does
        not exist.
        TypeError: If the inline template in config is not a string.
    """
    file_path = config.get(file_key)
    if file_path is not None:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Prompt template file not found: {file_path}")
        return load_prompt_template_from_file(file_path).rstrip("\n")
    if template_key in config and config[template_key] is not None:
        template = config[template_key]
        if not isinstance(template, str):
            raise TypeError(
                f"Config key '{template_key}' must be a string, got {type(template).__name__}"
            )
        return template
    if default_file is not None:
        if not Path(default_file).is_file():
            raise FileNotFoundError(f"Default prompt template file not found: {default_file}")
        return load_prompt_template_from_file(default_file).rstrip("\n")
    return default


def resolve_prompts(
    config: Dict[str, Any],
    defaults: Optional[Dict[str, str]] = None,
    *,
    prompts_key: str = "prompts",
    file_key: str = "prompts_file",
    default_file: Optional[str] = None,
) -> Dict[str, str]:
    """Select a metric-to-prompt mapping from config, file, and defaults.

    Resolution order:
      1. ``config[file_key]`` — load metric prompts from the referenced file.
      2. ``config[prompts_key]`` — inline metric-to-prompt mapping.
      3. ``defaults`` — caller-provided fallback mapping.
      4. ``default_file`` — caller-provided fallback prompts file path.

    File and inline prompts are merged, with inline entries taking precedence over
    file entries. Defaults are used only for metrics not overridden above.

    Raises:
        FileNotFoundError: If a configured or fallback prompts file does not exist.
        ValueError: If the loaded prompts file is not a mapping.
    """
    custom: Dict[str, str] = dict(config.get(prompts_key) or {})
    file_path = config.get(file_key)
    if file_path is not None:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Prompts file not found: {file_path}")
        file_prompts = _strip_prompt_values(load_prompts_from_file(file_path))
        custom = {**file_prompts, **custom}

    base: Dict[str, str] = dict(defaults or {})
    if default_file is not None:
        if not Path(default_file).is_file():
            raise FileNotFoundError(f"Default prompts file not found: {default_file}")
        file_defaults = _strip_prompt_values(load_prompts_from_file(default_file))
        base = {**file_defaults, **base}

    return {**base, **custom}
