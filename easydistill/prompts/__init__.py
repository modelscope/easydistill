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

"""Prompt-loading helpers for synthesis and evaluation operators.

All built-in prompt templates live as plain text / YAML files under
``configs/prompts/`` and are loaded directly by operators. This package only
provides shared resolution utilities so that every operator follows the same
priority: config file path > inline config value > default config file path.
"""

from easydistill.prompts._loader import (
    load_prompt_template_from_file,
    load_prompts_from_file,
    resolve_prompt,
    resolve_prompts,
)

__all__ = [
    "load_prompt_template_from_file",
    "load_prompts_from_file",
    "resolve_prompt",
    "resolve_prompts",
]
