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

"""Command-line interface for EasyDistill 2."""

from easydistill.utils import expand_env_vars

from .backend_factory import build_backend
from .data_loaders import (
    load_eval_samples,
    load_problem_answer_pairs,
    load_problem_column,
    load_requests,
    load_string_column,
)
from .main import main
from .runners import (
    run_advanced_cot_distill,
    run_advanced_instruct_distill,
    run_augmented_instruct_distill,
    run_balanced_instruct_distill,
    run_cot_eval,
    run_cot_generation,
    run_cot_long2short,
    run_cot_short2long,
    run_distill,
    run_instruct_eval,
    run_synthesis,
)

# Backward-compatible aliases used by older tests and scripts.
_build_backend = build_backend
_load_requests = load_requests
_load_string_column = load_string_column
_load_problem_column = load_problem_column
_load_problem_answer_pairs = load_problem_answer_pairs
_load_eval_samples = load_eval_samples
_expand_env_vars = expand_env_vars

__all__ = [
    "main",
    "build_backend",
    "load_requests",
    "load_string_column",
    "load_problem_column",
    "load_problem_answer_pairs",
    "load_eval_samples",
    "run_distill",
    "run_synthesis",
    "run_cot_generation",
    "run_cot_long2short",
    "run_cot_short2long",
    "run_cot_eval",
    "run_instruct_eval",
    "run_advanced_instruct_distill",
    "run_advanced_cot_distill",
    "run_augmented_instruct_distill",
    "run_balanced_instruct_distill",
]
