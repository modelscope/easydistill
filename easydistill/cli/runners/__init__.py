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

"""CLI runner functions for each job family."""

from .cot import run_cot_generation, run_cot_long2short, run_cot_short2long
from .distill import run_distill
from .eval import run_cot_eval, run_instruct_eval, run_pe_rewrite_eval
from .mm import (
    run_mm_cot_eval,
    run_mm_cot_generation,
    run_mm_cot_long2short,
    run_mm_cot_short2long,
    run_mm_generation,
    run_mm_instruct_eval,
)
from .pe_rewrite import run_pe_rewrite_build_sft, run_pe_rewrite_filter
from .pipeline import (
    run_advanced_cot_distill,
    run_advanced_instruct_distill,
    run_advanced_mm_cot_distill,
    run_advanced_mm_distill,
    run_agent_distill,
    run_augmented_instruct_distill,
    run_balanced_instruct_distill,
    run_pe_rewrite_distill,
    run_search_agent_distill,
)
from .preference import run_dpo_data_build
from .synthesis import run_synthesis
from .t2i import (
    run_advanced_t2i_distill,
    run_prompt_optimize,
    run_t2i_distill,
    run_t2i_eval,
    run_t2i_generation,
)
from .t2i_ti2i_eval import (
    run_t2i_multi_model_eval,
    run_t2i_single_model_eval,
    run_ti2i_multi_model_eval,
    run_ti2i_single_model_eval,
)
from .t2v import (
    run_advanced_t2v_distill,
    run_t2v_distill,
    run_t2v_eval,
    run_t2v_generation,
    run_t2v_prompt_optimize,
)

__all__ = [
    "run_distill",
    "run_synthesis",
    "run_cot_generation",
    "run_cot_long2short",
    "run_cot_short2long",
    "run_cot_eval",
    "run_instruct_eval",
    "run_pe_rewrite_eval",
    "run_pe_rewrite_filter",
    "run_pe_rewrite_build_sft",
    "run_pe_rewrite_distill",
    "run_advanced_instruct_distill",
    "run_advanced_cot_distill",
    "run_augmented_instruct_distill",
    "run_balanced_instruct_distill",
    "run_dpo_data_build",
    "run_mm_generation",
    "run_mm_cot_generation",
    "run_mm_cot_long2short",
    "run_mm_cot_short2long",
    "run_mm_instruct_eval",
    "run_mm_cot_eval",
    "run_advanced_mm_distill",
    "run_advanced_mm_cot_distill",
    "run_t2i_distill",
    "run_prompt_optimize",
    "run_t2i_generation",
    "run_t2i_single_model_eval",
    "run_t2i_multi_model_eval",
    "run_ti2i_single_model_eval",
    "run_ti2i_multi_model_eval",
    "run_t2i_eval",
    "run_advanced_t2i_distill",
    "run_t2v_distill",
    "run_t2v_prompt_optimize",
    "run_t2v_generation",
    "run_t2v_eval",
    "run_advanced_t2v_distill",
    "run_agent_distill",
    "run_search_agent_distill",
]
