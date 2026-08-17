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

"""Operators for search-agent data synthesis and trajectory distillation."""

from .judge import answer_equivalent, judge_trajectory, run_quality_gate
from .solver import SearchTrajectoryOperator, evaluate_task, solve_search_task
from .task_evolver import SearchTaskEvolverOperator
from .tools import SEARCH_TOOLS, SearchToolset

__all__ = [
    "SEARCH_TOOLS",
    "SearchTaskEvolverOperator",
    "SearchToolset",
    "SearchTrajectoryOperator",
    "answer_equivalent",
    "evaluate_task",
    "judge_trajectory",
    "run_quality_gate",
    "solve_search_task",
]
