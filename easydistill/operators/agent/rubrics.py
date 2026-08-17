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

"""Rubric generation + best-trajectory selection operator."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely

from .base import PromptAgentOperator
from .utils import extract_best_solution_filename, extract_tag, format_trajectory_for_comparison

logger = logging.getLogger(__name__)


class AgentRubricOperator(PromptAgentOperator[Dict[str, Any], Dict[str, Any]]):
    """Compare trajectories for a task and produce rubrics + best solution.

    Input rows must be trajectory rows (one per trajectory) with ``id``,
    ``solution_id``, ``trajectory``, ``fuzzy_task``, ``task_background``,
    ``restriction``, and optionally ``workflow``.

    The operator groups rows by ``id`` and invokes a judge LLM to compare all
    trajectories for the same task.

    Configurable fields:
      - prompt_template / prompt_template_file.
      - temperature, max_tokens, max_workers, show_progress.
      - solution_top_k: max trajectories to compare (default 3).

    Output is one row per task containing:
      - id, task, fuzzy_task, task_background, restriction, workflow, checked_tools
      - trajectories: list of input trajectory rows for this task
      - rubrics, alignment_check, final, best_solution_id
    """

    name = "agent_rubrics"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            default_file="configs/prompts/agent_rubrics_prompt.txt",
        )
        self.solution_top_k = int(self.config.get("solution_top_k") or 3)

    def _group_by_task(self, rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            task_id = str(row.get("id", "unknown"))
            groups[task_id].append(row)
        return groups

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        groups = self._group_by_task(inputs)
        for task_id, trajectories in groups.items():
            trajectories = trajectories[: self.solution_top_k]
            if len(trajectories) < 1:
                continue

            summaries: List[str] = []
            for traj_row in trajectories:
                summary = format_trajectory_for_comparison(
                    traj_row.get("solution_id", "unknown"),
                    traj_row.get("trajectory", []),
                )
                summaries.append(summary)

            prompt = format_prompt_safely(
                self.prompt_template or "",
                task_description=trajectories[0].get("fuzzy_task", ""),
                task_background=trajectories[0].get("task_background", ""),
                high_level_workflow=trajectories[0].get("workflow", ""),
                restrict_policy=trajectories[0].get("restriction", ""),
                trajectories_summary="\n".join(["=" * 80] + summaries + ["=" * 80]),
            )
            requests.append(
                GenerationRequest(
                    id=f"agent_rubrics_{task_id}",
                    instruction=prompt,
                    metadata={
                        "task_id": task_id,
                        "trajectories": trajectories,
                        "task": "agent_rubrics",
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        raw_response = result.response or ""
        alignment_check = extract_tag(raw_response, "alignment_check") or ""

        if not alignment_check or "discard" in alignment_check.lower():
            logger.warning("Rubric judge discarded task %s", result.request.metadata.get("task_id"))
            return None

        rubrics = extract_tag(raw_response, "rubrics") or ""
        final = extract_tag(raw_response, "final") or ""
        best_solution_text = extract_tag(raw_response, "best_solution") or ""
        best_solution_id = extract_best_solution_filename(best_solution_text)

        if not rubrics or not final or not best_solution_id:
            logger.warning(
                "Incomplete rubric result for task %s",
                result.request.metadata.get("task_id"),
            )
            return None

        trajectories = result.request.metadata.get("trajectories", [])
        first_traj = trajectories[0] if trajectories else {}
        return {
            "id": result.request.metadata.get("task_id"),
            "task": first_traj.get("task", ""),
            "fuzzy_task": first_traj.get("fuzzy_task", ""),
            "task_background": first_traj.get("task_background", ""),
            "restriction": first_traj.get("restriction", ""),
            "workflow": first_traj.get("workflow", ""),
            "checked_tools": first_traj.get("checked_tools", []),
            "trajectories": trajectories,
            "rubrics": rubrics,
            "alignment_check": alignment_check,
            "final": final,
            "best_solution_id": best_solution_id,
            "raw_rubrics": raw_response,
        }
