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

"""Search-agent distillation pipeline: seed QA -> evolved multi-hop tasks ->
solver trajectories -> judge filter -> SFT dataset."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import Message, SFTSample
from easydistill.operators.search_agent import (
    SearchTaskEvolverOperator,
    SearchTrajectoryOperator,
)

from .base import BaseDistillationPipeline
from .common import run_quality_filter_stage

logger = logging.getLogger(__name__)

_SEED_QUESTION_KEYS = ("question", "q", "instruction", "problem", "question_text")
_SEED_ANSWER_KEYS = (
    "answer",
    "a_star",
    "short_answer",
    "golden_answer",
    "true_answer",
    "short_answer_text",
)
_SEED_ID_KEYS = ("id", "example_id")


def _normalize_seed_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize seed rows into ``{id, question, answer}`` with aliases.

    Accepts the original SearchSynthAgent field names (``q``/``a_star``/
    ``short_answer``) as well as generic instruction datasets. List-valued
    answers (e.g. ``true_answer`` alias lists from 2wiki/hqa) use the first
    entry as the canonical answer and keep the rest in ``answer_aliases``.
    """
    rows = []
    for idx, row in enumerate(data):
        question = next((str(row[k]).strip() for k in _SEED_QUESTION_KEYS if row.get(k)), "")
        answer = ""
        aliases: List[str] = []
        for key in _SEED_ANSWER_KEYS:
            value = row.get(key)
            if not value:
                continue
            if isinstance(value, list):
                values = [str(v).strip() for v in value if str(v).strip()]
                if values:
                    answer = values[0]
                    aliases = values[1:]
            else:
                answer = str(value).strip()
            break
        if not question or not answer:
            logger.warning("Seed row %d missing question/answer; dropped.", idx)
            continue
        seed_id = next(
            (str(row[k]) for k in _SEED_ID_KEYS if row.get(k) not in (None, "")),
            f"seed_{idx}",
        )
        normalized: Dict[str, Any] = {
            "id": seed_id,
            "question": question,
            "answer": answer,
        }
        if aliases:
            normalized["answer_aliases"] = aliases
        rows.append(normalized)
    return rows


def run_judge_filter_stage(
    data: List[Dict[str, Any]],
    stage_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Filter trajectories per task and select which ones feed SFT.

    Keeps only correct trajectories by default (``require_correct``) and
    ranks them by ``selection``: ``correct_shortest`` (default) prefers the
    correct trajectory with the fewest turns; ``correct_longest`` prefers the
    most tool-use-rich one; ``all_correct`` keeps every correct trajectory.
    """
    require_correct = bool(stage_config.get("require_correct", True))
    min_turns = int(stage_config.get("min_turns", 0))
    selection = str(stage_config.get("selection", "correct_shortest"))

    output = []
    for row in data:
        trajectories = row.get("trajectories", [])
        kept = [
            t
            for t in trajectories
            if (t.get("is_correct") or not require_correct) and t.get("turns", 0) >= min_turns
        ]
        if not kept:
            continue
        if selection == "all_correct":
            selected = kept
        elif selection == "correct_longest":
            selected = [max(kept, key=lambda t: t.get("turns", 0))]
        else:
            selected = [min(kept, key=lambda t: t.get("turns", 0))]
        new_row = dict(row)
        new_row["selected_trajectories"] = selected
        output.append(new_row)
    logger.info("Judge filter kept %d/%d tasks (selection=%s).", len(output), len(data), selection)
    return output


def run_build_search_sft_stage(
    data: List[Dict[str, Any]],
    sft_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert selected trajectories into standard SFT samples.

    The full solve history (system + user + assistant/tool_response turns)
    becomes ``messages``; task provenance, judge labels and per-run stats go
    into ``metadata`` so training and auditing consume one structure.
    """
    sft_config = sft_config or {}
    min_length = int(sft_config.get("min_length", 0))
    max_length = sft_config.get("max_length")

    samples = []
    for row in data:
        selected = row.get("selected_trajectories") or row.get("trajectories") or []
        for traj in selected:
            history = traj.get("trajectory", [])
            if not history:
                continue
            messages = [
                Message(role=m.get("role", "user"), content=m.get("content", "")) for m in history
            ]
            total_len = sum(len(str(m.content or "")) for m in messages)
            if total_len < min_length:
                continue
            if max_length is not None and total_len > int(max_length):
                continue
            sample = SFTSample(
                messages=messages,
                metadata={
                    "task_id": row.get("id"),
                    "solution_id": traj.get("solution_id"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "predicted_answer": traj.get("predicted_answer"),
                    "is_correct": traj.get("is_correct"),
                    "turns": traj.get("turns"),
                    "hops": row.get("hops"),
                    "seed_id": row.get("seed_id"),
                    "seed_question": row.get("seed_question"),
                    "difficulty_report": row.get("difficulty_report"),
                    "final_eval": row.get("final_eval"),
                },
            )
            samples.append(sample.model_dump())
    logger.info("Built %d search-agent SFT samples from %d tasks.", len(samples), len(data))
    return samples


class SearchAgentDistillationPipeline(BaseDistillationPipeline):
    """End-to-end pipeline for search-agent data synthesis and distillation.

    Recommended flow:
      1. search_task_evolve (Strategist/Expand/Refine/QualityGate/Verify/Judge loop)
      2. search_trajectory (repeat_times solver rollouts per task)
      3. search_judge_filter (keep correct trajectories, pick best)
      4. build_sft

    The last stage must be ``build_sft``.
    """

    _last_stage = "build_sft"
    _default_eval_metrics: Optional[List[str]] = None

    def __init__(
        self,
        backend: Any,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        search_agent_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            backend=backend,
            pipeline_config=pipeline_config,
            dataset_config=dataset_config,
            generation_config=generation_config,
            sft_config=sft_config,
            eval_config=eval_config,
        )
        # Shared roles/tools settings merged into every stage config.
        self.search_agent_config = search_agent_config or {}

    def _stage_config(self, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.search_agent_config)
        merged.update(stage_config)
        return merged

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == "search_task_evolve":
            seeds = _normalize_seed_rows(data)
            evolver = SearchTaskEvolverOperator(
                backend=self.backend, config=self._stage_config(stage_config)
            )
            data = evolver.run(seeds)
        elif stage_name == "search_trajectory":
            trajectory_op = SearchTrajectoryOperator(
                backend=self.backend, config=self._stage_config(stage_config)
            )
            data = trajectory_op.run(data)
        elif stage_name == "search_judge_filter":
            data = run_judge_filter_stage(data, stage_config)
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            data = run_build_search_sft_stage(data, self.sft_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
