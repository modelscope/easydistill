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

"""Agent distillation pipeline for synthetic tool-use tasks and trajectories."""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from easydistill.data.models import Message, SFTSample
from easydistill.operators.agent import (
    AgentFuzzyTaskOperator,
    AgentRubricOperator,
    AgentTaskSynthesisOperator,
    AgentToolCheckOperator,
    AgentTrajectoryOperator,
)
from easydistill.operators.preference import PreferenceDatasetBuilder, PreferencePairBuilder
from easydistill.operators.prompt_base import PromptGenerationOperator

from .base import BaseDistillationPipeline
from .common import run_quality_filter_stage

logger = logging.getLogger(__name__)

_AGENT_OPERATORS: Dict[str, Type[PromptGenerationOperator[Any, Any]]] = {
    "agent_task_synthesis": AgentTaskSynthesisOperator,
    "agent_fuzzy_task": AgentFuzzyTaskOperator,
    "agent_tool_check": AgentToolCheckOperator,
}


def _run_agent_prompt_stage(
    backend: Any,
    stage_name: str,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run a synthesis/fuzzy-task/tool-check stage and merge outputs into rows.

    Outputs are merged by ``row_idx`` so that parse failures do not shift
    subsequent results onto the wrong input row.
    """
    op_cls = _AGENT_OPERATORS[stage_name]
    operator = op_cls(backend=backend, config=stage_config)

    if stage_name == "agent_task_synthesis":
        inputs = [row.get("background") or row.get("persona", "") for row in data]
    else:
        inputs = data

    outputs = operator.run(inputs)
    by_idx: Dict[int, Dict[str, Any]] = {}
    for output in outputs:
        idx = output.get("row_idx")
        if idx is not None:
            try:
                by_idx[int(idx)] = output
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid row_idx %r from %s; skipping output.", idx, stage_name
                )

    merged = []
    for idx, row in enumerate(data):
        output = by_idx.get(idx)
        if output is None:
            logger.warning("Dropping row %d after failed %s parse.", idx, stage_name)
            continue
        new_row = dict(row)
        new_row.update(output)
        merged.append(new_row)
    return merged


def _run_trajectory_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate agent trajectories; output one row per trajectory."""
    operator = AgentTrajectoryOperator(backend=backend, config=stage_config)
    return operator.run(data)


def _run_rubric_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare trajectories per task and select the best one."""
    operator = AgentRubricOperator(backend=backend, config=stage_config)
    return operator.run(data)


def _extract_best_trajectory(
    row: Dict[str, Any],
    use_rubrics: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return the trajectory row marked as best by the rubric judge.

    When ``use_rubrics`` is False or no rubric selection is available, fall
    back to the first trajectory so that ``build_sft`` can still produce
    output from a pipeline that omits the rubric stage.
    """
    trajectories: List[Dict[str, Any]] = row.get("trajectories", [])
    best_id = row.get("best_solution_id")

    if best_id:
        for traj in trajectories:
            if traj.get("solution_id") == best_id:
                return traj
        # The rubric picked an id that is not present in the trajectory list.
        logger.warning(
            "Best solution id %r not found for task %s; falling back to first trajectory.",
            best_id,
            row.get("id"),
        )
        if trajectories:
            return trajectories[0]
        return None

    if use_rubrics:
        return None

    if trajectories:
        logger.warning(
            "No rubric selection for task %s; using first trajectory (use_rubrics=false).",
            row.get("id"),
        )
        return trajectories[0]
    return None


def _trajectory_to_sft_sample(
    row: Dict[str, Any],
    trajectory: Dict[str, Any],
    sft_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert an agent trajectory into an SFT message sample."""
    messages_raw = trajectory.get("trajectory", [])
    if not messages_raw:
        return None

    messages = []
    for msg in messages_raw:
        role = msg.get("role")
        content = msg.get("content")
        if role and content is not None:
            messages.append(Message(role=role, content=content))

    metadata = {
        "task_id": row.get("id"),
        "solution_id": trajectory.get("solution_id"),
        "task_finished": trajectory.get("task_finished"),
        "source": "teacher_model",
        "model": "agent_distill",
    }
    for key in ("task", "fuzzy_task", "restriction", "workflow"):
        if row.get(key):
            metadata[key] = row[key]

    sample = SFTSample(messages=messages, metadata=metadata)
    sample_dict = sample.model_dump()

    # Apply length filters if provided via sft_config.
    if sft_config:
        min_length = sft_config.get("min_length") or 0
        max_length = sft_config.get("max_length")
        total_len = sum(len(str(m.content or "")) for m in messages)
        if total_len < int(min_length):
            return None
        if max_length is not None and total_len > int(max_length):
            return None

    return sample_dict


def run_build_agent_sft_stage(
    data: List[Dict[str, Any]],
    sft_config: Optional[Dict[str, Any]] = None,
    use_rubrics: bool = True,
) -> List[Dict[str, Any]]:
    """Build SFT samples from the best trajectory of each task."""
    sft_config = sft_config or {}
    samples = []
    for row in data:
        best = _extract_best_trajectory(row, use_rubrics=use_rubrics)
        if best is None:
            continue
        sample = _trajectory_to_sft_sample(row, best, sft_config)
        if sample is not None:
            samples.append(sample)
    logger.info("Built %d agent SFT samples from %d tasks.", len(samples), len(data))
    return samples


def run_build_agent_preference_stage(
    data: List[Dict[str, Any]],
    pref_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build DPO preference pairs from rubric-scored trajectories.

    For each task every trajectory becomes a candidate.  The best trajectory
    receives score ``1.0`` and all others receive ``0.0``.  The
    ``PreferencePairBuilder`` selects the highest-scoring candidate as
    ``chosen`` and the lowest-scoring candidate as ``rejected``.  The prompt
    is the fuzzy task (optionally prefixed with a system prompt).
    """
    pref_config = pref_config or {}
    prompt_rows = []
    for row in data:
        trajectories = row.get("trajectories", [])
        best_id = row.get("best_solution_id")
        if not trajectories or not best_id:
            continue

        candidates = []
        candidate_scores = []
        for traj in trajectories:
            candidates.append(json.dumps(traj.get("trajectory", []), ensure_ascii=False))
            score = 1.0 if traj.get("solution_id") == best_id else 0.0
            candidate_scores.append(score)

        prompt_rows.append(
            {
                "id": row.get("id"),
                "instruction": row.get("fuzzy_task", ""),
                "system": pref_config.get("system_prompt"),
                "answer": row.get("task", ""),
                "candidates": candidates,
                "candidate_scores": candidate_scores,
                "task": row.get("task", ""),
                "fuzzy_task": row.get("fuzzy_task", ""),
                "restriction": row.get("restriction", ""),
            }
        )

    pair_builder = PreferencePairBuilder(config=pref_config)
    dataset_builder = PreferenceDatasetBuilder(config=pref_config)

    pairs = pair_builder.run(prompt_rows)
    return dataset_builder.run(pairs)


class AgentDistillationPipeline(BaseDistillationPipeline):
    """End-to-end pipeline for agent trajectory distillation.

    Recommended flow:
      1. agent_task_synthesis (from persona/background)
      2. agent_fuzzy_task
      3. agent_tool_check
      4. agent_trajectory (repeat_times rollouts)
      5. agent_rubrics (compare and select best)
      6. build_sft OR build_preference_dataset

    The last stage must be ``build_sft`` or ``build_preference_dataset``.
    """

    _last_stage = {"build_sft", "build_preference_dataset"}
    _default_eval_metrics: Optional[List[str]] = None

    def __init__(
        self,
        backend: Any,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            backend=backend,
            pipeline_config=pipeline_config,
            dataset_config=dataset_config,
            generation_config=generation_config,
            sft_config=sft_config,
            eval_config=eval_config,
        )
        self.agent_config = agent_config or {}

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name in _AGENT_OPERATORS:
            data = _run_agent_prompt_stage(self.backend, stage_name, stage_config, data)
        elif stage_name == "agent_trajectory":
            merged_cfg = {**self.agent_config, **stage_config}
            data = _run_trajectory_stage(self.backend, merged_cfg, data)
        elif stage_name == "agent_rubrics":
            data = _run_rubric_stage(self.backend, stage_config, data)
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            use_rubrics = bool(self.agent_config.get("use_rubrics", True))
            data = run_build_agent_sft_stage(
                data, self.sft_config, use_rubrics=use_rubrics
            )
        elif stage_name == "build_preference_dataset":
            pref_cfg = {**self.generation_config, **stage_config}
            data = run_build_agent_preference_stage(data, pref_cfg)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
