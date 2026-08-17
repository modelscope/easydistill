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

"""Preference distillation pipeline for DPO data."""

import logging
from typing import Any, Dict, List

from easydistill.operators.preference import (
    CandidateGenerationOperator,
    CoTScorer,
    LLMJudgeScorer,
    PreferenceDatasetBuilder,
    PreferencePairBuilder,
)

from .base import BaseDistillationPipeline

logger = logging.getLogger(__name__)

_SCORER_REGISTRY = {
    "llm_judge": LLMJudgeScorer,
    "cot": CoTScorer,
}


def _run_generate_candidates_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    global_generation_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gen_cfg = {**global_generation_config, **stage_config}
    generator = CandidateGenerationOperator(backend=backend, config=gen_cfg)
    return generator.run(data)


def _run_score_candidates_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    default_scorer: str,
) -> List[Dict[str, Any]]:
    scorer_name = stage_config.get("scorer") or default_scorer
    if scorer_name not in _SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    scorer_cls = _SCORER_REGISTRY[scorer_name]
    if scorer_name == "llm_judge":
        scorer = scorer_cls(backend=backend, config=stage_config)
    else:
        scorer = scorer_cls(config=stage_config)

    instruction_key = stage_config.get("instruction_key") or "instruction"
    answer_key = stage_config.get("answer_key") or "answer"

    output_rows = []
    for row in data:
        instruction = row.get(instruction_key, "")
        reference = row.get(answer_key)
        candidates = row.get("candidates", [])
        if not candidates:
            continue
        scores = scorer.score(instruction, candidates, reference)
        new_row = dict(row)
        new_row["candidate_scores"] = scores
        if scorer_name == "cot":
            # Mark correctness explicitly so the pair builder can enforce it.
            new_row["candidate_correctness"] = [score > 0.0 for score in scores]
        output_rows.append(new_row)

    logger.info("Scored candidates for %d prompts.", len(output_rows))
    return output_rows


def _run_build_preference_pairs_stage(
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    builder = PreferencePairBuilder(config=stage_config)
    return builder.run(data)


def _run_build_preference_dataset_stage(
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    global_generation_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cfg = {**global_generation_config, **stage_config}
    builder = PreferenceDatasetBuilder(config=cfg)
    return builder.run(data)


class PreferenceDistillationPipeline(BaseDistillationPipeline):
    """End-to-end pipeline for DPO preference data.

    Fields in the top-level `preference` config are used as defaults for every
    preference stage. Each stage can override them in its own `config` block.

    Recommended stage flow:
      1. generate_candidates
      2. score_candidates
      3. build_preference_pairs
      4. build_preference_dataset
    """

    _last_stage = "build_preference_dataset"

    def __init__(
        self,
        backend: Any,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Dict[str, Any],
        preference_config: Dict[str, Any],
    ):
        super().__init__(
            backend=backend,
            pipeline_config=pipeline_config,
            dataset_config=dataset_config,
            generation_config=generation_config,
        )
        self.preference_config = preference_config or {}
        self.default_scorer = self.preference_config.get("scorer") or "llm_judge"

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        # Top-level `preference` fields act as defaults; stage `config` overrides.
        merged_config = {**self.preference_config, **stage_config}
        if stage_name == "generate_candidates":
            return _run_generate_candidates_stage(
                self.backend, merged_config, data, self.generation_config
            )
        elif stage_name == "score_candidates":
            return _run_score_candidates_stage(
                self.backend, merged_config, data, self.default_scorer
            )
        elif stage_name == "build_preference_pairs":
            return _run_build_preference_pairs_stage(merged_config, data)
        elif stage_name == "build_preference_dataset":
            return _run_build_preference_dataset_stage(merged_config, data, self.generation_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
