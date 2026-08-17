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

"""Best-practice chain-of-thought distillation pipeline."""

from typing import Any, Dict, List, Tuple, Type

from easydistill.eval import CoTEvaluator
from easydistill.operators.cot import (
    CoTGenerationOperator,
    CoTRVCDMixer,
    CoTRVCDScorer,
)
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.rewrite import (
    CoTLong2ShortOperator,
    CoTShort2LongOperator,
)
from easydistill.utils.image import _extract_text_from_content

from .base import BaseDistillationPipeline
from .common import (
    run_build_preference_dataset_stage,
    run_build_sft_stage,
    run_eval_stage,
    run_quality_filter_stage,
)

_COT_OPERATORS: Dict[str, Type[PromptGenerationOperator[Any, Any]]] = {
    "cot_distill": CoTGenerationOperator,
    "cot_long2short": CoTLong2ShortOperator,
    "cot_short2long": CoTShort2LongOperator,
}

_DEFAULT_EVAL_METRICS = [
    "reasoning_verbosity",
    "cognitive_difficulty",
    "logical_correctness",
]


def _extract_problems(data: List[Dict[str, Any]]) -> List[str]:
    """Extract problem strings from rows, preferring a 'problem' key."""
    problems = []
    for row in data:
        problem = row.get("problem") or row.get("instruction")
        if problem:
            problems.append(_extract_text_from_content(problem))
    return problems


def _extract_problem_answer_pairs(data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Extract (problem, answer) pairs from rows."""
    pairs = []
    for row in data:
        problem = row.get("problem") or row.get("instruction")
        answer = row.get("response") or row.get("answer") or row.get("output")
        if problem and answer:
            pairs.append((_extract_text_from_content(problem), _extract_text_from_content(answer)))
    return pairs


def _run_cot_stage(
    backend: Any,
    stage_name: str,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run a CoT rewrite/generation stage."""
    op_cls = _COT_OPERATORS[stage_name]
    operator = op_cls(backend=backend, config=stage_config)

    if stage_name == "cot_distill":
        inputs = _extract_problems(data)
        return operator.run(inputs)

    pairs = _extract_problem_answer_pairs(data)
    return operator.run(pairs)


def _run_rvcd_score_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run RV/CD scoring on CoT rows."""
    scorer = CoTRVCDScorer(backend=backend, config=stage_config)
    return scorer.run(data)


def _run_rvcd_mix_stage(
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run RV/CD mixing on scored CoT rows."""
    mixer = CoTRVCDMixer(config=stage_config)
    return mixer.run(data)


class CoTDistillationPipeline(BaseDistillationPipeline):
    """End-to-end advanced pipeline for CoT distillation.

    The recommended flow uses RV/CD scores for curriculum mixing:
      1. cot_distill (teacher CoT generation)
      2. (optional) cot_long2short or cot_short2long
      3. cot_rvcd_score (score reasoning verbosity, cognitive difficulty,
         and logical correctness)
      4. cot_mix_by_rv_cd (mix rows per CD bin to build a curriculum)
      5. build_sft (final SFT dataset)

    An alternative flow uses standard LLM-as-judge evaluation:
      1. cot_distill
      2. cot_eval
      3. quality_filter
      4. build_sft

    The last stage must be `build_sft`.
    """

    _last_stage = "build_sft"
    _default_eval_metrics = _DEFAULT_EVAL_METRICS

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name in _COT_OPERATORS:
            data = _run_cot_stage(self.backend, stage_name, stage_config, data)
        elif stage_name == "cot_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            data = run_eval_stage(self.backend, eval_cfg, data, CoTEvaluator)
        elif stage_name == "cot_rvcd_score":
            score_cfg = {**self.eval_config, **stage_config}
            data = _run_rvcd_score_stage(self.backend, score_cfg, data)
        elif stage_name == "cot_mix_by_rv_cd":
            data = _run_rvcd_mix_stage(stage_config, data)
        elif stage_name == "quality_filter":
            data = run_quality_filter_stage(stage_config, data, eval_metrics)
        elif stage_name == "build_sft":
            data = run_build_sft_stage(data, self.generation_config, self.sft_config)
        elif stage_name == "build_preference_dataset":
            pref_cfg = {**self.generation_config, **stage_config}
            data = run_build_preference_dataset_stage(data, pref_cfg)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
