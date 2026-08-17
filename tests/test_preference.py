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

"""Unit tests for DPO preference data operators and pipeline."""

from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult
from easydistill.operators.preference import (
    CandidateGenerationOperator,
    CoTScorer,
    LLMJudgeScorer,
    PreferenceDatasetBuilder,
    PreferencePairBuilder,
)
from easydistill.pipeline import PreferenceDistillationPipeline


class CounterFakeBackend(ModelBackend):
    """Returns incrementing responses so candidates are distinguishable."""

    def __init__(self, prefix: str = "response"):
        self.prefix = prefix
        self.counter = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        request = build_generation_request(messages)
        response = f"{self.prefix} {self.counter}"
        self.counter += 1
        return GenerationResult(
            request=request,
            response=response,
            model=model_id or "fake",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    def health_check(self) -> bool:
        return True


class JudgeFakeBackend(ModelBackend):
    """Returns a fixed <score> tag so LLMJudgeScorer can parse it."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        request = build_generation_request(messages)
        return GenerationResult(
            request=request,
            response="<score>8</score>",
            model=model_id or "fake",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    def health_check(self) -> bool:
        return True


def test_candidate_generation_operator():
    backend = CounterFakeBackend()
    operator = CandidateGenerationOperator(backend=backend, config={"n": 3})
    rows = [
        {"id": "1", "instruction": "What is 1+1?"},
        {"id": "2", "instruction": "Explain gravity."},
    ]
    out = operator.run(rows)
    assert len(out) == 2
    assert len(out[0]["candidates"]) == 3
    assert out[0]["candidates"] == ["response 0", "response 1", "response 2"]
    assert out[1]["candidates"] == ["response 3", "response 4", "response 5"]


def test_cot_scorer_with_reference():
    scorer = CoTScorer(config={"alpha": 0.0, "normalize_answer": True})
    candidates = [
        "The answer is 55.",
        "The answer is 42.",
    ]
    scores = scorer.score("sum 1..10", candidates, reference="55")
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_cot_scorer_length_penalty():
    scorer = CoTScorer(config={"alpha": 0.01, "normalize_answer": True})
    candidates = [
        "The answer is 55.",
        "The answer is 55. " + "x " * 100,
    ]
    scores = scorer.score("sum 1..10", candidates, reference="55")
    assert scores[0] > scores[1]


def test_llm_judge_scorer():
    backend = JudgeFakeBackend()
    scorer = LLMJudgeScorer(backend=backend, config={"metrics": ["helpfulness"]})
    scores = scorer.score("hello", ["a", "b", "c"])
    assert len(scores) == 3
    assert all(s == 8.0 for s in scores)


def test_pair_builder_basic():
    builder = PreferencePairBuilder(config={"min_margin": 0.0})
    rows = [
        {
            "id": "1",
            "instruction": "q1",
            "candidates": ["bad", "good"],
            "candidate_scores": [1.0, 5.0],
        }
    ]
    pairs = builder.run(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"
    assert pairs[0]["rejected"] == "bad"
    assert pairs[0]["chosen_score"] == 5.0
    assert pairs[0]["rejected_score"] == 1.0


def test_pair_builder_respects_margin():
    builder = PreferencePairBuilder(config={"min_margin": 2.0})
    rows = [
        {
            "id": "1",
            "instruction": "q1",
            "candidates": ["a", "b"],
            "candidate_scores": [5.0, 4.0],
        }
    ]
    pairs = builder.run(rows)
    assert len(pairs) == 0


def test_pair_builder_handles_short_correctness_mask():
    builder = PreferencePairBuilder(config={"min_margin": 0.0, "require_chosen_correct": True})
    rows = [
        {
            "id": "1",
            "instruction": "q1",
            "candidates": ["bad", "good", "ok"],
            "candidate_scores": [1.0, 5.0, 3.0],
            # Mask is shorter than candidates; missing entries should be treated False.
            "candidate_correctness": [False, True],
        }
    ]
    pairs = builder.run(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"


def test_dataset_builder_alpaca_format():
    builder = PreferenceDatasetBuilder(config={"format": "llama_factory_alpaca"})
    rows = [
        {"instruction": "q1", "chosen": "good", "rejected": "bad"},
    ]
    out = builder.run(rows)
    assert out == [{"instruction": "q1", "input": "", "chosen": "good", "rejected": "bad"}]


def test_dataset_builder_sharegpt_format():
    builder = PreferenceDatasetBuilder(config={"format": "llama_factory_sharegpt"})
    rows = [
        {"instruction": "q1", "chosen": "good", "rejected": "bad"},
    ]
    out = builder.run(rows)
    assert out[0]["conversations"] == [{"from": "human", "value": "q1"}]
    assert out[0]["chosen"] == {"from": "gpt", "value": "good"}
    assert out[0]["rejected"] == {"from": "gpt", "value": "bad"}


def test_dataset_builder_openai_format():
    builder = PreferenceDatasetBuilder(config={"format": "openai_messages"})
    rows = [
        {"instruction": "q1", "chosen": "good", "rejected": "bad"},
    ]
    out = builder.run(rows)
    assert out[0]["prompt"] == [{"role": "user", "content": "q1"}]
    assert out[0]["chosen"] == [{"role": "assistant", "content": "good"}]
    assert out[0]["rejected"] == [{"role": "assistant", "content": "bad"}]


def test_dpo_cot_pipeline_end_to_end():
    backend = CounterFakeBackend(prefix="The answer is")
    pipeline = PreferenceDistillationPipeline(
        backend=backend,
        pipeline_config=[
            {"stage": "generate_candidates", "config": {"n": 2}},
            {"stage": "score_candidates", "config": {"scorer": "cot", "answer_key": "answer"}},
            {"stage": "build_preference_pairs", "config": {"require_chosen_correct": True}},
            {"stage": "build_preference_dataset", "config": {"format": "openai_messages"}},
        ],
        dataset_config={"input_path": "unused", "output_path": "unused"},
        generation_config={},
        preference_config={"scorer": "cot"},
    )
    data = [
        {"id": "1", "instruction": "sum 1..10", "answer": "The answer is 1"},
    ]
    result = pipeline.run_with_data(data)
    assert len(result) == 1
    assert result[0]["chosen"][0]["content"] == "The answer is 1"
    assert "prompt" in result[0]


def test_dpo_pipeline_end_to_end():
    backend = JudgeFakeBackend()
    pipeline = PreferenceDistillationPipeline(
        backend=backend,
        pipeline_config=[
            {"stage": "generate_candidates", "config": {"n": 2}},
            {
                "stage": "score_candidates",
                "config": {"scorer": "llm_judge", "metrics": ["helpfulness"]},
            },
            {"stage": "build_preference_pairs", "config": {"min_margin": 0.0}},
            {"stage": "build_preference_dataset", "config": {"format": "llama_factory_alpaca"}},
        ],
        dataset_config={"input_path": "unused", "output_path": "unused"},
        generation_config={},
        preference_config={"scorer": "llm_judge"},
    )
    data = [{"id": "1", "instruction": "hello"}]
    result = pipeline.run_with_data(data)
    assert len(result) == 1
    assert result[0]["instruction"] == "hello"
    assert "chosen" in result[0]
    assert "rejected" in result[0]
