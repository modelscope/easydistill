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
# See the License for the specific language governing permissions or
# limitations under the License.
# ==============================================================================

"""Unit tests for the system-1 distillation operators and pipeline.

Covers:
- Module-level constants and contracts.
- ``parse_probability_output`` / ``parse_freq_output`` parsing edge cases.
- ``System1BuildCasesOperator``: state derivation, gold one-hot, label field.
- ``System1ElicitOperator``: end-to-end with a fake backend (k_verbalized_mean,
  single_verbalized, k_sample_freq).
- ``System1AggregateOperator``: consistency, shrinkage, labeled-rejection.
- ``System1BuildDatasetOperator``: distill gold nesting, labeled passthrough.
- Full 4-stage pipeline integration with a fake backend.
"""

import json
from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult
from easydistill.operators.system1.aggregate import (
    AGGREGATE_VARIANTS,
    DEFAULT_CONSISTENCY_THRESHOLD,
    DEFAULT_MIX_ALPHA,
    DEFAULT_SHRINKAGE,
    FREQ_LAPLACE_ALPHA,
    System1AggregateOperator,
    aggregate_question,
)
from easydistill.operators.system1.build import (
    DEFAULT_TOKEN_BUDGET,
    System1BuildCasesOperator,
)
from easydistill.operators.system1.build_dataset import (
    BUILD_DATASET_VARIANTS,
    System1BuildDatasetOperator,
)
from easydistill.operators.system1.elicit import (
    ELICIT_METHODS,
    System1ElicitOperator,
    parse_freq_output,
    parse_probability_output,
)
from easydistill.operators.system1.protocol import (
    MAX_FLAT_OPTIONS,
    QUESTION_TYPES,
)

# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class _FakeSystem1Backend(ModelBackend):
    """Return a fixed JSON probability distribution for every call."""

    def __init__(self, probabilities: Dict[str, float], model_id: str = "fake-teacher"):
        self._probabilities = probabilities
        self._model_id = model_id
        self.call_count = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        self.call_count += 1
        request = build_generation_request(messages)
        return GenerationResult(
            request=request,
            response=json.dumps(self._probabilities),
            model=model_id or self._model_id,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def health_check(self) -> bool:
        return True


class _FakeFreqBackend(ModelBackend):
    """Return a single option key (for k_sample_freq method)."""

    def __init__(self, pick: str, model_id: str = "fake-freq"):
        self._pick = pick
        self._model_id = model_id
        self.call_count = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        self.call_count += 1
        request = build_generation_request(messages)
        return GenerationResult(
            request=request,
            response=self._pick,
            model=model_id or self._model_id,
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        )

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMS_SCHEMA = {
    "name": "sms_spam",
    "question": "spam",
    "instructions": "Classify the SMS message.",
    "options": {
        "spam": "Unsolicited commercial or fraudulent message",
        "ham": "Legitimate personal, service, or transactional message",
    },
}

RAW_ROWS = [
    {"id": "sms-001", "label": "ham", "text": "Hey, running late."},
    {"id": "sms-002", "label": "spam", "text": "WINNER! Claim your prize now!"},
]


def _build_cases(
    rows: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    op = System1BuildCasesOperator({
        "schema": schema or SMS_SCHEMA,
        "token_budget": {"check": False},
        "show_progress": False,
    })
    return op.run(rows)


def _case_row() -> Dict[str, Any]:
    """Return a single minimal case row for elicit/aggregate/build_dataset tests."""
    return {
        "id": "sms-001",
        "workflow": "sms_spam",
        "state": {"text": "Hey, running late."},
        "questions": {
            "spam": {
                "type": "choice",
                "instructions": "Classify the SMS message.",
                "criteria": {
                    "spam": "Unsolicited commercial or fraudulent message",
                    "ham": "Legitimate personal, service, or transactional message",
                },
            }
        },
        "gold": {"spam": {"spam": 0.0, "ham": 1.0}},
        "source": {"id": "sms-001", "label": "ham", "text": "Hey, running late."},
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_elicit_methods(self):
        assert ELICIT_METHODS == (
            "single_verbalized", "k_sample_freq", "k_verbalized_mean", "two_stage",
        )

    def test_aggregate_variants(self):
        assert AGGREGATE_VARIANTS == ("distill", "mixed")

    def test_build_dataset_variants(self):
        assert BUILD_DATASET_VARIANTS == ("labeled", "distill", "mixed")

    def test_question_types(self):
        assert QUESTION_TYPES == ("choice", "score", "noul")

    def test_max_flat_options(self):
        assert MAX_FLAT_OPTIONS == 20

    def test_aggregate_defaults(self):
        assert FREQ_LAPLACE_ALPHA == 0.5
        assert DEFAULT_CONSISTENCY_THRESHOLD == 0.6
        assert DEFAULT_MIX_ALPHA == 0.7
        assert DEFAULT_SHRINKAGE == 0.02

    def test_token_budget_defaults(self):
        assert DEFAULT_TOKEN_BUDGET["check"] is True
        assert DEFAULT_TOKEN_BUDGET["tokenizer_id"] == "convaiinnovations/laya"
        assert DEFAULT_TOKEN_BUDGET["max_len"] == 1024
        assert DEFAULT_TOKEN_BUDGET["head_max_len"] == 256
        assert DEFAULT_TOKEN_BUDGET["chars_per_token"] == 4.0


# ---------------------------------------------------------------------------
# parse_probability_output
# ---------------------------------------------------------------------------


class TestParseProbabilityOutput:
    keys = ["spam", "ham"]

    def test_plain_json(self):
        result = parse_probability_output('{"spam": 0.3, "ham": 0.7}', self.keys)
        assert result == {"spam": 0.3, "ham": 0.7}

    def test_json_wrapped_in_prose(self):
        content = 'Here is my analysis:\n{"spam": 0.0, "ham": 1.0}\nDone.'
        result = parse_probability_output(content, self.keys)
        assert result == {"spam": 0.0, "ham": 1.0}

    def test_json_in_code_fence(self):
        content = '```json\n{"spam": 0.2, "ham": 0.8}\n```'
        result = parse_probability_output(content, self.keys)
        assert result == {"spam": 0.2, "ham": 0.8}

    def test_extra_keys_dropped(self):
        result = parse_probability_output('{"spam": 0.5, "ham": 0.5, "extra": 0.9}', self.keys)
        assert result == {"spam": 0.5, "ham": 0.5}
        assert "extra" not in result

    def test_missing_key_returns_none(self):
        result = parse_probability_output('{"spam": 1.0}', self.keys)
        assert result is None

    def test_invalid_value_returns_none(self):
        result = parse_probability_output('{"spam": "yes", "ham": 0.5}', self.keys)
        assert result is None

    def test_negative_value_returns_none(self):
        result = parse_probability_output('{"spam": -0.1, "ham": 1.1}', self.keys)
        assert result is None

    def test_boolean_rejected(self):
        result = parse_probability_output('{"spam": true, "ham": false}', self.keys)
        assert result is None

    def test_no_json_returns_none(self):
        result = parse_probability_output("I cannot decide.", self.keys)
        assert result is None

    def test_integer_values_accepted(self):
        result = parse_probability_output('{"spam": 0, "ham": 1}', self.keys)
        assert result == {"spam": 0.0, "ham": 1.0}

    def test_nested_object_found(self):
        content = '{"wrapper": {"spam": 0.4, "ham": 0.6}}'
        result = parse_probability_output(content, self.keys)
        assert result == {"spam": 0.4, "ham": 0.6}


# ---------------------------------------------------------------------------
# parse_freq_output
# ---------------------------------------------------------------------------


class TestParseFreqOutput:
    keys = ["spam", "ham"]

    def test_bare_key(self):
        assert parse_freq_output("ham", self.keys) == "ham"

    def test_fenced_key(self):
        assert parse_freq_output("```ham```", self.keys) == "ham"

    def test_quoted_fenced_key(self):
        assert parse_freq_output('```"ham"```', self.keys) == "ham"

    def test_key_in_sentence(self):
        assert parse_freq_output("The answer is ham.", self.keys) == "ham"

    def test_multiple_keys_returns_none(self):
        assert parse_freq_output("spam and ham both apply", self.keys) is None

    def test_no_key_returns_none(self):
        assert parse_freq_output("I cannot classify this.", self.keys) is None


# ---------------------------------------------------------------------------
# build_cases
# ---------------------------------------------------------------------------


class TestBuildCases:
    def test_state_derived_from_non_id_non_label(self):
        cases = _build_cases(RAW_ROWS)
        assert cases[0]["state"] == {"text": "Hey, running late."}
        assert cases[1]["state"] == {"text": "WINNER! Claim your prize now!"}

    def test_gold_one_hot(self):
        cases = _build_cases(RAW_ROWS)
        assert cases[0]["gold"]["spam"] == {"spam": 0.0, "ham": 1.0}
        assert cases[1]["gold"]["spam"] == {"spam": 1.0, "ham": 0.0}

    def test_source_preserved(self):
        cases = _build_cases(RAW_ROWS)
        assert cases[0]["source"]["id"] == "sms-001"
        assert cases[0]["source"]["label"] == "ham"

    def test_workflow_name(self):
        cases = _build_cases(RAW_ROWS)
        assert cases[0]["workflow"] == "sms_spam"

    def test_question_structure(self):
        cases = _build_cases(RAW_ROWS)
        q = cases[0]["questions"]["spam"]
        assert q["type"] == "choice"
        assert q["instructions"] == "Classify the SMS message."
        assert set(q["criteria"].keys()) == {"spam", "ham"}

    def test_custom_label_field(self):
        schema = dict(SMS_SCHEMA, label_field="category")
        rows = [{"id": "r1", "category": "spam", "text": "win now"}]
        cases = _build_cases(rows, schema=schema)
        assert cases[0]["gold"]["spam"] == {"spam": 1.0, "ham": 0.0}

    def test_prebuilt_questions_passthrough(self):
        rows = [{
            "id": "r1",
            "state": {"text": "hi"},
            "questions": {"q": {"type": "noul", "instructions": "Y/N", "criteria": {}}},
            "gold": {"q": {"false": 0.0, "true": 1.0}},
        }]
        cases = _build_cases(rows)
        assert cases[0]["questions"]["q"]["type"] == "noul"
        assert cases[0]["state"] == {"text": "hi"}


# ---------------------------------------------------------------------------
# elicit
# ---------------------------------------------------------------------------


class TestElicit:
    def test_k_verbalized_mean(self):
        backend = _FakeSystem1Backend({"spam": 0.0, "ham": 1.0})
        op = System1ElicitOperator(backend, {
            "method": "k_verbalized_mean",
            "samples": 3,
            "schema": SMS_SCHEMA,
            "shuffle_options": False,
            "show_progress": False,
            "max_workers": 1,
            "token_budget": {"check": False},
        })
        rows = op.run([_case_row()])
        elic = rows[0]["elicitations"]["spam"]
        assert elic["method"] == "k_verbalized_mean"
        assert elic["model"] == "fake-teacher"
        assert len(elic["samples"]) == 3
        assert all(s["ok"] for s in elic["samples"])
        assert all(s["probabilities"] == {"spam": 0.0, "ham": 1.0} for s in elic["samples"])
        assert backend.call_count == 3

    def test_single_verbalized_forces_one_sample(self):
        backend = _FakeSystem1Backend({"spam": 0.0, "ham": 1.0})
        op = System1ElicitOperator(backend, {
            "method": "single_verbalized",
            "samples": 5,
            "schema": SMS_SCHEMA,
            "show_progress": False,
            "max_workers": 1,
            "token_budget": {"check": False},
        })
        rows = op.run([_case_row()])
        assert len(rows[0]["elicitations"]["spam"]["samples"]) == 1
        assert backend.call_count == 1

    def test_k_sample_freq(self):
        backend = _FakeFreqBackend("ham")
        op = System1ElicitOperator(backend, {
            "method": "k_sample_freq",
            "samples": 4,
            "schema": SMS_SCHEMA,
            "show_progress": False,
            "max_workers": 1,
            "token_budget": {"check": False},
        })
        rows = op.run([_case_row()])
        elic = rows[0]["elicitations"]["spam"]
        assert elic["method"] == "k_sample_freq"
        assert len(elic["samples"]) == 4
        assert all(s["ok"] for s in elic["samples"])
        assert all(s["pick"] == "ham" for s in elic["samples"])

    def test_invalid_method_raises(self):
        backend = _FakeSystem1Backend({"spam": 0.0, "ham": 1.0})
        with pytest.raises(ValueError, match="method must be one of"):
            System1ElicitOperator(backend, {"method": "bogus", "schema": SMS_SCHEMA})

    def test_parse_failure_marks_not_ok(self):
        backend = _FakeSystem1Backend({})
        op = System1ElicitOperator(backend, {
            "method": "k_verbalized_mean",
            "samples": 1,
            "schema": SMS_SCHEMA,
            "show_progress": False,
            "max_workers": 1,
            "token_budget": {"check": False},
        })
        rows = op.run([_case_row()])
        sample = rows[0]["elicitations"]["spam"]["samples"][0]
        assert sample["ok"] is False
        assert "parse_failed" in sample["errors"]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _elicit_row(self, samples: List[Dict[str, float]]) -> Dict[str, Any]:
        row = _case_row()
        row["elicitations"] = {
            "spam": {
                "method": "k_verbalized_mean",
                "model": "fake-teacher",
                "samples": [
                    {
                        "id": f"sms-001|spam|{i}",
                        "question_id": "spam",
                        "sample": i,
                        "ok": True,
                        "probabilities": probs,
                        "model": "fake-teacher",
                        "errors": [],
                    }
                    for i, probs in enumerate(samples)
                ],
            }
        }
        return row

    def test_distill_consistent(self):
        row = self._elicit_row([
            {"spam": 0.0, "ham": 1.0},
            {"spam": 0.0, "ham": 1.0},
            {"spam": 0.0, "ham": 1.0},
        ])
        op = System1AggregateOperator({"variant": "distill", "show_progress": False})
        out = op.run([row])
        teacher = out[0]["teacher"]["spam"]
        assert teacher["ok"] is True
        assert teacher["consistency"] == 1.0
        assert teacher["label"] == "ham"
        assert teacher["samples_used"] == 3

    def test_shrinkage_prevents_one_hot(self):
        row = self._elicit_row([{"spam": 0.0, "ham": 1.0}] * 4)
        op = System1AggregateOperator({
            "variant": "distill",
            "shrinkage": 0.1,
            "show_progress": False,
        })
        out = op.run([row])
        probs = out[0]["teacher"]["spam"]["probabilities"]
        assert probs["ham"] < 1.0
        assert probs["spam"] > 0.0

    def test_no_shrinkage_allows_one_hot(self):
        row = self._elicit_row([{"spam": 0.0, "ham": 1.0}] * 4)
        op = System1AggregateOperator({
            "variant": "distill",
            "shrinkage": 0.0,
            "show_progress": False,
        })
        out = op.run([row])
        probs = out[0]["teacher"]["spam"]["probabilities"]
        assert probs["ham"] == 1.0
        assert probs["spam"] == 0.0

    def test_low_consistency_drops_question(self):
        row = self._elicit_row([
            {"spam": 1.0, "ham": 0.0},
            {"spam": 0.0, "ham": 1.0},
            {"spam": 1.0, "ham": 0.0},
            {"spam": 0.0, "ham": 1.0},
        ])
        op = System1AggregateOperator({
            "variant": "distill",
            "consistency_threshold": 0.6,
            "show_progress": False,
        })
        out = op.run([row])
        teacher = out[0]["teacher"]["spam"]
        assert teacher["ok"] is False
        assert teacher["label"] is None

    def test_labeled_variant_raises(self):
        with pytest.raises((ValueError, KeyError)):
            System1AggregateOperator({"variant": "labeled", "show_progress": False})

    def test_aggregate_question_unit(self):
        question = _case_row()["questions"]["spam"]
        payload = {
            "method": "k_verbalized_mean",
            "model": "fake",
            "samples": [
                {"ok": True, "probabilities": {"spam": 0.0, "ham": 1.0}, "errors": []},
                {"ok": True, "probabilities": {"spam": 0.0, "ham": 1.0}, "errors": []},
            ],
        }
        gold = {"spam": 0.0, "ham": 1.0}
        result = aggregate_question(
            question, payload, gold,
            variant="distill", alpha=0.7, consistency_threshold=0.6, shrinkage=0.02,
        )
        assert result["ok"] is True
        assert result["label"] == "ham"
        assert result["consistency"] == 1.0


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


class TestBuildDataset:
    def _aggregated_row(self) -> Dict[str, Any]:
        row = _case_row()
        row["teacher"] = {
            "spam": {
                "ok": True,
                "method": "k_verbalized_mean",
                "model": "fake-teacher",
                "probabilities": {"spam": 0.01, "ham": 0.99},
                "label": "ham",
                "consistency": 1.0,
                "samples_used": 4,
                "samples_failed": 0,
                "reason": None,
                "gold_tv": 0.01,
                "gold_kl": 0.01,
            }
        }
        return row

    def test_distill_gold_nesting(self):
        op = System1BuildDatasetOperator({"variant": "distill", "show_progress": False})
        out = op.run([self._aggregated_row()])
        gold = out[0]["gold"]["spam"]
        assert "probabilities" in gold
        assert "label" in gold
        assert gold["probabilities"] == {"spam": 0.01, "ham": 0.99}
        assert gold["label"] == "ham"

    def test_distill_drops_intermediate_fields(self):
        op = System1BuildDatasetOperator({"variant": "distill", "show_progress": False})
        out = op.run([self._aggregated_row()])
        assert "elicitations" not in out[0]
        assert "teacher" not in out[0]
        assert "source" not in out[0]

    def test_labeled_passthrough(self):
        row = _case_row()  # gold already one-hot flat
        op = System1BuildDatasetOperator({"variant": "labeled", "show_progress": False})
        out = op.run([row])
        gold = out[0]["gold"]["spam"]
        assert "probabilities" in gold
        assert "label" in gold
        assert gold["label"] == "ham"

    def test_drops_failed_teacher(self):
        row = self._aggregated_row()
        row["teacher"]["spam"]["ok"] = False
        op = System1BuildDatasetOperator({"variant": "distill", "show_progress": False})
        out = op.run([row])
        assert len(out) == 0


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_distill_pipeline_with_fake_backend(self):
        backend = _FakeSystem1Backend({"spam": 0.0, "ham": 1.0})
        cases = _build_cases(RAW_ROWS)
        assert len(cases) == 2

        elicit_op = System1ElicitOperator(backend, {
            "method": "k_verbalized_mean",
            "samples": 2,
            "schema": SMS_SCHEMA,
            "shuffle_options": False,
            "show_progress": False,
            "max_workers": 1,
            "token_budget": {"check": False},
        })
        elicited = elicit_op.run(cases)
        assert all("elicitations" in r for r in elicited)
        assert backend.call_count == 4  # 2 rows × 2 samples

        agg_op = System1AggregateOperator({"variant": "distill", "show_progress": False})
        aggregated = agg_op.run(elicited)
        assert all("teacher" in r for r in aggregated)

        bd_op = System1BuildDatasetOperator({"variant": "distill", "show_progress": False})
        dataset = bd_op.run(aggregated)
        assert len(dataset) == 2
        for row in dataset:
            assert "state" in row
            assert "questions" in row
            assert "gold" in row
            for _, gold in row["gold"].items():
                assert "probabilities" in gold
                assert "label" in gold
