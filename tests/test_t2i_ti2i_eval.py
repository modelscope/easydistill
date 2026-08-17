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

"""Unit tests for the T2I / TI2I single-file evaluators."""

import json
import re
from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval import (
    T2IMultiModelEvaluator,
    T2ISingleModelEvaluator,
    TI2IMultiModelEvaluator,
    TI2ISingleModelEvaluator,
)
from easydistill.eval.t2i_multi_model import _parse_json_block


class ScriptedJudgeBackend(ModelBackend):
    """Deterministic backend that answers teacher scoring and Debate prompts.

    ``scores`` maps L3 dimension name -> (score, confidence); scoring prompts
    are answered per requested dimension, and the three Debate steps plus the
    reason-synthesis prompt are recognized by their role headers.
    """

    def __init__(
        self,
        scores: Dict[str, Any],
        debate_score: int = 2,
        default_score: int = 3,
    ):
        self.scores = scores
        self.debate_score = debate_score
        self.default_score = default_score

    def _reply(self, prompt: str) -> str:
        if "评审理由规范器" in prompt:
            return json.dumps({"reason": "规范化理由"}, ensure_ascii=False)
        if "控辩双方书记员" in prompt:
            return json.dumps(
                {"prosecution": "低分论点", "defense": "高分论点"}, ensure_ascii=False
            )
        if "最终仲裁法官" in prompt:
            return json.dumps(
                {
                    "score": self.debate_score,
                    "applicable": True,
                    "confidence": 0.95,
                    "reason": "仲裁理由",
                    "adopted": "final",
                },
                ensure_ascii=False,
            )
        if "独立仲裁法官" in prompt:
            return json.dumps(
                {
                    "score": self.debate_score,
                    "applicable": True,
                    "confidence": 0.9,
                    "reason": "初评",
                },
                ensure_ascii=False,
            )
        # Teacher scoring prompt: answer every requested dimension line ("- 名称：...").
        judgments = []
        for name in re.findall(r"^- (.+?)：", prompt, flags=re.MULTILINE):
            score, confidence = self.scores.get(name, (self.default_score, 0.9))
            judgments.append(
                {
                    "dimension": name,
                    "applicable": True,
                    "score": score,
                    "confidence": confidence,
                    "reason": f"{name} 的理由",
                }
            )
        return json.dumps({"dimension_judgments": judgments}, ensure_ascii=False)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        prompt = messages[-1]["content"]
        if isinstance(prompt, list):
            prompt = " ".join(
                item.get("text", "") for item in prompt if item.get("type") == "text"
            )
        return GenerationResult(
            request=GenerationRequest(instruction=prompt),
            response=self._reply(prompt),
            model=model_id or "fake",
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def t2i_pool_path(tmp_path):
    pool = {
        "aggregation": {
            "method": "equal_weight_mean",
            "safety_veto": True,
            "total_excludes": ["Safety"],
            "na_handling": "exclude_from_mean",
        },
        "dimensions": [
            {
                "name": "Quality",
                "l2_groups": [
                    {
                        "items": [
                            {"name": "Clarity", "criteria": {"4": "清晰", "0": "模糊"}},
                            {"name": "Color", "criteria": {"4": "协调", "0": "失真"}},
                        ]
                    }
                ],
            },
            {
                "name": "Safety",
                "l2_groups": [{"items": [{"name": "Safety Compliance", "criteria": {}}]}],
            },
        ],
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _config(pool_path):
    return {"dimensions_path": pool_path, "max_workers": 1, "retry_delay_sec": 0.01}


class TestT2IMultiModel:
    def test_conflict_triggers_debate_and_export(self, t2i_pool_path):
        # Teacher A and B disagree on Clarity by 3 points -> Debate on Clarity only.
        teacher_a = ScriptedJudgeBackend(
            {"Clarity": (4, 0.9), "Color": (4, 0.9), "Safety Compliance": (4, 0.9)}
        )
        teacher_b = ScriptedJudgeBackend(
            {"Clarity": (1, 0.6), "Color": (4, 0.6), "Safety Compliance": (4, 0.6)}
        )
        arbiter = ScriptedJudgeBackend({}, debate_score=2)
        evaluator = T2IMultiModelEvaluator(
            teachers={"model-a": teacher_a, "model-b": teacher_b},
            arbiter=arbiter,
            config=_config(t2i_pool_path),
        )
        results = evaluator.run([{"prompt_id": "p1", "prompt": "一只猫"}])

        assert len(results) == 1
        result = results[0]
        assert result["case_id"] == "p1"
        assert result["conflict_dims"] == ["Clarity"]
        judgments = {j["dimension"]: j for j in result["final_judgments"]}
        assert judgments["Clarity"]["was_debated"] is True
        assert judgments["Clarity"]["final_score"] == 2
        assert judgments["Clarity"]["final_source"] == "debate_arbitration"
        assert judgments["Color"]["was_debated"] is False
        assert judgments["Color"]["final_score"] == 4
        # Quality dims only: Clarity 50 + Color 100 -> 75; Safety excluded.
        assert result["overall_score_100"] == 75.0

        summary = evaluator.aggregate(results)
        assert summary["cases"] == 1
        assert summary["cases_failed"] == 0
        assert summary["overall_score_stats"]["mean"] == 75.0
        assert summary["debated_dims_total"] == 1

        bins = evaluator.export_training_data(results)
        dpo_dims = [row["dimension"] for row in bins["dpo"]]
        assert dpo_dims == ["Clarity"]
        assert bins["dpo"][0]["chosen_score"] == 2
        assert bins["dpo"][0]["rejected_score"] == 4
        # Debated Clarity has confidence 0.95 -> sft; the consensus dims stay uncertain.
        assert [row["dimension"] for row in bins["sft"]] == ["Clarity"]
        assert len(bins["uncertain"]) == 2

    def test_debate_slots_prefer_largest_spread(self, t2i_pool_path):
        # Clarity spread=2, Color spread=4; with max_debate_dims=1 the more
        # severe Color conflict must win the slot (not alphabetical Clarity).
        teacher_a = ScriptedJudgeBackend(
            {"Clarity": (3, 0.9), "Color": (4, 0.9), "Safety Compliance": (4, 0.9)}
        )
        teacher_b = ScriptedJudgeBackend(
            {"Clarity": (1, 0.6), "Color": (0, 0.6), "Safety Compliance": (4, 0.6)}
        )
        evaluator = T2IMultiModelEvaluator(
            teachers={"model-a": teacher_a, "model-b": teacher_b},
            arbiter=ScriptedJudgeBackend({}, debate_score=2),
            config={**_config(t2i_pool_path), "max_debate_dims": 1},
        )
        result = evaluator.run([{"prompt_id": "p1", "prompt": "一只猫"}])[0]
        assert result["conflict_dims"] == ["Color"]
        judgments = {j["dimension"]: j for j in result["final_judgments"]}
        assert judgments["Color"]["was_debated"] is True
        assert judgments["Clarity"]["was_debated"] is False

    def test_safety_compliance_zero_vetoes_overall(self, t2i_pool_path):
        teacher = ScriptedJudgeBackend(
            {"Clarity": (4, 0.9), "Color": (4, 0.9), "Safety Compliance": (0, 0.9)}
        )
        evaluator = T2IMultiModelEvaluator(
            teachers={"model-a": teacher}, config=_config(t2i_pool_path)
        )
        results = evaluator.run([{"prompt_id": "p1", "prompt": "危险内容"}])
        result = results[0]
        # Single teacher: no conflicts, no debate.
        assert result["conflict_dims"] == []
        assert result["n_debated"] == 0
        assert result["overall_score_100"] == 0.0
        assert result["overall"]["safety_veto_triggered"] is True

    def test_failed_case_does_not_block_batch(self, t2i_pool_path):
        teacher = ScriptedJudgeBackend({"Clarity": (4, 0.9)})
        evaluator = T2IMultiModelEvaluator(
            teachers={"model-a": teacher}, config=_config(t2i_pool_path)
        )
        evaluator._extract_case = None  # force a failure on every case
        results = evaluator.run([{"prompt_id": "p1", "prompt": "x"}])
        assert len(results) == 1
        assert results[0]["error"]


class TestT2ISingleModel:
    def test_single_teacher_no_debate(self, t2i_pool_path):
        backend = ScriptedJudgeBackend(
            {"Clarity": (3, 0.9), "Color": (4, 0.9), "Safety Compliance": (4, 0.9)}
        )
        evaluator = T2ISingleModelEvaluator(
            teacher="fake-model", backend=backend, config=_config(t2i_pool_path)
        )
        results = evaluator.run([{"prompt_id": "p1", "prompt": "一只猫"}])
        result = results[0]
        judgments = {j["dimension"]: j for j in result["final_judgments"]}
        assert len(judgments) == 3
        assert all(j["was_debated"] is False for j in judgments.values())
        assert judgments["Clarity"]["final_source"] == "single_teacher_score"
        # Clarity 75 + Color 100 -> 87.5; Safety excluded from the total.
        assert result["overall_score_100"] == 87.5

    def test_export_training_data_bins(self, t2i_pool_path):
        backend = ScriptedJudgeBackend(
            {"Clarity": (3, 0.9), "Color": (4, 0.5), "Safety Compliance": (4, 0.9)}
        )
        evaluator = T2ISingleModelEvaluator(
            teacher="fake-model", backend=backend, config=_config(t2i_pool_path)
        )
        results = evaluator.run([{"prompt_id": "p1", "prompt": "一只猫"}])
        bins = evaluator.export_training_data(results)
        assert not bins["dpo"]  # no debate in single-model mode
        assert {row["dimension"] for row in bins["sft"]} == {"Clarity", "Safety Compliance"}
        assert [row["dimension"] for row in bins["uncertain"]] == ["Color"]


@pytest.fixture
def ti2i_pool_path(tmp_path):
    pool = {
        "aggregation": {"method": "equal_weight_mean", "na_handling": "exclude_from_mean"},
        "dimensions": [
            {
                "name": "Instruction Following",
                "l2_groups": [
                    {"items": [{"name": "Edit Accuracy", "criteria": {"4": "完全执行"}}]}
                ],
            },
            {
                "name": "Consistency",
                "l2_groups": [
                    {"items": [{"name": "Background Preservation", "criteria": {}}]}
                ],
            },
        ],
    }
    path = tmp_path / "ti2i_pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return str(path)


class TestTI2IMultiModel:
    def test_conflict_debate_flow(self, ti2i_pool_path):
        teacher_a = ScriptedJudgeBackend(
            {"Edit Accuracy": (4, 0.9), "Background Preservation": (3, 0.9)}
        )
        teacher_b = ScriptedJudgeBackend(
            {"Edit Accuracy": (0, 0.7), "Background Preservation": (3, 0.7)}
        )
        arbiter = ScriptedJudgeBackend({}, debate_score=2)
        evaluator = TI2IMultiModelEvaluator(
            teachers={"model-a": teacher_a, "model-b": teacher_b},
            arbiter=arbiter,
            config=_config(ti2i_pool_path),
        )
        results = evaluator.run([{"case_id": "c1", "instruction": "替换天空"}])
        result = results[0]
        assert result["conflict_dims"] == ["Edit Accuracy"]
        judgments = {j["dimension"]: j for j in result["final_judgments"]}
        assert judgments["Edit Accuracy"]["final_score"] == 2
        assert judgments["Edit Accuracy"]["was_debated"] is True
        # Equal-weight mean over both dims: (50 + 75) / 2 = 62.5, no excludes.
        assert result["overall_score_100"] == 62.5


class TestTI2ISingleModel:
    def test_runs_and_aggregates(self, ti2i_pool_path):
        backend = ScriptedJudgeBackend(
            {"Edit Accuracy": (4, 0.9), "Background Preservation": (2, 0.9)}
        )
        evaluator = TI2ISingleModelEvaluator(
            teacher="fake-model", backend=backend, config=_config(ti2i_pool_path)
        )
        results = evaluator.run(
            [
                {"case_id": "c1", "instruction": "替换天空"},
                {"case_id": "c2", "instruction": "去除水印"},
            ]
        )
        assert [r["case_id"] for r in results] == ["c1", "c2"]
        # (100 + 50) / 2 per case.
        assert all(r["overall_score_100"] == 75.0 for r in results)
        summary = evaluator.aggregate(results)
        assert summary["cases"] == 2
        assert summary["overall_score_stats"]["mean"] == 75.0


class TestParseJsonBlock:
    def test_tolerates_markdown_fences(self):
        text = '```json\n{"score": 3}\n```'
        assert _parse_json_block(text) == {"score": 3}

    def test_extracts_object_from_noise(self):
        text = '前置说明 {"reason": "ok"} 后置说明'
        assert _parse_json_block(text) == {"reason": "ok"}

    def test_rejects_non_object(self):
        with pytest.raises(ValueError):
            _parse_json_block("[1, 2, 3]")

    def test_rejects_missing_json(self):
        with pytest.raises(ValueError):
            _parse_json_block("no json here")


class TestMajorityVote:
    def test_confidence_weighted_majority(self):
        votes = [
            {"teacher": "a", "score": 4, "confidence": 0.9, "reason": "好"},
            {"teacher": "b", "score": 1, "confidence": 0.3, "reason": "差"},
        ]
        majority = T2IMultiModelEvaluator._majority(votes)
        assert majority["score"] == 4
        assert majority["applicable"] is True

    def test_all_na_votes(self):
        votes = [{"teacher": "a", "score": None, "applicable": False, "reason": "不适用"}]
        majority = T2IMultiModelEvaluator._majority(votes)
        assert majority["score"] is None
        assert majority["applicable"] is False
