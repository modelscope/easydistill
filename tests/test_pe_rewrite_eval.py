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

"""Unit tests for the PE rewrite judge evaluator (combined single-call)."""

import json
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult
from easydistill.eval import PERewriteEvaluator
from easydistill.eval.pe_rewrite import ALL_METRICS, _parse_judge_response

FULL_VERDICT = {
    "intent_fidelity": 8,
    "text_rendering_completeness": 9,
    "detail_enrichment": 7,
    "visual_concreteness": 7,
    "compositional_coverage": 8,
    "scene_alignment": 7,
    "usability": 9,
    "language_consistency": True,
    "no_conflict": True,
}

TEMPLATE = "[judge] scene={scene} lang={language}\nIN: {instruction}\nOUT: {output}"


class ScriptedJudgeBackend(ModelBackend):
    """Returns one scripted combined verdict per call; records prompts."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: List[str] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        request = build_generation_request(messages)
        self.prompts.append(str(request.instruction))
        return GenerationResult(request=request, response=self.response, model=model_id or "fake")

    def health_check(self) -> bool:
        return True


def _build_evaluator(backend, **overrides):
    config = {
        "prompt_template": TEMPLATE,
        "show_progress": False,
        "max_workers": 1,
    }
    config.update(overrides)
    return PERewriteEvaluator(backend=backend, config=config)


class TestPERewriteEvaluator:
    def test_single_call_scores_all_metrics_and_preserves_row_fields(self):
        backend = ScriptedJudgeBackend(json.dumps(FULL_VERDICT))
        evaluator = _build_evaluator(backend)
        rows = [
            {
                "id": "r1",
                "instruction": "画只猫",
                "response": "改写后的猫",
                "scene": "photographic_realism",
                "language": "zh",
                "source_seed_id": "s1",
                "round": 0,
                "agent_trace": {"plan": {"status": "ok"}},
            }
        ]
        results = evaluator.run(rows)

        # Exactly one judge call for all nine metrics.
        assert len(backend.prompts) == 1
        row = results[0]
        for metric, expected in FULL_VERDICT.items():
            assert row[metric] == expected
        # Every original field must survive scoring.
        assert row["scene"] == "photographic_realism"
        assert row["source_seed_id"] == "s1"
        assert row["round"] == 0
        assert row["agent_trace"] == {"plan": {"status": "ok"}}
        assert row["response"] == "改写后的猫"

    def test_scene_and_language_are_injected_into_prompt(self):
        backend = ScriptedJudgeBackend(json.dumps(FULL_VERDICT))
        evaluator = _build_evaluator(backend)
        evaluator.run(
            [
                {
                    "id": "r1",
                    "instruction": "画张海报",
                    "response": "改写稿",
                    "scene": "design_layout",
                    "language": "zh",
                }
            ]
        )

        assert "scene=design_layout" in backend.prompts[0]
        assert "lang=zh" in backend.prompts[0]

    def test_missing_scene_falls_back_to_general(self):
        backend = ScriptedJudgeBackend(json.dumps(FULL_VERDICT))
        evaluator = _build_evaluator(backend)
        evaluator.run([{"id": "r1", "instruction": "画只猫", "response": "改写稿"}])

        assert "scene=general" in backend.prompts[0]

    def test_bool_metrics_are_converted(self):
        verdict = dict(FULL_VERDICT, language_consistency=1, no_conflict="false")
        backend = ScriptedJudgeBackend(json.dumps(verdict))
        evaluator = _build_evaluator(backend)
        results = evaluator.run([{"id": "r1", "instruction": "画只猫", "response": "改写稿"}])

        assert results[0]["language_consistency"] is True
        assert results[0]["no_conflict"] is False

    def test_rows_without_id_get_index_ids(self):
        backend = ScriptedJudgeBackend(json.dumps(FULL_VERDICT))
        evaluator = _build_evaluator(backend)
        results = evaluator.run(
            [
                {"instruction": "画只猫", "response": "改写稿A"},
                {"instruction": "画只狗", "response": "改写稿B"},
            ]
        )

        assert [r["id"] for r in results] == ["0", "1"]
        assert len(backend.prompts) == 2

    def test_verdict_wrapped_in_prose_is_salvaged(self):
        backend = ScriptedJudgeBackend(
            "评估结果如下：\n```json\n" + json.dumps(FULL_VERDICT) + "\n```"
        )
        evaluator = _build_evaluator(backend)
        results = evaluator.run([{"id": "r1", "instruction": "画只猫", "response": "改写稿"}])

        assert results[0]["intent_fidelity"] == 8
        assert results[0]["no_conflict"] is True

    def test_unparseable_response_yields_none_scores(self):
        backend = ScriptedJudgeBackend("totally not json")
        evaluator = _build_evaluator(backend)
        results = evaluator.run([{"id": "r1", "instruction": "画只猫", "response": "改写稿"}])

        assert all(results[0][metric] is None for metric in ALL_METRICS)

    def test_default_prompts_file_loads_combined_template(self):
        backend = ScriptedJudgeBackend(json.dumps(FULL_VERDICT))
        evaluator = PERewriteEvaluator(backend=backend, config={"show_progress": False})
        assert set(evaluator.metrics) == set(ALL_METRICS)
        # The bundled template must expose all four placeholders.
        for placeholder in ("{instruction}", "{output}", "{scene}", "{language}"):
            assert placeholder in evaluator.prompt_template


class TestParseJudgeResponse:
    def test_truncated_json_is_salvaged_per_field(self):
        raw = (
            '{"intent_fidelity": 8, "text_rendering_completeness": 9, '
            '"language_consistency": true, "no_conf'
        )
        scores = _parse_judge_response(raw)
        assert scores["intent_fidelity"] == 8
        assert scores["text_rendering_completeness"] == 9
        assert scores["language_consistency"] is True
        assert scores["no_conflict"] is None

    def test_out_of_range_scores_are_rejected(self):
        verdict = dict(FULL_VERDICT, intent_fidelity=15)
        scores = _parse_judge_response(json.dumps(verdict))
        assert scores["intent_fidelity"] is None
        assert scores["usability"] == 9

    def test_empty_response_returns_all_none(self):
        assert all(v is None for v in _parse_judge_response(None).values())
        assert all(v is None for v in _parse_judge_response("  ").values())
