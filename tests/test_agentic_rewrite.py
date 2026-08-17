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

"""Unit tests for the agentic prompt rewrite operator."""

import json
from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult
from easydistill.rewrite import AgenticPromptRewriteOperator

PLAN_SP = "PLAN_SP"
REWRITE_SP_ZH = "REWRITE_SP_ZH"
REWRITE_SP_EN = "REWRITE_SP_EN"
REFLECT_SP = "REFLECT_SP"

PLAN_OK = json.dumps(
    {
        "scene": "photographic_realism",
        "language": "zh",
    },
    ensure_ascii=False,
)
REFLECT_PASS = json.dumps(
    {"changed": False, "rewritten_prompt": "", "notes": "ok"}, ensure_ascii=False
)


def _rewrite_json(gen_prompt: str, negative_prompt: str = "blurry") -> str:
    """Teacher-style JSON response; negative_prompt must be discarded."""
    return json.dumps(
        {"gen_prompt": gen_prompt, "negative_prompt": negative_prompt},
        ensure_ascii=False,
    )


class ScriptedAgentBackend(ModelBackend):
    """Routes responses by the system prompt marker of each step.

    Records ``(step, model_id, system_prompt)`` per call so tests can assert
    call order and per-step overrides.
    """

    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        if PLAN_SP in system:
            step = "plan"
        elif REWRITE_SP_ZH in system or REWRITE_SP_EN in system:
            step = "rewrite"
        elif REFLECT_SP in system:
            step = "reflection"
        else:
            step = "unknown"
        self.calls.append({"step": step, "model_id": model_id, "system": system})
        return GenerationResult(
            request=build_generation_request(messages),
            response=self.responses[step],
            model=model_id or "fake",
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def scene_dir(tmp_path):
    """Prompt dir with the two required general fallback prompts."""
    (tmp_path / "rewrite_general_zh.txt").write_text(REWRITE_SP_ZH, encoding="utf-8")
    (tmp_path / "rewrite_general_en.txt").write_text(REWRITE_SP_EN, encoding="utf-8")
    return tmp_path


def _build_operator(backend, scene_dir, **overrides):
    config = {
        "plan": {"prompt_template": PLAN_SP},
        "rewrite": {"scene_prompt_dir": str(scene_dir)},
        "reflection": {"prompt_template": REFLECT_SP},
        "max_workers": 1,
        "show_progress": False,
        "retry_attempts": 1,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return AgenticPromptRewriteOperator(backend=backend, config=config)


class TestHappyPath:
    def test_three_steps_run_in_order_and_record_is_complete(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("改写后的猫", "blurry, low quality"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(
            [{"instruction": "画只猫", "source_seed_id": "s1", "round": 0}]
        )

        assert [c["step"] for c in backend.calls] == ["plan", "rewrite", "reflection"]
        assert len(outputs) == 1
        record = outputs[0]
        assert record["instruction"] == "画只猫"
        assert record["response"] == "改写后的猫"
        assert record["scene"] == "photographic_realism"
        assert record["language"] == "zh"
        # Only gen_prompt is kept; other teacher JSON fields are discarded.
        assert "negative_prompt" not in record
        # Expansion lineage fields must be passed through.
        assert record["source_seed_id"] == "s1"
        assert record["round"] == 0
        trace = record["agent_trace"]
        assert trace["plan"]["status"] == "ok"
        assert trace["rewrite"]["status"] == "ok"
        assert trace["rewrite"]["draft"] == "改写后的猫"
        assert trace["reflection"]["status"] == "ok"
        assert trace["reflection"]["changed"] is False
        assert set(trace["durations"]) == {"plan", "rewrite", "reflection"}

    def test_scene_prompt_replaces_general_fallback(self, scene_dir):
        scene_sp = f"{REWRITE_SP_ZH} PHOTO_FULL_SP"
        (scene_dir / "rewrite_photographic_realism_zh.txt").write_text(
            scene_sp, encoding="utf-8"
        )
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        operator.run(["画只猫"])

        rewrite_call = next(c for c in backend.calls if c["step"] == "rewrite")
        assert rewrite_call["system"] == scene_sp

    def test_common_block_is_prepended_to_scene_prompts(self, scene_dir):
        (scene_dir / "rewrite_common_zh.txt").write_text("COMMON_ZH\n", encoding="utf-8")
        scene_sp = f"{REWRITE_SP_ZH} PHOTO_FULL_SP"
        (scene_dir / "rewrite_photographic_realism_zh.txt").write_text(
            scene_sp, encoding="utf-8"
        )
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        operator.run(["画只猫"])

        rewrite_call = next(c for c in backend.calls if c["step"] == "rewrite")
        # Common block first, then the scene body; general_zh also gets it.
        assert rewrite_call["system"] == f"COMMON_ZH\n\n{scene_sp}"
        assert operator.scene_prompts["general_zh"] == f"COMMON_ZH\n\n{REWRITE_SP_ZH}"
        # No English common file -> English prompts stay untouched.
        assert operator.scene_prompts["general_en"] == REWRITE_SP_EN

    def test_language_selects_prompt_variant(self, scene_dir):
        plan_en = json.dumps(
            {"scene": "photographic_realism", "language": "en"}
        )
        backend = ScriptedAgentBackend(
            {
                "plan": plan_en,
                "rewrite": _rewrite_json("a cat"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        operator.run(["a cat please"])

        rewrite_call = next(c for c in backend.calls if c["step"] == "rewrite")
        # No photographic_realism_en file exists -> general_en fallback.
        assert rewrite_call["system"] == REWRITE_SP_EN

    def test_missing_general_prompt_raises(self, tmp_path):
        backend = ScriptedAgentBackend({})
        with pytest.raises(FileNotFoundError, match="rewrite_general"):
            _build_operator(backend, tmp_path)

    def test_per_step_model_id_override(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(
            backend, scene_dir, model_id="default-model", reflection={"model_id": "checker"}
        )
        operator.run(["画只猫"])

        by_step = {c["step"]: c["model_id"] for c in backend.calls}
        assert by_step["plan"] == "default-model"
        assert by_step["rewrite"] == "default-model"
        assert by_step["reflection"] == "checker"

    def test_stream_output_path_writes_rows_incrementally(self, scene_dir, tmp_path):
        stream_path = tmp_path / "out" / "stream.jsonl"
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("改写后的猫"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(
            backend, scene_dir, stream_output_path=str(stream_path)
        )
        outputs = operator.run(["画只猫", "再画只狗"])

        lines = stream_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(outputs) == 2
        streamed = sorted(
            (json.loads(line) for line in lines), key=lambda r: r["instruction"]
        )
        expected = sorted(outputs, key=lambda r: r["instruction"])
        assert streamed == expected

    def test_reflection_revision_replaces_draft(self, scene_dir):
        reflect = json.dumps(
            {"changed": True, "rewritten_prompt": "修正稿", "notes": "fixed"},
            ensure_ascii=False,
        )
        backend = ScriptedAgentBackend(
            {"plan": PLAN_OK, "rewrite": _rewrite_json("初稿"), "reflection": reflect}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "修正稿"
        assert outputs[0]["agent_trace"]["rewrite"]["draft"] == "初稿"
        assert outputs[0]["agent_trace"]["reflection"]["changed"] is True


class TestRewriteParsing:
    def test_truncated_json_gen_prompt_is_salvaged(self, scene_dir):
        # Token-limit truncation: the closing quote and brace never arrive.
        truncated = '{"gen_prompt": "一只黑白相间的猫，柔和的窗光'
        backend = ScriptedAgentBackend(
            {"plan": PLAN_OK, "rewrite": truncated, "reflection": REFLECT_PASS}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "一只黑白相间的猫，柔和的窗光"
        assert outputs[0]["agent_trace"]["rewrite"]["status"] == "ok_salvaged"

    def test_plain_text_response_is_used_as_draft(self, scene_dir):
        backend = ScriptedAgentBackend(
            {"plan": PLAN_OK, "rewrite": "纯文本改写结果", "reflection": REFLECT_PASS}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "纯文本改写结果"
        assert outputs[0]["agent_trace"]["rewrite"]["status"] == "plain_text"


class TestFallbacks:
    def test_unparseable_plan_falls_back_to_general_and_cjk_detection(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": "not json at all",
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫", "a cat please"])

        assert len(outputs) == 2
        assert all(o["scene"] == "general" for o in outputs)
        assert outputs[0]["language"] == "zh"
        assert outputs[1]["language"] == "en"
        assert outputs[0]["agent_trace"]["plan"]["status"] == "parse_failed"
        # Language routing still applies on the fallback scene.
        rewrite_systems = [
            c["system"] for c in backend.calls if c["step"] == "rewrite"
        ]
        assert rewrite_systems == [REWRITE_SP_ZH, REWRITE_SP_EN]

    def test_invalid_scene_falls_back_to_general(self, scene_dir):
        plan = json.dumps({"scene": "made_up_scene", "language": "zh"})
        backend = ScriptedAgentBackend(
            {"plan": plan, "rewrite": _rewrite_json("draft"), "reflection": REFLECT_PASS}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["scene"] == "general"
        assert outputs[0]["agent_trace"]["plan"]["status"] == "invalid_scene"

    def test_unparseable_reflection_keeps_draft(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("初稿"),
                "reflection": "garbage output",
            }
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "初稿"
        assert outputs[0]["agent_trace"]["reflection"]["status"] == "parse_failed"
        assert outputs[0]["agent_trace"]["reflection"]["changed"] is False

    def test_broken_json_with_readable_pass_verdict_is_salvaged(self, scene_dir):
        # Unescaped quotes in `notes` break the JSON, but the verdict is clear.
        broken = '{"changed": false, "rewritten_prompt": "", "notes": "文字"光照"保留"}'
        backend = ScriptedAgentBackend(
            {"plan": PLAN_OK, "rewrite": _rewrite_json("初稿"), "reflection": broken}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "初稿"
        assert outputs[0]["agent_trace"]["reflection"]["status"] == "ok_salvaged"
        assert outputs[0]["agent_trace"]["reflection"]["changed"] is False

    def test_changed_true_with_identical_text_counts_as_unchanged(self, scene_dir):
        reflect = json.dumps({"changed": True, "rewritten_prompt": "初稿", "notes": ""})
        backend = ScriptedAgentBackend(
            {"plan": PLAN_OK, "rewrite": _rewrite_json("初稿"), "reflection": reflect}
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫"])

        assert outputs[0]["response"] == "初稿"
        assert outputs[0]["agent_trace"]["reflection"]["changed"] is False

    def test_empty_rewrite_drops_row_without_blocking_others(self, scene_dir):
        class FailFirstRewrite(ScriptedAgentBackend):
            def __init__(self):
                super().__init__(
                    {
                        "plan": PLAN_OK,
                        "rewrite": _rewrite_json("draft"),
                        "reflection": REFLECT_PASS,
                    }
                )
                self._rewrite_count = 0

            def generate(self, messages, **kwargs):
                system = next(
                    (m["content"] for m in messages if m.get("role") == "system"), ""
                )
                if REWRITE_SP_ZH in system or REWRITE_SP_EN in system:
                    self._rewrite_count += 1
                    if self._rewrite_count == 1:
                        self.responses["rewrite"] = ""
                    else:
                        self.responses["rewrite"] = _rewrite_json("draft")
                return super().generate(messages, **kwargs)

        backend = FailFirstRewrite()
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫", "画只狗"])

        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "画只狗"


class TestInputNormalization:
    def test_accepts_strings_and_skips_invalid_rows(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        outputs = operator.run(["画只猫", "", {"no_instruction": 1}, 42])

        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "画只猫"

    def test_empty_input_returns_empty(self, scene_dir):
        backend = ScriptedAgentBackend(
            {
                "plan": PLAN_OK,
                "rewrite": _rewrite_json("draft"),
                "reflection": REFLECT_PASS,
            }
        )
        operator = _build_operator(backend, scene_dir)
        assert operator.run([]) == []
