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

"""End-to-end tests for the PE rewrite distillation pipeline."""

import json
from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.backends.utils import build_generation_request
from easydistill.data.models import GenerationResult
from easydistill.pipeline import PERewriteDistillPipeline

PLAN_SP = "PLAN_SP"
REWRITE_SP_ZH = "REWRITE_SP_ZH"
REWRITE_SP_EN = "REWRITE_SP_EN"
REFLECT_SP = "REFLECT_SP"
EXPAND_SP = "EXPAND_SP"
JUDGE_MARK = "[judge]"

EXPANSIONS = json.dumps(
    [
        {"topic": "夜景", "prompt": "夜景下的好样本"},
        {"topic": "雨天", "prompt": "雨天里的好样本"},
    ],
    ensure_ascii=False,
)

PLAN_OK = json.dumps({"scene": "photographic_realism", "language": "zh"})
REFLECT_PASS = json.dumps(
    {"changed": False, "rewritten_prompt": "", "notes": "ok"}, ensure_ascii=False
)
GOOD_VERDICT = json.dumps(
    {
        "intent_fidelity": 8,
        "text_rendering_completeness": 9,
        "detail_enrichment": 8,
        "visual_concreteness": 8,
        "compositional_coverage": 8,
        "scene_alignment": 8,
        "usability": 9,
        "language_consistency": True,
        "no_conflict": True,
    }
)
BAD_VERDICT = json.dumps(
    {
        "intent_fidelity": 3,  # below the >=7 gate -> dropped by the filter
        "text_rendering_completeness": 9,
        "detail_enrichment": 8,
        "visual_concreteness": 8,
        "compositional_coverage": 8,
        "scene_alignment": 8,
        "usability": 9,
        "language_consistency": True,
        "no_conflict": True,
    }
)


class ScriptedPipelineBackend(ModelBackend):
    """Routes plan/rewrite/reflection/judge calls by prompt markers.

    The judge verdict is keyed by which original instruction appears in the
    judge prompt, so different rows can score differently. Records the
    ``model_id`` per step for model-separation assertions.
    """

    def __init__(self, verdicts: Dict[str, str]):
        self.verdicts = verdicts
        self.model_by_step: Dict[str, Optional[str]] = {}

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        request = build_generation_request(messages)
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        prompt = str(request.instruction)
        if PLAN_SP in system:
            step, response = "plan", PLAN_OK
        elif EXPAND_SP in system:
            step, response = "expand", EXPANSIONS
        elif REWRITE_SP_ZH in system or REWRITE_SP_EN in system:
            step, response = "rewrite", f"改写:{prompt.splitlines()[1]}"
        elif REFLECT_SP in system:
            step, response = "reflection", REFLECT_PASS
        elif JUDGE_MARK in prompt:
            step = "judge"
            response = next((v for key, v in self.verdicts.items() if key in prompt), GOOD_VERDICT)
        else:
            step, response = "unknown", ""
        self.model_by_step[step] = model_id
        return GenerationResult(request=request, response=response, model=model_id or "fake")

    def health_check(self) -> bool:
        return True


@pytest.fixture
def pipeline_env(tmp_path):
    scene_dir = tmp_path / "prompts"
    scene_dir.mkdir()
    (scene_dir / "rewrite_general_zh.txt").write_text(REWRITE_SP_ZH, encoding="utf-8")
    (scene_dir / "rewrite_general_en.txt").write_text(REWRITE_SP_EN, encoding="utf-8")
    (tmp_path / "sys_zh.txt").write_text("学生指令", encoding="utf-8")
    return tmp_path, scene_dir


def _build_pipeline(backend, tmp_path, scene_dir):
    pipeline_config = [
        {
            "stage": "agentic_rewrite",
            "config": {
                "plan": {"prompt_template": PLAN_SP},
                "rewrite": {"scene_prompt_dir": str(scene_dir)},
                "reflection": {"prompt_template": REFLECT_SP},
                "max_workers": 1,
                "show_progress": False,
                "retry_attempts": 1,
            },
        },
        {
            "stage": "pe_rewrite_eval",
            "config": {
                "model_id": "judge-model",
                "prompt_template": JUDGE_MARK + " {scene} {language}\n{instruction}\n{output}",
                "show_progress": False,
                "max_workers": 1,
            },
            "output_path": str(tmp_path / "scored.jsonl"),
        },
        {"stage": "quality_filter", "config": {}},
        {"stage": "build_sft"},
    ]
    return PERewriteDistillPipeline(
        backend=backend,
        pipeline_config=pipeline_config,
        dataset_config={
            "input_path": str(tmp_path / "in.jsonl"),
            "output_path": str(tmp_path / "sft.jsonl"),
        },
        sft_config={"system_prompt_zh_file": str(tmp_path / "sys_zh.txt")},
    )


class TestPERewriteDistillPipeline:
    def test_full_chain_filters_bad_rows_and_builds_sft(self, pipeline_env):
        tmp_path, scene_dir = pipeline_env
        backend = ScriptedPipelineBackend(verdicts={"坏样本": BAD_VERDICT})
        pipeline = _build_pipeline(backend, tmp_path, scene_dir)
        rows = [
            {"id": "a", "instruction": "好样本"},
            {"id": "b", "instruction": "坏样本"},
        ]

        samples = pipeline.run_with_data(rows)

        # The low intent_fidelity row is filtered out before SFT building.
        assert len(samples) == 1
        messages = samples[0]["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert messages[0]["content"] == "学生指令"
        assert messages[1]["content"] == "好样本"
        assert messages[2]["content"].startswith("改写:")
        # The intermediate scored jsonl is persisted for auditing.
        scored = [json.loads(line) for line in (tmp_path / "scored.jsonl").read_text().splitlines()]
        assert {r["instruction"] for r in scored} == {"好样本", "坏样本"}

    def test_judge_runs_on_its_own_model(self, pipeline_env):
        tmp_path, scene_dir = pipeline_env
        backend = ScriptedPipelineBackend(verdicts={})
        pipeline = _build_pipeline(backend, tmp_path, scene_dir)
        pipeline.run_with_data([{"id": "a", "instruction": "好样本"}])

        assert backend.model_by_step["judge"] == "judge-model"
        # Teacher steps keep the backend default (no per-step override here).
        assert backend.model_by_step["rewrite"] is None

    def test_last_stage_must_be_build_sft(self, pipeline_env):
        tmp_path, scene_dir = pipeline_env
        backend = ScriptedPipelineBackend(verdicts={})
        with pytest.raises(ValueError, match="build_sft"):
            PERewriteDistillPipeline(
                backend=backend,
                pipeline_config=[{"stage": "agentic_rewrite", "config": {}}],
                dataset_config={"input_path": "x.jsonl"},
            )

    def test_seed_expansion_stage_feeds_the_chain_with_lineage(self, pipeline_env):
        tmp_path, scene_dir = pipeline_env
        backend = ScriptedPipelineBackend(verdicts={})
        pipeline = _build_pipeline(backend, tmp_path, scene_dir)
        pipeline.pipeline_config.insert(
            0,
            {
                "stage": "seed_anchored_expansion",
                "config": {
                    "prompt_template": EXPAND_SP,
                    "rounds": 1,
                    "generations_per_round": 2,
                    "max_workers": 1,
                    "show_progress": False,
                },
            },
        )

        samples = pipeline.run_with_data([{"id": "seed1", "instruction": "好样本种子"}])

        # 1 seed -> 2 expansions -> rewrite/judge/filter -> 2 SFT samples.
        assert len(samples) == 2
        users = {s["messages"][1]["content"] for s in samples}
        assert users == {"夜景下的好样本", "雨天里的好样本"}
        # Expansion lineage must survive the whole chain into SFT metadata.
        for sample in samples:
            assert sample["metadata"]["source_seed_id"] == "seed1"
            assert sample["metadata"]["round"] == 0
