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

"""Unit tests for the PE rewrite local stage runners (filter + build_sft)."""

import json

from easydistill.cli.runners.pe_rewrite import (
    run_pe_rewrite_build_sft,
    run_pe_rewrite_filter,
)


def _scored_row(row_id, language="zh", scene="photographic_realism", **overrides):
    row = {
        "id": row_id,
        "instruction": f"原始 prompt {row_id}",
        "response": f"改写后 prompt {row_id}",
        "scene": scene,
        "language": language,
        "agent_trace": {"plan": {"status": "ok"}},
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
    row.update(overrides)
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestPERewriteFilter:
    def test_default_thresholds_drop_low_scores_and_bool_fails(self, tmp_path):
        rows = [
            _scored_row("good"),
            _scored_row("low_intent", intent_fidelity=6),
            _scored_row("bad_lang", language_consistency=False),
            _scored_row("missing_metric", usability=None),
        ]
        _write_jsonl(tmp_path / "scored.jsonl", rows)
        config = _write_yaml(
            tmp_path / "filter.yaml",
            f"""
job_type: pe_rewrite_filter
dataset:
  input_path: {tmp_path / "scored.jsonl"}
  output_path: {tmp_path / "filtered.jsonl"}
""",
        )
        run_pe_rewrite_filter(config)

        kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
        assert [r["id"] for r in kept] == ["good"]
        # Full rows (scene / trace fields) survive filtering untouched.
        assert kept[0]["agent_trace"] == {"plan": {"status": "ok"}}

    def test_keep_top_ratio_selects_by_average_score(self, tmp_path):
        rows = [
            _scored_row("high", detail_enrichment=9),
            _scored_row("mid", detail_enrichment=8),
            _scored_row("lowest_but_passing", detail_enrichment=6),
            _scored_row("second_lowest", detail_enrichment=7),
        ]
        _write_jsonl(tmp_path / "scored.jsonl", rows)
        config = _write_yaml(
            tmp_path / "filter.yaml",
            f"""
job_type: pe_rewrite_filter
quality_filter:
  keep_top_ratio: 0.5
dataset:
  input_path: {tmp_path / "scored.jsonl"}
  output_path: {tmp_path / "filtered.jsonl"}
""",
        )
        run_pe_rewrite_filter(config)

        kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
        assert [r["id"] for r in kept] == ["high", "mid"]

    def test_top_ratio_is_applied_per_scene_and_protects_small_scenes(self, tmp_path):
        # Global top-50% ranking would evict both low-scoring cultural rows;
        # per-scene quotas must keep ceil(2*0.5)=1 of them.
        rows = [
            _scored_row("photo_hi", scene="photographic_realism", detail_enrichment=9),
            _scored_row("photo_mid", scene="photographic_realism", detail_enrichment=8),
            _scored_row("cult_a", scene="cultural_heritage_art", visual_concreteness=6),
            _scored_row(
                "cult_b", scene="cultural_heritage_art", visual_concreteness=6, detail_enrichment=7
            ),
        ]
        _write_jsonl(tmp_path / "scored.jsonl", rows)
        config = _write_yaml(
            tmp_path / "filter.yaml",
            f"""
job_type: pe_rewrite_filter
quality_filter:
  keep_top_ratio: 0.5
dataset:
  input_path: {tmp_path / "scored.jsonl"}
  output_path: {tmp_path / "filtered.jsonl"}
""",
        )
        run_pe_rewrite_filter(config)

        kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
        by_scene = {}
        for r in kept:
            by_scene.setdefault(r["scene"], []).append(r["id"])
        assert by_scene["photographic_realism"] == ["photo_hi"]
        # cultural survives with its per-scene quota (higher-avg row wins).
        assert by_scene["cultural_heritage_art"] == ["cult_a"]

    def test_per_scene_false_restores_global_ranking(self, tmp_path):
        rows = [
            _scored_row("photo_hi", scene="photographic_realism", detail_enrichment=9),
            _scored_row("photo_mid", scene="photographic_realism", detail_enrichment=8),
            _scored_row("cult_a", scene="cultural_heritage_art", visual_concreteness=6),
            _scored_row(
                "cult_b", scene="cultural_heritage_art", visual_concreteness=6, detail_enrichment=7
            ),
        ]
        _write_jsonl(tmp_path / "scored.jsonl", rows)
        config = _write_yaml(
            tmp_path / "filter.yaml",
            f"""
job_type: pe_rewrite_filter
quality_filter:
  keep_top_ratio: 0.5
  per_scene: false
dataset:
  input_path: {tmp_path / "scored.jsonl"}
  output_path: {tmp_path / "filtered.jsonl"}
""",
        )
        run_pe_rewrite_filter(config)

        kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
        assert {r["scene"] for r in kept} == {"photographic_realism"}

    def test_min_scores_override(self, tmp_path):
        rows = [_scored_row("r1", intent_fidelity=5)]
        _write_jsonl(tmp_path / "scored.jsonl", rows)
        config = _write_yaml(
            tmp_path / "filter.yaml",
            f"""
job_type: pe_rewrite_filter
quality_filter:
  min_scores:
    intent_fidelity: 5
dataset:
  input_path: {tmp_path / "scored.jsonl"}
  output_path: {tmp_path / "filtered.jsonl"}
""",
        )
        run_pe_rewrite_filter(config)

        kept = (tmp_path / "filtered.jsonl").read_text().splitlines()
        assert len(kept) == 1


class TestPERewriteBuildSFT:
    def test_language_selects_student_system_prompt(self, tmp_path):
        (tmp_path / "sys_zh.txt").write_text("学生中文改写指令", encoding="utf-8")
        (tmp_path / "sys_en.txt").write_text("Student EN instruction", encoding="utf-8")
        rows = [
            _scored_row("zh1", language="zh"),
            _scored_row("en1", language="en"),
        ]
        _write_jsonl(tmp_path / "filtered.jsonl", rows)
        config = _write_yaml(
            tmp_path / "sft.yaml",
            f"""
job_type: pe_rewrite_build_sft
sft:
  system_prompt_zh_file: {tmp_path / "sys_zh.txt"}
  system_prompt_en_file: {tmp_path / "sys_en.txt"}
dataset:
  input_path: {tmp_path / "filtered.jsonl"}
  output_path: {tmp_path / "sft.jsonl"}
""",
        )
        run_pe_rewrite_build_sft(config)

        samples = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
        assert len(samples) == 2
        by_id = {}
        for sample in samples:
            messages = sample["messages"]
            roles = [m["role"] for m in messages]
            assert roles == ["system", "user", "assistant"]
            by_id[messages[1]["content"]] = messages
        zh = by_id["原始 prompt zh1"]
        en = by_id["原始 prompt en1"]
        assert zh[0]["content"] == "学生中文改写指令"
        assert en[0]["content"] == "Student EN instruction"
        assert zh[2]["content"] == "改写后 prompt zh1"

    def test_judge_scores_and_trace_are_stripped_from_metadata(self, tmp_path):
        (tmp_path / "sys_zh.txt").write_text("SYS", encoding="utf-8")
        _write_jsonl(tmp_path / "filtered.jsonl", [_scored_row("zh1")])
        config = _write_yaml(
            tmp_path / "sft.yaml",
            f"""
job_type: pe_rewrite_build_sft
sft:
  system_prompt_zh_file: {tmp_path / "sys_zh.txt"}
dataset:
  input_path: {tmp_path / "filtered.jsonl"}
  output_path: {tmp_path / "sft.jsonl"}
""",
        )
        run_pe_rewrite_build_sft(config)

        sample = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
        dumped = json.dumps(sample, ensure_ascii=False)
        for leaked in ("intent_fidelity", "agent_trace", "no_conflict"):
            assert leaked not in dumped
