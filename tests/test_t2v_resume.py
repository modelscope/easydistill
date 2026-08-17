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

"""Unit tests for T2V pipeline resume / row-level checkpointing."""

import json
from typing import Any, List, Optional

from easydistill.operators.t2v import T2VGenerationOperator
from easydistill.operators.t2v.resume import (
    RowCheckpointWriter,
    eval_row_complete,
    generate_row_complete,
    load_completed_rows,
    merge_resumed,
    optimize_row_complete,
    resume_key,
    split_pending,
)
from tests._fake_backend import FakeBackend
from tests.test_t2v_pipeline import FakeT2VBackend, _make_pipeline


class CountingT2VBackend(FakeT2VBackend):
    """Fake backend counting generate_video calls per prompt."""

    def __init__(self, video_urls: Optional[List[str]] = None):
        super().__init__(video_urls=video_urls)
        self.calls: List[str] = []
        self.kwargs_seen: List[dict] = []

    def generate_video(self, prompt: str, **kwargs: Any):
        self.calls.append(prompt)
        self.kwargs_seen.append(kwargs)
        return super().generate_video(prompt, **kwargs)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestResumeHelpers:
    """Tests for the low-level resume helper functions."""

    def test_resume_key_prefers_id(self):
        assert resume_key({"id": "7", "prompt": "x"}) == "id:7"
        # int and str ids collapse to the same key.
        assert resume_key({"id": 7}) == "id:7"

    def test_resume_key_falls_back_to_content_hash(self):
        row_a = {"prompt": "a cat", "first_frame_image": "img.jpg"}
        row_b = {"prompt": "a cat", "first_frame_image": "img.jpg"}
        row_c = {"prompt": "a cat"}
        assert resume_key(row_a) == resume_key(row_b)
        assert resume_key(row_a) != resume_key(row_c)

    def test_completion_predicates(self):
        assert optimize_row_complete({"optimized_prompt": "p"})
        assert not optimize_row_complete({"optimized_prompt": ""})
        assert generate_row_complete({"video_urls": ["v.mp4"]})
        assert not generate_row_complete({"video_urls": []})
        assert eval_row_complete({"visual_quality_confidence": 0.9})
        assert eval_row_complete({"vbench_motion_smoothness": 0.5})
        # None-valued metric columns mean the row was skipped, not scored.
        assert not eval_row_complete({"visual_quality": None, "visual_quality_confidence": None})
        # VBench skip reason alone must not make a row look complete.
        assert not eval_row_complete(
            {
                "vbench_skipped_reason": "vbench not installed",
                "vbench_motion_smoothness": None,
                "vbench_aesthetic_quality": None,
            }
        )

    def test_load_completed_rows_filters_and_dedupes(self, tmp_path):
        path = tmp_path / "out.jsonl"
        _write_jsonl(
            path,
            [
                {"id": "1", "video_urls": ["old.mp4"]},
                {"id": "2", "video_urls": []},  # incomplete -> retried
                {"id": "1", "video_urls": ["new.mp4"]},  # duplicate keeps last
            ],
        )
        # Simulate a torn tail line from a crashed append.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"id": "3", "video_ur')
        completed = load_completed_rows(str(path), generate_row_complete)
        assert set(completed) == {"id:1"}
        assert completed["id:1"]["video_urls"] == ["new.mp4"]

    def test_load_completed_rows_missing_file(self, tmp_path):
        assert load_completed_rows(str(tmp_path / "none.jsonl"), generate_row_complete) == {}

    def test_split_and_merge_preserve_input_order(self):
        data = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        completed = {"id:2": {"id": "2", "video_urls": ["cached.mp4"]}}
        done, pending = split_pending(data, completed)
        assert [r["id"] for r in done] == ["2"]
        assert [r["id"] for r in pending] == ["1", "3"]
        new_rows = [{"id": "3", "video_urls": ["fresh.mp4"]}]  # id 1 dropped (failed)
        merged = merge_resumed(data, completed, new_rows)
        assert [r["id"] for r in merged] == ["2", "3"]
        assert merged[0]["video_urls"] == ["cached.mp4"]

    def test_checkpoint_writer_appends(self, tmp_path):
        path = tmp_path / "sub" / "ckpt.jsonl"
        writer = RowCheckpointWriter(str(path))
        writer.append({"id": "1"})
        writer.append({"id": "2", "prompt": "中文"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["prompt"] == "中文"


class TestOperatorCheckpoint:
    """Tests for per-row checkpointing inside T2VGenerationOperator."""

    def test_each_completed_row_is_appended(self, tmp_path):
        ckpt = tmp_path / "gen.jsonl"
        backend = CountingT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "checkpoint_path": str(ckpt)},
        )
        rows = [
            {"id": "1", "optimized_prompt": "cat"},
            {"id": "2", "optimized_prompt": "dog"},
        ]
        results = operator.run(rows)
        assert len(results) == 2
        lines = [json.loads(line) for line in ckpt.read_text().strip().splitlines()]
        assert [r["id"] for r in lines] == ["1", "2"]
        assert all(r["video_urls"] for r in lines)

    def test_checkpoint_survives_midway_failure(self, tmp_path):
        ckpt = tmp_path / "gen.jsonl"

        class ExplodingBackend(CountingT2VBackend):
            def generate_video(self, prompt: str, **kwargs: Any):
                if prompt == "boom":
                    raise RuntimeError("hard failure")
                return super().generate_video(prompt, **kwargs)

        operator = T2VGenerationOperator(
            backend=ExplodingBackend(),
            config={
                "show_progress": False,
                "checkpoint_path": str(ckpt),
                "retry_attempts": 0,
            },
        )
        rows = [
            {"id": "1", "optimized_prompt": "cat"},
            {"id": "2", "optimized_prompt": "boom"},
            {"id": "3", "optimized_prompt": "dog"},
        ]
        results = operator.run(rows)
        assert [r["id"] for r in results] == ["1", "3"]
        lines = [json.loads(line) for line in ckpt.read_text().strip().splitlines()]
        assert [r["id"] for r in lines] == ["1", "3"]

    def test_checkpoint_keys_not_forwarded_to_backend(self, tmp_path):
        backend = CountingT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "checkpoint_path": str(tmp_path / "gen.jsonl"),
                "resume": True,
            },
        )
        operator.run([{"id": "1", "optimized_prompt": "cat"}])
        assert "checkpoint_path" not in backend.kwargs_seen[0]
        assert "resume" not in backend.kwargs_seen[0]

    def test_concurrent_mode_checkpoints(self, tmp_path):
        ckpt = tmp_path / "gen.jsonl"
        operator = T2VGenerationOperator(
            backend=CountingT2VBackend(),
            config={
                "show_progress": False,
                "checkpoint_path": str(ckpt),
                "max_workers": 2,
            },
        )
        rows = [{"id": str(i), "optimized_prompt": f"p{i}"} for i in range(4)]
        results = operator.run(rows)
        assert len(results) == 4
        lines = [json.loads(line) for line in ckpt.read_text().strip().splitlines()]
        assert {r["id"] for r in lines} == {"0", "1", "2", "3"}


class TestPipelineResume:
    """Tests for stage-level resume in T2VDistillationPipeline."""

    @staticmethod
    def _stages(tmp_path, resume=True):
        return [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {
                "stage": "t2v_generate",
                "config": {"show_progress": False, "max_workers": 1, "resume": resume},
                "output_path": str(tmp_path / "stage2_generated.jsonl"),
            },
            {"stage": "build_t2v_sft", "config": {}},
        ]

    @staticmethod
    def _seed_rows():
        return [
            {"id": "1", "prompt": "a cat walking"},
            {"id": "2", "prompt": "a dog running"},
        ]

    @staticmethod
    def _cached_generated_row(row_id="1"):
        return {
            "id": row_id,
            "prompt": "a cat walking",
            "optimized_prompt": "cached optimized prompt",
            "video_urls": ["https://cdn.example.com/cached.mp4"],
            "t2v_model": "cached-model",
            "t2v_mode": "t2v",
        }

    def test_resume_skips_completed_rows(self, tmp_path):
        stages = self._stages(tmp_path)
        _write_jsonl(stages[1]["output_path"], [self._cached_generated_row("1")])
        backend = CountingT2VBackend()
        pipeline = _make_pipeline(t2v_backend=backend, stages=stages)
        result = pipeline.run_with_data(self._seed_rows())
        # Only the missing row hit the backend.
        assert len(backend.calls) == 1
        assert len(result) == 2

    def test_resume_reuses_cached_row_content(self, tmp_path):
        stages = self._stages(tmp_path)
        # Keep only prompt_optimize + t2v_generate outputs visible: inspect
        # the stage output written after the resumed generate stage.
        _write_jsonl(stages[1]["output_path"], [self._cached_generated_row("1")])
        pipeline = _make_pipeline(t2v_backend=CountingT2VBackend(), stages=stages)
        pipeline.run_with_data(self._seed_rows())
        rows = _read_jsonl(stages[1]["output_path"])
        by_id = {r["id"]: r for r in rows if "video_urls" in r}
        assert by_id["1"]["video_urls"] == ["https://cdn.example.com/cached.mp4"]
        assert by_id["2"]["video_urls"] == ["https://cdn.example.com/pipeline_video.mp4"]

    def test_resume_retries_incomplete_rows(self, tmp_path):
        stages = self._stages(tmp_path)
        incomplete = self._cached_generated_row("1")
        incomplete["video_urls"] = []
        _write_jsonl(stages[1]["output_path"], [incomplete])
        backend = CountingT2VBackend()
        pipeline = _make_pipeline(t2v_backend=backend, stages=stages)
        pipeline.run_with_data(self._seed_rows())
        # The failed row is retried alongside the missing one.
        assert len(backend.calls) == 2

    def test_resume_disabled_runs_all_rows(self, tmp_path):
        stages = self._stages(tmp_path, resume=False)
        _write_jsonl(stages[1]["output_path"], [self._cached_generated_row("1")])
        backend = CountingT2VBackend()
        pipeline = _make_pipeline(t2v_backend=backend, stages=stages)
        pipeline.run_with_data(self._seed_rows())
        assert len(backend.calls) == 2

    def test_resume_without_output_path_runs_full(self, tmp_path):
        stages = self._stages(tmp_path)
        stages[1].pop("output_path")
        backend = CountingT2VBackend()
        pipeline = _make_pipeline(t2v_backend=backend, stages=stages)
        pipeline.run_with_data(self._seed_rows())
        assert len(backend.calls) == 2

    def test_resume_all_rows_completed_skips_stage(self, tmp_path):
        stages = self._stages(tmp_path)
        _write_jsonl(
            stages[1]["output_path"],
            [self._cached_generated_row("1"), self._cached_generated_row("2")],
        )
        backend = CountingT2VBackend()
        pipeline = _make_pipeline(t2v_backend=backend, stages=stages)
        result = pipeline.run_with_data(self._seed_rows())
        assert backend.calls == []
        assert len(result) == 2

    def test_prompt_optimize_resume(self, tmp_path):
        opt_path = tmp_path / "stage1_optimized.jsonl"
        stages = [
            {
                "stage": "prompt_optimize",
                "config": {"show_progress": False, "max_workers": 1, "resume": True},
                "output_path": str(opt_path),
            },
            {"stage": "t2v_generate", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "build_t2v_sft", "config": {}},
        ]
        _write_jsonl(
            opt_path,
            [{"id": "1", "prompt": "a cat walking", "optimized_prompt": "cached prompt"}],
        )
        backend = FakeBackend(response_template="<answer>fresh: {}</answer>")
        pipeline = _make_pipeline(backend=backend, stages=stages)
        pipeline.run_with_data(self._seed_rows())
        rows = _read_jsonl(opt_path)
        by_id = {r["id"]: r for r in rows}
        assert by_id["1"]["optimized_prompt"] == "cached prompt"
        assert by_id["2"]["optimized_prompt"] != "cached prompt"
