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

"""Unit tests for the composable T2V video checkers."""

from typing import Any, Dict, List

import pytest

from easydistill.eval import (
    BaseVideoChecker,
    OmniChecker,
    T2VVideoEvaluator,
    VBenchChecker,
    VLMChecker,
    build_video_checkers,
)
from easydistill.eval.t2v_checkers import CHECKER_REGISTRY
from tests._fake_backend import FakeVideoJudgeBackend

_SAMPLE_ROWS = [
    {
        "id": "1",
        "optimized_prompt": "a cat walking",
        "video_urls": ["https://cdn.example.com/v1.mp4"],
    },
    {
        "id": "2",
        "optimized_prompt": "animate the boat",
        "video_urls": ["https://cdn.example.com/v2.mp4"],
        "first_frame_image": "https://cdn.example.com/frame.png",
    },
]

_FAST = {"show_progress": False, "max_workers": 1, "call_retries": 0}


class TestBuildVideoCheckers:
    """Tests for the checker registry and builder."""

    def test_builds_configured_checkers_in_order(self):
        backend = FakeVideoJudgeBackend()
        checkers = build_video_checkers(
            [
                {"type": "vbench", "dimensions": ["motion_smoothness"]},
                {"type": "vlm", "metrics": ["prompt_consistency"], **_FAST},
            ],
            backend=backend,
        )
        assert [c.name for c in checkers] == ["vbench", "vlm"]

    def test_disabled_checker_is_skipped(self):
        backend = FakeVideoJudgeBackend()
        checkers = build_video_checkers(
            [
                {"type": "vbench", "enabled": False},
                {"type": "vlm", "metrics": ["prompt_consistency"], **_FAST},
            ],
            backend=backend,
        )
        assert [c.name for c in checkers] == ["vlm"]

    def test_unknown_checker_type_raises(self):
        with pytest.raises(ValueError, match="Unknown video checker type"):
            build_video_checkers([{"type": "nonexistent"}], backend=None)

    def test_registry_contains_builtin_checkers(self):
        assert set(CHECKER_REGISTRY) == {"vbench", "vlm", "omni"}

    def test_backend_required_checker_without_backend_raises(self):
        with pytest.raises(ValueError, match="requires a backend"):
            build_video_checkers(
                [{"type": "vlm", "metrics": ["prompt_consistency"]}],
                backend=None,
            )


class TestVBenchChecker:
    """Tests for the VBench objective checker."""

    def test_default_dimensions(self):
        checker = VBenchChecker()
        assert checker.metrics == [
            "vbench_motion_smoothness",
            "vbench_dynamic_degree",
            "vbench_imaging_quality",
            "vbench_temporal_flickering",
        ]

    def test_custom_dimensions_prefixed(self):
        checker = VBenchChecker(config={"dimensions": ["subject_consistency"]})
        assert checker.metrics == ["vbench_subject_consistency"]

    def test_missing_repo_skips_and_records_reason(self):
        """Without any VBench install configured the skip reason lands in the rows."""
        checker = VBenchChecker(
            config={"dimensions": ["motion_smoothness"], "vbench_repo": None}
        )
        rows = checker.check([dict(r) for r in _SAMPLE_ROWS])
        assert len(rows) == 2
        for row in rows:
            assert row["vbench_motion_smoothness"] is None
            assert "neither vbench_bin nor vbench_repo" in row["vbench_skipped_reason"]

    def test_bad_vbench_bin_skips_and_records_reason(self, tmp_path):
        """A non-executable vbench_bin is reported in the skip reason."""
        checker = VBenchChecker(
            config={
                "dimensions": ["motion_smoothness"],
                "vbench_bin": str(tmp_path / "missing" / "vbench"),
            }
        )
        rows = checker.check([dict(_SAMPLE_ROWS[0])])
        assert "not an executable file" in rows[0]["vbench_skipped_reason"]

    def test_cli_mode_command(self, tmp_path):
        """vbench_bin builds a `vbench evaluate` command and wins over repo mode."""
        bin_path = tmp_path / "venv" / "bin" / "vbench"
        bin_path.parent.mkdir(parents=True)
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        checker = VBenchChecker(
            config={
                "dimensions": ["temporal_flickering"],
                "vbench_bin": str(bin_path),
                "vbench_repo": str(tmp_path),  # present but must be ignored
                "require_gpu": False,
            }
        )
        assert checker._environment_issue() is None
        cmd = checker._build_command("/videos", "/out")
        assert cmd[:2] == [str(bin_path), "evaluate"]
        assert "evaluate.py" not in " ".join(cmd)
        assert "--ngpus" not in cmd  # single GPU: no flag

    def test_cli_mode_multi_gpu_uses_ngpus(self, tmp_path):
        bin_path = tmp_path / "vbench"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        checker = VBenchChecker(
            config={"vbench_bin": str(bin_path), "num_gpus": 2}
        )
        cmd = checker._build_command("/videos", "/out")
        assert cmd[:4] == [str(bin_path), "evaluate", "--ngpus", "2"]
        assert "torchrun" not in cmd

    def test_cli_mode_prepends_venv_bin_to_path(self, tmp_path, monkeypatch):
        """The pip CLI spawns a bare `python`; PATH must resolve it to the venv."""
        import subprocess as subprocess_module

        bin_dir = tmp_path / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        bin_path = bin_dir / "vbench"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        video = tmp_path / "gen.mp4"
        video.write_bytes(b"fake-video")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            completed = type("P", (), {})()
            completed.returncode = 0
            completed.stderr = ""
            completed.stdout = ""
            return completed

        monkeypatch.setattr(subprocess_module, "run", fake_run)
        checker = VBenchChecker(
            config={
                "dimensions": ["temporal_flickering"],
                "vbench_bin": str(bin_path),
                "require_gpu": False,
            }
        )
        rows = checker.check(
            [{"id": "1", "optimized_prompt": "a cat", "video_urls": [str(video)]}]
        )
        assert captured["env"]["PATH"].startswith(str(bin_dir))
        # No results were produced -> the skip reason must be recorded, not
        # silently blanked (the CLI wrapper swallows inner torchrun failures).
        assert "no per-video results" in rows[0]["vbench_skipped_reason"]
        assert rows[0]["vbench_temporal_flickering"] is None

    def test_missing_gpu_skips_and_records_reason(self, tmp_path, monkeypatch):
        """require_gpu (default) skips when nvidia-smi is absent."""
        import shutil as shutil_module

        repo = tmp_path / "VBench"
        repo.mkdir()
        (repo / "evaluate.py").write_text("# fake\n")
        monkeypatch.setattr(shutil_module, "which", lambda name: None)
        checker = VBenchChecker(
            config={"dimensions": ["motion_smoothness"], "vbench_repo": str(repo)}
        )
        rows = checker.check([dict(_SAMPLE_ROWS[0])])
        assert rows[0]["vbench_motion_smoothness"] is None
        assert "no GPU detected" in rows[0]["vbench_skipped_reason"]

    def test_parse_per_video_canonical_and_legacy(self, tmp_path):
        from easydistill.eval.t2v_checkers import _parse_vbench_per_video

        (tmp_path / "results_x_eval_results.json").write_text(
            '{"motion_smoothness": [0.9, ['
            '{"video_path": "/tmp/video_00000.mp4", "video_results": 0.95},'
            '{"video_path": "/tmp/video_00001.mp4", "video_results": 0.85}]]}'
        )
        (tmp_path / "dynamic_degree.json").write_text(
            '{"dynamic_degree": {"results": {"video_00000.mp4": true, "mean": 0.5}}}'
        )
        scores = _parse_vbench_per_video(tmp_path)
        assert scores["motion_smoothness"]["video_00000"] == 0.95
        assert scores["motion_smoothness"]["video_00001"] == 0.85
        assert scores["dynamic_degree"]["video_00000"] == 1.0

    def test_check_with_mocked_subprocess(self, tmp_path, monkeypatch):
        """Full flow: staging local videos, running evaluate.py, merging scores."""
        import subprocess as subprocess_module

        # Fake repo with an evaluate.py so the config check passes.
        repo = tmp_path / "VBench"
        repo.mkdir()
        (repo / "evaluate.py").write_text("# fake\n")
        # A local "video" file for row 0; row 1 keeps a remote URL.
        video = tmp_path / "gen.mp4"
        video.write_bytes(b"fake-video")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            output_dir = cmd[cmd.index("--output_path") + 1]
            videos_dir = cmd[cmd.index("--videos_path") + 1]
            staged = sorted(p.stem for p in __import__("pathlib").Path(videos_dir).iterdir())
            captured["staged"] = staged
            result_file = __import__("pathlib").Path(output_dir) / "r_eval_results.json"
            result_file.write_text(
                '{"motion_smoothness": [0.9, ['
                f'{{"video_path": "{staged[0]}.mp4", "video_results": 0.91}}]]}}'
            )
            completed = type("P", (), {})()
            completed.returncode = 0
            completed.stderr = ""
            return completed

        monkeypatch.setattr(subprocess_module, "run", fake_run)

        checker = VBenchChecker(
            config={
                "dimensions": ["motion_smoothness"],
                "vbench_repo": str(repo),
                "require_gpu": False,  # the mocked subprocess needs no GPU
            }
        )
        rows = [
            {"id": "1", "optimized_prompt": "a cat", "video_urls": [str(video)]},
            dict(_SAMPLE_ROWS[1]),  # remote URL -> skipped, blank column
        ]
        merged = checker.check(rows)
        assert merged[0]["vbench_motion_smoothness"] == 0.91
        assert merged[0]["vbench_skipped_reason"] is None
        assert merged[1]["vbench_motion_smoothness"] is None
        # Dimensions were passed through to the CLI.
        assert "motion_smoothness" in captured["cmd"]
        assert "--mode" in captured["cmd"]


class TestVLMChecker:
    """Tests for the structured multi-dimension VLM judge."""

    def test_one_call_scores_all_dimensions(self):
        """A single call per row fills every configured dimension column."""
        backend = FakeVideoJudgeBackend(
            scores={"prompt_consistency": (4, 0.95), "visual_quality": (2, 0.7)}
        )
        checker = VLMChecker(
            config={"metrics": ["prompt_consistency", "visual_quality"], **_FAST},
            backend=backend,
        )
        rows = checker.check([dict(_SAMPLE_ROWS[0])])
        assert backend.call_count == 1  # one call, both dimensions
        row = rows[0]
        assert row["prompt_consistency"] == 4
        assert row["prompt_consistency_confidence"] == 0.95
        assert row["prompt_consistency_reason"]
        assert row["visual_quality"] == 2
        # Original fields are preserved.
        assert "video_urls" in row

    def test_metrics_default_to_dimension_pool(self):
        backend = FakeVideoJudgeBackend()
        checker = VLMChecker(config=_FAST, backend=backend)
        assert checker.metrics == [
            "prompt_consistency",
            "visual_quality",
            "subject_consistency",
            "first_frame_consistency",
        ]

    def test_unknown_metric_raises(self):
        backend = FakeVideoJudgeBackend()
        with pytest.raises(ValueError, match="Unknown vlm judge dimensions"):
            VLMChecker(config={"metrics": ["nonexistent_dim"], **_FAST}, backend=backend)

    def test_i2v_only_dimension_not_applicable_for_t2v(self):
        """T2V rows get first_frame_consistency = None (judge marks N/A)."""
        backend = FakeVideoJudgeBackend()
        checker = VLMChecker(config=_FAST, backend=backend)
        rows = checker.check([dict(_SAMPLE_ROWS[0])])  # no first frame
        assert rows[0]["first_frame_consistency"] is None
        assert rows[0]["prompt_consistency"] == 3  # default score

    def test_i2v_row_prepends_first_frame_and_scores_consistency(self):
        backend = FakeVideoJudgeBackend(scores={"first_frame_consistency": (4, 0.9)})
        checker = VLMChecker(config=_FAST, backend=backend)
        rows = checker.check([dict(_SAMPLE_ROWS[1])])  # has first frame
        assert rows[0]["first_frame_consistency"] == 4
        # Conditioning frame + fallback video reference = 2 images.
        assert backend.last_image_count == 2
        assert "conditioning first frame" in backend.last_prompt

    def test_remote_video_falls_back_to_raw_reference(self):
        """Remote URLs skip frame sampling and pass the reference through."""
        backend = FakeVideoJudgeBackend()
        checker = VLMChecker(config=_FAST, backend=backend)
        checker.check([dict(_SAMPLE_ROWS[0])])
        assert "Raw video reference" in backend.last_prompt

    def test_pre_extracted_frames_used_directly(self):
        backend = FakeVideoJudgeBackend()
        checker = VLMChecker(config=_FAST, backend=backend)
        row = {
            **_SAMPLE_ROWS[0],
            "frame_urls": [
                "https://cdn.example.com/f1.jpg",
                "https://cdn.example.com/f2.jpg",
            ],
        }
        checker.check([row])
        assert backend.last_image_count == 2
        assert "pre-extracted frames" in backend.last_prompt

    def test_row_failure_is_isolated(self):
        """A row whose judge call keeps failing gets blank metric columns."""

        class ExplodingBackend(FakeVideoJudgeBackend):
            def generate(self, *args, **kwargs):
                raise ConnectionError("boom")

        checker = VLMChecker(
            config={"metrics": ["prompt_consistency"], **_FAST, "retry_delay_sec": 0.0},
            backend=ExplodingBackend(),
        )
        rows = checker.check([dict(r) for r in _SAMPLE_ROWS])
        assert len(rows) == 2
        assert all(row["prompt_consistency"] is None for row in rows)


class TestOmniChecker:
    """Tests for the Omni full-video judge."""

    def test_default_metrics_from_dimension_pool(self):
        backend = FakeVideoJudgeBackend()
        checker = OmniChecker(backend=backend, config=_FAST)
        assert checker.metrics == [
            "motion_quality",
            "temporal_execution",
            "camera_accuracy",
        ]

    def test_remote_url_transport(self):
        """Rows with remote URLs send the URL as a video_url content item."""
        backend = FakeVideoJudgeBackend(scores={"motion_quality": (4, 0.9)})
        checker = OmniChecker(
            backend=backend, config={"metrics": ["motion_quality"], **_FAST}
        )
        rows = checker.check([dict(_SAMPLE_ROWS[0])])  # https video_urls
        assert rows[0]["motion_quality"] == 4
        assert rows[0]["motion_quality_confidence"] == 0.9
        assert backend.last_video_count == 1
        assert backend.last_video_url == "https://cdn.example.com/v1.mp4"

    def test_prefers_row_remote_url_over_local(self, tmp_path):
        backend = FakeVideoJudgeBackend()
        checker = OmniChecker(
            backend=backend, config={"metrics": ["motion_quality"], **_FAST}
        )
        video = tmp_path / "v.mp4"
        video.write_bytes(b"vid")
        row = {
            "id": "1",
            "optimized_prompt": "a cat",
            "video_urls": [str(video)],
            "video_remote_urls": ["https://cdn.example.com/remote.mp4"],
        }
        checker.check([row])
        assert backend.last_video_url == "https://cdn.example.com/remote.mp4"

    def test_base64_transport_for_local_video(self, tmp_path):
        backend = FakeVideoJudgeBackend()
        checker = OmniChecker(
            backend=backend, config={"metrics": ["motion_quality"], **_FAST}
        )
        video = tmp_path / "v.mp4"
        video.write_bytes(b"vid-bytes")
        row = {"id": "1", "optimized_prompt": "a cat", "video_urls": [str(video)]}
        checker.check([row])
        assert backend.last_video_url.startswith("data:video/mp4;base64,")

    def test_no_video_reference_blanks_row(self):
        backend = FakeVideoJudgeBackend()
        checker = OmniChecker(
            backend=backend,
            config={"metrics": ["motion_quality"], **_FAST, "retry_delay_sec": 0.0},
        )
        rows = checker.check([{"id": "1", "optimized_prompt": "a cat", "video_urls": []}])
        assert rows[0]["motion_quality"] is None


class TestT2VVideoEvaluatorComposition:
    """Tests for the checker-composition orchestrator."""

    def test_backcompat_shortcut_builds_vlm(self):
        """Top-level metrics config (no `checkers`) builds one vlm checker."""
        backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (3, 0.9)})
        evaluator = T2VVideoEvaluator(
            backend=backend,
            config={"metrics": ["prompt_consistency"], **_FAST},
        )
        assert [c.name for c in evaluator.checkers] == ["vlm"]
        assert evaluator.metrics == ["prompt_consistency"]
        results = evaluator.run([dict(r) for r in _SAMPLE_ROWS])
        assert all(r["prompt_consistency"] == 3 for r in results)

    def test_composed_chain_merges_all_metric_columns(self):
        backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (4, 0.9)})
        evaluator = T2VVideoEvaluator(
            backend=backend,
            config={
                "checkers": [
                    {"type": "vbench", "dimensions": ["motion_smoothness"]},
                    {"type": "vlm", "metrics": ["prompt_consistency"], **_FAST},
                ]
            },
        )
        assert evaluator.metrics == ["vbench_motion_smoothness", "prompt_consistency"]
        results = evaluator.run([dict(r) for r in _SAMPLE_ROWS])
        for row in results:
            assert row["vbench_motion_smoothness"] is None  # placeholder
            assert row["prompt_consistency"] == 4

    def test_failing_checker_blanks_only_its_metrics(self):
        """A crashing checker must not kill the batch or later checkers."""

        class ExplodingChecker(BaseVideoChecker):
            name = "exploding"

            @property
            def metrics(self) -> List[str]:
                return ["exploding_score"]

            def check(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                raise RuntimeError("boom")

        backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (2, 0.9)})
        evaluator = T2VVideoEvaluator(
            backend=backend,
            config={"metrics": ["prompt_consistency"], **_FAST},
        )
        evaluator.checkers.insert(0, ExplodingChecker())
        results = evaluator.run([dict(r) for r in _SAMPLE_ROWS])
        for row in results:
            assert row["exploding_score"] is None
            assert row["prompt_consistency"] == 2

    def test_no_enabled_checkers_raises(self):
        backend = FakeVideoJudgeBackend()
        with pytest.raises(ValueError, match="no enabled checkers"):
            T2VVideoEvaluator(
                backend=backend,
                config={"checkers": [{"type": "vbench", "enabled": False}]},
            )

    def test_aggregate(self):
        backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (3, 0.9)})
        evaluator = T2VVideoEvaluator(
            backend=backend,
            config={"metrics": ["prompt_consistency"], **_FAST},
        )
        results = evaluator.run([dict(r) for r in _SAMPLE_ROWS])
        aggregates = evaluator.aggregate(results)
        assert aggregates["prompt_consistency"] == 3
