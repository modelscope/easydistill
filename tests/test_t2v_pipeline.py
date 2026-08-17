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

"""Unit tests for T2VDistillationPipeline."""

from typing import Any, List, Optional

import pytest

from easydistill.backends.t2v_base import T2VBackend
from easydistill.data.models import VideoGenerationResult
from easydistill.pipeline import T2VDistillationPipeline
from tests._fake_backend import FakeBackend, FakeVideoJudgeBackend

_FAST_EVAL = {"show_progress": False, "max_workers": 1, "call_retries": 0}


class FakeT2VBackend(T2VBackend):
    """Fake T2V backend for pipeline testing."""

    def __init__(self, video_urls: Optional[List[str]] = None):
        self._video_urls = video_urls or ["https://cdn.example.com/pipeline_video.mp4"]

    def generate_video(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1280*720",
        duration: Optional[float] = None,
        first_frame_image: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        return VideoGenerationResult(
            prompt=prompt,
            video_urls=list(self._video_urls),
            first_frame_image=first_frame_image,
            model=model_id or "fake-t2v",
            usage={"video_count": len(self._video_urls)},
        )

    def health_check(self) -> bool:
        return True


def _make_pipeline(
    backend=None,
    t2v_backend=None,
    stages=None,
    dataset_config=None,
    generation_config=None,
    sft_config=None,
    eval_config=None,
    eval_backend=None,
):
    """Build a T2VDistillationPipeline with sensible test defaults."""
    if backend is None:
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
    if t2v_backend is None:
        t2v_backend = FakeT2VBackend()
    if stages is None:
        stages = [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "t2v_generate", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "build_t2v_sft", "config": {}},
        ]
    if dataset_config is None:
        dataset_config = {"input_path": "dummy", "output_path": "dummy"}
    return T2VDistillationPipeline(
        backend=backend,
        t2v_backend=t2v_backend,
        pipeline_config=stages,
        dataset_config=dataset_config,
        generation_config=generation_config or {},
        sft_config=sft_config or {},
        eval_config=eval_config or {},
        eval_backend=eval_backend,
    )


class TestPipelineValidation:
    """Tests for pipeline configuration validation."""

    def test_last_stage_must_be_build_t2v_sft(self):
        """Test that the last stage must be build_t2v_sft."""
        stages = [
            {"stage": "prompt_optimize", "config": {}},
            {"stage": "build_sft", "config": {}},  # Wrong last stage.
        ]
        with pytest.raises(ValueError, match="build_t2v_sft"):
            _make_pipeline(stages=stages)

    def test_empty_pipeline_raises(self):
        """Test that an empty pipeline config raises ValueError."""
        with pytest.raises(ValueError, match="at least one stage"):
            _make_pipeline(stages=[])

    def test_default_eval_metrics(self):
        """Test that default eval metrics are set correctly."""
        pipeline = _make_pipeline()
        assert "prompt_consistency" in pipeline._default_eval_metrics
        assert "visual_quality" in pipeline._default_eval_metrics
        assert "subject_consistency" in pipeline._default_eval_metrics

    def test_t2v_backend_stored(self):
        """Test that the T2V backend is stored on the pipeline."""
        t2v_backend = FakeT2VBackend()
        pipeline = _make_pipeline(t2v_backend=t2v_backend)
        assert pipeline.t2v_backend is t2v_backend

    def test_eval_backend_falls_back_to_backend(self):
        """Test that eval_backend defaults to backend when not provided."""
        backend = FakeBackend(response_template="<score>7</score>")
        pipeline = _make_pipeline(backend=backend)
        assert pipeline.eval_backend is backend


class TestStageDispatch:
    """Tests for individual stage dispatch."""

    def test_prompt_optimize_stage(self):
        """Test prompt_optimize stage produces optimized prompts."""
        backend = FakeBackend(response_template="<answer>detailed: {}</answer>")
        pipeline = _make_pipeline(backend=backend)
        data = [{"id": "1", "prompt": "a cat walking"}]
        result = pipeline._dispatch_stage(
            "prompt_optimize", {"show_progress": False, "max_workers": 1}, data, []
        )
        assert len(result) == 1
        assert result[0]["raw_prompt"] == "a cat walking"
        assert "optimized_prompt" in result[0]

    def test_t2v_generate_stage(self):
        """Test t2v_generate stage produces video URLs."""
        pipeline = _make_pipeline()
        data = [{"id": "1", "optimized_prompt": "a cat walking on the moon"}]
        result = pipeline._dispatch_stage(
            "t2v_generate",
            {"show_progress": False, "max_workers": 1},
            data,
            [],
        )
        assert len(result) == 1
        assert "video_urls" in result[0]
        assert len(result[0]["video_urls"]) >= 1

    def test_t2v_eval_stage(self):
        """Test t2v_eval stage adds scores to rows."""
        eval_backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (3, 0.9)})
        pipeline = _make_pipeline(
            eval_backend=eval_backend,
            eval_config={"metrics": ["prompt_consistency"], **_FAST_EVAL},
        )
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat walking",
                "video_urls": ["https://cdn.example.com/v.mp4"],
            }
        ]
        result = pipeline._dispatch_stage(
            "t2v_eval",
            {},
            data,
            ["prompt_consistency"],
        )
        assert len(result) == 1
        assert result[0]["prompt_consistency"] == 3
        assert result[0]["prompt_consistency_confidence"] == 0.9

    def test_t2v_eval_uses_eval_backend(self):
        """Test that t2v_eval stage uses eval_backend, not backend."""
        backend = FakeBackend(response_template="not json")
        eval_backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (4, 0.9)})
        pipeline = _make_pipeline(
            backend=backend,
            eval_backend=eval_backend,
            eval_config={"metrics": ["prompt_consistency"], **_FAST_EVAL},
        )
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat walking",
                "video_urls": ["https://cdn.example.com/v.mp4"],
            }
        ]
        result = pipeline._dispatch_stage(
            "t2v_eval",
            {},
            data,
            ["prompt_consistency"],
        )
        # Score should come from eval_backend (the video judge), not backend.
        assert result[0]["prompt_consistency"] == 4

    def test_quality_filter_stage(self):
        """Test quality_filter stage filters by min scores."""
        pipeline = _make_pipeline()
        data = [
            {"id": "1", "motion_quality": 8, "optimized_prompt": "p", "video_urls": ["u"]},
            {"id": "2", "motion_quality": 3, "optimized_prompt": "p", "video_urls": ["u"]},
        ]
        result = pipeline._dispatch_stage(
            "quality_filter",
            {"min_scores": {"motion_quality": 5}},
            data,
            ["motion_quality"],
        )
        ids = {row["id"] for row in result}
        assert ids == {"1"}

    def test_build_t2v_sft_stage(self):
        """Test build_t2v_sft stage produces SFT samples."""
        pipeline = _make_pipeline()
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat walking",
                "video_urls": ["https://cdn.example.com/v.mp4"],
            }
        ]
        result = pipeline._dispatch_stage("build_t2v_sft", {}, data, [])
        assert len(result) == 1
        messages = result[0]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "video_url"

    def test_unknown_stage_raises(self):
        """Test that unknown stage raises ValueError."""
        pipeline = _make_pipeline()
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            pipeline._dispatch_stage("unknown_stage", {}, [{"id": "1"}], [])


class TestEndToEndSmoke:
    """End-to-end smoke test for the T2V pipeline."""

    def test_simple_pipeline_mixed_modes(self):
        """Test the full pipeline over mixed T2V and I2V rows."""
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        t2v_backend = FakeT2VBackend(
            video_urls=["https://cdn.example.com/e2e_video.mp4"]
        )
        pipeline = _make_pipeline(backend=backend, t2v_backend=t2v_backend)
        data = [
            {"id": "1", "prompt": "a cat walking on the moon"},
            {
                "id": "2",
                "prompt": "animate the boat",
                "first_frame_image": "https://cdn.example.com/frame.png",
            },
        ]
        result = pipeline.run_with_data(data)
        assert len(result) == 2
        by_id = {sample["metadata"]["request_id"]: sample for sample in result}
        # T2V row: plain text user message.
        t2v_messages = by_id["1"]["messages"]
        assert isinstance(t2v_messages[0]["content"], str)
        assert t2v_messages[1]["content"][0]["type"] == "video_url"
        assert by_id["1"]["metadata"]["t2v_mode"] == "t2v"
        # I2V row: multi-modal user message carrying the first frame.
        i2v_messages = by_id["2"]["messages"]
        assert isinstance(i2v_messages[0]["content"], list)
        assert by_id["2"]["metadata"]["t2v_mode"] == "i2v"

    def test_pipeline_with_eval(self):
        """Test pipeline with evaluation stage preserves scores in metadata."""
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        eval_backend = FakeVideoJudgeBackend(scores={"prompt_consistency": (4, 0.9)})
        stages = [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "t2v_generate", "config": {"show_progress": False, "max_workers": 1}},
            {
                "stage": "t2v_eval",
                "config": {},
            },
            {"stage": "quality_filter", "config": {"min_scores": {"prompt_consistency": 3}}},
            {"stage": "build_t2v_sft", "config": {}},
        ]
        pipeline = _make_pipeline(
            backend=backend,
            eval_backend=eval_backend,
            stages=stages,
            eval_config={"metrics": ["prompt_consistency"], **_FAST_EVAL},
        )
        data = [{"id": "1", "prompt": "a cat walking"}]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        assert result[0]["metadata"].get("prompt_consistency") == 4

    def test_empty_data_raises(self):
        """Test that run_with_data raises on empty data."""
        pipeline = _make_pipeline()
        with pytest.raises(ValueError, match="Input data is empty"):
            pipeline.run_with_data([])
