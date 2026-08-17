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

"""Unit tests for T2IDistillationPipeline."""

from typing import Any, List, Optional

import pytest

from easydistill.backends.t2i_base import T2IBackend
from easydistill.data.models import ImageGenerationResult
from easydistill.pipeline import T2IDistillationPipeline
from tests._fake_backend import FakeBackend


class FakeT2IBackend(T2IBackend):
    """Fake T2I backend for pipeline testing."""

    def __init__(self, image_urls: Optional[List[str]] = None):
        self._image_urls = image_urls or ["https://cdn.example.com/pipeline_img.png"]

    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            prompt=prompt,
            image_urls=self._image_urls[:n],
            model=model_id or "fake-t2i",
            usage={"image_count": n},
        )

    def health_check(self) -> bool:
        return True


def _make_pipeline(
    backend=None,
    t2i_backend=None,
    stages=None,
    dataset_config=None,
    generation_config=None,
    sft_config=None,
    eval_config=None,
    eval_backend=None,
):
    """Build a T2IDistillationPipeline with sensible test defaults."""
    if backend is None:
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
    if t2i_backend is None:
        t2i_backend = FakeT2IBackend()
    if stages is None:
        stages = [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "t2i_generate", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "build_t2i_sft", "config": {}},
        ]
    if dataset_config is None:
        dataset_config = {"input_path": "dummy", "output_path": "dummy"}
    return T2IDistillationPipeline(
        backend=backend,
        t2i_backend=t2i_backend,
        pipeline_config=stages,
        dataset_config=dataset_config,
        generation_config=generation_config or {},
        sft_config=sft_config or {},
        eval_config=eval_config or {},
        eval_backend=eval_backend,
    )


class TestPipelineValidation:
    """Tests for pipeline configuration validation."""

    def test_last_stage_must_be_build_t2i_sft(self):
        """Test that the last stage must be build_t2i_sft."""
        stages = [
            {"stage": "prompt_optimize", "config": {}},
            {"stage": "build_sft", "config": {}},  # Wrong last stage.
        ]
        with pytest.raises(ValueError, match="build_t2i_sft"):
            _make_pipeline(stages=stages)

    def test_empty_pipeline_raises(self):
        """Test that an empty pipeline config raises ValueError."""
        with pytest.raises(ValueError, match="at least one stage"):
            _make_pipeline(stages=[])

    def test_default_eval_metrics(self):
        """Test that default eval metrics are set correctly."""
        pipeline = _make_pipeline()
        assert "prompt_consistency" in pipeline._default_eval_metrics
        assert "aesthetic_quality" in pipeline._default_eval_metrics
        assert "detail_richness" in pipeline._default_eval_metrics
        assert "artifact_absence" in pipeline._default_eval_metrics

    def test_t2i_backend_stored(self):
        """Test that the T2I backend is stored on the pipeline."""
        t2i_backend = FakeT2IBackend()
        pipeline = _make_pipeline(t2i_backend=t2i_backend)
        assert pipeline.t2i_backend is t2i_backend

    def test_eval_backend_stored(self):
        """Test that eval_backend is stored when provided."""
        eval_backend = FakeBackend(response_template="<score>8</score>")
        pipeline = _make_pipeline(eval_backend=eval_backend)
        assert pipeline.eval_backend is eval_backend

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
        data = [{"id": "1", "prompt": "a cat"}]
        result = pipeline._dispatch_stage(
            "prompt_optimize", {"show_progress": False, "max_workers": 1}, data, []
        )
        assert len(result) == 1
        assert "raw_prompt" in result[0]
        assert "optimized_prompt" in result[0]
        assert result[0]["raw_prompt"] == "a cat"

    def test_t2i_generate_stage(self):
        """Test t2i_generate stage produces image URLs."""
        pipeline = _make_pipeline()
        data = [{"id": "1", "optimized_prompt": "a cat on the moon"}]
        result = pipeline._dispatch_stage(
            "t2i_generate",
            {"show_progress": False, "max_workers": 1},
            data,
            [],
        )
        assert len(result) == 1
        assert "image_urls" in result[0]
        assert len(result[0]["image_urls"]) >= 1

    def test_t2i_eval_stage(self):
        """Test t2i_eval stage adds scores to rows."""
        backend = FakeBackend(response_template="<score>8</score>")
        pipeline = _make_pipeline(
            backend=backend,
            eval_config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat",
                "image_urls": ["https://cdn.example.com/img.png"],
            }
        ]
        result = pipeline._dispatch_stage(
            "t2i_eval",
            {"show_progress": False, "max_workers": 1},
            data,
            ["prompt_consistency"],
        )
        assert len(result) == 1
        assert result[0]["prompt_consistency"] == 8

    def test_t2i_eval_uses_eval_backend(self):
        """Test that t2i_eval stage uses eval_backend, not backend."""
        # backend returns 3, eval_backend returns 9 — eval should get 9.
        backend = FakeBackend(response_template="<score>3</score>")
        eval_backend = FakeBackend(response_template="<score>9</score>")
        pipeline = _make_pipeline(
            backend=backend,
            eval_backend=eval_backend,
            eval_config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat",
                "image_urls": ["https://cdn.example.com/img.png"],
            }
        ]
        result = pipeline._dispatch_stage(
            "t2i_eval",
            {"show_progress": False, "max_workers": 1},
            data,
            ["prompt_consistency"],
        )
        assert len(result) == 1
        # Score should be 9 (from eval_backend), not 3 (from backend).
        assert result[0]["prompt_consistency"] == 9

    def test_quality_filter_stage(self):
        """Test quality_filter stage filters by min scores."""
        pipeline = _make_pipeline()
        data = [
            {"id": "1", "prompt_consistency": 8, "optimized_prompt": "p", "image_urls": ["u"]},
            {"id": "2", "prompt_consistency": 3, "optimized_prompt": "p", "image_urls": ["u"]},
        ]
        result = pipeline._dispatch_stage(
            "quality_filter",
            {"min_scores": {"prompt_consistency": 5}},
            data,
            ["prompt_consistency"],
        )
        ids = {row["id"] for row in result}
        assert "1" in ids
        assert "2" not in ids

    def test_build_t2i_sft_stage(self):
        """Test build_t2i_sft stage produces SFT samples."""
        pipeline = _make_pipeline()
        data = [
            {
                "id": "1",
                "optimized_prompt": "a cat",
                "image_urls": ["https://cdn.example.com/img.png"],
            }
        ]
        result = pipeline._dispatch_stage("build_t2i_sft", {}, data, [])
        assert len(result) == 1
        # SFT samples are dicts with "messages" key.
        assert "messages" in result[0]
        messages = result[0]["messages"]
        # User message is the prompt text.
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "a cat"
        # Assistant message is a multimodal list.
        assert messages[1]["role"] == "assistant"
        assert isinstance(messages[1]["content"], list)

    def test_unknown_stage_raises(self):
        """Test that unknown stage raises ValueError."""
        pipeline = _make_pipeline()
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            pipeline._dispatch_stage("unknown_stage", {}, [{"id": "1"}], [])


class TestEndToEndSmoke:
    """End-to-end smoke test for the T2I pipeline."""

    def test_simple_pipeline(self):
        """Test the full pipeline: prompt_optimize -> t2i_generate -> build_t2i_sft."""
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        t2i_backend = FakeT2IBackend(
            image_urls=["https://cdn.example.com/e2e_img.png"]
        )
        pipeline = _make_pipeline(
            backend=backend,
            t2i_backend=t2i_backend,
        )
        data = [
            {"id": "1", "prompt": "a cat on the moon"},
            {"id": "2", "prompt": "a dog in space"},
        ]
        result = pipeline.run_with_data(data)
        assert len(result) == 2
        for sample in result:
            assert "messages" in sample
            messages = sample["messages"]
            # User message contains the optimized prompt.
            assert messages[0]["role"] == "user"
            assert "enhanced:" in messages[0]["content"]
            # Assistant message is a multimodal list with image_url.
            assert messages[1]["role"] == "assistant"
            assert isinstance(messages[1]["content"], list)
            assert messages[1]["content"][0]["type"] == "image_url"
            assert messages[1]["content"][0]["image_url"]["url"] == "https://cdn.example.com/e2e_img.png"

    def test_pipeline_with_eval(self):
        """Test pipeline with evaluation stage."""
        backend = FakeBackend(response_template="<score>8</score>")
        t2i_backend = FakeT2IBackend()
        stages = [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "t2i_generate", "config": {"show_progress": False, "max_workers": 1}},
            {
                "stage": "t2i_eval",
                "config": {
                    "metrics": ["prompt_consistency"],
                    "show_progress": False,
                    "max_workers": 1,
                },
            },
            {"stage": "build_t2i_sft", "config": {}},
        ]
        pipeline = _make_pipeline(
            backend=backend,
            t2i_backend=t2i_backend,
            stages=stages,
            eval_config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        data = [{"id": "1", "prompt": "a cat"}]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        # The SFT sample should preserve eval scores in metadata.
        assert "messages" in result[0]
        metadata = result[0].get("metadata", {})
        assert metadata.get("prompt_consistency") == 8

    def test_pipeline_with_split_backends(self):
        """Test pipeline with separate backend (text) and eval_backend (VLM)."""
        # backend (text) returns optimized prompt, eval_backend (VLM) returns score.
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        eval_backend = FakeBackend(response_template="<score>9</score>")
        t2i_backend = FakeT2IBackend()
        stages = [
            {"stage": "prompt_optimize", "config": {"show_progress": False, "max_workers": 1}},
            {"stage": "t2i_generate", "config": {"show_progress": False, "max_workers": 1}},
            {
                "stage": "t2i_eval",
                "config": {
                    "metrics": ["prompt_consistency"],
                    "show_progress": False,
                    "max_workers": 1,
                },
            },
            {"stage": "build_t2i_sft", "config": {}},
        ]
        pipeline = _make_pipeline(
            backend=backend,
            eval_backend=eval_backend,
            t2i_backend=t2i_backend,
            stages=stages,
            eval_config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        data = [{"id": "1", "prompt": "a cat"}]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        metadata = result[0].get("metadata", {})
        # Score should be 9 (from eval_backend), not from backend.
        assert metadata.get("prompt_consistency") == 9

    def test_empty_data_raises(self):
        """Test that run_with_data raises on empty data."""
        pipeline = _make_pipeline()
        with pytest.raises(ValueError, match="Input data is empty"):
            pipeline.run_with_data([])
