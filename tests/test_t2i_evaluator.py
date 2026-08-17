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

"""Unit tests for T2IImageEvaluator."""


from easydistill.eval import T2IImageEvaluator
from tests._fake_backend import FakeBackend


class TestT2IImageEvaluatorExtract:
    """Tests for _extract_sample and _extract_images."""

    def test_extract_sample_with_optimized_prompt(self):
        """Test that optimized_prompt is used as the instruction."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {
            "id": "img1",
            "optimized_prompt": "a cat on the moon",
            "image_urls": ["https://cdn.example.com/img.png"],
        }
        sample_id, instruction, output = evaluator._extract_sample(sample)
        assert sample_id == "img1"
        assert instruction == "a cat on the moon"
        assert output == "https://cdn.example.com/img.png"

    def test_extract_sample_fallback_to_prompt(self):
        """Test that 'prompt' is used when optimized_prompt is missing."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {
            "id": "img2",
            "prompt": "a dog in space",
            "image_urls": ["https://cdn.example.com/dog.png"],
        }
        sample_id, instruction, output = evaluator._extract_sample(sample)
        assert sample_id == "img2"
        assert instruction == "a dog in space"
        assert output == "https://cdn.example.com/dog.png"

    def test_extract_sample_no_images(self):
        """Test that empty output is returned when no image_urls."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {"id": "img3", "optimized_prompt": "prompt", "image_urls": []}
        sample_id, instruction, output = evaluator._extract_sample(sample)
        assert output == ""

    def test_extract_images(self):
        """Test that _extract_images returns the first image URL."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {
            "id": "1",
            "optimized_prompt": "prompt",
            "image_urls": [
                "https://cdn.example.com/img1.png",
                "https://cdn.example.com/img2.png",
            ],
        }
        images = evaluator._extract_images(sample)
        assert len(images) == 1
        assert images[0] == "https://cdn.example.com/img1.png"

    def test_extract_images_string(self):
        """Test that a single string image_url is handled."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {
            "id": "1",
            "optimized_prompt": "prompt",
            "image_urls": "https://cdn.example.com/single.png",
        }
        images = evaluator._extract_images(sample)
        assert len(images) == 1
        assert images[0] == "https://cdn.example.com/single.png"

    def test_extract_images_empty(self):
        """Test that no images are returned for empty image_urls."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"metrics": ["prompt_consistency"], "show_progress": False},
        )
        sample = {"id": "1", "optimized_prompt": "prompt", "image_urls": []}
        images = evaluator._extract_images(sample)
        assert images == []


class TestT2IImageEvaluatorRun:
    """Tests for the full evaluator run."""

    def test_run_with_score(self):
        """Test that scores are extracted from the judge response."""
        backend = FakeBackend(response_template="<score>8</score>")
        evaluator = T2IImageEvaluator(
            backend=backend,
            config={
                "metrics": ["prompt_consistency"],
                "show_progress": False,
                "max_workers": 1,
            },
        )
        samples = [
            {
                "id": "1",
                "optimized_prompt": "a cat",
                "image_urls": ["https://cdn.example.com/cat.png"],
            }
        ]
        results = evaluator.run(samples)
        assert len(results) == 1
        assert results[0]["prompt_consistency"] == 8

    def test_run_multiple_metrics(self):
        """Test evaluation with multiple metrics."""
        backend = FakeBackend(response_template="<score>7</score>")
        evaluator = T2IImageEvaluator(
            backend=backend,
            config={
                "metrics": ["prompt_consistency", "aesthetic_quality"],
                "show_progress": False,
                "max_workers": 1,
            },
        )
        samples = [
            {
                "id": "1",
                "optimized_prompt": "a dog",
                "image_urls": ["https://cdn.example.com/dog.png"],
            }
        ]
        results = evaluator.run(samples)
        assert len(results) == 1
        assert results[0]["prompt_consistency"] == 7
        assert results[0]["aesthetic_quality"] == 7

    def test_run_skips_no_image_samples(self):
        """Test that samples without images are skipped."""
        backend = FakeBackend(response_template="<score>9</score>")
        evaluator = T2IImageEvaluator(
            backend=backend,
            config={
                "metrics": ["prompt_consistency"],
                "show_progress": False,
                "max_workers": 1,
            },
        )
        samples = [
            {"id": "1", "optimized_prompt": "prompt", "image_urls": []},
        ]
        results = evaluator.run(samples)
        assert len(results) == 0

    def test_run_multiple_samples(self):
        """Test evaluation with multiple samples."""
        backend = FakeBackend(response_template="<score>6</score>")
        evaluator = T2IImageEvaluator(
            backend=backend,
            config={
                "metrics": ["prompt_consistency"],
                "show_progress": False,
                "max_workers": 1,
            },
        )
        samples = [
            {"id": "1", "optimized_prompt": "cat", "image_urls": ["https://x.com/1.png"]},
            {"id": "2", "optimized_prompt": "dog", "image_urls": ["https://x.com/2.png"]},
        ]
        results = evaluator.run(samples)
        assert len(results) == 2
        for r in results:
            assert r["prompt_consistency"] == 6

    def test_aggregate(self):
        """Test that aggregate computes average scores."""
        backend = FakeBackend(response_template="<score>8</score>")
        evaluator = T2IImageEvaluator(
            backend=backend,
            config={
                "metrics": ["prompt_consistency"],
                "show_progress": False,
                "max_workers": 1,
            },
        )
        samples = [
            {"id": "1", "optimized_prompt": "a", "image_urls": ["https://x.com/1.png"]},
            {"id": "2", "optimized_prompt": "b", "image_urls": ["https://x.com/2.png"]},
        ]
        results = evaluator.run(samples)
        agg = evaluator.aggregate(results)
        assert agg["prompt_consistency"] == 8.0

    def test_default_metrics(self):
        """Test that default metrics are loaded from DEFAULT_PROMPTS."""
        evaluator = T2IImageEvaluator(
            backend=FakeBackend(),
            config={"show_progress": False},
        )
        assert "prompt_consistency" in evaluator.metrics
        assert "aesthetic_quality" in evaluator.metrics
        assert "detail_richness" in evaluator.metrics
        assert "artifact_absence" in evaluator.metrics
