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

"""Unit tests for T2I operators."""

from typing import Any, List, Optional

from easydistill.backends.t2i_base import T2IBackend
from easydistill.data.models import ImageGenerationResult, SFTSample
from easydistill.operators.t2i import (
    T2IGenerationOperator,
    T2IPromptOptimizer,
    T2ISFTBuilder,
)
from tests._fake_backend import FakeBackend


class FakeT2IBackend(T2IBackend):
    """Fake T2I backend for testing T2I operators."""

    def __init__(
        self,
        image_urls: Optional[List[str]] = None,
        fail_count: int = 0,
        exc_class: type = ConnectionError,
    ):
        self._image_urls = image_urls or ["https://cdn.example.com/img.png"]
        self._fail_count = fail_count
        self._call_count = 0
        self._exc_class = exc_class

    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._exc_class(f"Transient error (call {self._call_count})")
        return ImageGenerationResult(
            prompt=prompt,
            image_urls=self._image_urls[:n],
            model=model_id or "fake-t2i",
            usage={"image_count": n},
        )

    def health_check(self) -> bool:
        return True


class TestT2IPromptOptimizer:
    """Tests for T2IPromptOptimizer."""

    def test_build_requests(self):
        """Test that _build_requests reads prompt from input rows."""
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        optimizer = T2IPromptOptimizer(backend=backend, config={"show_progress": False})
        rows = [
            {"id": "1", "prompt": "a cat"},
            {"id": "2", "prompt": "a dog"},
        ]
        requests = optimizer._build_requests(rows)
        assert len(requests) == 2
        assert requests[0].id == "1"
        assert "a cat" in requests[0].instruction
        assert requests[0].metadata["raw_prompt"] == "a cat"

    def test_parse_result(self):
        """Test that _parse_result extracts <answer> tag."""
        backend = FakeBackend(response_template="<answer>enhanced: {}</answer>")
        optimizer = T2IPromptOptimizer(backend=backend, config={"show_progress": False})
        rows = [{"id": "1", "prompt": "a cat"}]
        results = optimizer.run(rows)
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["raw_prompt"] == "a cat"
        # The FakeBackend formats the template with the full instruction text,
        # so the optimized_prompt starts with "enhanced: " and contains the seed prompt.
        assert results[0]["optimized_prompt"].startswith("enhanced: ")
        assert "a cat" in results[0]["optimized_prompt"]

    def test_parse_result_no_answer_tag(self):
        """Test fallback when no <answer> tag is present."""
        # Use a template without {} so the response doesn't include the
        # instruction text (which itself contains <answer>...</answer>).
        backend = FakeBackend(response_template="plain optimized prompt without tags")
        optimizer = T2IPromptOptimizer(backend=backend, config={"show_progress": False})
        rows = [{"id": "1", "prompt": "a cat"}]
        results = optimizer.run(rows)
        assert len(results) == 1
        # Fallback: the stripped response is used as the optimized prompt.
        assert results[0]["optimized_prompt"] == "plain optimized prompt without tags"

    def test_run_with_multiple_rows(self):
        """Test running the optimizer on multiple rows."""
        backend = FakeBackend(response_template="<answer>detailed: {}</answer>")
        optimizer = T2IPromptOptimizer(
            backend=backend, config={"show_progress": False, "max_workers": 1}
        )
        rows = [
            {"id": "1", "prompt": "sunset"},
            {"id": "2", "prompt": "mountain"},
            {"id": "3", "prompt": "ocean"},
        ]
        results = optimizer.run(rows)
        assert len(results) == 3
        for r in results:
            assert "raw_prompt" in r
            assert "optimized_prompt" in r
            assert r["optimized_prompt"].startswith("detailed:")


class TestT2IGenerationOperator:
    """Tests for T2IGenerationOperator."""

    def test_run_basic(self):
        """Test basic image generation."""
        backend = FakeT2IBackend()
        operator = T2IGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [{"id": "1", "optimized_prompt": "a cat on the moon"}]
        results = operator.run(rows)
        assert len(results) == 1
        assert "image_urls" in results[0]
        assert len(results[0]["image_urls"]) == 1
        assert results[0]["image_urls"][0] == "https://cdn.example.com/img.png"

    def test_run_fallback_prompt_key(self):
        """Test that the operator falls back to 'prompt' key."""
        backend = FakeT2IBackend()
        operator = T2IGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [{"id": "1", "prompt": "a dog in space"}]
        results = operator.run(rows)
        assert len(results) == 1
        assert len(results[0]["image_urls"]) == 1

    def test_run_skips_empty_prompt(self):
        """Test that rows with empty prompts are skipped."""
        backend = FakeT2IBackend()
        operator = T2IGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [{"id": "1", "optimized_prompt": ""}]
        results = operator.run(rows)
        assert len(results) == 0

    def test_retry_on_transient_error(self):
        """Test that the operator retries on transient errors."""
        backend = FakeT2IBackend(fail_count=2, exc_class=ConnectionError)
        operator = T2IGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "retry_attempts": 3,
                "retry_backoff_base": 0.01,
                "retry_max_wait": 0.1,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "retry test"}]
        results = operator.run(rows)
        assert len(results) == 1
        assert len(results[0]["image_urls"]) == 1
        assert backend._call_count == 3  # 2 failures + 1 success

    def test_retry_exhausted(self):
        """Test that the operator gives up after max retries."""
        backend = FakeT2IBackend(fail_count=99, exc_class=ConnectionError)
        operator = T2IGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "retry_attempts": 2,
                "retry_backoff_base": 0.01,
                "retry_max_wait": 0.1,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "will fail"}]
        results = operator.run(rows)
        assert len(results) == 0  # All retries exhausted, row skipped.

    def test_concurrent_run(self):
        """Test concurrent generation with ThreadPoolExecutor."""
        backend = FakeT2IBackend()
        operator = T2IGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 3},
        )
        rows = [
            {"id": str(i), "optimized_prompt": f"prompt {i}"} for i in range(5)
        ]
        results = operator.run(rows)
        assert len(results) == 5
        for r in results:
            assert len(r["image_urls"]) == 1


class TestT2ISFTBuilder:
    """Tests for T2ISFTBuilder."""

    def test_build_basic(self):
        """Test basic SFT sample construction."""
        builder = T2ISFTBuilder(config={})
        rows = [
            {
                "id": "1",
                "optimized_prompt": "a cat on the moon",
                "image_urls": ["https://cdn.example.com/img1.png"],
            }
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, SFTSample)
        # User message is the prompt text.
        assert sample.messages[0].role == "user"
        assert sample.messages[0].content == "a cat on the moon"
        # Assistant message is a multimodal content list.
        assert sample.messages[1].role == "assistant"
        assert isinstance(sample.messages[1].content, list)
        assert sample.messages[1].content[0]["type"] == "image_url"
        assert "url" in sample.messages[1].content[0]["image_url"]

    def test_skip_empty_images(self):
        """Test that rows with no images are skipped."""
        builder = T2ISFTBuilder(config={"skip_empty": True})
        rows = [
            {"id": "1", "optimized_prompt": "prompt", "image_urls": []},
        ]
        samples = builder.run(rows)
        assert len(samples) == 0

    def test_skip_empty_prompt(self):
        """Test that rows with empty prompts are skipped."""
        builder = T2ISFTBuilder(config={"skip_empty": True})
        rows = [
            {"id": "1", "optimized_prompt": "", "image_urls": ["https://x.com/img.png"]},
        ]
        samples = builder.run(rows)
        assert len(samples) == 0

    def test_min_prompt_length(self):
        """Test that short prompts are filtered."""
        builder = T2ISFTBuilder(config={"min_prompt_length": 10})
        rows = [
            {"id": "1", "optimized_prompt": "short", "image_urls": ["https://x.com/i.png"]},
            {"id": "2", "optimized_prompt": "this is a long prompt", "image_urls": ["https://x.com/i.png"]},
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].messages[0].content == "this is a long prompt"

    def test_preserves_eval_scores(self):
        """Test that evaluation scores are preserved in metadata."""
        builder = T2ISFTBuilder(config={})
        rows = [
            {
                "id": "1",
                "optimized_prompt": "prompt",
                "image_urls": ["https://x.com/i.png"],
                "prompt_consistency": 8,
                "aesthetic_quality": 7,
            }
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].metadata["prompt_consistency"] == 8
        assert samples[0].metadata["aesthetic_quality"] == 7

    def test_fallback_prompt_key(self):
        """Test that the builder falls back to 'prompt' key."""
        builder = T2ISFTBuilder(config={})
        rows = [
            {"id": "1", "prompt": "fallback prompt", "image_urls": ["https://x.com/i.png"]},
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].messages[0].content == "fallback prompt"

    def test_system_prompt(self):
        """Test that system prompt is included when configured."""
        builder = T2ISFTBuilder(config={"system_prompt": "You are an image generator."})
        rows = [
            {"id": "1", "optimized_prompt": "prompt", "image_urls": ["https://x.com/i.png"]},
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].messages[0].role == "system"
        assert samples[0].messages[0].content == "You are an image generator."
