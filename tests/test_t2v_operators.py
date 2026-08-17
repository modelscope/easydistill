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

"""Unit tests for T2V operators."""

from typing import Any, List, Optional

import pytest

from easydistill.backends.t2v_base import T2VBackend
from easydistill.data.models import (
    GenerationRequest,
    GenerationResult,
    SFTSample,
    VideoGenerationResult,
)
from easydistill.operators.t2v import (
    T2VGenerationOperator,
    T2VPromptOptimizer,
    T2VSFTBuilder,
)
from easydistill.operators.t2v.prompt_optimizer import _draft_aspect_ratio
from easydistill.operators.t2v.t2v_generation import _image_dimensions
from tests._fake_backend import FakeBackend


class FakeT2VBackend(T2VBackend):
    """Fake T2V backend for testing T2V operators."""

    def __init__(
        self,
        video_urls: Optional[List[str]] = None,
        fail_count: int = 0,
        exc_class: type = ConnectionError,
    ):
        self._video_urls = video_urls or ["https://cdn.example.com/video.mp4"]
        self._fail_count = fail_count
        self._call_count = 0
        self._exc_class = exc_class
        self.last_call_kwargs: dict = {}

    def generate_video(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        size: str = "1280*720",
        duration: Optional[float] = None,
        first_frame_image: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        self._call_count += 1
        self.last_call_kwargs = {
            "prompt": prompt,
            "model_id": model_id,
            "size": size,
            "duration": duration,
            "first_frame_image": first_frame_image,
            **kwargs,
        }
        if self._call_count <= self._fail_count:
            raise self._exc_class(f"Transient error (call {self._call_count})")
        return VideoGenerationResult(
            prompt=prompt,
            video_urls=list(self._video_urls),
            first_frame_image=first_frame_image,
            model=model_id or ("fake-i2v" if first_frame_image else "fake-t2v"),
            usage={"video_count": len(self._video_urls)},
        )

    def health_check(self) -> bool:
        return True


class TestT2VPromptOptimizer:
    """Tests for the two-stage (extract -> compose) T2VPromptOptimizer."""

    def test_extract_stage_build_requests(self):
        """Stage 1 reads the prompt and formats the extract template."""
        backend = FakeBackend(response_template="<answer>draft json</answer>")
        optimizer = T2VPromptOptimizer(backend=backend, config={"show_progress": False})
        rows = [
            {"id": "1", "prompt": "a cat walking"},
            {"id": "2", "prompt": "a flying dog"},
        ]
        requests = optimizer.extract_stage._build_requests(rows)
        assert len(requests) == 2
        assert requests[0].id == "1"
        assert "a cat walking" in requests[0].instruction
        assert requests[0].metadata["raw_prompt"] == "a cat walking"

    def test_extract_stage_i2v_attaches_first_frame(self):
        """I2V rows produce multi-modal extract requests with the first frame."""
        backend = FakeBackend(response_template="<answer>draft json</answer>")
        optimizer = T2VPromptOptimizer(backend=backend, config={"show_progress": False})
        rows = [
            {
                "id": "1",
                "prompt": "make it move",
                "first_frame_image": "https://cdn.example.com/frame.png",
            }
        ]
        requests = optimizer.extract_stage._build_requests(rows)
        assert len(requests) == 1
        assert isinstance(requests[0].instruction, list)
        image_items = [
            item for item in requests[0].instruction if item.get("type") == "image_url"
        ]
        assert len(image_items) == 1
        assert requests[0].metadata["first_frame_image"] == "https://cdn.example.com/frame.png"
        # The I2V-specific template (first-frame inventory -> evolution) is used.
        text = " ".join(
            item.get("text", "")
            for item in requests[0].instruction
            if item.get("type") == "text"
        )
        assert "first_frame_inventory" in text
        assert "IMAGE-TO-VIDEO" in text

    def test_extract_stage_t2v_uses_generic_template(self):
        """Plain T2V rows must NOT use the I2V inventory template."""
        backend = FakeBackend(response_template="<answer>draft json</answer>")
        optimizer = T2VPromptOptimizer(backend=backend, config={"show_progress": False})
        requests = optimizer.extract_stage._build_requests(
            [{"id": "1", "prompt": "a cat walking"}]
        )
        assert isinstance(requests[0].instruction, str)
        assert "first_frame_inventory" not in requests[0].instruction

    def test_compose_stage_injects_schema_and_draft(self):
        """Stage 2 formats the compose template with schema + draft + prompt."""
        backend = FakeBackend(response_template="<answer>final caption</answer>")
        optimizer = T2VPromptOptimizer(
            backend=backend,
            config={"show_progress": False, "schema": "CUSTOM SCHEMA RULES"},
        )
        rows = [{"id": "1", "raw_prompt": "a cat walking", "draft": '{"subject": "cat"}'}]
        requests = optimizer.compose_stage._build_requests(rows)
        assert len(requests) == 1
        instruction = requests[0].instruction
        assert "CUSTOM SCHEMA RULES" in instruction
        assert '{"subject": "cat"}' in instruction
        assert "a cat walking" in instruction

    def test_compose_stage_default_schema_built_in(self):
        """Without config, the schema falls back to the built-in generic one."""
        backend = FakeBackend(response_template="<answer>final caption</answer>")
        optimizer = T2VPromptOptimizer(backend=backend, config={"show_progress": False})
        assert "caption" in optimizer.compose_stage.schema.lower()

    def test_run_two_stage_end_to_end(self):
        """Full run produces draft + optimized_prompt with two model calls."""
        backend = FakeBackend(response_template="<answer>stage output</answer>")
        optimizer = T2VPromptOptimizer(
            backend=backend, config={"show_progress": False, "max_workers": 1}
        )
        rows = [
            {"id": "1", "prompt": "sunset timelapse"},
            {
                "id": "2",
                "prompt": "animate the boat",
                "first_frame_image": "https://cdn.example.com/frame.png",
            },
        ]
        results = optimizer.run(rows)
        assert len(results) == 2
        by_id = {r["id"]: r for r in results}
        for r in results:
            assert r["raw_prompt"]
            assert r["draft"] == "stage output"
            assert r["optimized_prompt"] == "stage output"
        # I2V row keeps its first frame through both stages.
        assert by_id["2"]["first_frame_image"] == "https://cdn.example.com/frame.png"
        assert "first_frame_image" not in by_id["1"]

    def test_compose_backend_split(self):
        """A separate compose backend is used for stage 2 when provided."""
        extract_backend = FakeBackend(response_template="<answer>the draft</answer>")
        compose_backend = FakeBackend(response_template="<answer>the caption</answer>")
        optimizer = T2VPromptOptimizer(
            backend=extract_backend,
            config={"show_progress": False},
            compose_backend=compose_backend,
        )
        rows = [{"id": "1", "prompt": "a cat walking"}]
        results = optimizer.run(rows)
        assert len(results) == 1
        assert results[0]["draft"] == "the draft"
        assert results[0]["optimized_prompt"] == "the caption"


class TestT2VGenerationOperator:
    """Tests for T2VGenerationOperator."""

    def test_run_basic_t2v(self):
        """Test basic T2V video generation."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [{"id": "1", "optimized_prompt": "a cat walking on the moon"}]
        results = operator.run(rows)
        assert len(results) == 1
        assert results[0]["video_urls"] == ["https://cdn.example.com/video.mp4"]
        assert results[0]["t2v_mode"] == "t2v"
        assert backend.last_call_kwargs["first_frame_image"] is None

    def test_run_i2v_passes_first_frame(self):
        """Test that I2V rows forward the first frame to the backend."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [
            {
                "id": "1",
                "optimized_prompt": "animate the boat",
                "first_frame_image": "https://cdn.example.com/frame.png",
            }
        ]
        results = operator.run(rows)
        assert len(results) == 1
        assert results[0]["t2v_mode"] == "i2v"
        assert (
            backend.last_call_kwargs["first_frame_image"]
            == "https://cdn.example.com/frame.png"
        )

    def test_run_skips_empty_prompt(self):
        """Test that rows with empty prompts are skipped."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [{"id": "1", "optimized_prompt": ""}]
        results = operator.run(rows)
        assert len(results) == 0

    def test_retry_on_transient_error(self):
        """Test that the operator retries on transient errors."""
        backend = FakeT2VBackend(fail_count=2, exc_class=ConnectionError)
        operator = T2VGenerationOperator(
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
        assert backend._call_count == 3  # 2 failures + 1 success

    def test_retry_exhausted(self):
        """Test that the operator gives up after max retries."""
        backend = FakeT2VBackend(fail_count=99, exc_class=ConnectionError)
        operator = T2VGenerationOperator(
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
        assert len(results) == 0

    def test_concurrent_run(self):
        """Test concurrent generation with ThreadPoolExecutor."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 3},
        )
        rows = [{"id": str(i), "optimized_prompt": f"prompt {i}"} for i in range(5)]
        results = operator.run(rows)
        assert len(results) == 5

    def test_row_level_resolution_override(self):
        """T2V rows may override the configured resolution per row."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "resolution": "720P",
                "ratio": "16:9",
            },
        )
        rows = [{"id": "1", "optimized_prompt": "p", "resolution": "1080P"}]
        operator.run(rows)
        assert backend.last_call_kwargs["resolution"] == "1080P"
        assert backend.last_call_kwargs["ratio"] == "16:9"

    def test_row_level_ratio_override(self):
        """T2V rows may override the configured ratio per row."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1, "ratio": "16:9"},
        )
        rows = [{"id": "1", "optimized_prompt": "p", "ratio": "9:16"}]
        operator.run(rows)
        assert backend.last_call_kwargs["ratio"] == "9:16"

    def test_auto_resolution_keeps_configured_tier(self):
        """`resolution: auto` keeps the configured tier and sends no override."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1, "resolution": "720P"},
        )
        rows = [{"id": "1", "optimized_prompt": "p", "resolution": "auto"}]
        operator.run(rows)
        assert backend.last_call_kwargs["resolution"] == "720P"

    def test_i2v_drops_framing_knobs(self):
        """I2V rows never send resolution / ratio / size to the backend."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "resolution": "720P",
                "ratio": "16:9",
                "size": "1280*720",
            },
        )
        rows = [
            {
                "id": "1",
                "optimized_prompt": "p",
                "first_frame_image": "https://cdn.example.com/frame.png",
            }
        ]
        operator.run(rows)
        kwargs = backend.last_call_kwargs
        assert "resolution" not in kwargs
        assert "ratio" not in kwargs
        assert kwargs["size"] is None
        assert kwargs["first_frame_image"] == "https://cdn.example.com/frame.png"

    def test_i2v_ignores_row_level_resolution(self):
        """Per-row resolution on an I2V row is ignored, not sent."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [
            {
                "id": "1",
                "optimized_prompt": "p",
                "resolution": "1080P",
                "first_frame_image": "https://cdn.example.com/frame.png",
            }
        ]
        operator.run(rows)
        assert "resolution" not in backend.last_call_kwargs
        assert "ratio" not in backend.last_call_kwargs

    def test_i2v_frame_check_skip_small_frame(self, tmp_path):
        """i2v_frame_check=skip drops rows whose first frame is too small."""
        import cv2
        import numpy as np

        frame = tmp_path / "small.png"
        cv2.imwrite(str(frame), np.full((100, 100, 3), 128, dtype=np.uint8))
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "i2v_frame_check": "skip",
                "i2v_frame_min_edge": 256,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "p", "first_frame_image": str(frame)}]
        results = operator.run(rows)
        assert results == []
        assert backend._call_count == 0

    def test_i2v_frame_check_raise_small_frame(self, tmp_path):
        """i2v_frame_check=raise raises on a degenerate first frame."""
        import cv2
        import numpy as np

        frame = tmp_path / "wide.png"
        cv2.imwrite(str(frame), np.full((100, 500, 3), 128, dtype=np.uint8))
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "i2v_frame_check": "raise",
                "i2v_frame_max_aspect": 2.0,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "p", "first_frame_image": str(frame)}]
        with pytest.raises(ValueError, match="aspect ratio"):
            operator.run(rows)

    def test_i2v_frame_check_warn_still_generates(self, tmp_path):
        """i2v_frame_check=warn logs the issue but still generates."""
        import cv2
        import numpy as np

        frame = tmp_path / "small.png"
        cv2.imwrite(str(frame), np.full((100, 100, 3), 128, dtype=np.uint8))
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "i2v_frame_check": "warn",
                "i2v_frame_min_edge": 256,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "p", "first_frame_image": str(frame)}]
        results = operator.run(rows)
        assert len(results) == 1
        assert backend._call_count == 1

    def test_i2v_frame_check_remote_frame_passes(self):
        """http(s) first frames cannot be checked locally and pass through."""
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "i2v_frame_check": "skip",
                "i2v_frame_min_edge": 256,
            },
        )
        rows = [
            {
                "id": "1",
                "optimized_prompt": "p",
                "first_frame_image": "https://cdn.example.com/frame.png",
            }
        ]
        results = operator.run(rows)
        assert len(results) == 1

    def test_i2v_frame_check_ok_large_frame(self, tmp_path):
        """A healthy first frame passes the size check."""
        import cv2
        import numpy as np

        frame = tmp_path / "ok.png"
        cv2.imwrite(str(frame), np.full((300, 400, 3), 128, dtype=np.uint8))
        backend = FakeT2VBackend()
        operator = T2VGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 1,
                "i2v_frame_check": "skip",
                "i2v_frame_min_edge": 256,
            },
        )
        rows = [{"id": "1", "optimized_prompt": "p", "first_frame_image": str(frame)}]
        results = operator.run(rows)
        assert len(results) == 1

    def test_invalid_i2v_frame_check(self):
        """Unknown i2v_frame_check values fail fast at construction."""
        with pytest.raises(ValueError, match="i2v_frame_check"):
            T2VGenerationOperator(
                backend=FakeT2VBackend(),
                config={"i2v_frame_check": "explode"},
            )


class TestImageDimensions:
    """Tests for the module-level first-frame dimension probe."""

    def test_local_path(self, tmp_path):
        import cv2
        import numpy as np

        frame = tmp_path / "f.png"
        cv2.imwrite(str(frame), np.full((480, 640, 3), 128, dtype=np.uint8))
        assert _image_dimensions(str(frame)) == (640, 480)

    def test_file_url(self, tmp_path):
        import cv2
        import numpy as np

        frame = tmp_path / "f.png"
        cv2.imwrite(str(frame), np.full((240, 320, 3), 128, dtype=np.uint8))
        assert _image_dimensions(f"file://{frame}") == (320, 240)

    def test_data_url(self):
        import base64

        import cv2
        import numpy as np

        ok, buf = cv2.imencode(".png", np.full((240, 320, 3), 128, dtype=np.uint8))
        assert ok
        data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        assert _image_dimensions(data_url) == (320, 240)

    def test_http_returns_none(self):
        assert _image_dimensions("https://cdn.example.com/frame.png") is None

    def test_missing_file_returns_none(self):
        assert _image_dimensions("/nonexistent/frame.png") is None

    def test_invalid_data_url_returns_none(self):
        assert _image_dimensions("data:image/png;base64,!!!") is None
        assert _image_dimensions("data:text/plain,hello") is None

    def test_non_string_returns_none(self):
        assert _image_dimensions("") is None
        assert _image_dimensions(None) is None


class TestT2VExtractAutoResolution:
    """Tests for `resolution: auto` -> LLM-inferred aspect ratio on T2V rows."""

    @staticmethod
    def _parse(draft: str, **meta: Any):
        stage = T2VPromptOptimizer(
            backend=FakeBackend(), config={"show_progress": False}
        ).extract_stage
        request = GenerationRequest(
            id="1", instruction="x", metadata={"raw_prompt": "p", **meta}
        )
        return stage._parse_result(
            GenerationResult(request=request, response=f"<answer>{draft}</answer>")
        )

    def test_draft_aspect_ratio_valid(self):
        assert _draft_aspect_ratio('{"aspect_ratio": "16:9"}') == "16:9"
        assert _draft_aspect_ratio('{"aspect_ratio": " 9 : 16 "}') == "9:16"
        assert _draft_aspect_ratio('text {"aspect_ratio": "1:1"} more') == "1:1"

    def test_draft_aspect_ratio_invalid(self):
        assert _draft_aspect_ratio("no json here") is None
        assert _draft_aspect_ratio('{"broken": ') is None
        assert _draft_aspect_ratio('{"aspect_ratio": "ultrawide"}') is None
        assert _draft_aspect_ratio('{"aspect_ratio": ""}') is None
        assert _draft_aspect_ratio("") is None

    def test_auto_resolves_ratio_for_t2v(self):
        output = self._parse('{"aspect_ratio": "9:16"}', resolution="auto")
        assert output["ratio"] == "9:16"
        assert output["resolution"] == "auto"

    def test_auto_ignored_for_i2v(self):
        output = self._parse(
            '{"aspect_ratio": "9:16"}',
            resolution="auto",
            first_frame_image="data:image/png;base64,AAAA",
        )
        assert "ratio" not in output

    def test_explicit_ratio_wins(self):
        output = self._parse('{"aspect_ratio": "9:16"}', resolution="auto", ratio="1:1")
        assert output["ratio"] == "1:1"

    def test_auto_without_draft_ratio_falls_back(self):
        output = self._parse("plain text", resolution="auto")
        assert "ratio" not in output

    def test_no_auto_no_ratio(self):
        output = self._parse('{"aspect_ratio": "9:16"}', resolution="720P")
        assert "ratio" not in output


class TestT2VSFTBuilder:
    """Tests for T2VSFTBuilder."""

    def test_build_basic_t2v(self):
        """Test basic T2V SFT sample construction."""
        builder = T2VSFTBuilder(config={})
        rows = [
            {
                "id": "1",
                "optimized_prompt": "a cat walking on the moon",
                "video_urls": ["https://cdn.example.com/v1.mp4"],
            }
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, SFTSample)
        # User message is the prompt text.
        assert sample.messages[0].role == "user"
        assert sample.messages[0].content == "a cat walking on the moon"
        # Assistant message is a multimodal content list with the video.
        assert sample.messages[1].role == "assistant"
        assert isinstance(sample.messages[1].content, list)
        assert sample.messages[1].content[0]["type"] == "video_url"
        assert sample.messages[1].content[0]["video_url"]["url"] == "https://cdn.example.com/v1.mp4"

    def test_build_i2v_user_content(self):
        """Test that I2V samples embed the first frame in the user message."""
        builder = T2VSFTBuilder(config={})
        rows = [
            {
                "id": "1",
                "optimized_prompt": "animate the boat",
                "first_frame_image": "https://cdn.example.com/frame.png",
                "video_urls": ["https://cdn.example.com/v1.mp4"],
                "t2v_mode": "i2v",
            }
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        user_content = samples[0].messages[0].content
        assert isinstance(user_content, list)
        types = [item["type"] for item in user_content]
        assert "text" in types
        assert "image_url" in types
        assert samples[0].metadata["t2v_mode"] == "i2v"

    def test_skip_empty_videos(self):
        """Test that rows with no videos are skipped."""
        builder = T2VSFTBuilder(config={"skip_empty": True})
        rows = [{"id": "1", "optimized_prompt": "prompt", "video_urls": []}]
        samples = builder.run(rows)
        assert len(samples) == 0

    def test_preserves_eval_scores(self):
        """Test that evaluation scores are preserved in metadata."""
        builder = T2VSFTBuilder(config={})
        rows = [
            {
                "id": "1",
                "optimized_prompt": "prompt",
                "video_urls": ["https://x.com/v.mp4"],
                "prompt_consistency": 4,
                "subject_consistency": 3,
            }
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].metadata["prompt_consistency"] == 4
        assert samples[0].metadata["subject_consistency"] == 3

    def test_system_prompt(self):
        """Test that system prompt is included when configured."""
        builder = T2VSFTBuilder(config={"system_prompt": "You are a video generator."})
        rows = [
            {"id": "1", "optimized_prompt": "prompt", "video_urls": ["https://x.com/v.mp4"]},
        ]
        samples = builder.run(rows)
        assert len(samples) == 1
        assert samples[0].messages[0].role == "system"
