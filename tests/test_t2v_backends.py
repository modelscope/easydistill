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

"""Unit tests for T2V backends (httpx-based, mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from easydistill.backends import PaiTokenVideoBackend, PAIVideoBackend, T2VBackend

_PAI_TOKEN_HTTPX = "easydistill.backends.pai_token_video_backend.httpx.Client"
_PAI_VIDEO_HTTPX = "easydistill.backends.pai_video_backend.httpx.Client"


def _mock_client(post_payloads=None, get_payloads=None):
    """Build a MagicMock httpx.Client with scripted post/get JSON payloads."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    def _responses(payloads):
        responses = []
        for payload in payloads or []:
            resp = MagicMock()
            resp.json.return_value = payload
            resp.raise_for_status = MagicMock()
            responses.append(resp)
        return responses

    post_responses = _responses(post_payloads)
    if post_responses:
        client.post.side_effect = post_responses
    get_responses = _responses(get_payloads)
    if get_responses:
        client.get.side_effect = get_responses
    return client


class TestPaiTokenVideoBackend:
    """Tests for the PAI-Token gateway video backend."""

    def _backend(self, **kwargs):
        return PaiTokenVideoBackend(
            api_key="test-key",
            base_url="https://gw.example.com/api/v1",
            poll_interval=0.01,
            max_poll_wait=5.0,
            **kwargs,
        )

    def test_t2v_submit_and_poll(self):
        """T2V flow: async submit -> poll -> SUCCEEDED with video_url."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"output": {"task_id": "t-1", "task_status": "PENDING"}}],
            get_payloads=[
                {"output": {"task_id": "t-1", "task_status": "RUNNING"}},
                {
                    "output": {
                        "task_id": "t-1",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                },
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            result = backend.generate_video(prompt="a cat walking", size="1280*720")

        assert result.video_urls == ["https://cdn.example.com/v.mp4"]
        assert result.model == "happyhorse-1.1-t2v"  # T2V default model
        assert result.metadata["task_id"] == "t-1"
        # Submit payload follows the DashScope-style envelope.
        submit_kwargs = client.post.call_args.kwargs
        assert submit_kwargs["json"]["input"]["prompt"] == "a cat walking"
        assert "img_url" not in submit_kwargs["json"]["input"]
        assert submit_kwargs["json"]["parameters"]["size"] == "1280*720"
        assert submit_kwargs["headers"]["X-DashScope-Async"] == "enable"

    def test_i2v_switches_model_and_sends_media(self):
        """I2V flow: first_frame_image switches the default model and payload."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            result = backend.generate_video(
                prompt="animate the boat",
                first_frame_image="https://cdn.example.com/frame.png",
            )

        assert result.model == "happyhorse-1.1-i2v"  # I2V default model
        assert result.first_frame_image == "https://cdn.example.com/frame.png"
        submit_kwargs = client.post.call_args.kwargs
        # Current DashScope protocol carries the frame in `input.media`.
        assert submit_kwargs["json"]["input"]["media"] == [
            {"type": "first_frame", "url": "https://cdn.example.com/frame.png"}
        ]
        assert "img_url" not in submit_kwargs["json"]["input"]

    def test_i2v_legacy_img_url_field(self):
        """i2v_image_field='img_url' keeps the legacy wan-style payload."""
        backend = self._backend(i2v_image_field="img_url")
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            backend.generate_video(
                prompt="animate the boat",
                first_frame_image="https://cdn.example.com/frame.png",
            )

        submit_kwargs = client.post.call_args.kwargs
        assert (
            submit_kwargs["json"]["input"]["img_url"]
            == "https://cdn.example.com/frame.png"
        )
        assert "media" not in submit_kwargs["json"]["input"]

    def test_invalid_i2v_image_field_raises(self):
        with pytest.raises(ValueError, match="i2v_image_field"):
            self._backend(i2v_image_field="bogus")

    def test_i2v_local_first_frame_normalized_to_data_url(self, tmp_path):
        """Local first-frame paths are converted to base64 data URLs."""
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"\x89PNG-fake")
        backend = self._backend()
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            backend.generate_video(prompt="animate it", first_frame_image=str(frame))

        media = client.post.call_args.kwargs["json"]["input"]["media"]
        assert media[0]["url"].startswith("data:image/png;base64,")

    def test_duration_normalized_to_integer(self):
        """DashScope video models require an integer number of seconds."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            backend.generate_video(prompt="a cat walking", duration=5.0)

        duration = client.post.call_args.kwargs["json"]["parameters"]["duration"]
        assert duration == 5
        assert isinstance(duration, int)

    def test_size_yields_to_resolution(self):
        """Legacy `size` is dropped when new-style resolution/ratio are set."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            backend.generate_video(
                prompt="a cat walking", size="1280*720", resolution="720P"
            )

        parameters = client.post.call_args.kwargs["json"]["parameters"]
        assert parameters["resolution"] == "720P"
        assert "size" not in parameters

    def test_failed_task_raises(self):
        """FAILED task status raises RuntimeError."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"output": {"task_id": "t-2", "task_status": "PENDING"}}],
            get_payloads=[
                {"output": {"task_id": "t-2", "task_status": "FAILED", "message": "boom"}}
            ],
        )
        with patch(_PAI_TOKEN_HTTPX, return_value=client), pytest.raises(
            RuntimeError, match="boom"
        ):
            backend.generate_video(prompt="a cat walking")

    def test_output_dir_downloads_video(self, tmp_path):
        """With output_dir set, remote videos are downloaded to local files."""
        backend = self._backend(output_dir=str(tmp_path))
        stream_resp = MagicMock()
        stream_resp.__enter__ = MagicMock(return_value=stream_resp)
        stream_resp.__exit__ = MagicMock(return_value=False)
        stream_resp.iter_bytes.return_value = [b"vid-bytes"]
        stream_resp.raise_for_status = MagicMock()
        client = _mock_client(
            post_payloads=[
                {
                    "output": {
                        "task_id": "t-7",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/v.mp4",
                    }
                }
            ],
        )
        client.stream.return_value = stream_resp
        with patch(_PAI_TOKEN_HTTPX, return_value=client):
            result = backend.generate_video(prompt="a cat walking")

        assert result.video_urls[0].endswith("t-7.mp4")
        assert (tmp_path / "t-7.mp4").read_bytes() == b"vid-bytes"
        assert result.metadata["remote_urls"] == ["https://cdn.example.com/v.mp4"]

    def test_is_t2v_backend(self):
        assert isinstance(self._backend(), T2VBackend)


class TestPAIVideoBackend:
    """Tests for the PAI-EAS video backend."""

    def _backend(self, **kwargs):
        return PAIVideoBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            poll_interval=0.01,
            max_poll_wait=5.0,
            **kwargs,
        )

    def test_sync_mode(self):
        """Sync response returns video URLs directly."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"data": [{"url": "https://cdn.example.com/v.mp4"}]}],
        )
        with patch(_PAI_VIDEO_HTTPX, return_value=client):
            result = backend.generate_video(prompt="a cat walking")

        assert result.video_urls == ["https://cdn.example.com/v.mp4"]
        assert result.metadata["mode"] == "sync"

    def test_async_mode_with_result_url(self):
        """Async response polls until completed and reads the video URL."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"task_id": "task-9"}],
            get_payloads=[
                {"status": "running"},
                {"status": "completed", "video_url": "https://cdn.example.com/v.mp4"},
            ],
        )
        with patch(_PAI_VIDEO_HTTPX, return_value=client):
            result = backend.generate_video(prompt="a cat walking")

        assert result.video_urls == ["https://cdn.example.com/v.mp4"]
        assert result.metadata["mode"] == "async"
        assert result.metadata["task_id"] == "task-9"

    def test_i2v_payload_includes_image_url(self):
        """I2V rows send image_url in the payload."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"data": [{"url": "https://cdn.example.com/v.mp4"}]}],
        )
        with patch(_PAI_VIDEO_HTTPX, return_value=client):
            backend.generate_video(
                prompt="animate the boat",
                first_frame_image="https://cdn.example.com/frame.png",
            )
        payload = client.post.call_args.kwargs["json"]
        assert payload["image_url"] == "https://cdn.example.com/frame.png"

    def test_async_failed_raises(self):
        """Failed async task raises RuntimeError."""
        backend = self._backend()
        client = _mock_client(
            post_payloads=[{"task_id": "task-x"}],
            get_payloads=[{"status": "failed", "error": "oom"}],
        )
        with patch(_PAI_VIDEO_HTTPX, return_value=client), pytest.raises(
            RuntimeError, match="oom"
        ):
            backend.generate_video(prompt="a cat walking")

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="protocol"):
            self._backend(protocol="bogus")

    def test_sglang_t2v_submit_poll_download(self, tmp_path):
        """sglang flow: POST /v1/videos -> poll -> download /content binary."""
        backend = self._backend(
            protocol="sglang", output_dir=str(tmp_path), auth_prefix=""
        )
        content_resp = MagicMock()
        content_resp.content = b"vid-bytes"
        content_resp.raise_for_status = MagicMock()
        client = _mock_client(
            post_payloads=[{"id": "vid-1", "status": "queued"}],
            get_payloads=[
                {"id": "vid-1", "status": "in_progress"},
                {"id": "vid-1", "status": "completed"},
            ],
        )
        # The binary download is a third GET returning raw content.
        client.get.side_effect = list(client.get.side_effect) + [content_resp]
        with patch(_PAI_VIDEO_HTTPX, return_value=client):
            result = backend.generate_video(prompt="a cat walking", duration=5)

        assert result.video_urls[0].endswith("vid-1.mp4")
        assert (tmp_path / "vid-1.mp4").read_bytes() == b"vid-bytes"
        assert result.metadata["mode"] == "sglang"
        assert result.metadata["task"] == "t2va"
        assert result.metadata["video_id"] == "vid-1"
        submit_kwargs = client.post.call_args.kwargs
        assert submit_kwargs["json"]["task"] == "t2va"
        assert submit_kwargs["json"]["target"] == {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
        }
        # Raw-token auth: auth_prefix="" yields no "Bearer " prefix.
        assert submit_kwargs["headers"]["Authorization"] == "test-token"
        # Status polled at /v1/videos/{id}, download at /v1/videos/{id}/content.
        get_urls = [c.args[0] for c in client.get.call_args_list]
        assert get_urls[0].endswith("/v1/videos/vid-1")
        assert get_urls[-1].endswith("/v1/videos/vid-1/content")

    def test_sglang_i2v_uses_i2v_task_and_reference_url(self, tmp_path):
        """sglang I2V rows switch the task and carry reference_url."""
        backend = self._backend(protocol="sglang", output_dir=str(tmp_path))
        content_resp = MagicMock()
        content_resp.content = b"vid-bytes"
        content_resp.raise_for_status = MagicMock()
        client = _mock_client(
            post_payloads=[{"id": "vid-2", "status": "queued"}],
            get_payloads=[{"id": "vid-2", "status": "completed"}],
        )
        client.get.side_effect = list(client.get.side_effect) + [content_resp]
        with patch(_PAI_VIDEO_HTTPX, return_value=client):
            result = backend.generate_video(
                prompt="animate the boat",
                first_frame_image="https://cdn.example.com/frame.png",
            )

        assert result.metadata["task"] == "fl2va"
        payload = client.post.call_args.kwargs["json"]
        assert payload["task"] == "fl2va"
        assert payload["reference_url"] == "https://cdn.example.com/frame.png"

    def test_sglang_i2v_rejects_local_first_frame(self):
        """sglang I2V requires an http(s) frame URL (service fetches it)."""
        backend = self._backend(protocol="sglang", output_dir="/tmp")
        with pytest.raises(ValueError, match="http\\(s\\) first-frame"):
            backend.generate_video(
                prompt="animate it",
                first_frame_image="data:image/png;base64,AAAA",
            )

    def test_sglang_requires_output_dir(self):
        """sglang returns binary video data, so output_dir must be set."""
        backend = self._backend(protocol="sglang")
        content_resp = MagicMock()
        content_resp.content = b"vid-bytes"
        content_resp.raise_for_status = MagicMock()
        client = _mock_client(
            post_payloads=[{"id": "vid-3", "status": "queued"}],
            get_payloads=[{"id": "vid-3", "status": "completed"}],
        )
        client.get.side_effect = list(client.get.side_effect) + [content_resp]
        with patch(_PAI_VIDEO_HTTPX, return_value=client), pytest.raises(
            RuntimeError, match="output_dir"
        ):
            backend.generate_video(prompt="a cat walking")

    def test_sglang_failed_video_raises(self):
        backend = self._backend(protocol="sglang", output_dir="/tmp")
        client = _mock_client(
            post_payloads=[{"id": "vid-4", "status": "queued"}],
            get_payloads=[{"id": "vid-4", "status": "failed", "error": "oom"}],
        )
        with patch(_PAI_VIDEO_HTTPX, return_value=client), pytest.raises(
            RuntimeError, match="oom"
        ):
            backend.generate_video(prompt="a cat walking")
