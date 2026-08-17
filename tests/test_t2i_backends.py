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

"""Unit tests for T2I backends."""

from unittest.mock import MagicMock, patch

import pytest

from easydistill.backends import PAIDiffusionBackend, T2IBackend

_HTTPX_CLIENT = "easydistill.backends.pai_diffusion_backend.httpx.Client"


class TestPAIDiffusionBackend:
    """Tests for the PAI-Diffusion backend (httpx-based)."""

    # ------------------------------------------------------------------
    # Sync mode tests (OpenAI-compatible /images/generations)
    # ------------------------------------------------------------------

    def test_generate_image_with_url(self):
        """Test that image URLs are extracted from the sync response."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            model_id="sdxl",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"url": "https://cdn.example.com/img1.png"}],
            "usage": {"images": 1},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            result = backend.generate_image(prompt="a cat", size="1024*1024", n=1)

        assert result.prompt == "a cat"
        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "https://cdn.example.com/img1.png"
        assert result.model == "sdxl"
        assert result.usage == {"images": 1}
        assert result.metadata["mode"] == "sync"
        # Verify the API was called with size converted from * to x.
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["size"] == "1024x1024"
        # Verify width/height are also included for async-mode compatibility.
        assert call_kwargs["json"]["width"] == 1024
        assert call_kwargs["json"]["height"] == 1024

    def test_generate_image_with_b64(self):
        """Test that b64_json responses are converted to data URLs."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"b64_json": "AAAA"}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            result = backend.generate_image(prompt="a dog")

        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "data:image/png;base64,AAAA"

    def test_generate_image_no_images(self):
        """Test that empty data list returns no image URLs."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            result = backend.generate_image(prompt="empty")

        assert result.image_urls == []

    # ------------------------------------------------------------------
    # Async mode tests (Qwen-Image on EAS: submit → poll → download)
    # ------------------------------------------------------------------

    def test_async_generate_image_data_url(self):
        """Test async task-based generation returns data URL when no output_dir."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            model_id="Qwen-Image",
            auth_prefix="",
            poll_interval=0.01,
            max_poll_wait=5.0,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "test-task-123"}
        mock_post_resp.raise_for_status = MagicMock()

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "completed"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_image_resp = MagicMock()
        mock_image_resp.content = b"fake-image-bytes"
        mock_image_resp.headers = {"content-type": "image/jpeg"}
        mock_image_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.side_effect = [mock_status_resp, mock_image_resp]

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            result = backend.generate_image(prompt="a cat", size="1024*1024")

        assert result.prompt == "a cat"
        assert len(result.image_urls) == 1
        assert result.image_urls[0].startswith("data:image/jpg;base64,")
        assert result.model == "Qwen-Image"
        assert result.metadata["mode"] == "async"
        assert result.metadata["task_id"] == "test-task-123"
        # Verify auth header uses raw token (no Bearer prefix).
        post_headers = mock_client.post.call_args.kwargs["headers"]
        assert post_headers["Authorization"] == "test-token"

    def test_async_generate_image_with_output_dir(self, tmp_path):
        """Test async task-based generation saves image to output_dir."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            model_id="Qwen-Image",
            auth_prefix="",
            output_dir=str(tmp_path),
            poll_interval=0.01,
            max_poll_wait=5.0,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "test-task-456"}
        mock_post_resp.raise_for_status = MagicMock()

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "completed"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_image_resp = MagicMock()
        mock_image_resp.content = b"fake-png-bytes"
        mock_image_resp.headers = {"content-type": "image/png"}
        mock_image_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.side_effect = [mock_status_resp, mock_image_resp]

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            result = backend.generate_image(prompt="a dog")

        assert len(result.image_urls) == 1
        assert result.image_urls[0].endswith(".png")
        assert "test-task-456" in result.image_urls[0]

    def test_async_task_failure(self):
        """Test that async task failure raises RuntimeError."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            poll_interval=0.01,
            max_poll_wait=5.0,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "test-task-fail"}
        mock_post_resp.raise_for_status = MagicMock()

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "failed", "error": "GPU OOM"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.return_value = mock_status_resp

        with (
            patch(_HTTPX_CLIENT, return_value=mock_client),
            pytest.raises(RuntimeError, match="GPU OOM"),
        ):
            backend.generate_image(prompt="bad prompt")

    def test_async_timeout(self):
        """Test that async polling raises TimeoutError after max_poll_wait."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
            poll_interval=0.01,
            max_poll_wait=0.02,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "test-task-slow"}
        mock_post_resp.raise_for_status = MagicMock()

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "pending"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.return_value = mock_status_resp

        with (
            patch(_HTTPX_CLIENT, return_value=mock_client),
            patch("easydistill.backends.pai_diffusion_backend.time.sleep"),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            backend.generate_image(prompt="slow prompt")

    def test_async_passes_qwen_image_kwargs(self):
        """Test that Qwen-Image specific kwargs are passed through."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
            poll_interval=0.01,
            max_poll_wait=5.0,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"task_id": "t1"}
        mock_post_resp.raise_for_status = MagicMock()

        mock_status_resp = MagicMock()
        mock_status_resp.json.return_value = {"status": "completed"}
        mock_status_resp.raise_for_status = MagicMock()

        mock_image_resp = MagicMock()
        mock_image_resp.content = b"x"
        mock_image_resp.headers = {"content-type": "image/jpeg"}
        mock_image_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_post_resp
        mock_client.get.side_effect = [mock_status_resp, mock_image_resp]

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            backend.generate_image(
                prompt="test",
                seed=42,
                negative_prompt="blurry",
                infer_steps=50,
                cfg_scale=4,
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["seed"] == 42
        assert payload["negative_prompt"] == "blurry"
        assert payload["infer_steps"] == 50
        assert payload["cfg_scale"] == 4

    # ------------------------------------------------------------------
    # Helper / utility tests
    # ------------------------------------------------------------------

    def test_auth_prefix(self):
        """Test that auth_prefix controls the Authorization header."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="my-token",
            auth_prefix="",
        )
        assert backend._headers["Authorization"] == "my-token"

        backend2 = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="my-token",
            auth_prefix="Bearer ",
        )
        assert backend2._headers["Authorization"] == "Bearer my-token"

    def test_parse_size_to_wh(self):
        """Test size string parsing to width/height."""
        assert PAIDiffusionBackend._parse_size_to_wh("1024*1024") == (1024, 1024)
        assert PAIDiffusionBackend._parse_size_to_wh("1664x928") == (1664, 928)
        assert PAIDiffusionBackend._parse_size_to_wh("invalid") == (1024, 1024)

    def test_base_url(self):
        """Test that _base_url strips /v1 suffix."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
        )
        assert backend._base_url == "https://eas.example.com"

        backend2 = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com",
            token="t",
        )
        assert backend2._base_url == "https://eas.example.com"

    # ------------------------------------------------------------------
    # Health check tests
    # ------------------------------------------------------------------

    def test_health_check_success(self):
        """Test health_check returns True on 200 from /models."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            assert backend.health_check() is True

    def test_health_check_fallback(self):
        """Test health_check falls back to base URL when /models returns 404."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
        )
        mock_404 = MagicMock()
        mock_404.status_code = 404

        mock_200 = MagicMock()
        mock_200.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [mock_404, mock_200]

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            assert backend.health_check() is True

    def test_health_check_failure_500(self):
        """Test health_check returns False when both /models and base URL return 500."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
        )
        mock_500 = MagicMock()
        mock_500.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_500

        with patch(_HTTPX_CLIENT, return_value=mock_client):
            assert backend.health_check() is False

    def test_health_check_failure_exception(self):
        """Test health_check returns False on connection exception."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="test-token",
        )
        with patch(
            "easydistill.backends.pai_diffusion_backend.httpx.Client",
            side_effect=Exception("connection error"),
        ):
            assert backend.health_check() is False

    # ------------------------------------------------------------------
    # Other tests
    # ------------------------------------------------------------------

    def test_default_model_id(self):
        """Test that the default model ID is stable-diffusion-xl."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
        )
        assert backend.model_id == "stable-diffusion-xl"

    def test_context_manager(self):
        """Test that PAIDiffusionBackend supports context manager protocol."""
        backend = PAIDiffusionBackend(
            endpoint_url="https://eas.example.com/v1",
            token="t",
        )
        with backend as ctx:
            assert ctx is backend


class TestWanxBackend:
    """Tests for the Wanx (Tongyi Wanxiang) backend via dashscope SDK."""

    def _make_mock_dashscope(self, sync_result=False):
        """Create a mock dashscope module."""
        mock_ds = MagicMock()

        if sync_result:
            # Synchronous: call() returns results immediately.
            mock_rsp = MagicMock()
            mock_rsp.status_code = 200
            mock_rsp.output = {
                "results": [{"url": "https://cdn.example.com/sync_img.png"}],
                "task_id": "task-sync",
            }
            mock_rsp.usage = {"image_count": 1}
            mock_ds.ImageSynthesis.call.return_value = mock_rsp
        else:
            # Async: call() returns task_id, fetch() returns SUCCEEDED.
            submit_rsp = MagicMock()
            submit_rsp.status_code = 200
            submit_rsp.output = {"task_id": "task-async-123"}
            submit_rsp.usage = None
            mock_ds.ImageSynthesis.call.return_value = submit_rsp

            fetch_rsp = MagicMock()
            fetch_rsp.output = {
                "task_status": "SUCCEEDED",
                "results": [{"url": "https://cdn.example.com/async_img.png"}],
                "task_id": "task-async-123",
            }
            fetch_rsp.usage = {"image_count": 1}
            mock_ds.ImageSynthesis.fetch.return_value = fetch_rsp

        return mock_ds

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_sync(self, mock_dashscope):
        """Test synchronous generation (results returned immediately)."""
        mock_ds = self._make_mock_dashscope(sync_result=True)
        mock_dashscope.ImageSynthesis.call.return_value = mock_ds.ImageSynthesis.call.return_value

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key", model_id="wanx-test")
        result = backend.generate_image(prompt="a cat", size="1024*1024", n=1)

        assert result.prompt == "a cat"
        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "https://cdn.example.com/sync_img.png"
        assert result.model == "wanx-test"
        # The configured HTTP timeout must reach the dashscope SDK call.
        call_kwargs = mock_dashscope.ImageSynthesis.call.call_args.kwargs
        assert call_kwargs["request_timeout"] == backend.timeout
        # Should not have called fetch (synchronous response).
        mock_dashscope.ImageSynthesis.fetch.assert_not_called()

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_async_poll(self, mock_dashscope):
        """Test asynchronous generation with task polling."""
        mock_ds = self._make_mock_dashscope(sync_result=False)
        mock_dashscope.ImageSynthesis.call.return_value = mock_ds.ImageSynthesis.call.return_value
        mock_dashscope.ImageSynthesis.fetch.return_value = mock_ds.ImageSynthesis.fetch.return_value

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(
            api_key="test-key",
            model_id="wanx-test",
            poll_interval=0.01,
            max_poll_wait=5.0,
        )
        result = backend.generate_image(prompt="a dog")

        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "https://cdn.example.com/async_img.png"
        # fetch should have been called at least once.
        assert mock_dashscope.ImageSynthesis.fetch.call_count >= 1

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_request_timeout_override(self, mock_dashscope):
        """Extra kwargs can override the default request_timeout."""
        mock_rsp = MagicMock()
        mock_rsp.status_code = 200
        mock_rsp.output = {"results": [{"url": "https://cdn.example.com/img.png"}]}
        mock_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = mock_rsp

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key", timeout=30.0)
        backend.generate_image(prompt="a cat", request_timeout=90.0)

        call_kwargs = mock_dashscope.ImageSynthesis.call.call_args.kwargs
        assert call_kwargs["request_timeout"] == 90.0

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_submit_failure(self, mock_dashscope):
        """Test that a non-200 status code raises RuntimeError."""
        mock_rsp = MagicMock()
        mock_rsp.status_code = 400
        mock_rsp.message = "Invalid prompt"
        mock_dashscope.ImageSynthesis.call.return_value = mock_rsp

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key")
        with pytest.raises(RuntimeError, match="Wanx submit failed"):
            backend.generate_image(prompt="bad prompt")

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_submit_retries_on_transient_error(self, mock_dashscope):
        """Test that transient submit errors are retried."""
        ok_rsp = MagicMock()
        ok_rsp.status_code = 200
        ok_rsp.output = {"results": [{"url": "https://cdn.example.com/img.png"}]}
        ok_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.side_effect = [ConnectionError("reset"), ok_rsp]

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key", retry_attempts=1, retry_backoff_base=0.01)
        result = backend.generate_image(prompt="a cat")
        assert result.image_urls == ["https://cdn.example.com/img.png"]
        assert mock_dashscope.ImageSynthesis.call.call_count == 2

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_poll_retries_fetch_on_transient_error(self, mock_dashscope):
        """Test that transient fetch errors during polling are retried."""
        submit_rsp = MagicMock()
        submit_rsp.status_code = 200
        submit_rsp.output = {"task_id": "task-retry"}
        submit_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = submit_rsp

        ok_fetch = MagicMock()
        ok_fetch.output = {
            "task_status": "SUCCEEDED",
            "results": [{"url": "https://cdn.example.com/poll.png"}],
            "task_id": "task-retry",
        }
        ok_fetch.usage = None
        mock_dashscope.ImageSynthesis.fetch.side_effect = [TimeoutError("slow"), ok_fetch]

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(
            api_key="test-key",
            poll_interval=0.01,
            max_poll_wait=5.0,
            retry_attempts=1,
            retry_backoff_base=0.01,
        )
        result = backend.generate_image(prompt="a dog")
        assert result.image_urls == ["https://cdn.example.com/poll.png"]
        assert mock_dashscope.ImageSynthesis.fetch.call_count == 2

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_generate_image_poll_re_raises_keyboard_interrupt(self, mock_dashscope):
        """Test that KeyboardInterrupt during polling is not swallowed."""
        submit_rsp = MagicMock()
        submit_rsp.status_code = 200
        submit_rsp.output = {"task_id": "task-interrupt"}
        submit_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = submit_rsp
        mock_dashscope.ImageSynthesis.fetch.side_effect = KeyboardInterrupt

        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key", poll_interval=0.01, max_poll_wait=5.0)
        with pytest.raises(KeyboardInterrupt):
            backend.generate_image(prompt="a dog")

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.wanx_backend.dashscope")
    def test_health_check(self, mock_dashscope):
        """Test health_check returns True when API key is set."""
        from easydistill.backends.wanx_backend import WanxBackend

        backend = WanxBackend(api_key="test-key")
        assert backend.health_check() is True

    @patch("easydistill.backends.wanx_backend._HAS_DASHSCOPE", False)
    def test_import_error_without_dashscope(self):
        """Test that WanxBackend raises ImportError when dashscope is not installed."""
        from easydistill.backends.wanx_backend import WanxBackend

        with pytest.raises(ImportError, match="dashscope"):
            WanxBackend(api_key="test-key")


class TestQwenImageBackend:
    """Tests for the Qwen-Image backend via dashscope SDK."""

    def _make_mock_dashscope(self, sync_result=False):
        """Create a mock dashscope module."""
        mock_ds = MagicMock()

        if sync_result:
            mock_rsp = MagicMock()
            mock_rsp.status_code = 200
            mock_rsp.output = {
                "results": [{"url": "https://cdn.example.com/qwen_sync_img.png"}],
                "task_id": "qwen-task-sync",
            }
            mock_rsp.usage = {"image_count": 1}
            mock_ds.ImageSynthesis.call.return_value = mock_rsp
        else:
            submit_rsp = MagicMock()
            submit_rsp.status_code = 200
            submit_rsp.output = {"task_id": "qwen-task-async-123"}
            submit_rsp.usage = None
            mock_ds.ImageSynthesis.call.return_value = submit_rsp

            fetch_rsp = MagicMock()
            fetch_rsp.output = {
                "task_status": "SUCCEEDED",
                "results": [{"url": "https://cdn.example.com/qwen_async_img.png"}],
                "task_id": "qwen-task-async-123",
            }
            fetch_rsp.usage = {"image_count": 1}
            mock_ds.ImageSynthesis.fetch.return_value = fetch_rsp

        return mock_ds

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_sync(self, mock_dashscope):
        """Test synchronous generation (results returned immediately)."""
        mock_ds = self._make_mock_dashscope(sync_result=True)
        mock_dashscope.ImageSynthesis.call.return_value = mock_ds.ImageSynthesis.call.return_value

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key", model_id="qwen-image-test")
        result = backend.generate_image(prompt="a cat", size="1024*1024", n=1)

        assert result.prompt == "a cat"
        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "https://cdn.example.com/qwen_sync_img.png"
        assert result.model == "qwen-image-test"
        call_kwargs = mock_dashscope.ImageSynthesis.call.call_args.kwargs
        assert call_kwargs["request_timeout"] == backend.timeout
        mock_dashscope.ImageSynthesis.fetch.assert_not_called()

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_async_poll(self, mock_dashscope):
        """Test asynchronous generation with task polling."""
        mock_ds = self._make_mock_dashscope(sync_result=False)
        mock_dashscope.ImageSynthesis.call.return_value = mock_ds.ImageSynthesis.call.return_value
        mock_dashscope.ImageSynthesis.fetch.return_value = mock_ds.ImageSynthesis.fetch.return_value

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(
            api_key="test-key",
            model_id="qwen-image-test",
            poll_interval=0.01,
            max_poll_wait=5.0,
        )
        result = backend.generate_image(prompt="a dog")

        assert len(result.image_urls) == 1
        assert result.image_urls[0] == "https://cdn.example.com/qwen_async_img.png"
        assert mock_dashscope.ImageSynthesis.fetch.call_count >= 1

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_request_timeout_override(self, mock_dashscope):
        """Extra kwargs can override the default request_timeout."""
        mock_rsp = MagicMock()
        mock_rsp.status_code = 200
        mock_rsp.output = {"results": [{"url": "https://cdn.example.com/img.png"}]}
        mock_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = mock_rsp

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key", timeout=30.0)
        backend.generate_image(prompt="a cat", request_timeout=90.0)

        call_kwargs = mock_dashscope.ImageSynthesis.call.call_args.kwargs
        assert call_kwargs["request_timeout"] == 90.0

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_submit_failure(self, mock_dashscope):
        """Test that a non-200 status code raises RuntimeError."""
        mock_rsp = MagicMock()
        mock_rsp.status_code = 400
        mock_rsp.message = "Invalid prompt"
        mock_dashscope.ImageSynthesis.call.return_value = mock_rsp

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key")
        with pytest.raises(RuntimeError, match="Qwen-Image submit failed"):
            backend.generate_image(prompt="bad prompt")

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_submit_retries_on_transient_error(self, mock_dashscope):
        """Test that transient submit errors are retried."""
        ok_rsp = MagicMock()
        ok_rsp.status_code = 200
        ok_rsp.output = {"results": [{"url": "https://cdn.example.com/qwen.png"}]}
        ok_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.side_effect = [ConnectionError("reset"), ok_rsp]

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key", retry_attempts=1, retry_backoff_base=0.01)
        result = backend.generate_image(prompt="a cat")
        assert result.image_urls == ["https://cdn.example.com/qwen.png"]
        assert mock_dashscope.ImageSynthesis.call.call_count == 2

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_poll_retries_fetch_on_transient_error(self, mock_dashscope):
        """Test that transient fetch errors during polling are retried."""
        submit_rsp = MagicMock()
        submit_rsp.status_code = 200
        submit_rsp.output = {"task_id": "qwen-retry"}
        submit_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = submit_rsp

        ok_fetch = MagicMock()
        ok_fetch.output = {
            "task_status": "SUCCEEDED",
            "results": [{"url": "https://cdn.example.com/qwen_poll.png"}],
            "task_id": "qwen-retry",
        }
        ok_fetch.usage = None
        mock_dashscope.ImageSynthesis.fetch.side_effect = [TimeoutError("slow"), ok_fetch]

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(
            api_key="test-key",
            poll_interval=0.01,
            max_poll_wait=5.0,
            retry_attempts=1,
            retry_backoff_base=0.01,
        )
        result = backend.generate_image(prompt="a dog")
        assert result.image_urls == ["https://cdn.example.com/qwen_poll.png"]
        assert mock_dashscope.ImageSynthesis.fetch.call_count == 2

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_generate_image_poll_re_raises_keyboard_interrupt(self, mock_dashscope):
        """Test that KeyboardInterrupt during polling is not swallowed."""
        submit_rsp = MagicMock()
        submit_rsp.status_code = 200
        submit_rsp.output = {"task_id": "qwen-interrupt"}
        submit_rsp.usage = None
        mock_dashscope.ImageSynthesis.call.return_value = submit_rsp
        mock_dashscope.ImageSynthesis.fetch.side_effect = KeyboardInterrupt

        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key", poll_interval=0.01, max_poll_wait=5.0)
        with pytest.raises(KeyboardInterrupt):
            backend.generate_image(prompt="a dog")

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_health_check(self, mock_dashscope):
        """Test health_check returns True when API key is set."""
        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key")
        assert backend.health_check() is True

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", False)
    def test_import_error_without_dashscope(self):
        """Test that QwenImageBackend raises ImportError when dashscope is not installed."""
        from easydistill.backends.qwen_image_backend import QwenImageBackend

        with pytest.raises(ImportError, match="dashscope"):
            QwenImageBackend(api_key="test-key")

    @patch("easydistill.backends.qwen_image_backend._HAS_DASHSCOPE", True)
    @patch("easydistill.backends.qwen_image_backend.dashscope")
    def test_default_model_id(self, mock_dashscope):
        """Test that the default model ID is qwen-image2.0-pro."""
        from easydistill.backends.qwen_image_backend import QwenImageBackend

        backend = QwenImageBackend(api_key="test-key")
        assert backend.model_id == "qwen-image2.0-pro"


class TestT2IBackendABC:
    """Test the T2IBackend abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that T2IBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            T2IBackend()
