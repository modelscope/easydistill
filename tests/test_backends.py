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

"""Unit tests for model backends."""

from unittest.mock import MagicMock, patch

import pytest

from easydistill.backends import (
    EASBackend,
    OpenAIBackend,
    PaiTokenBackend,
)


class TestOpenAIBackend:
    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_generate(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.model_dump.return_value = {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        }
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        backend = OpenAIBackend(
            api_key="key",
            base_url="https://api.example.com/v1",
            model_id="gpt-4",
        )
        result = backend.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert result.response == "Hello!"
        assert result.model == "gpt-4"
        assert result.usage["total_tokens"] == 5
        mock_client.chat.completions.create.assert_called_once()

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_resolve_model_from_list(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "model-a"
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        mock_openai_cls.return_value = mock_client

        backend = OpenAIBackend(api_key="key", base_url="https://api.example.com/v1")
        assert backend._resolve_model(None) == "model-a"

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_generate_handles_empty_choices(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = []
        mock_completion.usage = None
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        backend = OpenAIBackend(
            api_key="key",
            base_url="https://api.example.com/v1",
            model_id="gpt-4",
        )
        result = backend.generate(messages=[{"role": "user", "content": "hi"}])
        assert result.response == ""
        assert result.model == "gpt-4"

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_generate_with_multimodal_messages(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "I see a red square."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = None
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        backend = OpenAIBackend(
            api_key="key",
            base_url="https://api.example.com/v1",
            model_id="gpt-4v",
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        result = backend.generate(messages=messages)
        assert result.response == "I see a red square."
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == messages

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_close_logs_error_on_failure(self, mock_openai_cls, caplog):
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("boom")
        mock_openai_cls.return_value = mock_client

        backend = OpenAIBackend(api_key="key", base_url="https://api.example.com/v1")
        backend.close()

        mock_client.close.assert_called_once()
        assert "Failed to close OpenAI client" in caplog.text
        assert "boom" in caplog.text


class TestPaiTokenBackend:
    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_uses_api_key_and_default_base_url(self, mock_openai_cls):
        PaiTokenBackend(api_key="key", model_id="kimi-k2.6")
        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "key"
        assert call_kwargs["base_url"] == "https://cn-beijing.pai-token.aliyuncs.com/v1"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("PAI_TOKEN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="PAI_TOKEN_API_KEY"):
            PaiTokenBackend(model_id="kimi-k2.6")

    def test_missing_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            PaiTokenBackend(api_key="key")

    def test_uses_env_model_id(self, monkeypatch):
        monkeypatch.setenv("PAI_TOKEN_MODEL_ID", "kimi-k2.6")
        backend = PaiTokenBackend(api_key="key")
        assert backend.model_id == "kimi-k2.6"

    def test_uses_provided_model_id(self):
        backend = PaiTokenBackend(api_key="key", model_id="qwen2.5-72b-instruct")
        assert backend.model_id == "qwen2.5-72b-instruct"


class TestEASBackend:
    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_init_uses_token_and_endpoint(self, mock_openai_cls):
        EASBackend(
            endpoint_url="https://eas.example.com/v1",
            token="token123",
            model_id="custom-model",
        )
        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "token123"
        assert call_kwargs["base_url"] == "https://eas.example.com/v1"

    def test_init_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("EAS_ENDPOINT_URL", raising=False)
        with pytest.raises(ValueError, match="endpoint_url"):
            EASBackend(token="token123")

    def test_init_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("EAS_TOKEN", raising=False)
        with pytest.raises(ValueError, match="token"):
            EASBackend(endpoint_url="https://eas.example.com/v1")

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_init_preserves_predict_path(self, mock_openai_cls):
        EASBackend(
            endpoint_url="http://example.aliyuncs.com/api/predict/my_eas_service",
            token="token123",
            model_id="Qwen2.5-VL-3B-Instruct",
        )
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["base_url"] == (
            "http://example.aliyuncs.com/api/predict/my_eas_service/v1"
        )

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_health_check_falls_back_to_chat_probe(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        backend = EASBackend(
            endpoint_url="https://eas.example.com/v1",
            token="token123",
            model_id="custom-model",
        )
        mock_client.models.list.side_effect = Exception("model list not supported")
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="pong"))],
            usage=None,
        )
        assert backend.health_check() is True
        mock_client.chat.completions.create.assert_called_once()

    @patch("easydistill.backends.openai_backend.OpenAI")
    def test_health_check_fails_when_both_probes_fail(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        backend = EASBackend(
            endpoint_url="https://eas.example.com/v1",
            token="token123",
            model_id="custom-model",
        )
        mock_client.models.list.side_effect = Exception("model list not supported")
        mock_client.chat.completions.create.side_effect = Exception("unreachable")
        assert backend.health_check() is False
