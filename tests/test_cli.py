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

"""Unit tests for CLI helpers."""

import os
import sys
from unittest.mock import patch

import pytest

from easydistill.cli import _build_backend, _expand_env_vars, _load_requests
from easydistill.cli.backend_factory import close_backends
from easydistill.cli.data_loaders import load_string_column
from easydistill.data.models import GenerationRequest


class TestExpandEnvVars:
    def test_expands_both_var_syntaxes(self):
        os.environ["CLI_TEST_KEY"] = "secret"
        cfg = {
            "backend": {
                "api_key": "${CLI_TEST_KEY}",
                "base_url": "$CLI_TEST_KEY/path",
            },
            "list": ["${CLI_TEST_KEY}"],
        }
        expanded = _expand_env_vars(cfg)
        assert expanded["backend"]["api_key"] == "secret"
        assert expanded["backend"]["base_url"] == "secret/path"
        assert expanded["list"] == ["secret"]

    def test_missing_env_raises(self):
        os.environ.pop("CLI_MISSING", None)
        cfg = {"key": "${CLI_MISSING}"}
        with pytest.raises(ValueError, match="unset environment variable"):
            _expand_env_vars(cfg)


class TestBuildBackend:
    @patch("easydistill.cli.backend_factory.OpenAIBackend")
    def test_build_openai_backend(self, mock_cls):
        os.environ["CLI_OPENAI_KEY"] = "key"
        _build_backend(
            {
                "type": "openai",
                "api_key": "${CLI_OPENAI_KEY}",
                "base_url": "https://api.example.com/v1",
                "model_id": "gpt-4",
            }
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://api.example.com/v1"
        assert kwargs["model_id"] == "gpt-4"

    @patch("easydistill.cli.backend_factory.PaiTokenBackend")
    def test_build_pai_token_backend(self, mock_cls):
        os.environ["CLI_PT_KEY"] = "pt_key"
        _build_backend(
            {
                "type": "pai_token",
                "api_key": "${CLI_PT_KEY}",
                "base_url": "https://cn-beijing.pai-token.aliyuncs.com/v1",
                "model_id": "qwen2.5-72b-instruct",
            }
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["api_key"] == "pt_key"
        assert kwargs["base_url"] == "https://cn-beijing.pai-token.aliyuncs.com/v1"
        assert kwargs["model_id"] == "qwen2.5-72b-instruct"

    @patch("easydistill.cli.backend_factory.EASBackend")
    def test_build_eas_backend(self, mock_cls):
        _build_backend(
            {
                "type": "pai_eas",
                "endpoint_url": "https://eas.example.com/v1",
                "token": "token123",
            }
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["endpoint_url"] == "https://eas.example.com/v1"
        assert kwargs["token"] == "token123"

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend type"):
            _build_backend({"type": "unknown"})

    def test_openai_backend_requires_api_key(self):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OpenAI backend requires"):
            _build_backend({"type": "openai"})

    def test_pai_token_backend_requires_api_key(self):
        os.environ.pop("PAI_TOKEN_API_KEY", None)
        with pytest.raises(ValueError, match="PAI-Token backend requires"):
            _build_backend({"type": "pai_token"})

    def test_pai_eas_backend_requires_endpoint_and_token(self):
        os.environ.pop("EAS_ENDPOINT_URL", None)
        os.environ.pop("EAS_TOKEN", None)
        with pytest.raises(ValueError, match="PAI-EAS backend requires"):
            _build_backend({"type": "pai_eas"})

    def test_invalid_timeout_raises(self):
        os.environ["CLI_OPENAI_KEY"] = "key"
        with pytest.raises(ValueError, match="Config field 'timeout'"):
            _build_backend({"type": "openai", "api_key": "${CLI_OPENAI_KEY}", "timeout": "fast"})

    def test_negative_timeout_raises(self):
        os.environ["CLI_OPENAI_KEY"] = "key"
        with pytest.raises(ValueError, match="Config field 'timeout' must be >= 0.0"):
            _build_backend({"type": "openai", "api_key": "${CLI_OPENAI_KEY}", "timeout": -1})

    def test_negative_max_retries_raises(self):
        os.environ["CLI_OPENAI_KEY"] = "key"
        with pytest.raises(ValueError, match="Config field 'max_retries' must be >= 0.0"):
            _build_backend({"type": "openai", "api_key": "${CLI_OPENAI_KEY}", "max_retries": -1})


class TestBuildT2IBackend:
    @patch("easydistill.cli.backend_factory.WanxBackend")
    def test_wanx_invalid_poll_interval_raises(self, mock_cls):
        os.environ["CLI_DS_KEY"] = "key"
        from easydistill.cli.backend_factory import build_t2i_backend

        with pytest.raises(ValueError, match="Config field 'poll_interval'"):
            build_t2i_backend(
                {"type": "wanx", "api_key": "${CLI_DS_KEY}", "poll_interval": "soon"}
            )

    @patch("easydistill.cli.backend_factory.QwenImageBackend")
    def test_qwen_image_negative_max_poll_wait_raises(self, mock_cls):
        os.environ["CLI_DS_KEY"] = "key"
        from easydistill.cli.backend_factory import build_t2i_backend

        with pytest.raises(ValueError, match="Config field 'max_poll_wait' must be >= 0.0"):
            build_t2i_backend(
                {"type": "qwen_image", "api_key": "${CLI_DS_KEY}", "max_poll_wait": -10}
            )


class TestLoadRequests:
    def test_load_requests_from_rows(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(
            '{"id": "a", "instruction": "Q1"}\n{"id": "b", "instruction": "Q2"}\n',
            encoding="utf-8",
        )
        config = {
            "dataset": {
                "input_path": str(input_path),
                "instruction_key": "instruction",
                "system_key": "system",
            },
            "generation": {"system_prompt": "SYS"},
        }
        requests = _load_requests(config)
        assert len(requests) == 2
        assert all(isinstance(r, GenerationRequest) for r in requests)
        assert requests[0].id == "a"
        assert requests[0].instruction == "Q1"
        assert requests[0].system_prompt == "SYS"

    def test_load_requests_missing_instruction(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text('{"id": "a", "instruction": "Q1"}\n{"id": "b"}\n')
        config = {
            "dataset": {
                "input_path": str(input_path),
                "instruction_key": "instruction",
            }
        }
        requests = _load_requests(config)
        assert len(requests) == 1
        assert requests[0].id == "a"

    def test_load_requests_empty_file_raises(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text("")
        config = {"dataset": {"input_path": str(input_path)}}
        with pytest.raises(ValueError, match="No data found"):
            _load_requests(config)


class TestLoadStringColumn:
    def test_uses_default_column_names(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(
            '{"instruction": "Q1", "text": "T1"}\n'
            '{"instruction": "Q2", "text": "T2"}\n',
            encoding="utf-8",
        )
        config = {"dataset": {"input_path": str(input_path)}}
        assert load_string_column(config, "instruction_key") == ["Q1", "Q2"]
        assert load_string_column(config, "text_key") == ["T1", "T2"]

    def test_null_config_key_falls_back_to_default(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text('{"instruction": "Q1"}\n', encoding="utf-8")
        config = {"dataset": {"input_path": str(input_path), "instruction_key": None}}
        assert load_string_column(config, "instruction_key") == ["Q1"]


class TestMainCLI:
    def test_list_jobs_prints_supported_jobs(self, capsys):
        from easydistill.cli.main import main

        with patch.object(sys, "argv", ["easydistill", "--list-jobs"]):
            main()
        captured = capsys.readouterr()
        assert "instruct_distill" in captured.out
        assert "advanced_instruct_distill" in captured.out

    def test_list_models_prints_distilqwen_models(self, capsys):
        from easydistill.cli.main import main

        with patch.object(sys, "argv", ["easydistill", "--list-models"]):
            main()
        captured = capsys.readouterr()
        assert "DistilQwen2.5-7B-Instruct" in captured.out
        assert "DistilQwen-ThoughtX" in captured.out
        assert "Model Zoo" in captured.out

    def test_missing_config_raises(self):
        from easydistill.cli.main import main

        with pytest.raises(SystemExit), patch.object(sys, "argv", ["easydistill"]):
            main()

    def test_unsupported_job_type_lists_valid_ones(self, tmp_path):
        from easydistill.cli.main import main

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "job_type: not_a_job\n"
            "backend:\n  type: openai\n  api_key: key\n"
            "dataset:\n  input_path: /dev/null\n"
        )
        with pytest.raises(ValueError, match="not_a_job") as exc_info, patch.object(
            sys, "argv", ["easydistill", "--config", str(config_path)]
        ):
            main()
        assert "instruct_distill" in str(exc_info.value)


class TestCloseBackends:
    def test_close_backends_returns_true_when_all_ok(self):
        ok_backend = type("OkBackend", (), {"close": lambda self: None})()
        assert close_backends(ok_backend, None, ok_backend) is True

    def test_close_backends_returns_false_and_logs_on_failure(self, caplog):
        class BadBackend:
            def close(self):
                raise RuntimeError("close failed")

        ok_backend = type("OkBackend", (), {"close": lambda self: None})()
        assert close_backends(ok_backend, BadBackend()) is False
        assert "Failed to close backend" in caplog.text
        assert "close failed" in caplog.text
