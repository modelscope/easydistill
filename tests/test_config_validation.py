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

"""Unit tests for config validation schemas."""

import pytest

from easydistill.utils import validate_config


def test_valid_minimal_config():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl", "output_path": "out.jsonl"},
    }
    validated = validate_config(cfg)
    assert validated["job_type"] == "instruct_distill"
    assert validated["backend"]["type"] == "openai"
    assert validated["dataset"]["input_path"] == "data.jsonl"


def test_unknown_backend_type_rejected():
    cfg = {
        "backend": {"type": "unknown"},
        "dataset": {"input_path": "data.jsonl"},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_missing_backend_rejected():
    cfg = {"dataset": {"input_path": "data.jsonl"}}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_missing_dataset_input_path_rejected():
    cfg = {"backend": {"type": "openai"}, "dataset": {}}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_pipeline_stage_requires_stage_name():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl"},
        "pipeline": [{"stage": "build_sft"}, {"config": {}}],
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_backend_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai", "api_key": "key", "custom_arg": True},
        "dataset": {"input_path": "data.jsonl"},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_dataset_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl", "unknown_key": True},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_generation_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl"},
        "generation": {"temperture": 0.7},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_pipeline_stage_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl"},
        "pipeline": [
            {"stage": "build_sft", "config": {}, "unknown_key": True}
        ],
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_top_level_extra_keys_allowed():
    cfg = {
        "backend": {"type": "openai", "api_key": "key"},
        "dataset": {"input_path": "data.jsonl"},
        "custom_section": {"foo": "bar"},
    }
    validated = validate_config(cfg)
    assert validated["custom_section"] == {"foo": "bar"}


def test_validate_config_paths_checks_input_exists(tmp_path):
    from easydistill.utils.config import validate_config_paths

    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="Input path does not exist"):
        validate_config_paths({"dataset": {"input_path": str(missing)}})


def test_mm_config_valid_and_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl"},
        "mm": {"system_prompt": "You are helpful.", "temperature": 0.5},
    }
    validated = validate_config(cfg)
    assert validated["mm"]["temperature"] == 0.5

    cfg["mm"]["unknown_key"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_agent_config_valid_and_extra_keys_rejected():
    cfg = {
        "backend": {"type": "openai"},
        "dataset": {"input_path": "data.jsonl"},
        "agent": {"max_steps": 15, "repeat_times": 3, "max_tool_calls": 10, "use_rubrics": False},
    }
    validated = validate_config(cfg)
    assert validated["agent"]["max_steps"] == 15
    assert validated["agent"]["use_rubrics"] is False

    cfg["agent"]["unknown_key"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_paths_rejects_directory_output(tmp_path):
    from easydistill.utils.config import validate_config_paths

    with pytest.raises(ValueError, match="Output path is a directory"):
        validate_config_paths({"dataset": {"output_path": str(tmp_path)}})


def _base_t2v_cfg():
    return {
        "job_type": "advanced_t2v_distill",
        "backend": {"type": "pai_token", "api_key": "sk-x", "model_id": "m"},
        "dataset": {"input_path": "data.jsonl"},
    }


def test_t2v_backend_valid_configs():
    cfg = _base_t2v_cfg()
    cfg["t2v_backend"] = {
        "type": "pai_token_video",
        "api_key": "sk-x",
        "base_url": "https://example.com/api/v1",
        "model_id": "happyhorse-1.1-t2v",
        "i2v_model_id": "happyhorse-1.1-i2v",
        "output_dir": "outputs/videos",
    }
    validate_config(cfg)

    cfg["t2v_backend"] = {
        "type": "pai_video",
        "endpoint_url": "https://example.com",
        "token": "tok",
        "protocol": "sglang",
        "sglang_short_edge": 768,
    }
    validate_config(cfg)


def test_t2v_backend_extra_keys_rejected():
    cfg = _base_t2v_cfg()
    cfg["t2v_backend"] = {
        "type": "pai_token_video",
        "api_keyy": "sk-typo",  # misspelled -> must be rejected
    }
    with pytest.raises(ValueError, match="api_keyy"):
        validate_config(cfg)

    cfg["t2v_backend"] = {
        "type": "pai_video",
        "endpoint_url": "https://example.com",
        "sglang_short_edeg": 768,  # misspelled -> must be rejected
    }
    with pytest.raises(ValueError, match="sglang_short_edeg"):
        validate_config(cfg)


def test_t2v_backend_unknown_type_rejected():
    cfg = _base_t2v_cfg()
    cfg["t2v_backend"] = {"type": "wanx_video"}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_t2v_backend_schema_covers_factory_keys():
    """Guard against drift: every config key consumed by build_t2v_backend
    must be declared in one of the T2V backend schemas, otherwise adding a
    factory parameter without updating the schema would reject valid configs.
    """
    import inspect
    import re

    from easydistill.cli import backend_factory
    from easydistill.utils.schemas import (
        PaiTokenVideoBackendConfig,
        PAIVideoBackendConfig,
    )

    source = inspect.getsource(backend_factory.build_t2v_backend)
    consumed = set(re.findall(r"config\.get\(\s*\"([a-z0-9_]+)\"", source))
    consumed |= set(
        re.findall(r"_resolve_backend_value\(\s*config,\s*\"([a-z0-9_]+)\"", source)
    )
    schema_fields = (
        set(PaiTokenVideoBackendConfig.model_fields)
        | set(PAIVideoBackendConfig.model_fields)
    )
    missing = consumed - schema_fields
    assert not missing, f"Factory keys missing from T2V backend schemas: {missing}"
