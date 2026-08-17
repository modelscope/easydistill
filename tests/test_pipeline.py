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

"""Basic smoke tests for the distillation pipeline."""

import os
from typing import Any, Dict, List

from easydistill.cli import _expand_env_vars
from easydistill.data.models import GenerationRequest
from easydistill.operators import SFTDatasetBuilder, TextGenerationOperator
from easydistill.pipeline.base import BaseDistillationPipeline
from easydistill.pipeline.common import run_build_sft_stage, run_quality_filter_stage
from easydistill.utils import load_jsonl
from tests._fake_backend import FakeBackend


class _FailingPipeline(BaseDistillationPipeline):
    """Pipeline subclass that fails at a configured stage for testing."""

    _last_stage = {"build_sft"}

    def __init__(self, failing_stage: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._failing_stage = failing_stage

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == self._failing_stage:
            raise RuntimeError(f"Simulated failure in {stage_name}")
        return [{**row, "stage": stage_name} for row in data]


def test_fake_backend_pipeline():
    backend = FakeBackend(response_template="Answer: {}")
    generator = TextGenerationOperator(backend=backend, config={"show_progress": False})
    requests = [
        GenerationRequest(id="1", instruction="What is 1+1?"),
        GenerationRequest(id="2", instruction="Say hello."),
    ]
    results = generator.run(requests)
    assert len(results) == 2
    assert "What is 1+1?" in results[0].response

    builder = SFTDatasetBuilder(config={})
    samples = builder.run(results)
    assert len(samples) == 2
    assert samples[0].messages[0].role == "user"
    assert samples[0].messages[1].role == "assistant"


def test_expand_env_vars():
    os.environ["ED_TEST_KEY"] = "secret"
    cfg = {"backend": {"api_key": "${ED_TEST_KEY}", "url": "$ED_TEST_KEY"}}
    expanded = _expand_env_vars(cfg)
    assert expanded["backend"]["api_key"] == "secret"
    assert expanded["backend"]["url"] == "secret"


def test_quality_filter_bool_threshold_treats_float_strictly():
    data = [
        {"id": "1", "correctness": True},
        {"id": "2", "correctness": False},
        {"id": "3", "correctness": 0.5},
    ]
    result = run_quality_filter_stage(
        {"min_scores": {"correctness": True}},
        data,
        ["correctness"],
    )
    ids = {row["id"] for row in result}
    assert "1" in ids
    assert "2" not in ids
    assert "3" not in ids


def test_pipeline_saves_recovery_checkpoint_on_stage_failure(tmp_path):
    output_path = str(tmp_path / "output.jsonl")
    pipeline_config = [
        {"stage": "expand", "output_path": str(tmp_path / "expand.jsonl")},
        {"stage": "filter", "output_path": str(tmp_path / "filter.jsonl")},
        {"stage": "build_sft"},
    ]
    pipeline = _FailingPipeline(
        failing_stage="filter",
        backend=FakeBackend(),
        pipeline_config=pipeline_config,
        dataset_config={"input_path": str(tmp_path / "input.jsonl"), "output_path": output_path},
    )
    seed = [{"id": "1", "instruction": "hello"}]
    try:
        pipeline.run_with_data(seed)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "filter" in str(exc)

    recovery_files = list(tmp_path.glob("output.jsonl.recovery.*"))
    assert len(recovery_files) == 1
    recovered = load_jsonl(str(recovery_files[0]))
    assert recovered == [{"id": "1", "instruction": "hello", "stage": "expand"}]


def test_run_build_sft_stage_supports_pai_output_field():
    data = [
        {
            "instruction": "Q1",
            "output": "A1",
        },
        {
            "instruction": "Q2",
            "output": "A2",
        },
    ]
    samples = run_build_sft_stage(data)
    assert len(samples) == 2
    assert samples[0]["messages"][0]["content"] == "Q1"
    assert samples[0]["messages"][1]["content"] == "A1"
    assert samples[1]["messages"][0]["content"] == "Q2"
    assert samples[1]["messages"][1]["content"] == "A2"


def test_run_build_sft_stage_respects_response_key_config():
    data = [
        {"instruction": "Q1", "response": "from-response", "output": "from-output"},
    ]
    # Default tries response first.
    samples = run_build_sft_stage(data)
    assert samples[0]["messages"][1]["content"] == "from-response"

    # Explicit response_key forces output only.
    samples = run_build_sft_stage(data, global_sft_config={"response_key": "output"})
    assert samples[0]["messages"][1]["content"] == "from-output"


def test_run_build_sft_stage_respects_images_key_config():
    data = [
        {
            "instruction": "Describe the image.",
            "response": "A red square.",
            "img_refs": ["examples/mm_sample_image.png"],
        },
    ]
    samples = run_build_sft_stage(data, global_sft_config={"images_key": "img_refs"})
    assert len(samples) == 1
    user_content = samples[0]["messages"][0]["content"]
    assert isinstance(user_content, list)
    assert any(item["type"] == "image_url" for item in user_content)
    assert any(item["type"] == "text" for item in user_content)
