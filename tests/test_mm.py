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

"""Unit tests for multi-modal operators, evaluators, and pipelines."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from easydistill.cli.data_loaders import load_multimodal_inputs
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval import MMCoTEvaluator, MMInstructionFollowingEvaluator
from easydistill.operators.mm import (
    MMCoTGenerationOperator,
    MMGenerationOperator,
)
from easydistill.pipeline import MMCoTDistillationPipeline, MMDistillationPipeline
from easydistill.pipeline.common import run_build_sft_stage
from easydistill.rewrite import (
    MMCoTLong2ShortOperator,
    MMCoTShort2LongOperator,
)
from easydistill.utils import (
    build_multimodal_user_content,
    is_image_url,
    load_image_to_data_url,
    normalize_image_reference,
)
from tests._fake_backend import FakeBackend


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a tiny PNG file for testing image normalization.

    A hardcoded minimal 1x1 PNG keeps the test free of any image library
    dependency; the code under test only base64-encodes raw bytes.
    """
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
        "h6FO1AAAAABJRU5ErkJggg=="
    )
    path = tmp_path / "test.png"
    path.write_bytes(png_bytes)
    return str(path)


class TestImageUtilities:
    def test_is_image_url(self):
        assert is_image_url("http://example.com/img.png")
        assert is_image_url("https://example.com/img.png")
        assert is_image_url("data:image/png;base64,AAAA")
        assert not is_image_url("/path/to/image.png")
        assert not is_image_url("just text")

    def test_normalize_image_reference_url(self):
        url = "https://example.com/img.png"
        assert normalize_image_reference(url) == url

    def test_normalize_image_reference_local_file(self, sample_image_path):
        data_url = normalize_image_reference(sample_image_path)
        assert data_url.startswith("data:image/png;base64,")

    def test_load_image_to_data_url_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_image_to_data_url("/tmp/nonexistent_image.png")

    def test_build_multimodal_user_content(self):
        content = build_multimodal_user_content("What is this?", ["https://example.com/img.png"])
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "What is this?"


class TestMMGenerationOperator:
    def test_run_with_fake_backend(self):
        backend = FakeBackend(response_template="Answer: {}")
        operator = MMGenerationOperator(backend=backend, config={"max_workers": 1})
        inputs = [
            {"instruction": "What is in the image?", "images": ["https://example.com/img.png"]}
        ]
        outputs = operator.run(inputs)
        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "What is in the image?"
        assert outputs[0]["images"] == ["https://example.com/img.png"]
        assert "Answer:" in outputs[0]["response"]

    def test_prompt_template(self):
        backend = FakeBackend(response_template="{}")
        operator = MMGenerationOperator(
            backend=backend,
            config={"prompt_template": "Question: {instruction}\nAnswer:", "max_workers": 1},
        )
        outputs = operator.run([{"instruction": "Hello", "images": []}])
        # FakeBackend returns the formatted instruction text, not the raw template.
        assert "Question: Hello" in outputs[0]["response"]

    def test_none_system_prompt_uses_default(self):
        """Explicitly configured None system prompt must not become the string 'None'."""
        default_system_prompt = (
            Path("configs/prompts/mm_generation_prompt.txt").read_text().rstrip("\n")
        )

        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="Hello"),
            response="OK",
            model="mock",
        )
        operator = MMGenerationOperator(
            backend=backend,
            config={"system_prompt": None, "max_workers": 1},
        )
        operator.run([{"instruction": "Hello", "images": []}])
        messages = backend.generate.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == default_system_prompt

    def test_list_instruction_preserved(self):
        class RecordingBackend(FakeBackend):
            def __init__(self):
                super().__init__(response_template="{}")
                self.messages = None

            def generate(self, messages, **kwargs):
                self.messages = messages
                return super().generate(messages, **kwargs)

        backend = RecordingBackend()
        operator = MMGenerationOperator(
            backend=backend,
            config={
                "prompt_template": "Question: {instruction}\nAnswer:",
                "max_workers": 1,
            },
        )
        instruction_list = [{"type": "text", "text": "Hello"}]
        outputs = operator.run([{"instruction": instruction_list, "images": []}])
        assert len(outputs) == 1
        assert outputs[0]["instruction"] == instruction_list
        assert backend.messages is not None
        user_content = backend.messages[-1]["content"]
        assert isinstance(user_content, list)
        assert user_content == instruction_list

    def test_default_system_prompt(self):
        backend = FakeBackend(response_template="{}")
        operator = MMGenerationOperator(backend=backend, config={"max_workers": 1})
        # The default system prompt should be applied to the generator config.
        assert operator.system_prompt is not None
        assert "visual assistant" in operator.system_prompt.lower()

    def test_system_prompt_passed_to_backend(self):
        class RecordingBackend(FakeBackend):
            def __init__(self):
                super().__init__(response_template="{}")
                self.messages = None

            def generate(self, messages, **kwargs):
                self.messages = messages
                return super().generate(messages, **kwargs)

        backend = RecordingBackend()
        operator = MMGenerationOperator(backend=backend, config={"max_workers": 1})
        operator.run([{"instruction": "Hello", "images": []}])
        assert backend.messages is not None
        assert backend.messages[0]["role"] == "system"
        assert "visual assistant" in backend.messages[0]["content"].lower()


class TestMMCoTGenerationOperator:
    def test_run_extracts_thought_and_solution(self):
        backend = FakeBackend(
            response_template=(
                "<|begin_of_thought|>thinking<|end_of_thought|>"
                "<|begin_of_solution|>42<|end_of_solution|>"
            )
        )
        operator = MMCoTGenerationOperator(backend=backend, config={"max_workers": 1})
        outputs = operator.run(
            [{"instruction": "Solve this", "images": ["https://example.com/img.png"]}]
        )
        assert len(outputs) == 1
        assert outputs[0]["thought"] == "thinking"
        assert outputs[0]["solution"] == "42"

    def test_list_instruction_does_not_crash(self):
        backend = FakeBackend(
            response_template=(
                "<|begin_of_thought|>thinking<|end_of_thought|>"
                "<|begin_of_solution|>42<|end_of_solution|>"
            )
        )
        operator = MMCoTGenerationOperator(backend=backend, config={"max_workers": 1})
        instruction_list = [{"type": "text", "text": "Solve this"}]
        outputs = operator.run(
            [{"instruction": instruction_list, "images": ["https://example.com/img.png"]}]
        )
        assert len(outputs) == 1
        assert outputs[0]["thought"] == "thinking"


class TestMMCoTRewriteOperators:
    def test_long2short(self):
        backend = FakeBackend(response_template="short")
        operator = MMCoTLong2ShortOperator(backend=backend, config={"max_workers": 1})
        outputs = operator.run(
            [
                {
                    "instruction": "Problem",
                    "images": ["https://example.com/img.png"],
                    "response": "a very long reasoning process",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["response"] == "short"
        assert outputs[0]["compression_ratio"] < 1.0

    def test_long2short_with_list_instruction(self):
        backend = FakeBackend(response_template="short")
        operator = MMCoTLong2ShortOperator(backend=backend, config={"max_workers": 1})
        instruction_list = [{"type": "text", "text": "Problem"}]
        outputs = operator.run(
            [
                {
                    "instruction": instruction_list,
                    "images": ["https://example.com/img.png"],
                    "response": "a very long reasoning process",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["response"] == "short"

    def test_short2long(self):
        backend = FakeBackend(response_template="extended reasoning\n\nwith steps")
        operator = MMCoTShort2LongOperator(backend=backend, config={"max_workers": 1})
        outputs = operator.run(
            [
                {
                    "instruction": "Problem",
                    "images": ["https://example.com/img.png"],
                    "response": "short",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["expansion_ratio"] > 1.0
        assert outputs[0]["step_count"] >= 1

    def test_short2long_with_list_instruction(self):
        backend = FakeBackend(response_template="extended reasoning\n\nwith steps")
        operator = MMCoTShort2LongOperator(backend=backend, config={"max_workers": 1})
        instruction_list = [{"type": "text", "text": "Problem"}]
        outputs = operator.run(
            [
                {
                    "instruction": instruction_list,
                    "images": ["https://example.com/img.png"],
                    "response": "short",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["expansion_ratio"] > 1.0


class TestMMEvaluators:
    def test_mm_instruct_evaluator(self):
        backend = FakeBackend(response_template="<score>8</score>")
        evaluator = MMInstructionFollowingEvaluator(
            backend=backend,
            config={"metrics": ["helpfulness"], "max_workers": 1},
        )
        results = evaluator.run(
            [
                {
                    "id": "1",
                    "instruction": "What is this?",
                    "output": "A cat.",
                    "images": ["https://example.com/img.png"],
                }
            ]
        )
        assert len(results) == 1
        assert results[0]["helpfulness"] == 8
        assert results[0]["images"] == ["https://example.com/img.png"]

    def test_mm_instruct_evaluator_with_list_instruction(self):
        backend = FakeBackend(response_template="<score>8</score>")
        evaluator = MMInstructionFollowingEvaluator(
            backend=backend,
            config={"metrics": ["helpfulness"], "max_workers": 1},
        )
        instruction_list = [{"type": "text", "text": "What is this?"}]
        results = evaluator.run(
            [
                {
                    "id": "1",
                    "instruction": instruction_list,
                    "output": "A cat.",
                    "images": ["https://example.com/img.png"],
                }
            ]
        )
        assert len(results) == 1
        assert results[0]["helpfulness"] == 8

    def test_mm_cot_evaluator(self):
        backend = FakeBackend(response_template="<score>1</score>")
        evaluator = MMCoTEvaluator(
            backend=backend,
            config={"metrics": ["logical_correctness"], "max_workers": 1},
        )
        results = evaluator.run(
            [
                {
                    "id": "1",
                    "instruction": "Solve",
                    "output": "Answer is 42.",
                    "images": ["https://example.com/img.png"],
                }
            ]
        )
        assert results[0]["logical_correctness"] is True


class TestMMPipelines:
    def test_mm_distillation_pipeline(self):
        backend = FakeBackend(response_template="I see a red square.")
        pipeline = MMDistillationPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "mm_instruct_distill", "config": {"max_workers": 1}},
                {
                    "stage": "mm_instruct_eval",
                    "config": {"metrics": ["helpfulness"], "max_workers": 1},
                },
                {"stage": "quality_filter", "config": {}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "ignored"},
            generation_config={"system_prompt": "You are helpful."},
            sft_config={},
            eval_config={},
        )
        data = [
            {
                "id": "0",
                "instruction": "Describe the image.",
                "images": ["https://example.com/img.png"],
            }
        ]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        messages = result[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "image_url"

    def test_mm_cot_distillation_pipeline(self):
        backend = FakeBackend(
            response_template=(
                "<|begin_of_thought|>thinking<|end_of_thought|>"
                "<|begin_of_solution|>42<|end_of_solution|>"
            )
        )
        pipeline = MMCoTDistillationPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "mm_cot_distill", "config": {"max_workers": 1}},
                {
                    "stage": "mm_cot_eval",
                    "config": {"metrics": ["logical_correctness"], "max_workers": 1},
                },
                {"stage": "quality_filter", "config": {}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "ignored"},
            generation_config={},
            sft_config={},
            eval_config={},
        )
        data = [
            {
                "id": "0",
                "instruction": "What is 6x7?",
                "images": ["https://example.com/img.png"],
            }
        ]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        user_message = next((m for m in result[0]["messages"] if m["role"] == "user"), None)
        assert user_message is not None
        assert user_message["content"][0]["type"] == "image_url"


class TestMMDataLoading:
    def test_load_multimodal_inputs_preserves_list_instruction(self, tmp_path):
        input_path = tmp_path / "mm_inputs.jsonl"
        row = {
            "instruction": [{"type": "text", "text": "Hello"}],
            "images": ["https://example.com/img.png"],
        }
        input_path.write_text(json.dumps(row) + "\n")
        config = {"dataset": {"input_path": str(input_path)}}
        inputs = load_multimodal_inputs(config)
        assert len(inputs) == 1
        assert isinstance(inputs[0]["instruction"], list)
        assert inputs[0]["instruction"][0]["text"] == "Hello"


class TestMMSFTStage:
    def test_run_build_sft_stage_preserves_list_instruction(self):
        data = [
            {
                "id": "0",
                "instruction": [{"type": "text", "text": "What is this?"}],
                "response": "A cat.",
            }
        ]
        result = run_build_sft_stage(data, {}, {})
        assert len(result) == 1
        user_message = next((m for m in result[0]["messages"] if m["role"] == "user"), None)
        assert user_message is not None
        assert user_message["content"] == [{"type": "text", "text": "What is this?"}]
