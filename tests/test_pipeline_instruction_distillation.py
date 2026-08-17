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

"""Unit tests for AdvancedInstructDistillPipeline."""

from unittest.mock import MagicMock

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.pipeline import AdvancedInstructDistillPipeline


class TestAdvancedInstructDistillPipeline:
    def _make_backend(self, response: str):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            prompt = messages[-1]["content"]
            req = GenerationRequest(instruction=prompt)
            return GenerationResult(request=req, response=response, model="mock")

        backend.generate.side_effect = fake_generate
        return backend

    def test_generate_and_build_sft(self):
        backend = self._make_backend("Paris")
        pipeline = AdvancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "generate", "config": {"max_workers": 1}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            generation_config={"system_prompt": "You are helpful."},
        )
        data = [{"instruction": "What is the capital of France?"}]
        results = pipeline.run_with_data(data)
        assert len(results) == 1
        messages = results[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is the capital of France?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Paris"

    def test_quality_filter_keeps_high_scores(self):
        backend = self._make_backend("<score>8</score>")
        pipeline = AdvancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "generate", "config": {"max_workers": 1}},
                {"stage": "instruct_eval", "config": {"max_workers": 1}},
                {
                    "stage": "quality_filter",
                    "config": {
                        "min_scores": {"informativeness": 6},
                        "require_all_metrics": False,
                    },
                },
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            eval_config={"metrics": ["informativeness"], "max_workers": 1},
        )
        data = [
            {"instruction": "What is 2+2?", "response": "4"},
        ]
        results = pipeline.run_with_data(data)
        assert len(results) == 1

    def test_quality_filter_drops_low_scores(self):
        backend = self._make_backend("<score>3</score>")
        pipeline = AdvancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "generate", "config": {"max_workers": 1}},
                {"stage": "instruct_eval", "config": {"max_workers": 1}},
                {
                    "stage": "quality_filter",
                    "config": {
                        "min_scores": {"informativeness": 6},
                        "require_all_metrics": False,
                    },
                },
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            eval_config={"metrics": ["informativeness"], "max_workers": 1},
        )
        data = [
            {"instruction": "What is 2+2?", "response": "4"},
        ]
        results = pipeline.run_with_data(data)
        assert len(results) == 0

    def test_generate_skips_and_aligns_rows_without_instruction(self):
        backend = self._make_backend("Paris")
        pipeline = AdvancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "generate", "config": {"max_workers": 1}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            generation_config={"system_prompt": "You are helpful."},
        )
        data = [
            {"id": "first", "instruction": "What is the capital of France?"},
            {"id": "skip", "instruction": ""},
            {"id": "second", "instruction": "What is the capital of Italy?"},
        ]
        results = pipeline.run_with_data(data)
        assert len(results) == 2
        # Ensure responses align with the correct original rows.
        assert results[0]["metadata"]["request_id"] == "first"
        assert results[0]["messages"][-1]["content"] == "Paris"
        assert results[1]["metadata"]["request_id"] == "second"
        assert results[1]["messages"][-1]["content"] == "Paris"

    def test_last_stage_must_be_build_sft(self):
        backend = self._make_backend("x")
        try:
            AdvancedInstructDistillPipeline(
                backend=backend,
                pipeline_config=[{"stage": "generate", "config": {}}],
                dataset_config={"input_path": "dummy.jsonl"},
            )
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "build_sft" in str(e)
