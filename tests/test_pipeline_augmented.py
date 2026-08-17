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

"""Unit tests for AugmentedInstructDistillPipeline."""

from unittest.mock import MagicMock

import pytest

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.pipeline import AugmentedInstructDistillPipeline


class TestAugmentedInstructDistillPipeline:
    def test_expansion_then_distill(self):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            prompt = messages[-1]["content"]
            if "NEW instruction" in prompt:
                return GenerationResult(
                    request=GenerationRequest(instruction=prompt),
                    response="<answer>expanded instruction</answer>",
                    model="mock",
                )
            return GenerationResult(
                request=GenerationRequest(instruction=prompt),
                response="final answer",
                model="mock",
            )

        backend.generate.side_effect = fake_generate

        pipeline = AugmentedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {
                    "stage": "instruction_expansion",
                    "config": {
                        "num_in_context_samples": 1,
                        "num_output_samples": 2,
                        "show_progress": False,
                    },
                },
                {
                    "stage": "instruct_distill",
                    "config": {"show_progress": False},
                },
            ],
            dataset_config={"input_path": "dummy"},
            generation_config={"system_prompt": "You are helpful."},
            sft_config={},
        )

        data = [{"instruction": "seed1"}, {"instruction": "seed2"}]
        result = pipeline.run_with_data(data)
        assert len(result) == 2
        assert result[0]["messages"][0]["content"] == "You are helpful."
        assert result[0]["messages"][-1]["content"] == "final answer"

    def test_refinement_then_distill(self):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            prompt = messages[-1]["content"]
            if "rewrite and improve" in prompt:
                return GenerationResult(
                    request=GenerationRequest(instruction=prompt),
                    response="<answer>refined instruction</answer>",
                    model="mock",
                )
            return GenerationResult(
                request=GenerationRequest(instruction=prompt),
                response="answer",
                model="mock",
            )

        backend.generate.side_effect = fake_generate

        pipeline = AugmentedInstructDistillPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "instruction_refinement", "config": {"show_progress": False}},
                {"stage": "instruct_distill", "config": {"show_progress": False}},
            ],
            dataset_config={"input_path": "dummy"},
        )

        data = [{"instruction": "seed1"}]
        result = pipeline.run_with_data(data)
        assert len(result) == 1
        assert result[0]["messages"][-1]["content"] == "answer"

    def test_last_stage_must_be_basic_distill(self):
        backend = MagicMock()
        with pytest.raises(ValueError):
            AugmentedInstructDistillPipeline(
                backend=backend,
                pipeline_config=[{"stage": "instruction_expansion"}],
                dataset_config={"input_path": "dummy"},
            )
