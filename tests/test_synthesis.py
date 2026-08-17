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

"""Unit tests for synthesis operators."""

import random
from unittest.mock import MagicMock

import pytest

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.synthesis.utils import (
    extract_instruction_response,
    extract_tagged_answer,
    format_in_context_examples,
    sample_in_context_examples,
)
from easydistill.rewrite import (
    InstructionExpansionOperator,
    InstructionRefinementOperator,
    InstructionResponseExtractionOperator,
)


class TestSynthesisUtils:
    def test_sample_in_context_examples(self):
        pool = ["a", "b", "c", "d"]
        examples = sample_in_context_examples(pool, 2, seed=42)
        assert len(examples) == 2
        assert len(set(examples)) == 2
        assert all(e in pool for e in examples)

    def test_sample_in_context_examples_exclude(self):
        pool = ["a", "b", "c"]
        examples = sample_in_context_examples(pool, 2, exclude="a", seed=42)
        assert "a" not in examples

    def test_sample_in_context_examples_does_not_mutate_global_random_state(self):
        random.seed(123)
        before = random.random()
        random.seed(123)
        sample_in_context_examples(["a", "b", "c"], 1, seed=42)
        after = random.random()
        assert after == before

    def test_extract_tagged_answer(self):
        text = "Some text <answer>the answer</answer> more"
        assert extract_tagged_answer(text) == "the answer"

    def test_extract_tagged_answer_missing(self):
        assert extract_tagged_answer("no tags") is None

    def test_extract_instruction_response(self):
        text = "<instruction>Q</instruction> <response>A</response>"
        ins, resp = extract_instruction_response(text)
        assert ins == "Q"
        assert resp == "A"

    def test_format_in_context_examples(self):
        formatted = format_in_context_examples(["a", "b"])
        assert "Example: 1\na" in formatted
        assert "Example: 2\nb" in formatted


class TestInstructionExpansionOperator:
    def test_generates_new_instructions(self):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            return GenerationResult(
                request=GenerationRequest(instruction=messages[-1]["content"]),
                response="<answer>expanded instruction</answer>",
                model="mock",
            )

        backend.generate.side_effect = fake_generate
        operator = InstructionExpansionOperator(
            backend=backend,
            config={"num_output_samples": 2, "num_in_context_samples": 2, "show_progress": False},
        )
        outputs = operator.run(["seed1", "seed2", "seed3"])
        assert len(outputs) == 2
        assert outputs[0] == "expanded instruction"
        assert backend.generate.call_count == 2

    def test_falls_back_to_whole_response(self):
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="prompt"),
            response="whole response",
            model="mock",
        )
        operator = InstructionExpansionOperator(
            backend=backend,
            config={"num_output_samples": 1, "num_in_context_samples": 2, "show_progress": False},
        )
        outputs = operator.run(["a", "b", "c"])
        assert outputs == ["whole response"]

    def test_too_few_seeds_raises(self):
        backend = MagicMock()
        operator = InstructionExpansionOperator(
            backend=backend,
            config={"num_output_samples": 1, "num_in_context_samples": 5, "show_progress": False},
        )
        with pytest.raises(ValueError):
            operator.run(["a", "b"])


class TestInstructionRefinementOperator:
    def test_refinement(self):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            return GenerationResult(
                request=GenerationRequest(instruction=messages[-1]["content"]),
                response="<answer>refined</answer>",
                model="mock",
            )

        backend.generate.side_effect = fake_generate
        operator = InstructionRefinementOperator(
            backend=backend,
            config={"show_progress": False},
        )
        outputs = operator.run(["ins1", "ins2"])
        assert len(outputs) == 2
        assert outputs[0] == "refined"


class TestInstructionResponseExtractionOperator:
    def test_llm_extraction(self):
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="prompt", metadata={"text": "raw"}),
            response="<instruction>Q</instruction>\n<response>A</response>",
            model="mock",
        )
        operator = InstructionResponseExtractionOperator(
            backend=backend,
            config={"show_progress": False},
        )
        outputs = operator.run(["raw text"])
        assert len(outputs) == 1
        assert outputs[0] == ("Q", "A")

    def test_no_llm_regex_extraction(self):
        backend = MagicMock()
        operator = InstructionResponseExtractionOperator(
            backend=backend,
            config={"use_llm": False},
        )
        outputs = operator.run(["<instruction>Q1</instruction><response>A1</response>"])
        assert outputs == [("Q1", "A1")]
        backend.generate.assert_not_called()

    def test_fallback_to_input_text(self):
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(
                instruction="prompt",
                metadata={"text": "<instruction>Q</instruction><response>A</response>"},
            ),
            response="no tags here",
            model="mock",
        )
        operator = InstructionResponseExtractionOperator(
            backend=backend,
            config={"show_progress": False},
        )
        outputs = operator.run(["<instruction>Q</instruction><response>A</response>"])
        assert outputs == [("Q", "A")]
