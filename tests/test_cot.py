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

"""Unit tests for CoT operators, evaluator, and pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval import CoTEvaluator
from easydistill.operators.cot import (
    CoTGenerationOperator,
    CoTRVCDMixer,
    CoTRVCDScorer,
    extract_between_tags,
    extract_cot_sections,
)
from easydistill.pipeline import CoTDistillationPipeline
from easydistill.rewrite import (
    CoTLong2ShortOperator,
    CoTShort2LongOperator,
)
from tests._fake_backend import FakeBackend

_COT_GENERATION_PROMPT = Path("configs/prompts/cot_generation_prompt.txt").read_text().rstrip("\n")
_COT_LONG2SHORT_PROMPT = Path("configs/prompts/cot_long2short_prompt.txt").read_text().rstrip("\n")
_COT_SHORT2LONG_PROMPT = Path("configs/prompts/cot_short2long_prompt.txt").read_text().rstrip("\n")


class TestCoTUtils:
    def test_extract_between_tags(self):
        text = "<|begin_of_thought|>think<|end_of_thought|>"
        assert extract_between_tags(text, "<|begin_of_thought|>", "<|end_of_thought|>") == "think"

    def test_extract_cot_sections(self):
        text = (
            "<|begin_of_thought|>reasoning<|end_of_thought|>\n"
            "<|begin_of_solution|>answer<|end_of_solution|>"
        )
        thought, solution = extract_cot_sections(text)
        assert thought == "reasoning"
        assert solution == "answer"

    def test_extract_cot_sections_missing(self):
        thought, solution = extract_cot_sections("no tags here")
        assert thought is None
        assert solution is None

    def test_extract_cot_sections_thinking_answer_format(self):
        text = "<thinking>step by step reasoning</thinking>\n<answer>final answer</answer>"
        thought, solution = extract_cot_sections(text)
        assert thought == "step by step reasoning"
        assert solution == "final answer"

    def test_extract_cot_sections_prefers_legacy_tags(self):
        text = (
            "<|begin_of_thought|>legacy thought<|end_of_thought|>\n"
            "<|begin_of_solution|>legacy solution<|end_of_solution|>\n"
            "<thinking>other</thinking><answer>other</answer>"
        )
        thought, solution = extract_cot_sections(text)
        assert thought == "legacy thought"
        assert solution == "legacy solution"


class TestCoTGenerationOperator:
    def test_generates_cot_output(self):
        backend = FakeBackend(response_template="{}")
        operator = CoTGenerationOperator(
            backend=backend,
            config={"show_progress": False, "prompt_template": "Solve: {problem}"},
        )
        outputs = operator.run(["2+2"])
        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "2+2"
        assert outputs[0]["response"] == "Solve: 2+2"
        assert "thought" in outputs[0]
        assert "solution" in outputs[0]

    def test_extracts_thought_and_solution(self):
        backend = FakeBackend(
            response_template=(
                "<|begin_of_thought|>think<|end_of_thought|>"
                "<|begin_of_solution|>4<|end_of_solution|>"
            )
        )
        operator = CoTGenerationOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(["2+2"])
        assert outputs[0]["thought"] == "think"
        assert outputs[0]["solution"] == "4"

    def test_skips_empty_responses(self):
        backend = FakeBackend(response_template="   ")
        operator = CoTGenerationOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(["2+2"])
        assert len(outputs) == 0

    def test_default_prompt_template(self):
        backend = FakeBackend(response_template="{}")
        operator = CoTGenerationOperator(backend=backend, config={"show_progress": False})
        assert operator.prompt_template == _COT_GENERATION_PROMPT


class TestCoTLong2ShortOperator:
    def test_simplifies_cot(self):
        backend = FakeBackend(response_template="short: {}")
        operator = CoTLong2ShortOperator(
            backend=backend,
            config={"show_progress": False, "prompt_template": "{problem} | {answer}"},
        )
        outputs = operator.run([("2+2", "long reasoning")])
        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "2+2"
        assert outputs[0]["response"] == "short: 2+2 | long reasoning"
        assert outputs[0]["original_response"] == "long reasoning"

    def test_default_prompt_template(self):
        backend = FakeBackend(response_template="x")
        operator = CoTLong2ShortOperator(backend=backend, config={"show_progress": False})
        assert operator.prompt_template == _COT_LONG2SHORT_PROMPT

    def test_metadata_and_compression_ratio(self):
        backend = FakeBackend(response_template="short answer")
        operator = CoTLong2ShortOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run([("2+2", "a" * 40)])
        assert len(outputs) == 1
        assert "original_tokens" in outputs[0]
        assert "simplified_tokens" in outputs[0]
        assert "compression_ratio" in outputs[0]
        assert outputs[0]["original_tokens"] == 10
        assert outputs[0]["simplified_tokens"] == 3

    def test_max_length_truncates(self):
        backend = FakeBackend(response_template="word " * 20)
        operator = CoTLong2ShortOperator(
            backend=backend,
            config={"show_progress": False, "max_length": 30},
        )
        outputs = operator.run([("2+2", "long")])
        assert len(outputs) == 1
        assert len(outputs[0]["response"]) <= 30

    def test_verify_solution_tags_drops_mismatched(self):
        backend = FakeBackend(response_template="short without solution tag")
        operator = CoTLong2ShortOperator(
            backend=backend,
            config={"show_progress": False, "verify_solution_tags": True},
        )
        outputs = operator.run([("2+2", "<|begin_of_solution|>4<|end_of_solution|>")])
        assert len(outputs) == 0

    def test_verify_solution_tags_keeps_matched(self):
        backend = FakeBackend(response_template="short <|begin_of_solution|>4<|end_of_solution|>")
        operator = CoTLong2ShortOperator(
            backend=backend,
            config={"show_progress": False, "verify_solution_tags": True},
        )
        outputs = operator.run([("2+2", "<|begin_of_solution|>4<|end_of_solution|>")])
        assert len(outputs) == 1


class TestCoTShort2LongOperator:
    def test_extends_cot(self):
        backend = FakeBackend(response_template="long: {}")
        operator = CoTShort2LongOperator(
            backend=backend,
            config={"show_progress": False, "prompt_template": "{problem} | {answer}"},
        )
        outputs = operator.run([("2+2", "short reasoning")])
        assert len(outputs) == 1
        assert outputs[0]["instruction"] == "2+2"
        assert outputs[0]["response"] == "long: 2+2 | short reasoning"
        assert outputs[0]["original_response"] == "short reasoning"

    def test_default_prompt_template(self):
        backend = FakeBackend(response_template="x")
        operator = CoTShort2LongOperator(backend=backend, config={"show_progress": False})
        assert operator.prompt_template == _COT_SHORT2LONG_PROMPT

    def test_metadata_and_step_count(self):
        backend = FakeBackend(response_template="step1\n\nstep2\n\nstep3")
        operator = CoTShort2LongOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run([("2+2", "short")])
        assert len(outputs) == 1
        assert outputs[0]["step_count"] == 3
        assert "original_tokens" in outputs[0]
        assert "extended_tokens" in outputs[0]
        assert "expansion_ratio" in outputs[0]


class TestCoTRVCDScorer:
    def _make_backend(self, response: str):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            request = GenerationRequest(instruction=messages[-1]["content"])
            return GenerationResult(request=request, response=response, model="fake")

        backend.generate.side_effect = fake_generate
        return backend

    def test_scores_merged_into_rows(self):
        backend = self._make_backend("<score>5</score>")
        scorer = CoTRVCDScorer(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        rows = [
            {"instruction": "2+2", "response": "4"},
            {"instruction": "3+3", "response": "6"},
        ]
        scored = scorer.run(rows)
        assert len(scored) == 2
        for row in scored:
            assert row["reasoning_verbosity"] == 5
            assert row["cognitive_difficulty"] == 5
            assert row["logical_correctness"] is True

    def test_respects_prompts_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.yaml"
            prompts_path.write_text(
                yaml.safe_dump(
                    {"reasoning_verbosity": "Custom prompt for RV."},
                    allow_unicode=True,
                )
            )
            backend = self._make_backend("<score>5</score>")
            scorer = CoTRVCDScorer(
                backend=backend,
                config={
                    "show_progress": False,
                    "max_workers": 1,
                    "prompts_file": str(prompts_path),
                },
            )
            assert scorer.evaluator.prompts["reasoning_verbosity"] == "Custom prompt for RV."


class TestCoTRVCDMixer:
    def _make_row(self, instruction, response, rv, cd, correctness):
        return {
            "instruction": instruction,
            "response": response,
            "reasoning_verbosity": rv,
            "cognitive_difficulty": cd,
            "logical_correctness": correctness,
        }

    def test_sft_selects_closest_to_target(self):
        rows = [
            self._make_row("easy", "a", rv=2, cd=1, correctness=1),
            self._make_row("easy", "b", rv=7, cd=1, correctness=1),
            self._make_row("hard", "c", rv=8, cd=8, correctness=1),
        ]
        mixer = CoTRVCDMixer(config={"mode": "sft", "cd_bins": [0, 3, 6, 10], "samples_per_bin": 1})
        selected = mixer.run(rows)
        assert len(selected) == 2
        # Low CD bin target is low RV, so row with rv=2 is chosen.
        assert selected[0]["response"] == "a"
        # High CD bin target is high RV, so row with rv=8 is chosen.
        assert selected[1]["response"] == "c"

    def test_sft_filters_by_correctness(self):
        rows = [
            self._make_row("easy", "a", rv=2, cd=1, correctness=0),
            self._make_row("easy", "b", rv=3, cd=1, correctness=1),
        ]
        mixer = CoTRVCDMixer(config={"mode": "sft", "cd_bins": [0, 3, 6, 10], "samples_per_bin": 1})
        selected = mixer.run(rows)
        assert len(selected) == 1
        assert selected[0]["response"] == "b"


class TestCoTEvaluator:
    def _make_backend(self, response: str):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            request = GenerationRequest(instruction=messages[-1]["content"])
            return GenerationResult(request=request, response=response, model="fake")

        backend.generate.side_effect = fake_generate
        return backend

    def test_evaluates_all_metrics(self):
        backend = self._make_backend("<score>8</score>")
        evaluator = CoTEvaluator(
            backend=backend,
            config={"show_progress": False, "max_workers": 1},
        )
        results = evaluator.run([{"instruction": "2+2", "output": "4"}])
        assert len(results) == 1
        assert results[0]["reasoning_verbosity"] == 8
        assert results[0]["cognitive_difficulty"] == 8
        assert results[0]["logical_correctness"] is True

    def test_logical_correctness_false(self):
        backend = self._make_backend("<score>0</score>")
        evaluator = CoTEvaluator(
            backend=backend,
            config={"metrics": ["logical_correctness"], "show_progress": False, "max_workers": 1},
        )
        results = evaluator.run([{"instruction": "2+2", "output": "5"}])
        assert results[0]["logical_correctness"] is False

    def test_aggregate(self):
        backend = self._make_backend("<score>6</score>")
        evaluator = CoTEvaluator(
            backend=backend,
            config={"metrics": ["reasoning_verbosity"], "show_progress": False, "max_workers": 1},
        )
        results = evaluator.run(
            [
                {"instruction": "a", "output": "b"},
                {"instruction": "c", "output": "d"},
            ]
        )
        aggregates = evaluator.aggregate(results)
        assert aggregates["reasoning_verbosity"] == 6.0

    def test_extract_score(self):
        from easydistill.eval.base import _extract_score

        assert _extract_score("Score: <score>7</score>") == 7
        assert _extract_score("No score") is None


class TestCoTDistillationPipeline:
    def _make_backend(self, response: str):
        backend = MagicMock()

        def fake_generate(messages, **kwargs):
            prompt = messages[-1]["content"]
            req = GenerationRequest(instruction=prompt)
            return GenerationResult(request=req, response=response, model="fake")

        backend.generate.side_effect = fake_generate
        return backend

    def test_generate_and_build_sft(self):
        backend = self._make_backend(
            "<|begin_of_thought|>think<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"
        )
        pipeline = CoTDistillationPipeline(
            backend=backend,
            pipeline_config=[
                {"stage": "cot_distill", "config": {"max_workers": 1}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
        )
        data = [{"problem": "What is 2+2?"}]
        results = pipeline.run_with_data(data)
        assert len(results) == 1
        messages = results[0]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"
        assert messages[1]["role"] == "assistant"
        assert "4" in messages[1]["content"]

    def test_quality_filter_keeps_high_scores(self):
        backend = self._make_backend("<score>8</score>")
        pipeline = CoTDistillationPipeline(
            backend=backend,
            pipeline_config=[
                {
                    "stage": "cot_distill",
                    "config": {"max_workers": 1, "prompt_template": "{problem}"},
                },
                {"stage": "cot_eval", "config": {"max_workers": 1}},
                {
                    "stage": "quality_filter",
                    "config": {
                        "min_scores": {"reasoning_verbosity": 6},
                        "require_all_metrics": False,
                    },
                },
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            eval_config={"metrics": ["reasoning_verbosity"], "max_workers": 1},
        )
        data = [{"problem": "What is 2+2?"}]
        results = pipeline.run_with_data(data)
        assert len(results) == 1

    def test_last_stage_must_be_build_sft(self):
        backend = self._make_backend("x")
        try:
            CoTDistillationPipeline(
                backend=backend,
                pipeline_config=[{"stage": "cot_distill", "config": {}}],
                dataset_config={"input_path": "dummy.jsonl"},
            )
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "build_sft" in str(e)

    def test_rvcd_score_stage(self):
        backend = self._make_backend("<score>5</score>")
        pipeline = CoTDistillationPipeline(
            backend=backend,
            pipeline_config=[
                {
                    "stage": "cot_distill",
                    "config": {"max_workers": 1, "prompt_template": "{problem}"},
                },
                {"stage": "cot_rvcd_score", "config": {"max_workers": 1}},
                {"stage": "cot_mix_by_rv_cd", "config": {"mode": "sft"}},
                {"stage": "build_sft", "config": {}},
            ],
            dataset_config={"input_path": "dummy.jsonl"},
            eval_config={"metrics": ["reasoning_verbosity"], "max_workers": 1},
        )
        data = [{"problem": "What is 2+2?"}]
        results = pipeline.run_with_data(data)
        assert len(results) == 1
