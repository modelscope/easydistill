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

"""Unit tests for instruction-following evaluator."""

from unittest.mock import MagicMock

import pytest

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval import InstructionFollowingEvaluator


class TestInstructionFollowingEvaluator:
    def test_evaluates_all_metrics(self):
        backend = MagicMock()

        metric_order = ["informativeness", "helpfulness", "generalization", "correctness"]

        def fake_generate(messages, **kwargs):
            prompt = messages[-1]["content"]
            metric = None
            for m in metric_order:
                if m in prompt:
                    metric = m
                    break
            response = (
                "<score>1</score>" if metric == "correctness" else "<score>7</score>"
            )
            return GenerationResult(
                request=GenerationRequest(instruction=prompt, metadata={"metric": metric}),
                response=response,
                model="fake",
            )

        backend.generate.side_effect = fake_generate
        evaluator = InstructionFollowingEvaluator(backend=backend, config={"show_progress": False})
        samples = [{"instruction": "What is 2+2?", "output": "4"}]
        results = evaluator.run(samples)

        assert len(results) == 1
        assert results[0]["informativeness"] == 7
        assert results[0]["helpfulness"] == 7
        assert results[0]["generalization"] == 7
        assert results[0]["correctness"] is True

    def test_aggregate(self):
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="prompt", metadata={"metric": "informativeness"}),
            response="<score>5</score>",
            model="fake",
        )
        evaluator = InstructionFollowingEvaluator(
            backend=backend,
            config={"metrics": ["informativeness"], "show_progress": False},
        )
        results = evaluator.run(
            [
                {"instruction": "a", "output": "b"},
                {"instruction": "c", "output": "d"},
            ]
        )
        aggregates = evaluator.aggregate(results)
        assert aggregates["informativeness"] == 5.0

    def test_extract_score(self):
        from easydistill.eval.base import _extract_score

        assert _extract_score("The score is <score>7</score>.") == 7
        assert _extract_score("No score here") is None

    def test_extract_score_boolean_anywhere(self):
        from easydistill.eval.base import _extract_score

        assert _extract_score("After analysis, the response is correct.") == 1
        assert _extract_score("I conclude the answer is incorrect.") == 0

    def test_skips_empty_instruction_or_output(self):
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="prompt", metadata={"metric": "informativeness"}),
            response="<score>5</score>",
            model="fake",
        )
        evaluator = InstructionFollowingEvaluator(
            backend=backend,
            config={"metrics": ["informativeness"], "show_progress": False},
        )
        results = evaluator.run(
            [
                {"instruction": "a", "output": "b"},
                {"instruction": "", "output": "b"},
                {"instruction": "a", "output": ""},
            ]
        )
        assert len(results) == 1
        assert results[0]["informativeness"] == 5

    def test_strict_mode_raises_on_empty_sample(self):
        backend = MagicMock()
        evaluator = InstructionFollowingEvaluator(
            backend=backend,
            config={
                "metrics": ["informativeness"],
                "show_progress": False,
                "strict_mode": True,
            },
        )
        with pytest.raises(ValueError, match="empty instruction or output"):
            evaluator.run([{"instruction": "a", "output": ""}])

    def test_returns_empty_when_all_samples_invalid(self):
        backend = MagicMock()
        evaluator = InstructionFollowingEvaluator(
            backend=backend,
            config={"metrics": ["informativeness"], "show_progress": False},
        )
        results = evaluator.run([{"instruction": "", "output": ""}])
        assert results == []
