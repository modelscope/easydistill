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

"""Unit tests for operators."""

from unittest.mock import MagicMock

import pytest

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators import SFTDatasetBuilder, TextGenerationOperator
from tests._fake_backend import FakeBackend


class TestTextGenerationOperator:
    def test_generates_responses_for_requests(self):
        backend = FakeBackend(response_template="Ans: {}")
        operator = TextGenerationOperator(backend=backend, config={"show_progress": False})
        requests = [
            GenerationRequest(id="1", instruction="Q1"),
            GenerationRequest(id="2", instruction="Q2"),
        ]
        results = operator.run(requests)
        assert len(results) == 2
        assert results[0].response == "Ans: Q1"
        assert results[1].response == "Ans: Q2"
        assert results[0].request.id == "1"

    def test_uses_system_prompt(self):
        backend = FakeBackend()
        operator = TextGenerationOperator(
            backend=backend,
            config={"system_prompt": "You are a teacher.", "show_progress": False},
        )
        requests = [GenerationRequest(instruction="Q")]
        results = operator.run(requests)
        assert results[0].request.system_prompt == "You are a teacher."

    def test_none_temperature_defaults_to_default(self):
        """Explicitly configured None temperature must not crash."""
        backend = MagicMock()
        backend.generate.return_value = GenerationResult(
            request=GenerationRequest(instruction="Q"),
            response="OK",
            model="mock",
        )
        operator = TextGenerationOperator(
            backend=backend,
            config={"temperature": None, "show_progress": False},
        )
        requests = [GenerationRequest(instruction="Q")]
        operator.run(requests)
        assert backend.generate.call_args.kwargs["temperature"] == 0.7

    def test_skips_failed_requests_by_default(self):
        backend = MagicMock()
        backend.generate.side_effect = [
            RuntimeError("boom"),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="OK",
                model="mock",
            ),
        ]
        operator = TextGenerationOperator(
            backend=backend,
            config={"show_progress": False, "retry_attempts": 1},
        )
        requests = [
            GenerationRequest(id="1", instruction="Q1"),
            GenerationRequest(id="2", instruction="Q2"),
        ]
        results = operator.run(requests)
        assert len(results) == 1
        assert results[0].response == "OK"

    def test_raise_on_error(self):
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("boom")
        operator = TextGenerationOperator(
            backend=backend,
            config={"show_progress": False, "raise_on_error": True, "retry_attempts": 1},
        )
        with pytest.raises(RuntimeError):
            operator.run([GenerationRequest(instruction="Q")])

    def test_concurrent_generation_preserves_order(self):
        backend = FakeBackend(response_template="Ans: {}")
        operator = TextGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 3},
        )
        requests = [GenerationRequest(id=str(i), instruction=f"Q{i}") for i in range(10)]
        results = operator.run(requests)
        assert len(results) == 10
        # Results should be returned in input order.
        for i, result in enumerate(results):
            assert result.request.id == str(i)
            assert result.response == f"Ans: Q{i}"

    def test_concurrent_generation_with_raise_on_error_returns_partial(self):
        """When raise_on_error=True and a task fails concurrently, other results survive."""
        backend = MagicMock()
        backend.generate.side_effect = [
            GenerationResult(
                request=GenerationRequest(instruction="Q0"),
                response="A0",
                model="mock",
            ),
            RuntimeError("boom"),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="A2",
                model="mock",
            ),
        ]
        operator = TextGenerationOperator(
            backend=backend,
            config={
                "show_progress": False,
                "max_workers": 2,
                "retry_attempts": 0,
                "raise_on_error": True,
            },
        )
        requests = [
            GenerationRequest(id=str(i), instruction=f"Q{i}") for i in range(3)
        ]
        # Previously this would crash the whole loop via uncaught future.result().
        results = operator.run(requests)
        assert len(results) == 2
        assert results[0].request.id == "0"
        assert results[1].request.id == "2"

    def test_concurrent_generation_skips_failed_requests(self):
        backend = MagicMock()
        # Fail every other request.
        backend.generate.side_effect = [
            GenerationResult(
                request=GenerationRequest(instruction="Q0"),
                response="A0",
                model="mock",
            ),
            RuntimeError("boom"),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="A2",
                model="mock",
            ),
            RuntimeError("boom"),
        ]
        operator = TextGenerationOperator(
            backend=backend,
            config={"show_progress": False, "max_workers": 2, "retry_attempts": 1},
        )
        requests = [GenerationRequest(id=str(i), instruction=f"Q{i}") for i in range(4)]
        results = operator.run(requests)
        assert len(results) == 2
        assert results[0].request.id == "0"
        assert results[1].request.id == "2"

    def test_retries_transient_failures(self, monkeypatch):
        backend = MagicMock()
        backend.generate.side_effect = [
            TimeoutError("boom"),
            GenerationResult(
                request=GenerationRequest(instruction="Q"),
                response="OK",
                model="mock",
            ),
        ]
        monkeypatch.setattr("easydistill.operators.generation.time.sleep", lambda _: None)
        operator = TextGenerationOperator(
            backend=backend,
            config={"show_progress": False, "retry_attempts": 2, "retry_backoff_base": 0.1},
        )
        results = operator.run([GenerationRequest(id="1", instruction="Q")])
        assert len(results) == 1
        assert results[0].response == "OK"
        assert backend.generate.call_count == 2


class TestSFTDatasetBuilder:
    def test_builds_samples(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1", system_prompt="SYS"),
                response="A1",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="A2",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={})
        samples = builder.run(results)
        assert len(samples) == 2
        assert samples[0].messages[0].role == "system"
        assert samples[0].messages[1].content == "Q1"
        assert samples[0].messages[2].content == "A1"
        assert samples[1].messages[0].role == "user"  # no system prompt

    def test_skips_empty_responses(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="   ",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="A2",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={"skip_empty": True})
        samples = builder.run(results)
        assert len(samples) == 1
        assert samples[0].messages[0].content == "Q2"

    def test_min_length_filter(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="short",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="this is a much longer response",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={"min_length": 10})
        samples = builder.run(results)
        assert len(samples) == 1
        assert samples[0].messages[0].content == "Q2"

    def test_max_length_filter(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="this is a very long response that should be filtered out",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="short",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={"max_length": 20})
        samples = builder.run(results)
        assert len(samples) == 1
        assert samples[0].messages[0].content == "Q2"

    def test_dedup_by_instruction(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="A1",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="A1-variant",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q2"),
                response="A2",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={"dedup_key": "instruction"})
        samples = builder.run(results)
        assert len(samples) == 2
        assert samples[0].messages[0].content == "Q1"
        assert samples[1].messages[0].content == "Q2"

    def test_dedup_by_instruction_and_response(self):
        results = [
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="A1",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="A1",
                model="teacher",
            ),
            GenerationResult(
                request=GenerationRequest(instruction="Q1"),
                response="A2",
                model="teacher",
            ),
        ]
        builder = SFTDatasetBuilder(config={"dedup_key": "instruction_response"})
        samples = builder.run(results)
        assert len(samples) == 2
