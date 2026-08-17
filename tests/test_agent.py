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

"""Unit tests for agent distillation operators and pipeline."""

import sys
from unittest.mock import patch

import pytest

from easydistill.cli.main import _JOB_DESCRIPTIONS, _JOB_DISPATCH
from easydistill.data.models import GenerationResult
from easydistill.operators.agent import (
    AgentFuzzyTaskOperator,
    AgentRubricOperator,
    AgentTaskSynthesisOperator,
    AgentToolCheckOperator,
    AgentTrajectoryOperator,
)
from easydistill.pipeline.agent_distillation import (
    AgentDistillationPipeline,
    _run_agent_prompt_stage,
    run_build_agent_preference_stage,
    run_build_agent_sft_stage,
)
from easydistill.utils import validate_config

from ._fake_backend import FakeBackend


class _FixedBackend(FakeBackend):
    """Fake backend that always returns a fixed response string."""

    def __init__(self, response: str):
        super().__init__()
        self._response = response

    def generate(self, messages, model_id=None, temperature=0.7, max_tokens=2048, **kwargs):
        from easydistill.backends.utils import build_generation_request

        return GenerationResult(
            request=build_generation_request(messages),
            response=self._response,
            model=model_id or "fake",
        )


class _QueuedBackend(FakeBackend):
    """Fake backend that returns queued responses in order."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._index = 0

    def generate(self, messages, model_id=None, temperature=0.7, max_tokens=2048, **kwargs):
        response_text = self._responses[self._index] if self._index < len(self._responses) else ""
        self._index += 1
        from easydistill.backends.utils import build_generation_request

        return GenerationResult(
            request=build_generation_request(messages),
            response=response_text,
            model=model_id or "fake",
        )


class TestAgentTaskSynthesisOperator:
    def test_parses_xml_tags(self):
        response = (
            "<task>Plan a local concert for Afrikaans music.</task>"
            '<tools>[{"name": "search_venues", "description": "Search venues"}]</tools>'
            "<workflow>1. Find venues 2. Book artists 3. Promote</workflow>"
            "<restriction>Stay within the stated budget.</restriction>"
        )
        backend = _FixedBackend(response)
        operator = AgentTaskSynthesisOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(["Afrikaans music fan"])
        assert len(outputs) == 1
        assert outputs[0]["task"] == "Plan a local concert for Afrikaans music."
        assert outputs[0]["tools"] == [{"name": "search_venues", "description": "Search venues"}]
        assert outputs[0]["workflow"] == "1. Find venues 2. Book artists 3. Promote"
        assert outputs[0]["restriction"] == "Stay within the stated budget."

    def test_skips_incomplete_result(self):
        backend = _FixedBackend("<task>Only task</task>")
        operator = AgentTaskSynthesisOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(["background"])
        assert outputs == []


class TestAgentFuzzyTaskOperator:
    def test_parses_fuzzy_task_and_background(self):
        response = (
            "<task>Help organize a small concert on a limited budget.</task>"
            "<background>The user is an Afrikaans music fan with no event-planning "
            "experience and needs step-by-step guidance.</background>"
        )
        backend = _FixedBackend(response)
        operator = AgentFuzzyTaskOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(
            [
                {
                    "task": "Plan concert",
                    "tools": [{"name": "search"}],
                    "workflow": "find venues",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["fuzzy_task"] == "Help organize a small concert on a limited budget."
        assert "Afrikaans music fan" in outputs[0]["task_background"]


class TestAgentToolCheckOperator:
    def test_parses_checked_tools(self):
        response = (
            '<tools>[{"name": "search_venues", "description": "Search venues"}, '
            '{"name": "book_artist", "description": "Book an artist"}]</tools>'
        )
        backend = _FixedBackend(response)
        operator = AgentToolCheckOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(
            [
                {
                    "fuzzy_task": "Plan concert",
                    "tools": [{"name": "search_venues"}],
                }
            ]
        )
        assert len(outputs) == 1
        assert len(outputs[0]["checked_tools"]) == 2
        assert outputs[0]["checked_tools"][0]["name"] == "search_venues"

    def test_skips_non_list_tools(self):
        backend = _FixedBackend('<tools>{"name": "x"}</tools>')
        operator = AgentToolCheckOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run([{"fuzzy_task": "Plan concert", "tools": []}])
        assert outputs == []


class TestAgentPromptStageMerge:
    def test_merge_by_row_idx_skips_failed_middle_parse(self):
        """Parse failures must not shift later outputs onto earlier rows."""
        good = (
            "<task>Plan a local concert for Afrikaans music.</task>"
            '<tools>[{"name": "search_venues", "description": "Search venues"}]</tools>'
            "<workflow>1. Find venues</workflow>"
            "<restriction>Budget.</restriction>"
        )
        bad = "<task>Only task</task>"  # incomplete parse
        backend = _QueuedBackend([good, bad, good])
        data = [
            {"id": "row_0", "background": "A"},
            {"id": "row_1", "background": "B"},
            {"id": "row_2", "background": "C"},
        ]
        merged = _run_agent_prompt_stage(
            backend=backend,
            stage_name="agent_task_synthesis",
            stage_config={"show_progress": False},
            data=data,
        )
        assert len(merged) == 2
        assert merged[0]["id"] == "row_0"
        assert merged[0]["task"] == "Plan a local concert for Afrikaans music."
        assert merged[1]["id"] == "row_2"
        assert merged[1]["background"] == "C"


class TestAgentTrajectoryOperator:
    def test_generates_trajectory_with_tool_call_and_answer(self, monkeypatch):
        # Speed up retry backoff during this test.
        monkeypatch.setattr("easydistill.operators.agent.trajectory.time.sleep", lambda _: None)
        responses = [
            # Solve: first reasoning step with a tool call.
            'I will search for venues.<tool_call>{"name": "search_venues", '
            '"arguments": {"city": "Cape Town"}}</tool_call>',
            # Mock tool: simulated response.
            "<tool_response_start>Found 3 suitable venues.</tool_response_start>",
            # Solve: final answer.
            "<answer>Book the Community Hall for the concert.</answer>",
        ]
        backend = _QueuedBackend(responses)
        operator = AgentTrajectoryOperator(
            backend=backend,
            config={"max_steps": 5, "repeat_times": 1},
        )
        outputs = operator.run(
            [
                {
                    "id": "task_1",
                    "fuzzy_task": "Plan a concert in Cape Town.",
                    "task_background": "Afrikaans music fan.",
                    "checked_tools": [{"name": "search_venues"}],
                    "restriction": "Stay within budget.",
                }
            ]
        )
        assert len(outputs) == 1
        row = outputs[0]
        assert row["solution_id"] == "task_1_solution_1.json"
        assert row["task_finished"] == "Terminated"
        trajectory = row["trajectory"]
        assert any(
            m.get("role") == "assistant" and "<answer>" in m.get("content", "") for m in trajectory
        )
        assert any(
            m.get("role") == "assistant" and "<tool_call>" in m.get("content", "")
            for m in trajectory
        )

    def test_respects_max_steps(self, monkeypatch):
        monkeypatch.setattr("easydistill.operators.agent.trajectory.time.sleep", lambda _: None)
        backend = FakeBackend(response_template="thinking...")
        operator = AgentTrajectoryOperator(
            backend=backend,
            config={"max_steps": 2, "repeat_times": 1},
        )
        outputs = operator.run(
            [
                {
                    "id": "task_2",
                    "fuzzy_task": "Plan a concert.",
                    "task_background": "Fan.",
                    "checked_tools": [],
                    "restriction": "None.",
                }
            ]
        )
        assert len(outputs) == 1
        assert outputs[0]["task_finished"] == "Terminated"

    def test_respects_max_tool_calls(self, monkeypatch):
        monkeypatch.setattr("easydistill.operators.agent.trajectory.time.sleep", lambda _: None)
        responses = [
            # Solve: keep calling tools without ever answering.
            'Need a tool.<tool_call>{"name": "search", "arguments": {}}</tool_call>',
            # Mock tool: simulated response (new_bg=NO so history does not grow).
            "<tool_response_start>result</tool_response_start><new_bg_introduced>NO</new_bg_introduced>",
        ]
        backend = _QueuedBackend(responses * 10)
        operator = AgentTrajectoryOperator(
            backend=backend,
            config={"max_steps": 100, "max_tool_calls": 2, "repeat_times": 1},
        )
        outputs = operator.run(
            [
                {
                    "id": "task_3",
                    "fuzzy_task": "Plan a concert.",
                    "task_background": "Fan.",
                    "checked_tools": [{"name": "search"}],
                    "restriction": "None.",
                }
            ]
        )
        assert len(outputs) == 1
        trajectory = outputs[0]["trajectory"]
        assert any(
            m.get("role") == "assistant"
            and "maximum tool call limit" in m.get("content", "").lower()
            for m in trajectory
        )


class TestAgentRubricOperator:
    def test_selects_best_solution(self):
        response = (
            "<alignment_check>The trajectories align with the task.</alignment_check>"
            "<rubrics>1. Correctness 2. Efficiency</rubrics>"
            "<final>Solution 1 is best.</final>"
            "<best_solution>task_1_solution_1.json</best_solution>"
        )
        backend = _FixedBackend(response)
        operator = AgentRubricOperator(backend=backend, config={"show_progress": False})
        trajectories = [
            {
                "id": "task_1",
                "solution_id": "task_1_solution_1.json",
                "trajectory": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Plan concert."},
                    {"role": "assistant", "content": "<answer>Done</answer>"},
                ],
                "fuzzy_task": "Plan concert.",
                "task_background": "Fan.",
                "restriction": "Budget.",
                "workflow": "Plan.",
                "checked_tools": [],
            },
            {
                "id": "task_1",
                "solution_id": "task_1_solution_2.json",
                "trajectory": [
                    {"role": "user", "content": "Plan concert."},
                    {"role": "assistant", "content": "<answer>Done2</answer>"},
                ],
                "fuzzy_task": "Plan concert.",
                "task_background": "Fan.",
                "restriction": "Budget.",
                "workflow": "Plan.",
                "checked_tools": [],
            },
        ]
        outputs = operator.run(trajectories)
        assert len(outputs) == 1
        assert outputs[0]["best_solution_id"] == "task_1_solution_1.json"
        assert "rubrics" in outputs[0]
        assert len(outputs[0]["trajectories"]) == 2

    def test_discards_task_when_alignment_check_fails(self):
        backend = _FixedBackend("<alignment_check>discard</alignment_check>")
        operator = AgentRubricOperator(backend=backend, config={"show_progress": False})
        outputs = operator.run(
            [
                {
                    "id": "task_1",
                    "solution_id": "task_1_solution_1.json",
                    "trajectory": [],
                    "fuzzy_task": "Plan concert.",
                    "task_background": "Fan.",
                    "restriction": "Budget.",
                }
            ]
        )
        assert outputs == []


class TestAgentSFTAndPreferenceBuilders:
    def test_build_sft_from_best_trajectory(self):
        row = {
            "id": "task_1",
            "fuzzy_task": "Plan concert.",
            "restriction": "Budget.",
            "workflow": "Plan.",
            "best_solution_id": "task_1_solution_1.json",
            "trajectories": [
                {
                    "solution_id": "task_1_solution_1.json",
                    "task_finished": "Terminated",
                    "trajectory": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "Plan concert."},
                        {"role": "assistant", "content": "<answer>Done</answer>"},
                    ],
                }
            ],
        }
        samples = run_build_agent_sft_stage([row])
        assert len(samples) == 1
        assert samples[0]["messages"][0]["role"] == "system"
        assert samples[0]["metadata"]["task_id"] == "task_1"

    def test_build_sft_without_rubrics_uses_first_trajectory(self):
        row = {
            "id": "task_1",
            "fuzzy_task": "Plan concert.",
            "restriction": "Budget.",
            "workflow": "Plan.",
            "trajectories": [
                {
                    "solution_id": "task_1_solution_1.json",
                    "task_finished": "Terminated",
                    "trajectory": [
                        {"role": "user", "content": "Plan concert."},
                        {"role": "assistant", "content": "<answer>Done</answer>"},
                    ],
                }
            ],
        }
        samples = run_build_agent_sft_stage([row], use_rubrics=False)
        assert len(samples) == 1
        assert samples[0]["metadata"]["task_id"] == "task_1"

    def test_build_preference_from_rubrics(self):
        row = {
            "id": "task_1",
            "fuzzy_task": "Plan concert.",
            "task": "Plan concert.",
            "restriction": "Budget.",
            "best_solution_id": "task_1_solution_1.json",
            "trajectories": [
                {
                    "solution_id": "task_1_solution_1.json",
                    "trajectory": [{"role": "assistant", "content": "Good"}],
                },
                {
                    "solution_id": "task_1_solution_2.json",
                    "trajectory": [{"role": "assistant", "content": "Bad"}],
                },
            ],
        }
        result = run_build_agent_preference_stage(
            [row], {"system_prompt": "sys", "format": "openai_messages"}
        )
        assert isinstance(result, list)
        assert len(result) == 1
        sample = result[0]
        assert sample["prompt"][0]["content"] == "sys"
        assert sample["prompt"][1]["content"] == "Plan concert."
        assert sample["chosen"][0]["content"] == '[{"role": "assistant", "content": "Good"}]'
        assert sample["rejected"][0]["content"] == '[{"role": "assistant", "content": "Bad"}]'


class TestAgentDistillationPipeline:
    def test_dispatches_prompt_stages(self):
        from easydistill.pipeline.agent_distillation import _run_agent_prompt_stage

        backend = _FixedBackend(
            '<task>T</task><tools>[{"name": "n"}]</tools>'
            "<workflow>W</workflow><restriction>R</restriction>"
        )
        data = [{"id": "1", "persona": "fan"}]
        result = _run_agent_prompt_stage(
            backend, "agent_task_synthesis", {"show_progress": False}, data
        )
        assert len(result) == 1
        assert result[0]["task"] == "T"

    def test_last_stage_validation(self):
        backend = FakeBackend(response_template="")
        with pytest.raises(ValueError, match="last pipeline stage"):
            AgentDistillationPipeline(
                backend=backend,
                pipeline_config=[{"stage": "agent_task_synthesis", "config": {}}],
                dataset_config={"input_path": "examples/seed_agent_personas.jsonl"},
            )


class TestAgentConfigValidation:
    def test_agent_section_valid_and_extra_keys_rejected(self):
        cfg = {
            "job_type": "agent_distill",
            "backend": {"type": "openai", "api_key": "key"},
            "dataset": {"input_path": "data.jsonl"},
            "agent": {"max_steps": 15, "repeat_times": 3},
        }
        validated = validate_config(cfg)
        assert validated["agent"]["max_steps"] == 15
        assert validated["agent"]["repeat_times"] == 3

        cfg["agent"]["unknown_key"] = True
        with pytest.raises(ValueError):
            validate_config(cfg)


class TestAgentCLI:
    def test_agent_distill_in_dispatch(self):
        assert "agent_distill" in _JOB_DISPATCH
        assert "agent_distill" in _JOB_DESCRIPTIONS

    def test_list_jobs_includes_agent_distill(self, capsys):
        from easydistill.cli.main import main

        with patch.object(sys, "argv", ["easydistill", "--list-jobs"]):
            main()
        captured = capsys.readouterr()
        assert "agent_distill" in captured.out
