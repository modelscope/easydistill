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

"""Unit tests for the search-agent operators and pipeline."""

import json
from typing import Any, Dict, List, Optional

import pytest

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.search_agent import (
    SearchTaskEvolverOperator,
    SearchToolset,
    SearchTrajectoryOperator,
    judge_trajectory,
    run_quality_gate,
    solve_search_task,
)
from easydistill.pipeline.search_agent_distill import (
    SearchAgentDistillationPipeline,
    _normalize_seed_rows,
    run_build_search_sft_stage,
    run_judge_filter_stage,
)

SEED_QUESTION = "What genre is Our Town by Thornton Wilder?"
SEED_ANSWER = "drama"
REWRITTEN_QUESTION = (
    "What genre is Our Town by the author who wrote the 1927 novel " "The Bridge of San Luis Rey?"
)
BRIDGE_FACT = (
    "Thornton Wilder wrote the 1927 novel The Bridge of San Luis Rey, "
    "which won the Pulitzer Prize for the Novel in 1928."
)

SEARCH_RESULTS = {
    "results": [
        {
            "title": "Thornton Wilder - Encyclopedia",
            "url": "https://example.com/wilder",
            "snippet": BRIDGE_FACT + " He was an American playwright and novelist.",
        }
    ]
}


class ScriptedSearchBackend(ModelBackend):
    """Routes responses by prompt markers to emulate every pipeline role."""

    def __init__(self, judge_difficulty: str = "good", judge_correct: bool = True):
        self.judge_difficulty = judge_difficulty
        self.judge_correct = judge_correct
        self.calls: List[str] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        system = ""
        if messages and messages[0].get("role") == "system":
            system = str(messages[0].get("content", ""))
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        response = self._route(system, last_user, messages)
        return GenerationResult(
            request=GenerationRequest(instruction=last_user or "[empty]"),
            response=response,
            model=model_id or "scripted",
        )

    def _route(self, system: str, last_user: str, messages: List[Dict[str, Any]]) -> str:
        if "You are Google Search" in system:
            self.calls.append("search_sim")
            return json.dumps(SEARCH_RESULTS)
        if "mock web browser" in system:
            self.calls.append("browser_sim")
            return json.dumps(
                {
                    "url": "https://example.com/wilder",
                    "title": "Thornton Wilder",
                    "content": BRIDGE_FACT,
                }
            )
        if "strategic decision-making agent" in system:
            self.calls.append("strategist")
            decision = "FINALIZE" if "Difficulty Level: good" in last_user else "EXPAND"
            return json.dumps(
                {
                    "problem_diagnosis": "too easy",
                    "root_cause": "single hop",
                    "suggested_fix": "expand the author entity",
                    "decision": decision,
                    "expand_target": "Thornton Wilder",
                    "rollback_reason": "",
                    "confidence": "high",
                    "reasoning": "scripted",
                }
            )
        if "generating atomic QA" in system:
            self.calls.append("atomic_qa")
            return json.dumps(
                {
                    "atomic_question": "Who wrote the 1927 novel The Bridge of San Luis Rey?",
                    "atomic_answer": "Thornton Wilder",
                    "bridge_entity": "The Bridge of San Luis Rey",
                    "relationship": "author",
                    "bridge_fact": BRIDGE_FACT,
                    "reasoning": "scripted",
                }
            )
        if "rewriting questions" in system:
            self.calls.append("rewrite")
            return json.dumps(
                {
                    "replacement_phrase": (
                        "the author who wrote the 1927 novel The Bridge of San Luis Rey"
                    ),
                    "new_question": REWRITTEN_QUESTION,
                }
            )
        if "question quality gate" in system:
            self.calls.append("quality_gate")
            return json.dumps({"pass": True, "failed_checks": [], "rollback_reason": ""})
        if "expert evaluator for multi-hop" in system:
            self.calls.append("judge")
            return json.dumps(
                {
                    "is_correct": self.judge_correct,
                    "total_steps": 6,
                    "difficulty_level": self.judge_difficulty,
                    "has_shortcut": False,
                    "recommended_action": "FINALIZE",
                    "reason": "scripted",
                    "suggestions": [],
                }
            )
        if "answer evaluator" in system:
            self.calls.append("answer_check")
            return json.dumps({"equivalent": True})
        if "designed to solve tasks" in system:
            self.calls.append("solver")
            has_tool_response = any(
                "<tool_response>" in str(m.get("content", ""))
                for m in messages
                if m.get("role") == "user"
            )
            if not has_tool_response:
                return (
                    "I need to search for this first.\n"
                    '<tool_call>{"name": "web_search", '
                    '"arguments": {"query": "Our Town genre"}}</tool_call>'
                )
            return (
                "Based on the search results, the genre is drama "
                "(evidence: 'three-act play').\n<answer>drama</answer>"
            )
        raise AssertionError(f"Unrouted prompt. system={system[:80]!r}")


@pytest.fixture()
def backend():
    return ScriptedSearchBackend()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_toolset_mock_search_and_browse(backend):
    toolset = SearchToolset(backend, {"tools": {"mode": "mock", "num_results": 3}})
    result = toolset.search("Thornton Wilder")
    assert result["results"][0]["url"] == "https://example.com/wilder"

    history = [
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name": "web_search", '
                '"arguments": {"query": "Thornton Wilder"}}</tool_call>'
            ),
        },
        {"role": "user", "content": f"<tool_response>{json.dumps(SEARCH_RESULTS)}</tool_response>"},
    ]
    response = toolset.execute("web_browse", {"url": "https://example.com/wilder"}, history)
    page = json.loads(response)
    assert page["title"] == "Thornton Wilder"


def test_toolset_unknown_tool_and_missing_args(backend):
    toolset = SearchToolset(backend, {})
    assert "Unknown tool" in toolset.execute("web_upload", {}, [])
    assert "Missing 'query'" in toolset.execute("web_search", {}, [])
    assert "Missing 'url'" in toolset.execute("web_browse", {}, [])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def test_solve_search_task_produces_react_trajectory(backend):
    toolset = SearchToolset(backend, {})
    history = solve_search_task(backend, {}, SEED_QUESTION, toolset, max_steps=5)
    roles = [m["role"] for m in history]
    assert roles[0] == "system" and roles[1] == "user"
    assert any("<tool_call>" in m["content"] for m in history if m["role"] == "assistant")
    assert any("<tool_response>" in m["content"] for m in history if m["role"] == "user")
    assert "<answer>drama</answer>" in history[-1]["content"]


def test_trajectory_operator_marks_correct_runs(backend):
    operator = SearchTrajectoryOperator(
        backend=backend, config={"repeat_times": 2, "max_steps": 5, "max_workers": 1}
    )
    rows = operator.run([{"id": "t1", "question": SEED_QUESTION, "answer": SEED_ANSWER}])
    assert len(rows) == 1
    trajectories = rows[0]["trajectories"]
    assert len(trajectories) == 2
    assert all(t["is_correct"] for t in trajectories)
    assert all(t["turns"] >= 2 for t in trajectories)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_judge_trajectory_report(backend):
    trajectory = [
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>',
        },
        {"role": "user", "content": "<tool_response>{}</tool_response>"},
        {"role": "assistant", "content": "<answer>drama</answer>"},
    ]
    report = judge_trajectory(backend, {}, SEED_QUESTION, SEED_QUESTION, SEED_ANSWER, trajectory)
    assert report["is_correct"] is True
    assert report["difficulty_level"] == "good"
    assert report["recommended_action"] == "FINALIZE"


def test_quality_gate_fails_open_on_garbage():
    class GarbageBackend(ScriptedSearchBackend):
        def _route(self, system, last_user, messages):
            return "not json at all"

    gate = run_quality_gate(GarbageBackend(), {}, SEED_QUESTION, SEED_QUESTION, SEED_ANSWER)
    assert gate["pass"] is True


# ---------------------------------------------------------------------------
# Task evolver
# ---------------------------------------------------------------------------


def test_evolver_saves_good_candidate(backend):
    operator = SearchTaskEvolverOperator(
        backend=backend,
        config={"max_levels": 3, "max_iterations": 6, "final_eval_runs": 1, "max_workers": 1},
    )
    rows = operator.run([{"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER}])
    assert len(rows) == 1
    candidate = rows[0]
    assert candidate["evolve_status"] == "saved"
    assert candidate["question"] == REWRITTEN_QUESTION
    assert candidate["hops"] == 1
    assert candidate["path"][0]["from_id"] == "Thornton Wilder"
    assert "The Bridge of San Luis Rey" in candidate["entity_set"]
    assert candidate["final_eval"]["accuracy"] == 1.0
    # The loop must have exercised all agent roles.
    assert {"strategist", "atomic_qa", "rewrite", "quality_gate", "judge", "solver"} <= set(
        backend.calls
    )


def test_evolver_filters_bad_difficulty():
    backend = ScriptedSearchBackend(judge_difficulty="too_easy")
    operator = SearchTaskEvolverOperator(
        backend=backend,
        config={"max_levels": 1, "max_iterations": 3, "final_eval_runs": 1, "max_workers": 1},
    )
    rows = operator.run([{"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER}])
    assert rows == []

    operator_keep = SearchTaskEvolverOperator(
        backend=ScriptedSearchBackend(judge_difficulty="too_easy"),
        config={
            "max_levels": 1,
            "max_iterations": 3,
            "final_eval_runs": 1,
            "max_workers": 1,
            "keep_filtered": True,
        },
    )
    rows = operator_keep.run([{"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER}])
    assert len(rows) == 1
    assert rows[0]["evolve_status"] == "filtered"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _make_trajectory_row(is_correct: bool = True, turns: int = 3) -> Dict[str, Any]:
    return {
        "id": "task1",
        "question": REWRITTEN_QUESTION,
        "answer": SEED_ANSWER,
        "hops": 1,
        "difficulty_report": {"difficulty_level": "good"},
        "trajectories": [
            {
                "solution_id": "task1_solution1",
                "trajectory": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<answer>drama</answer>"},
                ],
                "predicted_answer": "drama",
                "is_correct": is_correct,
                "turns": turns,
            }
        ],
    }


def test_seed_normalization_accepts_aliases():
    rows = _normalize_seed_rows(
        [
            {"id": "a", "q": "Q1?", "a_star": "A1"},
            {"instruction": "Q2?", "short_answer": "A2"},
            {"question": "no answer"},
            {"id": "c", "question": "Q3?", "true_answer": ["NYC", "New York City"]},
        ]
    )
    assert len(rows) == 3
    assert rows[0] == {"id": "a", "question": "Q1?", "answer": "A1"}
    assert rows[1]["id"] == "seed_1"
    assert rows[2]["answer"] == "NYC"
    assert rows[2]["answer_aliases"] == ["New York City"]


def test_judge_filter_and_build_sft():
    rows = [_make_trajectory_row(), _make_trajectory_row(is_correct=False)]
    rows[1]["id"] = "task2"
    kept = run_judge_filter_stage(rows, {"require_correct": True})
    assert [r["id"] for r in kept] == ["task1"]

    samples = run_build_search_sft_stage(kept, {"min_length": 1})
    assert len(samples) == 1
    sample = samples[0]
    assert sample["messages"][-1]["content"] == "<answer>drama</answer>"
    assert sample["metadata"]["task_id"] == "task1"
    assert sample["metadata"]["is_correct"] is True
    assert sample["metadata"]["difficulty_report"] == {"difficulty_level": "good"}


def test_pipeline_dispatch_end_to_end(backend):
    pipeline = SearchAgentDistillationPipeline(
        backend=backend,
        pipeline_config=[
            {
                "stage": "search_task_evolve",
                "config": {
                    "max_levels": 3,
                    "max_iterations": 6,
                    "final_eval_runs": 1,
                    "max_workers": 1,
                },
            },
            {
                "stage": "search_trajectory",
                "config": {"repeat_times": 1, "max_steps": 5, "max_workers": 1},
            },
            {"stage": "search_judge_filter", "config": {"require_correct": True}},
            {"stage": "build_sft"},
        ],
        dataset_config={"input_path": "unused.jsonl"},
        sft_config={"min_length": 1},
        search_agent_config={"tools": {"mode": "mock"}},
    )
    samples = pipeline.run_with_data(
        [{"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER}]
    )
    assert len(samples) == 1
    assert samples[0]["metadata"]["question"] == REWRITTEN_QUESTION
    roles = [m["role"] for m in samples[0]["messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "assistant"


def test_pipeline_rejects_wrong_last_stage(backend):
    with pytest.raises(ValueError):
        SearchAgentDistillationPipeline(
            backend=backend,
            pipeline_config=[{"stage": "search_task_evolve"}],
            dataset_config={},
        )


# ---------------------------------------------------------------------------
# Role config / backend parameter passing
# ---------------------------------------------------------------------------


class RecordingBackend(ScriptedSearchBackend):
    """Captures generate() kwargs for parameter-passing assertions."""

    def __init__(self):
        super().__init__()
        self.last_call: Dict[str, Any] = {}

    def generate(self, messages, model_id=None, temperature=0.7, max_tokens=2048, **kwargs):
        self.last_call = {
            "messages": messages,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return super().generate(messages, model_id, temperature, max_tokens, **kwargs)


def test_call_role_passes_role_settings_to_backend():
    from easydistill.operators.search_agent.utils import call_role, resolve_role_config

    backend = RecordingBackend()
    config = {
        "roles": {"judge": {"model_id": "judge-model", "temperature": 0.1, "max_tokens": 512}}
    }
    call_role(backend, config, "judge", "Predicted Answer: x", "answer evaluator system")
    assert backend.last_call["model_id"] == "judge-model"
    assert backend.last_call["temperature"] == 0.1
    assert backend.last_call["max_tokens"] == 512

    # fast_verify falls back to the solver role when unset.
    cfg = resolve_role_config({"roles": {"solver": {"model_id": "s1"}}}, "fast_verify")
    assert cfg["model_id"] == "s1"
    # Unknown roles get generic defaults and never crash.
    cfg = resolve_role_config({}, "strategist")
    assert cfg["model_id"] is None and cfg["no_think"] is False


def test_no_think_suffix_applied_per_role():
    from easydistill.operators.search_agent.solver import build_solver_messages
    from easydistill.operators.search_agent.utils import call_role

    backend = RecordingBackend()
    config = {"roles": {"judge": {"no_think": True}}}
    call_role(backend, config, "judge", "Predicted Answer: x", "answer evaluator system")
    assert backend.last_call["messages"][-1]["content"].endswith("/no_think")

    # Solver: marker lands on the first user prompt only when enabled.
    messages = build_solver_messages("Q?", {"roles": {"solver": {"no_think": True}}})
    assert messages[1]["content"].endswith("/no_think")
    messages = build_solver_messages("Q?", {})
    assert not messages[1]["content"].endswith("/no_think")


# ---------------------------------------------------------------------------
# Real-mode tool plumbing (no network)
# ---------------------------------------------------------------------------


def test_sqlite_cache_roundtrip(tmp_path):
    from easydistill.operators.search_agent.tools import SqliteCache

    cache = SqliteCache(str(tmp_path / "cache.db"))
    assert cache.get("missing") is None
    cache.set("k", "v1")
    cache.set("k", "v2")
    assert cache.get("k") == "v2"


def test_real_search_requires_api_key(tmp_path, monkeypatch):
    from easydistill.operators.search_agent.tools import real_web_search

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="google_api_key"):
        real_web_search({"tools": {}}, "query", 3, None)


def test_real_search_requires_cx(monkeypatch):
    from easydistill.operators.search_agent.tools import real_web_search

    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    monkeypatch.delenv("GOOGLE_CX", raising=False)
    with pytest.raises(ValueError, match="google_cx"):
        real_web_search({"tools": {}}, "query", 3, None)


def test_real_browse_requires_http_url():
    from easydistill.operators.search_agent.tools import real_web_browse

    with pytest.raises(ValueError, match="HTTP\\(S\\) URL"):
        real_web_browse({"tools": {}}, "not-a-url")


def test_toolset_real_mode_uses_cache(tmp_path, backend):
    from easydistill.operators.search_agent.tools import SearchToolset, SqliteCache

    db_path = str(tmp_path / "cache.db")
    toolset = SearchToolset(backend, {"tools": {"mode": "real", "cache_db_path": db_path}})
    # Pre-seed the cache so no network call is needed.
    SqliteCache(db_path).set("search::cached query::5", json.dumps(SEARCH_RESULTS))
    result = toolset.search("cached query")
    assert result["results"][0]["url"] == "https://example.com/wilder"


# ---------------------------------------------------------------------------
# Filter / SFT edge cases
# ---------------------------------------------------------------------------


def test_judge_filter_selection_strategies():
    row = _make_trajectory_row()
    row["trajectories"].append(
        {
            "solution_id": "task1_solution2",
            "trajectory": [{"role": "assistant", "content": "<answer>drama</answer>"}],
            "predicted_answer": "drama",
            "is_correct": True,
            "turns": 7,
        }
    )
    longest = run_judge_filter_stage([dict(row)], {"selection": "correct_longest"})
    assert longest[0]["selected_trajectories"][0]["turns"] == 7
    shortest = run_judge_filter_stage([dict(row)], {"selection": "correct_shortest"})
    assert shortest[0]["selected_trajectories"][0]["turns"] == 3
    all_correct = run_judge_filter_stage([dict(row)], {"selection": "all_correct"})
    assert len(all_correct[0]["selected_trajectories"]) == 2


def test_build_sft_length_bounds():
    row = _make_trajectory_row()
    row["selected_trajectories"] = row["trajectories"]
    assert run_build_search_sft_stage([row], {"min_length": 10_000}) == []
    assert run_build_search_sft_stage([row], {"max_length": 3}) == []
    assert len(run_build_search_sft_stage([row], {"min_length": 1})) == 1


# ---------------------------------------------------------------------------
# Original-feature parity: num_generations / resume / seed schema / defaults
# ---------------------------------------------------------------------------


def test_seed_normalization_original_schema():
    # Original SearchSynthAgent seed files use example_id/question_text/short_answer_text.
    rows = _normalize_seed_rows(
        [{"example_id": 42, "question_text": "Q?", "short_answer_text": "A"}]
    )
    assert rows == [{"id": "42", "question": "Q?", "answer": "A"}]


def test_evolver_num_generations_and_gen_suffix(backend):
    operator = SearchTaskEvolverOperator(
        backend=backend,
        config={
            "max_levels": 3,
            "max_iterations": 6,
            "final_eval_runs": 1,
            "max_workers": 1,
            "num_generations": 2,
        },
    )
    rows = operator.run([{"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER}])
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {"seed1-evolved#gen0", "seed1-evolved#gen1"}
    assert {r["gen_index"] for r in rows} == {0, 1}
    # Original v3_metadata equivalents live on each row.
    for row in rows:
        assert row["iteration_count"] >= 1
        assert row["final_level"] == 1
        assert row["layers"][0]["rewritten_question"] == SEED_QUESTION


def test_evolver_resume_from_skips_completed(tmp_path, backend):
    resume_file = tmp_path / "stage1.jsonl"
    resume_file.write_text(json.dumps({"id": "seed1-evolved#gen0", "seed_id": "seed1"}) + "\n")
    operator = SearchTaskEvolverOperator(
        backend=backend,
        config={
            "max_levels": 3,
            "max_iterations": 6,
            "final_eval_runs": 1,
            "max_workers": 1,
            "resume_from": str(resume_file),
        },
    )
    rows = operator.run(
        [
            {"id": "seed1", "question": SEED_QUESTION, "answer": SEED_ANSWER},
            {"id": "seed2", "question": SEED_QUESTION, "answer": SEED_ANSWER},
        ]
    )
    assert [r["seed_id"] for r in rows] == ["seed2"]


def test_toolset_browse_default_matches_original(backend):
    from easydistill.operators.search_agent.tools import SearchToolset

    # 500 chars is the original mock_fetch_page default.
    assert SearchToolset(backend, {}).browse_max_chars == 500
