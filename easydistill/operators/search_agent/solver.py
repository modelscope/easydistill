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

"""Search solver: native ReAct rollout over web_search/web_browse tools."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.operators.agent.utils import (
    extract_tool_call,
    extract_tool_call_lenient,
    format_tools_description,
    parse_tool_call_json,
)
from easydistill.operators.base import Operator

from .judge import answer_equivalent
from .tools import SEARCH_TOOLS, SearchToolset
from .utils import (
    ROLE_SOLVER,
    call_role_messages,
    count_assistant_turns,
    extract_final_answer,
    resolve_role_config,
)

logger = logging.getLogger(__name__)

SOLVER_SYSTEM_TEMPLATE = "\n".join(
    [
        "You are a helpful assistant designed to solve tasks using the provided tools. "
        "Your task is to answer the user's question by calling the appropriate tools.",
        "1. You MUST provide your brief reasoning process before using any tool.",
        "2. You MUST call at least one tool before giving a final answer, "
        "even if you think you know the answer.",
        "3. Tool calls must be wrapped in <tool_call>...</tool_call> and contain only JSON.",
        "4. Final answers must be wrapped in <answer></answer>.",
        "",
        "# Tools",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        "{tools_description}",
        "</tools>",
        "",
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}",
        "</tool_call>",
    ]
)

SOLVER_USER_TEMPLATE = "\n".join(
    [
        "Task: Please answer the following question. You must use the provided tools "
        "to search or browse for information before answering.",
        "",
        "Question: {question}",
        "",
        "### Requirements:",
        "1. Call at least one tool before answering, even if you think you know the answer.",
        "2. Do NOT answer from internal knowledge. You must base your answer only on "
        "information returned by the tools.",
        "3. The final answer must cite evidence from the tool results (e.g., include a "
        "short quote or a snippet identifier from the tool output).",
        "4. Please call only one tool at a time, and you must provide your brief "
        "reasoning process before using any tool. You can not just give a tool call "
        "without providing your reasoning process.",
        "5. Once the task is complete, output the final answer, wrapping the answer in "
        "`<answer></answer>` as a termination signal.",
    ]
)


def build_solver_messages(question: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the initial system/user messages for the solver rollout.

    When the solver role sets ``no_think: true``, the ``/no_think`` marker is
    appended to the first user prompt only, matching the original
    SearchSynthAgent behavior for Qwen3-style hybrid-thinking models.
    """
    tools_description = format_tools_description(SEARCH_TOOLS)
    system_prompt = SOLVER_SYSTEM_TEMPLATE.format(tools_description=tools_description)
    user_prompt = SOLVER_USER_TEMPLATE.format(question=question)
    suffix = str(config.get("user_prompt_suffix", ""))
    if suffix:
        user_prompt += suffix
    if resolve_role_config(config, ROLE_SOLVER)["no_think"]:
        user_prompt += "/no_think"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def solve_search_task(
    backend: ModelBackend,
    config: Dict[str, Any],
    question: str,
    toolset: SearchToolset,
    max_steps: int = 12,
) -> List[Dict[str, Any]]:
    """Run one solver rollout and return the full ``solve_history``.

    Message convention matches the original SearchSynthAgent trajectories:
    assistant messages carry reasoning + ``<tool_call>`` JSON, tool outputs
    come back as user messages wrapped in ``<tool_response>``, and the final
    assistant message contains ``<answer>...</answer>``.
    """
    solve_history = build_solver_messages(question, config)
    for _ in range(max_steps):
        response = call_role_messages(backend, config, ROLE_SOLVER, solve_history)
        solve_history.append({"role": "assistant", "content": response})

        if "<answer>" in response:
            break
        tool_call_text = extract_tool_call(response) or extract_tool_call_lenient(response)
        if not tool_call_text:
            solve_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "I couldn't find a way to proceed. "
                        "<answer>Information not found.</answer>"
                    ),
                }
            )
            break

        tool_name, arguments = parse_tool_call_json(tool_call_text)
        try:
            response_text = toolset.execute(tool_name, arguments or {}, solve_history)
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the solver
            logger.warning("Tool execution failed (%s): %s", tool_name, exc)
            response_text = f"Error executing tool '{tool_name}': {exc}"
        solve_history.append(
            {"role": "user", "content": f"<tool_response>{response_text}</tool_response>"}
        )
    else:
        solve_history.append(
            {
                "role": "assistant",
                "content": "Maximum steps reached. <answer>Information not found.</answer>",
            }
        )
    return solve_history


# Upper bound for in-task rollout parallelism (evaluate_task final eval and
# SearchTrajectoryOperator repeat rollouts). Kept small: the outer seed/task
# level pools already provide the bulk of the concurrency, and stacking large
# inner pools on top risks hitting backend rate limits.
_INNER_ROLLOUT_WORKERS = 4


def evaluate_task(
    backend: ModelBackend,
    config: Dict[str, Any],
    question: str,
    gold_answer: str,
    toolset: SearchToolset,
    num_runs: int = 4,
    max_steps: int = 12,
) -> Dict[str, Any]:
    """Run the solver ``num_runs`` times and report accuracy / average turns.

    The runs are independent, so they are rolled out concurrently in a small
    pool; a broken run is skipped instead of aborting the whole evaluation.
    """

    def _single_run(run_idx: int) -> Optional[Dict[str, Any]]:
        try:
            solve_history = solve_search_task(backend, config, question, toolset, max_steps)
        except Exception as exc:  # noqa: BLE001 - one broken run must not kill the eval
            logger.error("Final-eval run %d failed: %s", run_idx, exc)
            return None
        predicted = extract_final_answer(solve_history)
        turns = count_assistant_turns(solve_history)
        is_correct = answer_equivalent(backend, config, question, predicted, gold_answer)
        return {
            "run_idx": run_idx,
            "turns": turns,
            "predicted": predicted,
            "is_correct": is_correct,
            "_trajectory": solve_history,
        }

    workers = min(num_runs, _INNER_ROLLOUT_WORKERS)
    run_results: List[Optional[Dict[str, Any]]] = [None] * num_runs
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_single_run, idx): idx for idx in range(num_runs)}
        for future in as_completed(futures):
            run_results[futures[future]] = future.result()

    runs: List[Dict[str, Any]] = []
    trajectories: List[List[Dict[str, Any]]] = []
    for result in run_results:
        if result is None:
            continue
        trajectories.append(result.pop("_trajectory"))
        runs.append(result)
    turns_list = [r["turns"] for r in runs]
    correct_list = [r["is_correct"] for r in runs]
    return {
        "accuracy": (sum(correct_list) / len(correct_list)) if correct_list else 0.0,
        "avg_turns": (sum(turns_list) / len(turns_list)) if turns_list else 0.0,
        "runs": runs,
        "trajectories": trajectories,
    }


class SearchTrajectoryOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Roll out solver trajectories for synthesized search tasks.

    Input rows must carry ``question`` and ``answer``. Each row gains a
    ``trajectories`` list where every entry records the solution id, the full
    message history, the predicted answer, correctness and turn count.
    """

    name = "search_trajectory"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend = backend
        self.toolset = SearchToolset(backend, self.config)

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        repeat_times = int(self.config.get("repeat_times", 2))
        max_steps = int(self.config.get("max_steps", 12))
        max_workers = int(self.config.get("max_workers", 4))

        def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            question = row.get("question", "")
            gold_answer = row.get("answer", "")
            if not question:
                return None

            def _rollout(run_idx: int) -> Optional[Dict[str, Any]]:
                try:
                    solve_history = solve_search_task(
                        self.backend, self.config, question, self.toolset, max_steps
                    )
                except Exception as exc:  # noqa: BLE001 - skip broken rollouts
                    logger.error(
                        "Trajectory rollout failed for %s run %d: %s",
                        row.get("id"),
                        run_idx,
                        exc,
                    )
                    return None
                predicted = extract_final_answer(solve_history)
                is_correct = answer_equivalent(
                    self.backend, self.config, question, predicted, gold_answer
                )
                return {
                    "solution_id": f"{row.get('id', 'task')}_solution{run_idx + 1}",
                    "trajectory": solve_history,
                    "predicted_answer": predicted,
                    "is_correct": is_correct,
                    "turns": count_assistant_turns(solve_history),
                }

            # repeat rollouts for one task are independent; run them in a small
            # inner pool (outer pool already parallelizes across tasks).
            workers = min(repeat_times, _INNER_ROLLOUT_WORKERS)
            rollout_results: List[Optional[Dict[str, Any]]] = [None] * repeat_times
            with ThreadPoolExecutor(max_workers=max(1, workers)) as inner:
                futures = {inner.submit(_rollout, idx): idx for idx in range(repeat_times)}
                for future in as_completed(futures):
                    rollout_results[futures[future]] = future.result()
            trajectories = [t for t in rollout_results if t is not None]
            if not trajectories:
                return None
            new_row = dict(row)
            new_row["trajectories"] = trajectories
            return new_row

        results: List[Optional[Dict[str, Any]]] = [None] * len(input_data)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, row): idx for idx, row in enumerate(input_data)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        output = [row for row in results if row is not None]
        logger.info(
            "Generated trajectories for %d/%d tasks (%d rollouts each).",
            len(output),
            len(input_data),
            repeat_times,
        )
        return output
