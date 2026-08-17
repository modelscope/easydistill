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

"""Agent trajectory rollout operator using a langgraph ReAct loop."""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, TypedDict

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationResult
from easydistill.operators.base import Operator
from easydistill.prompts import resolve_prompt
from easydistill.utils import DEFAULT_MAX_TOKENS, DEFAULT_RETRY_ATTEMPTS, format_prompt_safely

from .utils import (
    extract_tag,
    extract_tool_call,
    extract_tool_response,
    format_tools_description,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    """Return True if an exception looks transient."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    exc_module = type(exc).__module__
    exc_name = type(exc).__name__
    if exc_module == "httpx" and exc_name in (
        "ConnectError",
        "ReadError",
        "WriteError",
        "TimeoutException",
        "NetworkError",
    ):
        return True
    return exc_module == "openai" and exc_name in (
        "RateLimitError",
        "InternalServerError",
        "APITimeoutError",
        "APIConnectionError",
    )


class TrajectoryState(TypedDict, total=False):
    """Langgraph state for a single trajectory rollout."""

    fuzzy_task: str
    task_background: str
    checked_tools: List[Dict[str, Any]]
    restrict: str
    solve_history: List[Dict[str, Any]]
    tool_call_history: List[str]
    tool_call_count: int
    current_tool_call: Optional[str]
    task_finished: str
    step_count: int


class AgentTrajectoryOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Generate multi-turn agent trajectories for virtual tool-use tasks.

    Configurable fields:
      - max_steps: max ReAct steps per rollout (default 10).
      - repeat_times: number of trajectories to generate per task (default 2).
      - solve: dict passed to the solve generator (model_id, temperature, ...).
      - mock_tool: dict passed to the mock-tool generator.
      - mock_user: dict passed to the mock-user generator.
      - solve_system_prompt_template_file: optional custom solve system prompt.
      - mock_tool_prompt_template_file: optional custom mock-tool prompt.
      - mock_user_prompt_template_file: optional custom mock-user prompt.

    Input rows must contain ``fuzzy_task``, ``task_background``,
    ``checked_tools``, ``restriction``, and optionally ``workflow``.

    Output is one row per trajectory with:
      - id, solution_id, fuzzy_task, task_background, restriction, checked_tools
      - trajectory: list of OpenAI-style messages.
      - tool_call_history: list of "Query:\n...\nResponse:\n..." strings.
      - task_finished: termination reason.
    """

    name = "agent_trajectory"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend = backend
        self.max_steps = int(self.config.get("max_steps") or 10)
        self.repeat_times = int(self.config.get("repeat_times") or 2)
        self.max_tool_calls = int(self.config.get("max_tool_calls") or 20)
        self.solve_cfg = self.config.get("solve", {})
        self.mock_tool_cfg = self.config.get("mock_tool", {})
        self.mock_user_cfg = self.config.get("mock_user", {})

        self.solve_system_prompt_template = resolve_prompt(
            self.config,
            file_key="solve_system_prompt_template_file",
            template_key="solve_system_prompt_template",
            default_file="configs/prompts/agent_solve_system_prompt.txt",
        )
        self.mock_tool_prompt_template = resolve_prompt(
            self.config,
            file_key="mock_tool_prompt_template_file",
            template_key="mock_tool_prompt_template",
            default_file="configs/prompts/agent_mock_tool_prompt.txt",
        )
        self.mock_user_prompt_template = resolve_prompt(
            self.config,
            file_key="mock_user_prompt_template_file",
            template_key="mock_user_prompt_template",
            default_file="configs/prompts/agent_mock_user_prompt.txt",
        )

        from langgraph.graph import END, StateGraph

        builder = StateGraph(TrajectoryState)
        builder.add_node("reason_and_act", self._reason_and_act_node)
        builder.add_node("mock_tools", self._mock_tools_node)
        builder.add_node("mock_user", self._mock_user_node)
        builder.set_entry_point("reason_and_act")
        builder.add_conditional_edges(
            "reason_and_act",
            self._should_call_tool,
            {"tool_call": "mock_tools", "user": "mock_user", "end": END},
        )
        builder.add_edge("mock_tools", "reason_and_act")
        builder.add_edge("mock_user", "reason_and_act")
        self.graph = builder.compile()

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        gen_cfg: Dict[str, Any],
    ) -> GenerationResult:
        """Call the backend with retry/backoff."""
        model_id = gen_cfg.get("model_id")
        temperature = gen_cfg.get("temperature")
        if temperature is None:
            temperature = 0.7
        max_tokens = int(gen_cfg.get("max_tokens") or DEFAULT_MAX_TOKENS)
        retry_attempts = int(gen_cfg.get("retry_attempts") or DEFAULT_RETRY_ATTEMPTS)
        retry_backoff_base = float(gen_cfg.get("retry_backoff_base") or 1.0)
        retry_max_wait = float(gen_cfg.get("retry_max_wait") or 30.0)
        raise_on_error = bool(gen_cfg.get("raise_on_error") or False)

        last_exc: Optional[Exception] = None
        total_attempts = retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            try:
                return self.backend.generate(
                    messages=messages,
                    model_id=model_id,
                    temperature=float(temperature),
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == total_attempts or not _is_retryable(exc):
                    break
                wait = min(
                    retry_backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random()),
                    retry_max_wait,
                )
                logger.warning(
                    "Agent generation failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("Agent generation failed after %d attempts: %s", total_attempts, last_exc)
        if raise_on_error and last_exc is not None:
            raise last_exc
        # Return an empty result so the graph can continue gracefully.
        return GenerationResult(
            request=None,  # type: ignore[arg-type]
            response="",
            model=model_id or "agent",
        )

    def _build_solve_messages(
        self,
        fuzzy_task: str,
        checked_tools: List[Dict[str, Any]],
        restrict: str,
    ) -> List[Dict[str, Any]]:
        tools_description = format_tools_description(checked_tools)
        system_prompt = format_prompt_safely(
            self.solve_system_prompt_template or "",
            available_tools=tools_description,
            restrict=restrict,
        )
        user_prompt = f"""Task Description: {fuzzy_task}.

### Requirements:
1. Please call only one tool at a time, and you must provide your brief
   reasoning process before using any tool. You cannot give a tool call
   without providing your reasoning process.

2. Once the task is complete, output the final answer, wrapping the answer
   in `<answer></answer>` as a termination signal.

3. IMPORTANT: The user most likely provided insufficient information; you
   are encouraged to interact with the user to gather more information if
   needed. Before calling any tool, if **any required parameter is uncertain,
   missing, ambiguous, or not explicitly provided by the user**, you **MUST
   ask the user for clarification first**. Do NOT guess or fabricate
   parameters!!!
"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _reason_and_act_node(self, state: TrajectoryState) -> Dict[str, Any]:
        solve_history = state.get("solve_history", [])
        step_count = state.get("step_count", 0)

        if not solve_history:
            solve_history = self._build_solve_messages(
                state["fuzzy_task"],
                state["checked_tools"],
                state["restrict"],
            )

        if step_count >= self.max_steps:
            solve_history.append(
                {
                    "role": "assistant",
                    "content": "<answer>Reached maximum step limit.</answer>",
                }
            )
            return {
                "solve_history": solve_history,
                "task_finished": "Terminated",
                "step_count": step_count,
            }

        if state.get("tool_call_count", 0) >= self.max_tool_calls:
            solve_history.append(
                {
                    "role": "assistant",
                    "content": "<answer>Reached maximum tool call limit.</answer>",
                }
            )
            return {
                "solve_history": solve_history,
                "task_finished": "Terminated",
                "step_count": step_count,
            }

        result = self._generate(solve_history, self.solve_cfg)
        content = result.response or ""
        solve_history.append({"role": "assistant", "content": content})

        if "<answer>" in content:
            task_finished = "Terminated"
            current_tool_call = None
        else:
            tool_call = extract_tool_call(content)
            if tool_call:
                task_finished = "Tool call"
                current_tool_call = tool_call
            else:
                task_finished = "Transfer to user"
                current_tool_call = None

        return {
            "solve_history": solve_history,
            "current_tool_call": current_tool_call,
            "task_finished": task_finished,
            "step_count": step_count + 1,
        }

    def _mock_tools_node(self, state: TrajectoryState) -> Dict[str, Any]:
        tool_call = state.get("current_tool_call")
        if not tool_call:
            return {}

        checked_tools = state.get("checked_tools", [])
        tool_call_history = state.get("tool_call_history", [])
        solve_history = state.get("solve_history", [])

        prompt = format_prompt_safely(
            self.mock_tool_prompt_template or "",
            tools=json.dumps(checked_tools, ensure_ascii=False, indent=2),
            world_state=json.dumps(tool_call_history, ensure_ascii=False, indent=2),
            query=tool_call,
        )
        messages = [{"role": "user", "content": prompt}]
        result = self._generate(messages, self.mock_tool_cfg)
        tool_response, new_bg_introduced = extract_tool_response(result.response or "")
        if tool_response is None:
            tool_response = result.response or ""

        solve_history.append(
            {"role": "user", "content": f"<tool_response>{tool_response}</tool_response>"}
        )
        if new_bg_introduced:
            tool_call_history.append(f"Query:\n{tool_call}\nResponse:\n{tool_response}")

        return {
            "solve_history": solve_history,
            "tool_call_history": tool_call_history,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        }

    def _mock_user_node(self, state: TrajectoryState) -> Dict[str, Any]:
        solve_history = state.get("solve_history", [])
        prompt = format_prompt_safely(
            self.mock_user_prompt_template or "",
            task=state.get("fuzzy_task", ""),
            background=state.get("task_background", ""),
            restrict=state.get("restrict", ""),
            interaction=json.dumps(solve_history, ensure_ascii=False, indent=2),
        )
        messages = [{"role": "user", "content": prompt}]
        result = self._generate(messages, self.mock_user_cfg)
        reply = extract_tag(result.response or "", "reply")
        if reply is None:
            reply = result.response or ""

        solve_history.append({"role": "user", "content": reply})
        return {"solve_history": solve_history}

    @staticmethod
    def _should_call_tool(state: TrajectoryState) -> str:
        finished = state.get("task_finished", "")
        if finished == "Terminated":
            return "end"
        if finished == "Tool call":
            return "tool_call"
        return "user"

    def _run_single_trajectory(
        self,
        base_row: Dict[str, Any],
        solution_idx: int,
    ) -> Optional[Dict[str, Any]]:
        task_id = str(base_row.get("id", "unknown"))
        solution_id = f"{task_id}_solution_{solution_idx + 1}.json"
        initial_state: TrajectoryState = {
            "fuzzy_task": base_row.get("fuzzy_task", ""),
            "task_background": base_row.get("task_background", ""),
            "checked_tools": base_row.get("checked_tools", []),
            "restrict": base_row.get("restriction", ""),
            "solve_history": [],
            "tool_call_history": [],
            "tool_call_count": 0,
            "current_tool_call": None,
            "task_finished": "",
            "step_count": 0,
        }
        try:
            final_state = self.graph.invoke(
                initial_state,
                {"recursion_limit": max(self.max_steps * 3 + 10, 100)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Trajectory rollout failed for %s: %s", solution_id, exc)
            return None

        trajectory = final_state.get("solve_history", [])
        if not trajectory:
            return None

        row = dict(base_row)
        row.update(
            {
                "solution_id": solution_id,
                "trajectory": trajectory,
                "tool_call_history": final_state.get("tool_call_history", []),
                "task_finished": final_state.get("task_finished", "Unknown"),
            }
        )
        return row

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for row in data:
            for repeat_idx in range(self.repeat_times):
                trajectory_row = self._run_single_trajectory(row, repeat_idx)
                if trajectory_row is not None:
                    outputs.append(trajectory_row)
        logger.info(
            "AgentTrajectoryOperator produced %d trajectories from %d tasks.",
            len(outputs),
            len(data),
        )
        return outputs
