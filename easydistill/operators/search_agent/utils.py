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

"""Shared helpers for search-agent distillation operators."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.operators.agent.utils import _extract_balanced_json_object

logger = logging.getLogger(__name__)

# Extra retry layer on top of the backend/SDK retries, to survive transient
# connection drops during long evolve loops.
_CALL_RETRIES = 3
_CALL_RETRY_DELAY = 5.0

# Role names used across the search-agent pipeline. Each role can be mapped to
# its own model/temperature/max_tokens in the ``roles`` section of the stage
# config, mirroring the original SearchSynthAgent ``step_models`` layout.
ROLE_STRATEGIST = "strategist"
ROLE_SYNTHESIS = "synthesis"
ROLE_SEARCH_SIM = "search_sim"
ROLE_JUDGE = "judge"
ROLE_SOLVER = "solver"
ROLE_FAST_VERIFY = "fast_verify"

_ROLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    ROLE_STRATEGIST: {"temperature": 0.0, "max_tokens": 4096},
    ROLE_SYNTHESIS: {"temperature": 0.7, "max_tokens": 4096},
    ROLE_SEARCH_SIM: {"temperature": 0.7, "max_tokens": 4096},
    ROLE_JUDGE: {"temperature": 0.0, "max_tokens": 4096},
    ROLE_SOLVER: {"temperature": 0.7, "max_tokens": 4096},
    ROLE_FAST_VERIFY: {"temperature": 0.0, "max_tokens": 2048},
}


def resolve_role_config(config: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Resolve generation kwargs for a role from the stage config.

    ``config["roles"][role]`` may define ``model_id``, ``temperature``,
    ``max_tokens`` and ``no_think``. Unset fields fall back to role defaults;
    ``fast_verify`` falls back to the ``solver`` role when absent, matching
    the original FastVerifyAgent -> SolveAgent fallback. ``no_think: true``
    appends the ``/no_think`` marker to user prompts, mirroring the original
    SearchSynthAgent behavior for Qwen3-style hybrid-thinking models.
    """
    roles = config.get("roles") or {}
    role_cfg = roles.get(role)
    if role_cfg is None and role == ROLE_FAST_VERIFY:
        role_cfg = roles.get(ROLE_SOLVER)
    role_cfg = role_cfg or {}
    defaults = _ROLE_DEFAULTS.get(role, {"temperature": 0.7, "max_tokens": 4096})
    return {
        "model_id": role_cfg.get("model_id"),
        "temperature": float(role_cfg.get("temperature", defaults["temperature"])),
        "max_tokens": int(role_cfg.get("max_tokens", defaults["max_tokens"])),
        "no_think": bool(role_cfg.get("no_think", False)),
    }


def _generate_with_retry(
    backend: ModelBackend,
    messages: List[Dict[str, Any]],
    model_id: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the backend with an outer retry layer for transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(_CALL_RETRIES):
        try:
            result = backend.generate(
                messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result.response or ""
        except Exception as exc:  # noqa: BLE001 - transient network errors
            last_exc = exc
            if attempt < _CALL_RETRIES - 1:
                delay = _CALL_RETRY_DELAY * (attempt + 1)
                logger.warning(
                    "Backend call failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt + 1,
                    _CALL_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def call_role(
    backend: ModelBackend,
    config: Dict[str, Any],
    role: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the backend with the model settings of a pipeline role."""
    role_cfg = resolve_role_config(config, role)
    if role_cfg["no_think"]:
        user_prompt = user_prompt + "/no_think"
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return _generate_with_retry(
        backend,
        messages,
        role_cfg["model_id"],
        role_cfg["temperature"] if temperature is None else temperature,
        role_cfg["max_tokens"] if max_tokens is None else max_tokens,
    )


def call_role_messages(
    backend: ModelBackend,
    config: Dict[str, Any],
    role: str,
    messages: List[Dict[str, Any]],
) -> str:
    """Call the backend with a full message history for a pipeline role."""
    role_cfg = resolve_role_config(config, role)
    return _generate_with_retry(
        backend,
        messages,
        role_cfg["model_id"],
        role_cfg["temperature"],
        role_cfg["max_tokens"],
    )


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object extraction from model output.

    Tries the whole string, then a fenced ```json block, then the first
    balanced ``{...}`` object.
    """
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return _extract_balanced_json_object(text)


def extract_final_answer(solve_history: List[Dict[str, Any]]) -> str:
    """Extract the last ``<answer>...</answer>`` from assistant messages."""
    for msg in reversed(solve_history or []):
        if msg.get("role") != "assistant":
            continue
        match = re.search(r"<answer>(.*?)</answer>", msg.get("content", ""), re.DOTALL)
        if match:
            return match.group(1).strip()
    for msg in reversed(solve_history or []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def count_assistant_turns(solve_history: List[Dict[str, Any]]) -> int:
    """Number of assistant messages, used as the turn count of a trajectory."""
    return sum(1 for m in solve_history or [] if m.get("role") == "assistant")


def count_tool_steps(solve_history: List[Dict[str, Any]]) -> int:
    """Number of web_search/web_browse calls in a trajectory."""
    count = 0
    for msg in solve_history or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "web_search" in content:
            count += 1
        if "web_browse" in content:
            count += 1
    return count


def format_trajectory_for_judge(
    trajectory: List[Dict[str, Any]],
    max_message_chars: int = 15000,
    max_total_chars: int = 8000,
) -> str:
    """Render a solver trajectory as compact text for the judge."""
    if not trajectory:
        return "No trajectory available"
    parts: List[str] = []
    for i, msg in enumerate(trajectory):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        if role == "system":
            continue
        if len(content) > max_message_chars:
            content = content[:max_message_chars] + "... [truncated]"
        if role == "assistant":
            parts.append(f"### Assistant (Turn {i}):\n{content}\n")
        elif role == "user":
            if "<tool_response>" in content:
                response_text = (
                    content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                )
                if len(response_text) > 500:
                    response_text = response_text[:500] + "... [truncated]"
                parts.append(f"### Tool Response:\n{response_text}\n")
            else:
                parts.append(f"### User (Turn {i}):\n{content}\n")
    full_text = "\n".join(parts)
    if len(full_text) > max_total_chars:
        full_text = full_text[:max_total_chars] + "\n\n... [trajectory truncated for length]"
    return full_text


def summarize_trajectory_for_strategist(trajectory: List[Dict[str, Any]]) -> str:
    """Short action-level summary of a solver trajectory for the strategist."""
    if not trajectory:
        return "No solver trajectory available (first iteration)"
    parts: List[str] = []
    search_count = 0
    browse_count = 0
    for msg in trajectory:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "assistant":
            if "web_search" in content:
                search_count += 1
                query_match = re.search(r'"query":\s*"([^"]+)"', content)
                if query_match:
                    parts.append(f"[Search {search_count}] Query: {query_match.group(1)}")
            if "web_browse" in content:
                browse_count += 1
                url_match = re.search(r'"url":\s*"([^"]+)"', content)
                if url_match:
                    parts.append(f"[Browse {browse_count}] URL: {url_match.group(1)[:50]}...")
            if "<answer>" in content:
                answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                if answer_match:
                    parts.append(f"[Final Answer] {answer_match.group(1).strip()[:100]}")
        elif role == "user" and "<tool_response>" in content and parts:
            parts.append("  \u2514\u2500 Got response...")
    if not parts:
        return "Solver trajectory is empty or unparseable"
    return "\n".join(parts[:15])
