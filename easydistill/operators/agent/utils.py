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

"""Utility helpers for agent distillation operators."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract the last occurrence of ``<tag>...</tag>`` from text."""
    matches = re.findall(rf"<{tag}>(.+?)</{tag}>", text, re.DOTALL)
    if matches:
        return str(matches[-1]).strip()
    return None


def extract_all_tags(text: str, tag: str) -> List[str]:
    """Extract all occurrences of ``<tag>...</tag>`` from text."""
    return [m.strip() for m in re.findall(rf"<{tag}>(.+?)</{tag}>", text, re.DOTALL)]


def extract_tool_call(text: str) -> Optional[str]:
    """Extract the last ``<tool_call>...</tool_call>`` content from text."""
    return extract_tag(text, "tool_call")


def _extract_balanced_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find the first balanced ``{...}`` in text that parses as a JSON dict."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if in_str:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


# Native special-token tool-call format emitted by some teacher models
# (e.g. kimi): <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:0
# <|tool_call_argument_begin|>{...}<|tool_call_end|><|tool_calls_section_end|>
# Hardened variants: optional ``<|tool_call_begin|>``, tool names with dots or
# dashes, optional ``:index`` suffix, and a missing ``<|tool_call_end|>``.
_NATIVE_TOOL_CALL_RE = re.compile(
    r"(?:<\|tool_call_begin\|>\s*)?functions\.([\w.\-]+?)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>(.*?)"
    r"(?:<\|tool_call_end\|>|<\|tool_calls_section_end\|>|<\|tool_call_begin\|>|$)",
    re.DOTALL,
)


def extract_tool_call_lenient(text: str) -> Optional[str]:
    """Hardened tool-call extraction for non-standard model output formats.

    Fallback used ONLY after the standard ``extract_tool_call`` (the
    ``<tool_call>...</tool_call>`` tag format) returns None, so the default
    parsing path is byte-for-byte unchanged. Shared by the mock/sandbox
    trajectory operator, the real_exec generator, and the pipeline boundary
    adapters — the kimi special-token parsing lives in this single place.

    Returns the same contract as ``extract_tool_call``: a JSON string
    ``{"name": ..., "arguments": {...}}`` consumable by
    ``parse_tool_call_json``, or None when no tool call is recognized.
    """
    if not text:
        return None

    # Variant A: mixed/unofficial <tool_call> closers, e.g.
    # <tool_call>{...}<|tool_call_end|> (the strict tag regex misses these).
    match = re.search(
        r"<tool_call>(.*?)(?:</tool_call>|<\|tool_call_end\|>)", text, re.DOTALL
    )
    if match:
        data = _extract_balanced_json_object(match.group(1))
        if data and data.get("name"):
            return json.dumps(
                {"name": data["name"], "arguments": data.get("arguments") or {}},
                ensure_ascii=False,
            )

    # Variant B: native special-token format (kimi) and tolerant variants.
    match = _NATIVE_TOOL_CALL_RE.search(text)
    if match:
        name = match.group(1)
        args_text = match.group(2).strip()
        arguments: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(args_text) if args_text else {}
            if isinstance(parsed, dict):
                arguments = parsed
        except json.JSONDecodeError:
            arguments = _extract_balanced_json_object(args_text)
        if arguments is not None:
            return json.dumps(
                {"name": name, "arguments": arguments}, ensure_ascii=False
            )

    # Variant C: markdown fenced block  ```tool_call\n{...}\n```
    match = re.search(r"```tool_call\s*\n(.*?)```", text, re.DOTALL)
    if match:
        data = _extract_balanced_json_object(match.group(1))
        if data and data.get("name"):
            return json.dumps(
                {"name": data["name"], "arguments": data.get("arguments") or {}},
                ensure_ascii=False,
            )

    # Variant D: bare functions.NAME({...}) call style.
    match = re.search(r"functions\.([\w.\-]+)\s*\(\s*(\{.*?\})\s*\)", text, re.DOTALL)
    if match:
        arguments = _extract_balanced_json_object(match.group(2))
        if arguments is not None:
            return json.dumps(
                {"name": match.group(1), "arguments": arguments}, ensure_ascii=False
            )

    return None


def format_tools_description(tools: List[Dict[str, Any]]) -> str:
    """Format a list of tool schemas as OpenAI function-calling strings."""
    lines = []
    for tool in tools:
        tool_copy = dict(tool)
        # Drop outputs section; the solver only needs parameters.
        tool_copy.pop("outputs", None)
        lines.append(json.dumps({"type": "function", "function": tool_copy}))
    return "\n".join(lines)


def parse_tool_call_json(tool_call_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse a tool-call JSON string into (name, arguments).

    Returns ``(None, None)`` if parsing fails.
    """
    try:
        data = json.loads(tool_call_text)
        if isinstance(data, dict):
            return data.get("name"), data.get("arguments")
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call JSON: %s", tool_call_text[:200])
    return None, None


def extract_tool_response(text: str) -> Tuple[Optional[str], bool]:
    """Extract the simulated tool response and new-background flag."""
    tool_response = extract_tag(text, "tool_response_start")
    new_bg_text = extract_tag(text, "new_bg_introduced")
    new_bg_introduced = False
    if new_bg_text:
        new_bg_introduced = "YES" in new_bg_text.upper()
    return tool_response, new_bg_introduced


def extract_answer(text: str) -> Optional[str]:
    """Extract content wrapped in ``<answer>...</answer>``."""
    return extract_tag(text, "answer")


def extract_best_solution_filename(text: str) -> Optional[str]:
    """Extract the best solution identifier from rubric output.

    The judge writes the ``solution_id`` (often with a ``.json`` suffix)
    inside the ``<best_solution>`` tag.  Return the stripped text when it
    looks like a solution reference; otherwise fall back to a regex search.
    """
    if not text:
        return None
    cleaned = text.strip().replace("\n", " ").replace("\r", " ")
    if cleaned and (".json" in cleaned.lower() or "solution" in cleaned.lower()):
        return cleaned
    match = re.search(r"[\w\-_]+solution[\w\-_]*\.json", text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def format_trajectory_for_comparison(solution_id: str, trajectory: List[Dict[str, Any]]) -> str:
    """Summarize a single trajectory for the rubric judge."""
    tool_count = 0
    tools_used: List[str] = []
    for message in trajectory:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            tool_count += content.count("<tool_call>")
            for tc in extract_all_tags(content, "tool_call"):
                name, _ = parse_tool_call_json(tc)
                if name:
                    tools_used.append(name)

    summary = f"### Solution: {solution_id}\n\n"
    summary += f"**Tool Call Count**: {tool_count}\n"
    summary += f"**Tools Used**: {', '.join(tools_used)}\n\n"
    summary += "**Trajectory Summary**:\n"

    for i, message in enumerate(trajectory):
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            continue
        if role == "user":
            if i == 1:
                summary += f"\n[User Request]\n{content[:800]}...\n"
            elif "tool_response" in content or role == "tool":
                summary += f"\n[Tool Response]\n{content[:500]}...\n"
        elif role == "assistant":
            if "<tool_call>" in content:
                reasoning = content.split("<tool_call>")[0].strip()
                start = content.find("<tool_call>")
                end = content.find("</tool_call>") + len("</tool_call>")
                tool_part = content[start:end]
                summary += (
                    f"\n[Assistant-Step{i // 2}]\n"
                    f"Reasoning: {reasoning[:800]}...\n"
                    f"Tool Call: {tool_part}\n"
                )
            else:
                summary += f"\n[Assistant-Final Answer]\n{content[:1000]}...\n"
    return summary
