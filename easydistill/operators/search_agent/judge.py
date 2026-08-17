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

"""Judge helpers: trajectory difficulty report, answer check and quality gate."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend

from .utils import (
    ROLE_JUDGE,
    call_role,
    count_tool_steps,
    extract_final_answer,
    format_trajectory_for_judge,
    parse_json_safely,
)

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ("too_easy", "good", "too_hard", "broken")
ACTIONS = ("EXPAND", "REFINE", "ROLLBACK", "FINALIZE")

JUDGE_SYSTEM_PROMPT = "\n".join(
    [
        "You are an expert evaluator for multi-hop question generation quality.",
        "Your task is to analyze a solver agent's attempt and provide a concise assessment.",
        "",
        "## Action Definitions (IMPORTANT)",
        "- EXPAND: The question is correct but too easy (few steps). Add another hop.",
        "- REFINE: The question has shortcuts/leakage/overly direct phrasing. "
        "Keep hops, improve wording.",
        "- ROLLBACK: The question is broken/unsolvable or points to wrong entities. "
        "Revert and try a different path.",
        "- FINALIZE: The question is correct and has good difficulty.",
        "",
        "## Difficulty Guidance",
        "- too_easy: 0-5 search steps or obvious shortcut",
        "- good: 6+ search steps with a coherent path",
        "- too_hard: many steps but still incorrect",
        "- broken: logical/ factual issues make it unsolvable",
        "",
        "Return ONLY a compact JSON object, no extra text.",
    ]
)

JUDGE_USER_TEMPLATE = """## Question Information
Original Question: {original_question}
Current Question: {current_question}
Ground Truth Answer: {ground_truth}

## Solver Trajectory
{solver_trajectory}

## Solver Final Answer
{final_answer}

Please output a compact JSON with this schema:
{{
  "is_correct": true/false,
  "total_steps": <int>,
  "difficulty_level": "too_easy" | "good" | "too_hard" | "broken",
  "has_shortcut": true/false,
  "recommended_action": "EXPAND" | "REFINE" | "ROLLBACK" | "FINALIZE",
  "reason": "",
  "suggestions": ["short suggestion 1", "short suggestion 2"]
}}

Return ONLY JSON."""

ANSWER_CHECK_SYSTEM_PROMPT = (
    """You are an answer evaluator. Determine if the predicted answer matches """
    """the ground truth. Consider semantic equivalence, not just exact """
    """string match. Respond with ONLY a JSON object: {"equivalent": true} """
    """or {"equivalent": false}"""
)

QUALITY_GATE_SYSTEM_PROMPT = "\n".join(
    [
        "You are a question quality gate for multi-hop question generation.",
        "Your task is to check whether the question should proceed to solver verification.",
        "",
        "Checks:",
        "1) Uniqueness: the question must have a unique, unambiguous answer.",
        "2) Pseudo multi-hop: detect structural shortcuts that allow answering without "
        "resolving the bridge/indirection.",
        "   - IMPORTANT: Ignore whether a clue is \"well-known\" or guessable. "
        "Do NOT reject just because it is famous.",
        "   - You MUST NOT use your own world knowledge to resolve clues. Treat any "
        "unresolved description as requiring",
        "     an external lookup. If identifying the target would require knowledge "
        "beyond the text, it is NOT a shortcut.",
        "   - Only mark a shortcut if the question text itself explicitly contains the "
        "target/answer or directly equates",
        "     them (e.g., apposition or explicit definition: \"the play Our Town, "
        "set in Grover's Corners\").",
        "   - Example (NOT a shortcut): \"the world's second-largest island country\" "
        "without naming it.",
        "   - Example (shortcut): \"the world's second-largest island country, "
        "Papua New Guinea\".",
        "3) Leakage: the question must not directly reveal the answer or "
        "obvious near-synonyms.",
        "",
        "Output JSON only.",
    ]
)

QUALITY_GATE_USER_TEMPLATE = """## Original Question
{original_question}

## Current Question
{current_question}

## Gold Answer
{answer}

## Bridge Context (if any)
{bridge_info}

Return JSON:
{{
  "pass": true/false,
  "failed_checks": ["uniqueness" | "pseudo_multihop" | "leakage"],
  "uniqueness": {{
    "is_unique": true/false,
    "alternative_answers": [],
    "reasoning": ""
  }},
  "pseudo_multihop": {{
    "has_shortcut": true/false,
    "reasoning": ""
  }},
  "leakage": {{
    "leaks_answer": true/false,
    "leak_spans": [],
    "reasoning": ""
  }},
  "rollback_reason": "short reason if pass=false, else empty"
}}"""


def judge_trajectory(
    backend: ModelBackend,
    config: Dict[str, Any],
    original_question: str,
    current_question: str,
    ground_truth: str,
    solver_trajectory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze a solver trajectory and return a compact difficulty report."""
    trajectory_text = format_trajectory_for_judge(solver_trajectory)
    final_answer = extract_final_answer(solver_trajectory)
    user_prompt = JUDGE_USER_TEMPLATE.format(
        original_question=original_question,
        current_question=current_question,
        ground_truth=ground_truth,
        solver_trajectory=trajectory_text,
        final_answer=final_answer or "No answer provided",
    )
    try:
        content = call_role(
            backend, config, ROLE_JUDGE, user_prompt, JUDGE_SYSTEM_PROMPT, temperature=0.0
        )
        result = parse_json_safely(content)
        if not isinstance(result, dict):
            raise ValueError("Judge output is not a valid JSON object")
    except Exception as exc:  # noqa: BLE001 - degrade to rule-based judgment
        logger.error("Judge LLM call failed, using fallback: %s", exc)
        return _fallback_judge(solver_trajectory, ground_truth)

    total_steps = result.get("total_steps")
    if not isinstance(total_steps, int):
        total_steps = count_tool_steps(solver_trajectory)
    difficulty_level = result.get("difficulty_level", "broken")
    if difficulty_level not in DIFFICULTY_LEVELS:
        difficulty_level = "broken"
    recommended_action = result.get("recommended_action", "FINALIZE")
    if recommended_action not in ACTIONS:
        recommended_action = "FINALIZE"
    suggestions = result.get("suggestions", [])
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if not isinstance(suggestions, list):
        suggestions = []

    return {
        "is_correct": bool(result.get("is_correct", False)),
        "total_steps": int(total_steps),
        "difficulty_level": difficulty_level,
        "has_shortcut": bool(result.get("has_shortcut", False)),
        "recommended_action": recommended_action,
        "reason": result.get("reason", "") or "",
        "suggestions": suggestions,
    }


def _fallback_judge(solver_trajectory: List[Dict[str, Any]], ground_truth: str) -> Dict[str, Any]:
    """Rule-based judgment used when the judge LLM output is unusable."""
    total_steps = count_tool_steps(solver_trajectory)
    final_answer = extract_final_answer(solver_trajectory)
    is_correct = False
    if final_answer and ground_truth:
        is_correct = (
            ground_truth.lower() in final_answer.lower()
            or final_answer.lower() in ground_truth.lower()
        )
    if is_correct and total_steps >= 4:
        difficulty_level = "good"
    elif is_correct:
        difficulty_level = "too_easy"
    elif total_steps >= 4:
        difficulty_level = "too_hard"
    else:
        difficulty_level = "broken"
    recommended_action = (
        "FINALIZE"
        if difficulty_level == "good"
        else ("EXPAND" if difficulty_level == "too_easy" else "ROLLBACK")
    )
    return {
        "is_correct": is_correct,
        "total_steps": total_steps,
        "difficulty_level": difficulty_level,
        "has_shortcut": False,
        "recommended_action": recommended_action,
        "reason": "Fallback rule-based judgment",
        "suggestions": [],
    }


def answer_equivalent(
    backend: ModelBackend,
    config: Dict[str, Any],
    question: str,
    predicted_answer: str,
    ground_truth: str,
) -> bool:
    """LLM semantic-equivalence check with string-match fallback."""
    if not predicted_answer or not ground_truth:
        return False
    user_prompt = (
        f"Question: {question}\n"
        f"Predicted Answer: {predicted_answer}\n"
        f"Ground Truth: {ground_truth}\n\n"
        "Are they equivalent?"
    )
    try:
        content = call_role(
            backend, config, ROLE_JUDGE, user_prompt, ANSWER_CHECK_SYSTEM_PROMPT, temperature=0.0
        )
        result = parse_json_safely(content)
        if isinstance(result, dict):
            return bool(result.get("equivalent", False))
    except Exception as exc:  # noqa: BLE001 - degrade to string matching
        logger.warning("LLM answer check failed: %s", exc)
    return (
        ground_truth.lower().strip() in predicted_answer.lower().strip()
        or predicted_answer.lower().strip() in ground_truth.lower().strip()
    )


def run_quality_gate(
    backend: ModelBackend,
    config: Dict[str, Any],
    original_question: str,
    current_question: str,
    answer: str,
    bridge_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check uniqueness / pseudo multi-hop / leakage before verification.

    Parse failures pass the gate, matching the original fail-open behavior.
    """
    user_prompt = QUALITY_GATE_USER_TEMPLATE.format(
        original_question=original_question,
        current_question=current_question,
        answer=answer,
        bridge_info=bridge_info or {},
    )
    try:
        content = call_role(
            backend, config, ROLE_JUDGE, user_prompt, QUALITY_GATE_SYSTEM_PROMPT, temperature=0.0
        )
        result = parse_json_safely(content)
    except Exception as exc:  # noqa: BLE001 - fail open like the original gate
        logger.warning("Quality gate failed to run: %s", exc)
        return {"pass": True}
    if not isinstance(result, dict):
        return {"pass": True}
    result["pass"] = bool(result.get("pass", True))
    return result
