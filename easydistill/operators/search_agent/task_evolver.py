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

"""Search task evolver: native closed-loop multi-hop question generation.

Ports the SearchSynthAgent v3 builder (Strategist -> Expand/Refine/Rollback ->
QualityGate -> Verify -> Judge loop -> Finalize) onto a plain Python loop with
all model calls going through the easydistill backend abstraction.
"""

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.operators.base import Operator

from .judge import answer_equivalent, judge_trajectory, run_quality_gate
from .solver import evaluate_task, solve_search_task
from .tools import SearchToolset
from .utils import (
    ROLE_FAST_VERIFY,
    ROLE_STRATEGIST,
    ROLE_SYNTHESIS,
    call_role,
    parse_json_safely,
    summarize_trajectory_for_strategist,
)

logger = logging.getLogger(__name__)

ACTIONS = ("EXPAND", "REFINE", "ROLLBACK", "FINALIZE")

STRATEGIST_SYSTEM_PROMPT = "\n".join(
    [
        "You are a strategic decision-making agent for multi-hop question generation.",
        "Your task is to analyze the current state, diagnose problems, and decide the next action.",
        "",
        "## Your Goal",
        "Generate challenging multi-hop questions that require multiple search steps to solve.",
        "The question should be:",
        "1. Difficult enough that a search agent needs multiple search steps to find "
        "the answer (6+ is a guideline, not a hard rule)",
        "2. Logically coherent - the answer path should be traceable",
        "3. Not too convoluted that it becomes unsolvable",
        "",
        "## Available Actions",
        "",
        "1. **EXPAND**: Add another hop to the question by searching a new entity and "
        "replacing it with a description.",
        "   - Use when: The question is still too easy",
        "   - **Guidance for EXPAND**: You MUST specify the exact target entity "
        "(a phrase that appears verbatim in the Current Question) to expand.",
        "     The target MUST be a named entity / proper noun (person, place, event, "
        "organization, work title, award, year, etc.).",
        "",
        "2. **REFINE**: Make the current question harder without adding new hops.",
        "   - Use when: Multiple EXPANDs haven't increased difficulty",
        "   - Use when: The question needs more constraints or ambiguity",
        "   - **Guidance for REFINE**: Suggest specific refinement technique "
        "(temporal, exclusionary, vague, contextual)",
        "",
        "3. **ROLLBACK**: Undo the last change and try a different path.",
        "   - Use when: The solver couldn't find the answer (broken logic)",
        "   - Use when: The rewritten question became unsolvable or illogical",
        "   - Use when: The entity description leaked the answer",
        "",
        "4. **FINALIZE**: Complete the generation and output the result.",
        '   - Use when: Judge recommends FINALIZE and difficulty is "good"',
        "   - Use when: Max iterations reached",
        "   - **IMPORTANT**: If Judge says `recommended_action: FINALIZE` and "
        'difficulty is "good", you SHOULD finalize!',
        "",
        "**Expand Guidance (IMPORTANT)**:",
        "1. First analyze the whole question and identify the *core entity* that is "
        "most central to finding the answer.",
        "2. Prefer expanding the core entity or the most information-carrying proper noun.",
        "3. Avoid expanding peripheral or decorative entities that do not change the "
        "main reasoning path.",
        "4. If the core entity was already expanded in a previous step, choose the "
        "next most central entity.",
        "",
        "## Decision Factors",
        "- **Solver Performance**: How many steps did the solver take? "
        "Did it find the correct answer?",
        "- **Question Quality**: Is the question grammatically correct and logically coherent?",
        "- **Difficulty Balance**: Not too easy (trivial), not too hard (unsolvable)",
        "- **Action History**: Avoid repeating failed strategies",
        "",
        "## IMPORTANT: Your analysis must include specific guidance for the next action!",
    ]
)

STRATEGIST_USER_TEMPLATE = "\n".join(
    [
        "## Current State",
        "",
        "**Original Question**: {original_question}",
        "**Current Question**: {current_question}",
        "**Correct Answer**: {answer}",
        "",
        "**Current Level**: {current_level} / {max_levels}",
        "**Iteration Count**: {iteration_count}",
        "",
        "## Action History",
        "{action_history}",
        "",
        "## Latest Difficulty Report",
        "{difficulty_report}",
        "",
        "## Last Rollback Reason",
        "{rollback_reason}",
        "",
        "## Solver Trajectory Summary",
        "{solver_summary}",
        "",
        "",
        "Output your decision in JSON format:",
        "{{",
        '  "problem_diagnosis": "What is wrong with the current question and why",',
        '  "root_cause": "The specific reason for the problem",',
        (
            '  "suggested_fix": "Specific guidance for the next agent (e.g., which entity '
            'to target, what technique to use)",'
        ),
        '  "decision": "EXPAND" | "REFINE" | "ROLLBACK" | "FINALIZE",',
        (
            '  "expand_target": "If decision is EXPAND, the exact entity phrase from the '
            'Current Question to search and replace; otherwise empty string",'
        ),
        (
            '  "rollback_reason": "If decision is ROLLBACK, a short reason for rollback; '
            'otherwise empty string",'
        ),
        '  "confidence": "high" | "medium" | "low",',
        '  "reasoning": "Full step-by-step analysis"',
        "}}",
        "",
        "Return ONLY JSON.",
    ]
)

ATOMIC_QA_PROMPT = "\n".join(
    [
        "You are an expert at generating atomic QA pairs from evidence.",
        "",
        'Given search snippets about "{target_entity}", generate ONE atomic QA such that:',
        "- The atomic question is about a specific fact in the snippets",
        '- The atomic answer MUST be exactly "{target_entity}"',
        "- The atomic question MUST NOT contain the answer string verbatim",
        "- The atomic question should be clear, solvable, and require lookup",
        "- You must also identify a bridge entity used in the atomic question "
        "(a concrete entity from snippets)",
        "- The bridge fact must be at least 50 characters and 1-2 sentences, "
        "and MUST mention the bridge entity",
        "- Prefer the bridge entity to be NEW and not already central to the original question",
        "- Prefer a bridge entity from a different domain (e.g., "
        "archive/institution/award body/biographical source),",
        "  rather than repeating obvious identifiers like title/setting/characters",
        "- Prefer a NEW relation type (e.g., archive location, education, employer, "
        "biographical event)",
        '  over "is this work/setting/character"',
        "",
        "## Search Results:",
        "{search_snippets}",
        "",
        "## Input",
        "Target Entity (must be the atomic answer): {target_entity}",
        "Original Question: {question}",
        "Forbidden entities (MUST NOT appear in atomic question or bridge entity): "
        "{forbidden_entities}",
        "",
        "## Output JSON",
        "{{",
        '  "atomic_question": "a question whose answer is the target entity",',
        '  "atomic_answer": "{target_entity}",',
        '  "bridge_entity": "the key entity referenced in the atomic question",',
        '  "relationship": "how bridge_entity relates to target",',
        '  "bridge_fact": "1-2 sentences factual statement mentioning bridge_entity and target",',
        '  "reasoning": "why this atomic QA supports a multi-hop merge"',
        "}}",
    ]
)

REWRITE_PROMPT = "\n".join(
    [
        'Rewrite the question by replacing "{target_entity}" with a clause derived '
        'from the atomic question.',
        "",
        "## Requirements:",
        "1. The rewritten question MUST incorporate the atomic question's clue so that "
        "solving it is required.",
        '2. The rewritten question MUST NOT mention "{target_entity}" directly.',
        '3. The rewritten question MUST have the same answer: "{answer}".',
        "4. The rewritten question MUST be grammatically correct and natural.",
        "5. Keep the replacement phrase concise but informative.",
        "6. **CRITICAL**: The replacement phrase must NOT repeat any other entities "
        "already in the question.",
        "7. The rewritten question must have a **unique, unambiguous answer**.",
        "8. Avoid **pseudo multi-hop**: do NOT include the answer or obvious "
        "near-synonyms of the answer.",
        "9. The rewritten question MUST NOT include any forbidden entities listed below.",
        "",
        "## Atomic QA:",
        "- Atomic Question: {atomic_question}",
        "- Atomic Answer: {target_entity}",
        "- Bridge Entity: {bridge_entity}",
        "- Bridge Fact: {bridge_fact}",
        "",
        "## Example:",
        'Original: "What genre is Our Town by Thornton Wilder?"',
        'Target: "Thornton Wilder"',
        'Atomic Question: "Who wrote the 1927 novel The Bridge of San Luis Rey?"',
        'Rewritten: "What genre is Our Town by the author who wrote the 1927 novel '
        'The Bridge of San Luis Rey?"',
        "",
        "## Input",
        "Original Question: {question}",
        "Answer: {answer}",
        "Forbidden entities (MUST NOT appear in the replacement or new question): "
        "{forbidden_entities}",
        "",
        "## Output JSON",
        "{{",
        '    "replacement_phrase": "the phrase that replaces {target_entity} (must not '
        'repeat other entities in question)",',
        '    "new_question": "the complete rewritten question"',
        "}}",
    ]
)

REFINE_SYSTEM_PROMPT = "\n".join(
    [
        "You are a question difficulty enhancer for multi-hop question generation.",
        "Your task is to FUZZ (obscure) information in the question to make it more challenging.",
        "",
        "## FUZZ Rules (strict)",
        "1. **Fuzz exactly ONE piece of information** in the question (one phrase or one entity).",
        "2. The resulting question must remain **clear** and have a **unique answer**.",
        "3. Do NOT leave obvious direct hints that make the answer trivial.",
        "",
        "## Techniques You Can Use",
        "1. **Remove Direct Hints**: Remove or soften the most obvious clue.",
        "2. **Attribute Substitution**: Replace a name with a distinctive attribute or descriptor.",
        "3. **Generalization**: Use a broader but still accurate description.",
        "",
        "## FORBIDDEN",
        "- Do NOT introduce constraints that make the question ambiguous or unsolvable",
        "- Do NOT contradict the Known Facts",
        "- Do NOT change the answer",
        "",
        "## CRITICAL Rules",
        "1. The answer MUST remain exactly the same",
        "2. The question must still be answerable with correct reasoning",
    ]
)

REFINE_USER_TEMPLATE = """## Current Question
{current_question}

## Correct Answer (MUST NOT CHANGE)
{answer}

## Known Facts (Optional context — you may use these, but are not required to)
{known_facts}

## Strategist Analysis
**Problem Diagnosis**: {problem_diagnosis}
**Suggested Fix**: {suggested_fix}

## Previous Attempts
{previous_attempts}

## Task
FUZZ this question to make it harder while keeping the same answer: {answer}

Key requirements:
1. The answer must still be: {answer}
2. Fuzz exactly ONE piece of information (one phrase or one entity)
3. The question must remain clear and have a unique answer
4. Do NOT leave obvious direct hints that make the answer trivial
5. Do NOT contradict the Known Facts (if any)

Output Format:
{{
  "refined_question": "Your rewritten question here",
  "technique_used": "remove_direct_hints" | "attribute_substitution" | "generalization",
  "fact_used": "Which fact from Known Facts you used (if any), or 'none' if just removing hints",
  "reasoning": "Brief explanation of what you fuzzed and why"
}}

Return ONLY JSON."""

FAST_VERIFY_SYSTEM_PROMPT = "\n".join(
    [
        "You are simulating a search-based solver.",
        "You MUST NOT use internal knowledge to answer. Assume you can only answer "
        "after searching.",
        "Your job is to estimate how many searches are needed and provide a plausible "
        "answer based on hypothetical search results.",
        "Return ONLY JSON. No extra text.",
    ]
)

FAST_VERIFY_USER_TEMPLATE = """Question:
{question}

Requirements:
1) Propose a realistic search plan: one query per step.
2) Estimate the number of searches required (integer >= 1).
3) Provide a final answer based on the hypothetical search results (do not use
   internal knowledge).
4) Include a short evidence phrase that could plausibly appear in a search snippet.

Output JSON with this schema:
{{
  "estimated_steps": <int>,
  "search_plan": ["query 1", "query 2", "..."],
  "answer": "final answer",
  "evidence": "short phrase",
  "needs_search": true/false
}}
Return ONLY JSON."""


class SearchTaskEvolverOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Evolve seed QA pairs into verified multi-hop search tasks.

    Each input row needs ``id``, ``question`` and ``answer``. Rows that pass
    the final quality gate (solver-correct with ``good`` difficulty) become
    task candidates carrying provenance, difficulty report and final-eval
    statistics; other rows are dropped unless ``keep_filtered`` is set.
    """

    name = "search_task_evolve"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend = backend
        self.toolset = SearchToolset(backend, self.config)
        self.max_levels = int(self.config.get("max_levels", 4))
        self.max_iterations = int(self.config.get("max_iterations", 20))
        self.fast_verify = bool(self.config.get("fast_verify", False))
        self.fast_verify_runs = int(self.config.get("fast_verify_runs", 4))
        self.final_eval_runs = int(self.config.get("final_eval_runs", 4))
        self.solver_max_steps = int(self.config.get("solver_max_steps", 12))
        self.keep_filtered = bool(self.config.get("keep_filtered", False))
        # Independent evolution rounds per seed (original --num_generations).
        self.num_generations = max(1, int(self.config.get("num_generations", 1)))

    # ------------------------------------------------------------------
    # Agent steps
    # ------------------------------------------------------------------

    def _strategist_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        report = state["difficulty_report"]
        if report:
            difficulty_text = (
                f"\n- Solver Steps: {report.get('total_steps', 'N/A')}"
                f"\n- Answer Correct: {report.get('is_correct', 'N/A')}"
                f"\n- Difficulty Level: {report.get('difficulty_level', 'N/A')}"
                f"\n- Has Shortcut: {report.get('has_shortcut', 'N/A')}"
                f"\n- Judge Recommended Action: {report.get('recommended_action', 'N/A')}"
                f"\n- Reason: {report.get('reason', 'N/A')}"
                f"\n- Suggestions: {report.get('suggestions', [])}\n"
            )
        else:
            difficulty_text = "No difficulty report yet (first iteration)"
        action_history = state["action_history"]
        user_prompt = STRATEGIST_USER_TEMPLATE.format(
            original_question=state["seed_question"],
            current_question=state["current_question"],
            answer=state["answer"],
            current_level=state["current_level"],
            max_levels=self.max_levels,
            iteration_count=state["iteration_count"],
            action_history=(
                "\n".join(f"- {a}" for a in action_history)
                if action_history
                else "No previous actions"
            ),
            difficulty_report=difficulty_text,
            rollback_reason=state.get("last_rollback_reason") or "None",
            solver_summary=summarize_trajectory_for_strategist(state["solver_trajectory"]),
        )
        try:
            content = call_role(
                self.backend,
                self.config,
                ROLE_STRATEGIST,
                user_prompt,
                STRATEGIST_SYSTEM_PROMPT,
                temperature=0.0,
            )
            result = parse_json_safely(content)
            if not isinstance(result, dict):
                raise ValueError("strategist output is not a JSON object")
            action = str(result.get("decision", "FINALIZE")).upper()
            if action not in ACTIONS:
                logger.warning("Invalid strategist action '%s', defaulting to FINALIZE", action)
                action = "FINALIZE"
            return {
                "action": action,
                "reasoning": result.get("reasoning", ""),
                "problem_diagnosis": result.get("problem_diagnosis", ""),
                "suggested_fix": result.get("suggested_fix", ""),
                "expand_target": result.get("expand_target", ""),
                "rollback_reason": result.get("rollback_reason", ""),
            }
        except Exception as exc:  # noqa: BLE001 - fall back to rule-based decision
            logger.error("Strategist failed, using rule fallback: %s", exc)
            return self._fallback_decision(state)

    def _fallback_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "reasoning": "",
            "problem_diagnosis": "",
            "suggested_fix": "",
            "expand_target": "",
            "rollback_reason": "",
        }
        report = state["difficulty_report"]
        if state["iteration_count"] >= self.max_iterations:
            return {**base, "action": "FINALIZE", "reasoning": "Max iterations reached"}
        if not report:
            return {**base, "action": "EXPAND", "reasoning": "First iteration"}
        difficulty_level = report.get("difficulty_level")
        if difficulty_level == "good":
            return {**base, "action": "FINALIZE", "reasoning": "Good difficulty achieved"}
        if difficulty_level == "too_easy" and state["current_level"] < self.max_levels:
            return {**base, "action": "EXPAND", "reasoning": "Too easy"}
        if difficulty_level == "broken":
            rollbacks = sum(1 for a in state["action_history"] if "ROLLBACK" in a)
            if rollbacks < 2:
                return {**base, "action": "ROLLBACK", "reasoning": "Broken logic"}
        return {**base, "action": "FINALIZE", "reasoning": "Default"}

    def _expand(self, state: Dict[str, Any], decision: Dict[str, Any]) -> bool:
        """Atomic-QA merge expansion (ExpandAgent V3). Returns success."""
        target = str(decision.get("expand_target") or "").strip()
        if not target:
            logger.warning("[Expand] No target entity provided by strategist")
            return False
        question = state["current_question"]
        if target.lower() not in question.lower():
            logger.warning("[Expand] Target '%s' does not appear verbatim in the question", target)

        # Forbidden entities: historical targets/bridges, used bridges, target, answer.
        forbidden: List[str] = []
        for layer in state["layers"]:
            for key in ("target_entity", "bridge_entity"):
                if layer.get(key):
                    forbidden.append(str(layer[key]))
        forbidden.extend(state["used_bridges"])
        forbidden.append(target)
        if state["answer"]:
            forbidden.append(state["answer"])
        forbidden = [e for e in dict.fromkeys(e.strip() for e in forbidden if e) if e]
        forbidden_text = ", ".join(forbidden) if forbidden else "(none)"

        # Step 1: search the target itself with long snippets.
        try:
            search_results = self.toolset.search(target, long_snippet=True)
            snippets = [
                r.get("snippet", "") for r in search_results.get("results", []) if r.get("snippet")
            ]
        except Exception as exc:  # noqa: BLE001 - treat as expansion failure
            logger.warning("[Expand] Search failed for '%s': %s", target, exc)
            return False
        if not snippets:
            logger.warning("[Expand] No search results for '%s'", target)
            return False

        # Step 2: generate an atomic QA whose answer is the target.
        atomic_prompt = ATOMIC_QA_PROMPT.format(
            target_entity=target,
            question=question,
            search_snippets="\n".join(f"- {s}" for s in snippets),
            forbidden_entities=forbidden_text,
        )
        atomic = self._call_synthesis_json(
            atomic_prompt, "You are an expert at generating atomic QA. Output valid JSON only."
        )
        if not atomic:
            return False
        required = (
            "atomic_question",
            "atomic_answer",
            "bridge_entity",
            "relationship",
            "bridge_fact",
        )
        if not all(key in atomic for key in required):
            logger.warning("[Expand] Atomic QA missing required fields")
            return False
        atomic_question = str(atomic["atomic_question"]).strip()
        atomic_answer = str(atomic["atomic_answer"]).strip()
        bridge = str(atomic["bridge_entity"]).strip()
        bridge_fact = str(atomic["bridge_fact"]).strip()
        if not atomic_answer or atomic_answer.lower() != target.strip().lower():
            logger.warning("[Expand] atomic_answer mismatch (got '%s')", atomic_answer)
            return False
        if not atomic_question or target.strip().lower() in atomic_question.lower():
            logger.warning("[Expand] atomic_question invalid or contains target")
            return False
        if bridge and bridge.lower() in [e.lower() for e in forbidden]:
            logger.warning("[Expand] bridge entity is forbidden")
            return False
        if len(bridge_fact) < 50:
            logger.warning("[Expand] bridge_fact too short")
            return False

        # Step 3: rewrite the question, replacing the target with the clue.
        rewrite_prompt = REWRITE_PROMPT.format(
            question=question,
            answer=state["answer"],
            target_entity=target,
            atomic_question=atomic_question,
            bridge_entity=bridge,
            bridge_fact=bridge_fact,
            forbidden_entities=forbidden_text,
        )
        rewrite = self._call_synthesis_json(
            rewrite_prompt, "You are an expert at rewriting questions. Output valid JSON only."
        )
        new_question = str((rewrite or {}).get("new_question") or "").strip()
        if not new_question:
            logger.warning("[Expand] Rewrite produced empty question")
            return False

        next_level = state["current_level"] + 1
        state["layers"].append(
            {
                "level": next_level,
                "target_entity": target,
                "bridge_entity": bridge,
                "bridge_fact": bridge_fact,
                "relationship": atomic.get("relationship", ""),
                "rewritten_question": new_question,
            }
        )
        state["graph_edges"].append(
            {
                "from_id": target,
                "to_id": bridge,
                "relation": atomic.get("relationship", ""),
            }
        )
        state["used_bridges"].append(bridge)
        state["current_level"] = next_level
        state["current_question"] = new_question
        logger.info("[Expand] L%d: '%s' -> '%s'", next_level, target, bridge)
        return True

    def _refine(self, state: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """FUZZ the question without adding hops. Failure keeps the question."""
        known_facts = [
            layer["bridge_fact"] for layer in state["layers"] if layer.get("bridge_fact")
        ]
        previous_refines = sum(1 for a in state["action_history"] if "REFINE" in a)
        user_prompt = REFINE_USER_TEMPLATE.format(
            current_question=state["current_question"],
            answer=state["answer"],
            known_facts=(
                "\n".join(f"- {fact}" for fact in known_facts)
                if known_facts
                else (
                    "(No known facts available - you can only use "
                    "'remove_direct_hints' or 'generalization' techniques)"
                )
            ),
            problem_diagnosis=decision.get("problem_diagnosis") or "No specific diagnosis provided",
            suggested_fix=decision.get("suggested_fix") or "No specific suggestion provided",
            previous_attempts=(
                (
                    f"This is attempt #{previous_refines + 1}. Previous attempts did not "
                    "sufficiently increase difficulty."
                )
                if previous_refines
                else "This is the first refinement attempt."
            ),
        )
        try:
            content = call_role(
                self.backend, self.config, ROLE_SYNTHESIS, user_prompt, REFINE_SYSTEM_PROMPT
            )
        except Exception as exc:  # noqa: BLE001 - keep the current question
            logger.warning("[Refine] LLM call failed: %s", exc)
            return
        result = parse_json_safely(content)
        refined = str((result or {}).get("refined_question") or "").strip()
        if refined:
            logger.info("[Refine] technique=%s", (result or {}).get("technique_used"))
            state["current_question"] = refined

    def _verify(self, state: Dict[str, Any]) -> None:
        """Run the solver (or fast verify) and store the trajectory."""
        if self.fast_verify:
            state["solver_trajectory"] = self._fast_verify(state)
            return
        try:
            state["solver_trajectory"] = solve_search_task(
                self.backend,
                self.config,
                state["current_question"],
                self.toolset,
                self.solver_max_steps,
            )
        except Exception as exc:  # noqa: BLE001 - empty trajectory means broken
            logger.error("[Verify] Solver failed: %s", exc)
            state["solver_trajectory"] = []
        state["fast_verify_summary"] = None

    def _fast_verify(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cheap plan-based verification instead of a full solver rollout."""
        import json as _json

        results = []
        for _ in range(self.fast_verify_runs):
            content = call_role(
                self.backend,
                self.config,
                ROLE_FAST_VERIFY,
                FAST_VERIFY_USER_TEMPLATE.format(question=state["current_question"]),
                FAST_VERIFY_SYSTEM_PROMPT,
                temperature=0.0,
            )
            parsed = parse_json_safely(content) or {}
            try:
                parsed["estimated_steps"] = int(parsed.get("estimated_steps", 0))
            except (TypeError, ValueError):
                parsed["estimated_steps"] = 0
            parsed["answer"] = str(parsed.get("answer", ""))
            parsed["is_correct"] = answer_equivalent(
                self.backend,
                self.config,
                state["current_question"],
                parsed["answer"],
                state["answer"],
            )
            results.append(parsed)
        steps = [r["estimated_steps"] for r in results]
        median_steps = int(statistics.median(steps)) if steps else 0
        correct_count = sum(1 for r in results if r["is_correct"])
        selected_answer = next(
            (r["answer"] for r in results if r["is_correct"]),
            results[0]["answer"] if results else "",
        )
        summary = {
            "num_runs": self.fast_verify_runs,
            "median_steps": median_steps,
            "correct_count": correct_count,
            "any_correct": correct_count > 0,
        }
        state["fast_verify_summary"] = summary
        return [
            {
                "role": "assistant",
                "content": (
                    "[FastVerify Summary]\n"
                    f"{_json.dumps(summary, ensure_ascii=False)}\n"
                    f"<answer>{selected_answer}</answer>"
                ),
            }
        ]

    def _judge(self, state: Dict[str, Any]) -> None:
        report = judge_trajectory(
            self.backend,
            self.config,
            state["seed_question"],
            state["current_question"],
            state["answer"],
            state["solver_trajectory"],
        )
        # Align with fast-verify statistics when available.
        summary = state.get("fast_verify_summary")
        if isinstance(summary, dict):
            if summary.get("median_steps", 0) > 0:
                report["total_steps"] = summary["median_steps"]
            report["is_correct"] = bool(summary.get("any_correct"))
        state["difficulty_report"] = report
        logger.info(
            "[Judge] correct=%s steps=%s difficulty=%s recommended=%s",
            report.get("is_correct"),
            report.get("total_steps"),
            report.get("difficulty_level"),
            report.get("recommended_action"),
        )

    def _rollback(self, state: Dict[str, Any], reason: str) -> bool:
        """Revert the last change. Returns False when nothing can be reverted."""
        state["rollback_history"].append(reason)
        state["last_rollback_reason"] = reason
        gate_failed = (state.get("quality_gate_result") or {}).get("pass") is False
        if gate_failed and state.get("last_action") == "REFINE":
            state["current_question"] = state["pre_question"]
            state["last_action"] = ""
            logger.info("[Rollback] REFINE rollback: restored pre_question")
            return True
        if state["current_level"] <= 0 or len(state["layers"]) <= 1:
            logger.warning("[Rollback] Cannot rollback further")
            return False
        state["layers"].pop()
        state["graph_edges"] = state["graph_edges"][: state["current_level"] - 1]
        state["used_bridges"] = state["used_bridges"][: state["current_level"] - 1]
        state["current_level"] -= 1
        prev_question = (
            state["pre_question"]
            or state["layers"][-1].get("rewritten_question")
            or state["seed_question"]
        )
        state["current_question"] = prev_question
        state["pre_question"] = prev_question
        state["last_action"] = ""
        logger.info("[Rollback] Reverted to level %d", state["current_level"])
        return True

    def _call_synthesis_json(
        self, user_prompt: str, system_prompt: str
    ) -> Optional[Dict[str, Any]]:
        try:
            content = call_role(
                self.backend, self.config, ROLE_SYNTHESIS, user_prompt, system_prompt
            )
        except Exception as exc:  # noqa: BLE001 - propagate as parse failure
            logger.warning("Synthesis LLM call failed: %s", exc)
            return None
        return parse_json_safely(content)

    # ------------------------------------------------------------------
    # Evolve loop
    # ------------------------------------------------------------------

    def _evolve_seed(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seed_id = str(row.get("id", "seed"))
        seed_question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not seed_question or not answer:
            logger.warning("Seed %s missing question/answer, skipped.", seed_id)
            return None

        state: Dict[str, Any] = {
            "seed_question": seed_question,
            "answer": answer,
            "current_question": seed_question,
            "current_level": 0,
            "layers": [{"level": 0, "rewritten_question": seed_question}],
            "graph_edges": [],
            "used_bridges": [],
            "iteration_count": 0,
            "action_history": [],
            "difficulty_report": {},
            "solver_trajectory": [],
            "fast_verify_summary": None,
            "quality_gate_result": {},
            "last_action": "",
            "pre_question": seed_question,
            "last_rollback_reason": "",
            "rollback_history": [],
            "strategist_log": [],
        }

        while True:
            if state["iteration_count"] >= self.max_iterations:
                state["action_history"].append("FINALIZE(max_iter)")
                break

            decision = self._strategist_decide(state)
            action = decision["action"]
            state["action_history"].append(
                f"{action}(llm:{(decision.get('problem_diagnosis') or 'no_diag')[:30]})"
            )
            state["strategist_log"].append({"iteration": state["iteration_count"], **decision})
            state["last_action"] = action
            state["pre_question"] = state["current_question"]
            state["last_rollback_reason"] = (
                (
                    decision.get("rollback_reason")
                    or decision.get("problem_diagnosis")
                    or "strategist_rollback"
                )
                if action == "ROLLBACK"
                else ""
            )

            if action == "FINALIZE":
                break

            if action == "ROLLBACK":
                if not self._rollback(
                    state, state["last_rollback_reason"] or "strategist_rollback"
                ):
                    break
                continue

            if action == "EXPAND":
                state["iteration_count"] += 1
                if not self._expand(state, decision):
                    # Expansion failure ends the loop, matching the original break.
                    break
            elif action == "REFINE":
                state["iteration_count"] += 1
                self._refine(state, decision)

            # Quality gate before spending solver budget.
            last_layer = state["layers"][-1]
            gate = run_quality_gate(
                self.backend,
                self.config,
                seed_question,
                state["current_question"],
                answer,
                {
                    "target_entity": last_layer.get("target_entity"),
                    "bridge_entity": last_layer.get("bridge_entity"),
                    "bridge_fact": last_layer.get("bridge_fact"),
                },
            )
            state["quality_gate_result"] = gate
            if not gate.get("pass", True):
                reason = gate.get("rollback_reason") or "quality_gate_failed"
                if not self._rollback(state, reason):
                    break
                continue

            self._verify(state)
            self._judge(state)

        return self._finalize(row, state)

    def _finalize(self, row: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        report = state["difficulty_report"]
        base = {
            "id": f"{row.get('id', 'seed')}-evolved",
            "question": state["current_question"],
            "answer": state["answer"],
            "seed_id": row.get("id"),
            "seed_question": state["seed_question"],
            "hops": len(state["layers"]) - 1,
            "path": state["graph_edges"],
            "entity_set": _collect_entities(state),
            # Full per-level provenance (original ``seed_trace`` / layers).
            "layers": state["layers"],
            "iteration_count": state["iteration_count"],
            "final_level": state["current_level"],
            "difficulty_report": report,
            "action_history": state["action_history"],
            "rollback_history": state["rollback_history"],
            "strategist_log": state["strategist_log"],
        }
        is_correct = bool(report.get("is_correct", False))
        difficulty_level = report.get("difficulty_level", "unknown")
        if not is_correct or difficulty_level != "good":
            reason = "incorrect_answer" if not is_correct else f"difficulty_{difficulty_level}"
            logger.warning("[Finalize] FILTERED %s: %s", base["id"], reason)
            if self.keep_filtered:
                return {**base, "evolve_status": "filtered", "filter_reason": reason}
            return None

        try:
            eval_result = evaluate_task(
                self.backend,
                self.config,
                state["current_question"],
                state["answer"],
                self.toolset,
                num_runs=self.final_eval_runs,
                max_steps=self.solver_max_steps,
            )
        except Exception as exc:  # noqa: BLE001 - keep the candidate on eval failure
            logger.error(
                "[Finalize] Final eval failed for %s, saving without stats: %s",
                base["id"],
                exc,
            )
            return {
                **base,
                "evolve_status": "saved",
                "final_eval": {"error": str(exc), "num_runs": 0, "runs": []},
            }
        logger.info(
            "[Finalize] SAVED %s: hops=%d accuracy=%.2f avg_turns=%.1f",
            base["id"],
            base["hops"],
            eval_result["accuracy"],
            eval_result["avg_turns"],
        )
        return {
            **base,
            "evolve_status": "saved",
            "final_eval": {
                "accuracy": eval_result["accuracy"],
                "avg_turns": eval_result["avg_turns"],
                "num_runs": self.final_eval_runs,
                "runs": eval_result["runs"],
            },
        }

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_workers = int(self.config.get("max_workers", 2))
        max_tasks = self.config.get("max_tasks")
        if max_tasks is not None:
            input_data = input_data[: int(max_tasks)]

        # Resume support (original entry-script behavior): skip seeds whose
        # ids already appear in a previous stage-1 output file.
        completed_ids = _load_completed_seed_ids(self.config.get("resume_from"))
        if completed_ids:
            before = len(input_data)
            input_data = [r for r in input_data if str(r.get("id")) not in completed_ids]
            logger.info(
                "Resume: skipped %d/%d seeds already present in resume_from.",
                before - len(input_data),
                before,
            )

        # num_generations independent evolution rounds per seed, with the
        # original ``#genN`` id suffix so downstream rows stay unique.
        jobs: List[Dict[str, Any]] = []
        for row in input_data:
            for gen_idx in range(self.num_generations):
                job = dict(row)
                job["_gen_index"] = gen_idx
                jobs.append(job)

        results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._evolve_seed, job): idx for idx, job in enumerate(jobs)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad seed must not kill the run
                    logger.error("Evolve failed for job %d: %s", idx, exc)
                    continue
                if result is not None:
                    gen_idx = jobs[idx]["_gen_index"]
                    result["gen_index"] = gen_idx
                    if self.num_generations > 1:
                        result["id"] = f"{result['id']}#gen{gen_idx}"
                results[idx] = result
        output = [r for r in results if r is not None]
        saved = sum(1 for r in output if r.get("evolve_status") == "saved")
        logger.info(
            "Task evolution finished: %d saved, %d kept-filtered, %d/%d jobs "
            "(%d seeds x %d generations).",
            saved,
            len(output) - saved,
            len(output),
            len(jobs),
            len(input_data),
            self.num_generations,
        )
        return output


def _load_completed_seed_ids(resume_path: Optional[str]) -> set:
    """Read seed ids already covered by a previous evolve output file."""
    import json
    import os

    completed: set = set()
    if not resume_path or not os.path.exists(str(resume_path)):
        return completed
    with open(str(resume_path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seed_id = obj.get("seed_id") or str(obj.get("id", "")).split("#gen")[0].replace(
                "-evolved", ""
            )
            if seed_id:
                completed.add(str(seed_id))
    return completed


def _collect_entities(state: Dict[str, Any]) -> List[str]:
    """Collect the deduplicated entity list of the whole evolution."""
    entities: List[str] = []
    for layer in state["layers"]:
        for key in ("target_entity", "bridge_entity"):
            value = layer.get(key)
            if value:
                entities.append(str(value).strip())
    for edge in state["graph_edges"]:
        for key in ("from_id", "to_id"):
            value = edge.get(key)
            if value:
                entities.append(str(value).strip())
    for value in state["used_bridges"]:
        if value:
            entities.append(str(value).strip())
    if state["answer"]:
        entities.append(str(state["answer"]).strip())
    unique: List[str] = []
    for entity in entities:
        if entity and entity not in unique:
            unique.append(entity)
    return unique
