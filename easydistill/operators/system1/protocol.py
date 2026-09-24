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

"""Typed-decision question and schema protocol for system-1 distillation.

System-1 distillation turns raw states into typed-decision cases for the Laya
student model. Cases carry dataset-format questions (``type`` / ``instructions``
/ ``criteria``) and gold answers as probability distributions. This module
mirrors the verbatim rendering contracts of ``laya.common`` so prompts, token
budgets, and training targets stay byte-compatible with the official
``build_training_item`` path, while adding the schema validation used to
synthesize cases from raw rows.
"""

import json
import math
from typing import Any, Dict, List, Optional, Set

QUESTION_TYPES = ("choice", "score", "noul")
MAX_FLAT_OPTIONS = 20
_NOUL_FALSE_TEXT = "no, the statement does not hold"
_NOUL_TRUE_TEXT = "yes, the statement holds"


def serialize_state(state: Any) -> str:
    """Serialize a case state for prompts (mirrors laya.common.serialize_state)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    """Render one criterion value as text (mirrors laya.common.render_criterion).

    Strings pass through; structured values become JSON with the exact
    separators laya uses, so rendered options match the student's training
    format byte for byte.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_question_options(question: Dict[str, Any]) -> List[str]:
    """Render option texts in label-index order (mirrors laya render_options).

    Noul always renders exactly two options (``false`` then ``true``); missing
    criteria fall back to laya's default phrases.
    """
    qtype = question["type"]
    criteria = question.get("criteria") or {}
    if qtype == "choice":
        return [
            key if value is None or value == "" else f"{key}: {render_criterion(value)}"
            for key, value in criteria.items()
        ]
    if qtype == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(criteria)]
    false_crit = criteria.get("false")
    true_crit = criteria.get("true")
    false_text = render_criterion(false_crit) if false_crit not in (None, "") else _NOUL_FALSE_TEXT
    true_text = render_criterion(true_crit) if true_crit not in (None, "") else _NOUL_TRUE_TEXT
    return ["false: " + false_text, "true: " + true_text]


def question_keys(question: Dict[str, Any]) -> List[str]:
    """Option keys in target-vector order (mirrors build_training_item).

    Score questions key levels by index string; the 4-level fallback for a
    non-list criteria value is kept verbatim from the official notebook so
    malformed prebuilt rows degrade identically.
    """
    qtype = question["type"]
    criteria = question.get("criteria", {})
    if qtype == "choice":
        return list(criteria.keys())
    if qtype == "score":
        n_levels = len(criteria) if isinstance(criteria, list) else 4
        return [str(i) for i in range(n_levels)]
    return ["false", "true"]


def validate_question(question: Any, context: str) -> Dict[str, Any]:
    """Validate one dataset-format question; returns it unchanged."""
    if not isinstance(question, dict):
        raise ValueError(f"{context}: question must be an object")
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(
            f"{context}: question type must be one of {QUESTION_TYPES}, got {qtype!r}"
        )
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"{context}: 'instructions' must be a non-empty string")
    criteria = question.get("criteria", {})
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"{context}: choice 'criteria' must be a non-empty object")
        for key in criteria:
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"{context}: choice option keys must be non-empty strings, got {key!r}"
                )
    elif qtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f"{context}: score 'criteria' must be a non-empty list")
    else:
        if not isinstance(criteria, dict):
            raise ValueError(f"{context}: noul 'criteria' must be an object")
        for key in criteria:
            if key not in ("false", "true"):
                raise ValueError(
                    f"{context}: noul criteria keys must be 'false'/'true', got {key!r}"
                )
    return question


def validate_schema(schema: Any) -> Dict[str, Any]:
    """Validate and normalize a workflow schema declaration.

    Returns a schema with defaults filled in (``type`` -> "choice",
    ``label_field`` -> "label") carrying exactly one of ``options`` (flat) or
    ``groups`` (hierarchical choice). Flat schemas and every group are capped
    at MAX_FLAT_OPTIONS options; larger decisions must be split into groups so
    both prompts and the student's option markers stay in budget.
    """
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    normalized: Dict[str, Any] = {}
    for field in ("name", "question", "instructions"):
        value = schema.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"schema field {field!r} must be a non-empty string")
        normalized[field] = value
    qtype = schema.get("type", "choice")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"schema 'type' must be one of {QUESTION_TYPES}, got {qtype!r}")
    normalized["type"] = qtype
    label_field = schema.get("label_field", "label")
    if not isinstance(label_field, str) or not label_field:
        raise ValueError("schema 'label_field' must be a non-empty string")
    normalized["label_field"] = label_field
    options = schema.get("options")
    groups = schema.get("groups")
    if options is not None and groups is not None:
        raise ValueError("schema cannot declare both 'options' and 'groups'; pick one")
    if groups is not None:
        normalized["groups"] = _validate_groups(groups, qtype)
    else:
        normalized["options"] = _validate_flat_options(options, qtype)
    return normalized


def _validate_flat_options(options: Any, qtype: str) -> Any:
    """Validate flat options (dict) or score levels (list) for one question type."""
    if options is None:
        if qtype == "noul":
            return {}
        raise ValueError("schema must declare 'options' or 'groups'")
    if qtype == "noul":
        if not isinstance(options, dict):
            raise ValueError("noul 'options' must be an object with 'false'/'true' descriptions")
        for key in options:
            if key not in ("false", "true"):
                raise ValueError(f"noul options may only contain 'false'/'true', got {key!r}")
        return options
    if qtype == "score":
        if not isinstance(options, list) or not options:
            raise ValueError("score 'options' must be a non-empty list of level descriptions")
        if len(options) > MAX_FLAT_OPTIONS:
            raise ValueError(
                f"score schema declares {len(options)} levels; at most "
                f"{MAX_FLAT_OPTIONS} are supported"
            )
        return options
    if not isinstance(options, dict) or not options:
        raise ValueError("choice 'options' must be a non-empty object")
    for key in options:
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"choice option keys must be non-empty strings, got {key!r} "
                "(quote keys in YAML/JSON)"
            )
    if len(options) > MAX_FLAT_OPTIONS:
        raise ValueError(
            f"choice schema declares {len(options)} options; at most {MAX_FLAT_OPTIONS} are "
            "supported - use 'groups' to split the decision hierarchically"
        )
    return options


def _validate_groups(groups: Any, qtype: str) -> Dict[str, Dict[str, Any]]:
    """Validate hierarchical choice groups and normalize their definitions."""
    if qtype != "choice":
        raise ValueError("'groups' is only supported with type 'choice'")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("'groups' must be a non-empty object mapping group names to definitions")
    validated: Dict[str, Dict[str, Any]] = {}
    seen_keys: Set[str] = set()
    for group_name, group_def in groups.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"group names must be non-empty strings, got {group_name!r}")
        if not isinstance(group_def, dict):
            raise ValueError(f"group {group_name!r} must be an object")
        group_options = group_def.get("options")
        if not isinstance(group_options, dict) or not group_options:
            raise ValueError(f"group {group_name!r} must declare a non-empty 'options' object")
        for key in group_options:
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"group {group_name!r} option keys must be non-empty strings, got {key!r}"
                )
        if len(group_options) > MAX_FLAT_OPTIONS:
            raise ValueError(
                f"group {group_name!r} declares {len(group_options)} options; at most "
                f"{MAX_FLAT_OPTIONS} are supported per group"
            )
        overlap = seen_keys & set(group_options)
        if overlap:
            raise ValueError(
                f"option keys must be unique across groups; "
                f"{sorted(overlap)[0]!r} appears in more than one group"
            )
        seen_keys.update(group_options)
        description = group_def.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"group {group_name!r} 'description' must be a string or null")
        group_instructions = group_def.get("instructions")
        if group_instructions is not None and (
            not isinstance(group_instructions, str) or not group_instructions.strip()
        ):
            raise ValueError(
                f"group {group_name!r} 'instructions' must be a non-empty string or null"
            )
        validated[group_name] = {
            "options": group_options,
            "description": description,
            "instructions": group_instructions,
        }
    return validated


def gold_from_label(label: Any, question: Dict[str, Any]) -> Dict[str, float]:
    """Build a one-hot gold distribution from a dataset label."""
    qtype = question["type"]
    keys = question_keys(question)
    match = str(label).strip().lower() if qtype == "noul" else str(label)
    if match not in keys:
        raise ValueError(
            f"label {label!r} is not a valid option for {qtype} questions; expected one of {keys}"
        )
    return {key: (1.0 if key == match else 0.0) for key in keys}


def normalize_probabilities(values: List[Any]) -> Optional[List[float]]:
    """Normalize numeric values to a distribution, or None when invalid."""
    numeric: List[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number < 0 or not math.isfinite(number):
            return None
        numeric.append(number)
    total = sum(numeric)
    if total <= 0:
        return None
    return [value / total for value in numeric]


def argmax_key(probabilities: Dict[str, float]) -> str:
    """Return the first key holding the maximum value."""
    if not probabilities:
        raise ValueError("argmax_key requires a non-empty mapping")
    best_key = next(iter(probabilities))
    best_value = probabilities[best_key]
    for key, value in probabilities.items():
        if value > best_value:
            best_key = key
            best_value = value
    return best_key


def is_prebuilt_case(row: Dict[str, Any]) -> bool:
    """Return True when a raw row already carries prebuilt questions."""
    questions = row.get("questions")
    if isinstance(questions, dict):
        return True
    if isinstance(questions, str):
        try:
            return isinstance(json.loads(questions), dict)
        except json.JSONDecodeError:
            return False
    return False


def parse_case_field(value: Any, field: str) -> Any:
    """Parse a case field, tolerating malformed JSON only for state."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        if field == "state":
            return value
        raise ValueError(
            f"case field {field!r} must be valid JSON when given as a string"
        ) from exc


def validate_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one case row and parse its state/questions/gold fields."""
    context = f"case {str(row.get('id', '<unknown>'))!r}"
    for field in ("state", "questions"):
        if field not in row:
            raise ValueError(f"{context}: missing required field {field!r}")
    state = parse_case_field(row["state"], "state")
    questions = parse_case_field(row["questions"], "questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{context}: 'questions' must parse to a non-empty object")
    for question_id, question in questions.items():
        validate_question(question, f"{context} question {question_id!r}")
    gold = parse_case_field(row.get("gold", {}), "gold")
    if not isinstance(gold, dict):
        raise ValueError(f"{context}: 'gold' must parse to an object")
    return {"state": state, "questions": questions, "gold": gold}


def build_schema_questions(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the coarse dataset questions from a validated schema."""
    groups = schema.get("groups")
    if groups is not None:
        criteria = {group_name: group["description"] for group_name, group in groups.items()}
    else:
        criteria = schema.get("options") or {}
    return {
        schema["question"]: {
            "type": schema["type"],
            "instructions": schema["instructions"],
            "criteria": criteria,
        }
    }


def build_fine_question(schema: Dict[str, Any], group: str) -> Dict[str, Any]:
    """Build the fine-grained choice question for one schema group."""
    group_def = schema["groups"][group]
    instructions = group_def.get("instructions") or schema["instructions"]
    return {"type": "choice", "instructions": instructions, "criteria": group_def["options"]}


def fine_question_id(question_id: str, group: str) -> str:
    return f"{question_id}::{group}"
