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

"""Build typed-decision cases from raw rows or prebuilt case files.

The build stage joins raw dataset rows with a workflow schema into the case
format every later stage shares: ``{id, workflow, state, questions, gold}``
(plus ``source`` while ``keep_source`` is on). Rows that already carry a
``questions`` field are prebuilt cases and pass through validation unchanged.

The token budget check mirrors ``laya.common.build_sequence`` token by token.
It loads the student's real tokenizer when ``transformers`` and
``huggingface_hub`` are installed, falls back to a chars-per-token heuristic
otherwise, and rejects up front anything build_sequence would silently
truncate: long options, oversized question heads, and states that do not fit
``max_len``.
"""

import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from easydistill.operators.base import Operator
from easydistill.utils import (
    config_list,
    config_number,
    config_section,
    load_config,
    progress,
)

from .protocol import (
    build_fine_question,
    build_schema_questions,
    fine_question_id,
    gold_from_label,
    is_prebuilt_case,
    render_question_options,
    serialize_state,
    validate_case_row,
    validate_schema,
)

logger = logging.getLogger(__name__)

#: Case fields this module owns. A raw row carrying them (short of being a
#: full prebuilt case) keeps them out of the state so nothing leaks sideways.
_RESERVED_ROW_KEYS = frozenset({"questions", "gold", "workflow", "source"})

#: Canonical case keys; a prebuilt row's remaining fields become its source.
_CASE_KEYS = frozenset({"id", "workflow", "state", "questions", "gold", "source"})

#: Budget defaults matching the student's training config (max_len 1024,
#: head_max_len 256) and build_sequence's hard 48-token per-option cap.
DEFAULT_TOKEN_BUDGET: Dict[str, Any] = {
    "check": True,
    "tokenizer_id": "convaiinnovations/laya",
    "head_max_len": 256,
    "max_len": 1024,
    "option_max_tokens": 48,
    "chars_per_token": 4.0,
}


def _fix_tokenizer_config(path: str) -> None:
    """Make a laya snapshot's tokenizer_config.json loadable (mirrors laya.agent).

    Checkpoints built with recent ``tokenizers`` releases write
    ``tokenizer_class: "TokenizersBackend"`` and a list-valued
    ``extra_special_tokens``; older ``transformers`` releases refuse both, so
    the fields are normalized in place exactly as
    ``laya.agent._fix_tokenizer_config`` does before AutoTokenizer loads.
    """
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tokenizer_config = json.load(f)
        changed = False
        if tokenizer_config.get("tokenizer_class") in (None, "TokenizersBackend"):
            tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
            tokenizer_config.pop("backend", None)
            tokenizer_config.pop("is_local", None)
            changed = True
        extra = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra, list):
            tokenizer_config["extra_special_tokens"] = {
                f"extra_{i}": token for i, token in enumerate(extra)
            }
            changed = True
        if changed:
            with open(cfg_file, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
    except Exception as exc:
        logger.debug("could not fix tokenizer config at %s: %s", cfg_file, exc)


def _load_tokenizer(tokenizer_id: str) -> Any:
    """Load the student tokenizer, or None when it is unavailable."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers/huggingface_hub are not installed; the token budget "
            "check falls back to the chars-per-token heuristic"
        )
        return None
    try:
        model_dir = snapshot_download(tokenizer_id, allow_patterns=["tokenizer/*"])
        _fix_tokenizer_config(model_dir)
        return AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    except Exception as exc:
        logger.warning(
            "could not load tokenizer %r (%s); the token budget check falls "
            "back to the chars-per-token heuristic",
            tokenizer_id,
            exc,
        )
        return None


def load_token_counter(budget: Dict[str, Any]) -> Tuple[Callable[[str], int], Optional[str]]:
    """Return ``(count_tokens, mask_token)`` for a merged token budget.

    An empty ``tokenizer_id`` selects the heuristic counter outright, which
    keeps tests offline and lets air-gapped runs opt in explicitly. The
    heuristic counts ``ceil(len(text) / chars_per_token)`` tokens.
    """
    chars_per_token = config_number(
        budget, "chars_per_token", 4.0, where="system1 token_budget"
    )
    if chars_per_token <= 0:
        logger.warning(
            "token_budget.chars_per_token = %r is not positive; using 4.0 instead.",
            budget.get("chars_per_token"),
        )
        chars_per_token = 4.0

    def heuristic_count(text: str) -> int:
        return max(1, math.ceil(len(text) / chars_per_token))

    tokenizer_id = str(budget.get("tokenizer_id") or "")
    if not tokenizer_id:
        return heuristic_count, None
    tokenizer = _load_tokenizer(tokenizer_id)
    if tokenizer is None:
        return heuristic_count, None

    def tokenizer_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    mask_token = tokenizer.mask_token
    return tokenizer_count, mask_token if isinstance(mask_token, str) else None


def check_question_budget(
    question: Dict[str, Any],
    state_text: str,
    budget: Dict[str, Any],
    count_tokens: Callable[[str], int],
    mask_token: Optional[str],
    context: str,
) -> None:
    """Reject anything ``laya.common.build_sequence`` would silently truncate.

    build_sequence never raises on oversized input. It caps every option at
    ``option_max_tokens``, shrinks the question head (and, once the options
    alone overflow ``head_max_len``, the options too) to as little as eight
    tokens, and truncates the state so the sequence fits ``max_len``. Every
    cut is silent, so this check mirrors the exact accounting and fails
    loudly instead of shipping degraded training items.
    """
    instructions = str(question["instructions"])
    options = render_question_options(question)
    if mask_token:
        instructions = instructions.replace(mask_token, " ")
        options = [option.replace(mask_token, " ") for option in options]
        state_text = state_text.replace(mask_token, " ")
    head_len = count_tokens(f"{question['type']} question: {instructions}")
    option_max_tokens = config_number(
        budget, "option_max_tokens", 48, where="system1 token_budget"
    )
    option_lens = [count_tokens(f" {option}") for option in options]
    for option, option_len in zip(options, option_lens):
        if option_len > option_max_tokens:
            raise ValueError(
                f"{context}: option {option!r} needs {option_len} tokens, over "
                f"the {option_max_tokens}-token cap; shorten the option text"
            )
    total_options = sum(option_len + 1 for option_len in option_lens)
    head_max_len = config_number(budget, "head_max_len", 256, where="system1 token_budget")
    if total_options + max(head_len, 16) > head_max_len:
        raise ValueError(
            f"{context}: question head ({head_len} tokens) plus options "
            f"({total_options} tokens) exceed head_max_len ({head_max_len}); "
            "split the decision into 'groups' or raise head_max_len"
        )
    state_len = count_tokens(state_text)
    max_len = config_number(budget, "max_len", 1024, where="system1 token_budget")
    total_len = head_len + total_options + state_len + 4
    if total_len > max_len:
        raise ValueError(
            f"{context}: full sequence needs {total_len} tokens, over max_len "
            f"({max_len}); shorten the state or raise max_len"
        )


class System1BuildCasesOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Build typed-decision cases from raw rows, or validate prebuilt ones.

    Raw rows are joined with the workflow schema into case rows; a row that
    already carries a ``questions`` field is a prebuilt case and passes
    through validation. Hierarchical labeled cases also carry the matching
    group's fine question, so elicitation and aggregation see the same
    question set the student will train on.

    Configurable fields:
      - schema / schema_path: workflow schema object, or a JSON/YAML file path.
      - id_field: raw-row field holding the case id ("id"; missing ids fall
        back to "line<N>").
      - workflow: case workflow name (defaults to the schema name).
      - state_fields: raw-row fields copied into the state. Defaults to every
        field except ``id_field``, ``label_field``, and the case-reserved
        keys; a raw row that already has a ``state`` field uses it as-is.
      - token_budget: overrides merged over ``DEFAULT_TOKEN_BUDGET``. An
        empty ``tokenizer_id`` selects the chars-per-token heuristic.
      - keep_source: keep the raw row under "source" for debugging (default
        true; stripped when the final dataset is written).
      - show_progress: whether to show a progress bar.
    """

    name = "system1_build_cases"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        schema = self.config.get("schema")
        if not isinstance(schema, dict):
            schema_path = self.config.get("schema_path")
            if not isinstance(schema_path, str) or not schema_path:
                raise ValueError(
                    "system1_build_cases requires a workflow schema: set "
                    "'schema' (object) or 'schema_path' (JSON/YAML file) in its config"
                )
            schema = load_config(schema_path)
        self.schema = validate_schema(schema)
        self.id_field = str(self.config.get("id_field") or "id")
        self.label_field = self.schema["label_field"]
        self.workflow = str(self.config.get("workflow") or self.schema["name"])
        state_fields = config_list(self.config, "state_fields", where=self.name)
        self.state_fields = [str(field) for field in state_fields] or None
        self.keep_source = bool(self.config.get("keep_source", True))
        self.token_budget = {
            **DEFAULT_TOKEN_BUDGET,
            **config_section(self.config, "token_budget"),
        }
        self.check_tokens = bool(self.token_budget.get("check", True))
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self._count_tokens: Optional[Callable[[str], int]] = None
        self._mask_token: Optional[str] = None

    def _ensure_counter(self) -> Tuple[Callable[[str], int], Optional[str]]:
        if self._count_tokens is None:
            self._count_tokens, self._mask_token = load_token_counter(self.token_budget)
        return self._count_tokens, self._mask_token

    def _build_gold(
        self, label: Any, questions: Dict[str, Dict[str, Any]], context: str
    ) -> Dict[str, Dict[str, float]]:
        """Build the gold distributions for a raw row's label.

        Hierarchical schemas also insert the matching group's fine question
        into *questions*: the coarse gold is a one-hot over group names and
        the fine gold a one-hot over the group's options, mirroring how the
        distill variant elicits only the selected group.
        """
        if label is None or label == "":
            return {}
        question_id = self.schema["question"]
        groups = self.schema.get("groups")
        if groups is None:
            return {question_id: gold_from_label(label, questions[question_id])}
        option_label = str(label)
        matching = next(
            (name for name, group in groups.items() if option_label in group["options"]),
            None,
        )
        if matching is None:
            valid = sorted(key for group in groups.values() for key in group["options"])
            raise ValueError(
                f"{context}: label {label!r} is not an option in any group; "
                f"valid labels: {valid}"
            )
        fine_id = fine_question_id(question_id, matching)
        fine_question = build_fine_question(self.schema, matching)
        questions[fine_id] = fine_question
        return {
            question_id: {name: (1.0 if name == matching else 0.0) for name in groups},
            fine_id: gold_from_label(option_label, fine_question),
        }

    def _build_raw_case(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        case_id = row.get(self.id_field)
        case_id = str(case_id) if case_id is not None else f"line{idx + 1}"
        context = f"case {case_id!r}"
        state: Any
        if self.state_fields is not None:
            state = {}
            for field in self.state_fields:
                if field not in row:
                    raise ValueError(
                        f"{context}: state field {field!r} is missing from the raw row"
                    )
                state[field] = row[field]
        elif "state" in row:
            state = row["state"]
        else:
            state = {
                key: row[key]
                for key in row
                if key not in (self.id_field, self.label_field)
                and key not in _RESERVED_ROW_KEYS
            }
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError(
                f"{context}: raw row resolves to an empty or scalar state "
                f"({type(state).__name__}); give the row data fields or set 'state_fields'"
            )
        questions = build_schema_questions(self.schema)
        gold = self._build_gold(row.get(self.label_field), questions, context)
        case: Dict[str, Any] = {
            "id": case_id,
            "workflow": self.workflow,
            "state": state,
            "questions": questions,
            "gold": gold,
        }
        if self.keep_source:
            case["source"] = dict(row)
        return case

    def _pass_through_case(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        parsed = validate_case_row(row)
        case_id = row.get("id")
        workflow = row.get("workflow")
        case: Dict[str, Any] = {
            "id": str(case_id) if case_id is not None else f"line{idx + 1}",
            "workflow": str(workflow) if workflow else self.workflow,
            "state": parsed["state"],
            "questions": parsed["questions"],
            "gold": parsed["gold"],
        }
        if self.keep_source:
            source = row.get("source")
            if not isinstance(source, dict):
                source = {
                    key: value for key, value in row.items() if key not in _CASE_KEYS
                }
            case["source"] = source
        return case

    def _build_row(self, row: Any, idx: int) -> Dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError(
                f"system1_build_cases expected a JSON object row, got "
                f"{type(row).__name__} at line {idx + 1}"
            )
        if is_prebuilt_case(row):
            case = self._pass_through_case(row, idx)
        else:
            case = self._build_raw_case(row, idx)
        if self.check_tokens:
            count_tokens, mask_token = self._ensure_counter()
            state_text = serialize_state(case["state"])
            for question_id, question in case["questions"].items():
                check_question_budget(
                    question,
                    state_text,
                    self.token_budget,
                    count_tokens,
                    mask_token,
                    f"case {case['id']!r} question {question_id!r}",
                )
        return case

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n_prebuilt = sum(1 for row in data if isinstance(row, dict) and is_prebuilt_case(row))
        if n_prebuilt:
            logger.info(
                "Passing through %d prebuilt cases; building %d from raw rows.",
                n_prebuilt,
                len(data) - n_prebuilt,
            )
        return [
            self._build_row(row, idx)
            for idx, row in enumerate(
                progress(
                    data,
                    enabled=self.show_progress,
                    total=len(data),
                    desc="Building system1 cases",
                )
            )
        ]
