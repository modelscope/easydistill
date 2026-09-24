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

"""Assemble RLCD training-ready case JSONL from aggregated teacher labels.

Stage 4 of the system-1 pipeline. It converts the aggregate stage's output
(or, for the ``labeled`` variant, the build stage's gold-only output) into
the training format the official Laya notebook consumes -- ``{id, workflow,
state, questions, gold}`` rows where every ``gold`` entry uses the official
nested shape ``{"probabilities": {...}, "label": key}``:

- ``distill`` / ``mixed``: each question's target is the aggregate stage's
  ``teacher`` distribution (already blended with gold under ``mixed``);
- ``labeled``: each target is the build stage's gold passed through
  unchanged (this variant skips elicit/aggregate by design).

The official ``build_training_item`` silently drops or degrades malformed
rows, so this stage enforces the notebook's hard invariants itself: every
``probabilities`` mapping carries exactly the question's option keys
(criteria keys for choice, ``"0".."n-1"`` for score, ``"false"/"true"`` for
noul), holds non-negative values with a positive sum, and ``label`` is the
argmax of the distribution. Cases where any question lacks a usable target
(not-ok teacher entries, unusable gold) are dropped whole and counted,
never emitted half-labeled.

Run statistics -- case and question counts, drop breakdown, target success
rate, teacher consistency spread, and label agreement with gold when the
rows carry it -- are logged; the operator writes no files itself (the
pipeline or a standalone runner saves the JSONL).
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from easydistill.operators.base import Operator
from easydistill.utils import progress

from .aggregate import parse_gold_distribution
from .protocol import (
    argmax_key,
    question_keys,
    validate_case_row,
)

logger = logging.getLogger(__name__)

BUILD_DATASET_VARIANTS = ("labeled", "distill", "mixed")


# ---------------- distribution validation ----------------


def validated_distribution(
    probabilities: Any, keys: List[str]
) -> Optional[Dict[str, float]]:
    """Validate a target distribution over ``keys``, or return None.

    Enforces the official notebook's hard invariants: the mapping carries
    exactly ``keys`` (no extras, none missing), every value is a finite
    non-negative number, and the values sum above zero. Valid distributions
    are returned unchanged (no renormalization) in ``keys`` order.
    """
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        return None
    values: List[float] = []
    for key in keys:
        value = probabilities[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number < 0 or not math.isfinite(number):
            return None
        values.append(number)
    if sum(values) <= 0:
        return None
    return dict(zip(keys, values))


class System1BuildDatasetOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Emit training-ready system-1 case rows with final gold targets.

    Reads build-stage rows (``labeled``) or aggregate-stage rows
    (``distill`` / ``mixed``) and returns ``{id, workflow, state,
    questions, gold}`` rows in the official RLCD format (see the module
    docstring for the invariants). Intermediate fields (``teacher``,
    ``source``) are consumed, not carried into the final dataset.

    Configurable fields:
      - variant: ``labeled`` takes targets from the build stage's gold;
        ``distill`` (default) and ``mixed`` take them from the aggregate
        stage's ``teacher`` payload (already blended under ``mixed``).
      - id_key: case field holding the row id ("id").
      - show_progress: whether to show a progress bar.
    """

    name = "system1_build_dataset"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        variant = self.config.get("variant", "distill")
        if variant not in BUILD_DATASET_VARIANTS:
            raise ValueError(
                f"variant must be one of {BUILD_DATASET_VARIANTS}, got {variant!r}"
            )
        self.variant = str(variant)
        self.id_key = str(self.config.get("id_key") or "id")
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = list(input_data)
        if not rows:
            return []
        self._validate_inputs(rows)
        stats: Dict[str, Any] = {
            "cases_in": len(rows),
            "cases_out": 0,
            "questions": 0,
            "questions_ok": 0,
            "consistency_sum": 0.0,
            "consistency_n": 0,
            "consistency_min": None,
            "consistency_max": None,
            "gold_n": 0,
            "gold_agree": 0,
            "drops": Counter(),
        }
        outputs: List[Dict[str, Any]] = []
        for idx, row in enumerate(
            progress(rows, enabled=self.show_progress, desc="Building system1 dataset")
        ):
            case = self._build_case(row, idx, stats)
            if case is not None:
                outputs.append(case)
        self._log_stats(stats)
        return outputs

    def _validate_inputs(self, rows: List[Dict[str, Any]]) -> None:
        """Check the input shape matches the configured variant."""
        if self.variant == "labeled":
            offenders = [
                str(row.get(self.id_key) or f"line{idx + 1}")
                for idx, row in enumerate(rows)
                if isinstance(row.get("teacher"), dict)
            ]
            if offenders:
                raise ValueError(
                    "variant 'labeled' skips elicit/aggregate by design, but input "
                    f"rows carry a 'teacher' payload (first: {offenders[0]!r}); feed "
                    "the system1_build_cases output or switch variant to "
                    "'distill'/'mixed'"
                )
            return
        for idx, row in enumerate(rows):
            if not isinstance(row.get("teacher"), dict):
                row_id = str(row.get(self.id_key) or f"line{idx + 1}")
                raise ValueError(
                    f"case {row_id!r}: variant {self.variant!r} requires aggregate "
                    "output carrying a 'teacher' payload; run the system1_elicit "
                    "and system1_aggregate stages first (variant 'labeled' skips "
                    "elicit/aggregate by design)"
                )

    def _build_case(
        self, row: Dict[str, Any], idx: int, stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build one output case, or None when the case is dropped."""
        row_id = str(row.get(self.id_key) or f"line{idx + 1}")
        parsed = validate_case_row(row)
        stats["questions"] += len(parsed["questions"])
        teacher = row.get("teacher") if self.variant != "labeled" else None
        final_gold: Dict[str, Any] = {}
        for question_id, question in parsed["questions"].items():
            keys = question_keys(question)
            dist: Optional[Dict[str, float]]
            if teacher is not None:
                entry = teacher.get(question_id)
                if not isinstance(entry, dict):
                    stats["drops"]["missing_teacher_entry"] += 1
                    return None
                if not entry.get("ok"):
                    stats["drops"]["teacher_not_ok"] += 1
                    return None
                dist = validated_distribution(entry.get("probabilities"), keys)
                if dist is None:
                    stats["drops"]["invalid_probabilities"] += 1
                    return None
                self._record_consistency(stats, entry.get("consistency"))
            else:
                dist = parse_gold_distribution(parsed["gold"].get(question_id), keys)
                if dist is None:
                    stats["drops"]["gold_unusable"] += 1
                    return None
            label = argmax_key(dist)
            final_gold[question_id] = {"probabilities": dist, "label": label}
            stats["questions_ok"] += 1
            if teacher is not None:
                gold_dist = parse_gold_distribution(parsed["gold"].get(question_id), keys)
                if gold_dist is not None:
                    stats["gold_n"] += 1
                    if argmax_key(gold_dist) == label:
                        stats["gold_agree"] += 1
        stats["cases_out"] += 1
        return {
            "id": row_id,
            "workflow": row.get("workflow"),
            "state": parsed["state"],
            "questions": parsed["questions"],
            "gold": final_gold,
        }

    # ---------------- stats ----------------

    @staticmethod
    def _record_consistency(stats: Dict[str, Any], value: Any) -> None:
        """Fold one teacher consistency reading into the stats."""
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            reading = float(value)
            stats["consistency_sum"] += reading
            stats["consistency_n"] += 1
            if stats["consistency_min"] is None or reading < stats["consistency_min"]:
                stats["consistency_min"] = reading
            if stats["consistency_max"] is None or reading > stats["consistency_max"]:
                stats["consistency_max"] = reading

    def _log_stats(self, stats: Dict[str, Any]) -> None:
        dropped = stats["cases_in"] - stats["cases_out"]
        logger.info(
            "System1 build_dataset (%s): %d cases in, %d out, %d dropped.",
            self.variant,
            stats["cases_in"],
            stats["cases_out"],
            dropped,
        )
        if stats["drops"]:
            breakdown = ", ".join(
                f"{count} {reason}" for reason, count in stats["drops"].most_common()
            )
            logger.info("Dropped cases by reason: %s.", breakdown)
        if stats["questions"]:
            logger.info(
                "Question targets: %d/%d usable (%.1f%%).",
                stats["questions_ok"],
                stats["questions"],
                100.0 * stats["questions_ok"] / stats["questions"],
            )
        if stats["consistency_n"]:
            logger.info(
                "Teacher consistency: mean %.3f, min %.3f, max %.3f over %d questions.",
                stats["consistency_sum"] / stats["consistency_n"],
                float(stats["consistency_min"]),
                float(stats["consistency_max"]),
                stats["consistency_n"],
            )
        if stats["gold_n"]:
            logger.info(
                "Label agreement with gold: %d/%d (%.1f%%) over questions carrying gold.",
                stats["gold_agree"],
                stats["gold_n"],
                100.0 * stats["gold_agree"] / stats["gold_n"],
            )
