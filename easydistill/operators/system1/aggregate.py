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

"""Aggregate elicited teacher samples into per-question system-1 labels.

Stage 3 of the system-1 pipeline. It turns the elicit stage's raw
``elicitations`` payload into one compact ``teacher`` label per question and
writes no files itself (the pipeline or a standalone runner saves the labels
JSONL):

- verbalized methods (``single_verbalized``, ``k_verbalized_mean``,
  ``two_stage``): every usable sample's distribution is normalized, the
  samples are averaged, and the mean is shrunk away from one-hots with
  ``p' = (p + eps / K) / (1 + eps)`` (K = option count, eps = ``shrinkage``,
  default 0.02);
- ``k_sample_freq``: picks are counted into a distribution with Laplace
  smoothing (pseudo-count 0.5 per option), then shrunk the same way.

Questions whose top-1 agreement rate across usable samples falls below
``consistency_threshold`` (default 0.6) are filtered: the entry is marked
``ok: false`` with reason ``low_consistency`` and counted, never silently
dropped. When a question carries gold (the build stage's flat one-hot shape
or the official ``{"probabilities": ..., "label": ...}`` shape), the
pre-blend teacher distribution is checked against it -- total variation and
KL(gold || teacher), floored at 1e-12 -- as distillation quality metrics.
With ``variant: mixed`` the emitted label becomes
``alpha * gold + (1 - alpha) * teacher`` (default alpha 0.7); questions
without usable gold fall back to the pure teacher distribution.

Output rows keep every case field but replace the bulky ``elicitations``
payload with ``teacher: {qid: entry}``. Each entry carries ``ok``, ``method``,
``model``, ``probabilities``, ``label``, ``consistency``, ``samples_used``,
``samples_failed``, ``reason``, plus ``gold_tv`` / ``gold_kl`` when gold was
usable and ``blended`` under the ``mixed`` variant. The elicit stage's own
output file retains the raw per-sample records.
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from easydistill.operators.base import Operator
from easydistill.utils import progress

from .protocol import (
    argmax_key,
    normalize_probabilities,
    question_keys,
    validate_case_row,
)

logger = logging.getLogger(__name__)

AGGREGATE_VARIANTS = ("distill", "mixed")
FREQ_LAPLACE_ALPHA = 0.5
DEFAULT_CONSISTENCY_THRESHOLD = 0.6
DEFAULT_MIX_ALPHA = 0.7
DEFAULT_SHRINKAGE = 0.02
_KL_FLOOR = 1e-12


# ---------------- gold and distribution helpers ----------------


def parse_gold_distribution(entry: Any, keys: List[str]) -> Optional[Dict[str, float]]:
    """Parse a gold answer into a distribution over ``keys``, or None.

    Accepts the build stage's flat one-hot shape (``{key: 1.0}``) and the
    official nested ``{"probabilities": {...}, "label": ...}`` shape. Keys
    missing from a nested mapping default to 0; unknown keys, non-numeric
    values, or a non-positive total make the gold unusable.
    """
    if not isinstance(entry, dict):
        return None
    if isinstance(entry.get("probabilities"), dict):
        entry = entry["probabilities"]
    if any(key not in keys for key in entry):
        return None
    normalized = normalize_probabilities([entry.get(key, 0.0) for key in keys])
    if normalized is None:
        return None
    return dict(zip(keys, normalized))


def shrink_distribution(dist: Dict[str, float], epsilon: float) -> Dict[str, float]:
    """Blend a distribution toward uniform: ``(p + eps / K) / (1 + eps)``."""
    if epsilon <= 0:
        return dict(dist)
    count = len(dist)
    return {
        key: (value + epsilon / count) / (1.0 + epsilon)
        for key, value in dist.items()
    }


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total variation distance ``0.5 * sum |p - q|`` over ``p``'s keys."""
    return 0.5 * sum(abs(value - q.get(key, 0.0)) for key, value in p.items())


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(p || q) over ``p``'s keys, floored at 1e-12 to stay finite.

    Only terms with ``p > 0`` contribute, so a one-hot gold stays finite
    against any shrunk teacher distribution.
    """
    total = 0.0
    for key, p_value in p.items():
        if p_value <= 0:
            continue
        q_value = max(q.get(key, 0.0), _KL_FLOOR)
        total += p_value * math.log(max(p_value, _KL_FLOOR) / q_value)
    return total


# ---------------- per-question aggregation ----------------


def aggregate_question(
    question: Dict[str, Any],
    payload: Optional[Dict[str, Any]],
    gold_dist: Optional[Dict[str, float]],
    *,
    variant: str,
    alpha: float,
    consistency_threshold: float,
    shrinkage: float,
) -> Dict[str, Any]:
    """Aggregate one question's elicitation payload into a teacher entry.

    ``payload`` is the elicit stage's ``elicitations[qid]`` object; None means
    the question was never elicited. ``gold_tv`` / ``gold_kl`` are computed
    on the post-shrinkage, pre-blend teacher distribution whenever gold is
    usable, even for entries the consistency filter later rejects.
    """
    keys = question_keys(question)
    entry: Dict[str, Any] = {
        "ok": False,
        "method": None,
        "model": None,
        "probabilities": None,
        "label": None,
        "consistency": None,
        "samples_used": 0,
        "samples_failed": 0,
        "reason": "not_elicited",
    }
    if not isinstance(payload, dict):
        return entry
    entry["method"] = payload.get("method")
    entry["model"] = payload.get("model")
    samples = payload.get("samples") or []
    if not samples:
        entry["reason"] = "no_usable_samples"
        return entry
    distributions: List[Dict[str, float]] = []
    picks: List[str] = []
    for record in samples:
        if not isinstance(record, dict) or not record.get("ok"):
            continue
        if entry["method"] == "k_sample_freq":
            pick = record.get("pick")
            if isinstance(pick, str) and pick in keys:
                picks.append(pick)
            continue
        probabilities = record.get("probabilities")
        if not isinstance(probabilities, dict):
            continue
        normalized = normalize_probabilities([probabilities.get(key) for key in keys])
        if normalized is not None:
            distributions.append(dict(zip(keys, normalized)))
    top_ones: List[str]
    if picks:
        counts = Counter(picks)
        teacher: Dict[str, float] = {
            key: (counts.get(key, 0) + FREQ_LAPLACE_ALPHA)
            / (len(picks) + FREQ_LAPLACE_ALPHA * len(keys))
            for key in keys
        }
        top_ones = picks
    elif distributions:
        teacher = {
            key: sum(dist[key] for dist in distributions) / len(distributions)
            for key in keys
        }
        top_ones = [argmax_key(dist) for dist in distributions]
    else:
        entry["reason"] = "no_usable_samples"
        entry["samples_failed"] = len(samples)
        return entry
    entry["samples_used"] = len(top_ones)
    entry["samples_failed"] = len(samples) - len(top_ones)
    consistency = max(Counter(top_ones).values()) / len(top_ones)
    entry["consistency"] = consistency
    teacher = shrink_distribution(teacher, shrinkage)
    if gold_dist is not None:
        entry["gold_tv"] = total_variation(gold_dist, teacher)
        entry["gold_kl"] = kl_divergence(gold_dist, teacher)
    if consistency_threshold > 0 and consistency < consistency_threshold:
        entry["reason"] = "low_consistency"
        return entry
    final = teacher
    if variant == "mixed":
        if gold_dist is not None:
            final = {
                key: alpha * gold_dist[key] + (1.0 - alpha) * teacher[key]
                for key in keys
            }
            entry["blended"] = True
        else:
            entry["blended"] = False
    entry["ok"] = True
    entry["reason"] = None
    entry["probabilities"] = final
    entry["label"] = argmax_key(final)
    return entry


class System1AggregateOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Replace elicitation payloads with aggregated per-question labels.

    Reads elicit-stage rows and returns the same rows with the bulky
    ``elicitations`` payload replaced by a compact ``teacher`` mapping (see
    the module docstring for the entry schema). The elicit stage's own
    output file keeps the raw per-sample records for auditing.

    Configurable fields:
      - variant: ``distill`` (default) keeps pure teacher labels; ``mixed``
        blends gold into the emitted label. The ``labeled`` variant is
        rejected here -- it skips elicit/aggregate by design.
      - alpha: gold weight for ``mixed`` labels (default 0.7).
      - consistency_threshold: drop questions whose top-1 agreement rate
        across usable samples falls below this (default 0.6; 0 disables the
        filter).
      - shrinkage: anti-one-hot epsilon applied to every distribution
        (default 0.02; 0 disables).
      - id_key: case field holding the row id ("id").
      - show_progress: whether to show a progress bar.
    """

    name = "system1_aggregate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        variant = self.config.get("variant", "distill")
        if variant not in AGGREGATE_VARIANTS:
            raise ValueError(
                f"variant must be one of {AGGREGATE_VARIANTS}, got {variant!r}; the "
                "'labeled' variant skips elicit/aggregate by design"
            )
        self.variant = str(variant)
        alpha = self.config.get("alpha")
        self.alpha = float(alpha) if alpha is not None else DEFAULT_MIX_ALPHA
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {self.alpha!r}")
        threshold = self.config.get("consistency_threshold")
        self.consistency_threshold = (
            float(threshold) if threshold is not None else DEFAULT_CONSISTENCY_THRESHOLD
        )
        if not 0.0 <= self.consistency_threshold <= 1.0:
            raise ValueError(
                "consistency_threshold must be within [0, 1], got "
                f"{self.consistency_threshold!r}"
            )
        shrinkage = self.config.get("shrinkage")
        self.shrinkage = float(shrinkage) if shrinkage is not None else DEFAULT_SHRINKAGE
        if self.shrinkage < 0:
            raise ValueError(f"shrinkage must be >= 0, got {self.shrinkage!r}")
        self.id_key = str(self.config.get("id_key") or "id")
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = list(input_data)
        if not rows:
            return []
        has_payload = any(
            isinstance(row.get("elicitations"), dict) and row["elicitations"] for row in rows
        )
        if not has_payload:
            raise ValueError(
                "input rows carry no 'elicitations' payload; run the system1_elicit "
                "stage first (variant 'labeled' skips elicit/aggregate by design)"
            )
        stats: Dict[str, Any] = {
            "questions": 0,
            "ok": 0,
            "not_elicited": 0,
            "no_usable_samples": 0,
            "low_consistency": 0,
            "samples_total": 0,
            "samples_used": 0,
            "consistency_sum": 0.0,
            "consistency_n": 0,
            "gold_n": 0,
            "tv_sum": 0.0,
            "kl_sum": 0.0,
            "blended": 0,
            "blend_fallback": 0,
            "leftover_payloads": 0,
        }
        outputs: List[Dict[str, Any]] = []
        for idx, row in enumerate(
            progress(rows, enabled=self.show_progress, desc="Aggregating system1 labels")
        ):
            outputs.append(self._aggregate_row(row, idx, stats))
        self._log_stats(stats)
        return outputs

    def _aggregate_row(
        self, row: Dict[str, Any], idx: int, stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate one case row; raises on malformed case fields."""
        row_id = str(row.get(self.id_key) or f"line{idx + 1}")
        parsed = validate_case_row(row)
        elicitations = row.get("elicitations")
        if elicitations is not None and not isinstance(elicitations, dict):
            raise ValueError(f"case {row_id!r}: 'elicitations' must be an object")
        teacher: Dict[str, Any] = {}
        for question_id, question in parsed["questions"].items():
            payload = (elicitations or {}).get(question_id)
            gold_dist = parse_gold_distribution(
                parsed["gold"].get(question_id), question_keys(question)
            )
            entry = aggregate_question(
                question,
                payload if isinstance(payload, dict) else None,
                gold_dist,
                variant=self.variant,
                alpha=self.alpha,
                consistency_threshold=self.consistency_threshold,
                shrinkage=self.shrinkage,
            )
            teacher[question_id] = entry
            self._record_stats(stats, entry)
        stats["questions"] += len(parsed["questions"])
        if isinstance(elicitations, dict):
            stats["leftover_payloads"] += sum(
                1 for qid in elicitations if qid not in parsed["questions"]
            )
        out = {key: value for key, value in row.items() if key != "elicitations"}
        out["questions"] = parsed["questions"]
        out["teacher"] = teacher
        return out

    # ---------------- stats ----------------

    @staticmethod
    def _record_stats(stats: Dict[str, Any], entry: Dict[str, Any]) -> None:
        """Fold one question entry into the run-level counters."""
        if entry["ok"]:
            stats["ok"] += 1
        else:
            stats[str(entry["reason"])] += 1
        stats["samples_total"] += entry["samples_used"] + entry["samples_failed"]
        stats["samples_used"] += entry["samples_used"]
        if entry["consistency"] is not None:
            stats["consistency_sum"] += float(entry["consistency"])
            stats["consistency_n"] += 1
        if "gold_tv" in entry:
            stats["gold_n"] += 1
            stats["tv_sum"] += float(entry["gold_tv"])
            stats["kl_sum"] += float(entry["gold_kl"])
        if entry.get("blended"):
            stats["blended"] += 1
        elif "blended" in entry:
            stats["blend_fallback"] += 1

    def _log_stats(self, stats: Dict[str, Any]) -> None:
        logger.info(
            "System1 aggregate (%s): %d questions, %d ok, %d low_consistency, "
            "%d no_usable_samples, %d not_elicited.",
            self.variant,
            stats["questions"],
            stats["ok"],
            stats["low_consistency"],
            stats["no_usable_samples"],
            stats["not_elicited"],
        )
        if stats["samples_total"]:
            logger.info(
                "Elicitation samples: %d total, %d usable (%.1f%%).",
                stats["samples_total"],
                stats["samples_used"],
                100.0 * stats["samples_used"] / stats["samples_total"],
            )
        if stats["consistency_n"]:
            logger.info(
                "Consistency: mean %.3f over %d aggregated questions.",
                stats["consistency_sum"] / stats["consistency_n"],
                stats["consistency_n"],
            )
        if stats["gold_n"]:
            logger.info(
                "Gold agreement: %d questions, mean TV %.3f, mean KL %.3f.",
                stats["gold_n"],
                stats["tv_sum"] / stats["gold_n"],
                stats["kl_sum"] / stats["gold_n"],
            )
        if self.variant == "mixed":
            if stats["gold_n"] == 0:
                logger.error(
                    "variant=mixed but no question carried usable gold; every "
                    "label fell back to the pure teacher distribution."
                )
            else:
                logger.info(
                    "Mixed blend: %d questions blended with gold, %d fell back "
                    "to the pure teacher distribution.",
                    stats["blended"],
                    stats["blend_fallback"],
                )
        if stats["leftover_payloads"]:
            logger.warning(
                "Ignored %d elicitation payloads without matching questions.",
                stats["leftover_payloads"],
            )
