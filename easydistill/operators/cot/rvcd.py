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

"""RV / CD scorer and mixer for CoT reasoning data.

RV (Reasoning Verbosity) and CD (Cognitive Difficulty) are LLM-as-judge scores
used by the OmniThought pipeline to describe how appropriate a CoT's length is
for a given problem and how cognitively demanding the chain is to follow. This
module provides:

  - CoTRVCDScorer: score existing CoT rows on RV, CD, and logical correctness.
  - CoTRVCDMixer: mix scored rows into an SFT subset.
"""

import bisect
import logging
from typing import Any, Dict, List, Optional

from easydistill.utils.image import _extract_text_from_content

logger = logging.getLogger(__name__)

DEFAULT_CD_BINS = [0.0, 3.0, 6.0, 10.0]
RV_TARGET_MAP = {"low": 2.0, "medium": 5.0, "high": 8.0}


class CoTRVCDScorer:
    """Score CoT rows on reasoning verbosity, cognitive difficulty, and correctness.

    Configurable fields:
      - metrics: list of metrics to compute (default: rv, cd, correctness).
      - max_workers, temperature, max_tokens, show_progress: passed to CoTEvaluator.
      - instruction_key: field name for the problem/instruction (default: "instruction").
      - output_key: field name for the CoT response (default: "response").

    Input: list of dict rows, each containing at least the instruction and response.
    Output: the same rows with additional keys reasoning_verbosity,
            cognitive_difficulty, and logical_correctness.
    """

    name = "cot_rvcd_scorer"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        # Import here to avoid a circular import: easydistill.eval.base imports
        # operators.generation, and operators/__init__.py imports operators.cot.
        from easydistill.eval import CoTEvaluator

        self.config = config or {}
        self.instruction_key = self.config.get("instruction_key", "instruction")
        self.output_key = self.config.get("output_key", "response")
        # RV/CD scoring always needs the three core metrics; any user-supplied
        # metrics are appended rather than replacing them. Other evaluator
        # settings (e.g. prompts_file, prompts) are preserved from the config.
        required_metrics = [
            "reasoning_verbosity",
            "cognitive_difficulty",
            "logical_correctness",
        ]
        metrics = list(self.config.get("metrics", required_metrics))
        for metric in required_metrics:
            if metric not in metrics:
                metrics.append(metric)
        evaluator_cfg = {
            **self.config,
            "metrics": metrics,
            "max_workers": self.config.get("max_workers", 10),
            "temperature": self.config.get("temperature", 0.0),
            "max_tokens": self.config.get("max_tokens", 512),
            "show_progress": self.config.get("show_progress", True),
        }
        self.evaluator = CoTEvaluator(backend=backend, config=evaluator_cfg)

    def _to_eval_sample(self, row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        instruction = row.get(self.instruction_key)
        output = row.get(self.output_key)
        if not instruction or not output:
            return None
        return {
            "id": str(row.get("id", idx)),
            "instruction": instruction,
            "output": output,
        }

    def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score rows and merge scores back into the original rows."""
        if not rows:
            return []

        eval_samples: List[Dict[str, Any]] = []
        idx_to_row_idx: Dict[int, int] = {}
        for row_idx, row in enumerate(rows):
            sample = self._to_eval_sample(row, row_idx)
            if sample is None:
                continue
            idx_to_row_idx[len(eval_samples)] = row_idx
            eval_samples.append(sample)

        if not eval_samples:
            logger.warning("No valid samples found for RV/CD scoring.")
            return rows

        scored = self.evaluator.run(eval_samples)
        scored_by_id = {s["id"]: s for s in scored}

        output_rows = []
        for row_idx, row in enumerate(rows):
            new_row = dict(row)
            sample_id = str(row.get("id", row_idx))
            scores = scored_by_id.get(sample_id, {})
            for metric in self.evaluator.metrics:
                if metric in scores:
                    new_row[metric] = scores[metric]
            output_rows.append(new_row)

        logger.info("Scored %d CoT rows on RV/CD/correctness.", len(eval_samples))
        return output_rows


class CoTRVCDMixer:
    """Mix scored CoT rows by RV/CD bins for SFT selection.

    Configurable fields:
      - mode: "sft".
      - cd_bins: bin edges for cognitive difficulty (default [0, 3, 6, 10]).
      - rv_target: "matched", "low", "medium", "high", or a numeric value.
                   "matched" maps each CD bin to an increasing RV target.
      - samples_per_bin: max rows to keep per CD bin.
      - min_correctness: minimum logical_correctness score to include (default 1).
      - instruction_key, output_key: field names for instruction/response.

    Input: rows already scored by CoTRVCDScorer.
    Output: rows whose RV is closest to the target within each CD bin.
    """

    name = "cot_rvcd_mixer"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "sft")
        if self.mode != "sft":
            raise ValueError(f"Unsupported RV/CD mixer mode: {self.mode}; only 'sft' is supported")
        self.cd_bins = list(self.config.get("cd_bins", DEFAULT_CD_BINS))
        if len(self.cd_bins) < 2:
            raise ValueError("cd_bins must contain at least two edges.")
        self.rv_target = self.config.get("rv_target", "matched")
        self.samples_per_bin = int(self.config.get("samples_per_bin") or 0) or None
        self.min_correctness = self.config.get("min_correctness", 1)
        self.instruction_key = self.config.get("instruction_key", "instruction")
        self.output_key = self.config.get("output_key", "response")

    def _bin_for_cd(self, cd_score: Optional[Any]) -> Optional[int]:
        if cd_score is None:
            return None
        try:
            value = float(cd_score)
        except (TypeError, ValueError):
            return None
        if value < self.cd_bins[0] or value > self.cd_bins[-1]:
            return None
        # bisect_right returns the index where value should be inserted to keep
        # sorted order; subtract 1 to get the bin index.
        return bisect.bisect_right(self.cd_bins, value) - 1

    def _target_rv_for_bin(self, bin_idx: int) -> float:
        if isinstance(self.rv_target, (int, float)):
            return float(self.rv_target)
        if self.rv_target in RV_TARGET_MAP:
            return RV_TARGET_MAP[self.rv_target]
        if self.rv_target == "matched":
            # Linearly map bin index to a mid-range RV target.
            n_bins = len(self.cd_bins) - 1
            if n_bins <= 1:
                return 5.0
            step = 8.0 / max(1, n_bins - 1)
            return round(2.0 + bin_idx * step, 2)
        raise ValueError(f"Unsupported rv_target: {self.rv_target}")

    def _rv_distance(self, row: Dict[str, Any], target: float) -> float:
        rv = row.get("reasoning_verbosity")
        if rv is None:
            return float("inf")
        return abs(float(rv) - target)

    def _is_valid(self, row: Dict[str, Any]) -> bool:
        correctness = row.get("logical_correctness")
        if correctness is None:
            return False
        try:
            return int(correctness) >= int(self.min_correctness)
        except (TypeError, ValueError):
            return False

    def _extract_text(self, row: Dict[str, Any], key: str) -> str:
        return _extract_text_from_content(row.get(key) or "")

    def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mix rows into SFT selection."""
        if not rows:
            return []

        # Group rows by CD bin, keeping only valid (correct enough) rows.
        bins: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            bin_idx = self._bin_for_cd(row.get("cognitive_difficulty"))
            if bin_idx is None:
                continue
            if not self._is_valid(row):
                continue
            bins.setdefault(bin_idx, []).append(row)

        return self._mix_sft(bins)

    def _mix_sft(self, bins: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for bin_idx in sorted(bins):
            target_rv = self._target_rv_for_bin(bin_idx)
            candidates = sorted(bins[bin_idx], key=lambda r: self._rv_distance(r, target_rv))
            if self.samples_per_bin:
                candidates = candidates[: self.samples_per_bin]
            for row in candidates:
                new_row = dict(row)
                new_row["cd_bin"] = bin_idx
                new_row["rv_target"] = target_rv
                selected.append(new_row)
        logger.info(
            "Selected %d rows across %d CD bins for SFT mixing.",
            len(selected),
            len(bins),
        )
        return selected
