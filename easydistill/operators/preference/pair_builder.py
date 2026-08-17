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

"""Build chosen/rejected preference pairs from scored candidates."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.operators.base import Operator

logger = logging.getLogger(__name__)


class PreferencePairBuilder(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Build chosen/rejected preference pairs from scored candidates.

    For each prompt the candidates are ranked by score. The highest-scoring
    valid candidate becomes `chosen`; the lowest-scoring candidate becomes
    `rejected`. If `min_margin` is 0.0 (the default) pairs with equal scores
    are allowed, which is useful for small example datasets but can be
    tightened in production.

    Configurable fields:
      - min_margin: minimum score gap between chosen and rejected (default 0.0).
      - max_pairs_per_prompt: number of pairs to emit per prompt (default 1).
      - require_chosen_correct: require chosen to match the reference.
      - instruction_key: prompt field (default "instruction").
      - answer_key: optional reference answer field (default "answer").
      - system_key: optional system prompt field (default "system").
    """

    name = "preference_pair_builder"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_margin = float(self.config.get("min_margin") or 0.0)
        self.max_pairs_per_prompt = int(self.config.get("max_pairs_per_prompt") or 1)
        self.require_chosen_correct = bool(self.config.get("require_chosen_correct") or False)
        self.instruction_key = self.config.get("instruction_key") or "instruction"
        self.answer_key = self.config.get("answer_key") or "answer"
        self.system_key = self.config.get("system_key") or "system"

    def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        output_rows = []
        for row in rows:
            candidates = row.get("candidates", [])
            scores = row.get("candidate_scores", [])
            if len(candidates) < 2 or len(scores) != len(candidates):
                continue

            indexed = list(zip(range(len(candidates)), candidates, scores))
            indexed.sort(key=lambda x: x[2], reverse=True)

            correct_mask = row.get("candidate_correctness", [])
            if self.require_chosen_correct and correct_mask:
                chosen_pool = [
                    item
                    for item in indexed
                    if item[0] < len(correct_mask) and correct_mask[item[0]]
                ]
                if not chosen_pool:
                    continue
            else:
                chosen_pool = indexed

            # Rejected is always the lowest-scoring candidate overall.
            rejected_candidates = list(indexed)

            chosen_list = chosen_pool[: self.max_pairs_per_prompt]

            for chosen in chosen_list:
                rejected = None
                for candidate in reversed(rejected_candidates):
                    if candidate[0] != chosen[0]:
                        rejected = candidate
                        break
                if rejected is None:
                    logger.warning(
                        "Skipping prompt %s: no rejected candidate different from chosen.",
                        row.get("id"),
                    )
                    continue
                if chosen[2] - rejected[2] < self.min_margin:
                    logger.warning(
                        "Skipping pair for prompt %s: score margin %.3f < %.3f.",
                        row.get("id"),
                        chosen[2] - rejected[2],
                        self.min_margin,
                    )
                    continue

                new_row = {
                    "id": row.get("id"),
                    self.instruction_key: row.get(self.instruction_key),
                    self.system_key: row.get(self.system_key),
                    "chosen": chosen[1],
                    "rejected": rejected[1],
                    "chosen_score": chosen[2],
                    "rejected_score": rejected[2],
                    self.answer_key: row.get(self.answer_key),
                }
                # Carry over extra metadata except internal fields.
                for key, value in row.items():
                    if key not in new_row and key not in {
                        "candidates",
                        "candidate_scores",
                        "candidate_correctness",
                        "candidate_results",
                    }:
                        new_row[key] = value
                output_rows.append(new_row)

        logger.info("Built %d preference pairs from %d prompts.", len(output_rows), len(rows))
        return output_rows
