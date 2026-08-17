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

"""Generate multiple candidate responses per prompt for preference data."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.base import Operator
from easydistill.operators.generation import TextGenerationOperator

logger = logging.getLogger(__name__)


class CandidateGenerationOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Generate N candidate responses for each seed prompt.

    Rows whose `instruction_key` field is missing or empty are dropped, because
    they cannot form preference pairs.

    Configurable fields:
      - n: number of candidate responses per prompt (default 2).
      - system_prompt: optional default system prompt.
      - model_id: model identifier passed to the backend.
      - temperature: sampling temperature.
      - max_tokens: max tokens per response.
      - show_progress: whether to show tqdm progress bar.
      - max_workers: number of concurrent workers (default 1).
      - raise_on_error: if True, raise on first generation failure.
      - instruction_key: key in input rows containing the prompt (default "instruction").
      - system_key: key in input rows containing an optional system prompt.
    """

    name = "candidate_generation"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend = backend
        self.n = int(self.config.get("n") or 2)
        self.instruction_key = self.config.get("instruction_key") or "instruction"
        self.system_key = self.config.get("system_key") or "system"
        self._generator = TextGenerationOperator(backend=backend, config=config)

    def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        requests = []
        row_indices = []
        valid_rows = []
        for row_idx, row in enumerate(rows):
            instruction = row.get(self.instruction_key)
            if not instruction:
                # Skip rows without a prompt entirely; they cannot produce
                # preference pairs.
                continue
            valid_rows.append((row_idx, row))
            system_prompt = row.get(self.system_key) or self.config.get("system_prompt")
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {self.instruction_key, self.system_key, "id"}
            }
            for candidate_idx in range(self.n):
                requests.append(
                    GenerationRequest(
                        id=f"{row.get('id', row_idx)}_c{candidate_idx}",
                        instruction=(
                            instruction if isinstance(instruction, list) else str(instruction)
                        ),
                        system_prompt=system_prompt,
                        metadata={
                            "row_idx": row_idx,
                            "candidate_idx": candidate_idx,
                            **metadata,
                        },
                    )
                )
                row_indices.append(row_idx)

        results = self._generator.run(requests)
        grouped: Dict[int, List[GenerationResult]] = {idx: [] for idx in range(len(rows))}
        for result in results:
            result_row_idx = result.request.metadata.get("row_idx")
            if result_row_idx is None:
                continue
            grouped.setdefault(int(result_row_idx), []).append(result)

        output_rows = []
        for row_idx, row in valid_rows:
            candidates = grouped.get(row_idx, [])
            if len(candidates) < self.n:
                logger.warning(
                    "Row %s generated only %d/%d candidates.",
                    row.get("id", row_idx),
                    len(candidates),
                    self.n,
                )
            new_row = dict(row)
            new_row["candidates"] = [c.response for c in candidates]
            new_row["candidate_results"] = [c.model_dump() for c in candidates]
            output_rows.append(new_row)

        logger.info(
            "Generated candidates for %d rows (%d candidates each).",
            len(output_rows),
            self.n,
        )
        return output_rows
