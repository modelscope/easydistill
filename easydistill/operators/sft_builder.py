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

"""SFT dataset builder: generation results -> sharegpt-format SFT data."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationResult, SFTSample

from .base import Operator

logger = logging.getLogger(__name__)


class SFTDatasetBuilder(Operator[List[GenerationResult], List[SFTSample]]):
    """Convert generation results into SFT training samples.

    Configurable fields:
      - system_prompt: default system prompt if not present in result.
      - skip_empty: skip responses that are empty after stripping.
      - min_length: minimum response length (in characters).
      - max_length: maximum response length (in characters); longer responses are skipped.
      - dedup_key: optional key(s) used to drop duplicate samples. Pass a string
        (e.g. "instruction", "response", or "instruction_response") or a list
        of field names. Defaults to None (no deduplication).
    """

    name = "sft_dataset_builder"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.system_prompt = self.config.get("system_prompt")
        skip_empty = self.config.get("skip_empty")
        self.skip_empty = bool(skip_empty) if skip_empty is not None else True
        self.min_length = int(self.config.get("min_length") or 0)
        max_length = self.config.get("max_length")
        self.max_length = int(max_length) if max_length is not None else None
        dedup_key = self.config.get("dedup_key")
        if isinstance(dedup_key, str):
            dedup_key = dedup_key.split("_") if "_" in dedup_key else [dedup_key]
        self.dedup_key = dedup_key

    def _is_valid(self, result: GenerationResult) -> bool:
        response = (result.response or "").strip()
        if self.skip_empty and not response:
            return False
        if len(response) < self.min_length:
            return False
        return not (self.max_length is not None and len(response) > self.max_length)

    def _dedup_key(self, result: GenerationResult) -> Optional[str]:
        """Return a hashable deduplication key for ``result`` or None if disabled."""
        if not self.dedup_key:
            return None
        parts = []
        for field in self.dedup_key:
            if field == "instruction":
                parts.append(str(result.request.instruction))
            elif field == "response":
                parts.append(str(result.response))
            else:
                parts.append(str(result.metadata.get(field, "")))
        return "|".join(parts)

    def run(self, results: List[GenerationResult]) -> List[SFTSample]:
        samples = []
        seen: set[str] = set()
        duplicates = 0
        for result in results:
            if not self._is_valid(result):
                logger.info("Skipping invalid result for instruction: %s", result.request.id)
                continue
            system = result.request.system_prompt or self.system_prompt
            metadata = {
                "source": "teacher_model",
                "model": result.model,
                "request_id": result.request.id,
                **result.metadata,
            }
            if result.usage:
                metadata["usage"] = result.usage
            sample = SFTSample.from_instruction_response(
                instruction=result.request.instruction,
                response=result.response,
                system=system,
                metadata=metadata,
            )
            key = self._dedup_key(result)
            if key is not None:
                if key in seen:
                    duplicates += 1
                    logger.info(
                        "Skipping duplicate SFT sample for instruction: %s", result.request.id
                    )
                    continue
                seen.add(key)
            samples.append(sample)
        if duplicates:
            logger.info("Removed %d duplicate SFT samples.", duplicates)
        logger.info("Built %d SFT samples from %d generation results.", len(samples), len(results))
        return samples
