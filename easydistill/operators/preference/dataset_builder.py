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

"""Convert preference pairs into training framework formats."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import PreferenceSample
from easydistill.operators.base import Operator
from easydistill.utils.image import _extract_text_from_content, content_has_images

logger = logging.getLogger(__name__)


class PreferenceDatasetBuilder(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Export chosen/rejected pairs to DPO / LLaMA-Factory formats.

    Each row must contain the prompt field (`instruction_key`), a `chosen`
    response, and a `rejected` response. Rows missing any of these are skipped.

    Configurable fields:
      - format: one of
          "llama_factory_alpaca" (default)
          "llama_factory_sharegpt"
          "openai_messages"
      - system_prompt: default system prompt if not present in the row.
      - instruction_key: prompt field (default "instruction").
      - system_key: system prompt field (default "system").
      - skip_empty: skip pairs with empty chosen or rejected (default True).
      - min_length: minimum response length in characters.
      - max_length: maximum response length in characters.
    """

    name = "preference_dataset_builder"
    SUPPORTED_FORMATS = {
        "llama_factory_alpaca",
        "llama_factory_sharegpt",
        "openai_messages",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.format = self.config.get("format") or "llama_factory_alpaca"
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported preference format: {self.format}")
        self.system_prompt = self.config.get("system_prompt")
        self.instruction_key = self.config.get("instruction_key") or "instruction"
        self.system_key = self.config.get("system_key") or "system"
        skip_empty = self.config.get("skip_empty")
        self.skip_empty = bool(skip_empty) if skip_empty is not None else True
        self.min_length = int(self.config.get("min_length") or 0)
        max_length = self.config.get("max_length")
        self.max_length = int(max_length) if max_length is not None else None

    def _is_valid(self, chosen: str, rejected: str) -> bool:
        chosen = (chosen or "").strip()
        rejected = (rejected or "").strip()
        if self.skip_empty and (not chosen or not rejected):
            return False
        if len(chosen) < self.min_length or len(rejected) < self.min_length:
            return False
        return not (
            self.max_length is not None
            and (len(chosen) > self.max_length or len(rejected) > self.max_length)
        )

    def _build_sample(self, row: Dict[str, Any]) -> Optional[PreferenceSample]:
        instruction = row.get(self.instruction_key)
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not instruction or not chosen or not rejected:
            return None
        if isinstance(chosen, list):
            logger.warning("Extracting text from list-valued chosen response.")
            chosen = _extract_text_from_content(chosen)
        if isinstance(rejected, list):
            logger.warning("Extracting text from list-valued rejected response.")
            rejected = _extract_text_from_content(rejected)
        if not self._is_valid(chosen, rejected):
            return None

        system = row.get(self.system_key) or self.system_prompt
        return PreferenceSample.from_instruction_responses(
            instruction=instruction if isinstance(instruction, list) else str(instruction),
            chosen=str(chosen),
            rejected=str(rejected),
            system=system,
            chosen_score=row.get("chosen_score"),
            rejected_score=row.get("rejected_score"),
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    self.instruction_key,
                    self.system_key,
                    "chosen",
                    "rejected",
                    "chosen_score",
                    "rejected_score",
                }
            },
        )

    def _to_openai_messages(self, sample: PreferenceSample) -> Dict[str, Any]:
        return {
            "prompt": [m.model_dump() for m in sample.prompt],
            "chosen": [m.model_dump() for m in sample.chosen],
            "rejected": [m.model_dump() for m in sample.rejected],
        }

    def _to_llama_factory_alpaca(self, sample: PreferenceSample) -> Dict[str, Any]:
        instruction = ""
        for m in sample.prompt:
            if m.role == "user":
                if content_has_images(m.content):
                    logger.warning(
                        "Stripping image content for llama_factory_alpaca export."
                    )
                instruction = _extract_text_from_content(m.content)
                break
        chosen = ""
        rejected = ""
        if sample.chosen:
            chosen = _extract_text_from_content(sample.chosen[0].content)
        if sample.rejected:
            rejected = _extract_text_from_content(sample.rejected[0].content)
        return {
            "instruction": instruction,
            "input": "",
            "chosen": chosen,
            "rejected": rejected,
        }

    def _to_llama_factory_sharegpt(self, sample: PreferenceSample) -> Dict[str, Any]:
        conversations = []
        for m in sample.prompt:
            if content_has_images(m.content):
                logger.warning(
                    "Stripping image content for llama_factory_sharegpt export."
                )
            role_map = {"system": "system", "user": "human", "assistant": "gpt"}
            conversations.append(
                {"from": role_map[m.role], "value": _extract_text_from_content(m.content)}
            )
        chosen = {}
        if sample.chosen:
            chosen = {
                "from": "gpt",
                "value": _extract_text_from_content(sample.chosen[0].content),
            }
        rejected = {}
        if sample.rejected:
            rejected = {
                "from": "gpt",
                "value": _extract_text_from_content(sample.rejected[0].content),
            }
        return {
            "conversations": conversations,
            "chosen": chosen,
            "rejected": rejected,
        }

    def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples = []
        for row in rows:
            sample = self._build_sample(row)
            if sample is None:
                continue
            if self.format == "openai_messages":
                samples.append(self._to_openai_messages(sample))
            elif self.format == "llama_factory_alpaca":
                samples.append(self._to_llama_factory_alpaca(sample))
            elif self.format == "llama_factory_sharegpt":
                samples.append(self._to_llama_factory_sharegpt(sample))

        logger.info("Built %d preference samples in %s format.", len(samples), self.format)
        return samples
