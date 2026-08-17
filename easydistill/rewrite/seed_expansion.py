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

"""Seed-anchored instruction expansion operator.

Unlike :class:`InstructionExpansionOperator` (random in-context mixing, one new
instruction per LLM call), this operator expands each seed independently so the
generated instructions inherit the seed's scenario, which makes scenario quotas
directly controllable through the seed set composition. Each LLM call returns a
JSON array of ``{"topic", "prompt"}`` items, and later rounds pass the topics
generated so far back to the model as a lightweight semantic dedup signal.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest
from easydistill.operators.base import Operator
from easydistill.operators.generation import TextGenerationOperator
from easydistill.prompts import resolve_prompt
from easydistill.utils import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_BASE,
    DEFAULT_RETRY_MAX_WAIT,
    progress,
)

logger = logging.getLogger(__name__)

SeedInput = Union[str, Dict[str, Any]]

_DEFAULT_FIRST_MESSAGE_TEMPLATE = (
    "Seed prompt:\n"
    '"""\n'
    "{seed}\n"
    '"""\n\n'
    "Generate {count} new prompts for the same scenario, simulating natural "
    "inputs from real users."
)

_DEFAULT_FOLLOWUP_MESSAGE_TEMPLATE = (
    "Seed prompt:\n"
    '"""\n'
    "{seed}\n"
    '"""\n\n'
    "The following topics have already been covered, avoid them:\n"
    "{topics}\n\n"
    "Generate {count} brand-new prompts for the same scenario, simulating "
    "natural inputs from real users."
)


def _parse_expansion_array(text: Optional[str]) -> List[Tuple[str, str]]:
    """Parse an LLM response into ``[(topic, prompt), ...]`` pairs.

    Accepts a JSON array of ``{"topic", "prompt"}`` objects, optionally wrapped
    in a markdown code fence. Truncated arrays are repaired by cutting at the
    last ``]``. Plain-string items are accepted with an empty topic.
    """
    if not text:
        return []
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        array = json.loads(text)
    except json.JSONDecodeError:
        end = text.rfind("]")
        if end < 0:
            return []
        try:
            array = json.loads(text[: end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(array, list):
        return []
    pairs: List[Tuple[str, str]] = []
    for item in array:
        if isinstance(item, dict):
            topic = str(item.get("topic") or "").strip()
            prompt = str(item.get("prompt") or "").strip()
            if prompt:
                pairs.append((topic, prompt))
        elif isinstance(item, str) and item.strip():
            pairs.append(("", item.strip()))
    return pairs


class SeedAnchoredExpansionOperator(Operator[List[SeedInput], List[Dict[str, Any]]]):
    """Expand each seed instruction into multiple same-scenario instructions.

    Execution model: seeds run concurrently (``max_workers``), while the rounds
    within one seed run sequentially because each round feeds the accumulated
    topic list back to the model for dedup.

    Input: list of seed instructions, either plain strings or dicts with an
    ``instruction`` field (and an optional ``id`` used for lineage).

    Output: one dict per generated instruction with lineage fields:
    ``{"instruction", "source_seed_id", "round", "topic"}``.

    Configurable fields:
      - prompt_template / prompt_template_file: expansion system prompt.
      - rounds: number of expansion rounds per seed.
      - generations_per_round: instructions requested per round.
      - first_message_template: user message template for round 0 with
        {seed} and {count} placeholders.
      - followup_message_template: user message template for later rounds with
        {seed}, {count} and {topics} placeholders.
      - model_id / temperature / max_tokens: generation parameters.
      - max_workers: number of seeds expanded concurrently.
      - show_progress: whether to show tqdm progress bar over seeds.
      - retry_attempts / retry_backoff_base / retry_max_wait: retry behavior
        per LLM call.
      - round_retry_attempts: extra full-round retries when a round yields no
        parseable expansions (0 disables round-level retry).
    """

    name = "seed_anchored_expansion"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        system_prompt = resolve_prompt(
            self.config, default_file="configs/prompts/pe_rewrite/expansion_prompt.txt"
        )
        self.rounds = int(self.config.get("rounds") or 3)
        if self.rounds <= 0:
            raise ValueError("rounds must be a positive integer.")
        self.generations_per_round = int(self.config.get("generations_per_round") or 10)
        if self.generations_per_round <= 0:
            raise ValueError("generations_per_round must be a positive integer.")
        # 0 is a valid value here, so only substitute the default on None.
        round_retry_attempts = self.config.get("round_retry_attempts")
        self.round_retry_attempts = 2 if round_retry_attempts is None else int(round_retry_attempts)
        if self.round_retry_attempts < 0:
            raise ValueError("round_retry_attempts must be >= 0.")
        # Validated configs may carry explicit None values, so fall back with
        # `or` instead of dict.get defaults.
        self.first_message_template = (
            self.config.get("first_message_template") or _DEFAULT_FIRST_MESSAGE_TEMPLATE
        )
        self.followup_message_template = (
            self.config.get("followup_message_template") or _DEFAULT_FOLLOWUP_MESSAGE_TEMPLATE
        )
        max_workers = int(self.config.get("max_workers") or DEFAULT_MAX_WORKERS)
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        self.max_workers = max_workers
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        temperature = self.config.get("temperature")
        # Inner generator handles a single (seed, round) call at a time; seed
        # level concurrency is owned by this operator's own thread pool.
        self.generator = TextGenerationOperator(
            backend=backend,
            config={
                "system_prompt": system_prompt,
                "model_id": self.config.get("model_id"),
                "temperature": float(temperature) if temperature is not None else 0.9,
                "max_tokens": int(self.config.get("max_tokens") or 4096),
                "show_progress": False,
                "max_workers": 1,
                "raise_on_error": False,
                "retry_attempts": int(self.config.get("retry_attempts") or DEFAULT_RETRY_ATTEMPTS),
                "retry_backoff_base": float(
                    self.config.get("retry_backoff_base") or DEFAULT_RETRY_BACKOFF_BASE
                ),
                "retry_max_wait": float(
                    self.config.get("retry_max_wait") or DEFAULT_RETRY_MAX_WAIT
                ),
            },
        )

    @staticmethod
    def _normalize_seed(seed: SeedInput, index: int) -> Optional[Dict[str, str]]:
        """Normalize a raw seed into ``{"id", "instruction"}`` or None."""
        if isinstance(seed, str):
            instruction = seed.strip()
            seed_id = f"seed_{index}"
        elif isinstance(seed, dict):
            instruction = str(seed.get("instruction") or "").strip()
            seed_id = str(seed.get("id") or f"seed_{index}")
        else:
            return None
        if not instruction:
            return None
        return {"id": seed_id, "instruction": instruction}

    def _build_user_message(self, instruction: str, existing_topics: List[str]) -> str:
        if not existing_topics:
            return self.first_message_template.format(
                seed=instruction, count=self.generations_per_round
            )
        return self.followup_message_template.format(
            seed=instruction,
            count=self.generations_per_round,
            topics="\n".join(f"- {topic}" for topic in existing_topics),
        )

    def _expand_one_seed(self, seed: Dict[str, str]) -> List[Dict[str, Any]]:
        """Run all rounds for a single seed sequentially, deduping by topic."""
        outputs: List[Dict[str, Any]] = []
        existing_topics: List[str] = []
        seen_topics: set = set()
        for round_idx in range(self.rounds):
            # Request-level retries only cover transport errors; a full-round
            # retry additionally covers malformed/truncated JSON responses.
            pairs: List[Tuple[str, str]] = []
            for attempt in range(self.round_retry_attempts + 1):
                suffix = f"_retry{attempt}" if attempt else ""
                request = GenerationRequest(
                    id=f"{seed['id']}_r{round_idx}{suffix}",
                    instruction=self._build_user_message(seed["instruction"], existing_topics),
                    metadata={"task": self.name, "source_seed_id": seed["id"]},
                )
                results = self.generator.run([request])
                pairs = _parse_expansion_array(results[0].response) if results else []
                if pairs:
                    break
                logger.warning(
                    "Seed %s round %d attempt %d/%d produced no parseable expansions.",
                    seed["id"],
                    round_idx,
                    attempt + 1,
                    self.round_retry_attempts + 1,
                )
            if not pairs:
                logger.warning(
                    "Seed %s round %d abandoned after %d attempts.",
                    seed["id"],
                    round_idx,
                    self.round_retry_attempts + 1,
                )
                continue
            dropped = 0
            for topic, prompt in pairs:
                if topic and topic in seen_topics:
                    dropped += 1
                    continue
                outputs.append(
                    {
                        "instruction": prompt,
                        "source_seed_id": seed["id"],
                        "round": round_idx,
                        "topic": topic,
                    }
                )
                if topic:
                    seen_topics.add(topic)
                    existing_topics.append(topic)
            if dropped:
                logger.info(
                    "Seed %s round %d dropped %d expansions with duplicate topics.",
                    seed["id"],
                    round_idx,
                    dropped,
                )
        return outputs

    def run(self, input_data: List[SeedInput]) -> List[Dict[str, Any]]:
        seeds = []
        for index, raw_seed in enumerate(input_data or []):
            seed = self._normalize_seed(raw_seed, index)
            if seed is None:
                logger.warning("Skipping seed at index %d: empty or invalid.", index)
                continue
            seeds.append(seed)
        if not seeds:
            return []

        # Seeds are independent, so they run concurrently; per-seed outputs are
        # collected by seed order to keep the result deterministic.
        per_seed_outputs: List[List[Dict[str, Any]]] = [[] for _ in seeds]
        if self.max_workers <= 1:
            for idx, seed in enumerate(
                progress(
                    seeds,
                    enabled=self.show_progress,
                    total=len(seeds),
                    desc="Expanding seeds",
                )
            ):
                per_seed_outputs[idx] = self._expand_one_seed(seed)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._expand_one_seed, seed): idx
                    for idx, seed in enumerate(seeds)
                }
                for future in progress(
                    as_completed(futures),
                    enabled=self.show_progress,
                    total=len(futures),
                    desc="Expanding seeds",
                ):
                    idx = futures[future]
                    try:
                        per_seed_outputs[idx] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Seed expansion task %d raised: %s",
                            idx,
                            exc,
                        )

        outputs = [record for records in per_seed_outputs for record in records]
        logger.info(
            "%s expanded %d seeds x %d rounds into %d instructions.",
            self.name,
            len(seeds),
            self.rounds,
            len(outputs),
        )
        return outputs
