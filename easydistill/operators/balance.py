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

"""Instruction balancing / task-aware curriculum planning operator."""

import logging
import random
import re
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.utils import format_prompt_safely

from .base import Operator
from .generation import TextGenerationOperator

logger = logging.getLogger(__name__)

# Target distribution from the original DistilQwen2 recipe.
DEFAULT_DISTILQWEN2_DISTRIBUTION: Dict[str, float] = {
    "Math": 0.167,
    "Code Generation": 0.083,
    "Writing": 0.017,
    "Computer Science": 0.017,
    "Reasoning": 0.167,
    "Complex Format": 0.017,
    "Code Debug": 0.083,
    "Common-Sense": 0.017,
    "Counterfactual": 0.017,
    "Multilingual": 0.017,
    "Roleplay": 0.017,
    "Biology": 0.017,
    "Technology": 0.017,
    "Ethics": 0.017,
    "Sport": 0.017,
    "Law": 0.017,
    "Medicine": 0.017,
    "Literature": 0.017,
    "Entertainment": 0.017,
    "Art": 0.017,
    "Music": 0.017,
    "Toxicity": 0.017,
    "Economy": 0.017,
    "Physics": 0.017,
    "History": 0.017,
    "Chemistry": 0.017,
    "Philosophy": 0.017,
    "Health": 0.017,
    "Ecology": 0.017,
    "Grammar": 0.017,
    "Paraphrase": 0.017,
    "Others": 0.041,
}

DEFAULT_CATEGORY_PROMPT = (
    "You are a data annotation expert. Please classify the task type or "
    "domain of #Given Instruction#.\n"
    "The task type or domain should be in the list: [{categories}]. "
    "You should place your answer enclosed within <answer></answer> tags, "
    "such as <answer>Math</answer>. Do not return anything else.\n"
    "\n"
    "#Given Instruction#:\n"
    "{instruction}"
)


class InstructionBalancer(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Classify instructions by task/domain and resample to a target distribution.

    This operator performs two steps:
      1. Uses a backend (teacher model) to classify each instruction into a
         category such as Math, Code Generation, or Reasoning.
      2. Resamples the dataset so each category matches the target distribution.
         Categories with too many samples are downsampled; categories with too
         few are upsampled by repeating existing samples.

    Configurable fields:
      - instruction_key: the field that contains the instruction text
        (default: "instruction").
      - category_key: the field to store the assigned category
        (default: "category").
      - categories: list of valid categories (defaults to DistilQwen2 list).
      - target_distribution: dict mapping category to target ratio
        (defaults to DistilQwen2 distribution).
      - category_prompt: prompt template for classification; must contain
        "{categories}" and "{instruction}" placeholders.
      - system_prompt: optional system prompt for the classifier.
      - max_workers: concurrency for classification (default 1).
      - show_progress: whether to show progress bars.
      - seed: random seed for resampling.
      - model_id, temperature, max_tokens: passed to the generation backend.
    """

    name = "instruction_balance"

    def __init__(
        self,
        backend: ModelBackend,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.backend = backend
        self.instruction_key = self.config.get("instruction_key") or "instruction"
        self.category_key = self.config.get("category_key") or "category"
        self.categories: List[str] = self.config.get("categories") or list(
            DEFAULT_DISTILQWEN2_DISTRIBUTION.keys()
        )
        raw_distribution = (
            self.config.get("target_distribution") or DEFAULT_DISTILQWEN2_DISTRIBUTION
        )
        self.target_distribution = self._normalize_distribution(raw_distribution)
        self.category_prompt = self.config.get("category_prompt") or DEFAULT_CATEGORY_PROMPT
        self.system_prompt = self.config.get("system_prompt")
        self.max_workers = int(self.config.get("max_workers") or 1)
        show_progress = self.config.get("show_progress")
        self.show_progress = True if show_progress is None else bool(show_progress)
        seed_value = self.config.get("seed")
        self.seed = int(seed_value) if seed_value is not None else 42
        self.model_id = self.config.get("model_id")
        self.temperature = float(self.config.get("temperature") or 0.0)
        self.max_tokens = int(self.config.get("max_tokens") or 512)

        self._generation_op = TextGenerationOperator(
            backend=backend,
            config={
                "system_prompt": self.system_prompt,
                "model_id": self.model_id,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "show_progress": self.show_progress,
                "max_workers": self.max_workers,
                "raise_on_error": False,
            },
        )

    @staticmethod
    def _normalize_distribution(
        distribution: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalize a target distribution so its ratios sum to 1.

        Always returns a shallow copy so callers cannot accidentally mutate a
        shared module-level constant or the caller's own dict.
        """
        if not distribution:
            return dict(distribution)
        total = sum(distribution.values())
        if total == 0:
            return dict(distribution)
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Target distribution ratios sum to %.6f; normalizing to 1.0.",
                total,
            )
            return {k: v / total for k, v in distribution.items()}
        return dict(distribution)

    def _extract_category(self, text: str) -> str:
        """Extract a category from a model response using <answer>...</answer>."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            category = match.group(1).strip()
            if category in self.categories:
                return category
        # Fall back to a whole-phrase match if the model omitted the XML tags.
        for category in self.categories:
            pattern = re.compile(
                r"\b" + re.escape(category) + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(text):
                return category
        return "Others"

    def _build_classification_requests(self, rows: List[Dict[str, Any]]) -> List[GenerationRequest]:
        """Build classification requests for each input row."""
        categories_text = ", ".join(self.categories)
        requests: List[GenerationRequest] = []
        for idx, row in enumerate(rows):
            instruction = row.get(self.instruction_key, "")
            prompt = format_prompt_safely(
                self.category_prompt,
                categories=categories_text,
                instruction=instruction,
            )
            requests.append(
                GenerationRequest(
                    id=str(idx),
                    instruction=prompt,
                    metadata={"row_index": idx},
                )
            )
        return requests

    def _classify_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify every row and attach the category field."""
        if not rows:
            return []
        requests = self._build_classification_requests(rows)
        results: List[GenerationResult] = self._generation_op.run(requests)
        classifications: Dict[int, str] = {}
        for result in results:
            idx = result.request.metadata.get("row_index")
            if idx is not None:
                classifications[idx] = self._extract_category(result.response)

        classified: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            category = classifications.get(idx, "Others")
            new_row = dict(row)
            new_row[self.category_key] = category
            classified.append(new_row)
            if idx not in classifications:
                logger.warning(
                    "Failed to classify instruction at index %s; defaulting to 'Others'.",
                    idx,
                )
        return classified

    def _resample(
        self,
        classified: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resample classified rows to match the target distribution exactly."""
        total = len(classified)
        if total == 0:
            return []

        rng = random.Random(self.seed)
        by_category: Dict[str, List[Dict[str, Any]]] = {
            category: [] for category in self.categories
        }
        for row in classified:
            by_category.setdefault(row[self.category_key], []).append(row)

        # Compute fractional targets for categories that actually appear.
        targets = {
            category: total * ratio
            for category, ratio in self.target_distribution.items()
            if by_category.get(category)
        }
        if not targets:
            logger.warning("No classified categories matched the target distribution.")
            return []

        floors = {category: int(count) for category, count in targets.items()}
        assigned = sum(floors.values())
        remaining = total - assigned
        if remaining > 0:
            # Distribute the leftover rows to categories with the largest
            # fractional remainders so the final size equals the input size.
            sorted_categories = sorted(
                targets.keys(),
                key=lambda c: (targets[c] - floors[c]),
                reverse=True,
            )
            for category in sorted_categories[:remaining]:
                floors[category] += 1
        elif remaining < 0:
            # Floors exceed total (e.g. target ratios sum > 1.0 due to
            # floating-point). Trim from categories with the largest floors.
            sorted_categories = sorted(
                floors.keys(),
                key=lambda c: floors[c],
                reverse=True,
            )
            for category in sorted_categories:
                if remaining >= 0:
                    break
                trim = min(floors[category] - 1, -remaining)
                floors[category] -= trim
                remaining += trim

        resampled: List[Dict[str, Any]] = []
        for category, target_count in floors.items():
            if target_count == 0:
                continue
            samples = by_category.get(category, [])
            if not samples:
                logger.warning(
                    "Category '%s' has no samples after classification; skipping.",
                    category,
                )
                continue
            if len(samples) >= target_count:
                chosen = rng.sample(samples, target_count)
            else:
                repeats = target_count // len(samples)
                remainder = target_count % len(samples)
                chosen = samples * repeats
                if remainder:
                    chosen.extend(rng.sample(samples, remainder))
            resampled.extend(chosen)

        rng.shuffle(resampled)
        if len(resampled) != total:
            logger.warning(
                "Resampled size (%d) differs from input size (%d); some categories "
                "may have had no valid samples.",
                len(resampled),
                total,
            )
        return resampled

    def run(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify and resample the instruction dataset."""
        classified = self._classify_rows(input_data)
        return self._resample(classified)
