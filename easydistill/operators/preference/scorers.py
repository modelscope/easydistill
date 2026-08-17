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

"""Scorers for preference candidate responses."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.operators.cot.utils import extract_between_tags
from easydistill.utils.metrics import compute_average_score

logger = logging.getLogger(__name__)


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.strip(".$")
    return text


def _default_extract_answer(text: str) -> Optional[str]:
    """Extract a final answer from a CoT response or reference explanation.

    Tries, in order:
      - \boxed{...}
      - #### answer (GSM8K style)
      - "the answer is ..."
      - the last standalone number (handles textual references like "...= 55.")
      - last non-empty line.
    """
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*(.+)",
        r"the answer is\s*[:\-]?\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback to the last standalone numeric token, allowing commas/decimals.
    numbers: List[str] = re.findall(r"\d[\d,\.]*", text)
    if numbers:
        return numbers[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip() or None


class BaseScorer(ABC):
    """Base class for candidate scorers."""

    @abstractmethod
    def score(
        self,
        instruction: Any,
        candidates: List[str],
        reference: Optional[str] = None,
    ) -> List[float]:
        """Return a score for each candidate. Higher is better."""
        raise NotImplementedError


class LLMJudgeScorer(BaseScorer):
    """Score candidates with an LLM-as-judge evaluator.

    The final score for each candidate is the average of the requested metric
    scores. Boolean metrics (such as `correctness`) are treated as 1.0/0.0.

    Configurable fields:
      - metrics: list of metric names to evaluate (default ["helpfulness",
        "correctness"]).
      - prompts: custom metric prompt templates.
      - prompts_file: path to YAML/JSON file with metric prompts.
      - max_workers, temperature, max_tokens, show_progress, raise_on_error:
        passed to the underlying judge generator.
    """

    name = "llm_judge"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = self.config.get("metrics") or ["helpfulness", "correctness"]
        evaluator_config = {
            **self.config,
            "metrics": self.metrics,
        }
        # Lazy import to avoid a circular dependency between eval and operators.preference.
        from easydistill.eval.instruction_following import InstructionFollowingEvaluator

        self._evaluator = InstructionFollowingEvaluator(backend=backend, config=evaluator_config)

    def score(
        self,
        instruction: Any,
        candidates: List[str],
        reference: Optional[str] = None,
    ) -> List[float]:
        if not candidates:
            return []
        samples = [
            {"id": str(idx), "instruction": instruction, "output": candidate}
            for idx, candidate in enumerate(candidates)
        ]
        results = self._evaluator.run(samples)
        scores = []
        for result in results:
            avg = compute_average_score(result, self.metrics)
            scores.append(avg if avg is not None else 0.0)
        return scores


class CoTScorer(BaseScorer):
    """Score chain-of-thought candidates by correctness and conciseness.

    For each candidate the final answer is extracted and compared against the
    reference answer. The score is:

        correct * (1 / (1 + alpha * reasoning_length))

    so a correct but shorter reasoning chain scores higher. When no reference
    is provided, correctness is 0.0 to avoid treating unverified answers as
    correct.

    Configurable fields:
      - alpha: length penalty coefficient (default 0.001 per character).
      - answer_extractor_pattern: custom regex to extract the final answer.
      - reasoning_tag_start / reasoning_tag_end: tags delimiting reasoning.
      - normalize_answer: whether to normalize before comparison (default True).
    """

    name = "cot"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        alpha = self.config.get("alpha")
        self.alpha = float(alpha) if alpha is not None else 0.001
        self.pattern = self.config.get("answer_extractor_pattern")
        self.reasoning_tag_start = self.config.get("reasoning_tag_start")
        self.reasoning_tag_end = self.config.get("reasoning_tag_end")
        self.normalize_answer = bool(self.config.get("normalize_answer", True))

    def _extract_answer(self, text: str) -> Optional[str]:
        if self.pattern:
            match = re.search(self.pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() if match.groups() else match.group(0).strip()
        return _default_extract_answer(text)

    def _reasoning_length(self, text: str) -> int:
        if self.reasoning_tag_start and self.reasoning_tag_end:
            reasoning = extract_between_tags(text, self.reasoning_tag_start, self.reasoning_tag_end)
            if reasoning:
                return len(reasoning)
        return len(text)

    def score(
        self,
        instruction: Any,
        candidates: List[str],
        reference: Optional[str] = None,
    ) -> List[float]:
        if not candidates:
            return []

        reference_answer = reference
        ref = None
        if reference_answer:
            ref = self._extract_answer(reference_answer)
            if self.normalize_answer and ref:
                ref = _normalize_answer(ref)

        scores = []
        for candidate in candidates:
            extracted = self._extract_answer(candidate)
            if self.normalize_answer and extracted:
                extracted = _normalize_answer(extracted)

            correct = 0.0
            if ref is not None and extracted is not None:
                correct = 1.0 if extracted == ref else 0.0

            length = self._reasoning_length(candidate)
            score = correct * (1.0 / (1.0 + self.alpha * length))
            scores.append(score)

        return scores
