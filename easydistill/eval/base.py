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

"""Base class for LLM-as-judge evaluators."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest
from easydistill.operators.generation import TextGenerationOperator
from easydistill.prompts import resolve_prompts
from easydistill.utils import build_multimodal_user_content, format_prompt_safely

logger = logging.getLogger(__name__)


def _extract_score(text: str) -> Optional[int]:
    """Extract an integer score from a judge response.

    First look for `<score>...</score>` tags. If they are missing, fall back
    to the first standalone integer, and finally to leading boolean words
    ("true"/"false"/"correct"/"incorrect"). Returns None if no score is found.
    """
    text = text.strip()
    match = re.search(r"<score>(\d+)</score>", text)
    if match:
        return int(match.group(1))
    # Fallback: look for the first standalone integer or boolean word.
    num_match = re.search(r"\b(\d+)\b", text)
    if num_match:
        return int(num_match.group(1))
    lowered = text.lower()
    bool_match = re.search(r"\b(true|false|correct|incorrect)\b", lowered)
    if bool_match:
        return 1 if bool_match.group(1) in {"true", "correct"} else 0
    return None


class LLMJudgeEvaluator(ABC):
    """Evaluate instruction/output pairs with an LLM judge.

    Subclasses define:
      - name: evaluator identifier.
      - DEFAULT_PROMPTS_FILE: optional path to a YAML/JSON file containing
        metric -> prompt templates.
      - BOOL_METRICS: optional set of metrics that should be converted to bool.
      - _extract_sample: how to extract (sample_id, instruction, output) from a row.

    Configurable fields:
      - metrics: list of metric names to compute.
      - prompts: dict of custom prompt templates per metric.
      - prompts_file: path to a YAML/JSON file containing metric -> prompt.
      - max_workers: concurrency for calling the judge model.
      - temperature, max_tokens: generation params for the judge.
      - show_progress: whether to show a progress bar.
      - raise_on_error: if True, raise on first generation failure.
      - strict_mode: if True, raise ValueError when any sample has an empty
        instruction or output instead of silently skipping it.
    """

    name = "llm_judge"
    DEFAULT_PROMPTS_FILE: Optional[str] = None
    BOOL_METRICS: Optional[Set[str]] = None

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.config = config or {}
        defaults = getattr(self, "DEFAULT_PROMPTS", None) or {}
        default_file = getattr(self, "DEFAULT_PROMPTS_FILE", None)
        self.prompts = resolve_prompts(self.config, defaults=defaults, default_file=default_file)
        self.metrics = self.config.get("metrics", list(self.prompts.keys()))
        missing_metrics = [m for m in self.metrics if m not in self.prompts]
        if missing_metrics:
            raise ValueError(
                f"No prompt template available for metrics: {missing_metrics}. "
                f"Supported metrics are: {list(self.prompts.keys())}."
            )
        self.max_workers = int(self.config.get("max_workers") or 10)
        self.temperature = float(self.config.get("temperature") or 0.0)
        self.max_tokens = int(self.config.get("max_tokens") or 512)
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self.raise_on_error = bool(self.config.get("raise_on_error") or False)
        self.strict_mode = bool(self.config.get("strict_mode") or False)

        self.generator = TextGenerationOperator(
            backend=backend,
            config={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "show_progress": self.show_progress,
                "max_workers": self.max_workers,
                "raise_on_error": self.raise_on_error,
            },
        )

    @abstractmethod
    def _extract_sample(self, sample: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract (sample_id, instruction, output) from a sample row."""
        raise NotImplementedError

    def _extract_images(self, sample: Dict[str, Any]) -> List[str]:
        """Extract optional image references for multi-modal evaluation."""
        return []

    def _build_metric_request(
        self,
        sample_id: str,
        instruction: Any,
        output: Any,
        images: List[str],
        metric: str,
    ) -> GenerationRequest:
        prompt_template = self.prompts[metric]
        prompt = format_prompt_safely(prompt_template, instruction=instruction, output=output)
        content = build_multimodal_user_content(prompt, images) if images else prompt
        return GenerationRequest(
            id=f"{sample_id}_{metric}",
            instruction=content,
            metadata={"sample_id": sample_id, "metric": metric},
        )

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a list of samples on all configured metrics."""
        if not samples:
            return []

        all_requests = []
        sample_info = []
        skipped = 0
        for sample in samples:
            sample_id, instruction, output = self._extract_sample(sample)
            if not instruction or not output:
                skipped += 1
                if self.strict_mode:
                    raise ValueError(
                        f"Sample {sample_id} has an empty instruction or output. "
                        "Set strict_mode=False to skip invalid samples."
                    )
                logger.warning("Skipping sample %s with empty instruction or output.", sample_id)
                continue
            images = self._extract_images(sample)
            sample_info.append((sample_id, instruction, output, images))
            for metric in self.metrics:
                all_requests.append(
                    self._build_metric_request(sample_id, instruction, output, images, metric)
                )

        if skipped:
            pct = 100.0 * skipped / len(samples)
            log_level = logging.ERROR if pct > 50 else logging.WARNING
            logger.log(
                log_level,
                "Skipped %d of %d evaluation samples (%.1f%%) due to missing "
                "instruction or output.",
                skipped,
                len(samples),
                pct,
            )
        if not sample_info:
            logger.error("No valid evaluation samples remain after filtering.")
            return []

        results = self.generator.run(all_requests)

        scores_by_sample: Dict[str, Dict[str, Any]] = {}
        for result in results:
            result_sample_id = result.request.metadata.get("sample_id")
            result_metric = result.request.metadata.get("metric")
            if result_sample_id is None or result_metric is None:
                continue
            score = _extract_score(result.response)
            if self.BOOL_METRICS and result_metric in self.BOOL_METRICS and score is not None:
                score = bool(score)
            scores_by_sample.setdefault(str(result_sample_id), {})[str(result_metric)] = score

        evaluated = []
        for sample_id, instruction, output, images in sample_info:
            row = {
                "id": sample_id,
                "instruction": instruction,
                "output": output,
                **scores_by_sample.get(sample_id, {}),
            }
            if images:
                row["images"] = images
            evaluated.append(row)

        logger.info(
            "Evaluated %d samples on metrics: %s",
            len(evaluated),
            ", ".join(self.metrics),
        )
        return evaluated

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average scores per metric."""
        aggregates = {}
        for metric in self.metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            if values:
                aggregates[metric] = sum(values) / len(values)
            else:
                aggregates[metric] = None
        return aggregates
