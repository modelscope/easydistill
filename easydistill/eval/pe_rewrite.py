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

"""LLM-as-judge evaluation for PE rewrite distillation data.

Scores each (original prompt, rewritten prompt) pair on nine dimensions:
seven 0-9 anchored metrics plus two boolean hard checks (see
docs/pe_rewrite_distill_plan_zh.md §4 Step 3). All nine dimensions are judged
in a SINGLE model call per sample (the rewritten prompt is long, so per-metric
calls would re-read the same input nine times); the judge answers with one
JSON object carrying every score. Unlike the generic evaluators, the output
rows keep every field of the input row (scene, lineage, agent_trace, ...) so
downstream quality filtering can group scores by scene.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest
from easydistill.eval._common import extract_first_json_object
from easydistill.operators.generation import TextGenerationOperator
from easydistill.prompts import resolve_prompts
from easydistill.utils.image import format_prompt_safely

logger = logging.getLogger(__name__)

# Canonical metric order; also the JSON keys the judge must return.
SCORE_METRICS = (
    "intent_fidelity",
    "text_rendering_completeness",
    "detail_enrichment",
    "visual_concreteness",
    "compositional_coverage",
    "scene_alignment",
    "usability",
)
BOOL_METRICS = ("language_consistency", "no_conflict")
ALL_METRICS = SCORE_METRICS + BOOL_METRICS


def _parse_judge_response(raw: Optional[str]) -> Dict[str, Any]:
    """Parse the judge's combined JSON verdict into metric fields.

    Degrades gracefully: strict JSON -> first {...} block -> per-field regex
    salvage. Missing or unparseable metrics come back as None so downstream
    filtering can treat them as failures.
    """
    scores: Dict[str, Any] = dict.fromkeys(ALL_METRICS)
    if not raw or not raw.strip():
        return scores
    text = raw.strip()

    parsed: Optional[Dict[str, Any]] = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            parsed = candidate
    except (json.JSONDecodeError, ValueError):
        block = extract_first_json_object(text)
        if block:
            try:
                candidate = json.loads(block)
                if isinstance(candidate, dict):
                    parsed = candidate
            except (json.JSONDecodeError, ValueError):
                parsed = None

    def _coerce_score(value: Any) -> Optional[int]:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        return score if 0 <= score <= 9 else None

    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return None

    if parsed is not None:
        for metric in SCORE_METRICS:
            scores[metric] = _coerce_score(parsed.get(metric))
        for metric in BOOL_METRICS:
            scores[metric] = _coerce_bool(parsed.get(metric))
        return scores

    # Field-level regex salvage for broken JSON (e.g. truncated output).
    for metric in SCORE_METRICS:
        match = re.search(rf'"{metric}"\s*:\s*(\d+)', text)
        if match:
            scores[metric] = _coerce_score(match.group(1))
    for metric in BOOL_METRICS:
        match = re.search(rf'"{metric}"\s*:\s*(true|false|[01])', text, re.IGNORECASE)
        if match:
            scores[metric] = _coerce_bool(match.group(1))
    return scores


class PERewriteEvaluator:
    """Judge prompt rewrites for the PE rewrite distillation pipeline.

    Expects rows shaped like the agentic rewrite output:
    ``{"instruction": original prompt, "response": rewritten prompt,
    "scene": str, "language": str, ...}``. One judge call per row scores all
    nine metrics at once; the per-sample ``scene`` / ``language`` fields are
    injected into the combined prompt template.

    Config fields: ``prompt_template`` / ``prompts_file`` (template with a
    ``combined`` key), ``model_id`` (judge model override, keeping the judge
    separate from the teacher on a shared backend), ``max_workers``,
    ``temperature``, ``max_tokens``, ``show_progress``, ``raise_on_error``,
    ``strict_mode``.
    """

    name = "pe_rewrite_evaluator"
    DEFAULT_PROMPTS_FILE = "configs/prompts/pe_rewrite/eval_prompts.yaml"
    PROMPT_KEY = "combined"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.config = config or {}
        self.metrics = list(ALL_METRICS)

        template = self.config.get("prompt_template")
        if not template:
            prompts = resolve_prompts(
                self.config, defaults={}, default_file=self.DEFAULT_PROMPTS_FILE
            )
            template = prompts.get(self.PROMPT_KEY)
        if not template:
            raise ValueError(
                f"Combined judge prompt not found: expected key "
                f"'{self.PROMPT_KEY}' in {self.DEFAULT_PROMPTS_FILE} or a "
                f"'prompt_template' config entry."
            )
        self.prompt_template = template

        self.max_workers = int(self.config.get("max_workers") or 10)
        self.temperature = float(self.config.get("temperature") or 0.0)
        # One call carries all nine verdicts; keep headroom above the tag-only
        # responses the per-metric judges used to return.
        self.max_tokens = int(self.config.get("max_tokens") or 512)
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self.raise_on_error = bool(self.config.get("raise_on_error") or False)
        self.strict_mode = bool(self.config.get("strict_mode") or False)

        self.generator = TextGenerationOperator(
            backend=backend,
            config={
                "model_id": self.config.get("model_id"),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "show_progress": self.show_progress,
                "max_workers": self.max_workers,
                "raise_on_error": self.raise_on_error,
            },
        )

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate rows and merge the nine scores back into each row."""
        if not samples:
            return []

        prepared: List[Dict[str, Any]] = []
        skipped = 0
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                logger.warning("Skipping non-dict sample at index %d.", index)
                continue
            row = dict(sample)
            row.setdefault("id", str(index))
            row["id"] = str(row["id"])
            instruction = row.get("instruction") or ""
            output = row.get("response") or row.get("output") or ""
            if not instruction or not output:
                skipped += 1
                if self.strict_mode:
                    raise ValueError(
                        f"Sample {row['id']} has an empty instruction or output. "
                        "Set strict_mode=False to skip invalid samples."
                    )
                logger.warning("Skipping sample %s with empty instruction or output.", row["id"])
                continue
            prepared.append(row)

        if skipped:
            pct = 100.0 * skipped / len(samples)
            logger.log(
                logging.ERROR if pct > 50 else logging.WARNING,
                "Skipped %d of %d evaluation samples (%.1f%%) due to missing "
                "instruction or output.",
                skipped,
                len(samples),
                pct,
            )
        if not prepared:
            logger.error("No valid evaluation samples remain after filtering.")
            return []

        requests = []
        for row in prepared:
            prompt = format_prompt_safely(
                self.prompt_template,
                instruction=row.get("instruction") or "",
                output=row.get("response") or row.get("output") or "",
                scene=str(row.get("scene") or "general"),
                language=str(row.get("language") or ""),
            )
            requests.append(
                GenerationRequest(
                    id=f"{row['id']}_judge",
                    instruction=prompt,
                    metadata={"sample_id": row["id"]},
                )
            )

        results = self.generator.run(requests)
        scores_by_sample = {
            str(result.request.metadata.get("sample_id")): _parse_judge_response(result.response)
            for result in results
        }

        merged = []
        for row in prepared:
            scores = scores_by_sample.get(row["id"], dict.fromkeys(ALL_METRICS))
            merged.append({**row, **scores})
        logger.info("Evaluated %d samples on metrics: %s", len(merged), ", ".join(self.metrics))
        return merged

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average scores per metric."""
        aggregates: Dict[str, Any] = {}
        for metric in self.metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            aggregates[metric] = sum(values) / len(values) if values else None
        return aggregates
