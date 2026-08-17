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

"""PE rewrite distillation pipeline (text segment).

End-to-end text pipeline for the PE rewrite distillation plan
(docs/pe_rewrite_distill_plan_zh.md):

  1. (optional) seed_anchored_expansion — grow seeds into more prompts
  2. agentic_rewrite                    — plan -> rewrite -> reflection chain
  3. pe_rewrite_eval                    — combined 9-metric judge
  4. quality_filter                     — plan-doc thresholds + top-ratio
  5. build_sft                          — student SFT samples (per-language
                                          end-to-end system prompt)

The image stages (image_gen / image_judge / image_filter) are intentionally
not part of this pipeline yet. Per-step model separation (teacher plus vs
judge max) is configured through the stage configs' ``model_id`` overrides on
one shared backend endpoint.
"""

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.eval.pe_rewrite import ALL_METRICS, SCORE_METRICS, PERewriteEvaluator
from easydistill.operators import SFTDatasetBuilder
from easydistill.rewrite import AgenticPromptRewriteOperator, SeedAnchoredExpansionOperator
from easydistill.utils.metrics import compute_average_score

from .base import BaseDistillationPipeline
from .common import run_quality_filter_stage

logger = logging.getLogger(__name__)

# Score thresholds from docs/pe_rewrite_distill_plan_zh.md §3 (filter roles):
# strictest gates on intent fidelity / text rendering / usability, boolean
# hard checks drop on False.
DEFAULT_MIN_SCORES = {
    "intent_fidelity": 7,
    "text_rendering_completeness": 7,
    "usability": 7,
    "detail_enrichment": 6,
    "visual_concreteness": 6,
    "compositional_coverage": 5,
    "scene_alignment": 5,
    "language_consistency": True,
    "no_conflict": True,
}


def run_pe_quality_filter_stage(
    stage_config: Dict[str, Any], data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Quality filter with the PE rewrite plan-doc thresholds as defaults.

    Two passes: global hard gates (``min_scores``), then top selection
    (``keep_top_k`` / ``keep_top_ratio``). The top selection runs **per
    scene** by default (``per_scene: false`` restores global ranking):
    global average-score ranking systematically evicts whole scenes whose
    style yields lower judge scores (observed twice with
    cultural_heritage_art), which would leave scene-shaped holes in the
    training set. Per-scene quotas use ceiling rounding and keep at least
    one row per surviving scene.
    """
    # Validated configs inject explicit None for unset keys, which must not
    # shadow the built-in defaults.
    stage_cfg = {k: v for k, v in (stage_config or {}).items() if v is not None}
    stage_cfg.setdefault("min_scores", DEFAULT_MIN_SCORES)
    per_scene = bool(stage_cfg.pop("per_scene", True))
    keep_top_k = stage_cfg.pop("keep_top_k", None)
    keep_top_ratio = stage_cfg.pop("keep_top_ratio", None)

    # Pass 1: hard score gates, always global.
    survivors = run_quality_filter_stage(stage_cfg, data, list(SCORE_METRICS))
    if keep_top_k is None and keep_top_ratio is None:
        return survivors

    # Pass 2: top selection, per scene (default) or global.
    if not per_scene:
        top_cfg = {"keep_top_k": keep_top_k, "keep_top_ratio": keep_top_ratio}
        return run_quality_filter_stage(top_cfg, survivors, list(SCORE_METRICS))

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in survivors:
        groups.setdefault(str(row.get("scene") or "unknown"), []).append(row)
    kept_ids: Set[int] = set()
    for _scene, rows in groups.items():
        scored = []
        for row in rows:
            avg = compute_average_score(row, list(SCORE_METRICS))
            if avg is not None:
                scored.append((avg, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        if keep_top_k is not None:
            quota = int(keep_top_k)
        else:
            # Ceiling with a floor of 1 so small scene groups are never
            # rounded down to extinction.
            quota = max(1, math.ceil(len(scored) * float(keep_top_ratio)))
        kept_ids.update(id(row) for _, row in scored[:quota])
    filtered = [row for row in survivors if id(row) in kept_ids]
    logger.info(
        "Per-scene top selection kept %d of %d rows: %s",
        len(filtered),
        len(survivors),
        dict(Counter(str(r.get("scene") or "unknown") for r in filtered)),
    )
    return filtered


def run_pe_build_sft_stage(
    sft_config: Dict[str, Any], data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build SFT samples from filtered rewrite rows.

    Each row becomes system + user(original prompt) + assistant(rewritten
    prompt). The student system prompt is an end-to-end rewrite instruction
    (NOT the teacher agent's per-scene SP) chosen per row language via the
    ``system_prompt_zh_file`` / ``system_prompt_en_file`` keys.
    """
    sft_cfg = {k: v for k, v in (sft_config or {}).items() if v is not None}
    system_prompts = {}
    for lang in ("zh", "en"):
        path = sft_cfg.pop(f"system_prompt_{lang}_file", None)
        if path:
            system_prompts[lang] = Path(path).read_text(encoding="utf-8").strip()

    # Build GenerationResults directly (rather than via the shared
    # run_build_sft_stage) so row fields like expansion lineage land in
    # result.metadata, which is what SFTDatasetBuilder copies into samples.
    results = []
    for idx, row in enumerate(data):
        instruction = row.get("instruction")
        response = row.get("response")
        if not instruction or not response:
            continue
        system = row.get("system") or system_prompts.get(str(row.get("language") or "").lower())
        # Judge scores and traces are audit fields; keep them out of SFT
        # sample metadata.
        metadata = {
            k: v
            for k, v in row.items()
            if k not in {"instruction", "response", "system", "id", "agent_trace"}
            and k not in ALL_METRICS
        }
        results.append(
            GenerationResult(
                request=GenerationRequest(
                    id=str(row.get("id", idx)),
                    instruction=str(instruction),
                    system_prompt=system,
                ),
                response=str(response),
                model="pipeline",
                metadata=metadata,
            )
        )
    builder = SFTDatasetBuilder(config=sft_cfg)
    return [sample.model_dump() for sample in builder.run(results)]


class PERewriteDistillPipeline(BaseDistillationPipeline):
    """End-to-end text pipeline for PE rewrite distillation.

    The recommended flow is:
      1. (optional) seed_anchored_expansion
      2. agentic_rewrite
      3. pe_rewrite_eval
      4. quality_filter
      5. build_sft

    The last stage must be `build_sft`. Stage configs may override the model
    per step (e.g. judge on qwen3.7-max while the teacher runs qwen3.7-plus).
    """

    _last_stage = "build_sft"
    _default_eval_metrics = list(SCORE_METRICS)

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        if stage_name == "seed_anchored_expansion":
            expansion_input: List[Union[str, Dict[str, Any]]] = list(data)
            data = SeedAnchoredExpansionOperator(backend=self.backend, config=stage_config).run(
                expansion_input
            )
        elif stage_name == "agentic_rewrite":
            rewrite_input: List[Union[str, Dict[str, Any]]] = list(data)
            data = AgenticPromptRewriteOperator(backend=self.backend, config=stage_config).run(
                rewrite_input
            )
        elif stage_name == "pe_rewrite_eval":
            eval_cfg = {**self.eval_config, **stage_config}
            evaluator = PERewriteEvaluator(backend=self.backend, config=eval_cfg)
            data = evaluator.run(data)
            logger.info("PE rewrite judge aggregates: %s", evaluator.aggregate(data))
        elif stage_name == "quality_filter":
            data = run_pe_quality_filter_stage(stage_config, data)
        elif stage_name == "build_sft":
            sft_cfg = {**self.sft_config, **stage_config}
            data = run_pe_build_sft_stage(sft_cfg, data)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        return data
