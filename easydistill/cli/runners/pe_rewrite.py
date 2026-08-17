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

"""Local (no-LLM) runners for the PE rewrite pipeline text stages.

``pe_rewrite_filter`` (stage 4) and ``pe_rewrite_build_sft`` (stage 8) are
thin CLI wrappers over the PERewriteDistillPipeline stage helpers, for
running a single stage on an intermediate jsonl. Neither touches a model
backend, so their configs need no ``backend`` section.
"""

import logging
from collections import Counter

from easydistill.pipeline.pe_rewrite_distill import (
    run_pe_build_sft_stage,
    run_pe_quality_filter_stage,
)
from easydistill.utils import load_dataset_rows, load_expanded_config, save_jsonl

logger = logging.getLogger(__name__)


def run_pe_rewrite_filter(config_path: str) -> None:
    """Filter judged rewrite rows by score thresholds + top-ratio selection.

    Config section ``quality_filter`` supports ``min_scores`` (defaults to the
    plan-doc thresholds), ``keep_top_k`` / ``keep_top_ratio`` and
    ``require_all_metrics``; the top selection averages the seven 0-9 metrics.
    """
    cfg = load_expanded_config(config_path)
    rows = load_dataset_rows(cfg["dataset"]["input_path"])

    filtered = run_pe_quality_filter_stage(cfg.get("quality_filter") or {}, rows)

    scene_kept = Counter(str(r.get("scene") or "unknown") for r in filtered)
    logger.info("Kept rows by scene: %s", dict(scene_kept))

    output_path = cfg["dataset"]["output_path"]
    save_jsonl(output_path, filtered)
    logger.info("pe_rewrite_filter kept %d/%d rows -> %s", len(filtered), len(rows), output_path)


def run_pe_rewrite_build_sft(config_path: str) -> None:
    """Build SFT samples from filtered rewrite rows.

    Each row becomes system + user(original prompt) + assistant(rewritten
    prompt); the per-language student system prompt comes from the
    ``sft.system_prompt_zh_file`` / ``sft.system_prompt_en_file`` config keys.
    """
    cfg = load_expanded_config(config_path)
    rows = load_dataset_rows(cfg["dataset"]["input_path"])

    samples = run_pe_build_sft_stage(cfg.get("sft") or {}, rows)

    output_path = cfg["dataset"]["output_path"]
    save_jsonl(output_path, samples)
    logger.info(
        "pe_rewrite_build_sft built %d/%d SFT samples -> %s",
        len(samples),
        len(rows),
        output_path,
    )
