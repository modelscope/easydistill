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

"""Synthesis operator runner."""

import logging
from typing import Any, List

from easydistill.rewrite import (
    AgenticPromptRewriteOperator,
    InstructionExpansionOperator,
    InstructionRefinementOperator,
    InstructionResponseExtractionOperator,
    SeedAnchoredExpansionOperator,
)
from easydistill.utils import load_expanded_config, save_jsonl

from ..backend_factory import build_backend, check_backend_health, close_backends
from ..data_loaders import load_instruction_rows, load_seed_records, load_string_column

logger = logging.getLogger(__name__)


def _save_synthesis_outputs(
    output_path: str,
    outputs: List[Any],
    output_format: str,
) -> None:
    """Save synthesis outputs to JSONL."""
    rows = []
    for item in outputs:
        if output_format == "instruction" and isinstance(item, str):
            rows.append({"instruction": item})
        elif output_format == "instruction_response" and isinstance(item, tuple):
            rows.append({"instruction": item[0], "response": item[1]})
        else:
            rows.append(item if isinstance(item, dict) else {"output": item})
    save_jsonl(output_path, rows)
    logger.info("Saved synthesis outputs to %s.", output_path)


def run_synthesis(config_path: str, job_type: str) -> None:
    """Run a synthesis operator."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        synth_cfg = cfg.get("synthesis", {})
        output_path = cfg["dataset"]["output_path"]
        output_format = cfg["dataset"].get("output_format", "instruction")

        operator: Any
        outputs: List[Any]
        if job_type == "instruction_expansion":
            seeds = load_string_column(cfg, "instruction_key")
            operator = InstructionExpansionOperator(backend=backend, config=synth_cfg)
            outputs = operator.run(seeds)
        elif job_type == "seed_anchored_expansion":
            seed_records = load_seed_records(cfg)
            operator = SeedAnchoredExpansionOperator(backend=backend, config=synth_cfg)
            outputs = operator.run(seed_records)
        elif job_type == "agentic_rewrite":
            rows = load_instruction_rows(cfg)
            # This operator has its own config section (nested per-step blocks)
            # instead of the flat `synthesis` section.
            operator = AgenticPromptRewriteOperator(
                backend=backend, config=cfg.get("agentic_rewrite", {})
            )
            outputs = operator.run(rows)
        elif job_type == "instruction_refinement":
            seeds = load_string_column(cfg, "instruction_key")
            operator = InstructionRefinementOperator(backend=backend, config=synth_cfg)
            outputs = operator.run(seeds)
        elif job_type == "instruction_response_extraction":
            texts = load_string_column(cfg, "text_key")
            operator = InstructionResponseExtractionOperator(backend=backend, config=synth_cfg)
            outputs = operator.run(texts)
        else:
            raise ValueError(f"Unknown synthesis job_type: {job_type}")

        _save_synthesis_outputs(output_path, outputs, output_format)
    finally:
        close_backends(backend)
