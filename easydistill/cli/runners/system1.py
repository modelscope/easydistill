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

"""System-1 distillation runner: raw rows + schema -> RLCD training cases.

Builds the teacher backend from the config, then drives the four-stage
system1 pipeline (build_cases -> elicit -> aggregate -> build_dataset) with
stage-level resume: the longest prefix of stages whose output file already
exists is skipped and its last output feeds the remaining stages. The
``labeled`` variant drops elicit/aggregate (gold passes straight through),
so those runs never call the teacher. Each stage is also exposed as a
standalone job for debugging or resuming from an intermediate JSONL.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from easydistill.backends.base import ModelBackend
from easydistill.operators.system1 import (
    System1AggregateOperator,
    System1BuildCasesOperator,
    System1BuildDatasetOperator,
    System1ElicitOperator,
)
from easydistill.pipeline import System1DistillationPipeline
from easydistill.utils import load_dataset_rows, load_expanded_config, save_jsonl

from ..backend_factory import build_backend, check_backend_health, close_backends

logger = logging.getLogger(__name__)

_VARIANTS = ("labeled", "distill", "mixed")

# Stages whose operators can reuse previously finished work.
_RESUMABLE_STAGES = ("elicit",)


def _configured_stages(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy stages, propagating ``output_path`` into each stage config.

    Operators stream partial output to ``stage_config["output_path"]``; the
    pipeline-level field alone is only written when a stage finishes.
    """
    configured = []
    for stage in stages:
        stage = {**stage, "config": dict(stage.get("config") or {})}
        path = stage.get("output_path")
        if path and not stage["config"].get("output_path"):
            stage["config"]["output_path"] = path
        configured.append(stage)
    return configured


def _apply_resume_plan(
    stages: List[Dict[str, Any]],
    resume: bool,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Return (input source or None, stages to run).

    With resume enabled the longest prefix of stages whose output file
    already exists is considered complete and its last output feeds the
    remaining stages. The final stage always re-runs, so re-invoking a
    finished pipeline is safe. The elicit stage additionally picks up its
    own previous output or streaming partial file to reuse finished
    records.
    """
    if not resume:
        return None, stages
    done = 0
    for idx, stage in enumerate(stages):
        path = stage.get("output_path")
        if idx == done and path and os.path.exists(path) and os.path.getsize(path) > 0:
            done = idx + 1
    done = min(done, len(stages) - 1)
    if done == 0:
        return None, stages
    trimmed = stages[done:]
    source = stages[done - 1]["output_path"]
    logger.info(
        "Resume: %d of %d stages already have output; restarting from stage %d (%s).",
        done,
        len(stages),
        done + 1,
        trimmed[0]["stage"],
    )
    first = trimmed[0]
    first_path = first.get("output_path")
    if (
        first["stage"] in _RESUMABLE_STAGES
        and first_path
        and os.path.exists(first_path)
        and not first["config"].get("resume_from")
    ):
        first["config"]["resume_from"] = first_path
    return source, trimmed


def _effective_stages(
    stages: List[Dict[str, Any]],
    variant: str,
) -> List[Dict[str, Any]]:
    """Drop the teacher-dependent stages for the gold-passthrough variant."""
    if variant == "labeled":
        return [
            stage for stage in stages if stage.get("stage") not in ("elicit", "aggregate")
        ]
    return list(stages)


def _run_pipeline(
    cfg: Dict[str, Any],
    system1_cfg: Dict[str, Any],
    stages: List[Dict[str, Any]],
    dataset_cfg: Dict[str, Any],
    backend: ModelBackend,
    resume: bool,
) -> None:
    """Run the system1 pipeline over the configured dataset."""
    source, trimmed = _apply_resume_plan(_configured_stages(stages), resume)
    pipeline = System1DistillationPipeline(
        backend=backend,
        pipeline_config=trimmed,
        dataset_config=dataset_cfg,
        generation_config=cfg.get("generation", {}),
        sft_config=cfg.get("sft", {}),
        system1_config=system1_cfg,
    )
    if source:
        pipeline.run_with_data(load_dataset_rows(source))
    else:
        pipeline.run()


def run_system1_build_cases(config_path: str) -> None:
    """Build typed-decision cases from raw rows and a schema (local, no LLM)."""
    cfg = load_expanded_config(config_path)
    rows = load_dataset_rows(cfg["dataset"]["input_path"])

    built = System1BuildCasesOperator(config=cfg.get("system1") or {}).run(rows)

    output_path = cfg["dataset"]["output_path"]
    save_jsonl(output_path, built)
    logger.info(
        "system1_build_cases built %d cases from %d rows -> %s",
        len(built),
        len(rows),
        output_path,
    )


def run_system1_elicit(config_path: str) -> None:
    """Elicit teacher probability distributions for every case question."""
    cfg = load_expanded_config(config_path)
    system1_cfg = dict(cfg.get("system1") or {})
    dataset_cfg = cfg["dataset"]
    system1_cfg.setdefault("output_path", dataset_cfg["output_path"])

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        rows = load_dataset_rows(dataset_cfg["input_path"])
        elicited = System1ElicitOperator(backend=backend, config=system1_cfg).run(rows)

        output_path = dataset_cfg["output_path"]
        save_jsonl(output_path, elicited)
        logger.info(
            "system1_elicit elicited %d/%d rows -> %s",
            len(elicited),
            len(rows),
            output_path,
        )
    finally:
        close_backends(backend)


def run_system1_aggregate(config_path: str) -> None:
    """Aggregate elicitation records into teacher labels (local, no LLM)."""
    cfg = load_expanded_config(config_path)
    rows = load_dataset_rows(cfg["dataset"]["input_path"])

    aggregated = System1AggregateOperator(config=cfg.get("system1") or {}).run(rows)

    output_path = cfg["dataset"]["output_path"]
    save_jsonl(output_path, aggregated)
    logger.info(
        "system1_aggregate wrote %d/%d rows -> %s",
        len(aggregated),
        len(rows),
        output_path,
    )


def run_system1_build_dataset(config_path: str) -> None:
    """Emit RLCD training-ready case rows (local, no LLM)."""
    cfg = load_expanded_config(config_path)
    rows = load_dataset_rows(cfg["dataset"]["input_path"])

    built = System1BuildDatasetOperator(config=cfg.get("system1") or {}).run(rows)

    output_path = cfg["dataset"]["output_path"]
    save_jsonl(output_path, built)
    logger.info(
        "system1_build_dataset built %d/%d cases -> %s",
        len(built),
        len(rows),
        output_path,
    )


def run_system1_distill(config_path: str) -> None:
    """Run the system-1 distillation pipeline from a config file."""
    cfg = load_expanded_config(config_path)

    system1_cfg = cfg.setdefault("system1", {})
    variant = system1_cfg.get("variant", "distill")
    if variant not in _VARIANTS:
        raise ValueError(f"system1.variant must be one of {_VARIANTS}, got {variant!r}.")
    stages = _effective_stages(cfg["pipeline"], variant)
    dataset_cfg = cfg["dataset"]

    resume_cfg = system1_cfg.get("resume")
    resume = True if resume_cfg is None else bool(resume_cfg)

    needs_backend = any(stage.get("stage") == "elicit" for stage in stages)
    backend = build_backend(cfg["backend"]) if needs_backend else None
    try:
        if needs_backend:
            check_backend_health(backend)
        _run_pipeline(cfg, system1_cfg, stages, dataset_cfg, backend, resume)
    finally:
        if backend is not None:
            close_backends(backend)
