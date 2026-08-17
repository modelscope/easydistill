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

"""CLI runners for T2I/TI2I standalone evaluators.

These runners expose the standalone ``t2i_single_model`` / ``t2i_multi_model`` /
``ti2i_single_model`` / ``ti2i_multi_model`` evaluator modules through the main
``easydistill`` CLI, using the same config files that their standalone scripts
accept.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from easydistill.backends.base import ModelBackend
from easydistill.utils import load_dataset_rows, load_expanded_config

from ..backend_factory import close_backends

logger = logging.getLogger(__name__)


def _output_dir(cfg: Dict[str, Any]) -> Path:
    """Resolve the evaluator output directory from the dataset config."""
    dataset = cfg.get("dataset", {})
    if dataset.get("output_dir"):
        return Path(dataset["output_dir"])
    if dataset.get("output_path"):
        return Path(dataset["output_path"]).parent
    raise ValueError("dataset.output_dir or dataset.output_path is required.")


def _resolve_teacher(cfg: Dict[str, Any], teachers: Dict[str, ModelBackend]) -> str:
    """Return the configured teacher label, validating it exists in the pool."""
    teacher = str(cfg.get("teacher") or "")
    if not teacher:
        raise ValueError(
            f"'teacher' must be set in config. Available teachers: {sorted(teachers)}"
        )
    if teacher not in teachers:
        raise ValueError(
            f"Teacher '{teacher}' not in configured teachers: {sorted(teachers)}"
        )
    return teacher


def _resolve_image_paths(sample: Dict[str, Any], keys: List[str], base_dir: Path) -> None:
    """Resolve relative image paths against the seed file directory."""
    for key in keys:
        value = sample.get(key)
        if value and not str(value).startswith(("http://", "https://", "/")):
            sample[key] = str((base_dir / str(value)).resolve())


# ---------------------------------------------------------------------------
# T2I runners
# ---------------------------------------------------------------------------


def run_t2i_single_model_eval(config_path: str) -> None:
    """Run single-teacher T2I evaluation via the main CLI."""
    from easydistill.eval.t2i_single_model import (
        T2ISingleModelEvaluator,
        _build_backends_from_config,
        _write_artifacts,
    )

    cfg = load_expanded_config(config_path)
    teachers, reason_model = _build_backends_from_config(cfg)
    teacher = _resolve_teacher(cfg, teachers)

    try:
        evaluator = T2ISingleModelEvaluator(
            teacher=teacher,
            backend=teachers[teacher],
            reason_model=reason_model,
            config=dict(cfg.get("eval") or {}),
        )

        seed_path = Path(cfg["dataset"]["input_path"])
        samples = load_dataset_rows(str(seed_path))
        for sample in samples:
            _resolve_image_paths(sample, ["image"], seed_path.parent)

        results = evaluator.run(samples)
        summary = evaluator.aggregate(results)
        _write_artifacts(_output_dir(cfg), results, summary)
        logger.info("T2I single-model eval saved %d cases.", len(results))
    finally:
        close_backends(*teachers.values(), reason_model)


def run_t2i_multi_model_eval(config_path: str) -> None:
    """Run multi-teacher T2I evaluation via the main CLI."""
    from easydistill.eval.t2i_multi_model import (
        T2IMultiModelEvaluator,
        _build_teachers_from_config,
        _write_artifacts,
    )

    cfg = load_expanded_config(config_path)
    teachers, arbiter, reason_model = _build_teachers_from_config(cfg)

    try:
        evaluator = T2IMultiModelEvaluator(
            teachers=teachers,
            arbiter=arbiter,
            reason_model=reason_model,
            config=dict(cfg.get("eval") or {}),
        )

        seed_path = Path(cfg["dataset"]["input_path"])
        samples = load_dataset_rows(str(seed_path))
        for sample in samples:
            _resolve_image_paths(sample, ["image"], seed_path.parent)

        results = evaluator.run(samples)
        summary = evaluator.aggregate(results)
        _write_artifacts(_output_dir(cfg), results, summary)
        logger.info("T2I multi-model eval saved %d cases.", len(results))
    finally:
        close_backends(*teachers.values(), arbiter, reason_model)


# ---------------------------------------------------------------------------
# TI2I runners
# ---------------------------------------------------------------------------


def run_ti2i_single_model_eval(config_path: str) -> None:
    """Run single-teacher TI2I evaluation via the main CLI."""
    from easydistill.eval.ti2i_single_model import (
        TI2ISingleModelEvaluator,
        _build_backends_from_config,
        _write_artifacts,
    )

    cfg = load_expanded_config(config_path)
    teachers, reason_model = _build_backends_from_config(cfg)
    teacher = _resolve_teacher(cfg, teachers)

    try:
        evaluator = TI2ISingleModelEvaluator(
            teacher=teacher,
            backend=teachers[teacher],
            reason_model=reason_model,
            config=dict(cfg.get("eval") or {}),
        )

        seed_path = Path(cfg["dataset"]["input_path"])
        samples = load_dataset_rows(str(seed_path))
        for sample in samples:
            _resolve_image_paths(sample, ["before_image", "after_image"], seed_path.parent)
            refs = sample.get("reference_images") or []
            sample["reference_images"] = [
                str((seed_path.parent / str(r)).resolve())
                if r and not str(r).startswith(("http://", "https://", "/"))
                else str(r)
                for r in refs
            ]

        results = evaluator.run(samples)
        summary = evaluator.aggregate(results)
        _write_artifacts(_output_dir(cfg), results, summary)
        logger.info("TI2I single-model eval saved %d cases.", len(results))
    finally:
        close_backends(*teachers.values(), reason_model)


def run_ti2i_multi_model_eval(config_path: str) -> None:
    """Run multi-teacher TI2I evaluation via the main CLI."""
    from easydistill.eval.ti2i_multi_model import (
        TI2IMultiModelEvaluator,
        _build_teachers_from_config,
        _write_artifacts,
    )

    cfg = load_expanded_config(config_path)
    teachers, arbiter, reason_model = _build_teachers_from_config(cfg)

    try:
        evaluator = TI2IMultiModelEvaluator(
            teachers=teachers,
            arbiter=arbiter,
            reason_model=reason_model,
            config=dict(cfg.get("eval") or {}),
        )

        seed_path = Path(cfg["dataset"]["input_path"])
        samples = load_dataset_rows(str(seed_path))
        for sample in samples:
            _resolve_image_paths(sample, ["before_image", "after_image"], seed_path.parent)
            refs = sample.get("reference_images") or []
            sample["reference_images"] = [
                str((seed_path.parent / str(r)).resolve())
                if r and not str(r).startswith(("http://", "https://", "/"))
                else str(r)
                for r in refs
            ]

        results = evaluator.run(samples)
        summary = evaluator.aggregate(results)
        _write_artifacts(_output_dir(cfg), results, summary)
        logger.info("TI2I multi-model eval saved %d cases.", len(results))
    finally:
        close_backends(*teachers.values(), arbiter, reason_model)
