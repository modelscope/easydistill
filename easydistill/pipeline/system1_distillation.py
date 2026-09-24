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

"""System-1 distillation pipeline: typed decisions -> RLCD training cases.

Four stages, run in order:

  1. build_cases   traffic rows + schema -> protocol-validated case rows
                   (questions + gold where labels exist)
  2. elicit        teacher probability elicitation for every question
  3. aggregate     elicitation samples -> per-question teacher labels
  4. build_dataset cases + teacher labels -> RLCD training-ready JSONL

The top-level ``system1`` config provides defaults for every stage; each
stage can override them in its own ``config`` block. The ``labeled``
variant skips elicit/aggregate by design (gold passes straight through),
so its stage list is ``build_cases`` -> ``build_dataset``. Only the elicit
stage talks to the teacher backend.
"""

import logging
from typing import Any, Dict, List, Optional

from easydistill.backends.base import ModelBackend
from easydistill.operators.system1 import (
    System1AggregateOperator,
    System1BuildCasesOperator,
    System1BuildDatasetOperator,
    System1ElicitOperator,
)

from .base import BaseDistillationPipeline

logger = logging.getLogger(__name__)


def _run_build_cases_stage(
    system1_config: Dict[str, Any],
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate rows and build typed-decision case rows."""
    cfg = {**system1_config, **stage_config}
    return System1BuildCasesOperator(config=cfg).run(data)


def _run_elicit_stage(
    backend: ModelBackend,
    system1_config: Dict[str, Any],
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Elicit teacher distributions for every case question."""
    cfg = {**system1_config, **stage_config}
    return System1ElicitOperator(backend=backend, config=cfg).run(data)


def _run_aggregate_stage(
    system1_config: Dict[str, Any],
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate elicitation samples into teacher labels."""
    cfg = {**system1_config, **stage_config}
    return System1AggregateOperator(config=cfg).run(data)


def _run_build_dataset_stage(
    system1_config: Dict[str, Any],
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Emit RLCD training-ready case rows with final targets."""
    cfg = {**system1_config, **stage_config}
    return System1BuildDatasetOperator(config=cfg).run(data)


class System1DistillationPipeline(BaseDistillationPipeline):
    """End-to-end pipeline building RLCD training cases from typed decisions.

    Fields in the top-level ``system1`` config act as defaults for every
    stage; each stage can override them in its own ``config`` block. Only
    the elicit stage talks to the teacher backend.

    Recommended stage flow:
      1. build_cases
      2. elicit
      3. aggregate
      4. build_dataset
    """

    _last_stage = "build_dataset"

    def __init__(
        self,
        backend: ModelBackend,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        system1_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            backend=backend,
            pipeline_config=pipeline_config,
            dataset_config=dataset_config,
            generation_config=generation_config,
            sft_config=sft_config,
        )
        self.system1_config = system1_config or {}

    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        # Top-level `system1` fields act as defaults; stage `config` overrides.
        if stage_name == "build_cases":
            return _run_build_cases_stage(self.system1_config, stage_config, data)
        elif stage_name == "elicit":
            return _run_elicit_stage(
                self.backend, self.system1_config, stage_config, data
            )
        elif stage_name == "aggregate":
            return _run_aggregate_stage(self.system1_config, stage_config, data)
        elif stage_name == "build_dataset":
            return _run_build_dataset_stage(self.system1_config, stage_config, data)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
