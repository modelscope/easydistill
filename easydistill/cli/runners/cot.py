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

"""Chain-of-thought operator runners."""

import logging
from typing import Any, List

from easydistill.data.models import SFTSample
from easydistill.operators import CoTGenerationOperator
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.rewrite import (
    CoTLong2ShortOperator,
    CoTShort2LongOperator,
)
from easydistill.utils import load_expanded_config, save_jsonl

from ..backend_factory import build_backend, check_backend_health, close_backends
from ..data_loaders import load_problem_answer_pairs, load_problem_column
from .synthesis import _save_synthesis_outputs

logger = logging.getLogger(__name__)


def _run_cot_operator(config_path: str, job_type: str) -> None:
    """Run a CoT operator and save outputs."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        cot_cfg = cfg.get("cot", {})
        output_path = cfg["dataset"]["output_path"]
        output_format = cfg["dataset"].get("output_format", "cot")

        operator: PromptGenerationOperator[Any, Any]
        if job_type == "cot_distill":
            problems = load_problem_column(cfg)
            operator = CoTGenerationOperator(backend=backend, config=cot_cfg)
            sft_samples: List[SFTSample] = operator.run_to_sft(
                problems, sft_config=cfg.get("dataset", {})
            )
            save_jsonl(output_path, [sample.model_dump() for sample in sft_samples])
            logger.info("Saved CoT SFT dataset to %s.", output_path)
            return

        if job_type == "cot_long2short":
            pairs = load_problem_answer_pairs(cfg)
            operator = CoTLong2ShortOperator(backend=backend, config=cot_cfg)
            outputs = operator.run(pairs)
        elif job_type == "cot_short2long":
            pairs = load_problem_answer_pairs(cfg)
            operator = CoTShort2LongOperator(backend=backend, config=cot_cfg)
            outputs = operator.run(pairs)
        else:
            raise ValueError(f"Unknown CoT job_type: {job_type}")

        _save_synthesis_outputs(output_path, outputs, output_format)
    finally:
        close_backends(backend)


def run_cot_generation(config_path: str) -> None:
    """Run CoT generation: problems -> reasoning + solution."""
    _run_cot_operator(config_path, "cot_distill")


def run_cot_long2short(config_path: str) -> None:
    """Run CoT long-to-short simplification."""
    _run_cot_operator(config_path, "cot_long2short")


def run_cot_short2long(config_path: str) -> None:
    """Run CoT short-to-long extension."""
    _run_cot_operator(config_path, "cot_short2long")
