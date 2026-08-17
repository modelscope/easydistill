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

"""Multi-modal operator runners."""

import logging
from typing import Any, List

from easydistill.data.models import SFTSample
from easydistill.eval import MMCoTEvaluator, MMInstructionFollowingEvaluator
from easydistill.operators.mm import (
    MMCoTGenerationOperator,
    MMGenerationOperator,
)
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.rewrite import (
    MMCoTLong2ShortOperator,
    MMCoTShort2LongOperator,
)
from easydistill.utils import load_expanded_config, save_jsonl

from ..backend_factory import build_backend, check_backend_health, close_backends
from ..data_loaders import (
    load_multimodal_eval_samples,
    load_multimodal_inputs,
    load_multimodal_problem_answer_pairs,
)
from .synthesis import _save_synthesis_outputs

logger = logging.getLogger(__name__)


def run_mm_generation(config_path: str) -> None:
    """Run multi-modal teacher response generation -> SFT dataset."""
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        mm_cfg = cfg.get("mm", {})
        inputs = load_multimodal_inputs(cfg)
        operator = MMGenerationOperator(backend=backend, config=mm_cfg)
        sft_samples: List[SFTSample] = operator.run_to_sft(
            inputs, sft_config=cfg.get("dataset", {})
        )

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, [sample.model_dump() for sample in sft_samples])
        logger.info("Saved multi-modal SFT dataset to %s.", output_path)
    finally:
        close_backends(backend)


def _run_mm_cot_operator(config_path: str, job_type: str) -> None:
    """Run an MMCoT operator and save outputs."""
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        cot_cfg = cfg.get("cot", {})
        output_path = cfg["dataset"]["output_path"]
        output_format = cfg["dataset"].get("output_format", "cot")

        operator: PromptGenerationOperator[Any, Any]
        if job_type == "mm_cot_distill":
            inputs = load_multimodal_inputs(cfg)
            operator = MMCoTGenerationOperator(backend=backend, config=cot_cfg)
            sft_samples: List[SFTSample] = operator.run_to_sft(
                inputs, sft_config=cfg.get("dataset", {})
            )
            save_jsonl(output_path, [sample.model_dump() for sample in sft_samples])
            logger.info("Saved multi-modal CoT SFT dataset to %s.", output_path)
            return

        if job_type == "mm_cot_long2short":
            pairs = load_multimodal_problem_answer_pairs(cfg)
            operator = MMCoTLong2ShortOperator(backend=backend, config=cot_cfg)
            outputs = operator.run(pairs)
        elif job_type == "mm_cot_short2long":
            pairs = load_multimodal_problem_answer_pairs(cfg)
            operator = MMCoTShort2LongOperator(backend=backend, config=cot_cfg)
            outputs = operator.run(pairs)
        else:
            raise ValueError(f"Unknown MMCoT job_type: {job_type}")

        _save_synthesis_outputs(output_path, outputs, output_format)
    finally:
        close_backends(backend)


def run_mm_cot_generation(config_path: str) -> None:
    """Run multi-modal CoT generation."""
    _run_mm_cot_operator(config_path, "mm_cot_distill")


def run_mm_cot_long2short(config_path: str) -> None:
    """Run multi-modal CoT long-to-short simplification."""
    _run_mm_cot_operator(config_path, "mm_cot_long2short")


def run_mm_cot_short2long(config_path: str) -> None:
    """Run multi-modal CoT short-to-long extension."""
    _run_mm_cot_operator(config_path, "mm_cot_short2long")


def _run_mm_instruct_eval(config_path: str, evaluator_cls) -> None:
    """Run multi-modal evaluation with the given evaluator class."""
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        eval_cfg = cfg.get("eval", {})
        samples = load_multimodal_eval_samples(cfg)
        evaluator = evaluator_cls(backend=backend, config=eval_cfg)
        results = evaluator.run(samples)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved MM evaluation results to %s.", output_path)
    finally:
        close_backends(backend)


def run_mm_instruct_eval(config_path: str) -> None:
    """Run multi-modal instruction-following evaluation."""
    _run_mm_instruct_eval(config_path, MMInstructionFollowingEvaluator)


def run_mm_cot_eval(config_path: str) -> None:
    """Run multi-modal CoT evaluation."""
    _run_mm_instruct_eval(config_path, MMCoTEvaluator)
