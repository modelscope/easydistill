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

"""LLM-as-judge evaluation runners."""

import logging

from easydistill.eval import CoTEvaluator, InstructionFollowingEvaluator, PERewriteEvaluator
from easydistill.utils import load_dataset_rows, load_expanded_config, save_jsonl

from ..backend_factory import build_backend, check_backend_health, close_backends
from ..data_loaders import load_eval_samples

logger = logging.getLogger(__name__)


def run_cot_eval(config_path: str) -> None:
    """Run LLM-as-judge CoT evaluation."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        samples = load_eval_samples(cfg)
        evaluator = CoTEvaluator(backend=backend, config=cfg.get("eval", {}))
        results = evaluator.run(samples)
        aggregates = evaluator.aggregate(results)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved per-sample CoT evaluation results to %s.", output_path)
        logger.info("Aggregate CoT scores: %s", aggregates)
    finally:
        close_backends(backend)


def run_instruct_eval(config_path: str) -> None:
    """Run LLM-as-judge instruction-following evaluation."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        samples = load_eval_samples(cfg)
        evaluator = InstructionFollowingEvaluator(backend=backend, config=cfg.get("eval", {}))
        results = evaluator.run(samples)
        aggregates = evaluator.aggregate(results)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved per-sample evaluation results to %s.", output_path)
        logger.info("Aggregate scores: %s", aggregates)
    finally:
        close_backends(backend)


def run_pe_rewrite_eval(config_path: str) -> None:
    """Run the PE rewrite multi-dimension judge.

    Rows are loaded verbatim (not via load_eval_samples) because the evaluator
    passes scene/lineage/trace fields through to the scored output.
    """
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        rows = load_dataset_rows(cfg["dataset"]["input_path"])
        evaluator = PERewriteEvaluator(backend=backend, config=cfg.get("eval", {}))
        results = evaluator.run(rows)
        aggregates = evaluator.aggregate(results)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved per-sample PE rewrite scores to %s.", output_path)
        logger.info("Aggregate PE rewrite scores: %s", aggregates)
    finally:
        close_backends(backend)
