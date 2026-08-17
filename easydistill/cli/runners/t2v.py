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

"""T2V/I2V (text/image-to-video) distillation runners."""

import logging
from typing import Any, Dict, List

from easydistill.data.models import SFTSample
from easydistill.eval import T2VVideoEvaluator
from easydistill.operators.t2v import (
    T2VGenerationOperator,
    T2VPromptOptimizer,
    T2VSFTBuilder,
)
from easydistill.pipeline import T2VDistillationPipeline
from easydistill.utils import load_dataset_rows, load_expanded_config, save_jsonl

from ..backend_factory import (
    build_backend,
    build_t2v_backend,
    check_backend_health,
    close_backends,
)
from ..data_loaders import load_t2v_seed_prompts

logger = logging.getLogger(__name__)


def run_t2v_distill(config_path: str) -> None:
    """Run basic T2V distillation: seed prompts -> videos -> SFT dataset.

    This is the simplest T2V flow — no prompt optimization, no evaluation.
    Seed prompts (with optional first-frame images for I2V rows) are sent
    directly to the T2V backend, and the resulting videos are packaged into
    multi-modal SFT samples.
    """
    cfg = load_expanded_config(config_path)
    t2v_backend = build_t2v_backend(cfg["t2v_backend"])
    try:
        check_backend_health(t2v_backend)

        gen_cfg = cfg.get("generation", {})
        sft_cfg = cfg.get("sft", {})

        # Load seed prompts (T2V and I2V rows may be mixed).
        rows = load_t2v_seed_prompts(cfg)

        # Generate videos.
        gen_operator = T2VGenerationOperator(backend=t2v_backend, config=gen_cfg)
        generated_rows = gen_operator.run(rows)

        # Build SFT samples.
        builder = T2VSFTBuilder(config=sft_cfg)
        sft_samples: List[SFTSample] = builder.run(generated_rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, [sample.model_dump() for sample in sft_samples])
        logger.info("Saved T2V SFT dataset (%d samples) to %s.", len(sft_samples), output_path)
    finally:
        close_backends(t2v_backend)


def run_t2v_prompt_optimize(config_path: str) -> None:
    """Run standalone two-stage T2V/I2V prompt optimization.

    Exactly two model calls per row: extract (generic video parsing into a
    structured draft; I2V rows are grounded in their first-frame image) and
    compose (rewriting the draft into the target video model's caption
    schema, provided via ``prompt_optimize.schema_file``).
    """
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        prompt_cfg = cfg.get("prompt_optimize", cfg.get("generation", {}))

        # Load seed prompts.
        rows = load_t2v_seed_prompts(cfg)

        # Optimize prompts.
        optimizer = T2VPromptOptimizer(backend=backend, config=prompt_cfg)
        optimized_rows = optimizer.run(rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, optimized_rows)
        logger.info("Saved %d optimized prompts to %s.", len(optimized_rows), output_path)
    finally:
        close_backends(backend)


def run_t2v_generation(config_path: str) -> None:
    """Run standalone T2V/I2V video generation.

    Sends prompts (seed or pre-optimized) to the T2V backend and saves the
    generated video URLs.  No SFT building is performed.
    """
    cfg = load_expanded_config(config_path)
    t2v_backend = build_t2v_backend(cfg["t2v_backend"])
    try:
        check_backend_health(t2v_backend)

        gen_cfg = cfg.get("generation", {})

        # Load input prompts (seed or optimized).
        rows = load_t2v_seed_prompts(cfg)

        # Generate videos.
        gen_operator = T2VGenerationOperator(backend=t2v_backend, config=gen_cfg)
        generated_rows = gen_operator.run(rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, generated_rows)
        logger.info("Saved %d T2V generation results to %s.", len(generated_rows), output_path)
    finally:
        close_backends(t2v_backend)


def run_t2v_eval(config_path: str) -> None:
    """Run standalone T2V video evaluation.

    Loads rows containing optimized prompts and video URLs, then scores them
    with the T2V evaluation chain (precheck -> frame-based VLM-as-judge ->
    optional omni check).

    Uses ``eval_backend`` if configured (for VLM), otherwise falls back to
    ``backend``.
    """
    cfg = load_expanded_config(config_path)

    # Use eval_backend (VLM) if configured, otherwise fall back to backend.
    eval_backend_cfg = cfg.get("eval_backend", cfg["backend"])
    backend = build_backend(eval_backend_cfg)
    try:
        check_backend_health(backend)

        eval_cfg = cfg.get("eval", {})
        dataset_cfg = cfg["dataset"]
        input_path = dataset_cfg["input_path"]

        # Load evaluation data (rows with optimized_prompt and video_urls).
        samples: List[Dict[str, Any]] = load_dataset_rows(input_path)
        logger.info("Loaded %d T2V eval samples from %s.", len(samples), input_path)

        evaluator = T2VVideoEvaluator(backend=backend, config=eval_cfg)
        results = evaluator.run(samples)

        output_path = dataset_cfg["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved T2V evaluation results to %s.", output_path)
    finally:
        close_backends(backend)


def run_advanced_t2v_distill(config_path: str) -> None:
    """Run end-to-end T2V/I2V distillation pipeline.

    Full flow: prompt optimization -> T2V generation -> video evaluation ->
    quality filtering -> multi-modal SFT dataset building.

    Uses ``backend`` (text/VLM model) for prompt optimization and
    ``eval_backend`` (VLM) for video evaluation.  If ``eval_backend`` is
    not configured, ``backend`` is used for both.
    """
    cfg = load_expanded_config(config_path)

    backends: List[Any] = []
    try:
        backend = build_backend(cfg["backend"])
        backends.append(backend)
        check_backend_health(backend)

        t2v_backend = build_t2v_backend(cfg["t2v_backend"])
        backends.append(t2v_backend)
        check_backend_health(t2v_backend)

        # Use a separate VLM backend for evaluation if configured;
        # otherwise the main backend is used for both optimization and eval.
        eval_backend = None
        if cfg.get("eval_backend"):
            eval_backend = build_backend(cfg["eval_backend"])
            backends.append(eval_backend)
            check_backend_health(eval_backend)

        pipeline = T2VDistillationPipeline(
            backend=backend,
            t2v_backend=t2v_backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
            eval_backend=eval_backend,
        )
        pipeline.run()
    finally:
        close_backends(*backends)
