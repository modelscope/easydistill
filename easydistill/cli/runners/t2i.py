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

"""T2I (text-to-image) distillation runners."""

import logging
from typing import Any, Dict, List

from easydistill.data.models import SFTSample
from easydistill.eval import T2IImageEvaluator
from easydistill.operators.t2i import (
    T2IGenerationOperator,
    T2IPromptOptimizer,
    T2ISFTBuilder,
)
from easydistill.pipeline import T2IDistillationPipeline
from easydistill.utils import load_dataset_rows, load_expanded_config, save_jsonl

from ..backend_factory import (
    build_backend,
    build_t2i_backend,
    check_backend_health,
    close_backends,
)
from ..data_loaders import load_t2i_seed_prompts

logger = logging.getLogger(__name__)


def run_t2i_distill(config_path: str) -> None:
    """Run basic T2I distillation: seed prompts -> images -> SFT dataset.

    This is the simplest T2I flow — no prompt optimization, no evaluation.
    Seed prompts are sent directly to the T2I backend, and the resulting
    images are packaged into multi-modal SFT samples.
    """
    cfg = load_expanded_config(config_path)
    t2i_backend = build_t2i_backend(cfg["t2i_backend"])
    try:
        check_backend_health(t2i_backend)

        gen_cfg = cfg.get("generation", {})
        sft_cfg = cfg.get("sft", {})

        # Load seed prompts.
        rows = load_t2i_seed_prompts(cfg)

        # Generate images.
        gen_operator = T2IGenerationOperator(backend=t2i_backend, config=gen_cfg)
        generated_rows = gen_operator.run(rows)

        # Build SFT samples.
        builder = T2ISFTBuilder(config=sft_cfg)
        sft_samples: List[SFTSample] = builder.run(generated_rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, [sample.model_dump() for sample in sft_samples])
        logger.info("Saved T2I SFT dataset (%d samples) to %s.", len(sft_samples), output_path)
    finally:
        close_backends(t2i_backend)


def run_prompt_optimize(config_path: str) -> None:
    """Run standalone T2I prompt optimization.

    Uses an LLM/VLM to rewrite simple seed prompts into rich, descriptive
    prompts suitable for text-to-image models.
    """
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        prompt_cfg = cfg.get("prompt_optimize", cfg.get("generation", {}))

        # Load seed prompts.
        rows = load_t2i_seed_prompts(cfg)

        # Optimize prompts.
        optimizer = T2IPromptOptimizer(backend=backend, config=prompt_cfg)
        optimized_rows = optimizer.run(rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, optimized_rows)
        logger.info("Saved %d optimized prompts to %s.", len(optimized_rows), output_path)
    finally:
        close_backends(backend)


def run_t2i_generation(config_path: str) -> None:
    """Run standalone T2I image generation.

    Sends prompts (seed or pre-optimized) to the T2I backend and saves the
    generated image URLs.  No SFT building is performed.
    """
    cfg = load_expanded_config(config_path)
    t2i_backend = build_t2i_backend(cfg["t2i_backend"])
    try:
        check_backend_health(t2i_backend)

        gen_cfg = cfg.get("generation", {})

        # Load input prompts (seed or optimized).
        rows = load_t2i_seed_prompts(cfg)

        # Generate images.
        gen_operator = T2IGenerationOperator(backend=t2i_backend, config=gen_cfg)
        generated_rows = gen_operator.run(rows)

        output_path = cfg["dataset"]["output_path"]
        save_jsonl(output_path, generated_rows)
        logger.info("Saved %d T2I generation results to %s.", len(generated_rows), output_path)
    finally:
        close_backends(t2i_backend)


def run_t2i_eval(config_path: str) -> None:
    """Run standalone T2I image evaluation.

    Loads rows containing optimized prompts and image URLs, then scores them
    with a VLM-as-judge evaluator across multiple quality dimensions.

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

        # Load evaluation data (rows with optimized_prompt and image_urls).
        samples: List[Dict[str, Any]] = load_dataset_rows(input_path)
        logger.info("Loaded %d T2I eval samples from %s.", len(samples), input_path)

        evaluator = T2IImageEvaluator(backend=backend, config=eval_cfg)
        results = evaluator.run(samples)

        output_path = dataset_cfg["output_path"]
        save_jsonl(output_path, results)
        logger.info("Saved T2I evaluation results to %s.", output_path)
    finally:
        close_backends(backend)


def run_advanced_t2i_distill(config_path: str) -> None:
    """Run end-to-end T2I distillation pipeline.

    Full flow: prompt optimization -> T2I generation -> VLM evaluation ->
    quality filtering -> multi-modal SFT dataset building.

    Uses ``backend`` (text model) for prompt optimization and
    ``eval_backend`` (VLM) for image evaluation.  If ``eval_backend`` is
    not configured, ``backend`` is used for both.
    """
    cfg = load_expanded_config(config_path)

    backends: List[Any] = []
    try:
        backend = build_backend(cfg["backend"])
        backends.append(backend)
        check_backend_health(backend)

        t2i_backend = build_t2i_backend(cfg["t2i_backend"])
        backends.append(t2i_backend)
        check_backend_health(t2i_backend)

        # Use a separate VLM backend for evaluation if configured;
        # otherwise the main backend is used for both optimization and eval.
        eval_backend = None
        if cfg.get("eval_backend"):
            eval_backend = build_backend(cfg["eval_backend"])
            backends.append(eval_backend)
            check_backend_health(eval_backend)

        pipeline = T2IDistillationPipeline(
            backend=backend,
            t2i_backend=t2i_backend,
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
