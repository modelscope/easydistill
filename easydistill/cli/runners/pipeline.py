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

"""Best-practice pipeline runners."""

import logging

from easydistill.pipeline import (
    AdvancedInstructDistillPipeline,
    AgentDistillationPipeline,
    AugmentedInstructDistillPipeline,
    BalancedInstructDistillPipeline,
    CoTDistillationPipeline,
    MMCoTDistillationPipeline,
    MMDistillationPipeline,
    PERewriteDistillPipeline,
    SearchAgentDistillationPipeline,
)
from easydistill.utils import load_expanded_config

from ..backend_factory import build_backend, check_backend_health, close_backends

logger = logging.getLogger(__name__)


def run_agent_distill(config_path: str) -> None:
    """Run agent distillation pipeline for synthetic tasks and trajectories."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = AgentDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
            agent_config=cfg.get("agent", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_search_agent_distill(config_path: str) -> None:
    """Run search-agent pipeline: task evolution -> trajectories -> SFT."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = SearchAgentDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
            search_agent_config=cfg.get("search_agent", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_pe_rewrite_distill(config_path: str) -> None:
    """Run the PE rewrite text pipeline: rewrite -> judge -> filter -> SFT."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = PERewriteDistillPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_advanced_instruct_distill(config_path: str) -> None:
    """Run advanced instruction distillation pipeline."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = AdvancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_advanced_cot_distill(config_path: str) -> None:
    """Run advanced CoT distillation pipeline: generate -> eval -> filter -> SFT."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = CoTDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_balanced_instruct_distill(config_path: str) -> None:
    """Run balanced instruction distillation pipeline."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = BalancedInstructDistillPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_augmented_instruct_distill(config_path: str) -> None:
    """Run augmented instruction distillation pipeline."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = AugmentedInstructDistillPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_advanced_mm_distill(config_path: str) -> None:
    """Run advanced multi-modal instruction distillation pipeline."""
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = MMDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)


def run_advanced_mm_cot_distill(config_path: str) -> None:
    """Run advanced multi-modal CoT distillation pipeline."""
    cfg = load_expanded_config(config_path)
    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        pipeline = MMCoTDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
            eval_config=cfg.get("eval", {}),
        )
        pipeline.run()
    finally:
        close_backends(backend)
