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

"""CLI argument parsing and job dispatch."""

import argparse
import logging
from typing import Callable, Dict

from dotenv import load_dotenv

from easydistill.models.zoo import format_model_table
from easydistill.utils import load_expanded_config

from .runners import (
    run_advanced_cot_distill,
    run_advanced_instruct_distill,
    run_advanced_mm_cot_distill,
    run_advanced_mm_distill,
    run_advanced_t2i_distill,
    run_advanced_t2v_distill,
    run_agent_distill,
    run_augmented_instruct_distill,
    run_balanced_instruct_distill,
    run_cot_eval,
    run_cot_generation,
    run_cot_long2short,
    run_cot_short2long,
    run_distill,
    run_dpo_data_build,
    run_instruct_eval,
    run_mm_cot_eval,
    run_mm_cot_generation,
    run_mm_cot_long2short,
    run_mm_cot_short2long,
    run_mm_generation,
    run_mm_instruct_eval,
    run_pe_rewrite_build_sft,
    run_pe_rewrite_distill,
    run_pe_rewrite_eval,
    run_pe_rewrite_filter,
    run_prompt_optimize,
    run_search_agent_distill,
    run_synthesis,
    run_t2i_distill,
    run_t2i_eval,
    run_t2i_generation,
    run_t2i_multi_model_eval,
    run_t2i_single_model_eval,
    run_t2v_distill,
    run_t2v_eval,
    run_t2v_generation,
    run_t2v_prompt_optimize,
    run_ti2i_multi_model_eval,
    run_ti2i_single_model_eval,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure root logging once when the CLI entry point runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Dispatch table for all supported job_type values.
_JOB_DISPATCH: Dict[str, Callable[[str], None]] = {
    "instruct_distill": run_distill,
    "instruction_expansion": lambda path: run_synthesis(path, "instruction_expansion"),
    "seed_anchored_expansion": lambda path: run_synthesis(path, "seed_anchored_expansion"),
    "agentic_rewrite": lambda path: run_synthesis(path, "agentic_rewrite"),
    "instruction_refinement": lambda path: run_synthesis(path, "instruction_refinement"),
    "instruction_response_extraction": lambda path: run_synthesis(
        path, "instruction_response_extraction"
    ),
    "augmented_instruct_distill": run_augmented_instruct_distill,
    "instruct_eval": run_instruct_eval,
    "pe_rewrite_eval": run_pe_rewrite_eval,
    "pe_rewrite_filter": run_pe_rewrite_filter,
    "pe_rewrite_build_sft": run_pe_rewrite_build_sft,
    "pe_rewrite_distill": run_pe_rewrite_distill,
    "advanced_instruct_distill": run_advanced_instruct_distill,
    "balanced_instruct_distill": run_balanced_instruct_distill,
    "cot_distill": run_cot_generation,
    "cot_long2short": run_cot_long2short,
    "cot_short2long": run_cot_short2long,
    "cot_eval": run_cot_eval,
    "advanced_cot_distill": run_advanced_cot_distill,
    "dpo_data_build": run_dpo_data_build,
    "mm_instruct_distill": run_mm_generation,
    "mm_cot_distill": run_mm_cot_generation,
    "mm_cot_long2short": run_mm_cot_long2short,
    "mm_cot_short2long": run_mm_cot_short2long,
    "mm_instruct_eval": run_mm_instruct_eval,
    "mm_cot_eval": run_mm_cot_eval,
    "advanced_mm_distill": run_advanced_mm_distill,
    "advanced_mm_cot_distill": run_advanced_mm_cot_distill,
    "t2i_distill": run_t2i_distill,
    "prompt_optimize": run_prompt_optimize,
    "t2i_generation": run_t2i_generation,
    "t2i_single_model_eval": run_t2i_single_model_eval,
    "t2i_multi_model_eval": run_t2i_multi_model_eval,
    "ti2i_single_model_eval": run_ti2i_single_model_eval,
    "ti2i_multi_model_eval": run_ti2i_multi_model_eval,
    "t2i_eval": run_t2i_eval,
    "advanced_t2i_distill": run_advanced_t2i_distill,
    "t2v_distill": run_t2v_distill,
    "t2v_prompt_optimize": run_t2v_prompt_optimize,
    "t2v_generation": run_t2v_generation,
    "t2v_eval": run_t2v_eval,
    "advanced_t2v_distill": run_advanced_t2v_distill,
    "agent_distill": run_agent_distill,
    "search_agent_distill": run_search_agent_distill,
}

# Human-readable descriptions for CLI discovery.
_JOB_DESCRIPTIONS: Dict[str, str] = {
    "instruct_distill": "Generate teacher responses for seed instructions and build SFT data.",
    "instruction_expansion": "Synthesize new instructions from seed examples.",
    "seed_anchored_expansion": (
        "Expand each seed into same-scenario instructions with topic dedup and lineage."
    ),
    "agentic_rewrite": ("Rewrite prompts via a plan -> rewrite -> reflection teacher agent chain."),
    "instruction_refinement": "Rewrite and improve existing instructions.",
    "instruction_response_extraction": "Extract instruction/response pairs from raw text.",
    "augmented_instruct_distill": (
        "Refine, generate multiple responses, evaluate, filter, and build SFT data."
    ),
    "instruct_eval": "Run LLM-as-judge evaluation on instruction/response pairs.",
    "pe_rewrite_eval": (
        "Score prompt rewrites with a multi-dimension LLM judge (PE rewrite pipeline)."
    ),
    "pe_rewrite_filter": (
        "Filter judged rewrites by score thresholds and top-ratio (local, no LLM)."
    ),
    "pe_rewrite_build_sft": (
        "Build SFT samples from filtered rewrites with per-language student "
        "system prompts (local, no LLM)."
    ),
    "pe_rewrite_distill": (
        "End-to-end PE rewrite text pipeline: rewrite -> judge -> filter -> SFT."
    ),
    "advanced_instruct_distill": (
        "End-to-end pipeline: expansion, generation, evaluation, filter, SFT."
    ),
    "balanced_instruct_distill": (
        "Expand instructions, balance categories, generate responses, build SFT data."
    ),
    "cot_distill": "Generate chain-of-thought reasoning traces and build SFT data.",
    "cot_long2short": "Simplify existing CoT reasoning traces.",
    "cot_short2long": "Extend existing CoT reasoning traces with more detail.",
    "cot_eval": "Run LLM-as-judge evaluation on CoT reasoning traces.",
    "advanced_cot_distill": "End-to-end CoT pipeline with RV/CD scoring and curriculum mixing.",
    "dpo_data_build": "Generate candidates, score them, and build DPO preference data.",
    "mm_instruct_distill": ("Generate teacher responses for multi-modal instruction/image pairs."),
    "mm_cot_distill": "Generate multi-modal chain-of-thought reasoning traces.",
    "mm_cot_long2short": "Simplify multi-modal CoT reasoning traces.",
    "mm_cot_short2long": "Extend multi-modal CoT reasoning traces.",
    "mm_instruct_eval": ("Run LLM-as-judge evaluation on multi-modal instruction responses."),
    "mm_cot_eval": "Run LLM-as-judge evaluation on multi-modal CoT traces.",
    "advanced_mm_distill": "End-to-end multi-modal instruction distillation pipeline.",
    "advanced_mm_cot_distill": "End-to-end multi-modal CoT distillation pipeline.",
    "t2i_distill": (
        "Basic T2I distillation: generate images from seed prompts and build SFT data."
    ),
    "prompt_optimize": "Optimize seed T2I prompts into rich, descriptive prompts via LLM.",
    "t2i_generation": "Generate images from prompts via T2I backend (no SFT building).",
    "t2i_single_model_eval": "Run single-teacher T2I evaluation with a dimension pool judge.",
    "t2i_multi_model_eval": "Run multi-teacher T2I evaluation with cross-model debate.",
    "ti2i_single_model_eval": "Run single-teacher TI2I evaluation with a dimension pool judge.",
    "ti2i_multi_model_eval": "Run multi-teacher TI2I evaluation with cross-model debate.",
    "t2i_eval": "Run VLM-as-judge evaluation on generated images.",
    "advanced_t2i_distill": (
        "End-to-end T2I pipeline: prompt optimization, T2I generation, "
        "VLM evaluation, quality filtering, multi-modal SFT."
    ),
    "t2v_distill": (
        "Basic T2V/I2V distillation: generate videos from seed prompts "
        "(optionally with first-frame images) and build SFT data."
    ),
    "t2v_prompt_optimize": (
        "Two-stage T2V/I2V prompt optimization: extract a structured draft, "
        "then compose it into the target model's caption schema."
    ),
    "t2v_generation": "Generate videos from prompts via T2V backend (no SFT building).",
    "t2v_eval": (
        "Run T2V video evaluation: precheck, frame-based VLM-as-judge, "
        "optional omni consistency check."
    ),
    "advanced_t2v_distill": (
        "End-to-end T2V/I2V pipeline: prompt optimization, video generation, "
        "video evaluation, quality filtering, multi-modal SFT."
    ),
    "agent_distill": (
        "Synthesize virtual tool-use tasks and agent trajectories, "
        "then build SFT or DPO training data."
    ),
    "search_agent_distill": (
        "Evolve seed QA into multi-hop search tasks, roll out search-agent "
        "trajectories, and build SFT training data."
    ),
}


def _list_jobs() -> None:
    """Print all supported job types and exit."""
    print("Supported job_type values:")
    for job_type in sorted(_JOB_DISPATCH):
        desc = _JOB_DESCRIPTIONS.get(job_type, "")
        print(f"  {job_type:<36} {desc}")


def _list_models() -> None:
    """Print all models in the Model Zoo and exit."""
    print(format_model_table())


def main() -> None:
    # Load secrets from .env (if present); falls back to shell env vars.
    # This lets users run `easydistill --config xxx.yaml` without manually
    # exporting DASHSCOPE_API_KEY / EAS_TOKEN etc. in the shell.
    load_dotenv()
    _configure_logging()
    parser = argparse.ArgumentParser(description="EasyDistill 2 CLI")
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config file.")
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        dest="list_jobs",
        help="List all supported job_type values and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        dest="list_models",
        help="List all models in the Model Zoo and exit.",
    )
    args = parser.parse_args()

    if args.list_jobs:
        _list_jobs()
        return

    if args.list_models:
        _list_models()
        return

    if not args.config:
        parser.error("--config is required unless --list-jobs or --list-models is used.")

    cfg = load_expanded_config(args.config)
    job_type = cfg.get("job_type", "instruct_distill")

    runner = _JOB_DISPATCH.get(job_type)
    if runner is None:
        valid_jobs = ", ".join(sorted(_JOB_DISPATCH))
        raise ValueError(f"Unsupported job_type: {job_type}. Valid values: {valid_jobs}")
    runner(args.config)


if __name__ == "__main__":
    main()
