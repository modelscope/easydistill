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

"""Shared helpers for advanced distillation pipelines."""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators import InstructionBalancer, SFTDatasetBuilder, TextGenerationOperator
from easydistill.operators.preference import PreferenceDatasetBuilder
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.rewrite import (
    InstructionExpansionOperator,
    InstructionRefinementOperator,
    InstructionResponseExtractionOperator,
)
from easydistill.utils import build_multimodal_user_content, save_jsonl
from easydistill.utils.image import _extract_text_from_content
from easydistill.utils.metrics import compute_average_score

logger = logging.getLogger(__name__)


_SYNTHESIS_OPERATORS: Dict[str, Type[PromptGenerationOperator[Any, Any]]] = {
    "instruction_expansion": InstructionExpansionOperator,
    "instruction_refinement": InstructionRefinementOperator,
    "instruction_response_extraction": InstructionResponseExtractionOperator,
}


def _extract_strings(data: List[Dict[str, Any]], key: str) -> List[str]:
    values = []
    for row in data:
        value = row.get(key)
        if value:
            values.append(_extract_text_from_content(value))
    return values


def _format_synthesis_outputs(outputs: List[Any], stage_name: str) -> List[Dict[str, Any]]:
    rows = []
    if stage_name == "instruction_response_extraction":
        for instruction, response in outputs:
            rows.append({"instruction": instruction, "response": response})
    else:
        for instruction in outputs:
            rows.append({"instruction": instruction})
    return rows


def _run_synthesis_stage(
    backend: Any,
    stage_name: str,
    stage_config: Dict[str, Any],
    inputs: List[str],
) -> List[Any]:
    op_cls = _SYNTHESIS_OPERATORS[stage_name]
    operator = op_cls(backend=backend, config=stage_config)
    return operator.run(inputs)


def _run_instruction_balance_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify instructions by task/domain and resample to a target distribution."""
    balancer = InstructionBalancer(backend=backend, config=stage_config)
    return balancer.run(data)


def run_eval_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    evaluator_cls: Type[Any],
    image_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run an LLM-as-judge evaluation stage and merge scores back into the rows.

    Parameters
    ----------
    evaluator_cls:
        Evaluator class to instantiate (e.g., ``InstructionFollowingEvaluator``).
    image_key:
        Optional row key carrying multi-modal image URLs; when present, the
        images are attached to the eval sample under the same key.
    """
    evaluator = evaluator_cls(backend=backend, config=stage_config)
    eval_samples = []
    skipped = 0
    for idx, row in enumerate(data):
        if not row.get("instruction") or not row.get("response"):
            skipped += 1
            continue
        sample = {
            "id": str(row.get("id", idx)),
            "instruction": row["instruction"],
            "output": row["response"],
        }
        if image_key and row.get(image_key):
            sample[image_key] = row[image_key]
        eval_samples.append(sample)

    if skipped:
        pct = 100.0 * skipped / len(data) if data else 0.0
        log_level = logging.ERROR if pct > 50 else logging.WARNING
        logger.log(
            log_level,
            "run_eval_stage skipped %d of %d rows (%.1f%%) with missing "
            "instruction or response.",
            skipped,
            len(data),
            pct,
        )

    eval_results = evaluator.run(eval_samples)
    scores_by_id = {r["id"]: r for r in eval_results}

    output_rows = []
    for idx, row in enumerate(data):
        row_id = str(row.get("id", idx))
        scores = scores_by_id.get(row_id, {})
        new_row = dict(row)
        for metric in evaluator.metrics:
            new_row[metric] = scores.get(metric)
        output_rows.append(new_row)
    return output_rows


def _run_distill_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    global_generation_config: Optional[Dict[str, Any]] = None,
    global_sft_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run generation + SFT builder over instruction rows."""
    gen_cfg = {**(global_generation_config or {}), **stage_config.get("generation", {})}
    sft_cfg = {**(global_sft_config or {}), **stage_config.get("sft", {})}

    generator = TextGenerationOperator(backend=backend, config=gen_cfg)
    builder = SFTDatasetBuilder(config=sft_cfg)

    requests = []
    for idx, row in enumerate(data):
        instruction = row.get("instruction")
        if not instruction:
            continue
        requests.append(
            GenerationRequest(
                id=str(row.get("id", idx)),
                instruction=instruction if isinstance(instruction, list) else str(instruction),
                system_prompt=row.get("system") or gen_cfg.get("system_prompt"),
                metadata={
                    k: v
                    for k, v in row.items()
                    if k not in {"instruction", "system", "id", "response"}
                },
            )
        )

    results = generator.run(requests)
    samples = builder.run(results)
    return [sample.model_dump() for sample in samples]


def _save_intermediate_output(output_path: Optional[str], data: List[Dict[str, Any]]) -> None:
    if output_path:
        save_jsonl(output_path, data)
        logger.info("Saved intermediate output to %s.", output_path)


def run_quality_filter_stage(
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    eval_metrics: List[str],
) -> List[Dict[str, Any]]:
    """Filter rows by minimum scores and/or top-k/top-ratio selection."""
    min_scores = stage_config.get("min_scores", {})
    keep_top_k = stage_config.get("keep_top_k")
    keep_top_ratio = stage_config.get("keep_top_ratio")
    require_all_metrics = bool(stage_config.get("require_all_metrics", True))

    # First pass: minimum score filtering.
    filtered = []
    for row in data:
        skip = False
        for metric, threshold in min_scores.items():
            value = row.get(metric)
            if value is None:
                if require_all_metrics:
                    skip = True
                    break
                continue
            try:
                if float(value) < float(threshold):
                    skip = True
                    break
            except (TypeError, ValueError):
                # Non-numeric values are compared as booleans.
                if bool(value) != bool(threshold):
                    skip = True
                    break
        if not skip:
            filtered.append(row)

    # Second pass: top-k / top-ratio selection by average score.
    if keep_top_k is not None or keep_top_ratio is not None:
        if not eval_metrics:
            logger.warning(
                "quality_filter stage uses keep_top_k/keep_top_ratio but no "
                "eval_metrics are configured; skipping top-k/ratio selection."
            )
        else:
            scored = []
            for row in filtered:
                avg = compute_average_score(row, eval_metrics)
                if avg is not None:
                    scored.append((avg, row))
            scored.sort(key=lambda x: x[0], reverse=True)

            if keep_top_k is not None:
                filtered = [row for _, row in scored[: int(keep_top_k)]]
            elif keep_top_ratio is not None:
                k = max(0, int(len(scored) * float(keep_top_ratio)))
                filtered = [row for _, row in scored[:k]]

    logger.info(
        "Quality filter kept %d of %d rows.",
        len(filtered),
        len(data),
    )
    return filtered


def run_build_sft_stage(
    data: List[Dict[str, Any]],
    global_generation_config: Optional[Dict[str, Any]] = None,
    global_sft_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert instruction/response rows into SFT message format.

    By default the response is read from the ``response`` field, with a
    fallback to ``output`` (the field name used by PAI SFT datasets).  Set
    ``sft.response_key`` to a single field name or a list of field names to
    override this.

    Multi-modal image references are read from ``images`` by default; set
    ``sft.images_key`` to override.
    """
    sft_cfg = global_sft_config or {}
    builder = SFTDatasetBuilder(config=sft_cfg)

    response_keys = sft_cfg.get("response_key") or ["response", "output"]
    if isinstance(response_keys, str):
        response_keys = [response_keys]
    images_key = sft_cfg.get("images_key") or "images"
    exclude_keys = {"instruction", "system", "id", images_key, *response_keys}

    results = []
    for idx, row in enumerate(data):
        instruction = row.get("instruction")
        response = next((row.get(k) for k in response_keys if k in row), None)
        if not instruction or not response:
            continue
        system_prompt = row.get("system")
        if not system_prompt and global_generation_config:
            system_prompt = global_generation_config.get("system_prompt")
        images = row.get(images_key)
        instruction_content: Union[str, List[Dict[str, Any]]]
        if images:
            if isinstance(images, str):
                images = [images]
            instruction_content = build_multimodal_user_content(instruction, images)
        else:
            instruction_content = instruction if isinstance(instruction, list) else str(instruction)
        request = GenerationRequest(
            id=str(row.get("id", idx)),
            instruction=instruction_content,
            system_prompt=system_prompt,
            metadata={k: v for k, v in row.items() if k not in exclude_keys},
        )
        results.append(
            GenerationResult(
                request=request,
                response=str(response),
                model="pipeline",
            )
        )

    samples = builder.run(results)
    return [sample.model_dump() for sample in samples]


def run_build_preference_dataset_stage(
    data: List[Dict[str, Any]],
    stage_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert chosen/rejected rows into a preference dataset."""
    builder = PreferenceDatasetBuilder(config=stage_config)
    return builder.run(data)


def run_t2i_eval_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    evaluator_cls: Type[Any],
) -> List[Dict[str, Any]]:
    """Run a T2I image evaluation stage and merge scores back into rows.

    Each row must contain ``optimized_prompt`` (or ``prompt``) and
    ``image_urls``.  The evaluator sends the prompt text and the first image
    to a VLM judge, which returns a score per configured metric.
    """
    evaluator = evaluator_cls(backend=backend, config=stage_config)
    eval_samples = []
    skipped = 0
    for idx, row in enumerate(data):
        prompt = row.get("optimized_prompt") or row.get("prompt")
        image_urls = row.get("image_urls") or []
        if not prompt or not image_urls:
            skipped += 1
            continue
        sample = {
            "id": str(row.get("id", idx)),
            "optimized_prompt": prompt,
            "image_urls": image_urls,
        }
        eval_samples.append(sample)

    if skipped:
        pct = 100.0 * skipped / len(data) if data else 0.0
        log_level = logging.ERROR if pct > 50 else logging.WARNING
        logger.log(
            log_level,
            "run_t2i_eval_stage skipped %d of %d rows (%.1f%%) with missing "
            "prompt or image_urls.",
            skipped,
            len(data),
            pct,
        )

    eval_results = evaluator.run(eval_samples)
    scores_by_id = {r["id"]: r for r in eval_results}

    output_rows = []
    for idx, row in enumerate(data):
        row_id = str(row.get("id", idx))
        scores = scores_by_id.get(row_id, {})
        new_row = dict(row)
        for metric in evaluator.metrics:
            new_row[metric] = scores.get(metric)
        output_rows.append(new_row)
    return output_rows


def run_build_t2i_sft_stage(
    data: List[Dict[str, Any]],
    sft_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert T2I prompt/image rows into multi-modal SFT samples."""
    from easydistill.operators.t2i import T2ISFTBuilder

    builder = T2ISFTBuilder(config=sft_config or {})
    samples = builder.run(data)
    return [sample.model_dump() for sample in samples]


def run_t2v_eval_stage(
    backend: Any,
    stage_config: Dict[str, Any],
    data: List[Dict[str, Any]],
    evaluator_cls: Type[Any],
) -> List[Dict[str, Any]]:
    """Run a T2V video evaluation stage and merge scores back into rows.

    Each row must contain ``optimized_prompt`` (or ``prompt``) and
    ``video_urls``.  The evaluator samples frames from the first video and
    sends them with the prompt to a VLM judge, which returns a score per
    configured metric.  I2V rows additionally carry ``first_frame_image``.
    """
    evaluator = evaluator_cls(backend=backend, config=stage_config)
    eval_samples = []
    skipped = 0
    for idx, row in enumerate(data):
        prompt = row.get("optimized_prompt") or row.get("prompt")
        video_urls = row.get("video_urls") or []
        if not prompt or not video_urls:
            skipped += 1
            continue
        sample = {
            "id": str(row.get("id", idx)),
            "optimized_prompt": prompt,
            "video_urls": video_urls,
        }
        if row.get("frame_urls"):
            sample["frame_urls"] = row["frame_urls"]
        if row.get("first_frame_image"):
            sample["first_frame_image"] = row["first_frame_image"]
        eval_samples.append(sample)

    if skipped:
        pct = 100.0 * skipped / len(data) if data else 0.0
        log_level = logging.ERROR if pct > 50 else logging.WARNING
        logger.log(
            log_level,
            "run_t2v_eval_stage skipped %d of %d rows (%.1f%%) with missing "
            "prompt or video_urls.",
            skipped,
            len(data),
            pct,
        )

    eval_results = evaluator.run(eval_samples)
    scores_by_id = {r["id"]: r for r in eval_results}
    identity_keys = {
        "id",
        "optimized_prompt",
        "prompt",
        "video_urls",
        "frame_urls",
        "first_frame_image",
    }

    output_rows = []
    for idx, row in enumerate(data):
        row_id = str(row.get("id", idx))
        scores = scores_by_id.get(row_id, {})
        new_row = dict(row)
        for metric in evaluator.metrics:
            new_row[metric] = scores.get(metric)
        # Carry companion columns too (e.g. <metric>_confidence / _reason).
        for key, value in scores.items():
            if key not in identity_keys and key not in new_row:
                new_row[key] = value
        output_rows.append(new_row)
    return output_rows


def run_build_t2v_sft_stage(
    data: List[Dict[str, Any]],
    sft_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert T2V prompt/video rows into multi-modal SFT samples."""
    from easydistill.operators.t2v import T2VSFTBuilder

    builder = T2VSFTBuilder(config=sft_config or {})
    samples = builder.run(data)
    return [sample.model_dump() for sample in samples]
