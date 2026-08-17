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

"""Dataset loading helpers for CLI runners."""

import logging
from typing import Any, Dict, List, Tuple

from easydistill.data.models import GenerationRequest
from easydistill.utils import load_dataset_rows
from easydistill.utils.image import _extract_text_from_content

logger = logging.getLogger(__name__)


def load_requests(config: Dict[str, Any]) -> List[GenerationRequest]:
    """Load seed instructions from JSONL as GenerationRequest objects."""
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    instruction_key = dataset_cfg.get("instruction_key") or "instruction"
    system_key = dataset_cfg.get("system_key") or "system"

    rows = load_dataset_rows(input_path)

    requests = []
    for idx, row in enumerate(rows):
        instruction = row.get(instruction_key)
        if not instruction:
            logger.warning("Row %d missing '%s', skipping.", idx, instruction_key)
            continue
        requests.append(
            GenerationRequest(
                id=str(row.get("id", idx)),
                instruction=instruction if isinstance(instruction, list) else str(instruction),
                system_prompt=row.get(system_key)
                or config.get("generation", {}).get("system_prompt"),
                metadata={
                    k: v for k, v in row.items() if k not in {instruction_key, system_key, "id"}
                },
            )
        )
    logger.info("Loaded %d seed instructions from %s.", len(requests), input_path)
    return requests


def load_string_column(config: Dict[str, Any], column_key: str) -> List[str]:
    """Load a list of strings from a JSONL column.

    The default column name is inferred from ``column_key``:
      - ``instruction_key`` -> ``instruction``
      - ``text_key`` -> ``text``
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    default_key = "instruction" if column_key == "instruction_key" else "text"
    key = dataset_cfg.get(column_key) or default_key
    rows = load_dataset_rows(input_path)
    values = []
    for idx, row in enumerate(rows):
        value = row.get(key)
        if not value:
            logger.warning("Row %d missing '%s', skipping.", idx, key)
            continue
        values.append(_extract_text_from_content(value))
    if not values:
        raise ValueError(f"No values found for key '{key}' in {input_path}")
    logger.info("Loaded %d values for key '%s' from %s.", len(values), key, input_path)
    return values


def load_seed_records(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load seed rows as ``{"id", "instruction"}`` dicts, preserving ids.

    Used by operators that track lineage back to the source seed (e.g.
    seed-anchored expansion). The row index is used when a row has no ``id``.
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    key = dataset_cfg.get("instruction_key") or "instruction"
    rows = load_dataset_rows(input_path)
    records = []
    for idx, row in enumerate(rows):
        value = row.get(key)
        if not value:
            logger.warning("Row %d missing '%s', skipping.", idx, key)
            continue
        records.append(
            {"id": str(row.get("id", idx)), "instruction": _extract_text_from_content(value)}
        )
    if not records:
        raise ValueError(f"No values found for key '{key}' in {input_path}")
    logger.info("Loaded %d seed records from %s.", len(records), input_path)
    return records


def load_instruction_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load full rows as dicts with a normalized ``instruction`` field.

    Unlike :func:`load_seed_records`, all extra fields (e.g. expansion lineage
    like ``seed_id`` / ``round``) are preserved so downstream operators can pass
    them through to their outputs.
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    key = dataset_cfg.get("instruction_key") or "instruction"
    rows = load_dataset_rows(input_path)
    records = []
    for idx, row in enumerate(rows):
        value = row.get(key)
        if not value:
            logger.warning("Row %d missing '%s', skipping.", idx, key)
            continue
        record = {k: v for k, v in row.items() if k != key}
        record["instruction"] = _extract_text_from_content(value)
        record.setdefault("id", str(idx))
        records.append(record)
    if not records:
        raise ValueError(f"No values found for key '{key}' in {input_path}")
    logger.info("Loaded %d instruction rows from %s.", len(records), input_path)
    return records


def load_problem_column(config: Dict[str, Any]) -> List[str]:
    """Load problem strings from a JSONL column.

    Uses `problem_key` from the dataset config, falling back to `instruction`.
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    key = dataset_cfg.get("problem_key") or "problem"
    rows = load_dataset_rows(input_path)
    values = []
    for idx, row in enumerate(rows):
        value = row.get(key) or row.get("instruction")
        if not value:
            logger.warning("Row %d missing '%s' or 'instruction', skipping.", idx, key)
            continue
        values.append(_extract_text_from_content(value))
    if not values:
        raise ValueError(f"No problems found for key '{key}' in {input_path}")
    logger.info("Loaded %d problems from %s.", len(values), input_path)
    return values


def load_problem_answer_pairs(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Load (problem, answer) pairs from a JSONL file.

    Uses `problem_key` (default 'instruction') and `answer_key` (default 'response'),
    with fallbacks to common field names. Both problem and answer are normalized
    to plain text, so pre-built multi-modal content lists are accepted but their
    image content is dropped (text-only CoT operators do not consume images).
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    problem_key = dataset_cfg.get("problem_key") or "instruction"
    answer_key = dataset_cfg.get("answer_key") or "response"
    rows = load_dataset_rows(input_path)
    pairs = []
    for idx, row in enumerate(rows):
        problem = row.get(problem_key) or row.get("problem") or row.get("instruction")
        answer = row.get(answer_key) or row.get("answer") or row.get("output")
        if not problem or not answer:
            logger.warning("Row %d missing problem/answer, skipping.", idx)
            continue
        pairs.append((_extract_text_from_content(problem), _extract_text_from_content(answer)))
    if not pairs:
        raise ValueError(f"No problem/answer pairs found in {input_path}")
    logger.info("Loaded %d problem/answer pairs from %s.", len(pairs), input_path)
    return pairs


def load_eval_samples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load evaluation samples from JSONL.

    Supports two formats:
      - Plain: {"instruction": "...", "output": "..."} or {"response": "..."}
      - SFT messages: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    instruction_key = dataset_cfg.get("instruction_key") or "instruction"
    output_key = dataset_cfg.get("output_key") or "output"
    rows = load_dataset_rows(input_path)

    samples = []
    for idx, row in enumerate(rows):
        instruction = row.get(instruction_key) or row.get("input") or row.get("problem")
        output = row.get(output_key) or row.get("response") or row.get("answer")

        if not instruction and "messages" in row:
            messages = row["messages"]
            for msg in messages:
                if msg.get("role") == "user":
                    instruction = msg.get("content")
                elif msg.get("role") == "assistant":
                    output = msg.get("content")

        if not instruction or not output:
            logger.warning("Row %d missing instruction/output, skipping.", idx)
            continue

        sample = {"id": str(row.get("id", idx)), "instruction": instruction, "output": output}
        if row.get("images"):
            sample["images"] = row["images"]
        samples.append(sample)

    logger.info("Loaded %d evaluation samples from %s.", len(samples), input_path)
    return samples


def load_multimodal_inputs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load multi-modal instruction rows from JSONL.

    Expected format: {"instruction": "...", "images": ["path_or_url", ...]}
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    instruction_key = dataset_cfg.get("instruction_key") or "instruction"
    images_key = dataset_cfg.get("images_key") or "images"
    rows = load_dataset_rows(input_path)

    inputs = []
    for idx, row in enumerate(rows):
        instruction = row.get(instruction_key)
        images = row.get(images_key) or []
        if not instruction:
            logger.warning("Row %d missing '%s', skipping.", idx, instruction_key)
            continue
        if isinstance(images, str):
            images = [images]
        item = {
            "instruction": instruction if isinstance(instruction, list) else str(instruction),
            "images": images,
        }
        for k, v in row.items():
            if k not in {instruction_key, images_key, "id"}:
                item[k] = v
        inputs.append(item)

    logger.info("Loaded %d multi-modal inputs from %s.", len(inputs), input_path)
    return inputs


def load_multimodal_problem_answer_pairs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load multi-modal (problem, answer) rows from JSONL.

    Supported formats:
      - Plain: {"instruction": "...", "images": [...], "response": "..."}
      - SFT messages: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
                       "metadata": {"images": [...], ...}}
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    problem_key = dataset_cfg.get("problem_key") or "instruction"
    answer_key = dataset_cfg.get("answer_key") or "response"
    images_key = dataset_cfg.get("images_key") or "images"
    rows = load_dataset_rows(input_path)

    pairs = []
    for idx, row in enumerate(rows):
        problem = row.get(problem_key) or row.get("problem") or row.get("instruction")
        answer = row.get(answer_key) or row.get("answer") or row.get("output")
        images = row.get(images_key)

        # Fall back to SFT messages format.
        if (not problem or not answer) and "messages" in row:
            user_content = None
            assistant_content = None
            for msg in row["messages"]:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    user_content = content
                elif role == "assistant":
                    assistant_content = content
            problem = problem or user_content
            answer = answer or assistant_content

        # Images may live in metadata when reading SFT output from a previous stage.
        if not images and "metadata" in row and isinstance(row["metadata"], dict):
            images = row["metadata"].get(images_key) or row["metadata"].get("images")

        if not problem or not answer:
            logger.warning("Row %d missing problem/answer, skipping.", idx)
            continue
        if isinstance(images, str):
            images = [images]
        item = {
            "instruction": problem if isinstance(problem, list) else str(problem),
            "images": images or [],
            "response": answer if isinstance(answer, list) else str(answer),
        }
        # If the answer is a pre-built multi-modal content list, preserve it so
        # MMCoT operators can forward images alongside the response.
        for k, v in row.items():
            if k not in {problem_key, answer_key, images_key, "id", "messages", "metadata"}:
                item[k] = v
        pairs.append(item)

    logger.info("Loaded %d multi-modal problem/answer pairs from %s.", len(pairs), input_path)
    return pairs


def load_multimodal_eval_samples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load multi-modal evaluation samples from JSONL.

    Expected format:
      {"instruction": "...", "images": [...], "output": "..."}
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    instruction_key = dataset_cfg.get("instruction_key") or "instruction"
    output_key = dataset_cfg.get("output_key") or "output"
    images_key = dataset_cfg.get("images_key") or "images"
    rows = load_dataset_rows(input_path)

    samples = []
    for idx, row in enumerate(rows):
        instruction = row.get(instruction_key) or row.get("input") or row.get("problem")
        output = row.get(output_key) or row.get("response") or row.get("answer")

        if not instruction and "messages" in row:
            messages = row["messages"]
            for msg in messages:
                if msg.get("role") == "user":
                    instruction = msg.get("content")
                elif msg.get("role") == "assistant":
                    output = msg.get("content")

        if not instruction or not output:
            logger.warning("Row %d missing instruction/output, skipping.", idx)
            continue

        images = row.get(images_key)
        if images and isinstance(images, str):
            images = [images]

        sample = {
            "id": str(row.get("id", idx)),
            "instruction": instruction,
            "output": output,
        }
        if images:
            sample["images"] = images
        samples.append(sample)

    logger.info("Loaded %d multi-modal evaluation samples from %s.", len(samples), input_path)
    return samples


def load_t2i_seed_prompts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load seed T2I prompts from JSONL.

    Expected format: {"prompt": "..."} or {"instruction": "..."}.
    The ``prompt_key`` in dataset config overrides the default key name.
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    prompt_key = dataset_cfg.get("prompt_key") or "prompt"
    rows = load_dataset_rows(input_path)

    inputs = []
    for idx, row in enumerate(rows):
        prompt = row.get(prompt_key) or row.get("instruction") or ""
        if not prompt:
            logger.warning("Row %d missing '%s', skipping.", idx, prompt_key)
            continue
        item = {
            "id": str(row.get("id", idx)),
            "prompt": str(prompt),
        }
        for k, v in row.items():
            if k not in {prompt_key, "id"}:
                item[k] = v
        inputs.append(item)

    logger.info("Loaded %d seed T2I prompts from %s.", len(inputs), input_path)
    return inputs


def load_t2v_seed_prompts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load seed T2V/I2V prompts from JSONL.

    Expected format: {"prompt": "..."} for T2V rows, plus an optional
    {"first_frame_image": "url_or_path"} for I2V rows.  Both kinds may be
    mixed in one file.  The ``prompt_key`` / ``first_frame_key`` entries in
    the dataset config override the default key names.
    """
    dataset_cfg = config["dataset"]
    input_path = dataset_cfg["input_path"]
    prompt_key = dataset_cfg.get("prompt_key") or "prompt"
    first_frame_key = dataset_cfg.get("first_frame_key") or "first_frame_image"
    rows = load_dataset_rows(input_path)

    inputs = []
    for idx, row in enumerate(rows):
        prompt = row.get(prompt_key) or row.get("instruction") or ""
        if not prompt:
            logger.warning("Row %d missing '%s', skipping.", idx, prompt_key)
            continue
        item = {
            "id": str(row.get("id", idx)),
            "prompt": str(prompt),
        }
        first_frame = row.get(first_frame_key)
        if first_frame:
            item["first_frame_image"] = str(first_frame)
        for k, v in row.items():
            if k not in {prompt_key, first_frame_key, "id"}:
                item[k] = v
        inputs.append(item)

    n_i2v = sum(1 for item in inputs if item.get("first_frame_image"))
    logger.info(
        "Loaded %d seed T2V prompts from %s (%d I2V rows with first frame).",
        len(inputs),
        input_path,
        n_i2v,
    )
    return inputs
