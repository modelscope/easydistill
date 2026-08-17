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

"""Simple I/O utilities."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def safe_filename_stem(raw_id: str) -> str:
    """Sanitize an arbitrary identifier so it is safe to use as a filename stem.

    Replaces any character outside ``[a-zA-Z0-9_-]`` with ``_`` to prevent
    path traversal or directory separators in identifiers returned by external
    services from being interpreted as filesystem paths.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw_id))


def load_jsonl(path: str, strict: bool = False) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    Raises FileNotFoundError if the file does not exist. By default, malformed
    lines are logged and skipped. Pass ``strict=True`` to raise instead.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(
                        f"Malformed JSONL line {line_number} in {path}: {exc}"
                    ) from exc
                logger.warning(
                    "Skipping malformed JSONL line %d in %s: %s", line_number, path, exc
                )
    return rows


def load_dataset_rows(path: str, strict: bool = False) -> List[Dict[str, Any]]:
    """Load a JSONL dataset and return its rows, raising if empty."""
    rows = load_jsonl(path, strict=strict)
    if not rows:
        raise ValueError(f"No data found in {path}")
    return rows


def save_jsonl(path: str, data: Iterable[Any]) -> None:
    """Save an iterable of objects to a JSONL file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_alpaca(
    input_path: str,
    output_path: str,
    include_system_prompt: bool = True,
) -> int:
    """Convert a ShareGPT messages JSONL file to Alpaca format.

    Each output line contains only two fields:
        {"instruction": "...", "output": "..."}

    If *include_system_prompt* is True and a system message exists, it is
    prepended to the instruction field (separated by a blank line).

    Returns the number of converted samples.
    """
    rows = load_jsonl(input_path)
    samples = []
    for row in rows:
        messages = row.get("messages", [])
        system_content = ""
        user_content = ""
        assistant_content = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content
        if include_system_prompt and system_content:
            instruction = f"{system_content}\n\n{user_content}"
        else:
            instruction = user_content
        samples.append({"instruction": instruction, "output": assistant_content})
    save_jsonl(output_path, samples)
    logger.info("Converted %d samples to Alpaca format -> %s", len(samples), output_path)
    return len(samples)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    """Save an object to a JSON file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
