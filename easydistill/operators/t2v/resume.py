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

"""Opt-in resume helpers for the T2V/I2V pipeline.

Video generation is slow and expensive, so T2V stages support resuming
from their own ``output_path``: rows that are already present and complete
in a previous (possibly partial) output file are reused instead of re-run.
Rows are matched by their ``id`` when present, falling back to a content
hash of the seed prompt and first-frame image.

A row only counts as *complete* when it passes the stage's completion
predicate (e.g. non-empty ``video_urls`` for generation), so rows that
failed in the previous run are retried automatically.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

Row = Dict[str, Any]


def resume_key(row: Row) -> str:
    """Stable identity of a row across pipeline runs.

    Prefers the seed ``id``; rows without one are keyed by a hash of the
    original prompt and first-frame image, both of which are carried
    through every stage unchanged.
    """
    row_id = row.get("id")
    if row_id not in (None, ""):
        return f"id:{row_id}"
    basis = f"{row.get('prompt') or ''}\x1f{row.get('first_frame_image') or ''}"
    return "sha1:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()


def optimize_row_complete(row: Row) -> bool:
    """A prompt_optimize row is complete when it carries an optimized prompt."""
    return bool(row.get("optimized_prompt"))


def generate_row_complete(row: Row) -> bool:
    """A t2v_generate row is complete when at least one video was produced."""
    return bool(row.get("video_urls"))


def eval_row_complete(row: Row) -> bool:
    """A t2v_eval row is complete when any checker actually scored it.

    VLM/omni checkers emit ``<dim>_confidence`` companions and VBench emits
    ``vbench_``-prefixed metric columns; rows the evaluator skipped only carry
    ``None`` metric values (plus an optional skip reason) and therefore get
    re-evaluated on resume.
    """
    for key, value in row.items():
        if value is None:
            continue
        if key.endswith("_confidence"):
            return True
        if key.startswith("vbench_") and key != "vbench_skipped_reason":
            return True
    return False


def load_completed_rows(path: str, is_complete: Callable[[Row], bool]) -> Dict[str, Row]:
    """Index the complete rows of a previous stage output by resume key.

    Tolerates a torn trailing line (crash mid-append) and duplicate keys
    (append-mode checkpoints), keeping the last complete occurrence.
    """
    completed: Dict[str, Row] = {}
    if not path or not os.path.exists(path):
        return completed
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and is_complete(row):
                    completed[resume_key(row)] = row
    except OSError as exc:
        logger.warning("Could not read resume file %s: %s", path, exc)
    return completed


def split_pending(data: List[Row], completed: Dict[str, Row]) -> Tuple[List[Row], List[Row]]:
    """Split input rows into (reusable, pending) against completed keys."""
    done: List[Row] = []
    pending: List[Row] = []
    for row in data:
        (done if resume_key(row) in completed else pending).append(row)
    return done, pending


def merge_resumed(
    data: List[Row],
    completed: Dict[str, Row],
    new_rows: List[Row],
) -> List[Row]:
    """Merge reused and freshly produced rows back into input order.

    Input rows present in neither map were dropped by the stage itself
    (e.g. generation failed for them), matching non-resume behaviour.
    """
    fresh = {resume_key(row): row for row in new_rows}
    merged: List[Row] = []
    for row in data:
        key = resume_key(row)
        if key in completed:
            merged.append(completed[key])
        elif key in fresh:
            merged.append(fresh[key])
    return merged


class RowCheckpointWriter:
    """Thread-safe per-row JSONL appender for mid-stage crash recovery.

    Each completed row is appended and fsynced immediately so a crashed
    run loses at most the row in flight; ``load_completed_rows`` dedupes
    on the next resume.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, row: Row) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:  # pragma: no cover - disk-level failure
                logger.warning("Failed to append checkpoint row to %s: %s", self.path, exc)
