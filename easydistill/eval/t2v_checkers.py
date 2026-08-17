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

"""Composable video-evaluation checkers for the T2V/I2V eval chain.

Video evaluation is heterogeneous: some checks are objective local model
inference (VBench dimensions), some judge sampled frames with an image VLM,
and some need a video-native (Omni-style) understanding model.  Each
mechanism is its own :class:`BaseVideoChecker` subclass, and
:class:`~easydistill.eval.t2v.T2VVideoEvaluator` composes an arbitrary list
of checkers from config.

The three built-in checkers form a cost ladder with disjoint metric columns:

1. :class:`VBenchChecker` — objective per-video scores via the official
   Vchitect/VBench repo (subprocess; needs the repo + checkpoints + a GPU).
   Columns are prefixed ``vbench_``, raw 0-1 floats.
2. :class:`VLMChecker` — frames sampled from the video are scored by an
   image VLM judge in ONE structured multi-dimension call per row; judges
   only what sparse frames reliably show (per-frame content, cross-frame
   identity).  Raw 0-4 integer scores.
3. :class:`OmniChecker` — a video-native model consumes the complete video
   for the dynamic dimensions frames cannot show (motion quality, temporal
   execution, camera accuracy).  Raw 0-4 integer scores.

Adding a new evaluation = subclass ``BaseVideoChecker`` + one registry
entry.  Each checker contributes its own raw metric columns (no cross-source
normalization by design); a failing checker only blanks its own columns and
never kills the batch.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from easydistill.prompts import resolve_prompt
from easydistill.utils import (
    build_multimodal_user_content,
    format_prompt_safely,
    load_config,
    load_video_to_data_url,
    progress,
    sample_video_frames,
)

from ._common import clamp_score, parse_json_block

logger = logging.getLogger(__name__)


class BaseVideoChecker(ABC):
    """Base class for one composable video-evaluation step.

    Subclasses declare which metric columns they produce (``metrics``) and
    implement ``check(rows)``, returning the rows with those columns merged
    in.  Rows are plain dicts carrying at least ``id``, a prompt field and
    ``video_urls``.

    Common configurable fields:
      - enabled: set False to skip this checker (default True).
    """

    name = "base_video_checker"
    #: Whether this checker needs a ModelBackend (VLM / video model).
    requires_backend = False

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ):
        self.config = config or {}
        self.backend: Any = backend
        enabled = self.config.get("enabled")
        self.enabled = bool(enabled) if enabled is not None else True
        if self.requires_backend and self.enabled and backend is None:
            raise ValueError(f"Checker '{self.name}' requires a backend.")

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        """Metric column names this checker writes into each row."""
        raise NotImplementedError

    @abstractmethod
    def check(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all rows and return them with this checker's metrics merged."""
        raise NotImplementedError

    def _blank(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return rows with this checker's metrics set to None (failure path)."""
        blanked = []
        for row in rows:
            new_row = dict(row)
            for metric in self.metrics:
                new_row.setdefault(metric, None)
            blanked.append(new_row)
        return blanked

    @staticmethod
    def _local_video_path(row: Dict[str, Any]) -> Optional[str]:
        """Return the row's first local video path, or None."""
        video_urls = row.get("video_urls") or []
        if isinstance(video_urls, str):
            video_urls = [video_urls]
        if not video_urls:
            return None
        ref = str(video_urls[0])
        if ref.startswith(("http://", "https://")):
            return None
        ref = ref[len("file://") :] if ref.startswith("file://") else ref
        return ref if os.path.isfile(ref) else None


# ---------------------------------------------------------------------------
# VBench objective evaluation
# ---------------------------------------------------------------------------

# Default subset: static screen (dynamic_degree), motion artifacts
# (motion_smoothness), per-frame quality (imaging_quality) plus the cheapest
# consistency check (temporal_flickering, pure pixel diff).
DEFAULT_VBENCH_DIMENSIONS = [
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "temporal_flickering",
]


def _parse_vbench_per_video(output_dir: Path) -> Dict[str, Dict[str, float]]:
    """Parse VBench output JSONs into ``{dimension: {video_stem: score}}``.

    Canonical format (``*_eval_results.json``):
      ``{dim: [mean, [{"video_path": ..., "video_results": float}, ...]]}``
    Legacy format: ``{dim: {"results": {video_path: score}}}``.
    """
    scores: Dict[str, Dict[str, float]] = {}
    if not output_dir.exists():
        return scores
    for json_file in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for dim, val in data.items():
            per_video: Dict[str, float] = {}
            if isinstance(val, list) and len(val) >= 2 and isinstance(val[1], list):
                for item in val[1]:
                    if not isinstance(item, dict):
                        continue
                    video_path = item.get("video_path")
                    result = item.get("video_results")
                    if video_path is not None and isinstance(result, (int, float, bool)):
                        per_video[Path(str(video_path)).stem] = float(result)
            elif isinstance(val, dict) and isinstance(val.get("results"), dict):
                for video_path, result in val["results"].items():
                    if video_path in ("mean", "average"):
                        continue
                    if isinstance(result, (int, float, bool)):
                        per_video[Path(str(video_path)).stem] = float(result)
            if per_video:
                scores.setdefault(dim, {}).update(per_video)
    return scores


class VBenchChecker(BaseVideoChecker):
    """Objective per-video scores via VBench (https://github.com/Vchitect/VBench).

    Runs VBench in ``custom_input`` mode over the batch's local video files,
    then maps the per-video scores back into the rows.  Two invocation modes:

    - **CLI mode** (recommended): ``vbench_bin`` points at the ``vbench``
      executable of a dedicated virtualenv where ``pip install vbench`` was
      run (VBench pins heavy deps like ``transformers==4.33.2``, so keep it
      isolated from the easydistill environment; it needs Python <= 3.11).
    - **Repo mode**: ``vbench_repo`` points at a local clone of the VBench
      repo and ``evaluate.py`` is run with ``python_executable``.

    ``vbench_bin`` wins when both are configured.  A GPU is needed for
    reasonable throughput on most dimensions (``temporal_flickering`` is the
    only near-free dimension and runs fine on CPU).

    When the environment is not suitable — neither mode configured, no GPU
    detected (``require_gpu``), or the subprocess fails — the checker skips
    the batch: it logs a warning, blanks its metric columns AND records the
    reason in the ``vbench_skipped_reason`` column so the skip is visible in
    the saved results, not just in the logs.

    Configurable fields:
      - dimensions: VBench dimension names (default: fast 4-dim subset).
      - vbench_bin: path to the pip-installed ``vbench`` CLI
        (or env VBENCH_BIN).
      - vbench_repo: path to the cloned VBench repo (or env VBENCH_REPO).
      - python_executable: interpreter for repo mode (default: current;
        point it at VBench's own conda/venv to keep its heavy deps isolated).
      - require_gpu: skip when nvidia-smi is absent (default True; set False
        to allow slow CPU runs).
      - num_gpus: >1 switches to torchrun (repo mode) / ``--ngpus`` (CLI mode).
      - load_ckpt_from_local: pass --load_ckpt_from_local True (default True).
      - timeout: subprocess timeout in seconds (default 3600).
      - extra_args: extra CLI args appended to the command.

    Metric columns are prefixed ``vbench_`` (raw VBench scores, mostly 0-1
    floats; dynamic_degree is 0/1).
    """

    name = "vbench"
    requires_backend = False
    SKIP_REASON_COLUMN = "vbench_skipped_reason"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ):
        super().__init__(config, backend)
        self.dimensions = list(self.config.get("dimensions") or DEFAULT_VBENCH_DIMENSIONS)
        self.vbench_bin = self.config.get("vbench_bin") or os.getenv("VBENCH_BIN")
        self.vbench_repo = self.config.get("vbench_repo") or os.getenv("VBENCH_REPO")
        self.python_executable = self.config.get("python_executable") or sys.executable
        require_gpu = self.config.get("require_gpu")
        self.require_gpu = bool(require_gpu) if require_gpu is not None else True
        self.num_gpus = int(self.config.get("num_gpus", 1))
        load_local = self.config.get("load_ckpt_from_local")
        self.load_ckpt_from_local = bool(load_local) if load_local is not None else True
        self.timeout = float(self.config.get("timeout", 3600.0))
        self.extra_args = list(self.config.get("extra_args") or [])

    @property
    def metrics(self) -> List[str]:
        return [f"vbench_{dim}" for dim in self.dimensions]

    def _environment_issue(self) -> Optional[str]:
        """Return a human-readable reason when the VBench env is unusable."""
        if self.vbench_bin:
            if not (os.path.isfile(self.vbench_bin) and os.access(self.vbench_bin, os.X_OK)):
                return f"vbench_bin {self.vbench_bin!r} is not an executable file"
        elif self.vbench_repo:
            if not (Path(self.vbench_repo) / "evaluate.py").is_file():
                return f"evaluate.py not found under vbench_repo {self.vbench_repo!r}"
        else:
            return (
                "neither vbench_bin nor vbench_repo is configured (set one in "
                "the checker config or via the VBENCH_BIN / VBENCH_REPO env vars)"
            )
        if self.require_gpu and shutil.which("nvidia-smi") is None:
            return (
                "no GPU detected (nvidia-smi not found); set require_gpu: false "
                "to allow a slow CPU run"
            )
        return None

    def _skip(self, rows: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        """Blank the metric columns and record the skip reason in the rows."""
        logger.warning(
            "VBenchChecker skipped for %d rows: %s.", len(rows), reason
        )
        skipped = []
        for row in self._blank(rows):
            row[self.SKIP_REASON_COLUMN] = reason
            skipped.append(row)
        return skipped

    def _build_command(self, videos_dir: str, output_dir: str) -> List[str]:
        if self.vbench_bin:
            cmd = [self.vbench_bin, "evaluate"]
            if self.num_gpus > 1:
                cmd.extend(["--ngpus", str(self.num_gpus)])
        elif self.vbench_repo:
            evaluate_py = str(Path(self.vbench_repo) / "evaluate.py")
            if self.num_gpus > 1:
                cmd = [
                    "torchrun",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc_per_node={self.num_gpus}",
                    evaluate_py,
                ]
            else:
                cmd = [self.python_executable, evaluate_py]
        else:
            # Unreachable: _environment_issue() gates check() on one of the two
            # being configured.
            raise ValueError(
                "neither vbench_bin nor vbench_repo is configured"
            )
        cmd.extend(
            [
                "--videos_path",
                videos_dir,
                "--dimension",
                *self.dimensions,
                "--mode",
                "custom_input",
                "--output_path",
                output_dir,
            ]
        )
        if self.load_ckpt_from_local:
            cmd.extend(["--load_ckpt_from_local", "True"])
        cmd.extend(str(arg) for arg in self.extra_args)
        return cmd

    def check(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issue = self._environment_issue()
        if issue:
            return self._skip(rows, issue)

        # Stage local videos under index-based names so per-video scores can
        # be mapped back to rows regardless of their original filenames.
        stem_to_index: Dict[str, int] = {}
        with tempfile.TemporaryDirectory(prefix="vbench_videos_") as videos_dir, \
                tempfile.TemporaryDirectory(prefix="vbench_out_") as output_dir:
            for idx, row in enumerate(rows):
                local_path = self._local_video_path(row)
                if not local_path:
                    logger.warning(
                        "VBenchChecker: row %s has no local video file, skipping.",
                        row.get("id", idx),
                    )
                    continue
                stem = f"video_{idx:05d}"
                staged = os.path.join(
                    videos_dir, stem + os.path.splitext(local_path)[1]
                )
                try:
                    os.symlink(os.path.abspath(local_path), staged)
                except OSError:
                    shutil.copyfile(local_path, staged)
                stem_to_index[stem] = idx

            if not stem_to_index:
                return self._skip(rows, "no local video files to evaluate")

            cmd = self._build_command(videos_dir, output_dir)
            # The pip CLI internally spawns a bare `python -m torch.distributed.run`;
            # prepend the venv's bin dir to PATH so it resolves to the venv
            # interpreter (where vbench is installed) instead of the system one.
            env = None
            if self.vbench_bin:
                env = dict(os.environ)
                env["PATH"] = (
                    os.path.dirname(os.path.abspath(self.vbench_bin))
                    + os.pathsep
                    + env.get("PATH", "")
                )
            logger.info(
                "VBenchChecker: evaluating %d videos on %s.",
                len(stem_to_index),
                ", ".join(self.dimensions),
            )
            try:
                completed = subprocess.run(  # noqa: S603 - command built from config
                    cmd,
                    # Repo mode runs from the repo root (evaluate.py expects
                    # its relative asset paths); CLI mode has no such need.
                    cwd=self.vbench_repo if not self.vbench_bin else None,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                return self._skip(rows, f"vbench subprocess failed: {exc}")
            if completed.returncode != 0:
                logger.error(
                    "VBenchChecker: vbench exited with %d. stderr tail:\n%s",
                    completed.returncode,
                    (completed.stderr or "")[-2000:],
                )
                return self._skip(
                    rows,
                    f"vbench exited with code {completed.returncode} "
                    "(see logs for stderr)",
                )

            per_dim = _parse_vbench_per_video(Path(output_dir))
            if not per_dim:
                # The pip CLI wraps torchrun and can exit 0 even when the
                # inner worker crashed (e.g. NCCL init on a GPU-less host),
                # so an empty result dir must not pass silently.
                logger.error(
                    "VBenchChecker: no per-video results produced. stderr tail:\n%s",
                    (completed.stderr or completed.stdout or "")[-2000:],
                )
                return self._skip(
                    rows,
                    "vbench produced no per-video results (see logs; note "
                    "VBench requires a GPU — its distributed init is "
                    "NCCL-only on Linux)",
                )

        merged = []
        for idx, row in enumerate(rows):
            new_row = dict(row)
            new_row[self.SKIP_REASON_COLUMN] = None
            for dim in self.dimensions:
                stem_scores = per_dim.get(dim, {})
                score = None
                for stem, row_idx in stem_to_index.items():
                    if row_idx == idx and stem in stem_scores:
                        score = stem_scores[stem]
                        break
                new_row[f"vbench_{dim}"] = score
            merged.append(new_row)
        return merged


# ---------------------------------------------------------------------------
# Structured judge base (shared by the VLM and Omni checkers)
# ---------------------------------------------------------------------------


class _StructuredJudgeChecker(BaseVideoChecker):
    """Shared machinery for model judges with structured multi-dim output.

    Subclasses define a dimension pool asset and a judge prompt template,
    plus ``_build_message_content(row, is_i2v)`` returning the multimodal
    user content for one row.  The base class handles dimension selection,
    per-row calls with retries, JSON parsing, column mapping, row-level
    concurrency and failure isolation.

    Common configurable fields:
      - metrics: dimension subset (default: all pool dimensions).
      - dimensions_file: dimension pool override.
      - judge_prompt_template / judge_prompt_template_file: prompt override.
      - backend: optional backend config dict; when present, a dedicated
        backend is built for this checker (e.g. the omni endpoint usually
        differs from the VLM judge endpoint).
      - max_workers / call_retries / retry_delay_sec / temperature /
        max_tokens / model_id / show_progress.

    Output columns per dimension ``d``: ``d`` (0-4 int score or None),
    ``d_confidence``, ``d_reason``.
    """

    requires_backend = True
    DEFAULT_DIMENSIONS_FILE = ""
    DEFAULT_JUDGE_PROMPT_FILE = ""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ):
        backend_config = (config or {}).get("backend")
        if isinstance(backend_config, dict):
            # Lazy import: checker-level backend override (e.g. a dedicated
            # omni endpoint) reuses the CLI backend factory.
            from easydistill.cli.backend_factory import build_backend

            backend = build_backend(backend_config)
        super().__init__(config, backend)
        self.prompt_template = resolve_prompt(
            self.config,
            template_key="judge_prompt_template",
            file_key="judge_prompt_template_file",
            default_file=self.DEFAULT_JUDGE_PROMPT_FILE,
        )
        dimensions_file = self.config.get("dimensions_file", self.DEFAULT_DIMENSIONS_FILE)
        pool: Dict[str, Any] = load_config(dimensions_file)
        selected = self.config.get("metrics")
        if selected:
            missing = [name for name in selected if name not in pool]
            if missing:
                raise ValueError(
                    f"Unknown {self.name} judge dimensions {missing}; "
                    f"available: {sorted(pool)}."
                )
            self.dimensions = {name: pool[name] for name in selected}
        else:
            self.dimensions = dict(pool)

        self.max_workers = int(self.config.get("max_workers", 4))
        self.call_retries = int(self.config.get("call_retries", 2))
        self.retry_delay_sec = float(self.config.get("retry_delay_sec", 5.0))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 2048))
        self.model_id = self.config.get("model_id")
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True

    @property
    def metrics(self) -> List[str]:
        return list(self.dimensions)

    def _build_dim_block(self, is_i2v: bool) -> str:
        """Render the dimension definitions for the judge prompt."""
        lines = []
        for name, spec in self.dimensions.items():
            spec = spec or {}
            criteria = spec.get("criteria") or {}
            criteria_text = "; ".join(f"{k}={v}" for k, v in criteria.items())
            note = ""
            if spec.get("applicable") == "i2v_only":
                note = (
                    " (APPLICABLE: a conditioning first frame is provided)"
                    if is_i2v
                    else " (NOT applicable: no conditioning first frame)"
                )
            lines.append(
                f"- {name}{note}: {spec.get('description', '')} "
                f"Criteria: {criteria_text}"
            )
        return "\n".join(lines)

    @abstractmethod
    def _build_message_content(self, row: Dict[str, Any], is_i2v: bool) -> Any:
        """Build the multimodal user content for one row's judge call."""
        raise NotImplementedError

    def _check_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Score one row; returns the metric columns to merge."""
        is_i2v = bool(row.get("first_frame_image"))
        content = self._build_message_content(row, is_i2v)

        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(1 + self.call_retries):
            try:
                result = self.backend.generate(
                    messages=[{"role": "user", "content": content}],
                    model_id=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                payload = parse_json_block(result.response)
                return self._columns_from_payload(payload)
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last_exc = exc
                if attempt < self.call_retries:
                    logger.warning(
                        "%s judge retry %d/%d for row %s after error: %s",
                        self.name,
                        attempt + 1,
                        self.call_retries,
                        row.get("id"),
                        exc,
                    )
                    time.sleep(self.retry_delay_sec)
        raise last_exc

    def _columns_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Map the judge's dimension_judgments into row columns."""
        columns: Dict[str, Any] = {}
        for metric in self.metrics:
            columns[metric] = None
            columns[f"{metric}_confidence"] = None
            columns[f"{metric}_reason"] = None
        for item in payload.get("dimension_judgments") or []:
            name = str(item.get("dimension") or "")
            if name not in self.dimensions:
                continue
            applicable = item.get("applicable")
            score = item.get("score")
            if applicable is False or score is None:
                columns[name] = None
            else:
                columns[name] = clamp_score(score)
            confidence = item.get("confidence")
            columns[f"{name}_confidence"] = (
                float(confidence) if confidence is not None else None
            )
            columns[f"{name}_reason"] = str(item.get("reason") or "") or None
        return columns

    def check(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Optional[Dict[str, Any]]] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._check_row, row): idx
                for idx, row in enumerate(rows)
            }
            futures_iter = progress(
                futures.items(),
                enabled=self.show_progress,
                total=len(futures),
                desc=f"{self.name} judging videos",
            )
            for future, idx in futures_iter:
                new_row = dict(rows[idx])
                try:
                    new_row.update(future.result())
                except Exception as exc:  # noqa: BLE001 - isolate row failures
                    logger.error(
                        "%s judge failed for row %s: %s",
                        self.name,
                        rows[idx].get("id", idx),
                        exc,
                    )
                    for metric in self.metrics:
                        new_row.setdefault(metric, None)
                merged[idx] = new_row
        return [row for row in merged if row is not None]


# ---------------------------------------------------------------------------
# Frame-based VLM judge
# ---------------------------------------------------------------------------

DEFAULT_VLM_DIMENSIONS_FILE = "configs/eval/t2v/vlm_dimensions.yaml"
DEFAULT_VLM_JUDGE_PROMPT_FILE = "configs/prompts/t2v_vlm_judge_prompt.txt"


class VLMChecker(_StructuredJudgeChecker):
    """Frame-based VLM judge: one structured multi-dimension call per row.

    Frames are uniformly sampled from the generated video (cv2, timestamped,
    JPEG data URIs) and sent as an ordered multi-image payload — vision
    models interpret the sequence as a video.  I2V rows have their
    conditioning first frame prepended and labeled so the judge can score
    first-frame consistency; for plain T2V rows the judge marks that
    dimension as not applicable.

    The VLM judges only what sparse frames can reliably show (per-frame
    content, cross-frame identity); dynamic qualities belong to
    :class:`OmniChecker`.

    Row fields used:
      - ``optimized_prompt`` (or ``prompt``): the text prompt.
      - ``video_urls``: the first entry is evaluated (local path preferred;
        remote URLs fall back to being passed through directly).
      - ``frame_urls`` (optional): pre-extracted frames, skip sampling.
      - ``first_frame_image`` (optional, I2V): conditioning image.

    Extra configurable fields (on top of the shared judge fields):
      - frame_sample_count / frame_max_size / jpeg_quality: sampling knobs.
    """

    name = "vlm"
    DEFAULT_DIMENSIONS_FILE = DEFAULT_VLM_DIMENSIONS_FILE
    DEFAULT_JUDGE_PROMPT_FILE = DEFAULT_VLM_JUDGE_PROMPT_FILE

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ):
        super().__init__(config, backend)
        self.frame_sample_count = int(self.config.get("frame_sample_count", 8))
        self.frame_max_size = int(self.config.get("frame_max_size", 768))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 85))

    def _collect_frames(self, row: Dict[str, Any]) -> Tuple[List[str], str]:
        """Return (image refs, frames_info text) for one row.

        Priority: pre-extracted ``frame_urls`` > local-file sampling >
        raw video reference fallback (keeps judges that accept video URLs
        working when sampling is impossible).
        """
        pre_extracted = list(row.get("frame_urls") or [])
        if pre_extracted:
            info = (
                f"{len(pre_extracted)} pre-extracted frames in temporal order "
                f"(Frame 1 .. Frame {len(pre_extracted)})."
            )
            return pre_extracted, info

        video_urls = row.get("video_urls") or []
        if isinstance(video_urls, str):
            video_urls = [video_urls]
        if not video_urls:
            return [], "No video available."
        video_ref = str(video_urls[0])

        if not video_ref.startswith(("http://", "https://")):
            try:
                frames = sample_video_frames(
                    video_ref,
                    count=self.frame_sample_count,
                    max_size=self.frame_max_size,
                    jpeg_quality=self.jpeg_quality,
                )
                lines = []
                for i, frame in enumerate(frames, start=1):
                    if frame.timestamp is not None:
                        lines.append(
                            f"Frame {i}/{len(frames)}, t={frame.timestamp:.2f}s"
                        )
                    else:
                        lines.append(f"Frame {i}/{len(frames)}")
                info = (
                    f"{len(frames)} frames uniformly sampled in temporal order:\n"
                    + "\n".join(lines)
                )
                return [frame.data_url for frame in frames], info
            except (ImportError, ValueError) as exc:
                logger.warning(
                    "Frame sampling failed for row %s (%s); falling back to "
                    "the raw video reference.",
                    row.get("id"),
                    exc,
                )
        return [video_ref], "Raw video reference provided (frame sampling unavailable)."

    def _build_message_content(self, row: Dict[str, Any], is_i2v: bool) -> Any:
        prompt = row.get("optimized_prompt") or row.get("prompt") or ""
        images, frames_info = self._collect_frames(row)
        if is_i2v:
            images = [row["first_frame_image"], *images]
            frames_info = (
                "The FIRST image is the conditioning first frame (not a "
                "sampled frame). Subsequent images are the sampled frames.\n"
                + frames_info
            )
        judge_prompt = format_prompt_safely(
            self.prompt_template,
            prompt=prompt,
            frames_info=frames_info,
            dim_block=self._build_dim_block(is_i2v),
        )
        if images:
            return build_multimodal_user_content(judge_prompt, images)
        return judge_prompt


# ---------------------------------------------------------------------------
# Omni holistic video judge
# ---------------------------------------------------------------------------

DEFAULT_OMNI_DIMENSIONS_FILE = "configs/eval/t2v/omni_dimensions.yaml"
DEFAULT_OMNI_JUDGE_PROMPT_FILE = "configs/prompts/t2v_omni_judge_prompt.txt"
DEFAULT_OMNI_MAX_VIDEO_MB = 100.0


class OmniChecker(_StructuredJudgeChecker):
    """Full-video judge via a video-native (Omni-style) understanding model.

    The complete video is attached as a ``video_url`` content item
    (OpenAI-compatible multimodal extension), so the judge can score the
    dynamic dimensions sparse frames cannot show: motion quality, temporal
    execution of the prompt's beats, and camera accuracy.

    The judge endpoint usually differs from the frame-VLM endpoint —
    configure a dedicated one via the checker-level ``backend`` dict.

    Extra configurable fields (on top of the shared judge fields):
      - video_transport: "auto" (default), "url" or "base64".
        auto prefers the row's remote URL (``video_remote_urls``) and falls
        back to base64-encoding the local file.
      - max_video_mb: base64 transport size cap (default 100 MB); larger
        local files fall back to the remote URL or are skipped.
    """

    name = "omni"
    DEFAULT_DIMENSIONS_FILE = DEFAULT_OMNI_DIMENSIONS_FILE
    DEFAULT_JUDGE_PROMPT_FILE = DEFAULT_OMNI_JUDGE_PROMPT_FILE

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ):
        super().__init__(config, backend)
        self.video_transport = str(self.config.get("video_transport", "auto")).lower()
        self.max_video_bytes = int(
            float(self.config.get("max_video_mb", DEFAULT_OMNI_MAX_VIDEO_MB)) * 1024 * 1024
        )

    def _video_reference(self, row: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Resolve the video payload reference and a short note for the prompt."""
        remote_urls = row.get("video_remote_urls") or []
        if isinstance(remote_urls, str):
            remote_urls = [remote_urls]
        video_urls = row.get("video_urls") or []
        if isinstance(video_urls, str):
            video_urls = [video_urls]
        remote = str(remote_urls[0]) if remote_urls else None
        if not remote:
            for ref in video_urls:
                if str(ref).startswith(("http://", "https://")):
                    remote = str(ref)
                    break
        local = self._local_video_path(row)

        if self.video_transport == "url":
            return remote, "The complete video is provided via URL."
        if self.video_transport == "base64":
            if local:
                return (
                    load_video_to_data_url(local, max_bytes=self.max_video_bytes),
                    "The complete video is attached inline.",
                )
            return None, ""
        # auto: prefer the remote URL (no size limits), fall back to base64.
        if remote:
            return remote, "The complete video is provided via URL."
        if local:
            try:
                return (
                    load_video_to_data_url(local, max_bytes=self.max_video_bytes),
                    "The complete video is attached inline.",
                )
            except ValueError as exc:
                logger.warning(
                    "OmniChecker: base64 transport unavailable for row %s: %s",
                    row.get("id"),
                    exc,
                )
        return None, ""

    def _build_message_content(self, row: Dict[str, Any], is_i2v: bool) -> Any:
        prompt = row.get("optimized_prompt") or row.get("prompt") or ""
        video_ref, video_info = self._video_reference(row)
        if not video_ref:
            raise ValueError(
                f"OmniChecker: no usable video reference for row {row.get('id')} "
                f"(transport={self.video_transport})."
            )
        judge_prompt = format_prompt_safely(
            self.prompt_template,
            prompt=prompt,
            video_info=video_info,
            dim_block=self._build_dim_block(is_i2v),
        )
        return [
            {"type": "video_url", "video_url": {"url": video_ref}},
            {"type": "text", "text": judge_prompt},
        ]


# ---------------------------------------------------------------------------
# Registry & builder
# ---------------------------------------------------------------------------

CHECKER_REGISTRY: Dict[str, type] = {
    VBenchChecker.name: VBenchChecker,
    VLMChecker.name: VLMChecker,
    OmniChecker.name: OmniChecker,
}


def build_video_checkers(
    checker_configs: List[Dict[str, Any]],
    backend: Optional[Any] = None,
) -> List[BaseVideoChecker]:
    """Instantiate enabled checkers from a list of config dicts.

    Each entry needs a ``type`` key naming a registered checker; remaining
    keys are passed to the checker as its config.  Disabled entries
    (``enabled: false``) are skipped.
    """
    checkers: List[BaseVideoChecker] = []
    for entry in checker_configs:
        checker_type = str(entry.get("type") or "")
        checker_cls = CHECKER_REGISTRY.get(checker_type)
        if checker_cls is None:
            raise ValueError(
                f"Unknown video checker type: {checker_type!r}. "
                f"Registered types: {sorted(CHECKER_REGISTRY)}."
            )
        config = {k: v for k, v in entry.items() if k != "type"}
        enabled = config.get("enabled")
        if enabled is not None and not enabled:
            logger.info("Video checker '%s' is disabled, skipping.", checker_type)
            continue
        checkers.append(checker_cls(config=config, backend=backend))
    return checkers
