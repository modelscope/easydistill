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

"""Agentic prompt rewrite operator (plan -> rewrite -> reflection).

Teacher-side operator for the PE rewrite distillation pipeline: each input
prompt goes through three sequential LLM calls — scene/intent analysis (plan),
scene-aware rewriting (rewrite) and self-check correction (reflection). Rows
run concurrently while the three steps within one row stay sequential. Every
step degrades gracefully: a failed plan falls back to the general scene, and a
failed reflection keeps the rewrite draft.

The rewrite step selects one system prompt per (scene, language) built from
an optional shared ``rewrite_common_{lang}.txt`` block plus the scene's
``rewrite_{scene}_{lang}.txt`` file (10 scenes x zh/en). Scenes without a
dedicated file fall back to the ``general`` prompt of the same language. The
rewrite model answers with a JSON object whose ``gen_prompt`` field carries
the rewritten prompt.
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest
from easydistill.operators.base import Operator
from easydistill.operators.generation import TextGenerationOperator
from easydistill.prompts import resolve_prompt
from easydistill.utils import DEFAULT_MAX_WORKERS, progress

logger = logging.getLogger(__name__)

RewriteInput = Union[str, Dict[str, Any]]

# The 10-scene taxonomy used for plan routing, scene prompt loading and the
# judge's scene_alignment metric (see docs/pe_rewrite_distill_plan_zh.md §5).
# Every scene has a dedicated rewrite_{scene}_{zh,en}.txt system prompt under
# configs/prompts/pe_rewrite/.
SCENES = frozenset(
    {
        "general",
        "photographic_realism",
        "artistic_illustration",
        "design_layout",
        "structured_diagram",
        "ui_interface",
        "brand_commercial_ad",
        "narrative_panels",
        "cultural_heritage_art",
        "game_art_production",
    }
)

_FALLBACK_SCENE = "general"

_DEFAULT_PROMPT_DIR = "configs/prompts/pe_rewrite"

_DEFAULT_PLAN_MESSAGE_TEMPLATE = 'Original prompt:\n"""\n{instruction}\n"""'

_DEFAULT_REWRITE_MESSAGE_TEMPLATE = (
    'Original prompt:\n"""\n{instruction}\n"""\n\nScene: {scene}\nOutput language: {language}'
)

_DEFAULT_REFLECTION_MESSAGE_TEMPLATE = (
    'Original prompt:\n"""\n{instruction}\n"""\n\nRewritten prompt:\n"""\n{rewritten}\n"""'
)

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# Salvage pattern for reflection responses whose JSON is broken (typically by
# unescaped quotes inside `notes`) but whose verdict is still readable.
_CHANGED_FALSE_PATTERN = re.compile(r'"changed"\s*:\s*false', re.IGNORECASE)


def _detect_language(text: str) -> str:
    """Fallback zh/en detection used when the plan step cannot tell us."""
    return "zh" if _CJK_PATTERN.search(text) else "en"


def _parse_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse an LLM response into a JSON object.

    Strips an optional markdown code fence and repairs trailing junk by
    cutting at the last ``}``. Returns None when no object can be recovered.
    """
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _extract_string_field(text: str, key: str) -> Optional[str]:
    """Extract a JSON string value by scanning, tolerating a truncated tail.

    Used to salvage ``gen_prompt`` from responses whose JSON wrapper is broken
    or cut off by the token limit. Handles standard escapes; stops at the
    closing quote or at end-of-text (truncation).
    """
    match = re.search(rf'"{key}"\s*:\s*"', text)
    if not match:
        return None
    chars: List[str] = []
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
    i = match.end()
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            chars.append(escapes.get(text[i + 1], text[i + 1]))
            i += 2
            continue
        if char == '"':
            break
        chars.append(char)
        i += 1
    value = "".join(chars).strip()
    return value or None


class AgenticPromptRewriteOperator(Operator[List[RewriteInput], List[Dict[str, Any]]]):
    """Rewrite prompts through a plan -> rewrite -> reflection agent chain.

    Execution model: rows run concurrently (``max_workers``); the three steps
    within one row are sequential because each step consumes the previous
    step's output. Each step owns an inner single-shot
    :class:`TextGenerationOperator` so per-call retry/backoff and per-step
    ``model_id`` overrides come for free from the same backend endpoint.

    Input: list of prompts, either plain strings or dicts with an
    ``instruction`` field. Extra dict fields (e.g. expansion lineage) are
    passed through to the output row.

    Output: one dict per input row:
    ``{"instruction", "response", "scene", "language", "agent_trace", ...}``.
    ``response`` is the final rewritten prompt; ``agent_trace`` keeps the
    intermediate products for auditing and never enters SFT data. Rows whose
    rewrite step fails are dropped with an error log.

    Config is nested per step::

        plan:       prompt_template(_file) / model_id / temperature / max_tokens
        rewrite:    scene_prompt_dir / message_template / model_id /
                    temperature / max_tokens
        reflection: prompt_template(_file) / model_id / temperature / max_tokens

    plus top-level ``max_workers`` / ``show_progress`` / retry settings shared
    by all steps. The rewrite system prompt is chosen from
    ``{scene_prompt_dir}/rewrite_{scene}_{lang}.txt`` with a per-language
    ``general`` fallback (the general files are required).
    """

    name = "agentic_prompt_rewrite"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Validated configs may inject explicit None values, hence `or`.
        plan_cfg = self.config.get("plan") or {}
        rewrite_cfg = self.config.get("rewrite") or {}
        reflection_cfg = self.config.get("reflection") or {}

        plan_prompt = resolve_prompt(
            plan_cfg, default_file=f"{_DEFAULT_PROMPT_DIR}/plan_prompt.txt"
        )
        reflection_prompt = resolve_prompt(
            reflection_cfg, default_file=f"{_DEFAULT_PROMPT_DIR}/reflection_prompt.txt"
        )
        scene_prompt_dir = rewrite_cfg.get("scene_prompt_dir") or _DEFAULT_PROMPT_DIR
        self.scene_prompts = self._load_scene_prompts(scene_prompt_dir)
        for lang in ("zh", "en"):
            if f"{_FALLBACK_SCENE}_{lang}" not in self.scene_prompts:
                raise FileNotFoundError(
                    f"Required fallback rewrite prompt not found: "
                    f"{scene_prompt_dir}/rewrite_{_FALLBACK_SCENE}_{lang}.txt"
                )

        self.plan_message_template = (
            plan_cfg.get("message_template") or _DEFAULT_PLAN_MESSAGE_TEMPLATE
        )
        self.rewrite_message_template = (
            rewrite_cfg.get("message_template") or _DEFAULT_REWRITE_MESSAGE_TEMPLATE
        )
        self.reflection_message_template = (
            reflection_cfg.get("message_template") or _DEFAULT_REFLECTION_MESSAGE_TEMPLATE
        )

        max_workers = int(self.config.get("max_workers") or DEFAULT_MAX_WORKERS)
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        self.max_workers = max_workers
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True

        # One inner single-shot generator per step; row-level concurrency is
        # owned by this operator's own thread pool (same pattern as
        # SeedAnchoredExpansionOperator).
        self.plan_gen = self._build_step_generator(
            backend, plan_cfg, system_prompt=plan_prompt, default_temperature=0.3
        )
        self.rewrite_gen = self._build_step_generator(
            backend, rewrite_cfg, system_prompt=None, default_temperature=0.7
        )
        self.reflection_gen = self._build_step_generator(
            backend,
            reflection_cfg,
            system_prompt=reflection_prompt,
            default_temperature=0.3,
        )

    def _build_step_generator(
        self,
        backend: ModelBackend,
        step_cfg: Dict[str, Any],
        system_prompt: Optional[str],
        default_temperature: float,
    ) -> TextGenerationOperator:
        temperature = step_cfg.get("temperature")
        return TextGenerationOperator(
            backend=backend,
            config={
                "system_prompt": system_prompt,
                "model_id": step_cfg.get("model_id") or self.config.get("model_id"),
                "temperature": (
                    float(temperature) if temperature is not None else default_temperature
                ),
                "max_tokens": int(step_cfg.get("max_tokens") or 8192),
                "show_progress": False,
                "max_workers": 1,
                "raise_on_error": False,
                "retry_attempts": self.config.get("retry_attempts"),
                "retry_backoff_base": self.config.get("retry_backoff_base"),
                "retry_max_wait": self.config.get("retry_max_wait"),
            },
        )

    @staticmethod
    def _load_scene_prompts(scene_prompt_dir: str) -> Dict[str, str]:
        """Load ``rewrite_{scene}_{lang}.txt`` system prompts that exist.

        If a ``rewrite_common_{lang}.txt`` file exists in the same directory,
        it holds the iron laws shared by every scene (quote wrapping, text
        expansion, density budget, output contract, ...) and is prepended to
        each scene prompt of that language, so scene files only carry
        scene-specific content. Missing scene files are expected; those
        scenes fall back to the ``general`` prompt of the same language at
        request time.
        """
        common: Dict[str, str] = {}
        for lang in ("zh", "en"):
            common_path = Path(scene_prompt_dir) / f"rewrite_common_{lang}.txt"
            if common_path.is_file():
                common[lang] = common_path.read_text(encoding="utf-8").strip("\n")
        prompts: Dict[str, str] = {}
        for scene in SCENES:
            for lang in ("zh", "en"):
                path = Path(scene_prompt_dir) / f"rewrite_{scene}_{lang}.txt"
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8").rstrip("\n")
                if lang in common:
                    text = f"{common[lang]}\n\n{text}"
                prompts[f"{scene}_{lang}"] = text
        logger.info(
            "Loaded %d scene rewrite prompts from %s (common block: %s).",
            len(prompts),
            scene_prompt_dir,
            ", ".join(sorted(common)) or "none",
        )
        return prompts

    @staticmethod
    def _normalize_input(row: RewriteInput) -> Optional[Dict[str, Any]]:
        """Normalize a raw input row into a dict with an ``instruction``."""
        if isinstance(row, str):
            instruction = row.strip()
            extra: Dict[str, Any] = {}
        elif isinstance(row, dict):
            instruction = str(row.get("instruction") or "").strip()
            extra = {k: v for k, v in row.items() if k != "instruction"}
        else:
            return None
        if not instruction:
            return None
        return {"instruction": instruction, **extra}

    def _call_step(
        self, generator: TextGenerationOperator, request_id: str, message: str
    ) -> Optional[str]:
        request = GenerationRequest(
            id=request_id, instruction=message, metadata={"task": self.name}
        )
        results = generator.run([request])
        return results[0].response if results else None

    def _run_plan(self, instruction: str, request_id: str) -> Dict[str, Any]:
        """Plan step with fallback: never blocks the chain."""
        raw = self._call_step(
            self.plan_gen,
            f"{request_id}_plan",
            self.plan_message_template.format(instruction=instruction),
        )
        parsed = _parse_json_object(raw)
        status = "ok"
        if parsed is None:
            status = "failed" if raw is None else "parse_failed"
            parsed = {}
        scene = str(parsed.get("scene") or "").strip()
        if scene not in SCENES:
            if status == "ok" and scene:
                status = "invalid_scene"
            scene = _FALLBACK_SCENE
        language = str(parsed.get("language") or "").strip().lower()
        if language not in {"zh", "en"}:
            language = _detect_language(instruction)
        return {
            "scene": scene,
            "language": language,
            "status": status,
            "raw": raw,
        }

    def _run_rewrite(
        self, instruction: str, plan: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Rewrite step: full per-(scene, language) system prompt.

        The system prompt (ported from agentic-pe as-is) instructs the model
        to answer with a JSON object; only ``gen_prompt`` is extracted and the
        remaining fields (negative_prompt / reference_images) are discarded.
        Parsing degrades in order: strict JSON -> string-field salvage (broken
        or truncated JSON) -> plain text (custom non-JSON prompts).
        """
        language = plan["language"]
        system_prompt = self.scene_prompts.get(f"{plan['scene']}_{language}")
        if system_prompt is None:
            system_prompt = self.scene_prompts[f"{_FALLBACK_SCENE}_{language}"]
        message = self.rewrite_message_template.format(
            instruction=instruction,
            scene=plan["scene"],
            language=language,
        )
        request = GenerationRequest(
            id=f"{request_id}_rewrite",
            instruction=message,
            system_prompt=system_prompt,
            metadata={"task": self.name},
        )
        results = self.rewrite_gen.run([request])
        raw = results[0].response if results else None
        if not raw or not raw.strip():
            return {"draft": None, "status": "failed"}
        raw = raw.strip()

        parsed = _parse_json_object(raw)
        if parsed is not None and str(parsed.get("gen_prompt") or "").strip():
            return {"draft": str(parsed["gen_prompt"]).strip(), "status": "ok"}
        salvaged = _extract_string_field(raw, "gen_prompt")
        if salvaged:
            return {"draft": salvaged, "status": "ok_salvaged"}
        # No JSON wrapper at all: treat the whole response as the draft so
        # custom plain-text rewrite prompts keep working.
        return {"draft": raw, "status": "plain_text"}

    def _run_reflection(self, instruction: str, draft: str, request_id: str) -> Dict[str, Any]:
        """Reflection step with fallback: a failure keeps the rewrite draft."""
        raw = self._call_step(
            self.reflection_gen,
            f"{request_id}_reflection",
            self.reflection_message_template.format(instruction=instruction, rewritten=draft),
        )
        parsed = _parse_json_object(raw)
        if parsed is None:
            # A "changed": false verdict means the draft is already final, so
            # it is safe to salvage a pass even when the JSON around it is
            # broken (the notes are only kept for auditing anyway).
            if raw and _CHANGED_FALSE_PATTERN.search(raw):
                return {
                    "final": draft,
                    "changed": False,
                    "notes": "",
                    "status": "ok_salvaged",
                    "raw": raw,
                }
            return {
                "final": draft,
                "changed": False,
                "notes": "",
                "status": "failed" if raw is None else "parse_failed",
                "raw": raw,
            }
        changed = bool(parsed.get("changed"))
        revised = str(parsed.get("rewritten_prompt") or "").strip()
        # `changed` without a usable or actually different revision counts as
        # no change.
        if not changed or not revised or revised == draft:
            return {
                "final": draft,
                "changed": False,
                "notes": str(parsed.get("notes") or ""),
                "status": "ok",
                "raw": raw,
            }
        return {
            "final": revised,
            "changed": True,
            "notes": str(parsed.get("notes") or ""),
            "status": "ok",
            "raw": raw,
        }

    def _rewrite_one(self, row: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Run the full three-step chain for one row."""
        instruction = row["instruction"]
        request_id = str(row.get("id") or f"row_{index}")
        durations: Dict[str, float] = {}

        start = time.perf_counter()
        plan = self._run_plan(instruction, request_id)
        durations["plan"] = round(time.perf_counter() - start, 3)

        start = time.perf_counter()
        rewrite = self._run_rewrite(instruction, plan, request_id)
        durations["rewrite"] = round(time.perf_counter() - start, 3)
        draft = rewrite["draft"]
        if not draft:
            logger.error("Rewrite step produced no output for row %s; dropping row.", request_id)
            return None

        start = time.perf_counter()
        reflection = self._run_reflection(instruction, draft, request_id)
        durations["reflection"] = round(time.perf_counter() - start, 3)

        record = dict(row)
        record.update(
            {
                "instruction": instruction,
                "response": reflection["final"],
                "scene": plan["scene"],
                "language": plan["language"],
                "agent_trace": {
                    "plan": {
                        "status": plan["status"],
                        "raw": plan["raw"],
                    },
                    "rewrite": {
                        "status": rewrite["status"],
                        "draft": draft,
                    },
                    "reflection": {
                        "status": reflection["status"],
                        "changed": reflection["changed"],
                        "notes": reflection["notes"],
                        "raw": reflection["raw"],
                    },
                    "durations": durations,
                },
            }
        )
        return record

    def run(self, input_data: List[RewriteInput]) -> List[Dict[str, Any]]:
        rows = []
        for index, raw_row in enumerate(input_data or []):
            row = self._normalize_input(raw_row)
            if row is None:
                logger.warning("Skipping row at index %d: empty or invalid.", index)
                continue
            rows.append(row)
        if not rows:
            return []

        # Optional streaming sink: append each finished record immediately (in
        # completion order) so an interrupted run keeps its partial output.
        stream_path = self.config.get("stream_output_path")
        stream_ctx: AbstractContextManager[Optional[TextIO]]
        if stream_path:
            path = Path(stream_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream_ctx = path.open("w", encoding="utf-8")
        else:
            stream_ctx = nullcontext()

        with stream_ctx as stream_file:
            stream_lock = threading.Lock()

            def emit(record: Optional[Dict[str, Any]]) -> None:
                if stream_file is not None and record is not None:
                    with stream_lock:
                        stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream_file.flush()

            # Rows are independent, so they run concurrently; results are
            # collected by input order to keep the output deterministic.
            results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
            if self.max_workers <= 1:
                for idx, row in enumerate(
                    progress(
                        rows,
                        enabled=self.show_progress,
                        total=len(rows),
                        desc="Agentic rewrite",
                    )
                ):
                    results[idx] = self._rewrite_one(row, idx)
                    emit(results[idx])
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self._rewrite_one, row, idx): idx
                        for idx, row in enumerate(rows)
                    }
                    for future in progress(
                        as_completed(futures),
                        enabled=self.show_progress,
                        total=len(futures),
                        desc="Agentic rewrite",
                    ):
                        idx = futures[future]
                        try:
                            record = future.result()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Rewrite row %d raised: %s", idx, exc
                            )
                            record = None
                        results[idx] = record
                        emit(record)

        outputs = [record for record in results if record is not None]
        dropped = len(rows) - len(outputs)
        logger.info(
            "%s rewrote %d/%d rows (%d dropped).",
            self.name,
            len(outputs),
            len(rows),
            dropped,
        )
        return outputs
