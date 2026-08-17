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

"""Single-file T2I multi-model (multi-teacher) evaluation with Agentic Debate.

Self-contained single-file implementation of the T2I evaluation line, following the
same evaluator style as the other modules in ``easydistill/eval``. This file
does not depend on any other T2I/TI2I module; the sibling entries are the
independent files ``t2i_single_model.py`` / ``ti2i_multi_model.py`` /
``ti2i_single_model.py``. Functionality is preserved end to
end:

1. T2I 60-dim frozen pool (``t2i_dimensions.json``), L1 -> L3, 0-4.
2. Multi-teacher scoring at case x teacher x L1-group granularity through
   ``ModelBackend`` (PAI Token / PAI EAS / OpenAI-compatible).
3. Cross-teacher conflict detection per L3 dimension.
4. Three-step Agentic Debate arbitration (Step1 independent review,
   Step2 prosecution/defense, Step3 arbitration) on the arbiter model.
5. Weighted-majority merge, plus optional reason synthesis on a separate
   reason model (two models, two steps, same as the original pipeline), and
   per-case ``overall_score_100``: 0/25/50/75/100 map, NA excluded, Safety
   L1 excluded from total, Safety Compliance = 0 vetoes the total to 0.
6. Training-data export (SFT / DPO / uncertain bins) from final labels, so
   Debate outcomes land in standard training structures.

Only engineering conveniences of the original pipeline are dropped: resume /
incremental checkpointing, per-plan auto retry, and batch isolation knobs.

Public artifacts keep the ``teacher`` field as the actual model name
(e.g. ``qwen3.7-plus`` / ``kimi-k2.6``). Single-teacher mode reuses the same
flow; with one teacher there is no cross-teacher conflict so Debate is
skipped automatically.

Typical use::

    from easydistill.eval.t2i_multi_model import T2IMultiModelEvaluator

    evaluator = T2IMultiModelEvaluator(
        teachers={"qwen3.7-plus": backend1, "kimi-k2.6": backend2},
        arbiter=eas_backend,
    )
    results = evaluator.run(samples)
    summary = evaluator.aggregate(results)

CLI (same seed jsonl format as the full pipeline)::

    python -m easydistill.eval.t2i_multi_model \
        --config configs/eval/t2i_ti2i/t2i_multi_model_pai_token.yaml
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from easydistill.backends.base import ModelBackend
from easydistill.eval._common import (
    SCORE_MAP_5,
)
from easydistill.eval._common import (
    clamp_score as _clamp_score,
)
from easydistill.eval._common import (
    get_config_float as _get_float,
)
from easydistill.eval._common import (
    get_config_int as _get_int,
)
from easydistill.eval._common import (
    load_dimension_pool as _load_dimension_pool,
)
from easydistill.eval._common import (
    parse_json_block as _parse_json_block,
)
from easydistill.prompts import resolve_prompts
from easydistill.utils import build_multimodal_user_content

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIMENSIONS_FILE = _REPO_ROOT / "configs" / "eval" / "t2i_ti2i" / "t2i_dimensions.json"
DEFAULT_PROMPTS_FILE = _REPO_ROOT / "configs" / "prompts" / "t2i_multi_model_prompts.yaml"
REQUIRED_PROMPTS = ("teacher", "reason_synthesis", "debate_step1", "debate_step2", "debate_step3")


def load_dimension_pool(path: Optional[str] = None) -> Dict[str, Any]:
    return _load_dimension_pool(DEFAULT_DIMENSIONS_FILE, path)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class T2IMultiModelEvaluator:
    """Multi-teacher T2I evaluator with three-step Debate arbitration.

    Args:
        teachers: mapping of actual model name -> ModelBackend. One entry
            means single-teacher mode (Debate skipped, same output schema).
        arbiter: backend used only for the three Debate steps; defaults to
            the first teacher backend when omitted.
        reason_model: backend used only for final reason synthesis (a
            separate model and step from Debate, mirroring the original
            pipeline); defaults to the arbiter when omitted.
        config:
            - dimensions_path: optional dimension pool override.
            - prompts_file / prompts: prompt template overrides following the
              shared prompt resolution (file > inline > default file
              ``configs/prompts/t2i_multi_model_prompts.yaml``).
            - conflict_threshold: min cross-teacher score spread to trigger
              Debate for a dimension (default 2).
            - max_debate_dims: cap of arbitrated dims per case (default 6).
            - max_workers: teacher-call concurrency per run (default 4).
            - synthesize_reasons: normalize majority reasons via the
              reason model (default False).
            - call_retries / retry_delay_sec: per-call retry knobs.
            - temperature / max_tokens: generation params.
    """

    name = "t2i_multi_model_evaluator"
    line = "t2i"

    def __init__(
        self,
        teachers: Dict[str, ModelBackend],
        arbiter: Optional[ModelBackend] = None,
        reason_model: Optional[ModelBackend] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not teachers:
            raise ValueError("At least one teacher backend is required.")
        self.teachers = teachers
        self.arbiter = arbiter or next(iter(teachers.values()))
        self.reason_model = reason_model or self.arbiter
        self.config = config or {}
        self.pool = load_dimension_pool(self.config.get("dimensions_path"))
        self.prompts = resolve_prompts(self.config, default_file=str(DEFAULT_PROMPTS_FILE))
        missing = [key for key in REQUIRED_PROMPTS if not self.prompts.get(key)]
        if missing:
            raise ValueError(
                f"Missing prompt templates {missing}; provide them via "
                f"'prompts_file' / 'prompts' or keep {DEFAULT_PROMPTS_FILE}."
            )
        self.conflict_threshold = _get_int(self.config, "conflict_threshold", 2)
        self.max_debate_dims = _get_int(self.config, "max_debate_dims", 6)
        self.max_workers = _get_int(self.config, "max_workers", 4)
        self.synthesize_reasons = bool(self.config.get("synthesize_reasons") or False)
        self.temperature = _get_float(self.config, "temperature", 0.0)
        self.max_tokens = _get_int(self.config, "max_tokens", 8000)
        self.call_retries = _get_int(self.config, "call_retries", 2)
        self.retry_delay_sec = _get_float(self.config, "retry_delay_sec", 5.0)

    # -- sample extraction --------------------------------------------------

    def _extract_case(self, sample: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Return (case_id, prompt text, image list) for one T2I seed row."""
        case_id = str(sample.get("prompt_id") or sample.get("id") or sample.get("case_id") or "")
        instruction = str(sample.get("prompt") or sample.get("instruction") or "")
        images = [sample.get("image")] if sample.get("image") else []
        return case_id, instruction, [str(img) for img in images]

    def _case_context(self, instruction: str) -> str:
        return f"生成 prompt：{instruction}"

    # -- model calls ---------------------------------------------------------

    def _call_json(
        self, backend: ModelBackend, prompt: str, images: List[str]
    ) -> Dict[str, Any]:
        """Call a backend and parse strict JSON, with light per-call retries.

        Mirrors the original pipeline's per-plan auto retry: transient
        failures (e.g. gateway 503 / timeouts / malformed JSON) are retried
        ``call_retries`` times before the caller-level fallback kicks in.
        """
        content = build_multimodal_user_content(prompt, images) if images else prompt
        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(1 + self.call_retries):
            try:
                result = backend.generate(
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return _parse_json_block(result.response)
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_exc = exc
                if attempt < self.call_retries:
                    logger.warning(
                        "call retry %d/%d after error: %s", attempt + 1, self.call_retries, exc
                    )
                    time.sleep(self.retry_delay_sec)
        raise last_exc

    def _run_teacher_group(
        self,
        teacher: str,
        backend: ModelBackend,
        l1_group: str,
        dims: List[Dict[str, Any]],
        instruction: str,
        images: List[str],
    ) -> Dict[str, Any]:
        """One case x teacher x L1-group scoring call (same granularity as full pipeline)."""
        dim_block = "\n".join(
            f"- {dim['name']}：" + "；".join(f"{k}={v}" for k, v in (dim["criteria"] or {}).items())
            for dim in dims
        )
        prompt = self.prompts["teacher"].format(
            context=self._case_context(instruction),
            l1_group=l1_group,
            dim_block=dim_block,
        )
        row: Dict[str, Any] = {"teacher": teacher, "l1_group": l1_group, "judgments": []}
        try:
            payload = self._call_json(backend, prompt, images)
            valid_names = {dim["name"] for dim in dims}
            for item in payload.get("dimension_judgments") or []:
                name = str(item.get("dimension") or "")
                if name not in valid_names:
                    continue
                row["judgments"].append(
                    {
                        "dimension": name,
                        "applicable": item.get("applicable"),
                        "score": _clamp_score(item.get("score")),
                        "confidence": item.get("confidence"),
                        "reason": str(item.get("reason") or ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - one failed plan must not kill the case
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Teacher %s failed on L1 %s: %s", teacher, l1_group, exc)
        return row

    # -- pipeline stages -----------------------------------------------------

    def _run_teachers(self, instruction: str, images: List[str]) -> List[Dict[str, Any]]:
        tasks = [
            (teacher, backend, l1_group, dims)
            for teacher, backend in self.teachers.items()
            for l1_group, dims in self.pool["l1_groups"].items()
        ]
        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self._run_teacher_group, t, b, g, d, instruction, images)
                for t, b, g, d in tasks
            ]
            # Collect in submission order so teacher votes / artifacts stay
            # deterministic (majority tie-break depends on vote order).
            rows = [future.result() for future in futures]
        return rows

    def _collect_dim_scores(
        self, teacher_rows: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        by_dim: Dict[str, List[Dict[str, Any]]] = {}
        for row in teacher_rows:
            for judgment in row.get("judgments") or []:
                by_dim.setdefault(judgment["dimension"], []).append(
                    {**judgment, "teacher": row["teacher"]}
                )
        return by_dim

    def _detect_conflicts(self, by_dim: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Dims whose cross-teacher score spread >= threshold (needs >= 2 teachers).

        When conflicts exceed ``max_debate_dims``, the largest spreads win the
        debate slots (severity first; dim name as deterministic tie-break).
        """
        conflicts: List[Tuple[int, str]] = []
        for dim, votes in by_dim.items():
            scores = [v["score"] for v in votes if v.get("score") is not None]
            if len(scores) >= 2 and max(scores) - min(scores) >= self.conflict_threshold:
                conflicts.append((max(scores) - min(scores), dim))
        conflicts.sort(key=lambda item: (-item[0], item[1]))
        return [dim for _, dim in conflicts[: self.max_debate_dims]]

    def _criteria_text(self, dimension: str) -> str:
        l1 = self.pool["l3_to_l1"].get(dimension, "")
        for dim in self.pool["l1_groups"].get(l1, []):
            if dim["name"] == dimension:
                return "；".join(f"{k}={v}" for k, v in (dim["criteria"] or {}).items())
        return ""

    def _run_debate(
        self,
        dimension: str,
        votes: List[Dict[str, Any]],
        instruction: str,
        images: List[str],
    ) -> Dict[str, Any]:
        """Three-step Debate for one conflicted dimension; all steps recorded."""
        context = self._case_context(instruction)
        teacher_scores = {v["teacher"]: v.get("score") for v in votes}
        criteria = self._criteria_text(dimension)
        steps: Dict[str, Any] = {}
        try:
            step1 = self._call_json(
                self.arbiter,
                self.prompts["debate_step1"].format(
                    dimension=dimension, context=context, criteria=criteria,
                    teacher_scores=json.dumps(teacher_scores, ensure_ascii=False),
                ),
                images,
            )
            steps["step1"] = step1
            step2 = self._call_json(
                self.arbiter,
                self.prompts["debate_step2"].format(
                    dimension=dimension, context=context,
                    teacher_scores=json.dumps(teacher_scores, ensure_ascii=False),
                    step1_score=step1.get("score"),
                ),
                images,
            )
            steps["step2"] = step2
            step3 = self._call_json(
                self.arbiter,
                self.prompts["debate_step3"].format(
                    dimension=dimension, context=context, criteria=criteria,
                    teacher_scores=json.dumps(teacher_scores, ensure_ascii=False),
                    step1=json.dumps(step1, ensure_ascii=False),
                    prosecution=step2.get("prosecution", ""),
                    defense=step2.get("defense", ""),
                ),
                images,
            )
            steps["step3"] = step3
            return {
                "dimension": dimension,
                "success": True,
                "final_score": _clamp_score(step3.get("score")),
                "final_applicable": step3.get("applicable", True),
                "confidence": step3.get("confidence"),
                "reason": str(step3.get("reason") or ""),
                "steps": steps,
            }
        except Exception as exc:  # noqa: BLE001 - fall back to majority on failure
            logger.warning("Debate failed on dimension %s: %s", dimension, exc)
            return {"dimension": dimension, "success": False, "error": str(exc), "steps": steps}

    @staticmethod
    def _majority(votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Confidence-weighted majority over teacher votes for one dimension."""
        scored = [v for v in votes if v.get("score") is not None]
        if not scored:
            applicable_votes = [v for v in votes if v.get("applicable") is False]
            reason = applicable_votes[0].get("reason", "") if applicable_votes else ""
            return {"score": None, "applicable": False, "confidence": None, "reason": reason}
        weights: Dict[int, float] = {}
        for vote in scored:
            weight = float(vote.get("confidence") or 0.5)
            weights[vote["score"]] = weights.get(vote["score"], 0.0) + weight
        best = max(weights, key=lambda score: weights[score])
        support = [v for v in scored if v["score"] == best]
        confidence = sum(float(v.get("confidence") or 0.5) for v in support) / len(support)
        return {
            "score": best,
            "applicable": True,
            "confidence": round(confidence, 3),
            "reason": support[0].get("reason", ""),
        }

    def _synthesize_majority_reason(
        self, dimension: str, majority: Dict[str, Any], votes: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Normalize the majority reason via the reason model; failure falls back silently."""
        try:
            payload = self._call_json(
                self.reason_model,
                self.prompts["reason_synthesis"].format(
                    dimension=dimension,
                    score=majority["score"],
                    votes=json.dumps(
                        [
                            {
                                "teacher": v["teacher"],
                                "score": v.get("score"),
                                "reason": v.get("reason", ""),
                            }
                            for v in votes
                        ],
                        ensure_ascii=False,
                    ),
                ),
                [],
            )
            reason = str(payload.get("reason") or "").strip()
            return reason or None
        except Exception as exc:  # noqa: BLE001 - synthesis is best-effort
            logger.warning("Reason synthesis failed on %s: %s", dimension, exc)
            return None

    # -- overall score (same rules as the full pipeline) ---------------------

    def _compute_overall(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregation = self.pool["aggregation"]
        excludes = set(aggregation.get("total_excludes") or [])
        included: List[float] = []
        l1_buckets: Dict[str, List[float]] = {}
        n_na = n_unscored = 0
        safety_veto = False
        for judgment in judgments:
            if judgment.get("final_applicable") is False:
                n_na += 1
                continue
            score = judgment.get("final_score")
            if score is None:
                n_unscored += 1
                continue
            mapped = SCORE_MAP_5.get(int(score), float(score) / 4.0 * 100.0)
            l1 = self.pool["l3_to_l1"].get(judgment["dimension"], "unknown")
            l1_buckets.setdefault(l1, []).append(mapped)
            if l1 not in excludes:
                included.append(mapped)
            if (
                aggregation.get("safety_veto")
                and judgment["dimension"] == "Safety Compliance"
                and score == 0
            ):
                safety_veto = True
        overall = round(sum(included) / len(included), 2) if included else None
        if safety_veto:
            overall = 0.0
        return {
            "overall_score_100": overall,
            "aggregation_method": "equal_weight_mean+safety_veto",
            "l1_subscores_100": {
                name: round(sum(values) / len(values), 2)
                for name, values in sorted(l1_buckets.items())
            },
            "n_scored_dims": len(included),
            "n_na_dims": n_na,
            "n_unscored_dims": n_unscored,
            "safety_veto_triggered": safety_veto,
            "note_zh": (
                "L3 等权均值（0/25/50/75/100），NA 不进均值，"
                "Safety 组不进总分；Safety Compliance=0 一票否决置 0。"
            ),
        }

    # -- public API ----------------------------------------------------------

    def evaluate_case(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full chain for one case and return the final-label row."""
        case_id, instruction, images = self._extract_case(sample)
        teacher_rows = self._run_teachers(instruction, images)
        by_dim = self._collect_dim_scores(teacher_rows)
        conflict_dims = self._detect_conflicts(by_dim) if len(self.teachers) > 1 else []

        # Debate dims are independent; run them concurrently (3 steps inside
        # one dim stay sequential) and keep dim order deterministic.
        debate_results: Dict[str, Dict[str, Any]] = {}
        if conflict_dims:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    dim: pool.submit(self._run_debate, dim, by_dim[dim], instruction, images)
                    for dim in conflict_dims
                }
                debate_results = {dim: future.result() for dim, future in futures.items()}

        final_judgments: List[Dict[str, Any]] = []
        for dimension in sorted(by_dim):
            votes = by_dim[dimension]
            debate = debate_results.get(dimension)
            if debate and debate.get("success"):
                final_judgments.append(
                    {
                        "dimension": dimension,
                        "final_score": debate["final_score"],
                        "final_score_100": SCORE_MAP_5.get(debate["final_score"])
                        if debate["final_score"] is not None
                        else None,
                        "final_applicable": debate["final_applicable"],
                        "final_confidence": debate.get("confidence"),
                        "final_reason": debate.get("reason", ""),
                        "final_source": "debate_arbitration",
                        "was_debated": True,
                        "teacher_votes": votes,
                        "debate_steps": debate.get("steps"),
                    }
                )
                continue
            majority = self._majority(votes)
            reason = majority["reason"]
            reasoning_source = "majority_template"
            if self.synthesize_reasons and majority["score"] is not None:
                synthesized = self._synthesize_majority_reason(dimension, majority, votes)
                if synthesized:
                    reason = synthesized
                    reasoning_source = "arbiter_majority_reason_synthesizer"
            final_judgments.append(
                {
                    "dimension": dimension,
                    "final_score": majority["score"],
                    "final_score_100": SCORE_MAP_5.get(majority["score"])
                    if majority["score"] is not None
                    else None,
                    "final_applicable": majority["applicable"],
                    "final_confidence": majority["confidence"],
                    "final_reason": reason,
                    "reasoning_source": reasoning_source,
                    "final_source": (
                        "fallback_weighted_majority" if debate else "weighted_majority_consensus"
                    ),
                    "was_debated": bool(debate),
                    "teacher_votes": votes,
                }
            )

        overall = self._compute_overall(final_judgments)
        for judgment in final_judgments:
            judgment["case_overall_score_100"] = overall["overall_score_100"]
        return {
            "case_id": case_id,
            "line": self.line,
            "instruction": instruction,
            "images": images,
            "teachers": sorted(self.teachers),
            "teacher_outputs": teacher_rows,
            "conflict_dims": conflict_dims,
            "n_debated": sum(1 for j in final_judgments if j["was_debated"]),
            "final_judgments": final_judgments,
            "overall_score_100": overall["overall_score_100"],
            "overall": overall,
        }

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all cases; one failed case never blocks the rest."""
        results = []
        for index, sample in enumerate(samples):
            try:
                results.append(self.evaluate_case(sample))
            except Exception as exc:  # noqa: BLE001
                logger.error("Case %d failed: %s", index, exc)
                results.append({"case_id": str(sample.get("case_id") or index), "error": str(exc)})
        return results

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch-level overall score stats plus debate coverage."""
        scores = [r["overall_score_100"] for r in results if r.get("overall_score_100") is not None]
        return {
            "cases": len(results),
            "cases_failed": sum(1 for r in results if r.get("error")),
            "overall_score_stats": {
                "mean": round(sum(scores) / len(scores), 2) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            },
            "debated_dims_total": sum(r.get("n_debated") or 0 for r in results),
        }

    def export_training_data(
        self, results: List[Dict[str, Any]], confidence_threshold: float = 0.85
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Bin final labels into SFT / DPO / uncertain training records.

        Mirrors the full pipeline's export: every record keeps the dimension
        verdict, Chinese reason, evidence votes and ``case_overall_score_100``
        so Debate outcomes land in standard training structures.

        - sft: applicable dims with confidence >= threshold (verified).
        - dpo: debated dims where arbitration revised the pre-debate majority
          (chosen = arbitration, rejected = losing majority verdict).
        - uncertain: low-confidence or unresolved dims for later review.
        """
        sft: List[Dict[str, Any]] = []
        dpo: List[Dict[str, Any]] = []
        uncertain: List[Dict[str, Any]] = []
        for result in results:
            if result.get("error"):
                continue
            base = {
                "case_id": result["case_id"],
                "line": result["line"],
                "instruction": result["instruction"],
                "images": result["images"],
            }
            for judgment in result.get("final_judgments") or []:
                record = {
                    **base,
                    "dimension": judgment["dimension"],
                    "final_score": judgment.get("final_score"),
                    "final_score_100": judgment.get("final_score_100"),
                    "final_applicable": judgment.get("final_applicable"),
                    "final_reason": judgment.get("final_reason", ""),
                    "final_source": judgment.get("final_source"),
                    "was_debated": judgment.get("was_debated"),
                    "teacher_votes": judgment.get("teacher_votes"),
                    "case_overall_score_100": judgment.get("case_overall_score_100"),
                }
                confidence = judgment.get("final_confidence")
                if judgment.get("was_debated") and judgment.get("debate_steps"):
                    record["debate_steps"] = judgment["debate_steps"]
                    majority = self._majority(judgment.get("teacher_votes") or [])
                    majority_score = majority["score"]
                    if majority_score is not None and majority_score != judgment.get("final_score"):
                        dpo.append(
                            {
                                **record,
                                "chosen_score": judgment.get("final_score"),
                                "chosen_reason": judgment.get("final_reason", ""),
                                "rejected_score": majority["score"],
                                "rejected_reason": majority["reason"],
                            }
                        )
                if judgment.get("final_applicable") is False or judgment.get("final_score") is None:
                    uncertain.append(record)
                elif confidence is not None and float(confidence) >= confidence_threshold:
                    sft.append(record)
                else:
                    uncertain.append(record)
        return {"sft": sft, "dpo": dpo, "uncertain": uncertain}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_teachers_from_config(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, ModelBackend], Optional[ModelBackend], Optional[ModelBackend]]:
    """Build teacher backends (plus optional arbiter / reason_model) from yaml config."""
    from easydistill.cli.backend_factory import build_backend

    teachers: Dict[str, ModelBackend] = {}
    for entry in cfg.get("teachers") or []:
        name = str(entry.get("name") or entry.get("model_id") or "")
        if not name:
            raise ValueError("Each teacher entry needs 'name' (actual model name).")
        teachers[name] = build_backend(entry)
    arbiter_cfg = cfg.get("arbiter")
    arbiter = build_backend(arbiter_cfg) if arbiter_cfg else None
    reason_cfg = cfg.get("reason_model")
    reason_model = build_backend(reason_cfg) if reason_cfg else None
    return teachers, arbiter, reason_model


def _write_artifacts(out_dir: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Write the same artifact set as the full orchestrator output dir."""
    from easydistill.utils import save_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_outputs = [
        {"case_id": r["case_id"], **row}
        for r in results
        if not r.get("error")
        for row in r.get("teacher_outputs") or []
    ]
    conflict_report = [
        {"case_id": r["case_id"], "conflict_dims": r.get("conflict_dims") or []}
        for r in results
        if not r.get("error")
    ]
    debate_results = [
        {
            "case_id": r["case_id"],
            "n_conflict_dims": len(r.get("conflict_dims") or []),
            "n_debated": r.get("n_debated") or 0,
            "dimensions": {
                j["dimension"]: {
                    "final_score": j.get("final_score"),
                    "steps": j.get("debate_steps"),
                }
                for j in r.get("final_judgments") or []
                if j.get("was_debated")
            },
        }
        for r in results
        if not r.get("error")
    ]
    final_judgments = [
        {"case_id": r["case_id"], **j}
        for r in results
        if not r.get("error")
        for j in r.get("final_judgments") or []
    ]
    save_jsonl(str(out_dir / "teacher_outputs.jsonl"), teacher_outputs)
    save_jsonl(str(out_dir / "conflict_report.jsonl"), conflict_report)
    save_jsonl(str(out_dir / "debate_results.jsonl"), debate_results)
    save_jsonl(str(out_dir / "final_labels.jsonl"), results)
    save_jsonl(str(out_dir / "final_judgments.jsonl"), final_judgments)
    (out_dir / "final_labels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="T2I 多模型（多教师 Debate）单文件评估")
    parser.add_argument(
        "--config", required=True, help="yaml：teachers/arbiter/reason_model/dataset/eval 配置"
    )
    parser.add_argument(
        "--teacher", default=None, help="只用一个教师（实际模型名）；单教师自动跳过 Debate"
    )
    parser.add_argument(
        "--limit-cases", type=int, default=0, help="最多评估多少个 case；0 表示全部"
    )
    parser.add_argument(
        "--synthesize-reasons", action="store_true", help="用 reason 模型规范化多数票 reason"
    )
    parser.add_argument(
        "--export-training", action="store_true", help="同时导出 sft/dpo/uncertain 训练数据"
    )
    args = parser.parse_args()

    import yaml

    from easydistill.cli.backend_factory import close_backends
    from easydistill.utils import save_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    teachers, arbiter, reason_model = _build_teachers_from_config(cfg)
    try:
        if args.teacher:
            if args.teacher not in teachers:
                raise SystemExit(f"--teacher {args.teacher} 不在配置教师池：{sorted(teachers)}")
            teachers = {args.teacher: teachers[args.teacher]}

        eval_cfg = dict(cfg.get("eval") or {})
        if args.synthesize_reasons:
            eval_cfg["synthesize_reasons"] = True
        evaluator = T2IMultiModelEvaluator(
            teachers=teachers, arbiter=arbiter, reason_model=reason_model, config=eval_cfg
        )

        seed_path = Path(cfg["dataset"]["input_path"])
        samples = [
            json.loads(line)
            for line in seed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # seed 里的相对图片路径按 seed 文件目录解析
        for sample in samples:
            value = sample.get("image")
            if value and not str(value).startswith(("http://", "https://", "/")):
                sample["image"] = str((seed_path.parent / str(value)).resolve())
        if args.limit_cases > 0:
            samples = samples[: args.limit_cases]

        results = evaluator.run(samples)
        summary = evaluator.aggregate(results)

        output_dir = cfg["dataset"].get("output_dir")
        out_dir = Path(output_dir) if output_dir else Path(cfg["dataset"]["output_path"]).parent
        _write_artifacts(out_dir, results, summary)
        if args.export_training:
            bins = evaluator.export_training_data(results)
            for bin_name, rows in bins.items():
                save_jsonl(str(out_dir / f"{bin_name}_data.jsonl"), rows)
            logger.info(
                "Exported training data: sft=%d dpo=%d uncertain=%d",
                len(bins["sft"]), len(bins["dpo"]), len(bins["uncertain"]),
            )
        logger.info("Saved %d case results to %s.", len(results), out_dir)
        logger.info("Summary: %s", summary)
    finally:
        close_backends(*teachers.values(), arbiter, reason_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
