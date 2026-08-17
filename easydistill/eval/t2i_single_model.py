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

"""Single-file T2I single-model (single-teacher) evaluation.

Self-contained single-file implementation of the T2I single_model entry, following
the same evaluator style as the other modules in ``easydistill/eval``. This
file does not depend on any other T2I/TI2I module; the sibling entries are
the independent files ``t2i_multi_model.py`` / ``ti2i_multi_model.py``
/ ``ti2i_single_model.py``. Semantics match the original pipeline's
single-teacher mode:

1. T2I 60-dim frozen pool (``t2i_dimensions.json``), L1 -> L3, 0-4.
2. One scoring teacher (actual model name, e.g. ``qwen3.7-plus``) evaluated
   at case x L1-group granularity through ``ModelBackend`` (PAI Token /
   PAI EAS / OpenAI-compatible).
3. No cross-teacher conflict exists, so Debate is skipped by design — the
   final label of each dimension is the teacher's own verdict, with the
   same output schema as the multi-model entry (``was_debated=false``).
4. Optional reason synthesis: normalize the teacher reason via a separate
   reason model (mirror of the original pipeline's majority reason
   synthesis).
5. Per-case ``overall_score_100``: 0/25/50/75/100 map, NA excluded, Safety
   L1 excluded from total, Safety Compliance = 0 vetoes the total to 0.
6. Training-data export (SFT / uncertain bins; the DPO bin stays empty
   because there is no arbitration revision in single-model mode).

Only engineering conveniences of the original pipeline are dropped: resume /
incremental checkpointing, per-plan auto retry, and batch isolation knobs.

Typical use::

    from easydistill.eval.t2i_single_model import T2ISingleModelEvaluator

    evaluator = T2ISingleModelEvaluator(teacher="qwen3.7-plus", backend=backend)
    results = evaluator.run(samples)
    summary = evaluator.aggregate(results)

CLI (same seed jsonl format as the full pipeline)::

    python -m easydistill.eval.t2i_single_model \
        --config configs/eval/t2i_ti2i/t2i_single_model_pai_token.yaml \
        --teacher qwen3.7-plus
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
DEFAULT_PROMPTS_FILE = _REPO_ROOT / "configs" / "prompts" / "t2i_single_model_prompts.yaml"
REQUIRED_PROMPTS = ("teacher", "reason_synthesis")


def load_dimension_pool(path: Optional[str] = None) -> Dict[str, Any]:
    return _load_dimension_pool(DEFAULT_DIMENSIONS_FILE, path)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class T2ISingleModelEvaluator:
    """Single-teacher T2I evaluator (no cross-teacher conflict, no Debate).

    Args:
        teacher: actual model name used as the public teacher label
            (e.g. ``qwen3.7-plus`` / ``kimi-k2.6``).
        backend: ModelBackend of the scoring teacher.
        reason_model: optional backend to normalize teacher reasons
            (mirror of the original pipeline's reason synthesis); defaults
            to the teacher backend when ``synthesize_reasons`` is enabled.
        config:
            - dimensions_path: optional dimension pool override.
            - prompts_file / prompts: prompt template overrides following the
              shared prompt resolution (file > inline > default file
              ``configs/prompts/t2i_single_model_prompts.yaml``).
            - max_workers: L1-group call concurrency per case (default 4).
            - synthesize_reasons: normalize reasons via reason_model
              (default False).
            - call_retries / retry_delay_sec: per-call retry knobs.
            - temperature / max_tokens: generation params.
    """

    name = "t2i_single_model_evaluator"
    line = "t2i"

    def __init__(
        self,
        teacher: str,
        backend: ModelBackend,
        reason_model: Optional[ModelBackend] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not teacher:
            raise ValueError("'teacher' must be the actual model name, e.g. qwen3.7-plus.")
        self.teacher = teacher
        self.backend = backend
        self.reason_model = reason_model or backend
        self.config = config or {}
        self.pool = load_dimension_pool(self.config.get("dimensions_path"))
        self.prompts = resolve_prompts(self.config, default_file=str(DEFAULT_PROMPTS_FILE))
        missing = [key for key in REQUIRED_PROMPTS if not self.prompts.get(key)]
        if missing:
            raise ValueError(
                f"Missing prompt templates {missing}; provide them via "
                f"'prompts_file' / 'prompts' or keep {DEFAULT_PROMPTS_FILE}."
            )
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
        l1_group: str,
        dims: List[Dict[str, Any]],
        instruction: str,
        images: List[str],
    ) -> Dict[str, Any]:
        """One case x L1-group scoring call (same granularity as full pipeline)."""
        dim_block = "\n".join(
            f"- {dim['name']}：" + "；".join(f"{k}={v}" for k, v in (dim["criteria"] or {}).items())
            for dim in dims
        )
        prompt = self.prompts["teacher"].format(
            context=self._case_context(instruction),
            l1_group=l1_group,
            dim_block=dim_block,
        )
        row: Dict[str, Any] = {"teacher": self.teacher, "l1_group": l1_group, "judgments": []}
        try:
            payload = self._call_json(self.backend, prompt, images)
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
        except Exception as exc:  # noqa: BLE001 - one failed group must not kill the case
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Teacher %s failed on L1 %s: %s", self.teacher, l1_group, exc)
        return row

    def _run_teacher(self, instruction: str, images: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self._run_teacher_group, l1_group, dims, instruction, images)
                for l1_group, dims in self.pool["l1_groups"].items()
            ]
            # Collect in submission order so artifacts stay deterministic.
            rows = [future.result() for future in futures]
        return rows

    def _synthesize_reason(self, dimension: str, judgment: Dict[str, Any]) -> Optional[str]:
        """Normalize the teacher reason via reason_model; failure falls back silently."""
        try:
            payload = self._call_json(
                self.reason_model,
                self.prompts["reason_synthesis"].format(
                    dimension=dimension,
                    score=judgment.get("score"),
                    votes=json.dumps(
                        [
                            {
                                "teacher": self.teacher,
                                "score": judgment.get("score"),
                                "reason": judgment.get("reason", ""),
                            }
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
        """Run the single-teacher chain for one case (Debate skipped by design)."""
        case_id, instruction, images = self._extract_case(sample)
        teacher_rows = self._run_teacher(instruction, images)

        by_dim: Dict[str, Dict[str, Any]] = {}
        for row in teacher_rows:
            for judgment in row.get("judgments") or []:
                by_dim[judgment["dimension"]] = {**judgment, "teacher": self.teacher}

        final_judgments: List[Dict[str, Any]] = []
        for dimension in sorted(by_dim):
            vote = by_dim[dimension]
            applicable = vote.get("applicable")
            score = vote.get("score") if applicable is not False else None
            reason = vote.get("reason", "")
            reasoning_source = "single_teacher"
            if self.synthesize_reasons and score is not None:
                synthesized = self._synthesize_reason(dimension, vote)
                if synthesized:
                    reason = synthesized
                    reasoning_source = "reason_synthesizer"
            final_judgments.append(
                {
                    "dimension": dimension,
                    "final_score": score,
                    "final_score_100": SCORE_MAP_5.get(score) if score is not None else None,
                    "final_applicable": applicable if applicable is not None else True,
                    "final_confidence": vote.get("confidence"),
                    "final_reason": reason,
                    "reasoning_source": reasoning_source,
                    "final_source": "single_teacher_score",
                    "was_debated": False,
                    "teacher_votes": [vote],
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
            "teachers": [self.teacher],
            "teacher_outputs": teacher_rows,
            "conflict_dims": [],
            "n_debated": 0,
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
        """Batch-level overall score stats (same schema as the multi-model entry)."""
        scores = [r["overall_score_100"] for r in results if r.get("overall_score_100") is not None]
        return {
            "cases": len(results),
            "cases_failed": sum(1 for r in results if r.get("error")),
            "overall_score_stats": {
                "mean": round(sum(scores) / len(scores), 2) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            },
            "debated_dims_total": 0,
        }

    def export_training_data(
        self, results: List[Dict[str, Any]], confidence_threshold: float = 0.85
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Bin final labels into SFT / uncertain training records.

        The DPO bin stays empty: single-model mode has no arbitration
        revision to pair chosen/rejected verdicts.
        """
        sft: List[Dict[str, Any]] = []
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
                    "was_debated": False,
                    "teacher_votes": judgment.get("teacher_votes"),
                    "case_overall_score_100": judgment.get("case_overall_score_100"),
                }
                confidence = judgment.get("final_confidence")
                if judgment.get("final_applicable") is False or judgment.get("final_score") is None:
                    uncertain.append(record)
                elif confidence is not None and float(confidence) >= confidence_threshold:
                    sft.append(record)
                else:
                    uncertain.append(record)
        return {"sft": sft, "dpo": [], "uncertain": uncertain}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_backends_from_config(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, ModelBackend], Optional[ModelBackend]]:
    """Build the teacher pool (plus optional reason_model) from yaml config."""
    from easydistill.cli.backend_factory import build_backend

    teachers: Dict[str, ModelBackend] = {}
    for entry in cfg.get("teachers") or []:
        name = str(entry.get("name") or entry.get("model_id") or "")
        if not name:
            raise ValueError("Each teacher entry needs 'name' (actual model name).")
        teachers[name] = build_backend(entry)
    reason_cfg = cfg.get("reason_model")
    reason_model = build_backend(reason_cfg) if reason_cfg else None
    return teachers, reason_model


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
        {"case_id": r["case_id"], "conflict_dims": []}
        for r in results
        if not r.get("error")
    ]
    debate_results = [
        {"case_id": r["case_id"], "n_conflict_dims": 0, "n_debated": 0, "dimensions": {}}
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
    parser = argparse.ArgumentParser(description="T2I 单模型（单教师）单文件评估")
    parser.add_argument(
        "--config", required=True, help="yaml：teachers/teacher/reason_model/dataset/eval 配置"
    )
    parser.add_argument(
        "--teacher", default=None, help="教师模型名（如 qwen3.7-plus）；缺省用 yaml 的 teacher"
    )
    parser.add_argument(
        "--limit-cases", type=int, default=0, help="最多评估多少个 case；0 表示全部"
    )
    parser.add_argument(
        "--synthesize-reasons", action="store_true", help="用 reason 模型规范化教师理由"
    )
    parser.add_argument(
        "--export-training", action="store_true", help="同时导出 sft/dpo/uncertain 训练数据"
    )
    args = parser.parse_args()

    import yaml

    from easydistill.cli.backend_factory import close_backends
    from easydistill.utils import save_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    teachers, reason_model = _build_backends_from_config(cfg)
    try:
        teacher = args.teacher or str(cfg.get("teacher") or "")
        if not teacher:
            raise SystemExit(f"--teacher 必填（或在 yaml 里配 teacher）；可选：{sorted(teachers)}")
        if teacher not in teachers:
            raise SystemExit(f"--teacher {teacher} 不在配置教师池：{sorted(teachers)}")

        eval_cfg = dict(cfg.get("eval") or {})
        if args.synthesize_reasons:
            eval_cfg["synthesize_reasons"] = True
        evaluator = T2ISingleModelEvaluator(
            teacher=teacher, backend=teachers[teacher], reason_model=reason_model, config=eval_cfg
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
        close_backends(*teachers.values(), reason_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
