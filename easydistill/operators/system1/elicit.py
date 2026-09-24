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

"""Elicit teacher probability distributions for typed-decision cases.

The elicit stage asks a black-box teacher every question on every case, K
samples each, and attaches the raw per-sample records to the case under
``elicitations``. Four elicitation methods are supported:

- ``single_verbalized``: one greedy sample (temperature 0) returning a JSON
  probability distribution.
- ``k_sample_freq`` (K=16, temperature 0.7): each sample returns only the best
  option key; the aggregate stage counts picks into a distribution.
- ``k_verbalized_mean`` (K=4, temperature 0.7, default): each sample returns a
  JSON distribution; the aggregate stage averages them.
- ``two_stage`` (K=4, temperature 0.7): each sample first writes an analysis of
  the options, then turns that analysis into a distribution.

Hierarchical schemas elicit the coarse (group) question first, derive the
selected group from the top coarse answers, and then elicit only that group's
fine question, mirroring how labeled hierarchical cases carry exactly one fine
question. Rows that already carry fine questions are elicited as-is.

Every finished record is streamed to ``<output_path>.partial`` as it is
produced, and ``resume_from`` / the partial file feed a resume that skips
finished samples. Parse failures are retried once with reshuffled option
order; call failures rely on the generator's own retries and the resume.
"""

import json
import logging
import math
import os
import random
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from easydistill.backends.base import ModelBackend
from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.base import Operator
from easydistill.operators.generation import TextGenerationOperator
from easydistill.utils import config_section, load_config, load_jsonl, progress

from .build import DEFAULT_TOKEN_BUDGET, check_question_budget, load_token_counter
from .prompts import (
    ELICIT_FREQ_TEMPLATE,
    ELICIT_PROBABILITY_TEMPLATE,
    TWO_STAGE_ANALYSIS_TEMPLATE,
    TWO_STAGE_DISTRIBUTION_TEMPLATE,
    build_elicit_user_text,
    build_option_lines,
)
from .protocol import (
    argmax_key,
    build_fine_question,
    fine_question_id,
    question_keys,
    serialize_state,
    validate_case_row,
    validate_schema,
)

logger = logging.getLogger(__name__)

ELICIT_METHODS = ("single_verbalized", "k_sample_freq", "k_verbalized_mean", "two_stage")

_DEFAULT_SAMPLES: Dict[str, int] = {
    "single_verbalized": 1,
    "k_sample_freq": 16,
    "k_verbalized_mean": 4,
    "two_stage": 4,
}

_DEFAULT_TEMPERATURE: Dict[str, float] = {
    "single_verbalized": 0.0,
    "k_sample_freq": 0.7,
    "k_verbalized_mean": 0.7,
    "two_stage": 0.7,
}

DEFAULT_ELICIT_MAX_TOKENS = 2048
DEFAULT_ELICIT_CHUNK_SIZE = 256

_FREQ_FENCE_OPEN_RE = re.compile(r"^```[\w-]*\s*")
_FREQ_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def _is_valid_probability(value: Any) -> bool:
    """Mirror the numeric check behind protocol.normalize_probabilities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return number >= 0 and math.isfinite(number)


def parse_probability_output(content: str, keys: List[str]) -> Optional[Dict[str, float]]:
    """Extract a JSON object covering every option key with valid numbers.

    Scans ``{`` positions and decodes with ``raw_decode`` so prose around the
    object and trailing chatter are ignored; a nested object holding all the
    keys is accepted when the outer one does not. Extra keys are dropped, and
    values are kept raw -- normalization happens in aggregation.
    """
    decoder = json.JSONDecoder()
    start = content.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and all(
            key in obj and _is_valid_probability(obj[key]) for key in keys
        ):
            return {key: float(obj[key]) for key in keys}
        start = content.find("{", start + 1)
    return None


def parse_freq_output(content: str, keys: List[str]) -> Optional[str]:
    """Parse a single-key reply into one of ``keys``, or None.

    Accepts a fenced reply (optionally quoted) holding exactly one key.
    Otherwise the reply must mention exactly one key as a whole word -- two
    mentioned keys are treated as unparsable so the sample is retried.
    """
    text = content.strip()
    if _FREQ_FENCE_OPEN_RE.match(text):
        text = _FREQ_FENCE_OPEN_RE.sub("", text, count=1)
        text = _FREQ_FENCE_CLOSE_RE.sub("", text, count=1)
        text = text.strip().strip("\"'").strip()
        if text in keys:
            return text
    mentions = [key for key in keys if re.search(rf"\b{re.escape(key)}\b", content)]
    if len(mentions) == 1:
        return mentions[0]
    return None


def _empty_elicit_record(record_id: str, question_id: str, sample: int) -> Dict[str, Any]:
    return {
        "id": record_id,
        "question_id": question_id,
        "sample": sample,
        "ok": False,
        "probabilities": None,
        "pick": None,
        "analysis": None,
        "model": None,
        "usage": None,
        "errors": [],
    }


class System1ElicitOperator(Operator[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Elicit teacher decisions for every question on every case.

    Reads build-stage case rows and returns the same rows with per-question
    ``elicitations``: ``{qid: {"method", "model", "samples": [record, ...]}}``
    where each record carries the parsed ``probabilities`` (or ``pick`` for
    k_sample_freq), the two-stage ``analysis`` when present, model/usage
    metadata, and an ``errors`` list when the sample failed.

    Configurable fields:
      - method: one of ``single_verbalized``, ``k_sample_freq``,
        ``k_verbalized_mean`` (default), ``two_stage``.
      - samples: K samples per question (default per method; forced to 1 for
        single_verbalized).
      - schema / schema_path: optional workflow schema. A hierarchical schema
        (``groups``) elicits the coarse question first, then only the selected
        group's fine question.
      - temperature: sampling temperature (default per method).
      - max_tokens / max_workers / chunk_size: generator controls.
      - model_id / retry_attempts / retry_backoff_base / retry_max_wait:
        forwarded to the generator.
      - prompt_template / freq_prompt_template / two_stage_analysis_template /
        two_stage_distribution_template: prompt overrides.
      - shuffle_options: shuffle option order per sample (default true).
      - seed: RNG seed for reproducible shuffles.
      - id_key: case field holding the row id ("id").
      - token_budget: overrides merged over the build stage defaults.
      - output_path / resume_from / resume: partial-file streaming and resume
        (resume defaults to true).
      - show_progress: whether to show a progress bar.
    """

    name = "system1_elicit"

    def __init__(self, backend: ModelBackend, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        method = self.config.get("method", "k_verbalized_mean")
        if method not in ELICIT_METHODS:
            raise ValueError(f"method must be one of {ELICIT_METHODS}, got {method!r}")
        self.method = str(method)
        samples_value = self.config.get("samples")
        self.samples = (
            int(samples_value) if samples_value is not None else _DEFAULT_SAMPLES[self.method]
        )
        if self.samples <= 0:
            raise ValueError("samples must be a positive integer.")
        if self.method == "single_verbalized" and self.samples > 1:
            logger.warning("single_verbalized elicits one greedy sample; forcing samples=1.")
            self.samples = 1
        temperature = self.config.get("temperature")
        self.temperature = (
            float(temperature) if temperature is not None else _DEFAULT_TEMPERATURE[self.method]
        )
        schema: Any = self.config.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError("system1_elicit 'schema' must be an object when set")
        if schema is None:
            schema_path = self.config.get("schema_path")
            if isinstance(schema_path, str) and schema_path:
                schema = load_config(schema_path)
        self.schema: Optional[Dict[str, Any]] = (
            validate_schema(schema) if schema is not None else None
        )
        self.hierarchical = self.schema is not None and self.schema.get("groups") is not None
        self.probability_template = str(
            self.config.get("prompt_template") or ELICIT_PROBABILITY_TEMPLATE
        )
        self.freq_template = str(
            self.config.get("freq_prompt_template") or ELICIT_FREQ_TEMPLATE
        )
        self.analysis_template = str(
            self.config.get("two_stage_analysis_template") or TWO_STAGE_ANALYSIS_TEMPLATE
        )
        self.distribution_template = str(
            self.config.get("two_stage_distribution_template")
            or TWO_STAGE_DISTRIBUTION_TEMPLATE
        )
        self.id_key = str(self.config.get("id_key") or "id")
        self.shuffle_options = bool(self.config.get("shuffle_options", True))
        seed = self.config.get("seed")
        self.rng = random.Random(int(seed)) if seed is not None else random.Random()
        chunk_size_value = self.config.get("chunk_size")
        self.chunk_size = (
            int(chunk_size_value) if chunk_size_value is not None else DEFAULT_ELICIT_CHUNK_SIZE
        )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        max_tokens_value = self.config.get("max_tokens")
        self.max_tokens = (
            int(max_tokens_value) if max_tokens_value is not None else DEFAULT_ELICIT_MAX_TOKENS
        )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        max_workers_value = self.config.get("max_workers")
        self.max_workers = int(max_workers_value) if max_workers_value is not None else 1
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        show_progress = self.config.get("show_progress")
        self.show_progress = bool(show_progress) if show_progress is not None else True
        self.token_budget = {
            **DEFAULT_TOKEN_BUDGET,
            **config_section(self.config, "token_budget"),
        }
        self.check_tokens = bool(self.token_budget.get("check", True))
        self._count_tokens: Optional[Callable[[str], int]] = None
        self._mask_token: Optional[str] = None
        output_path = self.config.get("output_path")
        self.output_path = str(output_path) if output_path else None
        resume_from = self.config.get("resume_from")
        self.resume_from = str(resume_from) if resume_from else None
        resume = self.config.get("resume")
        self.read_resume = bool(resume) if resume is not None else True
        self.partial_path = f"{self.output_path}.partial" if self.output_path else None
        self.model_id = self.config.get("model_id")
        gen_config: Dict[str, Any] = {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_workers": self.max_workers,
            "show_progress": False,  # operator-level progress instead
        }
        for key in ("retry_attempts", "retry_backoff_base", "retry_max_wait"):
            if self.config.get(key) is not None:
                gen_config[key] = self.config[key]
        self.generator = TextGenerationOperator(backend=backend, config=gen_config)

    # ---------------- resume ----------------

    def _iter_sample_records(self, rows: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Yield sample records from grouped case rows or a bare record line."""
        for row in rows:
            elicitations = row.get("elicitations")
            if isinstance(elicitations, dict):
                for payload in elicitations.values():
                    if not isinstance(payload, dict):
                        continue
                    for record in payload.get("samples") or []:
                        if isinstance(record, dict) and record.get("id"):
                            yield record
            elif row.get("id") and row.get("question_id"):
                yield row

    def _load_done(self) -> Dict[str, Dict[str, Any]]:
        done: Dict[str, Dict[str, Any]] = {}
        if not self.read_resume:
            return done
        for path in (self.resume_from, self.partial_path):
            if not path or not os.path.exists(path):
                continue
            reused = 0
            for record in self._iter_sample_records(load_jsonl(path)):
                if record.get("ok"):
                    done[str(record["id"])] = record
                    reused += 1
            logger.info("System1 elicit resume: loaded %d finished records from %s.", reused, path)
        return done

    def _append_partial(self, records: List[Dict[str, Any]]) -> None:
        if not self.partial_path or not records:
            return
        directory = os.path.dirname(self.partial_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.partial_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------------- request building ----------------

    def _template_for(self, stage: int) -> str:
        if self.method == "k_sample_freq":
            return self.freq_template
        if self.method == "two_stage":
            if stage == 1:
                return self.analysis_template
            return self.distribution_template
        return self.probability_template

    def _build_request(
        self,
        record_id: str,
        question_id: str,
        question: Dict[str, Any],
        state_text: str,
        sample: int,
        stage: int,
        option_lines: Optional[List[str]] = None,
        analysis: Optional[str] = None,
    ) -> GenerationRequest:
        if option_lines is None:
            option_lines = build_option_lines(question)
            if self.shuffle_options:
                self.rng.shuffle(option_lines)
        else:
            option_lines = list(option_lines)
        metadata: Dict[str, Any] = {
            "question_id": question_id,
            "sample": sample,
            "stage": stage,
            "question": question,
            "state_text": state_text,
            "option_lines": option_lines,
        }
        if analysis is not None:
            metadata["analysis"] = analysis
        return GenerationRequest(
            id=record_id,
            instruction=build_elicit_user_text(
                question,
                state_text,
                option_lines=option_lines,
                template=self._template_for(stage),
                analysis=analysis,
            ),
            metadata=metadata,
        )

    def _missing_samples(
        self, row_id: str, question_id: str, records_by_id: Dict[str, Dict[str, Any]]
    ) -> List[int]:
        return [
            sample
            for sample in range(self.samples)
            if not (records_by_id.get(f"{row_id}|{question_id}|{sample}") or {}).get("ok")
        ]

    def _build_initial(
        self,
        rows: List[Dict[str, Any]],
        records_by_id: Dict[str, Dict[str, Any]],
        hier: Dict[Tuple[str, str], str],
    ) -> List[GenerationRequest]:
        requests: List[GenerationRequest] = []
        stage = 1 if self.method == "two_stage" else 0
        for idx, row in enumerate(rows):
            row_id = str(row.get(self.id_key) or f"line{idx + 1}")
            parsed = validate_case_row(row)
            state_text = serialize_state(parsed["state"])
            for question_id, question in parsed["questions"].items():
                if self.hierarchical and "::" not in question_id:
                    has_fine = any(
                        other.startswith(f"{question_id}::")
                        for other in parsed["questions"]
                    )
                    if not has_fine:
                        hier[(row_id, question_id)] = state_text
                for sample in self._missing_samples(row_id, question_id, records_by_id):
                    requests.append(
                        self._build_request(
                            f"{row_id}|{question_id}|{sample}",
                            question_id,
                            question,
                            state_text,
                            sample,
                            stage,
                        )
                    )
        return requests

    def _ensure_counter(self) -> Tuple[Callable[[str], int], Optional[str]]:
        if self._count_tokens is None:
            self._count_tokens, self._mask_token = load_token_counter(self.token_budget)
        return self._count_tokens, self._mask_token

    # ---------------- hierarchical follow-ups ----------------

    def _complete_hierarchical(
        self,
        records_by_id: Dict[str, Dict[str, Any]],
        hier: Dict[Tuple[str, str], str],
        selected_fine: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]],
    ) -> List[GenerationRequest]:
        schema = self.schema
        if not self.hierarchical or schema is None:
            return []
        requests: List[GenerationRequest] = []
        stage = 1 if self.method == "two_stage" else 0
        for key in list(hier):
            row_id, coarse_id = key
            state_text = hier[key]
            coarse_records = [
                records_by_id.get(f"{row_id}|{coarse_id}|{sample}")
                for sample in range(self.samples)
            ]
            if not any(record and record.get("ok") for record in coarse_records):
                continue
            del hier[key]
            usable = [
                record
                for record in coarse_records
                if record and record.get("ok") and self._usable(record)
            ]
            if not usable:
                logger.warning(
                    "System1 elicit: no usable coarse answers for case %r; "
                    "skipping its fine question.",
                    row_id,
                )
                continue
            try:
                group = self._top_group(usable)
                fine_id = fine_question_id(coarse_id, group)
                fine_question = build_fine_question(schema, group)
                if self.check_tokens:
                    count_tokens, mask_token = self._ensure_counter()
                    check_question_budget(
                        fine_question,
                        state_text,
                        self.token_budget,
                        count_tokens,
                        mask_token,
                        f"case {row_id!r} question {fine_id!r}",
                    )
            except (ValueError, KeyError) as exc:
                logger.warning("System1 elicit: %s; skipping this fine question.", exc)
                continue
            selected_fine[key] = (fine_id, fine_question)
            for sample in self._missing_samples(row_id, fine_id, records_by_id):
                requests.append(
                    self._build_request(
                        f"{row_id}|{fine_id}|{sample}",
                        fine_id,
                        fine_question,
                        state_text,
                        sample,
                        stage,
                    )
                )
        return requests

    def _run_requests(
        self, requests: List[GenerationRequest], desc: str = "Eliciting system1 decisions"
    ) -> Dict[str, GenerationResult]:
        results: Dict[str, GenerationResult] = {}
        chunks = [
            requests[i : i + self.chunk_size]
            for i in range(0, len(requests), self.chunk_size)
        ]
        for chunk in progress(
            chunks, enabled=self.show_progress, total=len(chunks), desc=desc
        ):
            for result in self.generator.run(chunk):
                results[result.request.id or ""] = result
        return results

    def _record_from_result(
        self, request: GenerationRequest, result: Optional[GenerationResult]
    ) -> Dict[str, Any]:
        meta = request.metadata
        record = _empty_elicit_record(request.id or "", meta["question_id"], meta["sample"])
        if result is None:
            record["errors"] = ["elicit_call_failed"]
            return record
        record["model"] = result.model
        record["usage"] = result.usage
        content = result.response or ""
        keys = question_keys(meta["question"])
        parsed = (
            parse_freq_output(content, keys)
            if self.method == "k_sample_freq"
            else parse_probability_output(content, keys)
        )
        if parsed is None:
            record["errors"] = ["parse_failed"]
            record["raw_content"] = content[:500]
            return record
        if self.method == "k_sample_freq":
            record["pick"] = parsed
        else:
            record["probabilities"] = parsed
        record["ok"] = True
        record["analysis"] = meta.get("analysis")
        return record

    def _retry_request(self, request: GenerationRequest) -> GenerationRequest:
        """Rebuild a parse-failed request; stage-2 keeps its option order."""
        meta = request.metadata
        option_lines = meta["option_lines"] if meta["stage"] == 2 else None
        return self._build_request(
            request.id or "",
            meta["question_id"],
            meta["question"],
            meta["state_text"],
            meta["sample"],
            meta["stage"],
            option_lines=option_lines,
            analysis=meta.get("analysis"),
        )

    # ---------------- round execution ----------------

    def _finalize_round(
        self,
        round_requests: List[GenerationRequest],
        round_results: Dict[str, GenerationResult],
    ) -> List[Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        retry_requests: List[GenerationRequest] = []
        for request in round_requests:
            if request.metadata.get("stage") == 1:
                continue
            record = self._record_from_result(request, round_results.get(request.id or ""))
            if record["errors"] == ["parse_failed"]:
                retry_requests.append(self._retry_request(request))
            records[request.id or ""] = record
        if retry_requests:
            retry_results = self._run_requests(
                retry_requests, desc="Eliciting system1 decisions (retry)"
            )
            for request in retry_requests:
                records[request.id or ""] = self._record_from_result(
                    request, retry_results.get(request.id or "")
                )
        return list(records.values())

    def _stage2_followups(
        self,
        round_requests: List[GenerationRequest],
        round_results: Dict[str, GenerationResult],
    ) -> Tuple[List[GenerationRequest], List[Dict[str, Any]]]:
        if self.method != "two_stage":
            return [], []
        followups: List[GenerationRequest] = []
        failed: List[Dict[str, Any]] = []
        for request in round_requests:
            meta = request.metadata
            if meta.get("stage") != 1:
                continue
            result = round_results.get(request.id or "")
            analysis = (result.response or "").strip() if result is not None else ""
            if not analysis:
                record = _empty_elicit_record(
                    request.id or "", meta["question_id"], meta["sample"]
                )
                record["errors"] = [
                    "empty_analysis" if result is not None else "elicit_call_failed"
                ]
                if result is not None:
                    record["model"] = result.model
                    record["usage"] = result.usage
                failed.append(record)
                continue
            followups.append(
                self._build_request(
                    request.id or "",
                    meta["question_id"],
                    meta["question"],
                    meta["state_text"],
                    meta["sample"],
                    2,
                    option_lines=meta["option_lines"],
                    analysis=analysis,
                )
            )
        return followups, failed

    def _usable(self, record: Dict[str, Any]) -> bool:
        if self.method == "k_sample_freq":
            return bool(record.get("pick"))
        probabilities = record.get("probabilities")
        return isinstance(probabilities, dict) and (
            sum(float(value) for value in probabilities.values()) > 0
        )

    def _top_group(self, records: List[Dict[str, Any]]) -> str:
        scores: Dict[str, float] = {}
        for record in records:
            if self.method == "k_sample_freq":
                pick = record.get("pick")
                if pick:
                    scores[str(pick)] = scores.get(str(pick), 0.0) + 1.0
                continue
            probabilities = record.get("probabilities")
            if not isinstance(probabilities, dict):
                continue
            total = sum(float(value) for value in probabilities.values())
            if total <= 0:
                continue
            for group, value in probabilities.items():
                scores[str(group)] = scores.get(str(group), 0.0) + float(value) / total
        if not scores:
            raise ValueError("no usable coarse answers to derive a group from")
        return argmax_key(scores)

    # ---------------- grouping ----------------

    def _regroup(
        self,
        row: Dict[str, Any],
        idx: int,
        records_by_id: Dict[str, Dict[str, Any]],
        fine_by_row: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        row_id = str(row.get(self.id_key) or f"line{idx + 1}")
        parsed = validate_case_row(row)
        questions = dict(parsed["questions"])
        for fine_id, fine_question in fine_by_row.get(row_id, []):
            questions[fine_id] = fine_question
        elicitations = dict(row.get("elicitations") or {})
        for question_id in questions:
            if question_id in elicitations:
                continue
            samples = [
                records_by_id.get(f"{row_id}|{question_id}|{sample}")
                for sample in range(self.samples)
            ]
            if not any(samples):
                continue
            model = next(
                (record["model"] for record in samples if record and record.get("model")),
                None,
            )
            elicitations[question_id] = {
                "method": self.method,
                "model": model,
                "samples": samples,
            }
        out = dict(row)
        out["questions"] = questions
        out["elicitations"] = elicitations
        return out

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.partial_path and os.path.exists(self.partial_path) and not self.read_resume:
            os.remove(self.partial_path)
            logger.info(
                "System1 elicit: discarded stale partial file %s (resume disabled).",
                self.partial_path,
            )
        records_by_id = self._load_done()
        hier: Dict[Tuple[str, str], str] = {}
        selected_fine: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}
        round_requests = self._build_initial(data, records_by_id, hier)
        while True:
            round_requests.extend(
                self._complete_hierarchical(records_by_id, hier, selected_fine)
            )
            if not round_requests:
                break
            round_results = self._run_requests(round_requests)
            round_records = self._finalize_round(round_requests, round_results)
            for record in round_records:
                records_by_id[str(record["id"])] = record
            self._append_partial(round_records)
            followups, failed = self._stage2_followups(round_requests, round_results)
            for record in failed:
                records_by_id[str(record["id"])] = record
            self._append_partial(failed)
            round_requests = followups
        fine_by_row: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for (row_id, _coarse_id), (fine_id, fine_question) in selected_fine.items():
            fine_by_row.setdefault(row_id, []).append((fine_id, fine_question))
        output: List[Dict[str, Any]] = []
        for idx, row in enumerate(
            progress(
                data,
                enabled=self.show_progress,
                total=len(data),
                desc="Grouping system1 elicitations",
            )
        ):
            output.append(self._regroup(row, idx, records_by_id, fine_by_row))
        n_ok = sum(1 for record in records_by_id.values() if record.get("ok"))
        n_failed = sum(1 for record in records_by_id.values() if not record.get("ok"))
        n_fine = sum(len(fine) for fine in fine_by_row.values())
        logger.info(
            "System1 elicit (%s, K=%d) finished: %d samples ok, %d failed, "
            "%d fine questions synthesized.",
            self.method,
            self.samples,
            n_ok,
            n_failed,
            n_fine,
        )
        return output
