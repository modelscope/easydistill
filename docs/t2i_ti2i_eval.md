# T2I / TI2I Evaluation (Four Independent Single-File Entries)

This module implements T2I (text-to-image) / TI2I (image editing) evaluation
as **four fully independent, self-contained single-file modules** under
`easydistill/eval/`, following the same evaluator style as the other modules in
that package (`cot.py` / `mm.py`, etc.). The four files have zero dependencies
on each other and on any other T2I/TI2I code directory, so each one can be
isolated and run on its own; at runtime they only need the two frozen dimension
pool JSONs and the entry YAMLs under `configs/eval/t2i_ti2i/`.

## The Four Entries

| Entry | File | Description |
|---|---|---|
| T2I multi-model | `easydistill/eval/t2i_multi_model.py` | multi-teacher scoring + conflict detection + 3-step Debate + synthesis + overall score |
| T2I single-model | `easydistill/eval/t2i_single_model.py` | one designated teacher; no cross-teacher conflict → no Debate, same artifact schema |
| TI2I multi-model | `easydistill/eval/ti2i_multi_model.py` | same as T2I multi-model with the image-editing pool/inputs |
| TI2I single-model | `easydistill/eval/ti2i_single_model.py` | same as T2I single-model with the image-editing pool/inputs |

System overview:

| | T2I | TI2I |
|---|---|---|
| Dimension pool | 60 L3 dims (`t2i_dimensions.json`) | 38 L3 dims (`ti2i_dimensions.json`) |
| Scoring | 0-4 five levels | 0-4 five levels |
| Call granularity | case × teacher × L1 group | case × teacher × L1 group |
| Seed schema | `prompt_id` / `prompt` / `image` | `case_id` / `instruction` / `before_image` / `after_image` / `reference_images` |

## Model Roles (multi-model entries)

Scoring, arbitration and synthesis are three roles; Debate and the final data
synthesis are **two separate models and two separate steps**:

| Role | Model (current config) | Responsibility |
|---|---|---|
| Scoring teachers | `qwen3.7-plus` / `qwen3.5-plus` / `kimi-k2.6` (PAI Token) | score L3 dims (0-4 + Chinese reason) per L1 group, independently |
| Debate arbiter (`arbiter`) | `kimi-k3` (PAI EAS) | only Step1 review → Step2 prosecution/defense → Step3 verdict on conflicted dims |
| Final synthesis (`reason_model`) | `kimi-k3` (PAI EAS, separate config section) | only normalizes majority-vote reasons (final data synthesis) |

Notes:

- Multi-model entries use **no OCR/VLM text-recognition tool teacher and no
  prompt-keyword activation**; every L3 dim is judged directly by the model
  teachers.
- The `teacher` field in all artifacts carries the actual model name.
- Single-model entries need one teacher backend plus an optional
  `reason_model`; there is no Debate.
- A dimension enters Debate when the cross-teacher score spread >=
  `conflict_threshold` (default 2); `max_debate_dims` caps arbitrated dims per
  case (largest spreads win the slots when over the cap).
- Every model call has built-in light retries (default 2, covering 503 /
  timeouts / malformed JSON).

## Quick Start

```bash
# 0) Env vars: PAI_TOKEN_API_KEY / PAI_TOKEN_BASE_URL / EAS_ENDPOINT_URL / EAS_TOKEN

# 1) T2I multi-model (multi-teacher Debate)
python -m easydistill.eval.t2i_multi_model \
  --config configs/eval/t2i_ti2i/t2i_multi_model_pai_token.yaml

# 2) T2I single-model (default teacher qwen3.7-plus, overridable via --teacher)
python -m easydistill.eval.t2i_single_model \
  --config configs/eval/t2i_ti2i/t2i_single_model_pai_token.yaml --teacher qwen3.7-plus

# 3) TI2I entries work the same way
python -m easydistill.eval.ti2i_multi_model  --config configs/eval/t2i_ti2i/ti2i_multi_model_pai_token.yaml
python -m easydistill.eval.ti2i_single_model --config configs/eval/t2i_ti2i/ti2i_single_model_pai_token.yaml --teacher qwen3.7-plus
```

Each entry ships a `*_pai_token.yaml` / `*_pai_eas.yaml` config pair (same
convention as the rest of `configs/eval/`); backends can also be mixed per
role inside one config.

Common flags: `--limit-cases N` (cap evaluated cases), `--synthesize-reasons`
(normalize majority reasons via the reason model), `--export-training` (export
sft/dpo/uncertain training data).

## Outputs

Each entry writes the same artifact set as the original orchestrator into
`dataset.output_dir`:

| File | Description |
|---|---|
| `teacher_outputs.jsonl` | raw judgments per teacher × L1 group; `teacher` is the actual model name |
| `conflict_report.jsonl` | conflicted dims per case (always empty in single-model mode) |
| `debate_results.jsonl` | Debate results with full step1/step2/step3 records per arbitrated dim |
| `final_labels.jsonl` | case-level final labels: `final_judgments` + `overall_score_100` + `overall` audit block |
| `final_judgments.jsonl` | dimension-level final verdicts, each carrying `case_overall_score_100` |
| `final_labels_summary.json` | batch-level `overall_score_stats` |
| `sft_data.jsonl` / `dpo_data.jsonl` / `uncertain_data.jsonl` | written with `--export-training`; Debate revisions become DPO pairs |

## Overall Score Rule

The overall score is computed only from the final L3 verdicts, with no extra
model calls:

- **T2I**: 0-4 mapped to 0/25/50/75/100; NA excluded from the mean; the Safety
  L1 group is excluded from the total; `Safety Compliance = 0` vetoes the total
  to 0.
- **TI2I**: simple baseline aligned with T2I; 0-4 mapped to 0/25/50/75/100;
  equal-weight mean over all applicable scored L3 dims; no edit-family /
  L1 / item weights and no Instruction Following gate capping.

Three score layers: case `overall_score_100` → L1 sub-scores
`overall.l1_subscores_100` → L3 scores `final_judgments[].final_score_100`
(raw 0-4 in `final_score`).

## Differences from the Full Pipeline

These entries keep every evaluation feature (teacher scoring, conflict
detection, 3-step Debate, two-model synthesis, overall score, training export)
and only drops engineering conveniences: resume / incremental checkpointing and
batch isolation knobs; per-call retries are built in.
