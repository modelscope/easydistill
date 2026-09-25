# System-1 Distillation

System-1 distillation produces RLCD (Reinforcement Learning from Contrastive Distillation) training data for a student model's typed-decision head. A teacher model is asked to express probability distributions over decision options for each case; the samples are aggregated into soft labels and packaged as a training-ready dataset.

## When to use this pipeline

Use this pipeline when you want to distill a teacher model's **fast, intuitive decisions** (system-1 thinking) into a student model. It is designed for cases where:

- You have a set of cases, each carrying a state (e.g., an SMS message, a customer-service ticket) and one or more typed-decision questions (multiple-choice, score-scale, or yes/no-unless).
- You want soft probability labels from a teacher model, not hard one-hot labels.
- You want to train a student model's decision head with GRPO-style policy gradients plus soft cross-entropy guidance.

The pipeline automatically:

1. Builds typed-decision case rows from raw rows and a schema (local, no LLM).
2. Elicits teacher probability distributions for every case question (K samples per question).
3. Aggregates elicitation samples into teacher labels with consistency checking and anti-one-hot shrinkage (local, no LLM).
4. Emits training-ready case rows with nested gold labels (local, no LLM).

## Pipeline stages

| Stage | Required? | Purpose |
|---|---|---|
| `build_cases` | Required (first) | Build typed-decision case rows from raw rows + schema. |
| `elicit` | Required | Elicit teacher probability distributions for every case question. |
| `aggregate` | Required | Aggregate elicitation samples into teacher labels. |
| `build_dataset` | Required (last) | Emit training-ready case rows with nested gold labels. |

All stages support stage-level resume: if a stage's `output_path` file already exists, that stage is skipped and its output feeds the remaining stages. The `elicit` stage additionally supports row-level checkpointing.

## Quick start

```bash
# 1. Set credentials for PAI-Token
export PAI_TOKEN_API_KEY=sk-pai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export PAI_TOKEN_BASE_URL=https://cn-beijing.pai-token.aliyuncs.com/v1

# 2. Run the full pipeline
easydistill --config configs/system1/system1_distill_pai_token.yaml
```

The shipped config distills 8 SMS spam-classification examples using `qwen3.7-max` as the teacher. The final training dataset is written to `outputs/system1_train_sms_pai_token.jsonl`.

## Input format

### Raw rows (JSONL)

Each row is a flat object with an `id`, a label field (specified by the schema's `label_field`, default `"label"`), and any state fields:

```json
{"id": "sms-001", "label": "ham", "text": "Hey, running late. Grab a table for 7:30 and I will meet you there."}
```

| Field | Required? | Description |
|---|---|---|
| `id` | Required | Stable case identifier, carried through every stage. |
| `<label_field>` | Required | The gold label (e.g., `"ham"` or `"spam"`). Field name comes from the schema. |
| *other fields* | Required | All non-id, non-label fields become the case `state`. |

Pre-built questions can also be supplied via a `questions` field (dict); if present, the questions are passed through as-is and the schema's question definition is not used.

### Schema (YAML)

```yaml
name: sms_spam
question: spam
instructions: "Classify the SMS message."
options:
  spam: "Unsolicited commercial or fraudulent message"
  ham: "Legitimate personal, service, or transactional message"
```

| Field | Required? | Description |
|---|---|---|
| `name` | Required | Workflow name, stored as `workflow` in case rows. |
| `question` | Required | Question key (e.g., `"spam"`). |
| `instructions` | Required | Instructions shown to the teacher. |
| `options` | Required (flat) | Flat option dict: `{key: description}`. Default `type` is `"choice"`. |
| `groups` | Required (hierarchical) | Hierarchical option groups (banking77-style). |
| `label_field` | Optional | Raw-row label field name (default `"label"`). |
| `id_field` | Optional | Raw-row id field name (default `"id"`). |
| `type` | Optional | Question type: `"choice"`, `"score"`, or `"noul"` (default `"choice"`). |

## Config reference

```yaml
job_type: system1_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  base_url: ${PAI_TOKEN_BASE_URL}
  model_id: qwen3.7-max

dataset:
  input_path: examples/system1_sms_raw.jsonl
  output_path: outputs/system1_train_sms_pai_token.jsonl

system1:
  schema_path: examples/system1_sms_schema.yaml
  variant: distill              # labeled | distill | mixed
  method: k_verbalized_mean     # single_verbalized | k_sample_freq | k_verbalized_mean | two_stage
  samples: 4                    # K samples per question
  temperature: 0.7
  max_tokens: 2048
  max_workers: 8
  seed: 7
  shuffle_options: true         # shuffle option order per call (anti-position-bias)
  show_progress: true
  resume: true                  # stage- and row-level resume
  consistency_threshold: 0.6    # drop questions with top-1 consistency < threshold
  shrinkage: 0.02               # anti-one-hot shrinkage (0 disables)
  alpha: 0.7                    # gold weight for the mixed variant

pipeline:
  - stage: build_cases
    output_path: outputs/system1_cases_sms_pai_token.jsonl
  - stage: elicit
    config:
      max_workers: 8
      resume: true
    output_path: outputs/system1_elicit_sms_pai_token.jsonl
  - stage: aggregate
    config:
      consistency_threshold: 0.6
    output_path: outputs/system1_labels_sms_pai_token.jsonl
  - stage: build_dataset
    config:
      variant: distill
    output_path: outputs/system1_train_sms_pai_token.jsonl
```

### Key descriptions

| Key | Default | Description |
|---|---|---|
| `system1.schema_path` | — | Path to the schema YAML. |
| `system1.variant` | `distill` | `labeled`: pass through gold labels (no teacher). `distill`: pure teacher labels. `mixed`: alpha-blend of gold and teacher. |
| `system1.method` | `k_verbalized_mean` | Elicitation method. `single_verbalized`: 1 call, T=0. `k_sample_freq`: K calls, frequency aggregation. `k_verbalized_mean`: K calls, mean of verbalized probabilities (recommended). `two_stage`: coarse-then-fine. |
| `system1.samples` | 1 / 16 / 4 / 4 | K — number of teacher samples per question (per method: single / k_sample_freq / k_verbalized_mean / two_stage). |
| `system1.temperature` | 0 / 0.7 / 0.7 / 0.7 | Sampling temperature for teacher calls (per method). |
| `system1.max_tokens` | 2048 | Max tokens per teacher response. |
| `system1.max_workers` | 1 | Concurrent API calls during elicit (shipped config uses 8). |
| `system1.seed` | random | Random seed for option shuffling (shipped config uses 7). |
| `system1.shuffle_options` | `true` | Shuffle option order per call to cancel teacher position bias. |
| `system1.resume` | `true` | Stage-level resume (skip stages whose output exists) + row-level checkpointing in elicit. |
| `system1.consistency_threshold` | 0.6 | Drop questions whose top-1 consistency across samples is below this threshold. |
| `system1.shrinkage` | 0.02 | Shrink aggregated probabilities toward uniform to prevent one-hot collapse (0 disables). |
| `system1.alpha` | 0.7 | Gold weight for the `mixed` variant (teacher weight = 1 - alpha). |
| `pipeline` | — | List of stage configs. Each entry has `stage`, optional `config` (per-stage overrides merged onto `system1`), and `output_path`. |

## Stage-by-stage I/O

The following shows the same row (`sms-001`, a legitimate SMS) flowing through all four stages, captured from a live `qwen3.7-max` run.

### Stage 1 — build_cases (input → cases)

**Input** (raw row):

```json
{"id": "sms-001", "label": "ham", "text": "Hey, running late. Grab a table for 7:30 and I will meet you there."}
```

**Output** (case row):

```json
{
  "id": "sms-001",
  "workflow": "sms_spam",
  "state": {"text": "Hey, running late. Grab a table for 7:30 and I will meet you there."},
  "questions": {
    "spam": {
      "type": "choice",
      "instructions": "Classify the SMS message.",
      "criteria": {
        "spam": "Unsolicited commercial or fraudulent message",
        "ham": "Legitimate personal, service, or transactional message"
      }
    }
  },
  "gold": {"spam": {"spam": 0.0, "ham": 1.0}},
  "source": {"id": "sms-001", "label": "ham", "text": "..."}
}
```

The raw row's `label` ("ham") is expanded into a one-hot gold distribution. The `state` is auto-derived from all non-id, non-label fields. The original row is preserved in `source`.

### Stage 2 — elicit (cases → elicitation records)

Each question gets K=4 teacher samples. Each sample is a separate API call with shuffled options. Here is one of the four samples for `sms-001`:

```json
{
  "id": "sms-001|spam|0",
  "question_id": "spam",
  "sample": 0,
  "ok": true,
  "probabilities": {"spam": 0.0, "ham": 1.0},
  "model": "qwen3.7-max",
  "usage": {
    "prompt_tokens": 126,
    "completion_tokens": 175,
    "total_tokens": 301,
    "completion_tokens_details": {"reasoning_tokens": 150}
  },
  "errors": []
}
```

All 4 samples for this row returned `{"spam": 0.0, "ham": 1.0}` — the teacher consistently and correctly identified this as ham. The `elicitations` field is added to each case row:

```json
{
  "elicitations": {
    "spam": {
      "method": "k_verbalized_mean",
      "model": "qwen3.7-max",
      "samples": [ /* 4 sample objects as above */ ]
    }
  }
}
```

### Stage 3 — aggregate (elicitation records → labels)

The 4 samples are aggregated into a single teacher label. Shrinkage (0.02) pulls the distribution slightly away from pure one-hot:

```json
{
  "teacher": {
    "spam": {
      "ok": true,
      "method": "k_verbalized_mean",
      "model": "qwen3.7-max",
      "probabilities": {"spam": 0.0098, "ham": 0.9902},
      "label": "ham",
      "consistency": 1.0,
      "samples_used": 4,
      "samples_failed": 0,
      "gold_tv": 0.0098,
      "gold_kl": 0.0099
    }
  }
}
```

- `consistency`: 1.0 means all 4 samples agreed on the top-1 option.
- `gold_tv` / `gold_kl`: total-variation and KL divergence from the gold one-hot distribution.
- `label`: argmax of the aggregated probabilities.

### Stage 4 — build_dataset (labels → training dataset)

The final training row drops intermediate fields and restructures `gold` into the nested format the training entry expects:

```json
{
  "id": "sms-001",
  "workflow": "sms_spam",
  "state": {"text": "Hey, running late. Grab a table for 7:30 and I will meet you there."},
  "questions": {
    "spam": {
      "type": "choice",
      "instructions": "Classify the SMS message.",
      "criteria": {
        "spam": "Unsolicited commercial or fraudulent message",
        "ham": "Legitimate personal, service, or transactional message"
      }
    }
  },
  "gold": {
    "spam": {
      "probabilities": {"spam": 0.0098, "ham": 0.9902},
      "label": "ham"
    }
  }
}
```

## Fine-tuning

After the pipeline produces the training dataset, fine-tune Laya's typed-decision head using the shipped training entry:

```bash
python examples/system1_train_entry.py \
  --input outputs/system1_train_sms_pai_token.jsonl \
  --output-dir outputs/system1_laya_finetuned \
  --nproc 2
```

The parent process downloads the Laya model from the Hugging Face Hub, preprocesses the dataset into tokenized training items, then re-executes itself under `torch.distributed.run`. Child processes perform DDP training with GRPO-style policy gradients plus soft cross-entropy guidance, and rank 0 exports the fine-tuned model with fitted calibration temperatures.

> **Requires GPUs and the `laya` research package.** Run this on a multi-GPU machine (2xT4 or better). The `--model-dir` flag reuses a local Laya snapshot instead of downloading.

### Training entry flags

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Path to the `system1_build_dataset` JSONL output. |
| `--output-dir` | `outputs/system1_laya_finetuned` | Directory for the exported model. |
| `--model-dir` | (auto-download) | Reuse a local Laya snapshot instead of downloading. |
| `--items` | `<output-dir>/train_items.pt` | Preprocessed training items cache. |
| `--nproc` | 2 | Number of GPUs for DDP. |

## Standalone stages

Every stage is also exposed as a standalone `job_type` for debugging or resuming from an intermediate JSONL:

| `job_type` | Purpose | Needs backend? |
|---|---|---|
| `system1_build_cases` | Build case rows from raw rows + schema. | No |
| `system1_elicit` | Elicit teacher distributions. | Yes |
| `system1_aggregate` | Aggregate elicitation samples into labels. | No |
| `system1_build_dataset` | Emit training-ready case rows. | No |

The `labeled` variant of `system1_distill` skips `elicit` and `aggregate` (gold labels pass through directly) and never calls the teacher.
