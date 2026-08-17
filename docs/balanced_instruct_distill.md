# Balanced Instruction Distillation

The `balanced_instruct_distill` pipeline synthesizes new instructions, classifies and resamples them to a target category distribution, generates teacher responses, and builds an SFT dataset. It is useful when you want the final dataset to follow a balanced curriculum instead of reflecting the raw distribution of generated instructions.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## When to use this pipeline

Use this pipeline when you want to:

1. Expand a small seed-instruction set into new instructions.
2. Classify the synthesized instructions by task or domain.
3. Resample them to a target distribution so no category dominates.
4. Generate teacher responses for the balanced set.
5. Build the final SFT dataset in OpenAI/ShareGPT message format.

## Pipeline stages

| Stage | Required? | Purpose |
|---|---|---|
| `instruction_expansion` | Optional | Generate new instructions from seed instructions. |
| `instruction_response_extraction` | Optional | Extract `(instruction, response)` pairs from raw text. |
| `instruction_refinement` | Optional | Rewrite/optimize instructions before balancing. |
| `instruction_balance` | Required | Classify instructions by task/domain and resample to a target distribution. |
| `generate` | Required | Call the teacher model to produce responses. |
| `instruct_eval` | Optional | Run the LLM-as-judge evaluator. |
| `quality_filter` | Optional | Drop rows that do not meet score thresholds. |
| `build_sft` | Required (last) | Convert the remaining rows into SFT messages. |

The last stage must always be `build_sft`.

## Config schema

```yaml
job_type: balanced_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant. Provide concise and accurate answers."
  temperature: 0.7
  max_tokens: 2048

eval:
  prompts_file: configs/prompts/default_eval_prompts.yaml
  metrics:
    - informativeness
    - helpfulness
    - generalization
    - correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4

pipeline:
  - stage: instruction_expansion
    config:
      prompt_template_file: configs/prompts/expansion_prompt.txt
      num_in_context_samples: 2
      num_output_samples: 3
      temperature: 0.8
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/balanced_stage1_expanded.jsonl

  - stage: instruction_balance
    config:
      instruction_key: instruction
      category_key: category
      categories: ["General"]
      target_distribution:
        General: 1.0
      category_prompt: "Classify the instruction below as General. Wrap the category in <answer>...</answer>.\nInstruction: {instruction}"
      max_workers: 4
      show_progress: true
      seed: 42
      temperature: 0.0
      max_tokens: 2048
    output_path: outputs/balanced_stage2_balanced.jsonl

  - stage: generate
    config:
      show_progress: true
      max_workers: 3
    output_path: outputs/balanced_stage3_generated.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/balanced_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### Top-level sections

- `job_type`: must be `balanced_instruct_distill`.
- `backend`: any supported backend (`pai_token`, `pai_eas`).
- `generation`: default generation parameters used by `generate` and `build_sft`.
- `eval`: default evaluator parameters. `metrics` lists the judge metrics to compute.
- `pipeline`: ordered list of stages.
- `dataset`: `input_path`, `output_path`, and SFT filter parameters (`skip_empty`, `min_length`, `max_length`).

### `instruction_balance` config

See [instruction_balancing.md](instruction_balancing.md) for the full schema. The example above uses a single `General` category; replace `categories` and `target_distribution` with your own task/domain labels.

## Stage data formats

The pipeline writes intermediate JSONL files between stages. Each row preserves the fields needed by the next stage.

| Stage | Output fields | Notes |
|---|---|---|
| `instruction_expansion` | `instruction` | One expanded instruction per row (optional first stage). |
| `instruction_refinement` | `instruction` | Refined version of the previous stage (optional). |
| `instruction_balance` | `instruction`, `category` | Original instruction plus assigned category. |
| `generate` | `instruction`, `output` | Teacher-generated response. |
| `instruct_eval` | `instruction`, `output`, `<metrics>` | Original fields plus per-metric scores. |
| `quality_filter` | same as `instruct_eval` | Only rows passing thresholds are kept. |
| `build_sft` | `messages`, `metadata` | Final OpenAI/ShareGPT SFT messages. |

## Running the pipeline

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/balanced_instruct_distill_pai_token.yaml
```

A PAI-EAS equivalent is provided as [`configs/pipeline/balanced_instruct_distill_pai_eas.yaml`](../configs/pipeline/balanced_instruct_distill_pai_eas.yaml).
