# Augmented Instruction Distillation

The `augmented_instruct_distill` pipeline chains instruction synthesis stages and ends with `instruct_distill` to produce an SFT dataset in one run. It is useful when you want to amplify a small seed instruction set before distilling teacher responses.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## Pipeline stages

| Stage | Purpose |
|---|---|
| `instruction_expansion` | Generate new instructions from seed instructions using in-context examples. |
| `instruction_refinement` | Rewrite or optimize the expanded instructions. |
| `instruct_distill` | Generate teacher responses and build the final SFT dataset. |

You can also add `instruction_response_extraction` or `instruction_balance` between synthesis stages if your input data and workflow require them.

The pipeline must end with `instruct_distill`.

## Config schema

### Top-level fields

```yaml
job_type: augmented_instruct_distill

backend:
  type: pai_token          # or pai_eas
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant. Provide concise and accurate answers."
  temperature: 0.7
  max_tokens: 2048

pipeline:
  - stage: instruction_expansion
    config: { ... }
    output_path: outputs/augmented_stage1_expanded.jsonl

  - stage: instruction_refinement
    config: { ... }
    output_path: outputs/augmented_stage2_refined.jsonl

  - stage: instruct_distill
    config: { ... }

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/augmented_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### `generation`

Default generation parameters applied to all generation stages unless overridden in a stage `config`.

### `dataset`

- `input_path`: JSONL file with seed instructions (`instruction` field).
- `output_path`: final SFT dataset path.
- `skip_empty`, `min_length`, `max_length`: SFT response filters.

### Stage configs

Each stage's `config` supports the same fields as the corresponding standalone job type. See:

- [instruction_distillation.md](instruction_distillation.md) for `instruction_expansion`, `instruction_refinement`, and `instruct_distill`.
- [instruction_balancing.md](instruction_balancing.md) for `instruction_balance`.

## Stage data formats

The pipeline writes intermediate JSONL files between stages:

| Stage | Output fields | Notes |
|---|---|---|
| `instruction_expansion` | `instruction` | One expanded instruction per row. |
| `instruction_refinement` | `instruction` | Refined version of the expanded instructions. |
| `instruct_distill` | `messages`, `metadata` | Final OpenAI/ShareGPT SFT messages. |

## Example

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/augmented_instruct_distill_pai_token.yaml
```

A PAI-EAS equivalent is provided as [`configs/pipeline/augmented_instruct_distill_pai_eas.yaml`](../configs/pipeline/augmented_instruct_distill_pai_eas.yaml).
