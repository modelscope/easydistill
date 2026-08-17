# Advanced Instruction Distillation

The `advanced_instruct_distill` pipeline connects all instruction-distillation features—expansion, refinement, teacher generation, LLM-as-judge evaluation, and quality filtering—into one end-to-end flow that keeps only the best data for supervised fine-tuning (SFT).

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## When to use this pipeline

Use this pipeline when you want to go from a small seed-instruction set to a curated, high-quality SFT dataset with one command. The pipeline automatically:

1. Expands seed instructions into new, diverse instructions.
2. Refines the expanded instructions for clarity and difficulty.
3. Generates teacher responses for every instruction.
4. Evaluates each `(instruction, response)` pair with an LLM judge.
5. Filters out low-quality rows based on the judge scores.
6. Builds the final SFT dataset in OpenAI/ShareGPT message format.

## Pipeline stages

| Stage | Required? | Purpose |
|---|---|---|
| `instruction_expansion` | Optional | Generate new instructions from seed instructions. |
| `instruction_refinement` | Optional | Rewrite/optimize instructions. |
| `instruction_response_extraction` | Optional | Extract `(instruction, response)` pairs from raw text. |
| `instruction_balance` | Optional | Classify instructions by task/domain and resample to a target distribution. |
| `generate` | Required | Call the teacher model to produce responses. |
| `instruct_eval` | Optional | Run the LLM-as-judge evaluator. |
| `quality_filter` | Optional | Drop rows that do not meet score thresholds. |
| `build_sft` | Required (last) | Convert the remaining rows into SFT messages. |

The last stage must always be `build_sft`.

## Config schema

```yaml
job_type: advanced_instruct_distill

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
    output_path: outputs/advanced_stage1_expanded.jsonl

  - stage: instruction_refinement
    config:
      prompt_template_file: configs/prompts/refinement_prompt.txt
      temperature: 0.7
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/advanced_stage2_refined.jsonl

  - stage: generate
    config:
      show_progress: true
      max_workers: 3
    output_path: outputs/advanced_stage3_generated.jsonl

  - stage: instruct_eval
    config:
      show_progress: true
      max_workers: 4
    output_path: outputs/advanced_stage4_evaluated.jsonl

  - stage: quality_filter
    config:
      min_scores:
        informativeness: 6
        helpfulness: 6
        generalization: 4
        correctness: true
      require_all_metrics: true
      keep_top_ratio: 0.7
    output_path: outputs/advanced_stage5_filtered.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/advanced_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### Top-level sections

- `job_type`: must be `advanced_instruct_distill`.
- `backend`: any supported backend (`pai_token`, `pai_eas`).
- `generation`: default generation parameters used by `generate` and `build_sft`.
- `eval`: default evaluator parameters. `metrics` lists the judge metrics to compute.
- `pipeline`: ordered list of stages.
- `dataset`: `input_path`, `output_path`, and SFT filter parameters (`skip_empty`, `min_length`, `max_length`).

### `quality_filter` config

- `min_scores`: minimum thresholds per metric.
  - Numeric metrics (`informativeness`, `helpfulness`, `generalization`): minimum score.
  - Boolean metric (`correctness`): `true` or `false`.
- `require_all_metrics`: if `true`, rows with any missing score are dropped.
- `keep_top_k`: keep only the top-k rows after minimum-score filtering.
- `keep_top_ratio`: keep only the top N% rows (for example, `0.7` keeps 70%).

If both `keep_top_k` and `keep_top_ratio` are set, `keep_top_k` wins.

## Stage data formats

The pipeline writes intermediate JSONL files between stages. Each row preserves the fields needed by the next stage.

| Stage | Output fields | Notes |
|---|---|---|
| `instruction_expansion` | `instruction` | One expanded instruction per row. |
| `instruction_refinement` | `instruction` | Refined version of the previous stage. |
| `generate` | `instruction`, `output` | Teacher-generated response. |
| `instruct_eval` | `instruction`, `output`, `<metrics>` | Original fields plus per-metric scores. |
| `quality_filter` | same as `instruct_eval` | Only rows passing thresholds are kept. |
| `build_sft` | `messages`, `metadata` | Final OpenAI/ShareGPT SFT messages. |

## Running the pipeline

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/advanced_instruct_distill_pai_token.yaml
```

A PAI-EAS equivalent is provided as [`configs/pipeline/advanced_instruct_distill_pai_eas.yaml`](../configs/pipeline/advanced_instruct_distill_pai_eas.yaml).
