# Basic Instruction Distillation

This document describes the basic instruction distillation module in EasyDistill 2. It covers data generation, augmentation, SFT dataset construction, and evaluation for instruction-following models. For chain-of-thought distillation, see [cot_distillation.md](cot_distillation.md).

## Overview

The module provides the following job types:

| Job type | Purpose |
|---|---|
| `instruct_distill` | Generate teacher responses for seed instructions and build an SFT dataset. |
| `instruction_expansion` | Generate new instructions from seed instructions using in-context examples. |
| `instruction_refinement` | Rewrite or optimize existing instructions. |
| `instruction_response_extraction` | Extract `<instruction>/<response>` pairs from raw text. |
| `instruction_balance` | Classify instructions by task/domain and resample to a target distribution. See [instruction_balancing.md](instruction_balancing.md). |
| `instruct_eval` | Evaluate `(instruction, response)` pairs with an LLM judge. |
| `cot_distill`, `cot_long2short`, `cot_short2long`, `cot_eval` | Chain-of-thought distillation. See [cot_distillation.md](cot_distillation.md). |

For end-to-end pipelines such as `augmented_instruct_distill` and `advanced_instruct_distill`, see [pipelines.md](pipelines.md).

All job types are backend-agnostic and can run on PAI-Token or PAI-EAS backends.

For the full JSONL schema reference used across jobs, see [data_formats.md](data_formats.md).

## Common data formats

### Seed instructions

A JSONL file with one instruction per line:

```jsonl
{"instruction": "What is the capital of France?"}
{"instruction": "Explain quantum computing in one sentence."}
```

### Raw text for extraction

A JSONL file with a `text` field:

```jsonl
{"text": "User: What is 2+2?\nAssistant: 2+2 equals 4."}
{"text": "Q: Define machine learning.\nA: Machine learning is ..."}
```

### SFT dataset output

EasyDistill 2 produces OpenAI/ShareGPT-style messages:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]}
```

## CLI

All jobs share the same command:

```bash
easydistill --config <path_to_config.yaml>
```

The `job_type` field in the config selects which pipeline to run.

## Customizing prompts

All synthesis operators and the evaluator load their default prompts from `easydistill/prompts.py`, but every prompt can be overridden via config.

### Synthesis operators

For `instruction_expansion`, `instruction_refinement`, and `instruction_response_extraction`, you can either inline a prompt template or reference an external text file:

```yaml
synthesis:
  prompt_template_file: configs/prompts/expansion_prompt.txt
```

Or inline:

```yaml
synthesis:
  prompt_template: |
    You are given some example instructions below. Your task is to write a NEW instruction.

    {examples}

    Now write a new instruction. Wrap your output in <answer>...</answer>.
```

Placeholders:

- `instruction_expansion`: `{examples}`
- `instruction_refinement`: `{instruction}`
- `instruction_response_extraction`: `{text}`

### Evaluator

For `instruct_eval`, you can override individual metric prompts inline or load them from a YAML/JSON file:

```yaml
eval:
  prompts_file: configs/prompts/default_eval_prompts.yaml
```

Or inline:

```yaml
eval:
  prompts:
    informativeness: |
      Rate the informativeness of the response on a scale of 0-9.
      Instruction: {instruction}
      Response: {output}
      Place your score in <score></score>.
```

Prompt files and inline config values are merged with the built-in defaults, so you only need to specify the prompts you want to change.

## Backends

Backend configs follow the same pattern across all job types.

### PAI-Token

```yaml
backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct
```

### EAS

```yaml
backend:
  type: pai_eas
  endpoint_url: ${EAS_ENDPOINT_URL}
  token: ${EAS_TOKEN}
```

## Job type: `instruct_distill`

Run black-box distillation: seed instructions -> teacher responses -> SFT dataset.

```yaml
job_type: instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant. Provide concise and accurate answers."
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4

sft:
  skip_empty: true
  min_length: 10
  max_length: 8192

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/instruct_distill_sft.jsonl
```

### SFT filtering

- `skip_empty`: drop empty assistant responses.
- `min_length`: minimum character length of the response.
- `max_length`: maximum character length of the response (useful for filtering overly verbose teachers).

## Job type: `instruction_expansion`

Given seed instructions, sample in-context examples and generate new instructions.

```yaml
job_type: instruction_expansion

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  num_in_context_samples: 2
  num_output_samples: 3
  temperature: 0.8
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/expanded_instructions.jsonl
  output_format: instruction
```

## Job type: `instruction_refinement`

Rewrite or optimize existing instructions.

```yaml
job_type: instruction_refinement

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/refined_instructions.jsonl
  output_format: instruction
```

## Job type: `instruction_response_extraction`

Extract `<instruction>/<response>` pairs from raw text. The operator first tries regex extraction; if that fails, it prompts an LLM to extract the pair.

```yaml
job_type: instruction_response_extraction

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/raw_texts.jsonl
  text_key: text
  output_path: outputs/extracted_pairs.jsonl
  output_format: instruction_response
```

## Job type: `instruct_eval`

Evaluate `(instruction, response)` pairs with an LLM judge. The judge scores four metrics:

| Metric | Range | Description |
|---|---|---|
| `informativeness` | 0-9 | How thoroughly and accurately the response addresses the instruction. |
| `helpfulness` | 0-9 | How well the response assists the user. |
| `generalization` | 0-9 | How transferable the reasoning is to similar tasks. |
| `correctness` | true/false | Whether the response is factually and logically correct. |

```yaml
job_type: instruct_eval

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

eval:
  metrics:
    - informativeness
    - helpfulness
    - generalization
    - correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/eval_samples.jsonl
  output_path: outputs/eval_results.jsonl
```

### Supported input formats

Plain `(instruction, output)` pairs:

```jsonl
{"instruction": "What is the capital of France?", "output": "Paris"}
```

SFT messages (auto-converted):

```jsonl
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]}
```

The output JSONL contains per-sample scores plus aggregate averages in the CLI log.

Example output row:

```jsonl
{"id": "0", "instruction": "What is the capital of France?", "output": "Paris", "informativeness": 8, "helpfulness": 9, "generalization": 7, "correctness": true}
```

## Tips

- Use `max_length` in `sft` to avoid overly verbose teacher outputs.
- Keep `max_workers` moderate for real APIs to avoid rate limits.
- The evaluator expects the judge model to return scores in `<score>...</score>` tags.
- For a full end-to-end workflow, use `advanced_instruct_distill`. It combines augmentation, evaluation, and quality filtering in one run.
