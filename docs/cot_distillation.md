# Chain-of-Thought Distillation

This document describes the chain-of-thought (CoT) distillation module in EasyDistill 2. It covers CoT generation, reasoning rewriting (long-to-short / short-to-long), LLM-as-judge evaluation, and the `advanced_cot_distill` end-to-end pipeline.

## Overview

The module provides five job types:

| Job type | Purpose |
|---|---|
| `cot_distill` | Generate `<|begin_of_thought|>` / `<|begin_of_solution|>` reasoning and solution for each input problem. |
| `cot_long2short` | Simplify an existing CoT reasoning process. |
| `cot_short2long` | Extend an existing CoT reasoning process with more details. |
| `cot_eval` | Evaluate `(problem, CoT-answer)` pairs with an LLM judge. |
| `advanced_cot_distill` | End-to-end pipeline: CoT generation -> RV/CD scoring -> RV/CD mixing -> SFT dataset. |

All job types support PAI-Token and PAI-EAS backends.

For the full JSONL schema reference used across jobs, see [data_formats.md](data_formats.md).

## Common data formats

### Seed problems

A JSONL file with one problem per line:

```jsonl
{"problem": "What is the sum of the first 10 positive integers?"}
{"problem": "A train travels 60 km in 30 minutes. What is its average speed in km/h?"}
```

For `cot_distill`, the `problem_key` in the dataset config selects the field to use (default `problem`, fallback `instruction`).

### Problem/answer pairs for rewriting

`cot_long2short` and `cot_short2long` read `(problem, answer)` pairs. By default they use `instruction` for the problem and `response` for the answer, with fallbacks to common field names:

```jsonl
{"instruction": "What is 2+2?", "response": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
```

You can override the keys via `problem_key` and `answer_key` in the dataset config.

### CoT generation output (basic feature)

`cot_distill` is a basic feature: it directly produces ShareGPT-format SFT messages that can be fed into LLaMA-Factory or ms-swift.

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}], "metadata": {"thought": "...", "solution": "4"}}
```

### SFT dataset output in `advanced_cot_distill`

The final `build_sft` stage in `advanced_cot_distill` converts `instruction`/`response` pairs into OpenAI/ShareGPT-style messages:

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}]}
```

## CLI

All jobs share the same command:

```bash
easydistill --config <path_to_config.yaml>
```

The `job_type` field in the config selects which pipeline to run.

## Customizing prompts

### CoT operators

For `cot_distill`, `cot_long2short`, and `cot_short2long`, you can either inline a prompt template or reference an external text file:

```yaml
cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
```

Or inline:

```yaml
cot:
  prompt_template: |
    Solve the following problem step by step.

    Problem: {problem}
```

Placeholders:

- `cot_distill`: `{problem}`
- `cot_long2short`: `{problem}`, `{answer}`
- `cot_short2long`: `{problem}`, `{answer}`

### Evaluator

For `cot_eval`, you can override individual metric prompts inline or load them from a YAML/JSON file:

```yaml
eval:
  prompts_file: configs/prompts/default_cot_eval_prompts.yaml
```

Or inline:

```yaml
eval:
  prompts:
    reasoning_verbosity: |
      Rate the reasoning verbosity of the following CoT on a scale of 0-9.
      Problem: {instruction}
      Answer with CoT: {output}
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

## Job type: `cot_distill`

Generate a chain-of-thought reasoning process and final solution for each problem.

```yaml
job_type: cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/seed_cot_problems.jsonl
  problem_key: problem
  output_path: outputs/cot_distill.jsonl
```

## Job type: `cot_long2short`

Simplify an existing CoT reasoning process. The input must contain both the problem and the full CoT answer.

```yaml
job_type: cot_long2short

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_long2short_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  problem_key: instruction
  answer_key: response
  output_path: outputs/cot_long2short.jsonl
  output_format: cot
```

## Job type: `cot_short2long`

Extend an existing CoT reasoning process with more intermediate steps and details.

```yaml
job_type: cot_short2long

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_short2long_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  problem_key: instruction
  answer_key: response
  output_path: outputs/cot_short2long.jsonl
  output_format: cot
```

## Job type: `cot_eval`

Evaluate `(problem, CoT-answer)` pairs with an LLM judge. The judge scores three metrics:

| Metric | Range | Description |
|---|---|---|
| `reasoning_verbosity` | 0-9 | How well the CoT length and step complexity match the problem difficulty. |
| `cognitive_difficulty` | 0-9 | The level of reasoning competence required to follow and reproduce the chain. |
| `logical_correctness` | true/false | Whether the reasoning process and final solution are logically sound. |

```yaml
job_type: cot_eval

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

eval:
  metrics:
    - reasoning_verbosity
    - cognitive_difficulty
    - logical_correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  output_path: outputs/cot_eval_results.jsonl
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
{"id": "0", "instruction": "What is the sum of the first 10 positive integers?", "output": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>55<|end_of_solution|>", "reasoning_verbosity": 6, "cognitive_difficulty": 5, "logical_correctness": true}
```

## Job type: `advanced_cot_distill`

Run the full CoT distillation workflow in one command:

1. `cot_distill` — generate CoT reasoning and solution for each problem.
2. `cot_rvcd_score` — score each generated CoT on reasoning verbosity, cognitive difficulty, and logical correctness.
3. `cot_mix_by_rv_cd` — mix rows per CD bin to build a curriculum SFT subset.
4. `build_sft` — produce the final SFT dataset.

The last stage must be `build_sft`.

```yaml
job_type: advanced_cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant. Think step by step and provide clear reasoning."
  temperature: 0.7
  max_tokens: 2048

sft:
  skip_empty: true
  min_length: 10
  max_length: 4096

eval:
  prompts_file: configs/prompts/default_cot_eval_prompts.yaml
  metrics:
    - reasoning_verbosity
    - cognitive_difficulty
    - logical_correctness
  temperature: 0.0
  max_tokens: 512
  max_workers: 4

pipeline:
  - stage: cot_distill
    config:
      prompt_template_file: configs/prompts/cot_generation_prompt.txt
      temperature: 0.7
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/cot_bp_stage1_generated.jsonl

  - stage: cot_rvcd_score
    config:
      show_progress: true
      max_workers: 4
    output_path: outputs/cot_bp_stage2_scored.jsonl

  - stage: cot_mix_by_rv_cd
    config:
      mode: sft
      cd_bins: [0, 3, 6, 10]
      rv_target: matched
      samples_per_bin: 100
      min_correctness: 1
    output_path: outputs/cot_bp_stage3_mixed.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_cot_problems.jsonl
  output_path: outputs/cot_bp_sft.jsonl
```

To add a rewrite stage, insert either `cot_long2short` or `cot_short2long` between `cot_distill` and `cot_rvcd_score`.

## Tips

- Use `max_length` in `sft` to avoid overly verbose teacher outputs.
- Keep `max_workers` moderate for real APIs to avoid rate limits.
- The evaluator expects the judge model to return scores in `<score>...</score>` tags.
- For a full end-to-end workflow, use `advanced_cot_distill`. It combines generation, RV/CD scoring, and curriculum mixing in one run.
- After generating the SFT dataset, see [training_guide.md](training_guide.md) for LLaMA-Factory and ms-swift fine-tuning examples, including LoRA.
