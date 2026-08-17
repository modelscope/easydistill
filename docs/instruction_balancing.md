# Instruction Balancing / Task-Aware Curriculum Planning

This document describes the `instruction_balance` stage in EasyDistill 2. It classifies instructions by task type or domain and resamples the dataset to match a target distribution. The goal is to avoid over-represented categories and make the resulting SFT dataset more balanced and curriculum-friendly.

`instruction_balance` is no longer exposed as a standalone `job_type`. Use it inside `advanced_instruct_distill`, `augmented_instruct_distill`, or the dedicated `balanced_instruct_distill` pipeline.

For the full JSONL schema reference used by these pipelines, see [data_formats.md](data_formats.md).

## When to use it

Use instruction balancing when:

- Your seed or synthesized instructions are skewed toward a few domains (for example, most are math or coding).
- You want the final dataset to follow a known target distribution, such as the DistilQwen2 recipe.
- You want to prepare a curriculum where each training batch has a predictable mix of task types.

## How it works

The `InstructionBalancer` operator performs two steps:

1. **Classification**: every instruction is sent to the configured backend with a classification prompt. The model returns a category wrapped in `<answer>...</answer>` tags.
2. **Resampling**: the operator counts the number of samples per category and resamples so that each category matches its target ratio.
   - If a category has too many samples, it is randomly downsampled.
   - If a category has too few samples, existing samples are repeated until the target count is reached.
   - The final order is shuffled with a configurable random seed.

The default category list and target distribution come from the DistilQwen2 recipe. You can override them in the config.

## Pipeline usage

`instruction_balance` is used as a stage inside `advanced_instruct_distill`, `augmented_instruct_distill`, or `balanced_instruct_distill` pipelines. It is usually placed after synthesis stages and before teacher generation. For an end-to-end pipeline where balancing is the main feature, see [docs/balanced_instruct_distill.md](balanced_instruct_distill.md).

```yaml
job_type: advanced_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

pipeline:
  - stage: instruction_expansion
    config:
      num_in_context_samples: 2
      num_output_samples: 5
      max_workers: 3
    output_path: outputs/stage1_expanded.jsonl

  - stage: instruction_balance
    config:
      max_workers: 4
      seed: 42
    output_path: outputs/stage2_balanced.jsonl

  - stage: generate
    config:
      max_workers: 4
    output_path: outputs/stage3_generated.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/advanced_instruct_distill_sft_balanced.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

## Config schema

### Top-level `balance` section

| Field | Type | Default | Description |
|---|---|---|---|
| `instruction_key` | `str` | `instruction` | Field that contains the instruction text. |
| `category_key` | `str` | `category` | Field name used to store the assigned category. |
| `categories` | `List[str]` | DistilQwen2 list | Valid category names. |
| `target_distribution` | `Dict[str, float]` | DistilQwen2 ratios | Target ratio for each category. Ratios should sum to `1.0`. |
| `category_prompt` | `str` | Built-in prompt | Prompt template for classification. Must contain `{categories}` and `{instruction}` placeholders. |
| `system_prompt` | `str` | `None` | Optional system prompt for the classifier. |
| `max_workers` | `int` | `1` | Concurrent classification requests. |
| `show_progress` | `bool` | `true` | Show a progress bar. |
| `seed` | `int` | `42` | Random seed for resampling and shuffling. |
| `model_id` | `str` | `None` | Override the backend model ID for classification. |
| `temperature` | `float` | `0.0` | Sampling temperature for classification. |
| `max_tokens` | `int` | `512` | Maximum tokens for the category response. |

### Stage config

When used as a pipeline stage, the `config` block accepts any of the fields above. The stage has no required fields; defaults are used for anything omitted.

## Customizing the category list and distribution

You can define your own curriculum by providing both `categories` and `target_distribution`:

```yaml
balance:
  categories: ["Easy", "Medium", "Hard"]
  target_distribution:
    Easy: 0.3
    Medium: 0.5
    Hard: 0.2
  category_prompt: |
    Classify the following instruction into one of {categories}.
    Wrap the answer in <answer></answer> tags.

    {instruction}
```

The classifier output is parsed in this order:

1. Look for `<answer>...</answer>` and check whether the value is in `categories`.
2. If no valid tag is found, check whether any category name appears as a substring.
3. Otherwise, fall back to `"Others"`.

## Output format

The output is a JSONL file where each row keeps the original fields and adds the assigned category:

```json
{"instruction": "What is 2 + 2?", "category": "Math"}
{"instruction": "Write a Python function to sort a list.", "category": "Code Generation"}
```
