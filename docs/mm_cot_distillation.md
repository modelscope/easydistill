# Multi-Modal Chain-of-Thought Distillation (MMCoT)

EasyDistill2 supports chain-of-thought distillation for multi-modal problems. The MMCoT module generates, rewrites, evaluates, and filters visual reasoning traces to build SFT datasets for vision-language models.

Only API backends are supported: `pai_token` and `pai_eas`.

For the full JSONL schemas, see [data_formats.md](data_formats.md).

## Data format

Input rows are JSON objects with a problem text and an `images` list:

```jsonl
{"id": "mmcot_0", "instruction": "How would you determine the dominant color?", "images": ["examples/mm_sample_image.png"]}
```

Image references follow the same rules as MMKD: local paths, URLs, or base64 data URLs.

## Standalone generation

Use `job_type: mm_cot_distill` to generate multi-modal CoT traces.

```bash
python -m easydistill.cli --config configs/basic/mm_cot_distill_pai_token.yaml
```

Example config:

```yaml
job_type: mm_cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen-vl-max

cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
  system_prompt: "You are a helpful visual reasoning assistant."
  temperature: 0.7
  max_tokens: 2048
  show_progress: true
  max_workers: 3

dataset:
  input_path: examples/mm_seed_cot_problems.jsonl
  output_path: outputs/mm_cot_distill_pai_token.jsonl
```

Output rows are ShareGPT-format SFT messages. The raw CoT response is stored as the assistant message; `thought` and `solution` are preserved in metadata.

## Rewriting CoT traces

Two optional rewrite operators are available:

- `mm_cot_long2short` - simplify a reasoning trace while preserving the final answer.
- `mm_cot_short2long` - extend a reasoning trace with additional details.

Both accept rows with `instruction`, `images`, and `response` fields, and also auto-convert SFT `messages` rows (reading images from `metadata.images`).

```bash
python -m easydistill.cli --config configs/rewrite/mm_cot_long2short_pai_token.yaml
python -m easydistill.cli --config configs/rewrite/mm_cot_short2long_pai_token.yaml
```

## Evaluation

Use `job_type: mm_cot_eval` to score MMCoT outputs.

```bash
python -m easydistill.cli --config configs/eval/mm_cot_eval_pai_token.yaml
```

Supported metrics: `reasoning_verbosity`, `cognitive_difficulty`, `logical_correctness`.

## `advanced_mm_cot_distill` pipeline

The recommended MMCoT pipeline mirrors the text CoT pipeline but handles image inputs.

```bash
python -m easydistill.cli --config configs/pipeline/advanced_mm_cot_distill_pai_token.yaml
```

Pipeline stages:

1. `mm_cot_distill` - generate visual reasoning traces.
2. `mm_cot_eval` - LLM-as-judge scoring.
3. `quality_filter` - filter by score thresholds.
4. `build_sft` - convert to ShareGPT-format SFT data with multi-modal user messages.

## OmniThoughtV

[OmniThoughtV](https://modelscope.cn/datasets/platformofai/OmniThoughtV_Filter_0.5M) is a large-scale multi-modal long chain-of-thought dataset built with this MMCoT module: visual problems from [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) are distilled through Qwen-VL-max, scored on reasoning verbosity (RV), cognitive difficulty (CD), and logical correctness, then filtered into a high-quality SFT subset. Fine-tuning Qwen3-VL 2B/4B/8B on the filtered subset improves both general visual understanding (AI2D, MMStar) and reasoning-heavy benchmarks (MMMU-Pro, MathVerse, MathVision).

Two releases are available (see [model_zoo.md](model_zoo.md)):

| Dataset | Size | Notes |
|---------|------|-------|
| `OmniThoughtV_Raw_1.8M` | 1.8M | Raw distilled traces |
| `OmniThoughtV_Filter_0.5M` | 0.5M | RV/CD-filtered subset used for fine-tuning |

Traces use the `<thinking>...</thinking><answer>...</answer>` format, which `mm_cot_distill` parses natively alongside the `<|begin_of_thought|>` format. To reproduce the recipe on your own seed problems:

```bash
python -m easydistill.cli --config configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml
```

The config uses `configs/prompts/mm_cot_thinking_prompt.txt` (the OmniThoughtV trace format) together with the system prompt used in the released dataset. Swap `backend.model_id` to your multi-modal teacher; the original datasets were distilled from Qwen-VL-max.

## Configuration reference

| Field | Description |
|-------|-------------|
| `backend.type` | `pai_token` or `pai_eas` |
| `backend.model_id` | Vision-language model ID |
| `cot.prompt_template_file` | Path to prompt template |
| `cot.system_prompt` | Optional system prompt |
| `cot.temperature` | Sampling temperature |
| `cot.max_tokens` | Max response tokens |
| `dataset.input_path` | Input JSONL path |
| `dataset.output_path` | Output JSONL path |
| `dataset.images_key` | Key for image list (default: `images`) |

For the EAS backend, replace `backend.api_key` with `endpoint_url` and `token`.
