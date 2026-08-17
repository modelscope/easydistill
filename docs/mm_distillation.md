# Multi-Modal Black-Box Knowledge Distillation (MMKD)

EasyDistill2 supports black-box knowledge distillation for multi-modal LLMs (MLLMs). Given a set of `(image, instruction)` pairs, the toolkit calls a teacher vision-language model and produces SFT training data in OpenAI/ShareGPT message format.

Only API backends are supported: `pai_token` and `pai_eas`.

For the full JSONL schemas, see [data_formats.md](data_formats.md).

## Data format

Input rows are JSON objects with an `instruction` text field and an `images` list:

```jsonl
{"id": "mm_0", "instruction": "Describe what you see.", "images": ["examples/mm_sample_image.png"]}
{"id": "mm_1", "instruction": "What color is the main object?", "images": ["https://example.com/img.png"]}
```

`images` can contain:

- Local file paths (relative or absolute).
- `file:///path/to/image` URIs.
- `http(s)://` URLs.
- Base64 data URLs such as `data:image/png;base64,...`.

Local files are automatically converted to base64 data URLs before being sent to the vision API.

## Standalone generation

Use `job_type: mm_instruct_distill` to generate teacher responses.

```bash
python -m easydistill.cli --config configs/basic/mm_instruct_distill_pai_token.yaml
```

Example config:

```yaml
job_type: mm_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen-vl-max

mm:
  system_prompt: "You are a helpful visual assistant."
  temperature: 0.7
  max_tokens: 2048
  show_progress: true
  max_workers: 3

dataset:
  input_path: examples/mm_seed_instructions.jsonl
  output_path: outputs/mm_instruct_distill_pai_token.jsonl
```

Output rows are ShareGPT-format SFT messages. The user message contains the multi-modal content with image references:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful visual assistant."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe what you see."}
    ]},
    {"role": "assistant", "content": "The image shows a solid red square."}
  ],
  "metadata": {"source": "teacher_model", "model": "Qwen2.5-VL-3B-Instruct", "request_id": "mm_gen_0", "backend": "pai_eas", "images": ["examples/mm_sample_image.png"]}
}
```

## Evaluation

Use `job_type: mm_instruct_eval` to score generated responses with an LLM judge.

```bash
python -m easydistill.cli --config configs/eval/mm_instruct_eval_pai_token.yaml
```

Supported metrics: `informativeness`, `helpfulness`, `generalization`, `correctness`.

## `advanced_mm_distill` pipeline

The recommended MMKD pipeline runs generation, evaluation, quality filtering, and SFT dataset construction in one command.

```bash
python -m easydistill.cli --config configs/pipeline/advanced_mm_distill_pai_token.yaml
```

Pipeline stages:

1. `mm_instruct_distill` - generate teacher responses for `(image, instruction)` pairs and build multi-modal SFT data.
2. `mm_instruct_eval` - LLM-as-judge scoring.
3. `quality_filter` - filter by minimum scores or top-k/top-ratio.
4. `build_sft` - convert to ShareGPT-format SFT data with multi-modal user messages.

The final SFT messages include image content items, so the dataset can be used directly to train MLLMs that accept the OpenAI vision format.

## Configuration reference

| Field | Description |
|-------|-------------|
| `backend.type` | `pai_token` or `pai_eas` |
| `backend.model_id` | Vision-language model ID |
| `mm.system_prompt` | Optional system prompt |
| `mm.temperature` | Sampling temperature |
| `mm.max_tokens` | Max response tokens |
| `mm.max_workers` | Concurrent API workers |
| `dataset.input_path` | Input JSONL path |
| `dataset.output_path` | Output JSONL path |
| `dataset.images_key` | Key for image list (default: `images`) |

For the EAS backend, replace `backend.api_key` with `endpoint_url` and `token`.
