# T2I Text-to-Image Distillation

EasyDistill 2 supports T2I (text-to-image) distillation: turning seed text prompts into multi-modal SFT training data where the assistant response is a generated image. The core pipeline is:

**seed prompt → prompt optimization → T2I teacher generation → VLM-as-judge evaluation → quality filtering → multi-modal SFT dataset**

The output format is `{messages: [{role: user, content: optimized_prompt}, {role: assistant, content: [image_url]}]}`, compatible with LLaMA-Factory and ms-swift multi-modal training.

> This document is the **user/runbook guide**: how to run T2I distillation jobs and configure backends. For architecture, stage-by-stage data schemas, and extension guides, see the **implementation document** [t2i_distillation_implementation.md](t2i_distillation_implementation.md).

## Architecture overview

T2I distillation uses **two separate backends**:

| Backend | Config key | Type | Purpose |
|---------|-----------|------|---------|
| VLM backend | `backend` | `ModelBackend` | Prompt optimization and image evaluation (e.g., Qwen-VL) |
| T2I backend | `t2i_backend` | `T2IBackend` | Image generation (e.g., Wanx, Qwen-Image, PAI-Diffusion) |

The T2I backend uses a different API protocol from chat completions, so it has its own abstraction (`T2IBackend`) parallel to `ModelBackend`.

## Supported T2I backends

| Backend | Config `type` | SDK | Notes |
|---------|--------------|-----|-------|
| Tongyi Wanxiang (Wanx) | `wanx` | dashscope | Async task submission + polling. Default model: `wanx2.1-t2i-turbo`. |
| Qwen-Image | `qwen_image` | dashscope | Async task submission + polling. Default model: `qwen-image2.0-pro`. |
| PAI-Diffusion | `pai_diffusion` | httpx | PAI-EAS deployed SD/Flux via OpenAI-compatible `/images/generations` endpoint. |

### Wanx backend

```yaml
t2i_backend:
  type: wanx
  api_key: ${DASHSCOPE_API_KEY}
  model_id: wanx2.1-t2i-turbo
```

Requires the `dashscope` package: `pip install easydistill[t2i]`.

The Wanx API is asynchronous: a generation request submits a task, then the backend polls until the task completes and image URLs are returned.

### Qwen-Image backend

```yaml
t2i_backend:
  type: qwen_image
  api_key: ${DASHSCOPE_API_KEY}
  model_id: qwen-image2.0-pro
```

Requires the `dashscope` package: `pip install easydistill[t2i]`.

The Qwen-Image API uses the same async task protocol as Wanx: submit a task, poll until completion, and return image URLs.

### PAI-Diffusion backend

```yaml
t2i_backend:
  type: pai_diffusion
  endpoint_url: ${EAS_ENDPOINT_URL}
  token: ${EAS_TOKEN}
  model_id: stable-diffusion-xl
```

Uses httpx to call the OpenAI-compatible `/images/generations` endpoint directly. No additional SDK required.

## Data format

### Input: seed prompts

Input rows are JSON objects with a `prompt` text field:

```jsonl
{"id": "1", "prompt": "一只在月球上喝咖啡的猫"}
{"id": "2", "prompt": "赛博朋克风格的城市街道，霓虹灯闪烁，下着小雨"}
```

Use `dataset.prompt_key` to override the default key name.

### Output: multi-modal SFT

Output rows are ShareGPT-format SFT messages where the assistant content is a multi-modal list containing the image URL:

```jsonl
{
  "messages": [
    {"role": "user", "content": "A cat drinking coffee on the moon, cinematic lighting, ultra-detailed"},
    {"role": "assistant", "content": [
      {"type": "image_url", "image_url": {"url": "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/..."}}
    ]}
  ],
  "metadata": {
    "source": "t2i_distillation",
    "t2i_model": "wanx2.1-t2i-turbo",
    "raw_prompt": "一只在月球上喝咖啡的猫",
    "request_id": "1"
  }
}
```

For the stage-by-stage intermediate JSONL schemas (prompt optimization, generation, evaluation), see [t2i_distillation_implementation.md](t2i_distillation_implementation.md).

## Standalone operations

### Basic T2I distillation (`t2i_distill`)

The simplest T2I flow — send seed prompts directly to the T2I backend and build SFT data. No prompt optimization, no evaluation.

```bash
# Wanx
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_distill_wanx.yaml

# Qwen-Image
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_distill_qwen_image.yaml
```

### Prompt optimization (`prompt_optimize`)

Use an LLM/VLM to rewrite simple seed prompts into rich, descriptive prompts suitable for T2I models.

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/t2i/prompt_optimize_pai_token.yaml

# PAI-EAS self-deployed endpoint / PAI-EAS 自部署端点
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/t2i/prompt_optimize_pai_eas.yaml
```

The optimizer adds details about subject, style, composition, lighting, and quality. Output rows contain `raw_prompt` and `optimized_prompt` fields.

### T2I generation (`t2i_generation`)

Send prompts to the T2I backend and save the generated image URLs. No SFT building.

```bash
# Wanx
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_generation_wanx.yaml

# Qwen-Image
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_generation_qwen_image.yaml
```

### T2I evaluation (`t2i_eval`)

Score generated images with a VLM-as-judge across multiple quality dimensions.

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/eval/t2i_eval_pai_token.yaml

# PAI-EAS self-deployed endpoint / PAI-EAS 自部署端点
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/eval/t2i_eval_pai_eas.yaml
```

## Evaluation metrics

All metrics are scored 0–9 by the VLM judge. For the implementation of the evaluator and how to add custom metrics, see [t2i_distillation_implementation.md](t2i_distillation_implementation.md).

| Metric | Description |
|--------|-------------|
| `prompt_consistency` | How faithfully the image depicts the text prompt. |
| `aesthetic_quality` | Visual appeal: composition, color, lighting. |
| `detail_richness` | Level of fine detail and texture. |
| `artifact_absence` | Freedom from generation artifacts and distortions. |

Custom evaluation prompts can be provided via `eval.prompts_file` (see `configs/prompts/t2i_eval_prompts.yaml`).

## Advanced T2I distillation pipeline

The recommended T2I pipeline runs all stages in one command:

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
export PAI_TOKEN_API_KEY=your_pai_token_key

# Wanx
easydistill --config configs/t2i/advanced_t2i_distill_wanx.yaml

# Qwen-Image
easydistill --config configs/t2i/advanced_t2i_distill_qwen_image.yaml
```

Pipeline stages:

1. `prompt_optimize` — enhance seed prompts via LLM/VLM.
2. `t2i_generate` — generate images from optimized prompts via T2I backend.
3. `t2i_eval` — VLM-as-judge quality scoring.
4. `quality_filter` — filter by minimum scores or top-k/top-ratio.
5. `build_t2i_sft` — convert to multi-modal SFT dataset.

Each stage writes intermediate output to the path specified in `pipeline[].output_path`.

## Configuration reference

This table covers the most commonly used fields. For the complete YAML structure and stage-level overrides, see [t2i_distillation_implementation.md](t2i_distillation_implementation.md).

| Field | Description |
|-------|-------------|
| `backend.type` | VLM backend type: `pai_token`, `pai_eas`, or `openai` |
| `backend.model_id` | VLM model ID (e.g., `qwen-vl-max`) |
| `t2i_backend.type` | T2I backend type: `wanx`, `qwen_image`, or `pai_diffusion` |
| `t2i_backend.api_key` | API key for Wanx/Qwen-Image (env: `DASHSCOPE_API_KEY`) |
| `t2i_backend.endpoint_url` | Endpoint URL for PAI-Diffusion (env: `EAS_ENDPOINT_URL`) |
| `t2i_backend.token` | Token for PAI-Diffusion (env: `EAS_TOKEN`) |
| `t2i_backend.model_id` | T2I model ID |
| `generation.size` | Image size (e.g., `"1024*1024"`) |
| `generation.n` | Number of images per prompt |
| `generation.max_workers` | Concurrent T2I API workers |
| `generation.retry_attempts` | Retries on transient failures |
| `eval.metrics` | List of evaluation metrics |
| `eval.temperature` | VLM judge temperature (recommended 0.0) |
| `eval.max_tokens` | Max tokens for VLM judge response |
| `sft.skip_empty` | Skip rows with no images or empty prompts |
| `sft.min_prompt_length` | Minimum prompt length in characters |
| `sft.max_images_per_prompt` | Max images per SFT sample (default 1) |
| `dataset.input_path` | Input JSONL path |
| `dataset.output_path` | Output JSONL path |
| `dataset.prompt_key` | Key for prompt field (default: `prompt`) |

## Installation

```bash
# Core install
pip install -e .

# With T2I (Wanx) support
pip install -e ".[t2i]"

# With all optional dependencies
pip install -e ".[all]"
```
