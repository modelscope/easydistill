# T2V/I2V Text-to-Video Distillation

EasyDistill 2 supports T2V (text-to-video) and I2V (image-to-video) distillation: turning seed text prompts — optionally conditioned on a first-frame image — into multi-modal SFT training data where the assistant response is a generated video. The core pipeline is:

**seed prompt → prompt optimization (extract → compose) → video teacher generation → video evaluation (VLM / omni / VBench) → quality filtering → multi-modal SFT dataset**

The output format is `{messages: [{role: user, content: optimized_prompt}, {role: assistant, content: [video_url]}]}`, compatible with LLaMA-Factory and ms-swift multi-modal training. I2V samples additionally carry the conditioning image in the user turn.

## Architecture overview

T2V distillation uses **three backend slots**, which can be mixed freely across providers (e.g. a PAI-Token VLM with an EAS-deployed video model):

| Backend | Config key | Type | Purpose |
|---------|-----------|------|---------|
| LLM/VLM backend | `backend` | `ModelBackend` | Two-stage prompt optimization (a VLM is required when seeds contain I2V rows) |
| Eval backend | `eval_backend` | `ModelBackend` | Video evaluation (VLM judge / omni checker). Falls back to `backend` when omitted |
| T2V backend | `t2v_backend` | `T2VBackend` | Video generation (T2V and I2V modes in one backend) |

Rows carrying a `first_frame_image` field run in **I2V mode**; rows without it run plain **T2V**. Both kinds may be mixed in one batch — every stage branches per row.

## Supported video backends

| Backend | Config `type` | Protocol | Notes |
|---------|--------------|----------|-------|
| PAI-Token gateway video | `pai_token_video` | DashScope-style async (submit + poll) | Models such as `happyhorse-1.1-t2v` / `happyhorse-1.1-i2v` / `wan2.7-t2v`. |
| PAI-EAS deployed video | `pai_video` | `legacy` (sync/async) or `sglang` (`/v1/videos`) | Self-deployed video models on PAI-EAS. |

### PAI-Token video backend

```yaml
t2v_backend:
  type: pai_token_video
  api_key: ${PAI_TOKEN_API_KEY}
  base_url: ${PAI_TOKEN_BASE_URL}
  model_id: happyhorse-1.1-t2v      # rows without first_frame_image
  i2v_model_id: happyhorse-1.1-i2v  # rows with first_frame_image
  output_dir: outputs/t2v_videos    # videos are downloaded here
```

The gateway proxies DashScope-style video synthesis: a request submits an async task (`X-DashScope-Async: enable`), then the backend polls until video URLs are returned and downloads them into `output_dir`. Notes learned from live runs:

- `generation.duration` must be an **integer** number of seconds (typical range 3–15).
- I2V passes the first frame through `input.media` (`{"type": "first_frame", "url": ...}`); set `i2v_image_field: img_url` for models that still use the older field.
- Supported `generation.resolution` tiers are model-specific (e.g. `wan2.7-t2v` accepts only `720P`/`1080P`); the API rejects unsupported tiers at submit time.
- Remote video URLs **expire** (typically 24 h) — always set `output_dir` so rows carry durable local paths.

### PAI-EAS video backend

```yaml
t2v_backend:
  type: pai_video
  endpoint_url: ${EAS_VIDEO_ENDPOINT_URL}
  token: ${EAS_VIDEO_TOKEN}
  protocol: sglang            # or: legacy
  auth_prefix: ""             # sglang deployments use the bare token
  t2v_task: t2va
  i2v_task: fl2va
  sglang_short_edge: 768
  output_dir: outputs/t2v_videos
```

Two protocols are supported through one backend class:

- **`legacy`** — the service's own JSON API; sync and async (task-id polling) response shapes are detected automatically.
- **`sglang`** — the sglang serving stack's videos API: `POST /v1/videos` → poll `GET /v1/videos/{id}` → download `GET /v1/videos/{id}/content`. `output_dir` is required. I2V rows pass the first frame as `reference_url`, which must be an **http(s) URL reachable from the EAS service** (local paths are rejected).

## Data format

### Input: seed prompts

Input rows are JSON objects with a `prompt` field, plus an optional `first_frame_image` for I2V:

```jsonl
{"id": "1", "prompt": "一只猫在月球表面缓慢行走，扬起细小的尘埃"}
{"id": "4", "prompt": "画面中的场景逐渐入夜，远处雷峰塔的灯光亮起", "first_frame_image": "examples/seed_t2v_first_frame.jpg"}
```

`first_frame_image` accepts local file paths, `file://` URLs, `http(s)://` URLs, and data URLs; local images are normalized to base64 data URLs before being sent to VLM or backend APIs (except the sglang path, which requires an http(s) URL).

Per-row resolution control is available for **T2V rows only**:

- `resolution` — per-row override of the configured tier (`480P` / `720P` / `1080P`), or `auto` to have the extract stage infer the aspect ratio from the prompt (via LLM) and record it in `ratio`.
- `ratio` — per-row override of the configured aspect ratio (e.g. `16:9`, `9:16`); `resolution: auto` writes its inferred value into this field.

```jsonl
{"id": "2", "prompt": "sunset city timelapse, camera pulls back to the skyline", "resolution": "auto"}
{"id": "3", "prompt": "an orange cat yawning on the windowsill", "resolution": "480P", "ratio": "4:3"}
```

**I2V rows always follow their first frame**: `resolution` / `ratio` on such rows are ignored (a warning is logged). To guard against degenerate first frames, set `i2v_frame_check` under `generation` to `off` (default) | `warn` | `skip` | `raise`, tuned by `i2v_frame_min_edge` (minimum short edge, default 256) and `i2v_frame_max_aspect` (max long/short ratio, default 3.0); `skip` drops the row with a warning. http(s) first frames cannot be inspected locally and always pass.

A runnable seed covering all four input forms (default, explicit `resolution` + `ratio`, `auto`, I2V) is in `examples/seed_t2v_prompts.jsonl`.

### Output: multi-modal SFT

```jsonl
{
  "messages": [
    {"role": "user", "content": "A cinematic 3D CG render of an adult gray tabby cat ... slowly pushes in with small amplitude."},
    {"role": "assistant", "content": [
      {"type": "video_url", "video_url": {"url": "outputs/t2v_videos/365be208-....mp4"}}
    ]}
  ],
  "metadata": {
    "source": "t2v_distillation",
    "t2v_model": "happyhorse-1.1-t2v",
    "t2v_mode": "t2v",
    "raw_prompt": "一只猫在月球表面缓慢行走，扬起细小的尘埃",
    "request_id": "1",
    "prompt_consistency": 3, "visual_quality": 3, "subject_consistency": 4,
    "motion_quality": 2, "temporal_execution": 2, "camera_accuracy": 4
  }
}
```

I2V samples use an image + video message pair: the user turn carries the first-frame image alongside the prompt, and `metadata.t2v_mode` is `"i2v"`. Evaluation scores are carried in `metadata` so downstream training can re-filter without re-running evaluation.

## Prompt optimization (extract → compose)

Stage 1 (**extract**) parses the seed prompt (grounded in the first-frame image for I2V rows) into a structured JSON draft: subject, appearance, action, setting, lighting, camera, temporal beats. Stage 2 (**compose**) rewrites the draft into the final caption following a **caption schema**. Exactly two model calls per row; the draft is kept in the `draft` column for inspection.

A generic model-agnostic schema is built in. Every video model has its own preferred prompt style (language, length, structure) documented by its vendor — **for production runs, write a schema from your target model's official prompt guideline** and point the stage at it:

```yaml
- stage: prompt_optimize
  config:
    schema_file: path/to/your_model_caption_schema.txt   # or inline: schema: |
```

## Video evaluation

Evaluation is a composable checker chain under `eval.checkers`; each entry can be toggled with `enabled` and failures are isolated per checker. Scores use a **0–4 scale** and each dimension emits three columns: `<dim>`, `<dim>_confidence`, `<dim>_reason`.

| Checker | `type` | What it scores | Requirements |
|---------|--------|----------------|--------------|
| VLM judge | `vlm` | Sparse-frame-visible aspects: `prompt_consistency`, `visual_quality`, `subject_consistency`, `first_frame_consistency` (I2V only) | VLM on `eval_backend`; frames sampled locally via OpenCV |
| Omni checker | `omni` | Dynamic qualities needing the full video: `motion_quality`, `temporal_execution`, `camera_accuracy` | Video-native omni model (e.g. `qwen3.5-omni-plus`); `video_transport: auto` picks URL or base64 by size |
| VBench | `vbench` | Objective metrics via [VBench](https://github.com/Vchitect/VBench) (`vbench_*` columns) | `pip install vbench` in an isolated venv (see below); GPU for most dimensions; skips gracefully with `vbench_skipped_reason` when unavailable |

Dimension pools are defined in `configs/eval/t2v/vlm_dimensions.yaml` and `configs/eval/t2v/omni_dimensions.yaml` (deliberately disjoint). Scores stay raw per source — no cross-checker normalization is applied.

### VBench setup

VBench is consumed as an external tool — nothing is vendored. Install it into its own virtualenv (Python ≤ 3.11; its pinned dependencies conflict with easydistill's):

```bash
python3.10 -m venv /path/to/vbench_env
/path/to/vbench_env/bin/pip install vbench "setuptools<81"
```

Then point the checker at the CLI:

```yaml
- type: vbench
  enabled: true
  vbench_bin: /path/to/vbench_env/bin/vbench   # or env VBENCH_BIN
  dimensions: [motion_smoothness, dynamic_degree, imaging_quality, temporal_flickering]
```

Model checkpoints are downloaded automatically on first use. Most dimensions require a GPU (`require_gpu: true` by default). Running from a repo clone is also supported via `vbench_repo` + `python_executable`; `vbench_bin` wins when both are set. When the environment is unusable, the checker records the reason in `vbench_skipped_reason` and never fails the pipeline.

## Advanced T2V distillation pipeline

```bash
export PAI_TOKEN_API_KEY=your_key
export PAI_TOKEN_BASE_URL=https://your-endpoint/v1
easydistill --config configs/pipeline/advanced_t2v_distill_pai_token.yaml

# EAS-deployed video model (LLM/VLM still via any backend):
export EAS_VIDEO_ENDPOINT_URL=https://your-video-service.pai-eas.aliyuncs.com
export EAS_VIDEO_TOKEN=your_token
easydistill --config configs/pipeline/advanced_t2v_distill_pai_eas.yaml
```

Pipeline stages:

1. `prompt_optimize` — two-stage extract → compose prompt enhancement.
2. `t2v_generate` — generate videos via the T2V backend (I2V rows use `i2v_model_id`).
3. `t2v_eval` — composable video evaluation (VLM / omni / VBench).
4. `quality_filter` — filter by `min_scores` per metric or top-k/top-ratio over `eval.metrics`.
5. `build_t2v_sft` — convert to the multi-modal SFT dataset.

Each stage writes intermediate output to `pipeline[].output_path`.

### Resume / crash recovery

Video generation is slow and expensive, so the expensive stages support opt-in resume:

```yaml
- stage: t2v_generate
  config:
    resume: true
  output_path: outputs/t2v_stage2_generated.jsonl
```

With `resume: true`, re-running the same command reuses completed rows from the stage's `output_path` and only runs the missing/failed ones. `t2v_generate` additionally checkpoints **each finished video row immediately**, so a crash mid-stage never re-generates finished videos. Rows are matched by `id` (or a content hash when absent); rows that failed previously (no `video_urls`) are retried automatically. Delete the stage output file to force a full re-run, e.g. after changing generation parameters.

## Standalone operations

Single-stage job types reuse the same config sections (`job_type` selects the stage): `t2v_distill` (seed → videos → SFT, no optimization/eval), `t2v_prompt_optimize`, `t2v_generation`, `t2v_eval`.

```bash
# Basic T2V distillation (no optimization / evaluation)
easydistill --config configs/basic/t2v_distill_pai_token.yaml
```

## Configuration reference

| Field | Description |
|-------|-------------|
| `backend.type` | LLM/VLM backend: `pai_token`, `pai_eas`, or `openai` |
| `eval_backend` | Optional separate backend for evaluation (defaults to `backend`) |
| `t2v_backend.type` | Video backend: `pai_token_video` or `pai_video` |
| `t2v_backend.model_id` / `i2v_model_id` | T2V / I2V model IDs |
| `t2v_backend.output_dir` | Local directory for downloaded videos (strongly recommended) |
| `t2v_backend.protocol` | `pai_video` only: `legacy` or `sglang` |
| `t2v_backend.i2v_image_field` | `pai_token_video` only: `media` (default) or `img_url` |
| `generation.resolution` / `ratio` | Resolution tier (e.g. `720P`) and aspect ratio |
| `generation.duration` | Video length in seconds (integer) |
| `generation.watermark` | Disable provider watermark where supported |
| `generation.max_workers` | Concurrent generation workers (default 1 — keep 1 for single-replica EAS) |
| `prompt_optimize.schema` / `schema_file` | Target model's caption schema (defaults to a generic built-in) |
| `eval.checkers` | Checker chain: `vlm` / `omni` / `vbench` entries with per-entry `enabled` |
| `eval.metrics` | Metrics used by quality_filter averaging (default: `prompt_consistency`, `visual_quality`, `subject_consistency`) |
| `quality_filter.min_scores` | Per-metric minimum scores (0–4 scale) |
| `sft.skip_empty` / `min_prompt_length` / `max_videos_per_prompt` | SFT building options |
| `pipeline[].config.resume` | Opt-in stage resume (see above) |
| `dataset.input_path` / `output_path` | Input seeds / final SFT output |

## Installation

```bash
pip install -e .          # core (httpx-based video backends included)
pip install -e ".[all]"   # all optional dependencies
```

Frame sampling for the VLM checker requires OpenCV (`opencv-python-headless`), installed with the default dependencies.
