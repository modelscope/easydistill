# Pipelines

EasyDistill 2 provides several end-to-end pipeline job types. Each pipeline chains multiple operators into a single config-driven run, saving intermediate stage outputs for inspection.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## Available pipelines

| Pipeline | Purpose | Config path |
|---|---|---|
| `augmented_instruct_distill` | Expand and refine seed instructions, then distill teacher responses into SFT data. | `configs/pipeline/augmented_instruct_distill_pai_token.yaml` |
| `advanced_instruct_distill` | Expand, refine, generate, evaluate, filter, and build a curated instruction SFT dataset. | `configs/pipeline/advanced_instruct_distill_pai_token.yaml` |
| `balanced_instruct_distill` | Synthesize instructions, balance by category, generate responses, and build an SFT dataset. | `configs/pipeline/balanced_instruct_distill_pai_token.yaml` |
| `advanced_cot_distill` | Generate CoT reasoning, score with RV/CD metrics, mix by difficulty bins, and build an SFT dataset. | `configs/pipeline/advanced_cot_distill_pai_token.yaml` |
| `advanced_mm_distill` | Generate multi-modal teacher responses, evaluate, filter, and build an SFT dataset. | `configs/pipeline/advanced_mm_distill_pai_token.yaml` |
| `advanced_mm_cot_distill` | Generate multi-modal CoT reasoning, evaluate, filter, and build an SFT dataset. | `configs/pipeline/advanced_mm_cot_distill_pai_token.yaml`<br>`configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml` (OmniThoughtV recipe) |
| `advanced_t2i_distill` | Optimize prompts, generate images, evaluate with a VLM judge, filter, and build a multi-modal SFT dataset. | `configs/t2i/advanced_t2i_distill_wanx.yaml`<br>`configs/t2i/advanced_t2i_distill_qwen_image.yaml`<br>`configs/t2i/advanced_t2i_distill_pai_diffusion.yaml` |
| `advanced_t2v_distill` | Optimize prompts (extract → compose), generate videos (T2V/I2V), evaluate (VLM / omni / VBench), filter, and build a multi-modal SFT dataset. Supports per-stage resume. | `configs/pipeline/advanced_t2v_distill_pai_token.yaml`<br>`configs/pipeline/advanced_t2v_distill_pai_eas.yaml` |
| `pe_rewrite_distill` | Expand seed prompts, rewrite them via a plan/rewrite/reflection teacher agent, judge, filter, and build a prompt-rewriting SFT dataset. | `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` |

## Common structure

Pipeline configs use the same top-level sections as other EasyDistill 2 configs:

```yaml
job_type: <pipeline_name>

backend:
  type: pai_token          # or pai_eas
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant."
  temperature: 0.7
  max_tokens: 2048

pipeline:
  - stage: <stage_name>
    config:
      ...
    output_path: outputs/stage1.jsonl

  - stage: <final_stage>
    config:
      ...

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/final.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

- `pipeline`: ordered list of stages. Each stage can write its output to `output_path`.
- `dataset`: input/output paths and SFT filter parameters (`skip_empty`, `min_length`, `max_length`).

See the sub-document for each pipeline for stage-level details.
