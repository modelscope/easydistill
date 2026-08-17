# EasyDistill 2 Job Type Index

This page maps every supported CLI `job_type` to a representative config file and the relevant documentation. Config paths are shown for the `pai_token` backend; each file has a matching `_pai_eas` variant unless noted otherwise.

For an interactive list, run `easydistill --list-jobs`.

## Basic distillation

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `instruct_distill` | Generate teacher responses for seed instructions and build SFT data. | `configs/basic/instruct_distill_pai_token.yaml` | [instruction_distillation.md](instruction_distillation.md) |
| `cot_distill` | Generate chain-of-thought reasoning traces and build SFT data. | `configs/basic/cot_distill_pai_token.yaml` | [cot_distillation.md](cot_distillation.md) |
| `mm_instruct_distill` | Generate teacher responses for `(image, instruction)` pairs. | `configs/basic/mm_instruct_distill_pai_token.yaml` | [mm_distillation.md](mm_distillation.md) |
| `mm_cot_distill` | Generate visual chain-of-thought reasoning traces. | `configs/basic/mm_cot_distill_pai_token.yaml` | [mm_cot_distillation.md](mm_cot_distillation.md) |
| `t2v_distill` | Generate videos from seed prompts (T2V/I2V) and build SFT data. | `configs/basic/t2v_distill_pai_token.yaml` | [t2v_distillation.md](t2v_distillation.md) |

## End-to-end pipelines

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `advanced_instruct_distill` | Expand → generate → judge → filter → SFT. | `configs/pipeline/advanced_instruct_distill_pai_token.yaml` | [advanced_instruct_distill.md](advanced_instruct_distill.md) |
| `balanced_instruct_distill` | Balance categories, generate, evaluate, filter, build SFT. | `configs/pipeline/balanced_instruct_distill_pai_token.yaml` | [balanced_instruct_distill.md](balanced_instruct_distill.md) |
| `augmented_instruct_distill` | Refine seeds, generate multiple responses, evaluate, filter, build SFT. | `configs/pipeline/augmented_instruct_distill_pai_token.yaml` | [augmented_instruct_distill.md](augmented_instruct_distill.md) |
| `advanced_cot_distill` | Generate CoT, score with RV/CD, mix by difficulty, build SFT. | `configs/pipeline/advanced_cot_distill_pai_token.yaml` | [cot_rvcd_mixer.md](cot_rvcd_mixer.md) |
| `advanced_mm_distill` | Multi-modal generate → eval → filter → SFT. | `configs/pipeline/advanced_mm_distill_pai_token.yaml` | [mm_distillation.md](mm_distillation.md) |
| `advanced_mm_cot_distill` | Visual CoT with RV/CD/correctness scoring. | `configs/pipeline/advanced_mm_cot_distill_pai_token.yaml` | [mm_cot_distillation.md](mm_cot_distillation.md) |
| `advanced_mm_cot_distill` (OmniThoughtV) | Same job_type; reproduces the OmniThoughtV recipe. | `configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml` | [mm_cot_distillation.md](mm_cot_distillation.md) |
| `advanced_t2i_distill` | Prompt optimization → T2I generation → VLM judge → filter → SFT. | `configs/t2i/advanced_t2i_distill_wanx.yaml` | [t2i_distillation.md](t2i_distillation.md) |
| `advanced_t2v_distill` | Prompt optimization → video generation → video eval → filter → SFT. | `configs/pipeline/advanced_t2v_distill_pai_token.yaml` | [t2v_distillation.md](t2v_distillation.md) |
| `pe_rewrite_distill` | Plan/rewrite/reflection → judge → filter → SFT for prompt rewriting. | `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` | [pe_rewrite.md](pe_rewrite.md) |
| `agent_distill` | Synthesize tool-use tasks and build agent trajectory SFT/DPO data. | `configs/pipeline/agent_distill_pai_token.yaml` (SFT) or `configs/pipeline/agent_distill_dpo_pai_token.yaml` (DPO) | [agent_distillation.md](agent_distillation.md) |
| `search_agent_distill` | Evolve seed QA into multi-hop search tasks and build SFT data. | `configs/pipeline/search_agent_distill_pai_token.yaml` | [search_agent_distillation.md](search_agent_distillation.md) |

## Preference data

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `dpo_data_build` | Generate candidates, score them, and build DPO preference pairs. | `configs/preference/dpo_instruct_pai_token.yaml` (set `dpo.task_type: instruct`) or `configs/preference/dpo_cot_pai_token.yaml` (set `dpo.task_type: cot`) | [dpo_distillation.md](dpo_distillation.md) |

## Text rewrite and synthesis operators

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `instruction_expansion` | Synthesize new instructions from seed examples. | `configs/rewrite/instruction_expansion_pai_token.yaml` | [instruction_balancing.md](instruction_balancing.md) |
| `seed_anchored_expansion` | Expand each seed into same-scenario instructions with dedup and lineage. | `configs/rewrite/seed_anchored_expansion_pai_token.yaml` | [instruction_balancing.md](instruction_balancing.md) |
| `instruction_refinement` | Rewrite and improve existing instructions. | `configs/rewrite/instruction_refinement_pai_token.yaml` | [instruction_balancing.md](instruction_balancing.md) |
| `instruction_response_extraction` | Extract instruction/response pairs from raw text. | `configs/rewrite/instruction_response_extraction_pai_token.yaml` | [instruction_balancing.md](instruction_balancing.md) |
| `agentic_rewrite` | Rewrite prompts via a plan → rewrite → reflection teacher agent chain. | `configs/rewrite/agentic_rewrite_pai_token.yaml` | [pe_rewrite.md](pe_rewrite.md) |
| `cot_long2short` | Simplify existing CoT reasoning traces. | `configs/rewrite/cot_long2short_pai_token.yaml` | [cot_distillation.md](cot_distillation.md) |
| `cot_short2long` | Extend existing CoT reasoning traces with more detail. | `configs/rewrite/cot_short2long_pai_token.yaml` | [cot_distillation.md](cot_distillation.md) |
| `mm_cot_long2short` | Simplify multi-modal CoT reasoning traces. | `configs/rewrite/mm_cot_long2short_pai_token.yaml` | [mm_cot_distillation.md](mm_cot_distillation.md) |
| `mm_cot_short2long` | Extend multi-modal CoT reasoning traces with more detail. | `configs/rewrite/mm_cot_short2long_pai_token.yaml` | [mm_cot_distillation.md](mm_cot_distillation.md) |

## PE rewrite pipeline stages

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `pe_rewrite_eval` | Score prompt rewrites with a multi-dimension LLM judge. | `configs/rewrite/pe_rewrite_eval_pai_token.yaml` | [pe_rewrite.md](pe_rewrite.md) |
| `pe_rewrite_filter` | Filter judged rewrites by score thresholds and top ratio. | `configs/rewrite/pe_rewrite_filter.yaml` | [pe_rewrite.md](pe_rewrite.md) |
| `pe_rewrite_build_sft` | Build SFT samples from filtered rewrites. | `configs/rewrite/pe_rewrite_build_sft.yaml` | [pe_rewrite.md](pe_rewrite.md) |

## Evaluation operators

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `instruct_eval` | Run LLM-as-judge evaluation on instruction/response pairs. | `configs/eval/instruct_eval_pai_token.yaml` | [data_formats.md](data_formats.md) |
| `cot_eval` | Run LLM-as-judge evaluation on CoT reasoning traces. | `configs/eval/cot_eval_pai_token.yaml` | [data_formats.md](data_formats.md) |
| `mm_instruct_eval` | Run LLM-as-judge evaluation on multi-modal instruction responses. | `configs/eval/mm_instruct_eval_pai_token.yaml` | [data_formats.md](data_formats.md) |
| `mm_cot_eval` | Run LLM-as-judge evaluation on multi-modal CoT traces. | `configs/eval/mm_cot_eval_pai_token.yaml` | [data_formats.md](data_formats.md) |
| `t2i_eval` | Run VLM-as-judge evaluation on generated images. | `configs/eval/t2i_eval_pai_token.yaml` | [data_formats.md](data_formats.md) |
| `t2i_single_model_eval` | Single-teacher T2I evaluation with a dimension pool judge. | `configs/eval/t2i_ti2i/t2i_single_model_pai_token.yaml` | [t2i_ti2i_eval.md](t2i_ti2i_eval.md) |
| `t2i_multi_model_eval` | Multi-teacher T2I evaluation with cross-model debate. | `configs/eval/t2i_ti2i/t2i_multi_model_pai_token.yaml` | [t2i_ti2i_eval.md](t2i_ti2i_eval.md) |
| `ti2i_single_model_eval` | Single-teacher TI2I evaluation with a dimension pool judge. | `configs/eval/t2i_ti2i/ti2i_single_model_pai_token.yaml` | [t2i_ti2i_eval.md](t2i_ti2i_eval.md) |
| `ti2i_multi_model_eval` | Multi-teacher TI2I evaluation with cross-model debate. | `configs/eval/t2i_ti2i/ti2i_multi_model_pai_token.yaml` | [t2i_ti2i_eval.md](t2i_ti2i_eval.md) |
| `t2v_eval` | Run T2V video evaluation (precheck, VLM judge, optional omni check). | Reuses a T2V pipeline config with `resume`/`eval` enabled, or `configs/eval/t2v/vlm_dimensions.yaml` for dimension definitions. | [t2v_distillation.md](t2v_distillation.md) |

## T2I/T2V generation operators

| `job_type` | Purpose | Representative config | Documentation |
|---|---|---|---|
| `prompt_optimize` | Optimize seed T2I prompts into rich, descriptive prompts. | `configs/t2i/prompt_optimize_pai_token.yaml` | [t2i_distillation.md](t2i_distillation.md) |
| `t2i_generation` | Generate images from prompts via a T2I backend (no SFT building). | `configs/t2i/t2i_generation_wanx.yaml` | [t2i_distillation.md](t2i_distillation.md) |
| `t2v_prompt_optimize` | Two-stage T2V/I2V prompt optimization. | Reuses `configs/basic/t2v_distill_pai_token.yaml` with prompt-optimization stages enabled. | [t2v_distillation.md](t2v_distillation.md) |
| `t2v_generation` | Generate videos from prompts via a T2V backend (no SFT building). | Reuses `configs/basic/t2v_distill_pai_token.yaml` with generation stage enabled. | [t2v_distillation.md](t2v_distillation.md) |

## Notes

- Config paths ending in `_pai_token.yaml` have `_pai_eas.yaml` counterparts under the same directory.
- T2I configs use backend-specific variants: `_wanx.yaml`, `_qwen_image.yaml`, or `_pai_diffusion.yaml`.
- Some standalone operators do not have dedicated config files; they reuse pipeline configs and are enabled by setting the appropriate stage flags or by resuming from an intermediate JSONL.
