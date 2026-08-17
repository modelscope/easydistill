# EasyDistill 2 job_type 索引

本页将每个 CLI `job_type` 映射到代表性配置文件与相关文档。配置路径以 `pai_token` 后端为例；除非特别说明，每个文件都有同目录下的 `_pai_eas` 对应版本。

如需交互式列表，请运行 `easydistill --list-jobs`。

## 基础蒸馏

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `instruct_distill` | 为种子指令生成教师回复并构建 SFT 数据。 | `configs/basic/instruct_distill_pai_token.yaml` | [instruction_distillation_zh.md](instruction_distillation_zh.md) |
| `cot_distill` | 生成思维链推理轨迹并构建 SFT 数据。 | `configs/basic/cot_distill_pai_token.yaml` | [cot_distillation_zh.md](cot_distillation_zh.md) |
| `mm_instruct_distill` | 为 `(图像, 指令)` 样本对生成教师回复。 | `configs/basic/mm_instruct_distill_pai_token.yaml` | [mm_distillation_zh.md](mm_distillation_zh.md) |
| `mm_cot_distill` | 生成视觉思维链推理轨迹。 | `configs/basic/mm_cot_distill_pai_token.yaml` | [mm_cot_distillation_zh.md](mm_cot_distillation_zh.md) |
| `t2v_distill` | 从种子提示生成视频（T2V/I2V）并构建 SFT 数据。 | `configs/basic/t2v_distill_pai_token.yaml` | [t2v_distillation_zh.md](t2v_distillation_zh.md) |

## 端到端流水线

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `advanced_instruct_distill` | 扩充 → 生成 → 裁判 → 过滤 → SFT。 | `configs/pipeline/advanced_instruct_distill_pai_token.yaml` | [advanced_instruct_distill_zh.md](advanced_instruct_distill_zh.md) |
| `balanced_instruct_distill` | 平衡类别、生成、评估、过滤并构建 SFT。 | `configs/pipeline/balanced_instruct_distill_pai_token.yaml` | [balanced_instruct_distill_zh.md](balanced_instruct_distill_zh.md) |
| `augmented_instruct_distill` | 精炼种子、生成多回复、评估、过滤并构建 SFT。 | `configs/pipeline/augmented_instruct_distill_pai_token.yaml` | [augmented_instruct_distill_zh.md](augmented_instruct_distill_zh.md) |
| `advanced_cot_distill` | 生成 CoT，按 RV/CD 评分、按难度分箱混合并构建 SFT。 | `configs/pipeline/advanced_cot_distill_pai_token.yaml` | [cot_rvcd_mixer_zh.md](cot_rvcd_mixer_zh.md) |
| `advanced_mm_distill` | 多模态生成 → 评估 → 过滤 → SFT。 | `configs/pipeline/advanced_mm_distill_pai_token.yaml` | [mm_distillation_zh.md](mm_distillation_zh.md) |
| `advanced_mm_cot_distill` | 视觉思维链，含 RV/CD/正确性评分。 | `configs/pipeline/advanced_mm_cot_distill_pai_token.yaml` | [mm_cot_distillation_zh.md](mm_cot_distillation_zh.md) |
| `advanced_mm_cot_distill`（OmniThoughtV） | 与上行为同一 job_type；复现 OmniThoughtV 配方。 | `configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml` | [mm_cot_distillation_zh.md](mm_cot_distillation_zh.md) |
| `advanced_t2i_distill` | Prompt 优化 → 文生图 → VLM 裁判 → 过滤 → SFT。 | `configs/t2i/advanced_t2i_distill_wanx.yaml` | [t2i_distillation_zh.md](t2i_distillation_zh.md) |
| `advanced_t2v_distill` | Prompt 优化 → 视频生成 → 视频评估 → 过滤 → SFT。 | `configs/pipeline/advanced_t2v_distill_pai_token.yaml` | [t2v_distillation_zh.md](t2v_distillation_zh.md) |
| `pe_rewrite_distill` | Plan/rewrite/reflection → 裁判 → 过滤 → SFT，用于 prompt 改写。 | `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` | [pe_rewrite_zh.md](pe_rewrite_zh.md) |
| `agent_distill` | 合成工具使用任务并构建 Agent 轨迹 SFT/DPO 数据。 | `configs/pipeline/agent_distill_pai_token.yaml`（SFT）或 `configs/pipeline/agent_distill_dpo_pai_token.yaml`（DPO） | [agent_distillation_zh.md](agent_distillation_zh.md) |
| `search_agent_distill` | 将种子 QA 演化为多跳搜索任务并构建 SFT 数据。 | `configs/pipeline/search_agent_distill_pai_token.yaml` | [search_agent_distillation_zh.md](search_agent_distillation_zh.md) |

## 偏好数据

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `dpo_data_build` | 生成候选回复、打分并构建 DPO 偏好对。 | `configs/preference/dpo_instruct_pai_token.yaml`（设置 `dpo.task_type: instruct`）或 `configs/preference/dpo_cot_pai_token.yaml`（设置 `dpo.task_type: cot`） | [dpo_distillation_zh.md](dpo_distillation_zh.md) |

## 文本改写与合成算子

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `instruction_expansion` | 从种子示例合成新指令。 | `configs/rewrite/instruction_expansion_pai_token.yaml` | [instruction_balancing_zh.md](instruction_balancing_zh.md) |
| `seed_anchored_expansion` | 将每个种子扩展为同场景指令，含去重与血缘。 | `configs/rewrite/seed_anchored_expansion_pai_token.yaml` | [instruction_balancing_zh.md](instruction_balancing_zh.md) |
| `instruction_refinement` | 重写并优化现有指令。 | `configs/rewrite/instruction_refinement_pai_token.yaml` | [instruction_balancing_zh.md](instruction_balancing_zh.md) |
| `instruction_response_extraction` | 从原始文本抽取指令/回复对。 | `configs/rewrite/instruction_response_extraction_pai_token.yaml` | [instruction_balancing_zh.md](instruction_balancing_zh.md) |
| `agentic_rewrite` | 通过 plan → rewrite → reflection 教师 Agent 链改写 prompt。 | `configs/rewrite/agentic_rewrite_pai_token.yaml` | [pe_rewrite_zh.md](pe_rewrite_zh.md) |
| `cot_long2short` | 简化现有 CoT 推理轨迹。 | `configs/rewrite/cot_long2short_pai_token.yaml` | [cot_distillation_zh.md](cot_distillation_zh.md) |
| `cot_short2long` | 为现有 CoT 推理轨迹补充更多细节。 | `configs/rewrite/cot_short2long_pai_token.yaml` | [cot_distillation_zh.md](cot_distillation_zh.md) |
| `mm_cot_long2short` | 简化多模态 CoT 推理轨迹。 | `configs/rewrite/mm_cot_long2short_pai_token.yaml` | [mm_cot_distillation_zh.md](mm_cot_distillation_zh.md) |
| `mm_cot_short2long` | 为多模态 CoT 推理轨迹补充更多细节。 | `configs/rewrite/mm_cot_short2long_pai_token.yaml` | [mm_cot_distillation_zh.md](mm_cot_distillation_zh.md) |

## PE 改写流水线阶段

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `pe_rewrite_eval` | 使用多维 LLM 裁判为 prompt 改写打分。 | `configs/rewrite/pe_rewrite_eval_pai_token.yaml` | [pe_rewrite_zh.md](pe_rewrite_zh.md) |
| `pe_rewrite_filter` | 按分数阈值与 top ratio 过滤已打分改写。 | `configs/rewrite/pe_rewrite_filter.yaml` | [pe_rewrite_zh.md](pe_rewrite_zh.md) |
| `pe_rewrite_build_sft` | 从过滤后的改写构建 SFT 样本。 | `configs/rewrite/pe_rewrite_build_sft.yaml` | [pe_rewrite_zh.md](pe_rewrite_zh.md) |

## 评估算子

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `instruct_eval` | 对指令/回复对运行 LLM-as-judge 评估。 | `configs/eval/instruct_eval_pai_token.yaml` | [data_formats_zh.md](data_formats_zh.md) |
| `cot_eval` | 对 CoT 推理轨迹运行 LLM-as-judge 评估。 | `configs/eval/cot_eval_pai_token.yaml` | [data_formats_zh.md](data_formats_zh.md) |
| `mm_instruct_eval` | 对多模态指令回复运行 LLM-as-judge 评估。 | `configs/eval/mm_instruct_eval_pai_token.yaml` | [data_formats_zh.md](data_formats_zh.md) |
| `mm_cot_eval` | 对多模态 CoT 轨迹运行 LLM-as-judge 评估。 | `configs/eval/mm_cot_eval_pai_token.yaml` | [data_formats_zh.md](data_formats_zh.md) |
| `t2i_eval` | 对生成图像运行 VLM-as-judge 评估。 | `configs/eval/t2i_eval_pai_token.yaml` | [data_formats_zh.md](data_formats_zh.md) |
| `t2i_single_model_eval` | 单教师 T2I 评估，使用维度池裁判。 | `configs/eval/t2i_ti2i/t2i_single_model_pai_token.yaml` | [t2i_ti2i_eval_zh.md](t2i_ti2i_eval_zh.md) |
| `t2i_multi_model_eval` | 多教师 T2I 评估，使用模型间辩论。 | `configs/eval/t2i_ti2i/t2i_multi_model_pai_token.yaml` | [t2i_ti2i_eval_zh.md](t2i_ti2i_eval_zh.md) |
| `ti2i_single_model_eval` | 单教师 TI2I 评估，使用维度池裁判。 | `configs/eval/t2i_ti2i/ti2i_single_model_pai_token.yaml` | [t2i_ti2i_eval_zh.md](t2i_ti2i_eval_zh.md) |
| `ti2i_multi_model_eval` | 多教师 TI2I 评估，使用模型间辩论。 | `configs/eval/t2i_ti2i/ti2i_multi_model_pai_token.yaml` | [t2i_ti2i_eval_zh.md](t2i_ti2i_eval_zh.md) |
| `t2v_eval` | 运行 T2V 视频评估（预检、VLM 裁判、可选 omni 一致性检查）。 | 复用 T2V 流水线配置并启用 `resume`/`eval`，或使用 `configs/eval/t2v/vlm_dimensions.yaml` 查看维度定义。 | [t2v_distillation_zh.md](t2v_distillation_zh.md) |

## T2I/T2V 生成算子

| `job_type` | 用途 | 代表性配置 | 文档 |
|---|---|---|---|
| `prompt_optimize` | 将种子 T2I prompt 优化为 rich、描述性 prompt。 | `configs/t2i/prompt_optimize_pai_token.yaml` | [t2i_distillation_zh.md](t2i_distillation_zh.md) |
| `t2i_generation` | 通过 T2I 后端从 prompt 生成图像（不构建 SFT）。 | `configs/t2i/t2i_generation_wanx.yaml` | [t2i_distillation_zh.md](t2i_distillation_zh.md) |
| `t2v_prompt_optimize` | 两阶段 T2V/I2V prompt 优化。 | 复用 `configs/basic/t2v_distill_pai_token.yaml` 并启用 prompt 优化阶段。 | [t2v_distillation_zh.md](t2v_distillation_zh.md) |
| `t2v_generation` | 通过 T2V 后端从 prompt 生成视频（不构建 SFT）。 | 复用 `configs/basic/t2v_distill_pai_token.yaml` 并启用生成阶段。 | [t2v_distillation_zh.md](t2v_distillation_zh.md) |

## 说明

- 以 `_pai_token.yaml` 结尾的配置路径在同目录下均有 `_pai_eas.yaml` 对应版本。
- T2I 配置按后端区分变体：`_wanx.yaml`、`_qwen_image.yaml` 或 `_pai_diffusion.yaml`。
- 部分独立算子没有专属配置文件；它们复用流水线配置，通过设置对应阶段开关或从中间 JSONL 续跑来启用。
