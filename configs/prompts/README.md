# Prompts

This directory contains prompt templates used by EasyDistill 2 operators. Two file formats are used intentionally:

- **`.txt` files** — single prompt templates used by synthesis, rewrite, and generation operators. They are loaded via `prompt_template_file` in the operator config.
- **`.yaml` files** — structured prompt collections used by LLM-as-judge evaluators. They map metric names to judge prompts and are loaded via `prompts_file` in the eval config.

| File | Type | Used by |
|---|---|---|
| `expansion_prompt.txt` | template | `instruction_expansion` |
| `refinement_prompt.txt` | template | `instruction_refinement` |
| `extraction_prompt.txt` | template | `instruction_response_extraction` |
| `cot_generation_prompt.txt` | template | `cot_distill`, `mm_cot_distill` |
| `cot_long2short_prompt.txt` | template | `cot_long2short`, `mm_cot_long2short` |
| `cot_short2long_prompt.txt` | template | `cot_short2long`, `mm_cot_short2long` |
| `mm_cot_thinking_prompt.txt` | template | `mm_cot_distill` (OmniThoughtV-style `<thinking>/<answer>` traces) |
| `mm_generation_prompt.txt` | system prompt | `mm_instruct_distill` |
| `t2i_prompt_optimize_prompt.txt` | template | `prompt_optimize`, `advanced_t2i_distill` |
| `agent_task_synthesis_prompt.txt` | template | `agent_distill` (task synthesis stage) |
| `agent_fuzzy_task_prompt.txt` | template | `agent_distill` (fuzzy task stage) |
| `agent_tool_check_prompt.txt` | template | `agent_distill` (tool check stage) |
| `agent_solve_system_prompt.txt` | system prompt | `agent_distill` (trajectory rollout) |
| `agent_mock_tool_prompt.txt` | template | `agent_distill` (trajectory rollout) |
| `agent_mock_user_prompt.txt` | template | `agent_distill` (trajectory rollout) |
| `agent_rubrics_prompt.txt` | template | `agent_distill` (rubric comparison stage) |
| `default_eval_prompts.yaml` | metric prompts | `instruct_eval`, `advanced_instruct_distill`, `augmented_instruct_distill`, `balanced_instruct_distill` |
| `default_cot_eval_prompts.yaml` | metric prompts | `cot_eval`, `advanced_cot_distill`, `cot_rvcd_score` |
| `t2i_eval_prompts.yaml` | metric prompts | `t2i_eval`, `advanced_t2i_distill` |
| `t2i_single_model_prompts.yaml` | judge prompts | `easydistill.eval.t2i_single_model` |
| `t2i_multi_model_prompts.yaml` | judge prompts | `easydistill.eval.t2i_multi_model` |
| `ti2i_single_model_prompts.yaml` | judge prompts | `easydistill.eval.ti2i_single_model` |
| `ti2i_multi_model_prompts.yaml` | judge prompts | `easydistill.eval.ti2i_multi_model` |

To customize a single prompt, copy the relevant `.txt` file, edit it, and point `prompt_template_file` to your copy. To customize evaluation metrics or judge prompts, copy the relevant `.yaml` file and update `prompts_file`.

---

# 提示词

本目录包含 EasyDistill 2 各算子使用的提示词模板。我们有意使用两种文件格式：

- **`.txt` 文件** —— 单条提示词模板，用于合成、改写与生成算子。在算子配置中通过 `prompt_template_file` 加载。
- **`.yaml` 文件** —— 结构化提示词集合，用于 LLM-as-judge 评估器。将指标名映射到裁判提示词，在评估配置中通过 `prompts_file` 加载。

| 文件 | 类型 | 使用方 |
|---|---|---|
| `expansion_prompt.txt` | 单模板 | `instruction_expansion` |
| `refinement_prompt.txt` | 单模板 | `instruction_refinement` |
| `extraction_prompt.txt` | 单模板 | `instruction_response_extraction` |
| `cot_generation_prompt.txt` | 单模板 | `cot_distill`、`mm_cot_distill` |
| `cot_long2short_prompt.txt` | 单模板 | `cot_long2short`、`mm_cot_long2short` |
| `cot_short2long_prompt.txt` | 单模板 | `cot_short2long`、`mm_cot_short2long` |
| `mm_cot_thinking_prompt.txt` | 单模板 | `mm_cot_distill`（OmniThoughtV 风格 `<thinking>/<answer>` 轨迹） |
| `mm_generation_prompt.txt` | 系统提示词 | `mm_instruct_distill` |
| `t2i_prompt_optimize_prompt.txt` | 单模板 | `prompt_optimize`、`advanced_t2i_distill` |
| `agent_task_synthesis_prompt.txt` | 单模板 | `agent_distill`（任务合成阶段） |
| `agent_fuzzy_task_prompt.txt` | 单模板 | `agent_distill`（模糊任务阶段） |
| `agent_tool_check_prompt.txt` | 单模板 | `agent_distill`（工具校验阶段） |
| `agent_solve_system_prompt.txt` | 系统提示词 | `agent_distill`（轨迹采样） |
| `agent_mock_tool_prompt.txt` | 单模板 | `agent_distill`（轨迹采样） |
| `agent_mock_user_prompt.txt` | 单模板 | `agent_distill`（轨迹采样） |
| `agent_rubrics_prompt.txt` | 单模板 | `agent_distill`（rubric 对比阶段） |
| `default_eval_prompts.yaml` | 指标提示词 | `instruct_eval`、`advanced_instruct_distill`、`augmented_instruct_distill`、`balanced_instruct_distill` |
| `default_cot_eval_prompts.yaml` | 指标提示词 | `cot_eval`、`advanced_cot_distill`、`cot_rvcd_score` |
| `t2i_eval_prompts.yaml` | 指标提示词 | `t2i_eval`、`advanced_t2i_distill` |
| `t2i_single_model_prompts.yaml` | 裁判提示词 | `easydistill.eval.t2i_single_model` |
| `t2i_multi_model_prompts.yaml` | 裁判提示词 | `easydistill.eval.t2i_multi_model` |
| `ti2i_single_model_prompts.yaml` | 裁判提示词 | `easydistill.eval.ti2i_single_model` |
| `ti2i_multi_model_prompts.yaml` | 裁判提示词 | `easydistill.eval.ti2i_multi_model` |

如需自定义单条提示词，可复制对应 `.txt` 文件并修改，然后将 `prompt_template_file` 指向你的副本。如需自定义评估指标或裁判提示词，可复制对应 `.yaml` 文件并更新 `prompts_file`。
