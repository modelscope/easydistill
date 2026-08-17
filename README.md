# EasyDistill 2

EasyDistill 2 is a config-driven knowledge distillation toolkit for turning black-box teacher models into high-quality training data. It composes atomic operators—generation, evaluation, filtering, rewriting, balancing, and preference scoring—into end-to-end pipelines that produce SFT datasets, DPO preference pairs, and multi-modal training data.

EasyDistill 2 是一个配置驱动的知识蒸馏工具包，用于将黑盒教师模型转化为高质量训练数据。它将生成、评估、过滤、改写、平衡与偏好评分等原子算子组合为端到端流水线，产出 SFT 数据集、DPO 偏好对以及多模态训练数据。

> **Upgrading from 1.0 / 从 1.0 升级**: EasyDistill 2 is a rewrite and is not backward compatible with 1.0 configs or APIs. The 1.0 implementation remains available on the [`1.0`](https://github.com/modelscope/easydistill/tree/1.0) branch and the `v1.0.0` tag. / EasyDistill 2 为重写版本，与 1.0 的配置和 API 不向后兼容。1.0 实现保留在 [`1.0`](https://github.com/modelscope/easydistill/tree/1.0) 分支与 `v1.0.0` 标签。

Key capabilities / 核心能力：

- **Backend-agnostic generation / 后端无关的生成**: works with any OpenAI-compatible endpoint—OpenAI API, Azure OpenAI, vLLM, local servers, PAI-Token, and PAI-EAS. / 支持任意 OpenAI 兼容端点：OpenAI API、Azure OpenAI、vLLM、本地服务、PAI-Token 与 PAI-EAS。
- **SFT and DPO data / SFT 与 DPO 数据**: build supervised fine-tuning datasets or chosen/rejected preference pairs for alignment. / 构建监督微调数据集或用于对齐训练的 chosen/rejected 偏好对。
- **Text and multi-modal / 文本与多模态**: distill instructions, chain-of-thought reasoning, vision-language conversations, agent trajectories, and text-to-image data. / 蒸馏指令、思维链推理、视觉语言对话、Agent 轨迹与文生图数据。
- **Composable pipelines / 可组合流水线**: use ready-made end-to-end pipelines or mix individual operators to match your data strategy. / 使用开箱即用的端到端流水线，或自由组合单个算子以匹配数据策略。
- **Training-ready outputs / 训练就绪输出**: exported formats work directly with LLaMA-Factory and ms-swift. / 导出格式可直接用于 LLaMA-Factory 与 ms-swift。

## Installation / 安装

```bash
pip install -e .
```

For the PAI-Token and PAI-EAS backends you also need the `openai` package: / 若使用 PAI-Token 和 PAI-EAS 后端，还需安装 `openai` 包：

```bash
pip install openai
```

## Quick start / 快速开始

Every job uses the same CLI entry point. The `job_type` field in the config selects the workflow, and each workflow ships paired `_pai_token` / `_pai_eas` configs.

每个任务使用相同的 CLI 入口。配置文件中的 `job_type` 字段选择具体工作流，每个工作流都提供成对的 `_pai_token` / `_pai_eas` 配置。

```bash
# 1. Set credentials for your backend / 按所用后端设置凭据
export PAI_TOKEN_API_KEY=your_key            # PAI-Token
export PAI_TOKEN_BASE_URL=https://your-endpoint/v1   # PAI-Token (optional / 可选)

export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1   # PAI-EAS
export EAS_TOKEN=your_token                  # PAI-EAS

export DASHSCOPE_API_KEY=your_key            # T2I (Wanx / Qwen-Image)

export EAS_VIDEO_ENDPOINT_URL=https://your-video-service.pai-eas.aliyuncs.com   # T2V (EAS video / EAS 视频)
export EAS_VIDEO_TOKEN=your_token            # T2V (EAS video / EAS 视频)

# 2. Run any workflow / 运行任意工作流
easydistill --config configs/basic/instruct_distill_pai_token.yaml
```

Discover supported jobs and models without a config:

无需配置文件即可查看支持的作业与模型：

```bash
easydistill --list-jobs      # list all job_type values / 列出所有 job_type
easydistill --list-models    # list Model Zoo entries / 列出模型库
```

Pick a config from the tables below; each linked doc covers stage-level options and data formats.

从下方表格选择配置；各阶段选项与数据格式见对应文档。

## Workflows / 工作流

### Basic distillation / 基础蒸馏

Single-stage distillation: generate teacher outputs for seed data and build an SFT dataset. / 单阶段蒸馏：为种子数据生成教师输出并构建 SFT 数据集。

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `instruct_distill` | Teacher responses for seed instructions. / 为种子指令生成教师回复。 | `configs/basic/instruct_distill_pai_token.yaml` | [EN](docs/instruction_distillation.md) · [中文](docs/instruction_distillation_zh.md) |
| `cot_distill` | Chain-of-thought reasoning traces. / 生成思维链推理轨迹。 | `configs/basic/cot_distill_pai_token.yaml` | [EN](docs/cot_distillation.md) · [中文](docs/cot_distillation_zh.md) |
| `mm_instruct_distill` | Responses for `(image, instruction)` pairs. / 为 `(图像, 指令)` 样本对生成回复。 | `configs/basic/mm_instruct_distill_pai_token.yaml` | [EN](docs/mm_distillation.md) · [中文](docs/mm_distillation_zh.md) |
| `mm_cot_distill` | Visual chain-of-thought traces. / 生成视觉思维链轨迹。 | `configs/basic/mm_cot_distill_pai_token.yaml` | [EN](docs/mm_cot_distillation.md) · [中文](docs/mm_cot_distillation_zh.md) |

### Instruction pipelines / 指令蒸馏流水线

End-to-end pipelines chaining synthesis, generation, evaluation, filtering, and SFT building. / 将合成、生成、评估、过滤与 SFT 构建串联的端到端流水线。

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `advanced_instruct_distill` | Expand → generate → judge → filter → SFT. / 扩充 → 生成 → 裁判 → 过滤 → SFT。 | `configs/pipeline/advanced_instruct_distill_pai_token.yaml` | [EN](docs/advanced_instruct_distill.md) · [中文](docs/advanced_instruct_distill_zh.md) |
| `balanced_instruct_distill` | Balance category distribution before generation. / 生成前先平衡指令类别分布。 | `configs/pipeline/balanced_instruct_distill_pai_token.yaml` | [EN](docs/balanced_instruct_distill.md) · [中文](docs/balanced_instruct_distill_zh.md) |
| `augmented_instruct_distill` | Refine seeds, then generate and distill. / 精练种子指令后生成并蒸馏。 | `configs/pipeline/augmented_instruct_distill_pai_token.yaml` | [EN](docs/augmented_instruct_distill.md) · [中文](docs/augmented_instruct_distill_zh.md) |

Standalone synthesis operators (`instruction_expansion`, `seed_anchored_expansion`, `instruction_refinement`, `instruction_response_extraction`) live in `configs/rewrite/`; see [instruction_balancing.md](docs/instruction_balancing.md).

独立的合成算子（`instruction_expansion`、`seed_anchored_expansion`、`instruction_refinement`、`instruction_response_extraction`）位于 `configs/rewrite/`，参见 [instruction_balancing_zh.md](docs/instruction_balancing_zh.md)。

### CoT distillation / 思维链蒸馏

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `advanced_cot_distill` | Generate CoT, score with RV/CD, mix by difficulty bins, build SFT. / 生成 CoT，按 RV/CD 评分、按难度分箱混合并构建 SFT。 | `configs/pipeline/advanced_cot_distill_pai_token.yaml` | [EN](docs/cot_rvcd_mixer.md) · [中文](docs/cot_rvcd_mixer_zh.md) |
| `cot_long2short` / `cot_short2long` | Rewrite CoT length in either direction. / 思维链长转短与短转长改写。 | `configs/rewrite/cot_long2short_pai_token.yaml` | [EN](docs/cot_distillation.md) · [中文](docs/cot_distillation_zh.md) |

### Multi-modal pipelines / 多模态流水线

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `advanced_mm_distill` | Multi-modal generate → eval → filter → SFT. / 多模态生成 → 评估 → 过滤 → SFT。 | `configs/pipeline/advanced_mm_distill_pai_token.yaml` | [EN](docs/mm_distillation.md) · [中文](docs/mm_distillation_zh.md) |
| `advanced_mm_cot_distill` | Visual CoT with RV/CD/correctness scoring. / 视觉思维链，含 RV/CD/正确性评分。 | `configs/pipeline/advanced_mm_cot_distill_pai_token.yaml` | [EN](docs/mm_cot_distillation.md) · [中文](docs/mm_cot_distillation_zh.md) |
| `advanced_mm_cot_distill` (OmniThoughtV config / OmniThoughtV 配置变体) | Same job_type as above; reproduces the OmniThoughtV `<thinking>/<answer>` trace recipe. / 与上行为同一 job_type；复现 OmniThoughtV `<thinking>/<answer>` 轨迹配方。 | `configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml` | [EN](docs/mm_cot_distillation.md) · [中文](docs/mm_cot_distillation_zh.md) |
| `mm_cot_long2short` / `mm_cot_short2long` | Rewrite visual CoT length. / 视觉思维链长度改写。 | `configs/rewrite/mm_cot_long2short_pai_token.yaml` | [EN](docs/mm_cot_distillation.md) · [中文](docs/mm_cot_distillation_zh.md) |

### Agent distillation / Agent 蒸馏

Synthesize virtual tool-use tasks from persona seeds, roll out multi-turn agent trajectories with a LangGraph ReAct loop, and build SFT or DPO training data.

从角色种子合成虚拟工具使用任务，通过 LangGraph ReAct 循环展开多轮 Agent 轨迹，并构建 SFT 或 DPO 训练数据。

| `job_type` | Config | Docs |
|---|---|---|
| `agent_distill` | `configs/pipeline/agent_distill_pai_token.yaml` | [EN](docs/agent_distillation.md) · [中文](docs/agent_distillation_zh.md) |

### Search agent distillation / 搜索 Agent 蒸馏

Evolve simple seed QA pairs into verified multi-hop search tasks through a Strategist-driven closed loop (expand via atomic-QA entity bridging, refine, quality gate, solver-verified difficulty rating), roll out tool-using ReAct search trajectories (mock-simulated or real Google/Jina search), and build SFT training data.

通过策略师驱动的闭环将简单种子问答进化为经验证的多跳搜索任务（原子 QA 实体桥接加跳、受控模糊化、质量门禁、解题实测定难度），展开带工具（模拟或真实 Google/Jina 检索）的 ReAct 搜索轨迹，并构建 SFT 训练数据。

| `job_type` | Config | Docs |
|---|---|---|
| `search_agent_distill` | `configs/pipeline/search_agent_distill_pai_token.yaml` | [EN](docs/search_agent_distillation.md) · [中文](docs/search_agent_distillation_zh.md) |

### T2I distillation / 文生图蒸馏

Distill seed text prompts into multi-modal SFT data with generated images. Supports Tongyi Wanxiang (Wanx), Qwen-Image, and PAI-Diffusion backends via the `t2i_backend` config section.


将种子文本提示蒸馏为含生成图片的多模态 SFT 数据。通过 `t2i_backend` 配置节支持通义万相（Wanx）、Qwen-Image 与 PAI-Diffusion 后端。

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `t2i_distill` | Basic text-to-image distillation. / 基础文生图蒸馏。 | `configs/t2i/t2i_distill_wanx.yaml` | [EN](docs/t2i_distillation.md) · [中文](docs/t2i_distillation_zh.md) |
| `advanced_t2i_distill` | Prompt optimization → T2I generation → VLM judge → filter → SFT. / Prompt 优化 → 文生图 → VLM 裁判 → 过滤 → SFT。 | `configs/t2i/advanced_t2i_distill_wanx.yaml` | [EN](docs/t2i_distillation.md) · [中文](docs/t2i_distillation_zh.md) |

### T2V distillation / 文生视频蒸馏

Distill seed text prompts — optionally conditioned on a first-frame image (I2V) — into multi-modal SFT data with generated videos. Supports PAI-Token gateway video models and PAI-EAS deployed video models via the `t2v_backend` config section; T2V and I2V rows may be mixed in one batch. Expensive stages support opt-in resume (`resume: true`) with row-level checkpointing.

将种子文本提示（可选以首帧图像为条件，即 I2V）蒸馏为含生成视频的多模态 SFT 数据。通过 `t2v_backend` 配置节支持 PAI-Token 网关视频模型与 PAI-EAS 自部署视频模型；T2V 与 I2V 行可混合在同一批数据中。高开销阶段支持可选断点续跑（`resume: true`）及行级 checkpoint。

| `job_type` | Purpose / 用途 | Config | Docs |
|---|---|---|---|
| `t2v_distill` | Basic text/image-to-video distillation. / 基础文生/图生视频蒸馏。 | `configs/basic/t2v_distill_pai_token.yaml` | [EN](docs/t2v_distillation.md) · [中文](docs/t2v_distillation_zh.md) |
| `advanced_t2v_distill` | Prompt optimization → video generation → video eval (VLM/omni/VBench) → filter → SFT. / Prompt 优化 → 视频生成 → 视频评估（VLM/omni/VBench）→ 过滤 → SFT。 | `configs/pipeline/advanced_t2v_distill_pai_token.yaml` | [EN](docs/t2v_distillation.md) · [中文](docs/t2v_distillation_zh.md) |
| `t2v_prompt_optimize` / `t2v_generation` / `t2v_eval` | Standalone single stages for debugging or resuming. / 独立单阶段，便于调试或续跑。 | — | [EN](docs/t2v_distillation.md) · [中文](docs/t2v_distillation_zh.md) |

### PE rewrite distillation / PE 改写蒸馏

Expand seed prompts, rewrite them via a plan/rewrite/reflection teacher agent chain, score with a combined nine-metric LLM judge, filter, and build a prompt-rewriting SFT dataset. Every stage is also exposed as a standalone `job_type` (`seed_anchored_expansion`, `agentic_rewrite`, `pe_rewrite_eval`, `pe_rewrite_filter`, `pe_rewrite_build_sft`) for debugging or resuming from an intermediate JSONL.

扩展种子 prompt，经 plan/rewrite/reflection 教师 Agent 链改写，由九维合并 LLM 裁判打分、过滤并构建 prompt 改写 SFT 数据集。每个阶段也均以独立 `job_type` 透出（`seed_anchored_expansion`、`agentic_rewrite`、`pe_rewrite_eval`、`pe_rewrite_filter`、`pe_rewrite_build_sft`），便于调试或从中间 JSONL 续跑。

| `job_type` | Config | Docs |
|---|---|---|
| `pe_rewrite_distill` | `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` | [EN](docs/pe_rewrite.md) · [中文](docs/pe_rewrite_zh.md) |

### Preference data / 偏好数据

Build chosen/rejected preference pairs for direct preference optimization (DPO). / 构建用于直接偏好优化（DPO）的 chosen/rejected 偏好对。

| `job_type` | Variant / 变体 | Purpose / 用途 | Config | Docs |
|---|---|---|---|---|
| `dpo_data_build` | `dpo_instruct_*` | Preference pairs for instruction data. / 指令数据偏好对。 | `configs/preference/dpo_instruct_pai_token.yaml` | [EN](docs/dpo_distillation.md) · [中文](docs/dpo_distillation_zh.md) |
| `dpo_data_build` | `dpo_cot_*` | Preference pairs for CoT data. / 思维链数据偏好对。 | `configs/preference/dpo_cot_pai_token.yaml` | [EN](docs/dpo_distillation.md) · [中文](docs/dpo_distillation_zh.md) |

The `dpo_instruct_*` and `dpo_cot_*` names are config-file naming patterns; the actual CLI `job_type` is always `dpo_data_build`. Set `dpo.task_type` inside the config to `instruct` or `cot`.

`dpo_instruct_*` 与 `dpo_cot_*` 是配置文件命名模式；实际 CLI `job_type` 始终为 `dpo_data_build`。在配置内通过 `dpo.task_type` 选择 `instruct` 或 `cot`。

### Evaluation / 评估

Score existing datasets with LLM/VLM judges, without regenerating them. / 使用 LLM/VLM 裁判为已有数据集打分，无需重新生成。

| Task / 任务 | Config | Docs |
|---|---|---|
| Instruction / CoT / multi-modal / T2I judging | `configs/eval/` | [EN](docs/data_formats.md) · [中文](docs/data_formats_zh.md) |
| T2I / TI2I image evaluation / 文生图与图文生图评测 | `configs/eval/t2i_ti2i/` | [EN](docs/t2i_ti2i_eval.md) · [中文](docs/t2i_ti2i_eval_zh.md) |

### Standalone operators / 独立算子

Every pipeline stage is also exposed as a standalone `job_type` for debugging, resuming from an intermediate JSONL, or building custom workflows. / 每个流水线阶段均作为独立 `job_type` 透出，便于调试、从中间 JSONL 续跑或构建自定义工作流。

| Category / 类别 | `job_type` | Config example / 配置示例 | Docs |
|---|---|---|---|
| Instruction synthesis / 指令合成 | `instruction_expansion`, `seed_anchored_expansion`, `instruction_refinement`, `instruction_response_extraction` | `configs/rewrite/instruction_expansion_pai_token.yaml` | [instruction_balancing.md](docs/instruction_balancing.md) |
| PE rewrite / PE 改写 | `agentic_rewrite`, `pe_rewrite_eval`, `pe_rewrite_filter`, `pe_rewrite_build_sft` | `configs/rewrite/pe_rewrite_eval_pai_token.yaml` | [pe_rewrite.md](docs/pe_rewrite.md) |
| CoT rewrite / CoT 改写 | `cot_long2short`, `cot_short2long` | `configs/rewrite/cot_long2short_pai_token.yaml` | [cot_distillation.md](docs/cot_distillation.md) |
| MM CoT rewrite / 多模态 CoT 改写 | `mm_cot_long2short`, `mm_cot_short2long` | `configs/rewrite/mm_cot_long2short_pai_token.yaml` | [mm_cot_distillation.md](docs/mm_cot_distillation.md) |
| T2I stages / 文生图阶段 | `prompt_optimize`, `t2i_generation`, `t2i_single_model_eval`, `t2i_multi_model_eval`, `ti2i_single_model_eval`, `ti2i_multi_model_eval`, `t2i_eval` | `configs/t2i/prompt_optimize_pai_token.yaml` | [t2i_distillation.md](docs/t2i_distillation.md), [t2i_ti2i_eval.md](docs/t2i_ti2i_eval.md) |
| T2V stages / 文生视频阶段 | `t2v_prompt_optimize`, `t2v_generation`, `t2v_eval` | `configs/basic/t2v_distill_pai_token.yaml` | [t2v_distillation.md](docs/t2v_distillation.md) |
| Evaluation / 评估 | `instruct_eval`, `cot_eval`, `mm_instruct_eval`, `mm_cot_eval` | `configs/eval/instruct_eval_pai_token.yaml` | [data_formats.md](docs/data_formats.md) |

A complete `job_type → config → doc` matrix is available in [docs/job_type_index.md](docs/job_type_index.md) / [docs/job_type_index_zh.md](docs/job_type_index_zh.md).

## Supported backends / 支持的后端

| Backend / 后端 | Config `type` | Notes / 说明 |
|---|---|---|
| OpenAI-compatible / OpenAI 兼容 | `openai` | Any OpenAI-compatible endpoint (OpenAI API, Azure OpenAI, vLLM, llama.cpp, etc.). / 任意 OpenAI 兼容端点（OpenAI API、Azure OpenAI、vLLM、llama.cpp 等）。 |
| PAI-Token | `pai_token` | OpenAI-compatible PAI-Token endpoint with API-key auth. / 使用 API Key 认证的 OpenAI 兼容 PAI-Token 端点。 |
| PAI-EAS | `pai_eas` | OpenAI-compatible EAS endpoint with token auth. / 使用 Token 认证的 OpenAI 兼容 EAS 端点。 |

For text-to-image jobs, a separate `t2i_backend` section selects the T2I backend: `wanx` (Tongyi Wanxiang via dashscope), `qwen_image` (Qwen-Image via dashscope), or `pai_diffusion` (PAI-EAS deployed diffusion models).

文生图任务通过独立的 `t2i_backend` 节选择 T2I 后端：`wanx`（通义万相，经 dashscope）、`qwen_image`（Qwen-Image，经 dashscope）或 `pai_diffusion`（PAI-EAS 部署的扩散模型）。

See [docs/backends.md](docs/backends.md) or [docs/backends_zh.md](docs/backends_zh.md) for backend configuration details, credential environment variables, and OpenAI/vLLM examples.

后端配置详情、凭据环境变量与 OpenAI/vLLM 示例见 [docs/backends.md](docs/backends.md) 或 [docs/backends_zh.md](docs/backends_zh.md)。

## Resources / 相关资源

- **[Model Zoo](docs/model_zoo.md)** — Open-source DistilQwen and AgenticQwen models and public datasets. / 开源 DistilQwen、AgenticQwen 模型与公开数据集。
- **[DistilQwen Series](docs/distilqwen_series.md)** — DistilQwen model benchmarks and download links. / DistilQwen 模型评测结果与下载链接。
- **[Papers & News](docs/papers.md)** — Academic papers and technical articles about EasyDistill. / EasyDistill 相关学术论文与技术文章。

## Documentation / 文档

**Reference / 总览与参考**

| Topic / 主题 | English | 中文 |
|---|---|---|
| Pipelines overview / 流水线总览 | [pipelines.md](docs/pipelines.md) | [pipelines_zh.md](docs/pipelines_zh.md) |
| Job type index / job_type 索引 | [job_type_index.md](docs/job_type_index.md) | [job_type_index_zh.md](docs/job_type_index_zh.md) |
| Backends / 后端 | [backends.md](docs/backends.md) | [backends_zh.md](docs/backends_zh.md) |
| Data formats / 数据格式 | [data_formats.md](docs/data_formats.md) | [data_formats_zh.md](docs/data_formats_zh.md) |
| Training guide / 训练指南 | [training_guide.md](docs/training_guide.md) | [training_guide_zh.md](docs/training_guide_zh.md) |
| Model zoo / 模型库 | [model_zoo.md](docs/model_zoo.md) | [model_zoo_zh.md](docs/model_zoo_zh.md) |
| DistilQwen series / DistilQwen 系列 | [distilqwen_series.md](docs/distilqwen_series.md) | [distilqwen_series_zh.md](docs/distilqwen_series_zh.md) |
| Papers & news / 论文与文章 | [papers.md](docs/papers.md) | [papers_zh.md](docs/papers_zh.md) |

**Text distillation / 文本蒸馏**

| Topic / 主题 | English | 中文 |
|---|---|---|
| Instruction distillation / 指令蒸馏 | [instruction_distillation.md](docs/instruction_distillation.md) | [instruction_distillation_zh.md](docs/instruction_distillation_zh.md) |
| Advanced instruction pipeline / 高级指令流水线 | [advanced_instruct_distill.md](docs/advanced_instruct_distill.md) | [advanced_instruct_distill_zh.md](docs/advanced_instruct_distill_zh.md) |
| Balanced instruction pipeline / 平衡指令流水线 | [balanced_instruct_distill.md](docs/balanced_instruct_distill.md) | [balanced_instruct_distill_zh.md](docs/balanced_instruct_distill_zh.md) |
| Augmented instruction pipeline / 增强指令流水线 | [augmented_instruct_distill.md](docs/augmented_instruct_distill.md) | [augmented_instruct_distill_zh.md](docs/augmented_instruct_distill_zh.md) |
| Instruction balancing / 指令平衡 | [instruction_balancing.md](docs/instruction_balancing.md) | [instruction_balancing_zh.md](docs/instruction_balancing_zh.md) |
| CoT distillation / 思维链蒸馏 | [cot_distillation.md](docs/cot_distillation.md) | [cot_distillation_zh.md](docs/cot_distillation_zh.md) |
| CoT RV/CD mixer / 思维链 RV/CD 混合器 | [cot_rvcd_mixer.md](docs/cot_rvcd_mixer.md) | [cot_rvcd_mixer_zh.md](docs/cot_rvcd_mixer_zh.md) |
| DPO distillation / DPO 蒸馏 | [dpo_distillation.md](docs/dpo_distillation.md) | [dpo_distillation_zh.md](docs/dpo_distillation_zh.md) |

**Multi-modal & T2I / 多模态与文生图**

| Topic / 主题 | English | 中文 |
|---|---|---|
| Multi-modal distillation / 多模态蒸馏 | [mm_distillation.md](docs/mm_distillation.md) | [mm_distillation_zh.md](docs/mm_distillation_zh.md) |
| Multi-modal CoT distillation / 多模态思维链蒸馏 | [mm_cot_distillation.md](docs/mm_cot_distillation.md) | [mm_cot_distillation_zh.md](docs/mm_cot_distillation_zh.md) |
| Agent distillation / Agent 蒸馏 | [agent_distillation.md](docs/agent_distillation.md) | [agent_distillation_zh.md](docs/agent_distillation_zh.md) |
| Search agent distillation / 搜索 Agent 蒸馏 | [search_agent_distillation.md](docs/search_agent_distillation.md) | [search_agent_distillation_zh.md](docs/search_agent_distillation_zh.md) |
| T2I distillation / 文生图蒸馏 | [t2i_distillation.md](docs/t2i_distillation.md) | [t2i_distillation_zh.md](docs/t2i_distillation_zh.md) |
| T2I / TI2I evaluation / 文生图评测 | [t2i_ti2i_eval.md](docs/t2i_ti2i_eval.md) | [t2i_ti2i_eval_zh.md](docs/t2i_ti2i_eval_zh.md) |
| PE rewrite distillation / PE 改写蒸馏 | [pe_rewrite.md](docs/pe_rewrite.md) | [pe_rewrite_zh.md](docs/pe_rewrite_zh.md) |

## Project structure / 项目结构

```
easydistill/
  backends/        # ModelBackend abstraction / 模型后端抽象
  basic/           # Basic distillation runners / 基础蒸馏运行器
  cli/             # CLI entry point and runners / CLI 入口与运行器
  data/            # Core data models / 核心数据模型
  eval/            # LLM-as-judge evaluators / LLM 裁判评估器
  models/          # Model zoo registry and model metadata / 模型仓库注册与元数据
  operators/       # Atomic generation, preference, and RV/CD operators / 原子生成、偏好与 RV/CD 算子
  pipeline/        # End-to-end pipelines / 端到端流水线
  prompts/         # Prompt loading helpers / 提示词加载工具
  rewrite/         # Instruction and CoT rewrite operators / 指令与 CoT 重写算子
  utils/           # I/O, config, and schema helpers / I/O、配置与 schema 辅助工具
configs/           # Example configs for each backend and workflow / 各后端与工作流示例配置
  t2i/             # T2I distillation configs / 文生图蒸馏配置
  basic/           # Basic distillation configs / 基础蒸馏配置
  eval/            # Evaluation configs / 评估配置
  pipeline/        # End-to-end pipeline configs / 端到端流水线配置
  preference/      # Preference distillation configs / 偏好蒸馏配置
  prompts/         # Prompt templates and eval prompt collections / 提示词模板与裁判提示词集合
  rewrite/         # Rewrite operator configs / 重写算子配置
examples/          # Seed instruction and problem examples / 种子指令与问题示例
docs/              # Documentation / 文档
tests/             # Unit and smoke tests / 单元与冒烟测试
```

## License / 许可证

This project is licensed under the [Apache License, Version 2.0](LICENSE). / 本项目采用 [Apache License, Version 2.0](LICENSE) 许可证。

See the [NOTICE](NOTICE) file for attribution and copyright information. / 归属与版权信息见 [NOTICE](NOTICE) 文件。
