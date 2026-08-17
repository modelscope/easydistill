# T2I 文生图蒸馏

EasyDistill 2 支持 T2I 文生图蒸馏：将种子文本提示转化为多模态 SFT 训练数据，其中助手回复为生成的图片。核心流水线为：

**种子 prompt → Prompt 优化 → T2I 教师生成 → VLM-as-judge 评测 → 质量过滤 → 多模态 SFT 数据集**

产出格式为 `{messages: [{role: user, content: 优化后prompt}, {role: assistant, content: [image_url]}]}`，兼容 LLaMA-Factory 与 ms-swift 多模态训练格式。

> 本文档是**用户使用/运行手册**：介绍如何运行 T2I 文生图蒸馏任务并配置后端。关于架构设计、各阶段数据 schema 与扩展指南，请见**实现文档** [t2i_distillation_implementation.md](t2i_distillation_implementation.md)。

## 架构设计概览

T2I 蒸馏使用**两个独立后端**：

| 后端 | 配置键 | 类型 | 用途 |
|------|--------|------|------|
| VLM 后端 | `backend` | `ModelBackend` | Prompt 优化与图片评测（如 Qwen-VL） |
| T2I 后端 | `t2i_backend` | `T2IBackend` | 图片生成（如通义万相、Qwen-Image、PAI-Diffusion） |

T2I 后端使用的 API 协议与 chat completions 不同，因此有独立的抽象层（`T2IBackend`），与 `ModelBackend` 平行。

## 支持的 T2I 后端

| 后端 | 配置 `type` | SDK | 说明 |
|------|------------|-----|------|
| 通义万相（Wanx） | `wanx` | dashscope | 异步任务提交 + 轮询。默认模型：`wanx2.1-t2i-turbo`。 |
| Qwen-Image | `qwen_image` | dashscope | 异步任务提交 + 轮询。默认模型：`qwen-image2.0-pro`。 |
| PAI-Diffusion | `pai_diffusion` | httpx | PAI-EAS 部署的 SD/Flux，走 OpenAI 兼容 `/images/generations` 端点。 |

### 通义万相后端

```yaml
t2i_backend:
  type: wanx
  api_key: ${DASHSCOPE_API_KEY}
  model_id: wanx2.1-t2i-turbo
```

需要安装 `dashscope` 包：`pip install easydistill[t2i]`。

万相 API 是异步的：生成请求会提交一个任务，后端轮询直到任务完成并返回图片 URL。

### Qwen-Image 后端

```yaml
t2i_backend:
  type: qwen_image
  api_key: ${DASHSCOPE_API_KEY}
  model_id: qwen-image2.0-pro
```

需要安装 `dashscope` 包：`pip install easydistill[t2i]`。

Qwen-Image API 与万相使用相同的异步任务协议：提交任务、轮询直到完成、返回图片 URL。

### PAI-Diffusion 后端

```yaml
t2i_backend:
  type: pai_diffusion
  endpoint_url: ${EAS_ENDPOINT_URL}
  token: ${EAS_TOKEN}
  model_id: stable-diffusion-xl
```

使用 httpx 直接调用 OpenAI 兼容的 `/images/generations` 端点，无需额外 SDK。

## 数据格式

### 输入：种子 prompt

输入行为包含 `prompt` 文本字段的 JSON 对象：

```jsonl
{"id": "1", "prompt": "一只在月球上喝咖啡的猫"}
{"id": "2", "prompt": "赛博朋克风格的城市街道，霓虹灯闪烁，下着小雨"}
```

可通过 `dataset.prompt_key` 覆盖默认键名。

### 输出：多模态 SFT

输出行为 ShareGPT 格式的 SFT 消息，助手内容为包含图片 URL 的多模态列表：

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

各阶段中间 JSONL 格式（prompt 优化、生成、评测）见 [t2i_distillation_implementation.md](t2i_distillation_implementation.md)。

## 独立操作

### 基础 T2I 蒸馏（`t2i_distill`）

最简单的 T2I 流程——将种子 prompt 直接发送到 T2I 后端并构建 SFT 数据。无 Prompt 优化，无评测。

```bash
# 通义万相
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_distill_wanx.yaml

# Qwen-Image
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_distill_qwen_image.yaml
```

### Prompt 优化（`prompt_optimize`）

使用 LLM/VLM 将简单的种子 prompt 改写为适合 T2I 模型的丰富描述性 prompt。

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/t2i/prompt_optimize_pai_token.yaml

# PAI-EAS 自部署端点
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/t2i/prompt_optimize_pai_eas.yaml
```

优化器会补充主体、风格、构图、光影和画质等描述。输出行包含 `raw_prompt` 和 `optimized_prompt` 字段。

### T2I 生成（`t2i_generation`）

将 prompt 发送到 T2I 后端并保存生成的图片 URL。不构建 SFT 数据。

```bash
# 通义万相
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_generation_wanx.yaml

# Qwen-Image
export DASHSCOPE_API_KEY=your_key
easydistill --config configs/t2i/t2i_generation_qwen_image.yaml
```

### T2I 评测（`t2i_eval`）

使用 VLM-as-judge 对生成的图片按多个质量维度打分。

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/eval/t2i_eval_pai_token.yaml

# PAI-EAS 自部署端点
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/eval/t2i_eval_pai_eas.yaml
```

## 评测维度

所有指标由 VLM 裁判按 0–9 分打分。评测器实现与自定义评测维度方法请见 [t2i_distillation_implementation.md](t2i_distillation_implementation.md)。

| 指标 | 说明 |
|------|------|
| `prompt_consistency` | 图片与 prompt 的语义一致性。 |
| `aesthetic_quality` | 美学质量：构图、色彩、光影。 |
| `detail_richness` | 细节丰富度与纹理。 |
| `artifact_absence` | 无伪影与变形。 |

可通过 `eval.prompts_file` 提供自定义评测 prompt（参见 `configs/prompts/t2i_eval_prompts.yaml`）。

## 高级 T2I 蒸馏流水线

推荐的 T2I 流水线一键运行所有阶段：

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
export PAI_TOKEN_API_KEY=your_pai_token_key

# 通义万相
easydistill --config configs/t2i/advanced_t2i_distill_wanx.yaml

# Qwen-Image
easydistill --config configs/t2i/advanced_t2i_distill_qwen_image.yaml
```

流水线阶段：

1. `prompt_optimize` — 通过 LLM/VLM 优化种子 prompt。
2. `t2i_generate` — 使用 T2I 后端从优化后 prompt 生成图片。
3. `t2i_eval` — VLM-as-judge 质量打分。
4. `quality_filter` — 按最低分数或 top-k/top-ratio 过滤。
5. `build_t2i_sft` — 转化为多模态 SFT 数据集。

每个阶段的中间输出会写入 `pipeline[].output_path` 指定的路径。

## 配置参考

本表覆盖最常用的字段。完整 YAML 结构与阶段级覆盖规则请见 [t2i_distillation_implementation.md](t2i_distillation_implementation.md)。

| 字段 | 说明 |
|------|------|
| `backend.type` | VLM 后端类型：`pai_token`、`pai_eas` 或 `openai` |
| `backend.model_id` | VLM 模型 ID（如 `qwen-vl-max`） |
| `t2i_backend.type` | T2I 后端类型：`wanx`、`qwen_image` 或 `pai_diffusion` |
| `t2i_backend.api_key` | 万相/Qwen-Image API Key（环境变量：`DASHSCOPE_API_KEY`） |
| `t2i_backend.endpoint_url` | PAI-Diffusion 端点 URL（环境变量：`EAS_ENDPOINT_URL`） |
| `t2i_backend.token` | PAI-Diffusion Token（环境变量：`EAS_TOKEN`） |
| `t2i_backend.model_id` | T2I 模型 ID |
| `generation.size` | 图片尺寸（如 `"1024*1024"`） |
| `generation.n` | 每个 prompt 生成的图片数 |
| `generation.max_workers` | 并发 T2I API 调用数 |
| `generation.retry_attempts` | 瞬时错误重试次数 |
| `eval.metrics` | 评测指标列表 |
| `eval.temperature` | VLM 裁判温度（建议 0.0） |
| `eval.max_tokens` | VLM 裁判回复最大 token 数 |
| `sft.skip_empty` | 跳过无图片或空 prompt 的行 |
| `sft.min_prompt_length` | 最小 prompt 长度（字符数） |
| `sft.max_images_per_prompt` | 每个 SFT 样本最大图片数（默认 1） |
| `dataset.input_path` | 输入 JSONL 路径 |
| `dataset.output_path` | 输出 JSONL 路径 |
| `dataset.prompt_key` | prompt 字段键名（默认：`prompt`） |

## 安装

```bash
# 核心安装
pip install -e .

# 含 T2I（万相）支持
pip install -e ".[t2i]"

# 含所有可选依赖
pip install -e ".[all]"
```
