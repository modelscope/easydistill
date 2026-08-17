# 多模态思维链蒸馏（MMCoT）

EasyDistill2 支持多模态问题的思维链蒸馏。MMCoT 模块可以生成、改写、评估和筛选视觉推理轨迹，以构建视觉语言模型的 SFT 数据集。

仅支持 API 后端：`pai_token` 和 `pai_eas`。

完整 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 数据格式

输入每行是一个包含问题文本和 `images` 图像列表的 JSON 对象：

```jsonl
{"id": "mmcot_0", "instruction": "如何确定主导颜色？", "images": ["examples/mm_sample_image.png"]}
```

图像引用规则与 MMKD 相同：本地路径、URL 或 base64 数据 URL。

## 独立生成

使用 `job_type: mm_cot_distill` 生成多模态 CoT 轨迹。

```bash
python -m easydistill.cli --config configs/basic/mm_cot_distill_pai_token.yaml
```

示例配置：

```yaml
job_type: mm_cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen-vl-max

cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
  system_prompt: "你是一个有帮助的视觉推理助手。"
  temperature: 0.7
  max_tokens: 2048
  show_progress: true
  max_workers: 3

dataset:
  input_path: examples/mm_seed_cot_problems.jsonl
  output_path: outputs/mm_cot_distill_pai_token.jsonl
```

输出行是 ShareGPT 格式的 SFT messages，原始 CoT 回复作为 assistant 消息，`thought` 和 `solution` 保留在 metadata 中。

## 改写 CoT 轨迹

提供两个可选的改写算子：

- `mm_cot_long2short`：在保留最终答案的前提下简化推理轨迹。
- `mm_cot_short2long`：为推理轨迹补充更多细节。

两者都接受包含 `instruction`、`images` 和 `response` 字段的行，也支持自动转换 SFT `messages` 行（图像从 `metadata.images` 读取）。

```bash
python -m easydistill.cli --config configs/rewrite/mm_cot_long2short_pai_token.yaml
python -m easydistill.cli --config configs/rewrite/mm_cot_short2long_pai_token.yaml
```

## 评估

使用 `job_type: mm_cot_eval` 对 MMCoT 输出打分。

```bash
python -m easydistill.cli --config configs/eval/mm_cot_eval_pai_token.yaml
```

支持指标：`reasoning_verbosity`、`cognitive_difficulty`、`logical_correctness`。

## `advanced_mm_cot_distill` 流水线

推荐的 MMCoT 流水线与文本 CoT 流水线结构相同，但会处理图像输入。

```bash
python -m easydistill.cli --config configs/pipeline/advanced_mm_cot_distill_pai_token.yaml
```

流水线阶段：

1. `mm_cot_distill`：生成视觉推理轨迹。
2. `mm_cot_eval`：LLM 裁判打分。
3. `quality_filter`：按分数阈值筛选。
4. `build_sft`：转换为包含多模态用户消息的 ShareGPT 格式 SFT 数据。

## OmniThoughtV

[OmniThoughtV](https://modelscope.cn/datasets/platformofai/OmniThoughtV_Filter_0.5M) 是基于本 MMCoT 模块构建的大规模多模态长思维链数据集：以 [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) 为种子问题，由 Qwen-VL-max 蒸馏推理轨迹，再按推理冗余度（RV）、认知难度（CD）与逻辑正确性评分过滤，得到高质量 SFT 子集。用过滤版微调 Qwen3-VL 2B/4B/8B，在通用视觉理解（AI2D、MMStar）与推理密集型基准（MMMU-Pro、MathVerse、MathVision）上均有提升。

已发布两个版本（另见 [model_zoo_zh.md](model_zoo_zh.md)）：

| 数据集 | 规模 | 说明 |
|--------|------|------|
| `OmniThoughtV_Raw_1.8M` | 1.8M | 原始蒸馏轨迹 |
| `OmniThoughtV_Filter_0.5M` | 0.5M | 经 RV/CD 过滤、用于微调的子集 |

轨迹采用 `<thinking>...</thinking><answer>...</answer>` 格式，`mm_cot_distill` 已原生支持解析该格式（与 `<|begin_of_thought|>` 格式共存）。在自己的种子问题上复现该配方：

```bash
python -m easydistill.cli --config configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml
```

该配置使用 `configs/prompts/mm_cot_thinking_prompt.txt`（OmniThoughtV 轨迹格式）与发布数据集中使用的系统提示。可将 `backend.model_id` 替换为你的多模态教师模型；原始数据集由 Qwen-VL-max 蒸馏得到。

## 配置参考

| 字段 | 说明 |
|------|------|
| `backend.type` | `pai_token` 或 `pai_eas` |
| `backend.model_id` | 视觉语言模型 ID |
| `cot.prompt_template_file` | 提示模板路径 |
| `cot.system_prompt` | 可选系统提示 |
| `cot.temperature` | 采样温度 |
| `cot.max_tokens` | 最大回复 token 数 |
| `dataset.input_path` | 输入 JSONL 路径 |
| `dataset.output_path` | 输出 JSONL 路径 |
| `dataset.images_key` | 图像列表字段名（默认 `images`） |

使用 EAS 后端时，将 `backend.api_key` 替换为 `endpoint_url` 和 `token`。
