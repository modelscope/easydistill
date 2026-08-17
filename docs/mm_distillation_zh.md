# 多模态黑盒知识蒸馏（MMKD）

EasyDistill2 支持多模态大模型（MLLM）的黑盒知识蒸馏。给定一组 `(图像, 指令)` 样本，工具会调用教师视觉语言模型，并生成 OpenAI/ShareGPT 消息格式的 SFT 训练数据。

仅支持 API 后端：`pai_token` 和 `pai_eas`。

完整 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 数据格式

输入每行是一个 JSON 对象，包含 `instruction` 文本字段和 `images` 图像列表：

```jsonl
{"id": "mm_0", "instruction": "描述你看到的内容。", "images": ["examples/mm_sample_image.png"]}
{"id": "mm_1", "instruction": "图中主要物体是什么颜色？", "images": ["https://example.com/img.png"]}
```

`images` 支持：

- 本地文件路径（相对或绝对）。
- `file:///path/to/image` 形式的 URI。
- `http(s)://` URL。
- Base64 数据 URL，例如 `data:image/png;base64,...`。

本地文件会在调用视觉 API 前自动转换为 base64 数据 URL。

## 独立生成

使用 `job_type: mm_instruct_distill` 生成教师回复。

```bash
python -m easydistill.cli --config configs/basic/mm_instruct_distill_pai_token.yaml
```

示例配置：

```yaml
job_type: mm_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen-vl-max

mm:
  system_prompt: "你是一个有帮助的视觉助手。"
  temperature: 0.7
  max_tokens: 2048
  show_progress: true
  max_workers: 3

dataset:
  input_path: examples/mm_seed_instructions.jsonl
  output_path: outputs/mm_instruct_distill_pai_token.jsonl
```

输出行是 ShareGPT 格式的 SFT messages，用户消息中包含带图像引用的多模态内容：

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是一个 helpful 的视觉助手。"},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "描述你看到的内容。"}
    ]},
    {"role": "assistant", "content": "图中是一个纯红色的正方形。"}
  ],
  "metadata": {"source": "teacher_model", "model": "Qwen2.5-VL-3B-Instruct", "request_id": "mm_gen_0", "backend": "pai_eas", "images": ["examples/mm_sample_image.png"]}
}
```

## 评估

使用 `job_type: mm_instruct_eval` 通过 LLM 裁判对生成结果打分。

```bash
python -m easydistill.cli --config configs/eval/mm_instruct_eval_pai_token.yaml
```

支持指标：`informativeness`、`helpfulness`、`generalization`、`correctness`。

## `advanced_mm_distill` 流水线

推荐的 MMKD 流水线在一个命令中完成生成、评估、质量过滤和 SFT 数据集构建。

```bash
python -m easydistill.cli --config configs/pipeline/advanced_mm_distill_pai_token.yaml
```

流水线阶段：

1. `mm_instruct_distill`：为 `(图像, 指令)` 样本生成教师回复并构建多模态 SFT 数据。
2. `mm_instruct_eval`：LLM 裁判打分。
3. `quality_filter`：按最低分数或 top-k/top-ratio 筛选。
4. `build_sft`：转换为包含多模态用户消息的 ShareGPT 格式 SFT 数据。

最终 SFT 消息中包含图像内容项，因此数据集可直接用于支持 OpenAI 视觉格式的 MLLM 训练。

## 配置参考

| 字段 | 说明 |
|------|------|
| `backend.type` | `pai_token` 或 `pai_eas` |
| `backend.model_id` | 视觉语言模型 ID |
| `mm.system_prompt` | 可选系统提示 |
| `mm.temperature` | 采样温度 |
| `mm.max_tokens` | 最大回复 token 数 |
| `mm.max_workers` | 并发 API 调用数 |
| `dataset.input_path` | 输入 JSONL 路径 |
| `dataset.output_path` | 输出 JSONL 路径 |
| `dataset.images_key` | 图像列表字段名（默认 `images`） |

使用 EAS 后端时，将 `backend.api_key` 替换为 `endpoint_url` 和 `token`。
