# 基础指令蒸馏

本文档介绍 EasyDistill 2 中基础指令蒸馏模块，涵盖指令数据生成、增强、SFT 数据集构建与评估。思维链（CoT）蒸馏请参见 [cot_distillation_zh.md](cot_distillation_zh.md)。

## 概览

本模块提供以下 job_type：

| job_type | 用途 |
|---|---|
| `instruct_distill` | 对种子指令调用教师模型生成回复，并构造 SFT 数据集。 |
| `instruction_expansion` | 从种子指令中采样 in-context 示例，生成新指令。 |
| `instruction_refinement` | 改写或优化已有指令。 |
| `instruction_response_extraction` | 从原始文本中抽取 `<instruction>/<response>` 对。 |
| `instruction_balance` | 按任务类型/领域分类指令并按目标分布重采样。详见 [instruction_balancing_zh.md](instruction_balancing_zh.md)。 |
| `instruct_eval` | 使用 LLM 作为裁判，评估 `(指令, 回复)` 对。 |
| `cot_distill`、`cot_long2short`、`cot_short2long`、`cot_eval` | 思维链蒸馏。详见 [cot_distillation_zh.md](cot_distillation_zh.md)。 |

端到端流水线（如 `augmented_instruct_distill`、`advanced_instruct_distill`）详见 [pipelines_zh.md](pipelines_zh.md)。


所有 job_type 都支持 PAI-Token 与 PAI-EAS 后端。

各任务使用的完整 JSONL 格式参考见 [data_formats_zh.md](data_formats_zh.md)。

## 通用数据格式

### 种子指令

每行一个 JSON 对象，包含 `instruction` 字段：

```jsonl
{"instruction": "法国的首都是哪里？"}
{"instruction": "用一句话解释量子计算。"}
```

### 用于抽取的原始文本

每行一个 JSON 对象，包含 `text` 字段：

```jsonl
{"text": "用户：2+2 等于多少？\n助手：2+2 等于 4。"}
{"text": "问：什么是机器学习？\n答：机器学习是 ..."}
```

### SFT 数据集输出

EasyDistill 2 输出 OpenAI/ShareGPT 风格的 messages 格式：

```jsonl
{"messages": [{"role": "system", "content": "你是一个 helpful 的助手。"}, {"role": "user", "content": "法国的首都是哪里？"}, {"role": "assistant", "content": "巴黎"}]}
```

## 命令行接口

所有任务都使用同一个命令：

```bash
easydistill --config <path_to_config.yaml>
```

配置中的 `job_type` 字段决定运行哪个流程。

## Prompt 自定义

所有合成算子和评估器都会从 `easydistill/prompts.py` 加载默认 prompt，但每个 prompt 都可以通过配置覆盖。

### 合成算子

对于 `instruction_expansion`、`instruction_refinement` 和 `instruction_response_extraction`，可以内联 prompt 模板，也可以引用外部文本文件：

```yaml
synthesis:
  prompt_template_file: configs/prompts/expansion_prompt.txt
```

或内联：

```yaml
synthesis:
  prompt_template: |
    You are given some example instructions below. Your task is to write a NEW instruction.

    {examples}

    Now write a new instruction. Wrap your output in <answer>...</answer>.
```

占位符：

- `instruction_expansion`：`{examples}`
- `instruction_refinement`：`{instruction}`
- `instruction_response_extraction`：`{text}`

### 评估器

对于 `instruct_eval`，可以内联覆盖单个指标的 prompt，也可以从 YAML/JSON 文件加载：

```yaml
eval:
  prompts_file: configs/prompts/default_eval_prompts.yaml
```

或内联：

```yaml
eval:
  prompts:
    informativeness: |
      Rate the informativeness of the response on a scale of 0-9.
      Instruction: {instruction}
      Response: {output}
      Place your score in <score></score>.
```

Prompt 文件和内联配置都会与内置默认值合并，因此只需指定你想修改的 prompt。

## 后端配置

后端配置在所有 job_type 中结构相同。

### PAI-Token

```yaml
backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct
```

### EAS

```yaml
backend:
  type: pai_eas
  endpoint_url: ${EAS_ENDPOINT_URL}
  token: ${EAS_TOKEN}
```

## job_type: `instruct_distill`

运行黑盒蒸馏：种子指令 -> 教师回复 -> SFT 数据集。

```yaml
job_type: instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "你是一个 helpful 的助手。请给出简洁准确的回答。"
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4

sft:
  skip_empty: true
  min_length: 10
  max_length: 8192

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/instruct_distill_sft.jsonl
```

### SFT 过滤配置

- `skip_empty`：跳过空的助手回复。
- `min_length`：回复的最小字符长度。
- `max_length`：回复的最大字符长度（用于过滤过于冗长的教师输出）。

## job_type: `instruction_expansion`

给定种子指令，采样 in-context 示例并生成新指令。

```yaml
job_type: instruction_expansion

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  num_in_context_samples: 2
  num_output_samples: 3
  temperature: 0.8
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/expanded_instructions.jsonl
  output_format: instruction
```

## job_type: `instruction_refinement`

改写或优化已有指令。

```yaml
job_type: instruction_refinement

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/refined_instructions.jsonl
  output_format: instruction
```

## job_type: `instruction_response_extraction`

从原始文本中抽取 `<instruction>/<response>` 对。该算子先尝试正则抽取；失败后调用 LLM 进行抽取。

```yaml
job_type: instruction_response_extraction

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

synthesis:
  temperature: 0.7
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/raw_texts.jsonl
  text_key: text
  output_path: outputs/extracted_pairs.jsonl
  output_format: instruction_response
```

## job_type: `instruct_eval`

使用 LLM 作为裁判，评估 `(指令, 回复)` 对。裁判模型对四个维度打分：

| 指标 | 范围 | 说明 |
|---|---|---|
| `informativeness` | 0-9 | 回复是否充分、准确地覆盖指令内容。 |
| `helpfulness` | 0-9 | 回复对用户的帮助程度。 |
| `generalization` | 0-9 | 回复中的推理是否可迁移到相似任务。 |
| `correctness` | true/false | 回复是否在事实与逻辑上正确。 |

```yaml
job_type: instruct_eval

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

eval:
  metrics:
    - informativeness
    - helpfulness
    - generalization
    - correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/eval_samples.jsonl
  output_path: outputs/eval_results.jsonl
```

### 支持的输入格式

普通 `(instruction, output)` 格式：

```jsonl
{"instruction": "法国的首都是哪里？", "output": "巴黎"}
```

SFT messages 格式（会自动转换）：

```jsonl
{"messages": [{"role": "user", "content": "法国的首都是哪里？"}, {"role": "assistant", "content": "巴黎"}]}
```

输出 JSONL 包含每个样本的打分，同时 CLI 日志会打印各项的平均分。

输出示例：

```jsonl
{"id": "0", "instruction": "法国的首都是哪里？", "output": "巴黎", "informativeness": 8, "helpfulness": 9, "generalization": 7, "correctness": true}
```

## 使用建议

- 在 `sft` 中使用 `max_length` 过滤过长的教师回复。
- 调用真实 API 时，将 `max_workers` 设置得保守一些，避免触发限流。
- 评估器期望裁判模型将分数放在 `<score>...</score>` 标签内返回。
- 如需完整的端到端流程，可使用 `advanced_instruct_distill`，它将增强、评估与过滤整合为一次运行。
