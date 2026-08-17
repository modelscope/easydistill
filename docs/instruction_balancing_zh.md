# 指令均衡 / 任务感知课程规划

本文档介绍 EasyDistill 2 中的 `instruction_balance` 阶段。它按任务类型或领域对指令进行分类，并根据目标分布对数据集进行重采样，从而避免某些类别占比过高，使最终的 SFT 数据集更均衡、更适合课程学习。

`instruction_balance` 不再作为独立 `job_type` 暴露。请在 `advanced_instruct_distill`、`augmented_instruct_distill` 或专用的 `balanced_instruct_distill` 流水线中使用。

这些流水线使用的完整 JSONL 格式参考见 [data_formats_zh.md](data_formats_zh.md)。

## 适用场景

在以下情况下可以使用指令均衡：

- 种子指令或合成后的指令集中偏向少数几个领域（例如大部分是数学或编程）。
- 希望最终数据集遵循已知的目标分布，例如 DistilQwen2 配方。
- 希望准备一种课程，使每个训练批次都包含可预测的任务类型混合。

## 工作原理

`InstructionBalancer` 算子分两步执行：

1. **分类**：将每条指令连同分类提示词一起发送给配置的后端模型。模型返回被 `<answer>...</answer>` 标签包裹的类别。
2. **重采样**：算子统计每个类别的样本数量，并按目标比例进行重采样。
   - 若某类别样本过多，则随机下采样。
   - 若某类别样本不足，则重复已有样本直到达到目标数量。
   - 最终结果会通过可配置的随机种子打乱顺序。

默认类别列表和目标分布来自 DistilQwen2 配方。你可以在配置中覆盖它们。

## 流水线用法

`instruction_balance` 作为 `advanced_instruct_distill`、`augmented_instruct_distill` 或 `balanced_instruct_distill` 流水线的一个阶段使用。通常放在合成阶段之后、教师模型生成之前。若希望平衡是核心环节的端到端流水线，见 [docs/balanced_instruct_distill_zh.md](balanced_instruct_distill_zh.md)。

```yaml
job_type: advanced_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

pipeline:
  - stage: instruction_expansion
    config:
      num_in_context_samples: 2
      num_output_samples: 5
      max_workers: 3
    output_path: outputs/stage1_expanded.jsonl

  - stage: instruction_balance
    config:
      max_workers: 4
      seed: 42
    output_path: outputs/stage2_balanced.jsonl

  - stage: generate
    config:
      max_workers: 4
    output_path: outputs/stage3_generated.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/advanced_instruct_distill_sft_balanced.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

## 配置说明

### 顶层 `balance` 字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `instruction_key` | `str` | `instruction` | 包含指令文本的字段名。 |
| `category_key` | `str` | `category` | 用于存储分配类别的字段名。 |
| `categories` | `List[str]` | DistilQwen2 列表 | 有效类别名称。 |
| `target_distribution` | `Dict[str, float]` | DistilQwen2 比例 | 每个类别的目标比例，建议总和为 `1.0`。 |
| `category_prompt` | `str` | 内置提示词 | 分类提示词模板，必须包含 `{categories}` 和 `{instruction}` 占位符。 |
| `system_prompt` | `str` | `None` | 可选的系统提示词。 |
| `max_workers` | `int` | `1` | 并发分类请求数。 |
| `show_progress` | `bool` | `true` | 是否显示进度条。 |
| `seed` | `int` | `42` | 重采样和打乱的随机种子。 |
| `model_id` | `str` | `None` | 覆盖分类时使用的后端模型 ID。 |
| `temperature` | `float` | `0.0` | 分类采样温度。 |
| `max_tokens` | `int` | `512` | 类别回复的最大 token 数。 |

### 阶段配置

作为流水线阶段使用时，`config` 块可以接受上述任意字段。该阶段没有必填字段，未填写项均使用默认值。

## 自定义类别列表与分布

你可以通过同时提供 `categories` 和 `target_distribution` 来定义自己的课程：

```yaml
balance:
  categories: ["Easy", "Medium", "Hard"]
  target_distribution:
    Easy: 0.3
    Medium: 0.5
    Hard: 0.2
  category_prompt: |
    Classify the following instruction into one of {categories}.
    Wrap the answer in <answer></answer> tags.

    {instruction}
```

分类输出的解析顺序如下：

1. 查找 `<answer>...</answer>`，并检查其值是否在 `categories` 中。
2. 若没有有效标签，则检查是否有类别名称作为子串出现。
3. 否则回退到 `"Others"`。

## 输出格式

输出为 JSONL 文件，每行保留原始字段并新增分配的类别：

```json
{"instruction": "What is 2 + 2?", "category": "Math"}
{"instruction": "Write a Python function to sort a list.", "category": "Code Generation"}
```
