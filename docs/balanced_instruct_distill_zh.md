# 平衡指令蒸馏

`balanced_instruct_distill` 流水线将合成新指令、按任务/领域分类并重采样到目标分布、生成教师回复、构建 SFT 数据集整合为一次运行。适用于希望最终数据集按平衡课程分布而非原始生成分布来组织的场景。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 适用场景

当你希望完成以下流程时使用该流水线：

1. 将小规模种子指令集扩充为新指令。
2. 对合成后的指令按任务或领域分类。
3. 按目标分布重采样，避免某一类别占比过高。
4. 为平衡后的指令生成教师回复。
5. 构建 OpenAI/ShareGPT 消息格式的 SFT 数据集。

## 流水线阶段

| 阶段 | 是否必需 | 用途 |
|---|---|---|
| `instruction_expansion` | 可选 | 从种子指令生成新指令。 |
| `instruction_response_extraction` | 可选 | 从原始文本抽取 `(指令, 回复)` 对。 |
| `instruction_refinement` | 可选 | 在平衡前改写/优化指令。 |
| `instruction_balance` | 必需 | 按任务/领域分类并重采样到目标分布。 |
| `generate` | 必需 | 调用教师模型生成回复。 |
| `instruct_eval` | 可选 | 运行 LLM 裁判评估。 |
| `quality_filter` | 可选 | 按分数阈值丢弃样本。 |
| `build_sft` | 必需（最后） | 将剩余样本转换为 SFT 消息格式。 |

流水线必须以 `build_sft` 结尾。

## 配置 Schema

```yaml
job_type: balanced_instruct_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant. Provide concise and accurate answers."
  temperature: 0.7
  max_tokens: 2048

eval:
  prompts_file: configs/prompts/default_eval_prompts.yaml
  metrics:
    - informativeness
    - helpfulness
    - generalization
    - correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4

pipeline:
  - stage: instruction_expansion
    config:
      prompt_template_file: configs/prompts/expansion_prompt.txt
      num_in_context_samples: 2
      num_output_samples: 3
      temperature: 0.8
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/balanced_stage1_expanded.jsonl

  - stage: instruction_balance
    config:
      instruction_key: instruction
      category_key: category
      categories: ["General"]
      target_distribution:
        General: 1.0
      category_prompt: "Classify the instruction below as General. Wrap the category in <answer>...</answer>.\nInstruction: {instruction}"
      max_workers: 4
      show_progress: true
      seed: 42
      temperature: 0.0
      max_tokens: 2048
    output_path: outputs/balanced_stage2_balanced.jsonl

  - stage: generate
    config:
      show_progress: true
      max_workers: 3
    output_path: outputs/balanced_stage3_generated.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/balanced_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### 顶层字段

- `job_type`：必须是 `balanced_instruct_distill`。
- `backend`：任意支持的后端（`pai_token`、`pai_eas`）。
- `generation`：`generate` 与 `build_sft` 使用的默认生成参数。
- `eval`：默认评估参数，`metrics` 指定要计算的裁判指标。
- `pipeline`：有序阶段列表。
- `dataset`：`input_path`、`output_path` 与 SFT 过滤参数（`skip_empty`、`min_length`、`max_length`）。

### `instruction_balance` 配置

完整 schema 见 [instruction_balancing_zh.md](instruction_balancing_zh.md)。上面的示例只使用单个 `General` 类别；请根据实际任务/领域替换 `categories` 与 `target_distribution`。

## 阶段数据格式

流水线在各阶段之间写入中间 JSONL 文件，每行保留下一阶段所需的字段。

| 阶段 | 输出行字段 | 说明 |
|---|---|---|
| `instruction_expansion` | `instruction` | 每行一条扩充后的指令（可选首阶段）。 |
| `instruction_refinement` | `instruction` | 上一阶段指令的精炼版本（可选）。 |
| `instruction_balance` | `instruction`、`category` | 原始指令与分配的类别。 |
| `generate` | `instruction`、`output` | 教师模型生成的回复。 |
| `instruct_eval` | `instruction`、`output`、`<指标>` | 原始字段加各指标分数。 |
| `quality_filter` | 同 `instruct_eval` | 仅保留通过阈值的行。 |
| `build_sft` | `messages`、`metadata` | 最终的 OpenAI/ShareGPT SFT 消息。 |

## 运行流水线

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/balanced_instruct_distill_pai_token.yaml
```

PAI-EAS 等价配置见 [`configs/pipeline/balanced_instruct_distill_pai_eas.yaml`](../configs/pipeline/balanced_instruct_distill_pai_eas.yaml)。
