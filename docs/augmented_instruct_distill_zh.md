# 增强指令蒸馏

`augmented_instruct_distill` 流水线将指令合成阶段串联起来，最后以 `instruct_distill` 收尾，一次性产出 SFT 数据集。它适用于在蒸馏教师回复之前先放大小规模种子指令集。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 流水线阶段

| 阶段 | 用途 |
|---|---|
| `instruction_expansion` | 使用 in-context 示例从种子指令生成新指令。 |
| `instruction_refinement` | 改写或优化扩充后的指令。 |
| `instruct_distill` | 生成教师回复并构建最终 SFT 数据集。 |

如果输入数据和工作流需要，也可以在合成阶段之间加入 `instruction_response_extraction` 或 `instruction_balance`。

流水线必须以 `instruct_distill` 结尾。

## 配置 Schema

### 顶层字段

```yaml
job_type: augmented_instruct_distill

backend:
  type: pai_token          # 或 pai_eas
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "你是一个 helpful 的助手。请给出简洁准确的回答。"
  temperature: 0.7
  max_tokens: 2048

pipeline:
  - stage: instruction_expansion
    config: { ... }
    output_path: outputs/augmented_stage1_expanded.jsonl

  - stage: instruction_refinement
    config: { ... }
    output_path: outputs/augmented_stage2_refined.jsonl

  - stage: instruct_distill
    config: { ... }

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/augmented_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### `generation`

默认生成参数，作用于所有生成阶段；可在阶段自身的 `config` 中覆盖。

### `dataset`

- `input_path`：包含种子指令（`instruction` 字段）的 JSONL 文件。
- `output_path`：最终 SFT 数据集路径。
- `skip_empty`、`min_length`、`max_length`：SFT 回复过滤参数。

### 阶段配置

每个阶段的 `config` 支持与对应独立 job_type 相同的字段。详见：

- [instruction_distillation_zh.md](instruction_distillation_zh.md)：`instruction_expansion`、`instruction_refinement`、`instruct_distill`。
- [instruction_balancing_zh.md](instruction_balancing_zh.md)：`instruction_balance`。

## 阶段数据格式

流水线在各阶段之间写入中间 JSONL 文件，每行保留下一阶段所需的字段。

| 阶段 | 输出行字段 | 说明 |
|---|---|---|
| `instruction_expansion` | `instruction` | 每行一条扩充后的指令。 |
| `instruction_refinement` | `instruction` | 上一阶段指令的精炼版本。 |
| `instruct_distill` | `messages`、`metadata` | 最终的 OpenAI/ShareGPT SFT 消息。 |

## 示例

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/augmented_instruct_distill_pai_token.yaml
```

PAI-EAS 等价配置见 [`configs/pipeline/augmented_instruct_distill_pai_eas.yaml`](../configs/pipeline/augmented_instruct_distill_pai_eas.yaml)。
