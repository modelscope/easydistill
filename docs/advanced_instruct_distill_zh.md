# 高级指令蒸馏

`advanced_instruct_distill` 流水线将指令蒸馏的各个环节——扩充、精炼、教师生成、LLM 裁判评估与质量过滤——串联为端到端流程，只保留最适合监督微调（SFT）的高质量数据。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 适用场景

当你希望用一条命令从小规模种子指令集得到经过筛选的高质量 SFT 数据集时，使用该流水线。它会自动完成：

1. 将种子指令扩充为更多样化的指令。
2. 精炼指令，提升清晰度与难度。
3. 为每条指令生成教师回复。
4. 使用 LLM 裁判评估每条 `(指令, 回复)` 对。
5. 根据裁判分数过滤低质量样本。
6. 构建最终 OpenAI/ShareGPT 消息格式的 SFT 数据集。

## 流水线阶段

| 阶段 | 是否必需 | 用途 |
|---|---|---|
| `instruction_expansion` | 可选 | 从种子指令生成新指令。 |
| `instruction_refinement` | 可选 | 改写/优化指令。 |
| `instruction_response_extraction` | 可选 | 从原始文本抽取 `(指令, 回复)` 对。 |
| `instruction_balance` | 可选 | 按任务/领域分类并重采样。 |
| `generate` | 必需 | 调用教师模型生成回复。 |
| `instruct_eval` | 可选 | 运行 LLM 裁判评估。 |
| `quality_filter` | 可选 | 按分数阈值丢弃样本。 |
| `build_sft` | 必需（最后） | 将剩余样本转换为 SFT 消息格式。 |

流水线必须以 `build_sft` 结尾。

## 配置 Schema

```yaml
job_type: advanced_instruct_distill

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
    output_path: outputs/advanced_stage1_expanded.jsonl

  - stage: instruction_refinement
    config:
      prompt_template_file: configs/prompts/refinement_prompt.txt
      temperature: 0.7
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/advanced_stage2_refined.jsonl

  - stage: generate
    config:
      show_progress: true
      max_workers: 3
    output_path: outputs/advanced_stage3_generated.jsonl

  - stage: instruct_eval
    config:
      show_progress: true
      max_workers: 4
    output_path: outputs/advanced_stage4_evaluated.jsonl

  - stage: quality_filter
    config:
      min_scores:
        informativeness: 6
        helpfulness: 6
        generalization: 4
        correctness: true
      require_all_metrics: true
      keep_top_ratio: 0.7
    output_path: outputs/advanced_stage5_filtered.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/advanced_instruct_distill_sft.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

### 顶层字段

- `job_type`：必须是 `advanced_instruct_distill`。
- `backend`：任意支持的后端（`pai_token`、`pai_eas`）。
- `generation`：`generate` 与 `build_sft` 使用的默认生成参数。
- `eval`：默认评估参数，`metrics` 指定要计算的裁判指标。
- `pipeline`：有序阶段列表。
- `dataset`：`input_path`、`output_path` 与 SFT 过滤参数（`skip_empty`、`min_length`、`max_length`）。

### `quality_filter` 配置

- `min_scores`：每个指标的最低阈值。
  - 数值指标（`informativeness`、`helpfulness`、`generalization`）：最低分数。
  - 布尔指标（`correctness`）：`true` 或 `false`。
- `require_all_metrics`：设为 `true` 时，缺少任一分数的行都会被丢弃。
- `keep_top_k`：保留分数最高的前 k 行。
- `keep_top_ratio`：保留前 N% 的行（例如 `0.7` 保留 70%）。

若同时设置 `keep_top_k` 与 `keep_top_ratio`，以 `keep_top_k` 为准。

## 阶段数据格式

流水线在各阶段之间写入中间 JSONL 文件，每行保留下一阶段所需的字段。

| 阶段 | 输出行字段 | 说明 |
|---|---|---|
| `instruction_expansion` | `instruction` | 每行一条扩充后的指令。 |
| `instruction_refinement` | `instruction` | 上一阶段指令的精炼版本。 |
| `generate` | `instruction`、`output` | 教师模型生成的回复。 |
| `instruct_eval` | `instruction`、`output`、`<指标>` | 原始字段加各指标分数。 |
| `quality_filter` | 同 `instruct_eval` | 仅保留通过阈值的行。 |
| `build_sft` | `messages`、`metadata` | 最终的 OpenAI/ShareGPT SFT 消息。 |

## 运行流水线

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/advanced_instruct_distill_pai_token.yaml
```

PAI-EAS 等价配置见 [`configs/pipeline/advanced_instruct_distill_pai_eas.yaml`](../configs/pipeline/advanced_instruct_distill_pai_eas.yaml)。
