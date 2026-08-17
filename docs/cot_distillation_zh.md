# 思维链（CoT）蒸馏

本文档介绍 EasyDistill 2 中的思维链（Chain-of-Thought，CoT）蒸馏模块，涵盖 CoT 生成、推理改写（长转短 / 短转长）、LLM 裁判评估以及 `advanced_cot_distill` 端到端流水线。

## 概览

本模块提供五种 job_type：

| job_type | 用途 |
|---|---|
| `cot_distill` | 为每个输入问题生成 `<|begin_of_thought|>` / `<|begin_of_solution|>` 思维链与最终解答。 |
| `cot_long2short` | 简化已有的 CoT 推理过程。 |
| `cot_short2long` | 为已有的 CoT 推理过程补充更多细节。 |
| `cot_eval` | 使用 LLM 作为裁判，评估 `(问题, CoT 答案)` 对。 |
| `advanced_cot_distill` | 端到端流水线：CoT 生成 -> RV/CD 评分 -> RV/CD 混合 -> SFT 数据集。 |

所有 job_type 都支持 PAI-Token 与 PAI-EAS 后端。

各任务使用的完整 JSONL 格式参考见 [data_formats_zh.md](data_formats_zh.md)。

## 通用数据格式

### 种子问题

每行一个 JSON 对象，包含 `problem` 字段：

```jsonl
{"problem": "前 10 个正整数的和是多少？"}
{"problem": "一列火车 30 分钟行驶 60 公里，平均速度是多少公里每小时？"}
```

对于 `cot_distill`，数据集配置中的 `problem_key` 指定使用哪个字段（默认 `problem`，可回退到 `instruction`）。

### 用于改写的问题/答案对

`cot_long2short` 和 `cot_short2long` 读取 `(问题, 答案)` 对。默认使用 `instruction` 作为问题、`response` 作为答案，并支持常见的字段名回退：

```jsonl
{"instruction": "2+2 等于多少？", "response": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
```

你可以通过数据集配置中的 `problem_key` 和 `answer_key` 覆盖这些字段名。

### CoT 生成输出（基础功能）

`cot_distill` 是基础功能，直接生成可输入 LLaMA-Factory 或 ms-swift 的 ShareGPT 格式 SFT messages。

```jsonl
{"messages": [{"role": "user", "content": "2+2 等于多少？"}, {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}], "metadata": {"thought": "...", "solution": "4"}}
```

### `advanced_cot_distill` 中的 SFT 数据集输出

`advanced_cot_distill` 最后的 `build_sft` 阶段将 `instruction`/`response` 对转换为 OpenAI/ShareGPT 风格的 messages 格式：

```jsonl
{"messages": [{"role": "user", "content": "2+2 等于多少？"}, {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}]}
```

## 命令行接口

所有任务都使用同一个命令：

```bash
easydistill --config <path_to_config.yaml>
```

配置中的 `job_type` 字段决定运行哪个流程。

## Prompt 自定义

### CoT 算子

对于 `cot_distill`、`cot_long2short` 和 `cot_short2long`，可以内联 prompt 模板，也可以引用外部文本文件：

```yaml
cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
```

或内联：

```yaml
cot:
  prompt_template: |
    请逐步解决以下问题。

    问题：{problem}
```

占位符：

- `cot_distill`：`{problem}`
- `cot_long2short`：`{problem}`、`{answer}`
- `cot_short2long`：`{problem}`、`{answer}`

### 评估器

对于 `cot_eval`，可以内联覆盖单个指标的 prompt，也可以从 YAML/JSON 文件加载：

```yaml
eval:
  prompts_file: configs/prompts/default_cot_eval_prompts.yaml
```

或内联：

```yaml
eval:
  prompts:
    reasoning_verbosity: |
      请对以下 CoT 的推理详尽程度在 0-9 之间打分。
      问题：{instruction}
      含 CoT 的答案：{output}
      将分数放在 <score></score> 中。
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

## job_type: `cot_distill`

为每个问题生成思维链推理过程与最终解答。

```yaml
job_type: cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_generation_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/seed_cot_problems.jsonl
  problem_key: problem
  output_path: outputs/cot_distill.jsonl
```

## job_type: `cot_long2short`

简化已有的 CoT 推理过程。输入必须同时包含问题与完整的 CoT 答案。

```yaml
job_type: cot_long2short

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_long2short_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  problem_key: instruction
  answer_key: response
  output_path: outputs/cot_long2short.jsonl
  output_format: cot
```

## job_type: `cot_short2long`

为已有的 CoT 推理过程补充更多中间步骤与细节。

```yaml
job_type: cot_short2long

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

cot:
  prompt_template_file: configs/prompts/cot_short2long_prompt.txt
  temperature: 0.7
  max_tokens: 2048
  max_workers: 3
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  problem_key: instruction
  answer_key: response
  output_path: outputs/cot_short2long.jsonl
  output_format: cot
```

## job_type: `cot_eval`

使用 LLM 作为裁判，评估 `(问题, CoT 答案)` 对。裁判模型对三个维度打分：

| 指标 | 范围 | 说明 |
|---|---|---|
| `reasoning_verbosity` | 0-9 | CoT 长度与步骤复杂度是否与问题难度相匹配。 |
| `cognitive_difficulty` | 0-9 | 模型要复现该推理链所需的推理能力水平。 |
| `logical_correctness` | true/false | 推理过程与最终解答是否在逻辑上成立。 |

```yaml
job_type: cot_eval

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

eval:
  metrics:
    - reasoning_verbosity
    - cognitive_difficulty
    - logical_correctness
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4
  show_progress: true

dataset:
  input_path: examples/cot_eval_samples.jsonl
  output_path: outputs/cot_eval_results.jsonl
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
{"id": "0", "instruction": "前 10 个正整数的和是多少？", "output": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>55<|end_of_solution|>", "reasoning_verbosity": 6, "cognitive_difficulty": 5, "logical_correctness": true}
```

## job_type: `advanced_cot_distill`

通过一条命令运行完整的 CoT 蒸馏流程：

1. `cot_distill` — 为每个问题生成思维链与解答。
2. `cot_rvcd_score` — 为生成的 CoT 按推理冗长度、认知难度和逻辑正确性评分。
3. `cot_mix_by_rv_cd` — 按 CD 分箱混合行，构建课程式 SFT 子集。
4. `build_sft` — 生成最终的 SFT 数据集。

最后一个阶段必须是 `build_sft`。

```yaml
job_type: advanced_cot_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "你是一个 helpful 的助手。请逐步思考并给出清晰的推理。"
  temperature: 0.7
  max_tokens: 2048

sft:
  skip_empty: true
  min_length: 10
  max_length: 4096

eval:
  prompts_file: configs/prompts/default_cot_eval_prompts.yaml
  metrics:
    - reasoning_verbosity
    - cognitive_difficulty
    - logical_correctness
  temperature: 0.0
  max_tokens: 512
  max_workers: 4

pipeline:
  - stage: cot_distill
    config:
      prompt_template_file: configs/prompts/cot_generation_prompt.txt
      temperature: 0.7
      max_tokens: 2048
      show_progress: true
      max_workers: 3
    output_path: outputs/cot_bp_stage1_generated.jsonl

  - stage: cot_rvcd_score
    config:
      show_progress: true
      max_workers: 4
    output_path: outputs/cot_bp_stage2_scored.jsonl

  - stage: cot_mix_by_rv_cd
    config:
      mode: sft
      cd_bins: [0, 3, 6, 10]
      rv_target: matched
      samples_per_bin: 100
      min_correctness: 1
    output_path: outputs/cot_bp_stage3_mixed.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_cot_problems.jsonl
  output_path: outputs/cot_bp_sft.jsonl
```

如需增加改写阶段，可在 `cot_distill` 与 `cot_rvcd_score` 之间插入 `cot_long2short` 或 `cot_short2long`。

## 使用建议

- 在 `sft` 中使用 `max_length` 过滤过长的教师回复。
- 调用真实 API 时，将 `max_workers` 设置得保守一些，避免触发限流。
- 评估器期望裁判模型将分数放在 `<score>...</score>` 标签内返回。
- 如需完整的端到端流程，可使用 `advanced_cot_distill`，它将生成、RV/CD 评分与课程混合整合为一次运行。
- 生成 SFT 数据集后，参考 [training_guide_zh.md](training_guide_zh.md) 中的 LLaMA-Factory 与 ms-swift 微调示例，包括 LoRA。
