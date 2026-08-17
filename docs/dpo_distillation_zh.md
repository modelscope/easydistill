# DPO 偏好数据

EasyDistill2 可以构建 DPO（直接偏好优化）训练所需的偏好数据集。本模块只负责数据构建，训练由 LLaMA-Factory 或 ms-swift 等框架完成。

各阶段使用的完整 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 支持的 job_type

- `dpo_data_build`：从种子提示构建偏好对。

提供两种变体：

- `dpo_instruct_*`：指令遵循偏好数据，使用 LLM 裁判评分。
- `dpo_cot_*`：思维链偏好数据，使用答案正确性与推理简洁度评分。

## 流水线阶段

偏好数据配置包含四个阶段：

1. `generate_candidates`：为每个提示生成 `n` 个候选回复。
2. `score_candidates`：使用评分器为每个候选打分。
3. `build_preference_pairs`：为每个提示选择分数最高的 `chosen` 和分数最低的 `rejected`。
4. `build_preference_dataset`：将偏好对导出为训练框架格式。

## 配置字段

### `backend`

与其他 EasyDistill2 配置相同，支持 `pai_token` 和 `pai_eas`。

### `dataset`

- `input_path`：包含种子提示的 JSONL 文件。
- `output_path`：最终数据集输出路径。

`dpo_instruct` 的输入通常包含 `instruction` 字段。
`dpo_cot` 的输入应包含 `problem` 和 `answer` 字段。

### `generation`

全局生成参数默认值：
- `system_prompt`、`temperature`、`max_tokens`、`max_workers`、`show_progress`。

### `preference`

偏好流水线的默认设置。以下字段可在各阶段的 `config` 块中单独覆盖。

- `scorer`：指令数据用 `llm_judge`，CoT 数据用 `cot`。
- `n`：每个提示生成的候选数量。
- `metrics`：LLM 裁判使用的指标（默认 `["helpfulness", "correctness"]`）。
- `min_margin`：chosen 与 rejected 之间的最小分数差（默认 `0.0`）。
- `max_pairs_per_prompt`：每个提示输出的偏好对数量（默认 `1`）。
- `require_chosen_correct`：要求 chosen 与参考答案一致。
- `format`：最终数据集格式。
- `instruction_key`：输入提示字段（默认 `"instruction"`）。CoT 数据可设为 `"problem"`。
- `answer_key`：参考答案字段（默认 `"answer"`）。
- `system_key`：可选的每行系统提示字段（默认 `"system"`）。

`cot` 评分器还可设置：
- `alpha`：长度惩罚系数（默认 `0.001`）。
- `normalize_answer`：比较前是否规范化答案（默认 `true`）。
- `answer_extractor_pattern`：自定义提取最终答案的正则。

## 阶段配置参考

每个阶段都可以在自身的 `config` 块中覆盖顶层 `preference` 的默认值。

### `generate_candidates`

- `n`：每个提示的候选数量。
- `instruction_key`、`system_key`：用作提示 / 系统提示的字段。
- `temperature`、`max_tokens`、`max_workers`、`show_progress`。

### `score_candidates`

- `scorer`：`llm_judge` 或 `cot`。
- `metrics`：用于 `llm_judge`。
- `instruction_key`、`answer_key`：使用的字段。
- `alpha`、`normalize_answer`、`answer_extractor_pattern`：用于 `cot`。
- `temperature`、`max_tokens`、`max_workers`、`show_progress`：裁判模型参数。

### `build_preference_pairs`

- `min_margin`：最小分数差（设为正值可避免同分情况）。
- `max_pairs_per_prompt`。
- `require_chosen_correct`。
- `instruction_key`、`answer_key`、`system_key`。

### `build_preference_dataset`

- `format`：`llama_factory_alpaca`、`llama_factory_sharegpt`、`openai_messages` 之一。
- `instruction_key`、`system_key`。
- `skip_empty`、`min_length`、`max_length`。

## 各阶段数据格式

### 输入

`dpo_instruct_*` 种子提示：

```jsonl
{"instruction": "用一段话解释知识蒸馏。"}
```

`dpo_cot_*` 种子问题与参考答案：

```jsonl
{"problem": "前 10 个正整数的和是多少？", "answer": "55"}
```

### 阶段输出

`generate_candidates` 为每个提示输出一行，包含 `n` 个候选回复：

```jsonl
{
  "id": "1",
  "instruction": "用一段话解释知识蒸馏。",
  "candidates": ["...", "..."],
  "candidate_results": [
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}},
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}}
  ]
}
```

`score_candidates` 会追加 `candidate_scores`（CoT 评分器还会追加 `candidate_correctness`）：

```jsonl
{
  "id": "1",
  "instruction": "用一段话解释知识蒸馏。",
  "candidates": ["...", "..."],
  "candidate_scores": [4.0, 4.0]
}
```

`build_preference_pairs` 为每个提示输出一个 `chosen` / `rejected` 对：

```jsonl
{
  "id": "1",
  "instruction": "用一段话解释知识蒸馏。",
  "system": null,
  "chosen": "...",
  "rejected": "...",
  "chosen_score": 4.0,
  "rejected_score": 4.0,
  "answer": null
}
```

CoT DPO 中 `instruction` 被替换为 `problem`，`answer` 为参考答案。

### 最终数据集格式

`build_preference_dataset` 根据 `preference.format` 导出以下格式之一：

`llama_factory_alpaca`：

```jsonl
{"instruction": "...", "input": "", "chosen": "...", "rejected": "..."}
```

`llama_factory_sharegpt`：

```jsonl
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ],
  "chosen": {"from": "gpt", "value": "..."},
  "rejected": {"from": "gpt", "value": "..."}
}
```

`openai_messages`：

```jsonl
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

## 示例：PAI-Token DPO instruct

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/preference/dpo_instruct_pai_token.yaml
```

## 示例：PAI-EAS DPO CoT

```bash
export EAS_ENDPOINT_URL=https://your-service.cn-shanghai.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/preference/dpo_cot_pai_eas.yaml
```

EAS 端点 URL 也可以以 `/v1/chat/completions` 结尾，系统会自动规范化。

## CoT 答案提取说明

`cot` 评分器按以下顺序提取最终答案：`\boxed{...}`、`#### ...`、`the answer is ...`、
最后一个独立数字，最后是最后一行非空内容。如果参考答案不是纯答案，请设置
`answer_extractor_pattern` 来捕获目标值。

## LLaMA-Factory dataset_info.json

Alpaca DPO 输出：

```json
{
  "my_dpo": {
    "file_name": "dpo_instruct_dataset_pai_token.json",
    "formatting": "alpaca",
    "ranking": "true",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

ShareGPT DPO 输出：

```json
{
  "my_dpo": {
    "file_name": "dpo_instruct_dataset_pai_token.json",
    "formatting": "sharegpt",
    "ranking": "true"
  }
}
```

## 训练

LLaMA-Factory 与 ms-swift 的全参数微调和 LoRA 训练命令见 [training_guide_zh.md](training_guide_zh.md)。
