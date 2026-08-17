# 使用蒸馏数据进行训练

EasyDistill 2 输出 SFT 数据（标准 OpenAI/ShareGPT `messages` 格式）和 DPO 偏好数据（Alpaca、ShareGPT 或 OpenAI message 格式）。本文档说明如何将这些数据用于 LLaMA-Factory、ms-swift 等主流训练框架，包括 LoRA 与全参数微调。

## 输出格式

### SFT 输出

运行任何以 `build_sft` 结尾的蒸馏任务后，输出 JSONL 文件的每一行如下：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个 helpful 的助手。"},
    {"role": "user", "content": "2+2 等于多少？"},
    {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
  ],
  "metadata": {"source": "teacher_model", "model": "qwen3-235b-a22b"}
}
```

`messages` 字段可直接被 LLaMA-Factory、ms-swift 等工具消费。

### DPO 输出

运行 `dpo_data_build` 任务后，输出格式取决于 `preference.format` 字段。三种支持的格式为：

- `llama_factory_alpaca`：单轮 Alpaca DPO 格式，包含 `instruction`、`input`、`chosen`、`rejected` 字段。
- `llama_factory_sharegpt`：ShareGPT DPO 格式，包含 `prompt`、`chosen`、`rejected` 的 `messages` 列表。
- `openai_messages`：扁平的 `prompt` / `chosen` / `rejected` 消息列表格式。

各格式的具体字段与 `dataset_info.json` 配置方式见 [dpo_distillation_zh.md](dpo_distillation_zh.md)。

EasyDistill 2 使用的所有输入与中间 JSONL 格式完整目录见 [data_formats_zh.md](data_formats_zh.md)。

## SFT 训练

### LLaMA-Factory

#### 注册数据集

在 `data/dataset_info.json` 中添加数据集：

```json
{
  "easydistill_sft": {
    "file_name": "cot_bp_sft_pai_token.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

#### 全参数微调示例

```bash
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset easydistill_sft \
  --template qwen \
  --finetuning_type full \
  --output_dir outputs/qwen_cot_full \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 100 \
  --plot_loss True
```

#### LoRA 微调示例

```bash
llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset easydistill_sft \
  --template qwen \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_dir outputs/qwen_cot_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.0e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 100 \
  --plot_loss True
```

请根据基座模型调整 `lora_rank`、`lora_alpha` 和 `lora_target`。对于 Qwen 模型，`lora_target all` 通常足够；对于 Llama 模型，可选择 `lora_target q_proj,v_proj`。

#### 合并 LoRA 权重（可选）

LoRA 训练结束后，可将 adapter 合并回基座模型：

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/qwen_cot_lora \
  --template qwen \
  --finetuning_type lora \
  --export_dir outputs/qwen_cot_merged
```

### ms-swift

#### 数据集格式

ms-swift 同样支持 `messages` 格式。将蒸馏得到的 JSONL 文件复制或软链接到 `custom_dataset` 目录，并在 swift 数据集配置中注册；也可以直接在命令行传入文件路径。

#### 全参数微调示例

```bash
swift sft \
  --model_type qwen2_5-7b-instruct \
  --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset easydistill_sft.jsonl \
  --sft_type full \
  --output_dir outputs/swift_qwen_cot_full \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1.0e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --save_steps 100 \
  --logging_steps 10
```

#### LoRA 微调示例

```bash
swift sft \
  --model_type qwen2_5-7b-instruct \
  --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset easydistill_sft.jsonl \
  --sft_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules ALL \
  --output_dir outputs/swift_qwen_cot_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.0e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --save_steps 100 \
  --logging_steps 10
```

在 ms-swift 中，`lora_target_modules ALL` 表示对所有线性层加 LoRA。也可以显式指定模块，例如 `q_proj,k_proj,v_proj,o_proj`。

#### 合并 LoRA 权重（可选）

```bash
swift export \
  --ckpt_dir outputs/swift_qwen_cot_lora \
  --merge_lora True \
  --output_dir outputs/swift_qwen_cot_merged
```

## DPO 训练

`dpo_data_build` 生成的 DPO 偏好数据可直接用于 LLaMA-Factory 或 ms-swift。以下示例假设使用默认的 `llama_factory_alpaca` 输出格式。

### LLaMA-Factory

#### 注册数据集

在 `data/dataset_info.json` 中添加数据集：

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

若使用 ShareGPT DPO 格式，设置 `"formatting": "sharegpt"` 与 `"ranking": "true"`，无需显式声明列。

#### 全参数微调示例

```bash
llamafactory-cli train \
  --stage dpo \
  --do_train True \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset my_dpo \
  --template qwen \
  --finetuning_type full \
  --output_dir outputs/qwen_dpo_full \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-7 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 100 \
  --plot_loss True
```

#### LoRA 微调示例

```bash
llamafactory-cli train \
  --stage dpo \
  --do_train True \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset my_dpo \
  --template qwen \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_dir outputs/qwen_dpo_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-6 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 100 \
  --plot_loss True
```

DPO 学习率通常比 SFT 低一个数量级。全参数微调建议从 `5e-7` 开始，LoRA 建议从 `5e-6` 开始。

### ms-swift

#### 数据集格式

ms-swift 接受 Alpaca 或 ShareGPT 格式的 DPO 数据。直接传入 JSON/JSONL 文件即可：

```bash
swift dpo \
  --model_type qwen2_5-3b-instruct \
  --model_id_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset dpo_instruct_dataset_pai_token.json \
  --sft_type full \
  --output_dir outputs/swift_qwen_dpo_full \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-7 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --save_steps 100 \
  --logging_steps 10
```

#### LoRA 微调示例

```bash
swift dpo \
  --model_type qwen2_5-3b-instruct \
  --model_id_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset dpo_instruct_dataset_pai_token.json \
  --sft_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules ALL \
  --output_dir outputs/swift_qwen_dpo_lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-6 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --save_steps 100 \
  --logging_steps 10
```

## 训练建议

### SFT / CoT 数据

- 助手的回复中包含 `<|begin_of_thought|>`、`<|begin_of_solution|>` 等特殊标签。请确保基座模型的 tokenizer 能够表示这些 token，或在训练前将它们添加为特殊 token。
- 如果基座模型不会主动输出 CoT 标签，可以考虑先用较小学习率做预热，或先仅对 reasoning 内容训练，再引入完整标签。
- 对于小模型（7B），LoRA rank >= 8 通常效果较好；对于更大模型（14B+），可以适当提高 rank，或在显存允许时进行全参数微调。
- 蒸馏长 CoT 链时，建议在 SFT 构建阶段设置 `max_length`，避免训练时出现超出模型上下文长度的超长序列。

### DPO 数据

- 确保 chosen 与 rejected 在质量上有明显差距；可在偏好流水线中设置 `min_margin > 0` 以避免同分。
- 对于 CoT 偏好数据，请先验证答案提取器能正确识别最终答案，再构建偏好对。
- DPO 对学习率敏感。若训练不稳定，可降低学习率或增大等效 batch size。

## 下一步

- 参考 [instruction_distillation_zh.md](instruction_distillation_zh.md) 和 [cot_distillation_zh.md](cot_distillation_zh.md) 了解如何生成 SFT 数据。
- 参考 [dpo_distillation_zh.md](dpo_distillation_zh.md) 了解如何生成 DPO 偏好数据。
- 查阅 LLaMA-Factory 与 ms-swift 官方文档，获取更多框架特定选项与分布式训练配置。
