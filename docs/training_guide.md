# Using Distilled Data for Training

EasyDistill 2 produces SFT data in the standard OpenAI/ShareGPT `messages` format and DPO preference data in Alpaca, ShareGPT, or OpenAI message formats. This document explains how to use the output with popular training frameworks such as LLaMA-Factory and ms-swift, including LoRA and full fine-tuning.

## Output formats

### SFT output

After running any distillation job that ends with `build_sft`, the output JSONL file contains rows like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
  ],
  "metadata": {"source": "teacher_model", "model": "qwen3-235b-a22b"}
}
```

The `messages` field is directly consumable by LLaMA-Factory, ms-swift, and most other toolkits.

### DPO output

After running a `dpo_data_build` job, the output format depends on the `preference.format` field. The three supported formats are:

- `llama_factory_alpaca`: single-turn Alpaca DPO format with `instruction`, `input`, `chosen`, and `rejected` fields.
- `llama_factory_sharegpt`: ShareGPT DPO format with `messages` lists for `prompt`, `chosen`, and `rejected`.
- `openai_messages`: a flat `prompt` / `chosen` / `rejected` message-list format.

See [dpo_distillation.md](dpo_distillation.md) for the exact schema and how to configure `dataset_info.json` for each format.

For a complete catalog of all input and intermediate JSONL schemas used by EasyDistill 2, see [data_formats.md](data_formats.md).

## SFT training

### LLaMA-Factory

#### Register the dataset

Add the dataset to `data/dataset_info.json`:

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

#### Full fine-tuning example

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

#### LoRA fine-tuning example

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

Adjust `lora_rank`, `lora_alpha`, and `lora_target` based on your base model. For Qwen models, `lora_target all` is usually sufficient; for Llama models you may prefer `lora_target q_proj,v_proj`.

#### Merge LoRA weights (optional)

After LoRA training, merge adapters back into the base model:

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/qwen_cot_lora \
  --template qwen \
  --finetuning_type lora \
  --export_dir outputs/qwen_cot_merged
```

### ms-swift

#### Dataset format

ms-swift supports the same `messages` format. Copy or symlink the distilled JSONL file into a `custom_dataset` folder and register it in the swift dataset config, or pass the file path directly.

#### Full fine-tuning example

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

#### LoRA fine-tuning example

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

For ms-swift, `lora_target_modules ALL` targets all linear layers. You can also specify explicit modules such as `q_proj,k_proj,v_proj,o_proj`.

#### Merge LoRA weights (optional)

```bash
swift export \
  --ckpt_dir outputs/swift_qwen_cot_lora \
  --merge_lora True \
  --output_dir outputs/swift_qwen_cot_merged
```

## DPO training

DPO preference data produced by `dpo_data_build` can be consumed directly by LLaMA-Factory or ms-swift. The examples below assume the default `llama_factory_alpaca` output format.

### LLaMA-Factory

#### Register the dataset

Add the dataset to `data/dataset_info.json`:

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

For ShareGPT DPO format, set `"formatting": "sharegpt"` and `"ranking": "true"` without explicit columns.

#### Full fine-tuning example

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

#### LoRA fine-tuning example

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

DPO learning rates are typically an order of magnitude lower than SFT learning rates. Start with `5e-7` for full fine-tuning and `5e-6` for LoRA.

### ms-swift

#### Dataset format

ms-swift accepts DPO data in Alpaca or ShareGPT format. Pass the JSON/JSONL file directly:

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

#### LoRA fine-tuning example

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

## Training tips

### SFT / CoT data

- The assistant responses contain special tags such as `<|begin_of_thought|>` and `<|begin_of_solution|>`. Make sure the tokenizer of the base model can represent these tokens, or add them as special tokens before training.
- If the base model is not trained to emit CoT tags, consider a warm-up phase with a small learning rate or train only on the reasoning content before introducing the tags.
- For small models (7B), LoRA with `rank >= 8` usually works well. For larger models (14B+), you can increase rank or use full fine-tuning if GPU memory allows.
- When distilling long CoT chains, set `max_length` in the SFT builder to avoid extremely long sequences that may exceed the model context limit during training.

### DPO data

- Ensure the chosen and rejected responses are clearly separable in quality; set `min_margin > 0` in the preference pipeline to avoid ties.
- For CoT preference data, verify that the answer extractor correctly identifies the final answer before building pairs.
- DPO is sensitive to hyperparameters. If training becomes unstable, reduce the learning rate or increase the effective batch size.

## Next steps

- See [instruction_distillation.md](instruction_distillation.md) and [cot_distillation.md](cot_distillation.md) for how to produce SFT data.
- See [dpo_distillation.md](dpo_distillation.md) for how to produce DPO preference data.
- Refer to the LLaMA-Factory and ms-swift documentation for framework-specific options and distributed training setups.
