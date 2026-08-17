# DPO Preference Data

EasyDistill2 can build preference datasets for DPO (Direct Preference Optimization) training. Only the data pipeline is included; training is delegated to frameworks such as LLaMA-Factory or ms-swift.

For the full JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## Supported job type

- `dpo_data_build`: build preference pairs from seed prompts.

Two variants are provided:

- `dpo_instruct_*`: instruction-following preference data, scored by an LLM judge.
- `dpo_cot_*`: chain-of-thought preference data, scored by answer correctness and reasoning conciseness.

## Pipeline stages

A preference config contains four stages:

1. `generate_candidates`: generate `n` responses per prompt.
2. `score_candidates`: score each candidate with a scorer.
3. `build_preference_pairs`: pick `chosen` (highest score) and `rejected` (lowest score) per prompt.
4. `build_preference_dataset`: export pairs to a training framework format.

## Config fields

### `backend`

Same as other EasyDistill2 configs. Supports `pai_token` and `pai_eas`.

### `dataset`

- `input_path`: JSONL file with seed prompts.
- `output_path`: final dataset output path.

For `dpo_instruct` the input usually contains an `instruction` field.
For `dpo_cot` the input should contain `problem` and `answer` fields.

### `generation`

Global generation defaults:
- `system_prompt`, `temperature`, `max_tokens`, `max_workers`, `show_progress`.

### `preference`

Default settings for the preference pipeline. Each field can be overridden in the
individual stage `config` blocks below.

- `scorer`: `llm_judge` (for instruction data) or `cot` (for CoT data).
- `n`: number of candidate responses per prompt.
- `metrics`: metrics used by the LLM judge (default `["helpfulness", "correctness"]`).
- `min_margin`: minimum score gap between chosen and rejected (default `0.0`).
- `max_pairs_per_prompt`: pairs emitted per prompt (default `1`).
- `require_chosen_correct`: require the chosen response to match the reference answer.
- `format`: final output format.
- `instruction_key`: input prompt field (default `"instruction"`). Use `"problem"` for CoT data.
- `answer_key`: reference answer field (default `"answer"`).
- `system_key`: optional per-row system prompt field (default `"system"`).

For the `cot` scorer you can also set:
- `alpha`: length penalty coefficient (default `0.001`).
- `normalize_answer`: normalize answers before comparison (default `true`).
- `answer_extractor_pattern`: custom regex to extract the final answer.

## Stage config reference

Each stage can override the top-level `preference` defaults in its own `config` block.

### `generate_candidates`

- `n`: candidates per prompt.
- `instruction_key`, `system_key`: row fields to use as prompt / system prompt.
- `temperature`, `max_tokens`, `max_workers`, `show_progress`.

### `score_candidates`

- `scorer`: `llm_judge` or `cot`.
- `metrics`: for `llm_judge`.
- `instruction_key`, `answer_key`: row fields to use.
- `alpha`, `normalize_answer`, `answer_extractor_pattern`: for `cot`.
- `temperature`, `max_tokens`, `max_workers`, `show_progress`: for the judge model.

### `build_preference_pairs`

- `min_margin`: minimum score gap (set to a positive value to avoid ties).
- `max_pairs_per_prompt`.
- `require_chosen_correct`.
- `instruction_key`, `answer_key`, `system_key`.

### `build_preference_dataset`

- `format`: one of `llama_factory_alpaca`, `llama_factory_sharegpt`, `openai_messages`.
- `instruction_key`, `system_key`.
- `skip_empty`, `min_length`, `max_length`.

## Data formats by pipeline stage

### Inputs

`dpo_instruct_*` seed prompts:

```jsonl
{"instruction": "Explain the concept of knowledge distillation in one paragraph."}
```

`dpo_cot_*` seed problems with reference answers:

```jsonl
{"problem": "What is the sum of the first 10 positive integers?", "answer": "55"}
```

### Stage outputs

`generate_candidates` emits one row per prompt with `n` candidate responses:

```jsonl
{
  "id": "1",
  "instruction": "Explain the concept of knowledge distillation in one paragraph.",
  "candidates": ["...", "..."],
  "candidate_results": [
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}},
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}}
  ]
}
```

`score_candidates` appends `candidate_scores` (and `candidate_correctness` for the CoT scorer):

```jsonl
{
  "id": "1",
  "instruction": "Explain the concept of knowledge distillation in one paragraph.",
  "candidates": ["...", "..."],
  "candidate_scores": [4.0, 4.0]
}
```

`build_preference_pairs` emits one `chosen` / `rejected` pair per prompt:

```jsonl
{
  "id": "1",
  "instruction": "Explain the concept of knowledge distillation in one paragraph.",
  "system": null,
  "chosen": "...",
  "rejected": "...",
  "chosen_score": 4.0,
  "rejected_score": 4.0,
  "answer": null
}
```

For CoT DPO, `instruction` is replaced by `problem` and `answer` contains the reference answer.

### Final dataset formats

`build_preference_dataset` exports one of the following formats based on `preference.format`:

`llama_factory_alpaca`:

```jsonl
{"instruction": "...", "input": "", "chosen": "...", "rejected": "..."}
```

`llama_factory_sharegpt`:

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

`openai_messages`:

```jsonl
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

## Example: DPO instruct with PAI-Token

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/preference/dpo_instruct_pai_token.yaml
```

## Example: DPO CoT with PAI-EAS

```bash
export EAS_ENDPOINT_URL=https://your-service.cn-shanghai.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/preference/dpo_cot_pai_eas.yaml
```

The endpoint URL may also end with `/v1/chat/completions`; it is normalized
automatically.

## Notes on CoT answer extraction

The `cot` scorer extracts a final answer from each candidate using, in order:
`\boxed{...}`, `#### ...`, "the answer is ...", the last standalone number,
and finally the last non-empty line. If your reference answers are not plain
answers, set `answer_extractor_pattern` to a regex that captures the desired
value.

## LLaMA-Factory dataset_info.json

For Alpaca DPO output:

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

For ShareGPT DPO output:

```json
{
  "my_dpo": {
    "file_name": "dpo_instruct_dataset_pai_token.json",
    "formatting": "sharegpt",
    "ranking": "true"
  }
}
```

## Training

See [training_guide.md](training_guide.md) for LLaMA-Factory and ms-swift training commands for both full fine-tuning and LoRA.
