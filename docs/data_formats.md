# EasyDistill 2 JSONL Data Formats

This document describes the JSONL schemas used as inputs and outputs across EasyDistill 2 pipelines and standalone jobs. All JSONL files use UTF-8 encoding and contain one valid JSON object per line.

## Common input formats

### Seed instructions

Used by instruction-distillation jobs and pipelines.

```jsonl
{"instruction": "What is the capital of France?"}
{"instruction": "Explain quantum computing in one sentence.", "system": "You are a concise tutor."}
```

Fields:
- `instruction` (string, required): the user prompt.
- `system` (string, optional): per-row system prompt. Falls back to the config-level `system_prompt`.
- `id` (string/integer, optional): row identifier; auto-generated if omitted.

### Seed problems for CoT

Used by `cot_distill` and `advanced_cot_distill`.

```jsonl
{"problem": "What is the sum of the first 10 positive integers?"}
{"instruction": "What is 2+2?"}
```

The problem field is configurable via `dataset.problem_key` (default `problem`, fallback `instruction`).

### Problem / answer pairs for CoT rewriting

Used by `cot_long2short` and `cot_short2long`.

```jsonl
{"instruction": "What is 2+2?", "response": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
```

Keys are configurable via `dataset.problem_key` (default `instruction`) and `dataset.answer_key` (default `response`). Common fallbacks (`problem`, `answer`, `output`) are also accepted.

### Multi-modal inputs

Used by `mm_instruct_distill` and `mm_cot_distill`.

```jsonl
{"id": "mm_0", "instruction": "Describe what you see.", "images": ["examples/mm_sample_image.png"]}
{"id": "mm_1", "instruction": "What color is the main object?", "images": ["https://example.com/img.png"]}
```

Fields:
- `instruction` (string, required): the text prompt.
- `images` (list of strings, required): image references. Each item may be a local path, `file://` URI, `http(s)://` URL, or base64 data URL such as `data:image/png;base64,...`.
- `id` (optional): row identifier.

### Evaluation inputs

Used by `instruct_eval`, `cot_eval`, `mm_instruct_eval`, and `mm_cot_eval`.

Plain format:

```jsonl
{"instruction": "What is the capital of France?", "output": "Paris"}
```

SFT messages format (auto-converted):

```jsonl
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]}
```

For multi-modal evaluation, `images` may also be present.

### Raw text for response extraction

Used by `instruction_response_extraction`.

```jsonl
{"text": "User: What is 2+2?\nAssistant: 2+2 equals 4."}
```

### DPO seed prompts

For `dpo_instruct_*`:

```jsonl
{"instruction": "Explain knowledge distillation in one paragraph."}
```

For `dpo_cot_*`:

```jsonl
{"problem": "What is the sum of the first 10 positive integers?", "answer": "55"}
```

Fields are configurable via `instruction_key` / `answer_key`.

## Common output formats

### SFT messages

Produced by any job that ends with `build_sft` or by standalone distillation jobs such as `instruct_distill`, `cot_distill`, `mm_instruct_distill`, and `mm_cot_distill`.

Text-only SFT:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ],
  "metadata": {
    "source": "teacher_model",
    "model": "Qwen2.5-3B-Instruct",
    "request_id": "1",
    "backend": "pai_eas",
    "usage": {"completion_tokens": 1, "prompt_tokens": 31, "total_tokens": 32}
  }
}
```

Multi-modal SFT:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful visual assistant."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe what you see."}
    ]},
    {"role": "assistant", "content": "The image shows a solid red square."}
  ],
  "metadata": {
    "source": "teacher_model",
    "model": "Qwen2.5-VL-3B-Instruct",
    "request_id": "mm_gen_0",
    "backend": "pai_eas",
    "instruction": "Describe what you see.",
    "images": ["examples/mm_sample_image.png"],
    "usage": {"completion_tokens": 235, "prompt_tokens": 103, "total_tokens": 338}
  }
}
```

Fields:
- `messages` (list, required): OpenAI/ShareGPT message objects. Multi-modal user messages contain a list of `image_url` and `text` content items.
- `metadata` (dict, optional): provenance information such as `source`, `model`, `request_id`, `backend`, `usage`, and original `instruction` / `images`.

### Evaluation outputs

Produced by `instruct_eval` and `cot_eval`.

```jsonl
{"id": "0", "instruction": "What is the capital of France?", "output": "Paris", "informativeness": 2, "helpfulness": 7, "generalization": 1, "correctness": true}
```

```jsonl
{"id": "0", "instruction": "What is the sum of the first 10 positive integers?", "output": "...", "reasoning_verbosity": 5, "cognitive_difficulty": 5, "logical_correctness": true}
```

The original row fields are preserved and the requested metrics are appended.

### Instruction expansion output

Produced by `instruction_expansion`.

```jsonl
{"instruction": "Write a new instruction similar to the examples but different in content."}
```

### Instruction refinement output

Produced by `instruction_refinement`.

```jsonl
{"instruction": "Rewrite the input instruction to be clearer and more specific."}
```

### Instruction balancing output

Produced by `instruction_balance`.

```jsonl
{"instruction": "What is 2+2?", "category": "Math"}
```

The original fields are preserved and a `category` field is added.

### Generated response rows (before SFT conversion)

Produced by the `generate` stage inside pipelines.

```jsonl
{"instruction": "What is 2+2?", "output": "4"}
```

### Quality-filtered rows

Produced by the `quality_filter` stage inside pipelines. The format is the same as the evaluated rows, with only rows that pass the thresholds retained.

```jsonl
{"instruction": "What is 2+2?", "output": "4", "correctness": true, "helpfulness": 7}
```

### CoT RV/CD scored rows

Produced by the `cot_rvcd_score` stage.

```jsonl
{
  "instruction": "What is the sum of the first 10 positive integers?",
  "response": "...",
  "reasoning_verbosity": 5,
  "cognitive_difficulty": 4,
  "logical_correctness": true
}
```

### CoT RV/CD mixed rows

Produced by the `cot_mix_by_rv_cd` stage.

```jsonl
{
  "instruction": "...",
  "response": "...",
  "reasoning_verbosity": 2,
  "cognitive_difficulty": 2,
  "logical_correctness": true,
  "cd_bin": 0,
  "rv_target": 2.0
}
```

### Agent distillation formats

Used by the `agent_distill` pipeline.

#### Agent seed personas

Input to `agent_task_synthesis`.

```jsonl
{"id": "persona_001", "background": "An Afrikaans music fan who wants to organize local events."}
```

Fields:
- `id` (string/integer, optional): row identifier.
- `background` or `persona` (string, required): persona or background description.

#### `agent_task_synthesis` output

```jsonl
{
  "id": "persona_001",
  "background": "An Afrikaans music fan...",
  "task": "Plan a local concert for Afrikaans music.",
  "tools": [{"name": "search_venues", "description": "Search venues"}],
  "workflow": "1. Find venues 2. Book artists 3. Promote",
  "restriction": "Stay within the stated budget.",
  "initial_toolset_create": "<task>...</task><tools>...</tools>..."
}
```

#### `agent_fuzzy_task` output

```jsonl
{
  "id": "persona_001",
  "fuzzy_task": "Help organize a small concert on a limited budget.",
  "task_background": "The user is an Afrikaans music fan with no event-planning experience...",
  "raw_fuzzy_task": "<task>...</task><background>...</background>"
}
```

#### `agent_tool_check` output

```jsonl
{
  "id": "persona_001",
  "checked_tools": [{"name": "search_venues", "description": "Search venues"}],
  "raw_tool_check": "<tools>...</tools>"
}
```

#### `agent_trajectory` output

One row per rollout.

```jsonl
{
  "id": "persona_001",
  "solution_id": "persona_001_solution_1.json",
  "fuzzy_task": "Help organize a small concert on a limited budget.",
  "task_background": "...",
  "restriction": "Stay within budget.",
  "checked_tools": [{"name": "search_venues"}],
  "trajectory": [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Help organize a small concert on a limited budget."},
    {"role": "assistant", "content": "I will search for venues.<tool_call>{...}</tool_call>"},
    {"role": "user", "content": "<tool_response>Found 3 venues.</tool_response>"},
    {"role": "assistant", "content": "<answer>Book the Community Hall.</answer>"}
  ],
  "tool_call_history": ["Query:\n...\nResponse:\n..."],
  "task_finished": "Terminated"
}
```

#### `agent_rubrics` output

One row per task, grouping trajectories and selecting the best solution.

```jsonl
{
  "id": "persona_001",
  "fuzzy_task": "Help organize a small concert on a limited budget.",
  "best_solution_id": "persona_001_solution_1.json",
  "alignment_check": "The trajectories align with the task.",
  "rubrics": "1. Correctness 2. Efficiency",
  "final": "Solution 1 is best.",
  "trajectories": [...]
}
```

#### `build_sft` output for agent distillation

Standard SFT `messages` format. The `metadata` block includes `task_id`, `solution_id`, `task_finished`, `task`, `fuzzy_task`, `restriction`, and `workflow`.

#### `build_preference_dataset` output for agent distillation

```jsonl
{
  "prompt": "Help organize a small concert on a limited budget.",
  "chosen": "[{...best trajectory messages...}]",
  "rejected": "[{...worst trajectory messages...}]",
  "system": "You are a helpful assistant..."
}
```

### DPO intermediate formats

#### `generate_candidates` output

```jsonl
{
  "id": "1",
  "instruction": "Explain knowledge distillation in one paragraph.",
  "candidates": ["...", "..."],
  "candidate_results": [
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}},
    {"request": {...}, "response": "...", "model": "...", "usage": {...}, "metadata": {...}}
  ]
}
```

For CoT DPO, the row contains `problem` and `answer` instead of `instruction`.

#### `score_candidates` output

Same as candidates output with `candidate_scores` and, for the CoT scorer, `candidate_correctness` added.

```jsonl
{
  "id": "1",
  "instruction": "Explain knowledge distillation in one paragraph.",
  "candidates": ["...", "..."],
  "candidate_scores": [4.0, 4.0]
}
```

#### `build_preference_pairs` output

```jsonl
{
  "id": "1",
  "instruction": "Explain knowledge distillation in one paragraph.",
  "system": null,
  "chosen": "...",
  "rejected": "...",
  "chosen_score": 4.0,
  "rejected_score": 4.0,
  "answer": null
}
```

For CoT DPO, `instruction` is replaced by `problem` and `answer` contains the reference answer.

#### `build_preference_dataset` output

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

### Multi-modal CoT rewrite input/output

`mm_cot_long2short` and `mm_cot_short2long` accept both raw rows and SFT message rows.

Raw input:

```jsonl
{"instruction": "Look at the image and determine the dominant color.", "images": ["examples/mm_sample_image.png"], "response": "..."}
```

SFT message input (auto-converted, images read from `metadata.images`):

```jsonl
{
  "messages": [
    {"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"images": ["examples/mm_sample_image.png"]}
}
```

`mm_cot_long2short` output includes `response` (simplified), `original_response`, `original_tokens`, `simplified_tokens`, and `compression_ratio`.

`mm_cot_short2long` output includes `response` (extended), `original_response`, `original_tokens`, `extended_tokens`, `expansion_ratio`, and `step_count`.

### T2I distillation formats

For T2I distillation input/output schemas and stage-by-stage JSONL formats, see [t2i_distillation.md](t2i_distillation.md) for an overview and [t2i_distillation_implementation.md](t2i_distillation_implementation.md) for the complete data-flow schemas.

### PE rewrite distillation formats

#### Seed PE prompts

Input for `pe_rewrite_distill` and `seed_anchored_expansion` (see `examples/seed_pe_prompts.jsonl`). `id` is optional and used for expansion lineage; the key is configurable via `dataset.instruction_key` (default `instruction`):

```jsonl
{"id": "pe_seed_001", "instruction": "画一张水循环的科普信息图，包含蒸发、凝结、降水几个环节，中文标注，图标简洁一点"}
```

#### `seed_anchored_expansion` output

One row per generated prompt, with lineage back to the source seed and the round-level dedup `topic`:

```jsonl
{"instruction": "画一张光合作用原理的科普长图...", "source_seed_id": "pe_seed_001", "round": 0, "topic": "光合作用原理图解"}
```

#### `agentic_rewrite` output

Adds the final rewritten prompt (`response`), the plan routing result (`scene` / `language`) and an `agent_trace` audit object; extra input fields (e.g. expansion lineage) pass through unchanged:

```jsonl
{"instruction": "画一张水循环的科普信息图...", "response": "一张竖版科普信息图，主标题\"水循环\"位于顶部...", "scene": "structured_diagram", "language": "zh", "agent_trace": {"plan": {"status": "ok", "raw": "..."}, "rewrite": {"status": "ok", "draft": "..."}, "reflection": {"status": "ok", "changed": false, "notes": "", "raw": "..."}, "durations": {"plan": 1.2, "rewrite": 8.5, "reflection": 3.1}}, "source_seed_id": "pe_seed_001", "round": 0, "topic": "..."}
```

#### `pe_rewrite_eval` output

Adds seven 0-9 integer metrics and two boolean hard checks to every row (unparseable metrics come back as `null`):

```jsonl
{"instruction": "...", "response": "...", "scene": "structured_diagram", "language": "zh", "intent_fidelity": 8, "text_rendering_completeness": 9, "detail_enrichment": 8, "visual_concreteness": 8, "compositional_coverage": 7, "scene_alignment": 8, "usability": 9, "language_consistency": true, "no_conflict": true, "agent_trace": {"...": "..."}, "source_seed_id": "pe_seed_001", "round": 0}
```

The `pe_rewrite_filter` stage keeps rows passing the score gates (plus the optional per-scene top selection) without changing the row schema.

#### `pe_rewrite_build_sft` output

SFT rows whose system message is the per-language student rewrite instruction. Judge scores and `agent_trace` are audit-only and never enter `metadata`, while scene routing and expansion lineage are carried over:

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是文生图 prompt 改写专家..."},
    {"role": "user", "content": "画一张水循环的科普信息图..."},
    {"role": "assistant", "content": "一张竖版科普信息图，主标题\"水循环\"位于顶部..."}
  ],
  "metadata": {"source": "teacher_model", "model": "pipeline", "request_id": "0", "scene": "structured_diagram", "language": "zh", "source_seed_id": "pe_seed_001", "round": 0, "topic": "..."}
}
```
