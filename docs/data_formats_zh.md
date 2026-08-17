# EasyDistill 2 JSONL 数据格式

本文档描述 EasyDistill 2 流水线与独立任务中使用的 JSONL 输入/输出格式。所有 JSONL 文件使用 UTF-8 编码，每行一个有效的 JSON 对象。

## 通用输入格式

### 种子指令

用于指令蒸馏任务与流水线。

```jsonl
{"instruction": "法国的首都是哪里？"}
{"instruction": "用一句话解释量子计算。", "system": "你是一位简洁的导师。"}
```

字段：
- `instruction`（字符串，必需）：用户提示。
- `system`（字符串，可选）：每行系统提示；未设置时使用配置级 `system_prompt`。
- `id`（字符串/整数，可选）：行标识符；省略时自动生成。

### CoT 种子问题

用于 `cot_distill` 和 `advanced_cot_distill`。

```jsonl
{"problem": "前 10 个正整数的和是多少？"}
{"instruction": "2+2 等于多少？"}
```

问题字段可通过 `dataset.problem_key` 配置（默认 `problem`，可回退到 `instruction`）。

### CoT 改写用问题/答案对

用于 `cot_long2short` 和 `cot_short2long`。

```jsonl
{"instruction": "2+2 等于多少？", "response": "<|begin_of_thought|>...<|end_of_thought|><|begin_of_solution|>4<|end_of_solution|>"}
```

字段可通过 `dataset.problem_key`（默认 `instruction`）和 `dataset.answer_key`（默认 `response`）覆盖。也接受常见的回退字段名（`problem`、`answer`、`output`）。

### 多模态输入

用于 `mm_instruct_distill` 和 `mm_cot_distill`。

```jsonl
{"id": "mm_0", "instruction": "描述你看到的内容。", "images": ["examples/mm_sample_image.png"]}
{"id": "mm_1", "instruction": "图中主要物体是什么颜色？", "images": ["https://example.com/img.png"]}
```

字段：
- `instruction`（字符串，必需）：文本提示。
- `images`（字符串列表，必需）：图像引用。每项可以是本地路径、`file://` URI、`http(s)://` URL 或 base64 数据 URL（如 `data:image/png;base64,...`）。
- `id`（可选）：行标识符。

### 评估输入

用于 `instruct_eval`、`cot_eval`、`mm_instruct_eval` 和 `mm_cot_eval`。

普通格式：

```jsonl
{"instruction": "法国的首都是哪里？", "output": "巴黎"}
```

SFT messages 格式（自动转换）：

```jsonl
{"messages": [{"role": "user", "content": "法国的首都是哪里？"}, {"role": "assistant", "content": "巴黎"}]}
```

多模态评估时，行中也可包含 `images`。

### 用于回复抽取的原始文本

用于 `instruction_response_extraction`。

```jsonl
{"text": "用户：2+2 等于多少？\n助手：2+2 等于 4。"}
```

### DPO 种子提示

`dpo_instruct_*` 输入：

```jsonl
{"instruction": "用一段话解释知识蒸馏。"}
```

`dpo_cot_*` 输入：

```jsonl
{"problem": "前 10 个正整数的和是多少？", "answer": "55"}
```

字段可通过 `instruction_key` / `answer_key` 配置。

## 通用输出格式

### SFT messages

由任何以 `build_sft` 结尾的任务或独立蒸馏任务（如 `instruct_distill`、`cot_distill`、`mm_instruct_distill`、`mm_cot_distill`）产出。

纯文本 SFT：

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是一个 helpful 的助手。"},
    {"role": "user", "content": "2+2 等于多少？"},
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

多模态 SFT：

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是一个 helpful 的视觉助手。"},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "描述你看到的内容。"}
    ]},
    {"role": "assistant", "content": "图中是一个纯红色的正方形。"}
  ],
  "metadata": {
    "source": "teacher_model",
    "model": "Qwen2.5-VL-3B-Instruct",
    "request_id": "mm_gen_0",
    "backend": "pai_eas",
    "instruction": "描述你看到的内容。",
    "images": ["examples/mm_sample_image.png"],
    "usage": {"completion_tokens": 235, "prompt_tokens": 103, "total_tokens": 338}
  }
}
```

字段：
- `messages`（列表，必需）：OpenAI/ShareGPT 风格消息对象。多模态用户消息的内容为 `image_url` 与 `text` 内容项的列表。
- `metadata`（字典，可选）：来源信息，包括 `source`、`model`、`request_id`、`backend`、`usage` 以及原始 `instruction` / `images`。

### 评估输出

由 `instruct_eval` 和 `cot_eval` 产出。

```jsonl
{"id": "0", "instruction": "法国的首都是哪里？", "output": "巴黎", "informativeness": 2, "helpfulness": 7, "generalization": 1, "correctness": true}
```

```jsonl
{"id": "0", "instruction": "前 10 个正整数的和是多少？", "output": "...", "reasoning_verbosity": 5, "cognitive_difficulty": 5, "logical_correctness": true}
```

原始行字段被保留，并在末尾追加所请求的指标。

### 指令扩充输出

由 `instruction_expansion` 产出。

```jsonl
{"instruction": "写一个与示例风格相似但内容不同的新指令。"}
```

### 指令精炼输出

由 `instruction_refinement` 产出。

```jsonl
{"instruction": "将输入指令改写得更清晰、更具体。"}
```

### 指令均衡输出

由 `instruction_balance` 产出。

```jsonl
{"instruction": "2+2 等于多少？", "category": "Math"}
```

原始字段被保留，并新增 `category` 字段。

### 生成回复行（SFT 转换前）

由流水线中的 `generate` 阶段产出。

```jsonl
{"instruction": "2+2 等于多少？", "output": "4"}
```

### 质量过滤后的行

由流水线中的 `quality_filter` 阶段产出。格式与评估后的行相同，仅保留通过阈值的行。

```jsonl
{"instruction": "2+2 等于多少？", "output": "4", "correctness": true, "helpfulness": 7}
```

### CoT RV/CD 评分行

由 `cot_rvcd_score` 阶段产出。

```jsonl
{
  "instruction": "前 10 个正整数的和是多少？",
  "response": "...",
  "reasoning_verbosity": 5,
  "cognitive_difficulty": 4,
  "logical_correctness": true
}
```

### CoT RV/CD 混合行

由 `cot_mix_by_rv_cd` 阶段产出。

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

### Agent 蒸馏格式

由 `agent_distill` 流水线使用。

#### Agent 种子角色

`agent_task_synthesis` 的输入。

```jsonl
{"id": "persona_001", "background": "一位想组织本地活动的南非荷兰语音乐爱好者。"}
```

字段：
- `id`（字符串/整数，可选）：行标识符。
- `background` 或 `persona`（字符串，必需）：角色或背景描述。

#### `agent_task_synthesis` 输出

```jsonl
{
  "id": "persona_001",
  "background": "一位南非荷兰语音乐爱好者...",
  "task": "为南非荷兰语音乐策划一场本地音乐会。",
  "tools": [{"name": "search_venues", "description": "Search venues"}],
  "workflow": "1. 寻找场地 2. 预约艺人 3. 宣传推广",
  "restriction": "保持在规定预算内。",
  "initial_toolset_create": "<task>...</task><tools>...</tools>..."
}
```

#### `agent_fuzzy_task` 输出

```jsonl
{
  "id": "persona_001",
  "fuzzy_task": "帮我用有限预算组织一场小型音乐会。",
  "task_background": "用户是一位没有活动策划经验的南非荷兰语音乐爱好者...",
  "raw_fuzzy_task": "<task>...</task><background>...</background>"
}
```

#### `agent_tool_check` 输出

```jsonl
{
  "id": "persona_001",
  "checked_tools": [{"name": "search_venues", "description": "Search venues"}],
  "raw_tool_check": "<tools>...</tools>"
}
```

#### `agent_trajectory` 输出

每次 rollout 输出一行。

```jsonl
{
  "id": "persona_001",
  "solution_id": "persona_001_solution_1.json",
  "fuzzy_task": "帮我用有限预算组织一场小型音乐会。",
  "task_background": "...",
  "restriction": "保持在预算内。",
  "checked_tools": [{"name": "search_venues"}],
  "trajectory": [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "帮我用有限预算组织一场小型音乐会。"},
    {"role": "assistant", "content": "I will search for venues.<tool_call>{...}</tool_call>"},
    {"role": "user", "content": "<tool_response>Found 3 venues.</tool_response>"},
    {"role": "assistant", "content": "<answer>Book the Community Hall.</answer>"}
  ],
  "tool_call_history": ["Query:\n...\nResponse:\n..."],
  "task_finished": "Terminated"
}
```

#### `agent_rubrics` 输出

每个任务一行，聚合轨迹并选择最优解。

```jsonl
{
  "id": "persona_001",
  "fuzzy_task": "帮我用有限预算组织一场小型音乐会。",
  "best_solution_id": "persona_001_solution_1.json",
  "alignment_check": "The trajectories align with the task.",
  "rubrics": "1. Correctness 2. Efficiency",
  "final": "Solution 1 is best.",
  "trajectories": [...]
}
```

#### Agent 蒸馏 `build_sft` 输出

标准 SFT `messages` 格式。`metadata` 包含 `task_id`、`solution_id`、`task_finished`、`task`、`fuzzy_task`、`restriction` 和 `workflow`。

#### Agent 蒸馏 `build_preference_dataset` 输出

```jsonl
{
  "prompt": "帮我用有限预算组织一场小型音乐会。",
  "chosen": "[{...最优轨迹消息...}]",
  "rejected": "[{...最差轨迹消息...}]",
  "system": "You are a helpful assistant..."
}
```

### DPO 中间格式

#### `generate_candidates` 输出

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

CoT DPO 使用 `problem` 和 `answer` 替代 `instruction`。

#### `score_candidates` 输出

与 candidates 输出相同，额外追加 `candidate_scores`；CoT 评分器还会追加 `candidate_correctness`。

```jsonl
{
  "id": "1",
  "instruction": "用一段话解释知识蒸馏。",
  "candidates": ["...", "..."],
  "candidate_scores": [4.0, 4.0]
}
```

#### `build_preference_pairs` 输出

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

#### `build_preference_dataset` 输出

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

### 多模态 CoT 改写输入/输出

`mm_cot_long2short` 和 `mm_cot_short2long` 同时接受原始行和 SFT message 行。

原始输入：

```jsonl
{"instruction": "看图并判断主导颜色。", "images": ["examples/mm_sample_image.png"], "response": "..."}
```

SFT message 输入（自动转换，图像从 `metadata.images` 读取）：

```jsonl
{
  "messages": [
    {"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"images": ["examples/mm_sample_image.png"]}
}
```

`mm_cot_long2short` 输出包含 `response`（简化后）、`original_response`、`original_tokens`、`simplified_tokens`、`compression_ratio`。

`mm_cot_short2long` 输出包含 `response`（扩展后）、`original_response`、`original_tokens`、`extended_tokens`、`expansion_ratio`、`step_count`。

### T2I 文生图蒸馏格式

T2I 文生图蒸馏的输入/输出 schema 与各阶段 JSONL 格式，请见 [t2i_distillation_zh.md](t2i_distillation_zh.md) 获取概览，或见 [t2i_distillation_implementation.md](t2i_distillation_implementation.md) 获取完整的数据流 schema。

### PE 改写蒸馏格式

#### PE 种子 prompt

`pe_rewrite_distill` 与 `seed_anchored_expansion` 的输入（见 `examples/seed_pe_prompts.jsonl`）。`id` 可选，用于扩展血统追溯；字段名可通过 `dataset.instruction_key` 配置（默认 `instruction`）：

```jsonl
{"id": "pe_seed_001", "instruction": "画一张水循环的科普信息图，包含蒸发、凝结、降水几个环节，中文标注，图标简洁一点"}
```

#### `seed_anchored_expansion` 输出

每条生成的 prompt 占一行，携带回溯到源种子的血统字段和轮次级去重 `topic`：

```jsonl
{"instruction": "画一张光合作用原理的科普长图...", "source_seed_id": "pe_seed_001", "round": 0, "topic": "光合作用原理图解"}
```

#### `agentic_rewrite` 输出

新增最终改写结果（`response`）、plan 路由结果（`scene` / `language`）与审计用的 `agent_trace` 对象；输入行的额外字段（如扩展血统）原样透传：

```jsonl
{"instruction": "画一张水循环的科普信息图...", "response": "一张竖版科普信息图，主标题\"水循环\"位于顶部...", "scene": "structured_diagram", "language": "zh", "agent_trace": {"plan": {"status": "ok", "raw": "..."}, "rewrite": {"status": "ok", "draft": "..."}, "reflection": {"status": "ok", "changed": false, "notes": "", "raw": "..."}, "durations": {"plan": 1.2, "rewrite": 8.5, "reflection": 3.1}}, "source_seed_id": "pe_seed_001", "round": 0, "topic": "..."}
```

#### `pe_rewrite_eval` 输出

为每行新增 7 个 0-9 整数评分维度与 2 个布尔硬校验（无法解析的维度为 `null`）：

```jsonl
{"instruction": "...", "response": "...", "scene": "structured_diagram", "language": "zh", "intent_fidelity": 8, "text_rendering_completeness": 9, "detail_enrichment": 8, "visual_concreteness": 8, "compositional_coverage": 7, "scene_alignment": 8, "usability": 9, "language_consistency": true, "no_conflict": true, "agent_trace": {"...": "..."}, "source_seed_id": "pe_seed_001", "round": 0}
```

`pe_rewrite_filter` 阶段保留通过分数门槛（及可选的分场景 top 筛选）的行，不改变行结构。

#### `pe_rewrite_build_sft` 输出

SFT 行的 system 消息为分语言的学生改写指令。裁判分数与 `agent_trace` 仅用于审计，不会进入 `metadata`；场景路由与扩展血统字段会携带过去：

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
