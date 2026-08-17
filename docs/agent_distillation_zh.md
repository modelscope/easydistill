# Agent 蒸馏

`agent_distill` 流水线从角色/背景种子合成虚拟工具使用任务，通过 LangGraph ReAct 循环展开多轮 Agent 轨迹，使用 LLM 裁判 rubric 对比轨迹并选出最优解，最后将最优轨迹导出为 SFT 多轮数据或 DPO 偏好对。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 适用场景

当你需要为工具使用或函数调用 Agent 生成训练数据时，使用该流水线。它适用于以下情况：

- 真实用户任务稀缺或过于垂直。
- 希望同时生成多样化合成任务与合理的工具定义。
- 需要用于监督微调的 `(用户, 助手, 工具)` 多轮对话轨迹。
- 希望获得 DPO 训练的 chosen/rejected 轨迹偏好对。

流水线自动完成：

1. 根据角色或背景种子合成虚拟任务、工具集、工作流与约束。
2. 将任务改写为模糊、信息不足的用户请求。
3. 针对模糊任务验证并重写工具集。
4. 通过 `SolveAgent / MockTool / MockUser` 的 LangGraph 循环展开 `repeat_times` 条 Agent 轨迹。
5. 使用 rubric 对轨迹打分并选择最优解。
6. 构建 OpenAI/ShareGPT `messages` 格式的 SFT 数据集，或构建 DPO 偏好数据集。

## 流水线阶段

| 阶段 | 是否必需 | 用途 |
|---|---|---|
| `agent_task_synthesis` | 必需（首位） | 从角色/背景种子合成 `task`、`tools`、`workflow`、`restriction`。 |
| `agent_fuzzy_task` | 必需 | 将任务改写为信息不足的用户请求（`fuzzy_task`）并生成世界背景。 |
| `agent_tool_check` | 必需 | 验证并重写适用于模糊任务的工具集（`checked_tools`）。 |
| `agent_trajectory` | 必需 | 为每个任务生成 `repeat_times` 条多轮轨迹。 |
| `agent_rubrics` | 可选 | 使用 LLM 裁判对比轨迹并选出最优解。 |
| `build_sft` | 必需（末位） | 将每个任务的最优轨迹导出为多轮 SFT messages。 |
| `build_preference_dataset` | 必需（末位） | 将轨迹导出为 DPO chosen/rejected 偏好对。 |

流水线必须以 `build_sft` 或 `build_preference_dataset` 结尾。

## 输入格式

每条种子是一个 JSON 对象，至少包含以下字段之一：

```json
{"id": "persona_001", "background": "一位想组织本地活动的南非荷兰语音乐爱好者。"}
```

| 字段 | 是否必需 | 说明 |
|---|---|---|
| `id` | 建议 | 稳定标识符，贯穿所有中间结果。 |
| `background` | 至少一个 | 用于任务合成的角色或背景描述。 |
| `persona` | 至少一个 | `background` 的别名；当 `background` 缺失时使用。 |

示例种子文件见 `examples/seed_agent_personas.jsonl`。

## 配置 Schema

```yaml
job_type: agent_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  base_url: ${PAI_TOKEN_BASE_URL}
  model_id: kimi-k2.6
  timeout: 120.0
  max_retries: 3

generation:
  system_prompt: "You are a helpful assistant that follows instructions precisely and uses XML tags for structured output."
  temperature: 0.7
  max_tokens: 4096

sft:
  skip_empty: true
  min_length: 10
  max_length: 16384

agent:
  max_steps: 10
  repeat_times: 2
  max_tool_calls: 20
  use_rubrics: true

pipeline:
  - stage: agent_task_synthesis
    config:
      prompt_template_file: configs/prompts/agent_task_synthesis_prompt.txt
      temperature: 0.7
      max_tokens: 4096
      show_progress: true
      max_workers: 3
    output_path: outputs/agent_stage1_tasks.jsonl

  - stage: agent_fuzzy_task
    config:
      prompt_template_file: configs/prompts/agent_fuzzy_task_prompt.txt
      temperature: 0.7
      max_tokens: 4096
      show_progress: true
      max_workers: 3
    output_path: outputs/agent_stage2_fuzzy.jsonl

  - stage: agent_tool_check
    config:
      prompt_template_file: configs/prompts/agent_tool_check_prompt.txt
      temperature: 0.7
      max_tokens: 4096
      show_progress: true
      max_workers: 3
    output_path: outputs/agent_stage3_tools.jsonl

  - stage: agent_trajectory
    config:
      max_steps: 10
      repeat_times: 2
      solve:
        temperature: 0.7
        max_tokens: 4096
        retry_attempts: 2
      mock_tool:
        temperature: 0.5
        max_tokens: 2048
        retry_attempts: 2
      mock_user:
        temperature: 0.5
        max_tokens: 2048
        retry_attempts: 2
    output_path: outputs/agent_stage4_trajectories.jsonl

  - stage: agent_rubrics
    config:
      prompt_template_file: configs/prompts/agent_rubrics_prompt.txt
      temperature: 0.3
      max_tokens: 4096
      show_progress: true
      max_workers: 2
      solution_top_k: 3
    output_path: outputs/agent_stage5_rubrics.jsonl

  - stage: build_sft
    config: {}

dataset:
  input_path: examples/seed_agent_personas.jsonl
  output_path: outputs/agent_sft_pai_token.jsonl
```

### 顶层配置

- `job_type`: 必须为 `agent_distill`。
- `backend`: 任意支持的后端（`openai`、`pai_token`、`pai_eas`）。
- `generation`: 默认生成参数，未覆盖时使用。
- `sft`: `build_sft` 应用的长度过滤。
- `agent`: Agent 轨迹展开的全局默认值。
  - `max_steps`: 每条轨迹 ReAct 步数硬上限。
  - `repeat_times`: 每个任务生成的轨迹数。
  - `max_tool_calls`: 每条轨迹工具调用次数硬上限。
  - `use_rubrics`: 是否计划使用 rubric 对比（文档标记位；添加 `agent_rubrics` 阶段以实际生效）。
- `pipeline`: 有序阶段列表。
- `dataset`: `input_path` 与 `output_path`。

### 阶段专属配置

#### agent_task_synthesis

- `prompt_template_file` / `prompt_template`: 从背景合成任务的提示模板。
- 通用生成参数：`temperature`、`max_tokens`、`max_workers`、`show_progress`。

该算子从回复中解析 `<task>`、`<tools>`、`<workflow>`、`<restriction>` XML 标签。缺失标签的行会被丢弃。

#### agent_fuzzy_task

- `prompt_template_file` / `prompt_template`: 将任务改写为模糊用户请求的提示模板。
- 解析 `<task>`（模糊任务）与 `<background>`（世界状态）标签。

#### agent_tool_check

- `prompt_template_file` / `prompt_template`: 验证工具的提示模板。
- 解析 `<tools>` 标签，内容必须是工具定义的 JSON 列表。

#### agent_trajectory

- `max_steps`: 每条轨迹最大 ReAct 步数。默认取 `agent.max_steps`（10）。
- `repeat_times`: 每个任务生成轨迹数。默认取 `agent.repeat_times`（2）。
- `max_tool_calls`: 每条轨迹最大工具调用数。默认取 `agent.max_tool_calls`（20）。
- `solve`: solve agent 的生成配置（model_id、temperature、max_tokens、retry_attempts、retry_backoff_base、retry_max_wait、raise_on_error）。
- `mock_tool`: 模拟工具的生成配置。
- `mock_user`: 模拟用户的生成配置。
- `solve_system_prompt_template_file` / `solve_system_prompt_template`: 可选自定义 solve system 提示。
- `mock_tool_prompt_template_file` / `mock_tool_prompt_template`: 可选自定义 mock-tool 提示。
- `mock_user_prompt_template_file` / `mock_user_prompt_template`: 可选自定义 mock-user 提示。

#### agent_rubrics

- `prompt_template_file` / `prompt_template`: rubric 裁判提示。
- `solution_top_k`: 每个任务参与对比的轨迹数。
- 解析 `<alignment_check>`、`<rubrics>`、`<final>`、`<best_solution>` 标签。`<best_solution>` 应为 `solution_id`，例如 `task_1_solution_1.json`。

#### build_sft

- 应用 `sft.min_length` 与 `sft.max_length` 过滤。
- 每个任务导出一条 rubric 选中的最优轨迹，格式为 `messages`。

#### build_preference_dataset

- 将每个任务的轨迹转换为 DPO 候选。最优轨迹得分为 `1.0`，其余为 `0.0`。
- 使用 `generation.system_prompt` 作为 prompt 字段的默认 system 提示。
- 通过 `PreferencePairBuilder` 与 `PreferenceDatasetBuilder` 输出 chosen/rejected 对。

## 输出格式

### SFT 输出（`build_sft`）

每行遵循 OpenAI/ShareGPT message 格式：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Plan a concert."},
    {"role": "assistant", "content": "I will search for venues.<tool_call>{...}</tool_call>"},
    {"role": "user", "content": "<tool_response>Found 3 venues.</tool_response>"},
    {"role": "assistant", "content": "<answer>Book the Community Hall.</answer>"}
  ],
  "metadata": {
    "task_id": "persona_001",
    "solution_id": "persona_001_solution_1.json",
    "task_finished": "Terminated",
    "source": "teacher_model",
    "model": "agent_distill",
    "task": "Plan a local concert...",
    "fuzzy_task": "Help organize a small concert...",
    "restriction": "Stay within budget.",
    "workflow": "1. Find venues 2. Book artists 3. Promote"
  }
}
```

### DPO 输出（`build_preference_dataset`）

每行是一个 DPO 偏好对：

```json
{
  "prompt": "Help organize a small concert on a limited budget.",
  "chosen": "[{...multi-turn messages...}]",
  "rejected": "[{...multi-turn messages...}]",
  "system": "You are a helpful assistant..."
}
```

`chosen` 为得分最高的最优轨迹，`rejected` 为得分最低的轨迹。

## CLI 用法

```bash
# PAI-Token 服务
export PAI_TOKEN_API_KEY=your_key
export PAI_TOKEN_BASE_URL=https://your-endpoint/v1
easydistill --config configs/pipeline/agent_distill_pai_token.yaml

# PAI-EAS 自部署端点
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/pipeline/agent_distill_pai_eas.yaml

# DPO 偏好数据集（PAI-Token）
easydistill --config configs/pipeline/agent_distill_dpo_pai_token.yaml

# DPO 偏好数据集（PAI-EAS）
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/pipeline/agent_distill_dpo_pai_eas.yaml
```

若需要 DPO 数据集而非 SFT，只需将末位的 `build_sft` 阶段替换为 `build_preference_dataset`，将 `dataset.output_path` 指向 `.json` 文件即可。完整示例参见 [`configs/pipeline/agent_distill_dpo_pai_token.yaml`](../configs/pipeline/agent_distill_dpo_pai_token.yaml) (PAI-Token) 或 [`configs/pipeline/agent_distill_dpo_pai_eas.yaml`](../configs/pipeline/agent_distill_dpo_pai_eas.yaml) (PAI-EAS)。
