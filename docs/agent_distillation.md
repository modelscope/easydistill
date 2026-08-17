# Agent Distillation

The `agent_distill` pipeline synthesizes virtual tool-use tasks from persona/background seeds, rolls out multi-turn agent trajectories with a LangGraph ReAct loop, compares the trajectories with an LLM rubric judge, and exports the best trajectories as SFT multi-turn data or DPO preference pairs.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## When to use this pipeline

Use this pipeline when you need training data for tool-use or function-calling agents. It is designed for cases where:

- Real user tasks are scarce or too domain-specific.
- You want to generate diverse, synthetic tasks together with plausible tool definitions.
- You need multi-turn `(user, assistant, tool)` conversation traces for supervised fine-tuning.
- You want preference pairs of better/worse trajectories for DPO training.

The pipeline automatically:

1. Synthesizes a virtual task, tool set, workflow, and restriction from a persona or background seed.
2. Rewrites the task into a fuzzy, under-specified user request.
3. Validates and rewrites the tool set for the fuzzy task.
4. Rolls out `repeat_times` agent trajectories using a `SolveAgent / MockTool / MockUser` LangGraph loop.
5. Scores the trajectories with rubrics and selects the best one.
6. Builds an SFT dataset in OpenAI/ShareGPT `messages` format, or a DPO preference dataset with chosen/rejected pairs.

## Pipeline stages

| Stage | Required? | Purpose |
|---|---|---|
| `agent_task_synthesis` | Required (first) | Synthesize `task`, `tools`, `workflow`, `restriction` from a persona/background seed. |
| `agent_fuzzy_task` | Required | Rewrite the task into an under-specified user request (`fuzzy_task`) and world background. |
| `agent_tool_check` | Required | Validate and rewrite the tool set for the fuzzy task (`checked_tools`). |
| `agent_trajectory` | Required | Generate `repeat_times` multi-turn trajectories per task. |
| `agent_rubrics` | Optional | Compare trajectories with an LLM judge and pick the best solution. |
| `build_sft` | Required (last) | Export the best trajectory per task as multi-turn SFT messages. |
| `build_preference_dataset` | Required (last) | Export trajectories as DPO chosen/rejected pairs. |

The last stage must be `build_sft` or `build_preference_dataset`.

## Input format

Each seed row is a JSON object with at least one of the following fields:

```json
{"id": "persona_001", "background": "An Afrikaans music fan who wants to organize local events."}
```

| Field | Required? | Description |
|---|---|---|
| `id` | Recommended | Stable identifier, carried through every intermediate row. |
| `background` | At least one | Persona or background description used for task synthesis. |
| `persona` | At least one | Alias for `background`. Used if `background` is missing. |

A sample seed file is provided at `examples/seed_agent_personas.jsonl`.

## Config schema

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

### Top-level sections

- `job_type`: must be `agent_distill`.
- `backend`: any supported backend (`openai`, `pai_token`, `pai_eas`).
- `generation`: default generation parameters used by prompt-based stages when not overridden.
- `sft`: length filters applied by `build_sft`.
- `agent`: global defaults for agent trajectory rollout.
  - `max_steps`: hard upper bound on ReAct steps per rollout.
  - `repeat_times`: number of trajectories generated per task.
  - `max_tool_calls`: hard upper bound on tool calls per rollout.
  - `use_rubrics`: whether rubric comparison is expected (documented flag; include `agent_rubrics` stage to enforce).
- `pipeline`: ordered list of stages.
- `dataset`: `input_path` and `output_path`.

### Stage-specific configs

#### agent_task_synthesis

- `prompt_template_file` / `prompt_template`: template for synthesizing tasks from background.
- Standard generation knobs: `temperature`, `max_tokens`, `max_workers`, `show_progress`.

The operator parses `<task>`, `<tools>`, `<workflow>`, and `<restriction>` XML tags from the response. Rows with missing tags are dropped.

#### agent_fuzzy_task

- `prompt_template_file` / `prompt_template`: template for rewriting the task into a fuzzy user request.
- Parses `<task>` (fuzzy task) and `<background>` (world state) tags.

#### agent_tool_check

- `prompt_template_file` / `prompt_template`: template for validating tools.
- Parses `<tools>` tag, which must contain a JSON list of tool definitions.

#### agent_trajectory

- `max_steps`: max ReAct steps per rollout. Defaults to `agent.max_steps` (10).
- `repeat_times`: number of trajectories per task. Defaults to `agent.repeat_times` (2).
- `max_tool_calls`: max tool calls per rollout. Defaults to `agent.max_tool_calls` (20).
- `solve`: generation config for the solve agent (model_id, temperature, max_tokens, retry_attempts, retry_backoff_base, retry_max_wait, raise_on_error).
- `mock_tool`: generation config for the simulated tool.
- `mock_user`: generation config for the simulated user.
- `solve_system_prompt_template_file` / `solve_system_prompt_template`: optional custom solve system prompt.
- `mock_tool_prompt_template_file` / `mock_tool_prompt_template`: optional custom mock-tool prompt.
- `mock_user_prompt_template_file` / `mock_user_prompt_template`: optional custom mock-user prompt.

#### agent_rubrics

- `prompt_template_file` / `prompt_template`: rubric judge prompt.
- `solution_top_k`: number of trajectories to compare per task.
- Parses `<alignment_check>`, `<rubrics>`, `<final>`, and `<best_solution>` tags. The `<best_solution>` value should be a `solution_id` such as `task_1_solution_1.json`.

#### build_sft

- Applies `sft.min_length` and `sft.max_length` filters.
- Exports one `messages`-format sample per task using the rubric-selected best trajectory.

#### build_preference_dataset

- Converts each task's trajectories into DPO candidates. The best trajectory scores `1.0`; all others score `0.0`.
- Uses `generation.system_prompt` as the default system prompt for the prompt field.
- Outputs chosen/rejected pairs via `PreferencePairBuilder` and `PreferenceDatasetBuilder`.

## Output formats

### SFT output (`build_sft`)

Each row follows the OpenAI/ShareGPT message format:

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

### DPO output (`build_preference_dataset`)

Each row is a DPO preference pair:

```json
{
  "prompt": "Help organize a small concert on a limited budget.",
  "chosen": "[{...multi-turn messages...}]",
  "rejected": "[{...multi-turn messages...}]",
  "system": "You are a helpful assistant..."
}
```

The `chosen` field contains the best trajectory (highest rubric score); `rejected` contains the lowest-scoring trajectory.

## CLI usage

```bash
# PAI-Token service
export PAI_TOKEN_API_KEY=your_key
export PAI_TOKEN_BASE_URL=https://your-endpoint/v1
easydistill --config configs/pipeline/agent_distill_pai_token.yaml

# PAI-EAS self-deployed endpoint
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/pipeline/agent_distill_pai_eas.yaml

# DPO preference dataset (PAI-Token)
easydistill --config configs/pipeline/agent_distill_dpo_pai_token.yaml

# DPO preference dataset (PAI-EAS)
export EAS_ENDPOINT_URL=https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
export EAS_TOKEN=your_token
easydistill --config configs/pipeline/agent_distill_dpo_pai_eas.yaml
```

To produce a DPO dataset instead of SFT, replace the final `build_sft` stage with `build_preference_dataset` and point `dataset.output_path` to a `.json` file. See [`configs/pipeline/agent_distill_dpo_pai_token.yaml`](../configs/pipeline/agent_distill_dpo_pai_token.yaml) (PAI-Token) or [`configs/pipeline/agent_distill_dpo_pai_eas.yaml`](../configs/pipeline/agent_distill_dpo_pai_eas.yaml) (PAI-EAS) for a complete example.
