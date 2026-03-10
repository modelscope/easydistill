# AgentKD Usage Tutorial

AgentKD extends EasyDistill with **virtual tool-use data synthesis and distillation**. It can **directly generate diverse tool-use metadata** from persona seeds—including tools, policies, and test cases—and produce teacher-model solution trajectories for knowledge distillation.

**Virtual tool-use model**: Tools and their possible return results are **pre-defined** before synthesis. During task solving, the **LLM simulates** tool execution (no real API calls). This enables flexible, safe, and scalable data generation without external services.

**RL training support**: The generated data can also be used for **reinforcement learning**. Evaluation rubrics are provided for each task, enabling reward modeling and RLHF-style training.

**For RL training**, you can refer to [this repository](https://github.com/haruhi-sudo/data_synth_and_rl)

This project is based on the LangGraph framework and supports end-to-end pipeline: synthesize virtual tasks → solve with mock tools/users → evaluate with rubrics & filter PASS data → distill on filtered data.

## Virtual Task Synthesis Pipeline

The pipeline synthesizes diverse virtual tool-use tasks from persona seeds, solves them with LLM-simulated tools and users, evaluates solutions against rubrics, filters to PASS-only data, and optionally runs SFT distillation. Rubrics evaluation and distillation are separate steps so you can inspect filtered data before training.




## Project Structure

```plain
.
├── configs/
│   ├── agentkd_data_gen.json       # Virtual task synthesis config
│   ├── agentkd_solve_task.json     # Virtual task solving config
│   ├── agentkd_rubrics_filter.json # Rubrics evaluation + filter PASS data
│   └── agentkd_distill.json        # SFT distillation training config
├── data/
│   ├── persona_5K.jsonl         # Persona seeds for virtual task synthesis
│   ├── virtual_tool_use_tasks.jsonl  # Synthesized virtual tasks
│   └── solve_output/            # Per-task solution outputs
├── easydistill/agentkd
│   ├── data_gen.py              # Virtual task synthesis (persona → tools + policy + test cases)
│   ├── solve_task.py            # Virtual task solving (mock tools + mock user)
│   ├── rubrics.py               # Rubrics evaluation + filter PASS → training data
│   └── train.py                 # Distillation training script
└── easydistill/agentkd/infer_utils/
    ├── graph/
    │   ├── virtual_tools.py     # LangGraph: toolset_gen → policy_task → final_task
    │   └── solve_task.py        # LangGraph: reason_and_act ⇄ mock_tools / mock_user
    └── functions/
        ├── call_llms.py         # LLM API calls
        ├── tool_set_policy_gen.py
        ├── policy_task.py
        ├── refine_policy_task.py
        ├── solve_task_fn.py
        ├── mock_tools.py
        └── mock_user.py
```

---

## Step 1: Data Generation (`agentkd_data_gen`)

**Input**: Persona seeds in JSONL format. Each entry:

```json
{
    "id": "uuid",
    "persona": "A passionate fan of Afrikaans music and die-hard supporter of Spoegwolf"
}
```

**Output**: `data/virtual_tool_use_tasks.jsonl` — synthesized tasks with tools, policy, and test cases.

```bash
easydistill --config configs/agentkd_data_gen.json
```

Configure `configs/agentkd_data_gen.json`:

- `paths.data_file`: Path to persona JSONL
- `logging.task_file_path`: Output path for virtual tasks
- `step_models`: ToolSetGenAgent, PolicyTaskAgent, FinalTaskAgent
- `api_configs`: API endpoints for each model (use env vars for keys)

---

## Step 2: Task Solving (`agentkd_solve_task`)

**Input**: Virtual tasks from Step 1.

**Output**: Per-task solution folders under `data/solve_output/{task_id}/` containing `solution*.json`, `more_info.json`, `tool_call_history.json`.

```bash
easydistill --config configs/agentkd_solve_task.json
```

Configure `configs/agentkd_solve_task.json`:

- `paths.data_file`: Path to virtual tasks JSONL (output of Step 1)
- `logging.solve_path`: Output directory for solutions
- `step_models`: SolveAgent, MockToolAgent

---

## Step 3a: Rubrics Evaluation and Filter (`agentkd_rubrics_filter`)

**Input**: Solution folders from Step 2.

**Output**: `rubrics_output.json` per task, filtered training data (PASS solutions only).

```bash
easydistill --config configs/agentkd_rubrics_filter.json
```

Configure `configs/agentkd_rubrics_filter.json`:

- `inference.paths.solution_path`: Path to solve output directory
- `dataset.labeled_path`: Output path for filtered training data
- `inference.step_models.RubricsAgent`: Model for evaluation

This step evaluates each solution against rubrics (PASS/FAIL), selects the best PASS solution per task, and converts them to the `conversations` format for training.

---

## Step 3b: Distillation Training (`agentkd_distill`)

**Input**: Filtered training data from Step 3a (`dataset.labeled_path`).

**Output**: Trained model in `training.output.output_dir`.

```bash
easydistill --config configs/agentkd_distill.json
```

Configure `configs/agentkd_distill.json`:

- `dataset.labeled_path`: Path to filtered training data (output of Step 3a)
- `models.student`: Student model to train
- `training`: SFT hyperparameters (output_dir, num_train_epochs, learning_rate, etc.)

This step runs SFT distillation on the PASS-filtered tool-use trajectories.

---

## Training Data Format

The pipeline produces data in the `conversations` format for SFT:

```json
[
    {
        "conversations": [
            { "from": "system", "value": "..." },
            { "from": "human", "value": "..." },
            { "from": "gpt", "value": "..." },
            ...
        ]
    }
]
```

---

## Generated Data Schema (`virtual_tool_use_tasks.jsonl`)

Each line in the output JSONL is a virtual task with the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique task ID (UUID). |
| `checked_tools` | List of pre-defined tools. Each tool has `name`, `description`, `parameters` (JSON schema), and `outputs` (JSON schema for possible return values). |
| `policy` | Policy text that defines authorized actions, constraints, refusal conditions, and transfer-to-human rules. |
| `tasks_and_backgrounds` | List of test cases. Each item contains: |
| ↳ `task` | User-facing task description. |
| ↳ `tool_return_expected` | Expected tool call sequences. `normal_path`: compliant sequence with `tool_name`, `input`, `expected_output`; `hack_path`: policy-violating sequence (may be empty). |
| ↳ `user_background` | Background for the mock user; parameters are revealed only when the agent asks. |
| ↳ `hack_success_user_background` | Optional. User justification for the hack path (e.g., to test refusal). |
| ↳ `additional_parameters` | Summary of which parameters are available and when to reveal them. |
| ↳ `test_policy` | What policy or precondition this test case validates. |
| ↳ `user_escape_strategy` | How the user might try to evade policy (for refusal tests). |
| ↳ `evaluation` | Rubric for judging whether the agent completed the task correctly. |

---

## Dependencies

In addition to the base easydistill dependencies, the virtual task pipeline requires:

- `json5` — for lenient JSON parsing in `refine_policy_task.py`
- `langgraph`, `langchain_core` — for graph-based agents
