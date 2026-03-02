# AgentKD Usage Tutorial

This project is based on the LangGraph framework and is used for virtual tool-use task synthesis, solution generation, and distillation training of small agent models.

## Virtual Task Synthesis Pipeline

The pipeline synthesizes virtual tool-use tasks from persona seeds, solves them with LLM-simulated tools and users, evaluates solutions against rubrics, filters to PASS-only data, and optionally runs SFT distillation. Rubrics evaluation and distillation are separate steps.

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
│   ├── rubrics.py               # Rubrics evaluation + filtered training data
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

This step evaluates each solution against rubrics (PASS/FAIL), selects the best PASS solution per task, and converts them to training format.

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
- `training`: SFT hyperparameters

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

## Dependencies

In addition to the base easydistill dependencies, the virtual task pipeline requires:

- `json5` — for lenient JSON parsing in `refine_policy_task.py`
- `langgraph`, `langchain_core` — for graph-based agents
