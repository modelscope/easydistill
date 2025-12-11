# AgentKD Usage Tutorial

This project is based on the langgraph framework and is used for Agent reasoning trajectory generation and distillation training of small agent models based on these trajectories.

## Project Structure

```plain
.
├── configs/
│   └── agentkd_local.json      # Main configuration file
├── data/
│   └── agent_demo.jsonl        # Raw data source
│   └── agent_demo_labeled.jsonl # Generated reasoning trajectories
├── easydistill/agentkd
│   └── infer.py                # Agent reasoning trajectory generation
│   └── train.py                # Distillation training script
```

## Quick Start

### Data Preparation

The dataset format supports `.jsonl`. We have provided sample data in `data/agent_demo.jsonl`. Each data entry has the following format:

```json
{
    "id": 0,
    "question": "TLDR",
    "solution": "TLDR",
    "true_answer": "TLDR"
}
```

### Configure Required Parameters

`configs/agentkd_local.json` contains all the parameter configuration information required for generating agent trajectories and distillation training:

```json
{
    "job_type": "agentkd_local",
    "dataset": {
        "instruction_path": "data/agent_demo.jsonl",
        "labeled_path_raw": "data/agent_demo_labeled_raw.jsonl",
        "labeled_path": "data/agent_demo_labeled.json"
    },
    "models": {
        "teacher": "Qwen/Qwen2.5-72B-Instruct",
        "student": "Qwen/Qwen2.5-7B-Instruct"
    },
    "inference": {
        ...
    },
    "training": {
        ...
    }
}
```

### Generate Agent Trajectory Data

Generate Agent Trajectory Data and automatically proceed to training using the `easydistill` command:

```bash
easydistill --config configs/agentkd_local.json
```

This will generate raw Agent reasoning trajectories based on the source data `data/agent_demo.jsonl`, perform format conversion, and ultimately produce trajectory data that can be directly used for training: `data/agent_demo_labeled.json`.

Example data format:

```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "A pool has 48 blue tiles and 32 red tiles. If the pool needs 100 tiles to be completed, how many more tiles are needed?"
            },
            {
                "from": "gpt",
                "value": "<first_thought>Calculate the total number of tiles currently in the pool, then subtract this number from the total number of tiles needed to find out how many more tiles are required.</first_thought>"
            },
            {
                "from": "human",
                "value": "Observation: None"
            },
            {
                "from": "gpt",
                "value": "<thought>I need to calculate the total number of tiles currently in the pool and then find out how many more tiles are needed to reach 100 tiles.</thought><code># Calculate the total number of tiles currently in the pool\nblue_tiles = 48\nred_tiles = 32\ntotal_tiles_current = blue_tiles + red_tiles\n\n# Calculate how many more tiles are needed\ntiles_needed = 100 - total_tiles_current\nprint(tiles_needed)</code>"
            },
            {
                "from": "human",
                "value": "Observation: 20"
            },
            {
                "from": "gpt",
                "value": "<thought>I have already calculated the number of tiles needed to complete the pool, which is 20. Now, I need to format this number as a final answer in LaTeX format.</thought><code>final_answer_print(r\"\\boxed{20}\")</code>"
            }
        ]
    }
]
```

You can modify the inference parameters in `configs/agentkd_local.json` as needed.

### Agent Model Training

Based on the generated Agent reasoning trajectories (`data/agent_demo_labeled.json`), start training using the `easydistill` command:

```bash
easydistill --config configs/agentkd_local.json
```

You can modify the training parameters in `configs/agentkd_local.json` as needed.