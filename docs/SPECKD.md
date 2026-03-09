# Speckd Usage Tutorial

This project is based on the EAGLE-3 framework for distillation training, enabling fast decoding of Large Language Models (LLMs) with provable performance maintenance.

## Project Structure

```plain
.
├── configs/
│   └── speckd_local.json               # Main configuration file
├── data/
│   └── speckd_demo.jsonl               # Raw data source
│   └── speckd_demo_labeled.jsonl       # Training data
│   └── speckd_demo_labeled_test.jsonl  # Test data
├── easydistill/speckd
│   ├── infer.py
│   ├── train.py                        # Distillation training script
│   ├── infer_utils/
│   └── train_utils/
```

## Quick Start

### Data Preparation

The dataset format supports `.jsonl`. We have provided sample data in `data/speckd_demo.jsonl`. Each data entry has the following format:

```json
{
    "instruction": "Three friends decide to split the cost of a $60 meal evenly. However, they later realize that they received a 20% discount on their bill. What should each friend pay if they want to split the discounted bill evenly?"
}
```

### Configure Required Parameters

`configs/speckd_local.json` contains all the parameter configuration information required for generating speckd trajectories and distillation training:

```json
{
    "job_type": "speckd_local",
    "dataset": {
        "instruction_path": "data/speckd_demo.jsonl",
        "labeled_progress_dir": "data/speckd_infer_progress",
        "labeled_path_raw": "data/speckd_demo_labeled_raw.jsonl",
        "labeled_path": "data/speckd_demo_labeled.jsonl"
    },
    "models": {
        "teacher": "Qwen/Qwen2.5-72B-Instruct",
        "student": "Qwen/Qwen2.5-7B-Instruct"
    },
    "inference": {
        "batch_size": 32,
        "vllm_config": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": true,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "trust_remote_code": true
        },
        "sampling_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048
        }
    },
    "training": {
        ...
    }
}
```

### Generate Model Output Data

Based on the vllm framework, generate trajectory data and automatically proceed to the next training step using the `easydistill` command:

```bash
easydistill --config configs/speckd_local.json
```

This will generate model outputs based on the source data `data/speckd_demo.jsonl`, perform format conversion, and ultimately produce data that can be directly used for training. Example data format can be found in `data/speckd_demo_labeled.jsonl`.

You can modify the inference parameters in `configs/speckd_local.json` as needed.

### Model Training

Based on the EAGLE-3 framework, use the generated model outputs (`data/speckd_demo_labeled.jsonl`) to start training using the `easydistill` command:

```bash
easydistill --config configs/speckd_local.json
```

You can modify the training parameters in `configs/speckd_local.json` as needed.