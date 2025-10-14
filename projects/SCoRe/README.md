<h1 align="center" style="margin-top: -50px;">FROM CORRECTION TO MASTERY: REINFORCED DISTILLATION OF LARGE LANGUAGE MODEL AGENTS</h1>

## 📑 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Cold-Start BC init](#cold-start-bc-init)
  - [Mentored Problem Solving and SCoRe-SFT](#mentored-problem-solving-and-score-sft)
  - [SCoRe-RL](#score-rl)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview
SCoRe (**S**tudent-**C**entered **o**ne-step **Re**inforcement) is a new training paradigm for LLM agents that replaces full imitation with **ability‑matched correction**.

**Key ideas:**

1. **Ability‑matched correction & deficiency localization** – The teacher minimally intervenes by correcting only the first critical mistake, producing trajectories that match the student’s ability and clearly exposing “correct prefix + key step” failure points.  
2. **Efficient RL optimization** – Short‑horizon rollouts starting from the correct prefix combined with key‑step rewards yield more stable training and encourage genuine problem‑solving beyond imitation.  



## Quick Start

To reproduce SCoRe training, follow three phases:

### Cold-Start BC init
Train the student on a small set of high-quality teacher trajectories to bootstrap reasoning–action skills.

1. **Environment Setup**
```bash
python -m venv mps_env
source mps_env/bin/activate

# For this cold‑start phase, the key packages are `llamafactory` and `langgraph`.
# The provided `requirements.txt` contains all dependencies of our environment.
# Normally, you can selectively install just the essentials.
pip install llamafactory
pip install langgraph
```

2. **BC Data Preparation**
```bash
cd MPS/
Modify .env to fit your needs.
Modity configs/BC_data_gen_config.yaml to fit your needs.
python BC_init_data_gen.py
```

The `.env` file should contain the following lines:  
```bash
QWEN25_72B_API_ADDRESS=.
QWEN25_72B_API_KEY=.
SEARCH_API_KEY=. # Not necessary if you are constructing the math dataset.
```

```
Examples of generated data: MPS/output/example/math_bc_init_trajectories.jsonl
```

### 🔑 Getting Qwen25‑72B API Address & Key  
You can set `QWEN25_72B_API_ADDRESS` and `QWEN25_72B_API_KEY` in `.env` either by **using the official API service**—sign up on the [Qwen platform](https://modelscope.cn/) (or your provider), copy the endpoint URL to `QWEN25_72B_API_ADDRESS` and the generated token to `QWEN25_72B_API_KEY`—or by **deploying locally with `llamafactory`**, running `llamafactory-cli api --model_name_or_path Qwen/Qwen2.5-72B` to get a local endpoint (e.g., `http://127.0.0.1:8000/v1`) and setting the key as `EMPTY` unless you enabled authentication.  

SEARCH_API_KEY is for the Search service at https://idealab.alibaba-inc.com/, which may be inaccessible outside Alibaba; for external web search, modify search_tools.py to use another API.

3. **BC init Training**

```bash
# convert to llamafactory format
python format/BC_init_llamafatory_format_convert.py
cd ../SFT/
Modify configs/train/BC_init.yaml and data/dataset_info.json to fit your needs.
llamafactory-cli train configs/train/BC_init.yaml
```


### Mentored Problem Solving and SCoRe-SFT
Let the initialized student try new tasks; teacher corrects the first wrong step, student resumes. Keep corrected trajectories for next SFT round.

1. **Deploy the student**:
An example configuration file for deploying the SCoRe student model:
```yaml
model_name_or_path: ../SFT/checkpoints/SCoRe/math/BC_init_qwen7b/
template: qwen
infer_backend: vllm  # choices: [huggingface, vllm, sglang]
trust_remote_code: true
skip_special_tokens: false

vllm_config:
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  dtype: bfloat16
  max_model_len: 8192

temperature: 0.6
top_p: 0.95
top_k: 20
repetition_penalty: 1.1
max_new_tokens: 4096
do_sample: true
```
Run the student model with ```llamafactory-cli api config.yaml```, and generate student-centerd, teacher-intervened trajectories:

```bash
cd MPS/
Modify configs/SCoRe_data_gen_config.yaml to fit your needs.
python SCoRe_data_gen.py
```

Train the student again:
```bash
# convert to llamafactory format
python format/SCoRe_SFT_llamafactory_format_convert.py
cd ../SFT/
Modify configs/train/SCoRe_SFT.yaml and data/dataset_info.json to fit your needs.
llamafactory-cli train configs/train/SCoRe_SFT.yaml
```

### SCoRe-RL
Take the part of MPS-generated trajectories, use **short rollouts + key-step rewards** to move from imitation to genuine problem solving.

1. **Environment Setup** Verl and llamafactory dependencies may conflict, so we need to create a new environment
```bash
python -m venv score_rl
source score_rl/bin/activate
```
Follow the **[Verl sglang worker installation guide](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html#installation)** to set up the RL environment.  

2. **Process the dataset**

We build the RL training data from trajectories generated in the MPS phase.
For best results, ensure **SCoRe-RL data does not overlap with SFT data**.
Use format/SCoRe_RL_convert.py to automatically filter out SFT samples.
```bash
cd MPS/
python format/SCoRe_RL_convert.py
```
Then convert the data to verl format.
```bash
cd ../RL/
Modify the config in SCoRe_Script/data_preprocess/convert_parquet.py to fit your needs
python SCoRe_Script/data_preprocess/convert_parquet.py
```

3. **Modify the reward config in .env**

Exact match rewards can be overly sensitive to formatting. We instead use an LLM API to judge semantic consistency between the reference and generated answers. We will consider the Exact match reward support in the future.
```bash
REWARD_API_BASE=.
REWARD_API_KEY=.
```

If you want to train the model to call web search tools, you will need a SEARCH_API_KEY from https://idealab.alibaba-inc.com/, which may be inaccessible outside Alibaba.
For external web search, modify verl/tools/utils/search_r1_like_utils.py and verl/tools/search_tool.py to use an alternative API.

4. **Train**

```bash
bash SCoRe_Script/run_qwen2.5-7b_agent_distill_tool_agent_mlflow.sh
```

## Inference
1. **Deploy the student**:
```bash
cd inference/
Modify configs/SCoRe.yaml to fit your needs
llamafactory-cli api configs/SCoRe.yaml
```

2. **Run the inference code**:
```bash
Modify inference_config.yaml to fit your needs
python inference_api.py
```
If you want to test the web search tool, update inference/tools/search_tools.py with your own API configuration.

## Acknowledgements
We thank [verl](https://github.com/volcengine/verl) and [LLaMA‑Factory](https://github.com/hiyouga/LLaMA-Factory) for providing the excellent training framework. We also thank [ARPO](https://github.com/dongguanting/ARPO) for organizing the evaluation datasets and providing baseline results, and [agent-distillation](https://github.com/Nardien/agent-distillation) for inspiring the prompts used in this project.  

## Contact

For any questions or feedback, please reach out to us at [s1583050085@gmail.com](s1583050085@gmail.com).
