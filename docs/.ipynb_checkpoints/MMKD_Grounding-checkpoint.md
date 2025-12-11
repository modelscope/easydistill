# 📌 MMKD-RL-Grounding 运行教程  MMKD-RL-Grounding Usage Tutorial

本项目基于 **GRPO（Group Relative Policy Optimization）** 强化学习框架，用于 **Qwen2.5-VL-Instruct** 视觉语言模型在，**单目标检测框定位任务（Single-box Grounding）** 和 **多目标检测框定位任务（Multi-box Grounding）** 上的优化。  
This project is based on the **GRPO (Group Relative Policy Optimization)** reinforcement learning framework and is designed to optimize the **Qwen2.5-VL-Instruct** vision-language model for **single-box grounding** and **multi-box grounding** tasks.

---

## 📁 项目结构  Project Structure

```
.
├── result/                         # 训练输出目录
├── configs/
│   └── mmkd_rl_grounding.json      # 主配置文件
├── data/
│   └── multi_box_grounding.yaml    # 数据集配置
├── easydistill/mmkd
│   └── mmkd_rl_grounding.py        # 主训练脚本
├── docs
│   └── mmkd_grounding.md           # 本文档
```

- `result/`：训练输出目录  
  `result/`: Training output directory  
- `configs/`：包含主要配置文件  
  `configs/`: Contains primary config files  
- `data/`：数据集及其配置文件  
  `data/`: Dataset and configuration files  
- `easydistill/mmkd/`：主要训练脚本  
  `easydistill/mmkd/`: Main training script  
- `docs/`：本教程文档  
  `docs/`: This documentation file  

---

## 🚀 快速开始  Quick Start

### 1. 准备数据  Prepare Data

- 数据集格式支持 `.json`  
  The dataset supports `.json` format  
- 检测框格式 $[x_{min},y_{min},x_{max},y_{max}]$  
  Detection box format $[x_{min},y_{min},x_{max},y_{max}]$  
- 每条数据格式如下：  
  Each data entry format is as follows:

```json
[
    {
        "image_path": "345.jpg",
        "height": 720,
        "width": 1280,
        "instruction": "Draw bounding boxes around product positions and label them with tags 0, 1, 2, 3, then output in JSON format",
        "abs_box": {
            "0": [[456,337,725,557]],
            "1": [],
            "2": [[849,407,961,482],[965,480,1099,550]],
            "3": []
        }
    },
    ...
]
```

> ⚠️ **对于仅需要单框检测的场景，请使用以下格式的json，并被配置单框检测的奖励函数。**  
> ⚠️ **For single-box detection, please use the following format and configure single-box reward functions.**

```json
[
    {
        "image_path": "APP_20250815_103045.png",
        "height": 1080,
        "width": 1920,
        "instruction": "请帮我找出包含“立即领取”按钮的区域，输出其坐标，格式为 [x_min, y_min, x_max, y_max]。",
        "bbox": [850, 720, 1070, 780]
    }
    ...
]
```

- 数据集配置文件 `data/multi_box_grounding.yaml` 示例：  
  Example configuration file `data/multi_box_grounding.yaml`:

```yaml
datasets:
  - json_path: "data/multi_box_grounding_RL_sample.json"
    sampling_strategy: "all" # or "first:100", "random:50%", "end:10" 
```

---

### 2. 配置训练参数  Configure Training Parameters

编辑配置文件 `configs/mmkd_rl_grounding.json`：  
Edit the configuration file `configs/mmkd_rl_grounding.json`:

```json
{
  "job_type": "mmkd_rl_grounding",
  "dataset": {
    "labeled_path": "data/multi_box_grounding.yaml"
  },
  "models": {
    "student": "Qwen/Qwen2.5-VL-3B-Instruct"
  },
  "training": {
    "reward_funcs": [
      "multi_box_format_reward",
      "multi_gaussian_point",
      "multi_gaussian_plane_reward"
    ],
    "deepspeed": "configs/accelerate_config/stage3.json",
    "output_dir": "./result/",
    "max_length": 4096,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "roll_out_num": 8,
    "save_steps": 400,
    "logging_steps": 4,
    "learning_rate": 1e-6,
    "max_pixels": 12845056
  }
}
```

---

### 3. 启动训练  Launch Training

使用 `easydistill` 命令启动训练：  
Start training using the `easydistill` command:

```bash
easydistill --config configs/mmkd_rl_grounding.json
```

---

## 🎯 奖励函数说明  Reward Function Explanation

| 奖励函数名                     | 适用场景 | 作用说明          |
| ------------------------- | ---- | ------------- |
| `multi_box_format_reward` | 多框检测 | 校验输出格式是否为合法列表 |
| `multi_gaussian_point`    | 多框检测 | 基于中心点的高斯奖励    |
| `multi_gaussian_plane`    | 多框检测 | 基于整个框的高斯奖励    |
| `format`                  | 单框检测 | 校验输出格式是否为合法列表 |
| `gaussian_point`          | 单框检测 | 基于中心点的高斯奖励    |
| `gaussian_plane`          | 单框检测 | 基于整个框的高斯奖励    |

| Reward Function            | Scenario      | Description                                   |
| ----------------------     | ------------ | --------------------------------------------- |
| `multi_box_format_reward`  | Multi-box     | Checks if output format is a valid list       |
| `multi_gaussian_point`     | Multi-box     | Gaussian reward based on center points        |
| `multi_gaussian_plane`     | Multi-box     | Gaussian reward based on whole box            |
| `format`                   | Single-box    | Validate output format is [x1, y1, x2, y2]    |
| `gaussian_point`           | Single-box    | Gaussian reward based on box center           |
| `gaussian_plane`           | Single-box    | Gaussian reward based on box coverage         |

> ⚠️ **请确保模型大部分回答的输出格式接近数据集中的检测框格式**  
> ⚠️ **Please ensure your model output format matches the dataset bounding box format most of the time.**  
>> 请先以相同格式微调模型，如 `data/multi_box_grounding_SFT_sample.json` 中的示例，或在 Prompt 中要求模型输出相应检测框格式  
>> First, finetune the model to output the same format as in `data/multi_box_grounding_SFT_sample.json`, or require the format in the prompt.

---

奖励函数介绍  
Reward Function Details

#### 1. 高斯点奖励（Gaussian Point Reward）

- **作用**：衡量预测中心与目标元素中心的精确对齐程度，鼓励精确定位。  
  **Purpose**: Measures the degree of alignment between predicted box center and ground truth center to encourage precise localization.
- **公式**：  
  **Formula**:  
  $$R_{\text{point}} = \exp\left(-\frac{1}{2}\left(\frac{(c_x^p - c_x^{gt})^2}{\sigma_x^{gt^2}} + \frac{(c_y^p - c_y^{gt})^2}{\sigma_y^{gt^2}}\right)\right)$$

  - $$(c_x^p, c_y^p)$$：预测框中心坐标  
    $$(c_x^p, c_y^p)$$: Predicted box center coordinates  
  - $$(c_x^{gt}, c_y^{gt})$$：真实框中心坐标  
    $$(c_x^{gt}, c_y^{gt})$$: Ground truth box center coordinates  
  - $$\sigma_x^{gt}, \sigma_y^{gt}$$：真实框高斯分布在x/y方向的标准差  
    $$\sigma_x^{gt}, \sigma_y^{gt}$$：Standard deviation in x/y direction for the ground truth box

---

#### 2. 高斯覆盖奖励（Gaussian Coverage Reward）

- **作用**：评估预测高斯分布与真实高斯分布的空间重叠程度，确保区域覆盖。  
  **Purpose**: Measures spatial overlap between predicted and ground-truth Gaussian distributions to ensure coverage.
- **公式**（基于Bhattacharyya系数）：  
  **Formula** (based on Bhattacharyya coefficient):  
  $$R_{\text{coverage}} = \exp\left(-\frac{1}{8}(\mu_p - \mu_{gt})^T \Sigma^{-1} (\mu_p - \mu_{gt}) - \frac{1}{2} \ln\frac{|\Sigma|}{\sqrt{|\Sigma_p||\Sigma_{gt}|}}\right)$$
  
  - $$\mu_p, \mu_{gt}$$：预测/真实分布的均值向量  
    $$\mu_p, \mu_{gt}$$: Mean vector of predicted / ground-truth distribution  
  - $$\Sigma_p, \Sigma_{gt}$$：预测/真实分布的协方差矩阵  
    $$\Sigma_p, \Sigma_{gt}$$: Covariance matrix of predicted / ground-truth distribution  
  - $$\Sigma = \frac{\Sigma_p + \Sigma_{gt}}{2}$$：平均协方差矩阵  
    $$\Sigma = \frac{\Sigma_p + \Sigma_{gt}}{2}$$: Average covariance matrix

---

#### 3. 格式奖励（Format Reward）（可选）  
#### 3. Format Reward (Optional)

- **作用**：确保模型输出的坐标格式严格符合 `[x1,y1,x2,y2]` 的四数值格式，避免因格式错误导致任务失败。  
  **Purpose**: Ensures model output strictly matches the four-value `[x1, y1, x2, y2]` box format to avoid failures due to format errors.
- **公式**（二元奖励）：  
  **Formula** (binary reward):  

  $$R_{\text{format}} = 1,  \text{若输出符合检测框格式} [x1,y1,x2,y2] ，\text{否则} R_{\text{format}} = 0$$

---

如需更多细节请参考源代码及相关配置文件。  
For more details, please refer to the source code and related configuration files.

---
