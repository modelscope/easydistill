# 📌 MMKD-RL-Grounding 运行教程

本项目基于 **GRPO（Group Relative Policy Optimization）** 强化学习框架，用于 **Qwen2.5-VL-Instruct** 视觉语言模型在， **单目标检测框定位任务（Multi-box Grounding** 和 **多目标检测框定位任务（Multi-box Grounding）** 上的优化。



## 📁 项目结构

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


## 🚀 快速开始

### 1. 准备数据

- 数据集格式支持 `.json`
- 检测框格式 $[x_{min},y_{min},x_{max},y_{max}]$
- 每条数据格式如下：

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
>⚠️ **对于仅需要单框检测的场景，请使用以下格式的json，并被配置单框检测的奖励函数。**

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

```yaml
datasets:
  - json_path: "data/multi_box_grounding_RL_sample.json"
    sampling_strategy: "all" # or "first:100", "random:50%", "end:10" 
```

---

### 2. 配置训练参数

编辑配置文件 `configs/mmkd_rl_grounding.json`：

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
### 3. 启动训练

使用 `easydistill` 命令启动训练：

```bash
easydistill --config configs/mmkd_rl_grounding.json
```
---

## 🎯 奖励函数说明

| 奖励函数名                     | 适用场景 | 作用说明          |
| ------------------------- | ---- | ------------- |
| `multi_box_format_reward` | 多框检测 | 校验输出格式是否为合法列表 |
| `multi_gaussian_point`    | 多框检测 | 基于中心点的高斯奖励    |
| `multi_gaussian_plane`    | 多框检测 | 基于整个框的高斯奖励    |
| `format`                  | 单框检测 | 校验输出格式是否为合法列表 |
| `gaussian_point`          | 单框检测 | 基于中心点的高斯奖励    |
| `gaussian_plane`          | 单框检测 | 基于整个框的高斯奖励    |

>⚠️ **请确保模型大部分回答的输出格式接近数据集中的检测框格式**
>>请先以相同格式微调模型，如`data/multi_box_grounding_SFT_sample.json`中的示例，或在Prompt中要求模型输出相应检测框格式

奖励函数介绍

#### 1. 高斯点奖励（Gaussian Point Reward）
- **作用**：衡量预测中心与目标元素中心的精确对齐程度，鼓励精确定位。
- **公式**：
  $$R_{\text{point}} = \exp\left(-\frac{1}{2}\left(\frac{(c_x^p - c_x^{gt})^2}{\sigma_x^{gt^2}} + \frac{(c_y^p - c_y^{gt})^2}{\sigma_y^{gt^2}}\right)\right)$$
  - $$(c_x^p, c_y^p)$$：预测框中心坐标
  - $(c_x^{gt}, c_y^{gt}) $：真实框中心坐标
  - $\sigma_x^{gt}, \sigma_y^{gt} $：真实框高斯分布在x/y方向的标准差

#### 2. 高斯覆盖奖励（Gaussian Coverage Reward）
- **作用**：评估预测高斯分布与真实高斯分布的空间重叠程度，确保区域覆盖。
- **公式**（基于Bhattacharyya系数）：
  $$R_{\text{coverage}} = \exp\left(-\frac{1}{8}(\mu_p - \mu_{gt})^T \Sigma^{-1} (\mu_p - \mu_{gt}) - \frac{1}{2} \ln\frac{|\Sigma|}{\sqrt{|\Sigma_p||\Sigma_{gt}|}}\right)$$
  - $\mu_p, \mu_{gt} $：预测/真实分布的均值向量
  - $\Sigma_p, \Sigma_{gt} $：预测/真实分布的协方差矩阵
  - $\Sigma = \frac{\Sigma_p + \Sigma_{gt}}{2} $：平均协方差矩阵

#### 3. 格式奖励（Format Reward）(可选)

- **作用**：确保模型输出的坐标格式严格符合 `[x1,y1,x2,y2]` 的四数值格式，避免因格式错误导致任务失败。
- **公式**（二元奖励）：
- $$R_{\text{format}} = 1,  \text{若输出符合检测框格式} [x1,y1,x2,y2] ，\text{否则} R_{\text{format}} = 0$$



## 🧠 参考论文

- **论文**： GUI-G2: GAUSSIAN REWARD MODELING FOR GUI GROUNDING
- **链接**： https://arxiv.org/pdf/2507.15846
