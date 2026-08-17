# EasyDistill 2 Model Zoo

本页面汇总开源的 **DistilQwen** 模型家族以及上游 [EasyDistill](https://github.com/modelscope/easydistill) 项目发布的公开数据集。所有模型均同时托管在 [HuggingFace](https://huggingface.co/alibaba-pai) 与 [ModelScope](https://modelscope.cn/organization/PAI)。

## 模型总览

| 模型 | 家族 | 规模 | 类型 | HuggingFace | ModelScope |
|---|---|---|---|---|---|
| AgenticQwen-30B-A3B | AgenticQwen | 30B | agentic | [HF](https://huggingface.co/alibaba-pai/AgenticQwen-30B-A3B) | [MS](https://modelscope.cn/models/PAI/AgenticQwen-30B-A3B) |
| AgenticQwen-8B | AgenticQwen | 8B | agentic | [HF](https://huggingface.co/alibaba-pai/AgenticQwen-8B) | [MS](https://modelscope.cn/models/PAI/AgenticQwen-8B) |
| DistilQwen-ThoughtX-32B | DistilQwen-ThoughtX | 32B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-32B) | [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-32B) |
| DistilQwen-ThoughtX-7B | DistilQwen-ThoughtX | 7B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-7B) | [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-7B) |
| DistilQwen2-1.5B-Instruct | DistilQwen2 | 1.5B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2-1.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2-1.5B-Instruct) |
| DistilQwen2-7B-Instruct | DistilQwen2 | 7B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2-7B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2-7B-Instruct) |
| DistilQwen2.5-0.5B-Instruct | DistilQwen2.5 | 0.5B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-0.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-0.5B-Instruct) |
| DistilQwen2.5-1.5B-Instruct | DistilQwen2.5 | 1.5B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-1.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-1.5B-Instruct) |
| DistilQwen2.5-3B-Instruct | DistilQwen2.5 | 3B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-3B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-3B-Instruct) |
| DistilQwen2.5-7B-Instruct | DistilQwen2.5 | 7B | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-7B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-7B-Instruct) |
| DistilQwen2.5-DS3-0324-14B | DistilQwen2.5-DS3-0324 | 14B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-14B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-14B) |
| DistilQwen2.5-DS3-0324-32B | DistilQwen2.5-DS3-0324 | 32B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-32B) |
| DistilQwen2.5-DS3-0324-3B | DistilQwen2.5-DS3-0324 | 3B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-3B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-3B) |
| DistilQwen2.5-DS3-0324-7B | DistilQwen2.5-DS3-0324 | 7B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-7B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-7B) |
| DistilQwen2.5-14B-R1 | DistilQwen2.5-R1 | 14B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-14B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-14B) |
| DistilQwen2.5-32B-R1 | DistilQwen2.5-R1 | 32B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-32B) |
| DistilQwen2.5-7B-R1 | DistilQwen2.5-R1 | 7B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-7B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-7B) |
| DistilQwen-ThoughtY-32B | DistilQwen-ThoughtY | 32B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-32B) |
| DistilQwen-ThoughtY-4B | DistilQwen-ThoughtY | 4B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-4B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-4B) |
| DistilQwen-ThoughtY-8B | DistilQwen-ThoughtY | 8B | reasoning | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-8B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-8B) |

## 按家族分类

### AgenticQwen

#### AgenticQwen-30B-A3B（30B，agentic）

基于 Qwen3-30B-A3B-Instruct 经多轮强化学习训练的 MoE 智能体语言模型。每次前向传播仅激活 3B 参数，但在 BFCL-V4、TAU-2 等智能体基准上可媲美甚至超越 8B 稠密模型。

**能力：**tool_use、multi_step_reasoning、agentic_planning、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/AgenticQwen-30B-A3B) | [ModelScope](https://modelscope.cn/models/PAI/AgenticQwen-30B-A3B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `agent_distillation`
- 后端: `pai_token`
- 说明: 当需要最强智能体能力，同时希望推理激活参数低于 8B 稠密模型时的推荐选择。

#### AgenticQwen-8B（8B，agentic）

基于 Qwen3-8B 经多轮强化学习训练的小型智能体语言模型。专为多步推理、工具编排与长程智能体任务（如航班预订、账户管理与数据分析）设计。

**能力：**tool_use、multi_step_reasoning、agentic_planning、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/AgenticQwen-8B) | [ModelScope](https://modelscope.cn/models/PAI/AgenticQwen-8B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `agent_distillation`
- 后端: `pai_token`
- 说明: 可作为智能体工具使用蒸馏的紧凑学生或教师模型。在工业智能体场景中具备优良的成本效益比。

### DistilQwen-ThoughtX

#### DistilQwen-ThoughtX-32B（32B，reasoning）

基于 OmniThought 训练的 32B 自适应思考模型。ThoughtX 系列中推理能力最强，支持课程感知的 CoT 生成。

**能力：**chain_of_thought、adaptive_thinking、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-32B) | [ModelScope](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-32B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 用于最高质量的自适应 CoT 蒸馏。

#### DistilQwen-ThoughtX-7B（7B，reasoning）

基于 OmniThought 数据集训练的 7B 自适应思考模型，采用 RV（Reasoning Verbosity）与 CD（Cognitive Difficulty）课程混合。生成的 CoT 在长度与难度上接近最优。

**能力：**chain_of_thought、adaptive_thinking、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-7B) | [ModelScope](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-7B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 当需要通过 RV/CD 混合器控制 CoT 冗长度与难度时的最佳教师模型。

### DistilQwen2

#### DistilQwen2-1.5B-Instruct（1.5B，instruction_following）

在 Qwen2-1.5B-Instruct 基础上蒸馏得到的指令跟随模型。使用 GPT-4 与 Qwen-max 作为教师，并在 SFT 前平衡了种子指令的任务分布。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2-1.5B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2-1.5B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 在推理成本可控的前提下，比 0.5B 家族具备更强指令跟随能力的小型学生模型。

#### DistilQwen2-7B-Instruct（7B，instruction_following）

DistilQwen2 系列中的 7B 指令跟随模型。使用平衡后的指令数据训练，并经过 DPO 排序优化。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2-7B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2-7B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 当 1.5B/3B 变体在目标领域能力不足时，可作为更高质量的教师模型。

### DistilQwen2.5

#### DistilQwen2.5-0.5B-Instruct（0.5B，instruction_following）

从 Qwen2.5-72B-Instruct 蒸馏得到的超轻量指令跟随模型，结合黑盒与白盒知识蒸馏。适用于低延迟教师生成或作为进一步微调的小巧学生模型。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-0.5B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-0.5B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 若通过 OpenAI 兼容推理服务托管该模型，请将 backend.model_id 设为 HuggingFace 或 ModelScope 的模型 ID。

#### DistilQwen2.5-1.5B-Instruct（1.5B，instruction_following）

从 Qwen2.5-72B-Instruct 蒸馏得到的中等规模指令跟随模型。结合黑盒 SFT 与白盒精馏，将复杂教师知识迁移到小型学生模型。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-1.5B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-1.5B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 高性价比的教师模型，适合高吞吐地产出中等质量的指令-回复对。

#### DistilQwen2.5-3B-Instruct（3B，instruction_following）

3B 指令跟随模型，在保持较低部署与运行成本的同时，大幅缩小了与 7B 基线的差距。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-3B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-3B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 教师生成与最终学生部署的强性价比选择。

#### DistilQwen2.5-7B-Instruct（7B，instruction_following）

DistilQwen2.5 系列的旗舰 7B 指令跟随模型。从 Qwen2.5-72B-Instruct 蒸馏并经白盒 KD 精修。

**能力：**instruction_following、chat

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-7B-Instruct) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-7B-Instruct)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_instruct_distill`
- 后端: `pai_token`
- 说明: 当推理预算允许时，推荐用于高质量指令蒸馏。

### DistilQwen2.5-DS3-0324

#### DistilQwen2.5-DS3-0324-14B（14B，reasoning）

从 DeepSeek-V3-0324 蒸馏得到的 14B 快思考推理模型。在强推理能力与可控推理成本之间取得平衡。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-14B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-14B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 当 7B DS3 质量不足而 32B 成本过高时使用。

#### DistilQwen2.5-DS3-0324-32B（32B，reasoning）

从 DeepSeek-V3-0324 蒸馏得到的 32B 快思考推理模型。DS3-0324 系列中质量最高的 System-2 模型。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-32B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-32B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 生成高质量数学/代码 CoT 训练数据的首选。

#### DistilQwen2.5-DS3-0324-3B（3B，reasoning）

从 DeepSeek-V3-0324 迁移快思考推理能力的 3B 模型。基于蒸馏 CoT 数据与缩短 CoT 数据集混合训练。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-3B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-3B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 面向低延迟推理任务的紧凑 System-2 模型。

#### DistilQwen2.5-DS3-0324-7B（7B，reasoning）

从 DeepSeek-V3-0324 蒸馏得到的 7B 快思考推理模型。在多个数学与代码基准上强于 R1-7B 变体。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-7B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-7B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 快思考 CoT 蒸馏的推荐默认教师模型。

### DistilQwen2.5-R1

#### DistilQwen2.5-14B-R1（14B，reasoning）

从 DeepSeek-R1 蒸馏并经 CogPO 精修的 14B 推理模型。推理能力强于 7B 变体，推理成本更高。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-14B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-14B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 当 7B 变体不足以生成数学/代码 CoT 数据时的良好教师模型。

#### DistilQwen2.5-32B-R1（32B，reasoning）

从 DeepSeek-R1 蒸馏得到的 32B 推理模型。在转向 DS3-0324 或 Thought 家族之前， R1 系列中推理质量最佳。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-32B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-32B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 当需要最高 CoT 质量且能承受推理成本时使用。

#### DistilQwen2.5-7B-R1（7B，reasoning）

从 DeepSeek-R1 蒸馏得到的 7B 推理模型。进一步经 CogPO 算法精修，使推理能力与模型内在认知容量对齐。

**能力：**chain_of_thought、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-7B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-7B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 可作为教师模型，为更小的学生模型生成长形式 CoT 轨迹。

### DistilQwen-ThoughtY

#### DistilQwen-ThoughtY-32B（32B，reasoning）

基于 Qwen3、从 DeepSeek-R1-0528 蒸馏得到的 32B 自适应思考模型。ThoughtY 系列中综合推理能力最强。

**能力：**chain_of_thought、adaptive_thinking、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-32B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-32B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 作为高难度推理任务的最高质量自适应思考教师模型。

#### DistilQwen-ThoughtY-4B（4B，reasoning）

基于 Qwen3 的 4B 自适应思考模型，以 DeepSeek-R1-0528 为教师。生成变长 CoT 轨迹，单位 token 推理能力强。

**能力：**chain_of_thought、adaptive_thinking、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-4B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-4B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 资源受限部署场景下的紧凑自适应推理模型。

#### DistilQwen-ThoughtY-8B（8B，reasoning）

基于 Qwen3、从 DeepSeek-R1-0528 蒸馏得到的 8B 自适应思考模型。在中等模型规模下具备强推理能力。

**能力：**chain_of_thought、adaptive_thinking、math、code

**下载：**[HuggingFace](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-8B) | [ModelScope](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-8B)

**在 EasyDistill 2 中的推荐用法：**

- 流水线: `advanced_cot_distill`
- 后端: `pai_token`
- 说明: 当 4B 过小时，推荐的自适应思考默认选择。

## 已发布数据集

| 数据集 | 规模 | 类型 | HuggingFace | ModelScope |
|---|---|---|---|---|
| AgenticQwen-Data | 80K | agentic | [HF](https://huggingface.co/alibaba-pai/AgenticQwen-Data) | [MS](https://modelscope.cn/datasets/PAI/AgenticQwen-Data) |
| DistilQwen_100K | 100K | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen_100k) | [MS](https://modelscope.cn/datasets/PAI/DistilQwen_100k) |
| DistilQwen_1M | 1M | instruction_following | [HF](https://huggingface.co/alibaba-pai/DistilQwen_1M) | [MS](https://modelscope.cn/datasets/PAI/DistilQwen_1M) |
| OmniThought | 2M | reasoning | [HF](https://huggingface.co/alibaba-pai/OmniThought) | [MS](https://modelscope.cn/datasets/PAI/OmniThought) |
| OmniThought-0528 | 365K | reasoning | [HF](https://huggingface.co/alibaba-pai/OmniThought-0528) | [MS](https://modelscope.cn/datasets/PAI/OmniThought-0528) |
| OmniThoughtV_Raw_1.8M | 1.8M | multimodal_reasoning | [HF](https://huggingface.co/alibaba-pai/OmniThoughtV_Raw_1.8M) |  |
| OmniThoughtV_Filter_0.5M | 0.5M | multimodal_reasoning | [HF](https://huggingface.co/alibaba-pai/OmniThoughtV_Filter_0.5M) | [MS](https://modelscope.cn/datasets/platformofai/OmniThoughtV_Filter_0.5M) |

### AgenticQwen-Data

用于训练 AgenticQwen 的双数据飞轮生成的合成智能体强化学习训练数据。包含行为树结构化任务、虚拟工具定义、多轮轨迹以及基于评分规则的质量标注。

### DistilQwen_100K

10 万条指令跟随样本，覆盖数学、代码、知识问答、指令遵循与创意生成。可用于微调 DistilQwen 模型时缓解灾难性遗忘。

### DistilQwen_1M

DistilQwen 指令跟随数据集的 100 万样本版本，覆盖范围更广，适合大规模微调。

### OmniThought

200 万条由 DeepSeek-R1 与 QwQ-32B 生成并验证的思维链推理轨迹。每条轨迹均标注 RV 与 CD 分数，支持课程混合。

### OmniThought-0528

36.5 万条由 DeepSeek-R1-0528 生成并验证的 CoT 推理轨迹。 OmniThought 基于更新教师模型的扩展版本。

### OmniThoughtV_Raw_1.8M

180 万条原始多模态长思维链轨迹，基于 FineVision 种子问题由 Qwen-VL-max 蒸馏得到。轨迹采用 <thinking>/<answer> 格式，图像以 base64 字符串存储。

### OmniThoughtV_Filter_0.5M

OmniThoughtV 经 RV/CD 与正确性评分筛选后的高质量子集，由 EasyDistill MMCoT 流水线构建。用其微调 Qwen3-VL 2B/4B/8B 可同时提升通用视觉理解与推理密集型基准表现。

---

_本页面由 [`easydistill/models/model_zoo.yaml`](../easydistill/models/model_zoo.yaml) 自动生成。如需更新，请修改 YAML 源文件或 generator 脚本，而非直接编辑本 Markdown。_
