# EasyDistill: Easy Knowledge Distillation for Large Language Models

<div align="center">

[中文](./README_zh.md) | [English](./README.md)

</div>

Introducing **EasyDistill**, a pioneering toolkit on knowledge distillation (KD) for large language models (LLMs). With the growing complexity and size of LLMs, **EasyDistill** offers a versatile and user-friendly platform to streamline the KD process, supporting both black-box and white-box methodologies. It facilitates efficient model training, enabling smaller models to emulate the performance of larger ones without compromising accuracy. **EasyDistill** boasts an extensive range of features, including data synthesis, supervised fine-tuning, ranking optimization, and reinforcement learning, all tailored for various KD scenarios. Designed to accommodate both System 1 (fast, intuitive) and System 2 (slow, analytical) cognitive models, the toolkit is modular and easy to use, with a simple command-line interface guiding users. Beyond academic exploration, **EasyDistill** anchors practical industrial solutions, offering robust distilled models and open-source datasets, while also showcasing seamless integration with Alibaba Cloud’s AI platform, PAI. Committed to bridging theoretical advancements with practical needs, **EasyDistill** empowers the NLP community, making state-of-the-art KD strategies accessible to researchers and industry practitioners alike. 


# News

 
- Dec 19th: We have released OmniThoughtV, multi-modal CoTs distilled and filtered from Qwen-VL-max.
- July 28th: We have released the functionalities of knowledge distillation from MLLM (aka MMKD). Refer to [Here](./easydistill/mmkd). Evaluations on the qualities of instruction-following and CoT datasets have been updated. Refer to [Here](./easydistill/eval).
- June 25th: We have released a new series of DistilQWen models named DistilQwen-ThoughtY, togeter with OmniThought-0528 (CoTs distilled from DeepSeek-R1-0528).


# Technical Articles

We have a series of technical articles on the functionalities of EasyDistill.

- [OmniThoughtV：面向多模态深度思考的高质量数据蒸馏](https://mp.weixin.qq.com/s/DoIpiauTxLwR9TtaZwbK5w)
- [基于模型蒸馏的大模型文案生成最佳实践](https://developer.aliyun.com/article/1675249)
- [DistillQwen-ThoughtY：通过变长思维链蒸馏，全面提升模型推理能力！](https://developer.aliyun.com/article/1669748)
- [DistilQwen-ThoughtX：变长思维链推理模型，能力超越DeepSeek蒸馏模型](https://developer.aliyun.com/article/1665220)
- [阿里云人工智能平台 PAI 开源 EasyDistill 框架助力大语言模型轻松瘦身](https://developer.aliyun.com/article/1664823)
- [人工智能平台 PAI DistilQwen2.5-DS3-0324发布：知识蒸馏+快思考=更高效解决推理难题](https://developer.aliyun.com/article/1661734)
- [DistilQwen2.5-R1发布：知识蒸馏助推小模型深度思考](https://developer.aliyun.com/article/1659288)
- [DistilQwen2.5发布：通义千问蒸馏小模型再升级](https://developer.aliyun.com/article/1653842)
- [DistilQwen2：通义千问大模型的知识蒸馏实践](https://developer.aliyun.com/article/1633882)
- [基于多轮课程学习的大语言模型蒸馏算法TAPIR](https://developer.aliyun.com/article/1635146)



## Overview

![EasyDistill Framework](resources/framework.png)

- **Toolkit Features**: EasyDistill provides versatile functionalities, including data synthesis, supervised fine-tuning, logits distillation, ranking optimization, and reinforcement learning techniques tailored for KD scenarios.
- **Compatibility**: It supports both System 1 (fast, intuitive) and System 2 (slow, analytical) models.
- **User-Friendly**: With its modular design and simple command-line interface, EasyDistill makes experimentation and implementation of KD strategies straightforward.
- **Industrial Integration**: Incorporates KD-based solutions and supports integration with platforms such as Alibaba Cloud’s Platform for AI (PAI).


## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/modelscope/easydistill
    cd EasyDistill
    ```

2. Install the required dependencies:
    ```bash
    python setup.py install
    ```

3. Explore the usage of EasyDistill through the command-line interface:
    ```bash
    easydistill --config <config-file-path>
    ```

    The config file expresses the detailed settings of any knowledge distillation jobs that **EasyDistill** supports. A sample of black-box distillation config can be shown below:
    ```json
    {
        "job_type": "kd_black_box_local",
        "dataset": {
            "instruction_path": "train.json",
            "labeled_path": "train_labeled.json",
            "template" : "chat_template/chat_template_kd.jinja",
            "seed": 42
        },
        "inference":{
            "enable_chunked_prefill": true,
            "seed": 777,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.8,
            "trust_remote_code": true,
            "enforce_eager": false,
            "max_model_len": 4096,
            "max_new_tokens": 512
        },
        "models": {
            "teacher": "teacher/Qwen/Qwen2.5-7B-Instruct/",
            "student": "student/Qwen/Qwen2.5-0.5B-Instruct/"
        },
        "training": {
            "output_dir": "./result/",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "save_steps": 1000,
            "max_length": 512,
            "logging_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine"
        }
    }
    ```

## DistilQwen Series

The **DistilQwen** models represent a robust suite of distilled language models derived from the **EasyDistill** toolkit. Designed to capitalize on the principles of knowledge distillation, DistilQwen models offer a significant reduction in model size while maintaining high performance, making them ideal for resource-constrained environments. Whether you're aiming for efficient deployment in industrial scenarios or seeking to explore advanced KD methodologies, **DistilQwen** models are poised to meet diverse application needs with agility and precision.


### What's New: Adaptive Thinking Models

The most recent **DistilQwen** series is **DistilQwen-ThoughtX** and **DistilQwen-ThoughtY**, which exhibits improved reasoning abilities and generates CoTs with more optimal lengths compared to its predecessors. The **DistilQwen-ThoughtX** model series is developed from the innovative **OmniThought** dataset by utilizing the novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which ensure that models receive rich, high-quality training data reflecting optimal CoT output length and difficulty. **DistilQwen-ThoughtY** is further trained based on Qwen3 as student models and DeepSeek-R1-0528 as the teacher model. The performance of **DistilQwen-ThoughtX** and **DistilQwen-ThoughtY** is shown below.


| **Model**                                     | **AIME2024** | **MATH500** | **GPQA-D** | **LCB V2** | **Avg.**  | **Download** |
|-----------------------------------------------|--------------|-------------|------------|------------|-----------|--------------|
| **DistillQwen-ThoughtY-4B**                   | **76.7**     | **95.2**    | **56.1**   | **75.8**   | **76.0**  |[HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-4B) & [MS](https://modelscope.cn/models/PAI/DistillQwen-ThoughtY-4B)|
| OpenThinker-7B                                | 31.3         | 83.0        | 42.4       | 39.9       | 49.1      |              |
| DeepSeek-R1-Distill-Qwen-7B                   | 57.3         | 89.6        | 47.3       | 48.4       | 60.6      |              |
| OpenThinker2-7B                               | 50.0         | 88.4        | 49.3       | 55.6       | 60.8      |              |
| **DistilQwen-ThoughtX-7B**                    | 56.7         | 90.2        | 50.0       | 56.8       | 63.4      |[HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-7B) & [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-7B)|
| **DistillQwen-ThoughtY-8B**                   | **76.7**     | **94.6**    | **62.1**   | **78.1**   | **77.9**  |[HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-8B) & [MS](https://modelscope.cn/models/PAI/DistillQwen-ThoughtY-8B)|
| LIMO-32B                                      | 56.7         | 86.6        | 58.1       | 60.0       | 65.3      |              |
| OpenThinker-32B                               | 66.0         | 90.6        | 61.6       | 68.9       | 71.7      |              |
| DeepSeek-R1-Distill-Qwen-32B                  | 74.7         | 90.0        | 62.4       | 72.3       | 74.8      |              |
| OpenThinker2-32B                              | 76.7         | 90.8        | **64.1**   | 72.5       | 76.0      |              |
| Light-R1-32B                                  | 74.7         | 90.4        | 62.0       | 56.0       | 70.7      |              |
| s1.1-32B                                      | 59.3         | 87.4        | 62.0       | 58.7       | 66.8      |              |
| **DistilQwen-ThoughtX-32B**                   | 80.0         | 92.6        | 64.0       | 73.4       | 77.5      |[HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-32B) & [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-32B)|
| **DistillQwen-ThoughtY-32B**                  | **90.0**     | **95.2**    | 63.6	      | **76.3**   | **81.3**  |[HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-32B) & [MS](https://modelscope.cn/models/PAI/DistillQwen-ThoughtY-32B)|

The **OmniThought** and **OmniThought-0528** datasets are also publicly available. Refer to the Datasets section.

### System 1 Models

**DistilQwen2** is an enhanced version of the Qwen2 models, equipped with improved instruction-following capabilities for various NLP tasks. We employ GPT-4 and Qwen-max as teacher models to generate high-quality responses, with the balance on the task distributions of input instructions. Following SFT, a rank optimization process is performed using the DPO algorithm to enhance alignment between the student models and the teacher models. **DistilQwen2.5** models are trained using a combination of black-box and white-box KD algorithms. We adhere to the same instruction data processing and black-box SFT procedure as employed in the production of **DistilQwen2**. Subsequently, white-box training is applied to refine the students' acquisition of intricate knowledge from the teacher models, specifically utilizing Qwen2.5-72B-Instruct as open-source teacher models. The performance of **DistilQwen2** and **DistilQwen2.5** is shown below.

| **Model**                          | **AlpacaEval 2.0 (length control)** | **MT-Bench** | **MT-Bench (single)** | **IFEval (instruct-loose)** | **IFEval (strict-prompt)** | **Download** |
|------------------------------------|-------------------------------------|--------------|-----------------------|-----------------------------|----------------------------|--------------|
| Qwen2.5-0.5B-Instruct              | 2.46                                | 5.49         | 6.26                  | 42.81                       | 30.31                      |              |
| **DistilQwen2.5-0.5B-Instruct**    | **4.89**                            | **5.78**     | **6.83**              | **52.61**                   | **37.82**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-0.5B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-0.5B-Instruct)|
| Qwen2-1.5B-Instruct                | 5.22                                | 5.85         | 6.45                  | 41.37                       | 28.10                      |              |
| **DistilQwen2-1.5B-Instruct**      | **8.28**                            | **6.42**     | **7.12**              | **49.76**                   | **36.04**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2-1.5B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2-1.5B-Instruct)|
| Qwen2.5-1.5B-Instruct              | 6.69                                | 7.09         | 7.66                  | 55.40                       | 40.11                      |              |
| **DistilQwen2.5-1.5B-Instruct**    | **13.69**                           | **7.35**     | **7.99**              | **61.10**                   | **74.49**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-1.5B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-1.5B-Instruct)|
| Qwen2.5-3B-Instruct                | 17.98                               | 7.92         | 8.40                  | 61.18                       | 74.58                      |              |
| **DistilQwen2.5-3B-Instruct**      | **20.91**                           | **8.37**     | **8.97**              | **67.03**                   | **77.36**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-3B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-3B-Instruct)|
| Qwen2-7B-Instruct                  | 24.33                               | 8.27         | 8.68                  | 66.67                       | 52.31                      |              |
| **DistilQwen2-7B-Instruct**        | **25.35**                           | **8.40**     | **9.03**              | **71.46**                   | **60.26**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2-7B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2-7B-Instruct)|
| Qwen2.5-7B-Instruct                | 31.43                               | 8.52         | 8.83                  | 81.53                       | 72.10                      |              |
| **DistilQwen2.5-7B-Instruct**      | **34.86**                           | **8.76**     | **9.22**              | **83.48**                   | **73.27**                  |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-7B-Instruct) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-7B-Instruct)|


We have released two instruction following datasets to public. Refer to the Datasets section.


### System 2 Models

The **DistilQwen2.5-R1** model series utilizes DeepSeek-R1 as the teacher model. To align the reasoning abilities of smaller distilled models with their intrinsic cognitive capacities, the models are further refined using our CogPO algorithm, which outperforms other training methods. Additionally, we transfer the fast-thinking reasoning capabilities from DeepSeek-V3-0324 to the **DistilQwen2.5-DS3-0324** models. To shorten the reasoning process, the CoT simplification operator are employed to reduce the number of tokens in the training data for **DistilQwen2.5-R1**. Combined with a rewritten dataset comprising DeepSeek-V3-0324's CoT distillation data, we develop the **DistilQwen2.5-DS3-0324** models. The performance of **DistilQwen2.5-R1** and **DistilQwen2.5-DS3-0324** is shown below.

| **Model**                             | **AIME2024** | **MATH-500** | **GPQA Diamond** | **LiveCodeBench V2** | **Download** |
|---------------------------------------|--------------|--------------|------------------|----------------------|--------------|
| Qwen2.5-3B-Instruct                   | 6.67         | 62.6         | 32.83            | 11.35                |              |
| **DistilQwen2.5-DS3-0324-3B**         | **16.67**    | **70.0**     | **34.34**        | **18.00**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-3B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-3B)|
| Qwen2.5-7B-Instruct                   | 10.0         | 73.6         | 33.30            | 30.72                |              |
| **DistilQwen2.5-7B-R1**               | **23.33**    | **77.8**     | **37.88**        | **36.40**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-7B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-7B)|
| **DistilQwen2.5-DS3-0324-7B**         | **43.33**    | **88.4**     | **42.93**        | **46.38**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-7B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-7B)|
| Qwen2.5-14B-Instruct                  | 16.7         | 78.2         | 43.43            | 37.38                |              |
| **DistilQwen2.5-14B-R1**              | **26.67**    | **82.6**     | **45.45**        | **41.49**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-14B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-14B)|
| **DistilQwen2.5-DS3-0324-14B**        | **46.67**    | **90.8**     | **51.52**        | **54.40**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-14B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-14B)|
| Qwen2.5-32B-Instruct                  | 16.67        | 81.4         | 45.50            | 47.36                |              |
| **DistilQwen2.5-32B-R1**              | **46.67**    | **87.0**     | **48.99**        | **55.97**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-32B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-32B)|
| **DistilQwen2.5-DS3-0324-32B**        | **70.00**    | **93.8**     | **62.12**        | **65.95**            |[HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-32B) & [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-32B)|

All the **DistilQwen** models are publicly available in HuggingFace and ModelScope.




## Released Datasets

We have also released several datasets based on the **EasyDistill** framework.

### Instruction Following Datasets

To assist community developers in avoiding catastrophic forgetting when fine-tuning the **DistilQwen** model, we have open-sourced two datasets: **DistilQwen_100K** and **DistilQwen_1M**. These datasets are intended to provide a solid foundation for model fine-tuning, enhancing adaptability to new tasks while retaining performance on previous tasks. Additionally, it can be utilized to improve instruction-following capabilities when fine-tuning other similar large language models. These datasets cover a range of contents, including mathematics, code, knowledge-based Q&A, instruction following, and creative generation, with a total dataset size of 100K and 1M entries. Users can integrate **DistilQwen_100K** and **DistilQwen_1M**, or its subsets, with their own data during model fine-tuning to ensure excellent downstream task performance while maintaining the model's general capabilities, thus preserving its ability to generalize.


### Chain-of-Thought Reasoning Datasets

**OmniThought** is a large-scale dataset featuring **2 million** Chain-of-Thought (CoT) processes generated and validated by DeepSeek-R1 and QwQ-32B. Each CoT process in **OmniThought** is annotated with novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which describe the appropriateness of CoT verbosity and cognitive difficulty level for models to comprehend these reasoning processes. Based on our **OmniThought** dataset, we further train and release a series of high-performing models (**DistilQwen-ThoughtX-7B** and **DistilQwen-ThoughtX-32B**), specifically equipped with stronger reasoning abilities and optimal CoT output length and difficulty level. Refer to `recipes/open_datasets` for details. In addition, **OmniThought-0528** is an extension to **OmniThought** featuring **365 thousand** Chain-of-Thought (CoT) processes generated and validated by DeepSeek-R1-0528. 

All the datasets are publicly available in HuggingFace and ModelScope.

| **Dataset**       | **Size**  | **Download**                                                                                                                  |
|-------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------|
| DistilQwen_100K   | 100K      | [HF](https://huggingface.co/datasets/alibaba-pai/DistilQwen_100k) & [MS](https://modelscope.cn/datasets/PAI/DistilQwen_100k)  |
| DistilQwen_1M     | 1M        | [HF](https://huggingface.co/datasets/alibaba-pai/DistilQwen_1M) & [MS](https://modelscope.cn/datasets/PAI/DistilQwen_1M)      |
| OmniThought       | 2M        | [HF](https://huggingface.co/datasets/alibaba-pai/OmniThought) & [MS](https://modelscope.cn/datasets/PAI/OmniThought)          |
| OmniThought-0528  | 365K      | [HF](https://huggingface.co/datasets/alibaba-pai/OmniThought-0528) & [MS](https://modelscope.cn/datasets/PAI/OmniThought-0528)|


## Reference

We have [an arxiv paper](https://arxiv.org/abs/2505.20888) for you to cite for the EasyDistill library. Below are papers related to our project.

- Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue, Jun Huang. EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models. arXiv preprint
- Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang. Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations. arXiv preprint
- Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang. Training Small Reasoning LLMs with Cognitive Preference Alignment. arXiv preprint
- Chengyu Wang, Junbing Yan, Yuanhao Yue, Jun Huang. DistilQwen2.5: Industrial Practices of Training Distilled Open Lightweight Language Models. **ACL 2025**
- Yuanhao Yue, Chengyu Wang, Jun Huang, Peng Wang. Building a Family of Data Augmentation Models for Low-cost LLM Fine-tuning on the Cloud. **COLING 2025**
- Yuanhao Yue, Chengyu Wang, Jun Huang, Peng Wang. Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning. **EMNLP 2024**


## License

This project is licensed under the [Apache License (Version 2.0)](LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](NOTICE) file for more information.


## Join in the Discussion

We welcome community partners to collaborate and contribute to the development, and welcome to join the DingTalk group: 117440002081 to participate in the discussion.
