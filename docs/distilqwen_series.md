# DistilQwen Series

The DistilQwen family provides distilled language models derived from the EasyDistill toolkit. They keep most of the teacher capability while reducing model size, which makes them suitable for resource-constrained deployment.

## Adaptive thinking models

DistilQwen-ThoughtX and DistilQwen-ThoughtY generate chain-of-thought traces with near-optimal length and difficulty. ThoughtX is trained on the OmniThought dataset with RV/CD curriculum mixing; ThoughtY uses Qwen3 as the student and DeepSeek-R1-0528 as the teacher.

| Model | AIME2024 | MATH500 | GPQA-D | LCB V2 | Avg. | HuggingFace | ModelScope |
|---|---|---|---|---|---|---|---|
| DistilQwen-ThoughtY-4B | 76.7 | 95.2 | 56.1 | 75.8 | 76.0 | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-4B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-4B) |
| OpenThinker-7B | 31.3 | 83.0 | 42.4 | 39.9 | 49.1 | | |
| DeepSeek-R1-Distill-Qwen-7B | 57.3 | 89.6 | 47.3 | 48.4 | 60.6 | | |
| OpenThinker2-7B | 50.0 | 88.4 | 49.3 | 55.6 | 60.8 | | |
| DistilQwen-ThoughtX-7B | 56.7 | 90.2 | 50.0 | 56.8 | 63.4 | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-7B) | [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-7B) |
| DistilQwen-ThoughtY-8B | 76.7 | 94.6 | 62.1 | 78.1 | 77.9 | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-8B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-8B) |
| LIMO-32B | 56.7 | 86.6 | 58.1 | 60.0 | 65.3 | | |
| OpenThinker-32B | 66.0 | 90.6 | 61.6 | 68.9 | 71.7 | | |
| DeepSeek-R1-Distill-Qwen-32B | 74.7 | 90.0 | 62.4 | 72.3 | 74.8 | | |
| OpenThinker2-32B | 76.7 | 90.8 | 64.1 | 72.5 | 76.0 | | |
| Light-R1-32B | 74.7 | 90.4 | 62.0 | 56.0 | 70.7 | | |
| s1.1-32B | 59.3 | 87.4 | 62.0 | 58.7 | 66.8 | | |
| DistilQwen-ThoughtX-32B | 80.0 | 92.6 | 64.0 | 73.4 | 77.5 | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtX-32B) | [MS](https://modelscope.cn/models/pai/DistilQwen-ThoughtX-32B) |
| DistilQwen-ThoughtY-32B | 90.0 | 95.2 | 63.6 | 76.3 | 81.3 | [HF](https://huggingface.co/alibaba-pai/DistilQwen-ThoughtY-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen-ThoughtY-32B) |

## System 1 models

DistilQwen2 and DistilQwen2.5 are instruction-following models. DistilQwen2 uses GPT-4 and Qwen-max as teachers and applies DPO rank optimization; DistilQwen2.5 combines black-box SFT with white-box knowledge distillation from Qwen2.5-72B-Instruct.

| Model | AlpacaEval 2.0 (LC) | MT-Bench | MT-Bench (single) | IFEval (loose) | IFEval (strict) | HuggingFace | ModelScope |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 2.46 | 5.49 | 6.26 | 42.81 | 30.31 | | |
| DistilQwen2.5-0.5B-Instruct | 4.89 | 5.78 | 6.83 | 52.61 | 37.82 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-0.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-0.5B-Instruct) |
| Qwen2-1.5B-Instruct | 5.22 | 5.85 | 6.45 | 41.37 | 28.10 | | |
| DistilQwen2-1.5B-Instruct | 8.28 | 6.42 | 7.12 | 49.76 | 36.04 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2-1.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2-1.5B-Instruct) |
| Qwen2.5-1.5B-Instruct | 6.69 | 7.09 | 7.66 | 55.40 | 40.11 | | |
| DistilQwen2.5-1.5B-Instruct | 13.69 | 7.35 | 7.99 | 61.10 | 74.49 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-1.5B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-1.5B-Instruct) |
| Qwen2.5-3B-Instruct | 17.98 | 7.92 | 8.40 | 61.18 | 74.58 | | |
| DistilQwen2.5-3B-Instruct | 20.91 | 8.37 | 8.97 | 67.03 | 77.36 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-3B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-3B-Instruct) |
| Qwen2-7B-Instruct | 24.33 | 8.27 | 8.68 | 66.67 | 52.31 | | |
| DistilQwen2-7B-Instruct | 25.35 | 8.40 | 9.03 | 71.46 | 60.26 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2-7B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2-7B-Instruct) |
| Qwen2.5-7B-Instruct | 31.43 | 8.52 | 8.83 | 81.53 | 72.10 | | |
| DistilQwen2.5-7B-Instruct | 34.86 | 8.76 | 9.22 | 83.48 | 73.27 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-7B-Instruct) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-7B-Instruct) |

## System 2 models

DistilQwen2.5-R1 uses DeepSeek-R1 as the teacher and is refined with the CogPO algorithm. DistilQwen2.5-DS3-0324 transfers fast-thinking reasoning from DeepSeek-V3-0324.

| Model | AIME2024 | MATH-500 | GPQA Diamond | LiveCodeBench V2 | HuggingFace | ModelScope |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 6.67 | 62.6 | 32.83 | 11.35 | | |
| DistilQwen2.5-DS3-0324-3B | 16.67 | 70.0 | 34.34 | 18.00 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-3B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-3B) |
| Qwen2.5-7B-Instruct | 10.0 | 73.6 | 33.30 | 30.72 | | |
| DistilQwen2.5-7B-R1 | 23.33 | 77.8 | 37.88 | 36.40 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-7B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-7B) |
| DistilQwen2.5-DS3-0324-7B | 43.33 | 88.4 | 42.93 | 46.38 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-7B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-7B) |
| Qwen2.5-14B-Instruct | 16.7 | 78.2 | 43.43 | 37.38 | | |
| DistilQwen2.5-14B-R1 | 26.67 | 82.6 | 45.45 | 41.49 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-14B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-14B) |
| DistilQwen2.5-DS3-0324-14B | 46.67 | 90.8 | 51.52 | 54.40 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-14B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-14B) |
| Qwen2.5-32B-Instruct | 16.67 | 81.4 | 45.50 | 47.36 | | |
| DistilQwen2.5-32B-R1 | 46.67 | 87.0 | 48.99 | 55.97 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-R1-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-R1-32B) |
| DistilQwen2.5-DS3-0324-32B | 70.00 | 93.8 | 62.12 | 65.95 | [HF](https://huggingface.co/alibaba-pai/DistilQwen2.5-DS3-0324-32B) | [MS](https://modelscope.cn/models/PAI/DistilQwen2.5-DS3-0324-32B) |
