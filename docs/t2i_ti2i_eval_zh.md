# T2I / TI2I 评估（四个独立单文件入口）

单文件版把 T2I（文生图）/ TI2I（图像编辑）评估重写为 **四个完全独立、自包含的单文件模块**，
放在 `easydistill/eval/` 下，风格与该目录其他 evaluator（`cot.py` / `mm.py` 等）一致。
每个文件互相之间零依赖，也不依赖任何其他 T2I/TI2I 代码目录，可单独隔离运行；
运行时只需要 `configs/eval/t2i_ti2i/` 下的两个冻结维度池 JSON 与对应入口 YAML。

## 四个入口

| 入口 | 文件 | 说明 |
|---|---|---|
| T2I 多模型 | `easydistill/eval/t2i_multi_model.py` | 多教师打分 + 冲突检测 + 三步 Debate + 合成 + 总分 |
| T2I 单模型 | `easydistill/eval/t2i_single_model.py` | 指定单教师打分；无跨教师冲突 → 无 Debate，同 schema 产物 |
| TI2I 多模型 | `easydistill/eval/ti2i_multi_model.py` | 同 T2I 多模型，维度池/输入为图像编辑 |
| TI2I 单模型 | `easydistill/eval/ti2i_single_model.py` | 同 T2I 单模型，维度池/输入为图像编辑 |

体系概览：

| | T2I | TI2I |
|---|---|---|
| 维度池 | 60 维 L3（`t2i_dimensions.json`） | 38 维 L3（`ti2i_dimensions.json`） |
| 打分 | 0-4 五档 | 0-4 五档 |
| 调用粒度 | case × 教师 × L1 组 | case × 教师 × L1 组 |
| seed 输入 | `prompt_id` / `prompt` / `image` | `case_id` / `instruction` / `before_image` / `after_image` / `reference_images` |

## 模型分工（多模型入口）

评分、仲裁、合成是**三类角色、两阶段裁决**，Debate 与最终合成为两个独立模型、两个独立步骤：

| 角色 | 模型（当前配置） | 职责 |
|---|---|---|
| 评分教师 | `qwen3.7-plus` / `qwen3.5-plus` / `kimi-k2.6`（PAI Token） | 按 L1 组独立打 L3 维度 0-4 分 + 中文理由 |
| Debate 仲裁（`arbiter`） | `kimi-k3`（PAI EAS） | 仅执行冲突维度的 Step1 初评 → Step2 控辩 → Step3 裁决 |
| 最终合成（`reason_model`） | `kimi-k3`（PAI EAS，独立配置段） | 仅执行多数票维度的中文理由规范化（final 数据合成） |

说明：

- 多模型入口不使用文字识别（OCR/VLM）工具教师，也不做 prompt 关键词激活；所有 L3 维度都由模型教师直接评估。
- `teacher` 字段在所有产物中直接使用实际模型名。
- 单模型入口只需一个教师后端 + 可选 `reason_model`；无 Debate。
- 跨教师同维度分差 ≥ `conflict_threshold`（默认 2）触发 Debate；单 case 仲裁维度数受 `max_debate_dims` 限制（超上限时分差最大的维度优先获得仲裁名额）。
- 所有模型调用自带轻量重试（默认 2 次，覆盖 503/超时/坏 JSON）。

## 快速开始

```bash
# 0) 环境变量：PAI_TOKEN_API_KEY / PAI_TOKEN_BASE_URL / EAS_ENDPOINT_URL / EAS_TOKEN

# 1) T2I 多模型（多教师 Debate）
python -m easydistill.eval.t2i_multi_model \
  --config configs/eval/t2i_ti2i/t2i_multi_model_pai_token.yaml

# 2) T2I 单模型（默认教师 qwen3.7-plus，可 --teacher 覆盖）
python -m easydistill.eval.t2i_single_model \
  --config configs/eval/t2i_ti2i/t2i_single_model_pai_token.yaml --teacher qwen3.7-plus

# 3) TI2I 两个入口同理
python -m easydistill.eval.ti2i_multi_model  --config configs/eval/t2i_ti2i/ti2i_multi_model_pai_token.yaml
python -m easydistill.eval.ti2i_single_model --config configs/eval/t2i_ti2i/ti2i_single_model_pai_token.yaml --teacher qwen3.7-plus
```

每个入口均提供 `*_pai_token.yaml` / `*_pai_eas.yaml` 成对配置（与 `configs/eval/`
其余入口同一约定）；同一配置内各角色也可按需混用后端。

通用参数：`--limit-cases N`（最多评估 N 个 case）、`--synthesize-reasons`（用 reason_model
规范化多数票理由）、`--export-training`（导出 sft/dpo/uncertain 训练数据）。

## 输出产物

每个入口在 `dataset.output_dir` 下输出与原编排器同名的产物：

| 文件 | 说明 |
|---|---|
| `teacher_outputs.jsonl` | 每条教师 × L1 组的原始判断；`teacher` 为实际模型名 |
| `conflict_report.jsonl` | 每 case 冲突维度列表（单模型恒为空） |
| `debate_results.jsonl` | Debate 结果，含每个仲裁维度的 step1/step2/step3 全记录 |
| `final_labels.jsonl` | case 级最终标签：`final_judgments` + `overall_score_100` + `overall` 审计块 |
| `final_judgments.jsonl` | 维度级最终判断，每条含 `case_overall_score_100` |
| `final_labels_summary.json` | 批级 `overall_score_stats` 汇总 |
| `sft_data.jsonl` / `dpo_data.jsonl` / `uncertain_data.jsonl` | `--export-training` 时输出；Debate 改判自动成 DPO 对 |

## 总分计算规则

总分只基于最终 L3 维度裁决结果计算，不额外调用模型：

- **T2I**：0-4 分映射为 0/25/50/75/100；NA 不进均值；Safety 组不进总分；若 `Safety Compliance=0`，总分一票否决为 0。
- **TI2I**：简单 baseline，与 T2I 分制对齐；0-4 分同样映射为 0/25/50/75/100；所有 applicable 且有分的 L3 维度等权平均；不使用 edit family 权重、L1/item 权重，也不做 Instruction Following gate 封顶。

三层分数结构：case 总分 `overall_score_100` → L1 子分 `overall.l1_subscores_100` →
L3 小分 `final_judgments[].final_score_100`（原始 0-4 档在 `final_score`）。

## 与完整管线的差异

单文件版保留全部评估功能（教师打分、冲突检测、三步 Debate、双模型合成、总分、训练导出），
仅省略工程特性：断点续跑 / 增量 checkpoint、批次隔离参数；调用级重试已内置。
