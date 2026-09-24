# System-1 蒸馏

System-1 蒸馏为学生模型的类型化决策头生成 RLCD（基于对比蒸馏的强化学习）训练数据。教师模型被要求为每个案例的决策选项给出概率分布；这些样本被聚合为软标签并打包为训练就绪的数据集。

## 何时使用此流水线

当你想把教师模型的**快速直觉决策**（system-1 思维）蒸馏到学生模型时使用此流水线。适用于：

- 你有一组案例，每个案例携带一个状态（如短信、客服工单）和若干类型化决策问题（多选、评分量表、是否判断）。
- 你想要教师模型的软概率标签，而非硬 one-hot 标签。
- 你想用 GRPO 式策略梯度加软交叉熵引导来训练学生模型的决策头。

流水线自动完成：

1. 从原始行和 schema 构建类型化决策案例行（本地，无 LLM）。
2. 为每个案例问题获取教师概率分布（每问 K 个样本）。
3. 将获取的样本聚合为教师标签，含一致性检查和防 one-hot 收缩（本地，无 LLM）。
4. 输出训练就绪的案例行，含嵌套 gold 标签（本地，无 LLM）。

## 流水线阶段

| 阶段 | 是否必需 | 用途 |
|---|---|---|
| `build_cases` | 必需（首阶段） | 从原始行 + schema 构建类型化决策案例行。 |
| `elicit` | 必需 | 为每个案例问题获取教师概率分布。 |
| `aggregate` | 必需 | 将获取样本聚合为教师标签。 |
| `build_dataset` | 必需（末阶段） | 输出训练就绪的案例行，含嵌套 gold 标签。 |

所有阶段支持阶段级续跑：若某阶段的 `output_path` 文件已存在，则跳过该阶段，其输出直接喂给后续阶段。`elicit` 阶段还支持行级 checkpoint。

## 快速开始

```bash
# 1. 设置 PAI-Token 凭据
export PAI_TOKEN_API_KEY=sk-pai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export PAI_TOKEN_BASE_URL=https://cn-beijing.pai-token.aliyuncs.com/v1

# 2. 运行完整流水线
easydistill --config configs/system1/system1_distill_pai_token.yaml
```

自带配置使用 `qwen3.7-max` 作为教师，蒸馏 8 条 SMS 垃圾短信分类示例。最终训练数据集写入 `outputs/system1_train_sms_pai_token.jsonl`。

## 输入格式

### 原始行（JSONL）

每行是一个扁平对象，含 `id`、标签字段（由 schema 的 `label_field` 指定，默认 `"label"`）和任意状态字段：

```json
{"id": "sms-001", "label": "ham", "text": "Hey, running late. Grab a table for 7:30 and I will meet you there."}
```

| 字段 | 是否必需 | 说明 |
|---|---|---|
| `id` | 必需 | 稳定的案例标识，贯穿所有阶段。 |
| `<label_field>` | 必需 | gold 标签（如 `"ham"` 或 `"spam"`）。字段名来自 schema。 |
| *其他字段* | 必需 | 所有非 id、非标签字段构成案例的 `state`。 |

也可通过 `questions` 字段（字典）提供预构建问题；若存在，则直接透传，不使用 schema 中的问题定义。

### Schema（YAML）

```yaml
name: sms_spam
question: spam
instructions: "Classify the SMS message."
options:
  spam: "Unsolicited commercial or fraudulent message"
  ham: "Legitimate personal, service, or transactional message"
```

| 字段 | 是否必需 | 说明 |
|---|---|---|
| `name` | 必需 | 工作流名称，存为案例行的 `workflow`。 |
| `question` | 必需 | 问题键（如 `"spam"`）。 |
| `instructions` | 必需 | 展示给教师的指令。 |
| `options` | 必需（扁平） | 扁平选项字典：`{key: 描述}`。默认 `type` 为 `"choice"`。 |
| `groups` | 必需（层级） | 层级选项组（banking77 风格）。 |
| `label_field` | 可选 | 原始行标签字段名（默认 `"label"`）。 |
| `id_field` | 可选 | 原始行 id 字段名（默认 `"id"`）。 |
| `type` | 可选 | 问题类型：`"choice"`、`"score"` 或 `"noul"`（默认 `"choice"`）。 |

## 配置参考

```yaml
job_type: system1_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  base_url: ${PAI_TOKEN_BASE_URL}
  model_id: qwen3.7-max

dataset:
  input_path: examples/system1_sms_raw.jsonl
  output_path: outputs/system1_train_sms_pai_token.jsonl

system1:
  schema_path: examples/system1_sms_schema.yaml
  variant: distill              # labeled | distill | mixed
  method: k_verbalized_mean     # single_verbalized | k_sample_freq | k_verbalized_mean | two_stage
  samples: 4                    # 每问 K 个样本
  temperature: 0.7
  max_tokens: 2048
  max_workers: 8
  seed: 7
  shuffle_options: true         # 每次调用打乱选项序（防教师位置偏差）
  show_progress: true
  resume: true                  # 阶段级 + 行级续跑
  consistency_threshold: 0.6    # top-1 一致率低于阈值则丢弃
  shrinkage: 0.02               # 防 one-hot 退化收缩（0 禁用）
  alpha: 0.7                    # mixed 变体的 gold 权重

pipeline:
  - stage: build_cases
    output_path: outputs/system1_cases_sms_pai_token.jsonl
  - stage: elicit
    config:
      max_workers: 8
      resume: true
    output_path: outputs/system1_elicit_sms_pai_token.jsonl
  - stage: aggregate
    config:
      consistency_threshold: 0.6
    output_path: outputs/system1_labels_sms_pai_token.jsonl
  - stage: build_dataset
    config:
      variant: distill
    output_path: outputs/system1_train_sms_pai_token.jsonl
```

### 配置项说明

| 键 | 默认值 | 说明 |
|---|---|---|
| `system1.schema_path` | — | Schema YAML 路径。 |
| `system1.variant` | `distill` | `labeled`：直通 gold 标签（不调教师）。`distill`：纯教师标签。`mixed`：gold 与教师 alpha 混合。 |
| `system1.method` | `single_verbalized` | 获取方法。`single_verbalized`：1 次调用，T=0。`k_sample_freq`：K 次调用，频率聚合。`k_verbalized_mean`：K 次调用，言语化概率均值（推荐）。`two_stage`：粗筛再细判。 |
| `system1.samples` | 1 / 16 / 4 | K — 每问教师样本数（随方法不同）。 |
| `system1.temperature` | 0 / 0.7 | 教师调用采样温度。 |
| `system1.max_tokens` | 2048 | 每次教师响应最大 token 数。 |
| `system1.max_workers` | 8 | elicit 阶段并发 API 调用数。 |
| `system1.seed` | 7 | 选项打乱随机种子。 |
| `system1.shuffle_options` | `true` | 每次调用打乱选项序以消除教师位置偏差。 |
| `system1.resume` | `true` | 阶段级续跑（跳过输出已存在的阶段）+ elicit 行级 checkpoint。 |
| `system1.consistency_threshold` | 0.6 | 跨样本 top-1 一致率低于此阈值的问题被丢弃。 |
| `system1.shrinkage` | 0.02 | 将聚合概率向均匀分布收缩以防 one-hot 塌缩（0 禁用）。 |
| `system1.alpha` | 0.7 | `mixed` 变体的 gold 权重（教师权重 = 1 - alpha）。 |
| `pipeline` | — | 阶段配置列表。每项含 `stage`、可选 `config`（与 `system1` 合并的 per-stage 覆盖）和 `output_path`。 |

## 各阶段输入输出

以下展示同一条行（`sms-001`，一条正常短信）流经全部四个阶段的真实数据，采集自 `qwen3.7-max` 实测运行。

### 阶段 1 — build_cases（输入 → 案例）

**输入**（原始行）：

```json
{"id": "sms-001", "label": "ham", "text": "Hey, running late. Grab a table for 7:30 and I will meet you there."}
```

**输出**（案例行）：

```json
{
  "id": "sms-001",
  "workflow": "sms_spam",
  "state": {"text": "Hey, running late. Grab a table for 7:30 and I will meet you there."},
  "questions": {
    "spam": {
      "type": "choice",
      "instructions": "Classify the SMS message.",
      "criteria": {
        "spam": "Unsolicited commercial or fraudulent message",
        "ham": "Legitimate personal, service, or transactional message"
      }
    }
  },
  "gold": {"spam": {"spam": 0.0, "ham": 1.0}},
  "source": {"id": "sms-001", "label": "ham", "text": "..."}
}
```

原始行的 `label`（"ham"）被展开为 one-hot gold 分布。`state` 从所有非 id、非标签字段自动派生。原始行保留在 `source` 中。

### 阶段 2 — elicit（案例 → 获取记录）

每个问题获得 K=4 个教师样本。每个样本是一次独立的 API 调用（选项序已打乱）。以下是 `sms-001` 的 4 个样本之一：

```json
{
  "id": "sms-001|spam|0",
  "question_id": "spam",
  "sample": 0,
  "ok": true,
  "probabilities": {"spam": 0.0, "ham": 1.0},
  "model": "qwen3.7-max",
  "usage": {
    "prompt_tokens": 126,
    "completion_tokens": 175,
    "total_tokens": 301,
    "completion_tokens_details": {"reasoning_tokens": 150}
  },
  "errors": []
}
```

此行的全部 4 个样本均返回 `{"spam": 0.0, "ham": 1.0}` — 教师一致且正确地判定为 ham。`elicitations` 字段被添加到每个案例行：

```json
{
  "elicitations": {
    "spam": {
      "method": "k_verbalized_mean",
      "model": "qwen3.7-max",
      "samples": [ /* 4 个样本对象 */ ]
    }
  }
}
```

### 阶段 3 — aggregate（获取记录 → 标签）

4 个样本被聚合为单个教师标签。收缩（0.02）将分布略微拉离纯 one-hot：

```json
{
  "teacher": {
    "spam": {
      "ok": true,
      "method": "k_verbalized_mean",
      "model": "qwen3.7-max",
      "probabilities": {"spam": 0.0098, "ham": 0.9902},
      "label": "ham",
      "consistency": 1.0,
      "samples_used": 4,
      "samples_failed": 0,
      "gold_tv": 0.0098,
      "gold_kl": 0.0099
    }
  }
}
```

- `consistency`：1.0 表示 4 个样本在 top-1 选项上一致。
- `gold_tv` / `gold_kl`：与 gold one-hot 分布的全变差距离和 KL 散度。
- `label`：聚合概率的 argmax。

### 阶段 4 — build_dataset（标签 → 训练数据集）

最终训练行去掉中间字段，将 `gold` 重构为训练入口所需的嵌套格式：

```json
{
  "id": "sms-001",
  "workflow": "sms_spam",
  "state": {"text": "Hey, running late. Grab a table for 7:30 and I will meet you there."},
  "questions": {
    "spam": {
      "type": "choice",
      "instructions": "Classify the SMS message.",
      "criteria": {
        "spam": "Unsolicited commercial or fraudulent message",
        "ham": "Legitimate personal, service, or transactional message"
      }
    }
  },
  "gold": {
    "spam": {
      "probabilities": {"spam": 0.0098, "ham": 0.9902},
      "label": "ham"
    }
  }
}
```

## 微调

流水线产出训练数据集后，使用自带的训练入口微调 Laya 的类型化决策头：

```bash
python examples/system1_train_entry.py \
  --input outputs/system1_train_sms_pai_token.jsonl \
  --output-dir outputs/system1_laya_finetuned \
  --nproc 2
```

父进程从 Hugging Face Hub 下载 Laya 模型，将数据集预处理为 tokenized 训练项，然后在 `torch.distributed.run` 下重新执行自身。子进程以 DDP 进行 GRPO 式策略梯度加软交叉熵引导训练，rank 0 导出带校准温度的微调模型。

> **需要 GPU 和 `laya` 研究包。** 请在多 GPU 机器（2xT4 或更高）上运行。`--model-dir` 标志可复用本地 Laya 快照而非下载。

### 训练入口标志

| 标志 | 默认值 | 说明 |
|---|---|---|
| `--input` | （必需） | `system1_build_dataset` JSONL 输出路径。 |
| `--output-dir` | `outputs/system1_laya_finetuned` | 导出模型目录。 |
| `--model-dir` | （自动下载） | 复用本地 Laya 快照而非下载。 |
| `--items` | `<output-dir>/train_items.pt` | 预处理训练项缓存。 |
| `--nproc` | 2 | DDP GPU 数量。 |

## 独立阶段

每个阶段也作为独立 `job_type` 透出，便于调试或从中间 JSONL 续跑：

| `job_type` | 用途 | 需要 backend？ |
|---|---|---|
| `system1_build_cases` | 从原始行 + schema 构建案例行。 | 否 |
| `system1_elicit` | 获取教师分布。 | 是 |
| `system1_aggregate` | 将获取样本聚合为标签。 | 否 |
| `system1_build_dataset` | 输出训练就绪案例行。 | 否 |

`system1_distill` 的 `labeled` 变体跳过 `elicit` 和 `aggregate`（gold 标签直接透传），从不调用教师。
