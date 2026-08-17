# CoT 数据的 RV / CD 混合器

RV/CD 混合器对思维链（CoT）推理轨迹进行 **推理冗长度（Reasoning Verbosity, RV）**、**认知难度（Cognitive Difficulty, CD）** 和 **逻辑正确性（logical correctness）** 评分，然后将其混合为 SFT 子集。它借鉴了 OmniThought 课程混合思想，该思想被用于训练 DistilQwen-ThoughtX 系列模型。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 适用场景

- 你已经拥有 CoT 数据（来自 `cot_distill` 或外部来源），希望按难度级别选择长度最合适的轨迹。

## 流水线阶段

以下两个阶段现在是 `advanced_cot_distill` 流水线的默认阶段：

| 阶段 | 用途 |
|---|---|
| `cot_rvcd_score` | 对现有 CoT 数据计算 RV/CD/正确性分数，并保存带标注的行。 |
| `cot_mix_by_rv_cd` | 将评分后的行混合为 SFT 子集。 |

## 配置 Schema

### 阶段配置

在 `advanced_cot_distill` 中使用时，`cot_rvcd_score` 与 `cot_mix_by_rv_cd` 阶段接受以下字段。`cot_rvcd_score` 还会继承顶层 `eval` 的默认配置（提示词、指标、温度、最大 token 数、并发数）。

### `cot_rvcd_score`

控制 LLM-as-judge 评分器。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `metrics` | `List[str]` | `[reasoning_verbosity, cognitive_difficulty, logical_correctness]` | 要计算的指标。 |
| `prompts_file` | `str` | `null` | 自定义每个指标裁判提示词的 YAML/JSON 文件路径。 |
| `max_workers` | `int` | `10` | 裁判模型并发调用数。 |
| `temperature` | `float` | `0.0` | 裁判采样温度。 |
| `max_tokens` | `int` | `512` | 裁判最大 token 数。 |
| `show_progress` | `bool` | `true` | 是否显示进度条。 |
| `instruction_key` | `str` | `instruction` | 问题字段名。 |
| `output_key` | `str` | `response` | CoT 轨迹字段名。 |

### `cot_mix_by_rv_cd`

控制混合器。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `mode` | `str` | `sft` | 固定为 `sft`，按目标 RV 选择行。 |
| `cd_bins` | `List[float]` | `[0, 3, 6, 10]` | 认知难度分数的分箱边界。 |
| `rv_target` | `str \| float` | `matched` | 目标 RV。可选 `matched`、`low`、`medium`、`high` 或数值。 |
| `samples_per_bin` | `int` | `null` | 每个 CD 分箱保留的最大行数。 |
| `min_correctness` | `int` | `1` | 包含样本所需的最小逻辑正确性分数。 |

### `rv_target` 语义

- `matched`：RV 目标随 CD 分箱递增（简单问题 → 简洁，困难问题 → 详细）。
- `low` / `medium` / `high`：固定目标，分别映射到 RV 分数 2、5、8。
- 数值：每个分箱使用相同的固定目标。

## 在 `advanced_cot_distill` 中使用

默认的 `advanced_cot_distill` 配置已使用 RV/CD 评分与混合：

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/advanced_cot_distill_pai_token.yaml
```

PAI-EAS 等价配置见 [`configs/pipeline/advanced_cot_distill_pai_eas.yaml`](../configs/pipeline/advanced_cot_distill_pai_eas.yaml)。

## 阶段配置示例

在 `advanced_cot_distill` 配置中：

### SFT 流程

```yaml
pipeline:
  - stage: cot_distill
    config: {}
    output_path: outputs/cot_stage1.jsonl

  - stage: cot_rvcd_score
    config:
      max_workers: 4
    output_path: outputs/cot_stage2_scored.jsonl

  - stage: cot_mix_by_rv_cd
    config:
      mode: sft
      samples_per_bin: 100
    output_path: outputs/cot_stage3_mixed.jsonl

  - stage: build_sft
    config: {}
```

SFT 数据必须以 `build_sft` 结尾。

## 阶段输出格式

`cot_rvcd_score` 为每个输入输出一行带标注的行：

```jsonl
{"instruction": "前 10 个正整数的和是多少？", "response": "...", "reasoning_verbosity": 5, "cognitive_difficulty": 4, "logical_correctness": true}
```

`cot_mix_by_rv_cd` 输出选中的子集，并追加 `cd_bin` 与 `rv_target`：

```jsonl
{"instruction": "...", "response": "...", "reasoning_verbosity": 2, "cognitive_difficulty": 2, "logical_correctness": true, "cd_bin": 0, "rv_target": 2.0}
```

## 说明

- 评分器使用与 `cot_eval` 相同的 `CoTEvaluator` 和默认提示词，因此支持自定义 `prompts_file`。
- 混合器仅包含 `logical_correctness` 不低于 `min_correctness` 的行。

