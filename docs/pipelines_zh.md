# 流水线

EasyDistill 2 提供多个端到端流水线 job_type。每个流水线将多个算子串联为一次配置驱动的运行，并保存中间阶段输出以便检查。

各阶段使用的 JSONL 格式见 [data_formats_zh.md](data_formats_zh.md)。

## 可用的流水线

| 流水线 | 用途 | 配置路径 |
|---|---|---|
| `augmented_instruct_distill` | 对种子指令进行扩充与精炼，再蒸馏教师回复为 SFT 数据。 | `configs/pipeline/augmented_instruct_distill_pai_token.yaml` |
| `advanced_instruct_distill` | 对种子指令扩充、精炼、生成、评估、过滤，构建高质量指令 SFT 数据集。 | `configs/pipeline/advanced_instruct_distill_pai_token.yaml` |
| `balanced_instruct_distill` | 合成指令，按类别平衡后生成回复并构建 SFT 数据集。 | `configs/pipeline/balanced_instruct_distill_pai_token.yaml` |
| `advanced_cot_distill` | 生成 CoT 推理，按 RV/CD 指标评分、按难度分箱混合并构建 SFT 数据集。 | `configs/pipeline/advanced_cot_distill_pai_token.yaml` |
| `advanced_mm_distill` | 生成多模态教师回复，评估、过滤并构建 SFT 数据集。 | `configs/pipeline/advanced_mm_distill_pai_token.yaml` |
| `advanced_mm_cot_distill` | 生成多模态 CoT 推理，评估、过滤并构建 SFT 数据集。 | `configs/pipeline/advanced_mm_cot_distill_pai_token.yaml`<br>`configs/pipeline/omnithoughtv_mm_cot_distill_pai_token.yaml`（OmniThoughtV 配方） |
| `advanced_t2i_distill` | 优化 prompt、生成图片、使用 VLM 裁判评估、过滤并构建多模态 SFT 数据集。 | `configs/t2i/advanced_t2i_distill_wanx.yaml`<br>`configs/t2i/advanced_t2i_distill_qwen_image.yaml`<br>`configs/t2i/advanced_t2i_distill_pai_diffusion.yaml` |
| `advanced_t2v_distill` | 优化 prompt（抽取 → 组合）、生成视频（T2V/I2V）、评估（VLM / omni / VBench）、过滤并构建多模态 SFT 数据集。支持阶段级断点续跑。 | `configs/pipeline/advanced_t2v_distill_pai_token.yaml`<br>`configs/pipeline/advanced_t2v_distill_pai_eas.yaml` |
| `pe_rewrite_distill` | 扩展种子 prompt，经 plan/rewrite/reflection 教师 Agent 链改写，裁判、过滤并构建 prompt 改写 SFT 数据集。 | `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` |

## 通用结构

流水线配置与其他 EasyDistill 2 配置使用相同的顶层字段：

```yaml
job_type: <pipeline_name>

backend:
  type: pai_token          # 或 pai_eas
  model_id: qwen2.5-72b-instruct

generation:
  system_prompt: "You are a helpful assistant."
  temperature: 0.7
  max_tokens: 2048

pipeline:
  - stage: <stage_name>
    config:
      ...
    output_path: outputs/stage1.jsonl

  - stage: <final_stage>
    config:
      ...

dataset:
  input_path: examples/seed_instructions.jsonl
  output_path: outputs/final.jsonl
  skip_empty: true
  min_length: 10
  max_length: 8192
```

- `pipeline`：有序阶段列表。每个阶段可将自己的输出写入 `output_path`。
- `dataset`：输入/输出路径与 SFT 过滤参数（`skip_empty`、`min_length`、`max_length`）。

各流水线的阶段级详情请参见对应的子文档。
