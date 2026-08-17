# PE Rewrite 蒸馏

`pe_rewrite_distill` 流水线把一条多步 prompt 改写 Agent 链路（plan -> rewrite -> reflection）蒸馏为轻量学生模型的训练数据。与常规指令蒸馏不同，学生模型学习的不是回答问题，而是把简短的文生图 prompt 单次改写为细节丰富、可直接送入生图模型的完整 prompt。

各阶段 JSONL 数据格式见 [data_formats_zh.md](data_formats_zh.md)。English version: [pe_rewrite.md](pe_rewrite.md).

## 适用场景

当你希望用一条命令，从一批原始文生图 prompt（或一小批种子）得到一份高质量的"prompt 改写" SFT 数据集时，使用本流水线。流水线自动完成：

1. （可选）把每条种子扩展为多条同场景 prompt，带主题去重与血缘追踪。
2. 每条 prompt 走三步教师 Agent 链：场景/语言路由（plan）-> 分场景专业改写（rewrite）-> 自检修正（reflection）。
3. 用合并九维 LLM 裁判为每对（原始, 改写）打分——每行仅一次裁判调用。
4. 按分数门槛过滤不合格样本。
5. 构建最终 SFT 数据集（system + user + assistant 消息），学生 system 为端到端改写指令、按行语言注入中/英版本。

## 流水线阶段

| 阶段 | 必需？ | 作用 |
|---|---|---|
| `seed_anchored_expansion` | 可选 | 每条种子扩展为同场景新 prompt（主题去重，携带 `source_seed_id`/`round` 血缘） |
| `agentic_rewrite` | 必需 | 教师链：plan（场景+语言路由）-> rewrite（分场景专属 SP）-> reflection（自检） |
| `pe_rewrite_eval` | 可选 | 合并九维裁判，每行一次调用追加全部分数 |
| `quality_filter` | 可选 | 按分数门槛丢弃不合格行；可选分场景 top-k/top-ratio 择优 |
| `build_sft` | 必需（末位） | 将幸存行转为带学生 system prompt 的 SFT 消息 |

最后一个阶段必须是 `build_sft`。

## 场景体系与改写 SP

plan 步把每条 prompt 路由到 10 类场景之一：`general`、`photographic_realism`、`artistic_illustration`、`design_layout`、`structured_diagram`、`ui_interface`、`brand_commercial_ad`、`narrative_panels`、`cultural_heritage_art`、`game_art_production`。

每个（场景, 语言）组合对应一份场景专属改写 prompt：`configs/prompts/pe_rewrite/rewrite_{scene}_{zh|en}.txt`。缺失场景文件时回退同语言 `general` SP（两份 general 文件必须存在）。

公共铁律（忠实保真、信息密度、画面文字逐条展开、引号规范、语种规则、自检、输出格式）统一放在 `rewrite_common_{zh|en}.txt`：该文件存在时，加载阶段自动将对应语言的公共块拼接到每份场景 prompt 之前；不存在时场景 prompt 按原样使用。场景文件只保留场景增补规则，与全局铁律有意冲突的条款会显式声明覆盖（以 ⚠️ 标注）。

## 裁判维度与默认过滤门槛

七个 0-9 锚定评分维度 + 两个布尔硬校验，每行一次裁判调用全部产出：

| 维度 | 类型 | 默认门槛 |
|---|---|---|
| `intent_fidelity` 意图保真度 | 0-9 | >= 7 |
| `text_rendering_completeness` 画面文字渲染完整性 | 0-9 | >= 7 |
| `usability` 输出可用性 | 0-9 | >= 7 |
| `detail_enrichment` 细节丰富度 | 0-9 | >= 6 |
| `visual_concreteness` 视觉可渲染性 | 0-9 | >= 6 |
| `compositional_coverage` 画面要素覆盖度 | 0-9 | >= 5 |
| `scene_alignment` 场景适配度 | 0-9 | >= 5 |
| `language_consistency` 语言一致性 | bool | 必须为 `true` |
| `no_conflict` 无冲突添加 | bool | 必须为 `true` |

`quality_filter` 默认套用上表门槛（可用 `min_scores` 覆盖）。配置 `keep_top_k` / `keep_top_ratio` 可再按七维平均分做第二轮择优；择优默认**分场景**执行（向上取整、每场景至少保留 1 条），避免评分风格偏低的场景被全局排序整类挤出；`per_scene: false` 可回退全局排序。

## 配置示例

```yaml
job_type: pe_rewrite_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen3.7-plus         # 教师步骤默认模型

pipeline:
  - stage: seed_anchored_expansion   # 输入已是现成 prompt 时可省略本段
    config:
      rounds: 2
      generations_per_round: 5
    output_path: outputs/pe_expanded.jsonl
  - stage: agentic_rewrite
    config:
      reflection:
        model_id: qwen3.7-max        # 更强的自检模型
      stream_output_path: outputs/pe_rewrite.stream.jsonl  # 逐条实时落盘，防中断丢数据
    output_path: outputs/pe_rewrite.jsonl
  - stage: pe_rewrite_eval
    config:
      model_id: qwen3.7-max          # 裁判与教师模型分离
      temperature: 0.0
    output_path: outputs/pe_scored.jsonl
  - stage: quality_filter
    config:
      per_scene: false               # 仅硬门槛，不做择优裁剪
    output_path: outputs/pe_filtered.jsonl
  - stage: build_sft

sft:
  system_prompt_zh_file: configs/prompts/pe_rewrite/student_system_zh.txt
  system_prompt_en_file: configs/prompts/pe_rewrite/student_system_en.txt

dataset:
  input_path: examples/seed_pe_prompts.jsonl
  output_path: outputs/pe_sft.jsonl
```

参考配置：
- 从种子起跑：`configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` (PAI-Token) 与 `configs/pipeline/pe_rewrite_distill_from_seeds_pai_eas.yaml` (PAI-EAS)
- 从现成 prompt 起跑：`configs/pipeline/pe_rewrite_distill_pai_token.yaml` (PAI-Token) 与 `configs/pipeline/pe_rewrite_distill_pai_eas.yaml` (PAI-EAS)

教师（rewrite）与裁判必须在同一 backend 端点上使用不同模型（通过各阶段的 `model_id` 覆盖），以规避自评偏置。

## 输出行结构

中间产物每行保留全部输入字段，并追加：

```json
{"instruction": "原始 prompt", "response": "改写后 prompt", "scene": "photographic_realism",
 "language": "zh", "agent_trace": {"plan": {}, "rewrite": {}, "reflection": {}, "durations": {}},
 "source_seed_id": "s1", "round": 0, "intent_fidelity": 8, "...": "..."}
```

最终 SFT 样本为 `{"messages": [system, user, assistant], "metadata": {...}}`；裁判分数与 `agent_trace` 仅用于审计，不进入 SFT metadata，而扩展血缘字段会保留。

## 单步作业

每个阶段同时提供独立 `job_type`，便于调试或从任意中间 JSONL 续跑：

| 作业 | 是否调用 LLM | 配置示例 |
|---|---|---|
| `seed_anchored_expansion` | 是 | `configs/rewrite/seed_anchored_expansion_pai_token.yaml`<br>`configs/rewrite/seed_anchored_expansion_pai_eas.yaml` |
| `agentic_rewrite` | 是 | `configs/rewrite/agentic_rewrite_pai_token.yaml`<br>`configs/rewrite/agentic_rewrite_pai_eas.yaml` |
| `pe_rewrite_eval` | 是 | `configs/rewrite/pe_rewrite_eval_pai_token.yaml`<br>`configs/rewrite/pe_rewrite_eval_pai_eas.yaml` |
| `pe_rewrite_filter` | 否（纯本地） | `configs/rewrite/pe_rewrite_filter.yaml` |
| `pe_rewrite_build_sft` | 否（纯本地） | `configs/rewrite/pe_rewrite_build_sft.yaml` |

两个纯本地作业的配置无需 `backend` 段。
