# Search Agent 蒸馏

`search_agent_distill` 管线从种子 QA 出发合成经过验证的多跳搜索任务，并将
search agent（ReAct）轨迹蒸馏为标准 SFT 训练数据。它将 SearchSynthAgent 的
闭环生成系统移植为 easydistill 原生 operator —— 不依赖 LangGraph 运行时，
所有模型调用统一走 backend 抽象。

## 管线总览

```
种子 QA (id, question, answer)
  │
  ▼
search_task_evolve      每个种子的 Strategist 驱动闭环：
  │                       Strategist → EXPAND / REFINE / ROLLBACK / FINALIZE
  │                       EXPAND  = 原子 QA 合并（搜索目标实体 → 生成原子 QA → 改写问题）
  │                       REFINE  = FUZZ 单个线索，不增加跳数
  │                       QualityGate 三查（唯一性 / 伪多跳 / 信息泄露）
  │                       Verify（Solver 实跑或 fast verify）→ Judge（难度报告）
  │                       Finalize 门槛：Solver 答对 且 难度 == "good"，
  │                       再做 N 次终评（accuracy / avg_turns）
  ▼
search_trajectory       每个任务 repeat_times 次 Solver 实跑，逐次判定答案正确性
  │
  ▼
search_judge_filter     只保留答对的轨迹，选最优（默认轮数最少）
  │
  ▼
build_sft               完整对话历史 → messages；judge 标签、难度报告、
                        终评统计 → metadata
```

## 角色

角色配置位于 `search_agent.roles`，与原项目 `step_models` 一一对应：

| 角色 | 职责 | 默认温度 |
|---|---|---|
| `strategist` | 依据状态/历史/难度报告决策下一步动作 | 0.0 |
| `synthesis` | Expand（原子 QA 合并）与 Refine（FUZZ）改写 | 0.7 |
| `search_sim` | LLM 模拟 web 搜索 / 浏览（mock 模式） | 0.7 |
| `judge` | 难度报告、答案等价判定、质量门禁 | 0.0 |
| `solver` | 带 `web_search` / `web_browse` 的 ReAct 实跑 | 0.7 |
| `fast_verify` | 低成本计划式验证（可选） | 0.0 |

每个角色可设置 `model_id`、`temperature`、`max_tokens`，未设置的字段回落到
backend 默认模型。

## 工具

`search_agent.tools.mode` 选择工具实现：

- `mock`（默认）：由 `search_sim` 角色模拟 Google 风格搜索结果与页面内容，
  适合数据合成；
- `real`：Google Custom Search API（`google_api_key` / `google_cx`）+ Jina
  Reader API（`jina_api_key` 可选），可配 SQLite 缓存（`cache_db_path`），
  适合评测与真实数据轨迹。

## 轨迹格式

轨迹遵循 easydistill agent 管线统一的 messages 约定：assistant 轮包含简要
推理与 `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 块，工具输出
以 `<tool_response>...</tool_response>` 包裹的 user 轮返回，最后一条
assistant 消息以 `<answer>...</answer>` 终止。

## 使用

```bash
easydistill --config configs/pipeline/search_agent_distill_pai_token.yaml
```

种子字段支持别名：`question`/`q`/`instruction` 与
`answer`/`a_star`/`short_answer`。完整配置见
`examples/seed_search_qa.jsonl` 与
`configs/pipeline/search_agent_distill_pai_token.yaml`（或 `_pai_eas` 变体）。

## 输出样本

每条 SFT 样本的 `messages` 携带完整多轮对话，`metadata` 携带可独立审计的
全部信息：任务/种子溯源、跳数、judge 难度报告、逐次正确性以及终评统计
（`accuracy`、`avg_turns`）。被过滤的任务不会进入 SFT 阶段；在
`search_task_evolve` 上设置 `keep_filtered: true` 可将其保留在中间产物中
用于调试。
