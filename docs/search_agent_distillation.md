# Search Agent Distillation

The `search_agent_distill` pipeline synthesizes verified multi-hop search
tasks from seed QA pairs and distills search-agent (ReAct) trajectories into
standard SFT training data. It ports the SearchSynthAgent closed-loop
generation system onto native easydistill operators — no LangGraph runtime
dependency; all model calls go through the unified backend abstraction.

## Pipeline Overview

```
Seed QA (id, question, answer)
  │
  ▼
search_task_evolve      Strategist-driven closed loop per seed:
  │                       Strategist → EXPAND / REFINE / ROLLBACK / FINALIZE
  │                       EXPAND  = atomic-QA merge (search target → atomic QA → rewrite)
  │                       REFINE  = FUZZ one clue without adding hops
  │                       QualityGate (uniqueness / pseudo multi-hop / leakage)
  │                       Verify (solver rollout or fast verify) → Judge (difficulty report)
  │                       Finalize gate: solver-correct AND difficulty == "good",
  │                       then N-run final eval (accuracy / avg turns)
  ▼
search_trajectory       repeat_times solver rollouts per evolved task;
  │                     each run judged for answer correctness
  ▼
search_judge_filter     keep correct trajectories, select best (fewest turns by default)
  │
  ▼
build_sft               full solve history → messages; judge labels, difficulty
                        report and final-eval stats → metadata
```

## Roles

Role settings live in the `search_agent.roles` config section and mirror the
original `step_models`:

| Role | Purpose | Default temperature |
|---|---|---|
| `strategist` | Decide next action from state/history/judge report | 0.0 |
| `synthesis` | Expand (atomic-QA merge) and Refine (FUZZ) rewriting | 0.7 |
| `search_sim` | LLM-simulated web search / browse (mock mode) | 0.7 |
| `judge` | Difficulty report, answer equivalence, quality gate | 0.0 |
| `solver` | ReAct rollout with `web_search` / `web_browse` | 0.7 |
| `fast_verify` | Cheap plan-based verification (optional) | 0.0 |

Each role may set `model_id`, `temperature`, and `max_tokens`; unset fields
fall back to the backend default model.

## Tools

`search_agent.tools.mode` selects the tool implementation:

- `mock` (default): the `search_sim` role simulates Google-style search
  results and page contents. Best for data synthesis.
- `real`: Google Custom Search API (`google_api_key` / `google_cx`) plus the
  Jina Reader API (`jina_api_key`, optional) with an optional SQLite cache
  (`cache_db_path`). Best for evaluation and real-data trajectories.

## Trajectory Format

Trajectories follow the messages convention used across easydistill agent
pipelines: assistant turns contain brief reasoning plus a
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` block, tool outputs
return as user turns wrapped in `<tool_response>...</tool_response>`, and the
final assistant turn terminates with `<answer>...</answer>`.

## Usage

```bash
easydistill --config configs/pipeline/search_agent_distill_pai_token.yaml
```

Seed rows accept `question`/`q`/`instruction` and
`answer`/`a_star`/`short_answer` aliases. See
`examples/seed_search_qa.jsonl` and
`configs/pipeline/search_agent_distill_pai_token.yaml` (or the `_pai_eas`
variant) for a complete configuration.

## Output Sample

Each SFT sample carries the full multi-turn conversation in `messages` and a
self-contained audit trail in `metadata`: task/seed provenance, hops, the
judge difficulty report, per-run correctness, and the final-eval statistics
(`accuracy`, `avg_turns`). Filtered tasks never reach the SFT stage; set
`keep_filtered: true` on `search_task_evolve` to keep them in intermediate
outputs for debugging.
