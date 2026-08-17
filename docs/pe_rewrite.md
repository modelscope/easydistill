# PE Rewrite Distillation

The `pe_rewrite_distill` pipeline distills a multi-step prompt-rewriting agent (plan -> rewrite -> reflection) into training data for a lightweight student model. Instead of answering questions, the student learns to rewrite short text-to-image prompts into detail-rich, production-ready prompts in a single call.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md). 中文版见 [pe_rewrite_zh.md](pe_rewrite_zh.md)。

## When to use this pipeline

Use this pipeline when you want to go from a set of raw text-to-image prompts (or a small seed set) to a curated prompt-rewriting SFT dataset with one command. The pipeline automatically:

1. (Optional) Expands each seed prompt into more same-scenario prompts with topic dedup and lineage tracking.
2. Rewrites every prompt through a three-step teacher agent chain: scene/language routing (plan), scene-specific rewriting (rewrite), and self-check correction (reflection).
3. Scores each `(original, rewritten)` pair with a combined nine-metric LLM judge in a single call per row.
4. Filters out rows that fail the score gates.
5. Builds the final SFT dataset (system + user + assistant messages) with a per-language end-to-end student instruction.

## Pipeline stages

| Stage | Required? | Purpose |
|---|---|---|
| `seed_anchored_expansion` | Optional | Expand each seed into same-scenario prompts (topic dedup, `source_seed_id`/`round` lineage). |
| `agentic_rewrite` | Required | Teacher chain: plan (scene + language routing) -> rewrite (scene-specific system prompt) -> reflection (self-check). |
| `pe_rewrite_eval` | Optional | Combined nine-metric judge; one call per row appends all scores. |
| `quality_filter` | Optional | Drop rows below the score gates; optional per-scene top-k/top-ratio selection. |
| `build_sft` | Required (last) | Convert surviving rows into SFT messages with the student system prompt. |

The last stage must always be `build_sft`.

## Scene taxonomy and rewrite prompts

The plan step routes every prompt into one of 10 scenes: `general`, `photographic_realism`, `artistic_illustration`, `design_layout`, `structured_diagram`, `ui_interface`, `brand_commercial_ad`, `narrative_panels`, `cultural_heritage_art`, `game_art_production`.

Each `(scene, language)` pair has a scene-specific rewrite prompt at `configs/prompts/pe_rewrite/rewrite_{scene}_{zh|en}.txt`. Scenes without a dedicated file fall back to the `general` prompt of the same language (the two general files are required).

Shared iron laws (fidelity, information density, on-screen text expansion, quoting, language rules, self-check, output format) live in `rewrite_common_{zh|en}.txt`. When present, the common block of the matching language is automatically prepended to every scene prompt at load time; when absent, scene prompts are used as-is. Scene files only carry scene-specific additions, and clauses that intentionally deviate from a global iron law declare the override explicitly (marked with ⚠️).

## Judge metrics and default filter gates

Seven 0-9 anchored metrics plus two boolean hard checks, all scored in one judge call per row:

| Metric | Type | Default gate |
|---|---|---|
| `intent_fidelity` | 0-9 | >= 7 |
| `text_rendering_completeness` | 0-9 | >= 7 |
| `usability` | 0-9 | >= 7 |
| `detail_enrichment` | 0-9 | >= 6 |
| `visual_concreteness` | 0-9 | >= 6 |
| `compositional_coverage` | 0-9 | >= 5 |
| `scene_alignment` | 0-9 | >= 5 |
| `language_consistency` | bool | must be `true` |
| `no_conflict` | bool | must be `true` |

`quality_filter` applies these gates by default (override via `min_scores`). Adding `keep_top_k` / `keep_top_ratio` enables a second selection pass ranked by the average of the seven scores; it runs per scene by default (ceiling rounding, at least one row per scene) so no scene is evicted wholesale — set `per_scene: false` for global ranking.

## Config schema

```yaml
job_type: pe_rewrite_distill

backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen3.7-plus         # default model for teacher steps

pipeline:
  - stage: seed_anchored_expansion   # omit when inputs are ready prompts
    config:
      rounds: 2
      generations_per_round: 5
    output_path: outputs/pe_expanded.jsonl
  - stage: agentic_rewrite
    config:
      reflection:
        model_id: qwen3.7-max        # stronger checker
      stream_output_path: outputs/pe_rewrite.stream.jsonl  # crash-safe sink
    output_path: outputs/pe_rewrite.jsonl
  - stage: pe_rewrite_eval
    config:
      model_id: qwen3.7-max          # judge separated from the teacher
      temperature: 0.0
    output_path: outputs/pe_scored.jsonl
  - stage: quality_filter
    config:
      per_scene: false               # hard gates only
    output_path: outputs/pe_filtered.jsonl
  - stage: build_sft

sft:
  system_prompt_zh_file: configs/prompts/pe_rewrite/student_system_zh.txt
  system_prompt_en_file: configs/prompts/pe_rewrite/student_system_en.txt

dataset:
  input_path: examples/seed_pe_prompts.jsonl
  output_path: outputs/pe_sft.jsonl
```

Reference configs:
- From seeds: `configs/pipeline/pe_rewrite_distill_from_seeds_pai_token.yaml` (PAI-Token) and `configs/pipeline/pe_rewrite_distill_from_seeds_pai_eas.yaml` (PAI-EAS)
- From ready prompts: `configs/pipeline/pe_rewrite_distill_pai_token.yaml` (PAI-Token) and `configs/pipeline/pe_rewrite_distill_pai_eas.yaml` (PAI-EAS)

The teacher (rewrite) and judge must run on different models sharing one backend endpoint (`model_id` overrides per stage) to avoid self-evaluation bias.

## Output row schema

Intermediate rows keep every input field plus:

```json
{"instruction": "...", "response": "...", "scene": "photographic_realism",
 "language": "zh", "agent_trace": {"plan": {}, "rewrite": {}, "reflection": {}, "durations": {}},
 "source_seed_id": "s1", "round": 0, "intent_fidelity": 8, "...": "..."}
```

The final SFT samples are `{"messages": [system, user, assistant], "metadata": {...}}`; judge scores and `agent_trace` are audit-only fields and never enter the SFT metadata, while expansion lineage does.

## Standalone jobs

Every stage is also exposed as a standalone `job_type` for debugging or resuming from an intermediate JSONL:

| Job | LLM calls? | Config example |
|---|---|---|
| `seed_anchored_expansion` | Yes | `configs/rewrite/seed_anchored_expansion_pai_token.yaml`<br>`configs/rewrite/seed_anchored_expansion_pai_eas.yaml` |
| `agentic_rewrite` | Yes | `configs/rewrite/agentic_rewrite_pai_token.yaml`<br>`configs/rewrite/agentic_rewrite_pai_eas.yaml` |
| `pe_rewrite_eval` | Yes | `configs/rewrite/pe_rewrite_eval_pai_token.yaml`<br>`configs/rewrite/pe_rewrite_eval_pai_eas.yaml` |
| `pe_rewrite_filter` | No (local) | `configs/rewrite/pe_rewrite_filter.yaml` |
| `pe_rewrite_build_sft` | No (local) | `configs/rewrite/pe_rewrite_build_sft.yaml` |

The two local jobs need no `backend` section.
