# RV / CD Mixer for CoT Data

The RV/CD mixer scores chain-of-thought (CoT) reasoning traces on **Reasoning Verbosity (RV)**, **Cognitive Difficulty (CD)**, and **logical correctness**, then mixes them into an SFT subset. It mirrors the OmniThought curriculum-mixing idea used to train the DistilQwen-ThoughtX series.

For the JSONL schemas used by each stage, see [data_formats.md](data_formats.md).

## When to use it

- You already have CoT data (from `cot_distill` or an external source) and want to select the best-length traces per difficulty level.

## Pipeline stages

These two stages are now used inside the `advanced_cot_distill` pipeline by default:

| Stage | Purpose |
|---|---|
| `cot_rvcd_score` | Score existing CoT data on RV/CD/correctness and save annotated rows. |
| `cot_mix_by_rv_cd` | Mix scored rows into an SFT subset. |

## Config schema

### Stage config

When used inside `advanced_cot_distill`, the `cot_rvcd_score` and `cot_mix_by_rv_cd` stages accept the fields below. `cot_rvcd_score` also inherits defaults from the top-level `eval` section (prompts, metrics, temperature, max_tokens, max_workers).

### `cot_rvcd_score`

Controls the LLM-as-judge scorer.

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `List[str]` | `[reasoning_verbosity, cognitive_difficulty, logical_correctness]` | Metrics to compute. |
| `prompts_file` | `str` | `null` | Path to a YAML/JSON file with custom judge prompts per metric. |
| `max_workers` | `int` | `10` | Concurrent judge calls. |
| `temperature` | `float` | `0.0` | Judge sampling temperature. |
| `max_tokens` | `int` | `512` | Judge max tokens. |
| `show_progress` | `bool` | `true` | Show progress bar. |
| `instruction_key` | `str` | `instruction` | Field name for the problem. |
| `output_key` | `str` | `response` | Field name for the CoT trace. |

### `cot_mix_by_rv_cd`

Controls the mixer.

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `sft` | Fixed to `sft`; selects rows whose RV is closest to the target. |
| `cd_bins` | `List[float]` | `[0, 3, 6, 10]` | Bin edges for cognitive-difficulty scores. |
| `rv_target` | `str \| float` | `matched` | Target RV. `matched`, `low`, `medium`, `high`, or a number. |
| `samples_per_bin` | `int` | `null` | Max rows per CD bin. |
| `min_correctness` | `int` | `1` | Minimum `logical_correctness` score to include. |

### `rv_target` semantics

- `matched`: RV target increases with the CD bin (easy problems → concise, hard problems → verbose).
- `low` / `medium` / `high`: fixed targets mapped to RV scores 2, 5, 8.
- numeric value: fixed target for every bin.

## Usage inside `advanced_cot_distill`

The default `advanced_cot_distill` config uses RV/CD scoring and mixing:

```bash
export PAI_TOKEN_API_KEY=your_key
easydistill --config configs/pipeline/advanced_cot_distill_pai_token.yaml
```

A PAI-EAS equivalent is provided as [`configs/pipeline/advanced_cot_distill_pai_eas.yaml`](../configs/pipeline/advanced_cot_distill_pai_eas.yaml).

## Example stage config

Inside an `advanced_cot_distill` config:

### SFT flow

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

The pipeline must end with `build_sft` for SFT data.

## Stage output formats

`cot_rvcd_score` writes one annotated row per input:

```jsonl
{"instruction": "What is the sum of the first 10 positive integers?", "response": "...", "reasoning_verbosity": 5, "cognitive_difficulty": 4, "logical_correctness": true}
```

`cot_mix_by_rv_cd` writes the selected subset, adding `cd_bin` and `rv_target`:

```jsonl
{"instruction": "...", "response": "...", "reasoning_verbosity": 2, "cognitive_difficulty": 2, "logical_correctness": true, "cd_bin": 0, "rv_target": 2.0}
```

## Notes

- The scorer uses the same `CoTEvaluator` and default prompts as `cot_eval`, so custom `prompts_file` values are supported.
- The mixer only includes rows whose `logical_correctness` is at least `min_correctness`.

