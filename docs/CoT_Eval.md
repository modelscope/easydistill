# CoT Eval Usage Tutorial

This tutorial will briefly explain how to use EasyDistill to score your chain-of-thought data. Including both text-based chain-of-thought data and multimodal (text-and-image) chain-of-thought data.

## Config

You can refer to the following JSON file for configuration.

```json
{
    "job_type": "cot_eval_api",
    "dataset": {
      "input_path": "cot_input.json",
      "output_path": "cot_output.json"
    },
    "inference":{
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key": "YOUR KEY",
      "max_new_tokens": 8196
    }
}
```
For reference, the formats for text and multimodal data are provided in `data/alpaca_en_demo.json` and `data/mllm_demo.json`, respectively.

To score text-based and multimodal chain-of-thought (CoT) data, set the `"job_type"` field to `"cot_eval_api"` or `"mmcot_eval_api"`, respectively.

The `base_url` can be replaced with any other URL that is OpenAI-compatible.

## Command

Run your CoT data evaluation.

```bash
export JUDGE_MODEL=qwen-plus

easydistill --config mmcot_eval_api.json
```