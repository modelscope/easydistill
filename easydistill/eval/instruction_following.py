# Copyright 2026 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LLM-as-judge evaluation for instruction-following datasets."""

from typing import Any, Dict, Tuple

from .base import LLMJudgeEvaluator


class InstructionFollowingEvaluator(LLMJudgeEvaluator):
    """Evaluate instruction-following data with an LLM judge."""

    name = "instruct_evaluator"
    DEFAULT_PROMPTS_FILE = "configs/prompts/default_eval_prompts.yaml"
    BOOL_METRICS = {"correctness"}

    def _extract_sample(self, sample: Dict[str, Any]) -> Tuple[str, str, str]:
        sample_id = str(sample.get("id", sample.get("index", 0)))
        instruction = sample.get("instruction") or sample.get("input") or ""
        output = sample.get("output") or sample.get("response") or ""
        return sample_id, instruction, output
