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

"""Balanced instruction distillation pipeline."""

from .advanced_instruct_distill import AdvancedInstructDistillPipeline


class BalancedInstructDistillPipeline(AdvancedInstructDistillPipeline):
    """End-to-end pipeline that synthesizes, balances, and distills instructions.

    The recommended flow is:
      1. instruction_expansion (or instruction_response_extraction) - synthesize
         new instructions from seeds or raw text.
      2. instruction_balance - classify instructions by task/domain and resample
         to a target distribution.
      3. generate - produce teacher responses for the balanced instructions.
      4. build_sft - package the results into an SFT dataset.

    Optional stages such as instruction_refinement, instruct_eval, and
    quality_filter can also be inserted between synthesis and the final
    build_sft stage.

    The last stage must be `build_sft`.
    """
