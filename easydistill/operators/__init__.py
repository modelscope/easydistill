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

from .agent import (
    AgentFuzzyTaskOperator,
    AgentRubricOperator,
    AgentTaskSynthesisOperator,
    AgentToolCheckOperator,
    AgentTrajectoryOperator,
)
from .balance import InstructionBalancer
from .base import Operator
from .cot import (
    CoTGenerationOperator,
    CoTRVCDMixer,
    CoTRVCDScorer,
)
from .generation import TextGenerationOperator
from .preference import (
    CandidateGenerationOperator,
    CoTScorer,
    LLMJudgeScorer,
    PreferenceDatasetBuilder,
    PreferencePairBuilder,
)
from .sft_builder import SFTDatasetBuilder
from .t2i import T2IGenerationOperator, T2IPromptOptimizer, T2ISFTBuilder

# Backward-compatible re-exports from easydistill.rewrite, resolved lazily to
# break the import cycle: easydistill.rewrite modules import from
# easydistill.operators (base/generation), so an eager import here would fail
# whenever easydistill.rewrite is imported first.
_REWRITE_REEXPORTS = {
    "InstructionExpansionOperator",
    "InstructionRefinementOperator",
    "InstructionResponseExtractionOperator",
}


def __getattr__(name):
    if name in _REWRITE_REEXPORTS:
        import easydistill.rewrite

        return getattr(easydistill.rewrite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Operator",
    "TextGenerationOperator",
    "SFTDatasetBuilder",
    "InstructionExpansionOperator",
    "InstructionRefinementOperator",
    "InstructionResponseExtractionOperator",
    "CoTGenerationOperator",
    "CoTRVCDScorer",
    "CoTRVCDMixer",
    "InstructionBalancer",
    "CandidateGenerationOperator",
    "CoTScorer",
    "LLMJudgeScorer",
    "PreferenceDatasetBuilder",
    "PreferencePairBuilder",
    "T2IPromptOptimizer",
    "T2IGenerationOperator",
    "T2ISFTBuilder",
    "AgentTaskSynthesisOperator",
    "AgentFuzzyTaskOperator",
    "AgentToolCheckOperator",
    "AgentTrajectoryOperator",
    "AgentRubricOperator",
]
