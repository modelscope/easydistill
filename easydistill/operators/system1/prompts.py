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

"""Elicitation prompt templates for system-1 distillation.

Every template shares the ``Input / Question / Options`` layout: the state
text, the question instructions, and one ``- key: description`` line per
option, keyed exactly like the student's training targets.

`ELICIT_PROBABILITY_TEMPLATE` (single_verbalized, k_verbalized_mean) asks for
a JSON probability distribution over the option keys; `ELICIT_FREQ_TEMPLATE`
(k_sample_freq) asks for the single best key, which the aggregate stage counts
over K samples. The two-stage pair first elicits a written analysis, then
turns it into a distribution. All of them can be overridden through the
elicitation stage config.
"""

from typing import Any, Dict, List, Optional

from .protocol import render_criterion, render_question_options

ELICIT_PROBABILITY_TEMPLATE = (
    "You are a careful annotator. Assign a probability to each option.\n\n"
    "Input: {state}\n"
    "Question: {instructions}\n"
    "Options:\n"
    "{options}\n\n"
    "Consider each option against the input before deciding. Output ONLY a JSON "
    "object mapping every option key to a probability in [0, 1]. Probabilities "
    "must sum to 1."
)


ELICIT_FREQ_TEMPLATE = (
    "You are a careful annotator. Pick the single best option for the input.\n\n"
    "Input: {state}\n"
    "Question: {instructions}\n"
    "Options:\n"
    "{options}\n\n"
    "Consider each option against the input before deciding. Output ONLY the key "
    "of the best option, nothing else."
)


TWO_STAGE_ANALYSIS_TEMPLATE = (
    "You are a careful annotator. Analyze the question against the input.\n\n"
    "Input: {state}\n"
    "Question: {instructions}\n"
    "Options:\n"
    "{options}\n\n"
    "Briefly evaluate each option against the input. Do not give a final answer "
    "yet."
)


TWO_STAGE_DISTRIBUTION_TEMPLATE = (
    "You are a careful annotator. Using the analysis below, assign a probability "
    "to each option.\n\n"
    "Input: {state}\n"
    "Question: {instructions}\n"
    "Options:\n"
    "{options}\n\n"
    "Analysis:\n"
    "{analysis}\n\n"
    "Output ONLY a JSON object mapping every option key to a probability in "
    "[0, 1]. Probabilities must sum to 1."
)


def build_option_lines(question: Dict[str, Any]) -> List[str]:
    """Render ``- key: description`` lines keyed by the student's target keys.

    Choice and noul questions reuse ``render_question_options`` verbatim
    (their rendered texts already begin with the target key); score questions
    re-render as ``- 0: <description>`` because the student's target keys are
    the level indexes, not the rendered ``level 0`` labels.
    """
    if question["type"] == "score":
        criteria = question.get("criteria") or []
        return [f"- {i}: {render_criterion(c)}" for i, c in enumerate(criteria)]
    return [f"- {line}" for line in render_question_options(question)]


def build_elicit_user_text(
    question: Dict[str, Any],
    state_text: str,
    *,
    option_lines: Optional[List[str]] = None,
    template: Optional[str] = None,
    analysis: Optional[str] = None,
) -> str:
    """Format one elicitation user turn.

    *option_lines* overrides the canonical option order (the elicitation
    operator shuffles lines per sample and passes the same list to the
    two-stage follow-up so both stages see one order); *analysis* splices the
    stage-one reply into the two-stage distribution template. Format fields
    the chosen template does not use are ignored.
    """
    if option_lines is None:
        option_lines = build_option_lines(question)
    return (template or ELICIT_PROBABILITY_TEMPLATE).format(
        state=state_text,
        instructions=question["instructions"],
        options="\n".join(option_lines),
        analysis=analysis or "",
    )
