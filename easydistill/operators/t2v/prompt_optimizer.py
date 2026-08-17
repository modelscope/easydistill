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

"""Two-stage T2V/I2V prompt optimization operator: extract -> compose.

Stage 1 (**extract**) is model-agnostic video parsing: one LLM/VLM call per
row that turns the short user prompt (grounded in the first-frame image for
I2V rows) into a structured JSON draft of everything a video caption needs
(subject, action, setting, camera, temporal beats, ...).

Stage 2 (**compose**) adapts the draft to a specific target video model: one
text-only LLM call that rewrites the draft into the model's caption schema.
A generic model-agnostic schema is built in as the default; every video
model has its own preferred prompt style documented by its vendor, so for
best quality users point ``schema_file`` (or inline ``schema``) at a schema
written from their generation backend's official prompt guideline.

Exactly two model calls per row, no fix/retry loop by design.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.operators.prompt_base import PromptGenerationOperator
from easydistill.operators.synthesis.utils import extract_tagged_answer
from easydistill.prompts import resolve_prompt
from easydistill.utils import build_multimodal_user_content, format_prompt_safely

logger = logging.getLogger(__name__)

_DEFAULT_EXTRACT_PROMPT_FILE = "configs/prompts/t2v_extract_prompt.txt"
_DEFAULT_I2V_EXTRACT_PROMPT_FILE = "configs/prompts/i2v_extract_prompt.txt"
_DEFAULT_COMPOSE_PROMPT_FILE = "configs/prompts/t2v_compose_prompt.txt"

_ASPECT_RATIO_PATTERN = re.compile(r"^\d{1,2}\s*:\s*\d{1,2}$")


def _draft_aspect_ratio(draft: str) -> Optional[str]:
    """Read the LLM-inferred ``aspect_ratio`` from a JSON draft, if valid."""
    text = draft.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    value = str(data.get("aspect_ratio") or "").strip()
    if _ASPECT_RATIO_PATTERN.match(value):
        return value.replace(" ", "")
    return None

# Generic model-agnostic caption schema used when the job config supplies
# neither `schema` nor `schema_file`.  It fills the compose template's
# {schema} placeholder verbatim.
_DEFAULT_SCHEMA = """\
The caption is a single fluent paragraph of 60-150 words, written in English,
present tense, third person, containing in natural narrative order:

1. Style anchor — open with the visual style (e.g. "Cinematic live-action shot
   of ...", "A 2D animated scene of ...").
2. Subject and appearance — the main subject with its specific visual
   attributes.
3. Action with temporal progression — what the subject does, as 2-3 causally
   connected beats ("... first A, then B, finally C").
4. Setting — location, key props, spatial layout.
5. Lighting and mood — light quality and the atmospheric tone it creates.
6. Camera — end with the camera framing and motion (e.g. "slow push-in",
   "handheld tracking shot").

Constraints:
- One paragraph, no line breaks, no field labels, no lists.
- Every visual fact from the extraction must appear; do not invent new
  subjects or events.
- Motion must be explicit — the subject or environment must visibly change
  over the clip, even when the camera itself is locked off.
"""


class T2VExtractStage(PromptGenerationOperator[Dict[str, Any], Dict[str, Any]]):
    """Stage 1: parse the user prompt into a structured JSON draft.

    Model-agnostic video parsing.  I2V rows (carrying ``first_frame_image``)
    use a dedicated template: the VLM first inventories what is visible in
    the conditioning frame (recorded as ``first_frame_inventory`` in the
    draft), then extracts only the evolution from that frame, with the image
    taking precedence over the prompt on conflicts.
    """

    name = "t2v_extract"
    default_max_tokens = 1024

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            template_key="extract_prompt_template",
            file_key="extract_prompt_template_file",
            default_file=_DEFAULT_EXTRACT_PROMPT_FILE,
        )
        self.i2v_prompt_template = resolve_prompt(
            self.config,
            template_key="i2v_extract_prompt_template",
            file_key="i2v_extract_prompt_template_file",
            default_file=_DEFAULT_I2V_EXTRACT_PROMPT_FILE,
        )
        self.first_frame_key = self.config.get("first_frame_key", "first_frame_image")

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            prompt = row.get("prompt") or row.get("optimized_prompt") or ""
            if not prompt:
                logger.warning("Row %d has empty prompt, skipping.", idx)
                continue
            first_frame_image = row.get(self.first_frame_key) or None
            if first_frame_image:
                formatted = format_prompt_safely(self.i2v_prompt_template, prompt=prompt)
                instruction: Any = build_multimodal_user_content(
                    formatted, [first_frame_image]
                )
            else:
                formatted = format_prompt_safely(self.prompt_template, prompt=prompt)
                instruction = formatted
            requests.append(
                GenerationRequest(
                    id=str(row.get("id", idx)),
                    instruction=instruction,
                    system_prompt=self.config.get("system_prompt"),
                    metadata={
                        "raw_prompt": prompt,
                        "first_frame_image": first_frame_image,
                        **{
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "prompt",
                                "optimized_prompt",
                                "id",
                                self.first_frame_key,
                            }
                        },
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        draft = extract_tagged_answer(result.response, "answer")
        if not draft:
            draft = result.response.strip()
        if not draft:
            return None
        meta = result.request.metadata
        output: Dict[str, Any] = {
            "id": result.request.id,
            "raw_prompt": meta.get("raw_prompt", ""),
            "draft": draft,
        }
        if meta.get("first_frame_image"):
            output["first_frame_image"] = meta["first_frame_image"]
        for k, v in meta.items():
            if k not in {"raw_prompt", "first_frame_image"}:
                output.setdefault(k, v)
        # `resolution: auto` (T2V rows only): resolve the aspect ratio from
        # the draft's LLM-inferred `aspect_ratio` field.  An explicit seed
        # `ratio` always wins; I2V rows follow their first frame instead.
        if (
            str(output.get("resolution", "")).lower() == "auto"
            and not output.get("first_frame_image")
            and not output.get("ratio")
        ):
            ratio = _draft_aspect_ratio(draft)
            if ratio:
                output["ratio"] = ratio
            else:
                logger.warning(
                    "Row %s requested resolution=auto but the draft carries no "
                    "valid aspect_ratio; generation will fall back to the "
                    "configured ratio.",
                    output.get("id"),
                )
        return output


class T2VComposeStage(PromptGenerationOperator[Dict[str, Any], Dict[str, Any]]):
    """Stage 2: rewrite the draft into the target model's caption schema.

    Text-only call — the first frame's visual understanding is already baked
    into the draft, so the compose model does not need to see the image.
    """

    name = "t2v_compose"
    default_max_tokens = 1024

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            template_key="compose_prompt_template",
            file_key="compose_prompt_template_file",
            default_file=_DEFAULT_COMPOSE_PROMPT_FILE,
        )
        # The caption schema defaults to a generic built-in; users supply
        # their video model's schema via schema_file / inline schema.
        self.schema = resolve_prompt(
            self.config,
            template_key="schema",
            file_key="schema_file",
            default=_DEFAULT_SCHEMA,
        )

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            draft = row.get("draft") or ""
            if not draft:
                logger.warning("Row %s has empty draft, skipping.", row.get("id", idx))
                continue
            formatted = format_prompt_safely(
                self.prompt_template,
                schema=self.schema,
                draft=draft,
                prompt=row.get("raw_prompt", ""),
            )
            requests.append(
                GenerationRequest(
                    id=str(row.get("id", idx)),
                    instruction=formatted,
                    system_prompt=self.config.get("system_prompt"),
                    metadata={k: v for k, v in row.items() if k != "id"},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        caption = extract_tagged_answer(result.response, "answer")
        if not caption:
            caption = result.response.strip()
        if not caption:
            return None
        meta = result.request.metadata
        output: Dict[str, Any] = {
            "id": result.request.id,
            "raw_prompt": meta.get("raw_prompt", ""),
            "draft": meta.get("draft", ""),
            "optimized_prompt": caption,
        }
        if meta.get("first_frame_image"):
            output["first_frame_image"] = meta["first_frame_image"]
        for k, v in meta.items():
            if k not in {"raw_prompt", "draft", "first_frame_image"}:
                output.setdefault(k, v)
        return output


class T2VPromptOptimizer:
    """Two-stage T2V/I2V prompt optimizer: extract draft -> compose caption.

    Exactly two model calls per row:
      1. extract — generic video parsing into a structured JSON ``draft``
         (VLM-grounded for I2V rows carrying ``first_frame_image``).
      2. compose — schema-adapted caption writing into ``optimized_prompt``.

    Configurable fields:
      - extract_prompt_template / extract_prompt_template_file: stage-1
        template with {prompt} (default configs/prompts/t2v_extract_prompt.txt).
      - i2v_extract_prompt_template / i2v_extract_prompt_template_file:
        stage-1 template for I2V rows (first-frame inventory -> evolution;
        default configs/prompts/i2v_extract_prompt.txt).
      - compose_prompt_template / compose_prompt_template_file: stage-2
        template with {schema} / {draft} / {prompt}
        (default configs/prompts/t2v_compose_prompt.txt).
      - schema / schema_file: target model's caption schema; write it from
        your video model's official prompt guideline (defaults to a generic
        built-in schema).
      - first_frame_key: row key of the conditioning image
        (default "first_frame_image").
      - compose_backend: optional separate ModelBackend for stage 2 (pass as
        constructor arg); defaults to the extract backend.
      - model_id, temperature, max_tokens, max_workers, show_progress and
        retry knobs are shared by both stages.

    Input: list of dicts with key ``prompt`` (and optional ``id``,
    ``first_frame_image``).
    Output: list of dicts with keys ``id``, ``raw_prompt``, ``draft``,
    ``optimized_prompt`` (plus ``first_frame_image`` passed through).
    """

    name = "t2v_prompt_optimize"

    def __init__(
        self,
        backend,
        config: Optional[Dict[str, Any]] = None,
        compose_backend: Optional[Any] = None,
    ):
        self.config = config or {}
        self.extract_stage = T2VExtractStage(backend=backend, config=self.config)
        self.compose_stage = T2VComposeStage(
            backend=compose_backend or backend, config=self.config
        )

    def run(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not inputs:
            return []
        drafts = self.extract_stage.run(inputs)
        logger.info(
            "T2V prompt optimize: extract produced %d drafts from %d rows.",
            len(drafts),
            len(inputs),
        )
        captions = self.compose_stage.run(drafts)
        logger.info(
            "T2V prompt optimize: compose produced %d captions from %d drafts.",
            len(captions),
            len(drafts),
        )
        return captions
