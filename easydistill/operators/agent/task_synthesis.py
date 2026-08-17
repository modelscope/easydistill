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

"""Virtual task + tool-set synthesis operator."""

import json
import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely

from .base import PromptAgentOperator
from .utils import extract_tag

logger = logging.getLogger(__name__)


class AgentTaskSynthesisOperator(PromptAgentOperator[str, Dict[str, Any]]):
    """Synthesize a virtual tool-use task from a persona/background seed.

    Configurable fields:
      - prompt_template / prompt_template_file: template with {background_info}.
      - temperature, max_tokens, max_workers, show_progress.

    Output dict keys:
      - background: the original seed background.
      - task: high-level task description.
      - tools: list of tool JSON schemas.
      - workflow: high-level workflow text.
      - restriction: policy restriction text.
      - initial_toolset_create: raw model output.
    """

    name = "agent_task_synthesis"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            default_file="configs/prompts/agent_task_synthesis_prompt.txt",
        )

    def _build_requests(self, inputs: List[str]) -> List[GenerationRequest]:
        requests = []
        for idx, background in enumerate(inputs):
            prompt = format_prompt_safely(self.prompt_template, background_info=background)
            requests.append(
                GenerationRequest(
                    id=f"agent_task_{idx}",
                    instruction=prompt,
                    metadata={
                        "row_idx": idx,
                        "background": background,
                        "task": "agent_task_synthesis",
                    },
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        raw_response = result.response or ""
        background = result.request.metadata.get("background", "")

        task = extract_tag(raw_response, "task")
        tools_text = extract_tag(raw_response, "tools")
        workflow = extract_tag(raw_response, "workflow")
        restriction = extract_tag(raw_response, "restriction")

        tools: Any = None
        if tools_text:
            try:
                tools = json.loads(tools_text)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool JSON: %s", exc)

        if not task or not tools or not workflow or not restriction:
            logger.warning("Incomplete task synthesis result for request %s", result.request.id)
            return None

        return {
            "row_idx": result.request.metadata.get("row_idx"),
            "background": background,
            "task": task,
            "tools": tools,
            "workflow": workflow,
            "restriction": restriction,
            "initial_toolset_create": raw_response,
        }
