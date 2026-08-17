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

"""Tool validation / refinement operator."""

import json
import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely

from .base import PromptAgentOperator
from .utils import extract_tag

logger = logging.getLogger(__name__)


class AgentToolCheckOperator(PromptAgentOperator[Dict[str, Any], Dict[str, Any]]):
    """Review and refine virtual tools for LLM-simulatability.

    Input rows must contain ``fuzzy_task`` and ``tools``.

    Configurable fields:
      - prompt_template / prompt_template_file.
      - temperature, max_tokens, max_workers, show_progress.

    Output row is enriched with:
      - checked_tools: validated list of tool JSON schemas.
      - raw_tool_check: raw model output.
    """

    name = "agent_tool_check"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            default_file="configs/prompts/agent_tool_check_prompt.txt",
        )

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            prompt = format_prompt_safely(
                self.prompt_template,
                task_description=row.get("fuzzy_task", ""),
                tool_description=json.dumps(row.get("tools", []), ensure_ascii=False, indent=2),
            )
            requests.append(
                GenerationRequest(
                    id=f"agent_toolcheck_{idx}",
                    instruction=prompt,
                    metadata={"row_idx": idx, "task": "agent_tool_check"},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        raw_response = result.response or ""
        tools_text = extract_tag(raw_response, "tools")

        if not tools_text:
            logger.warning("No tools found in tool-check result for %s", result.request.id)
            return None

        try:
            checked_tools = json.loads(tools_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse checked tools JSON: %s", exc)
            return None

        if not isinstance(checked_tools, list):
            logger.warning("Checked tools is not a list for %s", result.request.id)
            return None

        return {
            "row_idx": result.request.metadata.get("row_idx"),
            "checked_tools": checked_tools,
            "raw_tool_check": raw_response,
        }
