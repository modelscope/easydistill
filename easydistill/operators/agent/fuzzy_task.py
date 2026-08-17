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

"""Fuzzy task + background generation operator."""

import json
import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationRequest, GenerationResult
from easydistill.prompts import resolve_prompt
from easydistill.utils import format_prompt_safely

from .base import PromptAgentOperator
from .utils import extract_tag

logger = logging.getLogger(__name__)


class AgentFuzzyTaskOperator(PromptAgentOperator[Dict[str, Any], Dict[str, Any]]):
    """Convert a detailed task specification into a fuzzy task + background.

    Input rows must contain ``task``, ``tools``, and ``workflow``.

    Configurable fields:
      - prompt_template / prompt_template_file: template with {initial_task_info}.
      - temperature, max_tokens, max_workers, show_progress.

    Output row is enriched with:
      - fuzzy_task: concise task description.
      - task_background: detailed background for the agent.
    """

    name = "agent_fuzzy_task"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        super().__init__(backend, config)
        self.prompt_template = resolve_prompt(
            self.config,
            default_file="configs/prompts/agent_fuzzy_task_prompt.txt",
        )

    def _build_requests(self, inputs: List[Dict[str, Any]]) -> List[GenerationRequest]:
        requests = []
        for idx, row in enumerate(inputs):
            initial_task_info = {
                "task": row.get("task", ""),
                "tools": row.get("tools", []),
                "workflow": row.get("workflow", ""),
            }
            prompt = format_prompt_safely(
                self.prompt_template,
                initial_task_info=json.dumps(initial_task_info, ensure_ascii=False, indent=2),
            )
            requests.append(
                GenerationRequest(
                    id=f"agent_fuzzy_{idx}",
                    instruction=prompt,
                    metadata={"row_idx": idx, "task": "agent_fuzzy_task"},
                )
            )
        return requests

    def _parse_result(self, result: GenerationResult) -> Optional[Dict[str, Any]]:
        raw_response = result.response or ""
        fuzzy_task = extract_tag(raw_response, "task")
        background = extract_tag(raw_response, "background")

        if not fuzzy_task or not background:
            logger.warning("Incomplete fuzzy task result for request %s", result.request.id)
            return None

        return {
            "row_idx": result.request.metadata.get("row_idx"),
            "fuzzy_task": fuzzy_task,
            "task_background": background,
            "raw_fuzzy_task": raw_response,
        }
