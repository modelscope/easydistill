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

"""PAI-Token backend for OpenAI-compatible model calls."""

import logging
import os
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationResult

from .openai_backend import OpenAIBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://cn-beijing.pai-token.aliyuncs.com/v1"


class PaiTokenBackend(OpenAIBackend):
    """Backend that calls the PAI-Token service.

    PAI-Token exposes an OpenAI-compatible chat completion endpoint. Users
    must provide an API key and a model ID (for example, `kimi-k2.6`,
    `qwen2.5-72b-instruct`, or a custom deployed model name). The model ID
    can be set via the `model_id` argument or the `PAI_TOKEN_MODEL_ID`
    environment variable. No implicit default is used.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("PAI_TOKEN_API_KEY")
        self.base_url = base_url or os.getenv("PAI_TOKEN_BASE_URL", DEFAULT_BASE_URL)
        if not self.api_key:
            raise ValueError("PAI-Token backend requires 'api_key' or PAI_TOKEN_API_KEY env var.")
        if not self.base_url:
            raise ValueError("PAI-Token backend requires 'base_url' or PAI_TOKEN_BASE_URL env var.")
        resolved_model_id = model_id or os.getenv("PAI_TOKEN_MODEL_ID")
        if not resolved_model_id:
            raise ValueError(
                "PAI-Token backend requires a 'model_id' or PAI_TOKEN_MODEL_ID env var."
            )
        super().__init__(
            api_key=self.api_key,
            base_url=self.base_url,
            model_id=resolved_model_id,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        result = super().generate(
            messages=messages,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        result.metadata["backend"] = "pai_token"
        return result

    def health_check(self) -> bool:
        # PAI-Token does not expose the OpenAI model list endpoint, so treat the
        # backend as healthy if we have an API key and base URL.
        return bool(self.api_key and self.base_url)
