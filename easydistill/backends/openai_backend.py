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

"""OpenAI-compatible backend supporting PAI-Token, EAS, and OpenAI endpoints."""

import logging
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationResult

from .base import ModelBackend
from .utils import build_generation_request

logger = logging.getLogger(__name__)


try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore


class OpenAIBackend(ModelBackend):
    """Backend that calls any OpenAI-compatible chat completion endpoint.

    This covers:
      - PAI-Token (base_url=https://cn-beijing.pai-token.aliyuncs.com/v1)
      - PAI-EAS custom endpoints
      - OpenAI API
      - Local servers such as vLLM or llama.cpp
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 0,
    ):
        if not _HAS_OPENAI:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with: pip install openai"
            )
        # By default max_retries is 0 so that operators (e.g. TextGenerationOperator)
        # own the retry/backoff policy and avoid compounding delays. If a user
        # explicitly sets backend.max_retries, that value is passed through to the
        # underlying OpenAI client instead.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model_id = model_id

    def _resolve_model(self, model_id: Optional[str]) -> str:
        if model_id is not None:
            return model_id
        if self.model_id is not None:
            return self.model_id
        # Try to list available models and pick the first one.
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as exc:
            logger.warning("Failed to list models: %s", exc)
        raise ValueError("No model_id provided and backend could not determine a default model.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> GenerationResult:
        model = self._resolve_model(model_id)
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if not completion.choices:
            logger.warning("Backend returned no completion choices for model %s.", model)
            content = ""
        else:
            content = completion.choices[0].message.content or ""
        request = build_generation_request(messages, model=model)
        return GenerationResult(
            request=request,
            response=content,
            model=model,
            usage=completion.usage.model_dump() if completion.usage else None,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception as exc:
            logger.error("Failed to close OpenAI client: %s", exc)

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:
            logger.warning("Backend health check failed: %s", exc)
            return False
