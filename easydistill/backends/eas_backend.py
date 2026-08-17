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

"""PAI-EAS backend for self-deployed models."""

import logging
import os
from typing import Any, Dict, List, Optional

from easydistill.data.models import GenerationResult

from .openai_backend import OpenAIBackend

logger = logging.getLogger(__name__)


class EASBackend(OpenAIBackend):
    """Backend that calls a model deployed on PAI-EAS.

    PAI-EAS exposes OpenAI-compatible chat completion endpoints. Users provide
    the service endpoint URL and an access token. The endpoint URL should end
    with the API base path, e.g.:
      https://<service>-<id>.cn-beijing.pai-eas.aliyuncs.com/v1
    URLs ending in `/v1/chat/completions` are normalized automatically.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        raw_url = endpoint_url or os.getenv("EAS_ENDPOINT_URL")
        self.token = token or os.getenv("EAS_TOKEN")
        if not raw_url:
            raise ValueError("EAS backend requires 'endpoint_url' or EAS_ENDPOINT_URL env var.")
        if not self.token:
            raise ValueError("EAS backend requires 'token' or EAS_TOKEN env var.")
        # Normalize the endpoint URL: accept both .../v1 and .../v1/chat/completions.
        # URLs that already point to a service path (e.g. PAI-EAS predict URLs) keep
        # that path and have /v1 appended.
        normalized = raw_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1/chat/completions"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        self.endpoint_url = normalized
        # EAS uses the token as the API key for OpenAI-compatible endpoints.
        super().__init__(
            api_key=self.token,
            base_url=self.endpoint_url,
            model_id=model_id,
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
        # Tag the backend source in metadata.
        result.metadata["backend"] = "pai_eas"
        return result

    def health_check(self) -> bool:
        """Check EAS health.

        First try the OpenAI-compatible model list endpoint. If the EAS
        deployment does not expose ``/models``, fall back to a minimal chat
        completion probe. Any probe failure is treated as unhealthy so that
        configuration problems are caught before the pipeline runs.
        """
        try:
            self.client.models.list()
            return True
        except Exception as model_exc:  # noqa: BLE001
            logger.warning("EAS model-list health check failed: %s", model_exc)

        try:
            self.generate(messages=[{"role": "user", "content": "ping"}], max_tokens=1)
            return True
        except Exception as chat_exc:  # noqa: BLE001
            logger.warning("EAS chat-completion health check failed: %s", chat_exc)
            return False
