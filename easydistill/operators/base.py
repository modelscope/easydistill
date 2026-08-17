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

"""Base operator class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Operator(ABC, Generic[InputType, OutputType]):
    """Atomic operator with clear input/output/config contract.

    Each operator does one thing: receives typed input, runs processing, and
    returns typed output together with metadata for observability.
    """

    name: str = "operator"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def run(self, input_data: InputType) -> OutputType:
        """Execute the operator."""
        raise NotImplementedError

    def __call__(self, input_data: InputType) -> OutputType:
        return self.run(input_data)
