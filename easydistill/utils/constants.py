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

"""Shared default constants for operators and backends."""

# Generation defaults
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_TEMPERATURE: float = 0.7

# Concurrency defaults
DEFAULT_MAX_WORKERS: int = 1

# Retry defaults
DEFAULT_RETRY_ATTEMPTS: int = 3
DEFAULT_RETRY_BACKOFF_BASE: float = 1.0
DEFAULT_RETRY_MAX_WAIT: float = 30.0
