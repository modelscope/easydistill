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

"""Shared metric/score helpers used across pipelines and operators."""

from typing import Any, Dict, List, Optional


def compute_average_score(row: Dict[str, Any], metrics: List[str]) -> Optional[float]:
    """Compute the average of selected metrics from a row.

    Boolean values are treated as 1.0 (True) or 0.0 (False). Other values are
    cast to float. Missing or non-numeric values are skipped.

    Returns None if no metric values are available.
    """
    values = []
    for metric in metrics:
        value = row.get(metric)
        if value is None:
            continue
        try:
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            else:
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None
