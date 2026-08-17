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

"""Shared helpers for the T2I / TI2I single-file evaluators.

The four evaluator modules (``t2i_single_model``, ``t2i_multi_model``,
``ti2i_single_model``, ``ti2i_multi_model``) share identical utility
functions for config reading, JSON parsing, score clamping and dimension-pool
loading.  This module is the single source of truth for those helpers; each
evaluator re-imports them so the public API stays backward-compatible.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

SCORE_MAP_5: Dict[int, float] = {0: 0.0, 1: 25.0, 2: 50.0, 3: 75.0, 4: 100.0}


def get_config_int(config: Dict[str, Any], key: str, default: int) -> int:
    """Read an int config value; only a missing/None key falls back to default."""
    value = config.get(key)
    return default if value is None else int(value)


def get_config_float(config: Dict[str, Any], key: str, default: float) -> float:
    """Read a float config value; only a missing/None key falls back to default."""
    value = config.get(key)
    return default if value is None else float(value)


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced ``{...}`` block from ``text``.

    Respects JSON string literals so braces inside quoted text do not affect
    balance.  Returns ``None`` if no complete object can be found.
    """
    start = text.find("{")
    if start < 0:
        return None

    in_string = False
    escaped = False
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            continue
        if ch == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_block(text: str) -> Dict[str, Any]:
    """Parse the first JSON object from a model response, tolerating fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    block = extract_first_json_object(text)
    if block is None:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    parsed = json.loads(block)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def clamp_score(value: Any) -> Optional[int]:
    """Clamp a numeric value to the 0-4 integer scale, or None on failure."""
    if value is None:
        return None
    try:
        return max(0, min(4, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def load_dimension_pool(
    default_path: Path,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a frozen dimension pool JSON and index L3 dims by L1 group.

    Args:
        default_path: fallback path when *path* is not given.
        path: optional explicit override path.

    Returns:
        ``{"l1_groups": {l1: [{name, criteria}, ...]}, "l3_to_l1": {...},
        "aggregation": {...}}``.
    """
    pool_path = Path(path) if path else default_path
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    l1_groups: Dict[str, List[Dict[str, Any]]] = {}
    l3_to_l1: Dict[str, str] = {}
    for l1 in data.get("dimensions") or []:
        l1_name = str(l1.get("name") or "")
        for l2 in l1.get("l2_groups") or []:
            for item in l2.get("items") or []:
                name = str(item.get("name") or "")
                if not name:
                    continue
                l1_groups.setdefault(l1_name, []).append(
                    {"name": name, "criteria": item.get("criteria") or {}}
                )
                l3_to_l1[name] = l1_name
    return {
        "l1_groups": l1_groups,
        "l3_to_l1": l3_to_l1,
        "aggregation": data.get("aggregation") or {},
    }


def approx_token_count(text: str) -> int:
    """Approximate token count using character-based heuristics (~4 chars/token)."""
    return max(1, len(text) // 4)
