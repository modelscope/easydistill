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

"""Shared utilities for synthesis operators."""

import random
import re
from typing import List, Optional, Tuple


def sample_in_context_examples(
    pool: List[str],
    k: int,
    exclude: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample k distinct examples from a pool, optionally excluding one."""
    candidates = [item for item in pool if item != exclude] if exclude is not None else pool
    if k > len(candidates):
        raise ValueError(f"Cannot sample {k} examples from pool of size {len(candidates)}")
    rng = random.Random(seed)
    return rng.sample(candidates, k)


def extract_tagged_answer(text: str, tag: str = "answer") -> Optional[str]:
    """Extract content wrapped in <tag>...</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_instruction_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract <instruction> and <response> pairs from text."""
    instruction = extract_tagged_answer(text, "instruction")
    response = extract_tagged_answer(text, "response")
    return instruction, response


def format_in_context_examples(examples: List[str], prefix: str = "Example:") -> str:
    """Format a list of examples as a numbered string."""
    lines = []
    for idx, example in enumerate(examples, 1):
        lines.append(f"{prefix} {idx}\n{example}")
    return "\n\n".join(lines)
