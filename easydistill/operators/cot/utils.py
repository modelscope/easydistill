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

"""Utilities for parsing CoT model outputs."""

import re
from typing import Optional, Tuple


def extract_between_tags(text: str, begin_tag: str, end_tag: str) -> Optional[str]:
    """Extract content between two literal tags."""
    if not text:
        return None
    pattern = re.compile(
        re.escape(begin_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_cot_sections(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract thought and solution sections from a CoT generation.

    Expected formats (tried in order):
      <|begin_of_thought|> ... <|end_of_thought|>
      <|begin_of_solution|> ... <|end_of_solution|>
    or the OmniThoughtV-style trace format:
      <thinking> ... </thinking>
      <answer> ... </answer>

    Returns (thought, solution). Missing sections are returned as None.
    """
    thought = extract_between_tags(text, "<|begin_of_thought|>", "<|end_of_thought|>")
    solution = extract_between_tags(text, "<|begin_of_solution|>", "<|end_of_solution|>")
    if thought is None and solution is None:
        thought = extract_between_tags(text, "<thinking>", "</thinking>")
        solution = extract_between_tags(text, "<answer>", "</answer>")
    return thought, solution
