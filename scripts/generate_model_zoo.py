#!/usr/bin/env python3
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

"""Regenerate docs/model_zoo.md and docs/model_zoo_zh.md from easydistill/models/model_zoo.yaml."""

import os
import sys

# Allow running from repo root without installing the package.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from easydistill.models.zoo import (  # noqa: E402, I001
    render_model_zoo_markdown,
)


_OUTPUT_DIR = os.path.join(repo_root, "docs")
_OUTPUT_FILES = {
    "en": os.path.join(_OUTPUT_DIR, "model_zoo.md"),
    "zh": os.path.join(_OUTPUT_DIR, "model_zoo_zh.md"),
}


def main() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    for lang, output_path in _OUTPUT_FILES.items():
        markdown = render_model_zoo_markdown(lang=lang)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
