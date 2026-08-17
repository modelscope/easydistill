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

"""Validate SFT output format compatibility with training frameworks."""

import json

from easydistill.data.models import SFTSample


class TestSFTFormat:
    def test_llamafactory_openai_format(self, tmp_path):
        """LLaMA-Factory accepts OpenAI messages as a special ShareGPT format."""
        sample = SFTSample.from_instruction_response(
            instruction="Q",
            response="A",
            system="SYS",
        )
        path = str(tmp_path / "sft.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")

        with open(path, encoding="utf-8") as f:
            loaded = json.loads(f.readline())

        assert "messages" in loaded
        roles = [m["role"] for m in loaded["messages"]]
        assert roles == ["system", "user", "assistant"]
        for msg in loaded["messages"]:
            assert "role" in msg
            assert "content" in msg

    def test_last_message_is_assistant(self):
        sample = SFTSample.from_instruction_response(instruction="Q", response="A")
        assert sample.messages[-1].role == "assistant"
