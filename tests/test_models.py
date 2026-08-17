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

"""Unit tests for data models."""

from easydistill.data.models import (
    GenerationRequest,
    GenerationResult,
    Message,
    SFTSample,
)


def test_message_model():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_sft_sample_from_instruction_response_with_system():
    sample = SFTSample.from_instruction_response(
        instruction="Q",
        response="A",
        system="SYS",
        metadata={"source": "test"},
    )
    assert len(sample.messages) == 3
    assert sample.messages[0].role == "system"
    assert sample.messages[0].content == "SYS"
    assert sample.messages[1].role == "user"
    assert sample.messages[2].role == "assistant"
    assert sample.metadata["source"] == "test"


def test_sft_sample_from_instruction_response_without_system():
    sample = SFTSample.from_instruction_response(instruction="Q", response="A")
    assert len(sample.messages) == 2
    assert sample.messages[0].role == "user"
    assert sample.messages[1].role == "assistant"


def test_generation_result_model():
    request = GenerationRequest(instruction="Q", id="r1")
    result = GenerationResult(request=request, response="A", model="teacher")
    assert result.request.instruction == "Q"
    assert result.response == "A"
    assert result.model == "teacher"
