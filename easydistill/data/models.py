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

"""Core data models for EasyDistill 2."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message in OpenAI-compatible format."""

    role: str = Field(..., description="One of system/user/assistant/tool.")
    content: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="Text or multi-modal content of the message."
    )


class GenerationRequest(BaseModel):
    """Request to generate a teacher response for a seed instruction."""

    id: Optional[str] = Field(default=None, description="Optional request identifier.")
    instruction: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="Seed instruction from the user (text or multi-modal content)."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt override."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class GenerationResult(BaseModel):
    """Result of a teacher generation."""

    request: GenerationRequest = Field(..., description="Original request.")
    response: str = Field(..., description="Generated teacher response.")
    model: Optional[str] = Field(default=None, description="Model identifier.")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage info.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class SFTSample(BaseModel):
    """A single SFT training sample in sharegpt/messages format."""

    messages: List[Message] = Field(..., description="Conversation messages.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sample metadata.")

    @classmethod
    def from_instruction_response(
        cls,
        instruction: Union[str, List[Dict[str, Any]]],
        response: str,
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SFTSample":
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=instruction))
        messages.append(Message(role="assistant", content=response))
        return cls(messages=messages, metadata=metadata or {})

    @classmethod
    def from_prompt_image(
        cls,
        prompt: str,
        image_url: str,
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SFTSample":
        """Build a T2I SFT sample: user=prompt, assistant=image (multi-modal)."""
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        assistant_content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        messages.append(Message(role="assistant", content=assistant_content))
        return cls(messages=messages, metadata=metadata or {})

    @classmethod
    def from_prompt_video(
        cls,
        prompt: str,
        video_url: str,
        first_frame_image: Optional[str] = None,
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SFTSample":
        """Build a T2V/I2V SFT sample: user=prompt (+optional first frame),
        assistant=video (multi-modal).

        When ``first_frame_image`` is provided (I2V mode), the user message
        becomes a multi-modal content list carrying the prompt text and the
        conditioning first-frame image.
        """
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        if first_frame_image:
            user_content: List[Dict[str, Any]] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": first_frame_image}},
            ]
            messages.append(Message(role="user", content=user_content))
        else:
            messages.append(Message(role="user", content=prompt))
        assistant_content: List[Dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_url}}
        ]
        messages.append(Message(role="assistant", content=assistant_content))
        return cls(messages=messages, metadata=metadata or {})


class PreferenceSample(BaseModel):
    """A single preference training sample in DPO format."""

    prompt: List[Message] = Field(..., description="Prompt messages.")
    chosen: List[Message] = Field(..., description="Preferred assistant messages.")
    rejected: List[Message] = Field(..., description="Dispreferred assistant messages.")
    chosen_score: Optional[float] = Field(default=None, description="Score of the chosen response.")
    rejected_score: Optional[float] = Field(
        default=None, description="Score of the rejected response."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sample metadata.")

    @classmethod
    def from_instruction_responses(
        cls,
        instruction: Union[str, List[Dict[str, Any]]],
        chosen: str,
        rejected: str,
        system: Optional[str] = None,
        chosen_score: Optional[float] = None,
        rejected_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PreferenceSample":
        prompt = []
        if system:
            prompt.append(Message(role="system", content=system))
        prompt.append(Message(role="user", content=instruction))
        chosen_messages = [Message(role="assistant", content=chosen)]
        rejected_messages = [Message(role="assistant", content=rejected)]
        return cls(
            prompt=prompt,
            chosen=chosen_messages,
            rejected=rejected_messages,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            metadata=metadata or {},
        )


class ImageGenerationResult(BaseModel):
    """Result of a T2I (text-to-image) generation call."""

    prompt: str = Field(..., description="The prompt sent to the T2I model.")
    image_urls: List[str] = Field(
        default_factory=list, description="Generated image URLs or base64 data URLs."
    )
    model: Optional[str] = Field(default=None, description="Model identifier.")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token/credit usage info.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class T2ISample(BaseModel):
    """Intermediate T2I data row passed between pipeline stages.

    Tracks the full provenance from raw seed prompt to optimized prompt to
    generated images and evaluation scores.
    """

    id: str = Field(..., description="Sample identifier.")
    raw_prompt: str = Field("", description="Original seed prompt.")
    optimized_prompt: str = Field(
        "", description="Prompt after optimization (may equal raw_prompt)."
    )
    image_urls: List[str] = Field(default_factory=list, description="Generated image URLs.")
    system: Optional[str] = Field(default=None, description="Optional system prompt.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata / scores.")


class VideoGenerationResult(BaseModel):
    """Result of a T2V/I2V (text-to-video / image-to-video) generation call."""

    prompt: str = Field(..., description="The prompt sent to the T2V model.")
    video_urls: List[str] = Field(
        default_factory=list, description="Generated video URLs or local file paths."
    )
    first_frame_image: Optional[str] = Field(
        default=None, description="Conditioning first-frame image URL (I2V mode only)."
    )
    model: Optional[str] = Field(default=None, description="Model identifier.")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token/credit usage info.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class T2VSample(BaseModel):
    """Intermediate T2V/I2V data row passed between pipeline stages.

    Tracks the full provenance from raw seed prompt (and optional first-frame
    image) to optimized prompt to generated videos and evaluation scores.
    """

    id: str = Field(..., description="Sample identifier.")
    raw_prompt: str = Field("", description="Original seed prompt.")
    optimized_prompt: str = Field(
        "", description="Prompt after optimization (may equal raw_prompt)."
    )
    first_frame_image: Optional[str] = Field(
        default=None, description="Conditioning first-frame image (I2V mode only)."
    )
    video_urls: List[str] = Field(default_factory=list, description="Generated video URLs.")
    system: Optional[str] = Field(default=None, description="Optional system prompt.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata / scores.")
