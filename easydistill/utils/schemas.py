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

"""Pydantic validation schemas for EasyDistill 2 configs."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBackendConfig(BaseModel):
    """OpenAI-compatible backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai"] = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None


class PaiTokenBackendConfig(BaseModel):
    """PAI-Token backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pai_token"]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None


class PaiEASBackendConfig(BaseModel):
    """PAI-EAS backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pai_eas"]
    endpoint_url: Optional[str] = None
    token: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None


class WanxBackendConfig(BaseModel):
    """Tongyi Wanxiang (Wanx) T2I backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["wanx"]
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    poll_interval: Optional[float] = None
    max_poll_wait: Optional[float] = None


class PAIDiffusionBackendConfig(BaseModel):
    """PAI-EAS deployed diffusion (T2I) backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pai_diffusion"]
    endpoint_url: Optional[str] = None
    token: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    auth_prefix: Optional[str] = None
    output_dir: Optional[str] = None
    poll_interval: Optional[float] = None
    max_poll_wait: Optional[float] = None


class QwenImageBackendConfig(BaseModel):
    """Qwen-Image T2I backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["qwen_image"]
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    poll_interval: Optional[float] = None
    max_poll_wait: Optional[float] = None


BackendConfig = Union[
    OpenAIBackendConfig,
    PaiTokenBackendConfig,
    PaiEASBackendConfig,
]


T2IBackendConfig = Union[
    WanxBackendConfig,
    PAIDiffusionBackendConfig,
    QwenImageBackendConfig,
]


class PaiTokenVideoBackendConfig(BaseModel):
    """PAI-Token gateway video (T2V/I2V) backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pai_token_video"]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    i2v_model_id: Optional[str] = None
    i2v_image_field: Optional[str] = None
    submit_path: Optional[str] = None
    timeout: Optional[float] = None
    poll_interval: Optional[float] = None
    max_poll_wait: Optional[float] = None
    output_dir: Optional[str] = None


class PAIVideoBackendConfig(BaseModel):
    """PAI-EAS deployed video (T2V/I2V) backend configuration schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pai_video"]
    endpoint_url: Optional[str] = None
    token: Optional[str] = None
    model_id: Optional[str] = None
    timeout: Optional[float] = None
    auth_prefix: Optional[str] = None
    output_dir: Optional[str] = None
    poll_interval: Optional[float] = None
    max_poll_wait: Optional[float] = None
    protocol: Optional[str] = None
    t2v_task: Optional[str] = None
    i2v_task: Optional[str] = None
    sglang_short_edge: Optional[int] = None
    sglang_aspect_ratio: Optional[str] = None
    sglang_duration_seconds: Optional[float] = None


T2VBackendConfig = Union[
    PaiTokenVideoBackendConfig,
    PAIVideoBackendConfig,
]


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""

    model_config = ConfigDict(extra="forbid")

    input_path: str
    output_path: Optional[str] = None
    # Directory output used by standalone T2I/TI2I evaluators.
    output_dir: Optional[str] = None
    instruction_key: Optional[str] = None
    system_key: Optional[str] = None
    problem_key: Optional[str] = None
    answer_key: Optional[str] = None
    output_key: Optional[str] = None
    text_key: Optional[str] = None
    images_key: Optional[str] = None
    output_format: Optional[str] = None
    skip_empty: Optional[bool] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # T2I field
    prompt_key: Optional[str] = None


class GenerationConfig(BaseModel):
    """Generation configuration schema."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None
    prompt_template: Optional[str] = None
    prompt_template_file: Optional[str] = None
    raise_on_error: Optional[bool] = None
    retry_attempts: Optional[int] = None
    retry_backoff_base: Optional[float] = None
    retry_max_wait: Optional[float] = None
    # T2I fields
    model_id: Optional[str] = None
    size: Optional[str] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    infer_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    prompt_key: Optional[str] = None
    # T2V fields (new-protocol resolution control passed through to the
    # backend; legacy `size` above still applies).
    resolution: Optional[str] = None
    ratio: Optional[str] = None
    duration: Optional[float] = None
    watermark: Optional[bool] = None
    # sglang /v1/videos target spec (short_edge / aspect_ratio / ...).
    target: Optional[Dict[str, Any]] = None
    # I2V first-frame size guard (off | warn | skip | raise).
    i2v_frame_check: Optional[str] = None
    i2v_frame_min_edge: Optional[int] = None
    i2v_frame_max_aspect: Optional[float] = None


class SFTConfig(BaseModel):
    """SFT dataset builder configuration schema."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: Optional[str] = None
    # Per-language student system prompt files used by pe_rewrite_build_sft
    # (rows carry a zh/en language field from the plan step).
    system_prompt_zh_file: Optional[str] = None
    system_prompt_en_file: Optional[str] = None
    skip_empty: Optional[bool] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # Field mapping
    response_key: Optional[Union[str, List[str]]] = None
    images_key: Optional[str] = None
    # Deduplication: single key or list of keys (e.g. ["instruction"], ["instruction", "response"]).
    dedup_key: Optional[Union[str, List[str]]] = None
    # T2I fields
    min_prompt_length: Optional[int] = None
    max_images_per_prompt: Optional[int] = None
    # T2V fields
    max_videos_per_prompt: Optional[int] = None


class QualityFilterConfig(BaseModel):
    """Standalone quality filter stage configuration schema."""

    model_config = ConfigDict(extra="forbid")

    min_scores: Optional[Dict[str, Any]] = None
    keep_top_k: Optional[int] = None
    keep_top_ratio: Optional[float] = None
    require_all_metrics: Optional[bool] = None
    # PE rewrite filter: run the top selection per scene (default) so no
    # scene is evicted wholesale by global average-score ranking.
    per_scene: Optional[bool] = None


class EvalConfig(BaseModel):
    """Evaluation configuration schema."""

    model_config = ConfigDict(extra="forbid")

    metrics: Optional[List[str]] = Field(default_factory=list)
    prompts_file: Optional[str] = None
    # Judge model override so the judge can run on a different model than the
    # generation steps sharing the same backend endpoint.
    model_id: Optional[str] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    raise_on_error: Optional[bool] = None
    strict_mode: Optional[bool] = None
    # T2I/TI2I single-model evaluator fields.
    synthesize_reasons: Optional[bool] = None
    call_retries: Optional[int] = None
    retry_delay_sec: Optional[float] = None
    # T2I/TI2I multi-model evaluator fields.
    conflict_threshold: Optional[int] = None
    max_debate_dims: Optional[int] = None
    # T2V composable checker chain (vbench / vlm / omni entries, each a
    # free-form dict consumed by T2VVideoEvaluator).
    checkers: Optional[List[Dict[str, Any]]] = None


class SynthesisConfig(BaseModel):
    """Synthesis operator configuration schema."""

    model_config = ConfigDict(extra="forbid")

    prompt_template: Optional[str] = None
    prompt_template_file: Optional[str] = None
    num_in_context_samples: Optional[int] = None
    num_output_samples: Optional[int] = None
    seed: Optional[int] = None
    rounds: Optional[int] = None
    generations_per_round: Optional[int] = None
    round_retry_attempts: Optional[int] = None
    first_message_template: Optional[str] = None
    followup_message_template: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None


class AgenticRewriteStepConfig(BaseModel):
    """Per-step overrides for the agentic prompt rewrite operator.

    ``prompt_template(_file)`` applies to the plan and reflection steps;
    ``scene_prompt_dir`` applies to the rewrite step only (one full system
    prompt per scene and language: ``rewrite_{scene}_{lang}.txt``).
    """

    model_config = ConfigDict(extra="forbid")

    prompt_template: Optional[str] = None
    prompt_template_file: Optional[str] = None
    scene_prompt_dir: Optional[str] = None
    message_template: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgenticRewriteConfig(BaseModel):
    """Agentic prompt rewrite (plan -> rewrite -> reflection) configuration."""

    model_config = ConfigDict(extra="forbid")

    plan: AgenticRewriteStepConfig = Field(default_factory=AgenticRewriteStepConfig)
    rewrite: AgenticRewriteStepConfig = Field(default_factory=AgenticRewriteStepConfig)
    reflection: AgenticRewriteStepConfig = Field(default_factory=AgenticRewriteStepConfig)
    model_id: Optional[str] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None
    retry_attempts: Optional[int] = None
    retry_backoff_base: Optional[float] = None
    retry_max_wait: Optional[float] = None
    # Optional JSONL sink written incrementally (completion order) so an
    # interrupted run keeps its partial output.
    stream_output_path: Optional[str] = None


class CotConfig(BaseModel):
    """CoT operator configuration schema."""

    model_config = ConfigDict(extra="forbid")

    prompt_template: Optional[str] = None
    prompt_template_file: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None


class MMConfig(BaseModel):
    """Multi-modal operator configuration schema."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    prompt_template_file: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None


class PreferenceConfig(BaseModel):
    """Preference data build configuration schema.

    Fields set here are used as defaults for the four preference pipeline
    stages. Each stage can override them in its own `config` block.
    """

    model_config = ConfigDict(extra="forbid")

    scorer: Optional[str] = None
    n: Optional[int] = None
    metrics: Optional[List[str]] = Field(default_factory=list)
    min_margin: Optional[float] = None
    max_pairs_per_prompt: Optional[int] = None
    require_chosen_correct: Optional[bool] = None
    instruction_key: Optional[str] = None
    answer_key: Optional[str] = None
    system_key: Optional[str] = None
    format: Optional[str] = None
    alpha: Optional[float] = None
    normalize_answer: Optional[bool] = None
    # PreferenceDatasetBuilder filter fields
    system_prompt: Optional[str] = None
    skip_empty: Optional[bool] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    raise_on_error: Optional[bool] = None


class BalanceConfig(BaseModel):
    """Instruction balancing / curriculum planning configuration schema."""

    model_config = ConfigDict(extra="forbid")

    instruction_key: Optional[str] = None
    category_key: Optional[str] = None
    categories: Optional[List[str]] = None
    target_distribution: Optional[Dict[str, float]] = None
    category_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    max_workers: Optional[int] = None
    show_progress: Optional[bool] = None
    seed: Optional[int] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    raise_on_error: Optional[bool] = None


class AgentConfig(BaseModel):
    """Agent distillation configuration schema.

    These defaults are used when a pipeline stage does not override them.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps: Optional[int] = 10
    repeat_times: Optional[int] = 2
    max_tool_calls: Optional[int] = 20
    use_rubrics: Optional[bool] = True


class PipelineStageConfig(BaseModel):
    """Single pipeline stage configuration schema."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    config: Dict[str, Any] = Field(default_factory=dict)
    output_path: Optional[str] = None


class TeacherConfig(BaseModel):
    """Teacher entry used by T2I/TI2I standalone evaluators.

    ``name`` is the public teacher label; all other keys are passed through
    to the backend factory (e.g. ``type``, ``api_key``, ``model_id``).
    """

    model_config = ConfigDict(extra="allow")

    name: str


class AppConfig(BaseModel):
    """Top-level application configuration schema.

    Top-level extra keys are allowed so users can add extension sections or
    forward-compatible fields without breaking validation. All nested
    configuration sections (backend, dataset, generation, etc.) are strict.
    """

    model_config = ConfigDict(extra="allow")

    job_type: str = "instruct_distill"
    # Local-only jobs never touch a model service, so they may omit the
    # backend section; every other job type still requires it (validated
    # below), preserving the original fail-fast contract.
    backend: Optional[BackendConfig] = Field(None, discriminator="type")
    t2i_backend: Optional[T2IBackendConfig] = Field(None, discriminator="type")
    t2v_backend: Optional[T2VBackendConfig] = Field(None, discriminator="type")
    eval_backend: Optional[BackendConfig] = Field(None, discriminator="type")
    # T2I/TI2I standalone evaluators declare their own teacher pool instead
    # of a single backend.
    teachers: Optional[List[TeacherConfig]] = None
    arbiter: Optional[Dict[str, Any]] = None
    dataset: DatasetConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    quality_filter: QualityFilterConfig = Field(default_factory=QualityFilterConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    agentic_rewrite: AgenticRewriteConfig = Field(default_factory=AgenticRewriteConfig)
    cot: CotConfig = Field(default_factory=CotConfig)
    mm: MMConfig = Field(default_factory=MMConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    preference: PreferenceConfig = Field(default_factory=PreferenceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    pipeline: List[PipelineStageConfig] = Field(default_factory=list)

    # Jobs that run purely on local files, without any model backend.
    _LOCAL_JOBS = frozenset({"pe_rewrite_filter", "pe_rewrite_build_sft"})

    # Jobs that provide their own teacher pool for evaluation.
    _TEACHER_JOBS = frozenset({
        "t2i_single_model_eval",
        "t2i_multi_model_eval",
        "ti2i_single_model_eval",
        "ti2i_multi_model_eval",
    })

    @model_validator(mode="after")
    def _require_some_backend(self) -> "AppConfig":
        if self.job_type in self._LOCAL_JOBS:
            return self
        if self.job_type in self._TEACHER_JOBS:
            if not self.teachers:
                raise ValueError(
                    f"Config for job_type '{self.job_type}' must define "
                    f"'teachers' (only local jobs {sorted(self._LOCAL_JOBS)} "
                    "may omit a backend)."
                )
            return self
        if self.backend is None and self.t2i_backend is None and self.t2v_backend is None:
            raise ValueError(
                f"Config for job_type '{self.job_type}' must define at least "
                "one of 'backend', 't2i_backend' or 't2v_backend' (only "
                f"local jobs {sorted(self._LOCAL_JOBS)} may omit them)."
            )
        return self


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a raw config dict and return it with defaults filled in.

    Uses ``exclude_none=True`` so that optional fields not set by the user
    are omitted from the output.  This ensures that downstream operators
    which use ``config.get("field", default)`` correctly fall back to their
    own defaults instead of receiving explicit ``None`` values.
    """
    return AppConfig(**config).model_dump(exclude_none=True)
