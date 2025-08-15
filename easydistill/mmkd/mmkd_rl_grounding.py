# Copyright 2025 The HuggingFace Team, Alibaba Group Holding Limited. All rights reserved.
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

# Reference:
# - Paper: https://huggingface.co/papers/2507.15846
# - Code:  https://github.com/ZJU-REAL/GUI-G2/

import os
import re
import json
import math
import copy
import random
import logging
import warnings
import textwrap
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union, Sized

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from filelock import FileLock
from packaging import version
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

# Transformers and related
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
from transformers.utils import is_peft_available

# TRL-related imports
from trl import (
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

# Datasets
from datasets import Dataset, IterableDataset

# Accelerate
from accelerate.utils import is_peft_model, set_seed

# PEFT (if available)
if is_peft_available():
    from peft import PeftConfig, get_peft_model

# Weights & Biases (if available)
if is_wandb_available():
    import wandb

# Custom/local imports
from torch.utils.data import Dataset, Sampler
import yaml

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import torch

import ast

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type alias
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class VLMBaseModule(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_vlm_key(self):
        pass

    @abstractmethod
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        pass

    def post_model_init(self, model, processing_class):
        pass

    def is_embeds_input(self):
        return False
    
    @abstractmethod
    def get_processing_class(self):
        pass

    @abstractmethod
    def get_vision_modules_keywords(self):
        pass

    @abstractmethod
    def get_custom_multimodal_keywords(self):
        pass

    @abstractmethod
    def get_non_generate_params(self):
        pass

    @abstractmethod
    def get_custom_processing_keywords(self):
        pass

    @abstractmethod
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        pass
    
    @abstractmethod
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors, padding, padding_side, add_special_tokens):
        pass

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        else:
            model_cls = Qwen2_5_VLForConditionalGeneration
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        padding_side="left"
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
        return rewards


@dataclass
class GRPOConfig(TrainingArguments):

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and "
            "the hardware support this feature."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )

class RepeatRandomSampler(Sampler):

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
   
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        model_init_kwargs = args.model_init_kwargs or {}

        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params}")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, padding_side="left", trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            processing_class.tokenizer.padding_side  = 'left'
            
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            processing_class.tokenizer.padding_side  = 'left'
            if getattr(processing_class, "tokenizer",  None) is not None:

                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.tokenizer.padding_side  = 'left'
                
                if hasattr(processing_class.tokenizer, 'padding_side'):
                    processing_class.tokenizer.padding_side = 'left'
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
                processing_class.padding_side  = 'left'
        processing_class.tokenizer.padding_side  = 'left'

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_class.padding_side = "left"
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
            padding_side="left",
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon = args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        processing_class.padding_side="left"
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        # print("prompt:------------------------------",prompts)
        self.processing_class.tokenizer.padding_side = "left"
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        # Handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]

            for img in imgs:
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)
                

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            )
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():

            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                if output_reward_func and isinstance(output_reward_func[0], tuple):
        
                    reward_values = [r[1] for r in output_reward_func]  
                    reward_types = [r[0] for r in output_reward_func]   
       
                    unique_types = set(reward_types)
                    
            
                    if not hasattr(self, '_reward_by_type'):
                        self._reward_by_type = {}
                    
              
                    for reward_type in unique_types:
                
                        indices = [i for i, t in enumerate(reward_types) if t == reward_type]
                        type_values = [reward_values[i] for i in indices]
                        
                        type_tensor = torch.tensor(type_values, dtype=torch.float32, device=device)
                        
                        if reward_type not in self._reward_by_type:
                            self._reward_by_type[reward_type] = []
                        
                        self._reward_by_type[reward_type].append({
                            'tensor': type_tensor,
                            'type': reward_type
                        })
                    rewards_per_func[:, i] = torch.tensor(reward_values, dtype=torch.float32, device=device)
                else:
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        gpu_id = torch.cuda.current_device()

        rewards = rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
   
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()

        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            
        if hasattr(self, '_reward_by_type'):

            local_data = []
            for reward_type, data_list in self._reward_by_type.items():
                for data in data_list:
                    local_data.append({
                        'type': reward_type,
                        'tensor': data['tensor']
                    })
            
            all_data = self.accelerator.gather_for_metrics(local_data)
            
            type_to_tensors = {}
            for data in all_data:
                reward_type = data['type']
                if reward_type not in type_to_tensors:
                    type_to_tensors[reward_type] = []
                # type_to_tensors[reward_type].append(data['tensor'])
                tensor = data['tensor'].to(device)
                type_to_tensors[reward_type].append(tensor)

            for reward_type, tensors in type_to_tensors.items():
                if tensors:

                    all_values = torch.cat(tensors)
                    mean_value = all_values.mean().item()
                    
                    metric_name = f"rewards/{reward_type}"
                    if metric_name not in self._metrics:
                        self._metrics[metric_name] = []
                    self._metrics[metric_name].append(mean_value)  

        if hasattr(self, '_reward_by_type'):
            delattr(self, '_reward_by_type')

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def _get_train_sampler(self,_) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            logging.info(logs)
        self._metrics.clear()


def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        # q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0))
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["gaussian_point","gaussian_plane","format"],
        metadata={"help": "List of reward functions. Possible values: 'point', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "user", "content": example["instruction"]},
                ],
            }
     
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": example["instruction"]},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image_path' in example:
            image_path = example['image_path']
            # In case the image is not found
            attempts = 0
            max_attempts = 5
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
                attempts += 1
                if attempts==5:
                    break
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, Image.UnidentifiedImageError) as e:
                logger.warning(f"Could not open or identify image {image_path}: {e}")
                image = None
        else:
            image = None
        return {
            'image': image,
            'image_path': image_path,
            'problem': example['instruction'],
            'solution': example['abs_box'],
            'prompt': make_conversation_image(example)['prompt'],
        }

def gaussian_plane_reward(completions, solution, **kwargs):
    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps   = 1e-8
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        pred_width = pred_x2 - pred_x1
        pred_height = pred_y2 - pred_y1
        # pred_μ
        pred_mu = np.array([pred_center_x, pred_center_y])

        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        # gt_μ
        gt_mu = np.array([gt_center_x, gt_center_y])
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1

        # 1 sigma
        pred_sigma_x = pred_width * alpha
        pred_sigma_y = pred_height * alpha
        gt_sigma_x   = gt_width * alpha
        gt_sigma_y = gt_height * alpha

        pred_cov = np.array([[pred_sigma_x**2, 0], 
                            [0, pred_sigma_y**2]])
        
        # Σ2 (ground truth distribution covariance matrix)  
        gt_cov = np.array([[gt_sigma_x**2, 0], 
                        [0, gt_sigma_y**2]])
        
        sigma_avg = (pred_cov + gt_cov) / 2
        # 
        mu_diff = pred_mu - gt_mu
        
        # (1/8) * (μ1 - μ2)^T * Σ^(-1) * (μ1 - μ2)
        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1/8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))
        
        # (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))
        det_sigma_avg = np.linalg.det(sigma_avg)
        det_pred_cov = np.linalg.det(pred_cov)
        det_gt_cov = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_pred_cov * det_gt_cov + eps)))
        except:
            return 0.0
        bhattacharyya_distance = term1 + term2

        # 转换为奖励
        plane_reward = np.exp(-bhattacharyya_distance)
        plane_reward = round(plane_reward,3)
        return plane_reward

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        content = content.split('assistant\n')[-1]
        bbox_match = re.search(bbox_pattern, content.strip(), re.DOTALL)
        try:
            if bbox_match:
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                sol = [float(num) for num in sol]
                reward = g_plane_reward(bbox, sol)
        except Exception:
            print(Exception, content, sol)
            pass  
        
        rewards.append(reward)
    return rewards


def gaussian_point_reward(completions, solution, **kwargs):
    def g_point_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        
        sigma_x = alpha * gt_width
        sigma_y = alpha * gt_height

        x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
        y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
        exponent = -0.5 * (x_term + y_term)
        point_reward = math.exp(exponent)
        point_reward = round(point_reward,3)
        return point_reward

    contents = [completion[0]["content"] for completion in completions]

    rewards = []
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        content = content.split('assistant\n')[-1]
        bbox_match = re.search(bbox_pattern, content.strip(), re.DOTALL)
        try:
            if bbox_match:
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                sol = [float(num) for num in sol]
                reward = g_point_reward(bbox, sol)

        except Exception:
            print(Exception, content, sol)
            pass  
        
        rewards.append(reward)
       
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
    completion_contents = [completion[0]["content"] for completion in completions]

    matches = [re.fullmatch(pattern, content.split('assistant\n')[-1], re.DOTALL) for content in completion_contents]
                
    return [1.0 if match else 0.0 for match in matches]

def point_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific JSON format."""
    rewards = []
    
    for completion in completions:
        try:

            content = completion[0]["content"]
            json_str = content.split('assistant\n')[-1].strip()
            
            json_str=json_str.replace("```json","").replace("```","")
            
            parsed_data = json.loads(json_str)
            
            if not isinstance(parsed_data, list):
                rewards.append(0.0)
                continue
                
            valid = True
            for item in parsed_data:
                    
                point_2d = item["Point_2d"]
                if not isinstance(point_2d, list) or len(point_2d) != 2 or not all(isinstance(x, int) for x in point_2d):
                    valid = False
                    break
            
            rewards.append(1.0 if valid else 0.0)
            
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            rewards.append(0.0)
    
    return rewards

def multi_box_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific JSON format."""
    rewards = []
    
    for completion in completions:
        try:
            content = completion[0]["content"]
            response = eval(content.split('assistant\n')[-1].strip())
            for kk in [0,1,2,3]:
                if not isinstance(response[kk], list):
                    raise ValueError("Wrong Format")
 
            rewards.append(1.0)
            
        except (SyntaxError, NameError,json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            rewards.append(0.0)
    
    return rewards

def iou(box1, box2):
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = area1 + area2 - inter_area
    iou_value = inter_area / union_area if union_area > 0 else 0
    
    return iou_value

def match_boxes(predicted_boxes, target_boxes, iou_threshold=0.5):
    """
    Match predicted boxes with target boxes using the Hungarian algorithm.

    Args:
        predicted_boxes: Dictionary of predicted boxes {class_id: [boxes]}
        target_boxes: Dictionary of target boxes {class_id: [boxes]}
        iou_threshold: IoU threshold for matching

    Returns:
        matches: Dictionary of matching results {class_id: [(pred_idx, target_idx, iou_value)]}
        unmatched_preds: Unmatched predicted box indices {class_id: [pred_indices]}
        unmatched_targets: Unmatched target box indices {class_id: [target_indices]}
    """
    
    matches = {}
    unmatched_preds = {}
    unmatched_targets = {}
    
    all_class_ids = set(list(predicted_boxes.keys()) + list(target_boxes.keys()))
    
    for class_id in all_class_ids:
        preds = predicted_boxes.get(class_id, [])
        targets = target_boxes.get(class_id, [])
        
        if len(preds) == 0 and len(targets) == 0:
            matches[class_id] = []
            unmatched_preds[class_id] = []
            unmatched_targets[class_id] = []
            continue
            
        if len(preds) == 0:
            matches[class_id] = []
            unmatched_preds[class_id] = []
            unmatched_targets[class_id] = list(range(len(targets)))
            continue
            
        if len(targets) == 0:
            matches[class_id] = []
            unmatched_preds[class_id] = list(range(len(preds)))
            unmatched_targets[class_id] = []
            continue
        
        cost_matrix = np.zeros((len(preds), len(targets)))
        
        for i, pred_box in enumerate(preds):
            for j, target_box in enumerate(targets):
                iou_value = iou(pred_box, target_box)
                cost_matrix[i][j] = 1 - iou_value  # 成本 = 1 - IoU
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        matched_pred_indices = set()
        matched_target_indices = set()
        
        for pred_idx, target_idx in zip(row_indices, col_indices):
            iou_value = 1 - cost_matrix[pred_idx][target_idx]
            if iou_value >= iou_threshold:
                matched_pairs.append((pred_idx, target_idx, iou_value))
                matched_pred_indices.add(pred_idx)
                matched_target_indices.add(target_idx)
        
        unmatched_pred_list = [i for i in range(len(preds)) if i not in matched_pred_indices]
        unmatched_target_list = [i for i in range(len(targets)) if i not in matched_target_indices]
        
        matches[class_id] = matched_pairs
        unmatched_preds[class_id] = unmatched_pred_list
        unmatched_targets[class_id] = unmatched_target_list
    
    return matches, unmatched_preds, unmatched_targets

def multi_gaussian_point_reward(completions, solution, **kwargs):
    def g_point_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        
        sigma_x = alpha * gt_width
        sigma_y = alpha * gt_height

        x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
        y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
        exponent = -0.5 * (x_term + y_term)
        point_reward = math.exp(exponent)
        point_reward = round(point_reward,3)
        return point_reward

    contents = [completion[0]["content"] for completion in completions]

    rewards = []

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            #response = eval(content.split('assistant\n')[-1].strip())
            response = ast.literal_eval(content.split('assistant\n')[-1].strip())
            sol = {int(key): value for key, value in sol.items()}
            bbox_matches,_,__ = match_boxes(response,sol)
            
            reward=0.0
            for match in bbox_matches[0]:
                reward += g_point_reward(response[0][match[0]], sol[0][match[1]])
            if len(sol[0])+len(response[0])>0:
                reward=(reward*2)/(len(sol[0])+len(response[0]))
            else:
                reward=1.0
            rewards.append(reward)
        except (SyntaxError, NameError,json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            rewards.append(0.0)
            
        
    return rewards

def multi_gaussian_plane_reward(completions, solution, **kwargs):
    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps   = 1e-8
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        pred_width = pred_x2 - pred_x1
        pred_height = pred_y2 - pred_y1
        # pred_μ
        pred_mu = np.array([pred_center_x, pred_center_y])

        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        # gt_μ
        gt_mu = np.array([gt_center_x, gt_center_y])
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1

        # 1 sigma
        pred_sigma_x = pred_width * alpha
        pred_sigma_y = pred_height * alpha
        gt_sigma_x   = gt_width * alpha
        gt_sigma_y = gt_height * alpha

        pred_cov = np.array([[pred_sigma_x**2, 0], 
                            [0, pred_sigma_y**2]])
        
        # Σ2 (ground truth distribution covariance matrix)  
        gt_cov = np.array([[gt_sigma_x**2, 0], 
                        [0, gt_sigma_y**2]])
        
        sigma_avg = (pred_cov + gt_cov) / 2
        # 
        mu_diff = pred_mu - gt_mu

        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1/8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))
        
        # (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))
        det_sigma_avg = np.linalg.det(sigma_avg)
        det_pred_cov = np.linalg.det(pred_cov)
        det_gt_cov = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_pred_cov * det_gt_cov + eps)))
        except:
            return 0.0
        bhattacharyya_distance = term1 + term2

        plane_reward = np.exp(-bhattacharyya_distance)
        plane_reward = round(plane_reward,3)
        return plane_reward

    contents = [completion[0]["content"] for completion in completions]

    rewards = []

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            #response = eval(content.split('assistant\n')[-1].strip())
            response = ast.literal_eval(content.split('assistant\n')[-1].strip())
            sol = {int(key): value for key, value in sol.items()}
            bbox_matches,_,__ = match_boxes(response,sol)
            reward=0.0
            for match in bbox_matches[0]:
                reward += g_plane_reward(response[0][match[0]], sol[0][match[1]])
            if len(sol[0])+len(response[0])>0:
                reward=(reward*2)/(len(sol[0])+len(response[0]))
            else:
                reward=1.0
            rewards.append(reward)
        except (SyntaxError, NameError,json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            rewards.append(0.0)
    return rewards

def object_to_dict(obj):

    return {key: value for key, value in obj.__dict__.items()}

reward_funcs_registry = {
    "gaussian_point": gaussian_point_reward,
    "gaussian_plane": gaussian_plane_reward,
    "format": format_reward,
    "multi_gaussian_point":multi_gaussian_point_reward,
    "multi_box_format_reward":multi_box_format_reward,
    "multi_gaussian_plane_reward":multi_gaussian_plane_reward
}

def get_vlm_module(model_name_or_path):
    
    return Qwen2VLModule

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    script_args.reward_funcs="".join(script_args.reward_funcs)
    script_args.reward_funcs=script_args.reward_funcs[1:-1].replace(" ","").split(",")

    print(script_args.reward_funcs)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
   

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )
    
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
