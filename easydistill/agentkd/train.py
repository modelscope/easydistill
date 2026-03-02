# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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


import json
import argparse
import logging
import os
from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def prepare_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Load and configure model and tokenizer with special tokens.
    
    Args:
        config: Configuration dictionary containing model settings
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config["models"]
    training_config = config["training"]
    
    logger.info(f"Loading tokenizer from: {model_config['student']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["student"],
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    
    new_tokens_added = False
    if special_tokens_to_add := training_config.get("add_special_tokens"):
        vocab = set(tokenizer.get_vocab().keys())
        new_tokens = [token for token in special_tokens_to_add if token not in vocab]
        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} new special tokens: {new_tokens}")
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            new_tokens_added = True
        else:
            logger.info("All special tokens from config already exist in tokenizer vocabulary")

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token; setting it to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    deepspeed_enabled = training_config.get("deepspeed") is not None
    # device_map='auto' is incompatible with distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    device_map = "auto" if (not deepspeed_enabled and not is_distributed) else None

    logger.info(f"Loading model from: {model_config['student']}")
    logger.info(f"DeepSpeed enabled: {deepspeed_enabled}, Distributed: {is_distributed}, Device map: {device_map}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config["student"],
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if training_config.get("bf16", False) else torch.float32,
        device_map=device_map
    )
    
    if training_config.get("resize_vocab", False) and new_tokens_added:
        original_size = model.get_input_embeddings().weight.size(0)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings: {original_size} -> {len(tokenizer)}")
    
    return model, tokenizer

def load_and_prepare_dataset(config: Dict[str, Any]) -> Dataset:
    """
    Load dataset in original format for SFTTrainer to process.
    
    Args:
        config: Configuration dictionary containing dataset settings
        
    Returns:
        Dataset with original conversation format
    """
    dataset_config = config["dataset"]
    training_config = config["training"]["dataset"]
    
    logger.info(f"Loading dataset from: {dataset_config['labeled_path']}")
    dataset = load_dataset("json", data_files=dataset_config["labeled_path"], split="train")
    logger.info(f"Original dataset size: {len(dataset)}")
    
    if max_samples := training_config.get("max_samples"):
        if max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Dataset truncated to {max_samples} samples")
    
    logger.info(f"Final dataset size: {len(dataset)}")
    return dataset

def create_formatting_func(tokenizer: PreTrainedTokenizerBase):
    """
    Create a formatting function for SFTTrainer.
    
    Args:
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Formatting function that converts conversations to list of text strings
    """
    def formatting_func(examples: Dict[str, Any]) -> list[str]:
        """Format conversations using tokenizer's chat template."""

        conversations_list = examples["conversations"]
        
        if not isinstance(conversations_list[0], list):
            conversations_list = [conversations_list]
        
        formatted_texts = []
        for conversations in conversations_list:
            try:
                if not conversations:
                    formatted_texts.append("")
                    continue
                    
                transformed = []
                for turn in conversations:
                    if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                        logger.warning(f"Skipping malformed turn: {turn}")
                        continue
                    role = "user" if turn["from"] == "human" else "assistant"
                    content = turn["value"]
                    transformed.append({"role": role, "content": content})
                
                if not transformed:
                    formatted_texts.append("")
                    continue
                    
                formatted_text = tokenizer.apply_chat_template(
                    transformed, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
                
            except Exception as e:
                logger.warning(f"Error formatting conversation: {e}")
                formatted_texts.append("")
        
        return formatted_texts
    
    return formatting_func

def create_training_config(config: Dict[str, Any]) -> SFTConfig:
    """
    Create SFTConfig from configuration dictionary.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        SFTConfig object with all training parameters
    """
    output_config = config["training"]["output"]
    training_config = config["training"]
    dataset_config = config["training"]["dataset"]
    
    report_to_value = output_config.get("report_to", "none")
    report_to = [] if report_to_value == "none" else [report_to_value]
    
    sft_config = SFTConfig(
        output_dir=output_config["output_dir"],
        overwrite_output_dir=output_config.get("overwrite_output_dir", True),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", True),
        dataloader_num_workers=dataset_config.get("dataloader_num_workers", 4),
        logging_steps=output_config.get("logging_steps", 10),
        save_steps=output_config.get("save_steps", 500),
        save_only_model=output_config.get("save_only_model", False),
        max_seq_length=dataset_config.get("cutoff_len", 2048),
        packing=False,  
        deepspeed=training_config.get("deepspeed"),
        ddp_timeout=training_config.get("ddp_timeout", 1800),
        report_to=report_to,
        dataloader_pin_memory=True,
    )
    if training_config.get("gradient_checkpointing", True):
        sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

    return sft_config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary for required keys and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required keys are missing or invalid values found
    """
    required_sections = ["models", "dataset", "training"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
def train(config: Dict[str, Any]) -> None:
    """
    Execute complete SFT training pipeline.
    
    Args:
        config: Complete configuration dictionary
    """
    try:
        logger.info("Starting SFT training job")
        
        validate_config(config)
        
        logger.info("Step 1/4: Preparing model and tokenizer")
        model, tokenizer = prepare_model_and_tokenizer(config)
        
        logger.info("Step 2/4: Loading and preparing dataset")
        train_dataset = load_and_prepare_dataset(config)
        logger.info(f"Successfully loaded {len(train_dataset)} training samples")
        
        logger.info("Step 3/4: Creating training configuration")
        training_args = create_training_config(config)
        
        logger.info("Step 4/4: Initializing SFT Trainer")
        formatting_func = create_formatting_func(tokenizer)
        
        trainer = SFTTrainer(
            model=model, 
            tokenizer=tokenizer, 
            args=training_args, 
            train_dataset=train_dataset,
            formatting_func=formatting_func,  
            packing=False,
        )
        
        logger.info("Training process started")
        resume_checkpoint = config["training"].get("resume_from_checkpoint")
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        output_dir = config["training"]["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving final model and tokenizer to: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the JSON configuration file'
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    logger.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    train(config)

if __name__ == "__main__":
    main()
