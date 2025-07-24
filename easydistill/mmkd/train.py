
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
import torch
import numpy as np
import jsonlines
import torch.nn.functional as F
import os
import argparse
import logging
from datasets import load_dataset, Dataset
from typing import Optional, Dict, Union, List
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import PreTrainedModel, PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from torch.utils.data import Dataset
from PIL import Image
import os

class MMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[int(idx)]

class DistillSFTTrainer(SFTTrainer):

    def __init__(
        self,
        logits_dir: str = None,  
        teacher_vocab_size = None,  
        kd_ratio: float = 0.5,    
        max_seq_length : int = 1024,
        distillation_type: str = "forward_kld",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logits_dir = logits_dir
        self.teacher_vocab_size = teacher_vocab_size
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type
        self.teacher_logits = []
        with jsonlines.open(self.logits_dir) as reader:
            for obj in reader:
                self.teacher_logits.append(obj)


    def _load_teacher_logits(self, batch_size: int, it: int, dp_rank: int, device: torch.device, no_model_batch: Dict):
        start_idx = dp_rank * batch_size + batch_size * it
        end_idx = dp_rank * batch_size + batch_size * (it + 1)
        loaded_data = self.teacher_logits[start_idx:end_idx]
        arr = np.zeros((batch_size, self.max_seq_length, self.teacher_vocab_size))
        for i in range(len(loaded_data)):
            for j in range(len(loaded_data[i])):
                keys = np.array(list(loaded_data[i][j].keys()), dtype=int)
                values = np.array(list(loaded_data[i][j].values()))
                arr[i, j, keys] = values
                
        logits_tensor = torch.tensor(arr, dtype=torch.bfloat16, device=device)
        return self._shift_tensor_right(logits_tensor, no_model_batch['label'], pad_value=0)
    

    def _compute_white_box_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor]):
        student_logits = student_logits[:, :self.max_seq_length, :]
        teacher_probs = teacher_logits[:, :student_logits.size(1), :student_logits.size(-1)]
        mask = (labels != -100).float() if labels is not None else torch.ones_like(student_logits[:, :, 0])
        
        if self.distillation_type == "forward_kld":
            # Forward KLD: student learns from teacher (original implementation)
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
        elif self.distillation_type == "reverse_kld":
            # Reverse KLD: teacher provides certainty to student
            loss = F.kl_div(
                torch.log(teacher_probs.clamp(min=1e-10)),  # avoid log(0)
                F.softmax(student_logits, dim=-1),
                reduction='none',
                log_target=False
            ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
        else:
            raise ValueError(f"Unsupported distillation type: {self.distillation_type}. Use 'forward_kld' or 'reverse_kld'")
            
        return (loss * mask).sum() / mask.sum()


    @staticmethod
    def _shift_tensor_right(inputs: torch.Tensor, labels: torch.Tensor, pad_value: float = 0.0):
        batch_size, seqlen, vocab_size = inputs.shape
        device = inputs.device
        labels_ne = labels != -100
        shift_distances = torch.argmax(labels_ne.int(), dim=1)
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch_size, seqlen)
        shifted_idx = idx - shift_distances.unsqueeze(1)
        mask = shifted_idx >= 0
        shifted_idx = shifted_idx.clamp(min=0)
        inputs_flat = inputs.view(batch_size, seqlen, vocab_size)
        shifted_idx = shifted_idx.unsqueeze(2).expand(-1, -1, vocab_size)
        gathered = torch.gather(inputs_flat, 1, shifted_idx)
        mask = mask.unsqueeze(2).expand(-1, -1, vocab_size)
        return torch.where(mask, gathered, torch.full_like(gathered, pad_value))


    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        lm_loss = outputs.loss
        if self.logits_dir:
            teacher_logits = self._load_teacher_logits(
                batch_size=inputs['input_ids'].size(0),
                it=self.state.global_step,
                dp_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                device=model.device,
                no_model_batch={'label': inputs.get('labels', None)}
            )
            distil_loss = self._compute_white_box_distillation_loss(
                student_logits=outputs.logits,
                teacher_logits=teacher_logits,
                labels=inputs.get('labels', None)
            )
            total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
        else:
            total_loss = lm_loss
        return (total_loss, outputs) if return_outputs else total_loss

def train(config):
    with open(config["dataset"]["labeled_path"], "r") as f:
        raw_data = json.load(f)
    dataset = MMDataset(raw_data)
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(config["models"]["student"])

    training_arguments = SFTConfig(**config["training"])
    training_arguments.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_arguments.remove_unused_columns = False
    training_arguments.dataset_kwargs = {"skip_prepare_dataset": True}

    def collate_fn(examples):
                texts = []
                images = []
                for example in examples:
                
                    chat = example
                    text = processor.apply_chat_template(chat, tokenize=False)
                    texts.append(text)
                    
                    image, _ = process_vision_info(example)
                    images.append(image)

                batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
                labels = batch["input_ids"].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100
                
                if isinstance(processor, Qwen2_5_VLProcessor):
                    image_tokens = [151652, 151653, 151655]
                else:
                    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
                    
                for image_token_id in image_tokens:
                    labels[labels == image_token_id] = -100
                batch["labels"] = labels
                return batch
    
    try:
        job_type =  config["job_type"]
        if "mmkd_black_box" in job_type:
            
            
            trainer = SFTTrainer(
                model=student_model,
                data_collator=collate_fn,
                processing_class=processor.tokenizer,
                args=training_arguments,
                train_dataset=dataset
            )
        elif "mmkd_white_box" in job_type:
            teacher_vocab_size=json.load(open(os.path.join(config["models"]["teacher"], 'config.json')))['vocab_size']
            trainer = DistillSFTTrainer(
                logits_dir=config["dataset"]["logits_path"],
                data_collator=collate_fn,
                teacher_vocab_size=teacher_vocab_size,
                kd_ratio=config["distillation"]["kd_ratio"], 
                max_seq_length=config["distillation"]["max_seq_length"],
                distillation_type=config["distillation"].get("distillation_type", "forward_kld"),
                model=student_model,
                processing_class=processor.tokenizer,
                args=training_arguments,
                train_dataset=dataset,
            )
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return

    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    processor.tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)


if __name__ == "__main__":
    main()