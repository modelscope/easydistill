#!/usr/bin/env python
import json
import argparse
import os
import logging
from jinja2 import Environment, FileSystemLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import torch
import jsonlines
import numpy as np
import torch.nn.functional as F


def formatting_func(examples):
    """
    Formats a single example for student training using the loaded template.
    """
    try:
        message = {"content": examples["instruction"], "output": examples["output"]}
        full_text = template.render(
            message=message,
            add_generation_prompt=False,
            add_output=True
        )
        return full_text
    except Exception as e:
        logging.warning(f"Error processing sample: {str(e)}")
        return ""


class MultiDistillSFTTrainer(SFTTrainer):
    """
    Extension of SFTTrainer to support multiple teacher models in white-box distillation.
    """
    def __init__(
        self,
        logits_dirs: list,
        teacher_vocab_sizes: list,
        kd_ratio: float,
        max_seq_length: int,
        distillation_type: str = "forward_kld",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logits_dirs = logits_dirs
        self.teacher_vocab_sizes = teacher_vocab_sizes
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type
        # Load and cache each teacher's logits
        self.teacher_logits_list = []
        for path in self.logits_dirs:
            entries = []
            with jsonlines.open(path) as reader:
                for item in reader:
                    entries.append(item)
            self.teacher_logits_list.append(entries)

    def _load_teacher_logits_for(self, t_idx: int, batch_size: int, step: int, rank: int, device: torch.device, labels: torch.Tensor):
        """
        Slice and shift the teacher logits for teacher index t_idx.
        """
        data = self.teacher_logits_list[t_idx]
        vocab_size = self.teacher_vocab_sizes[t_idx]
        start = rank * batch_size + batch_size * step
        end = start + batch_size
        slice_ = data[start:end]
        arr = np.zeros((batch_size, self.max_seq_length, vocab_size), dtype=np.float32)
        for i, sample in enumerate(slice_):
            for pos, dist in enumerate(sample):
                idxs = np.fromiter(dist.keys(), dtype=int)
                vals = np.fromiter(dist.values(), dtype=float)
                arr[i, pos, idxs] = vals
        tensor = torch.tensor(arr, dtype=torch.bfloat16, device=device)
        return self._shift_tensor_right(tensor, labels, pad_value=0.0)

    def _compute_white_box_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        student_logits = student_logits[:, :self.max_seq_length, :]
        teacher_probs = teacher_logits[:, :student_logits.size(1), :student_logits.size(-1)]
        mask = (labels != -100).float() if labels is not None else torch.ones_like(student_logits[..., 0])
        if self.distillation_type == "forward_kld":
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1) / mask.sum()
        elif self.distillation_type == "reverse_kld":
            loss = F.kl_div(
                torch.log(teacher_probs.clamp(min=1e-10)),
                F.softmax(student_logits, dim=-1),
                reduction='none',
                log_target=False
            ).sum(dim=-1) / mask.sum()
        else:
            raise ValueError(f"Unsupported distillation type: {self.distillation_type}")
        return (loss * mask).sum() / mask.sum()

    @staticmethod
    def _shift_tensor_right(inputs: torch.Tensor, labels: torch.Tensor, pad_value: float = 0.0):
        batch, seqlen, vocab = inputs.shape
        device = inputs.device
        ne = labels != -100
        shift = torch.argmax(ne.int(), dim=1)
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch, seqlen)
        shifted = idx - shift.unsqueeze(1)
        mask = shifted >= 0
        shifted = shifted.clamp(min=0)
        flat = inputs.view(batch, seqlen, vocab)
        shifted = shifted.unsqueeze(2).expand(-1, -1, vocab)
        gathered = torch.gather(flat, 1, shifted)
        mask = mask.unsqueeze(2).expand_as(gathered)
        return torch.where(mask, gathered, torch.full_like(gathered, pad_value))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        lm = outputs.loss
        if not self.logits_dirs:
            return (lm, outputs) if return_outputs else lm
        batch = inputs['input_ids'].size(0)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        step = self.state.global_step
        labels = inputs.get('labels', None)
        dist_losses = []
        for i in range(len(self.logits_dirs)):
            t_logits = self._load_teacher_logits_for(
                i, batch, step, rank, model.device, labels
            )
            dist_losses.append(
                self._compute_white_box_distillation_loss(
                    outputs.logits, t_logits, labels
                )
            )
        total_dist = sum(dist_losses)
        loss = (1 - self.kd_ratio) * lm + self.kd_ratio * total_dist
        return (loss, outputs) if return_outputs else loss


def train_multi(config):
    # Load data
    ds = load_dataset("json", data_files=config["dataset"]["labeled_path"])["train"]
    # Student setup
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"], trust_remote_code=True)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], trust_remote_code=True)
    # Template
    global template
    tpl = config["dataset"]["template"]
    tpl_dir, tpl_file = os.path.split(tpl)
    env = Environment(loader=FileSystemLoader(tpl_dir))
    template = env.get_template(tpl_file)
    # Training args
    args = SFTConfig(**config["training"])
    # Multi-teacher config
    t_paths = config["models"]["teacher"]
    if not isinstance(t_paths, list): t_paths = [t_paths]
    l_paths = config["dataset"]["logits_path"]
    if not isinstance(l_paths, list): l_paths = [l_paths]
    assert len(t_paths) == len(l_paths), "Mismatch teachers vs logits paths"
    sizes = []
    for tp in t_paths:
        c = json.load(open(os.path.join(tp, 'config.json')))
        sizes.append(c['vocab_size'])
    # Trainer
    trainer = MultiDistillSFTTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        args=args,
        train_dataset=ds,
        formatting_func=formatting_func,
        logits_dirs=l_paths,
        teacher_vocab_sizes=sizes,
        kd_ratio=config["distillation"]["kd_ratio"],
        max_seq_length=config["distillation"]["max_seq_length"],
        distillation_type=config["distillation"].get("distillation_type", "forward_kld"),
    )
    # Train and save
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    student_tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    opt = p.parse_args()
    conf = json.load(open(opt.config, 'r'))
    train_multi(conf)

if __name__ == "__main__":
    main()
