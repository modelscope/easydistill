"""Fine-tune Laya's system-1 typed-decision head on a distilled dataset.

Verbatim port of the Laya research notebook's DDP recipe: preprocess a
typed-decisions dataset, train with GRPO-style policy gradients plus soft
cross-entropy guidance, then export a temperature-calibrated model.

Mechanical deviations from the notebook:

1. Training data comes from ``system1_build_dataset`` JSONL output (rows of
   ``{id, workflow, state, questions, gold}``) instead of the
   ``LocalLLaMA/typed-decisions`` Hugging Face dataset.
2. Paths come from CLI flags, and the parent process launches
   ``torch.distributed.run`` instead of hard-coded Kaggle working paths.
3. Questions dropped by marker-count mismatch are counted and reported with
   a WARNING instead of being silently discarded.
4. Lines longer than 100 characters were re-wrapped, and log wording was
   generalized away from the original 2xT4 hardware.

Usage (build the dataset first, then fine-tune):

    easydistill --config configs/system1/system1_distill_pai_token.yaml
    python examples/system1_train_entry.py --input outputs/system1_train_sms_pai_token.jsonl

Add --output-dir and --nproc to override the export directory (default
outputs/system1_laya_finetuned) and the number of DDP GPUs (default 2).

The parent process downloads the Laya model from the Hugging Face Hub (on
first run), preprocesses the dataset, then re-executes this file under
torchrun; child processes perform DDP training, and rank 0 exports the
fine-tuned model with fitted calibration temperatures.

Requires the laya research package (model, tokenizer, and training utilities).
"""
# allow: SIZE_OK - over 250 pure LOC by design (verbatim notebook port)

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Pin transformers to its PyTorch backend before its first import (as the
# laya research scripts do).
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import (  # noqa: E402
    QTYPES,
    build_model,
    build_sequence,
    proper_reward,
    render_options,
)
from safetensors.torch import load_file, save_file  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "convaiinnovations/laya"


def preprocess(args: argparse.Namespace) -> str:
    """Tokenize the dataset into training items; return the Laya model dir."""
    print(f"Fetching tokenizer and config from {MODEL_ID}...")
    model_dir = str(args.model_dir) if args.model_dir else str(snapshot_download(MODEL_ID))
    _fix_tokenizer_config(model_dir)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    def build_training_item(state, q, gold_q):
        t = q["type"]
        crit = q.get("criteria", {})
        if t == "choice":
            keys = list(crit.keys())
            target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
        elif t == "noul":
            target = [
                gold_q["probabilities"].get("false", 0.5),
                gold_q["probabilities"].get("true", 0.5),
            ]
        elif t == "score":
            n_levels = len(crit) if isinstance(crit, list) else 4
            target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]

        s = sum(target)
        target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
        label = target.index(max(target))
        k = len(render_options({"t": t, "crit": crit}))

        seq, markers = build_sequence(
            tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
            cfg["max_len"], cfg["head_max_len"],
        )
        if len(markers) != k:
            return None
        return {
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[t],
            "target": target,
            "label": label,
        }

    items = []
    n_dropped = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            state = row["state"]
            questions = row["questions"]
            gold = row["gold"]
            for qid, q in questions.items():
                if qid in gold:
                    it = build_training_item(state, q, gold[qid])
                    if it:
                        items.append(it)
                    else:
                        n_dropped += 1

    if n_dropped:
        print(f"WARNING: dropped {n_dropped} items whose marker count did not match.")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(items, args.items)
    print(f"Preprocessed {len(items)} training sequences from {args.input}.")
    print(f"Saved preprocessed items to {args.items}")
    return model_dir


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        k = len(it["markers"])
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items])
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def train(args: argparse.Namespace) -> None:
    """Run DDP fine-tuning as a torchrun child process."""
    if args.model_dir is None:
        raise SystemExit("--model-dir is required when running as a DDP child process")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir = args.model_dir
    output_dir = args.output_dir

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(args.items, weights_only=False)
    my_items = all_items[rank::world_size]

    EPOCHS = 4
    MICRO_BATCH = 8      # 8 sequences per forward pass per GPU
    GRAD_ACCUM = 4       # Effective batch across 2 GPUs = 64 sequences (8 * 2 * 4)
    GROUP_SIZE = 4       # GRPO baseline samples
    LR_ENCODER = 2.5e-5  # Encoder adaptation rate
    LR_HEAD = 1.0e-4     # Head adaptation rate
    SIGMA_START = 0.4    # Exploration noise
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(
            f"Starting DDP training: {len(all_items)} total items | "
            f"{len(my_items)} per rank | {EPOCHS} epochs"
        )
    t0 = time.time()

    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device)
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(
                    q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                    w_sph=0.75, w_rps=1.0,
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + full 1.0 soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(
                (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            )
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | "
                    f"Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | "
                    f"LR: {cur_lr:.2e}"
                )

        if rank == 0:
            print(
                f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | "
                f"Avg Loss: {epoch_loss/max(1, n_batches):.4f} ==="
            )

    dist.barrier()

    # Post-training temperature calibration on rank 0 (micro-batched in chunks of 16 to prevent OOM)
    if rank == 0:
        print()
        print("Fitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_items = all_items[::15][:400]
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device)
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print(
                "Fitted calibration temperatures (choice, score, noul):",
                [round(t, 3) for t in fitted_temps]
            )
        except Exception as e:
            print("Temperature fitting fallback:", e)
        os.makedirs(output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-typed-decisions"
        cfg["temperature"] = fitted_temps
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model successfully saved to {output_dir}!")

    dist.destroy_process_group()


def launch(args: argparse.Namespace, model_dir: str) -> None:
    """Spawn the DDP child processes through torch.distributed.run."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    n_gpu = torch.cuda.device_count()
    if n_gpu < args.nproc:
        raise SystemExit(
            f"DDP training needs {args.nproc} GPUs, but torch.cuda.device_count() "
            f"returned {n_gpu}."
        )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc}",
        os.path.abspath(__file__),
        "--input",
        args.input,
        "--output-dir",
        args.output_dir,
        "--model-dir",
        model_dir,
        "--items",
        args.items,
    ]
    print("Starting DDP training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Parse flags, then act as the parent launcher or a DDP child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="system1_build_dataset JSONL")
    parser.add_argument("--output-dir", default="outputs/system1_laya_finetuned")
    parser.add_argument("--model-dir", default=None, help="reuse a local Laya snapshot")
    parser.add_argument(
        "--items",
        default=None,
        help="preprocessed .pt path (default: OUTPUT_DIR/train_items.pt)",
    )
    parser.add_argument("--nproc", type=int, default=2, help="number of GPUs for DDP")
    args = parser.parse_args()
    if args.items is None:
        args.items = os.path.join(args.output_dir, "train_items.pt")

    if os.environ.get("LOCAL_RANK") is not None:
        train(args)
        return

    model_dir = preprocess(args)
    launch(args, model_dir)


if __name__ == "__main__":
    main()
