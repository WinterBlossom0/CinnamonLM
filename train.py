"""Train a linear-attention LM (KDA, OSDN, or Parallax) on TinyStories with
gelu_new MLPs, param-matched to activation_comparison's GPT-Neo replica --
same vocab (tokenizer/tokenizer.json, copied from there), same context
length, same data, so the only thing actually being compared is the attention
mechanism. Results land in results/<attention>/seed_<seed>/metrics.json,
same schema as activation_comparison.

Optimizer is AdamW for KDA/OSDN; Parallax specifically trains under Muon
(see _make_optimizers) -- its paper reports the covariance-correction gate
collapses under AdamW, so AdamW would not be testing the mechanism as
proposed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

from model import LM
from muon import Muon, split_params

ROOT = Path(__file__).parent
TOKENIZER_DIR = ROOT / "tokenizer"
DATA_DIR = ROOT / "tinystories"
RESULTS_DIR = ROOT / "results"

VRAM_CAP_GB = 15


def apply_vram_cap(device: torch.device, gb: float = VRAM_CAP_GB) -> None:
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_memory
    # A ceiling, so a card smaller than the cap just gets all of it: a T4 has
    # 14.56 GiB, and an unclamped 15/14.56 is rejected ("Allowed range: 0~1").
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * (1 << 30) / total), idx)


CONTEXT_LENGTH = 128
D_MODEL = 16
D_FF = 64
NUM_LAYERS = 4
CHUNK = 64
BATCH_SIZE = 1024
LR = 5e-4
WARMUP_STEPS = 500
EPOCHS = 3

# (n_heads, head_dim) per attention type, each independently searched for the
# closest total-param match to activation_comparison's ~23200.
HEAD_CONFIG = {
    "kda": (5, 3),        # 23168 params (still closest under gelu_new, re-verified)
    "osdn": (11, 2),      # 23252 params -- no gate/decay params, needs more heads*dim than kda
    "parallax": (1, 13),  # 23212 params -- extra W_R + position embedding leave little head budget
}


def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_DIR / "tokenizer.json"),
        bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>",
    )


class PackedBlocks(Dataset):
    def __init__(self, ids: torch.Tensor):
        n = (ids.shape[0] // CONTEXT_LENGTH) * CONTEXT_LENGTH
        self.blocks = ids[:n].view(-1, CONTEXT_LENGTH)

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def _pack(split_rows, tokenizer: PreTrainedTokenizerFast, cache_path: Path) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path)
    eos = tokenizer.eos_token_id
    chunks = []
    batch_size = 10000
    texts = split_rows["text"]
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]
        for row in enc:
            chunks.append(torch.tensor(row + [eos], dtype=torch.int32))
        print(f"  tokenized {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    ids = torch.cat(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_path)
    return ids


def _make_optimizers(attention, model, total_steps):
    """Parallax specifically needs Muon: its paper reports the covariance-
    correction gate collapses under AdamW (W_R stable rank 134 -> ~9), so
    training it under AdamW is not a fair test of the mechanism (see
    parallax/attention.py). KDA/OSDN have no such documented dependency and
    stay on plain AdamW, unchanged.

    Two separate optimizer instances (not param groups on one), since Muon and
    AdamW are different classes -- each needs its own scheduler, both stepped
    together every batch.
    """
    if attention != "parallax":
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = get_cosine_schedule_with_warmup(opt, WARMUP_STEPS, total_steps)
        return [opt], [sched]

    muon_params, adamw_params = split_params(model)
    muon_opt = Muon(muon_params, lr=0.02)          # Muon's own reference default; untuned here
    adamw_opt = torch.optim.AdamW(adamw_params, lr=LR)
    opts = [muon_opt, adamw_opt]
    scheds = [get_cosine_schedule_with_warmup(o, WARMUP_STEPS, total_steps) for o in opts]
    return opts, scheds


def run_seed(attention: str, seed: int, epochs: int, device: torch.device, vocab_size: int,
            train_ids: torch.Tensor, eval_ids: torch.Tensor, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    n_heads, head_dim = HEAD_CONFIG[attention]
    model = LM(attention, vocab_size, D_MODEL, D_FF, NUM_LAYERS, n_heads, head_dim,
              chunk=CHUNK, context_length=CONTEXT_LENGTH).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds, eval_ds = PackedBlocks(train_ids), PackedBlocks(eval_ids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    total_steps = len(train_loader) * epochs
    optimizers, schedulers = _make_optimizers(attention, model, total_steps)

    def run_eval() -> float:
        # token-weighted mean, not mean-of-batch-means: the last eval batch
        # is ragged (not drop_last), so equal-weighting it with full batches
        # biases the result toward its tokens by ~2x per token
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device).long()
                _, loss = model(batch, labels=batch)
                n_tokens = batch.shape[0] * (batch.shape[1] - 1)  # shifted labels
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        model.train()
        return total_loss / total_tokens

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device).long()
            _, loss = model(batch, labels=batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f"  seed {seed} step {step}/{total_steps} loss {loss.item():.4f} "
                     f"({time.time() - t0:.0f}s)", flush=True)
        eval_loss = run_eval()
        history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        print(f"  seed {seed} epoch {epoch + 1}/{epochs} eval_loss {eval_loss:.4f}", flush=True)

    result = {
        "attention": attention, "seed": seed, "n_params": n_params,
        "history": history, "final_eval_loss": history[-1]["eval_loss"],
        "wall_time_s": time.time() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", default="kda", choices=list(HEAD_CONFIG))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    apply_vram_cap(device)

    tokenizer = load_tokenizer()
    ds = load_from_disk(str(DATA_DIR))
    cache_dir = ROOT / "packed_cache"
    print("tokenizing/packing train split...", flush=True)
    train_ids = _pack(ds["train"], tokenizer, cache_dir / "train.pt")
    print("tokenizing/packing validation split...", flush=True)
    eval_ids = _pack(ds["validation"], tokenizer, cache_dir / "validation.pt")
    print(f"train tokens: {train_ids.shape[0]:,}  eval tokens: {eval_ids.shape[0]:,}", flush=True)

    for seed in args.seeds:
        out_dir = RESULTS_DIR / args.attention / f"seed_{seed}"
        print(f"=== attention={args.attention} seed={seed} ===", flush=True)
        run_seed(args.attention, seed, args.epochs, device, len(tokenizer), train_ids, eval_ids, out_dir)


if __name__ == "__main__":
    main()
