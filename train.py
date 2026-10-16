"""Train GPT-Neo on TinyStories, comparing activation functions across seeds.
Results land in results/<activation>/seed_<seed>/metrics.json."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

import xielu  # noqa: F401
import polyrelu  # noqa: F401
import swiglu
import powlu

GATED_ACTIVATIONS = {"swiglu": swiglu.apply_swiglu, "powlu": powlu.apply_powlu}

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
    fraction = min(1.0, gb * (1 << 30) / total)  # smaller GPUs: use it all instead of crashing
    torch.cuda.set_per_process_memory_fraction(fraction, idx)


CONTEXT_LENGTH = 128
HIDDEN_SIZE = 16
NUM_LAYERS = 4
NUM_HEADS = 4    # TinyStories-1M's head_dim=4 (64/16), not its raw head count (would give head_dim=1 here)
WINDOW_SIZE = 16  # TinyStories-1M's window/context ratio (256/2048), scaled to this context_length
BATCH_SIZE = 640
LR = 1e-2
WARMUP_STEPS = 500
EPOCHS = 3


def load_tokenizer() -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_DIR / "tokenizer.json"),
        bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    return tok


def build_config(vocab_size: int, activation: str, eos_token_id: int = 0) -> GPTNeoConfig:
    act_fn = "gelu_new" if activation in GATED_ACTIVATIONS else activation
    return GPTNeoConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        attention_types=[[["global", "local"], NUM_LAYERS // 2]],
        window_size=WINDOW_SIZE,
        max_position_embeddings=CONTEXT_LENGTH,
        activation_function=act_fn,
        attention_dropout=0, embed_dropout=0, resid_dropout=0,
        tie_word_embeddings=True,
        bos_token_id=eos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id,
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


def run_seed(activation: str, seed: int, epochs: int, device: torch.device,
            train_ids: torch.Tensor, eval_ids: torch.Tensor, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    tokenizer = load_tokenizer()
    config = build_config(len(tokenizer), activation, tokenizer.eos_token_id)
    model = GPTNeoForCausalLM(config)
    if activation in GATED_ACTIVATIONS:
        GATED_ACTIVATIONS[activation](model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds, eval_ds = PackedBlocks(train_ids), PackedBlocks(eval_ids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    def run_eval() -> float:
        # token-weighted mean, not mean-of-batch-means: the last eval batch
        # is ragged (not drop_last), so equal-weighting it with full batches
        # biases the result toward its tokens by ~2x per token
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device).long()
                out = model(input_ids=batch, labels=batch)
                n_tokens = batch.shape[0] * (batch.shape[1] - 1)  # shifted labels
                total_loss += out.loss.item() * n_tokens
                total_tokens += n_tokens
        model.train()
        return total_loss / total_tokens

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device).long()
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f"  seed {seed} step {step}/{total_steps} loss {out.loss.item():.4f} "
                     f"({time.time() - t0:.0f}s)", flush=True)
        eval_loss = run_eval()
        history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        print(f"  seed {seed} epoch {epoch + 1}/{epochs} eval_loss {eval_loss:.4f}", flush=True)

    result = {
        "activation": activation, "seed": seed, "n_params": n_params,
        "history": history, "final_eval_loss": history[-1]["eval_loss"],
        "wall_time_s": time.time() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", default="gelu_new")
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
        out_dir = RESULTS_DIR / args.activation / f"seed_{seed}"
        print(f"=== activation={args.activation} seed={seed} ===", flush=True)
        run_seed(args.activation, seed, args.epochs, device, train_ids, eval_ids, out_dir)


if __name__ == "__main__":
    main()
