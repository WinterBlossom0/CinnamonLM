"""Download TinyStories at a pinned revision and pack each split into one int32 token stream (each story followed by
<|endoftext|>), saved as <out>/train.pt and <out>/validation.pt. It fails unless each stream has the SHA-256 of the
stream every reported run used: `python prepare_data.py [--out DIR]`."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
# SHA-256 of the int32 little-endian bytes of each packed stream (822,344,370 and 8,280,406 tokens).
DATA_SHA256 = {
    "train": "7b538b8dd9b0215426c7db7ea7bbffbe585e7895c0f5f45041bd3fb5022a6f2e",
    "validation": "a971d8a0d19ac4b173099766d8a3a9ecb4979eaac5c086aa3ab2d351e215bac6",
}
ENCODE_BATCH = 10000


def token_sha256(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def pack(texts: list[str], tokenizer, eos: int) -> torch.Tensor:
    """The token ids of every text, each followed by eos, in order, as one int32 tensor."""
    parts = []
    for start in range(0, len(texts), ENCODE_BATCH):
        encodings = tokenizer.encode_batch(texts[start:start + ENCODE_BATCH], add_special_tokens=False)
        lengths = np.fromiter((len(encoding.ids) + 1 for encoding in encodings), dtype=np.int64, count=len(encodings))
        ends = np.cumsum(lengths)
        flat = np.empty(int(ends[-1]), dtype=np.int32)
        for encoding, end, length in zip(encodings, ends, lengths):
            flat[end - length:end - 1] = encoding.ids
        flat[ends - 1] = eos
        parts.append(flat)
        print(f"  tokenized {min(start + ENCODE_BATCH, len(texts))}/{len(texts)}", flush=True)
    return torch.from_numpy(np.concatenate(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "packed_cache")
    args = parser.parse_args()
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|endoftext|>")
    dataset = load_dataset("roneneldan/TinyStories", revision=TINYSTORIES_REVISION)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, expected in DATA_SHA256.items():
        print(f"packing {split}...", flush=True)
        ids = pack(dataset[split]["text"], tokenizer, eos)
        digest = token_sha256(ids)
        if digest != expected:
            raise RuntimeError(f"{split}: token SHA-256 {digest}, expected {expected}")
        torch.save(ids, args.out / f"{split}.pt")
        print(f"{split}: {ids.numel():,} tokens, SHA-256 matches", flush=True)


if __name__ == "__main__":
    main()
