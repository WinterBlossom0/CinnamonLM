"""Train a GPT-2-style byte-level BPE tokenizer on TinyStories itself, vocab
10K -- matching the TinyStories paper's own setup ("GPT-Neo tokenizer but
only keep the top 10K most common tokens", arXiv:2305.07759 footnote 2),
except trained fresh on this corpus rather than filtering GPT-Neo's existing
50257-token vocab down to its 10K most-frequent entries."""
from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 512
OUT_DIR = Path(__file__).parent / "tokenizer"


def main():
    ds = load_from_disk(str(Path(__file__).parent / "tinystories"))["train"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        (r["text"] for r in ds),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print(f"saved BPE tokenizer (vocab={tokenizer.get_vocab_size()}) to {OUT_DIR}")


if __name__ == "__main__":
    main()
