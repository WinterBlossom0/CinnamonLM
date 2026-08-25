# activation_comparision

Does a newer activation function actually help, or is that just what the paper
says? One architecture, one dataset, one activation swapped at a time —
results land side by side so they're comparable.

Self-contained: everything needed to reproduce this lives under this
directory (own `.gitignore`, no imports outside `activation_comparision/`).

## Setup

Architecture replicates `roneneldan/TinyStories-1M` (GPT-Neo, alternating
global/local attention), shrunk further to keep iteration fast:

| | TinyStories-1M | here |
|---|---|---|
| vocab | 50257 (top-10K kept) | 512 (BPE trained fresh on TinyStories text) |
| hidden_size | 64 | 16 |
| num_layers | 8 | 4 |
| num_heads | 16 | 16 |
| window_size | 256 | 256 |
| context_length | 512 | 128 |

BPE vocab is trained from the corpus itself, not filtered down from GPT-Neo's
vocab (`train_tokenizer.py`) — same idea as the paper's "top 10K", different
mechanism.

```bash
torch-python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories').save_to_disk('tinystories')"
torch-python train_tokenizer.py
torch-python train.py --activation gelu_new --seeds 0 1 2 --epochs 3
```

`train.py` packs+caches tokens on first run (`packed_cache/`, ~3.1GB, gitignored),
hard-caps VRAM at 15GB (`apply_vram_cap`), and trains one activation across
however many seeds are passed.

## Adding an activation

1. Write `<name>.py`: an `nn.Module` implementing the paper's exact formula
   (verify against the primary source, not a summary — see `xielu.py` and
   `polyrelu.py` for the pattern), registered as a class (not instance) into
   `transformers.activations.ACT2FN["<name>"]`. Registering the class matters:
   `ACT2FN` is a `ClassInstantier` that calls `cls(**kwargs)` fresh on every
   lookup, so each transformer layer gets its own independent trainable
   instance instead of sharing one.
2. `import <name>` (for its registration side effect) in `train.py`.
3. `torch-python train.py --activation <name> --seeds 0 1 2 --epochs 3`.

Gated (GLU-style) activations needing a second projection — `x1 * gate_fn(x2)`,
not a plain `ACT2FN` entry — go through `glu_mlp.py` instead (see `swiglu.py`/
`powlu.py`): write `<name>_gate(x2)`, an `apply_<name>(model)` that calls
`glu_mlp.apply_gated_mlp`, and add `"<name>": apply_<name>` to `train.py`'s
`GATED_ACTIVATIONS`. `apply_gated_mlp` defaults to a param-matched
`intermediate_size` (not the plain MLP's `4*hidden_size`) so the extra
projection matrix doesn't inflate params versus the non-gated activations.

Results land in `results/<name>/seed_<seed>/metrics.json` — same schema every
time, so any two activations are a direct diff.

## Results

Eval loss (nats), 3 seeds, mean ± sample-stdev:

| activation | seed 0 | seed 1 | seed 2 | mean ± std | params |
|---|---|---|---|---|---|
| gelu_new | 3.2324 | 3.2504 | 3.2509 | 3.2446 ± 0.0105 | 23200 |
| xielu | 3.2050 | 3.2141 | 3.2224 | 3.2138 ± 0.0087 | 23208 |
| polyrelu | 3.1946 | 3.2053 | 3.2167 | 3.2055 ± 0.0110 | 23216 |
| swiglu | 3.2303 | 3.2365 | 3.2296 | 3.2321 ± 0.0038 | 23152 (param-matched) |
| powlu | 3.2428 | 3.2512 | 3.2426 | 3.2456 ± 0.0049 | 23152 (param-matched) |

- **gelu_new** — GPT-Neo's default, `transformers.GPTNeoConfig` unstated
  default, matches TinyStories paper's own (also unstated) choice.
- **xielu** — Huang & Schlag, "Deriving Activation Functions Using
  Integration," [arXiv:2411.13010](https://arxiv.org/abs/2411.13010).
  Trainable piecewise, 2 params/layer.
- **polyrelu** — Zhuo et al., "Polynomial Composition Activations,"
  ICLR 2025, [arXiv:2411.03884](https://arxiv.org/abs/2411.03884). Trainable
  cubic-in-ReLU(x), 4 params/layer.
- **swiglu** — Shazeer, "GLU Variants Improve Transformer,"
  [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) (2020). Classic gated
  baseline; `x1 * SiLU(x2)`, two projections instead of one — see `glu_mlp.py`.
- **powlu** — Jiang et al., Ant Group Ling Team, "PowLU: An Activation
  Function for Stable Pre-Training of LLMs,"
  [arXiv:2605.25704](https://arxiv.org/abs/2605.25704) (25 May 2026). SwiGLU
  with a bounded-growth gate; matches SwiGLU exactly for x2≤0.

## Constraints

- VRAM hard-capped at 15GB (`VRAM_CAP_GB` in `train.py`).
- Batch size (1024) was found via a probe-then-slowdown search matching the
  real training step exactly (grad clip included) — that search script isn't
  part of this repo, scratch-only, since it's a one-off tool not a result.
