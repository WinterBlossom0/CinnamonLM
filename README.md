# CinnamonLM

> **⚠️ Work in progress — research code.** The architecture works and the
> optimisation tests pass, but **no full training run has been done yet**. Numbers
> below are from local single-GPU measurement, not from a trained model. Expect
> things to change.

A decoder-only language model with **adaptive per-token depth** and a
**hypernetwork-generated MoE**.

Instead of stacking `N` distinct transformer blocks, CinnamonLM recurses on a
single block per position and decides **per token** how many times to go round.
Instead of holding `n` copies of an expert, it holds **one body plus a bank of
hypernetworks**, and routes by choosing which hypernetwork *generates* the
adapter weights for that token.

```
tokens → Embedding → B1 → B2 → B_mid → B3 → B4 → Norm → LM head
                     ↑     ↑     ↑      ↑     ↑
                  shared recur shared recur shared
                   once   ·32   once   ·32   once
```

## What is unusual about it

**Adaptive depth, per token.** Each token recurses until its hidden state stops
changing, up to a cap. Easy tokens leave early. Measured on a memorised batch:
depth falls from 13.4 to 6.2 as the model learns the sequence — fewer recurrences
once the answer is easy.

**One body, a bank of hypernetworks.** The body is quadratic in `d_model`, a
hypernetwork only linear, so `n` specialisations cost far less than `n` copies of
the block. Decisively, attention runs **once per recurrence instead of once per
expert** — the cost of specialisation stops scaling with how many you have.

**Weights generated per token, never materialised.** Each token gets its own
effective FFN weight matrix, generated from its own hidden state. Building those
matrices would cost ~536 MB per batch; the DoRA row-norm is instead computed in
closed form from the low-rank factors, so the widest live tensor is
`[tokens, d_ff, rank]` rather than `[tokens, d_ff, d_model]`. **Peak VRAM 14.95 GB
→ 3.55 GB**, verified identical to the naive version to 1e-10 in float64.

**No positional encoding.** Position enters only through KDA's ordered recurrent
state (Kimi Linear's NoPE arrangement). Verified live: permuting the input changes
the output, and a change to token 0 propagates to every later position.

## Status

| | |
|---|---|
| Architecture | implemented, 110.17 M params (44.63 M non-embedding) |
| Tests | 42 passing, 1 expected `xfail` |
| Overfit (512 tokens) | ✅ loss 11.81 → **0.0112**, through the 6.236 unigram floor |
| Real training run | ❌ **not done yet** |
| Multi-GPU | ❌ gathered dispatch is not DDP-safe — see `architecture.md` §7 |

## Quick start

```bash
pip install torch tokenizers numpy datasets

# sanity: the architecture can actually optimise
python overfit_check.py --lr 1e-3 --warmup 20 --steps 150

# is the positional pathway alive?
python perm_check.py

# tests
python -m pytest tests/ -q

# train
python train.py --domain babylm --tokenizer tokenizer.json
```

Multi-GPU is auto-detected from `torchrun`, no flag — but see the DDP caveat above
before relying on it.

```bash
torchrun --nproc_per_node=4 train.py --domain babylm --tokenizer tokenizer.json
```

## Layout

```
cinnamon/
  model.py      top-level forward
  routed.py     the recurrence: routing, halting, per-token dispatch
  blocks.py     shared scaffold and the expert body
  hypernet.py   weight generation + factorised DoRA application
  kda.py        Kimi Delta Attention (chunked gated delta rule)
  attention.py  MLA and the fixed FFN
  attnres.py    attention over depth, in place of plain residual accumulation
  router.py     hypernetwork router + load-balancing aux loss
  halting.py    the convergence measure
  data.py       corpora, packing
  ewc.py        elastic weight consolidation for sequential domains
train.py        training loop (CPU / CUDA / XLA, DDP auto-detected)
kg/             Kaggle push + launch helpers
tests/          42 tests, 1 expected xfail
architecture.md what is actually built and measured, and what is still open
```

## Honest notes

A long stretch of this project was spent on a model that **could not learn** — loss
pinned at the unigram marginal with every position predicting the same token. The
cause turned out to be a single **post-norm** inside the recurrence: applied up to
32 times, it produced gradient norms of ~1.5e5 against `clip=1.0`, so the real
learning signal was divided into nothing. Pre-norm cut that to ~70.

Two plausible hypotheses were tested and **refuted** first (missing positional
signal; structural oversmoothing from repeated attention). `architecture.md` §11
records both, and the diagnostic that was itself wrong — a collapse metric that
read 0.995 on a healthy model because it was measuring the shared unigram bias
rather than degeneracy.

Kept because the wrong turns are the useful part.

## Reading

- Kimi Linear / KDA — gated delta rule, NoPE, 3:1 linear:full attention
- DoRA — weight-decomposed low-rank adaptation
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* —
  why post-norm explodes with depth
- Geiping et al. 2025, recurrent-depth LMs — input injection at every step

## License

Not yet chosen.
