# CinnamonLM

> **WIP research code.** Architecture works, optimisation tests pass, **no full
> training run completed.** Numbers are local single-GPU measurements.

Decoder-only LM: **adaptive per-token depth** + **hypernetwork-generated MoE**.

One block per position, recursed; per-token depth set by a halting rule. One
expert body plus a bank of hypernetworks; the router selects which hypernetwork
*generates* that token's adapter weights.

```
tokens → Embedding → B1 → B2 → B_mid → B3 → B4 → Norm → LM head
                     ↑     ↑     ↑      ↑     ↑
                  shared recur shared recur shared
                   once   ·32   once   ·32   once
```

## Properties

**Adaptive depth.** Token recurses until `rel_error < halt_tol`, capped at
`c_max2`. Measured on a memorised batch: depth 13.4 → 6.2 as the model learns.

**One body + hypernetwork bank.** Body quadratic in `d_model`, hypernetwork
linear ⇒ `n` specialisations ≪ `n` block copies. `stage_attn` runs once per
recurrence, not once per expert ⇒ attention cost independent of bank size.

**Per-token weights, never materialised.** Materialising costs ~536 MB/batch.
Instead the DoRA row-norm is expanded in closed form from the factors:
`‖V_j‖² = ‖Wn_j‖² + 2s⟨Wn_j,(BA)_j⟩ + s²‖(BA)_j‖²`, via `G = AAᵀ` and `Q = A Wnᵀ`.
Widest live tensor `[tokens, d_ff, rank]` not `[tokens, d_ff, d_model]`.
**14.95 → 3.55 GB**, equal to the naive form at 1e-10 (fp64).

**NoPE.** Position enters only via KDA's ordered recurrent state. Verified: not
permutation-equivariant; token-0 edit propagates to all later positions.

## Status

| | |
|---|---|
| params | 110.17 M (44.63 M non-embedding) |
| context | 1024 (not architectural — see below) |
| tests | 41 passing |
| overfit, 512 tok | loss 11.81 → **0.0112**, past the 6.236 unigram floor |
| full training run | **not done** |
| multi-GPU | unsupported by design |

## Run

```bash
pip install torch tokenizers numpy datasets

python overfit_check.py --lr 1e-3 --warmup 20 --steps 150   # architecture sanity
python perm_check.py                                        # positional pathway
python -m pytest tests/ -q
python pilot/run_pilot.py                                   # go/no-go, 1 epoch
python train.py                                             # WikiText-2
python train.py --domain babylm --max-hours 8
```

`--set KEY=VALUE` overrides any `Config` field. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Single GPU by design

No `torch.distributed`, `torchrun`, rank guards, or sharding.

Each hypernetwork runs only on tokens routed to it, skipping empty ones — a
data-dependent branch. Two ranks would skip different hypernetworks and desync
the all-reduce. DDP-safe alternative (every hypernetwork on every token):
**~6 min/step**.

16 GB card, seq 1024, checkpointing on:

| batch | tok/s | peak |
|---|---|---|
| 1 | 217 | 4.82 GB |
| **2** (default) | **395** | **7.91 GB** |
| 3 | 553 | 11.11 GB |
| 4 | OOM | — |

Default 2 × accum 2, not 3: depth is data-dependent, a batch halting later exceeds
the table. Checkpointing on is a *throughput* win, not only memory — off: 214 tok/s
@ batch 1; on: 433 @ batch 4. Batch 4 exhausts the logits tensor `[B,seq,128000]`,
not the model.

**Context 1024 is a packing choice.** `seq_len` appears nowhere in `Config`; NoPE
⇒ no positional table, KDA state size-invariant. Raising it costs a repack.
Measured @ batch 1: 512 = 131 tok/s, 1024 = 217, 2048 = 342, 4096 = OOM — longer
is more efficient per token (fixed per-recurrence overhead amortises).

## Layout

```
cinnamon/
  model.py      top-level forward
  routed.py     recurrence: routing, halting, per-token dispatch
  blocks.py     shared scaffold + expert body
  hypernet.py   weight generation + factorised DoRA
  kda.py        Kimi Delta Attention (chunked gated delta rule)
  attention.py  MLA, fixed FFN
  attnres.py    attention over depth, replaces residual accumulation
  router.py     router + load-balancing aux loss
  halting.py    convergence measure
  data.py       corpora, packing
  ewc.py        elastic weight consolidation
train.py        training loop (CPU/CUDA/XLA, single-GPU)
pilot/          contained go/no-go experiment
kg/             Kaggle push + launch helpers
tests/          41 tests
architecture.md what is built and measured; open problems
```

## Prior failure (kept as context)

Model could not learn for an extended period: loss pinned at the unigram
marginal, all positions emitting the same token. Cause: a single **post-norm
inside the recurrence**, applied up to 32×, gradient norms ~1.5e5 against
`clip=1.0`. Pre-norm → ~70.

Two hypotheses tested and refuted first (missing positional signal; structural
oversmoothing). `architecture.md` §11 records both, plus a diagnostic that was
itself wrong — a collapse metric reading 0.995 on a healthy model because it
measured shared unigram bias, not degeneracy.

## References

- Kimi Linear / KDA — gated delta rule, NoPE, linear:full attention ratio
- DoRA — weight-decomposed low-rank adaptation
- Xiong et al. 2020 — post-norm gradient explosion with depth
- Geiping et al. 2025 — recurrent-depth LMs, input injection

## License

Unset.
