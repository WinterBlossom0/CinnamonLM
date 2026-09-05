# CinnamonLM

> **WIP research code.** Architecture works, optimisation tests pass, **no full
> training run completed.** Numbers are local single-GPU measurements.

Decoder-only LM: **adaptive per-token depth** + **hypernetwork-generated MoE**.

One block per position, recursed; per-token depth set by a halting rule. One
expert body plus a bank of hypernetworks; the router selects which hypernetwork
*generates* that token's adapter weights.

```
tokens → Embedding → L1 → [L2 + L3] → L4 → Norm → LM head (untied)
                     ↑      ↑    ↑     ↑
                    once  recurrent   once
                          up to ·32
```

Four transformer layers, four KDA attentions, four FFNs. Only L3's FFN has generated weights.

## Properties

**Adaptive depth.** Fixed-point convergence (arXiv:2606.18206 Algorithm 1),
checked every recurrence, capped at `c_max2`: `r = fixed_point_residual(z_cand,
h).mean()` over the batch, damped update `h = eta*z_cand + (1-eta)*h`, halt once
`r < fp_tau`. Batch-uniform — depth is one value per batch, adaptive across
batches rather than within one. Untrained it does not fire at all — open problem
P1, though P1 was measured against the previous relative-improvement rule and
does not automatically carry over to this one.

**One body + hypernetwork bank.** Body quadratic in `d_model`, hypernetwork
linear ⇒ `n` specialisations ≪ `n` block copies. `stage_attn` runs once per
recurrence, not once per expert ⇒ attention cost independent of bank size.

**Per-token weights, never materialised.** Materialising costs ~536 MB/batch.
Instead the DoRA row-norm is expanded in closed form from the factors:
`‖V_j‖² = ‖Wn_j‖² + 2s⟨Wn_j,(BA)_j⟩ + s²‖(BA)_j‖²`, via `G = AAᵀ` and `Q = A Wnᵀ`.
Widest live tensor `[tokens, d_ff, rank]` not `[tokens, d_ff, d_model]`.
**14.95 → 3.55 GB**, equal to the naive form at 1e-10 (fp64).

**Canonical Kimi Delta Attention** (arXiv:2510.26692), via Flash Linear
Attention's `chunk_kda`. Gated delta-rule recurrence — per-key-channel decay,
beta write gate, causal short convolution, gated output norm — carries sequence
order in its state, so Cinnamon applies no positional encoding, no RoPE, no MLA
rank, and needs no compressed-attention configuration: the recurrence is already
linear in T, not quadratic, so 1024 → 16384 is 16× the tokens and ~16× the
attention compute, not 256×.

**Muon + AdamW.** Muon on matmul weights, AdamW on embeddings, norms and the
`phi`/`err_slot` lookup tables — split by role, not shape.

## Status

| | |
|---|---|
| params | 23.61 M (6.83 M non-embedding), 4 hypernets |
| context | **16384** (pilot default), via `loss_chunk` |
| tests | 39 total, **3 failing** — all one open problem (P1, halting calibration) |
| overfit, 512 tok | loss 11.81 → **0.0112**, past the 6.236 unigram floor |
| pilot, 1 epoch WikiText-2 | **GO** — dev **6.0216** vs unigram bar 6.3016 (0.280 nats) |
| full training run | **not done** |
| multi-GPU | unsupported by design |

Open problems are in [`problems.md`](problems.md) — observed failures only, each
with a measurement behind it.

## Run

```bash
pip install torch tokenizers numpy datasets transformers accelerate safetensors flash-linear-attention

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

16 GB card (13 GB allocator cap), seq 1024, checkpointing on, measured on the
4-layer architecture during pilot runs:

| batch | tok/s | peak |
|---|---|---|
| 2 × accum 2 | 702 | 6.6 GB |
| 4 × accum 1 | **948** | **12.3 GB** |

Same effective batch either way, so that 1.35× is pure launch amortisation.
Checkpointing on is a *throughput* win, not only memory. What runs out first is the
logits tensor `[B, seq, 32768]` and its fp32 copy, not the model — which is what
`loss_chunk` exists for.

Older numbers on the pre-rewrite 5-block architecture (217 / 395 / 553 tok/s at
batch 1 / 2 / 3, OOM at 4) are not comparable and were removed rather than carried
forward.

**Context is a packing choice.** `seq_len` appears nowhere in `Config`; KDA's
recurrent state and causal convolution carry order for any length. Changing it
costs a repack. Long context only needs `loss_chunk` for the `[B,T,32768]` logits
tensor (the pilot sets it automatically) — KDA's attention is already linear in T.
Longer is more efficient per token, since fixed per-recurrence overhead amortises:
16384 measured **506 tok/s at 5.06 GB**, against 217 at 1024 on the same batch 1.

## Layout

```
cinnamon/
  model.py      top-level forward
  routed.py     recurrence: routing, halting, per-token dispatch
  blocks.py     L1/L4 layers + the recurrent L2+L3 block
  hypernet.py   weight generation + factorised DoRA
  attention.py  canonical Kimi Delta Attention, xIELU FFN
  muon.py       Muon optimiser + the Muon/AdamW parameter split
  router.py     router + load-balancing aux loss
  halting.py    convergence measure
  data.py       corpora, packing
train.py        training loop (CPU/CUDA/XLA, single-GPU)
pilot/          contained go/no-go experiment
kg/             Kaggle push + launch helpers
tests/          39 tests
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

- Kimi Linear (arXiv:2510.26692) — gated delta-rule attention (KDA)
- Fixed-Point Reasoners (arXiv:2606.18206) — halting + residual-scaled recurrence
- xIELU (Huang & Schlag, arXiv:2411.13010) — the FFN activation
- Muon (Jordan et al.) — momentum orthogonalised by Newton–Schulz
- DoRA — weight-decomposed low-rank adaptation
- Xiong et al. 2020 — post-norm gradient explosion with depth
- Geiping et al. 2025 — recurrent-depth LMs, input injection

## License

Unset.
