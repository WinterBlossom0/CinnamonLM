# CinnamonLM — implemented architecture

What is built and measured, vs the spec (`recurrent-moe-architecture.md`) which is
design intent. `§n` = spec section. Deviations are marked.

One model; no staged pretraining. Modification guide: `CONTRIBUTING.md`.

## 1. Shape

```
tokens → Embedding → B1 → B2 → B_mid → B3 → B4 → Norm → LM head
                     ↑     ↑     ↑      ↑     ↑
                  shared recur shared recur shared
                   once   ·32   once   ·32   once
```

- **B1, B_mid, B4** — shared scaffold, separate parameter sets, run once.
- **B2, B3** — routed positions: one expert body + `n_hypernets` hypernetworks +
  a router. B2 and B3 bodies are distinct weights (§4.1) ⇒ recurrence counters
  reset between them.

Measured (`Config()`, `param_report()`):

| | |
|---|---|
| shared scaffold (B1+B_mid+B4) | 13.65 M |
| expert body (B2+B3) | 6.82 M |
| 8 hypernets (B2+B3) | 24.15 M |
| routers | 0.00 M |
| **non-embedding** | **44.63 M** |
| embedding + LM head (untied) | 65.54 M |
| total | 110.17 M |

Nothing vocab-sized is charged against the ~50 M budget: embedding and untied head
are `vocab × d_model`, set by the tokenizer, not a layer width. Ratios preserved:
`d_ff = 4d`, `q_lora = d/2`, `kv_lora = d/4`, `alpha = 2·rank`.

**Hypernet/body ratio.** Hypernetwork output layer is
`r_ceiling × (2·rank·(d + d_ff) + d + d_ff)` — **linear in d** — while the body it
modulates is **quadratic**. Shrinking the model inverts the ratio: 0.37× the body
at d=512, 0.83× at d=224/rank-16, >1 at d=160.
`test_default_config_hits_the_50m_budget` asserts <1; the failure is silent (model
still trains, budget just goes to generating adapters instead of layers).

Bank still outweighs the body collectively: 24.15 M vs 6.82 M. Open (§12).

## 2. Depth

Depth = recurrences one token spends at one block position. Per token (§7).

```
r <= r_free(2)   plain base weights, no hypernetwork
then             one commitment = turns(2) recurrences under one hypernetwork
cap              c_max2 = 32 per block position
```

Halting tested only at commitment boundaries ⇒ depth is always `r_free + k·turns`.

| counter | scope | resets |
|---|---|---|
| `depth` | per token, per block position | each block position |
| `r` | loop counter, same for all tokens | each block position |

`r` indexes the step embedding and per-step norms; both clamp at `r_ceiling`,
which must therefore be ≥ `c_max2`. A halted token stops being updated while the
loop continues, so all live tokens share the same `r`.

Measured, trained, 512 tok, `c_max2=32`: **B2 6.2 [4–8], B3 4.0 [4–4]**, from
13.4 [10–18] / 6.0 [4–6] at step 0. Depth *falls* as the model learns.

## 3. Shared blocks

```
KDA → FFN → KDA → FFN → KDA → FFN → MLA → FFN
```

Kimi Linear 3:1 ratio. Pre-RMSNorm every sublayer. **AttnRes** (§3.2) replaces
residual accumulation with softmax attention over depth — Full variant, single
head, RMSNorm on keys, pseudo-queries zero-init ⇒ starts as equal-weight average.

**NoPE throughout.** Position enters only via KDA's ordered recurrent state; every
MLA has a KDA upstream in the same block. Verified (`perm_check.py`): not
permutation-equivariant (max |diff| 2.36); a token-0 edit moves every later
position by ~3.6e-01 at all distances.

## 4. KDA

Gated delta rule, channelwise decay:

```
S_t = (I − b_t k_tᵀ k_t) Diag(a_t) S_{t−1} + b_t k_tᵀ v_t
```

Chunkwise: gate factored out inside a chunk leaves an ungated delta rule with
asymmetric keys, solved in closed form by the UT transform. `seq/chunk` sequential
steps instead of `seq`.

**(a) The inverse.** `(I+M)⁻¹` by repeated squaring overflows fp32 at chunk 64
(`inf`; 33% error at 32), which had forced `kda_chunk=16` = 4× more sequential
steps on the hottest path. Replaced with **blocked forward substitution**: invert
16×16 diagonal blocks (squaring is stable there), recover off-diagonals by
substitution. Chunk 64 safe and *more* accurate than what it replaced (fp64 @ C=64:
2.84e-13 vs 7.18e-13). **2.22× end-to-end, identical math.**

**(b) The decay clamp.** Chunked form splits decay into `k·exp(g)` and `k·exp(−g)`;
the second overflowed, so it was clamped — but once clamped the cumsum no longer
equals true cumulative decay and the chunked path silently stops matching the
recurrence (**1.4% of values clamped → 6.25e-02 error**, no NaN, no loss spike).

Fixed by removing the need: state-read/state-write factors are `exp(gcum)` and
`exp(g_end − gcum)`, both non-positive exponents, bounded by 1, **no clamp**. Only
the pairwise intra-chunk term must split; centring it on the chunk midpoint is
exact (constant cancels) and halves the range.

| decay | chunk | before | after |
|---|---|---|---|
| init | 64 | 3.33e-07 | 3.34e-07 |
| init | 128 | **6.25e-02** | **3.25e-07** |

**(c) The gate bound.** The split is accurate while per-chunk `|gcum|` ≲ 10. That
was enforced by a **per-step** clamp of 10 — `chunk`× too loose, since `gcum` is a
sum over the chunk. A real run reached **`gcum` 23.3 by step 250**, where the
chunked path had stopped matching the exact recurrence (1e-1 relative error at 9.5,
4e-1 at 18) with no NaN and no loss spike.

Kimi's kernel has no such limit: it **subtracts before exponentiating**
(`exp2(b_gn - b_gk)`, one exp, difference inside), so the exponent is never
positive. Implemented here: exact at any drift (**1.5e-15 @ `|gcum|` 640, 64× the
old range**) but needs a `[C,C,D]` intermediate + reduce instead of a tensor-core
matmul — **2.4× time, 1.5× memory** (395→165 tok/s, 7.9→12.1 GB), OOM at chunk 64.
Reverted.

Shipped: keep the split, bound `gcum` properly — log-decay floor of
`_G_RANGE/chunk` rather than `_G_RANGE`. This is fla's `safe_gate` tightened to
what the PyTorch split needs; `|gcum|` is bounded by construction. Cost:
retention/token ≥ `exp(-10/64)` = 0.855, though still ~total forgetting across a
full chunk (4.5e-5).

**Not what Kimi did — what Kimi did that is reachable without a Triton kernel.**
A kernel port removes the bound; correct fix if the constraint costs accuracy.

`gcum` is still logged but is now a *modelling* signal (retention/token =
`exp(-gcum/chunk)`), not a numerical watchdog.

## 5. Expert block + hypernetwork bank

One body per block position, not `n` copies. The router selects **which
hypernetwork generates the DoRA** for the dynamic FFN.

```
r <= r_free    KDA → FixedFFN → MLA → DynFFN(plain base W)
r >  r_free    KDA → FixedFFN → MLA → DynFFN(DoRA from hypernet i)
```

Body is 2:1 KDA:MLA. `stage_attn` (KDA/FixedFFN/MLA) is split from `stage_dyn`
(dynamic FFN, pointwise) so attention runs **once per recurrence, not once per
hypernetwork**. §4.3 requires each expert to produce its own K/V over the whole
history — batching cannot remove that; one body can.
`test_attention_runs_once_per_recurrence_not_once_per_hypernet` asserts
`stage_attn` calls == `depth.max()`.

### 5.1 Per-token generation — §5.5 deviation

Spec makes the hypernetwork content-blind (inputs `φ(r)` + base column norms) ⇒
one row per `r`, identical weights for all tokens at that `r`.

**Overridden.** `Hypernet.forward_token(r, content, base_norms)` returns a distinct
code per token, every recurrence, with no discretisation between the content signal
and the applied weights, so gradient reaches the content path.
`test_generation_is_content_sensitive_per_token` asserts both.

Content signal = `h` itself, RMS-normalised: the state entering the recurrence,
already per-token and context-mixed by B1 and prior recurrences. An earlier version
spent a dedicated KDA (0.81 M params + a KDA forward per recurrence) producing this;
raw `h` measured no worse.

### 5.2 Generated weight never materialised

DoRA: `V = Wn + s·B·A`, `W^(r) = m ⊙ V / ‖V‖`. Per token that is an `[out, in]`
matrix each — ~1 MB/token/projection at `d_ff=1024, d_model=256`, **~536 MB per
512-token batch**. That is what put peak VRAM at 14.95 GB.

`dora_linear` never builds it. Value factorises: `V x = Wn x + s·B(A x)`, widest
intermediate `[.., rank]`. The row norm appears to need the assembled matrix but
does not:

```
‖V_j‖²        = ‖Wn_j‖² + 2s·⟨Wn_j,(BA)_j⟩ + s²·‖(BA)_j‖²
‖(BA)_j‖²     = B_jᵀ (A Aᵀ) B_j            via G = A Aᵀ   [rank, rank]
⟨Wn_j,(BA)_j⟩ = Σ_k B[j,k] (A Wnᵀ)[k,j]    via Q = A Wnᵀ  [rank, out]
```

Widest tensor `[.., out, in]` → `[.., out, rank]`; rank replaces `d_model`.
**Peak VRAM 14.95 → 3.55 GB.** Pure optimisation, verified as one:
`test_dora_factorised_matches_reference` checks against the matrix-building
reference to **1e-10 in fp64**. `compose_dora` retained solely as that oracle (and
by depth-dependence tests that inspect `W^(r)` directly).

### 5.3 Step conditioning

- **Learned step embedding.** `φ` is `nn.Parameter` initialised *to* the sinusoid,
  so the ordering prior is the starting point and training can move it. Excluded
  from weight decay in both optimisers — it is 2-D and would land in the decay
  group by shape, and decaying it collapses every step's code onto one point.
- **Per-step RMSNorm.** `pre` is `[r_ceiling][6]`, one set per recurrence. All
  init weight 1 ⇒ unchanged behaviour until training separates them.

## 6. Pre-norm, input injection, threshold halting

These took the model from "cannot learn" to "learns" (§11).

**Pre-norm residual.** `stage_dyn` returns `h + down` raw. It previously returned
`out_norm(h + down)` — the block's one post-norm, applied up to 32× per forward.
Post-norm pins ‖h‖ constant, so the identity path can never dominate and every
recurrence injects a full-magnitude update forever. Post-LN transformers have
gradients that explode with depth at init (Xiong et al. 2020); a recurrence
amplifies it.

**Measured: gradient norm 151,951 → 71.7 at init.** With `clip_grad_norm_(1.0)`
the old value divided every update by ~10⁵, leaving no usable learning signal —
which is why the model both collapsed at high LR and refused to learn at low LR.

`out_norm` now normalises **the router's input only**, at commitment boundaries:
the router is a bare Linear with no norm of its own, and the stream grows under
pre-norm, so otherwise its logits grow with depth and the softmax saturates. Single
norm, not per-step — per-step copies would leave non-boundary steps with no
gradient.

**Input injection.** `h ← Block(h) + h₀`. Every position retains its distinct
signal at any depth. Standard for recurrent-depth LMs (Geiping et al. 2025).

**Threshold halting.** Injection makes the recurrence a clean contraction, so
`rel_error` falls **monotonically forever** and `active &= (cur < best)` can never
fire (measured: mean depth 104 vs a 64 cap). Corollary: the old rule only ever
worked because convergence was noisy enough to stall.

Measured decay, untrained:

| r | 2 | 4 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|
| `rel_error` | 0.3947 | 0.0954 | 0.0300 | 0.0178 | 0.0116 | 0.0053 |

So halting tests both: `active = active & (cur < best) & (cur > halt_tol)`.

`halt_tol` swept on the real model (cap 32, B2/B3 mean): 0.02 → 26.9/8.1,
0.03 → 25.3/6.8, 0.05 → 20.6/6.1, **0.08 → 14.9/5.9**, 0.12 → 11.6/5.6. Shipped
0.08. **Provisional** — swept untrained, because that was all that existed.

## 7. Routing

**Router (§11.0).** One per block position. Single Linear, no bias, softmax,
**fp32 even under autocast**, init std 0.01 (near-uniform start ⇒ collapse must be
learned, not handed over at step 0).

**Selection.** `argmax` at each commitment boundary. No entry router: the first
`r_free` recurrences run unadapted.

**Gate multiply (§11.3).** `argmax` is non-differentiable ⇒ without a gate the
router gets zero gradient from `L_LM`. `router_gate="straight"` uses
`g/g.detach()`: identity forward, correct `d/dg`, no ~1/n scaling of the stream.

**Load balancing (§10).** `L_aux = N·Σ f_i P_i`, weight 0.01, floor exactly 1.0
when uniform. `f_i` discrete (argmax counts, no gradient), `P_i` continuous and
carries it. Per-hypernet fractions and router entropy logged every eval — §10 is
explicit that collapse is invisible in the loss curve.

### Gathered dispatch ⇒ single-GPU

Each hypernetwork runs only on tokens it owns (`nonzero`/`index_copy`): O(B·S)
total, not O(n·B·S). Unconditional dispatch measured ~6 min/step.

`if idx.numel() == 0: continue` is a **data-dependent branch**. Two ranks with
different routing skip different hypernetworks and desync the all-reduce, so this
design and DDP are mutually exclusive. The gather won.

**All multi-GPU code removed** — no `torch.distributed`, `torchrun`, rank guards,
sharding. Restoring it means giving up the gather first.

## 8. Tokenizer and data

**128,000-vocab BPE** over 8 corpora: BabyLM 2026, WikiText-103-raw, AMPS-Khan,
xlam-function-calling-60k, reasoning-core/procedural-warmup, CodeSearchNet,
WikiTableT, izumi-lab/open-text-books.

Rule-checked over the whole vocabulary: digits are always separate single-character
tokens; punctuation/symbols separate; neither merges with letters or each other.

WikiTableT has no HF mirror (1.1 GB Drive zip) and ships pre-BPE'd with 30k merges
(`Au@@ tism`). `extract_wikitablet.py` strips `@@ ` markers; adding it changed
**18,291 / 128,000 tokens (14%)**, forcing a pack-cache rebuild.

BabyLM splits come from the organisers' own train/dev repos — never randomly split
(same-document sentences would leak across both sides).

BabyLM packed: 326,456 train blocks (167.1 M tokens), 33,326 dev (17.1 M) @
`seq_len=512`. That cache predates the move to 1024.

**Default domain `wikitext2`** (`Salesforce/wikitext`, `wikitext-2-raw-v1`):
36,718 / 3,760 / 4,358 rows, 10.89 M chars train, 2,670 blocks @ 1024. Organiser
splits used directly by `dev_iter`, not sliced from train. `raw` is required — the
non-raw configs are word-level with `<unk>` substitution and destroyed casing.

Chosen for **iteration speed, not to build a model on**: ~2.7 M tokens ≈ 0.06
tokens per non-embedding parameter (Chinchilla-optimal ~20), so it memorises rather
than generalises, and a 128 k vocab leaves most embedding rows with no gradient.
Buys a ~1 h epoch vs BabyLM's ~4.5 days.

## 9. Training infrastructure

- **Checkpointing** — `best.pt` (best dev only) and `last.pt` (resume point), both
  via `os.replace` so a crash mid-write cannot destroy the previous file.
  Auto-resume: rerunning the identical command continues from `last.pt`.
- **`--set KEY=VALUE`** — override any `Config` field; types from the dataclass,
  unknown key or bad cast is a hard error.
- **EWC** on the shared scaffold for sequential domains; routed-block params
  unprotected. `L_LM + (λ/2)·Σ F_j (θ_j − θ_j*)²`.
- **Single GPU only.** No distributed path (§7).
- **TF32 + `cudnn.benchmark`** on CUDA. TF32 keeps fp32's exponent range, trims
  mantissa; benchmark pays a one-off search for shapes fixed after step 1.
- **Mixed precision** — `resolve_amp` per hardware. T4 (sm_75) has no bf16 and no
  TF32 ⇒ fp16 + GradScaler. `torch.cuda.is_bf16_supported()` returns True on a T4
  via *emulation*; the check uses compute capability ≥ 8.
- **`--max-hours`** times a short window then re-fits the step count so the cosine
  schedule anneals fully.
- **Gradient checkpointing stays on** — a throughput win, not only memory: off =
  214 tok/s @ batch 1 (13.57 GB); on = 141 @ batch 1 but **433 @ batch 4** (8.20
  GB). Freed memory buys a larger batch than the recompute costs.
- **Batch 2 × accum 2 @ seq 1024** (16 GB card): batch 1 = 217 tok/s / 4.82 GB,
  2 = 395 / 7.91, 3 = 553 / 11.11, 4 = OOM. 2 not 3 because depth is data-dependent
  and 3 has no headroom. What runs out is the logits tensor, not the model.
- **Context length is not architectural.** `seq_len` appears nowhere in `Config`:
  NoPE ⇒ no positional table, KDA state is size-invariant. Raising it costs a
  repack. Measured @ batch 1: 512 = 131 tok/s / 3.56 GB, 1024 = 217 / 4.82,
  2048 = 342 / 8.71, 4096 = OOM — longer is *more* efficient per token.
- **Kaggle** — `machine_shape="NvidiaTeslaT4"`; the default P100 (sm_60) is
  unsupported by PyTorch 2.10. Single GPU used.

## 10. Tests

41 passing.

| suite | pins |
|---|---|
| `test_model` | KDA chunked == exact recurrence, causality, correlated-key blow-up, chunk-64 safety, **drifted-gate exactness**, AttnRes uniform at init, DoRA identity at init, **factorised DoRA == reference @ 1e-10 fp64**, halting bounds, halted tokens frozen, embedding tying, shipped widths run, 50 M budget + hypernet/body ratio |
| `test_routing` | router gradient + fp32, budget never exceeded, commitments never interrupted, warm-up unadapted, every hypernet gets gradient, attention once per recurrence, **per-token content sensitivity**, aux floor 1.0, halting fires and depth varies |
| `test_bf16` | fp16/bf16 finite, close to fp32, adversarial KDA under autocast, disabled is a true no-op |
| `test_learning` | overfits a fixed batch, depth is data-dependent, checkpointing matches plain |
| `test_resume` | kill and rerun continues; best.pt tracks best dev; writes atomic |

## 10b. Bugs found by measurement

| bug | symptom | fix |
|---|---|---|
| **post-norm in the recurrence** | gnorm 151,951 clipped to 1.0; collapse @ lr 3e-3, no learning @ 3e-4 | pre-norm residual; `out_norm` → router input |
| **halting cannot fire on a smooth contraction** | mean depth 104 vs a 64 cap after input injection | absolute `halt_tol` alongside "still improving" |
| **`gcum` clamp in the wrong units** | per-step clamp permits `chunk`× the intended chunk total; real run hit 23.3 | floor `_G_RANGE/chunk` |
| generated weight materialised per token | 536 MB/invocation; 14.95 GB peak | `dora_linear`, closed-form row norm |
| KDA chunk math downcast by autocast | 30 matmuls in fp16 despite `promote_types`; product 4.9e8 vs fp16's 65504 | scan wrapped in `autocast(enabled=False)` |
| hypernet recomputed per expert-turn | 256 identical 10.3 MB tables retained = 2.6 GB, OOM | hoisted; later superseded |
| ~528 GPU syncs/forward | `keep.any()`, `torch.unique(r_tok)`, `int(r)` per expert per turn | one host transfer per slot-turn |
| fp32 `ctx` into a bf16 buffer | `index_copy_` dtype mismatch under autocast | cast at the write |

**Tests that were wrong** (passed while the bug was present):
- `assert std >= 0` on depth is vacuous; green while every token ran to the cap.
- all KDA tests ran at init, where the gate has not drifted. The gate is learned;
  a real run passed the safe range by step 250 undetected.

**A metric that was wrong.** Collapse (mean pairwise cosine between per-position
logit vectors) read **0.995 on a healthy run** — every position shares a large
common component (unigram bias) that swamps the position-specific part. Now
centred (subtract mean logit), cross-checked against `same-token`, the fraction of
positions sharing one top-1 prediction.

## 11. The learning failure — RESOLVED

Loss fell to the **unigram marginal** and froze, gradient → 0. Plateau was exactly
`log(n_distinct)`: 6.2451 vs a 6.236 floor. Confirmed directly — **all 512
positions predicted the same token** (`same-token 1.000`).

Two hypotheses tested and **refuted**:

| hypothesis | how it died |
|---|---|
| NoPE leaves no positional signal | not permutation-equivariant; token-0 change propagates to every later position |
| structural oversmoothing / rank collapse | at init, **15 recurrences produce 0.0004 collapse** — training drove it, not the architecture |

Cause: **post-norm inside the recurrence** (§6). Gradient norms ~1.5e5 against
`clip=1.0` meant the update direction was dominated by the exploding component —
explaining both failure modes at once (large steps collapsed the model, small steps
did nothing).

Stagewise collapse (0 = distinct, 1 = identical):

| after | collapsed | fixed |
|---|---|---|
| embed | +0.0003 | +0.0002 |
| B1 | +0.6166 | +0.0005 |
| B2 | +0.5957 | +0.0084 |
| B_mid | +0.6130 | +0.0086 |
| B3 | **+1.0000** | +0.0797 |
| B4 | +1.0001 | +0.1132 |

B1's 0.62 was an *effect* of downstream collapse, not a cause — 0.0005 now with no
change to `SharedBlock`.

Overfit, shipped config, 512 tokens:

```
step   0   loss 11.8090   gnorm  71.70   depth 13.4[10-18]/6.0[4-6]
step  30   loss  6.6703   gnorm   5.06        <- earlier runs died here, at 6.24
step  35   loss  4.5719   gnorm  30.21        <- through the unigram floor
step  50   loss  0.5219   gnorm   1.00
step 149   loss  0.0112   gnorm   0.00   depth  6.2[4-8]/4.0[4-4]
```

**final 0.0112, beats the 6.236 floor.** 218 tok/s, 3.55 GB.

**Attribution caveat:** pre-norm and input injection landed together. Pre-norm is
credited on mechanism (injection has no route to a 2100× gradient reduction), not
by ablation.

## 12. Open

1. **No real-data training run completed.** Every pre-fix number measured a model
   that could not learn.
2. **LR wants re-tuning.** 1e-3 + 20-step warmup works for the overfit;
   `train.py` defaults to 3e-4 + 500. The old sweep measured post-norm dynamics
   that no longer exist — discard it.
3. **`halt_tol = 0.08` is provisional** — swept untrained (§6). A trained run
   showed depth collapsing to 4.7/4.0, i.e. too loose.
4. **Logits tensor `[B, seq, 128000]` caps the batch**, not the model. Chunked
   cross-entropy raises it directly.
5. **KDA gate bound is active**, not slack: a real run sat at `gcum` 9.4 of 10, so
   the model wants to forget faster than the bound allows. A Triton port (§4c)
   removes the constraint.
6. **Hypernet/body split** 24.15 M vs 6.82 M — untested whether correct.
7. **Domain list and order** for sequential training (§14.5) — order matters due
   to the scaffold-drift limitation in §9.
