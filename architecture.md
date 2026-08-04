# CinnamonLM — implemented architecture

What is actually built and measured, as distinct from the spec
(`recurrent-moe-architecture.md`), which is the design intent. Where the two
differ, that is called out explicitly.

Section numbers in `§n` refer to the spec.

There is **one model**. An earlier plan pretrained a single expert and then
assembled a routed bank from several such; that staging is gone, and what follows
is the whole architecture rather than a stage of it.

---

## 1. Top-level shape

```
tokens → Embedding → B1 → B2 → B_mid → B3 → B4 → Norm → LM head
                     ↑     ↑     ↑      ↑     ↑
                  shared recur shared recur shared
                   once   ·32   once   ·32   once
```

- **B1, B_mid, B4** — shared scaffold, separate parameter sets, run exactly once.
- **B2, B3** — routed block *positions*. Each holds **one expert body**, a bank of
  `n_hypernets` hypernetworks, and a router that picks between them. B2's body and
  B3's body are **different weights** (§4.1), which is why recurrence counters
  reset between them.

### Widths and the 50 M budget

Widths are scaled down from the spec's `d_model=512 / d_ff=2048`. The target is
**50 M excluding the embedding and LM head**: both are `vocab × d_model` and vocab
is fixed at 128 k by the tokenizer, so neither is a layer size anyone can tune —
and at these widths they would otherwise dominate the budget, capacity spent on a
vocabulary a model this size cannot exploit.

Embeddings are **untied**: the LM head is its own `[128000, 256]` matrix.

Measured at the shipped config (`Config()`, `param_report()`):

| | |
|---|---|
| shared scaffold (B1+B_mid+B4) | 13.65 M |
| expert body (B2+B3) | 6.82 M |
| 8 hypernets (B2+B3) | 24.15 M |
| routers | 0.00 M |
| **non-embedding total** | **44.63 M** |
| embedding + LM head | 65.54 M *(untied)* |
| grand total | 110.17 M |

Ratios are preserved — `d_ff = 4d`, `q_lora = d/2`, `kv_lora = d/4`,
`lora_alpha = 2·rank` — so nothing about the architecture changes, only widths.
**256 = 8 × 32**, power-of-two width and head count. DoRA rank stays at 16.

**Watch the hypernetwork/body ratio.** The hypernetwork's output layer is
`r_ceiling × (2·rank·(d + d_ff) + d + d_ff)` — **linear in `d`** — while the body
it modulates is **quadratic**. Shrinking the model inverts their ratio: at d=512 a
hypernet is 0.37× the body, at d=224/rank-16 it is 0.83×, and at d=160 it exceeds
the body outright. `test_default_config_hits_the_50m_budget` asserts the ratio
stays < 1, because the failure is silent — the model still trains, it just spends
its budget generating adapters instead of on the layers being adapted.

The bank still outweighs the body collectively: **8 hypernets are 24.15 M against
a 6.82 M body.** Most parameters generate adapters. Whether that split is right is
open — see §11.

---

## 2. Token depth

**Depth is how many *recurrences* one token spends at one block position.**
It is decided per token (§7), so two tokens in the same batch routinely take
different amounts of compute. This is the point of the architecture and also its
main cost risk.

### The unit: one recurrence

```
h → KDA → FixedFFN → MLA → DynamicFFN(r) → h'      (pre-norm, residual raw)
```

### The loop

```
r = 1 … r_free(2)     plain base weights, no hypernetwork selected
then, forever:        one commitment = turns(2) recurrences under one hypernetwork
budget c_max2         = 32 recurrences per block position
```

A commitment is never interrupted: halting is only tested at its end, and the
router picks the next hypernetwork immediately after. So a token's depth is always
`r_free + k·turns` — with the shipped values, an **even number in [2, 32]**.

### Two counters, easy to confuse

| counter | scope | range | resets |
|---|---|---|---|
| `depth` | per **token**, per block position | 2…32 | at each block position |
| `r` (recurrence index) | the loop counter — same for every token | 1…32 | at each block position |

`r` indexes the hypernetwork's step embedding and the per-step RMSNorms, which is
why `r_ceiling` must be ≥ `c_max2`; both clamp at `r_ceiling` rather than indexing
off the end.

A halted token simply stops being updated while the loop continues for everyone
else, so at recurrence `r` all live tokens are at the same `r`. What differs
between tokens is **when they stop** and which hypernetwork was chosen for them.

### Measured depth

Trained, 512 tokens, `c_max2=32`: **B2 6.2 [4–8], B3 4.0 [4–4]** at convergence,
having started at 13.4 [10–18] / 6.0 [4–6] on the same batch at step 0. Depth
*falls* as the model learns the sequence, which is what adaptive depth is for —
fewer recurrences once the answer is easy.

---

## 3. Shared blocks (B1, B_mid, B4)

```
KDA → FFN → KDA → FFN → KDA → FFN → MLA → FFN
```

Kimi Linear's 3:1 ratio: three linear-attention layers for cheap compressed
sequence processing, then one MLA to restore global interaction. Pre-RMSNorm on
every sublayer. **AttnRes** (§3.2) replaces plain residual accumulation with
softmax attention over depth — Full variant, single head, softmax kernel, RMSNorm
on keys, pseudo-queries initialised to **zero** so it starts as an equal-weight
average.

**NoPE throughout** — no positional encoding anywhere. Position enters through
KDA's inherently ordered recurrent state, and every MLA in the model has a KDA
upstream of it in the same block.

That this actually works is measured, not assumed (`perm_check.py`): permuting the
input changes the output (max |diff| 2.36, i.e. **not** permutation-equivariant),
and changing token 0 alone moves every later position by ~3.6e-01 at all
distances. The positional pathway is live.

---

## 4. KDA (Kimi Delta Attention)

Gated delta rule with **channelwise** decay:

```
S_t = (I − b_t k_tᵀ k_t) Diag(a_t) S_{t−1} + b_t k_tᵀ v_t
```

Computed chunkwise: the gate is factored out inside a chunk, leaving an ungated
delta rule with asymmetric keys that the UT transform solves in closed form. Cost
is `seq/chunk` sequential steps instead of `seq`.

Two numerical fixes here were load-bearing, both found by measurement:

**(a) The inverse.** `(I+M)⁻¹` by repeated squaring overflows fp32 at chunk 64
(measured: `inf`; 33% error at 32). That had forced `kda_chunk=16`, costing 4× more
sequential steps — and the chunk loop is the hottest sequential path in the model.

Replaced with **blocked forward substitution**: invert 16×16 diagonal blocks where
squaring is stable, recover off-diagonals by substitution. Chunk 64 is now safe,
and the result is *more* accurate than what it replaced (fp64 at C=64: 2.84e-13 vs
7.18e-13). **Measured 2.22× end-to-end speedup, identical math.**

**(b) The decay clamp.** The chunked form splits decay into `k·exp(g)` and
`k·exp(−g)`; the second grows exponentially and overflowed, so it was clamped. But
once the clamp binds, the cumsum no longer equals true cumulative decay and the
chunked path silently stops matching the recurrence — **1.4% of values clamped
produced 6.25e-02 error**, with no NaN and no loss spike to notice.

Fixed by removing the *need* for the clamp: the state-read and state-write factors
are `exp(gcum)` and `exp(g_end − gcum)`, both with non-positive exponents, so they
are bounded by 1 and now carry **no clamp at all**. Only the pairwise intra-chunk
term must be split, and centring it on the chunk midpoint is exact (the shift
cancels in the product) and halves the range.

| decay | chunk | before | after |
|---|---|---|---|
| init | 64 | 3.33e-07 | 3.34e-07 |
| init | 128 | **6.25e-02** | **3.25e-07** |

**(c) The gate bound.** The split above is accurate while the per-chunk total
`|gcum|` stays inside ~10. That was enforced by a per-step clamp of 10 — which is
`chunk` times too loose, since `gcum` is a *sum over the chunk*. A real training
run walked straight through it: **`gcum` 23.3 by step 250**, where the chunked
path had silently stopped matching the exact recurrence (measured 1e-1 relative
error at 9.5, 4e-1 at 18) with no NaN and no loss spike to notice it by.

Kimi Linear's own kernel has no such limit — it **subtracts before
exponentiating**, `exp2(b_gn - b_gk)`, one exponential with the difference inside,
so the exponent is never positive. Implemented that way here it is exact at any
drift (**1.5e-15 relative error at `|gcum|` 640, 64× the old range**) and
unaffordable: without a kernel it needs a `[C, C, D]` intermediate and a reduce
instead of a tensor-core matmul, measured at **2.4× the time and 1.5× the memory**
(395 → 165 tok/s, 7.9 → 12.1 GB), OOMing outright at chunk 64.

So the fast split is kept and the bound is enforced properly instead — a floor on
log-decay of `_G_RANGE / chunk` rather than `_G_RANGE`, which is fla's own
`safe_gate` idea tightened to what the PyTorch split needs. `|gcum|` is now
bounded by construction. The cost: retention per token cannot fall below
`exp(-10/64) = 0.855`, so the model cannot learn to forget faster than ~15% per
token — though it can still forget essentially everything across a full chunk
(`exp(-10)` = 4.5e-5).

**This is not what Kimi did; it is what Kimi did that is reachable without writing
a Triton kernel.** A kernel port removes the bound entirely, and is the right fix
if the constraint ever proves to cost accuracy.

`gcum` is still logged every eval, but it is no longer a numerical watchdog — it
cannot degrade any more. It is now a *modelling* signal: retention/token is
`exp(-gcum/chunk)`, so a large value means KDA has stopped acting as memory.

---

## 5. The expert block and the hypernetwork bank

There is **one expert body** per block position, not `n` copies. What the router
selects is **which hypernetwork generates the DoRA** for the dynamic FFN.

```
r ≤ r_free    KDA → FixedFFN → MLA → DynFFN(plain base W)
r > r_free    KDA → FixedFFN → MLA → DynFFN(DoRA from hypernet i)
              ↳ at each commitment boundary the router reads the state and
                picks the hypernetwork for the next `turns` recurrences
```

The body is 2:1 KDA:MLA. `stage_attn` (KDA/FixedFFN/MLA) is split from `stage_dyn`
(the dynamic FFN, pointwise) so that attention runs **once per recurrence rather
than once per hypernetwork**. Under a multi-expert design each expert needed its
own keys and values over the whole history (§4.3) — that is §4.3's requirement, not
an implementation artefact, so batching could only make it cheaper, never remove
it. One body removes it.

`test_attention_runs_once_per_recurrence_not_once_per_hypernet` asserts
`stage_attn` calls == `depth.max()`, because a regression there is invisible in the
loss and silently costs n× the attention.

### 5.1 Per-token generation — §5.5 deliberately overridden

The spec makes the hypernetwork **content-blind**: its only inputs are `φ(r)` and
the base column norms, so one call produces the whole `r`-table and every token at
the same `r` gets byte-identical weights.

That is **overridden**. `Hypernet.forward_token(r, content, base_norms)` takes the
per-token hidden state and returns a **distinct generated code per token**, every
recurrence. There is no discretisation step between the content signal and the
weights applied, so gradient reaches the content path for real —
`test_generation_is_content_sensitive_per_token` asserts both that two tokens in
one sequence generate different codes and that `ctx_proj` receives gradient.

The content signal is `h` itself, RMS-normalised — the state entering the
recurrence, which is already per-token and already context-mixed by B1 and every
prior recurrence. An earlier version spent a whole dedicated KDA (0.81 M params
plus a full KDA forward *per recurrence*) producing this signal; raw `h` is free
and measured no worse.

### 5.2 The generated weight is never materialised

DoRA composes `V = Wn + s·B·A` then `W^(r) = m ⊙ V / ‖V‖`. Building that per token
means an `[out, in]` matrix each — at `d_ff=1024, d_model=256` about 1 MB per token
per projection, **~536 MB for a 512-token batch**. That is what put peak VRAM at
14.95 GB and made gradient checkpointing non-optional.

`dora_linear` never builds it. The value factorises directly:

```
V x = Wn x + s · B (A x)                    largest intermediate [.., rank]
```

The obstacle is DoRA's row normalisation, which appears to need the assembled
matrix. It does not — expanding the square:

```
‖V_j‖² = ‖Wn_j‖² + 2s·⟨Wn_j, (BA)_j⟩ + s²·‖(BA)_j‖²
```

and every term is reachable from the factors alone, with `(BA)_j = Aᵀ B_j`:

```
‖(BA)_j‖²      = B_jᵀ (A Aᵀ) B_j            via G = A Aᵀ,  [rank, rank]
⟨Wn_j,(BA)_j⟩  = Σ_k B[j,k] (A Wnᵀ)[k,j]    via Q = A Wnᵀ, [rank, out]
```

The widest live tensor drops from `[.., out, in]` to `[.., out, rank]` — rank
replaces `d_model`. **Measured peak VRAM 14.95 GB → 3.55 GB.**

This is a pure optimisation, so it is verified as one:
`test_dora_factorised_matches_reference` checks `dora_linear` against the
matrix-building reference to **1e-10 in float64**. `compose_dora` is retained
solely as that oracle (and by the depth-dependence tests, which inspect `W^(r)`
itself rather than its action).

### 5.3 Step conditioning

- **Learned step embedding.** `φ` is an `nn.Parameter` initialised *to* the
  sinusoid rather than replacing it, so the ordering prior is the starting point
  and training can move it. Excluded from weight decay in both optimisers —
  it is 2-D and would otherwise land in the decay group by shape, and decaying it
  toward zero collapses every step's code onto the same point.
- **Per-step RMSNorm.** `pre` is `[r_ceiling][6]`, one set per recurrence rather
  than one shared across all of them. All init at weight 1, so step-0 behaviour is
  unchanged and the sets only diverge if training separates them.

---

## 6. Pre-norm, input injection, and why they mattered

These three changes are what took the model from "cannot learn" to "learns". See
§11 for the full diagnosis.

**Pre-norm residual.** `stage_dyn` returns `h + down` raw. It previously returned
`out_norm(h + down)` — the block's one post-norm, applied once per recurrence, up
to 32 times. Post-norm pins ‖h‖ constant, so the identity path can never grow to
dominate and every recurrence keeps injecting a full-magnitude update forever.
Post-LN transformers have gradients that explode with depth at initialisation
(Xiong et al. 2020); a recurrence amplifies that enormously.

**Measured: gradient norm 151,951 → 71.7 at init.** With `clip_grad_norm_(1.0)`
the old value meant every update was divided by ~10⁵, leaving no usable learning
signal — which is why the model both collapsed at high LR and refused to learn at
low LR.

`out_norm` still exists but now normalises **the router's input only**, at
commitment boundaries. The router is a bare linear with no norm of its own, and
the stream grows under pre-norm, so without it the router's logits grow with depth
and its softmax saturates. It is a single norm, not per-step: per-step copies would
leave the non-boundary steps' parameters with no gradient at all.

**Input injection.** `h ← Block(h) + h₀`, where `h₀` is what entered the block.
Every position permanently retains its own distinct signal no matter how deep it
goes. Standard for recurrent-depth LMs (Geiping et al. 2025 inject the embedding
at every step for the same reason).

**Threshold halting.** Injection makes the recurrence a clean contraction toward a
fixed point, so `rel_error` falls **monotonically forever** and the old rule —
`active &= (cur < best)`, "is the error still falling?" — can never fire. Measured:
mean depth 104 against a 64 cap. The uncomfortable corollary is that the old rule
had only ever worked because convergence was *noisy* enough to stall.

Measured decay (untrained, per recurrence):

| r | 2 | 4 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|
| `rel_error` | 0.3947 | 0.0954 | 0.0300 | 0.0178 | 0.0116 | 0.0053 |

Smooth, monotone, still falling at r=32. So halting now tests both:

```python
active = active & (cur < best) & (cur > halt_tol)
```

`halt_tol` swept on the real model (cap 32, B2 mean / B3 mean): 0.02 → 26.9/8.1,
0.03 → 25.3/6.8, 0.05 → 20.6/6.1, **0.08 → 14.9/5.9**, 0.12 → 11.6/5.6. Shipped at
**0.08**, which keeps both blocks well under the cap with real per-token spread.
**Provisional** — swept on an *untrained* model, because that was all that existed
to sweep on. It is a compute/quality knob and wants revisiting.

---

## 7. Routing

**Router (§11.0).** One per block position. Single linear layer, no bias, softmax,
**fp32 even under mixed precision**, init std 0.01 — small-random rather than zero,
so the distribution starts near-uniform and collapse has to be learned rather than
handed over at step 0.

**Selection.** `argmax` at each commitment boundary. There is no entry router: the
first `r_free` recurrences run on plain base weights, so nothing needs selecting
until the state has been shaped a little.

**Gate multiply (§11.3).** `argmax` is non-differentiable, so without a gate the
router receives zero gradient from `L_LM` and never trains. `router_gate="straight"`
uses `g/g.detach()` — identity in the forward pass, correct `d/dg` — so the router
learns without the residual stream being scaled by ~1/n at every boundary.

**Load balancing (§10).** `L_aux = N · Σ f_i P_i`, weight 0.01, floor exactly 1.0
when both are uniform. `f_i` is discrete (argmax counts, no gradient); `P_i` is
continuous and carries it. Logged with per-hypernet fractions and router entropy
every eval, because §10 is explicit that collapse is invisible in the loss curve.

### Dispatch is gathered — and this is why the model is single-GPU

Each hypernetwork runs only on the tokens it actually owns (`nonzero` /
`index_copy`), so total work across all `n` calls is O(B·S) once rather than
O(n·B·S). Running every hypernetwork on every token was measured at ~6 min/step.

The `if idx.numel() == 0: continue` is a **data-dependent branch**. Two ranks with
different routing would skip different hypernetworks and desync a gradient
all-reduce, so this design and DDP are mutually exclusive. The gather won: it is
what makes the model affordable to run at all.

**All multi-GPU code has been removed** — no `torch.distributed`, no `torchrun`
path, no rank guards, no sharding. Restoring it means giving up the gather first.

---

## 8. Tokenizer and data

**128,000-vocab BPE**, trained on all 8 corpora: BabyLM 2026, WikiText-103-raw,
AMPS-Khan, xlam-function-calling-60k, reasoning-core/procedural-warmup,
CodeSearchNet, **WikiTableT**, izumi-lab/open-text-books.

Enforced by rule check on the whole vocabulary: **digits are always separate
single-character tokens; punctuation/symbols are separate; neither ever merges with
letters or with each other.**

WikiTableT needed real work: it has no HF mirror (a 1.1 GB Google Drive zip), and
its text ships **pre-BPE'd with 30k merges** (`Au@@ tism`, `child@@ 's`).
`extract_wikitablet.py` strips the `@@ ` markers to recover natural text — adding it
changed **18,291 of 128,000 tokens (14%)**, which is why the pack cache had to be
rebuilt.

**BabyLM 2026 splits** come from the organisers' own `train`/`dev` repos and are
never randomly split — a random split would put sentences from the same document on
both sides and leak.

Packed: **326,456 train blocks (167.1 M tokens)**, 33,326 dev blocks (17.1 M), at
`seq_len=512`. That cache predates the move to 1024 and needs rebuilding.

**The default domain is now `wikitext2`** (`Salesforce/wikitext`,
`wikitext-2-raw-v1`): 36,718 / 3,760 / 4,358 rows, 10.89 M characters of train
text, with organiser-defined splits that `dev_iter` uses directly rather than
slicing train. The `raw` config is required -- the non-raw ones are word-level
with `<unk>` substitution and destroyed casing.

It is chosen for **iteration speed, not for building a model on**: ~2.4 M tokens
is ~0.05 tokens per non-embedding parameter against a Chinchilla-optimal ~20, so
it will memorise rather than generalise, and a 128 k vocabulary trained on eight
corpora leaves most embedding rows with no gradient at all. What it buys is a
~1 h epoch instead of BabyLM's ~4.5 days, which is what a first real run needs.

---

## 9. Training infrastructure

- **Checkpointing** — `best.pt` (best dev only, never overwritten by a later worse
  eval) and `last.pt` (resume point), both written via `os.replace` so a crash
  mid-write cannot destroy the previous file. **Auto-resume**: rerunning the
  identical command continues from `last.pt`.
- **EWC** on the shared scaffold for sequential domain training; routed-block
  parameters are unprotected. Objective `L_LM + (λ/2)·Σ F_j (θ_j − θ_j*)²`.
- **Single GPU only.** No distributed code path at all — see §7.
- **TF32 and `cudnn.benchmark`** are enabled on CUDA. TF32 keeps fp32's exponent
  range and only trims mantissa bits; benchmark mode pays a one-off kernel search
  for shapes that stop changing after the first step.
- **Mixed precision** — `resolve_amp` picks per hardware. T4 (sm_75) has **no bf16
  and no TF32**, so it gets fp16 + GradScaler. `torch.cuda.is_bf16_supported()`
  returns True on a T4 via *emulation*, which would have silently wasted an entire
  run; the check uses compute capability ≥ 8.
- **Wall-clock budget** — `--max-hours` times a short window, then re-fits the step
  count so the cosine schedule always anneals fully.
- **Gradient checkpointing stays on**, and it is a throughput win, not just a
  memory one. Measured on a 16 GB card at seq 512: off gives 214 tok/s at batch 1
  (13.57 GB); on gives 141 tok/s at batch 1 but **433 tok/s at batch 4** (8.20 GB).
  The memory it saves buys a larger batch, which more than repays the recompute.
- **Batch 4 is the default**, because batch 8 OOMs. What runs out is the logits
  tensor `[B, 512, 128000]` and its fp32 copy in the loss, not the model.
- **Kaggle** — `machine_shape="NvidiaTeslaT4"`. The default GPU is a **P100**,
  which PyTorch 2.10 cannot use at all (`sm_60` dropped).

---

## 10. Test coverage

40 passing, no expected failures.

| suite | what it pins |
|---|---|
| `test_model` (17) | KDA chunked == exact recurrence, causality, correlated-key blow-up, chunk-64 safety, AttnRes uniform at init, DoRA identity at init, **factorised DoRA == reference to 1e-10 (fp64)**, halting bounds, halted tokens stay frozen, embedding tying honoured both ways, shipped widths run, 50 M budget + hypernet/body ratio |
| `test_routing` (12) | router gets gradient and stays fp32, budget never exceeded, commitments never interrupted, warm-up runs unadapted, every hypernet gets gradient, attention runs once per recurrence, **generation is per-token and content-sensitive**, aux loss floor 1.0, halting fires and depth varies |
| `test_bf16` (6) | fp16/bf16 finite, close to fp32, adversarial KDA survives autocast, disabled is a true no-op |
| `test_learning` (3) | overfits a fixed batch, depth is data-dependent, checkpointing matches plain |
| `test_resume` (1) | kill and rerun continues; best.pt tracks best dev; writes atomic |

---

## 10b. Bugs found by measurement

| bug | symptom | fix |
|---|---|---|
| **post-norm inside the recurrence** | gnorm 151,951 clipped to 1.0; model collapsed at lr 3e-3 and would not learn at 3e-4 | pre-norm residual; `out_norm` moved to the router's input |
| **halting could not fire on a smooth contraction** | mean depth 104 of a 64 cap after input injection | added an absolute `halt_tol` alongside "still improving" |
| generated weight materialised per token | 536 MB per invocation; 14.95 GB peak; checkpointing mandatory | `dora_linear` — closed-form row norm from the factors |
| KDA chunk math downcast by autocast | all 30 matmuls ran fp16 despite `promote_types`; centred gate factors reach `exp(±10)`, product 4.9e8 vs fp16's 65504 | scan wrapped in `autocast(enabled=False)` |
| hypernet recomputed per expert-turn | 256 identical 10.3 MB tables retained by autograd = **2.6 GB**; OOM | hoisted; later superseded by per-token generation |
| ~528 GPU syncs per forward | `keep.any()`, `torch.unique(r_tok)`, `int(r)` per expert per turn | one host transfer per slot-turn |
| fp32 `ctx` written into a bf16 buffer | `index_copy_` dtype mismatch under autocast | cast at the write, and keep `ctx` in `h`'s dtype |

**Tests that were wrong** — both passed while the bug was present:

* `std >= 0` on depth is vacuously true, and stayed green while every token ran to
  the cap. Replaced with "the rule must bite for the population", plus non-zero
  spread and commitment-boundary landing.
* a DDP test used 4 experts and 32 tokens, so every expert was populated on both
  ranks and nothing ever diverged. It was later replaced, and then deleted
  outright when the model committed to single-GPU.

**A metric that was wrong.** The collapse diagnostic (mean pairwise cosine between
per-position logit vectors) read **0.995 on a perfectly healthy run**, because
every position shares a large common component — the unigram bias — that swamps
the position-specific part. It briefly had a working model diagnosed as collapsed.
Now centred (subtract the mean logit first), and cross-checked against
`same-token`, the fraction of positions sharing one top-1 prediction, which needs
no interpretation.

---

## 11. The learning failure, and its resolution

**RESOLVED.** For a long stretch the model could not learn at the shipped config:
on 512 tokens — trivially memorisable — loss fell to the **unigram marginal** and
froze there, with the gradient decaying to ~0.

The plateau was exactly `log(n_distinct_tokens)`: 6.2451 measured against a
6.236 floor. That is the score of a model emitting the token marginal and ignoring
its input entirely, confirmed directly — **all 512 positions predicted the same
token** (`same-token 1.000`).

Two hypotheses were tested and **refuted**:

| hypothesis | how it died |
|---|---|
| NoPE leaves no positional signal | model is *not* permutation-equivariant, and a change to token 0 propagates to every later position |
| structural oversmoothing / rank collapse from repeated attention | at init, **15 recurrences produce 0.0004 collapse**. The architecture does not smooth; training drove it there |

The cause was **post-norm inside the recurrence** (§6). Gradient norms of ~1.5e5
against `clip=1.0` meant the update direction was dominated by the exploding
component and the real learning signal was divided into irrelevance — which
explains both failure modes at once: large steps in that direction collapsed the
model, small steps did nothing at all.

Stagewise collapse, before and after (0 = positions distinct, 1 = identical):

| after | collapsed | fixed |
|---|---|---|
| embed | +0.0003 | +0.0002 |
| B1 | +0.6166 | +0.0005 |
| B2 | +0.5957 | +0.0084 |
| B_mid | +0.6130 | +0.0086 |
| B3 | **+1.0000** | +0.0797 |
| B4 | +1.0001 | +0.1132 |

B1's 0.62 was an *effect* of the collapse downstream, not a cause — it is 0.0005
now with no change to `SharedBlock` at all.

### Overfit, shipped config, 512 tokens

```
step  0   loss 11.8090   gnorm  71.70   depth 13.4[10-18]/6.0[4-6]
step 30   loss  6.6703   gnorm   5.06        ← every earlier run died here, at 6.24
step 35   loss  4.5719   gnorm  30.21        ← through the unigram floor
step 50   loss  0.5219   gnorm   1.00
step 149  loss  0.0112   gnorm   0.00   depth 6.2[4-8]/4.0[4-4]
```

**`final loss 0.0112 — BEATS the floor` (6.236).** 218 tok/s, 3.55 GB peak.

**Caveat on attribution:** pre-norm and input injection landed together, so their
contributions are not separated. Pre-norm is credited on mechanism — injection has
no plausible route to a 2100× gradient-norm reduction — but that is reasoning, not
measurement.

---

## 12. Open problems

1. **Nothing has been trained on real data since the fix.** Every throughput and
   loss number above the fix measured a model that could not learn. The overfit
   test passes; a real run has not been done.
2. **LR wants re-tuning.** `1e-3` with 20-step warmup works for the overfit; the
   `train.py` default is `3e-4` with 500-step warmup. The old sweep measured
   post-norm dynamics that no longer exist and should be discarded.
3. **`halt_tol = 0.08` is provisional** — swept on an untrained model (§6).
6. **KDA gate drift** past retention ≈ 0.86 degrades the chunked path silently.
   Monitored (6.87–6.93 at init, margin intact), not solved.
5. **Memory ceiling is the logits tensor**, `[B, 512, 128000]`, not the model.
   It is what caps the batch at 4; chunked cross-entropy is the fix and would
   raise the ceiling directly.
7. **Hypernet/body split.** 24.15 M generating adapters for a 6.82 M body. Whether
   that is right is untested.
8. **Domain list and order** for sequential training (§14.5) — still to be
   supplied; order matters because of the scaffold-drift limitation in §9.
