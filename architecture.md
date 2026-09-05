# CinnamonLM — implemented architecture

What is built and measured, vs the spec (`recurrent-moe-architecture.md`) which is
design intent. `§n` = spec section. Deviations are marked.

One model; no staged pretraining. Modification guide: `CONTRIBUTING.md`.

## 1. Shape

```
tokens → Embedding → L1 → [L2 + L3] → L4 → Norm → LM head (tied)
                     ↑      ↑    ↑     ↑
                    once  recurrent   once
                          up to ·32
```

**Four transformer layers, four MLAs, four FFNs.** L1 and L4 run once. L2+L3 are
one recurrent block holding the hypernetwork bank and the router; only **L3's FFN
has generated weights**, L2's is fixed. There is one recurrence counter and one
depth value.

| | |
|---|---|
| L1 + L4 | 1.65 M |
| L2 + L3 body | 1.68 M |
| 4 hypernet trunks | 0.80 M |
| shared adapter basis | 2.70 M |
| router | 0.00 M |
| **non-embedding** | **6.83 M** |
| embedding + LM head (untied) | 16.78 M |
| total | 23.61 M |

(`Config().param_report()` is the source of this table — re-run it after any
width change rather than trusting these numbers stale.)

**Embeddings are 71% of the model** — 32,768 vocab against `d_model` 256, untied
head. Was 128,000 (86%, tied) until the vocab shrink below. Measured fresh
against the current `tokenizer.json` on WikiText-2: **25,817 of 32,768 ids ever
appear** (79%, was 45% at 128k), 6,951 rows receive no gradient at all (was
69,933), and 1,606 are seen exactly once (was 83,832). The vocab is still sized
for the eventual 8-corpus mix's tokenizer default (§7), not the pilot corpus —
right-sizing it further needs the actual training corpus, not WikiText-2. Open (§11).

**Weight generation costs more than the layer it generates.** See §4.6 — this is
the model's dominant cost and the least examined part of its design.

**Checkpoint compatibility.** Nothing before this rewrite loads: KDA and the three
shared blocks are gone, `(d2, d3)` became a single depth, and `l3` became a shared
`AdapterBasis`. `tie_embeddings` defaults to False (original untied head) —
despite an earlier version of this doc claiming it flipped to True.

## 2. Depth

Depth = passes through the L2+L3 recurrent block.

```
r <= r_free(1)   plain base weights, no hypernetwork
then             router re-selects every recurrence (turns=1)
cap              c_max2 = 32
```

Halting (fixed-point convergence, arXiv:2606.18206 Algorithm 1) is checked every
recurrence, not just at commitment boundaries; the minimum depth is 1.

`r` indexes the step embedding and per-step norms; both clamp at `r_ceiling`,
which must therefore be ≥ `c_max2`.

**One counter, not two.** There was a second routed position (B3) and a shared
block between them; both are gone, so `forward` returns a single `depth` rather
than `(d2, d3)`.

## 3. Attention

Canonical Kimi Delta Attention (arXiv:2510.26692) via Flash Linear Attention's
differentiable `chunk_kda` Triton kernel (`cinnamon/attention.py`) — Moonshot's
own FlashKDA is inference-only and is not a training backend here. Gated
delta-rule recurrence: per-key-channel decay, a beta write gate, a causal short
convolution, gated output normalisation. The recurrent state and the
convolution carry sequence order, so there is **no positional encoding, no
RoPE, no MLA rank, and no compressed-attention configuration**. This replaced
an intermediate MLA + RoPE + CSA (compressed sparse attention) design that once
lived in this section — that design's own GFLOP/memory measurements do not
carry over and are not repeated here.

The recurrence is **O(T)** compute and memory, not quadratic: 1024 → 16384 is
16× the tokens and ~16× the attention compute, unlike a dense or CSA-style
design here, which would have paid ~256× for the same range. No profiled
GFLOP/tok-s numbers for this backend exist yet in this repo.

**CUDA only.** `KDA.forward` raises if the input is not on a CUDA device; no
approximate CPU path is provided.

**Tuning knobs** (`config.py`, all `kda_*`): `kda_short_conv` / `kda_conv_size`
for the causal convolution; `kda_safe_gate` / `kda_gate_lower_bound`, which
clamps the per-step log-decay gate to `[lower_bound, 0)`. The paper's own
chunked kernel divides by the *cumulative* product of gates across a whole
chunk, not the per-step gate, which the per-step clamp does not bound — a
suspected forward-pass NaN source under bf16 at long training runs, tested by
tightening the bound (-5.0 → -3.0) without a confirmed fix yet; see the field's
own comment in `config.py` for the current, inconclusive result.

## 4. Recurrent block + hypernetwork bank

One body, not `n` copies. The router selects **which hypernetwork generates the
DoRA** for L3's FFN.

```
r <= r_free    KDA → FixedFFN → KDA → DynFFN(plain base W)
r >  r_free    KDA → FixedFFN → KDA → DynFFN(DoRA from hypernet i)
```

`stage_attn` (L2 in full plus L3's attention) is split from `stage_dyn` (L3's FFN,
pointwise) so attention runs **once per recurrence, not once per hypernetwork**.
§4.3 requires each expert to produce its own K/V over the whole history — batching
cannot remove that; one body can.
`test_attention_runs_once_per_recurrence_not_once_per_hypernet` asserts
`stage_attn` calls == `depth.max()`.

### 4.1 Per-token generation — §5.5 deviation

Spec makes the hypernetwork content-blind (inputs `φ(r)` + base column norms) ⇒
one row per `r`, identical weights for all tokens at that `r`.

**Overridden.** `Hypernet.forward_token(r, content, base_norms, err_best, err_last)`
returns a distinct code per token, every recurrence, with no discretisation between
the content signal and the applied weights, so gradient reaches the content path.
`test_generation_is_content_sensitive_per_token` asserts both.

Content signal = `h` itself, RMS-normalised: the state entering the recurrence,
already per-token and context-mixed by B1 and prior recurrences. An earlier version
spent a dedicated summariser (0.81 M params + a forward per recurrence) producing
this; raw `h` measured no worse.

### 4.2 Generated weight never materialised

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

### 4.3 Step conditioning

- **Learned step embedding.** `φ` is `nn.Parameter` initialised *to* the sinusoid,
  so the ordering prior is the starting point and training can move it. Excluded
  from weight decay in both optimisers — it is 2-D and would land in the decay
  group by shape, and decaying it collapses every step's code onto one point.
- **Per-step RMSNorm.** `pre` is `[r_ceiling][6]`, one set per recurrence. All
  init weight 1 ⇒ unchanged behaviour until training separates them.

### 4.4 Halting-error inputs

Input widened from three signals to five:

```
[ φ(r) ; ctx ; norm_proj(‖W_dyn‖_c) ; enc(err_best, 0) ; enc(err_last, 1) ]
```

- **`err_best`** — running minimum `fixed_point_residual`, the value the halting rule
  compares against. Kept **per token** even though the decision itself is reduced
  to a batch mean — averaging these away would discard the per-token convergence
  signal they exist to provide.
- **`err_last`** — the previous recurrence's reading.

`φ(r)` says *where* in the loop the token is; these say *how it is going*. `content`
alone cannot: two tokens can sit at the same state with very different rates of
change, and only the second pair distinguishes "converging" from "stalled".

**Fourier-encoded, not raw.** `fixed_point_residual` is one scalar in ~[0,1]; concatenated raw
it is one input against 320 and is effectively ignored — the same reason `φ(r)` is a
sinusoid rather than an integer. Expanded to `phi_dim` features at geometric
frequencies 1…10³, dense near 0 where errors live once a token starts converging.

**Slot-tagged.** A learned `err_slot [2, phi_dim]` is added after encoding, so the
two arrive distinguishable rather than as interchangeable vectors. Verified: the
same numeric value in different slots produces different generated weights
(‖Δ‖ 0.148 vs 0.099), and both tags receive gradient.

**Read-only.** The halting path is under `no_grad`, so these are conditioning
inputs, not a gradient route. Deliberate: they are what halting compares, so a
gradient into them would let the model lower the *error* rather than the loss —
halt early, skip work, fake convergence. Backpropagating through halting is a real
family of methods (PonderNet, ACT), but it is incompatible with a hand-written stop
rule, which is what §12 specifies.

**Ordering.** Both are read *before* the halting block updates them, so at
recurrence `r` the generator sees only readings from `r−1` and earlier. No leakage
from the current step.

Threaded through both dispatch paths;
`test_batched_dispatch_matches_the_loop` still holds with them present.

### 4.5 Shared adapter basis

```
per hypernet   inputs -> 256 -> code_dim          Hypernet     0.23 M each
shared         code_dim -> n_out                  AdapterBasis 2.70 M
```

The output layer used to be per-hypernet and `r_ceiling`-wide. Measured, that was
**87.6% of each hypernet's parameters (1.35 M of 1.54 M) spent on a rank-32 output
space**:

```
generated codes  (1024 varied inputs)  : (1024, 42240)
numerical rank                         : 32
```

Everything a hypernet could produce — over every token, every `r`, every halting
error — lay on a 32-dimensional linear manifold. The docstring claimed content
was exempt (*"Content enters on a separate width and is not subject to that
bound"*); it is not. There is no separate path, `l1 → 256 → l2 → 32 → l3`.

The original justification — a function of `r` over `r_ceiling` values has rank
≤ `r_ceiling` at any width — was **valid only while generation was content-blind**.
Per-token generation (§4.1) made the input space far larger and the bound stopped
being free. It also tied adapter capacity to the recurrence cap for no reason:
raising `c_max2` forced `r_ceiling` up and doubled the bank as a side effect.
`code_dim` is now its own field.

| | per-hypernet basis | shared basis |
|---|---|---|
| rank available to a hypernet | 32 | **64** |
| bank, per block position | 12.34 M | **9.96 M** |
| non-embedding total | 45.14 M | **40.39 M** (6.23 M after the 4-layer rewrite, code_dim 64, rank 4) |

6× the rank for 4.75 M *fewer* parameters — **but 2.55× the compute**, which the
parameter table hides and which was missed when this landed.

Sharing cut parameters because one matrix stopped being duplicated 8×. It cut no
compute: a token only ever passed through the ONE hypernet it was routed to, so
per-token cost was always a single `code_dim × n_out` matmul. Widening that 32 →
192 multiplied it by 6, on what is the dominant term (since reverted to 64):

| MACs per token per recurrence | | share |
|---|---|---|
| adapter basis, `code_dim=32` | 1,351,680 | 31% |
| adapter basis, `code_dim=192` | 8,110,080 | **73%** |
| recurrent body (MLA + FFNs, approx) | 3,014,656 | |

Predicted 2.55× slower per token-recurrence; **measured 2.82×** (395 → 140 tok/s at
batch 2 × seq 1024), which turns a ~2 h pilot epoch into ~5.5 h. `code_dim` is
therefore a compute knob as much as a capacity one — 64 would be 1.31×, 128 would be
1.93×. Shipped at 64; nothing yet shows any of these widths is used. See §11 and §4.6.

Two further consequences:

- **A starved hypernet no longer freezes 1.35 M parameters.** The basis trains on
  every token regardless of routing; only the ~0.23 M trunk goes idle. Directly
  reduces the cost of **P2**.
- **Dispatch loses its largest batched operand.** The basis applies once to the
  whole `[n, K, code_dim]` result instead of as an `n`-way `bmm` — one large
  matmul, on the path that §6 measured as launch-bound.

Specialisation now lives in the trunk, which is the right place: every hypernet
adapts the *same* base weights, so a shared output basis is a prior rather than a
compromise.

**Init.** `reset_out` scales std by `1/sqrt(code_dim)`, so widening the code does
not silently multiply adapter strength — a generated `A` entry is one basis row
dotted with the code, and its magnitude would otherwise grow with the width. The
anchor is a fixed constant, not the old `r_ceiling`-dependent value: what matters
is `‖s·(BA)_j‖` against `‖Wn_j‖ = 1`, a property of the adapter rather than of the
generator's shape. Consequence: TINY/SMALL init ~2× stronger than before,
deliberately; the default config is unchanged.

### 4.6 Why generation is the dominant cost

The hypernetwork must emit a complete DoRA vector **for every token, every
recurrence**:

| | numbers | share |
|---|---|---|
| `A_up [rank, d]` | 4,096 | 9.7% |
| `B_up [d_ff, rank]` | 16,384 | **38.8%** |
| `m_up [d_ff]` | 1,024 | 2.4% |
| `A_dn [rank, d_ff]` | 16,384 | **38.8%** |
| `B_dn [d, rank]` | 4,096 | 9.7% |
| `m_dn [d]` | 256 | 0.6% |
| **n_out** | **42,240** | |

Emitting 42,240 numbers per token costs `code_dim × n_out` MACs. Using them costs
`2·d·d_ff` (xIELU FFN, no gate matrix -- was `3·d·d_ff` under SwiGLU). So:

| | MACs / token / recurrence | vs using it |
|---|---|---|
| generate @ `code_dim=192` | 8,110,080 | **15.5×** |
| generate @ `code_dim=64` | 2,703,360 | **5.2×** |
| use (xIELU `2·d·d_ff`) | 524,288 | 1.0× |

**The adapter is more expensive to create than the layer is to run.** That is
inherent to per-token generation: the generated object (42,240) is far larger than
the activation it acts on (256). It is not a bug, but it was never accounted for,
and it is why throughput fell 395 → 140 tok/s when `code_dim` went 32 → 192.

**`code_dim` is the cheap lever, `rank` is the real one.** 78% of `n_out` is
`rank·d_ff` (`B_up` + `A_dn`), so halving `rank` halves generation:

| `rank` | `n_out` | generate @ 64 | vs using it |
|---|---|---|---|
| 16 (current) | 42,240 | 2,703,360 | 3.4× |
| 8 | 21,760 | 1,392,640 | 1.8× |
| 4 | 11,520 | 737,280 | **0.9×** |

At `rank=4` generation becomes cheaper than the FFN itself. Untested — `rank` sets
the adapter's expressiveness and `lora_alpha` must track it (`__post_init__`), so
this is a real trade, not a free win. Open (§11).

## 5. Pre-norm, input injection, threshold halting

These took the model from "cannot learn" to "learns" (§10).

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

**Halting is spec 12 alone:** continue while the error falls, stop when it stops.
No tolerance, no threshold.

**The comparison is BATCH-UNIFORM**, in training and inference alike. The reading
is averaged over the batch *first*, and every later step works on those scalars:

```
k_r  = fixed_point_residual(...).mean()   one number per recurrence
cur  = min(k_r) over the commitment
best = min(cur) over all commitments
stop when  cur >= best
```

Order matters: averaging only at the comparison, with per-token mins before it, is
a different and lower rule, since `mean(min over turns) != min(mean over turns)` —
different tokens bottom out on different recurrences. Measured on one batch, that
form reported a commitment min of 0.0236 when no single recurrence averaged below
0.0239.

One scalar decision, so every token in a batch enters and leaves the recurrence
together. `err_best`/`err_last` stay **per token** — they feed the generator, not
the decision. Depth stays adaptive **across** batches, not within one. Two reasons: it
removes a discrete per-token decision that rounding noise can flip (measured:
editing token 40 changed token 12's depth 28 → 16), and it makes depth a property
of the batch rather than of each token's private noise.

Deliberately the same rule in both modes — a mode-dependent one would mean the
model never trains under the rule it is served with.

**Untrained, it does not fire:**

| | B2 | B3 | cap |
|---|---|---|---|
| batch mean | 32.0 [32–32] | 32.0 [32–32] | 32 |
| per token (previous) | 29.7 [8–32] | 31.7 [6–32] | 32 |

Weak evidence, and explicitly *not* being treated as a verdict. This repo has twice
drawn the wrong conclusion from an untrained measurement — `halt_tol` swept at init
pinned a *trained* model to the depth floor, and every KDA test ran at init and
missed a gate drift that appeared by step 250 (that component is now deleted, but
the lesson stands). A trained model converges
differently. Open; see problems.md P1.

If it turns out not to fire when trained, the fix that keeps a batch-uniform
decision is to aggregate per-token **verdicts** rather than raw errors — halt when
a majority of tokens would individually halt. Keeps one decision per batch while
keeping the signal the mean smooths away.

*This section previously documented the opposite, and the reasoning was wrong.* The
claim was that injection makes the recurrence a clean contraction, so `fixed_point_residual`
falls monotonically and the rule can never fire — and injection was removed on that
basis, with an absolute `halt_tol` added to compensate. Both are reverted:

| | with injection | without |
|---|---|---|
| depth as fraction of cap | 81% | 77.5% |
| fixed-batch overfit | **< 0.5** | 2.14 |

Injection barely moves halting and is worth a 4× convergence difference. Halting
firing late is real (**P1**) but is not caused by that line.

`halt_tol` is gone entirely. Swept untrained at 0.08, it pinned a *trained* B3 at
exactly 4.0 — `min == max == 4`, the minimum legal depth — i.e. every token halting
at the first opportunity. `Config` carries a comment where the field was, because
re-adding it is the obvious wrong move.

Measured decay, untrained (L2 formula, pre-`fixed_point_residual` rename -- the
metric is now L-infinity, this table needs remeasuring under it):

| r | 2 | 4 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|
| `rel_error` (L2) | 0.3947 | 0.0954 | 0.0300 | 0.0178 | 0.0116 | 0.0053 |

## 6. Routing

**Router (§11.0).** One per block position. Single Linear, no bias, softmax,
**fp32 even under autocast**, init std 0.01 (near-uniform start ⇒ collapse must be
learned, not handed over at step 0).

**Selection.** `argmax` at each commitment boundary. No entry router: the first
`r_free` recurrences run unadapted.

**Gate multiply (§11.3).** `argmax` is non-differentiable ⇒ without a gate the
router gets zero gradient from `L_LM`. `router_gate="straight"` uses
`g/g.detach()`: identity forward, correct `d/dg`, no ~1/n scaling of the stream.

**Load balancing (§10).** `L_aux = N·Σ f_i P_i`, **weight 0.1**, floor exactly 1.0
when uniform. `f_i` discrete (argmax counts, no gradient), `P_i` continuous and
carries it. Per-hypernet fractions and router entropy logged every eval — §10 is
explicit that collapse is invisible in the loss curve.

Raised from the spec's 0.01: at 0.01 three of eight hypernets reached exactly 0.000
share by step 250 of a real run (entropy 2.069 → 1.823 against a 2.079 max, `aux`
peaking 1.336 over the 1.0 floor). 0.1 did **not** fix it — the GO run still logged
a 0.000 share at step 175. See **P2**.

### Gathered dispatch ⇒ single-GPU

Each hypernetwork runs only on tokens it owns (`nonzero`/`index_copy`): O(B·S)
total, not O(n·B·S). Unconditional dispatch measured ~6 min/step.

`if idx.numel() == 0: continue` is a **data-dependent branch**. Two ranks with
different routing skip different hypernetworks and desync the all-reduce, so this
design and DDP are mutually exclusive. The gather won.

**All multi-GPU code removed** — no `torch.distributed`, `torchrun`, rank guards,
sharding. Restoring it means giving up the gather first.

### Batched dispatch — the default, with a size-based fallback

The gather is correct but **launch-bound**: 8 dispatches per recurrence × `c_max2`
= 256 gathered dispatches per forward, each firing ~15-20 tiny kernels. Holding
total token-work fixed and varying only `n_hypernets`, forward time went
198 → 715 ms for 1→8. Identical arithmetic; the gap is launches.

`forward_token_bank` runs the whole bank as one `bmm` per layer.
**No longer eval-only** — `test_batched_dispatch_gradients_match_the_loop` pins
every parameter's gradient to the loop's at 1e-10 in fp64, which is what the gate
was waiting on. Mutation-checked: a 1e-6 perturbation in the bmm path produces a
7.6e-6 gradient divergence, so the test is not vacuous.

**It is not universally better, and "always use the faster path" would be wrong.**
Batching every group is exactly why it saves launches and why it costs memory —
it materialises `[n, K, n_out]` for the whole bank at once:

| `n·K·n_out` | 11.8M | 23.6M | 47.2M | 94.4M | 189M |
|---|---|---|---|---|---|
| bmm vs loop | **4.57×** | 2.96× | 1.70× | 1.21× | **0.72×** |

It wins while launch-bound and loses once memory-bound. Chunking the bmm does not
rescue it: the chunks reintroduce exactly the launches it exists to remove.

So `_dispatch_bmm` falls back to the loop on two conditions — padding waste
(`n·K > 3·total`, i.e. the router collapsed) and size
(`n·K·n_out > bmm_max_elems`, default 128M). Measured end-to-end with the shipped
defaults: **4.14× faster at seq 1024**, and at 16384 it correctly declines
(0.99×, i.e. it picks the loop).

## 7. Tokenizer and data

**BPE over 8 corpora**: BabyLM 2026, WikiText-103-raw, AMPS-Khan,
xlam-function-calling-60k, reasoning-core/procedural-warmup, CodeSearchNet,
WikiTableT, izumi-lab/open-text-books. `tokenizer_train.py --vocab` still
defaults to 128,000, sized for this eventual full mix; the `tokenizer.json`
actually checked in was rebuilt at **32,768** (§1) to fit the corpora presently
being trained on, so the default and the shipped file now disagree on purpose —
pass `--vocab 32768` to reproduce what's checked in.

Rule-checked over the whole vocabulary: digits are always separate single-character
tokens; punctuation/symbols separate; neither merges with letters or each other.

WikiTableT has no HF mirror (1.1 GB Drive zip) and ships pre-BPE'd with 30k merges
(`Au@@ tism`). `extract_wikitablet.py` strips `@@ ` markers; adding it changed
**18,291 / 128,000 tokens (14%)** — measured against the 128k build before the
vocab shrink, not re-verified at 32,768.

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

## 8. Training infrastructure

- **Checkpointing** — `best.pt` (best dev only) and `last.pt` (resume point), both
  via `os.replace` so a crash mid-write cannot destroy the previous file.
  Auto-resume: rerunning the identical command continues from `last.pt`.
- **`--set KEY=VALUE`** — override any `Config` field; types from the dataclass,
  unknown key or bad cast is a hard error.
- **Muon + AdamW.** Muon on the 66 matmul weights (13.81 M), AdamW on the other
  167 tensors (32.82 M — embedding, RMSNorm weights, biases, and the `phi` /
  `err_slot` tables). The split is **by role, not by shape**: `phi` and `err_slot`
  are 2-D but are lookup tables, and orthogonalising across their rows would blend
  unrelated recurrence steps. Momentum 0.95, Nesterov, 5 Newton–Schulz steps,
  quintic `(3.4445, −4.7750, 2.0315)`, iterated in bf16 on the matrix's short
  side. Frobenius normalisation first guarantees the iteration starts inside its
  convergence region; the update is then scaled by `max(1, out/in)^0.5` because an
  orthogonalised update has RMS `~1/sqrt(max(out,in))` regardless of gradient
  scale. Two optimisers rather than two groups — sane LRs differ by ~100×
  (`--muon-lr` 0.02 vs 3e-4). Both share the cosine+warmup schedule.
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
- **Batch, measured on the 4-layer architecture @ seq 1024** (13 GB allocator cap):
  2 × accum 2 = 702 tok/s / 6.6 GB, 4 × accum 1 = 948 / 12.3. Same effective batch,
  so the 1.35× is pure launch amortisation. What runs out first is the logits
  tensor and its fp32 copy, not the model — hence `loss_chunk`. Pre-rewrite numbers
  (217 / 395 / 553 at batch 1 / 2 / 3, OOM at 4) measured a different model and are
  not comparable.
- **Context length is not architectural.** `seq_len` appears nowhere in `Config`:
  KDA's recurrent state and causal convolution carry order for any length.
  Changing it costs a repack. Longer is *more* efficient per token, since fixed
  per-recurrence overhead amortises: **16384 = 506 tok/s / 5.06 GB** against 217
  at 1024 on the same batch 1 -- measured before the vocab shrink (§1), not
  re-profiled at 32,768. The pilot now defaults to 16384, which only needs
  `loss_chunk` for the `[B,T,32768]` logits tensor -- KDA's attention is already
  linear in T.
- **Kaggle** — `machine_shape="NvidiaTeslaT4"`; the default P100 (sm_60) is
  unsupported by PyTorch 2.10. Single GPU used.

## 9. Tests

**Needs CUDA.** Canonical KDA has no CPU path (§3), unlike the MLA attention
this design used before -- every test that runs a real forward pass now
constructs its model/tensors on `cuda`. The pure AuxLoss/Router math (no
attention involved) stays CPU-testable and untouched.

**39 total; 36 pass, 3 fail — all one cause.** Every failure is the batch-mean
halting rule not firing on an untrained model (§5): mean depth below 0.6·cap,
depth varying across batches, depth bounded across caps. Left red on purpose —
they assert the behaviour of a *trained* model.

Nine tests were deleted with the modules they covered (KDA chunked-vs-recurrence,
gate drift, blocked inverse, AttnRes) and two more (`test_inv_unit_lower`,
`test_shared_parameters_excludes_routed_blocks`) referenced code that no longer
exists. Two further tests (`test_compressed_attention_is_causal`,
`test_query_chunking_is_exact`) covered CSA, which is also gone (§3) — deleted
along with the `attn_compress`/`LatentPool` exemptions two other tests carried
for them.

Also found and fixed while restoring CUDA execution: a fixture bug
(`test_routing.py`'s `ROUTED` set `c_max2=16` but left `r_ceiling=8`, violating
the config's own invariant) and two stale `Config(**kw)` kwargs
(`kv_lora_rank`/`q_lora_rank`, MLA-era fields that no longer exist) in
`test_learning.py`'s `SMALL`. `test_halted_tokens_stay_frozen` and the two
bmm-vs-loop dispatch-equivalence tests dropped from fp64 to fp32 (`fla`'s
chunk_kda Triton kernel has no fp64 path), with correspondingly looser
tolerances (`1e-5`, not `1e-10`).


| suite | pins |
|---|---|
| `test_model` | DoRA identity at init, **factorised DoRA == reference @ 1e-10 fp64**, depth-dependence under training, halting bounds, halted tokens frozen (fp32), embedding tying, shipped widths run, budget + basis/body ratio |
| `test_routing` | router gradient + fp32, budget never exceeded, commitments never interrupted, warm-up unadapted, every hypernet gets gradient, attention once per recurrence, **per-token content sensitivity**, **batched dispatch == loop @ 1e-5 fp32**, aux floor 1.0, halting fires and depth varies |
| `test_bf16` | fp16/bf16 finite, close to fp32, disabled is a true no-op |
| `test_learning` | overfits a fixed batch, depth is data-dependent, checkpointing matches plain |
| `test_resume` | kill and rerun continues; best.pt tracks best dev; writes atomic |

## 9b. Bugs found by measurement

| bug | symptom | fix |
|---|---|---|
| **post-norm in the recurrence** | gnorm 151,951 clipped to 1.0; collapse @ lr 3e-3, no learning @ 3e-4 | pre-norm residual; `out_norm` → router input |
| **a fix built on a misread number** | "injection prevents halting" — the depth figure behind it was a **sum over B2+B3** (104/128, not 104/64), and injection turned out to change halting by 3.5 points | injection restored, `halt_tol` deleted; see §5 |
| **`halt_tol` swept on an untrained model** | 0.08 pinned trained B3 at exactly 4.0, `min == max` — the minimum legal depth | removed; halting is spec 12 alone |
| **`--set c_max2=N` collided with `--c-max2`** | `got multiple values for argument` | `--set` wins |
| **`DOMAIN` NameError in the report path** | pilot run 2 trained 5 h, then crashed writing its verdict | fixed; `--report-only` added so a verdict is recoverable from `hist.json` |
| **chunk not a multiple of the inverse block** | generation crashed on a 17-token prompt; invisible in training, which only sees `seq_len` multiples | round `C` up to `_INV_BLOCK` (component since deleted) |
| **`.gitignore` trailing comments** | `raw/  # comment` is a literal pattern in git, not a comment — a 153 MB file staged | comments on their own line |
| **`gcum` clamp in the wrong units** | per-step clamp permits `chunk`× the intended chunk total; real run hit 23.3 | floor `_G_RANGE/chunk` |
| generated weight materialised per token | 536 MB/invocation; 14.95 GB peak | `dora_linear`, closed-form row norm |
| KDA chunk math downcast by autocast | 30 matmuls in fp16 despite `promote_types`; product 4.9e8 vs fp16's 65504 | scan pinned to fp32 (component since deleted) |
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

## 10. The learning failure — RESOLVED

Loss fell to the **unigram marginal** and froze, gradient → 0. Plateau was exactly
`log(n_distinct)`: 6.2451 vs a 6.236 floor. Confirmed directly — **all 512
positions predicted the same token** (`same-token 1.000`).

Two hypotheses tested and **refuted**:

| hypothesis | how it died |
|---|---|
| NoPE leaves no positional signal | not permutation-equivariant; token-0 change propagates to every later position |
| structural oversmoothing / rank collapse | at init, **15 recurrences produce 0.0004 collapse** — training drove it, not the architecture |

Cause: **post-norm inside the recurrence** (§5). Gradient norms ~1.5e5 against
`clip=1.0` meant the update direction was dominated by the exploding component —
explaining both failure modes at once (large steps collapsed the model, small steps
did nothing).

Stagewise collapse (0 = distinct, 1 = identical):

| after | collapsed | fixed |
|---|---|---|
| embed | +0.0003 | +0.0002 |
| B1 | +0.6166 | +0.0005 |
| B2 | +0.5957 | +0.0084 |
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

## 10b. Pilot — GO

`pilot/` is a self-contained go/no-go on whether this architecture is worth scaling.
One full epoch on WikiText-2. All artifacts stay under `pilot/`.

Bar: **beat the unigram entropy of the run's own dev split.** A model sitting on it
has learned token frequencies and nothing about context — which is exactly where
this architecture was stuck before the pre-norm fix, and is not visible in a loss
curve that still looks like it is descending.

Run `20260804-192902`, **7 of 8 checks pass, verdict GO**:

| | |
|---|---|
| uniform `log(vocab)` | 11.7598 |
| **unigram entropy (the bar)** | **6.3016** |
| **best dev** | **6.1587** (ppl 472.8) |

0.143 nats below the bar ⇒ the model uses context. The one failure is **router
collapse** (worst share 0.000 @ step 175 against a 0.031 floor) — P2, unfixed by
`lambda_aux = 0.1`. Depth 24.0/22.4 of cap 32 is P1.

**What this does not show.** ~2.7 M tokens against 6.23 M non-embedding parameters
is **0.44 tokens/param** (Chinchilla-optimal ~20), and the 128 k vocab leaves most
embedding rows with no gradient. A good number here is partly memorisation. GO is a
floor on capability and a licence to spend real compute, **not** a forecast.

Generation quality is consistent with ppl 473: `"Hi, how are you?"` → two whitespace
tokens then `<eos>`. `pilot/sample.py` runs it; qualitative only, not part of the
verdict.

## 11. Open

Observed failures live in **`problems.md`** (P1–P7), which is measurement-only.
This list is design questions, which do not belong there.

1. **Vocab was 128,000 against `d_model` 256** (untied embeddings were 86% of the
   model, 69,933 rows never got a gradient on WikiText-2). Shrunk to 32,768 (§1):
   now 71% of the model, 6,951 rows dead. Still sized for the eventual 8-corpus
   mix's tokenizer default (§7), not any one pilot corpus, and still sets the
   logits tensor that forces `loss_chunk` at long context — just a smaller one.
2. **LR wants re-tuning.** 1e-3 + 20-step warmup works for the overfit; `train.py`
   defaults to 3e-4 + 500. The old sweep measured post-norm dynamics that no longer
   exist — discard it. Cosine + warmup is now in `train.py`, `overfit_check.py` and
   `test_learning.py`; the last two had been running at constant LR.
3. **`code_dim = 64` is a guess, and so was 192.** §4.5 established that 32 was
   an unjustified *ceiling* — it says nothing about where the real requirement
   sits, so 64, 128 and 192 are all unmeasured. Shipped at 64 because it is 1.95×
   faster and the run that produces the measurement therefore costs half as much.
   `inspect_hypernets.py` reports effective rank on a trained checkpoint: a rank
   saturating at 64 means go wider, one well below it means 192 was waste.
4. **`rank = 16` is the untouched lever.** 78% of `n_out` is `rank·d_ff`, so
   `rank` 16 → 4 would make generating the adapter *cheaper* than running the FFN
   (§4.6) — a far bigger win than any `code_dim` change. It trades adapter
   expressiveness and `lora_alpha` must track it, so it needs measuring, not
   assuming.
5. **`bmm_max_elems = 128M` is a single-point calibration.** The loop/bmm
   crossover was measured on one laptop GPU at batch 1; it will sit elsewhere on
   different memory or batch size, and nothing re-derives it at runtime.
6. **Sequential multi-domain training has no forgetting protection.** EWC (was §9)
   is removed as unused — nothing has trained more than one domain yet. If §14.5's
   sequential plan is still intended, the shared scaffold has no anchor against
   drift between domains; that mechanism would need to come back before a second
   domain is trained, not after.
7. **Halting-error inputs are unmeasured.** §4.4 is verified correct (distinct per
   slot, gradient reaches both tags) but no run has yet shown they help. They were
   added on the argument that the generator should see convergence rate, not on
   evidence that it does.
