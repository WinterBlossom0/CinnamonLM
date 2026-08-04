# Open problems

State after pilot run `20260804-172619` (NO-GO). Ordered by blocking-ness.
Each entry: evidence, what is known, what is not.

---

## P1 — Adaptive depth is inert

**Evidence.** Depth fell 13.6/6.0 (step 0) → 5.2/4.1 (step 375). B3 pinned at
exactly 4.0, `min == max == 4`, for the entire run. `r_free=2`, `turns=2`, so 4 is
the **minimum legal depth**: every B3 token halted at the first opportunity.
Cap is 32.

**Cause.** `halt_tol = 0.08` was swept on an **untrained** model (config.py records
the sweep). A trained model's hidden state converges far faster per recurrence, so
a threshold calibrated on noise fires immediately on signal.

**Consequence.** The architecture's premise — per-token compute — is not
operating. The model is effectively a fixed 4–5 layer recurrent stack, and every
throughput number is flattered by it.

**Unknown.** Correct value, and whether a fixed scalar is the right mechanism at
all. Candidates: sweep `halt_tol` on a *trained* checkpoint; make it relative to
observed `rel_error` at that step rather than absolute; schedule it.

**Interaction.** Input injection makes the recurrence a clean contraction
(§6 architecture.md), so `rel_error` falls monotonically and `cur < best` can never
fire. `halt_tol` is now the **only** thing that stops a token. It carries the whole
halting decision alone.

---

## P2 — Router collapse

**Evidence.**

```
step   0  entropy 2.069  frac min 0.101   aux 1.000
step 150  entropy 1.974  frac [0.004, 0.015, 0.165, 0.083, 0.274, 0.431, 0.0, 0.028]
step 250  entropy 1.823  frac [0.0, 0.0, 0.0, 0.304, 0.089, 0.214, 0.348, 0.045]
step 375  entropy 1.846  frac min 0.057   aux 1.058
```

Max entropy is `ln 8 = 2.079`. Three of eight hypernets at **exactly 0.000** at
step 250. `aux` peaked at **1.336** against a floor of 1.0.

**Consequence.** ~9 M of 24.15 M hypernetwork parameters received no gradient for a
large part of the run. The bank is the majority of the model's non-embedding
budget.

**Unknown.** Whether `lambda_aux = 0.01` is simply too weak, or whether the
collapse is downstream of P1 (shallow depth ⇒ few routing decisions ⇒ weak
balancing pressure). Note the two are coupled: depth 4 on B3 means **one**
commitment boundary, so B3's router gets one decision per token per forward.

**Note.** Collapse partially recovered by step 375, so a check that samples only
the final log entry reports PASS. Fixed to take the worst over the run.

---

## P3 — Dev loss only reaches the unigram bar

**Evidence.** Final dev **6.2969** vs unigram entropy **6.3016** = **0.0047 nats**
below. Train loss reached 6.1841 and was still descending, but noisy:
6.59 → 6.73 → 6.74 → 6.30 → 6.58 → 6.51 → 6.18.

**Reading.** Inconclusive, not a refutation. The run had P1 and P2 active and saw
only **0.60 epochs** (old wall-clock budget, on a contended GPU). But it is also
not evidence of success: clearing the bar by 0.5% of a nat is what a model does
when it has learned the marginal and a trace of context.

**Unknown.** Whether P1/P2/data-budget explain the gap, or whether something
further upstream limits it. **Rerun on the epoch budget with P1 and P2 addressed
before drawing any conclusion.**

---

## P4 — KDA gate bound is fully binding

**Evidence.** `gcum` 9.4 → **9.9** against a hard bound of `_G_RANGE = 10`. The
clamp is active, not slack: the model is pushing against it continuously.

**Meaning.** Retention/token is floored at `exp(-10/64) = 0.855`. The model wants
to forget faster than the implementation permits.

**Cause.** The intra-chunk term `exp(g_t − g_s)` is evaluated as a matmul, which
forces the split `exp(g_t)·exp(−g_s)`; the second factor is unbounded, so `gcum`
must be bounded to keep the split accurate.

**Known fix.** Kimi's kernel subtracts before exponentiating (`exp2(b_gn − b_gk)`),
so the exponent is never positive and no bound is needed. Measured in PyTorch:
exact at any drift (1.5e-15 @ `|gcum|` 640) but **2.4× time, 1.5× memory**, OOM at
chunk 64 — unaffordable without a Triton kernel.

**Unknown.** Whether the bound actually costs accuracy. Testable: run with
`--set kda_chunk=32` (halves the per-chunk total, so the same `_G_RANGE` permits
2× the per-token forgetting) and compare.

---

## P5 — Learning rate not tuned for the current architecture

**Evidence.** Train loss oscillated ±0.4 between logged steps at `lr=1e-3`.

**Cause.** Every prior LR measurement was taken under post-norm dynamics that no
longer exist (gradient norms were ~1.5e5; they are now ~70). The old sweep is
void. `train.py` defaults to 3e-4/500-warmup; the pilot used 1e-3/200.

**Unknown.** Everything. Needs a fresh sweep, cheaply, on the pilot harness.

---

## P6 — Logits tensor caps batch size

**Evidence.** batch 4 @ seq 1024 OOMs on a 16 GB card; batch 3 = 11.11 GB. What
runs out is `[B, seq, 128000]` and its fp32 copy in the loss, **not the model**
(3.55 GB).

**Fix.** Chunked or fused cross-entropy (compute logits and loss in slices,
recomputing in backward). Raises batch directly, which raises throughput.

**Not attempted.** Straightforward but not free — naive chunking retains each
chunk's graph and saves nothing; it needs recompute.

---

## P7 — Hypernet/body parameter split untested

24.15 M generating adapters for a 6.82 M body. `n_hypernets=8`. No experiment has
tested whether this ratio is right, or whether 4 larger hypernetworks / 16 smaller
ones is better. Cheap to sweep now via `--set n_hypernets=N`.

---

## P8 — Attribution gap: pre-norm vs input injection

Both landed in the same commit. Pre-norm is credited with the 2100× gradient-norm
reduction on mechanism (injection has no plausible route to it), but this was never
ablated. If input injection is *not* pulling weight, it is worth removing — it is
what forced the halting rule to change (P1's interaction note).

**Test.** One run with injection removed, pre-norm kept.

---

## P9 — WikiText-2 is small relative to the model

2.7 M tokens vs 44.63 M non-embedding parameters ≈ **0.06 tokens/param**
(Chinchilla-optimal ~20). A 128 k vocab trained on eight corpora leaves most
embedding rows with no gradient on this corpus.

Deliberate — it buys a ~1 h epoch vs BabyLM's ~4.5 days — but it means a GO here is
a floor on capability, and it may not discriminate well between architectures that
differ mainly in generalisation.

**Consider.** Once P1–P3 are resolved, confirm on BabyLM before scaling.

---

## Suggested order

1. **P1** (halt_tol) — cheapest, and P2 may be downstream of it
2. **P2** (lambda_aux) — re-measure after P1
3. Rerun pilot on the **full epoch budget**, then re-read P3
4. **P5** (LR sweep) if P3 is still marginal
5. P4/P6 are performance ceilings, not correctness blockers
6. P7/P8 are cheap ablations worth running once the above is stable

## Fixed since the last pilot

- `train.py` dropped the final eval from `hist.json`, so the pilot reported 6.5026
  (step 250) instead of the actual best 6.2969 (final).
- Router-collapse check sampled only the last log entry and reported PASS while
  three hypernets had hit exactly 0.000 mid-run. Now worst-over-run.
- `--set c_max2=N` collided with `--c-max2` (`got multiple values`). `--set` wins.
