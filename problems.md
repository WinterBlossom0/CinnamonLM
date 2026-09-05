# Open problems

Observed failures only. Every entry has a measurement behind it. Speculation and
untested design questions do not belong here — those live in `architecture.md` §11.

State: after a rewrite to **four transformer layers** (L1, recurrent L2+L3, L4),
RoPE replacing KDA, tied embeddings, Muon+AdamW, and CSA compressed attention.

**One trained run now exists on this architecture**: `20260805-184616`, 1 epoch of
WikiText-2 at seq 1024, 668 steps, 0.87 h. Verdict **GO** -- dev **6.0216** against a
unigram bar of **6.3016**. Numbers marked "run 3" are from it; anything unmarked
predates the rewrite.

**P4 (KDA gate bound) is gone**: KDA was removed, so the problem cannot occur.
Since reversed -- the current architecture is back on canonical KDA (RoPE/CSA
removed instead, see `architecture.md` §3), so the gate-bound question is live
again: see `config.py`'s `kda_gate_lower_bound` comment for the current,
unresolved state of that investigation. This note is kept for the history of
*this* run, not as a claim about what ships now.

---


## P1 — Halting calibration, now under a batch-uniform rule

**The rule changed.** Halting now compares batch MEANS in both training and
inference (`mean(cur) < mean(best)`), so every token in a batch runs the same
depth. Adaptivity moved from within-batch to across-batch. See architecture.md §5.

**Observed, untrained:** it does not fire at all.

| | depth | cap |
|---|---|---|
| batch mean (current) | 32.0 [32–32] | 32 |
| per token (previous) | 29.7 [8–32] | 32 |

Measured on the *previous* 5-block architecture; the 4-layer rewrite has not been
run trained.

Averaging smooths away the per-token noise that made the old rule stop, so the
mean falls monotonically at every boundary.

**This is NOT yet a verdict.** The measurement is on an untrained model, and this
repo has twice been wrong doing exactly that: `halt_tol = 0.08` swept at init
pinned a *trained* B3 to 4.0 (`min == max == 4`, the minimum legal depth), and
every KDA test ran at init and missed a gate drift that appeared by step 250
(that component is now deleted, but the lesson stands). A
trained model converges differently. **The next pilot run decides this.**

**Observed under the old per-token rule (pilot run 2, trained):** mean depth
**24.0 / 22.4 of a 32 cap** — 75% and 70%. So even trained and per-token, the
operating point sat near the ceiling.

**Observed once on the current rule, trained (run `20260805-131040`, stopped at
step ~200):** the recurrent block's depth *did* vary batch to batch once training
started — 22, 10, 20, 12, 16 — against a fixed 32 for the block that never fired.
First evidence the averaged signal is not inert on a trained model. That run's
artifacts were deleted, so this is a log reading, not a reproducible measurement.

**DECIDED (run 3, trained, this architecture, full epoch).** The rule is not inert,
but it stops firing as soon as the model settles:

| | value |
|---|---|
| training steps logged | 27 |
| at the cap (`32.0[32-32]`) | **25** |
| off the cap | 2 — `8.0[8-8]` @ step 125, `14.0[14-14]` @ step 150 |
| eval depth, 9 evals | 32.0 everywhere except **31.6** @ step 249 |

Both off-cap steps fall inside the 200-step warmup. After warmup, depth never left
the cap again. So the question P1 said the trained run would settle is settled: the
signal *can* fire — it fires hard, 8 of 32 — and then never does. That is a
calibration failure, not an inert rule, and it rules out "the mean is too smooth to
ever cross" as the whole story.

The gap between the two off-cap values (8 and 14) and the cap (32) is also the
shape the old per-token rule showed: this signal has no middle. It either stops
almost immediately or not at all.

**Four tests fail on this**, all for the same reason:

| test | asserts |
|---|---|
| `test_halting_actually_fires_and_depth_varies` | mean depth < 0.6·cap, two batches differ |
| `test_gradient_norm_stays_bounded_with_depth` | mean depth barely moves between caps |
| `test_depth_is_data_dependent` | depth varies across batches |
| `test_halted_tokens_stay_frozen` | needs one early halt to have anything to freeze |

Left red deliberately. Two of them previously asserted depth varies *within* a
batch; that contract is invalid under the new rule and was corrected to
across-batch rather than relaxed.

**Both directions of the old rule were measured.** With `halt_tol = 0.08`, depth
fell to 5.2/4.1 and B3 pinned at the floor. Without it, depth pinned near the
ceiling. The threshold had been swept untrained; a trained model converges faster,
so it fired immediately.

**Refuted:** that input injection causes this. Measured, depth is 81% of cap
**with** injection and 77.5% **without** — a 3.5-point difference — while removing
injection cost convergence (fixed-batch overfit < 0.5 → 2.14).

**Candidate fix if the trained run also fails to halt.** Aggregate the per-token
**verdicts** instead of the raw errors — halt when a majority of tokens would
individually halt. Keeps one decision per batch (so the batch still starts and
ends together) while keeping the signal the mean smooths away.

**Also unknown.** Whether the residual problem is in the rule, in the
`min`-over-commitment reduction, or in the RMS normalisation of `ctx` — which makes
`fixed_point_residual` measure *direction* only, discarding magnitude change. A token whose
state is still growing in magnitude but has settled in direction reads as
converged.

---

## P2 — Router collapse, not fixed by `lambda_aux`

**Bank size reduced 8 -> 4.** 8 was never justified anywhere in this repo — no
measurement, no design note. It cost 26% of the non-embedding parameters (1.59 M of
6.13 M) and 3.6x the dispatch time (198 ms at `n=1` -> 715 ms at `n=8` for identical
token-work), defending a specialisation benefit that the rest of this entry shows has
never been validly measured. Non-embedding is now **5.34 M**. Every share/entropy
figure below was recorded at `n=8`; the floor `0.25/n_hypernets` moves 0.031 -> 0.0625.

**Observed (pilot run 1, `lambda_aux = 0.01`):**

```
step   0  entropy 2.069  frac min 0.101   aux 1.000
step 150  entropy 1.974  frac [0.004, 0.015, 0.165, 0.083, 0.274, 0.431, 0.0, 0.028]
step 250  entropy 1.823  frac [0.0, 0.0, 0.0, 0.304, 0.089, 0.214, 0.348, 0.045]
step 375  entropy 1.846  frac min 0.057   aux 1.058
```

Max entropy is `ln 8 = 2.079`. Three of eight at **exactly 0.000** at step 250; `aux`
peaked at **1.336** against a 1.0 floor.

**Observed (pilot run 2, `lambda_aux = 0.1`):** worst share **0.000 @ step 175**
against a 0.031 floor; final 0.067. Worst `aux` 1.202, final 1.008.

**Observed (run 3, same `lambda_aux = 0.1`, 4-layer architecture):** worst share
**0.000 @ step 50**, and it did *not* recover — **final 0.029, still under the
0.031 floor**. Worst `aux` 1.478, final 1.285. Run 2 at least came back to 0.067.

**And a second, separate collapse the checks do not catch.** Router entropy fell to
near zero while load balance stayed near its floor:

| step | 0 | 100 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|---|
| entropy | 2.069 | 0.895 | 0.093 | 0.063 | **0.017** | 0.028 |
| min share | 0.049 | 0.077 | 0.080 | 0.079 | 0.057 | — |

Max entropy is `ln 8 = 2.079`. So tokens are spread evenly *across* the bank while
each individual token's choice is essentially one-hot. That is not the collapse the
`frac` check looks for — the shares stay balanced — and it is not obviously a defect
either: confident routing is what a working router should converge to. **No check
covers it, and no measurement distinguishes it from premature commitment.**

**So the 10× raise did not fix it.** It is the only failing check in an otherwise-GO
run: some hypernets receive no gradient at all for stretches of it.

**The counter-evidence was not evidence.** `inspect_hypernets.py` reported run 2's
bank as "specialised". That verdict was a bug: it compared the raw cosine to fixed
thresholds and printed the untrained control **without using it**. Trained functional
similarity was −0.0007 against a control of −0.0003 — identical. The same code would
have declared an untrained bank specialised.

Its weight metric was worse. Measured at init:

```
weight cosine, ALL params      : 0.6074
weight cosine, phi+slot removed: 0.0004
```

`phi` is the same deterministic sinusoid in every hypernet — 0.13% of the parameters
carrying the whole apparent similarity. Same shared-common-component error as the
collapse metric reading 0.995 on a healthy run.

Both are fixed: the verdict is now a delta against the control, shared-at-init
parameters are detected and excluded, and effective rank is reported. **So there is
currently no valid measurement of whether the bank specialises.** The next run
produces the first one.

**Cost reduced, cause untouched.** Sharing the adapter basis (architecture.md §4.5)
means a starved hypernet now freezes ~0.23 M parameters instead of 1.35 M — the basis
keeps training on every token regardless of routing. That makes collapse cheaper, not
rarer. The router still sends some members exactly zero tokens.

**Coupling to P1.** Depth controls how many commitment boundaries a token crosses,
and each boundary is one router decision. P1 changes the amount of routing signal per
forward, so P2 may be partly downstream of it.

---

## P3 — Small margin over the unigram bar

**Observed, three runs, same corpus and same bar (6.3016):**

| run | architecture | dev | margin |
|---|---|---|---|
| 1 (0.60 ep, contended, `halt_tol` bug) | 5-block | 6.2969 | 0.005 — **NO-GO** |
| 2 (1 ep) | 5-block | 6.1587 | 0.143 |
| **3 (1 ep)** | **4-layer** | **6.0216** | **0.280** |

The rewrite roughly doubled the margin, in 0.87 h against run 2's ~5 h. Still a
small margin in absolute terms, and ppl 412 is not a usable model.

**Expected, not necessarily a defect.** ~2.7 M tokens against 6.13 M non-embedding
parameters is 0.45 tokens/param against a Chinchilla-optimal ~20, and **69,933 of
128,000 embedding rows receive no gradient at all** (58,067 ids appear; 83,832 are
seen at most once) — the vocab is 84% of the model. Listed because it is the number
that decides whether to scale, not because the cause is unknown.

Run 3 was the last WikiText-2 pilot: the default corpus is now BabyLM 2026
Strict-Small at seq 16384, so the next margin will be against a different bar.

---

## P4 — Weight generation costs more than the layer it generates

**Observed.** 395 → **140 tok/s** at batch 2 × seq 1024 after `code_dim` 32 → 192,
turning a ~2 h pilot epoch into ~5.5 h.

**Cause, and it was predictable.** The shared basis was justified on parameters
(−4.75 M) and the compute side was never checked. Sharing removes an 8×
*duplication*, not 8× of work — a token only ever used the one hypernet it was
routed to, so per-token cost was always one `code_dim × n_out` matmul. Widening it
6× multiplies the dominant term 6×:

| MACs / token / recurrence | | share |
|---|---|---|
| basis @ `code_dim=32` | 1,351,680 | 31% |
| basis @ `code_dim=192` | 8,110,080 | **73%** |
| expert body (approx) | 3,014,656 | |

Predicted 2.55×, measured 2.82×.

**Reverted to `code_dim=64`** (1.95× faster than 192, 5.67 M fewer params, still
2× the broken original's rank). Unmeasured either way — `inspect_hypernets.py`
reports effective rank on a trained checkpoint, and that is what decides whether
64 is enough or 192 was warranted.

**The structural half is fixed.** Generation cost 3.4× the FFN it becomes, because
`n_out` was 42,240 numbers per token against a 256-wide activation, 78% of it
`rank·d_ff`. **`rank` 16 → 4** cuts `n_out` to 11,520 and generation to **0.94× the
FFN** — cheaper than the layer it produces. `kv_lora_rank` 64 → 32 alongside it.
What that trades is adapter expressiveness, and nothing measures that yet.

**Still unmeasured:** whether `code_dim=64` is enough. `inspect_hypernets.py`'s
effective-rank readout decides it and needs a trained checkpoint; run 3 produced the
first one on this architecture, and the readout has not been run against it.

---

## P5 — Logits tensor caps batch size

**Observed.** batch 4 @ seq 1024 OOMs on a 16 GB card; batch 3 = 11.11 GB. What runs
out is `[B, seq, 128000]` and its fp32 copy in the loss, **not the model** (3.55 GB).

**Fix.** Chunked or fused cross-entropy with recompute in backward. Naive chunking
retains each chunk's graph and saves nothing.

---

## P6 — Dispatch is launch-bound, and batching only helps while it is

**Observed.** Total token-work is independent of `n_hypernets` — each token goes
to exactly one hypernet — yet forward time scales with it:

```
n_hypernets   1      2      4      8
forward      198ms  253ms  400ms  715ms      (3.6x for identical work)
```

That gap is kernel-launch overhead: 8 dispatches per recurrence × `c_max2`
recurrences = 256 gathered dispatches per forward.

**Addressed, with a caveat.** The bmm path is now the default for training as
well as eval (gradients pinned to the loop's at 1e-10 in fp64) and is **4.14×
faster end-to-end at seq 1024**. But it materialises `[n, K, n_out]` for the whole
bank, so it inverts once memory-bound:

| `n·K·n_out` | 11.8M | 23.6M | 47.2M | 94.4M | 189M |
|---|---|---|---|---|---|
| bmm vs loop | 4.57× | 2.96× | 1.70× | 1.21× | **0.72×** |

Auto-selected by `bmm_max_elems`. At 16384 it declines and runs the loop.

**Still open:** no KV cache. Generation re-encodes the whole prefix every token,
and every one of those forwards runs the full recurrence, so cost grows
quadratically. This is why `pilot/sample.py` hard-caps at 256 tokens.

**Measurement caveat.** These are laptop-GPU timings taken back to back; the same
loop config measured 33.6 s and 59.3 s per step at 16384 in two runs, so absolute
numbers drift with thermals. The *ratios* were taken within a single run and are
the trustworthy part.

---

## Suggested order

1. **P1** — the operating point, and now the best-characterised of these: run 3
   showed it fires only during warmup, then never. P2 and P6's cost both scale with
   depth, so this is upstream of two others.
2. **P2** — measure whether it survives a P1 fix before reaching for a bigger
   `lambda_aux`; 0.01 → 0.1 already failed, and run 3 ended *below* the floor rather
   than recovering. The entropy collapse needs a measurement before it needs a fix.
3. **Effective rank on run 3's checkpoint** — decides P4's remaining question
   (`code_dim=64`) and produces P2's first valid specialisation measurement. One
   command against an existing artifact, no training.
4. **P5**/**P6** are ceilings, not correctness blockers.

## Fixed

- **Input injection restored, `halt_tol` deleted.** Both had been removed on the
  theory that injection prevented halting from firing. The measurement refuted it
  (81% of cap with, 77.5% without) and removal cost convergence (overfit < 0.5 →
  2.14). Halting is spec 12 alone again: continue while the error falls, stop when
  it stops.
- **Depth misread as per-block when it was a sum** over B2+B3 — reported as
  "104 against a 64 cap", which is impossible. It was 104/128.
- **KDA crashed on any prompt whose length was not a multiple of 16**
  (`chunk 17 must be a multiple of inverse block 16`). Invisible in training, which
  only ever sees `seq_len` multiples. `C` now rounds up to `_INV_BLOCK`.
- **`DOMAIN` NameError in the report path** destroyed run 2's verdict after 5 h of
  training. Fixed, and `--report-only` added so a verdict is recoverable from
  `hist.json` without retraining.
- `train.py` dropped the final eval from `hist.json`, so the pilot reported 6.5026
  (step 250) instead of the actual best 6.2969 (final).
- Router-collapse check sampled only the last log entry and reported PASS while three
  hypernets had hit exactly 0.000 mid-run. Now worst-over-run.
- `--set c_max2=N` collided with `--c-max2` (`got multiple values`). `--set` wins.
- Cosine annealing with warmup added to `overfit_check.py` and `test_learning.py`;
  both had been running at a constant LR, which cannot settle into a minimum.
  `train.py` already had it.
- **Parameter counts hardcoded in six files** went stale the moment the architecture
  changed. `run_pilot.py` now computes it; the rest are prose and were corrected.
