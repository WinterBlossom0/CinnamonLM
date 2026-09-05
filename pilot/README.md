# Pilot — go/no-go on scaling

One contained experiment: **does CinnamonLM learn language, or only look like it?**

```bash
python pilot/run_pilot.py                      # 1 epoch (default)
python pilot/run_pilot.py --epochs 3
python pilot/run_pilot.py --smoke              # ~minutes, wiring only
python pilot/run_pilot.py --domain wikitext2 --seq-len 1024   # fast iteration
```

Defaults: **BabyLM 2026 Strict-Small**, **seq 16384**. KDA's attention is already
linear in sequence length; the logits tensor is what still scales with context,
so the pilot sets `loss_chunk` automatically (an explicit `--set` wins). Without
it the `[B,16384,32768]` logits tensor alone is 1.07 GB bf16 plus a 2.15 GB fp32
copy.

Budget is **epochs, not wall clock**: steps derived from the packed block count,
so runs are comparable regardless of GPU contention (the same 2 h window gave
1.04 epochs idle and 0.60 with a game running). `--max-hours` = optional safety
valve, off by default.

```
pilot/
  run_pilot.py          driver
  cache/                packed token blocks
  runs/<name>/          checkpoints, train.log, hist.json
  results/<name>.{md,json}
```

Nothing written outside `pilot/`. `rm -rf pilot/{cache,runs,results}` restores the
repo. Shells out to the real `train.py` — measures what would be scaled, not a
drifted copy.

## Criterion

Two baselines computed from the run's own dev split:

| baseline | meaning |
|---|---|
| `log(vocab)` | learned nothing |
| **unigram entropy** | learned token frequencies, **nothing about context** |

**Beating uniform is trivial; beating unigram entropy is the test.** The verdict
gates on this and non-finite loss only. This architecture previously sat at
exactly the unigram marginal — all positions emitting one token — with a
normal-looking loss curve (`architecture.md` §11).

## Checks

| check | purpose |
|---|---|
| beats uniform | sanity |
| **BEATS UNIGRAM** | **the gate** — uses context vs frequency table |
| still improving at end | separates "budget exhausted" from "converged" |
| router not collapsed | MoE collapse is invisible in loss; floor `0.25/n_hypernets` |
| load balance | `aux` floor is exactly 1.0 |
| no non-finite loss | — |

**There is deliberately no depth check.** Halting calibration is P1's problem,
decided by tests and inspection — not by this experiment, which asks whether the
model learns language. Graded, it only ever added a FAIL row to runs that were
otherwise a clean GO. Depth prints as an ungraded observation line.

Thresholds derive from the run's own `Config` (read from its checkpoint), so they
track architecture changes. Missing metric ⇒ check skipped, not failed.

## Interpreting a GO

Token budget against 6.13 M non-embedding params is far under Chinchilla-optimal
~20 tokens/param either way (WikiText-2's 2.7 M tokens = 0.45), and the 128 k vocab
leaves most embedding rows with no gradient. A good number is partly memorisation.

GO = floor on capability + licence to spend real compute, **not** a forecast of
quality at scale.

NO-GO: read which check failed. Two plausible hypotheses have already been refuted
by measurement in this repo — measure the next one before acting.
