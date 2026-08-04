# Pilot — is this architecture worth scaling?

One contained experiment answering one question: **does CinnamonLM actually learn
language, or does it only look like it?**

```bash
python pilot/run_pilot.py              # ~2 h budget (default)
python pilot/run_pilot.py --hours 0.5  # shorter
python pilot/run_pilot.py --smoke      # minutes, wiring check only
```

Everything lands under `pilot/`:

```
pilot/
  run_pilot.py          the driver
  cache/                packed WikiText-2 token blocks
  runs/<name>/          checkpoints, train.log, hist.json
  results/<name>.md     the report
  results/<name>.json   the same, machine-readable
```

Nothing is written outside this directory. `rm -rf pilot/cache pilot/runs
pilot/results` returns the repo to exactly where it started.

It shells out to the real `train.py` rather than reimplementing the loop — the
pilot has to measure the thing that would actually be scaled, not a copy that has
drifted from it.

## The one number that matters

Loss is meaningless without something to compare it against, so the report
computes two baselines **from the run's own dev split**:

| baseline | what it means |
|---|---|
| `log(vocab)` | a model that has learned nothing |
| **unigram entropy** | a model that has learned which tokens are common, and *nothing about context* |

**Beating uniform is trivial. Beating unigram entropy is the whole test.**

This is not a theoretical concern. This architecture spent a long stretch pinned
at exactly the unigram marginal — all 512 positions emitting the same token —
while the loss curve looked like a perfectly normal descent. The cause was a
post-norm inside the recurrence producing gradient norms of ~1.5e5 against
`clip=1.0`. See `architecture.md` §11.

So the verdict gates on that check, not on "loss went down".

## Checks in the report

| check | why it is there |
|---|---|
| beats uniform | sanity; failing means something is broken, not subtle |
| **BEATS UNIGRAM** | **the go/no-go.** Below it the model uses context; on it it is a frequency table |
| still improving at the end | distinguishes "budget ran out" from "converged" |
| depth stays off the cap | adaptive depth is the architecture's premise; pinned at the cap means the halting rule is inert |
| KDA numerics safe | `gcum > 10` silently degrades the chunked path — no NaN, no loss spike |
| router did not collapse | MoE collapse is invisible in the loss curve (§10 of the spec is explicit) |
| load balance near floor | `aux` floor is exactly 1.0 when balanced |
| no non-finite loss | — |

## What a GO does and does not mean

WikiText-2 is ~2.4 M tokens against 44.63 M non-embedding parameters — roughly
**0.05 tokens per parameter**, where Chinchilla-optimal is ~20. A good number here
is therefore partly memorisation, and the 128 k vocabulary (trained on eight
corpora) leaves most embedding rows with no gradient at all.

That is deliberate. The corpus was chosen so an epoch takes **~1 hour instead of
BabyLM's ~4.5 days**, which is what a go/no-go needs. Treat a GO as *a floor on
capability and a licence to spend real compute* — not as a forecast of quality at
scale.

A NO-GO is more informative than it looks: the failing checks say *which* part
broke, and this repo has already had two plausible-sounding hypotheses refuted by
measurement. Measure the next one before acting on it.
