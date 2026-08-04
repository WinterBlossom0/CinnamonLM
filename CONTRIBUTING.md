# Modifying Cinnamon

`architecture.md` = what the model is. This = how to change it.

## Map

| change | file |
|---|---|
| widths, depths, caps, tolerances | `cinnamon/config.py` (single dataclass, source of truth) |
| recurrence / routing / halting | `cinnamon/routed.py` |
| block contents | `cinnamon/blocks.py` |
| adapter weight generation | `cinnamon/hypernet.py` |
| linear attention | `cinnamon/kda.py` |
| MLA, fixed FFN | `cinnamon/attention.py` |
| training loop | `train.py` |
| corpora | `cinnamon/data.py` `DATASETS` |

## Config overrides

Do not edit `config.py` for experiments:

```bash
python train.py --set d_model=512 --set n_hypernets=4 --set no_halt=true
python pilot/run_pilot.py --set c_max2=16      # passed through
```

Types from dataclass fields. Unknown key or bad cast → `SystemExit`. `--set` wins
over overlapping flags (`--c-max2`).

## Pilot adapts to Config

`pilot/run_pilot.py` reads cfg from the run's checkpoint (`train.py` stores
`cfg.__dict__`). Thresholds derived, not hardcoded:

- depth cap ← `c_max2`
- retention ← `kda_chunk`
- router-collapse floor ← `0.25/n_hypernets`
- missing metric → check skipped, not failed

Not configurable: the go/no-go criterion (dev loss < unigram entropy). Architecture-independent.

## Load-bearing invariants

| invariant | site | failure mode if broken |
|---|---|---|
| pre-norm in recurrence (`stage_dyn` returns `h + down`) | `blocks.py` | post-norm → gnorm 1.5e5 vs clip 1.0; no learning |
| `turns >= 2` | `config.py` | first post-switch reading straddles two weight sets; at 1 no clean reading exists, halting dead |
| `r_ceiling >= c_max2` | `config.py` | `r` indexes step embedding + per-step norms; both clamp → silent reuse of last step |
| `lora_alpha == 2*rank` | `Config.__post_init__` | `dora_scale` fixed at 2; overriding `rank` alone 4x's adapter strength |
| gate floor `_G_RANGE/chunk` | `kda.py` `_project` | chunked path diverges from exact recurrence; no NaN, no loss spike |
| `unit_row_col_norm` on base not update | `hypernet.py` `compose_dora` | all `r` produce identical matrix; dynamic FFN becomes static |
| router gate multiply | `routed.py` boundary | `argmax` has no gradient; router never trains |
| `grad_checkpoint=True` | `config.py` | not just memory: enables larger batch, 433 vs 214 tok/s |

## Tests

```bash
python -m pytest tests/ -q                # 41 tests, ~9 min CPU
python -m pytest tests/test_model.py -q   # ~10 s
```

CPU-only and tiny by design; they pin control flow and invariants, where this
design fails while still producing a plausible loss curve.

Two prior tests that passed while broken:
- `assert std >= 0` on depth is vacuous; stayed green with every token at cap.
- all KDA tests ran at init, where the gate has not drifted. Gate is learned; a
  real run passed the safe range by step 250 undetected. **Test learned quantities
  at trained values.**

## Sanity check before trusting a result

```bash
python overfit_check.py --lr 1e-3 --warmup 20 --steps 150
```

512 tokens, expect loss < 0.5. Plateau at `log(n_distinct)` = model emitting
unigram marginal, ignoring context — indistinguishable from normal training in a
loss curve. `perm_check.py` and the `collapse` / `same-token` metrics discriminate.

## Comment style

Comments state *why* + the measurement that settled it. Stale rationale is worse
than none. Update the comment when changing the decision.
