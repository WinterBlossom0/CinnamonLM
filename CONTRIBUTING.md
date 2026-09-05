# Modifying Cinnamon

`architecture.md` = what the model is. This = how to change it.

## Map

| change | file |
|---|---|
| widths, depths, caps, tolerances | `cinnamon/config.py` (single dataclass, source of truth) |
| recurrence / routing / halting | `cinnamon/routed.py` |
| block contents | `cinnamon/blocks.py` |
| adapter weight generation (`Hypernet` trunk + shared `AdapterBasis`) | `cinnamon/hypernet.py` |
| attention (canonical KDA), fixed FFN (xIELU) | `cinnamon/attention.py` |
| optimiser split | `cinnamon/muon.py` |
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
- router-collapse floor ← `0.25/n_hypernets`
- missing metric → check skipped, not failed

Not configurable: the go/no-go criterion (dev loss < unigram entropy). Architecture-independent.

## Load-bearing invariants

| invariant | site | failure mode if broken |
|---|---|---|
| pre-norm in recurrence (`stage_dyn` returns `h + down`) | `blocks.py` | post-norm → gnorm 1.5e5 vs clip 1.0; no learning |
| `r_ceiling >= c_max2` | `config.py` | `r` indexes step embedding + per-step norms; both clamp → silent reuse of last step |
| `lora_alpha == 2*rank` | `Config.__post_init__` | `dora_scale` fixed at 2; overriding `rank` alone 4x's adapter strength |
| Muon excludes `phi`/`err_slot` | `muon.py` `split_params` | orthogonalising a lookup table blends unrelated recurrence steps |
| `unit_row_col_norm` on base not update | `hypernet.py` `compose_dora` | all `r` produce identical matrix; dynamic FFN becomes static |
| `AdapterBasis` shared per block position | `blocks.py` | per-hypernet it was 87.6% of each hypernet for a rank-`code_dim` space, and a starved hypernet froze all of it |
| `_GEN_STD / sqrt(code_dim)` init | `hypernet.py` `reset_out` | widening `code_dim` silently scales adapter strength at init; capacity change becomes a behaviour change |
| router gate multiply | `routed.py` boundary | `argmax` has no gradient; router never trains |
| halting compares batch means in BOTH modes | `routed.py` | a mode-dependent rule means the model never trains under the rule it is served with |
| `grad_checkpoint=True` | `config.py` | not just memory: enables larger batch, 433 vs 214 tok/s |

## Tests

```bash
python -m pytest tests/ -q                # 39 tests, needs CUDA; ~3 min, mostly Triton JIT warmup
python -m pytest tests/test_model.py -q   # ~10 s once warm
```

Needs CUDA: canonical KDA (`fla`'s Kimi Delta Attention) has no CPU path, unlike
the MLA attention this design used before. Tiny by design otherwise; they pin
control flow and invariants, where this design fails while still producing a
plausible loss curve. A few fail on P1 (untrained model never halts early) --
left red on purpose, see `architecture.md` §9.

Two prior tests that passed while broken:
- `assert std >= 0` on depth is vacuous; stayed green with every token at cap.
- KDA tests ran at init, where the gate had not drifted. Gate was learned; a
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
