"""Did the hypernetwork bank specialise, or collapse to n copies of one function?

    python pilot/inspect_hypernets.py                    # latest run, best.pt
    python pilot/inspect_hypernets.py --run <name>

The bank is 19.92 M of 40.39 M non-embedding parameters.  If all n hypernets
learned the same mapping, that budget bought nothing and n should be smaller.

Three measurements, because they answer different questions:

  WEIGHT similarity  -- cosine between flattened parameters, EXCLUDING parameters
    that are identical across hypernets at init (see `shared_at_init`).  Weak
    evidence on its own (two networks can compute the same function with different
    weights), but a HIGH value is still damning.

  FUNCTIONAL similarity -- feed every hypernet the same (r, ctx, base_norms) and
    compare the generated DoRA codes.  Whether the router's choice actually
    changes the weights the token gets.

  EFFECTIVE RANK -- how many dimensions one hypernet's generatable codes actually
    span.  A property of the architecture, not of training.

EVERYTHING IS READ AGAINST THE UNTRAINED CONTROL.  This script previously compared
the raw cosine to fixed thresholds and printed the control without using it, so it
reported "specialised" at a value an untrained bank also produces.  Random networks
of these shapes already generate near-orthogonal codes; only DISTANCE FROM THE
CONTROL is attributable to training.

Note what none of this can show: that the specialisation is USEFUL.  Low similarity
is consistent with eight well-differentiated experts and with eight untrained ones.

Read-only.  Writes pilot/results/<run>.hypernets.md.
"""
import argparse
import itertools
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)


def pairwise_cos(vecs):
    """Mean / min / max cosine over all distinct pairs."""
    v = torch.stack([x / (x.norm() + 1e-12) for x in vecs])
    sim = v @ v.t()
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean()), float(off.min()), float(off.max())


def shared_at_init(ctrl_block):
    """Parameter names byte-identical across hypernets in the UNTRAINED control.

    `phi` is the same deterministic sinusoid in every hypernet and `err_slot` is
    zeros, so they carry no between-hypernet information -- but they dominate a
    flattened cosine.  Measured at init: 0.6074 with them, 0.0004 without.  The
    whole apparent similarity was one shared initialisation.

    Detected rather than hardcoded, so a future shared-init parameter cannot
    silently re-poison this the way `phi` did.  Same failure as the collapse
    metric reading 0.995 on a healthy run (architecture.md 10b).
    """
    hs = ctrl_block.expert.hypers
    if len(hs) < 2:
        return set()
    other = dict(hs[1].named_parameters())
    return {n for n, p in hs[0].named_parameters()
            if n in other and torch.equal(p.detach(), other[n].detach())}


def weight_vecs(block, skip=()):
    """Flatten each hypernet's parameters into one vector, minus `skip`."""
    return [torch.cat([p.detach().flatten().float()
                       for n, p in h.named_parameters() if n not in skip])
            for h in block.expert.hypers]


@torch.no_grad()
def functional_vecs(block, ctx, r):
    """The DoRA weights each hypernet produces for the SAME input.

    Compared AFTER the shared basis, i.e. what the token actually receives rather
    than the intermediate code -- the basis is common to the bank, so comparing
    codes would flatter the differences.
    """
    bn, basis = block.expert.base_norms(), block.expert.basis
    return [basis(h.forward_token(r, ctx, bn)).detach().flatten().float()
            for h in block.expert.hypers]


@torch.no_grad()
def code_rank(block, i, rs):
    """Effective rank of everything hypernet `i` can generate.

    The output is `basis @ (code_dim-wide code)`, so every adapter for every token
    at every depth lies on a linear manifold no wider than `code_dim`.  This is a
    capacity ceiling, not a training failure, and it is the number to watch when
    the width changes -- it was 32 of 42,240 output dims before the basis was
    shared, which is what motivated sharing it.

    Probed over varied r AND varied content AND varied halting errors, so a rank
    below `code_dim` means the hypernet is not even using what it has.
    """
    h, basis = block.expert.hypers[i], block.expert.basis
    bn = block.expert.base_norms()
    n_out = sum(basis.splits)
    outs = []
    for r in rs:
        for _ in range(8):
            ctx = torch.randn(1, 4, h.c.d_model)
            code = h.forward_token(r, ctx, bn, torch.rand(1, 4), torch.rand(1, 4))
            outs.append(basis(code).reshape(-1, n_out))
    M = torch.cat(outs).double()
    sv = torch.linalg.svdvals(M)
    return int((sv > sv[0] * 1e-7).sum()), n_out


def report_block(tag, block, ctrl, ctx, rs, w):
    w(f"### {tag}\n")
    n = len(block.expert.hypers)

    skip = shared_at_init(ctrl)
    tm, tlo, thi = pairwise_cos(weight_vecs(block, skip))
    cm, clo, chi = pairwise_cos(weight_vecs(ctrl, skip))
    w("| | mean | min | max |")
    w("|---|---|---|---|")
    w(f"| weights, trained | **{tm:+.4f}** | {tlo:+.4f} | {thi:+.4f} |")
    w(f"| weights, untrained control | {cm:+.4f} | {clo:+.4f} | {chi:+.4f} |")
    if skip:
        w(f"\nExcluded as identical at init: `{'`, `'.join(sorted(skip))}` "
          f"-- see `shared_at_init`.\n")

    w("Generated codes for identical input (functional similarity):\n")
    w("| r | trained mean | trained max | control mean | trained - control |")
    w("|---|---|---|---|---|")
    deltas = []
    for r in rs:
        fm, flo, fhi = pairwise_cos(functional_vecs(block, ctx, r))
        cfm, _, _ = pairwise_cos(functional_vecs(ctrl, ctx, r))
        deltas.append(fm - cfm)
        w(f"| {r} | **{fm:+.4f}** | {fhi:+.4f} | {cfm:+.4f} | **{fm - cfm:+.4f}** |")
    w("")

    # The delta column is the whole point.  Random networks of these shapes
    # already produce near-orthogonal codes, so a near-zero absolute cosine is the
    # CHANCE level, not evidence of anything.  Only movement away from the control
    # is attributable to training.
    rank, out_dim = code_rank(block, 0, rs)
    w(f"Effective rank of hypernet 0's generatable codes: **{rank}** of {out_dim} "
      f"output dims.\n")
    return sum(deltas) / len(deltas), rank, out_dim, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--domain", default="wikitext2")
    p.add_argument("--which", default="best", choices=["best", "last"])
    a = p.parse_args()

    from cinnamon.config import Config
    from cinnamon.model import CinnamonModel

    runs = sorted(d for d in os.listdir(RUNS)
                  if os.path.isdir(os.path.join(RUNS, d))) if os.path.isdir(RUNS) else []
    run = a.run or (runs[-1] if runs else sys.exit("no runs"))
    path = os.path.join(RUNS, run, f"{a.domain}.{a.which}.pt")
    if not os.path.exists(path):
        sys.exit(f"no checkpoint at {path}")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]) if isinstance(ck.get("cfg"), dict) else Config()
    m = CinnamonModel(cfg)
    m.load_state_dict(ck["model"])
    m.eval()

    torch.manual_seed(0)
    ctrl = CinnamonModel(cfg).eval()          # untrained control, same shapes
    ctx = torch.randn(1, 16, cfg.d_model)
    rs = [r for r in (1, cfg.r_free + 1, cfg.c_max2 // 2, cfg.c_max2)
          if 1 <= r <= cfg.r_ceiling]
    rs = sorted(set(rs))

    lines = []
    w = lines.append
    w("# Hypernetwork bank: specialised or collapsed?\n")
    w(f"run `{run}` / {a.which}.pt, step {ck.get('step')}, "
      f"dev {ck.get('dev_loss', float('nan')):.4f}, {cfg.n_hypernets} hypernets\n")
    w("Cosine +1.0 = identical, 0.0 = unrelated. The untrained control is the "
      "chance level for these shapes.\n")

    means = []
    for tag, blk, ctl in (("L2+L3", m.expert, ctrl.expert),):
        means.append((tag, *report_block(tag, blk, ctl, ctx, rs, w)))

    w("## Reading\n")
    for tag, d, rank, out_dim, n in means:
        # Judged against the CONTROL, never against an absolute constant.  The
        # previous version compared the raw cosine to 0.8/0.98 and printed the
        # control without using it, so it reported "specialised" at a value the
        # untrained model also produced -- it would have passed an untrained bank.
        if d > 0.30:
            v = (f"**{tag}: COLLAPSING.** Generated codes are {d:+.3f} more alike "
                 f"than the untrained control. Training is driving the bank toward "
                 f"one function; the router's choice is losing its effect.")
        elif d > 0.05:
            v = (f"**{tag}: converging** ({d:+.3f} vs control). Some shared "
                 f"structure is expected -- they adapt the same base weights -- but "
                 f"the trend is toward fewer distinct functions, not more.")
        else:
            v = (f"**{tag}: not collapsed** ({d:+.3f} vs control). "
                 f"**This is not evidence of specialisation.** Randomly initialised "
                 f"networks of this shape already generate near-orthogonal codes, so "
                 f"matching the control means training has not made them measurably "
                 f"more distinct than chance either.")
        w(v + f" Capacity ceiling: rank **{rank}** of {out_dim} output dims.\n")

    w("Caveats:\n")
    w("- Measured on one random ctx batch; real token states could differ.")
    w("- A hypernet the router never selected sits near its initialisation, which "
      "reads as 'distinct' for the wrong reason. Cross-check against the "
      "per-hypernet share in the pilot report.")
    w("- Rank is a property of the ARCHITECTURE, not of training: the output is "
      "`basis @ (bottleneck-wide code)`, so it cannot exceed the bottleneck width "
      "however long the model trains.\n")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{run}.hypernets.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"-> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
