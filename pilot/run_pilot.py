"""Pilot: go/no-go on whether this architecture is worth scaling.

    python pilot/run_pilot.py                 # 1 epoch (default)
    python pilot/run_pilot.py --epochs 3
    python pilot/run_pilot.py --smoke         # ~minutes, wiring only
    python pilot/run_pilot.py --domain babylm --set n_hypernets=4

All output under pilot/ (cache, checkpoints, logs, report).  Nothing outside is
written or modified; `rm -rf pilot/{cache,runs,results}` restores the repo.

Budget in EPOCHS, not wall clock: steps derive from the packed block count, so
the verdict does not depend on GPU contention.

Shells out to the real train.py rather than reimplementing the loop -- an earlier
helper in this repo reimplemented training and carried its own bug.  The pilot
must measure what would be scaled, not a drifted copy.

Thresholds derive from the run's own Config (read from its checkpoint), so this
file does not need updating when the architecture changes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
# Default only.  --domain overrides it, and every path below is derived from it,
# so pointing the pilot at a different corpus needs no edits here.
DEFAULT_DOMAIN = "wikitext2"

sys.path.insert(0, ROOT)


def baselines(dev_blocks, vocab):
    """(uniform, unigram) cross-entropy in nats, from the actual packed dev split.

    uniform = log(vocab): learned nothing.
    unigram = entropy of the dev token frequency distribution: learned which
      tokens are common, nothing about context.  A model sitting on this is not
      using context however plausible its loss looks -- where this architecture
      was stuck for a long time.
    """
    import numpy as np
    counts = np.bincount(np.asarray(dev_blocks).reshape(-1), minlength=vocab)
    p = counts[counts > 0] / counts.sum()
    return math.log(vocab), float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- #

def train(args, steps, hours, out_dir):
    """Run the real train.py, contained entirely under pilot/."""
    cmd = [sys.executable, os.path.join(ROOT, "train.py"),
           "--domain", args.domain,
           "--tokenizer", os.path.join(ROOT, "tokenizer.json"),
           "--seq-len", str(args.seq_len),
           "--batch", str(args.batch),
           "--accum", str(args.accum),
           "--steps", str(steps),
           "--lr", str(args.lr),
           "--warmup", str(args.warmup),
           "--eval-every", str(args.eval_every),
           "--log-every", str(args.log_every),
           "--cache", CACHE,
           "--out", out_dir,
           "--no-ewc-consolidate"]
    # Architecture overrides go straight through to train.py, so a pilot can be
    # run against a modified Config without touching either file.
    for kv in args.set:
        cmd += ["--set", kv]
    if hours:
        cmd += ["--max-hours", str(hours)]
    print("+ " + " ".join(cmd), flush=True)
    log_path = os.path.join(out_dir, "train.log")
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace",
                             env=dict(os.environ, PYTHONUTF8="1"))
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()          # or a killed run leaves an empty train.log
        p.wait()
    return p.returncode, log_path


def verdict(hist, uni, base, cfg, smoke=False):
    """go/no-go + per-check evidence.

    Architecture-dependent thresholds come from `cfg` (the run's own config), not
    from constants here, so verdicts stay correct after architecture changes.
    Checks with absent inputs are SKIPPED, not failed -- a model with no router
    must still be able to pass.

    Gate is `BEATS UNIGRAM` + finite loss only: below the unigram entropy the
    model uses context, on it it is a frequency table, and that distinction is
    invisible in a descending loss curve.
    """
    devs = [h for h in hist if "dev_loss" in h]
    trains = [h for h in hist if "loss" in h]
    if not devs:
        return [], None, "no eval ran -- the budget was too short to reach --eval-every"

    cap = cfg.get("c_max2") or 32
    chunk = cfg.get("kda_chunk") or 64
    best = min(d["dev_loss"] for d in devs)
    last = devs[-1]
    checks = []

    checks.append(("beats uniform", best < base - 0.1,
                   f"dev {best:.4f} vs log(vocab) {base:.4f}"))
    checks.append(("BEATS UNIGRAM (uses context)", best < uni - 0.05,
                   f"dev {best:.4f} vs unigram entropy {uni:.4f}"))

    if len(devs) >= 2:
        # Final vs best of the FIRST half.  Indexing the midpoint directly
        # compared devs[-1] to itself when only two evals existed -> FAIL always.
        earlier = devs[:max(1, len(devs) // 2)]
        ref = min(d["dev_loss"] for d in earlier)
        checks.append(("still improving at the end", devs[-1]["dev_loss"] < ref,
                       f"first half best {ref:.4f} -> final {devs[-1]['dev_loss']:.4f}"
                       " (FAIL means converged or diverged, not that it cannot learn)"))

    depth = last.get("dev_depth")
    if depth:
        # Pinned at the cap => halting rule inert.  Cap from cfg, not a constant.
        worst = max(depth)
        checks.append((f"depth stays off the cap ({cap})", 0 < worst < 0.9 * cap,
                       "mean depth " + "/".join(f"{d:.1f}" for d in depth)
                       + f" of cap {cap}"))

    gc = last.get("gcum")
    if gc is not None:
        # Not a numerics check (gate bound keeps the chunked path exact).  Asks
        # whether KDA still carries context: retention/token = exp(-gcum/chunk).
        ret = math.exp(-gc / chunk)
        checks.append(("KDA still acting as memory", ret > 0.5,
                       f"max |gcum| {gc:.1f} over chunk {chunk} -> "
                       f"retention/token {ret:.3f}"))

    routed = [t for t in trains if t.get("routing")]
    if routed:
        fracs = [t["routing"]["frac"] for t in routed if t["routing"].get("frac")]
        if fracs:
            # WORST over the run, not the last sample.  Collapse is transient in
            # the log: a run showed three of eight hypernets at exactly 0.000 at
            # step 250 and recovered to 0.057 by step 375, and sampling only the
            # end reported PASS.
            floor = 0.25 / len(fracs[0])          # quarter of uniform, scales with n
            worst = min(min(f) for f in fracs)
            at = routed[[min(f) for f in fracs].index(worst)]["step"]
            checks.append((f"router did not collapse ({len(fracs[0])} hypernets)",
                           worst > floor,
                           f"worst share {worst:.3f} @ step {at} vs floor "
                           f"{floor:.3f} (uniform {1/len(fracs[0]):.3f}); "
                           f"final {min(fracs[-1]):.3f}"))
        auxes = [t["routing"]["aux"] for t in routed if "aux" in t["routing"]]
        if auxes:
            checks.append(("load balance near floor", max(auxes) < 1.5,
                           f"worst aux {max(auxes):.3f}, final {auxes[-1]:.3f}, "
                           f"floor is 1.0"))

    finite = all(math.isfinite(t["loss"]) for t in trains)
    checks.append(("no non-finite loss", finite, f"{len(trains)} logged steps"))

    gate = dict((n, ok) for n, ok, _ in checks)
    go = (gate.get("BEATS UNIGRAM (uses context)", False)
          and gate.get("no non-finite loss", False))
    if smoke:
        # Wiring only.  NO-GO here would read as evidence against the
        # architecture when it is only evidence that 60 steps is 60 steps.
        return checks, best, "SMOKE"
    return checks, best, ("GO" if go else "NO-GO")


def run_config(out_dir, domain):
    """Config the run used, from its own checkpoint (train.py stores cfg.__dict__).

    Lets checks target what the model IS, not what it was when this file was
    written: a hardcoded "cap 32" becomes a wrong verdict the moment someone sets
    c_max2=16.  Returns {} if absent; every reader supplies a default, so checks
    degrade to skipped rather than crashing.
    """
    import torch
    for f in (f"{domain}.best.pt", f"{domain}.last.pt"):
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            try:
                return torch.load(path, map_location="cpu",
                                  weights_only=False).get("cfg", {}) or {}
            except Exception as e:                       # a corrupt ckpt is not
                print(f"  (could not read cfg from {f}: {e})", flush=True)
    return {}


def pack_and_count(domain, seq_len):
    """Pack up front, return train block count.  Packing before launching
    train.py is what lets the budget be measured in epochs; train.py then finds a
    warm cache."""
    from train import build_blocks
    data, _ = build_blocks(domain, os.path.join(ROOT, "tokenizer.json"),
                           seq_len, CACHE)
    return len(data["train"])


def main():
    p = argparse.ArgumentParser()
    # Budget is DATA, not wall clock: a 2 h window gave 1.04 epochs idle and 0.60
    # with a game running, which changes what a NO-GO means.
    p.add_argument("--epochs", type=float, default=1.0,
                   help="passes over the training set (the budget)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="optional wall-clock safety valve; off by default")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=0,
                   help="0 = auto, about 8 evals across the run")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--domain", default=DEFAULT_DOMAIN,
                   help="corpus key from cinnamon/data.py DATASETS")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Config override, passed through to train.py, repeatable: "
                        "--set n_hypernets=4 --set c_max2=16")
    p.add_argument("--name", default=None, help="run name; defaults to a timestamp")
    p.add_argument("--smoke", action="store_true",
                   help="minutes, not epochs: proves the wiring, proves nothing else")
    a = p.parse_args()

    name = a.name or time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if a.smoke else "")
    out_dir = os.path.join(RUNS, name)
    for d in (CACHE, out_dir, RESULTS):
        os.makedirs(d, exist_ok=True)

    blocks = pack_and_count(a.domain, a.seq_len)
    per_step = a.batch * a.accum
    steps = max(1, math.ceil(blocks * a.epochs / per_step))
    if a.smoke:
        steps, a.warmup, a.log_every = 60, 10, 5
    # Auto cadence: a fixed eval_every gave 2 dev points on a short run and 8 on a
    # long one, so "still improving" meant different things in each.
    if not a.eval_every:
        a.eval_every = max(10, steps // 8)

    print(f"pilot: {a.domain}  seq {a.seq_len}  batch {a.batch}x{a.accum}  "
          f"{blocks} blocks -> {a.epochs} epoch(s) = {steps} steps "
          f"({steps * per_step * a.seq_len / 1e6:.1f}M tokens), eval every "
          f"{a.eval_every}  -> {os.path.relpath(out_dir, ROOT)}", flush=True)

    t0 = time.time()
    rc, log_path = train(a, steps, a.max_hours, out_dir)
    elapsed = time.time() - t0

    hist_path = os.path.join(out_dir, f"{a.domain}.hist.json")
    if rc != 0 or not os.path.exists(hist_path):
        print(f"\ntraining exited {rc}; see {os.path.relpath(log_path, ROOT)}", flush=True)
        sys.exit(1)
    hist = json.load(open(hist_path))

    # Baselines from the packed dev split this run actually evaluated on.
    import numpy as np
    from tokenizers import Tokenizer
    dev = np.load(os.path.join(CACHE, f"{a.domain}.dev.{a.seq_len}.npy"), mmap_mode="r")
    vocab = Tokenizer.from_file(os.path.join(ROOT, "tokenizer.json")).get_vocab_size()
    base, uni = baselines(dev[:200], vocab)

    cfg = run_config(out_dir, a.domain)
    checks, best, call = verdict(hist, uni, base, cfg, smoke=a.smoke)

    lines = []
    w = lines.append
    w("# CinnamonLM pilot report\n")
    w(f"- run: `{name}`  ({a.epochs} epoch(s), {steps} steps, "
      f"{elapsed/3600:.2f} h wall clock)")
    w(f"- corpus: `{a.domain}`, {blocks} train blocks, "
      f"seq {a.seq_len}, batch {a.batch} x accum {a.accum}, lr {a.lr}")
    w(f"- vocab {vocab:,}\n")
    w("## Baselines, computed from this run's own dev split\n")
    w(f"| uniform `log(vocab)` | {base:.4f} |")
    w("|---|---|")
    w(f"| **unigram entropy** | **{uni:.4f}** |")
    w(f"| best dev loss | **{best:.4f}** (ppl {math.exp(min(20, best)):.1f}) |" if best
      else "| best dev loss | n/a |")
    w("\nThe unigram row is the one that matters. A model sitting on it has learned")
    w("token frequencies and nothing about context -- which is precisely where this")
    w("architecture was stuck before the pre-norm fix, and it is not visible in a")
    w("loss curve that still looks like it is going down.\n")
    w("## Checks\n")
    w("| check | result | evidence |")
    w("|---|---|---|")
    for n, ok, why in checks:
        w(f"| {n} | {'PASS' if ok else 'FAIL'} | {why} |")
    w(f"\n## Verdict: **{call}**\n")
    if call == "GO":
        w("The architecture learns language, not just token frequencies. Scaling is")
        w("justified on this evidence. Note what this does NOT show: WikiText-2 is")
        w("~2.4 M tokens against 44.63 M non-embedding parameters, so a good number")
        w("here is partly memorisation. It is a floor on capability, not a forecast.")
    else:
        w("Not yet worth scaling. Read the failing checks above before changing")
        w("anything -- and prefer measuring the next hypothesis over acting on it, ")
        w("since two plausible ones were already refuted by measurement in this repo.")
    w(f"\nArtifacts: `{os.path.relpath(out_dir, ROOT)}` "
      f"(checkpoints, `train.log`, `{DOMAIN}.hist.json`)\n")

    report = os.path.join(RESULTS, f"{name}.md")
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    json.dump({"name": name, "epochs": a.epochs, "steps": steps,
               "hours": elapsed / 3600, "uniform": base, "unigram": uni,
               "best_dev": best, "verdict": call,
               "checks": [{"check": n, "pass": ok, "evidence": w_} for n, ok, w_ in checks]},
              open(os.path.join(RESULTS, f"{name}.json"), "w"), indent=2)

    print("\n" + "\n".join(lines), flush=True)
    print(f"report -> {os.path.relpath(report, ROOT)}", flush=True)
    sys.exit(0 if call in ("GO", "SMOKE") else 2)


if __name__ == "__main__":
    main()
