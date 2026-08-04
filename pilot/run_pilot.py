"""Pilot: is this architecture worth scaling?

One self-contained experiment on WikiText-2, whose whole job is to answer a
go/no-go question -- does CinnamonLM learn language well enough to justify
spending real compute on it -- and to answer it with numbers rather than
impressions.

EVERYTHING it produces lands under pilot/.  The packed token cache, the
checkpoints, the logs, the report.  Nothing is written outside this directory and
nothing outside it is modified, so a pilot run can be deleted with `rm -rf pilot/`
and leave the repository exactly as it was.

    python pilot/run_pilot.py                 # ~2 h budget, the default
    python pilot/run_pilot.py --hours 0.5     # shorter
    python pilot/run_pilot.py --smoke         # minutes, wiring check only

It shells out to the real train.py rather than reimplementing the loop.  That is
deliberate: a previous helper in this repo reimplemented training and carried its
own bug for it.  The pilot must measure the thing that will actually be scaled,
not a copy of it that has drifted.
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
DOMAIN = "wikitext2"

sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
# Baselines.  A loss number means nothing on its own -- "4.9" is either good or
# catastrophic depending on what the data could have been predicted at without a
# model at all.  Both of these are computed from the ACTUAL packed dev split.
# --------------------------------------------------------------------------- #

def baselines(dev_blocks, vocab):
    """(uniform, unigram) cross-entropy in nats.

    uniform  -- log(vocab).  The score of a model that has learned nothing.
    unigram  -- entropy of the dev token frequency distribution.  The score of a
                model that has learned which tokens are common and nothing else.
                This is the one that matters: a model sitting on it is not using
                context, however plausible its loss looks.  It is exactly where
                this architecture was stuck for a long time.
    """
    import numpy as np
    counts = np.bincount(np.asarray(dev_blocks).reshape(-1), minlength=vocab)
    p = counts[counts > 0] / counts.sum()
    return math.log(vocab), float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- #

def train(args, steps, hours, out_dir):
    """Run the real train.py, contained entirely under pilot/."""
    cmd = [sys.executable, os.path.join(ROOT, "train.py"),
           "--domain", DOMAIN,
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


def verdict(hist, uni, base, cfg_vocab, smoke=False):
    """Turn the run into a go/no-go, with the reason attached to each check.

    Deliberately strict about the one that matters.  Beating the unigram entropy
    is not a nice-to-have: below it the model is using context, on it the model is
    a frequency table with extra steps, and that distinction is invisible in a
    loss curve that looks like it is descending.
    """
    devs = [h for h in hist if "dev_loss" in h]
    trains = [h for h in hist if "loss" in h]
    if not devs:
        return [], None, "no eval ran -- the budget was too short to reach --eval-every"

    best = min(d["dev_loss"] for d in devs)
    last = devs[-1]
    checks = []

    checks.append(("beats uniform", best < base - 0.1,
                   f"dev {best:.4f} vs log(vocab) {base:.4f}"))
    checks.append(("BEATS UNIGRAM (uses context)", best < uni - 0.05,
                   f"dev {best:.4f} vs unigram entropy {uni:.4f}"))

    if len(devs) >= 2:
        # Compare the final eval against the best of the FIRST half.  Indexing the
        # midpoint directly compared devs[-1] to itself whenever only two evals
        # existed, so this read FAIL on every short run.
        earlier = devs[:max(1, len(devs) // 2)]
        ref = min(d["dev_loss"] for d in earlier)
        checks.append(("still improving at the end", devs[-1]["dev_loss"] < ref,
                       f"first half best {ref:.4f} -> final {devs[-1]['dev_loss']:.4f}"
                       " (FAIL means converged or diverged, not that it cannot learn)"))

    d2, d3 = last.get("dev_depth", (0, 0))
    checks.append(("depth stays off the cap", 0 < max(d2, d3) < 0.9 * 32,
                   f"mean depth {d2:.1f}/{d3:.1f} of cap 32"))

    gc = last.get("gcum")
    if gc is not None:
        # Not a numerics check any more: the chunked path is exact at any gcum
        # since the subtract-before-exp fix.  This asks whether KDA is still
        # carrying context -- retention/token is exp(-gcum/chunk), so 44 at
        # chunk 64 is 0.5, i.e. half the state dropped every token.
        checks.append(("KDA still acting as memory", gc < 44.0,
                       f"max |gcum| {gc:.1f} -> retention/token "
                       f"{math.exp(-gc/64):.3f}"))

    routed = [t for t in trains if t.get("routing")]
    if routed:
        r = routed[-1]["routing"]
        frac = r.get("frac") or []
        if frac:
            checks.append(("router did not collapse", min(frac) > 0.02,
                           f"per-hypernet share {frac}"))
        checks.append(("load balance near floor", r.get("aux", 9) < 1.5,
                       f"aux {r.get('aux', float('nan')):.3f}, floor is 1.0"))

    finite = all(math.isfinite(t["loss"]) for t in trains)
    checks.append(("no non-finite loss", finite, f"{len(trains)} logged steps"))

    gate = dict((n, ok) for n, ok, _ in checks)
    go = gate.get("BEATS UNIGRAM (uses context)", False) and gate.get("no non-finite loss", False)
    if smoke:
        # A smoke run proves the wiring and nothing else.  Returning NO-GO here
        # would read as evidence against the architecture when it is only evidence
        # that 60 steps is 60 steps.
        return checks, best, "SMOKE"
    return checks, best, ("GO" if go else "NO-GO")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=2.0,
                   help="wall-clock budget; train.py re-fits the schedule to it")
    p.add_argument("--steps", type=int, default=20000, help="ceiling if --hours is not hit")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--name", default=None, help="run name; defaults to a timestamp")
    p.add_argument("--smoke", action="store_true",
                   help="minutes, not hours: proves the wiring, proves nothing else")
    a = p.parse_args()

    if a.smoke:
        a.hours, a.steps, a.warmup = 0, 60, 10
        a.eval_every, a.log_every = 20, 5

    name = a.name or time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if a.smoke else "")
    out_dir = os.path.join(RUNS, name)
    for d in (CACHE, out_dir, RESULTS):
        os.makedirs(d, exist_ok=True)

    print(f"pilot: {DOMAIN}  seq {a.seq_len}  batch {a.batch}x{a.accum}  "
          f"budget {a.hours}h  -> {os.path.relpath(out_dir, ROOT)}", flush=True)

    t0 = time.time()
    rc, log_path = train(a, a.steps, a.hours, out_dir)
    elapsed = time.time() - t0

    hist_path = os.path.join(out_dir, f"{DOMAIN}.hist.json")
    if rc != 0 or not os.path.exists(hist_path):
        print(f"\ntraining exited {rc}; see {os.path.relpath(log_path, ROOT)}", flush=True)
        sys.exit(1)
    hist = json.load(open(hist_path))

    # Baselines from the packed dev split this run actually evaluated on.
    import numpy as np
    from tokenizers import Tokenizer
    dev = np.load(os.path.join(CACHE, f"{DOMAIN}.dev.{a.seq_len}.npy"), mmap_mode="r")
    vocab = Tokenizer.from_file(os.path.join(ROOT, "tokenizer.json")).get_vocab_size()
    base, uni = baselines(dev[:200], vocab)

    checks, best, call = verdict(hist, uni, base, vocab, smoke=a.smoke)

    lines = []
    w = lines.append
    w("# CinnamonLM pilot report\n")
    w(f"- run: `{name}`  ({elapsed/3600:.2f} h wall clock)")
    w(f"- corpus: `{DOMAIN}` (WikiText-2 raw), seq {a.seq_len}, "
      f"batch {a.batch} x accum {a.accum}, lr {a.lr}")
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
    json.dump({"name": name, "hours": elapsed / 3600, "uniform": base, "unigram": uni,
               "best_dev": best, "verdict": call,
               "checks": [{"check": n, "pass": ok, "evidence": w_} for n, ok, w_ in checks]},
              open(os.path.join(RESULTS, f"{name}.json"), "w"), indent=2)

    print("\n" + "\n".join(lines), flush=True)
    print(f"report -> {os.path.relpath(report, ROOT)}", flush=True)
    sys.exit(0 if call in ("GO", "SMOKE") else 2)


if __name__ == "__main__":
    main()
