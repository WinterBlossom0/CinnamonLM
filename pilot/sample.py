"""Generate from a trained pilot checkpoint.  Standalone -- run_pilot.py does not
call this, and the verdict does not depend on it.  Run it by hand after a run.

    python pilot/sample.py                      # latest run, best.pt
    python pilot/sample.py --run 20260804-192902 --tokens 60
    python pilot/sample.py --temperature 0      # greedy

A 110 M model trained for one epoch on 2.7 M tokens of WikiText-2 is not an
instruction-following model and will not answer questions.  Expect Wikipedia-ish
fragments.  The point is to see WHAT it produces -- whether it emits well-formed
words, whether it tracks syntax, whether it repeats -- which a loss number does
not tell you.

Also reports mean recurrence depth per prompt, which is the one quantitative
thing here: the architecture claims per-token adaptive compute, so depth should
differ between prompts if that claim means anything.

Writes pilot/results/<run>.samples.md.  Nothing outside pilot/ is touched.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)

MAX_TOKENS = 256

PROMPTS = [
    "Hi, how are you?",
    "Explain the Python language.",
    "What is the capital of France?",
    "The history of the Roman Empire began",
    "Water boils at a temperature of",
    "In 1969, humans first landed on",
    "The difference between a cat and a dog is",
    "To make a cup of tea you should",
    "Albert Einstein was a physicist who",
    "The three primary colours are",
]


def latest_run():
    runs = [d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))] \
        if os.path.isdir(RUNS) else []
    if not runs:
        sys.exit("no runs under pilot/runs -- train one first with run_pilot.py")
    return sorted(runs)[-1]


def load(run, domain, which):
    """Rebuild the model from the checkpoint's OWN cfg, so this keeps working
    when the architecture changes."""
    from cinnamon.config import Config
    from cinnamon.model import CinnamonModel

    path = os.path.join(RUNS, run, f"{domain}.{which}.pt")
    if not os.path.exists(path):
        sys.exit(f"no checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]) if isinstance(ck.get("cfg"), dict) else Config()
    m = CinnamonModel(cfg)
    m.load_state_dict(ck["model"])
    return m.eval(), cfg, ck


@torch.no_grad()
def generate(m, tok, prompts, n_tokens, temperature, top_k, device, eos_id):
    """All prompts in ONE batch.  The 256 decode steps are sequential either way
    (no KV cache -- the whole prefix is re-encoded every step), so batching turns
    10 near-idle forwards per step into one, roughly 10x.

    LEFT-padded with eos, because the model has no padding mask: pad tokens
    genuinely enter attention rather than being masked out.  eos is the least harmful
    filler -- it is the document separator the model saw throughout training, so a
    short eos prefix reads as "start of a new document" rather than as noise.
    Exact only for the longest prompt; the rest carry 1-6 tokens of that prefix.
    """
    enc = [tok.encode(p).ids or [eos_id] for p in prompts]
    L = max(len(e) for e in enc)
    pads = [L - len(e) for e in enc]
    ids = torch.tensor([[eos_id] * p + e for e, p in zip(enc, pads)],
                       dtype=torch.long, device=device)
    B = ids.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    d2s, d3s = torch.zeros(B), torch.zeros(B)
    steps = 0

    for _ in range(n_tokens):
        logits, _, depth = m(ids)
        d2s += d2.float().mean(-1).cpu()
        d3s += d3.float().mean(-1).cpu()
        steps += 1
        nxt = logits[:, -1].float()
        if temperature <= 0:
            tid = nxt.argmax(-1)
        else:
            nxt = nxt / temperature
            if top_k:
                v, _ = torch.topk(nxt, min(top_k, nxt.shape[-1]), dim=-1)
                nxt[nxt < v[:, -1:]] = -float("inf")
            tid = torch.multinomial(torch.softmax(nxt, -1), 1).squeeze(-1)
        tid = torch.where(done, torch.full_like(tid, eos_id), tid)
        done = done | (tid == eos_id)
        ids = torch.cat([ids, tid.unsqueeze(-1)], -1)
        if bool(done.all()):
            break

    out = []
    for i, p in enumerate(pads):
        row = ids[i, p:].tolist()                 # drop this row's left padding
        if eos_id in row[len(enc[i]):]:           # cut at the first generated eos
            row = row[:len(enc[i]) + row[len(enc[i]):].index(eos_id)]
        out.append((tok.decode(row),
                    (float(d2s[i]) / steps, float(d3s[i]) / steps)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=None,
                   help="single prompt, streamed token by token; skips the 10-prompt set")
    p.add_argument("--run", default=None, help="run name; default = latest")
    p.add_argument("--domain", default="wikitext2")
    p.add_argument("--which", default="best", choices=["best", "last"])
    # 256 is a hard ceiling, not just a default.  Generation has no KV cache, so
    # the whole prefix is re-encoded every token and cost grows quadratically --
    # and every one of those forwards runs the full recurrence.
    p.add_argument("--tokens", type=int, default=MAX_TOKENS,
                   help=f"tokens per answer, capped at {MAX_TOKENS}")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--tokenizer", default=os.path.join(ROOT, "tokenizer.json"))
    a = p.parse_args()
    a.tokens = max(1, min(a.tokens, MAX_TOKENS))

    from tokenizers import Tokenizer
    run = a.run or latest_run()
    m, cfg, ck = load(run, a.domain, a.which)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev)
    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<eos>") or 0

    head = (f"run `{run}` / {a.which}.pt  (step {ck.get('step')}, "
            f"dev {ck.get('dev_loss', float('nan')):.4f})  "
            f"temp {a.temperature}, top_k {a.top_k}, {a.tokens} tokens")
    lines = ["# CinnamonLM pilot samples\n", head, ""]
    lines += ["Qualitative only, not part of the go/no-go. A 110 M model trained one",
              "epoch on 2.7 M tokens is not instruction-tuned and will not answer",
              "questions; the useful signal is whether output is well-formed, and",
              "whether recurrence depth differs between prompts.\n"]
    print(head + "\n", flush=True)

    if a.prompt:
        # Single prompt, streamed.  Decodes the whole id list each step and prints
        # only the delta -- BPE pieces do not map 1:1 to characters, so decoding
        # tokens individually mangles the text.
        ids = tok.encode(a.prompt).ids or [eos]
        print(f"> {a.prompt}\n  ", end="", flush=True)
        shown, d2s, d3s = len(tok.decode(ids)), 0.0, 0.0
        with torch.no_grad():
            for s in range(a.tokens):
                x = torch.tensor([ids], dtype=torch.long, device=dev)
                logits, _, depth = m(x)
                d2s += float(d2.float().mean()); d3s += float(d3.float().mean())
                nx = logits[0, -1].float()
                if a.temperature <= 0:
                    tid = int(nx.argmax())
                else:
                    nx = nx / a.temperature
                    if a.top_k:
                        v, _ = torch.topk(nx, min(a.top_k, nx.numel()))
                        nx[nx < v[-1]] = -float("inf")
                    tid = int(torch.multinomial(torch.softmax(nx, -1), 1))
                if tid == eos:
                    print(" <eos>", end="", flush=True)
                    break
                ids.append(tid)
                text = tok.decode(ids)
                print(text[shown:], end="", flush=True)
                shown = len(text)
        n = max(1, s + 1)
        print(f"\n  [depth {d2s/n:.1f}/{d3s/n:.1f} over {n} tokens]", flush=True)
        return

    results = generate(m, tok, PROMPTS, a.tokens, a.temperature, a.top_k, dev, eos)
    for q, (text, (d2, d3)) in zip(PROMPTS, results):
        cont = text[len(q):].strip() if text.startswith(q) else text
        print(f"> {q}\n  {cont}\n  [depth {d2:.1f}/{d3:.1f}]\n", flush=True)
        lines += [f"**{q}**", "", f"> {cont}", "", f"`depth {d2:.1f}/{d3:.1f}`", ""]

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{run}.samples.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"-> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
