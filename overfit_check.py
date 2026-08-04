"""Does the shipped architecture actually learn, at shipped scale?

test_learning.py proves this at SMALL scale (d_model=128) for CI speed. This
runs the same fixed-batch overfit at the real default Config -- the per-token
content-aware hypernetwork generation (ctx_kda -> forward_token -> compose_dora,
gathered per hypernet, checkpointed per recurrence) has never been exercised at
this scale. If loss collapses towards 0 same as SMALL does, the mechanism
generalises. If it plateaus, scale exposed something SMALL couldn't.

Run: python overfit_check.py
"""
import contextlib
import math
import os
import time

# 13, not 15: set_per_process_memory_fraction bounds PyTorch's caching allocator
# only.  The CUDA context and cuBLAS workspaces sit outside it and cost ~0.5GB,
# and a peak-reserve of 14.95 measured against a 15 cap left no margin at all.
VRAM_CAP_GB = 13.0

# Must be set before torch initialises CUDA.  expandable_segments stops the
# allocator from fragmenting into unusable holes, which is what otherwise forces
# it to reserve well above the live footprint to satisfy a large request.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from cinnamon.config import Config  # noqa: E402
from cinnamon.model import CinnamonModel  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    # The cap that actually binds.  PyTorch refuses the allocation itself rather
    # than calling cudaMalloc, so the Windows driver's system-memory fallback
    # never gets the chance to place a tensor in shared RAM -- that fallback is
    # what silently turns a 47 tok/s run into a 5 tok/s one with no error.
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB / total_gb))
    # TF32 for the fp32 matmuls, bf16 autocast for the forward.  Neither is a
    # shortcut around accuracy: bf16 keeps fp32's exponent range (no GradScaler
    # needed) and is what train.py's resolve_amp already picks on Ampere+, and
    # master weights stay fp32 either way.
    torch.set_float32_matmul_precision("high")
    print(f"vram cap {VRAM_CAP_GB:.0f}GB of {total_gb:.1f}GB dedicated", flush=True)

import sys  # noqa: E402

# Diagnostic switch (config.py no_halt): forces every token to c_max2, removing
# the discrete halting decision.  The plateau run showed depth collapsing to the
# earliest legal value with gnorm at 0.06, so the suspicion is that halting --
# not the generated weights -- is what kills the gradient.
no_halt = "--no-halt" in sys.argv
cfg = Config(no_halt=no_halt)  # else shipped defaults: d_model=256, n_hypernets=8, c_max2=32
m = CinnamonModel(cfg).to(dev)
rows, total, _ = m.param_report()
for k, v in rows:
    print(f"  {k:32s} {v/1e6:8.2f} M", flush=True)
print(f"  {'total':32s} {total/1e6:8.2f} M  device={dev}", flush=True)

def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


# Warmup is not a nicety here.  AdamW moves every weight by ~lr per step no matter
# how large the gradient is, and _init uses std=0.02 -- so at lr=3e-3 the entire
# initialisation is overwritten in 7 steps.  Measured: collapse reached 0.9997 by
# step 10, i.e. the model was destroyed before it could learn anything, and
# "predict the marginal" is the nearest stable attractor to fall into.
bs, seq = 1, 512
steps = arg("--steps", 150, int)
lr = arg("--lr", 3e-4)
warmup = arg("--warmup", 20, int)
ids = torch.randint(0, cfg.vocab, (bs, seq + 1), device=dev)
x, y = ids[:, :-1], ids[:, 1:]
# Same split as train.py: phi is the learned step embedding and must not decay.
nodec = lambda n, q: q.dim() < 2 or n.endswith("phi")
opt = torch.optim.AdamW(
    [{"params": [q for n, q in m.named_parameters() if not nodec(n, q)], "weight_decay": 0.1},
     {"params": [q for n, q in m.named_parameters() if nodec(n, q)], "weight_decay": 0.0}],
    lr=lr, betas=(0.9, 0.95))
amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()

base = math.log(cfg.vocab)
# The floor that actually applies: the batch is seq distinct random ids, so the
# best a context-blind model can do is the marginal over the ids present, not
# log(vocab).  The plateau run sat at 6.245 against log(512)=6.238.
floor = math.log(len(set(y.flatten().tolist())))
print(f"uniform baseline {base:.3f}   unigram floor {floor:.3f}   "
      f"(bs={bs} seq={seq} steps={steps} lr={lr} warmup={warmup} no_halt={no_halt})",
      flush=True)
def argmax_agreement(logits):
    """Fraction of positions whose top-1 prediction is the single most common one.

    The unambiguous version of `collapse`: cosine similarity measures direction,
    so 1.0 argues the predictions are the same.  This just counts them.  1.0 here
    means every position literally predicts the same next token.
    """
    top = logits[0].argmax(-1)
    return float(top.bincount().max()) / top.numel()


def stagewise_collapse(m, x):
    """Where does the sequence stop being a sequence?  Same cosine measure, but on
    the hidden state after each block, to localise which one flattens it."""
    out = {}
    with torch.no_grad():
        h = m.embed(x)
        out["embed"] = collapse_of(h)
        h = m.b1(h); out["b1"] = collapse_of(h)
        h, _ = m.e2(h); out["e2"] = collapse_of(h)
        h = m.bmid(h); out["bmid"] = collapse_of(h)
        h, _ = m.e3(h); out["e3"] = collapse_of(h)
        h = m.b4(h); out["b4"] = collapse_of(h)
    return out


def collapse_of(h):
    v = torch.nn.functional.normalize(h[0].float(), dim=-1)
    sim = v @ v.t()
    n = sim.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


def collapse(logits):
    """How position-independent has the model become?

    Mean pairwise cosine between per-position logit vectors, AFTER subtracting
    the mean logit.  The centring is the whole point: raw cosine reads ~0.99 for
    a perfectly healthy model, because every position shares a large common
    component (the unigram bias) that swamps the position-specific part.  During
    the successful run raw cosine sat at 0.995 while loss fell from 6.7 to 0.02
    and every position predicted a different token -- it was measuring the shared
    bias, not degeneracy, and it briefly had me calling a healthy run collapsed.

    Centred, this reports only whether the position-SPECIFIC parts are distinct,
    which is the actual question.  Cross-check with same-token, which needs no
    interpretation at all.
    """
    v = logits[0].float()
    v = v - v.mean(0, keepdim=True)
    v = torch.nn.functional.normalize(v, dim=-1)
    sim = v @ v.t()
    n = sim.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


t0 = time.time()
for s in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr * min(1.0, (s + 1) / max(1, warmup))
    opt.zero_grad(set_to_none=True)
    with amp:
        logits, loss, (d2, d3) = m(x, labels=y)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if dev == "cuda":
        # Reserved, not allocated: reserved is what the process actually holds
        # against the card, so this is the number the cap is about.
        held = torch.cuda.memory_reserved() / 1e9
        assert held <= VRAM_CAP_GB, f"step {s}: {held:.2f}GB reserved > {VRAM_CAP_GB}GB cap"
    if s % 5 == 0 or s == steps - 1:
        mem = torch.cuda.max_memory_reserved() / 1e9 if dev == "cuda" else 0.0
        el = time.time() - t0
        print(f"step {s:4d}  loss {float(loss):7.4f}  gnorm {float(gn):7.2f}  "
              f"collapse {collapse(logits):+.4f}  same-token {argmax_agreement(logits):.3f}  "
              f"depth {float(d2.float().mean()):.1f}[{int(d2.min())}-{int(d2.max())}]/"
              f"{float(d3.float().mean()):.1f}[{int(d3.min())}-{int(d3.max())}]  "
              f"gpu {mem:.2f}GB  {(s+1)*bs*seq/el:6.1f} tok/s  "
              f"{el/(s+1):5.2f}s/step  {el:.0f}s", flush=True)

print(f"\nfinal loss {float(loss):.4f}  unigram floor {floor:.3f}  "
      f"{'BEATS' if float(loss) < floor - 0.3 else 'STUCK AT'} the floor", flush=True)
print(f"positions sharing one top-1 token: {argmax_agreement(logits):.3f}", flush=True)
print("stagewise collapse (0 = positions distinct, 1 = all identical):", flush=True)
for k, v in stagewise_collapse(m, x).items():
    print(f"  after {k:6s} {v:+.4f}", flush=True)
