"""Behaviour checks: the architecture must actually optimise.

Structural tests prove the shapes and invariants hold.  These prove gradient
actually moves the loss, which is the failure mode a shape test cannot see -- a
recurrent stack with up to c_max2 checkpointed passes, a generated-weight FFN and
a data-dependent halt mask has plenty of ways to be silently untrainable.

Run: python -m tests.test_learning
"""
import math
import time

import torch

from cinnamon.config import Config
from cinnamon.model import CinnamonModel

# n_hypernets kept small: the routed model's per-token generation gathers only
# each hypernet's own tokens (routed.py), so more hypernets means more gathered
# calls per recurrence for the same token count -- this is a speed knob for the
# test, not a claim about what the shipped model should use.
SMALL = dict(d_model=128, d_ff=512, vocab=512, n_heads=4, head_dim=32,
             kda_head_dim=32,         # pinned: SMALL is the known-good reference
             kv_lora_rank=32, q_lora_rank=64, rank=8, lora_alpha=16, phi_dim=16,
             r_ceiling=8, r_free=2, turns=2, c_max2=8, n_hypernets=2, kda_chunk=32)


def _run(cfg, steps, bs=4, seq=64, lr=3e-4, seed=0, log_every=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CinnamonModel(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    ids = torch.randint(0, cfg.vocab, (bs, seq + 1), device=dev)
    x, y = ids[:, :-1], ids[:, 1:]
    losses = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss, d = m(x, labels=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if log_every and s % log_every == 0:
            print(f"    step {s:4d}  loss {losses[-1]:7.4f}  "
                  f"depth {float(d[0].float().mean()):.1f}/{float(d[1].float().mean()):.1f}",
                  flush=True)
    return m, losses, d


def test_overfits_a_fixed_batch():
    """The sharpest cheap signal: with nothing to generalise to, loss must collapse."""
    cfg = Config(**SMALL)
    t0 = time.time()
    m, losses, depth = _run(cfg, steps=300, lr=1e-3, log_every=50)
    base = math.log(cfg.vocab)
    print(f"  uniform baseline {base:.3f} -> final {losses[-1]:.4f} "
          f"({time.time()-t0:.0f}s)")
    assert losses[-1] < 0.5, f"did not overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert all(math.isfinite(l) for l in losses), "non-finite loss"


def test_depth_is_data_dependent():
    """Adaptive depth is the whole point; if every token always runs to c_max the
    halting rule is inert and the model is just a fixed deep stack."""
    torch.manual_seed(0)
    cfg = Config(**SMALL, grad_checkpoint=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CinnamonModel(cfg).to(dev).eval()
    seen = set()
    with torch.no_grad():
        for s in range(6):
            ids = torch.randint(0, cfg.vocab, (4, 64), device=dev)
            _, _, (d2, d3) = m(ids)
            seen |= set(d2.flatten().tolist()) | set(d3.flatten().tolist())
    print(f"  observed depths: {sorted(seen)}")
    assert min(seen) >= cfg.r_free, f"halted before r_free: {min(seen)}"
    assert max(seen) <= cfg.c_max2, f"exceeded c_max2: {max(seen)}"
    assert len(seen) > 1, f"depth never varies (always {seen}), halting rule is inert"


def test_grad_checkpointing_matches_plain():
    """Checkpointing must be invisible to the result, only to memory."""
    torch.manual_seed(0)
    a = Config(**SMALL, grad_checkpoint=True)
    b = Config(**SMALL, grad_checkpoint=False)
    torch.manual_seed(0)
    ma = CinnamonModel(a)
    torch.manual_seed(0)
    mb = CinnamonModel(b)
    mb.load_state_dict(ma.state_dict())
    ids = torch.randint(0, a.vocab, (2, 32))
    _, la, _ = ma(ids, labels=ids)
    _, lb, _ = mb(ids, labels=ids)
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, atol=1e-5), (float(la), float(lb))
    ga = ma.e2.expert.dyn_up.weight.grad
    gb = mb.e2.expert.dyn_up.weight.grad
    assert torch.allclose(ga, gb, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name, flush=True)
            fn()
            print("ok  ", name, flush=True)
    print("\nall passed")
