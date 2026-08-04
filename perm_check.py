"""Does the model see word order at all?

The overfit test lands on log(n_distinct) every time, across bucketed and
per-token generation, KDA-summary and hidden-state halting, shared and per-step
norms.  That number is the entropy of the token MARGINAL -- exactly the score of
a model that ignores position and emits "which tokens are present".

Memorising a fixed random batch requires position: "at index 137 emit 40213".
Under NoPE, MLA carries no positional information at all, so KDA's recurrent
state is the only thing in the model that knows about order.  If that pathway is
dead, the plateau is not a training failure, it is the information-theoretic
ceiling and no hypernetwork change can ever move it.

Two checks, no training needed:

  1. PERMUTATION.  Shuffle the input tokens.  If the output logits merely follow
     the shuffle, the model is permutation-equivariant -- it has no positional
     signal whatsoever.
  2. PREFIX.  Change ONLY token 0 and measure how much later positions move.  A
     working sequential model propagates that change forward; a dead one does not.

Run: python perm_check.py
"""
import torch

from cinnamon.config import Config
from cinnamon.model import CinnamonModel
from cinnamon.kda import KDA

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Config()
m = CinnamonModel(cfg).to(dev).eval()

S = 64
ids = torch.randint(0, cfg.vocab, (1, S), device=dev)

with torch.no_grad():
    base, _, _ = m(ids)

    # ---- 1. permutation ------------------------------------------------- #
    perm = torch.randperm(S, device=dev)
    permuted, _, _ = m(ids[:, perm])
    # if the model ignores order, permuting the input just permutes the output
    equivariant = torch.allclose(permuted[0], base[0][perm], atol=1e-3)
    gap = (permuted[0] - base[0][perm]).abs().max()
    print(f"permutation-equivariant: {equivariant}   max|diff| {float(gap):.4e}")
    print("  -> " + ("NO positional signal: the model cannot tell position apart"
                     if equivariant else "order changes the output, position survives"))

    # ---- 2. prefix propagation ------------------------------------------ #
    alt = ids.clone()
    alt[0, 0] = (ids[0, 0] + 1) % cfg.vocab        # change ONLY token 0
    moved, _, _ = m(alt)
    d = (moved[0] - base[0]).abs().mean(-1)        # per-position change
    print(f"\nchange token 0 only -> mean |dlogit| by position:")
    for p in (0, 1, 2, 8, 16, 32, S - 1):
        print(f"  pos {p:3d}: {float(d[p]):.4e}")
    tail = float(d[8:].mean())
    print("  -> " + ("prefix does NOT reach later positions -- sequential path dead"
                     if tail < 1e-6 else f"prefix propagates forward (tail mean {tail:.2e})"))

    # ---- 3. is KDA's gate saturated? ------------------------------------ #
    # architecture.md 11 documents an instability past retention ~0.86, and the
    # earlier gradient-norm blowup traced entirely to expert.kda.
    gc = [(n, float(mod._max_gcum)) for n, mod in m.named_modules()
          if isinstance(mod, KDA) and hasattr(mod, "_max_gcum")]
    if gc:
        worst = max(gc, key=lambda t: t[1])
        print(f"\nKDA max |gcum|: worst {worst[0]} = {worst[1]:.2f} "
              f"(>10 means the chunked path has stopped matching the exact recurrence)")
