"""PowLU (Jiang et al., Ant Group Ling Team, "PowLU: An Activation Function
for Stable Pre-Training of LLMs," arXiv:2605.25704, 25 May 2026, eq. 1):

    PowLU(x1, x2) = x1 * f(x2)
    f(x2) = x2 ** (m / (sqrt(x2) + 1)) * sigmoid(x2)   if x2 > 0
    f(x2) = x2 * sigmoid(x2)  (= SiLU(x2))              if x2 <= 0

m is a FIXED hyperparameter, not trainable -- the paper's own scaling-law
and large-scale (7.9B/124B) runs all use m=3.0 (sec 4.1.1: "we choose the
value 3.0 in our experiments"). x1, x2 are two independent linear
projections of the same MLP input (sec 3.1, "implemented as PowLU(x1,x2) =
x1 * f(x2)"), same gated structure as SwiGLU -- see glu_mlp.py for why this
needs a full GPTNeoMLP replacement, not an ACT2FN entry. Designed as a
drop-in stability fix for SwiGLU: identical for x2<=0, and for x2>0 the
exponent 1+m/(sqrt(x2)+1) shrinks from ~1+m toward 1 as x2 grows, so growth
degenerates toward linear instead of SwiGLU's unbounded ~x^2 (sec 3.2).
"""
from __future__ import annotations

import torch

from glu_mlp import apply_gated_mlp

M = 3.0


def powlu_gate(x2: torch.Tensor, m: float = M) -> torch.Tensor:
    sig = torch.sigmoid(x2)
    # Mask the base into the pow branch's own domain before computing it,
    # not just the output -- torch.where evaluates both branches, and
    # sqrt/pow on an unmasked negative x2 would poison the backward pass
    # with nan even though that branch is never selected (same pattern as
    # xielu.py).
    x2_safe = torch.where(x2 > 0, x2, torch.ones_like(x2))
    exponent = m / (torch.sqrt(x2_safe) + 1.0)
    pos = x2_safe**exponent * sig
    neg = x2 * sig  # SiLU, paper's x2<=0 branch
    return torch.where(x2 > 0, pos, neg)


def apply_powlu(model, m: float = M, intermediate_size: int | None = None) -> None:
    apply_gated_mlp(model, lambda x2: powlu_gate(x2, m), intermediate_size)
