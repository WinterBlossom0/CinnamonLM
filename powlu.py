"""PowLU (Ant Group Ling Team, arXiv:2605.25704): x1 * f(x2), f(x2) =
x2^(m/(sqrt(x2)+1)) * sigmoid(x2) for x2>0 else SiLU(x2). m=3.0, fixed."""
from __future__ import annotations

import torch

from glu_mlp import apply_gated_mlp

M = 3.0


def powlu_gate(x2: torch.Tensor, m: float = M) -> torch.Tensor:
    sig = torch.sigmoid(x2)
    # masked to pos domain: unmasked sqrt/pow on negative x2 -> nan grad
    x2_safe = torch.where(x2 > 0, x2, torch.ones_like(x2))
    exponent = m / (torch.sqrt(x2_safe) + 1.0)
    pos = x2_safe**exponent * sig
    neg = x2 * sig
    return torch.where(x2 > 0, pos, neg)


def apply_powlu(model, m: float = M, intermediate_size: int | None = None) -> None:
    apply_gated_mlp(model, lambda x2: powlu_gate(x2, m), intermediate_size)
