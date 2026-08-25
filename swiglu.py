"""SwiGLU (Shazeer, "GLU Variants Improve Transformer," arXiv:2002.05202,
2020): the classic gated baseline every newer GLU variant (PowLU included)
benchmarks itself against.

    SwiGLU(x1, x2) = x1 * SiLU(x2) = x1 * (x2 * sigmoid(x2))

x1, x2 are two independent linear projections of the same MLP input -- see
glu_mlp.py for why this needs a full GPTNeoMLP replacement, not an ACT2FN
entry.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from glu_mlp import apply_gated_mlp


def swiglu_gate(x2: torch.Tensor) -> torch.Tensor:
    return F.silu(x2)


def apply_swiglu(model, intermediate_size: int | None = None) -> None:
    apply_gated_mlp(model, swiglu_gate, intermediate_size)
