"""SwiGLU (Shazeer, arXiv:2002.05202): x1 * SiLU(x2), gated MLP baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from glu_mlp import apply_gated_mlp


def swiglu_gate(x2: torch.Tensor) -> torch.Tensor:
    return F.silu(x2)


def apply_swiglu(model, intermediate_size: int | None = None) -> None:
    apply_gated_mlp(model, swiglu_gate, intermediate_size)
