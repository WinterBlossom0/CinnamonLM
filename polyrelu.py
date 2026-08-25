"""PolyReLU (Zhuo et al., "Polynomial Composition Activations: Unleashing the
Dynamics of Large Language Models", ICLR 2025, arXiv:2411.03884, Type I
PolyCom / eq. 2):

    PolyReLU(x) = sum_{i=0}^{r} a_i * ReLU(x)^i

a_i are trainable scalars, learned INDEPENDENTLY PER LAYER (paper's FFN
diagram shows one a_0..a_r set per block). Registering the class (not an
instance) into ACT2FN gives each GPTNeoMLP its own instance, same mechanism
used for xielu.py.

Paper uses r=3, init a_0=0, a_1=a_2=a_3=1/r. Since ReLU(x)=0 for x<=0, every
i>=1 term vanishes there -- only the constant a_0 term survives, so the
negative side is flat at a_0 (not fixed at 0; a_0 is trainable and
unconstrained, same as every other a_i).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

ORDER = 3


class PolyReLU(nn.Module):
    def __init__(self, order: int = ORDER):
        super().__init__()
        init = torch.zeros(order + 1)
        init[1:] = 1.0 / order
        self.a = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = F.relu(x)
        out = self.a[0] + torch.zeros_like(x)
        power = torch.ones_like(x)
        for i in range(1, self.a.shape[0]):
            power = power * r
            out = out + self.a[i] * power
        return out


ACT2FN["polyrelu"] = PolyReLU
