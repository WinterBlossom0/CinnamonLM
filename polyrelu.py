"""PolyReLU (Zhuo et al., ICLR 2025, arXiv:2411.03884): PolyReLU(x) = sum_i
a_i * ReLU(x)^i, per-layer trainable coefficients."""
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
