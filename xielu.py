"""xIELU (Huang & Schlag, arXiv:2411.13010): alpha_p*x^2+0.5x for x>0, else
alpha_n*(e^x-1-x)+0.5x. Per-layer trainable alpha_p, alpha_n."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

BETA_N = 0.5


def _softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


class XIELU(nn.Module):
    def __init__(self, alpha_p_init: float = 0.8, alpha_n_init: float = 0.8):
        super().__init__()
        self.raw_alpha_p = nn.Parameter(torch.tensor(_softplus_inverse(alpha_p_init)))
        self.raw_alpha_n = nn.Parameter(torch.tensor(_softplus_inverse(alpha_n_init - BETA_N)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.raw_alpha_p)
        alpha_n = BETA_N + F.softplus(self.raw_alpha_n)
        # masked per-branch: torch.where evaluates both, unmasked expm1 overflows -> nan grad
        x_pos = torch.where(x > 0, x, torch.zeros_like(x))
        x_neg = torch.where(x > 0, torch.zeros_like(x), x)
        pos = alpha_p * x_pos.square() + 0.5 * x_pos
        neg = alpha_n * torch.expm1(x_neg) - alpha_n * x_neg + 0.5 * x_neg
        return torch.where(x > 0, pos, neg)


ACT2FN["xielu"] = XIELU
