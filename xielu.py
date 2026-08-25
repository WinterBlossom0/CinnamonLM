"""xIELU (Huang & Schlag, "Deriving Activation Functions Using Integration",
arXiv:2411.13010, eq. 9):

    xIELU(x) = alpha_p*x^2 + 0.5x                     if x > 0
               alpha_n*(e^x - 1) - alpha_n*x + 0.5x    if x <= 0

alpha_p, alpha_n are trainable scalars, learned INDEPENDENTLY PER LAYER
(paper sec 3.1). Registering the class (not an instance) into ACT2FN gives
each GPTNeoMLP its own instance -- transformers.activations.ClassInstantier
calls cls(**kwargs) fresh on every __getitem__, confirmed by reading its
source directly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

BETA_N = 0.5   # paper sec 3.2: beta_p = beta_n = 0.5, fixed (not trainable)


def _softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


class XIELU(nn.Module):
    def __init__(self, alpha_p_init: float = 0.8, alpha_n_init: float = 0.8):
        super().__init__()
        # Constraints (sec 3.3): alpha_p > 0 via softplus; alpha_n > BETA_N
        # via BETA_N + softplus. Init both to 0.8, matching the paper's runs.
        self.raw_alpha_p = nn.Parameter(torch.tensor(_softplus_inverse(alpha_p_init)))
        self.raw_alpha_n = nn.Parameter(torch.tensor(_softplus_inverse(alpha_n_init - BETA_N)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.raw_alpha_p)
        alpha_n = BETA_N + F.softplus(self.raw_alpha_n)
        # Mask each branch's input to its own domain before applying it, not
        # just the output -- torch.where evaluates both branches, and an
        # unmasked expm1(x) on the positive side can overflow for large x,
        # poisoning the backward pass with 0*inf=nan even though that branch
        # is never selected.
        x_pos = torch.where(x > 0, x, torch.zeros_like(x))
        x_neg = torch.where(x > 0, torch.zeros_like(x), x)
        pos = alpha_p * x_pos.square() + 0.5 * x_pos
        neg = alpha_n * torch.expm1(x_neg) - alpha_n * x_neg + 0.5 * x_neg
        return torch.where(x > 0, pos, neg)


ACT2FN["xielu"] = XIELU
