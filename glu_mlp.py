"""Gated MLP scaffold (SwiGLU, PowLU, ...): x1 * gate_fn(x2), replaces
GPTNeoMLP wholesale since ACT2FN only ever sees one projection."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class GatedMLP(nn.Module):
    def __init__(self, intermediate_size: int, config, gate_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc1 = nn.Linear(embed_dim, intermediate_size)
        self.c_fc2 = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.dropout = nn.Dropout(float(config.resid_dropout))
        self.gate_fn = gate_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1 = self.c_fc1(hidden_states)
        x2 = self.c_fc2(hidden_states)
        hidden_states = x1 * self.gate_fn(x2)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


def param_matched_intermediate_size(config) -> int:
    # exact search, not round(8/3*hidden) -- naive rounding can land further
    # from the plain-MLP param target than the true closest integer does
    hidden = config.hidden_size
    plain_d_ff = 4 * hidden
    plain_params = 2 * hidden * plain_d_ff + plain_d_ff + hidden
    best_d_ff, best_diff = 1, None
    for d_ff in range(1, plain_d_ff + 1):
        gated_params = 3 * hidden * d_ff + 2 * d_ff + hidden
        diff = abs(gated_params - plain_params)
        if best_diff is None or diff < best_diff:
            best_d_ff, best_diff = d_ff, diff
    return best_d_ff


def apply_gated_mlp(
    model,
    gate_fn: Callable[[torch.Tensor], torch.Tensor],
    intermediate_size: int | None = None,
) -> None:
    if intermediate_size is None:
        intermediate_size = param_matched_intermediate_size(model.config)
    for block in model.transformer.h:
        block.mlp = GatedMLP(intermediate_size, model.config, gate_fn)
