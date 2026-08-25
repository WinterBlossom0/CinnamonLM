"""Shared scaffolding for gated linear unit MLPs (SwiGLU, PowLU, ...): two
independent linear projections of the same input, x1 * gate_fn(x2), then
projected back down. ACT2FN only ever sees one projection's output, so a GLU
variant can't be a plain activation-function swap -- it replaces GPTNeoMLP
wholesale instead. c_fc/c_proj/dropout naming mirrors
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP (verified against the
installed transformers==5.14.1 source) so this drops into a GPTNeoBlock in
its place.

x1 (the value path, no nonlinearity) and x2 (the gate path, through gate_fn)
each get their own c_fc-sized projection, so a GLU MLP has roughly double
the up-projection parameters of a plain single-projection MLP at the same
intermediate_size -- this is not param-matched against gelu_new/xielu/
polyrelu as-is. (Shazeer 2020 compensates by shrinking intermediate_size to
8/3*hidden_size instead of 4*hidden_size; not applied here -- left as-is so
the choice is explicit rather than silently baked in.)
"""
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
    """d_ff for a 3-matrix GatedMLP (2 up-projections + 1 down-projection)
    whose per-layer param count is closest to a plain 2-matrix MLP's at
    intermediate_size=4*hidden_size (Shazeer 2020's ~(8/3)*hidden_size
    convention, also used by Llama/PaLM -- but searched exactly here rather
    than taking the naive round(8/3*hidden_size), which is off by one
    integer step: round(42.667)=43 lands further from the target than 42
    does once the +bias terms are accounted for)."""
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
    """Replace every GPTNeoBlock's MLP with a GatedMLP(gate_fn), in place.
    Defaults to the param-matched size, not the plain MLP's 4*hidden_size --
    a GLU's extra projection matrix would otherwise inflate total params
    ~19% versus gelu_new/xielu/polyrelu for no reason tied to the gate
    function itself."""
    if intermediate_size is None:
        intermediate_size = param_matched_intermediate_size(model.config)
    for block in model.transformer.h:
        block.mlp = GatedMLP(intermediate_size, model.config, gate_fn)
