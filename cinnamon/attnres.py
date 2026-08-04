"""3.2 AttnRes - attention residuals (Full variant).

Standard PreNorm unrolls to an equal-weight sum of every prior sublayer output,
so hidden-state magnitude grows as O(L) and early layers get diluted.  AttnRes
replaces those equal weights with softmax attention over depth, using a learned
input-independent pseudo-query per sublayer.

Settings are the paper's ablated optima and are not free choices: single head
(multi-head hurts), softmax kernel (sigmoid hurts), RMSNorm on keys (without it
sublayers with naturally large outputs dominate), pseudo-query init to zeros
(makes initial weights uniform and prevents training volatility).
"""
import torch
import torch.nn as nn

from .config import Config


class AttnRes(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(c.d_model))    # zero init is required
        self.norm = nn.RMSNorm(c.d_model, eps=c.eps)

    def forward(self, sources):
        """sources: list of [B, S, d] -- block input plus all prior sublayer outputs."""
        V = torch.stack(sources)                                     # [n,B,S,d]
        logits = torch.einsum('d,nbsd->nbs', self.w, self.norm(V))
        return torch.einsum('nbs,nbsd->bsd', logits.softmax(0), V)
