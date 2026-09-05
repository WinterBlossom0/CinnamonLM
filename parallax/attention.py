"""Parallax: full causal softmax attention plus a local-covariance
correction (Zuo, Pai, Zeng, Dewulf, Hu, Wang, arXiv:2605.29157). Unlike
KDA/OSDN this is O(T^2), not chunked/recurrent -- no positional carrier of
its own, so the model adds a learned absolute position embedding for this
arm (see model.py).

o_i = o_i^SA - Sigma_KV^(i) @ rho_i, where o_i^SA is standard softmax
attention output, rho_i = W_R x_i (a learned projection, replacing the
original LLA paper's per-token numerical solve), and Sigma_KV^(i) is the
attention-weighted covariance between V and K at position i:

    Sigma_KV^(i) = sum_j p_ij (v_j - v_bar_i)(k_j - k_bar_i)^T

Expanding the outer product and using v_bar_i = o_i^SA gives the streaming
form actually used in forward() (paper eq. 14):

    o_i = o_i^SA * (1 + k_bar_i . rho_i) - sum_j p_ij (k_j . rho_i) v_j

CRITICAL, easy to miss from the equations alone (found only by reading the
paper's Sec. 4.3, not just its core formulas): Parallax's benefit is
optimizer-dependent. Under Muon the paper reports consistent gains; under
AdamW they state gains "shrink markedly or even disappear," and their
gating ablation (Fig. 6b) shows the model learns to suppress the covariance
correction under AdamW (gate converges to ~0.26, vs. fully open under
Muon) -- their own spectral analysis (Table 4) shows W_R's stable rank
collapses from 134 (Muon) to ~9 (AdamW). train.py trains this arm under Muon
specifically (_make_optimizers), not the AdamW used for KDA/OSDN, so a weak
result here would not be attributable to the wrong-optimizer confound the
paper itself documents.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Parallax(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        del chunk  # accepted for interface uniformity with KDA/OSDN; O(T^2), no chunking
        h, d = n_heads, head_dim
        self.h, self.d, self.eps = h, d, eps
        self.q = nn.Linear(d_model, h * d, bias=False)
        self.k = nn.Linear(d_model, h * d, bias=False)
        self.v = nn.Linear(d_model, h * d, bias=False)
        self.r = nn.Linear(d_model, h * d, bias=False)
        self.o = nn.Linear(h * d, d_model, bias=False)
        self.qn = nn.RMSNorm(d, eps=eps)
        self.kn = nn.RMSNorm(d, eps=eps)
        self.rn = nn.RMSNorm(d, eps=eps)

    def _project(self, x):
        B, T, _ = x.shape
        H, D = self.h, self.d
        shp = lambda t: t.view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        q = self.qn(shp(self.q(x)))
        k = self.kn(shp(self.k(x)))
        v = shp(self.v(x))
        rho = self.rn(shp(self.r(x)))
        return q, k, v, rho

    def _scores(self, q, k, T, device):
        scale = self.d ** -0.5
        scores = (q @ k.transpose(-1, -2)) * scale
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        return F.softmax(scores, dim=-1)

    def _out(self, o, x, B, T):
        o = o.transpose(1, 2).reshape(B, T, self.h * self.d)
        return self.o(o.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v, rho = self._project(x)
        p = self._scores(q, k, T, x.device)
        o_sa = p @ v
        k_bar = p @ k
        s2 = (rho @ k.transpose(-1, -2)) * p
        corr = s2 @ v
        gate = (rho * k_bar).sum(-1, keepdim=True)
        out = o_sa * (1.0 + gate) - corr
        return self._out(out, x, B, T)

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """Literal covariance-matrix form (paper eq. 3/8, not the streaming
        simplification) -- test oracle for forward()'s efficient form."""
        B, T, _ = x.shape
        hp = torch.promote_types(x.dtype, torch.float32)
        q, k, v, rho = (t.to(hp) for t in self._project(x))
        p = self._scores(q, k, T, x.device)

        v_bar = p @ v
        k_bar = p @ k
        v_c = v.unsqueeze(-3) - v_bar.unsqueeze(-2)  # (B,H,Ti,Tj,Dv)
        k_c = k.unsqueeze(-3) - k_bar.unsqueeze(-2)  # (B,H,Ti,Tj,Dk)
        weighted = p.unsqueeze(-1) * v_c
        sigma = torch.einsum("bhijv,bhijk->bhivk", weighted, k_c)
        corr = torch.einsum("bhivk,bhik->bhiv", sigma, rho)
        out = v_bar - corr
        return self._out(out, x, B, T)
