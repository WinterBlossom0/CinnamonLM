"""OSDN: plain delta-rule attention with a per-feature online preconditioner
on the write-key (Zhou, Li, Liu, Lin, Ge, Ye, arXiv:2605.13473). KDA's chunk
algorithm (kda/attention.py) reduces exactly to this when decay is zero:
k plays every "read" role KDA's undecayed k_bar_c/k_abs/q_c play, and the
preconditioned key k_tilde = d*k plays every "write" role (k_til_c/k_end).
So the existing kda_scan kernel is reused unchanged; only the preconditioner
sweep and the (decay-free) Tm construction are new here.

State update (paper eq. 5, transposed to this project's [K,V] row-vector
convention, o_t = q_t @ S -- see kda/attention.py's docstring):
    S_t = S_{t-1} + b_t * k_tilde_t (col) (v_t - k_t (row) @ S_{t-1})

This is vanilla OSDN (Algorithm 1), not OSDN-APF (Adaptive Preconditioner
Forgetting, paper's Sec. 4.2) -- APF adds a learned retention gate on the
preconditioner and is what the paper's own 1.3B-scale experiments actually
use and recommend for stability; vanilla OSDN is only evaluated at 340M in
the paper. Read the paper in full before assuming vanilla is sufficient for
any conclusion beyond this project's tiny scale.

Per the paper's own stated limitations (Sec. 7): OSDN's gains are
recall-specific (JRT-style repeated-key retrieval), not general language-
modeling perplexity -- the authors explicitly state downstream WikiText/
LAMBADA/LongBench averages "remain at parity with DeltaNet rather than
improving uniformly" and they "do not claim a universal benchmark lift."
This project's eval loss is a general LM metric, not a recall benchmark, so
OSDN underperforming KDA here is consistent with the paper's own claims, not
contrary to them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kda.scan_triton import kda_scan
from kda.attention import inv_unit_lower
from kda.rmsnorm_triton import TritonRMSNorm
from .precondition_triton import osdn_precondition, ETA, D_MIN, D_MAX


class OSDN(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        h, d = n_heads, head_dim
        self.h, self.d, self.chunk, self.eps = h, d, chunk, eps
        self.q = nn.Linear(d_model, h * d, bias=False)
        self.k = nn.Linear(d_model, h * d, bias=False)
        self.v = nn.Linear(d_model, h * d, bias=False)
        self.o = nn.Linear(h * d, d_model, bias=False)
        self.beta = nn.Linear(d_model, h, bias=True)
        self.qn = TritonRMSNorm(d, eps=eps)
        self.kn = TritonRMSNorm(d, eps=eps)
        self.on = TritonRMSNorm(d, eps=eps)
        with torch.no_grad():
            self.beta.bias.fill_(0.0)

    def _project(self, x):
        B, T, _ = x.shape
        H, D = self.h, self.d
        q = self.qn(self.q(x).view(B, T, H, D))
        k = F.normalize(self.kn(self.k(x).view(B, T, H, D)), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        return q, k, v, beta

    def _out(self, o, x, B, T):
        o = self.on(o).transpose(1, 2).reshape(B, T, self.h * self.d)
        return self.o(o.to(x.dtype))

    def _precondition_plain(self, k, beta):
        """Phase 1 (paper Algorithm 1), plain PyTorch: sequential online
        sweep over the full sequence, before chunking -- direct port of the
        authors' reference (Lhongpei/OSDN, src/osdn/reference.py:
        online_preconditioner). Used only by reference() as an independent
        oracle; forward() uses the Triton kernel (precondition_triton.py),
        profiled at ~33% of step time as a plain loop here (CPU launch-
        bound: the box clamp breaks any closed form, so it's a genuine
        per-token loop, but a cheap one -- elementwise, no matmul).

        Returns d_traj (pre-update d used at each t), not k_tilde -- both
        callers multiply by k themselves so Phase 1 is identical either way.
        """
        B, T, H, D = k.shape
        d = k.new_ones(B, H, D)
        d_traj = torch.empty(B, T, H, D, device=k.device, dtype=k.dtype)
        beta = beta.squeeze(-1)
        for t in range(T):
            k_t, b_t = k[:, t], beta[:, t]
            d_traj[:, t] = d
            k_sq = k_t.square()
            inner = (d * k_sq).sum(-1, keepdim=True)
            b = b_t[..., None]
            step = b * (1.0 - b * inner) / k_sq.sum(-1, keepdim=True).clamp_min(self.eps)
            d = (d + ETA * step * k_sq).clamp(D_MIN, D_MAX)
        return d_traj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.d
        C = min(self.chunk, T)
        assert C & (C - 1) == 0, f"chunk size {C} must be a power of two (tl.arange requires it)"
        q, k, v, beta = self._project(x)
        k_tilde = osdn_precondition(k, beta.squeeze(-1), self.eps) * k

        pad = (-T) % C
        if pad:
            q, k, k_tilde, v, beta = (F.pad(t, (0, 0, 0, 0, 0, pad))
                                      for t in (q, k, k_tilde, v, beta))
        Tp, N = T + pad, (T + pad) // C

        hp = torch.promote_types(x.dtype, torch.float32)
        shp = lambda t: t.transpose(1, 2).reshape(B, H, N, C, -1).to(hp)
        q, k, k_tilde, v, beta = shp(q), shp(k), shp(k_tilde), shp(v), shp(beta)

        with torch.autocast(x.device.type, enabled=False):
            o = self._chunk_scan(q, k, k_tilde, v, beta, B, H, D, N, C, T, Tp)
        return self._out(o, x, B, T)

    def _chunk_scan(self, q, k, k_tilde, v, beta, B, H, D, N, C, T, Tp):
        Tm = inv_unit_lower(((beta * k) @ k_tilde.transpose(-1, -2)).tril(-1))
        ones = k.new_ones(B, H, N, D)
        outs = kda_scan(k, k_tilde, q, q, k_tilde, beta, v, Tm, ones)
        return outs.reshape(B, H, Tp, D)[:, :, :T]

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """Naive O(T) recurrence. Test oracle for the chunked path."""
        B, T, _ = x.shape
        hp = torch.promote_types(x.dtype, torch.float32)
        q, k, v, beta = (t.to(hp) for t in self._project(x))
        k_tilde = self._precondition_plain(k, beta) * k

        S = q.new_zeros(B, self.h, self.d, self.d)
        outs = []
        for t in range(T):
            kt, ktt = k[:, t].unsqueeze(-1), k_tilde[:, t].unsqueeze(-1)
            vt, bt = v[:, t].unsqueeze(-2), beta[:, t].unsqueeze(-1)
            S = S + bt * (ktt @ (vt - kt.transpose(-1, -2) @ S))
            outs.append((q[:, t].unsqueeze(-2) @ S).squeeze(-2))
        return self._out(torch.stack(outs, 2), x, B, T)
