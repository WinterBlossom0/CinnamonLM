"""OSDN: DeltaNet with an online diagonal preconditioner on the write key.

Zhou, Li, Liu, Lin, Ge, Ye, "Improving Delta Rule with Provable Online Preconditioning in Linear
Attention", arXiv:2605.13473. This is vanilla OSDN (Alg. 1, no APF), the variant of the paper's
matched 340M runs (Sec. 6, App. F, App. I). It adds no parameters to DeltaNet. The official code
(github.com/Lhongpei/OSDN) has no layer code. The layer is therefore the backbone that its 340M config
names, fla DeltaNet (fla/layers/delta_net.py), with the config values: conv_size 4, qk_activation
silu, qk_norm l2, use_beta, use_gate false, output norm. H = n_heads, D = head_dim, dk = dv = D.

    q, k, v = SiLU(ShortConv4(x W^T))              depthwise causal conv, no bias, one per projection
    q = L2Norm(q) * D^-1/2,  k = L2Norm(k)         per head; q scale as in App. C Listing 1 (q * K**-0.5)
    beta = sigmoid(x W_b^T)                        fla's b_proj, no bias
    d_t from phase 1 (Alg. 1, eq. 4), see precondition_triton.py
    S_t = S_{t-1} + beta_t (d_t * k_t)^T (v_t - k_t S_{t-1}),  o_t = q_t S_t
                                                   eq. 5 in this project's [K,V] row-vector convention
    y = RMSNorm_D(o) W_o^T                         one norm weight shared across heads, as fla's o_norm

Phase 2 is the chunked WY form (App. A.2, Listing 1): kda_scan with k as the read key, d*k as the
write key and no decay is exactly that form.

Init (post_init, after the model's blanket init): conv weights N(0, 0.02) as fla's model init,
output-norm weight 1.

Choices between sources, and protocol settings:
- beta has no bias, as fla's b_proj in the backbone of the paper's runs. The background formula in
  Sec. 3, sigma(w^T x + b), has a bias.
- Phase 1 is beta-aware (Alg. 1) with d_1 = 1 (App. F and the 340M config). The released
  DeltaNet chunk op (chunk_osgm.py) instead runs a phase 1 without beta and starts d at 0.
- expand_k = expand_v = H*D/d_model is a free knob here (1.25 at (4, 5)), not the paper's 1. With
  expand 1 the closest totals to the 23,200 budget are 22,740 at (16, 1), where a unit-norm key of
  one coordinate is only a sign, and 22,232 at (8, 2).
- Shared protocol of every arm: d_model 16 and 4 layers, not 1024 and 24 (App. F); d and S reset at
  each 128-token packed block, not at each document boundary; fp32 training with Muon + AdamW, not
  AMP bf16 with AdamW (App. F).
Numerics, measured at (4, 5) on validation tokens (8 and 32 rows of 128, seeds 0 to 2, every block):
- At init, sum(k^2) before the L2 norm (median 7e-6 to 1.1e-5) is near the L2 eps 1e-6, so the keys
  have a norm below 1: median 0.94 to 0.96, 5th percentile 0.76 to 0.84.
- At init, the mean square of o (0.9e-7 to 2.4e-7) is below the output-norm eps 1e-6, so the RMS of
  the normalized output is 0.27 to 0.42, not 1. In training (seed 0, batch 1024, protocol schedule),
  block 0 goes deeper into this regime (median mean square 3.8e-10 and normalized RMS 0.02 at step
  1000), and blocks 1 to 3 leave it (normalized RMS 0.81 to 0.91). Every parameter still had a
  nonzero gradient at each of the 1000 steps.
- With eta = 0.003 and T = 128, d stays in [0.85, 1.39] for any key with norm <= 1 (each step moves a
  coordinate up by at most eta and down by at most 0.39 eta), so the [0.5, 2] box never binds. With
  unit keys the range is [0.96, 1.10], so the write key differs from DeltaNet's by at most 10%; at
  init d is in [1.000, 1.061]. The paper trains at up to 4096 tokens (App. F).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# fla DeltaNet and fla KDA use the same short conv, L2 norm and UT inverse, so this arm uses the KDA arm's functions.
from kda.attention import (
    CONV_INIT_STD,
    CONV_SIZE,
    causal_conv_silu,
    inv_unit_lower,
    l2norm,
)
from kda.rmsnorm_triton import TritonRMSNorm
from kda.scan_triton import kda_scan

from .precondition_triton import osdn_precondition


class OSDN(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        # kda_scan uses tl.arange over the chunk, which needs a power of two, and tl.dot, which needs >= 16.
        if chunk < 16 or (chunk & (chunk - 1)) != 0:
            raise ValueError(f"chunk {chunk} must be a power of two >= 16")
        width = n_heads * head_dim
        self.h, self.d, self.chunk = n_heads, head_dim, chunk
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(d_model, width, bias=False)
        self.v = nn.Linear(d_model, width, bias=False)
        self.beta = nn.Linear(d_model, n_heads, bias=False)
        # 3-D weights, so muon.split_params sends them to AdamW, as fla's ShortConvolution weights.
        self.q_conv, self.k_conv, self.v_conv = (
            nn.Conv1d(width, width, CONV_SIZE, groups=width, bias=False) for _ in range(3))
        self.on = TritonRMSNorm(head_dim, eps=eps)
        self.o = nn.Linear(width, d_model, bias=False)
        self.post_init()

    def post_init(self):
        with torch.no_grad():
            for conv in (self.q_conv, self.k_conv, self.v_conv):
                conv.weight.normal_(0.0, CONV_INIT_STD)
            self.on.weight.fill_(1.0)

    def _project(self, x):
        """q, k, v as (B,T,H,D) and beta as (B,T,H,1)."""
        B, T, _ = x.shape
        heads = lambda t: t.reshape(B, T, self.h, self.d)
        q = l2norm(heads(causal_conv_silu(self.q_conv, self.q(x)))) * self.d ** -0.5
        k = l2norm(heads(causal_conv_silu(self.k_conv, self.k(x))))
        v = heads(causal_conv_silu(self.v_conv, self.v(x)))
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1)
        return q, k, v, beta

    def _out(self, o, B, T):
        """o is (B,H,T,D)."""
        return self.o(self.on(o).transpose(1, 2).reshape(B, T, self.h * self.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, C = self.h, self.d, self.chunk
        q, k, v, beta = self._project(x)
        k_tilde = osdn_precondition(k, beta.squeeze(-1)) * k

        # The padding comes after the last real token, so causality keeps every real output unchanged.
        pad = (-T) % C
        if pad > 0:
            q, k, k_tilde, v, beta = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, k_tilde, v, beta))
        Tp, N = T + pad, (T + pad) // C
        chunks = lambda t: t.transpose(1, 2).reshape(B, H, N, C, -1).float()
        q, k, k_tilde, v, beta = (chunks(t) for t in (q, k, k_tilde, v, beta))

        # kda_scan and solve_triangular are verified in fp32 only.
        with torch.autocast(x.device.type, enabled=False):
            Tm = inv_unit_lower(((beta * k) @ k_tilde.transpose(-1, -2)).tril(-1))
            no_decay = k.new_ones(B, H, N, D)
            o = kda_scan(k, k_tilde, q, q, k_tilde, beta, v, Tm, no_decay)
        return self._out(o.reshape(B, H, Tp, D)[:, :, :T].to(x.dtype), B, T)
