"""KDA: Kimi Delta Attention, the complete KDA layer of Kimi Linear (Kimi Team, arXiv:2510.26692).

H = n_heads, d = head_dim, d_k = d_v = d. x is the block input after the block RMSNorm: Fig. 3 puts
the norm in the block, before the layer.

    q, k, v = SiLU(ShortConv4(W x))                     Sec. 4; depthwise causal conv, no bias, one per projection
    q = L2Norm(q) * d^-1/2,  k = L2Norm(k)              Sec. 4; scale from App. C Listing 1 (q = q * K**-0.5)
    log alpha = -exp(A_log[h]) * softplus(W_up W_down x + dt_bias)    Sec. 4, rank d; f from the official code
    beta = sigmoid(W_beta x)                            Sec. 4
    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t      Eq. 1
    y = W_o (sigmoid(W_g_up W_g_down x) * RMSNorm(o))   Eq. 10; head-wise norm, one weight [d] for all heads

The chunked form is Sec. 3.1, Eqs. 6-9, with chunk length C = chunk. intra_triton.py computes the
intra-chunk matrices Aqk and Akk with the decay difference G_r - G_i inside the exp, as App. C Listing 1
does, so every decay factor is at most 1 and no gate value needs a clamp. inv_unit_lower gives the UT
transform (I + Diag(beta) Akk)^-1, and scan_triton.py does the sequential state update across chunks.

The paper does not give these details. They come from the official code (fla fla/layers/kda.py and
fla/models/kda, Moonshot modeling_kimi.py of Kimi-Linear-48B-A3B):
- f: A_log per head, dt_bias per channel, no lower bound. fla's bounded gate (safe_gate) is an optional
  speed path that the released model does not use.
- Output-gate rank d (Moonshot g_a_proj). The paper says only "similar to the forget gate".
- Init (fla _init_weights and KimiDeltaAttention.__init__): A_log = log U(1, 16); dt_bias = inverse
  softplus of a log-uniform dt in [1e-3, 1e-1]; conv weights N(0, 0.02); norm weight 1. model.py gives
  the Linear weights N(0, 0.02).
- No bias tensors: Eq. 10 and Moonshot's g_b_proj have no output-gate bias (fla adds one, zero at init).
- Output-norm eps is the constructor eps (1e-6, the fla KDAConfig default).

Differences from the paper and the official code. All come from the shared protocol of every arm:
- KDA in every layer. The paper's model is a 3:1 hybrid of KDA and MLA layers (Sec. 4), so this arm is
  the KDA layer type, not "Kimi Linear".
- d_model 16 and a small head_dim (train.py HEAD_CONFIG), not d = 128.
- Context 128: conv state and S start from zero at each 128-token packed block.

Numerics, measured:
- At d_model 16 with conv init N(0, 0.02), sum(k^2) at init is near the L2Norm eps 1e-6, so the keys at
  init have a norm below 1: median 0.96, 5th percentile 0.82 in block 0 at (4, 4) on real tokens.
- G = cumsum(log alpha) is fp32, so exp(G_r - G_i) has a rounding error of about |G| * 6e-8. After 600
  real steps at batch 1024 the chunk log-decay span reached 2,500, and the layer matched an fp64 token
  loop to 2.5e-5 (output) and 7.7e-5 (gradient), relative to the maximum.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .intra_triton import kda_intra
from .rmsnorm_triton import gated_rmsnorm
from .scan_triton import kda_scan_given_a

CONV_SIZE = 4
CONV_INIT_STD = 0.02  # fla _init_weights: nn.Conv1d weight ~ N(0, initializer_range 0.02)
L2_EPS = 1e-6  # fla l2norm: x / sqrt(sum(x^2) + eps)
A_INIT_RANGE = (1.0, 16.0)  # fla: A_log = log U(1, 16)
DT_INIT_RANGE = (1e-3, 1e-1)  # fla: softplus(dt_bias) is log-uniform in this range
DT_INIT_FLOOR = 1e-4


def inv_unit_lower(M: torch.Tensor) -> torch.Tensor:
    """(I + M)^-1 for strictly lower-triangular M, batched over the leading dims.

    unitriangular=True reads the diagonal as 1, and solve_triangular's backward discards the gradient
    of the diagonal and the upper triangle, which are not free entries of a unit lower-triangular matrix.
    """
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    return torch.linalg.solve_triangular(M, eye, upper=False, unitriangular=True)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + L2_EPS)


def causal_conv_silu(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """x is (B,T,W); left padding with CONV_SIZE-1 zeros makes output t depend on inputs t-CONV_SIZE+1..t only."""
    padded = F.pad(x.transpose(1, 2), (CONV_SIZE - 1, 0))
    return F.silu(F.conv1d(padded, conv.weight, groups=conv.groups)).transpose(1, 2)


class KDA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        width = n_heads * head_dim
        self.h, self.d, self.chunk, self.eps = n_heads, head_dim, chunk, eps
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(d_model, width, bias=False)
        self.v = nn.Linear(d_model, width, bias=False)
        # 3-D weights, so muon.split_params sends them to AdamW.
        self.q_conv, self.k_conv, self.v_conv = (
            nn.Conv1d(width, width, CONV_SIZE, groups=width, bias=False) for _ in range(3))
        self.f_down = nn.Linear(d_model, head_dim, bias=False)
        self.f_up = nn.Linear(head_dim, width, bias=False)
        self.A_log = nn.Parameter(torch.empty(n_heads))
        self.dt_bias = nn.Parameter(torch.empty(width))
        # fla kda.py marks both as exempt from weight decay; train._make_optimizers honours the flag.
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.beta = nn.Linear(d_model, n_heads, bias=False)
        self.g_down = nn.Linear(d_model, head_dim, bias=False)
        self.g_up = nn.Linear(head_dim, width, bias=False)
        self.o_norm_weight = nn.Parameter(torch.empty(head_dim))
        self.o = nn.Linear(width, d_model, bias=False)
        self.post_init()

    def post_init(self):
        with torch.no_grad():
            for conv in (self.q_conv, self.k_conv, self.v_conv):
                conv.weight.normal_(0.0, CONV_INIT_STD)
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(*A_INIT_RANGE).log())
            low, high = (math.log(bound) for bound in DT_INIT_RANGE)
            dt = (torch.rand_like(self.dt_bias) * (high - low) + low).exp().clamp(min=DT_INIT_FLOOR)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.o_norm_weight.fill_(1.0)

    def _project(self, x):
        """q (scaled), k, v and log alpha as (B,T,H,D); beta as (B,T,H,1)."""
        B, T, _ = x.shape
        heads = lambda t: t.reshape(B, T, self.h, self.d)
        q = l2norm(heads(causal_conv_silu(self.q_conv, self.q(x)))) * self.d ** -0.5
        k = l2norm(heads(causal_conv_silu(self.k_conv, self.k(x))))
        v = heads(causal_conv_silu(self.v_conv, self.v(x)))
        decay_input = heads(self.f_up(self.f_down(x)) + self.dt_bias)
        log_alpha = -self.A_log.exp()[:, None] * F.softplus(decay_input)
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1)
        return q, k, v, log_alpha, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.d
        C = min(self.chunk, T)
        assert (C & (C - 1)) == 0, f"chunk size {C} must be a power of two (tl.arange requires it)"
        q, k, v, log_alpha, beta = self._project(x)

        # The padding comes after the last real token, so causality keeps every real output unchanged.
        pad = (-T) % C
        if pad:
            q, k, v, log_alpha, beta = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, log_alpha, beta))
        Tp, N = T + pad, (T + pad) // C
        chunks = lambda t: t.transpose(1, 2).reshape(B, H, N, C, -1).float()
        q, k, v, log_alpha, beta = (chunks(t) for t in (q, k, v, log_alpha, beta))

        q_abs, k_abs, k_end, eg_last, aqk, akk = kda_intra(q, k, log_alpha)
        Tm = inv_unit_lower(beta * akk)
        o = kda_scan_given_a(k_abs, k_end, q_abs, aqk, beta, v, Tm, eg_last)
        o = o.reshape(B, H, Tp, D)[:, :, :T].transpose(1, 2).to(x.dtype)

        gate = self.g_up(self.g_down(x)).reshape(B, T, H, D)
        y = gated_rmsnorm(o, gate, self.o_norm_weight, self.eps)
        return self.o(y.reshape(B, T, H * D))
