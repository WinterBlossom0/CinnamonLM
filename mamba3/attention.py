"""Mamba-3 SISO layer (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, "Mamba-3: Improved Sequence
Modeling using State Space Principles", arXiv:2603.15569, ICLR 2026): the layer of a pure Mamba-3
model (Sec. 3.4, Fig. 2), in every block.

Bias-free projections of u_t give z_t, x_t (H heads of width P), B_t, C_t (width N, one group shared by
all heads, Sec. 3.3), per-head dd_dt, dd_A, u_lam, and N/2 angle inputs u_theta. Per head and token:

    dt_t    = softplus(dd_dt + dt_bias),  A_t = min(-softplus(dd_A), -1e-4)    data-dependent A (Sec. 2)
    alpha_t = exp(dt_t A_t),  lam_t = sigmoid(u_lam)                            (App. A.3, Table 8)
    beta_t  = (1 - lam_t) dt_t alpha_t,  gamma_t = lam_t dt_t                   exp.-trapezoidal (Prop. 1)
    B'_t    = w_B RMSNorm(B_t) + b_B,  C'_t = w_C RMSNorm(C_t) + b_C           BCNorm, head-wise bias (Sec. 3.4)
    phi_t   = sum_{s<=t} dt_s pi tanh(u_theta_s)                                N/2 cumulative angles
    Bbar_t  = Rot(phi_t) B'_t,  Cbar_t = Rot(phi_t) C'_t                        complex state (Props. 2-4)
    h_t     = alpha_t h_{t-1} + beta_t Bbar_{t-1} x_{t-1}^T + gamma_t Bbar_t x_t^T   (Prop. 4)
    y_t     = Cbar_t^T h_t + D x_t
    out_t   = W_out (y_t * SiLU(z_t))                                           output gate (App. C)

Rot is block-diagonal with one 2x2 rotation on each of the N/2 state pairs (Prop. 2), so the complex
state is a data-dependent RoPE on B and C (Prop. 4), and the layer needs no position embedding. The
layer has no short convolution and no activation on x, B, C (Sec. 3.4, Table 5a), and no output norm,
which pure Mamba-3 models remove (Sec. 3.4).

The recurrence is computed in its parallel form, eq. (7): y = (L * Cbar Bbar^T) x + D x with
L[t,t] = gamma_t and L[t,s] = alpha_{s+1} ... alpha_t (gamma_s + (1 - lam_{s+1}) dt_{s+1}) for s < t,
evaluated chunk by chunk with the state h carried between chunks (the chunked SSD algorithm, Dao and
Gu, "Transformers are SSMs", arXiv:2405.21060, Sec. 6), in fp32 with the phase sum in fp64.

Constructor knobs: n_heads = H and head_dim = P, so d_inner = H P; the state size N is STATE_SIZE.

From the reference code (mamba_ssm 2.3.2 modules/mamba3.py and ops/triton/mamba3): the pi tanh angle
bound, the phase taken mod 2 pi, rotation of the interleaved pairs (2i, 2i+1) by +phi, the D skip
inside the gate, BCNorm eps 1e-5, dt init log-uniform in [1e-3, 1e-1] with floor 1e-4, and D = 1.
From the paper: b_B = b_C = 1, trainable (App. F, Table 10a). BCNorm weights start at 1.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_SIZE = 16  # N; must be even, because Rot acts on N/2 pairs
DT_MIN, DT_MAX, DT_INIT_FLOOR = 1e-3, 1e-1, 1e-4
A_FLOOR = 1e-4
BC_NORM_EPS = 1e-5


def segment_sums(cumulative: torch.Tensor, diagonal: int) -> torch.Tensor:
    """out[..., i, j] = cumulative[..., i] - cumulative[..., j] where i - j >= -diagonal, else -inf."""
    length = cumulative.shape[-1]
    keep = torch.ones(length, length, dtype=torch.bool, device=cumulative.device).tril(diagonal)
    return (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~keep, -math.inf)


def strictly_causal_scan(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         log_decay: torch.Tensor, chunk: int) -> torch.Tensor:
    """y_t = sum_{s<t} exp(sum_{r=s+1..t} log_decay_r) (query_t . key_s) value_s, per head.

    query, key: (batch, H, T, N); value: (batch, H, T, P); log_decay: (batch, H, T), all entries <= 0.
    """
    length = query.shape[2]
    padding = (-length) % chunk
    if padding:
        # Zeros after the last token change no earlier output, because the scan is causal.
        query, key, value = (F.pad(tensor, (0, 0, 0, padding)) for tensor in (query, key, value))
        log_decay = F.pad(log_decay, (0, padding))
    n_chunks = (length + padding) // chunk
    query, key, value = (tensor.unflatten(2, (n_chunks, chunk)) for tensor in (query, key, value))
    log_decay = log_decay.unflatten(2, (n_chunks, chunk))  # (batch, H, chunks, chunk)
    cumulative = log_decay.cumsum(dim=-1)

    intra_decay = torch.exp(segment_sums(cumulative, -1))
    intra = ((query @ key.transpose(-1, -2)) * intra_decay) @ value

    decay_to_chunk_end = torch.exp(cumulative[..., -1:] - cumulative).unsqueeze(-1)
    chunk_states = (key * decay_to_chunk_end).transpose(-1, -2) @ value  # (batch, H, chunks, N, P)
    # Entry c of the padded sequence is the state that enters chunk c; entry 0 is the zero initial state.
    chunk_states = F.pad(chunk_states, (0, 0, 0, 0, 1, 0))
    chunk_totals = F.pad(cumulative[..., -1], (1, 0)).cumsum(dim=-1)
    chunk_decay = torch.exp(segment_sums(chunk_totals, 0))  # (batch, H, chunks+1, chunks+1)
    entering_states = torch.einsum("bhzc,bhcnp->bhznp", chunk_decay, chunk_states)[:, :, :-1]
    inter = (query @ entering_states) * torch.exp(cumulative).unsqueeze(-1)

    return (intra + inter).flatten(2, 3)[:, :, :length]


def cumulative_phase(dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """phi_t = sum_{s<=t} dt_s theta_s mod 2 pi; dt: (batch, H, T), theta: (batch, T, N/2) -> (batch, H, T, N/2)."""
    increments = dt.double().unsqueeze(-1) * theta.double().unsqueeze(1)
    # The sum runs in fp64 and is reduced mod 2 pi before the fp32 cast, so phi keeps fp32 precision at any |phi|.
    return torch.remainder(increments.cumsum(dim=2), 2.0 * math.pi).to(theta.dtype)


def rotate_pairs(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate each pair (2i, 2i+1) of the last dim of vectors by the unit complex number rotation[..., i]."""
    # Pair (e, o) as the complex number e + io, so the rotation by phi is one product with exp(i phi).
    pairs = torch.view_as_complex(vectors.unflatten(-1, (-1, 2)))
    return torch.view_as_real(pairs * rotation).flatten(-2)


class Mamba3(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        del eps  # the block norm eps; BCNorm uses BC_NORM_EPS
        self.n_heads, self.head_dim, self.chunk = n_heads, head_dim, chunk
        d_inner = n_heads * head_dim
        self.z_projection = nn.Linear(d_model, d_inner, bias=False)
        self.x_projection = nn.Linear(d_model, d_inner, bias=False)
        self.b_projection = nn.Linear(d_model, STATE_SIZE, bias=False)
        self.c_projection = nn.Linear(d_model, STATE_SIZE, bias=False)
        self.dt_projection = nn.Linear(d_model, n_heads, bias=False)
        self.a_projection = nn.Linear(d_model, n_heads, bias=False)
        self.lambda_projection = nn.Linear(d_model, n_heads, bias=False)
        self.theta_projection = nn.Linear(d_model, STATE_SIZE // 2, bias=False)
        self.out_projection = nn.Linear(d_inner, d_model, bias=False)
        self.b_norm = nn.RMSNorm(STATE_SIZE, eps=BC_NORM_EPS)
        self.c_norm = nn.RMSNorm(STATE_SIZE, eps=BC_NORM_EPS)
        self.dt_bias = nn.Parameter(torch.empty(n_heads))
        # The reference shape (H, R, N) with MIMO rank R = 1 broadcasts over T, and a 3-D bias stays off Muon.
        self.b_bias = nn.Parameter(torch.empty(n_heads, 1, STATE_SIZE))
        self.c_bias = nn.Parameter(torch.empty(n_heads, 1, STATE_SIZE))
        self.d_skip = nn.Parameter(torch.empty(n_heads))
        # mamba_ssm modules/mamba3.py marks dt_bias and D as exempt from weight decay.
        self.dt_bias._no_weight_decay = True
        self.d_skip._no_weight_decay = True
        self.post_init()

    @torch.no_grad()
    def post_init(self) -> None:
        dt = torch.exp(torch.rand_like(self.dt_bias) * (math.log(DT_MAX) - math.log(DT_MIN)) + math.log(DT_MIN))
        dt = dt.clamp(min=DT_INIT_FLOOR)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        for ones in (self.b_bias, self.c_bias, self.d_skip, self.b_norm.weight, self.c_norm.weight):
            ones.fill_(1.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, length, _ = u.shape

        def per_head(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, length, self.n_heads, -1).transpose(1, 2)  # (batch, H, T, width)

        # One matmul for the eight input projections (layer fwd+bwd 1.09x faster); each stays its own
        # nn.Linear, so Muon orthogonalises each projection matrix separately.
        projections = (self.z_projection, self.x_projection, self.b_projection, self.c_projection,
                       self.dt_projection, self.a_projection, self.lambda_projection, self.theta_projection)
        projected = F.linear(u, torch.cat([projection.weight for projection in projections]))
        z, x, b_input, c_input, dt_input, a_input, lam_input, theta_input = projected.split(
            [projection.out_features for projection in projections], dim=-1)
        z, x = per_head(z), per_head(x)
        dt = per_head(F.softplus(dt_input + self.dt_bias)).squeeze(-1)  # (batch, H, T)
        a = per_head((-F.softplus(a_input)).clamp(max=-A_FLOOR)).squeeze(-1)
        lam = per_head(torch.sigmoid(lam_input)).squeeze(-1)
        gamma = lam * dt
        # beta_{s+1} writes Bbar_s x_s one step late, so its weight (1 - lam_{s+1}) dt_{s+1} joins gamma_s off the diagonal.
        late_weight = F.pad(((1.0 - lam) * dt)[..., 1:], (0, 1))
        phase = cumulative_phase(dt, math.pi * torch.tanh(theta_input))
        rotation = torch.polar(torch.ones_like(phase), phase)  # exp(i phi), (batch, H, T, N/2)
        b_bar = rotate_pairs(self.b_norm(b_input).unsqueeze(1) + self.b_bias, rotation)  # (batch, H, T, N)
        c_bar = rotate_pairs(self.c_norm(c_input).unsqueeze(1) + self.c_bias, rotation)
        y = strictly_causal_scan(c_bar, b_bar, x * (gamma + late_weight).unsqueeze(-1), dt * a, self.chunk)
        diagonal = (c_bar * b_bar).sum(dim=-1) * gamma + self.d_skip.unsqueeze(-1)
        y = (y + diagonal.unsqueeze(-1) * x) * F.silu(z)
        return self.out_projection(y.transpose(1, 2).reshape(batch, length, self.n_heads * self.head_dim))
