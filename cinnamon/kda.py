"""KDA - Kimi Delta Attention: gated delta rule with channelwise decay.

Recurrence (row vectors, S in R^{d_k x d_v}, o_t = q_t S_t):

    S_t = (I - b_t k_t^T k_t) Diag(a_t) S_{t-1} + b_t k_t^T v_t

i.e. the state decays channelwise, then the delta rule corrects it toward v_t.

Chunked form.  Inside a chunk let g_t = cumsum(log a) and P_t = Diag(g_t)^-1 S_t.
The gate then cancels out of the recurrence entirely:

    P_t = P_{t-1} + b_t k_til_t^T (v_t - k_bar_t P_{t-1})

with k_bar = k*exp(g) (the read key) and k_til = k*exp(-g) (the write key).  That
is an ungated delta rule with asymmetric keys, which the UT transform solves for a
whole chunk in closed form.  Cost is seq/chunk sequential steps instead of seq.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

# Max per-chunk |gcum|.  Enforced in _project as a per-step floor _G_RANGE/chunk.
#
# Why a bound is needed at all: the intra-chunk term exp(g_t - g_s) is causal
# (t>=s) with gcum monotone decreasing, so the exponent is <=0 and the value <=1.
# But evaluated as a matmul it must split into exp(g_t)*exp(-g_s) -- a matmul
# needs each factor to depend on one index -- and exp(-g_s) is unbounded.
# Channelwise decay means it cannot be folded into a [C,C] mask either.
# Midpoint centring (see forward) halves the range but only moves the cliff.
#
# Split-form error vs exact recurrence:  3e-7 @ |gcum| 7.6 | 1e-1 @ 9.5 | 4e-1 @ 18
# Real run reached 23.3 by step 250.  No NaN, no loss spike -- silently wrong.
#
# fla-org/flash-linear-attention fla/ops/kda subtracts INSIDE the exp
# (exp2(b_gn - b_gk)), so no bound is needed.  Implemented and measured here:
# exact at any drift (1.5e-15 @ |gcum| 640) but needs a [C,C,D] intermediate +
# reduce instead of a tensor-core matmul -> 2.4x time, 1.5x memory
# (395->165 tok/s, 7.9->12.1 GB), OOM at chunk 64.  Reverted.
#
# Shipped: keep the split, bound gcum (fla's `safe_gate`, tightened from per-step
# to per-chunk).  Cost: retention/token >= exp(-_G_RANGE/chunk) = 0.855 @ 64.
# A Triton port removes the bound.
_G_RANGE = 10.0

# Repeated squaring is only conditionally stable.  M is bounded (|M| <= 1 by
# Cauchy-Schwarz on unit keys with beta <= 1), but (I+M)^-1 grows like 2^C in the
# worst case, and keys inside a chunk do become correlated during training.
# Measured against the exact sequential recurrence on a real blown-up batch:
#
#     chunk 64 -> inf          chunk 32 -> 33% relative error
#     chunk 16 -> 3.9e-06      chunk  8 -> 4.8e-07
#
# So 16 is the largest block the squaring form can be trusted with.  That used to
# cap kda_chunk itself at 16, which cost seq/16 sequential state-recurrence steps
# instead of seq/64 -- 4x more, and the chunk loop is the single hottest sequential
# path in the model (2336 iterations per forward at c_max=32, against 64 for the
# recurrence loop itself).
_INV_BLOCK = 16


def _inv_squaring(M: torch.Tensor) -> torch.Tensor:
    """(I + M)^-1 for strictly lower-triangular M, by repeated squaring.

    M is nilpotent (M^C = 0), and X_k (I+M) = I - M^(2^(k+1)), so log2(C) steps
    suffice.  Matmuls only, so it lowers cleanly to XLA -- unlike triangular_solve,
    which risks a host fallback on TPU.  Stable only up to _INV_BLOCK.
    """
    C = M.shape[-1]
    I = torch.eye(C, device=M.device, dtype=M.dtype)
    X, P, k = I - M, M @ M, 2
    while k < C:
        X = X @ (I + P)
        P = P @ P
        k *= 2
    return X


def inv_unit_lower(M: torch.Tensor, block: int = _INV_BLOCK) -> torch.Tensor:
    """(I + M)^-1 for strictly lower-triangular M, by blocked forward substitution.

    Partition L = I+M into `block`-sized blocks.  L is unit lower triangular, so
    with X = L^-1:

        X_ii = (I + M_ii)^-1                                  (squaring, safe at 16)
        X_ij = -X_ii @ sum_{k=j}^{i-1} M_ik X_kj    for j < i

    Each diagonal inverse stays inside the numerically safe block size, so the
    chunk itself is free to be 64.  This is not on the hot path: it runs once per
    KDA forward, fully batched over (B, H, N), while what it buys is a 4x cut in
    the sequential state-recurrence loop that follows.
    """
    C = M.shape[-1]
    if C <= block:
        return _inv_squaring(M)
    assert C % block == 0, f"chunk {C} must be a multiple of inverse block {block}"
    nb = C // block

    blk = lambda i, j: M[..., i*block:(i+1)*block, j*block:(j+1)*block]
    diag = [_inv_squaring(blk(i, i)) for i in range(nb)]

    X = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        X[i][i] = diag[i]
        for j in range(i):
            acc = blk(i, j) @ X[j][j]
            for k in range(j + 1, i):
                acc = acc + blk(i, k) @ X[k][j]
            X[i][j] = -diag[i] @ acc

    zero = torch.zeros_like(diag[0])
    return torch.cat([torch.cat([X[i][j] if j <= i else zero for j in range(nb)], dim=-1)
                      for i in range(nb)], dim=-2)


class KDA(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        h, d = c.n_heads, c.kda_head_dim
        assert c.kda_chunk % _INV_BLOCK == 0 or c.kda_chunk < _INV_BLOCK, (
            f"kda_chunk={c.kda_chunk} must be a multiple of the inverse block "
            f"{_INV_BLOCK} so blocked forward substitution stays exact")
        self.h, self.d, self.chunk, self.eps = h, d, c.kda_chunk, c.eps
        self.q = nn.Linear(c.d_model, h * d, bias=False)
        self.k = nn.Linear(c.d_model, h * d, bias=False)
        self.v = nn.Linear(c.d_model, h * d, bias=False)
        self.o = nn.Linear(h * d, c.d_model, bias=False)
        self.g_down = nn.Linear(c.d_model, 16, bias=False)    # low-rank decay gate
        self.g_up = nn.Linear(16, h * d, bias=True)
        self.beta = nn.Linear(c.d_model, h, bias=True)
        self.A_log = nn.Parameter(torch.empty(h * d))
        # 6 QK-RMSNorm: mandatory, these weights are reused across up to c_max
        # recurrences and that is exactly when attention logits drift upward.
        self.qn = nn.RMSNorm(d, eps=c.eps)
        self.kn = nn.RMSNorm(d, eps=c.eps)
        self.on = nn.RMSNorm(d, eps=c.eps)
        # Diagnostic; never trained or saved.  Device tensor, not float, so reading
        # it costs no host sync.  Now a MODELLING signal, not a numerical one: the
        # gate bound keeps the arithmetic exact, so |gcum| only reports how much
        # state KDA drops per chunk (retention/token = exp(-gcum/chunk)).
        self._max_gcum = torch.zeros(())
        self.reset_gate()

    def reset_gate(self):
        """log a = -exp(A_log) * softplus(g).  These inits give a ~ 0.96, so a
        64-step chunk accumulates only ~-2.6 of log-decay, well inside _G_RANGE."""
        with torch.no_grad():
            self.A_log.uniform_(math.log(1.0), math.log(16.0))
            self.g_up.bias.fill_(-5.0)
            self.beta.bias.fill_(0.0)

    def _project(self, x):
        B, T, _ = x.shape
        H, D = self.h, self.d
        q = self.qn(self.q(x).view(B, T, H, D))
        k = F.normalize(self.kn(self.k(x).view(B, T, H, D)), dim=-1)  # delta rule wants unit keys
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        g = F.softplus(self.g_up(self.g_down(x))).view(B, T, H, D)
        # Floor bounds the PER-CHUNK total: gcum is a sum over `chunk` steps, so a
        # per-step clamp of _G_RANGE (the old code) was `chunk`x too loose and let
        # a real run reach 23.3.  Cost: retention/token >= exp(-_G_RANGE/chunk)
        # = 0.855 @ 64; still permits ~total forgetting across a chunk (4.5e-5).
        floor = -_G_RANGE / self.chunk
        log_a = (-self.A_log.exp().view(1, 1, H, D) * g).clamp(min=floor)
        return q, k, v, beta, log_a

    def _out(self, o, x, B, T):
        o = self.on(o).transpose(1, 2).reshape(B, T, self.h * self.d)
        return self.o(o.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.d
        C = min(self.chunk, T)
        q, k, v, beta, log_a = self._project(x)

        pad = (-T) % C
        if pad:
            q, k, v, beta, log_a = (F.pad(t, (0, 0, 0, 0, 0, pad))
                                    for t in (q, k, v, beta, log_a))
        Tp, N = T + pad, (T + pad) // C

        hp = torch.promote_types(x.dtype, torch.float32)   # chunk math never below fp32
        shp = lambda t: t.transpose(1, 2).reshape(B, H, N, C, -1).to(hp)
        q, k, v, beta, log_a = shp(q), shp(k), shp(v), shp(beta), shp(log_a)

        # promote_types alone is insufficient: autocast re-casts every matmul back
        # to fp16 AFTER the promotion.  Centred gate factors reach exp(+-10)=22026,
        # their product 4.9e8 vs fp16's 65504 -> backward produced inf, GradScaler
        # halved forever (32768->128 over 9 steps), loss frozen.  Forward stayed
        # finite, which hid it.  Only the scan is pinned; _out stays under autocast.
        with torch.autocast(x.device.type, enabled=False):
            o = self._chunk_scan(q, k, v, beta, log_a, B, H, D, N, C, T, Tp)
        return self._out(o, x, B, T)

    def _chunk_scan(self, q, k, v, beta, log_a, B, H, D, N, C, T, Tp):

        gcum = log_a.cumsum(-2)
        g_end = gcum[..., -1:, :]                       # total decay across the chunk
        self._max_gcum = g_end.detach().abs().amax()    # see __init__

        # No clamp needed: log_a <= 0 => gcum monotone decreasing => both <= 0.
        eg = gcum.exp()                                 # read state at chunk start
        eg_end = (g_end - gcum).exp()                   # write state at chunk end

        # The one term needing a two-factor split.  Centred on the chunk midpoint:
        # exact (the constant cancels in the product) and halves the range.
        # _project bounds gcum to keep this inside the accurate range -- _G_RANGE.
        gc = (gcum - 0.5 * (gcum[..., :1, :] + g_end)).clamp(-_G_RANGE, _G_RANGE)
        eg_c, emg_c = gc.exp(), (-gc).exp()

        k_abs, k_end, q_abs = k * eg, k * eg_end, q * eg
        k_bar_c, k_til_c, q_c = k * eg_c, k * emg_c, q * eg_c

        Tm = inv_unit_lower(((beta * k_bar_c) @ k_til_c.transpose(-1, -2)).tril(-1))

        S = q.new_zeros(B, H, D, D)
        outs = []
        for n in range(N):
            ka, ke, qa, qc, kt, bt, vt, Tn = (
                t[:, :, n] for t in (k_abs, k_end, q_abs, q_c, k_til_c, beta, v, Tm))
            u = Tn @ (bt * (vt - ka @ S))
            outs.append(qa @ S + (qc @ kt.transpose(-1, -2)).tril(0) @ u)
            S = S * eg[:, :, n, -1].unsqueeze(-1) + ke.transpose(-1, -2) @ u

        return torch.stack(outs, 2).reshape(B, H, Tp, D)[:, :, :T]

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """Naive O(T) recurrence.  Test oracle for the chunked path, nothing else."""
        B, T, _ = x.shape
        hp = torch.promote_types(x.dtype, torch.float32)
        q, k, v, beta, log_a = (t.to(hp) for t in self._project(x))
        a = log_a.exp()

        S = q.new_zeros(B, self.h, self.d, self.d)
        outs = []
        for t in range(T):
            S = S * a[:, t].unsqueeze(-1)
            kt, vt, bt = k[:, t].unsqueeze(-1), v[:, t].unsqueeze(-2), beta[:, t].unsqueeze(-1)
            S = S + bt * (kt @ (vt - kt.transpose(-1, -2) @ S))
            outs.append((q[:, t].unsqueeze(-2) @ S).squeeze(-2))
        return self._out(torch.stack(outs, 2), x, B, T)
