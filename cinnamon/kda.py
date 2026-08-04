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

# Every decay factor the recurrence actually needs has a non-positive exponent,
# so it is bounded by 1 and cannot overflow.  Only the *pairwise* intra-chunk term
# exp(g_t - g_s) has to be split into two separate matmul factors -- and because
# the decay is channelwise, that split cannot be folded into a [C, C] mask the way
# a scalar decay could.  Splitting is what manufactures a large exponent out of a
# bounded quantity.
#
# Centering fixes half the problem exactly: the two factors only ever appear as a
# product, so subtracting any per-chunk constant cancels.  Centering on the chunk
# midpoint puts the range at +-R/2 instead of 0..R, which doubles the chunk length
# that survives fp32 at the same clamp.
#
# Measured %-of-gcum hitting the old floor (and resulting error vs the exact
# recurrence) before this change:
#     chunk  16 -> 0.0%  3.1e-07     chunk  64 -> 0.0%  3.3e-07
#     chunk 128 -> 1.4%  6.3e-02  <- 1.4% clamped was enough to destroy it
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
        # Diagnostic only, never trained or saved.  Kept as a device tensor rather
        # than a float so reading it costs no host sync on the hot path.  Measured
        # threshold: the chunked path matches the exact recurrence while per-chunk
        # |gcum| stays under ~_G_RANGE, and silently degrades past it (1e-1 relative
        # error at 9.5, 4e-1 at 18) without ever producing a NaN to trip on.
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
        log_a = (-self.A_log.exp().view(1, 1, H, D) * g).clamp(min=-_G_RANGE)
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

        # promote_types alone is NOT enough: autocast intercepts every matmul
        # *after* the cast and puts it back to fp16, which silently undoes the
        # promotion this whole routine depends on.  The centred gate factors reach
        # exp(+-10) = 22026, and a product of two of them is 4.9e8 against fp16's
        # 65504 ceiling -- so under autocast the backward pass produced inf, the
        # GradScaler halved forever (32768 -> 128 over 9 steps) and the loss never
        # moved.  Forward stayed finite, which is what made it hard to see.
        # Only the scan is forced to fp32; the output projection stays under
        # autocast, where a Linear is exactly what fp16 is good at.
        with torch.autocast(x.device.type, enabled=False):
            o = self._chunk_scan(q, k, v, beta, log_a, B, H, D, N, C, T, Tp)
        return self._out(o, x, B, T)

    def _chunk_scan(self, q, k, v, beta, log_a, B, H, D, N, C, T, Tp):

        gcum = log_a.cumsum(-2)
        g_end = gcum[..., -1:, :]                       # total decay across the chunk
        self._max_gcum = g_end.detach().abs().amax()    # see __init__: drift watchdog

        # Bounded exactly, no clamp possible or needed: log_a <= 0 makes gcum
        # monotonically decreasing, so both exponents below are <= 0.
        eg = gcum.exp()                                 # read state at chunk start
        eg_end = (g_end - gcum).exp()                   # write state at chunk end

        # The one term that must be split into two factors.  Centred on the chunk
        # midpoint, which is exact (cancels in the product) and halves the range.
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
