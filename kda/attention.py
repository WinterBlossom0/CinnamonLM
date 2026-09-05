"""KDA - Kimi Delta Attention: gated delta rule with channelwise decay
(Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture",
arXiv:2510.26692, eq. 1).

IMPORTANT: "Kimi Linear" the paper's actual architecture is a hybrid that
interleaves KDA layers with full Multi-Head Latent Attention (MLA) layers at
a 3:1 ratio -- the paper's own ablations found pure-KDA-every-layer
"fundamentally constrained by finite-state capacity" for long-range
retrieval, with MLA layers carrying that role. This module, and this
project's "kda" arm, is pure KDA in every layer, no MLA -- a deliberate,
narrower ablation (isolate the KDA layer type itself) that is NOT the paper's
proposed architecture. Don't describe this as "Kimi Linear."

Adapted from CinnamonLM's own cinnamon/kda.py (commit 3255ef8, since deleted
from the main project when it moved to a different architecture) -- ported
here standalone (explicit constructor args instead of a shared Config
object) so attention_comparison/ stays self-contained. The recurrence and
every numerical-stability bound below carry over unchanged -- they were
debugged against real training blow-ups, not written from a paper read.
`inv_unit_lower` is the one rewritten piece (see its docstring).

Simplified from the paper's q/k/v/gate pipeline at this scale (paper eq. in
Sec. 4 "Neural Parameterization"; verified by reading the full paper, not
just the recurrence equation):
  - paper: q,k = L2Norm(Swish(ShortConv(Wx))); this: q = RMSNorm(Wx),
    k = L2Norm(RMSNorm(Wx)) -- no ShortConv, no Swish on either.
  - paper: v = Swish(ShortConv(Wx)); this: v = Wx -- no ShortConv, no Swish.
  - paper: output gate o_t = W_o(Sigmoid(W_g-up W_g-down x) * RMSNorm(KDA_out))
    before the final projection; this has no output gate, just
    W_o(RMSNorm(KDA_out)).
  - paper: gate rank = head_dim exactly (their 48B model: rank=128=head_dim);
    this: gate_rank defaults to 16 regardless of head_dim (see __init__).

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

from .scan_triton import kda_scan
from .gate_triton import kda_gate, _G_RANGE
from .rmsnorm_triton import TritonRMSNorm

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
# A Triton port removes the bound. Value lives in gate_triton.py (imported
# above) since that kernel needs it too -- one definition, not two to keep in sync.


def inv_unit_lower(M: torch.Tensor) -> torch.Tensor:
    """(I + M)^-1 for strictly lower-triangular M.  unitriangular=True reads M's
    diagonal as 1, so M goes in as-is; cuBLAS trsm does the substitution, batched
    over (B, H, N).  Upper-triangle and diagonal gradients are discarded by
    solve_triangular's own backward, which is what this needs (X is structurally
    unit lower triangular, so those entries are not free).

    Replaced a hand-blocked substitution (nested Python loops over 16x16 slices,
    diagonal inverses by repeated squaring).  That form was BOTH slower and less
    accurate: ~44 tiny batched matmuls plus 14 slices of a (B,H,N,64,64) tensor,
    whose SliceBackward0 zero-filled a full-size grad per slice -- measured on the
    real training step at 76 ms/step forward + 111 ms/step of slice-backward, and
    at chunk 64 it drifted to 1.1e-03 relative on correlated keys (vs 4.4e-07
    here) because repeated squaring is only conditionally stable.
    """
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    return torch.linalg.solve_triangular(M, I, upper=False, unitriangular=True)


class KDA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6, gate_rank: int = 16):
        super().__init__()
        h, d = n_heads, head_dim
        self.h, self.d, self.chunk, self.eps = h, d, chunk, eps
        self.q = nn.Linear(d_model, h * d, bias=False)
        self.k = nn.Linear(d_model, h * d, bias=False)
        self.v = nn.Linear(d_model, h * d, bias=False)
        self.o = nn.Linear(h * d, d_model, bias=False)
        # decay gate: low-rank only when gate_rank < d_model (16 < 16 does
        # not hold at this scale's default, so this is a full-rank map here)
        self.g_down = nn.Linear(d_model, gate_rank, bias=False)
        self.g_up = nn.Linear(gate_rank, h * d, bias=True)
        self.beta = nn.Linear(d_model, h, bias=True)
        self.A_log = nn.Parameter(torch.empty(h * d))
        # QK-RMSNorm: mandatory, these weights are reused at every step of the
        # recurrence and that is exactly when attention logits drift upward.
        # TritonRMSNorm (rmsnorm_triton.py) replaces nn.RMSNorm here: at
        # head_dim=3, PyTorch's fused RMSNorm kernel over 655K independent
        # 3-element groups was the actual measured bottleneck (57% of total
        # time, profiling) -- verified against nn.RMSNorm, fwd and bwd, to
        # float32 precision at every trained head_dim first.
        self.qn = TritonRMSNorm(d, eps=eps)
        self.kn = TritonRMSNorm(d, eps=eps)
        self.on = TritonRMSNorm(d, eps=eps)
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
        assert C & (C - 1) == 0, f"chunk size {C} must be a power of two (tl.arange requires it)"
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

        g_end = log_a.sum(-2, keepdim=True)              # total decay across the chunk
        self._max_gcum = g_end.detach().abs().amax()    # see __init__

        # Fused Triton kernel (gate_triton.py) for what used to be here:
        # cumsum + exp/clamp + 6 elementwise products, ~10 separate
        # (B,H,N,C,D) tensors in plain PyTorch -- the actual measured
        # bottleneck (profiling), fused into one kernel launch. Verified
        # against this exact pre-loop math, fwd and bwd (including at
        # _project()'s real floor, the true worst case for the _G_RANGE
        # clamp) to float32 precision at every trained head_dim.
        k_abs, k_end, q_abs, k_bar_c, k_til_c, q_c = kda_gate(q, k, log_a)

        Tm = inv_unit_lower(((beta * k_bar_c) @ k_til_c.transpose(-1, -2)).tril(-1))

        # The sequential part (S_n -> S_{n+1} across chunks) as a Triton kernel
        # with a hand-derived backward pass (scan_triton.py) -- verified fwd and
        # bwd against the equivalent PyTorch chunk loop to float32 machine
        # precision at every trained head_dim before being trusted here.
        eg_last = g_end.squeeze(-2).exp()   # gcum's last entry per chunk == g_end (total decay)
        outs = kda_scan(k_abs, k_end, q_abs, q_c, k_til_c, beta, v, Tm, eg_last)

        return outs.reshape(B, H, Tp, D)[:, :, :T]

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
