"""Triton kernel for every decay-dependent factor of one KDA chunk (Kimi Linear,
arXiv:2510.26692, Sec. 3.1 Eqs. 6-9 and App. C Listing 1).

Per (batch, head, chunk), with C positions r, i and D channels c, G = cumsum(log_alpha):

    q_abs  = q * exp(G)                    k_abs = k * exp(G)
    k_end  = k * exp(G_C - G)              eg_last = exp(G_C)
    Aqk[r,i] = sum_c q[r,c] k[i,c] exp(G[r,c] - G[i,c])    for i <= r, else 0
    Akk[r,i] = sum_c k[r,c] k[i,c] exp(G[r,c] - G[i,c])    for i <  r, else 0

Every exponent is <= 0, so every factor is in [0, 1] for any gate value.
Aqk and Akk subtract inside the exp, as Listing 1 does ("secondary chunking for
numerical stability"). A split exp(G_r) * exp(-G_i) overflows once a chunk's
log-decay span passes about 88 in fp32.
The [C, C] tiles are built one channel at a time in registers, so the
[C, C, D] tensor of Listing 1 is never stored.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _column(tile, d_idx, d):
    return tl.sum(tl.where(d_idx[None, :] == d, tile, 0.0), axis=1)


@triton.jit
def _intra_fwd_kernel(
    q_ptr, k_ptr, loga_ptr,
    qabs_ptr, kabs_ptr, kend_ptr, egl_ptr, aqk_ptr, akk_ptr,
    C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    cd_base = chunk * C * D
    cc_base = chunk * C * C

    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    cd_ptrs = c_idx[:, None] * D + d_idx[None, :]
    cc_ptrs = c_idx[:, None] * C + c_idx[None, :]
    cd_mask = d_mask[None, :]

    q = tl.load(q_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    k = tl.load(k_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    log_a = tl.load(loga_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)

    gcum = tl.cumsum(log_a, axis=0)
    g_end = tl.sum(log_a, axis=0)
    eg = tl.exp(gcum)
    eg_end = tl.exp(g_end[None, :] - gcum)

    tl.store(qabs_ptr + cd_base + cd_ptrs, q * eg, mask=cd_mask)
    tl.store(kabs_ptr + cd_base + cd_ptrs, k * eg, mask=cd_mask)
    tl.store(kend_ptr + cd_base + cd_ptrs, k * eg_end, mask=cd_mask)
    tl.store(egl_ptr + chunk * D + d_idx, tl.exp(g_end), mask=d_mask)

    causal = c_idx[:, None] >= c_idx[None, :]
    aqk = tl.zeros((C, C), dtype=tl.float32)
    akk = tl.zeros((C, C), dtype=tl.float32)
    for d in tl.static_range(D):
        q_d = _column(q, d_idx, d)
        k_d = _column(k, d_idx, d)
        g_d = _column(gcum, d_idx, d)
        decay = tl.exp(tl.where(causal, g_d[:, None] - g_d[None, :], float("-inf")))
        k_decayed = k_d[None, :] * decay
        aqk += q_d[:, None] * k_decayed
        akk += k_d[:, None] * k_decayed
    akk = tl.where(c_idx[:, None] > c_idx[None, :], akk, 0.0)

    tl.store(aqk_ptr + cc_base + cc_ptrs, aqk)
    tl.store(akk_ptr + cc_base + cc_ptrs, akk)


@triton.jit
def _intra_bwd_kernel(
    q_ptr, k_ptr, loga_ptr,
    dqabs_ptr, dkabs_ptr, dkend_ptr, degl_ptr, daqk_ptr, dakk_ptr,
    dq_ptr, dk_ptr, dloga_ptr,
    C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    cd_base = chunk * C * D
    cc_base = chunk * C * C

    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    cd_ptrs = c_idx[:, None] * D + d_idx[None, :]
    cc_ptrs = c_idx[:, None] * C + c_idx[None, :]
    cd_mask = d_mask[None, :]

    q = tl.load(q_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    k = tl.load(k_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    log_a = tl.load(loga_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    gcum = tl.cumsum(log_a, axis=0)
    g_end = tl.sum(log_a, axis=0)
    eg = tl.exp(gcum)
    eg_end = tl.exp(g_end[None, :] - gcum)

    d_qabs = tl.load(dqabs_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    d_kabs = tl.load(dkabs_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    d_kend = tl.load(dkend_ptr + cd_base + cd_ptrs, mask=cd_mask, other=0.0)
    d_egl = tl.load(degl_ptr + chunk * D + d_idx, mask=d_mask, other=0.0)

    d_q = d_qabs * eg
    d_k = d_kabs * eg + d_kend * eg_end
    d_gcum = (d_qabs * q + d_kabs * k) * eg - d_kend * k * eg_end
    d_g_end = tl.sum(d_kend * k * eg_end, axis=0) + d_egl * tl.exp(g_end)

    causal = c_idx[:, None] >= c_idx[None, :]
    d_aqk = tl.where(causal, tl.load(daqk_ptr + cc_base + cc_ptrs), 0.0)
    d_akk = tl.where(c_idx[:, None] > c_idx[None, :], tl.load(dakk_ptr + cc_base + cc_ptrs), 0.0)
    for d in tl.static_range(D):
        q_d = _column(q, d_idx, d)
        k_d = _column(k, d_idx, d)
        g_d = _column(gcum, d_idx, d)
        decay = tl.exp(tl.where(causal, g_d[:, None] - g_d[None, :], float("-inf")))
        p_qk = d_aqk * decay
        p_kk = d_akk * decay
        # k takes the column role in Aqk and Akk, and the row role in Akk.
        dq_d = tl.sum(p_qk * k_d[None, :], axis=1)
        dk_d = (tl.sum(p_qk * q_d[:, None], axis=0) + tl.sum(p_kk * k_d[:, None], axis=0)
                + tl.sum(p_kk * k_d[None, :], axis=1))
        # d/dG of exp(G_r - G_i) is +1 for the row r and -1 for the column i.
        dexp = p_qk * (q_d[:, None] * k_d[None, :]) + p_kk * (k_d[:, None] * k_d[None, :])
        dg_d = tl.sum(dexp, axis=1) - tl.sum(dexp, axis=0)
        here = d_idx[None, :] == d
        d_q += tl.where(here, dq_d[:, None], 0.0)
        d_k += tl.where(here, dk_d[:, None], 0.0)
        d_gcum += tl.where(here, dg_d[:, None], 0.0)

    # G = cumsum(log_a), so d log_a is the reverse cumsum of dG; G_C adds to every position.
    d_loga = tl.cumsum(d_gcum, axis=0, reverse=True) + d_g_end[None, :]

    tl.store(dq_ptr + cd_base + cd_ptrs, d_q, mask=cd_mask)
    tl.store(dk_ptr + cd_base + cd_ptrs, d_k, mask=cd_mask)
    tl.store(dloga_ptr + cd_base + cd_ptrs, d_loga, mask=cd_mask)


_NUM_WARPS = 4


class _KDAIntra(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, log_a):
        *lead, C, D = q.shape
        chunks = q.numel() // (C * D)
        BD = _next_pow2(D)
        qf, kf, lf = (t.reshape(chunks, C, D).contiguous().float() for t in (q, k, log_a))
        cd = [torch.empty(chunks, C, D, device=q.device, dtype=torch.float32) for _ in range(3)]
        egl = torch.empty(chunks, D, device=q.device, dtype=torch.float32)
        cc = [torch.empty(chunks, C, C, device=q.device, dtype=torch.float32) for _ in range(2)]
        _intra_fwd_kernel[(chunks,)](qf, kf, lf, *cd, egl, *cc, C, D, BD, num_warps=_NUM_WARPS)
        ctx.save_for_backward(qf, kf, lf)
        ctx.dims = (tuple(lead), C, D, chunks, BD, q.dtype)
        q_abs, k_abs, k_end = (t.reshape(*lead, C, D).to(q.dtype) for t in cd)
        aqk, akk = (t.reshape(*lead, C, C).to(q.dtype) for t in cc)
        return q_abs, k_abs, k_end, egl.reshape(*lead, D).to(q.dtype), aqk, akk

    @staticmethod
    def backward(ctx, d_qabs, d_kabs, d_kend, d_egl, d_aqk, d_akk):
        qf, kf, lf = ctx.saved_tensors
        lead, C, D, chunks, BD, dtype = ctx.dims

        def flat(grad, *shape):
            if grad is None:
                return torch.zeros(chunks, *shape, device=qf.device, dtype=torch.float32)
            return grad.reshape(chunks, *shape).contiguous().float()

        grads = (flat(d_qabs, C, D), flat(d_kabs, C, D), flat(d_kend, C, D), flat(d_egl, D),
                 flat(d_aqk, C, C), flat(d_akk, C, C))
        dq, dk, dloga = (torch.empty_like(qf) for _ in range(3))
        _intra_bwd_kernel[(chunks,)](qf, kf, lf, *grads, dq, dk, dloga, C, D, BD, num_warps=_NUM_WARPS)
        return tuple(t.reshape(*lead, C, D).to(dtype) for t in (dq, dk, dloga))


def kda_intra_reference(q, k, log_a):
    """The same outputs as kda_intra in plain PyTorch, with the [C, C, D] tensor stored; test oracle."""
    C = q.shape[-2]
    gcum = log_a.cumsum(-2)
    g_end = gcum[..., -1:, :]
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    exponent = (gcum.unsqueeze(-2) - gcum.unsqueeze(-3)).masked_fill(~causal[..., None], float("-inf"))
    decay = exponent.exp()
    aqk = torch.einsum("...rc,...ic,...ric->...ri", q, k, decay)
    akk = torch.einsum("...rc,...ic,...ric->...ri", k, k, decay).tril(-1)
    return (q * gcum.exp(), k * gcum.exp(), k * (g_end - gcum).exp(), g_end.squeeze(-2).exp(), aqk, akk)


def kda_intra(q: torch.Tensor, k: torch.Tensor, log_a: torch.Tensor):
    """(q_abs, k_abs, k_end, eg_last, Aqk, Akk) for inputs of shape (..., C, D); see the module docstring."""
    if not q.is_cuda:
        return kda_intra_reference(q, k, log_a)
    return _KDAIntra.apply(q, k, log_a)
