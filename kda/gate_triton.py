"""Triton kernel fusing attention.py's _chunk_scan pre-loop gate/key math -- the
actual measured bottleneck (profiling showed ~10 separate (B,H,N,C,D) tensors
materialized here: eg, eg_end, gc, eg_c, emg_c, k_abs, k_end, k_bar_c,
k_til_c, q_abs, q_c -- each its own PyTorch kernel launch + full memory
round-trip). This fuses computing the six tensors the scan actually needs
(k_abs, k_end, q_abs, k_bar_c, k_til_c, q_c) from (q, k, log_a) into one
kernel launch, keeping every intermediate in registers.

Pure elementwise + one cumsum (no matmuls, unlike scan_triton.py's scan
kernel) -- lower risk: elementwise backward is just local-derivative times
upstream gradient, and cumsum's backward is the standard reverse-cumsum
identity, not hand-derived matrix calculus. Still verified numerically
against the existing PyTorch computation before being trusted, same bar
as the scan kernel.

Forward, per (batch, head, chunk), C timesteps of D channels:

    gcum = cumsum(log_a)                      along C
    g_end = sum(log_a)                        (= gcum's last element)
    mid = 0.5*(log_a[0] + g_end)              (log_a[0] = gcum's first element)
    gc = clamp(gcum - mid, -G_RANGE, G_RANGE)
    eg, eg_end, eg_c, emg_c = exp(gcum), exp(g_end-gcum), exp(gc), exp(-gc)
    k_abs, k_end, q_abs   = k*eg, k*eg_end, q*eg
    k_bar_c, k_til_c, q_c = k*eg_c, k*emg_c, q*eg_c
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_G_RANGE = 10.0


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _gate_fwd_kernel(
    q_ptr, k_ptr, loga_ptr,
    kabs_ptr, kend_ptr, qabs_ptr, kbar_ptr, ktil_ptr, qc_ptr,
    C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    stride_g_c, stride_g_d,   # strides within one (C,D) chunk (shared by all tensors)
    G_RANGE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * stride_g_c * C  # each program owns a contiguous (C,D) block

    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    ptrs = c_idx[:, None] * stride_g_c + d_idx[None, :] * stride_g_d

    q = tl.load(q_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    k = tl.load(k_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    log_a = tl.load(loga_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)

    gcum = tl.cumsum(log_a, axis=0)
    g_end = tl.sum(log_a, axis=0)
    first_row = tl.sum(tl.where(c_idx[:, None] == 0, log_a, 0.0), axis=0)
    mid = 0.5 * (first_row + g_end)

    gc_pre = gcum - mid[None, :]
    gc = tl.minimum(tl.maximum(gc_pre, -G_RANGE), G_RANGE)

    eg = tl.exp(gcum)
    eg_end = tl.exp(g_end[None, :] - gcum)
    eg_c = tl.exp(gc)
    emg_c = tl.exp(-gc)

    tl.store(kabs_ptr + base + ptrs, k * eg, mask=d_mask[None, :])
    tl.store(kend_ptr + base + ptrs, k * eg_end, mask=d_mask[None, :])
    tl.store(qabs_ptr + base + ptrs, q * eg, mask=d_mask[None, :])
    tl.store(kbar_ptr + base + ptrs, k * eg_c, mask=d_mask[None, :])
    tl.store(ktil_ptr + base + ptrs, k * emg_c, mask=d_mask[None, :])
    tl.store(qc_ptr + base + ptrs, q * eg_c, mask=d_mask[None, :])


@triton.jit
def _gate_bwd_kernel(
    q_ptr, k_ptr, loga_ptr,
    dkabs_ptr, dkend_ptr, dqabs_ptr, dkbar_ptr, dktil_ptr, dqc_ptr,
    dq_ptr, dk_ptr, dloga_ptr,
    C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    stride_g_c, stride_g_d,
    G_RANGE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * stride_g_c * C

    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    ptrs = c_idx[:, None] * stride_g_c + d_idx[None, :] * stride_g_d

    q = tl.load(q_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    k = tl.load(k_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    log_a = tl.load(loga_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)

    # ---- recompute forward ----
    gcum = tl.cumsum(log_a, axis=0)
    g_end = tl.sum(log_a, axis=0)
    first_row = tl.sum(tl.where(c_idx[:, None] == 0, log_a, 0.0), axis=0)
    mid = 0.5 * (first_row + g_end)
    gc_pre = gcum - mid[None, :]
    clamp_mask = (gc_pre >= -G_RANGE) & (gc_pre <= G_RANGE)
    gc = tl.minimum(tl.maximum(gc_pre, -G_RANGE), G_RANGE)
    eg = tl.exp(gcum)
    eg_end = tl.exp(g_end[None, :] - gcum)
    eg_c = tl.exp(gc)
    emg_c = tl.exp(-gc)

    d_kabs = tl.load(dkabs_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    d_kend = tl.load(dkend_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    d_qabs = tl.load(dqabs_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    d_kbar = tl.load(dkbar_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    d_ktil = tl.load(dktil_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)
    d_qc = tl.load(dqc_ptr + base + ptrs, mask=d_mask[None, :], other=0.0)

    # ---- adjoint ----
    d_k = d_kabs * eg + d_kend * eg_end + d_kbar * eg_c + d_ktil * emg_c
    d_q = d_qabs * eg + d_qc * eg_c
    tl.store(dq_ptr + base + ptrs, d_q, mask=d_mask[None, :])
    tl.store(dk_ptr + base + ptrs, d_k, mask=d_mask[None, :])

    d_eg = d_kabs * k + d_qabs * q
    d_eg_end = d_kend * k
    d_eg_c = d_kbar * k + d_qc * q
    d_emg_c = d_ktil * k

    d_gc = (d_eg_c * eg_c - d_emg_c * emg_c) * tl.where(clamp_mask, 1.0, 0.0)

    d_mid = -tl.sum(d_gc, axis=0)                       # (BD,)
    d_gcum = d_gc                                        # direct part, from gc_pre = gcum - mid
    d_gcum += d_eg * eg                                  # from eg = exp(gcum)
    d_gcum += -d_eg_end * eg_end                         # from eg_end = exp(g_end - gcum)

    d_g_end = tl.sum(d_eg_end * eg_end, axis=0)          # from eg_end's other arg
    d_g_end += 0.5 * d_mid                               # from mid = 0.5*(first_row+g_end)
    d_first_row = 0.5 * d_mid                            # from mid

    # gcum = cumsum(log_a): d_log_a[c] = sum_{c'>=c} d_gcum[c'] = reverse-cumsum(d_gcum)
    total = tl.sum(d_gcum, axis=0)
    cs = tl.cumsum(d_gcum, axis=0)
    d_loga = total[None, :] - cs + d_gcum                # reverse-cumsum identity
    d_loga += d_g_end[None, :]                           # g_end = sum(log_a): every c contributes
    d_loga += tl.where(c_idx[:, None] == 0, d_first_row[None, :], 0.0)  # first_row = log_a[0]

    tl.store(dloga_ptr + base + ptrs, d_loga, mask=d_mask[None, :])


class _KDAGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, log_a):
        shape = q.shape  # (..., C, D)
        C, D = shape[-2], shape[-1]
        G = q.numel() // (C * D)
        BD = _next_pow2(D)
        dev, dt = q.device, q.dtype

        qf = q.reshape(G, C, D).contiguous().float()
        kf = k.reshape(G, C, D).contiguous().float()
        lf = log_a.reshape(G, C, D).contiguous().float()

        outs = [torch.empty(G, C, D, device=dev, dtype=torch.float32) for _ in range(6)]
        grid = (G,)
        _gate_fwd_kernel[grid](
            qf, kf, lf, *outs,
            C, D, BD,
            qf.stride(1), qf.stride(2),
            _G_RANGE,
        )
        ctx.save_for_backward(qf, kf, lf)
        ctx.dims = (shape, C, D, G, BD)
        ctx.dt = dt
        return tuple(o.reshape(shape).to(dt) for o in outs)

    @staticmethod
    def backward(ctx, *grads):
        qf, kf, lf = ctx.saved_tensors
        shape, C, D, G, BD = ctx.dims
        grads_f = [g.reshape(G, C, D).contiguous().float() for g in grads]

        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dloga = torch.zeros_like(lf)
        grid = (G,)
        _gate_bwd_kernel[grid](
            qf, kf, lf, *grads_f, dq, dk, dloga,
            C, D, BD,
            qf.stride(1), qf.stride(2),
            _G_RANGE,
        )
        r = lambda t: t.reshape(shape).to(ctx.dt)
        return r(dq), r(dk), r(dloga)


def kda_gate(q: torch.Tensor, k: torch.Tensor, log_a: torch.Tensor):
    """Returns (k_abs, k_end, q_abs, k_bar_c, k_til_c, q_c), replacing
    attention.py's _chunk_scan pre-loop elementwise section."""
    return _KDAGate.apply(q, k, log_a)
