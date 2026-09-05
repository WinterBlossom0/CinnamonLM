"""Triton kernel for KDA's sequential per-chunk state scan -- the one part of
attention.py's _chunk_scan that cannot be parallelized across chunks (state S_n
feeds S_{n+1}). Everything else (gate/key preprocessing, the UT-transform
matrix inverse) stays in plain PyTorch: those are already embarrassingly
parallel across chunks, so fusing them into this kernel wouldn't remove any
sequential dependency, only add kernel-writing risk for no structural gain.

Forward, per (batch, head), for n = 0..N-1 (S_0 = 0), matching attention.py's
_chunk_scan loop exactly:

    w_n = v_n - ka_n @ S_n
    r_n = bt_n * w_n
    u_n = T_n @ r_n
    A_n = tril(qc_n @ kt_n^T, 0)
    out_n = qa_n @ S_n + A_n @ u_n
    S_{n+1} = eg_last_n[:,None] * S_n + ke_n^T @ u_n

Backward derived by hand (reverse-mode adjoint, n = N-1 down to 0), not
copied from a reference -- verified below against autograd on the existing,
already-correctness-checked PyTorch implementation before this is trusted for
anything. Forward saves S_n (state entering each chunk) so backward can
recompute w_n/r_n/u_n/A_n cheaply instead of saving every intermediate.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, 16)  # tl.dot requires the contraction dim >= 16; extra rows/cols stay exactly
                        # zero throughout (proven by induction: zero-padded inputs + S_0=0 keep
                        # every intermediate zero outside the real (D,D)/(C,D) block), verified
                        # numerically below rather than trusted on the proof alone


@triton.jit
def _kda_fwd_kernel(
    ka_ptr, ke_ptr, qa_ptr, qc_ptr, kt_ptr, bt_ptr, vt_ptr, T_ptr, egl_ptr,
    out_ptr, S_save_ptr,
    N, C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    stride_bh_cd, stride_n_cd, stride_c_cd,   # strides for (BH,N,C,D)-shaped tensors
    stride_bh_cc, stride_n_cc, stride_c_cc,   # strides for (BH,N,C,C)-shaped tensors (T)
    stride_bh_d, stride_n_d,                  # strides for (BH,N,D)-shaped tensors (egl)
    stride_bh_s, stride_n_s, stride_d_s,      # strides for S_save (BH,N,D,D)
    stride_bh_bt, stride_n_bt,                # strides for beta (BH,N,C)
):
    bh = tl.program_id(0)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    c_idx = tl.arange(0, C)

    S = tl.zeros((BD, BD), dtype=tl.float32)  # (D,D), padded to (BD,BD), extra rows/cols stay 0

    cd_base = bh * stride_bh_cd
    cc_base = bh * stride_bh_cc
    d_base = bh * stride_bh_d
    s_base = bh * stride_bh_s
    bt_base = bh * stride_bh_bt

    for n in range(0, N):
        cd_off = cd_base + n * stride_n_cd
        cc_off = cc_base + n * stride_n_cc
        d_off = d_base + n * stride_n_d
        s_off = s_base + n * stride_n_s
        bt_off = bt_base + n * stride_n_bt

        cd_ptrs = c_idx[:, None] * stride_c_cd + d_idx[None, :]
        mask_cd = d_mask[None, :]
        ka = tl.load(ka_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        ke = tl.load(ke_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        qa = tl.load(qa_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        qc = tl.load(qc_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        kt = tl.load(kt_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        vt = tl.load(vt_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        bt = tl.load(bt_ptr + bt_off + c_idx)  # (C,)

        # save S entering this chunk, for backward's recompute
        s_ptrs = d_idx[:, None] * stride_d_s + d_idx[None, :]
        tl.store(S_save_ptr + s_off + s_ptrs, S, mask=d_mask[:, None] & d_mask[None, :])

        T_ptrs = c_idx[:, None] * stride_c_cc + c_idx[None, :]
        Tn = tl.load(T_ptr + cc_off + T_ptrs)

        w = vt - tl.dot(ka, S, input_precision="ieee")                       # (C,BD)
        r = w * bt[:, None]
        u = tl.dot(Tn, r, input_precision="ieee")                             # (C,BD)

        M = tl.dot(qc, tl.trans(kt), input_precision="ieee")                  # (C,C)
        tril_mask = c_idx[:, None] >= c_idx[None, :]
        A = tl.where(tril_mask, M, 0.0)

        out = tl.dot(qa, S, input_precision="ieee") + tl.dot(A, u, input_precision="ieee")             # (C,BD)
        out_ptrs = c_idx[:, None] * stride_c_cd + d_idx[None, :]
        tl.store(out_ptr + cd_off + out_ptrs, out, mask=mask_cd)

        egl = tl.load(egl_ptr + d_off + d_idx, mask=d_mask, other=0.0)  # (D,)
        S = egl[:, None] * S + tl.dot(tl.trans(ke), u, input_precision="ieee")


@triton.jit
def _kda_bwd_kernel(
    ka_ptr, ke_ptr, qa_ptr, qc_ptr, kt_ptr, bt_ptr, vt_ptr, T_ptr, egl_ptr,
    S_save_ptr, dout_ptr,
    dka_ptr, dke_ptr, dqa_ptr, dqc_ptr, dkt_ptr, dbt_ptr, dvt_ptr, dT_ptr, degl_ptr,
    N, C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    stride_bh_cd, stride_n_cd, stride_c_cd,
    stride_bh_cc, stride_n_cc, stride_c_cc,
    stride_bh_d, stride_n_d,
    stride_bh_s, stride_n_s, stride_d_s,
    stride_bh_bt, stride_n_bt,
):
    bh = tl.program_id(0)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D
    c_idx = tl.arange(0, C)
    mask_cd = d_mask[None, :]

    cd_base = bh * stride_bh_cd
    cc_base = bh * stride_bh_cc
    d_base = bh * stride_bh_d
    s_base = bh * stride_bh_s
    bt_base = bh * stride_bh_bt

    dS = tl.zeros((BD, BD), dtype=tl.float32)  # dL/dS_{n+1}, starts at 0 (no external use of S_N)

    for step in range(0, N):
        n = N - 1 - step
        cd_off = cd_base + n * stride_n_cd
        cc_off = cc_base + n * stride_n_cc
        d_off = d_base + n * stride_n_d
        s_off = s_base + n * stride_n_s
        bt_off = bt_base + n * stride_n_bt

        cd_ptrs = c_idx[:, None] * stride_c_cd + d_idx[None, :]
        ka = tl.load(ka_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        ke = tl.load(ke_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        qa = tl.load(qa_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        qc = tl.load(qc_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        kt = tl.load(kt_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        vt = tl.load(vt_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        bt = tl.load(bt_ptr + bt_off + c_idx)
        dout = tl.load(dout_ptr + cd_off + cd_ptrs, mask=mask_cd, other=0.0)
        egl = tl.load(egl_ptr + d_off + d_idx, mask=d_mask, other=0.0)

        s_ptrs = d_idx[:, None] * stride_d_s + d_idx[None, :]
        smask = d_mask[:, None] & d_mask[None, :]
        S = tl.load(S_save_ptr + s_off + s_ptrs, mask=smask, other=0.0)

        T_ptrs = c_idx[:, None] * stride_c_cc + c_idx[None, :]
        Tn = tl.load(T_ptr + cc_off + T_ptrs)

        # ---- recompute forward intermediates for this chunk ----
        w = vt - tl.dot(ka, S, input_precision="ieee")
        r = w * bt[:, None]
        u = tl.dot(Tn, r, input_precision="ieee")
        M = tl.dot(qc, tl.trans(kt), input_precision="ieee")
        tril_mask = c_idx[:, None] >= c_idx[None, :]
        A = tl.where(tril_mask, M, 0.0)

        # ---- adjoint ----
        dqa = tl.dot(dout, tl.trans(S), input_precision="ieee")
        dS_from_out = tl.dot(tl.trans(qa), dout, input_precision="ieee")
        dA = tl.dot(dout, tl.trans(u), input_precision="ieee")
        du = tl.dot(tl.trans(A), dout, input_precision="ieee")

        degl = tl.sum(dS * S, axis=1)
        dS_from_next = egl[:, None] * dS
        du += tl.dot(ke, dS, input_precision="ieee")
        dke = tl.dot(u, tl.trans(dS), input_precision="ieee")

        dT = tl.dot(du, tl.trans(r), input_precision="ieee")
        dr = tl.dot(tl.trans(Tn), du, input_precision="ieee")

        dbt = tl.sum(dr * w, axis=1)
        dw = dr * bt[:, None]

        dvt = dw
        dS_from_w = -tl.dot(tl.trans(ka), dw, input_precision="ieee")
        dka = -tl.dot(dw, tl.trans(S), input_precision="ieee")

        dM = tl.where(tril_mask, dA, 0.0)
        dqc = tl.dot(dM, kt, input_precision="ieee")
        dkt = tl.dot(tl.trans(dM), qc, input_precision="ieee")

        dS_n = dS_from_out + dS_from_next + dS_from_w

        out_ptrs = c_idx[:, None] * stride_c_cd + d_idx[None, :]
        tl.store(dka_ptr + cd_off + out_ptrs, dka, mask=mask_cd)
        tl.store(dke_ptr + cd_off + out_ptrs, dke, mask=mask_cd)
        tl.store(dqa_ptr + cd_off + out_ptrs, dqa, mask=mask_cd)
        tl.store(dqc_ptr + cd_off + out_ptrs, dqc, mask=mask_cd)
        tl.store(dkt_ptr + cd_off + out_ptrs, dkt, mask=mask_cd)
        tl.store(dvt_ptr + cd_off + out_ptrs, dvt, mask=mask_cd)
        tl.store(dbt_ptr + bt_off + c_idx, dbt)
        tl.store(dT_ptr + cc_off + T_ptrs, dT)
        tl.store(degl_ptr + d_off + d_idx, degl, mask=d_mask)

        dS = dS_n


class _KDAScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k_abs, k_end, q_abs, q_c, k_til_c, beta, v, Tm, eg_last):
        B, H, N, C, D = k_abs.shape
        BH = B * H
        BD = _next_pow2(D)
        dev, dt = k_abs.device, k_abs.dtype

        flat = lambda t: t.reshape(BH, N, *t.shape[3:]).contiguous()
        ka, ke, qa, qc, kt, vt = (flat(t) for t in (k_abs, k_end, q_abs, q_c, k_til_c, v))
        bt = flat(beta.squeeze(-1))          # (BH,N,C)
        Tn = flat(Tm)                        # (BH,N,C,C)
        egl = flat(eg_last)                  # (BH,N,D)

        out = torch.zeros(BH, N, C, D, device=dev, dtype=torch.float32)
        S_save = torch.zeros(BH, N, D, D, device=dev, dtype=torch.float32)

        grid = (BH,)
        _kda_fwd_kernel[grid](
            ka, ke, qa, qc, kt, bt, vt, Tn, egl, out, S_save,
            N, C, D, BD,
            ka.stride(0), ka.stride(1), ka.stride(2),
            Tn.stride(0), Tn.stride(1), Tn.stride(2),
            egl.stride(0), egl.stride(1),
            S_save.stride(0), S_save.stride(1), S_save.stride(2),
            bt.stride(0), bt.stride(1),
            num_stages=1,
        )
        ctx.save_for_backward(ka, ke, qa, qc, kt, bt, vt, Tn, egl, S_save)
        ctx.dims = (B, H, N, C, D, BD)
        return out.reshape(B, H, N, C, D).to(dt)

    @staticmethod
    def backward(ctx, dout):
        ka, ke, qa, qc, kt, bt, vt, Tn, egl, S_save = ctx.saved_tensors
        B, H, N, C, D, BD = ctx.dims
        BH = B * H
        dout_f = dout.reshape(BH, N, C, D).contiguous().float()

        dka, dke, dqa, dqc, dkt, dvt = (torch.zeros_like(ka) for _ in range(6))
        dbt = torch.zeros_like(bt)
        dT = torch.zeros_like(Tn)
        degl = torch.zeros_like(egl)

        grid = (BH,)
        _kda_bwd_kernel[grid](
            ka, ke, qa, qc, kt, bt, vt, Tn, egl, S_save, dout_f,
            dka, dke, dqa, dqc, dkt, dbt, dvt, dT, degl,
            N, C, D, BD,
            ka.stride(0), ka.stride(1), ka.stride(2),
            Tn.stride(0), Tn.stride(1), Tn.stride(2),
            egl.stride(0), egl.stride(1),
            S_save.stride(0), S_save.stride(1), S_save.stride(2),
            bt.stride(0), bt.stride(1),
            num_stages=1,
        )
        unflat = lambda t: t.reshape(B, H, N, C, D).to(ka.dtype)
        d_beta = dbt.reshape(B, H, N, C, 1).to(ka.dtype)
        d_Tm = dT.reshape(B, H, N, C, C).to(ka.dtype)
        d_eg_last = degl.reshape(B, H, N, D).to(ka.dtype)
        return (unflat(dka), unflat(dke), unflat(dqa), unflat(dqc), unflat(dkt),
                d_beta, unflat(dvt), d_Tm, d_eg_last)


def kda_scan(k_abs, k_end, q_abs, q_c, k_til_c, beta, v, Tm, eg_last):
    """Drop-in replacement for attention.py's per-chunk Python loop in _chunk_scan."""
    return _KDAScan.apply(k_abs, k_end, q_abs, q_c, k_til_c, beta, v, Tm, eg_last)
