"""Parallax core op of arXiv:2605.29157: eq. 14 / Algorithm 1 (Sec. 3.3) forward and the closed-form
backward of Sec. 9, as a fused Triton kernel with a plain PyTorch path for other devices.

q and rho are (B, T, H, D); k and v are (B, T, H_kv, D); query head h reads K/V head h // (H / H_kv).
With s = 1/sqrt(D), p_ij = softmax_{j <= i}(s q_i . k_j) and t_ij = rho_i . k_j:

    o_i = vbar_i (1 + tbar_i) - sum_j p_ij t_ij v_j,    vbar_i = sum_j p_ij v_j,    tbar_i = sum_j p_ij t_ij

Backward (Sec. 9):

    a_ij = dO_i . v_j    tau_i = dO_i . o_i    beta_i = dO_i . vbar_i    delta_ij = a_ij - beta_i
    g1_ij = p_ij (a_ij - tau_i + (tbar_i - t_ij) delta_ij)    g2_ij = -p_ij delta_ij
    dq_i = s sum_j g1_ij k_j    drho_i = sum_j g2_ij k_j
    dk_j = s sum_i g1_ij q_i + sum_i g2_ij rho_i    dv_j = sum_i p_ij (1 + tbar_i - t_ij) dO_i
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 32  # rows per tile; at 64 the dk/dv kernel needs 68 KB of shared memory, over the 64 KB of sm_75 (T4)
NUM_WARPS = 2
NUM_STAGES = 1  # sm_75 has no async copy to pipeline, so one stage keeps one shared-memory size on every GPU
MIN_CAPABILITY = (7, 5)  # the oldest GPU generation that Triton 3 compiles for


@triton.jit
def _fwd_kernel(q_ptr, r_ptr, k_ptr, v_ptr, o_ptr, vbar_ptr, lse_ptr, tbar_ptr,
                T, H, H_KV, D, scale,
                stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // (H // H_KV)
    rows = pid_m * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    q_mask = (rows < T)[:, None] & (dims < D)[None, :]
    q_off = b * stride_qb + h * stride_qh + rows[:, None] * stride_qt + dims[None, :]
    q = tl.load(q_ptr + q_off, mask=q_mask, other=0.0)
    r = tl.load(r_ptr + q_off, mask=q_mask, other=0.0)

    m_run = tl.full((BM,), float("-inf"), tl.float32)
    d1 = tl.zeros((BM,), tl.float32)
    d2 = tl.zeros((BM,), tl.float32)
    o1 = tl.zeros((BM, BD), tl.float32)
    o2 = tl.zeros((BM, BD), tl.float32)
    kv_base = b * stride_kb + hk * stride_kh
    for n0 in range(0, tl.minimum((pid_m + 1) * BM, T), BN):
        cols = n0 + tl.arange(0, BN)
        kv_off = kv_base + cols[:, None] * stride_kt + dims[None, :]
        kv_mask = (cols < T)[:, None] & (dims < D)[None, :]
        k = tl.load(k_ptr + kv_off, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kv_off, mask=kv_mask, other=0.0)
        visible = (cols[None, :] <= rows[:, None]) & (cols < T)[None, :]
        logits = tl.where(visible, tl.dot(q, tl.trans(k), input_precision="ieee") * scale, float("-inf"))
        # Key 0 is in the first tile of every row, so m_new is finite and alpha has no inf - inf.
        m_new = tl.maximum(m_run, tl.max(logits, 1))
        alpha = tl.exp(m_run - m_new)
        w = tl.exp(logits - m_new[:, None])
        wt = w * tl.dot(r, tl.trans(k), input_precision="ieee")
        d1 = (d1 * alpha) + tl.sum(w, 1)
        d2 = (d2 * alpha) + tl.sum(wt, 1)
        o1 = (o1 * alpha[:, None]) + tl.dot(w, v, input_precision="ieee")
        o2 = (o2 * alpha[:, None]) + tl.dot(wt, v, input_precision="ieee")
        m_run = m_new

    vbar = o1 / d1[:, None]
    tbar = d2 / d1
    o = (vbar * (1.0 + tbar[:, None])) - (o2 / d1[:, None])
    tl.store(o_ptr + q_off, o, mask=q_mask)
    tl.store(vbar_ptr + q_off, vbar, mask=q_mask)
    stat_off = pid_bh * T + rows
    tl.store(lse_ptr + stat_off, m_run + tl.log(d1), mask=rows < T)
    tl.store(tbar_ptr + stat_off, tbar, mask=rows < T)


@triton.jit
def _bwd_dq_kernel(q_ptr, r_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, tbar_ptr, tau_ptr, beta_ptr, dq_ptr, dr_ptr,
                   T, H, H_KV, D, scale,
                   stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh,
                   BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // (H // H_KV)
    rows = pid_m * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    q_mask = (rows < T)[:, None] & (dims < D)[None, :]
    q_off = b * stride_qb + h * stride_qh + rows[:, None] * stride_qt + dims[None, :]
    q = tl.load(q_ptr + q_off, mask=q_mask, other=0.0)
    r = tl.load(r_ptr + q_off, mask=q_mask, other=0.0)
    do = tl.load(do_ptr + q_off, mask=q_mask, other=0.0)
    stat_off = pid_bh * T + rows
    lse = tl.load(lse_ptr + stat_off, mask=rows < T, other=0.0)
    tbar = tl.load(tbar_ptr + stat_off, mask=rows < T, other=0.0)
    tau = tl.load(tau_ptr + stat_off, mask=rows < T, other=0.0)
    beta = tl.load(beta_ptr + stat_off, mask=rows < T, other=0.0)

    dq = tl.zeros((BM, BD), tl.float32)
    dr = tl.zeros((BM, BD), tl.float32)
    kv_base = b * stride_kb + hk * stride_kh
    for n0 in range(0, tl.minimum((pid_m + 1) * BM, T), BN):
        cols = n0 + tl.arange(0, BN)
        kv_off = kv_base + cols[:, None] * stride_kt + dims[None, :]
        kv_mask = (cols < T)[:, None] & (dims < D)[None, :]
        k = tl.load(k_ptr + kv_off, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kv_off, mask=kv_mask, other=0.0)
        visible = (cols[None, :] <= rows[:, None]) & (cols < T)[None, :]
        logits = tl.where(visible, tl.dot(q, tl.trans(k), input_precision="ieee") * scale, float("-inf"))
        p = tl.exp(logits - lse[:, None])
        t = tl.dot(r, tl.trans(k), input_precision="ieee")
        a = tl.dot(do, tl.trans(v), input_precision="ieee")
        delta = a - beta[:, None]
        g1 = p * ((a - tau[:, None]) + ((tbar[:, None] - t) * delta))
        g2 = -(p * delta)
        dq += tl.dot(g1, k, input_precision="ieee")
        dr += tl.dot(g2, k, input_precision="ieee")
    tl.store(dq_ptr + q_off, dq * scale, mask=q_mask)
    tl.store(dr_ptr + q_off, dr, mask=q_mask)


@triton.jit
def _bwd_dkv_kernel(q_ptr, r_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, tbar_ptr, tau_ptr, beta_ptr, dk_ptr, dv_ptr,
                    T, H, H_KV, D, scale,
                    stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh,
                    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H_KV
    hk = pid_bh % H_KV
    group = H // H_KV
    cols = pid_n * BN + tl.arange(0, BN)
    dims = tl.arange(0, BD)
    kv_mask = (cols < T)[:, None] & (dims < D)[None, :]
    kv_off = b * stride_kb + hk * stride_kh + cols[:, None] * stride_kt + dims[None, :]
    k = tl.load(k_ptr + kv_off, mask=kv_mask, other=0.0)
    v = tl.load(v_ptr + kv_off, mask=kv_mask, other=0.0)

    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    # Each program owns its K/V rows and sums over the query heads of its group, so no atomics are needed.
    for g in range(0, group):
        h = (hk * group) + g
        for m0 in range((pid_n * BN // BM) * BM, T, BM):
            rows = m0 + tl.arange(0, BM)
            q_mask = (rows < T)[:, None] & (dims < D)[None, :]
            q_off = b * stride_qb + h * stride_qh + rows[:, None] * stride_qt + dims[None, :]
            q = tl.load(q_ptr + q_off, mask=q_mask, other=0.0)
            r = tl.load(r_ptr + q_off, mask=q_mask, other=0.0)
            do = tl.load(do_ptr + q_off, mask=q_mask, other=0.0)
            stat_off = (b * H + h) * T + rows
            lse = tl.load(lse_ptr + stat_off, mask=rows < T, other=0.0)
            tbar = tl.load(tbar_ptr + stat_off, mask=rows < T, other=0.0)
            tau = tl.load(tau_ptr + stat_off, mask=rows < T, other=0.0)
            beta = tl.load(beta_ptr + stat_off, mask=rows < T, other=0.0)
            visible = (cols[:, None] <= rows[None, :]) & (rows < T)[None, :]
            logits = tl.where(visible, tl.dot(k, tl.trans(q), input_precision="ieee") * scale, float("-inf"))
            p = tl.exp(logits - lse[None, :])
            t = tl.dot(k, tl.trans(r), input_precision="ieee")
            a = tl.dot(v, tl.trans(do), input_precision="ieee")
            delta = a - beta[None, :]
            g1 = p * ((a - tau[None, :]) + ((tbar[None, :] - t) * delta))
            g2 = -(p * delta)
            dv += tl.dot(p * (1.0 + (tbar[None, :] - t)), do, input_precision="ieee")
            dk += (tl.dot(g1, q, input_precision="ieee") * scale) + tl.dot(g2, r, input_precision="ieee")
    tl.store(dk_ptr + kv_off, dk, mask=kv_mask)
    tl.store(dv_ptr + kv_off, dv, mask=kv_mask)


def _launch_args(q: torch.Tensor, k: torch.Tensor) -> tuple:
    _, length, n_heads, head_dim = q.shape
    return (length, n_heads, k.shape[2], head_dim, head_dim ** -0.5,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2))


def _tile_args(head_dim: int) -> dict:
    # tl.dot needs every dimension >= 16; masked loads zero the padded columns.
    return dict(BM=BLOCK, BN=BLOCK, BD=max(16, triton.next_power_of_2(head_dim)), num_warps=NUM_WARPS,
                num_stages=NUM_STAGES)


class _ParallaxCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rho):
        q, k, v, rho = (t.contiguous() for t in (q, k, v, rho))
        batch, length, n_heads, head_dim = q.shape
        o, vbar = torch.empty_like(q), torch.empty_like(q)
        lse = q.new_empty(batch, n_heads, length)
        tbar = q.new_empty(batch, n_heads, length)
        grid = (triton.cdiv(length, BLOCK), batch * n_heads)
        _fwd_kernel[grid](q, rho, k, v, o, vbar, lse, tbar, *_launch_args(q, k), **_tile_args(head_dim))
        ctx.save_for_backward(q, k, v, rho, o, vbar, lse, tbar)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, rho, o, vbar, lse, tbar = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        batch, length, n_heads, head_dim = q.shape
        tau = (grad_o * o).sum(-1).transpose(1, 2).contiguous()
        beta = (grad_o * vbar).sum(-1).transpose(1, 2).contiguous()
        dq, drho = torch.empty_like(q), torch.empty_like(rho)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        args, tiles = _launch_args(q, k), _tile_args(head_dim)
        _bwd_dq_kernel[(triton.cdiv(length, BLOCK), batch * n_heads)](
            q, rho, k, v, grad_o, lse, tbar, tau, beta, dq, drho, *args, **tiles)
        _bwd_dkv_kernel[(triton.cdiv(length, BLOCK), batch * k.shape[2])](
            q, rho, k, v, grad_o, lse, tbar, tau, beta, dk, dv, *args, **tiles)
        return dq, dk, dv, drho


def parallax_core_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    group = q.shape[2] // k.shape[2]
    q, rho = q.transpose(1, 2), rho.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(group, dim=1)
    v = v.transpose(1, 2).repeat_interleave(group, dim=1)
    length, head_dim = q.shape[-2:]
    future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=q.device), diagonal=1)
    logits = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
    p = torch.softmax(logits.masked_fill(future, float("-inf")), dim=-1)
    composite = p * (rho @ k.transpose(-1, -2))
    o = ((p @ v) * (1.0 + composite.sum(-1, keepdim=True))) - (composite @ v)
    return o.transpose(1, 2)


def uses_kernel(t: torch.Tensor) -> bool:
    return t.is_cuda and (t.dtype == torch.float32) and (torch.cuda.get_device_capability(t.device) >= MIN_CAPABILITY)


def parallax_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    if uses_kernel(q):
        return _ParallaxCore.apply(q, k, v, rho)
    return parallax_core_torch(q, k, v, rho)
