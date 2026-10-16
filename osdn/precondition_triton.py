"""Triton kernel for OSDN phase 1, the online preconditioner sweep (arXiv:2605.13473, Alg. 1, eq. 4).

Per (batch, head), with d = 1 at each sequence start (App. F; config initial_d "ones"):

    d_traj[t] = d                                             pre-update value; write key = d_traj[t] * k_t
    n = max(||k_t||^2, EPS)
    step = ETA * beta_t * (1 - beta_t * <d, k_t^2>) / n       beta-aware form (App. F; config beta_aware true)
    d = clamp(d + step * k_t^2, D_MIN, D_MAX)                 box projection Pi_D

The box clamp makes the map piecewise affine, so the sweep runs token by token. A plain PyTorch token
loop launches several small kernels per token, so the whole sweep is one kernel launch here.
The backward is the hand-derived reverse sweep (paper Sec. 4.5). It gives zero gradient through
an active clamp or floor, with the same inclusive bounds as torch.clamp and torch.clamp_min.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Paper values for the 340M OSDN runs (Sec. 4.2, App. C Listing 1, App. F).
ETA = 0.003
D_MIN, D_MAX = 0.5, 2.0
EPS = 1e-6


@triton.jit
def _precond_fwd_kernel(
    k_ptr, beta_ptr, d_traj_ptr, d_pre_ptr,
    T, D: tl.constexpr, BD: tl.constexpr,
    eta, d_min, d_max, eps,
    stride_bh_td, stride_t_td,
    stride_bh_tb, stride_t_tb,
):
    bh = tl.program_id(0)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D

    d = tl.where(d_mask, 1.0, 0.0)
    td_base = bh * stride_bh_td
    tb_base = bh * stride_bh_tb

    for t in range(0, T):
        td_off = td_base + t * stride_t_td
        k_t = tl.load(k_ptr + td_off + d_idx, mask=d_mask, other=0.0)
        b_t = tl.load(beta_ptr + tb_base + t * stride_t_tb)

        tl.store(d_traj_ptr + td_off + d_idx, d, mask=d_mask)

        k_sq = k_t * k_t
        n = tl.maximum(tl.sum(k_sq, axis=0), eps)
        inner = tl.sum(d * k_sq, axis=0)
        step = b_t * (1.0 - b_t * inner) / n

        d_pre = d + eta * step * k_sq
        tl.store(d_pre_ptr + td_off + d_idx, d_pre, mask=d_mask)
        d = tl.minimum(tl.maximum(d_pre, d_min), d_max)


@triton.jit
def _precond_bwd_kernel(
    k_ptr, beta_ptr, d_traj_ptr, d_pre_ptr, grad_d_traj_ptr,
    grad_k_ptr, grad_beta_ptr,
    T, D: tl.constexpr, BD: tl.constexpr,
    eta, d_min, d_max, eps,
    stride_bh_td, stride_t_td,
    stride_bh_tb, stride_t_tb,
):
    bh = tl.program_id(0)
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D

    # dL/d(d_{t+1}); zero after the last token because the final d is not an output.
    grad_d_next = tl.zeros((BD,), dtype=tl.float32)
    td_base = bh * stride_bh_td
    tb_base = bh * stride_bh_tb

    for step_i in range(0, T):
        t = T - 1 - step_i
        td_off = td_base + t * stride_t_td
        tb_off = tb_base + t * stride_t_tb

        k_t = tl.load(k_ptr + td_off + d_idx, mask=d_mask, other=0.0)
        b_t = tl.load(beta_ptr + tb_off)
        d_t = tl.load(d_traj_ptr + td_off + d_idx, mask=d_mask, other=0.0)
        d_pre = tl.load(d_pre_ptr + td_off + d_idx, mask=d_mask, other=0.0)
        grad_d_out = tl.load(grad_d_traj_ptr + td_off + d_idx, mask=d_mask, other=0.0)

        k_sq = k_t * k_t
        n_raw = tl.sum(k_sq, axis=0)
        n = tl.maximum(n_raw, eps)
        inner = tl.sum(d_t * k_sq, axis=0)
        step = b_t * (1.0 - b_t * inner) / n

        in_box = (d_pre >= d_min) & (d_pre <= d_max)
        grad_d_pre = tl.where(in_box, grad_d_next, 0.0)
        grad_step = eta * tl.sum(grad_d_pre * k_sq, axis=0)        # dL/d(step)

        # d step / d d_t = -beta^2 k^2 / n;  d step / d k^2 = -(beta^2 d_t + step * [n_raw >= eps]) / n
        grad_d_t = grad_d_out + grad_d_pre - (b_t * b_t / n) * k_sq * grad_step
        step_over_n = tl.where(n_raw >= eps, step / n, 0.0)
        grad_k_sq = eta * step * grad_d_pre - (b_t * b_t * d_t / n + step_over_n) * grad_step
        grad_beta = (1.0 - 2.0 * b_t * inner) / n * grad_step

        tl.store(grad_k_ptr + td_off + d_idx, 2.0 * k_t * grad_k_sq, mask=d_mask)
        tl.store(grad_beta_ptr + tb_off, grad_beta)

        grad_d_next = grad_d_t


class _OSDNPrecondition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, beta):
        B, T, H, D = k.shape
        BH = B * H
        BD = triton.next_power_of_2(D)

        # (B,T,H,*) -> (B,H,T,*) -> (BH,T,*): H must sit next to B before the flatten.
        k_f = k.transpose(1, 2).reshape(BH, T, D).contiguous().float()
        beta_f = beta.transpose(1, 2).reshape(BH, T).contiguous().float()

        d_traj = torch.empty(BH, T, D, device=k.device, dtype=torch.float32)
        d_pre = torch.empty_like(d_traj)
        _precond_fwd_kernel[(BH,)](
            k_f, beta_f, d_traj, d_pre,
            T, D, BD, ETA, D_MIN, D_MAX, EPS,
            k_f.stride(0), k_f.stride(1),
            beta_f.stride(0), beta_f.stride(1),
        )
        ctx.save_for_backward(k_f, beta_f, d_traj, d_pre)
        ctx.dims = (B, H, T, D, BD)
        return d_traj.reshape(B, H, T, D).transpose(1, 2).to(k.dtype)

    @staticmethod
    def backward(ctx, grad_d_traj):
        k_f, beta_f, d_traj, d_pre = ctx.saved_tensors
        B, H, T, D, BD = ctx.dims
        BH = B * H

        grad_f = grad_d_traj.transpose(1, 2).reshape(BH, T, D).contiguous().float()
        grad_k = torch.empty_like(k_f)
        grad_beta = torch.empty_like(beta_f)
        _precond_bwd_kernel[(BH,)](
            k_f, beta_f, d_traj, d_pre, grad_f,
            grad_k, grad_beta,
            T, D, BD, ETA, D_MIN, D_MAX, EPS,
            k_f.stride(0), k_f.stride(1),
            beta_f.stride(0), beta_f.stride(1),
        )
        grad_k = grad_k.reshape(B, H, T, D).transpose(1, 2).to(grad_d_traj.dtype)
        grad_beta = grad_beta.reshape(B, H, T).transpose(1, 2).to(grad_d_traj.dtype)
        return grad_k, grad_beta


def osdn_precondition(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Pre-update preconditioner d_t for each token, shape (B,T,H,D). k is (B,T,H,D), beta is (B,T,H)."""
    return _OSDNPrecondition.apply(k, beta)
