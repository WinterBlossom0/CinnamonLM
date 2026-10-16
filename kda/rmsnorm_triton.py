"""Triton RMSNorm over a small last dimension, with an optional sigmoid output gate.

    y_i = x_i * rstd * w_i * s_i,   rstd = 1/sqrt(mean_i(x_i^2) + eps),   s = sigmoid(g), or s = 1 without a gate

Each program handles BG groups of D elements; nn.RMSNorm is slow for many groups of only a few elements.

Backward, with e_i = dy_i * s_i (the gradient at the norm output before the gate):

    dx_i = rstd*w_i*e_i - (rstd^3/D)*x_i*sum_j(w_j*e_j*x_j)
    dw_i = sum over all groups of e_i * x_i * rstd        (w is shared, so atomic_add across programs)
    dg_i = dy_i * x_i * rstd * w_i * s_i * (1 - s_i)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_GROUPS_PER_PROGRAM = 1024


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _rmsnorm_fwd_kernel(x_ptr, g_ptr, w_ptr, y_ptr, rstd_ptr, G, D: tl.constexpr, BD: tl.constexpr,
                        BG: tl.constexpr, eps, HAS_GATE: tl.constexpr):
    pid = tl.program_id(0)
    g_idx = pid * BG + tl.arange(0, BG)
    g_mask = g_idx < G
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D

    ptrs = g_idx[:, None] * D + d_idx[None, :]
    mask = g_mask[:, None] & d_mask[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptr + d_idx, mask=d_mask, other=0.0)

    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * rstd[:, None] * w[None, :]
    if HAS_GATE:
        y = y * tl.sigmoid(tl.load(g_ptr + ptrs, mask=mask, other=0.0))

    tl.store(y_ptr + ptrs, y, mask=mask)
    tl.store(rstd_ptr + g_idx, rstd, mask=g_mask)


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, g_ptr, w_ptr, rstd_ptr, dy_ptr, dx_ptr, dg_ptr, dw_ptr,
                        G, D: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, HAS_GATE: tl.constexpr):
    pid = tl.program_id(0)
    g_idx = pid * BG + tl.arange(0, BG)
    g_mask = g_idx < G
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D

    ptrs = g_idx[:, None] * D + d_idx[None, :]
    mask = g_mask[:, None] & d_mask[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptr + d_idx, mask=d_mask, other=0.0)
    rstd = tl.load(rstd_ptr + g_idx, mask=g_mask, other=0.0)
    r = rstd[:, None]
    xhat = x * r

    if HAS_GATE:
        s = tl.sigmoid(tl.load(g_ptr + ptrs, mask=mask, other=0.0))
        tl.store(dg_ptr + ptrs, dy * xhat * w[None, :] * s * (1.0 - s), mask=mask)
        dy = dy * s

    wdy = w[None, :] * dy
    c = tl.sum(wdy * x, axis=1) / D
    dx = r * wdy - (r * r * r) * x * c[:, None]
    tl.store(dx_ptr + ptrs, dx, mask=mask)

    dw_partial = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    tl.atomic_add(dw_ptr + d_idx, dw_partial, mask=d_mask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        shape = x.shape
        D = shape[-1]
        G = x.numel() // D
        BD = _next_pow2(D)
        has_gate = gate is not None
        xf = x.reshape(G, D).contiguous().float()
        gf = gate.reshape(G, D).contiguous().float() if has_gate else xf
        wf = weight.float()

        y = torch.empty_like(xf)
        rstd = torch.empty(G, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(G, _GROUPS_PER_PROGRAM),)
        _rmsnorm_fwd_kernel[grid](xf, gf, wf, y, rstd, G, D, BD, _GROUPS_PER_PROGRAM, eps, has_gate)

        ctx.save_for_backward(xf, gf, wf, rstd)
        ctx.dims = (shape, D, G, BD, has_gate)
        ctx.dt = x.dtype
        return y.reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, gf, wf, rstd = ctx.saved_tensors
        shape, D, G, BD, has_gate = ctx.dims
        dyf = dy.reshape(G, D).contiguous().float()

        dx = torch.empty_like(xf)
        dg = torch.empty_like(xf) if has_gate else dx
        dw = torch.zeros_like(wf)
        grid = (triton.cdiv(G, _GROUPS_PER_PROGRAM),)
        _rmsnorm_bwd_kernel[grid](xf, gf, wf, rstd, dyf, dx, dg, dw, G, D, BD, _GROUPS_PER_PROGRAM, has_gate)

        d_gate = dg.reshape(shape).to(ctx.dt) if has_gate else None
        return dx.reshape(shape).to(ctx.dt), d_gate, dw.to(ctx.dt), None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """nn.RMSNorm(x) with a plain (D,) weight."""
    return _RMSNormFn.apply(x, None, weight, eps)


def gated_rmsnorm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm(x) * weight * sigmoid(gate), as fla FusedRMSNormGated(activation='sigmoid')."""
    if not x.is_cuda:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight * torch.sigmoid(gate)
    return _RMSNormFn.apply(x, gate, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """nn.RMSNorm with one (dim,) weight, init ones, and no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)
