"""Triton kernel for RMSNorm over a tiny last-dimension with a huge number of
independent groups -- KDA's actual measured bottleneck at head_dim=3
(profiling: nn.RMSNorm + its internal moment kernel = 57% of total time,
from B*T*H=655,360 independent groups of just 3 elements each). PyTorch's
fused RMSNorm kernel isn't tuned for reductions this tiny relative to the
number of groups; this batches many groups per Triton program instead of
launching per-group work.

RMSNorm(x)_i = x_i * rstd * w_i,  rstd = 1/sqrt(mean_i(x_i^2) + eps)

Backward (standard RMSNorm adjoint, D the reduced dim):

    dL/dx_i = rstd*w_i*dy_i - (rstd^3/D)*x_i*sum_j(w_j*dy_j*x_j)
    dL/dw_i = sum over ALL groups of dy_i * (x_i*rstd)      -- w is SHARED
              across every group, so this needs a reduction across groups,
              not just within one (done via atomic_add into a (D,) buffer).
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
def _rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, G, D: tl.constexpr, BD: tl.constexpr,
                        BG: tl.constexpr, eps):
    pid = tl.program_id(0)
    g_idx = pid * BG + tl.arange(0, BG)
    g_mask = g_idx < G
    d_idx = tl.arange(0, BD)
    d_mask = d_idx < D

    ptrs = g_idx[:, None] * D + d_idx[None, :]
    mask = g_mask[:, None] & d_mask[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptr + d_idx, mask=d_mask, other=0.0)

    ms = tl.sum(x * x, axis=1) / D
    rstd = 1.0 / tl.sqrt(ms + eps)
    y = x * rstd[:, None] * w[None, :]

    tl.store(y_ptr + ptrs, y, mask=mask)
    tl.store(rstd_ptr + g_idx, rstd, mask=g_mask)


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, w_ptr, rstd_ptr, dy_ptr, dx_ptr, dw_ptr,
                        G, D: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr):
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

    xhat = x * rstd[:, None]
    wdy = w[None, :] * dy
    c = tl.sum(wdy * x, axis=1) / D                      # (BG,)
    r = rstd[:, None]
    dx = r * wdy - (r * r * r) * x * c[:, None]
    tl.store(dx_ptr + ptrs, dx, mask=mask)

    dw_partial = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)  # (BD,), this block's groups
    tl.atomic_add(dw_ptr + d_idx, dw_partial, mask=d_mask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        D = shape[-1]
        G = x.numel() // D
        BD = _next_pow2(D)
        BG = 1024
        xf = x.reshape(G, D).contiguous().float()
        wf = weight.float()

        y = torch.empty_like(xf)
        rstd = torch.empty(G, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(G, BG),)
        _rmsnorm_fwd_kernel[grid](xf, wf, y, rstd, G, D, BD, BG, eps)

        ctx.save_for_backward(xf, wf, rstd)
        ctx.dims = (shape, D, G, BD, BG)
        ctx.dt = x.dtype
        return y.reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, wf, rstd = ctx.saved_tensors
        shape, D, G, BD, BG = ctx.dims
        dyf = dy.reshape(G, D).contiguous().float()

        dx = torch.empty_like(xf)
        dw = torch.zeros_like(wf)
        grid = (triton.cdiv(G, BG),)
        _rmsnorm_bwd_kernel[grid](xf, wf, rstd, dyf, dx, dw, G, D, BD, BG)

        return dx.reshape(shape).to(ctx.dt), dw.to(ctx.dt), None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Drop-in for nn.RMSNorm(x) when weight is a plain (D,) nn.Parameter."""
    return _RMSNormFn.apply(x, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for nn.RMSNorm with matching state_dict (one
    (dim,) `weight` parameter, no bias) -- same init (ones)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)
