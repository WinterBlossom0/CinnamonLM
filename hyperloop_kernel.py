"""The read side of a Hyperloop hyper-connection (Zeitoun et al. 2026, arXiv:2604.21254, Sec. 3) as Triton kernels.

For n streams of width C (a row of y is the n*C stream values of one token) and the 3n rows of W (pre, post, res):

    z = RMSNorm(y) * norm_weight,   H = sigmoid(gate * (z W^T) + bias)
    mixed = sum_k H_pre[k] y_k,     write = H_post,   keep = H_res

One forward and one backward kernel replace about 12 compiled kernels per pass, so y is read 2 (forward) and
3 (backward) times instead of about 10. A program owns BLOCK_M tokens; the weight, gate, bias and norm gradients are
per-program partial sums that torch adds, so the result is deterministic. tl.dot runs in IEEE fp32, as the cuBLAS
matmul of the unfused model does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# T4, speed.py forward + backward at 81,920 rows: 32 / 1 warp is fastest in both sweeps (1.3596, 1.0469 ms).
BLOCK_M = 32
WARPS = 1


@triton.jit
def _stream(ptr, rows, row_mask, k, N: tl.constexpr, C: tl.constexpr):
    columns = tl.arange(0, C)
    return tl.load(ptr + rows[:, None] * (N * C) + k * C + columns[None, :], mask=row_mask[:, None], other=0.0)


@triton.jit
def _weight_t(w_ptr, k, N: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    """W[:, kC:(k+1)C]^T as (C, R), the rows past 3n zero."""
    hc = tl.arange(0, R)
    columns = tl.arange(0, C)
    return tl.load(w_ptr + hc[None, :] * (N * C) + k * C + columns[:, None], mask=hc[None, :] < 3 * N, other=0.0)


@triton.jit
def _column(tile, hc, j):
    return tl.sum(tl.where(hc[None, :] == j, tile, 0.0), 1)


@triton.jit
def _gates(s_ptr, w_ptr, g_ptr, b_ptr, nw_ptr, rows, row_mask, eps, N: tl.constexpr, C: tl.constexpr,
           R: tl.constexpr, BLOCK_M: tl.constexpr):
    """rstd, raw = (y * norm_weight) W^T, logits = rstd * raw, and H. rstd is a per-row factor of z, so it is
    applied after the matmul and y is read once."""
    hc = tl.arange(0, R)
    columns = tl.arange(0, C)
    sum_squares = tl.zeros((BLOCK_M,), tl.float32)
    raw = tl.zeros((BLOCK_M, R), tl.float32)
    for k in tl.static_range(N):
        stream = _stream(s_ptr, rows, row_mask, k, N, C)
        sum_squares += tl.sum(stream * stream, 1)
        norm_weight = tl.load(nw_ptr + k * C + columns)
        raw += tl.dot(stream * norm_weight[None, :], _weight_t(w_ptr, k, N, C, R), input_precision="ieee")
    rstd = tl.rsqrt(sum_squares / (N * C) + eps)
    logits = raw * rstd[:, None]
    gate = tl.load(g_ptr + hc, mask=hc < 3 * N, other=0.0)
    bias = tl.load(b_ptr + hc, mask=hc < 3 * N, other=0.0)
    return rstd, raw, logits, tl.sigmoid(logits * gate[None, :] + bias[None, :])


@triton.jit
def _forward_kernel(s_ptr, w_ptr, g_ptr, b_ptr, nw_ptr, mixed_ptr, write_ptr, keep_ptr, M, eps,
                    N: tl.constexpr, C: tl.constexpr, R: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    hc = tl.arange(0, R)
    columns = tl.arange(0, C)
    _, _, _, h = _gates(s_ptr, w_ptr, g_ptr, b_ptr, nw_ptr, rows, row_mask, eps, N, C, R, BLOCK_M)
    mixed = tl.zeros((BLOCK_M, C), tl.float32)
    for k in tl.static_range(N):
        mixed += _column(h, hc, k)[:, None] * _stream(s_ptr, rows, row_mask, k, N, C)
    tl.store(mixed_ptr + rows[:, None] * C + columns[None, :], mixed, mask=row_mask[:, None])
    is_write = (hc[None, :] >= N) & (hc[None, :] < 2 * N)
    is_keep = (hc[None, :] >= 2 * N) & (hc[None, :] < 3 * N)
    tl.store(write_ptr + rows[:, None] * N + (hc[None, :] - N), h, mask=row_mask[:, None] & is_write)
    tl.store(keep_ptr + rows[:, None] * N + (hc[None, :] - 2 * N), h, mask=row_mask[:, None] & is_keep)


@triton.jit
def _backward_kernel(s_ptr, w_ptr, g_ptr, b_ptr, nw_ptr, grad_mixed_ptr, grad_write_ptr, grad_keep_ptr,
                     grad_s_ptr, grad_w_ptr, grad_g_ptr, grad_b_ptr, grad_nw_ptr, M, eps,
                     N: tl.constexpr, C: tl.constexpr, R: tl.constexpr, BLOCK_M: tl.constexpr):
    program = tl.program_id(0)
    rows = program * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    hc = tl.arange(0, R)
    columns = tl.arange(0, C)
    rstd, raw, logits, h = _gates(s_ptr, w_ptr, g_ptr, b_ptr, nw_ptr, rows, row_mask, eps, N, C, R, BLOCK_M)
    grad_mixed = tl.load(grad_mixed_ptr + rows[:, None] * C + columns[None, :], mask=row_mask[:, None], other=0.0)
    is_write = (hc[None, :] >= N) & (hc[None, :] < 2 * N)
    is_keep = (hc[None, :] >= 2 * N) & (hc[None, :] < 3 * N)
    grad_h = tl.load(grad_write_ptr + rows[:, None] * N + (hc[None, :] - N), mask=row_mask[:, None] & is_write,
                     other=0.0)
    grad_h += tl.load(grad_keep_ptr + rows[:, None] * N + (hc[None, :] - 2 * N), mask=row_mask[:, None] & is_keep,
                      other=0.0)
    for k in tl.static_range(N):
        grad_read = tl.sum(grad_mixed * _stream(s_ptr, rows, row_mask, k, N, C), 1)
        grad_h += tl.where(hc[None, :] == k, grad_read[:, None], 0.0)
    grad_a = grad_h * h * (1.0 - h)
    tl.store(grad_b_ptr + program * R + hc, tl.sum(grad_a, 0))
    tl.store(grad_g_ptr + program * R + hc, tl.sum(grad_a * logits, 0))
    gate = tl.load(g_ptr + hc, mask=hc < 3 * N, other=0.0)
    grad_logits = grad_a * gate[None, :]
    # logits = rstd * raw: raw takes rstd * grad_logits, and rstd takes sum_j grad_logits_j raw_j, which flows to y
    # through d rstd / d y = -rstd^3 y / (nC).
    grad_raw = grad_logits * rstd[:, None]
    grad_rstd_coeff = -tl.sum(grad_logits * raw, 1) * rstd * rstd * rstd / (N * C)
    for k in tl.static_range(N):
        stream = _stream(s_ptr, rows, row_mask, k, N, C)
        norm_weight = tl.load(nw_ptr + k * C + columns)
        weight_t = _weight_t(w_ptr, k, N, C, R)
        grad_weight_t = tl.dot(tl.trans(stream * norm_weight[None, :]), grad_raw, input_precision="ieee")
        tl.store(grad_w_ptr + ((program * N + k) * C + columns[:, None]) * R + hc[None, :], grad_weight_t)
        grad_scaled = tl.dot(grad_raw, tl.trans(weight_t), input_precision="ieee")
        tl.store(grad_nw_ptr + program * (N * C) + k * C + columns, tl.sum(grad_scaled * stream, 0))
        grad_stream = (grad_scaled * norm_weight[None, :] + grad_rstd_coeff[:, None] * stream
                       + _column(h, hc, k)[:, None] * grad_mixed)
        tl.store(grad_s_ptr + rows[:, None] * (N * C) + k * C + columns[None, :], grad_stream,
                 mask=row_mask[:, None])


def _padded_rows(streams: int) -> int:
    # tl.dot needs each dimension >= 16 and a power of 2.
    return max(16, triton.next_power_of_2(3 * streams))


def _check_shapes(flat: torch.Tensor, weight: torch.Tensor, streams: int) -> None:
    rows, width = flat.shape
    if flat.dtype != torch.float32 or not flat.is_cuda:
        raise ValueError("the Hyperloop kernels take fp32 CUDA tensors")
    if width % streams or (width // streams) < 16 or (width // streams) & (width // streams - 1):
        raise ValueError(f"stream width {width}/{streams} must be a power of 2 >= 16 (tl.dot)")
    if weight.shape != (3 * streams, width):
        raise ValueError(f"weight {tuple(weight.shape)}, expected {(3 * streams, width)}")


@torch.library.custom_op("recursion_comparison::hyperloop_read", mutates_args=())
def hyperloop_read(flat: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor,
                   norm_weight: torch.Tensor, streams: int,
                   eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """flat: (M, n*C) streams; weight (3n, n*C); gate and bias (3n,) per row; norm_weight (n*C,).
    Returns mixed (M, C), write (M, n), keep (M, n)."""
    _check_shapes(flat, weight, streams)
    flat = flat.contiguous()
    rows, width = flat.shape
    mixed = flat.new_empty(rows, width // streams)
    write, keep = flat.new_empty(rows, streams), flat.new_empty(rows, streams)
    _forward_kernel[(triton.cdiv(rows, BLOCK_M),)](flat, weight.contiguous(), gate, bias, norm_weight, mixed, write,
                                                   keep, rows, eps, N=streams, C=width // streams,
                                                   R=_padded_rows(streams), BLOCK_M=BLOCK_M, num_warps=WARPS)
    return mixed, write, keep


@hyperloop_read.register_fake
def _(flat, weight, gate, bias, norm_weight, streams, eps):
    rows, width = flat.shape
    return flat.new_empty(rows, width // streams), flat.new_empty(rows, streams), flat.new_empty(rows, streams)


@torch.library.custom_op("recursion_comparison::hyperloop_read_backward", mutates_args=())
def hyperloop_read_backward(flat: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor,
                            norm_weight: torch.Tensor, grad_mixed: torch.Tensor, grad_write: torch.Tensor,
                            grad_keep: torch.Tensor, streams: int, eps: float
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = flat.contiguous()
    rows, width = flat.shape
    width_per_stream, padded = width // streams, _padded_rows(streams)
    programs = triton.cdiv(rows, BLOCK_M)
    grad_flat = torch.empty_like(flat)
    grad_weight_t = flat.new_empty(programs, streams, width_per_stream, padded)
    grad_gate, grad_bias = flat.new_empty(programs, padded), flat.new_empty(programs, padded)
    grad_norm_weight = flat.new_empty(programs, width)
    _backward_kernel[(programs,)](flat, weight.contiguous(), gate, bias, norm_weight, grad_mixed.contiguous(),
                                  grad_write.contiguous(), grad_keep.contiguous(), grad_flat, grad_weight_t,
                                  grad_gate, grad_bias, grad_norm_weight, rows, eps, N=streams, C=width_per_stream,
                                  R=padded, BLOCK_M=BLOCK_M, num_warps=WARPS)
    hc_rows = 3 * streams
    # (k, c, j) -> W[j, k C + c]
    grad_weight = grad_weight_t.sum(0)[..., :hc_rows].permute(2, 0, 1).reshape(hc_rows, width).contiguous()
    return (grad_flat, grad_weight, grad_gate.sum(0)[:hc_rows], grad_bias.sum(0)[:hc_rows],
            grad_norm_weight.sum(0))


@hyperloop_read_backward.register_fake
def _(flat, weight, gate, bias, norm_weight, grad_mixed, grad_write, grad_keep, streams, eps):
    return (torch.empty_like(flat), torch.empty_like(weight), torch.empty_like(gate), torch.empty_like(bias),
            torch.empty_like(norm_weight))


def _setup_context(ctx, inputs, output):
    flat, weight, gate, bias, norm_weight, streams, eps = inputs
    ctx.save_for_backward(flat, weight, gate, bias, norm_weight)
    ctx.streams, ctx.eps = streams, eps


def _backward(ctx, grad_mixed, grad_write, grad_keep):
    flat = ctx.saved_tensors[0]
    grad_mixed, grad_write, grad_keep = (torch.zeros(flat.shape[0], size, device=flat.device) if grad is None else grad
                                         for grad, size in ((grad_mixed, flat.shape[1] // ctx.streams),
                                                            (grad_write, ctx.streams), (grad_keep, ctx.streams)))
    grads = hyperloop_read_backward(*ctx.saved_tensors, grad_mixed, grad_write, grad_keep, ctx.streams, ctx.eps)
    return (*grads, None, None)


hyperloop_read.register_autograd(_backward, setup_context=_setup_context)
