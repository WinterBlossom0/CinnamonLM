"""GPT-Neo's masked softmax(q k^T) v (no 1/sqrt(d) scale) as Triton kernels for a small head dimension.

The kernels follow flash attention (Dao et al., arXiv:2205.14135): a program owns a block of queries (or keys) and
visits only the key (or query) blocks that the causal mask and the local window let it see, with an online softmax
in the forward and p recomputed from the saved log-sum-exp in the backward. The backward runs as two kernels (query
blocks for dq, key blocks for dk and dv), so no program needs an atomic add and the result is deterministic. With
D = 4 the products are broadcasts summed over D, not tl.dot, which needs D >= 16.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# A row with no visible key yet keeps this running max, so exp(-inf - MASKED_MAX) = 0 instead of NaN.
MASKED_MAX = tl.constexpr(-1e30)


@triton.jit
def _column(ptr, positions, d, stride_t, stride_d):
    return tl.load(ptr + positions * stride_t + d * stride_d)


@triton.jit
def _masked_scores(q_ptr, k_ptr, rows, cols, stride_t, stride_d, D: tl.constexpr, WINDOW: tl.constexpr):
    # q k^T as D rank-1 updates: 2-D products only (a 3-D broadcast summed over D measured 2x slower).
    scores = tl.zeros((rows.shape[0], cols.shape[0]), tl.float32)
    for d in tl.static_range(D):
        scores += _column(q_ptr, rows, d, stride_t, stride_d)[:, None] * _column(k_ptr, cols, d, stride_t, stride_d)[None, :]
    # Global layers pass WINDOW = T; local layers keep the WINDOW most recent keys, as GPT-Neo's local mask does.
    keep = (cols[None, :] <= rows[:, None]) & ((rows[:, None] - cols[None, :]) < WINDOW)
    return tl.where(keep, scores, float("-inf"))


@triton.jit
def _put(tile, d, values, dims):
    """tile with column d increased by values."""
    return tile + tl.where(dims[None, :] == d, values[:, None], 0.0)


@triton.jit
def _first_key_block(row_start, WINDOW: tl.constexpr, BLOCK_N: tl.constexpr):
    return (tl.maximum(row_start - WINDOW + 1, 0) // BLOCK_N) * BLOCK_N


@triton.jit
def _forward_kernel(q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, heads, stride_b, stride_h, stride_t, stride_d,
                    T: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr):
    batch_head = tl.program_id(0)
    offset = (batch_head // heads) * stride_b + (batch_head % heads) * stride_h
    row_start = tl.program_id(1) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    running_max = tl.full((BLOCK_M,), MASKED_MAX, tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for key_start in range(_first_key_block(row_start, WINDOW, BLOCK_N), row_start + BLOCK_M, BLOCK_N):
        cols = key_start + tl.arange(0, BLOCK_N)
        scores = _masked_scores(q_ptr + offset, k_ptr + offset, rows, cols, stride_t, stride_d, D, WINDOW)
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)
        acc = acc * rescale[:, None]
        for d in tl.static_range(D):
            v_d = _column(v_ptr + offset, cols, d, stride_t, stride_d)
            acc = _put(acc, d, tl.sum(weights * v_d[None, :], axis=1), dims)
        running_max = new_max
    dense = batch_head * T * D
    tl.store(out_ptr + dense + rows[:, None] * D + dims[None, :], acc / running_sum[:, None])
    tl.store(lse_ptr + batch_head * T + rows, running_max + tl.log(running_sum))


@triton.jit
def _grad_scores(q_ptr, k_ptr, v_ptr, grad_out_ptr, rows, cols, lse, delta, stride_t, stride_d,
                 D: tl.constexpr, WINDOW: tl.constexpr):
    """P and dS = P * (dP - delta) on the tile rows x cols, with dP = dO V^T."""
    probs = tl.exp(_masked_scores(q_ptr, k_ptr, rows, cols, stride_t, stride_d, D, WINDOW) - lse[:, None])
    grad_probs = tl.zeros((rows.shape[0], cols.shape[0]), tl.float32)
    for d in tl.static_range(D):
        grad_probs += _column(grad_out_ptr, rows, d, D, 1)[:, None] * _column(v_ptr, cols, d, stride_t, stride_d)[None, :]
    return probs, probs * (grad_probs - delta[:, None])


@triton.jit
def _delta(grad_out_ptr, out_ptr, rows, D: tl.constexpr):
    """delta_i = sum_d dO[i, d] O[i, d] = sum_j P[i, j] dP[i, j]."""
    delta = tl.zeros((rows.shape[0],), tl.float32)
    for d in tl.static_range(D):
        delta += _column(grad_out_ptr, rows, d, D, 1) * _column(out_ptr, rows, d, D, 1)
    return delta


@triton.jit
def _backward_query_kernel(q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, grad_out_ptr, grad_q_ptr, heads, stride_b,
                           stride_h, stride_t, stride_d, T: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    batch_head = tl.program_id(0)
    offset = (batch_head // heads) * stride_b + (batch_head % heads) * stride_h
    dense = batch_head * T * D
    row_start = tl.program_id(1) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    delta = _delta(grad_out_ptr + dense, out_ptr + dense, rows, D)
    lse = tl.load(lse_ptr + batch_head * T + rows)
    grad_q = tl.zeros((BLOCK_M, D), tl.float32)
    for key_start in range(_first_key_block(row_start, WINDOW, BLOCK_N), row_start + BLOCK_M, BLOCK_N):
        cols = key_start + tl.arange(0, BLOCK_N)
        _, grad_scores = _grad_scores(q_ptr + offset, k_ptr + offset, v_ptr + offset, grad_out_ptr + dense, rows,
                                      cols, lse, delta, stride_t, stride_d, D, WINDOW)
        for d in tl.static_range(D):
            k_d = _column(k_ptr + offset, cols, d, stride_t, stride_d)
            grad_q = _put(grad_q, d, tl.sum(grad_scores * k_d[None, :], axis=1), dims)
    tl.store(grad_q_ptr + dense + rows[:, None] * D + dims[None, :], grad_q)


@triton.jit
def _backward_key_kernel(q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, grad_out_ptr, grad_k_ptr, grad_v_ptr, heads,
                         stride_b, stride_h, stride_t, stride_d, T: tl.constexpr, D: tl.constexpr,
                         WINDOW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    batch_head = tl.program_id(0)
    offset = (batch_head // heads) * stride_b + (batch_head % heads) * stride_h
    dense = batch_head * T * D
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, D)
    grad_k = tl.zeros((BLOCK_N, D), tl.float32)
    grad_v = tl.zeros((BLOCK_N, D), tl.float32)
    # Query i sees key j only for j <= i < j + WINDOW, so the query blocks run from the block holding the first
    # key to the block holding the last such i.
    last_row = col_start + BLOCK_N + WINDOW - 2
    row_end = tl.minimum((last_row // BLOCK_M + 1) * BLOCK_M, T)
    for row_start in range((col_start // BLOCK_M) * BLOCK_M, row_end, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        delta = _delta(grad_out_ptr + dense, out_ptr + dense, rows, D)
        lse = tl.load(lse_ptr + batch_head * T + rows)
        probs, grad_scores = _grad_scores(q_ptr + offset, k_ptr + offset, v_ptr + offset, grad_out_ptr + dense,
                                          rows, cols, lse, delta, stride_t, stride_d, D, WINDOW)
        for d in tl.static_range(D):
            q_d = _column(q_ptr + offset, rows, d, stride_t, stride_d)
            grad_k = _put(grad_k, d, tl.sum(grad_scores * q_d[:, None], axis=0), dims)
            grad_v = _put(grad_v, d, tl.sum(probs * _column(grad_out_ptr + dense, rows, d, D, 1)[:, None], axis=0), dims)
    tl.store(grad_k_ptr + dense + cols[:, None] * D + dims[None, :], grad_k)
    tl.store(grad_v_ptr + dense + cols[:, None] * D + dims[None, :], grad_v)


# (query block, key block, warps) per kernel and layer type, the fastest at (640, 4, 128, 4) on a Kaggle T4.
CONFIGS = {
    "global": {"forward": (16, 128, 1), "backward_query": (32, 16, 1), "backward_key": (16, 64, 1)},
    "local": {"forward": (16, 16, 1), "backward_query": (16, 16, 1), "backward_key": (16, 16, 1)},
}


def _configs(length: int, window: int) -> dict[str, tuple[int, int, int]]:
    return CONFIGS["global" if window >= length else "local"]


def _check_shapes(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if not (query.shape == key.shape == value.shape and query.stride() == key.stride() == value.stride()):
        raise ValueError("query, key and value must have one shape and one stride layout")
    length, head_dim = query.shape[2], query.shape[3]
    for size, name in ((length, "sequence length"), (head_dim, "head_dim")):
        if size & (size - 1):
            raise ValueError(f"the {name} must be a power of two (tl.arange), got {size}")
    for kind in CONFIGS.values():
        for block_m, block_n, _ in kind.values():
            if length % block_m != 0 or length % block_n != 0:
                raise ValueError(f"the sequence length {length} must be a multiple of the blocks {block_m}, {block_n}")


@torch.library.custom_op("recursion_comparison::attention_forward", mutates_args=())
def attention_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      window: int) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shapes(query, key, value)
    batch, heads, length, head_dim = query.shape
    out = query.new_empty(batch, heads, length, head_dim)
    lse = query.new_empty(batch, heads, length)
    block_m, block_n, warps = _configs(length, window)["forward"]
    _forward_kernel[(batch * heads, length // block_m)](query, key, value, out, lse, heads, *query.stride(),
                                                        T=length, D=head_dim, WINDOW=window, BLOCK_M=block_m,
                                                        BLOCK_N=block_n, num_warps=warps)
    return out, lse


@attention_forward.register_fake
def _(query, key, value, window):
    batch, heads, length, head_dim = query.shape
    return query.new_empty(batch, heads, length, head_dim), query.new_empty(batch, heads, length)


@torch.library.custom_op("recursion_comparison::attention_backward", mutates_args=())
def attention_backward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor,
                       lse: torch.Tensor, grad_out: torch.Tensor,
                       window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, length, head_dim = query.shape
    grad_out = grad_out.contiguous()
    grad_q, grad_k, grad_v = (query.new_empty(batch, heads, length, head_dim) for _ in range(3))
    configs = _configs(length, window)
    block_m, block_n, warps = configs["backward_query"]
    _backward_query_kernel[(batch * heads, length // block_m)](query, key, value, out, lse, grad_out, grad_q, heads,
                                                               *query.stride(), T=length, D=head_dim, WINDOW=window,
                                                               BLOCK_M=block_m, BLOCK_N=block_n, num_warps=warps)
    block_m, block_n, warps = configs["backward_key"]
    _backward_key_kernel[(batch * heads, length // block_n)](query, key, value, out, lse, grad_out, grad_k, grad_v,
                                                             heads, *query.stride(), T=length, D=head_dim,
                                                             WINDOW=window, BLOCK_M=block_m, BLOCK_N=block_n,
                                                             num_warps=warps)
    return grad_q, grad_k, grad_v


@attention_backward.register_fake
def _(query, key, value, out, lse, grad_out, window):
    shape = query.shape
    return query.new_empty(shape), query.new_empty(shape), query.new_empty(shape)


def _setup_context(ctx, inputs, output):
    query, key, value, window = inputs
    out, lse = output
    ctx.save_for_backward(query, key, value, out, lse)
    ctx.window = window


def _backward(ctx, grad_out, grad_lse):
    query, key, value, out, lse = ctx.saved_tensors
    grad_q, grad_k, grad_v = attention_backward(query, key, value, out, lse, grad_out, ctx.window)
    return grad_q, grad_k, grad_v, None


attention_forward.register_autograd(_backward, setup_context=_setup_context)


def gpt_neo_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, window: int) -> torch.Tensor:
    """query, key, value: (batch, heads, T, head_dim); returns softmax(q k^T masked) v as (batch, heads, T, head_dim)."""
    return attention_forward(query, key, value, window)[0]
