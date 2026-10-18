"""One MeSH step after its layers (Yu et al. 2026, arXiv:2510.07739, Sec. 3.2) as Triton kernels: the read of the
next state from the kept outputs, and the write and read routers of the next step.

With the embeddings e, the kept outputs y_i with their write weights w_i (i <= s) and the read weights r of step s:

    x = r_0 e + sum_i <r, w_i> y_i                       (= sum_b r_b m_b of the slot buffer, without the buffer)
    w' = softmax(W_write x + b_write),  r' = softmax(W_read x + b_read)      (over the B slots, if a next step exists)

The slot weights are stored with their B columns and held in registers padded to SLOTS_PAD = 16 columns (zeros
past B), as tl.dot needs. A program owns BLOCK_M tokens; the
router weight and bias gradients are per-program partial sums that torch adds, so the result is deterministic.
tl.dot runs in IEEE fp32, as the cuBLAS matmul does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

SLOTS_PAD = 16  # tl.dot needs each dimension >= 16
# T4, speed.py forward + backward of 9 steps, 2 sweeps: 32 / 1 warp 10.05, 9.48 ms; 64 / 4 10.37, 8.46 ms, in the noise.
BLOCK_M = 32
WARPS = 1


@triton.jit
def _tile(ptr, rows, row_mask, WIDTH: tl.constexpr):
    columns = tl.arange(0, WIDTH)
    return tl.load(ptr + rows[:, None] * WIDTH + columns[None, :], mask=row_mask[:, None], other=0.0)


@triton.jit
def _store(ptr, rows, row_mask, values, WIDTH: tl.constexpr):
    columns = tl.arange(0, WIDTH)
    tl.store(ptr + rows[:, None] * WIDTH + columns[None, :], values, mask=row_mask[:, None])


@triton.jit
def _slots(ptr, rows, row_mask, SLOTS: tl.constexpr, SP: tl.constexpr):
    """(BLOCK_M, SP) slot weights of rows stored with SLOTS columns, zero past SLOTS."""
    slot = tl.arange(0, SP)
    return tl.load(ptr + rows[:, None] * SLOTS + slot[None, :], mask=row_mask[:, None] & (slot[None, :] < SLOTS),
                   other=0.0)


@triton.jit
def _store_slots(ptr, rows, row_mask, values, SLOTS: tl.constexpr, SP: tl.constexpr):
    slot = tl.arange(0, SP)
    tl.store(ptr + rows[:, None] * SLOTS + slot[None, :], values, mask=row_mask[:, None] & (slot[None, :] < SLOTS))


@triton.jit
def _router_weight(w_ptr, first_row, SLOTS: tl.constexpr, C: tl.constexpr, SP: tl.constexpr):
    """Rows first_row .. first_row + SLOTS - 1 of the (2B, C) router weight as (SP, C), zero past SLOTS."""
    slot = tl.arange(0, SP)
    columns = tl.arange(0, C)
    return tl.load(w_ptr + (first_row + slot[:, None]) * C + columns[None, :], mask=slot[:, None] < SLOTS, other=0.0)


@triton.jit
def _softmax_slots(logits, SLOTS: tl.constexpr, SP: tl.constexpr):
    slot = tl.arange(0, SP)
    logits = tl.where(slot[None, :] < SLOTS, logits, float("-inf"))
    exp = tl.exp(logits - tl.max(logits, 1)[:, None])
    return exp / tl.sum(exp, 1)[:, None]


@triton.jit
def _read_state(e_ptr, y_ptrs, w_ptrs, read, rows, row_mask, N: tl.constexpr, SLOTS: tl.constexpr, C: tl.constexpr,
                SP: tl.constexpr):
    slot = tl.arange(0, SP)
    state = tl.sum(tl.where(slot[None, :] == 0, read, 0.0), 1)[:, None] * _tile(e_ptr, rows, row_mask, C)
    for i in tl.static_range(N):
        weight = tl.sum(read * _slots(w_ptrs[i], rows, row_mask, SLOTS, SP), 1)
        state += weight[:, None] * _tile(y_ptrs[i], rows, row_mask, C)
    return state


@triton.jit
def _forward_kernel(e_ptr, y_ptrs, w_ptrs, r_ptr, rw_ptr, rb_ptr, state_ptr, write_ptr, read_ptr, M,
                    N: tl.constexpr, SLOTS: tl.constexpr, HAS_NEXT: tl.constexpr, C: tl.constexpr,
                    SP: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    state = _read_state(e_ptr, y_ptrs, w_ptrs, _slots(r_ptr, rows, row_mask, SLOTS, SP), rows, row_mask, N, SLOTS, C,
                        SP)
    _store(state_ptr, rows, row_mask, state, C)
    if HAS_NEXT:
        slot = tl.arange(0, SP)
        bias_mask = slot < SLOTS
        write_logits = tl.dot(state, tl.trans(_router_weight(rw_ptr, 0, SLOTS, C, SP)), input_precision="ieee")
        write_logits += tl.load(rb_ptr + slot, mask=bias_mask, other=0.0)[None, :]
        read_logits = tl.dot(state, tl.trans(_router_weight(rw_ptr, SLOTS, SLOTS, C, SP)), input_precision="ieee")
        read_logits += tl.load(rb_ptr + SLOTS + slot, mask=bias_mask, other=0.0)[None, :]
        _store_slots(write_ptr, rows, row_mask, _softmax_slots(write_logits, SLOTS, SP), SLOTS, SP)
        _store_slots(read_ptr, rows, row_mask, _softmax_slots(read_logits, SLOTS, SP), SLOTS, SP)


@triton.jit
def _backward_kernel(e_ptr, y_ptrs, w_ptrs, r_ptr, rw_ptr, state_ptr, write_ptr, read_ptr, grad_state_ptr,
                     grad_write_ptr, grad_read_ptr, grad_e_ptr, grad_y_ptrs, grad_w_ptrs, grad_r_ptr, grad_rw_ptr,
                     grad_rb_ptr, M, N: tl.constexpr, SLOTS: tl.constexpr, HAS_NEXT: tl.constexpr,
                     C: tl.constexpr, SP: tl.constexpr, BLOCK_M: tl.constexpr):
    program = tl.program_id(0)
    rows = program * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    slot = tl.arange(0, SP)
    columns = tl.arange(0, C)
    grad_state = _tile(grad_state_ptr, rows, row_mask, C)
    if HAS_NEXT:
        state = _tile(state_ptr, rows, row_mask, C)
        for half in tl.static_range(2):
            if half == 0:
                probs = _slots(write_ptr, rows, row_mask, SLOTS, SP)
                grad_probs = _slots(grad_write_ptr, rows, row_mask, SLOTS, SP)
            else:
                probs = _slots(read_ptr, rows, row_mask, SLOTS, SP)
                grad_probs = _slots(grad_read_ptr, rows, row_mask, SLOTS, SP)
            # softmax backward; the padded columns have probability 0, so their gradient is 0.
            grad_logits = probs * (grad_probs - tl.sum(grad_probs * probs, 1)[:, None])
            grad_state += tl.dot(grad_logits, _router_weight(rw_ptr, half * SLOTS, SLOTS, C, SP),
                                 input_precision="ieee")
            offsets = ((program * 2 + half) * SP + slot[:, None]) * C + columns[None, :]
            tl.store(grad_rw_ptr + offsets, tl.dot(tl.trans(grad_logits), state, input_precision="ieee"))
            tl.store(grad_rb_ptr + (program * 2 + half) * SP + slot, tl.sum(grad_logits, 0))
    read = _slots(r_ptr, rows, row_mask, SLOTS, SP)
    e = _tile(e_ptr, rows, row_mask, C)
    _store(grad_e_ptr, rows, row_mask, tl.sum(tl.where(slot[None, :] == 0, read, 0.0), 1)[:, None] * grad_state, C)
    grad_read = tl.where(slot[None, :] == 0, tl.sum(grad_state * e, 1)[:, None], 0.0)
    for i in tl.static_range(N):
        write = _slots(w_ptrs[i], rows, row_mask, SLOTS, SP)
        kept_grad = tl.sum(grad_state * _tile(y_ptrs[i], rows, row_mask, C), 1)
        _store(grad_y_ptrs[i], rows, row_mask, tl.sum(read * write, 1)[:, None] * grad_state, C)
        _store_slots(grad_w_ptrs[i], rows, row_mask, read * kept_grad[:, None], SLOTS, SP)
        grad_read += write * kept_grad[:, None]
    _store_slots(grad_r_ptr, rows, row_mask, grad_read, SLOTS, SP)


def _check(embeddings: torch.Tensor, kept: list[torch.Tensor], writes: list[torch.Tensor], read: torch.Tensor,
           slots: int) -> None:
    rows, width = embeddings.shape
    if embeddings.dtype != torch.float32 or not embeddings.is_cuda:
        raise ValueError("the MeSH kernels take fp32 CUDA tensors")
    if width < 16 or width & (width - 1) or not 0 < slots <= SLOTS_PAD:
        raise ValueError(f"width {width} must be a power of 2 >= 16 (tl.dot) and slots {slots} in 1..{SLOTS_PAD}")
    if len(kept) != len(writes) or any(t.shape != (rows, width) for t in kept) or \
            any(t.shape != (rows, slots) for t in [*writes, read]):
        raise ValueError("kept outputs (M, C) and slot weights (M, B), one write per kept output")


@torch.library.custom_op("recursion_comparison::mesh_step", mutates_args=())
def mesh_step(embeddings: torch.Tensor, kept: list[torch.Tensor], writes: list[torch.Tensor], read: torch.Tensor,
              router_weight: torch.Tensor, router_bias: torch.Tensor, slots: int,
              has_next: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """embeddings and kept outputs (M, C); writes and read (M, B); router_weight (2B, C) and router_bias (2B,) of
    the next step. Returns the next state (M, C) and, if has_next, its write and read weights (M, B)."""
    _check(embeddings, kept, writes, read, slots)
    rows, width = embeddings.shape
    state = torch.empty_like(embeddings)
    next_rows = rows if has_next else 0
    next_write, next_read = embeddings.new_empty(next_rows, slots), embeddings.new_empty(next_rows, slots)
    _forward_kernel[(triton.cdiv(rows, BLOCK_M),)](
        embeddings.contiguous(), tuple(t.contiguous() for t in kept), tuple(t.contiguous() for t in writes),
        read.contiguous(), router_weight.contiguous(), router_bias.contiguous(), state, next_write, next_read, rows,
        N=len(kept), SLOTS=slots, HAS_NEXT=has_next, C=width, SP=SLOTS_PAD, BLOCK_M=BLOCK_M, num_warps=WARPS)
    return state, next_write, next_read


@mesh_step.register_fake
def _(embeddings, kept, writes, read, router_weight, router_bias, slots, has_next):
    rows = embeddings.shape[0] if has_next else 0
    return torch.empty_like(embeddings), embeddings.new_empty(rows, slots), embeddings.new_empty(rows, slots)


@torch.library.custom_op("recursion_comparison::mesh_step_backward", mutates_args=())
def mesh_step_backward(embeddings: torch.Tensor, kept: list[torch.Tensor], writes: list[torch.Tensor],
                       read: torch.Tensor, router_weight: torch.Tensor, state: torch.Tensor, next_write: torch.Tensor,
                       next_read: torch.Tensor, grad_state: torch.Tensor, grad_write: torch.Tensor,
                       grad_read: torch.Tensor, slots: int, has_next: bool) -> list[torch.Tensor]:
    rows, width = embeddings.shape
    programs = triton.cdiv(rows, BLOCK_M)
    grad_embeddings, grad_r = torch.empty_like(embeddings), torch.empty_like(read)
    grad_kept = [torch.empty_like(t) for t in kept]
    grad_writes = [torch.empty_like(t) for t in writes]
    grad_rw = embeddings.new_zeros(programs, 2, SLOTS_PAD, width)
    grad_rb = embeddings.new_zeros(programs, 2, SLOTS_PAD)
    _backward_kernel[(programs,)](
        embeddings.contiguous(), tuple(t.contiguous() for t in kept), tuple(t.contiguous() for t in writes),
        read.contiguous(), router_weight.contiguous(), state, next_write, next_read, grad_state.contiguous(),
        grad_write.contiguous(), grad_read.contiguous(), grad_embeddings, tuple(grad_kept), tuple(grad_writes), grad_r,
        grad_rw, grad_rb, rows, N=len(kept), SLOTS=slots, HAS_NEXT=has_next, C=width, SP=SLOTS_PAD, BLOCK_M=BLOCK_M,
        num_warps=WARPS)
    grad_router_weight = grad_rw.sum(0)[:, :slots].reshape(2 * slots, width)
    grad_router_bias = grad_rb.sum(0)[:, :slots].reshape(2 * slots)
    return [grad_embeddings, grad_r, grad_router_weight, grad_router_bias, *grad_kept, *grad_writes]


@mesh_step_backward.register_fake
def _(embeddings, kept, writes, read, router_weight, state, next_write, next_read, grad_state, grad_write, grad_read,
      slots, has_next):
    return [torch.empty_like(embeddings), torch.empty_like(read), torch.empty_like(router_weight),
            router_weight.new_empty(router_weight.shape[0]), *[torch.empty_like(t) for t in kept],
            *[torch.empty_like(t) for t in writes]]


def _setup_context(ctx, inputs, output):
    embeddings, kept, writes, read, router_weight, router_bias, slots, has_next = inputs
    state, next_write, next_read = output
    ctx.save_for_backward(embeddings, read, router_weight, state, next_write, next_read, *kept, *writes)
    ctx.count, ctx.slots, ctx.has_next = len(kept), slots, has_next


def _backward(ctx, grad_state, grad_write, grad_read):
    embeddings, read, router_weight, state, next_write, next_read, *rest = ctx.saved_tensors
    kept, writes = rest[:ctx.count], rest[ctx.count:]
    if grad_write is None or not ctx.has_next:
        grad_write = torch.zeros_like(next_write)
    if grad_read is None or not ctx.has_next:
        grad_read = torch.zeros_like(next_read)
    grads = mesh_step_backward(embeddings, kept, writes, read, router_weight, state, next_write, next_read,
                               grad_state, grad_write, grad_read, ctx.slots, ctx.has_next)
    grad_embeddings, grad_r, grad_router_weight, grad_router_bias = grads[:4]
    grad_kept, grad_writes = grads[4:4 + ctx.count], grads[4 + ctx.count:]
    if not ctx.has_next:
        grad_router_weight, grad_router_bias = None, None
    return grad_embeddings, list(grad_kept), list(grad_writes), grad_r, grad_router_weight, grad_router_bias, None, None


mesh_step.register_autograd(_backward, setup_context=_setup_context)
