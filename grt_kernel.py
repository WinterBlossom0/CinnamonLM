"""The gated state update of a Gated Recurrent Transformer pass as Triton kernels, as the official code computes it
(github.com/Amr-Hegazy1/gated-recurrent-transformer, model.py; arXiv:2608.15062, Eqs. 4-5). With h the state, o
the output of the looped layers, n = LN_h(h_pre) (the same on every pass) and eps_g the gate noise:

    u = LN_2C([LN_x(h); n]),   g = sigmoid(W_2 silu(W_1 u) + b_2 + eps_g),   h <- g h + (1 - g) o

eps_g is one value per token for all channels, as the code draws it, and is optional (no noise in eval mode). It
is a constant input, so it gets no gradient; the backward recomputes g with it. All LayerNorms have a gain and no
bias, and W_1 has no bias, as in the code (bias = False). One forward and one
backward kernel replace the norms, the two small matmuls (four in the backward) and the elementwise steps. A program
owns BLOCK_M tokens; the weight, bias and gain gradients are per-program partial sums that torch adds, so the result
is deterministic. tl.dot runs in IEEE fp32, as the cuBLAS matmul does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# T4, speed.py forward + backward at 81,920 rows, 2 sweeps: 64 / 4 warps 0.7256, 0.7711 ms; 32 / 1 1.1478, 0.7409 ms.
BLOCK_M = 64
WARPS = 4


@triton.jit
def _tile(ptr, rows, row_mask, C: tl.constexpr):
    columns = tl.arange(0, C)
    return tl.load(ptr + rows[:, None] * C + columns[None, :], mask=row_mask[:, None], other=0.0)


@triton.jit
def _store(ptr, rows, row_mask, values, C: tl.constexpr):
    columns = tl.arange(0, C)
    tl.store(ptr + rows[:, None] * C + columns[None, :], values, mask=row_mask[:, None])


@triton.jit
def _matrix(ptr, row_stride, column_offset, C: tl.constexpr):
    """A (C, C) block of a matrix whose rows are row_stride apart, from column column_offset."""
    index = tl.arange(0, C)
    return tl.load(ptr + index[:, None] * row_stride + column_offset + index[None, :])


@triton.jit
def _normalize(values, C: tl.constexpr, eps):
    centered = values - (tl.sum(values, 1) / C)[:, None]
    rstd = tl.rsqrt(tl.sum(centered * centered, 1) / C + eps)
    return centered * rstd[:, None], rstd


@triton.jit
def _gate(state, normed_input, wx_ptr, w2c_ptr, w1_ptr, w2_ptr, b2_ptr, noise_ptr, rows, row_mask, eps,
          C: tl.constexpr, HAS_GATE_NOISE: tl.constexpr):
    columns = tl.arange(0, C)
    state_hat, state_rstd = _normalize(state, C, eps)
    first = state_hat * tl.load(wx_ptr + columns)[None, :]
    # The 2C-wide LayerNorm over [first; normed_input], from the sums of both halves.
    mean = (tl.sum(first, 1) + tl.sum(normed_input, 1)) / (2 * C)
    first_centered = first - mean[:, None]
    second_centered = normed_input - mean[:, None]
    joint_rstd = tl.rsqrt((tl.sum(first_centered * first_centered, 1)
                           + tl.sum(second_centered * second_centered, 1)) / (2 * C) + eps)
    first_hat = first_centered * joint_rstd[:, None]
    second_hat = second_centered * joint_rstd[:, None]
    first_u = first_hat * tl.load(w2c_ptr + columns)[None, :]
    second_u = second_hat * tl.load(w2c_ptr + C + columns)[None, :]
    pre_hidden = (tl.dot(first_u, tl.trans(_matrix(w1_ptr, 2 * C, 0, C)), input_precision="ieee")
                  + tl.dot(second_u, tl.trans(_matrix(w1_ptr, 2 * C, C, C)), input_precision="ieee"))
    hidden = pre_hidden * tl.sigmoid(pre_hidden)
    gate_logit = tl.dot(hidden, tl.trans(_matrix(w2_ptr, C, 0, C)), input_precision="ieee")
    gate_logit += tl.load(b2_ptr + columns)[None, :]
    if HAS_GATE_NOISE:
        # tau = 1, so eps_g is added to the output of f_g (Eq. 5).
        gate_logit += tl.load(noise_ptr + rows, mask=row_mask, other=0.0)[:, None]
    # 1 - g as sigmoid(-x): g starts near sigmoid(4) = 0.98, where 1 - g in fp32 cancels and turns the small bias of
    # Triton's fast exp into a coherent gradient error.
    return (state_hat, state_rstd, first_hat, second_hat, joint_rstd, first_u, second_u, pre_hidden, hidden,
            tl.sigmoid(gate_logit), tl.sigmoid(-gate_logit))


@triton.jit
def _forward_kernel(s_ptr, o_ptr, n_ptr, wx_ptr, w2c_ptr, w1_ptr, w2_ptr, b2_ptr, noise_ptr, out_ptr, M, eps,
                    C: tl.constexpr, BLOCK_M: tl.constexpr, HAS_GATE_NOISE: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    state = _tile(s_ptr, rows, row_mask, C)
    _, _, _, _, _, _, _, _, _, gate, complement = _gate(state, _tile(n_ptr, rows, row_mask, C), wx_ptr, w2c_ptr,
                                                        w1_ptr, w2_ptr, b2_ptr, noise_ptr, rows, row_mask, eps, C,
                                                        HAS_GATE_NOISE)
    _store(out_ptr, rows, row_mask, gate * state + complement * _tile(o_ptr, rows, row_mask, C), C)


@triton.jit
def _layer_norm_backward(grad_hat, hat, rstd, WIDTH: tl.constexpr):
    """Gradient of the input of x_hat = (x - mean) * rstd over WIDTH features: rstd (g - mean(g) - x_hat mean(g x_hat))."""
    return rstd[:, None] * (grad_hat - (tl.sum(grad_hat, 1) / WIDTH)[:, None]
                            - hat * (tl.sum(grad_hat * hat, 1) / WIDTH)[:, None])


@triton.jit
def _backward_kernel(s_ptr, o_ptr, n_ptr, wx_ptr, w2c_ptr, w1_ptr, w2_ptr, b2_ptr, noise_ptr, grad_new_ptr,
                     grad_s_ptr, grad_o_ptr, grad_n_ptr, grad_wx_ptr, grad_w2c_ptr, grad_w1_ptr, grad_w2_ptr,
                     grad_b2_ptr, M, eps, C: tl.constexpr, BLOCK_M: tl.constexpr, HAS_GATE_NOISE: tl.constexpr):
    program = tl.program_id(0)
    rows = program * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    columns = tl.arange(0, C)
    state = _tile(s_ptr, rows, row_mask, C)
    output = _tile(o_ptr, rows, row_mask, C)
    (state_hat, state_rstd, first_hat, second_hat, joint_rstd, first_u, second_u, pre_hidden, hidden, gate,
     complement) = _gate(state, _tile(n_ptr, rows, row_mask, C), wx_ptr, w2c_ptr, w1_ptr, w2_ptr, b2_ptr, noise_ptr,
                         rows, row_mask, eps, C, HAS_GATE_NOISE)
    grad_new = _tile(grad_new_ptr, rows, row_mask, C)
    _store(grad_o_ptr, rows, row_mask, grad_new * complement, C)
    grad_logit = grad_new * (state - output) * gate * complement
    matrix = program * C * C + columns[:, None] * C + columns[None, :]
    tl.store(grad_b2_ptr + program * C + columns, tl.sum(grad_logit, 0))
    tl.store(grad_w2_ptr + matrix, tl.dot(tl.trans(grad_logit), hidden, input_precision="ieee"))
    sigmoid = tl.sigmoid(pre_hidden)
    grad_pre = (tl.dot(grad_logit, _matrix(w2_ptr, C, 0, C), input_precision="ieee")
                * sigmoid * (1.0 + pre_hidden * (1.0 - sigmoid)))
    # W_1 is (C, 2C): its two (C, C) halves act on the two halves of u.
    halves = program * 2 * C * C + columns[:, None] * C + columns[None, :]
    tl.store(grad_w1_ptr + halves, tl.dot(tl.trans(grad_pre), first_u, input_precision="ieee"))
    tl.store(grad_w1_ptr + halves + C * C, tl.dot(tl.trans(grad_pre), second_u, input_precision="ieee"))
    grad_first_u = tl.dot(grad_pre, _matrix(w1_ptr, 2 * C, 0, C), input_precision="ieee")
    grad_second_u = tl.dot(grad_pre, _matrix(w1_ptr, 2 * C, C, C), input_precision="ieee")
    tl.store(grad_w2c_ptr + program * 2 * C + columns, tl.sum(grad_first_u * first_hat, 0))
    tl.store(grad_w2c_ptr + program * 2 * C + C + columns, tl.sum(grad_second_u * second_hat, 0))
    grad_first_hat = grad_first_u * tl.load(w2c_ptr + columns)[None, :]
    grad_second_hat = grad_second_u * tl.load(w2c_ptr + C + columns)[None, :]
    # Backward of the 2C-wide LayerNorm: its mean terms run over both halves.
    mean_grad = (tl.sum(grad_first_hat, 1) + tl.sum(grad_second_hat, 1)) / (2 * C)
    mean_grad_hat = (tl.sum(grad_first_hat * first_hat, 1) + tl.sum(grad_second_hat * second_hat, 1)) / (2 * C)
    grad_first = joint_rstd[:, None] * (grad_first_hat - mean_grad[:, None] - first_hat * mean_grad_hat[:, None])
    grad_second = joint_rstd[:, None] * (grad_second_hat - mean_grad[:, None] - second_hat * mean_grad_hat[:, None])
    _store(grad_n_ptr, rows, row_mask, grad_second, C)
    tl.store(grad_wx_ptr + program * C + columns, tl.sum(grad_first * state_hat, 0))
    grad_state = _layer_norm_backward(grad_first * tl.load(wx_ptr + columns)[None, :], state_hat, state_rstd, C)
    _store(grad_s_ptr, rows, row_mask, grad_state + grad_new * gate, C)


def _check_shapes(state: torch.Tensor, w1: torch.Tensor, gate_noise: torch.Tensor | None) -> None:
    rows, width = state.shape
    if state.dtype != torch.float32 or not state.is_cuda:
        raise ValueError("the GRT kernels take fp32 CUDA tensors")
    if width < 16 or width & (width - 1):
        raise ValueError(f"width {width} must be a power of 2 >= 16 (tl.dot)")
    if w1.shape != (width, 2 * width):
        raise ValueError(f"W_1 {tuple(w1.shape)}, expected ({width}, {2 * width})")
    if gate_noise is not None and (gate_noise.shape != (rows,) or gate_noise.dtype != torch.float32):
        raise ValueError(f"gate noise {tuple(gate_noise.shape)} {gate_noise.dtype}, expected ({rows},) float32")


def _noise_pointer(gate_noise: torch.Tensor | None, b2: torch.Tensor) -> torch.Tensor:
    # Without noise HAS_GATE_NOISE = False removes the load, and b2 only fills the pointer argument.
    return b2 if gate_noise is None else gate_noise.contiguous()


@torch.library.custom_op("recursion_comparison::grt_update", mutates_args=())
def grt_update(state: torch.Tensor, output: torch.Tensor, normed_input: torch.Tensor, state_gain: torch.Tensor,
               joint_gain: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
               eps: float, gate_noise: torch.Tensor | None = None) -> torch.Tensor:
    """state, output, normed_input = LN_h(h_pre): (M, C); state_gain (C,), joint_gain (2C,), W_1 (C, 2C), W_2 (C, C),
    b_2 (C,), gate_noise (M,) or None. Returns g * state + (1 - g) * output."""
    _check_shapes(state, w1, gate_noise)
    rows, width = state.shape
    new_state = torch.empty_like(state)
    _forward_kernel[(triton.cdiv(rows, BLOCK_M),)](
        state.contiguous(), output.contiguous(), normed_input.contiguous(), state_gain, joint_gain, w1.contiguous(),
        w2.contiguous(), b2, _noise_pointer(gate_noise, b2), new_state, rows, eps, C=width, BLOCK_M=BLOCK_M,
        HAS_GATE_NOISE=gate_noise is not None, num_warps=WARPS)
    return new_state


@grt_update.register_fake
def _(state, output, normed_input, state_gain, joint_gain, w1, w2, b2, eps, gate_noise=None):
    return torch.empty_like(state)


@torch.library.custom_op("recursion_comparison::grt_update_backward", mutates_args=())
def grt_update_backward(state: torch.Tensor, output: torch.Tensor, normed_input: torch.Tensor,
                        state_gain: torch.Tensor, joint_gain: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                        b2: torch.Tensor, grad_new: torch.Tensor, eps: float,
                        gate_noise: torch.Tensor | None = None) -> list[torch.Tensor]:
    _check_shapes(state, w1, gate_noise)
    rows, width = state.shape
    programs = triton.cdiv(rows, BLOCK_M)
    grad_state, grad_output, grad_normed = (torch.empty_like(state) for _ in range(3))
    grad_wx, grad_b2 = state.new_empty(programs, width), state.new_empty(programs, width)
    grad_w2c = state.new_empty(programs, 2 * width)
    grad_w1 = state.new_empty(programs, 2, width, width)
    grad_w2 = state.new_empty(programs, width, width)
    _backward_kernel[(programs,)](
        state.contiguous(), output.contiguous(), normed_input.contiguous(), state_gain, joint_gain, w1.contiguous(),
        w2.contiguous(), b2, _noise_pointer(gate_noise, b2), grad_new.contiguous(), grad_state, grad_output,
        grad_normed, grad_wx, grad_w2c, grad_w1, grad_w2, grad_b2, rows, eps, C=width, BLOCK_M=BLOCK_M,
        HAS_GATE_NOISE=gate_noise is not None, num_warps=WARPS)
    # (half, j, c) -> W_1[j, half * C + c]
    grad_w1 = grad_w1.sum(0).permute(1, 0, 2).reshape(width, 2 * width).contiguous()
    return [grad_state, grad_output, grad_normed, grad_wx.sum(0), grad_w2c.sum(0), grad_w1, grad_w2.sum(0),
            grad_b2.sum(0)]


@grt_update_backward.register_fake
def _(state, output, normed_input, state_gain, joint_gain, w1, w2, b2, grad_new, eps, gate_noise=None):
    return [torch.empty_like(state), torch.empty_like(state), torch.empty_like(state), torch.empty_like(state_gain),
            torch.empty_like(joint_gain), torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(b2)]


def _setup_context(ctx, inputs, output):
    *tensors, eps, gate_noise = inputs
    ctx.save_for_backward(*tensors, gate_noise)
    ctx.eps = eps


def _backward(ctx, grad_new):
    *tensors, gate_noise = ctx.saved_tensors
    # No gradient for eps and for the noise, which is a constant.
    return (*grt_update_backward(*tensors, grad_new, ctx.eps, gate_noise), None, None)


grt_update.register_autograd(_backward, setup_context=_setup_context)
