"""Runnable check of the Parallax arm on real tokens: `python -m parallax.check [--heads H D]` from the repo root.

It fails if the Triton kernel or the PyTorch path disagrees with the paper layer written as eq. 8 (the covariance
form, arXiv:2605.29157 Sec. 3.1) with Qwen3 RMSNorm and RoPE from transformers, if the core op disagrees with the
authors' Algorithm 1 reference in fla, if the layer with W_R = 0 is not plain causal softmax attention, if an output
or a gradient sees a later token, or if a parameter gets no gradient or a value is not finite in a short real
Muon + AdamW run.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from fla.ops.parallax.naive import naive_parallax
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402
from model import build_model  # noqa: E402
from parallax import attention, kernel  # noqa: E402
from parallax.attention import Parallax  # noqa: E402

REL_TOL = 1e-5  # fp32 layer against the fp64 reference
# The reference uses the paper values, not the layer's constants, so a wrong constant in the layer fails.
ROPE_THETA = 1e6  # Sec. 11.1 and Table 5
KV_GROUP = 2  # Table 5: 16 query heads share 8 KV heads
DEVICE = torch.device("cuda")


def require(condition: bool, detail) -> None:
    if not condition:
        raise AssertionError(detail)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double(), expected.detach().double()
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-30))


def reference_heads(layer: Parallax, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """q, k, v, rho in fp64 as (B, H, T, d), with K/V repeated over each query group as in Qwen3 repeat_kv."""
    batch, length, d_model = x.shape
    n_heads, head_dim = layer.n_heads, layer.head_dim
    config = Qwen3Config(hidden_size=d_model, num_attention_heads=n_heads, num_key_value_heads=n_heads // KV_GROUP,
                         head_dim=head_dim, rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA})
    positions = torch.arange(length, device=x.device)[None].expand(batch, -1)
    cos, sin = Qwen3RotaryEmbedding(config, device=x.device)(x.double(), positions)

    def heads(linear: torch.nn.Linear, norm: torch.nn.RMSNorm | None) -> torch.Tensor:
        z = (x.double() @ linear.weight.double().T).view(batch, length, -1, head_dim)
        if norm is not None:
            z = norm.weight.double() * z / (z.square().mean(-1, keepdim=True) + norm.eps).sqrt()
        return z.transpose(1, 2)

    q, k = apply_rotary_pos_emb(heads(layer.q, layer.q_norm), heads(layer.k, layer.k_norm), cos, sin)
    probe = heads(layer.r, layer.r_norm)
    rho, _ = apply_rotary_pos_emb(probe, probe, cos, sin)  # as modeling_parallax.py: rho takes the query's cos/sin
    v = heads(layer.v, None)
    k, v = k.repeat_interleave(KV_GROUP, dim=1), v.repeat_interleave(KV_GROUP, dim=1)
    return q, k, v, rho


def reference_layer(layer: Parallax, x: torch.Tensor) -> torch.Tensor:
    """Eq. 8, eta_i = 0: o_i = vbar_i - Sigma_KV^(i) rho_i, Sigma_KV^(i) = sum_j p_ij (v_j - vbar_i)(k_j - kbar_i)^T."""
    batch, length, _ = x.shape
    q, k, v, rho = reference_heads(layer, x)
    future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
    p = torch.softmax((q @ k.transpose(-1, -2) / math.sqrt(layer.head_dim)).masked_fill(future, float("-inf")), -1)
    vbar, kbar = p @ v, p @ k
    v_centred = v[:, :, None] - vbar[:, :, :, None]
    k_centred = k[:, :, None] - kbar[:, :, :, None]
    sigma = torch.einsum("bhij,bhijv,bhijk->bhivk", p, v_centred, k_centred)
    o = vbar - torch.einsum("bhivk,bhik->bhiv", sigma, rho)
    return o.transpose(1, 2).reshape(batch, length, -1) @ layer.o.weight.double().T


def check_constants() -> None:
    require((attention.ROPE_THETA, attention.KV_GROUP) == (ROPE_THETA, KV_GROUP),
            (attention.ROPE_THETA, attention.KV_GROUP))


def check_init(model: torch.nn.Module) -> None:
    """Fails if the arm gets an absolute position embedding or the layer leaves the authors' Qwen3 init."""
    require(model.wpe is None, "the Parallax arm must have no absolute position embedding")
    layers = [block.attn for block in model.blocks]
    require(all(linear.bias is None for layer in layers for linear in (layer.q, layer.k, layer.v, layer.r, layer.o)),
            "a projection has a bias")
    r_std = float(torch.cat([layer.r.weight.detach().flatten() for layer in layers]).std())
    require(0.018 < r_std < 0.022, r_std)  # N(0, 0.02) +-10%, more than 5 standard errors at the budget shapes
    require(all(bool((norm.weight == 1.0).all()) for layer in layers
                for norm in (layer.q_norm, layer.k_norm, layer.r_norm)), "a q/k/r RMSNorm weight is not one")
    print(f"init: no position embedding, no biases, W_R std {r_std:.4f} (0.02), q/k/r RMSNorm weight 1")


def check_core(name: str, batch: int, length: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
    """Triton kernel and PyTorch core against fla's fp32 Algorithm 1 reference and the fp64 PyTorch core."""
    shape_q, shape_kv = (batch, length, n_heads, head_dim), (batch, length, n_kv_heads, head_dim)
    q, rho = (torch.randn(shape_q, device=DEVICE) for _ in range(2))
    k, v = (torch.randn(shape_kv, device=DEVICE) for _ in range(2))
    upstream = torch.randn(shape_q, device=DEVICE)
    inputs = [t.requires_grad_() for t in (q, k, v, rho)]
    exact = [t.detach().double().requires_grad_() for t in inputs]
    expected = kernel.parallax_core_torch(*exact)
    expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), exact)
    algorithm_1 = naive_parallax(q.detach(), rho.detach(), k.detach(), v.detach())
    require(rel_err(algorithm_1, expected) < REL_TOL, (name, "fla naive_parallax", rel_err(algorithm_1, expected)))
    for path, core in (("Triton", kernel._ParallaxCore.apply), ("PyTorch", kernel.parallax_core_torch)):
        out = core(*inputs)
        grads = torch.autograd.grad((out * upstream).sum(), inputs)
        errors = [rel_err(out, expected)] + [rel_err(a, b) for a, b in zip(grads, expected_grads, strict=True)]
        print(f"core {path} vs fp64 [{name}]: rel err o/dq/dk/dv/drho " + "/".join(f"{e:.1e}" for e in errors))
        require(max(errors) < REL_TOL, (name, path, errors))


def check_layer(name: str, layer: Parallax, x: torch.Tensor) -> None:
    x = x.detach().requires_grad_()
    params = [x] + list(layer.parameters())
    upstream = torch.randn(x.shape, device=x.device)
    expected = reference_layer(layer, x)
    expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), params)
    names = ["input"] + [n for n, _ in layer.named_parameters()]
    for path, use_kernel in (("Triton", True), ("PyTorch", False)):
        with mock.patch.object(kernel, "uses_kernel", lambda tensor, forced=use_kernel: forced):
            out = layer(x)
            grads = torch.autograd.grad((out * upstream).sum(), params)
        forward_err = rel_err(out, expected)
        grad_errs = {n: rel_err(a, b) for n, a, b in zip(names, grads, expected_grads, strict=True)}
        worst = max(grad_errs, key=grad_errs.get)
        print(f"layer {path} vs eq. 8 reference [{name}]: fwd rel err {forward_err:.1e}, "
              f"worst grad rel err {grad_errs[worst]:.1e} ({worst})")
        require(forward_err < REL_TOL and grad_errs[worst] < REL_TOL, (name, path, forward_err, grad_errs))


def check_softmax_limit(layer: Parallax, x: torch.Tensor) -> None:
    """With W_R = 0 the correction vanishes and the layer is Qwen3 causal softmax attention."""
    with torch.no_grad():
        saved = layer.r.weight.clone()
        layer.r.weight.zero_()
        out = layer(x)
        q, k, v, _ = reference_heads(layer, x)
        heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        expected = heads.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ layer.o.weight.double().T
        layer.r.weight.copy_(saved)
    error = rel_err(out, expected)
    print(f"W_R = 0 vs causal softmax attention: rel err {error:.1e}")
    require(error < REL_TOL, error)


def check_causal(layer: Parallax, x: torch.Tensor) -> None:
    with torch.no_grad():
        base = layer(x)
        for t0 in (0, 1, 31, 32, 127):
            changed = x.clone()
            changed[:, t0] += torch.randn_like(changed[:, t0])
            out = layer(changed)
            leak = float((out[:, :t0] - base[:, :t0]).abs().max()) if t0 > 0 else 0.0
            require(leak == 0.0, (t0, leak))
            require(float((out[:, t0] - base[:, t0]).abs().max()) > 0.0, t0)
    for t0 in (0, 31, 32, 126):
        probe = x.detach().requires_grad_()
        (grad,) = torch.autograd.grad(layer(probe)[:, t0].sum(), probe)
        leak = float(grad[:, t0 + 1:].abs().max())
        require(leak == 0.0, (t0, leak))
        require(float(grad[:, t0].abs().max()) > 0.0, t0)
    print("causality: outputs before a changed token are bitwise unchanged at t0 in {0, 1, 31, 32, 127}, "
          "and the gradient of output t0 is exactly zero on every later token")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, nargs=2, metavar=("N_HEADS", "HEAD_DIM"),
                        default=train.HEAD_CONFIG["parallax"])
    n_heads, head_dim = parser.parse_args().heads

    check_constants()
    torch.manual_seed(0)
    check_core("GQA 2:1, T 128", 4, 128, 2, 1, head_dim)
    check_core("GQA 2:1, 4 heads, T 70", 2, 70, 4, 2, head_dim)
    check_core("MHA, d 16, T 33", 2, 33, 2, 2, 16)

    model = build_model("parallax", 512, train.D_MODEL, train.D_FF, train.NUM_LAYERS, n_heads, head_dim,
                        chunk=train.CHUNK, context_length=train.CONTEXT_LENGTH).to(DEVICE)
    print(f"parallax ({n_heads}, {head_dim}): {sum(p.numel() for p in model.parameters())} params")
    check_init(model)

    T = train.CONTEXT_LENGTH
    train_ids = torch.load(train.ROOT / "packed_cache" / "train.pt")
    eval_ids = torch.load(train.ROOT / "packed_cache" / "validation.pt")
    eval_batch = eval_ids[: 8 * T].view(8, T).long().to(DEVICE)

    def layer_input(block_index: int) -> torch.Tensor:
        with torch.no_grad():
            h = model.wte(eval_batch)
            for block in model.blocks[:block_index]:
                h = block(h)
            return model.blocks[block_index].ln_1(h)

    layer, x = model.blocks[0].attn, layer_input(0)
    check_layer("block 0 at init", layer, x)
    check_softmax_limit(layer, x)
    check_causal(layer, x)

    optimizers, schedulers = train._make_optimizers(model, total_steps=10_000)
    generator = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(30):
        rows = torch.randint(0, train_ids.shape[0] // T, (64,), generator=generator)
        batch = torch.stack([train_ids[r * T:(r + 1) * T] for r in rows.tolist()]).long().to(DEVICE)
        _, loss = model(batch, labels=batch)
        loss.backward()
        require(bool(torch.isfinite(loss)), losses)
        losses.append(float(loss.detach()))
        zero_grad = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
        require(not zero_grad, zero_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    non_finite = [n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
    require(not non_finite, non_finite)
    print(f"30 real Muon + AdamW steps at batch 64: loss {losses[0]:.4f} (ln 512 = {math.log(512):.4f}) -> "
          f"{losses[-1]:.4f}, every parameter had a nonzero gradient at every step and all parameters are finite")
    for block_index in (0, train.NUM_LAYERS - 1):
        check_layer(f"block {block_index} after 30 steps", model.blocks[block_index].attn, layer_input(block_index))
    print("Parallax check passed")


if __name__ == "__main__":
    main()
