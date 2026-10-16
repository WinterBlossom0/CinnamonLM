"""Runnable check of the OSDN arm on real tokens: `python -m osdn.check [--heads H D]` from the repo root.

It fails if the phase-1 kernel, the chunked layer (Triton scan and the PyTorch scan that GPUs below sm_80 use)
or the causal mask disagrees with a plain token recurrence, or if a parameter gets no gradient or a value is not
finite in a short real Muon + AdamW run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402
from kda import scan_triton  # noqa: E402
from model import build_model  # noqa: E402
from osdn import precondition_triton  # noqa: E402
from osdn.attention import OSDN  # noqa: E402
from osdn.precondition_triton import osdn_precondition  # noqa: E402

REL_TOL = 1e-5  # two fp32 evaluation orders of one computation
# The reference uses the source values, not the layer's constants, so a wrong constant in the layer fails.
ETA, D_MIN, D_MAX, EPS = 0.003, 0.5, 2.0, 1e-6  # paper App. C Listing 1 and App. F
CONV_SIZE = 4  # 340M config conv_size
L2_EPS = 1e-6  # fla l2norm
DEVICE = torch.device("cuda")


def require(condition: bool, detail) -> None:
    if not condition:
        raise AssertionError(detail)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach(), expected.detach()
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-30))


def check_init(model: torch.nn.Module) -> None:
    """Fails if the model does not have the layer-specific init (PyTorch's default conv init has std 0.29 here)."""
    layers = [block.attn for block in model.blocks]
    conv_weights = torch.cat([conv.weight.detach().flatten() for layer in layers
                              for conv in (layer.q_conv, layer.k_conv, layer.v_conv)])
    conv_std = float(conv_weights.std())
    require(0.018 < conv_std < 0.022, conv_std)  # fla N(0, 0.02) +-10%, about 4 standard errors at 960 weights
    require(all(layer.beta.bias is None for layer in layers), "beta projection has a bias; fla's b_proj has none")
    require(all(bool((layer.on.weight == 1.0).all()) for layer in layers), "output norm weight is not one")
    print(f"init: conv weight std {conv_std:.4f} (fla 0.02), output norm weight 1")


def precondition_plain(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Alg. 1 with the box projection, as a token loop; k is (B,T,H,D), beta is (B,T,H)."""
    B, T, H, D = k.shape
    d = k.new_ones(B, H, D)
    trajectory = []
    for t in range(T):
        trajectory.append(d)
        k_sq = k[:, t].square()
        b = beta[:, t, :, None]
        n = k_sq.sum(-1, keepdim=True).clamp_min(EPS)
        d = (d + ETA * b * (1.0 - b * (d * k_sq).sum(-1, keepdim=True)) / n * k_sq).clamp(D_MIN, D_MAX)
    return torch.stack(trajectory, 1)


def reference_layer(layer: OSDN, x: torch.Tensor) -> torch.Tensor:
    """The layer as a plain token recurrence (eq. 5), computed from the layer weights without OSDN's methods."""
    B, T, _ = x.shape
    H, D = layer.h, layer.d

    def conv_silu(linear, conv):
        lin = F.pad(x @ linear.weight.T, (0, 0, CONV_SIZE - 1, 0))
        taps = conv.weight[:, 0]
        return F.silu(sum(taps[:, j] * lin[:, j:j + T] for j in range(CONV_SIZE))).reshape(B, T, H, D)

    q, k, v = (conv_silu(layer.q, layer.q_conv), conv_silu(layer.k, layer.k_conv),
               conv_silu(layer.v, layer.v_conv))
    q = q / (q.square().sum(-1, keepdim=True) + L2_EPS).sqrt() * D ** -0.5
    k = k / (k.square().sum(-1, keepdim=True) + L2_EPS).sqrt()
    beta = torch.sigmoid(x @ layer.beta.weight.T)
    d = precondition_plain(k, beta)

    state = x.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        residual = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        write_key = d[:, t] * k[:, t]
        state = state + beta[:, t, :, None, None] * write_key[..., :, None] * residual[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    o = torch.stack(outs, 1)
    o = o / (o.square().mean(-1, keepdim=True) + layer.on.eps).sqrt() * layer.on.weight
    return o.reshape(B, T, H * D) @ layer.o.weight.T


def check_precondition_kernel(name: str, k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    k, beta = k.detach().requires_grad_(), beta.detach().requires_grad_()
    upstream = torch.randn_like(k)
    kernel_d = osdn_precondition(k, beta)
    kernel_grads = torch.autograd.grad((kernel_d * upstream).sum(), [k, beta])
    plain_d = precondition_plain(k, beta)
    plain_grads = torch.autograd.grad((plain_d * upstream).sum(), [k, beta])
    plain_d = plain_d.detach()
    errors = [rel_err(kernel_d, plain_d)] + [rel_err(a, b) for a, b in zip(kernel_grads, plain_grads)]
    print(f"phase-1 kernel vs loop [{name}]: d range [{float(plain_d.min()):.4f}, {float(plain_d.max()):.4f}], "
          f"rel err d/grad_k/grad_beta {errors[0]:.1e}/{errors[1]:.1e}/{errors[2]:.1e}")
    require(max(errors) < REL_TOL, (name, errors))
    return plain_d


def check_layer(name: str, layer: OSDN, x: torch.Tensor) -> None:
    x = x.detach().requires_grad_()
    params = [x] + list(layer.parameters())
    upstream = torch.randn(x.shape, device=x.device)
    chunked = layer(x)
    chunked_grads = torch.autograd.grad((chunked * upstream).sum(), params)
    plain = reference_layer(layer, x)
    plain_grads = torch.autograd.grad((plain * upstream).sum(), params)
    forward_err = rel_err(chunked, plain)
    names = ["input"] + [n for n, _ in layer.named_parameters()]
    grad_errs = {n: rel_err(a, b) for n, a, b in zip(names, chunked_grads, plain_grads)}
    worst = max(grad_errs, key=grad_errs.get)
    print(f"chunked layer vs plain recurrence [{name}]: fwd rel err {forward_err:.1e}, "
          f"worst grad rel err {grad_errs[worst]:.1e} ({worst})")
    require(forward_err < REL_TOL and grad_errs[worst] < REL_TOL, (name, forward_err, grad_errs))


def check_layer_both_scans(name: str, layer: OSDN, x: torch.Tensor) -> None:
    check_layer(f"{name}, Triton scan", layer, x)
    with mock.patch.object(scan_triton, "_use_kernel", lambda tensor: False):
        check_layer(f"{name}, PyTorch scan", layer, x)


def check_causal(layer: OSDN, x: torch.Tensor) -> None:
    with torch.no_grad():
        base = layer(x)
        for t0 in (0, 1, 63, 64, 127):
            changed = x.clone()
            changed[:, t0] += torch.randn_like(changed[:, t0])
            out = layer(changed)
            leak = float((out[:, :t0] - base[:, :t0]).abs().max()) if t0 > 0 else 0.0
            require(leak == 0.0, (t0, leak))
            require(float((out[:, t0] - base[:, t0]).abs().max()) > 0.0, t0)
    print("causality: outputs before a changed token are bitwise unchanged at t0 in {0, 1, 63, 64, 127}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, nargs=2, metavar=("N_HEADS", "HEAD_DIM"),
                        default=train.HEAD_CONFIG["osdn"])
    n_heads, head_dim = parser.parse_args().heads

    torch.manual_seed(0)
    model = build_model("osdn", 512, train.D_MODEL, train.D_FF, train.NUM_LAYERS, n_heads, head_dim,
                        chunk=train.CHUNK, context_length=train.CONTEXT_LENGTH).to(DEVICE)
    print(f"osdn ({n_heads}, {head_dim}): {sum(p.numel() for p in model.parameters())} params")
    check_init(model)

    T = train.CONTEXT_LENGTH
    # mmap reads only the sampled rows; a full load of train.pt takes 3.3 GB of RAM.
    train_ids = torch.load(train.ROOT / "packed_cache" / "train.pt", mmap=True)
    eval_ids = torch.load(train.ROOT / "packed_cache" / "validation.pt", mmap=True)
    eval_batch = eval_ids[: 8 * T].view(8, T).long().to(DEVICE)

    def layer_input(block_index: int) -> torch.Tensor:
        with torch.no_grad():
            h = model.wte(eval_batch)
            for block in model.blocks[:block_index]:
                h = block(h)
            return model.blocks[block_index].ln_1(h)

    kernel_constants = (precondition_triton.ETA, precondition_triton.D_MIN, precondition_triton.D_MAX,
                        precondition_triton.EPS)
    require(kernel_constants == (ETA, D_MIN, D_MAX, EPS), kernel_constants)
    layer = model.blocks[0].attn
    x = layer_input(0)
    with torch.no_grad():
        _, k, _, beta = layer._project(x)
    check_precondition_kernel("layer keys at init", k, beta.squeeze(-1))
    raw = torch.randn(4, T, 2, head_dim, device=DEVICE)
    gates = torch.rand(4, T, 2, device=DEVICE)
    check_precondition_kernel("floor: ||k||^2 < eps", raw * 1e-4, gates)
    lower = check_precondition_kernel("lower clamp: large keys", raw * 30.0, gates)
    require(float(lower.min()) == D_MIN, float(lower.min()))
    repeated = torch.zeros(2, 4096, 1, head_dim, device=DEVICE)
    repeated[..., 0] = 1.0
    upper = check_precondition_kernel("upper clamp: repeated key", repeated, torch.full((2, 4096, 1), 0.3, device=DEVICE))
    require(float(upper.max()) == D_MAX, float(upper.max()))

    check_layer_both_scans("block 0 at init", layer, x)
    for length in (37, 100):  # shorter than one chunk and not a multiple of it: both need padding
        check_layer(f"block 0 at init, T={length}", layer, x[:, :length])
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
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    non_finite = [n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
    require(not non_finite, non_finite)
    print(f"30 real Muon + AdamW steps at batch 64: loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"every parameter had a nonzero gradient at every step and all parameters are finite")
    for block_index in (0, train.NUM_LAYERS - 1):
        check_layer_both_scans(f"block {block_index} after 30 steps", model.blocks[block_index].attn,
                               layer_input(block_index))
    print("OSDN check passed")


if __name__ == "__main__":
    main()
