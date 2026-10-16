"""Runnable check of the KDA arm on real tokens: `python -m kda.check [--head-config H D]` from attention_comparison/.

It fails if the chunked layer disagrees with a plain token loop of Eq. 1 and Eq. 10 in forward or gradient
(at init, with a strong decay, and after training), if a wrong recurrence agrees with it, if an output
depends on a later token, if the init does not survive build_model, if the Muon/AdamW split is wrong, or
if a parameter gets no gradient or a loss is not finite in a short real run.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402
from kda.attention import (  # noqa: E402
    A_INIT_RANGE,
    CONV_INIT_STD,
    CONV_SIZE,
    DT_INIT_FLOOR,
    DT_INIT_RANGE,
    KDA,
    L2_EPS,
)
from model import build_model  # noqa: E402
from muon import split_params  # noqa: E402

DEVICE = torch.device("cuda")
REL_TOL = 1e-5  # the fp32 layer against an fp64 token loop
# A strong decay sums hundreds of log-decay steps in fp32, so exp(G_r - G_i) carries about |G| * 6e-8 relative error.
STRONG_DECAY_REL_TOL = 1e-4
CONTROL_MARGIN = 100.0  # a wrong recurrence must miss the layer by this factor times REL_TOL
SHORT_RUN_STEPS = 30


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach(), expected.detach()
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-30))


def chunk_spans(log_alpha: torch.Tensor) -> torch.Tensor:
    """-sum of log alpha over each chunk; log_alpha is (B,T,H,D)."""
    B, T, H, D = log_alpha.shape
    return -log_alpha.reshape(B, T // train.CHUNK, train.CHUNK, H, D).sum(2)


def reference_layer(layer: KDA, x: torch.Tensor, decay_after_write: bool = False) -> torch.Tensor:
    """Eq. 1 and Eq. 10 as a token loop, from the layer weights only; x is (B,T,d_model)."""
    B, T, _ = x.shape
    H, D = layer.h, layer.d

    def conv_silu(linear, conv):
        padded = F.pad(x @ linear.weight.T, (0, 0, CONV_SIZE - 1, 0))
        taps = conv.weight[:, 0]
        return F.silu(sum(taps[:, i] * padded[:, i:i + T] for i in range(CONV_SIZE))).reshape(B, T, H, D)

    q, k, v = (conv_silu(layer.q, layer.q_conv), conv_silu(layer.k, layer.k_conv),
               conv_silu(layer.v, layer.v_conv))
    q = q / (q.square().sum(-1, keepdim=True) + L2_EPS).sqrt() * D ** -0.5
    k = k / (k.square().sum(-1, keepdim=True) + L2_EPS).sqrt()
    decay_input = (x @ layer.f_down.weight.T @ layer.f_up.weight.T + layer.dt_bias).reshape(B, T, H, D)
    alpha = torch.exp(-layer.A_log.exp()[:, None] * F.softplus(decay_input))
    beta = torch.sigmoid(x @ layer.beta.weight.T)[..., None, None]

    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    state = x.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        k_t = k[:, t, :, :, None]
        transition = eye - beta[:, t] * (k_t @ k_t.mT)
        write = beta[:, t] * (k_t @ v[:, t, :, None, :])
        if decay_after_write:
            state = alpha[:, t, :, :, None] * (transition @ state + write)
        else:
            state = transition @ (alpha[:, t, :, :, None] * state) + write
        outs.append((state.mT @ q[:, t, :, :, None]).squeeze(-1))
    o = torch.stack(outs, 1)
    gate = (x @ layer.g_down.weight.T @ layer.g_up.weight.T).reshape(B, T, H, D)
    y = o / (o.square().mean(-1, keepdim=True) + layer.eps).sqrt() * layer.o_norm_weight * torch.sigmoid(gate)
    return y.reshape(B, T, H * D) @ layer.o.weight.T


def check_layer(name: str, layer: KDA, x: torch.Tensor, tol: float = REL_TOL, run_control: bool = False) -> None:
    x = x.detach().requires_grad_()
    upstream = torch.randn(x.shape, device=x.device)
    chunked = layer(x)
    chunked_grads = torch.autograd.grad((chunked * upstream).sum(), [x, *layer.parameters()])
    layer64, x64 = copy.deepcopy(layer).double(), x.detach().double().requires_grad_()
    plain = reference_layer(layer64, x64)
    plain_grads = torch.autograd.grad((plain * upstream.double()).sum(), [x64, *layer64.parameters()])
    chunked_grads = [g.double() for g in chunked_grads]
    forward_err = rel_err(chunked.double(), plain)
    names = ["input"] + [n for n, _ in layer.named_parameters()]
    grad_errs = {n: rel_err(a, b) for n, a, b in zip(names, chunked_grads, plain_grads)}
    worst = max(grad_errs, key=grad_errs.get)
    print(f"chunked layer vs token loop [{name}]: fwd rel err {forward_err:.1e}, "
          f"worst grad rel err {grad_errs[worst]:.1e} ({worst})")
    assert forward_err < tol and grad_errs[worst] < tol, (name, forward_err, grad_errs)
    if run_control:
        with torch.no_grad():
            control_err = rel_err(chunked.double(), reference_layer(layer64, x64, decay_after_write=True))
        print(f"  control, decay after the delta write: rel err {control_err:.1e}")
        assert control_err > CONTROL_MARGIN * REL_TOL, control_err


def check_causal(layer: KDA, x: torch.Tensor) -> None:
    with torch.no_grad():
        base = layer(x)
        for t0 in (0, 1, 63, 64, 127):
            changed = x.clone()
            changed[:, t0] += torch.randn_like(changed[:, t0])
            out = layer(changed)
            leak = float((out[:, :t0] - base[:, :t0]).abs().max()) if t0 else 0.0
            assert leak == 0.0, (t0, leak)
            assert float((out[:, t0] - base[:, t0]).abs().max()) > 0.0, t0
    print("causality: outputs before a changed token are bitwise unchanged at t0 in {0, 1, 63, 64, 127}")


@torch.no_grad()
def check_init(model) -> None:
    low, high = (math.log(bound) for bound in A_INIT_RANGE)
    dt_low = max(DT_INIT_RANGE[0], DT_INIT_FLOOR)
    for index, block in enumerate(model.blocks):
        layer = block.attn
        dt = F.softplus(layer.dt_bias)
        conv_std = torch.cat([c.weight.flatten() for c in (layer.q_conv, layer.k_conv, layer.v_conv)]).std()
        assert float(layer.A_log.min()) >= low and float(layer.A_log.max()) <= high, (index, layer.A_log)
        assert float(dt.min()) >= dt_low * (1 - 1e-5) and float(dt.max()) <= DT_INIT_RANGE[1] * (1 + 1e-5), (index, dt)
        assert abs(float(conv_std) / CONV_INIT_STD - 1.0) < 0.25, (index, float(conv_std))
        assert bool((layer.o_norm_weight == 1.0).all()), index
    print(f"init survives build_model: A_log in [log {A_INIT_RANGE[0]:g}, log {A_INIT_RANGE[1]:g}], "
          f"softplus(dt_bias) in [{DT_INIT_RANGE[0]:g}, {DT_INIT_RANGE[1]:g}], conv std ~{CONV_INIT_STD}, norm weight 1")


def check_optimizer_split(model) -> None:
    muon, adamw = split_params(model)
    muon_ids, adamw_ids = {id(p) for p in muon}, {id(p) for p in adamw}
    layer = model.blocks[0].attn
    expected_adamw = {"A_log", "dt_bias", "o_norm_weight", "q_conv.weight", "k_conv.weight", "v_conv.weight"}
    for name, param in layer.named_parameters():
        wanted = adamw_ids if name in expected_adamw else muon_ids
        assert id(param) in wanted, name
    print(f"optimizer split: AdamW gets {sorted(expected_adamw)}; Muon gets every Linear weight of the layer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-config", type=int, nargs=2, default=list(train.HEAD_CONFIG["kda"]),
                        metavar=("N_HEADS", "HEAD_DIM"))
    n_heads, head_dim = parser.parse_args().head_config

    torch.manual_seed(0)
    model = build_model("kda", 512, train.D_MODEL, train.D_FF, train.NUM_LAYERS, n_heads, head_dim,
                        chunk=train.CHUNK, context_length=train.CONTEXT_LENGTH).to(DEVICE)
    print(f"kda ({n_heads}, {head_dim}): {sum(p.numel() for p in model.parameters())} params")
    check_init(model)
    check_optimizer_split(model)

    T = train.CONTEXT_LENGTH
    # mmap: train.pt is 3.3 GB, and a full load per check process can exhaust the RAM of a small machine.
    train_ids = torch.load(train.ROOT / "packed_cache" / "train.pt", mmap=True)
    eval_ids = torch.load(train.ROOT / "packed_cache" / "validation.pt", mmap=True)
    eval_batch = eval_ids[: 8 * T].view(8, T).long().to(DEVICE)

    def layer_input(block_index: int) -> torch.Tensor:
        with torch.no_grad():
            h = model.wte(eval_batch)
            for block in model.blocks[:block_index]:
                h = block(h)
            return model.blocks[block_index].ln_1(h)

    layer = model.blocks[0].attn
    x = layer_input(0)
    with torch.no_grad():
        _, k, _, log_alpha, _ = layer._project(x)
        key_norm = k.norm(dim=-1).flatten()
        span = chunk_spans(log_alpha)
    print(f"block 0 at init: key norm median {float(key_norm.median()):.3f}, "
          f"5th percentile {float(key_norm.quantile(0.05)):.3f}; "
          f"chunk log-decay span max {float(span.max()):.1f}")
    check_layer("block 0 at init", layer, x, run_control=True)
    check_causal(layer, x)

    strong = KDA(train.D_MODEL, n_heads, head_dim, chunk=train.CHUNK, eps=layer.eps).to(DEVICE)
    strong.load_state_dict(layer.state_dict())
    with torch.no_grad():
        strong.A_log.fill_(math.log(2.0))
        strong.dt_bias.fill_(1.0)
        _, _, _, log_alpha, _ = strong._project(x)
    min_span = float(chunk_spans(log_alpha).min())
    print(f"strong decay: every chunk log-decay span >= {min_span:.0f} (exp(-G) overflows fp32 above 88)")
    assert min_span > 88.0, min_span
    check_layer("strong decay", strong, x, tol=STRONG_DECAY_REL_TOL)

    optimizers, schedulers = train._make_optimizers(model, total_steps=10_000)
    generator = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(SHORT_RUN_STEPS):
        rows = torch.randint(0, train_ids.shape[0] // T, (64,), generator=generator)
        batch = torch.stack([train_ids[r * T:(r + 1) * T] for r in rows.tolist()]).long().to(DEVICE)
        _, loss = model(batch, labels=batch)
        loss.backward()
        assert torch.isfinite(loss), losses
        losses.append(float(loss.detach()))
        zero_grad = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
        assert not zero_grad, zero_grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    print(f"{SHORT_RUN_STEPS} real Muon + AdamW steps at batch 64: loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"every parameter had a nonzero gradient at every step")
    for block_index in (0, train.NUM_LAYERS - 1):
        check_layer(f"block {block_index} after {SHORT_RUN_STEPS} steps", model.blocks[block_index].attn,
                    layer_input(block_index))
    print("KDA check passed")


if __name__ == "__main__":
    main()
