"""Runnable check of the Mamba-3 arm on real tokens at train.py's config: `python -m mamba3.check` from the
project root. It exits non-zero on the first failure.

It fails if the chunked layer disagrees with a token loop of the Prop. 4 recurrence (arXiv:2603.15569) in
the output or in the gradient of any single parameter, if a deliberately wrong recurrence agrees with the
loop, if an output depends on a later token, if the reference init does not survive build_model, if the
Muon/AdamW split is wrong, or if a parameter gets no gradient or a loss is not finite in a short real run.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from mamba3.attention import STATE_SIZE  # noqa: E402
from model import build_model  # noqa: E402
from muon import split_params  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEC_BC_NORM_EPS = 1e-5
SPEC_A_FLOOR = 1e-4
# Relative L2 error per tensor against the fp64 loop: fp64 layer (logic) and fp32 layer as trained (precision).
FP64_REL_TOL = 1e-10
FP32_REL_TOL = 1e-4
# A wrong recurrence must miss the loop by at least this much.
CONTROL_MIN_REL = 1e-2
SHORT_RUN_STEPS = 60
CHECK_SEQUENCES = 4
# Memory-mapped loads keep the 3.3 GB train split out of RAM; the short run samples from this prefix only.
SHORT_RUN_TOKENS = SHORT_RUN_STEPS * train.BATCH_SIZE * train.CONTEXT_LENGTH


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-300))


def rotate_pairs(vec: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (2i, 2i+1) of vec by +phase[..., i]; pairs past phase's width stay fixed."""
    n_rotated = 2 * phase.shape[-1]
    rotated, fixed = vec[..., :n_rotated], vec[..., n_rotated:]
    even, odd = rotated[..., 0::2], rotated[..., 1::2]
    cos, sin = phase.cos(), phase.sin()
    pairs = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((pairs, fixed), dim=-1)


def reference_layer(layer, layer_input: torch.Tensor, variant: str | None = None) -> torch.Tensor:
    """Prop. 4 as a token loop, from the layer weights only; layer_input is (batch, T, d_model).

    variant names a deliberately wrong recurrence: "euler" (lam = 1), "half_rotation" (N/4 pairs),
    "no_rotation", "no_trapezoid_shift" (beta = 0 but lam kept).
    """
    heads, head_dim = layer.n_heads, layer.head_dim
    batch, seq_len, _ = layer_input.shape

    def project(linear: torch.nn.Linear) -> torch.Tensor:
        return layer_input @ linear.weight.T

    def bc_norm(vec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return vec * torch.rsqrt(vec.square().mean(-1, keepdim=True) + SPEC_BC_NORM_EPS) * weight

    gate = project(layer.z_projection).view(batch, seq_len, heads, head_dim)
    values = project(layer.x_projection).view(batch, seq_len, heads, head_dim)
    dt = F.softplus(project(layer.dt_projection) + layer.dt_bias)
    decay_rate = (-F.softplus(project(layer.a_projection))).clamp(max=-SPEC_A_FLOOR)
    alpha = torch.exp(dt * decay_rate)
    lam_input = project(layer.lambda_projection)
    lam = torch.ones_like(lam_input) if variant == "euler" else torch.sigmoid(lam_input)
    beta = (1.0 - lam) * dt * alpha
    if variant == "no_trapezoid_shift":
        beta = torch.zeros_like(beta)
    gamma = lam * dt
    theta = math.pi * torch.tanh(project(layer.theta_projection))
    if theta.shape[-1] != STATE_SIZE // 2:
        raise AssertionError(f"{theta.shape[-1]} rotation angles, the paper rotates all N/2 = {STATE_SIZE // 2} pairs")
    if variant == "half_rotation":
        theta = theta[..., : STATE_SIZE // 4]
    if variant == "no_rotation":
        theta = theta[..., :0]
    b_normed = bc_norm(project(layer.b_projection), layer.b_norm.weight)
    c_normed = bc_norm(project(layer.c_projection), layer.c_norm.weight)

    head_outputs = []
    for head in range(heads):
        phase = layer_input.new_zeros(batch, theta.shape[-1])
        hidden = layer_input.new_zeros(batch, STATE_SIZE, head_dim)
        previous_write = layer_input.new_zeros(batch, STATE_SIZE, head_dim)
        steps = []
        for t in range(seq_len):
            phase = phase + dt[:, t, head, None] * theta[:, t]
            b_bar = rotate_pairs(b_normed[:, t] + layer.b_bias[head], phase)
            c_bar = rotate_pairs(c_normed[:, t] + layer.c_bias[head], phase)
            write = b_bar[:, :, None] * values[:, t, head, None, :]
            hidden = (alpha[:, t, head, None, None] * hidden + beta[:, t, head, None, None] * previous_write
                      + gamma[:, t, head, None, None] * write)
            previous_write = write
            steps.append(torch.einsum("bn,bnp->bp", c_bar, hidden) + layer.d_skip[head] * values[:, t, head])
        head_outputs.append(torch.stack(steps, dim=1))
    gated = torch.stack(head_outputs, dim=2) * F.silu(gate)
    return gated.reshape(batch, seq_len, heads * head_dim) @ layer.out_projection.weight.T


def layer_errors(layer, layer_input: torch.Tensor, dtype: torch.dtype, device: torch.device) -> dict[str, float]:
    """Relative L2 error of the layer run at (dtype, device) against the fp64 loop: output and each gradient."""
    fast = copy.deepcopy(layer).to(device=device, dtype=dtype)
    reference = copy.deepcopy(layer).to(device="cpu", dtype=torch.float64)
    reference_input = layer_input.detach().to("cpu", torch.float64).requires_grad_()
    upstream = torch.randn(layer_input.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    reference_out = reference_layer(reference, reference_input)
    reference_grads = torch.autograd.grad((reference_out * upstream).sum(),
                                          [reference_input, *reference.parameters()])
    fast_input = reference_input.detach().to(device, dtype).requires_grad_()
    fast_out = fast(fast_input)
    fast_grads = torch.autograd.grad((fast_out * upstream.to(device, dtype)).sum(), [fast_input, *fast.parameters()])
    errors = {"output": rel_l2(fast_out, reference_out)}
    for name, fast_grad, reference_grad in zip(["input", *(n for n, _ in fast.named_parameters())],
                                               fast_grads, reference_grads):
        errors[name] = rel_l2(fast_grad, reference_grad)
    return errors


def check_against_reference(name: str, layer, layer_input: torch.Tensor, dtype: torch.dtype,
                            device: torch.device, tolerance: float) -> None:
    errors = layer_errors(layer, layer_input, dtype, device)
    worst = max(errors, key=errors.get)
    print(f"{name}: {str(dtype).removeprefix('torch.')} on {device.type} vs fp64 loop, max rel L2 "
          f"{errors[worst]:.1e} ({worst}); output {errors['output']:.1e}")
    if errors[worst] > tolerance:
        raise AssertionError(f"[{name}] layer disagrees with the Prop. 4 recurrence: {errors}")


@torch.no_grad()
def check_controls(layer, layer_input: torch.Tensor) -> None:
    fast = copy.deepcopy(layer).double().cpu()
    layer_input = layer_input.double().cpu()
    fast_out = fast(layer_input)
    for variant in ("euler", "half_rotation", "no_rotation", "no_trapezoid_shift"):
        control_err = rel_l2(fast_out, reference_layer(fast, layer_input, variant))
        print(f"  wrong recurrence [{variant}]: output rel L2 {control_err:.2e}")
        if control_err < CONTROL_MIN_REL:
            raise AssertionError(f"wrong recurrence {variant} agrees with the layer")


def check_config(layer) -> None:
    expected = {
        "no bias on any projection": all(module.bias is None for module in layer.modules()
                                         if isinstance(module, torch.nn.Linear)),
        "one shared B and C group": layer.b_projection.out_features == layer.c_projection.out_features == STATE_SIZE,
        f"BCNorm eps {SPEC_BC_NORM_EPS}": layer.b_norm.eps == layer.c_norm.eps == SPEC_BC_NORM_EPS,
        "no conv module": not any(isinstance(module, torch.nn.Conv1d) for module in layer.modules()),
        "no output norm": sum(isinstance(module, torch.nn.RMSNorm) for module in layer.modules()) == 2,
    }
    print("layer config:", expected)
    failed = [name for name, ok in expected.items() if not ok]
    if failed:
        raise AssertionError(f"layer config differs from the paper layer: {failed}")


def check_init(model) -> None:
    for index, block in enumerate(model.blocks):
        layer = block.attn
        dt_init = F.softplus(layer.dt_bias)
        ones = {"b_bias": layer.b_bias, "c_bias": layer.c_bias, "d_skip": layer.d_skip,
                "b_norm.weight": layer.b_norm.weight, "c_norm.weight": layer.c_norm.weight}
        not_ones = [name for name, param in ones.items() if not torch.all(param == 1.0)]
        if not_ones or dt_init.min() < 1e-3 or dt_init.max() > 1e-1:
            raise AssertionError(f"block {index}: reference init lost: {not_ones}, dt {dt_init.tolist()}")
        print(f"init block {index}: b/c bias, D, BCNorm weights = 1; dt = {[round(v, 4) for v in dt_init.tolist()]}; "
              f"x_projection std {float(layer.x_projection.weight.detach().std()):.4f}")


@torch.no_grad()
def check_causality(model, ids: torch.Tensor) -> None:
    logits, _ = model(ids)
    for position in (1, 37, 63, 64, 100):
        changed = ids.clone()
        changed[:, position] = (changed[:, position] + 1) % model.wte.num_embeddings
        changed_logits, _ = model(changed)
        before = float((changed_logits[:, :position] - logits[:, :position]).abs().max())
        after = float((changed_logits[:, position:] - logits[:, position:]).abs().max())
        print(f"causality: token {position} changed -> max |dlogit| before {before:.1e}, from it on {after:.1e}")
        if before != 0.0 or after == 0.0:
            raise AssertionError(f"causality check failed at position {position}")


def check_optimizer_split(model) -> None:
    muon, adamw = split_params(model)
    muon_ids, adamw_ids = {id(p) for p in muon}, {id(p) for p in adamw}
    for index, block in enumerate(model.blocks):
        for name, param in block.attn.named_parameters():
            on_muon = name.endswith("projection.weight")
            if (id(param) in muon_ids) != on_muon or (id(param) in adamw_ids) == on_muon:
                raise AssertionError(f"block {index} {name}: wrong optimizer")
    print(f"optimizer split: Muon {len(muon)} tensors (9 projections per layer, MLP), AdamW {len(adamw)} tensors")


def attention_inputs(model, ids: torch.Tensor) -> list[torch.Tensor]:
    captured = []
    hooks = [block.attn.register_forward_hook(lambda module, args, output: captured.append(args[0].detach()))
             for block in model.blocks]
    with torch.no_grad():
        model(ids)
    for hook in hooks:
        hook.remove()
    return captured


def check_short_run(model, train_ids: torch.Tensor) -> None:
    blocks = train_ids[:SHORT_RUN_TOKENS].clone().view(-1, train.CONTEXT_LENGTH)
    generator = torch.Generator().manual_seed(0)
    optimizers, schedulers = train._make_optimizers(model, SHORT_RUN_STEPS)
    losses = []
    for step in range(SHORT_RUN_STEPS):
        batch = blocks[torch.randint(blocks.shape[0], (train.BATCH_SIZE,), generator=generator)].to(DEVICE).long()
        _, loss = model(batch, labels=batch)
        loss.backward()
        if step == 0:
            missing = [name for name, param in model.named_parameters()
                       if param.grad is None or not torch.isfinite(param.grad).all() or param.grad.abs().max() == 0]
            if missing:
                raise AssertionError(f"parameters with no, zero or non-finite gradient: {missing}")
            print(f"gradients: all {sum(1 for _ in model.parameters())} parameter tensors finite and non-zero")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    print(f"short real run: {SHORT_RUN_STEPS} steps at batch {train.BATCH_SIZE}, loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    if not all(math.isfinite(value) for value in losses) or losses[-1] >= losses[0]:
        raise AssertionError(f"short run: non-finite or non-decreasing loss {losses}")


def build(n_heads: int, head_dim: int, vocab_size: int):
    return build_model("mamba3", vocab_size, train.D_MODEL, train.D_FF, train.NUM_LAYERS, n_heads, head_dim,
                       chunk=train.CHUNK, context_length=train.CONTEXT_LENGTH)


def main() -> None:
    torch.manual_seed(0)
    vocab_size = len(train.load_tokenizer())
    n_heads, head_dim = train.HEAD_CONFIG["mamba3"]
    model = build(n_heads, head_dim, vocab_size)
    print(f"HEAD_CONFIG mamba3 (n_heads, head_dim) = {(n_heads, head_dim)}, N = {STATE_SIZE}: "
          f"{sum(p.numel() for p in model.parameters())} params")
    check_config(model.blocks[0].attn)
    check_init(model)
    check_optimizer_split(model)

    val_ids = torch.load(ROOT / "packed_cache" / "validation.pt", mmap=True)
    ids = val_ids[: CHECK_SEQUENCES * train.CONTEXT_LENGTH].clone().view(CHECK_SEQUENCES, train.CONTEXT_LENGTH).long()
    layer, layer_input = model.blocks[0].attn, attention_inputs(model, ids)[0]
    cpu = torch.device("cpu")
    check_against_reference("built init", layer, layer_input, torch.float64, cpu, FP64_REL_TOL)
    check_against_reference("built init, T = 100 (padded chunk)", layer, layer_input[:, :100], torch.float64, cpu,
                            FP64_REL_TOL)
    # At init dt and the angles are small; larger weights make decay, the trapezoid and the rotation matter.
    stressed = copy.deepcopy(layer)
    with torch.no_grad():
        for name, param in stressed.named_parameters():
            if name.endswith("projection.weight"):
                param.normal_(0.0, 0.3)
        stressed.dt_bias.add_(1.0)
    check_against_reference("stressed weights", stressed, layer_input, torch.float64, cpu, FP64_REL_TOL)
    check_against_reference("stressed weights", stressed, layer_input, torch.float32, DEVICE, FP32_REL_TOL)
    check_controls(stressed, layer_input)

    model = model.to(DEVICE)
    check_causality(model, ids.to(DEVICE))
    check_short_run(model, torch.load(ROOT / "packed_cache" / "train.pt", mmap=True))
    for index, block_input in enumerate(attention_inputs(model, ids.to(DEVICE))):
        check_against_reference(f"after short run, block {index}", model.blocks[index].attn, block_input,
                                torch.float32, DEVICE, FP32_REL_TOL)
    print("mamba3 check: PASS")


if __name__ == "__main__":
    main()
