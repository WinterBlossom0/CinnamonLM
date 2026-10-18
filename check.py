"""Runnable check of the arms on real tokens: `python check.py [arm ...]` from the project root. It exits non-zero
on the first failure.

For each arm (all arms and the check-only combinations when no arm is given) it fails if a parameter count, an init,
a layer order or a weight-sharing pattern is wrong, if a looped model disagrees with the same stock GPT-Neo blocks
applied by hand in the looped order (with each loop method's formula written from its paper, and the residual
scaling and Peri-LN branch formulas), if the Triton kernels or the chunked weight gradient change the loss or a
gradient of the stock model, if the compiled model disagrees with the eager model, if the model is further from fp64
GPT-Neo than the stock fp32 model is (2x margin), or if an output depends on a later token. It also fails if the
Parcae start noise or the GRT noise has the wrong distribution, or if the GRT noise is drawn in eval mode (eager or
compiled).

The run_seed checks (a short run whose loss must be finite and fall, run_seed against an independent training loop,
and the two-process gradient) run for every named arm of train.ARMS, but only for recursive_x2 and real_18 when no
arm is given.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention

import train

DEVICE = torch.device("cuda")
VOCAB_SIZE = 512
# One stock GPT-Neo layer at hidden 16: two LayerNorms, q/k/v, out_proj with bias, and the 16 -> 64 -> 16 MLP.
PARAMS_PER_LAYER = 2 * 2 * 16 + 3 * 16 * 16 + (16 * 16 + 16) + (16 * 64 + 64) + (64 * 16 + 16)
FOUR_LAYER_PARAMS = 23200  # the 4-layer GPT-Neo proxy of the activation comparison
REL_TOL = 1e-5
SHORT_RUN_STEPS = 60
TRAINING_LOOP_STEPS = 20
TRAINING_LOOP_TOL = 1e-4
# Parameters each loop method trains without weight decay, from its source (see train.exempt_from_weight_decay).
NO_DECAY = {
    "plain": set(),
    "reinject": {"transformer.h.1.mix_logit"},  # arXiv:2610.00673, App. F.4: zero except projections and LoRA
    "hyperloop": set(),  # arXiv:2604.21254, App. A: weight decay 0.1, no exemption; no public code
    # parcae_lm modules: log_A, dt_bias, B, C (_no_weight_decay); the input norm through the MuonAdamW launch path:
    # recpre/optim/muon_adamw.py:107,121 and recpre/optim/__init__.py:24, AdamW weight decay 0.0 (Tables 20-21)
    "parcae": {f"transformer.h.1.{name}" for name in ("log_A", "dt_bias", "B", "C", "input_norm.weight",
                                                      "input_norm.bias")},
    # GRT model.py configure_optimizers: every parameter with fewer than 2 dimensions
    "grt": {f"transformer.h.1.{name}" for name in ("state_norm.weight", "input_norm.weight", "joint_norm.weight",
                                                   "gate_out.bias")},
    "mesh": {"transformer.h.0.router_bias"},  # Hugging Face Trainer default: biases exempt
}
# Each loop method with a residual-branch change, so the changes are checked inside every loop module.
CHECK_ARMS = {
    "check_scaled_x8": train.Arm(4, 8, residual_scale=True),
    "check_peri_ln_x8": train.Arm(4, 8, peri_ln=True),
    "check_reinject_scaled_x8": train.Arm(4, 8, "reinject", residual_scale=True),
    "check_hyperloop_peri_ln_x8": train.Arm(4, 8, "hyperloop", peri_ln=True),
    "check_parcae_scaled_x8": train.Arm(4, 8, "parcae", residual_scale=True),
    "check_grt_peri_ln_x8": train.Arm(4, 8, "grt", peri_ln=True),
}
SPECS = {**train.ARMS, **CHECK_ARMS}


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double(), expected.detach().double()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


class StockHyperLoop(train.HyperLoop):
    """HyperLoop with its hyper-connection read in torch ops instead of the fused kernels."""

    def read(self, streams, pass_index):
        flat = F.rms_norm(streams.flatten(-2), (streams.shape[-2] * streams.shape[-1],), self.norm_weight[pass_index],
                          eps=train.RMS_NORM_EPS)
        logits = torch.einsum("...f,knf->...kn", flat, self.weight[pass_index])
        logits = self.gate[pass_index, :, None] * logits + self.bias[pass_index]
        read, write, keep = torch.sigmoid(logits).unbind(-2)
        return (read.unsqueeze(-1) * streams).sum(-2), write, keep


class StockGRTLoop(train.GRTLoop):
    """GRTLoop with its gated update in torch ops instead of the fused kernels."""

    def update(self, state, output, normed_input, gate_noise):
        joint = self.joint_norm(torch.cat([self.state_norm(state), normed_input], dim=-1))
        logit = self.gate_out(F.silu(F.linear(joint, self.gate_hidden.weight)))
        if gate_noise is not None:
            logit = logit + gate_noise.unsqueeze(-1)
        gate = torch.sigmoid(logit)
        return gate * state + (1 - gate) * output


class StockMeSHLoop(train.MeSHLoop):
    """MeSHLoop with each step's read and routers in torch ops instead of the fused kernels."""

    def forward(self, hidden_states, **kwargs):
        embeddings = state = hidden_states
        write, read = self.route(embeddings, 0)
        writes, outputs = [], []
        for step in range(self.passes + 1):
            if step == 0:
                output = self.first(state, **kwargs)[0]
            else:
                output = state
                for block in self.blocks:
                    output = block(output, **kwargs)[0]
            writes.append(write)
            outputs.append(output)
            state = read[..., :1] * embeddings
            for slot_weights, kept in zip(writes, outputs):
                state = state + (read * slot_weights).sum(-1, keepdim=True) * kept
            if step < self.passes:
                write, read = self.route(state, step + 1)
        return (state,)


def stock_copy(model: nn.Module) -> nn.Module:
    """The same weights with HF's eager attention, plain nn.Linear and the unfused Hyperloop, GRT and MeSH steps."""
    stock = copy.deepcopy(model)
    for module in stock.modules():
        if isinstance(module, train.GPTNeoFusedAttention):
            module.__class__ = GPTNeoSelfAttention
        elif isinstance(module, train.ChunkedGradLinear):
            module.__class__ = nn.Linear
        elif isinstance(module, train.HyperLoop):
            module.__class__ = StockHyperLoop
        elif isinstance(module, train.GRTLoop):
            module.__class__ = StockGRTLoop
        elif isinstance(module, train.MeSHLoop):
            module.__class__ = StockMeSHLoop
    return stock


def build(arm: str, batch_size: int) -> nn.Module:
    """The arm's model on the GPU. The Parcae start state and the GRT noise are fixed noise for batch_size sequences,
    so two models or two calls get the same noise; check_parcae_noise and check_grt_noise check the drawn noise. The
    GRT noise still goes through GRTLoop.noise, so it applies only in training mode."""
    torch.manual_seed(0)
    model = train.build_model(SPECS[arm], VOCAB_SIZE).to(DEVICE)
    like = torch.empty(batch_size, train.CONTEXT_LENGTH, train.HIDDEN_SIZE, device=DEVICE)
    for module in model.modules():
        # Rows from noise_offset on, so a check that splits a batch can give each part its rows.
        if isinstance(module, train.ParcaeLoop):
            module.fixed_noise = module.initial_state(like)
            module.noise_offset = 0
            module.initial_state = lambda like, loop=module: loop.fixed_noise[
                loop.noise_offset:loop.noise_offset + like.shape[0]].to(like.dtype)
        elif isinstance(module, train.GRTLoop):
            module.fixed_noise = module.draw_noise(like)
            module.noise_offset = 0
            module.draw_noise = lambda like, loop=module: tuple(
                part[:, loop.noise_offset:loop.noise_offset + like.shape[0]].to(like.dtype)
                for part in loop.fixed_noise)
    return model


def loss_and_grads(model: nn.Module, batch: torch.Tensor,
                   forward: nn.Module | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    if forward is None:
        forward = model
    loss = forward(input_ids=batch, labels=batch).loss
    loss.backward()
    return loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters()}


def applied_blocks(model: nn.Module) -> list[nn.Module]:
    """The blocks in the order the forward applies them, with each loop module expanded into its passes."""
    blocks = []
    for module in model.transformer.h:
        if isinstance(module, train.MeSHLoop):
            blocks.extend([module.first] + list(module.blocks) * module.passes)
        elif isinstance(module, (train.ReinjectedLoop, train.HyperLoop, train.ParcaeLoop, train.GRTLoop)):
            blocks.extend(list(module.blocks) * module.passes)
        else:
            blocks.append(module)
    return blocks


def check_structure(arm: str) -> None:
    spec = SPECS[arm]
    unique_layers, passes = spec.unique_layers, spec.passes
    model = train.build_model(spec, VOCAB_SIZE)
    order = train.layer_order(unique_layers, passes)
    blocks = applied_blocks(model)
    n_params = sum(p.numel() for p in model.parameters())
    streams, width = train.HYPERLOOP_STREAMS, train.HIDDEN_SIZE
    loop_params = {
        "plain": 0,
        "reinject": 1,  # the mixing logit a
        # per pass: W (3 x n x nC), b (3 x n), alpha (3), the RMSNorm weight (nC) and e_l (C)
        "hyperloop": passes * (3 * streams * streams * width + 3 * streams + 3 + streams * width + width),
        # the LayerNorm on its input (2C), log A and dt (C each), B and C (C x C each)
        "parcae": 2 * width + 2 * width + 2 * width * width,
        # W_proj (C x 2C), gains of the two C-wide and the 2C-wide LayerNorm, the gate MLP 2C -> C (no bias) -> C
        "grt": 2 * width * width + 4 * width + 2 * width * width + (width * width + width),
        # per step (the first layer and each pass) a write and a read router C -> B with bias, B = passes + 3
        "mesh": (passes + 1) * 2 * (width * (passes + 3) + (passes + 3)),
    }[spec.loop]
    # Peri-LN: two output LayerNorms (2C each) in each looped layer.
    peri_ln_params = (unique_layers - 2) * 2 * 2 * width * int(spec.peri_ln)
    expected = FOUR_LAYER_PARAMS + (unique_layers - 4) * PARAMS_PER_LAYER + loop_params + peri_ln_params
    require(n_params == expected, f"{arm}: {n_params} params, expected {expected}")
    no_decay = {name for name, p in model.named_parameters() if getattr(p, "_no_weight_decay", False)}
    require(no_decay == NO_DECAY[spec.loop], f"{arm}: weight-decay exemptions {sorted(no_decay)}, expected "
            f"{sorted(NO_DECAY[spec.loop])}")
    if spec.loop == "parcae":
        # The official code's C: an orthogonal matrix times std * sqrt(width), so every singular value is the gain.
        gain = train.INIT_STD * math.sqrt(width)
        singular = torch.linalg.svdvals(model.transformer.h[1].C.detach().double())
        require(bool(((singular / gain - 1).abs() < 1e-5).all()),
                f"{arm}: Parcae C singular values {singular.min():.4f}-{singular.max():.4f}, expected all {gain:.4f}")
    if spec.loop == "reinject":
        mix = torch.sigmoid(model.transformer.h[1].mix_logit).item()
        require(abs(mix - 0.99) < 1e-3, f"{arm}: initial mixing weight {mix:.4f}, expected 0.99 (arXiv:2610.00673)")
    if spec.residual_scale:
        scales = {block.branch_scale for block in blocks[1:-1]}
        require(scales == {1.0 / passes}, f"{arm}: residual branch scales {scales}, expected 1/{passes}")
    require(len(blocks) == len(order), f"{arm}: {len(blocks)} layer positions, expected {len(order)}")
    types = [block.attn.attention.attention_type for block in blocks]
    alternating = ["global" if position % 2 == 0 else "local" for position in range(len(order))]
    require(types == alternating, f"{arm}: layer types {types} do not alternate global/local from global")
    distinct = {id(block) for block in blocks}
    require(len(distinct) == unique_layers, f"{arm}: {len(distinct)} distinct layers, expected {unique_layers}")
    for position, index in enumerate(order):
        require(blocks[position] is blocks[order.index(index)], f"{arm}: position {position} is not layer {index}")
    require(model.config.use_cache is False, f"{arm}: the key/value cache must be off")
    print(f"{arm}: {n_params} params, {unique_layers} unique layers, effective depth {len(order)}, order {order}")


def move_off_init(model: nn.Module) -> None:
    """Values away from the init, where some terms of a formula vanish (B = I, alpha = 0.01, gain 1 and bias 0)."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, train.ReinjectedLoop):
                module.mix_logit.fill_(0.0)  # w = 0.5, so both terms of the mix carry weight
            elif isinstance(module, train.HyperLoop):
                module.gate.fill_(1.0)
                module.bias.normal_(0.0, 0.5)
                module.norm_weight.add_(torch.randn_like(module.norm_weight) * 0.3)
            elif isinstance(module, train.ParcaeLoop):
                for parameter in (module.log_A, module.dt_bias, module.B, module.input_norm.weight,
                                  module.input_norm.bias):
                    parameter.add_(torch.randn_like(parameter) * 0.3)
            elif isinstance(module, train.MeSHLoop):
                module.router_weight.normal_(0.0, 0.5)
                module.router_bias.normal_(0.0, 0.5)
            elif isinstance(module, train.GRTLoop):
                for linear in (module.project, module.gate_hidden, module.gate_out):
                    linear.weight.normal_(0.0, 0.3)
                module.gate_out.bias.normal_(0.0, 1.0)
                for norm in (module.state_norm, module.input_norm, module.joint_norm):
                    norm.weight.add_(torch.randn_like(norm.weight) * 0.3)
            elif isinstance(module, train.LoopedBlock) and isinstance(module.attn_out_norm, nn.LayerNorm):
                for norm in (module.attn_out_norm, module.mlp_out_norm):
                    norm.weight.add_(torch.randn_like(norm.weight) * 0.3)
                    norm.bias.normal_(0.0, 0.3)


def block_by_hand(block: nn.Module, hidden: torch.Tensor, mask: torch.Tensor, scale: float,
                  peri_ln: bool) -> torch.Tensor:
    """h + s N_out(f(LN(h))) per branch: Wang et al. 2026 (arXiv:2606.18524), Table 3; Kim et al. 2025
    (arXiv:2502.02732), Eq. 3."""
    def out_norm(norm: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if not peri_ln:
            return values
        return F.layer_norm(values, (values.shape[-1],), norm.weight, norm.bias, train.LAYER_NORM_EPS)

    attn_output = block.attn(block.ln_1(hidden), attention_mask=mask)[0]
    hidden = hidden + scale * out_norm(block.attn_out_norm, attn_output)
    return hidden + scale * out_norm(block.mlp_out_norm, block.mlp(block.ln_2(hidden)))


def check_unrolled(arm: str, batch: torch.Tensor) -> None:
    """The looped model against its unique stock blocks applied by hand in the looped order."""
    spec = SPECS[arm]
    model = stock_copy(build(arm, batch.shape[0]))
    move_off_init(model)
    unique_layers, passes = spec.unique_layers, spec.passes
    order = train.layer_order(unique_layers, passes)
    blocks = applied_blocks(model)
    unique = [blocks[order.index(index)] for index in range(unique_layers)]
    loop = model.transformer.h[0 if spec.loop == "mesh" else 1]
    changed_branches = spec.residual_scale or spec.peri_ln
    scale = 1.0 / passes if spec.residual_scale else 1.0
    causal = torch.full((batch.shape[1], batch.shape[1]), torch.finfo(torch.float32).min, device=DEVICE).triu(1)
    mask = causal[None, None]

    def layer(index: int, hidden: torch.Tensor) -> torch.Tensor:
        if changed_branches and 0 < index < unique_layers - 1:
            return block_by_hand(unique[index], hidden, mask, scale, spec.peri_ln)
        return unique[index](hidden, attention_mask=mask)[0]

    def middle(hidden: torch.Tensor) -> torch.Tensor:
        for index in range(1, unique_layers - 1):
            hidden = layer(index, hidden)
        return hidden

    with torch.no_grad():
        logits = model(input_ids=batch).logits
        embeddings = model.transformer.wte(batch) + model.transformer.wpe(torch.arange(batch.shape[1], device=DEVICE))
        hidden = embeddings if spec.loop == "mesh" else layer(0, embeddings)
        if spec.loop == "mesh":
            hidden = mesh_by_hand(loop, lambda values: layer(0, values), middle, embeddings)
        elif spec.loop == "plain":
            for _ in range(passes):
                hidden = middle(hidden)
        elif spec.loop == "reinject":
            # Marchenko et al. 2026, Eq. 1: z_t = w e + (1 - w) h_{t-1} for every pass after the first.
            mix, loop_input = torch.sigmoid(loop.mix_logit), hidden
            for pass_index in range(passes):
                if pass_index > 0:
                    hidden = mix * loop_input + (1 - mix) * hidden
                hidden = middle(hidden)
        elif spec.loop == "hyperloop":
            hidden = hyperloop_by_hand(loop, middle, hidden)
        elif spec.loop == "parcae":
            hidden = parcae_by_hand(loop, middle, hidden)
        else:
            hidden = grt_by_hand(loop, middle, hidden)
        manual = model.lm_head(model.transformer.ln_f(layer(unique_layers - 1, hidden)))
    error = rel_err(logits, manual)
    require(error < REL_TOL, f"{arm}: looped model vs hand-unrolled blocks, rel err {error:.1e}")
    print(f"{arm}: looped model = stock blocks unrolled by hand, logits rel err {error:.1e}")


def hyperloop_by_hand(loop: nn.Module, middle, loop_input: torch.Tensor) -> torch.Tensor:
    """Zeitoun et al. 2026 (arXiv:2604.21254), Sec. 3, one stream and one pass at a time."""
    n = loop.streams
    streams = [loop_input.clone() for _ in range(n)]
    for pass_index in range(loop.passes):
        flat = torch.cat(streams, dim=-1)
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + torch.finfo(flat.dtype).eps)
        flat = flat * loop.norm_weight[pass_index]
        h = []
        for kind in range(3):  # pre, post, res
            raw = flat @ loop.weight[pass_index, kind].T
            h.append(torch.sigmoid(loop.gate[pass_index, kind] * raw + loop.bias[pass_index, kind]))
        read, write, keep = h
        output = middle(sum(read[..., k:k + 1] * streams[k] for k in range(n))) + loop.loop_embedding[pass_index]
        streams = [keep[..., k:k + 1] * streams[k] + 2 * write[..., k:k + 1] * output for k in range(n)]
    return sum(streams) / n


def parcae_by_hand(loop: nn.Module, middle, loop_input: torch.Tensor) -> torch.Tensor:
    """Prairie et al. 2026 (arXiv:2604.12946): x <- A_bar x + B_bar e, x <- F(x), output C x, with A = -exp(log_A)
    diagonal, A_bar = exp(dt A), B_bar = dt B and e = LN(loop input), as matrices."""
    norm = loop.input_norm
    injected = F.layer_norm(loop_input, (loop_input.shape[-1],), norm.weight, norm.bias, train.LAYER_NORM_EPS)
    dt = F.softplus(loop.dt_bias)
    a_bar = torch.diag(torch.exp(dt * -torch.exp(loop.log_A)))
    b_bar = torch.diag(dt) @ loop.B
    state = loop.initial_state(loop_input)
    for _ in range(loop.passes):
        state = middle(state @ a_bar.T + injected @ b_bar.T)
    return state @ loop.C.T


def mesh_by_hand(loop: nn.Module, first, middle, embeddings: torch.Tensor) -> torch.Tensor:
    """Yu et al. 2026 (arXiv:2510.07739), Eqs. 2-5 with the explicit buffer of B slots, as the official code runs it
    (the first step's routers read the embeddings)."""
    slots = loop.slots
    memory = [embeddings] + [torch.zeros_like(embeddings) for _ in range(slots - 1)]
    state = embeddings
    for step in range(loop.passes + 1):
        output = first(state) if step == 0 else middle(state)
        weight, bias = loop.router_weight[step], loop.router_bias[step]
        write = torch.softmax(state @ weight[:slots].T + bias[:slots], dim=-1)
        read = torch.softmax(state @ weight[slots:].T + bias[slots:], dim=-1)
        memory = [memory[b] + output * write[..., b:b + 1] for b in range(slots)]
        state = sum(memory[b] * read[..., b:b + 1] for b in range(slots))
    return state


def grt_by_hand(loop: nn.Module, middle, loop_input: torch.Tensor) -> torch.Tensor:
    """Hegazy et al. 2026 (arXiv:2608.15062), Eqs. 2-5, as the official model.py computes a step: eps_x in the W_proj
    input only, the gate over LN_2C([LN_x(h); LN_h(h_pre)]) through Linear(no bias) -> SiLU -> Linear, gains only,
    tau = 1, plus one eps_g per token."""
    def layer_norm(values: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(-1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + train.LAYER_NORM_EPS) * gain

    noise = loop.noise(loop_input)
    state = loop_input
    for pass_index in range(loop.passes):
        state_noise = 0.0 if noise is None else noise[0][pass_index]
        gate_noise = 0.0 if noise is None else noise[1][pass_index].unsqueeze(-1)
        output = middle(torch.cat([state + state_noise, loop_input], dim=-1) @ loop.project.weight.T)
        joint = layer_norm(torch.cat([layer_norm(state, loop.state_norm.weight),
                                      layer_norm(loop_input, loop.input_norm.weight)], dim=-1), loop.joint_norm.weight)
        hidden = F.silu(joint @ loop.gate_hidden.weight.T)
        gate = torch.sigmoid(hidden @ loop.gate_out.weight.T + loop.gate_out.bias + gate_noise)
        state = gate * state + (1 - gate) * output
    return state


def check_against_stock(arm: str, batches: list[torch.Tensor], off_init: bool) -> None:
    """Kernels and chunked gradients against the stock model on batches[0], at init and off init (where every term
    carries weight). The fp64 fallback for a compiled gradient uses all batches."""
    batch = batches[0]
    model = build(arm, batch.shape[0])
    if off_init:
        move_off_init(model)
    stock = stock_copy(model)
    loss, grads = loss_and_grads(model, batch)
    stock_loss, stock_grads = loss_and_grads(stock, batch)
    require(grads.keys() == stock_grads.keys(), f"{arm}: parameter names differ from the stock model")
    worst_name = max(grads, key=lambda name: rel_err(grads[name], stock_grads[name]))
    worst = rel_err(grads[worst_name], stock_grads[worst_name])
    loss_error = abs(loss.item() - stock_loss.item()) / stock_loss.item()
    require(loss_error < REL_TOL and worst < REL_TOL,
            f"{arm}: vs stock GPT-Neo, loss rel err {loss_error:.1e}, worst grad {worst_name} rel err {worst:.1e}")
    require(all(g.abs().max() > 0 for g in grads.values()), f"{arm}: a parameter has a zero gradient")
    torch._dynamo.reset()
    compiled = torch.compile(model)
    compiled_loss, compiled_grads = loss_and_grads(model, batch, compiled)
    compiled_errors = {name: rel_err(compiled_grads[name], grads[name]) for name in grads}
    compiled_worst = max(compiled_errors.values())
    compiled_loss_error = abs(compiled_loss.item() - loss.item()) / loss.item()
    require(compiled_loss_error < REL_TOL,
            f"{arm}: compiled vs eager, loss rel err {compiled_loss_error:.1e}")
    # A gradient that is a sum with cancellation (the 0-dim re-injection logit) has an fp32 error near REL_TOL that
    # changes 30x between batches, and either model can be the worse one on a batch. Such a tensor passes if, as a
    # mean over the batches, compiled is within 2 max(eager, REL_TOL) of fp64; a 10x error is about 9 from fp64.
    failing = [name for name, error in compiled_errors.items() if error >= REL_TOL]
    compiled_vs_exact, eager_vs_exact = fp64_distances(model, compiled, batches, failing) if failing else ({}, {})
    for name in failing:
        bound = 2 * max(eager_vs_exact[name], REL_TOL)
        require(compiled_vs_exact[name] <= bound,
                f"{arm}: compiled vs eager, grad {name} rel err {compiled_errors[name]:.1e}; vs fp64 over "
                f"{len(batches)} batches compiled {compiled_vs_exact[name]:.1e}, eager {eager_vs_exact[name]:.1e}")
        print(f"{arm}: compiled vs eager grad {name} rel err {compiled_errors[name]:.1e} >= {REL_TOL:.0e}; vs fp64 "
              f"over {len(batches)} batches compiled {compiled_vs_exact[name]:.1e} <= 2 x max(eager "
              f"{eager_vs_exact[name]:.1e}, {REL_TOL:.0e})")
    print(f"{arm} ({'off init' if off_init else 'init'}): vs stock loss rel err {loss_error:.1e}, worst grad rel err "
          f"{worst:.1e} ({worst_name}); compiled vs eager {compiled_loss_error:.1e} / {compiled_worst:.1e}")


def fp64_distances(model: nn.Module, compiled: nn.Module, batches: list[torch.Tensor],
                   names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """The mean over batches of the relative distance from fp64 GPT-Neo of the compiled and of the eager gradient of
    each named parameter."""
    exact_model = float64_copy(model)
    compiled_mean, eager_mean = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0.0)
    for batch in batches:
        exact = next_token_grads(exact_model, batch)
        _, eager = loss_and_grads(model, batch)
        _, compiled_grads = loss_and_grads(model, batch, compiled)
        for name in names:
            compiled_mean[name] += rel_err(compiled_grads[name], exact[name]) / len(batches)
            eager_mean[name] += rel_err(eager[name], exact[name]) / len(batches)
    return compiled_mean, eager_mean


class Float64Attention(GPTNeoSelfAttention):
    """GPTNeoSelfAttention._attn without its cast of q and k to fp32, so an fp64 copy of the model stays fp64."""

    def _attn(self, query, key, value, attention_mask=None):
        length = query.shape[-2]
        scores = query @ key.transpose(-1, -2)
        scores = torch.where(self.bias[:, :, :length, :length], scores, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        return torch.softmax(scores, dim=-1) @ value, None


def float64_copy(model: nn.Module) -> nn.Module:
    """The stock model in fp64, attention included."""
    exact = stock_copy(model).double()
    for module in exact.modules():
        if isinstance(module, GPTNeoSelfAttention):
            module.__class__ = Float64Attention
    return exact


def next_token_grads(network: nn.Module, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """The gradients of the mean next-token loss in the network's own dtype."""
    # The loss is computed here, because GPT-Neo casts the logits to fp32 before its own loss.
    network.zero_grad(set_to_none=True)
    logits = network(input_ids=batch).logits
    nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1)).backward()
    return {name: p.grad.clone() for name, p in network.named_parameters()}


def check_precision(arm: str, batches: list[torch.Tensor]) -> None:
    """The fp32 model with the kernels must be as close to fp64 GPT-Neo as the fp32 stock model is, off init (every
    term of the fused kernels carries weight). The measure is the RMS over parameters of each gradient's relative
    error, averaged over batches: a single gradient that is a sum with cancellation (the re-injection logit) has an
    fp32 error that changes 30x between batches, and either implementation can be the worse one on a batch. A
    systematic loss (TF32, bf16) fails the 2x margin. A loss in one tensor fails it only if it is large: with n
    tensors of equal error (56 to 238 in these arms), one tensor must get more than sqrt(3n + 1) = 13x to 27x worse."""
    model = build(arm, batches[0].shape[0])
    move_off_init(model)
    stock = stock_copy(model)
    exact = float64_copy(model)

    def rms_error(grads: dict[str, torch.Tensor], exact_grads: dict[str, torch.Tensor]) -> float:
        return math.sqrt(sum(rel_err(g, exact_grads[name]) ** 2 for name, g in grads.items()) / len(grads))

    ours, baseline = 0.0, 0.0
    for batch in batches:
        exact_grads = next_token_grads(exact, batch)
        ours += rms_error(next_token_grads(model, batch), exact_grads) / len(batches)
        baseline += rms_error(next_token_grads(stock, batch), exact_grads) / len(batches)
    require(ours <= 2 * baseline, f"{arm}: gradient error vs fp64 {ours:.1e}, stock fp32 model {baseline:.1e}")
    print(f"{arm}: gradient error vs fp64 GPT-Neo {ours:.1e} (stock fp32 model {baseline:.1e}; RMS over parameters, "
          f"mean over {len(batches)} batches)")


def check_causality(arm: str, batch: torch.Tensor) -> None:
    model = build(arm, 1)
    sequence = batch[:1]
    with torch.no_grad():
        logits = model(input_ids=sequence).logits
        for position in (1, 15, 16, 64, 127):
            changed = sequence.clone()
            changed[0, position] = (changed[0, position] + 1) % VOCAB_SIZE
            new_logits = model(input_ids=changed).logits
            before = (new_logits[0, :position] - logits[0, :position]).abs().max().item() if position else 0.0
            after = (new_logits[0, position:] - logits[0, position:]).abs().max().item()
            require(before == 0.0 and after > 0.0, f"{arm}: token {position} changed, earlier logits moved {before}")
    print(f"{arm}: outputs before a changed token are bitwise unchanged at t0 in {{1, 15, 16, 64, 127}}")


def check_parcae_noise() -> None:
    """The start state, eager and compiled, against N(0, std) truncated at 3 std: bound, mean and std."""
    std = train.INIT_STD
    density, mass = math.exp(-4.5) / math.sqrt(2 * math.pi), math.erf(3 / math.sqrt(2))
    truncated_std = std * math.sqrt(1 - 2 * 3 * density / mass)
    like = torch.empty(1000, 1000, device=DEVICE)
    torch._dynamo.reset()
    for name, draw in (("eager", train.ParcaeLoop.initial_state), ("compiled", torch.compile(train.ParcaeLoop.initial_state))):
        noise = draw(like).double()
        bound, mean, spread = noise.abs().max().item(), noise.mean().item(), noise.std().item()
        require(bound <= 3 * std and abs(mean) < 1e-4 and abs(spread / truncated_std - 1) < 1e-2,
                f"Parcae noise ({name}): max |x| {bound:.4f}, mean {mean:.1e}, std {spread:.5f} vs {truncated_std:.5f}")
        print(f"Parcae noise ({name}): max |x| {bound:.4f} <= {3 * std}, mean {mean:.1e}, std {spread:.5f} "
              f"(truncated normal {truncated_std:.5f})")


def check_grt_noise(arm: str, batch: torch.Tensor) -> None:
    """GRT noise: none in eval mode (noise() is None, and two eager and two compiled forwards are bitwise equal; the
    compiled eval forward gives the reported eval loss); in training mode, eager and compiled, eps_x of shape
    (passes, *x.shape) and eps_g of one value per token, both N(0, GRT_NOISE_STD^2), and two forwards differ. The
    compiled training step must run as one graph, so the noise does not break it."""
    torch.manual_seed(0)
    model = train.build_model(SPECS[arm], VOCAB_SIZE).to(DEVICE)
    loop = next(module for module in model.modules() if isinstance(module, train.GRTLoop))
    like = torch.empty(256, train.CONTEXT_LENGTH, train.HIDDEN_SIZE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        first, second = (model(input_ids=batch).logits for _ in range(2))
    require(loop.noise(like) is None and torch.equal(first, second), f"{arm}: GRT noise is drawn in eval mode")
    torch._dynamo.reset()
    compiled = torch.compile(model)
    with torch.no_grad():
        compiled_first, compiled_second = (compiled(input_ids=batch, labels=batch) for _ in range(2))
    require(torch.equal(compiled_first.logits, compiled_second.logits)
            and torch.equal(compiled_first.loss, compiled_second.loss)
            and rel_err(compiled_first.logits, first) < REL_TOL,
            f"{arm}: the compiled eval forward draws GRT noise or differs from the eager eval forward")
    model.train()
    with torch.no_grad():
        compiled_train = compiled(input_ids=batch, labels=batch).logits
    require(not torch.equal(compiled_train, compiled_first.logits),
            f"{arm}: the compiled training forward draws no GRT noise")
    with torch.no_grad():
        first, second = (model(input_ids=batch).logits for _ in range(2))
    require(not torch.equal(first, second), f"{arm}: two training-mode forwards are equal, so no GRT noise is drawn")
    std = train.GRT_NOISE_STD
    torch._dynamo.reset()
    for name, draw in (("eager", loop.noise), ("compiled", torch.compile(loop.noise, fullgraph=True))):
        state_noise, gate_noise = draw(like)
        require(state_noise.shape == (loop.passes, *like.shape) and gate_noise.shape == (loop.passes, *like.shape[:-1]),
                f"GRT noise ({name}): shapes {tuple(state_noise.shape)} / {tuple(gate_noise.shape)}")
        for kind, noise in (("eps_x", state_noise), ("eps_g", gate_noise)):
            mean, spread = noise.double().mean().item(), noise.double().std().item()
            require(abs(mean) < 2e-3 and abs(spread / std - 1) < 1e-2,
                    f"GRT noise ({name}) {kind}: mean {mean:.1e}, std {spread:.5f} vs {std}")
            print(f"GRT noise ({name}) {kind}: shape {tuple(noise.shape)}, mean {mean:.1e}, std {spread:.5f} "
                  f"(expected {std})")
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model)(input_ids=batch, labels=batch)
    require(explanation.graph_break_count == 0,
            f"{arm}: the compiled training forward has {explanation.graph_break_count} graph breaks "
            f"{[str(reason.reason) for reason in explanation.break_reasons]}")
    print(f"{arm}: GRT noise off in eval mode (bitwise equal eager and compiled forwards), on in training mode; "
          f"compiled training forward is one graph")


def check_short_run(arm: str, eval_blocks: torch.Tensor) -> None:
    # The validation blocks stand in for the train split: the check needs real tokens, not the 3.3 GB train cache.
    train_blocks = eval_blocks[:SHORT_RUN_STEPS * train.BATCH_SIZE]
    with tempfile.TemporaryDirectory() as out_dir:
        result, trained = train.run_seed(arm, 0, 1, DEVICE, VOCAB_SIZE, 0, train_blocks,
                                         eval_blocks[-4 * train.BATCH_SIZE:], Path(out_dir))
        metrics = json.loads((Path(out_dir) / "metrics.json").read_text())
        saved = torch.load(Path(out_dir) / "model.pt", map_location=DEVICE)
    require(metrics["code_sha256"] == train.CODE_SHA256 and metrics["steps"] == SHORT_RUN_STEPS,
            f"{arm}: metrics.json provenance {metrics['code_sha256'][:12]} / {metrics['steps']} steps")
    state = trained.state_dict()
    require(saved.keys() == state.keys() and all(torch.equal(saved[name], state[name]) for name in state),
            f"{arm}: model.pt is not the final state_dict")
    loss = result["final_eval_loss"]
    require(torch.isfinite(torch.tensor(loss)).item() and loss < 6.0,
            f"{arm}: eval loss {loss} after {SHORT_RUN_STEPS} steps (ln 512 = 6.24 at uniform)")
    print(f"{arm}: {SHORT_RUN_STEPS} real steps through run_seed, eval loss {loss:.4f}")


def check_training_loop(arm: str, eval_blocks: torch.Tensor) -> None:
    """run_seed against a loop written independently from the protocol: AdamW at LR with the weight-decay
    exemptions of NO_DECAY, linear warmup then cosine to 0, gradient clipping at 1.0, data order from a randperm
    seeded with the seed. The warmup is shortened so both parts of the schedule run. The reference uses the same
    compiled model and fused AdamW: some gradients (the Hyperloop gates) are below Adam's eps of 1e-8, where the
    update magnifies the 1e-12 differences between compiled and eager code (checked in check_against_stock)."""
    spec = train.ARMS[arm]
    warmup, steps, seed = 5, TRAINING_LOOP_STEPS, 1
    train_blocks = eval_blocks[:steps * train.BATCH_SIZE]
    # Both loops get the same fixed Parcae start state and GRT noise, so the result does not depend on generator order.
    like = torch.empty(train.BATCH_SIZE, train.CONTEXT_LENGTH, train.HIDDEN_SIZE, device=DEVICE)
    noise = train.ParcaeLoop.initial_state(like)
    grt_noise = tuple(part * train.GRT_NOISE_STD for part in (torch.randn(spec.passes, *like.shape, device=DEVICE),
                                                             torch.randn(spec.passes, *like.shape[:-1], device=DEVICE)))
    saved = train.WARMUP_STEPS, train.ParcaeLoop.initial_state, train.GRTLoop.draw_noise
    saved_deterministic = (torch.are_deterministic_algorithms_enabled(),
                           torch.is_deterministic_algorithms_warn_only_enabled())
    train.WARMUP_STEPS = warmup
    train.ParcaeLoop.initial_state = staticmethod(lambda like: noise[:like.shape[0]].to(like.dtype))
    train.GRTLoop.draw_noise = lambda loop, like: tuple(part[:, :like.shape[0]].to(like.dtype) for part in grt_noise)
    # Compiled embedding backward uses atomic adds in random order; recursive_x8 increases that 1e-8 noise to 3e-3.
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            _, trained = train.run_seed(arm, seed, 1, DEVICE, VOCAB_SIZE, 0, train_blocks,
                                        eval_blocks[:train.BATCH_SIZE], Path(out_dir))
        train.WARMUP_STEPS = saved[0]
        reference_trained = _reference_training(spec, seed, warmup, steps, train_blocks)
    finally:
        train.WARMUP_STEPS, train.ParcaeLoop.initial_state, train.GRTLoop.draw_noise = (
            saved[0], staticmethod(saved[1]), saved[2])
        torch.use_deterministic_algorithms(saved_deterministic[0], warn_only=saved_deterministic[1])
    trained_params = dict(trained.named_parameters())
    named = dict(reference_trained.named_parameters())
    worst_name = max(named, key=lambda name: rel_err(trained_params[name], named[name]))
    worst = rel_err(trained_params[worst_name], named[worst_name])
    require(worst < TRAINING_LOOP_TOL, f"{arm}: run_seed vs reference loop after {steps} steps, worst param rel err "
            f"{worst:.1e} ({worst_name})")
    print(f"{arm}: run_seed = independent training loop after {steps} steps, worst param rel err {worst:.1e}")


def _reference_training(spec, seed: int, warmup: int, steps: int, train_blocks: torch.Tensor) -> nn.Module:
    """The protocol of run_seed written out by hand."""
    torch.manual_seed(seed)
    reference = train.build_model(spec, VOCAB_SIZE).to(DEVICE)
    torch._dynamo.reset()
    compiled = torch.compile(reference)
    named = dict(reference.named_parameters())
    exempt = [named[name] for name in sorted(NO_DECAY[spec.loop])]
    decayed = [p for name, p in named.items() if name not in NO_DECAY[spec.loop]]
    groups = [{"params": decayed}] + ([{"params": exempt, "weight_decay": 0.0}] if exempt else [])
    optimizer = torch.optim.AdamW(groups, lr=train.LR, fused=True)

    def learning_rate(step: int) -> float:
        if step < warmup:
            return train.LR * step / warmup
        return train.LR * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (steps - warmup)))

    shuffle = torch.Generator(device=DEVICE)
    shuffle.manual_seed(seed)
    order = torch.randperm(train_blocks.shape[0], device=DEVICE, generator=shuffle)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = train_blocks[order[step * train.BATCH_SIZE:(step + 1) * train.BATCH_SIZE]].long()
        compiled(input_ids=batch, labels=batch).loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return reference


def check_data_parallel(arm: str, batch: torch.Tensor) -> None:
    """The gradient run_seed uses with world_size = 2, the mean of the two half-batch gradients, against the
    full-batch gradient. The all-reduce itself runs only on several GPUs (the Kaggle smoke run)."""
    model = build(arm, batch.shape[0])
    noisy = [module for module in model.modules() if hasattr(module, "noise_offset")]
    half = batch.shape[0] // 2
    _, full = loss_and_grads(model, batch)
    _, first = loss_and_grads(model, batch[:half])
    for loop in noisy:
        loop.noise_offset = half
    _, second = loss_and_grads(model, batch[half:])
    worst = max(rel_err((first[name] + second[name]) / 2, full[name]) for name in full)
    require(worst < REL_TOL, f"{arm}: mean of half-batch gradients vs full batch, worst rel err {worst:.1e}")
    print(f"{arm}: mean of the two half-batch gradients = full-batch gradient, worst rel err {worst:.1e}")


def main() -> None:
    arms = sys.argv[1:] or list(SPECS)
    unknown = [arm for arm in arms if arm not in SPECS]
    require(not unknown, f"unknown arms {unknown}")
    eval_blocks = train.to_blocks(train.load_packed("validation")).to(DEVICE)
    batches = [eval_blocks[start:start + 32].long() for start in range(0, 128, 32)]
    batch = batches[0]
    for arm in arms:
        check_structure(arm)
        check_unrolled(arm, batch)
        check_against_stock(arm, batches, off_init=False)
        check_against_stock(arm, batches, off_init=True)
        check_precision(arm, batches)
        check_causality(arm, batch)
    if any(SPECS[arm].loop == "parcae" for arm in arms):
        check_parcae_noise()
    for arm in arms:
        if SPECS[arm].loop == "grt":
            check_grt_noise(arm, batch)
    short_runs = [arm for arm in arms if arm in train.ARMS] if sys.argv[1:] else ["recursive_x2", "real_18"]
    for arm in short_runs:
        check_short_run(arm, eval_blocks)
        check_training_loop(arm, eval_blocks)
        check_data_parallel(arm, batch)
    print("recursion check: PASS")


if __name__ == "__main__":
    main()
