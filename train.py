"""Train GPT-Neo on TinyStories with its middle layers looped, against GPT-Neo models whose layers are all unique
at the same effective depth. Each run writes results/<arm>/seed_<seed>/metrics.json and the final weights in
model.pt next to it; a smoke run (--max-steps-per-epoch) writes to results_smoke/ instead."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
import triton
from torch import nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoSelfAttention

from attention_kernel import gpt_neo_attention
from grt_kernel import grt_update
from hyperloop_kernel import hyperloop_read
from mesh_kernel import mesh_step
from prepare_data import DATA_SHA256, token_sha256
from provenance import code_sha256

ROOT = Path(__file__).parent
TOKENIZER_DIR = ROOT / "tokenizer"
PACKED_DIR = ROOT / "packed_cache"
RESULTS_DIR = ROOT / "results"
SMOKE_RESULTS_DIR = ROOT / "results_smoke"

VRAM_CAP_GB = 15


def apply_vram_cap(device: torch.device, gb: float = VRAM_CAP_GB) -> None:
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_memory
    fraction = min(1.0, gb * (1 << 30) / total)  # smaller GPUs: use it all instead of crashing
    torch.cuda.set_per_process_memory_fraction(fraction, idx)


CONTEXT_LENGTH = 128
HIDDEN_SIZE = 16
NUM_HEADS = 4    # TinyStories-1M's head_dim=4 (64/16), not its raw head count (would give head_dim=1 here)
WINDOW_SIZE = 16  # TinyStories-1M's window/context ratio (256/2048), scaled to this context_length
BATCH_SIZE = 640
LR = 1e-2
WARMUP_STEPS = 500
EPOCHS = 3
# A weight gradient sums over all batch*T tokens (81,920) into a 16 x 16 to 64 x 16 matrix; cuBLAS gives that sum
# to few threads. 256 partial sums in one batched matmul: 18-layer step 38.6 -> 29.0 ms on an RTX 5080 Laptop GPU.
WEIGHT_GRAD_CHUNKS = 256



@dataclass(frozen=True)
class Arm:
    """The first and the last layer run once and the middle layers run in order on each pass, so the effective depth
    is 2 + (unique_layers - 2) * passes. loop is how the passes connect: "plain", "reinject" (ReinjectedLoop),
    "hyperloop" (HyperLoop), "parcae" (ParcaeLoop), "grt" (GRTLoop) or "mesh" (MeSHLoop). residual_scale and peri_ln change the
    residual branches of the looped layers (LoopedBlock)."""
    unique_layers: int
    passes: int
    loop: str = "plain"
    residual_scale: bool = False
    peri_ln: bool = False


# The 4-layer baseline is the gelu_new arm of the activation comparison: the same model and hyperparameters, but there
# another data order per seed, no torch.compile, non-fused AdamW, stock HF attention and nn.Linear; seeds do not pair.
ARMS = {
    "recursive_x2": Arm(4, 2),
    "real_6": Arm(6, 1),
    "recursive_x4": Arm(4, 4),
    "real_10": Arm(10, 1),
    "recursive_x8": Arm(4, 8),
    "real_18": Arm(18, 1),
    "reinject_x8": Arm(4, 8, "reinject"),
    "hyperloop_x8": Arm(4, 8, "hyperloop"),
    "parcae_x8": Arm(4, 8, "parcae"),
    "grt_x8": Arm(4, 8, "grt"),
    "mesh_x8": Arm(4, 8, "mesh"),
}
LAYER_NORM_EPS = 1e-5  # GPTNeoConfig's layer_norm_epsilon, for the norms these methods add
# Parcae official code (github.com/sandyresearch/parcae, parcae_lm/models/parcae/init.py): the keep factor
# exp(-dt A) starts at sqrt(1/5) in every channel.
PARCAE_INIT_DECAY = 0.2 ** 0.5
# The code's start state is truncated N(0, INIT_STD) noise cut at 3 std. Drawn as torch.nn.init.trunc_normal_ draws
# it (erfinv of a uniform in (-erf(3/sqrt 2), erf(3/sqrt 2))), whose .item() calls break the compiled graph.
PARCAE_NOISE_BOUND = math.erf(3 / math.sqrt(2))
GRT_GATE_BIAS_INIT = 4.0  # g = sigmoid(4) = 0.98 at the start (official GRT code, model.py, gate_init_bias)
# sigma_x = sigma_g = 0.1 in all GRT runs (arXiv:2608.15062, Sec. 4; official code config/_common.py, noise_std).
GRT_NOISE_STD = 0.1
INIT_STD = 0.02  # GPTNeoConfig's default initializer_range
HYPERLOOP_STREAMS = 4  # n, the paper's setting (Zeitoun et al. 2026, Sec. 4)
HYPERLOOP_GATE_INIT = 0.01  # alpha_pre, alpha_post, alpha_res: mHC, Xie et al. 2025, arXiv:2512.24880, Table 5
RMS_NORM_EPS = torch.finfo(torch.float32).eps  # F.rms_norm's default for fp32, the training dtype
# a in w = sigmoid(a): 4.6 gives w = 0.99, the initial value of Marchenko et al. 2026 (arXiv:2610.00673, App. F.2).
REINJECT_INIT_LOGIT = 4.6


def exempt_from_weight_decay(*parameters: nn.Parameter) -> None:
    """Marks parameters of a loop method that its official code (or its paper, where no code exists) trains without
    weight decay; run_seed puts them in an AdamW group with weight_decay 0."""
    for parameter in parameters:
        parameter._no_weight_decay = True


def layer_order(unique_layers: int, passes: int) -> list[int]:
    return [0] + list(range(1, unique_layers - 1)) * passes + [unique_layers - 1]


def effective_depth(arm: str) -> int:
    return len(layer_order(ARMS[arm].unique_layers, ARMS[arm].passes))


def assign_jobs(jobs: list[tuple[str, int]], gpus: int) -> list[list[tuple[str, int]]]:
    """Longest job first onto the least-loaded GPU. A job's cost is its effective depth: the T4 step time is
    45, 72 and 126 ms at depth 6, 10 and 18."""
    loads = [0] * gpus
    groups: list[list[tuple[str, int]]] = [[] for _ in range(gpus)]
    for arm, seed in sorted(jobs, key=lambda job: -effective_depth(job[0])):
        index = loads.index(min(loads))
        groups[index].append((arm, seed))
        loads[index] += effective_depth(arm)
    return groups


class _ChunkedWeightGradLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        chunks = math.gcd(flat_inputs.shape[0], WEIGHT_GRAD_CHUNKS)
        partial = torch.bmm(flat_grad.reshape(chunks, -1, flat_grad.shape[-1]).transpose(1, 2),
                            flat_inputs.reshape(chunks, -1, flat_inputs.shape[-1]))
        grad_bias = flat_grad.sum(0) if ctx.has_bias else None
        return grad_output @ weight, partial.sum(0), grad_bias


class ChunkedGradLinear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _ChunkedWeightGradLinear.apply(inputs, self.weight, self.bias)


class ReinjectedLoop(nn.Module):
    """The passes through the middle layers with input re-injection (Marchenko et al. 2026, arXiv:2610.00673,
    Sec. 4.1, Eq. 1): pass t > 1 starts from w e + (1 - w) h_{t-1}, where h_{t-1} is the output of the previous
    pass, e is the input of the first pass, and w = sigmoid(a) is one learned scalar shared by all passes. The
    paper loops the whole decoder, so its e is the token embedding; here the loop starts after layer 1, so e is
    the output of layer 1. The paper trains a only without weight decay ("Weight decay is 0.1 for attention and MLP
    projections and for LoRA parameters, and zero otherwise", App. F.4). The paper's shared exit gate and its
    exit-weighted, smoothed loss over all passes (Eq. 2, Table 16) are not used, because the protocol trains on the
    output of the last pass only. a = 4.6 is the paper's value for a pretrained model that it converts into a
    looped model (Sec. 4.1, App. F.2), not for a model trained from the start."""

    def __init__(self, blocks: list[nn.Module], passes: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.passes = passes
        self.mix_logit = nn.Parameter(torch.tensor(REINJECT_INIT_LOGIT))
        exempt_from_weight_decay(self.mix_logit)

    def forward(self, hidden_states, **kwargs):
        loop_input = hidden_states
        mix = torch.sigmoid(self.mix_logit)
        for pass_index in range(self.passes):
            if pass_index > 0:
                hidden_states = mix * loop_input + (1 - mix) * hidden_states
            for block in self.blocks:
                hidden_states = block(hidden_states, **kwargs)[0]
        return (hidden_states,)


class HyperLoop(nn.Module):
    """The passes through the middle layers as a Hyperloop (Zeitoun, Torroba-Hennigen, Kim 2026, arXiv:2604.21254,
    Sec. 3): the loop input is copied into n streams y; each pass l has its own hyper-connection

        z = RMSNorm(flatten(y))
        H_pre = sigmoid(a_pre (W_pre z) + b_pre),  H_post = 2 sigmoid(a_post (W_post z) + b_post)
        H_res = diag(sigmoid(a_res (W_res z) + b_res))
        y <- H_res y + H_post (F(H_pre y) + e_l)

    with F the middle layers (their own skip connections kept) and e_l a learned loop position embedding; the
    streams are averaged after the last pass. The paper reports two init schemes: PyTorch defaults for every weight
    (main results), and in App. C N(0, 0.02) for every weight except the layers that write to the residual stream,
    which get N(0, 0.02 / sqrt(2 * unrolled depth)). W and e_l use the N(0, 0.02) of App. C; the shared GPT-Neo base
    keeps the unscaled N(0, 0.02) of Hugging Face for every weight. a starts at 0.01 as in mHC (Table 5). The biases
    start at 0, because neither paper states their init. The paper states weight decay 0.1 with no exemption
    (App. A) and has no public code, so no parameter is exempt."""

    def __init__(self, blocks: list[nn.Module], passes: int, width: int, streams: int = HYPERLOOP_STREAMS):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.passes, self.streams = passes, streams
        flat = streams * width
        # Rows pre, post, res of each pass, stacked: one parameter set per pass, as the paper's loop-specific HCs.
        self.weight = nn.Parameter(torch.empty(passes, 3, streams, flat).normal_(0.0, INIT_STD))
        self.bias = nn.Parameter(torch.zeros(passes, 3, streams))
        self.gate = nn.Parameter(torch.full((passes, 3), HYPERLOOP_GATE_INIT))
        self.norm_weight = nn.Parameter(torch.ones(passes, flat))
        self.loop_embedding = nn.Parameter(torch.empty(passes, width).normal_(0.0, INIT_STD))

    def read(self, streams: torch.Tensor, pass_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """mixed = sum_k H_pre[k] y_k, H_post and H_res of one pass, from the fused kernels."""
        lead, width = streams.shape[:-2], streams.shape[-2] * streams.shape[-1]
        mixed, write, keep = hyperloop_read(
            streams.reshape(-1, width), self.weight[pass_index].flatten(0, 1),
            self.gate[pass_index].repeat_interleave(self.streams), self.bias[pass_index].flatten(),
            self.norm_weight[pass_index], self.streams, RMS_NORM_EPS)
        return mixed.view(*lead, -1), write.view(*lead, -1), keep.view(*lead, -1)

    def forward(self, hidden_states, **kwargs):
        streams = hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], self.streams, -1)
        for pass_index in range(self.passes):
            mixed, write, keep = self.read(streams, pass_index)
            for block in self.blocks:
                mixed = block(mixed, **kwargs)[0]
            output = mixed + self.loop_embedding[pass_index]
            streams = keep.unsqueeze(-1) * streams + (2.0 * write).unsqueeze(-1) * output.unsqueeze(-2)
        return (streams.mean(-2),)


class ParcaeLoop(nn.Module):
    """The passes through the middle layers as Parcae (Prairie, Novack, Berg-Kirkpatrick, Fu 2026,
    arXiv:2604.12946), as its official code computes it (github.com/sandyresearch/parcae, DiagonalInjection and
    core_block_forward): with e = LN(loop input), each pass is

        x <- exp(-dt A) x + dt B e,   x <- F(x)

    with A = exp(log_A) > 0 and dt = softplus(dt_bias) per channel, so the keep factor exp(-dt A) is in (0, 1),
    and F the middle layers (their own skip connections kept). The state starts as small noise, and C maps the
    last state to the input of the next layer. Init as the code: log_A = 0, dt for a keep factor of sqrt(1/5),
    B = I. C is an orthogonal matrix times std * sqrt(width), the code's rule (parcae_lm/utils/init.py,
    wrapped_trunc_ortho; init_orthogonal=True in every released config) with this model's init std; the code's
    trunc_orthogonal_ takes Q of a truncated normal and torch's orthogonal_ Q of a normal, and both give every
    singular value equal to the gain. The paper's text says N(0, 2/(5d)) instead; this arm follows the code. The
    start noise also uses this model's init std. The code exempts log_A, dt_bias, B and C from weight decay
    (_no_weight_decay). Its launch path (MuonAdamW: recpre/optim/muon_adamw.py:107,121 and
    recpre/optim/__init__.py:24) puts the norms and these parameters in an AdamW group whose weight decay is 0.0 in
    every launch config (paper Tables 20-21), so the input norm is exempt too. Its per-sequence loop-count sampling
    and truncated backpropagation are training-protocol changes and are not used."""

    def __init__(self, blocks: list[nn.Module], passes: int, width: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.passes = passes
        self.input_norm = nn.LayerNorm(width, eps=LAYER_NORM_EPS)
        self.log_A = nn.Parameter(torch.zeros(width))
        dt = -math.log(PARCAE_INIT_DECAY)
        self.dt_bias = nn.Parameter(torch.full((width,), dt + math.log(-math.expm1(-dt))))  # inverse softplus
        self.B = nn.Parameter(torch.eye(width))
        self.C = nn.Parameter(nn.init.orthogonal_(torch.empty(width, width), gain=INIT_STD * math.sqrt(width)))
        exempt_from_weight_decay(self.log_A, self.dt_bias, self.B, self.C, self.input_norm.weight,
                                 self.input_norm.bias)

    @staticmethod
    def initial_state(like: torch.Tensor) -> torch.Tensor:
        """The code's "like-init" start: truncated N(0, std) noise, std the embedding init std."""
        uniform = (2 * torch.rand_like(like) - 1) * PARCAE_NOISE_BOUND
        return (torch.erfinv(uniform) * (INIT_STD * math.sqrt(2))).clamp(-3 * INIT_STD, 3 * INIT_STD)

    def forward(self, hidden_states, **kwargs):
        injected = F.linear(self.input_norm(hidden_states), self.B)
        dt = F.softplus(self.dt_bias)
        keep = torch.exp(-dt * torch.exp(self.log_A))
        state = self.initial_state(hidden_states)
        for _ in range(self.passes):
            state = keep * state + dt * injected
            for block in self.blocks:
                state = block(state, **kwargs)[0]
        return (F.linear(state, self.C),)


class MeSHLoop(nn.Module):
    """The first layer and the passes through the middle layers with a MeSH memory buffer (Yu et al. 2026,
    arXiv:2510.07739, Sec. 3.2 and App. C; official code github.com/LivingFutureLab/MeSH, modeling_gpt_neox.py).
    The buffer has B = passes + 3 slots per token, m_0 = the embeddings e and the others 0. Each step s (the first
    layer, then each pass) has input x_s, output y_s of its layers, and its own routers:

        w_s = softmax(W_write^s x_s + b_write^s),   r_s = softmax(W_read^s x_s + b_read^s)   (over the B slots)
        m_b <- m_b + w_s,b y_s,   x_{s+1} = sum_b r_s,b m_b

    and the last x goes to the last layer. The first step's routers read e, as the code does (the paper's
    Algorithm 2 feeds them the first layer's output). The routers use this model's N(0, 0.02) init with zero
    biases, as the code uses its own init scheme for them. The code trains with the Hugging Face Trainer, whose
    default optimizer exempts biases from weight decay, so the router biases are exempt.

    The buffer is not stored: m_b = [b = 0] e + sum_i w_i,b y_i, so sum_b r_b m_b = r_0 e + sum_i <r, w_i> y_i, a
    sum over the kept outputs with per-token weights, without B copies of the hidden states."""

    def __init__(self, first: nn.Module, blocks: list[nn.Module], passes: int, width: int):
        super().__init__()
        self.first = first
        self.blocks = nn.ModuleList(blocks)
        self.passes = passes
        self.slots = passes + 3  # the paper's B = N_loop + 3 (App. E.2)
        # Step 0 is the first layer, step p + 1 is pass p; rows are the write logits, then the read logits.
        self.router_weight = nn.Parameter(torch.empty(passes + 1, 2 * self.slots, width).normal_(0.0, INIT_STD))
        self.router_bias = nn.Parameter(torch.zeros(passes + 1, 2 * self.slots))
        exempt_from_weight_decay(self.router_bias)

    def route(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Write and read weights of a step."""
        logits = _ChunkedWeightGradLinear.apply(state, self.router_weight[step], self.router_bias[step])
        # Two softmaxes, not one over logits.unflatten(-1, (2, B)) then unbind: TorchInductor in torch 2.13.0
        # gives wrong gradients for that form (25-36% error on the embeddings; check.py compiled vs eager).
        return torch.softmax(logits[..., :self.slots], dim=-1), torch.softmax(logits[..., self.slots:], dim=-1)

    def forward(self, hidden_states, **kwargs):
        lead, width = hidden_states.shape[:-1], hidden_states.shape[-1]
        embeddings = hidden_states.reshape(-1, width)
        write, read = self.route(embeddings, 0)
        state, writes, kept = hidden_states, [], []
        for step in range(self.passes + 1):
            if step == 0:
                output = self.first(state, **kwargs)[0]
            else:
                output = state
                for block in self.blocks:
                    output = block(output, **kwargs)[0]
            writes.append(write)
            kept.append(output.reshape(-1, width))
            has_next = step < self.passes
            # After the last pass no router is needed; the kernels ignore the weight passed.
            router = step + 1 if has_next else step
            flat, write, read = mesh_step(embeddings, kept, writes, read, self.router_weight[router],
                                          self.router_bias[router], self.slots, has_next)
            state = flat.view(*lead, width)
        return (state,)


class GRTLoop(nn.Module):
    """The passes through the middle layers as a Gated Recurrent Transformer (Hegazy, Alanwar, Elhoushi 2026,
    arXiv:2608.15062, Sec. 3), as its official code computes it (github.com/Amr-Hegazy1/gated-recurrent-transformer,
    model.py). With h_pre the loop input and h_0 = h_pre, each pass is (Eqs. 2-5)

        o = F(W_proj [h + eps_x; h_pre]),   g = sigmoid(f_g(LN_2C([LN_x(h); LN_h(h_pre)])) / tau + eps_g),
        h <- g h + (1 - g) o

    with F the middle layers, f_g = Linear(2C -> C, no bias) -> SiLU -> Linear(C -> C) with its bias at +4, and
    tau = 1. As in the code (bias = False), W_proj and the first gate layer have no bias and the three LayerNorms
    have a gain and no bias; Linear weights start at N(0, 0.02). The code exempts every 1-D parameter (the gains and
    the gate bias) from weight decay.

    The state noise eps_x ~ N(0, 0.1^2 I) and the gate noise eps_g ~ N(0, 0.1^2) are part of the method and are
    drawn again on every pass. As in the code (model.py, forward), eps_x goes only into the input of W_proj, so the
    gate and the carried h stay clean, and eps_g is one value per token for all channels (randn_like(
    gate_features[..., :1])); the paper's text says "per-scalar". The noise is drawn only in training mode: the
    paper injects eps_g "during training" (Sec. 3.3), runs its diagnostics "with inference noise disabled" (App. E),
    and the code's config says both noises are "Applied during training only" (model.py, GPTConfig.noise_std),
    although its forward does not test self.training. The official validation loss keeps the noise (train.py,
    estimate_loss: model.eval(), then model(X, Y, ...)); this arm evaluates without noise, a choice that is the same
    for every arm. The random loop count r ~ U{1, ..., R} (Sec. 3.4) is not used, because it
    changes the training protocol, which fixes the passes of each arm."""

    def __init__(self, blocks: list[nn.Module], passes: int, width: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.passes = passes
        self.project = nn.Linear(2 * width, width, bias=False)
        self.state_norm = nn.LayerNorm(width, eps=LAYER_NORM_EPS, bias=False)
        self.input_norm = nn.LayerNorm(width, eps=LAYER_NORM_EPS, bias=False)
        self.joint_norm = nn.LayerNorm(2 * width, eps=LAYER_NORM_EPS, bias=False)
        self.gate_hidden = nn.Linear(2 * width, width, bias=False)
        self.gate_out = nn.Linear(width, width)
        for linear in (self.project, self.gate_hidden, self.gate_out):
            nn.init.normal_(linear.weight, 0.0, INIT_STD)
        nn.init.constant_(self.gate_out.bias, GRT_GATE_BIAS_INIT)
        exempt_from_weight_decay(self.state_norm.weight, self.input_norm.weight, self.joint_norm.weight,
                                 self.gate_out.bias)

    def forward(self, hidden_states, **kwargs):
        # W_proj [h; h_pre] = W_h h + W_pre h_pre: the h_pre half is the same on every pass, so it is computed once.
        loop_input = hidden_states
        width = loop_input.shape[-1]
        project = self.project.weight
        injected = _ChunkedWeightGradLinear.apply(loop_input, project[:, width:], None)
        normed_input = self.input_norm(loop_input)
        noise = self.noise(loop_input)
        state = loop_input
        for pass_index in range(self.passes):
            projected = state if noise is None else state + noise[0][pass_index]
            output = _ChunkedWeightGradLinear.apply(projected, project[:, :width], None) + injected
            for block in self.blocks:
                output = block(output, **kwargs)[0]
            state = self.update(state, output, normed_input, None if noise is None else noise[1][pass_index])
        return (state,)

    def noise(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """eps_x and eps_g of every pass in training mode, None in eval mode."""
        if not self.training:
            return None
        return self.draw_noise(like)

    def draw_noise(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """eps_x (passes, *like.shape) and eps_g (passes, *like.shape[:-1])."""
        state_noise = torch.randn(self.passes, *like.shape, device=like.device, dtype=like.dtype)
        gate_noise = torch.randn(self.passes, *like.shape[:-1], device=like.device, dtype=like.dtype)
        return state_noise * GRT_NOISE_STD, gate_noise * GRT_NOISE_STD

    def update(self, state: torch.Tensor, output: torch.Tensor, normed_input: torch.Tensor,
               gate_noise: torch.Tensor | None) -> torch.Tensor:
        """g h + (1 - g) o from the fused kernels."""
        width = state.shape[-1]
        return grt_update(state.reshape(-1, width), output.reshape(-1, width), normed_input.reshape(-1, width),
                          self.state_norm.weight, self.joint_norm.weight, self.gate_hidden.weight,
                          self.gate_out.weight, self.gate_out.bias, self.state_norm.eps,
                          None if gate_noise is None else gate_noise.reshape(-1)).view_as(state)


class LoopedBlock(GPTNeoBlock):
    """A GPT-Neo block of the looped middle layers with changed residual branches, h <- h + s * N_out(f(LN(h))) for
    the attention and the MLP branch f:

    - Residual scaling (Wang et al. 2026, arXiv:2606.18524, Table 3): s = lambda / N * (L / L_ref)^(-1/2) with
      lambda = 1 and N the passes. The learning rate of this proxy was set on the 4-layer model whose middle pair
      is looped, so L = L_ref, s = 1/N, and the paper's learning-rate factor (L / L_ref)^(-1/2) is 1.
    - Peri-LN (Kim et al. 2025, arXiv:2502.02732, Eq. 3): N_out is a LayerNorm (gain 1, bias 0 at init); otherwise
      N_out is the identity."""

    def forward(self, hidden_states, layer_past=None, attention_mask=None, use_cache=False, output_attentions=False,
                **kwargs):
        attn_output = self.attn(self.ln_1(hidden_states), layer_past=layer_past, attention_mask=attention_mask,
                                use_cache=use_cache, output_attentions=output_attentions)[0]
        hidden_states = hidden_states + self.branch_scale * self.attn_out_norm(attn_output)
        feed_forward = self.mlp(self.ln_2(hidden_states))
        return hidden_states + self.branch_scale * self.mlp_out_norm(feed_forward), None


def make_looped_block(block: GPTNeoBlock, branch_scale: float, peri_ln: bool) -> None:
    block.__class__ = LoopedBlock
    block.branch_scale = branch_scale
    for name in ("attn_out_norm", "mlp_out_norm"):
        setattr(block, name, nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS) if peri_ln else nn.Identity())


class GPTNeoFusedAttention(GPTNeoSelfAttention):
    """GPTNeoSelfAttention with q, k and v from one 16 -> 48 projection of the concatenated weights (one read of the
    input, one weight-gradient matmul) and the masked softmax(q k^T) v, without a 1/sqrt(d) scale, as Triton kernels."""

    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=False, output_attentions=False,
                **kwargs):
        # The model passes only its causal mask (no padding), and the kernels apply it with the local window.
        if layer_past is not None or output_attentions:
            raise ValueError("the fused attention has no key/value cache and returns no attention weights")
        weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = _ChunkedWeightGradLinear.apply(hidden_states, weight, None)
        query, key, value = (self._split_heads(part, self.num_heads, self.head_dim)
                             for part in qkv.split(self.embed_dim, dim=-1))
        length = query.shape[-2]
        window = self.config.window_size if self.attention_type == "local" else length
        attn_output = self._merge_heads(gpt_neo_attention(query, key, value, window), self.num_heads, self.head_dim)
        return self.resid_dropout(self.out_proj(attn_output)), None


def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_DIR / "tokenizer.json"),
        bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>",
    )


def build_model(arm: Arm, vocab_size: int, eos_token_id: int = 0) -> GPTNeoForCausalLM:
    unique_layers, passes = arm.unique_layers, arm.passes
    config = GPTNeoConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=unique_layers,
        num_heads=NUM_HEADS,
        attention_types=[[["global", "local"], unique_layers // 2]],
        window_size=WINDOW_SIZE,
        max_position_embeddings=CONTEXT_LENGTH,
        activation_function="gelu_new",
        attention_dropout=0, embed_dropout=0, resid_dropout=0,
        tie_word_embeddings=True,
        # A looped layer has one cache slot, so a cache would feed the keys of its earlier pass into its later pass.
        use_cache=False,
        bos_token_id=eos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id,
    )
    model = GPTNeoForCausalLM(config)
    blocks = model.transformer.h
    if arm.residual_scale or arm.peri_ln:
        for block in blocks[1:-1]:
            make_looped_block(block, 1.0 / passes if arm.residual_scale else 1.0, arm.peri_ln)
    middle = list(blocks[1:-1])
    if arm.loop == "reinject":
        model.transformer.h = nn.ModuleList([blocks[0], ReinjectedLoop(middle, passes), blocks[-1]])
    elif arm.loop == "hyperloop":
        model.transformer.h = nn.ModuleList([blocks[0], HyperLoop(middle, passes, HIDDEN_SIZE), blocks[-1]])
    elif arm.loop == "parcae":
        model.transformer.h = nn.ModuleList([blocks[0], ParcaeLoop(middle, passes, HIDDEN_SIZE), blocks[-1]])
    elif arm.loop == "grt":
        model.transformer.h = nn.ModuleList([blocks[0], GRTLoop(middle, passes, HIDDEN_SIZE), blocks[-1]])
    elif arm.loop == "mesh":
        model.transformer.h = nn.ModuleList([MeSHLoop(blocks[0], middle, passes, HIDDEN_SIZE), blocks[-1]])
    elif arm.loop != "plain":
        raise ValueError(f"unknown loop {arm.loop!r}")
    else:
        # A module at several positions shares one set of weights; parameters() gives each tensor once.
        model.transformer.h = nn.ModuleList([blocks[index] for index in layer_order(unique_layers, passes)])
    for block in blocks:
        block.attn.attention.__class__ = GPTNeoFusedAttention
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = ChunkedGradLinear
    return model


def load_packed(split: str) -> torch.Tensor:
    """The packed token stream of a split from prepare_data.py; it must be the stream every reported run used."""
    ids = torch.load(PACKED_DIR / f"{split}.pt")
    digest = token_sha256(ids)
    if digest != DATA_SHA256[split]:
        raise RuntimeError(f"{PACKED_DIR / split}.pt: token SHA-256 {digest}, expected {DATA_SHA256[split]}")
    return ids


# Hashed at import, when Python and Triton load the code, so a file edit during a long run cannot change it.
CODE_SHA256 = code_sha256()


def to_blocks(ids: torch.Tensor) -> torch.Tensor:
    n_blocks = ids.shape[0] // CONTEXT_LENGTH
    return ids[:n_blocks * CONTEXT_LENGTH].view(n_blocks, CONTEXT_LENGTH)


def _all_reduce_mean(tensors: list[torch.Tensor], world_size: int) -> None:
    """Averages tensors over the processes in place, through one flat buffer (one collective per step)."""
    flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
    dist.all_reduce(flat)
    flat /= world_size
    offset = 0
    for tensor in tensors:
        tensor.copy_(flat[offset:offset + tensor.numel()].view_as(tensor))
        offset += tensor.numel()


def run_seed(arm: str, seed: int, epochs: int, device: torch.device, vocab_size: int, eos_token_id: int,
             train_blocks: torch.Tensor, eval_blocks: torch.Tensor, out_dir: Path, rank: int = 0,
             world_size: int = 1, max_steps_per_epoch: int | None = None) -> tuple[dict, nn.Module]:
    """One training run. With world_size > 1, each process takes its equal share of every global batch of BATCH_SIZE
    sequences and the gradients are averaged; every sequence has CONTEXT_LENGTH tokens, so this is the gradient of
    the same global mean loss as one process. max_steps_per_epoch cuts each epoch short for a smoke run."""
    torch.manual_seed(seed)
    model = build_model(ARMS[arm], vocab_size, eos_token_id).to(device)
    if rank > 0:
        # Every process builds identical weights; random draws inside the forward (the Parcae start state, the GRT
        # noise) must differ.
        torch.cuda.manual_seed(seed + 1_000_003 * rank)
    n_params = sum(p.numel() for p in model.parameters())
    unique_layers = ARMS[arm].unique_layers
    if BATCH_SIZE % world_size:
        raise ValueError(f"batch {BATCH_SIZE} does not split over {world_size} processes")
    local_batch = BATCH_SIZE // world_size

    # The blocks stay on the GPU (3.3 GB at int32), so a batch is one gather instead of 640 host-side reads.
    steps_per_epoch = train_blocks.shape[0] // BATCH_SIZE
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    shuffle = torch.Generator(device=device)
    shuffle.manual_seed(seed)

    # Each seed builds a new model, and the compiled graphs of the previous one count toward dynamo's recompile
    # limit (8), above which dynamo silently runs the model eagerly.
    torch._dynamo.reset()
    # Compiled: 18-layer step 54.1 -> 29.0 ms on an RTX 5080 Laptop GPU (fewer launches, fused elementwise ops).
    compiled = torch.compile(model)
    # The base GPT-Neo parameters decay in every arm; a loop method's own parameters follow its source (see
    # exempt_from_weight_decay).
    no_decay = [p for p in model.parameters() if getattr(p, "_no_weight_decay", False)]
    decay = [p for p in model.parameters() if not getattr(p, "_no_weight_decay", False)]
    groups = [{"params": decay}] + ([{"params": no_decay, "weight_decay": 0.0}] if no_decay else [])
    optimizer = torch.optim.AdamW(groups, lr=LR, fused=device.type == "cuda")
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    def run_eval() -> float:
        # Token-weighted mean, not a mean of batch means: the last eval batch is ragged, and equal weights would
        # count each of its tokens about twice. With several processes each takes every world_size-th batch.
        model.eval()
        sums = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for index, start in enumerate(range(0, eval_blocks.shape[0], BATCH_SIZE)):
                if index % world_size != rank:
                    continue
                batch = eval_blocks[start:start + BATCH_SIZE].long()
                n_tokens = batch.shape[0] * (batch.shape[1] - 1)  # shifted labels
                sums[0] += compiled(input_ids=batch, labels=batch).loss.double() * n_tokens
                sums[1] += n_tokens
        if world_size > 1:
            dist.all_reduce(sums)
        model.train()
        return (sums[0] / sums[1]).item()

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(train_blocks.shape[0], device=device, generator=shuffle)
        for batch_index in range(steps_per_epoch):
            rows = order[batch_index * BATCH_SIZE + rank * local_batch:batch_index * BATCH_SIZE + (rank + 1) * local_batch]
            batch = train_blocks[rows].long()
            loss = compiled(input_ids=batch, labels=batch).loss
            loss.backward()
            if world_size > 1:
                _all_reduce_mean([p.grad for p in model.parameters()], world_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0 and rank == 0:
                print(f"  {arm} seed {seed} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        eval_loss = run_eval()
        history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        if rank == 0:
            print(f"  {arm} seed {seed} epoch {epoch + 1}/{epochs} eval_loss {eval_loss:.4f}", flush=True)

    result = {
        "arm": arm, "seed": seed, "n_params": n_params,
        "unique_layers": unique_layers, "effective_depth": effective_depth(arm),
        "history": history, "final_eval_loss": history[-1]["eval_loss"],
        "wall_time_s": time.time() - t0,
        "steps": total_steps, "processes": world_size,
        "gpus": [torch.cuda.get_device_name(device)] * world_size,
        "code_sha256": CODE_SHA256, "data_sha256": DATA_SHA256,
        "versions": {"torch": torch.__version__, "triton": triton.__version__,
                     "transformers": transformers.__version__},
    }
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        torch.save(model.state_dict(), out_dir / "model.pt")
    return result, model


def _launch(jobs_per_process: list[list[tuple[str, int]]], args, world_size: int = 1) -> None:
    """One child process per entry of jobs_per_process, child i on cuda:i; waits for all of them."""
    port = str(29500 + os.getpid() % 1000)
    procs = []
    for index, group in enumerate(jobs_per_process):
        command = [sys.executable, __file__, "--epochs", str(args.epochs), "--device", f"cuda:{index}",
                   "--jobs", *[f"{arm}:{seed}" for arm, seed in group]]
        if args.max_steps_per_epoch is not None:
            command += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
        if world_size > 1:
            command += ["--rank", str(index), "--world-size", str(world_size), "--master-port", port]
        procs.append(subprocess.Popen(command))
    exit_codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"training processes exited with {exit_codes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    # arm:seed pairs; the parent process gives each GPU process its share of the jobs this way.
    parser.add_argument("--jobs", nargs="+", default=[])
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="smoke runs only")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-port", default="29500")
    args = parser.parse_args()
    jobs = [(arm, seed) for arm in args.arm for seed in args.seeds]
    for job in args.jobs:
        arm, seed = job.split(":")
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r} in --jobs")
        jobs.append((arm, int(seed)))
    if not jobs:
        parser.error("give --arm or --jobs")

    gpus = torch.cuda.device_count() if args.device == "cuda" else 1
    if gpus > 1 and args.world_size == 1 and not args.jobs:
        # Kaggle's 2x T4: whole runs go to the GPUs in parallel rounds; a single run left over (3 seeds on 2 GPUs)
        # is split over all GPUs, so no GPU waits idle for it.
        rounds = len(jobs) // gpus * gpus
        if rounds:
            _launch([group for group in assign_jobs(jobs[:rounds], gpus) if group], args)
        leftover = jobs[rounds:]
        if len(leftover) == 1:
            _launch([leftover] * gpus, args, world_size=gpus)
        elif leftover:
            _launch([[job] for job in leftover], args)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        # Kernels launch on the current device, so a process given cuda:1 must make it current.
        torch.cuda.set_device(device)
    if args.world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = args.master_port
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size, device_id=device)
    apply_vram_cap(device)
    tokenizer = load_tokenizer()
    train_blocks = to_blocks(load_packed("train")).to(device)
    eval_blocks = to_blocks(load_packed("validation")).to(device)
    results_dir = RESULTS_DIR if args.max_steps_per_epoch is None else SMOKE_RESULTS_DIR
    for arm, seed in jobs:
        if args.rank == 0:
            print(f"=== arm={arm} seed={seed} processes={args.world_size} ===", flush=True)
        run_seed(arm, seed, args.epochs, device, len(tokenizer), tokenizer.eos_token_id, train_blocks, eval_blocks,
                 results_dir / arm / f"seed_{seed}", args.rank, args.world_size, args.max_steps_per_epoch)
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
