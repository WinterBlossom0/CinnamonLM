"""Canonical Kimi Delta Attention and Cinnamon's xIELU FFN.

KDA is supplied by Flash Linear Attention's differentiable ``chunk_kda``
implementation. It implements the Kimi Linear delta rule, including its
per-key-channel decay, beta write gate, short causal convolutions and gated
output normalisation. KDA carries order through its recurrent state and causal
convolution; Cinnamon deliberately applies no positional encoding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.models.utils import Cache as FLACache

from .config import Config


def _softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


class XIELU(nn.Module):
    """xIELU (Huang & Schlag, arXiv:2411.13010): alpha_p*x^2+0.5x for x>0, else
    alpha_n*(e^x-1-x)+0.5x.  Two learnable scalars per instance, softplus-
    parametrised to stay positive (alpha_n additionally floored at BETA_N).
    Won the activation_comparison sweep (gelu_new/xielu/polyrelu/swiglu/powlu,
    5 activations x 8 seeds, TinyStories): every seed beat gelu_new's best,
    seeds 4-5 reached 2.70-0.75 vs everyone else's ~2.91-2.93."""
    BETA_N = 0.5

    def __init__(self, alpha_p_init: float = 0.8, alpha_n_init: float = 0.8):
        super().__init__()
        self.raw_alpha_p = nn.Parameter(torch.tensor(_softplus_inverse(alpha_p_init)))
        self.raw_alpha_n = nn.Parameter(torch.tensor(_softplus_inverse(alpha_n_init - self.BETA_N)))

    # 14 elementwise kernels over a [.., d_ff] tensor, run ~560 times per training
    # step (every FFN, every recurrence, plus every checkpoint recompute) -- the
    # single densest pointwise chain in the model.  Inductor fuses the lot into 6
    # launches: measured 4.67 -> 0.67 ms fwd+bwd at [12,1024,1024], 64 -> 6
    # launches, 888 -> 756 MiB.  dynamic=True because the dispatch loop hands it a
    # different token count per hypernet per recurrence (measured: 2 graphs, 0
    # recompiles over varying batch and sequence).
    # Inductor donates backward buffers, so a caller wanting backward(retain_graph=
    # True) or a double backward through this must run it under
    # torch.compiler.set_stance("force_eager") -- see the gradcheck in tests.
    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.raw_alpha_p)
        alpha_n = self.BETA_N + F.softplus(self.raw_alpha_n)
        # masked per-branch: torch.where evaluates both, unmasked expm1 overflows
        # -> nan grad.  `0.0` is a scalar, not zeros_like: same value and same
        # gradient, minus a [.., d_ff] allocation and its fill kernel each.
        pos = x > 0
        x_pos = torch.where(pos, x, 0.0)
        x_neg = x - x_pos                       # == where(pos, 0, x), one kernel
        # 0.5*x_pos + 0.5*x_neg == 0.5*x, hoisted out of both branches.
        return torch.where(pos, alpha_p * x_pos * x_pos,
                           alpha_n * (torch.expm1(x_neg) - x_neg)) + 0.5 * x


class KDAStateCache(FLACache):
    """FLA's production cache protocol for Cinnamon's single KDA module.

    KDA owns a finite recurrent matrix state and three short-convolution
    states; it never owns a growing token KV tensor.  State creation and
    updates deliberately delegate unchanged to FLA's canonical ``Cache`` /
    ``FLALayer`` implementation.  The small override below supplies the
    state-aware batch reordering that FLA's generic HF cache layer cannot
    provide (its default assumes ``keys``/``values`` attributes).
    """

    @staticmethod
    def _select_batch(value, indices: torch.LongTensor):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, tuple):
            return tuple(KDAStateCache._select_batch(item, indices) for item in value)
        if isinstance(value, list):
            return [KDAStateCache._select_batch(item, indices) for item in value]
        if isinstance(value, dict):
            return {key: KDAStateCache._select_batch(item, indices) for key, item in value.items()}
        return value

    def batch_select_indices(self, indices: torch.LongTensor) -> None:
        for layer in self.layers:
            if layer.state is not None:
                layer.state = self._select_batch(layer.state, indices)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class KDA(nn.Module):
    """Kimi Delta Attention with canonical 128-dimensional heads.

    Flash Linear Attention's differentiable Triton ``chunk_kda`` path is used
    in training. Moonshot FlashKDA is inference-only and is not a backend for
    Cinnamon training.
    """

    def __init__(self, c: Config):
        super().__init__()
        if c.n_heads * c.head_dim != c.d_model:
            raise ValueError(
                "KDA requires n_heads * head_dim == d_model "
                f"(got {c.n_heads} * {c.head_dim} != {c.d_model})")
        try:
            from fla.layers.kda import KimiDeltaAttention
        except ImportError as e:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Kimi Delta Attention requires flash-linear-attention. "
                "Install the differentiable FLA backend before constructing CinnamonLM.") from e

        self.impl = KimiDeltaAttention(
            hidden_size=c.d_model,
            head_dim=c.head_dim,
            num_heads=c.n_heads,
            num_v_heads=c.n_heads,
            mode="chunk",
            use_short_conv=c.kda_short_conv,
            conv_size=c.kda_conv_size,
            safe_gate=c.kda_safe_gate,
            lower_bound=c.kda_gate_lower_bound,
            norm_eps=c.eps,
        )
        # FLA requires a stable layer index whenever a cache object is supplied.
        # A Cinnamon ``Layer`` contains exactly one KDA module.
        self.impl.layer_idx = 0

    def forward(self, x: torch.Tensor, *, attention_mask: torch.Tensor | None = None,
                past_key_values: KDAStateCache | None = None, use_cache: bool = False) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError(
                "FLA Kimi Delta Attention requires CUDA; no approximate CPU path is provided.")
        out, _, _ = self.impl(
            x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return out


class FFN(nn.Module):
    """Plain xIELU MLP, 2 * d_model * d_ff parameters (no gate matrix)."""

    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)
        self.act = XIELU()

    def forward(self, x):
        return self.down(self.act(self.up(x)))
