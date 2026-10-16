"""Minimal transformer LM with GPT-Neo's block shape (pre-norm -> sublayer -> residual,
twice per block) and a swappable attention sublayer (KDA, OSDN, Parallax, or Mamba3) in place of softmax
self-attention. KDA, OSDN and Mamba3 carry position through their recurrence, and Parallax applies
RoPE to q, k and rho inside the layer (arXiv:2605.29157 Sec. 11.1), so LM has no position embedding.
The softmax arm is a GPT-Neo model of the same size (SoftmaxLM).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from kda import KDA
from osdn import OSDN
from parallax import Parallax
from mamba3 import Mamba3

ATTENTIONS = {"kda": KDA, "osdn": OSDN, "parallax": Parallax, "mamba3": Mamba3}

INIT_STD = 0.02  # GPTNeoConfig's default initializer_range, the softmax arm's init
# TinyStories-1M's local window / context ratio (256/2048) at context_length 128.
SOFTMAX_WINDOW = 16
EOS_TOKEN_ID = 0  # <|endoftext|> in tokenizer/tokenizer.json


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.act = nn.GELU(approximate="tanh")  # == HF gelu_new, verified to 1e-16
        self.c_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, attention: str, d_model: int, d_ff: int, n_heads: int,
                 head_dim: int, chunk: int, eps: float):
        super().__init__()
        self.ln_1 = nn.RMSNorm(d_model, eps=eps)
        self.attn = ATTENTIONS[attention](d_model, n_heads, head_dim, chunk=chunk, eps=eps)
        self.ln_2 = nn.RMSNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LM(nn.Module):
    def __init__(self, attention: str, vocab_size: int, d_model: int, d_ff: int,
                 num_layers: int, n_heads: int, head_dim: int, chunk: int = 64,
                 eps: float = 1e-6):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = None  # muon.split_params reads wpe from every model
        self.blocks = nn.ModuleList([
            Block(attention, d_model, d_ff, n_heads, head_dim, chunk, eps)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied, matching tie_word_embeddings=True
        self.apply(_init_weights)
        # _init_weights zeroes every Linear bias; a layer's own init (e.g. KDA's gate bias) must run after it.
        for block in self.blocks:
            post_init = getattr(block.attn, "post_init", None)
            if post_init is not None:
                post_init()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.wte(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss


class SoftmaxLM(nn.Module):
    """GPT-Neo at TinyStories-1M's shape scaled to d_model 16: global/local attention alternating, learned
    positions, gelu_new MLP."""

    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_layers: int, n_heads: int,
                 context_length: int):
        super().__init__()
        config = GPTNeoConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=d_ff,
            num_layers=num_layers,
            num_heads=n_heads,
            attention_types=[[["global", "local"], num_layers // 2]],
            window_size=SOFTMAX_WINDOW,
            max_position_embeddings=context_length,
            activation_function="gelu_new",
            attention_dropout=0, embed_dropout=0, resid_dropout=0,
            tie_word_embeddings=True,
            bos_token_id=EOS_TOKEN_ID, eos_token_id=EOS_TOKEN_ID, pad_token_id=EOS_TOKEN_ID,
        )
        self.model = GPTNeoForCausalLM(config)

    # Properties, not attributes: muon.split_params reads wte/wpe, and an attribute would register them twice.
    @property
    def wte(self) -> nn.Embedding:
        return self.model.transformer.wte

    @property
    def wpe(self) -> nn.Embedding:
        return self.model.transformer.wpe

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        out = self.model(input_ids=input_ids, labels=labels)
        return out.logits, out.loss


def build_model(attention: str, vocab_size: int, d_model: int, d_ff: int, num_layers: int,
                n_heads: int, head_dim: int, chunk: int, context_length: int) -> nn.Module:
    if attention != "softmax":
        return LM(attention, vocab_size, d_model, d_ff, num_layers, n_heads, head_dim, chunk=chunk)
    if n_heads * head_dim != d_model:
        raise ValueError(f"GPT-Neo needs n_heads * head_dim == d_model, got {n_heads} * {head_dim} != {d_model}")
    return SoftmaxLM(vocab_size, d_model, d_ff, num_layers, n_heads, context_length)
