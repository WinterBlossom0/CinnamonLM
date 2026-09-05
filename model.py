"""Minimal transformer LM: same per-block shape as activation_comparison's
GPT-Neo replica (pre-norm -> sublayer -> residual, twice per block), with a
swappable attention sublayer (KDA, OSDN, or Parallax) in place of softmax
self-attention. KDA/OSDN carry position through their recurrence and need no
position embedding; Parallax is O(T^2) with no such carrier, so the model
adds a learned absolute position embedding only for that arm.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kda import KDA
from osdn import OSDN
from parallax import Parallax

ATTENTIONS = {"kda": KDA, "osdn": OSDN, "parallax": Parallax}
NEEDS_POSITION_EMBEDDING = {"parallax"}

INIT_STD = 0.02  # matches GPTNeoConfig's default initializer_range, used by activation_comparison


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
                 eps: float = 1e-6, context_length: int | None = None):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = None
        if attention in NEEDS_POSITION_EMBEDDING:
            self.wpe = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([
            Block(attention, d_model, d_ff, n_heads, head_dim, chunk, eps)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied, matching tie_word_embeddings=True
        self.apply(_init_weights)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.wte(input_ids)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
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
