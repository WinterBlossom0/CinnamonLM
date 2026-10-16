"""Parallax attention: Zuo, Pai, Zeng, Dewulf, Hu, Wang, "Parallax: Parameterized Local Linear
Attention", arXiv:2605.29157. The layer is Qwen3 attention (GQA, RMSNorm on q and k, RoPE) plus a
learned probe rho_i = W_R x_i. The authors' released layer is ParallaxAttention in
huggingface.co/YifeiZuo/Parallax-0.6B, modeling_parallax.py.

Per-head vectors of size d (Sec. 3.1, Sec. 4.2, Sec. 11.1, Table 5):

    q_i = RoPE_i(RMSNorm(W_Q x_i))     k_j = RoPE_j(RMSNorm(W_K x_j))     v_j = W_V x_j
    rho_i = RoPE_i(RMSNorm(W_R x_i))
    p_ij = softmax_{j <= i}(q_i . k_j / sqrt(d))

RoPE has base 1e6 for q, k and rho, and rho uses the position of its own token. W_R has the head
count and head grouping of W_Q: KV_GROUP query/probe heads share one K/V head. No projection has
a bias.

Output of head h at position i, eq. 8 (boundary amplification eta_i = 0):

    o_i = o_i^SA - Sigma_KV^(i) rho_i
    o_i^SA = sum_j p_ij v_j        kbar_i = sum_j p_ij k_j
    Sigma_KV^(i) = sum_j p_ij (v_j - o_i^SA)(k_j - kbar_i)^T

Streaming form, eq. 14 / Algorithm 1 (Sec. 3.3). The composite score p_ij (k_j . rho_i) has no
1/sqrt(d) scale:

    o_i = o_i^SA (1 + sum_j p_ij (k_j . rho_i)) - sum_j p_ij (k_j . rho_i) v_j

kernel.py computes eq. 14 in one fused pass over the keys (Algorithm 1).

W_O maps the concatenated heads to d_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .kernel import parallax_core

ROPE_THETA = 1e6  # paper Sec. 11.1 and Table 5
KV_GROUP = 2  # paper Table 5: 16 query heads share 8 KV heads


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class Parallax(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 chunk: int = 64, eps: float = 1e-6):
        super().__init__()
        del chunk  # accepted for one constructor across the arms; the scores cover the full causal context
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        if n_heads % KV_GROUP != 0:
            raise ValueError(f"n_heads must be a multiple of KV_GROUP={KV_GROUP}, got {n_heads}")
        self.n_heads, self.head_dim = n_heads, head_dim
        n_kv_heads = n_heads // KV_GROUP
        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.r = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.r_norm = nn.RMSNorm(head_dim, eps=eps)
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", 1.0 / (ROPE_THETA ** exponents), persistent=False)

    def _heads(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return proj(x).view(batch, length, -1, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)  # (T, 1, head_dim) against (B, T, H, head_dim)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        q = _apply_rope(self.q_norm(self._heads(self.q, x)), cos, sin)
        k = _apply_rope(self.k_norm(self._heads(self.k, x)), cos, sin)
        rho = _apply_rope(self.r_norm(self._heads(self.r, x)), cos, sin)
        v = self._heads(self.v, x)
        heads = parallax_core(q, k, v, rho)
        return self.o(heads.reshape(batch, length, self.n_heads * self.head_dim))
