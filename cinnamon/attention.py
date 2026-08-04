"""MLA (global attention) and SwiGLU FFN.

MLA is full attention -- every token attends to every prior token, quadratic.
The compression is only in how K and V are stored (a low-rank latent, expanded on
use), which shrinks the KV cache without changing what attention computes.

3.1 NoPE: no positional encoding anywhere.  Position enters through the KDA
layers, whose recurrent state is inherently ordered.  RoPE is not an option here
because it does not commute with MLA's low-rank KV compression -- you cannot
rotate a compressed latent, expand it later, and get the right answer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class MLA(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        h, d = c.n_heads, c.head_dim
        self.h, self.d = h, d
        self.q_a = nn.Linear(c.d_model, c.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(c.q_lora_rank, eps=c.eps)
        self.q_b = nn.Linear(c.q_lora_rank, h * d, bias=False)
        self.kv_a = nn.Linear(c.d_model, c.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(c.kv_lora_rank, eps=c.eps)
        self.kv_b = nn.Linear(c.kv_lora_rank, 2 * h * d, bias=False)
        self.o = nn.Linear(h * d, c.d_model, bias=False)
        self.qn = nn.RMSNorm(d, eps=c.eps)      # 6 QK-RMSNorm
        self.kn = nn.RMSNorm(d, eps=c.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.d
        q = self.qn(self.q_b(self.q_norm(self.q_a(x))).view(B, T, H, D)).transpose(1, 2)
        k, v = self.kv_b(self.kv_norm(self.kv_a(x))).view(B, T, H, 2 * D).chunk(2, -1)
        o = F.scaled_dot_product_attention(
            q, self.kn(k).transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, H * D))


class FFN(nn.Module):
    """SwiGLU, 3 * d_model * d_ff parameters."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
