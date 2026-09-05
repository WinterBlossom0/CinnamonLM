"""Muon: momentum orthogonalised by Newton-Schulz (Jordan et al.).

Ported from ../cinnamon/muon.py -- same optimizer, no _TABLES exclusion (this
project has no recurrence-indexed lookup tables to protect from
orthogonalisation, unlike cinnamon's phi/err_slot).

The update is the momentum buffer with its singular values flattened to 1 --
whatever direction the gradient points, every direction is stepped equally.
That is only meaningful for a matrix that acts as a linear map, so Muon
covers matmul weights ONLY. Everything else (embeddings, RMSNorm weights,
biases) stays on AdamW.

Used for the Parallax arm specifically: its paper (arXiv:2605.29157) reports
the covariance-correction gate collapses under AdamW (stable rank of W_R:
134 under Muon, ~9 under AdamW) -- see parallax/attention.py's docstring.
"""
import torch


@torch.no_grad()
def _orthogonalise(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz quintic, run in bf16. Converges to a matrix with the same
    singular vectors as G and singular values ~1; operates on the short side."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Single-device Muon. Params must be 2-D; route everything else to AdamW."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov, ns_steps=ns_steps,
                                            weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.lerp_(p.grad, 1.0 - g["momentum"])
                d = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                d = _orthogonalise(d, g["ns_steps"])
                d = d * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1.0 - g["lr"] * g["weight_decay"])
                p.add_(d, alpha=-g["lr"])
        return loss


def split_params(model):
    """(muon_params, adamw_params).  Matmul-shaped (2-D), not embedding/head ->
    Muon; everything else (embeddings, RMSNorm weights) -> AdamW."""
    muon, adamw = [], []
    embed_ids = {id(model.wte.weight)}
    if model.wpe is not None:
        embed_ids.add(id(model.wpe.weight))
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon if (p.dim() == 2 and id(p) not in embed_ids) else adamw).append(p)
    return muon, adamw
