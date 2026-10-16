"""Muon: momentum orthogonalised by Newton-Schulz (Jordan et al., https://kellerjordan.github.io/posts/muon/).

The update is the momentum buffer with its singular values flattened to 1 --
whatever direction the gradient points, every direction is stepped equally.
That is only meaningful for a matrix that acts as a linear map, so Muon
covers matmul weights ONLY. Everything else (embeddings, RMSNorm weights,
biases) stays on AdamW.
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
            params = [p for p in g["params"] if p.grad is not None]
            for p in params:
                if "buf" not in self.state[p]:
                    self.state[p]["buf"] = torch.zeros_like(p.grad)
            grads = [p.grad for p in params]
            bufs = [self.state[p]["buf"] for p in params]
            torch._foreach_lerp_(bufs, grads, 1.0 - g["momentum"])
            directions = [grad.lerp(buf, g["momentum"]) for grad, buf in zip(grads, bufs)] if g["nesterov"] else bufs
            # One batched Newton-Schulz per weight shape instead of one per matrix: optimizer step 11.4 -> 2.6 ms (softmax arm).
            by_shape = {}
            for p, direction in zip(params, directions):
                by_shape.setdefault(p.shape, []).append((p, direction))
            for shape, pairs in by_shape.items():
                update = _orthogonalise(torch.stack([direction for _, direction in pairs]), g["ns_steps"])
                update *= max(1.0, shape[-2] / shape[-1]) ** 0.5
                shape_params = [p for p, _ in pairs]
                if g["weight_decay"]:
                    torch._foreach_mul_(shape_params, 1.0 - g["lr"] * g["weight_decay"])
                torch._foreach_add_(shape_params, list(update.unbind(0)), alpha=-g["lr"])
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
