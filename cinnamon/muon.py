"""Muon: momentum orthogonalised by Newton-Schulz (Jordan et al.).

The update is the momentum buffer with its singular values flattened to 1 --
whatever direction the gradient points, every direction is stepped equally.  That
is only meaningful for a matrix that acts as a linear map, so Muon covers matmul
weights ONLY.  Everything else stays on AdamW:

    Muon     KDA/FFN/DoRA-base/hypernet/basis/router weights   [out, in]
    AdamW    embedding, LM head, RMSNorm weights, biases,
             and the lookup tables phi / err_slot

phi and err_slot are 2-D but are TABLES, not maps: row r is recurrence r's code,
and orthogonalising across rows would mix unrelated steps together.  Shape alone
cannot tell them apart from a weight matrix, which is why the split is by name.

Defaults follow the reference implementation: momentum 0.95, Nesterov on, five
Newton-Schulz steps.  lr and weight decay are the knobs worth tuning.
"""
import torch


@torch.no_grad()
def _orthogonalise(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz quintic, run in bf16.

    Converges to a matrix with the same singular VECTORS as G and all singular
    values ~1.  The quintic coefficients are deliberately tuned to be fast rather
    than exact -- the iteration does not converge to machine precision and does
    not need to, since only the direction is used.

    Operates on the short side (transposing if needed) so the [n, n] products are
    as small as possible.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if X.size(-2) > X.size(-1):
        X = X.mT
    # Spectral norm <= Frobenius norm, so this guarantees the iteration starts
    # inside its region of convergence.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Single-device Muon.  Params must be 2-D; route everything else to AdamW."""

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
                # Nesterov: step from where the momentum is about to be, not from
                # where it is.
                d = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                d = _orthogonalise(d, g["ns_steps"])
                # An orthogonalised update has RMS ~1/sqrt(max(out, in)) regardless
                # of the gradient's scale, so without this a wide matrix and a tall
                # one move at different rates under one lr.
                d = d * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1.0 - g["lr"] * g["weight_decay"])
                p.add_(d, alpha=-g["lr"])
        return loss


# Names that are 2-D but are lookup tables rather than linear maps.
_TABLES = ("phi", "err_slot")


def split_params(model):
    """(muon_params, adamw_params).  Split by role, not by shape."""
    muon, adamw = [], []
    embed_ids = {id(model.embed.weight), id(model.head.weight)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        table = any(n.endswith(t) for t in _TABLES)
        (muon if (p.dim() == 2 and not table and id(p) not in embed_ids)
         else adamw).append((n, p))
    return muon, adamw
