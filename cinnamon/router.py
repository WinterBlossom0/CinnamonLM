"""Hypernetwork router and 10 the load-balancing auxiliary loss.

One router per block position.  It reads the state at the end of a recurrence and
picks which hypernetwork generates the DoRA for the next routing commitment.

There is no entry router: the first `r_free` recurrences run on the plain base
weights, so nothing needs selecting until the state has been shaped a little.

fp32 throughout, even under mixed precision -- routers are the standard
precision-sensitivity casualty in MoE training (11.0).
"""
import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = c.n_hypernets
        # Single linear layer, no bias (11.0).  A deeper router would just give the
        # collapse dynamic in 10 more capacity to latch onto.
        self.w = nn.Parameter(torch.empty(c.d_model, self.n))
        # Small random, not zero: leaves the distribution near-uniform (good for
        # load balance) while still breaking symmetry between hypernets (11.0).
        nn.init.normal_(self.w, std=0.01)

    def forward(self, h):
        """h: [B, S, d] -> [B, S, n_hypernets] probabilities.

        autocast must be switched off, not just worked around with .float():
        einsum is on autocast's lower-precision list, so it re-casts float inputs
        straight back down to bf16/fp16 and the cast silently achieves nothing.
        """
        with torch.autocast(h.device.type, enabled=False):
            return torch.softmax(
                torch.einsum('bsd,dn->bsn', h.float(), self.w.float()), -1)


class AuxLoss:
    """10  L_aux = N * sum_i f_i * P_i, minimised at 1.0 when both are uniform.

    f_i counts argmax outcomes and is discrete, so it carries no gradient; P_i is
    continuous and does.  The product uses f_i as a coefficient measuring actual
    imbalance while gradient flows through P_i into the router weights.  An
    over-selected hypernet has large f_i, so f_i*P_i is large, so P_i is pushed
    down hard, while rarely-selected ones get almost no downward pressure.

    Collapse is invisible in the loss curve, so this also carries the per-hypernet
    selection counts and router entropy that 10 requires to be logged.
    """

    def __init__(self, n, device):
        self.n = n
        self.p_sum = torch.zeros(n, device=device)
        self.f_sum = torch.zeros(n, device=device)
        # Kept as a device tensor, not a Python int: it has to count only the
        # positions that contributed to p_sum, and reading a masked count on the
        # host would be a GPU sync on the hot path.
        self.w_sum = torch.zeros((), device=device)
        self.calls = 0
        self.ent = 0.0

    def add(self, probs, chosen, mask=None):
        """probs: [..., n] router output.  chosen: [...] argmax selection.

        A masked-out position contributes zero rather than being indexed away:
        boolean indexing needs the mask's popcount on the host, which is a GPU
        sync, and skipping on an empty mask is a data-dependent branch that two
        DDP ranks can disagree about.
        """
        flat = probs.reshape(-1, self.n)
        w = (mask.reshape(-1).float() if mask is not None
             else torch.ones(flat.shape[0], device=flat.device))
        self.p_sum = self.p_sum + (flat * w.unsqueeze(-1)).sum(0)
        self.f_sum = self.f_sum + torch.bincount(
            chosen.reshape(-1), weights=w, minlength=self.n).float()
        # w.sum(), NOT flat.shape[0].  p_sum only accumulates masked-in positions,
        # so dividing by the unmasked count scaled P down by the active fraction --
        # the balancing pressure quietly faded exactly as tokens halted, which is
        # when it is needed most.
        self.w_sum = self.w_sum + w.sum()
        self.calls += 1
        self.ent += float(-(flat * (flat + 1e-9).log()).sum(-1).mean().detach())

    def value(self):
        if not self.calls:
            return torch.zeros((), device=self.p_sum.device), {}
        f = self.f_sum / self.f_sum.sum().clamp(min=1)
        P = self.p_sum / self.w_sum.clamp(min=1)
        loss = self.n * (f * P).sum()
        stats = {"aux": float(loss.detach()), "entropy": self.ent / max(1, self.calls),
                 "frac": [round(float(x), 3) for x in f]}
        return loss, stats
