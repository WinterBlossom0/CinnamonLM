"""3 Shared blocks (B1, B_mid, B4) and 4 Expert blocks (B2, B3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FFN, MLA
from .attnres import AttnRes
from .config import Config
from .hypernet import Hypernet, dora_linear
from .kda import KDA


class SharedBlock(nn.Module):
    """KDA-FFN-KDA-FFN-KDA-FFN-MLA-FFN.

    Kimi Linear's 3:1 KDA:MLA ratio -- three linear-attention layers for cheap
    compressed sequence processing, then one MLA to restore global interaction.
    B1, B_mid and B4 share this structure but are separate parameter sets, and
    their FFNs are the only FFNs in the model shared across experts.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.subs = nn.ModuleList(
            [KDA(c), FFN(c), KDA(c), FFN(c), KDA(c), FFN(c), MLA(c), FFN(c)])
        self.pre = nn.ModuleList([nn.RMSNorm(c.d_model, eps=c.eps) for _ in self.subs])
        # Sublayer 0 sees only the block input, where AttnRes is provably the
        # identity (softmax over a single source), so it gets no module.
        # 8 modules = sublayers 1..7 plus the final output aggregation.
        self.ares = nn.ModuleList([AttnRes(c) for _ in range(len(self.subs))])

    def forward(self, h):
        src = [h]
        for i, f in enumerate(self.subs):
            src.append(f(self.pre[i](h if i == 0 else self.ares[i - 1](src))))
        return self.ares[-1](src)


class ExpertBlock(nn.Module):
    """Body at one block position.  B2 and B3 are distinct parameter sets (4.1),
    hence the recurrence counter resets between them.

    14.2: plain pre-norm residuals inside the recurrence, not AttnRes -- AttnRes
    across recurrences would retain every prior output, a memory blow-up at c_max2
    for a speculative gain.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        # 2:1 KDA:MLA.  MLA is NoPE and carries no positional info, so KDA is the
        # only positional carrier.
        self.kda = nn.ModuleList([KDA(c) for _ in range(2)])
        self.fixed = nn.ModuleList([FFN(c) for _ in range(2)])
        self.mla = MLA(c)
        # Dynamic FFN is SwiGLU; 5.3 specifies DoRA for exactly the up and down
        # matrices, so the gate projection stays static.
        self.dyn_gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.dyn_up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.dyn_down = nn.Linear(c.d_ff, c.d_model, bias=False)
        # Bank of hypernetworks over ONE body: specialisation is in which adapter
        # generator the router picks, not in a block copy.  Body quadratic in
        # d_model, hypernetwork linear => n specialisations for a fraction of n
        # copies, and stage_attn runs once per recurrence instead of n times.
        self.hypers = nn.ModuleList([Hypernet(c) for _ in range(c.n_hypernets)])
        # Per RECURRENCE STEP, not just per sublayer: 6 = 2x(KDA, FFN) + MLA +
        # dynFFN's own norm at index 5, one set per r.  A shared set forces
        # identical normalisation at depth 1 and depth 32 -- this is the only place
        # the step index acts on the residual stream directly rather than through
        # generated FFN weights.  All init weight=1, so behaviour is unchanged
        # until training separates them.  Cost ~57 k per block position.
        self.pre = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(c.d_model, eps=c.eps) for _ in range(6)])
            for _ in range(c.r_ceiling)])
        # Single, not per-step: applied only to the router's input at commitment
        # boundaries (see routed.py), so per-step copies would leave non-boundary
        # steps with no gradient.
        self.out_norm = nn.RMSNorm(c.d_model, eps=c.eps)

    def base_norms(self):
        """5.1 Column norms of the dynamic base.  Constant within a forward pass
        (carry no r information); their job is across training steps, letting the
        generated DoRA track the base as the optimiser moves it."""
        n = torch.cat([self.dyn_up.weight.norm(dim=1), self.dyn_down.weight.norm(dim=1)])
        return n.detach() if self.c.detach_norm_input else n

    def step_norms(self, r):
        """RMSNorm set for recurrence r.  Clamped to r_ceiling like phi(r), so
        c_max2 > r_ceiling reuses the last set rather than indexing off the end."""
        return self.pre[min(r, self.c.r_ceiling) - 1]

    def stage_attn(self, h, r):
        """4.2 up to the dynamic FFN.  Sublayer weights do not depend on r, the
        norms do.  Runs once per recurrence over the whole sequence; only the
        dynamic FFN is grouped per token, which is what makes per-token depth
        affordable."""
        pre = self.step_norms(r)
        for i in range(2):
            h = h + self.kda[i](pre[2 * i](h))
            h = h + self.fixed[i](pre[2 * i + 1](h))
        return h + self.mla(pre[4](h))

    def stage_dyn(self, h, r, factors=None):
        """r-dependent tail: dynamic FFN.  Pointwise, so it takes any leading shape
        and can be applied to a gathered subset.

        factors=None for r <= r_free (plain base weights = DoRA's identity case);
        otherwise per-token generated factors, applied by dora_linear without
        building the [.., d_ff, d_model] matrix.
        """
        x = self.step_norms(r)[5](h)
        gate = F.silu(self.dyn_gate(x))
        if factors is None:
            up = F.linear(x, self.dyn_up.weight)
            down = F.linear(gate * up, self.dyn_down.weight)
        else:
            (A_u, B_u, m_u), (A_d, B_d, m_d) = factors
            up = dora_linear(x, self.dyn_up.weight, A_u, B_u, m_u, self.c)
            down = dora_linear(gate * up, self.dyn_down.weight, A_d, B_d, m_d, self.c)
        # PRE-NORM: stream returned raw.  Normalising h+down here was the block's
        # one post-norm, applied up to 32x per forward.  Post-norm pins ||h||
        # constant so the identity path can never dominate and every recurrence
        # injects a full-magnitude update forever -- measured gnorm 1.5e5 vs
        # clip 1.0, model could not learn.  Letting ||h|| grow damps later updates.
        # out_norm is applied by routed.py to the ROUTER'S INPUT instead.
        return h + down
