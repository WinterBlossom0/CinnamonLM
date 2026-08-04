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
    """The body at one block position.  B2's and B3's bodies are distinct
    parameter sets (4.1), which is why the recurrence counter resets between them.

    14.2 decided: plain pre-norm residuals inside the recurrence, not AttnRes.
    AttnRes across recurrences would have to retain every prior recurrence's
    output, and with a cap up to c_max2 that is a memory blow-up for a
    speculative gain.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        # 2:1 KDA:MLA in the body.  A third KDA existed here; it now lives on
        # RoutedBlock instead (as ctx_kda), giving the hypernetwork's bucket
        # selection a real content signal rather than removing it.  MLA is NoPE
        # and carries no positional information, so KDA is still what supplies
        # position -- 2:1 keeps that while freeing one KDA's worth of parameters
        # for the content-extraction role.
        self.kda = nn.ModuleList([KDA(c) for _ in range(2)])
        self.fixed = nn.ModuleList([FFN(c) for _ in range(2)])
        self.mla = MLA(c)
        # Dynamic FFN is SwiGLU like the fixed one; 5.3 lists DoRA for exactly one
        # 512->2048 and one 2048->512 matrix, so the gate projection stays static.
        self.dyn_gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.dyn_up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.dyn_down = nn.Linear(c.d_ff, c.d_model, bias=False)
        # A bank of hypernetworks over ONE body.  Specialisation lives in which
        # adapter generator the router picks, not in a whole copy of the block:
        # the body is quadratic in d_model, a hypernetwork only linear, so this
        # buys n specialisations for a fraction of n copies' parameters -- and it
        # is what lets stage_attn run once per recurrence instead of n times.
        self.hypers = nn.ModuleList([Hypernet(c) for _ in range(c.n_hypernets)])
        # PER RECURRENCE STEP, not just per sublayer.  6 = 2x(KDA, FFN) + MLA +
        # the dynamic FFN's own norm at index 5, and there is now one such set for
        # every step r.  Sharing one set across every recurrence forced the body to
        # normalise identically at depth 1 and depth 32, which is the one place the
        # step index could act on the residual stream directly rather than only
        # through the generated FFN weights.  All start at weight=1, so step 0
        # behaviour is unchanged and the sets only diverge if training separates
        # them.  Cost is r_ceiling*7*d_model per block position -- ~57 k, negligible.
        self.pre = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(c.d_model, eps=c.eps) for _ in range(6)])
            for _ in range(c.r_ceiling)])
        # Single, NOT per step: this is no longer applied to the residual stream
        # every recurrence, only to the router's input at commitment boundaries.
        # Per-step copies would leave the non-boundary steps' parameters with no
        # gradient at all -- dead weights, and a DDP hazard.
        self.out_norm = nn.RMSNorm(c.d_model, eps=c.eps)

    def base_norms(self):
        """5.1  Column norms of the dynamic base.  Constant within a forward pass,
        so they carry no information about r; their job is across training steps,
        letting the generated DoRA track the base as the optimiser moves it."""
        n = torch.cat([self.dyn_up.weight.norm(dim=1), self.dyn_down.weight.norm(dim=1)])
        return n.detach() if self.c.detach_norm_input else n

    def step_norms(self, r):
        """The RMSNorm set for recurrence r.  Clamped to r_ceiling the same way
        the hypernetwork clamps phi(r), so a c_max2 above r_ceiling reuses the last
        set instead of indexing off the end."""
        return self.pre[min(r, self.c.r_ceiling) - 1]

    def stage_attn(self, h, r):
        """4.2 up to the dynamic FFN -- the sublayer weights do NOT depend on r,
        but the norms now do, so r is passed in.

        Run once per recurrence over the whole sequence; only the dynamic FFN's
        DoRA is grouped per token, which is what keeps per-token depth affordable.
        """
        pre = self.step_norms(r)
        for i in range(2):
            h = h + self.kda[i](pre[2 * i](h))
            h = h + self.fixed[i](pre[2 * i + 1](h))
        return h + self.mla(pre[4](h))

    def stage_dyn(self, h, r, factors=None):
        """The r-dependent tail: dynamic FFN + routing-boundary norm.  Pointwise,
        so it accepts any leading shape and can be applied to a gathered subset.

        factors is None for r <= r_free, where the dynamic FFN runs on its plain
        base weights (DoRA's identity case).  Otherwise it carries the generated
        DoRA factors per token, and dora_linear applies them without ever building
        the [.., d_ff, d_model] matrix they describe.
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
        # PRE-NORM: the residual stream is returned raw.  Normalising h + down here
        # was the block's one post-norm, applied once per recurrence, up to 32
        # times -- and post-norm pins ||h|| constant, so the identity path can
        # never grow to dominate and every recurrence keeps injecting a
        # full-magnitude update forever.  Letting ||h|| grow makes each successive
        # update relatively smaller, which is the damping a deep recurrence needs.
        # out_norm still exists; routed.py applies it to the ROUTER'S INPUT at
        # commitment boundaries, which is the job it was actually there for (the
        # router is a bare linear with no norm of its own, so it would otherwise
        # see the growing stream and saturate).
        return h + down
