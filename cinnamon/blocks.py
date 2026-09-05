"""The four Cinnamon layers.

    L1                KDA + FFN                     once
    L2                KDA + FixedFFN      \\ one recurrent block, up to c_max2
    L3                KDA + DynFFN        /  passes
    L4                KDA + FFN                     once

Four Kimi Delta Attention layers, four FFNs, nothing else. KDA's recurrent
state and causal short convolution carry sequence order; there is no RoPE or
other positional carrier.

Only L3's FFN has generated weights -- L2's is fixed.  The two are one recurrent
unit rather than two independently recursed layers, so a pass always applies the
fixed FFN before the generated one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FFN, KDA, XIELU
from .config import Config
from .hypernet import AdapterBasis, Hypernet, dora_linear, unit_row_col_norm


class Layer(nn.Module):
    """A plain pre-norm transformer layer, run once.  L1 and L4."""

    def __init__(self, c: Config):
        super().__init__()
        self.attn = KDA(c)
        self.ffn = FFN(c)
        self.n1 = nn.RMSNorm(c.d_model, eps=c.eps)
        self.n2 = nn.RMSNorm(c.d_model, eps=c.eps)

    def forward(self, h, *, attention_mask=None, past_key_values=None, use_cache=False):
        h = h + self.attn(self.n1(h), attention_mask=attention_mask,
                          past_key_values=past_key_values, use_cache=use_cache)
        return h + self.ffn(self.n2(h))


class ExpertBlock(nn.Module):
    """L2 + L3, the recurrent pair.

    Split into stage_attn (everything up to the generated FFN) and stage_dyn (the
    generated FFN itself) so attention runs ONCE per recurrence rather than once
    per hypernetwork -- spec 4.3 requires each expert to produce its own K/V over
    the whole history, which batching cannot remove but one shared body can.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.mla2 = KDA(c)           # L2 Kimi Delta Attention
        self.fixed = FFN(c)         # L2 FFN, static
        self.mla3 = KDA(c)           # L3 Kimi Delta Attention
        # Eq 2 residual scaling (arXiv:2606.18206): single global scalar, not
        # per-sublayer.  beta1 is derived from this and RoutedBlock's alpha2
        # (Theorem 1), passed into stage_attn/stage_dyn per call.
        self.alpha1 = nn.Parameter(torch.tensor(float(c.fp_alpha1_init)))
        # L3 FFN: plain xIELU MLP (no gate matrix), DoRA on up and down.
        self.dyn_up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.dyn_down = nn.Linear(c.d_ff, c.d_model, bias=False)
        self.dyn_act = XIELU()
        self.hypers = nn.ModuleList([Hypernet(c) for _ in range(c.n_hypernets)])
        # Shared across the bank: per-hypernet it was 87.6% of each hypernet spent
        # on a rank-code_dim output space, and a starved hypernet froze all of it.
        self.basis = AdapterBasis(c)
        # Per RECURRENCE STEP: 4 = L2's two + L3's two.  A shared set would force
        # identical normalisation at depth 1 and depth 32; this is the only place
        # the step index touches the residual stream directly rather than through
        # generated FFN weights.  All init weight=1, so behaviour is unchanged
        # until training separates them.
        self.pre = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(c.d_model, eps=c.eps) for _ in range(4)])
            for _ in range(c.r_ceiling)])
        # Single, not per-step: applied only to the router's input at commitment
        # boundaries (see routed.py), so per-step copies would leave non-boundary
        # steps with no gradient.
        self.out_norm = nn.RMSNorm(c.d_model, eps=c.eps)

    def base_norms(self):
        """5.1 Column norms of the dynamic base.  Constant within a forward pass;
        their job is across training steps, letting the generated DoRA track the
        base as the optimiser moves it."""
        n = torch.cat([self.dyn_up.weight.norm(dim=1), self.dyn_down.weight.norm(dim=1)])
        return n.detach() if self.c.detach_norm_input else n

    def step_norms(self, r):
        """Norm set for recurrence r, clamped to r_ceiling like phi(r) so
        c_max2 > r_ceiling reuses the last set rather than indexing off the end."""
        return self.pre[min(r, self.c.r_ceiling) - 1]

    def stage_attn(self, h, r, beta1):
        """L2 in full, plus L3's attention -- everything before the generated FFN.
        Weights do not depend on r, the norms do."""
        pre = self.step_norms(r)
        a1 = self.alpha1
        h = a1 * h + beta1 * self.mla2(pre[0](h))
        h = a1 * h + beta1 * self.fixed(pre[1](h))
        return a1 * h + beta1 * self.mla3(pre[2](h))

    def dyn_base(self):
        """Per projection, (unit_row_col_norm(W), AdapterBasis.q_basis's T).

        Both are functions of weights alone, constant within a forward pass and
        ~500x re-derived if left inside dora_linear.  T rides along because it is
        derived from Wn and is hoistable for exactly the same reason."""
        wns = (unit_row_col_norm(self.dyn_up.weight),
               unit_row_col_norm(self.dyn_down.weight))
        return tuple(zip(wns, self.basis.q_basis(wns)))

    def stage_dyn(self, h, r, beta1, factors=None, wn=(None, None)):
        """L3's FFN.  Pointwise, so it takes any leading shape and can be applied
        to a gathered subset of tokens.

        factors=None for r <= r_free (plain base weights = DoRA's identity case);
        otherwise per-token generated factors, applied by dora_linear without
        building the [.., d_ff, d_model] matrix.  wn is dyn_base() hoisted out of
        the recurrence by the caller.
        """
        x = self.step_norms(r)[3](h)
        if factors is None:
            up = F.linear(x, self.dyn_up.weight)
            down = F.linear(self.dyn_act(up), self.dyn_down.weight)
        else:
            (A_u, B_u, m_u, q_u), (A_d, B_d, m_d, q_d) = factors
            (wn_u, _), (wn_d, _) = wn
            up = dora_linear(x, self.dyn_up.weight, A_u, B_u, m_u, self.c, wn_u, q_u)
            down = dora_linear(self.dyn_act(up), self.dyn_down.weight, A_d, B_d,
                               m_d, self.c, wn_d, q_d)
        # PRE-NORM: stream returned raw.  Normalising h+down here was the block's
        # one post-norm, applied up to c_max2 times per forward, which pins ||h||
        # constant so the identity path can never dominate -- measured gnorm 1.5e5
        # against clip 1.0, model could not learn.  out_norm is applied by
        # routed.py to the ROUTER'S input instead.
        return self.alpha1 * h + beta1 * down
