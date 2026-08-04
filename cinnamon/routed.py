"""One routed expert-block position (B2 or B3).

ONE expert body, a bank of hypernetworks, and a router that picks between them.

The loop, per token:

    r = 1, 2          DynFFN runs on its plain base weights -- no hypernetwork
                      selected yet.  Two guaranteed passes.
    KL still falling? no  -> stop here
                      yes -> the router picks a hypernetwork
    r = 3, 4          DynFFN runs with that hypernetwork's DoRA applied.
                      Two guaranteed passes again.
    KL still falling? no  -> stop
                      yes -> the router picks again, applies the new
                             transformation, and another two passes run
    ...                until KL stops falling or the budget c_max2 runs out.

"Still falling" is measured against the best KL seen so far, not the previous
commitment: two guaranteed passes give two KLs per commitment, and a commitment
counts as progress if the *lower* of them beats that running best.  Using the min
is also what makes the rule robust to the switch itself -- the first KL after a
new hypernetwork is applied straddles two different weight sets and comes out
inflated, and a min ignores it without needing a special case.

Each recurrence is KDA -> FixedFFN -> MLA -> DynFFN -> Norm; only the DynFFN's
weights change with the selection.  Halting is only ever tested at the end of a
commitment, so a commitment is never interrupted mid-way.

Everything is per-token (7): depth, selection and halting are decided
independently for every position in the batch.

Why this shape.  The body is quadratic in d_model, a hypernetwork only linear, so
n specialisations cost far less than n copies of the block -- and, decisively,
stage_attn runs ONCE per recurrence instead of once per expert.  Under the old
multi-expert design each expert needed its own keys and values over the whole
history (4.3), which is 4.3's requirement and not something batching removes; the
only way to stop paying it n times is to have one body.

turns=2 is load-bearing, not a tuning knob: the first recurrence after a switch
produces a KL that straddles two different sets of weights and is discarded, so a
commitment needs a second pass to yield one usable KL.  At turns=1 no clean KL
ever exists and halting cannot work.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks import ExpertBlock
from .config import Config
from .halting import rel_error
from .router import Router


def _hyper_apply(hyper, r, ctx, base_norms, expert, h_attn):
    """One hypernetwork's whole per-token pipeline: generate -> apply.

    Still one checkpointed unit (see its call site).  The composed matrices this
    used to build are gone -- dora_linear applies the factors directly -- so the
    widest live tensor is now [.., d_ff, rank] rather than [.., d_ff, d_model].
    """
    vec = hyper.forward_token(r, ctx, base_norms)
    A_u, B_u, m_u, A_d, B_d, m_d = hyper.unpack(vec)
    return expert.stage_dyn(h_attn, r, ((A_u, B_u, m_u), (A_d, B_d, m_d)))


class RoutedBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.expert = ExpertBlock(c)
        self.router = Router(c)
        # No separate content KDA any more.  It cost 0.81 M parameters and a full
        # KDA forward every recurrence to produce a signal that ctx_proj then
        # squashed to phi_dim anyway -- while h, the state entering the recurrence,
        # is already per-token AND already context-mixed (B1's KDA and MLA, plus
        # every prior recurrence).  Ouroboros's controller gets by on a mean-pooled
        # summary; raw h is strictly more information than that and free.

    def forward(self, h, aux=None):
        c = self.c
        B, S, D = h.shape
        e, n = self.expert, c.n_hypernets
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if (
            c.grad_checkpoint and self.training) else (lambda f, *a: f(*a))

        # INPUT INJECTION.  h0 is what entered the block, re-added at every
        # recurrence so each position permanently keeps its own distinct signal no
        # matter how deep it goes -- the state can drift arbitrarily far but can
        # never lose what made this position different from its neighbours.
        # Standard for recurrent-depth LMs (Geiping et al. 2025 inject the
        # embedding at every step for exactly this reason).
        h0 = h

        active = torch.ones(B, S, dtype=torch.bool, device=h.device)
        depth = torch.zeros(B, S, dtype=torch.int32, device=h.device)
        # -1 until the router has spoken; r_free recurrences run unadapted.
        sel = torch.full((B, S), -1, dtype=torch.long, device=h.device)
        gate = torch.ones(B, S, 1, device=h.device)
        prev_ctx = None
        best = cur = None       # best error ever seen; min error within this commitment

        for r in range(1, c.c_max2 + 1):
            h_attn = run(e.stage_attn, h, r)       # ONCE -- the whole point
            # Content signal for THIS recurrence: the state ENTERING it, RMS
            # normalised.  Used below both for weight generation (r > r_free, WITH
            # gradient) and, detached, for the halting comparison -- so halting
            # measures convergence of the hidden state itself.
            #
            # The normalisation is load-bearing, not cosmetic.  Under pre-norm plus
            # input injection ||h|| grows every recurrence, so a raw relative error
            # ||h - prev|| / ||h|| shrinks monotonically no matter what the state is
            # doing -- "still improving" reads true forever and halting never fires
            # (measured: mean depth 121 against a 64 cap).  Comparing directions
            # instead makes the test scale free, and it keeps the hypernetwork's
            # content input bounded as the stream grows.
            # .to(h.dtype): the reduction promotes to fp32 under autocast, and a
            # fp32 ctx propagates through generation into a bf16 output buffer.
            ctx = (h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True)
                                   + c.eps)).to(h.dtype)

            if r <= c.r_free:
                # No hypernetwork selected yet: the dynamic FFN runs on its own
                # base weights, which is exactly DoRA's identity case.
                out = run(e.stage_dyn, h_attn, r)
            else:
                out = h_attn
                base_norms = e.base_norms()
                # GATHERED, not the unconditional full-batch-per-hypernet form
                # used elsewhere in this file (aux/router).  Running every
                # hypernet on every token would do n times the generation work
                # even though each token needs ONE hypernet -- measured at ~6
                # min/step back when it also materialised the weight matrices.
                # Gathering keeps total work across all n calls at O(B*S) once,
                # not O(n*B*S).
                #
                # Data-dependent (`if idx.numel()`) -- fine for the single-GPU
                # runs this is built for.  Under DDP two ranks could skip
                # different hypernets and desync the allreduce, same reasoning
                # as the unconditional form elsewhere; that version would need
                # to be restored (or gathering done DDP-safely) before any
                # multi-GPU run.
                flat_ctx = ctx.reshape(B * S, D)
                flat_attn = h_attn.reshape(B * S, D)
                flat_out = out.reshape(B * S, D)
                for i in range(n):
                    mask = (active & (sel == i)).reshape(-1)
                    idx = mask.nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    ctx_g = flat_ctx[idx].unsqueeze(0)         # [1, k, D]
                    attn_g = flat_attn[idx].unsqueeze(0)       # [1, k, D]
                    out_g = run(_hyper_apply, e.hypers[i], r, ctx_g, base_norms,
                                e, attn_g)
                    # .to(): dora_linear's closed-form norm mixes fp32 reductions
                    # with a bf16 value path, so the result can come back wider
                    # than the buffer it is written into.
                    flat_out = flat_out.index_copy(
                        0, idx, out_g.squeeze(0).to(flat_out.dtype))
                out = flat_out.reshape(B, S, D)

            h = torch.where(active.unsqueeze(-1), out + h0, h)
            depth = depth + active

            # A commitment boundary: the end of the warm-up, then every `turns`
            # recurrences after.  Halting is checked here and routing decided
            # immediately after, in that order.
            boundary = r >= c.r_free and (r - c.r_free) % c.turns == 0

            with torch.no_grad():            # 15: halting needs no gradient
                ctx_d = ctx.detach()
                K = rel_error(ctx_d, prev_ctx, c.eps) if prev_ctx is not None else None
                prev_ctx = ctx_d

                if K is None:
                    # r=1 straddles block-input -> recurrence.  Different
                    # distributions, not comparable to a recurrence-to-recurrence
                    # reading.
                    pass
                elif r < c.r_free:
                    # Warm-up readings seed the reference; they are not a
                    # commitment.
                    best = K if best is None else torch.minimum(best, K)
                else:
                    # `turns` guaranteed passes give `turns` readings per
                    # commitment.  Take the MIN, which is also what makes the
                    # rule robust to the adapter switch: the first reading right
                    # after a new hypernetwork is applied straddles two different
                    # weight sets and comes out inflated, and a min ignores it.
                    cur = K if cur is None else torch.minimum(cur, K)
                    if boundary:
                        if best is not None and not c.no_halt:
                            # Two ways to be done, and BOTH are needed.
                            #   cur < best   -- still converging.  Cannot fire on a
                            #                   smooth contraction, which falls
                            #                   monotonically forever.
                            #   cur > tol    -- still moving enough to be worth
                            #                   another commitment.  This is what
                            #                   actually stops a converged token.
                            # AND also keeps halting monotone: a frozen token has
                            # K = 0, which would otherwise read as "converging".
                            active = active & (cur < best) & (cur > c.halt_tol)
                        best = cur if best is None else torch.minimum(best, cur)
                        cur = None

            # Stopping once every token has halted is where the real speedup comes
            # from, but a per-rank active.any() is a batch-dependent trip count, and
            # two DDP ranks running a different number of recurrences deadlock on
            # the next collective.  all_reduce makes the decision unanimous.
            stop = (~active.any()).to(torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stop, op=dist.ReduceOp.MIN)
            if r == c.c_max2 or bool(stop):
                break

            if boundary:
                # The routing-boundary norm, now applied to the router's INPUT
                # rather than to the residual stream (stage_dyn returns h raw).
                # Router is a bare linear with no norm of its own, and the stream
                # grows under pre-norm, so without this its logits grow with depth
                # and the softmax saturates into a fixed choice.
                probs = self.router(e.out_norm(h))
                nxt = probs.argmax(-1)
                g = probs.gather(-1, nxt.unsqueeze(-1))
                if aux is not None:
                    aux.add(probs, nxt, active)
                # The gate multiply is what gives the router gradient at all, since
                # argmax has none.  It used to have to come after the boundary norm
                # or RMSNorm would cancel it (6); now that norm is only on the
                # router's input and never on the stream, so nothing downstream can
                # cancel it.  Halted tokens keep their old state.
                a3 = active.unsqueeze(-1)
                sel = torch.where(active, nxt, sel)
                gate = torch.where(a3, g.to(gate.dtype), gate)
                if c.router_gate != "off":
                    gm = g.to(h.dtype)
                    if c.router_gate == "straight":
                        # Identity forward, same d/dg: the router still learns, but
                        # the residual stream is not scaled by ~1/n every boundary.
                        gm = gm / gm.detach()
                    h = torch.where(a3, gm * h, h)

        return h, depth
