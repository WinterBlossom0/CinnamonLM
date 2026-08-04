"""One routed expert-block position (B2 or B3).

One expert body + a bank of hypernetworks + a router selecting between them.
Recurrence: KDA -> FixedFFN -> MLA -> DynFFN; only DynFFN's weights vary with the
selection.  depth/selection/halting are per token (spec 7).

    r <= r_free       DynFFN on plain base weights, no hypernetwork selected
    then per commitment (`turns` recurrences under one hypernetwork):
        converged?    -> halt
        else          -> router picks the next hypernetwork
    until halt or c_max2

Halting is tested only at commitment boundaries, never mid-commitment, and takes
the MIN reading over the commitment: the first reading after a switch straddles
two weight sets and is inflated, so a min discards it without a special case.

turns >= 2 is load-bearing: at turns=1 no clean reading ever exists (every
reading straddles a switch) and halting cannot work.

Shape rationale: body is quadratic in d_model, hypernetwork linear, so n
specialisations << n block copies.  Decisively, stage_attn runs once per
recurrence rather than once per expert -- spec 4.3 requires each expert to
produce its own K/V over the whole history, which batching cannot remove; one
body does.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks import ExpertBlock
from .config import Config
from .halting import rel_error
from .router import Router


def _hyper_apply(hyper, r, ctx, base_norms, expert, h_attn):
    """Generate -> apply, for one hypernetwork.  One checkpointed unit (see call
    site).  dora_linear applies factors directly, so the widest live tensor is
    [.., d_ff, rank], not [.., d_ff, d_model]."""
    vec = hyper.forward_token(r, ctx, base_norms)
    A_u, B_u, m_u, A_d, B_d, m_d = hyper.unpack(vec)
    return expert.stage_dyn(h_attn, r, ((A_u, B_u, m_u), (A_d, B_d, m_d)))


class RoutedBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.expert = ExpertBlock(c)
        self.router = Router(c)
        # No separate content KDA: it cost 0.81 M params + a KDA forward per
        # recurrence for a signal ctx_proj then squashed to phi_dim anyway.  h is
        # already per-token and context-mixed (B1 + prior recurrences).

    def forward(self, h, aux=None):
        c = self.c
        B, S, D = h.shape
        e, n = self.expert, c.n_hypernets
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if (
            c.grad_checkpoint and self.training) else (lambda f, *a: f(*a))

        # Input injection: h0 re-added every recurrence so each position retains
        # its distinct signal at any depth.  Standard for recurrent-depth LMs
        # (Geiping et al. 2025).
        h0 = h

        active = torch.ones(B, S, dtype=torch.bool, device=h.device)
        depth = torch.zeros(B, S, dtype=torch.int32, device=h.device)
        # -1 until the router has spoken; r_free recurrences run unadapted.
        sel = torch.full((B, S), -1, dtype=torch.long, device=h.device)
        gate = torch.ones(B, S, 1, device=h.device)
        prev_ctx = None
        best = cur = None       # best error ever seen; min error within this commitment

        for r in range(1, c.c_max2 + 1):
            h_attn = run(e.stage_attn, h, r)       # ONCE per recurrence, not per hypernet
            # Content signal = state entering this recurrence, RMS normalised.
            # Feeds weight generation (r > r_free, with gradient) and, detached,
            # the halting comparison.
            #
            # Normalisation is load-bearing: under pre-norm + input injection ||h||
            # grows every recurrence, so raw ||h-prev||/||h|| shrinks monotonically
            # regardless of state, "still improving" reads true forever, halting
            # never fires (measured: mean depth 121 vs a 64 cap).  Comparing
            # directions is scale-free and bounds the hypernetwork's input.
            # .to(h.dtype): the reduction promotes to fp32 under autocast; a fp32
            # ctx propagates into a bf16 output buffer downstream.
            ctx = (h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True)
                                   + c.eps)).to(h.dtype)

            if r <= c.r_free:
                # No hypernetwork selected yet: base weights = DoRA's identity case.
                out = run(e.stage_dyn, h_attn, r)
            else:
                out = h_attn
                base_norms = e.base_norms()
                # Gathered dispatch: each hypernet runs only on tokens it owns, so
                # total work is O(B*S), not O(n*B*S) (~6 min/step when measured
                # unconditional).  `if idx.numel()` is a data-dependent branch and
                # is why this model is single-GPU only -- it would desync a DDP
                # all-reduce.  First thing to change if that ever matters.
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
                    # .to(): dora_linear mixes fp32 reductions with a bf16 value
                    # path, so the result can be wider than the target buffer.
                    flat_out = flat_out.index_copy(
                        0, idx, out_g.squeeze(0).to(flat_out.dtype))
                out = flat_out.reshape(B, S, D)

            h = torch.where(active.unsqueeze(-1), out + h0, h)
            depth = depth + active

            # Commitment boundary: end of warm-up, then every `turns` after.
            # Halting checked here, routing decided immediately after, in that order.
            boundary = r >= c.r_free and (r - c.r_free) % c.turns == 0

            with torch.no_grad():            # spec 15: halting needs no gradient
                ctx_d = ctx.detach()
                K = rel_error(ctx_d, prev_ctx, c.eps) if prev_ctx is not None else None
                prev_ctx = ctx_d

                if K is None:
                    pass                     # r=1 straddles block-input -> recurrence
                elif r < c.r_free:
                    best = K if best is None else torch.minimum(best, K)   # seed only
                else:
                    # MIN over the commitment's `turns` readings: the first reading
                    # after a switch straddles two weight sets and is inflated.
                    cur = K if cur is None else torch.minimum(cur, K)
                    if boundary:
                        if best is not None and not c.no_halt:
                            # Both conditions needed:
                            #   cur < best  still converging; cannot fire on a
                            #               smooth contraction (falls forever)
                            #   cur > tol   still moving enough to be worth another
                            #               commitment; this is what stops a
                            #               converged token
                            # AND keeps halting monotone: a frozen token has K=0,
                            # which would otherwise read as "converging".
                            active = active & (cur < best) & (cur > c.halt_tol)
                        best = cur if best is None else torch.minimum(best, cur)
                        cur = None

            # Stopping once every token has halted is where the real speedup comes
            # from.  Only tested at a boundary: `active` is written in exactly one
            # place (just above, under `if boundary`), so between boundaries there
            # is provably nothing to see -- and reading it costs a GPU->CPU sync,
            # which was being paid every single recurrence for an unchanged value.
            if r == c.c_max2 or (boundary and not active.any()):
                break

            if boundary:
                # out_norm applies to the ROUTER'S INPUT, not the residual stream
                # (stage_dyn returns h raw).  Router is a bare Linear with no norm
                # of its own and the stream grows under pre-norm, so without this
                # its logits grow with depth and the softmax saturates.
                probs = self.router(e.out_norm(h))
                nxt = probs.argmax(-1)
                g = probs.gather(-1, nxt.unsqueeze(-1))
                if aux is not None:
                    aux.add(probs, nxt, active)
                # Gate multiply is the router's only gradient path (argmax has
                # none).  Nothing downstream normalises the stream, so it cannot be
                # cancelled.  Halted tokens keep their old state.
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
