"""L2 + L3, the model's one recurrent block.

One body + a bank of hypernetworks + a router selecting between them.
Recurrence: KDA -> FixedFFN -> KDA -> DynFFN; only DynFFN's weights vary with the
selection.  Selection is per token (spec 7); DEPTH AND HALTING ARE PER BATCH --
see the halting note below.

    r <= r_free       DynFFN on plain base weights, no hypernetwork selected
    then per commitment (`turns` recurrences under one hypernetwork):
        router picks the next hypernetwork (unrelated to halting)
    checked every recurrence: converged (r < fp_tau)? -> halt
    until halt or c_max2

Halting is fixed-point convergence (arXiv:2606.18206 Algorithm 1, "FPOpt"),
BATCH-UNIFORM: the residual between the undamped candidate and the current state
is averaged over the batch to one scalar per recurrence, and everything the rule
touches from there -- r_star, the patience counter, the damping factor eta -- is
that one scalar, not a per-token quantity:

    z_cand = alpha2 * out + beta2 * h0                    Eq 3
    r      = fixed_point_residual(z_cand, h, eps).mean()  Algorithm 1 line 5
    h      = eta * z_cand + (1 - eta) * h                 Algorithm 1 line 6 (damped)
    stop when r < fp_tau

r_star/patience/eta update every recurrence, not just at commitment boundaries --
Algorithm 1 has no concept of a boundary, so this pays a GPU->CPU sync every
recurrence rather than every `turns` (the previous rule's boundary-only cost),
accepted for exact fidelity to the paper's per-step procedure. Router selection
keeps its own `turns`-boundary cadence, unrelated to halting and untouched by
this change.

Every token runs to the same depth -- the batch enters and leaves the recurrence
together, by construction: no per-token masking exists in the state update
(`h = eta*z_cand + (1-eta)*h`, no `torch.where` gate). Depth stays adaptive
across BATCHES (a converging batch stops early) but not within one.

The generator inputs (err_best, err_last, hypernet.py 5.4) are the batch-uniform
r_star/r_last, broadcast to every token position -- there is no longer a
per-token signal to feed distinctly, since the residual itself is a batch mean.

P1 (whether halting fires once trained, untrained-only evidence so far) was about
the previous relative-improvement rule specifically and does not carry over: an
absolute-tolerance rule has no "always still falling" degenerate case to fail
into. Re-open only if the new rule shows its own analogous failure mode.

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
from .halting import fixed_point_residual
from .hypernet import forward_token_bank, forward_token_grouped, stack_bank
from .router import Router


class _ClipRecurGrad(torch.autograd.Function):
    """Identity forward; clips the incoming gradient's norm on the way back.

    Inserted once per recurrence on the carried state h.  Measured (bisecting
    RoutedBlock.forward's steps= against a real batch on a checkpoint that
    later diverged): forward activations stayed flat across depth while the
    backward gradient grew ~exponentially with recurrence depth (40 at depth
    16 -> 53,230 at depth 32) -- a per-step backward Jacobian with spectral
    norm > 1 compounding geometrically across c_max2 steps, invisible to the
    forward pass and to a clip applied only once at the very end.  Clipping
    here, once per recurrence, caps each step's contribution before it
    multiplies through the next step's Jacobian.  Norm-only (clip_grad_norm_'s
    convention): rescales magnitude, does not touch direction.
    """

    @staticmethod
    def forward(ctx, x, max_norm):
        ctx.max_norm = max_norm
        return x

    @staticmethod
    def backward(ctx, grad_output):
        norm = grad_output.float().norm()
        scale = (ctx.max_norm / (norm + 1e-6)).clamp(max=1.0)
        return (grad_output.float() * scale).to(grad_output.dtype), None


def _hyper_apply(hyper, r, ctx, base_norms, expert, h_attn, eb, el, beta1, wn):
    """Generate -> apply, for one hypernetwork.  One checkpointed unit (see call
    site).  dora_linear applies factors directly, so the widest live tensor is
    [.., d_ff, rank], not [.., d_ff, d_model]."""
    code = hyper.forward_token(r, ctx, base_norms, eb, el)
    return expert.stage_dyn(h_attn, r, beta1, expert.basis.factors(code, wn), wn)


def _bank_apply(bank, r, ctx, base_norms, expert, h_attn, eb, el, r_ceiling, beta1, wn):
    """Generate -> apply, for the WHOLE bank at once.  One checkpointed unit, the
    bmm counterpart of _hyper_apply."""
    code = forward_token_bank(bank, r, ctx, base_norms, r_ceiling, eb, el)
    # Basis is shared, so it applies once to the whole [n, K, code_dim] result
    # rather than as an n-way bmm -- one large matmul instead of n batched ones.
    return expert.stage_dyn(h_attn, r, beta1, expert.basis.factors(code, wn), wn)


def _grouped_apply(bank, r, ctx, base_norms, expert, h_attn, eb, el, seg, offs,
                   r_ceiling, beta1, wn):
    """Generate -> apply, for the WHOLE bank on segment-sorted tokens.  One
    checkpointed unit, the unpadded counterpart of _bank_apply."""
    code = forward_token_grouped(bank, r, ctx, base_norms, r_ceiling, seg, offs,
                                 eb, el)
    return expert.stage_dyn(h_attn, r, beta1, expert.basis.factors(code, wn), wn)


class RoutedBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.expert = ExpertBlock(c)
        self.router = Router(c)
        # Eq 3 iteration-wise input mixing (arXiv:2606.18206): single global
        # scalar.  beta2 derived from this and expert.alpha1 (Theorem 1).
        self.alpha2 = nn.Parameter(torch.tensor(float(c.fp_alpha2_init)))
        # The hypernetwork's content signal is h itself: already per-token and
        # context-mixed by L1 and prior recurrences, so a dedicated summariser
        # measured no better and cost a forward per recurrence.

    def _dispatch_loop(self, e, r, ctx, h_attn, sel, base_norms,
                       run, B, S, D, n, err, beta1, wn):
        """One hypernet at a time.  The reference path; used in training."""
        flat_ctx = ctx.reshape(B * S, D)
        flat_attn = h_attn.reshape(B * S, D)
        flat_out = h_attn.reshape(B * S, D)
        fb, fl = (None if t is None else t.reshape(B * S) for t in err)
        for i in range(n):
            idx = (sel == i).reshape(-1).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out_g = run(_hyper_apply, e.hypers[i], r, flat_ctx[idx].unsqueeze(0),
                        base_norms, e, flat_attn[idx].unsqueeze(0),
                        None if fb is None else fb[idx].unsqueeze(0),
                        None if fl is None else fl[idx].unsqueeze(0), beta1, wn)
            # .to(): dora_linear mixes fp32 reductions with a bf16 value path, so
            # the result can be wider than the target buffer.
            flat_out = flat_out.index_copy(
                0, idx, out_g.squeeze(0).to(flat_out.dtype))
        return flat_out.reshape(B, S, D)

    def _dispatch_sorted(self, e, r, ctx, h_attn, sel, base_norms,
                         bank, B, S, D, n, err, run, beta1, wn):
        """Whole bank in one checkpointed call, no padding.  THE DEFAULT.

        Sort the tokens by owning hypernet once (argsort is a permutation, so no
        group ever pads to another's size), then the trunk runs as three grouped
        GEMMs over the segment offsets and everything after it -- AdapterBasis and
        the DoRA FFN, ~97% of the arithmetic and hypernet-INDEPENDENT -- runs once
        over all M tokens instead of n times over slices of them.  That second half
        is the win: the loop was re-launching the shared tail per hypernet for no
        reason but that its input was gathered per hypernet.

        What the loop cost was not FLOPs.  Each of its n iterations called
        `.nonzero()`, which sizes its output from the data and so reads the tensor
        back to the host: 8 GPU syncs per recurrence, ~248 per forward, on top of
        n copies of every launch in the tail.  Neither survives here; a training
        step goes from 99,480 CUDA launches to 33,699.  See config.dispatch.

        Same arithmetic as _dispatch_loop, reassociated; gradients pinned to it in
        fp64 by test_sorted_dispatch_gradients_match_the_loop.  Requires sel >= 0,
        which holds by construction: the router fires at r == r_free and dispatch
        only runs at r > r_free.
        """
        M = B * S
        flat_sel = sel.reshape(-1)
        # stable so a rerun on the same routing sorts to the same order.  The
        # segment ends come off the sorted vector by searchsorted rather than from
        # a bincount, whose output size is read from the data and so would put a
        # host sync back on the hot path -- the one the loop is here to remove.
        order = torch.argsort(flat_sel, stable=True)
        seg = flat_sel[order]
        offs = torch.searchsorted(seg, torch.arange(n, device=seg.device),
                                  right=True).to(torch.int32)
        fb, fl = (None if t is None else t.reshape(M)[order] for t in err)
        flat_out = h_attn.reshape(M, D)
        ctx_s, attn_s = ctx.reshape(M, D)[order], flat_out[order]
        step = self.c.dispatch_chunk or M
        parts = []
        for lo in range(0, M, step):
            hi = min(lo + step, M)
            # Segment ends clamped into the slice: a segment straddling a chunk
            # boundary is simply split, which costs nothing -- rows are independent.
            parts.append(run(_grouped_apply, bank, r, ctx_s[lo:hi], base_norms, e,
                             attn_s[lo:hi],
                             None if fb is None else fb[lo:hi],
                             None if fl is None else fl[lo:hi],
                             seg[lo:hi], (offs - lo).clamp(0, hi - lo),
                             self.c.r_ceiling, beta1, wn))
        out = parts[0] if len(parts) == 1 else torch.cat(parts)
        # .to(): dora_linear mixes fp32 reductions with a bf16 value path.
        return flat_out.index_copy(
            0, order, out.to(flat_out.dtype)).reshape(B, S, D)

    def _dispatch_bmm(self, e, r, ctx, h_attn, sel, base_norms,
                      bank, B, S, D, n, err, run, beta1, wn):
        """Whole bank in one set of bmms.  Same arithmetic as _dispatch_loop,
        reassociated -- see forward_token_bank.  Superseded as the default by
        _dispatch_sorted, which reassociates the same way WITHOUT the padding that
        makes this fall back below; kept because
        test_batched_dispatch_gradients_match_the_loop pins its gradients to the
        loop's at 1e-10 in fp64 and it is the only padded path.

        Groups are padded to the largest, so this loses if the router collapses:
        K -> total tokens and the bank does n times the work.  Falls back when
        padding would cost more than the launches it saves.
        """
        flat_ctx = ctx.reshape(B * S, D)
        flat_attn = h_attn.reshape(B * S, D)
        flat_out = h_attn.reshape(B * S, D)

        act = (sel >= 0).reshape(-1)
        groups = [(act & (sel.reshape(-1) == i)).nonzero(as_tuple=True)[0]
                  for i in range(n)]
        K = max((g.numel() for g in groups), default=0)
        if K == 0:
            return flat_out.reshape(B, S, D)
        total = sum(g.numel() for g in groups)
        # Two ways batching stops paying: padding waste (router collapsed, so
        # most of the [n,K] block is dead), and sheer size (the generated
        # [n,K,n_out] no longer fits comfortably, and memory pressure costs more
        # than the launches saved).
        if (not self.c.force_padded_dispatch and
                (n * K > 3 * max(1, total)
                 or n * K * sum(e.basis.splits) > self.c.bmm_max_elems)):
            return self._dispatch_loop(e, r, ctx, h_attn, sel, base_norms,
                                       run, B, S, D, n, err, beta1, wn)

        idx = flat_ctx.new_zeros((n, K), dtype=torch.long)
        valid = flat_ctx.new_zeros((n, K), dtype=torch.bool)
        for i, g in enumerate(groups):
            k = g.numel()
            if k:
                idx[i, :k] = g
                valid[i, :k] = True

        fb, fl = (None if t is None else t.reshape(B * S)[idx] for t in err)
        # Checkpointed as ONE unit, the same way the loop checkpoints each
        # hypernet: generation + application is the memory-heavy part either way.
        out = run(_bank_apply, bank, r, flat_ctx[idx], base_norms, e,
                  flat_attn[idx], fb, fl, self.c.r_ceiling, beta1, wn)
        return flat_out.index_copy(
            0, idx[valid], out[valid].to(flat_out.dtype)).reshape(B, S, D)

    def _diversity_loss(self, bank, r, ctx, base_norms, sample=256):
        """Off-diagonal squared cosine similarity between hypernets' codes on the
        SAME sampled tokens -- penalises redundancy, not collapse (that's aux's
        job).  Only meaningful under router_warmup, where dispatch is uniform
        random and gives no other pressure to differentiate the bank.  Codes
        only (not the basis/DoRA factors), so this is one small trunk forward on
        a token subsample, not a full generate+apply."""
        if self.c.n_hypernets < 2:
            return ctx.new_zeros(())
        B, S, D = ctx.shape
        flat = ctx.reshape(B * S, D)
        idx = torch.randperm(flat.shape[0], device=flat.device)[:min(sample, flat.shape[0])]
        sub = flat[idx].unsqueeze(0).expand(self.c.n_hypernets, -1, -1)
        codes = forward_token_bank(bank, r, sub, base_norms, self.c.r_ceiling)
        codes = torch.nn.functional.normalize(codes, dim=-1)
        sim = torch.einsum('nkd,mkd->nmk', codes, codes).mean(-1)
        eye = torch.eye(self.c.n_hypernets, device=sim.device, dtype=torch.bool)
        return sim[~eye].pow(2).mean()

    def forward(self, h, aux=None, *, return_states=False, steps=None, router_warmup=False):
        """Run the shared recurrent C2 body.

        ``steps`` is used by activation distillation to expose exactly one state
        after each continuous internal pass.  It does not reset the router,
        hypernetwork bank, DoRA factors, or input-injection state between passes.
        A complete target sequence requires every requested pass, so halting is
        observed but cannot stop a ``steps``-forced trace early.

        ``router_warmup``: dispatch ignores the router's argmax and assigns each
        token a uniform-random hypernet instead, so every hypernet gets roughly
        equal early training signal regardless of the router's (still near-random)
        preferences -- avoids the router latching onto whichever hypernet an
        arbitrary early imbalance favours.  The router itself still runs (so aux
        stats stay meaningful) but its output does not affect dispatch this call.
        Read back afterward via ``self._div_loss`` (a zero tensor when unused, so
        the return signature does not change and existing callers are unaffected).
        """
        self._div_loss = h.new_zeros(())
        c = self.c
        max_steps = c.c_max2 if steps is None else steps
        if not 1 <= max_steps <= c.c_max2:
            raise ValueError(f"steps must be in [1, {c.c_max2}], got {max_steps}")
        B, S, D = h.shape
        e, n = self.expert, c.n_hypernets
        bank = None                     # stacked weights, built lazily
        wn = None                       # dyn_base(), built lazily
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if (
            c.grad_checkpoint and self.training) else (lambda f, *a: f(*a))

        # Input injection: h0 re-added every recurrence (now via Eq 3's weighted
        # mixing, not a plain add), so every position keeps its own signal at any
        # depth and gradient keeps a short path to the block input.  Standard for
        # recurrent-depth LMs (Geiping et al. 2025); still true under the new
        # halting rule that injection materially helps convergence, even though
        # the specific measured numbers here were against the previous rule --
        # see problems.md P1.
        h0 = h

        depth = torch.zeros(B, S, dtype=torch.int32, device=h.device)
        # -1 until the router has spoken; r_free recurrences run unadapted.
        sel = torch.full((B, S), -1, dtype=torch.long, device=h.device)
        gate = torch.ones(B, S, 1, device=h.device)

        # Eq 2/3 residual scaling (arXiv:2606.18206), single global scalars.
        alpha1, alpha2 = e.alpha1, self.alpha2
        a1_pow2L = alpha1 ** 4                 # 2L=4: 2 KDA + 2 FFN per recurrence
        beta2 = 1.0 - alpha2 * a1_pow2L
        beta1 = beta2 * (1.0 - alpha1) / (1.0 - a1_pow2L)

        # Fixed-point halting (Algorithm 1, "FPOpt"), batch-uniform -- see module
        # docstring.  eta/patience/r_star are Python scalars, not tensors: no
        # gradient is meant to flow through this control state (spec 15).
        eta = c.fp_eta0
        patience = c.fp_patience
        r_star = None            # best residual seen; None until the first reading
        r_last = None            # previous recurrence's reading, batch scalar
        states = []

        for r in range(1, max_steps + 1):
            # ONCE per recurrence, not per hypernet.  Two ways of NOT checkpointing
            # this one leaf were measured and both lost:
            #
            #   un-checkpointed        gradients bit-identical, and it holds every
            #                          recurrence's KDA/FFN activations live:
            #                          3.15 -> 10.28 GB allocated at batch 8, i.e.
            #                          +0.89 GB per unit of batch, so batch 16 wants
            #                          ~20 GB and OOMs.  No speed to show for it
            #                          either (paired median -1.4% at batch 8,
            #                          inside this card's drift).
            #   torch.compile'd        correct here, unlike an earlier reading:
            #                          bit-identical gradients in bf16 both under
            #                          and without checkpoint (torch 2.13, 27 graphs,
            #                          24 breaks at FLA's disabled kernels).  It
            #                          saves 1.0 GB reserved and costs 4.25% of a
            #                          step, paired, negative in every round.
            h_attn = run(e.stage_attn, h, r, beta1)
            # Content signal for weight generation only now -- the halting
            # residual (below) compares the raw pre-damping candidate to h
            # directly, per Algorithm 1, not this normalised view.
            ctx = (h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True)
                                   + c.eps)).to(h.dtype)

            if r <= c.r_free:
                # No hypernetwork selected yet: base weights = DoRA's identity case.
                out = run(e.stage_dyn, h_attn, r, beta1)
            else:
                base_norms = e.base_norms()
                if wn is None:
                    # Outside `run`, so it is a checkpoint INPUT rather than a
                    # recomputed intermediate: derived once per forward, not once
                    # per hypernet per recurrence.
                    wn = e.dyn_base()
                # Gathered dispatch: each hypernet runs only on tokens it owns, so
                # total work is O(B*S), not O(n*B*S) (~6 min/step when measured
                # unconditional).  `if idx.numel()` is a data-dependent branch and
                # is why this model is single-GPU only -- it would desync a DDP
                # all-reduce.  First thing to change if that ever matters.
                eb = None if r_star is None else torch.full((B, S), r_star, device=h.device)
                el = None if r_last is None else torch.full((B, S), r_last, device=h.device)
                err = (eb, el)                # as known before this recurrence
                if c.dispatch == "loop":
                    out = self._dispatch_loop(e, r, ctx, h_attn, sel,
                                              base_norms, run, B, S, D, n, err,
                                              beta1, wn)
                else:
                    if bank is None:
                        bank = stack_bank(e.hypers)   # once per forward, not per r
                    fn = (self._dispatch_bmm if c.dispatch == "bmm"
                          else self._dispatch_sorted)
                    out = fn(e, r, ctx, h_attn, sel, base_norms, bank,
                             B, S, D, n, err, run, beta1, wn)

            # Eq 3: undamped candidate next state, before Algorithm 1's damping.
            z_cand = alpha2 * out + beta2 * h0

            # Checked from r=1: the algorithm may decide one recurrence was
            # enough, even within the unadapted (r<=r_free) phase -- r_free
            # bounds when ROUTING may start, not a minimum depth.  At r=1, h is
            # the raw pre-recurrence input; that is a legitimate, meaningful
            # comparison (Algorithm 1 has no warmup concept, z_0 is just whatever
            # the iteration starts from).
            with torch.no_grad():            # spec 15: halting needs no gradient
                r_val = fixed_point_residual(z_cand.detach(), h.detach(), c.eps).mean().item()

            # Algorithm 1 line 6: damped update replaces the undamped z <- z_cand.
            h = eta * z_cand + (1.0 - eta) * h
            if c.recur_val_clip:
                h = h.clamp(-c.recur_val_clip, c.recur_val_clip)
            if c.recur_grad_clip:
                h = _ClipRecurGrad.apply(h, c.recur_grad_clip)
            depth = depth + 1

            if steps is None and not c.no_halt:
                if r_star is None or r_val < r_star:
                    r_star, patience = r_val, c.fp_patience
                else:
                    patience -= 1
                    if patience <= 0 and r_val > c.fp_tau:
                        eta, patience = c.fp_gamma * eta, c.fp_patience
            r_last = r_val

            if return_states:
                states.append(h)

            # Commitment boundary: router re-selection, unconditional -- fires
            # BEFORE the halting check below, every time it's due, regardless of
            # whether this recurrence turns out to be the last.  Router
            # selection is not contingent on continuing to another depth (it
            # picks the adapter for a possible next depth, whether or not one
            # happens), so it must not be gated behind the decision to halt --
            # gating it there means a model that always halts at depth 1 never
            # exercises the router at all.
            boundary = r >= c.r_free and (r - c.r_free) % c.turns == 0
            if boundary:
                # out_norm applies to the ROUTER'S INPUT, not the residual stream
                # (stage_dyn returns h raw).  Router is a bare Linear with no norm
                # of its own and the stream grows under pre-norm, so without this
                # its logits grow with depth and the softmax saturates.
                probs = self.router(e.out_norm(h))
                if router_warmup:
                    nxt = torch.randint(0, n, sel.shape, device=h.device)
                else:
                    nxt = probs.argmax(-1)
                g = probs.gather(-1, nxt.unsqueeze(-1))
                if aux is not None:
                    aux.add(probs, nxt)
                if router_warmup:
                    if bank is None:
                        bank = stack_bank(e.hypers)
                    self._div_loss = self._div_loss + self._diversity_loss(
                        bank, r, ctx, e.base_norms())
                # Gate multiply is the router's only gradient path (argmax has
                # none).  Nothing downstream normalises the stream, so it cannot be
                # cancelled.
                sel = nxt
                gate = g.to(gate.dtype)
                if c.router_gate != "off":
                    gm = g.to(h.dtype)
                    if c.router_gate == "straight":
                        # Identity forward, same d/dg: the router still learns, but
                        # the residual stream is not scaled by ~1/n every boundary.
                        gm = gm / gm.detach()
                    h = gm * h

            # Checked every recurrence, not gated to a boundary -- see module
            # docstring on why this differs from the previous rule's sync budget.
            if r == max_steps or (steps is None and not c.no_halt and r_val < c.fp_tau):
                break

        return (h, depth, states) if return_states else (h, depth)
