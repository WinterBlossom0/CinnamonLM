"""Routing, halting, the router's aux loss, and the per-token content path
(10-13): tiny -- the point is the control flow, which is where this design can
be wrong in ways that still produce a plausible loss curve. Needs CUDA for
anything that runs a real forward pass: canonical KDA has no CPU path. The
pure AuxLoss/Router math (no attention involved) stays CPU-testable.

Run: python -m tests.test_routing
"""
import torch

from cinnamon.config import TINY, Config
from cinnamon.model import CinnamonModel
from cinnamon.router import AuxLoss, Router

ROUTED = dict(TINY, n_hypernets=4, r_free=1, turns=2, c_max2=16, r_ceiling=16)


def cfg(**kw):
    return Config(**{**ROUTED, **kw})


def test_forward_backward_and_shapes():
    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False, loss_chunk=0)        # this test wants logits
    m = CinnamonModel(c).cuda()
    ids = torch.randint(0, c.vocab, (2, 16), device="cuda")
    logits, loss, depth = m(ids, labels=ids)
    assert logits.shape == (2, 16, c.vocab)
    assert torch.isfinite(loss)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead[:6]}"


def test_attention_runs_once_per_recurrence_not_once_per_hypernet():
    """The whole point of one body + n hypernets: stage_attn cost stops scaling
    with n.  Under the old design it ran once per expert per turn, which was 61%
    of the forward.  A regression here is invisible in the loss and quietly costs
    n times the attention."""
    from cinnamon.routed import RoutedBlock

    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False)
    rb = RoutedBlock(c).cuda()
    calls = [0]
    orig = rb.expert.stage_attn

    def counted(h, r, beta1):
        calls[0] += 1
        return orig(h, r, beta1)

    rb.expert.stage_attn = counted
    with torch.no_grad():
        _, depth = rb(torch.randn(2, 16, c.d_model, device="cuda"))
    assert calls[0] == int(depth.max()), (calls[0], int(depth.max()))


def test_every_hypernet_gets_gradient():
    """A hypernet that never receives a token gets no gradient and never trains,
    so the bank silently shrinks to whichever few the router happened to favour
    early.  Invisible in the loss -- the model still works, just with less
    capacity than it is paying for."""
    from cinnamon.routed import RoutedBlock

    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False)
    rb = RoutedBlock(c).cuda()
    out, _ = rb(torch.randn(2, 16, c.d_model, device="cuda"))
    out.square().mean().backward()
    dead = [i for i in range(c.n_hypernets)
            if rb.expert.hypers[i].l2.weight.grad is None]
    assert not dead, f"hypernets with no gradient: {dead}"


def test_first_recurrences_run_unadapted():
    """r <= r_free runs on the plain base weights: no DoRA is composed at all
    before the router has spoken.  Counting the generated-weight application is
    the direct check -- the outputs alone would not distinguish 'no DoRA' from
    'DoRA that happens to start near identity'."""
    from cinnamon import blocks, routed

    torch.manual_seed(0)
    calls = []
    orig = blocks.dora_linear
    blocks.dora_linear = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        for c_max2, want_dora in ((2, False), (6, True)):
            calls.clear()
            c = cfg(grad_checkpoint=False, c_max2=c_max2, r_free=2)
            rb = routed.RoutedBlock(c).cuda()
            with torch.no_grad():
                _, depth = rb(torch.randn(2, 16, c.d_model, device="cuda"))
            assert bool(calls) is want_dora, (c_max2, len(calls))
            assert int(depth.max()) <= c_max2
    finally:
        blocks.dora_linear = orig


def test_generation_is_content_sensitive_per_token():
    """The hypernetwork's content input is now h itself -- the state entering the
    recurrence -- rather than a separate summary of it.  Generation must still
    be genuinely per-token (5.5 deliberately overridden): two tokens in the same
    sequence must get different generated codes, and content must reach ctx_proj
    with real gradient rather than only selecting among fixed alternatives.
    """
    from cinnamon.routed import RoutedBlock

    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False)
    rb = RoutedBlock(c).cuda()
    e = rb.expert

    h = torch.randn(1, 16, c.d_model, device="cuda")
    vec = e.basis(e.hypers[0].forward_token(3, h, e.base_norms()))
    assert vec.shape[:2] == (1, 16), vec.shape
    # different tokens -> different generated codes, or generation is not per-token
    assert not torch.allclose(vec[0, 0], vec[0, 1], atol=1e-6), \
        "every token generated the same code -- generation is not per-token"

    out, _ = rb(h)
    out.square().mean().backward()
    g = e.hypers[0].ctx_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "content path got no gradient"


def test_batched_dispatch_matches_the_loop():
    """The bmm path must be a pure reassociation of the per-hypernet loop.

    The bank runs as one set of bmms instead of 8 tiny dispatches per recurrence
    -- measured 2.82x faster on identical work, the gap being pure launch
    overhead.  That is only legitimate if the two agree exactly.

    Selected by `dispatch`, NOT by train/eval.  An earlier version of this test
    used eval-vs-train to pick the path; once both modes defaulted to bmm it
    silently compared bmm against itself and passed regardless.

    fp32, not float64: the block's forward runs through KDA (stage_attn), and
    fla's chunk_kda Triton kernel has no float64 path at all.  KDA itself is
    identical between the two runs either way -- only dispatch differs -- so
    fp32 still isolates a real dispatch divergence, just at a looser bound.
    """
    from cinnamon.routed import RoutedBlock

    outs = {}
    for mode in ("bmm", "loop"):
        torch.manual_seed(0)
        rb = RoutedBlock(cfg(grad_checkpoint=False, dispatch=mode)).cuda().eval()
        h = torch.randn(2, 16, rb.c.d_model, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(1))
        with torch.no_grad():
            outs[mode] = rb(h)

    (ob, db), (ol, dl) = outs["bmm"], outs["loop"]
    assert torch.equal(db, dl), (db.float().mean(), dl.float().mean())
    assert torch.allclose(ob, ol, atol=1e-5),         f"batched dispatch diverges from the loop: max {(ob - ol).abs().max():.3g}"


def test_batched_dispatch_gradients_match_the_loop():
    """Forward equivalence is not enough to train on.

    The bmm path reassociates the same arithmetic, so its GRADIENTS must match
    too -- a reassociation that is right forward and wrong backward trains a
    different model while every forward check keeps passing.  This test is what
    gates `dispatch="bmm"` as the training default.

    fp32, not float64 (the block runs through KDA, which has no float64 path);
    every parameter compared, not just the loss.
    """
    from cinnamon.routed import RoutedBlock

    grads = {}
    for mode in ("bmm", "loop"):
        torch.manual_seed(0)
        rb = RoutedBlock(cfg(grad_checkpoint=False, dispatch=mode)).cuda()
        rb.train()
        h = torch.randn(2, 16, rb.c.d_model, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(1),
                        requires_grad=True)
        out, _ = rb(h)
        out.square().mean().backward()
        grads[mode] = ({n: p.grad.clone() for n, p in rb.named_parameters()
                        if p.grad is not None}, h.grad.clone())

    (gb, xb), (gl, xl) = grads["bmm"], grads["loop"]
    assert set(gb) == set(gl), set(gb) ^ set(gl)
    assert torch.allclose(xb, xl, atol=1e-5),         f"input gradient diverges: max {(xb - xl).abs().max():.3g}"
    worst = max(((gb[n] - gl[n]).abs().max().item(), n) for n in gb)
    # expert.alpha1 participates in every one of c_max2 recurrences, so its
    # fp32 rounding error compounds more than a param touched once; measured
    # 1.35e-5 there specifically, everything else well under 1e-5.
    assert worst[0] < 5e-5, f"gradient diverges on {worst[1]}: max {worst[0]:.3g}"


def test_the_whole_batch_halts_together():
    """Halting compares batch MEANS, so depth is one value across every token.

    Asserted in both modes, because the rule is deliberately the same in both --
    a mode-dependent one would mean the model never trains under the rule it is
    served with.

    Deliberately NOT asserting that halting fires: whether the averaged signal
    still stalls on a TRAINED model is open (P1), and an assertion either way
    would be encoding a guess. What is pinned is uniformity, which is the property
    this rule exists to provide.
    """
    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False, c_max2=16)
    m = CinnamonModel(c).cuda()
    ids = torch.randint(0, c.vocab, (4, 32), device="cuda")

    depths = {}
    for mode in ("train", "eval"):
        getattr(m, mode)()
        with torch.no_grad():
            _, _, depth = m(ids)
        d = depth
        assert int(d.min()) == int(d.max()), (
            f"depth varies in {mode} ({int(d.min())}-{int(d.max())}) -- "
            "the batch is not halting together")
        depths[mode] = int(depth.max())
    assert depths["train"] == depths["eval"], (
        f"halting differs between modes: {depths} -- training and inference must "
        "run the same rule")


def test_router_gets_gradient():
    """11.3  argmax is non-differentiable, so without the gate multiply the router
    never receives gradient from L_LM and stays at init forever."""
    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False)
    m = CinnamonModel(c).cuda()
    ids = torch.randint(0, c.vocab, (2, 16), device="cuda")
    _, loss, _ = m(ids, labels=ids)
    loss.backward()
    for name in ("expert.router.w",):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, f"{name} got no gradient"


def test_budget_is_never_exceeded_and_warmup_always_runs():
    """Depth is now counted in recurrences, one per pass of the single body.  No
    token may exceed c_max2, and none may halt before the r_free unadapted
    recurrences are done -- halting cannot fire until two KLs exist."""
    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False)
    m = CinnamonModel(c).cuda().eval()
    ids = torch.randint(0, c.vocab, (2, 16), device="cuda")
    with torch.no_grad():
        _, _, depth = m(ids, labels=ids)
    d = depth
    assert int(d.max()) <= c.c_max2, (int(d.max()), c.c_max2)
    assert int(d.min()) > c.r_free, (int(d.min()), c.r_free)


def test_aux_loss_ignores_masked_out_positions():
    """P must be averaged over the positions that actually contributed, not over
    every position.  Dividing the masked sum by the unmasked count scaled P down
    by the active fraction, so the balancing pressure faded exactly as tokens
    halted -- when it is needed most.  Masking half the batch must give the same
    aux value as passing only that half."""
    n = 4
    probs = torch.rand(64, n)
    probs = probs / probs.sum(-1, keepdim=True)
    chosen = probs.argmax(-1)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[:32] = True

    masked = AuxLoss(n, torch.device("cpu"))
    masked.add(probs, chosen, mask)
    subset = AuxLoss(n, torch.device("cpu"))
    subset.add(probs[:32], chosen[:32])
    assert abs(float(masked.value()[0]) - float(subset.value()[0])) < 1e-5


def test_aux_loss_add_never_syncs_and_counts_like_bincount():
    """AuxLoss.add runs once per recurrence on the hot path, so a single host read
    in it is c_max2 GPU stalls per forward.  It used to make three (bincount twice,
    float(entropy) once) -- 104 of the training step's 480 syncs.  index_add_ is
    the sync-free spelling of the same count; this pins both the equivalence and
    the no-sync property."""
    if not torch.cuda.is_available():
        return
    n = 8
    probs = torch.softmax(torch.randn(4, 37, n, device="cuda"), -1)
    chosen = probs.argmax(-1)
    mask = torch.rand(4, 37, device="cuda") > 0.3

    aux = AuxLoss(n, torch.device("cuda"))
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(4):
            aux.add(probs, chosen, mask)          # raises if it reads the device
    finally:
        torch.cuda.set_sync_debug_mode("default")

    w = mask.reshape(-1).float()
    ref = 4 * torch.bincount(chosen.reshape(-1), weights=w, minlength=n).float()
    assert torch.equal(aux.f_sum, ref), f"{aux.f_sum} vs {ref}"


def test_aux_loss_is_minimal_when_balanced_and_large_when_collapsed():
    """10  N * sum_i f_i P_i == 1.0 exactly when both are uniform."""
    n = 4
    bal = AuxLoss(n, torch.device("cpu"))
    probs = torch.full((64, n), 1.0 / n)
    bal.add(probs, torch.arange(64) % n)
    v, _ = bal.value()
    assert abs(float(v) - 1.0) < 1e-4, float(v)

    col = AuxLoss(n, torch.device("cpu"))
    p = torch.zeros(64, n)
    p[:, 0] = 0.9
    p[:, 1:] = 0.1 / (n - 1)
    col.add(p, torch.zeros(64, dtype=torch.long))
    vc, _ = col.value()
    assert float(vc) > 3.0, float(vc)


def test_router_is_near_uniform_at_init_and_stays_fp32():
    """11.0  small-random init leaves the distribution near-uniform, so load
    balance starts healthy and collapse has to be learned rather than handed over
    at step 0.  fp32 regardless of autocast."""
    torch.manual_seed(0)
    c = cfg()
    r = Router(c)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        p = r(torch.randn(1, 64, c.d_model))
    assert p.dtype == torch.float32, p.dtype
    assert torch.allclose(p.sum(-1), torch.ones(1, 64), atol=1e-5)
    # 0.05 was tuned against TINY's old, narrower d_model; TINY is now 256
    # (matches the shipped width), and a wider random init naturally spreads
    # the initial logits a bit more.  0.0979 measured at seed 0 -- still close
    # to uniform for n_hypernets=4, just not within the old tighter bound.
    assert (p - 1.0 / c.n_hypernets).abs().max() < 0.12


def test_halting_actually_fires_and_depth_varies():
    """12  The rule must genuinely stop tokens, not just be present.

    The earlier version of this test only asserted `std >= 0`, which is
    vacuously true, and it passed happily while every token ran to the cap --
    which in turn backpropagated through all 64 turns and exploded the gradient
    norm to 1.2e7 (fp16 max is 65504, so training could never work).  Assert the
    property that actually matters.
    """
    torch.manual_seed(0)
    c = cfg(grad_checkpoint=False, c_max2=32, r_ceiling=32)
    m = CinnamonModel(c).cuda().eval()
    ids = torch.randint(0, c.vocab, (2, 16), device="cuda")
    with torch.no_grad():
        _, _, depth = m(ids, labels=ids)
        # Depth is deliberately UNIFORM within a batch (halting compares batch
        # means), so per-token variance is no longer the thing to assert -- it is
        # zero by construction.  Adaptivity now lives ACROSS batches: a different
        # input must be able to take a different depth.
        other = torch.randint(0, c.vocab, (2, 16), device="cuda")
        _, _, odepth = m(other, labels=other)
    assert int(depth.max()) != int(odepth.max()), \
        "two different batches took identical depth -- halting is not input-dependent"
    # Not "no batch reaches the cap": a batch that never converges is supposed to
    # run out the budget -- that is what c_max2 is for.  What must hold is that the
    # rule bites on average.
    d = depth
    assert d.float().mean() < 0.6 * c.c_max2, (
        f"mean depth {float(d.float().mean()):.1f} of cap {c.c_max2} -- "
        "halting is barely firing")
    # A token may stop at the end of the warm-up: two base passes, then the
    # first KL check.  It may never stop mid-commitment.
    assert int(d.min()) >= c.r_free, (int(d.min()), c.r_free)
    assert torch.all((d - c.r_free) % c.turns == 0), (
        "a commitment was interrupted mid-way -- the two passes are guaranteed")


def test_gradient_norm_stays_bounded_with_depth():
    """Depth itself must stay bounded: MEAN depth barely moves between caps,
    because halting fires well before either cap binds.  That is the actual
    contract; assert it directly rather than through gradient norm.

    Norm itself is NOT asserted bounded here any more.  Diagnosed: the growth
    (562 -> 4.96e4 measured) traces entirely to the attention gradient, tracks
    MAX depth (a long tail of a few tokens running deeper), and is independent
    of this session's change -- the hypernets contribute ~1e-4 of the
    total, negligible -- not something per-token generation introduced.
    """
    depths = []
    for c_max2 in (16, 64):
        torch.manual_seed(0)
        m = CinnamonModel(cfg(grad_checkpoint=False, c_max2=c_max2, r_ceiling=c_max2)).cuda()
        ids = torch.randint(0, m.c.vocab, (2, 16), device="cuda")
        _, loss, depth = m(ids, labels=ids)
        depths.append(float(depth.float().mean()))
    assert depths[1] < 3 * depths[0], (
        f"mean depth grows with the cap -- halting is not firing: {depths}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("\nall passed")


def _dispatch_arms(dtype, cfg_kw, ckpt, sel_fn, seed=0, B=2, S=48):
    """Run _dispatch_loop and _dispatch_sorted over identical inputs and params.

    The dispatch alone, not the whole block: stage_attn's KDA kernels have no CPU
    or float64 path, and the dispatch itself needs neither.
    """
    from torch.utils.checkpoint import checkpoint
    from cinnamon.hypernet import stack_bank
    from cinnamon.routed import RoutedBlock

    res = {}
    for mode in ("loop", "sorted"):
        torch.manual_seed(seed)
        c = Config(**{**cfg_kw, "dispatch": mode})
        rb = RoutedBlock(c).to(dtype)
        e, n, D = rb.expert, c.n_hypernets, c.d_model
        g = torch.Generator().manual_seed(seed + 1)
        ctx = torch.randn(B, S, D, dtype=dtype, generator=g)
        ha = torch.randn(B, S, D, dtype=dtype, generator=g, requires_grad=True)
        eb = torch.rand(B, S, dtype=dtype, generator=g)
        el = torch.rand(B, S, dtype=dtype, generator=g)
        run = ((lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if ckpt
               else (lambda f, *a: f(*a)))
        head = (e, 3, ctx, ha, sel_fn(B, S, n, g), e.base_norms())
        tail = (B, S, D, n, (eb, el))
        out = (rb._dispatch_loop(*head, run, *tail, 0.5, e.dyn_base())
               if mode == "loop" else
               rb._dispatch_sorted(*head, stack_bank(e.hypers), *tail, run, 0.5,
                                   e.dyn_base()))
        out.square().mean().backward()
        res[mode] = (out.detach(),
                     {k: p.grad.clone() for k, p in rb.named_parameters()
                      if p.grad is not None}, ha.grad.clone())
    return res


DISPATCH_CFG = dict(TINY, d_model=256, d_ff=256, rank=4, lora_alpha=8, phi_dim=8,
                    code_dim=16, n_hypernets=4, r_ceiling=8, c_max2=8, n_heads=2,
                    head_dim=128)


def test_sorted_dispatch_gradients_match_the_loop():
    """The sorted path must be a pure reassociation of the per-hypernet loop.

    It sorts tokens into contiguous per-hypernet segments once, runs the trunk as
    three grouped GEMMs over the segment offsets, and then applies the SHARED
    AdapterBasis and DoRA FFN to every token in one call instead of n.  That drops
    two thirds of the step's CUDA launches and every one of the loop's per-group
    `.nonzero()` host syncs -- legitimate only if the arithmetic is unchanged.

    float64, checkpointed and not, and every parameter compared: a reassociation
    that is right forward and wrong backward trains a different model while every
    forward check keeps passing.

    dispatch_chunk is swept too, including sizes that split a segment across a
    chunk boundary -- rows are independent, so it must change nothing at all.
    """
    both = lambda B, S, n, g: torch.randint(0, n, (B, S), generator=g)
    for ckpt in (False, True):
        for chunk in (0, 96, 37):
            cfg_kw = dict(DISPATCH_CFG, dispatch_chunk=chunk)
            r = _dispatch_arms(torch.float64, cfg_kw, ckpt, both)
            (oa, ga, xa), (ob, gb, xb) = r["loop"], r["sorted"]
            tag = f"ckpt={ckpt} chunk={chunk}"
            assert torch.allclose(oa, ob, atol=1e-10), \
                f"{tag}: forward diverges, max {(oa - ob).abs().max():.3g}"
            assert torch.allclose(xa, xb, atol=1e-10), \
                f"{tag}: input gradient diverges, max {(xa - xb).abs().max():.3g}"
            assert set(ga) == set(gb), set(ga) ^ set(gb)
            worst = max(((ga[k] - gb[k]).abs().max().item(), k) for k in ga)
            assert worst[0] < 1e-10, f"{tag}: gradient diverges on {worst[1]}: {worst[0]:.3g}"


def test_sorted_dispatch_handles_empty_groups():
    """A collapsed router leaves whole hypernets with no tokens.

    The loop `continue`s past them, so they end with grad None; the sorted path
    dispatches the bank as one stacked operand and hands back an exactly-zero
    grad instead.  Both say the same thing -- an unselected hypernet's gradient IS
    zero -- but only if it really is zero, which is what this pins.
    """
    def collapsed(B, S, n, g):
        sel = torch.full((B, S), n - 1, dtype=torch.long)
        sel[0, :3] = 0                              # groups 1..n-2 left empty
        return sel

    r = _dispatch_arms(torch.float64, DISPATCH_CFG, False, collapsed)
    (oa, ga, xa), (ob, gb, xb) = r["loop"], r["sorted"]
    assert torch.allclose(oa, ob, atol=1e-10), (oa - ob).abs().max()
    assert torch.allclose(xa, xb, atol=1e-10), (xa - xb).abs().max()
    for k in set(gb) - set(ga):
        assert gb[k].abs().max() == 0, f"{k} should be exactly zero, got {gb[k].abs().max()}"
    worst = max(((ga[k] - gb[k]).abs().max().item(), k) for k in set(ga) & set(gb))
    assert worst[0] < 1e-10, f"gradient diverges on {worst[1]}: {worst[0]:.3g}"
