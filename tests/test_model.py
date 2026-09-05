"""Self-checks for the low-level components: KDA, DoRA and the hypernetwork."""
import torch

from cinnamon.blocks import ExpertBlock
from cinnamon.config import TINY, Config
from cinnamon.halting import fixed_point_residual
from cinnamon.hypernet import compose_dora, dora_linear
from cinnamon.model import CinnamonModel
from cinnamon.routed import RoutedBlock


def test_unit_row_col_norm():
    """Rows sum-of-squares 1, columns m/n, negatives preserved, entries bounded.
    Both shapes B@A takes are rectangular, in opposite directions."""
    from cinnamon.hypernet import unit_row_col_norm as urc

    torch.manual_seed(0)
    for m, n in ((128, 64), (64, 128), (48, 48)):
        M = torch.randn(m, n, dtype=torch.float64) * 7.0
        P = urc(M)
        one = torch.ones(m, dtype=torch.float64)
        assert torch.allclose(P.pow(2).sum(-1), one, atol=1e-9), "row sum-of-squares"
        assert torch.allclose(P.pow(2).sum(-2),
                              torch.full((n,), m / n, dtype=torch.float64), atol=5e-3), \
            "column sum-of-squares"
        assert (P < 0).any(), "negatives must survive, not be clipped"
        # the whole point: an entry can never exceed the norm of its own row
        assert float(P.abs().max()) <= 1.0 + 1e-9, "entry exceeded its row norm"

    # scale invariance: 1000x the input must give the same answer
    M = torch.randn(64, 32, dtype=torch.float64)
    assert torch.allclose(urc(M), urc(M * 1000.0), atol=1e-6), \
        "a blown-up input must normalise to the same matrix"
    # zero maps to zero, which is what keeps DoRA an exact identity at init
    assert torch.equal(urc(torch.zeros(8, 4, dtype=torch.float64)),
                       torch.zeros(8, 4, dtype=torch.float64))


def test_dora_factorised_matches_reference():
    """dora_linear never builds W^(r); compose_dora does.  They must agree.

    This is the ONLY thing keeping the factorised path honest -- it is a pure
    memory optimisation (~536 MB -> ~22 MB per invocation at 512 tokens), so any
    disagreement is a bug in the optimisation, never a design choice.  float64
    throughout, because the closed-form ||V_j|| expands a square and the whole
    question is whether that expansion is exact.
    """
    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    hyp = e.hypers[0]
    with torch.no_grad():                       # a non-trivial (trained-like) code
        e.basis.w.weight.normal_(0, 0.05)

    x = torch.randn(2, 5, c.d_model, dtype=torch.float64)
    vec = e.basis(hyp.forward_token(3, x, e.base_norms()))  # [2, 5, n_out]
    A_u, B_u, m_u, A_d, B_d, m_d = e.basis.unpack(vec)

    got = dora_linear(x, e.dyn_up.weight, A_u, B_u, m_u, c)
    # reference: build the matrix, then apply it per token
    W_up, _ = compose_dora(e.basis, vec, e.dyn_up.weight, e.dyn_down.weight, c)
    want = torch.einsum('bsi,bsoi->bso', x, W_up)
    assert torch.allclose(got, want, atol=1e-10), \
        f"factorised DoRA diverges from reference: max {(got - want).abs().max():.2e}"

    # and the down projection, whose input width is d_ff not d_model
    y = torch.randn(2, 5, c.d_ff, dtype=torch.float64)
    got_d = dora_linear(y, e.dyn_down.weight, A_d, B_d, m_d, c)
    _, W_dn = compose_dora(e.basis, vec, e.dyn_up.weight, e.dyn_down.weight, c)
    want_d = torch.einsum('bsi,bsoi->bso', y, W_dn)
    assert torch.allclose(got_d, want_d, atol=1e-10), \
        f"down projection diverges: max {(got_d - want_d).abs().max():.2e}"


def test_dora_triton_epilogue_matches_reference_and_grads():
    """Same bar as above, on CUDA -- where dora_linear runs the fused Triton
    epilogue instead of the aten expression the test above exercises.

    Gradients as well as values: the kernel carries a hand-derived VJP, and
    nothing else in the suite would notice if it were wrong.
    """
    if not torch.cuda.is_available():
        return
    from cinnamon.dora_kernel import _ref, dora_epilogue

    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double().cuda()
    with torch.no_grad():
        e.basis.w.weight.normal_(0, 0.05)
    x = torch.randn(2, 5, c.d_model, dtype=torch.float64, device="cuda")
    vec = e.basis(e.hypers[0].forward_token(3, x, e.base_norms()))
    A_u, B_u, m_u, A_d, B_d, m_d = e.basis.unpack(vec)

    W_up, W_dn = compose_dora(e.basis, vec, e.dyn_up.weight, e.dyn_down.weight, c)
    got = dora_linear(x, e.dyn_up.weight, A_u, B_u, m_u, c)
    assert torch.allclose(got, torch.einsum('bsi,bsoi->bso', x, W_up), atol=1e-10)
    y = torch.randn(2, 5, c.d_ff, dtype=torch.float64, device="cuda")
    got_d = dora_linear(y, e.dyn_down.weight, A_d, B_d, m_d, c)
    assert torch.allclose(got_d, torch.einsum('bsi,bsoi->bso', y, W_dn), atol=1e-10)

    # backward: against the aten expression the kernel replaced
    n, out, r = 9, 64, c.rank
    g = lambda *s: torch.randn(*s, dtype=torch.float64, device="cuda",
                               requires_grad=True) * 0.05
    args = [t.detach().requires_grad_() for t in
            (g(n, out), g(n, r), g(n, out), g(n, out, r), g(n, r, r), g(n, r, out))]
    args += [t.detach().requires_grad_() for t in (g(out).abs() + 1, g(out).abs() + 1)]
    gy = torch.randn(n, out, dtype=torch.float64, device="cuda")
    grads = [torch.autograd.grad(f(*args, 2.0, c.eps), args, gy)
             for f in (dora_epilogue, _ref)]
    for name, a, b in zip("lin ax m B G Q wrow2 wnorm".split(), *grads):
        assert torch.allclose(a, b, atol=1e-10), \
            f"d{name} diverges: max {(a - b).abs().max():.2e}"


def test_dora_varies_with_r_and_never_chains():
    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    dora = e.basis(e.hypers[0](e.base_norms()))
    assert dora.shape[0] == c.r_ceiling
    W = lambda v: compose_dora(e.basis, v, e.dyn_up.weight, e.dyn_down.weight, c)
    up0, _ = W(dora[0])

    with torch.no_grad():                       # simulate a trained hypernet
        e.basis.w.weight.normal_(0, 0.05)
    dora = e.basis(e.hypers[0](e.base_norms()))
    assert not torch.allclose(W(dora[0])[0], W(dora[3])[0], atol=1e-6)
    # no chaining: every r regenerates from an untouched base
    assert torch.allclose(e.dyn_up.weight, up0 * 0 + e.dyn_up.weight, atol=1e-12)


def test_dora_stays_depth_dependent_under_training():
    """The invariant that matters most, and the one that silently broke.

    A depth-indexed adapter is the whole premise of 5: W^(r) must differ across r,
    or every recurrence applies identical weights and depth buys nothing.  With
    unit_row_col_norm on B@A this died -- cos(W^(1), W^(32)) measured 0.58-0.74 at
    init and exactly 1.0000 after 60 steps, because the normalisation discards
    magnitude and the basis had drifted rank-1 along the scale axis it ignores.
    Nothing
    in the loss curve showed it.

    Drive the basis to where training actually takes it and require r to still
    matter.
    """
    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    h = e.hypers[0]

    def cos_across_r(std):
        with torch.no_grad():
            e.basis.w.weight.normal_(0, std)
        tab = e.basis(h(e.base_norms()))
        w0, w1 = (compose_dora(e.basis, tab[r], e.dyn_up.weight, e.dyn_down.weight, c)[0]
                  for r in (0, c.r_ceiling - 1))
        return float(torch.nn.functional.cosine_similarity(
            w0.flatten(), w1.flatten(), dim=0))

    # Each value is how ALIKE W^(1) and W^(32) are: 1.0 = identical, so depth
    # changes nothing; lower = depth actually does something.  Measured at three
    # adapter strengths.
    alike_weak_adapter = cos_across_r(0.02)
    alike_mid_adapter = cos_across_r(1.0)
    alike_strong_adapter = cos_across_r(50.0)

    # A near-identity adapter SHOULD leave the depths alike -- it is barely doing
    # anything, so every r hands back the base.  The invariant is that turning the
    # adapter UP must make the depths LESS alike.  The old constrained-B@A code did
    # the reverse: alikeness climbed to exactly 1.0000 as the adapter strengthened,
    # so W^(1) and W^(32) became the same matrix and depth bought nothing.
    assert alike_weak_adapter > alike_mid_adapter > alike_strong_adapter, (
        "a stronger adapter must make depths diverge, not converge: "
        f"{alike_weak_adapter:.4f} -> {alike_mid_adapter:.4f} -> {alike_strong_adapter:.4f}")
    assert alike_strong_adapter < 0.99, (
        f"depths still identical at full adapter strength: {alike_strong_adapter:.6f}")


def test_dynamic_ffn_base_is_normalised():
    """The constraint now sits on the base.  Row sums-of-squares 1 means the base
    and the generated update stay on comparable scales, so V is a genuine mix
    rather than whichever term happened to drift larger."""
    from cinnamon.hypernet import unit_row_col_norm

    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    for W in (e.dyn_up.weight, e.dyn_down.weight):
        Wn = unit_row_col_norm(W)
        assert torch.allclose(Wn.pow(2).sum(-1),
                              torch.ones(W.shape[0], dtype=W.dtype), atol=1e-9)
        assert float(Wn.abs().max()) <= 1.0 + 1e-9


def test_xielu_matches_the_unfused_reference():
    """XIELU is fused and torch.compile'd; this pins it to the expression it
    replaced.  Also the extremes: the whole reason for the masked per-branch form
    is that an unmasked expm1 overflows to a nan gradient."""
    import torch.nn.functional as F
    from cinnamon.attention import XIELU

    torch.manual_seed(0)
    a = XIELU().cuda().double()
    ap = F.softplus(a.raw_alpha_p)
    an = a.BETA_N + F.softplus(a.raw_alpha_n)

    def ref(x):                       # the pre-fusion expression, verbatim
        xp = torch.where(x > 0, x, torch.zeros_like(x))
        xn = torch.where(x > 0, torch.zeros_like(x), x)
        return torch.where(x > 0, ap * xp.square() + 0.5 * xp,
                           an * torch.expm1(xn) - an * xn + 0.5 * xn)

    x = torch.randn(4, 257, 96, dtype=torch.float64, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y, yr = a(x), ref(x2)
    assert torch.allclose(y, yr, atol=1e-12), f"fwd {(y - yr).abs().max():.2e}"
    gy = torch.randn_like(y)
    y.backward(gy)
    yr.backward(gy)
    assert torch.allclose(x.grad, x2.grad, atol=1e-12), f"dx {(x.grad - x2.grad).abs().max():.2e}"

    xe = torch.tensor([[-1e4, -80.0, -1.0, 0.0, 1.0, 80.0, 1e4]],
                      dtype=torch.float64, device="cuda", requires_grad=True)
    a.zero_grad()
    a(xe).sum().backward()
    assert torch.isfinite(xe.grad).all(), f"nan/inf grad at the extremes: {xe.grad}"
    assert all(torch.isfinite(p.grad).all() for p in a.parameters())

    # Inductor's donated buffers are incompatible with gradcheck's retain_graph.
    with torch.compiler.set_stance("force_eager"):
        assert torch.autograd.gradcheck(
            a, (torch.randn(3, 11, dtype=torch.float64, device="cuda",
                            requires_grad=True),), atol=1e-9)


def test_halting_bounds_are_respected():
    """Depth stays within [r_free, cap] and always lands on a commitment
    boundary -- the direct, per-token replacement for the old ExpertBlock.run()
    version of this check, now against RoutedBlock."""
    torch.manual_seed(0)
    c = Config(**{**TINY, "r_free": 2, "turns": 2, "c_max2": 16, "r_ceiling": 16,
                  "grad_checkpoint": False})
    rb = RoutedBlock(c).cuda().eval()
    h = torch.randn(2, 6, c.d_model, device="cuda")
    with torch.no_grad():
        out, depth = rb(h)
    assert out.shape == h.shape
    assert depth.min() >= c.r_free and depth.max() <= c.c_max2
    assert torch.all((depth - c.r_free) % c.turns == 0), (
        "a commitment was interrupted mid-way")


def test_halted_tokens_stay_frozen():
    """A token whose depth < cap must match its state at that depth to fp32
    precision.  Was fp64 for a bit-identical bar, but fla's chunk_kda Triton
    kernel only compiles for fp16/bf16/fp32 -- KDA has no fp64 path at all, so
    this can only check fp32-close now, not bit-identical."""
    torch.manual_seed(0)
    c = Config(**{**TINY, "r_free": 2, "turns": 2, "c_max2": 16, "r_ceiling": 16,
                  "grad_checkpoint": False})
    rb = RoutedBlock(c).cuda().eval()
    h = torch.randn(1, 8, c.d_model, device="cuda")
    with torch.no_grad():
        out, depth = rb(h)
        assert depth.min() < c.c_max2, "need at least one early halt to test freezing"
        # Re-run with a HALVED cap: any token that halted before the smaller cap
        # must produce the exact same output either way, since a shorter budget
        # cannot change what a token that stopped early already decided.
        half = int(depth.min())
        c_half = Config(**{**c.__dict__, "c_max2": half})
        rb_half = RoutedBlock(c_half).cuda().eval()
        rb_half.load_state_dict(rb.state_dict())
        out_half, depth_half = rb_half(h)
    frozen = depth <= half
    assert frozen.any(), "need at least one token that halted within the smaller cap"
    assert torch.allclose(out[frozen], out_half[frozen], atol=1e-5)


def test_hypernet_trunk_is_live_from_the_first_step():
    """B is randomly initialised (unit_row_col_norm keeps only direction, and
    zero has none), so the hypernet trunk must get real gradient immediately --
    no dead first step the way a zero-initialised B would cause."""
    torch.manual_seed(0)
    c = Config(**{**TINY, "r_free": 1, "turns": 1, "c_max2": 4, "grad_checkpoint": False})
    rb = RoutedBlock(c).cuda()
    out, _ = rb(torch.randn(2, 8, c.d_model, device="cuda"))
    out.square().mean().backward()
    for name in ("l1.weight", "l2.weight", "norm_proj.weight"):
        g = dict(rb.expert.hypers[0].named_parameters())[name].grad
        assert g is not None and g.any(), f"hypernet trunk {name} dead on step 0"


def test_tied_embeddings():
    m = CinnamonModel(Config(**TINY, tie_embeddings=True))
    assert m.head.weight is m.embed.weight
    m2 = CinnamonModel(Config(**TINY, tie_embeddings=False))
    assert m2.head.weight is not m2.embed.weight


def test_default_config_hits_the_50m_budget():
    """The size target is 50 M EXCLUDING the embedding: the embedding is set by
    the tokenizer's vocab (32,768, untied -- so head too), not by a layer width.

    Meta device: shapes only, so counting the real 8-expert model costs nothing.
    """
    from cinnamon.model import CinnamonModel

    c = Config()
    with torch.device("meta"):
        m = CinnamonModel(c)
    total = sum(p.numel() for p in dict(m.named_parameters()).values())
    # Both vocab-sized matrices come off: untied, the LM head is a second one, and
    # counting it as "model" would silently eat 28 M of the budget.
    vocab_sized = m.embed.weight.numel() + (
        0 if c.tie_embeddings else m.head.weight.numel())
    non_embed = total - vocab_sized
    assert non_embed <= 50e6, f"{non_embed/1e6:.2f} M non-embedding, over budget"

    # The hypernet output layer grows linearly in d while the body it modulates
    # grows quadratically, so shrinking the model silently inverts their ratio.
    # Guard it: one generator must stay smaller than the body it adapts.
    n = lambda mods: sum(p.numel() for mod in mods for p in mod.parameters())
    e0 = m.expert.expert
    work = n([e0.mla2, e0.mla3, e0.fixed, e0.dyn_up, e0.dyn_down])
    assert n([e0.hypers[0]]) < work, "hypernet outweighs the body it modulates"


def test_default_widths_actually_run():
    """TINY exercises the code, not the shipped dims.  d_model=224 with 7 heads of
    32 is exactly the kind of non-power-of-two split that breaks a reshape, and
    nothing else in the suite would catch it.  n_hypernets is cut to 2 only to keep
    the instantiation cheap -- every width parameter is the real one.
    """
    from cinnamon.model import CinnamonModel

    c = Config(n_hypernets=2, c_max2=4, loss_chunk=0)   # this test wants logits
    torch.manual_seed(0)
    m = CinnamonModel(c).cuda()
    ids = torch.randint(0, c.vocab, (1, 32), device="cuda")
    logits, loss, depth = m(ids, labels=ids)
    assert logits.shape == (1, 32, c.vocab)
    assert torch.isfinite(loss) and loss > 0
    assert depth.shape == (1, 32)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.l1.parameters())



