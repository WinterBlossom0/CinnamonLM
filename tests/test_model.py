"""Self-checks for the low-level components: KDA, DoRA, the hypernetwork,
AttnRes.  Run: python -m tests.test_model"""
import torch

from cinnamon.attnres import AttnRes
from cinnamon.blocks import ExpertBlock
from cinnamon.config import TINY, Config
from cinnamon.halting import rel_error
from cinnamon.hypernet import compose_dora, dora_linear
from cinnamon.kda import KDA, inv_unit_lower
from cinnamon.model import CinnamonModel
from cinnamon.routed import RoutedBlock


def test_inv_unit_lower():
    torch.manual_seed(0)
    M = torch.randn(3, 16, 16, dtype=torch.float64).tril(-1)
    I = torch.eye(16, dtype=torch.float64)
    assert torch.allclose(inv_unit_lower(M) @ (I + M), I.expand_as(M), atol=1e-9)


def test_kda_chunked_matches_recurrence():
    """The whole point of the chunked form: it must equal the naive delta rule."""
    torch.manual_seed(0)
    for chunk in (4, 8, 16):
        k = KDA(Config(**{**TINY, 'kda_chunk': chunk})).double()
        x = torch.randn(2, 24, 64, dtype=torch.float64)      # 24 % 16 != 0 -> pads
        assert torch.allclose(k(x), k.reference(x), atol=1e-8), f"chunk={chunk}"


def test_kda_survives_correlated_keys():
    """Regression: the UT-transform inverse (I+M)^-1 grows like 2^chunk when keys
    inside a chunk align and beta approaches 1.  At chunk 64 this overflowed fp32
    ~180 steps into a real run; at 32 it was silently 33% wrong.  Drive the model
    into that regime deliberately and require it to still match the exact
    recurrence.
    """
    torch.manual_seed(0)
    k = KDA(Config(**TINY)).double().eval()
    with torch.no_grad():
        k.beta.bias.fill_(6.0)          # beta -> ~1, the hard-overwrite regime
        k.g_up.bias.fill_(-12.0)        # almost no decay, so nothing damps M
    # near-identical inputs => near-identical keys => worst-case conditioning
    base = torch.randn(1, 1, 64, dtype=torch.float64)
    x = base.expand(2, 96, 64) + 0.01 * torch.randn(2, 96, 64, dtype=torch.float64)
    out = k(x)
    assert torch.isfinite(out).all(), "chunked KDA overflowed on correlated keys"
    rel = (out - k.reference(x)).norm() / k.reference(x).norm()
    assert rel < 1e-5, f"chunked KDA diverged from exact recurrence: rel err {rel:.3g}"


def test_kda_survives_a_drifted_gate():
    """The failure this file did not catch for a long time.

    Every other KDA test runs at initialisation, where per-chunk |gcum| is ~7 and
    the old split form (exp(g_t) * exp(-g_s)) was still accurate.  But the gate is
    LEARNED, and training pushes it: a real run reached |gcum| 23.3 by step 250,
    where the split had silently stopped matching the exact recurrence -- 4e-1
    relative error at 18, and no NaN or loss spike to notice it by.

    Subtracting before exponentiating removes the failure mode rather than moving
    it, so this sweeps far past anything training could plausibly reach and
    requires exactness throughout.
    """
    from cinnamon.kda import _G_RANGE

    for chunk in (16, 64):
        for bias in (-3.0, -1.5, 0.5):                 # increasingly hard decay
            torch.manual_seed(0)
            k = KDA(Config(**{**TINY, 'kda_chunk': chunk})).double().eval()
            with torch.no_grad():
                k.g_up.bias.fill_(bias)                # drive the decay gate hard
            x = torch.randn(2, 256, 64, dtype=torch.float64)
            with torch.no_grad():
                out, ref = k(x), k.reference(x)

            # The bound is what makes the split form safe, so assert the bound
            # itself holds -- not merely that this particular input was fine.
            assert float(k._max_gcum) <= _G_RANGE + 1e-9, (
                f"chunk {chunk} bias {bias}: |gcum| {float(k._max_gcum):.2f} escaped "
                f"the {_G_RANGE} bound, so the split form is outside its safe range")
            assert torch.isfinite(out).all()
            rel = float((out - ref).norm() / ref.norm())
            assert rel < 1e-9, (
                f"chunk {chunk} bias {bias}, |gcum| {float(k._max_gcum):.2f}: "
                f"chunked diverged from exact, rel {rel:.3g}")


def test_kda_is_causal():
    """Changing token t must not move any output before t."""
    torch.manual_seed(0)
    k = KDA(Config(**TINY)).double().eval()
    x = torch.randn(1, 20, 64, dtype=torch.float64)
    y = k(x)
    x2 = x.clone()
    x2[:, 12] += 3.0
    y2 = k(x2)
    assert torch.allclose(y[:, :12], y2[:, :12], atol=1e-9)
    assert not torch.allclose(y[:, 12:], y2[:, 12:], atol=1e-6)


def test_attnres_zero_query_is_uniform_average():
    a = AttnRes(Config(**TINY))
    src = [torch.randn(2, 5, 64) for _ in range(4)]
    assert torch.allclose(a(src), torch.stack(src).mean(0), atol=1e-6)


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
        hyp.l3.weight.normal_(0, 0.05)

    x = torch.randn(2, 5, c.d_model, dtype=torch.float64)
    vec = hyp.forward_token(3, x, e.base_norms())          # [2, 5, n_out]
    A_u, B_u, m_u, A_d, B_d, m_d = hyp.unpack(vec)

    got = dora_linear(x, e.dyn_up.weight, A_u, B_u, m_u, c)
    # reference: build the matrix, then apply it per token
    W_up, _ = compose_dora(hyp, vec, e.dyn_up.weight, e.dyn_down.weight, c)
    want = torch.einsum('bsi,bsoi->bso', x, W_up)
    assert torch.allclose(got, want, atol=1e-10), \
        f"factorised DoRA diverges from reference: max {(got - want).abs().max():.2e}"

    # and the down projection, whose input width is d_ff not d_model
    y = torch.randn(2, 5, c.d_ff, dtype=torch.float64)
    got_d = dora_linear(y, e.dyn_down.weight, A_d, B_d, m_d, c)
    _, W_dn = compose_dora(hyp, vec, e.dyn_up.weight, e.dyn_down.weight, c)
    want_d = torch.einsum('bsi,bsoi->bso', y, W_dn)
    assert torch.allclose(got_d, want_d, atol=1e-10), \
        f"down projection diverges: max {(got_d - want_d).abs().max():.2e}"


def test_dora_varies_with_r_and_never_chains():
    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    dora = e.hypers[0](e.base_norms())
    assert dora.shape[0] == c.r_ceiling
    W = lambda v: compose_dora(e.hypers[0], v, e.dyn_up.weight, e.dyn_down.weight, c)
    up0, _ = W(dora[0])

    with torch.no_grad():                       # simulate a trained hypernet
        e.hypers[0].l3.weight.normal_(0, 0.05)
    dora = e.hypers[0](e.base_norms())
    assert not torch.allclose(W(dora[0])[0], W(dora[3])[0], atol=1e-6)
    # no chaining: every r regenerates from an untouched base
    assert torch.allclose(e.dyn_up.weight, up0 * 0 + e.dyn_up.weight, atol=1e-12)


def test_dora_stays_depth_dependent_under_training():
    """The invariant that matters most, and the one that silently broke.

    A depth-indexed adapter is the whole premise of 5: W^(r) must differ across r,
    or every recurrence applies identical weights and depth buys nothing.  With
    unit_row_col_norm on B@A this died -- cos(W^(1), W^(32)) measured 0.58-0.74 at
    init and exactly 1.0000 after 60 steps, because the normalisation discards
    magnitude and l3 had drifted rank-1 along the scale axis it ignores.  Nothing
    in the loss curve showed it.

    Drive l3 to where training actually takes it and require r to still matter.
    """
    torch.manual_seed(0)
    c = Config(**TINY)
    e = ExpertBlock(c).double()
    h = e.hypers[0]

    def cos_across_r(std):
        with torch.no_grad():
            h.l3.weight.normal_(0, std)
        tab = h(e.base_norms())
        w0, w1 = (compose_dora(h, tab[r], e.dyn_up.weight, e.dyn_down.weight, c)[0]
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


def test_halting_bounds_are_respected():
    """Depth stays within [r_free, cap] and always lands on a commitment
    boundary -- the direct, per-token replacement for the old ExpertBlock.run()
    version of this check, now against RoutedBlock."""
    torch.manual_seed(0)
    c = Config(**TINY, r_free=2, turns=2, c_max2=16, grad_checkpoint=False)
    rb = RoutedBlock(c).eval()
    h = torch.randn(2, 6, c.d_model)
    with torch.no_grad():
        out, depth = rb(h)
    assert out.shape == h.shape
    assert depth.min() >= c.r_free and depth.max() <= c.c_max2
    assert torch.all((depth - c.r_free) % c.turns == 0), (
        "a commitment was interrupted mid-way")


def test_halted_tokens_stay_frozen():
    """A token whose depth < cap must be bit-identical to its state at that depth."""
    torch.manual_seed(0)
    c = Config(**TINY, r_free=2, turns=2, c_max2=16, grad_checkpoint=False)
    rb = RoutedBlock(c).double().eval()
    h = torch.randn(1, 8, c.d_model, dtype=torch.float64)
    with torch.no_grad():
        out, depth = rb(h)
        assert depth.min() < c.c_max2, "need at least one early halt to test freezing"
        # Re-run with a HALVED cap: any token that halted before the smaller cap
        # must produce the exact same output either way, since a shorter budget
        # cannot change what a token that stopped early already decided.
        half = int(depth.min())
        c_half = Config(**{**c.__dict__, "c_max2": half})
        rb_half = RoutedBlock(c_half).double().eval()
        rb_half.load_state_dict(rb.state_dict())
        out_half, depth_half = rb_half(h)
    frozen = depth <= half
    assert frozen.any(), "need at least one token that halted within the smaller cap"
    assert torch.allclose(out[frozen], out_half[frozen], atol=1e-10)


def test_hypernet_trunk_is_live_from_the_first_step():
    """B is randomly initialised (unit_row_col_norm keeps only direction, and
    zero has none), so the hypernet trunk must get real gradient immediately --
    no dead first step the way a zero-initialised B would cause."""
    torch.manual_seed(0)
    c = Config(**TINY, r_free=1, turns=1, c_max2=4, grad_checkpoint=False)
    rb = RoutedBlock(c)
    out, _ = rb(torch.randn(2, 8, c.d_model))
    out.square().mean().backward()
    for name in ("l1.weight", "l2.weight", "norm_proj.weight"):
        g = dict(rb.expert.hypers[0].named_parameters())[name].grad
        assert g is not None and g.any(), f"hypernet trunk {name} dead on step 0"


def test_tied_embeddings():
    m = CinnamonModel(Config(**TINY, tie_embeddings=True))
    assert m.head.weight is m.embed.weight
    m2 = CinnamonModel(Config(**TINY, tie_embeddings=False))
    assert m2.head.weight is not m2.embed.weight


def test_shared_parameters_excludes_routed_blocks():
    m = CinnamonModel(Config(**TINY))
    shared = {n for n, _ in m.shared_parameters()}
    assert not any(n.startswith(('e2.', 'e3.')) for n in shared)
    assert 'b1.subs.0.q.weight' in shared and 'embed.weight' in shared


def test_blocked_inverse_keeps_chunk_64_exact():
    """chunk=64 is only safe because inv_unit_lower does blocked forward
    substitution; the old repeated-squaring form overflowed fp32 here (-> inf).
    Pins the property that buys the 4x cut in the hot sequential loop.
    """
    from cinnamon.kda import KDA, _inv_squaring, inv_unit_lower

    torch.manual_seed(0)
    cfg = Config(**{**TINY, "kda_chunk": 64})
    k = KDA(cfg)
    with torch.no_grad():                      # beta -> 1, decay -> 0: the blow-up regime
        k.beta.bias.fill_(6.0)
        k.g_up.bias.fill_(-12.0)
    base = torch.randn(1, 1, TINY["d_model"])
    x = base.expand(2, 128, TINY["d_model"]) + 0.01 * torch.randn(2, 128, TINY["d_model"])
    out, ref = k(x), k.reference(x)
    assert torch.isfinite(out).all()
    assert (out - ref).norm() / ref.norm() < 1e-4

    # and the blocked form must be no worse than squaring on a well-conditioned case
    M = (torch.randn(4, 64, 64).tril(-1) * 0.5).double()
    I = torch.eye(64, dtype=M.dtype)
    err = lambda X: ((I + M) @ X - I).abs().max()
    assert err(inv_unit_lower(M)) <= err(_inv_squaring(M))

def test_default_config_hits_the_50m_budget():
    """The size target is 50 M EXCLUDING the tied embedding: the embedding is set
    by the tokenizer's 128 k vocab, not by a layer width, and at this scale it
    would otherwise eat 40% of the budget.

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
    e0 = m.e2.expert
    work = n([e0.kda, e0.mla, e0.fixed, e0.dyn_gate, e0.dyn_up, e0.dyn_down])
    assert len(e0.kda) == 2 and len(e0.fixed) == 2, "expert body must be 2:1 KDA:MLA"
    assert n([e0.hypers[0]]) < work, "hypernet outweighs the body it modulates"


def test_default_widths_actually_run():
    """TINY exercises the code, not the shipped dims.  d_model=224 with 7 heads of
    32 is exactly the kind of non-power-of-two split that breaks a reshape, and
    nothing else in the suite would catch it.  n_hypernets is cut to 2 only to keep
    the instantiation cheap -- every width parameter is the real one.
    """
    from cinnamon.model import CinnamonModel

    c = Config(n_hypernets=2, c_max2=4, kda_chunk=16)
    torch.manual_seed(0)
    m = CinnamonModel(c)
    ids = torch.randint(0, c.vocab, (1, 32))
    logits, loss, (d2, d3) = m(ids, labels=ids)
    assert logits.shape == (1, 32, c.vocab)
    assert torch.isfinite(loss) and loss > 0
    assert d2.shape == d3.shape == (1, 32)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.b1.parameters())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("\nall passed")



