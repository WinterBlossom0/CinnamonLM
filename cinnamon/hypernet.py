"""spec 5: hypernetwork generating the DoRA modification for the dynamic FFN.

    per hypernet:  [ phi(r) ; ctx ; norm_proj(||W_dyn||_c) ; err_best ; err_last ]
                       -> 256 -> code_dim                      (Hypernet)
    shared:            code_dim -> n_out                       (AdapterBasis)

Spec 5.5 content-blindness is DELIBERATELY overridden: `forward_token` takes a
per-token content vector and returns a distinct code per token, not one shared
row per r.  `forward` (zero content, whole r-table) is retained for tests that
inspect W^(r) directly.

No chaining (5.5) holds: each recurrence regenerates from the unchanged base, so
generated-weight error cannot accumulate across passes.

THE OUTPUT LAYER IS SHARED ACROSS THE BANK.  It was previously per-hypernet and
r_ceiling-wide, on the argument that a function of r over r_ceiling values has
rank <= r_ceiling at any width -- true while generation was content-blind, and
invalidated by per-token generation.  What it actually bought was 1.35 M private
parameters each (87.6% of a hypernet) spent on a rank-32 output space.

Sharing it is what makes the width affordable: one basis at code_dim=192 costs
less than eight at 32 and gives every hypernet 6x the rank.  It also means a
hypernet the router never selects no longer freezes 1.35 M parameters -- the
basis trains on every token regardless of routing, and only the ~230 k trunk
goes idle.  Specialisation lives in the trunk, which is the right place: every
hypernet adapts the SAME base weights, so a shared output basis is a prior, not
a compromise.

dora_linear applies the generated factors WITHOUT materialising W^(r); see its
docstring.  compose_dora builds it explicitly and is the test oracle only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .dora_kernel import dora_epilogue


class AdapterBasis(nn.Module):
    """code_dim -> the full DoRA vector.  ONE per block position, shared by that
    position's whole hypernetwork bank.

    Owns the output-side concerns: what the generated vector means (`sizes`,
    `unpack`) and how it is initialised (`reset_out`).
    """

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        r, d, f = c.rank, c.d_model, c.d_ff
        # A_up, B_up, m_up, A_down, B_down, m_down  (5.3)
        self.sizes = [(r, d), (f, r), (f,), (r, f), (d, r), (d,)]
        self.splits = [math.prod(s) for s in self.sizes]
        self.w = nn.Linear(c.code_dim, sum(self.splits), bias=False)
        self.reset_out()

    def forward(self, code):
        """[.., code_dim] -> [.., n_out].  Leading dims untouched."""
        return self.w(code)

    def generate(self, code):
        """code -> the six DoRA factors directly, skipping the [.., n_out] vector.

        Same arithmetic as unpack(forward(code)): `w.weight` is [n_out, code_dim],
        so splitting dim 0 partitions the OUTPUT features and every entry is still
        the same dot product over code_dim -- forward is bit-identical, and only
        d(code) reassociates (6 partial sums instead of one over n_out; 6.7e-16 in
        fp64).  What it buys is layout.  Through the joint vector every factor is a
        stride-n_out view of it, so each is copied contiguous again before its bmm
        and the whole 1.4 GB buffer is rebuilt by cat in backward; born separate
        they are already contiguous and neither happens.  +4.69% paired at batch 16.

        unpack/forward stay for compose_dora and the tests, which want the vector
        itself.
        """
        lead = code.shape[:-1]
        return [F.linear(code, w).view(*lead, *s) for w, s in
                zip(self.w.weight.split(self.splits, 0), self.sizes)]

    def q_basis(self, wns):
        """[T_up, T_down]: what turns a code straight into dora_linear's Q.

        A = (W_A @ code).view(r, in) and Q = A @ Wn^T, so

            Q[.., k, o] = sum_c code[.., c] * T[c, k, o]
            T[c, k, o]  = sum_i W_A[k*in + i, c] * Wn[o, i]

        T is a function of weights alone -- constant within a forward pass, ~1.3 M
        elements total -- so this replaces a per-recurrence [tokens, r, out] GEMM
        contracting over in (256 up, 1024 down) with one contracting over code_dim
        (64).  Measured at batch 16: 22.4% of a training step, against a 24.8%
        ceiling for dropping Q's GEMM outright.

        This is the AdapterBasis-layout coupling dora_linear's `q` argument exists
        to keep on THIS side of the wall: the block indices below are the only
        place the two modules meet, and compose_dora still checks the result
        without knowing about any of it.
        """
        r = self.c.rank
        out = []
        for i, Wn in zip((0, 3), wns):          # A_up, A_down
            o = sum(self.splits[:i])
            wa = self.w.weight[o:o + self.splits[i]].view(r, -1, self.c.code_dim)
            out.append(torch.einsum('kic,oi->cko', wa, Wn.to(wa.dtype)))
        return out

    def factors(self, code, wn):
        """code + ExpertBlock.dyn_base() -> stage_dyn's ((A,B,m,Q) up, ... down)."""
        A_u, B_u, m_u, A_d, B_d, m_d = self.generate(code)
        lead = code.shape[:-1]
        q_u, q_d = ((code @ T.flatten(1)).view(*lead, *T.shape[1:]) for _, T in wn)
        return (A_u, B_u, m_u, q_u), (A_d, B_d, m_d, q_d)

    def reset_out(self):
        """A and B random, m zero.

        B canNOT be zero-initialised any more.  unit_row_col_norm is scale
        invariant, so it keeps only B@A's direction -- and a zero matrix has no
        direction.  The forward survives on the eps guard, but backward divides by
        eps eight times over the four iterations, amplifying the gradient by
        (1/eps)^8 = 1e48, past float32's 3.4e38, so inf and then nan.  There is no
        patch for it: an exactly-identity init needs the update to be zero, and a
        unit-norm matrix is never zero.  The two requirements are incompatible.

        m stays zero, so the magnitude term ||W||*(1 + m) still starts at exactly
        ||W|| -- the generated weight begins with the base's magnitude and a
        bounded rotation, rather than the base exactly.

        std scales as 1/sqrt(code_dim): a generated A entry is one row of `w`
        dotted with the code, so its magnitude grows with the code width.  Without
        this, widening the bottleneck would silently multiply adapter strength at
        init (192 vs the historical 32 is 2.4x) and change behaviour for a reason
        unrelated to capacity.
        """
        std = _GEN_STD / math.sqrt(self.c.code_dim)
        with torch.no_grad():
            self.w.weight.zero_()
            o = 0
            for i, n in enumerate(self.splits):
                if i in (0, 1, 3, 4):                 # the two A and two B matrices
                    self.w.weight[o:o + n].normal_(0, std)
                o += n

    def unpack(self, vec):
        """Leading-dimension agnostic: [n_out] table row or [B,S,n_out] per-token
        batch, identically.  vec.shape[:-1] is () or (B,S).

        split, not six `vec[..., o:o+n]` slices.  Same views forward; the whole
        difference is backward.  Slicing gives autograd six INDEPENDENT paths back
        to `vec`, each of which zero-fills a full [tokens, n_out] buffer to hold one
        piece, and then sums the six -- at batch 16 that is 6 fills and 5 adds of a
        1.4 GB tensor per recurrence, 155 adds and 186 fills per step, 38% of it.
        split's backward is one `cat`: one buffer, the same six copies, no adds.
        Bit-identical, since the pieces are disjoint so every sum was x + 0.
        """
        lead = vec.shape[:-1]
        return [p.view(*lead, *s)
                for p, s in zip(vec.split(self.splits, -1), self.sizes)]


# Target std for a generated A/B entry, held FIXED across configs.  What matters
# physically is ||s*(BA)_j|| against ||Wn_j|| = 1, which is a property of the
# adapter, not of how wide the generator happens to be.
#
# The old init (std 0.02 at fan-in r_ceiling) did not hold that: it made small
# configs' adapters weaker purely because their r_ceiling was smaller, which was
# an accident of the shape, not a decision.  Value chosen to reproduce the DEFAULT
# config's previous scale (0.02 at fan-in 32); TINY/SMALL now init ~2x stronger
# than before, deliberately.
_GEN_STD = 0.02 * 32 ** 0.5


class Hypernet(nn.Module):
    """Per-hypernet trunk: inputs -> code_dim.  The bank's AdapterBasis turns the
    code into weights.  This module is where specialisation lives, and it is cheap
    (~230 k) precisely so the bank can be wide without the basis being duplicated.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        d, f = c.d_model, c.d_ff
        assert c.phi_dim % 2 == 0, f"phi_dim must be even for Fourier features, got {c.phi_dim}"
        self.norm_proj = nn.Linear(f + d, c.phi_dim, bias=False)
        self.ctx_proj = nn.Linear(d, c.phi_dim, bias=False)     # per-token content -> phi_dim
        # Five inputs: phi(r), per-token content, base column norms, and the two
        # halting errors -- best-so-far and last turn (see _encode_err).
        self.l1 = nn.Linear(5 * c.phi_dim, 256, bias=False)
        self.l2 = nn.Linear(256, c.code_dim, bias=False)
        # LEARNED step embedding, initialised to the sinusoid rather than replacing
        # it.  A fixed sinusoid is a fine prior on "these steps are ordered and
        # distinct", but nothing about depth-r semantics says the r-axis should
        # keep that exact geometry; letting it move costs r_ceiling*phi_dim
        # parameters and keeps the prior as the starting point.
        # Excluded from weight decay at the optimiser (see train.py): decaying it
        # toward zero would collapse the steps back onto each other.
        self.phi = nn.Parameter(self._sinusoid(c.r_ceiling, c.phi_dim))
        # Halting-error inputs.  fixed_point_residual is a per-token scalar in ~[0, 1]; a raw
        # scalar into a 5*phi_dim concat would be one weight against hundreds and
        # effectively ignored, so it is expanded to phi_dim Fourier features --
        # the same reason phi(r) is sinusoidal rather than an integer.
        # Geometric frequencies 1..1e3 put most resolution near 0, which is where
        # the errors live once a token starts converging.
        self.register_buffer("err_freq", torch.exp(torch.linspace(
            0.0, math.log(1000.0), c.phi_dim // 2)), persistent=False)
        # Learned tag per slot, so "best so far" and "last turn" are distinguishable
        # after encoding.  Without it the two share one encoder and arrive as two
        # interchangeable vectors the model cannot tell apart.
        self.err_slot = nn.Parameter(torch.zeros(2, c.phi_dim))

    def _encode_err(self, x, slot):
        """Per-token scalar -> [..., phi_dim].  slot 0 = best so far, 1 = last turn."""
        e = x.unsqueeze(-1) * self.err_freq.to(x.dtype)
        return torch.cat([e.sin(), e.cos()], -1) + self.err_slot[slot].to(x.dtype)

    @staticmethod
    def _sinusoid(n_r, dim):
        r = torch.arange(1, n_r + 1, dtype=torch.float32).unsqueeze(1)
        w = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        e = torch.zeros(n_r, dim)
        e[:, 0::2], e[:, 1::2] = torch.sin(r * w), torch.cos(r * w)
        return e

    def forward(self, base_norms: torch.Tensor) -> torch.Tensor:
        """Content-blind, whole r-table at once.  Returns CODES [r_ceiling, code_dim];
        apply the block's AdapterBasis to get weights.

        Zero content and zero errors through the same trunk, so one set of weights
        serves this and forward_token.  Used by tests that inspect W^(r) directly.
        """
        R = self.c.r_ceiling
        u = self.norm_proj(base_norms).expand(R, -1)
        zero = u.new_zeros(R, self.c.phi_dim)
        z0 = u.new_zeros(R)
        z = torch.cat([self.phi.to(u.dtype), zero, u,
                       self._encode_err(z0, 0), self._encode_err(z0, 1)], -1)
        return F.silu(self.l2(F.silu(self.l1(z))))

    def forward_token(self, r: int, content: torch.Tensor, base_norms: torch.Tensor,
                      err_best=None, err_last=None) -> torch.Tensor:
        """Per-token.  content: [B, S, d_model] -> [B, S, code_dim], a distinct code
        per token, recomputed every recurrence.  Apply the block's AdapterBasis to
        turn the code into DoRA factors.

        err_best / err_last: per-token halting errors [B, S] as known BEFORE this
        recurrence -- the running minimum and the previous turn's reading.  They
        tell the generator how far this token still is from converged and whether
        it is still improving, which `content` alone does not say: two tokens can
        sit at the same state with very different rates of change.  Detached (the
        halting path carries no gradient), and zero when no reading exists yet.
        """
        u_norms = self.norm_proj(base_norms)                    # [phi_dim]
        u_ctx = self.ctx_proj(content)                           # [B, S, phi_dim]
        phi_r = self.phi[min(r, self.c.r_ceiling) - 1].to(u_ctx.dtype)   # [phi_dim]
        z0 = u_ctx.new_zeros(u_ctx.shape[:-1])
        eb = self._encode_err(z0 if err_best is None else err_best.to(u_ctx.dtype), 0)
        el = self._encode_err(z0 if err_last is None else err_last.to(u_ctx.dtype), 1)
        z = torch.cat([phi_r.expand_as(u_ctx), u_ctx,
                       u_norms.expand_as(u_ctx), eb, el], -1)
        return F.silu(self.l2(F.silu(self.l1(z))))


def stack_bank(hypers):
    """Stack a bank's TRUNK weights into [n, ...] tensors for batched dispatch.

    All hypernets share shapes, so the whole bank is one set of batched operands.
    The output basis is not here -- it is shared, so it stays a single matmul
    applied after the bmms rather than an [n, n_out, code_dim] stack.  That used to
    be ~43 MB restacked per forward and is now the largest operand NOT batched,
    which is the cheaper arrangement as well as the smaller one.
    """
    return {
        "norm": torch.stack([h.norm_proj.weight for h in hypers]),   # [n, phi, f+d]
        "ctx": torch.stack([h.ctx_proj.weight for h in hypers]),     # [n, phi, d]
        "l1": torch.stack([h.l1.weight for h in hypers]),            # [n, 256, 5phi]
        "l2": torch.stack([h.l2.weight for h in hypers]),            # [n, code, 256]
        "phi": torch.stack([h.phi for h in hypers]),                 # [n, rc, phi]
        "slot": torch.stack([h.err_slot for h in hypers]),           # [n, 2, phi]
        "freq": hypers[0].err_freq,                                  # deterministic
    }


def forward_token_bank(W, r, ctx, base_norms, r_ceiling, err_best=None, err_last=None):
    """forward_token for the WHOLE bank at once.  ctx: [n, K, d] -> [n, K, code_dim].

    Returns CODES; the caller applies the shared AdapterBasis once to the whole
    result, which is one big matmul rather than n batched ones.

    Identical arithmetic to Hypernet.forward_token, reassociated so each layer is
    one bmm over n groups instead of n separate matmuls.  The loop form launches
    ~15-20 tiny kernels per hypernet per recurrence and spends 41% of inference
    waiting on launches; this launches once per layer.

    Cost is padding: K is the largest group, so total work is n*K rather than the
    sum of group sizes.  Worth it only because this path is launch-bound, not
    FLOP-bound -- if the router collapses onto one hypernet, K approaches the full
    token count and this becomes strictly worse than the loop.
    """
    u_n = torch.einsum('npd,d->np', W["norm"].to(ctx.dtype), base_norms.to(ctx.dtype))
    u_c = torch.bmm(ctx, W["ctx"].to(ctx.dtype).transpose(1, 2))          # [n,K,phi]
    p_r = W["phi"].to(ctx.dtype)[:, min(r, r_ceiling) - 1]                # [n, phi]

    def enc(x, slot):                       # mirrors Hypernet._encode_err
        x = u_c.new_zeros(u_c.shape[:-1]) if x is None else x.to(u_c.dtype)
        e = x.unsqueeze(-1) * W["freq"].to(u_c.dtype)
        return torch.cat([e.sin(), e.cos()], -1) + \
            W["slot"].to(u_c.dtype)[:, slot].unsqueeze(1)

    z = torch.cat([p_r.unsqueeze(1).expand_as(u_c), u_c,
                   u_n.unsqueeze(1).expand_as(u_c),
                   enc(err_best, 0), enc(err_last, 1)], -1)               # [n,K,5phi]
    z = F.silu(torch.bmm(z, W["l1"].to(z.dtype).transpose(1, 2)))
    return F.silu(torch.bmm(z, W["l2"].to(z.dtype).transpose(1, 2)))      # [n,K,code]


def _gmm(x, W, offs):
    """[M, in] x [n, out, in] -> [M, out], row j using the weight of the segment it
    falls in.  `offs` is the int32 cumulative END of each segment, so segments are
    contiguous and MAY DIFFER IN SIZE -- no padding, unlike the bmm path.

    torch._grouped_mm is a CUTLASS grouped GEMM; verified bit-identical to the
    per-segment matmul below on sm_120.  The loop is the real fallback (CPU, fp64)
    and reads the offsets on the host, which the grouped path does not.
    """
    Wt = W.transpose(-1, -2)
    if x.is_cuda and x.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch._grouped_mm(x, Wt, offs=offs)
    out = x.new_empty(x.shape[0], Wt.shape[-1])
    s = 0
    for i, e in enumerate(offs.tolist()):
        if e > s:
            out[s:e] = x[s:e] @ Wt[i]
        s = e
    return out


def forward_token_grouped(W, r, ctx, base_norms, r_ceiling, seg, offs,
                          err_best=None, err_last=None):
    """forward_token for the whole bank on SEGMENT-SORTED tokens.

    ctx: [M, d] with rows already grouped by owning hypernet; `seg` [M] is each
    row's hypernet index and `offs` the segment ends.  Returns CODES [M, code_dim];
    the caller applies the shared AdapterBasis and the DoRA FFN ONCE to the whole
    result, which is the point -- those are hypernet-independent and were being
    re-launched n times per recurrence by the loop.

    Identical arithmetic to Hypernet.forward_token, reassociated: each trunk layer
    is one grouped GEMM over the bank instead of n separate matmuls, and every
    per-hypernet constant arrives by gather rather than by being a separate call.
    Unlike forward_token_bank there is NO padding, so total work is the sum of the
    group sizes however uneven the router's split is.
    """
    dt = ctx.dtype
    u_n = W["norm"].to(dt) @ base_norms.to(dt)                    # [n, phi]
    u_c = _gmm(ctx, W["ctx"].to(dt), offs)                        # [M, phi]
    p_r = W["phi"].to(dt)[:, min(r, r_ceiling) - 1]                # [n, phi]
    slot = W["slot"].to(dt)
    # ONE gather, not four.  p_r, u_n and the two err slots are all per-SEGMENT
    # constants read by the same `seg`, and index-backward on an [n, phi] operand
    # is essentially all fixed overhead -- measured, [8,256] costs 3% more than
    # [8,64], so four separate ones cost 4x one.  Same values, so bit-identical.
    ph = p_r.shape[-1]
    cs = torch.cat([p_r, u_n, slot[:, 0], slot[:, 1]], -1)[seg].split(ph, -1)

    def enc(x, i):                          # mirrors Hypernet._encode_err
        x = u_c.new_zeros(u_c.shape[0]) if x is None else x.to(dt)
        e = x.unsqueeze(-1) * W["freq"].to(dt)
        return torch.cat([e.sin(), e.cos()], -1) + cs[2 + i]

    z = torch.cat([cs[0], u_c, cs[1], enc(err_best, 0), enc(err_last, 1)], -1)
    z = F.silu(_gmm(z, W["l1"].to(z.dtype), offs))
    return F.silu(_gmm(z, W["l2"].to(z.dtype), offs))              # [M, code]


def unit_row_col_norm(M, iters: int = 4, eps: float = 1e-6):
    """Rows sum-of-squares 1, columns m/n (equal total mass m).  L2 analogue of
    Sinkhorn-Knopp; ends on rows so that constraint is exact.

    Applied to the BASE weights, not the generated update -- see compose_dora.

    Why constrained at all: the basis starts at zero but nothing keeps it there.
    Measured, a selected hypernetwork's table grew norm 0.9 -> 704 in 30 steps;
    once ||B@A|| >> ||W|| the update dominates V = W + s*B@A and the base is gone.
    DoRA's own normalisation hides this -- it pins ||W^(r)|| to ||W||, so
    magnitude looked correct (ratio 0.987) while direction rotated to cosine 0.027,
    i.e. orthogonal to the weight being adapted.

    Why L2 and not doubly-stochastic: negatives must survive (an adapter must
    subtract), and entries must be bounded.  ||row||_2 = 1 gives both.  The
    generalized doubly-stochastic form bounds only sums, so a row could read
    (+1000, -999) -- measured cosine -0.005 to base, i.e. unrelated.

    Zero maps to zero via the eps guard, which keeps the adapter an exact identity
    at init (reset_out zeroes the basis).  A true projection is undefined there.
    """
    # sqrt(sum x^2 + eps^2), never norm(x) + eps: the latter is exactly 0 at the
    # origin where d||x||/dx = 0/0 -> nan backward.  The basis is zero-init, so step 1
    # lands exactly there.  Forward looked fine; every gradient was nan.
    def nrm(P, dim):
        return (P.pow(2).sum(dim, keepdim=True) + eps * eps).sqrt()

    col_target = (M.shape[-2] / M.shape[-1]) ** 0.5
    P = M
    for _ in range(iters):
        P = P / nrm(P, -1)                        # rows  -> 1
        P = P * col_target / nrm(P, -2)           # cols  -> sqrt(m/n)
    return P / nrm(P, -1)                         # finish exact on rows


class _AxGram(torch.autograd.Function):
    """(A x, A A^T) from one node, so A has ONE gradient path instead of three.

    Same values as the two einsums it replaces; what changes is how autograd sees
    them.  A is [.., r, in] -- 537 MB for A_down at batch 16 -- and every extra
    path through it costs a full write of that plus an add:

        A A^T with A passed twice     two [.., r, r] x [.., r, in] bmms, added:
                                      dG @ A + dG^T @ A.  Here it is
                                      (dG + dG^T) @ A -- one bmm on the [.., 16, 16]
                                      sum, exact in exact arithmetic and a single
                                      accumulation instead of two summed after.
        A x as a separate einsum      a third dA, so aten adds two 537 MB tensors.
                                      Here it is an addcmul_ straight onto the
                                      bmm's output: a rank-1 update, no operand
                                      materialised, no add.

    fp64 against the einsum pair: forward bit-identical, gradients ~3e-16
    relative.  bf16, M=16384, fwd+bwd per recurrence: 2.40 -> 0.92 ms (up) and
    10.72 -> 3.74 ms (down) for the Gram alone; the merge takes another add of
    each size out of the backward.
    """

    @staticmethod
    def forward(ctx, A, x):
        ctx.save_for_backward(A, x)
        return (A @ x.unsqueeze(-1)).squeeze(-1), A @ A.transpose(-1, -2)

    @staticmethod
    def backward(ctx, dAx, dG):
        A, x = ctx.saved_tensors
        # .to(): no-op in every path that occurs (both grads come back in A's
        # dtype); here so a mixed-dtype caller cannot fail inside the bmm.
        dA = (dG + dG.transpose(-1, -2)).to(A.dtype) @ A
        dA.addcmul_(dAx.unsqueeze(-1).to(A.dtype), x.unsqueeze(-2))
        dx = (A.transpose(-1, -2) @ dAx.unsqueeze(-1).to(A.dtype)).squeeze(-1)
        return dA, dx


def dora_linear(x, W, A, B, m, c: Config, wn=None, q=None):
    """y = W^(r) x without materialising W^(r).

    Materialising costs ~1 MB/token/projection (d_ff=1024, d_model=256) = ~536 MB
    per 512-token batch; that is what put peak VRAM at 14.95 GB.

    Value factorises directly:   V x = Wn x + s*B(A x),  widest intermediate [..,r]

    DoRA's row norm ||V_j|| appears to need the assembled matrix.  It does not:

        ||V_j||^2 = ||Wn_j||^2 + 2s<Wn_j,(BA)_j> + s^2||(BA)_j||^2
        ||(BA)_j||^2   = B_j^T (A A^T) B_j            via G = A A^T   [r, r]
        <Wn_j,(BA)_j>  = sum_k B[j,k] (A Wn^T)[k,j]   via Q = A Wn^T  [r, out]

    Widest tensor [.., out, in] -> [.., out, r]: rank replaces d_model, ~24x less
    memory at r=16.  Identical to compose_dora + matmul; asserted to 1e-10 (fp64)
    by test_dora_factorised_matches_reference, which is why compose_dora is kept.

    `wn` is unit_row_col_norm(W) precomputed by the caller.  It depends on nothing
    but W, and this runs ~500 times per forward pass on the same two matrices, so
    the caller hoists it out of the recurrence (see RoutedBlock.forward).

    The two ROW STATISTICS below (Wn.pow(2).sum(-1) and W.norm(dim=-1)) are also
    pure functions of W and are also re-derived every call.  Hoisting them the same
    way was tried and does NOT pay: measured against this exact body, interleaved
    on a fixed batch over 9 rounds at batch 16 / seq 1024, -0.89% median and -1.78%
    paired, with the forward bit-identical and 266/268 gradients bit-identical.
    They are cheap reductions over [out, in] whose cost is already hidden behind
    the GEMMs; hoisting only turns them into long-lived checkpoint inputs.
    """
    s = c.dora_scale
    Wn = unit_row_col_norm(W) if wn is None else wn      # [out, in], shared
    # A/B/m keep their own dtype.  Every consumer re-derives its precision anyway
    # -- the einsums are autocast-downcast to bf16 whatever they are handed, and
    # dora_epilogue upcasts on load -- so casting them to x.dtype was ~53 MB of
    # dead copy per call.  It did buy fp32 accumulation of A's three and B's two
    # gradient contributions; measured against an fp64 reference that is worth
    # 0.18% of dA's error, both forms sitting at 3.72e-3, the bf16 gradient's own
    # quantisation.  Outside autocast they already ARE x.dtype.
    Wn = Wn.to(x.dtype)

    # Cast x HERE, not inside each consumer.  Under autocast all of them downcast
    # it to A's dtype anyway (A comes out of an autocast linear, so it is already
    # there), so this is the same arithmetic with one cast node instead of three,
    # and it lets _AxGram see one dtype on both operands.  Outside autocast it is
    # a no-op: A and x already agree.
    xa = x.to(A.dtype)

    # value: never wider than [.., r].  Ax and G share one node so A has a single
    # gradient path -- see _AxGram.
    Ax, G = _AxGram.apply(A, xa)                         # [.., r], [.., r, r]
    # val = F.linear(xa, Wn) + s*(B @ Ax) is NOT assembled here: the epilogue
    # kernel already holds B in registers, so it adds the low-rank term itself and
    # B keeps a single read and a single gradient.
    # `q` is the same tensor reached through the code rather than through A --
    # see AdapterBasis.q_basis, which is where the layout knowledge lives.  Same
    # value up to reassociation; this form contracts over in.
    Q = torch.einsum('...ri,io->...ro', A, Wn.t()) if q is None else q
    return dora_epilogue(F.linear(xa, Wn), Ax, m, B, G, Q, Wn.pow(2).sum(-1),
                         W.norm(dim=-1).to(x.dtype), s, c.eps)


def compose_dora(basis: AdapterBasis, vec, W_up, W_down, c: Config):
    """REFERENCE IMPLEMENTATION.  dora_linear above is what actually runs; this
    builds the matrix explicitly and exists to keep that optimisation honest --
    test_dora_factorised_matches_reference checks the two agree.  Also used by the
    depth-dependence tests, which inspect W^(r) itself rather than its action.

    5.4  V = W + (alpha/rank) DS(B A) ;  W^(r) = m * V / (||V||_c + eps).

    m is reparameterised as ||W||_c * (1 + m_raw) so that m_raw = 0 reproduces W
    exactly.  ||W||_c tracks the base as the optimiser moves it, which is the same
    reason the column norms are fed in as an input (5.1).

    Batch-dimension agnostic (dim=-1 throughout): composes a single [out,in] from
    a [n_out] vec, or [B,S,out,in] from [B,S,n_out].  W stays [out,in] (shared
    base); broadcasting handles the rest.
    """
    A_u, B_u, m_u, A_d, B_d, m_d = basis.unpack(vec)
    out = []
    for W, A, B, m in ((W_up, A_u, B_u, m_u), (W_down, A_d, B_d, m_d)):
        # Constraint on the BASE, not the update.  Constraining B@A instead
        # destroyed depth adaptation: unit_row_col_norm is scale-invariant, so the
        # basis drifts freely along the scale axis nothing penalises; once it is
        # effectively rank-1 every r maps to the same direction, and normalising
        # discards magnitude, the only remaining separator.  Measured:
        # cos(W^(1), W^(32)) 0.58-0.74 at init -> exactly 1.0000 after 60 steps,
        # i.e. the dynamic FFN became static.  Raw B@A keeps direction AND
        # magnitude varying with r, which is what makes the adapter depth-indexed.
        Wn = unit_row_col_norm(W)
        BA = B.to(W.dtype) @ A.to(W.dtype)
        V = Wn + c.dora_scale * BA
        # Magnitude from the RAW base, not the normalised one: lets the FFN learn
        # its own scale and keeps base_norms() (5.1) a live signal, not ones.
        mag = W.norm(dim=-1, keepdim=True) * (1.0 + m.to(W.dtype).unsqueeze(-1))
        out.append(mag * V / (V.norm(dim=-1, keepdim=True) + c.eps))
    return out
