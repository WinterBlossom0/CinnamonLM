"""5 Expert hypernetwork - generates the DoRA modification for the dynamic FFN.

    [ phi(r) 64 ; ctx(h) 64 ; proj(||W_dyn||_c) 64 ]  ->  256 -> 32 -> 84480

Content-blind (5.5) is DELIBERATELY overridden here, per direct instruction:
generation is now genuinely per-token.  `forward_token` takes a per-token content
vector (from RoutedBlock's ctx_kda, run over the actual sequence) alongside
phi(r) and the base norms, and returns a DISTINCT code per token -- there is no
longer one shared table row per (r, bucket); every token gets its own.

`forward` (no per-token content) still exists for Phase 1, which has no content
pathway; it feeds a zero content vector so the same trunk (l1/l2/l3) serves both.

No chaining (5.5) still holds: every recurrence regenerates from the unchanged
base, so generated-weight error cannot accumulate across recursive passes.

The 32-wide r-bottleneck is not a tuning choice: a function of r over 32 possible
values has rank <= 32 whatever the network width, so wider is provably unusable.
Content is a separate input width (phi_dim) and is not subject to that bound.

Cost, stated plainly: a genuinely distinct weight per token means the composed
[d_ff, d_model] matrices are materialised per token, not shared.  At B*S=512,
d_ff=1024, d_model=256, one matrix (up or down) is ~2 MB/token -> ~1 GB for the
batch, per hypernetwork that is actually invoked.  compose_dora is written to be
batch-dimension agnostic (dim=-1 throughout) so the SAME code composes either a
single [n_out] table row (Phase 1) or a [B,S,n_out] per-token batch (Phase 2).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Hypernet(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        r, d, f = c.rank, c.d_model, c.d_ff
        # A_up, B_up, m_up, A_down, B_down, m_down  (5.3)
        self.sizes = [(r, d), (f, r), (f,), (r, f), (d, r), (d,)]
        self.splits = [math.prod(s) for s in self.sizes]
        self.norm_proj = nn.Linear(f + d, c.phi_dim, bias=False)
        self.ctx_proj = nn.Linear(d, c.phi_dim, bias=False)     # per-token content -> phi_dim
        # Three inputs: phi(r), the (per-token) content projection, and the base
        # column norms.
        self.l1 = nn.Linear(3 * c.phi_dim, 256, bias=False)
        self.l2 = nn.Linear(256, c.r_ceiling, bias=False)
        self.l3 = nn.Linear(c.r_ceiling, sum(self.splits), bias=False)
        # LEARNED step embedding, initialised to the sinusoid rather than replacing
        # it.  A fixed sinusoid is a fine prior on "these steps are ordered and
        # distinct", but nothing about depth-r semantics says the r-axis should
        # keep that exact geometry; letting it move costs r_ceiling*phi_dim
        # parameters and keeps the prior as the starting point.
        # Excluded from weight decay at the optimiser (see train.py): decaying it
        # toward zero would collapse the steps back onto each other.
        self.phi = nn.Parameter(self._sinusoid(c.r_ceiling, c.phi_dim))
        self.reset_out()

    @staticmethod
    def _sinusoid(n_r, dim):
        r = torch.arange(1, n_r + 1, dtype=torch.float32).unsqueeze(1)
        w = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        e = torch.zeros(n_r, dim)
        e[:, 0::2], e[:, 1::2] = torch.sin(r * w), torch.cos(r * w)
        return e

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
        """
        with torch.no_grad():
            self.l3.weight.zero_()
            o = 0
            for i, n in enumerate(self.splits):
                if i in (0, 1, 3, 4):                 # the two A and two B matrices
                    self.l3.weight[o:o + n].normal_(0, 0.02)
                o += n

    def forward(self, base_norms: torch.Tensor) -> torch.Tensor:
        """Phase 1: content-blind, whole r-table at once.  Returns [r_ceiling, n_out].

        No content pathway exists in Phase 1 (no ctx_kda, no per-token loop), so
        this feeds a zero content vector through the same trunk `forward_token`
        uses -- one set of weights serves both, rather than duplicating l1/l2/l3.
        """
        R = self.c.r_ceiling
        u = self.norm_proj(base_norms).expand(R, -1)
        zero_ctx = u.new_zeros(R, self.c.phi_dim)
        z = torch.cat([self.phi.to(u.dtype), zero_ctx, u], -1)
        return self.l3(F.silu(self.l2(F.silu(self.l1(z)))))

    def forward_token(self, r: int, content: torch.Tensor,
                      base_norms: torch.Tensor) -> torch.Tensor:
        """Phase 2: genuinely per-token.  content: [B, S, d_model] (ctx_kda's
        output).  Returns [B, S, n_out] -- a distinct generated code per token,
        recomputed every recurrence since content changes every recurrence.
        """
        u_norms = self.norm_proj(base_norms)                    # [phi_dim]
        u_ctx = self.ctx_proj(content)                           # [B, S, phi_dim]
        phi_r = self.phi[min(r, self.c.r_ceiling) - 1].to(u_ctx.dtype)   # [phi_dim]
        z = torch.cat([phi_r.expand_as(u_ctx), u_ctx, u_norms.expand_as(u_ctx)], -1)
        return self.l3(F.silu(self.l2(F.silu(self.l1(z)))))

    def unpack(self, vec):
        """Leading-dimension agnostic: works for a single [n_out] table row
        (Phase 1) and a [B, S, n_out] per-token batch (Phase 2) identically --
        vec.shape[:-1] is () for the former, (B, S) for the latter."""
        out, o = [], 0
        lead = vec.shape[:-1]
        for s, n in zip(self.sizes, self.splits):
            out.append(vec[..., o:o + n].view(*lead, *s))
            o += n
        return out


def unit_row_col_norm(M, iters: int = 4, eps: float = 1e-6):
    """Normalise a matrix so every row and column has sum-of-squares 1.

    Applied to the dynamic FFN's BASE weights, not to the generated update -- see
    compose_dora for why putting it on the update killed the depth adaptation.

    This is the constraint the adapter needs, and it does two jobs the L1 versions
    could not do together:

      * negatives survive.  An adapter must be able to subtract, so the ordinary
        (non-negative) doubly stochastic form is wrong here.
      * it cannot blow up.  With ||row||_2 = 1 every entry is bounded by 1 -- an
        element can never exceed the norm of the row containing it.  The
        *generalized* doubly stochastic form allowed negatives but bounded only
        the sums, so a row could still read (+1000, -999); measured, it left the
        generated weight at cosine -0.005 to its base, i.e. unrelated.

    Rectangular reconciliation, as before: rows target 1, columns target m/n, so
    both give a total squared mass of m.  Alternating normalisation (the L2
    analogue of Sinkhorn-Knopp) converges quickly; the loop ends on rows so that
    constraint holds exactly and columns hold to within the iteration tolerance.

    Zero maps to zero, because of the eps guard rather than by construction -- and
    that is what keeps the adapter an exact identity at init, since reset_out
    zeroes l3.  A true projection would be undefined there (every unit-norm matrix
    is equidistant from 0); mapping it to 0 is both well-defined and what the
    initialisation wants.

    Why it is here: without a constraint the generated update is unbounded.  l3
    starts at zero but nothing keeps it there -- measured, a selected
    hypernetwork's table grew from norm 0.9 to 704 in 30 steps, and once
    ||B@A|| >> ||W|| the term dominates V = W + scale*B@A and the base weight is
    gone.  DoRA's own normalisation hides that completely: it pins ||W^(r)|| to
    ||W||, so the magnitude looked perfect (ratio 0.987) while the direction had
    rotated to cosine 0.027 -- orthogonal to the weight it was meant to adapt.
    """
    # sqrt(sum x^2 + eps^2), never norm(x) + eps: the latter is exactly zero at the
    # origin, where d||x||/dx = x/||x|| is 0/0, so backward returns nan.  l3 is
    # zero-initialised, so the very first step lands on precisely that point --
    # the forward looked fine and every gradient was nan.
    def nrm(P, dim):
        return (P.pow(2).sum(dim, keepdim=True) + eps * eps).sqrt()

    col_target = (M.shape[-2] / M.shape[-1]) ** 0.5
    P = M
    for _ in range(iters):
        P = P / nrm(P, -1)                        # rows  -> 1
        P = P * col_target / nrm(P, -2)           # cols  -> sqrt(m/n)
    return P / nrm(P, -1)                         # finish exact on rows


def dora_linear(x, W, A, B, m, c: Config):
    """y = W^(r) x, computed WITHOUT ever materialising W^(r).

    compose_dora below is the reference: it builds V = Wn + s*B@A explicitly, an
    [out, in] matrix PER TOKEN.  At d_ff=1024, d_model=256 that is ~1 MB a token,
    ~536 MB for a 512-token batch across both projections -- which is what put
    peak VRAM at 14.95 GB and made gradient checkpointing non-optional.

    The value is trivially factorable:

        V x = Wn x + s * B (A x)          largest intermediate [.., r]

    The obstacle is DoRA's row normalisation, which needs ||V_j|| -- seemingly a
    property of the assembled matrix.  It is not.  Expanding the square:

        ||V_j||^2 = ||Wn_j||^2 + 2s <Wn_j, (BA)_j> + s^2 ||(BA)_j||^2

    and every term is reachable from the factors alone, with (BA)_j = A^T B_j:

        ||(BA)_j||^2 = B_j^T (A A^T) B_j          via G = A A^T, [r, r]
        <Wn_j, (BA)_j> = sum_k B[j,k] (A Wn^T)[k,j]   via Q = A Wn^T, [r, out]

    So the widest tensor drops from [.., out, in] to [.., out, r] -- rank replaces
    d_model, ~24x less memory at r=16.  Mathematically identical to compose_dora
    followed by a matmul; test_dora_factorised_matches_reference asserts exactly
    that, and is the reason compose_dora is kept rather than deleted.
    """
    s = c.dora_scale
    Wn = unit_row_col_norm(W)                            # [out, in], shared
    A, B, m = A.to(x.dtype), B.to(x.dtype), m.to(x.dtype)
    Wn = Wn.to(x.dtype)

    # value: never wider than [.., r]
    Ax = torch.einsum('...ri,...i->...r', A, x)
    val = F.linear(x, Wn) + s * torch.einsum('...or,...r->...o', B, Ax)

    # ||V_j||, closed form from the factors
    G = torch.einsum('...ri,...qi->...rq', A, A)         # [.., r, r]
    quad = (torch.einsum('...or,...rq->...oq', B, G) * B).sum(-1)
    Q = torch.einsum('...ri,io->...ro', A, Wn.t())       # [.., r, out]
    cross = torch.einsum('...or,...ro->...o', B, Q)
    vn2 = Wn.pow(2).sum(-1) + 2.0 * s * cross + s * s * quad
    # clamp before sqrt: the three terms are exact in fp32 but can land a hair
    # below zero in bf16, and sqrt of a negative is nan with no other warning.
    vn = vn2.clamp_min(0).sqrt()

    mag = W.norm(dim=-1).to(x.dtype) * (1.0 + m)         # [.., out]
    return mag * val / (vn + c.eps)


def compose_dora(hyper: Hypernet, vec, W_up, W_down, c: Config):
    """REFERENCE IMPLEMENTATION.  dora_linear above is what actually runs; this
    builds the matrix explicitly and exists to keep that optimisation honest --
    test_dora_factorised_matches_reference checks the two agree.  Also used by the
    depth-dependence tests, which inspect W^(r) itself rather than its action.

    5.4  V = W + (alpha/rank) DS(B A) ;  W^(r) = m * V / (||V||_c + eps).

    m is reparameterised as ||W||_c * (1 + m_raw) so that m_raw = 0 reproduces W
    exactly.  ||W||_c tracks the base as the optimiser moves it, which is the same
    reason the column norms are fed in as an input (5.1).

    Batch-dimension agnostic throughout (dim=-1, never a hardcoded axis index), so
    the same function composes a single [out,in] matrix from a [n_out] vec (Phase
    1's table lookup) or a per-token BATCH of [B,S,out,in] matrices from a
    [B,S,n_out] vec (Phase 2's per-token generation) -- W stays [out,in] either
    way (the shared base), and broadcasting does the rest: `@` batches over any
    leading dims A/B carry, and `W` broadcasts against a batched V unmodified.
    """
    A_u, B_u, m_u, A_d, B_d, m_d = hyper.unpack(vec)
    out = []
    for W, A, B, m in ((W_up, A_u, B_u, m_u), (W_down, A_d, B_d, m_d)):
        # The constraint sits on the BASE, not on the update.
        #
        # Constraining B@A instead was measured to destroy the depth adaptation
        # outright: unit_row_col_norm is scale invariant, l3 drifts freely along
        # the scale axis it ignores (nothing penalises a direction the output does
        # not see), and once l3 is effectively rank-1 every r's code maps to the
        # same output direction.  The normalisation then discards the only thing
        # still separating them -- magnitude -- and W^(r) comes out IDENTICAL for
        # every r.  Measured: cos(W^(1), W^(32)) went 0.58-0.74 at init to exactly
        # 1.0000 after 60 steps, i.e. the "dynamic" FFN had become static.
        #
        # Left raw, B@A keeps both its direction and its magnitude varying with r,
        # which is what makes the adapter depth-indexed at all (5).  The base is
        # normalised instead: that is what keeps the two terms on a comparable
        # scale and stops V being dominated by whichever drifted larger.
        Wn = unit_row_col_norm(W)
        BA = B.to(W.dtype) @ A.to(W.dtype)
        V = Wn + c.dora_scale * BA
        # Magnitude still comes from the RAW base, not the normalised one: this is
        # what lets the FFN learn its own scale, and it keeps base_norms() (5.1)
        # a live signal instead of a constant vector of ones.
        mag = W.norm(dim=-1, keepdim=True) * (1.0 + m.to(W.dtype).unsqueeze(-1))
        out.append(mag * V / (V.norm(dim=-1, keepdim=True) + c.eps))
    return out
