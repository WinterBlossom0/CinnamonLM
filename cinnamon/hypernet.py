"""spec 5: hypernetwork generating the DoRA modification for the dynamic FFN.

    [ phi(r) ; ctx ; norm_proj(||W_dyn||_c) ]  ->  256 -> r_ceiling -> n_out

Spec 5.5 content-blindness is DELIBERATELY overridden: `forward_token` takes a
per-token content vector and returns a distinct code per token, not one shared
row per r.  `forward` (zero content, whole r-table) is retained for tests that
inspect W^(r) directly.

No chaining (5.5) holds: each recurrence regenerates from the unchanged base, so
generated-weight error cannot accumulate across passes.

r_ceiling-wide bottleneck is forced, not tuned: a function of r over r_ceiling
values has rank <= r_ceiling at any width.  Content enters on a separate width
(phi_dim) and is not subject to that bound.

dora_linear applies the generated factors WITHOUT materialising W^(r); see its
docstring.  compose_dora builds it explicitly and is the test oracle only.
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
        """Leading-dimension agnostic: [n_out] table row or [B,S,n_out] per-token
        batch, identically.  vec.shape[:-1] is () or (B,S)."""
        out, o = [], 0
        lead = vec.shape[:-1]
        for s, n in zip(self.sizes, self.splits):
            out.append(vec[..., o:o + n].view(*lead, *s))
            o += n
        return out


def unit_row_col_norm(M, iters: int = 4, eps: float = 1e-6):
    """Rows sum-of-squares 1, columns m/n (equal total mass m).  L2 analogue of
    Sinkhorn-Knopp; ends on rows so that constraint is exact.

    Applied to the BASE weights, not the generated update -- see compose_dora.

    Why constrained at all: l3 starts at zero but nothing keeps it there.
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
    at init (reset_out zeroes l3).  A true projection is undefined there.
    """
    # sqrt(sum x^2 + eps^2), never norm(x) + eps: the latter is exactly 0 at the
    # origin where d||x||/dx = 0/0 -> nan backward.  l3 is zero-init, so step 1
    # lands exactly there.  Forward looked fine; every gradient was nan.
    def nrm(P, dim):
        return (P.pow(2).sum(dim, keepdim=True) + eps * eps).sqrt()

    col_target = (M.shape[-2] / M.shape[-1]) ** 0.5
    P = M
    for _ in range(iters):
        P = P / nrm(P, -1)                        # rows  -> 1
        P = P * col_target / nrm(P, -2)           # cols  -> sqrt(m/n)
    return P / nrm(P, -1)                         # finish exact on rows


def dora_linear(x, W, A, B, m, c: Config):
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

    Batch-dimension agnostic (dim=-1 throughout): composes a single [out,in] from
    a [n_out] vec, or [B,S,out,in] from [B,S,n_out].  W stays [out,in] (shared
    base); broadcasting handles the rest.
    """
    A_u, B_u, m_u, A_d, B_d, m_d = hyper.unpack(vec)
    out = []
    for W, A, B, m in ((W_up, A_u, B_u, m_u), (W_down, A_d, B_d, m_d)):
        # Constraint on the BASE, not the update.  Constraining B@A instead
        # destroyed depth adaptation: unit_row_col_norm is scale-invariant, so l3
        # drifts freely along the scale axis nothing penalises; once l3 is
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
