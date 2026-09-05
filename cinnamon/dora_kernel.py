"""Fused epilogue for dora_linear: the value's low-rank term, quad, cross,
||V_j||, and the DoRA output in one Triton kernel (one more for the backward).

The einsum form materialises [.., out, r] for quad and reads [.., r, out] for
cross; both are ~7 M elements per call and both are consumed by a single
reduction, so they never need to exist.  Everything after Q and lin is register
work here.  The GEMMs that PRODUCE lin/G/Q stay on cuBLAS -- they are the FLOPs,
and a hand-rolled Triton GEMM would only be slower.  Measured, not assumed: a
Triton kernel taking (A x, A A^T) in ONE pass over A instead of cuBLAS's three
reads runs 0.416 ms against 0.359 ms at [16384, 16, 256] and 1.550 against 1.624
at [16384, 16, 1024] -- it moves a third of the bytes and does not win, because
the re-reads come out of cache, not HBM.

s*(B @ ax) is the exception, and it is here because it is NOT a GEMM saving: B is
[.., out, r], 537 MB for the up projection at batch 16, and this kernel already
holds the tile in registers for quad and cross.  As a separate einsum it was a
second full read of B in the forward and a second dB in the backward for autograd
to add to this one; folded in it costs one [.., r] load.  +9.68% of a training
step, paired at batch 16, 9/9 rounds.

Reductions run in fp32 (fp64 for fp64 inputs) regardless of input dtype, which is
at least as precise as the aten path: there `quad` lands in fp32 by promotion but
`cross` is a bf16 einsum under autocast.

# Q (A @ Wn^T, the O(r*d*out) term) is still materialised, but it is no longer
# BUILT here-shaped: dora_linear takes it as `q` from AdapterBasis.q_basis, which
# reaches the same tensor as code @ T with T = (W_A contracted with Wn) constant
# within a forward pass, so the per-token contraction is over code_dim=64 instead
# of d_model=256 (up) / d_ff=1024 (down).  Worth +15.35% of a training step,
# paired at batch 16, 9/9 rounds.  It was declined at an earlier 7.6%; what
# changed is the denominator, not the idea -- the unpack/generate fixes took ~30%
# out of the step around it.
#
# ponytail: the remaining option, folding the GEMM into this kernel, is still not
# taken -- it saves the [.., r, out] round-trip, not the FLOPs, and means
# hand-writing a GEMM cuBLAS already runs well.  Measured ceiling for removing
# Q's GEMM outright is 24.8% of a step against the old A-route, and the code
# route above already collects most of that.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bg(bn, b, G, n, sg0, sg1, sg2, rr, mr, R: tl.constexpr, RP: tl.constexpr,
        BO: tl.constexpr, ACC: tl.constexpr, DOT: tl.constexpr, BOTH: tl.constexpr):
    """(B[n] @ G[n], B[n] @ G[n]^T) over one [BO, RP] tile of B.  The transposed
    product is only built when BOTH -- forward needs one, backward needs both,
    because quad's two B operands are independent to autograd and G is symmetric
    only up to the error of the bmm that built it.

    DOT is the tensor-core path: [BO,RP] x [RP,RP], operands left in their own
    dtype so a bf16 pair multiplies exactly into an fp32 accumulator -- the same
    arithmetic as the fallback, reassociated.  The fallback walked G a column at a
    time and re-read B's whole tile from memory RP times over at stride sb1;
    measured on the dominant [up] shape (N=2048, OUT=1024, r=16, bf16) that cost
    1.574 ms forward against 0.294 ms here.  It stays for fp64 (no tensor cores)
    and for RP < 16 (tl.dot's minimum tile), and both paths are pinned to the same
    fp64 reference by test_dora_triton_epilogue_matches_reference_and_grads.
    """
    if DOT:
        g = tl.load(G + n * sg0 + rr[:, None] * sg1 + rr[None, :] * sg2,
                    mask=mr[:, None] & mr[None, :], other=0.0)              # [RP, RP]
        bg = tl.dot(bn, g, out_dtype=ACC)
        return bg, (tl.dot(bn, tl.trans(g), out_dtype=ACC) if BOTH else bg)
    acc = tl.zeros([BO, RP], dtype=ACC)
    acct = tl.zeros([BO, RP], dtype=ACC)
    for p in tl.static_range(RP):
        if p < R:
            # column p of the tile already in registers, rather than re-loading it
            bp = tl.sum(b * (rr == p)[None, :], axis=1)
            grow = tl.load(G + n * sg0 + p * sg1 + rr * sg2, mask=mr, other=0.0)
            acc += bp[:, None] * grow.to(ACC)[None, :]
            if BOTH:
                gcol = tl.load(G + n * sg0 + rr * sg1 + p * sg2, mask=mr, other=0.0)
                acct += bp[:, None] * gcol.to(ACC)[None, :]
    return acc, (acct if BOTH else acc)


@triton.jit
def _fwd_kernel(LIN, AX, M, B, G, Q, WROW2, WNORM, Y, DEN,
                sv0, sv1, sax0, sax1, sm0, sm1, sb0, sb1, sb2,
                sg0, sg1, sg2, sq0, sq1, sq2, sy0, sy1,
                OUT, S: tl.constexpr, EPS: tl.constexpr,
                R: tl.constexpr, RP: tl.constexpr, BO: tl.constexpr,
                ACC: tl.constexpr, DOT: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BO + tl.arange(0, BO)
    mo = o < OUT
    rr = tl.arange(0, RP)
    mr = rr < R

    bn = tl.load(B + n * sb0 + o[:, None] * sb1 + rr[None, :] * sb2,
                 mask=mo[:, None] & mr[None, :], other=0.0)                 # [BO, RP]
    b = bn.to(ACC)
    q = tl.load(Q + n * sq0 + rr[:, None] * sq1 + o[None, :] * sq2,
                mask=mr[:, None] & mo[None, :], other=0.0).to(ACC)          # [RP, BO]

    bg, _ = _bg(bn, b, G, n, sg0, sg1, sg2, rr, mr, R, RP, BO, ACC, DOT, False)

    quad = tl.sum(bg * b, axis=1)
    cross = tl.sum(b * tl.trans(q), axis=1)

    w2 = tl.load(WROW2 + o, mask=mo, other=0.0).to(ACC)
    vn2 = w2 + 2.0 * S * cross + S * S * quad
    # clamp before sqrt: the three terms are exact in fp32 but can land a hair
    # below zero in bf16, and sqrt of a negative is nan with no other warning.
    vn = tl.sqrt(tl.maximum(vn2, 0.0))
    den = vn + EPS

    # value's low-rank term off the B tile already in registers -- see dora_epilogue.
    ax = tl.load(AX + n * sax0 + rr * sax1, mask=mr, other=0.0).to(ACC)     # [RP]
    lin = tl.load(LIN + n * sv0 + o * sv1, mask=mo, other=0.0).to(ACC)
    val = lin + S * tl.sum(b * ax[None, :], axis=1)
    m = tl.load(M + n * sm0 + o * sm1, mask=mo, other=0.0).to(ACC)
    wn = tl.load(WNORM + o, mask=mo, other=0.0).to(ACC)
    y = wn * (1.0 + m) * val / den

    tl.store(Y + n * sy0 + o * sy1, y, mask=mo)
    tl.store(DEN + n * sy0 + o * sy1, den, mask=mo)


@triton.jit
def _bwd_kernel(GY, LIN, AX, M, B, G, Q, WNORM, DEN,
                DLIN, DAXP, DM, DB, DQ, DVN2, DWN,
                sv0, sv1, sax0, sax1, sm0, sm1, sb0, sb1, sb2,
                sg0, sg1, sg2, sq0, sq1, sq2, sy0, sy1,
                sdv0, sdv1, sdp0, sdp1, sdp2,
                sdm0, sdm1, sdb0, sdb1, sdb2, sdq0, sdq1, sdq2,
                OUT, S: tl.constexpr, EPS: tl.constexpr,
                R: tl.constexpr, RP: tl.constexpr, BO: tl.constexpr,
                ACC: tl.constexpr, DOT: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)
    o = t * BO + tl.arange(0, BO)
    mo = o < OUT
    rr = tl.arange(0, RP)
    mr = rr < R

    bn = tl.load(B + n * sb0 + o[:, None] * sb1 + rr[None, :] * sb2,
                 mask=mo[:, None] & mr[None, :], other=0.0)                 # [BO, RP]
    b = bn.to(ACC)
    q = tl.load(Q + n * sq0 + rr[:, None] * sq1 + o[None, :] * sq2,
                mask=mr[:, None] & mo[None, :], other=0.0).to(ACC)          # [RP, BO]
    ax = tl.load(AX + n * sax0 + rr * sax1, mask=mr, other=0.0).to(ACC)     # [RP]

    gy = tl.load(GY + n * sy0 + o * sy1, mask=mo, other=0.0).to(ACC)
    lin = tl.load(LIN + n * sv0 + o * sv1, mask=mo, other=0.0).to(ACC)
    val = lin + S * tl.sum(b * ax[None, :], axis=1)     # recomputed, not stored
    m = tl.load(M + n * sm0 + o * sm1, mask=mo, other=0.0).to(ACC)
    wn = tl.load(WNORM + o, mask=mo, other=0.0).to(ACC)
    # other=1.0, not 0.0: dval is reduced over o below, so a masked lane must be a
    # clean zero rather than gy(0) * mag / den(0) = nan poisoning the sum.
    den = tl.load(DEN + n * sy0 + o * sy1, mask=mo, other=1.0).to(ACC)

    mag = wn * (1.0 + m)
    dmag = gy * val / den
    dval = gy * mag / den
    dden = -gy * mag * val / (den * den)
    vn = den - EPS
    # vn == 0 is exactly the clamped branch, where clamp_min kills the gradient.
    # The aten path reaches it as inf*0 = nan instead; 0 is the value that
    # composition actually implies.
    dvn2 = tl.where(vn > 0.0, dden * 0.5 / vn, 0.0)
    dquad = S * S * dvn2
    dcross = 2.0 * S * dvn2

    tl.store(DLIN + n * sdv0 + o * sdv1, dval, mask=mo)
    tl.store(DM + n * sdm0 + o * sdm1, dmag * wn, mask=mo)
    tl.store(DVN2 + n * sy0 + o * sy1, dvn2, mask=mo)
    tl.store(DWN + n * sy0 + o * sy1, dmag * (1.0 + m), mask=mo)

    # Both B@G and B@G^T: quad's two B operands are independent to autograd, and
    # G is only symmetric up to the error of the bmm that built it.
    bg, bgt = _bg(bn, b, G, n, sg0, sg1, sg2, rr, mr, R, RP, BO, ACC, DOT, True)

    # dB now carries the value term too: val = lin + s*(B @ ax).
    db = (dquad[:, None] * (bg + bgt) + dcross[:, None] * tl.trans(q)
          + (S * dval)[:, None] * ax[None, :])
    tl.store(DB + n * sdb0 + o[:, None] * sdb1 + rr[None, :] * sdb2, db,
             mask=mo[:, None] & mr[None, :])
    tl.store(DQ + n * sdq0 + rr[:, None] * sdq1 + o[None, :] * sdq2,
             dcross[None, :] * tl.trans(b), mask=mr[:, None] & mo[None, :])
    # dax = s * sum_o dval[o] B[o, r] spans every o tile, so each tile drops a
    # partial and the caller sums the tile axis.  A [tiles, N, R] fp32 buffer is
    # 17 MB at the widest shape here, and the deterministic order costs nothing
    # against atomics on the same addresses.
    tl.store(DAXP + t * sdp0 + n * sdp1 + rr * sdp2,
             S * tl.sum(dval[:, None] * b, axis=0), mask=mr)


def _ref(lin, ax, m, B, G, Q, wrow2, wnorm, s, eps):
    """The aten expression dora_linear used to inline.  Non-CUDA fallback and the
    oracle the Triton path is tested against."""
    val = lin + s * torch.einsum('...or,...r->...o', B, ax)
    quad = (torch.einsum('...or,...rq->...oq', B, G) * B).sum(-1)
    cross = torch.einsum('...or,...ro->...o', B, Q)
    vn = (wrow2 + 2.0 * s * cross + s * s * quad).clamp_min(0).sqrt()
    return wnorm * (1.0 + m) * val / (vn + eps)


class _DoraEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lin, ax, m, B, G, Q, wrow2, wnorm, s, eps):
        N, OUT = lin.shape
        R = B.shape[-1]
        RP = max(triton.next_power_of_2(R), 1)
        BO = 64 if RP <= 16 else 32
        dt = lin.dtype
        for t in (ax, m, B, G, Q, wrow2, wnorm):
            dt = torch.promote_types(dt, t.dtype)
        acc = tl.float64 if dt == torch.float64 else tl.float32
        adt = torch.float64 if dt == torch.float64 else torch.float32
        # tl.dot needs a 16x16 minimum tile, and both operands in one dtype.
        # Restricted to bf16/fp16 on purpose: there the tensor core's fp32
        # accumulate is EXACTLY the fallback's arithmetic (a bf16 product is
        # representable in fp32), so this is a reassociation and nothing else.
        # On fp32 operands tl.dot would silently drop to TF32's 10-bit mantissa,
        # which is a precision regression, not an optimisation.
        dot = (RP >= 16 and B.dtype == G.dtype
               and B.dtype in (torch.bfloat16, torch.float16))

        y = torch.empty((N, OUT), dtype=dt, device=lin.device)
        den = torch.empty((N, OUT), dtype=adt, device=lin.device)
        grid = (N, triton.cdiv(OUT, BO))
        _fwd_kernel[grid](lin, ax, m, B, G, Q, wrow2, wnorm, y, den,
                          *lin.stride(), *ax.stride(), *m.stride(), *B.stride(),
                          *G.stride(), *Q.stride(), *y.stride(),
                          OUT, float(s), float(eps), R, RP, BO, acc, dot)

        ctx.save_for_backward(lin, ax, m, B, G, Q, wnorm, den)
        ctx.meta = (s, eps, R, RP, BO, acc, adt, wrow2.dtype, dot)
        return y

    @staticmethod
    def backward(ctx, gy):
        lin, ax, m, B, G, Q, wnorm, den = ctx.saved_tensors
        s, eps, R, RP, BO, acc, adt, w2dt, dot = ctx.meta
        N, OUT = lin.shape
        gy = gy.contiguous()

        dlin = torch.empty_like(lin)
        dm = torch.empty_like(m)
        dB = torch.empty(B.shape, dtype=B.dtype, device=B.device)
        dQ = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        dvn2 = torch.empty((N, OUT), dtype=adt, device=B.device)
        dwn = torch.empty((N, OUT), dtype=adt, device=B.device)
        tiles = triton.cdiv(OUT, BO)
        daxp = torch.empty((tiles, N, R), dtype=adt, device=B.device)
        _bwd_kernel[(N, tiles)](gy, lin, ax, m, B, G, Q, wnorm, den,
                                dlin, daxp, dm, dB, dQ, dvn2, dwn,
                                *lin.stride(), *ax.stride(), *m.stride(),
                                *B.stride(), *G.stride(), *Q.stride(), *gy.stride(),
                                *dlin.stride(), *daxp.stride(),
                                *dm.stride(), *dB.stride(), *dQ.stride(),
                                OUT, float(s), float(eps), R, RP, BO, acc, dot)

        need = ctx.needs_input_grad
        # dG[p,q] = sum_o dquad[o] B[o,p] B[o,q].  Three aten calls; a fourth
        # Triton kernel for it would save two launches per backward.
        dG = ((B * (dvn2 * (s * s)).unsqueeze(-1).to(B.dtype)).transpose(-1, -2) @ B
              ).to(G.dtype) if need[4] else None
        return (dlin, daxp.sum(0).to(ax.dtype), dm, dB, dG, dQ,
                dvn2.sum(0).to(w2dt) if need[6] else None,
                dwn.sum(0).to(wnorm.dtype) if need[7] else None,
                None, None)


def dora_epilogue(lin, ax, m, B, G, Q, wrow2, wnorm, s, eps):
    """lin/m: [.., out]  ax: [.., r]  B: [.., out, r]  G: [.., r, r]
    Q: [.., r, out]  wrow2/wnorm: [out].  Returns [.., out].

    Takes the value split as `lin` (W_n x) and `ax` (A x) rather than assembled:
    val = lin + s*(B @ ax) is computed off the B tile the kernel already has in
    registers.  B is [.., out, r] -- 537 MB for the up projection at batch 16 --
    and as a separate einsum it was a second full read of that in the forward and
    a SECOND dB in the backward, which autograd then added to this one.  Folding
    it in costs one [.., r] load and leaves B with a single gradient path.
    """
    if not lin.is_cuda:
        return _ref(lin, ax, m, B, G, Q, wrow2, wnorm, s, eps)
    lead, out = lin.shape[:-1], lin.shape[-1]
    r = B.shape[-1]
    y = _DoraEpilogue.apply(
        lin.reshape(-1, out), ax.reshape(-1, r), m.reshape(-1, out),
        B.reshape(-1, out, r), G.reshape(-1, r, r), Q.reshape(-1, r, out),
        wrow2.contiguous(), wnorm.contiguous(), s, eps)
    return y.view(*lead, out)
