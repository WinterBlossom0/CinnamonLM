"""Model configuration.  Single source of truth; every field is reachable from
the CLI via `train.py --set KEY=VALUE` (see CONTRIBUTING.md)."""
import math
from dataclasses import dataclass


@dataclass
class Config:
    # Restored original Cinnamon width. Teacher residual streams are projected
    # into this shared PCA coordinate space before C1/C2/C3 training.
    d_model: int = 256
    d_ff: int = 1024
    # Shrunk from 128,000 (tokenizer_train.py's own --vocab default, sized for the
    # eventual 8-corpus mix -- see architecture.md 7).  At 128k on WikiText-2, only
    # 58,067 ids ever appeared and 69,933 never fired (problems.md P3) -- most of
    # the vocab was dead weight on a small corpus.  Re-measured fresh against the
    # rebuilt 32,768-vocab tokenizer.json on the same corpus: 25,817 appear, 6,951
    # never fire, 1,606 seen exactly once -- 79% live instead of 45%.
    vocab: int = 32_768
    n_heads: int = 8
    head_dim: int = 32
    # 78% of the generated vector is rank*d_ff, so rank is the dominant term in
    # what the hypernetwork has to EMIT per token -- not in what the adapter costs
    # to apply.  At 4, generating the adapter is cheaper than running the FFN it
    # becomes (0.9x); at 16 it was 3.4x.  See architecture.md 4.6.
    rank: int = 16              # DoRA low-rank dimension
    lora_alpha: int = 32        # must be 2*rank, asserted below
    phi_dim: int = 64           # step-embedding width
    eps: float = 1e-6

    r_ceiling: int = 32         # step-embedding table size; must be >= c_max2

    # Width of the code each hypernet emits, expanded to the full DoRA vector by
    # the bank's SHARED AdapterBasis.  This is the hard ceiling on per-token
    # adaptation: the generated code is `basis @ code`, so everything a hypernet
    # can produce -- over every token, every r, every halting error -- lies on a
    # linear manifold of exactly this rank (measured: 32 of 42,240 output dims).
    #
    # Was `r_ceiling`, on the argument that a function of r over r_ceiling values
    # has rank <= r_ceiling.  True while the hypernet was content-blind; per-token
    # generation (5.5 overridden) made the input space far larger and the bound
    # stopped being free.  It also tied adapter capacity to the recurrence cap for
    # no reason -- raising c_max2 forced r_ceiling up and doubled the bank.
    #
    # 64, not 192: the basis matmul is code_dim * n_out PER TOKEN PER RECURRENCE
    # and is the model's dominant cost -- 73% of it at 192, 47% at 64.  192 was
    # measured at 2.82x slower with no evidence the width was used.  64 is
    # 2x the broken original and 1.95x faster than 192; whether even that is
    # needed is what inspect_hypernets.py's effective-rank readout decides.
    code_dim: int = 64

    # One expert body per block position + a bank of hypernetworks + a router.
    # Spec 5.5 content-blindness deliberately overridden: each hypernetwork
    # generates a distinct code PER TOKEN every recurrence, with no discretisation
    # between the content signal and the applied weights, so gradient reaches the
    # content path.
    # 4, not 8.  8 was an unjustified default: nothing in this repo ever measured
    # it, and it carried 26% of the non-embedding parameters (1.59 M of 6.13 M)
    # plus 3.6x the dispatch cost (198 ms at n=1 -> 715 ms at n=8 for identical
    # token-work) in defence of a specialisation benefit no valid measurement has
    # yet demonstrated -- see problems.md P2.
    n_hypernets: int = 4        # see comment above -- was reverted to 8, contradicting it; problems.md P2 measured non-embedding at 5.34 M under 4
    r_free: int = 1             # unadapted recurrences before any routing; router fires right after
    turns: int = 1              # recurrences per routing commitment; router re-selects every recurrence
    c_max2: int = 32            # original recurrence cap
    # 0.1, not the spec's 0.01: at 0.01 three of eight hypernets reached exactly
    # 0.000 share by step 250 of a real run (entropy 2.069 -> 1.823 against a
    # 2.079 max, aux peaking 1.336 over a 1.0 floor).  Balancing pressure was too
    # weak to prevent collapse.
    lambda_aux: float = 0.1     # see comment above -- 0.01 measured collapsing, this wasn't applied
    # Off-diagonal squared-cosine penalty between hypernet codes on the same
    # tokens, active only during router_warmup (train.py --warmup): uniform
    # random dispatch during warmup gives every hypernet equal exposure, but
    # with nothing to differentiate them they can converge to the same
    # function.  Not paper-sourced, not yet tuned -- a starting value.
    lambda_diversity: float = 0.05

    # Router gradient path (argmax has none):
    #   "scale"    h <- g*h.  g ~ 1/n_hypernets, shrinks the stream every boundary.
    #   "straight" h <- h * g/g.detach().  Identity forward, same d/dg.
    #   "off"      aux loss only.
    router_gate: str = "straight"

    # Diagnostic: disable halting, run every token to c_max2.  Makes depth
    # constant, removing the discrete decision that rounding noise flips
    # (measured: editing token 40 changed token 12's depth 28 -> 16).
    no_halt: bool = False

    # Fixed-point halting (arXiv:2606.18206 Algorithm 1), batch-uniform.  Values
    # below verified against the paper's own appendix tables (4, 5, 7), not
    # recalled/inferred -- each is a real number from a real reported config,
    # though not all from the same one:
    # 1/(pi^e * e^pi) = 0.0019241..., not the paper's 0.1 -- deliberately stricter,
    # to counter depth pinning at 2-3 of c_max2=32 observed under this project's
    # (alpha1, alpha2) choice (see fp_alpha1_init/fp_alpha2_init below).  Not
    # re-verified against a real run yet.
    fp_tau: float = 1 / (math.pi**math.e * math.e**math.pi)
    fp_eta0: float = 1.0         # stepsize (eval); Table 7 (ARC-AGI), exact
    fp_gamma: float = 0.9        # decay; Table 7 (ARC-AGI) -- Sudoku/Maze use 0.996-0.997 instead, a much weaker decay
    fp_patience: int = 10        # Sudoku/Maze (Table 4/5); ARC-AGI uses 5 -- paper's own Fig 8 says patience has only minor effect
    # (0.25, 0.75): the actual best cell (94.23%) in the paper's own Table 2
    # sensitivity sweep on Sudoku-Extreme, out of 9 (a1, a2) pairs tested.  Not
    # what the paper's own headline configs use -- Sudoku/Maze/ARC-AGI all ship
    # with (0.75, 0.25) instead (83.24% in that same sweep, second-worst cell),
    # for a reason the paper doesn't state.
    fp_alpha1_init: float = 0.25
    fp_alpha2_init: float = 0.75

    # Safety net, not a fix for the root cause: bisecting RoutedBlock.forward's
    # steps= against a real batch on a checkpoint that later diverged showed
    # forward activations flat across depth while the backward gradient grew
    # ~exponentially with recurrence depth (40 at depth 16 -> 53,230 at depth
    # 32) -- alpha1/beta1's mixing has no constraint keeping its backward
    # Jacobian contractive, so a per-step amplification invisible in the
    # forward pass compounds geometrically over c_max2 steps.  This clips the
    # carried state's gradient norm once per recurrence, before it multiplies
    # through the next step.  0.0 disables it.
    # Swept on the same checkpoint+batch: 10.0 -> depth-32 gnorm 2,247 (barely
    # engages before depth ~20); 1.0 -> 307; 0.3 -> 132, roughly linear growth
    # with depth (30 at 16 -> 132 at 32) instead of exponential.  0.3 chosen;
    # not verified yet against a full training run.
    recur_grad_clip: float = 0.3

    # Complements recur_grad_clip on the other direction: that one bounds the
    # BACKWARD gradient through the carried state; this bounds the FORWARD
    # value.  They guard against different failures -- a live run with the
    # gradient clip active still hit a single-step NaN (689 finite -> NaN in
    # one step, not a gradual ramp), which gradient clipping structurally
    # cannot prevent (nan * anything = nan; clipping only shrinks large-but-
    # finite gradients).  A real forward-value overflow needs a forward-side
    # guard.  Per-element clamp, not a norm rescale (like recur_grad_clip) --
    # the failure mode is a single element reaching bf16's range, which a
    # norm bound alone would not stop.  50.0 is generous: measured h_max
    # (real batch, real checkpoint) stayed under 4 through depth 32 even near
    # a since-diverged checkpoint, so this should be a no-op in the healthy
    # regime and only bind if something is already going wrong.  0.0 disables.
    recur_val_clip: float = 50.0

    # Canonical Kimi Delta Attention. Its decaying recurrent state and causal
    # short convolution carry sequence order, so Cinnamon has no positional
    # encoding, MLA rank, or compressed-attention configuration.
    kda_short_conv: bool = True
    kda_conv_size: int = 4
    kda_safe_gate: bool = True
    # -5.0 (fla's own default) bounds only the PER-STEP log-decay g in
    # [lower_bound, 0), i.e. each step keeps >= exp(-5) ~ 0.67% of state.  The
    # paper (arXiv:2510.26692 Kimi Linear, p.5) flags that its chunked kernel
    # divides by the CUMULATIVE product of gates across a whole chunk, not
    # the per-step gate -- exp(-5)^C underflows to exactly 0 in bf16 for any
    # normal chunk length C, and dividing by that zero is a genuine forward-
    # pass NaN source the per-step clamp cannot prevent (it bounds one step,
    # not the product).  Matches what was observed: a real training run's
    # loss (not just its gradient) went non-finite abruptly after enough
    # steps for gates to drift toward this bound.  -3.0 (~5%/step floor)
    # tightens the per-chunk product's underflow margin; not yet verified
    # against a full run.
    kda_gate_lower_bound: float = -3.0

    # Positions per cross-entropy tile; 0 = one shot over the whole sequence.
    # The logits tensor is [B, T, vocab] and never compressed by anything on the
    # attention side -- at T=16384, batch 1, current vocab=32,768 that is 1.07 GB
    # bf16 plus a 2.15 GB fp32 copy; it scales with batch from there, and was
    # fatal at batch 1 back when vocab was 128,000 (4.2 GB + 8.4 GB). Tiles
    # are checkpointed, so peak is one tile and `logits` comes back None.
    #
    # ON BY DEFAULT.  It was 0, and that single default was where this model's
    # memory actually went -- not the recurrence, not the hypernetworks.  Measured
    # at batch 4, seq 1024, one full train step (params 0.85 GB live throughout),
    # BEFORE the vocab shrink -- vocab=128,000 at the time, not re-profiled since:
    #
    #   whole 32-recurrence routed block, checkpointed   0.95 GB
    #   logits bf16 + fp32 copy + cross_entropy           4.94 GB
    #   step peak                                         7.97 GB   (0), 3.62 GB (1024)
    #
    # i.e. 62% of peak was a tensor the trainer immediately discards.  Freeing it
    # is what lets batch rise.  Best batch that fits a 16 GB card, seq 1024:
    #
    #            best batch   reserved   tok/s
    #   at 0          6        14.12 GB   2155     (batch 7 reserves 16.32 GB and
    #   at 1024      24         9.72 GB   3568      spills to host memory)
    #
    # 1024 rather than 256: the tile is recomputed in backward, so smaller tiles
    # buy memory nobody needs any more (2.90 GB at 512, 2.55 GB at 256) for real
    # throughput.  It costs ~9% at batch 4, where the logits never dominated;
    # that is the price of the batch headroom, not a free win.
    loss_chunk: int = 1024

    # How the bank is dispatched.  All three are the same arithmetic with the same
    # gradients (pinned at 1e-10 in fp64), so this is purely a speed choice --
    # none of them is a different model.
    #   "sorted"  tokens sorted into per-hypernet segments once, trunk as grouped
    #             GEMMs, the shared basis + DoRA FFN applied ONCE.  No padding.
    #   "loop"    one hypernet at a time.  The reference.
    #   "bmm"     whole bank padded to the largest group; falls back to the loop
    #             when padding would cost more than the launches it saves, which
    #             at any realistic batch is always -- so this selected the loop.
    #
    # "sorted" is THE DEFAULT.  Against the loop it drops a training step from
    # 99,480 CUDA launches to 33,699 (-66.1%; the forward, which is the launch-bound
    # half, 23,874 -> 8,405) and removes all ~248 of the loop's per-group nonzero()
    # host syncs.  Batch 16, seq 1024, interleaved paired over 7 rounds: +6.44%
    # median, positive in every round (+4.98% to +9.94%).  Best batch against best
    # batch, round-robin interleaved so drift hits both: 3062 tok/s at batch 20
    # against the loop's 2969 at batch 24.
    #
    # It costs memory, and that is what bounds the batch: one merged call holds the
    # whole generated adapter live during checkpoint recompute, where the loop held
    # an eighth of it.  Measured seq 1024, allocated / reserved, one process each:
    #
    #   batch      16      20      24      28      32 (chunk 8192)
    #   alloc    8.07    9.99   11.90   13.81    9.99  GiB
    #   resv     8.39   10.31   12.28   15.04   10.13  GiB
    #   tok/s    5746    4857    4987    4926    4691
    #
    # Batch 16 is both the fastest and the smallest, so nothing above it is worth
    # taking.  28 reserves 15.04 GiB, over the 15 GiB working ceiling; 32 needs
    # dispatch_chunk to run at all.
    #
    # The tok/s row above predates _AxGram and the epilogue's fused value term
    # (hypernet.py, dora_kernel.py), which took ~20% out of a step without moving
    # memory.  Re-measured since, seq 1024: batch 12  7410 tok/s  6.16/6.33 GiB;
    # 14  7553  7.12/7.39; 16  7620  8.08/8.30; 20  7564  9.99/10.26.  The shape
    # of the curve is unchanged -- 16 is still the pick.  These are much lower than the numbers this
    # comment used to carry (10.31 GiB at 16, no fit at 24) because unpack, the
    # generated factors and Q's GEMM stopped allocating what they were allocating
    # -- see AdapterBasis.unpack / .generate / .q_basis.
    dispatch: str = "sorted"
    # Tokens per merged "sorted" call; 0 = the whole batch in one.  Rows are
    # independent so the arithmetic is identical at any value -- this is purely
    # where the card sits on the launches-vs-live-memory curve, and it is the valve
    # that keeps "sorted" runnable past batch 20.
    #
    # It buys memory and nothing else, and the curve is MONOTONE -- there is no
    # sweet spot to find.  Swept round-robin at batch 16 (M = 16384), 4 rounds:
    #
    #   chunk        0    2048    4096    8192   12288
    #   vs 0      +0.0%  -25.7%  -12.0%   -3.6%   -3.9%
    #   alloc      8.07    4.86    5.21    6.17    7.15  GiB
    #
    # Every value loses, and the loss shrinks as the chunk grows back toward M.
    # It does extend the ceiling: batch 32 will not run unchunked, and at 8192 it
    # fits in 9.99/10.13 GiB and does 4691 tok/s -- still under batch 16's 5746 in
    # 8.39 GiB.  So chunking exists to make a too-large batch run at all, and no
    # batch is faster for having it.
    dispatch_chunk: int = 0
    # Keep the default memory-safe dispatch fallback.  C2 activation training
    # can explicitly force the algebraically equivalent padded-bank path when
    # measuring throughput on a known batch/sequence shape.
    force_padded_dispatch: bool = False

    # bmm batches every group, which is exactly why it saves launches AND why it
    # costs memory: it materialises [n, K, n_out] for the whole bank at once.
    # Measured end-to-end, batch 1:
    #
    #   n*K*n_out   11.8M   23.6M   47.2M   94.4M    189M
    #   bmm vs loop  4.57x   2.96x   1.70x   1.21x   0.72x
    #
    # So it wins while launch-bound and loses once memory-bound.  Chunking the
    # bmm does not rescue it -- the chunks reintroduce the launches it exists to
    # remove.  Above this many elements, fall back to the loop.
    bmm_max_elems: int = 128_000_000

    # Implementation knobs, not spec.
    tie_embeddings: bool = False     # original untied head
    grad_checkpoint: bool = True     # also a throughput win: enables larger batch
    detach_norm_input: bool = False  # 5.5 implementation note

    def __post_init__(self):
        # dora_scale fixed at 2.  Overriding `rank` alone and inheriting the
        # default alpha silently 4x's adapter strength -- TINY did exactly that.
        assert self.lora_alpha == 2 * self.rank, (
            f"lora_alpha must be 2*rank (got {self.lora_alpha} vs rank {self.rank}); "
            f"dora_scale is fixed at 2")
        assert self.r_ceiling >= self.c_max2, (
            f"r_ceiling={self.r_ceiling} must be >= c_max2={self.c_max2}")
        assert self.n_heads * self.head_dim == self.d_model, (
            f"KDA width mismatch: {self.n_heads} * {self.head_dim} != {self.d_model}")
        assert 0.0 < self.fp_gamma < 1.0, f"fp_gamma must be in (0,1), got {self.fp_gamma}"
        assert 0.0 < self.fp_eta0 <= 1.0, f"fp_eta0 must be in (0,1], got {self.fp_eta0}"

    @property
    def dora_scale(self) -> float:
        return self.lora_alpha / self.rank      # always 2.0, asserted above


TINY = dict(d_model=256, d_ff=768, vocab=256, n_heads=2, head_dim=128,
            rank=4, lora_alpha=8, phi_dim=8, r_ceiling=8, c_max2=8,
            n_hypernets=1)
