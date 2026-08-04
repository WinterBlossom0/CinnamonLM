"""Model configuration.  Values from recurrent-moe-architecture.md section 1."""
from dataclasses import dataclass


@dataclass
class Config:
    # Widths are scaled down from the spec's 512/2048 to a ~50 M budget.
    # The budget counts NO vocab-sized matrix: the embedding and the (untied) LM
    # head are set by the 128 k tokenizer, not by any layer width, so neither is
    # tunable here and neither is charged against the target.  40.11 M of 50 M.
    # Ratios are preserved: d_ff = 4*d, q_lora = d/2, kv_lora = d/4, alpha = 2*rank.
    # 256 = 8 x 32: power-of-two width and head count, unlike 224 = 7 x 32.
    d_model: int = 256
    d_ff: int = 1024
    vocab: int = 128_000
    n_heads: int = 8
    head_dim: int = 32
    kv_lora_rank: int = 64
    q_lora_rank: int = 128
    # KDA gets its own head width, separate from MLA's.  Its recurrent state is a
    # [kda_head_dim, kda_head_dim] matrix PER HEAD -- the entire memory of the
    # prefix -- so capacity grows quadratically here while parameters only grow
    # linearly.  At 32 that is 1024 slots per head to hold 128 tokens of context;
    # 48 more than doubles it.  MLA is left at head_dim: it is NoPE and carries no
    # positional information, so widening it would not help what KDA is for.
    kda_head_dim: int = 48
    rank: int = 16              # DoRA low-rank dimension
    lora_alpha: int = 32
    phi_dim: int = 64           # sinusoidal recurrence encoding width
    eps: float = 1e-6

    r_ceiling: int = 32         # 5.2 hypernet bottleneck == max reachable r

    # ONE expert body per block position (KDA / FixedFFN / MLA / dynamic-FFN
    # base), a bank of hypernetworks, and a router that picks between them.
    #
    # Content-blindness (5.5) is deliberately overridden: each hypernetwork
    # generates a genuinely distinct code PER TOKEN, every recurrence, from a
    # real content signal (RoutedBlock.ctx_kda -> Hypernet.forward_token) -- not
    # one shared table row looked up by every token.  There is no discretisation
    # step between that signal and the weights actually applied, so gradient
    # reaches the content path for real.
    n_hypernets: int = 8
    r_free: int = 2             # recurrences run on the plain base weights, no DoRA
    turns: int = 2              # recurrences one routing decision commits for
    c_max2: int = 32            # recurrence cap per block position
    lambda_aux: float = 0.01    # 10 load-balancing weight
    # How the router earns gradient, since argmax has none.
    #   "scale"    h <- g*h.  Simple, but g is ~1/n_hypernets, so it shrinks the
    #              whole residual stream ~5x at every commitment boundary.
    #   "straight" h <- h * g/g.detach().  Identity in the forward pass, same
    #              gradient w.r.t. g, no effect on the state.
    #   "off"      no multiply; the router learns from the aux loss alone.
    router_gate: str = "straight"
    # 12 which two KL values the block-end check compares.  "turn" reads the
    # spec's D_r vs D_{r-1} literally; "block" compares each block's final D to
    # the previous block's.  A handoff perturbs the state between blocks, so the
    # first turn after one always has high KL and the second lower -- making the
    # within-block comparison read "still converging" every single time.
    halt_compare: str = "block"
    # 12 Absolute convergence threshold, checked ALONGSIDE "is the error still
    # falling".  The relative rule alone cannot stop a well-behaved iteration: with
    # input injection the recurrence converges to a fixed point, so the error falls
    # monotonically forever and "still improving" reads true at every boundary --
    # measured, mean depth 104 against a 64 cap.  The old rule only ever fired
    # because convergence was noisy enough to stall.  A token whose state has
    # stopped moving by more than this is done, however smooth the approach.
# Swept on the real model rather than guessed (cap 32, B2 mean / B3 mean):
    #   0.02 -> 26.9 / 8.1     0.05 -> 20.6 / 6.1     0.12 -> 11.6 / 5.6
    #   0.03 -> 25.3 / 6.8     0.08 -> 14.9 / 5.9
    # 0.08 keeps both blocks well under the cap with real per-token spread
    # (B2 10-24) instead of pinning at either end.  PROVISIONAL: swept on an
    # UNTRAINED model, because that is all that exists to sweep on -- it is a
    # compute/quality knob and wants revisiting once training actually works.
    halt_tol: float = 0.08
    # Diagnostic: disable halting entirely and run every token to c_max2.  Depth
    # becomes a constant, so the discrete decision that rounding noise was
    # flipping (measured: editing token 40 changed token 12's depth 28 -> 16)
    # disappears and the network becomes a continuous function of its input.
    # If the loss then drops below the unigram floor, that chaos was the blocker.
    no_halt: bool = False

    # implementation knobs, not part of the spec
    kda_chunk: int = 64              # blocked inverse makes 64 safe; 128 saturates _G_CLAMP
    tie_embeddings: bool = False     # 14.6 revisited: untied, separate LM head
    grad_checkpoint: bool = True     # required: up to c_max2 recurrences of activations
    detach_norm_input: bool = False  # 5.5 implementation note

    def __post_init__(self):
        # dora_scale is fixed at 2.  It is easy to break by overriding rank alone
        # and inheriting the default alpha -- TINY did exactly that and ran at 8.0,
        # a 4x stronger adapter than anything else in the suite.
        assert self.lora_alpha == 2 * self.rank, (
            f"lora_alpha must be 2*rank (got {self.lora_alpha} vs rank {self.rank}); "
            f"dora_scale is fixed at 2")

    @property
    def dora_scale(self) -> float:
        return self.lora_alpha / self.rank      # always 2.0, asserted above


TINY = dict(d_model=64, d_ff=128, vocab=256, n_heads=2, head_dim=32,
            kda_head_dim=32,          # pinned: only the shipped config widens KDA
            kv_lora_rank=16, q_lora_rank=32, rank=4, lora_alpha=8, phi_dim=8,
            r_ceiling=8, kda_chunk=8, n_hypernets=1)
