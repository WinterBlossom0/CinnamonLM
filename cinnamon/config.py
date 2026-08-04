"""Model configuration.  Single source of truth; every field is reachable from
the CLI via `train.py --set KEY=VALUE` (see CONTRIBUTING.md)."""
from dataclasses import dataclass


@dataclass
class Config:
    # Scaled down from the spec's 512/2048 to a ~50 M non-embedding budget
    # (44.63 M actual).  Nothing vocab-sized is charged: embedding and untied LM
    # head are set by the tokenizer, not by any layer width.
    # Ratios preserved: d_ff = 4*d, q_lora = d/2, kv_lora = d/4, alpha = 2*rank.
    d_model: int = 256
    d_ff: int = 1024
    vocab: int = 128_000
    n_heads: int = 8
    head_dim: int = 32
    kv_lora_rank: int = 64
    q_lora_rank: int = 128
    # KDA head width, separate from MLA's.  Recurrent state is
    # [kda_head_dim, kda_head_dim] PER HEAD -- the entire prefix memory -- so
    # capacity grows quadratically while parameters grow linearly.  MLA stays at
    # head_dim: it is NoPE and carries no positional info, so widening it does not
    # help what KDA is for.
    kda_head_dim: int = 48
    rank: int = 16              # DoRA low-rank dimension
    lora_alpha: int = 32        # must be 2*rank, asserted below
    phi_dim: int = 64           # step-embedding width
    eps: float = 1e-6

    r_ceiling: int = 32         # 5.2 hypernet bottleneck; must be >= c_max2

    # One expert body per block position + a bank of hypernetworks + a router.
    # Spec 5.5 content-blindness deliberately overridden: each hypernetwork
    # generates a distinct code PER TOKEN every recurrence, with no discretisation
    # between the content signal and the applied weights, so gradient reaches the
    # content path.
    n_hypernets: int = 8
    r_free: int = 2             # unadapted recurrences before any routing
    turns: int = 2              # recurrences per routing commitment; >=2 required
    c_max2: int = 32            # recurrence cap per block position
    lambda_aux: float = 0.01    # spec 10 load-balancing weight

    # Router gradient path (argmax has none):
    #   "scale"    h <- g*h.  g ~ 1/n_hypernets, shrinks the stream every boundary.
    #   "straight" h <- h * g/g.detach().  Identity forward, same d/dg.
    #   "off"      aux loss only.
    router_gate: str = "straight"

    # spec 12: which readings the block-end check compares.  "turn" = D_r vs
    # D_{r-1} literally; "block" = each block's final D vs the previous block's.
    halt_compare: str = "block"

    # Absolute convergence threshold, checked ALONGSIDE "error still falling".
    # The relative rule alone cannot stop a smooth contraction: with input
    # injection the recurrence converges to a fixed point, error falls
    # monotonically forever, "still improving" is always true (measured: mean
    # depth 104 vs a 64 cap).  The old rule only fired because convergence was
    # noisy enough to stall.
    #
    # Swept on the real model (cap 32, B2 mean / B3 mean):
    #   0.02 -> 26.9/8.1   0.03 -> 25.3/6.8   0.05 -> 20.6/6.1
    #   0.08 -> 14.9/5.9   0.12 -> 11.6/5.6
    # PROVISIONAL: swept on an UNTRAINED model.  Compute/quality knob; revisit.
    halt_tol: float = 0.08

    # Diagnostic: disable halting, run every token to c_max2.  Makes depth
    # constant, removing the discrete decision that rounding noise flips
    # (measured: editing token 40 changed token 12's depth 28 -> 16).
    no_halt: bool = False

    # Implementation knobs, not spec.
    kda_chunk: int = 64              # blocked inverse makes 64 safe
    tie_embeddings: bool = False     # 14.6 revisited: untied head
    grad_checkpoint: bool = True     # also a throughput win: enables larger batch
    detach_norm_input: bool = False  # 5.5 implementation note

    def __post_init__(self):
        # dora_scale fixed at 2.  Overriding `rank` alone and inheriting the
        # default alpha silently 4x's adapter strength -- TINY did exactly that.
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
