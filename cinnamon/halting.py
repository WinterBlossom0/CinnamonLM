"""7 Halting rule.  Parameter-free control signal, no gradient (15)."""
import torch


def halt_signal(h: torch.Tensor, eps: float) -> torch.Tensor:
    """p_j = (h_j^2 + eps) / sum_k (h_k^2 + eps).

    Squaring makes every component non-negative so the normalisation is valid;
    sign is deliberately discarded.  eps keeps entries strictly positive because
    KL's log diverges on an exact zero.  No temperature or pre-scale: scaling
    collapses the distribution toward one-hot and destroys monotonicity.
    """
    q = h.pow(2) + eps
    return q / q.sum(-1, keepdim=True)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.log() - q.log())).sum(-1)


def rel_error(h: torch.Tensor, prev: torch.Tensor, eps: float) -> torch.Tensor:
    """Relative Frobenius error between consecutive states, per token.

        ||h - prev||_F / (||h||_F + eps)

    Replaces the KL of halt_signal as the convergence measure.  Three reasons it
    is the better signal here:

      * it measures the state directly.  KL went through h^2 normalised into a
        distribution, which discards sign and magnitude -- two states of very
        different size could give the same KL.
      * it is scale-relative, so "how much did I move, compared to how big I am"
        means the same thing at every depth and width.
      * it is smooth in h.  KL's log is steep near zero, which magnified exactly
        the rounding-level differences that were flipping halting decisions.

    Per token the Frobenius norm is just the L2 norm over the feature axis.
    """
    return (h - prev).norm(dim=-1) / (h.norm(dim=-1) + eps)
