"""7 Halting rule.  Parameter-free control signal, no gradient (15)."""
import torch


def fixed_point_residual(h: torch.Tensor, prev: torch.Tensor, eps: float) -> torch.Tensor:
    """Relative max-abs distance between consecutive states, per token."""
    return (h - prev).abs().amax(dim=-1) / (h.abs().amax(dim=-1) + eps)
