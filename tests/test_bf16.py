"""bf16/fp16 autocast correctness.

Requires CUDA: canonical KDA (`fla`'s Kimi Delta Attention) has no CPU path, so
this can no longer run under `torch.autocast("cpu", ...)` the way it did under
the old MLA attention. Runs the real `torch.autocast("cuda", ...)` dispatch
instead. Catches what actually breaks under lower precision: NaN/inf, or the
loss diverging from the fp32 run -- the model's fp32 promotion already showed
once that "should be numerically fine" is not something to assume on this
architecture without checking.

Run: python -m tests.test_bf16
"""
import torch

from cinnamon.config import TINY, Config
from cinnamon.model import CinnamonModel
from train import autocast_ctx

DEV = "cuda"


def test_bf16_forward_backward_finite():
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).to(DEV)
    ids = torch.randint(0, cfg.vocab, (2, 32), device=DEV)
    with autocast_ctx(DEV, "bf16"):
        logits, loss, depth = m(ids, labels=ids)
    assert torch.isfinite(loss), float(loss)
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients under bf16: {bad}"


def test_bf16_close_to_fp32_at_init():
    """Same weights, same batch: bf16 and fp32 should agree to bf16's own
    precision floor (~1e-2 relative for a stack this deep), not diverge outright."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).to(DEV).eval()
    ids = torch.randint(0, cfg.vocab, (2, 32), device=DEV)
    with torch.no_grad():
        _, loss_fp32, _ = m(ids, labels=ids)
        with autocast_ctx(DEV, "bf16"):
            _, loss_bf16, _ = m(ids, labels=ids)
    rel = abs(float(loss_bf16) - float(loss_fp32)) / abs(float(loss_fp32))
    assert rel < 0.05, f"fp32 loss {float(loss_fp32):.4f} vs bf16 {float(loss_bf16):.4f} (rel {rel:.3f})"


def test_bf16_does_not_change_eval_when_disabled():
    """bf16=False must be a true no-op, not an accidental always-on cast."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).to(DEV).eval()
    ids = torch.randint(0, cfg.vocab, (2, 16), device=DEV)
    with torch.no_grad():
        _, l1, _ = m(ids, labels=ids)
        with autocast_ctx(DEV, "off"):
            _, l2, _ = m(ids, labels=ids)
    assert torch.equal(l1, l2)

def test_fp16_gradients_are_finite_through_the_recurrence():
    """The end-to-end symptom of the above: finite forward, non-finite backward."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).to(DEV)
    ids = torch.randint(0, cfg.vocab, (2, 32), device=DEV)
    with autocast_ctx(DEV, "fp16"):
        _, loss, _ = m(ids, labels=ids)
    assert torch.isfinite(loss)
    loss.backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients under fp16: {bad[:5]}"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok  ", _name)
    print()
    print("all passed (CUDA)")
