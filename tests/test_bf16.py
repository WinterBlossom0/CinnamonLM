"""bf16 autocast correctness, CPU-only -- no GPU required or touched.

torch.autocast("cpu", dtype=torch.bfloat16) exercises the identical dtype-dispatch
logic autocast uses on CUDA, just without hardware acceleration.  That is enough to
catch what actually breaks under lower precision: NaN/inf, or the loss diverging
from the fp32 run.  It cannot measure speed -- only cinnamon/kda.py's fp32 promotion
already showed once that "should be numerically fine" is not something to assume
on this architecture without checking.

Run: python -m tests.test_bf16
"""
import torch

from cinnamon.config import TINY, Config
from cinnamon.model import CinnamonModel
from train import autocast_ctx


def test_bf16_forward_backward_finite():
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg)
    ids = torch.randint(0, cfg.vocab, (2, 32))
    with autocast_ctx("cpu", "bf16"):
        logits, loss, (d2, d3) = m(ids, labels=ids)
    assert torch.isfinite(loss), float(loss)
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients under bf16: {bad}"


def test_bf16_close_to_fp32_at_init():
    """Same weights, same batch: bf16 and fp32 should agree to bf16's own
    precision floor (~1e-2 relative for a stack this deep), not diverge outright."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab, (2, 32))
    with torch.no_grad():
        _, loss_fp32, _ = m(ids, labels=ids)
        with autocast_ctx("cpu", "bf16"):
            _, loss_bf16, _ = m(ids, labels=ids)
    rel = abs(float(loss_bf16) - float(loss_fp32)) / abs(float(loss_fp32))
    assert rel < 0.05, f"fp32 loss {float(loss_fp32):.4f} vs bf16 {float(loss_bf16):.4f} (rel {rel:.3f})"


def test_bf16_kda_survives_the_same_adversarial_case_that_broke_fp32():
    """The exact regime that overflowed the UT-transform inverse in fp32 at
    chunk=64 (see test_model.test_kda_survives_correlated_keys): beta -> 1,
    near-zero decay, correlated keys within a chunk.  KDA promotes its internal
    math to at least fp32 regardless of the input dtype (cinnamon/kda.py), so this
    must hold even when the surrounding model runs under bf16 autocast.
    """
    from cinnamon.kda import KDA

    torch.manual_seed(0)
    k = KDA(Config(**TINY))
    with torch.no_grad():
        k.beta.bias.fill_(6.0)
        k.g_up.bias.fill_(-12.0)
    base = torch.randn(1, 1, TINY["d_model"])
    x = base.expand(2, 96, TINY["d_model"]) + 0.01 * torch.randn(2, 96, TINY["d_model"])
    with autocast_ctx("cpu", "bf16"):
        out = k(x)
    assert torch.isfinite(out).all(), "KDA overflowed under bf16 autocast"


def test_bf16_does_not_change_eval_when_disabled():
    """bf16=False must be a true no-op, not an accidental always-on cast."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab, (2, 16))
    with torch.no_grad():
        _, l1, _ = m(ids, labels=ids)
        with autocast_ctx("cpu", "off"):
            _, l2, _ = m(ids, labels=ids)
    assert torch.equal(l1, l2)

def test_autocast_never_downcasts_kda_chunk_math():
    """KDA promotes its chunk math to fp32 on purpose, but autocast intercepts
    matmuls *after* that cast and puts them back to fp16 unless explicitly
    disabled.  That silently broke fp16 training: the centred gate factors reach
    exp(+-10) = 22026, and a product of two is 4.9e8 against fp16's 65504 ceiling,
    so backward produced inf, GradScaler halved forever (32768 -> 128 over 9
    steps) and the loss never moved -- with a perfectly finite forward pass, so
    nothing ever raised.
    """
    from cinnamon.kda import KDA

    k = KDA(Config(**TINY))
    x = torch.randn(1, 32, TINY["d_model"])
    seen = {}
    orig = torch.Tensor.__matmul__

    def spy(self, other):
        r = orig(self, other)
        seen[r.dtype] = seen.get(r.dtype, 0) + 1
        return r

    torch.Tensor.__matmul__ = spy
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            out = k(x)
    finally:
        torch.Tensor.__matmul__ = orig
    assert torch.float16 not in seen, f"chunk math ran in fp16: {seen}"
    assert torch.isfinite(out).all()


def test_fp16_gradients_are_finite_through_the_recurrence():
    """The end-to-end symptom of the above: finite forward, non-finite backward."""
    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg)
    ids = torch.randint(0, cfg.vocab, (2, 32))
    with autocast_ctx("cpu", "fp16"):
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
    print("all passed (CPU-only, no GPU touched)")
