"""DDP gradient correctness, CPU-only via the gloo backend.

Only one physical GPU is available here, so real multi-GPU throughput can't be
measured locally -- that needs an actual multi-GPU rental. What CAN be checked
without one: that DistributedDataParallel's gradient all-reduce is actually
correct for this model, i.e. training on 2 ranks with different data per rank
produces the exact same gradient as manually averaging two independent
single-process backward passes over the same two batches. That is the entire
correctness promise DDP makes, and it doesn't need a GPU to verify.

Run: python -m tests.test_ddp
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from cinnamon.config import TINY, Config
from cinnamon.model import CinnamonModel


def _worker(rank, world_size, ids_per_rank, init_state, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    cfg = Config(**TINY, grad_checkpoint=False)
    m = CinnamonModel(cfg)
    m.load_state_dict(init_state)          # identical init on every rank, as train.py broadcasts
    ddp = DDP(m)

    ids = ids_per_rank[rank]
    _, loss, _ = ddp(ids, labels=ids)
    loss.backward()

    if rank == 0:
        torch.save({n: p.grad.clone() for n, p in m.named_parameters()}, out_path)
    dist.destroy_process_group()


def test_ddp_gradient_equals_manual_average(tmp_path=None):
    import tempfile
    tmp_path = tmp_path or tempfile.mkdtemp()
    out_path = os.path.join(tmp_path, "ddp_grad.pt")

    torch.manual_seed(0)
    cfg = Config(**TINY, grad_checkpoint=False)
    init_state = CinnamonModel(cfg).state_dict()

    torch.manual_seed(1)
    ids_per_rank = [torch.randint(0, cfg.vocab, (2, 16)) for _ in range(2)]

    mp.spawn(_worker, args=(2, ids_per_rank, init_state, out_path), nprocs=2, join=True)
    ddp_grad = torch.load(out_path, weights_only=True)

    # Reference: two independent single-process passes, gradients averaged by hand.
    m = CinnamonModel(cfg)
    m.load_state_dict(init_state)
    manual = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for ids in ids_per_rank:
        m.zero_grad(set_to_none=True)
        _, loss, _ = m(ids, labels=ids)
        loss.backward()
        for n, p in m.named_parameters():
            manual[n] += p.grad / 2

    for n in manual:
        assert torch.allclose(ddp_grad[n], manual[n], atol=1e-6), \
            f"{n}: DDP all-reduced grad diverges from the manual average"


def test_ddp_world_size_1_is_a_true_noop():
    """The auto-detect path (WORLD_SIZE unset or 1) must not wrap in DDP at all --
    single-GPU and single-process runs should be byte-identical to before this
    feature existed."""
    import importlib

    import train
    importlib.reload(train)
    os.environ.pop("WORLD_SIZE", None)
    rank, world_size, local_rank, is_ddp = train.ddp_info()
    assert (rank, world_size, local_rank, is_ddp) == (0, 1, 0, False)


if __name__ == "__main__":
    test_ddp_world_size_1_is_a_true_noop()
    print("ok   test_ddp_world_size_1_is_a_true_noop")
    test_ddp_gradient_equals_manual_average()
    print("ok   test_ddp_gradient_equals_manual_average")
    print("\nall passed (CPU-only via gloo, no GPU touched)")
