"""Trains the model: B1 -> B2(routed bank) -> B_mid -> B3(routed bank) -> B4 -> head.

There is one architecture, not a staged pretraining plan -- this script always
trains the full routed model.  Runs on CPU, CUDA or TPU: the device layer is the
only thing that branches.

    python train.py --domain babylm --tokenizer tokenizer.json

Multi-GPU (cloud rental): launch with torchrun, no extra flag needed --
DDP is auto-detected from the environment variables torchrun sets.

    torchrun --nproc_per_node=4 train.py --domain babylm --tokenizer tokenizer.json
"""
import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cinnamon.config import Config
from cinnamon.ewc import EWC
from cinnamon.model import CinnamonModel


def get_device():
    """(kind, device).  XLA is detected by import, not by a flag, so the same
    script runs unchanged on a Kaggle TPU."""
    try:
        import torch_xla.core.xla_model as xm
        return "xla", xm.xla_device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


def ddp_info():
    """(rank, world_size, local_rank, is_ddp), read from torchrun's env vars.

    Auto-detected rather than a flag: a script launched plain (`python train.py`)
    and one launched `torchrun --nproc_per_node=N train.py` should be the same
    file with no mode switch to remember.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 1, 0, False
    return int(os.environ["RANK"]), world_size, int(os.environ["LOCAL_RANK"]), True


def build_blocks(domain, tok_path, seq_len, cache_dir, log=print, limit_mb=None):
    """Tokenise and pack, caching token blocks so a rerun skips the work."""
    from tokenizers import Tokenizer

    from cinnamon.data import babylm_text, corpus_iter, dev_iter, pack

    os.makedirs(cache_dir, exist_ok=True)
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id("<eos>")
    train_bytes = (limit_mb or 2048) * 2**20
    out = {}
    for split in ("train", "dev"):
        f = os.path.join(cache_dir, f"{domain}.{split}.{seq_len}.npy")
        if os.path.exists(f):
            out[split] = np.load(f, mmap_mode="r")
            log(f"{split}: {len(out[split])} blocks (cached)")
            continue
        if domain == "babylm":
            # Official organiser-released splits: train and dev live in separate
            # repos.  Never random-split the train file (see cinnamon/data.py).
            texts = babylm_text(split, log=log)
        elif split == "dev":
            texts = dev_iter(domain, train_bytes, log=log)
        else:
            texts = corpus_iter(domain, max_bytes=train_bytes, log=log)
        blocks = pack(texts, tok, seq_len, eos, log=log)
        np.save(f, blocks)
        out[split] = blocks
        log(f"{split}: {len(blocks)} blocks, {len(blocks)*seq_len/1e6:.1f}M tokens")
    return out, tok


def batches(blocks, bs, device, seed=0, shuffle=True, epochs=10**9):
    g = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = g.permutation(len(blocks)) if shuffle else np.arange(len(blocks))
        for i in range(0, len(idx) - bs + 1, bs):
            sel = np.sort(idx[i:i + bs])                 # mmap wants ascending reads
            b = torch.from_numpy(np.asarray(blocks[sel], dtype=np.int64))
            yield b[:, :-1].to(device), b[:, 1:].to(device)


def gate_health(model):
    """Worst per-chunk |gcum| over every KDA in the model.

    KDA's chunked path silently stops matching the exact recurrence once this
    passes ~10 -- no NaN, no loss spike, just a quietly wrong attention.  Cheap
    to watch (one already-computed scalar per KDA), so it is worth watching.
    """
    from cinnamon.kda import KDA
    vals = [m._max_gcum for m in model.modules() if isinstance(m, KDA)]
    return float(max(float(v) for v in vals)) if vals else 0.0


def save_ckpt(path, payload):
    """Write atomically, so a crash mid-write cannot destroy the last good file.

    torch.save straight onto `path` truncates it first: interrupt it there and
    both the new and the previous checkpoint are gone.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)              # atomic on the same filesystem


def resolve_amp(mode, kind):
    """Pick a precision that the hardware can actually execute.

    'auto' matters because the target GPUs differ in what they support:
      T4    (sm_75, Turing) -- no bf16, no TF32; fp16 is its only tensor-core path
      P100  (sm_60, Pascal) -- no tensor cores at all, fp32 is the only sane option
      A100+ (sm_80, Ampere) -- bf16, and fp32 matmuls already run on TF32

    Asking for bf16 on a T4 silently falls back to something slower than fp16
    rather than erroring, which is exactly the kind of thing that looks like it
    worked.
    """
    if kind != "cuda":
        return "off" if mode in ("auto", "off") else mode
    if mode != "auto":
        return mode
    # is_bf16_supported() reports True on a T4 because it counts *emulated* bf16,
    # which is slower than fp16 and saves nothing.  Native bf16 starts at Ampere.
    return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def autocast_ctx(kind, amp):
    """bf16 shares fp32's exponent range, so it needs no GradScaler.  fp16 does --
    see the scaler in main(); its exponent range genuinely underflows gradients.
    """
    if amp in (None, False, "off"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    if kind in ("cuda", "cpu"):
        return torch.autocast(kind, dtype=dtype)
    return contextlib.nullcontext()      # XLA/TPU has its own precision path


def evaluate(model, blocks, bs, device, max_batches=40, kind="cuda", amp="off"):
    model.eval()
    tot, n, d2s, d3s = 0.0, 0, 0.0, 0.0
    with torch.no_grad(), autocast_ctx(kind, amp):
        for i, (x, y) in enumerate(batches(blocks, bs, device, shuffle=False, epochs=1)):
            if i >= max_batches:
                break
            _, loss, (d2, d3) = model(x, labels=y)
            tot += float(loss)
            d2s += float(d2.float().mean())
            d3s += float(d3.float().mean())
            n += 1
    model.train()
    return (tot / n, (d2s / n, d3s / n)) if n else (float("nan"), (0.0, 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="babylm")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--c-max2", type=int, default=32, help="recurrence cap per block position")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="ckpt")
    p.add_argument("--limit-mb", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--ewc-lambda", type=float, default=5000.0)
    p.add_argument("--prev-ewc", default=None, help="EWC state from the previous domain")
    p.add_argument("--no-ewc-consolidate", action="store_true")
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; the step count is re-fit to it after a "
                        "timing probe so the cosine schedule always anneals fully")
    p.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"],
                   help="mixed precision; 'auto' picks bf16 on Ampere+, fp16 on T4")
    a = p.parse_args()

    rank, world_size, local_rank, is_ddp = ddp_info()
    p0 = rank == 0            # only rank 0 logs, evals, checkpoints, consolidates EWC
    if is_ddp:
        assert torch.cuda.is_available(), "DDP path is CUDA-only"
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        kind, device = "cuda", torch.device(f"cuda:{local_rank}")
    else:
        kind, device = get_device()
    amp = resolve_amp(a.amp, kind)
    # fp16 needs loss scaling: its exponent range is narrow enough that small
    # gradients flush to zero before the optimizer ever sees them.  bf16 does not.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    if p0:
        print(f"device: {kind} {device}  world_size {world_size}  amp {amp}", flush=True)
    os.makedirs(a.out, exist_ok=True)

    # Every torchrun process would otherwise race to write the same pack cache
    # file.  Rank 0 packs first; everyone else waits, then hits the now-warm cache.
    if p0:
        data, tok = build_blocks(a.domain, a.tokenizer, a.seq_len, a.cache, limit_mb=a.limit_mb)
    if is_ddp:
        dist.barrier()
    if not p0:
        data, tok = build_blocks(a.domain, a.tokenizer, a.seq_len, a.cache, limit_mb=a.limit_mb)

    cfg = Config(vocab=tok.get_vocab_size(), c_max2=a.c_max2)
    raw_model = CinnamonModel(cfg).to(device)
    if is_ddp:
        # Every rank must start from identical weights, or the all-reduced
        # gradient is averaging updates to different models.
        for t in raw_model.state_dict().values():
            dist.broadcast(t, src=0)
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    if p0:
        rows, total, _ = raw_model.param_report()
        for k, v in rows:
            print(f"  {k:32s} {v/1e6:8.2f} M", flush=True)
        print(f"  {'total':32s} {total/1e6:8.2f} M", flush=True)

    ewc = EWC(lam=a.ewc_lambda)
    if a.prev_ewc:
        ewc.load_state_dict(torch.load(a.prev_ewc, map_location=device))
        if p0:
            print(f"EWC anchored to {len(ewc.anchor)} shared tensors", flush=True)

    # phi is the learned step embedding: 2-D, so it would land in the decay group
    # by shape, but decaying it toward zero pulls every recurrence's code back onto
    # the same point and erases the depth signal it exists to carry.
    nodec = lambda n, q: q.dim() < 2 or n.endswith("phi")
    decay = [q for n, q in raw_model.named_parameters() if not nodec(n, q)]
    nodecay = [q for n, q in raw_model.named_parameters() if nodec(n, q)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    # Auto-resume: default to this domain's own last.pt so an interrupted run is
    # continued by re-running the identical command, with no flag to remember.
    start, best_dev = 0, float("inf")
    resume = a.resume or os.path.join(a.out, f"{a.domain}.last.pt")
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        best_dev = ck.get("best_dev", ck.get("dev_loss", float("inf")))
        if p0:
            print(f"resumed from {resume} at step {start} (best dev {best_dev:.4f})", flush=True)

    # Disjoint shard per rank -- each GPU sees different data, not a replay of
    # the same stream, which is what makes the all-reduced gradient meaningful.
    shard = data["train"][rank::world_size] if is_ddp else data["train"]
    it = batches(shard, a.batch, device, seed=rank)
    hist, t0 = [], time.time()
    model.train()

    # Throughput is not known until it runs, and a run cut off mid-cosine leaves a
    # model that was never annealed.  With --max-hours, time a short steady window
    # and re-fit the step count to the budget.  Under DDP every rank must land on
    # the same number: disagree by one step and the next collective deadlocks.
    budget, t_probe, PROBE = a.steps, None, 40
    sched = lambda s, n=None: (s / max(1, a.warmup) if s < a.warmup else
                               0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) /
                                                          max(1, (n or a.steps) - a.warmup)))))

    step = start - 1
    while True:
        step += 1
        if step >= budget:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step, budget)
        opt.zero_grad(set_to_none=True)
        lm = 0.0
        for i in range(a.accum):
            x, y = next(it)
            with autocast_ctx(kind, amp):
                _, loss, (d2, d3) = model(x, labels=y)
            lm += float(loss.detach())
            # Only all-reduce gradients on the accum step that actually optimizes;
            # no_sync() skips DDP's hook on every earlier micro-batch.
            last = i == a.accum - 1
            sync = model.no_sync() if (is_ddp and not last) else contextlib.nullcontext()
            with sync:
                scaler.scale((loss + ewc.penalty(raw_model)) / a.accum).backward()
        # unscale before clipping, or the clip threshold is applied to scaled grads
        if amp == "fp16":
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.clip)

        if kind == "xla":
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(opt, barrier=True)
        else:
            scaler.step(opt)
            scaler.update()

        if a.max_hours and step == start + 2:
            t_probe = time.time()
        if a.max_hours and step == start + PROBE and t_probe:
            per = (time.time() - t_probe) / (PROBE - 2)
            fits = torch.tensor([float((a.max_hours*3600 - (time.time()-t0)) / per)],
                                device=device)
            if is_ddp:
                dist.all_reduce(fits, op=dist.ReduceOp.MIN)     # in-place, returns None
            budget = max(a.warmup + 200, min(a.steps, int(fits.item()) + step))
            if p0:
                tok = budget * a.batch * a.accum * world_size * a.seq_len
                print(f"  >> {per:.2f}s/step -> budget {budget} steps ({tok/1e6:.0f}M tokens, "
                      f"{tok/(len(data['train'])*a.seq_len):.2f} epochs) in {a.max_hours}h",
                      flush=True)

        if p0 and step % a.log_every == 0:
            lm /= a.accum
            tok_s = (step - start + 1) * a.batch * a.accum * a.seq_len * world_size / (time.time() - t0)
            # Depth is per token (7), so log the spread and not just the mean.
            # A mean alone reads as a plausible number even when every token is
            # pinned at the cap -- which is exactly how a halting rule that never
            # fired went unnoticed until the std turned out to be 0.00.
            print(f"step {step:6d}  loss {lm:.4f}  ppl {math.exp(min(20, lm)):9.2f}  "
                  f"gnorm {float(gn):.2f}  depth {float(d2.float().mean()):.1f}"
                  f"[{int(d2.min())}-{int(d2.max())}]/{float(d3.float().mean()):.1f}"
                  f"[{int(d3.min())}-{int(d3.max())}]  {tok_s:.0f} tok/s (global)", flush=True)
            st = getattr(raw_model, "aux_stats", None)
            if st:
                print(f"     aux {st['aux']:.3f}  entropy {st['entropy']:.3f}  "
                      f"frac {st['frac']}", flush=True)
            hist.append({"step": step, "loss": lm, "gnorm": float(gn), "routing": st})

        if p0 and a.eval_every and step and step % a.eval_every == 0:
            vl, (vd2, vd3) = evaluate(raw_model, data["dev"], a.batch, device, kind=kind, amp=amp)
            gh = gate_health(raw_model)
            warn = "  !! KDA gate drift, chunked path degrading" if gh > 10.0 else ""
            print(f"  == dev loss {vl:.4f}  ppl {math.exp(min(20, vl)):.2f}  "
                  f"depth {vd2:.1f}/{vd3:.1f}  gcum {gh:.1f}{warn}", flush=True)
            hist.append({"step": step, "dev_loss": vl, "dev_depth": [vd2, vd3], "gcum": gh})

            payload = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                       "step": step + 1, "cfg": cfg.__dict__, "dev_loss": vl,
                       "best_dev": min(vl, best_dev)}
            # last.pt always: the resume point.  best.pt only on improvement, and
            # written to a separate file so a later worse eval cannot overwrite it.
            save_ckpt(os.path.join(a.out, f"{a.domain}.last.pt"), payload)
            if vl < best_dev:
                best_dev = vl
                save_ckpt(os.path.join(a.out, f"{a.domain}.best.pt"), payload)
                print(f"     new best dev {vl:.4f} -> {a.domain}.best.pt", flush=True)
            json.dump(hist, open(os.path.join(a.out, f"{a.domain}.hist.json"), "w"))
        if is_ddp:
            dist.barrier()        # keep ranks from drifting apart around the p0-only eval/save

    if p0:
        vl, (vd2, vd3) = evaluate(raw_model, data["dev"], a.batch, device, kind=kind, amp=amp)
        print(f"FINAL dev loss {vl:.4f}  ppl {math.exp(min(20, vl)):.2f}  "
              f"depth {vd2:.1f}/{vd3:.1f}  gcum {gate_health(raw_model):.1f}", flush=True)
        payload = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                   "step": step + 1, "cfg": cfg.__dict__, "dev_loss": vl,
                   "best_dev": min(vl, best_dev)}
        save_ckpt(os.path.join(a.out, f"{a.domain}.last.pt"), payload)
        if vl < best_dev:
            best_dev = vl
            save_ckpt(os.path.join(a.out, f"{a.domain}.best.pt"), payload)
        print(f"best dev {best_dev:.4f} -> {a.domain}.best.pt", flush=True)

        # 9 steps 5-6: hand the next domain a Fisher-weighted anchor on the scaffold.
        if not a.no_ewc_consolidate:
            ewc.consolidate(raw_model, batches(data["train"], a.batch, device, seed=1))
            torch.save(ewc.state_dict(), os.path.join(a.out, f"{a.domain}.ewc.pt"))
        json.dump(hist, open(os.path.join(a.out, f"{a.domain}.hist.json"), "w"))
        print("done", flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
