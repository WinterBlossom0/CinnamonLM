"""Kernel body for training on Kaggle's 2x T4 GPUs.

Not imported locally -- push.py reads this as text, prepends the source-locating
prelude, and substitutes __ARGS__.

This is deliberately a thin wrapper rather than a second copy of the training
loop.  An earlier version reimplemented the loop here and had a real bug for it:
torch.multiprocessing.spawn re-imports __main__ in every child, so each worker
would have re-run this whole file -- repacking the corpus, and re-spawning.
torchrun exists to do this correctly, and train.py already reads the env vars it
sets, so the only job left here is to pack once up front and then shell out.

T4 is Turing (sm_75): no bf16, no TF32, and fp16 cannot hold this model's
gradients (1e3-1e5 against a 65504 ceiling).  So amp is passed as "off" and the
run is fp32 -- resolve_amp()'s fp16 default is deliberately overridden.
"""
import json
import os
import subprocess
import sys
import time

ARGS = json.loads('__ARGS__')
ARGS["tokenizer"] = ARGS["tokenizer"] or os.path.join(SRC, "tokenizer.json")  # noqa: F821
OUT, CACHE = "/kaggle/working", "/kaggle/working/cache"
print("args:", ARGS, flush=True)

import torch
N = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", N,
      "|", [torch.cuda.get_device_name(i) for i in range(N)], flush=True)

# Pack once, here, before any worker starts: N processes tokenising the same
# 543 MB corpus would only race each other for the same cache file.
# If the pre-packed dataset is attached, seed the cache from it instead -- the
# pack is deterministic given (corpus, tokenizer, seq_len), and re-deriving it
# costs 12 minutes of a session that is already wall-clock limited.
import glob
import shutil

os.makedirs(CACHE, exist_ok=True)
for f in glob.glob("/kaggle/input/**/babylm.*.512.npy", recursive=True):
    dst = os.path.join(CACHE, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy(f, dst)
        print("seeded cache from", f, flush=True)

from train import build_blocks

t0 = time.time()
data, tok = build_blocks(ARGS["domain"], ARGS["tokenizer"], ARGS["seq_len"], CACHE)
print(f"packed in {(time.time()-t0)/60:.1f} min | vocab {tok.get_vocab_size()} | "
      f"train {len(data['train'])} blocks, dev {len(data['dev'])}", flush=True)
del data, tok

cmd = ([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={N}",
        "--master_port=29500"] if N > 1 else [sys.executable]) + [
    os.path.join(SRC, "train.py"),                                    # noqa: F821
    "--domain", ARGS["domain"],
    "--amp", ARGS["amp"],
    "--tokenizer", ARGS["tokenizer"],
    "--seq-len", str(ARGS["seq_len"]),
    "--batch", str(ARGS["batch"]),
    "--accum", str(ARGS["accum"]),
    "--c-max2", str(ARGS["c_max2"]),
    "--steps", str(ARGS["steps"]),
    "--lr", str(ARGS["lr"]),
    "--warmup", str(ARGS["warmup"]),
    "--max-hours", str(ARGS["max_hours"]),
    "--eval-every", str(ARGS["eval_every"]),
    "--log-every", str(ARGS["log_every"]),
    "--ewc-lambda", str(ARGS["ewc_lambda"]),
    "--cache", CACHE,
    "--out", OUT,
]
print("+", " ".join(cmd), flush=True)
env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1")           # noqa: F821
raise SystemExit(subprocess.run(cmd, env=env, cwd=SRC).returncode)     # noqa: F821
