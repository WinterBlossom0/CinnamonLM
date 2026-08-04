"""Ship source to Kaggle and launch a kernel.

Kaggle kernels cannot see a local folder, so the package is uploaded once as a
private dataset and every kernel mounts it at /kaggle/input/cinnamonlm-src.  The same
dataset serves the tokenizer job and the training job, so there is no second copy
of the code to keep in sync.

    python -m kg.push src                 # upload / version the source dataset
    python -m kg.push tokenizer           # launch the tokenizer kernel
    python -m kg.push packed              # ship the packed corpus as a dataset
    python -m kg.push wikitablet          # ship the extracted WikiTableT text
    python -m kg.push gpu-train           # launch training on 2x T4
    python -m kg.push status <slug>       # poll a kernel
    python -m kg.push pull <slug> <dir>   # download a kernel's output

TPU is gone: the queue never cleared.  Training is single-GPU (see train.py),
so the second T4 of Kaggle's pair sits idle.
"""
import json
import os
import shutil
import sys

USER = "electroknight"
DATASET = f"{USER}/cinnamonlm-src"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, ".kaggle_stage")
SRC_FILES = ["cinnamon", "tokenizer_train.py", "train.py", "tokenizer.json"]


def api():
    """The console script is not on PATH on Windows and `python -m kaggle` has no
    __main__, so both fail silently.  Use the library directly."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def push_src():
    src = os.path.join(STAGE, "src")
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(src)
    for f in SRC_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(src, f))
    shutil.rmtree(os.path.join(src, "cinnamon", "__pycache__"), ignore_errors=True)
    json.dump({"title": "cinnamonlm-src", "id": DATASET,
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(src, "dataset-metadata.json"), "w"), indent=1)

    a = api()
    try:
        exists = "ready" in str(a.dataset_status(DATASET)).lower()
    except Exception:
        exists = False
    if exists:
        print(a.dataset_create_version(src, "update", dir_mode="zip", quiet=True))
    else:
        print(a.dataset_create_new(src, public=False, dir_mode="zip", quiet=True))


def push_data(slug, files):
    """A side dataset for corpora that are not on the Hub (WikiTableT).  Kept out
    of cinnamonlm-src so every code push does not re-upload 150 MB."""
    d = os.path.join(STAGE, slug)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for f in files:
        shutil.copy(f, d)
    ref = f"{USER}/{slug}"
    json.dump({"title": slug, "id": ref, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(d, "dataset-metadata.json"), "w"), indent=1)
    a = api()
    try:
        exists = "ready" in str(a.dataset_status(ref)).lower()
    except Exception:
        exists = False
    print(a.dataset_create_version(d, "update", quiet=True) if exists
          else a.dataset_create_new(d, public=False, quiet=True))
    return ref


def push_kernel(slug, title, run_py, *, gpu=False, internet=True, extra_data=()):
    d = os.path.join(STAGE, slug)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "run.py"), "w").write(run_py)
    meta = {
        "id": f"{USER}/{slug}",
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": [DATASET, *extra_data],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if gpu:
        # enable_gpu alone gets whatever Kaggle defaults to, which was a P100 --
        # and PyTorch 2.10 dropped sm_60, so a P100 cannot run this at all
        # ("supports sm_70 ... sm_120").  T4 is not a preference here, it is the
        # only usable option.  The shape is a pair, but only one is used.
        # The key must be machine_shape; an "accelerator" key is silently ignored.
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    r = api().kernels_push(d)
    print("pushed:", getattr(r, "url", r), "error:", getattr(r, "error", None), flush=True)
    print(f"  https://www.kaggle.com/code/{USER}/{slug}", flush=True)


# Kaggle mounts a zip-mode dataset differently depending on how it was uploaded,
# so locate the source rather than assuming a path, and unpack a zip if that is
# what landed.  Both kernels share this.
PRELUDE = '''\
import sys, os, glob, zipfile, time
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/kaggle/working/_src")
hits = glob.glob("/kaggle/input/**/tokenizer_train.py", recursive=True) + \\
       glob.glob("/kaggle/working/_src/**/tokenizer_train.py", recursive=True)
assert hits, f"source not found; /kaggle/input = {glob.glob('/kaggle/input/**', recursive=True)[:40]}"
SRC = os.path.dirname(hits[0])
sys.path.insert(0, SRC)
print("source:", SRC, os.listdir(SRC), flush=True)
'''

TOKENIZER_RUN = PRELUDE + '''
import tokenizer_train
sys.argv = ["tokenizer_train.py", "--vocab", "128000", "--mb-per-source", "80"]
t0 = time.time()
tokenizer_train.main()
print("total minutes:", (time.time() - t0) / 60, flush=True)
'''


def gpu_train_run(**kw):
    body = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gpu_train.py"), encoding="utf-8").read()
    assert "__ARGS__" in body
    return PRELUDE + body.replace("__ARGS__", json.dumps(kw).replace("'", "\'"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "src"
    if cmd == "src":
        push_src()
    elif cmd == "packed":
        # The pack is deterministic given (corpus, tokenizer, seq_len) and takes
        # 12 min in-kernel.  Shipping it as a dataset makes a restarted run begin
        # training immediately instead of re-tokenising 543 MB.
        push_data("cinnamonlm-packed", [os.path.join(ROOT, "cache", f)
                                     for f in ("babylm.train.512.npy", "babylm.dev.512.npy")])
    elif cmd == "wikitablet":
        push_data("cinnamonlm-wikitablet", [os.path.join(ROOT, "raw", "wikitablet.txt")])
    elif cmd == "tokenizer":
        push_src()
        push_kernel("cinnamonlm-tokenizer", "CinnamonLM tokenizer", TOKENIZER_RUN,
                    extra_data=[f"{USER}/cinnamonlm-wikitablet"])
    elif cmd == "gpu-train":
        push_src()
        # fp32.  Measured on an RTX 5080 at the one-body/8-hypernet config, fp32
        # beats bf16 at batch 4 (325 vs 308 tok/s): the model is still launch-bound
        # enough that autocast knocking RMSNorm off its fused kernel costs more
        # than tensor cores win.  fp16 is separately wrong on a T4 -- gradient
        # norms run 1e3-1e5 against a 65504 ceiling.
        #
        # batch 4 is the measured ceiling at 9.6 GB; batch 6 OOM'd at a 13 GB cap.
        # What runs out is not the model -- it is the logits, [B,512,128000] fp32,
        # 1.6 GB at batch 6 before the .float() copy and its gradient.  Chunked
        # cross-entropy is what unlocks a bigger batch, not a smaller model.
        run = gpu_train_run(domain="babylm", tokenizer=None, seq_len=512, batch=4, accum=2,
                            amp="off",
                            c_max2=int(sys.argv[2]) if len(sys.argv) > 2 else 32,
                            steps=40000, lr=3e-4, warmup=500, max_hours=8.0,
                            ewc_lambda=5000.0, log_every=25, eval_every=500)
        push_kernel("cinnamonlm-gpu", "cinnamonlm gpu", run, gpu=True,
                    extra_data=[f"{USER}/cinnamonlm-packed"])
    elif cmd == "status":
        print(api().kernels_status(f"{USER}/{sys.argv[2]}"))
    elif cmd == "pull":
        os.makedirs(sys.argv[3], exist_ok=True)
        files, log = api().kernels_output(f"{USER}/{sys.argv[2]}", sys.argv[3])
        print("files:", files)
        print(log[-3000:] if log else "(no log)")
    else:
        print(__doc__)
