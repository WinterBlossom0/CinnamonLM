"""Upload this directory to Kaggle as a dataset and launch a training kernel on it.

TinyStories is NOT shipped as a dataset: .gitignore already pins the exact
reproduction command (revision-pinned `load_dataset`, so a future Hub-side
update can't silently change the corpus), and the kernel just runs that
directly -- avoids a 1.8 GB upload for something already reproducible exactly.

requirements.txt is pip-installed inside the kernel (needs enable_internet
regardless, for the Hub download above): the KDA, OSDN and Parallax Triton kernels were
verified locally at torch==2.13.0, triton==3.7.1 specifically -- Kaggle's base
image is not guaranteed to match, and a silent version drift in Triton
semantics is exactly the kind of thing that would not show up as an error,
just a wrong number.

    python push.py src                    # upload / version the source dataset
    python push.py train kda              # launch training for one attention type
    python push.py train kda --seeds 0 1 2 --epochs 3
    python push.py train osdn --seeds 0 --suffix=-s0   # one kernel per seed
    python push.py status <slug>
    python push.py pull <slug> <dir>
"""
import argparse
import json
import os
import shutil
import sys
import time

USER = "electroknight"
DATASET = f"{USER}/attn-comparison-src"
ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(ROOT, ".kaggle_stage")
# Every top-level module, globbed rather than listed: an explicit list silently drops a newly added module.
SRC_FILES = ["kda", "osdn", "parallax", "mamba3", "tokenizer", "requirements.txt"] + \
    sorted(f for f in os.listdir(ROOT)
           if f.endswith(".py") and f != os.path.basename(__file__))
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned in .gitignore


def api():
    """The console script is not on PATH on Windows and `python -m kaggle` has no
    __main__, so both fail silently.  Use the library directly."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def _exists(a, ref):
    try:
        return "ready" in str(a.dataset_status(ref)).lower()
    except Exception:
        return False


def push_src():
    src = os.path.join(STAGE, "src")
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(src)
    for f in SRC_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(src, f))
    for pkg in ("kda", "osdn", "parallax", "mamba3"):
        shutil.rmtree(os.path.join(src, pkg, "__pycache__"), ignore_errors=True)
    json.dump({"title": "attn-comparison-src", "id": DATASET,
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(src, "dataset-metadata.json"), "w"), indent=1)

    a = api()
    if _exists(a, DATASET):
        print(a.dataset_create_version(src, "update", dir_mode="zip", quiet=True))
    else:
        print(a.dataset_create_new(src, public=False, dir_mode="zip", quiet=True))

    # A kernel launched immediately after dataset_create_new can start with an empty /kaggle/input/,
    # so poll the dataset status rather than trust the create call's return.
    for _ in range(30):
        if _exists(a, DATASET):
            return
        time.sleep(5)
    raise RuntimeError(f"{DATASET} did not report ready after 150s")


def push_kernel(slug, title, run_py, *, gpu=True, internet=True, extra_data=()):
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
        # T4: PyTorch dropped sm_60, so Kaggle's P100 cannot run the pinned torch.
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    r = api().kernels_push(d)
    print("pushed:", getattr(r, "url", r), "error:", getattr(r, "error", None), flush=True)
    print(f"  https://www.kaggle.com/code/{USER}/{slug}", flush=True)


# Locate the mounted source by kda/attention.py, a file unique to this project; train.py then runs unmodified.
# The source is always copied to /kaggle/working: train.py writes results/, tinystories/ and packed_cache/
# next to itself, and /kaggle/input is read-only.
PRELUDE = f'''\
import sys, os, glob, zipfile, shutil, subprocess
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/kaggle/working/_" + os.path.basename(z)[:-4])
hits = glob.glob("/kaggle/input/**/kda/attention.py", recursive=True) + \\
       glob.glob("/kaggle/working/**/kda/attention.py", recursive=True)
assert hits, f"source not found; /kaggle/input = {{glob.glob('/kaggle/input/**', recursive=True)[:40]}}"
FOUND = os.path.dirname(os.path.dirname(hits[0]))
SRC = "/kaggle/working/src"
if FOUND != SRC:
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.copytree(FOUND, SRC)
sys.path.insert(0, SRC)
print("source:", SRC, os.listdir(SRC), flush=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                       os.path.join(SRC, "requirements.txt")])
# Kaggle's image ships torchvision/torchaudio built against its own torch; the
# pin above replaces torch under them, so their C extensions no longer register
# ("RuntimeError: operator torchvision::nms does not exist").  transformers
# reaches them through get_cosine_schedule_with_warmup -> trainer_utils -> peft
# -> BloomPreTrainedModel -> image_utils, so a stale torchvision takes down the
# import train.py needs.  Nothing here uses either package; removing them makes
# transformers' is_torchvision_available() (a find_spec, no import) say no.
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-q", "-y",
                       "torchvision", "torchaudio"])

# Same revision-pinned load .gitignore documents for local reproduction --
# not shipped as a dataset, since this is exactly as reproducible and 1.8 GB
# smaller to launch.
from datasets import load_dataset
load_dataset("roneneldan/TinyStories", revision={TINYSTORIES_REVISION!r}
             ).save_to_disk(os.path.join(SRC, "tinystories"))
'''


def train_run(attention, seeds, epochs):
    argv = ["train.py", "--attention", attention, "--epochs", str(epochs),
            "--seeds", *[str(s) for s in seeds]]
    return PRELUDE + f'''
os.chdir(SRC)
sys.argv = {argv!r}
import train
train.main()
'''


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "src"
    if cmd == "src":
        push_src()
    elif cmd == "train":
        parser = argparse.ArgumentParser(prog="push.py train")
        parser.add_argument("attention")
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--epochs", type=int, default=3)
        # Kaggle stops a session at 12 h; a suffix gives each seed its own kernel when three seeds exceed that.
        parser.add_argument("--suffix", default="")
        args = parser.parse_args(sys.argv[2:])
        push_src()
        run = train_run(args.attention, args.seeds, args.epochs)
        # The title must slugify to the same id, or Kaggle silently gives the kernel the title's slug.
        slug = f"attn-comparison-{args.attention}{args.suffix}"
        push_kernel(slug, slug, run)
    elif cmd == "status":
        print(api().kernels_status(f"{USER}/{sys.argv[2]}"))
    elif cmd == "pull":
        os.makedirs(sys.argv[3], exist_ok=True)
        files, log = api().kernels_output(f"{USER}/{sys.argv[2]}", sys.argv[3])
        print("files:", files)
        print(log[-3000:] if log else "(no log)")
    else:
        print(__doc__)
