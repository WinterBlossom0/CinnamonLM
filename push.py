"""Kaggle runs of this project under any Kaggle account (2x T4 per training kernel). KAGGLE.md gives the procedure.

Names, with OWNER the Kaggle username:
    OWNER/recursion-comparison-src         dataset: the source files
    OWNER/recursion-comparison-data        CPU kernel: TinyStories packed into tokens (prepare_data.py), run once
    OWNER/recursion-comparison-<arm>       GPU kernel per arm, e.g. recursion-comparison-grt-x8 for grt_x8
    OWNER/recursion-comparison-smoke       GPU kernel: every check and a shortened run of the given arms

    python push.py build [--owner OWNER]   # folders for the kaggle CLI in kaggle_build/ (no network)
    python push.py src                     # create or version the source dataset, wait until it is ready
    python push.py data                    # push the data kernel
    python push.py train <arm>             # push one arm kernel
    python push.py smoke <arm> [<arm> ...] # push the smoke kernel
    python push.py status <slug> [--owner OWNER]
    python push.py logs <slug> [--owner OWNER]
    python push.py pull <arm|smoke> [--owner OWNER]    # metrics.json, model.pt and the log, into kaggle_output/<name>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

from provenance import code_sha256

ROOT = Path(__file__).parent
BUILD = ROOT / "kaggle_build"
OUTPUT = ROOT / "kaggle_output"
PREFIX = "recursion-comparison"
SRC_FILES = ["train.py", "provenance.py", "check.py", "prepare_data.py", "attention_kernel.py", "hyperloop_kernel.py",
             "grt_kernel.py", "mesh_kernel.py", "requirements.txt", "tokenizer"]
# The arms of train.ARMS, listed here so building needs no torch.
ARMS = ["recursive_x2", "real_6", "recursive_x4", "real_10", "recursive_x8", "real_18", "reinject_x8",
        "hyperloop_x8", "parcae_x8", "grt_x8", "mesh_x8"]
SEEDS = [0, 1, 2]
EPOCHS = 3
SMOKE_STEPS_PER_EPOCH = 40
# Pinned because they decide the tokenization. requirements.txt is part of the code SHA-256, so it stays unchanged.
DATA_PACKAGES = ["datasets==5.0.1", "tokenizers==0.22.2"]


def kernel_slug(name: str) -> str:
    return f"{PREFIX}-{name.replace('_', '-')}"


# Shared start of every kernel: find the source dataset (zip folders unpacked) and copy it to /kaggle/working/src,
# because the scripts write results/ and packed_cache/ next to themselves and /kaggle/input is read-only.
_FIND_SOURCE = '''\
import glob, os, shutil, subprocess, sys, zipfile
for archive in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(archive).extractall("/kaggle/working/_unzipped/" + os.path.basename(archive)[:-4])
hits = glob.glob("/kaggle/input/**/prepare_data.py", recursive=True)
assert hits, f"source dataset not attached; /kaggle/input has {glob.glob('/kaggle/input/*')}"
SRC = "/kaggle/working/src"
shutil.rmtree(SRC, ignore_errors=True)
shutil.copytree(os.path.dirname(hits[0]), SRC)
for unzipped in glob.glob("/kaggle/working/_unzipped/*"):
    target = os.path.join(SRC, os.path.basename(unzipped))
    if not os.path.exists(target):
        shutil.copytree(unzipped, target)
os.chdir(SRC)
print("source:", sorted(os.listdir(SRC)), flush=True)
'''

_LINK_PACKED = '''\
packed = glob.glob("/kaggle/input/**/packed_cache/train.pt", recursive=True)
assert packed, "packed tokens missing: run the recursion-comparison-data kernel first and attach it"
PACKED_LINK = os.path.join(SRC, "packed_cache")
os.symlink(os.path.dirname(packed[0]), PACKED_LINK)
'''

_GPU_PREPARE = '''\
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
# Kaggle's torchvision/torchaudio are built for its own torch; transformers imports torchvision when it is
# installed, and the pinned torch breaks its C extensions ("operator torchvision::nms does not exist").
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchvision", "torchaudio"])
import torch
print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
'''


def gpu_script(body: str) -> str:
    # Kaggle saves /kaggle/working; remove the link also after a failure, so the output cannot copy the 3.3 GB tokens.
    return (_FIND_SOURCE + _LINK_PACKED + "try:\n" + textwrap.indent(_GPU_PREPARE + body, "    ")
            + "finally:\n    os.unlink(PACKED_LINK)\n")


def data_script() -> str:
    return _FIND_SOURCE + f'''\
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *{DATA_PACKAGES!r}])
subprocess.check_call([sys.executable, "prepare_data.py", "--out", "/kaggle/working/packed_cache"])
shutil.rmtree(SRC)
shutil.rmtree("/kaggle/working/_unzipped", ignore_errors=True)
'''


def train_script(arm: str) -> str:
    return gpu_script(f'''\
# The arm's checks on the Kaggle GPU first, so a fault costs minutes, not the run.
subprocess.check_call([sys.executable, "check.py", {arm!r}])
subprocess.check_call([sys.executable, "train.py", "--arm", {arm!r}, "--epochs", "{EPOCHS}", "--seeds",
                       *{list(map(str, SEEDS))!r}])
for seed in {SEEDS!r}:
    print(open(f"results/{arm}/seed_{{seed}}/metrics.json").read(), flush=True)
''')


def smoke_script(arms: list[str]) -> str:
    return gpu_script(f'''\
import time
subprocess.check_call([sys.executable, "check.py", *{arms!r}])
for arm in {arms!r}:
    # Two rounds of the real path: seeds 0 and 1 on one GPU each, seed 2 split over both GPUs.
    start = time.time()
    subprocess.check_call([sys.executable, "train.py", "--arm", arm, "--epochs", "2", "--seeds",
                           *{list(map(str, SEEDS))!r}, "--max-steps-per-epoch", "{SMOKE_STEPS_PER_EPOCH}"])
    for seed in {SEEDS!r}:
        metrics = open(f"results_smoke/{{arm}}/seed_{{seed}}/metrics.json").read()
        print(metrics, flush=True)
    print(f"smoke {{arm}}: {{time.time() - start:.0f}} s", flush=True)
subprocess.check_call([sys.executable, "speed.py", *{arms!r}])
''')


def kernel_metadata(owner: str, slug: str, gpu: bool) -> dict:
    return {
        # The title must slugify to the same id, or Kaggle gives the kernel the title's slug.
        "id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [f"{owner}/{PREFIX}-src"], "competition_sources": [],
        "kernel_sources": [f"{owner}/{PREFIX}-data"] if gpu else [], "model_sources": [],
        # T4 x2: PyTorch dropped sm_60, so Kaggle's P100 cannot run the pinned torch.
        "machine_shape": "NvidiaTeslaT4" if gpu else "",
    }


def write_kernel(owner: str, name: str, script: str, gpu: bool) -> Path:
    folder = BUILD / "kernels" / kernel_slug(name)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(script)
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(owner, kernel_slug(name), gpu), indent=1))
    return folder


def build(owner: str, smoke_arms: list[str]) -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    src = BUILD / "src"
    src.mkdir(parents=True)
    for name in SRC_FILES + ["speed.py"]:
        path = ROOT / name
        if path.is_dir():
            shutil.copytree(path, src / name)
        else:
            shutil.copy(path, src / name)
    (src / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"{PREFIX}-src", "id": f"{owner}/{PREFIX}-src", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    write_kernel(owner, "data", data_script(), gpu=False)
    for arm in ARMS:
        write_kernel(owner, arm, train_script(arm), gpu=True)
    write_kernel(owner, "smoke", smoke_script(smoke_arms), gpu=True)
    print(f"built {BUILD} for owner {owner}", flush=True)


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    return kaggle_api


def own_username() -> str:
    return api().get_config_value("username")


def dataset_ready(kaggle_api, ref: str) -> bool:
    try:
        return "ready" in str(kaggle_api.dataset_status(ref)).lower()
    except Exception:
        return False


def push_src(owner: str) -> None:
    kaggle_api = api()
    ref = f"{owner}/{PREFIX}-src"
    if dataset_ready(kaggle_api, ref):
        print(kaggle_api.dataset_create_version(str(BUILD / "src"), "update", dir_mode="zip", quiet=True))
    else:
        print(kaggle_api.dataset_create_new(str(BUILD / "src"), public=False, dir_mode="zip", quiet=True))
    # A kernel launched right after a new dataset version can start with an empty /kaggle/input, so wait.
    time.sleep(10)
    for _ in range(60):
        if dataset_ready(kaggle_api, ref):
            print(f"{ref} ready", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"{ref} did not report ready after 300 s")


def push_kernel(name: str) -> None:
    result = api().kernels_push(str(BUILD / "kernels" / kernel_slug(name)))
    print("pushed:", getattr(result, "url", result), "error:", getattr(result, "error", None), flush=True)


def pull(owner: str, name: str) -> None:
    """Downloads the outputs of a kernel and warns for each metrics.json whose code is not the local code."""
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    files, _ = api().kernels_output(f"{owner}/{kernel_slug(name)}", str(folder),
                                    file_pattern=r".*(metrics\.json|model\.pt|\.log)$")
    print("files:", files)
    local = code_sha256()
    for path in sorted(folder.rglob("metrics.json")):
        recorded = json.loads(path.read_text()).get("code_sha256")
        if recorded != local:
            print(f"WARNING: {path} has code_sha256 {recorded}, the local code has {local}; its run used other code",
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "src", "data", "train", "smoke", "status", "logs", "pull"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--owner", help="Kaggle username; default: the username of the credentials")
    args = parser.parse_args()
    owner = args.owner or own_username()
    smoke_arms = ["reinject_x8", "parcae_x8", "grt_x8", "mesh_x8"]
    if args.command == "build":
        build(owner, smoke_arms)
    elif args.command == "src":
        build(owner, smoke_arms)
        push_src(owner)
    elif args.command == "data":
        build(owner, smoke_arms)
        push_kernel("data")
    elif args.command == "train":
        unknown = [arm for arm in args.names if arm not in ARMS]
        if len(args.names) != 1 or unknown:
            parser.error(f"give exactly one arm of {ARMS}")
        build(owner, smoke_arms)
        push_kernel(args.names[0])
    elif args.command == "smoke":
        unknown = [arm for arm in args.names if arm not in ARMS]
        if unknown:
            parser.error(f"unknown arms {unknown}")
        build(owner, args.names or smoke_arms)
        push_kernel("smoke")
    elif args.command == "status":
        print(api().kernels_status(f"{owner}/{args.names[0]}"))
    elif args.command == "logs":
        for event in api().kernels_logs_stream(f"{owner}/{args.names[0]}"):
            print(event.get("data", ""), end="", flush=True)
    elif args.command == "pull":
        if len(args.names) != 1 or args.names[0] not in ARMS + ["smoke"]:
            parser.error(f"give exactly one of {ARMS + ['smoke']}")
        pull(owner, args.names[0])


if __name__ == "__main__":
    sys.exit(main())
