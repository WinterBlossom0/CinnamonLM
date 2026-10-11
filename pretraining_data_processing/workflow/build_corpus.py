"""Build the commercial-use corpus from the pinned Hugging Face originals, one stage after the other.

Stages, in order: download -> licence (remove non-commercial content) -> clean (source pipeline) -> decontam -> dedup -> publish.
Each stage checks that the build shards equal the output record of the stage before it and writes its own record
to reports/<source>/build/<stage>.json (input and output shard sha256, counts, code sha256).

Usage (run with data_pipeline/.venv/bin/python from any directory):
  build_corpus.py download SOURCE              download one source into build/<source>/ (the source command uses it)
  build_corpus.py source SOURCE [--procs N]    download, licence and clean one source into build/<source>/
  build_corpus.py clean SOURCE [--procs N]     run or resume the clean stage of one source (the pipeline resumes its work dir)
  build_corpus.py decontam-index               build the benchmark 8-gram index
  build_corpus.py decontam [SOURCE] [--procs N]  scan and apply one source, or every source
  build_corpus.py dedup-before-last [--procs N]  check and insert every source but the last; save the filter
  build_corpus.py dedup-last [--procs N]       check the last source against the saved filter, then apply every source
  build_corpus.py publish                      replace data/common_pile/<source>/ with build/<source>/ for every source
  build_corpus.py all [--procs N]              every stage above for every source
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow import cross_dedup, decontam  # noqa: E402
from workflow.corpus import (BUILD, DATA, REPORTS, ROOT, SOURCES, VENV_PYTHON, WORK_ROOT, now_ist,  # noqa: E402
                             read_stage_record, require_input, sha256_of, shard_digests, stage_record_path,
                             write_json_atomically)

# The Xet transfer backend stalled at one connection on this network; plain HTTP downloads the same files (sha256-checked).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
MANIFEST = ROOT / "workflow/raw_manifest.json"
LICENCE_DIR = ROOT / "licence"
LANGUAGE_MODEL = ROOT / "data_pipeline/models/lid.176.bin"


def code_digests(names: list[str]) -> dict[str, str]:
    return {name: sha256_of(ROOT / name) for name in names}


def run(command: list[str]) -> None:
    print(f"{now_ist()} $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})


def require_no_record(source: str, stage: str) -> None:
    if stage_record_path(source, stage).exists():
        raise RuntimeError(f"{source}: stage {stage} has a record already; delete build/{source} and "
                           f"reports/{source}/build to build again")


def download(source: str) -> None:
    from huggingface_hub import snapshot_download
    manifest = json.loads(MANIFEST.read_text())[source]
    if stage_record_path(source, "download").exists():
        # A download made ahead of the other stages is used when the shards still equal its record.
        require_input(source, "download")
        return
    target = BUILD / source
    started = now_ist()
    snapshot_download(manifest["repo"], repo_type="dataset", revision=manifest["revision"], local_dir=target, max_workers=16)
    shutil.rmtree(target / ".cache", ignore_errors=True)
    digests = shard_digests(target)
    expected = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    if digests != expected:
        raise RuntimeError(f"{source}: downloaded shards differ from workflow/raw_manifest.json")
    write_json_atomically(stage_record_path(source, "download"), {
        "stage": "download", "started_at_ist": started, "finished_at_ist": now_ist(), "repo": manifest["repo"],
        "revision": manifest["revision"], "code_sha256": code_digests(["workflow/build_corpus.py", "workflow/raw_manifest.json"]),
        "output_sha256": digests})


def licence(source: str, procs: int) -> None:
    require_no_record(source, "licence")
    inputs = require_input(source, "download")
    out_dir = REPORTS / source / "build/licence"
    started = now_ist()
    run([str(VENV_PYTHON), str(LICENCE_DIR / "licence_filter.py"), str(BUILD / source), "--name", source,
         "--out-dir", str(out_dir), "--workers", str(procs)])
    run([str(VENV_PYTHON), str(LICENCE_DIR / "apply_licence.py"), str(BUILD / source), "--name", source,
         "--actions-dir", str(out_dir), "--report", str(out_dir / "applied.json"), "--workers", str(procs)])
    applied = json.loads((out_dir / "applied.json").read_text())
    write_json_atomically(stage_record_path(source, "licence"), {
        "stage": "licence", "started_at_ist": started, "finished_at_ist": now_ist(), "totals": applied["totals"],
        "summary_checks": applied["summary_checks"],
        "code_sha256": code_digests(["licence/licence_filter.py", "licence/apply_licence.py", "workflow/build_corpus.py"]),
        "actions_sha256": applied["actions_sha256"], "input_sha256": inputs, "output_sha256": shard_digests(BUILD / source)})


def clean(source: str, procs: int) -> None:
    require_no_record(source, "clean")
    inputs = require_input(source, "licence")
    meta = REPORTS / source / "build/clean"
    started = now_ist()
    extra = {}
    if source == "doab":
        # The cross-book boilerplate table is counted on the licence-filtered input; the pipeline reads it.
        table = ROOT / "reports/doab/boilerplate_paragraphs.npy"  # the path that doab.BOILERPLATE_TABLE_PATH reads
        run([str(VENV_PYTHON), "-m", "data_pipeline.sources.doab", "boilerplate-table", "--input", str(BUILD / source),
             "--out", str(table), "--procs", str(procs)])
        extra["boilerplate_table_sha256"] = sha256_of(table)
    run([str(VENV_PYTHON), "-m", f"data_pipeline.sources.{source}", "--input", str(BUILD / source), "--in-place",
         "--meta-out", str(meta), "--procs", str(procs)])
    # Removed-document samples hold full texts; only counts are kept.
    (meta / "removed_samples.jsonl").unlink(missing_ok=True)
    stats = json.loads((meta / "stats.json").read_text())
    pipeline_code = [f"data_pipeline/sources/{source}.py"] + sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "data_pipeline/common").glob("*.py"))
    write_json_atomically(stage_record_path(source, "clean"), {
        "stage": "clean", "started_at_ist": started, "finished_at_ist": now_ist(), "steps": stats.get("steps"),
        "input": stats.get("input"), "output": stats.get("output"),
        "code_sha256": code_digests(pipeline_code + ["licence/licence_filter.py", "data_pipeline/uv.lock",
                                                     "workflow/build_corpus.py"]),
        "language_model_sha256": sha256_of(LANGUAGE_MODEL), **extra,
        "input_sha256": inputs, "output_sha256": shard_digests(BUILD / source)})


def build_source(source: str, procs: int) -> None:
    download(source)
    licence(source, procs)
    clean(source, procs)
    print(f"{now_ist()} {source}: download, licence and clean finished", flush=True)


def decontam_index() -> None:
    run([str(VENV_PYTHON), str(ROOT / "workflow/decontam_index.py")])


def decontam_source(source: str, procs: int) -> None:
    require_no_record(source, "decontam")
    decontam.scan(source, procs)
    decontam.apply(source, procs)


def dedup_before_last(procs: int) -> None:
    for source in SOURCES:
        require_no_record(source, "dedup")
    cross_dedup.scan_before_last(procs)


def dedup_last(procs: int) -> None:
    cross_dedup.scan_last(procs)
    for source in SOURCES:
        cross_dedup.apply(source, procs)


def publish() -> None:
    """Replaces data/common_pile/<source>/ with the finished build; the old copy is deleted, so one copy remains.

    A source that an interrupted earlier call already moved is checked against its dedup record and not moved again.
    """
    pending = [source for source in SOURCES if (BUILD / source).exists()]
    for source in pending:
        require_input(source, "dedup")
    for source in SOURCES:
        target = DATA / source
        expected = read_stage_record(source, "dedup")["output_sha256"]
        if source in pending:
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(BUILD / source, target)
        if shard_digests(target) != expected:
            raise RuntimeError(f"{source}: published shards differ from the dedup record")
        write_json_atomically(stage_record_path(source, "publish"), {
            "stage": "publish", "finished_at_ist": now_ist(), "path": str(target.relative_to(WORK_ROOT)),
            "output_sha256": expected})
        print(f"{now_ist()} published {source}", flush=True)
    if BUILD.exists() and not any(BUILD.iterdir()):
        BUILD.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "source", "clean", "decontam-index", "decontam", "dedup-before-last",
                                            "dedup-last", "publish", "all"])
    parser.add_argument("source", nargs="?", choices=SOURCES)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    if args.command in ("download", "source", "clean") and args.source is None:
        parser.error(f"{args.command} needs a SOURCE")
    if args.command == "download":
        download(args.source)
    elif args.command == "source":
        build_source(args.source, args.procs)
    elif args.command == "clean":
        clean(args.source, args.procs)
    elif args.command == "decontam-index":
        decontam_index()
    elif args.command == "decontam":
        for source in ([args.source] if args.source else SOURCES):
            decontam_source(source, args.procs)
    elif args.command == "dedup-before-last":
        dedup_before_last(args.procs)
    elif args.command == "dedup-last":
        dedup_last(args.procs)
    elif args.command == "publish":
        publish()
    else:
        for source in SOURCES:
            build_source(source, args.procs)
        decontam_index()
        for source in SOURCES:
            decontam_source(source, args.procs)
        dedup_before_last(args.procs)
        dedup_last(args.procs)
        publish()


if __name__ == "__main__":
    main()
