"""Paths and helpers that every corpus build stage shares."""
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

from data_pipeline.common.io import GZIP_LEVEL

ROOT = Path(__file__).resolve().parents[1]
# CORPUS_WORK moves build/, data/ and reports/ to another directory and CORPUS_SOURCES limits the sources; both are for a
# trial build on a few sources that must not touch the corpus.
WORK_ROOT = Path(os.environ.get("CORPUS_WORK", ROOT))
BUILD = WORK_ROOT / "build"
DATA = WORK_ROOT / "data/common_pile"
REPORTS = WORK_ROOT / "reports"
VENV_PYTHON = ROOT / "data_pipeline/.venv/bin/python"
# Cross-source dedup keeps the copy in the earliest source. Whole Gutenberg books come before OER anthologies, which
# reproduce public-domain books chapter by chapter; sources with source-specific HTML cleaning come before the
# aggregator (oercommons) and before PDF extraction (doab); Gutenberg comes before Wikisource transcriptions.
ALL_SOURCES = ["public_domain_review", "project_gutenberg", "libretexts", "pressbooks", "oercommons", "doab", "wikimedia"]
SOURCES = [source for source in ALL_SOURCES if source in os.environ.get("CORPUS_SOURCES", " ".join(ALL_SOURCES)).split()]
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def now_ist() -> str:
    return datetime.datetime.now(IST).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_paths(source_dir: Path) -> list[str]:
    """Shard paths relative to source_dir, sorted; hidden files and the Hugging Face .cache directory are not shards."""
    return sorted(str(path.relative_to(source_dir)) for path in Path(source_dir).rglob("*.gz")
                  if not path.name.startswith(".") and ".cache" not in path.parts)


def shard_digests(source_dir: Path) -> dict[str, str]:
    return {relative: sha256_of(Path(source_dir) / relative) for relative in shard_paths(source_dir)}


def iter_documents(path: Path):
    """Documents of one shard in line order; blank lines are skipped and do not get a line number."""
    with gzip.open(path, "rt", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                yield json.loads(line)


def write_json_atomically(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def rewrite_shard(path: Path, transform) -> dict:
    """Rewrites one shard in place through a temp file; transform(line_number, document) returns a document or None (drop)."""
    temp = path.with_name(f".tmp_{path.name}")
    counts = {"docs_in": 0, "docs_out": 0, "chars_in": 0, "chars_out": 0}
    # An empty filename and mtime=0 keep the gzip header free of the temp name and the time, so output is reproducible.
    with open(temp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=GZIP_LEVEL,
                                                mtime=0) as compressed, \
            io.TextIOWrapper(compressed, encoding="utf-8") as writer:
        for line_number, document in enumerate(iter_documents(path)):
            counts["docs_in"] += 1
            counts["chars_in"] += len(document["text"])
            result = transform(line_number, document)
            if result is None:
                continue
            counts["docs_out"] += 1
            counts["chars_out"] += len(result["text"])
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
    with open(temp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return counts


def stage_record_path(source: str, stage: str) -> Path:
    return REPORTS / source / "build" / f"{stage}.json"


def read_stage_record(source: str, stage: str) -> dict:
    return json.loads(stage_record_path(source, stage).read_text())


def require_input(source: str, previous_stage: str) -> dict[str, str]:
    """The shard digests of the build directory, checked against the output record of the previous stage."""
    expected = read_stage_record(source, previous_stage)["output_sha256"]
    actual = shard_digests(BUILD / source)
    if actual != expected:
        raise RuntimeError(f"{source}: build shards differ from the {previous_stage} output record")
    return actual
