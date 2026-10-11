"""Cross-source dedup stage: drop a document when most of its word 13-grams occur in a source of higher priority.

Method: the Common Pile global dedup (Dolma BFF with the whole document as the unit, a document-vs-corpus test),
with the BFF parameters of DCLM (arXiv:2406.11794, App. L): 13-grams, overlap threshold 0.8, false-positive rate 0.01.
Each source is checked only against the sources before it in corpus.SOURCES, so the result does not depend on the order
of the documents inside a source; dedup inside a source is the job of the source pipeline. The last source is only
checked, so scan_before_last saves the filter and scan_last checks the last source against it.
"""
import json
import mmap
import os
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.common.dedup import MinHasher, MinHashParams, dedup_words, mix64  # noqa: E402
from workflow.corpus import (BUILD, REPORTS, SOURCES, iter_documents, now_ist, require_input, rewrite_shard,  # noqa: E402
                             sha256_of, shard_digests, stage_record_path, write_json_atomically)

OUT = REPORTS / "cross_dedup"
WORK = OUT / "work"
BLOOM_FILE = OUT / "filter_before_last.u8"
PARTIAL = OUT / "decisions_before_last.json"
NGRAM = 13
THRESHOLD = 0.8
HASH_COUNT = 7
# 52e9 bits for about 5.3e9 inserted 13-grams (30.5e9 chars before wikimedia) gives (1 - e^(-7n/m))^7 = 0.009.
# CORPUS_BLOOM_BITS sets a small filter for a trial build on a few sources; decisions.json records the size.
BLOOM_BITS = int(os.environ.get("CORPUS_BLOOM_BITS", 52_000_000_000))
# 2e6 hashes x 7 positions with sort buffers use about 300 MB per worker.
INSERT_CHUNK = 2_000_000
# Workers write disjoint byte ranges of the shared filter under one lock per region, so no update is lost.
LOCK_REGIONS = 256
# Per-function seeds: multiples of the 64-bit golden ratio, reduced modulo 2^64.
SEEDS = [np.uint64(((index + 1) * 0x9E3779B97F4A7C15) % 2**64) for index in range(HASH_COUNT)]
CODE_FILES = ["workflow/cross_dedup.py", "workflow/corpus.py", "data_pipeline/common/dedup.py"]

hasher = MinHasher(MinHashParams(ngram_size=NGRAM))
bloom = None
region_locks = None
region_starts = None


def code_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_of(root / name) for name in CODE_FILES}


def bit_positions(hashes: np.ndarray, index: int) -> np.ndarray:
    return mix64(hashes + SEEDS[index]) % np.uint64(BLOOM_BITS)


def contained(hashes: np.ndarray) -> np.ndarray:
    present = np.ones(len(hashes), dtype=bool)
    for index in range(HASH_COUNT):
        positions = bit_positions(hashes, index)
        present &= ((bloom[(positions >> np.uint64(3)).astype(np.int64)] >> (positions & np.uint64(7)).astype(np.uint8))
                    & np.uint8(1)).astype(bool)
    return present


def insert_chunk(job) -> int:
    """Sets the filter bits of one chunk of a 13-gram file; positions are merged per byte before the locked write."""
    hash_path, start, count = job
    hashes = np.fromfile(hash_path, dtype=np.uint64, count=count, offset=start * 8)
    positions = np.concatenate([bit_positions(hashes, index) for index in range(HASH_COUNT)])
    byte_index = (positions >> np.uint64(3)).astype(np.int64)
    masks = np.left_shift(np.uint8(1), (positions & np.uint64(7)).astype(np.uint8))
    order = np.argsort(byte_index, kind="stable")
    byte_index, masks = byte_index[order], masks[order]
    first = np.flatnonzero(np.concatenate([[True], byte_index[1:] != byte_index[:-1]]))
    unique_bytes, merged = byte_index[first], np.bitwise_or.reduceat(masks, first)
    bounds = np.searchsorted(unique_bytes, region_starts)
    for region in range(LOCK_REGIONS):
        low, high = bounds[region], bounds[region + 1]
        if low == high:
            continue
        with region_locks[region]:
            bloom[unique_bytes[low:high]] |= merged[low:high]
    return count


def check_shard(job):
    """Contained fraction of each document in line order; the 13-grams of kept documents go to a work file."""
    source, relative, check, keep_hashes = job
    fractions, chars = [], []
    hash_path = WORK / f"{source}__{relative.replace('/', '__')}.u64" if keep_hashes else None
    hash_file = open(hash_path, "wb") if keep_hashes else None
    for document in iter_documents(BUILD / source / relative):
        words = dedup_words(document["text"])
        hashes = hasher.shingles(words) if words else np.empty(0, dtype=np.uint64)
        fraction = float(contained(hashes).mean()) if (check and len(hashes)) else 0.0
        fractions.append(fraction)
        chars.append(len(document["text"]))
        if keep_hashes and fraction < THRESHOLD:
            hash_file.write(hashes.tobytes())
    if hash_file is not None:
        hash_file.close()
    return relative, np.array(fractions, dtype=np.float32), np.array(chars, dtype=np.int64), hash_path


def allocate_filter(path: Path | None = None) -> None:
    """Shared filter memory, empty or loaded from a saved filter file, plus the region locks of insert_chunk."""
    global bloom, region_locks, region_starts
    # Anonymous mmap is MAP_SHARED, so the inserts of forked workers reach every process.
    memory = mmap.mmap(-1, -(-BLOOM_BITS // 8))
    if path is not None:
        with open(path, "rb") as handle:
            if handle.readinto(memory) != len(memory):
                raise RuntimeError(f"{path} is shorter than the filter")
    bloom = np.frombuffer(memory, dtype=np.uint8)
    region_locks = [multiprocessing.Lock() for _ in range(LOCK_REGIONS)]
    region_starts = np.array([region * len(bloom) // LOCK_REGIONS for region in range(LOCK_REGIONS + 1)], dtype=np.int64)


def scan_source(pool, source: str, inputs: dict[str, str], check: bool, keep_hashes: bool) -> dict:
    """Checks one source against the filter, then inserts the 13-grams of its kept documents when keep_hashes."""
    started = time.time()
    per_shard, hash_paths = {}, []
    jobs = [(source, relative, check, keep_hashes) for relative in inputs]
    for relative, fractions, chars, hash_path in pool.imap_unordered(check_shard, jobs):
        per_shard[relative] = (fractions, chars)
        if hash_path is not None:
            hash_paths.append(hash_path)
    order = sorted(per_shard)
    fractions = np.concatenate([per_shard[relative][0] for relative in order])
    chars = np.concatenate([per_shard[relative][1] for relative in order])
    np.save(OUT / f"{source}.fractions.npy", fractions)
    # Every lookup of this source has finished, so the inserts below never race with a lookup.
    insert_jobs = []
    for hash_path in sorted(hash_paths):
        total = hash_path.stat().st_size // 8
        insert_jobs.extend((hash_path, start, min(INSERT_CHUNK, total - start)) for start in range(0, total, INSERT_CHUNK))
    inserted = sum(pool.imap_unordered(insert_chunk, insert_jobs))
    for hash_path in hash_paths:
        hash_path.unlink()
    is_dropped = fractions >= THRESHOLD
    print(f"{now_ist()} dedup scan {source}: {len(fractions):,} docs, {int(is_dropped.sum()):,} dropped, "
          f"{inserted:,} 13-grams inserted, {time.time() - started:.0f} s", flush=True)
    return {"docs": int(len(fractions)), "chars": int(chars.sum()),
            "dropped_docs": int(is_dropped.sum()), "dropped_chars": int(chars[is_dropped].sum()),
            "fraction_histogram": np.histogram(fractions, bins=10, range=(0, 1))[0].tolist(), "inserted_ngrams": inserted,
            "dropped": {relative: np.flatnonzero(per_shard[relative][0] >= THRESHOLD).tolist() for relative in order}}


def scan_before_last(procs: int) -> None:
    """Checks and inserts every source but the last; saves the filter, because the last source is only checked."""
    WORK.mkdir(parents=True, exist_ok=True)
    allocate_filter()
    sources = SOURCES[:-1]
    inputs = {source: require_input(source, "decontam") for source in sources}
    partial = {"started_at_ist": now_ist(), "code_sha256": code_digests(), "input_sha256": inputs,
               "method": {"ngram": NGRAM, "threshold": THRESHOLD, "hash_count": HASH_COUNT, "bloom_bits": BLOOM_BITS,
                          "priority": SOURCES}, "sources": {}}
    with multiprocessing.Pool(procs) as pool:
        for position, source in enumerate(sources):
            partial["sources"][source] = scan_source(pool, source, inputs[source], position > 0, True)
    bloom.tofile(BLOOM_FILE)
    partial["filter_sha256"] = sha256_of(BLOOM_FILE)
    partial["inserted_ngrams"] = sum(entry["inserted_ngrams"] for entry in partial["sources"].values())
    write_json_atomically(PARTIAL, partial)
    WORK.rmdir()


def scan_last(procs: int) -> None:
    """Checks the last source against the saved filter and writes reports/cross_dedup/decisions.json."""
    decisions = json.loads(PARTIAL.read_text())
    if decisions["code_sha256"] != code_digests() or sha256_of(BLOOM_FILE) != decisions["filter_sha256"]:
        raise RuntimeError("code or saved filter changed after scan_before_last")
    allocate_filter(BLOOM_FILE)
    source = SOURCES[-1]
    inputs = require_input(source, "decontam")
    decisions["input_sha256"][source] = inputs
    with multiprocessing.Pool(procs) as pool:
        decisions["sources"][source] = scan_source(pool, source, inputs, True, False)
    probe = mix64(np.arange(1, 10_000_001, dtype=np.uint64) + np.uint64(0x5DEECE66D))
    inserted = decisions["inserted_ngrams"]
    decisions["expected_false_positive_rate"] = float((1 - np.exp(-HASH_COUNT * inserted / BLOOM_BITS)) ** HASH_COUNT)
    decisions["measured_false_positive_rate"] = float(contained(probe).mean())
    decisions["finished_at_ist"] = now_ist()
    write_json_atomically(OUT / "decisions.json", decisions)
    BLOOM_FILE.unlink()
    PARTIAL.unlink()


def apply_shard(job):
    source, relative, dropped = job
    return relative, rewrite_shard(BUILD / source / relative,
                                   lambda line_number, document: None if line_number in dropped else document)


def apply(source: str, procs: int) -> None:
    """Drops the duplicate documents of one source in place and writes reports/<source>/build/dedup.json."""
    decisions = json.loads((OUT / "decisions.json").read_text())
    inputs = require_input(source, "decontam")
    if inputs != decisions["input_sha256"][source]:
        raise RuntimeError(f"{source}: build shards changed after the dedup scan")
    dropped = decisions["sources"][source]["dropped"]
    started = now_ist()
    totals = {}
    with multiprocessing.Pool(procs) as pool:
        for _, counts in pool.imap_unordered(apply_shard, [(source, relative, set(dropped[relative]))
                                                           for relative in inputs]):
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    if totals.get("docs_in", 0) - totals.get("docs_out", 0) != decisions["sources"][source]["dropped_docs"]:
        raise RuntimeError(f"{source}: dropped count differs from the scan decisions")
    write_json_atomically(stage_record_path(source, "dedup"), {
        "stage": "dedup", "started_at_ist": started, "finished_at_ist": now_ist(), "method": decisions["method"],
        "code_sha256": code_digests(), "decisions_sha256": sha256_of(OUT / "decisions.json"), "totals": totals,
        "input_sha256": inputs, "output_sha256": shard_digests(BUILD / source)})
    print(f"{now_ist()} dedup apply {source}: {totals}", flush=True)
