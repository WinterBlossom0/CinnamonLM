"""Benchmark decontamination stage: remove evaluation items of the benchmarks in decontam_index.py from the cleaned corpus.

Scan: a (document, item) pair is written when they share at least MIN_ROW_SHARED of the item's distinct 8-grams.
Apply: a pair is a match when they share at least MIN_SHARED 8-grams and at least MATCH_FRACTION of the item's or of the
document's 8-grams; the lines that hold an 8-gram of a matched item are removed, and the document is dropped when those
lines are at least DROP_SHARE of its characters.
"""
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.common.dedup import MinHasher, MinHashParams, dedup_words, hash_word, mix64  # noqa: E402
from workflow.corpus import (BUILD, REPORTS, iter_documents, now_ist, require_input, rewrite_shard,  # noqa: E402
                             sha256_of, shard_digests, stage_record_path, write_json_atomically)

INDEX = REPORTS / "decontam"
NGRAM = 8
# Rows below this count are not written; the decision thresholds are applied in apply().
MIN_ROW_SHARED = 3
# PaLM (arXiv:2204.02311, §8) calls an example contaminated when 70 % of its 8-grams occur in the training data.
MATCH_FRACTION = 0.7
# 8 shared 8-grams are at least 15 words; template sentences (e.g. "X is a Finnish surname. Notable people with the
# surname include") share up to 4 with short eval items of the same template.
MIN_SHARED = 8
DROP_SHARE = 0.5
CODE_FILES = ["workflow/decontam.py", "workflow/decontam_index.py", "workflow/corpus.py", "data_pipeline/common/dedup.py"]

hasher = MinHasher(MinHashParams(ngram_size=NGRAM))
eval_ngrams = eval_items_of_ngram = item_ngrams = None


def load_index() -> None:
    global eval_ngrams, eval_items_of_ngram, item_ngrams
    eval_ngrams = np.load(INDEX / "eval_ngrams.npy")
    eval_items_of_ngram = np.load(INDEX / "eval_ngram_items.npy")
    items = json.loads((INDEX / "eval_items.json").read_text())["items"]
    item_ngrams = np.array([item["ngrams"] for item in items], dtype=np.int64)


def index_digests() -> dict[str, str]:
    return {name: sha256_of(INDEX / name) for name in ("eval_ngrams.npy", "eval_ngram_items.npy", "eval_items.json")}


def code_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_of(root / name) for name in CODE_FILES}


def shared_counts(hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(item ids, shared 8-gram count) of every item that shares an 8-gram with the document."""
    left = np.searchsorted(eval_ngrams, hashes, side="left")
    right = np.searchsorted(eval_ngrams, hashes, side="right")
    hit = right > left
    if not hit.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left, right = left[hit], right[hit]
    lengths = right - left
    positions = np.repeat(left - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
    return np.unique(eval_items_of_ngram[positions], return_counts=True)


def scan_shard(job):
    source, relative = job
    rows = []
    for line_number, document in enumerate(iter_documents(BUILD / source / relative)):
        words = dedup_words(document["text"])
        hashes = hasher.shingles(words) if words else np.empty(0, dtype=np.uint64)
        item_ids, counts = shared_counts(hashes)
        for item_id, count in zip(item_ids.tolist(), counts.tolist()):
            if count >= MIN_ROW_SHARED:
                rows.append({"shard": relative, "line": line_number, "id": str(document["id"]), "item": item_id,
                             "shared": count, "item_fraction": round(count / int(item_ngrams[item_id]), 4),
                             "doc_fraction": round(count / max(1, len(hashes)), 4), "doc_ngrams": len(hashes),
                             "chars": len(document["text"])})
    return relative, rows


def scan(source: str, procs: int) -> None:
    """Writes reports/decontam/<source>.matches.json and reports/<source>/build/decontam_scan.json."""
    load_index()
    inputs = require_input(source, "clean")
    started = now_ist()
    rows = []
    with multiprocessing.Pool(procs) as pool:
        for _, shard_rows in pool.imap_unordered(scan_shard, [(source, relative) for relative in inputs]):
            rows.extend(shard_rows)
    rows.sort(key=lambda row: (row["shard"], row["line"], row["item"]))
    write_json_atomically(INDEX / f"{source}.matches.json", rows)
    write_json_atomically(stage_record_path(source, "decontam_scan"), {
        "stage": "decontam_scan", "started_at_ist": started, "finished_at_ist": now_ist(),
        "index_sha256": index_digests(), "code_sha256": code_digests(), "min_row_shared": MIN_ROW_SHARED,
        "rows": len(rows), "matches_sha256": sha256_of(INDEX / f"{source}.matches.json"), "input_sha256": inputs})
    print(f"{now_ist()} decontam scan {source}: {len(rows):,} rows", flush=True)


def is_match(row: dict) -> bool:
    return row["shared"] >= MIN_SHARED and max(row["item_fraction"], row["doc_fraction"]) >= MATCH_FRACTION


def matched_lines(text: str, items: set[int]) -> set[int]:
    """Indices of the lines of text that hold a word of an 8-gram that belongs to one of the given eval items."""
    words, line_of_word = [], []
    for line_index, line in enumerate(text.split("\n")):
        line_words = dedup_words(line)
        words.extend(line_words)
        line_of_word.extend([line_index] * len(line_words))
    if not words:
        return set()
    size = min(NGRAM, len(words))
    word_hashes = np.fromiter((hash_word(word) for word in words), dtype=np.uint64, count=len(words))
    windows = np.lib.stride_tricks.sliding_window_view(word_hashes, size)
    # Same window hash as MinHasher.shingles, but per position instead of de-duplicated.
    window_hashes = mix64((windows * hasher.position_multipliers[:size]).sum(axis=1, dtype=np.uint64))
    left = np.searchsorted(eval_ngrams, window_hashes, side="left")
    right = np.searchsorted(eval_ngrams, window_hashes, side="right")
    hit_lines = set()
    for start in np.flatnonzero(right > left).tolist():
        if set(eval_items_of_ngram[left[start]:right[start]].tolist()) & items:
            hit_lines.update(line_of_word[start:start + size])
    return hit_lines


def apply_shard(job):
    source, relative, matches = job
    counts = {"dropped": 0, "edited": 0, "lines_removed": 0, "chars_removed": 0}

    def transform(line_number, document):
        rows = matches.get(line_number)
        if not rows:
            return document
        text_lines = document["text"].split("\n")
        removed = matched_lines(document["text"], {row["item"] for row in rows})
        removed_chars = sum(len(text_lines[index]) + 1 for index in removed)
        if removed_chars >= DROP_SHARE * len(document["text"]):
            counts["dropped"] += 1
            return None
        if not removed:
            return document
        counts["edited"] += 1
        counts["lines_removed"] += len(removed)
        counts["chars_removed"] += removed_chars
        return {**document, "text": "\n".join(line for index, line in enumerate(text_lines) if index not in removed)}

    shard_counts = rewrite_shard(BUILD / source / relative, transform)
    return relative, {**shard_counts, **counts}


def apply(source: str, procs: int) -> None:
    """Rewrites the build shards of one source in place and writes reports/<source>/build/decontam.json."""
    scan_record = json.loads(stage_record_path(source, "decontam_scan").read_text())
    if scan_record["index_sha256"] != index_digests():
        raise RuntimeError("the eval index changed after the scan")
    inputs = require_input(source, "clean")
    if inputs != scan_record["input_sha256"]:
        raise RuntimeError(f"{source}: build shards changed after the scan")
    if sha256_of(INDEX / f"{source}.matches.json") != scan_record["matches_sha256"]:
        raise RuntimeError(f"{source}: matches file changed after the scan")
    load_index()
    rows = json.loads((INDEX / f"{source}.matches.json").read_text())
    items = json.loads((INDEX / "eval_items.json").read_text())["items"]
    matches = {relative: {} for relative in inputs}
    matched_items = {}
    for row in rows:
        if is_match(row):
            matches[row["shard"]].setdefault(row["line"], []).append(row)
            benchmark = items[row["item"]]["benchmark"]
            matched_items[benchmark] = matched_items.get(benchmark, 0) + 1
    started = now_ist()
    totals = {}
    with multiprocessing.Pool(procs) as pool:
        for _, counts in pool.imap_unordered(apply_shard, [(source, relative, matches[relative]) for relative in inputs]):
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    write_json_atomically(stage_record_path(source, "decontam"), {
        "stage": "decontam", "started_at_ist": started, "finished_at_ist": now_ist(),
        "rule": {"ngram": NGRAM, "min_shared": MIN_SHARED, "match_fraction": MATCH_FRACTION, "drop_share": DROP_SHARE},
        "index_sha256": scan_record["index_sha256"], "code_sha256": code_digests(),
        "matched_docs": sum(len(lines) for lines in matches.values()), "matched_pairs_per_benchmark": matched_items,
        "totals": totals, "input_sha256": inputs, "output_sha256": shard_digests(BUILD / source)})
    print(f"{now_ist()} decontam apply {source}: {totals}", flush=True)
