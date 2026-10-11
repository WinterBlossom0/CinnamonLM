"""Apply licence_filter actions to Common Pile shards: drop rows, strip credit spans, keep all other lines byte-identical."""
import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys

FILTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILTER_DIR)
import licence_filter  # noqa: E402

GZIP_LEVEL = 6


def load_actions(actions_path):
    rows = collections.defaultdict(dict)
    with open(actions_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["shard"]][row["ordinal"]] = row
    return rows


def iter_records(path):
    """Yield (ordinal, raw line bytes, doc) with the ordinal numbering of licence_filter.iter_docs (blank lines skipped)."""
    ordinal = 0
    with gzip.open(path, "rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield ordinal, raw, json.loads(raw)
            ordinal += 1


def text_of(doc):
    return doc.get("text", "") or ""


def rewrite_shard(source_path, target_path, rows):
    stats = {"docs_before": 0, "chars_before": 0, "docs_after": 0, "chars_after": 0,
             "drops": 0, "drop_chars": 0, "strips": 0, "strip_chars_removed": 0}
    seen = set()
    with open(target_path, "wb") as raw_out:
        with gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=GZIP_LEVEL, mtime=0) as out:
            for ordinal, raw, doc in iter_records(source_path):
                text = text_of(doc)
                stats["docs_before"] += 1
                stats["chars_before"] += len(text)
                row = rows.get(ordinal)
                if row is None:
                    out.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    stats["docs_after"] += 1
                    stats["chars_after"] += len(text)
                    continue
                seen.add(ordinal)
                if row["id"] != doc.get("id"):
                    raise ValueError(f"id mismatch at ordinal {ordinal}: {row['id']} != {doc.get('id')}")
                new_doc = licence_filter.apply_action(doc, row)
                if new_doc is None:
                    stats["drops"] += 1
                    stats["drop_chars"] += len(text)
                    continue
                new_text = text_of(new_doc)
                stats["strips"] += 1
                stats["strip_chars_removed"] += len(text) - len(new_text)
                stats["docs_after"] += 1
                stats["chars_after"] += len(new_text)
                out.write((json.dumps(new_doc) + "\n").encode("utf-8"))
        raw_out.flush()
        os.fsync(raw_out.fileno())
    missing = set(rows) - seen
    if missing:
        raise ValueError(f"{len(missing)} action ordinals not found in {source_path}, e.g. {sorted(missing)[:5]}")
    return stats


def merged_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def collapse_spaces(text):
    # apply_strip also trims the spaces a removed span leaves at a line boundary.
    return re.sub(r" *\n *", "\n", re.sub(r" +", " ", text))


def strip_only_removes_spans(old_text, new_text, spans):
    """Independent of apply_strip: outside the spans only runs of spaces may change."""
    merged = merged_spans(spans)
    kept, cursor = [], 0
    for start, end in merged:
        kept.append(old_text[cursor:start])
        cursor = end
    kept.append(old_text[cursor:])
    removed = sum(end - start for start, end in merged)
    return collapse_spaces("".join(kept)) == collapse_spaces(new_text) and len(old_text) - len(new_text) >= removed


def verify_shard(source_path, target_path, rows):
    """Walk the original and rewritten shard together; raise on any difference that the actions do not explain."""
    target = iter_records(target_path)
    dropped_ids, retained_ids = set(), set()
    for ordinal, raw, doc in iter_records(source_path):
        row = rows.get(ordinal)
        if row is not None and row["action"] == "drop":
            dropped_ids.add(doc.get("id"))
            continue
        retained_ids.add(doc.get("id"))
        _, new_raw, new_doc = next(target)
        if row is None:
            if new_raw.rstrip(b"\n") != raw.rstrip(b"\n"):
                raise ValueError(f"kept record {ordinal} is not byte-identical")
            continue
        if {key: value for key, value in new_doc.items() if key != "text"} != {key: value for key, value in doc.items() if key != "text"}:
            raise ValueError(f"stripped record {ordinal} changed a field other than text")
        if not strip_only_removes_spans(text_of(doc), text_of(new_doc), row["spans"]):
            raise ValueError(f"stripped record {ordinal} differs by more than its credit spans")
    if next(target, None) is not None:
        raise ValueError("rewritten shard has extra records")
    # Source ids repeat; a dropped id is expected only where another record with that id was retained.
    must_be_absent = dropped_ids - retained_ids
    output_ids = {doc.get("id") for _, _, doc in iter_records(target_path)}
    present = must_be_absent & output_ids
    if present:
        raise ValueError(f"{len(present)} dropped ids still present, e.g. {sorted(present)[:3]}")
    return {"dropped_ids": len(dropped_ids), "dropped_ids_shared_with_retained_records": len(dropped_ids & retained_ids),
            "dropped_ids_absent": len(must_be_absent)}


def count_shard(path):
    docs = chars = 0
    for _, _, doc in iter_records(path):
        docs += 1
        chars += len(text_of(doc))
    return docs, chars


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_shard(job):
    source_path, out_dir, rows = job
    name = os.path.basename(source_path)
    in_place = out_dir is None
    target_dir = os.path.dirname(source_path) if in_place else out_dir
    target_path = os.path.join(target_dir, f".tmp_licence_{name}" if in_place else name)
    try:
        stats = rewrite_shard(source_path, target_path, rows)
        stats["verify"] = verify_shard(source_path, target_path, rows)
        if in_place:
            os.replace(target_path, source_path)
            fsync_directory(target_dir)
    except BaseException:
        if in_place and os.path.exists(target_path):
            os.remove(target_path)
        raise
    final_path = source_path if in_place else target_path
    stats["recount_docs"], stats["recount_chars"] = count_shard(final_path)
    stats["recount_matches"] = (stats["recount_docs"], stats["recount_chars"]) == (stats["docs_after"], stats["chars_after"]) \
        and stats["recount_docs"] == stats["docs_before"] - stats["drops"]
    print(f"[apply] {name} {stats}", file=sys.stderr, flush=True)
    return name, stats


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", help="write rewritten shards here; default rewrites in place")
    parser.add_argument("--report", help="JSON report path")
    parser.add_argument("--shards", nargs="*", help="basenames; default all shards")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--actions-dir", default=FILTER_DIR, help="directory of <name>_actions.jsonl and <name>_summary.json")
    args = parser.parse_args()

    actions_path = os.path.join(args.actions_dir, f"{args.name}_actions.jsonl")
    with open(os.path.join(args.actions_dir, f"{args.name}_summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = load_actions(actions_path)
    paths = licence_filter.shard_paths(args.source_dir)
    if args.shards:
        paths = [path for path in paths if os.path.basename(path) in set(args.shards)]
    unknown = set(rows) - {os.path.basename(path) for path in licence_filter.shard_paths(args.source_dir)}
    if unknown:
        raise ValueError(f"actions name shards not in source: {sorted(unknown)}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(path, args.out_dir, rows.get(os.path.basename(path), {})) for path in paths]
    with multiprocessing.Pool(min(args.workers, len(jobs))) as pool:
        per_shard = dict(pool.imap(process_shard, jobs))

    totals = {key: sum(stats[key] for stats in per_shard.values())
              for key in ("docs_before", "chars_before", "docs_after", "chars_after", "drops", "drop_chars", "strips", "strip_chars_removed")}
    full_run = len(paths) == summary["shards_total"]
    summary_checks = {
        "docs_before": totals["docs_before"] == summary["docs"],
        "chars_before": totals["chars_before"] == summary["chars"],
        "drops": totals["drops"] == summary["action_docs"].get("drop", 0),
        "drop_chars": totals["drop_chars"] == summary["action_chars"].get("drop", 0),
        "strips": totals["strips"] == summary["action_docs"].get("strip", 0),
        "docs_after": totals["docs_after"] == summary["docs"] - summary["action_docs"].get("drop", 0),
    } if full_run else None
    report = {
        "source": args.name, "source_dir": args.source_dir, "in_place": args.out_dir is None,
        "filter_path": os.path.join(FILTER_DIR, "licence_filter.py"),
        "filter_sha256": sha256_of(os.path.join(FILTER_DIR, "licence_filter.py")),
        "actions_path": actions_path, "actions_sha256": sha256_of(actions_path),
        "summary_strip_span_chars": summary["strip_chars_removed"],
        "per_shard": per_shard, "totals": totals, "summary_checks": summary_checks,
    }
    output = json.dumps(report, indent=1)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")
    print(output)
    failed = [name for name, stats in per_shard.items() if not stats["recount_matches"]]
    if failed or (summary_checks and not all(summary_checks.values())):
        sys.exit(f"verification failed: shards={failed} summary_checks={summary_checks}")


if __name__ == "__main__":
    main()
