import argparse
from collections.abc import Callable
from pathlib import Path

from .pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one Common Pile source.")
    parser.add_argument("--input", type=Path, required=True, help="directory with the .json.gz/.jsonl.gz shards")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--in-place", action="store_true", help="rewrite the input shards in place")
    mode.add_argument("--sample-out", type=Path, help="copy --limit documents here and process the copy")
    parser.add_argument("--limit", type=int, help="number of documents for --sample-out")
    parser.add_argument("--meta-out", type=Path, required=True,
                        help="directory for stats.json, removed_samples.jsonl and resumable work files")
    parser.add_argument("--procs", type=int, default=1, help="worker processes")
    args = parser.parse_args(argv)
    if args.sample_out is not None and args.limit is None:
        parser.error("--sample-out needs --limit")
    if args.in_place and args.limit is not None:
        parser.error("--limit applies only to --sample-out")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.procs < 1:
        parser.error("--procs must be at least 1")
    return args


def format_stats(stats: dict) -> str:
    lines = [f"{stats['pipeline']}: {stats['input']['docs']:,} docs / {stats['input']['chars']:,} chars in, "
             f"{stats['output']['docs']:,} docs / {stats['output']['chars']:,} chars out"]
    for step in stats["steps"]:
        lines.append(f"  {step['name']}: removed {step['docs_removed']:,} docs ({step['chars_removed']:,} chars), "
                     f"modified {step['docs_modified']:,} docs "
                     f"({step['chars_before_modified'] - step['chars_after_modified']:,} chars removed)")
    return "\n".join(lines)


def run_cli(build_pipeline: Callable[[], Pipeline], argv: list[str] | None = None) -> dict:
    """Entry point of python -m data_pipeline.sources.<source>."""
    args = parse_args(argv)
    pipeline = build_pipeline()
    if args.in_place:
        stats = pipeline.run_in_place(args.input, args.meta_out, processes=args.procs)
    else:
        stats = pipeline.run_sample(args.input, args.sample_out, args.meta_out, args.limit, processes=args.procs)
    print(format_stats(stats))
    return stats
