import difflib
import heapq
import itertools
import multiprocessing
import os
import random
import shutil
import sys
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import orjson
import regex

from .dedup import Deduplicator, load_duplicate_of, shard_offsets
from .io import (
    AtomicShardWriter, list_shards, read_documents, replace_durably, temp_path_for, write_bytes_atomically,
    write_sample_shards,
)

MAX_SAMPLES_PER_STEP = 50
SAMPLE_TEXT_CHARS = 10_000
DUPLICATE_PREVIEW_CHARS = 2_000
MAX_CHANGES_PER_SAMPLE = 100
MAX_DIFF_TOKENS = 2_000
CHANGE_SNIPPET_CHARS = 300
CHANGE_CONTEXT_CHARS = 60
TOKEN_WITH_SPACE = regex.compile(r"(\s+)")
IST = timezone(timedelta(hours=5, minutes=30))


class Step:
    """Per-document step: apply returns the document, a copy with new text, or None to drop the document."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, document: dict) -> dict | None:
        raise NotImplementedError


class TextMap(Step):
    def __init__(self, name: str, transform: Callable[[str], str]):
        super().__init__(name)
        self.transform = transform

    def apply(self, document: dict) -> dict:
        text = self.transform(document["text"])
        if text == document["text"]:
            return document
        return {**document, "text": text}


class Filter(Step):
    def __init__(self, name: str, keep: Callable[[dict], bool]):
        super().__init__(name)
        self.keep = keep

    def apply(self, document: dict) -> dict | None:
        return document if self.keep(document) else None


@dataclass
class StepStats:
    docs_in: int = 0
    docs_removed: int = 0
    chars_removed: int = 0
    docs_modified: int = 0
    chars_before_modified: int = 0
    chars_after_modified: int = 0

    def merge(self, other: "StepStats") -> None:
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)


class Reservoir:
    """Keeps the records with the smallest random keys, so a merge of reservoirs is still a uniform sample."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.heap: list[tuple[float, int, dict]] = []
        self.counter = itertools.count()

    def offer(self, make_record: Callable[[], dict]) -> None:
        key = self.rng.random()
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (-key, next(self.counter), make_record()))
        elif key < -self.heap[0][0]:
            heapq.heapreplace(self.heap, (-key, next(self.counter), make_record()))

    def items(self) -> list[tuple[float, dict]]:
        return [(-negative_key, record) for negative_key, _, record in self.heap]


def common_affix_lengths(before: str, after: str) -> tuple[int, int]:
    """Lengths of the common prefix and of the common suffix that does not overlap it."""
    prefix = len(os.path.commonprefix([before, after]))
    limit = min(len(before), len(after)) - prefix
    suffix = len(os.path.commonprefix([before[::-1][:limit], after[::-1][:limit]]))
    return prefix, suffix


def describe_changes(before: str, after: str) -> list[dict]:
    """Word-level edits from before to after, each with a short context, for audit records.

    difflib is quadratic in the worst case, so it runs on lines first and on words only inside changed line blocks
    of at most MAX_DIFF_TOKENS tokens; a larger block is one change between its common prefix and suffix.
    """
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    old_line_starts = list(itertools.accumulate((len(line) for line in old_lines), initial=0))
    changes = []
    line_matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for operation, old_start, old_stop, new_start, new_stop in line_matcher.get_opcodes():
        if operation == "equal":
            continue
        old_block = "".join(old_lines[old_start:old_stop])
        new_block = "".join(new_lines[new_start:new_stop])
        block_offset = old_line_starts[old_start]
        prefix, suffix = common_affix_lengths(old_block, new_block)
        old_middle = old_block[prefix:len(old_block) - suffix]
        new_middle = new_block[prefix:len(new_block) - suffix]
        old_tokens = TOKEN_WITH_SPACE.split(old_middle)
        new_tokens = TOKEN_WITH_SPACE.split(new_middle)
        if len(old_tokens) + len(new_tokens) > MAX_DIFF_TOKENS:
            edits = [(0, old_middle, new_middle)]
        else:
            old_token_starts = list(itertools.accumulate((len(token) for token in old_tokens), initial=0))
            token_matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
            edits = [(old_token_starts[a_start], "".join(old_tokens[a_start:a_stop]),
                      "".join(new_tokens[b_start:b_stop]))
                     for operation, a_start, a_stop, b_start, b_stop in token_matcher.get_opcodes()
                     if operation != "equal"]
        for middle_offset, old_text, new_text in edits:
            change_start = block_offset + prefix + middle_offset
            changes.append({
                "context": before[max(0, change_start - CHANGE_CONTEXT_CHARS):change_start],
                "before": old_text[:CHANGE_SNIPPET_CHARS],
                "after": new_text[:CHANGE_SNIPPET_CHARS],
            })
            if len(changes) == MAX_CHANGES_PER_SAMPLE:
                return changes
    return changes


def removed_record(step_name: str, document: dict) -> dict:
    text = document["text"]
    return {"step": step_name, "action": "removed", "id": document.get("id"), "chars": len(text),
            "text": text[:SAMPLE_TEXT_CHARS]}


def modified_record(step_name: str, document: dict, before: str, after: str) -> dict:
    return {"step": step_name, "action": "modified", "id": document.get("id"), "chars_before": len(before),
            "chars_after": len(after), "changes": describe_changes(before, after)}


@dataclass
class PassPlan:
    """One read of every shard: drop the duplicates marked by the previous deduplicators, apply the steps, and
    index the output for the next deduplicators. Consecutive deduplicators share one index pass and one drop pass.
    """
    index: int
    drop: list[Deduplicator]
    steps: list[Step]
    index_for: list[Deduplicator]

    @property
    def rewrites(self) -> bool:
        return bool(self.drop or self.steps)

    @property
    def step_names(self) -> list[str]:
        return [deduplicator.name for deduplicator in self.drop] + [step.name for step in self.steps]


@dataclass
class ShardTask:
    pass_index: int
    shard_index: int
    global_offset: int


@dataclass
class RunContext:
    pipeline: "Pipeline"
    plans: list[PassPlan]
    shards: list[Path]
    work_dir: Path

    def step_dir(self, deduplicator: Deduplicator) -> Path:
        return self.work_dir / "index" / deduplicator.name

    def build_summary_path(self, deduplicator: Deduplicator) -> Path:
        return self.step_dir(deduplicator) / "build.json"

    def pass_dir(self, pass_index: int) -> Path:
        return self.work_dir / f"pass_{pass_index}"

    def result_path(self, pass_index: int, shard_index: int) -> Path:
        return self.pass_dir(pass_index) / f"{shard_index:05d}.json"

    def pass_summary_path(self, pass_index: int) -> Path:
        return self.work_dir / f"pass_{pass_index}.json"


# Set before the worker pool forks, so workers read the pipeline from memory instead of unpickling it.
_ACTIVE_RUN: RunContext | None = None


def _apply_steps(steps: list[Step], document: dict, stats: dict[str, StepStats],
                 reservoirs: dict[str, Reservoir]) -> dict | None:
    for step in steps:
        step_stats = stats[step.name]
        step_stats.docs_in += 1
        before = document["text"]
        result = step.apply(document)
        if result is None:
            step_stats.docs_removed += 1
            step_stats.chars_removed += len(before)
            reservoirs[step.name].offer(lambda: removed_record(step.name, document))
            return None
        if list(result) != list(document):
            raise ValueError(f"step {step.name} changed the fields of document {document.get('id')}")
        after = result["text"]
        if after != before:
            step_stats.docs_modified += 1
            step_stats.chars_before_modified += len(before)
            step_stats.chars_after_modified += len(after)
            reservoirs[step.name].offer(lambda: modified_record(step.name, document, before, after))
        document = result
    return document


def _process_shard(task: ShardTask) -> dict:
    """Runs one pass over one shard; the result file is the commit point, written before the shard is replaced."""
    run = _ACTIVE_RUN
    plan = run.plans[task.pass_index]
    shard_path = run.shards[task.shard_index]
    rng = random.Random(f"{run.pipeline.seed}:{task.pass_index}:{task.shard_index}")
    stats = {name: StepStats() for name in plan.step_names}
    reservoirs = {step.name: Reservoir(run.pipeline.max_samples, rng) for step in plan.steps}
    drop_maps = [(deduplicator.name, load_duplicate_of(run.step_dir(deduplicator))) for deduplicator in plan.drop]
    indexers = [deduplicator.open_indexer(run.step_dir(deduplicator), task.shard_index)
                for deduplicator in plan.index_for]
    writer = AtomicShardWriter(shard_path) if plan.rewrites else None
    totals = {"input_docs": 0, "input_chars": 0, "output_docs": 0, "output_chars": 0}
    try:
        for local_id, document in enumerate(read_documents(shard_path)):
            chars = len(document["text"])
            totals["input_docs"] += 1
            totals["input_chars"] += chars
            if _is_marked_duplicate(drop_maps, task.global_offset + local_id, chars, stats):
                continue
            document = _apply_steps(plan.steps, document, stats, reservoirs)
            if document is None:
                continue
            for indexer in indexers:
                indexer.add(totals["output_docs"], document)
            if writer is not None:
                writer.write(document)
            totals["output_docs"] += 1
            totals["output_chars"] += len(document["text"])
        if writer is not None:
            writer.prepare()
        for indexer in indexers:
            indexer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    result = {
        "shard_index": task.shard_index,
        **totals,
        "stats": {name: asdict(step_stats) for name, step_stats in stats.items()},
        "samples": {name: reservoir.items() for name, reservoir in reservoirs.items()},
    }
    run.pass_dir(task.pass_index).mkdir(parents=True, exist_ok=True)
    write_bytes_atomically(run.result_path(task.pass_index, task.shard_index), orjson.dumps(result))
    if writer is not None:
        writer.commit()
    return result


def _is_marked_duplicate(drop_maps: list[tuple[str, np.ndarray]], global_id: int, chars: int,
                         stats: dict[str, StepStats]) -> bool:
    for name, duplicate_of in drop_maps:
        step_stats = stats[name]
        step_stats.docs_in += 1
        if duplicate_of[global_id] >= 0:
            step_stats.docs_removed += 1
            step_stats.chars_removed += chars
            return True
    return False


def _recover_shard(run: RunContext, pass_index: int, shard_index: int) -> dict | None:
    """Result of a shard that an earlier, interrupted run finished; None when the shard must be processed again.

    A result file with its temp shard still present means the crash came between the commit point and os.replace.
    """
    shard_path = run.shards[shard_index]
    result_path = run.result_path(pass_index, shard_index)
    if not result_path.exists():
        temp_path_for(shard_path).unlink(missing_ok=True)
        return None
    if temp_path_for(shard_path).exists():
        replace_durably(temp_path_for(shard_path), shard_path)
    return orjson.loads(result_path.read_bytes())


def _merge_results(plan: PassPlan, results: list[dict], max_samples: int) -> dict:
    stats = {}
    for name in plan.step_names:
        merged = StepStats()
        for result in results:
            merged.merge(StepStats(**result["stats"][name]))
        stats[name] = asdict(merged)
    samples = {}
    for step in plan.steps:
        candidates = sorted((item for result in results for item in result["samples"][step.name]),
                            key=lambda item: item[0])
        samples[step.name] = [record for _, record in candidates[:max_samples]]
    return {
        "shard_counts": [result["output_docs"] for result in results],
        "input": {"docs": sum(result["input_docs"] for result in results),
                  "chars": sum(result["input_chars"] for result in results)},
        "output": {"docs": sum(result["output_docs"] for result in results),
                   "chars": sum(result["output_chars"] for result in results)},
        "stats": stats,
        "samples": samples,
    }


def _fetch_shard_documents(job: tuple[Path, list[int]]) -> dict[int, dict]:
    shard_path, local_ids = job
    wanted = set(local_ids)
    last = max(local_ids)
    found = {}
    for local_id, document in enumerate(read_documents(shard_path)):
        if local_id in wanted:
            found[local_id] = document
        if local_id == last:
            break
    return found


def _log(message: str) -> None:
    print(f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} IST {message}", file=sys.stderr, flush=True)


class Pipeline:
    """Runs per-document steps and corpus deduplicators over a directory of JSON-lines shards, rewriting each
    shard in place. Every pass is resumable: rerunning the same command after a crash continues where it stopped.
    """

    def __init__(self, name: str, steps: list[Step | Deduplicator], seed: int = 0,
                 max_samples: int = MAX_SAMPLES_PER_STEP):
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique: {names}")
        self.name = name
        self.steps = steps
        self.seed = seed
        self.max_samples = max_samples

    def plan(self) -> list[PassPlan]:
        segments: list[list[Step]] = [[]]
        groups: list[list[Deduplicator]] = []
        for step in self.steps:
            if not isinstance(step, Deduplicator):
                segments[-1].append(step)
            elif groups and not segments[-1]:
                groups[-1].append(step)
            else:
                groups.append([step])
                segments.append([])
        return [
            PassPlan(index=pass_index, drop=groups[pass_index - 1] if pass_index > 0 else [], steps=segment,
                     index_for=groups[pass_index] if pass_index < len(groups) else [])
            for pass_index, segment in enumerate(segments)
        ]

    def run_in_place(self, data_dir: Path, meta_dir: Path, processes: int = 1, run_info: dict | None = None) -> dict:
        """Rewrites every shard under data_dir; writes stats.json and removed_samples.jsonl to meta_dir."""
        global _ACTIVE_RUN
        data_dir, meta_dir = Path(data_dir).resolve(), Path(meta_dir).resolve()
        if meta_dir.is_relative_to(data_dir) or data_dir.is_relative_to(meta_dir):
            raise ValueError("data and metadata directories must not contain each other")
        if (meta_dir / "stats.json").exists():
            raise FileExistsError(f"{meta_dir} holds the stats of a finished run; the data is already processed")
        shards = list_shards(data_dir)
        if not shards:
            raise FileNotFoundError(f"no .json.gz or .jsonl.gz shards under {data_dir}")
        work_dir = meta_dir / "_work"
        identity = {"pipeline": self.name, "steps": [step.name for step in self.steps], "data_dir": str(data_dir),
                    "shards": [shard.relative_to(data_dir).as_posix() for shard in shards]}
        identity_path = work_dir / "run.json"
        if identity_path.exists():
            if orjson.loads(identity_path.read_bytes()) != identity:
                raise RuntimeError(f"{work_dir} belongs to a different run; the data may be partly processed")
            _log(f"{self.name}: resuming the run in {work_dir}")
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            write_bytes_atomically(identity_path, orjson.dumps(identity))
        plans = self.plan()
        _ACTIVE_RUN = RunContext(self, plans, shards, work_dir)
        for step in self.steps:
            if isinstance(step, Deduplicator):
                _ACTIVE_RUN.step_dir(step).mkdir(parents=True, exist_ok=True)
        try:
            stats, samples = self._run_passes(_ACTIVE_RUN, processes)
        finally:
            _ACTIVE_RUN = None
        stats.update({"data_dir": str(data_dir), "meta_dir": str(meta_dir), **(run_info or {})})
        with open(meta_dir / "removed_samples.jsonl", "wb") as samples_file:
            for record in samples:
                samples_file.write(orjson.dumps(record) + b"\n")
        write_bytes_atomically(meta_dir / "stats.json", orjson.dumps(stats, option=orjson.OPT_INDENT_2))
        shutil.rmtree(work_dir)
        return stats

    def run_sample(self, input_dir: Path, sample_dir: Path, meta_dir: Path, limit: int, processes: int = 1) -> dict:
        """Copies up to `limit` documents to sample_dir and runs the in-place pipeline on that copy."""
        input_dir, sample_dir = Path(input_dir).resolve(), Path(sample_dir).resolve()
        if sample_dir.is_relative_to(input_dir) or input_dir.is_relative_to(sample_dir):
            raise ValueError("input and sample directories must not contain each other")
        if sample_dir.exists() and any(sample_dir.iterdir()):
            raise FileExistsError(f"{sample_dir} is not empty")
        write_sample_shards(input_dir, sample_dir, limit)
        return self.run_in_place(sample_dir, meta_dir, processes,
                                 run_info={"sample_of": str(input_dir), "limit": limit})

    def _run_passes(self, run: RunContext, processes: int) -> tuple[dict, list[dict]]:
        step_stats: dict[str, dict] = {}
        step_samples: dict[str, list[dict]] = {}
        builds: dict[str, dict] = {}
        summaries = []
        shard_counts = [0] * len(run.shards)
        for plan in run.plans:
            summary = self._run_pass(run, plan, shard_counts, processes)
            summaries.append(summary)
            step_stats.update(summary["stats"])
            step_samples.update(summary["samples"])
            shard_counts = summary["shard_counts"]
            dropped = None
            for deduplicator in plan.index_for:
                build = self._build(run, plan, deduplicator, shard_counts, dropped, processes)
                builds[deduplicator.name] = build["summary"]
                step_samples[deduplicator.name] = build["samples"]
                marked = np.asarray(load_duplicate_of(run.step_dir(deduplicator))) >= 0
                dropped = marked if dropped is None else dropped | marked
        stats = {
            "pipeline": self.name,
            "seed": self.seed,
            "input": {**summaries[0]["input"], "shards": len(run.shards)},
            "output": summaries[-1]["output"],
            "steps": [{"name": step.name, **step_stats.get(step.name, asdict(StepStats())),
                       **({"build": builds[step.name]} if step.name in builds else {})} for step in self.steps],
        }
        samples = [record for step in self.steps for record in step_samples.get(step.name, [])]
        return stats, samples

    def _run_pass(self, run: RunContext, plan: PassPlan, input_counts: list[int], processes: int) -> dict:
        summary_path = run.pass_summary_path(plan.index)
        if summary_path.exists():
            return orjson.loads(summary_path.read_bytes())
        offsets = shard_offsets(input_counts)
        results = {}
        tasks = []
        for shard_index in range(len(run.shards)):
            recovered = _recover_shard(run, plan.index, shard_index)
            if recovered is None:
                tasks.append(ShardTask(plan.index, shard_index, int(offsets[shard_index])))
            else:
                results[shard_index] = recovered
        _log(f"{self.name} pass {plan.index} ({', '.join(plan.step_names) or 'index only'}): "
             f"{len(tasks)} of {len(run.shards)} shards to process")
        for result in self._map(_process_shard, tasks, processes):
            results[result["shard_index"]] = result
            _log(f"{self.name} pass {plan.index}: shard {result['shard_index']} done "
                 f"({result['input_docs']:,} docs in, {result['output_docs']:,} out)")
        summary = _merge_results(plan, [results[index] for index in range(len(run.shards))], self.max_samples)
        write_bytes_atomically(summary_path, orjson.dumps(summary))
        shutil.rmtree(run.pass_dir(plan.index), ignore_errors=True)
        return summary

    def _build(self, run: RunContext, plan: PassPlan, deduplicator: Deduplicator, shard_counts: list[int],
               dropped: np.ndarray | None, processes: int) -> dict:
        summary_path = run.build_summary_path(deduplicator)
        if summary_path.exists():
            return orjson.loads(summary_path.read_bytes())
        _log(f"{self.name}: building {deduplicator.name}")
        summary = deduplicator.build(run.step_dir(deduplicator), shard_counts, dropped)
        samples = self._sample_duplicates(run, plan, deduplicator, shard_counts, processes)
        build = {"summary": summary, "samples": samples}
        write_bytes_atomically(summary_path, orjson.dumps(build))
        _log(f"{self.name}: {deduplicator.name} marks {summary['duplicates']:,} duplicates")
        return build

    def _sample_duplicates(self, run: RunContext, plan: PassPlan, deduplicator: Deduplicator,
                           shard_counts: list[int], processes: int) -> list[dict]:
        """Up to max_samples random removed duplicates with their kept copies, read before the drop pass
        overwrites the shards they come from.
        """
        duplicate_of = load_duplicate_of(run.step_dir(deduplicator))
        removed_ids = np.flatnonzero(np.asarray(duplicate_of) >= 0)
        rng = np.random.default_rng([self.seed, plan.index, zlib.crc32(deduplicator.name.encode())])
        chosen = np.sort(rng.choice(removed_ids, size=min(self.max_samples, len(removed_ids)), replace=False))
        kept = np.asarray(duplicate_of[chosen])
        documents = self._fetch_documents(run, shard_counts, np.concatenate([chosen, kept]), processes)
        records = []
        for removed_id, kept_id in zip(chosen.tolist(), kept.tolist()):
            kept_document = documents[kept_id]
            records.append({**removed_record(deduplicator.name, documents[removed_id]),
                            "duplicate_of": {"id": kept_document.get("id"),
                                             "text": kept_document["text"][:DUPLICATE_PREVIEW_CHARS]}})
        return records

    def _fetch_documents(self, run: RunContext, shard_counts: list[int], global_ids: np.ndarray,
                         processes: int) -> dict[int, dict]:
        offsets = shard_offsets(shard_counts)
        by_shard: dict[int, list[int]] = {}
        for global_id in np.unique(global_ids).tolist():
            shard_index = int(np.searchsorted(offsets, global_id, side="right") - 1)
            by_shard.setdefault(shard_index, []).append(global_id - int(offsets[shard_index]))
        shard_indices = sorted(by_shard)
        jobs = [(run.shards[shard_index], by_shard[shard_index]) for shard_index in shard_indices]
        documents = {}
        for shard_index, found in zip(shard_indices, self._map(_fetch_shard_documents, jobs, processes,
                                                               ordered=True)):
            for local_id, document in found.items():
                documents[int(offsets[shard_index]) + local_id] = document
        return documents

    @staticmethod
    def _map(function: Callable, jobs: list, processes: int, ordered: bool = False):
        if processes <= 1 or len(jobs) <= 1:
            yield from map(function, jobs)
            return
        with multiprocessing.get_context("fork").Pool(min(processes, len(jobs))) as pool:
            if ordered:
                yield from pool.imap(function, jobs)
            else:
                yield from pool.imap_unordered(function, jobs)
