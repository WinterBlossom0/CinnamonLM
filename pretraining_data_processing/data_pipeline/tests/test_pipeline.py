import hashlib
import json
import re
import shutil

import pytest

from data_pipeline.common import io as shard_io
from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import ExactDedup, NearDedup
from data_pipeline.common.io import ShardWriter, list_shards, read_documents, temp_path_for
from data_pipeline.common.pii import PHONE_TOKEN, mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, Step, TextMap, describe_changes
from data_pipeline.common.text import normalize_text

SHARD_NAMES = ["00000_demo.json.gz", "00001_demo.json.gz", "v0/00002_demo.jsonl.gz"]


def make_record(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "text": text, "source": "test", "added": "2025-03-17T19:52:06", "created": "2020",
            "metadata": {"license": "CC BY", "url": doc_id}}


@pytest.fixture
def corpus(tmp_path, snippets):
    english = snippets["english"]["text"]
    near_first, near_second = snippets["near_duplicate_pair"]["texts"]
    distinct_first, distinct_second = snippets["distinct_pair"]["texts"]
    shards = [
        [make_record("english", english), make_record("near_first", near_first)],
        [make_record("distinct_first", distinct_first), make_record("spanish", snippets["spanish"]["text"]),
         make_record("contacts", snippets["letter_phone"]["text"] + "\r\n\r\n\r\nEnd \n")],
        [make_record("english_copy", re.sub(r"\s+", "  ", english.upper())), make_record("near_second", near_second),
         make_record("distinct_second", distinct_second)],
    ]
    input_dir = tmp_path / "raw" / "demo"
    for name, records in zip(SHARD_NAMES, shards):
        with ShardWriter(input_dir / name) as writer:
            for record in records:
                writer.write(record)
    return input_dir


class FailOnce(Step):
    """Raises on one document while the marker file exists, to simulate a crash in the middle of a pass."""

    def __init__(self, name, document_id, marker):
        super().__init__(name)
        self.document_id = document_id
        self.marker = marker

    def apply(self, document):
        if document["id"] == self.document_id and self.marker.exists():
            raise RuntimeError("simulated crash")
        return document


def build_pipeline(language_identifier, extra_steps=(), leading_steps=()) -> Pipeline:
    return Pipeline("demo", [
        *leading_steps,
        TextMap("normalize", normalize_text),
        Filter("english", lambda document: language_identifier.is_english(document["text"])),
        TextMap("pii", mask_pii),
        ExactDedup(seen_batch=2),
        NearDedup(max_bucket_entries=4),
        *extra_steps,
    ])


def read_corpus(directory) -> list[dict]:
    return [document for path in list_shards(directory) for document in read_documents(path)]


def digests(directory) -> dict:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.rglob("*"))
            if path.is_file()}


def copy_corpus(corpus, destination):
    shutil.copytree(corpus, destination)
    return destination


def test_plan_groups_consecutive_deduplicators(language_identifier):
    plans = build_pipeline(language_identifier).plan()
    assert [(plan.step_names, [d.name for d in plan.index_for], plan.rewrites) for plan in plans] == [
        (["normalize", "english", "pii"], ["exact_dedup", "near_dedup"], True),
        (["exact_dedup", "near_dedup"], [], True),
    ]
    split = Pipeline("x", [ExactDedup(), TextMap("strip", str.strip), NearDedup()]).plan()
    assert [(plan.step_names, [d.name for d in plan.index_for], plan.rewrites) for plan in split] == [
        ([], ["exact_dedup"], False), (["exact_dedup", "strip"], ["near_dedup"], True), (["near_dedup"], [], True)]


def test_in_place_run_rewrites_shards_keeps_schema_and_records_stats(tmp_path, corpus, language_identifier):
    raw_by_id = {document["id"]: document for document in read_corpus(corpus)}
    meta_dir = tmp_path / "processed" / "demo"
    stats = build_pipeline(language_identifier).run_in_place(corpus, meta_dir, processes=1)

    assert [path.relative_to(corpus).as_posix() for path in sorted(corpus.rglob("*")) if path.is_file()] == SHARD_NAMES
    per_shard = [[document["id"] for document in read_documents(corpus / name)] for name in SHARD_NAMES]
    assert per_shard == [["english", "near_first"], ["distinct_first", "contacts"], ["distinct_second"]]
    documents = read_corpus(corpus)
    for document in documents:
        assert list(document) == list(raw_by_id[document["id"]])
        assert {key: value for key, value in document.items() if key != "text"} == {
            key: value for key, value in raw_by_id[document["id"]].items() if key != "text"}
    contacts = next(document for document in documents if document["id"] == "contacts")
    assert contacts["text"].count(PHONE_TOKEN) == 2 and contacts["text"].endswith("\n\nEnd")
    assert sorted(path.name for path in meta_dir.iterdir()) == ["removed_samples.jsonl", "stats.json"]

    steps = {step["name"]: step for step in stats["steps"]}
    assert stats["input"] == {"docs": 8, "chars": sum(len(doc["text"]) for doc in raw_by_id.values()), "shards": 3}
    assert stats["output"] == {"docs": 5, "chars": sum(len(document["text"]) for document in documents)}
    assert steps["normalize"]["docs_in"] == 8 and steps["normalize"]["docs_modified"] >= 2
    assert steps["english"]["docs_removed"] == 1
    assert steps["english"]["chars_removed"] == len(normalize_text(raw_by_id["spanish"]["text"]))
    assert steps["pii"]["docs_modified"] == 1
    assert {key: value for key, value in steps["exact_dedup"].items() if key != "build"} == {
        "name": "exact_dedup", "docs_in": 7, "docs_removed": 1,
        "chars_removed": len(normalize_text(raw_by_id["english_copy"]["text"])),
        "docs_modified": 0, "chars_before_modified": 0, "chars_after_modified": 0}
    assert steps["exact_dedup"]["build"] == {"duplicates": 1}
    assert steps["near_dedup"]["docs_in"] == 6 and steps["near_dedup"]["docs_removed"] == 1
    assert steps["near_dedup"]["build"]["duplicates"] == 1
    assert json.loads((meta_dir / "stats.json").read_text())["steps"] == stats["steps"]

    samples = [json.loads(line) for line in (meta_dir / "removed_samples.jsonl").read_text().splitlines()]
    by_step = {}
    for sample in samples:
        by_step.setdefault(sample["step"], []).append(sample)
    assert [sample["id"] for sample in by_step["english"]] == ["spanish"]
    assert by_step["near_dedup"][0]["id"] == "near_second"
    assert by_step["near_dedup"][0]["duplicate_of"]["id"] == "near_first"
    assert by_step["exact_dedup"][0]["id"] == "english_copy"
    assert by_step["exact_dedup"][0]["duplicate_of"]["id"] == "english"
    assert any(change["after"].startswith(PHONE_TOKEN) for change in by_step["pii"][0]["changes"])

    with pytest.raises(FileExistsError):
        build_pipeline(language_identifier).run_in_place(corpus, meta_dir)


def test_parallel_run_matches_serial_run(tmp_path, corpus, language_identifier):
    serial_dir = copy_corpus(corpus, tmp_path / "serial")
    parallel_dir = copy_corpus(corpus, tmp_path / "parallel")
    serial = build_pipeline(language_identifier).run_in_place(serial_dir, tmp_path / "meta_serial", processes=1)
    parallel = build_pipeline(language_identifier).run_in_place(parallel_dir, tmp_path / "meta_parallel", processes=3)
    assert digests(serial_dir) == digests(parallel_dir)
    assert serial["steps"] == parallel["steps"]
    assert (tmp_path / "meta_serial" / "removed_samples.jsonl").read_text() == (
        tmp_path / "meta_parallel" / "removed_samples.jsonl").read_text()


@pytest.mark.parametrize("failing_id, crash_pass, processes", [
    ("near_second", 0, 1), ("distinct_second", 1, 1), ("distinct_second", 1, 3)])
def test_resume_after_crash_matches_uninterrupted_run(tmp_path, corpus, language_identifier, failing_id, crash_pass,
                                                      processes):
    marker = tmp_path / "crash_marker"
    fail_once = [FailOnce("fail_once", failing_id, marker)]
    # The failing document is in the last shard, so with one process shards 0 and 1 are rewritten before the crash.
    if crash_pass == 0:
        pipeline = build_pipeline(language_identifier, leading_steps=fail_once)
    else:
        pipeline = build_pipeline(language_identifier, extra_steps=fail_once)
    reference_dir = copy_corpus(corpus, tmp_path / "reference")
    reference = pipeline.run_in_place(reference_dir, tmp_path / "meta_reference", processes=processes)

    marker.touch()
    meta_dir = tmp_path / "meta_resumed"
    with pytest.raises(RuntimeError, match="simulated crash"):
        pipeline.run_in_place(corpus, meta_dir, processes=processes)
    assert (meta_dir / "_work" / "pass_0.json").exists() == (crash_pass == 1)
    if processes == 1:
        assert digests(corpus) != digests(reference_dir)
        assert len(list((meta_dir / "_work" / f"pass_{crash_pass}").iterdir())) == 2
    marker.unlink()
    resumed = pipeline.run_in_place(corpus, meta_dir, processes=processes)

    assert digests(corpus) == digests(reference_dir)
    assert not any(path.name.endswith(".tmp") for path in corpus.rglob("*"))
    assert resumed["steps"] == reference["steps"]
    assert resumed["input"] == reference["input"] and resumed["output"] == reference["output"]


def test_resume_after_crash_between_commit_point_and_replace(tmp_path, corpus, language_identifier, monkeypatch):
    reference_dir = copy_corpus(corpus, tmp_path / "reference")
    reference = build_pipeline(language_identifier).run_in_place(reference_dir, tmp_path / "meta_reference")
    original_replace = shard_io.replace_durably
    target = corpus / SHARD_NAMES[1]

    def crash_before_replace(source, destination):
        if destination == target:
            raise KeyboardInterrupt("power loss")
        original_replace(source, destination)

    monkeypatch.setattr(shard_io, "replace_durably", crash_before_replace)
    meta_dir = tmp_path / "meta"
    with pytest.raises(KeyboardInterrupt):
        build_pipeline(language_identifier).run_in_place(corpus, meta_dir)
    assert temp_path_for(target).exists()
    assert (meta_dir / "_work" / "pass_0" / "00001.json").exists()
    monkeypatch.setattr(shard_io, "replace_durably", original_replace)
    resumed = build_pipeline(language_identifier).run_in_place(corpus, meta_dir)
    assert digests(corpus) == digests(reference_dir)
    assert resumed["steps"] == reference["steps"]


def test_work_dir_of_a_different_run_is_refused(tmp_path, corpus, language_identifier):
    marker = tmp_path / "crash_marker"
    marker.touch()
    meta_dir = tmp_path / "meta"
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_pipeline(language_identifier, [FailOnce("fail_once", "english", marker)]).run_in_place(corpus, meta_dir)
    with pytest.raises(RuntimeError, match="different run"):
        build_pipeline(language_identifier).run_in_place(corpus, meta_dir)


def test_samples_are_capped(tmp_path):
    input_dir = tmp_path / "raw"
    with ShardWriter(input_dir / "00000.jsonl.gz") as writer:
        for index in range(120):
            writer.write(make_record(f"doc{index}", f"short text {index}"))
    stats = Pipeline("cap", [Filter("drop_all", lambda document: False)], max_samples=50).run_in_place(
        input_dir, tmp_path / "meta")
    lines = (tmp_path / "meta" / "removed_samples.jsonl").read_text().splitlines()
    assert len(lines) == 50 and len({json.loads(line)["id"] for line in lines}) == 50
    assert stats["output"] == {"docs": 0, "chars": 0}
    assert list(read_documents(input_dir / "00000.jsonl.gz")) == []


def test_step_that_changes_fields_is_rejected(tmp_path, corpus):
    class AddField(Step):
        def apply(self, document):
            return {**document, "extra": 1}

    with pytest.raises(ValueError, match="changed the fields"):
        Pipeline("x", [AddField("add_field")]).run_in_place(corpus, tmp_path / "meta")
    assert len(read_corpus(corpus)) == 8


def test_cli_sample_run_leaves_input_untouched(tmp_path, corpus, language_identifier, capsys):
    before = digests(corpus)
    sample_dir = tmp_path / "sample"
    stats = run_cli(lambda: build_pipeline(language_identifier),
                    ["--input", str(corpus), "--sample-out", str(sample_dir), "--limit", "5",
                     "--meta-out", str(tmp_path / "meta"), "--procs", "2"])
    assert digests(corpus) == before
    assert stats["input"]["docs"] == 5 and stats["limit"] == 5 and stats["sample_of"] == str(corpus.resolve())
    assert [path.relative_to(sample_dir).as_posix() for path in list_shards(sample_dir)] == SHARD_NAMES
    assert [document["id"] for document in read_corpus(sample_dir)] == ["english", "near_first", "distinct_first"]
    assert "demo: 5 docs" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        build_pipeline(language_identifier).run_sample(corpus, sample_dir, tmp_path / "meta2", limit=5)


def test_cli_in_place_run(tmp_path, corpus, language_identifier):
    stats = run_cli(lambda: build_pipeline(language_identifier),
                    ["--input", str(corpus), "--in-place", "--meta-out", str(tmp_path / "meta"), "--procs", "3"])
    assert stats["output"]["docs"] == 5 and len(read_corpus(corpus)) == 5


@pytest.mark.parametrize("arguments", [
    ["--in-place", "--limit", "5"],
    ["--sample-out", "SAMPLE"],
    ["--in-place", "--sample-out", "SAMPLE", "--limit", "5"],
    ["--sample-out", "SAMPLE", "--limit", "0"],
    ["--in-place", "--procs", "0"],
    [],
])
def test_cli_rejects_invalid_arguments(tmp_path, corpus, arguments):
    arguments = [argument.replace("SAMPLE", str(tmp_path / "sample")) for argument in arguments]
    with pytest.raises(SystemExit):
        run_cli(lambda: Pipeline("x", []), ["--input", str(corpus), "--meta-out", str(tmp_path / "meta"), *arguments])
    assert not (tmp_path / "sample").exists()


def test_directories_that_contain_each_other_are_refused(tmp_path, corpus, language_identifier):
    with pytest.raises(ValueError):
        build_pipeline(language_identifier).run_in_place(corpus, corpus / "processed")
    with pytest.raises(ValueError):
        build_pipeline(language_identifier).run_sample(corpus, corpus / "sample", tmp_path / "meta", limit=3)


def test_duplicate_step_names_rejected():
    with pytest.raises(ValueError):
        Pipeline("x", [TextMap("same", str.strip), Filter("same", bool)])


def test_describe_changes_reports_word_edits_with_context(snippets):
    before = snippets["toll_free_phone"]["text"]
    changes = describe_changes(before, mask_pii(before))
    assert changes == [{"context": before[before.index("1-800") - 60:before.index("1-800")],
                        "before": "1-800-222-1222", "after": PHONE_TOKEN}]
    multi_line = "Title  \n\n\n\nFirst line\nSecond   line\n"
    assert [(change["before"], change["after"]) for change in describe_changes(
        multi_line, normalize_text(multi_line))] == [("  \n\n", ""), ("\n", "")]
    long_line = " ".join(f"word{index}" for index in range(5000))
    edited = long_line.replace("word10 ", "WORD10 ").replace("word4990", "WORD4990")
    big_change = describe_changes(long_line, edited)
    assert len(big_change) == 1 and big_change[0]["before"].startswith("word10 ")
