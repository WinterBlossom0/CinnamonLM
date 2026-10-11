import gzip
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from data_pipeline.common.io import (
    AtomicShardWriter, ShardWriter, list_shards, read_documents, temp_path_for, write_bytes_atomically,
    write_sample_shards,
)

PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def real_records(snippets) -> list[dict]:
    return [item["record"] for item in snippets["records"]]


def write_shard(path: Path, records: list[dict]) -> None:
    with ShardWriter(path) as writer:
        for record in records:
            writer.write(record)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_keeps_every_field_and_key_order(tmp_path, snippets):
    records = real_records(snippets)
    write_shard(tmp_path / "00000.jsonl.gz", records)
    read_back = list(read_documents(tmp_path / "00000.jsonl.gz"))
    assert read_back == records
    assert [list(record) for record in read_back] == [list(record) for record in records]


def test_lone_surrogate_and_wide_integer_round_trip(tmp_path):
    record = {"id": "x", "text": "bad \ud800 char", "metadata": {"big": 2**70}}
    write_shard(tmp_path / "s.jsonl.gz", [record])
    assert list(read_documents(tmp_path / "s.jsonl.gz")) == [record]


def test_list_shards_skips_cache_temp_files_and_non_shards(tmp_path):
    for relative in ["v0/documents/00001_pg.jsonl.gz", "v0/documents/00000_pg.jsonl.gz", "00000_a.json.gz",
                     ".cache/huggingface/x.jsonl.gz", "README.md", ".00000_b.json.gz.tmp", ".00000_c.json.gz"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(b""))
    names = [path.relative_to(tmp_path).as_posix() for path in list_shards(tmp_path)]
    assert names == ["00000_a.json.gz", "v0/documents/00000_pg.jsonl.gz", "v0/documents/00001_pg.jsonl.gz"]


def test_atomic_rewrite_replaces_shard_in_place(tmp_path, snippets):
    records = real_records(snippets)
    shard = tmp_path / "00000_doab.json.gz"
    write_shard(shard, records)
    with AtomicShardWriter(shard) as writer:
        for record in records[:2]:
            writer.write({**record, "text": record["text"].upper()})
        assert temp_path_for(shard).exists()
        assert list(read_documents(shard)) == records
    assert not temp_path_for(shard).exists()
    assert [path.name for path in tmp_path.iterdir()] == ["00000_doab.json.gz"]
    rewritten = list(read_documents(shard))
    assert [record["text"] for record in rewritten] == [record["text"].upper() for record in records[:2]]
    assert [list(record) for record in rewritten] == [list(record) for record in records[:2]]
    assert b".tmp" not in shard.read_bytes()[:64]


def test_exception_during_rewrite_leaves_old_shard_intact(tmp_path, snippets):
    shard = tmp_path / "00000.jsonl.gz"
    write_shard(shard, real_records(snippets))
    before = digest(shard)
    with pytest.raises(RuntimeError):
        with AtomicShardWriter(shard) as writer:
            writer.write(real_records(snippets)[0])
            raise RuntimeError("worker failed")
    assert digest(shard) == before
    assert not temp_path_for(shard).exists()


def test_killed_process_leaves_old_shard_intact(tmp_path, snippets):
    shard = tmp_path / "00000.jsonl.gz"
    write_shard(shard, real_records(snippets))
    before = digest(shard)
    script = f"""
import os, signal
from data_pipeline.common.io import AtomicShardWriter, read_documents
writer = AtomicShardWriter({str(shard)!r})
for repeat in range(200):
    for document in read_documents({str(shard)!r}):
        writer.write(document)
writer._gzip_file.flush()
os.kill(os.getpid(), signal.SIGKILL)
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=PACKAGE_PARENT)
    assert completed.returncode == -9
    assert digest(shard) == before
    assert list(read_documents(shard)) == real_records(snippets)
    assert temp_path_for(shard).stat().st_size > 0
    assert list_shards(tmp_path) == [shard]


def test_empty_rewrite_gives_valid_empty_shard(tmp_path, snippets):
    shard = tmp_path / "00000.jsonl.gz"
    write_shard(shard, real_records(snippets))
    with AtomicShardWriter(shard):
        pass
    assert list(read_documents(shard)) == []


def test_write_bytes_atomically(tmp_path):
    target = tmp_path / "stats.json"
    write_bytes_atomically(target, b"{}")
    write_bytes_atomically(target, b'{"a": 1}')
    assert target.read_bytes() == b'{"a": 1}'
    assert [path.name for path in tmp_path.iterdir()] == ["stats.json"]


def test_sample_shards_take_documents_from_every_shard(tmp_path, snippets):
    records = real_records(snippets)
    source = tmp_path / "raw"
    write_shard(source / "v0" / "00000_a.jsonl.gz", records)
    write_shard(source / "v0" / "00001_b.jsonl.gz", records[::-1])
    write_shard(source / "00002_c.json.gz", records)
    paths = write_sample_shards(source, tmp_path / "sample", limit=5)
    assert [path.relative_to(tmp_path / "sample").as_posix() for path in paths] == [
        "00002_c.json.gz", "v0/00000_a.jsonl.gz", "v0/00001_b.jsonl.gz"]
    counts = [len(list(read_documents(path))) for path in paths]
    assert counts == [2, 2, 1]
    assert list(read_documents(paths[2])) == [records[-1]]
