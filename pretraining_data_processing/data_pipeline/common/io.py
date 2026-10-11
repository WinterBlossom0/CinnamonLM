import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path

import orjson

SHARD_SUFFIXES = (".json.gz", ".jsonl.gz")
GZIP_LEVEL = 6


def list_shards(input_dir: Path) -> list[Path]:
    """Shard files under input_dir in relative-path order; hidden files and directories are skipped."""
    input_dir = Path(input_dir)
    shards = []
    for path in input_dir.rglob("*"):
        relative_parts = path.relative_to(input_dir).parts
        # Hidden directories hold the Hugging Face download cache; hidden files are unfinished temp shards.
        if any(part.startswith(".") for part in relative_parts):
            continue
        if path.is_file() and path.name.endswith(SHARD_SUFFIXES):
            shards.append(path)
    return sorted(shards, key=lambda path: path.relative_to(input_dir).as_posix())


def decode_document(line: bytes) -> dict:
    try:
        return orjson.loads(line)
    except orjson.JSONDecodeError:
        # orjson rejects lone surrogate escapes and integers wider than 64 bits; json accepts both.
        return json.loads(line)


def encode_document(document: dict) -> bytes:
    try:
        return orjson.dumps(document) + b"\n"
    except TypeError:
        # json escapes the lone surrogates and wide integers that orjson cannot encode, so the record round-trips.
        return json.dumps(document, ensure_ascii=True).encode("ascii") + b"\n"


def read_documents(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                yield decode_document(line)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_durably(source: Path, target: Path) -> None:
    """os.replace, then fsync of the directory, so the new name survives a power loss."""
    os.replace(source, target)
    fsync_directory(Path(target).parent)


def temp_path_for(target: Path) -> Path:
    """Hidden name in the target's directory, so os.replace stays on one file system and list_shards skips it."""
    target = Path(target)
    return target.with_name(f".{target.name}.tmp")


def write_bytes_atomically(path: Path, data: bytes) -> None:
    path = Path(path)
    temp_path = temp_path_for(path)
    with open(temp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    replace_durably(temp_path, path)


class ShardWriter:
    """Writes documents as one gzipped JSON-lines file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.documents_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_file = open(self._write_path(), "wb")
        # An empty filename keeps the temp name out of the gzip header; mtime=0 makes output reproducible.
        self._gzip_file = gzip.GzipFile(filename="", fileobj=self._raw_file, mode="wb",
                                        compresslevel=GZIP_LEVEL, mtime=0)

    def _write_path(self) -> Path:
        return self.path

    def write(self, document: dict) -> None:
        self._gzip_file.write(encode_document(document))
        self.documents_written += 1

    def _finish(self) -> None:
        if self._gzip_file is None:
            return
        self._gzip_file.close()
        self._raw_file.flush()
        os.fsync(self._raw_file.fileno())
        self._raw_file.close()
        self._gzip_file = None

    def close(self) -> None:
        self._finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class AtomicShardWriter(ShardWriter):
    """Rewrites a shard in place: documents go to a temp file in the same directory, and commit() replaces the
    target only after the temp file is complete and on disk. Until commit() the old shard stays intact.

    As a context manager it commits on a normal exit and removes the temp file when an exception leaves the block.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        self.committed = False

    def _write_path(self) -> Path:
        return temp_path_for(self.path)

    def prepare(self) -> None:
        """Closes and fsyncs the temp file; after prepare() a crash leaves a complete temp file next to the old shard."""
        self._finish()

    def commit(self) -> None:
        if self.committed:
            return
        self.prepare()
        replace_durably(temp_path_for(self.path), self.path)
        self.committed = True

    def abort(self) -> None:
        if self._gzip_file is not None:
            self._gzip_file.close()
            self._raw_file.close()
            self._gzip_file = None
        temp_path_for(self.path).unlink(missing_ok=True)

    def close(self) -> None:
        self.commit()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()


def write_sample_shards(input_dir: Path, output_dir: Path, limit: int) -> list[Path]:
    """Copies up to `limit` documents into output_dir with the same relative shard paths.

    Each shard gives its first ceil(limit / shard count) documents, so every shard (for example every wiki of
    wikimedia) is in the sample.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    shards = list_shards(input_dir)
    if not shards:
        raise FileNotFoundError(f"no .json.gz or .jsonl.gz shards under {input_dir}")
    per_shard = -(-limit // len(shards))
    remaining = limit
    written = []
    for shard in shards:
        if remaining <= 0:
            break
        quota = min(per_shard, remaining)
        with ShardWriter(output_dir / shard.relative_to(input_dir)) as writer:
            for document in read_documents(shard):
                writer.write(document)
                if writer.documents_written == quota:
                    break
        remaining -= writer.documents_written
        written.append(writer.path)
    return written
