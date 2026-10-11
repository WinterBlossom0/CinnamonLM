import os
import sqlite3
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex
import xxhash
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io import replace_durably, temp_path_for

PUNCTUATION = regex.compile(r"\p{P}+")
COMBINING_MARK = regex.compile(r"\p{Mn}+")
# One LSH key bucket of 2^23 entries needs about 0.5 GB of RAM for its arrays and sort buffers.
MAX_BUCKET_ENTRIES = 2**23
# Connected components of 2^23 edges need about 0.5 GB; above that the edge list is compacted to stars.
MAX_PENDING_EDGES = 2**23
# Signature pairs compared per batch: 2^17 pairs x 112 hashes x 4 bytes x 2 sides is about 117 MB.
VERIFY_BATCH_PAIRS = 2**17
SHINGLE_BATCH = 8192
SEEN_BATCH = 2**18
SEEN_CACHE_BYTES = 256 * 2**20
DUPLICATE_OF_FILE = "duplicate_of.npy"


def dedup_words(text: str) -> list[str]:
    # RefinedWeb (arXiv:2306.01116, App. G.3.1) hashes text after lowercasing and removing punctuation and accents.
    text = PUNCTUATION.sub(" ", text.lower())
    text = COMBINING_MARK.sub("", unicodedata.normalize("NFD", text))
    return text.split()


def hash_word(word: str) -> int:
    # surrogatepass keeps text with lone surrogates hashable; normalize_text replaces them, raw text may not.
    return xxhash.xxh3_64_intdigest(word.encode("utf-8", "surrogatepass"))


def exact_key(text: str) -> int | None:
    """64-bit hash of the dedup-normalized words; None when the text has no words, so it is never a duplicate."""
    words = dedup_words(text)
    if not words:
        return None
    return hash_word(" ".join(words))


def mix64(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer (Steele et al., OOPSLA 2014): spreads every input bit over all output bits."""
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


@dataclass(frozen=True)
class MinHashParams:
    # Word 5-gram shingles: Lee et al. (arXiv:2107.06499) and FineWeb (arXiv:2406.17557 §3.4).
    ngram_size: int = 5
    # FineWeb: 112 hashes = 14 bands x 8 rows, match probability 0.77 / 0.92 at Jaccard 0.75 / 0.8.
    num_bands: int = 14
    rows_per_band: int = 8
    # Lee et al. NEARDUP and Gopher (arXiv:2112.11446 A.1.1) call a pair a duplicate at Jaccard 0.8.
    min_jaccard: float = 0.8
    seed: int = 0

    @property
    def num_hashes(self) -> int:
        return self.num_bands * self.rows_per_band


class MinHasher:
    def __init__(self, params: MinHashParams = MinHashParams()):
        self.params = params
        rng = np.random.default_rng(params.seed)
        # Multiply-add-shift hashing (Dietzfelbinger 1996) needs odd multipliers.
        self.multipliers = rng.integers(0, 2**64, params.num_hashes, dtype=np.uint64, endpoint=False) | np.uint64(1)
        self.addends = rng.integers(0, 2**64, params.num_hashes, dtype=np.uint64, endpoint=False)
        self.position_multipliers = rng.integers(
            0, 2**64, params.ngram_size, dtype=np.uint64, endpoint=False) | np.uint64(1)

    def shingles(self, words: list[str]) -> np.ndarray:
        """Distinct 64-bit hashes of the word n-grams; a text shorter than ngram_size words is one shingle."""
        word_hashes = np.fromiter((hash_word(word) for word in words), dtype=np.uint64, count=len(words))
        size = min(self.params.ngram_size, len(words))
        windows = sliding_window_view(word_hashes, size)
        # A position-weighted sum (modulo 2^64) hashes each window from the word hashes without joining strings.
        return np.unique(mix64((windows * self.position_multipliers[:size]).sum(axis=1, dtype=np.uint64)))

    def signature(self, text: str) -> np.ndarray | None:
        """uint32 MinHash signature over word n-gram shingles of dedup_words(text); None when the text has no words."""
        words = dedup_words(text)
        if not words:
            return None
        shingles = self.shingles(words)
        signature = np.full(self.params.num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(shingles), SHINGLE_BATCH):
            batch = shingles[start:start + SHINGLE_BATCH, None]
            # uint64 overflow is the intended arithmetic modulo 2^64; the top 32 bits are the hash value.
            hashed = (batch * self.multipliers + self.addends) >> np.uint64(32)
            np.minimum(signature, hashed.min(axis=0), out=signature)
        return signature.astype(np.uint32)

    def band_keys(self, signature: np.ndarray) -> np.ndarray:
        """One uint64 key per band; the band index seeds the hash so equal rows in different bands never collide."""
        bands = signature.reshape(self.params.num_bands, self.params.rows_per_band)
        return np.array(
            [xxhash.xxh3_64_intdigest(bands[band].tobytes(), seed=band) for band in range(self.params.num_bands)],
            dtype=np.uint64,
        )


def save_array_atomically(path: Path, array: np.ndarray) -> None:
    temp_path = temp_path_for(path)
    with open(temp_path, "wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    replace_durably(temp_path, path)


def shard_prefix(directory: Path, shard_index: int) -> Path:
    return Path(directory) / f"{shard_index:05d}"


def keys_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".keys.npy")


def ids_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".ids.npy")


def signatures_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".sig.u32")


def shard_offsets(shard_counts: list[int]) -> np.ndarray:
    """Global id of the first document of each shard; global ids number documents in shard order."""
    return np.concatenate([[0], np.cumsum(shard_counts, dtype=np.int64)[:-1]]).astype(np.int64)


class KeyIndexWriter:
    """Collects (key, local document id) pairs of one shard and saves them, optionally sorted by key."""

    def __init__(self, path_prefix: Path, sort_by_key: bool):
        self.path_prefix = Path(path_prefix)
        self.sort_by_key = sort_by_key
        self.keys: list[np.ndarray] = []
        self.local_ids: list[np.ndarray] = []

    def add(self, local_id: int, keys: np.ndarray) -> None:
        self.keys.append(keys.astype(np.uint64))
        self.local_ids.append(np.full(len(keys), local_id, dtype=np.uint32))

    def close(self) -> None:
        keys = np.concatenate(self.keys) if self.keys else np.empty(0, dtype=np.uint64)
        local_ids = np.concatenate(self.local_ids) if self.local_ids else np.empty(0, dtype=np.uint32)
        if self.sort_by_key:
            order = np.argsort(keys, kind="stable")
            keys, local_ids = keys[order], local_ids[order]
        save_array_atomically(keys_path(self.path_prefix), keys)
        save_array_atomically(ids_path(self.path_prefix), local_ids)


def load_index(directory: Path, shard_index: int, mmap: bool) -> tuple[np.ndarray, np.ndarray]:
    """(keys, local ids) of one shard's index."""
    prefix = shard_prefix(directory, shard_index)
    mode = "r" if mmap else None
    keys = np.load(keys_path(prefix), mmap_mode=mode)
    local_ids = np.load(ids_path(prefix), mmap_mode=mode)
    return keys, local_ids


class DuplicateOfWriter:
    """Disk-backed array (memmap) of length corpus size: the global id of the kept copy, or -1 for a kept document.

    The array becomes DUPLICATE_OF_FILE only when the block exits without an exception.
    """

    def __init__(self, directory: Path, size: int):
        self.path = Path(directory) / DUPLICATE_OF_FILE
        self.array = open_memmap(temp_path_for(self.path), mode="w+", dtype=np.int64, shape=(size,))
        self.array[:] = -1

    def __enter__(self) -> np.ndarray:
        return self.array

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.array.flush()
        del self.array
        if exc_type is not None:
            temp_path_for(self.path).unlink(missing_ok=True)
            return
        with open(temp_path_for(self.path), "rb+") as handle:
            os.fsync(handle.fileno())
        replace_durably(temp_path_for(self.path), self.path)


def load_duplicate_of(directory: Path) -> np.ndarray:
    return np.load(Path(directory) / DUPLICATE_OF_FILE, mmap_mode="r")


class Deduplicator:
    """Corpus step: one pass indexes every document, build() marks duplicates, the next pass drops them.

    All index data lives in files under the step directory, so RAM stays bounded for any corpus size.
    """

    def __init__(self, name: str):
        self.name = name

    def open_indexer(self, directory: Path, shard_index: int):
        raise NotImplementedError

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        """Writes DUPLICATE_OF_FILE; documents flagged in `dropped` (removed by an earlier step) are never kept copies."""
        raise NotImplementedError


class SeenSet:
    """Disk-backed set of 64-bit keys (SQLite B-tree), each with the global id of the first document that had it.

    The file is a rebuildable work file, so journaling and fsync are off.
    """

    def __init__(self, path: Path, cache_bytes: int = SEEN_CACHE_BYTES):
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(f"PRAGMA cache_size={-(cache_bytes // 1024)}")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen (key INTEGER PRIMARY KEY, global_id INTEGER NOT NULL) WITHOUT ROWID")
        self.connection.execute(
            "CREATE TEMP TABLE batch (key INTEGER PRIMARY KEY, global_id INTEGER NOT NULL) WITHOUT ROWID")

    def first_seen(self, keys: np.ndarray, global_ids: np.ndarray) -> np.ndarray:
        """For each entry the global id of an earlier entry with the same key, or -1; then adds the new keys.

        Entries must arrive in ascending global id, so the first document of each key is the one kept.
        """
        # SQLite integers are signed 64-bit; the bit pattern of the uint64 key is kept.
        signed_keys = np.ascontiguousarray(keys, dtype=np.uint64).view(np.int64)
        unique_keys, first_positions, inverse = np.unique(signed_keys, return_index=True, return_inverse=True)
        representatives = global_ids[first_positions].astype(np.int64)
        connection = self.connection
        connection.execute("BEGIN")
        connection.execute("DELETE FROM batch")
        connection.executemany("INSERT INTO batch VALUES (?, ?)", zip(unique_keys.tolist(), representatives.tolist()))
        rows = connection.execute(
            "SELECT batch.key, seen.global_id FROM batch JOIN seen ON seen.key = batch.key").fetchall()
        connection.execute("INSERT OR IGNORE INTO seen SELECT key, global_id FROM batch")
        connection.execute("COMMIT")
        if rows:
            row_keys, row_ids = np.array(rows, dtype=np.int64).T
            representatives[np.searchsorted(unique_keys, row_keys)] = row_ids
        per_entry = representatives[inverse.reshape(-1)]
        return np.where(per_entry == global_ids, -1, per_entry)

    def close(self) -> None:
        self.connection.close()


class ExactIndexer:
    def __init__(self, directory: Path, shard_index: int):
        self.keys = KeyIndexWriter(shard_prefix(directory, shard_index), sort_by_key=False)

    def add(self, local_id: int, document: dict) -> None:
        key = exact_key(document["text"])
        if key is not None:
            self.keys.add(local_id, np.array([key], dtype=np.uint64))

    def close(self) -> None:
        self.keys.close()


class ExactDedup(Deduplicator):
    """Keeps the first document (in shard order) of each group with equal exact_key, using a SeenSet."""

    def __init__(self, name: str = "exact_dedup", seen_batch: int = SEEN_BATCH):
        super().__init__(name)
        self.seen_batch = seen_batch

    def open_indexer(self, directory: Path, shard_index: int) -> ExactIndexer:
        return ExactIndexer(directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        directory = Path(directory)
        seen_path = directory / "seen.sqlite"
        seen_path.unlink(missing_ok=True)
        seen = SeenSet(seen_path)
        with DuplicateOfWriter(directory, sum(shard_counts)) as duplicate_of:
            for shard_index, offset in enumerate(shard_offsets(shard_counts)):
                if not shard_counts[shard_index]:
                    continue
                keys, local_ids = load_index(directory, shard_index, mmap=False)
                global_ids = local_ids.astype(np.int64) + offset
                if dropped is not None:
                    live = ~dropped[global_ids]
                    keys, global_ids = keys[live], global_ids[live]
                for start in range(0, len(keys), self.seen_batch):
                    batch_ids = global_ids[start:start + self.seen_batch]
                    duplicate_of[batch_ids] = seen.first_seen(keys[start:start + self.seen_batch], batch_ids)
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        seen.close()
        seen_path.unlink()
        return {"duplicates": duplicates}


def iter_key_buckets(directory: Path, shard_counts: list[int],
                     max_bucket_entries: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (keys, global ids) per range of the key space; each range holds about max_bucket_entries entries.

    Each shard's keys are sorted on disk, so a range is one memmap slice per shard.
    """
    offsets = shard_offsets(shard_counts)
    parts = [(*load_index(directory, shard_index, mmap=True), int(offset))
             for shard_index, offset in enumerate(offsets) if shard_counts[shard_index]]
    total_entries = sum(len(keys) for keys, _, _ in parts)
    bucket_count = max(1, -(-total_entries // max_bucket_entries))
    boundaries = [(bucket << 64) // bucket_count for bucket in range(bucket_count + 1)]
    for bucket in range(bucket_count):
        key_pieces, id_pieces = [], []
        for keys, local_ids, offset in parts:
            start = int(np.searchsorted(keys, np.uint64(boundaries[bucket])))
            is_last = bucket == bucket_count - 1
            stop = len(keys) if is_last else int(np.searchsorted(keys, np.uint64(boundaries[bucket + 1])))
            key_pieces.append(np.asarray(keys[start:stop]))
            id_pieces.append(local_ids[start:stop].astype(np.int64) + offset)
        if key_pieces:
            yield np.concatenate(key_pieces), np.concatenate(id_pieces)


def duplicate_pairs(keys: np.ndarray, global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each group of equal keys returns pairs (smallest id of the group, other id of the group)."""
    if len(keys) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    order = np.lexsort((global_ids, keys))
    keys = keys[order]
    global_ids = global_ids[order]
    starts_group = np.empty(len(keys), dtype=bool)
    starts_group[0] = True
    starts_group[1:] = keys[1:] != keys[:-1]
    group_first_ids = global_ids[starts_group][np.cumsum(starts_group) - 1]
    is_pair = ~starts_group & (group_first_ids != global_ids)
    return group_first_ids[is_pair], global_ids[is_pair]


def star_edges(sources: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of the edge list as pairs (smallest id of the component, other member)."""
    nodes, inverse = np.unique(np.concatenate([sources, targets]), return_inverse=True)
    inverse = inverse.reshape(-1)
    edge_count = len(sources)
    graph = coo_matrix(
        (np.ones(edge_count, dtype=np.int8), (inverse[:edge_count], inverse[edge_count:])),
        shape=(len(nodes), len(nodes)),
    )
    _, labels = connected_components(graph, directed=False)
    # nodes is sorted, so the first node with each label is the smallest id of that component.
    _, first_positions = np.unique(labels, return_index=True)
    representatives = nodes[first_positions][labels]
    is_member = representatives != nodes
    return representatives[is_member], nodes[is_member]


class SignatureStore:
    """Reads MinHash signatures of all shards by global id from the per-shard files on disk."""

    def __init__(self, directory: Path, shard_counts: list[int], num_hashes: int):
        self.offsets = shard_offsets(shard_counts)
        self.num_hashes = num_hashes
        self.arrays = {}
        for shard_index, count in enumerate(shard_counts):
            if count:
                path = signatures_path(shard_prefix(directory, shard_index))
                self.arrays[shard_index] = np.memmap(path, dtype=np.uint32, mode="r", shape=(count, num_hashes))

    def gather(self, global_ids: np.ndarray) -> np.ndarray:
        # side="right" maps an id to the last shard starting at or before it, which skips empty shards.
        shard_indices = np.searchsorted(self.offsets, global_ids, side="right") - 1
        signatures = np.empty((len(global_ids), self.num_hashes), dtype=np.uint32)
        for shard_index in np.unique(shard_indices):
            selected = shard_indices == shard_index
            signatures[selected] = self.arrays[int(shard_index)][global_ids[selected] - self.offsets[shard_index]]
        return signatures


class MinHashIndexer:
    def __init__(self, hasher: MinHasher, directory: Path, shard_index: int):
        self.hasher = hasher
        prefix = shard_prefix(directory, shard_index)
        self.keys = KeyIndexWriter(prefix, sort_by_key=True)
        self.signature_path = signatures_path(prefix)
        self.signature_file = open(temp_path_for(self.signature_path), "wb")
        # A text without words gets no band keys, so its all-zero row is never compared.
        self.empty_signature = np.zeros(hasher.params.num_hashes, dtype=np.uint32).tobytes()
        self.next_local_id = 0

    def add(self, local_id: int, document: dict) -> None:
        # Signature rows are addressed by local id, so documents must arrive in output order.
        if local_id != self.next_local_id:
            raise ValueError(f"expected local id {self.next_local_id}, got {local_id}")
        self.next_local_id += 1
        signature = self.hasher.signature(document["text"])
        if signature is None:
            self.signature_file.write(self.empty_signature)
            return
        self.signature_file.write(signature.tobytes())
        self.keys.add(local_id, self.hasher.band_keys(signature))

    def close(self) -> None:
        self.signature_file.flush()
        os.fsync(self.signature_file.fileno())
        self.signature_file.close()
        replace_durably(temp_path_for(self.signature_path), self.signature_path)
        self.keys.close()


class NearDedup(Deduplicator):
    """MinHash-LSH near-duplicate removal: candidate pairs share a band key, and a pair counts only when the
    signature agreement (the Jaccard estimate) reaches min_jaccard; each connected component keeps its first document.
    """

    def __init__(self, name: str = "near_dedup", params: MinHashParams = MinHashParams(),
                 max_bucket_entries: int = MAX_BUCKET_ENTRIES, max_pending_edges: int = MAX_PENDING_EDGES):
        super().__init__(name)
        self.params = params
        self.hasher = MinHasher(params)
        self.max_bucket_entries = max_bucket_entries
        self.max_pending_edges = max_pending_edges

    def open_indexer(self, directory: Path, shard_index: int) -> MinHashIndexer:
        return MinHashIndexer(self.hasher, directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        signatures = SignatureStore(directory, shard_counts, self.params.num_hashes)
        source_pieces, target_pieces = [], []
        pending_edges = candidate_pairs = verified_pairs = 0
        for keys, global_ids in iter_key_buckets(directory, shard_counts, self.max_bucket_entries):
            if dropped is not None:
                live = ~dropped[global_ids]
                keys, global_ids = keys[live], global_ids[live]
            first_ids, other_ids = duplicate_pairs(keys, global_ids)
            candidate_pairs += len(first_ids)
            for start in range(0, len(first_ids), VERIFY_BATCH_PAIRS):
                sources = first_ids[start:start + VERIFY_BATCH_PAIRS]
                targets = other_ids[start:start + VERIFY_BATCH_PAIRS]
                agreement = (signatures.gather(sources) == signatures.gather(targets)).mean(axis=1)
                similar = agreement >= self.params.min_jaccard
                source_pieces.append(sources[similar])
                target_pieces.append(targets[similar])
                verified_pairs += int(np.count_nonzero(similar))
                pending_edges += int(np.count_nonzero(similar))
            if pending_edges > self.max_pending_edges:
                compacted = star_edges(np.concatenate(source_pieces), np.concatenate(target_pieces))
                source_pieces, target_pieces = [compacted[0]], [compacted[1]]
                pending_edges = len(compacted[0])
        with DuplicateOfWriter(directory, sum(shard_counts)) as duplicate_of:
            if verified_pairs:
                representatives, members = star_edges(np.concatenate(source_pieces), np.concatenate(target_pieces))
                duplicate_of[members] = representatives
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "candidate_pairs": candidate_pairs, "verified_pairs": verified_pairs}
