import re

import numpy as np

from data_pipeline.common.dedup import (
    ExactDedup, MinHasher, MinHashParams, NearDedup, SeenSet, dedup_words, duplicate_pairs, exact_key,
    load_duplicate_of, star_edges,
)


def true_jaccard(first: str, second: str, size: int = 5) -> float:
    def shingles(text):
        words = dedup_words(text)
        return {" ".join(words[start:start + size]) for start in range(max(1, len(words) - size + 1))}
    first_set, second_set = shingles(first), shingles(second)
    return len(first_set & second_set) / len(first_set | second_set)


def test_dedup_words_normalization():
    assert dedup_words("Ça, c'est  l'Été!\nOK") == ["ca", "c", "est", "l", "ete", "ok"]


def test_exact_key_ignores_case_punctuation_accents_whitespace(snippets):
    text = snippets["english"]["text"]
    variant = re.sub(r"\s+", "  ", text.upper()).replace(",", "")
    assert exact_key(text) == exact_key(variant)
    assert exact_key("Table 1: 2019") != exact_key("Table 1: 2020")
    assert exact_key("...  ") is None


def test_identical_texts_share_signature_and_band_keys(snippets):
    hasher = MinHasher()
    text = snippets["near_duplicate_pair"]["texts"][0]
    assert np.array_equal(hasher.signature(text), hasher.signature(text.upper()))
    signature = hasher.signature(text)
    assert signature.dtype == np.uint32 and signature.shape == (112,)
    keys = hasher.band_keys(signature)
    assert keys.shape == (14,) and len(set(keys.tolist())) == 14


def test_signature_agreement_estimates_jaccard_on_real_pairs(snippets):
    params = MinHashParams(num_bands=64, rows_per_band=8)
    hasher = MinHasher(params)
    for pair in ("near_duplicate_pair", "distinct_pair"):
        first, second = snippets[pair]["texts"]
        estimate = float((hasher.signature(first) == hasher.signature(second)).mean())
        assert abs(estimate - true_jaccard(first, second)) < 0.06


def test_short_and_empty_texts():
    hasher = MinHasher()
    assert hasher.signature("") is None
    assert hasher.signature("--- !!") is None
    assert np.array_equal(hasher.signature("two words"), hasher.signature("Two, words."))


def test_duplicate_pairs_keeps_smallest_id_per_group():
    keys = np.array([7, 3, 7, 3, 9, 7], dtype=np.uint64)
    ids = np.array([10, 4, 2, 8, 1, 6], dtype=np.int64)
    kept, removed = duplicate_pairs(keys, ids)
    assert sorted(zip(kept.tolist(), removed.tolist())) == [(2, 6), (2, 10), (4, 8)]


def test_star_edges_joins_chains_into_components():
    representatives, members = star_edges(np.array([5, 9, 30]), np.array([9, 12, 31]))
    assert sorted(zip(representatives.tolist(), members.tolist())) == [(5, 9), (5, 12), (30, 31)]


def test_shingles_are_distinct_word_ngrams(snippets):
    hasher = MinHasher()
    words = dedup_words(snippets["english"]["text"])
    distinct_ngrams = {tuple(words[start:start + 5]) for start in range(len(words) - 4)}
    assert len(hasher.shingles(words)) == len(distinct_ngrams)
    assert len(hasher.shingles(["two", "words"])) == 1
    assert not np.array_equal(hasher.shingles(["a", "b", "c", "d", "e"]), hasher.shingles(["b", "a", "c", "d", "e"]))


def test_seen_set_keeps_first_global_id_across_batches(tmp_path):
    seen = SeenSet(tmp_path / "seen.sqlite")
    first = seen.first_seen(np.array([5, 2**64 - 1, 5, 9], dtype=np.uint64), np.array([0, 1, 2, 3]))
    assert first.tolist() == [-1, -1, 0, -1]
    second = seen.first_seen(np.array([9, 2**64 - 1, 11, 11], dtype=np.uint64), np.array([4, 5, 6, 7]))
    assert second.tolist() == [3, 1, -1, 6]
    seen.close()
    reopened = SeenSet(tmp_path / "seen.sqlite")
    assert reopened.first_seen(np.array([11], dtype=np.uint64), np.array([8])).tolist() == [6]
    reopened.close()


def index_corpus(deduplicator, directory, shards: list[list[str]]) -> list[int]:
    directory.mkdir(parents=True, exist_ok=True)
    for shard_index, texts in enumerate(shards):
        indexer = deduplicator.open_indexer(directory, shard_index)
        for local_id, text in enumerate(texts):
            indexer.add(local_id, {"text": text})
        indexer.close()
    return [len(texts) for texts in shards]


def test_exact_dedup_build_marks_later_copies(tmp_path, snippets):
    english = snippets["english"]["text"]
    spanish = snippets["spanish"]["text"]
    shards = [[english, "...", spanish], [], [spanish.upper(), "...", english + " ", "other"]]
    deduplicator = ExactDedup(seen_batch=2)
    counts = index_corpus(deduplicator, tmp_path, shards)
    assert deduplicator.build(tmp_path, counts) == {"duplicates": 2}
    assert load_duplicate_of(tmp_path).tolist() == [-1, -1, -1, 2, -1, 0, -1]
    assert not (tmp_path / "seen.sqlite").exists()
    dropped = np.zeros(7, dtype=bool)
    dropped[0] = True
    deduplicator.build(tmp_path, counts, dropped)
    assert load_duplicate_of(tmp_path).tolist() == [-1, -1, -1, 2, -1, -1, -1]


def test_near_dedup_build_on_real_pairs_with_compaction(tmp_path, snippets):
    near_first, near_second = snippets["near_duplicate_pair"]["texts"]
    distinct_first, distinct_second = snippets["distinct_pair"]["texts"]
    english = snippets["english"]["text"]
    shards = [[near_first, distinct_first], [english + " Extra closing sentence."], [near_second, distinct_second,
                                                                                      english]]
    deduplicator = NearDedup(max_bucket_entries=8, max_pending_edges=0)
    counts = index_corpus(deduplicator, tmp_path, shards)
    summary = deduplicator.build(tmp_path, counts)
    assert summary["duplicates"] == 2 and summary["verified_pairs"] >= 2
    assert load_duplicate_of(tmp_path).tolist() == [-1, -1, -1, 0, -1, 2]
    dropped = np.array([True, False, False, False, False, False])
    deduplicator.build(tmp_path, counts, dropped)
    assert load_duplicate_of(tmp_path).tolist() == [-1, -1, -1, -1, -1, 2]
