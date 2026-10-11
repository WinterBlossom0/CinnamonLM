"""Quality analysis of common-pile/oercommons (raw) against common-pile/oercommons_filtered.

Usage: python3 -I analyze.py RAW_SHARD FILTERED_SHARD OUT_JSON [OTHER_SOURCE_SHARD ...]
"""
import collections
import gzip
import hashlib
import json
import re
import sys

import numpy

RAW_PATH, FILTERED_PATH, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
OTHER_SOURCE_PATHS = sys.argv[4:]

ENGLISH_STOPWORDS = frozenset(
    "the of and to a in is that for it as was with be by on not he i this are or his from at which but have an "
    "they you were their one all we can her has there been if more when will would who so no she other its may "
    "these what them than some him time into only do about could also my any your how our then should each".split()
)

ISSUE_PATTERNS = {
    # Site navigation captured instead of the resource body.
    "site_chrome_login_block": re.compile(r"Sign in to see your Hubs"),
    "off2class_promo": re.compile(r"Off2Class"),
    "openstax_feedback_footer": re.compile(r"(?m)^Did you have an idea for improving this content\? We’d love your input\.$"),
    "markdown_table_residue_line": re.compile(r"(?m)^[ \t]*\|[ \t|\-:]*$"),
    "table_row_line": re.compile(r"(?m)^[ \t]*\|.*\|[ \t]*$"),
    "url": re.compile(r"https?://\S+|www\.\S+"),
    "url_only_line": re.compile(r"(?m)^[ \t]*(?:https?://|www\.)\S+[ \t]*$"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_us": re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.]\d{4}(?!\d)"),
    "replacement_char": re.compile("\ufffd"),
    "mojibake_cp1252": re.compile(r"â€|Ã[\x80-\xbf©¨¡]"),
    "html_or_js_residue": re.compile(r"</?(?:div|span|p|br|img|a|table|tr|td|iframe|script|style|html)\b[^>\n]*>|function\s+\w+\s*\([^)]*\)\s*\{"),
    "latex_or_katex_source": re.compile(r"\\(?:frac|text|times|mathrm|left|right|sqrt|cdot|displaystyle|begin)\b"),
    # MathML/MathJax dropped during HTML-to-text, leaving a sentence with a hole where a formula was.
    "dropped_math_orphan_punct_line": re.compile(r"(?m)^[,.;:][ \t]"),
    "dropped_math_where_is_the": re.compile(r",\s*where\s+(?:is|are)\s+(?:the|a|an)\b"),
    "dropped_math_equation_colon_then_prose": re.compile(r"(?im)(?:following|the|this)\s+(?:equations?|formulas?)\s*:\s*\n(?![^\n]*[=0-9+×÷−])"),
    # OpenStax figure/table numbers removed, leaving "(Figure)".
    "stripped_figure_ref": re.compile(r"\((?:Figure|Table)\)"),
    "embedded_nc_nd_license": re.compile(
        r"(?i)\bCC[ -]?BY[ -]?(?:NC|ND)\b|\bNon-?Commercial\b|\bNoDeriv(?:atives|s)\b|creativecommons\.org/licenses/by-n[cd]"
    ),
    # Stricter: an NC/ND licence stated for the text or the whole work, on a line that names no image or media credit.
    "nc_nd_work_level_declaration": re.compile(
        r"(?im)^(?![^\n]*(?:image|photo|picture|video|figure|flickr|poster|graphic|icon|logo|cover|map|music|song|clip|youtu|youtube|ted\.com|vimeo))"
        r"[^\n]*(?:except where otherwise noted|text source|this (?:work|book|text|resource|oer|chapter|course|module|unit|lesson)|"
        r"\bis licensed under|\bis shared under|\bLicense:)[^\n]{0,250}"
        r"(?:Non-?Commercial|NoDeriv(?:atives|s)|BY-NC|BY-ND)[^\n]*$"
    ),
    "all_rights_reserved": re.compile(r"(?i)all rights reserved"),
    "attribution_boilerplate_line": re.compile(
        r"(?im)^(?:CC LICENSED CONTENT[^\n]*|Provided by:[^\n]*|License:[^\n]*|Located at:[^\n]*|Authored by:[^\n]*|"
        r"Licenses? and Attributions?|Attributions?|Attribution and License|Licensing and Attribution|"
        r"[^\n]{0,200}is licensed under a Creative Commons[^\n]*|Images? courtesy of[^\n]*)$"
    ),
    "single_letter_answer_line": re.compile(r"(?m)^[A-E]$"),
}

OPENSTAX_TITLE_PREFIXES = (
    "Biology", "U.S. History", "Principles of Macroeconomics", "American Government", "Introduction to Sociology",
    "Psychology", "Anatomy and Physiology", "Texas Government", "Principles of Economics", "Concepts of Biology",
)

# Frequent function words of Spanish, Portuguese, French, Italian, Indonesian, Turkish and German; English homographs removed.
OTHER_LATIN_STOPWORDS = frozenset(
    "de la que el en y los las del se por una para con es al lo como más pero sus ya este porque esta entre cuando muy "
    "sin sobre también hasta hay donde desde todo nos durante todos uno les ni contra otros ese eso ante ellos esto "
    "do da em um os na não é ao das dos mais são le les des du et est une dans pour qui pas sur au avec ce il elle sont "
    "di che il della per è non dan yang ini itu dengan untuk ve bir bu için ile der die und ist nicht ein eine mit von zu".split()
)

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def load_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def metadata_of(record):
    metadata = record["metadata"]
    if isinstance(metadata, str):
        import ast
        metadata = ast.literal_eval(metadata)
    return metadata


def normalize_for_hash(text):
    return " ".join(text.lower().split())


def text_hash(text):
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()


def percentile_summary(values):
    ordered = numpy.sort(numpy.asarray(values, dtype=numpy.int64))
    summary = {f"p{p}": int(numpy.percentile(ordered, p, method="lower")) for p in PERCENTILES}
    summary["min"] = int(ordered[0])
    summary["max"] = int(ordered[-1])
    summary["mean"] = round(float(ordered.mean()), 1)
    return summary


def language_features(text):
    letters = [character for character in text if character.isalpha()]
    ascii_letter_ratio = (sum(1 for character in letters if character.isascii()) / len(letters)) if letters else 0.0
    tokens = re.findall(r"[a-zA-Z']+|\S+", text.lower())
    stopword_ratio = (sum(1 for token in tokens if token in ENGLISH_STOPWORDS) / len(tokens)) if tokens else 0.0
    other_ratio = (sum(1 for token in tokens if token in OTHER_LATIN_STOPWORDS) / len(tokens)) if tokens else 0.0
    return len(letters), ascii_letter_ratio, stopword_ratio, other_ratio


def is_non_english(text):
    letter_count, ascii_letter_ratio, stopword_ratio, other_ratio = language_features(text)
    if letter_count < 50:
        return None
    return (ascii_letter_ratio < 0.80) or ((other_ratio > stopword_ratio) and (other_ratio >= 0.05))


def overview_preamble_chars(text):
    """Chars of attachment/resource titles that OER Commons prints before the title line and 'Overview'."""
    lines = text.split("\n")
    for index, line in enumerate(lines[:30]):
        if line.strip() == "Overview":
            title_line_index = index - 1
            if title_line_index <= 0:
                return 0
            return sum(len(previous) + 1 for previous in lines[:title_line_index])
    return 0


def repeated_line_chars(text, minimum_length=20):
    seen = set()
    repeated = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < minimum_length:
            continue
        if stripped in seen:
            repeated += len(line) + 1
        else:
            seen.add(stripped)
    return repeated


def short_line_fraction(text):
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(1 for line in lines if len(line.split()) < 5) / len(lines)


def measure_issues(records):
    """Doc fraction and char fraction of every issue over the given records."""
    total_chars = sum(len(record["text"]) for record in records) or 1
    total_docs = len(records) or 1
    results = {}
    examples = collections.defaultdict(list)

    def add(name, doc_hit, chars, record, snippet):
        entry = results.setdefault(name, {"docs": 0, "chars": 0})
        if doc_hit:
            entry["docs"] += 1
            entry["chars"] += chars
            if len(examples[name]) < 4 and snippet is not None:
                examples[name].append(f"[{record['id']}] {snippet!r}")

    for record in records:
        text = record["text"]
        for name, pattern in ISSUE_PATTERNS.items():
            matches = list(pattern.finditer(text))
            if not matches:
                add(name, False, 0, record, None)
                continue
            if name in ("site_chrome_login_block", "off2class_promo"):
                chars = len(text)
            else:
                chars = sum(match.end() - match.start() for match in matches)
            first = matches[0]
            add(name, True, chars, record, text[max(0, first.start() - 100): first.end() + 100])
        add("empty_text", len(text) == 0, 0, record, "")
        word_count = len(text.split())
        add("under_50_words", 0 < word_count < 50, len(text), record, text[:200])
        preamble = overview_preamble_chars(text)
        add("overview_preamble_titles", preamble > 0, preamble, record, text[:preamble][:200])
        repeated = repeated_line_chars(text)
        add("intra_doc_repeated_lines_ge20chars", repeated > 0, repeated, record, None)
        fragment_fraction = short_line_fraction(text)
        add("short_line_fraction_gt_0.5", fragment_fraction > 0.5 and word_count >= 50, len(text), record, text[:200])
        non_english = is_non_english(text)
        add("non_english_heuristic", bool(non_english), len(text), record, text[:200])
        title = metadata_of(record).get("title", "")
        openstax = title.startswith(OPENSTAX_TITLE_PREFIXES) or ("OpenStax" in text)
        add("openstax_derived", openstax, len(text), record, None)

    report = {}
    for name, entry in results.items():
        report[name] = {
            "docs": entry["docs"],
            "doc_fraction": round(entry["docs"] / total_docs, 4),
            "chars": entry["chars"],
            "char_fraction": round(entry["chars"] / total_chars, 5),
            "examples": examples.get(name, []),
        }
    return report


def shingle_hashes(text, width=5):
    words = normalize_for_hash(text).split()
    if len(words) < width:
        return numpy.zeros(0, dtype=numpy.uint64)
    shingles = {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}
    return numpy.fromiter(
        (int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "little") for shingle in shingles),
        dtype=numpy.uint64,
        count=len(shingles),
    )


def minhash_signature(hashes, salts, multipliers, chunk=20000):
    signature = numpy.full(len(salts), numpy.iinfo(numpy.uint64).max, dtype=numpy.uint64)
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        # Multiplication wraps modulo 2**64; xor-salt plus odd multiplier gives one hash family member per row.
        permuted = (block[None, :] ^ salts[:, None]) * multipliers[:, None]
        signature = numpy.minimum(signature, permuted.min(axis=1))
    return signature


def near_duplicates(records, permutations=128, bands=32, threshold=0.8):
    rng = numpy.random.default_rng(12345)
    salts = rng.integers(0, 2**63, size=permutations, dtype=numpy.uint64)
    multipliers = rng.integers(0, 2**63, size=permutations, dtype=numpy.uint64) * numpy.uint64(2) + numpy.uint64(1)
    rows = permutations // bands
    signatures = {}
    shingle_sets = {}
    for index, record in enumerate(records):
        hashes = shingle_hashes(record["text"])
        if len(hashes) == 0:
            continue
        signatures[index] = minhash_signature(hashes, salts, multipliers)
        shingle_sets[index] = hashes
    buckets = collections.defaultdict(list)
    for index, signature in signatures.items():
        for band in range(bands):
            buckets[(band, signature[band * rows:(band + 1) * rows].tobytes())].append(index)
    candidate_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidate_pairs.add((members[i], members[j]))
    parent = {}

    def find(node):
        while parent.get(node, node) != node:
            node = parent[node]
        return node

    confirmed = []
    for left, right in candidate_pairs:
        a, b = shingle_sets[left], shingle_sets[right]
        intersection = len(numpy.intersect1d(a, b, assume_unique=True))
        jaccard = intersection / (len(a) + len(b) - intersection)
        if jaccard >= threshold:
            confirmed.append((left, right, jaccard))
            parent[find(left)] = find(right)
    clusters = collections.defaultdict(list)
    for node in {n for pair in confirmed for n in pair[:2]}:
        clusters[find(node)].append(node)
    docs_in_clusters = sum(len(members) for members in clusters.values())
    removable = docs_in_clusters - len(clusters)
    removable_chars = 0
    for members in clusters.values():
        lengths = sorted((len(records[m]["text"]) for m in members), reverse=True)
        removable_chars += sum(lengths[1:])
    examples = []
    for left, right, jaccard in sorted(confirmed, key=lambda item: item[2])[:3] + sorted(confirmed, key=lambda item: -len(records[item[0]]["text"]))[:3]:
        examples.append(
            f"J={jaccard:.3f} [{records[left]['id']}] {records[left]['text'][:120]!r} <-> [{records[right]['id']}] {records[right]['text'][:120]!r}"
        )
    return {
        "method": f"MinHash {permutations} perms over lowercased whitespace word 5-shingles (blake2b-64), LSH {bands}x{rows}, exact Jaccard verification >= {threshold}",
        "docs_with_shingles": len(signatures),
        "candidate_pairs": len(candidate_pairs),
        "confirmed_pairs": len(confirmed),
        "clusters": len(clusters),
        "docs_in_clusters": docs_in_clusters,
        "removable_docs_keep_longest": removable,
        "removable_chars": removable_chars,
        "examples": examples,
    }


SEGMENT_SPLIT = re.compile(r"(?<=[.!?])[ \t]+|\n")


def segments(text):
    return [piece.strip() for piece in SEGMENT_SPLIT.split(text) if piece.strip()]


def compare_with_filtered(raw_records, filtered_records):
    raw_by_id = collections.defaultdict(list)
    for record in raw_records:
        raw_by_id[record["id"]].append(record)
    filtered_ids = collections.Counter(record["id"] for record in filtered_records)
    dropped_ids = [identifier for identifier in raw_by_id if identifier not in filtered_ids]
    dropped_records = [raw_by_id[identifier][0] for identifier in dropped_ids]
    kept_unique_records = [raw_by_id[identifier][0] for identifier in raw_by_id if identifier in filtered_ids]

    def profile(records):
        word_counts = [len(record["text"].split()) for record in records]
        non_english = sum(1 for record in records if is_non_english(record["text"]))
        chrome = sum(1 for record in records if ISSUE_PATTERNS["site_chrome_login_block"].search(record["text"]))
        return {
            "docs": len(records),
            "chars": sum(len(record["text"]) for record in records),
            "words": percentile_summary(word_counts) if word_counts else {},
            "empty": sum(1 for count in word_counts if count == 0),
            "under_50_words": sum(1 for count in word_counts if count < 50),
            "under_100_words": sum(1 for count in word_counts if count < 100),
            "non_english_heuristic": non_english,
            "site_chrome": chrome,
        }

    segment_doc_frequency = collections.Counter()
    seen_texts = set()
    for record in raw_records:
        if record["text"] in seen_texts:
            continue
        seen_texts.add(record["text"])
        segment_doc_frequency.update(set(segments(record["text"])))

    identical = modified = 0
    deleted_segment_categories = collections.Counter()
    deleted_chars = 0
    raw_chars_of_modified = 0
    filtered_chars_of_modified = 0
    modified_examples = []
    for record in filtered_records:
        candidates = raw_by_id[record["id"]]
        if any(candidate["text"] == record["text"] for candidate in candidates):
            identical += 1
            continue
        modified += 1
        raw_text = min(candidates, key=lambda candidate: abs(len(candidate["text"]) - len(record["text"])))["text"]
        raw_chars_of_modified += len(raw_text)
        filtered_chars_of_modified += len(record["text"])
        local_counts = collections.Counter(segments(raw_text))
        for segment in local_counts:
            if segment in record["text"]:
                continue
            if "@" in segment:
                deleted_segment_categories["contains_@_(pii_redaction)"] += 1
                continue
            deleted_chars += len(segment) * local_counts[segment]
            frequency = segment_doc_frequency[segment]
            if frequency == 1 and local_counts[segment] > 1:
                category = "unique_to_doc_repeated_inside_doc"
            elif frequency == 1:
                category = "unique_to_doc_single_occurrence"
            elif frequency <= 5:
                category = "in_2_to_5_raw_docs"
            else:
                category = "in_over_5_raw_docs"
            deleted_segment_categories[category] += 1
            if len(modified_examples) < 6 and category == "unique_to_doc_single_occurrence":
                modified_examples.append(f"[{record['id']}] deleted: {segment[:200]!r}")
    placeholders = collections.Counter(match for record in filtered_records for match in re.findall(r"<[A-Z_]{4,}>", record["text"]))
    placeholder_docs = sum(1 for record in filtered_records if re.search(r"<(?:EMAIL_ADDRESS|PHONE_NUMBER|IP_ADDRESS)>", record["text"]))
    leftover_space_artifacts = sum(1 for record in filtered_records if re.search(r"(?m)(?:^ \S| $)", record["text"]))
    leftover_space_artifacts_raw = sum(1 for record in kept_unique_records if re.search(r"(?m)(?:^ \S| $)", record["text"]))
    return {
        "raw_docs": len(raw_records),
        "raw_unique_ids": len(raw_by_id),
        "filtered_docs": len(filtered_records),
        "filtered_unique_ids": len(filtered_ids),
        "filtered_ids_with_multiple_rows": [identifier for identifier, count in filtered_ids.items() if count > 1],
        "filtered_ids_not_in_raw": sum(1 for identifier in filtered_ids if identifier not in raw_by_id),
        "dropped_unique_ids": len(dropped_ids),
        "dropped_profile": profile(dropped_records),
        "kept_profile": profile(kept_unique_records),
        "dropped_examples": [f"[{record['id']}] {record['text'][:160]!r}" for record in dropped_records if len(record["text"].split()) >= 150][:8],
        "kept_text_identical_to_a_raw_copy": identical,
        "kept_text_modified": modified,
        "modified_raw_chars": raw_chars_of_modified,
        "modified_filtered_chars": filtered_chars_of_modified,
        "deleted_segment_categories": dict(deleted_segment_categories),
        "deleted_segment_chars_excluding_pii": deleted_chars,
        "deleted_examples": modified_examples,
        "pii_placeholders": dict(placeholders),
        "docs_with_pii_placeholders": placeholder_docs,
        "docs_with_space_artifacts_filtered": leftover_space_artifacts,
        "docs_with_space_artifacts_raw_kept": leftover_space_artifacts_raw,
    }


def cross_source_overlap(records, other_paths, minimum_line_chars=100):
    """Share of docs whose long lines appear verbatim in another Common Pile source."""
    line_owner = collections.defaultdict(set)
    doc_long_lines = {}
    for index, record in enumerate(records):
        long_lines = {line.strip() for line in record["text"].split("\n") if len(line.strip()) >= minimum_line_chars}
        if long_lines:
            doc_long_lines[index] = len(long_lines)
            for line in long_lines:
                line_owner[hashlib.sha1(line.encode("utf-8")).digest()].add(index)
    results = {}
    union_matched_lines = collections.defaultdict(set)
    for path in other_paths:
        matched_per_doc = collections.Counter()
        seen_lines = set()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for raw_line in handle:
                text = json.loads(raw_line)["text"]
                for line in text.split("\n"):
                    stripped = line.strip()
                    if len(stripped) < minimum_line_chars:
                        continue
                    digest = hashlib.sha1(stripped.encode("utf-8")).digest()
                    if digest in line_owner and digest not in seen_lines:
                        seen_lines.add(digest)
                        for owner in line_owner[digest]:
                            matched_per_doc[owner] += 1
                            union_matched_lines[owner].add(digest)
        half_or_more = [index for index, count in matched_per_doc.items() if count / doc_long_lines[index] >= 0.5]
        results[path] = {
            "docs_with_long_lines": len(doc_long_lines),
            "docs_with_any_long_line_match": len(matched_per_doc),
            "docs_with_ge50pct_long_lines_matched": len(half_or_more),
            "chars_of_ge50pct_docs": sum(len(records[index]["text"]) for index in half_or_more),
            "examples": [f"[{records[index]['id']}] {metadata_of(records[index]).get('title', '')[:80]!r}" for index in half_or_more[:6]],
        }
    union_half = [index for index, lines in union_matched_lines.items() if len(lines) / doc_long_lines[index] >= 0.5]
    results["union_of_all_other_sources"] = {
        "docs_with_any_long_line_match": len(union_matched_lines),
        "docs_with_ge50pct_long_lines_matched": len(union_half),
        "chars_of_ge50pct_docs": sum(len(records[index]["text"]) for index in union_half),
        "total_docs": len(records),
        "total_chars": sum(len(record["text"]) for record in records),
        "openstax_title_share_of_ge50pct_docs": round(
            sum(1 for index in union_half if metadata_of(records[index]).get("title", "").startswith(OPENSTAX_TITLE_PREFIXES)) / max(1, len(union_half)), 3
        ),
    }
    return results


def main():
    raw_records = load_jsonl_gz(RAW_PATH)
    filtered_records = load_jsonl_gz(FILTERED_PATH)
    report = {}

    ids = collections.Counter(record["id"] for record in raw_records)
    hashes = [text_hash(record["text"]) for record in raw_records]
    hash_counts = collections.Counter(hashes)
    exact_text_counts = collections.Counter(record["text"] for record in raw_records)
    dup_ids_same_text = sum(
        1 for identifier, count in ids.items()
        if count > 1 and len({record["text"] for record in raw_records if record["id"] == identifier}) == 1
    ) if len(raw_records) < 50000 else None
    seen = set()
    unique_records = []
    for record, digest in zip(raw_records, hashes):
        if digest in seen:
            continue
        seen.add(digest)
        unique_records.append(record)
    report["counts"] = {
        "raw_docs": len(raw_records),
        "unique_ids": len(ids),
        "copies_per_id_histogram": dict(sorted(collections.Counter(ids.values()).items())),
        "duplicate_ids_whose_copies_have_identical_text": dup_ids_same_text,
        "unique_exact_texts": len(exact_text_counts),
        "unique_normalized_text_hashes": len(hash_counts),
        "docs_that_are_extra_normalized_copies": len(raw_records) - len(hash_counts),
        "chars_in_extra_normalized_copies": sum(len(record["text"]) for record in raw_records) - sum(len(record["text"]) for record in unique_records),
        "total_chars_raw": sum(len(record["text"]) for record in raw_records),
        "total_chars_unique": sum(len(record["text"]) for record in unique_records),
        "total_words_raw": sum(len(record["text"].split()) for record in raw_records),
        "total_words_unique": sum(len(record["text"].split()) for record in unique_records),
        "distinct_texts_sharing_a_normalized_hash": len(exact_text_counts) - len(hash_counts),
    }
    report["length_raw"] = {
        "chars": percentile_summary([len(record["text"]) for record in raw_records]),
        "words": percentile_summary([len(record["text"].split()) for record in raw_records]),
    }
    report["length_unique"] = {
        "chars": percentile_summary([len(record["text"]) for record in unique_records]),
        "words": percentile_summary([len(record["text"].split()) for record in unique_records]),
    }
    report["length_filtered"] = {
        "chars": percentile_summary([len(record["text"]) for record in filtered_records]),
        "words": percentile_summary([len(record["text"].split()) for record in filtered_records]),
    }

    licenses = collections.Counter(metadata_of(record).get("license") for record in raw_records)
    licenses_unique = collections.Counter(metadata_of(record).get("license") for record in unique_records)
    licenses_filtered = collections.Counter(metadata_of(record).get("license") for record in filtered_records)
    report["licenses"] = {"raw": dict(licenses), "unique": dict(licenses_unique), "filtered": dict(licenses_filtered)}
    report["metadata_keys"] = dict(collections.Counter(key for record in raw_records for key in metadata_of(record)))
    report["source_values"] = dict(collections.Counter(record.get("source") for record in raw_records))
    report["created_formats"] = dict(collections.Counter(re.sub(r"\d", "9", str(record.get("created"))) for record in raw_records))
    report["id_url_mismatch"] = sum(1 for record in raw_records if metadata_of(record).get("url") != record["id"])

    letter_ratios = []
    stop_ratios = []
    other_ratios = []
    for record in unique_records:
        letter_count, ascii_ratio, stop_ratio, other_ratio = language_features(record["text"])
        if letter_count >= 50:
            letter_ratios.append(ascii_ratio)
            stop_ratios.append(stop_ratio)
            other_ratios.append(other_ratio)
    report["language"] = {
        "method": "docs with >=50 letters; non-English if ASCII share of letters < 0.80 OR (other-Latin-language stopword share > English stopword share AND >= 0.05)",
        "docs_measured_unique": len(letter_ratios),
        "ascii_letter_ratio_lt_0.80": int(sum(1 for value in letter_ratios if value < 0.80)),
        "latin_other_language": int(sum(1 for value, stop, other in zip(letter_ratios, stop_ratios, other_ratios) if value >= 0.80 and other > stop and other >= 0.05)),
        "non_english_total": int(sum(1 for value, stop, other in zip(letter_ratios, stop_ratios, other_ratios) if value < 0.80 or (other > stop and other >= 0.05))),
        "stopword_ratio_lt_0.08": int(sum(1 for value in stop_ratios if value < 0.08)),
        "stopword_ratio_lt_0.15": int(sum(1 for value in stop_ratios if value < 0.15)),
        "stopword_ratio_percentiles": {f"p{p}": round(float(numpy.percentile(stop_ratios, p)), 3) for p in PERCENTILES},
    }

    report["issues_raw_all"] = measure_issues(raw_records)
    report["issues_raw_unique"] = measure_issues(unique_records)
    report["issues_filtered"] = measure_issues(filtered_records)
    report["near_duplicates_unique"] = near_duplicates(unique_records)
    report["near_duplicates_filtered"] = near_duplicates(filtered_records)
    report["filtered_comparison"] = compare_with_filtered(raw_records, filtered_records)
    if OTHER_SOURCE_PATHS:
        report["cross_source_overlap_unique"] = cross_source_overlap(unique_records, OTHER_SOURCE_PATHS)
        report["cross_source_overlap_filtered"] = cross_source_overlap(filtered_records, OTHER_SOURCE_PATHS)

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
