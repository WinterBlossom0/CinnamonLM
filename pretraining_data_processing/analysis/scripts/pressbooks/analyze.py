"""Quality analysis of common-pile/pressbooks (raw) and comparison with pressbooks_filtered.

Run: PYTHONHASHSEED=0 python3 -I analyze.py > report.txt
Reads raw shards read-only; writes only into this script's directory.
"""
import collections
import difflib
import gzip
import hashlib
import json
import os
import random
import re
import sys

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    # Word hashes for MinHash use the builtin str hash; a fixed seed makes signatures reproducible.
    os.execve(sys.executable, [sys.executable, "-I"] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "common_pile", "pressbooks") + os.sep
RAW_SHARDS = ["00000_pressbooks.json.gz", "00001_pressbooks.json.gz"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED = os.path.join(OUT_DIR, "filtered", "pressbooks-0000.json.gz")
EXAMPLES_PER_ISSUE = 4
SMOKE_LIMIT = int(os.environ.get("PB_SMOKE_LIMIT", "0"))  # 0 means all docs; >0 caps docs per shard for a quick check
rng = random.Random(0)


def assert_shards_complete():
    cache = os.path.join(RAW_DIR, ".cache")
    for root, _, files in os.walk(cache):
        for name in files:
            if name.endswith(".incomplete"):
                raise SystemExit(f"incomplete download present: {os.path.join(root, name)}")


def iter_raw():
    for shard in RAW_SHARDS:
        with gzip.open(RAW_DIR + shard, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if SMOKE_LIMIT and line_number >= SMOKE_LIMIT:
                    break
                yield shard, line_number, json.loads(line)


def iter_filtered():
    with gzip.open(FILTERED, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize_for_exact_dup(text):
    return " ".join(text.lower().split())


def book_key(doc):
    url = doc["id"]
    match = re.match(r"https?://([^/]+)/([^/]+)/", url)
    return f"{match.group(1)}/{match.group(2)}" if match else url


# ---------------------------------------------------------------- language heuristics
STOPWORDS = {
    "en": "the of and to in is that for it as with was on are by this be from or an which at not have but they can their".split(),
    "es": "de la que el en los del se las por un para con una su al es lo como más pero sus le ya o este".split(),
    "fr": "de la le et les des en un du une que est pour qui dans par sur au pas plus ne se ce il avec".split(),
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden".split(),
    "pt": "de a o que e do da em um para é com não uma os no se na por mais as dos como mas ao".split(),
    "it": "di e il la che in a per un è del non della le si una con da i sono gli al alla anche".split(),
    "nl": "de en van het een in is dat op te zijn met voor niet die aan er om ook als bij door".split(),
}
# One-letter words ("a", "e", "o") are dropped: they also occur as list markers in English text.
STOPWORD_SETS = {lang: {word for word in words if len(word) > 1} for lang, words in STOPWORDS.items()}
MIN_WORDS_FOR_LANGUAGE = 30
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def script_of(char):
    code = ord(char)
    if code < 0x250:
        return "latin"
    if 0x370 <= code < 0x400:
        return "greek"
    if 0x400 <= code < 0x530:
        return "cyrillic"
    if 0x590 <= code < 0x600:
        return "hebrew"
    if 0x600 <= code < 0x700:
        return "arabic"
    if 0x900 <= code < 0xE00:
        return "indic"
    if 0x1400 <= code < 0x1680:
        return "canadian_syllabics"
    if 0x1E00 <= code < 0x1F00:
        return "latin"
    if 0x3040 <= code < 0x3100 or 0x4E00 <= code < 0xA000 or 0x3400 <= code < 0x4DC0 or 0xAC00 <= code < 0xD7B0:
        return "cjk"
    return "other"


def language_profile(text):
    """Returns (label, ascii_letter_ratio, english_stopword_ratio)."""
    sample = text[:20000]
    letters = [char for char in sample if char.isalpha()]
    if not letters:
        return "no_letters", 0.0, 0.0
    ascii_ratio = sum(1 for char in letters if char < "\x80") / len(letters)
    scripts = collections.Counter(script_of(char) for char in letters)
    top_script, top_count = scripts.most_common(1)[0]
    words = [word.lower() for word in WORD_RE.findall(sample)]
    ratios = {lang: (sum(1 for word in words if word in stop) / len(words) if words else 0.0) for lang, stop in STOPWORD_SETS.items()}
    if top_script != "latin" and top_count / len(letters) >= 0.5:
        return f"script:{top_script}", ascii_ratio, ratios["en"]
    if len(words) < MIN_WORDS_FOR_LANGUAGE:
        return "too_short", ascii_ratio, ratios["en"]
    best = max(ratios, key=ratios.get)
    if ratios[best] < 0.08:
        return "unknown_latin", ascii_ratio, ratios["en"]
    return best, ascii_ratio, ratios["en"]


# ---------------------------------------------------------------- issue detectors
# Line-level detectors classify whole lines (after strip); span detectors match substrings.
PROMO_LINE = "Want to create or adapt books like this? Learn more about how Pressbooks supports open publishing practices."
NAV_START = "Primary Navigation"
NAV_END = "Previous/next navigation"
PIPE_ONLY_RE = re.compile(r"^[|\-: ]*\|[|\-: ]*$")
TABLE_ROW_RE = re.compile(r"^\|.*\|$|^\|\s|\s\|$")
LICENSE_LINE_RE = re.compile(
    r"(?i)(licen[sc]ed under|creative commons|\bCC[ -]?BY\b|\bCC0\b|except where otherwise noted|"
    r"to the extent possible under law|all rights reserved|waived all copyright|is licensed under|"
    r"^licen[sc]e:|^(?:text |media )?attributions?$|^licen[sc]es? and attributions|^cc licensed content|"
    r"^public domain content|^all rights reserved content|^licen[sc]e$)"
)
NONPERMISSIVE_RE = re.compile(
    r"\b[Cc][Cc][ -]BY[ -](?:NC|ND)\b|\b[Bb][Yy]-[Nn][CcDd]\b|NonCommercial|Non-Commercial|NoDerivatives|NoDerivs|"
    r"(?i:all rights reserved|license: copyright|used with permission|reprinted (?:here )?(?:with|by) permission)"
)
MEDIA_ATTR_HEADER = "Media Attributions"
FOOTNOTE_LINE_RE = re.compile(r"↵\s*$")
CAPTION_LINE_RE = re.compile(
    r"^(?:Figure|Fig\.|Image|Photo|Picture|Illustration|Diagram)\s*\d+(?:[.\-–]\d+)*[a-z]?\s*[:.\-–|]\s|"
    r"\((?:image |photo |photograph )?credit[s]?\s*:?[^)]*\)|"
    r"^(?:image|photo|picture) (?:credit|by|source|courtesy)\b|"
    r"^(?:image|long|figure|text) description\b|^alt[- ]text\b",
    re.IGNORECASE,
)
IMAGE_FILENAME_LINE_RE = re.compile(r"^- ?\S*(?:\d{2,4}px-|\.(?:jpe?g|png|gif|svg|webp|tiff?)\b|_[A-Za-z0-9]+_[A-Za-z0-9_]+$)", re.IGNORECASE)
UI_LINE_RE = re.compile(
    r"^(?:(?:show|hide|reveal|toggle) (?:answers?|solutions?|glossary|hints?|explanation|feedback|transcript)|"
    r"licenses and attributions \(click to expand\)|\(?click to expand\)?|"
    r"jump to the next .* box\..*|check your answers?|"
    r"an interactive (?:h5p )?element has been excluded.*|you can view it online here.*)$",
    re.IGNORECASE,
)
LATEX_RE = re.compile(r"\[latex\].{0,4000}?\[/latex\]|\[/?latexpage\]", re.DOTALL)
MATHML_JUNK_RE = re.compile(r"\\phantom\{\\rule\{[^{}]*\}\{[^{}]*\}\}")
SHORTCODE_RE = re.compile(
    r"\[/?(?:h5p|table|caption|embed|audio|video|footnote|glossary|pb_glossary|textbox|qrcode|iframe|gallery|"
    r"wpvideo|youtube|vimeo|pdf-embedder|pdfviewer|anchor|heading|reveal-answer|hidden-answer|equation)(?:\s[^\]\n]{0,200})?\]"
)
H5P_RE = re.compile(r"(?i)\[h5p[^\]\n]{0,60}\]|interactive h5p element|h5p\.(?:com|org)|\bh5p\b")
DANGLING_EMBED_RE = re.compile(
    r"(?i)\b(?:the|this) (?:following|below|embedded) (?:video|quiz|activity|simulation|interactive|exercise|flashcards?|"
    r"podcast|audio|slideshow|presentation|game|self-check|knowledge check)|"
    r"\b(?:video|quiz|activity|simulation|interactive|exercise|flashcards?|podcast|audio clip|slideshow|game|self-check|"
    r"knowledge check)s? (?:below|above)\b|"
    r"^interactive learning activity\b|\bcomplete the (?:quiz|activity|exercise)s? below\b"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+\.\S+")
MOJIBAKE_RE = re.compile(r"â€[™œ\x9d“”˜¦]|Ã[©¨¢§±¶¼]|ï¿½|\ufffd")
SECTION_ID_RE = re.compile(r"#chapter-\d+-section-\d+$")
LEADING_NUMBER_RE = re.compile(r"^\d{1,3} \S")

LINE_ISSUES = ["promo_line", "nav_chrome", "pipe_only_debris", "table_row", "license_attribution_line",
               "media_attribution_list", "footnote_line", "caption_or_credit_line", "ui_widget_line",
               "dangling_embed_reference", "repeated_line_within_doc"]
SPAN_ISSUES = ["latex_shortcode", "mathml_phantom_junk", "wp_shortcode", "h5p_mention", "url", "mojibake",
               "nonpermissive_license_mention"]
REMOVABLE = {"promo_line", "nav_chrome", "pipe_only_debris", "license_attribution_line", "media_attribution_list",
             "ui_widget_line", "repeated_line_within_doc"}


def classify_lines(text):
    """Returns dict issue -> (chars, example_line) for line-level issues and removable-line chars."""
    lines = text.split("\n")
    result = {}
    removable_chars = 0
    in_nav = False
    in_media = False
    seen_long_lines = set()
    for raw_line in lines:
        line = raw_line.strip()
        size = len(raw_line) + 1
        hits = []
        if line == NAV_START:
            in_nav = True
        if in_nav:
            hits.append("nav_chrome")
            if line == NAV_END:
                in_nav = False
        if line == PROMO_LINE:
            hits.append("promo_line")
        if line == MEDIA_ATTR_HEADER:
            in_media = True
        elif in_media:
            if line.startswith("- ") or line.startswith("-\t") or line == "-":
                hits.append("media_attribution_list")
            else:
                in_media = False
        if line and PIPE_ONLY_RE.match(line):
            hits.append("pipe_only_debris")
        elif line and TABLE_ROW_RE.search(line):
            hits.append("table_row")
        if line and len(line) < 600 and LICENSE_LINE_RE.search(line):
            hits.append("license_attribution_line")
        if FOOTNOTE_LINE_RE.search(line):
            hits.append("footnote_line")
        if line and (CAPTION_LINE_RE.search(line) or (in_media and IMAGE_FILENAME_LINE_RE.match(line))):
            hits.append("caption_or_credit_line")
        if line and UI_LINE_RE.match(line):
            hits.append("ui_widget_line")
        if line and DANGLING_EMBED_RE.search(line):
            hits.append("dangling_embed_reference")
        if len(line) >= 30:
            if line in seen_long_lines:
                hits.append("repeated_line_within_doc")
            else:
                seen_long_lines.add(line)
        for issue in hits:
            chars, example = result.get(issue, (0, None))
            result[issue] = (chars + size, example if example is not None else line[:240])
        if any(issue in REMOVABLE for issue in hits):
            removable_chars += size
    return result, removable_chars


def span_issues(text):
    result = {}
    for issue, pattern in (("latex_shortcode", LATEX_RE), ("mathml_phantom_junk", MATHML_JUNK_RE),
                           ("wp_shortcode", SHORTCODE_RE), ("h5p_mention", H5P_RE), ("url", URL_RE),
                           ("mojibake", MOJIBAKE_RE), ("nonpermissive_license_mention", NONPERMISSIVE_RE)):
        chars = 0
        example = None
        for match in pattern.finditer(text):
            chars += match.end() - match.start()
            if example is None:
                start = max(0, match.start() - 80)
                example = text[start: match.end() + 80].replace("\n", "\\n")
        if chars:
            result[issue] = (chars, example)
    return result


# ---------------------------------------------------------------- MinHash (one-permutation hashing)
NUM_BINS = 128
BANDS, ROWS = 16, 8  # LSH S-curve midpoint about (1/16)^(1/8) = 0.71 Jaccard
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(values):
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def minhash_signature(text):
    """One-permutation MinHash with rotation densification (Shrivastava 2017) over word 5-shingles."""
    words = text.lower().split()
    if not words:
        words = ["<empty>"]
    word_hashes = np.array([hash(word) for word in words], dtype=np.int64).view(np.uint64)
    if len(word_hashes) < 5:
        shingles = splitmix64(np.array([np.bitwise_xor.reduce(word_hashes)], dtype=np.uint64))
    else:
        with np.errstate(over="ignore"):
            prime = np.uint64(1099511628211)
            combined = word_hashes[:-4].copy()
            for offset in range(1, 5):
                combined = combined * prime + word_hashes[offset: len(word_hashes) - 4 + offset]
            shingles = splitmix64(combined)
    bins = (shingles & np.uint64(NUM_BINS - 1)).astype(np.int64)
    values = (shingles >> np.uint64(32)).astype(np.uint32)
    signature = np.full(NUM_BINS, np.iinfo(np.uint32).max, dtype=np.uint64)
    np.minimum.at(signature, bins, values.astype(np.uint64))
    empty = signature == np.iinfo(np.uint32).max
    if empty.any() and not empty.all():
        filled = np.where(~empty)[0]
        for index in np.where(empty)[0]:
            position = np.searchsorted(filled, index) % len(filled)
            source = filled[position]
            distance = (source - index) % NUM_BINS
            signature[index] = (signature[source] + np.uint64(distance) * np.uint64(0x9E3779B1)) & np.uint64(0xFFFFFFFF)
    return signature.astype(np.uint32)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[max(root_first, root_second)] = min(root_first, root_second)


def percentiles(values):
    array = np.asarray(values)
    points = [1, 5, 25, 50, 75, 95, 99]
    return {f"p{point}": int(np.percentile(array, point)) for point in points} | {"max": int(array.max()), "mean": round(float(array.mean()), 1)}


def keep_example(store, issue, doc_id, snippet):
    bucket = store.setdefault(issue, [])
    if len(bucket) < 400:
        bucket.append(f"[{doc_id}] {snippet}")


def pick_examples(store):
    return {issue: rng.sample(bucket, min(EXAMPLES_PER_ISSUE, len(bucket))) for issue, bucket in store.items()}


# ---------------------------------------------------------------- pass 1: raw corpus
def analyze_raw():
    records = []  # per doc: tuple fields below
    signatures = []
    issue_docs = collections.Counter()
    issue_chars = collections.Counter()
    examples = {}
    exact_groups = collections.defaultdict(list)
    licenses = collections.Counter()
    languages = collections.Counter()
    language_chars = collections.Counter()
    id_counts = collections.Counter()
    removable_total = 0
    total_chars = 0
    for index, (shard, line_number, doc) in enumerate(iter_raw()):
        text = doc["text"]
        doc_id = doc["id"]
        id_counts[doc_id] += 1
        total_chars += len(text)
        licenses[doc["metadata"].get("license")] += 1
        label, ascii_ratio, en_ratio = language_profile(text)
        languages[label] += 1
        language_chars[label] += len(text)
        if label != "en":
            keep_example(examples, f"lang:{label}", doc_id, text[:200].replace("\n", "\\n"))
        line_result, removable_chars = classify_lines(text)
        removable_total += removable_chars
        flags = set()
        for issue, (chars, example) in list(line_result.items()) + list(span_issues(text).items()):
            issue_docs[issue] += 1
            issue_chars[issue] += chars
            flags.add(issue)
            keep_example(examples, issue, doc_id, example)
        if LEADING_NUMBER_RE.match(text):
            issue_docs["leading_chapter_number_title"] += 1
            flags.add("leading_chapter_number_title")
            keep_example(examples, "leading_chapter_number_title", doc_id, text[:80].replace("\n", "\\n"))
        if SECTION_ID_RE.search(doc_id):
            flags.add("section_anchor_id")
        non_table = sum(line_result.get(name, (0, None))[0] for name in ("pipe_only_debris", "table_row"))
        if len(text) and non_table / len(text) > 0.5:
            issue_docs["table_dominated_doc(>50%_chars)"] += 1
            issue_chars["table_dominated_doc(>50%_chars)"] += len(text)
        lines = text.split("\n")
        bare_pipe_lines = sum(1 for line in lines if line.strip() in ("|", "||", "|||"))
        if bare_pipe_lines >= 5 and bare_pipe_lines / len(lines) >= 0.10:
            issue_docs["flattened_one_cell_per_line_table_doc"] += 1
            issue_chars["flattened_one_cell_per_line_table_doc"] += len(text)
            flags.add("flattened_table")
            keep_example(examples, "flattened_one_cell_per_line_table_doc", doc_id, text[:240].replace("\n", "\\n"))
        content_chars = len(text) - removable_chars
        if content_chars < 200:
            issue_docs["stub_after_boilerplate(<200_chars)"] += 1
            issue_chars["stub_after_boilerplate(<200_chars)"] += len(text)
            flags.add("stub")
            keep_example(examples, "stub_after_boilerplate(<200_chars)", doc_id, text[:240].replace("\n", "\\n"))
        if len(text) > 200_000:
            issue_docs["giant_doc(>200k_chars)"] += 1
            issue_chars["giant_doc(>200k_chars)"] += len(text)
            keep_example(examples, "giant_doc(>200k_chars)", doc_id, f"{len(text)} chars, title={doc['metadata'].get('title')!r}")
        normalized_hash = md5(normalize_for_exact_dup(text))
        exact_groups[normalized_hash].append(index)
        signatures.append(minhash_signature(text))
        records.append({
            "shard": shard, "line": line_number, "id": doc_id, "chars": len(text), "words": len(text.split()),
            "raw_md5": md5(text), "norm_md5": normalized_hash, "lang": label, "ascii": ascii_ratio, "en_stop": en_ratio,
            "book": book_key(doc), "license": doc["metadata"].get("license"), "flags": flags,
            "removable": removable_chars,
        })
        if (index + 1) % 20000 == 0:
            print(f"# raw pass: {index + 1} docs", file=sys.stderr, flush=True)
    return {
        "records": records, "signatures": np.vstack(signatures), "issue_docs": issue_docs, "issue_chars": issue_chars,
        "examples": examples, "exact_groups": exact_groups, "licenses": licenses, "languages": languages,
        "language_chars": language_chars, "id_counts": id_counts, "removable_total": removable_total,
        "total_chars": total_chars,
    }


def near_duplicates(signatures, records, threshold=0.8):
    count = len(records)
    union_find = UnionFind(count)
    candidate_pairs = set()
    for band in range(BANDS):
        block = signatures[:, band * ROWS:(band + 1) * ROWS]
        keys = collections.defaultdict(list)
        for index in range(count):
            keys[block[index].tobytes()].append(index)
        for members in keys.values():
            if 1 < len(members) <= 2000:
                anchor = members[0]
                for other in members[1:]:
                    candidate_pairs.add((anchor, other))
            elif len(members) > 2000:
                for position in range(1, len(members)):
                    candidate_pairs.add((members[position - 1], members[position]))
    confirmed = 0
    cross_book_pairs = 0
    for first, second in candidate_pairs:
        similarity = float(np.mean(signatures[first] == signatures[second]))
        if similarity >= threshold:
            confirmed += 1
            union_find.union(first, second)
            if records[first]["book"] != records[second]["book"]:
                cross_book_pairs += 1
    clusters = collections.defaultdict(list)
    for index in range(count):
        clusters[union_find.find(index)].append(index)
    multi = [members for members in clusters.values() if len(members) > 1]
    docs_in_clusters = sum(len(members) for members in multi)
    removable_docs = docs_in_clusters - len(multi)
    removable_chars = sum(sum(records[i]["chars"] for i in members) - max(records[i]["chars"] for i in members) for members in multi)
    cross_book_clusters = [members for members in multi if len({records[i]["book"] for i in members}) > 1]
    docs_with_cross_book_dup = sum(len(members) for members in cross_book_clusters)
    biggest = sorted(multi, key=len, reverse=True)[:5]
    return {
        "candidate_pairs": len(candidate_pairs), "confirmed_pairs": confirmed, "cross_book_pairs": cross_book_pairs,
        "clusters": len(multi), "docs_in_clusters": docs_in_clusters, "removable_docs": removable_docs,
        "removable_chars": removable_chars, "cross_book_clusters": len(cross_book_clusters),
        "docs_in_cross_book_clusters": docs_with_cross_book_dup,
        "biggest_clusters": [(len(m), len({records[i]["book"] for i in m}), [records[i]["id"] for i in m[:3]]) for m in biggest],
        "cluster_of": {i: union_find.find(i) for m in multi for i in m},
        "cross_book_examples": [[(records[i]["book"], records[i]["id"]) for i in m[:3]] for m in rng.sample(cross_book_clusters, min(6, len(cross_book_clusters)))],
    }


# ---------------------------------------------------------------- pass 2: filtered comparison
MAX_CHAR_DIFF = 20000  # char-level SequenceMatcher is quadratic; larger changed blocks are counted at line level
PLACEHOLDER_RE = re.compile(r"<(IP_ADDRESS|EMAIL_ADDRESS|PHONE_NUMBER)>")


def classify_replaced(original):
    original = original.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", original):
        octets = [int(part) for part in original.split(".")]
        if all(octet <= 255 for octet in octets) and octets[0] > 9:
            return "ipv4_like"
        return "section_number_like(e.g._1.2.3.4)"
    if re.search(r"https?://|www\.|\.(?:com|org|edu|ca|gov|net)\b", original) and "@" not in original:
        return "url_not_email"
    if "@" in original:
        return "email"
    if re.fullmatch(r"[\d\s\-().+]{7,}", original):
        if re.fullmatch(r"(?:97[89][\-\s]?)?\d[\d\-\s]{8,}", original.replace(" ", "")) and len(re.sub(r"\D", "", original)) in (10, 13):
            return "digits_10_or_13(phone_or_isbn)"
        return "phone_like_digits"
    return "other"


def compare_filtered(raw):
    records = raw["records"]
    filtered_by_id = collections.defaultdict(list)
    filtered_count = 0
    placeholder_counts = collections.Counter()
    for doc in iter_filtered():
        filtered_count += 1
        filtered_by_id[doc["id"]].append(md5(doc["text"]))
        placeholder_counts.update(PLACEHOLDER_RE.findall(doc["text"]))
    status = []
    for record in records:
        hashes = filtered_by_id.get(record["id"])
        if hashes is None:
            status.append("dropped")
        elif record["raw_md5"] in hashes:
            status.append("kept_identical")
        else:
            status.append("kept_modified")
    raw_ids = {record["id"] for record in records}
    filtered_not_in_raw = sum(1 for doc_id in filtered_by_id if doc_id not in raw_ids)

    def drop_rate(selector):
        selected = [s for r, s in zip(records, status) if selector(r)]
        if not selected:
            return "n=0"
        dropped = sum(1 for s in selected if s == "dropped")
        return f"n={len(selected)} dropped={dropped} ({dropped / len(selected):.1%})"

    breakdown = {}
    for low, high in [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 100000), (100000, 10**9)]:
        breakdown[f"chars[{low},{high})"] = drop_rate(lambda r, lo=low, hi=high: lo <= r["chars"] < hi)
    for label in sorted(raw["languages"], key=lambda x: -raw["languages"][x])[:8]:
        breakdown[f"lang={label}"] = drop_rate(lambda r, lab=label: r["lang"] == lab)
    for low, high in [(0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 1.01)]:
        breakdown[f"en_stopword_ratio[{low},{high})"] = drop_rate(lambda r, lo=low, hi=high: lo <= r["en_stop"] < hi)
    for flag in ["section_anchor_id", "stub", "table_row", "pipe_only_debris", "nav_chrome", "promo_line",
                 "latex_shortcode", "h5p_mention", "nonpermissive_license_mention", "dangling_embed_reference",
                 "repeated_line_within_doc", "mojibake"]:
        breakdown[f"flag={flag}"] = drop_rate(lambda r, fl=flag: fl in r["flags"])
    for license_name in raw["licenses"]:
        breakdown[f"license={license_name}"] = drop_rate(lambda r, lic=license_name: r["license"] == lic)

    # exact-duplicate handling by the filter
    groups = [members for members in raw["exact_groups"].values() if len(members) > 1]
    copies_kept = collections.Counter()
    for members in groups:
        kept = sum(1 for i in members if status[i] != "dropped")
        copies_kept["0" if kept == 0 else ("1" if kept == 1 else ">1")] += 1

    # diff of modified docs
    modified_indices = [i for i, s in enumerate(status) if s == "kept_modified"]
    modified_ids = {records[i]["id"] for i in modified_indices}
    raw_texts = {}
    for shard, line_number, doc in iter_raw():
        if doc["id"] in modified_ids and doc["id"] not in raw_texts:
            raw_texts[doc["id"]] = doc["text"]
    change_docs = collections.Counter()
    change_chars = collections.Counter()
    replaced_categories = collections.Counter()
    replaced_examples = collections.defaultdict(list)
    deleted_segments = []
    whitespace_only_docs = 0
    for doc in iter_filtered():
        if doc["id"] not in raw_texts:
            continue
        original = raw_texts.pop(doc["id"])
        new = doc["text"]
        if original.split() == new.split():
            whitespace_only_docs += 1
            continue
        doc_categories = set()
        old_lines, new_lines = original.split("\n"), new.split("\n")
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            old_block = "\n".join(old_lines[i1:i2])
            new_block = "\n".join(new_lines[j1:j2])
            if len(old_block) + len(new_block) <= MAX_CHAR_DIFF:
                inner_opcodes = difflib.SequenceMatcher(None, old_block, new_block, autojunk=False).get_opcodes()
            else:
                change_docs["changed_blocks_too_large_for_char_diff"] += 1
                inner_opcodes = [("replace", 0, len(old_block), 0, len(new_block))]
            for inner_tag, a1, a2, b1, b2 in inner_opcodes:
                if inner_tag == "equal":
                    continue
                old_piece, new_piece = old_block[a1:a2], new_block[b1:b2]
                if PLACEHOLDER_RE.search(new_piece):
                    category = classify_replaced(old_piece)
                    for placeholder in PLACEHOLDER_RE.findall(new_piece):
                        replaced_categories[(placeholder, category)] += 1
                        if len(replaced_examples[(placeholder, category)]) < 50:
                            replaced_examples[(placeholder, category)].append(f"[{doc['id']}] {old_piece.strip()[:80]!r} -> {new_piece.strip()[:40]!r}")
                    doc_categories.add("pii_placeholder")
                    change_chars["pii_placeholder_old_chars"] += len(old_piece)
                elif not new_piece.strip():
                    doc_categories.add("deletion")
                    change_chars["deleted_chars"] += len(old_piece)
                    if len(old_piece.strip()) >= 20:
                        deleted_segments.append((doc["id"], old_piece.strip()))
                else:
                    doc_categories.add("other_edit")
                    change_chars["other_edit_old_chars"] += len(old_piece)
        for category in doc_categories:
            change_docs[category] += 1
    # Are deleted segments repeated elsewhere in the raw corpus? Count raw docs containing each sampled segment.
    sampled = rng.sample(deleted_segments, min(40, len(deleted_segments)))
    segment_doc_freq = [0] * len(sampled)
    for shard, line_number, doc in iter_raw():
        text = doc["text"]
        for position, (_, segment) in enumerate(sampled):
            if segment in text:
                segment_doc_freq[position] += 1
    return {
        "filtered_docs": filtered_count, "filtered_unique_ids": len(filtered_by_id), "filtered_not_in_raw": filtered_not_in_raw,
        "status": collections.Counter(status), "status_list": status,
        "kept_chars": sum(r["chars"] for r, s in zip(records, status) if s != "dropped"),
        "dropped_chars": sum(r["chars"] for r, s in zip(records, status) if s == "dropped"),
        "breakdown": breakdown, "exact_dup_groups_copies_kept": copies_kept, "placeholder_counts": placeholder_counts,
        "whitespace_only_modified": whitespace_only_docs, "change_docs": change_docs, "change_chars": change_chars,
        "replaced_categories": replaced_categories,
        "replaced_examples": {f"{k[0]}|{k[1]}": rng.sample(v, min(3, len(v))) for k, v in replaced_examples.items()},
        "deleted_segment_count": len(deleted_segments),
        "deleted_segment_doc_freq": sorted(zip(segment_doc_freq, [s[:100] for _, s in sampled], [d for d, _ in sampled]), reverse=True),
    }


def _group_kept(cluster_of, status_list):
    groups = collections.defaultdict(list)
    for index, root in cluster_of.items():
        if status_list[index] != "dropped":
            groups[root].append(index)
    return groups


def main():
    assert_shards_complete()
    raw = analyze_raw()
    records = raw["records"]
    total_docs = len(records)
    total_chars = raw["total_chars"]
    report = {"total_docs": total_docs, "total_chars": total_chars, "books": len({r["book"] for r in records})}
    report["chars_percentiles"] = percentiles([r["chars"] for r in records])
    report["words_percentiles"] = percentiles([r["words"] for r in records])
    report["total_words"] = sum(r["words"] for r in records)
    report["issues"] = {
        issue: {"docs": raw["issue_docs"][issue], "doc_frac": round(raw["issue_docs"][issue] / total_docs, 4),
                "chars": raw["issue_chars"][issue], "char_frac": round(raw["issue_chars"][issue] / total_chars, 5)}
        for issue in sorted(raw["issue_docs"])
    }
    report["removable_line_union_chars"] = raw["removable_total"]
    report["removable_line_union_frac"] = round(raw["removable_total"] / total_chars, 5)
    report["languages_docs"] = dict(raw["languages"].most_common())
    report["languages_char_frac"] = {k: round(v / total_chars, 5) for k, v in raw["language_chars"].most_common()}
    ascii_values = np.array([r["ascii"] for r in records])
    report["ascii_letter_ratio_lt_0.9_docs"] = int((ascii_values < 0.9).sum())
    report["en_stopword_ratio_percentiles"] = {f"p{p}": round(float(np.percentile([r["en_stop"] for r in records], p)), 3) for p in (1, 5, 25, 50, 75)}
    report["licenses"] = dict(raw["licenses"].most_common())
    duplicate_id_values = [doc_id for doc_id, count in raw["id_counts"].items() if count > 1]
    report["duplicate_id_values"] = len(duplicate_id_values)
    report["docs_with_duplicate_id"] = sum(raw["id_counts"][doc_id] for doc_id in duplicate_id_values)
    groups = [m for m in raw["exact_groups"].values() if len(m) > 1]
    report["exact_dup_groups"] = len(groups)
    report["exact_dup_docs_in_groups"] = sum(len(m) for m in groups)
    report["exact_dup_removable_docs"] = sum(len(m) - 1 for m in groups)
    report["exact_dup_removable_chars"] = sum(sum(records[i]["chars"] for i in m[1:]) for m in groups)
    report["exact_dup_cross_book_groups"] = sum(1 for m in groups if len({records[i]["book"] for i in m}) > 1)
    report["exact_dup_cross_book_removable_docs"] = sum(len(m) - 1 for m in groups if len({records[i]["book"] for i in m}) > 1)
    section_docs = [i for i, r in enumerate(records) if "section_anchor_id" in r["flags"]]
    norm_counts = collections.Counter(r["norm_md5"] for r in records)
    base_hash = {}
    for record in records:
        if "#" not in record["id"]:
            base_hash.setdefault(record["id"], record["norm_md5"])
    report["section_anchor_docs"] = len(section_docs)
    report["section_anchor_docs_exact_dup_of_some_doc"] = sum(1 for i in section_docs if norm_counts[records[i]["norm_md5"]] > 1)
    report["section_anchor_docs_identical_to_base_url_doc"] = sum(1 for i in section_docs if base_hash.get(records[i]["id"].split("#")[0]) == records[i]["norm_md5"])
    report["section_anchor_chars"] = sum(records[i]["chars"] for i in section_docs)
    print("# near-dup", file=sys.stderr, flush=True)
    near = near_duplicates(raw["signatures"], records)
    cluster_of = near.pop("cluster_of")
    report["near_dup"] = near
    print("# filtered comparison", file=sys.stderr, flush=True)
    comparison = compare_filtered(raw)
    status_list = comparison.pop("status_list")
    report["filtered"] = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in comparison.items()}
    report["filtered"]["replaced_categories"] = {f"{k[0]}|{k[1]}": v for k, v in comparison["replaced_categories"].most_common()}
    # residual issues in what the filter kept
    residual = collections.Counter()
    for record, state in zip(records, status_list):
        if state == "dropped":
            continue
        residual["kept_docs"] += 1
        for flag in ("section_anchor_id", "stub", "flattened_table", "table_row", "nav_chrome", "promo_line", "pipe_only_debris", "latex_shortcode",
                     "wp_shortcode", "h5p_mention", "nonpermissive_license_mention", "dangling_embed_reference",
                     "repeated_line_within_doc", "license_attribution_line", "media_attribution_list", "mojibake"):
            if flag in record["flags"]:
                residual[flag] += 1
        if record["lang"] != "en":
            residual["non_en_label"] += 1
        residual["removable_chars"] += record["removable"]
        residual["kept_chars"] += record["chars"]
    kept_groups = collections.Counter()
    for members in groups:
        kept = [i for i in members if status_list[i] != "dropped"]
        if len(kept) > 1:
            kept_groups["groups_with_multiple_kept"] += 1
            kept_groups["redundant_kept_docs"] += len(kept) - 1
    kept_near = collections.defaultdict(int)
    for index, root in cluster_of.items():
        if status_list[index] != "dropped":
            kept_near[root] += 1
    kept_groups["near_dup_clusters_with_multiple_kept"] = sum(1 for count in kept_near.values() if count > 1)
    kept_groups["near_dup_redundant_kept_docs"] = sum(count - 1 for count in kept_near.values() if count > 1)
    kept_groups["near_dup_redundant_kept_chars"] = sum(
        sum(records[i]["chars"] for i in members) - max(records[i]["chars"] for i in members)
        for members in _group_kept(cluster_of, status_list).values() if len(members) > 1)
    report["filtered_residual"] = dict(residual) | dict(kept_groups)
    report["examples"] = pick_examples(raw["examples"])
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- filtered-centric follow-up (run with --filtered-followup)
# Raw ids repeat (2104 id values), so raw-centric status over-counts "kept" copies; this pass counts from the filtered side.
SENSITIVE_LEXICON_RE = re.compile(
    r"(?i)\b(?:kill\w*|murder\w*|blood\w*|dead|die|died|death|suicid\w*|gun\w*|shot|bomb\w*|blow up|rape\w*|sex\w*|"
    r"virgin\w*|naked|nude|breast\w*|penis|vagina\w*|porn\w*|drug\w*|cocaine|heroin|fuck\w*|shit\w*|damn\w*|hell|bitch\w*|"
    r"whore\w*|bastard\w*|nigg\w*|fag\w*|slut\w*|crazy|stupid|idiot\w*|hate\w*|black|white|race|racis\w*|jew\w*|muslim\w*|gay|lesbian)\b"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def filtered_followup():
    filtered_hash = {}
    for doc in iter_filtered():
        filtered_hash.setdefault(doc["id"], set()).add(md5(doc["text"]))
    raw_hashes = collections.defaultdict(set)
    raw_candidates = {}
    for _, _, doc in iter_raw():
        digest = md5(doc["text"])
        raw_hashes[doc["id"]].add(digest)
        if doc["id"] in filtered_hash and digest not in filtered_hash[doc["id"]] and doc["id"] not in raw_candidates:
            raw_candidates[doc["id"]] = doc["text"]
    status = collections.Counter()
    status_chars = collections.Counter()
    norm_groups = collections.Counter()
    signatures = []
    books = []
    deleted_segments = []
    kept_sentences_sample = []
    deleted_chars = 0
    for position, doc in enumerate(iter_filtered()):
        text = doc["text"]
        norm_groups[md5(normalize_for_exact_dup(text))] += 1
        signatures.append(minhash_signature(text))
        books.append(book_key(doc))
        if position % 50 == 0:
            sentences = [sentence for sentence in SENTENCE_SPLIT_RE.split(text) if len(sentence.strip()) >= 20]
            if sentences:
                kept_sentences_sample.append(rng.choice(sentences))
        if md5(text) in raw_hashes.get(doc["id"], ()):
            status["identical_to_a_raw_copy"] += 1
            continue
        original = raw_candidates.get(doc["id"])
        if original is None:
            status["no_raw_text_found"] += 1
            continue
        status["modified"] += 1
        status_chars["modified_doc_chars"] += len(text)
        old_lines, new_lines = original.split("\n"), text.split("\n")
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            old_block, new_block = "\n".join(old_lines[i1:i2]), "\n".join(new_lines[j1:j2])
            if len(old_block) + len(new_block) > MAX_CHAR_DIFF:
                continue
            for inner_tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old_block, new_block, autojunk=False).get_opcodes():
                if inner_tag != "equal" and not new_block[b1:b2].strip() and len(old_block[a1:a2].strip()) >= 20:
                    deleted_segments.append((doc["id"], old_block[a1:a2].strip()))
                    deleted_chars += a2 - a1
    exact_redundant = sum(count - 1 for count in norm_groups.values() if count > 1)
    matrix = np.vstack(signatures)
    pseudo_records = [{"book": book, "chars": 0, "id": ""} for book in books]
    near = near_duplicates(matrix, pseudo_records)
    deleted_hits = sum(1 for _, segment in deleted_segments if SENSITIVE_LEXICON_RE.search(segment))
    kept_hits = sum(1 for sentence in kept_sentences_sample if SENSITIVE_LEXICON_RE.search(sentence))
    result = {
        "filtered_docs": len(signatures), "status": dict(status), "status_chars": dict(status_chars),
        "exact_dup_redundant_docs_within_filtered": exact_redundant,
        "near_dup_within_filtered": {k: near[k] for k in ("clusters", "docs_in_clusters", "removable_docs", "cross_book_clusters", "docs_in_cross_book_clusters")},
        "deleted_segments_ge20chars": len(deleted_segments), "deleted_segment_chars": deleted_chars,
        "deleted_segments_with_sensitive_lexicon": deleted_hits,
        "kept_sentence_sample_size": len(kept_sentences_sample), "kept_sentences_with_sensitive_lexicon": kept_hits,
        "deleted_segment_examples": [f"[{d}] {s[:160]!r}" for d, s in rng.sample(deleted_segments, min(12, len(deleted_segments)))],
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if "--filtered-followup" in sys.argv:
        filtered_followup()
    else:
        main()
