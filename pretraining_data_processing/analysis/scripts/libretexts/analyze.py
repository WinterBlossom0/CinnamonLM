"""Quality analysis of common-pile/libretexts (raw) versus common-pile/libretexts_filtered.

Usage: python3 -I analyze.py RAW_SHARD FILTERED_SHARD OUT_JSON [MAX_DOCS]
       python3 -I analyze.py --selftest
"""
import ast
import collections
import difflib
import gzip
import hashlib
import json
import random
import re
import sys

import numpy as np
import xxhash

SELFTEST = sys.argv[1] == "--selftest"
if not SELFTEST:
    RAW_PATH, FILTERED_PATH, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
    MAX_DOCS = int(sys.argv[4]) if len(sys.argv) > 4 else None
SEED = 0
NUM_PERM = 128
BANDS = 16
ROWS = NUM_PERM // BANDS
NEAR_DUP_THRESHOLD = 0.8
MIN_WORDS_FOR_MINHASH = 30
MAX_EXAMPLES = 4

# ---------------------------------------------------------------- chrome rules
SITE_CHROME_END = re.compile(r"Expand/collapse global location\n.*?Page ID\n\d+\n?", re.S)
INLINE_HEADER = re.compile(
    r"^(?P<title>.{0,300}?) (?:Last updated|Última actualización) (?:Save as PDF|Guardar como PDF)(?: Page ID \d+)?"
)
CHROME_LINES = {
    "-", "- Last updated", "- Save as PDF", "Last updated", "Save as PDF",
    "- Última actualización", "- Guardar como PDF", "Última actualización", "Guardar como PDF",
    "Page ID", "Skip to main content", "selected template will load here",
    "This action is not available.",
}
SAVE_PDF_LINES = {"- Save as PDF", "Save as PDF", "- Guardar como PDF", "Guardar como PDF"}
TOC_MARKER = re.compile(r"(?:(?<=\s)|^)\d{1,3}(?:\.\d{1,3})*[A-Z]?: ")
TOC_LINE = re.compile(r"^(?:- )?\d{1,3}(?:\.\d{1,3})*[A-Z]?: .{1,150}$")
ATTRIBUTION_HEAD = re.compile(r"^(?:Contributors and Attributions|Contributors|Attributions?)$")
ATTRIBUTION_LINE = re.compile(
    r"(?i)^(?:- )?(?:Authored by ?:|Provided by ?:|License ?:|Located at ?:|"
    r"This page titled .{0,300} is shared under|CC licensed content)"
)
FOOTNOTE_RETURN = "↵"

# ---------------------------------------------------------------- artifact rules
ISSUE_PATTERNS = {
    "pageindex_macro": re.compile(r"\\\(\\PageIndex\{[^}]*\}\\\)"),
    "latex_layout_residue": re.compile(r"\\nonumber|\\label\{[^}]*\}|\\\\\[\d+pt\]|\\[hv]space\*?\{[^}]*\}"),
    "spaced_broken_latex": re.compile(
        r"\\ (?:\(|\)|dfrac|frac|begin|end|sqrt|left|right|cdot|text|mathrm|div|times|pi|alpha|beta|theta) ?\{?"
    ),
    "latex_crossref_residue": re.compile(r"(?m)\( ?\[[A-Za-z][\w:.\-]*\] ?\)|^\[[a-z][\w:.\-]*\](?= )"),
    "flattened_chem_subscript": re.compile(
        r"\b(?:H|He|Li|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Fe|Cu|Zn|Br|Mn|Cr|Co|Ni|Pb|Hg|Ag|CH|OH|NH|SO|NO|CO|PO) \d{1,2}"
        r"(?= ?(?:[A-Z]|\(|\)|\]|[+\-−]|$))|\b\d[spdf] \d{1,2}\b"
    ),
    "space_before_punctuation": re.compile(r"(?m)(?<=[A-Za-z]) [,;:](?= )|(?<=[a-z]) \.(?= [A-Z]|$)"),
    "split_ordinal_superscript": re.compile(r"\b\d+ (?:st|nd|rd|th)\b"),
    "inline_footnote_number": re.compile(r"(?<=[a-z][.,”\"]) \d{1,3}(?= [A-Z])"),
    "restrictive_license_mention": re.compile(
        r"(?i)\bCC[ -]BY[ -]NC|\bCC[ -]BY[ -](?:SA[ -])?ND\b|non-?commercial|all rights reserved|used (?:here )?with permission"
    ),
    "truncated_url": re.compile(r"https?://\S*\.\.\.\S*"),
    "interactive_placeholder": re.compile(
        r"(?i)watch (?:this|the following) video|\(Powered by DESMOS\)|Query \\\(\\PageIndex|"
        r"use the (?:simulation|applet|slider)|move the slider|interactive (?:element|figure|simulation)"
    ),
}
FIGURE_CAPTION_LINE = re.compile(r"^Figure (?:\\\(\\PageIndex\{[^}]*\}\\\)|\d+(?:\.\d+)*)\s*[:.]")
EMPTY_TABLE_LINE = re.compile(r"^\|(?:\s*\|)*\s*$")
INLINE_MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.S)
TOKEN = re.compile(r"[^\W\d_]+", re.U)

EN_STOP = set("""the of and to in a is that for it as was with be by on not he i this are or his from at which but
have an they you were their one all we can her has there been if more when will would who so no she other its may
these what them than some him time into only do could new about two our then most also any such should""".split())
ES_STOP = set("""el la de que y en los se del las un por con no una su para es al lo como más pero sus le ya o este
sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les
ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho""".split())
FR_STOP = set("""le la les de des du et en un une est que qui dans pour pas sur au aux avec ce ces il elle ils nous
vous je tu on son sa ses leur mais ou donc ni car par plus être avoir fait comme tout bien très""".split())


def classify_language(text):
    tokens = [t.lower() for t in TOKEN.findall(text)]
    if len(tokens) < 20:
        return "too_short", 0.0, 0.0
    en = sum(t in EN_STOP for t in tokens) / len(tokens)
    es = sum(t in ES_STOP for t in tokens) / len(tokens)
    fr = sum(t in FR_STOP for t in tokens) / len(tokens)
    letters = [c for c in text if c.isalpha()]
    ascii_ratio = sum(c.isascii() for c in letters) / max(1, len(letters))
    if en >= max(es, fr) and en >= 0.15:
        return "en", en, ascii_ratio
    if es > en and es >= 0.15:
        return "es", en, ascii_ratio
    if fr > en and fr >= 0.15:
        return "fr", en, ascii_ratio
    if ascii_ratio < 0.5:
        return "non_latin", en, ascii_ratio
    return "unclear_low_stopword", en, ascii_ratio


def strip_chrome(text):
    """Return (clean_text, removed_chars_by_category)."""
    removed = collections.Counter()
    body = text
    if body.startswith("Skip to main content"):
        match = SITE_CHROME_END.search(body)
        if match:
            removed["site_chrome_block"] += match.end()
            body = body[match.end():]
            lines = body.split("\n")
            byline = 0
            while byline < min(3, len(lines)) and 0 < len(lines[byline]) < 80 and not lines[byline].endswith((".", "?", "!", ":")):
                removed["byline"] += len(lines[byline]) + 1
                byline += 1
            body = "\n".join(lines[byline:])
    if "\n" not in body:
        match = INLINE_HEADER.match(body)
        if match:
            removed["inline_header"] += match.end() - len(match.group("title"))
            rest = body[match.end():]
            markers = [m.start() for m in TOC_MARKER.finditer(rest)]
            if markers:
                removed["byline"] += markers[0]
                toc_chars = 0
                for index, start in enumerate(markers):
                    stop = markers[index + 1] if index + 1 < len(markers) else len(rest)
                    if stop - start <= 150:
                        toc_chars += stop - start
                removed["toc_entries"] += toc_chars
                kept_rest = []
                for index, start in enumerate(markers):
                    stop = markers[index + 1] if index + 1 < len(markers) else len(rest)
                    if stop - start > 150:
                        kept_rest.append(rest[start:stop])
                body = match.group("title") + "\n" + "".join(kept_rest)
            else:
                removed["byline"] += len(rest)
                body = match.group("title")
        return body, removed

    lines = body.split("\n")
    kept = []
    index = 0
    in_attribution = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if in_attribution:
            removed["attribution"] += len(line) + 1
            index += 1
            continue
        if stripped == "-":
            removed["empty_bullet"] += len(line) + 1
            index += 1
            continue
        if stripped in CHROME_LINES or (re.fullmatch(r"\d+", stripped) and index > 0 and lines[index - 1].strip() == "Page ID"):
            removed["header_lines"] += len(line) + 1
            if stripped in SAVE_PDF_LINES:
                look = index + 1
                taken = 0
                while look < len(lines) and taken < 3 and lines[look].startswith("- ") and len(lines[look]) < 100 \
                        and not lines[look].rstrip().endswith((".", "?", "!", ":")):
                    removed["byline"] += len(lines[look]) + 1
                    look += 1
                    taken += 1
                index = look
                continue
            index += 1
            continue
        if ATTRIBUTION_HEAD.match(stripped) and len(text) - sum(len(x) + 1 for x in lines[:index]) < 3000:
            in_attribution = True
            removed["attribution"] += len(line) + 1
            index += 1
            continue
        if ATTRIBUTION_LINE.match(stripped):
            removed["attribution"] += len(line) + 1
            index += 1
            continue
        if stripped.endswith(FOOTNOTE_RETURN):
            removed["footnote_return_lines"] += len(line) + 1
            index += 1
            continue
        kept.append(line)
        index += 1

    # runs of >=3 table-of-contents lines
    final = []
    run = []
    for line in kept:
        if TOC_LINE.match(line.strip()):
            run.append(line)
            continue
        if len(run) >= 3:
            removed["toc_entries"] += sum(len(x) + 1 for x in run)
        else:
            final.extend(run)
        run = []
        final.append(line)
    if len(run) >= 3:
        removed["toc_entries"] += sum(len(x) + 1 for x in run)
    else:
        final.extend(run)
    return "\n".join(final), removed


def broken_fragment_chars(text):
    """Chars in short line fragments split from the surrounding sentence by inline HTML elements."""
    lines = text.split("\n")
    total = 0
    count = 0
    for index in range(1, len(lines) - 1):
        line = lines[index].strip()
        nxt = lines[index + 1].lstrip()
        prev = lines[index - 1].rstrip()
        if not line or line.startswith(("- ", "|", "\\")) or len(line.split()) > 4:
            continue
        if prev.endswith((".", "?", "!", ":", "|")) or len(prev) < 30:
            continue
        if (nxt[:1].islower() or nxt[:1] in ",.;:)") and len(nxt.split()) >= 3:
            total += len(lines[index]) + 1
            count += 1
    return count, total


def percentiles(values, points=(1, 5, 25, 50, 75, 95, 99)):
    array = np.sort(np.asarray(values))
    out = {f"p{p}": int(np.percentile(array, p, method="lower")) for p in points}
    out["max"] = int(array[-1])
    out["min"] = int(array[0])
    out["mean"] = float(array.mean())
    return out


rng = np.random.default_rng(SEED)
PERM_A = (rng.integers(1, 2**63 - 1, NUM_PERM, dtype=np.uint64) | np.uint64(1))
PERM_B = rng.integers(0, 2**63 - 1, NUM_PERM, dtype=np.uint64)


def minhash(text):
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < MIN_WORDS_FOR_MINHASH:
        return None
    shingles = {xxhash.xxh64_intdigest(" ".join(tokens[i:i + 5])) for i in range(len(tokens) - 4)}
    hashes = np.fromiter(shingles, dtype=np.uint64)
    signature = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    # multiply-shift universal hashing; uint64 overflow wraps by design
    with np.errstate(over="ignore"):
        for start in range(0, len(hashes), 20000):
            chunk = hashes[start:start + 20000]
            values = (PERM_A[:, None] * chunk[None, :] + PERM_B[:, None]) >> np.uint64(32)
            signature = np.minimum(signature, values.min(axis=1))
    return signature.astype(np.uint32)


def normalize_for_hash(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def add_example(store, key, doc_id, snippet):
    if len(store[key]) < MAX_EXAMPLES:
        store[key].append(f"[{doc_id}] {snippet[:280]}")


def self_test():
    # Snippets copied verbatim from real raw documents.
    toc_page = ("11: Employee Termination Last updated Save as PDF Page ID 46946 Nina Burokas Lumen Learning "
                "13.1: Navigating Employee Termination 13.2: Putting It Together- Employee Termination 13.3: Discussion- Employee Termination")
    clean, removed = strip_chrome(toc_page)
    assert clean.strip() == "11: Employee Termination", clean
    assert removed["toc_entries"] > 0 and removed["inline_header"] > 0 and removed["byline"] > 0
    header_doc = ("1.E: Fundamentals of Microbiology (Exercises)\n-\n- Last updated\n- Save as PDF\n- Ying Liu\n"
                  "- City College of San Francisco\nThese are homework exercises to accompany Kaiser's \"Microbiology\" TextMap.")
    clean, removed = strip_chrome(header_doc)
    assert clean == "1.E: Fundamentals of Microbiology (Exercises)\nThese are homework exercises to accompany Kaiser's \"Microbiology\" TextMap.", clean
    assert removed["byline"] == len("- Ying Liu\n- City College of San Francisco\n")
    site = ("Skip to main content\nTable of Contents\nExpand/collapse global location\n6.3: Inequalities\nLast updated\n"
            "Save as PDF\nPage ID\n35014\nIllustrative Mathematics\nOpenUp Resources")
    clean, removed = strip_chrome(site)
    assert clean == "" and removed["byline"] > 0 and removed["site_chrome_block"] > 0, (clean, removed)
    assert ISSUE_PATTERNS["space_before_punctuation"].search("between nominal , ordinal , interval")
    assert ISSUE_PATTERNS["split_ordinal_superscript"].search("Find the 40 th percentile.")
    assert ISSUE_PATTERNS["inline_footnote_number"].search("his acceptance speech. 1 Yet, Romney")
    assert ISSUE_PATTERNS["flattened_chem_subscript"].search("the formula H 2 O. |")
    assert ISSUE_PATTERNS["spaced_broken_latex"].search("Example \\ (\\PageIndex{2}\\): Calculating")
    assert classify_language("Las declaraciones de factores faltantes se pueden usar para responder preguntas como de que numero es la "
                             "fraccion y en el caso de los numeros que se utilicen puede ser verdadero o falso para la")[0] == "es"
    print("selftest ok")


if SELFTEST:
    self_test()
    sys.exit(0)


# ---------------------------------------------------------------- phase A: filtered index
filtered_info = {}
filtered_marker_docs = collections.Counter()
filtered_pii = collections.Counter()
filtered_pii_attached = collections.Counter()
filtered_len = []
FILTERED_MARKERS = {
    "last_updated_save_pdf": re.compile(r"Last updated\W+(?:- )?Save as PDF"),
    "mindtouch_blob": re.compile(r"MindTouch\.Deki"),
    "skip_to_main": re.compile(r"Skip to main content"),
    "selected_template": re.compile(r"selected template will load here"),
    "lone_dash_line": re.compile(r"(?m)^-$"),
    "pageindex_macro": ISSUE_PATTERNS["pageindex_macro"],
    "toc_inline_page": re.compile(r"^.{0,300}? Last updated Save as PDF Page ID \d+"),
}
for line in gzip.open(FILTERED_PATH, "rt"):
    doc = json.loads(line)
    text = doc["text"]
    filtered_info[doc["id"]] = (hashlib.md5(text.encode()).hexdigest(), len(text))
    filtered_len.append(len(text))
    for name, pattern in FILTERED_MARKERS.items():
        if pattern.search(text):
            filtered_marker_docs[name] += 1
    for token in re.findall(r"<(EMAIL_ADDRESS|IP_ADDRESS|PHONE_NUMBER|[A-Z_]{4,30})>", text):
        filtered_pii[token] += 1
    for token in re.findall(r"\S<(EMAIL_ADDRESS|IP_ADDRESS|PHONE_NUMBER)>", text):
        filtered_pii_attached[token] += 1

# ---------------------------------------------------------------- phase B: raw pass
doc_count = 0
total_chars = 0
char_lengths, word_lengths = [], []
clean_lengths = []
hosts = collections.Counter()
licenses = collections.Counter()
section_type = collections.Counter()
category_chars = collections.Counter()
category_docs = collections.Counter()
issue_docs = collections.Counter()
issue_chars = collections.Counter()
examples = collections.defaultdict(list)
language = collections.Counter()
language_chars = collections.Counter()
language_by_host = collections.defaultdict(collections.Counter)
math_chars = 0
math_docs = 0
figure_caption = [0, 0]
empty_table = [0, 0]
fragment_docs = 0
fragment_chars = 0
fragment_lines = 0
no_content_docs = 0
no_content_chars = 0
chrome_fraction = []
new_page_title_docs = 0
raw_hash_groups = collections.defaultdict(list)
clean_hash_groups = collections.defaultdict(list)
signatures = []
signature_index = []
doc_meta = []  # (id, raw_len, clean_len, host, lang, dropped, modified)
modified_ids = set()
dropped_reasons = collections.Counter()

CHROME_CATEGORIES = ("site_chrome_block", "inline_header", "header_lines", "byline", "toc_entries", "empty_bullet")
SOFT_CATEGORIES = ("attribution", "footnote_return_lines")

for line in gzip.open(RAW_PATH, "rt"):
    if MAX_DOCS is not None and doc_count >= MAX_DOCS:
        break
    doc = json.loads(line)
    doc_id = doc["id"]
    text = doc["text"]
    metadata = doc["metadata"] if isinstance(doc["metadata"], dict) else ast.literal_eval(doc["metadata"])
    doc_count += 1
    total_chars += len(text)
    char_lengths.append(len(text))
    word_lengths.append(len(text.split()))
    host = doc_id.split("/")[2]
    hosts[host] += 1
    licenses[(metadata.get("license") or "<missing>")] += 1
    section_type["Courses" if "/Courses/" in doc_id else "Bookshelves" if "/Bookshelves/" in doc_id else "other"] += 1
    if re.search(r"\bNew Page\b", (metadata.get("title") or "")):
        new_page_title_docs += 1
        add_example(examples, "new_page_title", doc_id, text)

    clean, removed = strip_chrome(text)
    for category, chars in removed.items():
        if chars > 0:
            category_chars[category] += chars
            category_docs[category] += 1
            add_example(examples, category, doc_id, text[:280] if category != "attribution" else text[-280:])
    chrome = sum(removed[c] for c in CHROME_CATEGORIES)
    chrome_fraction.append(chrome / len(text))
    clean_core = clean.strip()
    clean_lengths.append(len(clean_core))
    title = (metadata.get("title") or "")
    content_without_title = clean_core[len(title):].strip() if clean_core.startswith(title) else clean_core
    if len(content_without_title) < 200:
        no_content_docs += 1
        no_content_chars += len(text)
        add_example(examples, "no_content_after_chrome_removal", doc_id, text[:280])

    for name, pattern in ISSUE_PATTERNS.items():
        hits = [m.group(0) for m in pattern.finditer(text)]
        if hits:
            issue_docs[name] += 1
            issue_chars[name] += sum(len(h) for h in hits)
            m = pattern.search(text)
            add_example(examples, name, doc_id, text[max(0, m.start() - 80):m.end() + 120])

    doc_math = sum(len(m.group(0)) for m in INLINE_MATH.finditer(text))
    if doc_math:
        math_docs += 1
        math_chars += doc_math
    lines = text.split("\n")
    captions = [x for x in lines if FIGURE_CAPTION_LINE.match(x)]
    if captions:
        figure_caption[0] += 1
        figure_caption[1] += sum(len(x) + 1 for x in captions)
        add_example(examples, "figure_caption_without_image", doc_id, captions[0])
    empties = [x for x in lines if EMPTY_TABLE_LINE.match(x)]
    if empties:
        empty_table[0] += 1
        empty_table[1] += sum(len(x) + 1 for x in empties)
    count, chars = broken_fragment_chars(clean)
    if count:
        fragment_docs += 1
        fragment_chars += chars
        fragment_lines += count
        if count >= 5:
            add_example(examples, "broken_inline_fragments", doc_id, re.sub(r"\n", " ⏎ ", clean[:280]))

    lang_text = INLINE_MATH.sub(" ", clean)
    lang, en_ratio, ascii_ratio = classify_language(lang_text)
    language[lang] += 1
    language_chars[lang] += len(text)
    language_by_host[host][lang] += 1
    if lang in ("es", "fr", "non_latin"):
        add_example(examples, f"language_{lang}", doc_id, clean_core[:200])

    raw_hash_groups[hashlib.md5(normalize_for_hash(text).encode()).hexdigest()].append(doc_count - 1)
    if len(clean_core) >= 200:
        clean_hash_groups[hashlib.md5(normalize_for_hash(clean_core).encode()).hexdigest()].append(doc_count - 1)
    signature = minhash(clean_core)
    if signature is not None:
        signatures.append(signature)
        signature_index.append(doc_count - 1)

    info = filtered_info.get(doc_id)
    dropped = info is None
    modified = (not dropped) and info[0] != hashlib.md5(text.encode()).hexdigest()
    if modified:
        modified_ids.add(doc_id)
    doc_meta.append((doc_id, len(text), len(content_without_title), host, lang, dropped, modified,
                     bool(removed.get("site_chrome_block"))))

# ---------------------------------------------------------------- duplicates
raw_dup_extra = sum(len(g) - 1 for g in raw_hash_groups.values() if len(g) > 1)
raw_dup_docs = sum(len(g) for g in raw_hash_groups.values() if len(g) > 1)
clean_dup_extra = sum(len(g) - 1 for g in clean_hash_groups.values() if len(g) > 1)
clean_dup_docs = sum(len(g) for g in clean_hash_groups.values() if len(g) > 1)
clean_dup_extra_chars = sum(sum(doc_meta[i][1] for i in g[1:]) for g in clean_hash_groups.values() if len(g) > 1)

sig_matrix = np.vstack(signatures)
parent = list(range(len(signature_index)))


def find(node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


candidate_pairs = set()
for band in range(BANDS):
    buckets = collections.defaultdict(list)
    block = sig_matrix[:, band * ROWS:(band + 1) * ROWS]
    for row, key in enumerate(map(bytes, block)):
        buckets[key].append(row)
    for members in buckets.values():
        if 1 < len(members) <= 2000:
            anchor = members[0]
            for other in members[1:]:
                candidate_pairs.add((anchor, other))
        elif len(members) > 2000:
            for a, b in zip(members, members[1:]):
                candidate_pairs.add((a, b))
verified_pairs = 0
cross_section_pairs = 0
for a, b in candidate_pairs:
    similarity = float(np.mean(sig_matrix[a] == sig_matrix[b]))
    if similarity >= NEAR_DUP_THRESHOLD:
        verified_pairs += 1
        id_a, id_b = doc_meta[signature_index[a]][0], doc_meta[signature_index[b]][0]
        if ("/Courses/" in id_a) != ("/Courses/" in id_b):
            cross_section_pairs += 1
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b
clusters = collections.defaultdict(list)
for row in range(len(signature_index)):
    clusters[find(row)].append(signature_index[row])
dup_clusters = [c for c in clusters.values() if len(c) > 1]
near_dup_docs = sum(len(c) for c in dup_clusters)
near_dup_extra = sum(len(c) - 1 for c in dup_clusters)
near_dup_extra_chars = sum(sum(sorted((doc_meta[i][1] for i in c), reverse=True)[1:]) for c in dup_clusters)
in_cluster = set(i for c in dup_clusters for i in c)
largest = sorted(dup_clusters, key=len, reverse=True)[:5]
largest_examples = [[doc_meta[i][0] for i in c[:4]] + [f"(cluster size {len(c)})"] for c in largest]
cluster_example = []
for c in sorted(dup_clusters, key=len, reverse=True)[:40:8]:
    cluster_example.append([doc_meta[i][0] for i in c[:3]])

# ---------------------------------------------------------------- filtered comparison
kept_ids = set(filtered_info)
raw_ids = set(m[0] for m in doc_meta)
drop_stats = collections.Counter()
drop_buckets = collections.Counter()
for index, (doc_id, raw_len, content_len, host, lang, dropped, modified, site_chrome) in enumerate(doc_meta):
    group = "dropped" if dropped else "kept"
    drop_stats[f"{group}_total"] += 1
    if raw_len < 500:
        reason = "raw_len<500"
    elif host == "espanol.libretexts.org" or lang in ("es", "fr", "non_latin"):
        reason = "non_english_heuristic"
    elif site_chrome:
        reason = "site_chrome_doc"
    elif content_len < 200:
        reason = "no_content_after_chrome"
    elif index in in_cluster:
        reason = "in_near_dup_cluster"
    else:
        reason = "other"
    drop_buckets[(group, reason)] += 1

raw_kept_min_len = min(m[1] for m in doc_meta if not m[5])
raw_lt700_total = sum(1 for m in doc_meta if m[1] < 700)
raw_lt700_kept = sum(1 for m in doc_meta if m[1] < 700 and not m[5])
cluster_filter = collections.Counter()
for cluster in dup_clusters:
    kept_count = sum(1 for i in cluster if not doc_meta[i][5])
    cluster_filter["all_dropped" if kept_count == 0 else "one_kept" if kept_count == 1 else "multi_kept"] += 1
    cluster_filter["residual_extra_kept_docs"] += max(kept_count - 1, 0)
    cluster_filter["residual_extra_kept_chars"] += sum(sorted((doc_meta[i][1] for i in cluster if not doc_meta[i][5]), reverse=True)[1:])

# diff of modified docs
filtered_texts = {}
for line in gzip.open(FILTERED_PATH, "rt"):
    doc = json.loads(line)
    if doc["id"] in modified_ids:
        filtered_texts[doc["id"]] = doc["text"]
mod_stats = collections.Counter()


def classify_pii_replacement(token, source, before, after, window):
    """Label one redaction as an evident false positive or as plausibly real, from the raw context."""
    if token == "EMAIL_ADDRESS":
        if "://" in source or re.search(r"https?://\S*$", before) or re.search(r"\.@\dx\.|\.(?:jpe?g|png|gif|svg)\b", source + after):
            return "false_positive_url_or_filename"
        return "plausible_email" if REAL_EMAIL.fullmatch(source) else "false_positive_other"
    if token == "IP_ADDRESS":
        octets = re.findall(r"\d+", source)
        if any(int(o) > 255 for o in octets) or re.search(r"[\d./]$", before) or re.match(r"\.?\d", after) \
                or re.search(r"(?i)doi|section", before[-40:]):
            return "false_positive_dotted_number"
        if all(int(o) < 10 for o in octets) or before.endswith("-"):
            return "false_positive_section_number"
        return "plausible_ip"
    if re.search(r"[\d_.\-\\]$", before) or re.match(r"[\d_]", after) or len(re.findall(r"\d+", window)) >= 6:
        return "false_positive_numeric_context"
    return "plausible_phone"


PII_TOKEN = re.compile(r"<(EMAIL_ADDRESS|IP_ADDRESS|PHONE_NUMBER)>")
REAL_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
REAL_IP = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
pii_kinds = collections.Counter()
pii_examples = collections.defaultdict(list)
mod_examples = []
random.seed(SEED)
for line in gzip.open(RAW_PATH, "rt"):
    doc = json.loads(line)
    if doc["id"] not in filtered_texts:
        continue
    raw_text = doc["text"]
    new_text = filtered_texts[doc["id"]]
    mod_stats["docs"] += 1
    mod_stats["raw_chars"] += len(raw_text)
    mod_stats["filtered_chars"] += len(new_text)
    if len(new_text) == 0:
        mod_stats["emptied"] += 1
    has_pii = bool(re.search(r"<(EMAIL_ADDRESS|IP_ADDRESS|PHONE_NUMBER)>", new_text))
    raw_lines = raw_text.split("\n")
    new_lines = new_text.split("\n")
    matcher = difflib.SequenceMatcher(None, raw_lines, new_lines, autojunk=False)
    deleted_spans = 0
    text_deletion = False
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            continue
        old = "\n".join(raw_lines[a1:a2])
        new = "\n".join(new_lines[b1:b2])
        if PII_TOKEN.search(new):
            inner = difflib.SequenceMatcher(None, old, new, autojunk=False)
            for op2, c1, c2, d1, d2 in inner.get_opcodes():
                if op2 == "equal" or not PII_TOKEN.search(new[max(0, d1 - 2):d2 + 2]):
                    continue
                source = old[c1:c2].strip()
                token = PII_TOKEN.search(new[max(0, d1 - 2):d2 + 2]).group(1)
                kind = classify_pii_replacement(token, source, old[max(0, c1 - 60):c1], old[c2:c2 + 3], old[max(0, c1 - 40):c2 + 40])
                pii_kinds[(token, kind)] += 1
                if len(pii_examples[(token, kind)]) < 3:
                    pii_examples[(token, kind)].append(f"[{doc['id']}] raw='{old[max(0, c1 - 40):c2 + 20]}' -> '{new[max(0, d1 - 40):d2 + 20]}'")
        if not PII_TOKEN.search(new) and len(new) < len(old):
            mod_stats["nonpii_deleted_chars"] += len(old) - len(new)
            text_deletion = True
            if len(mod_examples) < 10 and random.random() < 0.004:
                mod_examples.append(f"[{doc['id']}] RAW: {old[:260]} || FILTERED: {new[:260]}")
        mod_stats["changed_line_raw_chars"] += len(old)
        mod_stats["changed_line_new_chars"] += len(new)
        deleted_spans += 1
    mod_stats["docs_with_nonpii_deletion"] += int(text_deletion)
    mod_stats["with_pii_token"] += int(has_pii)
    mod_stats["only_pii"] += int(has_pii and re.sub(r"<(?:EMAIL_ADDRESS|IP_ADDRESS|PHONE_NUMBER)>", "", new_text) and len(new_text) >= len(raw_text) - 200)
    mod_stats["shorter"] += int(len(new_text) < len(raw_text))

filtered_empty = sum(1 for x in filtered_len if x == 0)
filtered_short = sum(1 for x in filtered_len if x < 200)

# ---------------------------------------------------------------- report
total_removed_chrome = sum(category_chars[c] for c in CHROME_CATEGORIES)
total_removed_soft = sum(category_chars[c] for c in SOFT_CATEGORIES)
chrome_fraction_arr = np.asarray(chrome_fraction)
report = {
    "doc_count": doc_count,
    "total_chars": total_chars,
    "char_length": percentiles(char_lengths),
    "word_length": percentiles(word_lengths),
    "clean_char_length": percentiles(clean_lengths),
    "hosts": hosts.most_common(),
    "section_type": section_type,
    "licenses": licenses.most_common(),
    "new_page_title_docs": new_page_title_docs,
    "chrome_categories": {c: {"docs": category_docs[c], "doc_frac": category_docs[c] / doc_count,
                              "chars": category_chars[c], "char_frac": category_chars[c] / total_chars}
                          for c in CHROME_CATEGORIES + SOFT_CATEGORIES},
    "chrome_total_chars": total_removed_chrome,
    "chrome_total_char_frac": total_removed_chrome / total_chars,
    "soft_boilerplate_chars": total_removed_soft,
    "soft_boilerplate_char_frac": total_removed_soft / total_chars,
    "docs_with_any_chrome": int((chrome_fraction_arr > 0).sum()),
    "docs_chrome_over_50pct": int((chrome_fraction_arr > 0.5).sum()),
    "docs_chrome_over_90pct": int((chrome_fraction_arr > 0.9).sum()),
    "no_content_docs": no_content_docs,
    "no_content_raw_chars": no_content_chars,
    "issues": {k: {"docs": issue_docs[k], "doc_frac": issue_docs[k] / doc_count, "chars": issue_chars[k],
                   "char_frac": issue_chars[k] / total_chars} for k in ISSUE_PATTERNS},
    "latex_math_spans": {"docs": math_docs, "doc_frac": math_docs / doc_count, "chars": math_chars,
                         "char_frac": math_chars / total_chars},
    "figure_caption_lines": {"docs": figure_caption[0], "chars": figure_caption[1], "char_frac": figure_caption[1] / total_chars},
    "empty_table_lines": {"docs": empty_table[0], "chars": empty_table[1]},
    "broken_inline_fragments": {"docs": fragment_docs, "doc_frac": fragment_docs / doc_count, "lines": fragment_lines,
                                "chars": fragment_chars, "char_frac": fragment_chars / total_chars},
    "language_docs": language.most_common(),
    "language_chars": language_chars.most_common(),
    "language_by_host_nonen": {h: dict(c) for h, c in language_by_host.items() if sum(v for k, v in c.items() if k != "en") > 0},
    "exact_dup_raw": {"docs_in_groups": raw_dup_docs, "extra_copies": raw_dup_extra},
    "exact_dup_clean_ge200": {"docs_in_groups": clean_dup_docs, "extra_copies": clean_dup_extra, "extra_raw_chars": clean_dup_extra_chars},
    "near_dup": {"docs_hashed": len(signature_index), "candidate_pairs": len(candidate_pairs), "verified_pairs": verified_pairs,
                 "cross_courses_bookshelves_pairs": cross_section_pairs, "clusters": len(dup_clusters),
                 "docs_in_clusters": near_dup_docs, "extra_copies": near_dup_extra, "extra_raw_chars": near_dup_extra_chars,
                 "largest": largest_examples, "examples": cluster_example},
    "filtered": {"docs": len(filtered_info), "chars": sum(filtered_len), "kept_ids_not_in_raw": len(kept_ids - raw_ids),
                 "empty_docs": filtered_empty, "docs_lt200": filtered_short, "marker_docs": dict(filtered_marker_docs),
                 "pii_tokens": dict(filtered_pii), "pii_tokens_attached_to_nonspace": dict(filtered_pii_attached),
                 "raw_kept_min_len": raw_kept_min_len, "raw_lt700_total": raw_lt700_total, "raw_lt700_kept": raw_lt700_kept,
                 "near_dup_clusters_vs_filter": dict(cluster_filter), "drop_buckets": {f"{g}|{r}": v for (g, r), v in sorted(drop_buckets.items())},
                 "drop_stats": dict(drop_stats), "modified": dict(mod_stats), "modified_examples": mod_examples,
                 "pii_replacement_kinds": {f"{t}|{k}": v for (t, k), v in pii_kinds.items()},
                 "pii_replacement_examples": {f"{t}|{k}": v for (t, k), v in pii_examples.items()}},
    "examples": dict(examples),
}
with open(OUT_PATH, "w") as handle:
    json.dump(report, handle, indent=1, ensure_ascii=False, default=dict)
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1, ensure_ascii=False, default=dict))
