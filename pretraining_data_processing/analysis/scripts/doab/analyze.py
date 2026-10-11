"""Quality analysis of common-pile/doab (raw) with a comparison against common-pile/doab_filtered.

Run with the data_pipeline interpreter (fasttext, numpy, xxhash):
  python -I analyze.py filtered-index WORK_DIR FILTERED_SHARD...   # id -> (length, hash) of the filtered release, plus a text sample
  python -I analyze.py raw WORK_DIR RAW_SHARD...                   # per-document metrics, sketches, paragraph hashes, raw-vs-filtered diffs
  python -I analyze.py report WORK_DIR REPORT_DIR                  # aggregation into REPORT_DIR/report.json and examples.json
  python -I analyze.py sample WORK_DIR REPORT_DIR                  # language-band and issue samples for reading
  python -I analyze.py extra WORK_DIR RAW_SHARD...                 # empty reference sections, dropped tables, compilation notices, glued words
  python -I analyze.py extra-report WORK_DIR REPORT_DIR            # aggregation into REPORT_DIR/extra_report.json
"""
import collections
import difflib
import gzip
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import re
import sys

import fasttext
import numpy as np
import xxhash

PROCESSES = 2
LID_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data_pipeline", "models", "lid.176.bin")
WORDLIST = "/usr/share/dict/american-english"
SKETCH_K = 128
SHINGLE = 5
NEARDUP_MIN_WORDS = 50
POSTING_CAP = 200
PARA_HASH_MIN_CHARS = 200
LID_CHUNK_MIN_CHARS = 300
EXAMPLES_PER_ISSUE = 10
# Boilerplate patterns are searched only in the first 400 chars of a line: full-line search on long paragraphs dominated the run time.
BOILERPLATE_SCAN_CHARS = 400
# About 1 % of filtered docs keep their text for sentence-level diffs; long docs are skipped to bound memory and difflib time.
FILTERED_TEXT_SAMPLE_MOD = 100
FILTERED_TEXT_MAX_CHARS = 200_000

# ---------------------------------------------------------------- line classes
BOILERPLATE = re.compile("|".join([
    r"Additional information is available at the end of the chapter",
    r"Licensee (?:IntechOpen|InTech|MDPI)",
    r"distributed under the terms (?:and conditions )?of the Creative Commons",
    r"This (?:chapter|article|book|work|publication|title) is (?:an open access|distributed|licensed|published under|available)",
    r"licen[sc]ed under (?:the terms of )?(?:a |the )?Creative Commons",
    r"Address all correspondence to",
    r"Downloaded from https?://",
    r"We are IntechOpen|Interested in publishing with us|Numbers displayed above are based on latest data",
    r"Open Access books Built by scientists|Selection of our books indexed in the Book Citation Index",
    r"https?://dx\.doi\.org/10\.5772/|^\W*https?://(?:dx\.)?doi\.org/10\.\d{4,}/\S+\W*$",
    r"Frontiers Copyright Statement|^#* ?About Frontiers|Frontiers Journal Series|Dedication to Quality|What are Frontiers Research Topics",
    r"All rights reserved",
    r"Deutsche Nationalbibliothek|Deutschen Nationalbibliothek|CIP catalogue record|Library of Congress Cataloging",
    r"printed on acid-free paper|Gedruckt auf",
    r"Publisher.s Note: MDPI stays neutral|Disclaimer/Publisher.s Note",
    r"The authors? declares? (?:no|that there is no|that the research was conducted in the absence)",
    r"^\W*Citation: |^\W*Correspondence: |Received: .{0,40}(?:Accepted|Revised): ",
    r"(?:©|\(c\)|Copyright ©?) ?(?:19|20)\d\d",
    r"Editorial Office E-mail|MDPI St\. Alban-Anlage|www\.mdpi\.com",
    r"^Reprinted from: |This article was submitted to .{0,80}a section of the journal",
    r"\bISBN(?:-1[03])?:? ?(?:97[89])?[\d\- ]{9,}|\be?-?ISSN\b",
    r"^\W*(?:Cover (?:design|image|photo|illustration|art)|Typeset(?:ting)?(?: by)?|Proofreading|Jacket illustration)\b",
    r"\bDOI:? ?(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,}/\S+\s*$",
]), re.I)
ADMIN_STATEMENT = re.compile(r"^(?:Author Contributions?|Funding|Institutional Review Board Statement|Informed Consent Statement|Data Availability Statement|Conflicts? of Interest(?: Statement)?|Acknowledge?ments?|Supplementary Materials?|Specialty section|Edited by|Reviewed by|Academic Editors?|Keywords|Ethics Statement|Abbreviations)\s*:", re.I)
MARKUP_CHARS = re.compile(r"[*_#>\\]")
PAGE_NUMBER = re.compile(r"^[*_#\s]*(?:\d{1,4}|[ivxlc]{1,6})[*_\s]*$", re.I)
NUM_EDGE = re.compile(r"^[#*_\s]*\d{1,4}[*_]*\s+\S|\S\s+[*_]*\d{1,4}[*_\s]*$")
REF_HEAD = re.compile(r"^[#*\s]*(?:\d+\.?\s*)?(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY|Literature Cited|Works Cited|Reference List|Literatur|Literaturverzeichnis|Bibliografia|Bibliographie|Références|Referencias|Referências)\s*[*:\s]*$")
INDEX_HEAD = re.compile(r"^[#*\s]*(?:(?:Subject|Name|Author|General) )?Index(?: of [A-Za-z ]+)?[*\s]*$", re.I)
HEADING = re.compile(r"^(?:#{1,6}\s|\*\*[^*]{1,200}\*\*\s*$|\*[^*]{1,200}\*\s*$)")
REF_LINE = [
    re.compile(r"^\[\d{1,4}\]\s*\S"),
    re.compile(r"^\d{1,4}\.\s+[A-Z][A-Za-z'\-]+,?\s+[A-Z]{1,3}[.,\s]"),
    re.compile(r"\bdoi[:.]?\s*10\.\d{4,}|doi\.org/10\.\d{4,}", re.I),
]
REF_YEAR = re.compile(r"\b(?:19|20)\d\d[a-z]?\b")
REF_LOC = re.compile(r"\d+\s*[-–]\s*\d+|\bpp?\.\s*\d+|\bVol\.|\*\*\d+\*\*|\d+\(\d+\)")
REF_NAME = re.compile(r"et al\.|[A-Z][a-z]+,\s[A-Z]\.|\*[^*]{3,}\*")
REF_START = re.compile(r"^(?:[A-Z][A-Za-z'\-]+,?\s+(?:[A-Z]\.|[A-Z]{1,3}\b|[A-Z][a-z]+,)|\d{1,4}\.\s|\[\d{1,4}\]|\*)")
# IntechOpen template footer and Frontiers proof-stamp header, both printed on every page.
FIXED_RUNNING_HEADER = re.compile(r"Will-be-set-by-IN-TECH|^f[a-z]{2,8}-\d{2}-\d{5,6} \w+ \d{1,2}, \d{4} Time: \d")
NUMERIC_FRAGMENT = re.compile(r"^[\d;:().,\s\-–*/]+$")
INDEX_ENTRY = re.compile(r"[A-Za-z)][,.]? \d{1,4}(?:[–-]\d{1,4})?(?:, \d{1,4}(?:[–-]\d{1,4})?)*(?= [A-Z(]| *$)")
TOC_WORD = re.compile(r"\b(?:Chapter|Section|Part|Preface|Contents|Introduction|Appendix|Index|Acknowledg|Bibliography|References|Conclusion|Foreword)", re.I)
TOC_TAIL = re.compile(r"\s[*_]*\d{1,4}[*_]*\s*$")
DOT_LEADER = re.compile(r"(?:\.\s?){4,}\s*\d+\s*$")
CAPTION = re.compile(r"^[#*_\s]*(?:Figure|Fig\.|Table|Scheme|Plate|Chart|Graph|Box|Map|Illustration)\s*[A-Z]?\d+(?:\.\d+)*[.:\s]", re.I)
FOOTNOTE_LINE = re.compile(r"^\s*(?:<sup>\d+</sup>|[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s?\S|\d{1,3} \| )")
SECTION_NUMBERED = re.compile(r"^[#*\s]*(?:\d+(?:\.\d+)+\.?\s|[•▪\-–]\s)")
SENTENCE_END = re.compile(r"[a-z]{3,}[.?!] [A-Z]")
SUP_TAG = re.compile(r"</?su[pb]>")
WORD3 = re.compile(r"[^\W\d_]{3,}")
TOKEN = re.compile(r"\S+")
LINE_CLASSES = ("boilerplate", "admin_statement", "page_number", "running_header", "reference", "index", "toc", "caption", "footnote_text", "math_debris", "repeated_line", "heading", "body")

# --------------------------------------------------------------- inline issues
HYPHEN_BREAK = re.compile(r"\b([A-Za-z]{2,})[‐­-](?: |\n+)([a-z]{2,})\b")
HYPHEN_STOP = {"and", "or", "to", "nor", "as", "but", "versus", "vs", "than", "und", "oder", "bis", "et", "ou", "en", "e", "y"}
INLINE = {
    "soft_hyphen": re.compile("\u00ad"),
    "mojibake_utf8_as_latin1": re.compile(r"Ã[\u0080-¿]|Â[\u00a0-¿]|â€[\u0080-¿™œ“”˜]?"),
    "replacement_char": re.compile("\ufffd"),
    "replacement_char_then_space": re.compile(r"[A-Za-z]\ufffd ?[a-z]"),
    "private_use_char": re.compile("[\ue000-\uf8ff]"),
    "cid_glyph": re.compile(r"\(cid:\d+\)"),
    "ligature_codepoint": re.compile("[\ufb00-\ufb06]"),
    "control_char": re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "missing_space_after_punct": re.compile(r"(?<![@/.\w])[a-z]{3,}[.?!;,][A-Z][a-z]{2,}\b(?![@.])"),
    "glued_token_30plus_letters": re.compile(r"(?<![/\w.])[A-Za-z]{30,}(?![\w/.])"),
    "caesar_shift_font_garble": re.compile(r"\b(?:wkh|dqg|zlwk|iru|wkdw|WKH|DQG)\b"),
    "html_sup_sub_tag": re.compile(r"</?su[pb]>"),
    "footnote_marker_sup_after_word": re.compile(r"(?<=[A-Za-z.,;:)\]\"”’])(?:<sup>\d{1,3}</sup>|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)"),
    "html_entity": re.compile(r"&(?:amp|lt|gt|nbsp|quot|#\d+);"),
    "latex_display_math": re.compile(r"\$\$.*?\$\$", re.S),
    "latex_inline_math": re.compile(r"(?<![\\$])\$(?![\s$])[^$\n]{1,200}(?<![\s\\])\$(?!\$)"),
    "escaped_markdown_char": re.compile(r"\\[_*\[\]#]"),
    "markdown_bold_marker": re.compile(r"\*\*"),
    "markdown_heading_line": re.compile(r"^#{1,6} ", re.M),
    "markdown_table_row": re.compile(r"^\|.*\|\s*$", re.M),
    "image_placeholder": re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b", re.I),
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"https?://\S+|www\.\S+"),
    "gap_3plus_newlines": re.compile(r"\n{3,}"),
    "dangling_colon_before_gap": re.compile(r"[A-Za-z,)]:\s*(?:<sup>\d+</sup>|\d{1,3})?\n\n\n"),
    "nc_nd_licence_text": re.compile(r"CC[ -]BY[ -]N[CD]|BY-NC|BY-ND|Non-?Commercial|NoDerivatives|NoDerivs", re.I),
}
UNCOUNTED_EXAMPLES = {"markdown_bold_marker", "markdown_heading_line", "url", "gap_3plus_newlines"}
GARBLE_CHARS = re.compile("[ƀ-ȗȜ-ɏ¢¡]")
PARTIAL_LICENCE = re.compile(r"(?:excluding|except(?: for)?|with the exception of|does not apply to|not covered by)[^.\n]{0,160}(?:chapters?|figures|images|cover|photos|pictures|third[- ]party)[^.\n]{0,160}licen[sc]ed|licen[sc]ed[^.\n]{0,100}(?:excluding|except(?: for)?|with the exception of)[^.\n]{0,60}chapters?", re.I)
FRONTBACK_SECTION = re.compile(r"^(?:meet the editors?|about the editors?|preface|contents|table of contents|we are intechopen|author details|authors? contributions?|acknowledge?ments?|conflicts? of interest|funding|abbreviations|references|bibliography|data availability statement|notes on contributors|index|list of contributors|notes|appendix [a-z0-9]+|supplementary materials?)\b")
INTRO_HEAD = re.compile(r"^[#*\s]*(?:(?:Chapter )?1\.?\s+)?\**(?:Introduction|Preface|Foreword|Prologue|Abstract)\b", re.I | re.M)
PARA_SPLIT = re.compile(r"\n[ \t]*\n")
ORPHAN_START = re.compile(r"^[(\"'“‘]?[a-z]{2,}\b")
# Lowercase openers that legitimately follow a display equation or a list.
CONTINUATION_WORD = re.compile(r"^(?:where|wherein|with|and|or|which|i\.e|e\.g|respectively|if|for|then|so|but)\b")
MID_SENTENCE_END = re.compile(r"\b[a-z]{2,},?\s*$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]+>")
LID_STRIP = re.compile(r"<[^>]{1,10}>|[*_#|\\$]|https?://\S+")
LETTER = re.compile(r"[^\W\d_]")
LOWER_WORD = re.compile(r"(?<![\w\-'’*])[a-z]{2,}(?![\w\-'’*])")


def hash64(text):
    return xxhash.xxh64_intdigest(text.encode("utf-8", "surrogatepass"))


def load_wordlist():
    with open(WORDLIST, encoding="utf-8") as handle:
        return {w.strip().lower() for w in handle if w.strip().isalpha()}


class Reservoir:
    def __init__(self, size, seed):
        self.size, self.items, self.seen = size, [], 0
        self.rng = random.Random(seed)

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.size:
            self.items[slot] = item


def snippet(text, start, end, pad=80):
    return text[max(0, start - pad):min(len(text), end + pad)].replace("\n", "\\n")[:320]


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def is_reference_line(line):
    if any(pattern.search(line) for pattern in REF_LINE):
        return True
    if len(line) < 80 and NUMERIC_FRAGMENT.match(line) and (REF_YEAR.search(line) or re.search(r"\d+\s*[-–]\s*\d+", line)):
        return True
    if len(line) >= 300 or not REF_START.match(line):
        return False
    return bool(REF_YEAR.search(line) and REF_LOC.search(line) and REF_NAME.search(line))


def is_index_line(line):
    entries = len(INDEX_ENTRY.findall(line))
    if entries < 3:
        return False
    words = len(line.split())
    return entries / words >= 0.12 and len(LETTER.findall(line)) >= 0.3 * len(line) and len(SENTENCE_END.findall(line)) <= 1


def is_math_debris(line):
    if len(line) < 30 or line.startswith("$$"):
        return False
    tokens = TOKEN.findall(SUP_TAG.sub(" ", line).replace("*", ""))
    if len(tokens) < 8:
        return False
    wordlike = sum(1 for t in tokens if WORD3.fullmatch(t.strip("*.,;:()[]")))
    return wordlike / len(tokens) < 0.3


def edge_key(line):
    return re.sub(r"[\d*_#|]+", "", line).strip().lower()


def classify_lines(text):
    """Return (line, class) for each non-empty line; context comes from the whole document."""
    stripped = [line.strip() for line in text.split("\n")]
    repeat_count = collections.Counter(s for s in stripped if 10 <= len(s) <= 150 and not s.startswith(("#", "|", "$")))
    edge_keys = collections.Counter(edge_key(s) for s in stripped if len(s) <= 150 and NUM_EDGE.search(s))
    result, section = [], None
    for s in stripped:
        if not s:
            continue
        if REF_HEAD.match(s):
            section = "reference"
            result.append((s, "reference"))
            continue
        if INDEX_HEAD.match(s):
            section = "index"
            result.append((s, "index"))
            continue
        is_heading = bool(HEADING.match(s)) and len(s) < 200
        if is_heading:
            section = None
        bare = MARKUP_CHARS.sub("", s[:BOILERPLATE_SCAN_CHARS]).strip()
        if FIXED_RUNNING_HEADER.search(s):
            cls = "running_header"
        elif BOILERPLATE.search(bare):
            cls = "boilerplate"
        elif ADMIN_STATEMENT.match(bare):
            section = None
            cls = "admin_statement"
        elif PAGE_NUMBER.match(s):
            cls = "page_number"
        elif len(s) <= 150 and NUM_EDGE.search(s) and not SECTION_NUMBERED.match(s) and len(edge_key(s)) >= 8 and edge_keys[edge_key(s)] >= 2:
            cls = "running_header"
        elif CAPTION.match(s) and len(s) < 600:
            cls = "caption"
        elif section == "index" or is_index_line(s):
            cls = "index"
        elif section == "reference" or is_reference_line(s):
            cls = "reference"
        elif (20 <= len(s) < 250 and TOC_TAIL.search(s) and TOC_WORD.search(s)) or DOT_LEADER.search(s):
            cls = "toc"
        elif FOOTNOTE_LINE.match(s):
            cls = "footnote_text"
        elif is_math_debris(s):
            cls = "math_debris"
        elif is_heading:
            cls = "heading"
        elif repeat_count.get(s, 0) >= 3:
            cls = "repeated_line"
        else:
            cls = "body"
        result.append((s, cls))
    return result


def simple_line_class(line):
    bare = MARKUP_CHARS.sub("", line[:BOILERPLATE_SCAN_CHARS]).strip()
    if BOILERPLATE.search(bare) or ADMIN_STATEMENT.match(bare):
        return "boilerplate"
    if REF_HEAD.match(line) or is_reference_line(line):
        return "reference"
    if PAGE_NUMBER.match(line) or CAPTION.match(line) or FOOTNOTE_LINE.match(line):
        return "furniture"
    if HEADING.match(line) and len(line) < 200:
        return "heading"
    return "body"


def language_chunks(text):
    """Yield (chunk, chars) units of at least LID_CHUNK_MIN_CHARS letters-dominated text, built from consecutive paragraphs."""
    buffer, size = [], 0
    for para in PARA_SPLIT.split(text):
        clean = LID_STRIP.sub(" ", para)
        clean = " ".join(clean.split())
        if not clean:
            continue
        buffer.append(clean)
        size += len(clean)
        if size >= LID_CHUNK_MIN_CHARS:
            chunk = " ".join(buffer)
            buffer, size = [], 0
            if len(LETTER.findall(chunk)) >= 0.5 * len(chunk):
                yield chunk, len(chunk)
    if buffer:
        chunk = " ".join(buffer)
        if len(chunk) >= 100 and len(LETTER.findall(chunk)) >= 0.5 * len(chunk):
            yield chunk, len(chunk)


def lid(model, text):
    labels, scores = model.predict(text.lower(), k=1)
    return labels[0].removeprefix("__label__"), float(scores[0])


def word_hashes(words):
    vocab = {}
    inverse = np.fromiter((vocab.setdefault(w, len(vocab)) for w in words), dtype=np.int64, count=len(words))
    hashed = np.fromiter((hash64(w) for w in vocab), dtype=np.uint64, count=len(vocab))
    return hashed[inverse]


def splitmix64(values):
    with np.errstate(over="ignore"):
        z = values + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def bottom_k_sketch(words):
    hashed = word_hashes(words)
    count = len(hashed) - SHINGLE + 1
    combined = np.zeros(count, dtype=np.uint64)
    for offset in range(SHINGLE):
        combined = splitmix64(combined ^ hashed[offset:offset + count])
    return np.unique(combined)[:SKETCH_K]


def normalized_for_dedup(text):
    return re.sub(r"\s+", " ", re.sub(r"[*#_>\\|]", " ", text.lower())).strip()


def dictionary_word_issues(text, wordlist, doc_id, examples):
    """Count dropped f-ligatures (frst, specifc) and words split by a stray space (signifi cant, narrati ve)."""
    ligature_drop = split_word = 0
    tokens = [(m.group(), m.start(), m.end()) for m in LOWER_WORD.finditer(text)]
    for index, (word, start, end) in enumerate(tokens):
        if word in wordlist:
            continue
        if len(word) >= 4 and "f" in word:
            for pos in [i for i, ch in enumerate(word) if ch == "f"]:
                if any(word[:pos + 1] + extra + word[pos + 1:] in wordlist for extra in ("i", "l", "f")):
                    ligature_drop += 1
                    examples["ligature_dropped_fi_fl_ff"].add(f"{doc_id}: {snippet(text, start, end, 50)}")
                    break
        if index + 1 < len(tokens):
            nxt, nstart, nend = tokens[index + 1]
            if nstart == end + 1 and text[end] == " " and (word + nxt) in wordlist and (nxt not in wordlist or len(nxt) <= 3) and len(word + nxt) >= 6:
                split_word += 1
                examples["word_split_by_space"].add(f"{doc_id}: {snippet(text, start, nend, 50)}")
    return ligature_drop, split_word


def paragraph_flow(text, doc_id, examples):
    """Count body paragraphs that start lowercase or end mid-sentence: signs of page, column or footnote breaks inside a paragraph."""
    counts = collections.Counter()
    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    for index, para in enumerate(paragraphs):
        if len(para) < 20 or HEADING.match(para) or para.startswith(("|", "$", "-", "•", ">", "[", "<")):
            continue
        if simple_line_class(para.split("\n", 1)[0]) != "body":
            continue
        counts["body_paragraphs"] += 1
        counts["body_paragraph_chars"] += len(para)
        if ORPHAN_START.match(para) and not CONTINUATION_WORD.match(para):
            counts["orphan_start"] += 1
            counts["orphan_chars"] += len(para)
            examples["paragraph_starts_lowercase"].add(f"{doc_id}: {para[:220]!r}")
        if len(para.split()) >= 12 and MID_SENTENCE_END.search(para):
            counts["truncated_end"] += 1
            following = paragraphs[index + 1:index + 4]
            interrupt = 0
            for later in following:
                first = later.split("\n", 1)[0]
                if ORPHAN_START.match(later) and not HEADING.match(later):
                    counts["truncated_then_lowercase_within_3"] += 1
                    counts[f"interrupted_by:{interrupt_kind(paragraphs[index + 1])}" if interrupt else "joined_directly"] += 1
                    if interrupt:
                        examples["paragraph_interrupted_mid_sentence"].add(f"{doc_id}: ...{para[-120:]!r} || {paragraphs[index + 1][:120]!r} || {later[:80]!r}")
                    break
                if HEADING.match(first) and not FOOTNOTE_LINE.match(first) and not CAPTION.match(first) and "Will-be-set" not in first and not NUM_EDGE.search(first):
                    break
                interrupt += 1
    return counts


def interrupt_kind(block):
    first = block.split("\n", 1)[0].strip()
    if FOOTNOTE_LINE.match(first):
        return "footnote"
    if CAPTION.match(first):
        return "caption"
    if PAGE_NUMBER.match(first) or NUM_EDGE.search(first) or "Will-be-set" in first:
        return "page_header_or_number"
    if first.startswith("$$"):
        return "display_math"
    return "other"


def frontmatter_chars(text):
    """Offset of the first Introduction/Preface/Abstract heading within the first 30 % of the document, else -1."""
    limit = int(len(text) * 0.3)
    match = INTRO_HEAD.search(text, 0, limit)
    return match.start() if match else -1


def index_tail_chars(text):
    """Chars from the last 'Index' heading in the final 25 % of the document to the end."""
    tail_start = int(len(text) * 0.75)
    last = None
    for match in re.finditer(r"^[#*\s]*(?:(?:Subject|Name|Author|General) )?Index[*\s]*$", text[tail_start:], re.M | re.I):
        last = match
    return len(text) - tail_start - last.start() if last else 0


def duplicate_line_fraction(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0, 0
    counts = collections.Counter(lines)
    dup_chars = sum(len(line) for line in lines if counts[line] > 1 and len(line) >= 20)
    longest_run, run = 1, 1
    for prev, cur in zip(lines, lines[1:]):
        run = run + 1 if (cur == prev and len(cur) >= 20) else 1
        longest_run = max(longest_run, run)
    return dup_chars / max(1, sum(len(line) for line in lines)), longest_run


def analyze_doc(doc, model, wordlist, examples):
    text = doc["text"]
    meta = doc["metadata"]
    doc_id = doc["id"]
    row = {"id": doc_id, "book_id": meta.get("book_id"), "section_idx": meta.get("section_idx"), "is_section": "." in doc_id,
           "publisher": meta.get("publisher", ""), "license": meta.get("license", ""), "chars": len(text), "words": len(text.split())}
    classified = classify_lines(text)
    class_chars = collections.Counter()
    for line, cls in classified:
        class_chars[cls] += len(line) + 1
    row["line_chars"] = dict(class_chars)
    rng = random.Random(len(text))
    for cls in LINE_CLASSES[:-2]:
        hits = [line for line, c in classified if c == cls]
        if hits:
            examples[f"line:{cls}"].add(f"{doc_id}: {rng.choice(hits)[:260]}")
    first_line = next((line for line, _ in classified), "")
    first_key = re.sub(r"^[\d.\s]+", "", MARKUP_CHARS.sub("", first_line)).strip().lower()
    row["frontback_section"] = row["is_section"] and len(first_key) < 80 and bool(FRONTBACK_SECTION.match(first_key))
    if row["frontback_section"]:
        examples["frontback_section_doc"].add(f"{doc_id}: {text[:220]!r}")
    row["body_words"] = sum(len(line.split()) for line, cls in classified if cls in ("body", "heading"))
    if row["body_words"] < 30:
        examples["tiny_doc_body_words_lt_30"].add(f"{doc_id}: {text[:220]!r}")
    row["flow"] = dict(paragraph_flow(text, doc_id, examples))

    breaks = [m for m in HYPHEN_BREAK.finditer(text) if m.group(2).lower() not in HYPHEN_STOP]
    if breaks:
        words_lower = set(re.findall(r"[a-z]+", text.lower()))
        row["hyphen_break"] = len(breaks)
        row["hyphen_break_validated"] = sum(1 for m in breaks if (m.group(1) + m.group(2)).lower() in words_lower or (m.group(1) + m.group(2)).lower() in wordlist)
        m = rng.choice(breaks)
        examples["hyphen_line_break"].add(f"{doc_id}: {snippet(text, m.start(), m.end(), 60)}")
    inline = {}
    for name, rx in INLINE.items():
        found = list(rx.finditer(text))
        if found:
            inline[name] = (len(found), sum(m.end() - m.start() for m in found))
            if name not in UNCOUNTED_EXAMPLES:
                m = rng.choice(found)
                examples[f"inline:{name}"].add(f"{doc_id}: {snippet(text, m.start(), m.end(), 60)}")
    row["inline"] = inline
    row["ligature_drop"], row["split_word"] = dictionary_word_issues(text, wordlist, doc_id, examples)
    nonspace = len(text) - text.count(" ") - text.count("\n")
    row["garble_count"] = len(GARBLE_CHARS.findall(text))
    row["garble_ratio"] = row["garble_count"] / nonspace if nonspace else 0.0
    partial = PARTIAL_LICENCE.search(text[:20000])
    row["partial_licence"] = bool(partial)
    if partial:
        examples["partial_licence_statement"].add(f"{doc_id}: {snippet(text, partial.start(), partial.end(), 60)}")
    row["frontmatter_chars"] = frontmatter_chars(text) if not row["is_section"] else -1
    row["index_tail_chars"] = index_tail_chars(text)
    row["dup_line_frac"], row["longest_repeat_run"] = duplicate_line_fraction(text)
    if row["longest_repeat_run"] >= 5:
        examples["consecutive_repeated_line_run_ge5"].add(f"{doc_id}: run={row['longest_repeat_run']}")

    doc_label, doc_score = lid(model, " ".join(text.split()))
    row["lid_doc"] = (doc_label, round(doc_score, 3))
    lid_chars = collections.Counter()
    for chunk, size in language_chunks(text):
        label, score = lid(model, chunk)
        lid_chars[label if score >= 0.5 else "uncertain"] += size
    row["lid_chunks"] = dict(lid_chars.most_common(4))
    row["lid_chunk_total"] = sum(lid_chars.values())
    row["exact_hash"] = hashlib.sha1(normalized_for_dedup(text).encode()).hexdigest()
    return row


# ------------------------------------------------------------- filtered release
def filtered_index_shard(args):
    path, work_dir = args
    index, texts = {}, {}
    for doc in read_jsonl(path):
        text = doc["text"]
        index[doc["id"]] = (len(text), hash64(text))
        if hash64(doc["id"]) % FILTERED_TEXT_SAMPLE_MOD == 0 and len(text) <= FILTERED_TEXT_MAX_CHARS:
            texts[doc["id"]] = text
    return index, texts, sorted(doc["metadata"])


def build_filtered_index(work_dir, paths):
    index, texts, metadata_keys = {}, {}, set()
    with multiprocessing.Pool(PROCESSES) as pool:
        for part_index, part_texts, keys in pool.imap(filtered_index_shard, [(p, work_dir) for p in sorted(paths)]):
            index.update(part_index)
            texts.update(part_texts)
            metadata_keys.update(keys)
    with open(os.path.join(work_dir, "filtered_index.pkl"), "wb") as out:
        pickle.dump({"index": index, "texts": texts, "metadata_keys": sorted(metadata_keys), "shards": [os.path.basename(p) for p in sorted(paths)]}, out)
    print(f"filtered docs {len(index)}, sampled texts {len(texts)}")


def sentence_hash(sentence):
    return hashlib.md5(sentence.strip().encode()).hexdigest()[:16]


def compare_texts(doc_id, original, new, stats, examples):
    old_sents = SENTENCE_SPLIT.split(original)
    new_sents = SENTENCE_SPLIT.split(new)
    matcher = difflib.SequenceMatcher(None, old_sents, new_sents, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            stats["equal_sentences"] += i2 - i1
            continue
        old_part, new_part = " ".join(old_sents[i1:i2]), " ".join(new_sents[j1:j2])
        placeholders = PLACEHOLDER.findall(new_part)
        if placeholders and not PLACEHOLDER.search(old_part):
            for kind in placeholders:
                stats[f"placeholder:{kind}"] += 1
            stats["placeholder_edit_sentences"] += i2 - i1
            examples["filter_placeholder_edit"].add(f"{doc_id}: RAW={old_part[-200:]!r} FILT={new_part[-200:]!r}")
            continue
        if op == "delete" or (op == "replace" and new_part.strip() and new_part.strip() in old_part):
            removed = [s for s in old_sents[i1:i2] if s.strip() and s.strip() not in new_part]
            stats["deleted_sentences"] += len(removed)
            stats["deleted_sentence_chars"] += sum(len(s) + 1 for s in removed)
            if removed:
                examples["filter_deleted_sentence"].add(f"{doc_id}: {removed[0][:300]!r}")
            continue
        stats["other_edit_sentences"] += i2 - i1
        stats["other_edit_chars_delta"] += len(old_part) - len(new_part)
        examples["filter_other_edit"].add(f"{doc_id}: RAW={old_part[:200]!r} FILT={new_part[:200]!r}")


# ------------------------------------------------------------------ raw pass
_WORKER = {}


def init_worker(work_dir):
    with open(os.path.join(work_dir, "filtered_index.pkl"), "rb") as handle:
        _WORKER["filtered"] = pickle.load(handle)
    _WORKER["model"] = fasttext.load_model(LID_MODEL)
    _WORKER["wordlist"] = load_wordlist()


def raw_shard(args):
    raw_path, work_dir = args
    tag = os.path.basename(raw_path).split("_")[0]
    filtered = _WORKER["filtered"]
    model, wordlist = _WORKER["model"], _WORKER["wordlist"]
    examples = collections.defaultdict(lambda: Reservoir(EXAMPLES_PER_ISSUE, hash64(tag)))
    filter_stats = collections.Counter()
    sketch_ids, sketches = [], []
    para_hashes, para_lengths, para_book = [], [], []
    books = {}
    with open(os.path.join(work_dir, f"metrics_{tag}.jsonl"), "w") as out:
        for doc in read_jsonl(raw_path):
            row = analyze_doc(doc, model, wordlist, examples)
            text = doc["text"]
            entry = filtered["index"].get(doc["id"])
            if entry is None:
                row["filter_status"] = "dropped"
            elif entry[1] == hash64(text):
                row["filter_status"] = "same"
            else:
                row["filter_status"] = "modified"
                row["filtered_chars"] = entry[0]
                sample = filtered["texts"].get(doc["id"])
                if sample is not None and len(text) <= FILTERED_TEXT_MAX_CHARS:
                    filter_stats["diffed_docs"] += 1
                    compare_texts(doc["id"], text, sample, filter_stats, examples)
            out.write(json.dumps(row) + "\n")
            normalized = normalized_for_dedup(text)
            words = normalized.split()
            if len(words) >= NEARDUP_MIN_WORDS:
                sketch_ids.append(doc["id"])
                sketches.append(bottom_k_sketch(words))
            book_index = books.setdefault(row["book_id"], len(books))
            for para in PARA_SPLIT.split(text):
                key = normalized_for_dedup(para)
                if len(key) >= PARA_HASH_MIN_CHARS:
                    para_hashes.append(hash64(key))
                    para_lengths.append(len(key))
                    para_book.append(book_index)
    lengths = np.array([len(s) for s in sketches], dtype=np.int32)
    flat = np.concatenate(sketches) if sketches else np.zeros(0, dtype=np.uint64)
    np.savez(os.path.join(work_dir, f"sketch_{tag}.npz"), ids=np.array(sketch_ids), lengths=lengths, flat=flat)
    np.savez(os.path.join(work_dir, f"paras_{tag}.npz"), hashes=np.array(para_hashes, dtype=np.uint64),
             lengths=np.array(para_lengths, dtype=np.int32), book=np.array(para_book, dtype=np.int32),
             book_ids=np.array(list(books), dtype=object), allow_pickle=True)
    with open(os.path.join(work_dir, f"examples_{tag}.json"), "w") as out:
        json.dump({"examples": {k: v.items for k, v in examples.items()}, "filter_stats": dict(filter_stats)}, out, indent=1)
    return tag


def run_raw(work_dir, paths):
    with multiprocessing.Pool(PROCESSES, initializer=init_worker, initargs=(work_dir,)) as pool:
        for tag in pool.imap_unordered(raw_shard, [(p, work_dir) for p in sorted(paths)]):
            print("done", tag, flush=True)


# ------------------------------------------------------------------ report
def percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {f"p{q}": round(float(np.percentile(arr, q)), 4) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)} | {"min": float(arr.min()), "max": float(arr.max()), "mean": round(float(arr.mean()), 2)}


def english_share(row):
    total = row["lid_chunk_total"]
    return row["lid_chunks"].get("en", 0) / total if total else None


def is_garbled(row):
    caesar = row["inline"].get("caesar_shift_font_garble", (0, 0))[0]
    return (row["garble_ratio"] >= 0.02 and row["garble_count"] >= 20) or caesar >= 5


def prevalence(rows):
    n_docs, n_chars = len(rows) or 1, sum(r["chars"] for r in rows) or 1
    out = {}
    for cls in LINE_CLASSES:
        affected = [r for r in rows if r["line_chars"].get(cls)]
        out[f"line:{cls}"] = {"docs_frac": round(len(affected) / n_docs, 4), "chars_frac": round(sum(r["line_chars"].get(cls, 0) for r in rows) / n_chars, 5)}
    for name in INLINE:
        affected = [r for r in rows if name in r["inline"]]
        out[f"inline:{name}"] = {"docs_frac": round(len(affected) / n_docs, 4), "matches": sum(r["inline"][name][0] for r in affected),
                                 "match_chars_frac": round(sum(r["inline"][name][1] for r in affected) / n_chars, 6)}
    for key in ("ligature_drop", "split_word", "hyphen_break", "hyphen_break_validated"):
        affected = [r for r in rows if r.get(key)]
        out[key] = {"docs_frac": round(len(affected) / n_docs, 4), "matches": sum(r.get(key, 0) for r in rows),
                    "docs_frac_ge_10": round(sum(1 for r in rows if r.get(key, 0) >= 10) / n_docs, 4),
                    "per_1M_chars": round(sum(r.get(key, 0) for r in rows) / n_chars * 1e6, 2)}
    garbled = [r for r in rows if is_garbled(r)]
    out["garbled_font_encoding"] = {"docs": len(garbled), "chars_frac": round(sum(r["chars"] for r in garbled) / n_chars, 5)}
    tiny = [r for r in rows if r["body_words"] < 30]
    out["tiny_doc_body_words_lt_30"] = {"docs": len(tiny), "docs_frac": round(len(tiny) / n_docs, 4), "chars_frac": round(sum(r["chars"] for r in tiny) / n_chars, 5)}
    frontback = [r for r in rows if r["frontback_section"]]
    out["frontback_section_doc"] = {"docs": len(frontback), "docs_frac": round(len(frontback) / n_docs, 4), "chars_frac": round(sum(r["chars"] for r in frontback) / n_chars, 5)}
    partial = [r for r in rows if r["partial_licence"]]
    out["partial_licence_statement"] = {"docs": len(partial), "chars_frac": round(sum(r["chars"] for r in partial) / n_chars, 5)}
    repeated = [r for r in rows if r["longest_repeat_run"] >= 5]
    out["consecutive_repeated_line_run_ge5"] = {"docs": len(repeated), "chars_frac": round(sum(r["chars"] for r in repeated) / n_chars, 5)}
    dup_heavy = [r for r in rows if r["dup_line_frac"] >= 0.3 and r["chars"] >= 1000]
    out["dup_line_char_frac_ge_0.3"] = {"docs": len(dup_heavy), "chars_frac": round(sum(r["chars"] for r in dup_heavy) / n_chars, 5)}
    flow = collections.Counter()
    for r in rows:
        flow.update(r["flow"])
    paragraphs = flow["body_paragraphs"] or 1
    out["paragraph_flow"] = {"body_paragraphs": flow["body_paragraphs"], "starts_lowercase_frac": round(flow["orphan_start"] / paragraphs, 4),
                             "ends_mid_sentence_frac": round(flow["truncated_end"] / paragraphs, 4),
                             "ends_mid_sentence_then_lowercase_within_3": flow["truncated_then_lowercase_within_3"],
                             "joined_directly": flow["joined_directly"],
                             "interrupted_by": {k.split(":", 1)[1]: v for k, v in flow.items() if k.startswith("interrupted_by:")},
                             "docs_with_orphan_frac": round(sum(1 for r in rows if r["flow"].get("orphan_start")) / n_docs, 4)}
    return out


def language_report(rows):
    n_docs, n_chars = len(rows), sum(r["chars"] for r in rows)
    doc_labels, doc_label_chars = collections.Counter(), collections.Counter()
    chunk_chars = collections.Counter()
    for r in rows:
        doc_labels[r["lid_doc"][0]] += 1
        doc_label_chars[r["lid_doc"][0]] += r["chars"]
        chunk_chars.update(r["lid_chunks"])
    total_chunk = sum(chunk_chars.values()) or 1
    shares = [(english_share(r), r) for r in rows]
    bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 1.01)]
    band_stats = {}
    for lo, hi in bands:
        members = [r for share, r in shares if share is not None and lo <= share < hi]
        band_stats[f"{lo}-{min(hi, 1.0)}"] = {"docs": len(members), "chars": sum(r["chars"] for r in members),
                                              "whole_book_docs": sum(1 for r in members if not r["is_section"])}
    no_chunk = [r for share, r in shares if share is None]
    doc_scores_en = [r["lid_doc"][1] for r in rows if r["lid_doc"][0] == "en"]
    top_non_en = collections.Counter()
    for share, r in shares:
        if share is not None and share < 0.5:
            non_en = {k: v for k, v in r["lid_chunks"].items() if k not in ("en", "uncertain")}
            if non_en:
                top_non_en[max(non_en, key=non_en.get)] += 1
    return {"doc_level_fasttext_label_docs": dict(doc_labels.most_common(25)),
            "doc_level_fasttext_label_chars_frac": {k: round(v / n_chars, 5) for k, v in doc_label_chars.most_common(25)},
            "doc_level_en_score_percentiles": percentiles(doc_scores_en),
            "doc_level_en_score_lt_0.5_docs": sum(1 for r in rows if r["lid_doc"][0] != "en" or r["lid_doc"][1] <= 0.5),
            "chunk_level_chars_frac_by_label": {k: round(v / total_chunk, 5) for k, v in chunk_chars.most_common(25)},
            "english_chunk_share_bands": band_stats,
            "docs_without_lid_chunk": {"docs": len(no_chunk), "chars": sum(r["chars"] for r in no_chunk)},
            "dominant_non_english_label_for_share_lt_0.5": dict(top_non_en.most_common(20)),
            "docs": n_docs}


def near_duplicates(work_dir, tags, exact_rep):
    ids, sketches = [], []
    for tag in tags:
        with np.load(os.path.join(work_dir, f"sketch_{tag}.npz")) as data:
            flat, lengths, shard_ids = data["flat"], data["lengths"], data["ids"]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        for i, doc_id in enumerate(shard_ids):
            if exact_rep.get(str(doc_id), True):
                ids.append(str(doc_id))
                sketches.append(flat[offsets[i]:offsets[i + 1]])
    values = np.concatenate(sketches)
    owners = np.repeat(np.arange(len(sketches), dtype=np.int64), [len(s) for s in sketches])
    order = np.argsort(values, kind="stable")
    values, owners = values[order], owners[order]
    del order
    run_starts = np.flatnonzero(np.concatenate([[True], values[1:] != values[:-1]]))
    run_ends = np.concatenate([run_starts[1:], [len(values)]])
    run_sizes = run_ends - run_starts
    keep = (run_sizes > 1) & (run_sizes <= POSTING_CAP)
    n_docs = np.int64(len(ids))
    pair_keys = []
    for start, end in zip(run_starts[keep], run_ends[keep]):
        members = np.sort(owners[start:end])
        left, right = np.triu_indices(len(members), k=1)
        pair_keys.append(members[left] * n_docs + members[right])
    keys, shared_counts = np.unique(np.concatenate(pair_keys), return_counts=True)
    del pair_keys
    sketch_sizes = np.array([len(s) for s in sketches], dtype=np.int64)
    first, second = np.divmod(keys, n_docs)
    promising = shared_counts >= 0.25 * np.minimum(sketch_sizes[first], sketch_sizes[second])
    candidate_pairs = len(keys)
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs_05 = pairs_08 = same_book_08 = 0
    sample_pairs = []
    for a, b in zip(first[promising].tolist(), second[promising].tolist()):
        union = np.union1d(sketches[a], sketches[b])[:SKETCH_K]
        jaccard = float((np.isin(union, sketches[a]) & np.isin(union, sketches[b])).sum() / len(union))
        if jaccard >= 0.5:
            pairs_05 += 1
        if jaccard >= 0.8:
            pairs_08 += 1
            same_book_08 += ids[a].split(".")[0] == ids[b].split(".")[0]
            parent[find(a)] = find(b)
            if len(sample_pairs) < 40:
                sample_pairs.append((ids[a], ids[b], round(jaccard, 3)))
    groups = collections.defaultdict(list)
    for index in range(len(ids)):
        groups[find(index)].append(ids[index])
    clustered = [g for g in groups.values() if len(g) > 1]
    return {"docs_considered": len(ids), "candidate_pairs": candidate_pairs, "pairs_sharing_ge_25pct_of_sketch": int(promising.sum()), "sketch_values_skipped_posting_over_cap": int(np.sum(run_sizes > POSTING_CAP)),
            "pairs_jaccard_ge_0.5": pairs_05, "pairs_jaccard_ge_0.8": pairs_08, "pairs_ge_0.8_within_same_book": same_book_08,
            "clusters_jaccard_ge_0.8": len(clustered), "docs_in_clusters": sum(len(g) for g in clustered),
            "redundant_docs_jaccard_ge_0.8": sum(len(g) - 1 for g in clustered),
            "redundant_ids": [doc_id for g in clustered for doc_id in g[1:]],
            "sample_pairs": sample_pairs, "largest_clusters": [g[:6] + [f"...size {len(g)}"] for g in sorted(clustered, key=len, reverse=True)[:15]]}


def paragraph_duplicates(work_dir, tags, book_publisher):
    hashes, lengths, books = [], [], []
    global_books = {}
    for tag in tags:
        with np.load(os.path.join(work_dir, f"paras_{tag}.npz"), allow_pickle=True) as data:
            local_ids = data["book_ids"]
            mapping = np.array([global_books.setdefault(str(b), len(global_books)) for b in local_ids], dtype=np.int32)
            hashes.append(data["hashes"])
            lengths.append(data["lengths"])
            books.append(mapping[data["book"]] if len(data["book"]) else data["book"])
    hashes, lengths, books = np.concatenate(hashes), np.concatenate(lengths), np.concatenate(books)
    order = np.lexsort((books, hashes))
    hashes, lengths, books = hashes[order], lengths[order], books[order]
    starts = np.flatnonzero(np.concatenate([[True], hashes[1:] != hashes[:-1]]))
    sizes = np.diff(np.concatenate([starts, [len(hashes)]]))
    group_of = np.repeat(np.arange(len(starts)), sizes)
    first_book = books[starts][group_of]
    last_book = books[np.concatenate([starts[1:], [len(hashes)]]) - 1][group_of]
    in_group = sizes[group_of] > 1
    is_first = np.zeros(len(hashes), dtype=bool)
    is_first[starts] = True
    cross_book = in_group & (first_book != last_book)
    redundant = in_group & ~is_first
    total_chars = int(lengths.sum())
    id_of = {v: k for k, v in global_books.items()}
    per_book_total = np.bincount(books, weights=lengths, minlength=len(global_books))
    per_book_shared = np.bincount(books[cross_book], weights=lengths[cross_book], minlength=len(global_books))
    share = np.divide(per_book_shared, per_book_total, out=np.zeros_like(per_book_total), where=per_book_total > 0)
    top_groups = np.argsort(-sizes)[:12]
    top = [{"copies": int(sizes[g]), "chars": int(lengths[starts[g]]), "books": int(len(np.unique(books[starts[g]:starts[g] + sizes[g]])))} for g in top_groups]
    contained = [(id_of[i], round(float(share[i]), 3), book_publisher.get(id_of[i], "")) for i in np.argsort(-share)[:25] if share[i] > 0]
    pub_shared = collections.Counter()
    pub_total = collections.Counter()
    for i in range(len(global_books)):
        pub = book_publisher.get(id_of[i], "")
        pub_total[pub] += per_book_total[i]
        pub_shared[pub] += per_book_shared[i]
    return {"paragraphs_hashed_ge_200_chars": int(len(hashes)), "paragraph_chars": total_chars,
            "redundant_paragraph_copies": int(redundant.sum()), "redundant_paragraph_chars": int(lengths[redundant].sum()),
            "redundant_paragraph_chars_frac": round(float(lengths[redundant].sum()) / total_chars, 5),
            "paragraphs_shared_across_books": int(cross_book.sum()), "chars_in_paragraphs_shared_across_books_frac": round(float(lengths[cross_book].sum()) / total_chars, 5),
            "books": len(global_books),
            "books_with_cross_book_shared_paragraph_share": {f">={t}": int((share >= t).sum()) for t in (0.1, 0.3, 0.5, 0.8, 0.95)},
            "largest_paragraph_groups": top, "most_contained_books": contained,
            "cross_book_shared_chars_frac_by_publisher": {p: round(pub_shared[p] / pub_total[p], 4) for p, _ in pub_total.most_common(12)}}


def load_rows(work_dir, tags):
    rows = []
    for tag in tags:
        with open(os.path.join(work_dir, f"metrics_{tag}.jsonl")) as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def shard_tags(work_dir):
    return sorted(name[len("metrics_"):-len(".jsonl")] for name in os.listdir(work_dir) if name.startswith("metrics_"))


def report(work_dir, report_dir):
    tags = shard_tags(work_dir)
    rows = load_rows(work_dir, tags)
    with open(os.path.join(work_dir, "filtered_index.pkl"), "rb") as handle:
        filtered = pickle.load(handle)
    out = {"shards": tags, "docs": len(rows), "total_chars": sum(r["chars"] for r in rows),
           "chars": percentiles([r["chars"] for r in rows]), "words": percentiles([r["words"] for r in rows])}
    sections = [r for r in rows if r["is_section"]]
    whole = [r for r in rows if not r["is_section"]]
    books = collections.defaultdict(list)
    for r in rows:
        books[r["book_id"]].append(r)
    out["structure"] = {"books": len(books), "section_docs": len(sections), "section_chars": sum(r["chars"] for r in sections),
                        "whole_book_docs": len(whole), "whole_book_chars": sum(r["chars"] for r in whole),
                        "section_docs_chars": percentiles([r["chars"] for r in sections]), "whole_book_chars_pct": percentiles([r["chars"] for r in whole]),
                        "sections_per_book": percentiles([len(v) for v in books.values() if v[0]["is_section"]]),
                        "section_publishers": collections.Counter(r["publisher"] for r in sections).most_common(5),
                        "length_buckets": {f"{lo}-{hi}": {"docs": sum(1 for r in rows if lo <= r["chars"] < hi), "chars": sum(r["chars"] for r in rows if lo <= r["chars"] < hi)}
                                           for lo, hi in [(0, 200), (200, 1000), (1000, 10000), (10000, 100000), (100000, 1000000), (1000000, 10 ** 9)]}}
    pub_docs, pub_chars, pub_books = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        pub_docs[r["publisher"]] += 1
        pub_chars[r["publisher"]] += r["chars"]
    for members in books.values():
        pub_books[members[0]["publisher"]] += 1
    out["publishers_top"] = [(p, pub_books[p], pub_docs[p], c) for p, c in pub_chars.most_common(25)]
    out["publishers_distinct"] = len(pub_chars)
    lic_docs, lic_chars = collections.Counter(), collections.Counter()
    for r in rows:
        lic_docs[r["license"]] += 1
        lic_chars[r["license"]] += r["chars"]
    out["licenses"] = [(k, v, lic_chars[k]) for k, v in lic_docs.most_common()]
    out["issues_all"] = prevalence(rows)
    out["issues_section_docs"] = prevalence(sections)
    out["issues_whole_book_docs"] = prevalence(whole)
    out["issues_by_publisher"] = {p: {k: v for k, v in prevalence([r for r in rows if r["publisher"] == p]).items()
                                      if k in ("line:running_header", "line:reference", "line:index", "line:boilerplate", "line:math_debris", "ligature_drop", "split_word", "hyphen_break", "inline:footnote_marker_sup_after_word", "inline:replacement_char", "paragraph_flow")}
                                  for p, _ in pub_chars.most_common(8)}
    front = [r["frontmatter_chars"] for r in whole if r["frontmatter_chars"] >= 0]
    out["frontmatter_whole_books"] = {"docs_with_intro_heading_in_first_30pct": len(front), "of_whole_books": len(whole),
                                      "chars_before_intro": percentiles(front), "chars_before_intro_total_frac_of_whole_book_chars": round(sum(front) / max(1, sum(r["chars"] for r in whole)), 5)}
    tails = [r["index_tail_chars"] for r in rows if r["index_tail_chars"] > 0]
    out["index_tail"] = {"docs": len(tails), "whole_book_docs": sum(1 for r in whole if r["index_tail_chars"] > 0), "chars": sum(tails), "chars_pct": percentiles(tails)}
    out["language"] = language_report(rows)
    out["language_whole_books"] = language_report(whole)

    status = collections.Counter(r["filter_status"] for r in rows)
    status_chars = collections.Counter()
    for r in rows:
        status_chars[r["filter_status"]] += r["chars"]
    raw_ids = {r["id"] for r in rows}
    modified = [r for r in rows if r["filter_status"] == "modified"]
    dropped = [r for r in rows if r["filter_status"] == "dropped"]
    kept = [r for r in rows if r["filter_status"] != "dropped"]
    filter_stats = collections.Counter()
    for tag in tags:
        with open(os.path.join(work_dir, f"examples_{tag}.json")) as handle:
            filter_stats.update(json.load(handle)["filter_stats"])

    def drop_rate(predicate):
        members = [r for r in rows if predicate(r)]
        return {"docs": len(members), "dropped": sum(1 for r in members if r["filter_status"] == "dropped"),
                "chars": sum(r["chars"] for r in members), "dropped_chars": sum(r["chars"] for r in members if r["filter_status"] == "dropped")}

    out["filter"] = {"filtered_shards": filtered["shards"], "filtered_metadata_keys": filtered["metadata_keys"], "filtered_docs": len(filtered["index"]),
                     "filtered_chars": sum(v[0] for v in filtered["index"].values()),
                     "filtered_ids_not_in_raw": sum(1 for k in filtered["index"] if k not in raw_ids),
                     "status_docs": dict(status), "status_raw_chars": dict(status_chars),
                     "modified_chars_removed": sum(r["chars"] - r["filtered_chars"] for r in modified),
                     "modified_chars_delta_pct": percentiles([r["chars"] - r["filtered_chars"] for r in modified]),
                     "diff_sample_stats": dict(filter_stats),
                     "drop_rate_by_length": {f"{lo}-{hi}": drop_rate(lambda r, lo=lo, hi=hi: lo <= r["chars"] < hi) for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 50000), (50000, 10 ** 9)]},
                     "drop_rate_by_doc_lid": {lab: drop_rate(lambda r, lab=lab: r["lid_doc"][0] == lab) for lab in ("en", "de", "fr", "es", "it", "pt")},
                     "drop_rate_doc_en_score_le_0.5": drop_rate(lambda r: r["lid_doc"][0] != "en" or r["lid_doc"][1] <= 0.5),
                     "drop_rate_doc_en_score_gt_0.5": drop_rate(lambda r: r["lid_doc"][0] == "en" and r["lid_doc"][1] > 0.5),
                     "drop_rate_by_english_chunk_share": {f"{lo}-{hi}": drop_rate(lambda r, lo=lo, hi=hi: english_share(r) is not None and lo <= english_share(r) < hi) for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]},
                     "drop_rate_section_vs_whole": {"section": drop_rate(lambda r: r["is_section"]), "whole": drop_rate(lambda r: not r["is_section"])},
                     "drop_rate_by_publisher": {p: drop_rate(lambda r, p=p: r["publisher"] == p) for p, _ in pub_chars.most_common(10)},
                     "dropped_not_explained_by_length_le_200_or_lang": drop_rate(lambda r: r["filter_status"] == "dropped" and r["chars"] > 200 and r["lid_doc"][0] == "en" and r["lid_doc"][1] > 0.5),
                     "issues_kept_by_filter": {k: v for k, v in prevalence(kept).items() if not k.startswith("inline:markdown")},
                     "issues_dropped_by_filter": {k: v for k, v in prevalence(dropped).items() if not k.startswith("inline:markdown")}}

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["exact_hash"]].append(r)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    exact_rep = {r["id"]: i == 0 for g in groups.values() for i, r in enumerate(g)}
    out["exact_dup"] = {"groups": len(dup_groups), "redundant_docs": sum(len(g) - 1 for g in dup_groups),
                        "redundant_chars": sum(sum(r["chars"] for r in g[1:]) for g in dup_groups),
                        "redundant_docs_cross_book": sum(1 for g in dup_groups for r in g[1:] if r["book_id"] != g[0]["book_id"]),
                        "redundant_docs_ge_1000_chars": sum(1 for g in dup_groups for r in g[1:] if r["chars"] >= 1000),
                        "redundant_chars_ge_1000_chars": sum(r["chars"] for g in dup_groups for r in g[1:] if r["chars"] >= 1000),
                        "redundant_docs_kept_by_filter": sum(1 for g in dup_groups for r in g[1:] if r["filter_status"] != "dropped"),
                        "largest": [(len(g), g[0]["id"], g[0]["chars"]) for g in sorted(dup_groups, key=len, reverse=True)[:15]],
                        "largest_by_chars": [(len(g), g[0]["id"], g[0]["chars"], [r["id"] for r in g[1:4]]) for g in sorted(dup_groups, key=lambda g: -g[0]["chars"] * (len(g) - 1))[:10]]}
    near = near_duplicates(work_dir, tags, exact_rep)
    with open(os.path.join(work_dir, "near_dup_redundant_ids.json"), "w") as handle:
        json.dump(near.pop("redundant_ids"), handle)
    out["near_dup"] = near
    book_publisher = {r["book_id"]: r["publisher"] for r in rows}
    out["paragraph_dup"] = paragraph_duplicates(work_dir, tags, book_publisher)
    examples = collections.defaultdict(list)
    for tag in tags:
        with open(os.path.join(work_dir, f"examples_{tag}.json")) as handle:
            for key, items in json.load(handle)["examples"].items():
                examples[key].extend(items[:3])
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "report.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=str)
    with open(os.path.join(report_dir, "examples.json"), "w") as handle:
        json.dump(examples, handle, indent=1)


def sample_for_reading(work_dir, report_dir):
    """Write excerpts of docs per English-share band and per filter status, for manual calibration."""
    tags = shard_tags(work_dir)
    rows = load_rows(work_dir, tags)
    rng = random.Random(3)
    wanted = {}
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]:
        members = [r for r in rows if english_share(r) is not None and lo <= english_share(r) < hi and r["chars"] >= 2000]
        for r in rng.sample(members, min(8, len(members))):
            wanted[r["id"]] = f"en_share {lo}-{hi}"
    dropped_unexplained = [r for r in rows if r["filter_status"] == "dropped" and r["chars"] > 200 and r["lid_doc"][0] == "en" and r["lid_doc"][1] > 0.5]
    for r in rng.sample(dropped_unexplained, min(10, len(dropped_unexplained))):
        wanted[r["id"]] = "filter dropped, en>0.5, >200 chars"
    shard_of = {}
    for tag in tags:
        with open(os.path.join(work_dir, f"metrics_{tag}.jsonl")) as handle:
            for line in handle:
                doc_id = json.loads(line)["id"]
                if doc_id in wanted:
                    shard_of[doc_id] = tag
    by_id = {r["id"]: r for r in rows if r["id"] in wanted}
    samples = []
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "common_pile", "doab")
    for tag in sorted(set(shard_of.values())):
        for doc in read_jsonl(os.path.join(raw_dir, f"{tag}_doab.json.gz")):
            if doc["id"] in wanted:
                text = doc["text"]
                middle = len(text) // 2
                samples.append({"id": doc["id"], "reason": wanted[doc["id"]], "chars": len(text), "lid_doc": by_id[doc["id"]]["lid_doc"],
                                "lid_chunks": by_id[doc["id"]]["lid_chunks"], "title": doc["metadata"].get("title"),
                                "head": text[:700], "middle": text[middle:middle + 700]})
    with open(os.path.join(report_dir, "reading_samples.json"), "w") as handle:
        json.dump(samples, handle, indent=1, ensure_ascii=False)


# ------------------------------------------------------------- extra pass
# Patterns found while reading documents after the raw pass had run.
COMPILATION_RESTRICTION = re.compile(r"book as a whole[^.\n]{0,160}(?:CC BY-NC-ND|reserved|without INTECH)|book taken as a whole is ©[^.\n]{0,120}(?:CC BY-NC-ND|reserved)|compilation of articles constituting this e-book[^.\n]{0,80}exclusive property", re.I)
MATH_FONT_GARBLE = re.compile(r"ð Þ|(?<=\S) [¼þ] (?=\S)|<sup>[¼þ]</sup>")
CATALOGUE_AD = re.compile(r"\d+,\d\d € \((?:DE|A)\)|\bpb\., (?:ill\., )?\d+,\d\d €")
COVER_CREDIT = re.compile(r"(?:Photo|Image|Cover image|Cover photo)s? (?:by|courtesy of|credit:?)[^\n]{0,80}/ ?(?:iStock|Shutterstock|Getty)|© [^\n]{1,40} ?/ ?iStock", re.I)
GLUE_SCAN_CHARS = 300_000
GLUE_TOKEN = re.compile(r"(?<![\w\-'’\\/.@])[a-z]{6,}(?![\w\-'’/@]|\.[a-z])")
# A function word fused to a dictionary word is almost never a real compound, so it marks a lost space.
FUNCTION_WORDS = {"the", "of", "and", "to", "for", "with", "that", "was", "are", "from", "this", "which", "its", "their", "has", "have", "not", "were", "been"}


def glued_word_split(word, wordlist):
    """Return (left, right) if a token outside the word list is a function word fused to a word-list word of 4+ letters, else None."""
    if word in wordlist:
        return None
    for cut in range(2, len(word) - 1):
        left, right = word[:cut], word[cut:]
        if (left in FUNCTION_WORDS and len(right) >= 4 and right in wordlist) or (right in FUNCTION_WORDS and len(left) >= 4 and left in wordlist):
            return left, right
    return None


def next_content_line(lines, start):
    for line in lines[start:]:
        if line.strip():
            return line.strip()
    return None


def extra_doc(doc, wordlist, examples):
    text, doc_id = doc["text"], doc["id"]
    row = {"id": doc_id, "publisher": doc["metadata"].get("publisher", ""), "chars": len(text)}
    lines = text.split("\n")
    ref_empty = ref_full = caption_no_table = caption_with_table = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if REF_HEAD.match(stripped):
            following = next_content_line(lines, index + 1)
            if following is None or HEADING.match(following) or BOILERPLATE.search(MARKUP_CHARS.sub("", following[:BOILERPLATE_SCAN_CHARS])):
                ref_empty += 1
                examples["reference_heading_without_entries"].add(f"{doc_id}: {stripped!r} -> {str(following)[:120]!r}")
            else:
                ref_full += 1
        elif re.match(r"^[#*_\s]*Table\s*[A-Z]?\d+(?:\.\d+)*[.:]", stripped):
            following = next_content_line(lines, index + 1)
            caption_tail = next_content_line(lines, index + 2) if following and len(following) < 300 and following.startswith("*") else None
            if (following and following.startswith("|")) or (caption_tail and caption_tail.startswith("|")):
                caption_with_table += 1
            else:
                caption_no_table += 1
                examples["table_caption_without_table_rows"].add(f"{doc_id}: {stripped[:160]!r} -> {str(following)[:120]!r}")
    row["reference_heading_empty"], row["reference_heading_with_entries"] = ref_empty, ref_full
    row["table_caption_without_rows"], row["table_caption_with_rows"] = caption_no_table, caption_with_table
    for name, pattern in (("compilation_restriction", COMPILATION_RESTRICTION), ("math_font_garble", MATH_FONT_GARBLE),
                          ("catalogue_ad", CATALOGUE_AD), ("cover_credit", COVER_CREDIT)):
        found = list(pattern.finditer(text))
        row[name] = len(found)
        if found:
            match = found[len(found) // 2]
            examples[name].add(f"{doc_id}: {snippet(text, match.start(), match.end(), 80)}")
    glued = 0
    scanned = text[:GLUE_SCAN_CHARS]
    for match in GLUE_TOKEN.finditer(scanned):
        split = glued_word_split(match.group(), wordlist)
        if split:
            glued += 1
            examples["glued_words"].add(f"{doc_id}: {match.group()} = {split[0]}+{split[1]} :: {snippet(scanned, match.start(), match.end(), 40)}")
    row["glued_words"], row["glue_scanned_chars"] = glued, len(scanned)
    return row


def extra_shard(args):
    raw_path, work_dir = args
    tag = os.path.basename(raw_path).split("_")[0]
    wordlist = load_wordlist()
    examples = collections.defaultdict(lambda: Reservoir(EXAMPLES_PER_ISSUE * 3, hash64(tag)))
    with open(os.path.join(work_dir, f"extra_{tag}.jsonl"), "w") as out:
        for doc in read_jsonl(raw_path):
            out.write(json.dumps(extra_doc(doc, wordlist, examples)) + "\n")
    with open(os.path.join(work_dir, f"extra_examples_{tag}.json"), "w") as out:
        json.dump({k: v.items for k, v in examples.items()}, out, indent=1, ensure_ascii=False)
    return tag


def run_extra(work_dir, paths):
    with multiprocessing.Pool(PROCESSES) as pool:
        for tag in pool.imap_unordered(extra_shard, [(p, work_dir) for p in sorted(paths)]):
            print("done", tag, flush=True)


def extra_report(work_dir, report_dir):
    tags = sorted(name[len("extra_"):-len(".jsonl")] for name in os.listdir(work_dir) if name.startswith("extra_") and name.endswith(".jsonl"))
    counts = ("reference_heading_empty", "reference_heading_with_entries", "table_caption_without_rows", "table_caption_with_rows",
              "compilation_restriction", "math_font_garble", "catalogue_ad", "cover_credit", "glued_words")
    docs, chars, scanned = 0, 0, 0
    affected_docs, affected_chars, totals = collections.Counter(), collections.Counter(), collections.Counter()
    by_publisher = collections.defaultdict(collections.Counter)
    publisher_docs = collections.Counter()
    for tag in tags:
        with open(os.path.join(work_dir, f"extra_{tag}.jsonl")) as handle:
            for line in handle:
                row = json.loads(line)
                docs += 1
                chars += row["chars"]
                scanned += row["glue_scanned_chars"]
                publisher_docs[row["publisher"]] += 1
                for key in counts:
                    if row[key]:
                        affected_docs[key] += 1
                        affected_chars[key] += row["chars"]
                        totals[key] += row[key]
                        by_publisher[key][row["publisher"]] += 1
    examples = collections.defaultdict(list)
    for tag in tags:
        with open(os.path.join(work_dir, f"extra_examples_{tag}.json")) as handle:
            for key, items in json.load(handle).items():
                examples[key].extend(items[:4])
    out = {"shards": tags, "docs": docs, "chars": chars,
           "per_measure": {k: {"docs": affected_docs[k], "docs_frac": round(affected_docs[k] / docs, 5), "chars_of_affected_docs_frac": round(affected_chars[k] / chars, 5),
                               "matches": totals[k], "top_publishers_docs": by_publisher[k].most_common(6)} for k in counts},
           "glued_words_per_1M_scanned_chars": round(totals["glued_words"] / scanned * 1e6, 2),
           "publisher_docs_top": publisher_docs.most_common(10)}
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "extra_report.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    with open(os.path.join(report_dir, "extra_examples.json"), "w") as handle:
        json.dump(examples, handle, indent=1, ensure_ascii=False)


def main():
    command, work_dir = sys.argv[1], sys.argv[2]
    os.makedirs(work_dir, exist_ok=True)
    if command == "filtered-index":
        build_filtered_index(work_dir, sys.argv[3:])
    elif command == "raw":
        run_raw(work_dir, sys.argv[3:])
    elif command == "report":
        report(work_dir, sys.argv[3])
    elif command == "sample":
        sample_for_reading(work_dir, sys.argv[3])
    elif command == "extra":
        run_extra(work_dir, sys.argv[3:])
    elif command == "extra-report":
        extra_report(work_dir, sys.argv[3])
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
