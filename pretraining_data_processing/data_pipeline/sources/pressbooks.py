"""Cleaning pipeline for Common Pile pressbooks; "findings" means analysis/findings/pressbooks.json."""
from pathlib import Path

import numpy as np
import regex

from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import (
    Deduplicator, DuplicateOfWriter, ExactDedup, NearDedup, dedup_words, hash_word, save_array_atomically, shard_offsets,
    shard_prefix,
)
from data_pipeline.common.licence import LicenceCreditStrip
from data_pipeline.common.langid import ENGLISH_LABEL, LETTER, LanguageIdentifier, fasttext_input
from data_pipeline.common.pii import MASKERS, PII_KINDS, mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, TextMap
from data_pipeline.common.text import normalize_text

# Findings "site chrome": the book table of contents sits between these two lines; no chapter text is inside.
NAV_START = "Primary Navigation"
NAV_END = "Previous/next navigation"
NAV_HEADER_LINES = frozenset({NAV_START, "Book Contents Navigation", NAV_END})
# Lines repeated across many books, removed on exact match like the Common Pile line_tagger (LITERATURE §0).
BOILERPLATE_LINES = frozenset({
    "Want to create or adapt books like this? Learn more about how Pressbooks supports open publishing practices.",
    "Main Body",
    "Click play on the following audio player to listen along as you read this section.",
    "Answers to interactive elements are given within the interactive element.",
    "<!– pb_fixme –>",
    "<!-- pb_fixme -->",
    "Feedback/Errata",
    "Audio",
})
# Findings "UI widget strings": labels of buttons whose content was not scraped.
UI_LINE = regex.compile(
    r"(?i)(?:(?:show|hide|reveal|toggle) (?:answers?|solutions?|glossary|hints?|explanation|feedback|transcript)"
    r"|licen[cs]es and attributions \(click to expand\)|\(?click to expand\)?|jump to the next .* box\..*"
    r"|an interactive (?:h5p )?element has been excluded from this version of the text\..*"
    r"|(?:one or more )?interactive elements? (?:has|have) been excluded from this version of the text\..*"
    r"|you can view (?:it|them) online here:?.*)"
)
# Findings "dangling references to removed embeds": short lines that only point at a video, audio or quiz.
EMBED_POINTER = regex.compile(
    r"(?i)(?:(?:please )?(?:watch|view|play|listen to) (?:the|this) (?:following |short |brief )?"
    r"(?:videos?|audio|clip|podcast)(?: below| above)?.{0,150}"
    r"|(?:video|audio|podcast) (?:attribution|credit|source):.*"
    r"|listen to an audio version of this page.*"
    r"|interactive learning activity:?.*)"
)
# A one-sentence line that sends the reader to a video, audio or quiz "below/above" that the export does not hold.
EMBED_REFERENCE = regex.compile(
    r"(?i)\b(?:the|this) (?:following |short |brief )?(?:videos?|audio|podcast|quiz|clip)(?: clip)? (?:below|above)\b"
)
SENTENCE_END = regex.compile(r"[.!?](?=\s|$)")
MAX_POINTER_LINE_CHARS = 250

# Findings "math shortcodes": [latex]X[/latex] is unwrapped to $X$; \phantom{\rule{..}{..}} is MathML spacing junk.
# The body may not hold another tag, so an unpaired literal "[latex]" in a book about the shortcode stays as text.
LATEX_SHORTCODE = regex.compile(r"\[latex(?: [^\]\n]*)?\]((?:(?!\[/?latex)[\s\S]){0,4000}?)\[/latex\]")
LATEX_PAGE = regex.compile(r"\[/?latexpage\]")
PHANTOM_RULE = regex.compile(r"\\phantom\{\\rule\{[^{}]*\}\{[^{}]*\}\}")
# Findings "unrendered shortcodes": these hold no reader-visible content.
EMPTY_SHORTCODE = regex.compile(
    r"\[reveal-answer[^\]\n]*\].*?\[/reveal-answer\]|\[practice-area[^\]\n]*\](?:.*?\[/practice-area\])?"
    r"|\[ohm_question[^\]\n]*\].*?\[/ohm_question\]|\[h5p[^\]\n]*\]|\[table [^\]\n]*not found /\]",
    regex.DOTALL,
)
# A footnote stays next to the word it explains, in parentheses, so it does not run into that word.
FOOTNOTE_SHORTCODE = regex.compile(
    r"[^\S\n]*\[footnote(?:\s[^\]\n]{0,200})?\]((?:(?!\[/?footnote)[\s\S]){0,4000}?)\[/footnote\]"
)
# Wrappers whose inner text is the content (answers, glossary terms, footnotes) lose only the tags.
WRAPPER_SHORTCODE = regex.compile(
    r"\[/?(?:hidden-answer|glossary-page|glossary-term|glossary-definition|pb_glossary|footnote|textbox|caption"
    r"|heading|anchor|equation)(?:\s[^\]\n]{0,200})?\]"
)
# Findings "image credits": OpenStax "(credit: ...)" spans name a photographer, not content.
CREDIT_SPAN = regex.compile(r" ?\((?:image |photo |figure )?credits?(?: [a-z])?:[^()]*(?:\([^()]*\)[^()]*)*\)")
# Books that teach Pressbooks print shortcodes as examples, so a doc that names them keeps every tag literal.
SHORTCODE_TOPIC = regex.compile(r"(?i)\bshortcodes?\b")
FOOTNOTE_RETURN = regex.compile(r"\s*(?:↵|\[back\])\s*$")
# OpenStax-derived books keep "([link])" where a figure reference was; the figure is not in the text.
FIGURE_LINK = regex.compile(r" ?\(\[link\]\)")

# Findings "table debris": lone "|" lines between cells and empty rows carry no content.
TABLE_RESIDUE = regex.compile(r"[\s|]*\|[\s|]*")
TABLE_SEPARATOR = regex.compile(r"\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*")
# Findings "flattened tables": a row is a line "|" or "| first cell |", then one line per cell that ends in " |".
ROW_START = regex.compile(r"\|(?:\s*([^|]*?)\s*\|)?")
EMPTY_BULLET = regex.compile(r"[-*•]")
# Pressbooks prints the chapter ordinal before the chapter title, e.g. "14 7. Foreseeable Barriers" (findings).
CHAPTER_ORDINAL = regex.compile(r"\d{1,3}(?: (?=\S)|$)")
MAX_TITLE_CHARS = 200
TITLE_LINES = 2

# Findings "licence/attribution boilerplate": the licence stays in metadata.license (LITERATURE §f).
ATTRIBUTION_HEADING = regex.compile(
    r"(?i)(?:(?:text |media |image |photo |video )?attributions?|(?:image|photo) credits?|candela citations"
    r"|licen[cs]es?(?: (?:and|&) attributions?)?|attributions? (?:and|&) licen[cs]es?|licen[cs]e and attribution"
    r"|(?:cc|creative commons) licensed content(?:,.*)?|public domain content|all rights reserved content"
    r"|open content,.*):?"
)
MEDIA_ATTRIBUTIONS_HEADING = "Media Attributions"
# Credit fields of the attribution blocks read; prose about licences ("all CC licenses require ...") has none of them.
CREDIT_FIELD = regex.compile(
    r"(?i)©|\bCC[- ]?BY\b|\bCC0\b|provided by:|located at:|authored by:|\blicen[cs]e:|\blicen[cs]ed under\b"
    r"|\bused under\b|\bcourtesy of\b|unsplash|pixabay|pexels|wikimedia commons|flickr|standard youtube licen[cs]e"
    r"|^\W*(?:text |content |images? |photos? )?(?:adapted|remixed|derived) from\b|^(?:image |photo )?(?:source|credit)s?:"
)
# Media lists hold image names and credits; a longer item without a credit field is a footnote that lost its "↵".
MAX_MEDIA_NAME_WORDS = 12
# Lines that are a licence statement or an image credit wherever they appear.
LICENCE_STATEMENT = regex.compile(
    r"(?i).{0,300}\b(?:is|are) licen[cs]ed under (?:a |the )?(?:CC|Creative Commons|Public Domain|Standard YouTube).*"
    r"|.*\bcopyright ©.*\blicen[cs]ed\b.*|.*\bexcept where otherwise noted\b.*|to the extent possible under law.*"
    r"|.*\bis free of known copyright restrictions\b.*|.*\b(?:provided|authored) by:.*\b(?:licen[cs]e|located at):.*"
    r"|licen[cs]e: (?:CC|Public Domain|All Rights Reserved|Other)\b.*"
    r"|(?:access|download) (?:it |the original |this book )?for free at.*"
    r"|.*,? via Wikimedia Commons\.?"
    r"|this (?:work|chapter|page|section|book) is (?:a derivative of|adapted from)\b.*"
    r"|see the (?:attributions?|credits?|licen[cs]es?)(?: (?:and|&) (?:attributions?|licen[cs]es?))? page\b.*"
)
# An image credit names its medium and a licence or stock site; prose about photos names neither.
IMAGE_CREDIT_START = regex.compile(
    r"(?i)(?:unless otherwise noted, )?(?:all )?(?:image|photo|picture|illustration|figure|map|icon|graphic)s?\b"
    r".{0,200}\b(?:by|from|courtesy of|credits?:?)\b"
)
LICENCE_OR_STOCK_SITE = regex.compile(
    r"(?i)\bCC[- ]?BY\b|\bCC0\b|creative commons|public domain|\blicen[cs]ed\b|unsplash|pixabay|pexels|wikimedia"
    r"|flickr|\bopenstax\b"
)
# Credit lines are short; a longer line that matches is prose about licensing, which OER training books teach.
MAX_CREDIT_LINE_CHARS = 400
# Findings "repeated lines": in the sample only directly repeated blocks were artifacts; far repeats were exercises.
MIN_REPEATED_BLOCK_CHARS = 100
# Outline templates repeat short items on purpose; printed-twice blocks hold at least one prose line.
MIN_REPEATED_LINE_CHARS = 60

STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})
# Gopher rules (arXiv:2112.11446 A.1.1); its other rules removed mostly intact textbook pages in the sample.
MIN_WORDS = 50
MIN_MEAN_WORD_CHARS = 3
MAX_MEAN_WORD_CHARS = 10
MIN_STOP_WORDS = 2
# Markdown table pipes and separators are not words; counting them made pure tables fail the word-length rule.
TABLE_SYNTAX = regex.compile(r"[|:\-]+")
# Read calibration: 0.2 kept 144 of 157 English-taught language lessons and removed 80 of 144 non-English pages.
ENGLISH_MIN_SHARE = 0.2
# Below the clear share, non-English running prose drops a doc; read labels in that band: 67/70 English kept, 25/35 not.
ENGLISH_CLEAR_SHARE = 0.5
RUNNING_PROSE_MIN_WORDS = 25
MAX_FOREIGN_PROSE_SHARE = 0.3
# The licence filter keeps these NonCommercial text-source statements, which the commercial-use decision excludes.
NONCOMMERCIAL = r"(?:non-?commercial|\bCC[- ]?BY[- ]NC)"
NONCOMMERCIAL_TEXT_SOURCE = regex.compile(
    rf"(?im)^(?:by .{{1,300}}\bused under\b.{{0,40}}{NONCOMMERCIAL}"
    rf"|(?:this|the) (?:chapter|section|page|book|work|text|module|unit)\b.{{0,80}}\b(?:adapted|remixed|derived)\b"
    rf".{{0,800}}{NONCOMMERCIAL})"
)
# Placeholder Latin of template pages (C4, Raffel et al. §2.2); a page that only names "lorem ipsum" stays.
PLACEHOLDER_TEXT = regex.compile(r"(?i)lorem ipsum dolor")
PHONE_PATTERN, mask_phone = MASKERS["phone"]
# One "." between six and four digits is a course or catalogue code ("201809.1055"); NANP numbers are not written so.
DOTTED_CODE = regex.compile(r"\d{6}\.\d{4}")
# Findings "whole books embedded as one chapter": MinHash cannot match a chapter inside a book, but its long lines can.
CONTAINED_LINE_MIN_CHARS = 200
# Same share as the near-dedup Jaccard threshold (Lee et al., arXiv:2107.06499).
MIN_CONTAINED_SHARE = 0.8
# Long lines must also carry most of the doc, so a doc of short lines is not dropped for one shared paragraph.
MIN_CONTAINED_DOC_SHARE = 0.5
# A line shared by more docs is boilerplate; skipping it bounds the pairs at MAX_LINE_DOCS^2 per line.
MAX_LINE_DOCS = 50


def strip_navigation(text: str) -> str:
    lines = text.split("\n")
    if NAV_START in lines and NAV_END in lines:
        start, stop = lines.index(NAV_START), lines.index(NAV_END)
        if start < stop:
            lines = lines[:start] + lines[stop + 1:]
    return "\n".join(line for line in lines if line not in NAV_HEADER_LINES)


def strip_shortcodes(text: str) -> str:
    if SHORTCODE_TOPIC.search(text):
        return text
    text = FOOTNOTE_SHORTCODE.sub(lambda match: f" ({match.group(1).strip()})", text)
    text = LATEX_SHORTCODE.sub(lambda match: f"${PHANTOM_RULE.sub(' ', match.group(1)).strip()}$", text)
    text = PHANTOM_RULE.sub(" ", LATEX_PAGE.sub("", text))
    return FIGURE_LINK.sub("", WRAPPER_SHORTCODE.sub("", EMPTY_SHORTCODE.sub("", text)))


def is_cell_line(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) > 1 and stripped.endswith("|") and not stripped.startswith("|")


def rebuild_flattened_tables(text: str) -> str:
    """Joins each flattened row (row start, then one line per cell) into one "| a | b |" line."""
    lines = text.split("\n")
    rebuilt = []
    index = 0
    while index < len(lines):
        start = ROW_START.fullmatch(lines[index].strip())
        stop = index + 1
        while stop < len(lines) and is_cell_line(lines[stop]):
            stop += 1
        if start is None or stop == index + 1:
            rebuilt.append(lines[index])
            index += 1
            continue
        cells = [start.group(1)] if start.group(1) else []
        cells.extend(line.strip()[:-1].strip() for line in lines[index + 1:stop])
        rebuilt.append("| " + " | ".join(cells) + " |")
        index = stop
    return "\n".join(rebuilt)


def strip_chapter_ordinal(text: str) -> str:
    lines = text.split("\n")
    for index, line in enumerate(lines[:TITLE_LINES]):
        ordinal = CHAPTER_ORDINAL.match(line)
        if ordinal and len(line) <= MAX_TITLE_CHARS:
            remainder = line[ordinal.end():]
            lines[index:index + 1] = [remainder] if remainder else []
            break
    return "\n".join(lines)


def is_credit_line(line: str) -> bool:
    return len(line) <= MAX_CREDIT_LINE_CHARS and (bool(CREDIT_FIELD.search(line)) or is_licence_line(line))


def is_media_item(line: str) -> bool:
    return line.startswith("- ") and (is_credit_line(line) or len(line.split()) <= MAX_MEDIA_NAME_WORDS)


def is_embed_pointer(line: str) -> bool:
    if len(line) > MAX_POINTER_LINE_CHARS:
        return False
    if EMBED_POINTER.fullmatch(line):
        return True
    return bool(EMBED_REFERENCE.search(line)) and len(SENTENCE_END.findall(line)) <= 1


def is_licence_line(line: str) -> bool:
    if len(line) > MAX_CREDIT_LINE_CHARS:
        return False
    if LICENCE_STATEMENT.fullmatch(line):
        return True
    return bool(IMAGE_CREDIT_START.match(line)) and bool(LICENCE_OR_STOCK_SITE.search(line))


def is_chrome_line(line: str) -> bool:
    if line in BOILERPLATE_LINES or UI_LINE.fullmatch(line) or TABLE_RESIDUE.fullmatch(line) \
            or EMPTY_BULLET.fullmatch(line):
        return True
    return is_embed_pointer(line) or is_licence_line(line)


def strip_line_boilerplate(text: str) -> str:
    """Removes chrome lines, licence statements, attribution blocks and orphan table separators."""
    raw_lines = [CREDIT_SPAN.sub("", line) for line in text.split("\n")]
    # The "↵" return mark is the one sign that a "- " item is a footnote, so blocks end at it before it is removed.
    is_footnote = [bool(FOOTNOTE_RETURN.search(line)) for line in raw_lines]
    lines = [FOOTNOTE_RETURN.sub("", line) for line in raw_lines]
    # Matching uses stripped lines; kept lines keep their indentation, which code and verse need.
    keys = [line.strip() for line in lines]
    kept = []
    index = 0
    while index < len(lines):
        key = keys[index]
        if key == MEDIA_ATTRIBUTIONS_HEADING or ATTRIBUTION_HEADING.fullmatch(key):
            block_end = index + 1
            in_media_list = key == MEDIA_ATTRIBUTIONS_HEADING
            while block_end < len(lines) and not is_footnote[block_end] and (
                    is_credit_line(keys[block_end]) or (in_media_list and is_media_item(keys[block_end]))):
                block_end += 1
            # A heading with no credit under it points to an attribution page outside the doc, so it also goes.
            index = block_end
            continue
        index += 1
        if is_chrome_line(key):
            continue
        if TABLE_SEPARATOR.fullmatch(key) and not (kept and kept[-1].lstrip().startswith("|")):
            continue
        kept.append(lines[index - 1])
    return "\n".join(kept)


def drop_repeated_blocks(text: str) -> str:
    """Removes a block of at least MIN_REPEATED_BLOCK_CHARS chars that directly repeats the block before it."""
    lines = text.split("\n")
    last_seen = {}
    # Distance to the previous copy of the line; a run of k lines at distance k is a block printed twice.
    distances = []
    for index, line in enumerate(lines):
        previous = last_seen.get(line)
        is_candidate = previous is not None and line != "" and not line.startswith("|")
        distances.append(index - previous if is_candidate else 0)
        last_seen[line] = index
    kept = []
    start = 0
    while start < len(lines):
        stop = start + 1
        while stop < len(lines) and distances[stop] == distances[start]:
            stop += 1
        run = lines[start:stop]
        distance = distances[start]
        is_double = distance > 0 and stop - start >= distance
        is_artifact = is_double and sum(len(line) for line in run) >= MIN_REPEATED_BLOCK_CHARS \
            and any(len(line) >= MIN_REPEATED_LINE_CHARS for line in run)
        if not is_artifact:
            kept.extend(run)
        start = stop
    return "\n".join(kept)


def foreign_prose_share(text: str, language_identifier: LanguageIdentifier) -> float:
    """Fraction of the letters of text in non-table lines of RUNNING_PROSE_MIN_WORDS words that are not English."""
    total_letters = len(LETTER.findall(text))
    prose_lines = [line for line in text.split("\n") if len(line.split()) >= RUNNING_PROSE_MIN_WORDS
                   and not line.lstrip().startswith("|") and LETTER.search(line)]
    if not prose_lines or not total_letters:
        return 0.0
    labels, _ = language_identifier.model.predict([fasttext_input(line) for line in prose_lines], k=1)
    foreign_letters = sum(len(LETTER.findall(line)) for line, line_labels in zip(prose_lines, labels)
                          if line_labels[0] != ENGLISH_LABEL)
    return foreign_letters / total_letters


def is_english(text: str, language_identifier: LanguageIdentifier) -> bool:
    share = language_identifier.english_share(text)
    if share < ENGLISH_MIN_SHARE:
        return False
    if share >= ENGLISH_CLEAR_SHARE:
        return True
    return foreign_prose_share(text, language_identifier) < MAX_FOREIGN_PROSE_SHARE


def quality_failure(text: str) -> str | None:
    """Name of the first Gopher quality rule that the text fails, or None."""
    words = [word for word in text.split() if not TABLE_SYNTAX.fullmatch(word)]
    if len(words) < MIN_WORDS:
        return "min_words"
    mean_word_chars = sum(len(word) for word in words) / len(words)
    if not MIN_MEAN_WORD_CHARS <= mean_word_chars <= MAX_MEAN_WORD_CHARS:
        return "mean_word_length"
    if sum(1 for word in words if word.lower() in STOP_WORDS) < MIN_STOP_WORDS:
        return "stop_words"
    return None


def long_line_keys(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Distinct hashes of the dedup-normalized lines of at least CONTAINED_LINE_MIN_CHARS chars, with their chars."""
    chars_by_key = {}
    for line in text.split("\n"):
        words = dedup_words(line) if len(line) >= CONTAINED_LINE_MIN_CHARS else []
        if words:
            chars_by_key[hash_word(" ".join(words))] = len(line)
    return np.fromiter(chars_by_key, dtype=np.uint64), np.fromiter(chars_by_key.values(), dtype=np.int64)


def containment_index_path(directory: Path, shard_index: int, field: str) -> Path:
    prefix = shard_prefix(directory, shard_index)
    return prefix.with_name(f"{prefix.name}.{field}.npy")


class ContainmentIndexer:
    def __init__(self, directory: Path, shard_index: int):
        self.directory, self.shard_index = directory, shard_index
        self.keys, self.local_ids, self.line_chars, self.doc_chars = [], [], [], []

    def add(self, local_id: int, document: dict) -> None:
        keys, line_chars = long_line_keys(document["text"])
        self.keys.append(keys)
        self.local_ids.append(np.full(len(keys), local_id, dtype=np.int64))
        self.line_chars.append(line_chars)
        self.doc_chars.append(len(document["text"]))

    def close(self) -> None:
        fields = {"keys": (self.keys, np.uint64), "ids": (self.local_ids, np.int64),
                  "line_chars": (self.line_chars, np.int64)}
        for field, (parts, dtype) in fields.items():
            array = np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype=dtype)
            save_array_atomically(containment_index_path(self.directory, self.shard_index, field), array)
        save_array_atomically(containment_index_path(self.directory, self.shard_index, "doc_chars"),
                              np.array(self.doc_chars, dtype=np.int64))


class ContainmentDedup(Deduplicator):
    """Drops a doc when MIN_CONTAINED_SHARE of its long-line chars are in one longer kept doc.

    Docs are resolved from the longest down, so a container is never itself dropped after it absorbed a doc.
    The index holds only long lines (about 2 million for pressbooks), so it is built in RAM.
    """

    def open_indexer(self, directory: Path, shard_index: int) -> ContainmentIndexer:
        return ContainmentIndexer(directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        keys, global_ids, line_chars, doc_chars = [], [], [], []
        for shard_index, offset in enumerate(shard_offsets(shard_counts)):
            keys.append(np.load(containment_index_path(directory, shard_index, "keys")))
            global_ids.append(np.load(containment_index_path(directory, shard_index, "ids")) + offset)
            line_chars.append(np.load(containment_index_path(directory, shard_index, "line_chars")))
            doc_chars.append(np.load(containment_index_path(directory, shard_index, "doc_chars")))
        keys, global_ids, line_chars = np.concatenate(keys), np.concatenate(global_ids), np.concatenate(line_chars)
        doc_chars = np.concatenate(doc_chars)
        if dropped is not None:
            live = ~dropped[global_ids]
            keys, global_ids, line_chars = keys[live], global_ids[live], line_chars[live]
        long_chars = np.bincount(global_ids, weights=line_chars, minlength=len(doc_chars))
        # Rank orders docs by length; of two docs of equal length the earlier one ranks higher, as dedup keeps it.
        rank = np.empty(len(doc_chars), dtype=np.int64)
        rank[np.lexsort((-np.arange(len(doc_chars)), doc_chars))] = np.arange(len(doc_chars))
        shared = self.shared_chars(keys, global_ids, line_chars, rank)
        with DuplicateOfWriter(directory, len(doc_chars)) as duplicate_of:
            for doc in sorted(shared, key=lambda doc: -rank[doc]):
                containers = [(chars, container) for container, chars in shared[doc].items()
                              if duplicate_of[container] < 0]
                if not containers:
                    continue
                chars, container = max(containers)
                if chars >= MIN_CONTAINED_SHARE * long_chars[doc] and chars >= MIN_CONTAINED_DOC_SHARE * doc_chars[doc]:
                    duplicate_of[doc] = container
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "long_lines": int(len(keys))}

    @staticmethod
    def shared_chars(keys: np.ndarray, global_ids: np.ndarray, line_chars: np.ndarray,
                     rank: np.ndarray) -> dict[int, dict[int, int]]:
        """For each doc, the long-line chars it shares with each longer doc."""
        order = np.argsort(keys, kind="stable")
        keys, global_ids, line_chars = keys[order], global_ids[order], line_chars[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        stops = np.r_[starts[1:], len(keys)]
        shared: dict[int, dict[int, int]] = {}
        for start, stop in zip(starts.tolist(), stops.tolist()):
            if stop - start < 2 or stop - start > MAX_LINE_DOCS:
                continue
            docs = global_ids[start:stop].tolist()
            for position, doc in enumerate(docs):
                for container in docs:
                    if rank[container] > rank[doc]:
                        doc_shared = shared.setdefault(doc, {})
                        doc_shared[container] = doc_shared.get(container, 0) + int(line_chars[start + position])
        return shared


def mask_personal_data(text: str) -> str:
    text = mask_pii(text, kinds=tuple(kind for kind in PII_KINDS if kind != "phone"))
    return PHONE_PATTERN.sub(lambda match: match.group(0) if DOTTED_CODE.fullmatch(match.group(0)) else mask_phone(match),
                             text)


def build_pipeline() -> Pipeline:
    language_identifier = LanguageIdentifier()
    return Pipeline("pressbooks", [
        TextMap("normalize", normalize_text),
        Filter("noncommercial_source", lambda document: NONCOMMERCIAL_TEXT_SOURCE.search(document["text"]) is None),
        TextMap("navigation", strip_navigation),
        TextMap("shortcodes", strip_shortcodes),
        TextMap("tables", rebuild_flattened_tables),
        TextMap("boilerplate_lines", strip_line_boilerplate),
        TextMap("repeated_blocks", drop_repeated_blocks),
        TextMap("chapter_ordinal", strip_chapter_ordinal),
        # Line removals can leave blank-line runs and trailing spaces; normalize_text collapses them.
        TextMap("renormalize", normalize_text),
        Filter("gopher_quality", lambda document: quality_failure(document["text"]) is None),
        Filter("placeholder", lambda document: PLACEHOLDER_TEXT.search(document["text"]) is None),
        Filter("english", lambda document: is_english(document["text"], language_identifier)),
        ExactDedup("exact_dedup"),
        NearDedup("near_dedup"),
        ContainmentDedup("containment_dedup"),
        TextMap("pii", mask_personal_data),
        LicenceCreditStrip("licence_credits"),  # output audit: one media credit classified as strip
    ])


if __name__ == "__main__":
    run_cli(build_pipeline)
