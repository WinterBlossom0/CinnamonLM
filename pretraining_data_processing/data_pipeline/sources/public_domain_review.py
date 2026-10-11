"""Cleaning pipeline for Common Pile public_domain_review (findings: analysis/findings/public_domain_review.json)."""
from pathlib import Path

import numpy as np
import regex
import scipy.sparse

from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import (
    Deduplicator, DuplicateOfWriter, ExactDedup, KeyIndexWriter, MinHasher, dedup_words, load_index, shard_offsets,
    shard_prefix,
)
from data_pipeline.common.langid import LanguageIdentifier
from data_pipeline.common.pii import mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, Step, TextMap
from data_pipeline.common.text import normalize_text

# Every post has a title, an optional subtitle, an optional byline and a date line in its first lines (1412/1412).
DATE_LINE = regex.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}")
HEADER_MAX_LINES = 6

# Two or more NBSPs stand where a <br> was in verse and dialogue; a digit or a dot after them is a ledger column.
NBSP_LINE_BREAK = regex.compile(r"[ \t]*\xa0{2,}[\xa0 \t]*(?=[\p{Lu}\"“‘(])")

HTML_ANCHOR = regex.compile(r"<a\b[^>\n]*>([^<\n]*)(?:</a>)?", regex.IGNORECASE)
MARKDOWN_LINK = regex.compile(r"\[([^\]\n]+)\]\(https?://[^)\n]*\)")
BARE_URL = regex.compile(r"[ \t]*(?:https?://|www\.)\S+")

# Template lines in 3 or more posts, or links to the shop or comments (findings); exact lines as in Common Pile line_tagger.
BOILERPLATE_LINE = regex.compile(
    r"The text of this essay is published under a CC BY-SA license, see here for details\."
    r"|Okkult Motion Pictures is the brainchild .*"
    r"|Bill Domonkos is an experimental filmmaker and video artist\. .*"
    r"|You can see the full film featured .{0,80}\bhere\b.*"
    r"|Some people you think we've missed\? Please let us know in the comments!"
    r"|Wondering if \"bad things happen to works when they enter the public domain\"\? Wonder no more\."
    r"|\(Learn more about the situation in the U\.S\. .*\)\.?"
    r"|MP3 Download / Internet Archive Link|Download from Internet Archive"
    r"|You can scroll through our highlights below\."
    r"|And a few others that didn't make it to the class photo\.*"
)
SENTENCE_START = r"(?:^|(?<=[.!?)]))[ \t]*"
# A dot inside a word, as in "isaac-newton.org", does not end the sentence.
SENTENCE_REST = r"(?:[^.!?\n]|[.!?](?=[\p{Ll}\d]))"
# A word between the verb and "here" keeps quoted speech such as "See here comes a truck".
# In the passive form, a comma before "here" or a ", which" clause marks a main clause with content, so it stays.
LINK_SENTENCE = regex.compile(
    SENTENCE_START
    + r"\(?(?:You can (?:also )?|Also )?(?:See|see|Read|read|Find|find|Watch|View|Browse|browse)\s+\S[^.!?\n]{0,120}?"
      r"\bhere\b" + SENTENCE_REST + r"{0,80}(?:[.!:]\)?|$)"
    + r"|" + SENTENCE_START + r"For more\b[^.!?\n]{0,120}\bsee (?:our [^.!?\n]{0,40})?here\."
    + r"|" + SENTENCE_START + r"\(?(?=\p{Lu})(?:(?!,\s*which\b)[^.!?\n]){0,100}?\b(?:can|may) be "
      r"(?:seen|found|read|viewed|downloaded|heard|streamed)\b[^.!?\n,]{0,80}\bhere\b"
      + SENTENCE_REST + r"{0,80}(?:[.!:]\)?|$)"
    + r"|" + SENTENCE_START + r"Learn more\b[^.!?\n,]{0,60}\bhere\b[^.!?\n]{0,40}(?:[.!:]|$)",
    regex.MULTILINE,
)
SITE_SENTENCE = regex.compile(
    SENTENCE_START + r"(?:All (?:Okkult )?animated GIFs(?: by [\w ]+)? published here under a CC-BY-SA license\."
    r"|See more [^.!?\n]{0,80}in our Animated GIFs Collection\."
    r"|Below is a little bit more about each of their lives \(with each name linking through to their respective "
    r"Wikipedia pages[^)\n]*\)\.)",
    regex.MULTILINE,
)
# Requests to use the shop, the comments or the page widgets; a long lead clause holds content, so it stays.
READER_PROMPT_SENTENCE = regex.compile(
    SENTENCE_START + r"(?=\p{Lu})(?:[^.!?\n]{0,200}\bour (?:online )?shop\b[^.!?\n(]{0,120}(?:\([^)\n]*\))?"
    r"|[^.!?\n]{0,90}\b(?:let us know|share your thoughts|enlighten us)\b[^.!?\n]{0,60}\bin the comments\b[^.!?\n]{0,60}"
    r"|[^.!?\n]{0,80}\b(?:click (?:on|the)|embedded (?:reader|player|viewer))\b[^.!?\n]{0,160})(?:[.!?]|$)",
    regex.MULTILINE,
)
# The site's index widget ("※※Indexed under…" + term + subentry, 89 in 78 posts) ends its line and splits paragraphs.
INDEX_WIDGET = regex.compile(r"<?[ \t]*※※Indexed under…[^\n]*")
# A print-layout note glued between list items in two posts; it marks where an item ended, so it becomes a line break.
PRINT_NOTE = regex.compile(r"[ \t]*Scroll through the whole page to download all images before printing\.")
# Removing the Okkult credit block leaves its "*" separators at the end of the post with nothing after them.
TRAILING_SEPARATOR = regex.compile(r"[*\s]*")
# A forwarded e-mail header whose addresses the author had already starred out (the-primordial-gound).
MASKED_EMAIL_HEADER = regex.compile(r"From: [^\n]*?\*{3,}[^\n]*?(?=Subject:)")

# Essay notes were not scraped (findings: 40/40 sampled matches were markers); "Fm7" and "mp3" have under 3 letters.
# A number before " & <number>" is a list item such as crossword clue "borax25 & 30." (beastly-clues).
# Before ",;—–" a number after ":" is a page reference, as in "(Baldwin:41, also 81–2)".
FOOTNOTE_MARKER = regex.compile(
    r"(?<=\p{Ll}{3}|[\p{L}\d]\p{L}[.,;:!?]|\S[\"”’)\]][.,;:!?]?|[.,;:!?][\"”’)\]]"
    r"|[.!?…][.!?]|[—–][.,;:!?]|[’']\p{L}[.,;:!?]|[.!?] [\"”])"
    r"(?<!\b(?:[Nn]o|pp?|vol|fig|ch|ed|sec|cf|op)\.)"
    r"(?:\d{1,3}(?=\s|$)|(?<!:)\d{1,3}(?=[;,—–]))(?! & \d)"
)
NOTED_TYPES = frozenset({"essay", "conjecture"})

# Real names in this source with an inner capital after a capitalized part; every other "WordWord" is a joined line.
COMPOUND_NAMES = (
    ("You", "Tube"), ("Wiki", "Commons"), ("World", "Cat"), ("Bio", "Lib"), ("Bibli", "Odyssey"), ("Raw", "Pixel"),
    ("Trans", "Manche"), ("Guts", "Muths"), ("Cap", "Nap"), ("Hansetsu", "Roku"), ("Fore", "Bodings"),
)
NOT_COMPOUND_NAME = "(?!" + "|".join(f"(?<=\\b{head}){tail}\\b" for head, tail in COMPOUND_NAMES) + ")"
# Stripped <br> tags joined verse and list lines (findings: 40/40 sampled matches); "Dr.Caligari" has one lowercase letter.
# A line can start with a capitalized word of 1-2 letters ("mementoOf the dead"; 30/30 matches read were joins).
JOINED_LINE = regex.compile(
    r"(?<=\p{Ll}[’']?\p{Ll}[.!?,;:])(?<!\bMrs\.)(?=\p{Lu}\p{Ll}|I\b)"
    r"|(?<=\b\p{Ll}{3,})(?=\p{Lu}\p{Ll}{2})"
    r"|(?<=\b\p{Ll}{3,})(?=(?:I|A|O|Of|In|To|On|At|By|Or|An|As|My|No|So|Be|He|We|It|Is|If|Oh)\b)"
    r"|(?<=\b\p{Lu}\p{Ll}{2,})(?<!\b(?:Mac|Fitz))" + NOT_COMPOUND_NAME + r"(?=\p{Lu}\p{Ll}{2})"
    r"|(?<=\p{Ll}{3})(?=\d{4}\b)"
    r"|(?<=\b\d{4})(?=\p{Lu}\p{Ll}{2})"
)


def restore_nbsp_line_breaks(text: str) -> str:
    return NBSP_LINE_BREAK.sub("\n", text)


def split_header(text: str) -> tuple[list[str], str]:
    """Header lines up to and including the date line, and the body after it; no header when no date line is found."""
    lines = text.split("\n")
    for index, line in enumerate(lines[:HEADER_MAX_LINES]):
        if DATE_LINE.fullmatch(line.strip()):
            return [header_line for header_line in lines[:index + 1] if header_line.strip()], \
                "\n".join(lines[index + 1:]).strip()
    return [], text


def normalize_header(text: str) -> str:
    """Title, subtitle, byline and date on consecutive lines, then one blank line before the body."""
    header, body = split_header(text)
    if not header:
        return text
    return "\n".join(header) + ("\n\n" + body if body else "")


def remove_sentences(pattern: regex.Pattern, line: str) -> str:
    """Removes each match; a ')' that closes a '(' opened before the match stays, so parentheses stay balanced."""
    def replacement(match: regex.Match) -> str:
        before, removed = line[:match.start()], match.group()
        opened_before = before.count("(") > before.count(")")
        closes_outside = removed.count(")") > removed.count("(")
        return ")" if (opened_before and closes_outside) else ""
    return pattern.sub(replacement, line)


def remove_boilerplate(text: str) -> str:
    text = INDEX_WIDGET.sub("", text)
    text = MASKED_EMAIL_HEADER.sub("\n", text)
    text = PRINT_NOTE.sub("\n", text)
    lines = []
    for line in text.split("\n"):
        kept = line
        for pattern in (SITE_SENTENCE, LINK_SENTENCE, READER_PROMPT_SENTENCE):
            kept = remove_sentences(pattern, kept)
        kept = kept.rstrip()
        if not line[:1].isspace():
            kept = kept.lstrip()
        # A line that held only removed sentences goes too, so no blank line is left.
        if (line.strip() and not kept.strip()) or BOILERPLATE_LINE.fullmatch(line.strip()) \
                or BOILERPLATE_LINE.fullmatch(kept.strip()):
            continue
        lines.append(kept)
    while lines and TRAILING_SEPARATOR.fullmatch(lines[-1]):
        lines.pop()
    return normalize_text("\n".join(lines))


def remove_markup(text: str) -> str:
    text = HTML_ANCHOR.sub(r"\1", text)
    text = MARKDOWN_LINK.sub(r"\1", text)
    return BARE_URL.sub("", text)


class FootnoteMarkers(Step):
    def apply(self, document: dict) -> dict:
        if document.get("type") not in NOTED_TYPES:
            return document
        text = FOOTNOTE_MARKER.sub("", document["text"])
        return document if text == document["text"] else {**document, "text": text}


def restore_joined_lines(text: str) -> str:
    return JOINED_LINE.sub("\n", text)


class ShingleIndexer:
    def __init__(self, hasher: MinHasher, directory: Path, shard_index: int):
        self.hasher = hasher
        self.shingles = KeyIndexWriter(shard_prefix(directory, shard_index), sort_by_key=False)

    def add(self, local_id: int, document: dict) -> None:
        # Re-posts change the title and date, which lowers containment below the threshold (findings).
        words = dedup_words(split_header(document["text"])[1])
        if words:
            self.shingles.add(local_id, self.hasher.shingles(words))

    def close(self) -> None:
        self.shingles.close()


class ContainmentDedup(Deduplicator):
    """Drops a document when at least min_containment of its word 5-gram shingles are in one larger document.

    Re-posts with an added intro have Jaccard 0.64-0.76 but containment up to 1.0 (findings), so a Jaccard-0.8 test
    misses them; containment is the S2ORC partial-copy metric. The in-memory matrix limits this to small sources.
    """

    def __init__(self, name: str = "near_dedup", min_containment: float = 0.8):
        super().__init__(name)
        self.min_containment = min_containment
        self.hasher = MinHasher()

    def open_indexer(self, directory: Path, shard_index: int) -> ShingleIndexer:
        return ShingleIndexer(self.hasher, directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        shingle_pieces, id_pieces = [], []
        for shard_index, offset in enumerate(shard_offsets(shard_counts)):
            shingles, local_ids = load_index(directory, shard_index, mmap=False)
            shingle_pieces.append(shingles)
            id_pieces.append(local_ids.astype(np.int64) + offset)
        shingles, global_ids = np.concatenate(shingle_pieces), np.concatenate(id_pieces)
        if dropped is not None:
            live = ~dropped[global_ids]
            shingles, global_ids = shingles[live], global_ids[live]
        _, shingle_columns = np.unique(shingles, return_inverse=True)
        size = sum(shard_counts)
        incidence = scipy.sparse.csr_matrix(
            (np.ones(len(global_ids), dtype=np.int32), (global_ids, shingle_columns.reshape(-1))),
            shape=(size, int(shingle_columns.max(initial=-1)) + 1))
        set_sizes = np.asarray(incidence.sum(axis=1)).reshape(-1)
        overlaps = (incidence @ incidence.T).tocoo()
        smaller, larger, shared = overlaps.row, overlaps.col, overlaps.data
        # Rank by set size, then by earlier global id, so exactly one document of each pair is the larger one.
        is_larger = (set_sizes[larger] > set_sizes[smaller]) | (
            (set_sizes[larger] == set_sizes[smaller]) & (larger < smaller))
        contained = is_larger & (shared >= self.min_containment * set_sizes[smaller])
        smaller, larger = smaller[contained], larger[contained]
        # Each contained document points to its highest-ranked container: the last one after this sort.
        order = np.lexsort((-larger, set_sizes[larger], smaller))
        smaller, larger = smaller[order], larger[order]
        is_last = np.ones(len(smaller), dtype=bool)
        is_last[:-1] = smaller[1:] != smaller[:-1]
        with DuplicateOfWriter(directory, size) as duplicate_of:
            duplicate_of[smaller[is_last]] = larger[is_last]
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "contained_pairs": int(len(order))}


# Findings "Stub documents": under 200 body chars a post is a title, a date and at most one caption sentence.
MIN_BODY_CHARS = 200
# Dolma (arXiv:2402.00159) value; on all 1,412 documents the lowest share is 0.81 and every document is English.
MIN_ENGLISH_SHARE = 0.5


# Known limit: 54 "(Wikipedia)" paragraphs stay (42 of 53 posts have no other body); cross-source paragraph dedup must act.
def build_pipeline() -> Pipeline:
    language_identifier = LanguageIdentifier()
    return Pipeline("public_domain_review", [
        TextMap("nbsp_line_breaks", restore_nbsp_line_breaks),
        TextMap("normalize", normalize_text),
        TextMap("header", normalize_header),
        TextMap("markup", remove_markup),
        TextMap("boilerplate", remove_boilerplate),
        FootnoteMarkers("footnote_markers"),
        TextMap("joined_lines", restore_joined_lines),
        Filter("stub", lambda document: len(split_header(document["text"])[1]) >= MIN_BODY_CHARS),
        Filter("english", lambda document: language_identifier.english_share(document["text"]) >= MIN_ENGLISH_SHARE),
        # Gopher A.3.2 order: quality filters, then exact dedup, then fuzzy dedup; PII masking last as in Common Pile.
        ExactDedup(),
        ContainmentDedup(),
        TextMap("pii", mask_pii),
    ])


if __name__ == "__main__":
    run_cli(build_pipeline)
