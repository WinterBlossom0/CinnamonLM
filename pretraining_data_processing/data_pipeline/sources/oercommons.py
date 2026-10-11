"""Cleaning pipeline for Common Pile oercommons.

Measured findings: analysis/findings/oercommons.json. Literature: analysis/LITERATURE.md.
"""
import html
import os
from pathlib import Path

import numpy as np
import regex
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import (
    DuplicateOfWriter, ExactDedup, MinHashIndexer, NearDedup, dedup_words, iter_key_buckets, save_array_atomically,
    shard_offsets, shard_prefix,
)
from data_pipeline.common.io import replace_durably, temp_path_for
from data_pipeline.common.langid import LanguageIdentifier
from data_pipeline.common.pii import PHONE_TOKEN, mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, TextMap
from data_pipeline.common.text import normalize_text

# Findings "login/navigation chrome": all 92 pages with this line are 227-299 chars of site navigation.
LOGIN_CHROME = regex.compile(r"Sign in to see your Hubs")
# Spun merchandise promotion; a scan of every distinct page for spin and shop phrases found no other such page.
SPAM_PAGE_IDS = frozenset({"https://oercommons.org/courseware/lesson/103598/overview"})
# Findings "commercial promos": Off2Class pages advertise a vendor account, but 19 of 80 also hold 150+ words of
# English lesson prose, so only the sentences that name the vendor go.
VENDOR_NAME = regex.compile(r"(?i)off2class")
VENDOR_TITLE_SUFFIX = regex.compile(r"(?i)\s+[-–]\s+Off2Class (?:ESL )?Lesson Plan\s*$")
# A "." or "!" before a lowercase letter, a digit or "/" is inside a URL such as off2class.com, not a sentence end.
SENTENCE_BODY = r"(?:[^.!?\n]|[.!?](?=[a-z0-9/]))*"
VENDOR_SENTENCE = regex.compile(rf"{SENTENCE_BODY}(?i:off2class){SENTENCE_BODY}[.!?]*[ \t]*")

OVERVIEW_LINE = "Overview"
STANDARDS_HEADING = "Education Standards"
STANDARDS_DOMAIN = regex.compile(r"(?:Learning|Science) Domain:.*")
# These lines belong to the alignment block only directly after a domain line; elsewhere they can be lesson text.
# The code / framework / grade triple is the Wyoming layout, for example "WY.SCI.5.LS1.1", "... Standards", "Grade 5".
STANDARDS_DETAIL = regex.compile(
    r"(?:Standard|Cluster|Topic):.*|[A-Z]{2,}(?:\.[A-Za-z0-9-]+){2,}|.{0,80} Standards|Grades? [K\d][\d -]{0,8}"
)
# Findings "HTML residue": CC badge RDFa and Freepik credit anchors pasted as raw markup. A tag counts only with an
# attribute or as a closing tag, so lesson text about "<img>" or the vector "<a,b,c>" stays.
HTML_MARKUP = regex.compile(r"<(?:a|p|span|img|div)\s+[\w:-]+=|</(?:a|p|span|div)>|<br\s*/?>")
HTML_TAG = regex.compile(r"</?(?:a|p|span|img|br|div)\b[^<>]*>|\bahref=\"[^\"]*\">")
# A tag split over lines leaves a line with only its start or only its attributes.
HTML_TAG_FRAGMENT = regex.compile(r"\s*(?:<(?:a|p|span|img|div)\b[^<>]*|(?:[\w:-]+=\"[^\"]*\"\s*)+/?>)\s*")

# Findings "licence/attribution boilerplate": the licence is kept in metadata.license (LITERATURE §f).
ATTRIBUTION_HEADING = regex.compile(
    r"(?i)(?:(?:CC|Creative Commons) licensed content(?:,.*)?|Public domain content|All rights reserved content"
    r"|Licen[cs](?:es?|ing) (?:and|&) Attributions?|Attributions? (?:and|&) Licen[cs](?:es?|ing)"
    r"|At+r+ibutions?|Licen[cs]es?|Licensing|Image (?:credits?|attributions?)|Title image"
    r"|(?:\w+ ){0,3}(?:adapted|modified|remixed) from):?"
)
# Lines under an attribution heading are credits while they hold a link or a licence word (read on 100 real pages);
# a line of two or more sentences without a link is prose, such as a lesson on how to write attributions.
SENTENCE_BREAK = regex.compile(r"[.!?][\"”’)]?\s+[A-Z]")
LINK = regex.compile(r"https?://|www\.")
SECTION_CREDIT_HINT = regex.compile(
    r"(?i)https?://|www\.|\blicen[cs]e|creative ?commons|\bCC[- ]?(?:BY|0)\b|public domain|wikimedia|adapted from"
    r"|title image|provided by|authored by|located at|courtesy of|^-? ?(?:source|location|attribution):"
)
# A licence sentence counts as a credit only when it names its work by a quote, a "by" author or "this/the/all".
CREDITED_WORK = r"(?:[\"“'].*|.*\bby\b.*|(?i:(?:except where otherwise noted,? )?(?:this|the|all)\b).*)"
ATTRIBUTION_CREDIT = regex.compile(
    r"(?:.*\b(?:Authored by|Provided by|Located at):.*\bLicen[cs]e\b.*"
    r"|.*\bLicense: (?:CC|Public Domain|All Rights Reserved|Other)\b.*"
    r"|(?i:images? (?:courtesy of|from|credits?:?)\b.*)"
    r"|(?i:title image\b.*|(?:\w+ )?(?:photo|image|picture)s? (?:by|credits?:?) .*)"
    r"|(?i:access (?:for )?free at:? https?://\S+)"
    r"|(?:Content on this page|This (?:page|section|chapter)) is (?:a derivative|adapted|remixed) (?:of|from) .*\blicensed\b.*"
    rf"|{CREDITED_WORK}\b(?:is|are) (?:licensed|available|shared|released) under (?:a |the )?"
    r"(?:Creative Commons|CC[- ]|Public Domain).*"
    rf"|{CREDITED_WORK}\b(?:is|are) (?:in the Public Domain|marked with (?:CC0|(?:a )?Public Domain Mark)).*"
    r"|.*,? via Wikimedia Commons\.?"
    r"|.*\|\s*CC[- ]BY(?:-[A-Z]{2})*(?: \d\.\d)?\s*)"
)
# Credit lines are short; a longer line that matches is prose about licensing, which OER training material teaches.
MAX_CREDIT_LINE_CHARS = 400
# Site promos and disclaimers repeated across pages (Common Pile line_tagger method, LITERATURE §0); each in 3-35 pages.
BOILERPLATE_LINE = regex.compile(
    r"Did you have an idea for improving this content\? We’d love your input\."
    r"|Did you know that you can access the complete collection of Pathways Project .*"
    r"|Please Note: Many of our activities were created by upper-division students at Boise State University.*"
    r"|This (?:resource|document) (?:may )?contains? links to websites operated by third parties\..*"
    r"|\|? ?Scroll down to (?:continue to the next section|get started with the first section)! ?\|?"
)
# Canvas LMS screen-reader text after each external link; counted 112 times in 27 cleaned pages.
EXTERNAL_LINK_NOTE = regex.compile(r" ?Links to an external site\.")
# Findings "table residue": empty layout-table rows such as "| ||||||" carry no content.
TABLE_RESIDUE = regex.compile(r"[\s|]*\|[\s|]*")
TABLE_SEPARATOR = regex.compile(r"\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*")
# Findings "URLs": Drive/Canva/YouTube links on their own line carry no learnable text.
URL_ONLY = regex.compile(r"(?:[-*•]\s*)?(?:https?://|www\.)\S+")
# A URL after "Retrieved from" completes an APA reference; without it the reference breaks (read on real pages).
CITATION_LEAD = regex.compile(r"(?i).*\b(?:retrieved|available)\b[\w ,]{0,40}\b(?:from|at):?")
TRACKING_QUERY = regex.compile(r"[?&](?:utm_[a-z]+|usp|fbclid|gclid|ref)=[^\s&#)\]]*")
# Findings "lost math and figure references": OpenStax figure numbers were stripped, leaving "(Figure)".
STRIPPED_FIGURE_REFERENCE = regex.compile(r" ?\((?:Figure|Table)\)")
# Findings "repeated lines": the page prints the summary and whole blocks twice. A run of repeated lines shorter
# than this is kept, because outline templates and per-section headings repeat on purpose (read on real pages).
MIN_REPEATED_BLOCK_CHARS = 100

# Gopher (arXiv:2112.11446, App. A) keeps documents of at least 50 words; findings: stubs hold only a title.
MIN_WORDS = 50
# Gopher (arXiv:2112.11446, App. A) requires 2 of these 8 words; on real pages it flags only code and standard lists.
GOPHER_FUNCTION_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})
MIN_FUNCTION_WORDS = 2
WORD = regex.compile(r"\w+")
# FineWeb (arXiv:2406.17557 §3.6) drops a page when at least 0.67 of its lines are shorter than 30 chars. On this
# source that rule alone also dropped prose with long lines among many short ones (the Declaration of Independence
# with its signer list, an OpenStax introduction), so a page goes only when the short lines also hold half its words.
SHORT_LINE_CHARS = 30
MAX_SHORT_LINE_SHARE = 0.67
MAX_SHORT_LINE_WORD_SHARE = 0.5
# Calibrated on all 5,094 distinct pages that reach this step: no page lies between 0.30 and 0.40. Below are pages
# taught in Spanish, Turkish or Filipino; from 0.40 up are English-medium language lessons (Dolma's 0.5, LITERATURE §e,
# would drop English-medium French lessons at 0.40-0.47).
ENGLISH_MIN_SHARE = 0.35
# The common NANP rule skips a number next to a digit ("Room G19, (919) 530-7492", "714-446-5053 800-543-8312") as a
# possible table cell; an area code in parentheses or with a dash, and a dash before the line number, is not a cell.
DASHED_PHONE = regex.compile(r"(?<![\w.+/-])(?:\([2-9]\d{2}\) ?|[2-9]\d{2}-)[2-9]\d{2}-\d{4}(?![\w-]|\.\d)")


def strip_vendor_promo(text: str) -> str:
    if not VENDOR_NAME.search(text):
        return text
    kept = []
    for line in text.split("\n"):
        if VENDOR_NAME.search(line):
            line = VENDOR_SENTENCE.sub("", VENDOR_TITLE_SUFFIX.sub("", line)).rstrip()
            if not line.strip():
                continue
        kept.append(line)
    return "\n".join(kept)


def strip_page_scaffold(text: str) -> str:
    """Removes the OER Commons page frame: tag/licence sidebar, attachment names and standards before the title,
    the "Overview" label after it, and standards-alignment blocks."""
    lines = text.split("\n")
    if OVERVIEW_LINE in lines:
        overview_index = lines.index(OVERVIEW_LINE)
        title = [lines[overview_index - 1]] if overview_index > 0 else []
        lines = title + lines[overview_index + 1:]
    kept = []
    in_standards_block = False
    for line in lines:
        if line == STANDARDS_HEADING or STANDARDS_DOMAIN.fullmatch(line):
            in_standards_block = True
            continue
        if in_standards_block and STANDARDS_DETAIL.fullmatch(line):
            continue
        in_standards_block = False
        kept.append(line)
    return "\n".join(kept)


def is_boilerplate_line(line: str, previous_line: str) -> bool:
    stripped = line.strip()
    if BOILERPLATE_LINE.fullmatch(stripped) or ATTRIBUTION_HEADING.fullmatch(stripped):
        return True
    if TABLE_RESIDUE.fullmatch(stripped):
        return True
    if URL_ONLY.fullmatch(stripped):
        return not CITATION_LEAD.fullmatch(previous_line.strip())
    return is_credit_line(stripped)


def is_credit_line(stripped: str) -> bool:
    return len(stripped) <= MAX_CREDIT_LINE_CHARS and bool(ATTRIBUTION_CREDIT.fullmatch(stripped))


def is_section_credit_line(stripped: str) -> bool:
    if len(stripped) > MAX_CREDIT_LINE_CHARS or not SECTION_CREDIT_HINT.search(stripped):
        return False
    return bool(LINK.search(stripped)) or not SENTENCE_BREAK.search(stripped)


def is_orphan_table_separator(line: str, previous_line: str) -> bool:
    return bool(TABLE_SEPARATOR.fullmatch(line.strip())) and not previous_line.strip().startswith("|")


def strip_html_markup(line: str) -> str:
    if HTML_TAG_FRAGMENT.fullmatch(line):
        return ""
    if not HTML_MARKUP.search(line):
        return line
    return html.unescape(HTML_TAG.sub("", line))


def strip_boilerplate(text: str) -> str:
    kept = []
    in_attribution_section = False
    for line in text.split("\n"):
        if line.strip():
            line = strip_html_markup(line)
            if not line.strip():
                continue
        stripped = line.strip()
        previous_line = kept[-1] if kept else ""
        if in_attribution_section and stripped and is_section_credit_line(stripped):
            continue
        if stripped:
            in_attribution_section = bool(ATTRIBUTION_HEADING.fullmatch(stripped))
        if is_boilerplate_line(line, previous_line) or is_orphan_table_separator(line, previous_line):
            continue
        line = EXTERNAL_LINK_NOTE.sub("", TRACKING_QUERY.sub("", line))
        kept.append(STRIPPED_FIGURE_REFERENCE.sub("", line))
    return "\n".join(kept)


def drop_repeated_blocks(text: str) -> str:
    """Removes each run of consecutive lines that all occurred earlier in the text, when the run has at least
    MIN_REPEATED_BLOCK_CHARS chars; table rows never count as repeats."""
    lines = text.split("\n")
    seen = set()
    is_repeat = []
    for line in lines:
        key = line.strip()
        # Table rows repeat on purpose (answer options of each question) and often have no leading "|".
        is_repeat.append(bool(key) and key in seen and "|" not in key)
        seen.add(key)
    kept = []
    start = 0
    while start < len(lines):
        stop = start + 1
        while stop < len(lines) and is_repeat[stop] == is_repeat[start]:
            stop += 1
        run = lines[start:stop]
        if not (is_repeat[start] and sum(len(line) for line in run) >= MIN_REPEATED_BLOCK_CHARS):
            kept.extend(run)
        start = stop
    return "\n".join(kept)


def has_min_words(document: dict) -> bool:
    return len(document["text"].split()) >= MIN_WORDS


def is_list_like(document: dict) -> bool:
    lines = [line.strip() for line in document["text"].split("\n") if line.strip()]
    short_lines = [line for line in lines if len(line) < SHORT_LINE_CHARS]
    if len(short_lines) < MAX_SHORT_LINE_SHARE * len(lines):
        return False
    short_line_words = sum(len(line.split()) for line in short_lines)
    return short_line_words >= MAX_SHORT_LINE_WORD_SHARE * len(document["text"].split())


def has_function_words(document: dict) -> bool:
    words = set(WORD.findall(document["text"].lower()))
    return len(GOPHER_FUNCTION_WORDS & words) >= MIN_FUNCTION_WORDS


def shingles_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".shingles.u64")


def shingle_counts_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".shingle_counts.npy")


class ShingleIndexer(MinHashIndexer):
    """MinHash band keys, plus each document's exact shingle set for the Jaccard check of VerifiedNearDedup."""

    def __init__(self, hasher, directory: Path, shard_index: int):
        super().__init__(hasher, directory, shard_index)
        prefix = shard_prefix(directory, shard_index)
        self.shingle_path = shingles_path(prefix)
        self.counts_path = shingle_counts_path(prefix)
        self.shingle_file = open(temp_path_for(self.shingle_path), "wb")
        self.shingle_counts = []

    def add(self, local_id: int, document: dict) -> None:
        super().add(local_id, document)
        words = dedup_words(document["text"])
        shingles = self.hasher.shingles(words) if words else np.empty(0, dtype=np.uint64)
        self.shingle_file.write(shingles.tobytes())
        self.shingle_counts.append(len(shingles))

    def close(self) -> None:
        self.shingle_file.flush()
        os.fsync(self.shingle_file.fileno())
        self.shingle_file.close()
        replace_durably(temp_path_for(self.shingle_path), self.shingle_path)
        save_array_atomically(self.counts_path, np.array(self.shingle_counts, dtype=np.int64))
        super().close()


class ShingleStore:
    """Reads the sorted shingle set of a document by global id from the per-shard files on disk."""

    def __init__(self, directory: Path, shard_counts: list[int]):
        self.offsets = shard_offsets(shard_counts)
        self.parts = {}
        for shard_index, count in enumerate(shard_counts):
            if not count:
                continue
            prefix = shard_prefix(directory, shard_index)
            starts = np.concatenate([[0], np.cumsum(np.load(shingle_counts_path(prefix)))])
            # np.memmap cannot map an empty file.
            shingles = (np.memmap(shingles_path(prefix), dtype=np.uint64, mode="r") if starts[-1]
                        else np.empty(0, dtype=np.uint64))
            self.parts[shard_index] = (starts, shingles)

    def get(self, global_id: int) -> np.ndarray:
        shard_index = int(np.searchsorted(self.offsets, global_id, side="right") - 1)
        starts, shingles = self.parts[shard_index]
        local_id = global_id - int(self.offsets[shard_index])
        return np.asarray(shingles[starts[local_id]:starts[local_id + 1]])


def exact_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    shared = np.intersect1d(first, second, assume_unique=True).size
    union = first.size + second.size - shared
    return shared / union if union else 0.0


def key_group_pairs(keys: np.ndarray, global_ids: np.ndarray) -> set[tuple[int, int]]:
    """Every pair (smaller id, larger id) of documents that share a band key."""
    order = np.lexsort((global_ids, keys))
    keys, global_ids = keys[order], global_ids[order]
    group_starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1], [True]]))
    pairs = set()
    for start, stop in zip(group_starts[:-1], group_starts[1:]):
        if stop - start < 2:
            continue
        # Known limit: a key group of n documents gives n(n-1)/2 pairs; all groups of this source give 285 pairs.
        members = np.unique(global_ids[start:stop]).tolist()
        pairs.update((members[first], members[second])
                     for first in range(len(members)) for second in range(first + 1, len(members)))
    return pairs


def connected_groups(edges: list[tuple[int, int]]) -> list[list[int]]:
    if not edges:
        return []
    nodes, inverse = np.unique(np.array(edges, dtype=np.int64).reshape(-1), return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (inverse[:, 0], inverse[:, 1])),
                       shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph, directed=False)
    return [nodes[labels == label].tolist() for label in np.unique(labels)]


def resolve_component(members: list[int], shingles_of, min_jaccard: float) -> dict[int, int]:
    """Maps each removed member to its kept copy. Members are visited from the most shingles down (ties by id), and
    a member is removed only when its exact Jaccard with an already kept member reaches min_jaccard, so the longer
    version stays and no chain of similar pairs removes a document that is below the threshold to every kept one."""
    order = sorted(members, key=lambda member: (-len(shingles_of(member)), member))
    kept = []
    duplicate_of = {}
    for member in order:
        member_shingles = shingles_of(member)
        match = next((kept_member for kept_member in kept
                      if exact_jaccard(shingles_of(kept_member), member_shingles) >= min_jaccard), None)
        if match is None:
            kept.append(member)
        else:
            duplicate_of[member] = match
    return duplicate_of


class VerifiedNearDedup(NearDedup):
    """MinHash-LSH candidates (shared band key), decided by exact word 5-shingle Jaccard >= min_jaccard instead of
    the signature estimate; findings: the estimate removed 9 of 131 pages below 0.8 and dropped 50 longer versions."""

    def open_indexer(self, directory: Path, shard_index: int) -> ShingleIndexer:
        return ShingleIndexer(self.hasher, directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        store = ShingleStore(directory, shard_counts)
        candidates = set()
        for keys, global_ids in iter_key_buckets(directory, shard_counts, self.max_bucket_entries):
            if dropped is not None:
                live = ~dropped[global_ids]
                keys, global_ids = keys[live], global_ids[live]
            candidates.update(key_group_pairs(keys, global_ids))
        edges = [(first, second) for first, second in sorted(candidates)
                 if exact_jaccard(store.get(first), store.get(second)) >= self.params.min_jaccard]
        with DuplicateOfWriter(directory, sum(shard_counts)) as duplicate_of:
            for component in connected_groups(edges):
                for member, kept_member in resolve_component(component, store.get, self.params.min_jaccard).items():
                    duplicate_of[member] = kept_member
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "candidate_pairs": len(candidates), "verified_pairs": len(edges)}


def mask_contact_details(text: str) -> str:
    return DASHED_PHONE.sub(PHONE_TOKEN, mask_pii(text))


def build_pipeline() -> Pipeline:
    language_identifier = LanguageIdentifier()
    return Pipeline("oercommons", [
        TextMap("normalize", normalize_text),
        Filter("login_chrome", lambda document: not LOGIN_CHROME.search(document["text"])),
        Filter("spam_pages", lambda document: document["id"] not in SPAM_PAGE_IDS),
        TextMap("vendor_promo", strip_vendor_promo),
        TextMap("page_scaffold", strip_page_scaffold),
        TextMap("boilerplate_lines", strip_boilerplate),
        TextMap("repeated_blocks", drop_repeated_blocks),
        Filter("min_words", has_min_words),
        Filter("short_lines", lambda document: not is_list_like(document)),
        Filter("english", lambda document: language_identifier.is_english(document["text"], ENGLISH_MIN_SHARE)),
        # The function-word rule is defined for English text; Gopher applies its quality rules after language ID.
        Filter("function_words", has_function_words),
        ExactDedup("exact_dedup"),
        VerifiedNearDedup("near_dedup"),
        TextMap("pii", mask_contact_details),
    ])


if __name__ == "__main__":
    run_cli(build_pipeline)
