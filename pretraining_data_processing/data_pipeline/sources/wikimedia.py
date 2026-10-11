"""Cleaning pipeline for Common Pile wikimedia (wtf_wikipedia plain text of eight English Wikimedia wikis).

python -m data_pipeline.sources.wikimedia --input DIR (--in-place | --sample-out DIR --limit N) --meta-out DIR
"""
import html
import multiprocessing
import os
import bisect
import shutil
import sqlite3
import statistics
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import orjson
import regex

from ..common.cli import run_cli
from ..common.dedup import ExactDedup, ExactIndexer, NearDedup, exact_key
from ..common.io import list_shards, read_documents, write_bytes_atomically
from ..common.langid import LanguageIdentifier
from ..common.pii import mask_pii
from ..common.pipeline import Filter, Pipeline, Step, TextMap
from ..common.text import normalize_text

WORDLIST_PATH = Path("/usr/share/dict/american-english")

# Findings "non-article namespaces": talk, user, project, template, draft and similar pages are discussion or code.
WIKIBOOKS = "wikibooks.com"
WIKISOURCE = "wikisource.com"
CONTENT_NAMESPACES = {
    WIKIBOOKS: frozenset({"0", "102", "110"}),  # Cookbook, Wikijunior
    WIKISOURCE: frozenset({"0", "104", "114"}),  # Page (scanned books), Translation
}
# Sample: "Cookbook:Policy/Recipe template" is an editing guideline in the Cookbook namespace.
EDITORIAL_TITLE_PREFIXES = ("Cookbook:Policy",)
ARTICLE_NAMESPACES = frozenset({"0"})
PAGE_NAMESPACE = "104"
# Dev check: 16,362 Page docs end in U+FFFD, all bot-imported raw OCR (Statutes at Large); no proofread book does.
RAW_OCR_END = "\ufffd"

# Raw OCR of long-s print reads ſ as f; proofreaders type ſ or s, so these non-words mark unproofread text.
LONG_S_MISREADING = regex.compile(
    r"\b(?:fuch|faid|fhall|fhould|thefe|thofe|firft|moft|alfo|fome|himfelf|herfelf|itfelf|themfelves|muft|fince|"
    r"juft|laft|leaft|becaufe|houfe|ufe|ufed|feveral|fubject|prefent|perfon|perfons|fuppofe|reafon|cafe|purpofe|"
    r"poffible|neceffary|bufinefs|defire|fecond|fervice)\b")
# OCR-only shapes: punctuation inside a lowercase word ("wa.s", "an<l"), a capital inside one ("wTitten", "CathoUc")
# and stray marks next to letters ("\\rabia", "■nith"); foreign words and transliterations rarely have them.
# Commas, colons and "(" are left out: early printing sets "greater,then", phrasebooks write "ju:s", formulas "sin(x)".
OCR_DAMAGE = regex.compile(
    r"\p{Ll}{2}[.;<>)\\^■]\p{Ll}|\p{Ll}[.;<>)\\^■]\p{Ll}{2}"
    r"|\p{Ll}(?<!(?<!\S)(?:Mc|Mac|De|Da|Di|Du|La|Le|Van|Von|St|Fitz))\p{Lu}\p{Ll}|[■^\\]\p{L}|\p{L}[■^\\]")
WHITESPACE = regex.compile(r"\s")
# Reverse search from the match, so the cost is the length of the word, not of the text before it.
LAST_WHITESPACE = regex.compile(r"(?r)\s")
TOKEN = regex.compile(r"\S+")
# Product and project names repeat one camel-case spelling ("PlayStation", "UpCodes"); OCR errors vary.
MAX_COUNT_PER_SPELLING = 2
# Units ("mL"), formulas, table pipes, links, e-mail addresses and file names have such shapes without OCR errors.
NOT_PROSE_TOKEN = regex.compile(
    r"(?<!\S)(?:[^\s/@|$_{}]*+[/@|$_{}]\S*|\S+\.(?:jpe?g|png|gif|svg|ogg|pdf|djvu|tiff?|webm|html?|php|com|org|net)(?!\w))",
    regex.IGNORECASE)
PER_MILLE = 1000
# Calibration on all 23,752 assembled Page books, read by band: every book at 12 or more long-s misreadings per 1,000
# words was raw long-s OCR, books at 2-8 were mostly clean; at 8 or more damaged words per 1,000 the books read as
# garbage, while 4-8 was legible OCR with about 1% errors (Catholic Encyclopedia vol 2, Decline and Fall).
MAX_LONG_S_PER_MILLE = 8
MAX_DAMAGED_WORDS_PER_MILLE = 8
# Short pages reach the rates by chance (a Latvian phrasebook: 19 "ju:s" marks in 1,000 words).
MIN_OCR_EVIDENCE_WORDS = 20
# About one printed page.
OCR_WINDOW_WORDS = 500
# Calibration, the 661 Page books with any damage read by band: at a median of 4 or more nearly all were unproofread
# scans (Catholic Encyclopedia vols 2, 4-8, 10-15, Statutes at Large); at 2, proofread grammars mix with raw OCR.
MIN_MEDIAN_DAMAGED_WORDS_PER_MILLE = 4
PAGE_TITLE = regex.compile(r"\APage:(?P<book>.+)/(?P<number>\d+)\Z")
SCAN_EXTENSION = regex.compile(r"\.(?:djvu|pdf|tiff?|jpe?g|png|webm|ogv)\Z", regex.IGNORECASE)
ARCHIVE_ID = regex.compile(r"\s*\(IA [^()]*\)\s*\Z")
# The {{rh}} running-header template leaves its name where the header was (findings 104-2356556 "Rh starry heavens").
RUNNING_HEADER_RESIDUE = regex.compile(r"\A\s*\(?R[hv]\)?(?=\s)|(?<=\s)\(?R[hv]\)?\s*\Z|(?<!\S)(?:R[hv]\s+){2,}")
LEADING_NUMBER = regex.compile(r"\A\s*(\d{1,4})(?=\s)")
TRAILING_NUMBER = regex.compile(r"(?<=\s)(\d{1,4})\s*\Z")
MIN_PAGE_NUMBER_AGREEMENT = 3
SENTENCE_END = (".", "?", "!", ":", ";", '"', "”", "’", "'", ")", "]")
# Reverse search starts at the end of the page, so the cost does not grow with the page length.
HYPHENATED_END = regex.compile(r"(?r)(\p{L}+)-\Z")
FIRST_WORD = regex.compile(r"\A(\p{Ll}+)")
BOOK_STORE_DIR = "wikisource_books"
STORE_COMPLETE = "complete.json"
SQLITE_BATCH = 4096
ZLIB_LEVEL = 3

MATH_SPAN = regex.compile(r"\$\$.+?\$\$|(?<![\w$])\$[^$\n]{1,400}?\$(?![\w$])", regex.DOTALL)
FILE_LINK = regex.compile(r"\[\[(?:File|Image|Media):[^\[\]\n]*(?:\[\[[^\[\]\n]*\]\][^\[\]\n]*)*(?:\]\]|$)",
                          regex.IGNORECASE | regex.MULTILINE)
IMAGE_OPTIONS = regex.compile(r"\|\s*(?:thumb|thumbnail|frameless|framed|upright(?:=[\d.]+)?|left|right|center|"
                              r"\d+(?:x\d+)?px)\s*(?=\||\]\])", regex.IGNORECASE)
WIKILINK = regex.compile(r"\[\[(?:[^\[\]|\n]{0,200}\|)?([^\[\]|\n]{0,200})\]\]")
TEMPLATE = regex.compile(r"\{\{[^{}\n]{0,300}\}\}")
MATH_MASK = regex.compile(r"\x00(\d+)\x00")
# Templates whose argument is the quoted text itself (sample: {{blockquote|text=$$...$$}} lost a formula).
CONTENT_TEMPLATE_START = r"\{\{\s*(?:blockquote|quote|quotation|quote box|poem|center|centre|block center)\s*\|"
CONTENT_TEMPLATE = regex.compile(
    CONTENT_TEMPLATE_START
    + r"(?P<arguments>(?:[^{}]++|\{(?!\{)|\}(?!\})|(?P<nested>\{\{(?:[^{}]++|\{(?!\{)|\}(?!\})|(?&nested))*\}\}))*)"
    r"\}\}",
    regex.IGNORECASE,
)
# A longer span is an unclosed template that reached the "}}" of an unrelated one.
MAX_CONTENT_TEMPLATE_CHARS = 20_000
TEMPLATE_PARAMETER = regex.compile(r"\A\s*(?P<name>[\w -]{1,30}?)\s*=(?P<value>.*)\Z", regex.DOTALL)
CONTENT_PARAMETERS = frozenset({"text", "quote", "1", "content"})
UNCLOSED_CONTENT_TEMPLATE = regex.compile(
    CONTENT_TEMPLATE_START + r"(?:\s*(?:" + "|".join(sorted(CONTENT_PARAMETERS)) + r")\s*=)?", regex.IGNORECASE)
# Sample: "{{AuxTOC|width=100%|" opens a template whose "}}" is on a later line; named parameters only, no text.
TEMPLATE_START_LINE = regex.compile(r"\A\s*\{\{[^{}|\n]{1,60}(?:\|[^{}|=\n]{1,30}=[^{}|\n]*)*\|?\s*\Z")
TEMPLATE_END_LINE = regex.compile(r"\A\s*\}\}\s*\Z")
# Sample: "days ago|two days after tomorrow}}" is the end of a template whose start wtf_wikipedia dropped.
STRAY_TEMPLATE_END = regex.compile(r"\}\}\s*\Z")
# The closing bracket is missing in some sources ("[https://www.youtube.com/watch?v=IwQMpSxRPvc Video at YouTube").
LABELLED_EXTERNAL_LINK = regex.compile(
    r"\[(?:https?://|ftp://|mailto:)[^\s\[\]]+[ \t]+([^\[\]\n]+)(?:\]|(?=\n|\[|\Z))", regex.IGNORECASE)
CALENDAR_LINK_NOTE = " (link will display the full calendar)"
TABLE_LINE = regex.compile(r"\A\s*(?:\*\s*)?(?:\{\||\|\}|\|-|\|\+)")
HTML_ATTRIBUTE = regex.compile(
    r"\b(?:style|class|align|valign|width|height|bgcolor|border|cellpadding|cellspacing|colspan|rowspan|scope|"
    r"id|lang|dir|title)\s*=\s*(?:\"[^\"\n]*\"|'[^'\n]*')\s*\|?",
    regex.IGNORECASE,
)
ATTRIBUTE_LINE = regex.compile(
    r"\A[\s*|!-]*(?:(?:style|class|align|valign|width|height|bgcolor|border|cellpadding|cellspacing|colspan|rowspan|"
    r"scope)\s*=\s*\S+[\s|]*)+\Z",
    regex.IGNORECASE,
)
HTML_TAG = regex.compile(
    r"</?(?:hr|br|span|div|li|ul|ol|sup|sub|small|big|center|font|p|b|i|u|s|tt|blockquote|poem|ref|references|"
    r"table|tr|td|th|abbr|cite|code|nowiki|gallery|includeonly|noinclude|onlyinclude|section|syntaxhighlight|source|"
    r"pre|kbd|var|del|ins|strike|mark|caption|dl|dt|dd|score|timeline|imagemap|templatestyles|categorytree|"
    r"chem|ce|math|mapframe|maplink|graph|inputbox)\b[^<>\n]{0,300}?/?>",
    regex.IGNORECASE,
)
BOLD_QUOTES = regex.compile(r"'''(?:'')?")
ITALIC_QUOTES = regex.compile(r"''([^'\n]+?)''")
MAGIC_WORD = regex.compile(r"__[A-Z]{3,20}__")
CATEGORY_LINE = regex.compile(r"\A\s*\[\[Category:[^\]\n]*\]\]\s*\Z", regex.IGNORECASE)
# Residue of nested or unclosed markup that strip_markup leaves (0.5% of output docs).
# An orphan link keeps its label when its target has a left boundary: a '' or [ before it, or a title of capitalised words
# and connectors with an anchor or namespace ("…Constitution#Amendment XV|Fifteenth Amendment]]"); a target without either
# cannot be told apart from the prose before it.
TITLE_WORD = r"(?:\p{Lu}[^\s\[\]|#:]*|of|the|to|and|in|on|for|de|du|la|von|van)"
ORPHAN_LINK = regex.compile(
    r"(?:(?:\[\[?|'')[^\[\]|\n']{1,100}|\b\p{Lu}[^\s\[\]|#:]*(?: " + TITLE_WORD + r"){0,8}[#:][^\[\]|\n]{1,60})"
    r"\|([^\[\]|\n]{1,120})\]\]")
OPENED_LINK = regex.compile(r"\[\[([^\[\]|\n]{1,120}\|)?([^\[\]|\n]{1,200})\]( ?)\]")
# "{w|neuroplasticity|plasticity}}": a link template whose first brace was lost keeps its label.
BROKEN_LINK_TEMPLATE = regex.compile(r"\{(?!\{)\w{1,10}\|(?:[^{}|\n]{0,80}\|)?([^{}|\n]{0,80})\}\}")
QUOTE_TEMPLATE = regex.compile(r"\{\{\s*quote-[\w-]+\|[^\n]*")
QUOTE_PASSAGE = regex.compile(r"\|\s*passage\s*=\s*([^|{}\n]*)")
TEMPLATE_OPENER = regex.compile(r"\{\{\s*[\w#][\w# -]{0,30}\|(?:[\w.%]{1,8}\|)?")
BRACKET_RUN = regex.compile(r"\{\{|\}\}|\[\[|\]\]")
LINK_BRACKETS = regex.compile(r"\[\[|\]\]")
CURLY_BRACKETS = regex.compile(r"\{\{|\}\}")
# Sample: an unpaired $$ leaves LaTeX unmasked ("\frac{1 + 2z^{-1} +z^{-2}}"), so a line with LaTeX keeps its braces.
LATEX_HINT = regex.compile(r"\\[A-Za-z]+|[_^]\{")
# Output audit: gallery entries survive as plain text ("File:Parrots.jpg|Two parrots File:Owl.png|An owl").
# The lookbehind keeps Commons URLs ("wiki/File:Scan.pdf"); the extension keeps prose ("an image: an idol").
GALLERY_FILE = regex.compile(
    r"(?<![/\w])\[{0,2}(?:File|Image):[^|\n\[\]]{1,200}?\.(?:jpe?g|png|gif|svg|tiff?|webp|pdf|djvu|og[gv]|webm)"
    r"(?:\|(?:thumb(?:nail)?|frameless|framed?|border|upright(?:=[\d.]+)?|left|right|center|centre|none|"
    r"\d+(?:x\d+)?px|alt=[^|\n\]]*))*(?:\||\]\]|(?=\s|\Z))", regex.IGNORECASE)
# A capital word must follow a bare "<ref" opener, so XML code ("<ref bean=...>") keeps its tags.
REF_TAG = regex.compile(r"</?ref(?:\s+name\s*=[^<>\n]{0,120}?)?\s*/?>|<ref(?=\s+\p{Lu}\p{Ll}|\s*\Z)")
# Category names start with a letter or digit and hold no quotes or sentence marks (Wikisource prose can start "Category:").
BARE_CATEGORY_LINE = regex.compile(r"\A\s*Category:[\p{L}\p{N}][^\n\"“”?!]{0,150}\Z")
FILE_REF_HINT = regex.compile(r"(?i)File:|Image:|<ref|^\s*Category:", regex.MULTILINE)

# A space before the parenthesis separates template holes from code calls such as "main()".
EMPTY_PARENS = regex.compile(r"(?: |\A)\((?:\s|[,;:])*\)")
LEADING_PAREN_PUNCT = regex.compile(r"(?<= |\A)\(\s*(?:[,;:]\s*)+")
TRAILING_PAREN_PUNCT = regex.compile(r"(?:\s*[,;])+\s*\)")
EMPTY_CURLY_QUOTES = regex.compile(r" ?“”")
# Wiktionary citations lose their titles ('* 2006 Oct. 1,, "", ,'); elsewhere "" is mostly an empty string in code.
EMPTY_STRAIGHT_QUOTES = regex.compile(r" ?\"\"")
REPEATED_COMMA = regex.compile(r",(?:\s*,)+")
SPACE_BEFORE_PUNCT = regex.compile(r"(?<=[\p{L}\p{N})\]\"”’]) +([,.;])(?=\s|$)")
INNER_SPACES = regex.compile(r"(?<=\S) {2,}(?=\S)")
EMPTY_LIST_ITEM = regex.compile(r"\A\s*[*#]+[\s*#]*[\W_]*\Z")
# Wikivoyage phrasebook entries whose translation template was dropped: "* Hello. (informal) : . ( )".
EMPTY_PHRASEBOOK_LINE = regex.compile(r"\A ?\*[^:\n]{1,80} : [.?!]? ?\( ?[?.!]? ?\)\s*\Z")
# Wikivoyage listing names come from {{see|name=...}}; without them the item starts with the dash after the name.
NAMELESS_LISTING = regex.compile(r"\A\s*\*\s*[—–]\s")
LIST_NUMBER = regex.compile(r"\A\s*[*#]+\s*(?:\d+\)\s*)?")
MAX_REPEATED_RUN = 100
# Sample: repeated equation lines are steps of a derivation, so a list line with math is never removed.
LATEX_COMMAND = regex.compile(r"\\[A-Za-z]+")
# Pairs of '' on one line are already italics markup; a single '' left on a Wikiquote line opened a multi-line one.
UNPAIRED_ITALIC_QUOTES = regex.compile(r"(?<!')''(?!')")
# Findings "talk-page discussion": signed comments and sign-up lines leak into Wikiversity and Wikibooks pages.
SIGNATURE = regex.compile(r"\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")
GALLERY_FILE_LINE = regex.compile(r"\A\s*\*\s*[^\n]{1,200}\.(?:jpe?g|png|gif|svg|tiff?|webp)\s*\Z", regex.IGNORECASE)
# Wikibooks navigation: Cookbook category breadcrumbs ("| Vegan cuisine | Rice") and previous/next page links.
BREADCRUMB_LINE = regex.compile(r"\A\s*(?:\|\s*[^|\n]{1,40}?\s*)+\Z")
NAVIGATION_LINE = regex.compile(r"\A\s*(?:(?:[‹←]|<<?)\s.{0,100}|.{0,100}\s(?:[›→]|>>?))\s*\Z")
NAVIGATION_WIKIS = frozenset({WIKIBOOKS, "wikiversity.com"})
LINK_LIST_HEADINGS = frozenset({"sources", "source", "related news", "related stories", "external links",
                                "external link", "see also", "further reading", "links"})
HEADING_MAX_WORDS = 8
HEADING_MAX_CHARS = 80

# Interwiki links that wtf_wikipedia printed as text: titles of the page in other languages ("de:Mathematik").
INTERWIKI_PREFIX = regex.compile(r"(?<!\S)[a-z]{2,3}(?:-[a-z]{2,8})?:\S")
NON_LATIN_SCRIPT = regex.compile(r"[\p{Cyrillic}\p{Greek}\p{Hebrew}\p{Arabic}\p{Han}\p{Hangul}\p{Devanagari}\p{Thai}]")
MIN_RESIDUE_WORDS = 3
# Dev check, 437 removals in 24,000 Wikisource and Wikinews pages: titles reach 84 words; one 4,906-word line was not.
MAX_RESIDUE_WORDS = 120
TRANSLATED_WIKIS = frozenset({"wikinews.com", WIKISOURCE})
# Sample: "Piemont Autonomista/Lega Nord Piemont" ends a Wikipedia page; Wikibooks residue lists several book paths.
MIN_SLASHES_OUTSIDE_TRANSLATED_WIKIS = 2
# Sample: untagged translation titles ("Verklaring door Verenigde Naties") get 0.99 or more; the heading lid.176
# mislabels ("COLONEL DE CRAYE") gets 0.77.
MIN_TRANSLATED_TITLE_PROBABILITY = 0.9
WIKTIONARY = "wiktionary.com"
WIKIVOYAGE = "wikivoyage.com"
WIKIQUOTE = "wikiquote.com"
FLAG_EMOJI = regex.compile(r"[\U0001F1E6-\U0001F1FF]{2}")
EMPTY_LABEL_SLOT = regex.compile(r"\A ?\* [^:\n]{1,40}:[\s,;]*\Z")
WIKTIONARY_HEADINGS = frozenset({
    "etymology", "pronunciation", "noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction",
    "interjection", "determiner", "article", "numeral", "particle", "prefix", "suffix", "infix", "affix",
    "proper noun", "phrase", "proverb", "idiom", "abbreviation", "acronym", "initialism", "symbol", "letter",
    "contraction", "translations", "synonyms", "antonyms", "hypernyms", "hyponyms", "meronyms", "holonyms",
    "derived terms", "related terms", "descendants", "anagrams", "usage notes", "see also", "references",
    "further reading", "alternative forms", "coordinate terms", "han character", "kanji", "hanzi", "definitions",
    "readings", "glyph origin", "compounds", "related characters", "trivia", "statistics", "quotations",
    "conjugation", "declension", "inflection", "mutation", "hyphenation", "rhymes", "homophones", "external links",
})
ETYMOLOGY_CONNECTORS = frozenset({
    "from", "compare", "see", "also", "more", "at", "the", "and", "or", "of", "cognate", "with", "displaced",
    "native", "borrowed", "inherited", "derived", "learned", "semi", "calque", "doublet", "ultimately", "via",
    "to", "a", "an", "in", "by", "related", "akin", "possibly", "probably", "perhaps", "partly", "equivalent",
})
WORD = regex.compile(r"\p{L}+")

# Dolma (arXiv:2402.00159 App. N) removes Wikipedia pages of 25 words or fewer: short templated pages, parse errors.
MIN_BODY_WORDS = 26
DISAMBIGUATION = regex.compile(
    r"\b(?:may|can|might|could) (?:also )?(?:refer|stand) (?:to|for):"
    r"|There (?:is|are) more than one (?:place|article|page|person|destination)s? (?:called|named|titled)"
)
DISAMBIGUATION_SEARCH_CHARS = 600
# Gopher (arXiv:2112.11446 App. A) drops documents with more than 90% of lines starting with a bullet.
MAX_BULLET_LINE_FRACTION = 0.9
BULLET_LINE = regex.compile(r"\A\s*[*#•]")
# FineWeb (arXiv:2406.17557 §3.6) removes documents whose fraction of lines ending in punctuation is 0.12 or less.
MAX_PUNCTUATED_LINE_FRACTION = 0.12
PHRASEBOOK_TITLE = " phrasebook"
TERMINAL_PUNCTUATION = (".", "?", "!", '"', "”", "…")
# Gopher Table A1: duplicate line character fraction above 0.20 marks a repetitive document.
MAX_DUPLICATE_LINE_CHAR_FRACTION = 0.2
LATEX_SOURCE = regex.compile(r"\\documentclass|\\begin\{document\}|\\usepackage")
CODE_TITLE = regex.compile(r"\.(?:js|css|json|lua|py)\Z")

# ROOTS (arXiv:2303.03915 §3.1) closed-class word ratio; dev pages below 0.03 were indexes, move lists and code.
STOPWORDS = frozenset("the be to of and that have with a in is it for on as was by are this at from".split())
MIN_STOPWORD_FRACTION = 0.03
PROSE_WORD = regex.compile(r"\p{L}{2}")
LETTER = regex.compile(r"\p{L}")
# LID fails on short text (Caswell et al. arXiv:2010.14571 §3); pages with less prose are tables or formulas.
MIN_PROSE_LETTERS = 100
# Dev calibration (151,003 docs): below 0.1 almost all non-English; 0.1-0.2 is 62% English language-teaching pages.
ENGLISH_MIN_SHARE = 0.1
PII_KINDS = ("email", "ip", "ipv6", "phone")
# Sampled Wikivoyage phone matches were business listings (findings "PII redaction"); Wikinews ones named people.
# Content pages of books cite act.scene.line numbers ("3.4.19.24") that the IPv4 pattern masks.
PII_KINDS_BY_WIKI = {
    WIKIVOYAGE: ("email", "ip", "ipv6"),
    WIKIBOOKS: ("email", "ipv6", "phone"),
    WIKISOURCE: ("email", "ipv6", "phone"),
}


def metadata(document: dict) -> dict:
    return document.get("metadata") or {}


def wiki_of(document: dict) -> str:
    return metadata(document).get("wiki", "")


def namespace_of(document: dict) -> str:
    return str(metadata(document).get("namespace", ""))


def split_title(text: str) -> tuple[str, str]:
    """wtf_wikipedia text is the page title, a blank line, then the body."""
    title, _, body = text.partition("\n")
    return title, body.strip()


def body_of(text: str) -> str:
    return split_title(text)[1]


def is_content_page(document: dict) -> bool:
    if str(metadata(document).get("title", "")).startswith(EDITORIAL_TITLE_PREFIXES):
        return False
    return namespace_of(document) in CONTENT_NAMESPACES.get(wiki_of(document), ARTICLE_NAMESPACES)


def is_page_namespace(document: dict) -> bool:
    return wiki_of(document) == WIKISOURCE and namespace_of(document) == PAGE_NAMESPACE


def is_raw_ocr_page(document: dict) -> bool:
    return is_page_namespace(document) and document["text"].rstrip().endswith(RAW_OCR_END)


@lru_cache(maxsize=1)
def wordlist() -> frozenset[str]:
    return frozenset(word.strip().lower() for word in WORDLIST_PATH.read_text().split("\n") if word.strip())


def book_title(book: str) -> str:
    return ARCHIVE_ID.sub("", SCAN_EXTENSION.sub("", book)).strip()


def clean_page_body(body: str) -> str:
    return RUNNING_HEADER_RESIDUE.sub("", body).strip()


def dominant_offset(candidates: list[tuple[int, int]]) -> int | None:
    """Most common (scan number - printed number) over the pages; None when too few pages agree on one offset."""
    counts: dict[int, int] = {}
    for scan_number, printed in candidates:
        counts[scan_number - printed] = counts.get(scan_number - printed, 0) + 1
    if not counts:
        return None
    offset, count = max(counts.items(), key=lambda item: item[1])
    return offset if count >= MIN_PAGE_NUMBER_AGREEMENT else None


def strip_page_numbers(pages: list[tuple[int, str]]) -> list[str]:
    """Removes start or end numbers whose offset from the scan page number is the one most pages share (IB-ET §4.7)."""
    leading = [(number, int(match.group(1))) for number, page in pages if (match := LEADING_NUMBER.match(page))]
    trailing = [(number, int(match.group(1))) for number, page in pages if (match := TRAILING_NUMBER.search(page))]
    leading_offset, trailing_offset = dominant_offset(leading), dominant_offset(trailing)
    cleaned = []
    for number, page in pages:
        match = LEADING_NUMBER.match(page)
        if match and leading_offset is not None and number - int(match.group(1)) == leading_offset:
            page = page[match.end():]
        match = TRAILING_NUMBER.search(page)
        if match and trailing_offset is not None and number - int(match.group(1)) == trailing_offset:
            page = page[:match.start()]
        page = page.strip()
        if page:
            cleaned.append(page)
    return cleaned


def join_pages(pages: list[str]) -> str:
    """Joins pages in scan order; a page that does not end a sentence continues the paragraph on the next page."""
    if not pages:
        return ""
    parts = [pages[0]]
    for previous, page in zip(pages, pages[1:]):
        hyphenated = HYPHENATED_END.search(previous)
        first_word = FIRST_WORD.match(page)
        # IB-ET §4.5: 89% of end-of-line hyphens merge; the word list decides which ones.
        if hyphenated and first_word:
            if (hyphenated.group(1) + first_word.group(1)).lower() in wordlist():
                parts[-1] = parts[-1][:-1]
        elif previous.endswith(SENTENCE_END) and not page[:1].islower():
            parts.append("\n\n")
        else:
            parts.append(" ")
        parts.append(page)
    return "".join(parts)


def page_record(document: dict) -> tuple[int, str, int, str] | None:
    """(page id, book, page number, body) of a Page-namespace document that holds proofread text."""
    if not is_page_namespace(document) or is_raw_ocr_page(document):
        return None
    title, body = split_title(document["text"])
    match = PAGE_TITLE.match(title)
    body = clean_page_body(body)
    if match is None or not body:
        return None
    return int(document["id"].split("-", 1)[1]), match.group("book"), int(match.group("number")), body


def shard_wiki(shard: Path) -> str:
    for document in read_documents(shard):
        return wiki_of(document)
    return ""


def ocr_words(text: str) -> str:
    return NOT_PROSE_TOKEN.sub(" ", MATH_SPAN.sub(" ", body_of(text)))


def ocr_damage(text: str) -> tuple[int, int, int]:
    """(words, long-s misreadings, OCR-damaged words) of the body."""
    words = ocr_words(text)
    return len(words.split()), len(LONG_S_MISREADING.findall(words)), count_damaged_words(words)


def damaged_word_starts(words: str) -> list[int]:
    """Start offsets of words with an OCR_DAMAGE shape; each spelling counts at most MAX_COUNT_PER_SPELLING times."""
    spellings: dict[str, int] = {}
    starts = []
    previous_end = None
    for match in OCR_DAMAGE.finditer(words):
        if previous_end is not None and not WHITESPACE.search(words, previous_end, match.start()):
            previous_end = match.end()
            continue
        space = LAST_WHITESPACE.search(words, 0, match.start())
        start = 0 if space is None else space.end()
        spelling = words[start:match.end()]
        spellings[spelling] = spellings.get(spelling, 0) + 1
        if spellings[spelling] <= MAX_COUNT_PER_SPELLING:
            starts.append(start)
        previous_end = match.end()
    return starts


def count_damaged_words(words: str) -> int:
    return len(damaged_word_starts(words))


def median_window_damage(words: str, damaged_starts: list[int]) -> float:
    """Median damaged words per 1,000 over full OCR_WINDOW_WORDS windows; a few garbage pages do not move it."""
    token_starts = [match.start() for match in TOKEN.finditer(words)]
    windows = len(token_starts) // OCR_WINDOW_WORDS
    if windows == 0:
        return 0.0
    window_starts = token_starts[:windows * OCR_WINDOW_WORDS:OCR_WINDOW_WORDS]
    end = token_starts[windows * OCR_WINDOW_WORDS] if windows * OCR_WINDOW_WORDS < len(token_starts) else len(words)
    counts = [0] * windows
    for start in damaged_starts:
        if start < end:
            counts[bisect.bisect_right(window_starts, start) - 1] += 1
    return statistics.median(PER_MILLE * count / OCR_WINDOW_WORDS for count in counts)


def is_damaged_ocr(document: dict) -> bool:
    if wiki_of(document) != WIKISOURCE:
        return False
    words = ocr_words(document["text"])
    damaged_starts = damaged_word_starts(words)
    word_count, long_s, damaged = len(words.split()), len(LONG_S_MISREADING.findall(words)), len(damaged_starts)
    if any(count >= MIN_OCR_EVIDENCE_WORDS and PER_MILLE * count / word_count >= max_per_mille
           for count, max_per_mille in ((long_s, MAX_LONG_S_PER_MILLE), (damaged, MAX_DAMAGED_WORDS_PER_MILLE))):
        return True
    return (damaged >= MIN_OCR_EVIDENCE_WORDS
            and median_window_damage(words, damaged_starts) >= MIN_MEDIAN_DAMAGED_WORDS_PER_MILLE)


def page_store_path(store_dir: Path, shard_index: int) -> Path:
    return store_dir / f"pages_{shard_index:05d}.sqlite"


def index_shard_pages(job: tuple[Path, Path, int]) -> int:
    """Writes the proofread Page documents of one shard to a per-shard SQLite file; returns their count."""
    shard, store_dir, shard_index = job
    path = page_store_path(store_dir, shard_index)
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE pages (page_id INTEGER PRIMARY KEY, position INTEGER, book TEXT, "
                       "number INTEGER, body BLOB)")
    batch = []
    count = 0
    for position, document in enumerate(read_documents(shard)):
        record = page_record(document)
        if record is None:
            continue
        page_id, book, number, body = record
        batch.append((page_id, position, book, number, zlib.compress(body.encode("utf-8", "surrogatepass"),
                                                                       ZLIB_LEVEL)))
        if len(batch) == SQLITE_BATCH:
            connection.executemany("INSERT OR IGNORE INTO pages VALUES (?, ?, ?, ?, ?)", batch)
            count += len(batch)
            batch = []
    connection.executemany("INSERT OR IGNORE INTO pages VALUES (?, ?, ?, ?, ?)", batch)
    count += len(batch)
    connection.commit()
    connection.close()
    return count


def build_book_index(store_dir: Path, shard_indices: list[int]) -> dict:
    """Assigns every stored page to its book; the first page in shard order is the book's anchor document."""
    index_path = store_dir / "books.sqlite"
    index_path.unlink(missing_ok=True)
    index = sqlite3.connect(index_path)
    index.execute("PRAGMA journal_mode=OFF")
    index.execute("PRAGMA synchronous=OFF")
    index.execute("CREATE TABLE members (page_id INTEGER PRIMARY KEY, book_id INTEGER, number INTEGER, "
                  "shard INTEGER)")
    index.execute("CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT, anchor_page_id INTEGER)")
    book_ids: dict[str, int] = {}
    pages = 0
    for shard_index in shard_indices:
        connection = sqlite3.connect(page_store_path(store_dir, shard_index))
        rows = connection.execute("SELECT page_id, book, number FROM pages ORDER BY position")
        while batch := rows.fetchmany(SQLITE_BATCH):
            members = []
            for page_id, book, number in batch:
                if book not in book_ids:
                    book_ids[book] = len(book_ids)
                    index.execute("INSERT INTO books VALUES (?, ?, ?)", (book_ids[book], book_title(book), page_id))
                members.append((page_id, book_ids[book], number, shard_index))
            index.executemany("INSERT OR IGNORE INTO members VALUES (?, ?, ?, ?)", members)
            pages += len(members)
        connection.close()
    index.execute("CREATE INDEX members_by_book ON members (book_id, number)")
    index.commit()
    index.close()
    return {"books": len(book_ids), "pages": pages}


class BookAssembly(Step):
    """Puts each Wikisource book on its first page in shard order and drops its other pages (store built before the pass)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.store_dir: Path | None = None
        self._connections: dict[str, sqlite3.Connection] = {}
        self._pid = 0

    def connection(self, path: Path) -> sqlite3.Connection:
        # SQLite connections must not cross a fork, so every worker opens its own.
        if self._pid != os.getpid():
            self._connections = {}
            self._pid = os.getpid()
        key = str(path)
        if key not in self._connections:
            self._connections[key] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return self._connections[key]

    def book_text(self, book_id: int, title: str) -> str:
        index = self.connection(self.store_dir / "books.sqlite")
        members = index.execute("SELECT page_id, number, shard FROM members WHERE book_id = ? ORDER BY number",
                                (book_id,)).fetchall()
        pages = []
        for page_id, number, shard in members:
            (body,) = self.connection(page_store_path(self.store_dir, shard)).execute(
                "SELECT body FROM pages WHERE page_id = ?", (page_id,)).fetchone()
            pages.append((number, zlib.decompress(body).decode("utf-8", "surrogatepass")))
        return title + "\n\n" + join_pages(strip_page_numbers(pages))

    @staticmethod
    def single_page(document: dict) -> dict:
        """A Page document outside any book (one scan per page, e.g. "Page:NIE 1903 - v. 7, p. 884.jpg")."""
        title, body = split_title(document["text"])
        return {**document, "text": join_title(book_title(title.removeprefix("Page:")), clean_page_body(body))}

    def apply(self, document: dict) -> dict | None:
        if self.store_dir is None or not is_page_namespace(document):
            return document
        page_id = int(document["id"].split("-", 1)[1])
        index = self.connection(self.store_dir / "books.sqlite")
        found = index.execute("SELECT books.book_id, title, anchor_page_id FROM members JOIN books "
                              "ON members.book_id = books.book_id WHERE page_id = ?", (page_id,)).fetchone()
        if found is None:
            return self.single_page(document)
        book_id, title, anchor_page_id = found
        if anchor_page_id != page_id:
            return None
        return {**document, "text": self.book_text(book_id, title)}


def strip_markup(text: str) -> str:
    """Removes wikitext and HTML debris that wtf_wikipedia left; LaTeX math is masked, as its braces look like templates."""
    maths: list[str] = []

    def mask(match: regex.Match) -> str:
        maths.append(match.group(0))
        return f"\x00{len(maths) - 1}\x00"

    text = MATH_SPAN.sub(mask, text)
    text = HTML_TAG.sub("", text)
    text = html.unescape(text)
    text = FILE_LINK.sub("", text)
    text = IMAGE_OPTIONS.sub("", text)
    text = WIKILINK.sub(r"\1", text)
    text = LABELLED_EXTERNAL_LINK.sub(r"\1", text)
    text = text.replace(CALENDAR_LINK_NOTE, "")
    text = CONTENT_TEMPLATE.sub(unwrap_content_template, text)
    text = UNCLOSED_CONTENT_TEMPLATE.sub("", text)
    previous = None
    while previous != text:
        previous = text
        text = TEMPLATE.sub(lambda match: " ".join(MATH_MASK.findall(match.group(0))), text)
    text = MAGIC_WORD.sub("", text)
    text = BOLD_QUOTES.sub("", text)
    text = ITALIC_QUOTES.sub(r"\1", text)
    kept = []
    for line in drop_template_residue(text.split("\n")):
        if TABLE_LINE.match(line) or CATEGORY_LINE.match(line) or ATTRIBUTE_LINE.match(line):
            continue
        if "=" in line:
            line = HTML_ATTRIBUTE.sub("", line)
        kept.append(line)
    text = "\n".join(kept)
    return MATH_MASK.sub(lambda match: maths[int(match.group(1))], text)


def quote_passage(match: regex.Match) -> str:
    passage = QUOTE_PASSAGE.search(match.group(0))
    return passage.group(1).strip() if passage else ""


def strip_markup_residue(text: str) -> str:
    """Removes link and template fragments that survive strip_markup; LaTeX math is masked first."""
    maths: list[str] = []

    def mask(match: regex.Match) -> str:
        maths.append(match.group(0))
        return f"\x00{len(maths) - 1}\x00"

    masked = MATH_SPAN.sub(mask, text)
    if BRACKET_RUN.search(masked) is None:
        return text
    lines = []
    for line in masked.split("\n"):
        line = OPENED_LINK.sub(r"\2\3", ORPHAN_LINK.sub(r"\1", line))
        # Balanced brackets are content: nested phonetics "[b`il[e]nd[e]]", XML "<![CDATA[ x ]]>".
        if line.count("[") != line.count("]"):
            line = LINK_BRACKETS.sub("", line)
        if LATEX_HINT.search(line) or line.count("{") == line.count("}"):
            lines.append(line)
            continue
        line = QUOTE_TEMPLATE.sub(quote_passage, line)
        line = TEMPLATE_OPENER.sub("", BROKEN_LINK_TEMPLATE.sub(r"\1", line))
        lines.append(CURLY_BRACKETS.sub("", line))
    return MATH_MASK.sub(lambda match: maths[int(match.group(1))], "\n".join(lines))


def strip_file_ref_residue(text: str) -> str:
    """Removes gallery file names, <ref> tag fragments and bare category lines; gallery captions are kept."""
    if FILE_REF_HINT.search(text) is None:
        return text
    lines = []
    for line in text.split("\n"):
        if BARE_CATEGORY_LINE.match(line):
            continue
        cleaned = REF_TAG.sub("", GALLERY_FILE.sub(" ", line))
        if cleaned == line:
            lines.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        cleaned = INNER_SPACES.sub(" ", cleaned).strip()
        if cleaned:
            lines.append(indent + cleaned)
    return "\n".join(lines)


def drop_template_residue(lines: list[str]) -> list[str]:
    """Removes the open and close lines of templates that span lines, and the "}}" of templates without a start."""
    kept = []
    open_templates = 0
    for line in lines:
        if TEMPLATE_START_LINE.match(line):
            open_templates += 1
            continue
        if open_templates > 0 and TEMPLATE_END_LINE.match(line):
            open_templates -= 1
            continue
        stray_end = STRAY_TEMPLATE_END.search(line)
        # Math is masked here; a line with "{" is code or LaTeX outside "$".
        if stray_end is not None and "{" not in line and LETTER.search(line) is not None:
            line = line[:stray_end.start()]
            if "|" in line:
                continue
        kept.append(line)
    return kept


def split_template_arguments(arguments: str) -> list[str]:
    """Splits at the pipes that are not inside a nested template or link."""
    parts = []
    depth = start = 0
    for position, character in enumerate(arguments):
        if character in "{[":
            depth += 1
        elif character in "}]":
            depth = max(depth - 1, 0)
        elif character == "|" and depth == 0:
            parts.append(arguments[start:position])
            start = position + 1
    parts.append(arguments[start:])
    return parts


def unwrap_content_template(match: regex.Match) -> str:
    if len(match.group(0)) > MAX_CONTENT_TEMPLATE_CHARS:
        return match.group(0)
    return template_content(match.group("arguments"))


def template_content(arguments: str) -> str:
    """The quoted text of a blockquote, quote or poem template: its text parameter or first positional argument."""
    positional = []
    for argument in split_template_arguments(arguments):
        named = TEMPLATE_PARAMETER.match(argument)
        if named is None:
            positional.append(argument)
        elif named.group("name").strip().lower() in CONTENT_PARAMETERS:
            return named.group("value").strip()
    return positional[0].strip() if positional else ""


def is_heading_like(line: str) -> bool:
    stripped = line.strip()
    return (0 < len(stripped) <= HEADING_MAX_CHARS and len(stripped.split()) <= HEADING_MAX_WORDS
            and not BULLET_LINE.match(stripped) and not stripped.endswith(SENTENCE_END + (",",)))


def drop_link_list_sections(lines: list[str]) -> list[str]:
    """Removes "Sources", "Related news", "See also" and similar headings together with their list items."""
    kept = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.rstrip(":").lower() in LINK_LIST_HEADINGS:
            in_section = True
            continue
        if in_section and (not stripped or BULLET_LINE.match(stripped)):
            continue
        in_section = False
        kept.append(line)
    return kept


def is_dangling_label(line: str) -> bool:
    label = LIST_NUMBER.sub("", line).strip()
    return label.endswith(":") and is_heading_like(label[:-1])


def drop_trailing_label(lines: list[str]) -> list[str]:
    """Removes a label left at the page end after its list or table was dropped ("Key:") and the heading above it."""
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    if end == 0 or not is_dangling_label(lines[end - 1]):
        return lines
    while end > 0 and (not lines[end - 1].strip() or is_dangling_label(lines[end - 1])):
        end -= 1
    # One heading only: a longer run of short lines can be verse.
    heading = end
    while heading > 0 and not lines[heading - 1].strip():
        heading -= 1
    if heading > 0 and is_heading_like(lines[heading - 1]):
        end = heading - 1
    return lines[:end]


def drop_trailing_headings(lines: list[str]) -> list[str]:
    """Removes section headings left at the end of a page whose sections were empty templates."""
    while lines and (not lines[-1].strip() or is_heading_like(lines[-1])):
        lines = lines[:-1]
    return lines


def repair_line(line: str) -> str:
    line = EMPTY_PARENS.sub("", line)
    line = LEADING_PAREN_PUNCT.sub("(", line)
    line = TRAILING_PAREN_PUNCT.sub(")", line)
    line = EMPTY_CURLY_QUOTES.sub("", line)
    line = REPEATED_COMMA.sub(",", line)
    line = SPACE_BEFORE_PUNCT.sub(r"\1", line)
    return INNER_SPACES.sub(" ", line)


def list_item_key(line: str) -> str | None:
    """The item text without bullet and number; None for a line that is not a list item or must never be removed."""
    if not BULLET_LINE.match(line) or "$$" in line or LATEX_COMMAND.search(line):
        return None
    item = LIST_NUMBER.sub("", line).strip()
    return item if PROSE_WORD.search(item) else None


def drop_repeated_list_runs(lines: list[str]) -> list[str]:
    """Removes a run of list items that repeats the run just before it in the same list, the copy wtf_wikipedia
    prints for nested lists (sample: "Tunis, Tunis"; "Rome .. London, Rome .. London")."""
    kept: list[str] = []
    keys: list[str | None] = []
    positions: dict[str, list[int]] = {}
    index = 0
    while index < len(lines):
        key = list_item_key(lines[index])
        if key is None and not BULLET_LINE.match(lines[index]):
            positions = {}
        run = 0
        for position in reversed(positions.get(key, [])) if key is not None else ():
            length = len(kept) - position
            if length > MAX_REPEATED_RUN:
                break
            candidate = [list_item_key(line) for line in lines[index:index + length]]
            if None not in candidate and candidate == keys[position:]:
                run = length
                break
        if run:
            index += run
            continue
        if key is not None:
            positions.setdefault(key, []).append(len(kept))
        kept.append(lines[index])
        keys.append(key)
        index += 1
    return kept


def repair_holes(document: dict) -> str:
    """Repairs punctuation around dropped templates and removes the list items and sections they emptied."""
    title, body = split_title(document["text"])
    wiki = wiki_of(document)
    lines = []
    for line in body.split("\n"):
        if wiki == WIKIVOYAGE and (EMPTY_PHRASEBOOK_LINE.match(line) or NAMELESS_LISTING.match(line)):
            continue
        if SIGNATURE.search(line) or GALLERY_FILE_LINE.match(line):
            continue
        if wiki in NAVIGATION_WIKIS and (BREADCRUMB_LINE.match(line) or NAVIGATION_LINE.match(line)):
            continue
        line = repair_line(line)
        if wiki == WIKTIONARY:
            line = EMPTY_STRAIGHT_QUOTES.sub("", line)
        if wiki == WIKIQUOTE:
            line = UNPAIRED_ITALIC_QUOTES.sub("", line)
        if EMPTY_LIST_ITEM.match(line):
            continue
        lines.append(line.rstrip())
    # On Wikiquote a repeat is a source shared by two quotes; on Wikisource it is a refrain or a speaker's line.
    if wiki not in (WIKIQUOTE, WIKISOURCE):
        lines = drop_repeated_list_runs(lines)
    lines = drop_trailing_label(drop_link_list_sections(lines))
    # Findings "holes": 1,648 empty standard Wikivoyage sections; elsewhere a short last line can be verse.
    if wiki == WIKIVOYAGE:
        lines = drop_trailing_headings(lines)
    if wiki == WIKTIONARY:
        lines = drop_empty_wiktionary_sections(lines)
    return join_title(title, "\n".join(lines))


def is_interwiki_residue(language: LanguageIdentifier, line: str, wiki: str) -> bool:
    """Findings "language mix": Wikinews and Wikisource pages end (or start) with the titles of their translations."""
    stripped = line.strip()
    words = len(stripped.split())
    # Sample: the Wikisource chapter heading "COLONEL DE CRAYE" is not labelled en by lid.176.
    if (BULLET_LINE.match(stripped) or stripped.endswith(SENTENCE_END)
            or not (MIN_RESIDUE_WORDS <= words <= MAX_RESIDUE_WORDS) or stripped.isupper()):
        return False
    label, probability = language.predict(stripped)
    if label == "en":
        return False
    translated = wiki in TRANSLATED_WIKIS
    if INTERWIKI_PREFIX.search(stripped) or NON_LATIN_SCRIPT.search(stripped):
        return True
    if stripped.count("/") >= (1 if translated else MIN_SLASHES_OUTSIDE_TRANSLATED_WIKIS):
        return True
    return translated and probability >= MIN_TRANSLATED_TITLE_PROBABILITY


def strip_interwiki_residue(language: LanguageIdentifier, document: dict) -> str:
    """Removes a last body line (or a first one on Wikisource) that lists the page's titles in other languages."""
    if is_page_namespace(document):
        return document["text"]
    title, body = split_title(document["text"])
    wiki = wiki_of(document)
    lines = body.split("\n")
    if lines and is_interwiki_residue(language, lines[-1], wiki):
        lines = lines[:-1]
    if wiki == WIKISOURCE and lines and is_interwiki_residue(language, lines[0], wiki):
        lines = lines[1:]
    return join_title(title, "\n".join(lines))


def join_title(title: str, body: str) -> str:
    body = regex.sub(r"\n{3,}", "\n\n", body).strip()
    return f"{title}\n\n{body}" if body else title


def is_empty_etymology(line: str) -> bool:
    words = [word.lower() for word in WORD.findall(line)]
    return bool(line.strip()) and all(word in ETYMOLOGY_CONNECTORS for word in words)


def wiktionary_heading(line: str) -> str | None:
    key = regex.sub(r"\s+\d+\Z", "", line.strip()).lower()
    return key if key in WIKTIONARY_HEADINGS else None


def clean_wiktionary(document: dict) -> str:
    """Removes translation tables, empty language slots, emptied etymologies and flag residue of a Wiktionary entry."""
    if wiki_of(document) != WIKTIONARY:
        return document["text"]
    title, body = split_title(document["text"])
    body = FLAG_EMOJI.sub("", body)
    lines: list[str] = []
    section = None
    for line in body.split("\n"):
        heading = wiktionary_heading(line)
        if heading is not None:
            section = heading
            lines.append(line)
            continue
        if section == "translations" or EMPTY_LABEL_SLOT.match(line):
            continue
        if section == "etymology" and is_empty_etymology(line):
            continue
        lines.append(line)
    return join_title(title, "\n".join(drop_empty_wiktionary_sections(lines)))


def drop_empty_wiktionary_sections(lines: list[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        if wiktionary_heading(line) is not None and kept and wiktionary_heading(kept[-1]) is not None:
            kept[-1] = line
            continue
        if not line.strip() and kept and wiktionary_heading(kept[-1]) is not None:
            continue
        kept.append(line)
    while kept and (not kept[-1].strip() or wiktionary_heading(kept[-1]) is not None):
        kept.pop()
    return kept


def body_words(document: dict) -> int:
    return len(body_of(document["text"]).split())


def is_disambiguation(document: dict) -> bool:
    title, body = split_title(document["text"])
    return title.endswith("(disambiguation)") or DISAMBIGUATION.search(body[:DISAMBIGUATION_SEARCH_CHARS]) is not None


def is_generated_page(document: dict) -> bool:
    title, body = split_title(document["text"])
    return CODE_TITLE.search(title) is not None or LATEX_SOURCE.search(body) is not None


def bullet_line_fraction(text: str) -> float:
    lines = [line for line in body_of(text).split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(1 for line in lines if BULLET_LINE.match(line)) / len(lines)


def words_outside_lists(text: str) -> int:
    return sum(len(line.split()) for line in body_of(text).split("\n") if not BULLET_LINE.match(line))


def punctuated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in body_of(text).split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(1 for line in lines if line.endswith(TERMINAL_PUNCTUATION)) / len(lines)


def has_list_layout(document: dict) -> bool:
    """Almost every line is a list item and almost none ends a sentence. Bulleted sentences (rules, recipes) stay."""
    # Findings "list-only pages": Wikiquote lays quotes out as lists; phrasebooks list phrases without final stops.
    if wiki_of(document) == WIKIQUOTE or split_title(document["text"])[0].endswith(PHRASEBOOK_TITLE):
        return False
    if bullet_line_fraction(document["text"]) <= MAX_BULLET_LINE_FRACTION:
        return False
    return punctuated_line_fraction(document["text"]) <= MAX_PUNCTUATED_LINE_FRACTION


def is_bullet_list(document: dict) -> bool:
    """A page of bare list items: TOCs, link lists, name lists."""
    # Sample: list pages with 26-99 words of prose were biographies with a filmography; a stub of that prose stays.
    return has_list_layout(document) and words_outside_lists(document["text"]) < MIN_BODY_WORDS


def keep_prose_of_bullet_list(document: dict) -> str:
    """Sample: a season page (85 words of prose, its stats table as bare items) failed low_prose; the prose stays."""
    text = document["text"]
    if (not has_list_layout(document) or words_outside_lists(text) < MIN_BODY_WORDS
            or stopword_fraction(text) >= MIN_STOPWORD_FRACTION):
        return text
    title, body = split_title(text)
    # The headings and labels of the stub head the dropped items.
    lines = [line for line in body.split("\n")
             if not (BULLET_LINE.match(line) or is_heading_like(line) or is_dangling_label(line))]
    if len(" ".join(lines).split()) < MIN_BODY_WORDS:
        return text
    return join_title(title, "\n".join(lines))


def duplicate_line_char_fraction(text: str, wiki: str) -> float:
    """Repeats of a list item are left out: they are table cells, sub-lists or shared sources (sample: a drug list by
    class and then in full alphabetical order); on Wikiquote they still count, as there an item is a quote."""
    seen: set[str] = set()
    seen_items: set[str] = set()
    duplicate_chars = total_chars = 0
    for line in body_of(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        if wiki != WIKIQUOTE and BULLET_LINE.match(line):
            item = LIST_NUMBER.sub("", line).strip()
            if item in seen_items:
                continue
            seen_items.add(item)
        total_chars += len(line)
        if line in seen:
            duplicate_chars += len(line)
        seen.add(line)
    return duplicate_chars / total_chars if total_chars else 0.0


def is_repetitive(document: dict) -> bool:
    return duplicate_line_char_fraction(document["text"], wiki_of(document)) > MAX_DUPLICATE_LINE_CHAR_FRACTION


def prose_for_language_id(text: str) -> str:
    """The body without LaTeX math (lid.176 labels it English), one line per segment, so a gloss and its example
    are labelled apart (calibration: Wikiquote pages with original-language quotes rose from 0.1-0.5 to over 0.5)."""
    return "\n\n".join(line for line in MATH_SPAN.sub(" ", body_of(text)).split("\n") if PROSE_WORD.search(line))


def english_share(language: LanguageIdentifier, text: str) -> float | None:
    """Letter-weighted English share of the paragraphs; None when the prose is too short for language ID."""
    prose = prose_for_language_id(text)
    if sum(1 for _ in LETTER.finditer(prose)) < MIN_PROSE_LETTERS:
        return None
    return language.english_share(prose)


def mask_personal_data(document: dict) -> str:
    return mask_pii(document["text"], kinds=PII_KINDS_BY_WIKI.get(wiki_of(document), PII_KINDS))


def stopword_fraction(text: str) -> float:
    words = MATH_SPAN.sub(" ", body_of(text)).lower().split()
    return sum(1 for word in words if word in STOPWORDS) / len(words) if words else 0.0


def is_english(language: LanguageIdentifier, text: str) -> bool:
    share = english_share(language, text)
    return share is None or share >= ENGLISH_MIN_SHARE


class DocumentMap(Step):
    """A text transform that also reads the document metadata."""

    def __init__(self, name: str, transform):
        super().__init__(name)
        self.transform = transform

    def apply(self, document: dict) -> dict:
        text = self.transform(document)
        if text == document["text"]:
            return document
        return {**document, "text": text}


class BodyExactIndexer(ExactIndexer):
    def add(self, local_id: int, document: dict) -> None:
        key = exact_key(body_of(document["text"]))
        if key is not None:
            self.keys.add(local_id, np.array([key], dtype=np.uint64))


class BodyExactDedup(ExactDedup):
    """Exact dedup of the body: boilerplate pages differ only in their title (findings "boilerplate exact duplicates")."""

    def open_indexer(self, directory: Path, shard_index: int) -> BodyExactIndexer:
        return BodyExactIndexer(directory, shard_index)


class WikimediaPipeline(Pipeline):
    """Builds the Page store in the run's work directory before the first pass; a resumed run reuses it."""

    def __init__(self, name: str, steps: list, book_assembly: BookAssembly):
        super().__init__(name, steps)
        self.book_assembly = book_assembly

    def run_in_place(self, data_dir: Path, meta_dir: Path, processes: int = 1, run_info: dict | None = None) -> dict:
        data_dir, meta_dir = Path(data_dir).resolve(), Path(meta_dir).resolve()
        if (meta_dir / "stats.json").exists():
            raise FileExistsError(f"{meta_dir} holds the stats of a finished run; the data is already processed")
        store_dir = meta_dir / "_work" / BOOK_STORE_DIR
        if not (store_dir / STORE_COMPLETE).exists():
            self.build_store(data_dir, store_dir, processes)
        self.book_assembly.store_dir = store_dir
        try:
            return super().run_in_place(data_dir, meta_dir, processes, run_info)
        finally:
            self.book_assembly.store_dir = None

    @staticmethod
    def build_store(data_dir: Path, store_dir: Path, processes: int) -> None:
        if (store_dir.parent / "run.json").exists():
            raise RuntimeError(f"{store_dir} is incomplete but the run has started; the Page store cannot be rebuilt "
                               "from partly rewritten shards")
        shutil.rmtree(store_dir, ignore_errors=True)
        store_dir.mkdir(parents=True)
        shards = list_shards(data_dir)
        jobs = [(shard, store_dir, index) for index, shard in enumerate(shards) if shard_wiki(shard) == WIKISOURCE]
        if processes > 1 and len(jobs) > 1:
            with multiprocessing.get_context("fork").Pool(min(processes, len(jobs))) as pool:
                list(pool.imap_unordered(index_shard_pages, jobs))
        else:
            list(map(index_shard_pages, jobs))
        summary = build_book_index(store_dir, [index for _, _, index in jobs])
        write_bytes_atomically(store_dir / STORE_COMPLETE, orjson.dumps(summary))


def build_pipeline() -> WikimediaPipeline:
    language = LanguageIdentifier()
    book_assembly = BookAssembly("assemble_wikisource_books")
    return WikimediaPipeline("wikimedia", [
        Filter("content_namespace", is_content_page),  # "non-article namespaces": 72.6% of docs; metadata only
        Filter("raw_ocr_page", lambda document: not is_raw_ocr_page(document)),  # unproofread bot OCR dumps
        book_assembly,  # "Wikisource Page: namespace": books stored as per-page fragments
        TextMap("normalize", normalize_text),  # ftfy and NFC with FineWeb/datatrove settings (LITERATURE §c)
        TextMap("strip_markup", strip_markup),  # "residual wikitext and HTML markup": up to 25% of main docs
        TextMap("markup_residue", strip_markup_residue),  # output audit: fragments in 0.5% of docs
        TextMap("file_ref_residue", strip_file_ref_residue),  # output audit: gallery names, <ref> fragments
        Filter("damaged_ocr", lambda document: not is_damaged_ocr(document)),  # unproofread OCR in Page books
        DocumentMap("clean_wiktionary", clean_wiktionary),  # "Wiktionary skeletons": empty slots in 11.3% of docs
        DocumentMap("repair_holes", repair_holes),  # "holes where wtf_wikipedia dropped templates": up to 26% of docs
        DocumentMap("strip_interwiki_residue", lambda document: strip_interwiki_residue(language, document)),
        Filter("min_words", lambda document: body_words(document) >= MIN_BODY_WORDS),  # "title-only and stubs"
        Filter("disambiguation", lambda document: not is_disambiguation(document)),  # 4.9-6.5% of main docs
        Filter("generated_pages", lambda document: not is_generated_page(document)),  # "code and LaTeX pages"
        DocumentMap("bullet_list_prose", keep_prose_of_bullet_list),
        Filter("bullet_lists", lambda document: not is_bullet_list(document)),  # "list-only and low-prose pages"
        Filter("repetition", lambda document: not is_repetitive(document)),  # "templated, repetitive pages"
        Filter("low_prose", lambda document: stopword_fraction(document["text"]) >= MIN_STOPWORD_FRACTION),
        Filter("english", lambda document: is_english(language, document["text"])),  # "language mix"
        BodyExactDedup("exact_dedup"),  # Gopher A.3.2: exact then fuzzy dedup each lowered loss
        NearDedup("near_dedup"),  # "templated near-duplicate page series": 0.2-1.2% of long main docs
        DocumentMap("mask_pii", mask_personal_data),  # Common Pile §4.1 regex PII
    ], book_assembly)


if __name__ == "__main__":
    run_cli(build_pipeline)
