"""Cleaning pipeline for Common Pile project_gutenberg; "findings" means analysis/findings/project_gutenberg.json."""
import unicodedata
from pathlib import Path

import numpy as np
import regex

from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import (
    MAX_BUCKET_ENTRIES, Deduplicator, DuplicateOfWriter, ExactDedup, KeyIndexWriter, MinHasher, dedup_words,
    iter_key_buckets, load_index, shard_offsets, shard_prefix,
)
from data_pipeline.common.langid import LETTER, LanguageIdentifier, fasttext_input, language_segments
from data_pipeline.common.pii import mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, TextMap
from data_pipeline.common.text import normalize_text

# PG puts three or four blank lines between sections and one or two inside them (PG/DP formatting guidelines).
SECTION_BREAK = regex.compile(r"\n[^\S\n]*(?:\n[^\S\n]*){2,}\n")
LEADING_BLANK_LINES = regex.compile(r"(?:[^\S\n]*\n)+")
PARAGRAPH_BREAK = regex.compile(r"\n[^\S\n]*\n")

# Findings "PG licence header/footer": the Common Pile already cut at these markers; kept for files that still have them.
START_MARKER = regex.compile(r"(?im)^[^\n]*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*$")
END_MARKER = regex.compile(r"(?im)^[^\n]*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*$")
# Findings "residual trailers": the closing line names the ebook; everything after it is PG text.
TRAILER = regex.compile(
    r"(?im)^[ \t]*\**[ \t]*(?:the )?(?:end of (?:the |this )?(?:project gutenberg|p\.? ?g\.?\b|e-?text\b)"
    r"|ende (?:dieses|des) projekt(?:s)? gutenberg|fin (?:du|de ce|de la|del) (?:projet|project|proyecto) gutenberg"
    r"|\*{3}end of the collected etexts)"
)
TRAILER_WINDOW_CHARS = 3000
# Findings "producer credit preamble", extended with the credit openings counted in the sample's first paragraphs.
CREDIT_OPENING = regex.compile(
    r"(?i)(?:produced by|(?:this )?e-?text (?:was )?(?:prepared|produced|scanned|created|transcribed) by"
    r"|this e-?(?:book|text) (?:was|is|has been) "
    r"(?:produced|prepared|created|made|transcribed|scanned)|transcribed (?:from .{0,120}? )?by|digiti[sz]ation by"
    r"|(?:text )?scanned (?:and proofed |and proofread )?by|prepared by|e-?book prepared by|html version by|proofread by"
    r"|converted to (?:e-?text|html) by)\b"
)
HEAD_WINDOW_CHARS = 5000
# Findings "header fragment": the end of a START line split over several lines, e.g. "RELATED BY HIMSELF***".
HEADER_FRAGMENT = regex.compile(r"\A\s*(?:[^\S\n]*\S[^\n]*\n){0,3}?[^\n]*\p{L}[^\n]*\*{3,}[ \t]*(?=\n|\Z)")
# Findings "PG mentions": provenance lines name PG, its proofreading site or its mirrors.
PG_MENTION = regex.compile(
    r"(?i)project gutenberg|gutenberg\.org|ibiblio\.org|pgdp\.net|promo\.net/pg|distributed proofread"
    r"|gutenberg-tm|\bPGLAF\b|gutenberg literary archive|\bPG (?:edit(?:ion|or)|catalog)"
)
# Findings "old-style legal headers": phrases of the 1990s header and small print that do not name PG.
LEGAL_PHRASE = regex.compile(
    r"(?i)legal small print|small print!|plain vanilla electronic texts|prepared by (?:hundreds|thousands) of volunteers"
    r"|contributions are (?:only )?being solicited"
    r"|\*END\*|(?:this header )?should be the first thing seen|we are releasing two versions of this etext"
    r"|important information in this header|etexts readable by both humans|copyright laws are changing all over"
    r"|this (?:e-?)?text uses [^\n]{0,40}encoding|use the latin-1 version|short list of bookmarks"
)
# Notes on the scanned images (sample: "Images of the original pages are available through Internet Archive"); books
# from before 1928 never name the Internet Archive, so a short paragraph that does is about the ebook.
SCAN_SOURCE = regex.compile(
    r"(?i)internet archive|archive\.org|images of the original pages|produced from (?:page )?images|ocr software")
# "etext" and "ebook" postdate every PG book, so a short paragraph that uses them describes the file (81 English sample
# books keep such lines after the other rules, e.g. "Dates of addresses by Calvin Coolidge in this eBook:").
EBOOK_MENTION = regex.compile(r"(?i)\be-?texts?\b|\be-?books?\b")
HEADER_FIELD = regex.compile(
    r"(?im)^[ \t]*(?:title|author|release date|posting date|last updated|first posted|edition|language"
    r"|character set encoding|\[?e-?book #?\d+\]?|\[most recently updated[^\]]*\])\b.*$"
)
MIN_HEADER_FIELDS = 2
# Paragraphs that name PG are metadata about the ebook; a longer one is a book about PG (dropped by is_book_text).
MAX_PROVENANCE_PARAGRAPH_CHARS = 1500
URL_LINE = regex.compile(r"[ \t]*(?:or[ \t]*)?\(?(?:https?://|www\.)\S+\)?[ \t]*|[ \t]*See \S+-h\.htm.*")

# Findings "transcriber's notes": the bracketed form, the heading form and the boxed form.
TRANSCRIBER_NOTE_OPENING = r"(?:transcriber|transcription|end transcriber|note of transcriber|etext editor)"
TRANSCRIBER_BRACKET = regex.compile(
    rf"(?i)\[{TRANSCRIBER_NOTE_OPENING}"
    # Notes quote the book's own bracket codes, e.g. "shown as [M]", so one level of nested brackets is allowed.
    r"(?:[^\[\]]|\[[^\[\]\n]{0,40}\]){0,6000}\]"
    # Books that use [ ] themselves put the transcriber's notes in braces (sample: "{Transcriber's Note: ...}").
    rf"|\{{{TRANSCRIBER_NOTE_OPENING}(?:[^{{}}]|\{{[^{{}}\n]{{0,40}}\}}){{0,6000}}\}}"
    # Sample: "[Original text says "Tom". (note of etext transcriber.)]" and "[Greek text deleted from etext.]".
    r"|\[[^\[\]]{0,1000}\b(?:e-?text|e-?book)\b[^\[\]]{0,1000}\]"
)
TRANSCRIBER_HEADING = regex.compile(
    # Sample headings are also set in italic or bold markup: "_Transcriber's Notes_", "=Transcriber's Notes:=", "** ...".
    r"(?im)^[ \t]*[\[(●•○*=_-]{0,3}[ \t]*(?:the )?(?:transcriber['’]?s?['’]?|transcription|etext editor['’]?s?)[ \t]+"
    r"(?:notes?|remarks?|comments?|corrections?|errata|bookmarks|amendments)(?!\p{L})"
    # Sample: "Typographical errors corrected by the etext transcriber:", "The following obvious typographical errors
    # have been corrected:", "Corrections made to printed original.", "[Compiler's Notes", "ERRATA." before such a list.
    r"|^[ \t]*(?:the following )?(?:obvious )?typographical errors (?:were |have been )?(?:silently )?corrected\b"
    r"|^[ \t]*corrections made to (?:the )?printed original"
    r"|^[ \t]*\[?[ \t]*compiler['’]s notes?\b"
    # Sample: "[Errata Noted by Transcriber:" and "Errata Noted by Transcriber" at the end of three books.
    r"|^[ \t]*\[?[ \t]*(?:errata|corrections?|changes)[ \t]+(?:noted|made)[ \t]+by[ \t]+(?:the[ \t]+)?transcriber"
    # A bare errata heading elsewhere can be a contents line, so it counts only at the tail or over a printed list.
    r"|^[ \t]*(?P<errata>errata(?:[ \t]+(?:in|to)[ \t]+[^\n]{1,30}?)?)[.:]?[ \t]*$"
)
# Printed errata lists in front matter (sample: 'P. 25, line 6 from bottom, _for_ "_weis_" _read_ "_weiss_."').
ERRATA_ENTRY = regex.compile(r"(?i)(?<!\p{L})for(?!\p{L})[^\n]{1,80}?(?<!\p{L})read(?!\p{L})")
ERRATA_ENTRY_WINDOW_CHARS = 500
# A note at the end of a book runs to the end of the text; a note elsewhere ends at the next PG section break.
TAIL_NOTE_MAX_CHARS = 20000
# Short books are under TAIL_NOTE_MAX_CHARS, so a tail note must also start in the second half of the text.
TAIL_NOTE_MIN_POSITION_SHARE = 0.5
MAX_NOTE_CHARS = 10000
# DP closes a front note with a row of asterisks; when the section break is missing, the title page or a picture follows
# the note after one blank line (sample: "MY FATHER'S DRAGON", "THE AUGUSTAN REPRINT SOCIETY", "[Illustration: ...").
NOTE_STOP = regex.compile(
    r"\n[^\S\n]*\n(?=[ \t]*(?:\*[ \t]*){3,}\n|[ \t]*\[(?:Illustration|Frontispiece)\b"
    r"|[^\p{Ll}\n]*\p{Lu}[^\p{Ll}\n]*\n[^\S\n]*\n)"
)
# Footnotes can follow an errata list at the end of a book (sample: 17,068 chars after the list in one book).
FOOTNOTES_HEADING = regex.compile(r"(?im)^[ \t]*foot-?notes?[.:]?[ \t]*$")
BOX_LINE = regex.compile(r"[ \t]*(?:[+┌└├╔╚][-=+─━═┬┴┼]{3,}[+┐┘┤╗╝]?|[|│║].*)[ \t]*")
BOX_NOTE = regex.compile(
    r"(?i)transcriber|errata|corrections? (?:made|have been)|typographical errors corrected|corrected in (?:the )?text")

# Findings "tables of contents": a navigation block runs from its heading to the next PG section break. The heading is
# capitalized and opens a paragraph, or is all upper-case; a wrapped prose line such as "figures." is neither (sample: 6
# books lost text). A list may start right under its heading (sample: "CONTENTS." then "Page." then the entries).
CONTENTS_HEADING = regex.compile(
    r"(?i)(?:(?<=\A|\n[^\S\n]*\n)[ \t]*(?-i:(?=\p{Lu}))|(?<=\n)[ \t]*(?-i:(?=[^\p{Ll}\n]*(?:\n|\Z))))"
    r"(?:table of contents|contents|contents of (?:the )?(?:volume|vol\.|part|book)\b.*"
    r"|(?:list of (?:the )?)?(?:illustrations|plates|maps|figures|engravings|full-page illustrations))\.?:?[ \t]*"
    r"(?=\n|\Z)"
)
# Illustration lists hold captions of any length, so only contents lists must consist of short or page-numbered entries.
ILLUSTRATION_LIST = regex.compile(
    r"(?i)\s*(?:list of )?(?:the )?(?:illustrations|plates|maps|figures|engravings|full-page)")
# Sample headings may carry italic or bold markup ("_Index_"), a kind ("NAME INDEX", "AN Alphabetical INDEX TO THE
# FOURTH VOLUME.") or a decoration tag before them.
INDEX_HEADING = regex.compile(
    r"(?im)^[ \t]*(?:\[Illustration:?[^\[\]\n]{0,40}\][ \t]*)?[_=]?(?:(?:an|the)[ \t]+)?"
    r"(?:(?:general|analytical|alphabetical|glossarial|name|subject|authors?['’]?|topical)[ \t]+)?"
    r"index(?:[ \t]+(?:of|to)[ \t]+[^\n]{1,40})?\.?:?[_=]?[ \t]*$"
)
# A reference may carry a volume or chapter ("i. 197", "xliii. 4", "ch. 5") and notes ("276 _note_", "59 _n._").
REFERENCE_TARGET = (r"\s*(?:[ivxlc]{1,7}[.,]\s*)?(?:(?:chs?|chap|sect?)\.\s*)?_?(?:\d{1,4}|[ivxlc]{1,7}|ib(?:id)?\.)_?"
                    r"(?:\s*(?:[-–,;&]|f\.|ff\.|n\.|_?n\._?|_?(?:and )?notes?_?|(?:et )?s(?:e)?qq?\.|\(\d{1,2}\))\s*"
                    r"(?:[ivxlc]{1,7}[.,]\s*)?_?\d{0,4}_?)*\s*[.;,]?\s*$")
REFERENCE_SEPARATOR = r",|;|\.{2,}|\s{2,}|\bp\.|\bpp\.|\bpage"
PAGE_REFERENCE = regex.compile(rf"(?:{REFERENCE_SEPARATOR}){REFERENCE_TARGET}", regex.IGNORECASE)
# Inside an index a single space or a dash also separates an entry from its page ("Abiko 105", "Achechy--237"), a
# wrapped entry can end in a bare page ("473.") or carry its pages inside the line ("Accidents: At a launch, 22; to
# Volunteers, 41"), and a cross-reference ("_See_ Languages") stands for pages.
INDEX_REFERENCE = regex.compile(
    rf"(?:{REFERENCE_SEPARATOR}|\s|-{{2}}|—|^){REFERENCE_TARGET}|[,;]\s*(?:[ivxlc]{{1,7}}[.,]\s*)?\d{{1,4}}(?:[-–]\d{{1,4}})?[;,.]"
    r"|(?:^|[\s,;(.>])_?see(?: also)?_?\b",
    regex.IGNORECASE,
)
INDEX_LINE = regex.compile(r"(?m)^[^\n]*")
# Footnotes after an index cite volumes and pages like its entries (sample: "[245] Stortenbeker, _Zeitschr. physikal.
# Chem._, 1889, 3. 22."), so the index ends where they start.
FOOTNOTE_START = regex.compile(r"(?im)^[ \t]*(?:foot-?notes?[.:]?[ \t]*$|\[(?:\d{1,4}|footnote)\b)")
INDEX_CONTINUATION_START = regex.compile(r"[\p{Ll}\d(&]")
# Letter and group headings inside an index ("A", "— B —", "1 SAMUEL", "PAGE") are neither entries nor references; a
# line ending in a number ("II. 563") is a reference.
INDEX_SUBHEADING = regex.compile(r"(?=.*\p{Lu})[^\p{Ll}]{0,39}[^\p{Ll}\d\s]")
CHAPTER_LINE = regex.compile(r"(?i)[ \t]*(?:chapter|chap\.|book|part|canto|act|scene|section)\b.{0,60}")
MAX_ENTRY_CHARS = 60
MIN_ENTRY_SHARE = 0.5
MIN_INDEX_REFERENCE_SHARE = 0.6
MIN_INDEX_ENTRIES = 5
# An index ends where most of the next entries carry no reference (e.g. publisher's advertisements after it).
INDEX_WINDOW_ENTRIES = 10
# Sample: indexes open with up to three note paragraphs ("The numerals refer to the books, the figures to chapters").
MAX_INDEX_NOTE_ENTRIES = 5
MAX_INDEX_NOTE_CHARS = 1500
# Wrapped lines longer than an entry, with no page reference, are running text (e.g. an introduction after a list).
MIN_PROSE_LINES = 2
# Captions and their credits run to two lines; narrative after an illustration list is longer.
MIN_ILLUSTRATION_LIST_PROSE_LINES = 3
MAX_NAVIGATION_LINES = 1000
# Front-matter lists sit in the first part of a book; a CONTENTS heading later is often a catalogue or advertisement.
NAVIGATION_START_SHARE = 0.2
# A contents list is a small part of a book; a longer block means the section break was missed.
MAX_NAVIGATION_SHARE = 0.1
MIN_NAVIGATION_LIMIT_CHARS = 3000
# Back-of-book indexes sit in the last part of a book; an INDEX heading earlier is usually a section title.
INDEX_START_SHARE = 0.6

# Findings "hard-wrapped prose": about 70-column wrapping; verse and play lines are shorter than the wrap width.
WRAP_WIDTH_QUANTILE = 0.95
MIN_WRAP_SAMPLE_LINES = 20
# Findings: a book is hard-wrapped when its lines fall in the 50-80 band (per-book wrap width p25-p95 = 67-71).
MIN_WRAP_WIDTH = 50
MAX_WRAP_WIDTH = 80
# A paragraph whose every line is over MAX_WRAP_WIDTH was wrapped at its own width (sample: footnotes whose first line
# carries "[Footnote 47: ", transcribed manuscripts with lines of 85-100 characters).
MAX_PARAGRAPH_WIDTH = 100
# A line is full when it plus the next word fills 85% of the wrap width: the trough between the verse mode (0.65-0.7)
# and the prose mode (1.0-1.1) of that fill over 13.7M sample lines; DP books keep printed breaks, so fill varies.
FULL_LINE_FILL = 0.85
# Findings: right-margin numbers (verse line or section numbers) end lines in 27.1% of docs.
# Sample: some books put the printed page number in the right margin as "[121]" or "(p. 309)".
MARGIN_NUMBER = regex.compile(r"(?<=\S)(?:[ \t]{3,}\d{1,4}|[ \t]{2,}(?:\[\d{1,4}\]|\(p\. ?\d{1,4}\)))[ \t]*$")
# Verse capitalizes every line; in prose a few continuation lines start with a capital, so 2+ of them rarely all do.
MIN_VERSE_CHECK_LINES = 3
# Verse lines can open with an elision, a dash, a German or French quote ("’Twixt", "—And", "„Wollt") or a number.
CAPITALIZED_LINE = regex.compile(r"[\"'“‘’‛„‚«»(\[—–_]*(?:\p{Lu}|\d|[ivxlc]+\.)")
# A speaker label or heading line ("BERTRAM.", "1 PETITIONER.", "CHAPTER I.") over a prose paragraph keeps its own line.
LABEL_LINE = regex.compile(r"[ \t]*(?:\d[ \t]+)?\p{Lu}[\p{Lu}\p{M}\d .,'’&-]{0,40}\.")
# DP puts sidenotes just before their paragraph; some books run them into its first line.
LEADING_SIDENOTES = regex.compile(r"(?:[ \t]*\[Sidenote:[^\[\]]{0,500}\])+[ \t]*")
# A block quotation, a letter or a transcribed manuscript can be wrapped narrower than the book; such a paragraph is
# judged by its own longest line. Its lines after the first share one indent; verse with alternating indents does not.
MIN_BLOCK_INDENT = 2
MIN_BLOCK_LINES = 3
# An unindented narrow paragraph can be verse, whose lines mostly end in punctuation; wrapped prose lines end
# mid-sentence (sample: a German stanza ends 4 of 5 lines in punctuation, "The Declaration of Independence" 1 of 6).
END_STOPPED_LINE = regex.compile(r"[,;:.!?][\"'’”)\]]*[ \t]*$")
MAX_END_STOPPED_SHARE = 0.5
# Prose wrapped by the transcriber can keep one printed short line after its first line; the next line starting in
# lower case shows that the sentence goes on, which a salutation, a letter's closing or a verse break does not.
MAX_SHORT_INTERNAL_LINES = 1
LOWER_CASE_START = regex.compile(r"[ \t]*\p{Ll}")
# Table rows and aligned lists have a gap of 3+ spaces, a tab or a bar inside the line; they are never wrapped prose.
# Old etexts put up to three spaces after a sentence, so a short gap after punctuation does not count; a gap of five or
# more spaces after punctuation separates parallel columns (sample: Latin and English gospel texts side by side).
ALIGNED_LINE = regex.compile(r"[^\s.!?:;,\"'”’)\]](?: {3,}|\t)\S|\S {5,}\S|\|")

# Findings "typographic markup".
ILLUSTRATION = regex.compile(
    r"(?:(?<=\S)[^\S\n])?\[(?:Illustration|Frontispiece|Image|Plate|Portrait|Decoration|Map|Picture)\b"
    r"[^:\[\]\n]{0,20}:?\s*([^\[\]]{0,3000})\]",
    regex.IGNORECASE,
)
# A caption on its own lines is book text (findings: keep it as its own paragraph); one inside a sentence breaks it.
# Captions that only name the kind of image; the most frequent captions of the 35,092 tags in the sample.
IMAGE_KIND_CAPTION = regex.compile(
    r"(?i)\(?(?:(?:book )?cover(?: art)?|frontispiece|logo|colophon|(?:text )?decoration|ornament(?:al block)?"
    r"|uncaptioned|unlabell?ed|initial|illuminated capital|silhouette|hieroglyph|musical notation"
    r"|decorative [\w ]{1,30}|(?:end of )?chapter (?:icon|heading)|end of chapter icon|fig(?:ure)?\.? ?\d+)[.)]*"
)
# A one-letter caption directly before letters is a drop cap: "[Illustration: T]HE day" must keep its "T".
DROP_CAP = regex.compile(r"\p{L}")
OWN_LINE_BEFORE = regex.compile(r"(?:\A|\n)[^\S\n]*\Z")
OWN_LINE_AFTER = regex.compile(r"[^\S\n]*(?:\n|\Z)")
# DP writes subscripts and superscripts as _{2} and ^{2}, so a braced number after _ or ^ is not a page number.
PAGE_MARKER = regex.compile(
    r"(?<![_^])\{(?:pg|p\.?|page)?\s?[\dixvlc]{1,6}(?:\s?n\.)?\}|\[(?:Pg|pg|Page|p\.)\s?[\dixvlc]+\]|<pb\b[^>]*>",
    regex.IGNORECASE,
)
PAGE_MARKER_LINE = regex.compile(rf"\n[^\S\n]*(?:{PAGE_MARKER.pattern})[^\S\n]*(?=\n)", regex.IGNORECASE)
BRACED_SCRIPT_DIGITS = regex.compile(r"([_^])\{([-+−]?\d{1,6})\}")
# Abbreviations printed with raised letters ("w^{ch}", "M^{r}.", "L^{TD}") read as plain letters; a single letter that is
# a common maths variable ("X^{n}") stays an exponent.
BRACED_SCRIPT_LETTERS = regex.compile(r"\^\{(?![abijkmnpqxyz]\})(\p{L}{1,4})\}")
SCRIPT_DIGITS = {"_": str.maketrans("0123456789+-−", "₀₁₂₃₄₅₆₇₈₉₊₋₋"),
                 "^": str.maketrans("0123456789+-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻")}
HTML_TAG = regex.compile(r"</?(?:i|b|em|strong|sup|sub|sc|br|p|span|div|u|tt|small|big|font|a)\b[^>\n]{0,200}>")
# Digits may touch the underscores: DP writes italic money units as "6_s_." (six shillings).
ITALIC = regex.compile(r"(?<![\p{L}_])_([^_\n]{1,200}?)_(?![\p{L}_])")
BOLD = regex.compile(r"(?<![\w=])=(?=\S)([^=\n]{1,100}?)(?<=\S)=(?![\w=])")
# Some books mark bold with tildes instead (their transcriber's notes say so); 54 of 3,000 sample books have ~spans~.
TILDE_BOLD = regex.compile(r"(?<![\w~])~(?=\S)([^~\n]{1,100}?)(?<=\S)~(?![\w~])")
# Equals signs also write equations; a book uses =bold= only when it has several spans (findings: 3 or more).
MIN_BOLD_SPANS = 3
FOREIGN_SCRIPT = regex.compile(
    r"\[(?:Greek|Hebrew|Arabic|Sanskrit|Syriac|Coptic|Gothic|Russian|Chinese):\s*([^\[\]]{0,500})\]")
# DP transliterations mark accents and breathings with ASCII codes, e.g. "O(\\lôs de\\ sêmei=on" (DP guidelines).
TRANSLITERATION_MARK = regex.compile(r"(?<=\p{L})[/\\=|+()]+|[()*/\\=|+]+(?=\p{L})")
# Distributed Proofreaders diacritic codes, e.g. [=a] for a with macron (DP formatting guidelines).
DIACRITIC_CODE = regex.compile(r"\[([=)~^`':.,v])([A-Za-z])\]")
# [h.] is a dot below only inside a word ("Pin[h.]as"); alone, "[L.]" is an etymology label and "[p.]" a page label.
BELOW_DIACRITIC_CODE = regex.compile(r"(?|(?<=\p{L})\[([A-Za-z])([,.])\]|\[([A-Za-z])([,.])\](?=\p{L}))")
COMBINING_MARKS = {"=": "̄", ")": "̆", "~": "̃", "^": "̂", "`": "̀", "'": "́",
                   ":": "̈", ".": "̇", "v": "̌", ",": "̧"}
BELOW_MARKS = {",": "̧", ".": "̣"}
LIGATURES = {"[oe]": "œ", "[OE]": "Œ", "[ae]": "æ", "[AE]": "Æ"}
ASTERISM_LINE = regex.compile(r"(?m)^[ \t]*(?:\*[ \t]*){3,}$")
BLANK_PAGE_LINE = regex.compile(r"(?im)^[ \t]*\[(?:blank page|this page (?:intentionally )?left blank)\][ \t]*$")
EMPTY_LINES = regex.compile(r"\n{3,}")

# Findings "stubs": placeholders for audio, music or image ebooks with no book text; the sample adds audio-reading
# certificates (LibriVox) and readmes of zipped word lists (Moby).
STUB_MEDIA = regex.compile(
    r"(?i)\.(?:mus|mp3|midi?|avi|mpg|mpeg|ly|wav|ogg|pdf|tex|sib|zip)\b|musical score|no text version|audio ?book"
    r"|sheet music|human-read|computer-generated reading|librivox|audio reading|zip file"
)
MAX_STUB_CHARS = 5000
# Dolma (arXiv:2402.00159) drops Gutenberg books under 25 words.
MIN_WORDS = 25
WORD = regex.compile(r"\w+")
# Findings: digit share above 0.10 marks registers, census tables and catalogues (1.21% of docs, 3.18% of chars).
MAX_DIGIT_SHARE = 0.10
# Gopher (arXiv:2112.11446 A.1.1) keeps web pages with 80% of words containing a letter; in the sample all 8 English
# books between 0.70 and 0.80 are scientific bulletins whose tables lower the share around good prose, and the one
# below 0.70 is mostly word-frequency lists, so books use 0.70.
MIN_ALPHABETIC_WORD_SHARE = 0.7
# PG-made catalogues of its own files, e.g. "Index of the Project Gutenberg Works of George Eliot" (sample).
PG_CATALOGUE_TITLE = regex.compile(r"(?i)^\s*index of the project gutenberg works")
# Dolma (arXiv:2402.00159) keeps a Gutenberg book when English is at least 0.5; the sample has no book between 0.23 and
# 0.51 after the confidence rule, so the cut is insensitive there.
ENGLISH_MIN_SHARE = 0.5
# lid.176 has no label for e.g. Ojibwe and calls it English with low confidence (sample: mean 0.17 on a catechism,
# English books p0.5 0.76); Caswell et al. (arXiv:2010.14571) list unseen languages as an LID failure mode.
MIN_ENGLISH_CONFIDENCE = 0.3


def cut_at_markers(text: str) -> str:
    start = START_MARKER.search(text)
    if start:
        text = text[start.end():]
    end = END_MARKER.search(text)
    if end:
        text = text[:end.start()]
    return text


def cut_trailer(text: str) -> str:
    window_start = max(0, len(text) - TRAILER_WINDOW_CHARS)
    trailer = TRAILER.search(text, window_start)
    return text[:trailer.start()] if trailer else text


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """(start, end) of each paragraph between blank lines."""
    spans = []
    start = 0
    for match in PARAGRAPH_BREAK.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def is_url_paragraph(paragraph: str) -> bool:
    lines = [line for line in paragraph.split("\n") if line.strip()]
    return bool(lines) and all(URL_LINE.fullmatch(line) for line in lines)


def is_header_block(paragraph: str) -> bool:
    lines = [line for line in paragraph.split("\n") if line.strip()]
    fields = sum(1 for line in lines if HEADER_FIELD.fullmatch(line))
    return fields >= MIN_HEADER_FIELDS and fields == len(lines)


def is_provenance_paragraph(paragraph: str, in_head: bool) -> bool:
    if len(paragraph) > MAX_PROVENANCE_PARAGRAPH_CHARS:
        return False
    patterns = (PG_MENTION, LEGAL_PHRASE, SCAN_SOURCE, EBOOK_MENTION)
    if is_header_block(paragraph) or any(pattern.search(paragraph) for pattern in patterns):
        return True
    return in_head and bool(CREDIT_OPENING.match(paragraph.lstrip()) or is_url_paragraph(paragraph))


def strip_provenance(text: str) -> str:
    """Drops paragraphs about the ebook (credits, PG notes, legal text, URL lists); credits only near the start."""
    kept = []
    for start, end in split_paragraphs(text):
        if not is_provenance_paragraph(text[start:end], in_head=start < HEAD_WINDOW_CHARS):
            kept.append(text[start:end])
    return "\n\n".join(kept)


def section_end(text: str, start: int, limit: int) -> int | None:
    """Offset of the next PG section break after start, or None when there is none within limit chars."""
    match = SECTION_BREAK.search(text, start, min(len(text), start + limit))
    return match.start() if match else None


def is_tail_note(text: str, start: int) -> bool:
    return len(text) - start <= TAIL_NOTE_MAX_CHARS and start >= TAIL_NOTE_MIN_POSITION_SHARE * len(text)


def strip_transcriber_notes(text: str) -> str:
    text = TRANSCRIBER_BRACKET.sub("", text)
    heading = next_note_heading(text, 0)
    while heading is not None:
        if is_tail_note(text, heading.start()):
            footnotes = FOOTNOTES_HEADING.search(text, heading.end())
            text = text[:heading.start()] + (text[footnotes.start():] if footnotes else "")
        else:
            text = text[:heading.start()] + text[note_end(text, heading.end()):]
        heading = next_note_heading(text, heading.start())
    return strip_transcriber_boxes(text)


def note_end(text: str, body_start: int) -> int:
    """End of a note that is not at the tail: the next section break, or an earlier asterism, picture or title line."""
    end = section_end(text, body_start, MAX_NOTE_CHARS)
    if end is None:
        paragraph_break = PARAGRAPH_BREAK.search(text, body_start)
        end = paragraph_break.start() if paragraph_break else len(text)
    stop = NOTE_STOP.search(text, body_start)
    return min(end, stop.start()) if stop is not None else end


def is_errata_list(text: str, body_start: int) -> bool:
    return ERRATA_ENTRY.search(text, body_start, body_start + ERRATA_ENTRY_WINDOW_CHARS) is not None


def next_note_heading(text: str, start: int) -> regex.Match | None:
    for heading in TRANSCRIBER_HEADING.finditer(text, start):
        if heading.group("errata") is None or is_tail_note(text, heading.start()) or is_errata_list(text, heading.end()):
            return heading
    return None


def strip_transcriber_boxes(text: str) -> str:
    """Drops ASCII-framed boxes (runs of | and +--- lines) that hold a transcriber's note or an errata list."""
    lines = text.split("\n")
    kept = []
    index = 0
    while index < len(lines):
        stop = index
        while stop < len(lines) and BOX_LINE.fullmatch(lines[stop]):
            stop += 1
        if stop == index:
            kept.append(lines[index])
            index += 1
            continue
        box = "\n".join(lines[index:stop])
        if not BOX_NOTE.search(box):
            kept.append(box)
        index = stop
    return "\n".join(kept)


def is_entry_line(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) <= MAX_ENTRY_CHARS or bool(PAGE_REFERENCE.search(stripped)) or bool(CHAPTER_LINE.fullmatch(line))


def non_blank_lines(block: str) -> list[str]:
    return [line for line in block.split("\n") if line.strip()]


def is_prose_paragraph(paragraph: str, min_lines: int) -> bool:
    lines = non_blank_lines(paragraph)
    if len(lines) < min_lines or any(PAGE_REFERENCE.search(line) or is_aligned(line) for line in lines):
        return False
    return not any(is_entry_line(line) for line in lines[:-1])


def prose_start(text: str, start: int, end: int, min_lines: int) -> int:
    """Start of the first prose paragraph in text[start:end], or end; a section break can miss the end of a list."""
    paragraph_start = start
    for match in PARAGRAPH_BREAK.finditer(text, start, end):
        if is_prose_paragraph(text[paragraph_start:match.start()], min_lines):
            return paragraph_start
        paragraph_start = match.end()
    return paragraph_start if is_prose_paragraph(text[paragraph_start:end], min_lines) else end


def navigation_end(text: str, heading: regex.Match) -> int | None:
    """End of the list under a navigation heading, or None when the block does not look like a list."""
    body_start = heading.end() + 1
    blank = LEADING_BLANK_LINES.match(text, body_start)
    if blank:
        body_start = blank.end()
    limit = max(MIN_NAVIGATION_LIMIT_CHARS, int(MAX_NAVIGATION_SHARE * len(text)))
    end = section_end(text, body_start, limit)
    if end is None:
        return None
    illustration_list = ILLUSTRATION_LIST.match(heading.group(0)) is not None
    end = prose_start(text, body_start, end, MIN_ILLUSTRATION_LIST_PROSE_LINES if illustration_list else MIN_PROSE_LINES)
    lines = non_blank_lines(text[body_start:end])
    if not lines or len(lines) > MAX_NAVIGATION_LINES:
        return None
    if illustration_list:
        return end
    if sum(1 for line in lines if is_entry_line(line)) < MIN_ENTRY_SHARE * len(lines):
        return None
    return end


def strip_navigation(text: str) -> str:
    """Drops front-matter contents and illustration lists that end at a section break, and a back-of-book index."""
    front_matter_end = int(NAVIGATION_START_SHARE * len(text))
    heading = CONTENTS_HEADING.search(text, 0, front_matter_end)
    while heading is not None:
        end = navigation_end(text, heading)
        if end is None:
            heading = CONTENTS_HEADING.search(text, heading.end(), front_matter_end)
            continue
        front_matter_end -= end - heading.start()
        text = text[:heading.start()] + text[end:]
        heading = CONTENTS_HEADING.search(text, heading.start(), front_matter_end)
    return strip_back_index(text)


def strip_back_index(text: str) -> str:
    """Drops each index after INDEX_START_SHARE of the text: its heading, a short note and its entries."""
    start = int(len(text) * INDEX_START_SHARE)
    heading = INDEX_HEADING.search(text, start)
    while heading is not None:
        end = index_end(text, heading.end())
        if end is not None:
            text = text[:heading.start()] + text[end:]
        heading = INDEX_HEADING.search(text, heading.start() if end is not None else heading.end())
    return text


def index_entries(text: str, start: int) -> list[tuple[int, bool]]:
    """(end offset, has a reference) of each entry after start; indented or lower-case lines continue an entry."""
    entries = []
    head_indent = 0
    previous_blank = True
    footnotes = FOOTNOTE_START.search(text, start)
    for line in INDEX_LINE.finditer(text, start, len(text) if footnotes is None else footnotes.start()):
        stripped = line.group(0).strip()
        if not stripped:
            previous_blank = True
            continue
        if INDEX_SUBHEADING.fullmatch(stripped):
            continue
        indent = len(line.group(0)) - len(line.group(0).lstrip())
        has_reference = INDEX_REFERENCE.search(stripped) is not None
        if entries and not previous_blank and (indent > head_indent or INDEX_CONTINUATION_START.match(stripped)):
            entries[-1] = (line.end(), entries[-1][1] or has_reference)
        else:
            entries.append((line.end(), has_reference))
            head_indent = indent
        previous_blank = False
    return entries


def index_end(text: str, body_start: int) -> int | None:
    """End of the index whose heading ends at body_start, or None when the entries do not carry references."""
    entries = index_entries(text, body_start)
    first = 0
    while (first < min(MAX_INDEX_NOTE_ENTRIES, len(entries)) and not entries[first][1]
           and entries[first][0] - body_start <= MAX_INDEX_NOTE_CHARS):
        first += 1
    stop = first
    while stop < len(entries):
        window = entries[stop:stop + INDEX_WINDOW_ENTRIES]
        window_references = sum(has_reference for _, has_reference in window)
        if not entries[stop][1] and window_references < MIN_INDEX_REFERENCE_SHARE * len(window):
            break
        stop += 1
    while stop > first and not entries[stop - 1][1]:
        stop -= 1
    references = sum(has_reference for _, has_reference in entries[first:stop])
    if references < MIN_INDEX_ENTRIES or references < MIN_INDEX_REFERENCE_SHARE * (stop - first):
        return None
    return entries[stop - 1][0]


def clean_pg_apparatus(text: str) -> str:
    """Runs on the raw text, because section breaks (runs of blank lines) mark where PG notes and lists end."""
    text = cut_trailer(cut_at_markers(text.replace("\r\n", "\n")))
    text = HEADER_FRAGMENT.sub("", text, count=1)
    text = strip_transcriber_notes(text)
    text = strip_navigation(text)
    return strip_provenance(text)


def is_aligned(line: str) -> bool:
    return ALIGNED_LINE.search(line.strip()) is not None


def wrap_width(paragraphs: list[list[str]]) -> int | None:
    """Wrap width of a hard-wrapped book: a high quantile of the lengths of non-final lines of multi-line paragraphs.

    Lines above MAX_WRAP_WIDTH are left out, so long table or manuscript lines cannot push the width out of range.
    """
    lengths = sorted(len(line) for lines in paragraphs for line in lines[:-1]
                     if len(line) <= MAX_WRAP_WIDTH and not is_aligned(line))
    if len(lengths) < MIN_WRAP_SAMPLE_LINES:
        return None
    width = lengths[int(WRAP_WIDTH_QUANTILE * (len(lengths) - 1))]
    return width if width >= MIN_WRAP_WIDTH else None


def is_end_stopped(lines: list[str]) -> bool:
    end_stopped = sum(1 for line in lines[:-1] if END_STOPPED_LINE.search(line))
    return end_stopped > MAX_END_STOPPED_SHARE * (len(lines) - 1)


def paragraph_width(lines: list[str], width: int) -> int:
    """The paragraph's own width when it is narrower than the book or a manuscript line is over MAX_WRAP_WIDTH."""
    if len(lines) < MIN_BLOCK_LINES:
        return width
    indents = {len(line) - len(line.lstrip()) for line in lines[1:]}
    block_width = max(len(line) for line in lines[:-1])
    if len(indents) > 1 or not (MIN_WRAP_WIDTH <= block_width <= MAX_PARAGRAPH_WIDTH):
        return width
    if min(indents) < MIN_BLOCK_INDENT and is_end_stopped(lines):
        return width
    return block_width if block_width > MAX_WRAP_WIDTH else min(width, block_width)


def is_full_line(line: str, next_line: str, width: int) -> bool:
    next_word = next_line.split(maxsplit=1)[0] if next_line.strip() else ""
    return bool(next_word) and len(line) + 1 + len(next_word) >= FULL_LINE_FILL * width


def is_wrapped_paragraph(lines: list[str], numbered: list[bool], width: int) -> bool:
    """Every line but the last is full; one short or aligned line marks verse, a play, a table or a letter.

    A line that had a margin number counts as full, because the number took the rest of its width.
    """
    if len(lines) < 2 or any(is_aligned(line) for line in lines):
        return False
    if len(lines) >= MIN_VERSE_CHECK_LINES and all(CAPITALIZED_LINE.match(line.strip()) for line in lines[1:]):
        return False
    if any(len(line) > max(width, MAX_WRAP_WIDTH) for line in lines[:-1]):
        return False
    short_lines = [index for index, (line, has_number, next_line) in enumerate(zip(lines, numbered, lines[1:]))
                   if not (has_number or is_full_line(line, next_line, width))]
    if len(short_lines) > MAX_SHORT_INTERNAL_LINES:
        return False
    return all(index > 0 and LOWER_CASE_START.match(lines[index + 1]) for index in short_lines)


def join_lines(lines: list[str]) -> str:
    text = lines[0].rstrip()
    for line in lines[1:]:
        # PG keeps real hyphens and "--" dashes at line ends; it does not split words with soft hyphens.
        text += ("" if text.endswith("-") else " ") + line.strip()
    return text


def strip_page_markers(text: str) -> str:
    """Runs before unwrap: a marker on its own line inside a paragraph would otherwise split the paragraph."""
    text = PAGE_MARKER_LINE.sub("", text)
    return EMPTY_LINES.sub("\n\n", PAGE_MARKER.sub("", text))


def unwrap_lines(text: str) -> str:
    """Joins the lines of paragraphs that are wrapped prose; every other paragraph keeps its line breaks."""
    paragraphs = [paragraph.split("\n") for paragraph in text.split("\n\n")]
    width = wrap_width(paragraphs)
    if width is None:
        return text
    return "\n\n".join(unwrap_paragraph(lines, width) for lines in paragraphs)


def split_paragraph_label(lines: list[str]) -> tuple[str | None, list[str]]:
    """A leading speaker label or sidenote run, and the body lines; the body keeps the first line's width."""
    if len(lines) > 2 and LABEL_LINE.fullmatch(lines[0]):
        return lines[0], lines[1:]
    text = "\n".join(lines)
    sidenotes = LEADING_SIDENOTES.match(text)
    if sidenotes is None or sidenotes.end() == len(text):
        return None, lines
    line_start = text.rfind("\n", 0, sidenotes.end()) + 1
    body = text[sidenotes.end():].split("\n")
    body[0] = " " * (sidenotes.end() - line_start) + body[0]
    return " ".join(text[:sidenotes.end()].split()), body


def unwrap_paragraph(lines: list[str], width: int) -> str:
    label, body = split_paragraph_label(lines)
    without_margin_numbers = [MARGIN_NUMBER.sub("", line) for line in body]
    numbered = [stripped != line for stripped, line in zip(without_margin_numbers, body)]
    if not is_wrapped_paragraph(without_margin_numbers, numbered, paragraph_width(without_margin_numbers, width)):
        return "\n".join(lines)
    joined = join_lines(without_margin_numbers)
    return joined if label is None else label + "\n" + joined.lstrip()


def compose(letter: str, mark: str) -> str:
    return unicodedata.normalize("NFC", letter + mark)


def illustration(match: regex.Match) -> str:
    text, caption = match.string, match.group(1).strip()
    if DROP_CAP.fullmatch(caption) and DROP_CAP.match(text, match.end()):
        return (" " if match.group(0)[0].isspace() else "") + caption
    if IMAGE_KIND_CAPTION.fullmatch(caption):
        return ""
    on_own_lines = OWN_LINE_BEFORE.search(text, max(0, match.start() - 200), match.start()) is not None
    on_own_lines = on_own_lines and OWN_LINE_AFTER.match(text, match.end()) is not None
    return caption if on_own_lines else ""


def foreign_script(match: regex.Match) -> str:
    # Dropping a transliteration leaves a hole in the sentence; without its accent codes it reads as romanized text.
    return TRANSLITERATION_MARK.sub("", match.group(1)).strip()


def normalize_markup(text: str) -> str:
    text = ILLUSTRATION.sub(illustration, text)
    text = HTML_TAG.sub("", text)
    text = BLANK_PAGE_LINE.sub("", text)
    text = FOREIGN_SCRIPT.sub(foreign_script, text)
    text = BRACED_SCRIPT_DIGITS.sub(lambda match: match.group(2).translate(SCRIPT_DIGITS[match.group(1)]), text)
    text = BRACED_SCRIPT_LETTERS.sub(r"\1", text)
    text = DIACRITIC_CODE.sub(lambda match: compose(match.group(2), COMBINING_MARKS[match.group(1)]), text)
    text = BELOW_DIACRITIC_CODE.sub(lambda match: compose(match.group(1), BELOW_MARKS[match.group(2)]), text)
    for code, letter in LIGATURES.items():
        text = text.replace(code, letter)
    text = ITALIC.sub(r"\1", text)
    for bold in (BOLD, TILDE_BOLD):
        if len(bold.findall(text)) >= MIN_BOLD_SPANS:
            text = bold.sub(r"\1", text)
    text = ASTERISM_LINE.sub("* * *", text)
    text = regex.sub(r"[ \t]+(?=\n|\Z)", "", text)
    return EMPTY_LINES.sub("\n\n", text).strip()


def digit_share(text: str) -> float:
    return sum(character.isdigit() for character in text) / max(1, len(text))


def alphabetic_word_share(text: str) -> float:
    words = text.split()
    return sum(1 for word in words if LETTER.search(word)) / max(1, len(words))


def is_book_text(document: dict) -> bool:
    text = document["text"]
    if len(WORD.findall(text)) < MIN_WORDS:
        return False
    if len(text) < MAX_STUB_CHARS and STUB_MEDIA.search(text):
        return False
    if digit_share(text) > MAX_DIGIT_SHARE or alphabetic_word_share(text) < MIN_ALPHABETIC_WORD_SHARE:
        return False
    if PG_CATALOGUE_TITLE.match(document.get("metadata", {}).get("title") or ""):
        return False
    return not PG_MENTION.search(text)


def english_share(identifier: LanguageIdentifier, text: str) -> float:
    """Share of letters in segments that lid.176 labels English with at least MIN_ENGLISH_CONFIDENCE."""
    segments = language_segments(text)
    if not segments:
        return 0.0
    labels, scores = identifier.model.predict([fasttext_input(segment) for segment in segments], k=1)
    english_letters = total_letters = 0
    for segment, segment_labels, segment_scores in zip(segments, labels, scores):
        letters = len(LETTER.findall(segment))
        total_letters += letters
        if segment_labels[0] == "__label__en" and segment_scores[0] >= MIN_ENGLISH_CONFIDENCE:
            english_letters += letters
    return english_letters / total_letters


# Findings "duplicate editions": containment of a part volume in its "Complete" edition (6.2% of docs at >= 0.8),
# measured on 1/64 of the word 5-gram shingles with posting lists capped at 50 docs; S2ORC's containment metric.
MIN_CONTAINMENT = 0.8
# 1/16 sampling with at least 20 sampled shingles covers texts from about 320 words, e.g. single tales in collections.
SHINGLE_SAMPLE_MODULUS = 16
MIN_SAMPLED_SHINGLES = 20
# Shingles in more than 50 documents are stock phrases (findings: 409 such hashes); they are left out of both sides.
MAX_POSTING_DOCS = 50
# Pair counts go to disk partitions by pair code, so one partition at a time is in RAM during the merge.
PAIR_PARTITIONS = 64
PAIR_RECORD = np.dtype([("pair", np.int64), ("shared", np.int32)])


class SampledShingleIndexer:
    def __init__(self, hasher: MinHasher, directory: Path, shard_index: int):
        self.hasher = hasher
        self.keys = KeyIndexWriter(shard_prefix(directory, shard_index), sort_by_key=True)

    def add(self, local_id: int, document: dict) -> None:
        words = dedup_words(document["text"])
        if not words:
            return
        shingles = self.hasher.shingles(words)
        self.keys.add(local_id, shingles[shingles % np.uint64(SHINGLE_SAMPLE_MODULUS) == 0])

    def close(self) -> None:
        self.keys.close()


def group_pairs(global_ids: np.ndarray, group_starts: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """Codes smaller_id * 2^32 + larger_id of every pair of documents inside each group of equal keys."""
    pieces = []
    for size in np.unique(group_sizes):
        members = global_ids[group_starts[group_sizes == size][:, None] + np.arange(size)]
        first, second = np.triu_indices(size, k=1)
        low = np.minimum(members[:, first], members[:, second])
        high = np.maximum(members[:, first], members[:, second])
        pieces.append(((low << 32) | high).reshape(-1))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


class ContainmentDedup(Deduplicator):
    """Drops a document when at least MIN_CONTAINMENT of its sampled word 5-gram shingles are in one larger document.

    A part volume has low Jaccard with its omnibus but containment near 1.0 (findings), so a Jaccard test misses it;
    containment >= 0.8 also covers every pair with Jaccard >= 0.8. Pair counts are spilled to disk partitions.
    """

    def __init__(self, name: str = "near_dedup"):
        super().__init__(name)
        self.hasher = MinHasher()

    def open_indexer(self, directory: Path, shard_index: int) -> SampledShingleIndexer:
        return SampledShingleIndexer(self.hasher, directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        directory = Path(directory)
        size = sum(shard_counts)
        sampled = np.zeros(size, dtype=np.int64)
        for shard_index, offset in enumerate(shard_offsets(shard_counts)):
            if shard_counts[shard_index]:
                _, local_ids = load_index(directory, shard_index, mmap=True)
                sampled[offset:offset + shard_counts[shard_index]] = np.bincount(
                    np.asarray(local_ids), minlength=shard_counts[shard_index])
        stock = np.zeros(size, dtype=np.int64)
        partitions = [open(directory / f"pairs_{index:02d}.bin", "wb") for index in range(PAIR_PARTITIONS)]
        try:
            for keys, global_ids in iter_key_buckets(directory, shard_counts, MAX_BUCKET_ENTRIES):
                if dropped is not None:
                    live = ~dropped[global_ids]
                    keys, global_ids = keys[live], global_ids[live]
                self._spill_bucket(keys, global_ids, stock, partitions)
        finally:
            for partition in partitions:
                partition.close()
        usable = sampled - stock
        rank = np.lexsort((-np.arange(size), sampled))
        rank_of = np.empty(size, dtype=np.int64)
        rank_of[rank] = np.arange(size)
        smaller_pieces, larger_pieces = [], []
        for index in range(PAIR_PARTITIONS):
            path = directory / f"pairs_{index:02d}.bin"
            records = np.fromfile(path, dtype=PAIR_RECORD)
            path.unlink()
            pairs, inverse = np.unique(records["pair"], return_inverse=True)
            shared = np.bincount(inverse.reshape(-1), weights=records["shared"]).astype(np.int64)
            first, second = pairs >> 32, pairs & 0xFFFFFFFF
            smaller = np.where(rank_of[first] < rank_of[second], first, second)
            larger = np.where(rank_of[first] < rank_of[second], second, first)
            contained = (usable[smaller] >= MIN_SAMPLED_SHINGLES) & (shared >= MIN_CONTAINMENT * usable[smaller])
            smaller_pieces.append(smaller[contained])
            larger_pieces.append(larger[contained])
        smaller, larger = np.concatenate(smaller_pieces), np.concatenate(larger_pieces)
        # Each contained document points to its highest-ranked container: the last one after this sort.
        order = np.lexsort((rank_of[larger], smaller))
        smaller, larger = smaller[order], larger[order]
        is_last = np.ones(len(smaller), dtype=bool)
        is_last[:-1] = smaller[1:] != smaller[:-1]
        with DuplicateOfWriter(directory, size) as duplicate_of:
            duplicate_of[smaller[is_last]] = larger[is_last]
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "contained_pairs": int(len(smaller))}

    @staticmethod
    def _spill_bucket(keys: np.ndarray, global_ids: np.ndarray, stock: np.ndarray, partitions: list) -> None:
        if len(keys) == 0:
            return
        order = np.lexsort((global_ids, keys))
        keys, global_ids = keys[order], global_ids[order]
        starts = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
        sizes = np.diff(np.append(starts, len(keys)))
        is_stock = sizes > MAX_POSTING_DOCS
        np.add.at(stock, global_ids[np.repeat(is_stock, sizes)], 1)
        shared_groups = (sizes >= 2) & ~is_stock
        pairs, shared = np.unique(group_pairs(global_ids, starts[shared_groups], sizes[shared_groups]),
                                  return_counts=True)
        records = np.empty(len(pairs), dtype=PAIR_RECORD)
        records["pair"], records["shared"] = pairs, shared
        partition_of = pairs % PAIR_PARTITIONS
        for index, partition in enumerate(partitions):
            records[partition_of == index].tofile(partition)


# PG books predate IP addresses: the only IP match in the 3,000-book sample was a museum number ("No. 88.8.8.18").
PII_KINDS = ("email", "phone")


def build_pipeline() -> Pipeline:
    identifier = LanguageIdentifier()
    return Pipeline("project_gutenberg", [
        TextMap("pg_apparatus", clean_pg_apparatus),
        TextMap("normalize", normalize_text),
        TextMap("page_markers", strip_page_markers),
        TextMap("unwrap", unwrap_lines),
        TextMap("markup", normalize_markup),
        Filter("book_text", is_book_text),
        Filter("english", lambda document: english_share(identifier, document["text"]) >= ENGLISH_MIN_SHARE),
        ExactDedup(),
        ContainmentDedup(),
        TextMap("pii", lambda text: mask_pii(text, kinds=PII_KINDS)),
    ])


if __name__ == "__main__":
    run_cli(build_pipeline)
