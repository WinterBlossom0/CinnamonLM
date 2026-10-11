"""Cleaning pipeline for Common Pile libretexts (MindTouch page text).

python -m data_pipeline.sources.libretexts --input DIR (--in-place | --sample-out DIR --limit N) --meta-out DIR
"""

import regex

from ..common.cli import run_cli
from ..common.dedup import ExactDedup, NearDedup
from ..common.langid import LanguageIdentifier
from ..common.licence import load_licence_filter
from ..common.pii import mask_pii
from ..common.pipeline import Filter, Pipeline, Step, TextMap
from ..common.text import normalize_text

TERMINAL_PUNCTUATION = (".", "?", "!", ":")
# The espanol host prints the same MindTouch chrome with Spanish labels (4 pages).
SKIP_LINKS = ("Skip to main content", "Saltar al contenido principal")
# Findings libretexts.json "site chrome": a MindTouch UI dump ends at the breadcrumb "Page ID" number.
MINDTOUCH_BLOCK = regex.compile(
    r"\A(?:Skip to main content|Saltar al contenido principal)\n.*?"
    r"(?:Expand/collapse global location|Expandir/contraer ubicación global)\n.*?Page ID\n\d+\n?",
    regex.DOTALL,
)
INLINE_HEADER = regex.compile(
    r"\A(?P<title>.{0,300}?) (?:Last updated|Última actualización) (?:Save as PDF|Guardar como PDF)(?: Page ID \d+)?"
)
SAVE_AS_PDF_LINES = frozenset({"- Save as PDF", "Save as PDF", "- Guardar como PDF", "Guardar como PDF"})
HEADER_LINES = SAVE_AS_PDF_LINES | {
    "- Last updated", "Last updated", "- Última actualización", "Última actualización", "Page ID", *SKIP_LINKS,
    "selected template will load here", "This action is not available.", "Next page »", "« Previous page",
}
BYLINE_MAX_CHARS = 100
# Author lists and consortium bylines ("... via ASCCC ...") run past 100 chars; the longest measured has 140.
BYLINE_LONG_MAX_CHARS = 250
BYLINE_MAX_LINES = 3
INSTITUTION = regex.compile(r"\b(?:College|University|Universit|Institute|School|Academy|via|Learning|OpenStax)\b")
PAGE_ID_BULLET = regex.compile(r"\A- \d+\Z")
NAME_CONNECTORS = r"(?:of|the|and|&|for|at|in|on|de)"
INSTITUTION_WORD = rf"(?:\(?[A-Z][\w.'’&,()-]*|{NAME_CONNECTORS})"
PUBLISHER_WORD = rf"(?:[A-Z][\w.'’&-]*|[\w-]+\.(?:org|com|net)|{NAME_CONNECTORS})"
# "<institution> via <publisher>": a capitalised word before a lowercase word starts the body ("The following ...").
INSTITUTION_VIA = regex.compile(
    rf"\A{INSTITUTION_WORD}(?: {INSTITUTION_WORD}){{0,15}} via {PUBLISHER_WORD}"
    rf"(?: (?![A-Z][\w'’-]* (?:{NAME_CONNECTORS} )?(?!{NAME_CONNECTORS}\b)[a-z]){PUBLISHER_WORD})*"
    r"(?: \([A-Z]{2,}\))?(?= |\Z)"
)
# Stand-ins for widgets, embeds and Lumen quizzes that the export left out (findings "missing media").
PLACEHOLDER_LINE = regex.compile(
    r"\AQuery \\\(\\PageIndex\{[^{}]*\}\\\)\Z|\A\(Powered by DESMOS\)\Z"
    r"|\A(?:An?|One or more|The) [\w ]{0,40}elements? (?:has|have|is) (?:been )?excluded from (?:this|the print) "
    r"version of the text\. You can (?:view|listen to) (?:it|them) online here:.*\Z"
    r"|\AAnswers to interactive elements are given within the interactive element\.\Z"
    r"|\AAnswer the question\(s\) below to see how well you understand the topics covered in the previous section\..*\Z"
    r"|\AUse this quiz to check your understanding and decide whether to \(1\) study the previous section.*\Z"
    r"|\Ahttps://assessments\.lumenlearning\.com/\S*\Z|\ATemplate:\S+\Z"
)
MINDTOUCH_TAGS = regex.compile(r'\A\[\s*"[\w-]+[:@][^"]*"(?:\s*,\s*"[^"]*")*\s*\]?\Z')

TOC_ENTRY = regex.compile(r"\A(?:- )?\d{1,3}(?:\.\d{1,3})*[A-Z]?: .{1,150}\Z")
TOC_MARKER = regex.compile(r"(?:(?<=\s)|\A)\d{1,3}(?:\.\d{1,3})*[A-Z]?: ")
TOC_SEGMENT_MAX_CHARS = 150
# A child entry with its summary: "14.1: Impulse-Momentum Equations ... Includes example problems."
TOC_SUMMARY_MAX_CHARS = 400
TOC_SUMMARY_MAX_SENTENCES = 2
SENTENCE_END = regex.compile(r"[.!?](?=\s|\Z)")
TOC_RUN_MIN_LINES = 3
PAGE_NUMBER = regex.compile(r"\A(\d{1,3}(?:\.\d{1,3})*[A-Z]?): ")

ATTRIBUTION_HEADINGS = frozenset({
    "contributors and attributions", "contributors", "contributor", "attributions", "attribution",
    "licenses and attributions", "licenses and attribution", "cc licensed content, original",
    "cc licensed content, shared previously", "cc licensed content, previously shared", "cc licensed content, specific attribution",
    "all rights reserved content", "public domain content", "contributions and attributions",
    "contributions and affiliations", "contributors and affiliations",
    "remixed and/or curated from the following works", "contribuyentes y atribuciones", "contribuyentes",
    "license and attributions", "adaptions", "adaptations",
})
# Lumen blocks list "Adaptions: Reformatted ..." under "Licenses and Attributions:"; the line counts only inside a block.
ADAPTATION_NOTE = regex.compile(r"\A(?:- )?Adapt(?:at)?ions? ?:", regex.IGNORECASE)
# "Adaptions:" (a Lumen misspelling) heads only adaptation notes, so it goes outside a block too; "Adaptations:" may be a
# biology heading.
ADAPTIONS_NOTE = regex.compile(r"\A(?:- )?Adaptions ?:")
# Some pages put the attribution block above the body; these headings start the body again.
BODY_HEADINGS = frozenset({
    "key points", "key terms", "learning objectives", "summary", "glossary", "references", "footnotes", "exercises",
    "review questions", "introduction",
})
# A heading followed by a prose paragraph ends an attribution block ("The Human Organ Systems\nInformation\n...").
BODY_PROSE_MIN_CHARS = 200
BODY_HEADING_RUN = 2
LICENCE_LINE_MAX_CHARS = 400
# Derivation notes that start with "This work" list every source work; all 1,422 measured notes are under 600 chars.
DERIVATION_NOTE = regex.compile(r"\A(?:- )?This work\b")
DERIVATION_NOTE_MAX_CHARS = 600
LICENCE_LINE = regex.compile(
    r"\b(?:Provided by|Authored by|Located at) ?: "
    r"|\bLicen[cs]e ?: ?(?:CC|Public Domain|All Rights Reserved|GNU|Copyright|Other\b)|\bLicen[cs]e Terms ?:"
    r"|\blicensed (?:under |as )?(?:a |the )?(?:CC|Creative Commons|GNU|Public Domain)"
    r"|\bused under (?:an? |the )*CC\b"
    r"|\(\s?CC[ -]BY[^;)]*;[^)]* via "
    r"|\b(?:Curated|Modified|Adapted|Remixed) and (?:authored|adapted|curated|remixed) by\b"
    r"|\bis in the Public Domain\W*\Z"
    r"|\AThis page titled .{1,300} is shared under"
    r"|\bThis work\b.{0,300}?\bis a derivative of\b"
    r"|\b(?:Textbook content|Content) produced by\b|\bOriginal content (?:by|via|from) OpenStax\b"
    r"|\(CC[ -]BY[^)]*?(?:;\s*|\)\.?\s*)(?:Access|Download) for free at"
    r"|\bcopyrighted (?:under|with) an? (?:Creative Commons|CC)"
    r"|\A\W*CC[ -]?BY(?:[ -](?:SA|NC|ND))*(?:[ -]\d\.\d)?\W*\Z"
    r"|©[^.]{0,80}\.\s*(?:CC[ -]?BY|Public Domain)\b",
    regex.IGNORECASE,
)
# OpenStax author lists ("Connie Rye (...), ... among other contributing authors.") run to 445 chars.
AUTHOR_CREDIT_MAX_CHARS = 450
AUTHOR_CREDIT = regex.compile(r"\b(?:with (?:many )?(?:other )?|among other )contributing authors\b", regex.IGNORECASE)
THUMBNAIL_LINE = regex.compile(r"\A(?:- )?(?:Thumbnail(?: image)?\s*(?::|-|by\b)|Chapter thumbnail\b)")
# A credit inside a sentence ("Escucha a Cristina (by Orlando R. Kelm, licensed under CC BY 3.0) y") loses its span.
PARENTHETICAL_CREDIT = regex.compile(
    r" ?\((?:[^()]{0,200}?, )?(?:licensed|used) under (?:a |the )?(?:CC|Creative Commons)[^()]{0,60}\)",
    regex.IGNORECASE,
)
# Anthology pages glue title, licence and first line of the work: "Ode to the West Wind License: Public DomainO wild".
INLINE_LICENCE = regex.compile(r"(?m)^(.{1,200}?) License: Public Domain ?(.{40,})$")
ALPHANUMERIC = regex.compile(r"[^\W_]")

# Also malformed forms: "\(PageIndex{3}\)", "\(\ PageIndex {9 c}\ )", "\(\PageIndex{3\)".
PAGE_INDEX = regex.compile(r"\\ ?\(\s*[\\|]? ?PageIndex ?\{\s*([^{}\\]*?)\s*\}?\s*\\ ?\)")
# Forms that no pattern resolves ("Table \PageIndex3\PageIndex3.2", "\(\PageIndex{Exhibit 14.5."): the macro name goes.
RESIDUAL_PAGE_INDEX = regex.compile(r"\\\(\s*\\?PageIndex\s*\{|\\?PageIndex\s*\{?")
BARE_PAGE_INDEX = regex.compile(r"\\PageIndex\{\s*([^{}]*?)\s*\}")
# "\\nonumber_{4}": an underscore after the command is a word character, so \\b does not end it.
LATEX_LAYOUT = regex.compile(r"\\nonumber(?![A-Za-z])|\\notag(?![A-Za-z])|\\label\{[^{}]*\}|\\[hv]space\*?\{[^{}]*\}")
LATEX_LINE_SKIP = regex.compile(r"(\\\\)\[-?\d*\.?\d+(?:pt|em|ex|mm|cm)\]")
# Findings "broken LaTeX": machine translation inserted a space after the backslash of delimiters and commands.
SPACED_COMMAND = regex.compile(
    r"\\ (?=(?:dfrac|frac|begin|end|sqrt|left|right|cdot|text|mathrm|mathbf|div|times|pi|alpha|beta|"
    r"theta|lambda|mu|sigma|Delta|delta|quad|qquad|le|ge|ne|neq|pm|infty|sum|int|lim|log|ln|sin|cos|tan|"
    r"overline|underline|hat|vec|bar)\b)"
)
# "\ (" is also a LaTeX control space ("\equiv 0\ \ (\text{mod } p)"), so only paired or missing delimiters change.
MATH_DELIMITER = regex.compile(r"(?<!\\)\\( ?)([()\[\]])")
OPENERS = {"(": ")", "[": "]"}
NOT_BEFORE_SPACED_OPENER = frozenset(" \t\n\\+-=*/^_|<>&{(")
LATEX_CROSSREF = regex.compile(r" ?\( ?\[[A-Za-z][\w:.\-]*\] ?\)| ?\[link\]")

# Raw HTML leaked into 184 pages (480 lines): CSS styles, image attributes, embeds and bare table tags.
HTML_RESIDUE_LINE = regex.compile(
    r'data-(?:quail-id|mt-width|mt-height)="|\bstyle="[^"]*[:;]|<iframe\b|\bclass="[^"]*"|title="Rendered by QuickLaTeX'
    r"|property get \[Map MindTouch"
)
# Attribute runs cut from their tag ('degrees." src="/@api/deki/files/97446/a.png">'); a run inside a shown tag stays.
ORPHAN_ATTRIBUTES = regex.compile(r'"?\s*(?:\b[\w-]+="[^"]*"\s*)+/?>|(?<=\|\s?)\d\\\)">')
# What is left of an image alt text after its attributes go: 'rdquo;.rdquo;.'.
ENTITY_RESIDUE_LINE = regex.compile(r"\A(?:&?(?:[lr][ds]quo|quot|amp|nbsp);[.,\"]?\s*)+\Z")
DOUBLED_TAG = regex.compile(r"<</?(?:td|em|strong|span|p)>|(?<=\A|\s)</?(?:td|em)>(?=\s|\Z)")
TAG_ONLY_LINE = regex.compile(r"\A(?:\s*</?[a-z][a-z0-9]*\b[^<>]*>)+\s*\Z")
LINK_TAG = regex.compile(r'<a\s+href="[^"]*"\s*>\s*|\s*</a>')

# MindTouch inserts this editor prompt into new pages; 248 lines are only the prompt, a few have it inline.
TEMPLATE_PROMPT = "Add texts here. Do not delete this text first."
LUMEN_SHORTCODE = regex.compile(r"\[ohm_question[^\]]*\].*?\[/ohm_question\]|\[/?(?:hidden-answer|reveal-answer)[^\]]*\]")
FOOTNOTE_RETURN = regex.compile(r" ?↵")
EMPTY_TABLE_LINE = regex.compile(r"\A\|[\s|]*\Z")
# A bare truncated URL ends before the sentence's closing punctuation; one adjoining space goes with it.
TRUNCATED_URL_BODY = (
    r"(?:\(\s*(?:https?://|www\.)\S*?\.\.\.\S*?\s*\)|(?:https?://|www\.)\S*?\.\.\.\S*?(?=[).,;:\]]*(?:\s|\Z)))"
)
TRUNCATED_URL = regex.compile(rf" {TRUNCATED_URL_BODY}|(?m:^){TRUNCATED_URL_BODY} ?|{TRUNCATED_URL_BODY}")
SPACE_BEFORE_PUNCTUATION = regex.compile(r"(?m)(?<=[\p{L})\]”’\"']) ([,;:])(?=\s)|(?<=[\p{Ll})”’\"]) (\.)(?=\s|$)")
SPLIT_ORDINAL = regex.compile(r"\b(\d+) (st|nd|rd|th)\b")

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og".split()
)
# Symbols that are also English words ("In 10 years", "At 20 °C") join only inside a longer formula ("Be 3 Al 2").
WORD_SYMBOLS = frozenset({"I", "In", "As", "At", "Be", "He", "No", "Am", "Pa", "Re", "Ho", "Es", "Ga", "Os", "La",
                          "Ti", "Mo", "Co", "Ba", "Bi", "Ne", "Pr", "Sc"})
# Prefixes: bond, bracket, arrow, hydrate dot or function ("(–NH 2 )", "·H 2 O", "ln[H 2 O2]"); "(en)" is a ligand.
FORMULA_PREFIX = r"(?:[–\-=()\[\]·←→⟶⇄⇌]*|(?:[a-z]{1,3}|\p{Greek})[(\[])"
FORMULA_PIECE = regex.compile(
    rf"\A{FORMULA_PREFIX}(?:(?:[A-Z][a-z]?|\([a-z]{{2,4}}\))\d{{0,2}}[)\]]*[–\-=]?[(\[]*)+(?P<punctuation>[.,;:]*)\Z"
)
DIGIT = regex.compile(r"\d")
# "CC-BY 4.0" reads as carbon, boron and yttrium.
LICENCE_CODE = regex.compile(r"CC[ -]?BY")
BOND = regex.compile(r"\A[–\-=]|=")
ELEMENT_SYMBOL = regex.compile(r"[A-Z][a-z]?")
CLOSING_PIECE = regex.compile(r"\A[)\]]+\Z")
COUNT_PIECE = regex.compile(r"\A\d{1,2}([.,;:)\]]*)\Z")
CHARGE_PIECE = regex.compile(r"\A\d{0,2}[+−–]([.,;:)\]]*)\Z")
FLATTENED_SUBSCRIPT = regex.compile(r"[A-Z][a-z]?[)\]]* \d")
# The export wrote <sup> orbital occupancies as "1s 2 2s 2" and hybrid orbitals as "sp 3 d 2".
ORBITAL_OCCUPANCY = regex.compile(r"(?<![\w.])([1-7][spdf]) (\d{1,2})\b")
HYBRID_ORBITAL = regex.compile(r"\bsp ([23])(?: (d)(?: ([12]))?)?\b")
# 70 pages print their summary right above an identical first paragraph; Gopher (App. A) also counts repeated lines.
REPEATED_LINE_MIN_CHARS = 40
COLLAPSIBLE_LABEL = regex.compile(r"\A- ((?:Answer|Solution|Hint|Proof)(?: [a-z0-9]{1,3})?)\Z")
# Findings "text broken at removed inline elements"; glossaries share the shape, so fragments need a sentence context.
INLINE_BEFORE = regex.compile(
    r"(?:\b(?:a|an|the|of|to|in|on|at|by|for|with|from|as|is|are|was|were|be|and|or|but|that|than|called|its|"
    r"their|this|these)|[(\[“‘]|(?<!\S)[\"'])\Z",
    regex.IGNORECASE,
)
# Code listings ("string = \"Hello\"\nprint(...)") share the shape; statements join only as fragments of prose.
CODE_LINE = regex.compile(
    r"\A(?:[A-Za-z_][\w.$\[\]'\"]*\s*(?:<-|[-+*/%]?=(?!=))\s*\S|[A-Za-z_][\w.$]*\s*\(.*\)\s*;?\s*(?:#.*)?\Z"
    r"|(?:#|//|%|>\s)|.*[;{}]\s*\Z)"
)
INLINE_AFTER = regex.compile(
    r"\A(?:[,.;:)\]”\"’!?]|(?:is|are|was|were|and|or|of|to|in|for|that|which|with|as|by|from|has|have|can|will)\b)"
)
# A footnote marker line ("cunning,\n[5]\nof a self-conceit,") belongs above; counted, it would mark verse exploded.
FOOTNOTE_MARKER = regex.compile(r"\A\[\d{1,3}\]\Z")
FRAGMENT_MAX_WORDS = 4
FRAGMENT_SYMBOL_MAX_CHARS = 2
SENTENCE_START_WORDS = frozenset({"A", "I"})
HEADING_MAX_WORDS = 6
HEADING_CONNECTORS = frozenset({
    "a", "an", "and", "at", "for", "in", "of", "on", "or", "the", "to", "vs", "with", "/", "&", "-",
})
OPENING_MARKS = ("(", "[", "“", "‘")
EXPLODED_MIN_FRAGMENTS = 3
CONTINUATION_PUNCTUATION = regex.compile(r"\A(?:[,;:)\]]|\.(?:\s|\Z))")
BLOCK_LINE_STARTS = ("- ", "|", "\\")
# The value of an exploded attribution field ("Located at\n:\nhttps://...") belongs on the label's line.
ATTRIBUTION_LABEL_END = regex.compile(
    r"(?:Provided by|Authored by|Located at|Licen[cs]e(?: Terms)?|Project|Adapted by) ?:\Z"
)
WORD = regex.compile(r"\p{L}+")

# Gopher (arXiv:2112.11446 App. A) keeps documents of at least 50 words.
MIN_WORDS = 50
# FineWeb short-line rule (arXiv:2406.17557 §3.6); its 0.67 cut removed poems and exercises here, 0.97 only tables.
SHORT_LINE_CHARS = 30
MAX_SHORT_LINE_FRACTION = 0.97
# lid.176 labels LaTeX as English, which lifted Spanish-language algebra pages to 0.3-0.5 English share.
LATEX_MATH = regex.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$", regex.DOTALL)
# Devanagari words alternate letters with combining marks.
PROSE_WORD = regex.compile(r"\p{L}[\p{L}\p{M}]")
LETTER = regex.compile(r"\p{L}")
# LID fails on short text (Caswell et al. arXiv:2010.14571 §3); pages with less prose are tables or formulas.
MIN_PROSE_LETTERS = 100
# calibration.json: 0.3-0.5 holds English lessons on Spanish, Italian, Arabic, Hindi; below 0.3 foreign text dominates.
ENGLISH_MIN_SHARE = 0.3
# espanol.libretexts.org pages are Spanish translations; a bilingual exercise page there has an English share of 0.37.
ESPANOL_HOST = "espanol.libretexts.org"
ESPANOL_MIN_SHARE = 0.5
# Bare key-term pages (World History 17.4.04: 60 terms, 0.94 short lines) pass MAX_SHORT_LINE_FRACTION.
TERM_LIST_TITLE = regex.compile(r"key[ _-]?terms|key[ _-]?words|glossary", regex.IGNORECASE)
TERM_LIST_MIN_SHORT_FRACTION = 0.9
# Every IPv4 match read in LibreTexts was a section, figure or EC number ("13.2.4.2") or a networking example.
PII_KINDS = ("email", "ipv6", "phone")

# Line ends where a split licence statement continues ("is licensed under a\nCreative Commons ...").
CONTINUED_LINE_END = regex.compile(
    r"(?:\b(?:a|an|the|by|under|of|is|are|was|and|or|to|at|from|with|in|for|as|licensed|available)|[,(])\Z",
    regex.IGNORECASE,
)
ATTRIBUTION_FIELD = regex.compile(r"\A(?:(?:Provided|Authored|Adapted) by\b|Located at ?:|Licen[cs]e(?: Terms)? ?:)")
ATTRIBUTION_ITEM_MAX_CHARS = 300
# Length-preserving rewrites ("BY–NC–ND", "BY-SA-NC", "Exhibit 2.4") so the filter's strip spans fit the original text.
LICENCE_CODE_DASH = regex.compile(r"(?<=\b(?:BY|NC|ND|SA))[–—](?=(?:BY|NC|ND|SA)\b)")
SA_BEFORE_NC = regex.compile(r"\bBY-SA-NC\b")
EXHIBIT_LABEL = regex.compile(r"(?<![A-Za-z])Exhibit(?= \d)")
# User decision: third-party text used with permission drops the page; same-length stand-ins for unknown verbs.
PERMISSION_VERB = regex.compile(
    r"\b(?:remixed|modified|taken|retrieved)\b"
    r"(?=(?:[\w,'’-]+\s+|[,\s]+){0,5}?with\s+(?:his\s+|her\s+|their\s+)?permission\b)",
    regex.IGNORECASE,
)
PERMISSION_VERB_STAND_IN = {"remixed": "adapted", "modified": "adapted ", "taken": "used ", "retrieved": "reprinted"}
PERMISSION_PRONOUN = regex.compile(r"(?<=\bwith\s)(?:his|her|their)(?=\s+permission\b)", regex.IGNORECASE)
# A credit in the two lines under a caption is a media credit ("Figure 7.9 ...\n...\nTitle – CC BY-NC 2.0.").
CAPTION_CREDIT_LINES = 2
CAPTION_CREDIT_MAX_CHARS = 300
CAPTION_START = regex.compile(
    r"\A\W*(?:Figure|Fig\.|Table|Exhibit|Image|Photo)\s*(?:\d|\\\(\\PageIndex|[IVX]+\b)", regex.IGNORECASE
)
# A footnote naming an NC/ND text source makes the page non-commercial; one naming media is a credit to strip.
NC_SOURCE_NOTE = regex.compile(
    r"\A\[\d+\] .{1,300}?\bis licensed under (?:a |the )?"
    r"(?:CC[ -]BY[ -](?:SA[ -])?N[CD]|Creative Commons Attribution[ -]No)"
)


def page_lines(text: str) -> list[str]:
    return text.split("\n")


def join_lines(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def ends_sentence(line: str) -> bool:
    return line.rstrip(" \"'”’)]").endswith(TERMINAL_PUNCTUATION)


def is_heading_like(line: str) -> bool:
    return len(line.split()) <= 8 and line[:1].isupper() and not ends_sentence(line)


def starts_new_paragraph(previous: str, line: str) -> bool:
    """Exploded pages put every inline element on its own line; only these breaks are real paragraph breaks."""
    first = line[:1]
    if ends_sentence(previous) and (first.isupper() or first.isdigit() or first in "\"“‘'(["):
        return True
    return is_heading_like(previous) and first.isupper() and len(line) >= 40


def rejoin_exploded(text: str) -> str:
    """Rejoins the inline fragments of a page whose HTML elements were each written on a separate line.

    A lone "-" line marks a block element there, so it always ends the current paragraph.
    """
    paragraphs: list[str] = []
    current = ""
    for line in page_lines(text):
        stripped = line.strip()
        if stripped in ("-", ""):
            if current:
                paragraphs.append(current)
            current = ""
            continue
        if not current:
            current = stripped
        elif starts_new_paragraph(current, stripped):
            paragraphs.append(current)
            current = stripped
        elif CONTINUATION_PUNCTUATION.match(stripped):
            current += stripped
        else:
            current += " " + stripped
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def strip_site_chrome(text: str) -> str:
    """Removes the MindTouch UI dump and its byline; pages exported without it are rejoined instead."""
    if not text.startswith(SKIP_LINKS):
        return text
    block = MINDTOUCH_BLOCK.match(text)
    if block is None:
        return rejoin_exploded(text.split("\n", 1)[-1])
    lines = page_lines(text[block.end():])
    byline = 0
    while byline < min(BYLINE_MAX_LINES, len(lines)) and 0 < len(lines[byline]) < 80 \
            and not ends_sentence(lines[byline]):
        byline += 1
    return join_lines(lines[byline:])


def normalized_author(document: dict) -> str:
    return " ".join(((document.get("metadata") or {}).get("author") or "").split())


def names_author(line: str, author: str) -> bool:
    return bool(author) and " ".join(line.removeprefix("- ").split()) == author


def is_byline(lines: list[str], author: str) -> bool:
    """True when the bullets after "Save as PDF" are the author and institution, not a content list."""
    if names_author(lines[0].strip(), author):
        return True
    return len(lines) >= 2 and bool(INSTITUTION.search(lines[1]))


def is_byline_candidate(line: str, author: str, after_byline: bool) -> bool:
    if not line.startswith("- "):
        return False
    if names_author(line, author):
        return len(line) <= BYLINE_LONG_MAX_CHARS
    if ends_sentence(line):
        return False
    if len(line) < BYLINE_MAX_CHARS:
        return True
    return after_byline and len(line) <= BYLINE_LONG_MAX_CHARS and bool(INSTITUTION.search(line))


def byline_end(lines: list[str], start: int, author: str) -> int:
    """Index after the Page ID and byline bullets that begin at start; bullets that are content stay."""
    look = start
    while look < len(lines) and lines[look].strip() == "-":
        look += 1
    if look < len(lines) and PAGE_ID_BULLET.match(lines[look].strip()):
        start = look = look + 1
    candidates: list[int] = []
    while look < len(lines) and len(candidates) < BYLINE_MAX_LINES:
        line = lines[look].strip()
        if line == "-":
            look += 1
            continue
        if not is_byline_candidate(line, author, bool(candidates)):
            break
        candidates.append(look)
        look += 1
    if not candidates:
        return start
    return candidates[-1] + 1 if is_byline([lines[position] for position in candidates], author) else start


def page_number(text: str, document: dict) -> tuple[str | None, bool]:
    """The page's section number ("2.1") and whether the text itself starts with the numbered title.

    The metadata title decides, because a page without its title line starts with its first child entry ("11.1: ...").
    """
    in_text = PAGE_NUMBER.match(text)
    in_title = PAGE_NUMBER.match((document.get("metadata") or {}).get("title") or "")
    if in_title is None:
        return (in_text.group(1), True) if in_text is not None else (None, False)
    return in_title.group(1), in_text is not None and in_text.group(1) == in_title.group(1)


def is_child_summary(segment: str, number: str | None) -> bool:
    """A child TOC entry of page number with a short summary, such as "4.1: Title This page covers ... ."."""
    return (number is not None and segment.startswith(number + ".") and len(segment) <= TOC_SUMMARY_MAX_CHARS
            and len(SENTENCE_END.findall(segment)) <= TOC_SUMMARY_MAX_SENTENCES)


def strip_inline_header(text: str, author: str) -> str:
    """Single-line pages: "Title Last updated Save as PDF Page ID N Author Institution 1.1: Entry 1.2: ..."."""
    header = INLINE_HEADER.match(text)
    if header is None:
        return text
    title = header.group("title")
    rest = text[header.end():].strip()
    if author and rest.startswith(author):
        rest = rest[len(author):].strip()
    rest = INSTITUTION_VIA.sub("", rest, count=1).strip()
    markers = [marker.start() for marker in TOC_MARKER.finditer(rest)]
    if not markers:
        return join_lines([title, rest])
    number = PAGE_NUMBER.match(title)
    bounds = list(zip(markers, markers[1:] + [len(rest)]))
    segments = [rest[start:stop].strip() for start, stop in bounds]
    kept = [segment for segment in segments if len(segment) > TOC_SEGMENT_MAX_CHARS
            and not is_child_summary(segment, number.group(1) if number else None)]
    return join_lines([title, *kept])


def strip_page_header(document: dict) -> str:
    """Removes the "Last updated / Save as PDF / Page ID" header, the byline under it and empty list bullets."""
    text = document["text"]
    author = normalized_author(document)
    if "\n" not in text:
        return strip_inline_header(text, author)
    lines = page_lines(text)
    kept = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        # A Page ID bullet also follows the title directly when the export dropped "Save as PDF".
        is_page_id = (stripped.isdigit() and index > 0 and lines[index - 1].strip() == "Page ID") or (
            index == 1 and PAGE_ID_BULLET.match(stripped) is not None)
        if stripped == "-" or PLACEHOLDER_LINE.match(stripped) or MINDTOUCH_TAGS.match(stripped):
            index += 1
            continue
        if stripped not in HEADER_LINES and not is_page_id:
            match = COLLAPSIBLE_LABEL.match(stripped)
            kept.append(match.group(1) if match else lines[index])
            index += 1
            continue
        index += 1
        if stripped in SAVE_AS_PDF_LINES:
            index = byline_end(lines, index, author)
    return join_lines(kept)


def continues_sentence(line: str) -> bool:
    return bool(line) and (line[0].islower() or CONTINUATION_PUNCTUATION.match(line) is not None)


def is_open_line(line: str) -> bool:
    return bool(line) and not ends_sentence(line) and not line.startswith(("|", "\\"))


def follows_heading(previous: str, line: str) -> bool:
    """True when previous is a title-case heading of two or more words, or one word that line repeats.

    "Rounding\nRounding\nis the process" has a heading; "Press\nSTAT\nand arrow over" has an inline element.
    """
    words = previous.split()
    if (not words or len(words) > HEADING_MAX_WORDS or previous.endswith((*TERMINAL_PUNCTUATION, ",", ";"))
            or INLINE_BEFORE.search(previous) is not None):
        return False
    if not all(word[0].isupper() or word[0].isdigit() or word.lower() in HEADING_CONNECTORS for word in words):
        return False
    if len(words) > 1:
        return True
    return bool({word.lower() for word in WORD.findall(line)} & {word.lower() for word in WORD.findall(previous)})


def is_inline_fragment(previous: str, line: str, following: str) -> bool:
    if (not line or len(line.split()) > FRAGMENT_MAX_WORDS or line.startswith(BLOCK_LINE_STARTS)
            or FOOTNOTE_MARKER.match(line) or not continues_sentence(following)):
        return False
    if len(line) <= FRAGMENT_SYMBOL_MAX_CHARS and line not in SENTENCE_START_WORDS:
        return True
    if INLINE_BEFORE.search(previous) is not None:
        return True
    return INLINE_AFTER.match(following) is not None


def count_split_fragments(lines: list[str]) -> int:
    """Lines that are inline elements split from an unfinished sentence.

    A fragment that only precedes a function word ("and the price\nbade pay\nin gold for him") is not counted, because
    verse and vocabulary lists break their lines there too.
    """
    count = 0
    for index in range(1, len(lines) - 1):
        previous, line, following = lines[index - 1].strip(), lines[index].strip(), lines[index + 1].strip()
        if (is_open_line(previous) and is_inline_fragment(previous, line, following)
                and (len(line) <= FRAGMENT_SYMBOL_MAX_CHARS or INLINE_BEFORE.search(previous) is not None
                     or CONTINUATION_PUNCTUATION.match(following) is not None)):
            count += 1
    return count


def glue(joined: str, line: str, vocabulary: set[str]) -> str:
    """Appends a continuation; a styled first letter ("The\nt\nheory") rejoins its word when the page uses that word."""
    if CONTINUATION_PUNCTUATION.match(line) or joined.endswith(OPENING_MARKS):
        return joined.rstrip() + line
    last_token = joined.rsplit(" ", 1)[-1]
    first_word = WORD.match(line)
    if (len(last_token) == 1 and last_token.islower() and last_token not in "ai" and first_word is not None
            and (last_token + first_word.group(0)).lower() in vocabulary):
        return joined + line
    return joined.rstrip() + " " + line


def join_fragments(text: str) -> str:
    """Rejoins sentences that the export split at inline elements.

    Every page gets lines that start with a sentence's punctuation and values of "Located at :" style labels joined;
    exploded pages also get short inline fragments and their lowercase continuations joined. A fragment under the
    title line, or a capitalised one under a heading ("Role Theory\nA role\nis a position"), starts the paragraph
    instead. Code lines stay apart because a line that ends a sentence (or a colon) is never continued.
    """
    lines = page_lines(text)
    exploded = count_split_fragments(lines) >= EXPLODED_MIN_FRAGMENTS
    vocabulary = {word.lower() for word in WORD.findall(text)} if exploded else set()
    joined: list[str] = []
    chained = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        last = joined[-1].strip() if joined else ""
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        joinable = bool(last) and bool(stripped) and not last.startswith(("|", "\\"))
        marker = joinable and exploded and FOOTNOTE_MARKER.match(stripped) is not None
        inline = (joinable and exploded and is_open_line(last)
                  and not stripped.startswith(BLOCK_LINE_STARTS)
                  and not CODE_LINE.match(last) and not (CODE_LINE.match(stripped) and CODE_LINE.match(following)))
        fragment = inline and is_inline_fragment(last, stripped, following)
        opens_paragraph = fragment and (len(joined) == 1 or (stripped[0].isupper() and follows_heading(last, stripped)))
        joins = marker or (joinable and (
            CONTINUATION_PUNCTUATION.match(stripped) is not None or ATTRIBUTION_LABEL_END.search(last) is not None
            or (fragment and not opens_paragraph)
            or (inline and len(joined) > 1 and chained and continues_sentence(stripped))))
        if joins:
            joined[-1] = glue(joined[-1], stripped, vocabulary)
        else:
            joined.append(line)
        chained = (joins and not marker) or opens_paragraph
    return "\n".join(joined)


def continues_statement(previous: str, line: str) -> bool:
    if not previous or not line:
        return False
    if ATTRIBUTION_LABEL_END.search(previous):
        return True
    if ends_sentence(previous):
        return False
    if ATTRIBUTION_FIELD.match(line):
        return len(previous) <= ATTRIBUTION_ITEM_MAX_CHARS
    return line[0].islower() or CONTINUED_LINE_END.search(previous) is not None


def licence_view(text: str) -> str:
    """The text as the licence filter reads it: a statement split over lines on one line, licence codes normalised.

    Every rewrite keeps the length, so the filter's strip spans apply to the original text.
    """
    lines = page_lines(text)
    pieces = lines[:1]
    for previous, line in zip(lines, lines[1:]):
        pieces.append((" " if continues_statement(previous.strip(), line.strip()) else "\n") + line)
    text = LICENCE_CODE_DASH.sub("-", "".join(pieces))
    text = SA_BEFORE_NC.sub("BY-NC-SA", text)
    text = PERMISSION_VERB.sub(lambda match: PERMISSION_VERB_STAND_IN[match.group(0).lower()], text)
    text = PERMISSION_PRONOUN.sub(lambda match: "the".ljust(len(match.group(0))), text)
    return EXHIBIT_LABEL.sub("Figure ", text)


def is_nc_media_note(line: str, licence) -> bool:
    return NC_SOURCE_NOTE.match(line.strip()) is not None and licence.RE_MEDIA.search(line) is not None


def strip_nc_media_credits(text: str, licence) -> str:
    """Removes NC/ND media credits that the licence filter keeps: footnotes and lines just under a caption."""
    lines = page_lines(text)
    kept = []
    for index, line in enumerate(lines):
        above = [previous for previous in lines[max(0, index - CAPTION_CREDIT_LINES):index] if previous.strip()]
        under_caption = (len(line) <= CAPTION_CREDIT_MAX_CHARS and bool(licence.nc_nd_tokens(licence_view(line)))
                         and any(CAPTION_START.match(previous.strip()) for previous in above))
        if not (under_caption or is_nc_media_note(line, licence)):
            kept.append(line)
    return join_lines(kept)


class LicenceRecheck(Step):
    """Applies the commercial-use licence rules again on text whose split lines are joined.

    The licence pass ran on the raw export, where declarations split over lines ("by\nA. David Lewis\nis licensed
    under a\nCreative Commons Attribution-NoDerivatives ...") were not seen as one statement.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.licence = load_licence_filter()

    def apply(self, document: dict) -> dict | None:
        text = document["text"]
        lines = page_lines(text)
        if any(NC_SOURCE_NOTE.match(line.strip()) and not is_nc_media_note(line, self.licence) for line in lines):
            return None
        action, _, spans = self.licence.classify({**document, "text": licence_view(text)})
        if action == "drop":
            return None
        text = strip_nc_media_credits(self.licence.apply_strip(text, spans), self.licence)
        return document if text == document["text"] else {**document, "text": text}


def is_attribution_line(line: str) -> bool:
    limit = DERIVATION_NOTE_MAX_CHARS if DERIVATION_NOTE.match(line) else LICENCE_LINE_MAX_CHARS
    if len(line) <= limit and LICENCE_LINE.search(line):
        return True
    return len(line) <= AUTHOR_CREDIT_MAX_CHARS and bool(AUTHOR_CREDIT.search(line))


def heading_key(line: str) -> str:
    return line.removeprefix("- ").strip().removesuffix(":").strip().lower()


def heads_prose(lines: list[str], index: int) -> bool:
    """True for a heading line (not a bullet) directly above body prose, possibly under one more heading."""
    if lines[index].strip().startswith("- "):
        return False
    for line in lines[index + 1:index + 1 + BODY_HEADING_RUN]:
        stripped = line.strip()
        if len(stripped) >= BODY_PROSE_MIN_CHARS:
            return True
        if stripped.startswith("- ") or len(stripped) >= BYLINE_MAX_CHARS or ends_sentence(stripped):
            return False
    return False


def strip_attribution(text: str) -> str:
    """Removes attribution blocks, licence and image-credit lines, page thumbnail captions and MindTouch tag lists.

    A block starts at an attribution heading and runs while its lines are licences, names (short lines without
    final punctuation) or further headings; a prose line, a body heading or a heading above prose ends it, because
    some pages put the attribution heading at the top. A line that continues a removed statement goes with it.
    """
    text = INLINE_LICENCE.sub(r"\1\n\2", PARENTHETICAL_CREDIT.sub("", text))
    lines = page_lines(text)
    kept = []
    in_block = False
    previous_removed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        heading = heading_key(stripped)
        continues_removed = previous_removed and (not ALPHANUMERIC.search(stripped) or (
            len(stripped) <= LICENCE_LINE_MAX_CHARS and continues_statement(lines[index - 1].strip(), stripped)))
        removed = (heading in ATTRIBUTION_HEADINGS or is_attribution_line(stripped) or continues_removed
                   or ADAPTIONS_NOTE.match(stripped) is not None
                   or THUMBNAIL_LINE.match(stripped) is not None or MINDTOUCH_TAGS.match(stripped) is not None)
        previous_removed = removed
        if heading in ATTRIBUTION_HEADINGS:
            in_block = True
        if removed:
            continue
        if in_block and ADAPTATION_NOTE.match(stripped):
            previous_removed = True
            continue
        if in_block:
            is_credit_like = stripped in HEADER_LINES or (
                len(stripped) < BYLINE_MAX_CHARS and not ends_sentence(stripped))
            if is_credit_like and heading not in BODY_HEADINGS and not heads_prose(lines, index):
                previous_removed = True
                continue
            in_block = False
        kept.append(line)
    return join_lines(kept)


def strip_orphan_attributes(line: str) -> str:
    def replace(match: regex.Match) -> str:
        inside_tag = line.rfind("<", 0, match.start()) > line.rfind(">", 0, match.start())
        return match.group(0) if inside_tag else ""
    return ORPHAN_ATTRIBUTES.sub(replace, line)


def strip_html_residue(text: str) -> str:
    """Removes lines that hold leaked HTML markup, cut attribute runs and doubled tag brackets, and unwraps link tags."""
    kept = [DOUBLED_TAG.sub("", strip_orphan_attributes(LINK_TAG.sub("", line))) for line in page_lines(text)
            if HTML_RESIDUE_LINE.search(line) is None and TAG_ONLY_LINE.match(line) is None]
    return join_lines([line for line in kept if line.strip() and ENTITY_RESIDUE_LINE.match(line.strip()) is None])


def landing_toc_spans(lines: list[str], number: str, first_entry: int) -> list[tuple[int, int]]:
    """Line spans of TOC entries and their one-paragraph summaries on a chapter landing page.

    An entry numbered as a child of the page ("2.1:" on page "2:") with at most one content line before the next
    entry is a TOC entry; numbered subsections with real content are not. A single entry counts only as a bullet.
    """
    child = regex.compile(rf"\A(?:- )?{regex.escape(number)}\.(?:\d{{1,3}}[A-Z]?|[A-Z]): .{{1,150}}\Z")
    entries = [index for index, line in enumerate(lines) if index >= first_entry and child.match(line.strip())]
    spans = []
    for position, start in enumerate(entries):
        stop = entries[position + 1] if position + 1 < len(entries) else len(lines)
        content = [line for line in lines[start + 1:stop] if line.strip() not in ("", "-")]
        if len(content) <= 1:
            spans.append((start, stop))
    if len(entries) == 1:
        return spans if lines[entries[0]].strip().startswith("- ") else []
    if len(spans) < 2 or 2 * len(spans) < len(entries):
        return []
    return spans


def strip_toc(document: dict) -> str:
    """Removes chapter-landing TOC entries with their summaries and other runs of TOC lines."""
    text = document["text"]
    lines = page_lines(text)
    number, titled = page_number(text, document)
    dropped = set()
    if number is not None:
        for start, stop in landing_toc_spans(lines, number, 1 if titled else 0):
            dropped.update(range(start, stop))
    run: list[int] = []
    for index in range(1, len(lines) + 1):
        if index < len(lines) and index not in dropped and TOC_ENTRY.match(lines[index].strip()):
            run.append(index)
            continue
        if len(run) >= TOC_RUN_MIN_LINES:
            dropped.update(run)
        run = []
    return join_lines([line for index, line in enumerate(lines) if index not in dropped])


def spaced_delimiter_repairs(text: str) -> list[int]:
    """Positions of the space in spaced math delimiters ("\\ (" ... "\\ )") that are delimiters, not control spaces."""
    delimiters = list(MATH_DELIMITER.finditer(text))
    repairs: list[int] = []
    open_kind = None
    for position, delimiter in enumerate(delimiters):
        spaced, bracket = delimiter.group(1) == " ", delimiter.group(2)
        if open_kind is None:
            if bracket not in OPENERS:
                continue
            if not spaced:
                open_kind = bracket
                continue
            before = text[delimiter.start() - 1] if delimiter.start() > 0 else "\n"
            closer = delimiters[position + 1] if position + 1 < len(delimiters) else None
            if before not in NOT_BEFORE_SPACED_OPENER and closer is not None and closer.group(2) == OPENERS[bracket]:
                repairs.append(delimiter.start(1))
                open_kind = bracket
            continue
        if bracket != OPENERS[open_kind]:
            continue
        if spaced:
            closer = next((later for later in delimiters[position + 1:] if later.group(2) in OPENERS or later.group(2)
                           == bracket), None)
            if closer is not None and closer.group(2) == bracket and closer.group(1) == "":
                continue
            repairs.append(delimiter.start(1))
        open_kind = None
    return repairs


def repair_spaced_delimiters(text: str) -> str:
    repairs = spaced_delimiter_repairs(text)
    if not repairs:
        return text
    pieces = []
    cursor = 0
    for position in repairs:
        pieces.append(text[cursor:position])
        cursor = position + 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def clean_latex(text: str) -> str:
    """Resolves the \\PageIndex macro, removes MathJax layout commands and repairs spaced delimiters and commands."""
    text = PAGE_INDEX.sub(r"\1", text)
    text = repair_spaced_delimiters(SPACED_COMMAND.sub(r"\\", text))
    text = BARE_PAGE_INDEX.sub(r"\1", text)
    text = RESIDUAL_PAGE_INDEX.sub("", text)
    text = LATEX_LAYOUT.sub("", text)
    text = LATEX_LINE_SKIP.sub(r"\1", text)
    return LATEX_CROSSREF.sub("", text)


def ordinal_suffix(number: str) -> str:
    if number[-2:] in ("11", "12", "13"):
        return "th"
    return {"1": "st", "2": "nd", "3": "rd"}.get(number[-1], "th")


def join_ordinal(match: regex.Match) -> str:
    number, suffix = match.group(1), match.group(2)
    if ordinal_suffix(number) != suffix:
        return match.group(0)
    return number + suffix


def formula_piece(piece: str) -> regex.Match | None:
    """Match of an element group such as "(H", "O)" or "O," whose capitalised symbols are all elements."""
    match = FORMULA_PIECE.match(piece)
    symbols = ELEMENT_SYMBOL.findall(piece)
    if match is None or not symbols or not all(symbol in ELEMENTS for symbol in symbols) or LICENCE_CODE.search(piece):
        return None
    return match


def formula_run_end(pieces: list[str], start: int) -> int:
    """End (exclusive) of the flattened formula that starts at pieces[start], or start when there is none.

    A formula alternates element groups with subscript counts ("C 2 H 5"), may close brackets ("(H 2 O) 6 ]") and
    may end with a charge ("2+"). A group that already holds its count ("CH3") starts a formula only before a bond
    ("CH3 -CH 2"), because labels such as "H0 H1" or "B12" look the same.
    """
    first = formula_piece(pieces[start])
    if first is None or first.group("punctuation"):
        return start
    first_counted = DIGIT.search(pieces[start]) is not None
    if first_counted and (start + 1 >= len(pieces) or BOND.search(pieces[start + 1]) is None):
        return start
    previous = "count" if first_counted else "group"
    counts, groups, has_charge, end = int(first_counted), 1, False, start + 1
    while end < len(pieces):
        piece = pieces[end]
        count = COUNT_PIECE.match(piece)
        charge = CHARGE_PIECE.match(piece)
        group = formula_piece(piece)
        if previous in ("group", "closing") and count:
            counts, previous, end = counts + 1, "count", end + 1
            if count.group(1):
                break
        elif charge and (previous != "group" or piece[0].isdigit()):
            if piece[0].isdigit():
                counts += 1
            has_charge, end = True, end + 1
            break
        elif previous == "count" and group:
            counted = DIGIT.search(piece) is not None
            counts, groups, end = counts + int(counted), groups + 1, end + 1
            previous = "count" if counted else "group"
            if group.group("punctuation"):
                break
        elif previous == "count" and CLOSING_PIECE.match(piece):
            previous, end = "closing", end + 1
        else:
            break
    if counts == 0 or (groups == 1 and not has_charge and pieces[start] in WORD_SYMBOLS):
        return start
    return end


def join_formula_pieces(line: str) -> str:
    pieces = line.split(" ")
    joined: list[str] = []
    index = 0
    while index < len(pieces):
        end = formula_run_end(pieces, index)
        if end == index:
            joined.append(pieces[index])
            index += 1
            continue
        formula = "".join(pieces[index:end])
        if end < len(pieces) and CONTINUATION_PUNCTUATION.match(pieces[end]):
            formula += pieces[end]
            end += 1
        joined.append(formula)
        index = end
    return " ".join(joined)


def join_hybrid(match: regex.Match) -> str:
    return "sp" + match.group(1) + (match.group(2) or "") + (match.group(3) or "")


def join_subscripts(text: str) -> str:
    """Rejoins chemical formulas and orbital labels whose <sub>/<sup> digits the export split off with a space."""
    text = HYBRID_ORBITAL.sub(join_hybrid, text)
    text = ORBITAL_OCCUPANCY.sub(r"\1\2", text)
    return "\n".join(join_formula_pieces(line) if FLATTENED_SUBSCRIPT.search(line) else line
                     for line in page_lines(text))


def repair_inline(text: str) -> str:
    """Repairs damage from removed inline HTML: editor prompts, footnote return arrows, empty table rows, truncated
    link text, spaces before punctuation, split ordinal superscripts, lines that start with a sentence's punctuation
    and a line printed twice in a row.
    """
    text = text.replace(TEMPLATE_PROMPT, "")
    text = FOOTNOTE_RETURN.sub("", text)
    text = TRUNCATED_URL.sub("", text)
    text = LUMEN_SHORTCODE.sub("", text)
    text = SPACE_BEFORE_PUNCTUATION.sub(lambda match: match.group(1) or match.group(2), text)
    text = SPLIT_ORDINAL.sub(join_ordinal, text)
    kept: list[str] = []
    for line in page_lines(text):
        stripped = line.strip()
        if not stripped or stripped == "-" or EMPTY_TABLE_LINE.match(stripped):
            continue
        if kept and len(stripped) >= REPEATED_LINE_MIN_CHARS and stripped == kept[-1].strip():
            continue
        if kept and CONTINUATION_PUNCTUATION.match(stripped) and not kept[-1].lstrip().startswith(("|", "\\")):
            kept[-1] = kept[-1].rstrip() + stripped
            continue
        kept.append(line)
    return join_lines(kept)


def word_count(text: str) -> int:
    return len(text.split())


def short_line_fraction(text: str) -> float:
    lines = [line for line in page_lines(text) if line.strip()]
    return sum(1 for line in lines if len(line) < SHORT_LINE_CHARS) / len(lines)


def is_term_list(document: dict) -> bool:
    """A key-term or glossary page that lists bare terms: mostly short lines and no line that ends a sentence."""
    metadata = document.get("metadata") or {}
    if not TERM_LIST_TITLE.search(f"{metadata.get('title') or ''} {document.get('id', '')}"):
        return False
    lines = [line.strip() for line in page_lines(document["text"]) if line.strip()]
    if any(line.rstrip(" \"'”’)]").endswith((".", "?", "!")) for line in lines):
        return False
    return short_line_fraction(document["text"]) >= TERM_LIST_MIN_SHORT_FRACTION


def prose_for_language_id(text: str) -> str:
    """The text without LaTeX math and without lines that hold no word, such as "- a. b. c." of exercise lists."""
    return "\n".join(line for line in page_lines(LATEX_MATH.sub("", text)) if PROSE_WORD.search(line))


def is_english(language: LanguageIdentifier, document: dict) -> bool:
    prose = prose_for_language_id(document["text"])
    if sum(1 for _ in LETTER.finditer(prose)) < MIN_PROSE_LETTERS:
        return True
    minimum = ESPANOL_MIN_SHARE if ESPANOL_HOST in str(document.get("id", "")) else ENGLISH_MIN_SHARE
    return language.is_english(prose, minimum)


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


def build_pipeline() -> Pipeline:
    language = LanguageIdentifier()
    return Pipeline("libretexts", [
        TextMap("normalize", normalize_text),  # ftfy and NFC with FineWeb/datatrove settings (LITERATURE §c)
        TextMap("strip_site_chrome", strip_site_chrome),  # "site chrome": 1.69% of pages are a MindTouch UI dump
        DocumentMap("strip_page_header", strip_page_header),  # "header chrome lines, bylines": 30% of pages
        TextMap("join_fragments", join_fragments),  # "text broken at removed inline elements" on exploded pages
        LicenceRecheck("licence_recheck"),  # user decision: commercial use only; declarations split over lines
        TextMap("strip_attribution", strip_attribution),  # "attribution footers": 12.9%; licence lives in metadata (§f)
        TextMap("strip_html_residue", strip_html_residue),  # measured: raw HTML in 184 pages (0.3%), C4 §2.2
        DocumentMap("strip_toc", strip_toc),  # "chapter landing pages": 5.6% hold only a header and a TOC
        TextMap("clean_latex", clean_latex),  # "\PageIndex macro and LaTeX residue": 21.7%; math itself is kept
        TextMap("join_subscripts", join_subscripts),  # "flattened subscripts": 6.0% of pages ("H 2 O", "1s 2")
        TextMap("repair_inline", repair_inline),  # "text broken at removed inline elements": 53.7%
        LicenceRecheck("licence_final"),  # later steps change line structure, so the licence decision runs again
        Filter("min_words", lambda document: word_count(document["text"]) >= MIN_WORDS),  # "almost no content": 7.9%
        Filter("short_lines", lambda document: short_line_fraction(document["text"]) < MAX_SHORT_LINE_FRACTION),
        Filter("term_lists", lambda document: not is_term_list(document)),  # sample audit: bare key-term pages
        Filter("english", lambda document: is_english(language, document)),  # "non-English": 2.3% of pages
        ExactDedup("exact_dedup"),  # Gopher A.3.2: exact then fuzzy dedup each lowered loss
        NearDedup("near_dedup"),  # "near-duplicates": 33.9% of pages are in Courses/Bookshelves clusters
        TextMap("mask_pii", lambda text: mask_pii(text, kinds=PII_KINDS)),  # Common Pile §4.1 regex PII
    ])


if __name__ == "__main__":
    run_cli(build_pipeline)
