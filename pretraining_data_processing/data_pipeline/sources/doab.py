"""Cleaning pipeline for Common Pile doab: open-access books converted from PDF to Markdown by Marker.

Measured findings: analysis/findings/doab.json. Literature: analysis/LITERATURE.md.
The cross-book boilerplate table must exist before a run:
python -m data_pipeline.sources.doab boilerplate-table --input DIR --out FILE --procs P
"""
import argparse
import html
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import regex
import xxhash

from data_pipeline.common.cli import run_cli
from data_pipeline.common.dedup import (
    Deduplicator, DuplicateOfWriter, ExactDedup, NearDedup, save_array_atomically, shard_offsets, shard_prefix,
)
from data_pipeline.common.io import list_shards, read_documents, write_bytes_atomically
from data_pipeline.common.licence import LicenceCreditStrip
from data_pipeline.common.langid import LanguageIdentifier, fasttext_input, language_segments
from data_pipeline.common.pii import mask_pii
from data_pipeline.common.pipeline import Filter, Pipeline, Step, TextMap
from data_pipeline.common.text import normalize_text

BOILERPLATE_TABLE_PATH = Path(__file__).resolve().parents[2] / "reports/doab/boilerplate_paragraphs.npy"
WORDLIST_PATH = Path("/usr/share/dict/american-english")

# CCNet paragraph-hash normalization (Wenzek et al., arXiv:1911.00359 §3.3): lowercase, digits to 0, no punctuation
# or accents. Keys shorter than 40 chars are headings such as "1. Introduction", which repeat in every book.
MIN_PARAGRAPH_KEY_CHARS = 40
# Read paragraphs found in 2-4 books include reprinted chapter text; every one read in 5 or more books was licence,
# publisher, series or proof-stamp text (13,728 hashes on the licence-filtered shards).
MIN_BOILERPLATE_BOOKS = 5
# Section titles such as "4.3.2 Fourier Transform Infrared Spectroscopy (FTIR)" recur across books; italic licence
# sentences that Marker formats like headings are longer.
MAX_SECTION_HEADING_CHARS = 100
NON_WORD = regex.compile(r"[^\p{L}\p{N}\s]+")
NUMBER = regex.compile(r"\p{N}")
WHITESPACE = regex.compile(r"\s+")
PARAGRAPH_BREAK = regex.compile(r"\n[^\S\n]*\n")

MARKUP = regex.compile(r"[*_#>\\]")
SUP_OR_SUB_TAG = regex.compile(r"</?su[pb]>")
HEADING = regex.compile(r"#{1,6}\s.*|\*\*[^*]{1,200}\*\*|\*[^*]{1,200}\*")
MAX_HEADING_CHARS = 200
LIST_ITEM = regex.compile(r"\s*(?:[-*•‣◦▪●]|\d{1,3}[.)])\s")
TABLE_OR_MATH = regex.compile(r"\s*(?:\||\$\$)")

# Findings "publisher_boilerplate_and_front_matter": the analyst's 23 patterns (25/25 non-prose on read lines),
# matched in the first 400 chars of a markup-free line, plus the cover-credit, catalogue-advert and
# compilation-notice patterns of the extra pass.
BOILERPLATE_SCAN_CHARS = 400
# Licence paragraphs are 500-650 chars; every longer matching line read had prose glued on by Marker, so only its
# matching sentences go.
MAX_BOILERPLATE_LINE_CHARS = 700
SENTENCE_BREAK = regex.compile(r"(?<=[.!?])\s+(?=[\p{Lu}\"“(\[])")
BOILERPLATE = regex.compile("|".join([
    r"Additional information is available at the end of the chapter",
    r"Licensee (?:IntechOpen|InTech|MDPI)",
    r"distributed under the terms (?:and conditions )?of the Creative Commons",
    r"This (?:chapter|article|book|work|publication|title) is (?:an open access|distributed|licensed|published under"
    r"|available)",
    r"licen[sc]ed under (?:the terms of )?(?:a |the )?Creative Commons",
    r"Address all correspondence to",
    r"Downloaded from https?://",
    r"We are IntechOpen|Interested in publishing with us|Numbers displayed above are based on latest data",
    r"Open Access books Built by scientists|Selection of our books indexed in the Book Citation Index",
    r"Open access books available|International authors and editors|Contributors from top 500 universities"
    r"|Our authors are among the|^Provisional chapter$",
    r"^\W*(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,}/\S+\W*$",
    r"Frontiers (?:eBook )?Copyright Statement|^About Frontiers|Frontiers Journal Series|Dedication to Quality"
    r"|What are Frontiers Research Topics",
    r"All rights reserved",
    r"Deutsche Nationalbibliothek|Deutschen Nationalbibliothek|CIP catalogue record|Library of Congress Cataloging",
    r"printed on acid-free paper|Gedruckt auf",
    r"Publisher.s Note: MDPI stays neutral|Disclaimer/Publisher.s Note",
    r"The authors? declares? (?:no|that there is no|that the research was conducted in the absence)",
    r"^\W*Citation: |^\W*Correspondence: |Received: .{0,40}(?:Accepted|Revised): ",
    r"(?:©|\(c\)|Copyright ©?) ?(?:19|20)\d\d",
    r"Editorial Office E-mail|MDPI St\. Alban-Anlage|www\.mdpi\.com",
    r"^Reprinted from: |This article was submitted to .{0,80}a section of the journal"
    r"|Printed Edition of the Special Issue",
    r"\bISBN(?:-1[03])?:? ?(?:97[89])?[\d\- ]{9,}|\be?-?ISSN\b",
    r"^\W*(?:Cover (?:design|image|photo|illustration|art|and concept of design)|Typeset(?:ting)?(?: by)?|Proofreading"
    r"|Jacket illustration)\b",
    r"\bDOI:? ?(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,}/\S+\s*$",
    r"(?:Photo|Image|Cover image|Cover photo)s? (?:by|courtesy of|credit:?).{0,80}/ ?(?:iStock|Shutterstock|Getty)"
    r"|© .{1,40} ?/ ?iStock",
    r"\d+,\d\d € \((?:DE|A)\)|\bpb\., (?:ill\., )?\d+,\d\d €",
    r"book as a whole.{0,160}(?:CC BY-NC-ND|reserved|without INTECH)|book taken as a whole is ©"
    r"|compilation of articles constituting this e-?book",
]), regex.IGNORECASE)
ADMIN_STATEMENT = regex.compile(
    r"(?:Author Contributions?|Funding|Institutional Review Board Statement|Informed Consent Statement"
    r"|Data Availability Statement|Conflicts? of Interest(?: Statement)?|Acknowledge?ments?|Supplementary Materials?"
    r"|Specialty section|Edited by|Reviewed by|Academic Editors?|Keywords|Ethics Statement|Abbreviations)\s*:",
    regex.IGNORECASE,
)
# Findings "running_headers": IntechOpen template footer, Frontiers and MDPI proof stamps, and the IntechOpen page head
# "<chapter title> DOI: http://dx.doi.org/10.5772/..." (or "<title> http://dx.doi.org/10.5772/... <page>"). Marker
# glues them into prose lines, so only the stamp span goes. An unitalicized title is the run of capitalized words and
# title function words before the DOI.
TITLE_WORD = (r"(?:\(?\p{Lu}[\p{L}\p{N}'’\-]*\)?[,:;?!…]*\.{0,3}|\p{N}+[,:]?|[–—\-:…&]"
              r"|(?:a|an|the|of|in|on|for|and|or|to|with|from|at|by|as|into|via|vs\.|its|their)[,:]?)")
INTECH_DOI_URL = r"(?:DOI:?\s*)?https?://dx\.doi\.org/10\.5772/"
RUNNING_STAMP = regex.compile("|".join([
    r"(?:\b\d{1,4}\s+)?Will-be-set-by-IN-TECH\b(?:\s+\d{1,4}\b)?",
    r"\bf[a-z]{2,8}-\d{2}-\d{5,6} \w+ \d{1,2}, \d{4} Time: \d{1,2}:\d{1,2}(?: # ?\d{1,4})?",
    r"(?:\*[^*\n]{1,40}\*\s*|\b\p{Lu}\p{L}+\*?\s+)?(?:\*\*)?(?:19|20)\d\d(?:\*\*)?,\s*(?:\*?\d+\*?)?,\s*"
    r"(?:[x×](?:;\s*doi:)?\s*)?FOR PEER REVIEW(?:\s+\d{1,4}(?:\s+of\s+\d{1,4})?)?",
    r"(?<!\w)(?:[x×](?:;\s*doi:)?\s*)?FOR PEER REVIEW(?:\s+\d{1,4}(?:\s+of\s+\d{1,4})?)?",
    rf"\*(?:[^*\n]{{1,250}}?\s)?{INTECH_DOI_URL}[^*\s]+\*",
    rf"(?:(?<!\S)\p{{Lu}}\S*(?:\s+{TITLE_WORD})*\s+)?{INTECH_DOI_URL}\S+(?:\s+\d{{1,4}}(?=\s|$))?",
]))
STAMP_HINT = regex.compile(r"10\.5772/|PEER REVIEW|Will-be-set|Time: \d")
MULTIPLE_SPACES = regex.compile(r"(?<=\S) {2,}")
# Findings "publisher_boilerplate_and_front_matter": the statistics lines of the "We are IntechOpen" advert.
INTECH_AD_START = regex.compile(r"We are IntechOpen", regex.IGNORECASE)
INTECH_AD_TOKEN = (r"(?:[\d,.]+(?:M\+?|\+|%)?|Top 1%|Countries delivered to|Downloads|most cited scientists"
                   r"|Our authors are among the|Open access books available|International authors and editors"
                   r"|Contributors from top 500 universities)")
INTECH_AD_LINE = regex.compile(rf"{INTECH_AD_TOKEN}(?:\s+{INTECH_AD_TOKEN})*", regex.IGNORECASE)
PAGE_NUMBER = regex.compile(r"(?:\d{1,4}|[ivxlc]{1,6})", regex.IGNORECASE)
EDGE_NUMBER = regex.compile(r"^(?:\d{1,4}|[IVXLC]{1,6})\s+\S|\S\s+(?:\d{1,4}|[IVXLC]{1,6})$")
EDGE_NUMBERS = regex.compile(r"^(?:\d{1,4}|[IVXLC]{1,6})\s+|\s+(?:\d{1,4}|[IVXLC]{1,6})$")
SECTION_NUMBERED = regex.compile(r"(?:\d+(?:\.\d+)+\.?\s|[•▪\-–]\s)")
MAX_RUNNING_HEADER_CHARS = 150
MIN_RUNNING_HEADER_KEY_CHARS = 8
# "Fig. 14 shows ..." is prose; a caption has a separator or a capitalized word after the number.
CAPTION = regex.compile(
    r"(?P<label>(?i:Figure|Fig\.|Table|Scheme|Plate|Chart|Graph|Box|Map|Illustration)\s*[A-Z]?\d+(?:\.\d+)*)"
    r"(?:(?P<separator>\s*[.:|])|\s+[\p{Lu}(]|$)"
)
# "Figure 14 (1) is the contrast diagram ...", "Table 12 (El-Sherif et al., 2003; El-Sherif, 2006). The stability
# constants ...": the label is the subject of a sentence, or a citation follows it.
CAPTION_SUBJECT = regex.compile(r"\s*(?:\(\p{L}{0,2}\d{0,2}\)\s*)?(?:is|are|was|were|has|have)\b"
                                r"|\s*\([^)]*\b(?:19|20)\d\d[a-z]?\b")
MAX_CAPTION_CHARS = 600
# "Figure 4 (B) illustrates ...", "Figure 7 A and B, repeated-measures ANOVA revealed ...": without a separator, a
# verb near the start marks a sentence. Springer captions ("Table 2.1 Summary of adiabatic invariants") have no
# separator and prose function-word shares, so the share does not separate them.
CAPTION_VERB = regex.compile(
    r"\b(?:illustrates|shows|showed|depicts|presents|revealed|reveals|indicates|demonstrates|displays"
    r"|summari[sz]es|compares|represents)\b", regex.IGNORECASE)
CAPTION_VERB_SCAN_CHARS = 60
AFFILIATION = regex.compile(
    r"(?:\d{1,2}\s*|\*\s*)?(?:Department|Dept\.|Faculty|School|Institute|University|Laboratory|Centre|Center"
    r"|Division|College|Hospital|Graduate School)\b[^.]{0,250}"
)
MAX_AFFILIATION_CHARS = 300
# Read false positives without these marks were headings ("Laboratory Processing") and wrapped prose lines.
AFFILIATION_MARK = regex.compile(r",|^\s*(?:\d|\*|<sup>)")
# Findings "section_split_fragments": editor credits and author bylines head every IntechOpen/MDPI title page.
EDITOR_CREDIT = regex.compile(
    r"\W*(?:[Ee]dited by|(?:Academic |Guest |Series |Volume |Topic )?Editors?(?=\s*(?::|\p{Lu}|$)))")
MAX_CREDIT_LINE_CHARS = 600
NAME_SEPARATOR = regex.compile(r"\s*(?:[,;&]|\band\b)\s*")
NAME_TOKEN = regex.compile(
    r"\p{Lu}[\p{L}'’.\-]*|\p{Lu}\.|(?:de|da|do|dos|das|del|della|di|van|von|der|den|la|le|e|y|al|el|bin)")
# The comma after a marker separates names, so the marker match must not take it. Markdown removal leaves
# "Hooft 1,,†, Guido" from "Hooft 1,*,†, Guido", so a marker can also follow a comma.
AFFILIATION_MARKER = regex.compile(
    r"(?<=\p{L})[\d*†‡§⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:,[\d*†‡§⁰¹²³⁴⁵⁶⁷⁸⁹]+)*"
    r"|(?<![^\s,])[\d*†‡§#⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:,[\d*†‡§#⁰¹²³⁴⁵⁶⁷⁸⁹]+)*(?=[\s,;]|$)")
# Marker splits Polish letters into a spacing accent and a letter: "Wo´s", "Szyma ´nska", "Ró˙za".
SPACING_DIACRITIC = regex.compile(r"\s?[´˙¨ˇ˛˝]")
# Bylines read in the sample listed 3 to 40 names of 2-5 tokens; two names are often a wrapped line fragment.
MIN_BYLINE_NAMES = 3
MIN_NAME_TOKENS = 2
MAX_NAME_TOKENS = 5
# MDPI bylines that Marker formats as headings, and bylines with one short name ("Yan-Zhu1", "Hochella, Jr. 1,7"),
# carry affiliation markers; title lines such as "Quality, Safety, and Authenticity" do not.
MIN_BYLINE_MARKERS = 2
MAX_ONE_TOKEN_NAMES = 1
# IntechOpen author blocks glue numbered italic affiliations to the names: "S. Abdullah1, T. E. Putra2 and M. Z.
# Nuawi1 *1Universiti Kebangsaan Malaysia 2Universitas Syiah Kuala*", "... Yoshimura2 ¹*UNESP - Univ Estadual*".
AFFILIATION_TAIL = regex.compile(r"\s(?:[¹²³⁴⁵⁶⁷⁸⁹]\*|\*[\d¹²³⁴⁵⁶⁷⁸⁹])\p{Lu}")
MIN_AFFILIATED_BYLINE_NAMES = 1
# Software watermarks repeated across figures ("Versión Estudiantil Versión Estudiantil ..."); Gopher (Rae et al.,
# arXiv:2112.11446, Table A1) drops documents whose top 2-gram covers more than 20% of chars.
MIN_REPEATED_PHRASE_WORDS = 20
MAX_TOP_BIGRAM_SHARE = 0.3
# Findings "pii_contacts": author e-mails sit in correspondence and affiliation lines, which are not prose.
EMAIL = regex.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

REFERENCE_HEADING = regex.compile(
    r"(?:\d+\.?\s*)?(?:References?|Bibliography|Literature Cited|Works Cited|Reference List|Sources|Literatur"
    r"|Literaturverzeichnis|Bibliografia|Bibliographie|Références|Referencias|Referências)\s*:?",
    regex.IGNORECASE,
)
INDEX_HEADING = regex.compile(r"(?:(?:Subject|Name|Author|General) )?Index(?: of [A-Za-z ]+)?", regex.IGNORECASE)
CONTENTS_HEADING = regex.compile(r"(?:Table of )?Contents|Brief Contents|Detailed Contents", regex.IGNORECASE)
ADMIN_HEADING = regex.compile(
    r"Author details|Author information|Authors? contributions?|(?:List of )?Abbreviations|Notes on contributors"
    r"|(?:List of )?Contributors|Editorial board|Series editors?|Editors?|Topic editors?",
    regex.IGNORECASE,
)
# A heading whose section the cleaning emptied is removed with it (findings: 38,583 docs with empty References).
EMPTIED_HEADING = regex.compile(
    r"(?:\d+\.?\s*)?(?:References?|Bibliography|Author details|Conflicts? of interest|Acknowledge?ments?|Funding"
    r"|(?:List of )?Abbreviations|(?:Table of )?Contents|Notes|Authors? contributions?|Data availability(?: statement)?"
    r"|Supplementary materials?|Index|Meet the editors?|About the editors?|Notice|Contributors)\s*:?",
    regex.IGNORECASE,
)
# Index tails sit in the final quarter of a book (findings: 1,717 docs, 20.8M chars after the last Index heading).
INDEX_TAIL_START_SHARE = 0.75

REFERENCE_PREFIX = [
    regex.compile(r"\[\d{1,4}\]\s+\S"),
    regex.compile(r"\d{1,4}\.\s+[A-Z][A-Za-z'\-]+,?\s+(?:[A-Z]{1,3}[.,]|[A-Z]\.\s?[A-Z]\.)"),
    # "Molgedey L. & Schuster, H. G. (1994).", "Taylor, S. L. & Moffat, J. F. (2005)."
    regex.compile(r"\p{Lu}[\p{L}'’\-]+,?\s(?:\p{Lu}\.\s?){1,3},?\s?&\s\p{Lu}"),
    # "Pinheiro H, Gaspar R, ... (2018)" and "Lerma-Canto, A.; Domingez-Candela, I.; ...": two authors in a row.
    regex.compile(r"\p{Lu}[\p{L}'’\-]+ \p{Lu}{1,3},? (?:and )?\p{Lu}[\p{L}'’\-]+ \p{Lu}{1,3}\b"),
    regex.compile(r"\p{Lu}[\p{L}'’\-]+,\s(?:\p{Lu}\.\s?){1,3}[;,]\s\p{Lu}[\p{L}'’\-]+(?: \p{Lu}[\p{L}'’\-]+)?,\s\p{Lu}\."),
]
# "Recently, Giesen et al. (2020) have reported" and "Likewise, Trouillot (1995)" are prose; entries with a full
# given name ("Berger, Johannes (1996) 'Was behauptet ...'") have few function words.
AUTHOR_YEAR_START = regex.compile(
    r"[A-Z][\p{L}'\-]+,\s(?:(?P<initial>[A-Z]\.|[A-Z]{1,3}\b)|[A-Z][\p{L}]+).{0,150}?\(\d{4}[a-z]?\)")
# Chicago author-date entries, "Wieviorka, Michel. 2001. La violencia: ...": foreign-language reference lists of this
# form counted as foreign chapters in the language decision. Only names stand between the comma and the year, so
# "In Italy, the town ... March 7, 2005." is prose.
NAME_WORD = r"(?:(?:de|da|do|dos|das|del|della|di|du|van|von|der|den|la|le) )*\p{Lu}[\p{L}'’.\-]*"
AUTHOR_DATE_START = regex.compile(
    rf"\p{{Lu}}[\p{{L}}'’\-]+(?: \p{{Lu}}[\p{{L}}'’\-]+)?,\s(?:and |& )?{NAME_WORD}"
    rf"(?:(?:,? (?:and |& )?| ){NAME_WORD}){{0,12}}(?:,? eds?\.)?[.,]?\s(?:19|20)\d\d[a-z]?\.\s")
# Reference entries with a DOI are up to about 500 chars; a longer prose line can cite a DOI.
MAX_DOI_REFERENCE_CHARS = 600
DOI = regex.compile(r"\bdoi[:.]?\s*10\.\d{4,}|doi\.org/10\.\d{4,}", regex.IGNORECASE)
# "... can be downloaded at http://dx.doi.org/10.1016/...", "... is available at https://doi.org/10.7488/...": a DOI
# link after a lowercase word, without a year, page locator or author before it, is part of a sentence.
DOI_LINK_AFTER_WORD = regex.compile(r"\b\p{Ll}{2,}:?\s+https?://(?:dx\.)?$")
NUMERIC_FRAGMENT = regex.compile(r"[\d;:().,\s\-–*/]+")
# Reference entries glued into one line by Marker: "2001;44:S58-S64", "1999;135:1522–1525", "(1996): 171-178".
CITATION_LOCATOR = regex.compile(
    r"\b(?:19|20)\d\d[a-z]?\)?\s*[;:]\s*\**\d+\**\s*(?:\(\d+\))?\s*[:,]\s*[A-Z]?\d+"
    r"|\((?:19|20)\d\d\):\s*\d+\s*[-–]\s*\d+")
MIN_GLUED_REFERENCES = 2
# Page locators left from split reference entries: "p. 613", "1968. pp. 1-31", "ch7".
PAGE_LOCATOR = regex.compile(r"[\d;:().,\s\-–*/]*(?:(?:pp?|ch|vol|no)\.?\s?\d[\d;:().,\s\-–*/]*)+", regex.IGNORECASE)
REFERENCE_YEAR = regex.compile(r"\b(?:19|20)\d\d[a-z]?\b")
REFERENCE_LOCATOR = regex.compile(r"\d+\s*[-–]\s*\d+|\bpp?\.\s*\d+|\bVol\.|\*\*\d+\*\*|\d+\(\d+\)")
REFERENCE_NAME = regex.compile(r"et al\.|[A-Z][a-z]+,\s[A-Z]\.|\*[^*]{3,}\*")
# "The WJ Picture Vocabulary subtest" is not a surname with initials.
REFERENCE_START = regex.compile(
    r"(?:(?!(?:The|This|That|These|Those|In|On|At|For|From|With|By|As|An?|It|We|Our|If|When|While|To|Of)\b)"
    r"[A-Z][A-Za-z'\-]+,?\s+(?:[A-Z]\.|[A-Z]{1,3}\b|[A-Z][a-z]+,)|\d{1,4}\.\s|\[\d{1,4}\]|\*)"
)
# Reference fragments left by page splits: "*Lancet*, Vol. 2, No. 7892,", "pp. (1600-1605), 1022-1336.".
MAX_FRAGMENT_REFERENCE_CHARS = 200
JOURNAL_LOCATORS = [regex.compile(pattern) for pattern in (
    r"\bVol\.\s*\d", r"\bNo\.\s*\d", r"\bpp?\.\s*\(?\d", r"\bISSN\b|\b(?!(?:1[5-9]|20)\d\d-(?:1[5-9]|20)\d\d\b)\d{4}-\d{3}[\dX]\b",
    r";\s*\**\d+\**\s*[:(]", r"\bv\.\s*\d")]
# Contents entries "Chapter 7 **Title**" and "**Title 167** Author Names" break the contents section at each bold
# line; two in the first 15% of a document start a contents page, which goes on while entries follow each other.
# Marker also writes "Chapter N" entries as headings ("#### Chapter 21 **Gestational Diabetes ... 363**"); a bold
# heading with a number ("### **3.1. Case study 1**") is a section title in the text.
CONTENTS_ENTRY = regex.compile(
    r"(?:(?:#{1,6}\s+|[-*]\s+)?(?P<chapter>Chapter\s+\d{1,3}\s+\*\*[^*]+\*\*)"
    r"|\*\*[^*]*\S\s+(?:\d{1,4}|[IVXLC]{1,6})\*\*)(?:\s+\p{Lu}[^!?]{0,200})?")
CONTENTS_START_SHARE = 0.15
MIN_CONTENTS_ENTRIES = 2
MAX_SHORT_FRAGMENT_CHARS = 80
MAX_REFERENCE_CHARS = 300
INDEX_ENTRY = regex.compile(r"[A-Za-z)][,.]? \d{1,4}(?:[–-]\d{1,4})?(?:, \d{1,4}(?:[–-]\d{1,4})?)*(?= [A-Z(]| *$)")
SENTENCE_END = regex.compile(r"[a-z]{3,}[.?!] [A-Z]")
LETTER = regex.compile(r"\p{L}")
# Prose lines with three numbers before capitals ("at 400 MHz, 13C-NMR at 100 MHz") matched with 3 entries.
MIN_INDEX_ENTRIES = 4
MAX_INDEX_STOP_WORD_SHARE = 0.15
MIN_INDEX_ENTRIES_PER_WORD = 0.12
MIN_INDEX_LETTER_SHARE = 0.3
TOC_WORD = regex.compile(
    r"\b(?:Chapter|Section|Part|Preface|Contents|Introduction|Appendix|Index|Bibliography|References"
    r"|Conclusions?|Foreword)\b|\bAcknowledg", regex.IGNORECASE)
TOC_TAIL = regex.compile(r"\s[*_]*(?:\d{1,4}|[IVX]{1,6})[*_]*\s*$")
# A contents entry stands next to other entries; a lone match is a heading such as "Preface to Volume 2".
MAX_TOC_NEIGHBOUR_DISTANCE = 3
DOT_LEADER = regex.compile(r"(?:\.\s?){4,}\s*\d+\s*$")
MIN_TOC_CHARS = 20
MAX_TOC_CHARS = 250

# A prose line ends a contents, author or reference section; these lists and entries have few function words.
STOP_WORDS = frozenset({"the", "of", "and", "to", "in", "is", "that", "for", "with", "as", "was", "are", "be", "by",
                        "this", "which", "it", "on", "from", "we", "were", "not", "or", "an", "a", "their"})
MIN_PROSE_WORDS = 25
MIN_PROSE_STOP_WORD_SHARE = 0.2
MIN_CONTENT_STOP_WORD_SHARE = 0.12
MIN_LOWERCASE_WORD_SHARE = 0.5
NON_LETTER = regex.compile(r"\P{L}+")
WORD_INITIAL = regex.compile(r"(?<![\p{L}\p{N}])\p{L}(?=\p{L})")
LATEX_MATH = regex.compile(r"\$\$|\\(?:frac|begin|mathbf|mathrm|left|sum|int|sqrt|tag)\b")

OPEN_END = regex.compile(r"[\p{Ll},]$")
# Unmarked headings and running heads ("4.1.1 Antiplatelet agents", "Analysis of Chemical Processes, Determination")
# also end in a lowercase letter; a sentence cut by a page break has more words and function words.
MAX_HEADING_LIKE_CHARS = 120
MIN_CONTINUED_WORDS = 8
MIN_CONTINUED_STOP_WORD_SHARE = 0.1
FIRST_LETTER = regex.compile(r"\p{L}")
SENTENCE_FINAL = regex.compile(r"[.?!:;,)]\W*$")
# Running-head parts ("by Indoor Environment") have two or more capitalized words; sentence ends such as "and Na+" or
# "from the GlcNAc" have one.
MIN_RUNNING_HEAD_WORDS = 2
HYPHEN_END = regex.compile(r"(\p{L}{2,})[-‐]$")
LOWERCASE_START = regex.compile(r"[(\"'“‘]?\p{Ll}{2,}")
LOWERCASE_WORD_START = regex.compile(r"\p{Ll}{2,}")

# Findings "hyphenation_across_line_breaks": "techni‐ cally", "Univer-\n\nsity"; joined only when the joined word
# is known, the analyst's validation (858,090 of 1,016,481 matches).
LINE_BREAK_HYPHEN = regex.compile(r"\b(\p{L}{2,})[-‐](?: |\n\n)(\p{Ll}{2,})\b")
SUSPENDED_HYPHEN_NEXT = frozenset({"and", "or", "to", "nor", "as", "but", "versus", "vs", "than", "und", "oder",
                                   "bis", "et", "ou", "en", "e", "y"})
# Findings "ligature_and_glyph_loss": dropped fi/fl/ff glyphs ("frst", "efect") and split ones ("di fference").
WORD_TOKEN = regex.compile(r"(?<![\p{L}\p{N}\-'’])\p{L}+(?![\p{L}\p{N}\-'’])")
LIGATURE_INSERTS = ("i", "l", "f")
# Acronyms such as "FSH" or "NF" are not dictionary words but are not damaged.
MIN_LIGATURE_WORD_CHARS = 4
SPLIT_LIGATURE = regex.compile(r"\b(\p{L}*f{1,2}i?)\s(f?[fil]?\p{Ll}+)\b|\b(\p{L}+)\s(f{1,2}[il]?\p{Ll}*)\b")
# Springer and Open Book Publishers books show 260-400 drops per 1M chars, IntechOpen 0.84 (findings); the repair
# runs only on documents above noise level.
MIN_LIGATURE_REPAIRS_PER_MILLION_CHARS = 50
MIN_LIGATURE_REPAIRS = 3
# Findings "font_encoding_garble" (c): IntechOpen symbol-font math, '=' as ¼, '+' as þ, '( )' as ð Þ, minus as U+FFFD.
SYMBOL_FONT_EVIDENCE = regex.compile(r"ð[^ðÞ\n]{0,80}Þ|(?<=\S) ¼ (?=\S)")
SYMBOL_FONT_CHARS = str.maketrans({"ð": "(", "Þ": ")", "¼": "=", "þ": "+"})
SYMBOL_FONT_MINUS = regex.compile(r"\ufffd(?=\s?[\d\p{L}(])")
MIN_SYMBOL_FONT_MATCHES = 3
# Findings "footnotes_and_markers": <sup>n</sup> after a word is a note or citation marker, after a number or unit
# it is an exponent.
FOOTNOTE_MARKER = regex.compile(r"(?:(?<=\p{L}{3}|[.,;:)\]\"”’])\s?|(?<=\p{L}{3}) )<sup>[\d,\s–\-*†‡§]+</sup>")
ORDINAL_SUFFIX = regex.compile(r"(?<=\d)<sup>(st|nd|rd|th)</sup>")
SUP_OR_SUB = regex.compile(r"<(su[pb])>(.*?)</\1>")
SUPERSCRIPT = str.maketrans("0123456789+-−=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱ")
SUBSCRIPT = str.maketrans("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎")
SUPERSCRIPT_CHARS = frozenset("0123456789+-−=()ni")
SUBSCRIPT_CHARS = frozenset("0123456789+-−=()")
ESCAPED_MARKDOWN = regex.compile(r"\\([_*])")
# Marker splits "&lt;" as "<sup>&</sup>lt;", so the entity appears only once the tags are gone.
HTML_ENTITY = regex.compile(r"&(?:lt|gt|amp|quot|#\d{1,7}|#x[\da-fA-F]{1,6});")

# Findings "font_encoding_garble" (a, b): unreadable custom-font text and Caesar-shifted words; "iru" is left out
# because it is also Japanese romaji.
GARBLE_CHARS = regex.compile(r"[ƀ-ȗȜ-ɏ¢¡]")
LATIN1_LETTER = regex.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
# Counted per token: garbled fonts give tokens without any Latin-1 letter ("ǰ", "řǯŘǯ"), while African orthographies in
# English grammars and dictionaries (Ik "ɨƙáƙá") mix these letters with Latin-1 ones.
MIN_GARBLE_TOKENS = 20
MAX_GARBLE_TOKEN_SHARE = 0.02
# Garbled MDPI pages sit next to readable ones in the same article; most tokens of their paragraphs and headings are
# garbled.
MIN_GARBLE_PARAGRAPH_TOKEN_SHARE = 0.3
CAESAR_WORD = regex.compile(r"\b(?:wkh|dqg|zlwk|wkdw|WKH|DQG)\b")
MAX_CAESAR_WORD_SHARE = 0.005

# Gopher minimum document length (Rae et al., arXiv:2112.11446, App. A.1.1); findings: 13.8% of docs have <30 body
# words (bare headings, editor lines).
MIN_WORDS = 50
# Read docs below 0.1 were reference debris, title pages, taxon and affiliation lists; 0.1-0.3 held real sections.
MIN_CONTENT_SHARE = 0.1
# Findings language_mix: math, code and reference chunks get unreliable labels, so only segments that lid.176
# decides with probability >= 0.5 count, as in the analyst's calibrated share.
MIN_SEGMENT_LANGUAGE_PROBABILITY = 0.5
# Calibrated on read books: English monographs quoting Italian/Latin (Dante and Aquinas, 0.68), grammars of German
# written in English (0.74) and catalogues with Polish names (0.75) stay; a French/English bilingual edition (0.59)
# and foreign reference lists, title pages and tables (0.5-0.65) go.
ENGLISH_MIN_SHARE = 0.65
# A run of >= 5000 letters (about two pages) of non-English segments is a foreign chapter or document, not a quote;
# read books with >= 10% of their text in such runs were multilingual collections.
MIN_FOREIGN_BLOCK_LETTERS = 5000
MAX_FOREIGN_BLOCK_SHARE = 0.1
# When lid.176 decides < 10% of the letters, the decided rest must be clearly English: Tibetan Wylie transliteration
# (The Ocean of Heroes) scored 0.5-0.8 on 2-6% decided letters, math sections 0.9-1.0.
MIN_DECIDED_LETTER_SHARE = 0.1
LOW_EVIDENCE_ENGLISH_MIN_SHARE = 0.9
# English acknowledgement and caption paragraphs that name foreign institutions ("Fundação de Amparo à Pesquisa do
# Estado do Rio de Janeiro") get English as the second label at 0.25-0.35; in 3,421 decided foreign segments of
# multilingual books English had 0.03 at the 95th percentile and >= 0.25 in 0.2%.
MIN_SEGMENT_ENGLISH_PROBABILITY = 0.25
# Read English sections at 0.48-0.64 lost their share to short foreign segments (reference entries, affiliations,
# questionnaire items); the bilingual edition (0.46) and the multilingual books of shards 5, 8 and 11 (0.14-0.47)
# have that share of their letters in foreign segments of >= 200 letters.
GRAY_ENGLISH_MIN_SHARE = 0.4
MIN_FOREIGN_PROSE_LETTERS = 200
MAX_FOREIGN_PROSE_SHARE = 0.1

# Findings "paragraph_level": containment over paragraphs of >= 200 chars finds the 1,487 books that are >= 95%
# inside other books.
MIN_HASHED_PARAGRAPH_CHARS = 200
# Drops read in the 0.8-0.9 band were IntechOpen back-cover blurbs inside the preface and chapter PDFs inside their
# books; below 0.8 a dropped document keeps more than a fifth of unseen text.
MIN_CONTAINED_SHARE = 0.8
MAX_CONTAINMENT_ROUNDS = 50


def paragraph_key(paragraph: str) -> str:
    text = unicodedata.normalize("NFD", paragraph.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return WHITESPACE.sub(" ", NUMBER.sub("0", NON_WORD.sub(" ", text))).strip()


def paragraph_hash(paragraph: str) -> int | None:
    key = paragraph_key(paragraph)
    if len(key) < MIN_PARAGRAPH_KEY_CHARS:
        return None
    return xxhash.xxh64_intdigest(key.encode())


def shard_paragraph_books(shard: Path) -> np.ndarray:
    """Distinct (paragraph hash, book hash) pairs of one shard, as an (n, 2) uint64 array."""
    pairs = set()
    for document in read_documents(shard):
        book = xxhash.xxh64_intdigest((document["metadata"].get("book_id") or document["id"]).encode())
        for paragraph in PARAGRAPH_BREAK.split(document["text"]):
            key = paragraph_hash(paragraph)
            if key is not None:
                pairs.add((key, book))
    return np.array(sorted(pairs), dtype=np.uint64).reshape(-1, 2)


def build_boilerplate_table(input_dir: Path, out_path: Path, processes: int) -> int:
    """Writes the sorted hashes of paragraphs that occur in at least MIN_BOILERPLATE_BOOKS books."""
    with ProcessPoolExecutor(processes) as pool:
        pairs = np.unique(np.concatenate(list(pool.map(shard_paragraph_books, list_shards(input_dir)))), axis=0)
    hashes, book_counts = np.unique(pairs[:, 0], return_counts=True)
    table = hashes[book_counts >= MIN_BOILERPLATE_BOOKS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomically(out_path, table.astype(np.uint64).tobytes())
    return len(table)


@cache
def boilerplate_table() -> np.ndarray:
    return np.fromfile(BOILERPLATE_TABLE_PATH, dtype=np.uint64)


@cache
def wordlist() -> frozenset[str]:
    return frozenset(word.strip().lower() for word in WORDLIST_PATH.read_text().split("\n") if word.strip())


def bare(line: str) -> str:
    return MARKUP.sub("", SUP_OR_SUB_TAG.sub("", line)).strip()


def is_heading(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) < MAX_HEADING_CHARS and bool(HEADING.fullmatch(stripped))


def is_boilerplate_line(line: str) -> bool:
    plain = bare(line[:BOILERPLATE_SCAN_CHARS])
    return bool(BOILERPLATE.search(plain) or ADMIN_STATEMENT.match(plain))


def strip_running_stamps(line: str) -> str:
    if not STAMP_HINT.search(line):
        return line
    return MULTIPLE_SPACES.sub(" ", RUNNING_STAMP.sub(" ", line)).strip()


def strip_boilerplate_sentences(line: str) -> str:
    return " ".join(sentence for sentence in SENTENCE_BREAK.split(line) if not BOILERPLATE.search(bare(sentence)))


def is_cross_book_paragraph(line: str, table: np.ndarray) -> bool:
    if is_heading(line) and len(bare(line)) < MAX_SECTION_HEADING_CHARS:
        return False
    key = paragraph_hash(line)
    if key is None:
        return False
    position = np.searchsorted(table, np.uint64(key))
    return position < len(table) and table[position] == key


def is_author_year_start(line: str) -> bool:
    match = AUTHOR_YEAR_START.match(line)
    return match is not None and (match.group("initial") is not None
                                  or stop_word_share(line) < MIN_PROSE_STOP_WORD_SHARE)


def starts_like_reference(line: str) -> bool:
    return (any(pattern.match(line) for pattern in REFERENCE_PREFIX) or is_author_year_start(line)
            or bool(AUTHOR_DATE_START.match(line)))


def is_reference_fragment(line: str) -> bool:
    if len(line) >= MAX_FRAGMENT_REFERENCE_CHARS:
        return False
    locators = sum(1 for pattern in JOURNAL_LOCATORS if pattern.search(line))
    return locators >= 2 or (locators == 1 and stop_word_share(line) < MIN_PROSE_STOP_WORD_SHARE)


def is_reference_doi(before: str) -> bool:
    """Whether a DOI ends a reference entry, given the text before it."""
    if is_prose(before):
        return False
    has_reference_marks = REFERENCE_YEAR.search(before) or REFERENCE_LOCATOR.search(before) \
        or REFERENCE_NAME.search(before)
    return bool(has_reference_marks) or not DOI_LINK_AFTER_WORD.search(before)


def is_reference_line(line: str) -> bool:
    """Analyst rule (23/25 reference lines on a read sample), with prefixes tightened after reading removed prose."""
    # Humanities prose cites "(Sperber and Wilson 1995: 227, 247)" and can open "Recently, Giesen et al. (2020)".
    if len(line) >= MAX_REFERENCE_CHARS and is_prose(line):
        return False
    if len(CITATION_LOCATOR.findall(line)) >= MIN_GLUED_REFERENCES:
        return True
    if any(pattern.match(line) for pattern in REFERENCE_PREFIX):
        return True
    # Read lines starting "Surname, X. (1994)." outside a References section were reference entries.
    if len(line) <= MAX_DOI_REFERENCE_CHARS and (is_author_year_start(line) or AUTHOR_DATE_START.match(line)):
        return True
    doi = DOI.search(line) if len(line) <= MAX_DOI_REFERENCE_CHARS else None
    if doi is not None and is_reference_doi(line[:doi.start()]):
        return True
    if is_reference_fragment(line):
        return True
    if len(line) < MAX_SHORT_FRAGMENT_CHARS and PAGE_LOCATOR.fullmatch(line):
        return True
    if len(line) < MAX_SHORT_FRAGMENT_CHARS and NUMERIC_FRAGMENT.fullmatch(line) \
            and (REFERENCE_YEAR.search(line) or regex.search(r"\d+\s*[-–]\s*\d+", line)):
        return True
    if len(line) >= MAX_REFERENCE_CHARS or not REFERENCE_START.match(line):
        return False
    return bool(REFERENCE_YEAR.search(line) and REFERENCE_LOCATOR.search(line) and REFERENCE_NAME.search(line))


def is_index_line(line: str) -> bool:
    entries = len(INDEX_ENTRY.findall(line))
    if entries < MIN_INDEX_ENTRIES:
        return False
    return (entries / len(line.split()) >= MIN_INDEX_ENTRIES_PER_WORD
            and len(LETTER.findall(line)) >= MIN_INDEX_LETTER_SHARE * len(line)
            and len(SENTENCE_END.findall(line)) <= 1
            and stop_word_share(line) < MAX_INDEX_STOP_WORD_SHARE)


def is_toc_line(line: str) -> bool:
    plain = line.strip()
    if DOT_LEADER.search(plain):
        return True
    return (MIN_TOC_CHARS <= len(plain) < MAX_TOC_CHARS and bool(TOC_TAIL.search(plain))
            and bool(TOC_WORD.search(plain)) and not SENTENCE_END.search(plain))


def plain_words(line: str) -> list[str]:
    return [word for word in (NON_LETTER.sub("", token) for token in bare(line).lower().split()) if word]


def stop_word_share(line: str) -> float:
    words = plain_words(line)
    return sum(1 for word in words if word in STOP_WORDS) / len(words) if words else 0.0


def is_prose(line: str) -> bool:
    """At least 25 words with a function-word share of 0.2; reference entries read had a median share of 0.15."""
    plain = bare(line)
    if starts_like_reference(plain):
        return False
    return len(plain_words(line)) >= MIN_PROSE_WORDS and stop_word_share(line) >= MIN_PROSE_STOP_WORD_SHARE


def is_credit_line(line: str, plain: str) -> bool:
    """Editor credit or author byline: names separated by commas and "and", optionally with affiliation markers."""
    if len(plain) > MAX_CREDIT_LINE_CHARS or plain.endswith("."):
        return False
    marked = len(AFFILIATION_MARKER.findall(plain)) >= MIN_BYLINE_MARKERS
    if line.lstrip().startswith("#") and not marked:
        return False
    if EDITOR_CREDIT.match(plain):
        return True
    affiliations = AFFILIATION_TAIL.search(line)
    byline = bare(line[:affiliations.start()]) if affiliations else plain
    unmarked = AFFILIATION_MARKER.sub(" ", SPACING_DIACRITIC.sub("", byline))
    names = [name.split() for name in NAME_SEPARATOR.split(unmarked) if name.strip()]
    if not all(len(tokens) <= MAX_NAME_TOKENS and all(NAME_TOKEN.fullmatch(token) for token in tokens)
               for tokens in names):
        return False
    # A name has a given name and a surname; one-word items are title words ("Quality, Safety, and Authenticity").
    one_token_names = sum(1 for tokens in names if len(tokens) < MIN_NAME_TOKENS)
    if len(names) - one_token_names < (MIN_AFFILIATED_BYLINE_NAMES if affiliations else MIN_BYLINE_NAMES):
        return False
    return one_token_names == 0 or (marked and one_token_names <= MAX_ONE_TOKEN_NAMES)


def is_repeated_phrase(plain: str) -> bool:
    words = plain.lower().split()
    if len(words) < MIN_REPEATED_PHRASE_WORDS:
        return False
    bigrams = Counter(zip(words, words[1:]))
    return bigrams.most_common(1)[0][1] >= MAX_TOP_BIGRAM_SHARE * (len(words) - 1)


def is_caption(plain: str) -> bool:
    match = CAPTION.match(plain)
    if match is None or len(plain) >= MAX_CAPTION_CHARS:
        return False
    if match.group("separator"):
        return True
    return not (CAPTION_VERB.search(plain[:CAPTION_VERB_SCAN_CHARS])
                or CAPTION_SUBJECT.match(plain, match.end("label")))


def is_affiliation(line: str, plain: str) -> bool:
    return (not line.lstrip().startswith("#") and len(plain) <= MAX_AFFILIATION_CHARS
            and bool(AFFILIATION.fullmatch(plain.rstrip("."))) and bool(AFFILIATION_MARK.search(line)))


def running_header_key(line: str) -> str:
    return paragraph_key(EDGE_NUMBERS.sub("", bare(line)))


def title_key(title: str) -> str:
    return paragraph_key(bare(title))


def is_title_header(key: str, book_title: str) -> bool:
    """IntechOpen heads carry the full book title while the metadata title can stop at the subtitle dash."""
    return len(book_title) >= MIN_RUNNING_HEADER_KEY_CHARS and key.startswith(book_title)


def edge_header_candidate(line: str) -> bool:
    plain = bare(line)
    return (not line.lstrip().startswith("#") and len(plain) <= MAX_RUNNING_HEADER_CHARS
            and bool(EDGE_NUMBER.search(plain))
            and not SECTION_NUMBERED.match(plain) and not plain.endswith((".", "?", "!", ":", ";")))


def furniture_rule(line: str, plain: str, book_title: str, header_keys: Counter, table: np.ndarray
                   ) -> tuple[str, str] | None:
    if PAGE_NUMBER.fullmatch(plain):
        return "page_number", ""
    if is_repeated_phrase(plain) and not LATEX_MATH.search(line):
        return "repeated_phrase", ""
    if is_boilerplate_line(line):
        remainder = ""
        if len(plain) > MAX_BOILERPLATE_LINE_CHARS and not ADMIN_STATEMENT.match(plain):
            remainder = strip_boilerplate_sentences(line)
        if len(remainder.split()) >= MIN_PROSE_WORDS:
            return "boilerplate_sentences", remainder
        return "boilerplate", ""
    if is_cross_book_paragraph(line, table):
        return "cross_book", ""
    if edge_header_candidate(line):
        key = running_header_key(line)
        if is_title_header(key, book_title) or (len(key) >= MIN_RUNNING_HEADER_KEY_CHARS and header_keys[key] >= 2):
            return "running_header", ""
    if is_caption(plain):
        return "caption", ""
    if is_affiliation(line, plain):
        return "affiliation", ""
    if is_credit_line(line, plain):
        return "credit", ""
    if EMAIL.search(plain) and not is_prose(line):
        return "contact", ""
    return None


def furniture_lines(lines: list[str], title: str) -> dict[int, tuple[str, str]]:
    """Rule name and replacement ("" to remove) for each page number, running header or stamp, caption,
    affiliation or boilerplate line."""
    book_title = title_key(title)
    unstamped = [strip_running_stamps(line) for line in lines]
    header_keys = Counter(running_header_key(line) for line in unstamped if edge_header_candidate(line))
    table = boilerplate_table()
    removed = {}
    in_intech_ad = False
    for index, line in enumerate(unstamped):
        plain = bare(line)
        if not plain:
            if line != lines[index]:
                removed[index] = ("running_stamp", "")
            continue
        if in_intech_ad and INTECH_AD_LINE.fullmatch(plain):
            removed[index] = ("boilerplate", "")
            continue
        rule = furniture_rule(line, plain, book_title, header_keys, table)
        in_intech_ad = bool(INTECH_AD_START.search(plain)) or (in_intech_ad and rule == ("boilerplate", ""))
        if rule is not None:
            removed[index] = rule
        elif line != lines[index]:
            removed[index] = ("running_stamp", line)
    return removed


def back_matter_lines(lines: list[str]) -> dict[int, str]:
    """Rule name for each line of a reference list, contents or author list (to the next heading or prose line),
    index tail, or a stray reference, index or contents line."""
    removed = {}
    toc_candidates = []
    contents_entries = []
    total_chars = sum(len(line) + 1 for line in lines)
    offset = 0
    content_position = 0
    section = None
    for index, line in enumerate(lines):
        line_start = offset
        offset += len(line) + 1
        plain = bare(line)
        if not plain:
            continue
        content_position += 1
        entry = CONTENTS_ENTRY.fullmatch(line.strip())
        contents_entry = entry is not None and not SENTENCE_END.search(line)
        if contents_entry:
            contents_entries.append((content_position, index, line_start < CONTENTS_START_SHARE * total_chars,
                                     entry.group("chapter") is not None))
        heading = is_heading(line) or len(plain) < MAX_HEADING_CHARS // 4
        if heading and INDEX_HEADING.fullmatch(plain) and line_start >= INDEX_TAIL_START_SHARE * total_chars:
            removed.update(dict.fromkeys(range(index, len(lines)), "index_tail"))
            break
        if heading and REFERENCE_HEADING.fullmatch(plain):
            section = "references"
        elif heading and (CONTENTS_HEADING.fullmatch(plain) or ADMIN_HEADING.fullmatch(plain)):
            section = "list"
        # A Markdown heading ends a contents list even when it is an entry: the shuffled preface lines after
        # "#### **Preface XI**" are sentence ends, not entries.
        elif is_heading(line) and not (section == "references" and is_reference_line(plain)) \
                and not (section == "list" and contents_entry and not line.lstrip().startswith("#")):
            section = None
        if section is not None:
            if not is_prose(line) or (section == "references" and is_reference_line(plain)):
                removed[index] = f"{section}_section"
                continue
            section = None
        if is_reference_line(plain):
            removed[index] = "reference_line"
        elif is_index_line(plain):
            removed[index] = "index_line"
        elif is_toc_line(line):
            toc_candidates.append((content_position, index))
    for position, (content_index, index) in enumerate(toc_candidates):
        neighbours = toc_candidates[max(0, position - 1):position] + toc_candidates[position + 1:position + 2]
        if any(abs(content_index - other) <= MAX_TOC_NEIGHBOUR_DISTANCE for other, _ in neighbours):
            removed[index] = "toc_line"
    removed.update(contents_page_lines(lines, contents_entries))
    return removed


def contents_page_lines(lines: list[str], entries: list[tuple[int, int, bool, bool]]) -> dict[int, str]:
    """Contents entries (content position, line index, early, "Chapter N" form) that follow each other from a start.
    Two early entries, or two "Chapter N **Title**" entries next to each other, start a contents page. In a run with
    a "Chapter N" entry, which only contents pages have, the lines between entries that are not prose (author names,
    wrapped title parts) go too."""
    early_start = sum(1 for _, _, early, _ in entries if early) >= MIN_CONTENTS_ENTRIES
    chapter_positions = [position for position, _, _, chapter in entries if chapter]
    runs = []
    for entry in entries:
        position, _, early, chapter = entry
        if runs and position - runs[-1][-1][0] <= MAX_TOC_NEIGHBOUR_DISTANCE:
            runs[-1].append(entry)
        elif (early_start and early) or (chapter and any(
                0 < abs(position - other) <= MAX_TOC_NEIGHBOUR_DISTANCE for other in chapter_positions)):
            runs.append([entry])
    removed = {}
    for run in runs:
        removed.update(dict.fromkeys((index for _, index, _, _ in run), "toc_line"))
        if not any(chapter for _, _, _, chapter in run):
            continue
        for (_, start, _, _), (_, end, _, _) in zip(run, run[1:]):
            for between in range(start + 1, end):
                if bare(lines[between]) and not is_prose(lines[between]):
                    removed[between] = "toc_line"
    return removed


def drop_repeated_paragraphs(text: str) -> str:
    """Keeps the first copy of each paragraph of >= 200 chars; IntechOpen PDFs with two text layers repeat
    paragraphs a few paragraphs later (IB-ET, arXiv:2608.19026, dedups paragraphs within a book)."""
    seen = set()
    kept = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        stripped = paragraph.strip()
        key = paragraph_hash(stripped) if len(stripped) >= MIN_HASHED_PARAGRAPH_CHARS else None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept)


def drop_emptied_headings(lines: list[str]) -> list[str]:
    """Removes admin headings (References, Author details, ...) followed directly by another heading or the end."""
    content = [index for index, line in enumerate(lines) if line.strip()]
    emptied = set()
    for position, index in enumerate(content):
        following = lines[content[position + 1]] if position + 1 < len(content) else None
        if EMPTIED_HEADING.fullmatch(bare(lines[index])) and (following is None or is_heading(following)):
            emptied.add(index)
    return [line for index, line in enumerate(lines) if index not in emptied]


def strip_page_furniture(document: dict) -> str:
    lines = document["text"].split("\n")
    removed = furniture_lines(lines, document["metadata"].get("title") or "")
    return "\n".join(removed[index][1] if index in removed else line for index, line in enumerate(lines))


def strip_back_matter(text: str) -> str:
    lines = text.split("\n")
    removed = back_matter_lines(lines)
    return "\n".join(drop_emptied_headings([line for index, line in enumerate(lines) if index not in removed]))


def document_vocabulary(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in WORD_TOKEN.findall(text))


def is_known_word(word: str, vocabulary: frozenset[str]) -> bool:
    lowered = word.lower()
    return lowered in vocabulary or lowered in wordlist()


def dehyphenate(text: str) -> str:
    vocabulary = document_vocabulary(text)

    def join(match: regex.Match) -> str:
        first, second = match.group(1), match.group(2)
        if second in SUSPENDED_HYPHEN_NEXT:
            return match.group(0)
        if is_known_word(first + second, vocabulary):
            return first + second
        if f"{first.lower()}-{second}" in text.lower():
            return f"{first}-{second}"
        return match.group(0)

    return LINE_BREAK_HYPHEN.sub(join, text)


def ligature_candidate(word: str, vocabulary: frozenset[str]) -> str | None:
    """The known word made by inserting i, l or f after an f of an unknown word, when exactly one exists."""
    if len(word) < MIN_LIGATURE_WORD_CHARS or not (word.islower() or word.istitle()):
        return None
    if "f" not in word.lower() or word.lower() in wordlist():
        return None
    candidates = set()
    for position, char in enumerate(word):
        if char.lower() != "f":
            continue
        for insert in LIGATURE_INSERTS:
            candidate = word[:position + 1] + insert + word[position + 1:]
            if is_known_word(candidate, vocabulary):
                candidates.add(candidate)
    return candidates.pop() if len(candidates) == 1 else None


def join_split_ligature(match: regex.Match) -> str:
    first = match.group(1) or match.group(3)
    second = match.group(2) or match.group(4)
    joined = first + second
    # "e ffect": the single letter is a dictionary word, so only one part needs to be unknown.
    if joined.lower() in wordlist() and not (first.lower() in wordlist() and second.lower() in wordlist()):
        return joined
    return match.group(0)


def repair_ligatures(text: str) -> str:
    text = SPLIT_LIGATURE.sub(join_split_ligature, text)
    vocabulary = document_vocabulary(text)
    replacements = {}
    for word in set(WORD_TOKEN.findall(text)):
        candidate = ligature_candidate(word, vocabulary)
        if candidate is not None:
            replacements[word] = candidate
    occurrences = sum(text.count(word) for word in replacements)
    if occurrences < max(MIN_LIGATURE_REPAIRS, MIN_LIGATURE_REPAIRS_PER_MILLION_CHARS * len(text) / 1_000_000):
        return text
    return WORD_TOKEN.sub(lambda match: replacements.get(match.group(0), match.group(0)), text)


def repair_symbol_font(text: str) -> str:
    if len(SYMBOL_FONT_EVIDENCE.findall(text)) < MIN_SYMBOL_FONT_MATCHES:
        return text
    return SYMBOL_FONT_MINUS.sub("−", text.translate(SYMBOL_FONT_CHARS))


def render_script(match: regex.Match) -> str:
    tag, content = match.group(1), match.group(2)
    allowed, table = (SUPERSCRIPT_CHARS, SUPERSCRIPT) if tag == "sup" else (SUBSCRIPT_CHARS, SUBSCRIPT)
    if content and set(content) <= allowed:
        return content.translate(table)
    return content


def clean_markup(text: str) -> str:
    """Drops note markers, writes exponents and indices as Unicode scripts and removes Markdown escapes."""
    text = FOOTNOTE_MARKER.sub("", text)
    text = ORDINAL_SUFFIX.sub(r"\1", text)
    text = SUP_OR_SUB.sub(render_script, text)
    text = HTML_ENTITY.sub(lambda match: html.unescape(match.group(0)), SUP_OR_SUB_TAG.sub("", text))
    return ESCAPED_MARKDOWN.sub(r"\1", text)


def is_flowing_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    return bool(stripped) and not is_heading(stripped) and not LIST_ITEM.match(stripped) \
        and not TABLE_OR_MATH.match(stripped)


def is_title_case(plain: str) -> bool:
    initials = [FIRST_LETTER.search(word) for word in plain.split()
                if NON_LETTER.sub("", word).lower() not in STOP_WORDS]
    initials = [initial.group(0) for initial in initials if initial is not None]
    return bool(initials) and all(initial.isupper() for initial in initials)


def is_short_line(paragraph: str) -> bool:
    stripped = paragraph.strip()
    return "\n" not in stripped and len(stripped) < MAX_HEADING_LIKE_CHARS


def can_continue(previous: str, header_keys: Counter, book_title: str) -> bool:
    """Whether a paragraph can be the first part of a sentence split by a page break."""
    if len(plain_words(previous)) < MIN_CONTINUED_WORDS or stop_word_share(previous) < MIN_CONTINUED_STOP_WORD_SHARE:
        return False
    if not is_short_line(previous):
        return True
    plain = bare(previous)
    key = running_header_key(previous)
    return not (SECTION_NUMBERED.match(plain) or is_title_case(plain) or header_keys[key] >= 2
                or is_title_header(key, book_title))


def can_be_continued(paragraph: str) -> bool:
    """Whether a paragraph can be the second part of a sentence split by a page break; the second line of a
    two-line running head ("from Normal and Dissociated Strabismus Subjects Through Morphological Transformations")
    starts in lowercase. Repetition and the book title are no evidence here: IntechOpen PDFs with two text layers
    repeat every line, and sentences continue with the words of the title ("crisis management work in ...")."""
    if not is_short_line(paragraph):
        return True
    plain = bare(paragraph)
    content_words = sum(1 for word in plain_words(paragraph) if word not in STOP_WORDS)
    return not (content_words >= MIN_RUNNING_HEAD_WORDS and is_title_case(plain) and not SENTENCE_FINAL.search(plain))


def rejoin_paragraphs(document: dict) -> str:
    """Joins a paragraph that ends mid-sentence with a following paragraph that starts in lowercase.

    Findings "broken_reading_order": 836,414 such pairs are adjacent after page breaks, and more become adjacent
    once running headers, captions and boilerplate between them are removed.
    """
    text = document["text"]
    book_title = title_key(document["metadata"].get("title") or "")
    vocabulary = document_vocabulary(text)
    paragraphs = [paragraph for paragraph in PARAGRAPH_BREAK.split(text) if paragraph.strip()]
    header_keys = Counter(key for key in (running_header_key(paragraph) for paragraph in paragraphs
                                          if is_short_line(paragraph))
                          if len(key) >= MIN_RUNNING_HEADER_KEY_CHARS)
    joined: list[str] = []
    for paragraph in paragraphs:
        previous = joined[-1] if joined else ""
        if previous and is_flowing_paragraph(previous) and is_flowing_paragraph(paragraph) \
                and can_continue(previous, header_keys, book_title) \
                and can_be_continued(paragraph):
            hyphen = HYPHEN_END.search(previous)
            second = LOWERCASE_WORD_START.match(paragraph.lstrip())
            if hyphen and second and is_known_word(hyphen.group(1) + second.group(0), vocabulary):
                joined[-1] = previous[:-1] + paragraph.lstrip()
                continue
            if OPEN_END.search(previous.rstrip()) and LOWERCASE_START.match(paragraph.lstrip()):
                joined[-1] = previous.rstrip() + " " + paragraph.lstrip()
                continue
        joined.append(paragraph)
    return "\n\n".join(joined)


def garble_token_counts(text: str) -> tuple[int, int]:
    """Tokens that hold a garble char and no Latin-1 letter, and all tokens."""
    tokens = bare(text).split()
    garbled = sum(1 for token in tokens if GARBLE_CHARS.search(token) and not LATIN1_LETTER.search(token))
    return garbled, len(tokens)


def drop_garbled_paragraphs(text: str) -> str:
    kept = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        garbled, tokens = garble_token_counts(paragraph)
        if garbled and garbled >= MIN_GARBLE_PARAGRAPH_TOKEN_SHARE * tokens:
            continue
        kept.append(paragraph)
    return "\n\n".join(kept)


def is_garbled(text: str) -> bool:
    garbled, tokens = garble_token_counts(text)
    if garbled >= MIN_GARBLE_TOKENS and garbled >= MAX_GARBLE_TOKEN_SHARE * tokens:
        return True
    words = len(text.split())
    return words > 0 and len(CAESAR_WORD.findall(text)) / words >= MAX_CAESAR_WORD_SHARE


def is_content_paragraph(paragraph: str) -> bool:
    """Running text or LaTeX math. Methods prose read in the sample has function-word shares down to 0.12; author,
    affiliation and reference paragraphs are mostly capitalized words."""
    if LATEX_MATH.search(paragraph):
        return True
    words = plain_words(paragraph)
    if len(words) < MIN_PROSE_WORDS:
        return False
    # Numbers and units ("(145 mg, 0.7 mmol)") are not words, so they do not count against the lowercase share.
    initials = [match.group(0) for match in WORD_INITIAL.finditer(bare(paragraph))]
    lowercase = sum(1 for initial in initials if initial.islower())
    return (stop_word_share(paragraph) >= MIN_CONTENT_STOP_WORD_SHARE
            and lowercase >= MIN_LOWERCASE_WORD_SHARE * len(initials))


def content_share(text: str) -> float:
    """Share of chars in content paragraphs; the rest is tables, lists, bylines, title pages and reference debris."""
    content = sum(len(paragraph.strip()) for paragraph in PARAGRAPH_BREAK.split(text)
                  if is_content_paragraph(paragraph.strip()))
    return content / len(text) if text else 0.0


def has_min_words(text: str) -> bool:
    return len(text.split()) >= MIN_WORDS


@dataclass(frozen=True)
class LanguageProfile:
    english_share: float
    foreign_block_share: float
    decided_letter_share: float
    foreign_prose_share: float


class EnglishProfiler:
    """Shares of English letters, foreign-block letters and foreign-prose letters in the segments that are English
    or decided, and the share of decided letters, over the language segments of a text."""

    def __init__(self):
        self.identifier = LanguageIdentifier()

    def __call__(self, text: str) -> LanguageProfile:
        segments = language_segments(text)
        if not segments:
            return LanguageProfile(0.0, 0.0, 0.0, 0.0)
        labels, probabilities = self.identifier.model.predict([fasttext_input(segment) for segment in segments], k=2)
        english_letters = counted_letters = decided_letters = all_letters = 0
        block_letters = run_letters = prose_letters = 0
        for segment, segment_labels, segment_probabilities in zip(segments, labels, probabilities):
            letters = len(LETTER.findall(segment))
            all_letters += letters
            decided = segment_probabilities[0] >= MIN_SEGMENT_LANGUAGE_PROBABILITY
            english_probability = dict(zip(segment_labels, segment_probabilities)).get("__label__en", 0.0)
            english = (english_probability >= MIN_SEGMENT_ENGLISH_PROBABILITY
                       or (decided and segment_labels[0] == "__label__en"))
            if decided:
                decided_letters += letters
            if english:
                counted_letters += letters
                english_letters += letters
                if run_letters >= MIN_FOREIGN_BLOCK_LETTERS:
                    block_letters += run_letters
                run_letters = 0
            elif decided:
                counted_letters += letters
                run_letters += letters
                if letters >= MIN_FOREIGN_PROSE_LETTERS:
                    prose_letters += letters
        if run_letters >= MIN_FOREIGN_BLOCK_LETTERS:
            block_letters += run_letters
        if not counted_letters:
            # No segment gives evidence, so one prediction on the whole text decides at the low-evidence floor.
            language, _ = self.identifier.predict(text)
            return LanguageProfile(1.0 if language == "en" else 0.0, 0.0, 0.0, 0.0)
        return LanguageProfile(english_letters / counted_letters, block_letters / counted_letters,
                               decided_letters / all_letters, prose_letters / counted_letters)


def is_english_document(profile: LanguageProfile) -> bool:
    if profile.foreign_block_share >= MAX_FOREIGN_BLOCK_SHARE:
        return False
    if profile.decided_letter_share < MIN_DECIDED_LETTER_SHARE:
        return profile.english_share >= LOW_EVIDENCE_ENGLISH_MIN_SHARE
    return profile.english_share >= ENGLISH_MIN_SHARE or (
        profile.english_share >= GRAY_ENGLISH_MIN_SHARE and profile.foreign_prose_share < MAX_FOREIGN_PROSE_SHARE)


class ContainmentIndexer:
    def __init__(self, directory: Path, shard_index: int):
        self.prefix = shard_prefix(directory, shard_index)
        self.keys: list[int] = []
        self.local_ids: list[int] = []
        self.chars: list[int] = []
        self.lengths: list[int] = []

    def add(self, local_id: int, document: dict) -> None:
        text = document["text"]
        self.lengths.append(len(text))
        seen = set()
        for paragraph in PARAGRAPH_BREAK.split(text):
            paragraph = paragraph.strip()
            if len(paragraph) < MIN_HASHED_PARAGRAPH_CHARS:
                continue
            key = paragraph_hash(paragraph)
            if key is None or key in seen:
                continue
            seen.add(key)
            self.keys.append(key)
            self.local_ids.append(local_id)
            self.chars.append(len(paragraph))

    def close(self) -> None:
        for suffix, values, dtype in ((".ckeys.npy", self.keys, np.uint64), (".cids.npy", self.local_ids, np.uint32),
                                      (".cchars.npy", self.chars, np.uint32),
                                      (".clengths.npy", self.lengths, np.uint64)):
            save_array_atomically(self.prefix.with_name(self.prefix.name + suffix), np.array(values, dtype=dtype))


def contained_documents(keys: np.ndarray, doc_ids: np.ndarray, chars: np.ndarray, lengths: np.ndarray,
                        live: np.ndarray, min_share: float) -> tuple[np.ndarray, int]:
    """Kept copy for each document whose paragraphs cover at least min_share of its text inside longer kept
    documents, else -1; and the number of fixpoint rounds.

    Longer documents win (ties: smaller id), as when documents are scanned from the longest down, so a chapter is
    dropped in favour of its book. A document's status depends only on longer documents, so the rounds settle
    after at most the depth of the longest containment chain.
    """
    document_count = len(lengths)
    priority = np.empty(document_count, dtype=np.int64)
    priority[np.lexsort((-np.arange(document_count), lengths))] = np.arange(document_count)
    rows = live[doc_ids]
    keys, doc_ids, chars = keys[rows], doc_ids[rows], chars[rows].astype(np.int64)
    order = np.lexsort((-priority[doc_ids], keys))
    keys, doc_ids, chars = keys[order], doc_ids[order], chars[order]
    starts = np.ones(len(keys), dtype=bool)
    starts[1:] = keys[1:] != keys[:-1]
    group = np.cumsum(starts) - 1
    kept = live.copy()
    duplicate_of = np.full(document_count, -1, dtype=np.int64)
    for rounds in range(1, MAX_CONTAINMENT_ROUNDS + 1):
        kept_rows = np.flatnonzero(kept[doc_ids])
        first_groups, first_positions = np.unique(group[kept_rows], return_index=True)
        owner_of_group = np.full(group[-1] + 1 if len(group) else 0, -1, dtype=np.int64)
        owner_of_group[first_groups] = doc_ids[kept_rows[first_positions]]
        owners = owner_of_group[group]
        matched = (owners >= 0) & (owners != doc_ids)
        matched &= priority[np.maximum(owners, 0)] > priority[doc_ids]
        matched_chars = np.bincount(doc_ids[matched], weights=chars[matched], minlength=document_count)
        contained = live & (lengths > 0) & (matched_chars >= min_share * lengths)
        new_kept = live & ~contained
        if np.array_equal(new_kept, kept):
            break
        kept = new_kept
    matched_rows = np.flatnonzero(matched & contained[doc_ids])
    biggest_first = matched_rows[np.lexsort((-chars[matched_rows], doc_ids[matched_rows]))]
    documents, first_positions = np.unique(doc_ids[biggest_first], return_index=True)
    duplicate_of[documents] = owners[biggest_first[first_positions]]
    return duplicate_of, rounds


class ContainmentDedup(Deduplicator):
    """Drops a document whose paragraphs mostly occur in one or more longer documents (findings "paragraph_level":
    chapter PDFs inside whole books, which document-level Jaccard does not see)."""

    def __init__(self, name: str = "containment_dedup", min_share: float = MIN_CONTAINED_SHARE):
        super().__init__(name)
        self.min_share = min_share

    def open_indexer(self, directory: Path, shard_index: int) -> ContainmentIndexer:
        return ContainmentIndexer(directory, shard_index)

    def build(self, directory: Path, shard_counts: list[int], dropped: np.ndarray | None = None) -> dict:
        offsets = shard_offsets(shard_counts)
        parts = {suffix: [] for suffix in (".ckeys.npy", ".cids.npy", ".cchars.npy", ".clengths.npy")}
        for shard_index, offset in enumerate(offsets):
            prefix = shard_prefix(directory, shard_index)
            for suffix in parts:
                array = np.load(prefix.with_name(prefix.name + suffix))
                parts[suffix].append(array.astype(np.int64) + offset if suffix == ".cids.npy" else array)
        keys, doc_ids, chars, lengths = (np.concatenate(parts[suffix]) for suffix in parts)
        live = np.ones(len(lengths), dtype=bool) if dropped is None else ~np.asarray(dropped)
        duplicate_of, rounds = contained_documents(keys, doc_ids, chars, lengths.astype(np.int64), live,
                                                   self.min_share)
        with DuplicateOfWriter(directory, sum(shard_counts)) as output:
            output[:] = duplicate_of
            duplicates = int(np.count_nonzero(duplicate_of >= 0))
        return {"duplicates": duplicates, "rounds": rounds}


class DocumentTextMap(Step):
    """Text transform that also reads the document's metadata."""

    def __init__(self, name: str, transform):
        super().__init__(name)
        self.transform = transform

    def apply(self, document: dict) -> dict:
        text = self.transform(document)
        if text == document["text"]:
            return document
        return {**document, "text": text}


def build_pipeline() -> Pipeline:
    if not BOILERPLATE_TABLE_PATH.exists():
        raise FileNotFoundError(f"{BOILERPLATE_TABLE_PATH} is missing; run the boilerplate-table command first")
    english_profile = EnglishProfiler()
    return Pipeline("doab", [
        TextMap("normalize", normalize_text),
        TextMap("garbled_paragraphs", drop_garbled_paragraphs),
        DocumentTextMap("page_furniture", strip_page_furniture),
        TextMap("back_matter", strip_back_matter),
        # Symbol-font glyphs inside tags ("<sup>¼</sup>") would become superscript operators if repaired first.
        TextMap("markup", clean_markup),
        TextMap("symbol_font", repair_symbol_font),
        TextMap("ligatures", repair_ligatures),
        TextMap("dehyphenate", dehyphenate),
        DocumentTextMap("rejoin_paragraphs", rejoin_paragraphs),
        # Copies match only after page breaks inside them are rejoined.
        TextMap("repeated_paragraphs", drop_repeated_paragraphs),
        # Removed lines leave blank-line runs; normalize_text collapses them.
        TextMap("renormalize", normalize_text),
        Filter("garbled_font", lambda document: not is_garbled(document["text"])),
        Filter("min_words", lambda document: has_min_words(document["text"])),
        Filter("content", lambda document: content_share(document["text"]) >= MIN_CONTENT_SHARE),
        Filter("english", lambda document: is_english_document(english_profile(document["text"]))),
        ExactDedup("exact_dedup"),
        NearDedup("near_dedup"),
        ContainmentDedup("containment_dedup"),
        # Every IP-address match read in the sample was an EC number, a software version or a section number.
        TextMap("pii", lambda text: mask_pii(text, kinds=("email", "phone"))),
        LicenceCreditStrip("licence_credits"),  # output audit: rejoined caption credits classify as strip
    ])


def table_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Build the doab cross-book boilerplate paragraph table.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=BOILERPLATE_TABLE_PATH)
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args(argv)
    print(f"{build_boilerplate_table(args.input, args.out, args.procs):,} paragraph hashes written to {args.out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["boilerplate-table"]:
        table_command(sys.argv[2:])
    else:
        run_cli(build_pipeline)
