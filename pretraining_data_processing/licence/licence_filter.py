"""Licence-laundering filter for Common Pile v0.1 sources (commercial-use corpus).

classify(doc) -> (action, reasons, spans); action in {"keep", "strip", "drop"}; spans are [start, end) str offsets.
Drop if the doc's own text restricts the whole work or the doc is mostly third-party copyrighted material;
strip only the credit clause that restricts an embedded image/video/figure/table; keep the rest.
"""
import argparse
import collections
import functools
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import sys

PD_CUTOFF_YEAR = 1930  # US: works published in or before 1930 are public domain on 2026-01-01.
CONTEXT_LINES_BEFORE = 2
MAX_DECLARATION_LINE = 600
MAX_STRIP_SPAN = 500
MAX_LICENCE_AFTER_SUBJECT = 300
CC_CONTEXT_WINDOW = 150
MAX_CREDIT_SENTENCE = 150
MAX_GLUED_TAIL = 120
MIN_LEFTOVER_WORDS = 6
MAX_MEDIA_INTRO = 200
MAX_TITLE_CHARS = 200
PG_CLEARED_WINDOW = 300
PG_RESTRICTED_WINDOW = 150
PG_HEAD_CHARS = 15000
PG_RENEWAL_CONTEXT_LINES = 4
WIKISOURCE_TITLE_PAGE_CHARS = 2000
EXAMPLES_PER_REASON = 15

NC_ND = r"(?:Non-?Commercial|NoDeriv(?:ative)?s?|(?:(?<![A-Za-z])|(?<=CC))BY[ -](?:NC|(?:SA[ -])?ND)\b|\bCC[ -]NC\b|creativecommons\.org/licenses/by-n[cd])"
RE_NC_ND = re.compile(NC_ND, re.I)
# Uppercase "BY NC" or hyphenated "by-nc" is a CC token by itself; "by Nd:YAG" is not.
RE_BY_FORM = re.compile(r"(?-i:BY[ -])|by-n[cd]|creativecommons", re.I)
RE_CC_CONTEXT = re.compile(r"creative\s*commons|creativecommons|\bCC(?![A-Za-z])|\bCCBY|\battribution\b", re.I)
RE_ARR = re.compile(r"all rights reserved", re.I)
RE_PERMISSION = re.compile(
    r"\b(?:used|reused|reprinted|reproduced|included|printed|published|adapted)\s+"
    r"(?:(?!(?:or|and|but|when|where|if|unless|while|only|providing|provided|except|without|until|after|before)\b)[\w,'’-]+\s+){0,5}?"
    r"(?:with|by)\s+(?:the\s+)?"
    r"(?:kind\s+|express\s+|written\s+)?(?:copyright\s+)?permission\b", re.I)
# "with permission of Springer" restricts reproduced text only when the sentence names a copyrighted source.
RE_PERMISSION_GRANTED = re.compile(
    r"(?<!\byou\s)\b(?:gave|give|gives|given|granted|grant|grants)\s+(?:us|me|the\s+authors?|our\s+\w+)\s+(?:(?:written|express|kind)\s+)?permission\s+to\s+"
    r"(?:use|reproduce|include|reprint|publish|share|adapt)\b"
    # The granted item must be text ("their words", "it", "materials"); fiction grants leave to use a lodge or a forest.
    r"(?=\s*[,.;)]|\s+(?:[\w'’-]+\s+){0,3}?(?:it|them|this|text|texts|material|materials|work|works|selections?|excerpts?|articles?"
    r"|chapters?|essays?|poems?|stor(?:y|ies)|words|writings|content|passages?|extracts?|letters?)\b)", re.I)
# A permission sentence inside quoted dialogue or a quotation is narrative, not a credit of the page.
RE_QUOTED_LINE_START = re.compile(r"^\W{0,3}[\"“‘']")
RE_PERMISSION_OF = re.compile(r"\bwith\s+(?:the\s+)?(?:kind\s+)?permission\s+of\b", re.I)
RE_PERMISSION_OF_SOURCE = re.compile(r"©|\bmaterial\s+from\b|\bexcerpt", re.I)
RE_NO_REPRODUCE = re.compile(
    r"no part of this (?:book|publication|work|text|document|material)[^\n]{0,80}may be (?:reproduced|copied|stored|transmitted|used)"
    r"|may not be (?:reproduced|copied|redistributed|republished|reprinted)", re.I)
# \s+ instead of spaces: Project Gutenberg wraps these notices over several lines.
RE_RESTRICTED_USE = re.compile(
    r"no\s+permission\s+is\s+granted\s+for\s+commercial|\bpersonal\s+use\s+only\b|\beducational\s+(?:purposes|use)\s+(?:and|or)\s+personal\s+use\b"
    r"|\bnot\s+(?:be\s+)?(?:distributed|sold|used)(?:\s+or\s+(?:distributed|sold|used))?\s+commercially\b"
    r"|\bprohibited\s+commercial\s+distribution\b|\bnot\s+for\s+commercial\s+(?:distribution|use)\b", re.I)
# "It may not be sold" restricts an OER line; in a Project Gutenberg book it is often narrative about a historical edition.
RE_PG_RESTRICTED_USE = RE_RESTRICTED_USE
RE_RESTRICTED_USE = re.compile(RE_PG_RESTRICTED_USE.pattern + r"|\b(?:it|this|these|they)\s+may\s+not\s+be\s+sold\b", re.I)
RE_PG_COPYRIGHTED = re.compile(r"copyrighted\s+Project\s+Gutenberg|Project\s+Gutenberg[^\n]{0,40}\bCOPYRIGHTED\b", re.I)
RE_PG_CLEARED = re.compile(
    r"not\s+renewed|no\s+(?:record\s+of\s+)?(?:a\s+)?(?:copyright\s+)?renewal|evidence\s+that(?:U\.\s?S\.|[^.]){0,100}?\s+renewed"
    r"|no\s+evidence\s+(?:of|that)(?:U\.\s?S\.|[^.]){0,80}?renew|did\s+not\s+uncover\s+any\s+evidence|renewal\s+(?:has|was|could)\s+not\s+(?:been\s+|be\s+)?found|\brule\s+6\b"
    r"|now\s+in\s+the\s+public\s+domain|will\s+be\s+in\s+the\s+public\s+domain|public\s+domain\s+in\s+the\s+(?:USA|U\.\s?S\.|United\s+States)",
    re.I)
PG_LICENCE_TERMS = ("gutenberg-tm", "gutenberg tm", "foundation", "small print", "literary archive")
RE_COPYRIGHT_YEAR = re.compile(r"(?:\bcopyright(?:ed)?\b|©|\(c\))[^\n]{0,60}?\b((?:1[5-9]|20)\d\d)\b", re.I)
RE_YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
RE_ROMAN_YEAR = re.compile(r"\bM\.?\s?(?:D\.?\s?C{0,4}|C\.?\s?M)\.?\s?(?:[LXVI]\.?\s?){0,8}\b")
RE_NO_CLAIM = re.compile(r"no\s+(?:domestic\s+)?copyright\s+(?:is\s+)?claimed", re.I)

TRIGGER = re.compile(
    r"non-?commercial|noderiv|by[ -]n[cd]\b|\bcc[ -]nc\b|by[ -]sa[ -]nd\b|licenses/by-n[cd]|all rights reserved|permission|may not be|no part of this"
    r"|personal\s+use|commercially|commercial\s+(?:use|distribution)|copyrighted", re.I)

RE_MEDIA = re.compile(
    r"\b(?:image|images|photo|photos|photograph|photographs|photography|picture|pictures|video|videos|youtube|youtu\.be|vimeo|ted\.com"
    r"|flickr|flic\.kr|figure|figures|illustration|illustrations|drawing|drawings|painting|paintings|plate(?=\s*(?:\d|[IVX]+\b))|cover"
    r"|map(?=\s*\d|\s+(?:by|from|courtesy|credit|source)\b)|maps|graphic|graphics"
    r"|icon|icons|logo|logos|diagram|diagrams|chart|charts|poster|clip|screenshot|screenshots|screen shots?|animation|gif|audio|podcast"
    r"|music|song|soundtrack|film|footage|artwork|cartoon|table(?=\s*(?:\d|[IVX]+\b))|tables|scheme|schemes|panels?|plots?|infographic|infographics"
    r"|getty|shutterstock|istock|wikimedia commons|courtesy|TED(?:x|-Ed)? talks?)\b|(?-i:\bTED(?:x|-Ed)?\b)|\bfigs?\.(?=\s|\d)|\.(?:jpe?g|png|gif|svg|tiff?)\b", re.I)
# LibreTexts numbers captions with the MathJax macro "Figure \(\PageIndex{3}\)".
CAPTION_NUMBER = r"(?:\d+|[IVX]+\b|\\\(\\PageIndex\{\d+\}\\\))"
RE_CAPTION_LABEL = re.compile(r"(?<![A-Za-z])(?:figure|figs?\.|table|scheme|plate)\s*" + CAPTION_NUMBER, re.I)
RE_TABLE_LABEL = re.compile(r"(?<![A-Za-z])table\s*" + CAPTION_NUMBER, re.I)
RE_FIGURE_PERMISSION = re.compile(
    r"\b(?:reprinted|reproduced|adapted|used)\s+(?:\w+\s+){0,2}?(?:with|by)\s+(?:\w+\s+){0,2}?permission\s*"
    r"(?:(?:from|of)\s+(?:ref(?:erence)?s?\b\.?\s*)?)?(?:[\[(]\d+|(?:from|of)\s+[^.;\n]{0,80}?\bet al\b)"
    r"|[\[(]\d+(?:\s*[,–-]\s*\d+)*[\])]\s*[-–—,:]?\s*(?:reprinted|reproduced|adapted|used)\s+(?:\w+\s+){0,2}?(?:with|by)\s+(?:\w+\s+){0,2}?permission",
    re.I)
# DOAB figure credits that cite a journal ("Reprinted with permission from Califano M. ... Appl. Phys. Lett., 84, 2409-2411").
RE_JOURNAL_PERMISSION = re.compile(r"\b(?:reproduced|reprinted|adapted)\s+(?:\w+\s+){0,2}?with\s+permission\s+from\b", re.I)
RE_CITATION = re.compile(
    r"(?i:\bet al\b|\beditors?\b|\b(?:vol|pp|no)\.\s*\d|\bISSN\b|\bdoi\b|copyright\s*(?:©\s*)?\(?(?:19|20)\d\d)"
    r"|\b\d+\s*,\s*\d+\s*[-–]\s*\d+|\((?:19|20)\d\d[a-z]?\)|\b(?:19|20)\d\d[a-z]?\)"
    r"|\b(?:J|Phys|Chem|Lett|Rev|Proc|Sci|Soc|Am|Appl)\.\s|\b[A-Z][a-z]+,?\s+[A-Z]{1,2}\.")
# Panel labels, "et al." or "Copyright 2010 American Institute of Physics" mark a long PDF caption fragment.
RE_LONG_CAPTION_MARK = re.compile(
    r"\(\s*[a-hA-H](?:\s*[,–-]\s*[a-hA-H])*\s*\)|\*\*\([A-Ha-h]\)\*\*|\bet al\b|(?i:copyright)\s*\(?(?:19|20)\d\d\)?,?\s+[A-Z]"
    r"|(?i:copyright)\s+[A-Z][^.\n]{0,60}(?:19|20)\d\d")
# PDF extraction splits captions from their figures; these marks identify the caption fragments.
RE_CAPTION_FRAGMENT = re.compile(r"^\s*(?:(?!https?:|www\.)[a-z)\]]|\d+[),])|^\s*\*[^*][^\n]*\*\s*$|\([a-hA-H]\)\s|\bscale bar\b")
RE_URL_ONLY = re.compile(r"^\W*(?:https?://|www\.)\S+\s*$", re.I)
# "Adapted from:" heads a text-source block whose licence URLs follow on their own lines.
RE_ADAPTED_HEADER = re.compile(r"(?i)^\W*(?:(?:section|chapter|part|unit|module)\s+[\w.]+\s+)?(?:text\s+|content\s+)?adapted\s+from\b")
MAX_SOURCE_LINES_BEFORE_URL = 2
MAX_URL_BLOCK_LINES = 10
MAX_CAPTION_FRAGMENT = 300
RE_CAPTION_START = re.compile(
    r"^\W*(?:(?:figure|figs?\.|table|scheme|plate)\s*(?:\d|[IVX]+\b|\\\(\\PageIndex)"
    r"|(?:image|photo|photograph|picture|video|illustration|map|chart|diagram|screenshot|infographic)s?\b\s*(?:\d|:|credits?\b|source\b|attribution\b))",
    re.I)
RE_CREDIT_ANCHOR = re.compile(
    r"(?:\(?\b(?:title|cover|header|featured?|banner|background|hero)\s+)?\b(?:image|photo|photograph|picture|video|illustration|map"
    r"|chart|graphic|diagram|screenshot|drawing|painting|artwork|cartoon|logo|icon|clip|audio|podcast|infographic)s?\s*"
    r"(?:\b(?:by|credits?|courtesy|source|attribution|from)\b|:|©)"
    r"|(?<![A-Za-z])(?:figure|figs?\.|table|scheme|plate)\s*" + CAPTION_NUMBER + r"|[\"“][^\"”\n]{1,200}[\"”]\s*(?:by|from|©)"
    r"|\b(?:credits?|source|courtesy)\s*(?::|of\b)|©|\(c\)", re.I)
RE_CREDIT_WORD = re.compile(
    r"\b(?:source|credits?|courtesy|copyright|attribution|licen[cs]ed?|authored by|provided by|located at|flickr|wikimedia|unsplash"
    r"|pixabay|pexels|getty)\b|©|\(c\)|https?://", re.I)
RE_SENTENCE_BREAK = re.compile(r"[.!?][\"”’)\]]*\s+(?=[\"“(\[*_]*[A-Z])|(?<=\d\.\d)(?=[A-Z][a-z])")
RE_WORD_BEFORE_DOT = re.compile(r"([A-Za-z]+)[.!?][\"”’)\]]*$")
ABBREVIATIONS = {"fig", "figs", "no", "nos", "vol", "vols", "pp", "p", "ed", "eds", "inc", "ltd", "co", "corp", "dr", "mr", "mrs",
                 "ms", "prof", "st", "jr", "sr", "vs", "etc", "al", "ca", "cf", "approx", "dept", "univ"}
RE_LICENCE_TAIL = re.compile(
    r"(?:[ -](?:BY|SA|NC|ND))*(?:\s*\d\.\d)?(?:\s*(?:International|Generic|Unported|United States|License|Licence|Deed))*\s*[)\]]?\s*[.;,]?",
    re.I)

RE_WORK_SUBJECT = re.compile(
    r"except where otherwise noted|unless otherwise noted|text (?:source|credit)|content (?:section )?(?:adapted )?from"
    r"|\bthis (?:work|book|text|textbook|resource|oer|chapter|section|course|module|unit|lesson|page|site|publication|material|content"
    r"|article|document|edition|ebook|e-book)\b|\bthe (?:whole|entire) (?:work|book|text|textbook|resource|course|module|site|publication)\b"
    r"|\b(?:licen[sc]ed|shared|available|published|released|distributed|offered)\s+(?:under|as)\b|\bis licensed\b"
    r"|\blicensed\s+(?=\(?(?:CC|Creative))|\bunder (?:a |the )?licen[sc]e\b|\blicen[sc]e\s*:|\badapted (?:from|by)|\bremix(?:ed)?\b|\bderivative\b",
    re.I)
RE_STRONG_SUBJECT = re.compile(
    r"except where otherwise noted|unless otherwise noted|text (?:source|credit)|content (?:section )?(?:adapted )?from"
    r"|this (?:work|book|text|textbook|resource|oer|chapter|section|course|module|unit|lesson|page|publication|edition|ebook|e-book)"
    r"|the (?:whole|entire) (?:work|book|text|textbook|resource|course|module|site|publication)|adapted (?:from|by)", re.I)
# "Title by Author is licensed under ...": words of the title are not licence discussion.
RE_TITLE_BY = re.compile(r"\bby\s*:?\s*[A-Z]")
RE_OTHERWISE_NOTED = re.compile(r"(?:except|unless) (?:where )?otherwise noted", re.I)
RE_LICENSING_VERB = re.compile(r"\blicen[sc]ed\b|\blicen[sc]e\s*:|\bunder\b|\breleased\b|\bavailable\b|\bdistributed\b|\bshared\b|\bpublished\b", re.I)
# Flickr attributions ("Title" by user is licensed under CC BY-NC 2.0) credit photos; Flickr offers only 2.0 licences.
RE_FLICKR_TASL = re.compile(r"[\"“][^\"”\n]{1,200}[\"”]\s*by\s+[^\n]{1,120}?licensed under\s+(?:a\s+)?CC[ -]?BY[^\n]{0,20}2\.0", re.I)
# Lumen "All rights reserved content" heads a list; each listed item is classified on its own line.
RE_ARR_HEADING = re.compile(r"(?i)^\W*all rights reserved content\W*$")
DEED_TITLE = r"Attribution(?:[ \-–](?:Non-?Commercial|ShareAlike|NoDeriv(?:ative)?s))+(?:\s+\d\.\d)?(?:\s+(?:International|Unported|Generic|IGO))?"
CC_NC_ND_CODE = r"CC[ -]?(?:BY[ -])?(?:SA[ -])?N[CD](?:[ -](?:SA|ND))?(?:[ -]\d\.\d)?(?:\s+IGO)?"
RE_BARE_NC_ND_LINE = re.compile(
    rf"(?i)^\W*(?:(?:licen[sc]e|original content|conditions of use|terms of use)(?:\s*\([^)\n]{{0,40}}\))?\s*:?\s*)?(?:creative\s+commons\s+)?"
    rf"(?:{DEED_TITLE}(?:\s*\(\s*{CC_NC_ND_CODE}\s*\))?|{CC_NC_ND_CODE}(?:\s*:?\s*{DEED_TITLE})?)(?:\s+licen[sc]e)?"
    r"\W*(?:attribution\W*)?$")
# "These materials may be adopted for any learning (non-commercial) activities": the work restricts itself without a CC token.
RE_OWN_WORK_START = re.compile(
    r"^\W*(?:this|these|our)\s+(?:open educational resources?\s*(?:\(oers?\))?|oers?|works?|materials?|resources?|content|lessons?|modules?"
    r"|units?|course|book|textbook|text|website|site)\b", re.I)
RE_NC_WORD = re.compile(r"non-?commercial", re.I)
RE_USE_GRANT = re.compile(
    r"\b(?:may|can|could)\s+(?:freely\s+|only\s+)?(?:be\s+)?(?:used|adopted|shared|share|copied|reproduced|distributed|reused|remixed|repurposed?|adapted)\b"
    r"|\blicen[sc]ed\b|\bpurposes?\b|\buse only\b", re.I)
# "excepted from the Creative Commons Attribution-NonCommercial-ShareAlike license" names the licence of the rest of the work.
RE_EXCEPTED_FROM = re.compile(r"\b(?:excepted|excluded|exempt(?:ed)?)\s+from\s+(?:the|this)\b", re.I)
MAX_EXCEPTED_TO_LICENCE = 120
# LibreTexts HTML splits "Image credit: Title by Owner, is licensed under a" over several lines above the licence name.
RE_LICENSED_UNDER_END = re.compile(r"licen[sc]ed\s+under(?:\s+(?:a|the))?\s*$", re.I)
MAX_SPLIT_CREDIT_LINES = 6
# "| Tool | description | link\n(CC-BY-NC-SA license) |" continues a table row that names an external resource.
RE_TABLE_ROW_END = re.compile(r"^[^|]*\|\s*$")
# A header line that lists the source works of the page: "This chapter ... is a derivative of the following texts:", "Derived From".
RE_DERIVATION_HEADER = re.compile(
    r"(?i)^\W*(?:(?:this|the) (?:chapter|section|page|work|book|text|textbook|module|unit|lesson|resource|content)\b[^\n]{0,80}?"
    r"\b(?:(?:is |was )?(?:a |an )?(?:modified |remixed |adapted )?derivative (?:work )?of|(?:is |was )?(?:derived|adapted|remixed|compiled) from)"
    r"|derived from|(?:a |an )?(?:modified )?derivative work (?:of|from)|text attributions?"
    r"|(?:(?:portions|parts|some) of (?:the )?)?(?:content|text) (?:adapted |derived |remixed )?from"
    r"|adapted from|remixed from|sources?|(?:contributors|contributions) and attributions"
    r"|attributions?(?:\s*\([^)\n]{1,30}\)| for [^:\n]{1,40})?)"
    r"\W*(?:the following(?: texts?| works?| sources?)?)?\s*[:.]?\s*$")
# A bare "Attributions" heading also heads photo credits; Flickr offers only 2.0 licences.
RE_BARE_ATTRIBUTIONS = re.compile(r"(?i)^\W*attributions?\W*$")
RE_PHOTO_BY = re.compile(r"\sby\s")
MAX_DERIVATION_LIST_LINES = 30
MAX_DERIVATION_ITEM = 300
RE_BOTH_USES = re.compile(r"(?i)\bcommercial(?:ly)?\s*(?:and|or|&|/)\s*non-?commercial(?:ly)?|\bnon-?commercial(?:ly)?\s*(?:and|or|&|/)\s*commercial(?:ly)?")
# Frontiers eBooks: the compilation is reserved, each article inside is CC BY (stated in the same front matter).
RE_COMPILATION_NOTICE = re.compile(r"(?i)Frontiers Media SA\.? All rights reserved|the book as a whole is distributed by MDPI")
RE_QUOTED = re.compile(r"[\"“][^\"”\n]{0,200}[\"”]")
RE_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:org|com|edu|net|gov|pub)/\S*", re.I)
# Pressbooks media-attribution lists: "- Title © owner is licensed under a CC BY-NC-ND license".
RE_TASL_LIST_ITEM = re.compile(r"^\s*[-•*]\s*[^\n]{1,150}©[^\n]{1,120}?is licensed under", re.I)
RE_DISCUSSION = re.compile(
    r"\b(?:means|meaning|allows?|lets|permits?|prohibits?|restricts?|require[sd]?|forbids?|court|lawsuits?|condition|conditions|clause|choose|choosing|types? of"
    r"|kinds of|for example|e\.g\.|such as|if you|you can|you may|you cannot|cannot|can't|whether|compatib\w*|versus|vs\.|difference"
    r"|options?|elements?|copyrightable|copyright law|fair use|infring\w*|default|without an open|anyone|stands for"
    r"|same as|licen[cs]es|restriction|typically|needs to|needed to|permitted|intriguingly)\b|(?<!United )(?-i:\bstates\b)|\?", re.I)
RE_CC_LICENCE = re.compile(
    r"creative\s*commons|creativecommons\.org|\bCC[ -]?BY\b|\bCC0\b|open access|public domain|GNU Free Documentation|\bGFDL\b", re.I)
RE_STANDARDS_NOTICE = re.compile(r"Common Core State Standards|National Governors Association|Next Generation Science Standards|\bNGSS\b", re.I)
RE_PLACEHOLDER = re.compile(r"\{[^{}\n]{2,80}\}|\[(?:insert|name of|your)[^\]\n]{0,80}\]", re.I)
RE_QUIZ_OPTION = re.compile(r"^\s*(?:[-*•]\s*)?[A-Ea-e][).]\s")
MAX_QUIZ_OPTION = 120
RE_CODE_LINE = re.compile(r"^\s*(?:[A-Za-z]:\\[^\n>]{0,120}>|[$%]\s|>>>\s|\w+@[\w.-]+[:~]|//|/\*|<!--|REM\b)")
RE_SOFTWARE_VERSION = re.compile(r"\bversion\s+\d+(?:\.\d+)+|\[version\s+\d", re.I)
RE_RESOURCE_LIST = re.compile(
    r"resource[ -]?list|resource[ -]?sets?\b|resource[ -]?guide|reading[ -]?list|bibliograph|directory|compendium|oer[ -]?by"
    r"|by[ -]?(?:discipline|subject)|oer[ -]?(?:guide|discipline|finder)|peer[ -]?guide|website[ -]?guidance", re.I)
RE_LIST_TITLE = re.compile(r"(?i)(?:wikis|websites|web sites|links)\s*$")
RE_OWN_TEXT_CUE = re.compile(
    r"\bthis\b|otherwise noted|\badapted\b|\bremix|\bderivative\b|text (?:source|credit)|content (?:section )?(?:adapted )?from", re.I)
RE_WORK_OWN_CUE = re.compile(
    r"\bthis (?:work|book|text|textbook|resource|oer|chapter|section|course|module|unit|lesson|page|site|publication|material|content|article"
    r"|document|edition|ebook|e-book)\b|otherwise noted|\badapted\b|\bremix|\bderivative\b|text (?:source|credit)|content (?:section )?(?:adapted )?from",
    re.I)
RE_REFERENCE_ENTRY = re.compile(r"^\s*(?:\[\d+\]|\d+\.\s+[A-Z][\w'’-]+,\s)")
# Unnumbered author entries ("- Surname, Given.", "Surname, I. (2018).") also start TASL attributions of adapted text, so they count only without an NC token.
RE_AUTHOR_ENTRY = re.compile(
    r"^\s*(?:[-•*]\s*[A-Z][\w'’-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+)(?:\s+[A-Z]\.)*\s*[.(]|[A-Z][\w'’-]+,\s+(?:[A-Z]\.\s*){1,3}[^\n]{0,200}?\((?:19|20)\d\d[a-z]?\))")
RE_ACKNOWLEDGEMENT = re.compile(
    r"^\W*(?:funding|acknowledg(?:e)?ments?|conflicts? of interest|data availability)\b"
    r"|^\s*this (?:study|research|work|project) (?:is|was|has been) (?:\w+\s+){0,2}?(?:supported|funded)", re.I)
# "IEEE Xplore (Copyright 2021 IEEE - All rights reserved ...)" quotes the notice of a database the text names.
RE_THIRD_PARTY_NOTICE = re.compile(r"(?i)\((?:copyright|©)\s*(?:©\s*)?(?:19|20)\d\d[^()]{0,80}?all rights reserved[^()]*\)")
RE_EXTERNAL_ACCESS = re.compile(
    r"\b(?:accessed|access(?:ible)?|available|found|viewed|downloaded)\s+(?:\w+\s+)?(?:through|at|from|on|via)\b|\bfollowing\s+(?:website|link|site|url|web ?page)",
    re.I)
RE_ATTRIBUTION_ITEM = re.compile(
    r"\b(?:authored|provided)\s+by\s*:|\blocated\s+at\s*:|\blicen[cs]e\s*:|\badapt(?:ed|ation)\b|\bremix|\bderiv(?:ed|ative)\b|\bbased on\b"
    r"|\bmodified\b|\b(?:source|attribution|credits?)\s*:|text (?:source|credit)|content (?:section )?(?:adapted )?from|otherwise noted", re.I)
# "CK12. (2015). Food and nutrients. Accessed ... at <url>. Available under CC BY-NC": a source citation, not a link description.
RE_DATED_CITATION = re.compile(r"\((?:19|20)\d\d[a-z]?(?:,[^)]{0,30})?\)\.")
RE_LINK_LIST_INTRO = re.compile(r"(?i)\blinks?\b[^\n]{0,60}\b(?:below|following)\b|\b(?:below|following)\b[^\n]{0,40}\blinks?\b")
RE_RIGHTS_CONTEXT = re.compile(r"permission|copyright|©|licen[cs]|distribut|reproduc|\bcop(?:y|ies)\b|download|reprint|electronic", re.I)
RE_TEXT_UNIT_REFERENCE = re.compile(
    r"\bthis (?:chapter|article|paper|essay|text|book|section|poem|story|content|material|work|piece|excerpt|reading|page|lesson|module|unit)\b"
    r"|originally published|first published|previously published|reprint of|version of this", re.I)
RE_CREDIT_OPENING = re.compile(r"^\W*(?:reprinted|reproduced|adapted|used|printed)\b", re.I)
RE_LICENCE_TEXT_TITLE = re.compile(r"\blicen[cs]e\b|\bEULA\b|terms of (?:use|service)|copyright (?:act|law|notice)", re.I)
# Lumen text-source items: "- 6.4: Theory of Plate Tectonics. Provided by: CK-12. Located at: ... License: CC BY-NC".
RE_ATTRIBUTION_FIELD = re.compile(r"\b(?:authored|provided)\s+by\s*:|\blocated\s+at\s*:|\blicen[cs]e\s*:|\bproject\s*:", re.I)
RE_STRONG_MEDIA = re.compile(
    r"\b(?:images?|photos?|photographs?|pictures?|videos?|youtube|vimeo|flickr|illustrations?|diagrams?|screenshots?|infographics?|icons?|logos?"
    r"|clip ?art|animations?)\b|\.(?:jpe?g|png|gif|svg|tiff?)\b", re.I)
RE_TALK = re.compile(r"(?-i:\bTED(?:x|-Ed)?\b)|\b(?:talks?|speech|lecture)\b", re.I)
MIN_TRANSCRIPT_PROSE_SHARE = 0.6
PROSE_LINE_CHARS = 200
TALK_TITLE_HEAD_CHARS = 300
MIN_QUOTED_TITLE_CHARS = 10
RE_TEXT_OBJECT = re.compile(r"\b(?:stor(?:y|ies)|poems?|poetry|texts?|excerpts?|passages?|readings?|chapters?|articles?|essays?|lyrics|transcripts?)\b", re.I)
RE_MEDIA_BLOCK_HEADING = re.compile(r"(?i)^\W*(?:media|image|photo|figure|video)\s+(?:attributions?|credits?)\W*$")
MAX_BLOCK_LINES = 40
RE_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s")
RE_TABLE_ROW = re.compile(r"^\s*\||\|\s*$")
RE_FIRST_PERSON = re.compile(r"(?i)\b(?:our|my|we|I)\b")
RE_EXAMPLE_LINE = re.compile(r"(?i)^\W*(?:for\s+)?example\s*[:.–-]")
RE_MEDIA_HEADING = re.compile(r"(?i)^\W*(?:videos?|images?|photos?|photographs?|pictures?|illustrations?|diagrams?|maps?|charts?|infographics?)\W*$")
RE_FLICKR_VERSION = re.compile(r"\b2\.0\b")
RE_ENDS_BY_NAME = re.compile(r"\bby\s+[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}\s*[.,]?\s*$")
MAX_TABLE_CHARS = 20000
MAX_TABLE_GAP_LINES = 2
# Pages that teach licences list and compare licence names; their lists, rows, examples and headings are not declarations.
RE_CC_LICENCE_NAME = re.compile(
    r"\bCC[ -]?BY(?:[ -](?:NC|SA|ND))*\b|\bCC0\b|\bAttribution(?:[ \-–](?:Non-?Commercial|ShareAlike|NoDeriv\w*))+", re.I)
MIN_TEACHING_LICENCE_NAMES = 4
# Media-attribution lists also name several licences; a teaching page explains them.
RE_TEACHING_PROSE = re.compile(
    r"\bthis licen[cs]e\b|\blicen[cs]e (?:lets|allows|requires|permits)\b|\b(?:lets|allows) others\b|\btypes of (?:creative commons |cc |open )?licen"
    r"|\blicen[cs]e types\b|\bcreative commons licen[cs]es\b|\bsix (?:cc |creative commons |main )?licen", re.I)
RE_TEACHING_TITLE = re.compile(
    r"\bOER\b|\bopen\s+(?:licens\w*|educational|textbooks?|access|pedagogy|publishing)\b|\b(?:creative\s+commons|CC)\s+licen[cs]\w*|\blicen[cs]e\s+types\b"
    r"|\bcopyright\b|\bcreative\s+commons\b"
    r"|\bfair\s+(?:use|dealing)\b|\bintellectual\s+property\b", re.I)
RE_TEACHING_HEADING = re.compile(
    r"fair (?:use|dealing)|copyright (?:law|basics|exceptions|guidelines)|open licen|creative commons licen|types of licen|licen[cs]e types", re.I)
MAX_HEADING_CHARS = 100
RE_TEACHING_DISCUSSION = re.compile(r"\b(?:should|example|statement|as follows|looks? like|sample)\b", re.I)

RE_LYRIC_MARKER = re.compile(
    r"(?im)^\s*[\[(]?(?:verse(?:\s*\d+)?|chorus|pre-?chorus|bridge|refrain|hook|outro|intro)[\])]?\s*:?\s*$")
# Plays use "Chorus" as a speaker, so a song also needs a verse marker.
RE_VERSE_MARKER = re.compile(r"(?im)^\s*[\[(]?verse(?:\s*\d+)?[\])]?\s*:?\s*$")
RE_WIKIQUOTE_MEDIA_TITLE = re.compile(
    r"\((?:[^()]*\s)?(?:film|TV series|television series|miniseries|video game|musical|band|album|song|season \d+)\)\s*$|/(?:Season|Series) \d",
    re.I)
RE_WIKIQUOTE_MEDIA_HEADING = re.compile(
    r"(?im)^(?:taglines|cast|voice cast|main cast|seasons|season (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"|series (?:\d+|one|two|three|four|five|six)|quotes by season|episodes|episode list|main characters|recurring characters)\s*$")
RE_WIKIQUOTE_DIALOGUE_RULE = re.compile(r'<hr width="50%"')
RE_WIKIQUOTE_CREDITS = re.compile(r"(?im)^\s*(?:\*\s*)?(?:directed|produced|written|created)(?: and (?:written|directed|produced))? by\b")
RE_WIKIQUOTE_SPEAKER = re.compile(r"(?m)^\s*\*\s*[A-Z][\w .'-]{1,30}:\s")
MIN_SPEAKER_LINES = 5
RE_LYRICS_HEADING = re.compile(r"(?im)^\s*(?:song )?lyrics\s*$")
WIKIQUOTE_CREDITS_CHARS = 3000
# Lead sentence of a Wikiquote page about one work: "Choke is a 2001 novel by", "is an action-adventure platform video game".
WIKIQUOTE_LEAD_CHARS = 800
WIKIQUOTE_STOP_WORDS = (r"known|who|which|that|by|for|of|in|from|with|about|on|at|as|actor|actress|director|writer|author|producer|presenter|host"
                        r"|character|company|studio|singer|band|musician|comedian|franchise|person|episode|season")
WIKIQUOTE_SCREEN_KINDS = (r"film|movie|sitcom|mini-?series|web series|anime|soap opera|video game|cartoon series|animated series"
                          r"|(?:television|TV)(?:\s+[\w'’-]+){0,2}?\s+(?:series|programme|program|show|special|film|serial|sitcom|drama)")
WIKIQUOTE_BOOK_KINDS = r"novel|novella|graphic novel|manga"
WIKIQUOTE_PERSON_NOUNS = (r"actor|actress|director|producer|writer|screenwriter|composer|critic|personality|presenter|host|maker|star|editor"
                          r"|executive|designer|developer|author|novelist|historian|scholar|journalist|coach|player|company|studio|series of")
RE_WIKIQUOTE_LEAD_WORK = re.compile(
    rf"\b(?:is|was|are|were)\s+(?:an?|the)\s+(?:(?!(?:{WIKIQUOTE_STOP_WORDS})\b)[\w'’.()-]+[\s,]+){{0,8}}?"
    rf"(?:(?P<screen>{WIKIQUOTE_SCREEN_KINDS})|(?P<book>{WIKIQUOTE_BOOK_KINDS})|(?P<game>game))\b"
    rf"(?![\w\s/,-]{{0,25}}\b(?:{WIKIQUOTE_PERSON_NOUNS})\b)", re.I)
# A person's lead gives a birth date: "(born 4 April 1960)", "(30 June 1939 – 18 March 2016)".
RE_PERSON_LEAD = re.compile(
    r"\((?:born|b\.)\s|\((?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
WIKIQUOTE_LEAD_SENTENCES = 2
RE_WIKIQUOTE_NOVEL_TITLE = re.compile(r"\((?:[^()]*\s)?novel\)\s*$", re.I)
RE_WIKIQUOTE_DIALOGUE_LINE = re.compile(r"(?m)^\s*\*+\s*(?:\[[^\]\n]{0,80}\]\s*)?(?:'{2,3})?[A-Z][\w .'’\"-]{0,30}(?:'{2,3})?:\s?\S")
MIN_GAME_DIALOGUE_LINES = 3
RE_LYRICS_CUE = re.compile(r"(?i)\blyrics of (?:the |this )?song\b|\bsong lyrics\s*:|^\W*lyrics\s*:|\btranscri\w*(?: version)? of the song\b")
SHORT_LINE_CHARS = 60
MIN_LYRIC_BLOCK_LINES = 8
MIN_LYRIC_LINE_WORDS = 3
MIN_LYRIC_SHORT_SHARE = 0.6
RE_RENEWAL = re.compile(r"renew", re.I)
RE_NOT_COPYRIGHT = re.compile(r"\bnot\s+copyright", re.I)

NON_COMMERCIAL_METADATA = re.compile(r"non-?commercial|noderiv|\bnc\b|\bnd\b|by-nc|by-nd|all rights reserved", re.I)
LYRIC_RULE_EXEMPT = ("project gutenberg", "public-domain-review", "wikisource")


class Page:
    def __init__(self, doc):
        meta = doc.get("metadata") or {}
        if isinstance(meta, str):
            meta = {}
        self.text = doc.get("text", "") or ""
        self.source = str(doc.get("source", "")).lower()
        self.is_pg = "gutenberg" in self.source
        self.is_doab = self.source == "doab"
        self.wiki = meta.get("wiki")
        self.is_non_main_wiki = self.wiki is not None and not is_content_namespace(str(meta.get("namespace")))
        self.is_wikisource = self.wiki == "wikisource.com" and not self.is_non_main_wiki
        self.title = str(meta.get("title") or "")
        self.is_licence_text = self.wiki is not None and RE_LICENCE_TEXT_TITLE.search(self.title) is not None
        self.is_resource_list = (RE_RESOURCE_LIST.search(self.title + " " + str(doc.get("id", ""))) is not None
                                 or RE_LIST_TITLE.search(self.title) is not None)
        self.page_titles = {normalise_title(self.title), normalise_title(self.text.split("\n", 1)[0][:MAX_TITLE_CHARS])} - {""}

    @functools.cached_property
    def teaches_licences(self):
        if RE_TEACHING_TITLE.search(self.title):
            return True
        if any(is_heading(line) and RE_TEACHING_HEADING.search(line) for line in self.text.split("\n")):
            return True
        names = {licence_name_key(match.group(0)) for match in RE_CC_LICENCE_NAME.finditer(self.text)}
        return len(names) >= MIN_TEACHING_LICENCE_NAMES and RE_TEACHING_PROSE.search(self.text) is not None

    @functools.cached_property
    def prose_share(self):
        lines = self.text.split("\n")
        return sum(len(line) for line in lines if len(line) >= PROSE_LINE_CHARS) / max(1, len(self.text))


def is_content_namespace(namespace):
    """Main namespace or an even namespace number from 100 up (Cookbook, Page, Wikijunior); odd numbers are talk pages."""
    if not namespace.isdigit():
        return False
    number = int(namespace)
    return number == 0 or (number >= 100 and number % 2 == 0)


def normalise_title(title):
    return re.sub(r"\W+", " ", title).strip().lower()


def names_this_page(line, page):
    return any(normalise_title(quoted.group(0)) in page.page_titles for quoted in RE_QUOTED.finditer(line))


def licence_name_key(name):
    upper = name.upper()
    if upper == "CC0":
        return upper
    parts = [part for part, pattern in (("NC", r"\bNC\b|NON-?COMMERCIAL"), ("ND", r"\bND\b|NODERIV"), ("SA", r"\bSA\b|SHAREALIKE"))
             if re.search(pattern, upper)]
    return "-".join(["BY"] + parts)


def is_heading(line):
    stripped = line.strip()
    return 0 < len(stripped) <= MAX_HEADING_CHARS and not stripped.endswith((".", ";", ","))


def metadata_licence(doc):
    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        return None
    return meta.get("license")


def metadata_licence_ok(licence):
    if not licence:
        return False
    return NON_COMMERCIAL_METADATA.search(licence) is None


def line_bounds(text, pos):
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return start, end


def context_before(text, line_start, lines):
    start = line_start
    for _ in range(lines):
        if start <= 0:
            break
        start = text.rfind("\n", 0, start - 1) + 1
    return text[start:line_start]


def previous_line(before):
    lines = [line for line in before.split("\n") if line.strip()]
    return lines[-1].rstrip() if lines else None


def lines_above(text, line_start):
    """Yield (start, line) for each line above line_start, nearest first."""
    end = line_start - 1
    while end >= 0:
        start = text.rfind("\n", 0, end) + 1
        yield start, text[start:end]
        end = start - 1


def previous_line_is_caption(before):
    previous = previous_line(before)
    if previous is None:
        return False
    if RE_CAPTION_START.match(previous):
        return True
    return len(previous) <= MAX_MEDIA_INTRO and previous.endswith(":") and RE_MEDIA.search(previous) is not None


def list_intro(text, line_start):
    """Return the nearest non-blank line above the list that holds line_start, or None when the list is longer than MAX_BLOCK_LINES."""
    for count, (_, line) in enumerate(lines_above(text, line_start)):
        if count >= MAX_BLOCK_LINES:
            return None
        if line.strip() and not RE_BULLET.match(line):
            return line
    return None


def in_media_attribution_block(text, line_start, line):
    # Pressbooks footnotes ("... ↵") follow the media list without a heading of their own.
    if line.rstrip().endswith("↵"):
        return False
    intro = list_intro(text, line_start)
    return intro is not None and RE_MEDIA_BLOCK_HEADING.match(intro) is not None


def in_link_list(text, line_start, line):
    if not RE_BULLET.match(line):
        return False
    intro = list_intro(text, line_start)
    return intro is not None and RE_LINK_LIST_INTRO.search(intro) is not None


def prose(line):
    return RE_URL.sub(" ", RE_QUOTED.sub(" ", line))


def nc_nd_tokens(line):
    tokens = []
    for match in RE_NC_ND.finditer(line):
        if RE_BY_FORM.match(match.group(0)):
            tokens.append(match)
            continue
        window = line[max(0, match.start() - CC_CONTEXT_WINDOW):match.end() + CC_CONTEXT_WINDOW]
        if RE_CC_CONTEXT.search(window):
            tokens.append(match)
    return tokens


def nc_nd_declaration(line, tokens):
    """Return ("strong" | "weak" | None, subject start) for a work-level NC/ND declaration in one line."""
    for subject in RE_WORK_SUBJECT.finditer(line):
        licence = next((token for token in tokens if token.start() >= subject.start()), None)
        if licence is None or licence.end() - subject.start() > MAX_LICENCE_AFTER_SUBJECT:
            continue
        segment = line[subject.start():licence.end()]
        if RE_DISCUSSION.search(prose(segment)) or RE_MEDIA.search(segment):
            continue
        if (RE_STRONG_SUBJECT.match(subject.group(0)) or RE_OTHERWISE_NOTED.search(line)) and RE_LICENSING_VERB.search(segment):
            return "strong", subject.start()
        return "weak", subject.start()
    return None, 0


def sentence_start(line, position):
    return next((begin for begin, end in sentence_spans(line) if end > position), 0)


def media_before_subject(line, subject_start):
    """"The original video on this page © X is licensed ...": the media word names what the subject credits."""
    return RE_MEDIA.search(line, sentence_start(line, subject_start), subject_start) is not None


def media_scope(line):
    """The part of the line that can show media: the title of a Lumen text-source item names the source, not media."""
    fields = list(RE_ATTRIBUTION_FIELD.finditer(line))
    if len(fields) < 2:
        return line
    title = line[:fields[0].start()]
    if RE_CAPTION_START.match(title) or RE_STRONG_MEDIA.search(title):
        return line
    return line[fields[0].start():]


def caption_fragment(line, doab):
    if RE_URL_ONLY.match(line) or not RE_CAPTION_FRAGMENT.search(line):
        return False
    limit = MAX_DECLARATION_LINE if doab and RE_LONG_CAPTION_MARK.search(line) else MAX_CAPTION_FRAGMENT
    return len(line) <= limit


def journal_permission_credit(line):
    match = RE_JOURNAL_PERMISSION.search(line)
    return (match is not None and len(line) <= MAX_DECLARATION_LINE and RE_CITATION.search(line, match.end()) is not None
            and RE_TEXT_UNIT_REFERENCE.search(line) is None)


def own_media(line, doab=False):
    scope = media_scope(line)
    if (RE_MEDIA.search(scope) or RE_CAPTION_LABEL.search(scope) or RE_FIGURE_PERMISSION.search(line) or caption_fragment(line, doab)
            or RE_FLICKR_TASL.search(line) or RE_TASL_LIST_ITEM.match(line)):
        return True
    # In DOAB monographs a permission credit that cites a journal reproduces a figure, not text.
    return doab and journal_permission_credit(line)


def permission_covers_text(line):
    """True when the sentence of the permission phrase names text (stories, poems, excerpts) as the reproduced item."""
    match = RE_PERMISSION.search(line)
    if match is None or RE_CAPTION_START.match(line):
        return False
    return RE_TEXT_OBJECT.search(line, sentence_start(line, match.start()), match.start()) is not None


def talk_text_source(line, page):
    """A talk credit that closes a full transcript credits text, not a video."""
    if RE_TALK.search(line) is None or page.prose_share < MIN_TRANSCRIPT_PROSE_SHARE:
        return False
    if RE_TEXT_UNIT_REFERENCE.search(line) or "transcript" in line.lower():
        return True
    head = normalise_title(page.text[:TALK_TITLE_HEAD_CHARS])
    titles = (normalise_title(quoted.group(0)) for quoted in RE_QUOTED.finditer(line))
    return any(len(title) >= MIN_QUOTED_TITLE_CHARS and title in head for title in titles)


def bare_licence_after_media_credit(line, before):
    """"Diagram by Nikki Harris" on one line and "CC BY-NC" on the next are one credit; so are "Video", a description and the licence."""
    if RE_BARE_NC_ND_LINE.match(line) is None:
        return False
    above = [previous for previous in before.split("\n") if previous.strip()]
    if any(RE_CAPTION_START.match(previous) or RE_MEDIA_HEADING.match(previous) for previous in above):
        return True
    # Flickr offers only 2.0 licences, so "Title | by user" above a 2.0 licence line is a photo credit.
    return bool(above) and RE_ENDS_BY_NAME.search(above[-1]) is not None and (own_media(above[-1]) or RE_FLICKR_VERSION.search(line) is not None)


def split_media_credit(line, wide_before):
    """True when a bare licence line ends a credit sentence that starts with a media word some lines above."""
    if RE_BARE_NC_ND_LINE.match(line) is None:
        return False
    above = [previous.strip() for previous in wide_before.split("\n") if previous.strip()]
    if not above or RE_LICENSED_UNDER_END.search(above[-1]) is None:
        return False
    parts = [above.pop()]
    while above and not above[-1].endswith((".", "!", "?")):
        parts.insert(0, above.pop())
    credit = " ".join(parts)
    return RE_CAPTION_START.match(credit) is not None or (own_media(credit) and RE_STRONG_SUBJECT.search(credit) is None)


def teaching_line(line, arr_only):
    """A list item, table row, example, bare licence name or ownerless "all rights reserved" line on a licence-teaching page.

    Attribution items and notices with an owner or a year stay declarations.
    """
    if RE_EXAMPLE_LINE.match(line):
        return True
    if RE_STRONG_SUBJECT.search(line) or RE_ATTRIBUTION_ITEM.search(line) or "©" in line or RE_COPYRIGHT_YEAR.search(line):
        return False
    if RE_TITLE_BY.search(line) and re.search(r"(?i)\blicen[sc]ed\b", line):
        return False
    # "our overall collection is licensed CC BY-NC" declares the author's own licence.
    if not arr_only and RE_FIRST_PERSON.search(line):
        return False
    return arr_only or RE_BULLET.match(line) is not None or RE_TABLE_ROW.search(line) is not None or RE_BARE_NC_ND_LINE.match(line) is not None


def follows_adapted_from(text, line_start):
    """True when an "Adapted from:" text-source line sits above this licence URL, with only URLs or up to 2 citation lines between."""
    citation_lines = 0
    for count, (_, line) in enumerate(lines_above(text, line_start)):
        if count >= MAX_URL_BLOCK_LINES:
            return False
        if not line.strip() or RE_URL_ONLY.match(line):
            continue
        if RE_ADAPTED_HEADER.match(line):
            return True
        citation_lines += 1
        if own_media(line) or citation_lines > MAX_SOURCE_LINES_BEFORE_URL:
            return False
    return False


def is_reference_entry(line, tokens):
    return RE_REFERENCE_ENTRY.match(line) is not None or (not tokens and RE_AUTHOR_ENTRY.match(line) is not None)


def continues_reference(line, before):
    previous = previous_line(before)
    return previous is not None and is_reference_entry(previous, []) and RE_CITATION.search(line) is not None


def external_resource_line(line, matches):
    """"These tips can be accessed through the following website (used with permission): https://...": the restriction is the website's."""
    if (RE_URL.search(line) is None or RE_EXTERNAL_ACCESS.search(line) is None or RE_ATTRIBUTION_ITEM.search(line) or RE_DATED_CITATION.search(line)
            or RE_TEXT_UNIT_REFERENCE.search(line)):
        return False
    begin, end = next((begin, end) for begin, end in sentence_spans(line) if end > matches[0].start())
    return RE_WORK_OWN_CUE.search(line[begin:end]) is None


def roman_to_int(numeral):
    values = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    digits = [values[char] for char in numeral if char in values]
    total = 0
    for index, value in enumerate(digits):
        if index + 1 < len(digits) and digits[index + 1] > value:
            total -= value
        else:
            total += value
    return total


def latest_year(text):
    years = [int(match.group(1)) for match in RE_YEAR.finditer(text)]
    years += [roman_to_int(match.group(0)) for match in RE_ROMAN_YEAR.finditer(text)]
    years = [year for year in years if 1500 <= year <= 2029]
    return max(years) if years else None


def wikisource_source_year(text, line_start, line_end):
    year = latest_year(context_before(text, line_start, CONTEXT_LINES_BEFORE) + text[line_start:line_end])
    if year is not None:
        return year
    return latest_year(text[:WIKISOURCE_TITLE_PAGE_CHARS])


def is_credit_line(line):
    if RE_TEXT_UNIT_REFERENCE.search(line):
        return False
    return len(line) <= MAX_CREDIT_SENTENCE or (len(line) <= MAX_STRIP_SPAN and RE_CREDIT_OPENING.match(line) is not None)


def restriction_is_declared(line):
    restricted = RE_RESTRICTED_USE.search(line)
    begin, end = next((begin, end) for begin, end in sentence_spans(line) if end > restricted.start())
    return RE_DISCUSSION.search(prose(line[begin:end])) is None


def in_derivation_list(text, line_start):
    """True when the line is an item of a list that starts at a derivation header; every item between them is short."""
    item = text[line_start:text.find("\n", line_start) if "\n" in text[line_start:] else len(text)]
    above = [line for line in context_before(text, line_start, MAX_DERIVATION_LIST_LINES).split("\n") if line.strip()]
    for line in reversed(above):
        if RE_DERIVATION_HEADER.match(line):
            return not (RE_BARE_ATTRIBUTIONS.match(line) and RE_FLICKR_VERSION.search(item) and RE_PHOTO_BY.search(item))
        if len(line) > MAX_DERIVATION_ITEM:
            return False
    return False


def own_nc_grant(line):
    """True when a sentence opens with the work itself ("These materials ...") and limits its use to non-commercial use."""
    masked = RE_BOTH_USES.sub(lambda match: " " * len(match.group(0)), line)
    for begin, end in sentence_spans(masked):
        sentence = masked[begin:end]
        nc_word = RE_NC_WORD.search(sentence)
        if (nc_word and RE_OWN_WORK_START.match(sentence) and RE_USE_GRANT.search(sentence) and "?" not in sentence
                and not RE_MEDIA.search(sentence[:nc_word.start()])):
            return True
    return False


def excepted_from_licence(line, tokens):
    return any(token.start() >= excepted.end() and token.end() - excepted.end() <= MAX_EXCEPTED_TO_LICENCE
               for excepted in RE_EXCEPTED_FROM.finditer(line) for token in tokens)


def has_permission_phrase(line):
    granted = RE_PERMISSION_GRANTED.search(line) is not None and RE_QUOTED_LINE_START.match(line) is None
    return RE_PERMISSION.search(line) is not None or granted or (RE_PERMISSION_OF.search(line) is not None and RE_PERMISSION_OF_SOURCE.search(line) is not None)


def restriction_matches(line, tokens):
    found = list(tokens)
    for pattern in (RE_ARR, RE_PERMISSION, RE_PERMISSION_GRANTED, RE_PERMISSION_OF, RE_NO_REPRODUCE, RE_RESTRICTED_USE):
        found.extend(pattern.finditer(line))
    return sorted(found, key=lambda match: match.start())


def classify_line(text, line_start, line_end, page):
    """Return (category, reason) for one triggered line; category in {"a", "b", "c", None}."""
    line = text[line_start:line_end]
    tokens = nc_nd_tokens(RE_BOTH_USES.sub(lambda match: " " * len(match.group(0)), line))
    has_arr = RE_ARR.search(line) is not None
    has_perm = has_permission_phrase(line)
    has_noreprod = RE_NO_REPRODUCE.search(line) is not None
    has_restricted = len(line) <= MAX_DECLARATION_LINE and RE_RESTRICTED_USE.search(line) is not None and RE_RIGHTS_CONTEXT.search(line) is not None
    has_reservation = has_arr or has_perm or has_noreprod or has_restricted
    own_nc = not tokens and own_nc_grant(line)
    if not (tokens or has_reservation or own_nc):
        return None, None
    if page.is_pg and not tokens:
        return "c", "pg_printed_source_notice"

    before = context_before(text, line_start, CONTEXT_LINES_BEFORE)
    if "project gutenberg" in (before + line).lower():
        return "c", "project_gutenberg_boilerplate"
    if RE_COMPILATION_NOTICE.search(line):
        return "c", "publisher_compilation_notice"
    if RE_ARR_HEADING.match(line):
        return "c", "attribution_block_heading"
    if RE_PLACEHOLDER.search(line):
        return "c", "template_placeholder"
    if RE_QUIZ_OPTION.match(line) and len(line) <= MAX_QUIZ_OPTION:
        return "c", "quiz_option"
    if not tokens and RE_STANDARDS_NOTICE.search(line):
        return "c", "quoted_standards_notice"
    if not tokens and (RE_CODE_LINE.match(line) or RE_SOFTWARE_VERSION.search(line)):
        return "c", "software_banner_or_code"
    if page.is_resource_list and not RE_OWN_TEXT_CUE.search(line) and not names_this_page(line, page):
        return "c", "catalogue_entry_licence"
    if not has_perm and (is_reference_entry(line, tokens) or (not tokens and continues_reference(line, before))) and not RE_WORK_OWN_CUE.search(line):
        return "c", "reference_list_entry"
    if RE_ACKNOWLEDGEMENT.search(line):
        return "c", "acknowledgement_or_funding_line"
    if not tokens and not has_perm and RE_THIRD_PARTY_NOTICE.search(line):
        return "c", "catalogue_entry_licence"
    if external_resource_line(line, restriction_matches(line, tokens)) or (in_link_list(text, line_start, line) and not RE_WORK_OWN_CUE.search(line)):
        return "c", "catalogue_entry_licence"
    if RE_EXAMPLE_LINE.match(line) or (page.teaches_licences and teaching_line(line, has_arr and not tokens and not has_perm)):
        return "c", "licence_discussion"
    if tokens and RE_URL_ONLY.match(line) and follows_adapted_from(text, line_start):
        return "a", "work_level_nc_nd_declaration"

    # A caption or a media word before the subject makes "this page/work" the media item.
    caption = RE_CAPTION_START.match(line) is not None or (page.is_doab and RE_CAPTION_LABEL.search(line) is not None)
    declaration, subject_start = nc_nd_declaration(line, tokens) if tokens else (None, 0)
    if declaration == "strong" and not caption and not media_before_subject(line, subject_start):
        return "a", "work_level_nc_nd_declaration"
    if not caption and ((tokens and excepted_from_licence(line, tokens))
                        or (own_nc and not page.teaches_licences and not page.is_non_main_wiki)):
        return "a", "work_level_nc_nd_declaration"
    media_context = declaration == "strong" or caption or in_media_attribution_block(text, line_start, line)
    if has_restricted and not media_context and not own_media(line, page.is_doab) and restriction_is_declared(line):
        return "a", "restricted_use_notice"
    discussed = line[subject_start:] if RE_TITLE_BY.search(line[:subject_start]) else line
    if not RE_CAPTION_LABEL.search(line) and not RE_BARE_NC_ND_LINE.match(line) and RE_DISCUSSION.search(prose(discussed)):
        return "c", "licence_discussion"
    if page.teaches_licences and not RE_CAPTION_LABEL.search(line) and RE_TEACHING_DISCUSSION.search(prose(discussed)):
        return "c", "licence_discussion"
    if len(line) > MAX_DECLARATION_LINE:
        return "c", "long_line_without_strong_declaration"
    talk_source = talk_text_source(line, page)
    # DOAB books are scholarly monographs, where a permission line on its own credits a figure or a table.
    media = not talk_source and (own_media(line, page.is_doab) or media_context or (page.is_doab and has_perm and is_credit_line(line))
                                 or (tokens and bare_licence_after_media_credit(line, before))
                                 or (tokens and split_media_credit(line, context_before(text, line_start, MAX_SPLIT_CREDIT_LINES))))
    if media and has_perm and permission_covers_text(line):
        media = False
    after_caption = not media and not talk_source and len(line) <= MAX_STRIP_SPAN and previous_line_is_caption(before)
    if media or after_caption:
        if page.is_non_main_wiki:
            return "c", "wiki_non_main_namespace"
        return "b", "media_credit_restricted"
    if declaration == "weak":
        return "a", "work_level_nc_nd_declaration"
    if tokens and RE_BARE_NC_ND_LINE.match(line) and not RE_TABLE_ROW_END.match(line):
        if page.is_non_main_wiki:
            return "c", "wiki_non_main_namespace"
        return "a", "bare_nc_nd_licence_line"
    if tokens and not has_reservation and len(line) <= MAX_DERIVATION_ITEM and not page.is_non_main_wiki and in_derivation_list(text, line_start):
        return "a", "nc_nd_source_of_derived_text"
    if not has_reservation:
        return "c", "nc_nd_mention_without_work_subject"

    if page.is_non_main_wiki:
        return "c", "wiki_non_main_namespace"
    if page.is_licence_text:
        return "c", "licence_text_page"
    if RE_NO_CLAIM.search(line):
        return "c", "no_copyright_claimed_notice"
    year_match = RE_COPYRIGHT_YEAR.search(before + line)
    if year_match and int(year_match.group(1)) <= PD_CUTOFF_YEAR:
        return "c", "expired_copyright_notice"
    if page.is_wikisource:
        source_year = wikisource_source_year(text, line_start, line_end)
        if source_year is not None and source_year <= PD_CUTOFF_YEAR:
            return "c", "wikisource_source_year_expired"
    # An NC/ND token next to the reservation means the open licence on the line is not a commercial grant.
    if RE_CC_LICENCE.search(line) and not tokens:
        return "c", "reservation_next_to_open_licence"
    if has_restricted:
        return "a", "restricted_use_notice"
    if has_perm:
        return "a", "third_party_text_used_with_permission"
    if has_noreprod:
        return "a", "may_not_be_reproduced"
    return "a", "all_rights_reserved"


def sentence_spans(line):
    spans = []
    start = 0
    for match in RE_SENTENCE_BREAK.finditer(line):
        word = RE_WORD_BEFORE_DOT.search(line[start:match.end()].rstrip())
        if word and (word.group(1).lower() in ABBREVIATIONS or (len(word.group(1)) == 1 and word.group(1).isupper())):
            continue
        end = match.start() + len(match.group(0).rstrip())
        spans.append((start, end))
        start = match.end()
    spans.append((start, len(line)))
    return [(begin, end) for begin, end in spans if end > begin]


def is_credit_sentence(sentence):
    if len(sentence) > MAX_CREDIT_SENTENCE:
        return False
    if RE_MEDIA.search(sentence) or RE_CREDIT_WORD.search(sentence):
        return True
    words = sentence.split()
    capitalised = sum(1 for word in words if word[:1].isupper())
    return RE_TITLE_BY.search(sentence) is not None and capitalised >= len(words) / 2


def credit_span(line, matches):
    """Return the [start, end) part of the line that holds the restricted credit."""
    sentences = sentence_spans(line)
    first = next(index for index, (_, end) in enumerate(sentences) if end > matches[0].start())
    last = next(index for index, (_, end) in enumerate(sentences) if end > matches[-1].start())
    anchor = RE_CREDIT_ANCHOR.search(line, sentences[first][0], matches[0].start())
    start = anchor.start() if anchor else sentences[first][0]
    while first > 0 and start == sentences[first][0] and is_credit_sentence(line[slice(*sentences[first - 1])]):
        first -= 1
        start = sentences[first][0]
    tail = RE_LICENCE_TAIL.match(line, matches[-1].end())
    end = sentences[last][1]
    if end - tail.end() > MAX_GLUED_TAIL:
        end = tail.end()
    else:
        while last + 1 < len(sentences) and is_credit_sentence(line[slice(*sentences[last + 1])]):
            last += 1
            end = sentences[last][1]
    return start, end


def strip_span(text, line_start, line_end):
    """Return the [start, end) text span to remove for a category-b line, or None when it exceeds MAX_STRIP_SPAN."""
    line = text[line_start:line_end]
    tokens = nc_nd_tokens(RE_BOTH_USES.sub(lambda match: " " * len(match.group(0)), line))
    matches = restriction_matches(line, tokens)
    start, end = credit_span(line, matches) if matches else (0, len(line))
    leftover = (line[:start] + " " + line[end:]).strip()
    if len(leftover) <= MAX_CREDIT_SENTENCE or len(re.findall(r"[A-Za-z0-9]{2,}", leftover)) < MIN_LEFTOVER_WORDS:
        start, end = 0, len(line)
    if end - start > MAX_STRIP_SPAN:
        return None
    if start == 0 and end == len(line):
        return [line_start, line_end + 1 if line_end < len(text) else line_end]
    return [line_start + start, line_start + end]


def table_rows_spans(text, line_start, line_end):
    """Return the [start, end) spans of the '|' table rows directly above and directly below a table credit line."""
    spans = []
    top = rows_edge(text, line_start, above=True)
    if top is not None:
        spans.append([top, line_start])
    bottom = rows_edge(text, line_end, above=False)
    if bottom is not None:
        spans.append([line_end + 1, min(len(text), bottom + 1)])
    return spans


def rows_edge(text, position, above):
    """Return the far edge of the '|' rows next to position; up to MAX_TABLE_GAP_LINES spilled cell lines may sit between rows."""
    edge = None
    gap = 0
    for line_start, line in (lines_above(text, position) if above else lines_below(text, position)):
        if abs(line_start - position) > MAX_TABLE_CHARS:
            break
        if line.strip() and RE_TABLE_ROW.search(line):
            edge = line_start if above else line_start + len(line)
            gap = 0
            continue
        if edge is None and line.strip():
            break
        gap += 1
        if gap > MAX_TABLE_GAP_LINES:
            break
    return edge


def lines_below(text, line_end):
    """Yield (start, line) for each line below the line that ends at line_end, nearest first."""
    start = line_end + 1
    while start <= len(text):
        end = text.find("\\n", start)
        if end == -1:
            end = len(text)
        yield start, text[start:end]
        start = end + 1


def pg_modern_copyright(text):
    head = text[:PG_HEAD_CHARS]
    for match in RE_COPYRIGHT_YEAR.finditer(head):
        if int(match.group(1)) <= PD_CUTOFF_YEAR:
            continue
        line_start, line_end = line_bounds(head, match.start())
        line = head[line_start:line_end]
        if "gutenberg" in line.lower() or RE_DISCUSSION.search(line) or RE_NOT_COPYRIGHT.search(line):
            continue
        if renews_old_copyright(head, line_start, line, match):
            continue
        return line.strip()
    return None


def renews_old_copyright(head, line_start, line, match):
    """"Copyright, 1904 ... Copyright renewed, 1932": a renewal of a work published in or before PD_CUTOFF_YEAR."""
    if RE_RENEWAL.search(match.group(0)) is None:
        return False
    nearby = list(itertools.islice((above for _, above in lines_above(head, line_start) if above.strip()), PG_RENEWAL_CONTEXT_LINES))
    return any(int(year.group(1)) <= PD_CUTOFF_YEAR for part in nearby + [line] for year in RE_COPYRIGHT_YEAR.finditer(part))


def pg_cleared(text):
    """True when the book itself, not the Project Gutenberg licence boilerplate, says it is not renewed or public domain."""
    for match in RE_PG_CLEARED.finditer(text):
        window = text[max(0, match.start() - PG_CLEARED_WINDOW):match.end() + PG_CLEARED_WINDOW].lower()
        if not any(term in window for term in PG_LICENCE_TERMS):
            return True
    return False


def pg_drop_reasons(text):
    reasons = []
    header = RE_PG_COPYRIGHTED.search(text)
    if header:
        reasons.append({"reason": "pg_copyrighted_ebook_header", "detail": text[header.start():header.end() + 100]})
    for restricted in RE_PG_RESTRICTED_USE.finditer(text):
        window = text[max(0, restricted.start() - PG_RESTRICTED_WINDOW):restricted.end() + PG_RESTRICTED_WINDOW]
        if RE_RIGHTS_CONTEXT.search(window):
            reasons.append({"reason": "pg_restricted_ebook_licence", "detail": window})
            break
    modern = pg_modern_copyright(text)
    if modern and not pg_cleared(text):
        reasons.append({"reason": "pg_post_1930_copyright_notice", "detail": modern[:300]})
    return reasons


def lyrics_reason(text):
    """Return the drop detail for a doc that is mostly song lyrics, or None."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 10:
        return None
    short_share = sum(1 for line in lines if len(line) < SHORT_LINE_CHARS) / len(lines)
    markers = len(RE_LYRIC_MARKER.findall(text))
    if markers >= 3 and short_share > 0.5 and RE_VERSE_MARKER.search(text):
        return f"markers={markers} short_line_share={short_share:.2f}"
    if short_share <= MIN_LYRIC_SHORT_SHARE:
        return None
    # Without Verse/Chorus markers: a "lyrics of the song" cue line followed by a block of short lines.
    block_lines = 0
    for index, line in enumerate(lines):
        if not RE_LYRICS_CUE.search(line):
            continue
        for following in lines[index + 1:]:
            if not is_lyric_line(following) or RE_LYRICS_CUE.search(following):
                break
            block_lines += 1
    if block_lines >= MIN_LYRIC_BLOCK_LINES:
        return f"lyrics_cue_block_lines={block_lines} short_line_share={short_share:.2f}"
    return None


def is_lyric_line(line):
    """A short line of words, not a quiz item, list item or markup."""
    return len(line) < SHORT_LINE_CHARS and line[:1].isalpha() and len(line.split()) >= MIN_LYRIC_LINE_WORDS


def wikiquote_reason(page):
    if page.wiki != "wikiquote.com" or page.is_non_main_wiki:
        return None
    if RE_WIKIQUOTE_MEDIA_TITLE.search(page.title):
        return "wikiquote_film_tv_music_title"
    text = page.text
    if RE_WIKIQUOTE_MEDIA_HEADING.search(text) or RE_WIKIQUOTE_DIALOGUE_RULE.search(text):
        return "wikiquote_film_tv_quote_collection"
    if RE_WIKIQUOTE_CREDITS.search(text[:WIKIQUOTE_CREDITS_CHARS]) and len(RE_WIKIQUOTE_SPEAKER.findall(text)) >= MIN_SPEAKER_LINES:
        return "wikiquote_film_tv_quote_collection"
    if RE_LYRICS_HEADING.search(text):
        return "wikiquote_song_lyrics_section"
    if wikiquote_modern_work(page):
        return "wikiquote_modern_work_lead"
    return None


def wikiquote_lead(text):
    """First 2 sentences of the first paragraph after the title line that is not a navigation list."""
    for paragraph in text.split("\n", 1)[-1][:WIKIQUOTE_LEAD_CHARS].split("\n"):
        stripped = paragraph.strip()
        if stripped and not stripped.startswith(("*", "|", "#", "{")):
            spans = sentence_spans(stripped)[:WIKIQUOTE_LEAD_SENTENCES]
            return stripped[:spans[-1][1]]
    return ""


def wikiquote_modern_work(page):
    """The lead says the page is about one film, TV series, video game or novel published after PD_CUTOFF_YEAR."""
    lead = wikiquote_lead(page.text)
    if RE_PERSON_LEAD.search(lead):
        return False
    match = RE_WIKIQUOTE_LEAD_WORK.search(lead)
    if match is None and not RE_WIKIQUOTE_NOVEL_TITLE.search(page.title):
        return False
    sentence_end = lead.find(". ", match.end()) if match else lead.find(". ")
    years = [int(year) for year in RE_YEAR.findall(lead[:sentence_end if sentence_end != -1 else len(lead)])]
    year = years[0] if years else None
    if match is not None and match.group("screen"):
        return year is None or year > PD_CUTOFF_YEAR
    if match is not None and match.group("game"):
        return (year is None or year > PD_CUTOFF_YEAR) and len(RE_WIKIQUOTE_DIALOGUE_LINE.findall(page.text)) >= MIN_GAME_DIALOGUE_LINES
    return year is not None and year > PD_CUTOFF_YEAR


def classify(doc):
    page = Page(doc)
    text = page.text
    reasons = []
    spans = []
    drop_reasons = []

    licence = metadata_licence(doc)
    if not metadata_licence_ok(licence):
        drop_reasons.append({"reason": "metadata_licence_not_commercial", "detail": str(licence)})

    wikiquote = wikiquote_reason(page)
    if wikiquote:
        drop_reasons.append({"reason": wikiquote, "detail": page.title})

    if not any(name in page.source for name in LYRIC_RULE_EXEMPT):
        lyrics = lyrics_reason(text)
        if lyrics:
            drop_reasons.append({"reason": "predominantly_song_lyrics", "detail": lyrics})

    if page.is_pg:
        drop_reasons.extend(pg_drop_reasons(text))

    seen_lines = set()
    for match in TRIGGER.finditer(text):
        line_start, line_end = line_bounds(text, match.start())
        if line_start in seen_lines:
            continue
        seen_lines.add(line_start)
        category, reason = classify_line(text, line_start, line_end, page)
        if category is None:
            continue
        line = text[line_start:line_end]
        item = {"reason": reason, "detail": line.strip()[:300], "offset": line_start}
        if category == "a":
            drop_reasons.append(item)
        elif category == "b":
            span = strip_span(text, line_start, line_end)
            if span is None:
                item["reason"] = "media_credit_too_long_to_strip"
            else:
                spans.append(span)
            reasons.append(item)
            # Rows of a restricted table are the restricted content, so they go with the credit.
            for rows in (table_rows_spans(text, line_start, line_end) if span is not None and RE_TABLE_LABEL.search(line) else []):
                spans.append(rows)
                reasons.append({"reason": "restricted_table_rows", "detail": text[rows[0]:rows[1]].strip()[:300], "offset": rows[0]})
        else:
            reasons.append(item)

    if drop_reasons:
        return "drop", drop_reasons + reasons, []
    if spans:
        return "strip", reasons, spans
    return "keep", reasons, []


def apply_strip(text, spans):
    pieces = []
    cursor = 0
    for start, end in sorted(spans):
        start = max(start, cursor)
        piece = text[cursor:start]
        if pieces and piece[:1] == " " and pieces[-1][-1:] in (" ", "\n", ""):
            piece = piece.lstrip(" ")
        pieces.append(piece)
        cursor = max(cursor, end)
    rest = text[cursor:]
    if pieces and rest[:1] == " " and pieces[-1][-1:] in (" ", "\n", ""):
        rest = rest.lstrip(" ")
    pieces.append(rest)
    return "".join(pieces)


def text_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def apply_action(doc, row):
    """Return the rewritten doc for one actions row, or None for a drop; raise if the row is for another text."""
    text = doc.get("text", "") or ""
    if row["sha1"] != text_sha1(text):
        raise ValueError(f"sha1 mismatch for record {row['ordinal']} ({row['id']})")
    if row["action"] == "drop":
        return None
    return dict(doc, text=apply_strip(text, row["spans"]))


def iter_docs(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def scan_shard(args):
    path, out_dir, tag = args
    stats = {
        "docs": 0, "chars": 0,
        "metadata_licence": collections.Counter(),
        "action_docs": collections.Counter(), "action_chars": collections.Counter(),
        "reason_docs": collections.Counter(), "reason_chars": collections.Counter(),
        "strip_chars": 0,
        "examples": collections.defaultdict(list),
    }
    partial = os.path.join(out_dir, f".part_{tag}_{os.path.basename(path)}.jsonl")
    with open(partial, "w", encoding="utf-8") as out:
        for ordinal, doc in enumerate(iter_docs(path)):
            text = doc.get("text", "") or ""
            stats["docs"] += 1
            stats["chars"] += len(text)
            stats["metadata_licence"][str(metadata_licence(doc))] += 1
            action, reasons, spans = classify(doc)
            stats["action_docs"][action] += 1
            stats["action_chars"][action] += len(text)
            for name in {item["reason"] for item in reasons}:
                stats["reason_docs"][name] += 1
                stats["reason_chars"][name] += len(text)
            for item in reasons:
                bucket = stats["examples"][item["reason"]]
                if len(bucket) < EXAMPLES_PER_REASON and all(ex["id"] != doc.get("id") for ex in bucket):
                    bucket.append({"id": doc.get("id"), "shard": os.path.basename(path), "ordinal": ordinal, "text": item["detail"]})
            if action == "strip":
                stats["strip_chars"] += sum(end - start for start, end in spans)
            if action != "keep":
                record = {"shard": os.path.basename(path), "ordinal": ordinal, "sha1": text_sha1(text), "id": doc.get("id"),
                          "action": action, "reasons": reasons, "spans": spans}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path, partial, stats


def merge_stats(total, part):
    total["docs"] += part["docs"]
    total["chars"] += part["chars"]
    total["strip_chars"] += part["strip_chars"]
    for key in ("metadata_licence", "action_docs", "action_chars", "reason_docs", "reason_chars"):
        total[key].update(part[key])
    for name, items in part["examples"].items():
        bucket = total["examples"][name]
        bucket.extend(items[:max(0, EXAMPLES_PER_REASON - len(bucket))])


def shard_paths(source_dir):
    found = []
    for root, _, files in os.walk(source_dir):
        if ".cache" in root:
            continue
        found.extend(os.path.join(root, name) for name in files if name.endswith(".gz"))
    return sorted(found)


REGEX_DOC = {
    "trigger": TRIGGER.pattern, "nc_nd": NC_ND, "cc_context": RE_CC_CONTEXT.pattern, "all_rights_reserved": RE_ARR.pattern,
    "used_with_permission": RE_PERMISSION.pattern, "may_not_be_reproduced": RE_NO_REPRODUCE.pattern,
    "restricted_use": RE_RESTRICTED_USE.pattern, "pg_copyrighted_header": RE_PG_COPYRIGHTED.pattern,
    "pg_cleared": RE_PG_CLEARED.pattern, "copyright_year": RE_COPYRIGHT_YEAR.pattern,
    "media_words": RE_MEDIA.pattern, "caption_label": RE_CAPTION_LABEL.pattern, "caption_start": RE_CAPTION_START.pattern,
    "work_subject": RE_WORK_SUBJECT.pattern, "strong_subject": RE_STRONG_SUBJECT.pattern, "discussion_words": RE_DISCUSSION.pattern,
    "open_licence_nearby": RE_CC_LICENCE.pattern, "resource_list": RE_RESOURCE_LIST.pattern, "lyric_marker": RE_LYRIC_MARKER.pattern,
    "wikiquote_media_title": RE_WIKIQUOTE_MEDIA_TITLE.pattern, "wikiquote_media_heading": RE_WIKIQUOTE_MEDIA_HEADING.pattern,
    "metadata_not_commercial": NON_COMMERCIAL_METADATA.pattern, "pd_cutoff_year": PD_CUTOFF_YEAR, "max_strip_span": MAX_STRIP_SPAN,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--shards", nargs="*", help="basenames to scan; default all shards")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    paths = shard_paths(args.source_dir)
    all_count = len(paths)
    if args.shards:
        paths = [path for path in paths if os.path.basename(path) in set(args.shards)]
    os.makedirs(args.out_dir, exist_ok=True)
    total = {"docs": 0, "chars": 0, "metadata_licence": collections.Counter(), "action_docs": collections.Counter(),
             "action_chars": collections.Counter(), "reason_docs": collections.Counter(),
             "reason_chars": collections.Counter(), "strip_chars": 0, "examples": collections.defaultdict(list)}
    actions_path = os.path.join(args.out_dir, f"{args.name}_actions.jsonl")
    jobs = [(path, args.out_dir, args.name) for path in paths]
    with open(actions_path, "w", encoding="utf-8") as out, multiprocessing.Pool(args.workers) as pool:
        for path, partial, stats in pool.imap(scan_shard, jobs):
            merge_stats(total, stats)
            with open(partial, encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
            os.remove(partial)
            print(f"[{args.name}] {os.path.basename(path)} docs={stats['docs']} {dict(stats['action_docs'])}", file=sys.stderr, flush=True)

    summary = {
        "source": args.name, "source_dir": args.source_dir,
        "shards_scanned": [os.path.basename(path) for path in paths], "shards_total": all_count,
        "docs": total["docs"], "chars": total["chars"],
        "metadata_licence": dict(total["metadata_licence"].most_common()),
        "metadata_licence_not_commercial": {k: v for k, v in total["metadata_licence"].items() if not metadata_licence_ok(None if k == "None" else k)},
        "action_docs": dict(total["action_docs"]), "action_chars": dict(total["action_chars"]),
        "strip_chars_removed": total["strip_chars"],
        "reason_docs": dict(total["reason_docs"].most_common()), "reason_chars": dict(total["reason_chars"].most_common()),
        "examples": dict(total["examples"]), "regexes": REGEX_DOC,
    }
    with open(os.path.join(args.out_dir, f"{args.name}_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
