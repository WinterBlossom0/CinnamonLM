import unicodedata

import ftfy
import regex

# Encoding repairs as in datatrove FTFYFormatter (FineWeb, arXiv:2406.17557); format-enforcing fixes stay off.
FTFY_CONFIG = ftfy.TextFixerConfig(
    unescape_html="auto",
    remove_terminal_escapes=True,
    fix_encoding=True,
    restore_byte_a0=True,
    replace_lossy_sequences=True,
    decode_inconsistent_utf8=True,
    fix_c1_controls=True,
    fix_latin_ligatures=False,
    fix_character_width=False,
    uncurl_quotes=False,
    fix_line_breaks=False,
    fix_surrogates=True,
    remove_control_chars=False,
    normalization=None,
)

LINE_BREAK = regex.compile(r"\r\n?|[  ]")
# Soft hyphens and zero-width characters split words for language ID (Caswell et al., arXiv:2010.14571 §3).
INVISIBLE = regex.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\p{Cs}­​⁠⁪-⁯﻿￹-￻￾￿]"
)
NON_ASCII_SPACE = regex.compile(r"[^\S\n\t ]")
TRAILING_SPACE = regex.compile(r"[ \t]+(?=\n|\Z)")
BLANK_LINE_RUN = regex.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Repairs encoding, applies NFC, drops invisible characters, strips line ends and keeps one blank line per break.

    Leading indentation and runs of spaces inside a line stay, because verse, code and plain-text tables use them.
    """
    text = LINE_BREAK.sub("\n", text)
    text = ftfy.fix_text(text, config=FTFY_CONFIG)
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)
    text = NON_ASCII_SPACE.sub(" ", text)
    text = TRAILING_SPACE.sub("", text)
    text = BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()
