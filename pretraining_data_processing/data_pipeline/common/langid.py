from pathlib import Path

import fasttext
import regex

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lid.176.bin"
# Dolma (arXiv:2402.00159, Project Gutenberg) drops a book whose per-paragraph English score averages below 0.5.
ENGLISH_MIN_SHARE = 0.5
# Institutional Books 1.0 (arXiv:2506.08300 §4.4) identifies language on 768-character chunks.
SEGMENT_CHARS = 768
PARAGRAPH_BREAK = regex.compile(r"\n[^\S\n]*\n")
# A chunk ends at whitespace when one exists within SEGMENT_CHARS; a longer unbroken run is cut at SEGMENT_CHARS.
CHUNK = regex.compile(rf"\S(?:.{{0,{SEGMENT_CHARS - 2}}}\S(?=\s|\Z)|\S{{0,{SEGMENT_CHARS - 2}}})", regex.DOTALL)
LETTER = regex.compile(r"\p{L}")
ENGLISH_LABEL = "__label__en"


def language_segments(text: str) -> list[str]:
    """Paragraphs (split at blank lines), with paragraphs longer than SEGMENT_CHARS cut into chunks at whitespace.

    Pages without blank lines (most LibreTexts pages are one line) still get several segments this way.
    """
    segments = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        if len(paragraph) <= SEGMENT_CHARS:
            segments.append(paragraph)
        else:
            segments.extend(match.group(0) for match in CHUNK.finditer(paragraph))
    return [segment for segment in segments if LETTER.search(segment)]


def fasttext_input(text: str) -> str:
    # Dolma lowercases before prediction; fastText raises on newlines.
    return text.lower().replace("\n", " ").strip()


class LanguageIdentifier:
    """fastText lid.176 classifier; the model loads at construction so forked workers share its memory."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model = fasttext.load_model(str(model_path))

    def predict(self, text: str) -> tuple[str, float]:
        """Top language code and its probability for the whole text."""
        labels, scores = self.model.predict(fasttext_input(text), k=1)
        return labels[0].removeprefix("__label__"), float(scores[0])

    def english_share(self, text: str) -> float:
        """Fraction of the letters of text that are in segments whose top lid.176 label is English.

        Weights are letter counts, so digits, formulas and punctuation do not count; text without letters gives 0.0.
        """
        segments = language_segments(text)
        if not segments:
            return 0.0
        labels, _ = self.model.predict([fasttext_input(segment) for segment in segments], k=1)
        english_letters = total_letters = 0
        for segment, segment_labels in zip(segments, labels):
            letters = len(LETTER.findall(segment))
            total_letters += letters
            if segment_labels[0] == ENGLISH_LABEL:
                english_letters += letters
        return english_letters / total_letters

    def is_english(self, text: str, min_share: float = ENGLISH_MIN_SHARE) -> bool:
        """Document decision: keep when at least min_share of the letters are in English segments."""
        return self.english_share(text) >= min_share
