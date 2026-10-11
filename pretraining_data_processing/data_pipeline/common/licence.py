"""Loads the reviewed commercial-use licence filter (licence/licence_filter.py) and applies its credit strips."""
import hashlib
import importlib.util
from functools import cache
from pathlib import Path

from .pipeline import Step

LICENCE_FILTER_PATH = Path(__file__).resolve().parents[2] / "licence/licence_filter.py"
# The hash pins the reviewed filter; a changed filter must be reviewed before pipelines use it.
LICENCE_FILTER_SHA256_PREFIX = "c00d5d95"


@cache
def load_licence_filter():
    source = LICENCE_FILTER_PATH.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if not digest.startswith(LICENCE_FILTER_SHA256_PREFIX):
        raise RuntimeError(f"{LICENCE_FILTER_PATH} has sha256 {digest}, expected prefix {LICENCE_FILTER_SHA256_PREFIX}")
    spec = importlib.util.spec_from_file_location("licence_filter", LICENCE_FILTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_restricted_credits(document: dict) -> str:
    """Removes the media-credit spans that the filter marks 'strip' in the final text.

    Drop decisions are not applied here: the licence pass decides drops on the source text, and cleaning removes the
    notes (for example Project Gutenberg's renewal research) that clear some documents.
    """
    licence = load_licence_filter()
    text = document["text"]
    if not licence.TRIGGER.search(text):
        return text
    action, _, spans = licence.classify(document)
    return licence.apply_strip(text, spans) if action == "strip" else text


class LicenceCreditStrip(Step):
    """The final licence step: applies strip_restricted_credits to the cleaned text."""

    def apply(self, document: dict) -> dict:
        text = strip_restricted_credits(document)
        return document if text == document["text"] else {**document, "text": text}
