import json
from pathlib import Path

import pytest

from data_pipeline.common.langid import LanguageIdentifier

SNIPPETS_PATH = Path(__file__).parent / "data" / "real_snippets.json"


@pytest.fixture(scope="session")
def snippets() -> dict:
    return json.loads(SNIPPETS_PATH.read_text())


@pytest.fixture(scope="session")
def language_identifier() -> LanguageIdentifier:
    return LanguageIdentifier()
