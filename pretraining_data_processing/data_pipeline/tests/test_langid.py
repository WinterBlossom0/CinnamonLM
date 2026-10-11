from data_pipeline.common.langid import ENGLISH_MIN_SHARE, SEGMENT_CHARS, language_segments


def test_english_and_spanish_libretexts_pages(snippets, language_identifier):
    assert language_identifier.english_share(snippets["english"]["text"]) == 1.0
    assert language_identifier.is_english(snippets["english"]["text"])
    assert language_identifier.english_share(snippets["spanish"]["text"]) < 0.1
    assert not language_identifier.is_english(snippets["spanish"]["text"])


def test_french_gutenberg_novel_is_removed(snippets, language_identifier):
    assert language_identifier.english_share(snippets["french_gutenberg"]["text"]) < 0.1
    assert not language_identifier.is_english(snippets["french_gutenberg"]["text"])


def test_english_lesson_on_spanish_grammar_is_kept(snippets, language_identifier):
    text = snippets["english_teaching_spanish"]["text"]
    share = language_identifier.english_share(text)
    assert 0.6 < share < 0.8
    assert language_identifier.is_english(text)
    assert not language_identifier.is_english(text, min_share=0.8)


def test_english_module_with_long_spanish_passage_is_kept(snippets, language_identifier):
    text = snippets["english_with_spanish_passage"]["text"]
    # One prediction over the whole page ranks English below 0.5, so a document-level score would drop it.
    assert language_identifier.predict(text)[1] < 0.5
    assert language_identifier.is_english(text)


def test_default_threshold_is_inclusive_half(language_identifier, monkeypatch):
    assert ENGLISH_MIN_SHARE == 0.5
    monkeypatch.setattr(language_identifier, "english_share", lambda text: 0.5)
    assert language_identifier.is_english("x")
    monkeypatch.setattr(language_identifier, "english_share", lambda text: 0.4999)
    assert not language_identifier.is_english("x")


def test_segments_split_paragraphs_and_chunk_one_line_pages(snippets):
    page = snippets["english_with_spanish_passage"]["text"].replace("\n", " ")
    segments = language_segments(page)
    assert len(segments) > 1
    assert all(len(segment) <= SEGMENT_CHARS for segment in segments)
    assert "".join(segments).replace(" ", "") == page.replace(" ", "")
    assert language_segments("Title\n\n  \n12 34 %\n\nBody text\nsecond line") == ["Title", "Body text\nsecond line"]
    assert [len(segment) for segment in language_segments("x" * 2000)] == [767, 767, 466]


def test_text_without_letters_and_newlines_do_not_raise(language_identifier):
    assert language_identifier.english_share("") == 0.0
    assert language_identifier.english_share("12 345\n\n= 6.7 %") == 0.0
    assert language_identifier.predict("first line\nsecond line\n")[0] == "en"
    assert language_identifier.english_share("The first line of this paragraph.\nThe second line of it.") == 1.0
