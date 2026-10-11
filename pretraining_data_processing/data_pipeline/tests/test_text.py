from data_pipeline.common.text import normalize_text


def test_repairs_mojibake_and_html_entities():
    assert normalize_text("cafÃ© â€œquotedâ€\x9d") == "café “quoted”"
    assert normalize_text("Fish &amp; chips") == "Fish & chips"


def test_composes_to_nfc():
    assert normalize_text("Café") == "Café"


def test_maps_cp1252_c1_bullet_from_gutenberg(snippets):
    text = normalize_text(snippets["c1_bullet"]["text"])
    assert "Purgatorio • Canto I" in text
    assert "\x95" not in text


def test_removes_form_feed_from_doab_formula(snippets):
    text = normalize_text(snippets["form_feed"]["text"])
    assert "\x0c" not in text
    assert "*rj*  ≤ 1 and *sj*  ≤ 1" in text


def test_removes_soft_hyphen_and_word_joiner(snippets):
    assert "by shipping drugs" in normalize_text(snippets["soft_hyphen"]["text"])
    assert "arise.'—Aristotle" in normalize_text(snippets["word_joiner"]["text"])


def test_maps_no_break_space_to_space(snippets):
    assert "background.svg Not done" in normalize_text(snippets["nbsp"]["text"])


def test_line_breaks_trailing_space_and_blank_lines():
    raw = "  Title \r\n\r\n\r\n\r\nFirst para line one\t \rline two \n\n\n\nSecond para  \n \n"
    assert normalize_text(raw) == "Title\n\nFirst para line one\nline two\n\nSecond para"


def test_keeps_paragraph_breaks_indentation_and_inner_spacing(snippets):
    verse = "Para one.\n\n  Per correr miglior acque alza le vele\n  omai la navicella\n\nName      Age\nAnn       31"
    assert normalize_text(verse) == verse


def test_ftfy_replaces_lone_surrogate_and_tabs_stay():
    assert normalize_text("a\ud800b\tc") == "a\ufffdb\tc"


def test_is_idempotent_on_real_snippets(snippets):
    for name in ("c1_bullet", "form_feed", "nbsp", "soft_hyphen", "english", "spanish"):
        once = normalize_text(snippets[name]["text"])
        assert normalize_text(once) == once


def test_empty_and_whitespace_only():
    assert normalize_text("") == ""
    assert normalize_text(" \n\n\t \n") == ""
