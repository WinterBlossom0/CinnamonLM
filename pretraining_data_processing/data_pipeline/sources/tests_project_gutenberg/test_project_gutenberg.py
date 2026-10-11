"""Checks of the project_gutenberg cleaning rules on real excerpts from the source (data/real_snippets.json)."""
import json
from pathlib import Path

import pytest

from data_pipeline.common.dedup import ExactDedup
from data_pipeline.common.io import ShardWriter, read_documents
from data_pipeline.common.langid import LanguageIdentifier
from data_pipeline.common.pipeline import Pipeline
from data_pipeline.common.text import normalize_text
from data_pipeline.sources import project_gutenberg as pg

SNIPPETS = json.loads((Path(__file__).parent / "data" / "real_snippets.json").read_text())


@pytest.fixture(scope="module")
def identifier() -> LanguageIdentifier:
    return LanguageIdentifier()


def clean(text: str) -> str:
    text = normalize_text(pg.clean_pg_apparatus(text))
    return pg.normalize_markup(pg.unwrap_lines(pg.strip_page_markers(text)))


def document(doc_id: str, text: str, title: str = "") -> dict:
    return {"id": doc_id, "text": text, "source": "project gutenberg", "metadata": {"title": title}}


def test_note_at_the_top_of_a_short_book_keeps_the_story():
    text = pg.clean_pg_apparatus(SNIPPETS["transcriber_note_short_book"])
    assert "Transcriber" not in text and "Galaxy Science Fiction" not in text
    assert "My name is Andrews, third assistant vice president" in text
    assert len(text) > 0.9 * len(SNIPPETS["transcriber_note_short_book"])


def test_bracketed_note_that_quotes_bracket_codes_is_removed_whole():
    text = pg.clean_pg_apparatus(SNIPPETS["transcriber_note_nested_brackets"])
    assert "Transcriber" not in text and "female symbol" not in text
    assert "Pipistrellus cinnamomeus Miller 1902" in text and "Miller (Proc. Acad. Nat. Sci." in text


def test_notes_at_the_head_and_the_tail_are_removed_and_the_book_stays():
    text = pg.clean_pg_apparatus(SNIPPETS["transcriber_note_head_and_tail"])
    assert "Transcriber" not in text and "obvious typographical errors" not in text
    assert "The point of this trick consists in divining a word" in text
    assert "keep up a conversation." in text


def test_bracketed_note_naming_the_etext_is_removed():
    text = pg.clean_pg_apparatus(SNIPPETS["etext_note_in_brackets"])
    assert "note of etext transcriber" not in text and "Original text says" not in text
    assert "Of course,\" began Tom" in text


def test_errata_list_at_the_end_is_removed_and_the_footnotes_after_it_stay():
    # The rest of a real book comes first, because a tail note starts in the second half of a book.
    text = pg.clean_pg_apparatus(SNIPPETS["book"] + "\n\n\n\n\n" + SNIPPETS["errata_then_footnotes"])
    assert "griping his arm=> gripping his arm" not in text
    assert "FOOTNOTES:" in text and "Abnakis." in text


def test_editor_bookmarks_at_the_end_are_removed():
    text = pg.clean_pg_apparatus(SNIPPETS["book"] + "\n\n\n\n\n" + SNIPPETS["editor_bookmarks"])
    assert "BOOKMARKS" not in text and "Write so illegibly" not in text
    assert "A HERO OF OUR TIME" in text


def test_header_fragment_and_credit_line_are_removed():
    text = pg.clean_pg_apparatus(SNIPPETS["header_fragment"])
    assert "RELATED BY HIMSELF***" not in text and "Martin Schub" not in text
    assert text.lstrip().startswith("A NARRATIVE OF THE LIFE AND ADVENTURES")


def test_contents_and_illustration_lists_are_removed_but_the_title_stays():
    # The rest of a real book follows, because front matter is the first fifth of a book.
    book = SNIPPETS["contents_and_illustrations"] + "\n\n\n\n\n" + SNIPPETS["book"]
    text = pg.strip_navigation(book)
    assert "XVIII. GENERAL SURVEY OF ROMANCE LITERATURE" not in text
    assert "THE CID'S LAST VICTORY--Rochegrosse" not in text
    assert "LEGENDS OF THE MIDDLE AGES" in text and "A HERO OF OUR TIME" in text


def test_contents_list_stops_at_the_introduction_that_follows_it():
    # The rest of a real book follows, because the list limits scale with the book length.
    text = pg.clean_pg_apparatus(SNIPPETS["contents_then_introduction"] + "\n\n\n\n\n" + SNIPPETS["book"])
    assert "Collections from Jackson County, North Carolina          434" not in text
    assert "Mr. Palmer began his explorations early in July, 1881" in text


def test_prose_with_margin_numbers_is_joined_and_the_numbers_dropped():
    lines = SNIPPETS["margin_numbered_prose"].split("\n\n")[0].split("\n")
    joined = pg.unwrap_paragraph(lines, width=75)
    assert "\n" not in joined
    assert joined.startswith("I wish to place on record the proceedings in heaven October 13 last")


def test_prose_with_printed_line_breaks_is_joined():
    lines = SNIPPETS["printed_line_breaks"].split("\n\n")[0].split("\n")
    joined = pg.unwrap_paragraph(lines, width=71)
    assert "\n" not in joined and "(Nos. 93 and 87) salivation" in joined


def test_sentence_spacing_is_not_a_table_gap():
    lines = SNIPPETS["sentence_spacing"].split("\n\n")[0].split("\n")
    assert "\n" not in pg.unwrap_paragraph(lines, width=70)


def test_verse_and_tables_keep_their_lines():
    book = "\n\n".join([SNIPPETS["printed_line_breaks"].split("\n\n")[0]] * 4 + [SNIPPETS["verse"], SNIPPETS["table"]])
    text = pg.unwrap_lines(book)
    assert "Little Jack Jingle,\n    Played truant at school,\n  They made his bum tingle" in text
    assert SNIPPETS["table"].strip("\n") in text
    assert "(Nos. 93 and 87)" in text


def test_markup_keeps_captions_on_their_own_lines_and_drops_inline_tags():
    text = "He said [Illustration] hi there.\n\n[Illustration: Madame Waddington.\nFrom a photograph.]\n\nEnd."
    assert pg.normalize_markup(text) == "He said hi there.\n\nMadame Waddington.\nFrom a photograph.\n\nEnd."


def test_markup_keeps_drop_caps_and_drops_captions_that_only_name_the_image_kind():
    assert pg.normalize_markup("[Illustration: T]HE day was fine.") == "THE day was fine."
    text = "x\n\n[Illustration: Book Cover]\n\n[Illustration: Fig. 3.]\n\n[Illustration: Fig. 3. Section of a loom]"
    assert pg.normalize_markup(text) == "x\n\nFig. 3. Section of a loom"


def test_transliterated_greek_loses_its_accent_codes_and_stays_in_the_sentence():
    text = "marked by the character [Greek: to\\ du/nasthai lo/gon dido/nai]. You do not"
    assert pg.normalize_markup(text) == "marked by the character to dunasthai logon didonai. You do not"


def test_italic_money_units_lose_their_underscores():
    assert "6s.;" in pg.normalize_markup(SNIPPETS["italic_money"])


def test_english_share_rejects_an_unknown_language_that_lid_calls_english(identifier):
    ojibwe = clean(SNIPPETS["ojibwe"])
    assert identifier.english_share(ojibwe) > 0.2
    assert pg.english_share(identifier, ojibwe) < 0.1


def test_english_share_keeps_english_with_latin_names_and_drops_german(identifier):
    assert pg.english_share(identifier, clean(SNIPPETS["english_with_latin_names"])) >= pg.ENGLISH_MIN_SHARE
    assert pg.english_share(identifier, clean(SNIPPETS["book"])) > 0.95
    assert pg.english_share(identifier, clean(SNIPPETS["german"])) < pg.ENGLISH_MIN_SHARE


def test_book_text_filter_drops_stubs_and_catalogues_and_keeps_a_book():
    assert pg.is_book_text(document("1", clean(SNIPPETS["book"])))
    assert not pg.is_book_text(document("2", "README FILE ONLY:\n\nBEZETTE STAD\n\nBy Paul van Ostayen"))
    readme = ("Moby (tm) Hyphenator II for MSDOS operating systems is compressed and distributed as a single zip "
              "file. After decompression the hyphenation file included with this product is in ordinary ASCII.")
    assert not pg.is_book_text(document("3", readme))
    assert not pg.is_book_text(document("4", clean(SNIPPETS["book"]),
                                        title="Index of the Project Gutenberg Works of George Eliot"))


def write_shard(path: Path, documents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ShardWriter(path) as writer:
        for item in documents:
            writer.write(item)


def test_part_volume_and_exact_copy_are_dropped_and_partial_overlap_is_kept(tmp_path):
    book = clean(SNIPPETS["book"])
    other = clean(SNIPPETS["english_with_latin_names"])
    half_overlap = book[40000:50000] + "\n\n" + clean(SNIPPETS["contents_then_introduction"])
    write_shard(tmp_path / "data" / "00000.jsonl.gz",
                [document("part", book[40000:70000] + "\n\n" + SNIPPETS["german"]), document("complete", book), document("other", other)])
    write_shard(tmp_path / "data" / "00001.jsonl.gz",
                [document("copy", other), document("half", half_overlap)])
    Pipeline("test", [ExactDedup(), pg.ContainmentDedup()]).run_in_place(tmp_path / "data", tmp_path / "meta")
    kept = [item["id"] for shard in sorted((tmp_path / "data").glob("*.jsonl.gz")) for item in read_documents(shard)]
    assert kept == ["complete", "other", "half"]


def test_subscripts_survive_page_marker_removal_and_become_unicode():
    # Book 34457 (The Phase Rule); its page numbers are written as "{25}".
    text = ("the solid will be expressed by Na_{2}SO_{4}, 10H_{2}O; that of the solution\n"
            "by Na_{2}SO_{4} + _x_H_{2}O, while the vapour phase will be H_{2}O.\n\n{25}\n\nThe system")
    assert pg.normalize_markup(pg.strip_page_markers(text)) == (
        "the solid will be expressed by Na₂SO₄, 10H₂O; that of the solution\n"
        "by Na₂SO₄ + _x_H₂O, while the vapour phase will be H₂O.\n\nThe system")


def test_wrapped_line_naming_figures_or_plates_is_not_a_list_heading():
    # Books 14626 and 65997: "figures." and "plates." end wrapped sentences in the first fifth of the book.
    story = ("When the smoke cleared away, the water nearby was filled with struggling\nfigures.\n\n"
             "\"Lower the boats,\" shouted Jack.\n\nInstantly men sprang to obey the command, while others of the "
             "British tars\nstill stood quietly behind their guns, their eyes scanning the sea.")
    governor = ("drums to it for gallantry in the field; and then sent in a bill for the\nplates.\n\n"
                "He is not the only peculiar governor with which the Gambia has been\nafflicted; one in particular "
                "I can remember who was notorious for his\nparsimony throughout West Africa.")
    for excerpt in (story, governor):
        assert excerpt in pg.strip_navigation(excerpt + "\n\n\n\n\n" + SNIPPETS["book"])


def test_index_with_volume_and_single_space_references_is_removed():
    # Lines from books 72881 ("Ablaincourt, i. 197") and 14613 ("Abiko 105", the glossary note).
    index = ("INDEX\n\n\n_This Index may be regarded as a Glossary inasmuch as every Japanese\nword which occurs in the "
             "book will be found in it._\n\n  Abbat, Max, ii. 56\n\n  Abbeville, i. 26, 211\n\n"
             "  Abingley Camp, i. 231; ii. 156-157\n\n  Ablaincourt, i. 197\n\nAbiko 105\n\nAbility 66\n\n"
             "Abortion 65, 303;\n  Abortifacient 332\n\nAcreage, see Agriculture\n\nActing 115 (2), 320\n\n"
             "  Air warfare:\n    British activity on the Somme, i. 159, 196-197\n")
    tail = "\n\n\nTHE END."
    text = pg.strip_navigation(SNIPPETS["book"] + "\n\n\n\n\n" + index + tail)
    assert "Ablaincourt" not in text and "Abiko" not in text and "Glossary" not in text
    assert text.endswith(tail) and "A HERO OF OUR TIME" in text


def test_index_headings_with_a_kind_are_found_and_the_index_stops_at_the_notes():
    # Book 34457: "NAME INDEX", then "SUBJECT INDEX", then the endnotes, whose citations look like index entries.
    index = ("NAME INDEX\n\n          A\n  Abegg, 52\n  Adriani, 186, 217, 220\n  Alexejeff, 97, 125\n  Allan, 298\n"
             "  Allen, L. E., 109\n  Allen, R. W., 63\n  Ampolla, 213\n  Andreä, 109\n\nSUBJECT INDEX\n\n          Z\n"
             "  Zeolites, 176\n  Zinc, lead, silver, 246\n  ---- chloride in water, solubility of, 157\n\nTHE END\n\n"
             "PRINTED BY WILLIAM CLOWES AND SONS, LIMITED, LONDON AND BECCLES.\n\n"
             "       *       *       *       *       *\n\n\nNOTES\n\n[245] Stortenbeker, _Zeitschr. physikal. Chem._, 1889, "
             "3. 22.\n\n[246] Ramsay and Young, _Journ. Chem. Soc._, 1886, 49. 458.\n")
    text = pg.strip_navigation(SNIPPETS["book"] + "\n\n\n\n\n" + index)
    assert "Abegg" not in text and "Zeolites" not in text
    assert "[245] Stortenbeker" in text and "[246] Ramsay and Young" in text
    # Book 38519: an article and a mixed-case kind before the heading.
    index = ("              AN Alphabetical INDEX TO THE FOURTH VOLUME.\n\n\n          A.\n\n  AGRIPPA, 132.\n\n"
             "  AICHSTED, _t._, 266.\n\n  _St. Aignan_ (Duke of), 3.\n    Order’d to quit _Spain_, 7, 8.\n\n"
             "  AIX, in _Provence_, _t._, 100.\n\n  _Alberoni_ (Cardinal), 2, 7, 8, 19, 20, 94, 211, 213, _&c._\n\n"
             "  ALCALA, _t._, 204.\n")
    assert "AGRIPPA" not in pg.strip_navigation(SNIPPETS["book"] + "\n\n\n\n\n" + index)


def test_speaker_label_sidenote_and_margin_page_do_not_stop_unwrapping():
    # Book 100 (The Complete Works of Shakespeare); labels may carry a number ("1 PETITIONER.").
    speech = ["BERTRAM.", "And I in going, madam, weep o’er my father’s death anew; but I must",
              "attend his majesty’s command, to whom I am now in ward, evermore in", "subjection."]
    assert pg.unwrap_paragraph(speech, width=69) == (
        "BERTRAM.\nAnd I in going, madam, weep o’er my father’s death anew; but I must attend his majesty’s "
        "command, to whom I am now in ward, evermore in subjection.")
    petition = ["1 PETITIONER.", "Mine is, an ’t please your grace, against John Goodman, my Lord",
                "Cardinal’s man, for keeping my house and lands, and wife and all, from", "me."]
    assert pg.unwrap_paragraph(petition, width=69) == (
        "1 PETITIONER.\nMine is, an ’t please your grace, against John Goodman, my Lord Cardinal’s man, for keeping "
        "my house and lands, and wife and all, from me.")
    # Book 63420 runs a sidenote into the first line; book 12456 puts page numbers in the right margin.
    sidenoted = ["[Sidenote: 36.", "The Drug in the Nilgiri Hills.] There are magnificent landscapes at the",
                 "edge of the Nilgiris, where the mountains descend abruptly to the", "plains."]
    assert pg.unwrap_paragraph(sidenoted, width=71) == (
        "[Sidenote: 36. The Drug in the Nilgiri Hills.]\nThere are magnificent landscapes at the edge of the "
        "Nilgiris, where the mountains descend abruptly to the plains.")
    margin = ["he remains the greatest of troubadour poets, as modern taste regards      [50]", "poetry."]
    assert pg.unwrap_paragraph(margin, width=72).endswith("as modern taste regards poetry.")


def test_book_with_long_lines_still_gets_a_wrap_width():
    # Book 42479 (La Ronge Journal): many manuscript lines of 81-100 characters.
    prose = ["The sick one drew near in a very slow and thoughtful manner. However,"] * 30
    long_lines = ["attached two long swan quills inclosing the foot thus  with a tuft of down at each end"] * 10
    assert pg.wrap_width([prose + ["end"], long_lines + ["end"]]) is not None


def test_paragraphs_wrapped_at_their_own_width_are_joined_and_narrow_stanzas_are_not():
    # Book 1 wraps "The Declaration of Independence" at about 65 columns, narrower than the book.
    declaration = ["He has refused for a long time, after such dissolutions,",
                   "to cause others to be elected; whereby the Legislative Powers,",
                   "incapable of Annihilation, have returned to the People at large",
                   "for their exercise; the State remaining in the mean time exposed",
                   "to all the dangers of invasion from without, and convulsions within."]
    assert "\n" not in pg.unwrap_paragraph(declaration, width=77)
    # Book 5826: the first line of a footnote is longer than MAX_WRAP_WIDTH.
    footnote = ["[Footnote 47: \"My master\" was a Common appellation for Mr. Thrale,--and one which",
                "he seems, in earnest, to have deserved. \"I know no man,\" said Johnson,",
                "\"who is more master of his wife and family than Thrale, he but holds up",
                "a finger, he is obeyed.\" (Boswell.)--ED.]"]
    assert "\n" not in pg.unwrap_paragraph(footnote, width=72)
    # Book 38930: a stanza whose lines end in punctuation and open with a German quote stays verse.
    stanza = ["Da zog ein lieblich Lächeln um der Holden Lippen,", "Und fragend schaute sie zum Vater aus.",
              "Doch weil der schwieg, so galt es selbst zu reden,", "Und so erwiederte sie sittsam draus:",
              "„Wollt Ihr so gut sein und mein Körblein tragen,", "Darf mit zu gehen Euch ich nicht versagen.“"]
    assert pg.unwrap_paragraph(stanza, width=54) == "\n".join(stanza)
    assert pg.is_end_stopped(stanza) and not pg.is_end_stopped(declaration)


def test_verse_lines_opening_with_an_elision_keep_their_breaks():
    # Book 1004 (Longfellow's Divine Comedy).
    tercet = ("He shall not feed on either earth or pelf,\n    But upon wisdom, and on love and virtue;\n"
              "    ’Twixt Feltro and Feltro shall his nation be;")
    assert pg.unwrap_paragraph(tercet.split("\n"), width=54) == tercet


def test_brace_box_and_printed_original_notes_are_removed():
    # Brace note as in books 21872 and 17000; box as in books 19378, 19379 and 21915.
    assert pg.strip_transcriber_notes(
        "Start.\n\n{Transcriber’s Note:\n\nMaterial added by the\ntranscriber is in braces {}.}\n\nStory.") == (
        "Start.\n\n\n\nStory.")
    box = ("    +------------------------------------------+\n    | Typographical errors corrected in text:  |\n"
           "    | Page  60: argumeut replaced with argument |\n    +------------------------------------------+")
    assert "argumeut" not in pg.strip_transcriber_boxes("Story.\n\n" + box)
    # Books 27009 and 38538, then book 72813.
    tail = "\n\n\nCorrections made to printed original.\n\npage 497, \"This is on the floor\": 'This in' in original."
    assert "Corrections made" not in pg.strip_transcriber_notes(SNIPPETS["book"] + tail)
    errata = "\n\n\nERRATA.\n\nThe following obvious typographical errors have been corrected:\n\nOn page 9: Changed"
    assert "ERRATA" not in pg.strip_transcriber_notes(SNIPPETS["book"] + errata)


def test_errata_noted_by_transcriber_and_printed_errata_are_removed():
    # Book 18816 ends with "Errata Noted by Transcriber:".
    tail = ("\n\n    LEE AND SHEPARD Publishers BOSTON\n\n       *       *       *       *       *\n\n"
            "Errata Noted by Transcriber:\n\nInvisible punctuation-- chiefly quotation marks-- has been silently\n"
            "supplied. The spelling \"cockswain\" is standard for this text.\n\ncertain irregular enterprizes\n"
            "  _spelling as in original_\n")
    text = pg.strip_transcriber_notes(SNIPPETS["book"] + tail)
    assert "cockswain" not in text and "LEE AND SHEPARD" in text
    # Book 27229 prints its errata list in the front matter.
    front = ("retained for the sake of local colour.\"\n\n\n                                ERRATA\n\n"
             "     P. 25, line 6 from bottom, _for_ \"_weis_\" _read_ \"_weiss_.\"\n\n"
             "     P. 43, line 14, _for_ \"balolaika\" _read_ \"balalaika.\"\n\n\n\n\nCHAPTER I")
    text = pg.strip_transcriber_notes(front + "\n\n" + SNIPPETS["book"])
    assert "ERRATA" not in text and "balolaika" not in text
    assert "local colour" in text and "CHAPTER I" in text


def test_front_note_stops_at_the_title_page_or_the_asterisks():
    # Book 30017: the title follows the note after two blank lines.
    note = ("                         Transcriber's Note:\n\n       Extensive research did not uncover any evidence "
            "that the\n       U.S. copyright on this publication was renewed.\n\n\n"
            "                          MY FATHER'S DRAGON\n\n\n\n                               STORY BY\n")
    text = pg.strip_transcriber_notes(note + "\n\n\n\n" + SNIPPETS["book"])
    assert "Extensive research" not in text and "MY FATHER'S DRAGON" in text and "STORY BY" in text
    # Book 34450: the note ends at a row of asterisks before the series page.
    note = ("Transcriber's Notes\n\nChanges to the text have been made only in the case of obvious spelling\n"
            "or type-setting errors. These are listed at the end of the book.\n\n"
            "       *       *       *       *       *\n\n\n             MONOGRAPHS ON EXPERIMENTAL BIOLOGY\n\n"
            "                         EDITED BY\n\n            JACQUES LOEB, Rockefeller Institute\n\n\n\n\n")
    text = pg.strip_transcriber_notes(note + SNIPPETS["book"])
    assert "type-setting errors" not in text and "MONOGRAPHS ON EXPERIMENTAL BIOLOGY" in text


def test_dot_below_code_applies_only_inside_a_word():
    # "Pin[h.]as" is a real code (audit example); book 38538 labels etymologies "[L.]" next to the code "[=u]".
    assert pg.normalize_markup("an Essene Tannaite, Pin[h.]as ben Jaïr") == "an Essene Tannaite, Pinḥas ben Jaïr"
    assert pg.normalize_markup("EUGE, [=u]'j[=e], _interj._ well! well done! [L.]") == (
        "EUGE, ū'jē, interj. well! well done! [L.]")
    # Book 38492 ("Miss [M.]") and book 10080 ("J.W. 21 [o.]") use one-letter labels.
    assert pg.normalize_markup("seen by another Miss\n[M.] ... come up") == "seen by another Miss\n[M.] ... come up"
    assert pg.normalize_markup("Reference No: J.W. 21 [o.]") == "Reference No: J.W. 21 [o.]"


def test_braced_abbreviation_letters_become_plain_and_exponents_stay():
    # 24611 "SONS L^{TD}", 24950 "M^r. ... a m^{r}chant"; X^{n} is an exponent in an algebra book.
    assert pg.normalize_markup("BENROSE & SONS L^{TD}; a m^{r}chant; w^{ch} you; X^{n}") == \
        "BENROSE & SONS LTD; a mrchant; wch you; X^{n}"
