"""Checks of the libretexts cleaning rules on real excerpts from the source."""
import json
from pathlib import Path

import pytest

from data_pipeline.common.langid import LanguageIdentifier
from data_pipeline.sources import libretexts as lt

SNIPPETS = json.loads((Path(__file__).parent / "data" / "real_snippets.json").read_text(encoding="utf-8"))


def document(name: str) -> dict:
    snippet = SNIPPETS[name]
    return {"id": snippet["id"], "text": snippet["text"], "metadata": {"author": snippet.get("author", "")}}


def apply_steps(item: dict, names: tuple[str, ...]) -> dict | None:
    steps = {step.name: step for step in lt.build_pipeline().steps}
    for name in names:
        item = steps[name].apply(item)
        if item is None:
            return None
    return item


@pytest.fixture(scope="module")
def language() -> LanguageIdentifier:
    return LanguageIdentifier()


def test_spanish_site_chrome_page_is_emptied():
    assert lt.strip_site_chrome(SNIPPETS["spanish_chrome"]["text"]) == ""


def test_byline_with_period_in_author_name_is_removed():
    text = lt.strip_page_header(document("openrn_byline"))
    assert text.startswith("2.2: Ethical and Professional Foundations of Safe Medication Administration by Nurses\n"
                           "ANA Code of Ethics for Nurses\n")
    assert "Ernstmeyer" not in text and "Chippewa Valley" not in text


def test_attribution_block_stops_at_body_heading():
    text = lt.strip_attribution(SNIPPETS["boundless_attribution"]["text"])
    assert "Provided by" not in text and "Contributions and Attributions" not in text
    assert "\nKey Points\n- The three major Domains of Life include" in text


def test_quiz_stub_is_dropped_by_min_words():
    names = ("normalize", "strip_site_chrome", "strip_page_header", "strip_attribution", "min_words")
    assert apply_steps(document("lumen_self_check"), names) is None


@pytest.mark.parametrize("name", ["english_lesson_spanish_grammar", "english_lesson_french_phonetics",
                                  "english_lesson_spanish_weather", "english_c_code_examples",
                                  "english_math_exercises", "formation_constants_table"])
def test_english_and_language_neutral_pages_are_kept(language, name):
    assert lt.is_english(language, document(name))


@pytest.mark.parametrize("name", ["spanish_algebra_exam", "spanish_algebra_lesson", "spanish_dialogue_lesson"])
def test_foreign_language_pages_are_dropped(language, name):
    assert not lt.is_english(language, document(name))


def test_math_does_not_count_as_english(language):
    lesson = SNIPPETS["spanish_algebra_lesson"]["text"]
    assert language.english_share(lesson) >= lt.ENGLISH_MIN_SHARE
    assert language.english_share(lt.prose_for_language_id(lesson)) < lt.ENGLISH_MIN_SHARE


def test_cell_per_line_table_is_dropped_and_code_listing_kept():
    assert lt.short_line_fraction(SNIPPETS["atomic_mass_table"]["text"]) >= lt.MAX_SHORT_LINE_FRACTION
    assert lt.short_line_fraction(SNIPPETS["english_c_code_examples"]["text"]) < lt.MAX_SHORT_LINE_FRACTION


def test_section_numbers_are_not_masked_as_ip_addresses():
    line = SNIPPETS["pii_section_number"]["text"]
    masked = apply_steps({"id": "x", "text": line, "metadata": {}}, ("mask_pii",))["text"]
    assert "Table 15.3.7.1" in masked
    email = "Questions: Email me at jennifer.paris@canyons.edu"
    assert apply_steps({"id": "x", "text": email, "metadata": {}}, ("mask_pii",))["text"].endswith("<EMAIL_ADDRESS>")


def test_flattened_formulas_and_orbitals_are_rejoined():
    table = lt.join_subscripts(SNIPPETS["chem_formula_table"]["text"])
    assert "| NaCl, sodium chloride | Na2O, sodium oxide |" in table
    assert "| CaI2, calcium iodide | Mg3N2, magnesium nitride |" in table
    ions = lt.join_subscripts(SNIPPETS["chem_complex_ion"]["text"])
    assert "P(CH3)3, can act as a ligand" in ions and "[Co(en)3]Cl3" in ions and "Mg3[Cr(CN)6]2" in ions
    configuration = lt.join_subscripts(SNIPPETS["electron_configuration"]["text"])
    assert "1s2 2s2 2p6 3s2 3p6 4s1 and" in configuration


@pytest.mark.parametrize("name", ["interest_word_symbol", "legal_citation"])
def test_numbers_after_words_and_citations_stay_apart(name):
    text = SNIPPETS[name]["text"]
    assert lt.join_subscripts(text) == text


def test_leaked_html_line_is_removed_and_table_kept():
    text = lt.strip_html_residue(SNIPPETS["html_alt_residue"]["text"])
    assert "data-quail-id" not in text and "subscript 4 calcium sulfate" not in text
    assert "| NaHCO 3 , sodium bicarbonate | CaSO 4 , calcium sulfate |" in text


def test_inline_licence_is_cut_from_the_first_line_of_a_poem():
    text = lt.strip_attribution(SNIPPETS["anthology_inline_licence"]["text"])
    assert "License" not in text
    assert "\nO wild West Wind, thou breath of Autumn's being,\n" in text


def test_parenthetical_credit_loses_only_its_span():
    text = lt.strip_attribution(SNIPPETS["parenthetical_credit"]["text"])
    assert "Paso 1. Escucha a Cristina y decide si estas afirmaciones son ciertas" in text


@pytest.mark.parametrize("name, sentence", [
    ("glossary_license_term", "- License : The legal terms under which a person is allowed to use a product."),
    ("exercise_public_domain", "Exercise 1. In the following paragraph, identify all the nouns, verbs, and adjectives:"),
])
def test_prose_that_mentions_licences_is_kept(name, sentence):
    assert sentence in lt.strip_attribution(SNIPPETS[name]["text"])


def test_summary_printed_above_identical_paragraph_appears_once():
    names = ("normalize", "strip_site_chrome", "strip_page_header", "strip_attribution", "repair_inline")
    text = apply_steps(document("repeated_summary"), names)["text"]
    assert text.count("Work through the Respiratory System Worksheet") == 1


def licence_document(name: str) -> dict:
    snippet = SNIPPETS[name]
    return {"id": snippet["id"], "text": snippet["text"], "source": "libretexts",
            "metadata": {"license": snippet["license"], "title": snippet["title"]}}


def test_control_spaces_in_math_are_kept_and_a_missing_closer_is_repaired():
    text = SNIPPETS["control_space_in_math"]["text"]
    assert lt.clean_latex(text) == text
    assert lt.clean_latex(SNIPPETS["spaced_closing_delimiter"]["text"]) == "There is one solution: \\(2\\)."


@pytest.mark.parametrize("text, expected", [
    ("Figure \\(PageIndex{3}\\): Vertical position", "Figure 3: Vertical position"),
    ("as illustrated in Figure \\(\\ PageIndex {1}\\),", "as illustrated in Figure 1,"),
    ("Figure \\(\\PageIndex{3\\)", "Figure 3"),
])
def test_malformed_page_index_is_resolved(text, expected):
    assert lt.clean_latex(text) == expected


def test_truncated_url_keeps_closing_punctuation():
    text = "(retrieved 13 may, 2010 from www.norc.org/GSS+Website/Brow...Subject+Index/). Most see continuity."
    assert lt.repair_inline(text) == "(retrieved 13 may, 2010 from). Most see continuity."


@pytest.mark.parametrize("flattened, joined", [
    ("amino group (–NH 2 )", "amino group (–NH2)"), ("ln[H 2 O2]", "ln[H2O2]"), ("Δ[N 2 ]", "Δ[N2]"),
    ("H2 C=CH 2", "H2C=CH2"), ("CH3 -CH 2 -CH 3", "CH3-CH2-CH3"), ("rate = k[C 4 H8]", "rate = k[C4H8]"),
    ("H0 H1 and B12", "H0 H1 and B12"), ("CC-BY 4.0", "CC-BY 4.0"),
])
def test_prefixed_formulas_are_rejoined_and_labels_kept(flattened, joined):
    assert lt.join_subscripts(flattened) == joined


def test_exploded_page_rejoins_inline_symbols():
    text = lt.join_fragments(SNIPPETS["exploded_inline_symbols"]["text"])
    assert "If G is a group consisting of a finite number of elements" in text
    assert "the point group m 3 m has order 48." in text


def test_exploded_attribution_fields_join_into_licence_lines():
    text = lt.join_fragments("MCAT .\nProvided by\n: Khan Academy.\nLocated at\n:\nhttps://www.youtube.com/watch?v=Jt3\n.\n"
                             "License\n:\nOther\n.")
    assert lt.strip_attribution(text) == "MCAT ."


def test_glossary_and_code_lines_stay_apart():
    glossary = ("organic compound with a hydroxyl group (–OH) bonded to a carbon atom\naldehyde\n"
                "organic compound containing a carbonyl group\nalkane\nmolecule consisting of only carbon and hydrogen\n"
                "alkene\nmolecule consisting of carbon and hydrogen")
    assert lt.join_fragments(glossary) == glossary
    code = ('Here is the code:\nstring = "Hello"\nprint("string: " + string)\nprint("lower: " + string.lower())\n'
            'for i in range(3):\nprint(i)')
    assert lt.join_fragments(code) == code


def test_split_nd_declaration_drops_the_page():
    assert lt.LicenceRecheck("licence_recheck").apply(licence_document("split_nd_declaration")) is None


def test_nc_credit_under_caption_is_stripped():
    item = licence_document("nc_credit_under_caption")
    text = lt.LicenceRecheck("licence_recheck").apply(item)["text"]
    assert "CC BY-NC" not in text and "Resist the temptation to cheat" in text


def test_long_consortium_byline_is_removed():
    text = lt.strip_page_header(document("long_consortium_byline"))
    assert text.startswith("2.1: What is a Theory?\nSociologists study social events")


def test_single_toc_entry_and_chapter_thumbnail_are_removed():
    item = {**document("single_toc_entry"), "metadata": {"title": SNIPPETS["single_toc_entry"]["title"]}}
    text = lt.strip_toc({**item, "text": lt.strip_attribution(lt.strip_page_header(item))})
    assert "12.1:" not in text and "Chapter thumbnail" not in text and "Stratigraphy is the study" in text


def test_headings_after_attribution_block_are_kept():
    text = lt.strip_attribution(SNIPPETS["headings_after_attribution_block"]["text"])
    assert text.startswith("The Human Organ Systems\nInformation\nOrgan systems are groups")


def test_bare_key_term_page_is_a_term_list():
    snippet = SNIPPETS["bare_key_terms"]
    assert lt.is_term_list({"id": snippet["id"], "text": snippet["text"], "metadata": {"title": snippet["title"]}})


def test_nc_source_footnote_drops_and_nc_image_footnote_is_stripped():
    recheck = lt.LicenceRecheck("licence_recheck")
    metadata = {"license": "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/", "title": "x"}
    body = "Attitudes are our evaluations or feelings toward a person, idea, or object and typically are positive or negative.\n"
    source = body + "[2] Principles of Social Psychology by University of Minnesota is licensed under a CC BY-NC-SA 4.0"
    image = body + "[1] Image by Muriel HEARD-COLLIER is licensed under CC BY-NC-ND 2.0"
    assert recheck.apply({"id": "x", "text": source, "source": "libretexts", "metadata": metadata}) is None
    assert recheck.apply({"id": "x", "text": image, "source": "libretexts", "metadata": metadata})["text"] == body.strip()


def test_verse_with_footnote_markers_keeps_its_lines():
    verse = SNIPPETS["verse_with_footnote_markers"]["text"]
    assert lt.join_fragments(verse) == verse


# Post-audit defects; each input is an excerpt of a real page (page path in the test name or the comment).
def test_mindtouch_subpage_property_blob_is_removed():
    text = ('Archaeology\n{ "6.01:_Abstract": "property get [Map MindTouch.Deki.Logic.ExtensionProcessorQueryProvider+<>'
            'c__DisplayClass230_0.<PageSubPageProperty>b__1]()" }\nBody')
    assert lt.strip_html_residue(text) == "Archaeology\nBody"


@pytest.mark.parametrize("text, expected", [
    # Cosumnes Math 372 8.01
    ('per second.\nrdquo;." data-type="media">rdquo;." data-media-type="image/jpg" src="/@api/deki/pages/=B/files/8bf" '
     'fileid="106213" />\nDirect Variation', "per second.\nDirect Variation"),
    # Mathematics for Game Developers (Burzynski)
    ('The result is 85.6647 degrees." src="/@api/deki/files/97446/clipboard_e3.png">', "The result is 85.6647 degrees."),
    # Intermediate Algebra 1e 10.03
    ('| 1\\)">Domain | \\((-\\infty, \\infty)\\) |', "| Domain | \\((-\\infty, \\infty)\\) |"),
    # Cosumnes Math 372 2.03
    ("as shown in Figure 5.\n<<em>Figure 5", "as shown in Figure 5.\nFigure 5"),
    # HTML shown in a lesson stays.
    ('Use <img src="cat.png" alt="A cat"> to show an image.', 'Use <img src="cat.png" alt="A cat"> to show an image.'),
])
def test_cut_html_attributes_are_removed(text, expected):
    assert lt.strip_html_residue(text) == expected


def test_lumen_colon_headed_attribution_block_is_removed():
    # Blueprint for Success in College and Career (Dillon) 1.01
    text = ("This textbook is a journey.\nLicenses and Attributions:\nContent previously copyrighted, published in Blueprint "
            "for Success in College (by Dave Dillon). Now licensed as CC BY: Attribution.\nCC licensed content, Shared "
            "previously:\nAdaptions: Reformatted and edited for a wider audience.\nKen Robinson: Do Schools Kill Creativity? "
            "Authored by TED.com\nAll rights reserved content:\nDon't Just Follow Your Passion. Authored by TEDxTalks")
    assert lt.strip_attribution(text) == "This textbook is a journey."
    # Blueprint (Dillon) 1.06 spells the heading "Previously shared".
    variant = "Soft skills matter.\nCC licensed content, Previously shared:\nAdaptions: Reformatted, some content edited."
    assert lt.strip_attribution(variant) == "Soft skills matter."


def test_espanol_host_page_needs_majority_english(language):
    text = SNIPPETS["english_lesson_spanish_grammar"]["text"]
    english_share = language.english_share(lt.prose_for_language_id(text))
    if not lt.ENGLISH_MIN_SHARE <= english_share < lt.ESPANOL_MIN_SHARE:
        pytest.skip("the snippet share is outside the band between the two thresholds")
    assert lt.is_english(language, {"id": "https://socialsci.libretexts.org/x", "text": text})
    assert not lt.is_english(language, {"id": "https://espanol.libretexts.org/x", "text": text})


def test_devanagari_lines_reach_language_id():
    assert lt.prose_for_language_id("हिन्दी\nक़ब्र, क़ानून") == "हिन्दी\nक़ब्र, क़ानून"


def test_bullets_emptied_by_truncated_urls_are_removed():
    # Prince Georges INT 1010 12.12.05
    text = "- ACM Code of Ethics 10/16/92.\n- http://www.copyright.gov/fls/fl102...\n- http://www.e.com/a...\n- From the US Patent"
    assert lt.repair_inline(text) == "- ACM Code of Ethics 10/16/92.\n- From the US Patent"


def test_lumen_shortcodes_are_removed():
    # Accounting for Managers (Lumen) 1.02
    text = "TRY It\n[ohm_question]146224[/ohm_question]\n[ohm_question]146225[/ohm_question]\nNow we will\n[/hidden-answer]"
    assert lt.repair_inline(text) == "TRY It\nNow we will"


def test_malformed_page_index_is_removed():
    # NWTC Team Building 8.04, UWMilwaukee CHE 125 13.E, Lumen Business 2.04
    assert lt.clean_latex("in Figure \\(\\PageIndex{Exhibit 14.5. The") == "in Figure Exhibit 14.5. The"
    assert "PageIndex" not in lt.clean_latex("Table \\PageIndex3\\PageIndex3.2 and \\(|PageIndex{1}\\)")


def test_navigation_and_template_lines_are_removed():
    # SUNY Potsdam Sedimentary Geology 4.02, UWMilwaukee CHE 125 09
    item = {"id": "x", "text": "Figure 4: Dune crest\nNext page »\nFigure 5: Photographs\nTemplate:HideTOC", "metadata": {}}
    assert lt.strip_page_header(item) == "Figure 4: Dune crest\nFigure 5: Photographs"


def test_licence_decision_runs_on_final_text():
    names = [step.name for step in lt.build_pipeline().steps]
    assert names.index("licence_final") == names.index("repair_inline") + 1


def test_glued_nonumber_and_lone_adaptions_note_are_removed():
    # CHEM 145 6.06 "\nonumber_{4}"; Blueprint (Dillon) 5.01 "Adaptions:" note without an attribution heading.
    assert "nonumber" not in lt.clean_latex(r"\ce{NO}(g)ΔH^\circ \nonumber_{4}=+90.2")
    text = "Soft skills matter.\nCC-BY: Attribution.\nAdaptions: Changed formatting, removed image, removed pre-test link."
    assert "Adaptions" not in lt.strip_attribution(text)
