"""Checks of the wikimedia cleaning rules on real documents from the source (Hugging Face copies of the shards)."""
import gzip
import json
import time
from functools import lru_cache
from pathlib import Path

import pytest

from data_pipeline.common.io import encode_document, read_documents
from data_pipeline.common.langid import LanguageIdentifier
from data_pipeline.sources import wikimedia as wm

DATA_DIR = Path(__file__).parent / "data"
DOCUMENTS = json.loads((DATA_DIR / "real_documents.json").read_text(encoding="utf-8"))
# Excerpts of the documents named in the sample audit, from the frozen sample of data/common_pile/wikimedia.
AUDITED = json.loads((DATA_DIR / "audit_documents.json").read_text(encoding="utf-8"))
MAISIE_PAGES = [f"maisie_page_{number}" for number in range(105, 111)]


@lru_cache(maxsize=1)
def pipeline_steps() -> dict:
    return {step.name: step for step in wm.build_pipeline().steps}


def apply_steps(step_names: tuple[str, ...], document: dict) -> str | None:
    for step_name in step_names:
        document = pipeline_steps()[step_name].apply(document)
        if document is None:
            return None
    return document["text"]


def text_after(step_names: tuple[str, ...], name: str) -> str | None:
    return apply_steps(step_names, DOCUMENTS[name])


def audited_text_after(step_names: tuple[str, ...], name: str) -> str | None:
    return apply_steps(step_names, AUDITED[name])


CLEANING = ("normalize", "strip_markup", "clean_wiktionary", "repair_holes", "strip_interwiki_residue")


@pytest.fixture(scope="module")
def language() -> LanguageIdentifier:
    return LanguageIdentifier()


def test_talk_pages_are_dropped_and_article_and_scan_pages_kept():
    assert not wm.is_content_page(DOCUMENTS["wikipedia_user_talk"])
    assert wm.is_content_page(DOCUMENTS["wikipedia_disambiguation"])
    assert wm.is_content_page(DOCUMENTS["maisie_page_105"])


def test_raw_ocr_pages_are_recognized():
    assert wm.is_raw_ocr_page(DOCUMENTS["statutes_raw_ocr"])
    assert not any(wm.is_raw_ocr_page(DOCUMENTS[name]) for name in MAISIE_PAGES)


def test_page_numbers_and_running_header_residue_are_removed_when_pages_join():
    records = [wm.page_record(DOCUMENTS[name]) for name in MAISIE_PAGES]
    text = wm.join_pages(wm.strip_page_numbers([(number, body) for _, _, number, body in records]))
    assert "When he asked the pair how they liked the games they always replied" in text
    assert "Rh " not in text and " 94 " not in text and " 96 " not in text


def test_word_hyphenated_across_a_page_break_is_merged():
    records = [wm.page_record(DOCUMENTS[name]) for name in ("solo_page_20", "solo_page_21")]
    assert "appallingly old-fashioned clothes" in wm.join_pages([body for _, _, _, body in records])


def join_seconds(pages: list[str]) -> float:
    fastest = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        wm.join_pages(pages)
        fastest = min(fastest, time.perf_counter() - start)
    return fastest


def test_join_pages_time_grows_linearly_with_book_length():
    """Audit #1: join_pages searched the whole joined text for each page; 498 pages of Ruffhead took 48.5 s."""
    bodies = [wm.page_record(DOCUMENTS[name])[3] for name in MAISIE_PAGES]
    small, large = bodies * 150, bodies * 600
    assert join_seconds(large) < 8 * join_seconds(small) + 0.01
    assert join_seconds(large) < 1.0


def test_book_is_assembled_on_its_first_page_and_other_pages_are_dropped(tmp_path):
    data_dir, meta_dir = tmp_path / "data", tmp_path / "meta"
    data_dir.mkdir()
    names = MAISIE_PAGES + ["statutes_raw_ocr", "wikipedia_user_talk"]
    with gzip.open(data_dir / "00000_wikisource.com.jsonl.gz", "wb") as shard:
        for name in names:
            shard.write(encode_document(DOCUMENTS[name]))
    stats = wm.build_pipeline().run_in_place(data_dir, meta_dir)
    output = list(read_documents(data_dir / "00000_wikisource.com.jsonl.gz"))
    assert [document["id"] for document in output] == [DOCUMENTS["maisie_page_105"]["id"]]
    assert output[0]["text"].startswith("What Maisie Knew (Chicago & New York, Herbert S. Stone & Co., 1897)\n\n")
    assert "they liked the games they always replied" in output[0]["text"]
    steps = {step["name"]: step for step in stats["steps"]}
    assert steps["assemble_wikisource_books"]["docs_removed"] == len(MAISIE_PAGES) - 1
    assert steps["raw_ocr_page"]["docs_removed"] == 1
    assert not (meta_dir / "_work").exists()


def test_page_store_is_not_rebuilt_after_shards_were_rewritten(tmp_path):
    store_dir = tmp_path / "_work" / wm.BOOK_STORE_DIR
    store_dir.parent.mkdir()
    (store_dir.parent / "run.json").write_text("{}")
    with pytest.raises(RuntimeError):
        wm.WikimediaPipeline.build_store(tmp_path, store_dir, 1)


def test_wikitable_markup_is_removed_and_prose_kept():
    text = text_after(CLEANING, "wikipedia_wikitable")
    assert "{|" not in text and "class=" not in text and "\n * }" not in text
    assert "Textual corruption is probably the reason" in text


def test_html_entities_are_decoded():
    assert "— Russia's highest court" in text_after(CLEANING, "wikinews_entities")


def test_parentheses_emptied_by_dropped_pronunciation_templates_are_repaired():
    assert "Aeschylus (Αἰσχύλος; c. 525/524 – c. 456/455 BC)" in text_after(CLEANING, "wikipedia_empty_parens")


def test_wiktionary_skeleton_is_removed_and_definitions_kept():
    text = text_after(CLEANING, "wiktionary_twig")
    assert "Translations" not in text and "Abkhaz" not in text and "🇨🇬" not in text
    assert "From, , from" not in text
    assert text.count("Somebody, or one of his body parts, not looking developed.") == 1
    assert "A small thin branch of a tree or bush." in text


def test_repeated_wikivoyage_listings_are_kept_once():
    text = text_after(CLEANING, "voyage_duplicated_listings")
    assert text.count("Acqui Terme. A beautiful Roman town") == 1


def test_emptied_phrasebook_lines_and_nameless_listings_are_removed():
    phrasebook = text_after(CLEANING, "voyage_phrasebook")
    assert "(informal) : . ( )" not in phrasebook and "Hello." in phrasebook
    assert "— island with small population" not in text_after(CLEANING, "voyage_nameless_listings")


def test_disambiguation_page_is_dropped():
    assert text_after(CLEANING + ("min_words", "disambiguation"), "wikipedia_disambiguation") is None


def test_table_of_contents_is_dropped_and_recipe_is_kept():
    filters = CLEANING + ("min_words", "disambiguation", "generated_pages", "bullet_lists")
    assert text_after(filters, "wikibooks_toc") is None
    assert text_after(filters, "cookbook_recipe") is not None
    assert text_after(filters, "wikipedia_genus_with_species_list") is not None
    assert text_after(filters, "voyage_phrasebook") is not None


@pytest.mark.parametrize("name", ["wikiquote_foreign_quotes", "wikibooks_spanish_lesson", "maisie_page_106"])
def test_english_documents_with_foreign_text_are_kept(language, name):
    assert wm.is_english(language, text_after(CLEANING, name))


def test_interlingue_chapter_is_dropped(language):
    assert not wm.is_english(language, text_after(CLEANING, "interlingue_chapter"))


def test_personal_phone_is_masked_and_wikivoyage_business_phone_kept():
    assert "413-755-4906" not in wm.mask_personal_data(DOCUMENTS["wikinews_personal_phone"])
    business = DOCUMENTS["voyage_business_phone"]
    assert wm.mask_personal_data(business) == business["text"]


def test_wikibooks_navigation_and_breadcrumbs_are_removed():
    navigation = text_after(CLEANING, "wikibooks_navigation")
    assert "Return to the Contents" not in navigation and "Between the Wars (1920-1938) ›" not in navigation
    assert "16,000 Australian and New Zealand troops" in navigation
    assert "| Vegan cuisine | Rice" not in text_after(CLEANING, "cookbook_breadcrumb")


OCR_STEPS = ("normalize", "strip_markup", "damaged_ocr")


@pytest.mark.parametrize("name", ["ruffhead_statutes_vol4_pages_40_41", "catholic_encyclopedia_vol16_pages_13_14",
                                  "catholic_encyclopedia_vol2_pages_58_59"])
def test_unproofread_ocr_books_are_removed(name):
    """Audit #0: Ruffhead vol 4 (104-2828947), Catholic Encyclopedia vol 16 (104-2451142) and vol 2 (104-2451145)."""
    assert audited_text_after(OCR_STEPS, name) is None


@pytest.mark.parametrize("name", ["burmese_grammar_pages_20_31", "biloxi_dictionary_pages_11_22"])
def test_proofread_grammars_with_transliteration_marks_are_kept(name):
    """Audit #0 asks that proofread foreign-language grammars stay; these have 21 and 40 damaged-word shapes."""
    assert audited_text_after(OCR_STEPS, name) is not None


FILTERS = CLEANING + ("min_words", "disambiguation", "generated_pages", "bullet_list_prose", "bullet_lists",
                      "repetition", "low_prose")


def test_repeated_list_run_dedup_keeps_verse_attributions_and_equation_steps():
    """Audit #2: wikisource 0-66 (Richard II), 0-28 (A Letter to a Hindu), wikibooks 0-122; wikipedia 0-662."""
    verse = wm.body_of(AUDITED["richard_ii_verse"]["text"]).split("\n")
    assert wm.drop_repeated_list_runs(verse) == verse
    epigraphs = audited_text_after(CLEANING, "tolstoy_krishna_epigraphs")
    assert epigraphs.count(" * Krishna") == AUDITED["tolstoy_krishna_epigraphs"]["text"].count(" * Krishna")
    steps = audited_text_after(CLEANING, "wikibooks_align_steps")
    assert steps.count(" * $$\\begin{align}") == 4 and steps.count("$$") % 2 == 0
    references = audited_text_after(CLEANING, "wikipedia_repeated_reference_run")
    assert references.count("Apollo 11: As They Photographed It") == 1


def test_list_page_with_a_prose_stub_keeps_the_prose():
    """Audit #3: wikipedia 0-19340053 (Amitabh Bhattacharjee) and 0-28997248 (2005-06 New Jersey Nets season)."""
    biography = audited_text_after(FILTERS, "amitabh_bhattacharjee_filmography")
    assert biography is not None and "Indian actor who works in Bengali and Hindi films" in biography
    season = audited_text_after(FILTERS, "nets_season_table")
    assert season is not None and season.endswith("the eventual champions, the Miami Heat.")
    assert " * Vince Carter" not in season
    assert text_after(FILTERS, "wikibooks_toc") is None


def test_wiktionary_keeps_etymology_paragraphs_and_drops_emptied_sections():
    """Audit #4: wiktionary 0-19 (free) and 00001_wiktionary 0-765252 (analfabet)."""
    free = audited_text_after(CLEANING, "wiktionary_free")
    assert "\n\nGermanic and Celtic are the only" in free and "\n\nThe verb comes from" in free
    analfabet = audited_text_after(CLEANING, "wiktionary_analfabet")
    headings = [line for line in analfabet.split("\n") if wm.wiktionary_heading(line) is not None]
    assert analfabet.count("Etymology") == 1 and len(headings) == 7
    lines = analfabet.split("\n")
    for index, line in enumerate(lines):
        if wm.wiktionary_heading(line) is not None:
            assert index + 1 < len(lines) and wm.wiktionary_heading(lines[index + 1]) is None


def test_templates_keep_their_math_and_leave_no_debris():
    """Audit #5: wikibooks 0-122 (Signum), 00001_wikibooks 0-407971, wikisource 0-35, wiktionary 0-131."""
    signum = audited_text_after(("normalize", "strip_markup"), "wikibooks_signum_blockquote")
    assert "defined as$$\\mbox{sgn}(x)=" in signum and "{{" not in signum
    stevens = AUDITED["wikibooks_stevens_blockquote"]
    assert "remains weak in arithmetic COMMONWEALTH PUBLIC SERVICE." in wm.strip_markup(stevens["text"])
    # The audit saw this template without its "}}"; the cut reproduces that form from the same real line.
    unclosed = wm.strip_markup(stevens["text"].replace("1913. }}", "1913."))
    assert "{{" not in unclosed and "remains weak in arithmetic COMMONWEALTH PUBLIC SERVICE." in unclosed
    sonnets = audited_text_after(CLEANING, "shakespeare_sonnets_auxtoc")
    assert "{{" not in sonnets and "}}" not in sonnets and "/Sonnet 151/" in sonnets
    day = audited_text_after(CLEANING, "wiktionary_day_template_tail")
    assert "}}" not in day and "two days after tomorrow" not in day


def test_interwiki_residue_needs_a_title_shape(language):
    """Audit #6: wikisource 0-178358 (The Egoist), wikipedia 0-357391, wikisource 0-1042530; 0-3 is real residue."""
    egoist = wm.strip_interwiki_residue(language, AUDITED["egoist_chapter_heading"])
    assert "\n\nCOLONEL DE CRAYE\n" in egoist
    lega_nord = wm.strip_interwiki_residue(language, AUDITED["wikipedia_lega_nord_last_line"])
    assert lega_nord.endswith("Piemont Autonomista/Lega Nord Piemont")
    catalogue = AUDITED["belfast_species_catalogue"]
    assert wm.strip_interwiki_residue(language, catalogue) == catalogue["text"]
    dutch = wm.strip_interwiki_residue(language, AUDITED["wikisource_dutch_interwiki_title"])
    assert "Verklaring door Verenigde Naties" not in dutch


def test_small_wtf_wikipedia_junk_is_removed():
    """Audit #7: wikipedia 0-35111, 0-828240, 0-57361648; wikiquote 0-57, 0-167; wikivoyage 0-108; wikibooks 102-43."""
    year = audited_text_after(CLEANING, "wikipedia_year_843_calendar_note")
    assert "(link will display the full calendar)" not in year and "Year 843 (DCCCXLIII)" in year
    hamlet = audited_text_after(CLEANING, "wikiquote_hamlet_unpaired_italics")
    assert "''" not in hamlet and "Doubt thou the stars are fire;" in hamlet
    voyage = audited_text_after(CLEANING, "wikivoyage_mailto_link")
    abisko = next(line for line in voyage.split("\n") if "STF Abisko" in line)
    assert "mailto:" not in abisko and "[" not in abisko
    assert not wm.is_content_page(AUDITED["cookbook_policy_page"])
    assert "[https://" not in audited_text_after(CLEANING, "wikiquote_unclosed_link")
    ministers = audited_text_after(CLEANING, "wikipedia_ministers_key_label")
    assert ministers.endswith("public safety and emergency preparedness portfolio.")
    elicinae = audited_text_after(CLEANING, "elicinae_data_sources")
    assert elicinae.endswith("Data sources: i = ITIS, c = Catalogue of Life, g = GBIF, b = Bugguide.net")


def test_act_scene_line_citation_is_not_masked_as_an_ip_address():
    """Audit #8: 00001_wikibooks 0-408208 cites "Two Noble Kinsmen (3.4.19.24)"; wikivoyage 0-108 has a real e-mail."""
    kinsmen = AUDITED["two_noble_kinsmen_citation"]
    assert "Two Noble Kinsmen (3.4.19.24)" in wm.mask_personal_data(kinsmen)
    assert "<EMAIL_ADDRESS>" in wm.mask_personal_data(AUDITED["wikivoyage_mailto_link"])


@pytest.mark.parametrize("text, expected", [
    ("Argentina had attempts before Sergio Agüero}} scored the opening goal",  # 0-2845635
     "Argentina had attempts before Sergio Agüero scored the opening goal"),
    ("modifying for instance, the {w|neuroplasticity|plasticity}} of the brain",  # 0-2893319
     "modifying for instance, the plasticity of the brain"),
    ("Noosa defeated ''Toads Rugby Union Club|Toads]] 18 points to 13", "Noosa defeated Toads 18 points to 13"),  # 0-1460223
    ("told [[Agence France-Presse|AFP] ]he expected", "told AFP he expected"),  # 0-1523865
    ("under the Additional amendments to the United States Constitution#Amendment XV|Fifteenth Amendment]]]]]], of",
     "under the Additional amendments to the Fifteenth Amendment, of"),
])
def test_markup_residue_keeps_text(text, expected):
    assert wm.strip_markup_residue(text) == expected


def test_markup_residue_leaves_unmasked_latex_alone():
    # 0-41033: an unpaired $$ leaves this formula outside the math mask.
    line = r"\frac{Y(z)}{X(z)} = \frac{1 + 2z^{-1} +z^{-2}} {1 +\frac{1}{4} z^{-1}} $$"
    assert wm.strip_markup_residue(line) == line


@pytest.mark.parametrize("line", [
    " * Bilander, [b`il[e]nd[e]], [b`a`il[e]nd[e]], bijlander.",  # 104-3487967 Dutch phonetic dictionary
    " * Character data example: <![CDATA[ John Smith ]]>",  # 0-134064
])
def test_markup_residue_keeps_balanced_brackets(line):
    assert wm.strip_markup_residue(line) == line


def test_markup_residue_keeps_nested_list_code():
    # 0-239947: OpenSCAD source with nested lists.
    line = "polyhedron(points = [[x, y, z], ... ], triangles = [[p1, p2, p3..], ... ], convexity = N);"
    assert wm.strip_markup_residue(line) == line


@pytest.mark.parametrize("text, expected", [
    ("File:Burung Nuri Biru Kuning.jpg|Two true parrots in the branches", "Two true parrots in the branches"),  # 0-201246
    ("Image:V_For_Chanology_Hawaii.JPG Image:HPD_Chanology_1.JPG\nNext line", "Next line"),  # 0-99084
    ("Added: proportional]] [[File:01 Siebzehneck-1818-Animation.gif|535px|thumb|center|Construction according to",
     "Added: proportional]] Construction according to"),  # 0-196890
    ("Aconitum uncinatum File:Aconitum uncinatum illustration.jpg pollen nectar",
     "Aconitum uncinatum pollen nectar"),  # 0-411511
    ("for the station wagon body design.<ref From 1950 through 1991, the trim",
     "for the station wagon body design. From 1950 through 1991, the trim"),  # 0-168037
    (" * a fictional pair who commented on the letters page from 1935 to 1990.<ref",
     " * a fictional pair who commented on the letters page from 1935 to 1990."),  # 0-235688
    ("Text.\nCategory:Living people\nMore text.", "Text.\nMore text."),
])
def test_file_ref_residue_removed(text, expected):
    assert wm.strip_file_ref_residue(text) == expected


@pytest.mark.parametrize("line", [
    " * http://commons.wikimedia.org/wiki/File:Wikipedia_video_tutorial-2-Reliability-en.ogv",  # 104-1546845
    "Eilun, n. a copy; a resemblance; an image: an idol",  # 104-1706740
    "FILE: IntegerStack.i3",  # 0-241545
    'Inject it with <ref bean="dataSource"/> in the context file.',
    "Not marked on the above image:",  # 0-167796
    'Category:"Why have you taken up arms? . . . All this can\'t be any of your business."',  # 104-1369694
])
def test_file_ref_residue_keeps_text(line):
    assert wm.strip_file_ref_residue(line) == line
