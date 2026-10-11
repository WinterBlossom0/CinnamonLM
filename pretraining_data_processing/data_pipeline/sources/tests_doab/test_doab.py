"""Checks of the doab cleaning rules on real excerpts from the source shards."""
import json
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.common.dedup import load_duplicate_of
from data_pipeline.sources import doab

SNIPPETS = json.loads((Path(__file__).parent / "data" / "real_snippets.json").read_text(encoding="utf-8"))
CLEANING_STEPS = ("normalize", "page_furniture", "back_matter", "markup", "symbol_font", "ligatures", "dehyphenate",
                  "rejoin_paragraphs", "repeated_paragraphs", "renormalize")
ENGLISH_SECTIONS = [name for name in SNIPPETS if name.startswith("english_section_")]


@pytest.fixture(scope="module")
def steps() -> dict:
    return {step.name: step for step in doab.build_pipeline().steps}


def document(name: str) -> dict:
    snippet = SNIPPETS[name]
    return {"id": snippet["id"], "text": snippet["text"], "metadata": {"title": snippet["title"]}}


def apply_steps(steps: dict, item: dict, names: tuple[str, ...]) -> dict | None:
    for name in names:
        item = steps[name].apply(item)
        if item is None:
            return None
    return item


def test_pipeline_order_cleans_then_filters_then_dedups_then_masks(steps):
    names = list(steps)
    assert names.index("renormalize") < names.index("garbled_font") < names.index("min_words")
    assert names.index("min_words") < names.index("content") < names.index("english") < names.index("exact_dedup")
    assert names.index("exact_dedup") < names.index("near_dedup") < names.index("containment_dedup")
    assert names[-2:] == ["pii", "licence_credits"]


def test_intech_author_block_licence_and_reference_list_are_removed(steps):
    text = apply_steps(steps, document("intech_back_matter"), CLEANING_STEPS)["text"]
    for removed in ("Author details", "Address all correspondence", "Licensee IntechOpen", "Department of Nephrology",
                    "[10] Konner K.", "**47**"):
        assert removed not in text


def test_intech_back_matter_fragment_is_dropped_by_quality_filters(steps):
    assert apply_steps(steps, document("intech_back_matter"), CLEANING_STEPS + ("min_words", "content")) is None


def test_split_ligatures_are_joined():
    text = doab.repair_ligatures(SNIPPETS["mdpi_split_ligature"]["text"])
    assert "interfacial effect and induction effect" in text
    assert "e ffect" not in text and "e fficiency" not in text


def test_repeated_paragraph_is_kept_once(steps):
    text = SNIPPETS["intech_repeated_paragraphs"]["text"]
    marker = "Chapter six shows how in perimenopausal period"
    assert text.count(marker) == 2
    item = apply_steps(steps, document("intech_repeated_paragraphs"), CLEANING_STEPS)
    assert item["text"].count(marker) == 1


def test_section_copied_inside_a_longer_section_is_marked_as_contained(steps, tmp_path):
    deduplicator = doab.ContainmentDedup()
    indexer = deduplicator.open_indexer(tmp_path, 0)
    for local_id, name in enumerate(["chapter_section_copy", "chapter_section_container"]):
        indexer.add(local_id, apply_steps(steps, document(name), CLEANING_STEPS))
    indexer.close()
    summary = deduplicator.build(tmp_path, [2])
    assert summary["duplicates"] == 1
    assert np.asarray(load_duplicate_of(tmp_path)).tolist() == [1, -1]


def test_containment_prefers_the_longer_document_and_skips_dropped_containers():
    keys = np.array([1, 2, 1, 2], dtype=np.uint64)
    doc_ids = np.array([0, 0, 1, 1])
    chars = np.array([400, 400, 400, 400])
    lengths = np.array([5000, 900])
    duplicate_of, _ = doab.contained_documents(keys, doc_ids, chars, lengths, np.ones(2, dtype=bool), 0.8)
    assert duplicate_of.tolist() == [-1, 0]
    duplicate_of, _ = doab.contained_documents(keys, doc_ids, chars, lengths, np.array([False, True]), 0.8)
    assert duplicate_of.tolist() == [-1, -1]


def test_enzyme_numbers_are_not_masked_as_ip_addresses(steps):
    item = document("enzyme_ec_numbers")
    assert "EC 1.15.1.1" in steps["pii"].apply(item)["text"]


def test_e_mail_is_masked(steps):
    item = {"id": "x", "text": "Contact the author at someone.name@example.org for data.", "metadata": {}}
    assert "<EMAIL_ADDRESS>" in steps["pii"].apply(item)["text"]


@pytest.mark.parametrize("name,line", [
    ("laboratory_heading", "#### Laboratory Processing"),
    ("wrapped_prose_line", "School board policies determine the eligibility for bus"),
    ("nmr_methods_line", "1H-NMR-spectra were measured on a MercuryPlus 400"),
])
def test_headings_and_prose_lines_are_kept(steps, name, line):
    text = apply_steps(steps, document(name), ("normalize", "page_furniture", "back_matter"))["text"]
    assert line in text


def test_running_header_with_full_book_title_is_removed(steps):
    text = apply_steps(steps, document("title_prefix_running_header"), ("normalize", "page_furniture"))["text"]
    assert "Reverse Engineering – Recent Advances and Applications" not in text
    assert "## **2. Review of related methods**" in text


@pytest.mark.parametrize("line", ["p. 613", "pp. 19-23", "ch7", "1968. pp. 1-31"])
def test_page_locator_fragments_are_reference_lines(line):
    assert doab.is_reference_line(line)


@pytest.fixture(scope="module")
def profiler() -> doab.EnglishProfiler:
    return doab.EnglishProfiler()


@pytest.mark.parametrize("name,expected", [
    ("german_grammar_in_english", True),
    ("english_with_italian_quotes", True),
    ("math_section", True),
    ("bilingual_edition", False),
    ("italian_chapter", False),
    ("english_book_with_italian_chapter", False),
    ("tibetan_transliteration", False),
])
def test_english_decision_on_real_books(steps, profiler, name, expected):
    text = apply_steps(steps, document(name), CLEANING_STEPS)["text"]
    assert doab.is_english_document(profiler(text)) is expected


def test_foreign_chapter_is_caught_by_block_share_not_by_english_share(steps, profiler):
    profile = profiler(apply_steps(steps, document("english_book_with_italian_chapter"), CLEANING_STEPS)["text"])
    assert profile.english_share >= doab.ENGLISH_MIN_SHARE
    assert profile.foreign_block_share >= doab.MAX_FOREIGN_BLOCK_SHARE


def test_math_section_passes_content_filter_and_title_page_does_not(steps):
    assert apply_steps(steps, document("math_section"), CLEANING_STEPS + ("min_words", "content")) is not None
    assert apply_steps(steps, document("title_and_byline"), CLEANING_STEPS + ("min_words", "content")) is None


# Bylines from 004acd65-0f8a-4342-8761-8da04dfc73fe.5, b0ccb31b-3482-4603-97dc-f394189fe4e9.46 and
# 9cbfcd3f-...27.11 that a marker match with the separating comma merged into one name.
@pytest.mark.parametrize("line", [
    "**Monica Rosa Loizzo 1, Paolo Lucci 2,\\*, Oscar Núñez 3, Rosa Tundis 1, Michele Balzano 4, Natale Giuseppe Frega 4,"
    " Lanfranco Conte 2, Sabrina Moret 2, Daria Filatova 3, Encarnación Moyano <sup>3</sup> and Deborah Pacetti <sup>4</sup>**",
    "**Aiah A. El-Rashidy 1,2, Sara El Moshy 2,3, Israa Ahmed Radwan 2,3, Dina Rady 2,3, Marwa M. S. Abbass 2,3 , Christof E."
    " Dörfer <sup>4</sup> and Karim M. Fawzy El-Sayed 2,4,5,\\***",
    "Angelica Staniloiu1,2,3, Irina Vitcu3 and Hans J. Markowitsch1",
    "*Eliete A. Alvin, Anna V.B. e Borges, Rhéltheer de P. Martins, Marcela R. Lemes and Anielle C.A. Silva*",
    "**Chiara Galber 1,2 , Stefania Carissimi 1 , Alessandra Baracca <sup>2</sup> and Valentina Giorgio 1,2, \\***",
    "*Edited by Metin Turan and Ertan Yildirim*",
])
def test_bylines_and_editor_credits_are_credit_lines(line):
    assert doab.is_credit_line(line, doab.bare(line))


@pytest.mark.parametrize("line", ["**Extra Virgin Olive Oil Quality, Safety, and Authenticity**",
                                  "Editors often disagree about this, and the authors know it",
                                  "Austria, Belgium, France and Germany joined the treaty in 1995"])
def test_titles_and_prose_are_not_credit_lines(line):
    assert not doab.is_credit_line(line, doab.bare(line))


def test_latex_matrix_is_not_a_repeated_phrase(steps):
    line = "$$D = \\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 1 & 0 & 0 \\end{bmatrix}$$"
    item = {"id": "x", "text": "Intro text.\n\n" + line, "metadata": {"title": ""}}
    assert line in steps["page_furniture"].apply(item)["text"]


@pytest.mark.parametrize("line", [
    "Wieviorka, Michel. 2001. La violencia: Destrucción y constitución del sujeto. *Espacio abierto* 10: 337–47.",
    "Silva, F.das C. & Soares-Silva, L.H. (2000). Arboreal flora of the Godoy Forest State Park,",
    "Kang S, Lucky A, Pariser D, et al. Long-term safety. *J Am Acad Dermatol.* 2001;44:S58-S64. Klein P, Clark R. "
    "An evidence-based review. *Arch Dermatol.* 1999;135:1522–1525.",
])
def test_author_date_and_glued_reference_entries_are_reference_lines(line):
    assert doab.is_reference_line(line)


def test_byline_ending_in_a_marker_is_removed(steps):
    line = "**Pamela Dalton 1,\\*, Anna-Sara Claeson <sup>2</sup> and Steve Horenziak <sup>3</sup>**"
    item = {"id": "x", "text": "Intro text.\n\n" + line, "metadata": {"title": ""}}
    assert "Pamela Dalton" not in steps["page_furniture"].apply(item)["text"]


# Prose with glued stamps; source ids are in the "glued_running_stamps" snippet.
def test_glued_running_stamps_go_and_the_prose_stays(steps):
    text = apply_steps(steps, document("glued_running_stamps"), ("normalize", "page_furniture", "back_matter"))["text"]
    for stamp in ("PEER REVIEW", "Will-be-set-by-IN-TECH", "10.5772", "*Foods* **2020**", "Perspective Chapter",
                  "The Role of Vascular Smooth Muscle Cells", "Noncoding RNAs in Myelodysplastic Syndromes"):
        assert stamp not in text
    for prose in ("Finally, since the results by a single nested cross validation can vary",
                  "Self-resolution has the same positive and negative premise and needs to",
                  "who are obliged to add grades for each of the students after the exam period.",
                  "early investigators of atherogenesis compared the proliferation",
                  "miRNA-based drugs are designed to"):
        assert prose in text


# Prose from 8955d7f7, 62dd9f37, 13e19fd0, 27f760a3, 66163b8f, 61c19e6e, d86a6cf0 and 4e576316 (ids in the snippet).
@pytest.mark.parametrize("line", SNIPPETS["prose_with_citations"]["text"].split("\n"))
def test_prose_with_citations_is_not_a_reference_line(line):
    assert not doab.is_reference_line(doab.bare(line))


# Fragments from 140adf61-6969-47d6-ae8b-751388d7cf92.110 and its book.
@pytest.mark.parametrize("line", [
    "Molgedey L. & Schuster, H. G. (1994). Separation of a mixture of independent signals using time delayed",
    "prevent the spread of varicella in children in hospital. *Lancet*, Vol. 2, No. 7892,",
    "pp. (1600-1605), 1022-1336.",
    "Berger, Johannes (1996) 'Was behauptet die Modernisierungstheorie wirklich?', *Leviathan*",
])
def test_reference_fragments_and_name_lists_are_reference_lines(line):
    assert doab.is_reference_line(line)


def test_split_reference_list_is_dropped(steps):
    item = apply_steps(steps, document("reference_fragments"), CLEANING_STEPS + ("min_words", "content"))
    assert item is None


# Contents page of 27f1e5af-41de-44b0-9820-6f7558745353.5.
def test_contents_entries_after_bold_lines_are_removed(steps):
    text = apply_steps(steps, document("contents_page"), ("normalize", "page_furniture", "back_matter"))["text"]
    assert "Introductory Chapter: Modern Bridges" not in text
    assert "Wind Action Phenomena Associated with Large-Span Bridges" not in text
    assert "Dr. Hamid Yaghoubi is the director of Iran Maglev Technology" in text


# 280d7b39-48b6-4d98-b3b2-6bec3ed8db6a.62 and 27c922ec-d9f1-4f9e-b5f4-b360faba3c5b.5.
@pytest.mark.parametrize("name,heading", [("numbered_heading_before_tail", "4.1.1 Antiplatelet agents"),
                                          ("running_title_before_prose", "Analysis of Chemical Processes, Determination")])
def test_unmarked_headings_are_not_joined_to_the_next_paragraph(steps, name, heading):
    text = apply_steps(steps, document(name), CLEANING_STEPS)["text"]
    paragraph = next(paragraph for paragraph in text.split("\n\n") if heading in paragraph)
    assert doab.bare(paragraph) == heading


def test_page_break_inside_a_sentence_is_joined():
    assert "The clinical phenotype of MADD is highly heterogeneous" in doab.rejoin_paragraphs(
        document("page_break_in_sentence"))


# c4ab1c11-f4cc-4aea-835e-492adb60d14b: input "<sup>p</sup> <sup>&</sup>lt; 0.05".
def test_split_html_entity_is_unescaped():
    assert doab.clean_markup("(\\*<sup>p</sup> <sup>&</sup>lt; 0.05)") == "(*p < 0.05)"


# 753012d5-d292-406d-9e63-d6b52ed43a01.21: "atr <sup>¼</sup> <sup>V</sup>\\_ <sup>t</sup> cos ... � <sup>q</sup>".
def test_symbol_font_operators_in_superscript_become_plain_operators(steps):
    text = apply_steps(steps, document("symbol_font_equations"), ("normalize", "markup", "symbol_font"))["text"]
    assert "⁼" not in text and "¼" not in text and "þ" not in text
    assert "atr = V_ t cos φt − q" in text


# Adverts of 13e19fd0-36bb-4dab-ae5b-e9be9a63ae06.2 and 003dfb30-3d3b-432a-9c4e-217966fc6cb3.
def test_intech_advert_is_removed(steps):
    text = apply_steps(steps, document("intech_advert"), ("normalize", "page_furniture"))["text"]
    assert doab.bare(text).split() == ["Meet", "the", "editors"]


# 4e5d9eb3-4bc4-47c1-8838-2bcfa6d28737.76, 0004847e-ab80-435d-9a35-50f7f1b13b00.6, d8bf0471-94b2-4291-81a9-4d42d19775b5.9
# and d8bbffcd-a666-4f4a-aa54-e43e0b406294.9.
@pytest.mark.parametrize("line", [
    "Figure 4 (B) illustrates the details of the ith self-feedback neuron (e.g. ith gene in the GRN), where the sum",
    "Figure 7 A and B, repeated-measures ANOVA revealed significant overall differences between untreated groups",
    "Figure 14 (1) is the contrast diagram of the equivalent load force of every moment of the symmetry plane and the "
    "equivalent load of the left side.",
    "Table 12 (El-Sherif et al., 2003; El-Sherif, 2006). The stability constants of the Pd(AMBI)- CBDCA complex, tend "
    "to decrease while increasing the chloride ion concentration.",
])
def test_prose_opening_with_a_figure_reference_is_not_a_caption(line):
    assert not doab.is_caption(line)


@pytest.mark.parametrize("line", ["Table 2.1 Summary of adiabatic invariants", "Fig. 3.2 The employee income response "
                                  "process in OHS 1997 and 1998", "Figure 3. Map of the study area"])
def test_captions_are_captions(line):
    assert doab.is_caption(line)


# The 13 English sections that the language filter dropped (ids in the snippet names and fields).
@pytest.mark.parametrize("name", ENGLISH_SECTIONS)
def test_english_sections_with_foreign_names_or_references_are_english(steps, profiler, name):
    text = apply_steps(steps, document(name), CLEANING_STEPS)["text"]
    assert doab.is_english_document(profiler(text))


@pytest.mark.parametrize("line", [
    # 27bcc842-3de5-4781-ba4a-d8ce1386a50f: an MDPI byline that Marker formats as a heading.
    "## *Catalina Alvarado-Rojas 1,2, Katia Lehongre1,2, Juliana Bagdasaryan1,2, Anatol Bragin3, Richard Staba3, "
    "Jerome Engel Jr. 3, Vincent Navarro1,2,4 and Michel Le Van Quyen1,2\\**",
    # 141ac00a-9004-4900-b231-9f41bc362067.2: "†" after a comma.
    "**Jan van Hooft 1,\\*,†, Guido Kielenstijn 1,\\*,†, Jeroen Liebregts 2, Frank Baan 3, Gert Meijer 2, "
    "Jan D'haese 1, Ewald Bronkhorst <sup>1</sup> and Luc Verhamme <sup>3</sup>**",
    # 4e50f609-e7a4-4318-9e22-ea71aef11372.4: split Polish letters.
    "**Leszek Szalewski 1, Magdalena Szalewska 1,2,\\*, Paweł Jarosz 3, Michał Wo´s <sup>4</sup> and "
    "Jolanta Szyma ´nska <sup>1</sup>**",
    # 89590caf-4e16-4fe0-8d53-1ef0b7d1ffe5.7: one one-word name.
    "Dong-qing Wang1 , Hai-tao Zhu1 , Yan-fang Liu2 , Rui-gen Yin1 , Liang Zhao1 , Zhi-jian Zhang3 , Zhao-liang Su3 , "
    "Yan-Zhu1 , Hui-qun Lu1 , Juan Hong1 and Jie Zhang4",
    # 00064f14-4c6c-4123-bed8-b5ae765fbfc8.3: affiliations glued to the names.
    "S. Abdullah1, T. E. Putra2 and M. Z. Nuawi1 *1Universiti Kebangsaan Malaysia 2Universitas Syiah Kuala 1Malaysia "
    "2Indonesia*",
])
def test_marked_heading_bylines_and_bylines_with_affiliations_are_credit_lines(line):
    assert doab.is_credit_line(line, doab.bare(line))


@pytest.mark.parametrize("line", ["## Karl Marx, Friedrich Engels and Vladimir Lenin",
                                  "Science, Technology Studies, Public Policy and Social Change",
                                  "Results are given in Table 1 *1Note on the data*"])
def test_unmarked_heading_and_title_lists_are_not_credit_lines(line):
    assert not doab.is_credit_line(line, doab.bare(line))


def test_bare_intech_doi_stamp_is_removed(steps):
    # 9ccdb135-80ca-45b5-909b-ded99ed40790.9: the page stamp "10.5772/53606" between the byline and the first heading.
    text = "Stéphane Le Ménec and Antonios Tsourdos\n\n10.5772/53606\n\n**1. Introduction**"
    item = {"id": "9ccdb135-80ca-45b5-909b-ded99ed40790.9", "text": text, "metadata": {"title": ""}}
    cleaned = steps["page_furniture"].apply(item)["text"]
    assert "10.5772" not in cleaned and "**1. Introduction**" in cleaned


@pytest.mark.parametrize("line", [
    # 27c922ec-d9f1-4f9e-b5f4-b360faba3c5b.7
    "We called the software PCFenzyme. An old version of the executable PCFenzyme can be downloaded at "
    "http://dx.doi.org/10.1016/j.clinbiochem.2008.11.016. The latest version of the executable PCFenzyme with new "
    "methods included is available upon request by e-mail.",
    # 612cec61-82fb-48c1-bb5c-5571ac6bdafd
    "All datasets generated for this study are included in the manuscript and/or the supplementary files. More Data is "
    "available at https://doi.org/10.7488/ds/2497.",
])
def test_sentences_with_a_doi_link_are_not_reference_lines(line):
    assert not doab.is_reference_line(doab.bare(line))


@pytest.mark.parametrize("line", [
    "disease and Alzheimer disease. Neurology 66, 529–534. doi: 10.1212/01.wnl. 0000198511.09968.b3",  # 0025fd58-dc3d
    "P. Szczepanik et al. (eds.), *Digital Peripheries*, Springer Series in Media Industries, "
    "https://doi.org/10.1007/978-3-030-44850-9\\_10",  # c4b7809d-b20c-43d0-a227-db3ff2356116
    "URL: *http://dx.doi.org/10.1002/andp.19083300302*",  # 9cbfcd3f-0075-497a-acc8-7c9d6a426317.16
])
def test_reference_tails_with_a_doi_are_reference_lines(line):
    assert doab.is_reference_line(doab.bare(line))


def test_late_contents_page_with_chapter_entries_is_removed(steps):
    text = apply_steps(steps, document("late_contents_page"), ("normalize", "page_furniture", "back_matter"))["text"]
    for removed in ("Chapter 16", "Shahla Mohammad Ganji", "Chapter 17", "M. Rita I. Young and David D. Walker",
                    "Implications for Squamous Cell Carcinoma 253"):
        assert removed not in text
    assert "He published a paper in Nature in 2003" in text


def test_second_line_of_a_running_head_is_not_joined(steps):
    text = apply_steps(steps, document("two_line_running_head"), CLEANING_STEPS)["text"]
    heads = [paragraph for paragraph in text.split("\n\n") if "Dissociated Strabismus Subjects" in paragraph]
    assert heads == ["from Normal and Dissociated Strabismus Subjects Through Morphological Transformations"] * 2


def test_sentence_split_before_a_short_tail_is_joined(steps):
    text = apply_steps(steps, document("sentence_split_before_heading"), CLEANING_STEPS)["text"]
    assert "there are at least two sample points per line width." in text


def test_peer_review_stamp_with_multiplication_sign_is_removed():
    # 7943991e-5999-4fba-8f0d-7251d9fb1231.53 writes the draft stamp with "×".
    line = "isolated by preparative LC–MS *J. Fungi* **2021**, *7*, × FOR PEER REVIEW 5 of 10"
    assert "PEER REVIEW" not in doab.strip_running_stamps(line)


def test_caption_credit_is_stripped_by_the_final_licence_step():
    # 39d280a5-da04-432b-8597-5c47a27a455d: a rejoined caption keeps a figure permission credit.
    item = {"id": "39d280a5", "source": "doab",
            "metadata": {"license": "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"},
            "text": "further investigation. Originally published in Ziros et al. (16); used here with permission from "
                    "Mary Ann Liebert Inc."}
    text = doab.LicenceCreditStrip("licence_credits").apply(item)["text"]
    assert "with permission" not in text
