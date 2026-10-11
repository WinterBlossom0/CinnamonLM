"""Checks of the oercommons cleaning rules on real pages from the source (data/real_pages.json)."""
import json
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.common.io import ShardWriter, read_documents
from data_pipeline.common.text import normalize_text
from data_pipeline.sources import oercommons as oer

PAGES = json.loads((Path(__file__).parent / "data" / "real_pages.json").read_text())


def page_text(name: str) -> str:
    return PAGES[name]["text"]


def cleaned(name: str) -> str:
    text = normalize_text(page_text(name))
    return oer.drop_repeated_blocks(oer.strip_boilerplate(oer.strip_page_scaffold(text)))


@pytest.fixture(scope="module")
def pipeline():
    return oer.build_pipeline()


def test_scaffold_keeps_title_and_drops_attachment_names():
    text = oer.strip_page_scaffold(page_text("attachment_preamble"))
    assert text.startswith("TDA - Text Dependent Analysis\nThis presentation is used")
    assert "TDA rubric breakdown blank" not in text
    assert "\nOverview\n" not in text


def test_attribution_section_is_removed_to_its_end():
    text = cleaned("boundless_attribution_tail")
    assert text == ("The standard interpretation in the People’s Republic of China presented the war as the beginning of "
                    "modern China and the emergence of the Chinese people’s resistance to imperialism and feudalism.")


def test_prose_about_attribution_is_kept():
    text = cleaned("attribution_lesson")
    assert "The Copyleft symbol by Zscout370 is in the Public Domain." not in text
    assert "Regardless of the Creative Commons license you choose for your OER" in text
    assert "This attribution generator from Open Washington can also be a useful tool." in text


def test_reference_urls_stay_and_canvas_notes_go():
    assert "Retrieved September 27, 2020, from\nhttp://www.jstor.org/stable/44430486" in cleaned("apa_references")
    links = cleaned("canvas_links")
    assert "Links to an external site." not in links
    assert "https://www.outofservice.com/bigfive/ and review your results" in links


def test_list_rule_drops_question_lists_but_not_prose_with_short_headings():
    assert oer.is_list_like({"text": cleaned("question_list")})
    assert not oer.is_list_like({"text": cleaned("openstax_introduction")})


@pytest.mark.parametrize("name, kept", [
    ("french_lesson_english_medium", True),
    ("english_lesson_spanish_medium", False),
    ("english_lesson_turkish_medium", False),
])
def test_english_threshold(pipeline, name, kept):
    document = {"id": PAGES[name]["id"], "text": page_text(name)}
    for step in pipeline.steps:
        if step.name == "exact_dedup":
            break
        document = step.apply(document)
        if document is None:
            break
    assert (document is not None) == kept


def test_pipeline_drops_chrome_and_copies(tmp_path, pipeline):
    data_dir, meta_dir = tmp_path / "data", tmp_path / "meta"
    data_dir.mkdir()
    lesson = {"id": PAGES["openstax_introduction"]["id"], "text": page_text("openstax_introduction"),
              "source": "oercommons", "metadata": {"license": "CC BY"}}
    chrome = {**lesson, "id": PAGES["login_chrome"]["id"], "text": page_text("login_chrome")}
    with ShardWriter(data_dir / "00000_oercommons.json.gz") as writer:
        for document in (lesson, chrome, lesson):
            writer.write(document)
    stats = pipeline.run_in_place(data_dir, meta_dir)
    output = list(read_documents(data_dir / "00000_oercommons.json.gz"))
    assert [document["id"] for document in output] == [lesson["id"]]
    assert output[0]["text"].startswith("Introduction\n- The War on Terror")
    assert "(Figure)" not in output[0]["text"]
    removed = {step["name"]: step["docs_removed"] for step in stats["steps"]}
    assert removed["login_chrome"] == 1
    assert removed["exact_dedup"] == 1


def test_repeated_answer_rows_without_leading_pipe_stay():
    text = cleaned("multiple_choice_table")
    assert text.count("ANSWER | A. Anterior cranial fossa and frontal sinus |") == 4
    assert text.count("| C. Maxillary sinus | D. Temporal fossa and middle cranial fossa |") == 4


def test_vendor_sentences_go_and_lesson_prose_stays():
    text = oer.strip_vendor_promo(page_text("off2class_lesson"))
    assert "off2class" not in text.lower()
    assert "Toward is used to show in the direction of." in text
    assert oer.has_min_words({"text": cleaned("off2class_lesson")})


def test_vendor_title_keeps_lesson_name():
    line = "Alphabet ( Lesson 2 ) - Off2Class ESL Lesson Plan"
    url_promo = ("To access 150+ free ESL lesson plans like this, sign up for a free Off2Class account: "
                 "https://app.off2class.com/auth/register!Objective: Introduce graphemes a_e, i_e, o_e.")
    assert oer.strip_vendor_promo(line) == "Alphabet ( Lesson 2 )"
    assert oer.strip_vendor_promo(url_promo) == "Objective: Introduce graphemes a_e, i_e, o_e."


def test_html_badge_markup_goes():
    text = cleaned("cc0_rdfa_badge")
    assert "<" not in text and "href=" not in text
    assert "has waived all copyright and related or neighboring rights to" in text
    assert oer.strip_html_markup("Update the src in the<img> tag") == "Update the src in the<img> tag"


def test_science_domain_standards_block_goes():
    assert cleaned("wyoming_standards").strip() == ""


def test_phone_after_room_number_is_masked():
    text = oer.mask_contact_details(page_text("advocacy_phone"))
    assert "530-7492" not in text
    assert "Room G19, <PHONE_NUMBER>, <EMAIL_ADDRESS>." in text
    table_row = "| Caregiver Resource Center Orange County | 714-446-5053 800-543-8312 | 130 W Bastanchury Rd |"
    assert oer.mask_contact_details(table_row).count("<PHONE_NUMBER>") == 2
    assert oer.mask_contact_details("ISBN 978-0-13-468599-1, catalogue 1978.004.0001") == (
        "ISBN 978-0-13-468599-1, catalogue 1978.004.0001")


def shingle_sets(*word_counts: tuple[int, int]) -> dict[int, np.ndarray]:
    """Document id -> shingle set; each document holds shingles start..stop-1 of one shared numbering."""
    return {document_id: np.arange(start, stop, dtype=np.uint64)
            for document_id, (start, stop) in enumerate(word_counts)}


def test_component_keeps_longest_and_checks_exact_jaccard():
    # Jaccard(0, 1) = 0.85 and Jaccard(1, 2) = 0.85, but Jaccard(0, 2) = 0.7225.
    shingles = shingle_sets((0, 7225), (0, 8500), (0, 10000))
    duplicate_of = oer.resolve_component([0, 1, 2], shingles.__getitem__, 0.8)
    assert duplicate_of == {1: 2}
    assert oer.exact_jaccard(shingles[0], shingles[2]) == pytest.approx(0.7225)


def test_near_dedup_keeps_the_longer_version(tmp_path, pipeline):
    data_dir, meta_dir = tmp_path / "data", tmp_path / "meta"
    data_dir.mkdir()
    base = page_text("french_lesson_english_medium")
    lines = base.split("\n")
    shorter = {"id": "short", "text": "\n".join(lines[:int(len(lines) * 0.9)]), "source": "oercommons",
               "metadata": {"license": "CC BY"}}
    longer = {**shorter, "id": "long", "text": base}
    with ShardWriter(data_dir / "00000_oercommons.json.gz") as writer:
        writer.write(shorter)
        writer.write(longer)
    pipeline.run_in_place(data_dir, meta_dir)
    assert [document["id"] for document in read_documents(data_dir / "00000_oercommons.json.gz")] == ["long"]


def test_known_spam_page_is_dropped(pipeline):
    spam = {"id": PAGES["spun_merchandise_spam"]["id"], "text": page_text("spun_merchandise_spam")}
    lesson = {"id": PAGES["openstax_introduction"]["id"], "text": page_text("openstax_introduction")}
    spam_filter = next(step for step in pipeline.steps if step.name == "spam_pages")
    assert spam_filter.apply(spam) is None
    assert spam_filter.apply(lesson) is lesson
