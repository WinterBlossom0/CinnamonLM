"""Checks of the pressbooks cleaning rules on real excerpts from the source."""
from data_pipeline.common.dedup import load_duplicate_of
from data_pipeline.sources import pressbooks as pb

# opentextbc.ca/abealf3/chapter/the-shooting-of-ginger-goodwin/ (raw), one flattened table.
FLATTENED_TABLE = (
    "Practice reading these words\n|\nlong /i/ = -igh |\nlong /a/ = -ain |\now = -ound |\n|\n"
    "fight light might night right sight tight |\ngain main pain rain brain drain grain train again plain explain |\n"
    "Watch out!"
)
# lmu.pressbooks.pub/introecon3e (raw): a header row, its separator, then a data row.
FLATTENED_WITH_SEPARATOR = (
    "|\n|\nValue of Exports |\nValue of Imports |\nBalance |\n|---|---|---|---|\n|\nGoods |\n$1,428.8 |\n$2,350.8 |\n"
    "–$922.0 |"
)
# library.achievingthedream.org/herkimerarthistory1 (raw): the paragraph pair is printed twice in a row.
DOUBLED = (
    "Medea Sarcophagus, 140–150 CE, marble, 65 × 227 cm (Altes Museum, Berlin)\n"
    "Dr. Beth Harris and Dr. Steven Zucker provide a description, historical perspective, and analysis of the Medea "
    "Sarcophagus.\n"
)
# louis.pressbooks.pub/introductorystatistics: the same exercise stem recurs far apart and must stay.
EXERCISE = "- In words, define the random variable X.\n- List the values that X may take on.\n- Give the distribution of X."


def clean(text: str) -> str:
    for transform in (pb.normalize_text, pb.strip_navigation, pb.strip_shortcodes, pb.rebuild_flattened_tables,
                      pb.strip_line_boilerplate, pb.drop_repeated_blocks, pb.strip_chapter_ordinal, pb.normalize_text):
        text = transform(text)
    return text


def test_flattened_rows_are_joined():
    assert clean(FLATTENED_TABLE).split("\n") == [
        "Practice reading these words",
        "| long /i/ = -igh | long /a/ = -ain | ow = -ound |",
        "| fight light might night right sight tight | gain main pain rain brain drain grain train again plain explain |",
        "Watch out!",
    ]
    assert clean(FLATTENED_WITH_SEPARATOR).split("\n") == [
        "| Value of Exports | Value of Imports | Balance |", "|---|---|---|---|", "| Goods | $1,428.8 | $2,350.8 | –$922.0 |",
    ]


def test_only_direct_repeats_are_removed():
    assert clean(DOUBLED + DOUBLED) == DOUBLED.strip()
    spaced = f"{EXERCISE}\nA store rents DVDs.\n{EXERCISE}"
    assert clean(spaced) == spaced
    refrain = "Lord, have mercy.\nLord, have mercy.\nChrist, have mercy."
    assert clean(refrain) == refrain


def test_chrome_licence_and_shortcodes_are_removed():
    raw = (
        "Primary Navigation\nWant to create or adapt books like this? Learn more about how Pressbooks supports open "
        "publishing practices.\nBook Contents Navigation\n1. Introduction\n2. Heat\nPrevious/next navigation\n"
        "Physics\n2 Heat Exchangers\n[latex size=”2″]Q = \\phantom{\\rule{0.25em}{0ex}}mc\\Delta T[/latex]\n"
        "A refractor bends light ([link]).\n-\nShow Answer\n"
        "Image by sebadelval is in the public domain.\n"
        "Media Attributions\n- 796px-MHS_47632_Astrolabe\n"
    )
    assert clean(raw) == "Physics\nHeat Exchangers\n$Q =  mc\\Delta T$\nA refractor bends light."


def test_quality_rules_ignore_table_syntax_and_placeholders_are_found():
    table = "\n".join(f"| Andromeda | Andromedae | Princess of Ethiopia | And | {row} | +40 |" for row in range(20))
    assert pb.quality_failure(table + "\nThe table lists the constellations and the names of their stars.") is None
    assert pb.PLACEHOLDER_TEXT.search("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    assert not pb.PLACEHOLDER_TEXT.search("The Science Post published a block of “lorem ipsum” text.")


# kpu.pressbooks.pub/ancientandmedievalworld/chapter/technological-advantages/ (raw): footnotes follow the media list.
MEDIA_THEN_FOOTNOTES = (
    "Media Attributions\n"
    "- Richard Marshal unhorses Baldwin Guines at a skirmish by Matthew Paris © Matthew Paris is licensed under a "
    "Public Domain license\n"
    "- The earliest known “treed” saddles were discovered in burial chambers at Pazyryk in southern Siberia, dated to "
    "the 5th century BCE. ↵\n"
    "- Hyland, The Medieval Warhorse, 51. ↵"
)
# ecampusontario.pressbooks.pub/facultyoertoolkit2/chapter/attribution-statements/ (raw): prose under the heading.
ATTRIBUTION_PROSE = (
    "Attributions\n"
    "Regardless of the Creative Commons license you choose for your OER, all CC licenses require at least an "
    "attribution to the original creator of the work (the “BY” part of a CC license).\n"
    "This attribution generator from Open Washington can also be a useful tool. Creative Commons also has advice "
    "about and examples of giving attribution."
)


def test_footnotes_and_licence_prose_survive_attribution_blocks():
    assert clean(MEDIA_THEN_FOOTNOTES).split("\n") == [
        "- The earliest known “treed” saddles were discovered in burial chambers at Pazyryk in southern Siberia, "
        "dated to the 5th century BCE.",
        "- Hyland, The Medieval Warhorse, 51.",
    ]
    assert clean(ATTRIBUTION_PROSE) == ATTRIBUTION_PROSE.split("\n", 1)[1]
    assert clean("Text\nAttributions\nSee the Attributions page near the end of this book.") == "Text"


def test_shortcode_examples_stay_and_footnotes_become_parenthetical():
    guide = "Use shortcodes like this:\n| This leads to a footnote[footnote]This is the footnote content. [/footnote]. |"
    assert pb.strip_shortcodes(guide) == guide
    assert pb.strip_shortcodes("a footnote[footnote]The content.[/footnote].") == "a footnote (The content.)."


def test_outline_repeats_stay_and_dangling_pointers_go():
    outline = "- Subdivision 1\n" + "- supporting detail\n- example 1\n- example 2\n" * 3 + "- Subdivision 2"
    assert clean(outline) == outline
    raw = ("Prose.\nReview the video below for a few suggestions on improving workplace communication.\n"
           "Podcast attribution: This podcast is © CBC/Radio-Canada 2021. All rights reserved.\nFeedback/Errata")
    assert clean(raw) == "Prose."


def test_course_code_is_not_a_phone_number():
    assert pb.mask_personal_data("| Delhi | 201809.1055 | Freshman |") == "| Delhi | 201809.1055 | Freshman |"
    assert pb.mask_personal_data("Call 604-555-1234.") == "Call <PHONE_NUMBER>."


def test_noncommercial_text_sources_are_found():
    assert pb.NONCOMMERCIAL_TEXT_SOURCE.search(
        "Text.\nBy Hill et al. (University of California, Davis), used under a CC-BY-NC-SA 4.0 international license.")
    assert pb.NONCOMMERCIAL_TEXT_SOURCE.search(
        "This chapter is adapted from Chapter 4 Opening eyes by Michelle Turner shared with a CC BY-NC license.")
    assert not pb.NONCOMMERCIAL_TEXT_SOURCE.search("This chapter is adapted from Psychology, licensed under CC BY 4.0.")


def test_chapter_inside_a_book_is_dropped(tmp_path):
    chapters = [" ".join(f"Chapter {number} sentence {word} of the narrative." for word in range(40))
                for number in range(3)]
    documents = [{"text": "\n".join(chapters)}, {"text": "Title\n" + chapters[1]}, {"text": "Other\n" + chapters[1][:150]}]
    deduplicator = pb.ContainmentDedup("containment_dedup")
    indexer = deduplicator.open_indexer(tmp_path, 0)
    for local_id, document in enumerate(documents):
        indexer.add(local_id, document)
    indexer.close()
    assert deduplicator.build(tmp_path, [len(documents)])["duplicates"] == 1
    assert list(load_duplicate_of(tmp_path)) == [-1, 0, -1]
