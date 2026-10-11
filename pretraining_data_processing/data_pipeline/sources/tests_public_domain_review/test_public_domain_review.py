"""Checks of the public_domain_review cleaning rules on real excerpts from the source."""
import json

from data_pipeline.common.dedup import ExactDedup
from data_pipeline.common.io import ShardWriter, read_documents
from data_pipeline.common.pipeline import Pipeline
from data_pipeline.sources import public_domain_review as pdr

FOOTER = "The text of this essay is published under a CC BY-SA license, see here for details."
KISS_2011 = (
    "The Kiss (1896)\n\nAug 13, 2011\n\n\n\n\"An osculatory performance by May Irwin and John Rice\".... Scene from the "
    "New York stage comedy, The Widow Jones, in which Irwin and Rice starred. According to Edison film historian C. "
    "Musser, the actors staged their kiss for the camera at the request of the New York world newspaper, and the "
    "resulting film was the most popular Edison Vitascope film in 1896. The first ever kiss to be caught on film."
)
KISS_2013 = (
    "The Kiss (1896)\n\nJul 9, 2013\n\n\n\nAn animated GIF excerpted by Okkult Motion Pictures from a 19th century "
    "Edison film. \"An osculatory performance by May Irwin and John Rice\".... Scene from the New York stage comedy, "
    "The Widow Jones, in which Irwin and Rice starred. According to Edison film historian C. Musser, the actors staged "
    "their kiss for the camera at the request of the New York world newspaper, and the resulting film was the most "
    "popular Edison Vitascope film in 1896. The first ever kiss to be caught on film.\n*\nOkkult Motion Pictures is "
    "the brainchild of Marco Calabrese and Alessandro Scali from Turin, Italy. With the Excerpts project, Okkult "
    "Motion Pictures aims to bring to light the most interesting and unusual out-of-copyright moving images occulted "
    "in Internet archives, through a series of animated gifs. A digital archivalism project for the diffusion of open "
    "knowledge. Okkult Motion Pictures official website / Twitter\n*\nAll Okkult animated GIFs published here under a "
    "CC-BY-SA license."
)
STUB = "Trapeze Disrobing Act (1901)\n\nAug 21, 2011\n\n\n\nA naughty little skit from 1901 filmed by the Edison company."


def document(doc_id: str, text: str, doc_type: str = "collection") -> dict:
    return {"id": doc_id, "text": text, "source": "public-domain-review", "type": doc_type}


def write_shard(path, documents: list[dict]) -> None:
    with ShardWriter(path) as writer:
        for item in documents:
            writer.write(item)


def test_header_lines_become_consecutive():
    text = "An Archaeology of Surf\n\nBy Melissa McCarthy\nMay 19, 2021\n\n“All plots tend to move deathward.”"
    assert pdr.normalize_header(text) == (
        "An Archaeology of Surf\nBy Melissa McCarthy\nMay 19, 2021\n\n“All plots tend to move deathward.”")


def test_nbsp_runs_before_verse_lines_become_line_breaks_but_ledger_columns_stay():
    verse = "To such concerning frogs we don’t give ear,\xa0\xa0\xa0\xa0\xa0\xa0Though well described by Pliny;"
    assert pdr.restore_nbsp_line_breaks(verse) == "To such concerning frogs we don’t give ear,\nThough well described by Pliny;"
    ledger = "evaluate at" + "\xa0" * 16 + ". . ." + "\xa0" * 8 + "3000 francs"
    assert pdr.restore_nbsp_line_breaks(ledger) == ledger


def test_joined_verse_lines_are_split_but_abbreviations_stay():
    assert pdr.restore_joined_lines("Ev’ry lady in this landHas twenty nails upon each handFive & twenty") == (
        "Ev’ry lady in this land\nHas twenty nails upon each hand\nFive & twenty")
    assert pdr.restore_joined_lines("as silly.But when we talk") == "as silly.\nBut when we talk"
    assert pdr.restore_joined_lines("with a good capon lin’d,With eyes severe") == "with a good capon lin’d,\nWith eyes severe"
    assert pdr.restore_joined_lines("Who said that love was fire?I know that love is ash.") == (
        "Who said that love was fire?\nI know that love is ash.")
    assert pdr.restore_joined_lines("Great auk\nPinguinus impennis1852\nClassed as") == (
        "Great auk\nPinguinus impennis\n1852\nClassed as")
    assert pdr.restore_joined_lines("I gave the Nuutti drinkI put the Nuutti in the corner") == (
        "I gave the Nuutti drink\nI put the Nuutti in the corner")
    assert pdr.restore_joined_lines("Can be the mementoOf the dead,Then the tombstone") == (
        "Can be the memento\nOf the dead,\nThen the tombstone")
    assert pdr.restore_joined_lines("The fruit batA mother's pregnant bellyAn outrigger canoe") == (
        "The fruit bat\nA mother's pregnant belly\nAn outrigger canoe")
    assert pdr.restore_joined_lines("what World offers is vainOh let us Heaven gain!") == (
        "what World offers is vain\nOh let us Heaven gain!")
    for kept in ("grapples with the deranged Dr.Caligari in a world", "etchings held by the Mu.Zee in Ostend.",
                 "recorded October 20/9, 1920 on matrix 7596-B"):
        assert pdr.restore_joined_lines(kept) == kept


def test_joins_after_capitalized_words_and_years_are_split_but_compound_names_stay():
    assert pdr.restore_joined_lines("by Virginia WoolfThe Prophet by Kahlil GibranThe Ego") == (
        "by Virginia Woolf\nThe Prophet by Kahlil Gibran\nThe Ego")
    assert pdr.restore_joined_lines("AurochsBos primigenius") == "Aurochs\nBos primigenius"
    assert pdr.restore_joined_lines("taking Laughing Gas in 1799From pages 497") == (
        "taking Laughing Gas in 1799\nFrom pages 497")
    kept = ("John MacGregor, Edward FitzGerald, MacFlecknoe, YouTube, WikiCommons, WorldCat, BioLib, BibliOdyssey, "
            "RawPixel, TransManche, GutsMuths, CapNap, the HansetsuRoku, the ForeBodings of the Creator")
    assert pdr.restore_joined_lines(kept) == kept


def test_footnote_markers_are_removed_in_essays_only():
    step = pdr.FootnoteMarkers("footnote_markers")
    essay = ("New York, Biblion Soc., 1927; for private circulation.1\nAnd beneath it. "
             "The difficulty lies in the psychological domain.—Sigfried Giedion1\n"
             "a “venal and corrupt master” — entirely.24 That did not hold. The ferry’s wake. Fm7 arpeggios. "
             "See vol. 2 and p. 12 of the 1923 edition.")
    cleaned = step.apply(document("essay-id", essay, "essay"))["text"]
    assert cleaned == ("New York, Biblion Soc., 1927; for private circulation.\nAnd beneath it. "
                       "The difficulty lies in the psychological domain.—Sigfried Giedion\n"
                       "a “venal and corrupt master” — entirely. That did not hold. The ferry’s wake. Fm7 arpeggios. "
                       "See vol. 2 and p. 12 of the 1923 edition.")
    after_punctuation = ("homage to Queen Lili’uokalani of Hawai’i.6 Barrett publicized. But if I fail what a lesson!!27\n"
                         "The same camels . . . . ”9 Melville was unusual. He may break his neck——.3\n")
    assert step.apply(document("essay-id", after_punctuation, "essay"))["text"] == (
        "homage to Queen Lili’uokalani of Hawai’i. Barrett publicized. But if I fail what a lesson!!\n"
        "The same camels . . . . ” Melville was unusual. He may break his neck——.\n")
    clues = "24. With enough beer and bromide and borax25 & 30. To fill a crustacean’s thorax"
    assert step.apply(document("essay-id", clues, "essay"))["text"] == clues
    before_punctuation = ("resemble Bede’s computus15; the hand-based techniques. is my soul.1—Heinrich Heine\n"
                          "Elsewhere1, I have emphasised. “Self-Reliance” (1840)12, thus. Guy Miège6—rather. "
                          "Woolley.1– one brand-new")
    assert step.apply(document("essay-id", before_punctuation, "essay"))["text"] == (
        "resemble Bede’s computus; the hand-based techniques. is my soul.—Heinrich Heine\n"
        "Elsewhere, I have emphasised. “Self-Reliance” (1840), thus. Guy Miège—rather. Woolley.– one brand-new")
    for kept in ("than are expressed in Maps” (Baldwin:41, also 81–2).", "a child in ’48, I had", "the “Law of ’84”.",
                 "in 1943 or '44 it didn't"):
        assert step.apply(document("essay-id", kept, "essay"))["text"] == kept
    assert step.apply(document("essay-id", "never be debtors?’14\nThe homme", "essay"))["text"] == (
        "never be debtors?’\nThe homme")
    collection = document("collection-id", essay, "collection")
    assert step.apply(collection) is collection


def test_boilerplate_lines_and_link_sentences_are_removed_and_quoted_speech_stays():
    text = ("Dotted throughout France are quiet villages.\n"
            "See more past visions of the future here, including additional works by Robida.\n"
            "“Look at that cable car coming — and the man going across the street. See here comes a truck.”\n"
            "———- Forwarded message ———-From: Ivars Skrastins <i****** @du.lv >To: Justin Smith "
            "<j*******@*mail.com >Cc:Bcc:Date: Fri, 6 Nov 2016 16:01:27 +0200Subject: KANT’S SUMATRA VOYAGE\n"
            "MP3 Download / Internet Archive Link\n" + FOOTER)
    assert pdr.remove_boilerplate(text) == (
        "Dotted throughout France are quiet villages.\n"
        "“Look at that cable car coming — and the man going across the street. See here comes a truck.”\n"
        "———- Forwarded message ———-\nSubject: KANT’S SUMATRA VOYAGE")


def test_link_shop_comment_and_widget_sentences_are_removed_and_content_stays():
    text = ("(A transcript below via Wikisource. You can also see the original handwritten manuscript of the account "
            "here at Wellcome Images)\n"
            "A useful overview, from 1908, of the history of the Bills of Mortality can be read here, and another from "
            "1843, here. And from a more poetic response.\n"
            "(HTML version of the books can be found here)\n"
            "Viewers will recognize the distinctive style from other Méliès' films, which can be found here.\n"
            "A complete copy can be seen at the Bibliothèque nationale de France, but we are not featuring them here.\n"
            "Lantern slides. Learn more about Lantern Slides here.\n"
            "Plates below. We've also made a few prints of the images which you can see in our shop here.\n"
            "We are also selling a print of the quilt in our online shop (though the signatures are small!).\n"
            "We are unsure what the flaming \"artichokes\" are all about, if any one has a clue do let us know in the "
            "comments!\n"
            "Pictures from Flickr: The Commons, click on images for more info and source.\n"
            "You can browse the whole book in the embedded reader above and find our highlights below. For more see X.\n"
            "Below is a little bit more about each of their lives (with each name linking through to their respective "
            "Wikipedia pages, if you would like to find out more). In the new year links shall be included.\n"
            "And a few others that didn't make it to the class photo....\n"
            "Some people you think we've missed? Please let us know in the comments!\n"
            "Some great extra reading on Newton's prophecy can be found here on the isaac-newton.org website")
    assert pdr.remove_boilerplate(text) == (
        "(A transcript below via Wikisource.)\n"
        "And from a more poetic response.\n"
        "Viewers will recognize the distinctive style from other Méliès' films, which can be found here.\n"
        "A complete copy can be seen at the Bibliothèque nationale de France, but we are not featuring them here.\n"
        "Lantern slides.\n"
        "Plates below.\n"
        "For more see X.\n"
        "In the new year links shall be included.")


def test_okkult_credit_block_and_its_separators_are_removed():
    cleaned = pdr.remove_boilerplate(pdr.normalize_text(KISS_2013))
    assert cleaned.endswith("The first ever kiss to be caught on film.")
    assert "Okkult Motion Pictures is" not in cleaned


def test_index_widget_entries_are_removed():
    text = ("presenting them. ※※Indexed under…Jewellerydesign's influence on blackwork\nIn addition to functioning\n"
            "Cackling farts — eggs< ※※Indexed under…EggSlang for\nCaptain queernabs — shabby ill-dressed fellow")
    assert pdr.remove_boilerplate(text) == ("presenting them.\nIn addition to functioning\n"
                                            "Cackling farts — eggs\nCaptain queernabs — shabby ill-dressed fellow")


def test_artist_biography_template_is_removed():
    text = ("He listed 53 emotions that could be classified in terms of muscular action.\n*\n"
            "Bill Domonkos is an experimental filmmaker and video artist. His work combines computer animation, still "
            "photography, live footage and manipulated archive film footage.")
    assert pdr.remove_boilerplate(text) == "He listed 53 emotions that could be classified in terms of muscular action."


def test_stub_filter_counts_body_characters_only():
    stub = pdr.normalize_header(pdr.normalize_text(STUB))
    caption = ("Short future history novel from John Ames Mitchell (1845–1918). First published in 1889, it is the "
               "fictional journal of Persian admiral Khan-Li, who in the year 2951 rediscovers North America by sailing across the Atlantic.")
    assert len(pdr.split_header(stub)[1]) < pdr.MIN_BODY_CHARS
    assert len(pdr.split_header(f"The Last American (1889)\nAug 19, 2011\n\n{caption}")[1]) >= pdr.MIN_BODY_CHARS


def test_markup_keeps_anchor_text_and_drops_urls():
    text = ('You can see the versions <a href="http://jonathanswiftarchive.org.uk/search" target="_blank">here at the '
            'Archive. While well-intentioned, [scholars such as Katherine Ott](https://books.google.fi/books?id=c) '
            'point out. Read it here: https://publicdomainreview.org/2013/03/07/the-redemption-of-saint-anthony/')
    assert pdr.remove_markup(text) == ("You can see the versions here at the Archive. While well-intentioned, "
                                       "scholars such as Katherine Ott point out. Read it here:")


def test_contained_repost_is_dropped_and_the_larger_copy_kept(tmp_path):
    data_dir = tmp_path / "data"
    other = document("other", "Bartholomaeus Anglicus wrote an encyclopedia of nineteen books on the properties "
                              "of things, from God and the angels down to colours, smells and tastes.")
    write_shard(data_dir / "a.jsonl.gz", [document("kiss-2011", KISS_2011), other])
    write_shard(data_dir / "b.jsonl.gz", [document("kiss-2013", KISS_2013)])
    Pipeline("containment", [ExactDedup(), pdr.ContainmentDedup()]).run_in_place(data_dir, tmp_path / "meta")
    kept = [item["id"] for shard in ("a", "b") for item in read_documents(data_dir / f"{shard}.jsonl.gz")]
    assert kept == ["other", "kiss-2013"]


def test_repost_with_a_new_title_and_date_is_dropped(tmp_path):
    data_dir = tmp_path / "data"
    body = ("A book of caricatures of famous men of the day, including the likes of Darwin, Swinburne, Tennyson and "
            "Browning, drawn by cartoonist Frederick Waddy and accompanied by biographical sketches of each of them.")
    first = document("men-of-the-day", f"Cartoon Portraits of Men of the Day (1873)\nFeb 2, 2012\n\n{body}")
    repost = document("leading-figures", "Cartoon Portraits of Leading 19th-Century Figures (1873)\nSep 30, 2014\n\n"
                                         f"A selection of the more well known of the figures. {body}")
    write_shard(data_dir / "a.jsonl.gz", [first, repost])
    Pipeline("containment", [pdr.ContainmentDedup()]).run_in_place(data_dir, tmp_path / "meta")
    assert [item["id"] for item in read_documents(data_dir / "a.jsonl.gz")] == ["leading-figures"]


def test_full_pipeline_on_real_documents(tmp_path):
    data_dir = tmp_path / "data"
    essay_body = ". ".join(["The Encyclopedia of Light is an essay on how writing hides its false starts"] * 8)
    essay = f"The Encyclopedia of Light\n\nBy Peter Schmidt\nApril 13, 2022\n\n{essay_body}.3\n{FOOTER}"
    write_shard(data_dir / "v0" / "00000_essays.jsonl.gz", [document("light", essay, "essay")])
    write_shard(data_dir / "v0" / "00000_collections.jsonl.gz", [document("trapeze", STUB)])
    stats = pdr.build_pipeline().run_in_place(data_dir, tmp_path / "meta")
    documents = list(read_documents(data_dir / "v0" / "00000_essays.jsonl.gz"))
    assert documents[0]["text"] == f"The Encyclopedia of Light\nBy Peter Schmidt\nApril 13, 2022\n\n{essay_body}."
    assert list(read_documents(data_dir / "v0" / "00000_collections.jsonl.gz")) == []
    assert {step["name"]: step["docs_removed"] for step in stats["steps"]}["stub"] == 1
    assert json.loads((tmp_path / "meta" / "stats.json").read_text())["output"]["docs"] == 1
