"""Check classify() on full real Common Pile v0.1 records addressed by (shard, ordinal, sha1); exit code 1 on failure.

The records live in test_docs.jsonl.gz. `--refresh --raw ROOT...` copies missing records from the first ROOT that holds the shard
(data/common_pile is valid only for sources that are not rewritten in place; a fresh Hugging Face download is valid for all).
"""
import argparse
import collections
import gzip
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import licence_filter as lf

RAW = os.path.join(os.path.dirname(HERE), "data", "common_pile")
FIXTURES = os.path.join(HERE, "test_docs.jsonl.gz")
OER = "oercommons/00000_oercommons.json.gz"
LT = "libretexts/00000_libretext.json.gz"
PB = "pressbooks/00000_pressbooks.json.gz"
PB1 = "pressbooks/00001_pressbooks.json.gz"
DOAB5 = "doab/00005_doab.json.gz"
WN = "wikimedia/00000_wikinews.com.jsonl.gz"
WN1 = "wikimedia/00001_wikinews.com.jsonl.gz"
WV = "wikimedia/00000_wikiversity.com.jsonl.gz"
WVOY = "wikimedia/00000_wikivoyage.com.jsonl.gz"
PG_DOCS = "project_gutenberg/v0/documents/"
PG = "project_gutenberg/v0/documents/00003_pg.jsonl.gz"
WQ = "wikimedia/00000_wikiquote.com.jsonl.gz"
WB = "wikimedia/00000_wikibooks.com.jsonl.gz"
WS = "wikimedia/00000_wikisource.com.jsonl.gz"
# Shards rewritten by the 4c6acc6b licence pass; their records are in the fixtures, so --refresh needs a root that maps these names.
OER_L = "licensed/oercommons/00000_oercommons.json.gz"
LT_L = "licensed/libretexts/input_before_inplace.json.gz"
PB_L = "licensed/pressbooks/00000_pressbooks.json.gz"
LT_R3 = "licensed_r3/libretexts/00000_libretext.json.gz"
LT_R4 = "licensed_r4/libretexts/00000_libretext.json.gz"
PB_R5 = "licensed_r5/pressbooks/00000_pressbooks.json.gz"
NOT_DROP = ("keep", "strip")

Case = collections.namedtuple("Case", "name shard ordinal sha1 actions reason removed kept", defaults=(None, (), ()))

CASES = [
    # Defect 1: media words in context lines must not turn a text-level NC declaration into a strip.
    Case("d1_lumen_umn_text_nc", OER, 1819, "dbc80d01c0", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_own_title_nc", OER, 2260, "efb17f3eea", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_own_title_nc_derivative", OER, 7858, "5002128fff", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_remix_title_nc", OER, 1422, "ffe7b92b46", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_text_credit_nc", OER, 3777, "20903144e0", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_ck12_plate_tectonics_context", LT, 37177, "33444c95f2", ("drop",), "work_level_nc_nd_declaration"),
    Case("d1_google_docs_licence_teaching", LT, 2776, "3757626774", NOT_DROP, kept=("The least permissive type of license is a CC BY-NC-ND",)),
    # Defect 2: declaration forms without "is" or with the strong subject after the licence.
    Case("d2_licensed_under_except_noted", LT, 298, "fa8b4c8147", ("drop",), "work_level_nc_nd_declaration"),
    Case("d2_content_section_from", PB, 38439, "b33d5319e7", ("drop",), "work_level_nc_nd_declaration"),
    Case("d2_under_license_quoted", OER, 5313, "4893e70e49", ("drop",), "work_level_nc_nd_declaration"),
    Case("d2_text_credit", OER, 4216, "32375b93ec", ("drop",), "work_level_nc_nd_declaration"),
    # Defect 3: licence-teaching lines and non-copyright "licensed" are not declarations.
    Case("d3_dmca_cc_teaching", LT, 5953, "9fe2447715", NOT_DROP,
         kept=("CC-BY-NC: This license is the same as CC-BY", "CC-BY-NC-ND: This license is the same as CC-BY-NC")),
    Case("d3_remix_commercial_teaching", LT, 10884, "226a226ed7", NOT_DROP, kept=("If you want to use a remix for commercial purposes",)),
    Case("d3_resource_states_teaching", OER, 5691, "cf942a7e99", NOT_DROP, kept=("The resource clearly states that content",)),
    Case("d3_broadcast_licensing", WB, 2440, "40db36b89b", NOT_DROP, kept=("Commercial broadcasting is licensed only on channels",)),
    # Defect 4: strip only the credit clause; keep discussion paragraphs and glued body text.
    Case("d4_six_flavors_discussion", PB, 2591, "57ce9ede3d", NOT_DROP, kept=("licenses come in six flavors",)),
    Case("d4_copyright_us_discussion", PB, 2594, "ee95d04c65", NOT_DROP, kept=("Copyright in the U.S. is automatically assigned",)),
    Case("d4_durand_glued_body", PB, 61620, "e9c71d3488", ("strip",), removed=("Source: Flickr. CC BY-NC-ND 2.0",),
         kept=("As a communications professional you must also conduct",)),
    Case("d4_vital_signs_glued_body", LT, 2426, "da4571b963", ("strip",), removed=("Image by knowablemag is licensed under CC BY-NC-ND 2.0",),
         kept=("transition right from taking a p",)),
    # Defect 5: figure and table captions with a permission credit are stripped, not dropped.
    Case("d5_table_reproduced", "doab/00005_doab.json.gz", 14789, "585941aa42", ("strip",),
         removed=("Reproduced with permission from [38]; published by Elsevier, 2013",)),
    Case("d5_wrapped_caption", "doab/00000_doab.json.gz", 217, "5872bc1f26", ("strip",),
         removed=("Reprinted with permission from [26], 2000, Elsevier Ltd.",)),
    Case("d5_fig_dot_caption", "doab/00000_doab.json.gz", 3734, "62977b47dc", ("strip",), removed=("Reprinted with permission from Gavial",)),
    Case("d5_screenshots", PB, 41891, "b4aa94b226", ("strip",), removed=("Screenshots used with permission",)),
    # Defect 6: printed-source notices in Project Gutenberg books are not eBook licences.
    Case("d6_pg_arr_1921", PG, 19, "d972577006", ("keep",)),
    Case("d6_pg_arr_no_year", PG, 33, "ace9e78035", ("keep",)),
    Case("d6_pg_arr_1891", PG, 112, "b78c17a339", ("keep",)),
    Case("d6_pg_permission_1898", PG, 1448, "a36d48caaa", ("keep",)),
    Case("d6_pg_permission_1903", PG, 1954, "a34b240433", ("keep",)),
    Case("d6_pg_permission_kipling", PG, 1924, "c30f1d77f5", ("keep",)),
    Case("d6_pg_now_public_domain", PG, 789, "bec182e345", ("keep",)),
    Case("d6_pg_world_library", PG, 792, "1926b283e2", ("drop",), "pg_restricted_ebook_licence"),
    # Defect 7: Wikiquote film, TV and music pages.
    Case("d7_tv_title", WQ, 1248, "fb2c5c0544", ("drop",), "wikiquote_film_tv_music_title"),
    Case("d7_film_title", WQ, 1528, "72c20d9beb", ("drop",), "wikiquote_film_tv_music_title"),
    Case("d7_band_title", WQ, 869, "a30553ae1b", ("drop",), "wikiquote_film_tv_music_title"),
    Case("d7_band_song_lyrics", WQ, 926, "42dafdfe7c", ("drop",), "wikiquote_song_lyrics_section"),
    Case("d7_musician_song_lyrics", WQ, 1213, "64f02a1ba4", ("drop",), "wikiquote_song_lyrics_section"),
    Case("d7_person_quotes", WQ, 14695, "8205b84d2f", ("keep",)),
    # Defect 8: Wikimedia talk pages, program output, licence texts, government and pre-1931 sources.
    Case("d8_compiler_output", WB, 6002, "a175de8448", ("keep",)),
    Case("d8_user_talk", WB, 50612, "0e2986ab3d", ("keep",)),
    Case("d8_cookbook_namespace_is_content", WB, 6360, "8fc7282c88", ("drop",), "third_party_text_used_with_permission"),
    Case("d8_bsd_licence_text", WS, 19711, "cad365c6cc", ("keep",)),
    Case("d8_us_government", WS, 20630, "d29896e2ba", ("keep",)),
    Case("d8_roman_year_1901", WS, 23996, "d38f556c3c", ("keep",)),
    Case("d8_permission_1901", WS, 424, "e779cf056b", ("keep",)),
    Case("d8_help_talk_pg", WS, 18807, "0ba707d446", ("keep",)),
    # Defect 9: standards notices, resource lists, quiz options and templates.
    Case("d9_common_core_dropped_before", OER, 1844, "2bfc34ad6f", ("keep",)),
    Case("d9_common_core_stripped_before", OER, 4348, "df374b0818", NOT_DROP, kept=("Common Core State Standards © Copyright 2010",)),
    Case("d9_resource_list", OER, 6676, "3c52735c5c", NOT_DROP),
    Case("d9_resource_sets", OER, 1055, "84bc28e894", NOT_DROP),
    Case("d9_quiz_option", OER, 7118, "37ba219301", NOT_DROP),
    Case("d9_template", OER, 1415, "945be8e58e", NOT_DROP),
    Case("d9_discipline_guide", PB, 15433, "b11098af5f", NOT_DROP),
    # Defect 10: restriction phrasings.
    Case("d10_sourcebook_restriction", OER, 1698, "c9f281c157", ("drop",), "restricted_use_notice"),
    Case("d10_ccby_nc_no_space", OER, 327, "803272705e", ("strip",), removed=("licensed under CCBY-NC-SA 2.0",)),
    Case("d10_reused_with_permission", OER, 7801, "92e4a8b317", ("strip",), removed=("Reused with permission",)),
    Case("d10_copyright_permission", "doab/00001_doab.json.gz", 13159, "91a5afc1cc", ("strip",),
         removed=("reproduced with copyright permission from Reference [14]",)),
    # Defects found while fixing the above.
    Case("x_by_nd_yag_is_not_cc", "doab/00007_doab.json.gz", 12376, "6d26dee0aa", NOT_DROP, kept=("A - When sound dentin is irradiated by Nd",)),
    Case("x_data_availability_nc", "doab/00000_doab.json.gz", 10446, "8f1e44f0b0", NOT_DROP),
    Case("x_pg_not_used_commercially_prose", "project_gutenberg/v0/documents/00002_pg.jsonl.gz", 263, "5d9ab80a79", ("keep",)),
    Case("x_pg_personal_use_only_prose", "project_gutenberg/v0/documents/00012_pg.jsonl.gz", 2056, "a25fdffca3", ("keep",)),
    # Lines of the earlier test, now in their full documents.
    Case("old_work_nc", OER, 365, "ae03d90dd7", ("drop",), "work_level_nc_nd_declaration"),
    Case("old_except_noted", OER, 1962, "9ca076bfd6", ("drop",), "work_level_nc_nd_declaration"),
    Case("old_used_with_permission", OER, 2745, "13ebee9828", ("drop",), "third_party_text_used_with_permission"),
    Case("old_should_not_be_altered", LT, 2997, "24b627fb4c", ("drop",), "third_party_text_used_with_permission"),
    Case("old_bare_nc_line_is_image_credit", "doab/00001_doab.json.gz", 18123, "7739088561", ("strip",),
         removed=("Courtesy of Digital Bodleian *CC BY-NC 4.0*",)),
    Case("old_image_attribution", OER, 139, "b426c2b46f", ("strip",), removed=("Image Attribution: Two by B*lly Frank",)),
    Case("old_title_image", OER, 1097, "e9391109eb", ("strip",), removed=("Si-MOCs is licensed under CC BY-NC-SA 2.0",)),
    Case("old_flickr_tasl", OER, 325, "4ce1697d1d", ("strip",), removed=("by Ronin Helsing is licensed under CC BY-ND 2.0",)),
    Case("old_youtube_arr", LT, 1849, "d3e4fff679", ("strip",), removed=("Standard YouTube License",)),
    Case("old_frontiers_compilation", "doab/00000_doab.json.gz", 3554, "0992bf78c1", NOT_DROP),
    Case("old_both_uses", "doab/00000_doab.json.gz", 15128, "53266daa67", NOT_DROP),
    Case("old_non_commercial_cooperation", "doab/00000_doab.json.gz", 10995, "71d857866c", NOT_DROP),
    # Round 2, defect 1: bare NC licence URLs under "Adapted from:" are text-source licences, not caption fragments.
    Case("r2_d1_adapted_from_url", OER, 1247, "fd97856f36", ("drop",), "work_level_nc_nd_declaration"),
    Case("r2_d1_section_adapted_from_urls", OER, 5269, "47951f99c1", ("drop",), "work_level_nc_nd_declaration"),
    Case("r2_d1_adapted_from_url_list", OER, 1672, "78987f2785", ("drop",), "work_level_nc_nd_declaration"),
    Case("r2_d1_parenthesised_lumen_item", PB, 5271, "5fbd00b37e", ("drop",), "work_level_nc_nd_declaration"),
    # Defect 2: media words inside a text-source title, and a talk credit that closes a full transcript.
    Case("r2_d2_plate_tectonics_title", LT, 37248, "f06318d7ff", ("drop",), "work_level_nc_nd_declaration"),
    Case("r2_d2_maps_title", LT, 37181, "0d88cfb641", ("drop",), "work_level_nc_nd_declaration"),
    Case("r2_d2_ted_transcript", PB, 72286, "56ba768df9", ("drop",), "work_level_nc_nd_declaration"),
    # Defect 3: permission phrases with words between the verb and "with permission", and "with permission of <publisher>".
    Case("r2_d3_permission_of_springer", PB1, 31583, "f1eeb9d3e6", ("drop",), "third_party_text_used_with_permission"),
    Case("r2_d3_stories_included_with_permission", OER, 5149, "00dbfb063e", ("drop",), "third_party_text_used_with_permission"),
    # Defect 4: CC deed-title licence lines and "CC NC SA".
    Case("r2_d4_deed_title_line", OER, 9139, "edfacbee18", ("drop",), "bare_nc_nd_licence_line"),
    Case("r2_d4_cc_nc_sa", PB, 72844, "11d1ea3e25", ("drop",), "bare_nc_nd_licence_line"),
    # Defect 5: song lyrics without Verse/Chorus markers.
    Case("r2_d5_lyrics_without_markers", OER, 7483, "058ee1b825", ("drop",), "predominantly_song_lyrics"),
    Case("r2_d5_poetry_anthology_is_not_lyrics", PB1, 3317, "a4d8707d64", NOT_DROP),
    # Defect 6: licence-teaching pages keep their lists, table rows, examples and headings.
    Case("r2_d6_six_licences_list", PB, 3043, "6afb773999", ("keep",)),
    Case("r2_d6_arr_typically", PB, 7559, "c7dcd6dddf", NOT_DROP, kept=("Works with copyright typically are ‘all rights reserved’",)),
    Case("r2_d6_table_row", PB, 8636, "b6a4c6bb75", NOT_DROP, kept=("| Free online resources under all rights reserved copyright |",)),
    Case("r2_d6_arr_headings", PB, 9882, "137d10e483", NOT_DROP, kept=("Permission to Include All Rights Reserved Content in an Open Resource",)),
    Case("r2_d6_fair_use_guidelines", PB, 27498, "140ce45c7c", NOT_DROP, kept=("- Pages from consumable materials may not be copied",)),
    Case("r2_d6_licence_table_cells", PB, 6099, "e7f4a3dbf2", NOT_DROP),
    Case("r2_d6_sample_notice", PB, 5214, "291480ba92", NOT_DROP),
    Case("r2_d6_open_license_types", LT, 10892, "00f757808e", NOT_DROP),
    Case("r2_d6_example_lines", OER, 2294, "5a43f42fbb", NOT_DROP,
         kept=("Example: CK-12 Biology for High School by CK-12 is licensed under a Creative Commons Attribution-NonCommercial 3.0 License.",)),
    # Defect 7: DOAB figure captions longer than 300 chars and journal-citation permission credits.
    Case("r2_d7_caption_et_al_copyright", DOAB5, 3, "e39cc765c6", ("strip",),
         removed=("Reproduced with permission (Breger et al., 2015a), Copyright American Chemical Society 2015.",)),
    Case("r2_d7_panel_label_caption", DOAB5, 11836, "678c07482b", ("strip",), removed=("OE: overexpression; KD: knockdown; Data are adapted with permission from the publication (Yokoyama, N., Golebiewska",)),
    Case("r2_d7_bold_panel_caption", DOAB5, 18457, "3a3086c4e3", ("strip",), removed=("Reprinted with permission from Macmillan Publishers Ltd.",)),
    Case("r2_d7_journal_citation_credit", DOAB5, 27616, "19f74b2378", ("strip",)),
    Case("r2_d7_aip_copyright_caption", DOAB5, 33512, "998bc42371", ("strip",), removed=("(MacKenzie et al., 2010). Reprinted with permission",)),
    Case("r2_d7_ref_before_permission", DOAB5, 12804, "e8cc7a1758", ("strip",), removed=("reproduced by permission of John Wiley and Sons.",)),
    # Defect 8: media evidence before the subject, PageIndex captions, Media Attributions lists, split credits.
    Case("r2_d8_video_on_this_page", PB, 52440, "1a4d0b9d2d", ("strip",), removed=("The original video on this page © Abhay Pawar",)),
    Case("r2_d8_pageindex_caption", LT, 19899, "8ad62cedb1", ("strip",), removed=("This work by Jimmy Smith is licensed under CC BY-NC-ND 4.0",)),
    Case("r2_d8_split_diagram_credit", LT, 10190, "c592fd058f", ("strip",), kept=("Diagram by Nikki Harris",)),
    Case("r2_d8_media_attributions_arr", PB, 32477, "a224f5d9dc", ("strip",), removed=("- STng Pins adapted by Rick Sellens is licensed under a All Rights Reserved license",)),
    Case("r2_d8_doab_figure_igo", DOAB5, 30485, "8622f62ecb", ("strip",), removed=("This work is available under the CC BY-NC-SA 3.0 IGO licence",)),
    Case("r2_d8_doab_figure_derivative", DOAB5, 20617, "5c561f6f4e", ("strip",)),
    # Defect 9: citations and descriptions of external resources.
    Case("r2_d9_bullet_reference_arr", LT, 5889, "69110d11a7", ("keep",)),
    Case("r2_d9_bullet_reference_arr_pb", PB, 40452, "0bffb62543", NOT_DROP),
    Case("r2_d9_reference_continuation", DOAB5, 10630, "15b290622e", ("keep",)),
    Case("r2_d9_database_notice", DOAB5, 29433, "ae326d4733", ("keep",)),
    Case("r2_d9_funding_line", DOAB5, 29907, "ac313cfdae", ("keep",)),
    Case("r2_d9_website_used_with_permission", OER, 7948, "d9c6876db5", ("keep",)),
    Case("r2_d9_author_year_source", OER, 7327, "b44bace7c5", NOT_DROP),
    Case("r2_d9_link_list_footer", OER, 8227, "463494e6ea", NOT_DROP),
    Case("r2_d9_website_guidance", OER, 5716, "2be7ebc077", NOT_DROP),
    Case("r2_d9_linked_lectures", WV, 72253, "cbcdec05fb", ("keep",)),
    Case("r2_d9_wiki_list", WV, 38, "8d31f90192", ("keep",)),
    # Defects 10 and 11: Project Gutenberg clearance note with "U.S.", renewals of pre-1931 copyrights, "not copyright".
    Case("r2_d10_pg_us_renewal_note", PG_DOCS + "00005_pg.jsonl.gz", 2501, "97403afd10", ("keep",)),
    Case("r2_d10_pg_us_renewal_note_2", PG_DOCS + "00026_pg.jsonl.gz", 1015, "c8ea900142", ("keep",)),
    Case("r2_d11_pg_renewed_1932", PG_DOCS + "00001_pg.jsonl.gz", 2216, "9367fdf909", ("keep",)),
    Case("r2_d11_pg_renewal_1953", PG_DOCS + "00024_pg.jsonl.gz", 1999, "a5a3409195", ("keep",)),
    Case("r2_d11_pg_not_copyright", PG_DOCS + "00008_pg.jsonl.gz", 2009, "7628362540", ("keep",)),
    # Defect 12: Wikiquote pages about one modern film, TV series, video game or novel.
    Case("r2_d12_game_no_disambiguator", WQ, 148531, "94b6d08d69", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_video_game", WQ, 95384, "63c515f033", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_sitcom", WQ, 150090, "be68352ebd", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_film", WQ, 149909, "7b05bbbd76", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_novel", WQ, 38333, "af539b7773", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_novel_disambiguator", WQ, 3129, "a8894e75cf", ("drop",), "wikiquote_modern_work_lead"),
    Case("r2_d12_person_film_actor", WQ, 4900, "deb8e9c63c", ("keep",)),
    Case("r2_d12_person_novelist", WQ, 134049, "a8b6c778ef", ("keep",)),
    # Defect 13: licence discussion is not stripped.
    Case("r2_d13_user_talk", WN, 7901, "d89d8f82ab", ("keep",), kept=("Because CC-BY-ND is not a permitted licence",)),
    Case("r2_d13_talk_policy", WVOY, 127732, "113e31fa80", ("keep",), kept=("If their policy is copyright, all rights reserved",)),
    Case("r2_d13_needs_permission_advice", OER, 3614, "da14f9e99b", ("strip",), kept=("needs to be used with permission",)),
    Case("r2_d13_attribution_teaching", PB, 19942, "cde2c59024", ("keep",), kept=("The attribution statement should say",)),
    Case("r2_d13_news_sentence", WN1, 1405155, "1d57631016", ("keep",)),
    # Defect 14: rows of a restricted table go with its credit.
    Case("r2_d14_table_rows", PB1, 18490, "e97c115622", ("strip",),
         removed=("Table 1 (Attribution: Copyright Rice University, OpenStax, under CC BY-NC-SA 4.0 license)",
                  "| Below adequate | Adequate | Full | Excellent | Superior |"),
         kept=("By using this method, if we assume that evaluator biases can be minimized",)),
    # Round 3: recall misses of 4c6acc6b found in cleaned oercommons output.
    Case("r3_full_deed_name_line", OER_L, 4118, "3fba63fa4e", ("drop",), "bare_nc_nd_licence_line"),
    Case("r3_conditions_of_use_field", OER_L, 4144, "b6dc07c089", ("drop",), "bare_nc_nd_licence_line"),
    Case("r3_own_materials_nc_without_cc", OER_L, 5143, "2b96fb361c", ("drop",), "work_level_nc_nd_declaration"),
    Case("r3_this_oer_nc_purposes", OER_L, 6579, "1d5e09cb05", ("drop",), "work_level_nc_nd_declaration"),
    Case("r3_excepted_from_nc_licence", OER_L, 5687, "e984537213", ("drop",), "work_level_nc_nd_declaration"),
    Case("r3_authors_give_us_permission", OER_L, 6760, "30bb35313e", ("drop",), "third_party_text_used_with_permission"),
    Case("r3_submission_rules_discussion", OER_L, 1501, "e5bbc09bb6", NOT_DROP),
    Case("r3_attribution_examples", OER_L, 2238, "cf989f5a17", NOT_DROP),
    Case("r3_external_tasks_catalogue", OER_L, 5545, "2be7ebc077", NOT_DROP),
    Case("r3_split_own_licence_sentence", LT_L, 4385, "049144810a", ("drop",), "bare_nc_nd_licence_line"),
    Case("r3_split_this_work_licence", LT_L, 22817, "e1e27ea0bc", ("drop",), "bare_nc_nd_licence_line"),
    Case("r3_text_adapted_with_permission_nc", LT_L, 19241, "061bf5ec21", ("drop",), "third_party_text_used_with_permission"),
    Case("r3_split_image_credit", LT_L, 60142, "a44e0be7a5", ("strip",), "media_credit_restricted",
         removed=("Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",)),
    Case("r3_table_row_external_template", PB_L, 22318, "b4e1c36353", NOT_DROP),
    Case("r3_unit_attribution_nd", PB_L, 23050, "3682574b1f", ("drop",), "bare_nc_nd_licence_line"),
    Case("r3_derivative_of_following_texts", LT_R3, 38651, "c147c0669e", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_derived_from_list", LT_R3, 40727, "902407f106", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_sources_list", LT_R3, 31182, "26e36ec0b9", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_chapter_licence_under_attributions", PB_L, 46876, "217c404e3e", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_flickr_credits_under_attributions", PB_L, 5083, "ea5cbe3601", NOT_DROP),
    Case("r3_portions_adapted_from", OER, 3426, "49b2d296c6", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_attributions_for_chapter", LT_R4, 54615, "1462d2cab3", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_attribution_section_label", LT_R4, 60207, "109ec0b637", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_long_sources_list", LT_R4, 31220, "d6431acd35", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_commercial_and_noncommercial_grant", "licensed/doab/00011_doab.json.gz", 514, "c5519fe873", NOT_DROP),
    Case("r3_survey_consent_permission", "licensed/doab/00008_doab.json.gz", 25339, "e691f96962", NOT_DROP),
    Case("r3_derivation_header_with_period", PB_R5, 27342, "881120afd1", ("drop",), "nc_nd_source_of_derived_text"),
    Case("r3_pg_historical_may_not_be_sold", "licensed/project_gutenberg/v0/documents/00021_pg.jsonl.gz", 976, "9003dd5cef", NOT_DROP),
    Case("r3_fiction_dialogue_permission", "licensed/wikimedia/00005_wikisource.com.jsonl.gz", 89899, "6424c625af", NOT_DROP),
    Case("r3_fiction_permission_forests", "licensed/wikimedia/00005_wikisource.com.jsonl.gz", 172271, "2a91e654c3", NOT_DROP),
    Case("r3_quoted_request_permission", "licensed/wikimedia/00087_wikipedia.com.jsonl.gz", 137, "61d20ce1aa", NOT_DROP),
]
DUPLICATE_ID_RECORDS = (Case("d11_same_id_drop", PB, 5872, "c99938cee7", ("drop",)), Case("d11_same_id_keep", PB, 6030, "9a03735451", ("keep",)))


def fixture_key(case):
    return f"{case.shard}#{case.ordinal}"


def refresh(roots):
    found = load_fixtures() if os.path.exists(FIXTURES) else {}
    wanted = collections.defaultdict(dict)
    for case in CASES + list(DUPLICATE_ID_RECORDS):
        if fixture_key(case) not in found:
            wanted[case.shard][case.ordinal] = case
    for shard, by_ordinal in wanted.items():
        root = next((root for root in roots if os.path.exists(os.path.join(root, shard))), None)
        if root is None:
            raise SystemExit(f"{shard}: not found under {roots}")
        remaining = dict(by_ordinal)
        for ordinal, doc in enumerate(lf.iter_docs(os.path.join(root, shard))):
            case = remaining.pop(ordinal, None)
            if case is None:
                continue
            if not lf.text_sha1(doc["text"]).startswith(case.sha1):
                raise SystemExit(f"{case.name}: sha1 differs in {root}; the shard changed")
            found[fixture_key(case)] = doc
            if not remaining:
                break
    with gzip.open(FIXTURES, "wt", encoding="utf-8") as handle:
        for key, doc in sorted(found.items()):
            handle.write(json.dumps({"key": key, "doc": doc}, ensure_ascii=False) + "\n")
    print(f"wrote {len(found)} records to {FIXTURES}")


def load_fixtures():
    with gzip.open(FIXTURES, "rt", encoding="utf-8") as handle:
        return {row["key"]: row["doc"] for row in map(json.loads, handle)}


def check_case(case, doc):
    problems = []
    if not lf.text_sha1(doc["text"]).startswith(case.sha1):
        return ["fixture sha1 differs"]
    action, reasons, spans = lf.classify(doc)
    if action not in case.actions:
        decisive = [(item["reason"], item["detail"][:160]) for item in reasons if item["reason"] not in ("pg_printed_source_notice",)][:4]
        problems.append(f"action {action}, expected {case.actions}: {decisive}")
    if case.reason and case.reason not in {item["reason"] for item in reasons}:
        problems.append(f"reason {case.reason} missing")
    output = lf.apply_strip(doc["text"], spans) if action == "strip" else doc["text"]
    for text in case.removed:
        if text not in doc["text"]:
            problems.append(f"removed text not in source: {text!r}")
        elif text in output:
            problems.append(f"not removed: {text!r}")
    for text in case.kept:
        if text not in output:
            problems.append(f"not kept: {text!r}")
    return problems


def check_actions_rows(docs):
    """Rows must address records by ordinal and sha1, because ids repeat within a shard with different texts."""
    problems = []
    records = [docs[fixture_key(case)] for case in DUPLICATE_ID_RECORDS]
    if records[0]["id"] != records[1]["id"] or records[0]["text"] == records[1]["text"]:
        return ["duplicate-id fixture records are not the same id with different texts"]
    with tempfile.TemporaryDirectory() as tmp:
        shard = os.path.join(tmp, "dup.jsonl.gz")
        with gzip.open(shard, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _, partial, _ = lf.scan_shard((shard, tmp, "test"))
        with open(partial, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
    expected = {ordinal: case.actions[0] for ordinal, case in enumerate(DUPLICATE_ID_RECORDS)}
    by_ordinal = {row["ordinal"]: row for row in rows}
    for ordinal, action in expected.items():
        row = by_ordinal.get(ordinal)
        if action == "keep":
            if row is not None:
                problems.append(f"record {ordinal} has a row but is a keep")
            continue
        if row is None or row["action"] != action or row["sha1"] != lf.text_sha1(records[ordinal]["text"]):
            problems.append(f"record {ordinal}: row {row and (row['action'], row['sha1'][:10])} does not match")
            continue
        try:
            lf.apply_action(records[1 - ordinal], row)
            problems.append("apply_action accepted a row for a different text with the same id")
        except ValueError:
            pass
    return problems


def check():
    docs = load_fixtures()
    failures = 0
    for case in CASES:
        doc = docs.get(fixture_key(case))
        problems = ["record missing from fixtures; run --refresh"] if doc is None else check_case(case, doc)
        for problem in problems:
            print(f"FAIL {case.name} ({case.shard}#{case.ordinal}): {problem}")
        if problems:
            failures += 1
    for problem in check_actions_rows(docs):
        print(f"FAIL actions_rows: {problem}")
        failures += 1
    print(f"{len(CASES) + 1} checks, {failures} failures")
    return failures == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--raw", nargs="+", default=[RAW], help="roots searched in order for each shard")
    args = parser.parse_args()
    if args.refresh:
        refresh(args.raw)
    sys.exit(0 if check() else 1)
