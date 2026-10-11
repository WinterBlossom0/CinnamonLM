import pytest

from data_pipeline.common.pii import EMAIL_TOKEN, IP_TOKEN, PHONE_TOKEN, mask_pii


def test_masks_staff_emails_and_phones_keeping_separators(snippets):
    masked = mask_pii(snippets["staff_contacts"]["text"])
    assert f"- Lisa Bickmore, AD: {EMAIL_TOKEN}, {PHONE_TOKEN}, AAB 165-F" in masked
    assert f"Co-Coordinator: {EMAIL_TOKEN}, {PHONE_TOKEN}, AAB 165J" in masked
    assert "@slcc.edu" not in masked


def test_masks_faculty_contacts_on_own_lines(snippets):
    masked = mask_pii(snippets["faculty_contacts"]["text"])
    assert masked.startswith(f"Please call {PHONE_TOKEN} or one")
    assert f"Indio Campus, 304-C\n{EMAIL_TOKEN}\n{PHONE_TOKEN}\nJessica Huffman" in masked


def test_masks_parenthesized_phone_before_period(snippets):
    masked = mask_pii(snippets["letter_phone"]["text"])
    assert f"contact him at {PHONE_TOKEN}.\n" in masked
    assert f"Sandra L. Reynolds\n{PHONE_TOKEN}\n" in masked


def test_phone_with_country_code_and_kind_selection(snippets):
    text = snippets["wikivoyage_phones"]["text"]
    assert f"phone line at {PHONE_TOKEN} or call the park at {PHONE_TOKEN}." in mask_pii(text)
    assert mask_pii(text, kinds=("email", "ip")) == text


def test_masks_ipv6_of_anonymous_editor(snippets):
    masked = mask_pii(snippets["talk_page_ipv6"]["text"])
    assert f"added by {IP_TOKEN} (talk)" in masked
    assert mask_pii(snippets["talk_page_ipv6"]["text"], kinds=("email", "ip", "phone")) != masked


def test_masks_toll_free_number_with_country_prefix(snippets):
    masked = mask_pii(snippets["toll_free_phone"]["text"])
    assert f"Poison Control should be contacted at {PHONE_TOKEN}. They" in masked


@pytest.mark.parametrize("name", [
    "isbn", "section_numbers", "image_filename", "private_ips", "museum_accession_numbers"])
def test_leaves_known_false_positives_of_common_pile_untouched(snippets, name):
    text = snippets[name]["text"]
    assert mask_pii(text) == text


@pytest.mark.parametrize("text, expected", [
    ("Server at 172.217.3.110 answered.", f"Server at {IP_TOKEN} answered."),
    ("(203.0.113.9 is documentation)", "(203.0.113.9 is documentation)"),
    ("DNS 8.8.8.8 resolves.", "DNS 8.8.8.8 resolves."),
    ("See 4.2.1.3 below.", "See 4.2.1.3 below."),
    ("version v1.22.33.44", "version v1.22.33.44"),
    ("doi 10.1186/s11658-022-00371-3", "doi 10.1186/s11658-022-00371-3"),
    ("ISBN 978-0-306-40615-7", "ISBN 978-0-306-40615-7"),
    ("ISBN-10: 030-640-6152", "ISBN-10: 030-640-6152"),
    ("call 5551234567 now", "call 5551234567 now"),
    ("by 2a03:1b20:4:f011::1d (talk)", f"by {IP_TOKEN} (talk)"),
    ("range 2602:306:C583:2370::/64 used", f"range {IP_TOKEN}/64 used"),
    ("range 2600:1:8000::/33 is", "range 2600:1:8000::/33 is"),
    ("DNS 2001:4860:4860::8888 here", "DNS 2001:4860:4860::8888 here"),
    ("doc 2001:db8::1:2:3 here", "doc 2001:db8::1:2:3 here"),
    ("Chapter 15:: Mr. Knight", "Chapter 15:: Mr. Knight"),
    ("at 15:18:16 1994, std::vector, A:: No", "at 15:18:16 1994, std::vector, A:: No"),
    ("dial +1 (415) 555-0100.", f"dial {PHONE_TOKEN}."),
    ("exchange 212-055-0100", "exchange 212-055-0100"),
    ("area 123-456-7890", "area 123-456-7890"),
    ("12 345 678 9012 13", "12 345 678 9012 13"),
    ("mail <a.b-c+d@mail.example.co.uk>.", f"mail <{EMAIL_TOKEN}>."),
    ("see https://medium.com/@writer/post", "see https://medium.com/@writer/post"),
    ("img logo@2x.png here", "img logo@2x.png here"),
])
def test_edge_cases(text, expected):
    assert mask_pii(text) == expected


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        mask_pii("x", kinds=("names",))
