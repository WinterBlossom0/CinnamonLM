import ipaddress
from collections.abc import Collection

import regex

# Replacement tokens of the Common Pile mixer configs (r-three/common-pile, filtering/mixer_configs).
EMAIL_TOKEN = "<EMAIL_ADDRESS>"
IP_TOKEN = "<IP_ADDRESS>"
PHONE_TOKEN = "<PHONE_NUMBER>"
PII_KINDS = ("email", "ip", "ipv6", "phone")

# The local part may not end with a dot (RFC 5322 dot-atom), which rejects image names such as "soseki-3.@2x.jpeg".
EMAIL_PATTERN = regex.compile(
    r"(?<![\w.+%-])[\w+%-](?:[\w.+%-]*[\w+%-])?@[\w-]+(?:\.[\w-]+)*\.(?P<tld>[^\W\d_]{2,})(?![\w-]|\.[\w-])"
)
FILE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "gif", "svg", "webp", "bmp", "tif", "tiff", "pdf"})

OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# Neighbouring dots or digits mean a longer dotted number such as section "2.2.1.2.8" or a DOI.
IP_PATTERN = regex.compile(rf"(?<![\w.]){OCTET}(?:\.{OCTET}){{3}}(?![\w]|\.\d)")
# StarCoder (arXiv:2305.06161 §4) does not redact public DNS resolvers: they are not personal data.
# Candidate IPv6 text: hex digits and at least two colons; _mask_ipv6 decides.
IPV6_PATTERN = regex.compile(r"(?<![\w:.])(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)[0-9A-Fa-f:]{2,39}(?![\w:])")
# Global unicast 2000::/3 holds all assigned public addresses; shorter forms such as "15::" or "A::" are list markers.
IPV6_FIRST_GROUP = regex.compile(r"[23][0-9A-Fa-f]{3}")
IPV6_MIN_GROUPS = 4
PUBLIC_DNS_SERVERS = frozenset({
    "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9", "149.112.112.112",
    "208.67.222.222", "208.67.220.220", "76.76.19.19", "76.223.122.150",
    "8.26.56.26", "8.20.247.20", "94.140.14.14", "94.140.15.15",
    "2001:4860:4860::8888", "2001:4860:4860::8844", "2606:4700:4700::1111", "2606:4700:4700::1001",
    "2620:fe::fe", "2620:fe::9",
})

# Dolma pii_regex_with_counts_fast_v2 phone shape (NANP), but a separator before the last four digits is required so
# that ISBN-10s and other bare ten-digit numbers do not match; a digit next to the match marks a numeric table.
# NANP area and exchange codes start with 2-9, and a bare "1" prefix needs a separator, which rejects catalogue
# numbers such as "1978.004.0001".
PHONE_PATTERN = regex.compile(
    r"(?<![\w.+/-])(?<!\d[ \t,;]{1,2})(?:\+1[-. ]?|1[-. ])?(?:\([2-9]\d{2}\)|[2-9]\d{2})[-. ]?[2-9]\d{2}[-. ]\d{4}"
    r"(?![\w-]|\.\d)(?![ \t,;]{1,2}\d)"
)
ISBN_CONTEXT = regex.compile(r"(?i)isbn[\w\s:-]{0,6}$")


def _mask_email(match: regex.Match) -> str:
    if match.group("tld").lower() in FILE_EXTENSIONS:
        return match.group(0)
    return EMAIL_TOKEN


def _mask_ip(match: regex.Match) -> str:
    address = match.group(0)
    octets = address.split(".")
    # Outline numbers such as "4.2.1.3" are valid public addresses; real host addresses with four one-digit octets are rare.
    if all(len(octet) == 1 for octet in octets):
        return address
    if address in PUBLIC_DNS_SERVERS or not ipaddress.IPv4Address(address).is_global:
        return address
    return IP_TOKEN


def _mask_ipv6(match: regex.Match) -> str:
    address = match.group(0)
    groups = [group for group in address.split(":") if group]
    if len(groups) < IPV6_MIN_GROUPS or not IPV6_FIRST_GROUP.fullmatch(address.split(":")[0]):
        return address
    try:
        parsed = ipaddress.IPv6Address(address)
    except ValueError:
        return address
    if not parsed.is_global or address.lower() in PUBLIC_DNS_SERVERS:
        return address
    return IP_TOKEN


def _mask_phone(match: regex.Match) -> str:
    preceding = match.string[max(0, match.start() - 16):match.start()]
    if ISBN_CONTEXT.search(preceding):
        return match.group(0)
    return PHONE_TOKEN


MASKERS = {
    "email": (EMAIL_PATTERN, _mask_email),
    "ip": (IP_PATTERN, _mask_ip),
    "ipv6": (IPV6_PATTERN, _mask_ipv6),
    "phone": (PHONE_PATTERN, _mask_phone),
}


def mask_pii(text: str, kinds: Collection[str] = PII_KINDS) -> str:
    """Replaces e-mail addresses, public IPv4/IPv6 addresses and NANP phone numbers with Common Pile tokens.

    Only the matched value is replaced; surrounding whitespace and punctuation stay.
    """
    unknown = set(kinds) - set(PII_KINDS)
    if unknown:
        raise ValueError(f"unknown PII kinds: {sorted(unknown)}")
    for kind in PII_KINDS:
        if kind not in kinds:
            continue
        pattern, masker = MASKERS[kind]
        text = pattern.sub(masker, text)
    return text
