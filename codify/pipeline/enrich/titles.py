"""A law's official long title, and the short name it is cited by.

Common-law acts declare a short name; EU acts are cited by form and number, so
there are two sources. Both return `None` rather than a damaged string.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from lxml import etree

from codify.akn._schema import parse_xml
from codify.jurisdictions import load_config

# The closer is paired to the opener that actually matched. Scanning a set of
# closers instead truncates any unquoted name at its first apostrophe, which
# turns "the People's Small Business Act" into "People".
_CLOSERS_FOR = {
    '"': '"”″“',
    "\u201c": "\u201d\u2033\u201c",
    "\u201d": "\u201d\u2033",
    "'": "'\u2019",
    "\u2018": "\u2019",
    "\u00ab": "\u00bb",
    "\u201e": "\u201c\u201d",
}
_OPEN_QUOTES = "".join(_CLOSERS_FOR)
_ALL_CLOSERS = "".join(dict.fromkeys("".join(_CLOSERS_FOR.values())))
# OCR leaves a stray letter on the end of a year ("Act of 2011l").
_TRAILING_JUNK = re.compile(rf"[\s.,;:{re.escape(_ALL_CLOSERS)}]+$")
# Terminal only: the stray letter is an OCR artefact on the end of a year,
# and an internal one is a legal identifier ("Section 123A Amendment Act").
# Trailing punctuation is stripped before this runs, so the year is last.
_YEAR_JUNK = re.compile(r"\b(\d{3,4})[^\W\d_]$")
_WHITESPACE = re.compile(r"\s+")

# A name shorter than this is a parse artefact; longer than this is not short.
_MIN_LEN = 4
_MAX_LEN = 120
# Capture well past _MAX_LEN so an over-length name is rejected as a decision,
# not truncated mid-word by the window and written as if it were the name.
_CAPTURE_LEN = 400

_EU_ORDINAL = (
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH"
    r"|ELEVENTH|TWELFTH|THIRTEENTH)?\s*"
)
_EU_FORM = (
    r"(?:COMMISSION|COUNCIL|EUROPEAN PARLIAMENT AND COUNCIL)?\s*"
    r"(?:DELEGATED|IMPLEMENTING)?\s*"
    r"(?:DIRECTIVE|REGULATION|DECISION|RECOMMENDATION)"
)
# Modern titles carry the number straight after the form; pre-1990 ones put it
# in a trailing parenthetical instead.
_EU_LEAD = re.compile(
    rf"^\s*({_EU_ORDINAL}{_EU_FORM}\s*(?:\(\w+\)\s*)?(?:No\s*)?[\d/]+[/\w]*)",
    re.IGNORECASE,
)
_EU_TAIL = re.compile(
    rf"^\s*({_EU_ORDINAL}{_EU_FORM})\b.*?\((\d+/\d+/[A-Z]+)\)\s*$",
    re.IGNORECASE,
)


def _tidy(name: str) -> str | None:
    """Strip the wrapping quotes and the noise sources leave behind."""
    cleaned = _WHITESPACE.sub(" ", name).strip()
    cleaned = cleaned.lstrip(_OPEN_QUOTES).strip()
    cleaned = _TRAILING_JUNK.sub("", cleaned)
    cleaned = _YEAR_JUNK.sub(r"\1", cleaned)
    if not _MIN_LEN <= len(cleaned) <= _MAX_LEN:
        return None
    # A name with no letter is a number that survived the pattern, not a name.
    return cleaned if any(ch.isalpha() for ch in cleaned) else None


def _citation_pattern(country: str) -> re.Pattern[str] | None:
    """Declared lead-ins for this jurisdiction's short-title clause."""
    if not country:
        return None
    config = load_config(country)
    structuring = config.structuring
    leads = [lead for lead in (structuring.citation_lead_ins if structuring else []) if lead]
    subjects = [s for s in (structuring.citation_subjects if structuring else []) if s]
    # Anchored on the instrument or not at all. The same phrasing names the
    # schools and offices a law creates, and an unanchored pattern took 147 of
    # those for law names; a fallback would leave that reachable by omission.
    if not leads or not subjects:
        return None
    articles = [a for a in (structuring.citation_articles if structuring else []) if a]
    article = f"(?:{'|'.join(re.escape(a) for a in articles)})\\s+" if articles else ""
    return re.compile(
        rf"(?:{'|'.join(re.escape(s) for s in subjects)})\s+"
        rf"(?:{'|'.join(re.escape(lead) for lead in leads)})\s+(?:{article})?"
        rf"(?P<quote>[{re.escape(_OPEN_QUOTES)}])?"
        rf"(?P<name>[^\n]{{{_MIN_LEN},{_CAPTURE_LEN}}})",
        re.IGNORECASE,
    )


def _designation_rule(country: str) -> str | None:
    if not country:
        return None
    config = load_config(country)
    structuring = config.structuring
    return structuring.designation_rule if structuring else None


def declared_short_title(text: str, country: str) -> str | None:
    """The name the law gives itself, from its own text."""
    pattern = _citation_pattern(country)
    if not pattern or not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    candidate = match.group("name")
    opener = match.group("quote")
    # A quoted name ends at its own closer; an unquoted one ends the sentence.
    for closer in _CLOSERS_FOR.get(opener or "", ""):
        head, sep, _ = candidate.partition(closer)
        if sep and head.strip():
            return _tidy(head)
    return _tidy(candidate.split(".")[0])


def derived_short_title(title: str, country: str) -> str | None:
    """The designation the title's own grammar carries."""
    if _designation_rule(country) != "eu_instrument" or not title:
        return None
    lead = _EU_LEAD.match(title)
    if lead and any(ch.isdigit() for ch in lead.group(1)):
        return _tidy(lead.group(1))
    tail = _EU_TAIL.match(title)
    if tail:
        return _tidy(f"{tail.group(1).strip()} {tail.group(2)}")
    return None


def declares_any_rule(country: str) -> bool:
    """Whether the jurisdiction declares anything to resolve a short name with."""
    return bool(_citation_pattern(country) or _designation_rule(country))


# Strip lowercase status suffixes; preserve title suffixes such as "(Wales)".
_MARKUP_CHROME = re.compile(r"\s*\([a-z][a-z0-9 .,/-]*\)\s*$")
# Enclosing elements whose own title names something else.
_NOT_THIS_DOCUMENT = {"attachment", "component", "quotedStructure", "embeddedStructure"}


def _names_this_document(element: etree._Element) -> bool:
    return not any(
        isinstance(a.tag, str) and etree.QName(a).localname in _NOT_THIS_DOCUMENT
        for a in element.iterancestors()
    )


def _is_a_name(candidate: str) -> bool:
    """Reject slash-separated identifiers without whitespace."""
    return not ("/" in candidate and not any(c.isspace() for c in candidate))


def markup_short_title(akn_xml: str) -> str | None:
    """Read the document's own `docTitle` from hardened AKN markup."""
    # Allow namespace-prefixed tags.
    if not akn_xml or "docTitle" not in akn_xml:
        return None
    try:
        # Reject publisher-supplied entities and DTDs.
        root = parse_xml(akn_xml)
    except etree.XMLSyntaxError:
        return None
    stated = [
        e for e in root.iter() if isinstance(e.tag, str) and etree.QName(e).localname == "docTitle"
    ]
    # Annex and embedded-document titles do not name the enclosing act.
    for element in (e for e in stated if _names_this_document(e)):
        # Flatten inline markup such as <abbr> and <noteRef>.
        stated_text = _WHITESPACE.sub(" ", "".join(element.itertext())).strip()
        name = _tidy(_MARKUP_CHROME.sub("", stated_text))
        if name and _is_a_name(name):
            return name
    return None


def resolve_short_title(
    *, title: str, body_text: str, country: str, akn_xml: str = ""
) -> str | None:
    """Prefer declared names, then designations, then AKN markup."""
    return (
        declared_short_title(body_text, country)
        or derived_short_title(title, country)
        or markup_short_title(akn_xml)
    )


def long_title_lead_ins(country: str) -> tuple[str, ...]:
    """Declared openings of a document's own long title, e.g. `An Act`."""
    if not country:
        return ()
    config = load_config(country)
    structuring = config.structuring
    return tuple(lead for lead in (structuring.long_title_lead_ins if structuring else []) if lead)


def opens_long_title(line: str, leads: tuple[str, ...]) -> bool:
    """Case-folded: sources use title case or full caps."""
    return any(line[: len(lead)].casefold() == lead.casefold() for lead in leads)


def long_title_from_paragraphs(paragraphs: Iterable[str], country: str) -> str | None:
    """The long title among paragraphs of opening matter, first match only.

    A title split across paragraphs is rejoined, but only where the next one
    opens lowercase. The enacting formula follows a title with no terminal
    punctuation in most documents, so a punctuation-based join would swallow it.
    """
    leads = long_title_lead_ins(country)
    if not leads:
        return None
    cleaned = [_WHITESPACE.sub(" ", p).strip() for p in paragraphs]
    for index, paragraph in enumerate(cleaned):
        if not paragraph or not opens_long_title(paragraph, leads):
            continue
        title = [paragraph]
        for following in cleaned[index + 1 :]:
            # Case is the continuation signal, so scripts without it (Arabic,
            # Hebrew, CJK) never join and keep the first paragraph alone.
            if not following or not following[0].islower():
                break
            title.append(following)
        return " ".join(title)
    return None
