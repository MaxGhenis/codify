"""Canonical FRBR URI construction and validation.

Work URIs use the jurisdiction's configured shape or the canonical default;
expression URIs add language and date segments. Parsing lives in
`codify.akn.frbr`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from datetime import date
from typing import NamedTuple

import structlog

from codify.jurisdictions import (
    SLUG_DIGEST_CHARS,
    TitleIdentity,
    resolve_frbr_country,
    try_load_config,
)
from codify.lang import normalise_digits, number_run, word_bounded

logger = structlog.get_logger()

# Law-number shapes include plain, slash-compound and dash-joined numbers. The
# separators remain distinct so date/filing strings stay invalid.
_LAW_NUMBER_SHAPE = re.compile(r"(?:[0-9]+(?:-[0-9]+)*|[0-9]+/[0-9]+)[A-Za-z]?")

# Empty path segments (`//` or a trailing slash) do not identify a document.
# Non-ASCII is allowed for cited instruments; document identities are restricted
# by `law_number_token` at their minting boundary.
_UNCITABLE_SEGMENT = re.compile(r"//|/$")

# Placeholder for an unresolved year. AKN requires a date segment and XSD 1.0
# has no year zero, so `0001` keeps the resulting FRBR date schema-valid.
UNKNOWN_YEAR = "0001"

#: Namespace of a content-addressed identity. `is_citable_work_uri` refuses a
#: number opening with it, so nothing derived may take the same form.
DRAFT_PREFIX = "draft-"

#: Namespace of a title-derived identity: a digest, citable unlike the one
#: above, and no shape a stated number or a content address can take.
TITLE_DIGEST_PREFIX = "t-"


_URI_YEAR_SEGMENT = re.compile(r"[0-9]{4}")


def is_citable_work_uri(uri: str) -> bool:
    """Whether a stored work URI names a real year and number.

    `build_frbr_work_uri` mints `/akn/xz/act/0001/draft-<sha>` when neither resolved;
    writing that back replaces a real enactment date with year 1 and a real number
    with a content hash. Year and number are shape-checked rather than compared,
    without which `/akn/xz/act/2008/` and `//` both pass. `0000` is refused because
    XSD 1.0 has no year zero.
    """
    tail = uri.rsplit("/", 2)[-2:]
    if len(tail) != 2:
        return False
    year, number = tail
    return (
        _URI_YEAR_SEGMENT.fullmatch(year) is not None
        and year not in (UNKNOWN_YEAR, "0000")
        and bool(number)
        and not number.startswith(DRAFT_PREFIX)
    )


class UncitableFrbrUri(ValueError):
    """A built URI nothing could cite. Raised rather than returned, because a
    stored identity nobody can write down is worse than a failed ingest."""


# `45/PUU-IX/2011`: a court case or docket number. The leading run is the number;
# the rest is the series and the year, which the URI carries in its own segment.
_SERIES_CITATION = re.compile(r"([0-9]+)/[A-Za-z][A-Za-z0-9.\u2010-\u2015-]*/([0-9]{4})")


def series_number(value: object) -> str:
    """The citable number inside a `{number}/{series}/{year}` citation, or "".

    Narrow on purpose: a document numbered this way states its one number first.
    Office routing codes are kept out of the path by the guard, not by this.
    """
    token = normalise_digits(str(value if value is not None else "")).strip()
    match = _SERIES_CITATION.fullmatch(token)
    return match.group(1) if match else ""


class TitleDerivedIdentity(NamedTuple):
    """What a title states about its own identity: the URI segment, a digest of
    the canonical `body`, `edition` and local `year`; all empty for no title."""

    segment: str
    body: str
    edition: str
    year: str


# Letters, numbers and combining marks: `\w` drops the vowel and tone marks an
# abugida writes a word with. Format characters are gone before this runs.
def _slug_char(ch: str) -> str:
    return ch if unicodedata.category(ch)[0] in "LNM" else "-"


def _slugify(text: str) -> str:
    # Folded, not lowered: a sharp s and its capitals must reach one identity.
    return re.sub(r"-+", "-", "".join(map(_slug_char, text))).strip("-").casefold()


def _alternation(words: Iterable[str]) -> str:
    """Longest first, so a word that prefixes another matches whole; and each
    bounded, so "of" does not match inside "proof" nor "Update" in "Updated"."""
    ordered = sorted({w for w in words if w}, key=lambda w: (-len(w), w))
    return "|".join(word_bounded(w) for w in ordered)


# `[^()]*`, not `[^)]*`: an unclosed run of openers would otherwise rescan
# to end of title from every one of them.
_PARENTHETICAL = re.compile(r"\([^()]*\)")


def _opens_with_prefix(body: str, prefix: str) -> bool:
    """A kind word opens the title, a Latin one whole: "Act" is not in "Action"."""
    return re.match(word_bounded(prefix), body, re.IGNORECASE) is not None


def _first_consolidation_paren(text: str, marker: re.Pattern[str] | None) -> int | None:
    """Offset of the first parenthetical carrying a re-publication marker."""
    if marker is None:
        return None
    return next(
        (m.start() for m in _PARENTHETICAL.finditer(text) if marker.search(m.group(0))), None
    )


def identity_from_title(title: str, rule: TitleIdentity) -> TitleDerivedIdentity:
    """A citable identity from a title alone, for instruments carrying no number.
    Pure, so a re-ingest mints the same URI; the year is the last one stated."""
    # Stripped before the composition: an invisible between a letter and its
    # mark blocks the two, so removing it after leaves a different string.
    folded = unicodedata.normalize(
        "NFC", "".join(c for c in title if unicodedata.category(c) != "Cf")
    )
    text = " ".join(normalise_digits(folded).split())
    edition_re = (
        re.compile(rf"\(\s*(?:{_alternation(rule.edition_markers)})\s*([0-9]+)\s*\)", re.IGNORECASE)
        if rule.edition_markers
        else None
    )
    consolidation_re = (
        re.compile(_alternation(rule.consolidation_markers), re.IGNORECASE)
        if rule.consolidation_markers
        else None
    )
    edition = ""
    own = None
    if edition_re is not None:
        # Inside a parenthetical only: the word can be part of a title. An
        # edition after a republication marker is the one folded into it.
        marker = _first_consolidation_paren(text, consolidation_re)
        eligible = [m for m in edition_re.finditer(text) if marker is None or marker > m.start()]
        if eligible:
            # The last, as the year is: an earlier one belongs to the instrument
            # being amended and stays in the base, telling two amendments apart.
            own = eligible[-1]
            # "03" and "3" are one edition. Stripped as text, so a number too
            # long for an int is unharmed.
            edition = number_run(own.group(1))
    body = text
    if edition_re is not None:
        # The document's own edition leaves the base, as does one after the
        # republication marker; a retained one is spelt the one declared way.
        own_span = own.span() if own is not None else None

        def _edition(m: re.Match[str]) -> str:
            if m.span() == own_span or (marker is not None and m.start() > marker):
                return " "
            return f"({rule.edition_markers[0]} {number_run(m.group(1))})"

        body = edition_re.sub(_edition, text)
    if consolidation_re is not None:
        body = _PARENTHETICAL.sub(
            lambda m: " " if consolidation_re.search(m.group(0)) else m.group(0), body
        )
    year_re = (
        re.compile(
            rf"(?:{_alternation(rule.year_particles)})\s*([0-9]{{3,4}})(?![0-9])", re.IGNORECASE
        )
        if rule.year_particles
        else None
    )
    years = list(year_re.finditer(body)) if year_re else []
    # A year is its number, as an edition is: a leading zero is typography.
    year = number_run(years[-1].group(1)) if years else ""
    if years and year_re is not None:
        # Only the own year leaves; an earlier one names another instrument and
        # stays, spelt one way, as does any text after: undeclared is identity.
        own_year = years[-1]
        body = body[: own_year.start()] + " " + body[own_year.end() :]
        body = year_re.sub(lambda m: f"{rule.year_particles[0]} {number_run(m.group(1))}", body)
    body = body.strip()
    for prefix in sorted(rule.strip_prefixes, key=len, reverse=True):
        if _opens_with_prefix(body, prefix):
            body = body[len(prefix) :].strip()
            break
    canonical = _slugify(body)
    if not canonical:
        return TitleDerivedIdentity(segment="", body="", edition=edition, year=year)
    # The segment is a digest of the whole identity, so what the grammar folds
    # decides it and nothing of the title's script reaches the URI.
    identity = f"{canonical}|{edition}|{year}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:SLUG_DIGEST_CHARS]
    return TitleDerivedIdentity(
        segment=f"{TITLE_DIGEST_PREFIX}{digest}", body=canonical, edition=edition, year=year
    )


def law_number_token(value: object) -> str:
    """A law number fit for a URI path segment, or "".

    The document's own number and a number cited in its text get the same guard.
    Without it an office routing code lands verbatim in the FRBR path.
    """
    token = normalise_digits(str(value if value is not None else "")).strip()
    if not token or token.lower() in ("none", "null"):
        return ""
    if _LAW_NUMBER_SHAPE.fullmatch(token) is None:
        logger.warning("law_number_discarded", value=token)
        return ""
    return token


# legislation.gov.uk /id/ type tokens → (jurisdiction, doctype); a UK token is its
# own doctype, minted as the FRBR subtype.
_UK_PRIMARY_TOKENS = (
    "ukpga",
    "ukla",
    "asp",
    "anaw",
    "asc",
    "nia",
    "aosp",
    "aep",
    "apgb",
    "aip",
    "mwa",
    "ukcm",
    "mnia",
    "apni",
    "gbla",
    "ukppa",
)
_UK_SECONDARY_TOKENS = (
    "uksi",
    "wsi",
    "ssi",
    "nisr",
    "nisi",
    "ukmd",
    "nisro",
    "uksro",
    "ukmo",
    "ukci",
)
#: Every legislation.gov.uk type token, primary and secondary. Exported so a
#: caller that has to recognise one reads this table rather than copying it.
UK_TYPE_TOKENS = frozenset(_UK_PRIMARY_TOKENS + _UK_SECONDARY_TOKENS)

_LEG_GOV_UK_CLASS: dict[str, tuple[str, str]] = {
    **{token: ("gb", token) for token in _UK_PRIMARY_TOKENS + _UK_SECONDARY_TOKENS},
    "eur": ("eu", "regulation"),
    "eudr": ("eu", "directive"),
    "eudn": ("eu", "decision"),
}
_EU_KIND = {"directive": "directive", "regulation": "regulation", "decision": "decision"}
_LEG_ID_RE = re.compile(r"/id/([a-z]+)/(\d{4})/(\d+)((?:/[^#?]+)?)")
_EU_HREF_RE = re.compile(r"/european/(directive|regulation|decision)/(\d{4})/0*(\d+)")
_AKN_URI_RE = re.compile(r"^(/akn/[a-z-]+/[^#]*?)(?:/main)?(?:#(.+))?$")


# Reverse of _LEG_GOV_UK_CLASS for URL construction: (jurisdiction, doctype) →
# the publisher's canonical type token. One-to-one now, so the inverse is exact.
_LEG_GOV_UK_TYPE: dict[tuple[str, str], str] = {
    cls: token for token, cls in _LEG_GOV_UK_CLASS.items()
}

_FRBR_PARSE_RE = re.compile(
    r"^/akn/(?P<country>[^/]+)/(?P<doctype>[^/]+)/(?P<year>[^/]+)/(?P<number>[^/@~.]+)"
)
# /akn/gb/act/{token}/{year}/{number}: the publisher's type token as subtype.
_GB_SUBTYPE_RE = re.compile(r"^/akn/gb/act/([a-z]+)/(\d{4})/([^/@~.]+)")
# A generic kind resolves to the series a bare citation means by convention.
_GB_DEFAULT_TOKEN = {"act": "ukpga", "si": "uksi"}


def token_family(token: str) -> str | None:
    """`act` for a primary-legislation token, `si` for a secondary one, else None."""
    if token in _UK_PRIMARY_TOKENS:
        return "act"
    if token in _UK_SECONDARY_TOKENS:
        return "si"
    return None


def token_country(token: str) -> str | None:
    """The country whose work URIs carry this publisher token as a subtype."""
    entry = _LEG_GOV_UK_CLASS.get(token)
    return entry[0] if entry is not None and token_family(token) else None


def default_token(jurisdiction_code: str, doctype: str) -> str:
    """The publisher token a generic doctype resolves to, read off the FRBR
    country the jurisdiction publishes under; the doctype itself elsewhere."""
    if resolve_frbr_country(jurisdiction_code).segment == "gb":
        return _GB_DEFAULT_TOKEN.get(doctype, doctype)
    return doctype


def original_expression_path(doctype: str) -> str | None:
    """The publisher's path segment for a document's original expression:
    `enacted` for primary legislation, `made` for secondary, None otherwise."""
    if doctype in _UK_PRIMARY_TOKENS:
        return "enacted"
    if doctype in _UK_SECONDARY_TOKENS:
        return "made"
    return None


def expand_source_template(template: str, frbr_uri: str) -> str | None:
    """Substitute a SourceAdapter url_template's placeholders from a FRBR URI.

    Available: `{frbr_uri}`, `{country}`, `{doctype}`, `{year}`, `{number}`, and
    `{source_path}` (the publisher's own path). None when the URI does not parse or a
    placeholder cannot be derived.
    """
    match = _FRBR_PARSE_RE.match(frbr_uri)
    if not match:
        return None
    parts: dict[str, str] = match.groupdict()
    parts["frbr_uri"] = frbr_uri
    # /akn/eu/act/reg/2016/679 parses doctype="act", recover the subtype.
    country, doctype = parts["country"], parts["doctype"]
    m_sub = re.match(r"^/akn/eu/act/(reg|dir|dec)(?:-[a-z]+)?/(\d{4})/(\d+)", frbr_uri)
    if m_sub:
        doctype = {"reg": "regulation", "dir": "directive", "dec": "decision"}[m_sub.group(1)]
        parts["year"], parts["number"] = m_sub.group(2), m_sub.group(3)
    m_gb = _GB_SUBTYPE_RE.match(frbr_uri)
    if m_gb and m_gb.group(1) in _LEG_GOV_UK_CLASS:
        doctype = m_gb.group(1)
        parts["year"], parts["number"] = m_gb.group(2), m_gb.group(3)
    leg_type = _LEG_GOV_UK_TYPE.get((country, doctype))
    if leg_type:
        parts["source_path"] = f"{leg_type}/{parts['year']}/{parts['number']}"
    try:
        return template.format(**parts)
    except KeyError:
        return None


def parse_source_ref(href: str) -> tuple[str, str | None] | None:
    """A legal-source href to `(canonical FRBR work URI, target eId | None)`.

    Handles legislation.gov.uk `/id/…` (provision paths become published eIds),
    EUR-Lex `/european/…` redirects, and our own `/akn/…` with an optional
    `/main#eId` fragment. None for unrecognised hrefs.
    """
    # An AKN country segment can equal an external publisher's /id/ prefix.
    m = _AKN_URI_RE.match(href)
    if m:
        return (m.group(1), m.group(2) or None)
    m = _LEG_ID_RE.search(href)
    if m:
        cls = _LEG_GOV_UK_CLASS.get(m.group(1))
        if cls is None:
            return None
        juris, doctype = cls
        work = build_frbr_work_uri(juris, doctype, m.group(2), str(int(m.group(3))))
        tail = m.group(4).strip("/")
        return (work, tail.replace("/", "-").lower() or None)
    m = _EU_HREF_RE.search(href)
    if m:
        work = build_frbr_work_uri("eu", _EU_KIND[m.group(1)], m.group(2), m.group(3))
        return (work, None)
    return None


def build_frbr_work_uri(country: str, doctype: str, year: int | str, number: str) -> str:
    """`("al", "vendim", 2021, "285")` → `/akn/al/act/vendim/2021/285`.

    Uses `config.frbr.uri_patterns[doctype]` when it needs only {year}/{number};
    otherwise the canonical default (which equals what the adapters emit, so a
    re-ingest is idempotent on the Law's work URI)."""
    # Resolve the no-year sentinels before any template formats the year:
    # a configured pattern would otherwise mint /None/ or /null/ segments.
    year_absent = year is None or str(year).strip().lower() in ("", "none", "null")
    if year_absent and str(year).strip().lower() in ("none", "null"):
        # A literal "None"/"null" is an upstream leak, not a draft.
        logger.warning("frbr_year_sentinel_absorbed", year=str(year), number=number)
    # Every URI gets a date segment, so the pattern applies whether or not the
    # year resolved. Gating on a known year split a year-less decree from its
    # numbered sibling into two different doctype paths.
    year_for_uri = UNKNOWN_YEAR if year_absent else year
    cfg = try_load_config(country)
    # The country segment is not always the jurisdiction code: `gb-eng` homes
    # under `gb`, `eac` under `aa-eac`. Map through `frbr.country_code` so the
    # canonical default agrees with `work_uri_is_homed` and the ingest guard.
    country_segment = country
    if cfg is not None and cfg.frbr is not None:
        country_segment = (cfg.frbr.country_code or country).strip().lower()
        template = cfg.frbr.uri_patterns.get(doctype)
        if template:
            try:
                return _assert_citable(
                    template.format(year=year_for_uri, number=number),
                    country=country,
                    doctype=doctype,
                    number=number,
                )
            except KeyError as exc:
                # Template wants a field we do not supply; fall back to the
                # canonical default and log, since such a pattern is silently
                # under-served. A malformed brace propagates.
                logger.warning(
                    "frbr_pattern_unsupported_field",
                    country=country,
                    doctype=doctype,
                    pattern=template,
                    missing_field=str(exc).strip("'"),
                )
    # Canonical default: the doctype is the AKN subtype segment, except a plain
    # "act" (or none), which has no subtype → /akn/{country}/act/{year}/{number}.
    base = (
        f"/akn/{country_segment}/act/{doctype}"
        if doctype and doctype != "act"
        else f"/akn/{country_segment}/act"
    )
    # AKN requires a date segment, so a year we could not resolve gets the
    # unknown-date placeholder rather than a gap or an omission. Cobalt refuses
    # both `/akn/xz/act//draft-x` and `/akn/xz/act/draft-x`.
    built = f"{base}/{year_for_uri}/{number}"
    return _assert_citable(built, country=country, doctype=doctype, number=number)


def _assert_citable(uri: str, *, country: str, doctype: str, number: str) -> str:
    """The last gate before an identity is stored. A URI that reaches here bad
    was bad in its inputs, so name them in the error rather than the result."""
    if _UNCITABLE_SEGMENT.search(uri):
        raise UncitableFrbrUri(
            f"{uri!r} is not citable (empty path segment); "
            f"country={country!r} doctype={doctype!r} number={number!r}"
        )
    return uri


def build_frbr_expression_uri(work_uri: str, language: str, date_: date | str | None) -> str:
    """`(work, "sqi", date(2021,5,19))` → `{work}/sqi@2021-05-19`.

    An empty string is no date, same as None. It is falsy but not None, so the
    `is None` test alone emitted a trailing bare `@` and minted an expression URI
    no row would ever match."""
    if not date_:
        return f"{work_uri}/{language}"
    iso = date_ if isinstance(date_, str) else date_.isoformat()
    return f"{work_uri}/{language}@{iso[:10]}"


# The `@date` segment of an expression or manifestation FRBR URI. Anchored on the
# `@` so a `/!component` tail or a `.akn` format suffix after it is left alone.
EXPRESSION_URI_DATE = re.compile(r"@(\d{4}-\d{2}-\d{2})")


def expression_uri_date(uri: str | None) -> str | None:
    """The date an expression URI names, or None when it names none.

    The inverse of `build_frbr_expression_uri`, and the authority on a stored row's
    expression date over the column of the same name: `versions.expression_uri` is
    immutable by trigger and is what save-time dedup and every identity join use,
    while `expression_date` only orders versions and can disagree.

    Stricter than the rewriter's pattern in `akn_meta`, which only has to find a
    segment to replace: `@2026-08-021` matched as a prefix, gave back `2026-08-02`
    and forked the version. The date must end the URI, or be followed by the `/` of
    a `/!component` tail or the `.` of a manifestation format, the only shapes
    `build_frbr_expression_uri` and `_rebase` produce.
    """
    if not uri:
        return None
    found = EXPRESSION_URI_DATE.search(uri)
    if found is None or uri[found.end() : found.end() + 1] not in ("", "/", "."):
        return None
    try:
        date.fromisoformat(found.group(1))
    except ValueError:
        return None
    return found.group(1)
