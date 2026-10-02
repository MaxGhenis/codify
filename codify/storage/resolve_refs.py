"""Post-ingest resolution of dangling cross_references.

Unit refs stamp `target_provision_id`, container refs `target_section_id`,
bare work-URI refs `target_law_id`; the rest stay dangling, counted by
reason. `target_uri` is never cleared (provenance).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Container
from dataclasses import dataclass, field
from functools import lru_cache

import structlog
from lxml import etree
from sqlalchemy import select
from sqlalchemy import text as text_clause
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from codify.frbr import UNKNOWN_YEAR, parse_source_ref
from codify.jurisdictions import try_load_config
from codify.storage.models import CrossReference, Law, Provision, Section, Version

logger = structlog.get_logger()

# Bump when a resolution rule changes what a row should have been stamped with.
# The selector then re-examines every row below it, clearing its ids first, so a
# rule found wrong can be corrected rather than lived with. 2: refuse a
# placeholder provision as a target, and require a law citation to name its
# instrument.
RESOLVER_VERSION = 2


@dataclass
class ResolveStats:
    """Per-reason outcome counts for one version's dangling rows."""

    examined: int = 0
    resolved_provision: int = 0
    resolved_law: int = 0
    resolved_section: int = 0
    unknown_law: int = 0
    missing_same_doc: int = 0  # '#eid' target in the document's AKN but carrying no row
    # '#eid' naming a container stored only as its parts: its first stored part.
    resolved_container: int = 0
    # '#eid' inside a stored row, such as quoted amending text: that row.
    resolved_enclosing: int = 0
    # '#eid' the document never carried under any name: an anchor from a
    # publisher's rendering, which no later acquisition makes resolvable.
    anchor_not_in_document: int = 0
    resolved_registry: int = 0  # upstream registry refs (e.g. rada /go/{ref})
    unparsed: int = 0
    # Diagnostic overlap: fragment refs whose unit wasn't found but whose law
    # was, counted inside resolved_law too.
    fragment_fallback: int = 0
    # Year-agnostic number join, for citations that name a number and no date.
    resolved_by_number: int = 0
    series_not_recorded: int = 0  # cited series the corpus does not distinguish
    series_unconfirmed: int = 0  # source text does not name the series the URI claims
    number_not_found: int = 0
    ambiguous_number: int = 0
    # A held law was found for the URI, but the citing text does not name it.
    law_unconfirmed: int = 0
    # The text names a held law, but another held law's name matches it longer.
    law_outmatched: int = 0

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items()}


def _origin_of(target_uri: str) -> str:
    """Where the target URI came from, from its own shape.

    An absolute URL is a link the publisher wrote into the document. `/go/…` is
    an upstream register's own identifier. A bare `/akn/…` path is one this
    pipeline minted while reading prose, and is therefore a reading rather than
    a citation.
    """
    if target_uri.startswith(("http://", "https://")):
        return "href"
    if target_uri.startswith("/go/"):
        return "registry"
    return "text"


def _text_names_law(source_text: str, work_uri: str, *titles: str | None) -> str | None:
    """Does the citing text name this instrument, by either stated rule?

    (a) The instrument's whole title, normalised, appears in the text. Every
    token, so a short connective cannot be dropped and turn a title into a
    pattern that never matches what a drafter wrote.

    (b) Both the number and the year appear, each on a word boundary. A number
    alone is not enough: a duration, a quantity or a date carries digits too,
    and a bare number matching anywhere in a provision is how an unrelated
    instrument came to be cited by the resolver.

    Anything else is refused and counted. A later tier can revisit the refusal;
    a wrong stamp leaves no trace to revisit.
    """
    text = _normalised(source_text)
    if not text:
        return None
    # Every name it answers to: ingest stores the long title in `title` and the
    # designation a drafter actually writes in `short_title`, so checking one
    # refuses citations made by the other.
    for title in titles:
        if _is_placeholder_title(title):
            continue
        whole = _normalised(title)
        # A name, and on word boundaries. A title of "30" is not a name: it
        # confirmed on any text carrying a 30, a duration included, which is
        # the coincidence this rule refuses. The test is "not only digits",
        # not a length: a real title can be one short word, and shorter still
        # in a script that writes denser words. A stub title of path shape is
        # already refused above, which is why nothing more is needed here.
        if (
            whole
            and not whole.isdigit()
            and re.search(rf"(?<![0-9\w]){re.escape(whole)}(?![0-9\w])", text)
        ):
            return "title"
    # Normalised on both sides, or a number carrying letters (a session-numbered
    # act, a series suffix) never matches the text it appears in.
    number, year = (_normalised(part) for part in _number_and_year(work_uri))
    if number and year:
        on_boundary = r"(?<![0-9\w]){}(?![0-9\w])"
        if re.search(on_boundary.format(re.escape(number)), text) and re.search(
            on_boundary.format(re.escape(year)), text
        ):
            return "number_and_year"
    return None


async def _law_is_named(
    session: AsyncSession,
    cache: _LawCache,
    ref: CrossReference,
    law_id: uuid.UUID,
    source_text: str,
    work_uri: str,
    stats: ResolveStats,
) -> bool:
    """May anything inside this law be stamped for this reference?

    Called before the unit branches, not after them. Guarding only the
    law-level stamp left the sharper hole open: the gate refused a wrong law
    and then a wrong *unit of that law* was stamped anyway, because the
    fragment branch ran first and had no confirmation of its own. A fragment
    like `art_2` exists in most documents, so a mis-derived work URI stamped a
    provision inside the wrong instrument almost every time.

    An authored href, or a register's own identifier, stands on its own. Only a
    URI one of our passes read out of prose has to be borne out by the words it
    was read from.
    """
    if ref.resolution_origin != "text":
        return True
    if law_id not in cache.titles:
        row = (
            await session.execute(select(Law.title, Law.short_title).where(Law.id == law_id))
        ).first()
        cache.titles[law_id] = (row.title, row.short_title) if row else (None, None)
    confirmed_by = _text_names_law(source_text, work_uri, *cache.titles[law_id])
    if not confirmed_by and ref.ref_type.startswith("amendment_"):
        opener = await _scoped_opener_text(session, cache, ref, source_text)
        if opener is not None:
            source_text = opener
            confirmed_by = _text_names_law(source_text, work_uri, *cache.titles[law_id])
    if not confirmed_by:
        stats.law_unconfirmed += 1
        return False
    # Only where a name was what confirmed. A citation naming the instrument by
    # number and year has said which one it means, and another law's longer
    # name nearby does not unsay it.
    if confirmed_by != "title":
        return True
    if await _a_longer_name_matches(session, cache, law_id, source_text):
        stats.law_outmatched += 1
        return False
    return True


async def _a_longer_name_matches(
    session: AsyncSession, cache: _LawCache, law_id: uuid.UUID, source_text: str
) -> bool:
    """Compare all jurisdiction candidates using the title matcher's normalisation."""
    if law_id not in cache.names_by_law:
        rows = (
            await session.execute(
                text_clause(
                    """
                    SELECT l.id, l.title, l.short_title
                    FROM laws l
                    WHERE l.jurisdiction_id = (
                        SELECT jurisdiction_id FROM laws WHERE id = :law_id
                    )
                    """
                ),
                {"law_id": law_id},
            )
        ).all()
        names = [(row.id, row.title, row.short_title) for row in rows]
        # Each law in this jurisdiction shares one candidate list for the run.
        # SQL must not discard names that Python's normalisation equates.
        normalised_names = _normalised_title_candidates(names)
        for candidate_id, title, short_title in names:
            cache.names_by_law[candidate_id] = normalised_names
            cache.titles[candidate_id] = (title, short_title)
        cache.names_by_law[law_id] = normalised_names
    return _normalised_occurrences_outmatched(law_id, source_text, cache.names_by_law[law_id])


def _title_occurrences_outmatched(
    law_id: uuid.UUID,
    source_text: str,
    names: list[tuple[uuid.UUID, str | None, str | None]],
) -> bool:
    """Refuse only when every target occurrence has an overlapping rival."""
    return _normalised_occurrences_outmatched(
        law_id, source_text, _normalised_title_candidates(names)
    )


def _normalised_title_candidates(
    names: list[tuple[uuid.UUID, str | None, str | None]],
) -> list[tuple[uuid.UUID, str]]:
    candidates = []
    for candidate_id, *titles in names:
        for title in titles:
            name = _normalised(title)
            if name and not name.isdigit() and not _is_placeholder_title(title):
                candidates.append((candidate_id, name))
    return candidates


def _normalised_occurrences_outmatched(
    law_id: uuid.UUID, source_text: str, names: list[tuple[uuid.UUID, str]]
) -> bool:
    text = _normalised(source_text)
    matches: list[tuple[uuid.UUID, int, int]] = []
    for candidate_id, name in names:
        if name not in text:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            matches.append((candidate_id, *match.span()))
    targets = [(start, end) for candidate, start, end in matches if candidate == law_id]
    return bool(targets) and all(
        any(
            candidate != law_id
            and rival_start < end
            and start < rival_end
            and rival_end - rival_start >= end - start
            for candidate, rival_start, rival_end in matches
        )
        for start, end in targets
    )


# A title that is a path or bears a "(latest)" marker is a stub the acquisition
# lane minted, not a name anyone cites. Matching on one confirms by accident.
# A stub is a path, or a name carrying a currency marker. "No spaces" is not
# the test: plenty of real names are one word, in any script.
_PLACEHOLDER_TITLE = re.compile(r"^[\w.-]*/[\w./-]*$|\(latest\)")


def _is_placeholder_title(title: str | None) -> bool:
    return bool(title) and bool(_PLACEHOLDER_TITLE.search(title.strip()))


def _normalised(value: str | None) -> str:
    """Case-folded, digit-folded, whitespace-collapsed."""
    if not value:
        return ""
    return " ".join(value.translate(_DIGIT_FOLD).casefold().split())


def _number_and_year(work_uri: str) -> tuple[str, str]:
    """The number and year an FRBR work URI carries, as written in the text.

    Leading zeros go: a URI pads where a drafter does not.
    """
    parts = [p for p in work_uri.rstrip("/").split("/") if p]
    number = parts[-1] if parts else ""
    # Only the slot before the number, which is the one the template names.
    # Scanning every segment read the number itself as a year wherever it
    # happens to be four digits, so an instrument numbered 1451 confirmed on
    # "1451" as though the text had written its year.
    year = ""
    if len(parts) >= 2:
        slot = parts[-2]
        head = slot[:4]
        if head.isdigit() and (len(slot) == 4 or slot[4:5] == "-"):
            year = head
    # Leading zeros go, because a URI pads where a drafter does not. A suffix
    # stays, because it is part of what the instrument is called.
    lead, dash, rest = number.partition("-")
    if lead.isdigit():
        number = str(int(lead)) + (dash + rest if dash else "")
    return number, year


_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_DIGIT_FOLD = {ord(c): str(i % 10) for i, c in enumerate(_ARABIC_DIGITS)}


@dataclass
class _LawCache:
    """Per-run memo: work URI → (law_id, latest version_id) or None."""

    by_uri: dict[str, tuple[uuid.UUID, uuid.UUID] | None] = field(default_factory=dict)
    by_number: dict[str, list[tuple[uuid.UUID, uuid.UUID]]] = field(default_factory=dict)
    # law_id -> title, for the check that the citing text names the instrument.
    titles: dict[uuid.UUID, tuple[str | None, str | None]] = field(default_factory=dict)
    names_by_law: dict[uuid.UUID, list[tuple[uuid.UUID, str]]] = field(default_factory=dict)
    scoped_evidence: dict[tuple[uuid.UUID, str, str], tuple[str, str]] | None = None


async def _scoped_opener_text(
    session: AsyncSession, cache: _LawCache, ref: CrossReference, source_text: str
) -> str | None:
    from codify.storage.scoped_refs import scoped_opener_evidence

    if cache.scoped_evidence is None:
        cache.scoped_evidence = {}
        version = (
            await session.execute(
                select(Version.id, Version.akn_xml)
                .join(Provision, Provision.version_id == Version.id)
                .where(Provision.id == ref.source_provision_id)
            )
        ).one_or_none()
        if version is not None and version.akn_xml:
            evidence = scoped_opener_evidence(version.akn_xml)
            if evidence:
                sources = (
                    await session.execute(
                        select(Provision.id, Provision.akn_eid).where(
                            Provision.version_id == version.id
                        )
                    )
                ).all()
                by_eid: dict[str, list[uuid.UUID]] = {}
                for source in sources:
                    by_eid.setdefault(source.akn_eid, []).append(source.id)
                for (eid, uri, kind), texts in evidence.items():
                    ids = by_eid.get(eid, [])
                    if len(ids) == 1:
                        cache.scoped_evidence[(ids[0], uri, kind)] = texts
    match = cache.scoped_evidence.get((ref.source_provision_id, ref.target_uri or "", ref.ref_type))
    if match is None or not match[1] or _normalised(match[0]) != _normalised(source_text):
        return None
    return match[1]


_DATE_SLOT = re.compile(r"^(/akn/[a-z-]+/[a-z-]+)/(\d{4})(?:-\d\d-\d\d)?/([^/]+)$")

# A citation whose year is the unknown placeholder, split into the part that
# identifies the work and the number that names it. A series segment between the
# doctype and the year (`/akn/xx/act/pd/0001/442`) is captured because its
# presence is what the resolver refuses on.
_NUMBER_ONLY = re.compile(rf"^(/akn/[a-z-]+/[a-z-]+)(/[a-z-]+)?/{UNKNOWN_YEAR}/([^/]+)$")

# How far back to read for the series. Long enough for "otherwise known as" and
# a short title, short enough not to reach the previous citation where several
# are listed together.
_SERIES_WINDOW = 90


def _naming_phrase(template: str) -> str:
    """The words a citation template uses to name its instrument.

    A template of the form "<name> No. {number}" gives "<name>". The connector
    and the number are dropped; what is left is what a drafter writes before the
    digits.
    """
    head = template.split("{")[0]
    words = [w for w in re.split(r"[\s.]+", head) if w]
    while words and words[-1].lower().rstrip(".") in {"no", "nos", "blg", "number", "numbered"}:
        words.pop()
    return " ".join(words)


def _phrase_pattern(phrase: str) -> str:
    """A phrase and the initials a drafter shortens it to, as one alternation.

    Deriving the abbreviation from the initials is mechanism rather than a fact
    about any jurisdiction: a two-word instrument name is cited by its letters
    with or without stops, and the config need not enumerate that.
    """
    words = phrase.split()
    forms = [r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"]
    if len(words) > 1:
        # Bounded on both sides. Unanchored, a two-letter abbreviation matches
        # the letters spanning a space inside another instrument's name, so a
        # correct citation reads as a contradicting one.
        initials = "".join(w[0] for w in words)
        forms.append(r"\b" + r"\.?\s?".join(re.escape(c) for c in initials) + r"\.?\b")
    return "|".join(forms)


@lru_cache(maxsize=64)
def _series_patterns(code: str) -> tuple[re.Pattern[str], re.Pattern[str]] | None:
    """(confirms, contradicts) for a jurisdiction, from its own citation templates.

    The instrument names belong to the jurisdiction, so they are read from its
    config rather than listed here. A jurisdiction that declares no citation
    templates gets no patterns and therefore no number join, which is the safe
    direction.
    """
    config = try_load_config(code)
    if not (config and config.numbering):
        return None
    fields = {
        k: v
        for k, v in config.numbering.model_dump().items()
        if k.endswith("_citation") and isinstance(v, str) and v
    }
    default = f"{config.default_document_class}_citation"
    if default not in fields:
        return None
    abbrev = config.numbering.model_dump().get(f"{config.default_document_class}_citation_abbrev")
    confirming = [_naming_phrase(fields[default])]
    if isinstance(abbrev, str) and abbrev:
        confirming.append(_naming_phrase(abbrev))
    others = [_naming_phrase(v) for k, v in fields.items() if k != default]
    if not others:
        return None
    return (
        re.compile("|".join(_phrase_pattern(p) for p in confirming if p), re.I),
        re.compile("|".join(_phrase_pattern(p) for p in others if p), re.I),
    )


def _text_confirms_bare_series(code: str, body: str | None, number: str) -> bool:
    """Does the source name the jurisdiction's default instrument beside this number?

    A reference minted before the grammar kept the series carries no way to tell
    one instrument from another, and both are numbered from one, so the join
    would stamp whichever the corpus holds. The provision still says which.

    Refuses on silence as well as on contradiction: a provision that does not
    say what it means is not evidence for the reading that happens to resolve.
    """
    patterns = _series_patterns(code)
    if patterns is None:
        return False
    confirms, contradicts = patterns
    hit = re.search(r"\b" + re.escape(number) + r"\b", body or "")
    if hit is None:
        return False
    window = (body or "")[max(0, hit.start() - _SERIES_WINDOW) : hit.end()]
    return not contradicts.search(window) and bool(confirms.search(window))


def _numbers_are_unique(base: str) -> bool:
    """Does the cited jurisdiction say an instrument number identifies a work?

    Off unless declared. Most jurisdictions restart numbering each year, so a
    year-less citation names one of many, and a corpus holding only one of them
    would let the join stamp that one with confidence. The claim is about the
    series and cannot be read off the rows held.
    """
    parts = base.split("/")
    if len(parts) < 3:
        return False
    # The country comes off the citation, not from the caller, so a cited
    # jurisdiction this deployment does not carry is ordinary.
    config = try_load_config(parts[2])
    return bool(config and config.numbering and config.numbering.numbers_unique_across_years)


async def _law_by_number(
    session: AsyncSession, cache: _LawCache, base: str, number: str
) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """Laws whose work URI is this base and number under any year.

    The citation carries a number and no date, so the year is the one segment
    that cannot be matched on. Every hit is returned rather than the first,
    because more than one is a reason to refuse and not a tie to break.
    """
    # Digits only. The number reaches here from a persisted href and is spliced
    # into a Postgres regex, where `(` fails the query and `.` matches a digit
    # that is not the one cited. A non-numeric segment is not a number anyway.
    if not number.isdigit():
        return []
    key = f"{base}//{number}"
    if key in cache.by_number:
        return cache.by_number[key]
    rows = list(
        (
            await session.execute(
                select(Law.id, Version.id)
                .join(Version, Version.law_id == Law.id)
                # A law under the placeholder year is itself undated, so it
                # cannot confirm the cited work: matching one would answer an
                # unknown date with another unknown date.
                .where(Law.frbr_work_uri.regexp_match(f"^{base}/[0-9]{{4}}/{number}$"))
                .where(Law.frbr_work_uri != f"{base}/{UNKNOWN_YEAR}/{number}")
                .where(Version.parent_version_id.is_(None))
                .order_by(Version.expression_date.desc(), Version.ingested_at.desc())
            )
        ).all()
    )
    seen: dict[uuid.UUID, uuid.UUID] = {}
    for law_id, version_id in rows:
        seen.setdefault(law_id, version_id)
    hits = list(seen.items())
    cache.by_number[key] = hits
    return hits


async def _law_latest_version(
    session: AsyncSession, cache: _LawCache, work_uri: str
) -> tuple[uuid.UUID, uuid.UUID] | None:
    if work_uri in cache.by_uri:
        return cache.by_uri[work_uri]
    row = (
        await session.execute(
            select(Law.id, Version.id)
            .join(Version, Version.law_id == Law.id)
            .where(Law.frbr_work_uri == work_uri)
            .where(Version.parent_version_id.is_(None))
            .order_by(Version.expression_date.desc(), Version.ingested_at.desc())
            .limit(1)
        )
    ).one_or_none()
    if row is None:
        # Date-slot tolerance (some sources cite a full date where the work is
        # homed under a year):
        # match year-form and dated-form alike, unique law only.
        m = _DATE_SLOT.match(work_uri)
        if m:
            base, year, number = m.groups()
            laws = (
                await session.execute(
                    select(Law.id, Version.id, Law.frbr_work_uri)
                    .join(Version, Version.law_id == Law.id)
                    .where(
                        (Law.frbr_work_uri == f"{base}/{year}/{number}")
                        | Law.frbr_work_uri.like(f"{base}/{year}-%/{number}")
                    )
                    .where(Version.parent_version_id.is_(None))
                    .order_by(Version.expression_date.desc(), Version.ingested_at.desc())
                )
            ).all()
            if len({r[2] for r in laws}) == 1:
                cache.by_uri[work_uri] = (laws[0][0], laws[0][1])
                return cache.by_uri[work_uri]
    cache.by_uri[work_uri] = (row[0], row[1]) if row else None
    return cache.by_uri[work_uri]


async def _provision_in_version(
    session: AsyncSession, version_id: uuid.UUID, eid: str
) -> uuid.UUID | None:
    exact: uuid.UUID | None = (
        (
            await session.execute(
                select(Provision.id)
                .where(Provision.version_id == version_id)
                .where((Provision.akn_eid == eid) | (Provision.akn_wid == eid))
                # A placeholder is not an answer. Where an explanatory
                # apparatus is printed under the eId of the unit it explains,
                # a citation of that unit resolved to the sentence saying it
                # needs no explanation.
                .where(Provision.excluded_from_pool.is_not(True))
            )
        )
        .scalars()
        .first()
    )
    if exact is not None:
        return exact
    # Depth-mismatched refs ("#chp_1__art_2" for chp_1__art_2__para_1,
    # pathless "#art_2"): segment-boundary match, unique candidate only.
    escaped = eid.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    candidates = (
        (
            await session.execute(
                select(Provision.id)
                .where(Provision.version_id == version_id)
                .where(Provision.excluded_from_pool.is_not(True))
                .where(
                    Provision.akn_eid.like(f"{escaped}\\_\\_%", escape="\\")
                    | Provision.akn_eid.like(f"%\\_\\_{escaped}", escape="\\")
                    | Provision.akn_eid.like(f"%\\_\\_{escaped}\\_\\_%", escape="\\")
                )
                .limit(2)
            )
        )
        .scalars()
        .all()
    )
    return candidates[0] if len(candidates) == 1 else None


async def _section_in_version(
    session: AsyncSession, version_id: uuid.UUID, eid: str
) -> uuid.UUID | None:
    exact: uuid.UUID | None = (
        (
            await session.execute(
                select(Section.id)
                .where(Section.version_id == version_id)
                .where((Section.akn_eid == eid) | (Section.akn_wid == eid))
            )
        )
        .scalars()
        .first()
    )
    if exact is not None:
        return exact
    # No prefix pattern here (unlike provisions): a container ref should
    # exact-match its container, not prefix-match into a descendant.
    escaped = eid.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    candidates = (
        (
            await session.execute(
                select(Section.id)
                .where(Section.version_id == version_id)
                .where(
                    Section.akn_eid.like(f"%\\_\\_{escaped}", escape="\\")
                    | Section.akn_eid.like(f"%\\_\\_{escaped}\\_\\_%", escape="\\")
                )
                .limit(2)
            )
        )
        .scalars()
        .all()
    )
    return candidates[0] if len(candidates) == 1 else None


_GO_REF = re.compile(
    r"^/go/(?P<ref>[^/#?]+)$|^https?://zakon\.rada\.gov\.ua/laws/show/(?P<ref2>[^/#?]+)"
)


async def _registry_target(
    session: AsyncSession, version_id: uuid.UUID, uri: str
) -> tuple[bool, uuid.UUID | None]:
    """(is_registry_href, held-law id) for an upstream registry href
    such as rada's `/go/{ref}`."""
    from urllib.parse import unquote

    match = _GO_REF.match(uri)
    if match is None:
        return False, None
    ref_raw = unquote(match.group("ref") or match.group("ref2") or "")
    row = (
        await session.execute(
            text_clause(
                """
                SELECT rw.law_id
                FROM registry_works rw
                JOIN versions v ON v.id = :version_id
                JOIN laws l ON l.id = v.law_id
                WHERE rw.jurisdiction_id = l.jurisdiction_id
                  AND rw.ref = :ref
                  AND rw.law_id IS NOT NULL
                LIMIT 1
                """
            ),
            {"version_id": str(version_id), "ref": ref_raw},
        )
    ).first()
    return True, (row[0] if row is not None else None)


_QUOTING = frozenset({"quotedStructure", "embeddedStructure"})


@dataclass
class _Anchors:
    """A version's elements by eId and wId, and its rows by eId."""

    index: dict[str, etree._Element]
    rows: dict[str, tuple[str, uuid.UUID]]  # eId -> ("provision" | "section", id)
    placeholders: set[str]


def element_index(root: etree._Element) -> dict[str, etree._Element]:
    """Elements by eId and wId: the names a document answers an anchor by."""
    out: dict[str, etree._Element] = {}
    for e in root.iter():
        if isinstance(e.tag, str):
            for name in (e.get("eId"), e.get("wId")):
                if name:
                    out.setdefault(name, e)
    return out


def nearest_stored_row(
    index: dict[str, etree._Element],
    eid: str,
    rows: Container[str],
    placeholders: Container[str] = frozenset(),
) -> tuple[str, str] | None:
    """The stored row an anchor with no row of its own lands on, and how.

    A container stored only as its parts lands on its first stored part
    ("container"); an element inside a stored row, quoted amending text above
    all, lands on that row ("enclosing"), never on a part of the quoted text.
    An element's rows are its eId and the mapper's `__intro` and `__content`.
    """
    el = index.get(eid)
    if el is None:
        return None

    def row(e: etree._Element) -> str | None:
        name = e.get("eId")
        if not name:
            return None
        return next((c for c in (name, f"{name}__intro", f"{name}__content") if c in rows), None)

    name = el.get("eId") or eid
    # A placeholder is not an answer, and neither is a part of one.
    if any(c in placeholders for c in (name, f"{name}__intro", f"{name}__content")):
        return None
    ancestors = [a for a in el.iterancestors() if isinstance(a.tag, str)]
    if not any(etree.QName(a).localname in _QUOTING for a in ancestors):
        # Its own lead-in or text first, then its first stored part.
        if own := row(el):
            return own, "container"
        for d in el.iterdescendants():
            if isinstance(d.tag, str) and (hit := row(d)):
                return hit, "container"
    for a in ancestors:
        if hit := row(a):
            return hit, "enclosing"
    return None


async def _anchors(
    session: AsyncSession, version_id: uuid.UUID, cache: dict[str, _Anchors | None]
) -> _Anchors | None:
    """Read once per version. None where no AKN is stored or it does not parse:
    the document cannot then say whether it ever carried the anchor."""
    from codify.akn._schema import parse_xml

    key = str(version_id)
    if key in cache:
        return cache[key]
    xml = (
        await session.execute(
            text_clause("SELECT akn_xml FROM versions WHERE id = :v"), {"v": str(version_id)}
        )
    ).scalar_one_or_none()
    try:
        root = parse_xml(xml) if xml else None
    except (ValueError, etree.XMLSyntaxError):
        root = None
    if root is None:
        cache[key] = None
        return None
    provisions = (
        await session.execute(
            select(Provision.akn_eid, Provision.id, Provision.excluded_from_pool).where(
                Provision.version_id == version_id
            )
        )
    ).all()
    sections = (
        await session.execute(
            select(Section.akn_eid, Section.id).where(Section.version_id == version_id)
        )
    ).all()
    rows: dict[str, tuple[str, uuid.UUID]] = {e: ("section", i) for e, i in sections}
    rows.update({e: ("provision", i) for e, i, excluded in provisions if excluded is not True})
    cache[key] = _Anchors(
        # The same names the mapper reads, over the same parse.
        index=element_index(root),
        rows=rows,
        placeholders={e for e, _, excluded in provisions if excluded is True},
    )
    return cache[key]


async def resolve_references_for_version(
    session: AsyncSession, version_id: uuid.UUID
) -> ResolveStats:
    """Resolve the dangling cross_references sourced from `version_id`'s
    provisions. Idempotent; caller commits."""
    stats = ResolveStats()
    cache = _LawCache()
    anchors: dict[str, _Anchors | None] = {}
    rows = (
        await session.execute(
            select(CrossReference, Provision.text)
            .join(Provision, Provision.id == CrossReference.source_provision_id)
            .where(Provision.version_id == version_id)
            .where(CrossReference.target_uri.is_not(None))
            .where(
                # Never resolved, or resolved by a policy since superseded, or
                # stamped current and since gone dangling: `target_law_id` and
                # `target_section_id` are ON DELETE SET NULL, so deleting a
                # cited law empties the columns and leaves the stamp behind. A
                # row stamped at the current version that still points at
                # something is settled.
                col(CrossReference.resolver_version).is_(None)
                | (col(CrossReference.resolver_version) < RESOLVER_VERSION)
                | (
                    col(CrossReference.target_provision_id).is_(None)
                    & col(CrossReference.target_section_id).is_(None)
                    & col(CrossReference.target_law_id).is_(None)
                )
            )
        )
    ).all()
    for ref, source_text in rows:
        # Re-examined rows carry a verdict from the superseded policy. Clear it
        # before re-deciding, or a stale stamp survives its own re-run.
        ref.target_provision_id = None
        ref.target_section_id = None
        ref.target_law_id = None
        # Not stamped here. Stamping every examined row sealed the refusals: a
        # reference to an instrument the corpus did not hold yet would never be
        # looked at again, so acquiring it would not connect it. Only a row that
        # resolved is settled; an unresolved one stays in the selector.
        ref.resolver_version = None
        stats.examined += 1
        uri = ref.target_uri or ""
        # The writer recorded it. Fall back to the URI's shape only for rows
        # written before it did, and treat that as the guess it is: a relative
        # `/akn/...` href from a publisher is indistinguishable by string from
        # one of our own passes' minting.
        if ref.resolution_origin is None and uri and not uri.startswith("#"):
            ref.resolution_origin = _origin_of(uri)
        if uri.startswith("#"):
            # Same-document target the ingest-time rewrite missed.
            pid = await _provision_in_version(session, version_id, uri[1:])
            if pid is not None:
                ref.target_provision_id = pid
                stats.resolved_provision += 1
                continue
            sid = await _section_in_version(session, version_id, uri[1:])
            if sid is not None:
                ref.target_section_id = sid
                stats.resolved_section += 1
                continue
            doc = await _anchors(session, version_id, anchors)
            near = (
                nearest_stored_row(doc.index, uri[1:], doc.rows, doc.placeholders)
                if doc is not None
                else None
            )
            if doc is not None and near is not None:
                kind, row_id = doc.rows[near[0]]
                if kind == "section":
                    ref.target_section_id = row_id
                else:
                    ref.target_provision_id = row_id
                if near[1] == "container":
                    stats.resolved_container += 1
                else:
                    stats.resolved_enclosing += 1
                continue
            if doc is not None and uri[1:] not in doc.index:
                stats.anchor_not_in_document += 1
            else:
                stats.missing_same_doc += 1
            continue
        is_registry, registry_law = await _registry_target(session, version_id, uri)
        if is_registry:
            if registry_law is not None:
                ref.target_law_id = registry_law
                stats.resolved_registry += 1
            else:
                stats.unknown_law += 1
            continue
        parsed = parse_source_ref(uri)
        if parsed is None:
            stats.unparsed += 1
            continue
        work_uri, eid = parsed
        # A citation carrying the unknown-year placeholder names no enactment
        # date: its number identifies the work and its year is a guess. Matching
        # on year and number would resolve against a law homed under the same
        # placeholder by an undated ingest, so the year is dropped and the number
        # joins on its own.
        number_only = _NUMBER_ONLY.match(work_uri)
        if number_only and _numbers_are_unique(number_only.group(1)):
            base, series, number = number_only.groups()
            if series:
                # The citation names a series the corpus does not record: every
                # law is homed under
                # the bare doctype. Joining on the number alone would stamp an
                # edge onto whichever instrument shares it, which is a confident
                # false answer where a refusal is a true one.
                stats.series_not_recorded += 1
                continue
            if not _text_confirms_bare_series(base.split("/")[2], source_text, number):
                # The URI cannot say which instrument this is and the provision
                # does not agree that it is the plain series, so there is no
                # reading to resolve on.
                stats.series_unconfirmed += 1
                continue
            hits = await _law_by_number(session, cache, base, number)
            if not hits:
                stats.number_not_found += 1
                continue
            if len(hits) > 1:
                # Two laws under one number is a data defect, not a tie to break.
                stats.ambiguous_number += 1
                continue
            # Not re-gated here. This branch exists for a citation that
            # carries no year, so the number-and-year rule cannot fire by
            # construction, and the confirmation it does have is the right one:
            # `_text_confirms_bare_series` above made the provision agree that
            # this is the plain series, and the number matched one held law.
            ref.target_law_id = hits[0][0]
            stats.resolved_by_number += 1
            continue
        hit = await _law_latest_version(session, cache, work_uri)
        if hit is None:
            stats.unknown_law += 1
            continue
        law_id, target_version_id = hit
        # Before the unit branches: a unit of an unconfirmed law is an
        # unconfirmed stamp wearing a narrower target.
        if not await _law_is_named(session, cache, ref, law_id, source_text, work_uri, stats):
            continue
        if eid:
            pid = await _provision_in_version(session, target_version_id, eid)
            if pid is not None:
                ref.target_provision_id = pid
                stats.resolved_provision += 1
                continue
            sid = await _section_in_version(session, target_version_id, eid)
            if sid is not None:
                ref.target_section_id = sid
                stats.resolved_section += 1
                continue
            stats.fragment_fallback += 1  # unheld unit; fall back to the law
        ref.target_law_id = law_id
        stats.resolved_law += 1
    # Stamped after the branches rather than inside them: a row is settled if
    # it resolved, whichever rung resolved it, and a rung added later cannot
    # forget to say so. An unresolved row keeps a null version and stays in the
    # selector, so acquiring the missing instrument is enough to connect it.
    for ref, _ in rows:
        if ref.target_provision_id or ref.target_section_id or ref.target_law_id:
            ref.resolver_version = RESOLVER_VERSION
    await session.flush()
    logger.info("resolve_references", version_id=str(version_id), **stats.as_dict())
    return stats
