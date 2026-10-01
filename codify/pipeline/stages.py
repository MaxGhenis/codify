"""Stage helpers shared by the event-generator pipeline and the durable ingest workflow."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import structlog

from codify.frbr import (
    UncitableFrbrUri,
    identity_from_title,
    law_number_token,
    series_number,
)
from codify.jurisdictions import (
    CONFIG_FAULTS,
    JurisdictionConfig,
    load_config,
    placeholder_statuses_for_code,
)
from codify.lang import to_iso639_3
from codify.pipeline.dating import resolve_dating
from codify.pipeline.enrich.akn_meta import normalise_akn_meta
from codify.pipeline.enrich.amendments import lift_amendment_markup
from codify.pipeline.enrich.asides import emit_marginal_notes
from codify.pipeline.enrich.cobalt import enrich_akn
from codify.pipeline.enrich.conclusions import emit_conclusions
from codify.pipeline.enrich.enacting import emit_enacting_formula
from codify.pipeline.enrich.hcontainers import postprocess_hcontainers
from codify.pipeline.enrich.inline_markup import emit_inline_markup
from codify.pipeline.enrich.metadata import number_from_title
from codify.pipeline.enrich.notes import emit_authorial_notes
from codify.pipeline.enrich.placeholder_status import mark_placeholder_status
from codify.pipeline.enrich.references import emit_references
from codify.pipeline.enrich.regions import (
    Region,
    RegionVocabulary,
    vocabulary_for_jurisdiction,
)

if TYPE_CHECKING:
    from codify.core.llm import LLMClient

logger = structlog.get_logger()


def resolve_year(metadata: dict[str, Any], country: str, *, title: str = "") -> str:
    """The Gregorian year for FRBR URI use. One reading of the shared
    resolution, so it cannot answer differently from the descriptor path."""
    return resolve_dating(metadata, country=country, title=title).uri_year


def _fold_language(raw: str | None, source: str, jurisdiction: str | None) -> str | None:
    """to_iso639_3 that logs and returns None on unfoldable values, so junk
    LLM detections and config sentinels ("both", "both_equal") never crash
    ingest; resolution falls through to the next candidate."""
    try:
        return to_iso639_3(raw) or None
    except ValueError:
        logger.warning("language_unrecognised", raw=raw, source=source, jurisdiction=jurisdiction)
        return None


def resolve_language(metadata: dict[str, Any], cfg: Any) -> str:
    """LLM-detected language → jurisdiction authoritative → the sole config
    language. A multilingual config can't disambiguate one expression, so it
    defaults to "eng" with a warning. Warns on off-corpus detections too, without
    blocking ingest."""
    jur = getattr(cfg, "code", None)
    detected = _fold_language(
        (metadata.get("language") or "").strip().lower() or None, "metadata", jur
    )
    if detected and cfg is not None and cfg.languages and detected not in cfg.languages:
        logger.warning(
            "metadata_language_off_corpus",
            detected=detected,
            jurisdiction_languages=cfg.languages,
        )
    if detected:
        return detected
    if cfg is not None:
        if cfg.authoritative_language:
            folded = _fold_language(cfg.authoritative_language, "authoritative_language", jur)
            if folded:
                return folded
        # Config can only name the language of a single-language jurisdiction;
        # a multilingual config list is unordered and cannot stand in for a
        # given expression's language, which callers supply via metadata.
        if cfg.languages and len(cfg.languages) == 1:
            folded = _fold_language(cfg.languages[0], "config_languages", jur)
            if folded:
                return folded
        if cfg.languages:
            logger.warning(
                "language_unresolved_defaulting_eng", jurisdiction_languages=cfg.languages
            )
    return "eng"


def draft_number(source: bytes | str) -> str:
    """Content-addressed FRBR number for un-numbered documents (drafts):
    deterministic across re-ingests so expression_uri idempotency holds.
    Accepts raw bytes or an already-computed sha256 hex digest."""
    sha = source if isinstance(source, str) else hashlib.sha256(source).hexdigest()
    return "draft-" + sha[:12]


@dataclass(frozen=True)
class Descriptors:
    """Everything parse/enrich need that comes out of metadata extraction."""

    title: str
    raw_date: str
    number: str
    year: str
    language: str
    doctype: str


# The enacting formula and any recitals sit at the top of the source, so a
# prefix is enough to test a preamble rule against without waiting for the
# document to be structured.
_PREAMBLE_PREFIX_CHARS = 2000


def preamble_from_akn(akn_xml: str) -> str | None:
    """The opening matter of a stored document, for classifying a law whose
    source text is gone. A derivative of the source preamble, not the source.
    ``None`` when the stored XML will not parse, so a caller counting how often
    the preamble decided a class cannot mistake unreadable for absent."""
    from lxml.etree import XMLSyntaxError

    from codify.akn._schema import AKN_NS, parse_xml

    if not akn_xml.strip():
        return ""
    try:
        root = parse_xml(akn_xml)
    except XMLSyntaxError as exc:
        logger.warning("akn_unparseable_for_classification", error=str(exc)[:200])
        return None
    # Preamble before preface: the classification rules match recitals, and a
    # long preface would otherwise fill the cap on its own.
    parts = [
        " ".join("".join(el.itertext()).split())
        for tag in ("preamble", "preface")
        for el in root.iter(f"{{{AKN_NS}}}{tag}")
    ]
    return "\n".join(p for p in parts if p)[:_PREAMBLE_PREFIX_CHARS]


def reclassify_stored_law(
    cfg: JurisdictionConfig | None,
    *,
    title: str,
    year: int | None,
    akn_xml: str,
) -> tuple[str | None, str]:
    """The class a stored law's own metadata implies, and the signal that
    decided it. ``None`` when nothing decided it, so the caller leaves the row
    alone rather than stamping it with the jurisdiction default, which is the
    silent fallback this reclassification exists to remove."""
    if cfg is None:
        return None, "no_config"
    if not title.strip():
        return None, "no_title"
    preamble = preamble_from_akn(akn_xml)
    if preamble is None:
        return None, "akn_unparseable"
    year_s = str(year or "")
    resolved = match_document_class(cfg, title=title, raw_date="", year=year_s, source=preamble)
    if resolved is None:
        return None, "no_rule_matched"
    if not preamble:
        return resolved, "title_and_year"
    bare = match_document_class(cfg, title=title, raw_date="", year=year_s, source="")
    return resolved, "preamble" if resolved != bare else "title_and_year"


def _unambiguous_date(raw_date: str, year: str) -> date | None:
    """The document's date, or ``None`` when only a year is known. A bare year cannot place a
    document either side of a bound falling inside it, and guessing 1 January would
    classify a whole year by the regime that ended in June.
    """
    try:
        return date.fromisoformat(raw_date[:10])
    except ValueError:
        return None


def resolve_doctype(
    cfg: JurisdictionConfig | None,
    *,
    title: str,
    raw_date: str,
    year: str,
    source: bytes | str,
    requested: str | None = None,
) -> str:
    """The caller's declared class, else a classification rule, else the default.

    The caller wins: overriding it structured a civil-law code as an act, where
    the chapter numbers collide. An undeclared class is refused, not ignored.
    """
    if cfg is None:
        return requested or "act"
    if requested:
        if requested not in cfg.document_classes:
            raise ValueError(
                f"jurisdiction {cfg.code!r} declares no document class {requested!r}; "
                f"known classes: {sorted(cfg.document_classes)}"
            )
        logger.info("doctype_from_caller", country=cfg.code, doctype=requested)
        return requested
    matched = match_document_class(cfg, title=title, raw_date=raw_date, year=year, source=source)
    if matched:
        logger.info("doctype_from_classification_rule", country=cfg.code, doctype=matched)
        return matched
    return cfg.default_document_class


def match_document_class(
    cfg: JurisdictionConfig,
    *,
    title: str,
    raw_date: str,
    year: str,
    source: bytes | str,
) -> str | None:
    """The class a rule chose, or None when none matched. Separated from
    `resolve_doctype` so a caller can tell a decision from a fallback."""
    text = source.decode("utf-8", "ignore") if isinstance(source, bytes) else source
    doc_date = _unambiguous_date(raw_date, year)
    if doc_date is None and year.isdigit():
        # A year alone still settles a bound that lies outside it entirely,
        # which is how era-bounded rules are written.
        doc_date = _year_only_date(cfg, int(year))
    return cfg.classify_document_class(
        title=title, preamble=text[:_PREAMBLE_PREFIX_CHARS], doc_date=doc_date
    )


def _year_only_date(cfg: JurisdictionConfig, year: int) -> date | None:
    """A date standing for the whole year, or ``None`` if any rule bound falls
    inside it and the year alone cannot decide which side the document is on."""
    rules = cfg.structuring.classification_rules if cfg.structuring else []
    for rule in rules:
        for bound in (rule.date_from, rule.date_until):
            if bound is not None and bound.year == year:
                return None
    return date(year, 1, 1)


def citable_number(value: str) -> str:
    """A path-safe law number from a model answer or a title, or "". A
    `{number}/{series}/{year}` citation is reduced first, so a court case number reaches
    the guard as the number it starts with rather than being discarded whole.
    """
    return law_number_token(series_number(value) or value)


def resolve_descriptors(
    metadata: dict[str, Any],
    *,
    jurisdiction_code: str,
    source_bytes: bytes | str,
    fallback_stem: str,
    requested_doctype: str | None = None,
    classification_text: str | None = None,
) -> Descriptors:
    model_title = str(metadata.get("title", "") or "")
    title = model_title or fallback_stem
    number = str(metadata.get("number") or "")
    cfg = load_config(jurisdiction_code)
    # The extracted text, never the source bytes: on the scanned route those are
    # the PDF file.
    source_text = classification_text if classification_text is not None else source_bytes
    dated = resolve_dating(
        metadata,
        country=jurisdiction_code,
        title=model_title,
        source_text=source_text if isinstance(source_text, str) else "",
        stem=fallback_stem,
    )
    year, raw_date = dated.uri_year, dated.raw_date
    # An instrument series that numbers nothing states its identity in its title,
    # which is the number this document is cited by.
    title_rule = cfg.frbr.title_identity if cfg.frbr is not None else None
    identity = identity_from_title(model_title, title_rule) if title_rule else None
    doctype = resolve_doctype(
        cfg,
        title=title,
        raw_date=raw_date,
        year=year,
        source=classification_text if classification_text is not None else source_bytes,
        requested=requested_doctype,
    )
    document_class = cfg.get_document_class(doctype)
    title_for_number = model_title
    if document_class and document_class.number_has_year_prefix:
        # Compact only numeric year/serial typography for this declared class.
        serial_pattern = r"(?<![\w-])([0-9]{4})\s*[-–]\s*([0-9]+)(?![\w-])"
        number = re.sub(serial_pattern, r"\1-\2", number)
        title_for_number = re.sub(serial_pattern, r"\1-\2", model_title)
    # The model sometimes returns the whole parenthetical a number sat in, routing
    # code and all. An unusable number is no number: fall through to the
    # content-addressed fallback rather than formatting it into an uncitable path.
    #
    # The title is a second source, as `date` is for the year, but only a title the
    # model actually read qualifies: a filename stem is not evidence of a number, and
    # an amending act's short title states the number of the act it amends.
    # The generic inference reads only the number inside a title, so every second
    # edition of a numberless series would collide. A stated number still wins.
    from_title = (
        ""
        if metadata.get("is_amendment") or identity is not None
        else number_from_title(title_for_number)
    )
    number = citable_number(number) or citable_number(from_title)
    # `4/2016` is a fine citation and a bad path segment: as the document's own
    # number it would split the FRBR path in two. A cited number keeps its
    # slash; this one cannot.
    number = number.replace("/", "-")
    if document_class and document_class.number_source == "title_identity":
        # A stated serial repeats across issuers and varies by run; the title does not.
        number = identity.segment if identity is not None else ""
    if not number and identity is not None:
        number = identity.segment
    if not number:
        number = draft_number(source_bytes)
    if document_class and document_class.number_has_year_prefix:
        prefix, separator, serial = number.partition("-")
        if separator and prefix == year and serial.isascii() and serial.isdecimal():
            number = serial
    language = resolve_language(metadata, cfg)
    return Descriptors(
        title=title,
        raw_date=raw_date,
        number=number,
        year=year,
        language=language,
        doctype=doctype,
    )


async def run_enrich_passes(
    akn_xml: str,
    *,
    llm: LLMClient,
    jurisdiction_code: str,
    desc: Descriptors,
    on_pass: Callable[[str], None] | None = None,
    skip_external_refs: bool = False,
    regions: dict[int, list[Region]] | None = None,
    work_uri: str | None = None,
) -> str:
    """The enrich sequence, each pass independent; failures log and skip so partial
    enrichment still ships a valid Document. `work_uri` pins identity to one the caller
    already holds, for a re-read of a stored law; absent it, cobalt derives as before.
    """
    try:
        vocab = vocabulary_for_jurisdiction(jurisdiction_code, desc.year)
    except CONFIG_FAULTS:
        raise
    except Exception as exc:  # noqa: BLE001
        # A bad config pattern costs the two region passes, not the sequence.
        logger.warning("region_vocabulary_failed", jurisdiction=jurisdiction_code, error=str(exc))
        vocab = RegionVocabulary()
    sync_passes: list[tuple[str, Callable[[str], str]]] = [
        (
            "cobalt",
            lambda x: enrich_akn(
                x,
                title=desc.title,
                country=jurisdiction_code,
                doctype=desc.doctype,
                year=desc.year,
                number=desc.number,
                work_uri=work_uri,
            ),
        ),
        # After cobalt, which rewrites `FRBRWork/FRBRdate` from the year-only work URI
        # and would undo this. `expression_undated` because nothing on this lane reads
        # an expression date off the source, so cobalt writes the day we ran.
        (
            "akn_meta",
            lambda x: normalise_akn_meta(x, work_date=desc.raw_date, expression_undated=True),
        ),
        # The same markers the row flag reads, so a unit the pool excludes says so
        # in the document too.
        (
            "placeholders",
            lambda x: mark_placeholder_status(x, placeholder_statuses_for_code(jurisdiction_code)),
        ),
        ("amendments", lift_amendment_markup),
        (
            "enacting",
            lambda x: emit_enacting_formula(
                x, jurisdiction_code, desc.doctype, desc.raw_date or None
            ),
        ),
        ("hcontainers", lambda x: postprocess_hcontainers(x, jurisdiction_code, desc.doctype)),
        ("references", lambda x: emit_references(x, jurisdiction_code)),
        # Both read regions and run before external refs, so a lifted note is
        # never linkified as though it were provision text.
        ("conclusions", lambda x: emit_conclusions(x, vocab=vocab, regions=regions)),
        ("notes", lambda x: emit_authorial_notes(x, vocab=vocab, regions=regions)),
        ("asides", lambda x: emit_marginal_notes(x, regions=regions)),
    ]
    for name, fn in sync_passes:
        try:
            akn_xml = fn(akn_xml)
        except UncitableFrbrUri:
            # Every other pass is decoration this document can ship without.
            # An identity nobody can cite is the document, so it propagates.
            raise
        except CONFIG_FAULTS:
            # So does a config nobody can read: every pass after it runs thinner.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("enrich_pass_failed", pass_name=name, error=str(exc))
            continue
        if on_pass is not None:
            on_pass(name)

    try:
        akn_xml = await emit_inline_markup(
            akn_xml,
            jurisdiction_code,
            desc.doctype,
            llm,
            skip_external_refs=skip_external_refs,
        )
    except CONFIG_FAULTS:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich_pass_failed", pass_name="inline_markup", error=str(exc))  # noqa: S106
    else:
        if on_pass is not None:
            on_pass("inline_markup")
    return akn_xml


__all__ = [
    "Descriptors",
    "citable_number",
    "draft_number",
    "resolve_descriptors",
    "match_document_class",
    "preamble_from_akn",
    "reclassify_stored_law",
    "resolve_doctype",
    "resolve_language",
    "resolve_year",
    "run_enrich_passes",
]
