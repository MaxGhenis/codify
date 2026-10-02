"""Cobalt wrapper: enrich AKN XML with metadata."""

from datetime import date
from typing import cast

import structlog
from cobalt.hierarchical import Act
from lxml import etree

from codify.frbr import UncitableFrbrUri, build_frbr_work_uri

logger = structlog.get_logger()


def set_work_uri(
    akn_xml: str,
    work_uri: str,
    expression_uri: str | None = None,
    expression_date: str | None = None,
) -> str | None:
    """Restate a document's identity from the caller's row. None if it already matches;
        raises where Cobalt cannot restate.

        Both halves come from the row, the expression being `versions.expression_uri`, never
    the document: Cobalt rebuilds `@date` from
        `FRBRExpression/FRBRdate`, which on a scanned corpus is the pipeline's generation
        date, and one UA row's language disagrees with its own expression. `expression_date`
        is written to that element and its manifestation twin, or Cobalt regenerates the old
        value and reverts the repair on the next write.

        `expression_uri` of None restates the work block only. A work date more precise than
        the URI's year is restored.
    """
    if not akn_xml.strip():
        # `Act("")` returns a synthetic empty act rather than raising, so an
        # empty row would be "repaired" into a fabricated stub.
        raise ValueError("set_work_uri: refusing to restate an empty document")
    if expression_uri is not None and not expression_uri.startswith(f"{work_uri}/"):
        # Otherwise the document would assert an expression outside its own
        # work, which is the nesting this repair exists to keep.
        raise ValueError(
            f"set_work_uri: expression {expression_uri!r} is not under work {work_uri!r}"
        )

    before = cast(str, Act(akn_xml).to_xml(encoding="unicode"))
    act = Act(akn_xml)
    # Every `<identification>`, not just the root's: Cobalt's setter rewrites each
    # component's blocks from the generation-date-and-stale-language source this
    # exists to stop trusting. Correcting only the root left most multi-component
    # documents asserting two expressions of one work, at two dates, with
    # `/!schedule1` no longer resolving. `write.py:_patch_frbr_meta` walks them all.
    saved = [_snapshot(el) for el in _identifications(act)]

    act.frbr_uri = work_uri

    language = _language_of(expression_uri, work_uri) if expression_uri else ""
    for ident, state in zip(_identifications(act), saved, strict=True):
        _restate(
            ident,
            state,
            expression_uri=expression_uri,
            expression_date=expression_date,
            language=language,
            work_uri=work_uri,
        )

    out = cast(str, act.to_xml(encoding="unicode"))
    return out if out != before else None


def _identifications(act: Act) -> list[etree._Element]:
    return [
        el
        for el in act.root.iter()
        if isinstance(el.tag, str) and etree.QName(el).localname == "identification"
    ]


def _child(parent: etree._Element | None, name: str) -> etree._Element | None:
    """A direct child by local name. `iterchildren` rather than iterating `parent`: Cobalt
    returns `lxml.objectify` elements whose `__iter__` yields siblings sharing the tag,
    not children, so the obvious loop found nothing and the repair applied to no document
    at all.
    """
    if parent is None:
        return None
    for el in parent.iterchildren():
        if isinstance(el.tag, str) and etree.QName(el).localname == name:
            return el
    return None


def _snapshot(ident: etree._Element) -> dict[str, str | None]:
    """What one block said before Cobalt rewrote it: the tails to carry over and
    the work date, which Cobalt replaces with the URI's bare year."""
    state: dict[str, str | None] = {}
    for block in ("FRBRExpression", "FRBRManifestation"):
        block_el = _child(ident, block)
        for name in ("FRBRthis", "FRBRuri"):
            el = _child(block_el, name)
            state[f"{block}/{name}"] = None if el is None else el.get("value")
    work_date = _child(_child(ident, "FRBRWork"), "FRBRdate")
    state["work_date"] = None if work_date is None else work_date.get("date")
    return state


def _restate(
    ident: etree._Element,
    state: dict[str, str | None],
    *,
    expression_uri: str | None,
    expression_date: str | None,
    language: str,
    work_uri: str,
) -> None:
    was, now_el = state["work_date"], _child(_child(ident, "FRBRWork"), "FRBRdate")
    now = None if now_el is None else now_el.get("date")
    if was and now and now_el is not None:
        if was.startswith(f"{now}-"):
            now_el.set("date", was)
        elif was != now:
            # Forward-only storage, so this value survives nowhere else.
            logger.warning("frbr_work_date_discarded", was=was, now=now, work_uri=work_uri)

    if expression_uri is None:
        return

    for block in ("FRBRExpression", "FRBRManifestation"):
        block_el = _child(ident, block)
        for name in ("FRBRthis", "FRBRuri"):
            el = _child(block_el, name)
            if el is None:
                continue
            el.set("value", expression_uri + _component_tail(state[f"{block}/{name}"]))
        # The element, not only the URI. Cobalt rebuilds the URI's `@date` from
        # here, so leaving it stale reverts this repair on the next write.
        date_el = _child(block_el, "FRBRdate")
        if expression_date and date_el is not None:
            date_el.set("date", expression_date)

    # `FRBRlanguage`, not the URI, is what the parser and the validator read, so
    # moving the URI alone relocates a language mislabel rather than resolving
    # it. `write.py` sets both for the same reason.
    lang_el = _child(_child(ident, "FRBRExpression"), "FRBRlanguage")
    if language and lang_el is not None:
        lang_el.set("language", language)


# Manifestation formats this corpus emits. A bare `\.[a-z]+$` would also eat a
# number like `1.2` off a work URI, so the set is closed rather than inferred.
_MANIFESTATION_FORMATS = (".akn", ".xml", ".pdf", ".html", ".docx", ".epub", ".rtf")


def _language_of(expression_uri: str, work_uri: str) -> str:
    """The language segment an expression URI names, or "" if it names none."""
    tail = expression_uri[len(work_uri) :].lstrip("/")
    return tail.split("@", 1)[0].split("/", 1)[0]


def _component_tail(value: str | None) -> str:
    """What an FRBR URI addressed beyond the expression: a `/!component`, a manifestation
    format, or both. The format sits on the manifestation's `FRBRuri` with no `/!` marker
    (`…/eng@2016-05-04.xml`), so splitting on `/!` alone drops it. Measured on demo that
    is 970 versions, 961 of them the EU acquis.
    """
    raw = value or ""
    head, sep, component = raw.partition("/!")
    if sep:
        return f"/!{component}"
    for fmt in _MANIFESTATION_FORMATS:
        if head.endswith(fmt):
            return fmt
    return ""


def enrich_akn(
    akn_xml: str,
    title: str,
    country: str,
    doctype: str,
    year: str,
    number: str,
    publication_name: str | None = None,
    publication_date: date | None = None,
    publication_number: str | None = None,
    work_uri: str | None = None,
) -> str:
    """Enrich AKN XML with metadata using Cobalt: title, canonical work URI, the expression
    derived from it, and optional publication details. `work_uri` pins the identity, so a
    re-read changes text and never identity. A pin Cobalt refuses raises
    `UncitableFrbrUri`, the one exception `run_enrich_passes` propagates rather than
    logging past, which would drop the pin and the title with it.
    """
    act = Act(akn_xml)
    act.title = title

    # Canonical work URI, doctype-keyed so a VKM never collides with a ligj. A
    # year-less document still gets one from `build_frbr_work_uri`: Bluebell mints
    # `/akn/xz/act//draft-x` for an
    # empty date, and declining to overwrite is how that gap reaches storage.
    if work_uri:
        try:
            act.frbr_uri = work_uri
        except ValueError as exc:
            raise UncitableFrbrUri(f"stored work URI {work_uri!r} is unusable: {exc}") from exc
    elif number:
        act.frbr_uri = build_frbr_work_uri(country, doctype, year, number)

    if publication_name:
        act.publication_name = publication_name
    if publication_date:
        act.publication_date = publication_date
    if publication_number:
        act.publication_number = publication_number

    return cast(str, act.to_xml(encoding="unicode"))
