"""Native authoritative AKN ingest, deterministic, no LLM, no enrichment.

For publishers that serve Akoma Ntoso themselves (legislation.gov.uk,
Laws.Africa). The document arrives structured and reference-marked, so the
pipeline's job is normalisation, not extraction: canonicalise the FRBR URIs
(publisher scheme → our `/akn/...`) and synthesise eIds for the rare
unnumbered elements that would violate the provisions eId/wId constraints.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import structlog
from lxml import etree

from codify.akn._schema import parse_xml
from codify.akn.document import Document
from codify.akn.eids import ensure_unique_eids
from codify.akn.io import parse_akn
from codify.frbr import build_frbr_expression_uri, parse_source_ref
from codify.pipeline.events import Complete, Failed, IngestionEvent, Parsed, ValidationIssued

logger = structlog.get_logger()


def normalise_native_akn(doc: Document) -> tuple[Document, int]:
    """Canonicalise the document's FRBR URIs and fill missing eIds in place.

    The work URI is derived from the document's own FRBRWork value via the
    core source-href tables, so any publisher `parse_source_ref` understands
    normalises without caller-supplied identifiers. Returns the document and
    the count of synthesised eIds.
    """
    parsed = parse_source_ref(doc.frbr_work_uri)
    if parsed is not None:
        work, _ = parsed
        if work != doc.frbr_work_uri:
            doc.frbr_work_uri = work
            doc.frbr_expression_uri = build_frbr_expression_uri(
                work, doc.language, doc.expression_date
            )
    synth = 0
    seen: dict[str, int] = {}

    def _fill(el: object) -> None:
        nonlocal synth
        eid = getattr(el, "akn_eid", None)
        if not eid:
            synth += 1
            eid = f"codify-synth-{synth}"
        elif eid in seen:
            # Publishers occasionally repeat an eId; the first keeps it.
            seen[eid] += 1
            eid = f"{eid}-dup{seen[eid]}"
        seen.setdefault(eid, 0)
        el.akn_eid = eid  # type: ignore[attr-defined]
        if not getattr(el, "akn_wid", None):  # keep real source wIds
            el.akn_wid = eid  # type: ignore[attr-defined]
        for child in getattr(el, "children", []) or []:
            _fill(child)

    for top in doc.body:
        _fill(top)
    return doc, synth


def canonicalise_identification(xml: str, work: str, expression: str | None) -> str:
    """Rewrite the top-level FRBRWork and FRBRExpression `uri` values to ours,
    so the stored XML names the same work the row does; `this` keeps any
    component suffix it carried. Nested component identifications and the
    manifestation stay the publisher's."""
    root = parse_xml(xml)
    identification = root.find("./{*}*/{*}meta/{*}identification")
    if identification is None:
        return xml
    for level, value in (("FRBRWork", work), ("FRBRExpression", expression)):
        node = identification.find(f"{{*}}{level}")
        if node is None or not value:
            continue
        uri = node.find("{*}FRBRuri")
        this = node.find("{*}FRBRthis")
        old = uri.get("value", "") if uri is not None else ""
        if uri is not None:
            uri.set("value", value)
        if this is not None:
            old_this = this.get("value", "")
            suffix = old_this[len(old) :] if old and old_this.startswith(old) else ""
            this.set("value", value + suffix)
    return cast(str, etree.tostring(root, encoding="unicode"))


async def ingest(
    source: Path | str,
    jurisdiction_code: str,
) -> AsyncIterator[IngestionEvent]:
    """Read published AKN, normalise, validate, Complete. Fully deterministic."""
    from codify.pipeline.enrich.validator import validate_akn

    try:
        path = source if isinstance(source, Path) else Path(source)
        xml = path.read_text(encoding="utf-8")
        yield Parsed(akn_xml_len=len(xml))

        document = parse_akn(xml)
        published_work = document.frbr_work_uri
        document, synth = normalise_native_akn(document)
        if synth:
            yield ValidationIssued(issue={"synthesised_eids": synth})
        if document.frbr_work_uri != published_work:
            xml = canonicalise_identification(
                xml, document.frbr_work_uri, document.frbr_expression_uri
            )

        # Publishers reuse eIds (legislation.gov.uk repeats term-* on every
        # inline mention); persist a uniqueness-guaranteed copy so the stored
        # AKN is valid and importable.
        xml, deduped = ensure_unique_eids(xml)
        if deduped:
            yield ValidationIssued(issue={"deduped_eids": deduped})

        try:
            for issue in validate_akn(xml, provenance="native"):
                yield ValidationIssued(issue=issue)
        except Exception as exc:  # noqa: BLE001
            logger.warning("validator_failed", error=str(exc))
            yield Failed(stage="validator", error=f"{type(exc).__name__}: {exc}")
            return

        logger.info(
            "akn_native_ingested",
            jurisdiction=jurisdiction_code,
            frbr=document.frbr_work_uri,
            synthesised_eids=synth,
        )
        yield Complete(document=document, akn_xml=xml)
    except Exception as exc:  # noqa: BLE001
        yield Failed(stage="akn_native", error=f"{type(exc).__name__}: {exc}")


__all__ = ["canonicalise_identification", "ingest", "normalise_native_akn"]
