"""Build a bounded, immutable evidence dossier for one version."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from codify.akn.structure_diff import StructureNode, structure_of
from codify.jurisdictions import CONFIG_FAULTS
from codify.pipeline.enrich.ocr import PageLayout, PageResult, PageSpan, furniture_inline_patterns
from codify.pipeline.enrich.regions import classify_layouts, vocabulary_for_jurisdiction
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.edit_ops import ACTIONABLE_CHECKS
from codify.repair.grounding import combine_with_spans, eid_to_page, spans_to_json

logger = structlog.get_logger()

SCHEMA_VERSION = 1

# Where the per-page evidence came from. "none" = text/html ingest (no pages);
# "missing" = a scan whose page reads predate retention (re-OCR is the only way back).
EvidenceSource = Literal["page_reads", "artifact", "reocr", "none", "missing"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OutlineNode(BaseModel):
    """One skeleton element, serialisable (StructureNode is a frozen dataclass)."""

    eid: str
    tag: str
    parent_eid: str | None = None
    num: str | None = None
    heading: str | None = None


class PageEvidence(BaseModel):
    """One page's read, summarised. Full text and layout stay in `page_reads`."""

    page: int
    engine: str = ""
    model: str = ""
    dpi: int | None = None
    text_chars: int = 0
    divergence: float | None = None
    verdict: str = ""
    reasons: list[str] = Field(default_factory=list)
    has_layout: bool = False


class CoverageEntry(BaseModel):
    """One assessed area: a finding or a page, with the evidence-backed status."""

    status: Literal["confirmed", "suspected", "unresolved"]
    kind: str  # the validator check, or "page_read"
    eid: str | None = None
    page: int | None = None
    reason: str = ""


class RepairDossier(BaseModel):
    """Identity, hashes, structure, findings and coverage for one repair run."""

    schema_version: int = SCHEMA_VERSION
    version_id: str
    expression_uri: str = ""
    language: str = ""
    country: str = ""
    doctype: str = "act"
    akn_sha256: str
    source_text_sha256: str = ""
    source_pdf_sha256: str = ""
    object_key: str = ""
    page_evidence_source: EvidenceSource = "none"
    # Rebuilt combined text differs from version_source_texts, evidence drifted.
    source_text_mismatch: bool = False
    outline: list[OutlineNode] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    spans: list[dict[str, Any]] = Field(default_factory=list)
    eid_page: dict[str, int] = Field(default_factory=dict)
    pages: list[PageEvidence] = Field(default_factory=list)
    flagged_regions: list[dict[str, Any]] = Field(default_factory=list)
    coverage: list[CoverageEntry] = Field(default_factory=list)
    # The jurisdiction's attestation vocabulary, resolved once at assembly so
    # the loop and apply layer never re-load config.
    closing_phrases: list[str] = Field(default_factory=list)

    def coverage_counts(self) -> dict[str, int]:
        out = {"confirmed": 0, "suspected": 0, "unresolved": 0}
        for entry in self.coverage:
            out[entry.status] += 1
        return out

    def summary(self) -> dict[str, Any]:
        """The artifact/result payload: everything but the bulk lists."""
        return self.model_dump(exclude={"spans", "eid_page"})


@dataclass
class DossierInputs:
    """Everything assembly needs, gathered by `storage.repair` in one read."""

    version_id: str
    akn_xml: str
    country: str
    doctype: str = "act"
    expression_uri: str = ""
    language: str = ""
    object_key: str = ""
    source_pdf_sha256: str = ""
    page_evidence_source: EvidenceSource = "none"
    page_reads: list[PageReadInput] = dc_field(default_factory=list)
    stored_source_text: str = ""
    # Only populated on the artifact fallback (pre-page_reads versions).
    fallback_text: str = ""
    fallback_spans: list[dict[str, Any]] = dc_field(default_factory=list)


class PageReadInput(BaseModel):
    """The slice of a `page_reads` row assembly consumes (storage stays out of codify.repair)."""

    page_number: int
    engine: str = ""
    model: str = ""
    dpi: int | None = None
    text: str = ""
    rival_text: str = ""
    divergence: float | None = None
    layout: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


def _page_evidence(read: PageReadInput) -> PageEvidence:
    metrics = read.metrics or {}
    return PageEvidence(
        page=read.page_number,
        engine=read.engine,
        model=read.model,
        dpi=read.dpi,
        text_chars=len(read.text),
        divergence=read.divergence,
        verdict=str(metrics.get("verdict", "")),
        reasons=[str(r) for r in metrics.get("reasons", [])],
        has_layout=read.layout is not None,
    )


def _flagged_regions(
    reads: list[PageReadInput], *, country: str, year: str
) -> list[dict[str, Any]]:
    """Region flags recomputed from persisted layouts; empty when nothing is
    classifiable. Regions are advisory evidence, so only a config fault raises."""
    layouts: dict[int, PageLayout] = {}
    for read in reads:
        if not read.layout:
            continue
        try:
            layouts[read.page_number] = PageLayout.model_validate(read.layout)
        except Exception as exc:  # noqa: BLE001, a drifted layout is absent evidence
            logger.debug("dossier_layout_unreadable", page=read.page_number, error=str(exc))
    if not layouts:
        return []
    try:
        vocab = vocabulary_for_jurisdiction(country, year)
        regions = classify_layouts(layouts, vocab=vocab)
    except CONFIG_FAULTS:
        raise
    except Exception as exc:  # noqa: BLE001, silence here would read as "all pages clean"
        logger.warning("dossier_regions_failed", country=country, year=year, error=str(exc))
        return []
    return [
        {
            "page": region.page_number,
            "block_index": region.block_index,
            "kind": region.kind,
            "block_type": region.block_type,
            "signals": list(region.signals),
            "detail": region.detail,
        }
        for page_regions in regions.values()
        for region in page_regions
        if region.flagged
    ]


def classify_coverage(
    findings: list[dict[str, Any]], pages: list[PageEvidence]
) -> list[CoverageEntry]:
    """Assessment: confirmed = actionable finding with a target; suspected =
    located but not yet repairable, or a page whose read verdict is not accept;
    unresolved = a finding no evidence can place."""
    entries: list[CoverageEntry] = []
    for finding in findings:
        check = str(finding.get("check", ""))
        eid = finding.get("eid") or (finding.get("eids") or [None])[0]
        if eid is None:
            entries.append(
                CoverageEntry(status="unresolved", kind=check, reason="no target element")
            )
        elif check == "number_gap" and finding.get("likely") == "defect":
            entries.append(
                CoverageEntry(
                    status="suspected",
                    kind=check,
                    eid=str(eid),
                    reason="structurer/OCR miss upstream; not a repair",
                )
            )
        elif check in ACTIONABLE_CHECKS:
            entries.append(CoverageEntry(status="confirmed", kind=check, eid=str(eid)))
        else:
            entries.append(
                CoverageEntry(
                    status="suspected", kind=check, eid=str(eid), reason="no safe operation"
                )
            )
    for page in pages:
        if page.verdict and page.verdict != "accept":
            entries.append(
                CoverageEntry(
                    status="suspected",
                    kind="page_read",
                    page=page.page,
                    reason=",".join(page.reasons) or page.verdict,
                )
            )
    return entries


def _closing_phrases(country: str, year: str) -> list[str]:
    """The jurisdiction's attestation vocabulary; empty disables the check."""
    if not country:
        return []
    try:
        return list(vocabulary_for_jurisdiction(country, year).closing_phrases)
    except CONFIG_FAULTS:
        raise
    except Exception as exc:  # noqa: BLE001, disables the check, but never silently
        logger.warning("closing_phrases_unavailable", country=country, error=str(exc))
        return []


_YEAR_RE = re.compile(r"/(\d{4})[-/]")


def _year_from_uri(expression_uri: str) -> str:
    match = _YEAR_RE.search(expression_uri)
    return match.group(1) if match else ""


def assemble_dossier(inputs: DossierInputs) -> tuple[RepairDossier, str]:
    """Build the dossier and its combined source text, deterministically.

    With page reads, the combined text and spans are rebuilt from the per-page
    texts (`combine_with_spans`; the cleaning steps are idempotent, so pages
    with text reproduce the ingest-time combine, a page whose read came back
    empty gets no span, and any resulting drift from the stored flat text is
    flagged, never fatal). Without them, the caller-supplied fallback stands.
    """
    reads = inputs.page_reads
    if reads:
        # Without the jurisdiction's furniture the combine here would not
        # reproduce the ingest-time one, which this docstring promises it does.
        furniture = furniture_inline_patterns(inputs.country)
        results = [
            PageResult(page_number=r.page_number, text=r.text, method=r.engine, furniture=furniture)
            for r in reads
        ]
        doc_text, span_models = combine_with_spans(results)
        spans = spans_to_json(span_models)
    else:
        doc_text = inputs.fallback_text or inputs.stored_source_text
        spans = list(inputs.fallback_spans)
    mismatch = (
        bool(inputs.stored_source_text) and bool(doc_text) and inputs.stored_source_text != doc_text
    )
    if mismatch:
        # The AKN was structured from the stored text; grounding and page maps
        # run on the rebuilt one. Drift here must be visible, not archival.
        logger.warning(
            "dossier_source_text_mismatch",
            version_id=inputs.version_id,
            page_evidence_source=inputs.page_evidence_source,
        )

    country, doctype = inputs.country, inputs.doctype
    closing_phrases = _closing_phrases(country, _year_from_uri(inputs.expression_uri))
    findings = validate_akn(
        inputs.akn_xml,
        source_text=doc_text or None,
        closing_phrases=closing_phrases,
    )
    nodes: tuple[StructureNode, ...] = structure_of(inputs.akn_xml) or ()
    eid_page = (
        eid_to_page(
            doc_text,
            [PageSpan.model_validate(x) for x in spans],
            country=country,
            doctype=doctype,
        )
        if doc_text and spans
        else {}
    )
    pages = [_page_evidence(r) for r in reads]
    dossier = RepairDossier(
        version_id=inputs.version_id,
        expression_uri=inputs.expression_uri,
        language=inputs.language,
        country=country,
        doctype=doctype,
        akn_sha256=sha256_text(inputs.akn_xml),
        source_text_sha256=sha256_text(doc_text) if doc_text else "",
        source_pdf_sha256=inputs.source_pdf_sha256,
        object_key=inputs.object_key,
        page_evidence_source=inputs.page_evidence_source,
        source_text_mismatch=mismatch,
        outline=[
            OutlineNode(eid=n.eid, tag=n.tag, parent_eid=n.parent_eid, num=n.num, heading=n.heading)
            for n in nodes
        ],
        findings=findings,
        spans=spans,
        eid_page=eid_page,
        pages=pages,
        flagged_regions=_flagged_regions(
            reads, country=country, year=_year_from_uri(inputs.expression_uri)
        ),
        coverage=[],
        closing_phrases=closing_phrases,
    )
    dossier.coverage = classify_coverage(findings, pages)
    return dossier, doc_text
