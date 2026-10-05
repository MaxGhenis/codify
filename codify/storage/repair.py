"""Storage reads for the agentic repair loop."""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from codify.repair.dossier import DossierInputs, EvidenceSource, PageReadInput, SpanTrim
from codify.storage.models import Law, SourceDocument, SourceSpan, Version
from codify.storage.page_reads import get_page_reads
from codify.storage.runs import get_latest_artifact_by_kind
from codify.storage.spans import span_for_version
from codify.storage.versions import get_version_source_text

logger = structlog.get_logger()


async def dossier_inputs_for_version(
    session: AsyncSession, version_id: uuid.UUID
) -> DossierInputs | None:
    """Evidence for a repair dossier, or None if the version is unknown.

    Preference order: `page_reads` (version-lifetime), then the ingest run's
    `page_texts` artifact (3-day window), then nothing. "missing" means a scan
    whose page evidence has aged out, the workflow may re-OCR; "none" means a
    text/html ingest that never had pages.
    """
    version = (
        await session.execute(select(Version).where(Version.id == version_id))
    ).scalar_one_or_none()
    if version is None:
        return None

    # A span-cut child, or its translation, reaches its upload through the span.
    span = None if version.source_sha256 else await span_for_version(session, version_id)
    source_sha = version.source_sha256 or (span.source_sha256 if span is not None else None)
    src = None
    if source_sha:
        src = (
            await session.execute(select(SourceDocument).where(SourceDocument.sha256 == source_sha))
        ).scalar_one_or_none()

    reads = await get_page_reads(session, version_id)
    trim = await _span_trim(session, version_id)
    stored_text = await get_version_source_text(session, version_id) or ""
    law = (await session.execute(select(Law).where(Law.id == version.law_id))).scalar_one_or_none()

    # The same dispatch ingest itself uses: only these suffixes skip page
    # extraction, so anything else with a source sha is a scan.
    object_key = src.object_key if src else ""
    is_textual = object_key.lower().endswith((".txt", ".html", ".htm"))

    fallback_text: str = ""
    fallback_spans: list[dict[str, Any]] = []
    source: EvidenceSource
    if reads:
        source = "page_reads"
    else:
        fallback_text, fallback_spans = await artifact_page_text(session, version_id)
        if fallback_spans:
            source = "artifact"
        elif source_sha and not is_textual:
            # Classified on the version's own sha, not the join: a dangling
            # source_documents row is still a scan whose evidence is gone, and
            # calling it "none" would assert it never had pages. Textual
            # uploads (.txt/.html) carry a sha too but never had pages, and
            # re-OCR on their bytes would fail.
            source = "missing"
            if src is None or not object_key:
                logger.warning(
                    "repair_source_object_gone",
                    version_id=str(version_id),
                    source_sha256=source_sha,
                )
        else:
            source = "none"

    return DossierInputs(
        version_id=str(version_id),
        akn_xml=version.akn_xml,
        country=(src.jurisdiction_code if src and src.jurisdiction_code else ""),
        doctype=(law.doctype if law and law.doctype else "act"),
        expression_uri=version.expression_uri,
        language=version.language,
        object_key=object_key,
        source_pdf_sha256=source_sha or "",
        page_evidence_source=source,
        page_reads=[
            PageReadInput(
                page_number=r.page_number,
                engine=r.engine,
                model=r.model,
                dpi=r.dpi,
                text=r.text,
                rival_text=r.rival_text,
                divergence=r.divergence,
                layout=r.layout,
                metrics=r.metrics,
            )
            for r in reads
        ],
        stored_source_text=stored_text,
        fallback_text=fallback_text,
        fallback_spans=fallback_spans,
        span_trim=trim,
    )


async def _span_trim(session: AsyncSession, version_id: uuid.UUID) -> SpanTrim | None:
    """Where a span-read version's text sits on its shared pages; None when it owns
    its reads or was never cut from a multi-act source."""
    owned = await session.execute(
        text("SELECT 1 FROM page_reads WHERE version_id = :vid LIMIT 1"), {"vid": version_id}
    )
    if owned.first() is not None:
        return None
    span = await span_for_version(session, version_id)
    if span is None:
        return None
    following = (
        await session.execute(
            select(SourceSpan)
            .where(SourceSpan.source_sha256 == span.source_sha256)
            .where(SourceSpan.generation == span.generation)
            .where(SourceSpan.ordinal == span.ordinal + 1)
        )
    ).scalar_one_or_none()
    shares = following is not None and following.first_page == span.last_page
    return SpanTrim(
        first_page=span.first_page,
        last_page=span.last_page,
        start_marker=span.start_marker,
        start_offset=span.start_offset,
        end_marker=following.start_marker if shares and following is not None else "",
        end_offset=span.end_offset,
    )


async def artifact_page_text(
    session: AsyncSession, version_id: uuid.UUID
) -> tuple[str, list[dict[str, Any]]]:
    """The ingest run's combined text + spans, while the artifact still exists."""
    run_id = (
        await session.execute(
            text(
                "SELECT id FROM runs WHERE kind = 'ingest' "
                "AND result->>'version_id' = :vid ORDER BY created_at DESC LIMIT 1"
            ).bindparams(vid=str(version_id))
        )
    ).scalar_one_or_none()
    if run_id is None:
        return "", []
    artifact = await get_latest_artifact_by_kind(session, run_id=run_id, kind="page_texts")
    if artifact is None:
        return "", []
    return artifact.content_text or "", list((artifact.content_json or {}).get("pages", []))
