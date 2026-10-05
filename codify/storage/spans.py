"""The acts one uploaded source holds, kept as spans over it, and the page reads
each span draws on. A span's version reads its source through the span: the
pages it covers, trimmed at its cuts."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from codify.pipeline.span_cuts import SpanCut
from codify.storage.models import PageRead, SourceSpan, SourceSpanPage

# Matches the source lineage walk: a translation carries no span of its own.
_LINEAGE_DEPTH_CAP = 100


async def write_span_generation(
    session: AsyncSession,
    *,
    source_sha256: str,
    cuts: Sequence[SpanCut],
    page_reads: Mapping[int, uuid.UUID] | None = None,
) -> list[SourceSpan]:
    """Retire the source's live spans and record `cuts` as the next generation.

    Never deletes: a re-split keeps the history it supersedes. `page_reads` maps a
    page number to the read each span covering that page draws on. Caller commits.
    """
    # Serialises generation numbering per source; the row exists before any split.
    await session.execute(
        text("SELECT 1 FROM source_documents WHERE sha256 = :sha FOR UPDATE"),
        {"sha": source_sha256},
    )
    latest = (
        await session.execute(
            text(
                "SELECT coalesce(max(generation), 0) FROM source_spans WHERE source_sha256 = :sha"
            ),
            {"sha": source_sha256},
        )
    ).scalar_one()
    await session.execute(
        text(
            "UPDATE source_spans SET retired_at = now() "
            "WHERE source_sha256 = :sha AND retired_at IS NULL"
        ),
        {"sha": source_sha256},
    )
    spans = [
        SourceSpan(
            source_sha256=source_sha256,
            generation=int(latest) + 1,
            ordinal=ordinal,
            kind=cut.kind,
            first_page=cut.first_page,
            last_page=cut.last_page,
            start_marker=cut.start_marker,
            start_offset=cut.start_offset,
            end_offset=cut.end_offset,
            issue=cut.issue,
            heading=cut.heading,
            act_key=cut.act_key,
            signals=list(cut.signals),
            reason=cut.reason,
        )
        for ordinal, cut in enumerate(cuts)
    ]
    session.add_all(spans)
    await session.flush()
    links = [
        SourceSpanPage(span_id=span.id, page_read_id=read_id)
        for span in spans
        for page, read_id in sorted((page_reads or {}).items())
        if span.first_page <= page <= span.last_page
    ]
    session.add_all(links)
    await session.flush()
    return spans


async def live_spans(session: AsyncSession, source_sha256: str) -> list[SourceSpan]:
    """The source's current split, in reading order; empty when it was never split."""
    rows = await session.execute(
        select(SourceSpan)
        .where(col(SourceSpan.source_sha256) == source_sha256)
        .where(col(SourceSpan.retired_at).is_(None))
        .order_by(col(SourceSpan.ordinal))
    )
    return list(rows.scalars().all())


# A version's own span, else the nearest ancestor's: a translation of a child
# version carries none of its own.
_NEAREST_SPAN_SQL = """
WITH RECURSIVE lineage(id, parent_version_id, source_span_id, depth) AS (
    SELECT id, parent_version_id, source_span_id, 0 FROM versions WHERE id = :vid
    UNION ALL
    SELECT v.id, v.parent_version_id, v.source_span_id, l.depth + 1
    FROM versions v JOIN lineage l ON v.id = l.parent_version_id
    WHERE l.source_span_id IS NULL AND l.depth < {cap}
)
SELECT source_span_id FROM lineage
WHERE source_span_id IS NOT NULL ORDER BY depth LIMIT 1
"""


async def span_for_version(session: AsyncSession, version_id: uuid.UUID) -> SourceSpan | None:
    """The span this version was cut from, walking up to an ancestor's; None for a
    version whose source was never split."""
    span_id = (
        await session.execute(
            text(_NEAREST_SPAN_SQL.format(cap=_LINEAGE_DEPTH_CAP)), {"vid": version_id}
        )
    ).scalar_one_or_none()
    if span_id is None:
        return None
    return await session.get(SourceSpan, span_id)


async def span_page_reads(session: AsyncSession, span_id: uuid.UUID) -> list[PageRead]:
    """The page reads a span draws on, in page order."""
    rows = await session.execute(
        select(PageRead)
        .join(SourceSpanPage, col(SourceSpanPage.page_read_id) == col(PageRead.id))
        .where(col(SourceSpanPage.span_id) == span_id)
        .order_by(col(PageRead.page_number))
    )
    return list(rows.scalars().all())


__all__ = [
    "SpanCut",
    "live_spans",
    "span_for_version",
    "span_page_reads",
    "write_span_generation",
]
