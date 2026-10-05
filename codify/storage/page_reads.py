"""Per-page OCR reads, retained for the life of the version they explain."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from codify.storage.models import PageRead, SourceSpanPage
from codify.storage.spans import span_for_version, span_page_reads


async def save_page_reads(
    session: AsyncSession, *, run_id: uuid.UUID, reads: Sequence[PageRead]
) -> int:
    """Record what a run made of a document's pages. Caller commits.

    The last attempt wins as a set. The step that writes these retries, and the
    retry's text is what the workflow goes on to structure, so keeping the first
    attempt's rows would leave the trace explaining text nobody used.
    """
    if not reads:
        return 0
    await session.execute(text("DELETE FROM page_reads WHERE run_id = :rid"), {"rid": run_id})
    for read in reads:
        read.run_id = run_id
    session.add_all(list(reads))
    # Explicit: a raw-SQL execute does not autoflush, so without this the version
    # stamp in the same transaction would pass over rows still pending.
    await session.flush()
    return len(reads)


async def attach_page_reads_to_version(
    session: AsyncSession, *, run_id: uuid.UUID, version_id: uuid.UUID
) -> int:
    """Make a run's reads the version's reads, and return how many moved.

    Claim first, then drop any earlier run's reads for that version: after a
    re-OCR replaces the AKN in place the read behind the old text no longer
    explains the row, but a run that wrote nothing must not erase what it cannot
    replace. Zero is the normal answer on the lanes that never read pages.
    """
    result = await session.execute(
        text("UPDATE page_reads SET version_id = :vid WHERE run_id = :rid"),
        {"vid": version_id, "rid": run_id},
    )
    moved = int(result.rowcount or 0)
    if moved:
        await session.execute(
            text("DELETE FROM page_reads WHERE version_id = :vid AND run_id IS DISTINCT FROM :rid"),
            {"vid": version_id, "rid": run_id},
        )
    return moved


async def get_page_reads(session: AsyncSession, version_id: uuid.UUID) -> list[PageRead]:
    """Every page of a version, in page order. A version cut from a multi-act
    source reads the pages of its span, which carry the source's own page numbers."""
    rows = await session.execute(
        select(PageRead)
        .where(col(PageRead.version_id) == version_id)
        .order_by(col(PageRead.page_number))
    )
    own = list(rows.scalars().all())
    if own:
        return own
    span = await span_for_version(session, version_id)
    return await span_page_reads(session, span.id) if span is not None else []


async def get_page_read(
    session: AsyncSession, version_id: uuid.UUID, page_number: int
) -> PageRead | None:
    """One page's read, or None. Row-scoped: a 300-page scan's rows carry tens
    of MB of text and layout, so per-page callers must not load them all."""
    rows = await session.execute(
        select(PageRead)
        .where(col(PageRead.version_id) == version_id)
        .where(col(PageRead.page_number) == page_number)
    )
    own = rows.scalars().first()
    if own is not None:
        return own
    span = await span_for_version(session, version_id)
    if span is None:
        return None
    rows = await session.execute(
        select(PageRead)
        .join(SourceSpanPage, col(SourceSpanPage.page_read_id) == col(PageRead.id))
        .where(col(SourceSpanPage.span_id) == span.id)
        .where(col(PageRead.page_number) == page_number)
    )
    return rows.scalars().first()


async def count_disputes_by_page_read(
    session: AsyncSession, version_id: uuid.UUID
) -> dict[uuid.UUID, int]:
    """How many disputes each of a version's page reads carries, keyed by read id.

    Counts, not rows: the trace needs to know which pages a reviewer has already
    ruled on, and the rows themselves are fetched per page. A read nobody has
    ruled on is absent, so callers default it to zero.
    """
    span = await span_for_version(session, version_id)
    rows = await session.execute(
        text(
            "SELECT d.page_read_id, count(*) FROM page_read_disputes d "
            "JOIN page_reads p ON p.id = d.page_read_id "
            "WHERE p.version_id = :vid OR p.id IN "
            "(SELECT page_read_id FROM source_span_pages WHERE span_id = :sid) "
            "GROUP BY d.page_read_id"
        ),
        {"vid": version_id, "sid": span.id if span is not None else None},
    )
    return {row[0]: int(row[1]) for row in rows.all()}


__all__ = [
    "attach_page_reads_to_version",
    "count_disputes_by_page_read",
    "get_page_read",
    "get_page_reads",
    "save_page_reads",
]
