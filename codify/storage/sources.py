"""Source-document provenance, uploaded file hash/filename, for re-ingest detection."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from codify.storage.models import Law, SourceDocument, Version

# Matches `version_lineage`'s policy. `amend_provision` creates a child version
# per applied patch, so a low cap silently reports "no source" on a heavily
# amended law whose root upload is still there. The cap exists only to stop a
# cycle becoming a hang.
_LINEAGE_DEPTH_CAP = 100


async def upsert_source_document(
    session: AsyncSession,
    *,
    sha256: str,
    original_filename: str,
    byte_size: int,
    object_key: str,
    jurisdiction_code: str | None = None,
    page_count: int | None = None,
    pdf_title: str | None = None,
    pdf_author: str | None = None,
    pdf_producer: str | None = None,
    pdf_creation_date: datetime | None = None,
) -> None:
    """Record an uploaded source by content hash. Idempotent on sha256; later
    ingests fill in NULLs from earlier rows (PDF info-dict, page count)
    without overwriting non-NULL values. Caller commits."""
    stmt = pg_insert(SourceDocument).values(
        sha256=sha256,
        original_filename=original_filename,
        byte_size=byte_size,
        object_key=object_key,
        jurisdiction_code=jurisdiction_code,
        page_count=page_count,
        pdf_title=pdf_title,
        pdf_author=pdf_author,
        pdf_producer=pdf_producer,
        pdf_creation_date=pdf_creation_date,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["sha256"],
        set_={
            "page_count": func.coalesce(stmt.excluded.page_count, SourceDocument.page_count),
            "pdf_title": func.coalesce(stmt.excluded.pdf_title, SourceDocument.pdf_title),
            "pdf_author": func.coalesce(stmt.excluded.pdf_author, SourceDocument.pdf_author),
            "pdf_producer": func.coalesce(stmt.excluded.pdf_producer, SourceDocument.pdf_producer),
            "pdf_creation_date": func.coalesce(
                stmt.excluded.pdf_creation_date, SourceDocument.pdf_creation_date
            ),
        },
    )
    await session.execute(stmt)


async def existing_ingest_for_source(session: AsyncSession, sha256: str) -> dict[str, Any] | None:
    """The earliest version produced from this source file (with its law), or
    None if the file has not been ingested before. Drives upload-time
    duplicate detection."""
    row = (
        await session.execute(
            select(Version.id, Version.law_id, Law.title, SourceDocument.first_ingested_at)
            .join(Law, Law.id == Version.law_id)
            .join(SourceDocument, SourceDocument.sha256 == Version.source_sha256)
            .where(Version.source_sha256 == sha256)
            .order_by(Version.created_at)
            .limit(1)
        )
    ).first()
    if row is None:
        return None
    return {
        "version_id": str(row.id),
        "law_id": str(row.law_id),
        "title": row.title,
        "first_ingested_at": row.first_ingested_at.isoformat(),
    }


# A version's own source, else its span's, else the nearest ancestor's. Translations never carry
# a `source_sha256` of their own (the translation workflow saves without one), so
# without the walk the English expression of a scanned law can never reach the
# scan it came from, allowing reviewers to compare the source and extraction.
_NEAREST_SOURCE_SQL = """
WITH RECURSIVE lineage(origin, id, parent_version_id, source_sha256, depth) AS (
    SELECT v.id, v.id, v.parent_version_id, coalesce(v.source_sha256, s.source_sha256), 0
    FROM versions v LEFT JOIN source_spans s ON s.id = v.source_span_id
    WHERE v.id = ANY(:ids)
    UNION ALL
    SELECT l.origin, v.id, v.parent_version_id, coalesce(v.source_sha256, s.source_sha256),
           l.depth + 1
    FROM versions v JOIN lineage l ON v.id = l.parent_version_id
    LEFT JOIN source_spans s ON s.id = v.source_span_id
    WHERE l.source_sha256 IS NULL AND l.depth < {cap}
)
SELECT DISTINCT ON (origin) origin, source_sha256
FROM lineage WHERE source_sha256 IS NOT NULL
ORDER BY origin, depth ASC
"""


async def source_document_for_version(
    session: AsyncSession, version_id: uuid.UUID
) -> SourceDocument | None:
    """The retained upload this version came from, walking up to an ancestor's.

    None covers the cases the caller cannot tell apart and does not need to:
    unknown version, acquisition-adapter version (never had an upload), and a
    source row cleared by the FK's ON DELETE SET NULL."""
    sha = (
        await session.execute(
            text(_NEAREST_SOURCE_SQL.format(cap=_LINEAGE_DEPTH_CAP)),
            {"ids": [version_id]},
        )
    ).first()
    if sha is None:
        return None
    return (
        await session.execute(
            select(SourceDocument).where(SourceDocument.sha256 == sha.source_sha256)
        )
    ).scalar_one_or_none()


async def versions_with_available_source(
    session: AsyncSession, version_ids: Sequence[uuid.UUID]
) -> set[uuid.UUID]:
    """Which of these versions can serve an original, own or inherited.

    Batched so a versions listing costs one extra query rather than one per row."""
    if not version_ids:
        return set()
    rows = (
        await session.execute(
            text(_NEAREST_SOURCE_SQL.format(cap=_LINEAGE_DEPTH_CAP)),
            {"ids": list(version_ids)},
        )
    ).all()
    return {row.origin for row in rows}
