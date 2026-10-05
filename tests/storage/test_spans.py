"""Spans over a multi-act source: generations, the version they name, and the
page reads a span's version reads through it."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from codify.pipeline.span_cuts import SpanCut
from codify.storage.models import Jurisdiction, Law, PageRead, SourceDocument, SourceSpan, Version
from codify.storage.page_reads import count_disputes_by_page_read, get_page_read, get_page_reads
from codify.storage.spans import (
    live_spans,
    span_for_version,
    span_page_reads,
    write_span_generation,
)

pytestmark = pytest.mark.integration

_AKN = "<akomaNtoso xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'><act/></akomaNtoso>"


def _url() -> str:
    raw = os.environ.get("POSTGRES_URL", "postgresql://codify:codify@localhost:5432/codify")
    return raw.replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(_url())
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()
    await engine.dispose()


async def _source(session: AsyncSession, pages: int) -> tuple[str, dict[int, uuid.UUID]]:
    """A three-page source and one read per page."""
    sha = uuid.uuid4().hex * 2
    session.add(
        SourceDocument(
            sha256=sha, original_filename="issue.pdf", byte_size=1, object_key=f"k/{sha}"
        )
    )
    await session.flush()
    reads = {n: PageRead(page_number=n, text=f"page {n}") for n in range(1, pages + 1)}
    session.add_all(reads.values())
    await session.flush()
    return sha, {n: r.id for n, r in reads.items()}


async def _version(
    session: AsyncSession, *, span_id: uuid.UUID | None, parent: uuid.UUID | None = None
) -> uuid.UUID:
    tag = uuid.uuid4().hex[:8]
    juris = Jurisdiction(code=f"zz-{tag}", name="Test", languages=["en"])
    session.add(juris)
    await session.flush()
    law = Law(
        jurisdiction_id=juris.id, title=f"Act {tag}", doctype="act", frbr_work_uri=f"/akn/zz/{tag}"
    )
    session.add(law)
    await session.flush()
    version = Version(
        law_id=law.id,
        expression_uri=f"/akn/zz/{tag}/eng@2024-01-01",
        language="eng",
        expression_date=date(2024, 1, 1),
        akn_xml=_AKN,
        source_span_id=span_id,
        parent_version_id=parent,
    )
    session.add(version)
    await session.flush()
    return version.id


def _cuts() -> list[SpanCut]:
    """Front matter on page 1, an act on pages 1-2, an act from mid-page 2 to page 3."""
    return [
        SpanCut("front_matter", 1, 1, 0, 40),
        SpanCut("act", 1, 2, 40, 120, start_marker="ACT No. 3 OF 2020", act_key="act 3"),
        SpanCut("act", 2, 3, 120, None, start_marker="ACT No. 4 OF 2020", act_key="act 4"),
    ]


async def test_a_generation_is_written_in_order_with_its_pages(session: AsyncSession) -> None:
    sha, reads = await _source(session, 3)
    spans = await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    assert [(s.generation, s.ordinal, s.kind, s.act_key) for s in spans] == [
        (1, 0, "front_matter", ""),
        (1, 1, "act", "act 3"),
        (1, 2, "act", "act 4"),
    ]
    # Page 2 is shared: both acts draw on its read.
    pages = [[r.page_number for r in await span_page_reads(session, s.id)] for s in spans]
    assert pages == [[1], [1, 2], [2, 3]]


async def test_a_re_split_retires_the_live_generation_and_keeps_it(
    session: AsyncSession,
) -> None:
    sha, reads = await _source(session, 3)
    first = await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    merged = [SpanCut("act", 1, 3, 0, None, start_marker="ACT No. 3 OF 2020", act_key="act 3")]
    second = await write_span_generation(session, source_sha256=sha, cuts=merged, page_reads=reads)
    live = await live_spans(session, sha)
    assert [(s.id, s.generation) for s in live] == [(second[0].id, 2)]
    kept = (
        await session.execute(
            text("SELECT count(*) FROM source_spans WHERE source_sha256 = :sha"), {"sha": sha}
        )
    ).scalar_one()
    assert kept == len(first) + len(second)
    for span in first:
        await session.refresh(span)
    assert all(s.retired_at is not None for s in first)


async def test_a_child_and_its_translation_read_the_spans_pages(session: AsyncSession) -> None:
    sha, reads = await _source(session, 3)
    spans = await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    child = await _version(session, span_id=spans[2].id)
    translation = await _version(session, span_id=None, parent=child)
    for version in (child, translation):
        found = await span_for_version(session, version)
        assert found is not None and found.id == spans[2].id
        assert [r.page_number for r in await get_page_reads(session, version)] == [2, 3]
    assert (await get_page_read(session, child, 3)) is not None
    assert await get_page_read(session, child, 1) is None


async def test_a_version_never_split_has_no_span(session: AsyncSession) -> None:
    version = await _version(session, span_id=None)
    assert await span_for_version(session, version) is None
    assert await get_page_reads(session, version) == []


async def test_a_versions_own_reads_come_before_its_spans(session: AsyncSession) -> None:
    sha, reads = await _source(session, 3)
    spans = await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    child = await _version(session, span_id=spans[1].id)
    session.add(PageRead(page_number=7, text="own", version_id=child))
    await session.flush()
    assert [r.page_number for r in await get_page_reads(session, child)] == [7]


async def test_a_dispute_on_a_span_page_counts_for_the_child(session: AsyncSession) -> None:
    sha, reads = await _source(session, 3)
    spans = await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    child = await _version(session, span_id=spans[2].id)
    await session.execute(
        text(
            "INSERT INTO page_read_disputes (id, page_read_id, verdict, actor) "
            "VALUES (:id, :read, 'disputed', 'reviewer')"
        ),
        {"id": uuid.uuid4(), "read": reads[3]},
    )
    assert await count_disputes_by_page_read(session, child) == {reads[3]: 1}


async def test_spans_are_unique_per_generation_and_ordinal(session: AsyncSession) -> None:
    sha, _reads = await _source(session, 1)
    session.add_all(
        [
            SourceSpan(
                source_sha256=sha,
                generation=1,
                ordinal=0,
                kind="act",
                first_page=1,
                last_page=1,
                start_offset=0,
            ),
            SourceSpan(
                source_sha256=sha,
                generation=1,
                ordinal=0,
                kind="act",
                first_page=1,
                last_page=1,
                start_offset=0,
            ),
        ]
    )
    with pytest.raises(Exception, match="source_spans_generation_ordinal_key"):
        await session.flush()


async def test_a_read_a_span_draws_on_cannot_be_deleted(session: AsyncSession) -> None:
    sha, reads = await _source(session, 3)
    await write_span_generation(session, source_sha256=sha, cuts=_cuts(), page_reads=reads)
    with pytest.raises(Exception, match="source_span_pages"):
        await session.execute(text("DELETE FROM page_reads WHERE id = :id"), {"id": reads[2]})
