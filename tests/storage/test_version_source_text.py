"""Source text and legibility survive the ingest that produced them. Without
these rows the structural checks that read the source abstain forever."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from codify.akn.io import parse_akn
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.storage.repository import save_document, supersede_document
from codify.storage.sources import upsert_source_document
from codify.storage.versions import get_version_source_text, save_version_source_text

pytestmark = pytest.mark.integration

_SOURCE = "مادة (1)\nنص الأولى.\n"


def _postgres_url() -> str:
    raw = os.environ.get("POSTGRES_URL", "postgresql://codify:codify@localhost:5432/codify")
    return raw.replace("postgresql://", "postgresql+asyncpg://", 1)


async def _reachable(url: str) -> bool:
    try:
        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect():
            pass
        await engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    url = _postgres_url()
    if not await _reachable(url):
        pytest.skip(f"postgres not reachable at {url}")
    engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
    async with engine.connect() as c:
        exists = (
            await c.execute(text("SELECT to_regclass('public.version_source_texts')"))
        ).scalar()
        if exists is None:
            await engine.dispose()
            pytest.skip("schema not migrated; run alembic upgrade head")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _akn(number: str) -> str:
    return parse_to_akn(
        "BODY\n  ARTICLE 1\n    Body text.\n",
        country="xz",
        doctype="act",
        number=number,
        date="2020-01-01",
        language="ara",
    )


async def _save(session: AsyncSession, **kw: object) -> uuid.UUID:
    suffix = uuid.uuid4().hex[:8]
    xml = _akn(suffix)
    return await save_document(
        session,
        parse_akn(xml),
        jurisdiction_code="xz",
        law_title=f"Source text {suffix}",
        akn_xml=xml,
        **kw,  # type: ignore[arg-type]
    )


async def _cleanup(session: AsyncSession, version_id: uuid.UUID) -> None:
    await session.execute(
        text("DELETE FROM laws WHERE id = (SELECT law_id FROM versions WHERE id = :v)"),
        {"v": version_id},
    )
    await session.commit()


async def test_an_ingest_retains_the_text_it_structured(session: AsyncSession) -> None:
    version_id = await _save(session, source_text=_SOURCE, source_legibility=131.5)
    await session.commit()
    try:
        assert await get_version_source_text(session, version_id) == _SOURCE
        stored = await session.execute(
            text("SELECT source_legibility FROM versions WHERE id = :v"), {"v": version_id}
        )
        assert stored.scalar_one() == pytest.approx(131.5)
    finally:
        await _cleanup(session, version_id)


async def test_a_version_with_no_source_text_stores_no_row(session: AsyncSession) -> None:
    """A translation has no extracted source, and an absent row says so. An
    empty string would read as a source that extracted to nothing."""
    version_id = await _save(session)
    await session.commit()
    try:
        assert await get_version_source_text(session, version_id) is None
        stored = await session.execute(
            text("SELECT source_legibility FROM versions WHERE id = :v"), {"v": version_id}
        )
        assert stored.scalar_one() is None
    finally:
        await _cleanup(session, version_id)


async def test_retaining_the_text_again_replaces_it(session: AsyncSession) -> None:
    """Re-OCR gives the same version better text; a second insert must not raise
    on the primary key."""
    version_id = await _save(session, source_text=_SOURCE)
    await session.commit()
    try:
        await save_version_source_text(session, version_id, "مادة (1)\nنص أوضح.\n")
        await session.commit()
        assert await get_version_source_text(session, version_id) == "مادة (1)\nنص أوضح.\n"
    finally:
        await _cleanup(session, version_id)


async def test_an_empty_extraction_is_stored_as_empty_not_absent(session: AsyncSession) -> None:
    """An OCR pass that returned nothing is a different state from a version
    that never had a source, and both checks read the absent row as the latter."""
    version_id = await _save(session, source_text="")
    await session.commit()
    try:
        assert await get_version_source_text(session, version_id) == ""
    finally:
        await _cleanup(session, version_id)


async def test_re_ingesting_the_same_file_backfills_the_source_text(
    session: AsyncSession,
) -> None:
    """`save_document` returns the existing version on a content-hash match. Not
    writing there made re-ingest, the operator's obvious backfill lever, a
    success that stored nothing."""
    suffix = uuid.uuid4().hex[:8]
    xml = _akn(suffix)
    await upsert_source_document(
        session,
        sha256=f"sha-{suffix}",
        original_filename=f"{suffix}.pdf",
        byte_size=1,
        object_key=f"uploads/{suffix}.pdf",
        jurisdiction_code="xz",
    )
    first = await save_document(
        session,
        parse_akn(xml),
        jurisdiction_code="xz",
        law_title=f"Backfill {suffix}",
        akn_xml=xml,
        source_sha256=f"sha-{suffix}",
    )
    await session.commit()
    try:
        assert await get_version_source_text(session, first) is None
        again = await save_document(
            session,
            parse_akn(xml),
            jurisdiction_code="xz",
            law_title=f"Backfill {suffix}",
            akn_xml=xml,
            source_sha256=f"sha-{suffix}",
            source_text=_SOURCE,
            source_legibility=118.0,
        )
        await session.commit()
        assert again == first
        assert await get_version_source_text(session, first) == _SOURCE
        stored = await session.execute(
            text("SELECT source_legibility FROM versions WHERE id = :v"), {"v": first}
        )
        assert stored.scalar_one() == pytest.approx(118.0)
    finally:
        await _cleanup(session, first)


async def test_superseding_carries_the_source_text_to_the_new_version(
    session: AsyncSession,
) -> None:
    """Supersede is the sanctioned re-ingest path. Dropping the kwargs there
    would reinstate the abstaining checks on exactly the corpus-wide lever."""
    suffix = uuid.uuid4().hex[:8]
    xml = _akn(suffix)
    first = await save_document(
        session,
        parse_akn(xml),
        jurisdiction_code="xz",
        law_title=f"Supersede {suffix}",
        akn_xml=xml,
    )
    await session.commit()
    replacement = (
        await supersede_document(
            session,
            parse_akn(xml),
            jurisdiction_code="xz",
            law_title=f"Supersede {suffix}",
            akn_xml=xml,
            source_text=_SOURCE,
            source_legibility=99.5,
        )
    ).version_id
    await session.commit()
    try:
        assert replacement != first
        assert await get_version_source_text(session, replacement) == _SOURCE
        stored = await session.execute(
            text("SELECT source_legibility FROM versions WHERE id = :v"), {"v": replacement}
        )
        assert stored.scalar_one() == pytest.approx(99.5)
    finally:
        await _cleanup(session, replacement)


async def test_deleting_the_law_takes_the_source_text_with_it(session: AsyncSession) -> None:
    """No FK cascade would leave the largest column in the schema orphaned."""
    version_id = await _save(session, source_text=_SOURCE)
    await session.commit()
    await _cleanup(session, version_id)
    assert await get_version_source_text(session, version_id) is None
