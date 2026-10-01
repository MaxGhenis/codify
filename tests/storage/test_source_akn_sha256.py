"""Integration tests for translation staleness: source_akn_sha256 stamping,
update_repaired_akn re-stamp, stale_translation_ids, and the SQL-digest vs
hashlib equivalence pin."""

from __future__ import annotations

import hashlib
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
from codify.storage.repository import save_document
from codify.storage.versions import stale_translation_ids, update_repaired_akn

pytestmark = pytest.mark.integration


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
        has_col = (
            await c.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name='versions' AND column_name='source_akn_sha256'"
                )
            )
        ).scalar()
        if has_col is None:
            await engine.dispose()
            pytest.skip("schema not migrated; run alembic upgrade head")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _akn(number: str, *, lang: str = "ara", body: str = "Body text.") -> str:
    bb = f"BODY\n  ARTICLE 1\n    {body}\n"
    return parse_to_akn(
        bb, country="xz", doctype="act", number=number, date="2020-01-01", language=lang
    )


def _sha(xml: str) -> str:
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()


async def _cleanup(session: AsyncSession, law_ids: list[uuid.UUID]) -> None:
    for law_id in law_ids:
        await session.execute(text("DELETE FROM laws WHERE id=:id"), {"id": law_id})
    await session.commit()


async def _law_id(session: AsyncSession, vid: uuid.UUID) -> uuid.UUID:
    row = await session.execute(text("SELECT law_id FROM versions WHERE id=:v"), {"v": vid})
    return row.scalar_one()


async def test_stale_lifecycle_and_hash_equivalence(session: AsyncSession) -> None:
    suffix = uuid.uuid4().hex[:8]
    source_xml = _akn(suffix)
    source_id = await save_document(
        session,
        parse_akn(source_xml),
        jurisdiction_code="xz",
        law_title=f"Stale {suffix}",
        akn_xml=source_xml,
    )
    await session.commit()
    law_id = await _law_id(session, source_id)

    try:
        # Translation stamped with the source hash at save time.
        translated_xml = _akn(suffix, lang="eng", body="Translated body.")
        target_id = await save_document(
            session,
            parse_akn(translated_xml).model_copy(update={"language": "eng"}),
            jurisdiction_code="xz",
            law_title=f"Stale {suffix}",
            akn_xml=translated_xml,
            parent_version_id=source_id,
            source_akn_sha256=_sha(source_xml),
        )
        await session.commit()

        stored = (
            await session.execute(
                text("SELECT source_akn_sha256 FROM versions WHERE id=:v"), {"v": target_id}
            )
        ).scalar_one()
        assert stored == _sha(source_xml)

        # Fresh: SQL digest of the parent equals the Python hash, so not stale.
        assert await stale_translation_ids(session, law_id) == set()

        # Repair the source: the translation flips stale.
        repaired_xml = source_xml.replace("Body text.", "Repaired body text.")
        await update_repaired_akn(session, source_id, repaired_xml, attribution="test")
        await session.commit()
        assert await stale_translation_ids(session, law_id) == {target_id}

        # Force-retranslate re-stamp clears it.
        await update_repaired_akn(
            session,
            target_id,
            translated_xml,
            attribution="translation_force_retranslate",
            source_akn_sha256=_sha(repaired_xml),
        )
        await session.commit()
        assert await stale_translation_ids(session, law_id) == set()

        # NULL-hash rows (pre-staleness translations) are unknown, never stale.
        legacy_xml = _akn(suffix, lang="heb", body="Legacy translated body.")
        legacy_id = await save_document(
            session,
            parse_akn(legacy_xml).model_copy(update={"language": "heb"}),
            jurisdiction_code="xz",
            law_title=f"Stale {suffix}",
            akn_xml=legacy_xml,
            parent_version_id=source_id,
        )
        await session.commit()
        assert legacy_id not in await stale_translation_ids(session, law_id)
    finally:
        await _cleanup(session, [law_id])


async def test_update_repaired_akn_leaves_hash_alone_when_not_passed(
    session: AsyncSession,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    xml = _akn(suffix)
    vid = await save_document(
        session,
        parse_akn(xml),
        jurisdiction_code="xz",
        law_title=f"StaleKeep {suffix}",
        akn_xml=xml,
        source_akn_sha256="deadbeef",
    )
    await session.commit()
    law_id = await _law_id(session, vid)
    try:
        await update_repaired_akn(session, vid, xml.replace("Body", "Fixed"), attribution="test")
        await session.commit()
        stored = (
            await session.execute(
                text("SELECT source_akn_sha256 FROM versions WHERE id=:v"), {"v": vid}
            )
        ).scalar_one()
        assert stored == "deadbeef"

        # Explicit None clears to unknown (legacy-artifact force resume).
        await update_repaired_akn(
            session, vid, xml.replace("Body", "Again"), attribution="test", source_akn_sha256=None
        )
        await session.commit()
        stored = (
            await session.execute(
                text("SELECT source_akn_sha256 FROM versions WHERE id=:v"), {"v": vid}
            )
        ).scalar_one_or_none()
        assert stored is None
    finally:
        await _cleanup(session, [law_id])
