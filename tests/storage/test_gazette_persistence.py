"""Integration tests for the gazette persistence contract: a new work stores its
gazette, a reuse fills it only when null, and a reuse never clobbers a stored one.

The export tests build `Law(gazette=...)` directly, so they cannot catch a break in
this threading or the fill-if-null behaviour; these exercise the real write path."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

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
from codify.storage.repository import SaveOutcome, save_document_reporting

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
                    "WHERE table_name='laws' AND column_name='gazette'"
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
        bb, country="xz", doctype="act", number=number, date="2005-01-01", language=lang
    )


async def _law_id(session: AsyncSession, vid: uuid.UUID) -> uuid.UUID:
    row = await session.execute(text("SELECT law_id FROM versions WHERE id=:v"), {"v": vid})
    return row.scalar_one()


async def _gazette(session: AsyncSession, law_id: uuid.UUID) -> dict[str, Any] | None:
    row = await session.execute(text("SELECT gazette FROM laws WHERE id=:id"), {"id": law_id})
    return row.scalar_one()


async def _cleanup(session: AsyncSession, law_id: uuid.UUID) -> None:
    await session.execute(text("DELETE FROM laws WHERE id=:id"), {"id": law_id})
    await session.commit()


async def _save(
    session: AsyncSession,
    xml: str,
    num: str,
    *,
    lang: str,
    gazette: dict[str, Any] | None,
    parent: uuid.UUID | None = None,
) -> SaveOutcome:
    doc = parse_akn(xml).model_copy(update={"language": lang})
    return await save_document_reporting(
        session,
        doc,
        jurisdiction_code="xz",
        law_title=f"Gaz {num}",
        akn_xml=xml,
        parent_version_id=parent,
        gazette=gazette,
    )


async def test_gazette_stored_on_a_new_work_and_not_clobbered_on_reuse(
    session: AsyncSession,
) -> None:
    num = uuid.uuid4().hex[:8]
    first = {"name": "Gazette A", "year": 2005, "issue": "1", "page": "5"}
    src = _akn(num)
    outcome = await _save(session, src, num, lang="ara", gazette=first)
    await session.commit()
    law_id = await _law_id(session, outcome.version_id)
    try:
        assert await _gazette(session, law_id) == first
        # A later ingest of the same work carrying a different gazette must not win.
        tr = _akn(num, lang="eng", body="Translated.")
        await _save(
            session,
            tr,
            num,
            lang="eng",
            gazette={"name": "Gazette B", "year": 2005, "issue": "2"},
            parent=outcome.version_id,
        )
        await session.commit()
        assert await _gazette(session, law_id) == first
    finally:
        await _cleanup(session, law_id)


async def test_gazette_fills_a_null_work_on_reuse(session: AsyncSession) -> None:
    num = uuid.uuid4().hex[:8]
    src = _akn(num)
    outcome = await _save(session, src, num, lang="ara", gazette=None)
    await session.commit()
    law_id = await _law_id(session, outcome.version_id)
    try:
        assert await _gazette(session, law_id) is None
        later = {"name": "Gazette C", "year": 2005, "issue": "3"}
        tr = _akn(num, lang="eng", body="Translated.")
        await _save(session, tr, num, lang="eng", gazette=later, parent=outcome.version_id)
        await session.commit()
        assert await _gazette(session, law_id) == later
    finally:
        await _cleanup(session, law_id)
