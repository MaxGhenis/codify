"""Integration test for rederive_version_rows: repair must rebuild the derived
provision rows (search text + document-view skeleton), not just akn_xml."""

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
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.edit_ops import apply_op
from codify.repair.ops import SetBody
from codify.storage.repository import rederive_version_rows, save_document
from codify.storage.versions import update_repaired_akn

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
        if (await c.execute(text("SELECT to_regclass('provisions')"))).scalar() is None:
            await engine.dispose()
            pytest.skip("schema not migrated; run alembic upgrade head")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _provision_texts(session: AsyncSession, vid: uuid.UUID) -> dict[str, str]:
    rows = (
        await session.execute(
            text("SELECT akn_eid, text FROM provisions WHERE version_id=:v"), {"v": vid}
        )
    ).all()
    return {r[0]: (r[1] or "") for r in rows}


async def test_rederive_rebuilds_provisions_after_repair(session: AsyncSession) -> None:
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      Body of article one.\n    ARTICLE 2\n"
    suffix = uuid.uuid4().hex[:8]
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    empty = next(i for i in validate_akn(akn) if i["check"] == "empty_article")

    vid = await save_document(
        session, parse_akn(akn), jurisdiction_code="xz", law_title=f"Rederive {suffix}", akn_xml=akn
    )
    await session.commit()
    before = await _provision_texts(session, vid)
    assert not any("art_2" in e for e in before)  # empty article has no content provision

    res = apply_op(
        akn,
        SetBody(eid=empty["eid"], bluebell="The restored body of the second article."),
        country="xz",
        target_finding=empty,
    )
    assert res.ok, res.error

    await update_repaired_akn(session, vid, res.xml, attribution="test")
    await rederive_version_rows(session, vid, res.xml)
    await session.commit()

    after = await _provision_texts(session, vid)
    art2 = next((t for e, t in after.items() if "art_2" in e), "")
    assert "restored body of the second article" in art2  # rows reflect the repair
    embedded_at = (
        await session.execute(text("SELECT embedded_at FROM versions WHERE id=:v"), {"v": vid})
    ).scalar()
    assert embedded_at is None  # cleared so a failed re-embed is a visible gap

    # cleanup
    await session.execute(
        text("DELETE FROM laws WHERE id=(SELECT law_id FROM versions WHERE id=:v)"), {"v": vid}
    )
    await session.commit()
