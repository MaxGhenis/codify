"""`retokenise_version_provisions` against a real database.

This function shipped broken with no test: an ORM-entity `update()` given
parameter dicts routes into SQLAlchemy's bulk-update-by-primary-key path, which
rejects extra WHERE criteria, so every call raised. Nothing caught it because
nothing exercised it. These tests do.
"""

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
from codify.search import TOKENISER_VERSION, tokenise_to_text
from codify.storage.repository import rederive_version_rows, save_document
from codify.storage.search_tokens import retokenise_version_provisions
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


async def _seed(session: AsyncSession, language: str = "eng") -> uuid.UUID:
    suffix = uuid.uuid4().hex[:8]
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      The rights of the accused.\n"
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    doc = parse_akn(akn)
    doc.language = language
    vid = await save_document(
        session, doc, jurisdiction_code="xz", law_title=f"Retokenise {suffix}", akn_xml=akn
    )
    await session.commit()
    return vid


async def _drop_tokens(session: AsyncSession, vid: uuid.UUID) -> None:
    """Return the version to its pre-tokeniser state."""
    await session.execute(
        text(
            "UPDATE provisions SET search_tokens=NULL, search_pipeline_version=NULL "
            "WHERE version_id=:v"
        ),
        {"v": vid},
    )
    await session.commit()


async def _cleanup(session: AsyncSession, vid: uuid.UUID) -> None:
    await session.execute(
        text("DELETE FROM laws WHERE id=(SELECT law_id FROM versions WHERE id=:v)"), {"v": vid}
    )
    await session.commit()


async def test_untokenised_rows_are_written(session: AsyncSession) -> None:
    """The case the backfill exists for: rows predating the tokeniser."""
    vid = await _seed(session)
    try:
        await _drop_tokens(session, vid)
        changed = await retokenise_version_provisions(session, vid)
        await session.commit()
        assert changed > 0

        rows = (
            await session.execute(
                text(
                    "SELECT text, search_tokens, search_pipeline_version "
                    "FROM provisions WHERE version_id=:v"
                ),
                {"v": vid},
            )
        ).all()
        assert len(rows) == changed
        for provision_text, tokens, version in rows:
            assert tokens == tokenise_to_text(provision_text, "eng")
            assert version == TOKENISER_VERSION
    finally:
        await _cleanup(session, vid)


async def test_a_second_pass_does_no_work(session: AsyncSession) -> None:
    """The idempotency the DBOS retry depends on: a re-run of a child that
    already succeeded must not rewrite rows."""
    vid = await _seed(session)
    try:
        await _drop_tokens(session, vid)
        assert await retokenise_version_provisions(session, vid) > 0
        await session.commit()
        assert await retokenise_version_provisions(session, vid) == 0
        await session.commit()
    finally:
        await _cleanup(session, vid)


async def test_a_version_bump_restages_every_row(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`IS DISTINCT FROM` rather than `IS NULL` is what makes a tokeniser
    change recoverable; without it a bump would find no work to do."""
    vid = await _seed(session)
    try:
        await _drop_tokens(session, vid)
        first = await retokenise_version_provisions(session, vid)
        await session.commit()
        assert first > 0
        assert await retokenise_version_provisions(session, vid) == 0

        monkeypatch.setattr("codify.storage.search_tokens.TOKENISER_VERSION", TOKENISER_VERSION + 1)
        assert await retokenise_version_provisions(session, vid) == first
        await session.rollback()
    finally:
        await _cleanup(session, vid)


async def test_a_missing_version_returns_zero_without_raising(session: AsyncSession) -> None:
    """A child pointed at a deleted version reports nothing done rather than
    failing the run."""
    assert await retokenise_version_provisions(session, uuid.uuid4()) == 0


async def test_tokens_follow_the_expression_language(session: AsyncSession) -> None:
    """The helper reads `versions.language`, so an Arabic expression is not
    retokenised as English."""
    vid = await _seed(session, language="ara")
    try:
        await _drop_tokens(session, vid)
        await retokenise_version_provisions(session, vid)
        await session.commit()
        rows = (
            await session.execute(
                text("SELECT text, search_tokens FROM provisions WHERE version_id=:v"), {"v": vid}
            )
        ).all()
        assert rows
        for provision_text, tokens in rows:
            assert tokens == tokenise_to_text(provision_text, "ara")
    finally:
        await _cleanup(session, vid)


async def test_both_writers_agree_when_the_akn_language_is_wrong(session: AsyncSession) -> None:
    """`document_to_rows` reads the AKN's FRBRlanguage; the backfill reads
    `versions.language`. They disagree for 959 EU versions, which carry `bul`
    over English text. The persisted row wins, so a rederive and a backfill must
    land on the same tokens."""
    vid = await _seed(session, language="eng")
    try:
        # The row says English; make the stored AKN claim otherwise, as the EU
        # corpus does. `versions.language` is immutable by trigger, so the AKN
        # is the side that moves, through the repair route that is allowed to.
        akn = (
            await session.execute(text("SELECT akn_xml FROM versions WHERE id=:v"), {"v": vid})
        ).scalar_one()
        akn = akn.replace('language="eng"', 'language="bul"')
        await update_repaired_akn(session, vid, akn, attribution="test")
        await session.commit()

        await rederive_version_rows(session, vid, akn)
        await session.commit()
        after_rederive = dict(
            (
                await session.execute(
                    text("SELECT akn_eid, search_tokens FROM provisions WHERE version_id=:v"),
                    {"v": vid},
                )
            ).all()
        )
        assert after_rederive

        await _drop_tokens(session, vid)
        await retokenise_version_provisions(session, vid)
        await session.commit()
        after_backfill = dict(
            (
                await session.execute(
                    text("SELECT akn_eid, search_tokens FROM provisions WHERE version_id=:v"),
                    {"v": vid},
                )
            ).all()
        )
        assert after_rederive == after_backfill
        # And they agree on the *right* language: English stems, Bulgarian does not.
        assert after_backfill != {
            eid: tokenise_to_text(t, "bul")
            for eid, t in (
                await session.execute(
                    text("SELECT akn_eid, text FROM provisions WHERE version_id=:v"), {"v": vid}
                )
            ).all()
        }
    finally:
        await _cleanup(session, vid)


_PLACEHOLDER = "[TIFF not transcribed: annex-page.tif]"


async def test_a_placeholder_only_provision_is_dropped_from_both_lexical_arms(
    session: AsyncSession,
) -> None:
    """Both arms read `search_tokens`: BM25 directly, the tsvector as a generated
    column over it. Emptying the tokens is what takes the placeholder out of each."""
    vid = await _seed(session)
    try:
        await session.execute(
            text("UPDATE provisions SET text = :t WHERE version_id = :v"),
            {"t": _PLACEHOLDER, "v": vid},
        )
        await _drop_tokens(session, vid)
        assert await retokenise_version_provisions(session, vid) > 0
        await session.commit()

        rows = (
            await session.execute(
                text(
                    "SELECT coalesce(search_tokens, ''), search_tsv::text "
                    "FROM provisions WHERE version_id = :v"
                ),
                {"v": vid},
            )
        ).all()
        assert rows
        for tokens, tsv in rows:
            assert tokens == ""  # BM25 arm
            assert tsv == ""  # generated tsvector arm

        # And the row is not returned by a match on the placeholder's own words.
        hits = (
            await session.execute(
                text(
                    "SELECT count(*) FROM provisions WHERE version_id = :v "
                    "AND search_tsv @@ plainto_tsquery('simple', 'transcribed')"
                ),
                {"v": vid},
            )
        ).scalar_one()
        assert hits == 0
    finally:
        await _cleanup(session, vid)


async def test_a_provision_that_merely_mentions_a_placeholder_keeps_its_tokens(
    session: AsyncSession,
) -> None:
    """The predicate matches the whole text, so real text beside a marker survives."""
    vid = await _seed(session)
    try:
        await session.execute(
            text("UPDATE provisions SET text = :t WHERE version_id = :v"),
            {"t": f"The rates in {_PLACEHOLDER} apply from 1 January.", "v": vid},
        )
        await _drop_tokens(session, vid)
        assert await retokenise_version_provisions(session, vid) > 0
        await session.commit()

        tokens = (
            await session.execute(
                text(
                    "SELECT coalesce(search_tokens, '') FROM provisions "
                    "WHERE version_id = :v ORDER BY position LIMIT 1"
                ),
                {"v": vid},
            )
        ).scalar_one()
        assert tokens != ""
    finally:
        await _cleanup(session, vid)
