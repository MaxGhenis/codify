"""What running the persist writes twice actually does to the corpus.

The ingest workflow's persist body used to run as workflow code, so recovery
re-executed it in full. The safety argument for that was `save_document`'s
dedup on (work, lang, hash). These establish where that argument holds and
where it does not, so the checkpoint that now covers it has a measured reason
rather than an asserted one.

Nothing here says `supersede_document` is wrong. Deleting before it saves is
what supersede is for. The finding is only that running it twice is not the
same as running it once.
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
from codify.storage.repository import (
    SaveOutcome,
    save_document_reporting,
    supersede_document,
)
from codify.storage.sources import upsert_source_document

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
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def akn_xml() -> str:
    """One document, reused across both passes: a replay re-parses the same
    artifact, so the second pass sees byte-identical input."""
    return parse_to_akn(
        "BODY\n  ARTICLE 1\n    Body text.\n",
        country="xz",
        doctype="act",
        number=uuid.uuid4().hex[:8],
        date="2020-01-01",
        language="ara",
    )


async def _law_at(session: AsyncSession, work_uri: str) -> uuid.UUID | None:
    return (
        await session.execute(text("SELECT id FROM laws WHERE frbr_work_uri = :u"), {"u": work_uri})
    ).scalar_one_or_none()


async def _cleanup(session: AsyncSession, work_uri: str) -> None:
    await session.rollback()  # a failed assertion can leave the session dirty
    await session.execute(text("DELETE FROM laws WHERE frbr_work_uri = :u"), {"u": work_uri})
    # Laws cascade to versions, which frees the provenance rows to go too.
    await session.execute(
        text("DELETE FROM source_documents WHERE object_key LIKE :p"),
        {"p": f"replay/{work_uri}/%"},
    )
    await session.commit()


async def _source(session: AsyncSession, work_uri: str) -> str:
    """A provenance row to hang the version off: `versions.source_sha256` is a
    foreign key, so the ingest upserts this before it persists.

    Keyed under the work URI so `_cleanup` can find every row this test made;
    these outlive the cascade, which reaches versions through the law and stops
    there."""
    sha = uuid.uuid4().hex
    await upsert_source_document(
        session,
        sha256=sha,
        original_filename="replay.pdf",
        byte_size=1,
        object_key=f"replay/{work_uri}/{sha}.pdf",
        jurisdiction_code="xz",
    )
    await session.flush()
    return sha


async def _persist(
    session: AsyncSession,
    akn_xml: str,
    *,
    supersede: bool,
    sha: str | None = None,
    run_id: uuid.UUID | None = None,
    selected: uuid.UUID | None = None,
) -> SaveOutcome:
    write = supersede_document if supersede else save_document_reporting
    extra = {"selected_version_id": selected} if supersede and selected else {}
    outcome = await write(
        session,
        parse_akn(akn_xml),
        jurisdiction_code="xz",
        law_title="Replay Act",
        akn_xml=akn_xml,
        source_sha256=sha,
        ingest_run_id=run_id,
        **extra,
    )
    await session.commit()
    return outcome


async def test_the_save_path_survives_being_run_twice(session: AsyncSession, akn_xml: str) -> None:
    """The dedup the old safety argument rested on. A replayed save returns the
    version the first pass stored and reports that it stored nothing."""
    work_uri = parse_akn(akn_xml).frbr_work_uri
    try:
        first = await _persist(session, akn_xml, supersede=False)
        second = await _persist(session, akn_xml, supersede=False)

        assert first.stored is True
        assert second.stored is False, "a replayed save must not store a second parse"
        assert second.version_id == first.version_id
        assert await _law_at(session, work_uri) is not None
    finally:
        await _cleanup(session, work_uri)


async def test_the_supersede_path_survives_being_run_twice_on_one_source(
    session: AsyncSession, akn_xml: str
) -> None:
    """The destructive path, repeated, when the source is known.

    A checkpoint alone does not cover this: a step result is recorded after the
    body returns, so a crash between the commit and that record replays the
    write regardless. Supersede therefore has to be safe to repeat on its own,
    and it is, by declining to replace a law with the source it already holds.
    """
    work_uri = parse_akn(akn_xml).frbr_work_uri
    try:
        sha = await _source(session, work_uri)
        first = await _persist(session, akn_xml, supersede=True, sha=sha)
        law_after_first = await _law_at(session, work_uri)

        second = await _persist(session, akn_xml, supersede=True, sha=sha)

        assert first.stored is True
        assert second.stored is False, "a replayed supersede must not write again"
        assert second.version_id == first.version_id
        assert await _law_at(session, work_uri) == law_after_first, "the law was re-homed"
    finally:
        await _cleanup(session, work_uri)


async def test_a_correction_still_supersedes(session: AsyncSession, akn_xml: str) -> None:
    """The skip above keys on the source, so it must not blunt a real
    correction: different bytes still replace what is there."""
    work_uri = parse_akn(akn_xml).frbr_work_uri
    try:
        first = await _persist(
            session, akn_xml, supersede=True, sha=await _source(session, work_uri)
        )
        second = await _persist(
            session, akn_xml, supersede=True, sha=await _source(session, work_uri)
        )

        assert second.stored is True
        assert second.version_id != first.version_id
    finally:
        await _cleanup(session, work_uri)


async def test_an_unhashed_supersede_still_cannot_be_repeated(
    session: AsyncSession, akn_xml: str
) -> None:
    """The limit of the guard, stated rather than left to be discovered.

    Neither a hash nor a run id, so this write leaves nothing to recognise it
    by and the second pass deletes and rebuilds. That is the floor, not the
    ingest path: every lane there sets a run id, and the test below covers it.
    """
    work_uri = parse_akn(akn_xml).frbr_work_uri
    try:
        first = await _persist(session, akn_xml, supersede=True)
        law_after_first = await _law_at(session, work_uri)

        second = await _persist(session, akn_xml, supersede=True)
        law_after_second = await _law_at(session, work_uri)

        assert first.stored is True
        assert second.stored is True, "with no hash there is nothing to dedup on"
        assert second.version_id != first.version_id
        assert law_after_second != law_after_first, "the replay re-homed the law"

        # The first pass's version is gone, not merely superseded.
        survived = (
            await session.execute(
                text("SELECT count(*) FROM versions WHERE id = :v"), {"v": first.version_id}
            )
        ).scalar_one()
        assert survived == 0

        # One law at the URI either way: the end state is defined, the identity is not.
        assert law_after_second is not None
    finally:
        await _cleanup(session, work_uri)


async def test_an_unhashed_supersede_is_repeatable_by_run_id(
    session: AsyncSession, akn_xml: str
) -> None:
    """The hashless write's own mark. Gating the guard on the hash meant a lane
    that persists without one skipped it entirely and re-homed the law on every
    replay, which is worse on the destructive path than on the save path: there
    the expression URI catches the repeat, here the law is already deleted."""
    work_uri = parse_akn(akn_xml).frbr_work_uri
    run_id = uuid.uuid4()
    try:
        first = await _persist(session, akn_xml, supersede=True, run_id=run_id)
        law_after_first = await _law_at(session, work_uri)

        second = await _persist(session, akn_xml, supersede=True, run_id=run_id)

        assert first.stored is True
        assert second.stored is False, "the replay wrote again with no hash to stop it"
        assert second.version_id == first.version_id
        assert await _law_at(session, work_uri) == law_after_first, "the law was re-homed"
    finally:
        await _cleanup(session, work_uri)


async def test_another_run_still_supersedes_an_unhashed_version(
    session: AsyncSession, akn_xml: str
) -> None:
    """The run id keys the replay, so it must not blunt a later correction the
    way a hash match would not: a different run replaces what is there."""
    work_uri = parse_akn(akn_xml).frbr_work_uri
    try:
        first = await _persist(session, akn_xml, supersede=True, run_id=uuid.uuid4())
        second = await _persist(session, akn_xml, supersede=True, run_id=uuid.uuid4())

        assert second.stored is True
        assert second.version_id != first.version_id
    finally:
        await _cleanup(session, work_uri)


async def test_a_hashless_replay_with_a_selection_does_not_delete_twice(
    session: AsyncSession, akn_xml: str
) -> None:
    """A hashless replay that also names a selection: without the guard it reached
    the compare, which found the selection gone and reported recovered-elsewhere
    for work the run had just done itself."""
    work_uri = parse_akn(akn_xml).frbr_work_uri
    run_id = uuid.uuid4()
    try:
        seed = await _persist(session, akn_xml, supersede=False)
        # The selection path is for a refused structure: the compare declines to
        # supersede a row carrying no halt, so without this the first pass is a
        # no-op and the replay proves nothing.
        await session.execute(
            text("UPDATE versions SET structure_halt = :h WHERE id = :v"),
            {"h": '{"findings": [{"check": "structure_halted"}]}', "v": seed.version_id},
        )
        await session.commit()
        first = await _persist(
            session, akn_xml, supersede=True, run_id=run_id, selected=seed.version_id
        )
        law_after_first = await _law_at(session, work_uri)

        second = await _persist(
            session, akn_xml, supersede=True, run_id=run_id, selected=seed.version_id
        )

        assert first.stored is True
        assert second.stored is False
        assert second.version_id == first.version_id, "the replay named a row it did not write"
        assert await _law_at(session, work_uri) == law_after_first, "the law was re-homed"
    finally:
        await _cleanup(session, work_uri)
