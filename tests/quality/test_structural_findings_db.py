"""The rate helpers against a real Postgres: a mis-scoped GROUP BY is the defect
that matters most and the one a unit test cannot see."""

from __future__ import annotations

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

from codify.storage.structural_findings import (
    finding_rates,
    record_findings,
    scanned_versions,
    versions_failing,
)
from codify.testing import postgres_url

pytestmark = pytest.mark.integration


async def _table_exists(engine: AsyncEngine, name: str) -> bool:
    async with engine.connect() as c:
        return (await c.execute(text("SELECT to_regclass(:n)"), {"n": name})).scalar() is not None


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    url = postgres_url()
    engine = create_async_engine(url, pool_pre_ping=True)
    if not await _table_exists(engine, "structural_findings"):
        await engine.dispose()
        pytest.skip("structural_findings missing; run alembic upgrade head")
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


async def _fixture_law(session: AsyncSession, code: str) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """A run, a law and a version to hang findings off, with the FKs satisfied."""
    run_id, law_id, version_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await session.execute(
        text("INSERT INTO runs (id, kind, status) VALUES (:i, 'admin_batch', 'succeeded')"),
        {"i": run_id},
    )
    juris = (
        await session.execute(
            text("SELECT id FROM jurisdictions WHERE code = :c LIMIT 1"), {"c": code}
        )
    ).scalar()
    if juris is None:
        pytest.skip(f"jurisdiction {code} not seeded")
    await session.execute(
        text(
            "INSERT INTO laws (id, jurisdiction_id, frbr_work_uri, title, doctype, year) "
            "VALUES (:i, :j, :u, 'fixture', 'act', 1999)"
        ),
        {"i": law_id, "j": juris, "u": f"/akn/{code}/act/1999/test-{law_id}"},
    )
    await session.execute(
        text(
            "INSERT INTO versions (id, law_id, expression_uri, language, expression_date, akn_xml) "
            "VALUES (:i, :l, :u, 'ara', '1999-01-01', '<x/>')"
        ),
        {"i": version_id, "l": law_id, "u": f"/akn/{code}/act/1999/test-{law_id}/ara@"},
    )
    return run_id, law_id, version_id


async def test_a_rate_keeps_could_not_run_out_of_the_denominator(session: AsyncSession) -> None:
    run_id, law_id, version_id = await _fixture_law(session, "xz")
    await record_findings(
        session,
        scan_run_id=run_id,
        version_id=version_id,
        law_id=law_id,
        jurisdiction_code="xz",
        era="assembly",
        doctype="act",
        findings=[
            ("body_units_present", True, {"count": 0}),
            ("eid_unique", False, {}),
            ("anchor_coverage", None, {"reason": "no_source_text"}),
        ],
    )
    await session.commit()

    rates = {r.check_name: r for r in await finding_rates(session, run_id, "xz")}
    assert rates["body_units_present"].failed == 1
    assert rates["body_units_present"].rate == 1.0
    assert rates["eid_unique"].rate == 0.0
    # The abstention is counted, and it is not in the denominator.
    assert rates["anchor_coverage"].not_run == 1
    assert rates["anchor_coverage"].measured == 0
    assert rates["anchor_coverage"].rate is None

    failing = await versions_failing(session, run_id, "body_units_present")
    assert len(failing) == 1
    assert failing[0][0].startswith("/akn/xz/act/1999/test-")
    # The denominator's size, not only the ratio it produced.
    assert await scanned_versions(session, run_id) == 1

    await session.execute(text("DELETE FROM runs WHERE id = :i"), {"i": run_id})
    await session.execute(text("DELETE FROM laws WHERE id = :i"), {"i": law_id})
    await session.commit()


async def test_two_jurisdictions_do_not_share_a_denominator(session: AsyncSession) -> None:
    """Every profile with no declared eras reports era "all", so grouping without
    the jurisdiction would pool unrelated corpora."""
    run_id, xz_law, xz_version = await _fixture_law(session, "xz")
    gb_run, gb_law, gb_version = await _fixture_law(session, "gb")
    for law_id, version_id, code in ((xz_law, xz_version, "xz"), (gb_law, gb_version, "gb")):
        await record_findings(
            session,
            scan_run_id=run_id,
            version_id=version_id,
            law_id=law_id,
            jurisdiction_code=code,
            era="all",
            doctype="act",
            findings=[("eid_unique", code == "xz", {})],
        )
    await session.commit()

    rates = {(r.jurisdiction_code, r.check_name): r for r in await finding_rates(session, run_id)}
    assert rates[("xz", "eid_unique")].failed == 1
    assert rates[("gb", "eid_unique")].failed == 0
    assert rates[("gb", "eid_unique")].rate == 0.0

    # A filtered rate needs the filtered population, or a filter matching nothing
    # reads as a scan with no defects.
    assert await scanned_versions(session, run_id) == 2
    assert await scanned_versions(session, run_id, "gb") == 1
    assert await scanned_versions(session, run_id, "al") == 0

    await session.execute(text("DELETE FROM runs WHERE id = ANY(:ids)"), {"ids": [run_id, gb_run]})
    await session.execute(text("DELETE FROM laws WHERE id = ANY(:ids)"), {"ids": [xz_law, gb_law]})
    await session.commit()


async def test_rescanning_a_version_replaces_its_rows(session: AsyncSession) -> None:
    run_id, law_id, version_id = await _fixture_law(session, "xz")
    args = {
        "scan_run_id": run_id,
        "version_id": version_id,
        "law_id": law_id,
        "jurisdiction_code": "xz",
        "era": "assembly",
        "doctype": "act",
    }
    await record_findings(session, **args, findings=[("eid_unique", True, {})])
    await record_findings(session, **args, findings=[("eid_unique", False, {})])
    await session.commit()

    rates = {r.check_name: r for r in await finding_rates(session, run_id, "xz")}
    assert rates["eid_unique"].failed == 0
    assert rates["eid_unique"].passed == 1

    await session.execute(text("DELETE FROM runs WHERE id = :i"), {"i": run_id})
    await session.execute(text("DELETE FROM laws WHERE id = :i"), {"i": law_id})
    await session.commit()
