"""Proposal lifecycle: record → decide → apply, with the DB as the last gate.

Integration: needs a migrated postgres (same skip rules as test_rederive)."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from codify.akn.io import parse_akn
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.edit_ops import apply_plan
from codify.repair.ops import SetBody
from codify.storage.models import RepairProposalRow
from codify.storage.repair_proposals import (
    decide_proposal,
    list_proposals,
    persist_approved_repair,
    record_proposals,
)
from codify.storage.repository import save_document

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
        if (await c.execute(text("SELECT to_regclass('repair_proposals')"))).scalar() is None:
            await engine.dispose()
            pytest.skip("schema not migrated; run alembic upgrade head")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed(session: AsyncSession) -> tuple[uuid.UUID, str, dict[str, object]]:
    bb = "BODY\n  ARTICLE 1\n    Body of article one.\n  ARTICLE 2\n"
    suffix = uuid.uuid4().hex[:8]
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    vid = await save_document(
        session, parse_akn(akn), jurisdiction_code="xz", law_title=f"Prop {suffix}", akn_xml=akn
    )
    await session.commit()
    finding = next(f for f in validate_akn(akn) if f["check"] == "empty_article")
    return vid, akn, finding


def _payload(finding: dict[str, object], *, risk: str, status: str) -> dict[str, object]:
    return {
        "eid": finding["eid"],
        "check": finding["check"],
        "finding": finding,
        "ops": [SetBody(eid=str(finding["eid"]), bluebell="Restored body.").model_dump()],
        "reasoning": "test",
        "risk": risk,
        "status": status,
        "audit": {"reasons": []},
    }


async def test_full_lifecycle_pending_to_applied(session: AsyncSession) -> None:
    vid, akn, finding = await _seed(session)
    sha = hashlib.sha256(akn.encode("utf-8")).hexdigest()
    await record_proposals(
        session,
        version_id=vid,
        run_id=None,
        proposals=[_payload(finding, risk="high", status="pending")],
        akn_sha256=sha,
        source_sha256="",
        model="test-model",
        prompt_version="2.0",
    )
    await session.commit()

    rows = await list_proposals(session, vid, status="pending")
    assert len(rows) == 1
    # The DoD line "records evidence, model, prompt version, operations": read
    # the row back, or key drift silently downgrades every field.
    stored = rows[0]
    assert stored.risk_class == "high"
    assert stored.model == "test-model"
    assert stored.prompt_version == "2.0"
    assert stored.ops and stored.ops[0]["op"] == "set_body"
    assert stored.audit == {"reasons": []}
    assert stored.finding["check"] == "empty_article"
    row = await decide_proposal(session, rows[0].id, decision="approved", actor="reviewer@x")
    await session.commit()
    assert row.status == "approved" and row.decided_by == "reviewer@x"

    res = apply_plan(
        akn,
        [SetBody(eid=str(finding["eid"]), bluebell="Restored body.")],
        country="xz",
        target_finding=finding,
    )
    assert res.ok, res.error
    applied = await persist_approved_repair(session, row.id, res.xml, attribution="test")
    await session.commit()
    assert applied.status == "applied"

    reviewed = (
        await session.execute(
            text("SELECT reviewed_by, reviewed_at FROM versions WHERE id = :v"), {"v": vid}
        )
    ).one()
    assert reviewed[0] == "reviewer@x" and reviewed[1] is not None


async def test_stale_pin_supersedes_instead_of_applying(session: AsyncSession) -> None:
    vid, akn, finding = await _seed(session)
    await record_proposals(
        session,
        version_id=vid,
        run_id=None,
        proposals=[_payload(finding, risk="high", status="pending")],
        akn_sha256="0" * 64,  # pin never matches
        source_sha256="",
        model="m",
        prompt_version="2.0",
    )
    await session.commit()
    row = (await list_proposals(session, vid))[0]
    await decide_proposal(session, row.id, decision="approved", actor="reviewer@x")
    await session.commit()

    result = await persist_approved_repair(session, row.id, akn, attribution="test")
    await session.commit()
    # Returned, never raised: a raise would roll the supersede back in the
    # caller's transaction and strand the row approved forever.
    assert result.status == "superseded"
    assert (await list_proposals(session, vid))[0].status == "superseded"


async def test_the_db_refuses_an_applied_undecided_high_risk_row(
    session: AsyncSession,
) -> None:
    vid, _, finding = await _seed(session)
    session.add(
        RepairProposalRow(
            version_id=vid,
            eid=str(finding["eid"]),
            check_name="empty_article",
            finding=finding,
            ops=[],
            risk_class="high",
            status="applied",
            audit={},
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_stored_ops_round_trip_through_the_adapter(session: AsyncSession) -> None:
    """apply_approved revives ops from JSONB; a schema drift must fail a test
    before it fails a reviewer."""
    from pydantic import TypeAdapter

    from codify.repair.ops import RepairOp

    vid, _, finding = await _seed(session)
    await record_proposals(
        session,
        version_id=vid,
        run_id=None,
        proposals=[_payload(finding, risk="high", status="pending")],
        akn_sha256="",
        source_sha256="",
        model="m",
        prompt_version="2.0",
    )
    await session.commit()
    row = (await list_proposals(session, vid))[0]
    adapter: TypeAdapter[list[RepairOp]] = TypeAdapter(list[RepairOp])
    ops = adapter.validate_python(row.ops)
    assert ops and ops[0].op == "set_body"


async def test_a_decided_row_is_never_redecided(session: AsyncSession) -> None:
    vid, _, finding = await _seed(session)
    await record_proposals(
        session,
        version_id=vid,
        run_id=None,
        proposals=[_payload(finding, risk="high", status="pending")],
        akn_sha256="",
        source_sha256="",
        model="m",
        prompt_version="2.0",
    )
    await session.commit()
    row = (await list_proposals(session, vid))[0]
    await decide_proposal(session, row.id, decision="rejected", actor="reviewer@x")
    await session.commit()
    with pytest.raises(ValueError, match="not pending"):
        await decide_proposal(session, row.id, decision="approved", actor="other@x")
