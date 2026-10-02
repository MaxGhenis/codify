"""Dossier inputs load from version-lifetime evidence, never OCR.

Integration: needs a migrated postgres (same skip rules as test_rederive)."""

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
from codify.repair.dossier import assemble_dossier
from codify.storage.models import PageRead, SourceDocument
from codify.storage.repair import dossier_inputs_for_version
from codify.storage.repository import save_document
from codify.storage.versions import save_version_source_text

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
        if (await c.execute(text("SELECT to_regclass('page_reads')"))).scalar() is None:
            await engine.dispose()
            pytest.skip("schema not migrated; run alembic upgrade head")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_version(session: AsyncSession) -> tuple[uuid.UUID, str]:
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      Body of article one.\n    ARTICLE 2\n"
    suffix = uuid.uuid4().hex[:8]
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    vid = await save_document(
        session, parse_akn(akn), jurisdiction_code="xz", law_title=f"Dossier {suffix}", akn_xml=akn
    )
    await session.commit()
    return vid, akn


async def test_page_reads_are_preferred_and_no_ocr_is_needed(session: AsyncSession) -> None:
    vid, akn = await _seed_version(session)
    session.add_all(
        [
            PageRead(version_id=vid, page_number=1, engine="mistral_ocr", text="SECTION 1"),
            PageRead(
                version_id=vid,
                page_number=2,
                engine="vision_ocr",
                text="ARTICLE 2 body",
                rival_text="rival",
                metrics={"verdict": "accept", "reasons": []},
            ),
        ]
    )
    await save_version_source_text(session, vid, "SECTION 1\n\nARTICLE 2 body")
    await session.commit()

    inputs = await dossier_inputs_for_version(session, vid)
    assert inputs is not None
    assert inputs.page_evidence_source == "page_reads"
    assert [r.page_number for r in inputs.page_reads] == [1, 2]

    dossier, doc_text = assemble_dossier(inputs)
    assert doc_text == "SECTION 1\n\nARTICLE 2 body"
    assert not dossier.source_text_mismatch
    assert [s["page"] for s in dossier.spans] == [1, 2]


async def test_scan_with_aged_out_evidence_classifies_missing(session: AsyncSession) -> None:
    """A version whose source PDF hash is stamped but whose reads and artifact
    are gone must read as "missing" (re-OCR possible), never "none"."""
    bb = "BODY\n  ARTICLE 1\n    Body text.\n"
    suffix = uuid.uuid4().hex[:8]
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    sha = uuid.uuid4().hex + uuid.uuid4().hex
    session.add(
        SourceDocument(
            sha256=sha,
            original_filename=f"{suffix}.pdf",
            byte_size=1,
            object_key=f"uploads/xz/{suffix}.pdf",
            jurisdiction_code="xz",
        )
    )
    await session.flush()
    from codify.storage.repository import save_document_reporting

    outcome = await save_document_reporting(
        session,
        parse_akn(akn),
        jurisdiction_code="xz",
        law_title=f"Missing {suffix}",
        akn_xml=akn,
        source_sha256=sha,
    )
    await session.commit()

    inputs = await dossier_inputs_for_version(session, outcome.version_id)
    assert inputs is not None
    assert inputs.page_evidence_source == "missing"
    assert inputs.object_key == f"uploads/xz/{suffix}.pdf"


async def test_no_evidence_classifies_none_for_textual_ingest(session: AsyncSession) -> None:
    vid, _ = await _seed_version(session)
    inputs = await dossier_inputs_for_version(session, vid)
    assert inputs is not None
    # No page reads, no artifact, no source PDF: a txt/html ingest.
    assert inputs.page_evidence_source == "none"
    assert inputs.page_reads == []


async def test_textual_upload_with_sha_is_none_not_missing(session: AsyncSession) -> None:
    """A .txt upload stamps a source sha too; it must never classify as a scan
    (re-OCR on text bytes fails)."""
    bb = "BODY\n  ARTICLE 1\n    Body text.\n"
    suffix = uuid.uuid4().hex[:8]
    akn = parse_to_akn(bb, country="xz", doctype="act", number=suffix, date="2020-01-01")
    sha = uuid.uuid4().hex + uuid.uuid4().hex
    session.add(
        SourceDocument(
            sha256=sha,
            original_filename=f"{suffix}.txt",
            byte_size=1,
            object_key=f"uploads/xz/{suffix}.txt",
            jurisdiction_code="xz",
        )
    )
    await session.flush()
    from codify.storage.repository import save_document_reporting

    outcome = await save_document_reporting(
        session,
        parse_akn(akn),
        jurisdiction_code="xz",
        law_title=f"Textual {suffix}",
        akn_xml=akn,
        source_sha256=sha,
    )
    await session.commit()

    inputs = await dossier_inputs_for_version(session, outcome.version_id)
    assert inputs is not None
    assert inputs.page_evidence_source == "none"
