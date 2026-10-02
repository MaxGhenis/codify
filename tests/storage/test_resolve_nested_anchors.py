"""Same-document anchors whose element has no row of its own.

A publisher's AKN names containers that the mapper stores only as their parts,
and amending text quoted inside a provision that stores it whole. Both are
elements the document carries, so a reference to one lands on the nearest row.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import date

import pytest
from lxml import etree
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from codify.akn import Article, Document, Paragraph
from codify.storage.models import CrossReference, Provision, Section
from codify.storage.repository import save_document
from codify.storage.resolve_refs import (
    element_index,
    nearest_stored_row,
    resolve_references_for_version,
)

_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

# A container stored only as its parts, a section stored as `__content`
# quoting a paragraph it inserts elsewhere, and a point inside a stored paragraph.
_AKN = f"""<akomaNtoso xmlns="{_NS}"><act><body>
<hcontainer eId="regulation-3" name="regulation">
  <paragraph eId="regulation-3-1"><content><p>One.</p></content></paragraph>
  <paragraph eId="regulation-3-2"><content><p>Two.</p></content></paragraph>
</hcontainer>
<section eId="section-8"><content><p>After paragraph 5 insert<mod>
  <quotedStructure><paragraph eId="d1e5"><num>5A</num>
    <subparagraph eId="d1e6"><content><p>Inserted.</p></content></subparagraph>
  </paragraph></quotedStructure></mod></p></content></section>
<paragraph eId="article-16-1"><list eId="article-16-1-list">
  <point eId="article-16-b"><content><p>A point.</p></content></point></list></paragraph>
<chapter eId="chp_1"><intro><p>Lead.</p></intro>
  <article eId="art_5" wId="art_5_w"><intro><p>The body shall:
    <term eId="art_5__term_1">act</term></p></intro>
    <paragraph eId="art_5__para_a"><content><p>(a) one.</p></content></paragraph></article>
</chapter>
<part eId="pt_2"><article eId="art_9"><paragraph eId="art_9__para_1">
  <content><p>Nine.</p></content></paragraph></article></part>
<paragraph eId="p_x"><list eId="p_x__list"><point eId="p_x__a"><content><p>Kept.</p></content>
</point></list></paragraph>
</body></act></akomaNtoso>"""

_STORED = {
    "regulation-3-1",
    "regulation-3-2",
    "section-8__content",
    "article-16-1",
    "chp_1__content",
    "art_5__intro",
    "art_5__para_a",
    "pt_2",  # a section row
    "p_x__a",
}


def _index() -> dict[str, etree._Element]:
    return element_index(etree.fromstring(_AKN.encode()))


@pytest.mark.parametrize(
    ("anchor", "lands_on"),
    [
        ("regulation-3", ("regulation-3-1", "container")),
        # Quoted text lands on the provision that quotes it, never on its own parts.
        ("d1e5", ("section-8__content", "enclosing")),
        ("d1e6", ("section-8__content", "enclosing")),
        ("article-16-b", ("article-16-1", "enclosing")),
        # A lead-in is the article's own row: never climb past it to the chapter's.
        ("art_5__term_1", ("art_5__intro", "enclosing")),
        # A wId names the same element as its eId.
        ("art_5_w", ("art_5__intro", "container")),
        # A section row encloses as surely as a provision does.
        ("art_9", ("pt_2", "enclosing")),
    ],
)
def test_an_anchor_lands_on_the_nearest_stored_row(anchor: str, lands_on: tuple[str, str]) -> None:
    assert nearest_stored_row(_index(), anchor, _STORED) == lands_on


def test_quoted_text_never_lands_on_its_own_parts() -> None:
    """Even a stored part of quoted text is text being inserted elsewhere."""
    assert nearest_stored_row(_index(), "d1e5", _STORED | {"d1e6"}) == (
        "section-8__content",
        "enclosing",
    )


def test_a_placeholder_is_not_an_answer_and_nor_are_its_parts() -> None:
    assert nearest_stored_row(_index(), "p_x", _STORED, placeholders={"p_x"}) is None


def test_an_anchor_the_document_lacks_lands_nowhere() -> None:
    assert nearest_stored_row(_index(), "regulation-9", _STORED) is None


def test_an_element_with_no_stored_row_near_it_lands_nowhere() -> None:
    assert nearest_stored_row(_index(), "regulation-3", {"article-16-1"}) is None


def _postgres_url() -> str:
    raw = os.environ.get("POSTGRES_URL", "postgresql://codify:codify@localhost:5432/codify")
    return raw.replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(_postgres_url(), pool_pre_ping=True)
    try:
        async with engine.connect() as c:
            if (await c.execute(text("SELECT to_regclass('provisions')"))).scalar() is None:
                pytest.skip("schema not migrated; run alembic upgrade head")
    except OSError:
        await engine.dispose()
        pytest.skip("postgres not reachable")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
        await s.rollback()
    await engine.dispose()


def _doc(work: str) -> Document:
    def para(eid: str, position: int) -> Paragraph:
        return Paragraph(akn_eid=eid, akn_type="paragraph", position=position, text="x")

    return Document(
        frbr_work_uri=work,
        frbr_expression_uri=f"{work}/eng@2020-01-01",
        language="eng",
        expression_date=date(2020, 1, 1),
        body=[
            # Stored as the section row `pt_2`, which encloses `art_9` in the AKN.
            Article(
                akn_eid="pt_2",
                akn_type="article",
                position=1,
                children=[para(e, i) for i, e in enumerate(sorted(_STORED - {"pt_2"}), start=1)],
            )
        ],
    )


@pytest.mark.integration
async def test_the_resolver_stamps_the_nearest_row(session: AsyncSession) -> None:
    work = f"/akn/xa/act/2020/{uuid.uuid4().hex[:8]}"
    version = await save_document(
        session,
        _doc(work),
        jurisdiction_code="xa",
        law_title="T",
        year=2020,
        akn_xml=_AKN,
    )
    rows = dict(
        (
            await session.execute(
                select(Provision.akn_eid, Provision.id).where(Provision.version_id == version)
            )
        ).all()
    )
    source = rows["article-16-1"]
    refs = {
        anchor: CrossReference(
            source_provision_id=source,
            target_uri=f"#{anchor}",
            ref_type="cross_reference",
            edge_class="freetext_reference",
        )
        for anchor in ("regulation-3", "d1e6", "regulation-9", "art_9")
    }
    session.add_all(refs.values())
    await session.flush()

    stats = await resolve_references_for_version(session, version)

    section = (
        await session.execute(
            select(Section.id).where(Section.version_id == version, Section.akn_eid == "pt_2")
        )
    ).scalar_one()
    landed = {a: (r.target_provision_id, r.target_section_id) for a, r in refs.items()}
    assert landed == {
        "regulation-3": (rows["regulation-3-1"], None),
        "d1e6": (rows["section-8__content"], None),
        "regulation-9": (None, None),
        "art_9": (None, section),
    }
    assert (stats.resolved_container, stats.resolved_enclosing) == (1, 2)
    assert stats.anchor_not_in_document == 1
