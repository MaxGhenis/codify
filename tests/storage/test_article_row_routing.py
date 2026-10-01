"""Which table an article lands in: `sections`, never `provisions`. Querying
`provisions` for articles therefore answers zero on every correct document."""

from __future__ import annotations

import uuid

from codify.akn import Article, Chapter, Point
from codify.akn.document import Document
from codify.storage.mappers import LEAF_KINDS, ROW_SECTION_KINDS, document_to_rows


def _doc() -> Document:
    chapter = Chapter(
        akn_eid="chp_1",
        akn_type="chapter",
        position=0,
        heading="General",
        children=[
            Article(
                akn_eid="art_1",
                akn_type="article",
                position=0,
                text="The authority shall:",
                children=[
                    Point(akn_eid="art_1__point_1", akn_type="point", position=0, text="a"),
                ],
            ),
            Article(akn_eid="art_2", akn_type="article", position=1, text="Body."),
        ],
    )
    return Document(
        id=uuid.uuid4(),
        frbr_work_uri="/akn/xz/act/2012/7",
        frbr_expression_uri="/akn/xz/act/2012/7/ara@2012",
        language="ara",
        expression_date="2012-01-01",
        body=[chapter],
    )


def test_an_article_is_a_section_row_never_a_provision_row() -> None:
    """A healthy law reports zero article-typed provisions."""
    _, sections, provisions, _, _, _ = document_to_rows(_doc(), law_id=uuid.uuid4())

    assert sorted(s.akn_eid for s in sections if s.akn_type == "article") == ["art_1", "art_2"]
    assert [p for p in provisions if p.akn_type == "article"] == []


def test_an_articles_own_text_becomes_a_paragraph_provision() -> None:
    """The `paragraph` rows under an article are synthetic: its lead-in and its
    body text. They are not a sign the article was mistyped."""
    _, _, provisions, _, _, _ = document_to_rows(_doc(), law_id=uuid.uuid4())

    synthetic = {p.akn_eid: p.akn_type for p in provisions if p.akn_eid.startswith("art_")}
    assert synthetic["art_1__intro"] == "paragraph"
    assert synthetic["art_2__content"] == "paragraph"
    assert synthetic["art_1__point_1"] == "point"


def test_the_two_row_vocabularies_stay_disjoint() -> None:
    """One kind in both sets would put the same element in two tables."""
    assert not ROW_SECTION_KINDS & LEAF_KINDS
    assert "article" in ROW_SECTION_KINDS
