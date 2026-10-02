"""Every provision row leaves the mapper carrying lexical-arm tokens.

The tokens are written once per document rather than at each `ProvisionRow`
construction site, so this asserts the property that arrangement buys: no row
escapes untokenised, whichever branch built it.
"""

from __future__ import annotations

import uuid

from codify.akn import Article, Chapter, Point
from codify.akn.document import Document
from codify.search import TOKENISER_VERSION, tokenise_to_text
from codify.storage.mappers import document_to_rows


def _doc(language: str) -> Document:
    chapter = Chapter(
        akn_eid="chp_1",
        akn_type="chapter",
        position=0,
        heading="General",
        children=[
            # Lead-in plus children: the `__intro` branch.
            Article(
                akn_eid="art_1",
                akn_type="article",
                position=0,
                text="The Authority shall determine the following:",
                children=[
                    Point(
                        akn_eid="art_1__point_1",
                        akn_type="point",
                        position=0,
                        text="the rights of the accused",
                    )
                ],
            ),
            # Text with no leaf children: the `__content` branch.
            Article(akn_eid="art_2", akn_type="article", position=1, text="Appeals are heard."),
        ],
    )
    # An attachment too: its provisions come from a separate walk, so a fixture
    # without one cannot catch a branch that appends after the token pass.
    attachment = Chapter(
        akn_eid="att_1__chp_1",
        akn_type="chapter",
        position=0,
        heading="Schedule",
        children=[
            Article(
                akn_eid="att_1__art_1",
                akn_type="article",
                position=0,
                text="Scheduled rates of compensation.",
            )
        ],
    )
    return Document(
        id=uuid.uuid4(),
        frbr_work_uri="/akn/xz/act/2012/7",
        frbr_expression_uri=f"/akn/xz/act/2012/7/{language}@2012",
        language=language,
        expression_date="2012-01-01",
        body=[chapter, attachment],
    )


def test_every_provision_row_carries_tokens_and_a_version() -> None:
    _, _, provisions, _, _, _ = document_to_rows(_doc("eng"), law_id=uuid.uuid4())
    assert provisions, "fixture should produce provisions"
    for row in provisions:
        assert row.search_tokens
        assert row.search_pipeline_version == TOKENISER_VERSION


def test_tokens_match_the_shared_tokeniser_for_the_expression_language() -> None:
    """The write path and the query path must agree, so the mapper may not
    apply its own variant."""
    for language in ("eng", "ara"):
        _, _, provisions, _, _, _ = document_to_rows(_doc(language), law_id=uuid.uuid4())
        for row in provisions:
            assert row.search_tokens == tokenise_to_text(row.text, language)


def test_language_selects_the_pipeline() -> None:
    """The expression's language picks the pipeline, so an Arabic expression is
    tokenised as Arabic rather than as English."""
    _, _, ara, _, _, _ = document_to_rows(_doc("ara"), law_id=uuid.uuid4())
    for row in ara:
        assert row.search_tokens == tokenise_to_text(row.text, "ara")
        assert row.search_tokens != tokenise_to_text(row.text, "eng")
