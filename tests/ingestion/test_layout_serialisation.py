"""The layout round-trip both ingest lanes share.

The DBOS lane used to rebuild `OcrBlock` from a four-key projection, defaulting
the rest to plausible zeros. These hold the two lanes to one shape.
"""

from __future__ import annotations

from typing import Any

from codify.pipeline.enrich.ocr import (
    OcrBlock,
    PageDimensions,
    PageLayout,
    PageResult,
    layout_from_json,
    layout_to_json,
)
from codify.pipeline.enrich.regions import classify_layouts

A4 = PageDimensions(dpi=87, width=720, height=1018)


def _layout() -> PageLayout:
    return PageLayout(
        engine="mistral",
        model="mistral-ocr-4-0",
        header="الوقائع الزرزورية",
        footer="٣٥",
        dimensions=A4,
        blocks=[
            OcrBlock(
                type="title",
                content="# **مادة (٧٩)**",
                top_left_x=40,
                top_left_y=551,
                bottom_right_x=670,
                bottom_right_y=572,
            ),
            OcrBlock(
                type="table",
                content="| a | b |",
                top_left_x=55,
                top_left_y=600,
                bottom_right_x=665,
                bottom_right_y=700,
                table_id="tbl-0",
            ),
            OcrBlock(
                type="image",
                content="",
                top_left_x=60,
                top_left_y=720,
                bottom_right_x=400,
                bottom_right_y=860,
                image_id="img-0",
            ),
        ],
        raw={"discarded": "x" * 10_000},
    )


def _pages() -> list[PageResult]:
    return [
        PageResult(page_number=3, text="t", method="mistral_ocr", layout=_layout()),
        PageResult(page_number=4, text="t", method="vision_ocr", layout=None),
    ]


def test_the_round_trip_keeps_every_block_field() -> None:
    """Four of eight survived; the rest came back as zeros."""
    back = layout_from_json(layout_to_json(_pages()))
    assert list(back) == [3]  # a page the layout engine never saw stays absent
    assert back[3].blocks == _layout().blocks
    assert back[3].dimensions == A4


def test_the_table_id_survives_the_artifact() -> None:
    """Extracted tables use these keys; the projection must preserve them."""
    back = layout_from_json(layout_to_json(_pages()))
    assert [b.table_id for b in back[3].blocks] == [None, "tbl-0", None]
    assert [b.image_id for b in back[3].blocks] == [None, None, "img-0"]


def test_the_raw_response_stays_out_of_the_artifact() -> None:
    assert "discarded" not in str(layout_to_json(_pages()))


def test_both_lanes_classify_the_same_page_the_same_way() -> None:
    """The divergence this shared adapter exists to prevent."""
    live = classify_layouts({3: _layout()})
    replayed = classify_layouts(layout_from_json(layout_to_json(_pages())))
    assert live == replayed


def test_a_legacy_artifact_still_classifies() -> None:
    """A run in flight at deploy replays against what the old code wrote."""
    legacy: dict[str, Any] = {
        "3": [
            {"type": "_dimensions", "height": 1018},
            {"type": "title", "content": "# **مادة (٧٩)**", "y0": 551, "y1": 572},
            {"type": "text", "content": "على جميع الجهات", "y0": 579, "y1": 634},
        ]
    }
    regions = classify_layouts(layout_from_json(legacy))
    assert [r.kind for r in regions[3]] == ["body", "body"]
    assert regions[3][0].top == 551


def test_an_unreadable_page_payload_raises_rather_than_vanishing() -> None:
    """A skipped page reads downstream as an empty one."""
    import pytest
    from pydantic import ValidationError

    for payload in ("text", None, 7):
        with pytest.raises(ValidationError):
            layout_from_json({"3": payload})


def test_the_artifact_carries_no_per_word_confidence() -> None:
    """~70% of a gazette page's dump, and `page_reads` already holds it."""
    layout = _layout().model_copy(
        update={"words": [{"text": "مادة", "confidence": 0.4}], "page_confidence": 0.9}
    )
    dumped = layout_to_json([PageResult(page_number=3, text="t", method="m", layout=layout)])
    assert dumped["3"].keys() == {"engine", "model", "header", "footer", "blocks", "dimensions"}


def test_a_drifted_artifact_key_raises_instead_of_emptying_the_page() -> None:
    """`{"engine": ..., "blockz": [...]}` used to validate to a blank page."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        layout_from_json({"3": {"engine": "mistral", "blockz": [{"type": "text"}]}})
