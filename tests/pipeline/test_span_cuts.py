"""A segmented source stored as cuts by page reads back as the same acts, and a
re-read finds each cut again by its marker or says it cannot."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.segment import PAGE_SEPARATOR, segment
from codify.pipeline.span_cuts import cuts_from_segmentation, locate_cut, span_text
from tests.config_fixtures import isolated_configs
from tests.pipeline.test_segment import COUNTRY, _config, _join, _three_act_issue


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[JurisdictionConfig]:
    with isolated_configs(monkeypatch, tmp_path / "jurisdictions", {COUNTRY: _config()}):
        yield load_config(COUNTRY)


def test_cuts_tile_the_source_and_rebuild_each_act(config: JurisdictionConfig) -> None:
    text, spans = _join(_three_act_issue())
    result = segment(text, spans, config=config)
    cuts = cuts_from_segmentation(result, text, spans, issue="12")
    assert [(c.kind, c.first_page, c.last_page) for c in cuts] == [
        ("front_matter", 1, 1),
        ("act", 2, 2),
        ("act", 2, 2),
        ("act", 3, 3),
    ]
    pages = {s.page: text[s.start : s.end] for s in spans}
    rebuilt = [
        span_text(pages, c.first_page, c.last_page, c.start_offset, c.end_offset) for c in cuts
    ]
    regions = [text[: result.front_matter[1]]] if result.front_matter else []
    # A region running to its page's end holds the separator after it; a page does not.
    expected = [
        r.removesuffix(PAGE_SEPARATOR) if c.end_offset is None else r
        for r, c in zip(regions + [s.text for s in result.segments], cuts, strict=True)
    ]
    assert rebuilt == expected
    assert [c.start_marker for c in cuts[1:]] == [s.heading for s in result.segments]
    assert {c.issue for c in cuts} == {"12"}


def test_a_held_region_is_cut_with_its_reason(config: JurisdictionConfig) -> None:
    text, spans = _join(_three_act_issue(listed_third=2))
    result = segment(text, spans, config=config)
    cuts = cuts_from_segmentation(result, text, spans)
    held = [c for c in cuts if c.kind == "held"]
    assert held and all(c.reason for c in held), cuts


def test_a_cut_is_found_again_after_the_page_is_re_read() -> None:
    before = "- 2 -\nACT No. 4 OF 2020\nTHE PILOTS ACT"
    after = "- 2 -\n\n  ACT  No. 4  OF 2020\nTHE PILOTS ACT"
    at = before.index("ACT No. 4")
    assert locate_cut(after, "ACT No. 4 OF 2020", at) == after.index("ACT  No. 4")


def test_the_nearest_of_two_markers_is_the_cut() -> None:
    page = "ACT No. 4 OF 2020\nbody\nACT No. 4 OF 2020\nmore"
    second = page.rindex("ACT No. 4")
    assert locate_cut(page, "ACT No. 4 OF 2020", second - 2) == second


def test_a_marker_no_longer_on_its_page_is_lost() -> None:
    assert locate_cut("- 2 -\nsomething else entirely", "ACT No. 4 OF 2020", 6) is None


def test_a_page_start_cut_stays_at_the_page_start_while_its_heading_is_there() -> None:
    page = "- 2 -\nACT No. 4 OF 2020\nTHE PILOTS ACT"
    assert locate_cut(page, "ACT No. 4 OF 2020", 0) == 0
    assert locate_cut("- 2 -\nTHE PILOTS ACT", "ACT No. 4 OF 2020", 0) is None


def test_a_segment_headed_as_a_kind_kept_apart_is_cut_as_skipped(
    config: JurisdictionConfig,
) -> None:
    text, spans = _join(_three_act_issue())
    result = segment(text, spans, config=config)
    cuts = cuts_from_segmentation(result, text, spans, skip=[r"ACT No\. 4 "])
    assert [(c.kind, c.act_key) for c in cuts if c.kind != "front_matter"] == [
        ("act", "act 3"),
        ("skipped", "act 4"),
        ("act", "act 5"),
    ]
    assert cuts[2].reason


def test_a_skip_pattern_that_cannot_compile_is_refused_at_load() -> None:
    from codify.jurisdictions import SegmentationConfig

    with pytest.raises(ValueError, match="heading pattern"):
        SegmentationConfig(skip_heading_patterns=["(unclosed"])
