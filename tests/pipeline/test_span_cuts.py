"""A segmented source stored as cuts by page reads back as the same acts, and a
re-read finds each cut again by its marker or says it cannot."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.segment import PAGE_SEPARATOR, segment
from codify.pipeline.span_cuts import (
    SpanCut,
    cuts_from_segmentation,
    locate_cut,
    locate_generation,
    span_text,
)
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
    # The cut is the heading's line start, indentation included.
    assert locate_cut(after, "ACT No. 4 OF 2020", at) == after.index("  ACT  No. 4")


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


def test_a_marker_does_not_match_a_longer_line_it_opens() -> None:
    page = "Header\nLaw No. 12 on taxes\nbody\n"
    assert locate_cut(page, "Law No. 1", 7) is None


def test_a_truncated_marker_matches_the_line_it_was_cut_from() -> None:
    line = "ACT " + "x" * 300
    marker = line[:200]
    assert locate_cut("- 2 -\n" + line, marker, 6) == 6


def test_a_re_read_generation_keeps_its_tiling(config: JurisdictionConfig) -> None:
    text, spans = _join(_three_act_issue())
    result = segment(text, spans, config=config)
    cuts = cuts_from_segmentation(result, text, spans)
    pages = {s.page: text[s.start : s.end] for s in spans}
    # A re-read that lifts the running head and shifts every line.
    reread = {n: "\n" + body.replace("- 2 -\n", "") for n, body in pages.items()}
    located = locate_generation(reread, cuts)
    assert located is not None
    rebuilt = [
        span_text(reread, c.first_page, c.last_page, start, end)
        for c, (start, end) in zip(cuts, located, strict=True)
    ]
    # Each act opens at its heading, or at its page's start with the furniture above it.
    for region, act in zip(rebuilt[1:], result.segments, strict=True):
        assert act.heading in [x for x in region.splitlines() if x.strip()][:2], region[:80]
    assert "\n\n".join(rebuilt).replace("\n", "") == "\n\n".join(reread.values()).replace("\n", "")


def test_a_generation_with_a_lost_start_is_not_located(config: JurisdictionConfig) -> None:
    text, spans = _join(_three_act_issue())
    result = segment(text, spans, config=config)
    cuts = cuts_from_segmentation(result, text, spans)
    pages = {s.page: text[s.start : s.end].replace("ACT No. 4", "ACT No. A") for s in spans}
    assert locate_generation(pages, cuts) is None


def test_a_next_act_moved_to_its_page_start_still_ends_the_one_before() -> None:
    cuts = [
        SpanCut("act", 1, 2, 0, 10, start_marker="ACT No. 3 OF 2020"),
        SpanCut("act", 2, 2, 10, None, start_marker="ACT No. 4 OF 2020"),
    ]
    # The re-read lost the first act's tail on page 2: the second now opens it.
    pages = {1: "ACT No. 3 OF 2020\nbody", 2: "ACT No. 4 OF 2020\nbody"}
    located = locate_generation(pages, cuts)
    assert located == [(0, 0), (0, None)]
    first = span_text(pages, 1, 2, *located[0])
    assert "ACT No. 4" not in first


@pytest.mark.parametrize(
    "page",
    ["ACT No. 3\nACT No. 5\nACT No. 4", "ACT No. 3\nACT No. 4"],
    ids=["reordered", "colliding"],
)
def test_starts_that_run_backwards_are_not_located(page: str) -> None:
    cuts = [
        SpanCut("act", 1, 1, 0, 10, start_marker="ACT No. 3"),
        SpanCut("act", 1, 1, 10, 20, start_marker="ACT No. 4"),
        SpanCut("act", 1, 1, 20, None, start_marker="ACT No. 5" if "5" in page else "ACT No. 4"),
    ]
    assert locate_generation({1: page}, cuts) is None


def test_a_reordered_heading_behind_a_page_start_cut_is_caught() -> None:
    cuts = [
        SpanCut("act", 1, 1, 0, 7, start_marker="ACT No. 3"),
        SpanCut("act", 1, 1, 7, None, start_marker="ACT No. 4"),
    ]
    page = "Header\nACT No. 4\nFour.\nACT No. 3\nThree."
    assert locate_generation({1: page}, cuts) is None


def test_repair_evidence_from_the_ingest_artifact_is_trimmed_to_the_span() -> None:
    from codify.repair.dossier import DossierInputs, SpanTrim, assemble_dossier

    pages = ["ACT No. 3\nThree.", "End of three.\nACT No. 4\nFour."]
    text = "\n\n".join(pages)
    spans = [
        {"page": 1, "method": "text_layer", "start": 0, "end": len(pages[0])},
        {"page": 2, "method": "text_layer", "start": len(pages[0]) + 2, "end": len(text)},
    ]
    cut = pages[1].index("ACT No. 4")
    cuts = (
        SpanCut("act", 1, 2, 0, cut, start_marker="ACT No. 3"),
        SpanCut("act", 2, 2, cut, None, start_marker="ACT No. 4"),
    )
    akn = "<akomaNtoso xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'><act/></akomaNtoso>"
    for index, own, other in ((0, "Three", "Four"), (1, "Four", "Three")):
        inputs = DossierInputs(
            version_id="v",
            akn_xml=akn,
            country="",
            fallback_text=text,
            fallback_spans=spans,
            span_trim=SpanTrim(cuts[index].first_page, cuts[index].last_page, index, cuts),
        )
        _dossier, source = assemble_dossier(inputs)
        assert own in source and other not in source, source


def test_an_indented_heading_is_cut_at_its_line_start() -> None:
    page = "Three.\n    ACT No. 4\nFour."
    assert locate_cut(page, "ACT No. 4", 7) == 7


def test_unmapped_artifact_text_never_stands_as_a_childs_evidence() -> None:
    from codify.repair.dossier import DossierInputs, SpanTrim, assemble_dossier

    cuts = (SpanCut("act", 1, 1, 0, None, start_marker="ACT No. 3"),)
    akn = "<akomaNtoso xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'><act/></akomaNtoso>"
    inputs = DossierInputs(
        version_id="v",
        akn_xml=akn,
        country="",
        stored_source_text="ACT No. 3\nThree.",
        fallback_text="ACT No. 3\nThree.\nACT No. 4\nFour.",
        fallback_spans=[],
        span_trim=SpanTrim(1, 1, 0, cuts),
    )
    _dossier, source = assemble_dossier(inputs)
    assert source == "ACT No. 3\nThree."
