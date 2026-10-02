"""Splitting a source that holds several acts, on fictional gazettes.

Every fixture is a fictional jurisdiction's issue. Each one either splits on a
heading another signal agrees with, keeps a hard negative whole, or holds what the
evidence cannot settle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import codify.jurisdictions as jurisdictions
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.enrich.ocr import PageSpan
from codify.pipeline.segment import (
    PAGE_SEPARATOR,
    Segmentation,
    SourcePage,
    _key,
    segment,
    segment_volume,
)
from tests.config_fixtures import isolated_configs

# A code no other test configures, so per-country caches cannot carry another shape.
COUNTRY = "xg"
CLOSING = "Given under the Seal of the Assembly"
SEGMENTATION = {
    "act_heading_patterns": [r"(?P<type>ACT|CODE|CONVENTION) No\. (?P<number>\d+) OF \d{4}"],
    "issue_heading_patterns": [r"ISSUE No\. (?P<number>\d+)"],
    "contents_keywords": ["Contents"],
    "printed_page_pattern": r"^- (\d+) -$",
}


def _config(**overrides: object) -> dict[str, object]:
    """The fictional `xa` configuration under its own code, declaring segmentation."""
    base = json.loads((jurisdictions.JURISDICTIONS_DIR / "xa" / "config.json").read_text())
    base.update(
        code=COUNTRY,
        closing_phrases=[CLOSING],
        adoption_markers=["is hereby ratified"],
        attachments=[{"caption": "ANNEX"}, {"caption": "ELUCIDATION", "normative": False}],
        segmentation=SEGMENTATION,
    )
    base.update(overrides)
    return base


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[JurisdictionConfig]:
    with isolated_configs(monkeypatch, tmp_path / "jurisdictions", {COUNTRY: _config()}):
        yield load_config(COUNTRY)


def _act(number: int, title: str, sections: int, *, doctype: str = "ACT") -> list[str]:
    lines = [f"{doctype} No. {number} OF 2020", title, "", "BE IT ENACTED by the Assembly:", ""]
    for i in range(1, sections + 1):
        lines += [f"Section {i}. Duty {i}", "The keeper of every harbour shall levy dues.", ""]
    return lines


def _signed() -> list[str]:
    return [f"{CLOSING} at Port Town.", "Speaker of the Assembly", ""]


def _join(pages: list[list[str]]) -> tuple[str, list[PageSpan]]:
    """Pages joined as extraction joins them, with each page's place in the text."""
    bodies = ["\n".join(lines) for lines in pages]
    spans, position = [], 0
    for number, body in enumerate(bodies, start=1):
        if number > 1:
            position += len(PAGE_SEPARATOR)
        spans.append(
            PageSpan(page=number, method="text_layer", start=position, end=position + len(body))
        )
        position += len(body)
    return PAGE_SEPARATOR.join(bodies), spans


def _conserved(result: Segmentation, text: str) -> None:
    """Front matter, acts and held regions tile the source with nothing lost."""
    pieces = sorted(
        [(s.start, s.end) for s in result.segments]
        + [(h.start, h.end) for h in result.held]
        + ([result.front_matter] if result.front_matter else [])
    )
    assert pieces[0][0] == 0
    assert pieces[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:], strict=False))
    assert all(s.text == text[s.start : s.end] for s in result.segments)


def test_two_acts_split_where_three_signals_agree(config: JurisdictionConfig) -> None:
    pages = [
        _act(3, "THE HARBOUR DUES ACT", 4) + _signed(),
        # A rule above the heading is decoration, not text preceding it.
        ["__________", *_act(4, "THE LIGHTHOUSE ACT", 3), *_signed()],
    ]
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    assert [s.heading for s in result.segments] == ["ACT No. 3 OF 2020", "ACT No. 4 OF 2020"]
    assert [s.key for s in result.segments] == ["act 3", "act 4"]
    assert result.segments[1].signals == ("closing", "restart", "page_start")
    assert [(s.first_page, s.last_page) for s in result.segments] == [(1, 1), (2, 2)]
    assert result.segments[1].text.startswith("__________\nACT No. 4 OF 2020\nTHE LIGHTHOUSE ACT")
    assert result.held == ()
    _conserved(result, text)


def _contents_page(entries: list[tuple[str, int]]) -> list[str]:
    lines = ["THE ATLANTIS GAZETTE", "ISSUE No. 12", "", "Contents", ""]
    for heading, page in entries:
        lines += [f"{heading} The ... Act", f"of the Assembly ............ {page}", ""]
    return lines


def _three_act_issue(third: str = "ACT No. 5 OF 2020", listed_third: int = 3) -> list[list[str]]:
    """A contents page, then three acts; the second starts mid-page."""
    contents = _contents_page(
        [("ACT No. 3 OF 2020", 2), ("ACT No. 4 OF 2020", 2), ("ACT No. 5 OF 2020", listed_third)]
    )
    number = int(third.split()[2])
    return [
        contents,
        [
            "- 2 -",
            "",
            *_act(3, "THE HARBOUR DUES ACT", 3),
            *_signed(),
            *_act(4, "THE PILOTS ACT", 3),
        ],
        ["- 3 -", "", *_act(number, "THE BUOYS ACT", 3), *_signed()],
    ]


def test_three_acts_split_against_their_contents_listing(config: JurisdictionConfig) -> None:
    text, spans = _join(_three_act_issue())
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    assert [s.key for s in result.segments] == ["act 3", "act 4", "act 5"]
    # The second act opens mid-page, so the page-start signal is absent.
    assert result.segments[1].signals == ("closing", "restart", "contents")
    # The second act is unsigned, so no closing stands before the third.
    assert result.segments[2].signals == ("restart", "page_start", "contents")
    front = result.front_matter
    assert front is not None and text[front[0] : front[1]].startswith("THE ATLANTIS GAZETTE")
    statuses = sorted((r.source, r.key, r.status, r.pdf_page) for r in result.reconciliation)
    assert statuses == [
        ("contents", "act 3", "matched", 2),
        ("contents", "act 4", "matched", 2),
        ("contents", "act 5", "matched", 3),
        ("heading", "act 3", "matched", 2),
        ("heading", "act 4", "matched", 2),
        ("heading", "act 5", "matched", 3),
    ]
    assert [r.printed_page for r in result.reconciliation if r.source == "contents"] == [2, 2, 3]
    _conserved(result, text)


def test_a_ratified_treaty_stays_with_its_act(config: JurisdictionConfig) -> None:
    act = _act(5, "THE LIGHTHOUSE CONVENTION (RATIFICATION) ACT", 2)
    act += ["Section 3. The Lighthouse Convention that follows is hereby ratified.", ""]
    treaty = ["CONVENTION No. 7 OF 2019", "ON LIGHTHOUSES", ""]
    treaty += [f"Section {i}. The parties shall keep a light." for i in range(1, 5)]
    text, spans = _join([act + _signed(), treaty])
    result = segment(text, spans, config=config)
    assert result.outcome == "single"
    assert result.segments[0].text is text
    vetoed = [(r.key, r.veto) for r in result.reconciliation if r.status == "vetoed"]
    assert vetoed == [("convention 7", "follows a declaration adopting another text")]


def test_an_issued_code_stays_with_the_act_issuing_it(config: JurisdictionConfig) -> None:
    act = _act(2, "THE HARBOUR CODE (ISSUING) ACT", 3) + _signed()
    code = ["ANNEX", "", *_act(1, "THE HARBOUR CODE", 6, doctype="CODE")]
    text, spans = _join([act, code])
    result = segment(text, spans, config=config)
    assert result.outcome == "single"
    assert result.segments[0].text is text
    vetoed = [(r.key, r.veto) for r in result.reconciliation if r.status == "vetoed"]
    assert vetoed == [("code 1", "follows an attachment caption")]


def test_an_elucidation_naming_its_act_stays_with_it(config: JurisdictionConfig) -> None:
    act = _act(5, "THE MOORINGS ACT", 3) + _signed()
    notes = ["ELUCIDATION", "OF", "ACT No. 5 OF 2020", "", "Section 1. Sufficiently clear."]
    text, spans = _join([act, notes])
    result = segment(text, spans, config=config)
    assert result.outcome == "single"
    vetoed = [r.veto for r in result.reconciliation if r.status == "vetoed"]
    assert vetoed == ["repeats the open act's own heading"]


def test_a_contents_listing_that_disagrees_holds_the_region(config: JurisdictionConfig) -> None:
    """Listed as act 5, printed as act 6: the first act still splits, the rest is held."""
    text, spans = _join(_three_act_issue(third="ACT No. 6 OF 2020"))
    result = segment(text, spans, config=config)
    assert result.outcome == "abstained"
    assert [s.key for s in result.segments] == ["act 3"]
    assert len(result.held) == 1
    held = result.held[0]
    assert text[held.start :].startswith("ACT No. 4 OF 2020")
    assert held.end == len(text)
    assert (held.first_page, held.last_page) == (2, 3)
    mismatched = sorted(
        (r.source, r.key) for r in result.reconciliation if r.status == "label_mismatch"
    )
    assert mismatched == [("contents", "act 5"), ("heading", "act 6")]
    assert "a different act stands where the contents places this one" in held.reason
    _conserved(result, text)


def test_a_contents_page_number_that_disagrees_holds_the_region(
    config: JurisdictionConfig,
) -> None:
    text, spans = _join(_three_act_issue(listed_third=2))
    result = segment(text, spans, config=config)
    assert result.outcome == "abstained"
    rows = [(r.source, r.status) for r in result.reconciliation if r.key == "act 5"]
    assert rows == [("contents", "page_mismatch"), ("heading", "page_mismatch")]


def test_an_uncorroborated_heading_is_held_not_split(config: JurisdictionConfig) -> None:
    """Mid-page, unsigned before, numbering carried on: nothing agrees it opens an act."""
    lines = _act(3, "THE HARBOUR DUES ACT", 3)
    lines += ["ACT No. 9 OF 2020 is amended as follows.", "", "Section 4. Duty 4", ""]
    text, spans = _join([lines])
    result = segment(text, spans, config=config)
    assert result.outcome == "abstained"
    assert result.segments == ()
    assert [(r.key, r.status) for r in result.reconciliation] == [
        ("act 3", "opening"),
        ("act 9", "uncorroborated"),
    ]
    assert result.held[0].reason.endswith("no signal agrees it opens an act")


def test_a_quoted_heading_is_not_a_boundary(config: JurisdictionConfig) -> None:
    lines = _act(3, "THE HARBOUR DUES ACT", 3) + _signed()
    lines += ["The schedule reads: “", "ACT No. 9 OF 2020", "THE OLD ACT”", ""]
    text, spans = _join([lines])
    result = segment(text, spans, config=config)
    assert result.outcome == "single"
    assert [r.veto for r in result.reconciliation if r.status == "vetoed"] == ["inside a quotation"]


def test_a_missing_enacting_formula_vetoes_where_the_acts_carry_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = _config(enacting_formula_markers=["BE IT ENACTED"])
    second = _act(4, "THE LIGHTHOUSE ACT", 3)
    unenacted = [line for line in second if not line.startswith("BE IT ENACTED")]
    with isolated_configs(monkeypatch, tmp_path / "j", {COUNTRY: fields}):
        cfg = load_config(COUNTRY)
        text, spans = _join([_act(3, "THE HARBOUR DUES ACT", 3) + _signed(), second])
        assert segment(text, spans, config=cfg).outcome == "decided"
        # Neither carries one: nothing to compare against, so no veto.
        bare = [line for line in _act(3, "A", 3) if not line.startswith("BE IT ENACTED")]
        text, spans = _join([bare + _signed(), unenacted])
        assert segment(text, spans, config=cfg).outcome == "decided"
        text, spans = _join([_act(3, "THE HARBOUR DUES ACT", 3) + _signed(), unenacted])
        result = segment(text, spans, config=cfg)
    assert result.outcome == "single"
    assert [r.veto for r in result.reconciliation if r.status == "vetoed"] == [
        "carries no enacting formula where the act before it does"
    ]


def test_a_single_act_comes_back_unchanged(config: JurisdictionConfig) -> None:
    text, spans = _join([_act(3, "THE HARBOUR DUES ACT", 4) + _signed()])
    result = segment(text, spans, config=config)
    assert result.outcome == "single"
    assert len(result.segments) == 1
    assert result.segments[0].text is text
    assert (result.segments[0].start, result.segments[0].end) == (0, len(text))
    assert [(r.key, r.status) for r in result.reconciliation] == [("act 3", "opening")]


def test_a_jurisdiction_declaring_nothing_is_never_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [_act(3, "A", 4) + _signed(), _act(4, "B", 3) + _signed()]
    with isolated_configs(monkeypatch, tmp_path / "j", {COUNTRY: _config(segmentation=None)}):
        text, spans = _join(pages)
        result = segment(text, spans, config=load_config(COUNTRY))
    assert result.outcome == "single"
    assert result.segments[0].text is text
    assert result.reconciliation == ()


def _issue(number: int, acts: list[tuple[int, str]]) -> list[list[str]]:
    pages = [["ISSUE No. " + str(number), "THE ATLANTIS GAZETTE", ""]]
    for page, (act, title) in enumerate(acts, start=2):
        pages.append([f"- {page} -", "", *_act(act, title, 3), *_signed()])
    return pages


def _pages(issues: list[list[list[str]]]) -> Iterator[SourcePage]:
    number = 0
    for issue in issues:
        for lines in issue:
            number += 1
            yield SourcePage(page=number, text="\n".join(lines))


def test_a_volume_splits_into_issues_and_each_issue_into_acts(config: JurisdictionConfig) -> None:
    issues = [
        _issue(11, [(1, "THE FERRIES ACT"), (2, "THE TOLLS ACT")]),
        _issue(12, [(3, "THE HARBOUR DUES ACT")]),
        _issue(13, [(4, "THE PILOTS ACT"), (5, "THE BUOYS ACT")]),
    ]
    found = list(segment_volume(_pages(issues), config=config))
    assert [(i.key, i.first_page, i.last_page, i.status) for i in found] == [
        ("11", 1, 3, "opening"),
        ("12", 4, 5, "corroborated"),
        ("13", 6, 8, "corroborated"),
    ]
    # Page start, the closing on the page before, and page numbers starting again.
    assert found[1].signals == ("page_start", "closing", "restart")
    assert [i.segmentation.outcome for i in found] == ["decided", "single", "decided"]
    assert [s.key for s in found[2].segmentation.segments] == ["act 4", "act 5"]
    # Each page once: the issue's text is its pages joined, nothing read twice.
    pages = ["\n".join(lines) for lines in issues[1]]
    assert found[1].segmentation.segments[0].text == PAGE_SEPARATOR.join(pages)
    assert [(s.first_page, s.last_page) for s in found[2].segmentation.segments] == [(7, 7), (8, 8)]


def test_a_volume_is_read_one_issue_at_a_time(config: JurisdictionConfig) -> None:
    """The first issue is out before the third issue's pages are read."""
    issues = [_issue(n, [(n, "THE TOLLS ACT")]) for n in (11, 12, 13)]
    read: list[int] = []

    def pages() -> Iterator[SourcePage]:
        for page in _pages(issues):
            read.append(page.page)
            yield page

    stream = segment_volume(pages(), config=config)
    first = next(stream)
    assert (first.first_page, first.last_page) == (1, 2)
    # One issue, plus the next issue's opening page and the page after it.
    assert max(read) == 4


def test_a_running_head_repeating_the_issue_is_not_a_boundary(config: JurisdictionConfig) -> None:
    issue = _issue(11, [(1, "THE FERRIES ACT"), (2, "THE TOLLS ACT")])
    issue[2] = ["ISSUE No. 11", *issue[2]]
    found = list(segment_volume(_pages([issue]), config=config))
    assert [(i.key, i.first_page, i.last_page) for i in found] == [("11", 1, 3)]


def test_an_issue_heading_nothing_agrees_with_holds_the_issue(config: JurisdictionConfig) -> None:
    # Unsigned, mid-page, and no page numbers to restart.
    issue = [["ISSUE No. 11", ""], _act(1, "THE FERRIES ACT", 3)]
    issue.append(["The tolls of", "ISSUE No. 12", "are reduced."])
    found = list(segment_volume(_pages([issue]), config=config))
    assert len(found) == 1
    assert found[0].status == "uncorroborated"
    assert found[0].segmentation.outcome == "abstained"
    assert found[0].segmentation.segments == ()
    assert "ISSUE No. 12" in found[0].segmentation.held[0].reason


def _with_line_in_first_act(line: list[str]) -> tuple[str, list[PageSpan]]:
    """The three-act issue with `line` inserted between the first act's sections."""
    pages = _three_act_issue()
    at = pages[1].index("Section 2. Duty 2")
    pages[1][at:at] = line
    return _join(pages)


def test_an_unlisted_heading_no_signal_supports_is_read_as_a_citation(
    config: JurisdictionConfig,
) -> None:
    """Neither the contents nor any signal says an act opens here, so nothing is held."""
    text, spans = _with_line_in_first_act(["ACT No. 9 OF 2019 is repealed.", ""])
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    assert [s.key for s in result.segments] == ["act 3", "act 4", "act 5"]
    row = next(r for r in result.reconciliation if r.key == "act 9")
    assert (row.status, row.signals) == ("heading_only", ())
    assert row.describe().endswith("read as a citation")


def test_an_unlisted_heading_a_signal_supports_holds_the_region(
    config: JurisdictionConfig,
) -> None:
    """A closing says an act may open where the contents lists none: they disagree."""
    text, spans = _with_line_in_first_act([*_signed(), "ACT No. 9 OF 2019", ""])
    result = segment(text, spans, config=config)
    assert result.outcome == "abstained"
    row = next(r for r in result.reconciliation if r.key == "act 9")
    assert (row.status, row.signals) == ("heading_only", ("closing",))
    assert [s.key for s in result.segments] == ["act 4", "act 5"]
    assert text[result.held[0].start :].startswith("ACT No. 3 OF 2020")


def test_a_citation_wrapped_inside_a_contents_entry_is_not_an_entry(
    config: JurisdictionConfig,
) -> None:
    pages = _three_act_issue()
    at = pages[0].index("of the Assembly ............ 2")
    pages[0][at:at] = ["amending", "ACT No. 1 OF 2019 and"]
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    listed = [(r.key, r.printed_page) for r in result.reconciliation if r.source == "contents"]
    assert listed == [("act 3", 2), ("act 4", 2), ("act 5", 3)]


def test_an_entry_running_on_past_any_listing_ends_the_contents(
    config: JurisdictionConfig,
) -> None:
    """A heading followed by prose and no page is body text, not a listed act."""
    pages = _three_act_issue()
    pages[0] += ["ACT No. 8 OF 2020", *["Text that runs on as a body would."] * 40, ""]
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    listed = [r.key for r in result.reconciliation if r.source == "contents"]
    assert listed == ["act 3", "act 4", "act 5"]


def test_a_quotation_opened_long_before_a_heading_does_not_claim_it(
    config: JurisdictionConfig,
) -> None:
    """One stray mark must not veto every heading after it."""
    pages = [
        _act(3, "THE HARBOUR DUES ACT", 4) + ["It cites “the old rule."] + _signed(),
        _act(4, "THE LIGHTHOUSE ACT", 3) + _signed(),
    ]
    pages[0][-3:-3] = ["The keeper shall keep the light burning at night."] * 12
    text, spans = _join(pages)
    assert text.index("ACT No. 4") - text.index("“") > 400
    assert segment(text, spans, config=config).outcome == "decided"


def test_a_closed_quotation_before_a_heading_does_not_claim_it(
    config: JurisdictionConfig,
) -> None:
    pages = [
        _act(3, "THE HARBOUR DUES ACT", 4) + ["It cites “the old rule”."] + _signed(),
        _act(4, "THE LIGHTHOUSE ACT", 3) + _signed(),
    ]
    text, spans = _join(pages)
    assert segment(text, spans, config=config).outcome == "decided"


def test_page_numbers_in_furniture_place_the_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lifted into furniture, printed by a second alternative of the pattern."""
    rules = {**SEGMENTATION, "printed_page_pattern": r"^- (\d+) -$|^Page (\d+)$"}
    pages = _three_act_issue()
    furniture = {2: "Page 2", 3: "Page 3"}
    for lines in pages[1:]:
        del lines[:2]
    with isolated_configs(monkeypatch, tmp_path / "j", {COUNTRY: _config(segmentation=rules)}):
        text, spans = _join(pages)
        result = segment(text, spans, config=load_config(COUNTRY), furniture=furniture)
    assert result.outcome == "decided"
    assert [(r.key, r.pdf_page) for r in result.reconciliation if r.source == "contents"] == [
        ("act 3", 2),
        ("act 4", 2),
        ("act 5", 3),
    ]


def test_space_inside_a_number_is_extraction_noise() -> None:
    match = re.match(r"(?P<type>ACT) No\. (?P<number>\d[\d ]*)", "ACT No. 1 2")
    assert match is not None
    assert _key(match) == "act 12"


def _mid_page_second_act(first: list[str]) -> tuple[str, list[PageSpan]]:
    """`first`, then a second act on the same page, so no page start can agree."""
    return _join([first + _act(4, "THE LIGHTHOUSE ACT", 3)])


def test_an_adoption_far_before_a_heading_does_not_veto_it(config: JurisdictionConfig) -> None:
    first = _act(3, "THE HARBOUR DUES ACT", 1)
    first += ["The Lighthouse Convention is hereby ratified."]
    first += ["The keeper shall keep the light burning at night."] * 10
    first += ["Section 2. Duty 2", "Section 3. Duty 3", *_signed()]
    text, spans = _mid_page_second_act(first)
    assert text.index("ACT No. 4") - text.index("hereby ratified") > 400
    assert segment(text, spans, config=config).outcome == "decided"


def test_a_closing_inside_the_act_is_not_a_closing_before_the_next(
    config: JurisdictionConfig,
) -> None:
    """Only text after the last numbered provision can close the act."""
    first = _act(3, "THE HARBOUR DUES ACT", 1)
    first += [f"Section 2. Done as if {CLOSING}.", "Section 3. Duty 3"]
    first += ["Section 4. Duty 4"]
    unsigned = [*first, *_act(4, "THE LIGHTHOUSE ACT", 3)]
    second = unsigned.index("ACT No. 4 OF 2020")
    unsigned[unsigned.index("Section 1. Duty 1", second)] = "Section 5. Duty 5"
    text, spans = _join([unsigned])
    result = segment(text, spans, config=config)
    row = next(r for r in result.reconciliation if r.key == "act 4")
    assert (row.status, row.signals) == ("uncorroborated", ())


def test_a_quoted_closing_is_not_a_closing(config: JurisdictionConfig) -> None:
    first = _act(3, "THE HARBOUR DUES ACT", 3) + [f"It cites \u201c{CLOSING}\u201d."]
    second = _act(4, "THE LIGHTHOUSE ACT", 3)
    second[second.index("Section 1. Duty 1")] = "Section 4. Duty 4"
    text, spans = _join([first + second])
    row = next(r for r in segment(text, spans, config=config).reconciliation if r.key == "act 4")
    assert (row.status, row.signals) == ("uncorroborated", ())


@pytest.mark.parametrize(("sections", "signals"), [(2, ()), (3, ("restart",))])
def test_a_restart_needs_a_sequence_to_leave(
    config: JurisdictionConfig, sections: int, signals: tuple[str, ...]
) -> None:
    """Back to 1 after 1, 2 is too short to read as a new act; after 1, 2, 3 it is not."""
    text, spans = _mid_page_second_act(_act(3, "THE HARBOUR DUES ACT", sections))
    row = next(r for r in segment(text, spans, config=config).reconciliation if r.key == "act 4")
    assert row.signals == signals


def test_a_source_opening_mid_act_keeps_that_act_as_a_segment(config: JurisdictionConfig) -> None:
    """Provisions before the first heading belong to an act begun elsewhere."""
    tail = ["Section 7. Duty 7", "Section 8. Duty 8", "Section 9. Duty 9", *_signed()]
    text, spans = _join([tail, _act(4, "THE LIGHTHOUSE ACT", 3) + _signed()])
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    assert [(s.start, s.heading) for s in result.segments] == [
        (0, ""),
        (spans[1].start, "ACT No. 4 OF 2020"),
    ]
    assert result.front_matter is None


def test_a_part_heading_above_the_first_act_is_not_a_provision(
    config: JurisdictionConfig,
) -> None:
    pages = [
        ["PART I", "ACTS OF THE ASSEMBLY", "", *_act(3, "THE HARBOUR DUES ACT", 4), *_signed()],
        _act(4, "THE LIGHTHOUSE ACT", 3) + _signed(),
    ]
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    assert [s.heading for s in result.segments] == ["ACT No. 3 OF 2020", "ACT No. 4 OF 2020"]
    assert result.front_matter == (0, text.index("ACT No. 3"))


def test_a_line_opening_with_the_contents_word_is_not_a_listing(
    config: JurisdictionConfig,
) -> None:
    first = _act(3, "THE HARBOUR DUES ACT", 4)
    first[first.index("Section 2. Duty 2") : first.index("Section 2. Duty 2")] = [
        "Contents of a vessel are liable to dues."
    ]
    text, spans = _join([first + _signed(), _act(4, "THE LIGHTHOUSE ACT", 3) + _signed()])
    result = segment(text, spans, config=config)
    assert result.segments[1].signals == ("closing", "restart", "page_start")


def test_an_act_starting_on_its_contents_page_stays_an_act(config: JurisdictionConfig) -> None:
    """The listing names the page it sits on, so only the body's provisions end it."""
    contents = _contents_page([("ACT No. 3 OF 2020", 1), ("ACT No. 4 OF 2020", 2)])
    pages = [
        [*contents, *_act(3, "THE HARBOUR DUES ACT", 12), *_signed()],
        ["- 2 -", "", *_act(4, "THE LIGHTHOUSE ACT", 3), *_signed()],
    ]
    pages[0].insert(0, "- 1 -")
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    assert result.outcome == "decided"
    assert [s.key for s in result.segments] == ["act 3", "act 4"]
    assert sorted((r.source, r.status) for r in result.reconciliation) == [
        ("contents", "matched"),
        ("contents", "matched"),
        ("heading", "matched"),
        ("heading", "matched"),
    ]


def test_a_listing_whose_pages_cannot_be_found_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Printed numbers declared but absent: no entry is placed, so signals decide."""
    pages = _three_act_issue()
    for lines in pages[1:]:
        del lines[:2]
    with isolated_configs(monkeypatch, tmp_path / "j", {COUNTRY: _config()}):
        text, spans = _join(pages)
        result = segment(text, spans, config=load_config(COUNTRY))
    assert [r.source for r in result.reconciliation].count("contents") == 0
    assert [s.key for s in result.segments] == ["act 3", "act 4", "act 5"]


def test_a_prose_line_opening_with_the_contents_word_lists_nothing(
    config: JurisdictionConfig,
) -> None:
    first = _act(3, "THE HARBOUR DUES ACT", 4)
    first += ["Contents of a cargo hold are as listed under", "ACT No. 9 OF 2019 at page 2"]
    second = ["- 2 -", "", *_act(4, "THE LIGHTHOUSE ACT", 3), *_signed()]
    text, spans = _join([["- 1 -", "", *first, *_signed()], second])
    result = segment(text, spans, config=config)
    assert [r for r in result.reconciliation if r.source == "contents"] == []


def test_an_entry_too_long_to_be_one_is_not_continued(config: JurisdictionConfig) -> None:
    """Past the length of an entry, a heading opens the next one even with no page yet."""
    pages = _three_act_issue()
    at = pages[0].index("ACT No. 3 OF 2020 The ... Act")
    pages[0][at + 1 : at + 2] = ["A description that runs on, and on."] * 40
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    listed = [r.key for r in result.reconciliation if r.source == "contents"]
    assert listed == ["act 4", "act 5"]


def test_the_page_an_entry_names_ends_the_listing(config: JurisdictionConfig) -> None:
    """A first act with no numbered provisions is still the body, not another entry."""
    pages = _three_act_issue()
    notice = ["- 2 -", "", "ACT No. 3 OF 2020", "THE HARBOUR DUES ACT", "", "Dues are abolished."]
    pages[1] = [*notice, *_signed(), *_act(4, "THE PILOTS ACT", 3)]
    text, spans = _join(pages)
    result = segment(text, spans, config=config)
    assert [s.key for s in result.segments] == ["act 3", "act 4", "act 5"]
    assert [r.status for r in result.reconciliation if r.key == "act 3"] == ["matched", "matched"]
