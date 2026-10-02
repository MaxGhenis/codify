"""The shipped Uganda profile, held to invented Uganda-style text: sections that
print as "1. Heading" with no keyword, Parts and Divisions above them, closing
schedules, and the revised edition's and Acts Supplement's running heads. None
of the text below is from a real statute."""

from __future__ import annotations

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors
from codify.pipeline.enrich.ocr import (
    _strip_furniture_lines,
    clean_page_text,
    furniture_inline_patterns,
    furniture_line_patterns,
)

_ACT = """\
PART I—PRELIMINARY

1. Short title
This Act may be cited as the Village Water Supply Act.

2. Interpretation
In this Act, "levy" means the levy set out in Schedule 1 to this Act.

PART II—WATER BOARDS

DIVISION I—ESTABLISHMENT

3. Establishment of water boards
(1) There is established in every district a water board.
(2) A water board shall consist of five members.

DIVISION II—FINANCE

4. Levy
A water board may charge the levy at the rates in Schedule 1.

5. Accounts
A water board shall keep proper accounts.

SCHEDULES

Schedule 1
Rates of levy, in shillings each month.

Schedule 2
Form of application for a connection.
"""


def _anchors() -> list:
    regex = build_anchor_regex(load_config("ug"), "act")
    return scan_anchors(_ACT, regex, country="ug", doctype="act")


def test_a_bare_numbered_heading_is_a_section() -> None:
    sections = [a for a in _anchors() if a.kind == "section"]
    assert [a.number for a in sections] == ["1", "2", "3", "4", "5"]


def test_parts_divisions_and_sections_nest() -> None:
    eids = [a.akn_eid for a in _anchors() if a.kind != "schedule"]
    assert eids == [
        "part_I",
        "part_I__sec_1",
        "part_I__sec_2",
        "part_II",
        "part_II__dvs_I",
        "part_II__dvs_I__sec_3",
        "part_II__dvs_II",
        "part_II__dvs_II__sec_4",
        "part_II__dvs_II__sec_5",
    ]


def test_closing_schedules_sit_outside_the_parts() -> None:
    schedules = [a for a in _anchors() if a.kind == "schedule"]
    assert [(a.akn_eid, a.depth) for a in schedules] == [("schedule_1", 0), ("schedule_2", 0)]


def test_a_schedule_named_in_prose_is_not_a_container() -> None:
    # Sections 2 and 4 both name Schedule 1; only the closing heading anchors it.
    schedules = [a for a in _anchors() if a.kind == "schedule"]
    assert all(a.char_offset > _ACT.index("SCHEDULES") for a in schedules)


_BODY = "(2) A water board shall consist of five members."


class TestRunningHeads:
    def test_revised_edition_heads_go_and_the_body_stays(self) -> None:
        page = (
            f"4410 Cap. 999.] Village Water Supply Act\n{_BODY}\n"
            "Village Water Supply Act [Cap. 999. 4411"
        )
        out = _strip_furniture_lines(page, furniture_line_patterns("ug"))
        assert out == _BODY

    def test_acts_supplement_head_goes_with_or_without_its_year(self) -> None:
        for head in (
            "Act 9          Village Water Supply Act          2031",
            "Act 9 Village Water Supply Act",
        ):
            out = _strip_furniture_lines(f"{head}\n{_BODY}", furniture_line_patterns("ug"))
            assert out == _BODY

    def test_a_body_line_that_starts_with_an_act_number_stays(self) -> None:
        line = "Act 3 of 2001 is repealed by this Act"
        assert _strip_furniture_lines(line, furniture_line_patterns("ug")) == line

    def test_a_head_fused_into_a_sentence_is_scrubbed(self) -> None:
        line = (
            "A water board shall consist 4412 Cap. 999.] Village Water Supply Act of five members."
        )
        out = clean_page_text(line, furniture_inline_patterns("ug"))
        assert "Cap. 999" not in out
        assert out.startswith("A water board shall consist")
        assert out.endswith("of five members.")
