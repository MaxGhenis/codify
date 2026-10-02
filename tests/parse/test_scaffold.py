"""Unit tests for the anchor-driven scaffold emitter + body-fill schema."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from codify.pipeline.enrich.anchors import StructuralAnchor
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.scaffold import (
    BodyBlock,
    BodyFillResponse,
    _marker_header,
    assemble_filled_scaffold,
    scaffold_from_anchors,
)
from codify.pipeline.enrich.validator import validate_akn


def _anchor(
    kind: str,
    keyword: str,
    number: str,
    *,
    char_offset: int = 0,
    depth: int = 0,
    parent_eid: str | None = None,
    akn_eid: str = "",
    quoted_amendment: bool = False,
    heading: str | None = None,
) -> StructuralAnchor:
    return StructuralAnchor(
        kind=kind,
        keyword=keyword,
        number=number,
        char_offset=char_offset,
        line=1,
        matched_text=f"{keyword} {number}",
        depth=depth,
        parent_eid=parent_eid,
        akn_eid=akn_eid,
        quoted_amendment=quoted_amendment,
        heading=heading,
    )


def test_scaffold_emits_flat_articles():
    anchors = [
        _anchor("article", "ARTICLE", "1", akn_eid="art_1"),
        _anchor("article", "ARTICLE", "2", char_offset=100, akn_eid="art_2"),
    ]
    skeleton, index = scaffold_from_anchors(anchors)
    assert "BODY" in skeleton
    assert "ARTICLE 1" in skeleton
    assert "ARTICLE 2" in skeleton
    assert set(index) == {"art_1", "art_2"}


def test_scaffold_indents_by_depth():
    anchors = [
        _anchor("chapter", "CHAPTER", "1", depth=0, akn_eid="chp_1"),
        _anchor(
            "article",
            "ARTICLE",
            "1",
            depth=1,
            parent_eid="chp_1",
            akn_eid="chp_1__art_1",
        ),
        _anchor(
            "article",
            "ARTICLE",
            "2",
            depth=1,
            parent_eid="chp_1",
            akn_eid="chp_1__art_2",
        ),
    ]
    skeleton, _ = scaffold_from_anchors(anchors)
    lines = skeleton.splitlines()
    chap_line = next(line for line in lines if "CHAPTER" in line)
    art_line = next(line for line in lines if "ARTICLE 1" in line)
    # CHAPTER lives directly under BODY (one indent level); ARTICLE is one deeper.
    assert chap_line.startswith("  CHAPTER")
    assert art_line.startswith("    ARTICLE")


def test_scaffold_handles_empty_anchors():
    skeleton, index = scaffold_from_anchors([])
    assert "BODY" in skeleton
    assert index == {}


def test_scaffold_includes_preface():
    anchors = [_anchor("article", "ARTICLE", "1", akn_eid="art_1")]
    skeleton, _ = scaffold_from_anchors(anchors, preface="Issued by the Council.")
    assert "PREFACE" in skeleton
    assert "Issued by the Council." in skeleton


def test_body_block_rejects_structural_keyword_in_lines():
    with pytest.raises(ValidationError) as excinfo:
        BodyBlock(eid="art_1", lines=["This is fine.", "ARTICLE 99 - smuggled"])
    assert "structural keyword" in str(excinfo.value)


def test_body_block_accepts_normal_body():
    block = BodyBlock(eid="art_1", heading="Definitions", lines=["A 'person' means …", "(1) blah"])
    assert block.eid == "art_1"
    assert len(block.lines) == 2


def test_body_fill_response_round_trips():
    payload = {
        "bodies": [
            {
                "eid": "art_1",
                "heading": "Application",
                "lines": ["This Code applies …"],
            },
            {"eid": "art_2", "lines": []},
        ]
    }
    parsed = BodyFillResponse.model_validate(payload)
    assert len(parsed.bodies) == 2
    assert parsed.bodies[0].heading == "Application"


def test_assemble_splices_body_under_correct_anchor():
    anchors = [
        _anchor("article", "ARTICLE", "1", akn_eid="art_1"),
        _anchor("article", "ARTICLE", "2", char_offset=100, akn_eid="art_2"),
    ]
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[
            BodyBlock(eid="art_1", heading="Scope", lines=["This law applies to all persons."]),
            BodyBlock(eid="art_2", heading="Definitions", lines=["'Person' includes …"]),
        ]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    assert "ARTICLE 1 - Scope" in filled
    assert "This law applies to all persons." in filled
    assert "ARTICLE 2 - Definitions" in filled
    assert "'Person' includes …" in filled


@pytest.mark.parametrize(
    ("kind", "keyword", "number", "depth", "heading", "expected"),
    [
        ("article", "ARTICLE", "1", 0, None, "  ARTICLE 1"),
        ("article", "ARTICLE", "1", 0, "Scope", "  ARTICLE 1 - Scope"),
        # Empty number: the trailing space is stripped, not left dangling.
        ("chapter", "CHAPTER", "", 0, None, "  CHAPTER"),
        ("chapter", "CHAPTER", "", 0, "General", "  CHAPTER - General"),
        # A schedule roots at column 0 (Bluebell parses it as an attachment there).
        ("schedule", "SCHEDULE", "1", 0, None, "SCHEDULE 1"),
        # Deeper body anchors indent two spaces per level (depth + 1).
        ("paragraph", "PARAGRAPH", "2", 1, None, "    PARAGRAPH 2"),
    ],
)
def test_marker_header_format(kind, keyword, number, depth, heading, expected):
    """The single header source both emit sites now share; pin its exact bytes
    so the format cannot drift from the _is_header matcher that re-detects it."""
    anchor = _anchor(kind, keyword, number, depth=depth, akn_eid="x")
    assert _marker_header(anchor, set(), heading) == expected


def test_numberless_header_round_trips_through_assemble():
    """A container with no number emits a bare `CHAPTER` header; assemble's
    `_is_header` must still recognise it (the `== kw` branch) and splice the
    body. This exercises the emit↔matcher agreement for the numberless case."""
    anchors = [_anchor("chapter", "CHAPTER", "", akn_eid="chp_1")]
    skeleton, index = scaffold_from_anchors(anchors)
    assert "CHAPTER\n" in skeleton  # bare keyword, no trailing number
    filled = assemble_filled_scaffold(
        skeleton,
        index,
        BodyFillResponse(bodies=[BodyBlock(eid="chp_1", lines=["Body text."])]),
    )
    assert "Body text." in filled


def test_assemble_drops_orphan_body_blocks():
    anchors = [_anchor("article", "ARTICLE", "1", akn_eid="art_1")]
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[
            BodyBlock(eid="art_1", lines=["fine"]),
            BodyBlock(eid="ghost_99", lines=["this has no home"]),
        ]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    assert "fine" in filled
    assert "this has no home" not in filled


def test_assemble_handles_missing_body_blocks():
    # Anchor present, but no body-fill entry (e.g. repealed article).
    anchors = [
        _anchor("article", "ARTICLE", "1", akn_eid="art_1"),
        _anchor("article", "ARTICLE", "2", char_offset=100, akn_eid="art_2"),
    ]
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[BodyBlock(eid="art_2", lines=["only the second is filled"])]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    assert "ARTICLE 1" in filled
    assert "ARTICLE 2" in filled
    assert "only the second is filled" in filled


def test_join_soft_wraps_merges_pdf_line_fragments():
    from codify.pipeline.enrich.scaffold import _join_soft_wraps

    fragments = [
        "1. Autoriteti nuk mund të nënshkruajë kontratën",
        "pa kaluar një periudhë, si më",
        "poshtë:",
        "a) të paktën 10 ditë kalendarike, mbi kufirin e",
        "lartë monetar;",
        "b) të paktën 7 ditë kalendarike, nën kufirin e",
        "lartë monetar;",
    ]
    out = _join_soft_wraps(fragments)
    # Wrapped continuations folded into their logical paragraph.
    assert out == [
        "1. Autoriteti nuk mund të nënshkruajë kontratën pa kaluar një periudhë, si më poshtë:",
        "a) të paktën 10 ditë kalendarike, mbi kufirin e lartë monetar;",
        "b) të paktën 7 ditë kalendarike, nën kufirin e lartë monetar;",
    ]


def test_join_soft_wraps_keeps_complete_paragraphs_apart():
    from codify.pipeline.enrich.scaffold import _join_soft_wraps

    # Each line already ends on terminal punctuation → no merging.
    lines = ["First sentence ends here.", "Second one too.", "Third."]
    assert _join_soft_wraps(lines) == lines


def test_join_soft_wraps_does_not_glue_prose_onto_a_table_row():
    """A pipe row ends in `|`, not terminal punctuation, so prose immediately
    following a table with no blank line must not merge onto the last row:
    that would leave a line no longer ending in `|` and nest_pipe_tables could
    miss the row, or the whole table."""
    from codify.pipeline.enrich.scaffold import _join_soft_wraps

    lines = ["| Alice | 30 |", "This sentence follows the table directly."]
    assert _join_soft_wraps(lines) == lines


def test_end_to_end_dot_paragraphs_and_bracketed_points_no_dupe_nums():
    """Bridge test: the W10h structurer-flatten bug, fixed at `nest_enumerated_lines`,
    flows correctly through scaffold → assemble → Bluebell → AKN → validate.

    The body-fill response carries the UA-style `1. paragraph / 1) 2) point / 2.
    paragraph / 1) point` pattern. Pre-fix the validator surfaced
    `duplicate_number` warnings because the points flattened into the paragraph
    siblings; post-fix the points nest under their paragraph and every <num> is
    unique among siblings.
    """
    anchors = [_anchor("article", "ARTICLE", "1", akn_eid="art_1")]
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[
            BodyBlock(
                eid="art_1",
                lines=[
                    "1. First paragraph defines key terms.",
                    "1) point alpha of paragraph 1",
                    "2) point bravo of paragraph 1",
                    "3) point charlie of paragraph 1",
                    "2. Second paragraph names the duties.",
                    "1) duty one",
                    "2) duty two",
                ],
            ),
        ]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    akn = parse_to_akn(filled, country="xa", doctype="act", date="2026-01-01", number="1")
    issues = validate_akn(akn)
    dupe = [i for i in issues if i["check"] == "duplicate_number"]
    assert dupe == [], f"expected zero duplicate_number issues, got: {dupe}"


def test_end_to_end_pipe_table_becomes_a_real_table_element():
    """Bridge test for issue #1168: body-fill transcribes an OCR table verbatim as
    flat `| cell | cell |` lines (its rule is transcribe, not restructure), so
    pre-fix these landed as six `<p>` paragraphs of literal pipe text. Post-fix
    `nest_pipe_tables` rewrites the run into `TABLE`/`TR`/`TH`/`TC` before Bluebell
    parses it, giving a real `<table>` with stable eIds, validator-clean.
    """
    anchors = [_anchor("section", "SECTION", "4", akn_eid="sec_4", heading="Schedule of penalties")]
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[
            BodyBlock(
                eid="sec_4",
                heading="Schedule of penalties",
                lines=[
                    "The following penalties apply to the offences specified:",
                    "",
                    "| Offence | Maximum fine | Maximum imprisonment |",
                    "",
                    "| --- | --- | --- |",
                    "",
                    "| Littering in a public place | 200 dollars | None |",
                    "",
                    "| Depositing waste unlawfully | 2,000 dollars | 6 months |",
                ],
            ),
        ]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    assert "| Offence |" not in filled, "a raw pipe row must not reach Bluebell text"
    akn = parse_to_akn(filled, country="xa", doctype="act", date="2026-01-01", number="1")
    assert re.search(r'<table eId="sec_4__table_1">', akn)
    assert "Littering in a public place" in akn
    issues = validate_akn(akn)
    # the uncitable_work_uri warning is this test's placeholder FRBR uri, not the table
    blocking = [i for i in issues if i["check"] != "uncitable_work_uri"]
    assert blocking == [], f"expected a validator-clean table, got: {blocking}"


class TestQuotedAmendmentSkip:
    """Quoted-amendment anchors are dropped from the scaffold, not emitted
    as peer headers. `<mod><quotedStructure>` emission is deferred; embedded
    articles must not surface as top-level siblings."""

    def test_quoted_anchor_omitted(self):
        anchors = [
            _anchor("article", "ARTICLE", "1", akn_eid="art_1"),
            _anchor("article", "ARTICLE", "2", akn_eid="art_2"),
            _anchor("article", "ARTICLE", "9", quoted_amendment=True),
            _anchor("article", "ARTICLE", "3", akn_eid="art_3"),
        ]
        skeleton, index = scaffold_from_anchors(anchors)
        assert "ARTICLE 9" not in skeleton
        assert set(index) == {"art_1", "art_2", "art_3"}


class TestContainerHeadingEmit:
    """Container anchors carrying a captured heading emit `KEYWORD N - title`
    so Bluebell produces `<heading>`; assemble preserves it even though
    containers never receive a BodyBlock."""

    def test_scaffold_emits_container_heading(self):
        anchors = [
            _anchor("chapter", "CHAPTER", "1", akn_eid="chp_1", heading="General provisions"),
            _anchor(
                "article",
                "ARTICLE",
                "1",
                depth=1,
                parent_eid="chp_1",
                akn_eid="chp_1__art_1",
            ),
        ]
        skeleton, _ = scaffold_from_anchors(anchors)
        assert "CHAPTER 1 - General provisions" in skeleton

    def test_container_heading_round_trips_to_akn_heading(self):
        anchors = [
            _anchor("chapter", "CHAPTER", "1", akn_eid="chp_1", heading="General provisions"),
            _anchor(
                "article",
                "ARTICLE",
                "1",
                depth=1,
                parent_eid="chp_1",
                akn_eid="chp_1__art_1",
            ),
        ]
        skeleton, index = scaffold_from_anchors(anchors)
        response = BodyFillResponse(bodies=[BodyBlock(eid="chp_1__art_1", lines=["Body text."])])
        filled = assemble_filled_scaffold(skeleton, index, response)
        akn = parse_to_akn(filled, country="xa", doctype="act", date="2026-01-01", number="1")
        assert "<heading>General provisions</heading>" in akn


def test_scaffold_emits_preamble_block():
    from codify.pipeline.enrich.scaffold import scaffold_from_anchors

    scaffold, _ = scaffold_from_anchors(
        [], preface="State of Nowhere", preamble="Having reviewed the Basic Law,"
    )
    lines = scaffold.splitlines()
    assert "PREFACE" in lines and "PREAMBLE" in lines
    assert lines.index("PREFACE") < lines.index("PREAMBLE") < lines.index("BODY")
    assert "  Having reviewed the Basic Law," in lines


class TestCapturedHeadingSurvivesFill:
    """A captured heading must survive filling and remain stable across runs."""

    def _assemble(self, anchor: StructuralAnchor, filled_heading: str | None) -> str:
        scaffold, eid_to_anchor = scaffold_from_anchors([anchor])
        bodies = [BodyBlock(eid=anchor.akn_eid, heading=filled_heading, lines=["Body text."])]
        return assemble_filled_scaffold(scaffold, eid_to_anchor, BodyFillResponse(bodies=bodies))

    def test_model_cannot_replace_a_captured_heading_with_a_bare_ordinal(self) -> None:
        # A numeric ordinal must not become `<heading>الأول</heading>`,
        # duplicating its own num, over a source reading "الفصل الأول / تعاريف".
        anchor = _anchor("chapter", "CHAPTER", "الأول", akn_eid="chp_1", heading="تعاريف")
        out = self._assemble(anchor, "الأول")
        assert "تعاريف" in out
        assert "CHAPTER الأول - الأول" not in out

    def test_model_cannot_replace_a_captured_heading_at_all(self) -> None:
        anchor = _anchor("chapter", "CHAPTER", "1", akn_eid="chp_1", heading="Definitions")
        out = self._assemble(anchor, "Something the model preferred")
        assert "Definitions" in out
        assert "Something the model preferred" not in out

    def test_model_heading_still_used_where_source_captured_none(self) -> None:
        anchor = _anchor("chapter", "CHAPTER", "1", akn_eid="chp_1", heading=None)
        out = self._assemble(anchor, "General Provisions")
        assert "General Provisions" in out

    def test_bare_ordinal_fill_is_rejected_when_no_heading_was_captured(self) -> None:
        # Better an untitled chapter than one whose title restates its number.
        anchor = _anchor("chapter", "CHAPTER", "1", akn_eid="chp_1", heading=None)
        out = self._assemble(anchor, "الأول")
        assert "CHAPTER 1" in out
        assert "- الأول" not in out


class TestHeadingRestatesNumValidator:
    """A bare ordinal heading is reported as a missing descriptive title."""

    def test_flags_a_heading_that_only_repeats_its_num(self) -> None:
        akn = (
            '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
            "<act><meta><identification source='#codify'/></meta><body>"
            '<chapter eId="chp_1"><num>الأول</num><heading>الأول</heading>'
            '<article eId="art_1"><num>1</num><content><p>x</p></content></article>'
            "</chapter></body></act></akomaNtoso>"
        )
        checks = [i["check"] for i in validate_akn(akn)]
        assert "container_heading_restates_num" in checks
        # It is not also reported as a missing title: one finding per container.
        assert "missing_container_title" not in checks

    def test_descriptive_heading_is_clean(self) -> None:
        akn = (
            '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
            "<act><meta><identification source='#codify'/></meta><body>"
            '<chapter eId="chp_1"><num>الأول</num><heading>تعاريف</heading>'
            '<article eId="art_1"><num>1</num><content><p>x</p></content></article>'
            "</chapter></body></act></akomaNtoso>"
        )
        checks = [i["check"] for i in validate_akn(akn)]
        assert "container_heading_restates_num" not in checks


class TestHeadingComparisonAcrossDigitScripts:
    """The scanner preserves the source's digit script, so the same ordinal
    reaches the check as a word on one side and a non-Latin digit on the
    other."""

    def test_ordinal_word_against_arabic_indic_digit(self) -> None:
        from codify.pipeline.enrich.arabic_normalise import heading_restates_number

        assert heading_restates_number("الأول", "١")
        assert heading_restates_number("الخامس", "٥")

    def test_comparison_is_symmetric(self) -> None:
        from codify.pipeline.enrich.arabic_normalise import heading_restates_number

        assert heading_restates_number("1", "الأول")
        assert heading_restates_number("١", "1")

    def test_a_real_title_still_passes_in_either_script(self) -> None:
        from codify.pipeline.enrich.arabic_normalise import heading_restates_number

        assert not heading_restates_number("تعاريف", "١")
        assert not heading_restates_number("Definitions", "1")


@pytest.mark.parametrize(
    "line",
    [
        "Section 6 of the same Act is hereby amended to read as follows:",
        "Sec. 6 of the same Act is hereby amended.",
        "the provisions of Section 17, Rule 119 shall apply.",
    ],
)
def test_a_citation_is_body_text_not_structure(line: str) -> None:
    """Bluebell reads these as prose. Refusing them failed every window of an
    amending act, whose sections nearly all open this way, and the document
    came back with no body at all."""
    assert BodyBlock(eid="sec_1", lines=[line]).lines == [line]


@pytest.mark.parametrize("line", ["SECTION 6", "SECTION 6. Coverage. - Applies.", "CHAPTER II"])
def test_an_upper_case_marker_is_still_refused(line: str) -> None:
    """The counterpart: Bluebell would make these a nested element."""
    with pytest.raises(ValueError, match="structural keyword"):
        BodyBlock(eid="sec_1", lines=[line])


def test_bluebell_agrees_with_the_boundary() -> None:
    """The rule is only right if it matches what Bluebell actually does."""

    def sections(body: str) -> list[str]:
        src = f"PREFACE\n\n  P.\n\nBODY\n\n  SECTION 1\n\n    {body}\n"
        akn = parse_to_akn(src, country="xa", doctype="act", date="2018-05-28", number="1")
        return re.findall(r'<section eId="([^"]+)"', akn)

    assert sections("Section 6 of the same Act is hereby amended.") == ["sec_1"]
    assert sections("SECTION 6 of the same Act is hereby amended.") != ["sec_1"]


def test_assembly_preserves_repeated_markers_after_unpunctuated_text(monkeypatch):
    import string

    labels = list(string.ascii_lowercase) + [c * 2 for c in string.ascii_lowercase]
    labels += [c * 3 for c in string.ascii_lowercase[:15]]
    anchors = [_anchor("section", "SECTION", "1", akn_eid="sec_1")]
    scaffold, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[BodyBlock(eid="sec_1", lines=[f"{n}) Example definition" for n in labels])]
    )
    filled = assemble_filled_scaffold(scaffold, index, response)
    points = [line.strip() for line in filled.splitlines() if line.strip().startswith("POINT ")]
    assert points == [f"POINT {n})" for n in labels]
    from codify.jurisdictions import JurisdictionConfig

    cfg = JurisdictionConfig(code="xx", name="Example", tradition=["common_law"], languages=["eng"])
    monkeypatch.setattr("codify.jurisdictions.load_config", lambda _: cfg)
    xml = parse_to_akn(filled, "xx", date="2031", number="7")
    from lxml import etree

    root = etree.fromstring(xml.encode())
    ns = {"a": root.nsmap[None]}
    section = root.find(".//a:section", ns)
    assert section is not None
    assert len(section.findall("a:point", ns)) == len(labels)
