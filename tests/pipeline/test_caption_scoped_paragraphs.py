"""Declared captions distinguish host paragraphs from forward cited judgments."""

import re

import pytest
from pydantic import ValidationError

from codify.jurisdictions import HierarchyEntry
from codify.pipeline.enrich import anchors as _anchors
from codify.pipeline.enrich.anchors import _scan_declared_markers, scan_anchors_with_ambiguity
from codify.pipeline.enrich.scaffold import BodyBlock


@pytest.fixture(autouse=True)
def _fresh_anchor_caches():
    """Fixture configs stand in for xa; no per-country cache may outlive them."""
    yield
    for fn in vars(_anchors).values():
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


ENTRIES = [
    HierarchyEntry(
        local_term="Heading",
        akn_element="section",
        level="higher",
        marker_form="caption",
        captions=["FACTS", "REASONS", "ORDER"],
    ),
    HierarchyEntry(
        local_term="Paragraph",
        akn_element="paragraph",
        level="basic",
        marker_form="bracketed_decimal",
    ),
]


def test_forward_cited_paragraph_cannot_advance_host_sequence() -> None:
    source = """2. FACTS
[2.1] The applicant cites another judgment:
[3.12.4] The earlier court held that the decision was valid.
[2.2] The respondent filed evidence.
3. REASONS
[3.1] The court has jurisdiction.
[3.2] The claim fails.
[3.2.1] The first reason.
[3.3] The next reason.
5. ORDER
[5.1] The claim is dismissed.
[6.1] A separate opinion follows.
"""
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == [
        "2.1",
        "2.2",
        "3.1",
        "3.2",
        "3.2.1",
        "3.3",
        "5.1",
        "6.1",
    ]


def test_quoted_unnumbered_caption_cannot_duplicate_numbered_caption() -> None:
    source = (
        "2. FACTS\n[2.1] Earlier judgment:\nORDER\n[1.1] Earlier order.\n"
        "[2.2] Evidence.\n5. ORDER\n[5.1] Present order."
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [(a.kind, a.number) for a in anchors if a.kind == "section"] == [
        ("section", "2"),
        ("section", "5"),
    ]


def test_unnumbered_caption_without_numbered_counterpart_is_retained() -> None:
    anchors, _ = _scan_declared_markers("FACTS\n[1.1] Facts.\nORDER\n[2.1] Order.", 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["1", "2"]


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
def test_embedded_structural_line_in_body_element_is_refused(separator: str) -> None:
    with pytest.raises(ValidationError, match="structural keyword"):
        BodyBlock(
            eid="sec_2__para_4-3",
            lines=[f"Ordinary text.{separator}  PARAGRAPH A whole sentence treated as a number"],
        )


def test_embedded_multiline_prose_is_preserved() -> None:
    lines = ["Ordinary text.\nAnother sentence.", "Paragraph 3 is discussed."]
    assert BodyBlock(eid="para_1", lines=lines).lines == [
        "Ordinary text.",
        "Another sentence.",
        "Paragraph 3 is discussed.",
    ]


def test_forward_citation_within_caption_cannot_skip_longer_host_sequence() -> None:
    source = """3. REASONS
[3.1] Jurisdiction.
[3.9] Another court's interlocutory decision.
[3.2] Evidence.
[3.3] Submissions.
[3.4] Findings.
[3.9] The present court's later conclusion.
[3.10] Further reasons.
"""
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    paragraphs = [a for a in anchors if a.kind == "paragraph"]
    assert [a.number for a in paragraphs] == ["3.1", "3.2", "3.3", "3.4", "3.9", "3.10"]
    assert next(a for a in paragraphs if a.number == "3.9").char_offset == source.index(
        "[3.9] The present"
    )


def test_long_forward_quotation_stays_outside_later_caption() -> None:
    source = """2. FACTS
[2.1] Quoted reasons:
[3.1] Earlier court.
[3.2] Earlier court.
[3.3] Earlier court.
[3.4] Earlier court.
[2.2] Present evidence.
3. REASONS
[3.1] Present court.
"""
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    paragraphs = [a for a in anchors if a.kind == "paragraph"]
    assert [a.number for a in paragraphs] == ["2.1", "2.2", "3.1"]
    assert paragraphs[-1].char_offset == source.index("[3.1] Present")


def test_heading_cannot_inject_a_structural_line() -> None:
    with pytest.raises(ValidationError, match="structural keyword"):
        BodyBlock(
            eid="para_1", heading="Reasons\n  PARAGRAPH A sentence as a number", lines=["Text."]
        )


def test_heading_keyword_on_same_line_is_plain_heading_text() -> None:
    assert (
        BodyBlock(eid="para_1", heading="PARAGRAPH numbering", lines=[]).heading
        == "PARAGRAPH numbering"
    )


@pytest.mark.parametrize("tail", ["[3.19] earlier reasons.", "3.19 earlier reasons."])
def test_keyword_cannot_capture_only_prefix_of_decimal_number(tail: str) -> None:
    regex = re.compile(
        r"^[ \t]*(?:(?P<k_section>Section)|(?P<k_paragraph>Paragraph))"
        r"[ \t\[]+(?P<num>\d+)",
        re.MULTILINE,
    )
    source = f"Section 2\nParagraph {tail}\nParagraph 4. Actual heading.\n"
    scan = scan_anchors_with_ambiguity(source, regex)
    assert [a.number for a in scan.anchors if a.kind == "paragraph"] == ["4"]
    assert any(s.detail.get("reason") == "partial_decimal_number" for s in scan.ambiguity)


@pytest.mark.parametrize("tail", ["3 Actual heading.", "3. Actual heading.", "[3] Actual heading."])
def test_complete_keyword_number_keeps_trailing_punctuation(tail: str) -> None:
    regex = re.compile(
        r"^[ \t]*(?:(?P<k_section>Section)|(?P<k_paragraph>Paragraph))"
        r"[ \t\[]+(?P<num>\d+)",
        re.MULTILINE,
    )
    scan = scan_anchors_with_ambiguity(f"Section 2\nParagraph {tail}\n", regex)
    assert [a.number for a in scan.anchors if a.kind == "paragraph"] == ["3"]


def test_complete_declared_decimal_number_remains_an_anchor() -> None:
    anchors, _ = _scan_declared_markers("3. REASONS\n[3.19] The court's reasons.\n", 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["3.19"]


def test_marker_denominator_refuses_the_same_partial_keyword_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codify.jurisdictions import DocumentClass, JurisdictionConfig
    from codify.pipeline.enrich import anchors as anchors_module
    from codify.pipeline.enrich.anchors import _marker_numbers

    config = JurisdictionConfig(
        code="zz",
        name="Fixture",
        tradition=["civil_law"],
        languages=["eng"],
        document_classes={
            "act": DocumentClass(label="Judgment", basic_unit="paragraph", hierarchy=ENTRIES)
        },
    )
    text = (
        "3. REASONS\nParagraph [8.19] Earlier reasons.\n"
        "[3.19] Present reasons.\nParagraph 4. Heading."
    )
    monkeypatch.setattr(anchors_module, "load_config", lambda _code: config)
    try:
        assert _marker_numbers(text, config, "act", "paragraph") == {"3.19", "4"}
    finally:
        anchors_module._prose_filter_for.cache_clear()


def test_lower_numbered_quotation_cannot_replace_active_caption_sequence() -> None:
    source = (
        "3. REASONS\n[3.1] Jurisdiction.\n[3.2] Evidence.\n"
        "[1.1] Earlier court.\n[1.2] Earlier court.\n[1.3] Earlier court.\n"
        "[1.4] Earlier court.\n[1.5] Earlier court.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["3.1", "3.2"]


def test_uncaptioned_sequence_and_final_separate_opinion_remain_supported() -> None:
    uncaptioned, _ = _scan_declared_markers("[1.1] Opening.\n[1.2] Next.\n", 0, ENTRIES)
    assert [a.number for a in uncaptioned] == ["1.1", "1.2"]
    source = (
        "[1.1] Opening.\n3. REASONS\n[3.1] Reasons.\n"
        "5. ORDER\n[5.1] Order.\n[6.1] Separate opinion.\n[6.2] Further opinion.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == [
        "1.1",
        "3.1",
        "5.1",
        "6.1",
        "6.2",
    ]


def test_uncaptioned_lower_quotation_cannot_replace_host_sequence() -> None:
    source = (
        "[3.1] Present reasons.\n[3.2] Earlier judgment follows:\n"
        "[1.1] Quoted.\n[1.2] Quoted.\n[1.3] Quoted.\n[1.4] Quoted.\n[1.5] Quoted.\n"
        "[3.3] Present conclusion.\n[4.1] Separate opinion.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES[1:])
    assert [a.number for a in anchors] == ["3.1", "3.2", "3.3", "4.1"]


def test_uncaptioned_sequence_does_not_infer_which_forward_marker_is_quoted() -> None:
    source = "[3.1] Reasons.\n[3.9] Decision.\n[3.2] Earlier text.\n[3.3] Earlier text.\n"
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors] == ["3.1", "3.9"]


def test_long_quotation_cannot_compete_across_numbered_caption_intervals() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n"
        "[9.1] Quoted.\n[9.2] Quoted.\n[9.3] Quoted.\n[9.4] Quoted.\n[9.5] Quoted.\n[9.6] Quoted.\n"
        "[2.2] Host.\n3. REASONS\n[3.1] Host.\n[3.2] Host.\n"
        "5. ORDER\n[5.1] Order.\n[6.1] Separate opinion.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == [
        "2.1",
        "2.2",
        "3.1",
        "3.2",
        "5.1",
        "6.1",
    ]


def test_later_caption_keeps_its_sequence_after_long_final_opinion_prefix() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n[8.1] Quoted.\n[8.2] Quoted.\n[8.3] Quoted.\n[8.4] Quoted.\n"
        "3. REASONS\n[3.1] Host.\n[3.2] Host.\n[4.1] Separate opinion.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "3.1", "3.2", "4.1"]


def test_quoted_numbered_caption_cannot_claim_later_host_boundary() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n[1.1] Quoted order.\n[2.2] Host.\n"
        "3. REASONS\n[3.1] Host.\n5. ORDER\n[5.1] Actual order.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1", "5.1"]
    sections = [a for a in anchors if a.kind == "section"]
    assert [a.number for a in sections] == ["2", "3", "5"]
    assert sections[-1].char_offset == source.rindex("5. ORDER")


@pytest.mark.parametrize("suffix", ["", "[6.1] Actual separate opinion.\n[6.2] Continued.\n"])
def test_separate_opinion_must_follow_final_caption_host_sequence(suffix: str) -> None:
    source = (
        "5. ORDER\n[5.1] Host.\n[6.1] Quoted.\n[6.2] Quoted.\n"
        "[6.3] Quoted.\n[6.4] Quoted.\n[5.2] Final host order.\n" + suffix
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    paragraphs = [a for a in anchors if a.kind == "paragraph"]
    assert [a.number for a in paragraphs] == (
        ["5.1", "5.2", "6.1", "6.2"] if suffix else ["5.1", "5.2"]
    )
    if suffix:
        assert paragraphs[2].char_offset == source.index("[6.1] Actual")


def test_unnumbered_final_order_keeps_following_separate_opinion() -> None:
    source = "5. ORDER\nThe claim is dismissed.\n[6.1] Separate opinion.\n[6.2] Reasons.\n"
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["6.1", "6.2"]


def test_equal_caption_waits_for_resumed_previous_host_sequence() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n3. REASONS\n[3.1] Quoted.\n[2.2] Host resumes.\n"
        "3. REASONS\n[3.1] Actual reasons.\n5. ORDER\n[5.1] Order.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1", "5.1"]
    assert next(
        a for a in anchors if a.kind == "section" and a.number == "3"
    ).char_offset == source.rindex("3. REASONS")


def test_caption_tie_keeps_earliest_maximal_sequence() -> None:
    entries = [
        ENTRIES[0].model_copy(update={"captions": ["FACTS", "REASONS", "CONCLUSION", "ORDER"]}),
        ENTRIES[1],
    ]
    source = (
        "2. FACTS\n[2.1] Facts.\n3. REASONS\n[3.1] Reasons.\n"
        "5. ORDER\n[5.1] Order.\n4. CONCLUSION\n[4.1] Quoted.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, entries)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "3", "5"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "3.1", "5.1"]


def test_repeated_host_caption_does_not_discard_preceding_host_paragraph() -> None:
    source = "3. REASONS\n[3.1] First page.\n3. REASONS\n[3.2] Next page.\n5. ORDER\n[5.1] Order.\n"
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert next(a for a in anchors if a.kind == "section" and a.number == "3").char_offset == 0
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["3.1", "3.2", "5.1"]


@pytest.mark.parametrize("keyword", ["ART", "CHAP", "PARA", "SEC", "SUBCHAP", "SUBPARA", "SUBSEC"])
def test_parser_alias_cannot_inject_body_or_heading_structure(keyword: str) -> None:
    from codify.pipeline.enrich.bluebell import parse_to_akn

    xml = parse_to_akn(
        f"BODY\n  SECTION 1\n    {keyword} Unexpected sentence\n      Body.\n",
        country="xa",
        doctype="act",
        date="2026-01-01",
        number="1",
    )
    assert "Unexpected sentence" in xml
    assert 'eId="sec_1__' in xml
    with pytest.raises(ValidationError, match="structural keyword"):
        BodyBlock(eid="sec_1", lines=[f"Prose.\n  {keyword} Unexpected sentence"])
    with pytest.raises(ValidationError, match="structural keyword"):
        BodyBlock(eid="sec_1", heading=f"Heading\n  {keyword} Unexpected sentence")


@pytest.mark.parametrize("keyword", ["Art", "Sec", "ARTISTIC", "SECTIONAL", "\\SEC"])
def test_alias_guard_keeps_prose_and_escaped_literal_controls(keyword: str) -> None:
    line = f"{keyword} words remain literal."
    assert BodyBlock(eid="sec_1", lines=[line]).lines == [line]


def test_structural_guard_covers_installed_parser_hierarchy_vocabulary() -> None:
    from importlib.resources import files

    from codify.pipeline.enrich.scaffold import _STRUCTURAL_LINE_RE

    grammar = files("bluebell").joinpath("akn.peg").read_text()
    definition = re.search(r"hier_element_name <- \((.*?)\)", grammar, re.DOTALL)
    assert definition is not None
    keywords = re.findall(r"'([A-Z]+)'", definition[1])
    assert keywords
    assert all(_STRUCTURAL_LINE_RE.match(f"{keyword} 1") for keyword in keywords)


def test_wrapped_citation_cannot_disambiguate_repeated_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codify.pipeline.enrich import anchors as anchors_module

    monkeypatch.setattr(
        anchors_module,
        "_prose_filter_for",
        lambda country: re.compile(r"\bunder\s*$" if country == "xa" else r"(?!)"),
    )
    source = (
        "3. REASONS\n[3.1] Host.\nThe earlier decision under\n[2.1] is cited.\n"
        "3. REASONS\n[3.2] Host continued.\n5. ORDER\n[5.1] Order.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES, country="xa")
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["3.1", "3.2", "5.1"]
    assert next(a for a in anchors if a.kind == "section" and a.number == "3").char_offset == 0


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("multiline_heading", [False, True])
def test_multiline_fill_keeps_heading_and_prose_under_its_section(
    separator: str, multiline_heading: bool
) -> None:
    from lxml import etree

    from codify.pipeline.enrich.anchors import StructuralAnchor
    from codify.pipeline.enrich.bluebell import parse_to_akn
    from codify.pipeline.enrich.scaffold import (
        BodyFillResponse,
        assemble_filled_scaffold,
        scaffold_from_anchors,
    )

    anchors = [
        StructuralAnchor("section", "Section", str(n), n, n, f"Section {n}", akn_eid=f"sec_{n}")
        for n in (1, 2)
    ]
    scaffold, mapping = scaffold_from_anchors(anchors)
    heading = f"Main{separator}heading" if multiline_heading else "Main heading"
    lines = (
        ["First sentence.", "Second sentence."]
        if multiline_heading
        else [f"First sentence.{separator}{separator}Second sentence."]
    )
    response = BodyFillResponse(
        bodies=[
            BodyBlock(eid="sec_1", heading=heading, lines=lines),
            BodyBlock(eid="sec_2", heading="Next", lines=["Neighbour."]),
        ]
    )
    assembled = assemble_filled_scaffold(scaffold, mapping, response)
    root = etree.fromstring(
        parse_to_akn(assembled, country="xa", doctype="act", date="2026-01-01", number="1").encode()
    )
    ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
    sections = root.findall(".//a:section", ns)
    assert len(sections) == 2
    assert "".join(sections[0].find("a:heading", ns).itertext()) == "Main heading"
    assert ["".join(p.itertext()) for p in sections[0].findall("a:content/a:p", ns)] == [
        "First sentence.",
        "Second sentence.",
    ]
    assert ["".join(p.itertext()) for p in sections[1].findall("a:content/a:p", ns)] == [
        "Neighbour."
    ]
    assert "\n    Second sentence.\n" in assembled


@pytest.mark.parametrize("quoted_body", ["[5.1] Quoted order.\n", "[1.1] Quoted order.\n"])
def test_single_forward_caption_yields_to_resumed_host(quoted_body: str) -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n"
        + quoted_body
        + "[2.2] Host resumes.\n3. REASONS\n[3.1] Actual reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "3"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1"]
    assert next(a for a in anchors if a.number == "3.1").char_offset == source.index("[3.1]")


def test_out_of_order_caption_keeps_earliest_without_host_resumption() -> None:
    source = "2. FACTS\n[2.1] Host.\n5. ORDER\n[5.1] Order.\n3. REASONS\n[3.1] Quoted.\n"
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "5"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "5.1"]


def test_resumed_caption_reaches_scaffold_and_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    from lxml import etree

    from codify.jurisdictions import DocumentClass, JurisdictionConfig
    from codify.pipeline.enrich import anchors as anchors_module
    from codify.pipeline.enrich.anchors import build_anchor_regex
    from codify.pipeline.enrich.bluebell import parse_to_akn
    from codify.pipeline.enrich.scaffold import scaffold_from_anchors

    config = JurisdictionConfig(
        code="xa",
        name="Fixture",
        tradition=["civil_law"],
        languages=["eng"],
        document_classes={
            "act": DocumentClass(label="Judgment", basic_unit="paragraph", hierarchy=ENTRIES)
        },
    )
    monkeypatch.setattr(anchors_module, "load_config", lambda _code: config)
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n[5.1] Quoted order.\n"
        "[2.2] Host resumes.\n3. REASONS\n[3.1] Actual reasons.\n"
    )
    try:
        scan = scan_anchors_with_ambiguity(
            source, build_anchor_regex(config, "act"), country="xa", doctype="act"
        )
        scaffold, _ = scaffold_from_anchors(scan.anchors)
        root = etree.fromstring(
            parse_to_akn(
                scaffold, country="xa", doctype="act", date="2026-01-01", number="1"
            ).encode()
        )
        ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
        assert root.xpath("//a:section/a:num/text()", namespaces=ns) == ["2", "3"]
        assert root.xpath("//a:section[a:num='3']/a:paragraph/a:num/text()", namespaces=ns) == [
            "3.1"
        ]
    finally:
        anchors_module._prose_filter_for.cache_clear()


def test_rejected_caption_cannot_hide_resumption_from_earlier_quote() -> None:
    entries = [
        ENTRIES[0].model_copy(update={"captions": ["FACTS", "REASONS", "CONCLUSION", "ORDER"]}),
        ENTRIES[1],
    ]
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n[5.1] Quoted.\n"
        "4. CONCLUSION\n[4.1] Quoted.\n[2.2] Host resumes.\n"
        "3. REASONS\n[3.1] Actual reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, entries)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "3"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1"]


def test_higher_caption_cannot_split_a_resumed_host_interval() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n[5.1] Quoted.\n[2.2] Host resumes.\n"
        "7. ORDER\n[7.1] Quoted.\n3. REASONS\n[3.1] Actual reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "3"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1"]


@pytest.mark.parametrize("quoted_number", ["1.2", "2.1"])
def test_lower_quotation_without_advancing_host_evidence_keeps_caption(quoted_number: str) -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n5. ORDER\n[5.1] Actual order.\n"
        f"[{quoted_number}] Earlier quotation.\n7. ORDER\n[7.1] Further order.\n"
        "3. REASONS\n[3.1] Earlier quoted reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "5", "7"]


def test_chp_remains_prose_in_installed_parser() -> None:
    from lxml import etree

    from codify.pipeline.enrich.bluebell import parse_to_akn

    block = BodyBlock(eid="sec_1", lines=["CHP Unexpected sentence"])
    root = etree.fromstring(
        parse_to_akn(
            f"BODY\n  SECTION 1\n    {block.lines[0]}\n      Body.\n",
            country="xa",
            doctype="act",
            date="2026-01-01",
            number="1",
        ).encode()
    )
    ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
    assert root.xpath("//a:section/a:content/a:p/text()", namespaces=ns) == [
        "CHP Unexpected sentence",
        "Body.",
    ]
    assert not root.xpath("//a:hcontainer|//a:section/a:section", namespaces=ns)


def test_forward_same_prefix_citation_does_not_hide_resumption() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n[2.9] Earlier citation.\n5. ORDER\n[5.1] Quoted.\n"
        "[2.2] Host resumes.\n3. REASONS\n[3.1] Actual reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "3"]
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "2.2", "3.1"]


def test_repeated_complete_contents_sequence_uses_body_captions() -> None:
    contents = "2. FACTS\n3. REASONS\n5. ORDER\n"
    source = (
        contents + "2. FACTS\n[2.1] Facts.\n3. REASONS\n[3.1] Reasons.\n5. ORDER\n[5.1] Order.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["2.1", "3.1", "5.1"]
    assert all(a.char_offset >= len(contents) for a in anchors)


def test_caption_only_contents_does_not_invent_a_body() -> None:
    source = "2. FACTS\n3. REASONS\n5. ORDER\n2. FACTS\n3. REASONS\n5. ORDER\n"
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.char_offset for a in anchors if a.kind == "section"] == [0, 9, 20]


def test_contents_sequence_keeps_body_after_opening_paragraph() -> None:
    contents = "2. FACTS\n3. REASONS\n5. ORDER\n"
    source = contents + (
        "[1.1] Opening.\n2. FACTS\n[2.1] Facts.\n3. REASONS\n[3.1] Reasons.\n"
        "5. ORDER\n[5.1] Order.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "paragraph"] == ["1.1", "2.1", "3.1", "5.1"]
    assert all(a.char_offset >= len(contents) for a in anchors)


def test_repeated_prior_paragraph_is_not_advancing_resumption() -> None:
    source = (
        "2. FACTS\n[2.1] Host.\n[2.2] Host continued.\n5. ORDER\n[5.1] Actual order.\n"
        "[2.2] Earlier quotation.\n7. ORDER\n[7.1] Further order.\n"
        "3. REASONS\n[3.1] Earlier quoted reasons.\n"
    )
    anchors, _ = _scan_declared_markers(source, 0, ENTRIES)
    assert [a.number for a in anchors if a.kind == "section"] == ["2", "5", "7"]


@pytest.mark.parametrize(
    "control",
    ["SCHEDULE 1", "APPENDIX 1", "ANNEXURE 1", "ATTACHMENT 1", "BODY", "PREFACE", "PREAMBLE"],
)
@pytest.mark.parametrize("in_heading", [False, True])
def test_document_controls_remain_literal_inside_filled_sections(
    control: str, in_heading: bool
) -> None:
    from lxml import etree

    from codify.pipeline.enrich.anchors import StructuralAnchor
    from codify.pipeline.enrich.bluebell import parse_to_akn
    from codify.pipeline.enrich.scaffold import (
        BodyFillResponse,
        assemble_filled_scaffold,
        scaffold_from_anchors,
    )

    anchors = [
        StructuralAnchor("section", "Section", str(n), n, n, f"Section {n}", akn_eid=f"sec_{n}")
        for n in (1, 2)
    ]
    scaffold, mapping = scaffold_from_anchors(anchors)
    block = BodyBlock(
        eid="sec_1",
        heading=f"Main\n{control}" if in_heading else "Main",
        lines=["Before.\n\nAfter." if in_heading else f"Before.\n\n{control}\n\nAfter."],
    )
    assembled = assemble_filled_scaffold(
        scaffold,
        mapping,
        BodyFillResponse(bodies=[block, BodyBlock(eid="sec_2", lines=["Neighbour."])]),
    )
    root = etree.fromstring(
        parse_to_akn(assembled, country="xa", date="2026-01-01", number="1").encode()
    )
    ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
    sections = root.findall(".//a:section", ns)
    assert [section.get("eId") for section in sections] == ["sec_1", "sec_2"]
    assert sections[0].findtext("a:heading", namespaces=ns) == (
        f"Main {control}" if in_heading else "Main"
    )
    assert sections[0].xpath("a:content/a:p/text()", namespaces=ns) == (
        ["Before.", "After."] if in_heading else ["Before.", control, "After."]
    )
    assert sections[1].xpath("a:content/a:p/text()", namespaces=ns) == ["Neighbour."]
    assert not root.xpath("//a:attachments|//a:preface|//a:preamble", namespaces=ns)
