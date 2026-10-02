# ruff: noqa: E501  # AKN XML test fixtures keep single-line elements
"""Unit tests for translation anchor traversal + scaffold builder."""

from __future__ import annotations

from codify.akn.io import parse_akn
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.translate.anchors import (
    build_translation_scaffold,
    walk_source_units,
)

_ACT_BLUEBELL = (
    "PREFACE\n  An Act\nBODY\n"
    "  CHAPTER 1 - General\n"
    "    ARTICLE 1 - Scope\n      This Act applies to all persons.\n"
    "    ARTICLE 2 - Definitions\n      In this Act, 'person' means an individual.\n"
    "  CHAPTER 2 - Duties\n"
    "    ARTICLE 3 - Duty of care\n      Every person shall act with care.\n"
)


def _doc():
    return parse_akn(parse_to_akn(_ACT_BLUEBELL, country="xa", doctype="act", number="0001"))


def test_walk_source_units_emits_one_per_element() -> None:
    units = list(walk_source_units(_doc()))
    # 2 chapters + 3 articles = 5 units; preface text isn't an element.
    kinds = [u.kind for u in units]
    assert kinds.count("chapter") == 2
    assert kinds.count("article") == 3


def test_walk_source_units_carries_eid_and_heading() -> None:
    units = list(walk_source_units(_doc()))
    headings = {u.heading for u in units if u.heading}
    assert headings == {"General", "Scope", "Definitions", "Duties", "Duty of care"}
    for u in units:
        assert u.akn_eid, "every unit carries the source eId verbatim"


def test_walk_source_units_body_text_excludes_children() -> None:
    """A chapter's body_text is its own (intro/text/wrap_up), never its children's."""
    units = list(walk_source_units(_doc()))
    chapters = [u for u in units if u.kind == "chapter"]
    for chapter in chapters:
        # Chapter has no own body in this fixture, bodies live on articles.
        assert "applies to all persons" not in chapter.body_text


def test_build_translation_scaffold_preserves_keyword_number_indent() -> None:
    """Scaffold uses Bluebell keywords + source numbers + depth-correct indent."""
    units = list(walk_source_units(_doc()))
    scaffold, eid_to_unit = build_translation_scaffold(units, preface_text="An Act")

    # The source numbers + Bluebell keywords appear verbatim, no translations.
    for line_no in (1, 2, 3):
        assert f"ARTICLE {line_no}" in scaffold
    assert "CHAPTER 1" in scaffold and "CHAPTER 2" in scaffold

    # eid_to_unit maps every walked unit by its source eId.
    assert {u.akn_eid for u in units} == set(eid_to_unit.keys())


def test_scaffold_roundtrip_through_parse_to_akn_preserves_eids() -> None:
    """CDFY-T1 invariant: parsing the scaffold (with empty bodies) yields the same eId set."""
    src_doc = _doc()
    units = list(walk_source_units(src_doc))
    scaffold, _ = build_translation_scaffold(units, preface_text="An Act")

    reparsed_xml = parse_to_akn(scaffold, country="xa", doctype="act", number="0001")
    reparsed_doc = parse_akn(reparsed_xml)
    reparsed_units = list(walk_source_units(reparsed_doc))

    assert {u.akn_eid for u in units} == {u.akn_eid for u in reparsed_units}


class TestAttachmentUnits:
    """Attachment content becomes translation units."""

    def _akn_with_attachment(self) -> str:
        from codify.pipeline.enrich.bluebell import parse_to_akn

        base = parse_to_akn(
            "BODY\n  ARTICLE 1\n    نص المادة.\n",
            country="xz",
            doctype="act",
            date="2016",
            number="18",
            language="ara",
        )
        attachments = """<attachments xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
      <attachment eId="att_1">
        <heading>سلم الدرجات والرواتب</heading>
        <doc name="schedule">
          <meta><identification source="#codify"/></meta>
          <mainBody>
            <point eId="att_1__point_1"><num>1.</num><content><p eId="att_1__point_1__p_1">الفئة الأولى.</p></content></point>
            <point eId="att_1__point_2"><num>2.</num><content><p eId="att_1__point_2__p_1">الفئة الثانية.</p></content></point>
          </mainBody>
        </doc>
      </attachment>
    </attachments>"""
        return base.replace("</act>", attachments + "</act>")

    def test_walk_includes_attachment_units(self) -> None:
        from codify.akn.io import parse_akn
        from codify.translate.anchors import walk_source

        xml = self._akn_with_attachment()
        units, _summary = walk_source(parse_akn(xml), xml)
        by_eid = {u.akn_eid: u for u in units}
        assert "att_1" in by_eid
        assert by_eid["att_1"].heading == "سلم الدرجات والرواتب"
        assert by_eid["att_1__point_1"].body_text == "الفئة الأولى."
        assert by_eid["att_1__point_2"].number == "2."

    def test_attachment_blocks_patch_into_translated_xml(self) -> None:
        from codify.translate.translate_bodies import TranslatedBlock
        from codify.translate.write import apply_translation_to_akn

        xml = self._akn_with_attachment()
        blocks = [
            TranslatedBlock(eid="art_1", heading=None, lines=["Article text."]),
            TranslatedBlock(eid="att_1", heading="Salary and grade scale", lines=[]),
            TranslatedBlock(eid="att_1__point_1", heading=None, lines=["First category."]),
            TranslatedBlock(eid="att_1__point_2", heading=None, lines=["Second category."]),
        ]
        out = apply_translation_to_akn(xml, blocks, [], target_language="English")
        assert "Salary and grade scale" in out
        assert "First category." in out and "Second category." in out
        assert "الفئة الأولى" not in out


class TestBarePAttachment:
    """Flat annex prose: bare <p eId> directly under <mainBody>."""

    def _akn(self) -> str:
        from codify.pipeline.enrich.bluebell import parse_to_akn

        base = parse_to_akn(
            "BODY\n  ARTICLE 1\n    نص المادة.\n",
            country="xz",
            doctype="act",
            date="2016",
            number="18",
            language="ara",
        )
        attachments = """<attachments xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
      <attachment eId="att_1">
        <heading>جدول الرسوم</heading>
        <doc name="schedule">
          <meta><identification source="#codify"/></meta>
          <mainBody>
            <p eId="att_1__p_1">رسم الترخيص مائة دينار.</p>
            <p eId="att_1__p_2">رسم التجديد خمسون دينارا.</p>
          </mainBody>
        </doc>
      </attachment>
    </attachments>"""
        return base.replace("</act>", attachments + "</act>")

    def test_bare_p_lines_become_units_and_patch(self) -> None:
        from codify.akn.io import parse_akn
        from codify.translate.anchors import walk_source
        from codify.translate.translate_bodies import TranslatedBlock
        from codify.translate.write import apply_translation_to_akn

        xml = self._akn()
        units, _ = walk_source(parse_akn(xml), xml)
        by_eid = {u.akn_eid: u for u in units}
        assert by_eid["att_1__p_1"].body_text == "رسم الترخيص مائة دينار."
        assert by_eid["att_1__p_2"].body_text == "رسم التجديد خمسون دينارا."
        blocks = [
            TranslatedBlock(eid="att_1", heading="Fee schedule", lines=[]),
            TranslatedBlock(
                eid="att_1__p_1", heading=None, lines=["Licence fee: one hundred dinars."]
            ),
            TranslatedBlock(eid="att_1__p_2", heading=None, lines=["Renewal fee: fifty dinars."]),
        ]
        out = apply_translation_to_akn(xml, blocks, [], target_language="English")
        assert "Licence fee: one hundred dinars." in out
        assert "رسم الترخيص" not in out

    def test_nested_attachment_units_not_duplicated(self) -> None:
        from codify.akn.io import parse_akn
        from codify.translate.anchors import walk_source

        xml = self._akn()
        units, _ = walk_source(parse_akn(xml), xml)
        eids = [u.akn_eid for u in units]
        assert len(eids) == len(set(eids))
