"""The schema check, and the splice that keeps `amend_provision` valid.

`amend_provision` round-trips through a body-only domain model, so without the
splice every repair-path write drops the front and back matter it never touched.
"""

from __future__ import annotations

import pytest
from lxml import etree

from codify.akn.io import parse_akn, to_akn
from codify.storage.schema_gate import AknSchemaError, gate_akn
from codify.storage.versions import _restore_non_body

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
# Strict AKN requires FRBRauthor and FRBRcountry; a meta without them is exactly
# what the gate exists to refuse, so the valid fixture has to carry them.
_META = """<meta><identification source="#codify">
  <FRBRWork><FRBRthis value="/akn/xz/act/2004/41"/><FRBRuri value="/akn/xz/act/2004/41"/>
    <FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>
    <FRBRcountry value="xz"/></FRBRWork>
  <FRBRExpression><FRBRthis value="/akn/xz/act/2004/41/eng@2004-04-12"/>
    <FRBRuri value="/akn/xz/act/2004/41/eng@2004-04-12"/>
    <FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>
    <FRBRlanguage language="eng"/></FRBRExpression>
  <FRBRManifestation><FRBRthis value="/akn/xz/act/2004/41/eng@2004-04-12.xml"/>
    <FRBRuri value="/akn/xz/act/2004/41/eng@2004-04-12.xml"/>
    <FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>
  </FRBRManifestation></identification></meta>"""
_BODY = '<body><article eId="art_1"><num>1</num><content><p>Body.</p></content></article></body>'


def _act(extra: str = "", body: str = _BODY) -> str:
    return f'<akomaNtoso xmlns="{NS}"><act name="act">{_META}{extra}{body}</act></akomaNtoso>'


class TestSchemaGate:
    def test_valid_akn_passes(self) -> None:
        gate_akn(_act(), where="test")

    def test_an_unknown_element_is_refused(self) -> None:
        bad = _act().replace("<p>Body.</p>", "<notAnAknElement>x</notAnAknElement>")
        with pytest.raises(AknSchemaError) as exc:
            gate_akn(bad, where="save_document_reporting")
        assert "save_document_reporting" in str(exc.value)

    def test_the_write_path_is_named_in_the_error(self) -> None:
        """A gate that says only "invalid" leaves the operator finding the
        producer by hand."""
        with pytest.raises(AknSchemaError, match="update_repaired_akn"):
            gate_akn("<akomaNtoso/>", where="update_repaired_akn")

    def test_an_empty_string_is_not_a_write(self) -> None:
        """Both writers default `akn_xml` to "", meaning caller supplied none."""
        gate_akn("", where="test")


class TestFrontMatterSurvivesTheRoundTrip:
    _FULL = _act(
        "<preface><p>PREFACE-TEXT</p></preface><preamble><p>PREAMBLE-TEXT</p></preamble>",
    ).replace("</act>", "<conclusions><p>CONCLUSIONS-TEXT</p></conclusions></act>")

    def test_the_emitter_alone_loses_all_of_it(self) -> None:
        """The premise. The domain model is body-shaped with `extra="forbid"`,
        so this is not a bug in `to_akn`, it is its scope."""
        emitted = to_akn(parse_akn(self._FULL))
        assert "PREFACE-TEXT" not in emitted
        assert "PREAMBLE-TEXT" not in emitted
        assert "CONCLUSIONS-TEXT" not in emitted

    def test_the_splice_puts_all_three_back(self) -> None:
        out = _restore_non_body(self._FULL, to_akn(parse_akn(self._FULL)))
        for marker in ("PREFACE-TEXT", "PREAMBLE-TEXT", "CONCLUSIONS-TEXT"):
            assert marker in out

    def test_akn_element_order_is_legal(self) -> None:
        """Front matter precedes the body and back matter follows it, so an
        append would put `<preface>` after `<body>` and fail the schema."""
        out = _restore_non_body(self._FULL, to_akn(parse_akn(self._FULL)))
        act = next(iter(etree.fromstring(out.encode())))
        assert [etree.QName(c).localname for c in act] == [
            "meta",
            "preface",
            "preamble",
            "body",
            "conclusions",
        ]

    def test_the_spliced_document_passes_the_gate(self) -> None:
        """The two halves of this change meet here: the splice has to produce
        something the gate accepts, or the repair path fails on every write."""
        gate_akn(_restore_non_body(self._FULL, to_akn(parse_akn(self._FULL))), where="test")

    def test_the_emitted_meta_wins(self) -> None:
        """`amend_provision` rewrites the FRBR expression URI before emitting,
        so carrying the source `<meta>` over would undo the version step."""
        emitted = to_akn(parse_akn(self._FULL)).replace(
            "/akn/xz/act/2004/41/eng@2004-04-12", "/akn/xz/act/2004/41/eng@2004-04-12.1"
        )
        out = _restore_non_body(self._FULL, emitted)
        assert "eng@2004-04-12.1" in out

    def test_a_document_with_no_front_matter_is_unchanged(self) -> None:
        emitted = to_akn(parse_akn(_act()))
        assert _restore_non_body(_act(), emitted) == emitted

    def test_two_back_matter_elements_keep_their_order(self) -> None:
        """Each one inserted after `<body>` would reverse the pair, and AKN
        fixes the order, so the schema refuses the result."""
        annex = (
            f'<attachments><attachment><doc name="annex">{_META}'
            "<mainBody><p>ANNEX</p></mainBody></doc></attachment></attachments>"
        )
        src = self._FULL.replace("</act>", f"{annex}</act>")
        out = _restore_non_body(src, to_akn(parse_akn(src)))
        act = next(iter(etree.fromstring(out.encode())))
        names = [etree.QName(c).localname for c in act]
        assert names[-2:] == ["conclusions", "attachments"]
        gate_akn(out, where="test")
