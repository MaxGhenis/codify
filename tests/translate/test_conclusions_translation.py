"""Conclusions has to be translated, or an English export ends in Arabic.

The attestation moved out of the final article into `<conclusions>`. Before
that it was translated as part of the article; after it, nothing reached it,
because the front-matter slot extractors covered `<preface>` and `<preamble>`
only, and the per-eId body loop never leaves `<body>`.
"""

from __future__ import annotations

from lxml import etree

from codify.translate.translate import _extract_conclusions_lines
from codify.translate.write import apply_translation_to_akn, conclusions_p_slots

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

# The shape `emit_conclusions` produces: place-and-date lines as direct `<p>`s,
# the signatory pair grouped into a `<blockContainer>`.
_AKN = f"""<akomaNtoso xmlns="{NS}"><act name="act">
  <meta><identification source="#src"><FRBRWork>
    <FRBRthis value="/akn/xz/act/2004/41"/><FRBRuri value="/akn/xz/act/2004/41"/>
    <FRBRdate date="2004-04-12" name="enacted"/></FRBRWork>
    <FRBRExpression><FRBRthis value="/akn/xz/act/2004/41/ara@2004-04-12"/>
    <FRBRuri value="/akn/xz/act/2004/41/ara@2004-04-12"/>
    <FRBRdate date="2004-04-12" name="validFrom"/><FRBRlanguage language="ara"/>
    </FRBRExpression></identification></meta>
  <body><article eId="art_1"><num>1</num><content><p>نص المادة</p></content></article></body>
  <conclusions>
    <p>صدر بمدينة زرزورة بتاريخ : ١٢ / ٤ / ٢٠٠٤ ميلادية</p>
    <p>الموافق : ٢٢ / صفر / ١٤٢٥ هجرية</p>
    <blockContainer eId="sig_1">
      <p>نادر بن سالم</p>
      <p>رئيس مجلس الوزراء</p>
    </blockContainer>
  </conclusions>
</act></akomaNtoso>"""


def _conclusions_texts(xml: str) -> list[str]:
    root = etree.fromstring(xml.encode())
    return ["".join(p.itertext()).strip() for p in conclusions_p_slots(root)]


class TestSlots:
    def test_place_date_and_signatory_are_all_slots(self) -> None:
        """Four lines: two dates as direct `<p>`s, name and role in the container."""
        root = etree.fromstring(_AKN.encode())
        assert len(conclusions_p_slots(root)) == 4

    def test_the_extractor_and_the_patcher_see_the_same_slots(self) -> None:
        """The patch side re-invokes the slot function on the cloned root, so a
        divergence here is what the strict zip exists to catch."""
        root = etree.fromstring(_AKN.encode())
        assert len(conclusions_p_slots(root)) == len(_extract_conclusions_lines(_AKN))

    def test_a_document_with_no_conclusions_yields_no_slots(self) -> None:
        bare = _AKN[: _AKN.index("<conclusions>")] + "</act></akomaNtoso>"
        assert conclusions_p_slots(etree.fromstring(bare.encode())) == []

    def test_a_mixed_content_paragraph_is_never_a_patch_slot(self) -> None:
        """`_set_p_text` flattens inline markup, so a `<p>` carrying semantic
        children must not be offered as a slot."""
        mixed = _AKN.replace(
            "<p>نادر بن سالم</p>", "<p>Signed by <docType>Decree</docType> holder</p>"
        )
        assert len(conclusions_p_slots(etree.fromstring(mixed.encode()))) == 3


class TestTheAuditMeasuresIt:
    def test_conclusions_text_reaches_the_language_ratio(self) -> None:
        """Translating the attestation but leaving it out of the ratio means a
        block that comes back in the source script passes the delivery gate."""
        from codify.translate.translate import _akn_text_content

        assert "نادر بن سالم" in _akn_text_content(_AKN)


class TestPatch:
    _LINES = [
        "Done at Zerzura on 12 April 2004",
        "corresponding to 22 Safar 1425",
        "Nadir bin Salem",
        "Prime Minister",
    ]

    def test_the_attestation_is_translated(self) -> None:
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", conclusions_lines=self._LINES
        )
        assert _conclusions_texts(out) == self._LINES

    def test_omitting_the_lines_leaves_the_source_attestation_alone(self) -> None:
        """The parameter is optional, so a caller that has not been taught about
        conclusions ships the source text rather than an empty block."""
        out = apply_translation_to_akn(_AKN, [], [], target_language="English")
        assert _conclusions_texts(out) == _conclusions_texts(_AKN)

    def test_the_signatory_block_survives_the_patch(self) -> None:
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", conclusions_lines=self._LINES
        )
        block = etree.fromstring(out.encode()).find(f".//{{{NS}}}blockContainer")
        assert block is not None and block.get("eId") == "sig_1"
        assert [p.text for p in block] == ["Nadir bin Salem", "Prime Minister"]

    def test_a_count_mismatch_fails_loudly(self) -> None:
        """Strict zip: half-translated attestation is worse than none, because it
        reads as authoritative."""
        import pytest

        with pytest.raises(ValueError):
            apply_translation_to_akn(
                _AKN, [], [], target_language="English", conclusions_lines=["only one"]
            )
