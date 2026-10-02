"""Tests for the enacting formula emitter."""

from __future__ import annotations

from datetime import date

import pytest
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.jurisdictions import JurisdictionConfigError
from codify.pipeline.enrich.enacting import (
    emit_enacting_formula,
    inject_enacting_formula,
    select_formula,
)


def _minimal_act(has_preface: bool = False) -> str:
    preface = "<preface><longTitle><p>An Act.</p></longTitle></preface>" if has_preface else ""
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    {preface}
    <body><section eId="sec_1"><num>1.</num><content><p>Short title.</p></content></section></body>
  </act>
</akomaNtoso>
'''


class TestSelectFormula:
    def test_the_later_era_is_selected_by_date(self):
        f = select_formula("gb", "act", date(2024, 1, 1))
        assert f is not None
        assert "King" in f.text

    def test_the_earlier_era_is_selected_by_date(self):
        f = select_formula("gb", "act", date(2020, 1, 1))
        assert f is not None
        assert "Queen" in f.text or "Majesty" in f.text

    def test_an_unknown_country_raises_rather_than_selecting_nothing(self):
        """A jurisdiction with no config and one whose formulae simply do not
        match both used to return None, so a run that never consulted a config
        was indistinguishable from one that consulted it and found no rule."""
        with pytest.raises(JurisdictionConfigError, match="zz"):
            select_formula("zz", "act", date(2024, 1, 1))

    def test_no_date_selects_nothing_when_all_formulae_are_era_bounded(self):
        # gb's formulae tile the timeline with from/to bounds; an undated
        # document cannot prove which era it is in, so nothing is injected.
        assert select_formula("gb", "act", None) is None

    def test_missing_jurisdiction_returns_none(self):
        # jp has no enacting_formulae per the corpus audit
        assert select_formula("jp", "act", date(2024, 1, 1)) is None

    def test_verified_preferred_on_ties(self):
        # Spot-check: where multiple entries match the same date, the
        # verified one wins if present. Not all configs have verified
        # entries, so we just assert the selection is stable for gb.
        f1 = select_formula("gb", "act", date(2024, 1, 1))
        f2 = select_formula("gb", "act", date(2024, 1, 1))
        assert f1 is not None and f2 is not None
        assert f1.text == f2.text


class TestDoctypeMatches:
    def test_scalar_document_class_matches_only_its_doctype(self):
        # 68 of 71 configs that set document_class use a scalar (ar's "ley" vs
        # "dnu"); a scalar must match one doctype, not act as a wildcard, else the
        # first formula is selected for every doctype.
        from codify.jurisdictions import EnactingFormula
        from codify.pipeline.enrich.enacting import _doctype_matches

        assert _doctype_matches(EnactingFormula(document_class="ley"), "ley") is True
        assert _doctype_matches(EnactingFormula(document_class="ley"), "dnu") is False
        assert _doctype_matches(EnactingFormula(document_class=None), "anything") is True
        assert _doctype_matches(EnactingFormula(document_class=["act", "si"]), "si") is True
        assert _doctype_matches(EnactingFormula(document_class=["act", "si"]), "loi") is False


class TestInjectEnactingFormula:
    def test_injects_into_new_preamble(self):
        from codify.jurisdictions import EnactingFormula

        f = EnactingFormula(text="ENACTED by the Parliament of Nowhere.", position="preamble")
        out = inject_enacting_formula(_minimal_act(), f)
        root = etree.fromstring(out.encode("utf-8"))
        formula = root.find(".//akn:preamble/akn:formula[@name='enactingFormula']", NS)
        assert formula is not None
        assert formula.find("akn:p", NS).text == "ENACTED by the Parliament of Nowhere."

    def test_preamble_sits_before_body(self):
        from codify.jurisdictions import EnactingFormula

        f = EnactingFormula(text="X", position="preamble")
        out = inject_enacting_formula(_minimal_act(), f)
        root = etree.fromstring(out.encode("utf-8"))
        act = root.find("akn:act", NS)
        children = [child.tag.split("}")[-1] for child in act]
        assert children.index("preamble") < children.index("body")

    def test_preserves_existing_preface(self):
        from codify.jurisdictions import EnactingFormula

        f = EnactingFormula(text="X", position="preamble")
        out = inject_enacting_formula(_minimal_act(has_preface=True), f)
        root = etree.fromstring(out.encode("utf-8"))
        act = root.find("akn:act", NS)
        children = [child.tag.split("}")[-1] for child in act]
        assert "preface" in children
        assert "preamble" in children
        # Order invariant: preface → preamble → body
        assert children.index("preface") < children.index("preamble") < children.index("body")

    def test_idempotent(self):
        from codify.jurisdictions import EnactingFormula

        f = EnactingFormula(text="Y", position="preamble")
        once = inject_enacting_formula(_minimal_act(), f)
        twice = inject_enacting_formula(once, f)
        root = etree.fromstring(twice.encode("utf-8"))
        formulae = root.findall(".//akn:formula[@name='enactingFormula']", NS)
        assert len(formulae) == 1


class TestEmitEnactingFormula:
    def test_end_to_end_current_era(self):
        out = emit_enacting_formula(_minimal_act(), "gb", "act", "2024-06-01")
        assert "King" in out
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is not None

    def test_an_unknown_country_raises_rather_than_emitting_a_formulaless_act(self):
        """The document this used to return is valid AKN with no enacting formula,
        which reads as a law that has none rather than a run that could not find
        the rule. Absence of a config is a fault, so it raises."""
        with pytest.raises(JurisdictionConfigError, match="zz"):
            emit_enacting_formula(_minimal_act(), "zz", "act", "2024-06-01")

    def test_noop_when_missing_from_config(self):
        out = emit_enacting_formula(_minimal_act(), "jp", "act", "2024-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None


def _arabic_act(preamble_text: str | None = None) -> str:
    preamble = f"<preamble><p>{preamble_text}</p></preamble>" if preamble_text else ""
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    {preamble}
    <body><article eId="art_1"><num>1</num><content><p>نص</p></content></article></body>
  </act>
</akomaNtoso>
'''


class TestSourceCarriesFormula:
    def test_marker_in_source_skips_injection(self):
        src = _arabic_act("رئيس جمهورية زرزورة، استناداً للقانون الأساسي، أصدر القرار الآتي بقانون:")
        out = emit_enacting_formula(src, "xz", "act", "2001-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_third_person_municipal_marker_skips_injection(self):
        # A municipal amendment names the council early, then enacts with a
        # third-person verb ("... قد أصدر التعديل التالي"), the council is not
        # adjacent to the verb, so only the self-enacting marker catches it.
        src = _arabic_act(
            "إن مجلس بلدية الواحة واستناداً إلى الصلاحيات المخولة له في المادة الخامسة "
            "عشر من قانون الهيئات المحلية رقم (2) لسنة 1995 قد أصدر التعديل التالي:-"
        )
        out = emit_enacting_formula(src, "xz", "act", "1999-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_assembly_era_without_marker_injects_fallback(self):
        out = emit_enacting_formula(_minimal_act(), "xz", "act", "2001-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        formula = root.find(".//akn:formula[@name='enactingFormula']", NS)
        assert formula is not None
        assert "مجلس الأمة" in formula.find("akn:p", NS).text

    def test_after_the_era_without_marker_injects_nothing(self):
        # The only config formula is era-bounded; it must never
        # leak outside its era via the date-filter fallback.
        out = emit_enacting_formula(_minimal_act(), "xz", "act", "2014-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_injected_formula_carries_codify_provenance(self):
        # refersTo="#codify" is what exempts the config fallback from the
        # fabricated_formula validator check; losing it would make every
        # injection self-report as fabrication. It is not @source because the
        # schema refuses that attribute on <formula>.
        out = emit_enacting_formula(_minimal_act(), "xz", "act", "2001-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        formula = root.find(".//akn:formula[@name='enactingFormula']", NS)
        assert formula.get("refersTo") == "#codify"
        assert formula.get("source") is None

    def test_marker_matches_despite_orthography_drift(self):
        # OCR text with bare alef vs config marker with hamza still matches.
        src = _arabic_act("رئيس جمهورية زرزورة اصدر القرار الاتي بقانون:")
        out = emit_enacting_formula(src, "xz", "act", "2001-06-01")
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_marker_check_idempotent_with_existing_formula(self):
        out = emit_enacting_formula(_minimal_act(), "xz", "act", "2001-06-01")
        again = emit_enacting_formula(out, "xz", "act", "2001-06-01")
        root = etree.fromstring(again.encode("utf-8"))
        assert len(root.findall(".//akn:formula[@name='enactingFormula']", NS)) == 1


class TestSplitOpeningMaterial:
    def test_recital_chain_splits_to_preamble(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        text = (
            "قانون رقم (7) لسنة 1999م بشأن المراعي\n"
            "رئيس جمهورية زرزورة\n"
            "بعد الاطلاع على القانون الأساسي،\n"
            "وبناءً على تنسيب مجلس الوزراء،\n"
            "أصدرنا القانون الآتي:\n"
        )
        preface, preamble = split_opening_material(text, "xz")
        assert preface is not None and "بشأن المراعي" in preface
        assert "رئيس جمهورية" in preface
        assert preamble is not None and preamble.startswith("بعد الاطلاع")
        assert "أصدرنا القانون الآتي" in preamble

    def test_no_opener_stays_preface(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        preface, preamble = split_opening_material("قانون رقم 5 لسنة 2000\n", "xz")
        assert preface is not None and preamble is None

    def test_empty(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        assert split_opening_material("  \n ", "xz") == (None, None)


class TestOpeningMaterialTerminator:
    """The truncation cut is `opening_material_terminators`, not
    `enacting_formula_markers`. An instrument that opens on an invocation lists
    it as formula evidence, and cutting there deletes the recitals."""

    PERBUP = (
        "BUPATI TANJUNG SERI\n"
        "PERATURAN BUPATI TANJUNG SERI NOMOR 7 TAHUN 2031\n"
        "DENGAN RAHMAT TUHAN YANG MAHA KUASA\n"
        "BUPATI TANJUNG SERI,\n"
        "Menimbang : a. bahwa berdasarkan ketentuan Pasal 34 Peraturan Menteri;\n"
        "Mengingat : 1. Undang-Undang Nomor 9 Tahun 2011;\n"
        "MEMUTUSKAN:\n"
        "Menetapkan : PERATURAN BUPATI TENTANG PERUBAHAN ATAS PERATURAN BUPATI.\n"
    )

    def test_invocation_does_not_truncate_the_recitals(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        preface, preamble = split_opening_material(self.PERBUP, "xl")
        both = f"{preface or ''}\n{preamble or ''}"
        assert "Menimbang" in both
        assert "Mengingat" in both
        assert "Menetapkan" in both

    def test_invocation_is_still_formula_evidence(self):
        """It stays in `enacting_formula_markers`, which is what suppresses the
        config fallback; the two fields answer different questions."""
        from codify.jurisdictions import load_config

        cfg = load_config("xl")
        assert "DENGAN RAHMAT TUHAN YANG MAHA KUASA" in cfg.enacting_formula_markers
        assert "DENGAN RAHMAT TUHAN YANG MAHA KUASA" not in cfg.opening_material_terminators

    def test_no_terminator_declared_means_no_cut(self):
        from codify.pipeline.enrich.enacting import _truncate_after_enacting_formula

        lines = ["TITLE", "recital", "tail"]
        assert _truncate_after_enacting_formula(lines, [], "xl") == lines


class TestFurnitureNeverBecomesOpeningMaterial:
    """Gazette furniture is dropped before the preface/preamble split, so it
    never reaches a paragraph slot."""

    def test_masthead_and_page_numbers_dropped_recitals_kept(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        text = (
            "الوقائع الزرزورية\n"
            "قانون رقم (3) لسنة 2000م بشأن الآبار\n"
            "صفحة 12\n"
            "بعد الاطلاع على القانون الأساسي،\n"
            "صفحة 13\n"
            "وبناءً على تنسيب مجلس الوزراء،\n"
            "أصدرنا القانون الآتي:\n"
        )
        preface, preamble = split_opening_material(text, "xz")
        assert preface is not None and preamble is not None
        both = f"{preface}\n{preamble}"
        assert "الوقائع الزرزورية" not in both
        assert "صفحة 12" not in both and "صفحة 13" not in both
        # The recitals interleaved with the footers survive, in order.
        assert preamble.startswith("بعد الاطلاع")
        assert "وبناءً على تنسيب" in preamble
        assert "أصدرنا القانون الآتي" in preamble

    def test_law_title_is_not_treated_as_furniture(self):
        # The running-title pattern in `ocr_header_patterns` matches the law's
        # own title, which is why whole-line disposal reads the narrower
        # `furniture_line_patterns` list instead. Dropping this line would
        # delete the document's title from its preface.
        from codify.pipeline.enrich.enacting import split_opening_material

        text = "قانون رقم (7) لسنة 1999 بشأن المراعي\nبعد الاطلاع على القانون الأساسي،\n"
        preface, _preamble = split_opening_material(text, "xz")
        assert preface is not None and "بشأن المراعي" in preface

    def test_partial_match_keeps_the_rest_of_the_line(self):
        # Provenance sharing a line with the masthead is retained: only the
        # matched phrase goes, not the issue number and year.
        from codify.pipeline.enrich.enacting import split_opening_material

        text = "الوقائع الزرزورية - العدد 45 - 2016\nبعد الاطلاع على القانون الأساسي،\n"
        preface, _preamble = split_opening_material(text, "xz")
        assert preface is not None
        assert "45" in preface and "2016" in preface
        assert "الوقائع الزرزورية" not in preface

    def test_all_furniture_opening_material_yields_nothing(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        assert split_opening_material("الوقائع الزرزورية\nصفحة 3\n", "xz") == (
            None,
            None,
        )

    def test_a_jurisdiction_declaring_no_patterns_is_untouched(self):
        """Declaring no furniture patterns is a real configuration, and the text
        passes through. Distinct from having no config at all, which raises."""
        from codify.pipeline.enrich.enacting import split_opening_material

        text = "Some Act 2020\nHaving regard to nothing,\n"
        preface, _preamble = split_opening_material(text, "xa")
        assert preface is not None and "Some Act 2020" in preface

    def test_an_absent_config_raises_rather_than_passing_text_through(self):
        from codify.pipeline.enrich.enacting import split_opening_material

        with pytest.raises(JurisdictionConfigError, match="zz"):
            split_opening_material("Some Act 2020\n", "zz")


class TestAuthorityGate:
    """A Council of Ministers instrument must not receive a
    legislative-council formula just because its date falls in that era."""

    def _com_regulation(self) -> str:
        return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preface><p>قرار مجلس الوزراء رقم (41) لسنة 2004 باللائحة التنفيذية</p></preface>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''

    def test_com_instrument_skips_assembly_formula(self):
        out = emit_enacting_formula(self._com_regulation(), "xz", "act", date(2004, 8, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_assembly_era_act_without_instrument_phrase_still_injected(self):
        plain = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preface><p>قانون رقم (3) لسنة 2000 بشأن الآبار</p></preface>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''
        out = emit_enacting_formula(plain, "xz", "act", date(2000, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is not None

    def test_recital_citing_com_does_not_block(self):
        # "بناء على تنسيب مجلس الوزراء" is a citation, not an enacting verb.
        recital = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preamble><p>بناء على تنسيب مجلس الوزراء بتاريخ 2000/5/1،</p></preamble>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''
        out = emit_enacting_formula(recital, "xz", "act", date(2000, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is not None

    def test_recital_citing_com_decision_number_does_not_block(self):
        # "وعلى قرار مجلس الوزراء رقم (5) لسنة 2003" in a recital cites a
        # prior instrument; only a preface instrument title blocks.
        recital = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preface><p>قانون رقم (7) لسنة 2005</p></preface>
    <preamble><p>وعلى قرار مجلس الوزراء رقم (5) لسنة 2003،</p></preamble>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''
        out = emit_enacting_formula(recital, "xz", "act", date(2005, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is not None

    def test_formula_naming_matching_authority_passes_gate(self):
        from codify.pipeline.enrich.enacting import _authority_conflict

        com_doc = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act><meta/><preface><p>قرار مجلس الوزراء رقم (41) لسنة 2004</p></preface><body/></act>
</akomaNtoso>
'''
        root = etree.fromstring(com_doc.encode("utf-8"))
        assert not _authority_conflict(root, "قرر مجلس الوزراء ما يلي")
        assert _authority_conflict(root, "أصدر مجلس الأمة الزرزوري القانون الآتي نصه:")

    def _municipal_bylaw(self, opener: str) -> str:
        # A city-council نظام ratified by the Minister of Local Governance.
        return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preamble><p>{opener} بعد موافقة وزير الحكم المحلي:</p></preamble>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''

    def test_municipal_bylaw_construct_form_skips_assembly_formula(self):
        # "أصدر مجلس بلدية الواحة النظام التالي", an assembly-era date must not pull
        # the legislative-council formula onto a municipal by-law.
        src = self._municipal_bylaw("أصدر مجلس بلدية الواحة النظام التالي")
        out = emit_enacting_formula(src, "xz", "act", date(2005, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_municipal_bylaw_adjective_form_skips_assembly_formula(self):
        # "أصدر المجلس البلدي لمدينة زرزورة النظام التالي".
        src = self._municipal_bylaw("أصدر المجلس البلدي لمدينة زرزورة النظام التالي")
        out = emit_enacting_formula(src, "xz", "act", date(1999, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is None

    def test_recital_citing_municipal_decision_does_not_block(self):
        # A national statute whose recitals cite a prior municipal decision by
        # its NOUN form ("قرار مجلس بلدية") must still receive its formula, only
        # the enacting-VERB form ("قرر/أصدر مجلس بلدية") pins municipal authority.
        recital = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act contains="originalVersion">
    <meta><identification source="#codify"/></meta>
    <preface><p>قانون رقم (5) لسنة 2001 بشأن الحكم المحلي</p></preface>
    <preamble><p>وعلى قرار مجلس بلدية الواحة رقم (12) لسنة 2000،</p></preamble>
    <body><section eId="sec_1"><num>1.</num><content><p>نص.</p></content></section></body>
  </act>
</akomaNtoso>
'''
        out = emit_enacting_formula(recital, "xz", "act", date(2001, 6, 1))
        root = etree.fromstring(out.encode("utf-8"))
        assert root.find(".//akn:formula[@name='enactingFormula']", NS) is not None

    def test_the_bidi_mangled_form_of_the_title_is_not_furniture(self):
        # Arabic OCR can emit the law's own title with its tokens in visual order.
        # That form matches a pattern kept for mid-paragraph scrubbing, and
        # promoting it to whole-line disposal deleted the statute's title.
        from codify.pipeline.enrich.enacting import split_opening_material

        mangled = "7 بشأن المراعي 1999 ) لسنة 7 ) قانون رقم"
        preface, _preamble = split_opening_material(
            f"{mangled}\nبعد الاطلاع على القانون الأساسي،\n", "xz"
        )
        assert preface is not None and mangled in preface


def test_opening_material_stops_at_the_enacting_formula() -> None:
    """Synthetic page furniture after a formula must not enter the preamble."""
    from codify.pipeline.enrich.enacting import split_opening_material

    text = "\n".join(
        [
            "قانون المرصد رقم ٤١ لسنة ٢٠٤٢",
            "بعد الإطلاع على نظام الأدوات التجريبية لسنة ٢٠٤٠،",
            "وبعد إقرار مجلس الأمة،",
            "أصدرنا القانون التالي:",
            "-٥-",
            "12062042",
            "00417/000",
            "يونيه ٢٠٤٢",
        ]
    )
    preface, preamble = split_opening_material(text, "xz")
    assert preamble is not None
    assert preamble.strip().endswith("أصدرنا القانون التالي:")
    assert "بعد الإطلاع" in preamble
    for furniture in ("-٥-", "12062042", "00417/000", "يونيه"):
        assert furniture not in preamble
        assert furniture not in (preface or "")


def test_opening_material_without_a_formula_is_not_truncated() -> None:
    """Without a formula marker, preserve the supplied opening text."""
    from codify.pipeline.enrich.enacting import split_opening_material

    text = "نظام\nبعد الإطلاع على شيء،\nسطر أخير"
    _, preamble = split_opening_material(text, "xz")
    assert preamble is not None
    assert preamble.strip().endswith("سطر أخير")


def test_truncation_declines_when_the_tail_carries_provisions() -> None:
    """If the first article anchor was missed, real articles sit in the tail.
    The guard reads source OCR, so it must know `مادة (١)`, not `ARTICLE 1`."""
    from codify.pipeline.enrich.enacting import split_opening_material

    text = "\n".join(
        [
            "قانون رقم ٣ لسنة ٢٠٠٠",
            "بعد الإطلاع على القانون الأساسي،",
            "أصدرنا القانون التالي:",
            "مادة (١)",
            "يعمل بأحكام هذا القانون اعتباراً من تاريخ نشره.",
        ]
    )
    _, preamble = split_opening_material(text, "xz")
    assert preamble is not None
    assert "يعمل بأحكام هذا القانون" in preamble


def test_truncation_cuts_at_the_first_formula_not_a_repeated_one() -> None:
    """A bled-in page repeats the formula; cutting at the later one keeps the
    furniture between the two."""
    from codify.pipeline.enrich.enacting import split_opening_material

    text = "\n".join(
        [
            "قانون المرصد رقم ٤١ لسنة ٢٠٤٢",
            "بعد الإطلاع على القانون الأساسي،",
            "أصدرنا القانون التالي:",
            "-٥-",
            "00417/000",
            "قانون المرصد رقم ٤١ لسنة ٢٠٤٢",
            "أصدرنا القانون التالي:",
        ]
    )
    _, preamble = split_opening_material(text, "xz")
    assert preamble is not None
    assert preamble.strip().endswith("أصدرنا القانون التالي:")
    assert "-٥-" not in preamble
    assert "00417/000" not in preamble


def test_truncation_declines_when_recitals_follow_the_formula() -> None:
    """A preamble quoting a formula early must keep the recitals that follow."""
    from codify.pipeline.enrich.enacting import split_opening_material

    text = "\n".join(
        [
            "قانون رقم ٣ لسنة ٢٠٠٠",
            "أصدرنا القانون التالي:",
            "وبناء على ما تقدم من مشروع القانون،",
        ]
    )
    _, preamble = split_opening_material(text, "xz")
    assert preamble is not None
    assert "وبناء على ما تقدم" in preamble


def test_document_with_no_preamble_marker_is_all_preface() -> None:
    """No opener and no formula: nothing cut, nothing becomes a preamble."""
    from codify.pipeline.enrich.enacting import split_opening_material

    preface, preamble = split_opening_material("ARBITRATION REGULATION\nSome title line", "xz")
    assert preamble is None
    assert preface == "ARBITRATION REGULATION\nSome title line"
