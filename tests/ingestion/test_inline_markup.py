"""Tests for Phase 3d inline semantic markup."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.jurisdictions import JurisdictionConfigError
from codify.pipeline.enrich.inline_markup import emit_inline_markup


def _act(body_inner: str, has_refs: bool = False) -> str:
    refs = '<references source="#codify"/>' if has_refs else ""
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/>{refs}</meta>
    <body>{body_inner}</body>
  </act>
</akomaNtoso>
'''


# --- Pass 1: internal refs ---------------------------------------------------


class TestInternalRefs:
    async def test_section_ref_wrapped(self):
        xml = _act("""
        <section eId="sec_1">
          <content><p>Subject to section 5(2), the duty applies.</p></content>
        </section>
        <section eId="sec_5">
          <subsection eId="sec_5__subsec_2"><content><p>x</p></content></subsection>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        refs = root.findall(".//akn:ref", NS)
        assert len(refs) >= 1
        assert refs[0].get("href") == "#sec_5__subsec_2"
        assert "section 5(2)" in (refs[0].text or "")

    async def test_alias_s_dot(self):
        xml = _act("""
        <section eId="sec_1">
          <content><p>See s. 12 for details.</p></content>
        </section>
        <section eId="sec_12"><content><p>x</p></content></section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        assert 'href="#sec_12"' in out

    async def test_multiple_refs(self):
        xml = _act("""
        <section eId="sec_1">
          <content><p>Under section 3 and section 7(1)(a), the court may act.</p></content>
        </section>
        <section eId="sec_3"><content><p>x</p></content></section>
        <section eId="sec_7"><content><p>y</p></content></section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        refs = root.findall(".//akn:ref", NS)
        assert len(refs) >= 2

    async def test_two_refs_same_text_node_document_order(self):
        """Regression: two refs in the same .text must appear in document order."""
        xml = _act("""
        <section eId="sec_1">
          <content><p>See section 3 and section 7 for details.</p></content>
        </section>
        <section eId="sec_3"><content><p>x</p></content></section>
        <section eId="sec_7"><content><p>y</p></content></section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        p = root.find(".//akn:p", NS)
        refs = p.findall("akn:ref", NS)
        assert len(refs) == 2
        assert refs[0].get("href") == "#sec_3"
        assert refs[1].get("href") == "#sec_7"

    async def test_idempotent(self):
        xml = _act("""
        <section eId="sec_1">
          <content><p>See section 5.</p></content>
        </section>
        <section eId="sec_5"><content><p>x</p></content></section>""")
        once = await emit_inline_markup(xml, "gb", "act")
        twice = await emit_inline_markup(once, "gb", "act")
        root = etree.fromstring(twice.encode())
        assert len(root.findall(".//akn:ref", NS)) == 1

    async def test_unresolved_ref_left_as_text(self):
        """A reference whose target isn't in the document must NOT become a
        dangling <ref>, leave it as plain text."""
        xml = _act("""
        <section eId="sec_1">
          <content><p>See section 999 which does not exist.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        assert root.findall(".//akn:ref", NS) == []

    async def test_an_unknown_country_raises(self):
        """It used to return the document unmarked, which reads as prose with no
        internal references rather than a pass that never ran."""
        xml = _act('<section eId="sec_1"><content><p>section 5</p></content></section>')
        with pytest.raises(JurisdictionConfigError, match="zz"):
            await emit_inline_markup(xml, "zz", "act")


# --- Pass 2: definitions -----------------------------------------------------


class TestDefinitions:
    async def test_quoted_term_extracted(self):
        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content>
            <p>"company" means a body corporate.</p>
            <p>"officer" means a director or secretary.</p>
          </content>
        </section>
        <section eId="sec_3">
          <content><p>The company shall register.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())

        defs = root.findall(".//akn:def", NS)
        assert len(defs) >= 2

        # Check refersTo attribute
        refers = {d.get("refersTo") for d in defs}
        assert "#term-company" in refers
        assert "#term-officer" in refers

    async def test_tlcterm_injected(self):
        xml = _act(
            """
        <section eId="sec_2">
          <heading>Definitions</heading>
          <content><p>"fund" means assets of every kind.</p></content>
        </section>""",
            has_refs=True,
        )
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())

        tlc = root.findall(".//akn:TLCTerm", NS)
        assert len(tlc) >= 1
        assert tlc[0].get("eId") == "term-fund"
        assert tlc[0].get("showAs") == "fund"

    async def test_curly_quotes(self):
        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content><p>\u201cvessel\u201d means any ship or boat.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        assert "term-vessel" in out


# --- Pass 3: term occurrences ------------------------------------------------


class TestTermOccurrences:
    async def test_term_marked_in_body(self):
        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content><p>"company" means a body corporate.</p></content>
        </section>
        <section eId="sec_5">
          <content><p>Every company shall file annual returns.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())

        terms = root.findall(".//akn:term", NS)
        assert len(terms) >= 1
        assert terms[0].get("refersTo") == "#term-company"

    async def test_term_not_marked_in_definition_section(self):
        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content>
            <p>"company" means a body corporate.</p>
            <p>In this Act, company has the meaning given above.</p>
          </content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        # <term> should NOT appear in the definition section
        terms = root.findall(".//akn:term", NS)
        assert len(terms) == 0


# --- Pass 4: external refs (mocked LLM) --------------------------------------


class TestExternalRefs:
    async def test_external_ref_with_mock(self):
        mock_client = AsyncMock()
        mock_client.chat_json = AsyncMock(
            return_value={
                "refs": [
                    {
                        "text": "the Companies Act",
                        "title": "Companies Act",
                        "year": "1982",
                        "number": "21",
                    }
                ]
            }
        )

        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content><p>"company" means a body corporate.</p></content>
        </section>
        <section eId="sec_5">
          <content><p>Subject to the Companies Act, the duty applies.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act", client=mock_client)
        root = etree.fromstring(out.encode())

        refs = root.findall(".//akn:ref", NS)
        ext = [r for r in refs if not r.get("href", "").startswith("#")]
        assert len(ext) >= 1
        assert "/akn/gb/act/ukpga/1982/21" in ext[0].get("href", "")

    async def test_a_named_series_keeps_its_token(self):
        """A series of the wrong family falls back to the default."""
        mock_client = AsyncMock()
        mock_client.chat_json = AsyncMock(
            return_value={
                "refs": [
                    {"text": "2020 asp 8", "year": "2020", "number": "8", "series": "asp"},
                    {
                        "text": "S.S.I. 2020/1",
                        "year": "2020",
                        "number": "1",
                        "kind": "secondary",
                        "series": "ssi",
                    },
                    {"text": "the Act of 2019", "year": "2019", "number": "3", "series": "uksi"},
                ]
            }
        )
        xml = _act("""
        <section eId="sec_1">
          <content><p>See 2020 asp 8, S.S.I. 2020/1 and the Act of 2019.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act", client=mock_client)
        root = etree.fromstring(out.encode())
        hrefs = [r.get("href", "") for r in root.findall(".//akn:ref", NS)]
        assert "/akn/gb/act/asp/2020/8" in hrefs
        assert "/akn/gb/act/ssi/2020/1" in hrefs
        assert "/akn/gb/act/ukpga/2019/3" in hrefs

    async def test_a_literal_si_citation_takes_the_default_secondary_token(self):
        xml = _act("""
        <section eId="sec_1">
          <content><p>As amended by S.I. 2004/3391 and S.I. 1976/1213 (N.I. 16).</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())
        hrefs = [r.get("href", "") for r in root.findall(".//akn:ref", NS)]
        assert "/akn/gb/act/uksi/2004/3391" in hrefs
        assert "/akn/gb/act/nisi/1976/1213" in hrefs

    async def test_a_series_names_the_country_that_publishes_it(self):
        """A UK series cited elsewhere targets the UK work."""
        from codify.pipeline.enrich.inline_markup import _extract_external_refs_llm

        mock_client = AsyncMock()
        mock_client.chat_json = AsyncMock(
            return_value={
                "refs": [
                    {"text": "2020 asp 8", "year": "2020", "number": "8", "series": "asp"},
                    {"text": "Law 5 of 2019", "year": "2019", "number": "5"},
                ]
            }
        )
        refs = await _extract_external_refs_llm("irrelevant", "xq", mock_client)
        assert [r["frbr_uri"] for r in refs] == ["/akn/gb/act/asp/2020/8", "/akn/xq/act/2019/5"]

    async def test_no_client_skips_external(self):
        xml = _act("""
        <section eId="sec_5">
          <content><p>The Companies Act applies.</p></content>
        </section>""")
        out = await emit_inline_markup(xml, "gb", "act", client=None)
        root = etree.fromstring(out.encode())
        refs = root.findall(".//akn:ref", NS)
        ext = [r for r in refs if not r.get("href", "").startswith("#")]
        assert len(ext) == 0

    async def test_llm_failure_graceful(self):
        mock_client = AsyncMock()
        mock_client.chat_json = AsyncMock(side_effect=ValueError("model error"))

        xml = _act("""
        <section eId="sec_5">
          <content><p>Under the Insurance Act, coverage is mandatory.</p></content>
        </section>""")
        # Should not raise, graceful fallback
        out = await emit_inline_markup(xml, "gb", "act", client=mock_client)
        assert out  # returns something, even if no external refs


# --- Integration: mixed content ----------------------------------------------


class TestMixedContent:
    async def test_refs_and_terms_coexist(self):
        xml = _act("""
        <section eId="sec_2">
          <heading>Interpretation</heading>
          <content><p>"registrar" means the Registrar of Companies.</p></content>
        </section>
        <section eId="sec_5">
          <content><p>The registrar shall, under section 3, maintain a register.</p></content>
        </section>
        <section eId="sec_3"><content><p>x</p></content></section>""")
        out = await emit_inline_markup(xml, "gb", "act")
        root = etree.fromstring(out.encode())

        refs = root.findall(".//akn:ref", NS)
        terms = root.findall(".//akn:term", NS)
        defs = root.findall(".//akn:def", NS)
        assert len(refs) >= 1
        assert len(defs) >= 1
        assert len(terms) >= 1


# --- Pass 4a: deterministic Arabic external refs -----------------------------


class TestArabicExternalRefs:
    """Arabic legal citations follow `قانون TITLE لسنة YYYY` (Law TITLE of year
    YYYY). The deterministic detector runs without an LLM and only when the
    body carries Arabic content, Latin-only docs skip it cleanly."""

    async def test_arabic_citation_with_year_only(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>المرجع: قانون الشركات لسنة 1929 ينطبق هنا.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        refs = root.findall(".//akn:ref", NS)
        ext = [r for r in refs if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        assert "1929" in ext[0].get("href", "")
        assert "xz/act" in ext[0].get("href", "")

    async def test_arabic_citation_with_number_and_year(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>راجع قانون رقم 5 لسنة 1995 للتفاصيل.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        # Number 5 + year 1995 land in the FRBR URI
        assert "1995" in ext[0].get("href", "")
        assert "/5" in ext[0].get("href", "")

    async def test_arabic_indic_year_normalised(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>قانون المكاييل لسنة ١٩٢٧ كان ساريا.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        # Year should be ASCII in the FRBR URI even though the source used ١٩٢٧
        assert "1927" in ext[0].get("href", "")

    async def test_protectorate_era_marsoum(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>صدر مرسوم الموانئ لسنة 1932.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        assert "1932" in ext[0].get("href", "")

    async def test_internal_madda_ref_not_caught_as_external(self):
        """`المادة 5` is an internal article reference handled by Pass 1; the
        external Arabic detector must not mistakenly wrap it."""
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>كما هو موضح في المادة 5.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 0

    async def test_latin_only_doc_skips_arabic_detector(self):
        """A UA / EN document with no Arabic letters in the body must not
        trigger the Arabic pass, content-based detection guards against
        wasted regex work on Cyrillic / Latin docs."""
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>This Act amends the Companies Act of 1929.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "ua", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 0  # Arabic regex doesn't match Latin

    async def test_arabic_title_slug_fallback(self):
        """When no canonical number is present, the FRBR target encodes a slug
        of the kind+title so distinct citations don't collapse to the same URI."""
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>راجع قانون الشركات لسنة 1929 وأيضا قانون السجون لسنة 1921.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "xz", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 2
        hrefs = {r.get("href") for r in ext}
        # Slugged URIs preserve the cited act's identity
        assert any("الشركات" in h for h in hrefs)
        assert any("السجون" in h for h in hrefs)


class TestCyrillicExternalRefs:
    """Ukrainian / Russian / Belarusian law citations canonically take the
    form `№ NNNN-IX від DD.MM.YYYY` (number + session marker, then date). The
    deterministic detector recognises both orderings and builds a date-anchored
    FRBR URI matching the form already in the corpus."""

    async def test_cyrillic_canonical_forward(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Зміни внесено Законом № 263-IX від 31.10.2019.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "ua", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        assert ext[0].get("href") == "/akn/ua/act/2019/263-IX"

    async def test_cyrillic_canonical_reverse(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Закон від 18.02.2016 № 1021-VIII визначає процедуру.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "ua", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        assert len(ext) == 1
        assert ext[0].get("href") == "/akn/ua/act/2016/1021-VIII"

    async def test_latin_only_doc_skips_cyrillic_detector(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Amended by Act No 263 of 31.10.2019.</p>"
            "</content></article>"
        )
        out = await emit_inline_markup(xml, "ua", "act", client=None)
        root = etree.fromstring(out.encode())
        ext = [r for r in root.findall(".//akn:ref", NS) if r.get("href", "").startswith("/akn/")]
        # The Cyrillic regex requires "від" / "№" which are Cyrillic, never matches Latin
        assert len(ext) == 0


class TestNumberedInstrumentRefs:
    """Citations by a yearless series number, declared in the jurisdiction's
    `numbering.series_citations`. The pass builds the FRBR work URI from the
    number alone (no year in the citation → unknown-year slot)."""

    async def test_a_list_of_instruments_is_refused_not_half_captured(self) -> None:
        """One number group captures one number, so a list would wrap the first
        and drop the rest without saying so.

        Refused rather than split. OCR-damaged lists ("Assembly Act 580,1577
        and5") decay to numbers that name other laws, so splitting invents
        targets a resolver would then point at confidently. Losing a list is
        visible; a wrong edge is not.
        """
        for text in (
            "Repealing Assembly Acts Nos. 312 and 313 hereby.",
            "Repealing Regency Decrees Nos. 1204, 1377 and 1590 hereby.",
            "Repealing Assembly Act 580,1577 and5 hereby.",
        ):
            assert await self._refs(f"<p>{text}</p>") == []

    async def test_a_plural_noun_with_one_number_still_resolves(self) -> None:
        """The refusal is of lists, not of the plural spelling: drafters write
        "Acts Nos." for a single instrument and that citation is unambiguous."""
        refs = await self._refs("<p>Amending Assembly Acts Nos. 312 hereby.</p>")
        assert refs == ["/akn/xa/act/0001/312"]

    async def _refs(self, body_inner: str, country: str = "xa") -> list[str]:
        out = await emit_inline_markup(_act(body_inner), country, "act", client=None)
        root = etree.fromstring(out.encode())
        return [
            r.get("href", "")
            for r in root.findall(".//akn:ref", NS)
            if r.get("href", "").startswith("/akn/")
        ]

    async def test_full_series_name(self):
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>This Act amends Assembly Act No. 7741.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/0001/7741"]

    async def test_series_abbreviation(self):
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>See AA 312 for the general rule.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/0001/312"]

    async def test_second_series_takes_its_own_doctype(self):
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>Regency Decree No. 518 is hereby repealed.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/rd/0001/518"]

    async def test_declared_connector(self):
        """A connector beyond the defaults ("Ser.") is read from the config."""
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>Harbour Ordinance Ser. 57 remains in force.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/ho/0001/57"]

    async def test_another_full_name_series(self):
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>Mint Statute No. 233 governs the coinage.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/ms/0001/233"]

    async def test_full_name_not_double_wrapped_by_abbreviation(self):
        """The full-name citation wraps first; the abbreviation pass must not
        mint a second ref from the same span."""
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>Assembly Act No. 7741 applies.</p></content></section>"
        )
        assert hrefs == ["/akn/xa/act/0001/7741"]

    async def test_a_jurisdiction_without_series_skips_pass(self):
        hrefs = await self._refs(
            '<section eId="sec_1"><num>1</num><content>'
            "<p>Assembly Act No. 7741 applies.</p></content></section>",
            country="gb",
        )
        assert hrefs == []

    async def test_repeal_citation_classed_as_amendment(self):
        """A numbered citation in a repeal paragraph carries an amendment class,
        which is what the AKN parser turns into an amendment_effects row."""
        out = await emit_inline_markup(
            _act(
                '<section eId="sec_1"><num>1</num><content>'
                "<p>Assembly Act No. 6093 is hereby repealed.</p></content></section>"
            ),
            "xa",
            "act",
            client=None,
        )
        root = etree.fromstring(out.encode())
        ref = root.find(".//akn:ref", NS)
        assert ref is not None
        assert ref.get("href") == "/akn/xa/act/0001/6093"
        # Composed: this pass minted the ref, so it carries the provenance
        # marker as well as the operation. Replacing the class erased the
        # first and the reference read as publisher-authored.
        assert (ref.get("class") or "").split() == ["derived", "amendment-delete"]


def test_digit_token_guards_llm_ref_fields() -> None:
    """JSON null and literal "None" year-number fields must read absent,
    never mint a /None/ FRBR segment."""
    from codify.pipeline.enrich.inline_markup import _digit_token

    assert _digit_token(None) == ""
    assert _digit_token("None") == ""
    assert _digit_token("null") == ""
    assert _digit_token("") == ""
    assert _digit_token("1998") == "1998"
    assert _digit_token("٤٦") == "46"
    assert _digit_token(2016) == "2016"


def test_number_token_keeps_compound_law_numbers() -> None:
    """Compound numbers ("4/2016", "19-06") are real citations; only None-ish,
    digitless and over-long compounds read absent. The guard now lives beside
    the builder it protects, shared with the document's own identity."""
    from codify.frbr import law_number_token

    assert law_number_token("4/2016") == "4/2016"
    assert law_number_token("19-06") == "19-06"
    assert law_number_token("٤٦") == "46"
    assert law_number_token(None) == ""
    # A dash-joined sequence is a real citation: a decision number can carry
    # its office series as `رقم (3 279 11 م.و إ.ه)`, and migration 0108 stores it.
    assert law_number_token("3-279-11") == "3-279-11"
    # Mixing the separators is a date and a filing reference run together, not a
    # number, so it stays rejected.
    assert law_number_token("2008-04-13/11/11") == ""
    assert law_number_token("1/2/2016") == ""
    assert law_number_token("None") == ""
    assert law_number_token("null") == ""
    assert law_number_token("EC") == ""
    assert law_number_token("article 7") == ""
    assert law_number_token("None/7") == ""
    assert law_number_token("7 extra") == ""


class TestSeriesCitations:
    def test_a_uk_series_cited_elsewhere_targets_the_uk_work(self) -> None:
        """The matcher runs without a title index and sends a recognised
        series to the country that publishes it."""
        from codify.pipeline.enrich import inline_markup as im

        body = etree.fromstring(
            f'<body xmlns="{AKN_NS}"><p>See S.S.I. 2020/1 and S.I. 1976/1213 (N.I. 16).</p></body>'
        )
        assert im._mark_title_refs(body, "xq") == 2
        hrefs = [r.get("href") for r in body.iter(f"{{{AKN_NS}}}ref")]
        assert hrefs == ["/akn/gb/act/ssi/2020/1", "/akn/gb/act/nisi/1976/1213"]
