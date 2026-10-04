"""The legacy OJ HTML rendering segments deterministically into AKN4EU."""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from codify.pipeline.enrich.validator import validate_akn
from codify.pipeline.formats import dispatch
from codify.pipeline.formats.eu_html import EurlexHtmlError, html_to_akn4eu, is_eurlex_html

NS = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}

# The shape Cellar serves for a pre-FORMEX act: a synthetic directive in it.
_HTML = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"><html lang="EN">
<head>
<meta name="DC.title" content="EUR-Lex - 31999L0999 - EN">
<meta name="DC.source" content="Official Journal 123 , 05/06/1999 P. 0010 - 0012; ">
<meta name="DC.type" http-equiv="Content-Type" content="text/html; charset=UNICODE-1-1-UTF-8">
</head>
<body>
<div id="banner"><p>Avis juridique important</p></div>
<h1>31999L0999</h1>
<p>Council Directive 99/999/EEC of 1 April 1999 on the registration of lantern keepers</p>
<div id="TexteOnly">
<p>COUNCIL DIRECTIVE of 1 April 1999 on the registration of lantern keepers (99/999/EEC)</p>
<p>COUNCIL DIRECTIVE of 1 April 1999 on the registration of lantern keepers (99/999/EEC)</p>
<p>THE COUNCIL OF THE UNION,</p>
<p>Having regard to the Treaty, and in particular Article 100 thereof;</p>
<p>Having regard to the proposal from the Commission;</p>
<p>Whereas lantern keepers should be registered;</p>
<p>Whereas a register needs a keeper in each Member State;</p>
<p>HAS ADOPTED THIS DIRECTIVE:</p>
<p></p>
<p>Article 1</p>
<p>Member States shall keep a register of lantern keepers, as provided in Article 3.</p>
<p>Article 2</p>
<p>This Directive shall apply: 1. to keepers of harbour lanterns;</p>
<p>2. to keepers of river lanterns.</p>
<p>Article 3</p>
<p>The register shall record, for each keeper:</p>
<p>(a) the name;</p>
<p>(b) the lantern, by its arrêté number.</p>
<p>Article 4</p>
<p>This Directive is addressed to the Member States.</p>
<p>Done at Brussels, 1 April 1999.</p>
<p>For the Council</p>
<p>The President</p>
<p>ANNEX I</p>
<p>Form of the register.</p>
</div>
</body></html>"""


def _convert(html: str = _HTML) -> tuple[str, dict[str, object], etree._Element]:
    provenance: dict[str, object] = {}
    xml = html_to_akn4eu(html, frbr_work_uri="/akn/eu/act/dir/1999/999", provenance=provenance)
    return xml, provenance, etree.fromstring(xml.encode())


def test_sniff_recognises_the_oj_rendering_and_not_xml() -> None:
    assert is_eurlex_html(_HTML)
    assert is_eurlex_html("  <html><body/></html>")
    assert not is_eurlex_html("<?xml version='1.0'?><FORMEX/>")


def test_articles_are_the_blocks_headed_article_n_and_nothing_else() -> None:
    """A cross-reference inside a paragraph ("as provided in Article 3") is
    prose, not a heading: four articles, in order, from the four headings."""
    _, provenance, root = _convert()
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_3",
        "art_4",
    ]
    assert provenance["html_source"] == {
        "articles": 4,
        "duplicate_articles": 0,
        "duplicate_paragraphs": 0,
        "tables": 0,
        "images": 0,
        "recitals": 2,
        "annexes": 1,
    }
    assert "body_discarded_text" not in provenance
    art_1 = root.find(".//a:article[@eId='art_1']//a:p", NS)
    assert art_1 is not None and "as provided in Article 3" in (art_1.text or "")


def test_numbered_blocks_become_numbered_paragraphs_and_prose_shares_one() -> None:
    _, _, root = _convert()
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    art_3 = root.find(".//a:article[@eId='art_3']", NS)
    assert art_2 is not None and art_3 is not None
    # Article 2's first block runs the chapeau and item 1 together: the chapeau is
    # the intro and item 1 a numbered paragraph. Article 3's lettered points carry
    # no `N.` numbering, so they are points of one shared paragraph.
    assert art_2.find("a:intro/a:p", NS).text == "This Directive shall apply:"
    assert [p.get("eId") for p in art_2.iterfind("a:paragraph", NS)] == [
        "art_2__para_1",
        "art_2__para_2",
    ]
    assert art_2.find("a:paragraph/a:content/a:p", NS).text == "to keepers of harbour lanterns;"
    assert [p.get("eId") for p in art_3.iterfind("a:paragraph", NS)] == ["art_3__para_1"]
    assert len(art_3.findall(".//a:p", NS)) == 3


def test_preamble_splits_citations_recitals_and_the_enacting_formula() -> None:
    _, _, root = _convert()
    assert [c.get("eId") for c in root.iterfind(".//a:citation", NS)] == [
        "citations__cit_1",
        "citations__cit_2",
    ]
    assert [r.get("eId") for r in root.iterfind(".//a:recital", NS)] == [
        "recitals__rec_1",
        "recitals__rec_2",
    ]
    formula = root.find(".//a:formula[@name='enactingFormula']/a:p", NS)
    assert formula is not None and formula.text == "HAS ADOPTED THIS DIRECTIVE:"
    # The institution line is neither a citation nor a recital.
    other = root.find(".//a:preamble/a:p", NS)
    assert other is not None and other.text == "THE COUNCIL OF THE UNION,"


def test_the_title_is_the_preface_and_its_repeat_is_dropped() -> None:
    _, _, root = _convert()
    title = root.find(".//a:preface/a:longTitle/a:p", NS)
    assert title is not None and title.text.startswith("COUNCIL DIRECTIVE of 1 April 1999")
    assert len(root.findall(".//a:preface//a:p", NS)) == 1


def test_signature_and_annex_leave_the_body() -> None:
    _, _, root = _convert()
    conclusions = [p.text for p in root.iterfind(".//a:conclusions/a:p", NS)]
    assert conclusions == ["Done at Brussels, 1 April 1999.", "For the Council", "The President"]
    annex = root.find(".//a:attachments/a:attachment[@eId='att_1']", NS)
    assert annex is not None
    assert annex.findtext("a:heading", namespaces=NS) == "ANNEX I"
    assert annex.findtext(".//a:mainBody/a:p", namespaces=NS) == "Form of the register."
    assert root.find(".//a:body//a:p[.='Form of the register.']", NS) is None


def test_dates_and_publication_come_from_the_title_and_the_oj_reference() -> None:
    _, _, root = _convert()
    assert root.find(".//a:FRBRExpression/a:FRBRdate", NS).get("date") == "1999-04-01"
    assert root.find(".//a:FRBRExpression/a:FRBRuri", NS).get("value") == (
        "/akn/eu/act/dir/1999/999/eng@1999-04-01"
    )
    publication = root.find(".//a:publication", NS)
    assert publication is not None
    assert (publication.get("date"), publication.get("number")) == ("1999-06-05", "123")
    assert root.find(".//a:act", NS).get("name") == "directive"


def test_utf8_text_survives_the_charset_the_page_declares() -> None:
    _, _, root = _convert()
    texts = [p.text or "" for p in root.iterfind(".//a:article[@eId='art_3']//a:p", NS)]
    assert any("arrêté" in t for t in texts)


def test_the_converted_act_validates_clean() -> None:
    xml, _, _ = _convert()
    assert validate_akn(xml) == []


def test_a_page_without_article_headings_is_refused() -> None:
    page = _HTML.replace("<p>Article 1</p>", "").replace("<p>Article 2</p>", "")
    page = page.replace("<p>Article 3</p>", "").replace("<p>Article 4</p>", "")
    with pytest.raises(EurlexHtmlError, match="Article N"):
        html_to_akn4eu(page, frbr_work_uri="/akn/eu/act/dir/1999/999")


async def test_an_eu_html_file_takes_the_directive_lane_not_the_llm_lane(tmp_path: Path) -> None:
    """Without a model the PDF lane refuses; an EU html file never reaches it."""
    source = tmp_path / "31999L0999.html"
    source.write_text(_HTML, encoding="utf-8")
    events = [
        e
        async for e in dispatch(
            source, "eu", frbr_work_uri="/akn/eu/act/dir/1999/999", language="eng"
        )
    ]
    names = [type(e).__name__ for e in events]
    assert "Complete" in names, names
    assert not any(getattr(e, "stage", "") == "dispatch" for e in events)


def test_sniff_sees_past_a_byte_order_mark() -> None:
    assert is_eurlex_html("\ufeff<!DOCTYPE HTML PUBLIC><html/>")


@pytest.mark.parametrize(
    "formula",
    ["HAVE ADOPTED THIS REGULATION:", "HAS DECIDED AS FOLLOWS:", "HAS ADOPTED THIS DECISION:"],
)
def test_each_enacting_formula_closes_the_preamble(formula: str) -> None:
    page = _HTML.replace("HAS ADOPTED THIS DIRECTIVE:", formula)
    _, _, root = etree_of(page)
    assert root.find(".//a:formula[@name='enactingFormula']/a:p", NS).text == formula
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)][:1] == ["art_1"]


def test_a_lettered_article_number_keeps_its_letter() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>Article 1a</p>")
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_1a",
        "art_3",
        "art_4",
    ]
    assert root.find(".//a:article[@eId='art_1a']/a:num", NS).text == "Article 1a"


def test_an_act_without_a_preamble_starts_at_its_first_article() -> None:
    start = _HTML.index("<p>THE COUNCIL OF THE UNION,</p>")
    end = _HTML.index("<p>Article 1</p>")
    page = _HTML[:start] + _HTML[end:]
    _, provenance, root = etree_of(page)
    assert root.find(".//a:preamble", NS) is None
    assert provenance["html_source"]["articles"] == 4  # type: ignore[index]


def test_prose_between_the_formula_and_the_first_article_stays_in_the_preamble() -> None:
    page = _HTML.replace("<p></p>\n<p>Article 1</p>", "<p>A stray line.</p>\n<p>Article 1</p>")
    _, _, root = etree_of(page)
    assert [p.text for p in root.iterfind(".//a:preamble/a:p", NS)] == [
        "THE COUNCIL OF THE UNION,",
        "A stray line.",
    ]
    assert root.find(".//a:body//a:p[.='A stray line.']", NS) is None


def test_without_an_oj_reference_the_date_falls_to_the_signature_and_publication_is_flagged() -> (
    None
):
    page = _HTML.replace(
        '<meta name="DC.source" content="Official Journal 123 , 05/06/1999 P. 0010 - 0012; ">', ""
    ).replace("of 1 April 1999 on", "on")
    xml, provenance, root = etree_of(page)
    assert root.find(".//a:FRBRExpression/a:FRBRdate", NS).get("date") == "1999-04-01"
    assert root.find(".//a:publication", NS) is None
    assert provenance["publication_missing"] is True


def etree_of(page: str) -> tuple[str, dict[str, object], etree._Element]:
    return _convert(page)


def test_a_titled_heading_is_an_article_and_a_sentence_starting_article_is_prose() -> None:
    page = _HTML.replace("<p>Article 1</p>", "<p>Article 1 Scope</p>").replace(
        "<p>Article 3</p>\n<p>The register shall record, for each keeper:</p>",
        "<p>Article 3 - Register</p>\n"
        "<p>Article 1 shall apply to the register, for each keeper:</p>",
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_3",
        "art_4",
    ]
    assert root.find(".//a:article[@eId='art_1']/a:heading", NS).text == "Scope"
    assert root.find(".//a:article[@eId='art_3']/a:heading", NS).text == "Register"
    first = root.find(".//a:article[@eId='art_3']//a:p", NS)
    assert first is not None and first.text.startswith("Article 1 shall apply")


def test_an_annex_before_the_signature_does_not_swallow_it() -> None:
    page = _HTML.replace(
        "<p>Done at Brussels, 1 April 1999.</p>\n<p>For the Council</p>\n<p>The President</p>\n"
        "<p>ANNEX I</p>\n<p>Form of the register.</p>",
        "<p>ANNEX I</p>\n<p>Form of the register.</p>\n"
        "<p>Done at Brussels, 1 April 1999.</p>\n<p>For the Council</p>\n<p>The President</p>",
    )
    _, _, root = etree_of(page)
    assert [p.text for p in root.iterfind(".//a:conclusions/a:p", NS)] == [
        "Done at Brussels, 1 April 1999.",
        "For the Council",
        "The President",
    ]
    assert [p.text for p in root.iterfind(".//a:attachment//a:mainBody/a:p", NS)] == [
        "Form of the register."
    ]


def test_a_chapeau_then_numbered_blocks_gives_an_intro_and_numbered_paragraphs() -> None:
    page = _HTML.replace(
        "<p>This Directive shall apply: 1. to keepers of harbour lanterns;</p>",
        "<p>This Directive shall apply:</p>\n<p>1. to keepers of harbour lanterns;</p>",
    )
    _, _, root = etree_of(page)
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    assert art_2 is not None
    assert art_2.find("a:intro/a:p", NS).text == "This Directive shall apply:"
    assert [p.get("eId") for p in art_2.iterfind("a:paragraph", NS)] == [
        "art_2__para_1",
        "art_2__para_2",
    ]
    assert art_2.find("a:paragraph[@eId='art_2__para_2']/a:num", NS).text == "2."


def test_with_no_date_in_title_or_signature_the_oj_date_is_the_expression_date() -> None:
    page = _HTML.replace("of 1 April 1999 on", "on").replace(
        "Done at Brussels, 1 April 1999.", "Done at Brussels."
    )
    _, _, root = etree_of(page)
    assert root.find(".//a:FRBRExpression/a:FRBRdate", NS).get("date") == "1999-06-05"


def test_with_no_date_anywhere_the_conversion_refuses() -> None:
    from codify.pipeline.formats.eu_directive import FormexDateMissingError

    page = (
        _HTML.replace("of 1 April 1999 on", "on")
        .replace("Done at Brussels, 1 April 1999.", "Done at Brussels.")
        .replace(
            '<meta name="DC.source" content="Official Journal 123 , 05/06/1999 P. 0010 - 0012; ">',
            "",
        )
    )
    with pytest.raises(FormexDateMissingError):
        html_to_akn4eu(page, frbr_work_uri="/akn/eu/act/dir/1999/999")


def test_an_act_without_signature_or_annex_emits_neither_and_validates() -> None:
    end = _HTML.index("<p>Done at Brussels, 1 April 1999.</p>")
    page = _HTML[:end] + "</div></body></html>"
    xml, provenance, root = etree_of(page)
    assert root.find(".//a:conclusions", NS) is None
    assert root.find(".//a:attachments", NS) is None
    assert provenance["html_source"]["annexes"] == 0  # type: ignore[index]
    assert validate_akn(xml) == []


async def test_an_eu_html_source_without_a_work_uri_fails_legibly(tmp_path: Path) -> None:
    source = tmp_path / "act.html"
    source.write_text(_HTML, encoding="utf-8")
    events = [e async for e in dispatch(source, "eu", language="eng")]
    failed = [e for e in events if type(e).__name__ == "Failed"]
    assert failed and "FRBR work URI" in failed[0].error


def test_a_lowercase_continuation_after_article_n_is_prose_even_without_punctuation() -> None:
    page = _HTML.replace(
        "<p>Member States shall keep a register of lantern keepers, as provided in Article 3.</p>",
        "<p>Article 1 shall apply to keepers</p>",
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_3",
        "art_4",
    ]
    assert (
        root.find(".//a:article[@eId='art_1']//a:p", NS).text == "Article 1 shall apply to keepers"
    )


def test_the_publication_event_carries_the_oj_date() -> None:
    _, _, root = etree_of(_HTML)
    event = root.find(".//a:lifecycle/a:eventRef[@eId='evt_publication']", NS)
    assert event is not None and event.get("date") == "1999-06-05"
    assert event.get("date") == root.find(".//a:publication", NS).get("date")


def test_a_url_with_an_html_suffix_is_not_a_local_html_act() -> None:
    from codify.pipeline.formats import _is_html_path

    assert _is_html_path(Path("act.html")) and _is_html_path("act.HTM")
    assert not _is_html_path("http://169.254.169.254/x.html")
    assert not _is_html_path("https://eur-lex.europa.eu/x.html")


def test_an_en_dash_or_no_break_space_after_article_still_heads_an_article() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>Article \u2013 2</p>").replace(
        "<p>Article 3</p>", "<p>Article\u00a03</p>"
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_3",
        "art_4",
    ]


def test_a_repeated_article_number_keeps_a_unique_eid_and_is_counted() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>Article 1</p>")
    xml, provenance, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_1_dup2",
        "art_3",
        "art_4",
    ]
    assert provenance["html_source"]["duplicate_articles"] == 1  # type: ignore[index]
    # The validator names the repeat too, so the run carries it twice over.
    assert [i["check"] for i in validate_akn(xml)] == ["duplicate_number"]


def test_a_lowercase_sentence_with_a_date_and_no_punctuation_is_prose() -> None:
    page = _HTML.replace(
        "<p>Member States shall keep a register of lantern keepers, as provided in Article 3.</p>",
        "<p>Article 1 shall apply from 1 January 2000</p>",
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_3",
        "art_4",
    ]


async def test_a_non_eu_html_file_does_not_take_the_eu_lane(tmp_path: Path) -> None:
    source = tmp_path / "act.html"
    source.write_text(_HTML, encoding="utf-8")
    events = [e async for e in dispatch(source, "xa")]
    assert [getattr(e, "stage", "") for e in events] == ["dispatch"]


async def test_the_article_count_reaches_the_run_as_a_finding(tmp_path: Path) -> None:
    source = tmp_path / "act.html"
    source.write_text(_HTML, encoding="utf-8")
    issued = [
        e.issue
        async for e in dispatch(source, "eu", frbr_work_uri="/akn/eu/act/dir/1999/999")
        if type(e).__name__ == "ValidationIssued"
    ]
    assert any("html_source" in i and i["html_source"]["articles"] == 4 for i in issued)


def test_the_sniff_accepts_xhtml_behind_an_xml_declaration() -> None:
    assert is_eurlex_html('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html><html/>')
    assert is_eurlex_html('<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"/>')
    assert not is_eurlex_html('<?xml version="1.0"?><FORMEX/>')


def test_lettered_points_nest_under_the_number_before_them() -> None:
    page = _HTML.replace(
        "<p>This Directive shall apply: 1. to keepers of harbour lanterns;</p>\n"
        "<p>2. to keepers of river lanterns.</p>",
        "<p>1. Scope</p>\n<p>(a) the first case;</p>\n<p>(b) the second case.</p>\n"
        "<p>2. Application</p>",
    )
    _, _, root = etree_of(page)
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    assert art_2 is not None and art_2.find("a:intro", NS) is None
    paras = art_2.findall("a:paragraph", NS)
    assert [p.get("eId") for p in paras] == ["art_2__para_1", "art_2__para_2"]
    assert paras[0].find("a:intro/a:p", NS).text == "Scope"
    points = paras[0].findall("a:point", NS)
    assert [p.get("eId") for p in points] == ["art_2__para_1__point_a", "art_2__para_1__point_b"]
    assert [(p.find("a:num", NS).text, p.find("a:content/a:p", NS).text) for p in points] == [
        ("(a)", "the first case;"),
        ("(b)", "the second case."),
    ]
    assert [p.text for p in paras[1].findall("a:content/a:p", NS)] == ["Application"]


def test_the_annex_doc_carries_its_meta_and_the_act_passes_the_strict_schema() -> None:
    from codify.akn._schema import validate_akn as schema_validate

    xml, _, root = etree_of(_HTML)
    meta = root.find(".//a:attachment[@eId='att_1']/a:doc/a:meta", NS)
    assert meta is not None
    this = meta.find(".//a:FRBRWork/a:FRBRthis", NS)
    assert this is not None and this.get("value") == "/akn/eu/act/dir/1999/999/!att_1"
    schema_validate(xml, strict=True)


def test_the_heading_word_is_article_or_capitals_and_never_lowercase() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>ARTICLE 2</p>").replace(
        "<p>Article 3</p>", "<p>article 3</p>"
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_4",
    ]
    # The lowercase line and its blocks stay prose under Article 2.
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    assert any((p.text or "") == "article 3" for p in art_2.iterfind(".//a:p", NS))


def test_a_repeated_paragraph_number_keeps_a_unique_eid_and_is_counted() -> None:
    page = _HTML.replace(
        "<p>This Directive shall apply: 1. to keepers of harbour lanterns;</p>\n"
        "<p>2. to keepers of river lanterns.</p>",
        "<p>1. to keepers of harbour lanterns;</p>\n<p>1. to keepers of river lanterns.</p>",
    )
    _, provenance, root = etree_of(page)
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    assert [p.get("eId") for p in art_2.iterfind("a:paragraph", NS)] == [
        "art_2__para_1",
        "art_2__para_1_dup2",
    ]
    assert provenance["html_source"]["duplicate_paragraphs"] == 1  # type: ignore[index]
    eids = [e.get("eId") for e in root.iter() if e.get("eId")]
    assert len(set(eids)) == len(eids)


def test_an_inserted_article_suffix_may_be_upper_case() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>Article 1A</p>").replace(
        "<p>Article 3</p>", "<p>Article 11B</p>"
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_1a",
        "art_11b",
        "art_4",
    ]


def test_an_annex_table_keeps_its_cells_as_row_blocks_and_is_counted() -> None:
    page = _HTML.replace(
        "<p>Form of the register.</p>",
        "<p>Rates:</p><table><tr><th>Lantern</th><th>Rate</th></tr>"
        "<tr><td>harbour</td><td>5%</td></tr><tr><td>river</td><td>2%</td></tr></table>",
    )
    _, provenance, root = etree_of(page)
    rows = [p.text for p in root.iterfind(".//a:attachment//a:mainBody/a:p", NS)]
    assert rows == ["Rates:", "Lantern | Rate", "harbour | 5%", "river | 2%"]
    assert provenance["html_source"]["tables"] == 1  # type: ignore[index]
    assert "body_discarded_text" not in provenance


def test_text_outside_any_block_element_is_reported_as_discarded() -> None:
    page = _HTML.replace(
        '<div id="TexteOnly">',
        '<div id="TexteOnly">'
        "<span>Sixty characters of notice text that no block element wraps here.</span>",
    )
    _, provenance, _ = etree_of(page)
    assert provenance["body_discarded_text"] >= 40  # type: ignore[operator]


def test_a_two_letter_title_is_a_heading_not_an_article_suffix() -> None:
    page = _HTML.replace("<p>Article 2</p>", "<p>Article 2 To</p>").replace(
        "<p>Article 3</p>", "<p>Article 4A</p>"
    )
    _, _, root = etree_of(page)
    assert [a.get("eId") for a in root.iterfind(".//a:body/a:article", NS)] == [
        "art_1",
        "art_2",
        "art_4a",
        "art_4",
    ]
    assert root.find(".//a:article[@eId='art_2']/a:heading", NS).text == "To"
    assert root.find(".//a:article[@eId='art_4a']/a:num", NS).text == "Article 4A"


def test_a_paragraph_inside_a_list_item_is_one_block_not_two() -> None:
    page = _HTML.replace(
        "<p>Article 1</p>", "<p>Article 1</p><ul><li><p>First point.</p></li></ul>"
    )
    _, _, root = etree_of(page)
    assert len(root.findall(".//a:body//a:p[.='First point.']", NS)) == 1


def test_the_preamble_keeps_source_order_and_the_formula_stays_before_later_prose() -> None:
    page = _HTML.replace(
        "<p>Having regard to the proposal from the Commission;</p>",
        "<p>Having regard to the proposal from the Commission;</p>"
        "<p>After consulting the Assembly,</p>",
    ).replace("<p></p>\n<p>Article 1</p>", "<p>A stray line.</p>\n<p>Article 1</p>")
    _, _, root = etree_of(page)
    preamble = root.find(".//a:preamble", NS)
    assert preamble is not None
    tags = [etree.QName(c).localname for c in preamble]
    assert tags == ["p", "citations", "p", "recitals", "formula", "p"]
    assert [p.text for p in preamble.iterfind("a:p", NS)] == [
        "THE COUNCIL OF THE UNION,",
        "After consulting the Assembly,",
        "A stray line.",
    ]


def test_a_short_unwrapped_span_counts_as_discarded() -> None:
    page = _HTML.replace('<div id="TexteOnly">', '<div id="TexteOnly"><span>Repealed.</span>')
    _, provenance, _ = etree_of(page)
    assert provenance["body_discarded_text"] == len("Repealed.")


def test_a_signal_reaches_the_run_graded_and_lost_text_is_a_warning() -> None:
    from codify.pipeline.formats.eu_directive import signal_finding
    from codify.quality.structural_quality_grade import structural_quality_grade

    lost = signal_finding("body_discarded_text", 9)
    assert lost["body_discarded_text"] == 9 and lost["severity"] == "warning"
    assert signal_finding("html_source", {"articles": 2})["severity"] == "info"
    assert structural_quality_grade([lost]).grade == "warning"
    assert structural_quality_grade([signal_finding("publication_missing", True)]).grade == "clean"


def test_a_line_break_is_a_space_not_a_join() -> None:
    page = _HTML.replace(
        "<p>Member States shall keep a register of lantern keepers, as provided in Article 3.</p>",
        "<p>Effective<br>immediately, as provided in Article 3.</p>",
    )
    _, _, root = etree_of(page)
    art_1 = root.find(".//a:article[@eId='art_1']", NS)
    assert art_1 is not None
    assert "Effective immediately," in "".join(art_1.itertext())


def test_an_image_is_named_in_place_and_graded_a_warning() -> None:
    from codify.pipeline.formats.eu_directive import signal_finding

    page = _HTML.replace(
        "<p>Article 2</p>",
        '<p>Article 2</p><img src="form.gif"><p>Applicants<img src="tick.gif"> sign.</p>',
    )
    _, provenance, root = etree_of(page)
    assert provenance["html_images"] == 2
    assert provenance["html_source"]["images"] == 2  # type: ignore[index]
    texts = [p.text or "" for p in root.iterfind(".//a:article[@eId='art_2']//a:p", NS)]
    assert "[image not transcribed: form.gif]" in texts
    assert any("Applicants [image not transcribed: tick.gif] sign." == t for t in texts)
    assert signal_finding("html_images", 2)["severity"] == "warning"


def test_an_act_nested_inside_an_unclosed_paragraph_still_segments() -> None:
    # Older Cellar pages wrap the act in `<p><TXT_TE>` and never close the `<p>`.
    wrapped = _HTML.replace('<div id="TexteOnly">', '<div id="TexteOnly">\n<p>\n<TXT_TE>', 1)
    assert wrapped != _HTML
    _, flat_provenance, flat = _convert()
    _, nested_provenance, nested = _convert(wrapped)
    assert nested_provenance.get("body_discarded_text") == flat_provenance.get(
        "body_discarded_text"
    )
    numbers = [n.text for n in nested.iterfind(".//a:article/a:num", NS)]
    assert numbers == [n.text for n in flat.iterfind(".//a:article/a:num", NS)]
    assert len(numbers) == 4


# Cellar's CONVEX XHTML for acts with no FORMEX: an encoding declaration first,
# then the eli-container layout. Synthetic text in that shape.
_CONVEX_XHTML = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML//EN"'
    ' "xhtml-strict.dtd"><html xmlns="http://www.w3.org/1999/xhtml">'
    "<!-- CONVEX # converter_version:9.17.0 --><head>\n"
    '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>\n'
    "<title>L_2042001EN.01000101.xml</title>\n</head>\n<body>\n"
    '<table width="100%"><tbody><tr><td><p class="oj-hd-date">2.1.2042</p></td>'
    '<td><p class="oj-hd-oj">L 1/1</p></td></tr></tbody></table>\n'
    '<div class="eli-container">\n'
    '<div class="eli-main-title" id="tit_1">'
    '<p class="oj-doc-ti">COUNCIL REGULATION (EC) No 7/2042</p>'
    '<p class="oj-doc-ti">of 1 January 2042</p>'
    '<p class="oj-doc-ti">on the registration of lantern keepers</p></div>\n'
    '<div class="eli-subdivision" id="pbl_1">'
    '<p class="oj-normal">THE COUNCIL OF THE EUROPEAN UNION,</p>'
    '<div class="eli-subdivision" id="cit_1"><p class="oj-normal">'
    "Having regard to the Treaty,</p></div>"
    '<p class="oj-normal">HAS ADOPTED THIS REGULATION:</p></div>\n'
    '<div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article 1</p>'
    '<div id="001.001"><p class="oj-normal">Each keeper shall hold an arrêté.</p></div></div>\n'
    '<div class="eli-subdivision" id="art_2"><p class="oj-ti-art">Article 2</p>'
    '<p class="oj-normal">The form is set out below.</p>'
    '<p class="oj-normal"><img src="data:image/jpg;base64,/9j/4AAQSkZJRgABAQEAyADIAAD"'
    ' alt="Image 1" class="oj-img"/></p></div>\n'
    '<div class="eli-subdivision" id="fnp_1"><div class="oj-final">'
    '<p class="oj-normal">Done at Brussels, January 2042.</p>'
    '<p class="oj-signatory">For the Council</p></div></div>\n'
    "</div>\n</body></html>"
)


def test_xhtml_with_an_encoding_declaration_converts() -> None:
    _, _, root = _convert(_CONVEX_XHTML)
    assert [n.text for n in root.iterfind(".//a:article/a:num", NS)] == ["Article 1", "Article 2"]
    texts = [p.text or "" for p in root.iterfind(".//a:article[@eId='art_1']//a:p", NS)]
    assert any("arrêté" in t for t in texts)


# The 1990s renderings use the same classes without the `oj-` prefix.
_CONVEX_UNPREFIXED = _CONVEX_XHTML.replace('class="oj-hd-', 'class="hd-').replace(
    'class="oj-doc-ti"', 'class="doc-ti"'
)


@pytest.mark.parametrize("page", [_CONVEX_XHTML, _CONVEX_UNPREFIXED], ids=["oj", "unprefixed"])
def test_xhtml_title_lines_are_one_title_and_the_running_header_is_not_it(page: str) -> None:
    _, _, root = _convert(page)
    title = " ".join(root.find(".//a:preface", NS).itertext()).split()
    assert " ".join(title) == (
        "COUNCIL REGULATION (EC) No 7/2042 of 1 January 2042 on the registration of lantern keepers"
    )
    # The signature carries no day, so the date can only have come from the title.
    assert root.find(".//a:FRBRExpression/a:FRBRdate", NS).get("date") == "2042-01-01"


@pytest.mark.parametrize("scheme", ["data:", "DATA:"])
def test_an_inline_image_is_named_by_its_alt_not_its_data(scheme: str) -> None:
    _, provenance, root = _convert(_CONVEX_XHTML.replace('src="data:', f'src="{scheme}'))
    texts = [" ".join(p.itertext()) for p in root.iterfind(".//a:article[@eId='art_2']//a:p", NS)]
    assert "[image not transcribed: Image 1]" in texts
    assert provenance["html_images"] == 1


async def test_an_xhtml_file_with_an_encoding_declaration_completes(tmp_path: Path) -> None:
    source = tmp_path / "32042R0007.html"
    source.write_bytes(_CONVEX_XHTML.encode("utf-8"))
    events = [e async for e in dispatch(source, "eu", frbr_work_uri="/akn/eu/act/reg/2042/7")]
    # Conversion is the stage that refused the declaration; enrichment needs config.
    failed = [e.error for e in events if getattr(e, "stage", "") == "eu_directive"]
    assert failed == []
    assert "Complete" in [type(e).__name__ for e in events]


def _cellar_page(body: str) -> str:
    """Body markup in the older Cellar wrapper: `<p><TXT_TE>`, the `<p>` never closed."""
    return (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"><html lang="EN">\n'
        '<head>\n<meta name="DC.source" content="Official Journal L 034 , 06/02/1991 P. 0012 '
        '- 0013; ">\n</head>\n<body>\n<div id="TexteOnly">\n<p>\n<TXT_TE>\n'
        f"{body}\n</TXT_TE>\n</p>\n</div>\n</body>\n</html>"
    )


def _convert_dated(page: str, frbr_work_uri: str) -> str:
    return html_to_akn4eu(page, frbr_work_uri=frbr_work_uri)


def _article_texts(root: etree._Element) -> dict[str, str]:
    return {
        a.get("eId"): " ".join(" ".join(a.itertext()).split())
        for a in root.iterfind(".//a:body/a:article", NS)
    }


# Each heading ends the block before its text, or runs on between two sentences.
_TRAILING_HEADINGS = _cellar_page(
    "<p>COMMISSION DECISION  of 14 January 1991  concerning lantern oil  (91/999/EEC)</p>"
    "<p>  THE COMMISSION OF THE EUROPEAN COMMUNITIES, </p>"
    "<p> Having regard to the Treaty, and in  particular Article 10 thereof, </p>"
    "<p> Whereas lantern oil should be labelled; </p>"
    "<p> HAS ADOPTED THIS DECISION:    Article 1 </p>"
    "<p> 1. Lantern oil shall be labelled. </p>"
    "<p> 2. The label shall name the refinery.  Article 2 </p>"
    "<p> Article 3 shall apply from 1 March 1991.  Article 3  The Member States shall "
    "amend their rules. They shall inform the Commission thereof.  Article 4 </p>"
    "<p> This Decision is addressed to the Member States.   Done at Brussels, "
    "14 January 1991. For the Commission </p>"
    "<p> Member of the Commission   (1) OJ No L 224, 18. 8. 1990, p. 29.   </p>"
    "<p> ANNEX  </p><p>Refineries listed. </p>"
)


def test_a_heading_run_into_the_text_around_it_still_heads_its_article() -> None:
    _, provenance, root = _convert(_TRAILING_HEADINGS)
    texts = _article_texts(root)
    assert list(texts) == ["art_1", "art_2", "art_3", "art_4"]
    assert root.find(".//a:formula/a:p", NS).text == "HAS ADOPTED THIS DECISION:"
    assert texts["art_1"].endswith("2. The label shall name the refinery.")
    # The next number opening a sentence in lower case is a reference, not a heading.
    assert texts["art_2"] == "Article 2 Article 3 shall apply from 1 March 1991."
    assert texts["art_3"].startswith("Article 3 The Member States shall amend")
    assert texts["art_4"] == "Article 4 This Decision is addressed to the Member States."
    conclusions = [p.text for p in root.iterfind(".//a:conclusions/a:p", NS)]
    assert conclusions[0] == "Done at Brussels, 14 January 1991. For the Commission"
    assert provenance["html_source"]["annexes"] == 1  # type: ignore[index]
    assert validate_akn(_convert_dated(_TRAILING_HEADINGS, "/akn/eu/act/dec/1991/999")) == []


# Each heading opens its block and the text runs on after it, behind any division title.
_LEADING_HEADINGS = _cellar_page(
    "<p>COUNCIL DIRECTIVE of 17 April 1989 on lantern keepers (89/999/EEC)</p>"
    "<p> THE COUNCIL OF THE EUROPEAN COMMUNITIES, </p>"
    "<p> Having regard to the Treaty, and in particular Article 54 thereof, </p>"
    "<p> HAS ADOPTED THIS DIRECTIVE: </p>"
    "<p>  TITLE I General provisions Article 1 The object of this Directive is: </p>"
    "<p> (a) to register keepers; </p>"
    "<p> Article 2 1.  Member States shall keep a register. </p>"
    "<p> 2.  The register shall be public. </p>"
    "<p> Article 3   Keepers shall register as laid down in Article 4 Member States' rules."
    " Article 1 Keepers may appeal. </p>"
    "<p> 31. 12. 88No L 374/ Article 4  Fees Member States may charge a fee. </p>"
    "<p>  SECTION II</p>"
    "<p> Cooperation between Member States Article 5 The bodies shall cooperate. </p>"
    "<p> Article 7 Member States shall inform the Commission. </p>"
    "<p> Done at Luxembourg, 17 April 1989. </p>"
)


def test_a_heading_opening_its_text_heads_its_article_behind_a_division_or_page_head() -> None:
    _, _, root = _convert(_LEADING_HEADINGS)
    texts = _article_texts(root)
    # Article 6 is absent from the page: Article 7 still heads, and the gap is visible.
    assert list(texts) == ["art_1", "art_2", "art_3", "art_4", "art_5", "art_7"]
    assert texts["art_1"].startswith("Article 1 The object of this Directive is:")
    art_2 = root.find(".//a:article[@eId='art_2']", NS)
    assert [p.get("eId") for p in art_2.iterfind("a:paragraph", NS)] == [
        "art_2__para_1",
        "art_2__para_2",
    ]
    # Mid-sentence, or out of sequence, a reference stays prose even before a capital.
    assert texts["art_3"] == (
        "Article 3 Keepers shall register as laid down in Article 4 Member States' rules."
        " Article 1 Keepers may appeal."
    )
    # The running head is dropped; a division heading is not modelled and stays as text.
    assert texts["art_4"] == (
        "Article 4 Fees Member States may charge a fee."
        " SECTION II Cooperation between Member States"
    )
    assert texts["art_5"] == "Article 5 The bodies shall cooperate."


# `Article` and its number arrive as two blocks.
_SPLIT_HEADINGS = _cellar_page(
    "<p>COUNCIL REGULATION (EEC) No 9999/92 of 28 December  1992 on lantern oil</p>"
    "<p>THE  COUNCIL OF THE EUROPEAN COMMUNITIES,</p>"
    "<p> Having regard to the Treaty, and in particular Article  43 thereof,</p>"
    "<p>HAS ADOPTED THIS REGULATION:</p><p> </p><p></p>"
    "<p>Article</p><p> 1</p><p>A levy shall be payable on lantern oil.</p>"
    "<p>Article</p><p> 2</p><p>The levy shall be paid yearly.</p><p> </p>"
    "<p> Done at Brussels, 28 December 1992.</p><p> For the Council</p>"
)


def test_a_heading_split_from_its_number_still_heads_its_article() -> None:
    _, _, root = _convert(_SPLIT_HEADINGS)
    assert _article_texts(root) == {
        "art_1": "Article 1 A levy shall be payable on lantern oil.",
        "art_2": "Article 2 The levy shall be paid yearly.",
    }
    assert validate_akn(_convert_dated(_SPLIT_HEADINGS, "/akn/eu/act/reg/1992/9999")) == []
