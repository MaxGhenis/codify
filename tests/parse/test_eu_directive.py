"""EU directive ingestion, unit tests + integration against real fixtures."""

from __future__ import annotations

import asyncio
import contextvars
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest

from codify.akn.elements import ElementBase
from codify.pipeline import ingest_document
from codify.pipeline.events import Complete
from codify.pipeline.formats import eu_directive
from codify.pipeline.formats.eu_directive import (
    FormexNotAnActError,
    _detect_format,
    _frbr_from_formex,
    celex_to_frbr,
    formex_to_akn4eu,
    ingest,
    parse_akn4eu,
)


@pytest.fixture(autouse=True)
def protocol_data(monkeypatch, tmp_path):
    from tests.acquisition.config_support import install_protocol_data

    install_protocol_data(monkeypatch, tmp_path)


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


# --- Unit -----------------------------------------------------------------


def test_detect_format_akn4eu() -> None:
    xml = '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"/>'
    assert _detect_format(xml) == "akn4eu"


def test_detect_format_formex() -> None:
    for root in ("DOCUMENT", "ACT", "DOC", "DIR", "REG"):
        assert _detect_format(f"<{root}/>") == "formex"


def test_detect_format_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown EU XML root"):
        _detect_format("<something-else/>")


@pytest.mark.parametrize(
    "celex, expected",
    [
        ("32016L0943", "/akn/eu/act/dir/2016/943"),
        ("32014L0024", "/akn/eu/act/dir/2014/24"),
        ("32014R0537", "/akn/eu/act/reg/2014/537"),
        ("32016D0001", "/akn/eu/act/dec/2016/1"),
        ("32016L0943R(01)", "/akn/eu/act/dir/2016/943"),
    ],
)
def test_celex_to_frbr(celex: str, expected: str) -> None:
    assert celex_to_frbr(celex) == expected


def test_celex_to_frbr_invalid() -> None:
    with pytest.raises(ValueError):
        celex_to_frbr("not-a-celex")


def test_formex_to_akn4eu_hierarchical_eids() -> None:
    """Sibling counters are per-parent; eIds are path-prefixed."""
    fmx = """
    <DOCUMENT>
      <DATE.OF.DOCUMENT ISO="2020-01-01">1 January 2020</DATE.OF.DOCUMENT>
      <ENACTING.TERMS>
        <CHAP><NO>1</NO><TI>One</TI>
          <ARTICLE><NO>1</NO><PAR><P>A.</P></PAR><PAR><P>B.</P></PAR></ARTICLE>
          <ARTICLE><NO>2</NO><PAR><P>C.</P></PAR></ARTICLE>
        </CHAP>
        <CHAP><NO>2</NO><TI>Two</TI>
          <ARTICLE><NO>1</NO><PAR><P>D.</P></PAR></ARTICLE>
        </CHAP>
      </ENACTING.TERMS>
    </DOCUMENT>
    """.strip()
    out = formex_to_akn4eu(fmx, frbr_work_uri="/akn/eu/act/dir/2020/1")
    doc = parse_akn4eu(out)
    counts = {"chapter": 0, "article": 0, "paragraph": 0}

    def walk(e: ElementBase, eids: list[str]) -> None:
        eids.append(e.akn_eid)
        counts[e.kind] = counts.get(e.kind, 0) + 1
        for c in e.children:
            walk(c, eids)

    eids: list[str] = []
    for top in doc.body:
        walk(top, eids)
    assert counts == {"chapter": 2, "article": 3, "paragraph": 4}
    # chapter 2's article is "chp_2__art_1", counter restarts per parent.
    assert "chp_2__art_1" in eids


def test_formex_extracts_real_date() -> None:
    fmx = (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2018-07-12'>12 July 2018</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS></DOCUMENT>"
    )
    out = formex_to_akn4eu(fmx)
    assert "2018-07-12" in out
    assert 'stub="expression_date"' not in out


def test_formex_raises_when_date_missing() -> None:
    from codify.pipeline.formats.eu_directive import FormexDateMissingError

    fmx = "<DOCUMENT><ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS></DOCUMENT>"
    with pytest.raises(FormexDateMissingError):
        formex_to_akn4eu(fmx)


def test_frbr_from_formex_bib_doc() -> None:
    fmx = (
        "<ACT><BIB.DOC><NO.DOC><NO.CURRENT>0024</NO.CURRENT>"
        "<YEAR>2014</YEAR></NO.DOC></BIB.DOC>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS></ACT>"
    )
    assert _frbr_from_formex(fmx) == "/akn/eu/act/dir/2014/24"


@pytest.mark.parametrize(
    "fmx, match",
    [
        ("<DOC><BIB.DOC/></DOC>", "OJ index wrapper"),
        ("<PUBLICATION><OJ/></PUBLICATION>", "OJ index wrapper"),
        ("<ACT><ENACTING.TERMS/></ACT>", "zero <ARTICLE>"),
    ],
)
def test_formex_rejects_toc_stubs(fmx: str, match: str) -> None:
    with pytest.raises(FormexNotAnActError, match=match):
        formex_to_akn4eu(fmx)


# --- Integration ----------------------------------------------------------


@pytest.fixture(scope="module")
def trade_secrets_xml() -> Path:
    p = FIXTURES_DIR / "eu" / "2016-943" / "eng.xml"
    if not p.exists():
        pytest.skip("fixture missing: run apps/api/scripts/run_ingest.py --skip-ingest")
    return p


@pytest.fixture(scope="module")
def procurement_xml() -> Path:
    p = FIXTURES_DIR / "eu" / "2014-24" / "eng.xml"
    if not p.exists():
        pytest.skip("fixture missing: run apps/api/scripts/run_ingest.py --skip-ingest")
    return p


def _ingest_sync(path: Path) -> Complete:
    async def go() -> list:
        return [e async for e in ingest_document(path, "eu")]

    events = asyncio.run(go())
    assert events, "no events yielded"
    last = events[-1]
    assert isinstance(last, Complete), f"expected Complete, got {last!r}"
    return last


def test_2016_943_ingests_with_21_articles(trade_secrets_xml: Path) -> None:
    final = _ingest_sync(trade_secrets_xml)
    assert final.document.frbr_work_uri == "/akn/eu/act/dir/2016/943"
    assert final.document.frbr_expression_uri.endswith("@2016-06-08")
    counts: dict[str, int] = {}

    def walk(e: ElementBase) -> None:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        for c in e.children:
            walk(c)

    for top in final.document.body:
        walk(top)
    assert counts.get("article", 0) == 21


def test_2014_24_ingests_with_real_metadata(procurement_xml: Path) -> None:
    final = _ingest_sync(procurement_xml)
    assert final.document.frbr_work_uri == "/akn/eu/act/dir/2014/24"
    assert final.document.frbr_expression_uri.endswith("@2014-02-26")
    # Procurement directive is large, chapters > 10, articles > 50.
    counts: dict[str, int] = {}

    def walk(e: ElementBase) -> None:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        for c in e.children:
            walk(c)

    for top in final.document.body:
        walk(top)
    # Source has 25 untyped DIVISIONs nested as TITLE/CHAPTER/SECTION;
    # heading-text inference splits them across the three AKN container kinds.
    div_total = (
        counts.get("title", 0)
        + counts.get("chapter", 0)
        + counts.get("section", 0)
        + counts.get("part", 0)
    )
    assert div_total >= 20
    assert counts.get("article", 0) >= 50


# --- Production-quality assertions ---------------------------------------


def test_2016_943_emits_substantive_body(trade_secrets_xml: Path) -> None:
    """Trade-secrets directive (FORMEX 81 KB) must emit ≥50 KB AKN body."""
    final = _ingest_sync(trade_secrets_xml)
    assert len(final.akn_xml) > 50_000, (
        f"trade-secrets AKN only {len(final.akn_xml)} bytes — empty-stub regression"
    )
    assert "<preface>" in final.akn_xml
    assert "<preamble>" in final.akn_xml
    assert "<conclusions>" in final.akn_xml
    assert final.akn_xml.count("<recital ") >= 30


def test_2014_24_emits_substantive_body(procurement_xml: Path) -> None:
    """Procurement directive (FORMEX 477 KB) must emit ≥200 KB AKN body."""
    final = _ingest_sync(procurement_xml)
    assert len(final.akn_xml) > 200_000, (
        f"procurement AKN only {len(final.akn_xml)} bytes — empty-stub regression"
    )
    assert "<preface>" in final.akn_xml
    assert "<preamble>" in final.akn_xml
    assert "<conclusions>" in final.akn_xml
    # 138 CONSID in source.
    assert final.akn_xml.count("<recital ") >= 100


def test_2016_943_validates_strict(trade_secrets_xml: Path) -> None:
    """Final enriched AKN passes the OASIS XSD in strict mode."""
    from codify.akn._schema import validate_akn

    final = _ingest_sync(trade_secrets_xml)
    validate_akn(final.akn_xml, strict=True)  # raises on any schema error


def test_2014_24_validates_strict(procurement_xml: Path) -> None:
    from codify.akn._schema import validate_akn

    final = _ingest_sync(procurement_xml)
    validate_akn(final.akn_xml, strict=True)


def test_meta_carries_publication_lifecycle_classification(
    trade_secrets_xml: Path,
) -> None:
    final = _ingest_sync(trade_secrets_xml)
    xml = final.akn_xml
    assert "<publication " in xml
    assert "Official Journal" in xml
    assert "<lifecycle " in xml
    assert "<references " in xml


def test_act_carries_required_name_attribute(trade_secrets_xml: Path) -> None:
    """AKN3 schema requires <act @name>; we set it from the FRBR doctype."""
    final = _ingest_sync(trade_secrets_xml)
    assert 'name="directive"' in final.akn_xml


def test_formex_to_akn_is_deterministic(trade_secrets_xml: Path) -> None:
    """Same input → byte-identical AKN. No date.today(), no UUIDs, no clock."""
    src_xml = trade_secrets_xml.read_text()
    out_a = formex_to_akn4eu(src_xml, frbr_work_uri="/akn/eu/act/dir/2016/943")
    out_b = formex_to_akn4eu(src_xml, frbr_work_uri="/akn/eu/act/dir/2016/943")
    assert out_a == out_b


# --- Per-element fragment tests -----------------------------------------


def _wrap(body_inner: str) -> str:
    return (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 January 2024</DATE.OF.DOCUMENT>"
        f"<ENACTING.TERMS><ARTICLE IDENTIFIER='001'>{body_inner}</ARTICLE></ENACTING.TERMS></DOCUMENT>"  # noqa: E501
    )


def test_ht_italic_maps_to_i() -> None:
    out = formex_to_akn4eu(_wrap("<PARAG><TXT>see <HT TYPE='ITALIC'>infra</HT> p.10</TXT></PARAG>"))
    assert "<i>infra</i>" in out


def test_ht_bold_maps_to_b() -> None:
    out = formex_to_akn4eu(_wrap("<PARAG><TXT>see <HT TYPE='BOLD'>warning</HT></TXT></PARAG>"))
    assert "<b>warning</b>" in out


def test_ht_uc_uppercases_text() -> None:
    """UC has no AKN semantic, apply text.upper() and emit plain text."""
    out = formex_to_akn4eu(_wrap("<PARAG><TXT>see <HT TYPE='UC'>schulz</HT></TXT></PARAG>"))
    assert "SCHULZ" in out
    assert '<inline name="uc">' not in out


def test_quot_pair_emits_curly_quotes_only() -> None:
    """AKN <quotedText> is reserved for amendment-mod contexts; for prose
    quotation we emit the curly marks as plain text."""
    out = formex_to_akn4eu(
        _wrap("<PARAG><TXT>he said <QUOT.START/>hello world<QUOT.END/> firmly</TXT></PARAG>")
    )
    assert '<inline name="quoted">' not in out
    assert "“" in out and "”" in out
    # The quoted term rides QUOT.START's tail; it must survive, marks and all.
    # (Regression: empty “” when the tail was dropped, e.g. Art.2 definitions.)
    assert "“hello world”" in out


def test_paragraph_num_has_no_trailing_dot() -> None:
    out = formex_to_akn4eu(_wrap("<PARAG><NO.PARAG>1</NO.PARAG><TXT>body</TXT></PARAG>"))
    assert "<num>1</num>" in out
    assert "<num>1.</num>" not in out


def test_recital_eId_is_path_prefixed() -> None:
    out = formex_to_akn4eu(
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<PREAMBLE><CONSID><NP><NO.P>(1)</NO.P><TXT>Whereas one.</TXT></NP></CONSID></PREAMBLE>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS></DOCUMENT>"
    )
    assert 'eId="recitals__rec_1"' in out


def test_date_inline_carries_iso() -> None:
    out = formex_to_akn4eu(
        _wrap(
            "<PARAG><TXT>signed <DATE ISO='20160608'>8 June 2016</DATE> in Strasbourg</TXT></PARAG>"
        )
    )
    assert '<date date="2016-06-08">8 June 2016</date>' in out


def test_item_with_np_no_p_txt_pattern_emits_content() -> None:
    """Stage 9 regression: real FORMEX ITEMs use <NP><NO.P>(1)</NO.P><TXT>…</TXT></NP>;
    earlier code only knew <ITEM.CONT> and dropped 90% of definitions."""
    out = formex_to_akn4eu(
        _wrap(
            "<PARAG><TXT>shall mean:</TXT>"
            "<LIST>"
            "<ITEM><NP><NO.P>(1)</NO.P><TXT>contracting authority means the State.</TXT></NP></ITEM>"  # noqa: E501
            "<ITEM><NP><NO.P>(2)</NO.P><TXT>economic operator means any natural person.</TXT></NP></ITEM>"  # noqa: E501
            "</LIST></PARAG>"
        )
    )
    assert "contracting authority means the State." in out
    assert "economic operator means any natural person." in out
    assert out.count("<num>(1)</num>") == 1
    assert out.count("<p> </p>") == 0  # no degenerate stub fallbacks


def test_note_marker_uses_note_id_not_numbering() -> None:
    """Stage 12 regression: NUMBERING is a style enum (ARAB/ROM/ALPHA),
    not the visible marker. Earlier code emitted marker=\"ARAB\" everywhere."""
    out = formex_to_akn4eu(
        _wrap(
            "<PARAG><TXT>see "
            '<NOTE NOTE.ID="E0001" NUMBERING="ARAB" TYPE="FOOTNOTE">'
            "<P>OJ L 1, 1.1.2024, p. 1.</P></NOTE>"
            " for the original.</TXT></PARAG>"
        )
    )
    assert 'marker="ARAB"' not in out
    assert 'eId="fn_E0001"' in out
    assert 'placement="bottom"' in out


def test_ft_number_type_emits_plain_text_not_authorialnote() -> None:
    """Stage 12 regression: FT@TYPE=NUMBER is a financial threshold, not a
    footnote marker. Earlier code wrapped these in <authorialNote>."""
    out = formex_to_akn4eu(
        _wrap('<PARAG><TXT>threshold <FT TYPE="NUMBER">750000</FT> EUR applies.</TXT></PARAG>')
    )
    # The threshold value flows as plain text inside the <p>.
    assert "750000" in out
    # And is NOT wrapped as an authorial note.
    assert out.count("<authorialNote") == 0


def test_division_heading_inference_picks_title_chapter_section() -> None:
    """Stage 12 regression: untyped DIVISIONs nested as TITLE/CHAPTER/SECTION
    were all defaulting to <chapter>. Heading-text inference now classifies
    them per AKN container kind."""
    out = formex_to_akn4eu(
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS>"
        "<DIVISION><TITLE><NO>I</NO><TI>TITLE I</TI></TITLE>"
        "  <DIVISION><TITLE><NO>1</NO><TI>CHAPTER 1</TI></TITLE>"
        "    <DIVISION><TITLE><NO>1</NO><TI>Section 1</TI></TITLE>"
        "      <ARTICLE IDENTIFIER='001'><TI.ART><STI.ART>Article 1</STI.ART></TI.ART><PARAG><TXT>x.</TXT></PARAG></ARTICLE>"  # noqa: E501
        "    </DIVISION>"
        "  </DIVISION>"
        "</DIVISION>"
        "</ENACTING.TERMS></DOCUMENT>"
    )
    assert "<title " in out
    assert "<chapter " in out
    assert "<section " in out


def test_2014_24_word_count_ratio(procurement_xml: Path) -> None:
    """Adversarial-review regression: word ratio output/source must stay >0.80.
    Earlier (stub-emitter) version was 0.71 due to NP-pattern blindness."""
    final = _ingest_sync(procurement_xml)
    src = procurement_xml.read_text()
    src_words = len(re.sub(r"<[^>]+>", " ", src).split())
    out_words = len(re.sub(r"<[^>]+>", " ", final.akn_xml).split())
    ratio = out_words / src_words
    assert ratio > 0.80, f"word ratio {ratio:.3f} below 0.80 — regression likely"


def test_eid_uniqueness_within_document(trade_secrets_xml: Path) -> None:
    """AKN spec: eIds unique within document scope."""
    import lxml.etree as et

    final = _ingest_sync(trade_secrets_xml)
    root = et.fromstring(final.akn_xml.encode())
    eids = root.xpath("//*/@eId")
    duplicates = {eid for eid in eids if eids.count(eid) > 1}
    assert not duplicates, f"duplicate eIds: {sorted(duplicates)[:5]}"


def test_validator_failure_yields_failed_not_complete(
    trade_secrets_xml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 13 regression: validator crash must yield Failed and suppress Complete."""
    from codify.pipeline import formats as _formats_pkg
    from codify.pipeline.events import Failed

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("simulated validator crash")

    monkeypatch.setattr(_formats_pkg.eu_directive, "validate_akn", _boom, raising=False)
    # Also intercept the `from ... import validate_akn` already-bound symbol
    # inside eu_directive.ingest.
    from codify.pipeline.enrich import validator as _validator_mod

    monkeypatch.setattr(_validator_mod, "validate_akn", _boom)

    async def go() -> list:
        return [
            ev
            async for ev in __import__(
                "codify.pipeline", fromlist=["ingest_document"]
            ).ingest_document(trade_secrets_xml, "eu")
        ]

    events = asyncio.run(go())
    failed = [e for e in events if isinstance(e, Failed) and e.stage == "validator"]
    completes = [e for e in events if isinstance(e, Complete)]
    assert failed, "expected Failed(stage='validator') after monkey-patched crash"
    assert not completes, "Complete must NOT fire after validator failure"


def test_only_formulas_and_figures_remain_unmapped_in_an_attachment() -> None:
    """Tables used to be counted here too. They are mapped now, so a count of
    three would mean the annex table had gone back to being a placeholder."""
    fmx = (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS>"
        "<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE>"
        "<TBL/><FORMULA/><FIGURE/></ANNEX></DOCUMENT>"
    )
    provenance: dict = {}
    out = formex_to_akn4eu(fmx, provenance=provenance)
    assert "<attachments>" in out
    assert "[TBL omitted: see source]" not in out
    assert provenance.get("unmapped_blocks") == 2


def test_ft_with_ref_note_emits_noteref() -> None:
    """Stage 14 regression: <FT TYPE='FOOTNOTE' REF.NOTE='E0001'>1</FT>
    emits a <noteRef href='#fn_E0001'> at the call site (link to the
    NOTE body). Currently zero coverage on this path."""
    fmx = (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><PARAG><TXT>"
        "see footnote <FT TYPE='FOOTNOTE' REF.NOTE='E0001'>1</FT> below."
        "</TXT></PARAG></ARTICLE></ENACTING.TERMS></DOCUMENT>"
    )
    out = formex_to_akn4eu(fmx)
    assert '<noteRef href="#fn_E0001"' in out
    assert 'marker="1"' in out


def test_publication_partial_when_fields_missing() -> None:
    """Stage 15 item 4: missing BIB.INSTANCE/DOCUMENT.REF fields produces a
    partial <publication> + provenance['publication_missing']."""
    fmx = (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<BIB.INSTANCE><DOCUMENT.REF><COLL>L</COLL></DOCUMENT.REF></BIB.INSTANCE>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO></ARTICLE></ENACTING.TERMS></DOCUMENT>"
    )
    provenance: dict = {}
    out = formex_to_akn4eu(fmx, provenance=provenance)
    assert "<publication " in out  # always emitted, even when partial
    assert "no_oj" in provenance.get("publication_missing", [])
    assert "year" in provenance.get("publication_missing", [])


def test_unresolved_refs_counted() -> None:
    """Stage 15 item 5: REF.DOC.OJ with missing fields and tocItems both
    produce href='#unresolved' anchors; the count surfaces in provenance."""
    fmx = (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<TOC><TOC.BLK>"
        "<TOC.ITEM><NO.ITEM>I</NO.ITEM><ITEM.CONT>SCOPE</ITEM.CONT></TOC.ITEM>"
        "</TOC.BLK></TOC>"
        "<ENACTING.TERMS><ARTICLE><NO>1</NO><PARAG><TXT>see "
        "<REF.DOC.OJ><COLL>L</COLL></REF.DOC.OJ> for details.</TXT></PARAG></ARTICLE>"
        "</ENACTING.TERMS></DOCUMENT>"
    )
    provenance: dict = {}
    formex_to_akn4eu(fmx, provenance=provenance)
    # 1 tocItem + 1 unresolvable REF.DOC.OJ = 2 unresolved anchors.
    assert provenance.get("unresolved_refs", 0) >= 2


def test_ht_stroke_preserved_as_inline_strike() -> None:
    """Stage 15 item 6: HT@TYPE=STROKE preserves the legal-amendment
    semantic via <inline name='strike'>, not silent plain text."""
    out = formex_to_akn4eu(
        _wrap("<PARAG><TXT>previous text <HT TYPE='STROKE'>removed</HT> stays.</TXT></PARAG>")
    )
    assert '<inline name="strike">removed</inline>' in out


def test_ingest_determinism_through_full_pipeline(trade_secrets_xml: Path) -> None:
    """ingest() must be byte-deterministic across runs."""

    async def run() -> str:
        async for ev in __import__(
            "codify.pipeline.formats.eu_directive", fromlist=["ingest"]
        ).ingest(trade_secrets_xml, "eu"):
            if isinstance(ev, Complete):
                return ev.akn_xml
        return ""

    a = asyncio.run(run())
    b = asyncio.run(run())
    assert a == b, "non-deterministic ingest output"


def test_list_emits_blocklist_with_items() -> None:
    out = formex_to_akn4eu(
        _wrap(
            "<PARAG><TXT>shall include:</TXT>"
            "<LIST>"
            "  <ITEM><NO.ITEM>(a)</NO.ITEM><ITEM.CONT><ALINEA><TXT>foo</TXT></ALINEA></ITEM.CONT></ITEM>"  # noqa: E501
            "  <ITEM><NO.ITEM>(b)</NO.ITEM><ITEM.CONT><ALINEA><TXT>bar</TXT></ALINEA></ITEM.CONT></ITEM>"  # noqa: E501
            "</LIST></PARAG>"
        )
    )
    assert "<blockList " in out
    assert out.count("<item ") == 2
    assert "<num>(a)</num>" in out
    assert "<num>(b)</num>" in out


# --- Quoted structures in amending acts ------------------------------------


def _amending(item_inner: str) -> str:
    return (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><TI.ART>Article 1</TI.ART>"
        "<ALINEA>Regulation (EU) 2017/2400 is amended as follows:"
        f"<LIST><ITEM>{item_inner}</ITEM></LIST></ALINEA>"
        "</ARTICLE></ENACTING.TERMS></DOCUMENT>"
    )


QUOTED_ARTICLE_ITEM = (
    "<NP><NO.P>(1)</NO.P><TXT>Article 1 is replaced by the following:</TXT>"
    "<P><QUOT.S LEVEL='1'><ARTICLE IDENTIFIER='001'>"
    "<TI.ART><QUOT.START CODE='2018'/>Article 1</TI.ART>"
    "<STI.ART>Subject matter</STI.ART>"
    "<ALINEA>This Regulation lays down licensing rules.<QUOT.END CODE='2019'/></ALINEA>"
    "</ARTICLE></QUOT.S></P></NP>"
)

QUOTED_LIST_ITEM = (
    "<NP><NO.P>(2)</NO.P><TXT>the following points are added:</TXT>"
    "<P><QUOT.S LEVEL='1'><LIST>"
    "<ITEM><NP><NO.P>(a)</NO.P><TXT>medium lorries;</TXT></NP></ITEM>"
    "<ITEM><NP><NO.P>(b)</NO.P><TXT>heavy buses.</TXT></NP></ITEM>"
    "</LIST></QUOT.S></P></NP>"
)


def test_quoted_article_becomes_a_mod() -> None:
    """The replacement text an amending act introduces was dropped entirely:
    QUOT.S has no branch, so only the lead-in survived."""
    out = formex_to_akn4eu(_amending(QUOTED_ARTICLE_ITEM))
    assert "<mod " in out
    assert "<quotedStructure " in out
    assert "This Regulation lays down licensing rules." in out
    assert "<num>Article 1</num>" in out
    assert "<heading>Subject matter</heading>" in out


def test_quoted_article_text_sits_inside_the_quote() -> None:
    out = formex_to_akn4eu(_amending(QUOTED_ARTICLE_ITEM))
    quoted = out[out.index("<quotedStructure ") : out.index("</quotedStructure>")]
    assert "This Regulation lays down licensing rules." in quoted
    # The instruction stays outside it, as prose.
    assert "is replaced by the following" not in quoted


def test_quoted_list_becomes_a_mod() -> None:
    out = formex_to_akn4eu(_amending(QUOTED_LIST_ITEM))
    quoted = out[out.index("<quotedStructure ") : out.index("</quotedStructure>")]
    assert "medium lorries;" in quoted
    assert "heavy buses." in quoted


def test_unquoted_item_is_not_a_mod() -> None:
    plain = "<NP><NO.P>(1)</NO.P><TXT>the words 'motor vehicle' are deleted.</TXT></NP>"
    out = formex_to_akn4eu(_amending(plain))
    assert "<mod " not in out
    assert "motor vehicle" in out


ALINEA_QUOTE = (
    "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
    "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><TI.ART>Article 1</TI.ART>"
    "<ALINEA>Article 3 is replaced by the following:"
    "<QUOT.S LEVEL='1'><ARTICLE IDENTIFIER='003'><TI.ART>Article 3</TI.ART>"
    "<ALINEA>Definitions apply.</ALINEA></ARTICLE></QUOT.S>"
    "</ALINEA></ARTICLE></ENACTING.TERMS></DOCUMENT>"
)


def test_quoted_block_directly_under_an_alinea() -> None:
    """Some dialects hang QUOT.S off the ALINEA rather than a list item."""
    out = formex_to_akn4eu(ALINEA_QUOTE)
    assert "<quotedStructure " in out
    assert "Definitions apply." in out


def test_quote_follows_its_lead_in_and_appears_once() -> None:
    """The ALINEA's own flatten would otherwise repeat the quote it contains,
    after emitting it ahead of the instruction that introduces it."""
    out = formex_to_akn4eu(ALINEA_QUOTE)
    assert out.count("Definitions apply.") == 1
    assert out.index("replaced by the following") < out.index("<mod ")


def test_quote_inside_an_np_appears_once() -> None:
    out = formex_to_akn4eu(
        _amending(
            "<NP><NO.P>(1)</NO.P>Point 2 is replaced by the following:"
            "<P><QUOT.S><ALINEA>New point text.</ALINEA></QUOT.S></P></NP>"
        )
    )
    assert out.count("New point text.") == 1
    assert out.count("Point 2 is replaced by the following:") == 1
    assert out.index("replaced by the following") < out.index("<mod ")


def test_two_quotes_in_one_item_get_distinct_eids() -> None:
    """Several branches feed one item, so a per-branch counter would collide."""
    out = formex_to_akn4eu(
        _amending(
            "<NP><NO.P>(1)</NO.P><TXT>the following are inserted:</TXT>"
            "<P><QUOT.S><ALINEA>First insert.</ALINEA></QUOT.S></P></NP>"
            "<P><QUOT.S><ALINEA>Second insert.</ALINEA></QUOT.S></P>"
        )
    )
    eids = re.findall(r'<mod eId="([^"]+)"', out)
    assert len(eids) == 2
    assert len(set(eids)) == 2
    assert "First insert." in out
    assert "Second insert." in out


def test_quote_inside_an_item_cont_appears_once() -> None:
    """ITEM.CONT with a quote but no ALINEA or TXT falls to the same flatten."""
    out = formex_to_akn4eu(
        _amending(
            "<NO.ITEM>(3)</NO.ITEM><ITEM.CONT>Point 5 is replaced by the following:"
            "<P><QUOT.S><ALINEA>Cont point text.</ALINEA></QUOT.S></P></ITEM.CONT>"
        )
    )
    assert out.count("Cont point text.") == 1
    assert out.count("Point 5 is replaced by the following:") == 1
    assert out.index("replaced by the following") < out.index("<mod ")


def test_mod_sits_in_a_block_of_its_own() -> None:
    """`mod` is inline in AKN 3.0, so a block container cannot hold it directly;
    the whole document fails schema validation when it does."""
    from lxml import etree

    root = etree.fromstring(formex_to_akn4eu(ALINEA_QUOTE).encode())
    mods = root.findall(".//{*}mod")
    assert mods
    assert all(etree.QName(m.getparent()).localname == "p" for m in mods)


def test_quoted_children_the_walker_does_not_model_survive() -> None:
    """A QUOT.S mixing a handled child with an unhandled one used to keep the
    first and drop the rest in silence."""
    out = formex_to_akn4eu(
        _amending(
            "<NP><NO.P>(1)</NO.P><TXT>replaced by the following:</TXT>"
            "<P><QUOT.S>"
            "<ALINEA>Handled alinea.</ALINEA>"
            "<PARAG><NO.PARAG>1.</NO.PARAG><ALINEA>Quoted paragraph.</ALINEA></PARAG>"
            "<DLIST><DLIST.ITEM><TERM>Term</TERM><DEFINITION>Its meaning.</DEFINITION>"
            "</DLIST.ITEM></DLIST>"
            "<NOTE.ALINEA>CO<HT TYPE='SUB'>2</HT> emissions apply.</NOTE.ALINEA>"
            "<TBL><CORPUS><ROW><CELL>A</CELL><CELL>B</CELL></ROW></CORPUS></TBL>"
            "</QUOT.S></P></NP>"
        )
    )
    quoted = out[out.index("<quotedStructure ") : out.index("</quotedStructure>")]
    assert "Handled alinea." in quoted
    assert "Quoted paragraph." in quoted
    assert "Its meaning." in quoted
    # Adjacent elements keep their boundary: flattening the block fuses them.
    assert "TermIts" not in quoted
    # Text around a child makes that child inline, so it is not split off.
    assert "<p>CO<sub>2</sub> emissions apply.</p>" in quoted


def test_quoted_table_eid_has_no_empty_segment() -> None:
    """eIds are __-segmented, so an empty one gives parse_eid a phantom
    ancestor."""
    out = formex_to_akn4eu(
        _amending(
            "<NP><NO.P>(1)</NO.P><TXT>replaced by the following:</TXT>"
            "<P><QUOT.S><TBL><CORPUS><ROW><CELL>A</CELL></ROW></CORPUS></TBL></QUOT.S></P></NP>"
        )
    )
    eids = re.findall(r'eId="([^"]+)"', out)
    assert not [e for e in eids if "____" in e]
    assert [e for e in eids if e.endswith("__qstr_1__table_1")]


_TBL = "<TBL><CORPUS><ROW><CELL>A</CELL><CELL>B</CELL></ROW></CORPUS></TBL>"
_INTRO = "The following rates apply."
_NOTE = "Rates are expressed per tonne."


def _annex(annex_inner: str) -> str:
    return (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><PARAG><TXT>Body.</TXT></PARAG></ARTICLE>"
        "</ENACTING.TERMS>"
        f"<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE>{annex_inner}</ANNEX></DOCUMENT>"
    )


class TestProseEitherSideOfAnAnnexTable:
    """Three placements of one table, as controls for each other. The walk
    descends a wrapper holding a table by element children, so a `<P>`'s own text
    and the table's tail had nowhere to go and were dropped uncounted."""

    @staticmethod
    def _placements() -> dict[str, str]:
        return {
            "inside a P": f"<P>{_INTRO}{_TBL}{_NOTE}</P>",
            "sibling blocks": f"<P>{_INTRO}</P>{_TBL}<P>{_NOTE}</P>",
            "loose text": f"{_INTRO}{_TBL}{_NOTE}",
        }

    def test_every_placement_keeps_the_prose_on_both_sides(self) -> None:
        for label, inner in self._placements().items():
            out = formex_to_akn4eu(_annex(inner))
            assert "<table" in out, label
            assert ">A<" in out, label
            assert _INTRO in out, label
            assert _NOTE in out, label

    def test_every_placement_keeps_the_reading_order(self) -> None:
        """The prose reads as an introduction and a note to the table between
        them, so emitting both after it would still lose the sense."""
        for label, inner in self._placements().items():
            out = formex_to_akn4eu(_annex(inner))
            assert out.index(_INTRO) < out.index("<table") < out.index(_NOTE), label

    def test_the_prose_is_its_own_paragraph_not_fused_to_the_table(self) -> None:
        out = formex_to_akn4eu(_annex(f"<P>{_INTRO}{_TBL}{_NOTE}</P>"))
        assert f"<p>{_INTRO}</p>" in out
        assert f"<p>{_NOTE}</p>" in out

    def test_loose_text_is_counted(self) -> None:
        """A count is what makes a future silent drop visible: the earlier loss
        reported clean."""
        provenance: dict = {}
        formex_to_akn4eu(_annex(f"<P>{_INTRO}{_TBL}{_NOTE}</P>"), provenance=provenance)
        assert provenance.get("annex_loose_text") == 2

    def test_whitespace_between_blocks_is_not_a_paragraph(self) -> None:
        provenance: dict = {}
        out = formex_to_akn4eu(
            _annex(f"<P>{_INTRO}</P>\n  {_TBL}\n  <P>{_NOTE}</P>\n"),
            provenance=provenance,
        )
        assert provenance.get("annex_loose_text", 0) == 0
        assert "<p></p>" not in out
        assert "<p/>" not in out


_MIN_FORMEX_ACT = (
    "<ACT><BIB.INSTANCE><DATE ISO='20200101'>20200101</DATE>"
    "<NO.DOC TYPE='OJ'><YEAR>2020</YEAR><NO.CURRENT>0001</NO.CURRENT></NO.DOC>"
    "</BIB.INSTANCE><ENACTING.TERMS><ARTICLE><TI.ART>Article 1</TI.ART>"
    "<PARAG><TXT>Member States shall comply.</TXT></PARAG></ARTICLE></ENACTING.TERMS></ACT>"
)


def test_the_cpu_bound_passes_run_on_the_pipeline_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversion and validation are the FORMEX lane's whole CPU cost and are
    synchronous. Left on the loop, one large act holds the health probe for its own
    duration whatever the queues are capped to: 19 to 31 s on the corpus's largest.

    On the pipeline's own pool, not the default executor, which is shared with the
    readiness probe, the PDF export and the OCR crops.
    """
    src = tmp_path / "act.xml"
    src.write_text(_MIN_FORMEX_ACT, encoding="utf-8")
    dispatched: list[str] = []
    threads: list[str] = []

    real_seam = eu_directive._on_the_cpu_pool

    async def recording_seam(fn: Callable[..., Any], *args: Any) -> Any:
        dispatched.append(getattr(fn, "__name__", repr(fn)))
        return await real_seam(fn, *args)

    real_frbr = eu_directive._frbr_from_formex

    def recording_frbr(xml: str) -> str:
        threads.append(threading.current_thread().name)
        return real_frbr(xml)

    marker = contextvars.ContextVar("marker", default="MISSING")
    seen_context: list[str] = []
    real_detect = eu_directive._detect_format

    def recording_detect(xml: str | bytes) -> Literal["akn4eu", "formex"]:
        seen_context.append(marker.get())
        threads.append(threading.current_thread().name)
        return real_detect(xml)

    monkeypatch.setattr(eu_directive, "_on_the_cpu_pool", recording_seam)
    monkeypatch.setattr(eu_directive, "_frbr_from_formex", recording_frbr)
    monkeypatch.setattr(eu_directive, "_detect_format", recording_detect)

    async def go() -> None:
        marker.set("SET-BY-CALLER")
        async for _ in ingest(src, "eu"):
            pass

    asyncio.run(go())

    # Both passes cross the handoff. The conversion goes as the closure that also
    # reads the file, sniffs the format and derives the URI, since each of those
    # touches the whole document and would otherwise run on the loop first.
    assert "formex_conversion" in dispatched, dispatched
    assert "_validate_native" in dispatched, dispatched
    # Dispatch alone cannot see where the work lands: only the thread name
    # distinguishes this pool from the shared default executor, and only the
    # thread distinguishes reading and sniffing inside the callable from before it.
    assert threads and all(t.startswith("codify-cpu") for t in threads), threads
    # run_in_executor drops contextvars unless the context is copied across.
    assert seen_context == ["SET-BY-CALLER"], seen_context


def _body(body_inner: str) -> str:
    return (
        "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
        f"<ENACTING.TERMS>{body_inner}</ENACTING.TERMS></DOCUMENT>"
    )


class TestTheBodyWalkCountsWhatItDiscards:
    """The body walk emits only DIVISION/CHAP/SECTION/ARTICLE. Everything else it
    meets is dropped, and it used to report a clean conversion while doing it. A
    bare paragraph is not valid under `<body>`, so these are counted, not guessed
    at: the count says the shape exists and wants mapping."""

    @staticmethod
    def _convert(body_inner: str) -> tuple[str, dict]:
        provenance: dict = {}
        return formex_to_akn4eu(_body(body_inner), provenance=provenance), provenance

    def test_a_sibling_of_the_articles_is_counted(self) -> None:
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "<GR.SEQ><P>Transitional note between the articles.</P></GR.SEQ>"
            "<ARTICLE IDENTIFIER='002'><PARAG><TXT>Two.</TXT></PARAG></ARTICLE>"
        )
        assert "Transitional note" not in out, "emitted after all; update the count too"
        assert prov.get("body_discarded_text") == 1

    def test_prose_before_a_chapters_first_article_is_counted(self) -> None:
        out, prov = self._convert(
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<P>Introductory prose standing before the first article.</P>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>"
        )
        assert "Introductory prose" not in out
        assert prov.get("body_discarded_text") == 1

    def test_a_lead_in_sentence_beside_numbered_points_is_counted(self) -> None:
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><TI.ART><STI.ART>1</STI.ART></TI.ART>"
            "<TXT>This introductory sentence precedes the numbered points.</TXT>"
            "<PARAG><NO.PARAG>1.</NO.PARAG><ALINEA><TXT>First point.</TXT></ALINEA></PARAG>"
            "</ARTICLE>"
        )
        assert "introductory sentence" not in out
        assert "First point." in out
        assert prov.get("body_discarded_text") == 1

    def test_a_lead_in_sentence_inside_a_chapter_is_counted(self) -> None:
        """The article scan has to run under a division too, not only at the top."""
        out, prov = self._convert(
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><TXT>A lead-in inside the chapter.</TXT>"
            "<PARAG><NO.PARAG>1.</NO.PARAG><ALINEA><TXT>First point.</TXT></ALINEA></PARAG>"
            "</ARTICLE></DIVISION>"
        )
        assert "lead-in inside the chapter" not in out
        assert prov.get("body_discarded_text") == 1

    def test_a_loss_nested_two_divisions_deep_is_counted(self) -> None:
        """Divisions nest title inside part inside chapter, so a scan that stops
        at the first level misses most of an act."""
        out, prov = self._convert(
            "<DIVISION TYPE='TITLE'><TITLE><TI>TITLE I</TI></TITLE>"
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<P>Prose buried two divisions down.</P>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "</DIVISION></DIVISION>"
        )
        assert "buried two divisions down" not in out
        assert prov.get("body_discarded_text") == 1

    def test_text_between_the_blocks_is_counted(self) -> None:
        """The walk reads elements, so a sentence sitting loose between two of
        them is not read at all. Not seen in the corpus; counted so it would be."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "A sentence loose between the articles."
            "<ARTICLE IDENTIFIER='002'><PARAG><TXT>Two.</TXT></PARAG></ARTICLE>"
        )
        assert "loose between the articles" not in out
        assert prov.get("body_discarded_text") == 1

    def test_whitespace_between_the_blocks_is_not_a_loss(self) -> None:
        _, prov = self._convert(
            "\n  <ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>\n  "
        )
        assert "body_discarded_text" not in prov

    def test_an_article_with_no_paragraphs_keeps_everything(self) -> None:
        """The itertext fallback already carries the whole element, so counting
        it would report a loss that did not happen."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><TXT>The only sentence.</TXT></ARTICLE>"
        )
        assert "The only sentence." in out
        assert "body_discarded_text" not in prov

    def test_a_subtitle_beside_the_heading_becomes_a_subheading(self) -> None:
        out, prov = self._convert(
            "<CHAP><NO>1</NO><TI>General provisions</TI><STI>Scope and application</STI>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></CHAP>"
        )
        assert "<heading>General provisions</heading>" in out
        assert "<subheading>Scope and application</subheading>" in out
        assert "body_discarded_text" not in prov

    def test_a_subtitle_inside_the_title_wrapper_becomes_a_subheading(self) -> None:
        """The wrapper holds both, and only its TI was ever read."""
        out, _ = self._convert(
            "<DIVISION><TITLE><TI>General provisions</TI><STI>Scope note</STI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>"
        )
        assert "<heading>General provisions</heading>" in out
        assert "<subheading>Scope note</subheading>" in out

    def test_the_subheading_follows_the_heading(self) -> None:
        """AKN orders num, heading, subheading; out of order it will not validate."""
        out, _ = self._convert(
            "<CHAP><NO>1</NO><TI>General provisions</TI><STI>Scope note</STI>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></CHAP>"
        )
        assert out.index("<num>1</num>") < out.index("<heading>") < out.index("<subheading>")

    def test_a_number_inside_the_title_wrapper_is_counted(self) -> None:
        """`find("NO")` looks at the division's own children, so a NO one level
        down is never read. Not recovered here, but no longer silent."""
        out, prov = self._convert(
            "<DIVISION><TITLE><NO>I</NO><TI>General provisions</TI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>"
        )
        assert "<heading>General provisions</heading>" in out
        assert "<num>I</num>" not in out
        assert prov.get("body_discarded_text") == 1

    def test_an_article_holding_only_a_reference_is_counted(self) -> None:
        """The text fallback keeps text, and an element carrying only attributes
        has none, so the article emits empty and the reference is gone."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><GRAPHIC FILE='diagram.tif'/></ARTICLE>"
        )
        assert "diagram.tif" not in out
        assert prov.get("body_discarded_text") == 1

    def test_a_processing_instruction_does_not_break_the_count(self) -> None:
        """A comment or PI is a node, not an element, and itertext raises on one:
        the count reached them inside a TITLE wrapper and took the conversion out."""
        out, prov = self._convert(
            "<DIVISION><TITLE><?fmx pagebreak?><TI>General provisions</TI>"
            "<!-- editorial note --></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>"
        )
        assert "<heading>General provisions</heading>" in out
        assert "body_discarded_text" not in prov

    def test_a_reference_beside_text_is_still_counted(self) -> None:
        """Text anywhere in the article used to mask this: the check ran over the
        whole element, so a heading was enough to report the reference kept."""
        for inner in (
            "<TI.ART><STI.ART>1</STI.ART><TI>Definitions</TI></TI.ART>"
            "<GRAPHIC FILE='diagram.tif'/>",
            "<TXT>Some text.</TXT><GRAPHIC FILE='diagram.tif'/>",
            "<PARAG><TXT>One.</TXT></PARAG><GRAPHIC FILE='diagram.tif'/>",
        ):
            out, prov = self._convert(f"<ARTICLE IDENTIFIER='001'>{inner}</ARTICLE>")
            assert "diagram.tif" not in out, inner
            assert prov.get("body_discarded_text") == 1, inner

    def test_a_child_trapped_in_the_article_title_wrapper_is_counted(self) -> None:
        """With a nested STI.ART the emitter reads that and TI, so anything else
        the wrapper holds is dropped; exempting the whole wrapper hid it."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'>"
            "<TI.ART><STI.ART>1</STI.ART><TI>Definitions</TI>"
            "<P>A note trapped in the wrapper.</P></TI.ART>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "A note trapped" not in out
        assert prov.get("body_discarded_text") == 1

    def test_a_flattened_title_wrapper_is_not_a_loss(self) -> None:
        """Without a nested STI.ART the whole wrapper is flattened into the number,
        so its children are read after all."""
        _, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><TI.ART>Article 1</TI.ART>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "body_discarded_text" not in prov

    def test_a_wrapper_holding_only_a_comment_is_not_a_loss(self) -> None:
        """`len` counts comment and processing-instruction nodes, so an empty
        wrapper read as content reported a discard that never happened."""
        _, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG>"
            "<GR.SEQ><!-- editorial --></GR.SEQ></ARTICLE>"
        )
        assert "body_discarded_text" not in prov

    def test_a_reference_inside_a_numbered_paragraph_is_counted(self) -> None:
        """`_emit_paragraph` reads NO.PARAG, ALINEA, LIST, TXT and P. It stops
        there, so a loss deeper than this is not counted and the total is a floor."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG><NO.PARAG>1.</NO.PARAG>"
            "<ALINEA><TXT>The rule.</TXT></ALINEA>"
            "<GRAPHIC FILE='inside-parag.tif'/></PARAG></ARTICLE>"
        )
        assert "inside-parag.tif" not in out
        assert prov.get("body_discarded_text") == 1

    def test_what_another_pass_emits_is_not_a_loss(self) -> None:
        """With no ENACTING.TERMS the walk falls back to the whole document, where
        the preamble and the date sit. Both are read elsewhere."""
        provenance: dict = {}
        formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<PREAMBLE><PREAMBLE.INIT>Having regard to the Treaty,</PREAMBLE.INIT></PREAMBLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DOCUMENT>",
            provenance=provenance,
        )
        assert "body_discarded_text" not in provenance

    def test_the_count_reaches_the_run_as_a_validation_issue(self, tmp_path: Path) -> None:
        """A provenance key nothing yields is a counter that does not report: the
        signal list is the only production reader."""
        from codify.pipeline.events import ValidationIssued

        source = tmp_path / "act.xml"
        source.write_text(
            _body(
                "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
                "<GR.SEQ><P>A note between the articles.</P></GR.SEQ>"
            ),
            encoding="utf-8",
        )

        async def go() -> list:
            return [e async for e in ingest_document(source, "eu")]

        issues = [e for e in asyncio.run(go()) if isinstance(e, ValidationIssued)]
        assert any("body_discarded_text" in e.issue for e in issues), issues

    def test_a_signature_block_the_conclusions_emit_is_not_a_loss(self) -> None:
        """Naming FINAL as handled would be a guess: the conclusions builder only
        emits when it finds a signature, so the counter asks it instead."""
        provenance: dict = {}
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "<FINAL><SIGNATURE><PL.DATE><P>Done at the seat, 1 January 2024.</P></PL.DATE>"
            "</SIGNATURE></FINAL></DOCUMENT>",
            provenance=provenance,
        )
        assert "Done at the seat" in out
        assert "body_discarded_text" not in provenance

    def test_a_closer_the_conclusions_drop_is_still_a_loss(self) -> None:
        """The other half of the same question: no signature, nothing emitted, so
        exempting the tag by name would have hidden a real drop."""
        provenance: dict = {}
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "<FINAL><P>An unsigned closer.</P></FINAL></DOCUMENT>",
            provenance=provenance,
        )
        assert "unsigned closer" not in out
        assert provenance.get("body_discarded_text") == 1

    @pytest.mark.parametrize("tag", ["ANNEX", "APPENDIX"])
    def test_neither_kind_of_attachment_is_a_loss(self, tag: str) -> None:
        """The attachments builder maps both, and naming one of them here left the
        other reported as discarded while its prose was in the output."""
        provenance: dict = {}
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            f"<{tag}><TITLE><TI><P>ANNEX I</P></TI></TITLE>"
            f"<P>Attachment prose.</P></{tag}></DOCUMENT>",
            provenance=provenance,
        )
        assert "Attachment prose." in out
        assert "body_discarded_text" not in provenance

    @pytest.mark.parametrize(
        "inner",
        [
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<GRAPHIC FILE='ref.tif'/>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>",
            "<GRAPHIC FILE='ref.tif'/>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>",
        ],
        ids=["under a division", "at the top level"],
    )
    def test_a_reference_is_counted_wherever_it_sits(self, inner: str) -> None:
        """One predicate for the whole walk. Two branches asked only whether the
        element had text, so a file reference was a loss in an article and not in
        a chapter."""
        out, prov = self._convert(inner)
        assert "ref.tif" not in out
        assert prov.get("body_discarded_text") == 1

    def test_text_loose_in_a_paragraph_is_counted(self) -> None:
        """`_emit_paragraph` reads its children, so a lead-in sitting directly in
        the PARAG is read by nothing."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG>Loose lead-in inside the paragraph."
            "<ALINEA><TXT>The rule.</TXT></ALINEA></PARAG></ARTICLE>"
        )
        assert "Loose lead-in" not in out
        assert "The rule." in out
        assert prov.get("body_discarded_text") == 1

    def test_a_loss_below_any_nesting_depth_is_counted(self) -> None:
        """The walk being mirrored has no depth cutoff, so a cutoff here would
        report a document it converts fine as carrying no loss at all."""
        inner = (
            "<GRAPHIC FILE='deep.tif'/>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        for _ in range(25):
            inner = f"<DIVISION TYPE='CHAPTER'><TITLE><TI>C</TI></TITLE>{inner}</DIVISION>"
        out, prov = self._convert(inner)
        assert "deep.tif" not in out
        assert prov.get("body_discarded_text") == 1

    def test_the_act_title_is_not_a_loss(self) -> None:
        provenance: dict = {}
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<TITLE><TI>An Act of some kind</TI></TITLE>"
            "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG>"
            "</ARTICLE></ENACTING.TERMS></DOCUMENT>",
            provenance=provenance,
        )
        assert "An Act of some kind" in out
        assert "body_discarded_text" not in provenance

    def test_a_title_one_level_down_is_counted(self) -> None:
        """The preface reads TITLE as a direct child of the document, so the same
        tag inside the enacting terms is read by nothing. Exempting it by name
        covered both."""
        out, prov = self._convert(
            "<TITLE><TI>A heading nothing reads</TI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "A heading nothing reads" not in out
        assert prov.get("body_discarded_text") == 1

    def test_a_flattened_wrapper_carries_its_children(self) -> None:
        """The emitter branches on the number having text, not on the element
        being there: with an empty STI.ART it flattens the whole wrapper, so a
        child beside it is read after all."""
        out, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><TI.ART><STI.ART></STI.ART><P>Trapped note.</P></TI.ART>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "Trapped note." in out
        assert "body_discarded_text" not in prov

    def test_the_sibling_subtitle_is_read_only_when_the_wrapper_gives_no_heading(
        self,
    ) -> None:
        """The emitter falls back to the sibling only when the wrapper produced no
        heading, so exempting it unconditionally hides it when the wrapper did."""
        dropped, prov_dropped = self._convert(
            "<ARTICLE IDENTIFIER='001'><TI.ART><STI.ART></STI.ART><TI>Definitions</TI></TI.ART>"
            "<STI.ART>A subtitle beside the wrapper.</STI.ART>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "A subtitle beside" not in dropped
        assert prov_dropped.get("body_discarded_text") == 1

        kept, prov_kept = self._convert(
            "<ARTICLE IDENTIFIER='001'><TI.ART><STI.ART></STI.ART></TI.ART>"
            "<STI.ART>A subtitle beside the wrapper.</STI.ART>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "A subtitle beside" in kept
        assert "body_discarded_text" not in prov_kept

    def test_what_another_pass_emits_is_not_a_loss_at_any_level(self) -> None:
        """The exemption was applied by the outer loop only, so the same element
        one level in was emitted by its builder and counted as lost anyway."""
        for inner in (
            "<ARTICLE IDENTIFIER='001'><TOC><TITLE><TI>Contents</TI></TITLE></TOC>"
            "<PARAG><TXT>One.</TXT></PARAG></ARTICLE>",
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE><P>Attachment prose.</P></ANNEX>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>",
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT>"
            "<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE><P>Attachment prose.</P></ANNEX>"
            "</PARAG></ARTICLE>",
        ):
            out, prov = self._convert(inner)
            assert "body_discarded_text" not in prov, inner

    def test_a_contents_listing_is_not_a_loss(self) -> None:
        """A TOC is navigation the walk drops on purpose; counting it would make
        every act with one report lossy."""
        _, prov = self._convert(
            "<TOC><TITLE><TI>Contents</TI></TITLE></TOC>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
        )
        assert "body_discarded_text" not in prov

    def test_a_heading_is_not_a_loss(self) -> None:
        _, prov = self._convert(
            "<DIVISION TYPE='CHAPTER'><TITLE><TI>CHAPTER 1</TI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE></DIVISION>"
        )
        assert "body_discarded_text" not in prov

    def test_a_clean_act_reports_no_loss(self) -> None:
        _, prov = self._convert(
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "<ARTICLE IDENTIFIER='002'><PARAG><TXT>Two.</TXT></PARAG></ARTICLE>"
        )
        assert "body_discarded_text" not in prov


class TestATitledAnnexGroupGetsAContainer:
    """A transparent wrapper is walked through, so its heading and its item label
    reached nothing. A title justifies a container; without one the wrapper stays
    transparent, because an ancestor with no heading moves eIds and says nothing."""

    @staticmethod
    def _annex(inner: str) -> str:
        return (
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "</ENACTING.TERMS>"
            f"<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE>{inner}</ANNEX></DOCUMENT>"
        )

    _TITLED = (
        "<GR.SEQ><NO.GR.SEQ>(a)</NO.GR.SEQ>"
        "<TITLE><TI>Part A</TI><STI>General provisions</STI></TITLE>"
        "<P>First item.</P></GR.SEQ><P>Outside the group.</P>"
    )
    # What the corpus actually holds: every title splits its number from its text
    # into sibling elements, and none is the plain string the other fixtures use.
    _STRUCTURED = (
        "<GR.SEQ><TITLE><TI><NP><NO.P>I.</NO.P>"
        "<TXT>Information regarding X</TXT></NP></TI></TITLE>"
        "<P>First item.</P></GR.SEQ>"
    )
    _UNTITLED = "<GR.SEQ><P>First item.</P></GR.SEQ><P>Outside the group.</P>"

    def test_the_heading_and_the_label_survive(self) -> None:
        out = formex_to_akn4eu(self._annex(self._TITLED))
        assert '<hcontainer name="group" eId="att_1__group_1">' in out
        assert "<num>(a)</num>" in out
        assert "<heading>Part A</heading>" in out
        assert "<subheading>General provisions</subheading>" in out

    def test_the_recovered_title_reaches_the_provision_row(self) -> None:
        """Recovering a heading nobody can search would be half the job: the row's
        text is what search and the embeddings read."""
        from codify.akn import parse_akn

        doc = parse_akn(formex_to_akn4eu(self._annex(self._TITLED)).encode())
        rows: list[str] = []

        def walk(node: object) -> None:
            eid = getattr(node, "akn_eid", None)
            if eid and "group" in eid:
                rows.append(getattr(node, "text", ""))
            for child in getattr(node, "children", None) or []:
                walk(child)

        for att in doc.attachments:
            for top in getattr(att, "children", None) or [att]:
                walk(top)
        assert rows, "the group produced no row"
        assert "Part A" in rows[0] and "(a)" in rows[0] and "General provisions" in rows[0]

    def test_the_group_is_an_ancestor_of_its_contents(self) -> None:
        """The point of the container: the eId lengthens visibly rather than the
        annex renumbering underneath a citation that still resolves."""
        out = formex_to_akn4eu(self._annex(self._TITLED))
        assert 'eId="att_1__group_1__paragraph_1"' in out
        assert "First item." in out

    def test_a_number_the_group_took_is_never_reused_outside_it(self) -> None:
        """The failure option a was rejected for. The group's contents leave the
        annex sequence, so without sharing the count the next outside paragraph
        slides into the vacated number and an old eId survives holding new text."""
        out = formex_to_akn4eu(self._annex(self._TITLED))
        assert 'eId="att_1__group_1__paragraph_1"' in out
        assert 'eId="att_1__paragraph_1"' not in out
        second = out.index('eId="att_1__paragraph_2"')
        assert "Outside the group." in out[second : second + 400]

    def test_no_annex_eid_survives_holding_different_text(self) -> None:
        """Stated as the invariant rather than as one case: convert the same annex
        with and without the wrapper and compare every eId that exists in both."""
        import re

        from codify.akn._schema import parse_xml

        def paragraphs(inner: str) -> dict[str, str]:
            root = parse_xml(formex_to_akn4eu(self._annex(inner)).encode())
            out: dict[str, str] = {}
            for el in root.iter("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}hcontainer"):
                eid = el.get("eId")
                if eid and el.get("name") == "paragraph":
                    out[eid] = " ".join("".join(el.itertext()).split())
            return out

        before = paragraphs("<P>First item.</P><P>Outside the group.</P>")
        after = paragraphs(
            "<GR.SEQ><TITLE><TI>Part A</TI></TITLE><P>First item.</P></GR.SEQ>"
            "<P>Outside the group.</P>"
        )
        assert before and after
        moved = {
            k: (before[k], after[k]) for k in before.keys() & after.keys() if before[k] != after[k]
        }
        assert not moved, moved
        assert re.search(r"att_\d+__group_\d+__paragraph_\d+", " ".join(after))

    def test_an_untitled_wrapper_is_still_transparent(self) -> None:
        """A container with no heading would add an ancestor that says nothing and
        move the contents' eIds for no reader's benefit."""
        out = formex_to_akn4eu(self._annex(self._UNTITLED))
        assert 'name="group"' not in out
        assert 'eId="att_1__paragraph_1"' in out
        assert 'eId="att_1__paragraph_2"' in out

    def test_a_nested_group_composes_and_never_reuses_a_number(self) -> None:
        """A titled wrapper inside a titled wrapper. The eId composes, and the
        annex's counters are shared throughout so no number is handed out twice."""
        out = formex_to_akn4eu(
            self._annex(
                "<GR.SEQ><TITLE><TI>Part A</TI></TITLE>"
                "<GR.SEQ><TITLE><TI>Section 1</TI></TITLE><P>Inner item.</P></GR.SEQ>"
                "</GR.SEQ><P>After both.</P>"
            )
        )
        import re as _re

        eids = [e for e in _re.findall(r'eId="([^"]+)"', out) if "att_" in e]
        assert len(eids) == len(set(eids)), eids
        assert "att_1__group_1__group_2__paragraph_1" in eids
        assert "<heading>Section 1</heading>" in out

    def test_a_structured_title_keeps_its_separator(self) -> None:
        """The shape every title in the corpus has: the number and the text are
        sibling elements, and concatenating them gives "I.Information"."""
        out = formex_to_akn4eu(self._annex(self._STRUCTURED))
        assert "<heading>I. Information regarding X</heading>" in out
        assert "I.Information" not in out

    def test_a_group_is_scaffolding_not_a_unit_of_law(self) -> None:
        """It carries a num, and everything else with a num in the corpus is a
        provision, so the preservation gate would count the wrapper as law."""
        from codify.akn._schema import parse_xml
        from codify.akn.vocabulary import basic_unit_numbers

        root = parse_xml(formex_to_akn4eu(self._annex(self._TITLED)).encode())
        assert not [u for u in basic_unit_numbers(root) if u.startswith("group")]

    def test_a_titled_group_validates(self, trade_secrets_xml: Path) -> None:
        from codify.akn._schema import validate_akn

        src = trade_secrets_xml.read_text()
        annex = "<ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE>" + self._TITLED + "</ANNEX>"
        out = formex_to_akn4eu(
            src.replace("</ENACTING.TERMS>", "</ENACTING.TERMS>" + annex, 1),
            frbr_work_uri="/akn/eu/act/dir/2016/943",
        )
        assert 'name="group"' in out
        assert not validate_akn(out.encode())


class TestATitleIsJoinedNotConcatenated:
    """Both heading paths share one join. Concatenating throughout welds a number
    to its text; separating throughout parts an inline element from the
    punctuation after it. Shapes and counts are from the stored corpus, where no
    title is the plain string the older fixtures used."""

    _WELDED = "<NP><NO.P>I.</NO.P><TXT>Information regarding X</TXT></NP>"
    _INLINE = "<NP><TXT>Part <HT TYPE='ITALIC'>A</HT>.</TXT></NP>"
    _BOTH = "<NP><NO.P>I.</NO.P><TXT>Rules on <HT TYPE='ITALIC'>ex ante</HT> review.</TXT></NP>"

    @staticmethod
    def _division_headings(ti: str) -> list[str]:
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            f"<ENACTING.TERMS><DIVISION TYPE='CHAPTER'><TITLE><TI>{ti}</TI></TITLE>"
            "<ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "</DIVISION></ENACTING.TERMS></DOCUMENT>"
        )
        return re.findall(r"<heading>([^<]*)</heading>", out)

    @staticmethod
    def _group_headings(ti: str) -> list[str]:
        out = formex_to_akn4eu(
            "<DOCUMENT><DATE.OF.DOCUMENT ISO='2024-01-01'>1 Jan 2024</DATE.OF.DOCUMENT>"
            "<ENACTING.TERMS><ARTICLE IDENTIFIER='001'><PARAG><TXT>One.</TXT></PARAG></ARTICLE>"
            "</ENACTING.TERMS><ANNEX><TITLE><TI><P>ANNEX I</P></TI></TITLE>"
            f"<GR.SEQ><TITLE><TI>{ti}</TI></TITLE><P>Body.</P></GR.SEQ></ANNEX></DOCUMENT>"
        )
        return [h for h in re.findall(r"<heading>([^<]*)</heading>", out) if "ANNEX" not in h]

    def test_a_number_and_its_text_are_separated(self) -> None:
        """Sibling NO.P and TXT: 15,737 of 15,737 titles in the corpus split this
        way, so concatenating welded every heading."""
        assert self._division_headings(self._WELDED) == ["I. Information regarding X"]
        assert self._group_headings(self._WELDED) == ["I. Information regarding X"]

    def test_inline_markup_keeps_its_tail(self) -> None:
        """633 inline elements in the corpus are followed by text, so separating
        every fragment left the punctuation adrift."""
        assert self._division_headings(self._INLINE) == ["Part A."]
        assert self._group_headings(self._INLINE) == ["Part A."]

    def test_an_inline_tag_other_than_ht_keeps_its_tail(self) -> None:
        """Anything not a block part is inline by design, so the rule is exercised
        on tags the corpus holds beside HT: QUOT.START at 437, NOTE at 234, DATE
        at 28, all followed by text."""
        for markup in (
            "<TXT>Adopted on <DATE ISO='20240101'>1 January 2024</DATE>.</TXT>",
            "<TXT>The <QUOT.START/>register<QUOT.END/>.</TXT>",
        ):
            headings = self._division_headings(f"<NP>{markup}</NP>")
            assert headings, markup
            assert " ." not in headings[0], headings
            assert headings[0].endswith("."), headings

    def test_a_block_tag_inside_an_inline_one_does_not_split(self) -> None:
        """A NOTE wraps its body in a P, so treating block parts at any depth split
        inside the note and stranded the stop after it as "See footnote .". Block
        parts are the ones on the title's own spine; below an inline element
        everything is inline, whatever it is named. The result matches `main` for
        this shape: the join only differs where a real block boundary sits."""
        headings = self._division_headings("<NP><TXT>See<NOTE><P>footnote</P></NOTE>.</TXT></NP>")
        assert headings == ["Seefootnote."], headings
        assert " ." not in headings[0]

    def test_both_shapes_in_one_title(self) -> None:
        assert self._division_headings(self._BOTH) == ["I. Rules on ex ante review."]
        assert self._group_headings(self._BOTH) == ["I. Rules on ex ante review."]


def test_a_real_act_with_a_welded_title_is_repaired() -> None:
    """The repo's other EU fixtures cannot detect this class: their headings are
    identical either way. This one is taken from the corpus because its stored
    output carries the defect, so the comparison has a case that must differ."""
    fixture = FIXTURES_DIR / "eu" / "2012-687" / "eng.xml"
    if not fixture.exists():
        pytest.skip("fixture missing: run apps/api/scripts/run_ingest.py --skip-ingest")
    src = fixture.read_text()
    out = formex_to_akn4eu(src, frbr_work_uri="/akn/eu/act/dec/2012/687")
    headings = re.findall(r"<heading>([^<]*)</heading>", out)
    assert "B. Entities" in headings
    assert "B.Entities" not in headings


def test_the_lane_validates_as_publisher_structured() -> None:
    """The act's own date is the publisher's, so plausibility is not graded."""
    from codify.akn import AKN_NS

    xml = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><meta><identification source="#src"><FRBRWork>'
        '<FRBRuri value="/akn/eu/act/1766/4"/><FRBRdate date="1766-03-18" name="enacted"/>'
        '</FRBRWork></identification></meta><body><article eId="art_1"><num>1</num>'
        "<content><p>t</p></content></article></body></act></akomaNtoso>"
    )
    assert not [i for i in eu_directive._validate_native(xml) if i["check"].startswith("identity_")]
