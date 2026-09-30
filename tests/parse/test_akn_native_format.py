"""Native-AKN ingest format: normalisation + dispatch routing."""

from __future__ import annotations

import uuid
from pathlib import Path

from lxml import etree

from codify.akn.io import parse_akn
from codify.pipeline import ingest_document
from codify.pipeline.events import Complete
from codify.pipeline.formats.akn_native import normalise_native_akn

_UK_AKN = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
 <act>
  <meta><identification source="#src"><FRBRWork>
    <FRBRthis value="http://www.legislation.gov.uk/id/ukpga/2018/12"/>
    <FRBRuri value="http://www.legislation.gov.uk/id/ukpga/2018/12"/>
    <FRBRdate date="2018-05-23" name="enacted"/>
  </FRBRWork>
  <FRBRExpression>
    <FRBRthis value="http://www.legislation.gov.uk/ukpga/2018/12/2026-06-19"/>
    <FRBRuri value="http://www.legislation.gov.uk/ukpga/2018/12/2026-06-19"/>
    <FRBRdate date="2026-06-19" name="validFrom"/>
    <FRBRlanguage language="eng"/>
  </FRBRExpression></identification></meta>
  <body>
    <section eId="section-1"><num>1</num><heading>Overview</heading>
      <content><p>This Act makes provision.</p></content>
    </section>
    <hcontainer name="definition">
      <content><p>An unnumbered definition container.</p></content>
    </hcontainer>
  </body>
 </act>
</akomaNtoso>"""


def test_normalise_rewrites_frbr_and_fills_eids() -> None:
    doc = parse_akn(_UK_AKN)
    doc, synth = normalise_native_akn(doc)
    assert doc.frbr_work_uri == "/akn/gb/act/ukpga/2018/12"
    assert doc.frbr_expression_uri.startswith("/akn/gb/act/ukpga/2018/12/eng@")
    assert synth >= 1  # the unnumbered hcontainer
    eids = {el.akn_eid for el in doc.body}
    assert "section-1" in eids  # published eIds untouched


def test_normalise_leaves_unknown_publisher_frbr() -> None:
    doc = parse_akn(_UK_AKN.replace("http://www.legislation.gov.uk/id/ukpga/2018/12", "urn:x"))
    work_before = doc.frbr_work_uri
    doc, _ = normalise_native_akn(doc)
    assert doc.frbr_work_uri == work_before


async def test_dispatch_routes_non_eu_xml_to_akn_native(tmp_path: Path) -> None:
    src = tmp_path / "ukpga-2018-12.akn"
    src.write_text(_UK_AKN, encoding="utf-8")

    events = [e async for e in ingest_document(src, "gb")]
    complete = [e for e in events if isinstance(e, Complete)]
    assert len(complete) == 1
    assert complete[0].document.frbr_work_uri == "/akn/gb/act/ukpga/2018/12"
    # The stored XML names the same work as the row.
    stored = etree.fromstring(complete[0].akn_xml.encode())
    work = stored.find(".//{*}FRBRWork")
    expression = stored.find(".//{*}FRBRExpression")
    assert work is not None and expression is not None
    assert {el.get("value") for el in work.iter("{*}FRBRthis", "{*}FRBRuri")} == {
        "/akn/gb/act/ukpga/2018/12"
    }
    assert {el.get("value") for el in expression.iter("{*}FRBRthis", "{*}FRBRuri")} == {
        complete[0].document.frbr_expression_uri
    }


def test_looks_like_eu_is_the_one_routing_predicate() -> None:
    # The shared predicate the ingest workflow now reuses: EU hosts (incl.
    # publications.europa.eu, which the old ingest copy dropped), case-insensitive
    # suffix, and no SSRF via a non-EU host.
    from codify.pipeline.formats import looks_like_eu

    assert looks_like_eu("https://publications.europa.eu/x.xml") is True
    assert looks_like_eu("https://eur-lex.europa.eu/legal-content/x") is True
    assert looks_like_eu(Path("foo.XML")) is True  # case-insensitive
    assert looks_like_eu("https://evil.example.com/x.xml") is False  # non-EU host
    assert looks_like_eu(Path("foo.pdf")) is False
    assert looks_like_eu("https://eur-lex.europa.eu:443/x") is True  # explicit port
    assert looks_like_eu("https://eur-lex.europa.eu.evil.com/x") is False  # subdomain spoof
    assert looks_like_eu("https://user@evil.com/x") is False  # userinfo spoof


def test_normalise_dedupes_repeated_eids() -> None:
    dup = _UK_AKN.replace(
        '<hcontainer name="definition">',
        '<section eId="section-1"><num>1A</num><content><p>Dup.</p></content></section>'
        '<hcontainer name="definition">',
    )
    doc = parse_akn(dup)
    doc, _ = normalise_native_akn(doc)
    eids = [el.akn_eid for el in doc.body]
    assert "section-1" in eids and "section-1-dup1" in eids


def test_targetless_refs_produce_no_cross_reference_rows() -> None:
    from codify.akn.references import Citation
    from codify.storage.mappers import _ref_to_row

    ref = Citation(target_uri=None, start_offset=0, end_offset=3, text_snippet="see")
    assert _ref_to_row(ref, uuid.uuid4()) is None


async def test_amendment_context_types_external_refs() -> None:
    """Pass 5: refs in amendment prose gain class="amendment-<op>" and parse
    as AmendmentReference -> mod_textual edges; plain citations stay untyped."""
    from codify.pipeline.enrich.inline_markup import emit_inline_markup
    from codify.storage.mappers import _ref_to_row

    akn = _UK_AKN.replace(
        "<content><p>This Act makes provision.</p></content>",
        '<content><p>In <ref href="/akn/gb/act/ukpga/2000/36">the 2000 Act</ref>, '
        "section 9 is repealed.</p>"
        '<p>Subject to <ref href="/akn/gb/act/1998/29">the 1998 Act</ref>.</p></content>',
    )
    out = await emit_inline_markup(akn, "gb", "act", None)
    doc = parse_akn(out)

    refs = []

    def _collect(el: object) -> None:
        refs.extend(getattr(el, "references", None) or [])
        for child in getattr(el, "children", []) or []:
            _collect(child)

    for top in doc.body:
        _collect(top)
    kinds = {r.text_snippet: r.kind for r in refs if "Act" in r.text_snippet}
    assert kinds["the 2000 Act"] == "amendment_reference"
    assert kinds["the 1998 Act"] == "citation"
    amendment = next(r for r in refs if r.kind == "amendment_reference")
    assert amendment.operation == "delete"
    row = _ref_to_row(amendment, uuid.uuid4())
    assert row is not None and row.edge_class == "mod_textual"


async def test_other_act_qualifiers_do_not_self_link() -> None:
    """ "section N of the X Act" and amendment-schedule bare refs must not
    resolve to this document's section N; "of this Act" still does."""
    akn = _UK_AKN.replace(
        "<content><p>This Act makes provision.</p></content>",
        "<content>"
        "<p>The repeal of section 1 of the 1998 Act does not affect it.</p>"
        "<p>See section 36 of the Opticians Act 1989 for meaning.</p>"
        "<p>Section 1 (information databases) of the Data Protection Act 1998 "
        "is amended as follows.</p>"
        "<p>Subject to section 1 of this Act.</p>"
        "</content>",
    )
    from codify.pipeline.enrich.inline_markup import emit_inline_markup

    out = await emit_inline_markup(akn, "gb", "act", None)
    root = etree.fromstring(out.encode())
    hrefs = [
        r.get("href")
        for r in root.iter("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}ref")
        if (r.get("href") or "").startswith("#")
    ]
    assert hrefs == ["#section-1"]  # only the "of this Act" form self-links


def test_tail_redirect_classification() -> None:
    from codify.pipeline.enrich.inline_markup import _tail_redirects_elsewhere

    redirects = [
        " of the 1998 Act",
        " of that Act",
        " of the said Regulations",
        " of the Education (Scotland) Act 1980",
        " of Schedule 1 to the 1998 Act",
        " to 42 of the Education (Scotland) Act 1980",
        " under the Proceeds of Crime Act 2002",
        " (information databases) is amended as follows.",
        " is repealed.",
    ]
    self_links = [
        " of this Act",
        " of these Regulations",
        " of Schedule 2 to this Act",
        " of Part 7 of this Act",
        " of Part 2",
        " of Schedule 1",
        " applies for the purposes of this Part",
    ]
    for tail in redirects:
        assert _tail_redirects_elsewhere(tail), tail
    for tail in self_links:
        assert not _tail_redirects_elsewhere(tail), tail


async def test_scoped_amendments_bind_to_opener_act() -> None:
    """Pass 6: op-paragraphs after "The X Act is amended as follows" gain
    typed refs to the opener's act; the scope ends with its container."""
    # Realistic shape: opener and ops in sibling numbered <paragraph>s
    # under a schedule-like container.
    akn = _UK_AKN.replace(
        "<content><p>This Act makes provision.</p></content>",
        '<hcontainer name="schedule" eId="schedule-1">'
        '<paragraph eId="schedule-1-paragraph-1"><content>'
        '<p>The <ref href="/akn/gb/act/ukpga/2000/36">Freedom of Information Act 2000</ref> '
        "is amended as follows.</p></content></paragraph>"
        '<paragraph eId="schedule-1-paragraph-2"><content>'
        "<p>In section 2, omit subsection (3).</p></content></paragraph>"
        '<paragraph eId="schedule-1-paragraph-3"><content>'
        "<p>After section 5 insert new text.</p></content></paragraph>"
        '<paragraph eId="schedule-1-paragraph-4"><content>'
        "<p>Nothing operative here.</p></content></paragraph>"
        "</hcontainer>",
    )
    from codify.pipeline.enrich.inline_markup import emit_inline_markup

    out = await emit_inline_markup(akn, "gb", "act", None)
    root = etree.fromstring(out.encode())
    typed = [
        # The operation token, not the whole class: a ref this pipeline
        # minted carries its provenance marker in the same attribute.
        (
            r.get("href"),
            next(t for t in (r.get("class") or "").split() if t.startswith("amendment-")),
        )
        for r in root.iter("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}ref")
        if "amendment-" in (r.get("class") or "")
    ]
    assert ("/akn/gb/act/ukpga/2000/36", "amendment-replace") in typed  # opener
    assert ("/akn/gb/act/ukpga/2000/36", "amendment-delete") in typed  # omit para
    assert ("/akn/gb/act/ukpga/2000/36", "amendment-insert") in typed  # insert para
    assert len(typed) == 3  # the inert paragraph gains nothing


async def test_op_cue_union_lets_pass6_classify_deleted_and_added() -> None:
    # Drift fix: pass 6 used to lack the `delet`/`add(ed)` cues that pass 5 had,
    # so "is deleted"/"added" op-paragraphs got no typed ref. The shared table
    # now classifies them the same way both passes would.
    akn = _UK_AKN.replace(
        "<content><p>This Act makes provision.</p></content>",
        '<hcontainer name="schedule" eId="schedule-1">'
        '<paragraph eId="schedule-1-paragraph-1"><content>'
        '<p>The <ref href="/akn/gb/act/ukpga/2000/36">Freedom of Information Act 2000</ref> '
        "is amended as follows.</p></content></paragraph>"
        '<paragraph eId="schedule-1-paragraph-2"><content>'
        "<p>Section 3 is deleted.</p></content></paragraph>"
        '<paragraph eId="schedule-1-paragraph-3"><content>'
        "<p>In section 4, new text is added at the end.</p></content></paragraph>"
        "</hcontainer>",
    )
    from codify.pipeline.enrich.inline_markup import emit_inline_markup

    out = await emit_inline_markup(akn, "gb", "act", None)
    root = etree.fromstring(out.encode())
    typed = [
        # The operation token, not the whole class: a ref this pipeline
        # minted carries its provenance marker in the same attribute.
        (
            r.get("href"),
            next(t for t in (r.get("class") or "").split() if t.startswith("amendment-")),
        )
        for r in root.iter("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}ref")
        if "amendment-" in (r.get("class") or "")
    ]
    assert ("/akn/gb/act/ukpga/2000/36", "amendment-delete") in typed  # "is deleted"
    assert ("/akn/gb/act/ukpga/2000/36", "amendment-insert") in typed  # "added"


async def test_short_title_index_links_named_acts() -> None:
    """Pass 4a2: "the X Act YYYY" links via the jurisdiction title index;
    unknown titles stay plain text."""
    from codify.pipeline.enrich.inline_markup import emit_inline_markup

    akn = _UK_AKN.replace(
        "<content><p>This Act makes provision.</p></content>",
        "<content><p>See the Freedom of Information Act 2000 and "
        "the Completely Fictional Act 1999.</p></content>",
    )
    out = await emit_inline_markup(akn, "gb", "act", None)
    root = etree.fromstring(out.encode())
    hrefs = [
        r.get("href")
        for r in root.iter("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}ref")
        if not (r.get("href") or "").startswith("#")
    ]
    assert "/akn/gb/act/ukpga/2000/36" in hrefs
    assert len(hrefs) == 1


def test_title_regex_nested_and_comma_forms() -> None:
    from codify.pipeline.enrich.inline_markup import _TITLE_REF

    nested = (
        "Under the Financial Services and Markets Act 2000 "
        "(Regulated Activities) Order 2001 it applies."
    )
    matches = [m.group(1) for m in _TITLE_REF.finditer(nested)]
    assert "Financial Services and Markets Act 2000" not in matches

    comma = "See the Anti-terrorism, Crime and Security Act 2001 for powers."
    matches = [m.group(1) for m in _TITLE_REF.finditer(comma)]
    assert matches == ["Anti-terrorism, Crime and Security Act 2001"]


def test_cyrillic_title_regex_extracts_quoted_names() -> None:
    from codify.pipeline.enrich.inline_markup import _TITLE_REF_CYR

    text = (
        "відповідно до Закону України «Про запобігання корупції» та "
        'Закон України "Про лобіювання" передбачає'
    )
    got = [m.group(1) for m in _TITLE_REF_CYR.finditer(text)]
    assert got == ["Про запобігання корупції", "Про лобіювання"]


def test_canonicalisation_leaves_component_identifications_alone() -> None:
    """A schedule's own identification names the component, not the parent."""
    from codify.pipeline.formats.akn_native import canonicalise_identification

    xml = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act>
  <meta><identification source="#src">
    <FRBRWork><FRBRthis value="http://x/id/xpa/2018/12/!main"/><FRBRuri value="http://x/id/xpa/2018/12"/></FRBRWork>
    <FRBRExpression><FRBRthis value="http://x/xpa/2018/12/2026-06-19"/><FRBRuri value="http://x/xpa/2018/12/2026-06-19"/></FRBRExpression>
  </identification></meta>
  <body><section eId="section-1"><content><p>x</p></content></section></body>
  <attachments><attachment eId="att_1"><doc name="schedule"><meta><identification source="#src">
    <FRBRWork><FRBRthis value="http://x/id/xpa/2018/12/!schedule_1"/><FRBRuri value="http://x/id/xpa/2018/12"/></FRBRWork>
  </identification></meta><mainBody><p>s</p></mainBody></doc></attachment></attachments>
</act></akomaNtoso>"""
    out = canonicalise_identification(
        xml, "/akn/xq/act/xpa/2018/12", "/akn/xq/act/xpa/2018/12/eng@2026-06-19"
    )
    root = etree.fromstring(out.encode())
    top = root.find("./{*}act/{*}meta/{*}identification")
    assert top.find("{*}FRBRWork/{*}FRBRuri").get("value") == "/akn/xq/act/xpa/2018/12"
    assert top.find("{*}FRBRWork/{*}FRBRthis").get("value") == "/akn/xq/act/xpa/2018/12/!main"
    assert (
        top.find("{*}FRBRExpression/{*}FRBRuri").get("value")
        == "/akn/xq/act/xpa/2018/12/eng@2026-06-19"
    )
    nested = root.find(".//{*}attachment//{*}identification/{*}FRBRWork")
    assert nested.find("{*}FRBRthis").get("value") == "http://x/id/xpa/2018/12/!schedule_1"
    assert nested.find("{*}FRBRuri").get("value") == "http://x/id/xpa/2018/12"


_OLD_NATIVE = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
 <act>
  <meta><identification source="#src"><FRBRWork>
    <FRBRthis value="/akn/xz/act/1766/4/!main"/>
    <FRBRuri value="/akn/xz/act/1766/4"/>
    <FRBRdate date="1766-03-18" name="enacted"/>
  </FRBRWork>
  <FRBRExpression>
    <FRBRthis value="/akn/xz/act/1766/4/eng@1766-03-18/!main"/>
    <FRBRuri value="/akn/xz/act/1766/4/eng@1766-03-18"/>
    <FRBRdate date="1766-03-18" name="enacted"/>
    <FRBRlanguage language="eng"/>
  </FRBRExpression></identification></meta>
  <body>
    <section eId="section-3"><num>3</num><content><p>Three.</p></content></section>
    <section eId="section-4"><num>4</num><content><p>Four.</p></content></section>
  </body>
 </act>
</akomaNtoso>"""


async def test_a_publisher_s_old_act_does_not_grade_as_a_misparse(tmp_path: Path) -> None:
    """A statute book's own markup holds acts from before 1800 and excerpts that
    start mid-sequence; neither is a reading error to grade against."""
    from codify.pipeline.enrich.validator import validate_akn
    from codify.pipeline.events import ValidationIssued
    from codify.pipeline.formats.akn_native import ingest

    src = tmp_path / "old.akn"
    src.write_text(_OLD_NATIVE, encoding="utf-8")
    checks = [
        e.issue.get("check") async for e in ingest(src, "xz") if isinstance(e, ValidationIssued)
    ]
    assert not [c for c in checks if c and c.startswith(("identity_", "number_gap"))], checks
    # The same document read without its provenance is still flagged.
    assert "identity_year_implausible" in {i["check"] for i in validate_akn(_OLD_NATIVE)}
