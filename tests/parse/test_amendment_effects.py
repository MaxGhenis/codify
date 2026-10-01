"""Parser invariants for AKN `<meta><analysis>` and `<meta><lifecycle>`.

`<textualMod>` metadata carries the per-modification detail across five
categories (textual, meaning, scope, force, efficacy) with a 25-value
action enum. `<eventRef>` under `<lifecycle>` carries the chronological
event journal (generation, amendment, repeal)."""

from __future__ import annotations

from textwrap import dedent

from codify.akn._parser import parse_document
from codify.akn.analysis import category_for


def _make_akn(analysis_xml: str = "", lifecycle_xml: str = "") -> str:
    return dedent(
        f"""<?xml version="1.0" encoding="utf-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <meta>
      <identification source="#tester">
        <FRBRWork>
          <FRBRuri value="/akn/xx/act/2020/1"/>
          <FRBRthis value="/akn/xx/act/2020/1/main"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRuri value="/akn/xx/act/2020/1/eng@2020-01-01"/>
          <FRBRlanguage language="eng"/>
          <FRBRdate date="2020-01-01" name="Original"/>
        </FRBRExpression>
      </identification>
      {analysis_xml}
      {lifecycle_xml}
    </meta>
    <body>
      <section eId="sec_1">
        <content><p>Body text.</p></content>
      </section>
    </body>
  </act>
</akomaNtoso>
"""
    )


def test_parses_textual_mods_across_categories() -> None:
    analysis = dedent(
        """<analysis source="#tester">
        <passiveModifications>
          <textualMod type="substitution" source="#sec_1__para_1" period="#p1"
                      authority="/akn/xx/act/2023/5">
            <destination href="/akn/xx/act/2020/1/main#sec_1__para_1" wId="sec_1__para_1"/>
            <old><quotedText>The Minister</quotedText></old>
            <new><quotedText>The Secretary</quotedText></new>
          </textualMod>
          <textualMod type="entryIntoForce" source="#sec_1"
                      authority="/akn/xx/act/2020/1">
            <destination href="/akn/xx/act/2020/1/main#sec_1" wId="sec_1"/>
          </textualMod>
          <textualMod type="extensionOfScope" source="#sec_1"
                      authority="/akn/xx/act/2021/2">
            <destination href="/akn/xx/act/2020/1/main#sec_1" wId="sec_1"/>
            <new><quotedText>and Wales</quotedText></new>
          </textualMod>
        </passiveModifications>
      </analysis>"""
    )
    doc = parse_document(_make_akn(analysis_xml=analysis))

    assert len(doc.textual_mods) == 3
    kinds = {(m.akn_category, m.akn_action) for m in doc.textual_mods}
    assert ("textual", "substitution") in kinds
    assert ("force", "entryIntoForce") in kinds
    assert ("scope", "extensionOfScope") in kinds

    sub = next(m for m in doc.textual_mods if m.akn_action == "substitution")
    assert sub.source_akn_wid == "sec_1__para_1"
    assert sub.target_frbr_uri == "/akn/xx/act/2020/1/main#sec_1__para_1"
    assert sub.target_akn_wid == "sec_1__para_1"
    assert sub.authority_uri == "/akn/xx/act/2023/5"
    assert sub.quoted is not None
    assert sub.quoted.old == ["The Minister"]
    assert sub.quoted.new == ["The Secretary"]


def test_parses_multi_block_quoted_content() -> None:
    analysis = dedent(
        """<analysis source="#tester">
        <passiveModifications>
          <textualMod type="substitution" source="#sec_1"
                      authority="/akn/xx/act/2023/5">
            <destination href="/akn/xx/act/2020/1/main#sec_1" wId="sec_1"/>
            <old><quotedText>a</quotedText></old>
            <old><quotedText>b</quotedText></old>
            <new><quotedText>c</quotedText></new>
            <new><quotedText>d</quotedText></new>
          </textualMod>
        </passiveModifications>
      </analysis>"""
    )
    doc = parse_document(_make_akn(analysis_xml=analysis))
    assert len(doc.textual_mods) == 1
    quoted = doc.textual_mods[0].quoted
    assert quoted is not None
    assert quoted.old == ["a", "b"]
    assert quoted.new == ["c", "d"]


def test_category_derivation_from_action() -> None:
    assert category_for("renumbering") == "textual"
    assert category_for("authenticInterpretation") == "meaning"
    assert category_for("exceptionOfScope") == "scope"
    assert category_for("prorogationOfForce") == "force"
    assert category_for("retroactivity") == "efficacy"


def test_parses_lifecycle_events() -> None:
    lifecycle = dedent(
        """<lifecycle source="#tester">
        <eventRef eId="e1" date="2020-01-01" type="generation" source="#tester"/>
        <eventRef eId="e2" date="2023-04-15" type="amendment"
                  source="/akn/xx/act/2023/5"/>
      </lifecycle>"""
    )
    doc = parse_document(_make_akn(lifecycle_xml=lifecycle))
    assert len(doc.lifecycle_events) == 2
    kinds = {(ev.event_date, ev.event_type) for ev in doc.lifecycle_events}
    assert ("2020-01-01", "generation") in kinds
    assert ("2023-04-15", "amendment") in kinds


def test_missing_analysis_and_lifecycle_produce_empty_lists() -> None:
    doc = parse_document(_make_akn())
    assert doc.textual_mods == []
    assert doc.lifecycle_events == []


def test_invalid_action_type_is_skipped() -> None:
    analysis = dedent(
        """<analysis source="#tester">
        <passiveModifications>
          <textualMod type="not_a_real_action" source="#sec_1">
            <destination href="/akn/xx/act/2020/1/main#sec_1"/>
          </textualMod>
        </passiveModifications>
      </analysis>"""
    )
    doc = parse_document(_make_akn(analysis_xml=analysis))
    assert doc.textual_mods == []


# ── End-to-end: quoted-amendment emission through the pipeline ───────────


class TestQuotedAmendmentPipeline:
    """End to end: amendment text with a `تعدل المادة` trigger
    lands as a top-level article for the amending instrument plus a
    `<mod>` + `<quotedStructure>` for the embedded content."""

    def test_taadel_shape_marks_and_skips_from_scaffold(self) -> None:
        """Anchors are marked and the scaffold skips them; `<mod><quotedStructure>`
        emission (Bluebell integration) is deferred."""
        from codify.jurisdictions import load_config
        from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors
        from codify.pipeline.enrich.scaffold import scaffold_from_anchors

        text = (
            "المادة ١\nنص أول.\n\n"
            "المادة ٢\nنص ثان.\n\n"
            "المادة ٣\nتعدل المادة ٩ من قانون العقوبات لتصبح على النحو التالي :--\n\n"
            "المادة ٩\nالنص الجديد المستبدل.\n\n"
            "المادة ٤\nأحكام ختامية.\n"
        )
        config = load_config("xz")
        assert config is not None
        regex = build_anchor_regex(config, "act")
        anchors = scan_anchors(text, regex, country="xz", doctype="act")

        articles = [a for a in anchors if a.kind == "article"]
        assert any(a.number == "٩" and a.quoted_amendment for a in articles)
        assert not any(a.number == "٩" and a.akn_eid for a in articles)
        host_nums = [a.number for a in articles if not a.quoted_amendment]
        assert host_nums == ["١", "٢", "٣", "٤"]

        scaffold, index = scaffold_from_anchors(anchors)
        assert "ARTICLE ٩" not in scaffold
        assert set(index) == {"art_1", "art_2", "art_3", "art_4"}
