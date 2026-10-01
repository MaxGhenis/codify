# ruff: noqa: E501  # AKN XML test fixtures: line wraps would change tested whitespace
"""Tests for post-enrichment AKN validator."""

from __future__ import annotations

import subprocess
import sys

import pytest

from codify.akn import AKN_NS
from codify.pipeline.enrich.validator import (
    _STRUCTURAL_ELEMENT_NAMES,
    validate_akn,
)


def _act(body: str, refs: str = "", meta_extra: str = "", after_body: str = "") -> str:
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta>
      <identification source="#codify"/>
      <references source="#codify">{refs}</references>
      {meta_extra}
    </meta>
    <body>{body}</body>{after_body}
  </act>
</akomaNtoso>
'''


class TestEidUniqueness:
    def test_no_duplicates(self):
        xml = _act('<section eId="sec_1"/><section eId="sec_2"/>')
        assert not any(i["check"] == "eid_uniqueness" for i in validate_akn(xml))

    def test_duplicate_flagged(self):
        xml = _act('<section eId="sec_1"/><section eId="sec_1"/>')
        issues = [i for i in validate_akn(xml) if i["check"] == "eid_uniqueness"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert "sec_1" in issues[0]["message"]


class TestRefResolution:
    def test_valid_ref(self):
        xml = _act(
            '<section eId="sec_5"><content><p><ref href="#sec_5">s5</ref></p></content></section>'
        )
        assert not any(i["check"] == "ref_resolution" for i in validate_akn(xml))

    def test_broken_ref(self):
        xml = _act(
            '<section eId="sec_1"><content><p><ref href="#sec_99">s99</ref></p></content></section>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "ref_resolution"]
        assert len(issues) == 1
        assert "#sec_99" in issues[0]["message"]

    def test_external_ref_ignored(self):
        xml = _act(
            '<section eId="sec_1"><content><p><ref href="/akn/zz/act/1982/21">Act</ref></p></content></section>'
        )
        assert not any(i["check"] == "ref_resolution" for i in validate_akn(xml))


class TestDefinitionCompleteness:
    def test_complete(self):
        refs = '<TLCTerm eId="term-company" href="/ontology/term/company" showAs="company"/>'
        body = """
        <section eId="sec_2"><content><p><def refersTo="#term-company">company</def></p></content></section>
        <section eId="sec_5"><content><p><term refersTo="#term-company">company</term></p></content></section>
        """
        assert not any(
            i["check"] == "definition_completeness" for i in validate_akn(_act(body, refs))
        )

    def test_orphan_term(self):
        body = '<section eId="sec_5"><content><p><term refersTo="#term-missing">x</term></p></content></section>'
        issues = [i for i in validate_akn(_act(body)) if i["check"] == "definition_completeness"]
        assert len(issues) == 1
        assert "term-missing" in issues[0]["message"]

    def test_term_missing_refers_to(self):
        body = '<section eId="sec_5"><content><p><term>x</term></p></content></section>'
        issues = [i for i in validate_akn(_act(body)) if i["check"] == "definition_completeness"]
        assert len(issues) == 1
        assert "missing" in issues[0]["message"].lower()


class TestSectionNumbering:
    def test_sequential(self):
        xml = _act("""
        <section eId="sec_1"><num>1.</num></section>
        <section eId="sec_2"><num>2.</num></section>
        <section eId="sec_3"><num>3.</num></section>
        """)
        assert not any(i["check"] == "section_numbering" for i in validate_akn(xml))

    def test_small_gap_ok(self):
        # Gap of 2 is OK (inserted sections like 5A)
        xml = _act("""
        <section eId="sec_5"><num>5.</num></section>
        <section eId="sec_7"><num>7.</num></section>
        """)
        assert not any(i["check"] == "section_numbering" for i in validate_akn(xml))

    def test_large_gap_flagged(self):
        xml = _act("""
        <section eId="sec_1"><num>1.</num></section>
        <section eId="sec_10"><num>10.</num></section>
        """)
        issues = [i for i in validate_akn(xml) if i["check"] == "section_numbering"]
        assert len(issues) == 1
        assert "gap" in issues[0]["message"].lower()

    def test_gap_after_alphanumeric_label_still_flagged(self):
        # The signal number_gap misses: it stops the scope at the non-digit "2A",
        # but section_numbering reads the leading digits and reports 2 -> 10.
        xml = _act("""
        <section eId="sec_1"><num>1.</num></section>
        <section eId="sec_2A"><num>2A.</num></section>
        <section eId="sec_10"><num>10.</num></section>
        """)
        assert any(i["check"] == "section_numbering" for i in validate_akn(xml))


class TestHierarchyCoherence:
    def test_valid_hierarchy(self):
        xml = _act("""
        <section eId="sec_1">
          <subsection eId="sec_1__subsec_1">
            <paragraph eId="sec_1__subsec_1__para_a"/>
          </subsection>
        </section>
        """)
        assert not any(i["check"] == "hierarchy_coherence" for i in validate_akn(xml))

    def test_orphan_subsection(self):
        # subsection directly under body (no parent section)
        xml = _act('<subsection eId="subsec_1"/>')
        issues = [i for i in validate_akn(xml) if i["check"] == "hierarchy_coherence"]
        assert len(issues) >= 1
        assert "subsection" in issues[0]["message"]


class TestIntegration:
    def test_clean_document(self):
        refs = '<TLCTerm eId="term-fund" href="/ontology/term/fund" showAs="fund"/>'
        xml = _act(
            """
        <section eId="sec_1"><num>1.</num><heading>Short title</heading></section>
        <section eId="sec_2"><num>2.</num><heading>Interpretation</heading>
          <content><p><def refersTo="#term-fund">"fund"</def> means assets.</p></content>
        </section>
        <section eId="sec_3"><num>3.</num>
          <content><p>The <term refersTo="#term-fund">fund</term> shall be used under <ref href="#sec_2">section 2</ref>.</p></content>
        </section>
        """,
            refs,
        )
        issues = validate_akn(xml)
        assert len(issues) == 0

    def test_multiple_issues(self):
        xml = _act("""
        <section eId="sec_1"><num>1.</num></section>
        <section eId="sec_1"><num>1.</num></section>
        <section eId="sec_10"><num>10.</num>
          <content><p><ref href="#sec_99">s99</ref> <term refersTo="#term-nope">x</term></p></content>
        </section>
        """)
        issues = validate_akn(xml)
        checks = {i["check"] for i in issues}
        assert "eid_uniqueness" in checks
        assert "ref_resolution" in checks
        assert "definition_completeness" in checks
        assert "section_numbering" in checks


class TestBodyArtefacts:
    """LLM body-fill occasionally leaks JSON/markdown fence fragments into the
    rendered body text when its stop token fails on long chunks. Real-world
    examples: Customs Code art 5 (~50KB) ending with ``]}]}\\u200a````, NABU asset
    agency art 15 trailing ``]}]}\\u200a````Or more precisely as valid JSON``."""

    def test_clean_body_passes(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>The Authority shall act.</p></content></article>'
        )
        assert not any(i["check"] == "body_artefact" for i in validate_akn(xml))

    def test_json_fence_artefact_flagged(self):
        xml = _act(
            '<article eId="art_5"><num>5</num><content><p>Some content here ]}]}```</p></content></article>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "body_artefact"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["eid"] == "art_5"

    def test_markdown_fence_flagged(self):
        xml = _act(
            '<article eId="art_2"><num>2</num><content><p>Body text ```more text</p></content></article>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "body_artefact"]
        assert len(issues) == 1

    def test_explanatory_tail_flagged(self):
        # The NABU pattern: LLM appends a clarification sentence at the very end
        xml = _act(
            '<article eId="art_15"><num>15</num><content><p>The agency shall publish reports. Or more precisely, all reports.</p></content></article>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "body_artefact"]
        assert len(issues) == 1

    def test_legitimate_brackets_in_prose_pass(self):
        # Lone "]" or "}" inside prose must not false-fire
        xml = _act(
            '<article eId="art_3"><num>3</num><content><p>See annex [5] and clause {a}.</p></content></article>'
        )
        assert not any(i["check"] == "body_artefact" for i in validate_akn(xml))


class TestOrphanArticles:
    """Structurer container-assembly failure: articles sitting at <body> root
    alongside sections/chapters instead of inside them. The UA Customs Code
    re-ingest produced 99 such orphans next to 9 sections + 9 chapters."""

    def test_flat_articles_no_containers_pass(self):
        # A small act with no sections / chapters, articles at root is fine.
        xml = _act("""
        <article eId="art_1"><num>1</num></article>
        <article eId="art_2"><num>2</num></article>
        """)
        assert not any(i["check"] == "orphan_articles" for i in validate_akn(xml))

    def test_articles_nested_in_sections_pass(self):
        xml = _act("""
        <section eId="sec_1"><num>I</num>
          <article eId="sec_1__art_1"><num>1</num></article>
        </section>
        """)
        assert not any(i["check"] == "orphan_articles" for i in validate_akn(xml))

    def test_orphan_articles_alongside_sections_flagged(self):
        xml = _act("""
        <section eId="sec_1"><num>I</num>
          <article eId="sec_1__art_1"><num>1</num></article>
        </section>
        <article eId="art_2"><num>2</num></article>
        <article eId="art_3"><num>3</num></article>
        """)
        issues = [i for i in validate_akn(xml) if i["check"] == "orphan_articles"]
        assert len(issues) == 1
        assert issues[0]["count"] == 2
        assert "art_2" in issues[0]["eids"]
        assert "art_3" in issues[0]["eids"]
        # root articles AFTER a container are interspersed: a real defect
        assert issues[0]["severity"] == "error"

    def test_orphan_articles_alongside_chapters_flagged(self):
        xml = _act("""
        <chapter eId="chp_1"><num>1</num>
          <article eId="chp_1__art_1"><num>1</num></article>
        </chapter>
        <article eId="art_2"><num>2</num></article>
        """)
        issues = [i for i in validate_akn(xml) if i["check"] == "orphan_articles"]
        assert len(issues) == 1
        assert issues[0]["count"] == 1
        assert issues[0]["severity"] == "error"

    def test_small_leading_run_grades_warning(self):
        # general provisions before the first chapter: faithful shape, text present
        chapter = (
            '<chapter eId="chp_1"><num>1</num>'
            + "".join(
                f'<article eId="chp_1__art_{i}"><num>{i}</num></article>' for i in range(1, 19)
            )
            + "</chapter>"
        )
        xml = _act(
            '<article eId="art_g1"><num>1</num></article>'
            '<article eId="art_g2"><num>2</num></article>' + chapter
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "orphan_articles"]
        assert len(issues) == 1 and issues[0]["severity"] == "warning"

    def test_quoted_amendment_articles_do_not_dilute_ratio(self):
        # 3 leading orphans beside 1 chapter article is 75% malformed; 30 borrowed
        # articles inside a quoted amendment must not shrink the fraction to warning
        quoted = (
            "<quotedStructure>"
            + "".join(f'<article eId="q_art_{i}"><num>{i}</num></article>' for i in range(1, 31))
            + "</quotedStructure>"
        )
        chapter = '<chapter eId="chp_1"><num>1</num><article eId="chp_1__art_1"><num>1</num></article></chapter>'
        roots = "".join(f'<article eId="art_{i}"><num>{i}</num></article>' for i in range(1, 4))
        xml = _act(roots + chapter + quoted)
        issues = [i for i in validate_akn(xml) if i["check"] == "orphan_articles"]
        assert len(issues) == 1 and issues[0]["severity"] == "error"

    def test_large_root_fraction_grades_error(self):
        # nine root articles beside one thin chapter: wholesale assembly failure
        roots = "".join(f'<article eId="art_{i}"><num>{i}</num></article>' for i in range(1, 10))
        xml = _act(
            roots + '<chapter eId="chp_1"><num>1</num>'
            '<article eId="chp_1__art_1"><num>1</num></article></chapter>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "orphan_articles"]
        assert len(issues) == 1 and issues[0]["severity"] == "error"


class TestOcrGarble:
    """Detect Arabic OCR mojibake the body-fill LLM passes through unchanged.
    The PS Monetary Authority 1997 re-ingest carried 106 instances of doubled
    ta-marbuta (e.g. ``الآتيةة``) straight from the PDF text-extraction layer."""

    def test_clean_arabic_passes(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>الآتية المعاني المخصصة لها</p></content></article>'
        )
        assert not any(i["check"] == "ocr_garble" for i in validate_akn(xml))

    def test_doubled_ta_marbuta_flagged(self):
        # The signature mojibake pattern from PS Monetary 1997 art 1
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>والعبارات الآتيةة المعاني</p></content></article>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "ocr_garble"]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_1"
        assert issues[0]["count"] == 1

    def test_multiple_garble_aggregated_per_anchor(self):
        # Two mojibake hits in the same article, aggregate into one issue with count=2
        xml = _act(
            '<article eId="art_5"><num>5</num><content>'
            "<p>الآتيةة المعةاني</p>"
            "<p>والمخصصةة لها</p>"
            "</content></article>"
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "ocr_garble"]
        assert len(issues) == 1
        assert issues[0]["count"] == 2

    def test_doubled_hamza_flagged(self):
        # Doubled hamza (ءء) is also virtually impossible in real Arabic
        xml = _act(
            '<article eId="art_2"><num>2</num><content><p>للحرف ءء غير شائع</p></content></article>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "ocr_garble"]
        assert len(issues) == 1


class TestEmptyArticles:
    """An article with only `<num>` and no body content. PS Criminal Code 1936
    had three such zombies (`art_٧`, `art_٩٨`, `art_١`) appended after
    art_391, the structurer recovered them as fragments but never landed any
    source text under them."""

    def test_article_with_body_passes(self):
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>Body text.</p></content></article>'
        )
        assert not any(i["check"] == "empty_article" for i in validate_akn(xml))

    def test_article_with_only_num_flagged(self):
        xml = _act('<article eId="art_99"><num>99</num></article>')
        issues = [i for i in validate_akn(xml) if i["check"] == "empty_article"]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_99"

    def test_article_with_num_and_heading_only_flagged(self):
        # Per the Criminal Code's `art_١` zombie shape
        xml = _act('<article eId="art_5"><num>5</num><heading>Title</heading></article>')
        issues = [i for i in validate_akn(xml) if i["check"] == "empty_article"]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_5"

    def test_article_with_empty_content_flagged(self):
        # `<num>` plus empty `<paragraph>`, the `art_٧` pattern
        xml = _act('<article eId="art_7"><num>7</num><paragraph eId="art_7__para_1"/></article>')
        issues = [i for i in validate_akn(xml) if i["check"] == "empty_article"]
        assert len(issues) == 1


class TestSwallowedEnumerators:
    """The body-fill LLM occasionally misses a sibling enumerator and bakes it
    into the previous sibling's text. PS Criminal Code 1936 has 12 of these on
    the `(د) → (ه)` boundary. Detection signature: parent's point sequence has
    a gap AND the missing letter appears inline in the preceding sibling's body."""

    def test_complete_sequence_passes(self):
        xml = _act(
            '<article eId="art_1"><num>1</num>'
            '<point eId="art_1__point_a"><num>(أ)</num><content><p>alpha</p></content></point>'
            '<point eId="art_1__point_b"><num>(ب)</num><content><p>beta</p></content></point>'
            '<point eId="art_1__point_c"><num>(ج)</num><content><p>gamma</p></content></point>'
            "</article>"
        )
        assert not any(i["check"] == "swallowed_enumerator" for i in validate_akn(xml))

    def test_swallowed_haa_flagged(self):
        # `(أ)(ب)(ج)(د)(و)`, `(ه)` missing AND appears inside `(د)`'s body
        xml = _act(
            '<article eId="art_1"><num>1</num>'
            '<point eId="art_1__point_a"><num>(أ)</num><content><p>alpha</p></content></point>'
            '<point eId="art_1__point_b"><num>(ب)</num><content><p>beta</p></content></point>'
            '<point eId="art_1__point_c"><num>(ج)</num><content><p>gamma</p></content></point>'
            '<point eId="art_1__point_d"><num>(د)</num><content>'
            "<p>delta (ه) the-real-haa-content was swallowed here</p>"
            "</content></point>"
            '<point eId="art_1__point_w"><num>(و)</num><content><p>waaw</p></content></point>'
            "</article>"
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "swallowed_enumerator"]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_1__point_d"
        assert "ه" in issues[0]["missing"]

    def test_too_few_siblings_no_check(self):
        # Need ≥3 siblings to infer a sequence shape
        xml = _act(
            '<article eId="art_1"><num>1</num>'
            '<point eId="art_1__point_a"><num>(أ)</num><content><p>(ب) — actually a citation</p></content></point>'
            "</article>"
        )
        assert not any(i["check"] == "swallowed_enumerator" for i in validate_akn(xml))


class TestHierarchyCoherenceRefinements:
    def test_subsection_in_quoted_structure_not_flagged(self):
        # Quoted amendment fragments carry their own hierarchy, exempt.
        xml = _act(
            '<section eId="sec_1"><content><p><mod>'
            '<quotedStructure><subsection eId="q_1"><num>(1)</num></subsection></quotedStructure>'
            "</mod></p></content></section>"
        )
        assert not any(i["check"] == "hierarchy_coherence" for i in validate_akn(xml))

    def test_subsection_under_chapter_allowed(self):
        # Grouping containers legitimately hold provisions across traditions.
        xml = _act(
            '<chapter eId="chp_1"><subsection eId="chp_1__subsec_1"><num>(1)</num></subsection></chapter>'
        )
        assert not any(i["check"] == "hierarchy_coherence" for i in validate_akn(xml))

    def test_subparagraph_under_article_still_flagged(self):
        # A basic unit at the wrong level is still a real structural error.
        xml = _act(
            '<article eId="art_1"><subparagraph eId="art_1__subpara_1"><num>a</num></subparagraph></article>'
        )
        assert any(i["check"] == "hierarchy_coherence" for i in validate_akn(xml))

    def test_duplicate_number_scoped_through_wrapper(self):
        # A subsection under an unlabelled wrapper still scopes to its section,
        # so first-subsections across sections are not false duplicates.
        xml = _act(
            '<section eId="sec_61"><hcontainer><subsection eId="sec_61__1"><num>(1)</num></subsection></hcontainer></section>'
            '<section eId="sec_215"><hcontainer><subsection eId="sec_215__1"><num>(1)</num></subsection></hcontainer></section>'
        )
        assert not any(i["check"] == "duplicate_number" for i in validate_akn(xml))


class TestNumberGapLikelihood:
    def _gap(self, nums: list[int]):
        body = "".join(f'<article eId="art_{n}"><num>{n}.</num></article>' for n in nums)
        gaps = [i for i in validate_akn(_act(body)) if i["check"] == "number_gap"]
        return gaps[0] if gaps else None

    def test_contiguous_run_is_a_defect_and_warns(self):
        # 4,5,6,7,8 missing between 3 and 9, a structuring drop.
        g = self._gap([1, 2, 3, 9, 10])
        assert g is not None and g["likely"] == "defect" and g["severity"] == "warning"

    def test_scattered_singletons_read_as_repeals_and_stay_info(self):
        # two isolated holes in a dense sequence, legitimate repeals.
        g = self._gap([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15])
        assert g is not None and g["likely"] == "repeal" and g["severity"] == "info"


class TestFabricatedFormula:
    def _act_with_formula(self, formula_text: str) -> str:
        return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <preamble><formula name="enactingFormula"><p>{formula_text}</p></formula></preamble>
    <body><article eId="art_1"><num>1</num><content><p>Body.</p></content></article></body>
  </act>
</akomaNtoso>
'''

    def test_formula_absent_from_source_flagged(self):
        xml = self._act_with_formula("ENACTED by the Parliament of Nowhere.")
        issues = [
            i
            for i in validate_akn(xml, source_text="Article 1. Body.")
            if i["check"] == "fabricated_formula"
        ]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"

    def test_formula_present_in_source_silent(self):
        xml = self._act_with_formula("ENACTED by the Parliament.")
        source = "Preamble text\nENACTED   by the\nParliament. Article 1."
        assert not any(
            i["check"] == "fabricated_formula" for i in validate_akn(xml, source_text=source)
        )

    def test_check_skipped_without_source_text(self):
        xml = self._act_with_formula("ENACTED by the Parliament of Nowhere.")
        assert not any(i["check"] == "fabricated_formula" for i in validate_akn(xml))

    def test_config_injected_formula_skipped_on_provenance(self):
        xml = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <preamble><formula name="enactingFormula" refersTo="#codify"><p>Config fallback text.</p></formula></preamble>
    <body><article eId="art_1"><num>1</num><content><p>Body.</p></content></article></body>
  </act>
</akomaNtoso>
'''
        assert not any(
            i["check"] == "fabricated_formula"
            for i in validate_akn(xml, source_text="Article 1. Body.")
        )

    def test_orthography_drift_not_flagged(self):
        # Transcription folded alef/hamza + dropped diacritics; still the
        # source's own formula, not fabrication.
        xml = self._act_with_formula("أَصدرنا القانون الآتي:")
        source = "الديباجة اصدرنا القانون الاتي: مادة 1"
        assert not any(
            i["check"] == "fabricated_formula" for i in validate_akn(xml, source_text=source)
        )


class TestFormulaIntegrity:
    def _act(self, formula_text: str, lang: str = "ara") -> str:
        return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify">
      <FRBRExpression><FRBRlanguage language="{lang}"/></FRBRExpression>
    </identification></meta>
    <preamble><formula name="enactingFormula"><p>{formula_text}</p></formula></preamble>
    <body><article eId="art_1"><num>1</num><content><p>نص</p></content></article></body>
  </act>
</akomaNtoso>
'''

    def test_ellipsis_terminated_formula_flagged(self):
        xml = self._act("وبعد الاطلاع على... أصدر القرار الآتي بقانون:...")
        issues = [i for i in validate_akn(xml) if i["check"] == "formula_truncated"]
        assert any("truncated" in i["message"] for i in issues)

    def test_script_mismatch_flagged_on_english_expression(self):
        xml = self._act("قرر المجلس التشريعي الفلسطيني القانون الآتي:", lang="eng")
        issues = [i for i in validate_akn(xml) if i["check"] == "formula_script_mismatch"]
        assert any("script" in i["message"] for i in issues)

    def test_matching_script_and_complete_formula_silent(self):
        xml = self._act("قرر المجلس التشريعي الفلسطيني القانون الآتي:", lang="ara")
        checks = {i["check"] for i in validate_akn(xml)}
        assert "formula_truncated" not in checks and "formula_script_mismatch" not in checks


def test_money_words_mismatch_fires_with_both_readings() -> None:
    xml = _act(
        '<article eId="art_8"><num>8</num><content>'
        "<p>يعاقب بغرامة لا تقل عن (500,000) خمسون ألف دينار أردني.</p>"
        "</content></article>"
    )
    found = [i for i in validate_akn(xml) if i["check"] == "money_words_mismatch"]
    assert len(found) == 1
    assert found[0]["severity"] == "warning"
    assert found[0]["eid"] == "art_8"
    assert found[0]["numeral_value"] == 500_000
    assert found[0]["words_value"] == 50_000


def test_money_words_consistent_is_silent() -> None:
    xml = _act(
        '<article eId="art_8"><num>8</num><content>'
        "<p>يعاقب بغرامة لا تقل عن (50,000) خمسون ألف دينار أردني.</p>"
        "</content></article>"
    )
    assert [i for i in validate_akn(xml) if i["check"] == "money_words_mismatch"] == []


def test_anchor_count_schedule_maps_to_attachment() -> None:
    """Bluebell parses SCHEDULE into <attachment>; the count check must
    speak that vocabulary or every annex-bearing act false-alarms."""
    xml = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <body><article eId="art_1"><num>1</num><content><p>نص.</p></content></article></body>
    <attachments><attachment eId="att_1"><doc name="schedule"><mainBody><p>جدول.</p></mainBody></doc></attachment></attachments>
  </act>
</akomaNtoso>'''
    issues = validate_akn(xml, expected_anchor_summary={"article": 1, "schedule": 1})
    assert [i for i in issues if i["check"] == "anchor_count_mismatch"] == []


def test_anchor_count_dropped_schedule_still_detected() -> None:
    xml = _act('<article eId="art_1"><num>1</num><content><p>نص.</p></content></article>')
    issues = validate_akn(xml, expected_anchor_summary={"schedule": 1})
    found = [i for i in issues if i["check"] == "anchor_count_mismatch"]
    assert len(found) == 1 and found[0]["kind"] == "schedule"


def test_anchor_count_unrelated_attachment_does_not_mask_dropped_schedule() -> None:
    xml = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <body><article eId="art_1"><num>1</num><content><p>نص.</p></content></article></body>
    <attachments><attachment eId="att_1"><doc name="annexure"><mainBody><p>x</p></mainBody></doc></attachment></attachments>
  </act>
</akomaNtoso>'''
    issues = validate_akn(xml, expected_anchor_summary={"schedule": 1})
    found = [i for i in issues if i["check"] == "anchor_count_mismatch"]
    assert len(found) == 1 and found[0]["kind"] == "schedule"


def _identified(date: str, uri: str) -> str:
    """The shape Cobalt actually emits: FRBRuri carries the bare work URI and
    FRBRthis the same path plus a `/!main` component. An earlier version of this
    fixture omitted the component and the URI-mismatch check passed against a
    document the pipeline never produces."""
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta>
      <identification source="#codify">
        <FRBRWork>
          <FRBRthis value="{uri}/!main"/>
          <FRBRuri value="{uri}"/>
          <FRBRdate date="{date}" name="enactment"/>
        </FRBRWork>
      </identification>
    </meta>
    <body><section eId="sec_1"/></body>
  </act>
</akomaNtoso>
'''


class TestIdentityConsistency:
    def _checks(self, date: str, uri: str) -> set[str]:
        return {
            i["check"]
            for i in validate_akn(_identified(date, uri))
            if i["check"].startswith("identity_")
        }

    def test_agreeing_identity_is_silent(self):
        assert not self._checks("2007-05-12", "/akn/xz/act/2007/11")

    def test_future_year_flagged(self):
        # Found on the stored corpus: laws.year 2037 against a /2007/ URI.
        assert "identity_year_in_future" in self._checks("2037-05-12", "/akn/xz/act/2037/11")

    def test_unconverted_hijri_year_flagged(self):
        # 1431 AH is 2010 CE; the Gregorian slot took the Hijri number verbatim.
        assert "identity_year_unconverted_hijri" in self._checks("1431-01-01", "/akn/xz/act/1431/7")

    def test_year_predating_the_corpus_flagged(self):
        assert "identity_year_implausible" in self._checks("1766-01-01", "/akn/xz/act/1766/7")

    def test_uri_disagreeing_with_the_date_flagged(self):
        # The defect that makes a law citable two ways, neither authoritative.
        assert "identity_year_uri_mismatch" in self._checks("2007-05-12", "/akn/xz/act/2009/11")

    def test_unknown_date_placeholder_is_not_an_implausible_year(self):
        # The placeholder says the year never resolved, which the URI already reports.
        # Reading it as a date graded every year-less document blocking.
        assert not self._checks("0001", "/akn/xz/act/0001/draft-abc")

    def test_missing_identification_is_not_a_finding(self):
        # Absent metadata is a different defect; this check only compares what is there.
        assert not [
            i
            for i in validate_akn(_act('<section eId="s"/>'))
            if i["check"].startswith("identity_")
        ]


def test_the_delivered_fixture_has_a_self_consistent_identity() -> None:
    """Run the checks over a pipeline-emitted document, not a hand-written one.

    A hand-written identification block can be wrong in the same direction as the
    code that reads it; a document the real parser produced cannot. This is a
    synthetic law parsed through the same parse_to_akn + normalise_akn_meta path."""
    from pathlib import Path

    fixture = Path(__file__).parents[1] / "fixtures" / "synthetic-zerzura-1994.akn.xml"
    findings = [
        i
        for i in validate_akn(fixture.read_text(encoding="utf-8"))
        if i["check"].startswith("identity_")
    ]
    assert findings == []


class TestCollapsedBisArticle:
    def _bis_body(self) -> str:
        # Two sub-point runs each restarting at (أ): two consecutive bis
        # articles collapsed into one element (the Inheritance Law 1944 shape).
        return (
            '<article eId="art_15bis"><num>15bis</num>'
            '<point eId="art_15bis__point_a1"><num>(أ)</num><content><p>first run alpha</p></content></point>'
            '<point eId="art_15bis__point_b1"><num>(ب)</num><content><p>first run beta</p></content></point>'
            '<point eId="art_15bis__point_a2"><num>(أ)</num><content><p>second run alpha</p></content></point>'
            '<point eId="art_15bis__point_b2"><num>(ب)</num><content><p>second run beta</p></content></point>'
            "</article>"
        )

    def test_no_bis_marker_keeps_generic_duplicate(self):
        # Same (أ),(ب),(أ),(ب) shape but the article is not marked bis: it is not
        # a collapsed_bis, and its duplicate_number stays actionable.
        body = self._bis_body().replace("<num>15bis</num>", "<num>5</num>")
        issues = validate_akn(_act(body))
        assert not any(i["check"] == "collapsed_bis_article" for i in issues)
        assert any(
            i["check"] == "duplicate_number" and i.get("scope") == "art_15bis" for i in issues
        )

    def test_flags_two_runs_restarting_at_alef(self):
        issues = [
            i for i in validate_akn(_act(self._bis_body())) if i["check"] == "collapsed_bis_article"
        ]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_15bis"
        assert issues[0]["severity"] == "error"

    def test_suppresses_redundant_duplicate_number(self):
        # The collapse also duplicates (أ)/(ب); the bis finding owns it so the
        # repair agent is not routed to a cosmetic renumber.
        issues = validate_akn(_act(self._bis_body()))
        assert not any(
            i["check"] == "duplicate_number" and i.get("scope") == "art_15bis" for i in issues
        )

    def test_no_fire_on_single_ordered_run(self):
        body = (
            '<article eId="art_1"><num>1</num>'
            '<point eId="art_1__point_a"><num>(أ)</num><content><p>alpha</p></content></point>'
            '<point eId="art_1__point_b"><num>(ب)</num><content><p>beta</p></content></point>'
            '<point eId="art_1__point_j"><num>(ج)</num><content><p>gamma</p></content></point>'
            '<point eId="art_1__point_d"><num>(د)</num><content><p>delta</p></content></point>'
            "</article>"
        )
        assert not any(i["check"] == "collapsed_bis_article" for i in validate_akn(_act(body)))

    def test_suppression_preserves_unrelated_duplicate(self):
        # Bis collapse (أ,ب,أ,ب) plus an independent (ج) double-emit: the collapse's
        # own letters suppress, the unrelated (ج) duplicate stays actionable.
        body = (
            self._bis_body()[: -len("</article>")]
            + '<point eId="art_15bis__point_j1"><num>(ج)</num><content><p>x</p></content></point>'
            + '<point eId="art_15bis__point_j2"><num>(ج)</num><content><p>y</p></content></point>'
            + "</article>"
        )
        issues = validate_akn(_act(body))
        dups = [
            i for i in issues if i["check"] == "duplicate_number" and i.get("scope") == "art_15bis"
        ]
        assert any(i["check"] == "collapsed_bis_article" for i in issues)
        assert len(dups) == 1 and "ج" in dups[0]["number"]


class TestSeamDuplication:
    _SENT = "يحظر ويمنع ذبح أي من الحيوانات خارج المسلخ في أي مكان داخل منطقة البلدية."

    def test_flags_page_break_restart(self):
        s = self._SENT
        body = (
            '<article eId="art_4"><num>4</num>'
            f'<point eId="art_4__point_1"><num>1.</num><content><p>{s}</p></content></point>'
            '<point eId="art_4__point_2"><num>2.</num><content><p>في حاله عدم تحقق</p></content></point>'
            '<paragraph eId="art_4__para_1"><num>1</num>'
            f'<point eId="art_4__para_1__point_1"><num>1.</num><content><p>{s}</p></content></point>'
            '<point eId="art_4__para_1__point_2"><num>2.</num><content><p>في حاله عدم تحقق الفقرة من هذه المادة.</p></content></point>'
            "</paragraph></article>"
        )
        issues = [i for i in validate_akn(_act(body)) if i["check"] == "seam_duplication"]
        assert len(issues) == 1
        assert issues[0]["eid"] == "art_4"
        assert issues[0]["severity"] == "warning"

    def test_no_fire_on_genuine_distinct_clauses(self):
        body = (
            '<article eId="art_1"><num>1</num>'
            '<point eId="art_1__point_1"><num>1.</num><content><p>The operator shall keep a register of every consignment received in full.</p></content></point>'
            '<point eId="art_1__point_2"><num>2.</num><content><p>The register shall be produced to an inspector on demand at any reasonable time.</p></content></point>'
            "</article>"
        )
        assert not any(i["check"] == "seam_duplication" for i in validate_akn(_act(body)))

    def test_no_fire_on_repeated_short_line(self):
        # Two identical short lines (a form field) must not trip the >=40-char
        # opening-span guard.
        body = (
            '<article eId="art_2"><num>2</num>'
            "<content><p>الاسم: ____</p><p>الاسم: ____</p></content>"
            "</article>"
        )
        assert not any(i["check"] == "seam_duplication" for i in validate_akn(_act(body)))


def _root(root: str, body_tag: str, body: str, after_body: str = "") -> str:
    """An `act`, `doc` or `judgment` root, each naming its body differently."""
    ns = 'xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"'
    meta = '<meta><identification source="#codify"/><references source="#codify"/></meta>'
    name = ' name="x"' if root != "act" else ""
    return (
        f"<akomaNtoso {ns}><{root}{name}>{meta}"
        f"<{body_tag}>{body}</{body_tag}>{after_body}</{root}></akomaNtoso>"
    )


class TestMarkupAsText:
    """A paragraph whose whole text is markup vocabulary: the model emitting its own
    scaffolding where the law should be."""

    def _p(self, text: str) -> str:
        return _act(f'<section eId="sec_1"><content><p>{text}</p></content></section>')

    def _checks(self, text: str) -> set[str]:
        return {i["check"] for i in validate_akn(self._p(text))}

    def test_a_repeated_numbered_element_name_is_an_error(self):
        issues = [
            i
            for i in validate_akn(self._p("HCONTAINER 12 HCONTAINER 9"))
            if i["check"] == "markup_as_text"
        ]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["eid"] == "sec_1"

    def test_a_single_element_name_is_caught_whatever_its_case(self):
        for text in ("HCONTAINER 1", "hcontainer 1", "quotedStructure", "QUOTEDSTRUCTURE"):
            assert "markup_as_text" in self._checks(text), text

    def test_a_heading_naming_a_document_part_is_left_alone(self):
        # The control that matters: `article` and `section` are ordinary law, so the
        # vocabulary deliberately excludes them or this check would take real text.
        for text in ("Article 2", "Section 1", "Chapter IV", "Paragraph"):
            assert "markup_as_text" not in self._checks(text), text

    def test_prose_that_opens_with_an_element_name_is_left_alone(self):
        # Exercises the anchoring: text that opens with the word and carries on is
        # what tests that the match is required to reach the end.
        for text in (
            "quotedStructure references are set out in Annex I.",
            "hcontainer 3 shall be read with the preceding subsection.",
            "The hcontainer element is described in Annex I.",
        ):
            assert "markup_as_text" not in self._checks(text), text

    def test_a_long_repeat_does_not_hang_the_validator(self):
        # Two separators per repetition let a space belong to either, so a failing
        # long repeat explores every partition. The corpus holds a fifty-repeat case.
        # Bounded in a child process because `re` takes no timeout and holds the GIL,
        # so in-process this would hang the run rather than fail it.
        script = (
            "from codify.pipeline.enrich.validator import validate_akn\n"
            "xml = %r %% ('HCONTAINER ' * 60 + 'and then some prose that stops it.')\n"
            "assert not [i for i in validate_akn(xml) "
            "if i['check'] == 'markup_as_text']\n" % self._p("%s")
        )
        try:
            done = subprocess.run([sys.executable, "-c", script], timeout=10, capture_output=True)
        except subprocess.TimeoutExpired:
            pytest.fail("the markup pattern did not terminate on a long repeat")
        assert done.returncode == 0, done.stderr.decode()

    def test_a_long_repeat_that_does_match_is_still_caught(self):
        assert "markup_as_text" in self._checks("HCONTAINER 1 " * 60)

    def test_law_in_a_neighbouring_paragraph_is_not_claimed_lost(self):
        # Per paragraph: a unit holding one scaffolding line beside real law is still
        # flagged, and the message must not claim the whole provision reads that.
        xml = _act(
            '<section eId="sec_1"><content><p>HCONTAINER 1</p>'
            "<p>The Authority shall act within thirty days.</p></content></section>"
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "markup_as_text"]
        assert len(issues) == 1
        assert "paragraph" in issues[0]["message"]
        assert "provision" not in issues[0]["message"]

    @pytest.mark.parametrize(
        "text",
        [
            "\u200fHCONTAINER 1",
            "HCONTAINER 1\u200e",
            "HCON\u200bTAINER 1",
            "\u202bHCONTAINER 12 HCONTAINER 9",
            "\u061cHCONTAINER 1",
            "HCON\u00adTAINER 1",
            "HCONTAINER\u2060 1",
            "\u200f HCONTAINER 1",
            "\u061c HCONTAINER 12 HCONTAINER 9",
        ],
    )
    def test_an_invisible_mark_does_not_defeat_the_match(self, text):
        # One mark anywhere breaks an anchored match. The last three fall outside the
        # ranges a character class listed, which is why the strip goes by category.
        assert "markup_as_text" in self._checks(text), repr(text)

    # Written out rather than derived: parametrising over the tuple itself cannot
    # fail when a name leaves it, it just runs one case fewer.
    VOCABULARY = frozenset(
        {
            "hcontainer",
            "blockList",
            "blockContainer",
            "crossHeading",
            "embeddedStructure",
            "authorialNote",
            "subFlow",
            "longTitle",
            "amendmentList",
            "portionBody",
            "tblock",
            "quotedStructure",
            "quotedText",
            "listIntroduction",
            "listWrapUp",
            "mainBody",
            "amendmentBody",
            "debateBody",
            "judgmentBody",
            "debateSection",
            "componentRef",
            "documentRef",
            "eventRef",
            "temporalGroup",
            "wrapUp",
        }
    )

    def test_the_vocabulary_is_exactly_what_is_tested(self):
        assert set(_STRUCTURAL_ELEMENT_NAMES) == self.VOCABULARY

    @pytest.mark.parametrize("name", sorted(VOCABULARY))
    def test_every_name_in_the_vocabulary_is_matched(self, name):
        assert "markup_as_text" in self._checks(f"{name} 1"), name

    @pytest.mark.parametrize("name", ["article", "section", "chapter", "paragraph", "clause"])
    def test_a_name_a_drafter_writes_is_not_in_the_vocabulary(self, name):
        # The exclusions are load-bearing: adding any of these would flag real law.
        assert name not in _STRUCTURAL_ELEMENT_NAMES
        assert "markup_as_text" not in self._checks(f"{name} 1"), name

    def test_a_long_run_is_truncated_in_the_message(self):
        message = next(
            i["message"]
            for i in validate_akn(self._p("HCONTAINER 1 " * 20))
            if i["check"] == "markup_as_text"
        )
        assert "…" in message
        assert len(message) < 260

    def test_the_message_reads_without_an_eid(self):
        # Straight into the body, so no ancestor carries an eId: building this inside
        # a section would inherit sec_1 and test nothing.
        issues = [
            i for i in validate_akn(_act("<p>HCONTAINER 1</p>")) if i["check"] == "markup_as_text"
        ]
        assert len(issues) == 1
        assert issues[0]["eid"] == ""
        assert "''" not in issues[0]["message"]

    def test_the_message_says_what_is_wrong_without_naming_a_tag(self):
        message = next(
            i["message"]
            for i in validate_akn(self._p("HCONTAINER 1"))
            if i["check"] == "markup_as_text"
        )
        for jargon in ("<p>", "AKN", "element name", "regex"):
            assert jargon not in message


class TestEmptyBody:
    """A body with neither provisions nor prose. Seventeen versions graded clean on
    this shape, holding only a title and identifying details."""

    def test_a_body_with_no_provisions_and_no_text_is_an_error(self):
        issues = [i for i in validate_akn(_act("")) if i["check"] == "empty_body"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"

    def test_a_body_holding_a_provision_is_not_flagged(self):
        xml = _act('<section eId="sec_1"><content><p>نص</p></content></section>')
        assert not any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_a_long_unstructured_body_stays_structureless_not_empty(self):
        # The two checks divide the space: text survived without structure is a
        # warning, nothing survived is an error. Neither should claim the other.
        xml = _act(
            f'<section eId="sec_1"><content><p>{TestStructurelessBody.JUDGMENT}</p></content></section>'
        )
        checks = {i["check"] for i in validate_akn(xml)}
        assert "structureless_body" in checks
        assert "empty_body" not in checks

    def test_an_annex_bearing_act_with_a_referring_body_is_not_empty(self):
        # An annex sits in <attachments>, a sibling of <body>, and is inlined rather
        # than referenced, so a body of one referring line is an ordinary shape.
        att = (
            '<attachments><attachment eId="att_1"><doc name="schedule"><mainBody>'
            '<article eId="a1"><num>1</num><content><p>'
            + ("Rate text. " * 80)
            + "</p></content></article></mainBody></doc></attachment></attachments>"
        )
        xml = _act("<p>The rates in Schedule 1 apply from 1 January.</p>", after_body=att)
        assert not any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_a_provisionless_body_between_the_two_floors_is_still_flagged(self):
        # Neither check saw this band: too long for empty_body, too short for
        # structureless_body's one-provision floor.
        xml = _act("<p>" + ("word " * 600) + "</p>")
        assert any(i["check"] == "structureless_body" for i in validate_akn(xml))

    def test_an_empty_attachment_does_not_exempt_a_document(self):
        # Presence of the element is not content: an empty annex would exempt a
        # document that holds nothing anywhere.
        xml = _act("", after_body='<attachments><attachment eId="att_1"/></attachments>')
        assert any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_a_schedule_bearing_act_with_a_long_referring_body_is_not_flagged(self):
        # The lowered provisionless floor needs the same exemption, or a referring
        # body of a few hundred characters trips structureless_body instead.
        att = (
            '<attachments><attachment eId="att_1"><doc name="schedule"><mainBody>'
            '<article eId="a1"><num>1</num><content><p>'
            + ("Rate text. " * 80)
            + "</p></content></article></mainBody></doc></attachment></attachments>"
        )
        body = "<p>" + ("The rates set out in Schedule 1 apply from 1 January. " * 8) + "</p>"
        checks = {i["check"] for i in validate_akn(_act(body, after_body=att))}
        assert "empty_body" not in checks
        assert "structureless_body" not in checks

    # Content-bearing, so a body that is not found and a body that is found empty
    # are distinguishable: only reaching the tag keeps these off the finding.
    def test_a_doc_root_with_content_is_not_flagged(self):
        xml = _root(
            "doc",
            "mainBody",
            '<article eId="a1"><num>1</num><content><p>Text.</p></content></article>',
        )
        assert not any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_a_judgment_root_with_content_is_not_flagged(self):
        xml = _root(
            "judgment",
            "judgmentBody",
            '<article eId="a1"><num>1</num><content><p>Text.</p></content></article>',
        )
        assert not any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_a_doc_root_with_an_empty_body_is_flagged(self):
        assert any(i["check"] == "empty_body" for i in validate_akn(_root("doc", "mainBody", "")))

    def test_an_annexes_own_body_is_not_taken_for_the_documents(self):
        # `body` is tried before `mainBody`, so without skipping an attachment's own
        # the annex's would be read as the document's. The annex here holds a short
        # note rather than content, so nothing else keeps the finding away.
        att = (
            '<attachments><attachment eId="att_1"><doc name="schedule"><body>'
            "<p>A short note.</p></body></doc></attachment></attachments>"
        )
        body = '<article eId="a1"><num>1</num><content><p>Real text.</p></content></article>'
        assert not any(
            i["check"] == "empty_body" for i in validate_akn(_root("doc", "mainBody", body, att))
        )

    def test_no_body_element_at_all_is_the_same_defect(self):
        ns = 'xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"'
        meta = '<meta><identification source="#c"/><references source="#c"/></meta>'
        xml = f"<akomaNtoso {ns}><act>{meta}</act></akomaNtoso>"
        assert any(i["check"] == "empty_body" for i in validate_akn(xml))

    def test_an_annex_does_not_excuse_a_body_of_its_own(self):
        # The exemption covers a referring line. A body substantial in its own right
        # is judged as one, or any provisionless body escapes to the higher floor.
        att = (
            '<attachments><attachment eId="att_1"><doc name="schedule"><mainBody>'
            '<article eId="a1"><num>1</num><content><p>'
            + ("Rate text. " * 80)
            + "</p></content></article></mainBody></doc></attachment></attachments>"
        )
        xml = _act("<p>" + ("word " * 1876) + "</p>", after_body=att)
        assert any(i["check"] == "structureless_body" for i in validate_akn(xml))

    def test_the_message_avoids_tag_names(self):
        message = next(i["message"] for i in validate_akn(_act("")) if i["check"] == "empty_body")
        for jargon in ("<body>", "eId", "AKN", "provision"):
            assert jargon not in message


class TestStructurelessBody:
    """A long one-provision body has no structure and every other check calls it clean."""

    # The shape a court judgment reaches: no markers, so it is kept whole.
    JUDGMENT = "Menimbang bahwa permohonan Pemohon beralasan menurut hukum. " * 400

    def test_a_long_single_provision_body_is_flagged(self):
        xml = _act(f'<section eId="sec_1"><content><p>{self.JUDGMENT}</p></content></section>')
        issues = [i for i in validate_akn(xml) if i["check"] == "structureless_body"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert issues[0]["chars"] > 10_000

    def test_the_message_avoids_tag_names_and_reads_as_a_sentence(self):
        xml = _act(f'<section eId="sec_1"><content><p>{self.JUDGMENT}</p></content></section>')
        message = next(
            i["message"] for i in validate_akn(xml) if i["check"] == "structureless_body"
        )
        # Surfaces to readers, so no element or pass names.
        for jargon in ("<section>", "eId", "AKN", "verbatim", "anchor"):
            assert jargon not in message
        assert "single block" in message

    def test_a_short_flat_instrument_stays_clean(self):
        """274 of the corpus's 288 one-provision bodies are this shape."""
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>This Decree enters into '
            "force on the date of its publication in the Official Gazette.</p></content></article>"
        )
        assert not any(i["check"] == "structureless_body" for i in validate_akn(xml))

    def test_a_structured_document_stays_clean_however_long(self):
        body = '<chapter eId="chp_1"><num>1</num>' + "".join(
            f'<article eId="chp_1__art_{n}"><num>{n}</num>'
            f"<content><p>{self.JUDGMENT}</p></content></article>"
            for n in (1, 2, 3)
        )
        assert not any(
            i["check"] == "structureless_body" for i in validate_akn(_act(body + "</chapter>"))
        )

    def test_the_floor_is_what_separates_them(self):
        under = "x " * 4_000  # 8,000 characters
        over = "x " * 6_000  # 12,000 characters
        for text, expected in ((under, False), (over, True)):
            xml = _act(f'<section eId="sec_1"><content><p>{text}</p></content></section>')
            fired = any(i["check"] == "structureless_body" for i in validate_akn(xml))
            assert fired is expected

    def test_an_empty_body_is_not_reported(self):
        """A different defect, already covered elsewhere."""
        assert not any(i["check"] == "structureless_body" for i in validate_akn(_act("")))

    def test_list_items_do_not_count_as_structure(self):
        """`<item>` tracks prose reflow, so a wall of text split into items is still one."""
        items = "".join(f"<item><p>{self.JUDGMENT[:200]}</p></item>" for _ in range(60))
        xml = _act(f'<section eId="sec_1"><blockList>{items}</blockList></section>')
        assert any(i["check"] == "structureless_body" for i in validate_akn(xml))

    def test_a_quoted_amendment_is_not_counted_as_this_act_s_prose(self):
        """`count_provisions_by_kind` skips quoted provisions, so the length must
        skip their text too, or a correctly structured amending act is flagged."""
        quoted = "".join(
            f'<article eId="q_art_{n}"><num>{n}</num>'
            f"<content><p>{self.JUDGMENT[:600]}</p></content></article>"
            for n in range(1, 30)
        )
        xml = _act(
            '<article eId="art_1"><num>1</num><content><p>The Schedule is replaced by:'
            f"<mod><quotedStructure>{quoted}</quotedStructure></mod></p></content></article>"
        )
        assert not any(i["check"] == "structureless_body" for i in validate_akn(xml))

    def test_indentation_is_not_counted_as_text(self):
        """The character count is reader-facing, so it must measure prose."""
        paras = "\n      ".join(f"<p>Paragraph {n} of the judgment.</p>" for n in range(300))
        xml = _act(
            f'<section eId="sec_1">\n      <content>\n      {paras}\n      </content>\n    </section>'
        )
        issues = [i for i in validate_akn(xml) if i["check"] == "structureless_body"]
        if issues:
            prose = 300 * len("Paragraph 000 of the judgment.") + 300
            assert issues[0]["chars"] < prose * 1.1


class TestEidUnusable:
    """An eId is the addressing key. One that nests past real hierarchy is a flat
    list structured as a chain of children, and cannot be cited or compared."""

    def test_ordinary_nesting_is_not_flagged(self):
        xml = _act('<section eId="sec_1"><paragraph eId="sec_1__para_1__point_3"/></section>')
        assert not any(i["check"] == "eid_unusable" for i in validate_akn(xml))

    def test_a_chain_past_the_depth_ceiling_is_an_error(self):
        deep = "att_1" + "".join(f"__point_{n}" for n in range(1, 40))
        issues = [
            i for i in validate_akn(_act(f'<point eId="{deep}"/>')) if i["check"] == "eid_unusable"
        ]
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["depth"] == 40

    def test_a_long_eid_is_an_error_even_within_the_depth_ceiling(self):
        long = "att_1__point_" + "9" * 300
        issues = [
            i for i in validate_akn(_act(f'<point eId="{long}"/>')) if i["check"] == "eid_unusable"
        ]
        assert len(issues) == 1
        assert issues[0]["length"] == len(long)

    def test_one_finding_per_document_not_one_per_link(self):
        """A chain trips the ceiling at every link below it; 400 copies of one
        defect would bury every other finding the agent needs to see."""
        nodes = "".join(
            f'<point eId="att_1{"".join(f"__point_{n}" for n in range(1, d))}"/>'
            for d in range(20, 30)
        )
        issues = [i for i in validate_akn(_act(nodes)) if i["check"] == "eid_unusable"]
        assert len(issues) == 1

    def test_the_finding_carries_the_whole_eid_however_long(self):
        """The validation panel jumps the reader to `eid`. Truncating it for display
        left the finding addressing a provision that does not exist."""
        long = "att_1__point_" + "9" * 300
        issues = [
            i for i in validate_akn(_act(f'<point eId="{long}"/>')) if i["check"] == "eid_unusable"
        ]
        assert issues[0]["eid"] == long
        assert issues[0]["length"] == len(long)

    def test_the_message_names_the_ceiling_that_fired(self):
        """A 2-deep eId of 313 characters is not a nested chain, and saying so sent a
        reader looking for nesting that is not there. Both measurements are reported;
        only the one that fired is named, and the chain diagnosis follows depth."""
        long = "att_1__point_" + "9" * 300
        msg = next(
            i for i in validate_akn(_act(f'<point eId="{long}"/>')) if i["check"] == "eid_unusable"
        )["message"]
        assert "too long" in msg and "too deep" not in msg
        assert "chain of children" not in msg
        assert "2 levels" in msg and "313 characters" in msg

        deep = "att_1" + "".join(f"__point_{n}" for n in range(1, 40))
        deep_msg = next(
            i for i in validate_akn(_act(f'<point eId="{deep}"/>')) if i["check"] == "eid_unusable"
        )["message"]
        assert "too deep" in deep_msg
        assert "chain of children" in deep_msg


class TestProvenanceCalibration:
    """Where the structure came from decides what a date or a numbering hole means."""

    def _checks(self, date: str, uri: str, provenance: str | None) -> set[str]:
        return {
            i["check"]
            for i in validate_akn(_identified(date, uri), provenance=provenance)  # type: ignore[arg-type]
            if i["check"].startswith("identity_")
        }

    def test_a_native_pre_1800_act_is_not_implausible(self):
        # An old statute book holds instruments from before 1800 in its own markup.
        assert self._checks("1766-01-01", "/akn/xz/act/1766/7", "native") == set()

    def test_a_native_year_in_the_hijri_band_is_not_unconverted(self):
        assert self._checks("1431-01-01", "/akn/xz/act/1431/7", "native") == set()

    def test_native_keeps_the_arithmetic_checks(self):
        assert self._checks("2037-05-12", "/akn/xz/act/2037/11", "native") == {
            "identity_year_in_future"
        }
        assert self._checks("2007-05-12", "/akn/xz/act/2009/11", "native") == {
            "identity_year_uri_mismatch"
        }

    def test_extracted_and_unstated_still_flag_both(self):
        for provenance in ("extracted", None):
            assert self._checks("1766-01-01", "/akn/xz/act/1766/7", provenance) == {
                "identity_year_implausible"
            }
            assert self._checks("1431-01-01", "/akn/xz/act/1431/7", provenance) == {
                "identity_year_unconverted_hijri"
            }


def _numbered(*nums: int, kind: str = "section") -> str:
    return _act(
        "".join(
            f'<{kind} eId="{kind[:3]}_{n}"><num>{n}</num><content><p>t</p></content></{kind}>'
            for n in nums
        )
    )


def _gaps(xml: str, **kwargs: object) -> list[tuple[str, str, list[int]]]:
    return [
        (i["severity"], i["likely"], i["range"])
        for i in validate_akn(xml, **kwargs)  # type: ignore[arg-type]
        if i["check"] == "number_gap"
    ]


class TestNumberGapCalibration:
    def test_a_leading_gap_in_extracted_text_warns(self):
        assert _gaps(_numbered(4, 5, 6), provenance="extracted") == [("warning", "leading", [1, 3])]

    def test_a_leading_gap_in_articles_warns(self):
        assert _gaps(_numbered(2, 3, kind="article"), provenance="extracted") == [
            ("warning", "leading", [1, 1])
        ]

    def test_numbering_from_one_has_no_leading_gap(self):
        assert _gaps(_numbered(1, 2, 3), provenance="extracted") == []

    def test_a_leading_gap_is_not_read_from_native_or_unstated_structure(self):
        assert _gaps(_numbered(4, 5, 6), provenance="native") == []
        assert _gaps(_numbered(4, 5, 6)) == []

    def test_a_single_hole_in_extracted_text_warns(self):
        # One missing number reads as a repeal, which is info elsewhere.
        assert _gaps(_numbered(1, 2, 4, 5, 6, 7, 8, 9, 10), provenance="extracted") == [
            ("warning", "repeal", [3, 3])
        ]

    def test_a_single_hole_stays_info_when_provenance_is_native_or_unstated(self):
        xml = _numbered(1, 2, 4, 5, 6, 7, 8, 9, 10)
        assert _gaps(xml, provenance="native") == [("info", "repeal", [3, 3])]
        assert _gaps(xml) == [("info", "repeal", [3, 3])]

    def test_a_remarked_lost_page_keeps_the_gap_after_it_from_reading_as_a_repeal(self):
        xml = _numbered(1, 2, 4, 5, 6, 7, 8, 9, 10).replace(
            '<section eId="sec_2"><num>2</num><content><p>t</p>',
            '<section eId="sec_2"><num>2</num><content><p>t</p>'
            '<p><remark status="editorial">[Page 5 of the source could not be read]</remark></p>',
        )
        found = [i for i in validate_akn(xml) if i["check"] == "number_gap"]
        assert [(i["severity"], i["likely"]) for i in found] == [("warning", "unreadable_page")]
        assert "unreadable page(s) 5" in found[0]["message"], found[0]["message"]

    def test_a_lost_page_elsewhere_does_not_relabel_an_unrelated_gap(self):
        xml = _numbered(1, 2, 4, 5, 6, 7, 8, 9, 10).replace(
            '<section eId="sec_9"><num>9</num><content><p>t</p>',
            '<section eId="sec_9"><num>9</num><content><p>t</p>'
            '<p><remark status="editorial">[Page 5 of the source could not be read]</remark></p>',
        )
        found = [
            i
            for i in validate_akn(xml, unreadable_pages={5: "empty_read"})
            if i["check"] == "number_gap"
        ]
        assert [(i["severity"], i["likely"]) for i in found] == [("info", "repeal")]


class TestUnreadablePagesAndBodyFill:
    def test_unreadable_pages_block(self):
        found = [
            i
            for i in validate_akn(
                _numbered(1, 2), unreadable_pages={5: "content_filter: RECITATION", 9: "empty_read"}
            )
            if i["check"] == "unreadable_page"
        ]
        assert [(i["severity"], i["pages"]) for i in found] == [("error", [5, 9])]
        assert found[0]["reasons"] == {"5": "content_filter: RECITATION", "9": "empty_read"}

    def test_no_unreadable_pages_no_finding(self):
        assert not [
            i
            for i in validate_akn(_numbered(1, 2), unreadable_pages={})
            if "unreadable" in i["check"]
        ]

    def test_bodies_the_model_never_wrote_are_graded(self):
        trace = {
            "windows": 2,
            "calls": 4,
            "calls_failed": 2,
            "expected": 5,
            "verbatim": ["sec_2", "sec_3"],
            "empty": ["sec_4"],
        }
        found = {
            i["check"]: (i["severity"], i["eids"])
            for i in validate_akn(_numbered(1, 2, 3, 4, 5), body_fill=trace)
            if i["check"].startswith("body_fill")
        }
        assert found == {
            "body_fill_incomplete": ("error", ["sec_4"]),
            "body_fill_verbatim": ("warning", ["sec_2", "sec_3"]),
        }

    def test_a_complete_body_fill_is_silent(self):
        trace = {"windows": 1, "calls": 1, "calls_failed": 0, "expected": 2}
        assert not [
            i
            for i in validate_akn(_numbered(1, 2), body_fill=trace)
            if i["check"].startswith("body_fill")
        ]
