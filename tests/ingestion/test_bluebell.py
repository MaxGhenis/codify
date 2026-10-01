"""Unit tests for Bluebell parser wrapper."""

from collections import Counter
from pathlib import Path

import pytest
from lxml import etree

from codify.akn.bluebell import akn_xml_to_bluebell
from codify.pipeline.enrich.bluebell import parse_to_akn

# The package root before the monorepo root: an sdist unpacked inside a checkout
# must skip rather than borrow fixtures the checkout has and it does not.
_HERE = Path(__file__).resolve()
_ROOTS = [_HERE.parents[i] for i in (2, 4) if i < len(_HERE.parents)]
_SYNTHETIC = next(
    (c for r in _ROOTS if (c := r / "data" / "fixtures" / "synthetic").is_dir()),
    _ROOTS[0] / "data" / "fixtures" / "synthetic",
)
_needs_fixtures = pytest.mark.skipif(
    not _SYNTHETIC.is_dir(), reason="synthetic fixtures are not in this tree"
)
_FIXTURE = _SYNTHETIC / "xa" / "legislation-hcontainer.akn.xml"


def _text_char_counts(xml: str) -> Counter:
    """Non-space character multiset of an AKN doc's text content (tags and
    attributes excluded), so the metric measures content, not structure."""
    root = etree.fromstring(xml.encode("utf-8"))
    text = "".join(root.itertext())
    return Counter(c for c in text if not c.isspace())


def test_parse_simple_act():
    """Parse a simple Bluebell text into valid AKN XML."""
    bluebell_text = """SECTION 1 - Definitions

  In this Act, \"term\" means a defined concept.

SECTION 2 - Purpose

  The purpose of this Act is to regulate.
"""
    xml = parse_to_akn(
        bluebell_text=bluebell_text,
        country="xa",
        doctype="act",
        date="2024",
        number="1",
    )

    assert "<akomaNtoso" in xml
    assert 'xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"' in xml
    assert "<act" in xml
    assert "<body" in xml
    assert "<meta" in xml
    assert "Definitions" in xml
    assert "Purpose" in xml


def test_parse_strips_control_chars():
    """Control chars from OCR/LLM output reach the parser and crash lxml; the
    wrapper strips them at the boundary so the parse still succeeds."""
    bluebell_text = "SECTION 1 - Scope\n\n  Body \x00with a \x07NUL and bell.\n"
    xml = parse_to_akn(bluebell_text=bluebell_text, country="xa", doctype="act", number="1")
    assert "\x00" not in xml and "\x07" not in xml
    assert "Body with a NUL and bell." in xml


def test_parse_with_chapters():
    """Parse Bluebell text with chapters and sections."""
    bluebell_text = """CHAPTER 1 - Preliminary

  SECTION 1 - Short title

    This Act is the Test Act.

CHAPTER 2 - Main provisions

  SECTION 2 - Application

    This Act applies to all persons.
"""
    xml = parse_to_akn(
        bluebell_text=bluebell_text,
        country="xa",
        date="2024",
        number="42",
    )

    assert "<chapter" in xml
    assert "Preliminary" in xml
    assert "Main provisions" in xml


def test_parse_empty_text_does_not_crash():
    """Bluebell should handle empty input gracefully."""
    xml = parse_to_akn(
        bluebell_text="",
        country="xa",
        date="2024",
        number="1",
    )
    assert "<akomaNtoso" in xml


def test_frbr_uri_in_output():
    """The FRBR URI should appear in the generated XML."""
    xml = parse_to_akn(
        bluebell_text="SECTION 1 - Test\n\n  Content here.\n",
        country="xa",
        date="2023",
        number="5",
    )
    assert "/akn/xa/act/2023/5" in xml


@_needs_fixtures
def test_akn_bluebell_roundtrip_preserves_schedules():
    """AKN -> Bluebell -> AKN must not drop content. Regression guard for the
    bug where `akn_xml_to_bluebell` emitted nothing for `hcontainer` bodies,
    so whole schedules silently vanished before reaching the translator.

    Uses a synthetic Indigo-style fixture rich in `hcontainer`s (the unstructured
    body form our own Bluebell never emits, but imported AKN does); asserts the
    round-trip reparses cleanly and retains ~all non-space characters, and that a
    known line inside an hcontainer body survives into the Bluebell.
    """
    source_xml = _FIXTURE.read_text()
    assert "hcontainer" in source_xml  # fixture really exercises the path

    bluebell = akn_xml_to_bluebell(source_xml)
    # A line that lives inside an hcontainer body, would be absent pre-fix.
    assert "framework for the publication" in " ".join(bluebell.split())

    # Reparse must succeed (parse_to_akn raises on failure) ...
    reparsed_xml = parse_to_akn(
        bluebell, country="xa", doctype="act", date="1992", number="7", language="eng"
    )
    # ... and retain ~all the source's text-content characters.
    src, got = _text_char_counts(source_xml), _text_char_counts(reparsed_xml)
    retained = sum((src & got).values()) / max(sum(src.values()), 1)
    assert retained >= 0.97, f"round-trip lost content: retention={retained:.3f}"


def test_akn_bluebell_roundtrip_preserves_tables():
    """A `<table>` must survive AKN -> Bluebell -> AKN byte-identical, not
    flatten to prose: that flattening is what breaks the source pane, which
    re-derives Bluebell from stored AKN on every reopen (issue #1168)."""
    bluebell_text = """ARTICLE 1 - Schedule

  The following schedule:

  TABLE
    TR
      TH{colspan 2}
        Fee schedule
    TR
      TH
        Item
      TH
        Fee
    TR
      TC{rowspan 2}
        Filing
      TC
        10
    TR
      TC
        20
"""
    xml1 = parse_to_akn(bluebell_text, country="xa", doctype="act", date="2024", number="1")
    assert "<table" in xml1

    reparsed = akn_xml_to_bluebell(xml1)
    assert "TABLE" in reparsed
    assert "TH{colspan 2}" in reparsed
    assert "TC{rowspan 2}" in reparsed

    xml2 = parse_to_akn(reparsed, country="xa", doctype="act", date="2024", number="1")
    table1 = xml1[xml1.index("<table") : xml1.index("</table>") + len("</table>")]
    table2 = xml2[xml2.index("<table") : xml2.index("</table>") + len("</table>")]
    assert table1 == table2, "table structure must round-trip byte-identical"


def test_akn_bluebell_roundtrip_survives_an_adjacent_edit():
    """Editing a neighbouring provision (the source-pane workflow) and
    reparsing must leave an unrelated table's structure untouched."""
    bluebell_text = """ARTICLE 1 - Intro

  Original text.

ARTICLE 2 - Schedule

  TABLE
    TR
      TH
        Item
      TH
        Fee
    TR
      TC
        Filing
      TC
        10
"""
    xml1 = parse_to_akn(bluebell_text, country="xa", doctype="act", date="2024", number="1")
    edited = akn_xml_to_bluebell(xml1).replace("Original text.", "Edited text.")
    xml2 = parse_to_akn(edited, country="xa", doctype="act", date="2024", number="1")

    table1 = xml1[xml1.index("<table") : xml1.index("</table>") + len("</table>")]
    table2 = xml2[xml2.index("<table") : xml2.index("</table>") + len("</table>")]
    assert table1 == table2
    assert "Edited text." in xml2


def test_source_map_anchors_a_table_and_its_cells():
    """A `TABLE` line and each cell keyword line get their own anchor, so
    linked scroll doesn't lerp the whole table as uniform-density prose. A
    numbered marker after the table must still pair with its own paragraph.
    """
    from codify.akn.bluebell import bluebell_to_akn

    bluebell_text = """ARTICLE 1 - Schedule

  (1) before the table

  TABLE
    TR
      TH
        Item
      TH
        Fee
    TR
      TC
        Filing
      TC
        10

  (2) after the table
"""
    _akn, anchors, errors = bluebell_to_akn(
        bluebell_text, country="xa", doctype="act", date="2024", number="1"
    )
    assert errors == []
    kinds_by_eid = {a.eid: a.kind for a in anchors}
    assert kinds_by_eid["art_1__table_1"] == "table"
    # one anchor per cell keyword line: TH, TH, TC, TC
    cell_lines = {
        a.line
        for a in anchors
        if a.kind == "p" and "table_1__p" in a.eid and a.line in {7, 9, 12, 14}
    }
    assert cell_lines == {7, 9, 12, 14}
    # row keyword lines anchor to each row's first cell to compress table-only
    # source rows that have no dedicated AKN element in the preview.
    row_anchor_pairs = {(a.line, a.eid) for a in anchors if a.line in {6, 11}}
    assert (6, "art_1__table_1__p_1") in row_anchor_pairs
    assert (11, "art_1__table_1__p_3") in row_anchor_pairs
    # the table's four cells must not shift the numbered marker after it
    after = next(a for a in anchors if a.eid == "art_1__p_2")
    assert after.line == 17


def test_source_map_a_cell_with_no_paragraph_does_not_shift_later_cells():
    """`table_cell_p_eids` must carry one slot per cell, even a `<p>`-less one,
    aligned with `cell_lines`. Built directly against `_build_source_map`:
    bluebell always emits an empty `<p>` for a blank content line, so a real
    `<p>`-less cell needs hand-built AKN to exercise.
    """
    from codify.akn.bluebell import _build_source_map

    text = """ARTICLE 1 - Schedule

  TABLE
    TR
      TH
        Item
      TH
        Fee
    TR
      TC
        Filing
      TC
    TR
      TC
        Renewal
      TC
        20
"""
    # Middle cell (row 2's second TC) has no <p> at all, matching the source's
    # blank cell; the third row's cells still carry their real eids.
    akn = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act name="act"><body><article eId="art_1"><content>
    <table eId="art_1__table_1">
      <tr><th><p eId="art_1__table_1__p_1">Item</p></th>
          <th><p eId="art_1__table_1__p_2">Fee</p></th></tr>
      <tr><td><p eId="art_1__table_1__p_3">Filing</p></td>
          <td/></tr>
      <tr><td><p eId="art_1__table_1__p_4">Renewal</p></td>
          <td><p eId="art_1__table_1__p_5">20</p></td></tr>
    </table>
  </content></article></body></act>
</akomaNtoso>"""
    anchors = _build_source_map(text, akn)
    pairs = {(a.eid, a.line) for a in anchors if a.kind == "p"}
    assert ("art_1__table_1__p_4", 10) not in pairs, "must not point at the next cell"
    assert ("art_1__table_1__p_5", 16) in pairs, "last cell's content must still be anchored"


def test_source_map_ignores_a_marker_like_number_inside_a_table_cell():
    """A cell body that happens to start "(1)" must not be read as a real
    enumerated paragraph marker: since table cell eids are excluded from
    `flat_para_eids`, a false match here would steal the zip slot meant for
    the next real numbered paragraph and shift everything after it."""
    bluebell_text = """ARTICLE 1 - Schedule

  TABLE
    TR
      TH
        Item
      TH
        Note
    TR
      TC
        Filing
      TC
        (1) see the general fee schedule

  (1) after the table
"""
    from codify.akn.bluebell import bluebell_to_akn

    _akn, anchors, errors = bluebell_to_akn(
        bluebell_text, country="xa", doctype="act", date="2024", number="1"
    )
    assert errors == []
    after = next(a for a in anchors if a.eid == "art_1__p_1")
    assert after.kind == "paragraph"


def test_persian_digits_in_num_normalise_to_ascii_in_eids():
    """Arabic-script OCR sometimes lands Persian (U+06F0-06F9) digits in `<num>`
    text. eIds must fold to ASCII so cross-refs stay resolvable regardless of
    which script the source spelled the number in."""
    bluebell = """ACT
BODY
  ARTICLE ۱۳۲
    HEADING
      Persian-digit article
    Body content.

  ARTICLE ١٣٣
    HEADING
      Arabic-Indic-digit article
    Body content.
"""
    akn = parse_to_akn(bluebell, country="xz", doctype="act", date="2024", number="1")
    import re

    eids = re.findall(r'eId="([^"]+)"', akn)
    for e in eids:
        assert all(ord(c) < 128 for c in e), f"eId {e!r} contains non-ASCII digits"
    assert 'eId="art_132"' in akn
    assert 'eId="art_133"' in akn
    # Display fidelity: the <num> text keeps its original script.
    assert "<num>۱۳۲</num>" in akn
    assert "<num>١٣٣</num>" in akn


def test_eid_digit_collision_disambiguates():
    """Persian ۱۳۲ and Arabic-Indic ١٣٢ both fold to ASCII 132. The post-pass
    mirrors Bluebell's _2 suffix scheme on collision instead of silently merging."""
    bluebell = """ACT
BODY
  ARTICLE ۱۳۲
    Persian version.

  ARTICLE ١٣٢
    Arabic-Indic version.
"""
    akn = parse_to_akn(bluebell, country="xz", doctype="act", date="2024", number="1")
    import re

    eids = re.findall(r'eId="(art_\d[^"]*)"', akn)
    assert "art_132" in eids
    assert "art_132_2" in eids, f"expected collision suffix, got {eids}"


@_needs_fixtures
def test_parse_id_simple_uu_bab_only():
    """Asserts the base Bahasa hierarchy (CHAPTER, ARTICLE, PARAGRAPH, POINT)
    parses through Bluebell without the higher divisions (BOOK, PART, DIVISION)
    the deep-hierarchy test also exercises.
    """
    fixture = _SYNTHETIC / "xl" / "keterbukaan-informasi-2018.bluebell"
    xml = parse_to_akn(
        bluebell_text=fixture.read_text(),
        country="xl",
        doctype="act",
        date="2018",
        number="14",
        language="ind",
    )
    assert "<akomaNtoso" in xml
    assert "/akn/xl/act/2018/14" in xml
    assert "<chapter" in xml
    assert "<article" in xml
    assert "<paragraph" in xml
    assert "<point" in xml
    assert "KETENTUAN UMUM" in xml
    assert "Informasi Publik" in xml


@_needs_fixtures
def test_parse_id_complex_uu_with_buku_bagian_paragraf():
    """Asserts the BOOK > CHAPTER > PART > DIVISION > ARTICLE cascade nests through
    Bluebell for the inverted Bahasa criminal-code hierarchy.

    Substring checks alone would pass a flattened tree, so this test
    additionally parses the output and confirms at least one nested cascade
    exists via XPath.
    """
    from codify.akn import AKN_NS

    fixture = _SYNTHETIC / "xl" / "criminal-code-2023.bluebell"
    xml = parse_to_akn(
        bluebell_text=fixture.read_text(),
        country="xl",
        doctype="act",
        date="2023",
        number="1",
        language="ind",
    )
    assert "<akomaNtoso" in xml
    assert "/akn/xl/act/2023/1" in xml
    assert "KITAB HUKUM PIDANA LANGKASUKA" in xml
    root = etree.fromstring(xml.encode("utf-8"))
    ns = {"a": AKN_NS}
    nested = root.xpath("//a:book//a:chapter//a:part//a:division//a:article", namespaces=ns)
    assert len(nested) >= 1, "inverted cascade did not nest under book"


@_needs_fixtures
def test_parse_id_amendment_act_roman_articles():
    """Asserts UU 15/2019's Roman-numeral Pasal I / Pasal II land as AKN
    `<article>` elements, per the profile's §Amendment Patterns claim.
    """
    fixture = _SYNTHETIC / "xl" / "keterbukaan-informasi-perubahan-2023.bluebell"
    xml = parse_to_akn(
        bluebell_text=fixture.read_text(),
        country="xl",
        doctype="act",
        date="2023",
        number="3",
        language="ind",
    )
    assert "<akomaNtoso" in xml
    assert "/akn/xl/act/2023/3" in xml
    assert "<article" in xml
    assert "Perubahan atas" in xml
    # Both Roman-numeral articles must land in the tree.
    assert xml.count("<article") >= 2


def test_eid_digit_collision_against_existing_ascii_eid():
    """Direct corpus-repair regression: an already-ASCII `art_7` PLUS a Persian
    `art_۷` that normalises to the same string. Pre-fix the normaliser silently
    let both keep `art_7`, surfacing 29 eid_uniqueness violations during the
    2026-06-23 corpus repair dry-run. Now the non-ASCII form must take the
    `_2` slot.
    """
    from lxml import etree

    from codify.akn import AKN_NS
    from codify.pipeline.enrich.bluebell import normalise_eid_digits

    xml = f'''<akomaNtoso xmlns="{AKN_NS}">
      <act><body>
        <article eId="art_7"><num>7</num></article>
        <article eId="art_۷"><num>۷</num></article>
      </body></act>
    </akomaNtoso>'''
    root = etree.fromstring(xml.encode())
    normalise_eid_digits(root)
    eids = sorted([el.get("eId") for el in root.iter() if el.get("eId")])
    assert eids == ["art_7", "art_7_2"], f"got {eids}"
    # Both eIds are now distinct, no eId_uniqueness violation
    eid_counter: dict[str, int] = {}
    for el in root.iter():
        e = el.get("eId")
        if e:
            eid_counter[e] = eid_counter.get(e, 0) + 1
    assert all(c == 1 for c in eid_counter.values()), f"dupes: {eid_counter}"


def test_marker_punctuation_never_reaches_an_eid():
    """A marker carrying a trailing stop, parentheses and a non-Latin digit names
    the same provision as the bare number, so none of it may reach the identifier.
    The same law in two languages was minting different eId sets on this alone.
    """
    from lxml import etree

    from codify.akn import AKN_NS
    from codify.pipeline.enrich.bluebell import normalise_eid_digits

    xml = f"""<akomaNtoso xmlns="{AKN_NS}">
      <act><body>
        <article eId="art_1"><num>1</num>
          <point eId="art_1__point_1."><num>1.</num><content><p>a</p></content></point>
          <point eId="art_1__point_(2)"><num>(2)</num><content><p>b</p></content></point>
          <point eId="art_1__point_(٣)"><num>(٣)</num><content><p>c</p></content></point>
          <point eId="art_1__point_4-"><num>4-</num><content><p>d</p></content></point>
        </article>
      </body></act>
    </akomaNtoso>"""
    root = etree.fromstring(xml.encode())
    normalise_eid_digits(root)
    eids = sorted(el.get("eId") for el in root.iter() if el.get("eId"))
    assert eids == [
        "art_1",
        "art_1__point_1",
        "art_1__point_2",
        "art_1__point_3",
        "art_1__point_4",
    ], f"got {eids}"


def test_punctuation_fold_suffixes_rather_than_merges():
    """`point_1` and `point_1.` fold to one string, so the fold must disambiguate
    or two distinct provisions become one identifier. Cross-references follow the
    point that moved."""
    from lxml import etree

    from codify.akn import AKN_NS
    from codify.pipeline.enrich.bluebell import normalise_eid_digits

    xml = f"""<akomaNtoso xmlns="{AKN_NS}">
      <act><body>
        <article eId="art_1"><num>1</num>
          <point eId="art_1__point_1"><num>1</num><content><p>a</p></content></point>
          <point eId="art_1__point_1."><num>1.</num><content><p>b</p></content></point>
          <p><ref href="#art_1__point_1.">see</ref></p>
        </article>
      </body></act>
    </akomaNtoso>"""
    root = etree.fromstring(xml.encode())
    normalise_eid_digits(root)
    eids = sorted(el.get("eId") for el in root.iter() if el.get("eId"))
    assert eids == ["art_1", "art_1__point_1", "art_1__point_1_2"], f"got {eids}"
    refs = [el.get("href") for el in root.iter() if el.get("href")]
    assert refs == ["#art_1__point_1_2"], f"got {refs}"


def test_every_terminator_the_point_parser_accepts_is_trimmed():
    """The parser takes U+2010 to U+2015 as marker terminators, so all six must
    fold. Covering only the two common dashes left four minting punctuated eIds."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for dash in (chr(c) for c in range(0x2010, 0x2016)):
        assert _ascii_fold_eid(f"art_1__point_4{dash}") == "art_1__point_4"
    for terminator in ")", ".", ":", "-":
        assert _ascii_fold_eid(f"art_1__point_4{terminator}") == "art_1__point_4"


def test_one_marker_letter_folds_in_every_form_it_arrives_in():
    """Bare, bracketed, carrying the decorative kashida the point parser allows,
    or carrying Bluebell's dedup suffix. Two letters are not a marker index and
    stay put, which is what keeps the backfill selector honest."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("art_1__point_\u0623") == "art_1__point_a"
    assert _ascii_fold_eid("art_1__point_(\u0623)") == "art_1__point_a"
    assert _ascii_fold_eid("art_1__point_\u0647\u0640") == "art_1__point_e"
    assert _ascii_fold_eid("art_1__point_(\u0623)_2") == "art_1__point_a_2"
    assert _ascii_fold_eid("art_1__point_\u0623\u0628") == "art_1__point_\u0623\u0628"


def test_the_collision_suffix_does_not_follow_document_order():
    """Documented, not endorsed. Whichever eId the fold leaves unchanged keeps the
    bare name, so a punctuated point ahead of a bare one takes `_2` while sitting
    first. Nothing merges and references resolve. Changing it would move eIds the
    digit path already assigns across the corpus, so it is not changed here."""
    from lxml import etree

    from codify.akn import AKN_NS
    from codify.pipeline.enrich.bluebell import normalise_eid_digits

    def eids(*nums: str) -> list[str]:
        points = "".join(
            f'<point eId="art_1__point_{n}"><num>{n}</num><content><p>{n}</p></content></point>'
            for n in nums
        )
        root = etree.fromstring(
            f'<akomaNtoso xmlns="{AKN_NS}"><act><body>'
            f'<article eId="art_1"><num>1</num>{points}</article>'
            f"</body></act></akomaNtoso>".encode()
        )
        normalise_eid_digits(root)
        return [
            el.get("eId") for el in root.iter() if (el.get("eId") or "").startswith("art_1__point")
        ]

    assert eids("1", "1.") == ["art_1__point_1", "art_1__point_1_2"]
    assert eids("1.", "1") == ["art_1__point_1_2", "art_1__point_1"]


def test_a_range_and_a_dedup_suffix_survive_the_fold():
    """Only the edges are trimmed: `1-2` is a range and `1_2` is Bluebell's own
    disambiguator, and trimming either would rename a provision."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("art_1__point_1-2") == "art_1__point_1-2"
    assert _ascii_fold_eid("art_1__point_1_2") == "art_1__point_1_2"
    assert _ascii_fold_eid("art_6bis") == "art_6bis"


def test_arabic_ordinal_container_eids_fold_to_latin():
    """Arabic ordinal chapters must not
    produce `chp_الأول` eIds; the ordinal word folds to its integer while
    the displayed `<num>` keeps the source script."""
    text = (
        "BODY\n"
        "  CHAPTER الأول - أحكام عامة\n"
        "    ARTICLE 1\n"
        "      نص المادة.\n"
        "  CHAPTER الحادي عشر - أحكام ختامية\n"
        "    ARTICLE 2\n"
        "      نص المادة.\n"
    )
    akn = parse_to_akn(
        text, country="xz", doctype="act", date="2004-01-01", number="39", language="ara"
    )
    assert 'eId="chp_1"' in akn
    assert 'eId="chp_11"' in akn
    assert "chp_الأول" not in akn
    # Displayed <num> stays in source script.
    assert "<num>الأول</num>" in akn


def test_duplicate_ordinal_chapters_fold_to_ascii_with_suffix():
    """Bluebell disambiguates a repeated `CHAPTER الأول` as `chp_الأول_2`;
    the fold must handle the suffixed shape so no non-ASCII eId ships."""
    text = (
        "BODY\n"
        "  CHAPTER الأول - أحكام عامة\n"
        "    ARTICLE 1\n"
        "      نص المادة.\n"
        "  CHAPTER الأول - أحكام مكررة\n"
        "    ARTICLE 2\n"
        "      نص المادة.\n"
    )
    akn = parse_to_akn(
        text, country="xz", doctype="act", date="2004-01-01", number="39", language="ara"
    )
    import re as _re

    eids = _re.findall(r'eId="([^"]+)"', akn)
    non_ascii = [e for e in eids if not e.isascii()]
    assert non_ascii == [], non_ascii
    assert 'eId="chp_1"' in akn


def test_ordinal_fold_collision_with_existing_ascii_chapter_disambiguates():
    """`CHAPTER الأول` folding to chp_1 while a literal `CHAPTER 1` exists
    must suffix, not merge."""
    text = (
        "BODY\n"
        "  CHAPTER 1 - General\n"
        "    ARTICLE 1\n"
        "      Text.\n"
        "  CHAPTER الأول - أحكام عامة\n"
        "    ARTICLE 2\n"
        "      نص المادة.\n"
    )
    akn = parse_to_akn(
        text, country="xz", doctype="act", date="2004-01-01", number="39", language="ara"
    )
    assert 'eId="chp_1"' in akn
    assert 'eId="chp_1_2"' in akn
    # Descendants of the suffixed chapter carry the suffixed prefix, not
    # a fold under the wrong parent.
    assert 'eId="chp_1_2__art_2"' in akn
    assert 'eId="chp_1__art_2"' not in akn


def test_arabic_end_to_end_scan_scaffold_assemble_chapter_heading():
    """Full chain: scan → scaffold → assemble → Bluebell on an Arabic law
    sample with a next-line chapter title. Pins the ` - ` separator
    convention, the assemble fallback for containers, RTL heading text,
    and the ordinal eId fold as one contract."""
    from codify.jurisdictions import load_config
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors
    from codify.pipeline.enrich.scaffold import (
        BodyBlock,
        BodyFillResponse,
        assemble_filled_scaffold,
        scaffold_from_anchors,
    )

    text = (
        "الفصل الأول\nأحكام عامة\n\nالمادة ١\nنص المادة الأولى.\n\nالمادة ٢\nنص المادة الثانية.\n"
    )
    config = load_config("xz")
    assert config is not None
    regex = build_anchor_regex(config, "act")
    anchors = scan_anchors(text, regex, country="xz", doctype="act")
    skeleton, index = scaffold_from_anchors(anchors)
    response = BodyFillResponse(
        bodies=[
            BodyBlock(eid=eid, lines=["نص المادة."])
            for eid in index
            if index[eid].kind == "article"
        ]
    )
    filled = assemble_filled_scaffold(skeleton, index, response)
    akn = parse_to_akn(
        filled, country="xz", doctype="act", date="2020-01-01", number="1", language="ara"
    )
    assert 'eId="chp_1"' in akn
    assert "<heading>أحكام عامة</heading>" in akn


def test_unfoldable_eid_logs_residual_warning():
    """An ordinal outside the table survives non-ASCII; the residual sweep
    must surface it rather than ship the naming violation silently."""
    from structlog.testing import capture_logs

    text = (
        "BODY\n"
        "  CHAPTER الأربعون - أحكام\n"
        "    ARTICLE 1\n"
        "      نص.\n"
        "  CHAPTER الأول - أحكام\n"
        "    ARTICLE 2\n"
        "      نص.\n"
    )
    with capture_logs() as logs:
        akn = parse_to_akn(
            text, country="xz", doctype="act", date="2004-01-01", number="39", language="ara"
        )
    residual_events = [e for e in logs if e["event"] == "eid_non_ascii_residual"]
    assert residual_events, logs
    # Only the genuinely unfoldable eId is reported; the folded chp_1 and
    # its remapped-away original are not.
    assert all("chp_1" not in str(e.get("eids")) for e in residual_events)
    assert 'eId="chp_1"' in akn


def test_latin_ordinal_container_eids_fold_to_integers():
    """A Bahasa Bagian is numbered with an ordinal word. `part_Kesatu` is
    URL-safe, so the ASCII shortcut used to return it untouched and the anchor
    layer's `part_1` never reached the document."""
    text = (
        "BODY\n"
        "  PART Kesatu - Umum\n"
        "    SEC 3 - Ruang lingkup\n"
        "      Ketentuan umum berlaku.\n"
        "  PART Kedua - Perizinan\n"
        "    SEC 4 - Izin\n"
        "      Izin diberikan oleh Menteri.\n"
    )
    akn = parse_to_akn(
        text, country="xl", doctype="act", date="2024-01-01", number="1", language="ind"
    )
    assert 'eId="part_1"' in akn
    assert 'eId="part_2"' in akn
    assert "part_Kesatu" not in akn
    # Descendants carry the folded prefix, so a citation into them resolves.
    assert 'eId="part_1__sec_3"' in akn
    # Displayed <num> keeps the ordinal word the source used.
    assert "<num>Kesatu</num>" in akn


def test_compound_ordinal_eids_fold_despite_the_collapsed_space():
    """Bluebell builds the eId from `<num>` with spaces removed, so `Kedua Belas`
    arrives as `KeduaBelas` and never matches the declared two-word form."""
    text = "BODY\n  PART Kedua Belas - Ketentuan\n    SEC 3 - Lingkup\n      Isi ketentuan.\n"
    akn = parse_to_akn(
        text, country="xl", doctype="act", date="2024-01-01", number="1", language="ind"
    )
    assert 'eId="part_12"' in akn
    assert "KeduaBelas" not in akn
    assert "<num>Kedua Belas</num>" in akn


def test_ordinal_fold_leaves_unrelated_ascii_eids_alone():
    """The fold table is consulted per num segment, so ordinary numbering and
    letter markers must pass through untouched."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for eid in ("art_15", "art_15__para_2", "list_b", "chp_1", "sec_4__point_a"):
        assert _ascii_fold_eid(eid) == eid, eid
    # A Bluebell duplicate suffix rides along rather than blocking the fold.
    assert _ascii_fold_eid("part_Kesatu_2") == "part_1_2"


def test_a_mixed_script_eid_folds_both_halves():
    """An Arabic chapter over a Bahasa part: the script-specific pass leaves the
    ASCII ordinal alone, so a half-folded eId would be selected for repair forever."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("chp_الأول__part_Kesatu") == "chp_1__part_1"
    assert _ascii_fold_eid("chp_الأول__part_KeduaBelas") == "chp_1__part_12"
    # Each half still folds on its own.
    assert _ascii_fold_eid("chp_الأول") == "chp_1"
    assert _ascii_fold_eid("chp_II__part_Kesatu") == "chp_II__part_1"


def test_a_subtype_survives_the_round_trip() -> None:
    """The editor re-parses under the stored work; a subtype-carrying URI keeps it."""
    from codify.akn.bluebell import bluebell_to_akn

    akn, _anchors, errors = bluebell_to_akn(
        "SECTION 1\n\n  Text.\n",
        country="xq",
        doctype="act",
        date="2018",
        number="12",
        subtype="xpa",
    )
    assert errors == []
    assert 'value="/akn/xq/act/xpa/2018/12"' in akn


def test_a_marker_letter_with_no_abjad_rank_is_left_whole():
    """`points.py` accepts every Arabic letter as a marker, but only the abjad ranks
    fold. Trimming the brackets and stopping would read as repaired while still naming
    the provision in a script citations cannot match, so the segment is restored."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for eid in ("point_(\u0629)", "point_(\u0621)", "point_(\u0623\u0628)"):
        assert _ascii_fold_eid(eid) == eid, eid


def test_one_unfoldable_segment_does_not_block_the_rest():
    """The restore is per segment: a chapter that folds still folds beside a point
    that cannot, so a mixed eId is repaired as far as the tables reach."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("chp_\u0627\u0644\u0623\u0648\u0644__point_(\u0629)") == (
        "chp_1__point_(\u0629)"
    )


def test_a_compound_marker_is_left_whole_rather_than_half_trimmed():
    """`sec_1(a)` carries punctuation inside the num, not at its edges. Trimming the
    outer half alone leaves `sec_1(a`, an unbalanced eId naming nothing, and it is
    ASCII so the unfoldable restore never sees it. Runs on every parse, not the
    backfill, so a half-trim would rename provisions corpus-wide."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for eid in (
        "sec_1(a)",
        "sec_(1)(a)",
        "sec_(i)(a)",
        "sec_1(a)(i)",
        "sec_1.2.",
        "sec_(1.2)",
        "sec_(1-2)",
        "art_1__point_(1-3)",
        "art_1__point_2(b)",
        "sec_1(a)_dup2",
    ):
        assert _ascii_fold_eid(eid) == eid, eid


def test_both_collision_suffixes_survive_the_fold():
    """The digit fold mints `_2` and `ensure_unique_eids` mints `_dup2`. A fold that
    knew only one left `sec_1._dup2` unrepaired beside a repaired `sec_1.`."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("sec_(1)_2") == "sec_1_2"
    assert _ascii_fold_eid("sec_(1)_dup2") == "sec_1_dup2"
    assert _ascii_fold_eid("sec_1._dup2") == "sec_1_dup2"
    assert _ascii_fold_eid("art_1__point_(أ)_dup3") == "art_1__point_a_dup3"


def test_every_fold_branch_tolerates_a_collision_suffix():
    """The ordinal and bis branches read the whole num, so a `_dup2` on either left
    the eId non-ASCII while the selector, which reads the whole num too, booked the
    version: work run to no effect, reported as done."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for eid, want in (
        ("chp_الأول_dup2", "chp_1_dup2"),
        ("chp_الأول_2", "chp_1_2"),
        ("art_٦مكرر_dup2", "art_6bis_dup2"),
        ("art_1__point_أ_dup3", "art_1__point_a_dup3"),
    ):
        assert _ascii_fold_eid(eid) == want, eid


def test_a_bis_num_keeps_its_collision_suffix_out_of_the_index():
    """`art_6bis2` is a real anchor: bis index 2. Folding a dedup suffix into the
    index would name that provision from a different one."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    assert _ascii_fold_eid("art_٦مكرر_2") == "art_6bis_2"


def test_a_marker_letter_sheds_every_joiner_the_ordinal_lookup_sheds():
    """Kashida, ZWNJ and ZWJ are all decorative here, and the ordinal path already
    strips the set. A letter branch stripping only kashida left detection wider
    than the fold."""
    from codify.pipeline.enrich.bluebell import _ascii_fold_eid

    for joiner in ("ـ", "‌", "‍"):
        assert _ascii_fold_eid(f"art_1__point_أ{joiner}") == "art_1__point_a"


def test_every_kind_the_scanner_emits_survives_the_round_trip() -> None:
    """A kind missing from the keyword table takes its whole subtree with it.

    The emitter skips a tag it cannot name and skips its children too, so the
    text under it leaves the source pane, the export and the translation input
    with the document still non-empty and nothing raised.
    """
    from codify.akn.bluebell import (
        _STRUCTURAL_TO_KEYWORD,
        akn_xml_to_bluebell,
        parse_to_akn,
    )
    from codify.pipeline.enrich.kinds import BLUEBELL_HIER_KEYWORDS

    # `ART` is Bluebell's own abbreviation for the article keyword.
    named = set(_STRUCTURAL_TO_KEYWORD.values()) | {"ARTICLE"}
    assert not sorted(BLUEBELL_HIER_KEYWORDS - named)

    # Two grouping containers and two body-bearing subdivisions: the emitter
    # drops all four the same way, and only the containers were covered before.
    for kind in ("subdivision", "subpart", "alinea", "indent"):
        akn = (
            '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
            '<act name="act"><body>'
            '<part eId="part_1"><num>1</num><heading>Part One</heading>'
            f'<{kind} eId="part_1__grp_1"><num>1</num>'
            "<content><p>Inner text here.</p></content>"
            f"</{kind}></part></body></act></akomaNtoso>"
        )
        bluebell = akn_xml_to_bluebell(akn)
        assert "Inner text here." in bluebell, kind

        # The name promises a round trip, so parse it back: an emitter that
        # names a kind the parser cannot read loses the subtree anyway.
        back = parse_to_akn(bluebell, "xa", "act")
        assert "Inner text here." in back, kind


def test_every_keyword_the_emitter_writes_is_one_the_source_map_reads() -> None:
    """The pane's gutter and linked scrolling hang off the source map.

    A keyword the emitter writes but the parse table lacks yields no anchor for
    that line, so the pane silently loses its handle on those provisions while
    the text itself renders correctly.
    """
    from codify.akn.bluebell import (
        _BLUEBELL_TO_AKN,
        _STRUCTURAL_TO_KEYWORD,
        _build_source_map,
        akn_xml_to_bluebell,
    )

    unreadable = sorted(set(_STRUCTURAL_TO_KEYWORD.values()) - set(_BLUEBELL_TO_AKN))
    assert not unreadable

    akn = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        '<act name="act"><body>'
        '<part eId="part_1"><num>1</num><heading>Part One</heading>'
        '<subtitle eId="part_1__subtitle_1"><num>1</num><heading>Sub One</heading>'
        '<article eId="part_1__subtitle_1__art_1"><num>1</num>'
        "<content><p>Text.</p></content></article>"
        "</subtitle></part></body></act></akomaNtoso>"
    )
    anchors = _build_source_map(akn_xml_to_bluebell(akn), akn)
    assert [a.eid for a in anchors] == [
        "part_1",
        "part_1__subtitle_1",
        "part_1__subtitle_1__art_1",
    ]
