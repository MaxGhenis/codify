"""Keyword-less anchoring for document classes that declare a marker form."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codify.jurisdictions import HierarchyEntry, load_config
from codify.pipeline.enrich.anchors import (
    _marker_numbers,
    anchor_coverage,
    build_anchor_regex,
    scan_anchors_with_ambiguity,
)
from codify.pipeline.enrich.structure import basic_unit_kind

# A constitutional-court judgment's shape: captions carry a numeral that
# matches the leading digit of the paragraphs beneath them, and the body quotes
# statute freely, which latches the quote mask over most of the document.
JUDGMENT = """PUTUSAN
NOMOR 12/PUL-III/2029

[1.1] Yang mengadili permohonan pengujian undang-undang.

2. DUDUK PERKARA

[2.1] Menimbang bahwa Pemohon mengajukan permohonan.
[2.2] Menimbang bahwa Pemohon mendalilkan “Pasal 5 ayat (1)” sebagaimana [2.1] di atas.

3. PERTIMBANGAN HUKUM

[3.1] Menimbang bahwa Mahkamah berwenang mengadili.
[3.2] Menimbang bahwa “Setiap orang berhak atas pengakuan” menurut Pasal 28D.

4. KONKLUSI

[4.1] Berdasarkan penilaian atas fakta.

5. AMAR PUTUSAN

[5.1] Mengadili.
"""


def _scan(text: str, country: str, doctype: str):
    config = load_config(country)
    return scan_anchors_with_ambiguity(
        text, build_anchor_regex(config, doctype), country=country, doctype=doctype
    )


def test_captions_take_their_printed_numeral() -> None:
    sections = [a for a in _scan(JUDGMENT, "xl", "putusan").anchors if a.kind == "section"]
    assert [(a.number, a.heading) for a in sections] == [
        ("2", "DUDUK PERKARA"),
        ("3", "PERTIMBANGAN HUKUM"),
        ("4", "KONKLUSI"),
        ("5", "AMAR PUTUSAN"),
    ]


def test_paragraphs_nest_under_the_section_sharing_their_prefix() -> None:
    paragraphs = [a for a in _scan(JUDGMENT, "xl", "putusan").anchors if a.kind == "paragraph"]
    assert [a.akn_eid for a in paragraphs] == [
        "sec_2__para_2.1",
        "sec_2__para_2.2",
        "sec_3__para_3.1",
        "sec_3__para_3.2",
        "sec_4__para_4.1",
        "sec_5__para_5.1",
    ]


def test_quoted_text_does_not_suppress_judgment_paragraphs() -> None:
    # The mask latches on the curly quotes in 2.2, which would otherwise eat
    # every paragraph after it.
    numbers = [a.number for a in _scan(JUDGMENT, "xl", "putusan").anchors if a.kind == "paragraph"]
    assert {"3.1", "3.2", "4.1", "5.1"} <= set(numbers)


def test_mid_sentence_bracketed_decimal_does_not_anchor() -> None:
    numbers = [a.number for a in _scan(JUDGMENT, "xl", "putusan").anchors]
    assert numbers.count("2.1") == 1


def test_declared_markers_are_stamped() -> None:
    scan = _scan(JUDGMENT, "xl", "putusan")
    assert scan.fires["scan_declared_markers"] == 11
    assert {a.source_pass for a in scan.anchors} == {"declared_markers"}


def test_class_declaring_no_marker_form_is_untouched() -> None:
    # A fixture jurisdiction, so declaring a form on a live one cannot silently
    # invalidate the premise. This test has been through that twice already.
    country, doctype = "xe", "act"
    declared = load_config(country).get_document_class(doctype)
    assert declared is not None
    assert not [e for e in declared.hierarchy if e.marker_form]

    text = "Глава I\nОБЩИЕ ПОЛОЖЕНИЯ\n\nСтатья 1\nВ настоящем законе.\n\nСтатья 2\nДействует.\n"
    scan = _scan(text, country, doctype)
    assert "scan_declared_markers" not in scan.fires
    assert [a.kind for a in scan.anchors] == ["chapter", "article", "article"]


def test_caption_level_without_captions_is_rejected_at_load() -> None:
    with pytest.raises(ValidationError, match="declares no captions"):
        HierarchyEntry(
            local_term="Bagian", akn_element="section", level="higher", marker_form="caption"
        )


def test_captions_without_a_caption_marker_form_are_rejected() -> None:
    with pytest.raises(ValidationError, match="without marker_form"):
        HierarchyEntry(
            local_term="Bagian", akn_element="section", level="higher", captions=["KONKLUSI"]
        )


# The real [2.1] comes first; PDF extraction soft-wraps a citation of it onto
# its own line later, where it looks exactly like a marker.
WRAPPED_CITATION = """PUTUSAN

2. DUDUK PERKARA

[2.1] Menimbang bahwa Pemohon mengajukan permohonan pengujian.
[2.2] Menimbang bahwa dalil Pemohon sebagaimana
[2.1] di atas telah dipertimbangkan.
[2.3] Menimbang lebih lanjut.
"""


def test_wrapped_citation_does_not_displace_the_real_paragraph() -> None:
    scan = _scan(WRAPPED_CITATION, "xl", "putusan")
    at = {a.number: a.char_offset for a in scan.anchors if a.kind == "paragraph"}
    assert at["2.1"] == WRAPPED_CITATION.index("[2.1] Menimbang")
    assert [a.number for a in scan.anchors if a.kind == "paragraph"] == ["2.1", "2.2", "2.3"]


def test_sub_numbered_paragraphs_still_advance() -> None:
    # (3, 11) < (3, 11, 1) < (3, 11, 2) < (3, 12), so the monotonic rule keeps them.
    text = "3. KONKLUSI\n\n[3.11] Satu.\n[3.11.1] Dua.\n[3.11.2] Tiga.\n[3.12] Empat.\n"
    numbers = [a.number for a in _scan(text, "xl", "putusan").anchors if a.kind == "paragraph"]
    assert numbers == ["3.11", "3.11.1", "3.11.2", "3.12"]


def test_coverage_gate_measures_the_citable_paragraph_unit() -> None:
    # The gate was inert for judgments: `basic_unit_kind` picked the caption
    # sections and the denominator was empty, so nothing was checked.
    config = load_config("xl")
    assert basic_unit_kind(config, "putusan") == "paragraph"
    assert _marker_numbers(JUDGMENT, config, "putusan", "paragraph") == {
        "1.1",
        "2.1",
        "2.2",
        "3.1",
        "3.2",
        "4.1",
        "5.1",
    }


def test_coverage_ratio_reports_the_orphaned_opening_paragraph() -> None:
    config = load_config("xl")
    scan = _scan(JUDGMENT, "xl", "putusan")
    coverage = anchor_coverage(JUDGMENT, scan.anchors, config, "putusan", "paragraph")
    # [1.1] sits above the first caption and drops as an orphan; the gate now
    # says so rather than skipping on an empty denominator.
    assert coverage.expected - coverage.captured == {"1.1"}
    assert coverage.ratio is not None and coverage.ratio >= config.min_anchor_coverage


def test_keyword_driven_classes_keep_their_basic_unit() -> None:
    config = load_config("xl")
    assert basic_unit_kind(config, "act") == "article"
    assert basic_unit_kind(config, "pp") == "article"


def test_a_parenthesised_number_anchors_the_subdivision_it_marks() -> None:
    """The enacted subdivision of an article prints no keyword, only "(1) ".

    Keyword aliases match only where the word is spelled out, which in the
    operative text it never is, so without a marker form of its own the level
    anchors nowhere and its content flattens into the level below.
    """
    from codify.pipeline.enrich.anchors import _scan_declared_markers

    entry = HierarchyEntry(
        local_term="Ayat",
        akn_element="paragraph",
        level="subdivision",
        marker_form="parenthesized_arabic",
    )
    text = (
        "Pasal 5\n"
        "(1) Ketentuan pertama.\n"
        "(2) Ketentuan kedua sebagaimana dimaksud pada ayat (1).\n"
        "Pasal 6\n"
        "(1) Ketentuan lain.\n"
    )
    anchors, _ = _scan_declared_markers(text, 0, [entry])
    assert [a.number for a in anchors] == ["1", "2", "1"]
    assert all(a.kind == "paragraph" for a in anchors)
    # The citation inside (2)'s own text sits mid-line, so it is not a marker.
    assert len(anchors) == 3


def test_a_declared_marker_form_adds_to_the_keyword_forms() -> None:
    """Both spellings must anchor, because the same corpus uses both.

    The enacted text prints the subdivision bare, "(1) ", while the explanatory
    memorandum spells it out as a heading. A declaration that replaced the
    keyword rather than joining it dropped every spelled-out one silently, and
    the document still parsed.
    """
    from codify.pipeline.enrich.anchors import _alias_terms_for

    entry = HierarchyEntry(
        local_term="Ayat",
        akn_element="paragraph",
        level="subdivision",
        marker_form="parenthesized_arabic",
    )
    assert "Ayat" in _alias_terms_for(entry)

    text = "Pasal 5\n(1) Ketentuan pertama.\nAyat (2)\nKetentuan kedua.\n"
    scan = _scan(text, "xl", "act")
    paragraphs = [(a.number, a.matched_text.strip()) for a in scan.anchors if a.kind == "paragraph"]
    assert paragraphs == [("1", "(1)"), ("2", "Ayat (2")], paragraphs


def test_the_coverage_denominator_counts_both_spellings_too() -> None:
    """The denominator has to match the scan, or the ratio is against a fraction.

    The scan anchors the bare marker and the spelled-out keyword; a denominator
    counting only the declared form scores a document above 1.0 and the gate
    stops meaning anything.
    """
    from codify.pipeline.enrich.anchors import (
        anchor_coverage,
        build_anchor_regex,
        scan_anchors,
    )

    config = load_config("xl")
    # The spelled-out ones carry a number the bare ones do not, or both sets
    # coincide and a denominator holding either alone still scores 1.0.
    text = (
        "Pasal 5\n(1) Ketentuan pertama.\n(2) Ketentuan kedua.\n"
        "PENJELASAN\nPasal 5\nAyat (3)\nCukup jelas.\n"
    )
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    coverage = anchor_coverage(text, anchors, config, "act", "paragraph")
    assert coverage.ratio is not None and coverage.ratio <= 1.0, coverage


def test_the_prose_filter_sees_the_line_above_a_declared_marker() -> None:
    """The commonest cross-reference idiom wraps onto the marker's own line.

    The filter's window walks back from a marker's newline, so a match anchored
    after that newline was handed an empty window and read every wrapped
    citation as a heading. A bare marker has no keyword to distinguish it, so
    nothing downstream would have caught it.
    """
    from codify.pipeline.enrich.anchors import _is_prose_reference, _probe_from

    text = "Ketentuan sebagaimana dimaksud pada ayat\n(7) di atas berlaku.\n"
    at_line_start = text.index("(7)")

    # What the declared-marker scan used to pass: the window is empty.
    assert not _is_prose_reference(text, at_line_start, "xl")
    # What it passes now.
    assert _is_prose_reference(text, _probe_from(text, at_line_start), "xl")


def test_a_quoted_amendment_is_not_the_host_act_s_own_subdivision() -> None:
    """An amending act prints the article it replaces, subdivisions and all.

    A quoted subdivision is written exactly as a real one, so unmasked it
    anchors and attaches to the amending act's own article, giving the host a
    structure that belongs to the law it amends. The bracketed form needs no
    mask, a quoted statute never being written that way, which is why the
    declared scan ran unmasked.
    """
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    config = load_config("xl")
    text = (
        "Pasal I\n"
        "Beberapa ketentuan diubah sebagai berikut:\n"
        "1. Ketentuan Pasal 5 diubah sehingga berbunyi sebagai berikut:\n"
        "“Pasal 5\n"
        "(1) Ketentuan yang diubah pertama.\n"
        "(2) Ketentuan yang diubah kedua.”\n"
        "Pasal II\n"
        "Undang-Undang ini mulai berlaku.\n"
    )
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    assert [a.akn_eid for a in anchors] == ["art_I", "art_II"], [a.akn_eid for a in anchors]


def test_the_denominator_filters_prose_with_the_jurisdiction_s_own_words() -> None:
    """Scanned without the country, the filter falls back to another script's
    precursors, so a wrapped citation leaves the numerator and stays in the
    denominator and the ratio is measured against markers that are not there.
    """
    from codify.pipeline.enrich.anchors import _marker_numbers

    config = load_config("xl")
    text = (
        "Pasal 5\n"
        "(1) Ketentuan pertama.\n"
        "(2) Ketentuan sebagaimana dimaksud pada ayat\n"
        "(7) di atas berlaku.\n"
    )
    assert _marker_numbers(text, config, "act", "paragraph") == {"1", "2"}


def test_the_production_check_reports_what_a_quote_span_hid() -> None:
    """Asserted through the scan the pipeline runs, not the helper.

    The check measures the basic unit, which declares no marker form, so a
    count filtered to the measured kind reads zero however many markers the
    mask took, and the finding says a flawless ratio on a document that lost
    most of its subdivisions.
    """
    from codify.quality.structural_scan import _check_anchor_coverage

    config = load_config("xl")
    text = (
        "Pasal 5\n"
        "(1) Ketentuan pertama dengan tanda “kutip yang tidak ditutup.\n"
        "(2) Ketentuan kedua.\n"
        "(3) Ketentuan ketiga.\n"
        "(4) Ketentuan keempat.\n"
    )
    finding = _check_anchor_coverage(text, config, "act")
    assert finding.detail["masked_markers"] == 3, finding.detail
    assert finding.failed, finding.detail
    assert finding.detail["reason"] == "markers_masked", finding.detail


def test_an_amending_act_that_quotes_properly_still_passes() -> None:
    """The control. A balanced quote masks the amendment it encloses, which is
    what the mask is for, so masking alone cannot be the failure.
    """
    from codify.quality.structural_scan import _check_anchor_coverage

    config = load_config("xl")
    text = (
        "Pasal I\n"
        "Ketentuan Pasal 5 diubah sehingga berbunyi:\n"
        "“Pasal 5\n"
        "(1) Ketentuan yang diubah pertama.\n"
        "(2) Ketentuan yang diubah kedua.”\n"
        "Pasal II\n"
        "Undang-Undang ini mulai berlaku.\n"
    )
    finding = _check_anchor_coverage(text, config, "act")
    assert finding.detail["masked_markers"] == 0, finding.detail
    assert not finding.failed, finding.detail


def test_an_amendment_quoting_more_than_one_block_still_passes() -> None:
    """The second control, and the ordinary shape: a quoted article runs to
    several paragraphs, so its quote spans a blank line. Counting unclosed
    quotes per block reads that as two unclosed ones and fails a correct
    document.
    """
    from codify.quality.structural_scan import _check_anchor_coverage

    config = load_config("xl")
    text = (
        "Pasal I\n"
        "Ketentuan Pasal 5 diubah sehingga berbunyi:\n"
        "“Pasal 5\n"
        "(1) Ketentuan yang diubah pertama.\n"
        "\n"
        "(2) Ketentuan yang diubah kedua.”\n"
        "Pasal II\n"
        "Undang-Undang ini mulai berlaku.\n"
    )
    finding = _check_anchor_coverage(text, config, "act")
    # Nothing was hidden by accident: the count is of markers a span nothing
    # closed took, and this quote closes.
    assert finding.detail["masked_markers"] == 0, finding.detail
    assert not finding.failed, finding.detail


def test_precursors_match_a_line_ending_in_any_case() -> None:
    """A scan yields the same word in whatever case the page carried it."""
    from codify.pipeline.enrich.anchors import _is_prose_reference, _probe_from

    for spelling in ("ayat", "AYAT", "Ayat"):
        text = f"Ketentuan sebagaimana dimaksud pada {spelling}\n(7) di atas berlaku.\n"
        at_line_start = text.index("(7)")
        assert _is_prose_reference(text, _probe_from(text, at_line_start), "xl"), spelling


def test_a_heading_ending_in_a_keyword_does_not_suppress_the_unit_below_it() -> None:
    """Why the article keyword is not a precursor, case-insensitively.

    The explanatory memorandum opens with a heading ending in that keyword, so
    as a precursor it suppresses the very first unit the memorandum describes.
    """
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    config = load_config("xl")
    text = "II. PASAL DEMI PASAL\nPasal 1\nCukup jelas.\nPasal 2\nCukup jelas.\n"
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    assert [a.number for a in anchors if a.kind == "article"] == ["1", "2"]


_PAD = "".join(f"Pasal {i}\n(1) Ketentuan nomor {i}.\n" for i in range(1, 21))


# Every shape the quote mask has to tell apart, as one set. Three consecutive
# fixes here each passed the shape in front of them and broke the next, because
# the controls were built one at a time.
_QUOTE_SHAPES = [
    (
        "one unclosed quote",
        _PAD + "Pasal 21\n(1) Satu dengan tanda “kutip.\n(2) Dua.\n(3) Tiga.\n(4) Empat.\n",
        True,
    ),
    (
        "two unclosed quotes in different articles",
        _PAD + "Pasal 21\n(1) Satu dengan “kutip.\n(2) Dua dengan “kutip lain.\n"
        "Pasal 22\n(1) Tiga.\n(2) Empat.\n",
        True,
    ),
    (
        "a balanced quote in one block",
        "Pasal I\nDiubah:\n“Pasal 5\n(1) Satu.\n(2) Dua.”\nPasal II\nBerlaku.\n",
        False,
    ),
    (
        "a balanced quote across a blank line",
        "Pasal I\nDiubah:\n“Pasal 5\n(1) Satu.\n\n(2) Dua.”\nPasal II\nBerlaku.\n",
        False,
    ),
    (
        # The OCR shape that puts the walk on its reversed path: a closer with
        # no opener anywhere. Nothing can be paired, so the count has only the
        # end of the text to come from.
        "a reversed unclosed quote, nothing after it to end the span",
        "Pasal 5\n(1) Satu dengan ”kutip terbalik.\n(2) Dua.\n(3) Tiga.\n(4) Empat.\n",
        True,
    ),
    (
        # The same quote, ended by a blank line: the span reaches nothing, so
        # the document measures honestly and there is nothing to report.
        "a reversed unclosed quote a blank line ends",
        "Pasal 5\n(1) Satu dengan ”kutip terbalik.\n\n(2) Dua.\n(3) Tiga.\n(4) Empat.\n",
        False,
    ),
    (
        "a balanced guillemet amendment",
        "Pasal I\nDiubah:\n«Pasal 5\n(1) Satu.\n(2) Dua.»\nPasal II\nBerlaku.\n",
        False,
    ),
    (
        # A directional opener nothing closes is skipped outright, so it masks
        # nothing and every subdivision anchors.
        "an unclosed guillemet",
        "Pasal 5\n(1) Satu dengan «kutip.\n(2) Dua.\n(3) Tiga.\n(4) Empat.\n",
        False,
    ),
    (
        # Worse than the mid-article case: the span reaches every marker, so
        # the document anchors nothing and its ratio still reads perfectly.
        "a reversed unclosed quote before the first marker",
        "Pasal 5\nPembukaan dengan ”kutip.\n(1) Satu.\n(2) Dua.\n(3) Tiga.\n",
        True,
    ),
    (
        # The same quote with nothing after it. A span that hides no marker is
        # not a failure: the document measures honestly and there is nothing
        # to report.
        "a reversed unclosed quote after the last marker",
        "Pasal 5\n(1) Satu.\n(2) Dua.\n(3) Tiga.\n(4) Empat.\nPenutup dengan ”kutip.\n",
        False,
    ),
    (
        # A stray after a balanced amendment must still be caught, the
        # amendment's own quotes having closed properly around it.
        "a balanced amendment followed by a stray opener",
        _PAD + "Pasal 21\nDiubah:\n“Pasal 5\n(1) Satu.\n(2) Dua.”\n"
        "Pasal 22\n(1) Tiga dengan “kutip.\n(2) Empat.\n",
        True,
    ),
    (
        # The same shape with a closer instead. The document opens its quotes
        # the ordinary way round, so a lone closer closes nothing and quotes
        # nothing after it; what it suggests is a dropped opener earlier.
        "a lone closer after a balanced amendment",
        _PAD + "Pasal 21\nDiubah:\n“Pasal 5\n(1) Satu.\n(2) Dua.”\n"
        "Pasal 22\n(1) Tiga dengan ”kutip.\n(2) Empat.\n",
        False,
    ),
    (
        # One character at both ends, which OCR of RTL typography produces and a
        # real statute in the corpus uses throughout. Nothing can be paired, so
        # it toggles; read as a pair it lost half that act's articles.
        "a quote using one character at both ends",
        _PAD + "Pasal 21\n(1) Yang dimaksud dengan ”keadilan” adalah asas.\n"
        "(2) Yang dimaksud dengan ”keberlanjutan” adalah asas lain.\n"
        "Pasal 22\n(1) Tiga.\n",
        False,
    ),
    (
        # Toggling and closed. A span is stray only if nothing closes it, so
        # marking every toggled one on open fails a correct amendment.
        "same-character quotes around an amendment",
        _PAD + "Pasal 21\nDiubah:\n”Pasal 5\n(1) Satu.\n(2) Dua.”\nPasal 22\n(1) Tiga.\n",
        False,
    ),
    (
        "a nested quote before an amendment",
        _PAD + "Pasal 21\n(1) Yang dimaksud dengan ”istilah” adalah asas.\n"
        "Pasal 22\nDiubah:\n“Pasal 5\n(1) Satu.\n(2) Dua.”\nPasal 23\n(1) Tiga.\n",
        False,
    ),
    (
        # A low-nine opener takes a curly closer of either hand, which OCR of
        # this corpus does emit.
        "a balanced low-nine quotation",
        _PAD + "Pasal 21\nDiubah:\n„Pasal 5\n(1) Satu.\n(2) Dua.“\nPasal 22\n(1) Tiga.\n",
        False,
    ),
    (
        # Skipped rather than bounded, as the unclosed guillemet is: a
        # directional opener nothing closes masks nothing, so the document
        # measures honestly and there is nothing to report.
        "an unclosed low-nine quotation",
        _PAD + "Pasal 21\n(1) Satu dengan „kutip.\n(2) Dua.\n(3) Tiga.\n",
        False,
    ),
    (
        # The only marker the span hides is a wrapped citation, which the scan
        # would never have anchored. A span costs nobody a marker that was
        # never going to be one.
        "a stray hiding only a wrapped citation",
        _PAD + "Pasal 21\n(1) Satu \u201cdengan kutip hilang sebagaimana dimaksud pada ayat\n"
        "(7) di atas berlaku.\n",
        False,
    ),
    (
        # Reversed and balanced: one of each, so counting them cannot say which
        # opens. Read the wrong way round its provisions are promoted into the
        # host act.
        "a reversed but balanced amendment",
        _PAD + "Pasal 21\nDiubah:\n”Pasal 5\n(1) Satu.\n(2) Dua.“\nPasal 22\n(1) Tiga.\n",
        False,
    ),
    (
        # Two independent facts are not one: the amendment's quote masks by
        # design, and a stray that reaches no marker hides nothing.
        "a balanced amendment and a stray that hides nothing",
        _PAD + "Pasal 21\nDiubah:\n“Pasal 5\n(1) Satu.\n(2) Dua.”\n"
        "Pasal 22\n(1) Tiga.\nPenutup dengan “kutip.\n",
        False,
    ),
    (
        # One hidden marker of fourteen. The count does not need the damage to
        # be large, only real.
        "a late opener in a long document",
        "Pasal 5\n"
        + "".join(f"({i}) Ketentuan nomor {i}.\n" for i in range(1, 13))
        + "(13) Tiga belas dengan “kutip.\n(14) Empat belas.\n",
        True,
    ),
    (
        # Long enough to clear the ratio, so only the count can fail it. The
        # amendment's own opener would otherwise close the stray span before
        # it and its closer open a fresh one, hiding the stray from the count.
        "a stray quote, then a balanced amendment, in a document that scores well",
        "".join(f"Pasal {i}\n(1) Ketentuan nomor {i}.\n" for i in range(1, 21))
        + "Pasal 21\n(1) Dengan “kutip.\n"
        + "Pasal 22\n(1) Berikutnya.\n"
        + "Pasal I\nDiubah:\n“Pasal 9\n(1) Satu.\n(2) Dua.”\nPasal II\nBerlaku.\n",
        True,
    ),
    (
        # The same class within one article: a dropped closer on its first
        # subdivision and a genuine quoted amendment on its fourth.
        "a stray quote and an amendment inside one article",
        _PAD + "Pasal 21\n"
        "(1) Satu dengan “kutip yang hilang penutupnya.\n"
        "(2) Dua.\n"
        "(3) Tiga.\n"
        "(4) Empat, diubah menjadi “Pasal 9 ayat baru.”\n"
        "Pasal 22\n(1) Berikutnya.\n",
        True,
    ),
    (
        "an unclosed quote before a balanced amendment",
        _PAD + "Pasal 21\n(1) Satu dengan “kutip.\n(2) Dua.\n"
        "Pasal I\nDiubah:\n“Pasal 9\n(1) Tiga.\n(2) Empat.”\nPasal II\nBerlaku.\n",
        True,
    ),
]


@pytest.mark.parametrize(("name", "text", "should_fail"), _QUOTE_SHAPES)
def test_the_coverage_check_tells_the_quote_shapes_apart(
    name: str, text: str, should_fail: bool
) -> None:
    from codify.quality.structural_scan import _check_anchor_coverage

    finding = _check_anchor_coverage(text, load_config("xl"), "act")
    assert bool(finding.failed) is should_fail, (name, finding.detail)


# Where the annex begins decides which numbering is a provision and which is an
# ordinary list. Every fault in this helper has been a boundary, never the
# counting, so the boundaries are one set.
_ANNEX_SHAPES = [
    # A span opening in the body reaches into the annex; scanning a slice from
    # the caption cannot see the quote that opened it.
    (
        "a body stray reaching the annex",
        "Pasal 1\n(1) Ketentuan dengan “kutip hilang.\n"
        "LAMPIRAN\nA. Bagian.\n1. Butir satu.\n2. Butir dua.\n",
        3,
    ),
    # A sentence opening with the caption word is prose. Taken as the boundary
    # it counts the body list after it as annex provisions.
    (
        "the caption word inside body prose",
        "Pasal 1\n(1) Ketentuan.\n"
        "LAMPIRAN sebagaimana dimaksud diatur tersendiri.\n"
        "(2) Lain dengan “kutip hilang.\n1. Daftar butir.\n2. Daftar dua.\n"
        "LAMPIRAN\nA. Bagian.\n1. Butir.\n",
        2,
    ),
    # No annex, so nothing can be inside one.
    (
        "no caption at all",
        "Pasal 1\n(1) Ketentuan dengan “kutip hilang.\n1. Daftar satu.\n2. Daftar dua.\n",
        0,
    ),
    (
        "a caption on the last line",
        "Pasal 1\n(1) Ketentuan dengan “kutip hilang.\n1. Daftar satu.\n2. Daftar dua.\nLAMPIRAN\n",
        0,
    ),
    (
        "two captions, the stray in the second",
        "Pasal 1\n(1) Ketentuan.\n"
        "LAMPIRAN I\nA. Bagian.\n1. Butir.\n"
        "LAMPIRAN II\nA. Bagian dengan “kutip hilang.\n1. Butir dua.\n2. Butir tiga.\n",
        2,
    ),
    (
        "a caption inside a quoted span",
        "Pasal 1\n(1) Ketentuan dengan “kutip hilang.\n"
        "LAMPIRAN\nA. Bagian.\n1. Butir.\n2. Butir dua.\n",
        3,
    ),
    (
        "a lowercase caption",
        "Pasal 1\n(1) Ketentuan.\n"
        "lampiran\nA. Bagian dengan “kutip hilang.\n1. Butir.\n2. Butir dua.\n",
        2,
    ),
    (
        "a caption with a numeral",
        "Pasal 1\n(1) Ketentuan.\n"
        "LAMPIRAN II\nA. Bagian dengan “kutip hilang.\n1. Butir.\n2. Butir dua.\n",
        2,
    ),
]


@pytest.mark.parametrize(("name", "text", "expected"), _ANNEX_SHAPES)
def test_the_annex_boundary_decides_what_counts(name: str, text: str, expected: int) -> None:
    from codify.pipeline.enrich.anchors import _masked_marker_count

    assert _masked_marker_count(text, load_config("xl"), "act") == expected, name


def test_an_unclosed_quote_in_an_annex_is_counted_too() -> None:
    """The annex declares its own levels, and the body class need not declare
    the same ones, so a count taken over the body alone sees nothing there.

    On the shipped config rather than the fixture one, which declares no
    attachment hierarchy to exercise.
    """
    from codify.pipeline.enrich.anchors import _masked_marker_count

    config = load_config("xl")
    text = (
        "Pasal 1\n(1) Ketentuan.\n"
        "LAMPIRAN\nA. Bagian pertama dengan “kutip yang hilang.\n"
        "1. Butir pertama.\n2. Butir kedua.\n3. Butir ketiga.\n"
    )
    assert _masked_marker_count(text, config, "act") == 3

    # The same numbering before the annex is an ordinary list and is not
    # counted; the annex's own heading after it is, the stray span reaching
    # that far. One, not four, and not zero.
    body_list = (
        "Pasal 1\n(1) Ketentuan dengan “kutip yang hilang.\n"
        "1. Daftar butir satu.\n2. Daftar butir dua.\n3. Daftar butir tiga.\n"
        "LAMPIRAN\nA. Bagian.\n"
    )
    assert _masked_marker_count(body_list, config, "act") == 1


def test_a_reversed_amendment_inside_a_normal_document_is_still_quoted() -> None:
    """Both orientations in one document, the ordinary case for a scan that
    reversed some pages and not others.

    Asserted on the anchors, not the gate: an unmasked quote adds structure
    rather than losing it, so the coverage verdict cannot see this. Its
    provisions arrive as the host act's own and nothing objects.
    """
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    config = load_config("xl")
    text = (
        "Pasal 21\nDiubah:\n\u201cPasal 2\n(1) Normal satu.\n(2) Normal dua.\u201d\n"
        "Pasal 22\nDiubah:\n\u201dPasal 9\n(1) Terbalik satu.\n(2) Terbalik dua.\u201c\n"
        "Pasal 23\n(1) Sesudahnya.\n"
    )
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    articles = [a.number for a in anchors if a.kind == "article"]
    paragraphs = [a.number for a in anchors if a.kind == "paragraph"]
    # Neither quoted article, and only the one subdivision the host itself has.
    assert articles == ["21", "22", "23"], articles
    assert paragraphs == ["1"], paragraphs


def test_an_orphan_closer_does_not_reach_a_later_amendment() -> None:
    """A dropped closer looks exactly like a reversed opener until you ask how
    far it reaches.

    With an ordinary amendment further down the document, the span runs from the
    orphan to that amendment's own opener, closes there, and is therefore not a
    stray: the articles between vanish, nothing is counted as hidden, and the
    ratio stays above its floor. An amendment quotes one article, so a span
    crossing a provision heading is a dropped closer, not an opener.
    """
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    config = load_config("xl")
    text = (
        "Pasal 21\nDiubah:\n“Pasal 2\n(1) A.\n(2) B.”\n"
        "Pasal 22\n(1) Dengan ”kutip yatim.\n"
        "Pasal 23\n(1) C.\n"
        "Pasal 24\n(1) D.\n"
        "Pasal 25\n(1) E.\n"
        "Pasal 26\nDiubah:\n“Pasal 3\n(1) F.\n(2) G.”\n"
        "Pasal 27\n(1) H.\n"
    )
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    articles = [a.number for a in anchors if a.kind == "article"]
    # Every host article, and neither quoted one.
    assert articles == ["21", "22", "23", "24", "25", "26", "27"], articles


def test_the_reported_unclosed_count_is_taken_under_the_same_country() -> None:
    """`masked` and `unclosed` describe one walk, so they must share its rules.

    The masked count is taken with the jurisdiction's own boundary; the unclosed
    count was taken without it, so the two described different documents and a
    halt could read "1 marker hidden by 0 unclosed quotes". It also evicted the
    single-entry walk cache the masked count had just filled, walking every
    measured document twice.
    """
    from codify.pipeline.enrich.anchors import (
        _unclosed_quote_spans,
        anchor_coverage,
        build_anchor_regex,
        scan_anchors,
    )

    config = load_config("xl")
    # A reversed opener the boundary refuses, so the span it would have closed
    # stays stray under the jurisdiction's rules and does not under none.
    text = "“(3) teks ”\n”(4) teks ”\nPasal 7\n„(7) teks “"
    assert _unclosed_quote_spans(text, "") != _unclosed_quote_spans(text, "xl")

    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")
    coverage = anchor_coverage(text, anchors, config, "act", "article", with_masked=True)

    assert coverage.unclosed == _unclosed_quote_spans(text, "xl")


def test_outline_markers_tolerate_markdown_bold() -> None:
    from codify.pipeline.enrich.anchors import _OUTLINE_RES

    rx = _OUTLINE_RES["arabic_period"]
    m1 = rx.search("**1.** First section")
    assert m1 is not None and m1.group("num") == "1"

    m2 = rx.search("**2**. Second section")
    assert m2 is not None and m2.group("num") == "2"

    m3 = rx.search("3. Third section")
    assert m3 is not None and m3.group("num") == "3"


def test_roman_outline_markers_tolerate_markdown_bold() -> None:
    from codify.jurisdictions import HierarchyEntry
    from codify.pipeline.enrich.anchors import _attachment_roman_markers

    level = HierarchyEntry(
        local_term="Unit",
        akn_element="article",
        level="basic",
        bluebell_keyword="ARTICLE",
        numbering="roman",
        marker_form="upper_roman_period",
    )
    text = "**I.** First item\n**II**. Second item\n"
    anchors = _attachment_roman_markers(text, [level], 0, "", [])

    assert [anchor.number for anchor in anchors] == ["I", "II"]


def test_line_anchored_boundary_rejects_midline_part_in_amendment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codify.jurisdictions import HierarchyEntry, JurisdictionConfig, StructuringConfig
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    cfg = JurisdictionConfig(
        code="xe",
        name="Synthetic",
        tradition=["common_law"],
        languages=["eng"],
        authoritative_language="eng",
        default_document_class="act",
        document_classes={
            "act": {
                "label": "Act",
                "akn_element": "act",
                "basic_unit": "section",
                "hierarchy": [
                    HierarchyEntry(
                        local_term="Part",
                        akn_element="part",
                        level="higher",
                        bluebell_keyword="PART",
                        numbering="roman",
                    ),
                    HierarchyEntry(
                        local_term="Section",
                        akn_element="section",
                        level="basic",
                        bluebell_keyword="SECTION",
                        numbering="arabic_continuous",
                        marker_form="arabic_period",
                    ),
                ],
            }
        },
        structuring=StructuringConfig(marker_boundary="line_anchored"),
    )
    monkeypatch.setattr("codify.pipeline.enrich.anchors.load_config", lambda _c: cfg)
    rx = build_anchor_regex(cfg, "act")
    text = (
        "1. Citation.\n"
        "This Order may be cited as Order 2022.\n\n"
        "2. Amendment.\n"
        "(1) Part C of the Schedule is amended.\n"
    )
    anchors = scan_anchors(text, rx, country="xe", doctype="act")
    assert [a.kind for a in anchors] == ["section", "section"]
    assert [a.number for a in anchors] == ["1", "2"]


def test_line_anchored_boundary_still_admits_a_column_gap_before_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three or more spaces are a column gap to this boundary, so the same
    amendment sentence laid out that way still anchors the part. The boundary
    protects against prose spacing, not against column-shaped prose."""
    from codify.jurisdictions import HierarchyEntry, JurisdictionConfig, StructuringConfig
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    cfg = JurisdictionConfig(
        code="xe",
        name="Synthetic",
        tradition=["common_law"],
        languages=["eng"],
        authoritative_language="eng",
        default_document_class="act",
        document_classes={
            "act": {
                "label": "Act",
                "akn_element": "act",
                "basic_unit": "section",
                "hierarchy": [
                    HierarchyEntry(
                        local_term="Part",
                        akn_element="part",
                        level="higher",
                        bluebell_keyword="PART",
                        numbering="roman",
                    ),
                    HierarchyEntry(
                        local_term="Section",
                        akn_element="section",
                        level="basic",
                        bluebell_keyword="SECTION",
                        numbering="arabic_continuous",
                        marker_form="arabic_period",
                    ),
                ],
            }
        },
        structuring=StructuringConfig(marker_boundary="line_anchored"),
    )
    monkeypatch.setattr("codify.pipeline.enrich.anchors.load_config", lambda _c: cfg)
    rx = build_anchor_regex(cfg, "act")
    text = "2. Amendment.\n(1)   Part C of the Schedule is amended.\n"  # three spaces
    anchors = scan_anchors(text, rx, country="xe", doctype="act")
    assert [(a.kind, a.number) for a in anchors] == [("section", "2"), ("part", "C")]
