"""Unit tests for the anchor-count validator + number-set continuity."""

from __future__ import annotations

from typing import Any

from codify.pipeline.enrich.anchors import StructuralAnchor
from codify.pipeline.enrich.validator import validate_akn

_AKN_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      {articles}
    </body>
  </act>
</akomaNtoso>
"""


def _make_akn_with_articles(n: int) -> str:
    inner = "\n".join(
        f'<article eId="art_{i}"><num>{i}</num><heading>A{i}</heading>'
        f"<content><p>Body {i}.</p></content></article>"
        for i in range(1, n + 1)
    )
    return _AKN_TEMPLATE.format(articles=inner)


def test_anchor_count_mismatch_fires_on_undershoot() -> None:
    akn = _make_akn_with_articles(5)  # AKN has 5 articles
    issues = validate_akn(akn, expected_anchor_summary={"article": 7})
    mismatch = [i for i in issues if i["check"] == "anchor_count_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "error"
    assert mismatch[0]["kind"] == "article"
    assert mismatch[0]["expected"] == 7
    assert mismatch[0]["got"] == 5


def test_anchor_count_mismatch_silent_when_count_matches() -> None:
    akn = _make_akn_with_articles(5)
    issues = validate_akn(akn, expected_anchor_summary={"article": 5})
    mismatch = [i for i in issues if i["check"] == "anchor_count_mismatch"]
    assert mismatch == []


def test_anchor_count_mismatch_warning_on_2x_overshoot() -> None:
    akn = _make_akn_with_articles(5)
    issues = validate_akn(akn, expected_anchor_summary={"article": 2})
    mismatch = [i for i in issues if i["check"] == "anchor_count_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "warning"


def test_cover_body_reconciliation_fires_on_shortfall() -> None:
    """Env 7/1999 shape: cover TOC declares 20 but the emitted body
    carries 5. Fires as error severity."""
    akn = _make_akn_with_articles(5)
    issues = validate_akn(akn, expected_cover_article_numbers=list(range(1, 21)))
    mismatch = [i for i in issues if i["check"] == "cover_body_article_count_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "error"
    assert mismatch[0]["cover_count"] == 20
    assert mismatch[0]["body_count"] == 5


def test_cover_body_reconciliation_silent_when_missing() -> None:
    """When no cover count is passed, the check is a no-op; callers with
    no TOC-shaped source page still get all other checks."""
    akn = _make_akn_with_articles(5)
    issues = validate_akn(akn)
    mismatch = [i for i in issues if i["check"] == "cover_body_article_count_mismatch"]
    assert mismatch == []


def test_cover_body_reconciliation_off_by_one_ignored() -> None:
    """Annexes / final articles routinely sit outside the TOC; off-by-one
    must not fire so a clean document does not carry a spurious warning."""
    akn = _make_akn_with_articles(5)
    issues = validate_akn(akn, expected_cover_article_numbers=list(range(1, 7)))
    mismatch = [i for i in issues if i["check"] == "cover_body_article_count_mismatch"]
    assert mismatch == []


def test_validate_akn_no_op_when_summary_missing() -> None:
    akn = _make_akn_with_articles(2)
    # Backwards-compatible signature: no expected_anchor_summary, no
    # anchor_count_mismatch issues regardless of count.
    issues = validate_akn(akn)
    assert all(i["check"] != "anchor_count_mismatch" for i in issues)


# ── Number-set continuity ──────────────────────────────────────────────────


def test_number_gap_detected_in_middle() -> None:
    inner = (
        '<article eId="art_1"><num>1</num><content><p>a</p></content></article>'
        '<article eId="art_2"><num>2</num><content><p>b</p></content></article>'
        '<article eId="art_5"><num>5</num><content><p>e</p></content></article>'
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    gaps = [i for i in issues if i["check"] == "number_gap"]
    assert len(gaps) == 1
    assert gaps[0]["kind"] == "article"
    assert gaps[0]["missing"] == [3, 4]
    assert gaps[0]["severity"] == "warning"


def test_duplicate_number_detected() -> None:
    inner = (
        '<article eId="art_1"><num>1</num><content><p>a</p></content></article>'
        '<article eId="art_2"><num>2</num><content><p>b</p></content></article>'
        '<article eId="art_2_2"><num>2</num><content><p>b2</p></content></article>'
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    dups = [i for i in issues if i["check"] == "duplicate_number"]
    assert len(dups) == 1
    assert dups[0]["number"] == "2"
    assert set(dups[0]["eids"]) == {"art_2", "art_2_2"}


def test_paragraph_duplicate_check_is_scoped_within_article() -> None:
    """Civil-law codes restart paragraph numbering per article. PARAGRAPH 1
    in art_8 and PARAGRAPH 1 in art_10 are NOT duplicates, they're
    sibling positions in different parents. Doc-wide grouping would
    fire false positives (the bg/ro round-1 regression)."""
    inner = (
        '<article eId="art_8"><num>8</num>'
        '<paragraph eId="art_8__para_1"><num>1</num><content><p>a</p></content></paragraph>'
        '<paragraph eId="art_8__para_2"><num>2</num><content><p>b</p></content></paragraph>'
        "</article>"
        '<article eId="art_10"><num>10</num>'
        '<paragraph eId="art_10__para_1"><num>1</num><content><p>c</p></content></paragraph>'
        '<paragraph eId="art_10__para_2"><num>2</num><content><p>d</p></content></paragraph>'
        "</article>"
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    dups = [i for i in issues if i["check"] == "duplicate_number"]
    assert dups == [], f"expected no duplicate_number issues; got {dups}"


def test_paragraph_duplicate_still_fires_within_same_article() -> None:
    """The dedup error is still a real signal, same article emitting
    PARAGRAPH 1 twice (cross-chunk duplication into the same article)
    must still produce a duplicate_number issue."""
    inner = (
        '<article eId="art_5"><num>5</num>'
        '<paragraph eId="art_5__para_1"><num>1</num><content><p>a</p></content></paragraph>'
        '<paragraph eId="art_5__para_1_2"><num>1</num><content><p>a2</p></content></paragraph>'
        "</article>"
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    dups = [i for i in issues if i["check"] == "duplicate_number" and i["kind"] == "paragraph"]
    assert len(dups) == 1
    assert dups[0]["number"] == "1"
    assert set(dups[0]["eids"]) == {"art_5__para_1", "art_5__para_1_2"}


def test_no_gap_when_continuous() -> None:
    inner = "".join(
        f'<article eId="art_{i}"><num>{i}</num><content><p>x</p></content></article>'
        for i in range(1, 6)
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    assert not any(i["check"] in ("number_gap", "duplicate_number") for i in issues)


def test_number_gap_silent_on_roman_or_mixed() -> None:
    # A consolidated act with "Article 22/1" sub-articles after 22 must not
    # false-positive between 22 and 22/1.
    inner = (
        '<article eId="art_22"><num>22</num><content><p>x</p></content></article>'
        '<article eId="art_22-1"><num>22/1</num><content><p>x</p></content></article>'
        '<article eId="art_23"><num>23</num><content><p>x</p></content></article>'
    )
    akn = _AKN_TEMPLATE.format(articles=inner)
    issues = validate_akn(akn)
    # No gap between 22 and 23 (arithmetic continuous), no duplicate.
    gaps = [i for i in issues if i["check"] == "number_gap"]
    assert gaps == []


class TestUndetectedAmendmentShape:
    """Warn when article numbering reveals an unrecognised amendment shape."""

    def test_non_monotonic_sequence_fires_warning(self) -> None:
        """AC 37/2018's `1, 2, 4, 5, 3, 6` shape: article 3 sits between
        articles 5 and 6, which is a strict decrease at position 4."""
        inner = "".join(
            f'<article eId="art_{n}"><num>{n}</num><content><p>x</p></content></article>'
            for n in [1, 2, 4, 5, 3, 6]
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns, [i["check"] for i in issues]
        assert warns[0]["severity"] == "warning"
        assert warns[0]["previous_number"] == 5
        assert warns[0]["number"] == 3

    def test_forward_jump_over_ceiling_fires_warning(self) -> None:
        """PP Amendment's `1, 2, 3, 16, 4, 5` shape: 3 to 16 is a jump
        of 13, above the plural-budget ceiling of 8."""
        inner = "".join(
            f'<article eId="art_{n}"><num>{n}</num><content><p>x</p></content></article>'
            for n in [1, 2, 3, 16, 4, 5]
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        # Both the 3 -> 16 forward jump and the 16 -> 4 strict decrease fire.
        assert len(warns) >= 1
        assert any(w["number"] == 16 for w in warns)

    def test_monotonic_sequence_no_warning(self) -> None:
        """A clean 1..N sequence must not fire; false positives here would
        cry wolf for every well-formed amendment-free act."""
        inner = "".join(
            f'<article eId="art_{n}"><num>{n}</num><content><p>x</p></content></article>'
            for n in range(1, 8)
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns == []

    def test_correctly_quoted_amendment_no_warning(self) -> None:
        """When an amendment is correctly emitted with the embedded
        article nested inside `<mod><quotedStructure>`, the top-level
        sequence stays clean. Nested articles are ignored by design."""
        inner = (
            '<article eId="art_1"><num>1</num><content><p>x</p></content></article>'
            '<article eId="art_2"><num>2</num><content><p>x</p></content></article>'
            '<article eId="art_3"><num>3</num>'
            "<content>"
            '<mod><quotedStructure><article eId="quoted_art_9">'
            "<num>9</num><content><p>quoted content</p></content>"
            "</article></quotedStructure></mod>"
            "</content>"
            "</article>"
            '<article eId="art_4"><num>4</num><content><p>x</p></content></article>'
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns == []


class TestUndetectedAmendmentShapeScope:
    """Check articles nested under containers, not only direct body children."""

    def test_chaptered_law_flags_non_monotonic(self) -> None:
        """AC 37/2018 shape but with articles nested under a chapter.
        Pre-fix this returned zero warnings silently."""
        inner = (
            '<chapter eId="chp_1"><num>1</num>'
            + "".join(
                f'<article eId="chp_1__art_{n}"><num>{n}</num><content><p>x</p></content></article>'
                for n in [1, 2, 4, 5, 3, 6]
            )
            + "</chapter>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns, [i["check"] for i in issues]
        assert warns[0]["previous_number"] == 5
        assert warns[0]["number"] == 3

    def test_arabic_indic_digits_detected(self) -> None:
        """Arabic AKN articles carry <num>١</num>, and the
        check must fold to compare. Pre-fix zero warnings would have been
        silent on an Arabic corpus."""
        inner = "".join(
            f'<article eId="art_{i}"><num>{n}</num><content><p>x</p></content></article>'
            for i, n in enumerate(["١", "٢", "٤", "٥", "٣", "٦"], start=1)
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns
        assert warns[0]["previous_number"] == 5
        assert warns[0]["number"] == 3


class TestUndetectedAmendmentShapePerContainerScope:
    """Allow numbering to restart in each chapter by checking each container."""

    def test_per_chapter_restart_no_warning(self) -> None:
        inner = (
            '<chapter eId="chp_1"><num>1</num>'
            + "".join(
                f'<article eId="chp_1__art_{n}"><num>{n}</num><content><p>x</p></content></article>'
                for n in [1, 2, 3]
            )
            + "</chapter>"
            + '<chapter eId="chp_2"><num>2</num>'
            + "".join(
                f'<article eId="chp_2__art_{n}"><num>{n}</num><content><p>x</p></content></article>'
                for n in [1, 2, 3]
            )
            + "</chapter>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "undetected_amendment_shape"]
        assert warns == [], warns


class TestMissingContainerTitles:
    """Warn when a structural container has no descriptive heading."""

    def test_headingless_chapter_fires_warning(self) -> None:
        inner = (
            '<chapter eId="chp_5"><num>5</num>'
            '<article eId="chp_5__art_1"><num>1</num><content><p>x</p></content></article>'
            "</chapter>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "missing_container_title"]
        assert len(warns) == 1
        assert warns[0]["severity"] == "warning"
        assert warns[0]["eid"] == "chp_5"
        assert warns[0]["container"] == "chapter"

    def test_chapter_with_heading_is_silent(self) -> None:
        inner = (
            '<chapter eId="chp_5"><num>5</num><heading>General provisions</heading>'
            '<article eId="chp_5__art_1"><num>1</num><content><p>x</p></content></article>'
            "</chapter>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        assert [i for i in issues if i["check"] == "missing_container_title"] == []

    def test_empty_heading_fires_warning(self) -> None:
        """A present-but-blank <heading> is the same reader-facing gap."""
        inner = (
            '<chapter eId="chp_1"><num>1</num><heading>  </heading>'
            '<article eId="chp_1__art_1"><num>1</num><content><p>x</p></content></article>'
            "</chapter>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "missing_container_title"]
        assert len(warns) == 1

    def test_headingless_part_and_book_fire_per_kind(self) -> None:
        inner = (
            '<part eId="part_1"><num>1</num>'
            '<book eId="bk_1"><num>1</num>'
            '<article eId="art_1"><num>1</num><content><p>x</p></content></article>'
            "</book></part>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = {i["container"] for i in issues if i["check"] == "missing_container_title"}
        assert warns == {"part", "book"}

    def test_headingless_section_not_flagged_without_articles(self) -> None:
        """In article-less doctypes (gb) sections are body-fill targets; a
        headingless section is legitimate and stays out of this check."""
        inner = '<section eId="sec_1"><num>1</num><content><p>x</p></content></section>'
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        assert [i for i in issues if i["check"] == "missing_container_title"] == []

    def test_headingless_section_flagged_when_document_has_articles(self) -> None:
        """Mirrors the capture pass: with articles present, `section` is a
        grouping level (Ukraine's РОЗДІЛ) and a headless one must warn, or
        a failed capture would ship gate-approved."""
        inner = (
            '<section eId="sec_1"><num>I</num>'
            '<article eId="sec_1__art_1"><num>1</num><content><p>x</p></content></article>'
            "</section>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "missing_container_title"]
        assert [w["container"] for w in warns] == ["section"]

    def test_headingless_tome_fires_warning(self) -> None:
        """`tome` is title-captured at scan time, so the validator checks
        it too."""
        inner = (
            '<tome eId="tome_1"><num>1</num>'
            '<article eId="art_1"><num>1</num><content><p>x</p></content></article>'
            "</tome>"
        )
        akn = _AKN_TEMPLATE.format(articles=inner)
        issues = validate_akn(akn)
        warns = [i for i in issues if i["check"] == "missing_container_title"]
        assert [w["container"] for w in warns] == ["tome"]


def _ar_source(n: int, prefix: str = "مادة") -> str:
    return "\n".join(f"{prefix} {i}\nنص المادة رقم {i} هنا.\n" for i in range(1, n + 1))


def test_header_coverage_fires_on_undersegmentation() -> None:
    # Source claims 10 article headers; the AKN carries 6. The self-referential
    # coverage probe cannot see this; the independent header count can.
    issues = validate_akn(_make_akn_with_articles(6), source_text=_ar_source(10))
    hc = [i for i in issues if i["check"] == "header_coverage"]
    assert len(hc) == 1
    assert hc[0]["severity"] == "warning"
    assert "10" in hc[0]["message"] and "6" in hc[0]["message"]


def test_header_coverage_silent_when_matched() -> None:
    issues = validate_akn(_make_akn_with_articles(10), source_text=_ar_source(10))
    assert [i for i in issues if i["check"] == "header_coverage"] == []


def test_header_coverage_counts_stray_paren_headers() -> None:
    # RTL reorder prefixes headers with a lone `)`; they still count as headers,
    # so a document missing them is flagged.
    src = "\n".join(f")مادة {i}\nنص هنا.\n" for i in range(1, 11))
    issues = validate_akn(_make_akn_with_articles(6), source_text=src)
    assert any(i["check"] == "header_coverage" for i in issues)


def test_header_coverage_abstains_below_minimum() -> None:
    # Too few headers to trust (a short order); no false shortfall.
    issues = validate_akn(_make_akn_with_articles(1), source_text=_ar_source(4))
    assert [i for i in issues if i["check"] == "header_coverage"] == []


def test_header_coverage_abstains_on_quoted_amendment_markers() -> None:
    # Markers inside «…» are amendments of other laws, not this document's own
    # articles, so their count is not the article count.
    inner = "\n".join(f"مادة {i}\nيستعاض عن النص.\n" for i in range(1, 11))
    issues = validate_akn(_make_akn_with_articles(2), source_text=f"«{inner}»")
    assert [i for i in issues if i["check"] == "header_coverage"] == []


def test_header_coverage_abstains_for_non_arabic_source() -> None:
    src = "\n".join(f"Article {i}\nBody of article {i}." for i in range(1, 11))
    issues = validate_akn(_make_akn_with_articles(3), source_text=src)
    assert [i for i in issues if i["check"] == "header_coverage"] == []


def test_page_yield_names_the_real_failed_page() -> None:
    # A page with ink but ~zero text is a failed OCR read; the message must name
    # the real page_number (7), not the list position (2).
    issues = validate_akn(
        _make_akn_with_articles(5), page_yield=[(1, 42000.0), (7, 0.0), (12, 45000.0)]
    )
    py = [i for i in issues if i["check"] == "page_yield"]
    assert len(py) == 1
    assert py[0]["severity"] == "error"  # blocking: missing legal text
    assert "page 7" in py[0]["message"]
    assert "1 of 3" in py[0]["message"]


def test_page_yield_silent_when_all_pages_yield() -> None:
    issues = validate_akn(
        _make_akn_with_articles(5), page_yield=[(1, 42000.0), (2, 45000.0), (3, 30000.0)]
    )
    assert [i for i in issues if i["check"] == "page_yield"] == []


def test_page_yield_abstains_when_absent() -> None:
    # Born-digital versions have no page reads; the check must not fire (and must
    # never false-clear).
    issues = validate_akn(_make_akn_with_articles(5))
    assert [i for i in issues if i["check"] == "page_yield"] == []


def test_page_yield_recoverable_page_is_a_warning_not_blocking() -> None:
    # Page 7 read empty but the caller verified its content is in the AKN: the
    # content is recoverable, so this warns rather than blocking delivery.
    issues = validate_akn(
        _make_akn_with_articles(5),
        page_yield=[(1, 42000.0), (7, 0.0), (12, 45000.0)],
        recoverable_pages={7},
    )
    py = [i for i in issues if i["check"] == "page_yield"]
    assert len(py) == 1
    assert py[0]["severity"] == "warning"
    assert "page 7" in py[0]["message"]
    assert "recoverable" in py[0]["message"]


def test_page_yield_blocks_when_page_not_recoverable() -> None:
    # No recoverable set: the content is not established in the AKN, still blocking.
    issues = validate_akn(_make_akn_with_articles(5), page_yield=[(1, 42000.0), (7, 0.0)])
    py = [i for i in issues if i["check"] == "page_yield"]
    assert len(py) == 1
    assert py[0]["severity"] == "error"


def test_page_yield_splits_lost_and_recoverable() -> None:
    # Page 7 lost (not in AKN), page 9 recoverable: one blocking error and one
    # warning, each naming its own pages.
    issues = validate_akn(
        _make_akn_with_articles(5),
        page_yield=[(7, 0.0), (9, 0.0)],
        recoverable_pages={9},
    )
    py = {i["severity"]: i for i in issues if i["check"] == "page_yield"}
    assert "page 7" in py["error"]["message"]
    assert "page 9" in py["warning"]["message"]


def test_recoverable_empty_pages_matches_content_present_in_akn() -> None:
    # The rival's words appear in the AKN body -> recoverable; a disjoint rival
    # (its content never reached the AKN) -> not recoverable, stays blocking.
    from codify.pipeline.enrich.validator import recoverable_empty_pages

    akn = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act>'
        '<meta><identification source="#c"/></meta>'
        "<body><p>alpha bravo charlie delta echo foxtrot</p></body></act></akomaNtoso>"
    )
    present = "Alpha Bravo charlie delta plus other"  # 4 of 6 tokens in the AKN
    absent = "zzqqx wibble frobnicate nonexistent gibberish tokens"  # none in the AKN
    assert recoverable_empty_pages(akn, {5: present, 7: absent}) == {5}
    assert recoverable_empty_pages(akn, {}) == set()
    assert recoverable_empty_pages("<not-xml", {5: present}) == set()


class TestSummaryExcludesQuotedAmendments:
    """The gap that had no failing test: every case above hand-writes the
    expected summary, so none exercised `anchor_summary()` over a real anchor
    list, which is where the two populations diverged."""

    @staticmethod
    def _anchors() -> list[StructuralAnchor]:
        from dataclasses import replace

        def anchor(number: str, offset: int) -> StructuralAnchor:
            return StructuralAnchor(
                kind="article",
                keyword="Pasal",
                number=number,
                char_offset=offset,
                line=offset // 80,
                matched_text=f"Pasal {number}",
            )

        quoted = [replace(anchor(str(n), 100 + n), quoted_amendment=True) for n in range(1, 20)]
        return [anchor("I", 10), *quoted, anchor("II", 9000)]

    def test_the_summary_counts_only_the_document_s_own_anchors(self) -> None:
        from codify.pipeline.enrich.anchors import anchor_summary

        assert anchor_summary(self._anchors()) == {"article": 2}

    def test_a_genuine_drop_in_an_amending_act_is_still_caught(self) -> None:
        """The filter must not blind the check it was written for. The act's own
        two articles are still expected, so losing one is still an error."""
        from codify.pipeline.enrich.anchors import anchor_summary

        issues = validate_akn(
            _make_akn_with_articles(1),
            expected_anchor_summary=anchor_summary(self._anchors()),
        )
        mismatch = [i for i in issues if i["check"] == "anchor_count_mismatch"]
        assert len(mismatch) == 1
        assert mismatch[0]["severity"] == "error"
        assert (mismatch[0]["expected"], mismatch[0]["got"]) == (2, 1)

    def test_an_amending_act_does_not_read_as_a_dropped_body(self) -> None:
        """An amending act quotes the articles it replaces. Counting them made
        the AKN look 19 articles short of a document that never existed."""
        from codify.pipeline.enrich.anchors import anchor_summary

        akn = _make_akn_with_articles(2)
        issues = validate_akn(akn, expected_anchor_summary=anchor_summary(self._anchors()))
        assert [i for i in issues if i["check"] == "anchor_count_mismatch"] == []


_AKN_WITH_IDENTITY = """<?xml version="1.0" encoding="UTF-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <meta>
      <identification source="#codify">
        <FRBRWork>
          <FRBRthis value="{uri}/!main"/>
          <FRBRuri value="{uri}"/>
        </FRBRWork>
      </identification>
    </meta>
    <body><article eId="art_1"><num>1</num><content><p>Body.</p></content></article></body>
  </act>
</akomaNtoso>
"""


class TestUncitableWorkUri:
    """A content-addressed number finishes an ingest and leaves a law nobody can
    cite. Silent, it is how five judgments lost their identity unnoticed."""

    @staticmethod
    def _findings(uri: str) -> list[dict[str, Any]]:
        issues = validate_akn(_AKN_WITH_IDENTITY.format(uri=uri))
        return [i for i in issues if i["check"] == "uncitable_work_uri"]

    def test_a_draft_number_is_reported(self) -> None:
        found = self._findings("/akn/xl/judgment/mk/2011/draft-fbf1614f137f")
        assert len(found) == 1
        assert found[0]["severity"] == "warning"

    def test_a_citable_uri_is_silent(self) -> None:
        assert self._findings("/akn/xl/judgment/mk/2011/45") == []

    def test_a_component_tail_does_not_fault_a_citable_law(self) -> None:
        """`FRBRthis` appends `/!main`, which shifts the year/number window by
        one. Reading it without unpicking faulted every document in the corpus."""
        akn = _AKN_WITH_IDENTITY.format(uri="/akn/xz/act/decree-law/2005/1")
        assert "/!main" in akn
        assert [i for i in validate_akn(akn) if i["check"] == "uncitable_work_uri"] == []

    def test_a_pipeline_document_is_not_faulted(self) -> None:
        """The fabricated fixtures above all passed while the check faulted
        every real document, so this reads one the pipeline actually produced."""
        from pathlib import Path

        akn = (
            Path(__file__).resolve().parents[1] / "fixtures" / "synthetic-atlantis-1992.akn.xml"
        ).read_text()
        assert [i for i in validate_akn(akn) if i["check"] == "uncitable_work_uri"] == []

    def test_a_document_with_no_identity_block_is_not_faulted(self) -> None:
        """Every fixture in this file omits meta; absence is not a finding."""
        assert [
            i
            for i in validate_akn(_make_akn_with_articles(1))
            if i["check"] == "uncitable_work_uri"
        ] == []


class TestAnchorCountsIgnoreQuotedStructures:
    @staticmethod
    def _quoted_body(wrapper: str, own: int, borrowed: int) -> str:
        quoted = "".join(
            f'<article eId="quoted_{n}"><num>{n}</num>'
            "<content><p>Replacement.</p></content></article>"
            for n in range(borrowed)
        )
        articles = "".join(
            f'<article eId="own_{n}"><num>{n}</num>'
            "<content><p>Operative text.</p></content></article>"
            for n in range(own)
        )
        return _AKN_TEMPLATE.format(articles=f"{articles}<{wrapper}>{quoted}</{wrapper}>")

    def test_quoted_articles_cannot_replace_a_missing_host_article(self) -> None:
        from codify.pipeline.enrich.anchors import anchor_summary

        for wrapper in ("mod", "quotedStructure", "embeddedStructure"):
            issues = validate_akn(
                self._quoted_body(wrapper, own=1, borrowed=1),
                expected_anchor_summary=anchor_summary(
                    TestSummaryExcludesQuotedAmendments._anchors()
                ),
            )
            mismatch = [i for i in issues if i["check"] == "anchor_count_mismatch"]
            assert [(i["severity"], i["expected"], i["got"]) for i in mismatch] == [
                ("error", 2, 1)
            ], wrapper

    def test_quoted_articles_do_not_inflate_an_intact_host(self) -> None:
        for wrapper in ("mod", "quotedStructure", "embeddedStructure"):
            issues = validate_akn(
                self._quoted_body(wrapper, own=2, borrowed=9),
                expected_anchor_summary={"article": 2},
            )
            assert [i for i in issues if i["check"] == "anchor_count_mismatch"] == [], wrapper
