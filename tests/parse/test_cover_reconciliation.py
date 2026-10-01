"""Synthetic covers and bodies exercise article-count reconciliation."""

from __future__ import annotations

from codify.pipeline.enrich.cover_reconciliation import (
    extract_cover_article_numbers,
    reconcile_cover_vs_body,
)


def _toc_page(count: int, marker: str = "Article", with_keyword: bool = True) -> str:
    """Fabricate a TOC-shaped cover page. Includes the ``Contents``
    keyword by default so the extractor gate opens; pass
    ``with_keyword=False`` to test the density-only path."""
    lines = ["Table of Contents"] if with_keyword else []
    lines.extend(f"{marker} {n}. heading here" for n in range(1, count + 1))
    return "\n".join(lines)


# --- extract_cover_article_numbers -----------------------------------------


def test_toc_with_keyword_returns_numbers() -> None:
    """A well-formed TOC block with the `Contents` keyword returns the
    ordered list of article numbers."""
    numbers = extract_cover_article_numbers(_toc_page(20))
    assert numbers == list(range(1, 21))


def test_dense_toc_without_keyword_returns_numbers() -> None:
    """A high-density block of marker lines with no other prose is enough
    on its own; density above the floor implies TOC shape."""
    text = _toc_page(20, with_keyword=False)
    numbers = extract_cover_article_numbers(text)
    assert numbers == list(range(1, 21))


def test_body_with_five_articles_at_top_is_not_a_toc() -> None:
    """Copilot #3 regression: five line-anchored article headings in a
    normal body page (Articles 1-5 with prose between them) must NOT be
    accepted as a TOC. Without the density gate this false-fired on
    every ordinary body extract."""
    lines = []
    for n in range(1, 6):
        lines.append(f"Article {n}")
        lines.append("The Council shall convene and consider matters brought before it.")
        lines.append("The relevant provisions of the preceding law shall apply.")
        lines.append("Each member has the right to speak and vote on any resolution.")
    text = "\n".join(lines)
    assert extract_cover_article_numbers(text) is None


def test_arabic_toc_page_returns_numbers() -> None:
    """Arabic gazette-style cover uses `المادة` markers; the pattern must
    match those the same as `Article`."""
    lines = ["المحتويات"]
    lines.extend(f"المادة {n} - عنوان" for n in range(1, 15))
    numbers = extract_cover_article_numbers("\n".join(lines))
    assert numbers == list(range(1, 15))


def test_gapped_toc_preserves_the_gap() -> None:
    """Copilot #4 regression: `max(numbers)` inflates the count when the
    TOC has gaps. Returning the number list lets the reconciliation
    compare set size against body count, which is what we actually want.
    Articles 1, 2, 5 = 3 declared, not 5."""
    lines = [
        "Table of Contents",
        "Article 1",
        "Article 2",
        "Article 5",
        "Article 8",
        "Article 10",
    ]
    numbers = extract_cover_article_numbers("\n".join(lines))
    assert numbers == [1, 2, 5, 8, 10]


def test_arabic_indic_digits_folded() -> None:
    text = "المحتويات\n" + "\n".join(
        f"المادة {ar_num} - عنوان" for ar_num in ["١", "٢", "٣", "٤", "٥"]
    )
    assert extract_cover_article_numbers(text) == [1, 2, 3, 4, 5]


def test_prose_page_returns_none() -> None:
    text = (
        "This law regulates something under Article 3 and Article 5. "
        "The Council convened under Article 7 last year."
    )
    assert extract_cover_article_numbers(text) is None


def test_empty_input_returns_none() -> None:
    assert extract_cover_article_numbers("") is None
    assert extract_cover_article_numbers("   \n\n  ") is None


def test_cover_scan_bounded_to_char_window() -> None:
    """A body-buried article list past the cover character window must
    not be treated as a TOC."""
    padding = "Prose about statutory purposes and definitions. " * 200
    text = padding + "\n" + _toc_page(20)
    assert len(padding) > 6000
    assert extract_cover_article_numbers(text) is None


def test_bidi_control_at_line_start_still_matches() -> None:
    lrm = "‎"
    lines = ["المحتويات"]
    lines.extend(f"{lrm}المادة {n} - عنوان" for n in range(1, 15))
    numbers = extract_cover_article_numbers("\n".join(lines))
    assert numbers == list(range(1, 15))


def test_short_marker_run_returns_none() -> None:
    """Below the min-markers-for-TOC floor, treat as non-TOC."""
    text = _toc_page(3)
    assert extract_cover_article_numbers(text) is None


# --- reconcile_cover_vs_body ---------------------------------------------


def test_matching_counts_return_none() -> None:
    assert reconcile_cover_vs_body(list(range(1, 21)), 20) is None


def test_missing_cover_returns_none() -> None:
    assert reconcile_cover_vs_body(None, 20) is None
    assert reconcile_cover_vs_body([], 20) is None


def test_off_by_one_is_ignored() -> None:
    """Annexes and final articles routinely sit outside the TOC."""
    assert reconcile_cover_vs_body(list(range(1, 21)), 19) is None
    assert reconcile_cover_vs_body(list(range(1, 21)), 21) is None


def test_env_7_1999_shape_returns_error() -> None:
    """Cover 93, body 82, 11-article shortfall fires at error severity."""
    issue = reconcile_cover_vs_body(list(range(1, 94)), 82)
    assert issue is not None
    assert issue["check"] == "cover_body_article_count_mismatch"
    assert issue["severity"] == "error"
    assert issue["cover_count"] == 93
    assert issue["body_count"] == 82
    assert issue["delta"] == -11


def test_off_by_one_swallows_reviewer_arb_3_2000_case() -> None:
    """Documenting the tolerance trade-off: reviewer's Arb 3/2000 case
    (cover 59, body 58) sits inside ±1 and does not fire. Trading
    slightly more silent-drop coverage against false-positive rate on
    documents whose annexes sit outside the TOC."""
    assert reconcile_cover_vs_body(list(range(1, 60)), 58) is None


def test_two_article_delta_is_warning() -> None:
    issue = reconcile_cover_vs_body(list(range(1, 21)), 18)
    assert issue is not None
    assert issue["severity"] == "warning"


def test_body_over_cover_flagged() -> None:
    """Civil Code 4/2014 shape: cover 1230, body 1302."""
    issue = reconcile_cover_vs_body(list(range(1, 1231)), 1302)
    assert issue is not None
    assert issue["delta"] == 72
    assert issue["severity"] == "error"


def test_gapped_cover_reports_set_size_not_max() -> None:
    """Copilot #4: TOC has articles 1, 2, 5 = 3 declared; body has 3
    articles. No mismatch. If we naively used max(numbers)=5, we would
    fire a spurious delta of -2."""
    assert reconcile_cover_vs_body([1, 2, 5], 3) is None


def test_duplicate_numbers_deduplicated() -> None:
    """A TOC that repeats an article number counts once in the set."""
    assert reconcile_cover_vs_body([1, 2, 2, 3], 3) is None
