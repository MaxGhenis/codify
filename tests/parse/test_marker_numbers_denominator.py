"""The coverage-gate denominator (`_marker_numbers`, regex path) is pinned.

Wave-2b anchor-grammar dedup: `_marker_numbers` now composes `_compile_anchor_regex`
(via a `column_only` boundary + `case_insensitive` flag) instead of hand-rebuilding
the scanner grammar. These cases were captured from the pre-composition code and
must stay byte-identical: a change to the denominator silently skews every
coverage-gate ratio."""

from __future__ import annotations

import pytest

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import _marker_numbers


@pytest.mark.parametrize(
    ("country", "doctype", "kind", "text", "expected"),
    [
        # Latin, column-anchored markers count.
        ("us", "act", "section", "SECTION 1. Foo.\nSECTION 2. Bar.\n", {"1", "2"}),
        ("fr", "loi", "article", "Article 1\ncorps\nArticle 4\n", {"1", "4"}),
        # A mid-prose reference (المادة (٢) أعلاه) must NOT count toward the denominator.
        ("xz", "act", "article", "المادة 1\nنص\nفي المادة (٢) أعلاه\n", {"1"}),
    ],
)
def test_marker_numbers_regex_path_is_stable(
    country: str, doctype: str, kind: str, text: str, expected: set[str]
) -> None:
    assert _marker_numbers(text, load_config(country), doctype, kind) == expected
