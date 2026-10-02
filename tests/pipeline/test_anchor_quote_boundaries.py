"""Quote orientation recovery respects every article-level heading."""

from __future__ import annotations

import pytest

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import (
    _quote_mask,
    _stray_quote_mask,
    build_anchor_regex,
    scan_anchors,
)


def _numbers(text: str, kind: str) -> list[str | None]:
    config = load_config("xu")
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xu", doctype="act")
    return [a.number for a in anchors if a.kind == kind and not a.quoted_amendment]


@pytest.mark.parametrize("kind", ["article", "section"])
def test_mixed_quote_closer_cannot_borrow_a_later_provisions_opener(kind: str) -> None:
    marker = kind.upper()
    text = (
        "The authority adopts the “Account Rules”.\n\n"
        f'{marker} 1. These are the "Inquiry Rules.”\n\n'
        f"{marker} 2. All doubts shall favour implementation.\n\n"
        f"{marker} 3. The “Deposit Law” applies.\n\n"
        f"{marker} 4. A “day” means a calendar day.\n"
    )
    assert _numbers(text, kind) == ["1", "2", "3", "4"]


@pytest.mark.parametrize("kind", ["article", "section"])
@pytest.mark.parametrize("quotes", [("“", "”"), ("”", "“")])
def test_balanced_amendment_stays_masked_across_blank_lines(
    kind: str, quotes: tuple[str, str]
) -> None:
    marker = kind.upper()
    opening, closing = quotes
    text = (
        "The authority adopts the “Account Rules”.\n\n"
        f"{marker} 1. Replace the provision with:\n"
        f"{opening}{marker} 9. Replacement text.\n\n"
        f"Additional replacement text.{closing}\n\n"
        f"{marker} 2. Commencement.\n"
    )
    assert _numbers(text, kind) == ["1", "2"]
    assert not any(_stray_quote_mask(text, "xu"))


@pytest.mark.parametrize("kind", ["article", "section"])
def test_unclosed_curly_span_still_reports_hidden_text_and_resets(kind: str) -> None:
    """A quote opening on a heading may quote several, so it masks to the blank line."""
    marker = kind.upper()
    text = (
        f"{marker} 1. Replace the provision with:\n"
        f"“{marker} 9. Replacement with no closer.\n"
        f"{marker} 10. Hidden text.\n\n"
        f"{marker} 2. Commencement.\n"
    )
    assert _numbers(text, kind) == ["1", "2"]
    stray = _stray_quote_mask(text, "xu")
    assert stray[text.index(f"{marker} 10")]
    assert not stray[text.index(f"{marker} 2")]


def test_a_prose_quote_nothing_closes_stops_at_the_next_heading() -> None:
    """A dropped closer on a quoted name must not take the chapter and article below it."""
    text = (
        "The authority founds the “Harbour Board, which keeps the moorings.\n"
        "It therefore enacts the following:\n"
        "CHAPTER I\nGeneral\n"
        "ARTICLE 1. This creates the “Harbour Board”.\n"
        "ARTICLE 2. Commencement.\n"
    )
    config = load_config("xu")
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xu", doctype="act")
    assert [(a.kind, a.number) for a in anchors] == [
        ("chapter", "I"),
        ("article", "1"),
        ("article", "2"),
    ]
    stray = _stray_quote_mask(text, "xu")
    assert stray[text.index("Harbour Board, which")]
    assert not stray[text.index("CHAPTER I")]


# cp1252 bytes 0x93/0x94 decoded as Latin-1: the curly quotes a text extractor
# leaves behind when it reads a Windows-encoded PDF byte for byte.
_MOJIBAKE_OPEN, _MOJIBAKE_CLOSE = "\x93", "\x94"


@pytest.mark.parametrize("kind", ["article", "section"])
def test_an_amendment_quoted_in_mojibake_glyphs_is_masked(kind: str) -> None:
    marker = kind.upper()
    text = (
        f"{marker} 1. Replace the provision with:\n"
        f"{_MOJIBAKE_OPEN}{marker} 9. Replacement text.{_MOJIBAKE_CLOSE}\n\n"
        f"{marker} 2. Commencement.\n"
    )
    assert _numbers(text, kind) == ["1", "2"]


@pytest.mark.parametrize("kind", ["article", "section"])
def test_the_same_document_without_the_glyphs_keeps_the_marker(kind: str) -> None:
    """Control: the mask, not the layout, is what suppresses the quoted heading."""
    marker = kind.upper()
    text = (
        f"{marker} 1. Replace the provision with:\n"
        f"{marker} 9. Replacement text.\n\n"
        f"{marker} 2. Commencement.\n"
    )
    assert _numbers(text, kind) == ["1", "9", "2"]


def test_folding_the_glyphs_moves_no_offset() -> None:
    """Every mask offset is an offset into the caller's own text."""
    text = f"ARTICLE 1. A {_MOJIBAKE_OPEN}quoted term{_MOJIBAKE_CLOSE} here.\n"
    mask = _quote_mask(text, "xu")
    assert len(mask) == len(text) + 1
    assert mask[text.index(_MOJIBAKE_OPEN)] and not mask[text.index("ARTICLE")]
