"""Unit tests for the Hebrew-target branch of the num canonicaliser.

`_canonicalise_num_text(root, target_language="he")` folds Arabic ordinal
words to Hebrew letter numerals (`الأول → א`), Arabic abjad enumerators
to Hebrew alphabet (`أ. → א.`), and Arabic-Indic digits to Latin (Hebrew
legal typography reads Latin digits natively). Unknown targets fall back
to the Latin default.
"""

from __future__ import annotations

import pytest
from lxml import etree

from codify.akn import AKN_NS
from codify.translate.write import _canonicalise_num_text


def _tree_with_nums(*nums: str) -> etree._Element:
    root = etree.Element(f"{{{AKN_NS}}}body")
    for num_text in nums:
        num = etree.SubElement(root, f"{{{AKN_NS}}}num")
        num.text = num_text
    return root


def _num_texts(root: etree._Element) -> list[str]:
    return [n.text or "" for n in root.iter(f"{{{AKN_NS}}}num")]


@pytest.mark.parametrize("tag", ["he", "heb", "Hebrew", "HE", "iw"])
def test_hebrew_ordinal_folds_to_letter_numeral(tag: str) -> None:
    """Every accepted Hebrew tag (ISO-639-1, ISO-639-2, name, legacy `iw`,
    case-insensitive) routes ordinal words to letter numerals."""
    root = _tree_with_nums("الأول", "الثاني", "العاشر")
    _canonicalise_num_text(root, tag)
    assert _num_texts(root) == ["א", "ב", "י"]


def test_hebrew_compound_ordinal_uses_taboo_free_forms() -> None:
    """11-19 use letter-numeral pairs; 15 and 16 avoid the religious
    taboo pairs (יה / יו) by using טו / טז."""
    root = _tree_with_nums("الحادي عشر", "الرابع عشر", "الخامس عشر", "السادس عشر")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["יא", "יד", "טו", "טז"]


def test_hebrew_abjad_enumerator_folds_to_hebrew_alphabet() -> None:
    """Sub-list enumerators map letter-for-letter: `أ. → א.`, `ب- → ב-`."""
    root = _tree_with_nums("أ.", "ب-", "ج", "ه.")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א.", "ב-", "ג", "ה."]


def test_hebrew_target_folds_arabic_indic_digits_to_hebrew_letter_numeral() -> None:
    """Arabic-Indic digits fold Latin-first then Latin-to-Hebrew, so a
    ``<num>٥</num>`` reaches the Hebrew reader as ``ה`` rather than a
    bare ``5``. Same fold the backfill relies on to repair legacy
    Hebrew AKN that shipped with Latin ``<num>`` values."""
    root = _tree_with_nums("٥", "١٢", "١")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["ה", "יב", "א"]


def test_none_target_falls_back_to_latin() -> None:
    """Unknown / missing target routes to the Latin default so nothing
    regresses on the English lane (and no reader ever sees a raw Arabic
    ordinal because the fallback is opinionated)."""
    root = _tree_with_nums("الأول", "أ.", "٥")
    _canonicalise_num_text(root, None)
    assert _num_texts(root) == ["1", "a.", "5"]


def test_english_target_falls_back_to_latin() -> None:
    """Explicit non-Hebrew targets also route to the Latin default."""
    root = _tree_with_nums("الأول", "أ.")
    _canonicalise_num_text(root, "en")
    assert _num_texts(root) == ["1", "a."]


def test_hebrew_ordinal_wins_over_abjad_first_char() -> None:
    """`الأول` starts with `ا` (Hebrew map: `א`); the ordinal branch
    runs first so the full compound wins the fold, not the letter."""
    root = _tree_with_nums("الأول")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א"]


def test_hebrew_tatweel_ordinal_still_folds() -> None:
    """Tatweel-decorated ordinals normalise the same way for Hebrew as
    for the Latin lane, via the shared ordinal-lookup helper."""
    root = _tree_with_nums("الـأول")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א"]


def test_hebrew_backfill_converts_latin_digit_to_letter_numeral() -> None:
    """Legacy Hebrew AKN shipped with Latin `<num>` text (`1`, `12`)
    from the target-blind write pass. A Hebrew backfill re-run converts
    those to the letter-numeral form the Hebrew reader expects."""
    root = _tree_with_nums("1", "12", "20")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א", "יב", "כ"]


def test_hebrew_backfill_converts_latin_abjad_to_hebrew_alphabet() -> None:
    """Legacy Hebrew AKN with Latin enumerators (`a.`, `b-`) folds
    forward into the Hebrew alphabet on backfill so a re-canonicalise
    matches what a fresh Hebrew translation would emit."""
    root = _tree_with_nums("a.", "b-", "c", "e.")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א.", "ב-", "ג", "ה."]


def test_hebrew_abjad_letter_without_hebrew_form_falls_back_to_latin() -> None:
    """`ث`, `خ`, `ذ`, `ض` have no direct Hebrew abjad equivalent. Rather
    than leave the raw Arabic glyph inside `<num>` (mixed script in a
    Hebrew document), the Hebrew map inherits the Latin form for those
    letters, no reader ever sees a foreign glyph."""
    root = _tree_with_nums("ث.", "خ.", "ذ.", "ض")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["w.", "x.", "y.", "z"]


def test_zwnj_stripped_from_abjad_enumerator() -> None:
    """ZWNJ (U+200C) between abjad letter and separator sneaks in from
    word-processor round-trips. `JOINER_STRIP_TABLE` collapses it before
    lookup on both the Latin and Hebrew lanes."""
    root = _tree_with_nums("أ‌.", "ب‌-")
    _canonicalise_num_text(root, None)
    assert _num_texts(root) == ["a.", "b-"]
    root = _tree_with_nums("أ‌.", "ب‌-")
    _canonicalise_num_text(root, "he")
    assert _num_texts(root) == ["א.", "ב-"]


def test_apply_translation_to_akn_threads_hebrew_target_end_to_end() -> None:
    """Locks the target_language kwarg on the top-level entry point so a
    future refactor that drops it can't silently regress Hebrew back to
    Latin without a test failing."""
    from codify.akn import AKN_NS
    from codify.translate.translate_bodies import TranslatedBlock
    from codify.translate.write import apply_translation_to_akn

    source = (
        f'<akomaNtoso xmlns="{AKN_NS}">'
        '<act name="act">'
        "<meta/>"
        "<preface><p>preface source</p></preface>"
        '<body><chapter eId="chp_1"><num>الأول</num>'
        '<article eId="chp_1__art_1"><num>أ.</num>'
        "<content><p>source body</p></content>"
        "</article></chapter></body>"
        "</act>"
        "</akomaNtoso>"
    )
    blocks = [
        TranslatedBlock(eid="chp_1__art_1", heading=None, lines=["translated body"]),
    ]
    result = apply_translation_to_akn(source, blocks, ["preface tr"], target_language="he")
    assert "<num>א</num>" in result
    assert "<num>א.</num>" in result

    latin = apply_translation_to_akn(source, blocks, ["preface tr"], target_language="en")
    assert "<num>1</num>" in latin
    assert "<num>a.</num>" in latin
