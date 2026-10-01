# ruff: noqa: E501  # Arabic test strings render shorter visually than their byte length
"""Tests for the Arabic cardinal-words parser and money words/numeral check."""

from __future__ import annotations

import pytest

from codify.pipeline.enrich.arabic_cardinals import (
    find_money_word_mismatches,
    parse_cardinal_words,
)


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("خمسون", 50),
        ("خمسة وعشرون", 25),
        ("أحد عشر", 11),
        ("اثنا عشر", 12),
        ("مائة", 100),
        ("مائتان", 200),
        ("ثلاثمائة", 300),
        ("ألف", 1000),
        ("ألفان", 2000),
        ("خمسون ألف", 50_000),
        ("خمسمائة ألف", 500_000),
        ("مائتان وخمسون ألفاً", 250_000),
        ("ثلاثة آلاف وخمسمائة", 3_500),
        ("عشرة آلاف", 10_000),
        ("مليون", 1_000_000),
        ("خمسة ملايين وثلاثمائة ألف", 5_300_000),
        ("تسعة وتسعون", 99),
    ],
)
def test_parse_cardinal_words(words: str, value: int) -> None:
    assert parse_cardinal_words(words) == value


def test_parse_rejects_non_number_words() -> None:
    assert parse_cardinal_words("كلمة عادية") is None
    assert parse_cardinal_words("") is None
    assert parse_cardinal_words("غرامة مالية") is None


class TestMoneyWordMismatch:
    def test_arb_reg_art8_numeral_words_disagree(self) -> None:
        # Synthetic disagreement: numeral says 500,000, words say 50,000.
        text = "بغرامة لا تقل عن (500,000) خمسون ألف دينار زرزوري أو ما يعادلها"
        found = find_money_word_mismatches(text)
        assert len(found) == 1
        assert found[0].numeral_value == 500_000
        assert found[0].words_value == 50_000

    def test_words_before_numeral_order(self) -> None:
        text = "غرامة قدرها خمسون ألف دينار زرزوري (500,000) تدفع فوراً"
        found = find_money_word_mismatches(text)
        assert len(found) == 1
        assert found[0].words_value == 50_000

    @pytest.mark.parametrize(
        "clean",
        [
            "بغرامة لا تقل عن (50,000) خمسون ألف دينار زرزوري",
            "بغرامة قدرها (1,000) ألف دينار زرزوري",
            "وفقاً لأحكام المادة (5) والفقرة (2) من هذا القانون",
            "خلال مدة أقصاها (30) يوماً من تاريخ النشر",
            "تتكون اللجنة من (7) أعضاء يعينهم المجلس",
            "مبلغ وديعة (1000) في حساب مصرفي",
            "تسري أحكام البند (3) على الجميع",
            "غرامة لا تتجاوز (خمسمائة) دينار",
            "نسبة لا تقل عن (51%) من رأس المال",
            "المادة (12) مكرر من القانون الأساسي",
        ],
    )
    def test_clean_text_zero_false_positives(self, clean: str) -> None:
        assert find_money_word_mismatches(clean) == []


def test_fused_hundred_short_spelling() -> None:
    assert parse_cardinal_words("خمسمئة") == 500
    assert parse_cardinal_words("ثلاثمئة ألف") == 300_000


def test_bare_scale_multiplier_notation_not_flagged() -> None:
    # "(50) ألف دينار" means 50,000 (numeral × scale), not a second reading.
    assert find_money_word_mismatches("غرامة قدرها (50) ألف دينار زرزوري") == []
    assert find_money_word_mismatches("مبلغ (2) مليون دينار") == []
