"""Legibility word lists key on language, not script."""

from __future__ import annotations

import pytest

from codify.jurisdictions import load_config
from codify.quality.legibility import function_word_rate, lexicon_rate, text_verdict
from codify.quality.lexicons import words_for

BAHASA = (
    "Dalam Undang-Undang ini yang dimaksud dengan hutan adalah suatu kesatuan "
    "ekosistem berupa hamparan lahan berisi sumber daya alam hayati yang "
    "didominasi pepohonan dalam persekutuan alam lingkungannya yang satu "
    "dengan lainnya tidak dapat dipisahkan. "
) * 12

# The shape behind the letter-spaced address block: every letter separated,
# so no closed-class word survives as a token.
LETTER_SPACED = ("D a l a m U n d a n g i n i y a n g d i m a k s u d d e n g a n h u t a n ") * 12

ENGLISH = (
    "This Act may be cited as the Forestry Act and shall come into force on "
    "such date as the Minister may by order appoint under section three. "
) * 12


def test_arabic_words_are_unchanged_by_the_move() -> None:
    # The bands and the ingest text-layer gate are calibrated against these
    # exact sets, so the move out of the script pack must not alter an entry.
    # Pinned by content because the pack no longer carries a copy to diff.
    # One term left for a jurisdiction's config: a word naming one jurisdiction
    # is its data, not the language's, and `extend` folds it back in per run.
    ara = words_for("ara")
    assert ara is not None
    assert len(ara.function_words) == 28
    assert len(ara.lexicon_words) == 17
    assert {"في", "من", "على", "التي", "يكون"} <= ara.function_words
    assert {"قانون", "المادة", "مرسوم"} <= ara.lexicon_words
    assert ara.prose_floor == 80.0 and ara.lexicon_floor == 12.0


def test_arabic_still_tokenises_on_its_own_letters() -> None:
    # A bilingual page is scored on the language being judged, which is what
    # stops Latin boilerplate inflating an Arabic reading.
    ara = words_for("ara")
    assert ara is not None
    assert ara.is_letter("م") and not ara.is_letter("m")


def test_a_latin_language_tokenises_on_letters() -> None:
    ind = words_for("ind")
    assert ind is not None
    assert ind.is_letter("a") and not ind.is_letter("5")


def test_bahasa_prose_scores_as_prose() -> None:
    words = words_for("ind")
    assert function_word_rate(BAHASA, words) is not None
    assert text_verdict(BAHASA, words) == "prose"


def test_english_prose_scores_as_prose() -> None:
    words = words_for("eng")
    assert text_verdict(ENGLISH, words) == "prose"


def test_letter_spaced_bahasa_reads_as_damaged() -> None:
    # No function word survives tokenisation, and no legal noun either.
    words = words_for("ind")
    assert function_word_rate(LETTER_SPACED, words) == 0.0
    assert lexicon_rate(LETTER_SPACED, words) == 0.0
    assert text_verdict(LETTER_SPACED, words) == "damaged"


def test_a_language_with_no_list_is_unscored_rather_than_zero() -> None:
    # Absence of a list is not evidence of illegibility.
    assert words_for("ukr") is None
    assert function_word_rate(BAHASA, words_for("ukr")) is None
    assert text_verdict(BAHASA, words_for("ukr")) is None


def test_the_corpora_that_were_unscored_now_resolve_a_list() -> None:
    for code, language in (("xl", "ind"), ("xa", "eng"), ("xz", "ara"), ("gb", "eng")):
        config = load_config(code)
        assert config is not None
        assert config.authoritative_language == language, code
        assert words_for(config.authoritative_language) is not None, code


LETTER_SPACED_ENGLISH = (
    "T h i s A c t m a y b e c i t e d a s t h e F o r e s t r y A c t a n d "
) * 12


def test_letter_spaced_english_reads_as_damaged() -> None:
    # A first attempt listed "a" as an English function word, so this text
    # scored off bare glyphs and read as prose. A single-character entry can
    # never evidence a surviving word.
    words = words_for("eng")
    assert function_word_rate(LETTER_SPACED_ENGLISH, words) == 0.0
    assert text_verdict(LETTER_SPACED_ENGLISH, words) == "damaged"


def test_no_list_admits_a_single_character_entry() -> None:
    from codify.quality.lexicons import WordList, languages_with_lists

    for language in languages_with_lists():
        entry = words_for(language)
        assert entry is not None
        assert all(len(w) > 1 for w in entry.function_words | entry.lexicon_words), language
    with pytest.raises(ValueError, match="single-character"):
        WordList(language="zz", function_words=frozenset({"a"}))


def test_a_foreign_script_run_does_not_deflate_a_latin_reading() -> None:
    # str.isalpha() accepts Arabic, so an annex used to become tokens that
    # could only ever drag the rate down.
    english = "This Act may be cited as the Forestry Act and shall come into force. " * 12
    with_annex = english + "المادة الأولى من هذا القانون وعلى الوزراء تنفيذ هذا القرار " * 20
    words = words_for("eng")
    assert function_word_rate(english, words) == function_word_rate(with_annex, words)


def test_each_language_carries_its_own_measured_floors() -> None:
    # Arabic clean prose runs 91 to 176 per 1,000; English runs 213 to 433.
    # One global floor cannot describe both.
    for language in ("ara", "ind", "eng"):
        entry = words_for(language)
        assert entry is not None and entry.prose_floor is not None, language


def test_capitalised_prose_is_not_read_as_damage() -> None:
    # Legal text is full of headings and whole sections in capitals, and every
    # word list is lowercase. Casing is not evidence about letter forms.
    words = words_for("eng")
    lower = (
        "This Act may be cited as the Forestry Act and shall come into force "
        "on such date as the Minister may by order appoint under section three. "
    ) * 12
    assert function_word_rate(lower, words) == function_word_rate(lower.upper(), words)
    assert function_word_rate(lower, words) == function_word_rate(lower.title(), words)
    assert text_verdict(lower.upper(), words) == "prose"


def test_case_folding_leaves_a_caseless_script_alone() -> None:
    # Arabic has no case, so the fold must be a no-op there and its calibration
    # cannot move.
    arabic = "تسري أحكام هذا القانون على كل من يعمل في القطاع العام أو في القطاع الخاص. " * 40
    words = words_for("ara")
    assert function_word_rate(arabic, words) == function_word_rate(arabic.casefold(), words)
