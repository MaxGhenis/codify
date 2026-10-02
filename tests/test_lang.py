import re

import pytest

from codify.lang import number_run, to_iso639_3, word_bounded


def test_two_letter_codes_normalise_to_iso639_3():
    assert to_iso639_3("en") == "eng"
    assert to_iso639_3("pl") == "pol"
    assert to_iso639_3("uk") == "ukr"


def test_already_three_letter_passes_through():
    assert to_iso639_3("ukr") == "ukr"
    assert to_iso639_3("eng") == "eng"
    # Valid 639-3 outside the 639-1 map still passes.
    assert to_iso639_3("mlt") == "mlt"


def test_case_and_whitespace_insensitive():
    assert to_iso639_3("  EN ") == "eng"
    assert to_iso639_3("ENG") == "eng"


def test_english_language_names_fold():
    assert to_iso639_3("English") == "eng"
    assert to_iso639_3("  ARABIC ") == "ara"
    assert to_iso639_3("hebrew") == "heb"


def test_empty():
    assert to_iso639_3("") == ""
    assert to_iso639_3(None) == ""


def test_eu_official_languages_fold():
    assert to_iso639_3("et") == "est"
    assert to_iso639_3("mt") == "mlt"
    assert to_iso639_3("cs") == "ces"
    assert to_iso639_3("Estonian") == "est"


def test_locale_tags_fold_on_primary_subtag():
    assert to_iso639_3("en-US") == "eng"
    assert to_iso639_3("uk_UA") == "ukr"
    assert to_iso639_3("ar-EG") == "ara"


def test_off_table_three_letter_passes_with_warning():
    # Legitimate rare 639-3 (Latin) is not rejected.
    assert to_iso639_3("lat") == "lat"


def test_unknown_raises():
    with pytest.raises(ValueError):
        to_iso639_3("klingon")
    with pytest.raises(ValueError):
        to_iso639_3("e n")


@pytest.mark.parametrize(
    ("literal", "inside", "alone"),
    [
        ("Update", "Updated", "Update 3"),
        ("of", "thereof", "of 1991"),
        ("dated", "undated", "dated 1"),
    ],
    ids=["a marker", "a particle", "a cue"],
)
def test_a_latin_literal_matches_a_whole_word_only(literal: str, inside: str, alone: str) -> None:
    pattern = re.compile(word_bounded(literal))
    assert pattern.search(inside) is None
    assert pattern.search(alone) is not None


@pytest.mark.parametrize(
    ("literal", "text"),
    [("ฉบับที่", "(ฉบับที่ 3)"), ("B.E.", "B.E.2567"), ("No.", "(No.3)")],
    ids=["a script without spaces", "a trailing full stop", "a marker ending in a stop"],
)
def test_a_literal_without_a_latin_word_edge_is_not_bounded(literal: str, text: str) -> None:
    assert re.compile(word_bounded(literal)).search(text) is not None


@pytest.mark.parametrize(
    ("digits", "number"), [("03", "3"), ("0999", "999"), ("0", "0"), ("000", "0"), ("", "")]
)
def test_a_digit_run_is_its_number(digits: str, number: str) -> None:
    assert number_run(digits) == number
