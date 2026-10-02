"""Script selection driven by the jurisdiction config.

The structurer used to decide "is this Arabic?" by counting codepoints at
runtime, and the Arabic tables sat inline in `anchors.py` and `validator.py`,
so a gb or eu-27 ingest ran Arabic handling it had no use for. Every config
already declares `display.script`; that declaration is now the selector and
codepoint detection is the fallback.
"""

from __future__ import annotations

from lxml import etree

from codify.jurisdictions import load_config
from codify.pipeline.enrich.arabic_normalise import normalise_arabic_in_tree
from codify.pipeline.enrich.scripts import (
    ARABIC_SCRIPT,
    detect_pack,
    dominant_script,
    pack_for,
    pack_for_config,
    resolve_pack,
)
from codify.pipeline.enrich.scripts.arabic import ARABIC

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def test_a_pack_is_found_without_anything_importing_it_first() -> None:
    """A lookup must not depend on which module happened to import a pack."""
    assert pack_for(ARABIC_SCRIPT) is ARABIC


def test_a_script_with_no_pack_returns_none() -> None:
    """None means "no script-specific handling", which is the right state for
    a Latin-script jurisdiction: the general algorithms carry it."""
    assert pack_for("latin") is None
    assert pack_for(None) is None


def test_the_config_declaration_selects_the_pack() -> None:
    assert pack_for_config(load_config("xz")) is ARABIC
    assert pack_for_config(load_config("gb")) is None


def test_content_is_the_fallback_when_the_config_declares_no_pack() -> None:
    """A document whose content disagrees with its jurisdiction still gets the
    handling its text needs."""
    assert resolve_pack(load_config("gb"), "المادة الأولى من هذا القانون") is ARABIC
    assert resolve_pack(load_config("gb"), "Section 1 of this Act") is None


def test_the_config_wins_over_the_content() -> None:
    """An Arabic act quoting an English treaty is still an Arabic act."""
    assert resolve_pack(load_config("xz"), "This Agreement shall enter into force") is ARABIC


def test_detection_ignores_a_stray_word() -> None:
    """One Arabic term inside English prose is a quotation, not a script."""
    assert detect_pack("The term حسبة appears once in this English sentence.") is None


def test_detection_needs_letters_to_look_at() -> None:
    assert detect_pack("123 456") is None
    assert detect_pack("") is None


def test_dominant_script_answers_for_scripts_no_pack_covers() -> None:
    """Parenthesised-alias filtering asks "same writing system?" of scripts
    that have no pack, so it stays general rather than per-pack."""
    assert dominant_script("Член") == "CYRILLIC"
    assert dominant_script("მუხლი") == "GEORGIAN"
    assert dominant_script("مادة") == "ARABIC"
    assert dominant_script("123") is None


def _doc(text: str) -> etree._Element:
    return etree.fromstring(
        f'<akomaNtoso xmlns="{NS}"><act><body><p>{text}</p></body></act></akomaNtoso>'.encode()
    )


def test_a_latin_jurisdiction_does_not_run_the_arabic_pass(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The pass was unconditional, so a gb ingest could have a lettered
    sub-list mapped onto Arabic-index letters."""
    seen: list[bool] = []
    import codify.pipeline.enrich.arabic_normalise as module

    original = module.normalise_arabic_text

    def spy(text: str, *, apply_arabic_only: bool = True) -> tuple[str, int]:
        seen.append(apply_arabic_only)
        return original(text, apply_arabic_only=apply_arabic_only)

    monkeypatch.setattr(module, "normalise_arabic_text", spy)
    normalise_arabic_in_tree(_doc("Paragraph (a) of section 3 applies."), country="gb")
    assert seen == [False]


def test_an_arabic_jurisdiction_runs_it(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: list[bool] = []
    import codify.pipeline.enrich.arabic_normalise as module

    original = module.normalise_arabic_text

    def spy(text: str, *, apply_arabic_only: bool = True) -> tuple[str, int]:
        seen.append(apply_arabic_only)
        return original(text, apply_arabic_only=apply_arabic_only)

    monkeypatch.setattr(module, "normalise_arabic_text", spy)
    normalise_arabic_in_tree(_doc("المادة (1) من هذا القانون"), country="xz")
    assert seen == [True]


def test_an_unknown_jurisdiction_falls_back_to_the_content(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """No config to read, so the document's own script decides."""
    seen: list[bool] = []
    import codify.pipeline.enrich.arabic_normalise as module

    original = module.normalise_arabic_text

    def spy(text: str, *, apply_arabic_only: bool = True) -> tuple[str, int]:
        seen.append(apply_arabic_only)
        return original(text, apply_arabic_only=apply_arabic_only)

    monkeypatch.setattr(module, "normalise_arabic_text", spy)
    normalise_arabic_in_tree(_doc("المادة (1) من هذا القانون"), country="")
    assert seen == [True]


def test_the_arabic_pack_carries_the_data_the_structurer_moved_out_of_anchors() -> None:
    """Guards the boundary: these tables were inline in the general modules,
    where a second script meant editing anchors.py again."""
    assert ARABIC.ordinal_to_int
    assert len(ARABIC.letter_order) == 28
    assert ARABIC.letter_alias == {"ه": "هـ"}
    assert ARABIC.prose_precursors and "بموجب" in ARABIC.prose_precursors
    assert ARABIC.sameline_precursors == ("مخالفة",)
    assert dict(ARABIC.ordinal_list_markers)["أولاً"] == "1"
    assert ARABIC.is_letter("م") and not ARABIC.is_letter("A")
    assert ARABIC.digit_re.match("(٣)") and ARABIC.plural_marker_re.search("المواد")


def test_anchors_and_validator_read_the_pack_rather_than_their_own_copies() -> None:
    """The point of the move: one definition per script, not one per module."""
    from codify.pipeline.enrich import anchors, validator

    assert anchors._AR_PROSE_PRECURSORS is ARABIC.prose_precursors
    assert anchors._AR_ORDINAL_LIST_MARKERS is ARABIC.ordinal_list_markers
    assert anchors._AR_SAMELINE_PRECURSORS is ARABIC.sameline_precursors
    assert anchors._ARABIC_JOINER_OPT is ARABIC.joiner_opt
    assert anchors._PLURAL_MARKER_RE is ARABIC.plural_marker_re
    assert validator._ARABIC_LETTER_ORDER is ARABIC.letter_order
    assert validator._ARABIC_LETTER_ALIAS is ARABIC.letter_alias
    assert validator._ARABIC_DIGIT_RE is ARABIC.digit_re
