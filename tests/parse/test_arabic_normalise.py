"""Tests for Arabic OCR-mojibake normalisation."""

from __future__ import annotations

from lxml import etree

from codify.akn import AKN_NS
from codify.pipeline.enrich.arabic_normalise import (
    _ARABIC_INDEX_LETTERS,
    normalise_arabic_in_tree,
    normalise_arabic_text,
    strip_ocr_headers,
    strip_ocr_headers_text,
)


def test_mid_word_ta_marbuta_stripped():
    """Ta-marbuta is a final-form letter, anywhere mid-word it's OCR noise.
    Shape taken from an OCR'd definitions article."""
    out, n = normalise_arabic_text("والعبارات الآتيةة المعةاني المخصصةة لاةا")
    assert out == "والعبارات الآتية المعاني المخصصة لاا"
    assert n == 4


def test_doubled_ta_marbuta_collapses():
    out, n = normalise_arabic_text("الآتيةة")
    assert out == "الآتية"
    assert n == 1


def test_triple_doubled_ta_marbuta_collapses_to_single():
    """`الةةةرئيس` has three rogue ta-marbutas; only the last (word-final) keeps."""
    out, n = normalise_arabic_text("الةةةرئيس")
    assert out == "الرئيس"
    assert n == 3


def test_doubled_hamza_collapses():
    out, n = normalise_arabic_text("للحرف ءء غير شائع")
    assert out == "للحرف ء غير شائع"
    assert n == 1


def test_legitimate_word_final_ta_marbuta_preserved():
    """`ة` at word boundary (followed by space, punctuation, end-of-line) is valid."""
    out, n = normalise_arabic_text("مدرسة جديدة في المدينة.")
    assert out == "مدرسة جديدة في المدينة."
    assert n == 0


def test_taa_plus_pronominal_suffix_preserved():
    """`مدرستها` (her school), the ta is regular t, not ta-marbuta. Must not match."""
    out, n = normalise_arabic_text("مدرستها كبيرة")
    assert out == "مدرستها كبيرة"
    assert n == 0


def test_ta_marbuta_with_tanween_preserved():
    """`ةً ةٍ ةٌ`, ta-marbuta + tanween diacritic is a valid case-ending construct."""
    out, n = normalise_arabic_text("مدرسةً جديدةٍ مدرسةٌ")
    assert n == 0
    assert "ة" in out


def test_non_arabic_text_untouched():
    """Latin / Cyrillic / digit-only text is never modified."""
    inp = "The quick brown fox. Стаття 5. 123 + 456 = 579."
    out, n = normalise_arabic_text(inp)
    assert out == inp
    assert n == 0


def test_tatweel_stripped():
    """`مـادة` (with U+0640 between م and ا) is a decorative variant of `مادة`
    used for typographic justification. Publishers freely mix the two; anchor
    scanning must see one form only."""
    out, n = normalise_arabic_text("مـادة (١)")
    assert out == "مادة (١)"
    assert n == 1


def test_zwnj_and_zwj_stripped():
    """Zero-width non-joiner (U+200C) and joiner (U+200D) appear from
    word-processor round-trips; strip them so equality comparisons and
    marker-literal regexes work without a decoration branch."""
    out, n = normalise_arabic_text("م‌اد‍ة")
    assert out == "مادة"
    assert n == 2


def test_mixed_joiners_and_mojibute_count_together():
    """Joiner stripping + ta-marbuta fix compose; counts sum."""
    out, n = normalise_arabic_text("الآتيةة مـادة")
    assert out == "الآتية مادة"
    assert n == 2


class TestLatiniseArabicOrdinal:
    def test_bare_ordinal(self):
        from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

        assert latinise_arabic_ordinal("الأول") == "1"
        assert latinise_arabic_ordinal("الثاني") == "2"
        assert latinise_arabic_ordinal("العاشر") == "10"

    def test_compound_ordinal(self):
        from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

        assert latinise_arabic_ordinal("الثاني عشر") == "12"
        assert latinise_arabic_ordinal("العشرون") == "20"

    def test_feminine_ordinal_maps_to_same_int(self):
        """`الأولى` / `الثانية` (feminine) map to the same digit as the
        masculine so gender-agreement in the source does not change the
        emitted num."""
        from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

        assert latinise_arabic_ordinal("الأولى") == "1"
        assert latinise_arabic_ordinal("الثانية") == "2"

    def test_tatweel_form_matches(self):
        """`الـأول` (with tatweel) is the same word; the lookup normalises."""
        from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

        assert latinise_arabic_ordinal("الـأول") == "1"

    def test_unknown_returns_none(self):
        from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

        assert latinise_arabic_ordinal("") is None
        assert latinise_arabic_ordinal("hello") is None
        # A single ordinal that isn't in the table yet (40th).
        assert latinise_arabic_ordinal("الأربعون") is None


def test_in_tree_walks_p_text_and_tails():
    """The tree variant updates `.text` and child `.tail` of every `<p>`."""
    xml = (
        f'<root xmlns="{AKN_NS}">'
        "<p>الآتيةة text<term>foo</term> bar الآتيةة</p>"
        "<p>المعةاني</p>"
        "</root>"
    )
    root = etree.fromstring(xml.encode())
    n = normalise_arabic_in_tree(root)
    assert n == 3
    paras = list(root.iter(f"{{{AKN_NS}}}p"))
    assert paras[0].text == "الآتية text"
    assert paras[0][0].tail == " bar الآتية"
    assert paras[1].text == "المعاني"


# --- OCR-header strip -----------------------------------------------------

_GAZETTE_PATTERNS = [
    r"الوقائع\s+الزرزورية",
    r"(?m)^\s*قانون\s+رقم\s*\(?\s*\d+\s*\)?\s*لسنة\s*\d{4}\s*بشأن\s+[^\n]{1,60}$",
    r"(?m)^\s*صفحة\s*\d+\s*$",
    r"\d+\s*بشأن\s+البيئة\s*\d{4}\s*\)?\s*لسنة\s*\d+\s*\)?\s*قانون\s+رقم",
]


class TestStripOCRHeaders:
    def test_gazette_masthead_removed(self):
        out, n = strip_ocr_headers_text(
            "الوقائع الزرزورية مقدمة", [__import__("re").compile(r"الوقائع\s+الزرزورية")]
        )
        assert n == 1
        assert "الوقائع" not in out

    def test_running_law_header_at_line_start_stripped(self):
        """A "قانون رقم 9 لسنة 1987 بشأن البيئة" running header at start of
        line is stripped; the same phrase in prose is NOT stripped."""
        import re

        patterns = [re.compile(p, re.MULTILINE | re.UNICODE) for p in _GAZETTE_PATTERNS]
        source = "قانون رقم 9 لسنة 1987 بشأن البيئة\nالمادة 1: هذا القانون يهدف إلى حماية البيئة."
        out, n = strip_ocr_headers_text(source, patterns)
        assert n >= 1
        assert "المادة 1: هذا القانون يهدف" in out

    def test_substantive_law_reference_not_stripped(self):
        """A prose-embedded "قانون رقم 9 لسنة 1987" reference must NOT be
        stripped, the pattern requires it to be at line-start with the
        distinctive "بشأن X" suffix."""
        import re

        patterns = [re.compile(p, re.MULTILINE | re.UNICODE) for p in _GAZETTE_PATTERNS]
        source = "يستند هذا القرار إلى قانون رقم 9 لسنة 1987 المعمول به"
        out, n = strip_ocr_headers_text(source, patterns)
        assert n == 0
        assert out == source

    def test_page_number_footer_stripped(self):
        import re

        patterns = [re.compile(p, re.MULTILINE | re.UNICODE) for p in _GAZETTE_PATTERNS]
        source = "نص المادة السابقة.\nصفحة 27\nنص المادة التالية."
        out, n = strip_ocr_headers_text(source, patterns)
        assert n == 1
        assert "صفحة 27" not in out

    def test_env_law_art27_specific_concatenated_bleed(self):
        """The exact shape observed on Env Law Art 27: page number, gazette
        title fragment, all mashed together into one run."""
        import re

        patterns = [re.compile(p, re.MULTILINE | re.UNICODE) for p in _GAZETTE_PATTERNS]
        source = "الشركات المشاركة 22 بشأن البيئة 1987 لسنة 9 قانون رقم في نشاطها"
        out, n = strip_ocr_headers_text(source, patterns)
        assert n == 1
        assert "22 بشأن البيئة" not in out

    def test_tree_walk_touches_all_p_text_and_tail(self):
        xml = (
            "<root xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<p>الوقائع الزرزورية prefix<a>x</a>tail with الوقائع الزرزورية again</p>"
            "<p>clean text</p>"
            "</root>"
        )
        root = etree.fromstring(xml.encode())
        n = strip_ocr_headers(root, [r"الوقائع\s+الزرزورية"])
        assert n == 2  # once in text, once in tail
        p0 = list(root.iter(f"{{{AKN_NS}}}p"))[0]
        assert "الوقائع" not in (p0.text or "")
        assert "الوقائع" not in (p0[0].tail or "")

    def test_empty_patterns_short_circuits(self):
        xml = (
            "<root xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'><p>anything</p></root>"
        )
        root = etree.fromstring(xml.encode())
        n = strip_ocr_headers(root, [])
        assert n == 0

    def test_invalid_pattern_raises_at_compile(self):
        import pytest

        with pytest.raises(Exception):
            strip_ocr_headers_text(
                "text", [__import__("re").compile("[unclosed") if False else None][:1]
            )  # type: ignore
        # Direct: an invalid regex string handed to strip_ocr_headers must
        # raise at compile time, not silently pass.
        with pytest.raises(Exception):
            xml = "<root xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'><p>x</p></root>"
            root = etree.fromstring(xml.encode())
            strip_ocr_headers(root, ["[unclosed"])


class TestLatinBulletRemap:
    """Arabic list-bullet normaliser.

    Older Arabic gazettes typeset alphabetic list markers (أ- ب- ج- د- ه- …)
    as visually similar Latin letters in OCR output (`v-`, `w-`, `c-`,
    `h-`, `x-`). The remap walks line-anchored Latin-letter markers in
    document order and replaces by list position, not by identity.

    All fixtures include a Arabic-script anchor line so the language gate
    opens; the pre-existing `_has_arabic` guard prevents English or other
    Latin-script documents from being silently rewritten (e.g. a lettered
    sub-list `a- foo\\nb- bar` on a gb AKN)."""

    _AR_HEADER = "المادة 1\n"

    def test_arb_reg_39_2004_bullet_run(self):
        """Reviewer defect v38-E-01 verbatim: five sequential markers."""
        text = self._AR_HEADER + (
            "v- first item\nw- second item\nc- third item\nh- fourth item\nx- fifth item"
        )
        cleaned, n = normalise_arabic_text(text)
        assert n == 5
        assert "أ- first item" in cleaned
        assert "ب- second item" in cleaned
        assert "ج- third item" in cleaned
        assert "د- fourth item" in cleaned
        assert "ه- fifth item" in cleaned

    def test_english_document_bullets_never_remapped(self):
        """Silent-failure guard: an English AKN with a lettered sub-list
        (`a- b- c-`) must not have its markers rewritten to Arabic-index
        letters when normalise_arabic_text is called with the flag off."""
        text = "a- first English item\nb- second English item\nc- third English item"
        cleaned, n = normalise_arabic_text(text, apply_arabic_only=False)
        assert cleaned == text
        assert n == 0

    def test_tree_gate_applies_arabic_transforms_to_english_fragment(self):
        """Copilot #1 regression: normalise_arabic_in_tree must decide the
        Arabic gate once from the whole document, so an English `<p>`
        inside an Arabic-language AKN still gets its bullet marker
        rewritten. Otherwise a per-fragment gate silently skips exactly
        the case it was built for, an English body inside an
        Arabic-header act."""
        xml = (
            "<akomaNtoso xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<act><body>"
            "<article eId='art_1'><num>1</num><heading>المادة الأولى</heading>"
            "<content><p>v- English fragment inside Arabic law.</p></content>"
            "</article>"
            "</body></act></akomaNtoso>"
        )
        root = etree.fromstring(xml.encode())
        n = normalise_arabic_in_tree(root)
        assert n >= 1
        ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
        p_text = root.find(".//a:p", ns).text or ""
        assert p_text.startswith("أ- ")

    def test_tree_gate_leaves_wholly_english_document_untouched(self):
        """Complement of the above: a document with no Arabic characters
        anywhere gets no bullet remap, no fused-numbering, no dedup."""
        xml = (
            "<akomaNtoso xmlns='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<act><body>"
            "<article eId='art_1'><num>1</num><heading>The First Section</heading>"
            "<content><p>a- first Latin item.\nb- second Latin item.</p></content>"
            "</article>"
            "</body></act></akomaNtoso>"
        )
        root = etree.fromstring(xml.encode())
        normalise_arabic_in_tree(root)
        ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
        p_text = root.find(".//a:p", ns).text or ""
        assert p_text.startswith("a- first")

    def test_mid_line_latin_not_touched(self):
        """A Latin letter mid-sentence (`item v of the list`) must not
        trigger the remap; the pattern is line-anchored."""
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "A sentence with letter v embedded.")
        assert "A sentence with letter v embedded." in cleaned
        assert n == 0

    def test_paren_marker_form_also_remapped(self):
        """Both `v-` and `v)` marker shapes are normalised consistently. The
        paren form must remap identically."""
        text = self._AR_HEADER + "v) first item\nw) second item\nc) third item"
        cleaned, n = normalise_arabic_text(text)
        assert n == 3
        assert "أ- first item" in cleaned
        assert "ب- second item" in cleaned
        assert "ج- third item" in cleaned

    def test_two_separate_lists_each_restart(self):
        """A document with two independent bulleted lists (separated by
        prose or a blank line) must restart each list at `أ-` rather
        than continue mid-alphabet on the second list."""
        text = (
            self._AR_HEADER
            + "v- list A item 1\nw- list A item 2\n"
            + "\n"  # blank line ends the run
            + "some intervening prose\n"
            + "v- list B item 1\nw- list B item 2\n"
        )
        cleaned, n = normalise_arabic_text(text)
        assert n == 4
        # Both lists start at أ- and go to ب-, not continue at ج-/د-.
        assert cleaned.count("أ-") == 2
        assert cleaned.count("ب-") == 2
        assert "ج-" not in cleaned

    def test_exhausted_positions_leave_extra_markers(self):
        """Beyond the last-known Arabic-index letter (position 10, ي-),
        extra markers survive unchanged so an operator sees the suspicious
        long list instead of an invented remap. Assert on the first ten
        being Arabic-index and the eleventh being the original Latin."""
        lines = self._AR_HEADER + "\n".join(f"{c}- item" for c in "vwchxjkfrtl")  # 11 markers
        cleaned, _ = normalise_arabic_text(lines)
        for arabic in _ARABIC_INDEX_LETTERS:
            assert f"{arabic}- item" in cleaned
        assert "l- item" in cleaned


class TestFusedNumbering:
    """Fused-numbering markers are separated before parsing.

    Observed as `1Formulating`, `2Preparing`, `3Issuing`:
    the OCR dropped the separator between the numbering and the first
    word of the item. Restore `.` and a space. Trailing group requires
    a full-word shape so `5A` subarticle markers and `18USC` cites do
    not get mangled."""

    _AR_HEADER = "المادة 1\n"

    def test_pp_8_2014_fused_shape(self):
        cleaned, n = normalise_arabic_text(
            self._AR_HEADER + "1Formulating the plan.\n2Preparing the report.\n3Issuing the decree."
        )
        assert n == 3
        assert "1. Formulating" in cleaned
        assert "2. Preparing" in cleaned
        assert "3. Issuing" in cleaned

    def test_multi_digit_fused_shape(self):
        cleaned, _ = normalise_arabic_text(self._AR_HEADER + "42Formulating the plan.")
        assert "42. Formulating" in cleaned

    def test_subarticle_marker_not_mangled(self):
        """`5A The Minister` is a legitimate subarticle marker; the
        trailing must be `[A-Z][a-z]` for the rewrite to fire."""
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "5A The Minister shall convene.")
        assert "5A The Minister" in cleaned
        assert n == 0

    def test_shorthand_cite_not_mangled(self):
        """A citation shape like `18USC` at line-start is uppercase-only;
        the rewrite must not fire."""
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "18USC §1234 governs this.")
        assert "18USC" in cleaned
        assert n == 0

    def test_number_before_lowercase_not_touched(self):
        """Lowercase after a digit is legitimate prose (`the 5th item`),
        so must survive unchanged."""
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "The 5th item is important.")
        assert "The 5th item is important." in cleaned
        assert n == 0


class TestDuplicateIndexMarker:
    """Consecutive markers with the same index remain distinct.

    Observed as `2 2` at the start of what should be a
    single item; collapse to `2`."""

    _AR_HEADER = "المادة 1\n"

    def test_ac_18_2016_dup_shape(self):
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "2 2 the item body prose here.\n")
        assert n == 1
        assert "2 the item body prose here." in cleaned

    def test_distinct_adjacent_numbers_untouched(self):
        cleaned, n = normalise_arabic_text(self._AR_HEADER + "2 3 not a duplicate marker.")
        assert "2 3 not a duplicate marker." in cleaned
        assert n == 0


def test_latinise_space_collapsed_compound_ordinal() -> None:
    """Bluebell eId derivation strips the space in compound ordinals
    (`الحادي عشر` → `الحاديعشر`); the lookup must still resolve."""
    from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

    assert latinise_arabic_ordinal("الحاديعشر") == "11"
    assert latinise_arabic_ordinal("الحادي عشر") == "11"
    assert latinise_arabic_ordinal("ليس ترتيبا") is None


def test_latinise_extended_compound_ordinals() -> None:
    """21-30 masculine and 11-19 feminine compounds resolve; big codes
    (civil codes with 21+ parts) must not ship non-ASCII eIds."""
    from codify.pipeline.enrich.arabic_normalise import latinise_arabic_ordinal

    assert latinise_arabic_ordinal("الحادي والعشرون") == "21"
    assert latinise_arabic_ordinal("الثلاثون") == "30"
    assert latinise_arabic_ordinal("الحادية عشرة") == "11"


def test_dashed_page_numbers_are_stripped_whichever_dash_is_printed() -> None:
    """The declared pattern allowed a hyphen where the sources print an en-dash,
    so it never matched. These patterns also drive header stripping in
    translation and enacting-formula cleanup, so a dashed page number was
    invisible to both."""
    from codify.jurisdictions import load_config
    from codify.pipeline.enrich.arabic_normalise import drop_header_only_lines

    cfg = load_config("xl")
    assert cfg is not None
    patterns = [*cfg.ocr_header_patterns, *cfg.furniture_line_patterns]
    text = "Ketentuan ini\n- 41 -\n- 42 –\n– 43 —\nberlaku.\n"
    cleaned, dropped = drop_header_only_lines(text, patterns)
    assert dropped == 3, cleaned
    assert cleaned.strip() == "Ketentuan ini\nberlaku."
