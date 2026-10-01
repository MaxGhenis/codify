"""Deterministic page-read signals, and the abstention contract around them."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from codify.pipeline.enrich.scripts.arabic import ARABIC
from codify.quality.page_read import (
    chars_per_ink,
    divergence,
    ink_ratio,
    letter_spaced_runs,
    measured,
    numeral_parity,
    presentation_form_share,
    repeat_density,
    score_page,
    script_purity,
)

_CLEAN_AR = (
    "المادة ١ تسري أحكام هذا النظام على الأدوات المقيدة في السجل وعلى من "
    "يزاول النشاط في نطاق الموانئ بما لا يخل بالشروط المقررة في اللائحة"
)


def _png(shade: int, size: int = 40) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (size, size), shade).save(buf, format="PNG")
    return buf.getvalue()


class TestInkRatio:
    def test_a_black_page_is_all_ink_and_a_white_page_is_none(self) -> None:
        assert ink_ratio(_png(0)) == 1.0
        assert ink_ratio(_png(255)) == 0.0

    def test_undecodable_bytes_abstain(self) -> None:
        assert ink_ratio(b"not an image") is None

    def test_a_broken_pillow_is_not_an_unreadable_page(self, monkeypatch) -> None:
        """Swallowing ImportError here would report every page as unmeasurable
        and read as a corpus of blank scans."""
        import builtins

        # Built before the hook goes in, or _png raises and the assertion passes
        # without ink_ratio ever being called.
        image = _png(0)
        real = builtins.__import__

        def _no_pillow(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("no pillow")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pillow)
        with pytest.raises(ImportError):
            ink_ratio(image)


class TestCharsPerInk:
    def test_a_page_dense_with_ink_returning_little_text_scores_low(self) -> None:
        assert chars_per_ink("x" * 20, 0.5) < chars_per_ink("x" * 2000, 0.5)

    def test_a_blank_render_abstains_rather_than_dividing(self) -> None:
        assert chars_per_ink("x" * 100, 0.0) is None
        assert chars_per_ink("x" * 100, None) is None


class TestRepeatDensity:
    def test_a_degeneracy_loop_scores_high(self) -> None:
        looped = "المادة الأولى تسري أحكام هذا القانون على كل من " * 12
        density = repeat_density(looped)
        assert density is not None and density > 0.8

    def test_ordinary_prose_scores_zero(self) -> None:
        assert repeat_density(_CLEAN_AR) == 0.0

    def test_text_too_short_to_judge_abstains(self) -> None:
        assert repeat_density("المادة ١") is None


class TestScriptPurity:
    def test_expected_script_is_pure(self) -> None:
        assert script_purity(_CLEAN_AR, ARABIC) == 1.0

    def test_a_read_that_drifted_to_another_script_scores_zero(self) -> None:
        assert script_purity("the quick brown fox jumps over the lazy dog again", ARABIC) == 0.0

    def test_no_script_pack_abstains(self) -> None:
        assert script_purity(_CLEAN_AR, None) is None


class TestPresentationForms:
    def test_standard_arabic_carries_no_presentation_forms(self) -> None:
        assert presentation_form_share(_CLEAN_AR) == 0.0

    def test_display_glyphs_are_detected(self) -> None:
        share = presentation_form_share("ﻟﺎﺮﺳ" * 20)
        assert share is not None and share > 0.9

    def test_digits_and_brackets_cannot_dilute_a_mangled_read(self) -> None:
        """The denominator is letters. Counting every non-space character let a
        page of article numbers pull a wholly glyph-mangled read toward clean."""
        mangled = "ﻟﺎﺮﺳ" * 20
        with_numbering = mangled + " (١٢٣) (٤٥٦) [٧٨٩] " * 20

        assert presentation_form_share(with_numbering) == presentation_form_share(mangled)


class TestMeasured:
    """The trap: four of five signals abstain on a short read, and a short read
    is one of the failures they exist to catch. Nothing may read all-abstained
    as clean."""

    def test_a_short_bad_read_abstains_nearly_everywhere(self) -> None:
        text = "no"
        signals = {
            "chars_per_ink": chars_per_ink(text, 0.4),
            "repeat_density": repeat_density(text),
            "presentation_forms": presentation_form_share(text),
            "script_purity": script_purity(text, ARABIC),
        }

        assert measured(signals) == ["chars_per_ink"]

    def test_nothing_measurable_is_distinguishable_from_nothing_wrong(self) -> None:
        nothing = dict.fromkeys(("chars_per_ink", "repeat_density", "script_purity"))
        clean = {"chars_per_ink": 900.0, "repeat_density": 0.0, "script_purity": 1.0}

        assert measured(nothing) == []
        assert measured(clean) == ["chars_per_ink", "repeat_density", "script_purity"]


class TestDivergence:
    """How far apart two engines read the same page. A proxy, not a ranking."""

    def test_identical_reads_do_not_diverge(self) -> None:
        assert divergence(_CLEAN_AR, _CLEAN_AR) == 0.0

    def test_reads_sharing_nothing_diverge_completely(self) -> None:
        assert divergence("alpha beta gamma", "delta epsilon zeta") == 1.0

    def test_order_and_repetition_are_ignored(self) -> None:
        """Token sets, so a reordered read is not a disagreement. That ceiling
        is why this localises doubt rather than measuring quality."""
        assert divergence("a b c", "c b a") == 0.0
        assert divergence("a b c", "a a b b c c") == 0.0

    def test_an_empty_side_abstains(self) -> None:
        assert divergence("", _CLEAN_AR) is None
        assert divergence(_CLEAN_AR, "   ") is None


class TestNumeralParity:
    """Figures, not prose. A summarising read drops amounts a faithful read keeps."""

    def test_a_dropped_figure_lowers_parity(self) -> None:
        assert numeral_parity("غرامة 500 دينار مادة 12", "غرامة دينار مادة 12") == 0.5

    def test_a_read_that_drops_every_figure_scores_zero_not_none(self) -> None:
        """The exact missing-penalty-figure case: one read keeps the amounts, the
        other has none. That must score 0, the strongest signal, not abstain."""
        assert numeral_parity("غرامة 500 دينار", "غرامة دينار فقط") == 0.0

    def test_the_same_figure_in_either_digit_script_agrees(self) -> None:
        assert numeral_parity("fine 500", "غرامة ٥٠٠") == 1.0

    def test_only_two_figureless_reads_abstain(self) -> None:
        assert numeral_parity("نص بلا أرقام", "نص آخر بلا أي أرقام") is None


class TestScorePage:
    def test_a_clean_read_with_no_flags_is_accepted(self) -> None:
        verdict = score_page(_CLEAN_AR, pack=ARABIC)
        assert verdict.verdict == "accept"
        assert verdict.reasons == []
        # It accepted because signals ran and passed, not because none could run.
        assert measured(verdict.signals)

    def test_a_dense_page_returning_little_text_escalates_on_ink_alone(self) -> None:
        """The 2026-08-02 failure mode: an inked page transcribed to a fragment,
        caught from the image and a character count with no second engine."""
        verdict = score_page("مادة ٧٩", ink=1.0)
        assert verdict.verdict == "escalate"
        assert "chars_per_ink:low" in verdict.reasons

    def test_a_degeneracy_loop_escalates(self) -> None:
        looped = "المادة الأولى تسري أحكام هذا القانون على كل من " * 12
        assert score_page(looped, pack=ARABIC).verdict == "escalate"

    def test_readers_that_disagree_are_sent_for_a_reread(self) -> None:
        verdict = score_page("alpha beta gamma delta", rival_text="one two three four")
        assert verdict.verdict == "reread"
        assert "divergence:high" in verdict.reasons

    def test_a_near_blank_page_abstains_rather_than_escalating(self) -> None:
        """Empty text on a page below the blank-ink floor is a blank leaf, not a
        summarising read; chars_per_ink must abstain, not score it a failure."""
        verdict = score_page("", ink=0.004)
        assert "chars_per_ink:low" not in verdict.reasons
        assert verdict.verdict != "escalate"

    def test_an_inked_page_that_returned_nothing_still_escalates(self) -> None:
        """Above the blank floor, an empty read is a total failure, not a leaf."""
        assert score_page("", ink=0.5).verdict == "escalate"

    def test_script_drift_is_caught_against_the_declared_pack(self) -> None:
        """A wholly Latin hallucination on an Arabic page is drift when scored
        against the declared Arabic pack; scored against its own detected script
        (the trap) it would look pure."""
        latin = "the quick brown fox jumps over the lazy dog again and again once more"
        assert score_page(latin, pack=ARABIC).verdict == "escalate"

    def test_a_page_nothing_could_measure_is_reread_not_accepted(self) -> None:
        """Unmeasured is not clean: a near-empty read abstains everywhere, and a
        near-empty read is itself the failure."""
        verdict = score_page("no")
        assert verdict.verdict == "reread"
        assert verdict.reasons == ["unmeasured"]

    def test_reasons_are_machine_readable_and_sorted(self) -> None:
        looped = "ﻟﺎﺮﺳ ﻟﺎﺮﺳ ﻟﺎﺮﺳ " * 20  # presentation forms and a loop
        reasons = score_page(looped).reasons
        assert reasons == sorted(reasons)
        assert all(":" in r for r in reasons)


class TestFaintScanBaseline:
    """A recorded calibration target, not a pass. A faint scan (ink ~0.027) means
    a summarising read of it clears the provisional absolute floor; robustly
    flagging such a page needs contrast-aware calibration. This pins that gap on a
    synthetic faint scan whose ink density stands in for the real-world case."""

    fixture = Path(__file__).parent.parent / "fixtures" / "ocr" / "synthetic-faint-p1.png"

    def test_the_scan_is_faint_not_a_dense_page(self) -> None:
        ink = ink_ratio(self.fixture.read_bytes())
        assert ink is not None and 0.02 < ink < 0.05

    def test_a_summarising_read_of_a_faint_scan_clears_the_absolute_floor(self) -> None:
        ink = ink_ratio(self.fixture.read_bytes())
        # ~200 chars over ink 0.027 is ~7.4k, above the 5k floor: the miss calibration must close.
        assert score_page("م" * 200, ink=ink).verdict == "accept"


class TestLetterSpacedRuns:
    def test_counts_a_damaged_word(self) -> None:
        assert letter_spaced_runs("A l a m a t : Dulang, RT. 001/RW. 000") == 1

    def test_counts_separated_runs_separately(self) -> None:
        """Adjacent damaged words merge into one run, which is what a count of
        damaged regions should say."""
        assert letter_spaced_runs("d i t e r i m a o l e h") == 1
        assert letter_spaced_runs("d i t e r i m a oleh M a h k a m a h") == 2

    def test_ignores_clean_prose(self) -> None:
        assert letter_spaced_runs("Mahkamah berpendapat bahwa dalil para Pemohon") == 0

    def test_ignores_a_lettered_list(self) -> None:
        """`a. b. c.` enumeration is the shape most likely to be mistaken for
        letter-spacing, and it keeps its stops."""
        assert letter_spaced_runs("a. Ketentuan b. Penjelasan c. Lampiran d. Sanksi") == 0

    def test_ignores_three_letters(self) -> None:
        """Initials and short abbreviations run to three; the floor sits above."""
        assert letter_spaced_runs("ditandatangani M A S di Kota Langkasuka") == 0

    def test_ignores_arabic(self) -> None:
        assert letter_spaced_runs(_CLEAN_AR) == 0

    def test_a_run_never_spans_a_line_break(self) -> None:
        assert letter_spaced_runs("a b c\nd e f") == 0

    def test_matches_extended_latin_and_hard_spaces(self) -> None:
        """A damaged accented-Latin page splits accented letters, and a bad text
        layer separates with NBSP as readily as with a space."""
        assert letter_spaced_runs("Ç \u00eb s h t j e") == 1
        assert letter_spaced_runs("A\u00a0l\u00a0a\u00a0m\u00a0a\u00a0t") == 1

    def test_scores_the_text_the_caller_cleaned(self) -> None:
        """The count is only meaningful on cleaned text: the damaged layer usually
        double-spaces, which reads as nothing until cleaning collapses it."""
        from codify.pipeline.enrich.ocr import clean_page_text

        assert letter_spaced_runs("A  l  a  m  a  t") == 0
        assert letter_spaced_runs(clean_page_text("A  l  a  m  a  t")) == 1

    def test_a_column_aligned_table_is_a_known_false_positive(self) -> None:
        """Cleaning collapses column padding, so single-letter columns read as a
        run. Recorded rather than guarded: measured at zero pages across several
        thousand judgment pages."""
        from codify.pipeline.enrich.ocr import clean_page_text

        assert letter_spaced_runs(clean_page_text("No.   Pasal   A     B     C     D")) == 1

    def test_ignores_the_maths_glyphs_inside_the_latin_block(self) -> None:
        """The Latin-1 range carries multiplication and division signs."""
        assert letter_spaced_runs("a \u00d7 b \u00f7 c d") == 0

    def test_counts_a_deliberately_spaced_heading(self) -> None:
        """Typographic letter-spacing is the same shape as damage and cannot be
        told apart by shape alone. Recorded, not guarded: no spaced heading
        appeared in the corpora this was calibrated against."""
        assert letter_spaced_runs("P U T U S A N") == 1
