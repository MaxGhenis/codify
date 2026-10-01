"""Rate-based legibility tests.

Private calibration fixtures are not distributed; their checks skip when absent.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from codify.jurisdictions import load_config
from codify.quality.legibility import text_verdict, verdict_from_rates
from codify.quality.lexicons import words_for

MARKED = Path(__file__).resolve().parents[1] / "fixtures/legibility/marked.jsonl"
ARABIC = words_for("ara")
assert ARABIC is not None


def _verdict(row: dict[str, float | None]) -> str | None:
    """The production rule over pre-measured rates, so the fixture needs no text
    and a change to the rule still moves these numbers."""
    return verdict_from_rates(row["function_word_rate"], row["lexicon_rate"])


def _rows() -> list[dict]:
    if not MARKED.exists():
        pytest.skip("legibility/marked.jsonl not in the open tree")
    return [json.loads(line) for line in MARKED.read_text(encoding="utf-8").splitlines() if line]


class TestCalibration:
    def test_the_verdict_never_calls_clean_text_damaged(self) -> None:
        """The floor is set for precision, not recall. A false damage call sends
        a reviewer to re-OCR a document that reads perfectly well, and the OCR
        decision is not this function's job in the first place."""
        wrong = [r["id"] for r in _rows() if r["mark"] != "damaged" and _verdict(r) == "damaged"]
        assert wrong == []

    def test_it_recognises_readable_prose(self) -> None:
        marked = [r for r in _rows() if r["mark"] == "prose"]
        assert marked and all(_verdict(r) == "prose" for r in marked)

    def test_it_catches_under_half_the_damage_and_that_is_the_ceiling(self) -> None:
        """Recorded so the number is not mistaken for a damage detector. Six of
        thirteen: the rest corrupt interior letters while leaving the instrument
        names spelled correctly. Deciding OCR is the page-level gate's job, and
        it is measured separately, per page rather than per document."""
        rows = _rows()
        caught = sum(1 for r in rows if r["mark"] == "damaged" and _verdict(r) == "damaged")
        assert (caught, sum(1 for r in rows if r["mark"] == "damaged")) == (6, 13)

    def test_it_abstains_rather_than_guessing_on_short_documents(self) -> None:
        """Five of the ten marked `not_prose` carry too few tokens for a prose
        rate. None is the honest answer there; calling them `not_prose` would be
        a guess that happens to be right."""
        rows = _rows()
        assert sum(1 for r in rows if _verdict(r) is None) == 5
        assert all(r["function_word_rate"] is None for r in rows if _verdict(r) is None)


class TestLanguagesWithoutALexicon:
    def test_a_language_with_no_lexicon_abstains_rather_than_reassuring(self) -> None:
        """The dangerous case. Without a lexicon there is nothing to separate a
        terse list from a ruin, and `not_prose` reads as a scope question rather
        than a quality one, so a wholly corrupt corpus would file itself as
        material we chose not to structure."""
        damaged = "مىذة ىلمىدة ىنص ىحكم ىباب " * 60
        assert text_verdict(damaged, ARABIC) == "damaged"
        no_lexicon = dataclasses.replace(ARABIC, lexicon_words=frozenset(), lexicon_floor=None)
        assert text_verdict(damaged, no_lexicon) is None

    def test_a_language_with_no_list_abstains(self) -> None:
        # Ukrainian has no list, so its documents report no verdict. English
        # now does have one, which is the point of keying on language.
        assert words_for(load_config("ua").authoritative_language) is None
        assert words_for(load_config("gb").authoritative_language) is not None
        assert text_verdict("the quick brown fox " * 60, None) is None


class TestTheDeltaBetweenTwoScans:
    """`compare_finding_rates` needs a Postgres session, so the arithmetic that
    can be tested without one is tested here: the properties a summariser reads
    before it touches either side."""

    def test_a_check_that_stopped_running_is_not_a_zero_delta(self) -> None:
        from codify.storage.structural_findings import FindingRate, RateDelta

        before = FindingRate(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            failed=3,
            passed=7,
            not_run=0,
        )
        gone = RateDelta(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            before=before,
            after=None,
        )
        assert gone.in_both is False
        assert gone.stopped_running is True
        # None, not 0.0: the check vanished rather than improving to perfect.
        assert gone.rate_delta is None

    def test_an_unmeasurable_side_is_told_apart_from_a_vanished_one(self) -> None:
        from codify.storage.structural_findings import FindingRate, RateDelta

        unmeasurable = FindingRate(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            failed=0,
            passed=0,
            not_run=9,
        )
        measured = FindingRate(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            failed=2,
            passed=8,
            not_run=0,
        )
        both = RateDelta(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            before=unmeasurable,
            after=measured,
        )
        assert both.rate_delta is None
        assert both.stopped_running is False

    def test_a_newly_added_check_is_not_a_regression(self) -> None:
        """`stopped_running` used to be `not in_both`, which fired for a check
        present only in the later run. A check that has just started reads as one
        that has just vanished, and the comparison reports the wrong direction."""
        from codify.storage.structural_findings import FindingRate, RateDelta

        after = FindingRate(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            failed=1,
            passed=9,
            not_run=0,
        )
        added = RateDelta(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            before=None,
            after=after,
        )
        assert added.started_running is True
        assert added.stopped_running is False

    def test_an_unchanged_corpus_gives_a_zero_delta(self) -> None:
        from codify.storage.structural_findings import FindingRate, RateDelta

        rate = FindingRate(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            failed=2,
            passed=8,
            not_run=1,
        )
        same = RateDelta(
            jurisdiction_code="xz",
            era="assembly",
            check_name="anchor_coverage",
            before=rate,
            after=rate,
        )
        assert same.rate_delta == 0.0
