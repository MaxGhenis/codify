"""The primitives shared by the gate and the corpus scan, so a defect here is a
defect in two places. Each test names the wrong number it prevents."""

from __future__ import annotations

import pytest

from codify.quality.invariants import (
    AmbiguitySpan,
    classify_gap,
    missing_between,
    order_key,
)


class TestOrderKey:
    def test_a_bis_article_sorts_between_its_neighbours(self) -> None:
        """Insertions are normal; reading one as out of order breaks them all."""
        assert order_key("art_5") < order_key("art_5bis") < order_key("art_6")  # type: ignore[operator]

    def test_the_dedup_suffix_is_visible(self) -> None:
        """The suffix is the defect. Keying on the whole eId hid it."""
        assert order_key("art_1_2") == (1, 0, 2)
        assert order_key("art_1") == (1, 0, 0)

    def test_arabic_indic_digits_fold(self) -> None:
        assert order_key("art_١٢") == order_key("art_12") == (12, 0, 0)

    def test_a_nested_eid_keys_on_its_own_segment(self) -> None:
        assert order_key("art_1__point_3") == (3, 0, 0)

    def test_a_word_numbered_eid_is_not_comparable(self) -> None:
        """The oldest acts spell ordinals; guessing invents a verdict."""
        assert order_key("art_alula") is None


class TestClassifyGap:
    def test_a_run_of_three_reads_as_a_defect(self) -> None:
        assert classify_gap(3, 3, 1, 20) == "defect"

    def test_isolated_singletons_read_as_repeals(self) -> None:
        """Consolidated texts drop repealed articles; that is not breakage."""
        assert classify_gap(1, 1, 1, 40) == "repeal"

    def test_a_wide_gap_is_a_defect_even_without_a_run(self) -> None:
        assert classify_gap(4, 1, 1, 10) == "defect"

    def test_no_holes_is_not_a_gap(self) -> None:
        assert classify_gap(0, 0, 1, 10) == "none"


class TestMissingBetween:
    def test_holes_are_found_within_the_observed_range(self) -> None:
        assert missing_between([1, 2, 5]) == ([3, 4], 2, 2, 1, 5)

    def test_one_number_is_not_a_sequence(self) -> None:
        """One number is not evidence of a dense sequence."""
        assert missing_between([7]) == ([], 0, 0, 0, 0)

    def test_duplicates_do_not_create_holes(self) -> None:
        assert missing_between([1, 1, 2]) == ([], 0, 0, 1, 2)

    def test_a_wild_ocr_number_does_not_expand_the_range(self) -> None:
        """Digit repair produced 6453 beside 1 on a real document. Expanding
        that eagerly costs more than the finding is worth."""
        examples, total, max_run, lo, hi = missing_between([1, 10**9], cap=5)
        assert len(examples) == 5
        assert total == 10**9 - 2
        assert max_run == total
        assert (lo, hi) == (1, 10**9)


class TestAmbiguitySpan:
    def test_an_unknown_kind_is_refused(self) -> None:
        """A typo in a kind would silently become non-blocking."""
        with pytest.raises(ValueError, match="unknown ambiguity kind"):
            AmbiguitySpan(kind="probably_fine", start=0, end=1)

    def test_only_an_unresolved_collision_blocks(self) -> None:
        assert AmbiguitySpan(kind="duplicate_number", start=0, end=1).blocking is True
        for kind in ("toc_without_body", "numbering_gap", "orphan_text", "unmatched_marker"):
            assert AmbiguitySpan(kind=kind, start=0, end=1).blocking is False

    def test_a_cover_listing_is_declared_rather_than_refused(self) -> None:
        """The check cannot yet tell a cover line from a body anchor, and
        refusing on it fails correct documents."""
        assert AmbiguitySpan(kind="toc_without_body", start=0, end=1).blocking is False

    def test_a_resolved_duplicate_does_not_block(self) -> None:
        """Blocking a dropped TOC twin refused 3 of 36 correct documents."""
        span = AmbiguitySpan(kind="duplicate_number", start=0, end=1, resolved=True)
        assert span.blocking is False


def test_a_subdivision_is_counted_as_a_grouping_container() -> None:
    """A document whose only grouping is a subdivision must score full coverage.

    The probe counts the level from the source, so omitting it on the AKN side
    reads a correctly structured document as having flattened its headings.
    """
    from lxml import etree

    from codify.quality.structural_scan import _akn_container_count

    akn = etree.fromstring(
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body>"
        '<subdivision eId="subdvs_1"><num>1</num><heading>Group One</heading>'
        '<article eId="subdvs_1__art_1"><num>1</num>'
        "<content><p>Text.</p></content></article>"
        "</subdivision></body></act></akomaNtoso>".encode()
    )
    assert _akn_container_count(akn) == 1


def test_every_gate_states_the_route_its_reader_should_take() -> None:
    """A remedy that names the wrong loop is worse than none.

    Repair edits the AKN, so a gate whose loss never reached the AKN has to say
    so. The default said "landed for repair" for every gate but one, and a gate
    added later took it silently, sending the reader to a loop that cannot help.
    """
    import re
    from pathlib import Path

    from codify.quality.invariants import _REMEDIES

    package = Path(__file__).resolve().parents[2]
    declared = {
        m.group(1)
        for path in package.rglob("*.py")
        if "/tests/" not in path.as_posix()
        for m in re.finditer(r'gate="([a-z_]+)"', path.read_text(encoding="utf-8"))
    }
    assert declared, "no gates found, so this test is measuring nothing"
    assert declared <= set(_REMEDIES), sorted(declared - set(_REMEDIES))


def test_a_hidden_provision_sends_the_reader_to_the_source_not_the_repair_loop() -> None:
    """The provisions a quote hid were never anchored, so they are not in the
    AKN and the loop that edits it cannot put them there."""
    from codify.quality.invariants import halt_finding

    message = halt_finding(
        {"gate": "markers_masked", "kind": "article", "detail": "2 marker(s) hidden"}
    )["message"]
    assert "source text" in message
    assert "for repair" not in message
