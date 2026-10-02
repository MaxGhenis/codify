"""The container-coverage probe: the source presented grouping headings
(Bab/Fasl) that structuring flattened into the article run. A config-independent
guard against the silent level-drop a config gap causes: the OCR transcribed
every الفصل heading faithfully, yet the run produced zero chapters and graded clean.

`xz/decree` is the standing config-gap fixture: `فصل` is a grouping keyword the
jurisdiction knows through its other doctypes, but `decree` declares no
container level, so the anchor scan emits no chapter for a `الفصل الأول` heading.
"""

from __future__ import annotations

from lxml import etree

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity
from codify.pipeline.enrich.container_coverage import (
    container_coverage_below_floor,
    container_coverage_probe,
)
from codify.pipeline.enrich.structure import normalise_rtl_extract
from codify.pipeline.enrich.validator import _check_container_coverage as _validator_finding
from codify.quality.structural_scan import _check_container_coverage as _scan_finding

_FUSUL = (
    "الفصل الأول\nالمادة 1\nنص اول.\n\n"
    "الفصل الثاني\nالمادة 2\nنص ثان.\n\n"
    "الفصل الثالث\nالمادة 3\nنص ثالث.\n"
)
_FLAT = "المادة 1\nنص.\n\nالمادة 2\nنص.\n"
# A prose sentence opening with the keyword but no ordinal ("the previous Fasl").
_PROSE = "الفصل السابق ينطبق على هذه الحالة.\nالمادة 1\nنص.\n"
# An amendment quoting another law's structure: the heading is line-initial and
# inside a «…» span, and `فصل` is undeclared for xz/decree so it carries no
# `inside_quoted_text` span. Only a source quote-mask read excludes it.
_QUOTED = (
    "المادة 1\nيُستبدل بالمادة الآتية ما يلي:\n«الأحكام العامة\nالفصل الأول\nالمادة 5\nنص معدل.»\n"
)

_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _akn(chapters: int) -> etree._Element:
    """A minimal AKN body carrying `chapters` grouping containers."""
    if chapters:
        inner = "".join(
            f'<chapter eId="chp_{i}"><article eId="art_{i}"/></chapter>'
            for i in range(1, chapters + 1)
        )
    else:
        inner = '<article eId="art_1"/><article eId="art_2"/>'
    return etree.fromstring(
        f'<akomaNtoso xmlns="{_NS}"><act><body>{inner}</body></act></akomaNtoso>'.encode()
    )


def _probe(text: str, code: str, doctype: str) -> dict:
    t = normalise_rtl_extract(text)
    scan = scan_anchors_with_ambiguity(
        t, cached_regex(code, doctype), country=code, doctype=doctype
    )
    return container_coverage_probe(t, scan, load_config(code), code, doctype)


def test_config_gap_flattening_fires() -> None:
    """Three Fusul, no container level declared: all present, none found."""
    probe = _probe(_FUSUL, "xz", "decree")
    assert probe["present"] == 3
    assert probe["found"] == 0
    assert container_coverage_below_floor(probe) is True


def test_recovered_structure_does_not_fire() -> None:
    """A cabinet decision declaring its Fusul detects them as chapters, so the
    probe finds every heading and does not warn. The fix and the probe agree."""
    probe = _probe(_FUSUL, "xz", "cabinet_decision")
    assert probe["present"] == probe["found"] == 3
    assert container_coverage_below_floor(probe) is False


def test_flat_document_is_silent() -> None:
    """No grouping headings in the source: nothing to lose, no finding."""
    assert _probe(_FLAT, "xz", "decree")["present"] == 0


def test_prose_keyword_without_ordinal_is_not_a_heading() -> None:
    """`الفصل السابق` opens a line but names no ordinal, so it is prose, not a
    heading, and must not reach the denominator."""
    assert _probe(_PROSE, "xz", "decree")["present"] == 0


def test_quoted_amendment_heading_is_excluded() -> None:
    """A heading a document only quotes is not its own structure."""
    assert _probe(_QUOTED, "xz", "decree")["present"] == 0


def test_floor_fires_only_when_most_headings_are_lost() -> None:
    """A single missed container in a well-structured document is not a finding;
    losing most of them is."""
    assert container_coverage_below_floor({"present": 7, "found": 6}) is False
    assert container_coverage_below_floor({"present": 7, "found": 0}) is True
    assert container_coverage_below_floor({"present": 0, "found": 0}) is False


def test_validator_emits_warning_never_error() -> None:
    findings = _validator_finding({"present": 3, "found": 0})
    assert len(findings) == 1
    assert findings[0]["check"] == "container_coverage"
    # Warning, not error: the detector is config-independent and so has residual
    # false positives; a flattened law is still readable. Blocking would exclude
    # good laws.
    assert findings[0]["severity"] == "warning"
    assert findings[0]["count"] == 3


def test_validator_silent_when_structure_recovered() -> None:
    assert _validator_finding({"present": 3, "found": 3}) == []
    assert _validator_finding({"present": 0, "found": 0}) == []


def test_corpus_scan_check_compares_against_real_akn_containers() -> None:
    """The per-version scan measures source headings against the containers the
    stored AKN actually kept, so it catches a dedicated parser that flattened
    them even though a re-scan of the source would still find every heading."""
    # Structured-parser flatten: three source Fusul, an AKN with zero chapters.
    gap = _scan_finding(_FUSUL, load_config("xz"), "decree", _akn(0))
    assert gap.check == "container_coverage"
    assert gap.failed is True
    assert gap.detail["present"] == 3 and gap.detail["found"] == 0

    # The AKN kept all three chapters: no flatten.
    recovered = _scan_finding(_FUSUL, load_config("xz"), "decree", _akn(3))
    assert recovered.failed is False
    assert recovered.detail["found"] == 3

    # No grouping headings in the source: not a miss whatever the AKN holds.
    flat = _scan_finding(_FLAT, load_config("xz"), "decree", _akn(0))
    assert flat.failed is None
