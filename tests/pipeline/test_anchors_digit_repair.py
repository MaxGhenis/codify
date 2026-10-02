"""Anchor-level integration tests for the digit-substitution OCR repair.

Complements `tests/parse/test_digit_confusion.py` (unit-level scoring) by
exercising the full `scan_anchors` path with realistic Arabic article
sequences carrying the observed OCR defects."""

from __future__ import annotations

import pytest
import structlog
from structlog.testing import capture_logs

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


# Synthetic sequence: five articles in ascending order where article 6
# has been misread as `1` due to the 1↔6 confusion (hooked stroke lost).
# The repair pass should walk the anchor list and rewrite that number.
_SYNTHETIC_DIGIT_SEQUENCE = """\
المادة 4
Body prose for the fourth provision.

المادة 5
Body prose for the fifth provision.

المادة 1
Body prose for the misread provision.

المادة 7
Body prose for the seventh provision.

المادة 8
Body prose for the eighth provision.
"""


def _regex():
    config = load_config("xz")
    assert config is not None
    return build_anchor_regex(config, "act")


def test_synthetic_six_read_as_one_is_repaired() -> None:
    """The independently constructed sequence 4,5,1,7,8
    should be rewritten to 4,5,6,7,8 by the digit-substitution pass."""
    with capture_logs() as captured:
        anchors = scan_anchors(_SYNTHETIC_DIGIT_SEQUENCE, _regex())
    structlog.reset_defaults()

    article_numbers = [a.number for a in anchors if a.kind == "article"]
    assert article_numbers == ["4", "5", "6", "7", "8"], article_numbers
    article_eids = [a.akn_eid for a in anchors if a.kind == "article"]
    assert "art_6" in article_eids
    repair_events = [r for r in captured if r.get("event") == "structure_digit_repair"]
    assert len(repair_events) == 1
    assert repair_events[0]["from_num"] == "1"
    assert repair_events[0]["to_num"] == "6"


def test_clean_sequence_leaves_anchors_untouched() -> None:
    """Monotonic 1..N sequence: no repair should fire, no log event
    should be emitted, and every original anchor number survives."""
    text = "\n\n".join(f"المادة {n}\nBody." for n in range(1, 6))
    with capture_logs() as captured:
        anchors = scan_anchors(text, _regex())
    structlog.reset_defaults()

    article_numbers = [a.number for a in anchors if a.kind == "article"]
    assert article_numbers == ["1", "2", "3", "4", "5"]
    assert not any(r.get("event") == "structure_digit_repair" for r in captured)


def test_single_article_short_circuits() -> None:
    """A one-article law has nothing to repair; the pass must not raise
    or emit spurious log events on the degenerate case."""
    with capture_logs() as captured:
        anchors = scan_anchors("المادة 1\nSolitary article.", _regex())
    structlog.reset_defaults()

    article_numbers = [a.number for a in anchors if a.kind == "article"]
    assert article_numbers == ["1"]
    assert not any(r.get("event") == "structure_digit_repair" for r in captured)
