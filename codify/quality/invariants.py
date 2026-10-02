"""Structural invariants and the readings the scanner would otherwise resolve
silently, in one implementation for the gate and the corpus scan.

No `codify.pipeline` imports, so `anchors.py` can emit spans without a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from codify.lang import normalise_digits

# Emitted where the scanner picks one reading over another.
AMBIGUITY_KINDS: tuple[str, ...] = (
    "duplicate_number",
    "toc_without_body",
    "numbering_gap",
    "orphan_text",
    "unmatched_marker",
    "untwinned_tail",
    "adoption_suppressed",
    "tail_excluded",
    "act_boundary_suspected",
)

# `toc_without_body` is absent on purpose: a cover-listed unit still yields an
# anchor from its own cover line, so the check cannot yet tell a listing from a
# body. It blocks once it can.
# `adoption_suppressed` is informational by nature: it records a check declining to
# run, so a landed instrument says why rather than looking unexamined.
# `untwinned_tail` is absent for a different reason: its own gate in the structurer
# decides what happens, and under the landing policy that is a blocking halt rather
# than a refusal. Counting it here as well would price one finding twice.
BLOCKING_KINDS: frozenset[str] = frozenset({"duplicate_number", "act_boundary_suspected"})


# Repair edits the AKN, so a gate whose loss is not in the AKN must say so, or
# the reader is sent to a loop that cannot help.
_REPAIR_REMEDY = "Landed for repair rather than refused."

_REMEDIES = {
    "coverage_below_floor": _REPAIR_REMEDY,
    "duplicate_anchor": _REPAIR_REMEDY,
    "source_truncated": (
        "Landed rather than refused; recovery is a complete copy of the source, "
        "not a re-run of this one."
    ),
    "markers_masked": (
        "Landed rather than refused; the hidden provisions were never anchored, "
        "so the route is the source text and its unclosed quote, not the AKN."
    ),
    "act_boundary_suspected": (
        "Landed rather than refused; the source reads as two acts, so the route is "
        "splitting the source, not repair."
    ),
    "body_fill_failed": (
        "Landed rather than refused; no model call wrote a body, so the route is "
        "a re-run once the model answers, not repair."
    ),
}


def halt_finding(halt: dict[str, Any]) -> dict[str, Any]:
    """One `structure_halted` finding, for every lane that reports halts.

    Both the durable workflow and the in-process bundle emit this. Built in two
    places it drifted, and a bundle that omits the remedy sends the reader to the
    repair loop for a document repair cannot help.
    """
    gate = str(halt.get("gate") or "")
    remedy = _REMEDIES.get(gate, _REPAIR_REMEDY)
    return {
        "check": "structure_halted",
        "severity": "error",
        "gate": gate,
        "kind": halt.get("kind"),
        "spans": halt.get("spans"),
        "offset": halt.get("first_offset"),
        "eid": halt.get("eid"),
        "ratio": halt.get("ratio"),
        "message": f"{gate} on {halt.get('kind')}: {halt.get('detail')}. {remedy}",
    }


@dataclass(frozen=True)
class AmbiguitySpan:
    """One place the scanner chose between readings. Offsets index the text it
    was given, which is RTL-normalised and so does not align with page spans."""

    kind: str
    start: int
    end: int
    eid: str = ""
    emitted_by: str = ""
    resolved: bool = False  # the scanner applied a rule it can defend
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in AMBIGUITY_KINDS:
            raise ValueError(f"unknown ambiguity kind {self.kind!r}")

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING_KINDS and not self.resolved


# `art_5`, `art_5bis`, `art_5bis2`, `art_1__point_3`, `art_1__point_1_2`.
_ORDER_KEY_RE = re.compile(r"^(?:[a-z]+_)?(\d+)(?:bis(\d*))?(?:_(\d+))?$")


def order_key(eid: str) -> tuple[int, int, int] | None:
    """Sibling position, or None if not comparable. `art_5bis` sorts as (5, 1, 0).
    The third component is the eId dedup suffix, so non-zero is itself a break."""
    match = _ORDER_KEY_RE.match(normalise_digits(eid.rsplit("__", 1)[-1]))
    if match is None:
        return None
    base, bis, dup = match.groups()
    return (
        int(base),
        (int(bis) if bis else 1) if bis is not None else 0,
        int(dup) if dup else 0,
    )


def classify_gap(count: int, max_run: int, lo: int, hi: int) -> str:
    """A gap reads as a structuring 'defect', a legitimate 'repeal', or
    'ambiguous'. Consolidated texts drop repealed articles, so holes alone are
    not breakage."""
    if not count:
        return "none"
    if max_run >= 3 or count / (hi - lo + 1) > 0.30:
        return "defect"
    if max_run == 1 and count <= 3:
        return "repeal"
    return "ambiguous"


def missing_between(numbers: list[int], *, cap: int = 500) -> tuple[list[int], int, int, int, int]:
    """Holes inside the observed range: capped examples, total, longest run, lo, hi.

    Never materialises the range. OCR misreads produce numbers like 6453 beside
    1, and expanding that eagerly costs the scan more than the finding is worth.
    """
    ordered = sorted(set(numbers))
    if len(ordered) < 2:
        return [], 0, 0, 0, 0
    lo, hi = ordered[0], ordered[-1]
    examples: list[int] = []
    total = max_run = 0
    for a, b in zip(ordered, ordered[1:], strict=False):
        run = b - a - 1
        if run <= 0:
            continue
        total += run
        max_run = max(max_run, run)
        if len(examples) < cap:
            examples.extend(range(a + 1, min(b, a + 1 + cap - len(examples))))
    return examples, total, max_run, lo, hi


__all__ = [
    "AMBIGUITY_KINDS",
    "BLOCKING_KINDS",
    "AmbiguitySpan",
    "classify_gap",
    "missing_between",
    "order_key",
]
