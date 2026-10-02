"""Corrections that must survive every later translation of a law. Regeneration follows
every pipeline change and the model never repeats itself, so a reviewer's correction
decays unless something holds it. A pin is a work URI, target language, eId and
substring; the delivery gate refuses a translation that has lost one. Pins assert a
phrase, not a provision, so rewording stays free; each ``why`` records its defect.
They live in ``pins/<jurisdiction>.jsonl`` beside this module. A pin is meaningful
only if its eId resolves, so resolve a new one against a real translation first::

    python -m codify.translate.regression_pins delivered.xml /akn/xa/act/2005/1 eng
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import structlog
from lxml import etree

from codify.akn._schema import parse_xml

logger = structlog.get_logger()

_PINS_PACKAGE = "codify.translate.pins"

# Prose inside a quoted amendment belongs to the act being amended, so it must
# not satisfy a pin on this document's own provision of the same eId.
_BORROWED = frozenset({"meta", "quotedStructure"})


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _own_text(el: etree._Element) -> str:
    """The element's text with borrowed subtrees left out. A provision can quote the act
    it amends, and ``itertext()`` would read that quotation as this document's own
    words, so a phrase missing from the operative text could be satisfied by the copy
    of the text being replaced.
    """
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if isinstance(child.tag, str) and _local(child.tag) in _BORROWED:
                # The quotation is skipped; the text that follows it is not.
                if child.tail:
                    parts.append(child.tail)
                continue
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join("".join(parts).split())


class PinFileError(RuntimeError):
    """A pins file that cannot be read; the gate has no assertions to apply."""


@dataclass(frozen=True)
class RegressionPin:
    """One phrase that a verified translation of ``eid`` is known to contain."""

    work_uri: str
    language: str
    eid: str
    must_contain: str
    why: str
    source: str  # pins file this came from, so a failure says where to edit

    @property
    def key(self) -> tuple[str, str]:
        return (self.work_uri, self.language)


@dataclass(frozen=True)
class PinViolation:
    """A pin whose phrase is absent from the delivered text."""

    pin: RegressionPin
    found: str | None  # None when the eId itself is missing

    @property
    def eid_missing(self) -> bool:
        """True when the provision is absent rather than reworded.

        Either the translation dropped it or the pin is stale, and those want
        different responses, so the failure message keeps them apart.
        """
        return self.found is None

    def describe(self) -> str:
        what = (
            f"reads {self.found[:60]!r}"
            if self.found is not None
            else f"eId absent, so either the provision was dropped or the pin "
            f"in {self.pin.source} is stale"
        )
        return f"{self.pin.eid} lost {self.pin.must_contain!r} ({what}; {self.pin.why})"


def _pin_from(raw: dict[str, Any], source: str) -> RegressionPin:
    return RegressionPin(
        work_uri=raw["work_uri"],
        language=raw["language"],
        eid=raw["eid"],
        must_contain=raw["must_contain"],
        why=raw["why"],
        source=source,
    )


@lru_cache(maxsize=1)
def load_pins() -> tuple[RegressionPin, ...]:
    """Every pin shipped with the package, across all jurisdictions.

    Raises rather than returning nothing: an empty load means the gate is
    disarmed, which must never pass for "no corrections to hold".
    """
    pins: list[RegressionPin] = []
    for entry in sorted(resources.files(_PINS_PACKAGE).iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".jsonl"):
            continue
        for number, line in enumerate(entry.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                pins.append(_pin_from(json.loads(line), entry.name))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise PinFileError(f"{entry.name} line {number}: {exc}") from exc
    if not pins:
        raise PinFileError(f"no regression pins found in {_PINS_PACKAGE}")
    logger.info("regression_pins_loaded", count=len(pins))
    return tuple(pins)


def pins_for(work_uri: str, language: str) -> tuple[RegressionPin, ...]:
    """Pins registered for one work in one target language."""
    return tuple(p for p in load_pins() if p.key == (work_uri, language))


def check_pins(akn_xml: str, work_uri: str, language: str) -> list[PinViolation] | None:
    """Pins violated by this translation, or ``None`` if the check could not run.

    ``None`` is returned for unparseable XML rather than an empty list, so a
    caller cannot mistake "could not look" for "nothing lost"."""
    try:
        root = parse_xml(akn_xml)
    except etree.XMLSyntaxError:
        # Parse before loading pins: an unparseable translation cannot be checked,
        # so the pins are never loaded and the check is recorded as not run.
        return None
    pins = pins_for(work_uri, language)
    if not pins:
        return []

    # One pass over the tree: a law can carry many pins and the Civil Code has
    # 1,300 articles.
    wanted = {p.eid for p in pins}
    text_by_eid: dict[str, str] = {}
    for el in root.iter():
        eid = el.get("eId") if isinstance(el.tag, str) else None
        if eid not in wanted or eid in text_by_eid:
            continue
        if any(_local(e.tag) in _BORROWED for e in (el, *el.iterancestors())):
            continue
        text_by_eid[eid] = _own_text(el)

    return [
        PinViolation(pin=pin, found=found)
        for pin in pins
        if (found := text_by_eid.get(pin.eid)) is None
        or pin.must_contain.lower() not in found.lower()
    ]


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m codify.translate.regression_pins FILE.xml WORK_URI LANGUAGE")
        return 2
    path, work_uri, language = argv
    pins = pins_for(work_uri, language)
    if not pins:
        print(f"no pins registered for {work_uri} in {language}")
        return 2
    violations = check_pins(open(path, encoding="utf-8").read(), work_uri, language)
    if violations is None:
        print(f"{path} will not parse")
        return 2
    violated = {v.pin.eid for v in violations}
    for pin in pins:
        print(f"{'VIOLATED' if pin.eid in violated else 'holds':10} {pin.eid} {pin.must_contain!r}")
    for violation in violations:
        print(f"  {violation.describe()}")
    return 1 if violations else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
