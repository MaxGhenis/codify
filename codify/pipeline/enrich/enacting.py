"""Inject <formula name="enactingFormula"> into AKN from jurisdiction config.

Selects the right structured enacting formula for a document's date + doctype
and injects it into the AKN preamble. No LLM involvement, pure data lookup.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import cast

import structlog
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.jurisdictions import EnactingFormula, load_config
from codify.pipeline.enrich.akn_meta import FORMULA_PROVENANCE
from codify.pipeline.enrich.anchors import cached_regex
from codify.pipeline.enrich.arabic_normalise import (
    drop_header_only_lines,
    fold_arabic_for_match,
)

logger = structlog.get_logger()


def select_formula(
    country: str,
    doctype: str,
    doc_date: str | date | None,
) -> EnactingFormula | None:
    """Pick the enacting formula that applies to a document. Filters the jurisdiction's
    ``enacting_formulae`` by doctype (wildcard allowed) and date range, then prefers
    ``verified: true``. None when the jurisdiction has no formulae or nothing matches.
    """
    cfg = load_config(country)
    if not cfg.enacting_formulae:
        return None

    target = _coerce_date(doc_date)
    candidates: list[EnactingFormula] = []

    for f in cfg.enacting_formulae:
        if not _doctype_matches(f, doctype):
            continue
        bounded = f.from_date is not None or f.to_date is not None
        # An era-bounded formula must never apply outside its era, and a
        # document with no usable date cannot prove it is inside one.
        if bounded and (target is None or not _date_in_range(f, target)):
            continue
        candidates.append(f)

    if not candidates:
        # Date filter ruled everything out: fall back only to date-unbounded
        # formulae.
        candidates = [
            f
            for f in cfg.enacting_formulae
            if _doctype_matches(f, doctype) and f.from_date is None and f.to_date is None
        ]

    if not candidates:
        return None

    candidates.sort(key=lambda f: (not f.verified,))
    return candidates[0]


def inject_enacting_formula(akn_xml: str, formula: EnactingFormula) -> str:
    """Add ``<preamble><formula name="enactingFormula">…</formula></preamble>``
    between <preface>/<meta> and <body>. Idempotent."""
    value = formula.text or ""
    slots = re.findall(r"\[([^\]]*)\]", value)
    unresolved = any(
        any(char.isalpha() for char in slot)
        and not re.fullmatch(r"\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*(?:\([A-Za-z0-9]+\))*", slot)
        for slot in slots
    )
    if unresolved or re.search(r"\{[^{}]*\}|\.{3}|…", value):
        logger.info("enacting_formula_skipped", reason="unexpanded_template")
        return akn_xml
    root = etree.fromstring(akn_xml.encode("utf-8"))

    act = _find_act_element(root)
    if act is None:
        return akn_xml

    preamble = act.find("akn:preamble", NS)
    if preamble is None:
        preamble = etree.SubElement(act, f"{{{AKN_NS}}}preamble")
        _move_before_body(act, preamble)

    existing = preamble.find("akn:formula[@name='enactingFormula']", NS)
    if existing is not None:
        # Idempotent: leave existing formula alone. If the config changed
        # we'd need an explicit re-enrich; this keeps repeat runs stable.
        return akn_xml

    # refersTo="#codify" marks the formula as tool-injected config fallback, not
    # document text; the fabricated_formula validator check skips it on that
    # provenance. The schema refuses @source here.
    fml = etree.SubElement(
        preamble,
        f"{{{AKN_NS}}}formula",
        attrib={"name": "enactingFormula", "refersTo": FORMULA_PROVENANCE},
    )
    p = etree.SubElement(fml, f"{{{AKN_NS}}}p")
    p.text = formula.text

    return cast(str, etree.tostring(root, pretty_print=True, encoding="unicode"))


def emit_enacting_formula(
    akn_xml: str,
    country: str,
    doctype: str,
    doc_date: str | date | None,
) -> str:
    """Select and inject the enacting formula in one call. No-op when nothing matches, when
    the source preface or preamble already carries its own, or when the source names a
    different enacting authority than the formula asserts: a Council of Ministers
    regulation must never receive a legislative-council formula because its date falls in
    that era.
    """
    cfg = load_config(country)
    root = etree.fromstring(akn_xml.encode("utf-8"))
    if cfg.enacting_formula_markers:
        if _source_carries_formula(root, cfg.enacting_formula_markers):
            logger.info(
                "enacting_formula_skipped",
                country=country,
                doctype=doctype,
                reason="present_in_source",
            )
            return akn_xml

    formula = select_formula(country, doctype, doc_date)
    if formula is None:
        logger.info(
            "enacting_formula_skipped",
            country=country,
            doctype=doctype,
            reason="no_match",
        )
        return akn_xml

    if _authority_conflict(root, formula.text or ""):
        logger.info(
            "enacting_formula_skipped",
            country=country,
            doctype=doctype,
            reason="authority_mismatch",
        )
        return akn_xml

    out = inject_enacting_formula(akn_xml, formula)
    if out == akn_xml:
        return out
    logger.info(
        "enacting_formula_injected",
        country=country,
        era=formula.era,
        verified=getattr(formula, "verified", False),
    )
    return out


# Generic Arabic preamble openers: citation clauses (having reviewed /
# pursuant to / based upon) that begin recital chains across legal eras.
# Config enacting_formula_markers extend this per jurisdiction.
_PREAMBLE_OPENERS = (
    "بعد الاطلاع",
    "استنادا",
    "وبناء على",
    "بناء على",
    "باسم الشعب",
)
_FOLDED_PREAMBLE_OPENERS = tuple(fold_arabic_for_match(o) for o in _PREAMBLE_OPENERS)


def split_opening_material(text: str, country: str) -> tuple[str | None, str | None]:
    """Split pre-first-anchor text into (preface, preamble). The preamble starts at the first
    line carrying a citation opener or a config enacting-formula marker; everything before
    it is preface, and no opener means all preface.

    Lines that are wholly gazette furniture by the jurisdiction's
    ``furniture_line_patterns`` are dropped first, so they never become preface or
    preamble paragraphs: carrying them invites the translator to write legal-sounding text
    over a page footer, and one it declines is one the front-matter contract must recover.
    """
    stripped = text.strip()
    if not stripped:
        return None, None
    cfg = load_config(country)
    if cfg.furniture_line_patterns:
        cleaned, dropped = drop_header_only_lines(stripped, cfg.furniture_line_patterns)
        if dropped:
            logger.info("opening_material_furniture_dropped", country=country, lines=dropped)
        stripped = cleaned.strip()
        if not stripped:
            return None, None
    formula_markers = [fold_arabic_for_match(m) for m in cfg.enacting_formula_markers]
    markers = formula_markers + [fold_arabic_for_match(m) for m in _PREAMBLE_OPENERS]
    terminators = [
        fold_arabic_for_match(m) for m in (cfg.opening_material_terminators if cfg else [])
    ]
    lines = _truncate_after_enacting_formula(stripped.splitlines(), terminators, country)
    for i, line in enumerate(lines):
        folded = fold_arabic_for_match(line)
        if folded and any(m in folded for m in markers):
            preface = "\n".join(lines[:i]).strip() or None
            preamble = "\n".join(lines[i:]).strip() or None
            return preface, preamble
    return "\n".join(lines).strip() or None, None


def _truncate_after_enacting_formula(
    lines: list[str], terminators: list[str], country: str
) -> list[str]:
    """Drop everything after the phrase closing the opening material; what follows is the
    next scan page bleeding in.

    Terminators rather than `enacting_formula_markers`, because the latter is evidence a
    source carries its own formula, and a phrase that opens an instrument is evidence too:
    cutting there deletes the recitals after it. No terminator means no cut, and a tail
    carrying real content is kept and warned about rather than deleted.
    """
    if not terminators:
        return lines
    # First, not last: a bled-in page repeats the formula with its masthead.
    first = next(
        (
            i
            for i, line in enumerate(lines)
            if (folded := fold_arabic_for_match(line)) and any(m in folded for m in terminators)
        ),
        -1,
    )
    if first < 0 or first == len(lines) - 1:
        return lines
    tail = lines[first + 1 :]
    # Content after the formula means the cut point is wrong, not the tail.
    # The jurisdiction's own anchor regex, not Bluebell keywords: this runs on
    # source OCR, where an Arabic article opens `مادة (١)`, not `ARTICLE 1`.
    anchors = cached_regex(country, "act")
    kept = [
        line
        for line in tail
        if anchors.search(line)
        or any(op in fold_arabic_for_match(line) for op in _FOLDED_PREAMBLE_OPENERS)
    ]
    if kept:
        logger.warning(
            "opening_material_truncation_declined",
            reason="tail_carries_content",
            lines=[line.strip()[:60] for line in kept[:3]],
        )
        return lines
    logger.info(
        "opening_material_truncated_at_formula",
        dropped=len(tail),
        text=[line.strip()[:60] for line in tail[:6]],
    )
    return lines[: first + 1]


# --- helpers -----------------------------------------------------------------


# Phrases pinning the enacting authority: when the source carries one the formula's
# text does not name, the formula belongs to another body and must not be injected.
# The noun form is an instrument title, so it counts only in the preface, recitals
# routinely citing prior Council of Ministers decisions. The verb forms pin enactment
# and count anywhere in the opening matter.
_INSTRUMENT_TITLE_AUTHORITIES: tuple[tuple[str, str], ...] = (
    ("قرار مجلس الوزراء", "مجلس الوزراء"),
)
_ENACTING_VERB_AUTHORITIES: tuple[tuple[str, str], ...] = (
    ("قرر مجلس الوزراء", "مجلس الوزراء"),
    ("أصدر مجلس الوزراء", "مجلس الوزراء"),
    # Municipal by-laws (نظام) enacted by a city council and ratified by the
    # Minister of Local Governance. A PLC-era date must not pull the
    # legislative-council formula onto a municipal instrument.
    ("أصدر مجلس بلدية", "مجلس بلدية"),
    ("قرر مجلس بلدية", "مجلس بلدية"),
    ("أصدر المجلس البلدي", "المجلس البلدي"),
    ("قرر المجلس البلدي", "المجلس البلدي"),
)


def _authority_conflict(root: etree._Element, formula_text: str) -> bool:
    def region(tag: str) -> str:
        return fold_arabic_for_match(
            " ".join(" ".join(el.itertext()) for el in root.iter(f"{{{AKN_NS}}}{tag}"))
        )

    preface = region("preface")
    opening = preface + " " + region("preamble")
    fml = fold_arabic_for_match(formula_text)
    for phrases, src in (
        (_INSTRUMENT_TITLE_AUTHORITIES, preface),
        (_ENACTING_VERB_AUTHORITIES, opening),
    ):
        for instrument, authority in phrases:
            if (
                fold_arabic_for_match(instrument) in src
                and fold_arabic_for_match(authority) not in fml
            ):
                return True
    return False


def _source_carries_formula(root: etree._Element, markers: list[str]) -> bool:
    """True when any marker phrase occurs in the preface/preamble text.
    Markers are literal substrings, orthography-folded on both sides, so a
    config typo can never become a regex that kills the pass."""
    parts: list[str] = []
    for tag in ("preface", "preamble"):
        for el in root.iter(f"{{{AKN_NS}}}{tag}"):
            parts.append(" ".join(el.itertext()))
    text = fold_arabic_for_match(" ".join(parts))
    return any(fold_arabic_for_match(m) in text for m in markers)


def _coerce_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _doctype_matches(formula: EnactingFormula, doctype: str) -> bool:
    """Formula applies to this doctype if it's a wildcard or explicitly lists it."""
    dc = formula.document_class
    if dc is None:
        return True
    if isinstance(dc, str):
        # A scalar document_class names one doctype, not a wildcard: 68 of the
        # 71 configs that set it are scalar (ar's "ley" vs "dnu", at's "act" vs
        # "vo"), so a wildcard here would pick the first formula for every doctype.
        return dc == doctype
    return doctype in dc


def _date_in_range(formula: EnactingFormula, target: date) -> bool:
    start = _coerce_date(formula.from_date)
    end = _coerce_date(formula.to_date)
    if start is not None and target < start:
        return False
    if end is not None and target >= end:
        return False
    return True


def _find_act_element(root: etree._Element) -> etree._Element | None:
    for tag in ("act", "bill", "amendment", "doc"):
        el = root.find(f"akn:{tag}", NS)
        if el is not None:
            return el
    return None


def _move_before_body(act: etree._Element, preamble: etree._Element) -> None:
    """Ensure <preamble> sits before <body> per AKN 3.0 element order."""
    body = act.find("akn:body", NS)
    if body is None:
        return
    act.remove(preamble)
    body_idx = list(act).index(body)
    act.insert(body_idx, preamble)
