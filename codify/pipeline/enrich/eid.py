"""Resolve informal provision references to canonical AKN eId paths.

`inline_markup.py` detects cross-references in prose ("section 5(2)(a)") and needs
paths like ``sec_5__subsec_2__para_a`` to point each `<ref>` at its target. Driven
deterministically from the jurisdiction's ``hierarchy`` config, no model.

Supported so far: the common-law parenthesised form. The civil-law inline form falls
through to the primary unit only, which is a gap rather than a decision; a caller
needing more should inspect the remainder.
"""

from __future__ import annotations

import re

import structlog

from codify.akn.eid import eid_abbrev
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.lang import normalise_digits
from codify.pipeline.enrich.anchors import _split_term_with_parens

logger = structlog.get_logger()

# Aliases every jurisdiction with a matching local_term picks up: a `section` entry
# also answers to "s.", "§" and "s". Synonyms, not overrides, so the real local_term
# still wins where present.
_COMMON_ALIASES: dict[str, tuple[str, ...]] = {
    "section": ("s", "sec", "§"),
    "subsection": ("ss", "subsec"),
    "article": ("art", "a"),
    "paragraph": ("para", "p", "par"),
    "subparagraph": ("subpara", "sp"),
    "chapter": ("ch", "chap"),
    "part": ("pt",),
    "schedule": ("sch",),
    "rule": ("r",),
    "clause": ("cl",),
}

# Captures unit (the structural word, case-insensitive), num (which may carry a letter
# suffix like 5A), and zero or more parenthesised subdivisions. The unit class covers
# Latin with diacritics, Cyrillic and Arabic; num covers ASCII plus both Arabic-Indic
# digit ranges, which `resolve_informal_reference` folds to ASCII before lookup.
_REF_RE = re.compile(
    r"""
    ^\s*
    (?P<unit>[A-Za-zÀ-ÿ§Ѐ-ӿ؀-ۿ]+)\.?
    \s*
    (?P<num>[0-9٠-٩۰-۹]+[A-Za-z]?(?:[-/][0-9٠-٩۰-۹A-Za-z]+)?)
    (?P<subs>(?:\s*\(\s*[0-9٠-٩۰-۹A-Za-z]+\s*\))*)
    \s*
    $
    """,
    re.VERBOSE,
)

_SUB_RE = re.compile(r"\(\s*([0-9A-Za-z]+)\s*\)")


def build_term_map(
    cfg: JurisdictionConfig, doctype: str | None
) -> tuple[dict[str, str], list[str]]:
    """Build a {normalised_term: abbrev} map and the ordered list of
    subdivision-level abbrevs for the given document class.
    """
    dc = cfg.get_document_class(doctype or cfg.default_document_class)
    if dc is None or not dc.hierarchy:
        return {}, []

    term_map: dict[str, str] = {}
    subdivision_order: list[str] = []

    for entry in dc.hierarchy:
        if not entry.akn_element:
            continue
        abbrev = entry.eid_abbrev or eid_abbrev(entry.akn_element)
        # Every same-script form of local_term maps to the abbrev. Configs often write
        # Arabic local_terms with a parenthesised transliteration, which
        # `_split_term_with_parens` unwraps so the key matches body text.
        for form in _split_term_with_parens(entry.local_term):
            _add_term_aliases(term_map, form, abbrev)
        term_map.setdefault(entry.akn_element.lower(), abbrev)
        # Aliases from the common list, keyed on the akn_element
        for aliases in _COMMON_ALIASES.get(entry.akn_element.lower(), ()):
            term_map.setdefault(aliases.lower(), abbrev)
        # Same parenthesised treatment for the multilingual local_terms store.
        for raw in (entry.local_terms or {}).values():
            if not raw:
                continue
            for form in _split_term_with_parens(raw):
                _add_term_aliases(term_map, form, abbrev)

        if entry.level == "subdivision":
            subdivision_order.append(abbrev)

    return term_map, subdivision_order


def _add_term_aliases(term_map: dict[str, str], form: str, abbrev: str) -> None:
    """Add a term and its script-specific morphological variants to the term_map. Arabic
    terms gain the definite-article prefix ("مادة" also "المادة"), since body text usually
    cites a specific unit in the definite form. Both keys point at the same abbrev, so
    either spelling resolves.
    """
    key = form.strip().lower()
    if not key:
        return
    term_map.setdefault(key, abbrev)
    if _is_arabic(key) and not key.startswith("ال"):
        term_map.setdefault(f"ال{key}", abbrev)


def _is_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)


def resolve_informal_reference(
    ref: str,
    jurisdiction_code: str,
    doctype: str | None = None,
) -> str | None:
    """Map an informal reference to a canonical eId path. ``None`` where the reference
    will not parse or the primary unit is not in the hierarchy. Raises
    ``JurisdictionConfigError`` where the jurisdiction has no config: the hierarchy is
    what resolves the path, so without one there is nothing to answer with.

    Examples (xa.act)::

        resolve_informal_reference("section 5", "xa")        -> "sec_5"
        resolve_informal_reference("section 5(2)", "xa")     -> "sec_5__subsec_2"
        resolve_informal_reference("section 5(2)(a)", "xa")  -> "sec_5__subsec_2__para_a"
        resolve_informal_reference("s. 5(2)", "xa")          -> "sec_5__subsec_2"

    Examples (fr.loi)::

        resolve_informal_reference("Article 5", "fr")        -> "art_5"
        resolve_informal_reference("Art. 5(2)", "fr")        -> "art_5__para_2"
    """
    if not ref:
        return None

    cfg = load_config(jurisdiction_code)

    term_map, subdivision_order = build_term_map(cfg, doctype)
    if not term_map:
        logger.warning(
            "eid_resolve_empty_term_map",
            code=jurisdiction_code,
            doctype=doctype or cfg.default_document_class,
        )
        return None

    # Normalise Arabic-Indic + Persian digits to ASCII before the match, the
    # subdivision parser and abbrev lookup expect ASCII numerics throughout.
    normalised_ref = normalise_digits(ref.strip())

    # Compound "X of Y" ("paragraph 3 of Schedule 2"), container-first eId path.
    compound = re.split(r"\s+of\s+", normalised_ref, maxsplit=1)
    if len(compound) == 2:
        outer = resolve_informal_reference(compound[1], jurisdiction_code, doctype)
        inner = resolve_informal_reference(compound[0], jurisdiction_code, doctype)
        return f"{outer}__{inner}" if outer and inner else None

    match = _REF_RE.match(normalised_ref)
    if not match:
        logger.debug("eid_resolve_no_match", ref=ref)
        return None

    unit = match.group("unit").lower().rstrip(".")
    num = match.group("num").lower()
    subs_raw = match.group("subs") or ""

    abbrev = term_map.get(unit)
    if abbrev is None:
        logger.debug("eid_resolve_unknown_unit", unit=unit, code=jurisdiction_code)
        return None

    parts: list[str] = [f"{abbrev}_{num}"]

    for i, sub_match in enumerate(_SUB_RE.finditer(subs_raw)):
        if i >= len(subdivision_order):
            logger.debug(
                "eid_resolve_too_many_subdivisions",
                ref=ref,
                depth=i + 1,
                supported=len(subdivision_order),
            )
            break
        sub_abbrev = subdivision_order[i]
        sub_num = sub_match.group(1).lower()
        parts.append(f"{sub_abbrev}_{sub_num}")

    return "__".join(parts)
