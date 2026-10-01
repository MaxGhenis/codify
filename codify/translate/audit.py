"""Deterministic consistency audit for translated AKN (no LLM).

Checks what can drift during body translation: terminology, volume, numeric
and sentinel fidelity, script purity, OCR-header bleed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, NamedTuple

import structlog
from lxml import etree

from codify.akn import AKN_NS
from codify.lang import to_iso639_3
from codify.pipeline.enrich.arabic_normalise import fold_arabic_for_match
from codify.pipeline.enrich.validator import check_missing_container_titles
from codify.translate.translate_bodies import _UNTRANSLATED_MARKER

_logger = structlog.get_logger()


def _content_chars(text: str) -> int:
    """Count non-whitespace characters as a language-neutral volume proxy."""
    return sum(1 for c in text if not c.isspace())


# Alien-script classes used to validate structural ``<num>`` values. Body
# prose may retain source terms, but structural values must use target script.
_ARABIC_LETTER_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_HEBREW_LETTER_RE = re.compile(r"[֐-׿יִ-ﭏ]")

_ALIEN_SCRIPT_FOR_TARGET: dict[str, list[re.Pattern[str]]] = {
    # English target: Arabic or Hebrew letters in <num> are alien.
    "eng": [_ARABIC_LETTER_RE, _HEBREW_LETTER_RE],
    "en": [_ARABIC_LETTER_RE, _HEBREW_LETTER_RE],
    # Hebrew target: Arabic letters in <num> are alien.
    "heb": [_ARABIC_LETTER_RE],
    "he": [_ARABIC_LETTER_RE],
}


def alien_patterns_for(target_lang: str | None) -> list[re.Pattern[str]]:
    """Alien-script patterns for a target language given as a code or a name.

    Runs carry the name ("English"), this map is keyed by code, so the name is
    folded before giving up. No patterns means "not checked", never "clean"."""
    key = (target_lang or "").strip().lower()
    if key in _ALIEN_SCRIPT_FOR_TARGET:
        return _ALIEN_SCRIPT_FOR_TARGET[key]
    try:
        return _ALIEN_SCRIPT_FOR_TARGET.get(to_iso639_3(target_lang or ""), [])
    except ValueError:
        return []


def _has_alien_letters(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """True when the text carries an alien-script *letter*.

    The alien classes span whole Unicode blocks, so they include Arabic-Indic
    digits. A digit is a number, not retained source.
    """
    return any(m.group().isalpha() for pat in patterns for m in pat.finditer(text))


# Deontic-verb realisations by target language; a mix indicates register drift.
# (label, pattern) so counts key on a readable name. English omits "will", which
# legal English uses as future indicative, so counting it false-fires.
_DEONTIC_TARGETS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "eng": [
        ("shall", re.compile(r"\bshall\b", re.IGNORECASE)),
        ("must", re.compile(r"\bmust\b", re.IGNORECASE)),
        ("is required to", re.compile(r"\bis\s+required\s+to\b", re.IGNORECASE)),
    ],
    "ara": [
        ("يجب", re.compile(r"يجب", re.UNICODE)),
        ("يتعين", re.compile(r"يتعين", re.UNICODE)),
        ("يلزم", re.compile(r"يلزم", re.UNICODE)),
    ],
    "heb": [
        ("חייב", re.compile(r"חייב", re.UNICODE)),
        ("אסור", re.compile(r"אסור", re.UNICODE)),
    ],
}
# Aliases the caller may pass in.
_DEONTIC_TARGETS["en"] = _DEONTIC_TARGETS["eng"]
_DEONTIC_TARGETS["he"] = _DEONTIC_TARGETS["heb"]

# Fraction of deontic hits a minor variant must exceed to count as drift.
_DEONTIC_DRIFT_THRESHOLD = 0.10
# Below this total, sample is too small for the ratio to be meaningful.
_DEONTIC_MIN_TOTAL = 20


def _register_drift(translated: str, target_language: str) -> tuple[dict[str, int], bool, bool]:
    """Count deontic-verb realisations in the translated body. Returns
    ``(counts_by_verb, drifted, applicable)`` where ``applicable`` is
    False when the target language has no registered deontic patterns
    (so the audit can distinguish "clean pass" from "check skipped")."""
    entries = _DEONTIC_TARGETS.get(target_language.lower(), [])
    if not entries:
        try:
            entries = _DEONTIC_TARGETS.get(to_iso639_3(target_language), [])
        except ValueError:
            entries = []
    if not entries:
        return {}, False, False
    counts: dict[str, int] = {}
    for label, pat in entries:
        matches = pat.findall(translated)
        if matches:
            counts[label] = len(matches)
    total = sum(counts.values())
    if total < _DEONTIC_MIN_TOTAL or len(counts) < 2:
        return counts, False, True
    dominant = max(counts.values())
    minor = total - dominant
    return counts, (minor / total) > _DEONTIC_DRIFT_THRESHOLD, True


# Body paragraphs tolerate an inline source term; headings use a stricter
# threshold because they are short labels rather than prose.
_BODY_ALIEN_DOMINANCE = 0.5
_HEADING_ALIEN_DOMINANCE = 0.2


def text_is_alien_dominated(
    text: str, patterns: list[re.Pattern[str]], threshold: float = _BODY_ALIEN_DOMINANCE
) -> bool:
    """True when more than `threshold` of the non-space characters are
    source-script (alien to the target). Shared with the pre-delivery repair
    selection so it marks exactly the lines the delivery gate would reject."""
    content = [c for c in text if not c.isspace()]
    if not content:
        return False
    alien = sum(1 for c in content if any(pat.match(c) for pat in patterns))
    return alien / len(content) > threshold


# Four or more separated letters are treated as a likely spacing artefact.
# Restricting the pattern to letters avoids matching legitimate spaced digits.
_LETTER_SPACED = re.compile(r"\b(?:[^\W\d_] ){3,}[^\W\d_]\b", re.UNICODE)


# Longest initialism worth exempting; longer runs are more likely corruption.
_MAX_INITIALISM_LETTERS = 6


def _is_spelled_initialism(run: str) -> bool:
    """Whether a run is a short all-caps acronym written with spaces."""
    letters = run.replace(" ", "")
    return letters.isupper() and letters.isalpha() and len(letters) <= _MAX_INITIALISM_LETTERS


def count_letter_spaced_runs(akn_xml: str) -> tuple[int, dict[str, list[str]]] | None:
    """Count letter-spaced words per owning eId, or ``None`` if unparseable."""
    try:
        root = etree.fromstring(akn_xml.encode())
    except etree.XMLSyntaxError:
        return None
    hits = 0
    by_eid: dict[str, list[str]] = {}
    body_tags = (f"{{{AKN_NS}}}body", f"{{{AKN_NS}}}mainBody")
    bodies = [el for tag in body_tags for el in root.iter(tag)]
    if not bodies:
        # Parsed, but nothing was swept: a fragment or a missing namespace is
        # the same "did not run" state as unparseable, never a clean zero.
        return None
    for body in bodies:
        for el in body.iter(f"{{{AKN_NS}}}p"):
            text = " ".join("".join(el.itertext()).split())
            found = [r for r in _LETTER_SPACED.findall(text) if not _is_spelled_initialism(r)]
            if not found:
                continue
            hits += len(found)
            owner = _owning_eid(el.get("eId") or _owning_eid_of_ancestor(el))
            if owner:
                by_eid.setdefault(owner, []).extend(found)
    return hits, by_eid


# Citation nouns the rewrite can inject. A plural already carries the noun for
# the whole list, so a member repeating it cites one provision and names two
# ("Articles (Article 20)"); two nouns in one parenthetical is the same fault.
_CITATION_NOUNS_EN = r"Article|Paragraph|Section|Chapter|Clause"
_NOUN_UNDER_PLURAL = re.compile(
    rf"\b(?:{_CITATION_NOUNS_EN})s\b[^.]{{0,80}}?\(\s*(?:{_CITATION_NOUNS_EN})\b",
    re.UNICODE,
)
_TWO_NOUNS_IN_ONE_PAREN = re.compile(
    rf"\(\s*(?:{_CITATION_NOUNS_EN})\b[^)]*?\b(?:{_CITATION_NOUNS_EN})\b[^)]*\)",
    re.UNICODE,
)


def count_doubled_citation_nouns(
    akn_xml: str, target_language: str | None
) -> tuple[int, dict[str, list[str]]] | None:
    """Count citations naming their noun twice, per owning eId. ``None`` when the
    check could not run: unparseable XML, no body, or a target language with no
    patterns, which is unchecked rather than clean. Only English is written.
    """
    try:
        iso = to_iso639_3(target_language or "")
    except ValueError:
        iso = ""
    if iso not in ("eng",):
        return None
    try:
        root = etree.fromstring(akn_xml.encode())
    except etree.XMLSyntaxError:
        return None
    bodies = [el for tag in (f"{{{AKN_NS}}}body", f"{{{AKN_NS}}}mainBody") for el in root.iter(tag)]
    if not bodies:
        return None
    hits = 0
    by_eid: dict[str, list[str]] = {}
    for body in bodies:
        for el in body.iter(f"{{{AKN_NS}}}p"):
            text = " ".join("".join(el.itertext()).split())
            found = _NOUN_UNDER_PLURAL.findall(text) + _TWO_NOUNS_IN_ONE_PAREN.findall(text)
            if not found:
                continue
            hits += len(found)
            owner = _owning_eid(el.get("eId") or _owning_eid_of_ancestor(el))
            if owner:
                by_eid.setdefault(owner, []).extend(found)
    return hits, by_eid


def _count_alien_hits_under_body(
    akn_xml: str, target_lang: str, tag_localname: str, threshold: float
) -> tuple[int, dict[str, int]] | None:
    """Count `tag_localname` elements under `<body>` or `<mainBody>` whose text is
    dominated by alien script, as `(hits, hits_by_owning_eid)`. `None` when the check
    could not run, never a clean zero. Threshold is passed in: `<p>` carries inline
    source terms at 50%, a heading does not.
    """
    patterns = alien_patterns_for(target_lang)
    if not patterns:
        # Could not run: same state as unparseable, never a clean zero.
        return None
    try:
        root = etree.fromstring(akn_xml.encode())
    except etree.XMLSyntaxError:
        return None
    hits = 0
    hits_by_eid: dict[str, int] = {}
    # `mainBody` is the attachment doc's body: an annex that ships in source
    # script is the same delivery defect as an untranslated article.
    body_tags = (f"{{{AKN_NS}}}body", f"{{{AKN_NS}}}mainBody")
    target_tag = f"{{{AKN_NS}}}{tag_localname}"
    for body in (el for tag in body_tags for el in root.iter(tag)):
        for el in body.iter(target_tag):
            text = "".join(el.itertext())
            if text_is_alien_dominated(text, patterns, threshold):
                hits += 1
                owner = _owning_eid(el.get("eId") or _owning_eid_of_ancestor(el))
                if owner:
                    hits_by_eid[owner] = hits_by_eid.get(owner, 0) + 1
    return (hits, hits_by_eid)


def _owning_eid_of_ancestor(el: etree._Element) -> str:
    """A `<heading>` carries no eId; its owning article/section is the
    nearest ancestor with an eId. Walks up to the body root; returns ``""``
    if none found."""
    p = el.getparent()
    while p is not None:
        eid = p.get("eId") if isinstance(p.tag, str) else None
        if eid:
            return str(eid)
        p = p.getparent()
    return ""


_P_SUFFIX_RE = re.compile(r"__p_\d+$")


def _owning_eid(p_eid: str) -> str:
    """A `<p>` eid like `chp_iv__art_1__intro__p_1` is owned by `chp_iv__art_1`:
    slice at `__intro__` / `__content__` / `__wrapUp__`. Direct-child eids like
    `art_1__p_1` have no such scope and need the tail `__p_N` trimmed instead.
    """
    for sep in ("__intro__", "__content__", "__wrapUp__"):
        idx = p_eid.find(sep)
        if idx > 0:
            return p_eid[:idx]
    return _P_SUFFIX_RE.sub("", p_eid)


def _target_text_by_owning_eid(akn_xml: str) -> dict[str, str]:
    """Concatenate every `<p>` under `<body>` into per-owning-eid buckets.
    Skips `<preface>` and `<meta>` (glossary and metadata are not readable
    prose). Returns eid -> joined text with single spaces between `<p>`s."""
    buckets: dict[str, list[str]] = {}
    try:
        root = etree.fromstring(akn_xml.encode())
    except etree.XMLSyntaxError:
        return {}
    body_tag = f"{{{AKN_NS}}}body"
    for body in root.iter(body_tag):
        for p in body.iter(f"{{{AKN_NS}}}p"):
            owner = _owning_eid(p.get("eId") or "")
            if not owner:
                continue
            buckets.setdefault(owner, []).append("".join(p.itertext()))
    return {eid: " ".join(parts) for eid, parts in buckets.items()}


def _money_multiset_deltas(
    source_surfaces_by_eid: dict[str, list[str]],
    target_text_by_eid: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Per-eid multiset compare of money surfaces: `missing` where source count exceeds
    target, `duplicated` where target exceeds source. Multisets because counts agree
    where the amounts differ: source `1,000` and `3,000` against target `1,000` twice
    is two tokens either way, with `3,000` silently dropped.
    """
    missing: dict[str, list[str]] = {}
    duplicated: dict[str, list[str]] = {}
    for eid, surfaces in source_surfaces_by_eid.items():
        target_text = target_text_by_eid.get(eid, "")
        expected: dict[str, int] = {}
        for surface in surfaces:
            expected[surface] = expected.get(surface, 0) + 1
        for surface, src_count in expected.items():
            pattern = re.compile(rf"(?<!\d){re.escape(surface)}(?!\d)")
            tgt_count = len(pattern.findall(target_text))
            if tgt_count < src_count:
                missing.setdefault(eid, []).extend([surface] * (src_count - tgt_count))
            elif tgt_count > src_count:
                duplicated.setdefault(eid, []).extend([surface] * (tgt_count - src_count))
    return missing, duplicated


def _count_alien_num_hits(akn_xml: str, target_lang: str) -> int | None:
    """Count <num> elements carrying a script alien to the target language; zero
    for source-language targets. ``None`` on unparseable XML, which the caller
    surfaces as ``audit_alien_num_unparseable`` rather than a clean zero. The
    narrow ``XMLSyntaxError`` catch leaves broader failures to propagate.
    """
    patterns = alien_patterns_for(target_lang)
    if not patterns:
        return None
    try:
        root = etree.fromstring(akn_xml.encode())
    except etree.XMLSyntaxError:
        return None
    hits = 0
    for num in root.iter(f"{{{AKN_NS}}}num"):
        text = "".join(num.itertext())
        if not text:
            continue
        if any(pat.search(text) for pat in patterns):
            hits += 1
    return hits


def _find_header_bleed(
    translated: str, patterns: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Sweep the translated body for leftover OCR-header patterns, returning
    ``(matches_by_pattern, invalid_patterns)``. Uncompilable patterns are named
    separately so a profile author sees the config bug, not a clean audit. These
    are the stripper's own regexes: a hit means it missed or the LLM re-added it.
    """
    out: dict[str, list[str]] = {}
    invalid: list[str] = []
    for pat_str in patterns:
        try:
            pat = re.compile(pat_str, re.UNICODE | re.MULTILINE)
        except re.error:
            invalid.append(pat_str)
            continue
        matches = pat.findall(translated)
        if matches:
            # findall returns list[str] for a single group / no groups,
            # list[tuple] when multiple groups. Normalise.
            snippets = [m if isinstance(m, str) else " ".join(m) for m in matches]
            out[pat_str] = snippets
    return out, invalid


def _front_matter_ps(root: etree._Element) -> list[tuple[str, str]]:
    """(eId, text) for every eId-bearing `<p>` under preface/preamble."""
    out: list[tuple[str, str]] = []
    for section in root.iter():
        if not isinstance(section.tag, str):
            continue
        if etree.QName(section).localname not in ("preface", "preamble"):
            continue
        for p in section.iter(f"{{{AKN_NS}}}p"):
            eid = p.get("eId")
            if eid:
                out.append((eid, "".join(p.itertext()).strip()))
    return out


class FrontMatterDefects(NamedTuple):
    """Per-slot front-matter parity, split by what a reader loses."""

    empty_by_eid: dict[str, str]  # eId -> source text that has no target text
    duplicate_slots: int
    repeated_furniture_by_eid: dict[str, str]  # empty, but the text is elsewhere

    @property
    def empty_slots(self) -> int:
        return len(self.empty_by_eid)

    @property
    def repeated_furniture_slots(self) -> int:
        return len(self.repeated_furniture_by_eid)


def front_matter_slot_defects(source_akn_xml: str, translated_akn_xml: str) -> FrontMatterDefects:
    """Per-slot parity between source and translated front matter, empty when either
    fails to parse. An empty target slot is a dropped paragraph unless that source
    text appeared in an earlier translated slot, masthead repeats being rendered once.
    Duplicates are adjacent identical targets whose sources differ after that folding.
    """
    try:
        src_root = etree.fromstring(source_akn_xml.encode("utf-8"))
        tgt_root = etree.fromstring(translated_akn_xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        # This counter feeds a persist-blocking gate; a parse failure must
        # be visible, not read as clean.
        _logger.warning("front_matter_parity_unverifiable")
        return FrontMatterDefects({}, 0, {})

    src_pairs = _front_matter_ps(src_root)
    src_by_eid = dict(src_pairs)
    tgt_pairs = _front_matter_ps(tgt_root)
    tgt_by_eid = dict(tgt_pairs)

    folded = [(eid, fold_arabic_for_match(src)) for eid, src in src_pairs if src]
    repeated = {text for text, count in Counter(t for _, t in folded).items() if count > 1}
    # Words the reader did get. A masthead often reaches the target split over
    # the two slots that carry the title and the subject, so the repeat's own
    # text never appears verbatim in a delivered slot; its words do.
    delivered_words = {
        word for eid, text in folded if tgt_by_eid.get(eid, "").strip() for word in text.split()
    }

    empty: dict[str, str] = {}
    furniture: dict[str, str] = {}
    for eid, src in src_pairs:
        if not src or tgt_by_eid.get(eid, "").strip():
            continue
        # Blank and absent read the same to a reader, so both land here. A line
        # the source repeats, all of whose words were delivered elsewhere, is
        # furniture; anything else is a dropped paragraph.
        text = fold_arabic_for_match(src)
        benign = text in repeated and set(text.split()) <= delivered_words
        (furniture if benign else empty)[eid] = src

    duplicate = 0
    prev_eid: str | None = None
    prev_text = ""
    for eid, tgt in tgt_pairs:
        src = src_by_eid.get(eid, "")
        if (
            tgt
            and prev_eid is not None
            and tgt == prev_text
            and fold_arabic_for_match(src_by_eid.get(prev_eid, "")) != fold_arabic_for_match(src)
        ):
            duplicate += 1
        prev_eid, prev_text = eid, tgt
    return FrontMatterDefects(empty, duplicate, furniture)


def front_matter_source_retained(
    source_akn_xml: str, translated_akn_xml: str, target_language: str | None = None
) -> int:
    """Count target preface and preamble slots left in the source language: every
    front-matter fallback writes the verbatim source line and no other counter sees
    it. A slot counts only when unchanged AND carrying script the target does not
    use, since acronyms and proper names survive translation and would false-count.
    """
    patterns = alien_patterns_for(target_language)
    if not patterns:
        return 0
    try:
        src_root = etree.fromstring(source_akn_xml.encode("utf-8"))
        tgt_root = etree.fromstring(translated_akn_xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        _logger.warning("front_matter_retention_unverifiable")
        return 0
    src_by_eid = dict(_front_matter_ps(src_root))
    return sum(
        1
        for eid, tgt in _front_matter_ps(tgt_root)
        if tgt and src_by_eid.get(eid) == tgt and _has_alien_letters(tgt, patterns)
    )


def audit_translation(
    source: str,
    translated: str,
    notes: dict[str, Any],
    *,
    sentinels_expected: int = 0,
    sentinels_missing: int = 0,
    sentinels_missing_by_eid: dict[str, list[str]] | None = None,
    stray_sentinels: int = 0,
    stray_sentinels_by_eid: dict[str, list[str]] | None = None,
    source_money_surfaces_by_eid: dict[str, list[str]] | None = None,
    target_language: str | None = None,
    header_patterns: list[str] | None = None,
    translated_akn_xml: str | None = None,
) -> dict[str, Any]:
    """Every deterministic check available on a completed translation: term surfaces,
    volume drift, numeric recall, stray sentinels, alien script in `<num>` and in
    body `<p>`/`<heading>`, money recall, OCR header bleed, containers with no
    `<heading>`. Missing money and absent headings block persist, the rest advise.
    """
    # Defined-term equivalents absent from the output. Surface-form,
    # case-insensitive; an inflected occurrence may still false-flag.
    defined = notes.get("defined_terms") or []
    terms_missing: list[str] = []
    haystack = translated.lower()
    for term in defined:
        target = (term.get("target") or "").strip()
        if target and target.lower() not in haystack:
            terms_missing.append(target)

    # Content coverage catches silently-dropped bodies. Languages differ in
    # length, so this only flags gross loss, not normal expansion/compression.
    src_chars = _content_chars(source)
    tgt_chars = _content_chars(translated)

    numeric_slot_recall = (
        (sentinels_expected - sentinels_missing) / sentinels_expected if sentinels_expected else 1.0
    )
    sentinel_preservation = numeric_slot_recall  # alias

    # The alien-script check needs the AKN tree to iterate <num>; text-only
    # callers skip it. None, not 0: only a sweep that ran may report a count.
    alien_hits: int | None = None
    body_alien_hits: int | None = None
    body_alien_by_eid: dict[str, int] = {}
    heading_alien_hits: int | None = None
    heading_alien_by_eid: dict[str, int] = {}
    if target_language:
        xml_for_alien = translated_akn_xml if translated_akn_xml is not None else translated
        alien_hits = _count_alien_num_hits(xml_for_alien, target_language)
        body_result = _count_alien_hits_under_body(
            xml_for_alien, target_language, "p", _BODY_ALIEN_DOMINANCE
        )
        if body_result is not None:
            body_alien_hits, body_alien_by_eid = body_result
        else:
            body_alien_hits = None
        heading_result = _count_alien_hits_under_body(
            xml_for_alien, target_language, "heading", _HEADING_ALIEN_DOMINANCE
        )
        if heading_result is not None:
            heading_alien_hits, heading_alien_by_eid = heading_result
        else:
            heading_alien_hits = None

    # Language-independent, so it runs whether or not a target language is
    # known; only an unparseable tree can stop it.
    letter_spaced_hits: int | None = None
    letter_spaced_by_eid: dict[str, list[str]] = {}
    letter_spaced_result = count_letter_spaced_runs(
        translated_akn_xml if translated_akn_xml is not None else translated
    )
    if letter_spaced_result is not None:
        letter_spaced_hits, letter_spaced_by_eid = letter_spaced_result

    # Money-token recall: missing is a dropped penalty, duplicated is the
    # `(500,000) 50,000 Dinar` shape. Multiset per eid, so source
    # `1,000 / 3,000` vs target `1,000 / 1,000` reports the dropped `3,000`.
    money_missing_by_eid: dict[str, list[str]] = {}
    money_duplicated_by_eid: dict[str, list[str]] = {}
    if source_money_surfaces_by_eid:
        xml_for_money = translated_akn_xml if translated_akn_xml is not None else translated
        target_text_by_eid = _target_text_by_owning_eid(xml_for_money)
        money_missing_by_eid, money_duplicated_by_eid = _money_multiset_deltas(
            source_money_surfaces_by_eid, target_text_by_eid
        )

    # A container without <heading> may be the source AKN's own gap, so
    # persist treats this as advisory: a warning and this key, not a block.
    missing_container_titles_by_eid: dict[str, str] = {}
    xml_for_titles = translated_akn_xml if translated_akn_xml is not None else translated
    try:
        titles_root: etree._Element | None = etree.fromstring(xml_for_titles.encode("utf-8"))
    except etree.XMLSyntaxError:
        titles_root = None
    if titles_root is not None:
        for issue in check_missing_container_titles(titles_root):
            missing_container_titles_by_eid[issue["eid"]] = issue["container"]
    missing_container_titles_unparseable = titles_root is None

    header_bleed_hits: dict[str, list[str]] = {}
    invalid_header_patterns: list[str] = []
    if header_patterns:
        header_bleed_hits, invalid_header_patterns = _find_header_bleed(translated, header_patterns)

    deontic_counts: dict[str, int] = {}
    register_drifted = False
    register_drift_applicable = False
    if target_language:
        deontic_counts, register_drifted, register_drift_applicable = _register_drift(
            translated, target_language
        )

    return {
        # Names the language in a "check did not run" warning.
        "target_language": target_language,
        "terms_checked": len(defined),
        "terms_missing": terms_missing,
        "content_chars_source": src_chars,
        "content_chars_translated": tgt_chars,
        "content_coverage": (tgt_chars / src_chars) if src_chars else 1.0,
        # Phase 1 additions
        "sentinels_expected": sentinels_expected,
        "sentinels_missing": sentinels_missing,
        "sentinels_missing_by_eid": sentinels_missing_by_eid or {},
        "stray_sentinels": stray_sentinels,
        "stray_sentinels_by_eid": stray_sentinels_by_eid or {},
        # Placeholder text is a pipeline artefact and fatal at persist. Count
        # the XML when present, not also its text, or every hit doubles.
        "untranslated_marker_hits": (
            translated_akn_xml if translated_akn_xml is not None else translated
        ).count(_UNTRANSLATED_MARKER),
        "numeric_slot_recall": numeric_slot_recall,
        "sentinel_preservation": sentinel_preservation,
        # `None` means the alien-script check couldn't run because the AKN
        # XML was unparseable; surfaced distinctly from a clean zero.
        "num_alien_script_hits": alien_hits if alien_hits is not None else 0,
        "num_alien_script_unparseable": alien_hits is None,
        # Whole untranslated paragraphs: `<p>` under `<body>` dominated by
        # source-script chars. Excludes the preface glossary, where source
        # terms legitimately sit beside their gloss.
        "body_alien_script_hits": body_alien_hits if body_alien_hits is not None else 0,
        "body_alien_script_hits_by_eid": body_alien_by_eid,
        "body_alien_script_unparseable": body_alien_hits is None,
        # Same for `<heading>`, at a 20% threshold: headings are short and
        # hold a name, so a lower ceiling still spares inline source terms.
        "heading_alien_script_hits": heading_alien_hits if heading_alien_hits is not None else 0,
        "heading_alien_script_hits_by_eid": heading_alien_by_eid,
        "heading_alien_script_unparseable": heading_alien_hits is None,
        # Words the model emitted with a space between every letter. Not prose
        # in any target language in scope, so any hit blocks the delivery.
        "letter_spaced_runs": letter_spaced_hits if letter_spaced_hits is not None else 0,
        "letter_spaced_runs_by_eid": letter_spaced_by_eid,
        "letter_spaced_unparseable": letter_spaced_hits is None,
        # Money-token recall per owning eid, both raised at persist. Multiset
        # compare, so a dropped amount is caught when aggregate counts match.
        "money_missing_by_eid": money_missing_by_eid,
        "money_duplicated_by_eid": money_duplicated_by_eid,
        # Containers with no descriptive heading (eid → container kind);
        # advisory in the persist step (warning, not a block). `_unparseable`
        # True means the check could not run on the translated XML.
        "missing_container_titles_by_eid": missing_container_titles_by_eid,
        "missing_container_titles_unparseable": missing_container_titles_unparseable,
        "header_bleed_hits": header_bleed_hits,
        "invalid_header_patterns": invalid_header_patterns,
        # Phase 3: register-drift check (deontic-verb split above threshold).
        "deontic_counts": deontic_counts,
        "register_drifted": register_drifted,
        "register_drift_applicable": register_drift_applicable,
    }


__all__ = ["alien_patterns_for", "audit_translation"]
