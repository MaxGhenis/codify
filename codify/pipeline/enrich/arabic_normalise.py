"""Arabic-text normalisation for OCR mojibake the body-fill LLM passes through.

Each fix captures an orthographic constraint valid Arabic cannot violate: ``ة`` is
final-form, so anything but a boundary, punctuation or a diacritic after it is an
inserted character, and ``ءء`` never occurs. Untouched: trailing or
diacritic-followed ``ة``, mid-word ``ت``, non-Arabic characters.

Also carries ``strip_ocr_headers``, patterned from
``JurisdictionConfig.ocr_header_patterns``, removing running headers and footers before
translation so the LLM never sees the bleed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from lxml import etree

from codify.akn import AKN_NS
from codify.lang import normalise_digits

# Arabic letter range, letters only, NOT diacritics (U+064B-U+065F).
# A ta-marbuta followed by a diacritic is a legitimate construct (e.g. ``ةً``);
# a ta-marbuta followed by a *letter* is OCR noise.
_ARABIC_LETTER = r"[ء-يٮ-ۓ]"

# ``ة`` immediately followed by an Arabic letter, strip the rogue ta-marbuta.
_MID_WORD_TA_MARBUTA = re.compile(r"ة(?=" + _ARABIC_LETTER + r")")

# Doubled hamza, collapse to one.
_DOUBLED_HAMZA = re.compile(r"ءء+")

# Joiners the anchor regex cannot see through: TATWEEL (U+0640) elongates for
# justification (`مـادة` is `مادة` with different bytes), and ZWNJ/ZWJ can arrive
# from any word-processor round-trip. Stripping keeps body text canonical, so downstream
# regexes match visually identical strings without a decoration branch.
ARABIC_JOINERS = "ـ‌‍"
JOINER_STRIP_TABLE = str.maketrans("", "", ARABIC_JOINERS)
_JOINERS = re.compile(f"[{ARABIC_JOINERS}]")

# Arabic-index letter markers. Gazette OCR can emit the visually similar Latin
# letter (`v-`, `w-`, `c-`), collapsing the list identity. Ordered by sequence
# position, so a rewrite matches how a jurist reads the list.
_ARABIC_INDEX_LETTERS = ("أ", "ب", "ج", "د", "ه", "و", "ز", "ح", "ط", "ي")

# Line-start Latin marker: one ASCII letter, then a dash, closing paren or period,
# then whitespace. Older gazettes typeset both dash and paren forms. The letter
# is a visual OCR of an Arabic index letter, so recovery goes by list position, not
# character identity, which is what the OCR destroyed.
_LATIN_BULLET_LINE_RE = re.compile(r"(?m)^([A-Za-z])\s*(?:[\-–—)]|\.)\s+")

# Fused numbering: a line-start digit run welded to a word, observed as
# `1Formulating`. The trailing group needs a capital plus a lowercase, or two
# Arabic-block characters, so a sub-article marker (`5A The`) or a shorthand cite
# (`18USC`) is not mangled.
_FUSED_NUMBERING_RE = re.compile(r"(?m)^(\d+)([A-Z][a-z]|[؀-ۿ]{2})")

# Consecutive duplicate marker: `\n2 2` at line-start (or start-of-string)
# collapses to `\n2`. Observed as `2 2`.
_DUP_INDEX_MARKER_RE = re.compile(r"(?m)^(\d+)\s+\1(?=\s|$)")


# Arabic-script presence check. The Arabic-only transforms would otherwise corrupt
# Latin-script documents: a lettered `a- foo\nb- bar` sub-list on a gb or us AKN
# would be rewritten to `أ- foo\nب- bar`.
_ARABIC_CHAR_RE = re.compile(r"[؀-ۿ]")


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC_CHAR_RE.search(text))


def _remap_latin_bullets(text: str) -> tuple[str, int]:
    """Replace each line-anchored Latin-letter marker with the Arabic-index letter at its
    position within the run.

    A run is consecutive lines whose every non-blank line starts with such a marker.
    Position resets between runs, so two independent lists both restart at `أ-` rather
    than continuing mid-alphabet. Matches beyond position 10 (`ي`) stay put, so an
    operator sees a suspicious long list rather than an invented remap.
    """
    lines = text.split("\n")
    out_lines: list[str] = []
    idx = 0
    total = 0
    for line in lines:
        m = _LATIN_BULLET_LINE_RE.match(line)
        if m and idx < len(_ARABIC_INDEX_LETTERS):
            rest = line[m.end() :]
            out_lines.append(f"{_ARABIC_INDEX_LETTERS[idx]}- {rest}")
            idx += 1
            total += 1
        else:
            if not line.strip():
                # Blank line ends a marker run.
                idx = 0
            elif not m:
                # Non-marker prose ends a marker run.
                idx = 0
            out_lines.append(line)
    return "\n".join(out_lines), total


# Fold for containment matching between raw OCR text and AKN text: strip
# harakat + dagger alef + tatweel, fold alef/ya/ta-marbuta variants, collapse
# whitespace. Both sides drift on exactly these axes during transcription.
_MATCH_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_MATCH_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def fold_arabic_for_match(text: str) -> str:
    """Orthography-insensitive form for substring checks (not for display)."""
    text = _MATCH_DIACRITICS.sub("", text).translate(_MATCH_FOLD)
    return re.sub(r"\s+", " ", text).strip()


def normalise_arabic_text(text: str, *, apply_arabic_only: bool = True) -> tuple[str, int]:
    """Return ``(cleaned_text, edit_count)``. Pure; safe on any string.

    The list-bullet remap, fused-numbering and duplicate-marker collapse run only when
    ``apply_arabic_only`` is set, so an English fragment inside an Arabic document is
    still rewritten while a wholly English one is untouched. Callers walking a tree
    decide the flag once for the document and pass it down per fragment. The joiner
    strip, ta-marbuta and doubled-hamza fixes are script-safe and always run.
    """
    cleaned, joiner_count = _JOINERS.subn("", text)
    cleaned, mid_count = _MID_WORD_TA_MARBUTA.subn("", cleaned)
    cleaned, hamza_count = _DOUBLED_HAMZA.subn("ء", cleaned)
    bullet_count = 0
    fused_count = 0
    dup_count = 0
    if apply_arabic_only:
        cleaned, bullet_count = _remap_latin_bullets(cleaned)
        cleaned, fused_count = _FUSED_NUMBERING_RE.subn(r"\1. \2", cleaned)
        cleaned, dup_count = _DUP_INDEX_MARKER_RE.subn(r"\1", cleaned)
    return cleaned, joiner_count + mid_count + hamza_count + bullet_count + fused_count + dup_count


def _apply_to_p_text(root: etree._Element, transform: Callable[[str], tuple[str, int]]) -> int:
    """Apply ``transform`` to every ``<p>`` descendant's ``.text`` and each
    child element's ``.tail``. Returns the total substitution count. The
    mark-up structure is untouched, only rendered body text changes."""
    total = 0
    for p in root.iter(f"{{{AKN_NS}}}p"):
        if p.text:
            fixed, n = transform(p.text)
            if n:
                p.text = fixed
                total += n
        for child in p:
            if child.tail:
                fixed, n = transform(child.tail)
                if n:
                    child.tail = fixed
                    total += n
    return total


def normalise_arabic_in_tree(root: etree._Element, *, country: str = "") -> int:
    """Fold Arabic OCR mojibake across every ``<p>`` descendant, returning the substitution
    count. The Arabic-only transforms run when the jurisdiction declares Arabic script,
    or when it declares nothing usable and the content is Arabic. Decided once for the
    tree, so an English fragment inside an Arabic act is rewritten while a gb, us or
    eu-27 corpus cannot have a lettered sub-list mapped to Arabic-index letters.
    """
    from codify.jurisdictions import load_config
    from codify.pipeline.enrich.scripts import ARABIC_SCRIPT, resolve_pack

    document_text = " ".join(t for t in root.itertext() if t)
    cfg = load_config(country) if country else None
    pack = resolve_pack(cfg, document_text)
    apply_arabic_only = pack is not None and pack.script == ARABIC_SCRIPT
    return _apply_to_p_text(
        root, lambda t: normalise_arabic_text(t, apply_arabic_only=apply_arabic_only)
    )


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """Compile jurisdiction OCR-header patterns. Multiline for line-anchored
    forms; unicode for the Arabic ranges. Invalid patterns raise at compile
    time so a broken profile is caught at load, not at translation."""
    return [re.compile(p, re.MULTILINE | re.UNICODE) for p in patterns]


def strip_ocr_headers_text(text: str, compiled: list[re.Pattern[str]]) -> tuple[str, int]:
    """Return ``(cleaned_text, edit_count)`` with each pattern substituted
    out. Consecutive-whitespace collapse afterwards keeps prose readable
    where a mid-paragraph header was removed."""
    if not text or not compiled:
        return text, 0
    total = 0
    cleaned = text
    for pat in compiled:
        cleaned, n = pat.subn("", cleaned)
        total += n
    if total:
        # Collapse multi-space runs into one; strip leading/trailing space
        # per line so a removed header doesn't leave dangling whitespace.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"^[ \t]+|[ \t]+$", "", cleaned, flags=re.MULTILINE)
    return cleaned, total


def drop_header_only_lines(text: str, patterns: list[str]) -> tuple[str, int]:
    """Remove lines that are nothing but gazette furniture; return ``(text, dropped_count)``.

    A line reduced to whitespace by the jurisdiction's header patterns carried no content
    and is dropped. A partial match keeps its remainder, preserving provenance that shares
    a line with a masthead. Blank lines already present survive, so callers reading
    blank-line structure are unaffected.
    """
    if not text or not patterns:
        return text, 0
    compiled = _compile_patterns(patterns)
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        cleaned, hits = strip_ocr_headers_text(line, compiled)
        if hits and line.strip() and not cleaned.strip():
            dropped += 1
            continue
        kept.append(cleaned if hits else line)
    return "\n".join(kept), dropped


def strip_ocr_headers(root: etree._Element, patterns: list[str]) -> int:
    """Strip OCR header/footer bleed from every ``<p>`` descendant. Returns
    the substitution count; an empty ``patterns`` list is a zero-work no-op."""
    if not patterns:
        return 0
    compiled = _compile_patterns(patterns)
    return _apply_to_p_text(root, lambda t: strip_ocr_headers_text(t, compiled))


# Arabic ordinal words used as chapter or part numbers, mapped to Latin integers so
# a translated render reads "Chapter 2" rather than "Chapter الثاني". Exact-key
# lookups. Feminine forms map to the same integer as the masculine, so gender
# agreement in the source does not change the digit.
AR_ORDINAL_TO_INT: dict[str, int] = {
    "الحادي عشر": 11,
    "الحادية عشرة": 11,
    "الثاني عشر": 12,
    "الثانية عشرة": 12,
    "الثالث عشر": 13,
    "الثالثة عشرة": 13,
    "الرابع عشر": 14,
    "الرابعة عشرة": 14,
    "الخامس عشر": 15,
    "الخامسة عشرة": 15,
    "السادس عشر": 16,
    "السادسة عشرة": 16,
    "السابع عشر": 17,
    "السابعة عشرة": 17,
    "الثامن عشر": 18,
    "الثامنة عشرة": 18,
    "التاسع عشر": 19,
    "التاسعة عشرة": 19,
    "العشرون": 20,
    "الحادي والعشرون": 21,
    "الثاني والعشرون": 22,
    "الثالث والعشرون": 23,
    "الرابع والعشرون": 24,
    "الخامس والعشرون": 25,
    "السادس والعشرون": 26,
    "السابع والعشرون": 27,
    "الثامن والعشرون": 28,
    "التاسع والعشرون": 29,
    "الثلاثون": 30,
    "الأول": 1,
    "الأولى": 1,
    "الاول": 1,
    "الثاني": 2,
    "الثانية": 2,
    "الثالث": 3,
    "الثالثة": 3,
    "الرابع": 4,
    "الرابعة": 4,
    "الخامس": 5,
    "الخامسة": 5,
    "السادس": 6,
    "السادسة": 6,
    "السابع": 7,
    "السابعة": 7,
    "الثامن": 8,
    "الثامنة": 8,
    "التاسع": 9,
    "التاسعة": 9,
    "العاشر": 10,
    "العاشرة": 10,
}


# Keys with spaces collapsed ("الحاديعشر") for inputs where the space was
# lost upstream (OCR line-joins, Bluebell eId derivation).
_AR_ORDINAL_NOSPACE: dict[str, int] = {k.replace(" ", ""): v for k, v in AR_ORDINAL_TO_INT.items()}


def arabic_ordinal_words() -> tuple[str, ...]:
    """Every ordinal `latinise_arabic_ordinal` folds, spaced and space-collapsed, for
    callers that must detect an eId the fold will change (the normalise-eids selector).
    These are not in the jurisdictions' declared word lists, so a caller reading only
    those sees `chp_الأول` as clean."""
    return tuple(sorted(set(AR_ORDINAL_TO_INT) | set(_AR_ORDINAL_NOSPACE), reverse=True))


def latinise_arabic_ordinal(text: str) -> str | None:
    """The Latin-digit form of an Arabic ordinal word, or None. Called on ``<num>`` text of
    containers numbered by ordinal. Input is joiner-normalised before lookup, so ``الأول``
    matches whatever TATWEEL the typesetter inserted, and compound ordinals match with
    their space collapsed (``الحاديعشر``). None lets callers fall through to abjad or
    digit-translate without a special case.
    """
    if not text:
        return None
    key = text.translate(JOINER_STRIP_TABLE).strip()
    n = AR_ORDINAL_TO_INT.get(key)
    if n is None:
        n = _AR_ORDINAL_NOSPACE.get(key.replace(" ", ""))
    return str(n) if n is not None else None


# Punctuation a heading or num may carry without changing what it says.
_HEADING_TRIM = " \t.-\u2013\u2014:\u060c,()[]"


def heading_restates_number(heading: str, number: str) -> bool:
    """True when a heading says nothing its own ``<num>`` already says: the bare-ordinal
    shape, where the heading is just the container's ordinal word and the num carries the
    same word or its digit. Either reads as a container with no title while looking
    deliberate. Ordinal words compare by integer value, so the two numbering styles
    collapse onto each other.
    """
    h = fold_arabic_for_match(heading).strip(_HEADING_TRIM)
    if not h:
        return True
    n = fold_arabic_for_match(number).strip(_HEADING_TRIM)
    # The scanner preserves the source's digit script, so the same ordinal can
    # reach here as a word on one side and an Arabic-Indic digit on the other.
    # Reduce both to a Latin-digit value before comparing, in either direction.
    h_val = latinise_arabic_ordinal(h) or normalise_digits(h)
    n_val = latinise_arabic_ordinal(n) or normalise_digits(n)
    return h == n or h_val == n_val


__all__ = [
    "ARABIC_JOINERS",
    "AR_ORDINAL_TO_INT",
    "JOINER_STRIP_TABLE",
    "latinise_arabic_ordinal",
    "normalise_arabic_text",
    "normalise_arabic_in_tree",
    "strip_ocr_headers",
    "strip_ocr_headers_text",
]
