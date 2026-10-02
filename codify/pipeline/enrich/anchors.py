"""Per-jurisdiction structural anchor scanning."""

from __future__ import annotations

import re
import statistics
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from functools import lru_cache

import structlog

from codify.akn.eid import eid_abbrev
from codify.jurisdictions import (
    HierarchyEntry,
    JurisdictionConfig,
    JurisdictionConfigError,
    fold_inserted_suffix,
    heading_line_pattern,
    load_config,
    ordinal_word_folds,
)
from codify.lang import normalise_digits
from codify.pipeline.enrich.adoption import TITLE_BLOCK_CHARS, adopts_external_text
from codify.pipeline.enrich.arabic_normalise import (
    AR_ORDINAL_TO_INT,
    fold_arabic_for_match,
    latinise_arabic_ordinal,
)
from codify.pipeline.enrich.cover_reconciliation import (
    COVER_TEXT_CHARS,
    extract_cover_article_numbers,
)
from codify.pipeline.enrich.digit_confusion import repair_number_sequence
from codify.pipeline.enrich.kinds import CONTAINER_KINDS, KIND_RANK
from codify.pipeline.enrich.scripts import dominant_script
from codify.pipeline.enrich.scripts.arabic import ARABIC, abjad_index_to_latin
from codify.quality.invariants import (
    AmbiguitySpan,
    classify_gap,
    missing_between,
    order_key,
)

logger = structlog.get_logger()

# Built-in keyword aliases, indexed by AKN element name. Layered on top of
# whatever ``local_term`` + ``bluebell_keyword`` the jurisdiction config
# supplies. Keys are AKN element names.
_BUILTIN_ALIASES: dict[str, tuple[str, ...]] = {
    "part": ("PART", "Part", "Partie", "Parte", "Teil"),
    "title": ("TITLE", "Title", "Titre", "Título", "Titel"),
    "chapter": (
        "CHAPTER",
        "Chapter",
        "Chap.",
        "Chapitre",
        "Capítulo",
        "Capitolo",
        "Kapitel",
        "Глава",
        "Бүлэг",
    ),
    "subchapter": ("SUBCHAPTER", "Subchapter"),
    "section": (
        "SECTION",
        "Section",
        "Sec.",
        "§",
        "Sección",
        "Section",
        "Abschnitt",
    ),
    "subsection": ("SUBSECTION", "Subsection"),
    "article": (
        "ARTICLE",
        "Article",
        "Art.",
        "Artículo",
        "Artikel",
        "Articolo",
        "Madde",
        "Статья",
    ),
    "paragraph": ("PARAGRAPH", "Paragraph", "Para.", "¶"),
    "subparagraph": ("SUBPARAGRAPH", "Subparagraph"),
    "point": ("POINT", "Point", "Punkt"),
    "item": ("ITEM", "Item"),
    "book": ("BOOK", "Book", "Livre", "Libro", "Buch"),
    "tome": ("TOME", "Tome"),
    "division": ("DIVISION", "Division"),
    "schedule": (
        "SCHEDULE",
        "Schedule",
        "Annex",
        "Annexe",
        "Anexo",
        "Anhang",
        # Arabic annex/table headings; the column-boundary anchor plus the
        # prose-precursor filter keep in-text "الملحق رقم (1)" references out.
        "الملحق رقم",
        "ملحق رقم",
        "الملحق",
        "ملحق",
        "الجدول رقم",
        "جدول رقم",
        "الجدول",
        "جدول",
    ),
}

# Arabic ordinal words as chapter/part numbers (الباب الأول): older codes
# use ordinals above article level. Longest-first so compounds beat prefixes.
_AR_ORDINAL_RE = "|".join(
    re.escape(w) for w in sorted(AR_ORDINAL_TO_INT, key=lambda w: (-len(w), w))
)

# Bis-article suffix ("المادة (6) مكرر" = 6bis), absorbed into the num group so
# it neither leaks into body text nor collides with the base number. The index
# paren is left unconsumed so `_MARKER_NUM_END` still sees an edge.
_BIS_SUFFIX = r"(?:\s*\)?\s*مكرر(?:ة)?(?:\s*\(?\s*(?:[0-9٠-٩۰-۹]+|[ء-ي](?=\s*\))))?)?"

# Between keyword and number. Parens run both ways because RTL extraction
# mirrors `(1)` and often drops the opener; dashes cover `مادة— ٢٧`.
_MARKER_NUM_SEPARATOR = r"[\s.)(\[\-–—]+"

# Every digit `_NUM_PATTERN` accepts, so the boundary rule below applies to all
# of them rather than to Western and Arabic-Indic alone.
_MARKER_DIGITS = r"0-9٠-٩۰-۹০-৯०-९๐-๙"

# A digit followed by a letter is no `\b`, hiding a rubric fused to its number
# (`مادة — ٢تفسير`). Relaxed only after a digit: Roman numerals and lone letters
# keep `\b`, or `Chapter Definitions` anchors as chapter `D`.
_MARKER_NUM_END = rf"(?:(?<=[{_MARKER_DIGITS}])(?![{_MARKER_DIGITS}])|(?<![{_MARKER_DIGITS}])\b)"

# No separator means the number must be the rest of the line, or `Pasal2lT`
# reads as article 2 and invents a wrong-numbered one.
_MARKER_NUM_TAIL = r"(?(sep)|(?=[ \t]*$))"


def _inserted_suffix_end(suffixes: Mapping[str, str]) -> str:
    """The number's end, or a declared insertion suffix ("5 bis") ending it instead.
    The suffix may wrap onto the next line and ends on a non-word character."""
    if not suffixes:
        return _MARKER_NUM_END
    words = "|".join(re.escape(w) for w in sorted(suffixes, key=lambda w: (-len(w), w)))
    # `(?-i:…)`: the denominator compiles case-blind, and a suffix is a word.
    return rf"(?:(?P<suffix>[ \t]*(?:\r?\n)?[ \t]*(?-i:{words}))(?!\w)|{_MARKER_NUM_END})"


def _separator_for(tolerances: AbstractSet[str]) -> str:
    """Keyword-to-number separator, optional where the sources lose it."""
    if "missing_separator" not in tolerances:
        return _MARKER_NUM_SEPARATOR
    return rf"(?P<sep>{_MARKER_NUM_SEPARATOR})?"


# Number alternation: Western Arabic, Roman, Arabic-Indic, Bengali,
# Devanagari, Thai digits, a single Latin letter, or an Arabic ordinal word.
_NUM_ALTS = (
    # Roman numerals, incl. the Cyrillic homoglyphs OCR yields for Ukrainian
    # section rubrics ("РОЗДІЛ І" uses Cyrillic І U+0406, Х U+0425).
    r"[IVXLCDMivxlcdmІіХх]+"
    # Hyphenated or slashed insertions ("13-1", spaced "13 - 1", "7/1").
    r"|\d+(?:\s?[-\u2013/]\s?\d+)+"
    r"|\d+[A-Za-z]?"  # Western Arabic, optional letter suffix (e.g. 5A)
    r"|[٠-٩]+"  # Arabic-Indic ٠-٩
    r"|[۰-۹]+"  # Extended Arabic-Indic (Persian/Urdu) ۰-۹
    r"|[০-৯]+"  # Bengali ০-৯
    r"|[०-९]+"  # Devanagari ०-९
    r"|[๐-๙]+"  # Thai ๐-๙
    rf"|{_AR_ORDINAL_RE}"  # Arabic ordinal words (compound first)
    r"|[ء-ي](?=\s*\))"  # abjad letter in parens: annex "الملحق (أ)"
    r"|[a-zA-Z]"
)

_NUM_PATTERN = rf"(?P<num>(?:{_NUM_ALTS}){_BIS_SUFFIX})"


# Digit misreads inside a number, only where the run also holds a real digit:
# `Pasal l1` is 11, while `Pasal I` and `Pasal II` are genuine Roman structure.
# `T` for 7 is attested (`Pasal2T` after a lead-in naming Pasal 27).
# ponytail: a real T suffix would fold to 147; unattested, twenty insertions
# deep. Spare the final character if one appears.
_DIGIT_GLYPHS = str.maketrans({"l": "1", "L": "1", "I": "1", "O": "0", "o": "0", "T": "7"})
_GLYPHED_NUM = r"(?=[\dlLIOoT]*\d)(?P<numglyph>[\dlLIOoT]+[A-Za-z]?)"
# A number split across a space: `Pasal 2 1` is 21, and the leading group alone
# reads 2 and displaces the real one. Groups must run to end of line so
# "Pasal 5 ayat (2)" still matches as 5.
_SPLIT_CHARS = "\\d"
_GLYPH_CHARS = "\\dlLIOoT"


def _split_num(chars: str) -> str:
    return rf"(?=[{chars}]*\d)(?P<numsplit>[{chars}]+(?:[ \t][{chars}]+)+[A-Za-z]?)(?=[ \t]*$)"


def _num_pattern_with(
    ordinals: Mapping[str, int], *, digit_glyphs: bool = False, split_numbers: bool = False
) -> str:
    """`_NUM_PATTERN`, with declared ordinal words tried first.

    Ordinals lead or `[a-zA-Z]` takes "Kesatu"'s K and strands the rest.
    Case-sensitive, separating heading "Bagian Kesatu" from prose "Bagian ketiga".
    """
    # The split form leads: a plain digit run would otherwise take the first
    # group and stop.
    # Digit-only groups unless `digit_glyph` is declared too, so enabling one
    # tolerance cannot quietly admit the other's damage.
    extra = (
        f"{_split_num(_GLYPH_CHARS if digit_glyphs else _SPLIT_CHARS)}|" if split_numbers else ""
    )
    if digit_glyphs:
        extra += f"{_GLYPHED_NUM}|"
    if not ordinals:
        return rf"(?P<num>(?:{extra}{_NUM_ALTS}){_BIS_SUFFIX})" if extra else _NUM_PATTERN
    forms = {f for w in ordinals for f in (w, w.upper())}
    # Compound ordinals wrap on centred headings, so inner spaces match any
    # whitespace. Longest-first with a tiebreak: set order is not stable.
    alts = "|".join(
        re.sub(r"\\ ", r"\\s+", re.escape(f)) for f in sorted(forms, key=lambda f: (-len(f), f))
    )
    # Scoped case-sensitivity: the coverage denominator compiles with (?i),
    # and a case-blind ordinal counts the prose the scanner refuses, which
    # reads back as lost coverage.
    return rf"(?P<num>(?:(?-i:{alts})|{extra}{_NUM_ALTS}){_BIS_SUFFIX})"


# Kinds whose anchors may carry a captured heading. `section` is included
# for jurisdictions where it groups articles (Ukraine's РОЗДІЛ); capture
# itself gates section on the document actually containing articles.
_HEADING_KINDS = CONTAINER_KINDS | {"section", "schedule"}
# Narrower vocabulary than the capture set: a lone subpart is ordinary internal
# structure, not a gazette wrapper.
_LONE_WRAPPER_KINDS = frozenset(
    {"book", "tome", "part", "title", "chapter", "subchapter", "division"}
)


@dataclass(frozen=True)
class StructuralAnchor:
    """Regex-detected structural boundary; depth and eIds are filled at scan time.

    ``quoted_amendment`` marks an embedded amendment article excluded from the host
    hierarchy. ``heading`` is a container title; ``source_pass`` names the pass.
    """

    kind: str
    keyword: str
    number: str | None
    char_offset: int
    line: int
    matched_text: str
    depth: int = 0
    parent_eid: str | None = None
    akn_eid: str = ""
    akn_wid: str = ""
    quoted_amendment: bool = False
    # An amendment item introducing another statute's article ("Angka 21"), so
    # it parents that article instead of being the deepest subdivision.
    amendment_item: bool = False
    # The amended statute's article, which nests under that item.
    amended_article: bool = False
    heading: str | None = None
    source_pass: str = ""

    def __post_init__(self) -> None:
        if self.heading is not None and self.kind not in _HEADING_KINDS:
            raise ValueError(f"heading on non-container anchor kind {self.kind!r}")


@dataclass(frozen=True)
class Window:
    """Anchor-grouped slice of raw text; body_start/_end mark the in-scope range."""

    anchors: tuple[StructuralAnchor, ...]
    text: str
    body_start: int
    body_end: int


def _split_term_with_parens(term: str) -> list[str]:
    """Split a config ``local_term`` into its primary form plus same-script
    parenthesised variants.

    Keeps declensions, which reference resolution needs to match inflected
    citations. Anchor scanning uses :func:`_heading_forms`, which excludes them.
    """
    if "(" not in term:
        return [term.strip()]
    head, _, tail = term.partition("(")
    head = head.strip()
    paren_body = tail.rsplit(")", 1)[0]
    out = [head] if head else []
    head_script = dominant_script(head)
    for piece in re.split(r"[/,]", paren_body):
        piece = piece.strip()
        if not piece:
            continue
        if head_script and dominant_script(piece) == head_script:
            out.append(piece)
    return out


def _heading_forms(term: str) -> list[str]:
    """Heading forms of a ``local_term``: the primary plus any parenthesised
    abbreviation, never a declension.

    Declensions appear only in prose, so admitting them promotes in-text references
    to phantom headings. The primary form is kept even when lowercase.
    """
    forms = _split_term_with_parens(term)
    if not forms:
        return forms
    head, *variants = forms
    kept = [head]
    for variant in variants:
        first = next((c for c in variant if c.isalpha()), "")
        if first and first.islower() and first.upper() != first.lower():
            continue
        kept.append(variant)
    return kept


def _alias_terms_for(entry: HierarchyEntry) -> tuple[str, ...]:
    """Every textual form that should match this hierarchy entry.

    A level declaring `marker_form` keeps its keyword forms too: the same corpus
    that prints the marker bare also spells it out, and a declaration that
    disabled the spelling would drop those provisions silently.
    """
    aliases: list[str] = []
    if entry.local_term:
        for form in _heading_forms(entry.local_term):
            aliases.append(form)
            aliases.append(form.title())
            aliases.append(form.upper())
    if entry.bluebell_keyword:
        aliases.append(entry.bluebell_keyword)
        aliases.append(entry.bluebell_keyword.title())
    if entry.local_terms:
        for term in entry.local_terms.values():
            if not term:
                continue
            for form in _heading_forms(term):
                aliases.append(form)
                aliases.append(form.title())
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    # Arabic legal headers oscillate between bare and ال-prefixed forms.
    arabic_prefixed: list[str] = []
    for a in out:
        if dominant_script(a) == "ARABIC" and not a.startswith("ال"):
            prefixed = f"ال{a}"
            if prefixed not in seen:
                seen.add(prefixed)
                arabic_prefixed.append(prefixed)
    out.extend(arabic_prefixed)
    return tuple(out)


# Optional Arabic joiners: TATWEEL (U+0640), ZWNJ/ZWJ (U+200C/U+200D). The
# normaliser strips these pre-scan, so this serves callers passing raw text,
# chiefly the coverage-delta counter.
_ARABIC_JOINER_OPT = ARABIC.joiner_opt


# Letter-for-letter misreads the vision OCR produces on this corpus, keyed by
# the true letter: `Pasal` reads `Pasai`, `Angka` reads `Ang)<a` when the
# ligature breaks. Narrow by evidence; a broad class turns prose into markers.
_KEYWORD_GLYPHS = {"l": "li1I", "k": "k", "i": "il1"}
_KEYWORD_LIGATURES = {"k": r"[)\]]<"}


def _glyph_class(ch: str) -> str:
    """The character, plus the glyphs the scan renders it as."""
    variants = _KEYWORD_GLYPHS.get(ch.lower())
    if variants is None:
        return re.escape(ch)
    cased = "".join(v.upper() if ch.isupper() else v for v in variants)
    alt = _KEYWORD_LIGATURES.get(ch.lower())
    cls = f"[{re.escape(cased)}]"
    return f"(?:{cls}|{alt})" if alt else cls


def alias_regex_fragment(alias: str, *, glyph_tolerant: bool = False) -> str:
    """Compile one alias literal into a regex fragment, inserting an
    optional-joiner class between adjacent Arabic letters so
    ``مادة``, ``مـادة``, ``م‌ادة``, and ``م‍ادة`` all match the same
    pattern. Non-Arabic aliases pass through ``re.escape`` unchanged."""
    if not any(ARABIC.is_letter(c) for c in alias):
        if glyph_tolerant:
            return "".join(_glyph_class(c) for c in alias)
        return re.escape(alias)
    # Drop any joiners in the config-supplied alias so its own byte-shape
    # does not disagree with the emitted pattern.
    letters = [c for c in alias if c not in "ـ‌‍"]
    pieces: list[str] = []
    for i, ch in enumerate(letters):
        pieces.append(re.escape(ch))
        if i + 1 < len(letters) and ARABIC.is_letter(ch) and ARABIC.is_letter(letters[i + 1]):
            pieces.append(_ARABIC_JOINER_OPT)
    return "".join(pieces)


@lru_cache(maxsize=64)
def _aliases_for(country: str, doctype: str) -> dict[str, tuple[str, ...]] | None:
    """Alias sets by kind. None for an unnamed jurisdiction; a named one without
    a config raises, because the alternation would otherwise be English."""
    if not country:
        return None
    return keyword_aliases(load_config(country), doctype)


def keyword_aliases(
    config: JurisdictionConfig | None,
    doctype: str,
) -> dict[str, tuple[str, ...]]:
    """Every textual form each hierarchy kind answers to, keyed by AKN element.

    ``_BUILTIN_ALIASES`` is English, so it is the answer only where there is no
    config to consult. A named jurisdiction whose config yields no hierarchy for
    this doctype raises rather than scanning a non-Latin document for `Chapter`.
    """
    kind_to_aliases: dict[str, tuple[str, ...]] = {}
    if config is not None:
        doc_class = config.get_document_class(doctype)
        if doc_class:
            for entry in doc_class.hierarchy:
                kind_to_aliases[entry.akn_element] = _alias_terms_for(entry)
    if not kind_to_aliases:
        if config is not None:
            raise JurisdictionConfigError(
                f"jurisdiction {config.code!r} declares no hierarchy for doctype "
                f"{doctype!r}; declared: {sorted(config.document_classes)}"
            )
        return {k: v for k, v in _BUILTIN_ALIASES.items() if v}
    # Configs rarely declare annexes in their hierarchy, yet annex material is
    # real law (schedules, salary tables, forms). Union the builtin schedule
    # aliases in so a config-driven scan still anchors them.
    merged = (*kind_to_aliases.get("schedule", ()), *_BUILTIN_ALIASES["schedule"])
    kind_to_aliases["schedule"] = tuple(dict.fromkeys(a for a in merged if a))
    return kind_to_aliases


# What may precede a marker for it to count as a header; `relaxed` admits one
# space for old scans that lost column layout. Both keep the marker's
# newline inside the match, which `_is_prose_reference` reads back from.
_BOUNDARIES = {
    "relaxed": r"(?:^\s{0,8}|\t\s*|\s+)",
    "line_anchored": r"(?:^[^\S\n]{0,8}|\t\s*|\n[^\S\n]*|[^\S\n]{3,})",
    # `relaxed` without the `|\s+` arm: the coverage denominator counts only
    # column-anchored markers, so a mid-prose alias+number is not expected.
    "column_only": r"(?:^\s{0,8}|\t\s*)",
}


def _policy(config: JurisdictionConfig | None) -> str:
    if config is None or config.structuring is None:
        return "relaxed"
    return config.structuring.marker_boundary


def markers_outside_boundary(
    text: str, config: JurisdictionConfig | None, doctype: str, kind: str
) -> int:
    """Markers of ``kind`` the relaxed boundary sees and the declared one does
    not. Non-zero against an empty coverage denominator means the source lost
    the line layout `line_anchored` assumes, so the policy cannot be trusted."""
    if _policy(config) == "relaxed":
        return 0
    group = _group_name(kind)

    def seen(rx: re.Pattern[str]) -> int:
        if group not in rx.groupindex:
            return 0
        return sum(1 for m in rx.finditer(text) if m.group(group))

    return seen(build_anchor_regex(config, doctype, boundary="relaxed")) - seen(
        build_anchor_regex(config, doctype)
    )


def build_anchor_regex(
    config: JurisdictionConfig | None,
    doctype: str,
    *,
    boundary: str | None = None,
) -> re.Pattern[str]:
    """Compile a multilingual structural-anchor regex."""
    return _compile_anchor_regex(keyword_aliases(config, doctype), config, boundary)


def _compile_anchor_regex(
    kind_to_aliases: Mapping[str, Iterable[str]],
    config: JurisdictionConfig | None,
    boundary: str | None = None,
    case_insensitive: bool = False,
) -> re.Pattern[str]:
    """`build_anchor_regex`, over an explicit kind-to-aliases map.

    ``case_insensitive`` compiles ``(?i)`` for the coverage denominator; the ordinal
    alternation is guarded by ``(?-i:…)`` so the flag never widens the number match.
    """
    structuring = config.structuring if config else None
    tolerances = set(structuring.marker_tolerances if structuring else ())
    glyphs = "keyword_glyph" in tolerances

    # Build one named alternation per kind so we can recover the AKN element
    # from the match. Names must be regex-safe; AKN element names are.
    parts: list[str] = []
    for kind, aliases in kind_to_aliases.items():
        # Longest-first ("Article" before "Art"); ties break by value so the
        # compiled pattern is identical across processes (set order is seed-dependent).
        ordered = sorted({a for a in aliases if a}, key=lambda a: (-len(a), a))
        if not ordered:
            continue
        alts = "|".join(alias_regex_fragment(a, glyph_tolerant=glyphs) for a in ordered)
        parts.append(rf"(?P<{_group_name(kind)}>{alts})")
    if not parts:
        return re.compile(r"$^")  # never matches
    keyword_group = "|".join(parts)
    num_pattern = _num_pattern_with(
        structuring.ordinal_words if structuring else {},
        digit_glyphs="digit_glyph" in tolerances,
        split_numbers="split_number" in tolerances,
    )
    flags = "(?im)" if case_insensitive else "(?m)"
    pattern = (
        rf"{flags}{_BOUNDARIES[boundary or _policy(config)]}(?:{keyword_group})"
        rf"{_separator_for(tolerances)}{num_pattern}"
        rf"{_inserted_suffix_end(structuring.insertion_suffixes if structuring else {})}"
        rf"{_MARKER_NUM_TAIL if 'missing_separator' in tolerances else ''}"
    )
    return re.compile(pattern)


# Scanned Arabic drops the hamza (أ إ آ -> ا), so a precursor spelled with one
# never matches the OCR text. Fold both sides before comparing.
_HAMZA_FOLD = str.maketrans("أإآ", "ااا")


def _fold_hamza(value: str) -> str:
    return value.translate(_HAMZA_FOLD)


def _precursor_alt(words: Iterable[str]) -> str:
    return "|".join(re.escape(_fold_hamza(w)) for w in words)


# Arabic prose precursors that signal "this المادة/الباب is an in-text
# cross-reference, not a structural marker". "في المادة ٥" = "in Article 5".
_AR_PROSE_PRECURSORS = ARABIC.prose_precursors
# Optional one-char conjunction prefix (`و/ف/ل/ب/ك`), Arabic glues
# conjunctions to the next word (`وبموجب` = "and by virtue of"), so a
# whitespace-only anchor before the precursor misses these forms.
_AR_PROSE_FILTER_RE = re.compile(
    rf"(?:^|\s|[،,(\[])[وفلبك]?(?:{_precursor_alt(_AR_PROSE_PRECURSORS)})\s*$"
)

# Precursors that mark a citation only when adjacent on one line: they also end
# ordinary sentences, where the next line may be a genuine header.
_AR_SAMELINE_PRECURSORS = ARABIC.sameline_precursors
_AR_SAMELINE_FILTER_RE = re.compile(
    rf"(?:^|\s|[،,(\[])[وفلبك]?(?:{_precursor_alt(_AR_SAMELINE_PRECURSORS)})\s*$"
)


# Scripts that run words together, so a word there touches its neighbours.
_UNSPACED_SCRIPTS = "\u0e00-\u0e7f\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff"


def _word_end(word: str) -> str:
    """A whole-word end for the regex, except in a script that runs words together."""
    return "" if re.search(f"[{_UNSPACED_SCRIPTS}]$", word) else r"(?!\w)"


def _precursor_re(words: tuple[str, ...]) -> re.Pattern[str]:
    if not words:
        return re.compile(r"(?!)")
    alts = _precursor_alt(words)
    # A trailing colon is part of the precursor, not a break from it:
    # a legal-basis line reads "Mengingat :   Pasal 5 ayat (1)".
    # Case-insensitive: a precursor is a word, and a line can end on it in any
    # case a scan produces.
    return re.compile(
        rf"(?:^|\s|[،,(\[]|(?<=[{_UNSPACED_SCRIPTS}]))[وفلبك]?(?:{alts})\s*[:：]?\s*$",
        re.IGNORECASE,
    )


@lru_cache(maxsize=32)
def _prose_filter_for(country: str) -> re.Pattern[str]:
    """The wrapped-citation filter, config-declared where a jurisdiction needs
    one. A boundary policy cannot see these: the marker legitimately starts its
    line and only the previous line's last word says it is a citation."""
    config = load_config(country) if country else None
    declared = config.structuring.prose_precursors if config and config.structuring else []
    return _precursor_re(tuple(declared)) if declared else _AR_PROSE_FILTER_RE


@lru_cache(maxsize=32)
def _sameline_filter_for(country: str) -> re.Pattern[str]:
    """Precursors that only mark a citation when adjacent on one line."""
    config = load_config(country) if country else None
    declared = config.structuring.sameline_precursors if config and config.structuring else []
    return _precursor_re(tuple(declared)) if declared else _AR_SAMELINE_FILTER_RE


# A window ending on a comma straight after a number or a closing paren is
# mid-enumeration: "Pasal 45 ayat (1),\nPasal 47" is one citation list, and no
# heading follows a line broken there.
_CITATION_LIST_TAIL_RE = re.compile(r"[)\d]\s*,\s*$")


def _probe_from(text: str, start: int) -> int:
    """Offset the prose filter reads a marker's previous line from.

    Its window walks back from the marker's own newline, so a match anchored
    after that newline hands it an empty window and every wrapped citation
    reads as a heading.
    """
    return start - 1 if start > 0 and text[start - 1] == "\n" else start


def _is_prose_reference(text: str, start: int, country: str = "") -> bool:
    """True when the match at ``start`` is preceded by an Arabic prose word, making it
    a cross-reference rather than an anchor.

    ``start`` sits on the marker's own newline, so the 30-char window back covers
    the previous line's tail, suppressing OCR-wrapped citations. It stops at any
    earlier newline.
    """
    window_start = max(0, start - 30)
    prev_nl = text.rfind("\n", window_start, start)
    if prev_nl >= 0:
        window_start = prev_nl + 1
    window = text[window_start:start]
    # The anchor match may start before a line break (its leading \s+
    # consumes it); count the newlines between ``start`` and the keyword.
    i = start
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    gap_newlines = text.count("\n", start, i)
    # A precursor separated from the marker by a blank line ended its own
    # sentence; only a same-line precursor or a single soft wrap is a
    # citation ("…اتفاق يقضي بغير ذلك\n\nمادة (812)" heads a real article).
    window = _fold_hamza(window)
    if gap_newlines <= 1 and _prose_filter_for(country).search(window):
        return True
    # Same-line-only precursors never look across the newline.
    if gap_newlines <= 1 and _CITATION_LIST_TAIL_RE.search(window):
        return True
    return gap_newlines == 0 and bool(_sameline_filter_for(country).search(window))


# Footnote brackets and amendment stars sit between a number and what follows it.
_NUMBER_DECORATIONS_RE = re.compile(r"^(?:[ \t]*(?:\*|\[[^\]\n]{1,6}\]))*[ \t]*")


@lru_cache(maxsize=64)
def _citation_successors_for(country: str) -> tuple[str, ...]:
    if not country:
        return ()
    config = load_config(country)
    return tuple(config.structuring.citation_successors if config.structuring else ())


@lru_cache(maxsize=64)
def _successor_re(country: str) -> re.Pattern[str] | None:
    """The declared successors as whole words, or None where none are declared."""
    successors = _citation_successors_for(country)
    if not successors:
        return None
    alts = "|".join(
        re.escape(w) + _word_end(w) for w in sorted(successors, key=lambda w: (-len(w), w))
    )
    return re.compile(rf"(?:{alts})")


def _opens_citation_run(text: str, end: int, country: str) -> bool:
    """Does a declared successor word follow the number on the marker's own line?"""
    pattern = _successor_re(country)
    if pattern is None:
        return False
    rest = text[end : text.find("\n", end) if text.find("\n", end) >= 0 else len(text)]
    rest = _NUMBER_DECORATIONS_RE.sub("", rest, count=1)
    return pattern.match(rest) is not None


_CYRILLIC_ROMAN = str.maketrans({"І": "I", "і": "i", "Х": "X", "х": "x"})

# Matches a captured num carrying a bis suffix: "6) مكرر", "9) مكرر (1".
_BIS_SPLIT_RE = re.compile(
    r"^(?P<base>.*?)\s*\)?\s*مكرر(?:ة)?\s*(?:\(?\s*(?P<idx>[0-9٠-٩۰-۹]+|[ء-ي]))?\s*\)?$"
)


def _split_bis(num: str) -> tuple[str, str] | None:
    """Split "6) مكرر (2" into ("6", "2"); None when there is no مكرر."""
    if "مكرر" not in num:
        return None
    m = _BIS_SPLIT_RE.match(num.strip())
    if m is None:
        # A مكرر shape the grammar doesn't cover would otherwise ship a
        # non-ASCII eid silently; surface the drift.
        logger.warning("bis_number_unmatched", num=num)
        return None
    # The index stays in its source script for display; eid derivation
    # normalises it separately.
    return m.group("base").strip(), (m.group("idx") or "").strip()


def _repair_damaged_num(match: re.Match[str]) -> str | None:
    """`_normalise_num`, first undoing the damage the matching branch admitted.

    Only a match that arrived through a declared tolerance is repaired, so
    `_normalise_number` stays country-agnostic and a clean `14T` keeps its suffix.
    """
    num = match.group("num")
    if num is None:
        return None
    groups = match.groupdict()
    if groups.get("numsplit") is not None:
        num = re.sub(r"[ \t]+", "", num)
    if groups.get("numglyph") is not None or groups.get("numsplit") is not None:
        num = num.translate(_DIGIT_GLYPHS)
    # The suffix is a word: repaired digits never reach it.
    suffix = groups.get("suffix")
    return _normalise_num(f"{num} {suffix.strip()}" if suffix else num)


def _normalise_num(num: str | None) -> str | None:
    """Collapse spaced hyphens ("13 - 1" → "13-1") and map Cyrillic Roman
    homoglyphs to Latin ("РОЗДІЛ І" → num "I") so eIds are ASCII and unique."""
    if num is None:
        return None
    bis = _split_bis(num)
    if bis is not None:
        base, idx = bis
        return f"{base} مكرر ({idx})" if idx else f"{base} مكرر"
    num = re.sub(r"\s*[-\u2013]\s*", "-", num)
    num = re.sub(r"\s*/\s*", "/", num)
    if all(c in "IVXLCDMivxlcdmІіХх" for c in num):
        num = num.translate(_CYRILLIC_ROMAN)
    return num


def _group_name(kind: str) -> str:
    return f"k_{kind.replace('-', '_').replace('.', '_')}"


def _stamp(anchors: list[StructuralAnchor], source_pass: str) -> list[StructuralAnchor]:
    """Name the pass that produced each anchor not already carrying a name.

    Applied per pass rather than per construction site, because passes rebuild
    the list they are handed; survivors keep the name they came in with.
    """
    return [a if a.source_pass else replace(a, source_pass=source_pass) for a in anchors]


@dataclass(frozen=True)
class AnchorScan:
    """Anchors, plus the readings the scan chose between to produce them."""

    anchors: list[StructuralAnchor]
    ambiguity: list[AmbiguitySpan]
    # Pass name -> effect in that pass's own unit (anchors added/removed,
    # rewritten, spans raised, eIds assigned). Not additive. Absent means the
    # pass did not run; 0 means it ran and changed nothing.
    fires: dict[str, int] = field(default_factory=dict)


def _fire(
    fires: dict[str, int], name: str, before: list[StructuralAnchor], after: list[StructuralAnchor]
) -> list[StructuralAnchor]:
    """Record what a pass did to the anchor list and hand the result on."""
    fires[name] = fires.get(name, 0) + len(after) - len(before)
    return after


def _fire_changed(
    fires: dict[str, int],
    name: str,
    before: list[StructuralAnchor],
    after: list[StructuralAnchor],
    attr: str,
) -> list[StructuralAnchor]:
    """Same, for a pass that rewrites a field rather than adding or removing.

    Counts anchors whose ``attr`` changed, matched on char offset. A drop in the
    same call is counted by `_fire`. Raises if two anchors share an offset.
    """
    was = {a.char_offset: getattr(a, attr) for a in before}
    if len(was) != len(before):
        raise AssertionError(f"{name}: anchors share a char offset, so the count is ambiguous")
    fires[name] = sum(
        1 for a in after if a.char_offset in was and getattr(a, attr) != was[a.char_offset]
    )
    return after


def _fire_dropped(
    fires: dict[str, int],
    name: str,
    before: list[StructuralAnchor],
    after: list[StructuralAnchor],
    spans: list[AmbiguitySpan],
    *,
    reads_as: str,
) -> list[StructuralAnchor]:
    """`_fire`, plus a resolved span for each anchor the pass removed.

    An undeclared drop leaves `_declare_orphaned_drops` charging the next visible
    drop with everything in between. The span is resolved, so it is advisory.
    """
    kept = {id(a) for a in after}
    for anchor in before:
        if id(anchor) in kept:
            continue
        spans.append(
            AmbiguitySpan(
                kind="orphan_text",
                start=anchor.char_offset,
                end=anchor.char_offset + len(anchor.matched_text),
                emitted_by=name,
                resolved=True,
                detail={"kind": anchor.kind, "number": anchor.number, "reads_as": reads_as},
            )
        )
    return _fire(fires, name, before, after)


def _declare_unclaimed_markers(
    text: str, anchors: list[StructuralAnchor], spans: list[AmbiguitySpan]
) -> None:
    """Marker-shaped lines no anchor claims.

    A keyword the regex never matched (`المادم`, from broken ToUnicode) never enters
    the scan loop, so the rejections declared there cannot see it.
    """
    # By offset, not line: the anchor regex consumes the leading newline, so an
    # anchor's recorded line can be one less than the marker's own.
    starts = sorted(a.char_offset for a in anchors)
    # A marker the closing boundary set aside is classified, not unclaimed.
    excluded = [(sp.start, sp.end) for sp in spans if sp.kind == "tail_excluded"]
    for m in _BARE_MARKER_LINE_RE.finditer(text):
        i = bisect_left(starts, m.start() - 2)
        if i < len(starts) and starts[i] <= m.end():
            continue
        if any(start <= m.start() < end for start, end in excluded):
            continue
        line = text.count("\n", 0, m.start()) + 1
        spans.append(
            AmbiguitySpan(
                kind="unmatched_marker",
                start=m.start(),
                end=m.end(),
                emitted_by="declare_unclaimed_markers",
                detail={
                    "reason": "marker_shaped_line_unclaimed",
                    "keyword": m.group("kw"),
                    "number": m.group("num"),
                    "line": line,
                },
            )
        )


def _basic_unit_for(country: str, doctype: str) -> str | None:
    """The doctype's basic unit. Deferred import: `structure` imports this."""
    from codify.pipeline.enrich.structure import basic_unit_kind

    if not country:
        return None
    return basic_unit_kind(load_config(country), doctype)


def _declare_out_of_order(
    members: list[StructuralAnchor], kind: str, parent: str, spans: list[AmbiguitySpan]
) -> None:
    """Siblings whose numbering does not ascend in document order.

    A gap check cannot see this: 1, 3, 2 sorts dense. `order_key` is the same
    comparison the corpus scan makes, so the two agree.
    """
    keyed = [(order_key(a.akn_eid), a) for a in members if a.akn_eid]
    seq = [(k, a) for k, a in keyed if k is not None]
    for (prev, _), (cur, anchor) in zip(seq, seq[1:], strict=False):
        if cur > prev:
            continue
        spans.append(
            AmbiguitySpan(
                kind="numbering_gap",
                start=anchor.char_offset,
                end=anchor.char_offset + len(anchor.matched_text),
                eid=anchor.akn_eid,
                emitted_by="declare_out_of_order",
                detail={
                    "kind": kind,
                    "parent": parent,
                    "reads_as": "out_of_order",
                    "number": anchor.number,
                },
            )
        )


def _declare_numbering_gaps(
    anchors: list[StructuralAnchor], spans: list[AmbiguitySpan], basic_kind: str | None
) -> None:
    """Holes in each container's numbering, classified but not closed."""
    groups: dict[tuple[str, str], list[StructuralAnchor]] = {}
    for a in anchors:
        if a.quoted_amendment:
            continue
        groups.setdefault((a.parent_eid or "", a.kind), []).append(a)
    # Also compared flat: per-parent alone reads a drop at a container boundary
    # as two dense runs. Skipped when the basic unit is already top-level, or the
    # same members would be counted twice.
    if basic_kind:
        flat = [a for a in anchors if a.kind == basic_kind and not a.quoted_amendment]
        if flat and groups.get(("", basic_kind)) != flat:
            groups[("", f"{basic_kind}:document")] = flat
    for (parent, kind), members in groups.items():
        numbered = [
            (int(n), a)
            for a in members
            if (n := _normalise_number(a.number)).isdigit() and n != "0"
        ]
        if len(numbered) < 2:
            continue
        _declare_out_of_order(members, kind, parent, spans)
        missing, total, max_run, lo, hi = missing_between([n for n, _ in numbered])
        if not total:
            continue
        first = numbered[0][1]
        spans.append(
            AmbiguitySpan(
                kind="numbering_gap",
                start=first.char_offset,
                end=numbered[-1][1].char_offset,
                eid=first.akn_eid,
                emitted_by="declare_numbering_gaps",
                detail={
                    "kind": kind,
                    "parent": parent,
                    "missing": missing[:20],
                    "missing_count": total,
                    "range": [lo, hi],
                    "reads_as": classify_gap(total, max_run, lo, hi),
                },
            )
        )


def _declare_toc_without_body(
    text: str, anchors: list[StructuralAnchor], spans: list[AmbiguitySpan], basic_kind: str | None
) -> None:
    """Cover-listed units no anchor claims. Keyed on the basic unit, not on
    `article`, or a `section`-based hierarchy reports every number absent."""
    if basic_kind is None:
        logger.info("toc_check_skipped", reason="no_basic_unit_for_doctype")
        return
    listed = extract_cover_article_numbers(text)
    if not listed:
        # Covers both "no listing" and "not recognised"; neither is asserted.
        logger.info("toc_check_skipped", reason="no_cover_listing")
        return
    present = {
        int(n)
        for a in anchors
        if a.kind == basic_kind
        and not a.quoted_amendment
        and (n := _normalise_number(a.number)).isdigit()
    }
    absent = sorted(n for n in set(listed) if n not in present)
    if not absent:
        return
    spans.append(
        AmbiguitySpan(
            kind="toc_without_body",
            start=0,
            end=min(len(text), COVER_TEXT_CHARS),
            emitted_by="declare_toc_without_body",
            detail={
                "listed": len(set(listed)),
                "absent": absent[:20],
                "absent_count": len(absent),
            },
        )
    )


def _declare_untwinned_tail(
    text: str,
    anchors: list[StructuralAnchor],
    spans: list[AmbiguitySpan],
    basic_kind: str | None,
    country: str = "",
) -> None:
    """Contents entries whose body is not there, which is what truncation leaves.

    A complete document pairs every listed unit: the contents line and the body
    heading share a number, so `_drop_toc_duplicates` drops one and names the other.
    Cut the tail off and the early units still pair while the rest survive as lone
    contents lines. The coverage gate cannot see that, because both its numbers come
    from the same text and shrink together.

    Nothing is read from duplicates alone. A duplicate proves a number was seen twice,
    not that a listing exists, and an annex restarting at 1 produces pair geometry
    identical to a truncated file: same count, same ordering, same reading. Only the
    text can separate them, so the cover recogniser has to find a listing first and
    this is silent without one. That recogniser is narrow, and its reach is the
    check's reach.

    An instrument adopting another body's text is exempt whatever its listing says. It
    carries two numbering series, so the adopted articles pair against the operative
    part's and the shortfall is the join rather than a cut file.
    """
    if basic_kind is None or extract_cover_article_numbers(text) is None:
        return
    if adopts_external_text(text, load_config(country) if country else None):
        # Recorded, not merely logged: a landed instrument otherwise reads as though
        # the truncation check had examined it and found nothing.
        spans.append(
            AmbiguitySpan(
                kind="adoption_suppressed",
                start=0,
                end=min(len(text), TITLE_BLOCK_CHARS),
                emitted_by="declare_untwinned_tail",
                resolved=True,
                detail={
                    "kind": basic_kind,
                    "reason": "the instrument declares it adopts another body's text",
                },
            )
        )
        return
    # `toc_twin` only: `_drop_toc_duplicates` separates `sequence_superseded` because
    # a restatement disproves the contents premise rather than evidencing it. Keyed on
    # the basic unit too, or a duplicate chapter 3 pairs section 3.
    pairs = [
        sp
        for sp in spans
        if sp.kind == "duplicate_number"
        and (sp.detail or {}).get("kind") == basic_kind
        and (sp.detail or {}).get("reads_as") == "toc_twin"
        and isinstance((sp.detail or {}).get("kept_at"), int)
    ]
    if not pairs:
        return
    # A pair names both halves: `kept_at` is the surviving body heading and the span's
    # own start the contents line that was dropped. Both are needed, since the dedup
    # has emptied the listing and only the pairs still say where it ran.
    body_starts = min(int(sp.detail["kept_at"]) for sp in pairs)
    listed_bodied = max(sp.start for sp in pairs)
    units = [a for a in anchors if a.kind == basic_kind and not a.quoted_amendment]
    # Before the body, so a contents line nothing bodied; and after the last listed
    # unit that was bodied, so the run reaches the end of the listing. One unbodied
    # entry among bodied ones is a gap in the listing, not a cut file.
    tail = [a for a in units if listed_bodied < a.char_offset < body_starts]
    if not tail or not units:
        return
    first = min(tail, key=lambda a: a.char_offset)
    spans.append(
        AmbiguitySpan(
            kind="untwinned_tail",
            start=first.char_offset,
            end=first.char_offset + len(first.matched_text),
            eid=first.akn_eid,
            emitted_by="declare_untwinned_tail",
            detail={
                "kind": basic_kind,
                "twinned": len(units) - len(tail),
                "present": len(units),
                "untwinned": [a.number for a in tail][:20],
                "untwinned_count": len(tail),
            },
        )
    )


def scan_anchors(
    text: str,
    regex: re.Pattern[str],
    *,
    country: str = "",
    doctype: str = "",
    toc_end: int = 0,
) -> list[StructuralAnchor]:
    """Walk text past toc_end and return anchors with depth + akn_eid filled."""
    return scan_anchors_with_ambiguity(
        text, regex, country=country, doctype=doctype, toc_end=toc_end
    ).anchors


_DECIMAL_CONTINUATION_RE = re.compile(r"\.\d")


def _partial_decimal_number(text: str, match: re.Match[str]) -> bool:
    return bool(
        match.groupdict().get("num") and _DECIMAL_CONTINUATION_RE.match(text, match.end("num"))
    )


def scan_anchors_with_ambiguity(
    text: str,
    regex: re.Pattern[str],
    *,
    country: str = "",
    doctype: str = "",
    toc_end: int = 0,
) -> AnchorScan:
    """`scan_anchors`, keeping the choices it made. Anchors are identical."""
    spans: list[AmbiguitySpan] = []
    fires: dict[str, int] = {}
    in_quote = _quote_mask(text, country)
    # UK: build the straight-quote mask before the first scan so keyworded
    # structures inside replacement blocks are suppressed by both passes.
    effective_quote = (
        _uk_amendment_mask(text, in_quote) if country in _UK_JURISDICTIONS else in_quote
    )
    raw: list[StructuralAnchor] = []
    for match in regex.finditer(text):
        if match.start() < toc_end:
            continue
        # A keyword match must not turn a dotted number into its integer prefix.
        if _partial_decimal_number(text, match):
            spans.append(
                AmbiguitySpan(
                    kind="unmatched_marker",
                    start=match.start(),
                    end=match.end(),
                    emitted_by="regex",
                    detail={"reason": "partial_decimal_number", "text": match.group(0).strip()},
                )
            )
            continue
        # The match can begin on leading whitespace/newlines; quote state
        # belongs to the marker's first real character (a mask that resets
        # at a blank line has already released it there).
        marker_start = match.start()
        while marker_start < match.end() and text[marker_start] in " \t\r\n":
            marker_start += 1
        if effective_quote[marker_start]:
            spans.append(
                AmbiguitySpan(
                    kind="unmatched_marker",
                    start=match.start(),
                    end=match.end(),
                    emitted_by="regex",
                    detail={"reason": "inside_quoted_text", "text": match.group(0).strip()},
                )
            )
            continue
        if _is_prose_reference(text, match.start(), country):
            spans.append(
                AmbiguitySpan(
                    kind="unmatched_marker",
                    start=match.start(),
                    end=match.end(),
                    emitted_by="regex",
                    detail={"reason": "read_as_prose_reference", "text": match.group(0).strip()},
                )
            )
            continue
        if _opens_citation_run(text, match.end(), country):
            spans.append(
                AmbiguitySpan(
                    kind="unmatched_marker",
                    start=match.start(),
                    end=match.end(),
                    emitted_by="regex",
                    detail={"reason": "read_as_citation_run", "text": match.group(0).strip()},
                )
            )
            continue
        kind = _kind_from_match(match)
        if kind is None:
            spans.append(
                AmbiguitySpan(
                    kind="unmatched_marker",
                    start=match.start(),
                    end=match.end(),
                    emitted_by="regex",
                    detail={"reason": "no_kind_resolved", "text": match.group(0).strip()},
                )
            )
            continue
        line = text.count("\n", 0, match.start()) + 1
        raw.append(
            StructuralAnchor(
                kind=kind,
                keyword=_keyword_from_match(match, kind),
                number=_repair_damaged_num(match) if "num" in match.groupdict() else None,
                char_offset=match.start(),
                line=line,
                matched_text=match.group(0).strip(),
            )
        )
    # Counted here, not after the UK block: that block extends `raw` in place,
    # so a later count credits the UK provision scan and both UK drops to the
    # regex.
    fires["regex"] = len(raw)
    declared = _declared_marker_entries(country, doctype)
    if declared:
        # The bracketed and caption forms run unmasked: a judgment quotes
        # constantly, latching the mask over most of its body, and a quoted
        # statute never writes those forms. The outline forms are masked.
        marked = _stamp(
            _scan_declared_markers(text, toc_end, declared, country)[0], "declared_markers"
        )
        fires["scan_declared_markers"] = len(marked)
        raw.extend(marked)
        raw.sort(key=lambda a: a.char_offset)
    marginal_kind = _marginal_note_kind(country, doctype)
    noted: list[StructuralAnchor] = []
    if marginal_kind:
        noted = _stamp(
            _scan_marginal_note_units(
                text,
                marginal_kind,
                _anchor_lines(text, raw),
                toc_end=toc_end,
                end=_first_attachment_offset(raw),
                in_quote=effective_quote,
            ),
            "marginal_note_units",
        )
        fires["scan_marginal_note_units"] = len(noted)
        raw = _fire_dropped(
            fires,
            "drop_references_in_marginal_bodies",
            raw,
            _drop_references_in_marginal_bodies(text, raw, noted, marginal_kind),
            spans,
            reads_as="prose_citation",
        )
        raw.extend(noted)
        raw.sort(key=lambda a: a.char_offset)
    if country in _UK_JURISDICTIONS:
        # Keyword-less UK sections/subsections; scan, merge in doc order, and
        # drop Arrangement-of-Sections contents entries. SI "(N)" subdivisions
        # are AKN paragraphs, not subsections (gb si hierarchy).
        sub_kind = "paragraph" if doctype == "si" else "subsection"
        raw = _fire_dropped(
            fires,
            "drop_uk_prose_containers",
            raw,
            _drop_uk_prose_containers(text, raw),
            spans,
            reads_as="prose_citation",
        )
        raw = _stamp(raw, "regex")
        uk = _stamp(_scan_uk_provisions(text, effective_quote, toc_end, sub_kind), "uk_provisions")
        if noted and marginal_kind:
            uk = _drop_references_in_marginal_bodies(text, uk, noted, marginal_kind)
        fires["scan_uk_provisions"] = len(uk)
        raw.extend(uk)
        raw.sort(key=lambda a: a.char_offset)
        raw = _fire_dropped(
            fires,
            "drop_uk_toc_sections",
            raw,
            _drop_uk_toc_sections(text, raw),
            spans,
            reads_as="contents_entry",
        )
    rank_map = _rank_map_for(country, doctype)
    # Order: amendment classification, then digit repair, then TOC dedup.
    # Classification keys off the original OCR numbers; repair must precede
    # dedup or a misread duplicate is collapsed with the genuine article and
    # the provision vanishes. Quoted-amendment anchors are excluded from repair.
    raw = _stamp(raw, "regex")
    raw = _fire_changed(
        fires,
        "mark_embedded_amendment_articles",
        raw,
        _mark_embedded_amendment_articles(text, raw, _trigger_phrases_for(country)),
        "quoted_amendment",
    )
    raw = _fire_changed(
        fires,
        "mark_amendments_under_roman_host",
        raw,
        _mark_amendments_under_roman_host(raw, country),
        "quoted_amendment",
    )
    # Before TOC dedup: it is the pass this corrects.
    raw = _fire_changed(
        fires,
        "mark_specimen_examples",
        raw,
        _mark_specimen_examples(text, raw, country),
        "quoted_amendment",
    )
    raw = _fire_changed(
        fires,
        "mark_amendment_items",
        raw,
        _mark_amendment_items(text, raw, country),
        "amendment_item",
    )
    raw = _fire_changed(
        fires, "repair_article_digit_ocr", raw, _repair_article_digit_ocr(raw), "number"
    )
    # Recovery passes precede TOC dedup so a recovered body marker's TOC
    # twin is collapsed like any other duplicate.
    raw = _fire(
        fires,
        "recover_misread_marker",
        raw,
        _stamp(
            _recover_misread_article_markers(
                text,
                raw,
                _aliases_for(country, doctype),
                in_quote=effective_quote,
                toc_end=toc_end,
            ),
            "recover_misread_marker",
        ),
    )
    annexes = _stamp(_scan_unnumbered_annexes(text, raw, toc_end, country), "unnumbered_annex")
    body_offsets = [a.char_offset for a in raw if a.kind != "schedule"]
    closing_floor = _closing_floor(text, country, min(body_offsets)) if body_offsets else None
    fires["unnumbered_annex"] = len(annexes)
    raw.extend(annexes)
    raw.sort(key=lambda a: a.char_offset)
    # Spans come from schedules that survive the embedded-caption test, run
    # here as a probe: a body table caption is not an annex, and scanning past
    # one would pull the rest of the document inside it.
    outlines = _stamp(
        _scan_attachment_outlines(
            text, _drop_embedded_schedule_captions(raw, closing_floor), toc_end, country
        ),
        "attachment_outline",
    )
    if outlines:
        fires["scan_attachment_outlines"] = len(outlines)
        raw.extend(outlines)
        raw.sort(key=lambda a: a.char_offset)
    # Before TOC dedup: an appended instrument restarts its numbers, and left in,
    # its twins would eject the body's own provisions.
    raw = _fire(
        fires, "exclude_closing_tail", raw, _exclude_closing_tail(text, raw, country, spans)
    )
    raw = _fire(fires, "drop_toc_duplicates", raw, _drop_toc_duplicates(raw, rank_map, spans))
    raw = _fire(
        fires,
        "drop_lone_compilation_container",
        raw,
        _drop_lone_compilation_container(raw, spans),
    )
    raw = _fire(
        fires,
        "drop_preamble_citation_articles",
        raw,
        _drop_preamble_citation_articles(raw, spans, country),
    )
    dropped_or_promoted = _drop_orphan_children(raw, rank_map, spans)
    _fire_changed(fires, "promote_orphan_children", raw, dropped_or_promoted, "kind")
    raw = _fire(fires, "drop_orphan_children", raw, dropped_or_promoted)
    raw = _fire_changed(
        fires,
        "capture_container_headings",
        raw,
        _capture_container_headings(text, raw, regex),
        "heading",
    )
    if not raw:
        # Early-20th-century regulations use a heading-then-number layout with no
        # explicit مادة keyword: "تعاريف\n١ - في هذا النظام". Try the
        # heading-pattern fallback, then the ordinal-list fallback for short
        # decrees (أولاً: … ثانياً: …), before the caller resorts to
        # the verbatim-single-section bucket.
        raw = _fire(
            fires,
            "numbered_heading_fallback",
            raw,
            _stamp(_scan_numbered_heading_fallback(text), "numbered_heading_fallback"),
        )
    if not raw:
        raw = _fire(
            fires,
            "ordinal_list_fallback",
            raw,
            _stamp(_scan_ordinal_list_fallback(text), "ordinal_list_fallback"),
        )
    raw = _fire_dropped(
        fires,
        "drop_embedded_schedule_captions",
        raw,
        _drop_embedded_schedule_captions(raw, closing_floor),
        spans,
        reads_as="table_caption",
    )
    # Every drop has run, so `raw` is the surviving set and the measure can ask
    # what each drop left behind. Declaring, not dropping: the count is spans.
    _declare_orphaned_drops(text, raw, spans)
    fires["declare_orphaned_drops"] = sum(
        1 for sp in spans if sp.emitted_by == "declare_orphaned_drops"
    )
    mark = len(spans)
    _declare_act_boundary_suspected(text, raw, spans, country)
    fires["declare_act_boundary_suspected"] = len(spans) - mark
    assigned = _assign_eids(raw, rank_map, spans, _abbrev_map_for(country, doctype))
    # eIds assigned, not collisions raised: this pass runs on every document and
    # a zero here would read as dead code in a table read for deletions.
    fires["assign_eids"] = sum(1 for a in assigned if a.akn_eid)
    basic_kind = _basic_unit_for(country, doctype)
    # Declaring passes touch no anchor, so their count is spans raised; 0 means
    # they found nothing. Counted by `emitted_by` because
    # `_declare_numbering_gaps` also calls `_declare_out_of_order`.
    mark = len(spans)
    _declare_numbering_gaps(assigned, spans, basic_kind)
    for name in ("declare_numbering_gaps", "declare_out_of_order"):
        fires[name] = sum(1 for sp in spans[mark:] if sp.emitted_by == name)
    mark = len(spans)
    _declare_toc_without_body(text, assigned, spans, basic_kind)
    fires["declare_toc_without_body"] = len(spans) - mark
    mark = len(spans)
    _declare_untwinned_tail(text, assigned, spans, basic_kind, country)
    fires["declare_untwinned_tail"] = len(spans) - mark
    mark = len(spans)
    _declare_unclaimed_markers(text, assigned, spans)
    fires["declare_unclaimed_markers"] = len(spans) - mark
    return AnchorScan(anchors=assigned, ambiguity=spans, fires=fires)


# Numberless annex headings: keyword + colon, or keyword alone with the heading
# on the next line. Caption-vs-annex stays with
# `_drop_embedded_schedule_captions`.
_UNNUMBERED_ANNEX_RE = re.compile(
    r"(?m)^[ \t]{0,8}(?:ال)?(?:ملحق|جدول)[ \t]*(?:(?P<colon>[:：])[ \t]*(?=\S)|$)"
)


@lru_cache(maxsize=32)
def _attachment_captions(country: str) -> tuple[tuple[str, bool], ...]:
    """Declared captions and whether each opens normative content."""
    from codify.jurisdictions import load_config

    config = load_config(country) if country else None
    if config is None:
        return ()
    return tuple((a.caption, a.normative) for a in config.attachments if a.caption)


# A prefix caption is a title line; longer is prose that happens to open with the word.
_PREFIX_CAPTION_MAX_CHARS = 80


def _exclude_closing_tail(
    text: str, anchors: list[StructuralAnchor], country: str, spans: list[AmbiguitySpan]
) -> list[StructuralAnchor]:
    """Drop body markers between the closing phrase and the first attachment;
    the span's text is not lost, the structurer keeps it as the conclusions."""
    body = [a for a in anchors if a.kind != "schedule"]
    if not body:
        return anchors
    start = _closing_floor(text, country, min(a.char_offset for a in body))
    if start is None:
        return anchors
    end = min(
        (a.char_offset for a in anchors if a.kind == "schedule" and a.char_offset >= start),
        default=len(text),
    )
    kept = [a for a in anchors if a.kind == "schedule" or not start <= a.char_offset < end]
    dropped = len(anchors) - len(kept)
    spans.append(
        AmbiguitySpan(
            kind="tail_excluded",
            start=start,
            end=end,
            emitted_by="exclude_closing_tail",
            resolved=True,
            detail={"anchors": dropped, "chars": end - start},
        )
    )
    return kept


def _closing_floor(text: str, country: str, after: int) -> int | None:
    """Offset of the jurisdiction's closing phrase past `after`, or None."""
    from codify.pipeline.enrich.closing import closing_offset, closing_phrases_for

    phrases = closing_phrases_for(country)
    if not phrases:
        return None
    return closing_offset(text, phrases, after=after, quoted=_closed_quote_mask(text, country))


def _closed_quote_mask(text: str, country: str = "") -> tuple[bool, ...]:
    """Per-char flag: inside a quotation something closes. A stray opener masks nothing here."""
    quoted, stray = _walk_quotes(text, country)
    return tuple(q and not s for q, s in zip(quoted, stray, strict=True))


@lru_cache(maxsize=32)
def _prefix_captions(country: str) -> tuple[str, ...]:
    """Captions declared as line openers rather than whole lines."""
    from codify.jurisdictions import load_config

    config = load_config(country) if country else None
    if config is None:
        return ()
    return tuple(a.caption for a in config.attachments if a.caption and a.prefix)


@lru_cache(maxsize=32)
def _annex_caption_re(country: str) -> re.Pattern[str]:
    """The annex pattern, extended by the jurisdiction's declared captions."""
    captions = [c for c, _ in _attachment_captions(country)]
    if not captions:
        return _UNNUMBERED_ANNEX_RE
    prefixes = set(_prefix_captions(country))
    alts = "|".join(re.escape(c) for c in captions if c not in prefixes)
    # A prefix caption takes the rest of its line as the heading.
    opener = "|".join(
        re.escape(c) + _word_end(c) for c in sorted(prefixes, key=lambda c: (-len(c), c))
    )
    prefixed = rf"|^[ \t]{{0,60}}(?P<prefixed>(?:{opener})[^\r\n]*)" if opener else ""
    # A declared caption tolerates a centred indent and its own numbering
    # ("LAMPIRAN I"); the inferred Arabic form keeps its tight margin and its
    # bare-keyword rule, so its behaviour does not move.
    return re.compile(
        rf"(?m)^(?:[ \t]{{0,8}}(?:ال)?(?:ملحق|جدول)"
        rf"|[ \t]{{0,60}}(?P<declared>{alts or '(?!)'})(?P<decnum>[ \t]+[IVXLCDM]+|[ \t]+\d+)?)"
        rf"[ \t]*(?:(?P<colon>[:：])[ \t]*(?=\S)|\r?$){prefixed}"
    )


def _scan_unnumbered_annexes(
    text: str, existing: list[StructuralAnchor], toc_end: int, country: str = ""
) -> list[StructuralAnchor]:
    taken = {a.char_offset for a in existing if a.kind == "schedule"}
    # Candidates must follow the first body anchor, or a colon-form TOC entry
    # escapes the continuity filter and owns the body as attachment.
    first_body = min((a.char_offset for a in existing if a.kind != "schedule"), default=None)
    if first_body is None:
        return []
    # Where the jurisdiction closes its body with a signature, a caption before
    # it titles a table in the body and a caption after it opens an attachment.
    floor = _closing_floor(text, country, first_body)
    next_no = max(
        (int(n) for a in existing if a.kind == "schedule" and (n := a.number) and n.isdigit()),
        default=0,
    )
    out: list[StructuralAnchor] = []
    for m in _annex_caption_re(country).finditer(text):
        if m.start() < toc_end or m.start() <= first_body or m.start() in taken:
            continue
        if floor is not None and m.start() < floor:
            continue
        prefixed = "prefixed" in m.re.groupindex and m.group("prefixed") is not None
        declared = prefixed or ("declared" in m.re.groupindex and m.group("declared") is not None)
        if m.group("colon") is None and not declared:
            # Bare-keyword form needs a blank line above: wrapped prose is a
            # continuation, an annex header never is.
            before = text[: m.start()].rstrip(" \t")
            if not (before.endswith("\n\n") or before in ("", "\n")):
                continue
        match_line_end = text.find("\n", m.end())
        if match_line_end == -1:
            match_line_end = len(text)
        if prefixed:
            heading = m.group("prefixed").strip()
            if len(heading) > _PREFIX_CAPTION_MAX_CHARS:
                continue
        elif declared:
            # The caption names the attachment. Taking the next line instead
            # titles a Penjelasan "ATAS", the first word of the heading below it.
            heading = (m.group("declared") + (m.group("decnum") or "")).strip()
        elif m.group("colon"):
            heading = text[m.end() : match_line_end].strip() or None
        else:
            # Bare-keyword form: the heading is the next non-empty line
            # (whitespace-only lines between keyword and heading skipped).
            heading = next(
                (line.strip() for line in text[match_line_end:].split("\n") if line.strip()),
                None,
            )
        # Heading-likeness: a definitions line ("الجدول: يقصد به الجدول
        # المرفق.") is prose, not an annex heading; promoting it would pull
        # the whole body inside an attachment. A declared prefix already vouched.
        if heading and not prefixed and (len(heading.split()) > 6 or re.search(r"[.؛،]", heading)):
            continue
        next_no += 1
        line = text.count("\n", 0, m.start()) + 1
        logger.info("unnumbered_annex_anchor", line=line, assigned_number=next_no)
        matched = m.group(0).strip()
        out.append(
            StructuralAnchor(
                kind="schedule",
                keyword=matched.strip(),
                number=str(next_no),
                char_offset=m.start(),
                line=line,
                matched_text=matched,
                heading=heading,
            )
        )
    return out


def _drop_embedded_schedule_captions(
    anchors: list[StructuralAnchor], floor: int | None = None
) -> list[StructuralAnchor]:
    """Drop schedule anchors that caption a table embedded in the body.

    A جدول/ملحق heading is a real annex or a caption for an embedded table, and
    promoting a caption pulls every later article into the attachment.
    Disambiguated by numbering continuity: a continuing sequence means caption
    (dropped, text stays in the body); a restart or nothing following means annex,
    and later anchors are its content.
    """
    ordered = sorted(anchors, key=lambda a: a.char_offset)

    def _num_of(a: StructuralAnchor) -> int | None:
        folded = normalise_digits(a.number or "").strip()
        return int(folded) if folded.isdigit() else None

    dropped: set[int] = set()
    for i, a in enumerate(ordered):
        if a.kind != "schedule" or (floor is not None and a.char_offset >= floor):
            continue
        prev_num = next(
            (
                n
                for b in reversed(ordered[:i])
                if b.kind != "schedule" and (n := _num_of(b)) is not None
            ),
            None,
        )
        next_num = next(
            (n for b in ordered[i + 1 :] if b.kind != "schedule" and (n := _num_of(b)) is not None),
            None,
        )
        if prev_num is not None and next_num is not None and next_num > prev_num:
            dropped.add(id(a))
            logger.info(
                "schedule_caption_dropped",
                number=a.number,
                prev_num=prev_num,
                next_num=next_num,
            )
    if not dropped:
        return anchors
    return [a for a in anchors if id(a) not in dropped]


def _repair_article_digit_ocr(anchors: list[StructuralAnchor]) -> list[StructuralAnchor]:
    """Post-detection pass: when the article sequence breaks its monotonic prior, look
    for a single-digit OCR substitution that fits. Non-article anchors are left
    alone.
    """
    schedule_starts = [i for i, a in enumerate(anchors) if a.kind == "schedule"]
    main_end = schedule_starts[0] if schedule_starts else len(anchors)
    if schedule_starts and main_end < len(anchors):
        logger.info(
            "digit_repair_stops_at_schedule",
            boundary_index=main_end,
            anchors_beyond=len(anchors) - main_end,
        )
    # Annex articles restart their numbering; repairing them against the main
    # body's monotonic prior would corrupt a legitimate restart.
    article_positions = [
        i
        for i, a in enumerate(anchors[:main_end])
        if a.kind == "article" and not a.quoted_amendment
    ]
    if len(article_positions) < 2:
        return anchors
    numbers = [anchors[i].number for i in article_positions]
    repaired_numbers, corrections = repair_number_sequence(numbers)
    if not corrections:
        return anchors
    out = list(anchors)
    for pos, new_num in zip(article_positions, repaired_numbers, strict=True):
        if new_num != anchors[pos].number:
            out[pos] = replace(anchors[pos], number=new_num)
    for corr in corrections:
        anchor_idx = article_positions[corr.index]
        logger.info(
            "structure_digit_repair",
            anchor_index=anchor_idx,
            char_offset=anchors[anchor_idx].char_offset,
            line=anchors[anchor_idx].line,
            from_num=corr.from_num,
            to_num=corr.to_num,
            reason=corr.reason,
        )
    return out


# A bare marker-shaped line: a single word plus a parenthesised number and
# nothing else, the shape of an article header whose keyword OCR mangled.
_BARE_MARKER_LINE_RE = re.compile(
    r"(?m)^[ \t]{0,8}(?P<kw>[^\s()]{2,12})[ \t]*\((?P<num>[0-9٠-٩۰-۹]+)\)[ \t]*$"
)
_MAX_MARKER_RECOVERY_GAP = 3

# How much of an alias OCR may drop before the remainder stops naming it. Two
# characters keeps `ماد` and `الماد`; more admits `Ch` for `Chapter` and `ال` for
# `الملحق`, which the builtin schedule aliases put in every jurisdiction.
_MAX_KEYWORD_TRUNCATION = 2


def _kind_for_mangled_keyword(keyword: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    """The single hierarchy kind a corrupted keyword can only be, or None.

    OCR drops a keyword's tail or swaps a letter, so a prefix of an alias counts and
    so does one same-length swap. Two candidates means refuse, or a mangled token
    picks a provision boundary at random.
    """
    folded = fold_arabic_for_match(keyword)
    if not folded:
        return None
    matched = {
        kind
        for kind, terms in aliases.items()
        for term in terms
        if (folded_term := fold_arabic_for_match(term))
        and (
            (
                folded_term.startswith(folded)
                and len(folded_term) - len(folded) <= _MAX_KEYWORD_TRUNCATION
            )
            or _one_substitution(folded, folded_term)
        )
    }
    return matched.pop() if len(matched) == 1 else None


def _one_substitution(a: str, b: str) -> bool:
    """True when `a` and `b` differ in exactly one character, same length.

    Substitution only: the truncation half is covered by the prefix test, and
    admitting an insertion would read `Party` as the container keyword `Part`.
    """
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b, strict=True) if x != y) == 1


def _recover_misread_article_markers(
    text: str,
    anchors: list[StructuralAnchor],
    aliases: dict[str, tuple[str, ...]] | None = None,
    *,
    in_quote: Sequence[bool] | None = None,
    toc_end: int = 0,
) -> list[StructuralAnchor]:
    """Reclaim marker lines whose keyword OCR corrupted past the alias set.

    Either evidence suffices: the number fills a short gap between two articles, or
    the keyword still names exactly one level. Both need `_BARE_MARKER_LINE_RE`, a
    line holding only a word and a parenthesised number.
    """
    taken_lines: set[int] = set()
    for a in anchors:
        o = a.char_offset
        while o < len(text) and text[o] in " \t\r\n":
            o += 1
        taken_lines.add(text.count("\n", 0, o) + 1)
    arts = [a for a in anchors if a.kind == "article" and not a.quoted_amendment]
    inserted: list[StructuralAnchor] = []
    for a, b in zip(arts, arts[1:]):
        na, nb = _normalise_number(a.number), _normalise_number(b.number)
        if not (na.isdigit() and nb.isdigit()):
            continue
        missing = list(range(int(na) + 1, int(nb)))
        if not missing or len(missing) > _MAX_MARKER_RECOVERY_GAP:
            continue
        want = missing.pop(0)
        for m in _BARE_MARKER_LINE_RE.finditer(text, a.char_offset, b.char_offset):
            folded = normalise_digits(m.group("num")).strip()
            line = text.count("\n", 0, m.start()) + 1
            if not folded.isdigit() or int(folded) != want or line in taken_lines:
                continue
            logger.info(
                "misread_article_marker_recovered",
                line=line,
                number=folded,
                keyword=m.group("kw"),
            )
            inserted.append(
                StructuralAnchor(
                    kind="article",
                    keyword=m.group("kw"),
                    number=m.group("num"),
                    char_offset=m.start(),
                    line=line,
                    matched_text=m.group(0).strip(),
                )
            )
            if not missing:
                break
            want = missing.pop(0)

    if aliases:
        claimed_lines = taken_lines | {a.line for a in inserted}
        for m in _BARE_MARKER_LINE_RE.finditer(text, toc_end):
            line = text.count("\n", 0, m.start()) + 1
            # The primary loop refuses quoted markers and everything before the
            # cover listing ends. Reclaiming those here would overrule it.
            if line in claimed_lines or (in_quote is not None and in_quote[m.start()]):
                continue
            kind = _kind_for_mangled_keyword(m.group("kw"), aliases)
            if kind is None:
                continue
            logger.info(
                "mangled_keyword_marker_recovered",
                line=line,
                kind=kind,
                number=m.group("num"),
                keyword=m.group("kw"),
            )
            claimed_lines.add(line)
            inserted.append(
                StructuralAnchor(
                    kind=kind,
                    keyword=m.group("kw"),
                    number=m.group("num"),
                    char_offset=m.start(),
                    line=line,
                    matched_text=m.group(0).strip(),
                )
            )

    if not inserted:
        return anchors
    out = anchors + inserted
    out.sort(key=lambda x: x.char_offset)
    return out


# Heading-pattern fallback: short Arabic line (1-4 words, no marker) followed
# by `[Arabic-digit] -` on the next line. Each hit becomes an article anchor.
# Conservative: requires ≥3 hits in the doc to fire, so a single bullet
# point won't get promoted to "Article 1".
_HEADING_PATTERN_RE = re.compile(
    r"(?m)^(?P<heading>[؀-ۿ\s]{2,40})\n\s*(?P<n>[٠-٩]+)\s*[-–—]\s*\S",
)
_MIN_HEADING_PATTERN_HITS = 3


def _scan_numbered_heading_fallback(text: str) -> list[StructuralAnchor]:
    """Detect "HEADING\n N - ..." and convert each to an article anchor.

    Used only when the primary regex finds zero anchors, and returns [] below
    ``_MIN_HEADING_PATTERN_HITS`` so single-clause decrees gain no spurious anchors.
    """
    matches = list(_HEADING_PATTERN_RE.finditer(text))
    if len(matches) < _MIN_HEADING_PATTERN_HITS:
        return []
    out: list[StructuralAnchor] = []
    for m in matches:
        num = m.group("n")
        # The anchor's char_offset points at the number, that's where the
        # body window starts. The keyword text is the heading above it.
        n_start = m.start("n")
        out.append(
            StructuralAnchor(
                kind="article",
                keyword=m.group("heading").strip(),
                number=num,
                char_offset=n_start,
                line=text.count("\n", 0, n_start) + 1,
                matched_text=text[m.start() : m.end()].strip(),
            )
        )
    return out


# A basic unit set marginal-note style: the note on its own line, the bare number
# (bold in a transcription) below it, the body on the next line or beside it.
_MARGINAL_NUMBER_RE = re.compile(
    r"(?m)^[ \t]{0,8}(?:\*\*)?(?P<num>\d{1,4}[A-Z]?)(?:\*\*)?\.(?:\*\*)?"
    r"(?:[ \t]+(?P<body>\S[^\r\n]*))?[ \t]*\r?$"
)
_MARGINAL_HEADING_MAX = 80
_MARGINAL_BODY_START_RE = re.compile(r"[(\"\u201c\u2018'A-Z]")
# A body beside the number is a sentence; a contents entry is a bare title.
_MARGINAL_BODY_MIN = 60
_SENTENCE_TAIL_RE = re.compile(r"[.;:\u2014\"\u201d]\s*$")
# An outline caption ("B. Dials") heads a list, not a provision.
_OUTLINE_CAPTION_RE = re.compile(r"^(?:[A-Za-z]|\d{1,3})[.)]\s")


def _marginal_note_kind(country: str, doctype: str) -> str | None:
    """The basic unit to scan for when the config says headings are marginal notes."""
    if not country:
        return None
    config = load_config(country)
    display = config.display
    if display is None or display.heading_type != "marginal_note":
        return None
    # Declared, not defaulted: the model's default is also marginal_note.
    if "heading_type" not in display.model_fields_set:
        return None
    return _basic_unit_for(country, doctype)


def _neighbour_line(text: str, offset: int, *, back: bool) -> tuple[int, str] | None:
    """The nearest non-blank line before (or after) the line at `offset`."""
    if back:
        end = _line_start(text, offset)
        while end > 0:
            start = text.rfind("\n", 0, end - 1) + 1
            line = text[start : end - 1]
            if line.strip():
                return start, line
            end = start
        return None
    start = text.find("\n", offset)
    while start != -1:
        start += 1
        stop = text.find("\n", start)
        line = text[start : stop if stop != -1 else len(text)]
        if line.strip():
            return start, line
        if stop == -1:
            break
        start = stop
    return None


def _reads_as_marginal_note(line: str) -> bool:
    """Short, sentence or title case, no terminal punctuation: a note, not prose."""
    bare = line.strip().strip("*_").strip()
    return (
        0 < len(bare) <= _MARGINAL_HEADING_MAX
        and not _OUTLINE_CAPTION_RE.match(bare)
        and bare[0].isupper()
        and any(c.islower() for c in bare)
        and bare[-1] not in ".;:,"
    )


def _opens_a_block(text: str, line_start: int) -> bool:
    """Whether the line at `line_start` follows a blank line or the text's start;
    a wrapped line of prose follows a line of prose."""
    if line_start == 0:
        return True
    prev_end = line_start - 1
    prev_start = text.rfind("\n", 0, prev_end) + 1
    return not text[prev_start:prev_end].strip()


def _scan_marginal_note_units(
    text: str,
    kind: str,
    taken_lines: AbstractSet[int],
    *,
    toc_end: int = 0,
    end: int | None = None,
    in_quote: Sequence[bool] | None = None,
) -> list[StructuralAnchor]:
    """Basic units whose number stands alone under a marginal note, before `end`;
    the guards below are each named by a test."""
    out: list[StructuralAnchor] = []
    last = 0
    for m in _MARGINAL_NUMBER_RE.finditer(text, toc_end, len(text) if end is None else end):
        if in_quote is not None and in_quote[m.start("num")]:
            continue
        number = int(m.group("num").rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        number_line = text.count("\n", 0, m.start()) + 1
        before = _neighbour_line(text, m.start(), back=True)
        if number < last or number_line in taken_lines or before is None:
            continue
        note_start, note = before
        note_line = text.count("\n", 0, note_start) + 1
        if note_line in taken_lines or not _reads_as_marginal_note(note):
            continue
        if not _opens_a_block(text, note_start):
            continue
        body = m.group("body")
        if body is not None and not (
            len(body) >= _MARGINAL_BODY_MIN or _SENTENCE_TAIL_RE.search(body)
        ):
            continue
        if body is None:
            after = _neighbour_line(text, m.start(), back=False)
            body = after[1] if after else ""
        if not _MARGINAL_BODY_START_RE.match(body.lstrip()):
            continue
        last = number
        heading = note.strip().strip("*_").strip()
        out.append(
            StructuralAnchor(
                kind=kind,
                keyword=heading,
                number=m.group("num"),
                char_offset=note_start,
                line=note_line,
                matched_text=note.strip(),
                heading=heading if kind in _HEADING_KINDS else None,
            )
        )
    return out


# A schedule anchors as the builtin kind or as the grouping hcontainer a config declares.
_ATTACHMENT_KINDS = frozenset({"schedule", "hcontainer"})


def _attachment_floor(markers: Iterable[tuple[int, str]]) -> int | None:
    """Where the schedules begin: the first schedule marker past the last
    container, so a contents list naming one does not floor the body."""
    markers = list(markers)
    last_container = max((o for o, k in markers if k in CONTAINER_KINDS), default=-1)
    return min(
        (o for o, k in markers if k in _ATTACHMENT_KINDS and o > last_container), default=None
    )


def _first_attachment_offset(anchors: Iterable[StructuralAnchor]) -> int | None:
    return _attachment_floor((a.char_offset, a.kind) for a in anchors)


def _first_attachment_marker(
    text: str, config: JurisdictionConfig, doctype: str, in_quote: Sequence[bool]
) -> int | None:
    """`_first_attachment_offset` for the denominator, which has no anchors yet."""
    pattern = _compile_anchor_regex(
        keyword_aliases(config, doctype), config, boundary="column_only"
    )
    markers = []
    for m in pattern.finditer(text):
        kind = _kind_from_match(m)
        if kind and not in_quote[_marker_line_start(text, m.start())]:
            markers.append((m.start(), kind))
    return _attachment_floor(markers)


def _anchor_lines(text: str, anchors: Iterable[StructuralAnchor]) -> set[int]:
    """Lines that open with an anchor's marker. A marker met mid-line leaves the
    line free: after a bare number it is the body's own cross-reference."""
    out: set[int] = set()
    for a in anchors:
        start = _marker_line_start(text, a.char_offset)
        marker = a.char_offset
        while marker < len(text) and text[marker] in " \t\r\n":
            marker += 1
        if not text[start:marker].strip():
            out.add(text.count("\n", 0, start) + 1)
    return out


def _drop_references_in_marginal_bodies(
    text: str, anchors: list[StructuralAnchor], noted: Sequence[StructuralAnchor], kind: str
) -> list[StructuralAnchor]:
    """A keyword match on a line a bare number opened is a cross-reference in
    that unit's first sentence, not a second unit."""
    number_lines: set[int] = set()
    for a in noted:
        below = _neighbour_line(text, a.char_offset, back=False)
        if below is not None:
            number_lines.add(text.count("\n", 0, below[0]) + 1)
    return [
        a
        for a in anchors
        if a.kind != kind
        or text.count("\n", 0, _marker_line_start(text, a.char_offset)) + 1 not in number_lines
    ]


# Arabic ordinal-list markers (`أولاً:`, `ثانياً:`, ...): common on
# short PA decrees whose provisions enumerate directly without an explicit
# مادة keyword. Order maps to integer 1..10. Requires colon or dash tail
# to avoid matching mid-sentence uses.
_AR_ORDINAL_LIST_MARKERS = ARABIC.ordinal_list_markers
_AR_ORDINAL_LIST_RE = re.compile(
    r"(?m)^\s*(?P<marker>"
    + "|".join(re.escape(w) for w, _ in _AR_ORDINAL_LIST_MARKERS)
    + r")\s*[:：\-–—]"
)
_MIN_ORDINAL_LIST_HITS = 3


def _scan_ordinal_list_fallback(text: str) -> list[StructuralAnchor]:
    """Detect PA `أولاً: … ثانياً: …` provisions; convert each to an
    article anchor. Runs only when the primary regex finds zero, and
    requires ``>= _MIN_ORDINAL_LIST_HITS`` hits so a stray "أولاً"
    somewhere in an ordinary decree can't get promoted."""
    marker_to_int = dict(_AR_ORDINAL_LIST_MARKERS)
    matches = list(_AR_ORDINAL_LIST_RE.finditer(text))
    if len(matches) < _MIN_ORDINAL_LIST_HITS:
        return []
    out: list[StructuralAnchor] = []
    for m in matches:
        marker = m.group("marker")
        out.append(
            StructuralAnchor(
                kind="article",
                keyword=marker,
                number=marker_to_int.get(marker, "0"),
                char_offset=m.start("marker"),
                line=text.count("\n", 0, m.start("marker")) + 1,
                matched_text=text[m.start() : m.end()].strip(),
            )
        )
    return out


# UK sections/subsections carry no keyword ("1 Overview", "(1)"), so the keyword
# regex misses them. Section = number + capitalised heading line, no trailing
# period and not a "N Month …" date; subsection = parenthesised number at line
# start (excludes "(a)" points and "(subject…" prose).
_UK_JURISDICTIONS = frozenset({"gb", "gb-eng", "gb-wls", "gb-sct", "gb-nir"})
_UK_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# Optional dot after num, SI regs print "2. Heading".
_UK_SECTION_RE = re.compile(
    rf"(?m)^(?P<num>\d+[A-Z]*)\.?[ \t]+(?!(?:{_UK_MONTHS})\b)"
    rf"(?P<heading>[A-Z][^\n]{{0,120}})(?<![.])$"
)
_UK_SUBSECTION_RE = re.compile(r"(?m)^\((?P<num>\d+[A-Z]*)\)")


_UK_PROSE_KINDS = frozenset({"part", "chapter", "section", "article"})
_UK_HEADING_MAX = 100


def _drop_uk_prose_containers(text: str, anchors: list[StructuralAnchor]) -> list[StructuralAnchor]:
    """UK structural headings are line-anchored, short, and unpunctuated; a
    keyword match mid-line ("under Part 2 of…", "means Regulation 2015/847")
    or on a sentence line ("Part 3 makes provision…;") is a citation, not
    structure."""
    out: list[StructuralAnchor] = []
    for a in anchors:
        if a.kind in _UK_PROSE_KINDS:
            pos = a.char_offset  # keyword matches include leading whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            line_start = text.rfind("\n", 0, pos) + 1
            line_end = text.find("\n", pos)
            line = text[line_start : line_end if line_end != -1 else len(text)].rstrip()
            if (
                text[line_start:pos].strip()
                or len(line) > _UK_HEADING_MAX
                or line.endswith((".", ";", ":", ",", ")"))
            ):
                continue
        out.append(a)
    return out


# Bracketed decimal paragraph marker at line start: "[1.1]", "[3.12]". Court
# judgments number their paragraphs this way and write no hierarchy keyword.
_BRACKETED_DECIMAL_RE = re.compile(r"(?m)^[^\S\n]{0,8}\[(?P<num>\d+(?:\.\d+)+)\]")

# Outline markers. An annex numbers its own content without a keyword, so each
# form is the whole marker: glyph class, terminator, then required space.
_OUTLINE_PASS = "attachment_outline"  # noqa: S105, pass name, not a credential

_OUTLINE_RES: dict[str, re.Pattern[str]] = {
    "upper_letter_period": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>[A-Z])(?:\*\*)?\.(?:\*\*)?[^\S\n]+(?=\S)"
    ),
    "lower_letter_period": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>[a-z])(?:\*\*)?\.(?:\*\*)?[^\S\n]+(?=\S)"
    ),
    "arabic_period": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>\d{1,3})(?:\*\*)?\.(?:\*\*)?[^\S\n]+(?=\S)"
    ),
    "arabic_closing_paren": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>\d{1,3})(?:\*\*)?\)(?:\*\*)?[^\S\n]+(?=\S)"
    ),
    "lower_letter_closing_paren": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>[a-z])(?:\*\*)?\)(?:\*\*)?[^\S\n]+(?=\S)"
    ),
    "parenthesized_arabic": re.compile(
        r"(?m)^[^\S\n]{0,12}(?:\*\*)?\((?:\*\*)?(?P<num>\d{1,3})(?:\*\*)?\)(?:\*\*)?[^\S\n]+(?=\S)"
    ),
}


def _caption_re(captions: Sequence[str]) -> re.Pattern[str]:
    """Match any declared caption as a whole heading line.

    The leading numeral is optional and captured: a judgment prints
    "3. PERTIMBANGAN HUKUM" over paragraphs "[3.1]", so it is the real number."""
    # Longest-first so a caption that prefixes another cannot shadow it; ties
    # break by value so the compiled pattern is stable across processes.
    ordered = sorted({c.strip() for c in captions if c.strip()}, key=lambda c: (-len(c), c))
    alts = "|".join(re.escape(c) for c in ordered)
    return re.compile(
        rf"(?m)^[^\S\n]{{0,8}}(?:(?P<secnum>\d+)[^\S\n]*[.)][^\S\n]*)?(?P<caption>{alts})[^\S\n]*$"
    )


def _declared_marker_entries(country: str, doctype: str) -> tuple[HierarchyEntry, ...]:
    """Hierarchy levels this document class marks without a keyword."""
    if not country:
        return ()
    config = load_config(country)
    doc_class = config.get_document_class(doctype)
    if not doc_class:
        return ()
    return tuple(e for e in doc_class.hierarchy if e.marker_form is not None)


def _decimal_rank(number: str) -> tuple[int, ...]:
    """A bracketed decimal as a comparable tuple: "3.11.1" -> (3, 11, 1)."""
    try:
        return tuple(int(part) for part in number.split("."))
    except ValueError:
        return ()


def _advancing_decimal_matches(
    matches: Sequence[re.Match[str]], *, number_group: str = "num"
) -> list[re.Match[str]]:
    """Keep the longest advancing sequence, retaining the first equal-number match."""
    tails: list[tuple[int, ...]] = []
    ends: list[int] = []
    previous = [-1] * len(matches)
    for index, match in enumerate(matches):
        rank = _decimal_rank(match.group(number_group))
        position = bisect_left(tails, rank)
        if position < len(tails) and tails[position] == rank:
            continue
        previous[index] = ends[position - 1] if position else -1
        if position == len(tails):
            tails.append(rank)
            ends.append(index)
        else:
            tails[position] = rank
            ends[position] = index
    selected: list[re.Match[str]] = []
    index = ends[-1] if ends else -1
    while index >= 0:
        selected.append(matches[index])
        index = previous[index]
    return list(reversed(selected))


def _host_caption_sequence(
    text: str, matches: Sequence[re.Match[str]], country: str = ""
) -> list[re.Match[str]]:
    """Prefer resumed host text, then the earliest maximal advancing caption sequence."""
    if not matches:
        return []
    first_paragraph = next(
        (
            m
            for m in _BRACKETED_DECIMAL_RE.finditer(text)
            if _decimal_rank(m.group("num"))[0] == int(matches[0].group("secnum"))
            and not _is_prose_reference(text, _probe_from(text, m.start()), country)
        ),
        None,
    )
    keys = [(m.group("secnum"), m.group("caption")) for m in matches]
    if first_paragraph is not None:
        for count in range(2, len(matches) // 2 + 1):
            if (
                matches[count].start() < first_paragraph.start() < matches[count + 1].start()
                and _decimal_rank(first_paragraph.group("num"))[0] == int(keys[count][0])
                and keys[:count] == keys[count : 2 * count]
                and all(int(a[0]) < int(b[0]) for a, b in zip(keys[: count - 1], keys[1:count]))
            ):
                matches = matches[count:]
                break
    candidates: list[re.Match[str]] = []
    for index in range(len(matches) - 1, -1, -1):
        match = matches[index]
        number = int(match.group("secnum"))
        boundary = next((m for m in reversed(candidates) if int(m.group("secnum")) <= number), None)
        host = next((m for m in reversed(matches[:index]) if int(m.group("secnum")) < number), None)
        if boundary is not None and host is not None:
            prefix = int(host.group("secnum"))
            preceding = [
                _decimal_rank(m.group("num"))
                for m in _BRACKETED_DECIMAL_RE.finditer(text, host.end(), match.start())
                if _decimal_rank(m.group("num"))[0] == prefix
                and not _is_prose_reference(text, _probe_from(text, m.start()), country)
            ]
            earliest = min(preceding, default=(prefix, 0))
            if any(
                _decimal_rank(m.group("num"))[0] == prefix
                and _decimal_rank(m.group("num")) > earliest
                and _decimal_rank(m.group("num")) not in preceding
                and not _is_prose_reference(text, _probe_from(text, m.start()), country)
                for m in _BRACKETED_DECIMAL_RE.finditer(text, match.end(), boundary.start())
            ):
                # A resumed host interval also excludes intervening forward captions.
                candidates = [m for m in candidates if m.start() >= boundary.start()]
                continue
        candidates.append(match)
    candidates.reverse()
    tails: list[int] = []
    lengths = [0] * len(candidates)
    for index in range(len(candidates) - 1, -1, -1):
        rank = -int(candidates[index].group("secnum"))
        position = bisect_left(tails, rank)
        lengths[index] = position + 1
        if position == len(tails):
            tails.append(rank)
        else:
            tails[position] = rank
    remaining = len(tails)
    previous = -1
    selected = []
    for match, length in zip(candidates, lengths, strict=True):
        number = int(match.group("secnum"))
        if number > previous and length >= remaining:
            selected.append(match)
            previous = number
            remaining -= 1
            if not remaining:
                break
    return selected


def _scan_declared_markers(
    text: str,
    toc_end: int,
    entries: Sequence[HierarchyEntry],
    country: str = "",
) -> tuple[list[StructuralAnchor], list[int]]:
    """Anchors for levels marked without a keyword, and where a quote hid one.

    Those offsets are the caller's only sight of a mask that took markers off
    both sides of its ratio, which then reads 1.0 however many it took. Only
    the outline forms are masked; an undeclared class is untouched.
    """
    out: list[StructuralAnchor] = []
    masked: list[int] = []
    mask: Sequence[bool] | None = None
    caption_matches = sorted(
        {
            m.start(): m
            for entry in entries
            if entry.marker_form == "caption"
            for m in _caption_re(entry.captions).finditer(text)
            if m.start() >= toc_end and m.group("secnum")
        }.values(),
        key=lambda m: m.start(),
    )
    selected_captions = _host_caption_sequence(text, caption_matches, country)
    selected_caption_offsets = {m.start() for m in selected_captions}
    numbered_captions = {m.group("caption") for m in selected_captions}
    caption_starts = {int(m.group("secnum")): m.start() for m in selected_captions}
    caption_boundaries = [(m.start(), int(m.group("secnum"))) for m in selected_captions]
    caption_offsets = [offset for offset, _number in caption_boundaries]
    for entry in entries:
        if entry.marker_form == "caption":
            emitted = 0
            for m in _caption_re(entry.captions).finditer(text):
                if (
                    m.start() < toc_end
                    or (m.group("secnum") and m.start() not in selected_caption_offsets)
                    or (not m.group("secnum") and m.group("caption") in numbered_captions)
                ):
                    continue
                emitted += 1
                out.append(
                    StructuralAnchor(
                        kind=entry.akn_element,
                        keyword=m.group("caption"),
                        # Fall back to the emitted position when the source
                        # prints no numeral; an unnumbered section has no eId.
                        number=m.group("secnum") or str(emitted),
                        char_offset=m.start(),
                        line=text.count("\n", 0, m.start()) + 1,
                        matched_text=m.group(0).strip(),
                        # Only heading-bearing kinds accept one; the caption is
                        # kept in `keyword` either way.
                        heading=m.group("caption") if entry.akn_element in _HEADING_KINDS else None,
                    )
                )
        elif entry.marker_form in _OUTLINE_RES:
            # Masked, unlike the bracketed form below: a quoted subdivision is
            # written exactly as a real one and would become host structure.
            if mask is None:
                mask = _quote_mask(text, country)
            stray = _stray_quote_mask(text, country)
            # Under the scan's own prose filter: a wrapped citation inside a
            # stray span was never going to anchor, so it is not a marker the
            # span cost anyone.
            masked.extend(
                m.start()
                for m in _OUTLINE_RES[entry.marker_form].finditer(text)
                if stray[m.start()]
                and m.start() >= toc_end
                and not _is_prose_reference(text, _probe_from(text, m.start()), country)
            )
            for m in _OUTLINE_RES[entry.marker_form].finditer(text):
                if (
                    m.start() < toc_end
                    or mask[m.start()]
                    or _is_prose_reference(text, _probe_from(text, m.start()), country)
                ):
                    continue
                out.append(
                    StructuralAnchor(
                        kind=entry.akn_element,
                        keyword=m.group(0).strip(),
                        number=m.group("num"),
                        char_offset=m.start(),
                        line=text.count("\n", 0, m.start()) + 1,
                        matched_text=m.group(0).strip(),
                    )
                )
        elif entry.marker_form == "bracketed_decimal":
            # Forward citations must not suppress the following host sequence.
            intervals: dict[int, list[re.Match[str]]] = {}
            for m in _BRACKETED_DECIMAL_RE.finditer(text):
                if m.start() < toc_end or _is_prose_reference(
                    text, _probe_from(text, m.start()), country
                ):
                    continue
                rank = _decimal_rank(m.group("num"))
                # A forward citation cannot start the later caption's sequence.
                if rank and m.start() < caption_starts.get(rank[0], 0):
                    continue
                active_caption = bisect_left(caption_offsets, m.start()) - 1
                # Quoted earlier prefixes cannot rewind an explicit numbered caption.
                if rank and active_caption >= 0 and rank[0] < caption_boundaries[active_caption][1]:
                    continue
                # Only the final caption may extend into uncaptioned separate opinions.
                if (
                    rank
                    and 0 <= active_caption < len(caption_boundaries) - 1
                    and rank[0] != caption_boundaries[active_caption][1]
                ):
                    continue
                intervals.setdefault(active_caption, []).append(m)
            selected: list[re.Match[str]] = []
            for interval, candidates in intervals.items():
                if interval >= 0:
                    prefix = caption_boundaries[interval][1]
                    host = _advancing_decimal_matches(
                        [m for m in candidates if _decimal_rank(m.group("num"))[0] == prefix]
                    )
                    selected.extend(host)
                    if interval == len(caption_boundaries) - 1:
                        host_end = host[-1].start() if host else caption_boundaries[interval][0]
                        selected.extend(
                            _advancing_decimal_matches(
                                [
                                    m
                                    for m in candidates
                                    if m.start() > host_end
                                    and _decimal_rank(m.group("num"))[0] > prefix
                                ]
                            )
                        )
                    continue
                # Before any numbered caption, retain the established monotonic policy.
                last_rank: tuple[int, ...] = ()
                for candidate in candidates:
                    rank = _decimal_rank(candidate.group("num"))
                    if rank > last_rank:
                        selected.append(candidate)
                        last_rank = rank
            for m in selected:
                out.append(
                    StructuralAnchor(
                        kind=entry.akn_element,
                        keyword=m.group(0).strip(),
                        number=m.group("num"),
                        char_offset=m.start(),
                        line=text.count("\n", 0, m.start()) + 1,
                        matched_text=m.group(0).strip(),
                    )
                )
    return out, masked


def _attachment_hierarchies(country: str) -> tuple[tuple[str, tuple[HierarchyEntry, ...]], ...]:
    """(caption, levels) for every attachment kind declaring its own levels."""
    if not country:
        return ()
    config = load_config(country)
    return tuple(
        (a.caption.strip().upper(), tuple(a.hierarchy)) for a in config.attachments if a.hierarchy
    )


def _scan_attachment_keywords(
    window: str,
    toc_end: int,
    levels: Sequence[HierarchyEntry],
    config: JurisdictionConfig | None,
    already: Sequence[StructuralAnchor],
) -> list[StructuralAnchor]:
    """Anchors for attachment levels named by a keyword rather than a marker.

    Roman levels retain keyword aliases; other marker forms use their own scanner."""
    aliases: dict[str, Iterable[str]] = {
        str(e.akn_element): terms
        for e in levels
        if e.marker_form in (None, "upper_roman_period") and (terms := _alias_terms_for(e))
    }
    if not aliases:
        return []
    taken = {a.char_offset for a in already}
    out: list[StructuralAnchor] = []
    country = config.code if config else ""
    for m in _compile_anchor_regex(aliases, config).finditer(window):
        kind = _kind_from_match(m)
        if kind is None or m.start() < toc_end or m.start() in taken:
            continue
        if _opens_citation_run(window, m.end(), country):
            continue
        out.append(
            StructuralAnchor(
                kind=kind,
                keyword=_keyword_from_match(m, kind),
                number=_repair_damaged_num(m) if "num" in m.groupdict() else None,
                char_offset=m.start(),
                line=window.count("\n", 0, m.start()) + 1,
                matched_text=m.group(0).strip(),
            )
        )
    return out


def _attachment_roman_markers(
    text: str,
    levels: Sequence[HierarchyEntry],
    toc_end: int,
    country: str,
    known: Sequence[StructuralAnchor],
) -> list[StructuralAnchor]:
    """Require an I→II sequence; a preceding H keeps I in its letter outline."""
    entry = next((e for e in levels if e.marker_form == "upper_roman_period"), None)
    if entry is None:
        return []
    markers = list(
        re.finditer(
            r"(?m)^[^\S\n]{0,12}(?:\*\*)?(?P<num>[A-Z]+)(?:\*\*)?\."
            r"(?:\*\*)?[^\S\n]+(?=\S)",
            text,
        )
    )
    mask = _quote_mask(text, country)
    candidates = [m for m in markers if m.start() >= toc_end and not mask[m.start()]]
    barriers = [a.char_offset for a in known if a.kind == entry.akn_element]
    selected: dict[int, re.Match[str]] = {}
    for i, first in enumerate(candidates):
        if first.group("num") != "I" or (i and candidates[i - 1].group("num") == "H"):
            continue
        chain = [first]
        expected = 2
        for candidate in candidates[i + 1 :]:
            if any(first.start() < boundary < candidate.start() for boundary in barriers):
                break
            numeral = candidate.group("num")
            if numeral == "I":
                break
            if _roman_or_digit(numeral) == expected:
                chain.append(candidate)
                expected += 1
        if len(chain) >= 2:
            selected.update((m.start(), m) for m in chain)
    return [
        StructuralAnchor(
            kind=entry.akn_element,
            keyword=m.group(0).strip(),
            number=m.group("num"),
            char_offset=m.start(),
            line=text.count("\n", 0, m.start()) + 1,
            matched_text=m.group(0).strip(),
            source_pass=_OUTLINE_PASS,
        )
        for m in selected.values()
    ]


def _scan_attachment_outlines(
    text: str,
    anchors: Sequence[StructuralAnchor],
    toc_end: int,
    country: str,
) -> list[StructuralAnchor]:
    """Outline anchors for content inside an attachment.

    An annex numbers itself as a lettered outline, so its levels are scanned per
    attachment span. Confined there because "1." is ordinary list text in the body.
    """
    declared = _attachment_hierarchies(country)
    if not declared:
        return []
    config = load_config(country)
    starts = sorted(a.char_offset for a in anchors if a.kind == "schedule")
    if not starts:
        return []
    bounds = list(zip(starts, starts[1:] + [len(text)], strict=True))
    out: list[StructuralAnchor] = []
    for start, end in bounds:
        # The opening line only: a header block naming another attachment
        # would otherwise lend it that attachment's grammar.
        line_end = text.find("\n", start)
        head = text[start : (end if line_end == -1 else min(line_end, end))].upper().strip()
        levels = next((lv for cap, lv in declared if head.startswith(cap)), None)
        if levels is None:
            continue
        window = text[start:end]
        local_toc = max(0, toc_end - start)
        found, _ = _scan_declared_markers(window, local_toc, levels, country)
        seen = [a for a in anchors if start <= a.char_offset < end]
        local_seen = [replace(a, char_offset=a.char_offset - start) for a in seen]
        romans = _attachment_roman_markers(window, levels, local_toc, country, local_seen)
        roman_offsets = {a.char_offset for a in romans}
        found = [a for a in found if a.char_offset not in roman_offsets] + romans
        # Keyworded annex levels ("BAB I") reach nothing if the body class does
        # not happen to declare the same element, so scan them over the span too.
        # Offsets the body regex already claimed here are excluded, or a class
        # that does declare the element would anchor it twice.
        found.extend(
            _scan_attachment_keywords(
                window,
                local_toc,
                levels,
                config,
                found + local_seen,
            )
        )
        for a in found:
            out.append(
                replace(
                    a,
                    char_offset=a.char_offset + start,
                    line=text.count("\n", 0, a.char_offset + start) + 1,
                )
            )
    return out


# UK amendment blocks open with a straight double quote on its own line after an
# em-dash terminator. legislation.gov.uk PDFs keep straight quotes here, so
# `_quote_mask` misses them. The opening " must start a line following a U+2014
# terminator, within a few blank lines; inline quotes are always mid-line.
_UK_AMENDMENT_OPEN_RE = re.compile(
    r"—[ \t]*\n(?:[ \t]*\n){0,3}[ \t]*\"",
    re.MULTILINE,
)
# Closing: optional terminal punctuation followed by straight " at end of line.
# Requiring the quote to *follow* punctuation (rather than precede it) prevents
# matching the inner closing quote of an inline replacement such as
# `(1) for "ten" substitute "twenty".` where the quote is mid-sentence.
_UK_AMENDMENT_CLOSE_RE = re.compile(r'[.;,\)]*"[ \t]*$', re.MULTILINE)


def _uk_amendment_mask(text: str, base: Sequence[bool]) -> list[bool]:
    """Extend *base* to also mask straight-quote amendment blocks in UK Acts.

    Returns a new mask; *base* is not mutated.  Positions already masked by
    *base* (curly/guillemet quotes) are preserved.
    """
    mask = list(base)
    opens = list(_UK_AMENDMENT_OPEN_RE.finditer(text))
    for index, open_m in enumerate(opens):
        # The open match ends just after the opening "; mask from there.
        start = open_m.end()
        # A block may not borrow the next block's closing quote. When OCR drops
        # one, an unbounded search finds the following block's and masks every
        # genuine provision between them. Same resolution as `_unclosed_openers`
        # takes for curly quotes: an opener nothing closes is skipped, because
        # under-masking shows up as a duplicate_number and over-masking is
        # silent.
        limit = opens[index + 1].start() if index + 1 < len(opens) else len(text)
        close_m = _UK_AMENDMENT_CLOSE_RE.search(text, start)
        if close_m is None or close_m.end() > limit:
            continue
        for i in range(start, min(close_m.end(), len(mask))):
            mask[i] = True
    return mask


def _scan_uk_provisions(
    text: str, in_quote: Sequence[bool], toc_end: int, sub_kind: str = "subsection"
) -> list[StructuralAnchor]:
    out: list[StructuralAnchor] = []
    for kind, regex in (("section", _UK_SECTION_RE), (sub_kind, _UK_SUBSECTION_RE)):
        for m in regex.finditer(text):
            if m.start() < toc_end or in_quote[m.start()]:
                continue
            num = m.group("num")
            # Year-led prose lines ("2015 The Money…"): section numbers top
            # out ~1300, cited years start ~1800, the ranges don't overlap.
            if kind == "section" and num.isdigit() and 1800 <= int(num) <= 2099:
                continue
            out.append(
                StructuralAnchor(
                    kind=kind,
                    keyword=(m.groupdict().get("heading") or f"({m.group('num')})").strip(),
                    number=m.group("num"),
                    char_offset=m.start(),
                    line=text.count("\n", 0, m.start()) + 1,
                    matched_text=m.group(0).strip(),
                )
            )
    return out


def _drop_uk_toc_sections(text: str, anchors: list[StructuralAnchor]) -> list[StructuralAnchor]:
    """Drop Arrangement-of-Sections TOC entries: a real section is followed by
    its subsections or body before the next section/container; a contents entry
    is immediately followed by the next heading with nothing between."""
    from codify.pipeline.enrich.kinds import KIND_RANK

    sec_rank = KIND_RANK.get("section", 6)
    out: list[StructuralAnchor] = []
    for i, a in enumerate(anchors):
        if a.kind != "section":
            out.append(a)
            continue
        end = len(text)
        for b in anchors[i + 1 :]:
            if KIND_RANK.get(b.kind, 8) <= sec_rank:
                end = b.char_offset
                break
        after_heading = text[a.char_offset : end].split("\n", 1)
        rest = after_heading[1] if len(after_heading) > 1 else ""
        # A genuine section has subsections or ANY body line before the next
        # anchor; a contents entry is followed by the next heading with
        # nothing between. (Short single-sentence sections are real, Finance
        # Act charging sections are one line.)
        if re.search(r"^\(\d", rest, re.MULTILINE) or rest.strip():
            out.append(a)
    return out


_QUOTE_OPEN = "«„"
_QUOTE_CLOSE = "»"
_QUOTE_TOGGLE = "“”"


# What may close a quoted span, by the glyph that opened it. A left curly opens
# wherever it appears, so meeting one inside its own span says the earlier one
# never closed; the others are ambiguous and close whichever kind is open.
_CLOSERS_FOR_OPENER = {
    "\u201c": frozenset({"\u201d"}),
    "\u201d": frozenset({"\u201d", "\u201c"}),
    "\u201e": frozenset({"\u201c", "\u201d"}),
    "\u00ab": frozenset({"\u00bb"}),
}


def _opens_reversed(text: str, at: int, boundary: re.Pattern[str] | None) -> bool:
    """Is this right curly an opener, in a document that otherwise opens left?

    Reversed typography closes with a left curly, so the next quote glyph after
    an opener of this kind is that one. A dropped closer looks the same whenever
    an ordinary quotation follows it later, so the span must also close before
    the next provision: an amendment quotes one article, and a span that runs
    over a heading is a dropped closer reaching a quotation further down.
    """
    nxt = min(
        (j for j in (text.find(q, at + 1) for q in _QUOTE_TOGGLE) if j != -1),
        default=-1,
    )
    if nxt == -1 or text[nxt] != "\u201c":
        return False
    return boundary is None or not boundary.search(text, at, nxt)


# The curly quotes a Latin-1 read of cp1252 leaves as C1 codepoints, which no
# legible document uses. Length-preserving, so every mask offset still holds.
_MOJIBAKE_QUOTES = str.maketrans({0x93: "“", 0x94: "”"})


@lru_cache(maxsize=1)
def _walk_quotes(raw: str, country: str = "") -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """The quoted mask, and the part of it inside spans nothing closes.

    A marker inside a closed quote is quoted and leaves both sides of a
    coverage ratio by design; one inside a span nothing closes leaves them by
    accident, and only the second is worth reporting. Straight quotes are
    deliberately unread, being ordinary punctuation here.

    Limit: a dropped opener immediately before an amendment's own quote reads
    as that quote opening, so the text between them is not reported.
    """
    text = raw.translate(_MOJIBAKE_QUOTES)
    # A document that has closed a left curly with a right one has shown its
    # orientation, and a later right curly there is a dropped opener's closer
    # rather than an opener of its own. One amendment inside it may still be
    # reversed, so the decision is per occurrence and not per document.
    openers = dict(_CLOSERS_FOR_OPENER)
    ltr = re.search("\u201c[^\u201c]*\u201d", text) is not None
    boundary = _basic_unit_line_re(country)
    # A directional opener nothing closes is skipped outright, as it always
    # was: left in, it masks to end of file, and it cannot be bounded the way
    # an ambiguous glyph can.
    skip = _unclosed_openers(text)
    spans: list[tuple[int, int, bool]] = []  # start, end, closed
    open_at, open_by = -1, ""
    for i, ch in enumerate(text):
        # A blank line ends a span nothing has closed yet, bounding its damage.
        # A closed quote may span as many blocks as it likes, and is not ended
        # here because its closer is still to come.
        if ch == "\n" and open_by:
            j = i + 1
            while j < len(text) and text[j] in " \t\r":
                j += 1
            if text.startswith("\n", j) and not _closes_later(text, i, open_by, boundary):
                spans.append((open_at, i, False))
                open_at, open_by = -1, ""
        if open_by and ch in _CLOSERS_FOR_OPENER[open_by]:
            spans.append((open_at, i, True))
            open_at, open_by = -1, ""
        elif (
            ch in openers
            and i not in skip
            and (ch != "\u201d" or not ltr or _opens_reversed(text, i, boundary if ltr else None))
        ):
            if open_by:
                spans.append((open_at, i, False))
            open_at, open_by = i, ch
    if open_by:
        spans.append((open_at, len(text), False))

    mask = [False] * (len(text) + 1)
    stray = [False] * (len(text) + 1)
    headings = _heading_line_re(country)
    for start, end, closed in spans:
        # A quote opening on prose quotes no heading, so if nothing closes it, it
        # stops at the next one. One opening on a heading may quote several.
        if (
            not closed
            and headings is not None
            and not headings.match(text[start + 1 : start + 80])
            and (h := headings.search(text, start + 1, end))
        ):
            end = h.start() - 1
        for k in range(start, min(end + 1, len(text))):
            mask[k] = True
            if not closed:
                stray[k] = True
    return tuple(mask), tuple(stray)


def _closes_later(text: str, at: int, opener: str, boundary: re.Pattern[str] | None = None) -> bool:
    """Does a closer for this opener appear after ``at``, before the next provision?
    A quotation does not run over a heading: a closer past one is a later quote's."""
    positions = [p for c in _CLOSERS_FOR_OPENER[opener] if (p := text.find(c, at)) != -1]
    if not positions:
        return False
    return boundary is None or not boundary.search(text, at, min(positions))


def _unclosed_quote_spans(text: str, country: str) -> int:
    """How many quoted spans nothing closes, each masking whatever it reaches.

    Takes the country the masked count was taken under: walking again without it
    reads a different document, and evicts the one-entry cache the caller just
    filled.
    """
    stray = _walk_quotes(text, country)[1]
    return sum(1 for i, on in enumerate(stray) if on and not (i and stray[i - 1]))


@lru_cache(maxsize=32)
def _basic_unit_line_re(country: str) -> re.Pattern[str] | None:
    """A line opening an article-level provision anywhere in the config, or None.

    Every class contributes, and the filter is on the AKN element rather than on
    the class's basic unit, which is not always an article. A config declaring no
    article or section term yields None and so bounds nothing.

    Bounds a reversed span: an amendment quotes one article, so a span running
    over a heading is a dropped closer reaching a later quotation.
    """
    if not country:
        return None
    config = load_config(country)
    entries = [
        entry
        for doc_class in (config.document_classes or {}).values()
        for entry in doc_class.hierarchy
        if entry.akn_element in {"article", "section", "rule"}
    ]
    keyword = _keyword_line_branch(config, entries)
    branches = [keyword] if keyword else []
    # A keyword-less unit ("2. …") is a heading by its declared marker form.
    for form in sorted({e.marker_form for e in entries if e.marker_form in _OUTLINE_RES}):
        branches.append(_OUTLINE_RES[form].pattern.removeprefix("(?m)").replace("(?P<num>", "(?:"))
    if not branches:
        return None
    return re.compile("(?m)" + "|".join(f"(?:{b})" for b in branches))


def _keyword_line_branch(config: JurisdictionConfig, entries: Sequence[HierarchyEntry]) -> str:
    """A line opening with one of these levels' keywords and a number, or ""."""
    terms = {form for entry in entries for form in _alias_terms_for(entry)}
    if not terms:
        return ""
    alts = "|".join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t)))
    # The scanner's own separator and number grammar: what it would anchor is a
    # provision heading here, and a keyword opening prose is not.
    tolerances = set(config.structuring.marker_tolerances if config.structuring else ())
    number = _num_pattern_with(
        config.structuring.ordinal_words if config.structuring else {},
        digit_glyphs="digit_glyph" in tolerances,
        split_numbers="split_number" in tolerances,
    )
    return rf"^[^\S\n]{{0,8}}(?:{alts}){_separator_for(tolerances)}{number}{_MARKER_NUM_END}"


@lru_cache(maxsize=32)
def _heading_line_re(country: str) -> re.Pattern[str] | None:
    """A line opening a keyword-led container or basic unit in any class, or None.

    Bounds a quote nothing closes. Keyword-led only: a bare "1." is as often a list.
    """
    if not country:
        return None
    config = load_config(country)
    entries = [
        entry
        for doc_class in (config.document_classes or {}).values()
        for entry in doc_class.hierarchy
        if entry.level in {"higher", "basic"}
    ]
    branch = _keyword_line_branch(config, entries)
    return re.compile("(?m)" + branch) if branch else None


def _quote_mask(text: str, country: str = "") -> tuple[bool, ...]:
    """Per-char flag: is this offset inside a quoted span?"""
    return _walk_quotes(text, country)[0]


def _stray_quote_mask(text: str, country: str = "") -> tuple[bool, ...]:
    """Per-char flag: is this offset inside a span nothing closes?"""
    return _walk_quotes(text, country)[1]


def _unclosed_openers(text: str) -> frozenset[int]:
    """Offsets of directional openers with no later character to close them.

    Left in, each masks every marker to end of file and is then dropped as quoted
    text with no gate tripping. A stray „ and a lone « in one long code together
    masked 81% of the document.

    Each opener is matched only against the closer it takes: a low-9 pairs with a
    curly, not a guillemet. One shared stack would let a » pop a „, so in `« … „ … »`
    the genuine guillemet is reported unclosed and its markers exposed.
    """
    stacks: dict[str, list[int]] = {opener: [] for opener in _CLOSES_FOR}
    for i, ch in enumerate(text):
        if ch in stacks:
            stacks[ch].append(i)
            continue
        for opener, closers in _CLOSES_FOR.items():
            if ch in closers and stacks[opener]:
                stacks[opener].pop()
                break
    return frozenset(i for stack in stacks.values() for i in stack)


# Opener → the characters that close it. A guillemet takes its mirror; a low-9
# quote takes a curly quote, which is why `_QUOTE_CLOSE` alone can never close
# one. Keyed by opener so each pairs only against its own kind.
_CLOSES_FOR = {"«": _QUOTE_CLOSE, "„": _QUOTE_TOGGLE}


_ORPHAN_PROMOTION_THRESHOLD = 5


def _drop_orphan_children(
    anchors: list[StructuralAnchor],
    rank_map: dict[str, float] | None = None,
    spans: list[AmbiguitySpan] | None = None,
) -> list[StructuralAnchor]:
    """Drop child anchors when no article or section is currently open.

    Rescue clause: short decrees enumerate provisions as points with
    no parent article, and dropping them yields zero anchors, which triggers the
    verbatim fallback that renders the decree as unstructured prose. When at least
    ``_ORPHAN_PROMOTION_THRESHOLD`` child anchors would be dropped and no basic unit
    ever opened, the run is promoted to ``article`` instead.
    """
    from codify.pipeline.enrich.kinds import BASIC_UNIT_KEYWORDS, CHILD_KEYWORDS

    basic = {kw.lower() for kw in BASIC_UNIT_KEYWORDS}
    children = {kw.lower() for kw in CHILD_KEYWORDS}
    basic_ranks = {_rank_of(k, rank_map) for k in basic}
    child_count = sum(1 for a in anchors if a.kind in children)
    basic_count = sum(1 for a in anchors if a.kind in basic)
    promote = child_count >= _ORPHAN_PROMOTION_THRESHOLD and basic_count == 0
    out: list[StructuralAnchor] = []
    stack: list[float] = []
    for a in anchors:
        rank = _rank_of(a.kind, rank_map)
        while stack and stack[-1] >= rank:
            stack.pop()
        if a.kind in children and not any(r in basic_ranks for r in stack):
            if promote:
                # Only the kind changes. Listing fields here dropped whichever
                # ones were added later (quoted_amendment, heading, source_pass).
                a = replace(a, kind="article")
                rank = _rank_of("article", rank_map)
            else:
                if spans is not None:
                    spans.append(
                        AmbiguitySpan(
                            kind="orphan_text",
                            start=a.char_offset,
                            end=a.char_offset + len(a.matched_text),
                            emitted_by="drop_orphan_children",
                            detail={
                                "kind": a.kind,
                                "number": a.number,
                                "reads_as": "no_basic_unit_open",
                            },
                        )
                    )
                continue
        out.append(a)
        stack.append(rank)
    return out


def _provisional_parents(
    anchors: list[StructuralAnchor], rank_map: dict[str, float] | None = None
) -> list[tuple[tuple[str, str] | None, ...]]:
    """Ancestor (kind, num) chain per anchor."""
    chains: list[tuple[tuple[str, str] | None, ...]] = []
    stack: list[tuple[float, tuple[str, str]]] = []
    for a in anchors:
        rank = _anchor_rank(a, rank_map)
        while stack and stack[-1][0] >= rank:
            stack.pop()
        chains.append(tuple(item for _, item in stack))
        stack.append((rank, (a.kind, _normalise_number(a.number))))
    return chains


# Roman numeral values, for reading a container number as an integer so a
# sibling sequence can be compared across `BAB IV`, `BAB V`, `BAB VI`.
_ROMAN_VALUE = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Kinds whose numbering runs the length of a document sparsely, so a
# neighbouring sibling is evidence about where a repeat belongs. Articles are
# excluded deliberately: an elucidation restates them densely, and two `Pasal
# 200` a page apart are a restatement rather than a contents entry and a body.
_SEQUENCE_KINDS = CONTAINER_KINDS

# Kinds whose run reads only outside an elucidation, where restatement is dense.
_BODY_SEQUENCE_KINDS = {"article", "section"}


# Scans read a zero as a capital O and a one as l or I. Folded for sequence
# comparison only: `_normalise_number` is country-agnostic and a fold there
# would rewrite a genuine `5I` in every jurisdiction.
_GLYPH_DIGITS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})


def _clean_number(num: str | None) -> bool:
    """Did the scan read this number without damage?"""
    if not num:
        return False
    text = normalise_digits(str(num)).strip()
    return text.isdigit() or bool(text) and all(c in _ROMAN_VALUE for c in text.upper())


def _roman_or_digit(num: str | None) -> int | None:
    """The integer a marker number denotes, for Roman or Arabic forms alike."""
    if not num:
        return None
    text = normalise_digits(str(num)).strip()
    # A token mixing digits with those letters is damage, not a Roman numeral:
    # `2O2` is 202, while a clean `II` or `XL` has no digit to give it away.
    if any(ch.isdigit() for ch in text) and not text.isdigit():
        folded = text.translate(_GLYPH_DIGITS)
        if folded.isdigit():
            return int(folded)
    if text.isdigit():
        return int(text)
    # An inserted unit sits between its base and the next number.
    if (inserted := fold_inserted_suffix(text)) is not None:
        base = re.match(r"\d+", inserted)
        if base is not None:
            return int(base.group(0))
    upper = text.upper()
    if not upper or any(c not in _ROMAN_VALUE for c in upper):
        return None
    total = 0
    for i, ch in enumerate(upper):
        value = _ROMAN_VALUE[ch]
        nxt = _ROMAN_VALUE.get(upper[i + 1]) if i + 1 < len(upper) else None
        total += -value if nxt and nxt > value else value
    return total


def _sequence_survivor(
    anchors: list[StructuralAnchor],
    siblings: list[int],
    positions: list[int],
    in_schedule: bool = False,
) -> int | None:
    """The candidate whose siblings read as `n-1` and `n+1`, or None for keep-last."""
    kind = anchors[positions[0]].kind
    # A container number runs the document, but an Ayat restarts every article.
    # An article's number runs the body too, so it reads there; inside a
    # Penjelasan the same articles restate densely and the run means nothing.
    if kind not in _SEQUENCE_KINDS and kind not in _BODY_SEQUENCE_KINDS:
        return None
    # Choosing between restated articles reparents everything beneath them, so
    # inside a Penjelasan the subtree has to be readable first. The glyph fold
    # makes a damaged run look ordered, so the damage is checked on the raw
    # numbers of the children, which is where the collision actually lands.
    if (
        in_schedule
        and kind in _BODY_SEQUENCE_KINDS
        and not _subtrees_read_cleanly(anchors, positions)
    ):
        return None
    # Adjacent siblings are a marker and its own heading: the contents shape keep-last wants.
    places = [siblings.index(p) for p in positions]
    if any(b - a == 1 for a, b in zip(sorted(places), sorted(places)[1:])):
        return None
    # A run that doubles back anywhere near a candidate says nothing about any of them.
    scored = [_sequence_score(anchors, siblings, p) for p in positions]
    fits = [score for score in scored if score is not None]
    if len(fits) != len(scored):
        return None
    # Keep-last is the incumbent, so a tie is not evidence; it has to be beaten.
    best: int | None = None
    best_score = fits[-1]
    for position, score in zip(positions[:-1], fits):
        if score > best_score:
            best, best_score = position, score
    return best


def _subtrees_read_cleanly(anchors: list[StructuralAnchor], positions: list[int]) -> bool:
    """Do the markers hanging under each candidate parse without damage?"""
    kind = anchors[positions[0]].kind
    for position in positions:
        for a in anchors[position + 1 :]:
            if a.kind == kind:
                break
            if a.kind in CONTAINER_KINDS:
                break
            if not _clean_number(a.number):
                return False
    return True


def _sequence_score(
    anchors: list[StructuralAnchor], siblings: list[int], position: int
) -> int | None:
    """How well a candidate's siblings read as `n-1` and `n+1`; None if nothing can be read."""
    number = _roman_or_digit(anchors[position].number)
    if number is None:
        return None
    at = siblings.index(position)
    before = _roman_or_digit(anchors[siblings[at - 1]].number) if at > 0 else None
    after = _roman_or_digit(anchors[siblings[at + 1]].number) if at + 1 < len(siblings) else None
    if before is not None and after is not None and before >= after:
        return None
    return (1 if before == number - 1 else 0) + (1 if after == number + 1 else 0)


def _drop_toc_duplicates(
    anchors: list[StructuralAnchor],
    rank_map: dict[str, float] | None = None,
    spans: list[AmbiguitySpan] | None = None,
) -> list[StructuralAnchor]:
    """Drop anchors sharing (ancestor chain, kind, number), keeping the one the sibling
    sequence places there and the later occurrence otherwise.

    The later is the body and the earlier the TOC twin, except where a document
    legitimately repeats a number. A rule rather than a reading, so each drop is
    recorded.
    """
    if not anchors:
        return anchors
    chains = _provisional_parents(anchors, rank_map)
    groups: dict[tuple[tuple[tuple[str, str] | None, ...], str, str], list[int]] = {}
    for idx, a in enumerate(anchors):
        key = (chains[idx], a.kind, _normalise_number(a.number))
        groups.setdefault(key, []).append(idx)
    siblings: dict[tuple[tuple[tuple[str, str] | None, ...], str], list[int]] = {}
    for idx, a in enumerate(anchors):
        siblings.setdefault((chains[idx], a.kind), []).append(idx)
    drop: set[int] = set()
    for key, positions in groups.items():
        if len(positions) > 1:
            in_schedule = any(c and c[0] == "schedule" for c in chains[positions[0]])
            chosen = _sequence_survivor(anchors, siblings[(key[0], key[1])], positions, in_schedule)
            # A sequence-selected drop is a restatement, not a contents twin, which is
            # the reading this pass otherwise asserts. `corpus_scan` aggregates the
            # reason, so sharing one would count these as evidence for the premise
            # they disprove.
            reads_as = "sequence_superseded" if chosen is not None else "toc_twin"
            if chosen is None:
                chosen = positions[-1]
            drop.update(p for p in positions if p != chosen)
            if spans is not None:
                kept = anchors[chosen]
                for i in (p for p in positions if p != chosen):
                    spans.append(
                        AmbiguitySpan(
                            kind="duplicate_number",
                            start=anchors[i].char_offset,
                            end=anchors[i].char_offset + len(anchors[i].matched_text),
                            emitted_by="drop_toc_duplicates",
                            resolved=True,
                            detail={
                                "kind": anchors[i].kind,
                                "number": anchors[i].number,
                                "reads_as": reads_as,
                                "kept_at": kept.char_offset,
                                # The grouping this drop was decided in. The pass
                                # already keys on it; without it on the span a reader
                                # sees only the number, and one number in two
                                # containers reads as one unit seen twice.
                                "container": [list(step) for step in chains[i] if step],
                            },
                        )
                    )
    return [a for i, a in enumerate(anchors) if i not in drop]


# Any phrase mentioning "المواد" (plural) counts as a plural cue for the
# multi-article budget in _mark_embedded_amendment_articles.
_PLURAL_MARKER_RE = ARABIC.plural_marker_re


# Arabic-Indic and extended Arabic-Indic digits, folded to ASCII.
_DIGIT_FOLD = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _names_number(span: str, number: str) -> bool:
    """Does the lead-in itself cite the number of the article it introduces?"""
    digits = re.sub(r"\D", "", number.translate(_DIGIT_FOLD))
    if not digits:
        return False
    return digits in re.findall(r"\d+", span.translate(_DIGIT_FOLD))


def _leadin_before(
    text: str,
    prev_offset: int,
    cand_offset: int,
    trigger_re: re.Pattern[str],
    cand_number: str = "",
) -> str | None:
    """Is the candidate anchor introduced by an amendment-replacement lead-in?

    The embedded header carries a colon within ~120 chars, and the text before it a
    "the following" pointer; the caller's forward-jump guard supplies the precision.
    Returns "plural", "singular", or None.
    """
    start = max(prev_offset, cand_offset - 300)
    span = text[start:cand_offset]
    if not span:
        return None
    colon = max(span.rfind(":"), span.rfind("："))
    if colon >= 0:
        if (len(span) - colon) > 120:
            return None
        head = span[:colon]
    else:
        # Without the colon, ordinary prose ending on "as follows" would
        # swallow the next heading, so the clause must name its own article.
        if not cand_number or not _names_number(span, cand_number):
            return None
        matches = list(trigger_re.finditer(span))
        if not matches or (len(span) - matches[-1].end()) > 40:
            return None
        head = span[: matches[-1].end()]
    if not trigger_re.search(head):
        return None
    return "plural" if _PLURAL_MARKER_RE.search(head) else "singular"


_ROMAN_NUM_RE = re.compile(r"[IVXLC]+")


@lru_cache(maxsize=32)
def _amending_body_structure(country: str) -> str:
    from codify.jurisdictions import load_config

    config = load_config(country) if country else None
    return (config.amendments.amending_body_structure or "") if config and config.amendments else ""


def _mark_amendments_under_roman_host(
    anchors: list[StructuralAnchor], country: str
) -> list[StructuralAnchor]:
    """In an instrument whose own articles are Roman, an Arabic-numbered article after
    the first of them is replacement text for another law.

    Declared as `amendments.amending_body_structure`, being a drafting rule rather
    than something to infer. The lead-in heuristic cannot reach these: an inserted
    article is introduced once, many lines above its text.
    """
    if _amending_body_structure(country) != "two_pasal_roman":
        return anchors
    arts = [i for i, a in enumerate(anchors) if a.kind == "article"]
    romans = [i for i in arts if _ROMAN_NUM_RE.fullmatch((anchors[i].number or "").strip())]
    # Two, per the declared shape. One is an OCR misread of an Arabic number,
    # and treating it as the host would quote the rest of the document.
    if len(romans) < 2 or romans[0] != arts[0]:
        return anchors
    host_rank = KIND_RANK.get("article", 8)
    mark: set[int] = set()
    # Between the Roman hosts only. `Pasal II` carries commencement, and an
    # annex or elucidation after it is the instrument's own, not quoted.
    for i in arts:
        if not romans[0] < i < romans[-1] or i in romans:
            continue
        mark.add(i)
        for m in range(i + 1, len(anchors)):
            r = KIND_RANK.get(anchors[m].kind)
            if r is None or r <= host_rank:
                break
            mark.add(m)
    if not mark:
        return anchors
    return [replace(a, quoted_amendment=True) if i in mark else a for i, a in enumerate(anchors)]


def _amendment_item_kind(country: str) -> str:
    from codify.jurisdictions import load_config

    config = load_config(country) if country else None
    return (config.amendments.amendment_item_kind or "") if config and config.amendments else ""


# A worked example runs to a few specimen headings, not to the end of the act.
_EXAMPLE_BUDGET = 1500
# A displayed specimen opens by naming the instrument it illustrates.
_SPECIMEN_TITLE_RE = re.compile(
    r"^(UNDANG-UNDANG|PERATURAN|PERPPU|PERPU|KEPUTUSAN|INSTRUKSI|QANUN)\b", re.IGNORECASE
)


@lru_cache(maxsize=64)
def _example_markers_for(country: str) -> tuple[str, ...]:
    """Declared lead-ins that introduce specimen legislation, e.g. `Contoh`."""
    if not country:
        return ()
    config = load_config(country)
    structuring = config.structuring
    return tuple(w for w in (structuring.example_markers if structuring else []) if w)


def _mark_specimen_examples(
    text: str, anchors: list[StructuralAnchor], country: str
) -> list[StructuralAnchor]:
    """Flag markers inside a worked example so they are not read as the document's own.

    A drafting manual quotes specimen legislation under `Contoh 1:`. Those are
    displayed rather than enacted, which `quoted_amendment` already models, and the
    example runs until the numbered rule resumes.
    """
    words = _example_markers_for(country)
    if not words or not anchors:
        return anchors
    trigger = re.compile(
        # Numbered only. A bare `Contoh:` introduces a worked example in an
        # elucidation ("Penghasilan Kena Pajak ..."); the numbered form is what
        # a drafting manual uses to display specimen legislation.
        r"(?m)^[ \t]*(?:" + "|".join(re.escape(w) for w in words) + r")\s+\d+\s*[:：]"
    )
    # A rule number (`2a. Penomoran`) at line start is the manual speaking again.
    resume = re.compile(r"(?m)^[ \t]*\d+[a-z]?\.\s+\S")
    spans: list[tuple[int, int]] = []
    for m in trigger.finditer(text):
        # The example must open on a specimen instrument title. A worked
        # arithmetic example ("Dalam hal Pengusaha Kena Pajak A ...") carries the
        # same lead-in and displays no legislation, so suppressing under it
        # would silence the host's own markers.
        head = next(
            (ln.strip() for ln in text[m.end() : m.end() + 200].split("\n") if ln.strip()),
            "",
        )
        if not _SPECIMEN_TITLE_RE.match(head):
            continue
        # An example that never hands back is a mismatch, not an example the
        # length of the rest of the act, so the budget bounds it either way.
        ceiling = m.end() + _EXAMPLE_BUDGET
        nxt = resume.search(text, m.end(), ceiling)
        spans.append((m.start(), nxt.start() if nxt else ceiling))
    if not spans:
        return anchors
    return [
        replace(a, quoted_amendment=True)
        if not a.quoted_amendment and any(lo <= a.char_offset < hi for lo, hi in spans)
        else a
        for a in anchors
    ]


def _mark_amendment_items(
    text: str, anchors: list[StructuralAnchor], country: str
) -> list[StructuralAnchor]:
    """Flag an item marker that introduces another statute's article.

    An omnibus restates dozens of statutes, each numbering from Pasal 1. With no
    container between them every article shares one ancestor chain, so
    `_drop_toc_duplicates` reads the repeats as TOC twins and collapses them.

    Adjacency in the anchor list is not enough, since a list's last item and the
    host's next article are consecutive too, and nesting the host would reparent the
    rest of the document. The amended article must follow the marker in the source
    with only whitespace between.
    """
    kind = _amendment_item_kind(country)
    if not kind:
        return anchors

    def _introduces(i: int) -> bool:
        """The article sits on the line under the marker, nothing between."""
        a = anchors[i]
        # `char_offset` sits on the newline the boundary consumed, so the scan
        # for the marker's own line end has to start past its matched text.
        line_end = text.find("\n", a.char_offset + len(a.matched_text))
        if line_end == -1:
            return False
        return not text[line_end : anchors[i + 1].char_offset].strip()

    mark = {
        i
        for i, a in enumerate(anchors)
        if a.kind == kind
        and i + 1 < len(anchors)
        and anchors[i + 1].kind == "article"
        and _introduces(i)
    }
    if not mark:
        return anchors
    out = list(anchors)
    for i in mark:
        out[i] = replace(out[i], amendment_item=True)
        out[i + 1] = replace(out[i + 1], amended_article=True)
    return out


def _mark_embedded_amendment_articles(
    text: str, anchors: list[StructuralAnchor], trigger_phrases: tuple[str, ...]
) -> list[StructuralAnchor]:
    """Mark embedded amendment-replacement articles ``quoted_amendment=True`` so
    ``_assign_eids`` skips them and they never surface as peers of the amending act.

    High-precision: marked only when preceded by a replacement lead-in AND not the
    host's next contiguous number. The conjunction catches forward jumps (host 3,
    embedded 151) and backward ones (host 5, article 3 quoted back), while sparing
    genuine additions at ``host_max + 1``.

    Nested child anchors following a marked article are marked too, up to the next
    article-or-higher, so a quoted article's paragraphs do not leak as top-level.
    """
    if not trigger_phrases:
        return anchors
    trigger_re = re.compile("|".join(re.escape(p) for p in trigger_phrases))
    arts = [i for i, a in enumerate(anchors) if a.kind == "article"]
    if not arts:
        return anchors

    def _is_bis(i: int) -> bool:
        return _normalise_number(anchors[i].number).endswith("bis")

    def _in_sequence(i: int, host_max: int) -> bool:
        """Is this the next article the host sequence expects? A bis inserts
        after its base, so it repeats `host_max` instead of advancing it."""
        n = _num(i)
        return n is not None and n == host_max + (0 if _is_bis(i) else 1)

    def _num(i: int) -> int | None:
        # Leading digits, so a bis number is comparable to its host sequence.
        n = _normalise_number(anchors[i].number)
        lead = re.match(r"\d+", n)
        return int(lead.group()) if lead else None

    article_rank = KIND_RANK.get("article", 8)

    def _mark_nested_after(idx: int, out: set[int]) -> None:
        m = idx + 1
        while m < len(anchors):
            r = KIND_RANK.get(anchors[m].kind)
            if r is None or r <= article_rank:
                return
            out.add(m)
            m += 1

    mark: set[int] = set()
    host_max = 0
    k = 0
    while k < len(arts):
        idx = arts[k]
        num = _num(idx)
        prev_offset = anchors[arts[k - 1]].char_offset if k > 0 else 0
        cue = _leadin_before(
            text, prev_offset, anchors[idx].char_offset, trigger_re, anchors[idx].number or ""
        )
        if cue and num is not None and host_max > 0 and not _in_sequence(idx, host_max):
            budget = 8 if cue == "plural" else 1
            j = k
            while j < len(arts) and budget > 0:
                if _in_sequence(arts[j], host_max):
                    break  # back in host sequence, embedded block ended
                mark.add(arts[j])
                _mark_nested_after(arts[j], mark)
                j += 1
                budget -= 1
            k = j
            continue
        if num is not None:
            host_max = max(host_max, num)
        k += 1
    if not mark:
        return anchors
    return [replace(a, quoted_amendment=True) if i in mark else a for i, a in enumerate(anchors)]


# Separators between a container marker and its inline title
# ("CHAPTER 1, Definitions", "الباب الأول: في التعاريف"), plus markdown
# fences the vision OCR wraps around title lines ("### تعاريف").
_HEADING_SEP_STRIP = " \t-–—:·#*_"
_HEADING_MAX_LEN = 80
# Terminal punctuation marking a prose sentence rather than a title.
_HEADING_PROSE_TERMINALS = (".", "؟", "?", "!", "،", ",", ";", "؛", ":")


def _plausible_heading(candidate: str) -> bool:
    if not (2 <= len(candidate) <= _HEADING_MAX_LEN):
        return False
    if not any(ch.isalpha() for ch in candidate):
        return False
    if candidate.endswith(_HEADING_PROSE_TERMINALS):
        return False
    # Enumerated items ("1. ...", "(a) ...") are body content, not titles.
    if candidate[0].isdigit() or candidate[0] in "([":
        return False
    return True


def _line_end(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl == -1 else nl


def _clean_heading(candidate: str) -> str:
    """Trim trailing separator residue; prose terminals were already vetted
    by `_plausible_heading` so only dash/middot/markdown cosmetics remain."""
    return candidate.rstrip(" \t-–—·#*_")


def _offset_in(offsets: list[int], lo: int, hi: int) -> bool:
    """Any anchor offset in [lo, hi)? `offsets` is sorted."""
    i = bisect_left(offsets, lo)
    return i < len(offsets) and offsets[i] < hi


def _capture_container_headings(
    text: str, anchors: list[StructuralAnchor], regex: re.Pattern[str]
) -> list[StructuralAnchor]:
    """Attach the descriptive title to container anchors.

    Titles sit on the marker line after the number, or on the next non-blank line. A
    next line that is marker-shaped or reads as wrapped prose is rejected, leaving
    ``heading=None`` for the validator to surface.

    ``section`` is eligible only when the document also has articles: there it is a
    grouping level (Ukraine's РОЗДІЛ contains Глава and Стаття), while in
    article-less doctypes it is the basic unit whose heading rides body-fill.

    Captures and rejections are structlogged; an anchor with no candidate gets no
    line, only ``heading=None``.
    """
    if not anchors:
        return anchors
    # Schedules never take a captured heading line: the colon form sets its
    # heading at scan time, and a numbered annex's next line is content.
    eligible = (
        _HEADING_KINDS - {"schedule"}
        if any(a.kind == "article" for a in anchors)
        else CONTAINER_KINDS
    )
    # A match can start on leading whitespace/newlines; normalise each
    # anchor's offset to its first marker character before comparing.
    offsets = []
    for a in anchors:
        o = a.char_offset
        while o < len(text) and text[o] in " \t\r\n":
            o += 1
        offsets.append(o)
    sorted_offsets = sorted(offsets)
    out = list(anchors)
    for i, anchor in enumerate(anchors):
        if anchor.kind not in eligible:
            continue
        heading = _heading_for(text, anchor, offsets[i], sorted_offsets, regex)
        if heading:
            logger.info(
                "container_heading_captured",
                kind=anchor.kind,
                number=anchor.number,
                line=anchor.line,
                heading=heading[:_HEADING_MAX_LEN],
            )
            out[i] = replace(anchor, heading=heading)
    return out


def _reject_heading(anchor: StructuralAnchor, candidate: str, reason: str) -> None:
    logger.info(
        "container_heading_rejected",
        kind=anchor.kind,
        number=anchor.number,
        line=anchor.line,
        candidate=candidate[:_HEADING_MAX_LEN],
        reason=reason,
    )


def _heading_for(
    text: str,
    anchor: StructuralAnchor,
    start: int,
    offsets: list[int],
    regex: re.Pattern[str],
) -> str | None:
    # Work in text coordinates from the end of the full match, so a marker
    # OCR-split across lines (`الفصل\nالأول`) does not leak the number line
    # into the walk below.
    match_end = start + len(anchor.matched_text)
    line_end = _line_end(text, match_end)
    # Another anchor on the marker line (old two-column layouts put the
    # next unit after a tab run) means the inline remainder is that unit's
    # marker, never this container's title.
    if _offset_in(offsets, match_end, line_end):
        return None
    # Strip marker separators from the left only: a trailing colon or full
    # stop is what the prose-terminal guard in _plausible_heading keys on
    # ("CHAPTER 1 - The following rules apply:" is a lead-in, not a title).
    rest = text[match_end:line_end].lstrip(_HEADING_SEP_STRIP).rstrip()
    # Some regex variants stop group(0) before the number; skip a leading
    # number token, but only when a separator follows so a title that merely
    # starts with the same string ("2nd Schedule ...") is left whole.
    num = anchor.number or ""
    if num and rest.startswith(num):
        tail = rest[len(num) :]
        if not tail or tail[0] in _HEADING_SEP_STRIP:
            rest = tail.lstrip(_HEADING_SEP_STRIP).rstrip()
    if rest:
        # Mirror of the walk's dropped-anchor guard: a marker-shaped
        # remainder whose anchor an upstream pass dropped is not a title.
        m = regex.search(rest)
        if m is not None and not rest[: m.start()].strip():
            _reject_heading(anchor, rest, "marker_shaped")
            return None
    if _plausible_heading(rest):
        return _clean_heading(rest)
    if rest:
        # Inline content present but rejected (a prose lead-in): the next
        # line is that lead-in's continuation, never this container's title.
        _reject_heading(anchor, rest, "implausible")
        return None
    # Marker alone on its line: the title, if any, is the next non-blank line.
    pos = line_end + 1
    while pos < len(text):
        nl = _line_end(text, pos)
        raw = text[pos:nl]
        if not raw.strip():
            pos = nl + 1
            continue
        if _offset_in(offsets, pos, nl):
            return None  # the next unit's marker: no title in source
        # A marker-shaped line whose anchor an upstream pass dropped
        # (TOC dedup, orphan drop) must not become the title.
        m = regex.search(raw)
        if m is not None and not raw[: m.start()].strip():
            _reject_heading(anchor, raw.strip(), "marker_shaped")
            return None
        candidate = raw.lstrip(_HEADING_SEP_STRIP).rstrip()
        if not _plausible_heading(candidate):
            _reject_heading(anchor, candidate, "implausible")
            return None
        # A title line is followed by a blank line or the next marker;
        # further prose means this is a soft-wrapped body sentence.
        after_end = _line_end(text, nl + 1)
        after = text[nl + 1 : after_end]
        if after.strip() and not _offset_in(offsets, nl + 1, after_end):
            _reject_heading(anchor, candidate, "continuation")
            return None
        return _clean_heading(candidate)
    return None


_PREAMBLE_CONTAINER_KINDS = CONTAINER_KINDS | {"section"}
_ROMAN_ONLY = frozenset("IVXLCDM")


@lru_cache(maxsize=32)
def _body_opens_with_container(country: str) -> bool:
    from codify.jurisdictions import load_config

    config = load_config(country) if country else None
    structuring = config.structuring if config else None
    return bool(structuring and structuring.body_opens_with_container)


def _drop_preamble_citation_articles(
    anchors: list[StructuralAnchor],
    spans: list[AmbiguitySpan] | None = None,
    country: str = "",
) -> list[StructuralAnchor]:
    """Drop articles preceding the first container, where the drafting standard puts
    every article inside one.

    Some laws cite the constitutional basis in a `Mengingat` recital. The
    scanner anchors on that citation, landing an article at <body> root that
    swallows the rest of the preamble, and truncating the preamble besides: opening
    material ends at the first anchor, and that anchor is inside it.

    An uppercase Roman number anywhere in the run means an amending act, whose
    `Pasal I` legitimately precedes a quoted container, and the pass declines rather
    than guessing per anchor.

    Only the main body is searched for that container: an annex carries its own
    hierarchy, so a LAMPIRAN opening with BAB I would otherwise pass as the body's
    first container and drop every real article ahead of it.
    """
    if not _body_opens_with_container(country):
        return anchors
    body_end = next((a.char_offset for a in anchors if a.kind == "schedule"), float("inf"))
    first_container = next(
        (
            a.char_offset
            for a in anchors
            if a.kind in _PREAMBLE_CONTAINER_KINDS and a.char_offset < body_end
        ),
        None,
    )
    # No container in the body: articles at root are the document's real structure.
    if first_container is None:
        return anchors
    candidates = [a for a in anchors if a.kind == "article" and a.char_offset < first_container]
    if not candidates:
        return anchors
    if any((n := (a.number or "").strip()) and set(n) <= _ROMAN_ONLY for a in candidates):
        return anchors
    dropped = {id(a) for a in candidates}
    if spans is not None:
        for a in candidates:
            spans.append(
                AmbiguitySpan(
                    kind="orphan_text",
                    start=a.char_offset,
                    end=a.char_offset + len(a.matched_text),
                    emitted_by="drop_preamble_citation_articles",
                    resolved=True,
                    detail={
                        "kind": a.kind,
                        "number": a.number,
                        "reads_as": "preamble_citation",
                    },
                )
            )
    return [a for a in anchors if id(a) not in dropped]


def _drop_lone_compilation_container(
    anchors: list[StructuralAnchor],
    spans: list[AmbiguitySpan] | None = None,
) -> list[StructuralAnchor]:
    """Drop a single top-level grouping anchor whose number is not 1.

    A standalone act's sole division is numbered 1; a lone container numbered 42 is
    a compilation wrapper, and dropping it reparents the articles. No-op with zero
    or multiple containers, or when the number starts the sequence.
    """
    containers = [i for i, a in enumerate(anchors) if a.kind in _LONE_WRAPPER_KINDS]
    if len(containers) != 1:
        return anchors
    i = containers[0]
    n = _normalise_number(anchors[i].number)
    # Only drop a purely-numeric number >= 2; Roman/letter/absent numbers (a
    # legitimate "PART I", an unnumbered "Preliminary") are left alone.
    if not n.isdigit() or int(n) <= 1:
        return anchors
    if spans is not None:
        spans.append(
            AmbiguitySpan(
                kind="orphan_text",
                start=anchors[i].char_offset,
                end=anchors[i].char_offset + len(anchors[i].matched_text),
                emitted_by="drop_lone_compilation_container",
                resolved=True,
                detail={
                    "kind": anchors[i].kind,
                    "number": anchors[i].number,
                    "reads_as": "compilation_wrapper",
                },
            )
        )
    return [a for j, a in enumerate(anchors) if j != i]


# A drop preserves content when the next anchor follows closely: typical drops
# sit near the document's own anchor gap, body-losing ones at 45x to 343x.
#
# `_ORPHAN_GAP_CHARS` floors every path: below it the loss cannot matter whatever
# the ratio. Above it the ratio decides, so a document with long provisions is
# not flagged for a proportionate drop. `_ORPHAN_GAP_ALWAYS` overrides the ratio,
# because a 7,000-character orphan is pages of law even at under 10x.
_ORPHAN_GAP_RATIO = 10.0
_ORPHAN_GAP_CHARS = 400
_ORPHAN_GAP_ALWAYS = 4000


def _declare_orphaned_drops(
    text: str, anchors: list[StructuralAnchor], spans: list[AmbiguitySpan]
) -> None:
    """Measure the text each drop left without an owner.

    A drop removes a marker; the text it governed does not leave with it, and falls
    to whatever precedes. For the first anchor that is the preamble, since
    `split_opening_material` ends at the first surviving anchor. Nothing else in the
    scan measures this, so a pass can move a run of definitions out of the body while
    appearing to remove one citation.

    Measured against the document's own median anchor gap, never a corpus norm: a
    jurisdiction's preamble length is not a defect signal, EU directives being mostly
    recitals by construction.

    Both gap and median come from the pre-drop list. Consecutive drops chain, so a
    dropped TOC entry whose next survivor sits past the listing would be charged with
    the whole run; and survivors thin as drops take more, so a document that lost
    nearly every anchor would have no denominator and escape unmeasured, which is the
    largest loss and the one most needing report.

    Every drop pass declares, via `_fire_dropped` where it does not record itself. A
    silent filter would put its anchors in neither set, and the next measured drop
    would be charged with the gap it left.
    """
    dropped = [s for s in spans if s.emitted_by.startswith("drop_")]
    if not dropped:
        return
    dropped_starts = {s.start for s in dropped}
    boundaries = _drop_boundaries(anchors, dropped)
    # The rhythm of the undisturbed document, so a gap opened by a drop does not
    # enter the median it is about to be compared against. In a document with
    # few anchors the orphaned gap would otherwise dominate its own denominator
    # and read as proportionate however much it took.
    gaps = [
        b - a for a, b in zip(boundaries, boundaries[1:], strict=False) if a not in dropped_starts
    ]
    # No undisturbed gap left to compare against, or colliding offsets median to
    # zero. Either way the absolute floor decides alone rather than nothing does.
    median = statistics.median(gaps) if gaps else 0.0
    for span in dropped:
        following = next((o for o in boundaries if o > span.start), len(text))
        orphaned = following - span.start
        if orphaned < _ORPHAN_GAP_CHARS:
            continue
        proportionate = bool(median) and orphaned < _ORPHAN_GAP_RATIO * median
        if proportionate and orphaned < _ORPHAN_GAP_ALWAYS:
            continue
        spans.append(
            AmbiguitySpan(
                kind="orphan_text",
                start=span.start,
                end=following,
                emitted_by="declare_orphaned_drops",
                detail={
                    "dropped_by": span.emitted_by,
                    "orphaned_chars": orphaned,
                    "median_gap": round(median),
                    "ratio": round(orphaned / median, 1) if median else None,
                },
            )
        )


def _past_indent(match: re.Match[str]) -> int:
    """Offset of a match's first non-whitespace character; its end if it has none."""
    return match.start() + len(match.group(0)) - len(match.group(0).lstrip())


def _drop_boundaries(anchors: list[StructuralAnchor], dropped: list[AmbiguitySpan]) -> list[int]:
    """Offsets that end a dropped marker's run: survivors and other drops alike."""
    return sorted({*(a.char_offset for a in anchors), *(s.start for s in dropped)})


def _declare_act_boundary_suspected(
    text: str, anchors: list[StructuralAnchor], spans: list[AmbiguitySpan], country: str
) -> None:
    """A fold whose run closes one instrument and opens another: two acts numbered
    from 1 read as one, and keep-last took the earlier. Declared, never undone."""
    from codify.pipeline.enrich.closing import _phrase_pattern, closing_phrases_for

    config = load_config(country) if country else None
    segmentation = config.segmentation if config is not None else None
    headings = segmentation.act_heading_patterns if segmentation is not None else []
    phrases = [p for p in closing_phrases_for(country) if p.strip()]
    folds = [s for s in spans if s.emitted_by == "drop_toc_duplicates"]
    if not headings or not phrases or not folds:
        return
    # Anywhere on a line: a closing phrase opening its own line ends the body
    # before the fold runs, so a fold only ever sees one that does not.
    closing = re.compile("|".join(map(_phrase_pattern, phrases)))
    # One pattern each: two may both name a `number` group.
    patterns = [heading_line_pattern(h) for h in headings]
    boundaries = _drop_boundaries(anchors, [s for s in spans if s.emitted_by.startswith("drop_")])
    quoted: tuple[bool, ...] | None = None
    for fold in folds:
        end = next((o for o in boundaries if o > fold.start), len(text))
        if closing.search(text, fold.start, end) is None:
            continue
        if quoted is None:
            quoted = _closed_quote_mask(text, country)
        mask = quoted
        # A quoted candidate is cited text: skip it and keep reading the run.
        shut = next(
            (m for m in closing.finditer(text, fold.start, end) if not mask[m.end() - 1]), None
        )
        candidates = (
            m for p in patterns for m in p.finditer(text, shut.end() if shut else end, end)
        )
        # Judged at the heading's first character: a whole-line match may end in a quote.
        opened = min(
            # A match of whitespace alone, a blank line included, names no heading.
            (m for m in candidates if _past_indent(m) < m.end() and not mask[_past_indent(m)]),
            key=lambda m: m.start(),
            default=None,
        )
        if shut is None or opened is None:
            continue
        line_end = text.find("\n", opened.start(), end)
        spans.append(
            AmbiguitySpan(
                kind="act_boundary_suspected",
                start=fold.start,
                end=end,
                emitted_by="declare_act_boundary_suspected",
                detail={
                    "number": fold.detail.get("number"),
                    "closing_at": shut.start(),
                    "heading_at": opened.start(),
                    "heading": text[opened.start() : line_end if line_end >= 0 else end].strip(),
                    "orphaned_chars": end - fold.start,
                },
            )
        )


_ABJAD_RANK = ARABIC.abjad_rank
_ABJAD_FOLD = ARABIC.abjad_fold


def _bis_idx_ascii(idx: str) -> str:
    """ASCII eid form of a bis index: digits fold to ASCII, an abjad letter to a
    latin letter (أ→a, ب→b) so a letter index stays distinct from a numeric one."""
    digits = normalise_digits(idx)
    if digits.isascii():
        return digits
    latin = abjad_index_to_latin(idx)
    if latin is None:
        # In-range but non-abjad (ة, ؤ, hamza): keep the eid ASCII, flag drift.
        logger.warning("bis_index_unmapped", idx=idx)
        return "x"
    return latin


def _normalise_number(num: str | None) -> str:
    """Canonicalise a number for eid derivation and TOC-vs-body dedup.

    Arabic-Indic digits fold to ASCII, and Arabic ordinal words to their integer via
    the `arabic_normalise` table, so `الفصل الأول` and `الفصل ١` both give `chp_1`
    and dedup collapses them. Also keeps eids ASCII-pure per the AKN 3.0 naming
    convention, since `chp_الأول` violates URL rules.
    """
    if not num:
        return "0"
    bis = _split_bis(num)
    if bis is not None:
        base, idx = bis
        return f"{_normalise_number(base)}bis{_bis_idx_ascii(idx)}"
    stripped = normalise_digits(num).strip()
    if not stripped:
        return "0"
    if (inserted := fold_inserted_suffix(stripped)) is not None:
        return inserted
    if (folded := ordinal_word_folds().get(re.sub(r"\s+", " ", stripped))) is not None:
        return folded
    if len(stripped) == 1 and "ء" <= stripped <= "ي":
        rank = _ABJAD_RANK.get(stripped.translate(_ABJAD_FOLD))
        if rank is not None:
            return str(rank)
    # A slashed insertion keys as the parser's own eId form.
    return latinise_arabic_ordinal(stripped) or re.sub(r"(?<=\d)/(?=\d)", "-", stripped)


@lru_cache(maxsize=64)
def _trigger_phrases_for(country: str) -> tuple[str, ...]:
    """Amendment trigger phrases declared on the jurisdiction config.
    Empty tuple when no country is named, the amendments block is absent, or the
    trigger list is empty; the caller short-circuits on empty."""
    if not country:
        return ()
    config = load_config(country)
    if config.amendments is None:
        return ()
    return tuple(config.amendments.trigger_phrases or ())


def _rank_map_for(country: str, doctype: str) -> dict[str, float] | None:
    """Containment rank per AKN element, from the jurisdiction's declared hierarchy.

    ``KIND_RANK`` encodes AKN's default nesting. Jurisdictions that invert it
    (Ukraine: Розділ contains Глава) need their own order or chapters collapse into
    their parent.
    """
    if not country:
        return None
    config = load_config(country)
    doc_class = config.get_document_class(doctype)
    if not doc_class or not doc_class.hierarchy:
        return None
    ranks: dict[str, float] = {e.akn_element: float(i) for i, e in enumerate(doc_class.hierarchy)}
    # An attachment level the body never declares still needs a place in the
    # order. Without one it falls past every declared kind, so an annex section
    # cannot scope the paragraphs under it and they all read as duplicates.
    for att in config.attachments:
        levels = list(att.hierarchy)
        # Above the topmost declared rank, and above the schedule sentinel.
        prev, i = -1.0, 0
        while i < len(levels):
            if levels[i].akn_element in ranks:
                prev = ranks[levels[i].akn_element]
                i += 1
                continue
            j = i
            while j < len(levels) and levels[j].akn_element not in ranks:
                j += 1
            run = levels[i:j]
            nxt = (
                ranks[levels[j].akn_element]
                if j < len(levels)
                else max(ranks.values(), default=0.0) + 1.0
            )
            # Sit just above the next shared level, spreading a run of gaps so
            # no two tie: equal ranks stop one level scoping the next.
            step = min(0.5, (nxt - prev) / (len(run) + 1))
            for k, entry in enumerate(run):
                ranks[entry.akn_element] = nxt - (len(run) - k) * step
            prev, i = ranks[run[-1].akn_element], j
    return ranks


def _abbrev_map_for(country: str, doctype: str) -> dict[str, str]:
    """Config eId overrides, keyed by AKN element.

    The generator only sees the element, so an element two local terms share
    cannot be told apart here; those are dropped and take the canonical value.
    """
    if not country:
        return {}
    config = load_config(country)
    doc_class = config.get_document_class(doctype)
    if not doc_class or not doc_class.hierarchy:
        return {}
    declared: dict[str, set[str]] = {}
    for entry in doc_class.hierarchy:
        if entry.akn_element:
            declared.setdefault(entry.akn_element, set()).add(
                entry.eid_abbrev or eid_abbrev(entry.akn_element)
            )
    return {el: next(iter(v)) for el, v in declared.items() if len(v) == 1}


def _rank_of(kind: str, rank_map: dict[str, float] | None) -> float:
    """Config rank when declared, else the global default placed after every
    declared kind so ad-hoc kinds never outrank the hierarchy. Schedules
    outrank everything: an annex closes the whole hierarchy and attaches at
    document level, never inside the preceding article."""
    if kind == "schedule":
        return -1
    if rank_map is not None and kind in rank_map:
        return rank_map[kind]
    base = len(rank_map) if rank_map else 0
    return base + KIND_RANK.get(kind, 8)


def _anchor_rank(anchor: StructuralAnchor, rank_map: dict[str, float] | None) -> float:
    """`_rank_of`, with an amendment item lifted to parent its article.

    The item's declared place is the deepest subdivision, which cannot hold the
    article it introduces. Fractional ranks seat host, item and amended article in
    order without disturbing other kinds.
    """
    article = _rank_of("article", rank_map)
    if anchor.amendment_item:
        # Under its own host article, not beside it: the host is an article too.
        return article + 0.5
    if anchor.amended_article:
        return article + 0.75
    return _rank_of(anchor.kind, rank_map)


def _note_restarted(
    spans: list[AmbiguitySpan] | None, anchor: StructuralAnchor, eid: str = ""
) -> None:
    """Record an outline anchor left to body text because its run restarted."""
    if spans is None:
        return
    spans.append(
        AmbiguitySpan(
            kind="duplicate_number",
            start=anchor.char_offset,
            end=anchor.char_offset + len(anchor.matched_text),
            eid=eid,
            emitted_by="assign_eids",
            resolved=True,
            detail={
                "kind": anchor.kind,
                "number": anchor.number,
                "reads_as": "restarted_annex_list",
            },
        )
    )


def _assign_eids(
    anchors: list[StructuralAnchor],
    rank_map: dict[str, float] | None = None,
    spans: list[AmbiguitySpan] | None = None,
    abbrevs: dict[str, str] | None = None,
) -> list[StructuralAnchor]:
    """Walk anchors in document order; derive depth, parent_eid and akn_eid.

    Ranks come from the declared hierarchy where available, else ``KIND_RANK``. The
    stack closes any open container whose rank is greater than or equal to the
    current anchor's, which is how a new ARTICLE pops out of the previous PARAGRAPH.
    """
    out: list[StructuralAnchor] = []
    stack: list[tuple[float, str]] = []  # (rank, akn_eid)
    sibling_counts: dict[tuple[str, str], int] = {}
    # Rank of a dropped restarted-list container, while its subtree is being
    # dropped with it. Without this its children re-parent onto the
    # grandparent instead of returning to body text.
    dropped_rank: float | None = None
    for anchor in anchors:
        if anchor.quoted_amendment:
            # Quoted embedded articles ride through the anchor list but
            # neither get a top-level eid nor push the container stack;
            # the scaffolder omits them from the emitted skeleton.
            out.append(anchor)
            continue
        rank = _anchor_rank(anchor, rank_map)
        if dropped_rank is not None:
            if rank > dropped_rank and anchor.source_pass == _OUTLINE_PASS:
                _note_restarted(spans, anchor)
                continue
            dropped_rank = None
        while stack and stack[-1][0] >= rank:
            stack.pop()
        parent_eid = stack[-1][1] if stack else None
        abbrev = (abbrevs or {}).get(anchor.kind) or eid_abbrev(anchor.kind)
        num = _normalise_number(anchor.number)
        own = f"{abbrev}_{num}"
        # The suffix makes the eId unique but hides why: a genuine repeat, a
        # missed TOC twin, or the "0" bucket unnumbered anchors land in.
        key = (parent_eid or "", own)
        sibling_counts[key] = sibling_counts.get(key, 0) + 1
        if sibling_counts[key] > 1 and anchor.source_pass == _OUTLINE_PASS:
            # An annex list that restarts under one heading has no container to
            # separate the runs, so a second "1." is presentation, not a
            # provision. Drop it back to body text rather than break round-trip.
            _note_restarted(spans, anchor, f"{parent_eid}__{own}" if parent_eid else own)
            sibling_counts[key] -= 1
            dropped_rank = rank
            continue
        if sibling_counts[key] > 1:
            if spans is not None:
                spans.append(
                    AmbiguitySpan(
                        kind="duplicate_number",
                        start=anchor.char_offset,
                        end=anchor.char_offset + len(anchor.matched_text),
                        eid=f"{parent_eid}__{own}" if parent_eid else own,
                        emitted_by="assign_eids",
                        detail={
                            "kind": anchor.kind,
                            "number": anchor.number,
                            "occurrence": sibling_counts[key],
                            "unnumbered": num == "0",
                        },
                    )
                )
            own = f"{own}_{sibling_counts[key]}"
        akn_eid = f"{parent_eid}__{own}" if parent_eid else own
        out.append(
            replace(
                anchor,
                depth=len(stack),
                parent_eid=parent_eid,
                akn_eid=akn_eid,
                akn_wid=akn_eid,
            )
        )
        stack.append((rank, akn_eid))
    return out


def _keyword_from_match(match: re.Match[str], kind: str) -> str:
    """The keyword literal this match fired on.

    Read from the kind's own capture group rather than by splitting on whitespace,
    which the separator's non-space characters break.
    """
    return match.group(_group_name(kind))


def _kind_from_match(match: re.Match[str]) -> str | None:
    for name, value in match.groupdict().items():
        if not name.startswith("k_") or value is None:
            continue
        return name[2:].replace("_", "-")
    return None


def _line_start(text: str, offset: int) -> int:
    """Start of the line ``offset`` sits on, skipping a leading newline first.

    An anchor's ``char_offset`` is the newline before its marker, because the
    scanner matches from there to read a wrapped line's tail. Comparing raw
    offsets against a pattern that anchors after the newline never matches.
    """
    while offset < len(text) and text[offset] == "\n":
        offset += 1
    return text.rfind("\n", 0, offset) + 1


def _marker_numbers(
    text: str,
    config: JurisdictionConfig | None,
    doctype: str,
    kind: str,
    exclude_starts: frozenset[int] = frozenset(),
) -> set[str]:
    """Distinct numbers whose hierarchy-kind marker sits at structural position.

    Anchored to the same column-boundary pattern as ``build_anchor_regex``, so
    in-prose cross-references are excluded and the count stays comparable to
    ``len(scan_anchors)`` for the coverage-delta gate.

    Empty when the config carries no aliases for the kind, which callers read as no
    expected markers and skip the gate.

    ``exclude_starts`` drops occurrences by position, never by number: a host and
    a quoted article can share one, and dropping the number would let the quoted
    occurrence stand in for a host the scan missed. Keyed on line-start offsets
    because an anchor's ``char_offset`` sits on the newline before its marker
    while this pattern anchors after it, and ``line`` counts from the former.
    """
    if config is None:
        return set()
    doc_class = config.get_document_class(doctype)
    if doc_class is None:
        return set()
    # Keyword-less levels answer to their own scanner, not to keyword aliases,
    # so count what that scanner would find. Without this the denominator is
    # empty and the gate silently skips the very provisions that are citable.
    declared = [e for e in doc_class.hierarchy if e.marker_form]
    # Unioned, not returned alone: the scan counts both spellings, and one of
    # them alone scores a document against a fraction of what it captured.
    declared_found = _scan_declared_markers(text, 0, declared, config.code)[0]
    if _marginal_note_kind(config.code, doctype) == kind:
        in_quote: Sequence[bool] = _quote_mask(text, config.code)
        if config.code in _UK_JURISDICTIONS:
            in_quote = _uk_amendment_mask(text, in_quote)
        declared_found.extend(
            _scan_marginal_note_units(
                text,
                kind,
                _anchor_lines(text, declared_found),
                end=_first_attachment_marker(text, config, doctype, in_quote),
                in_quote=in_quote,
            )
        )
    aliases: list[str] = []
    for entry in doc_class.hierarchy:
        if entry.akn_element != kind:
            continue
        aliases.extend(_alias_terms_for(entry))
    pattern = (
        _compile_anchor_regex(
            {kind: aliases}, config, boundary="column_only", case_insensitive=True
        )
        if aliases
        else None
    )
    # One floor for both grammars, from the earliest marker either reads: the
    # body may use the keyword form and an appended instrument the bare one.
    first_alias = next((m.start() for m in pattern.finditer(text)), None) if pattern else None
    candidates = [a.char_offset for a in declared_found]
    if first_alias is not None:
        candidates.append(first_alias)
    floor = _closing_floor(text, config.code, min(candidates)) if candidates else None
    from_declared = {
        _normalise_number(a.number)
        for a in declared_found
        if a.kind == kind
        and a.number
        and _line_start(text, a.char_offset) not in exclude_starts
        and (floor is None or a.char_offset < floor)
    }
    if pattern is None:
        return from_declared
    # Composes the scanner's own grammar, so declaring an ordinal word or a
    # damage tolerance cannot read as lost coverage: a denominator on the strict
    # grammar would let a recovered marker raise `captured` without raising
    # `expected`. A number after the keyword is what marks a structural header.
    # Distinct provision numbers, under the scanner's own prose filter. Raw
    # occurrences over-count TOC entries and cross-references, so a raw
    # denominator makes improved filtering read as lost coverage. A provision
    # the scanner misses still counts, its number being distinct.
    #
    # Approximation: numbering that restarts in a new scope collapses onto the
    # body's, so this is a floor against systemic drops. Single-provision gaps
    # belong to the anchor-count and cover-reconciliation validators.
    distinct: set[str] = set()
    for m in pattern.finditer(text):
        if _partial_decimal_number(text, m):
            continue
        # Past the closing phrase is not the body, for the denominator either.
        if floor is not None and floor <= m.start():
            break
        # The scanner's matches begin on the preceding newline, which is
        # what lets the prose window read a wrapped line's tail.
        probe = _probe_from(text, m.start())
        if _is_prose_reference(text, probe, config.code if config else ""):
            continue
        if _line_start(text, m.start()) in exclude_starts:
            continue
        if _opens_citation_run(text, m.end(), config.code if config else ""):
            continue
        distinct.add(_normalise_number(_repair_damaged_num(m)))
    return distinct | from_declared


# Line-start Arabic article keyword, digit-agnostic and tolerant of the stray `)`
# RTL artifact, followed by a boundary. Independent of the marker scan's
# keyword+number shape, so it sees headers a systemic digit/boundary drop hides.
_AR_ARTICLE_HEADER_RE = re.compile(
    r"(?m)^[^\S\n]{0,8}\)?[^\S\n]{0,4}(?P<kw>(?:ال)?مادة)(?=[\s(\[:.\-–—0-9٠-٩۰-۹]|$)"
)


def arabic_article_headers(text: str, country: str = "") -> tuple[int, int]:
    """(clean headers, quoted headers) at line start in Arabic source text.

    Quote-masked, prose-filtered and digit-agnostic, so it counts headers the marker
    scan drops on digit corruption. ``country`` selects declared prose precursors,
    defaulting to the generic Arabic filter.
    """
    mask = _quote_mask(text, country)
    clean = quoted = 0
    for m in _AR_ARTICLE_HEADER_RE.finditer(text):
        if mask[m.start("kw")]:
            quoted += 1
            continue
        # The preceding newline, so the filter reads the previous line's tail.
        if not _is_prose_reference(text, _probe_from(text, m.start()), country):
            clean += 1
    return clean, quoted


def _masked_marker_count(text: str, config: JurisdictionConfig | None, doctype: str) -> int:
    """Declared markers a span nothing closed hid, over every level declared.

    Not narrowed to the kind being measured: the gate measures the basic unit,
    which declares no marker form, so a count narrowed to it is always zero.
    """
    if config is None:
        return 0
    doc_class = config.get_document_class(doctype)
    if doc_class is None:
        return 0
    # Attachment levels too, an unclosed quote in an annex hiding markers the body
    # scan never looks at. One entry per form, or a form both declare is counted
    # twice; scanned over the whole text, since the annex spans are not known here.
    body = [e for e in doc_class.hierarchy if e.marker_form]
    total = len(_scan_declared_markers(text, 0, body, config.code)[1]) if body else 0
    # An annex form is structural only inside the annex: a body list numbered
    # the same way would otherwise read as a hidden provision. The first
    # caption is where the annexes begin.
    annex = [e for _c, lv in _attachment_hierarchies(config.code) for e in lv if e.marker_form]
    if not annex:
        return total
    # The caption alone on its line, bar a numeral: a sentence opening with the
    # word is prose, and taking it as the boundary counts body lists as annex
    # provisions.
    starts = [
        m.start()
        for cap, _lv in _attachment_hierarchies(config.code)
        for m in re.finditer(
            rf"(?mi)^[^\S\n]{{0,8}}{re.escape(cap)}(?:[^\S\n]+[IVXLC\d]+)?[^\S\n]*$", text
        )
    ]
    if not starts:
        return total
    # Scanned over the whole text and filtered by offset, not over a slice: a
    # span opening in the body reaches into the annex, and a slice cannot see
    # the quote that opened it.
    first = min(starts)
    total += sum(
        1 for offset in _scan_declared_markers(text, 0, annex, config.code)[1] if offset >= first
    )
    return total


@dataclass(frozen=True)
class AnchorCoverage:
    """What the scanner found of one hierarchy kind against what the source claims.

    ``ratio`` says a scan went wrong and ``missing`` says where. ``ratio`` is None
    when ``expected`` is empty, or a document whose markers are wholly unrecognised
    would score a perfect 1.0.
    """

    kind: str
    ratio: float | None
    captured: frozenset[str]
    expected: frozenset[str]
    masked: int = 0
    """Markers a span nothing closed hid from both sides of the ratio, which
    then reads 1.0 on a document that lost most of its provisions."""
    unclosed: int = 0
    """How many such spans there were, which the marker count alone cannot say."""

    @property
    def missing(self) -> frozenset[str]:
        return self.expected - self.captured


def _marker_line_start(text: str, offset: int) -> int:
    """Line start of the marker an anchor opens on.

    An anchor's offset can land in the whitespace *before* its marker's newline
    (a line ending "…: \n"), where `_line_start` alone keys the previous line
    and every position-keyed exclusion silently misses.
    """
    i = offset
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return _line_start(text, i)


def anchor_coverage(
    text: str,
    anchors: Iterable[StructuralAnchor],
    config: JurisdictionConfig | None,
    doctype: str,
    kind: str,
    *,
    with_masked: bool = False,
) -> AnchorCoverage:
    """Coverage of one hierarchy kind, itemised. ``ratio`` is None when the
    source claims no markers of that kind, which is both a short order with
    none and a document whose markers went unrecognised; the gate treats it as
    a pass and the caller can tell it apart from a measured 1.0."""
    anchors = list(anchors)
    # Same keying as the denominator: distinct normalised numbers, so a
    # duplicated anchor (art 77 emitted twice) cannot mask a missed one.
    # Quoted amendment articles are excluded here as they are in
    # `anchor_summary`: they borrow another act's structure and `_assign_eids`
    # never emits them, so counting them credits coverage never assembled.
    captured = {
        _normalise_number(a.number) if a.number else f"anon_{i}"
        for i, a in enumerate(anchors)
        if a.kind == kind and not a.quoted_amendment
    }
    # Their markers are in the source, so the denominator sees them too and
    # excluding one side alone would read as lost coverage. Dropped by position,
    # not by number: a host and a quoted article can share a number, and
    # subtracting it would let the quoted one stand in for a host the scan
    # missed, reporting full coverage over the omission the gate exists for.
    # Position rather than the quote mask, because `quoted_amendment` is also
    # set by replacement lead-ins outside any quote and the declared-marker lane
    # deliberately runs unmasked.
    quoted = frozenset(
        _marker_line_start(text, a.char_offset) for a in anchors if a.quoted_amendment
    )
    expected = _marker_numbers(text, config, doctype, kind, quoted)
    ratio = len(captured) / len(expected) if expected else None
    return AnchorCoverage(
        kind=kind,
        ratio=ratio,
        captured=frozenset(captured),
        expected=frozenset(expected),
        masked=_masked_marker_count(text, config, doctype) if with_masked else 0,
        unclosed=_unclosed_quote_spans(text, config.code if config else "") if with_masked else 0,
    )


def anchor_summary(anchors: Iterable[StructuralAnchor]) -> dict[str, int]:
    """Aggregate the document's own anchors by kind, in document order of first
    occurrence.

    Quoted amendment anchors are excluded as everywhere else: they borrow another
    act's structure, `_assign_eids` never emits them, and counting them describes a
    document that was never assembled.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for a in anchors:
        if a.quoted_amendment:
            continue
        counts[a.kind] += 1
        first_seen.setdefault(a.kind, a.char_offset)
    return dict(sorted(counts.items(), key=lambda kv: first_seen[kv[0]]))


# "schedule" is top-level so trailing annex material gets its own window
# instead of being swallowed into the last article's body.
_TOP_LEVEL_KINDS = frozenset({"article", "section", "paragraph", "schedule"})


def windows_from_anchors(
    text: str,
    anchors: list[StructuralAnchor],
    *,
    target_size: int = 6_000,
    overlap: int = 400,
    min_per_window: int = 3,
    max_per_window: int = 8,
) -> list[Window]:
    """Greedy-group consecutive top-level anchors into windows.

    Each window owns a contiguous run; its ``text`` spans from its first anchor's
    offset to the next window's, extended both sides by ``overlap`` so the LLM sees
    where the window begins and ends.

    Sizing starts at ``min_per_window`` and keeps adding while the body is under
    ``target_size`` and the count under ``max_per_window``. Returns [] when there are
    no top-level anchors, and callers then fall back to coarser splitting.
    """
    top = [a for a in anchors if a.kind in _TOP_LEVEL_KINDS and not a.quoted_amendment]
    if not top:
        return []
    top.sort(key=lambda a: a.char_offset)

    groups: list[list[StructuralAnchor]] = []
    current: list[StructuralAnchor] = []
    current_start_offset = top[0].char_offset
    for a in top:
        if not current:
            current.append(a)
            current_start_offset = a.char_offset
            continue
        prospective_end = a.char_offset
        prospective_size = prospective_end - current_start_offset
        if len(current) >= max_per_window or (
            len(current) >= min_per_window and prospective_size >= target_size
        ):
            groups.append(current)
            current = [a]
            current_start_offset = a.char_offset
        else:
            current.append(a)
    if current:
        groups.append(current)

    # Merge a small tail group back into the previous one so every window
    # carries at least ``min_per_window`` anchors. Only fires when the merge
    # doesn't push the previous window past 2x max_per_window, protects
    # against pathological all-tiny-tails inputs.
    if (
        len(groups) >= 2
        and len(groups[-1]) < min_per_window
        and len(groups[-2]) + len(groups[-1]) <= max_per_window * 2
    ):
        groups[-2].extend(groups[-1])
        groups.pop()

    ordered = sorted((a for a in anchors if not a.quoted_amendment), key=lambda a: a.char_offset)
    offsets = [a.char_offset for a in ordered]
    by_eid = {a.akn_eid: a for a in ordered if a.akn_eid}
    roots = {a.akn_eid for a in top if a.akn_eid}
    windows: list[Window] = []
    for i, group in enumerate(groups):
        body_start_abs = group[0].char_offset
        body_end_abs = groups[i + 1][0].char_offset if i + 1 < len(groups) else len(text)
        window_start = max(0, body_start_abs - overlap)
        window_end = min(len(text), body_end_abs + overlap)
        slice_text = text[window_start:window_end]
        body_start = body_start_abs - window_start
        body_end = body_end_abs - window_start
        # Grouping stays coarse; each body-fill target also needs its owned leaves.
        owned: list[StructuralAnchor] = []
        for anchor in ordered[
            bisect_left(offsets, body_start_abs) : bisect_left(offsets, body_end_abs)
        ]:
            parent = anchor.parent_eid
            seen: set[str] = set()
            while parent and parent not in roots and parent in by_eid and parent not in seen:
                seen.add(parent)
                parent = by_eid[parent].parent_eid
            if anchor.kind in _TOP_LEVEL_KINDS or parent in roots:
                owned.append(anchor)
        windows.append(
            Window(
                anchors=tuple(owned),
                text=slice_text,
                body_start=body_start,
                body_end=body_end,
            )
        )
    return windows


@lru_cache(maxsize=64)
def cached_regex(country: str, doctype: str) -> re.Pattern[str]:
    """Memo for the hot path, compiling the alternation per call is
    measurable on large corpora. Cached per ``(country, doctype)``.

    A named country must have a config and raises without one, since the
    alternation would otherwise be English. An empty country is a caller
    declaring it has none, and builds the alternation from the builtins.
    """
    config = load_config(country) if country else None
    return build_anchor_regex(config, doctype)


__all__ = [
    "StructuralAnchor",
    "Window",
    "AnchorCoverage",
    "anchor_coverage",
    "anchor_summary",
    "build_anchor_regex",
    "cached_regex",
    "scan_anchors",
    "windows_from_anchors",
]
