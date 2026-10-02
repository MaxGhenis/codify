"""Numeric tokens become ⟨N###⟩ sentinels before the text reaches the LLM, then decode
to a canonical target surface, so a lost digit names the token it lost. Bare digits,
article and statute refs, money and dates are covered; anything else stays prose.
When in doubt encode: a false positive only raises the audit's sentinel count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from codify.lang import NON_ASCII_DIGIT_CLASS, normalise_digits, to_iso639_3

# Angle quotation marks (U+27E8/U+27E9) are absent from legislative Arabic and
# Hebrew, so the LLM treats them as opaque and rarely paraphrases them.
# Single-guillemet is the fallback when a source ships the primary marker.
_PRIMARY_OPEN = "⟨"
_PRIMARY_CLOSE = "⟩"
_FALLBACK_OPEN = "‹"
_FALLBACK_CLOSE = "›"


TokenKind = Literal[
    "digits",  # bare digit sequence
    "article_ref",  # "Article (5)", "مادة (5)"
    "article_para_ref",  # "المادة (4/9)" — article 4, paragraph 9
    "para_of_article_ref",  # "الفقرة (3) من المادة (20)"
    "statute_ref",  # "Law No. 7 of 1999"
    "money",  # "1000 Dinars"
    "date",  # "3/2/2000"
    "unknown",  # numeric-adjacent but unclassified
]

# Compressed citations decode to target-language wording where available. A
# numeric-shape fallback preserves the citation when no template exists.
_CITATION_TEMPLATES: dict[str, dict[str, str]] = {
    "eng": {
        "article_para_ref": "Article {article} paragraph {paragraph}",
        "para_of_article_ref": "Paragraph {paragraph} of Article {article}",
    },
    # Israeli legal typography cites a subsection in parentheses after the
    # section number. Wording reviewed against Hebrew drafting convention is
    # still owed; the shape is what matters for the numerals to survive.
    "heb": {
        "article_para_ref": "סעיף {article}({paragraph})",
        "para_of_article_ref": "פסקה {paragraph} לסעיף {article}",
    },
}


@dataclass(frozen=True)
class NumericToken:
    """One numeric token extracted from a source body. ``surface`` is the source text
    replaced by the sentinel; ``expected_target_surface`` is what it decodes to,
    digits folded to ASCII and the source label stripped, because the LLM
    translates the label around the sentinel.
    """

    sentinel_id: str  # e.g. "N001"
    surface: str
    kind: TokenKind
    expected_target_surface: str
    start: int  # position in original source text
    end: int
    # Whether any citation label sits within reach before this token in the
    # source. Decoding strips a model-supplied noun only when none does.
    near_a_label: bool = True


@dataclass
class NumericManifest:
    """The set of numeric tokens extracted from one source body,
    together with the choice of sentinel delimiters."""

    tokens: list[NumericToken] = field(default_factory=list)
    open_char: str = _PRIMARY_OPEN
    close_char: str = _PRIMARY_CLOSE

    def sentinel_for(self, tok: NumericToken) -> str:
        return f"{self.open_char}{tok.sentinel_id}{self.close_char}"


# Extractor patterns. Order matters: specific patterns run first so "Law No. 7
# of 1999" consumes its inner digits before the bare-digit sweep sees them.

_DIGIT_CHARS = r"0-9" + NON_ASCII_DIGIT_CLASS

# Article/section/paragraph refs, singular only. A plural label governs a list,
# so claiming it into the first member ("Articles 9 and 20") swallows a noun
# belonging to all of them; left in prose the model translates it once.
_LABEL_WORDS = (
    r"Article|Section|Paragraph|Chapter"
    r"|مادة|المادة|فصل|الفصل|بند|البند|فقرة|الفقرة"
    r"|סעיף|הסעיף|פסקה|הפסקה"
)
# Arabic may attach prepositions and the definite article to citation labels;
# include those prefixes in the sentinel span.
_LABEL_PROCLITICS = r"(?:[وفبك]?لل|[وفبكل]?ال|[وفبكل])?"


# Parentheses are claimed as a balanced pair so closing punctuation is not
# stranded outside the sentinel.
def _paren_or_bare(*groups: str) -> str:
    """Match ``groups`` inside a balanced paren pair, or bare without either."""
    inner = r"\s*/\s*".join(rf"({g})" for g in groups)
    return rf"(?:\s*\(\s*{inner}\s*\)|\s*{inner})"


_NUM = rf"[{_DIGIT_CHARS}]+"
_ARTICLE_REF = re.compile(
    rf"{_LABEL_PROCLITICS}(?:{_LABEL_WORDS}){_paren_or_bare(_NUM)}",
    re.UNICODE,
)

# Compressed citations, claimed as one span including the label so the model
# cannot reinterpret the internals. The plain pattern stops at the slash,
# leaving `/9` to the bare-digit sweep: `المادة (4/9)` became "44/9)".
_ARTICLE_PARA_REF = re.compile(
    rf"{_LABEL_PROCLITICS}(?:المادة|مادة){_paren_or_bare(_NUM, _NUM)}",
    re.UNICODE,
)
_PARA_OF_ARTICLE_REF = re.compile(
    rf"{_LABEL_PROCLITICS}(?:الفقرة|فقرة){_paren_or_bare(_NUM)}\s*من\s+"
    rf"{_LABEL_PROCLITICS}(?:المادة|مادة){_paren_or_bare(_NUM)}",
    re.UNICODE,
)

# Statute references such as "Law No. 7 of 1999" and Arabic equivalents.
_STATUTE_WORDS_EN = r"Law|Act|Decree(?:-|\s+)?Law|Regulation|Decree|Order"
# Keep the closing-paren spacing constrained so the year suffix remains matchable.
_STATUTE_REF_EN = re.compile(
    rf"(?:{_STATUTE_WORDS_EN})\s+No\.?{_paren_or_bare(_NUM)}"
    rf"(?:\s+(?:of|for)\s+(\d{{4}}))?",
    re.UNICODE | re.IGNORECASE,
)
_STATUTE_REF_AR = re.compile(
    rf"{_LABEL_PROCLITICS}(?:قانون|قرار\s+بقانون|النظام|نظام|القانون|القرار)\s+رقم"
    rf"{_paren_or_bare(_NUM)}(?:\s+لسنة\s+(\d{{4}}))?",
    re.UNICODE,
)

# Monetary amounts: digits adjacent to a currency word (Arabic + English).
_CURRENCY_WORDS = (
    r"(?:[A-Z][a-z]+\s+)?"
    r"(?:Dinars?|Dirhams?|Riyals?|Dollars?|Shekels?|Euros?|Pounds?|Liras?|Lire|Francs?)"
    r"|دينار(?:اً|ا|ين)?|دنانير|درهم(?:اً|ا)?|دراهم|ريال(?:اً|ا|ات)?"
    r"|دولار(?:اً|ا|ات)?|شيقل(?:اً|ا)?|جنيه(?:اً|ا|ات)?|ليرة|ليرات|يورو|فرنك(?:اً|ا|ات)?"
    r"|שקל(?:ים)?|דינר|דולר|יורו|לירה"
)
_MONEY = re.compile(
    rf"([{_DIGIT_CHARS}]+(?:[,.][{_DIGIT_CHARS}]+)*)\s*(?:{_CURRENCY_WORDS})",
    re.UNICODE,
)

# Dates: DD/MM/YYYY, DD-MM-YYYY, DD/MM, YYYY.
_DATE = re.compile(
    rf"[{_DIGIT_CHARS}]{{1,4}}\s*[/-]\s*[{_DIGIT_CHARS}]{{1,2}}"
    rf"(?:\s*[/-]\s*[{_DIGIT_CHARS}]{{1,4}})?",
    re.UNICODE,
)

# Bare digit sequences: last-pass sweep after the labelled forms have
# consumed their runs.
_BARE_DIGITS = re.compile(rf"[{_DIGIT_CHARS}]+", re.UNICODE)


# --- Delimiter selection --------------------------------------------------


def _pick_delimiters(source_text: str) -> tuple[str, str]:
    """Return the pair of sentinel delimiters that don't clash with the
    source. Preferred is U+27E8 / U+27E9; falls back to U+2039 / U+203A."""
    if _PRIMARY_OPEN not in source_text and _PRIMARY_CLOSE not in source_text:
        return _PRIMARY_OPEN, _PRIMARY_CLOSE
    return _FALLBACK_OPEN, _FALLBACK_CLOSE


# --- Extraction pipeline --------------------------------------------------

_LABELLED_PATTERNS: tuple[tuple[re.Pattern[str], TokenKind], ...] = (
    # Compound citations first: they must claim the whole span before the
    # plain article-ref pattern splits one at the slash.
    (_PARA_OF_ARTICLE_REF, "para_of_article_ref"),
    (_ARTICLE_PARA_REF, "article_para_ref"),
    (_ARTICLE_REF, "article_ref"),
    (_STATUTE_REF_AR, "statute_ref"),
    (_STATUTE_REF_EN, "statute_ref"),
    (_MONEY, "money"),
    (_DATE, "date"),
)


def extract_tokens(source_text: str, *, start_id: int = 0) -> NumericManifest:
    """Extract numeric tokens from a source body in reading order. Labelled patterns
    run first so their inner digits are claimed, then the bare-digit sweep takes
    what is left. ``start_id`` lets one document allocate unique sentinel IDs
    across many bodies.
    """
    open_c, close_c = _pick_delimiters(source_text)

    # Collect (start, end, surface, kind) candidates from labelled patterns
    # first; each match records its span so subsequent patterns and the
    # bare-digit sweep skip already-claimed characters.
    candidates: list[tuple[int, int, str, TokenKind]] = []
    consumed: list[tuple[int, int]] = []

    def _claim(start: int, end: int, surface: str, kind: TokenKind) -> None:
        if any(s <= start and e >= end for s, e in consumed):
            return
        candidates.append((start, end, surface, kind))
        consumed.append((start, end))

    for pat, kind in _LABELLED_PATTERNS:
        for m in pat.finditer(source_text):
            _claim(m.start(), m.end(), m.group(0), kind)
    for m in _BARE_DIGITS.finditer(source_text):
        _claim(m.start(), m.end(), m.group(0), "digits")

    candidates.sort(key=lambda c: c[0])
    tokens = [
        NumericToken(
            sentinel_id=f"N{start_id + i:03d}",
            surface=surface,
            kind=kind,
            expected_target_surface=_target_surface(surface, kind),
            start=start,
            end=end,
            near_a_label=_near_a_label(source_text, start),
        )
        for i, (start, end, surface, kind) in enumerate(candidates)
    ]
    return NumericManifest(tokens=tokens, open_char=open_c, close_char=close_c)


# How far back a governing label may sit and still license a noun. Wide on
# purpose: ranges put connectives between the label and later members, and
# looking too far leaves a doubled noun where too near deletes a citation.
_LABEL_REACH = 48
# A trailing letter is refused so a plural does not match through its singular
# ("Article" inside "Articles", "סעיף" inside its plural). A plural governs the
# whole list, so its members do not each earn a noun.
_ANY_LABEL = re.compile(rf"(?:{_LABEL_WORDS})(?![^\W\d_])", re.UNICODE)


def _near_a_label(source_text: str, start: int) -> bool:
    """Whether a citation label sits within reach before the token at ``start``."""
    return _ANY_LABEL.search(source_text[max(0, start - _LABEL_REACH) : start]) is not None


def _target_surface(surface: str, kind: TokenKind) -> str:
    """Canonical target surface: ASCII-fold bare digits and dates, and for labelled
    refs and money keep every digit group joined by a slash. Keeping only digits
    fused `Law No. 8 of 2014` to `82014`, and adjacent sentinels then decoded to
    `8201482014`.
    """
    ascii_form = normalise_digits(surface)
    if kind in ("digits", "date"):
        return ascii_form
    if kind in ("article_para_ref", "para_of_article_ref"):
        # Language-neutral fallback; `decode_sentinels` renders the wording
        # when it knows the target language.
        return "/".join(re.findall(r"\d+", ascii_form))
    # Keep intra-group separators so "1,000" and "1.5" survive, and join
    # distinct groups with `/`: `Law No. 8 of 2014` is `8/2014`, not `82014`.
    digit_groups = re.findall(r"\d+(?:[.,]\d+)*", ascii_form)
    if len(digit_groups) >= 2:
        return "/".join(digit_groups)
    if digit_groups:
        return str(digit_groups[0])
    return ascii_form


# --- Encode / decode ------------------------------------------------------


def encode_sentinels(source_text: str, manifest: NumericManifest) -> str:
    """Replace every source token with its sentinel, right-to-left so
    indices stay valid. Returns the sentinel-marked text handed to the LLM.
    """
    if not manifest.tokens:
        return source_text
    out = source_text
    # Process in reverse position order so earlier indices don't shift.
    for tok in sorted(manifest.tokens, key=lambda t: t.start, reverse=True):
        sentinel = manifest.sentinel_for(tok)
        out = out[: tok.start] + sentinel + out[tok.end :]
    return out


def citation_surface(tok: NumericToken, target_language: str | None) -> str:
    """Target-language wording for a compound citation, or its numeric shape. Layout
    carries the meaning, so it decodes to words. An unknown target language or an
    unexpected digit count falls back to the numeric shape: an unworded citation
    is recoverable, a mis-worded one is not.
    """
    if tok.kind not in ("article_para_ref", "para_of_article_ref"):
        return tok.expected_target_surface
    parts = tok.expected_target_surface.split("/")
    if len(parts) != 2 or not target_language:
        return tok.expected_target_surface
    try:
        iso = to_iso639_3(target_language)
    except ValueError:
        return tok.expected_target_surface
    template = _CITATION_TEMPLATES.get(iso, {}).get(tok.kind)
    if not template:
        return tok.expected_target_surface
    first, second = parts
    if tok.kind == "article_para_ref":
        article, paragraph = first, second
    else:
        paragraph, article = first, second
    return template.format(article=article, paragraph=paragraph)


# Words a compound-citation surface supplies for itself. English only: Hebrew
# glues prepositions to the noun (בסעיף), so stripping the word takes the
# preposition with it. Hebrew needs its own prefix rule.
_CITATION_NOUNS: dict[str, frozenset[str]] = {
    "eng": frozenset({"article", "paragraph", "section", "clause"}),
}
_TRAILING_WORD = re.compile(r"(\w+)\s*$", re.UNICODE)


def strip_duplicated_citation_noun(before: str, surface: str, target_language: str | None) -> str:
    """Return ``before`` without a trailing citation noun the surface repeats. The
    extractor swallows the source noun into the sentinel, so the model supplies
    its own and decoding says it twice ("Article Paragraph 3 of Article 20"). Safe
    only because the surface carries the noun; an ordinary "Article ⟨N⟩" is left.
    """
    if not target_language:
        return before
    try:
        nouns = _CITATION_NOUNS.get(to_iso639_3(target_language), frozenset())
    except ValueError:
        return before
    match = _TRAILING_WORD.search(before)
    if match is None:
        return before
    word = match.group(1)
    if word.lower() not in nouns or word.lower() not in surface.lower():
        return before
    return before[: match.start(1)]


def strip_invented_citation_noun(before: str, target_language: str | None) -> str:
    """Return ``before`` without a citation noun no source label licenses. Called
    only when no label sits within reach, so the noun came from the model: an OCR
    page-footer `77` became "Article 77" in operative text.
    """
    if not target_language:
        return before
    try:
        nouns = _CITATION_NOUNS.get(to_iso639_3(target_language), frozenset())
    except ValueError:
        return before
    match = _TRAILING_WORD.search(before)
    if match is None or match.group(1).lower() not in nouns:
        return before
    return before[: match.start(1)]


def decode_sentinels(
    text: str,
    manifest: NumericManifest,
    *,
    target_language: str | None = None,
) -> tuple[str, list[str]]:
    """Substitute each sentinel with its expected target surface, returning
    ``(decoded_text, missing_ids)``. Extra sentinels such as a hallucinated
    ⟨N999⟩ are left in place for the audit's regex sweep to flag.
    """
    if not manifest.tokens:
        return text, []
    missing: list[str] = []
    out = text
    for tok in manifest.tokens:
        sentinel = manifest.sentinel_for(tok)
        pos = out.find(sentinel)
        if pos < 0:
            missing.append(tok.sentinel_id)
            continue
        # A thin space when the preceding character is an already-decoded
        # digit, so two adjacent decoded surfaces do not fuse.
        replacement = citation_surface(tok, target_language)
        # Only pad when the preceding digit is contiguous with the sentinel:
        # a digit on the prior word once produced a bogus space, "1 1.".
        lookback = out[max(0, pos - 3) : pos]
        _digit_word_before = (
            pos > 0 and out[pos - 1].isdigit() and not any(c.isspace() for c in lookback)
        )
        if _digit_word_before and replacement[:1].isdigit():
            replacement = " " + replacement
        head = out[:pos]
        if tok.kind in ("article_para_ref", "para_of_article_ref", "article_ref"):
            head = strip_duplicated_citation_noun(head, replacement, target_language)
        elif tok.kind == "digits" and not tok.near_a_label:
            head = strip_invented_citation_noun(head, target_language)
        out = head + replacement + out[pos + len(sentinel) :]
    return out, missing


def filter_real_missing(
    manifest: NumericManifest, candidate_missing: list[str], raw_joined: str
) -> list[str]:
    """Of the per-line missing-sentinel IDs, those whose marker is genuinely absent from
    ``raw_joined``, which must be the LLM output BEFORE per-line decode: decoding
    consumes sentinels, so a decoded string reports every candidate missing. Raw text
    also spares the case where the LLM moved a sentinel to another line. Deduplicated.
    """
    by_id = {t.sentinel_id: t for t in manifest.tokens}
    seen: set[str] = set()
    real: list[str] = []
    for mid in candidate_missing:
        if mid in seen:
            continue
        seen.add(mid)
        tok = by_id.get(mid)
        if tok is None:
            continue
        if manifest.sentinel_for(tok) not in raw_joined:
            real.append(mid)
    return real


# --- Post-hoc regex sweep -------------------------------------------------

_SENTINEL_RE = re.compile(
    rf"[{_PRIMARY_OPEN}{_FALLBACK_OPEN}]N\d{{3,}}[{_PRIMARY_CLOSE}{_FALLBACK_CLOSE}]"
)


def find_stray_sentinels(text: str) -> list[str]:
    """Return every sentinel marker still present in a string. Called
    after decode to catch sentinels that the manifest didn't know about
    (e.g. LLM hallucinated an extra one)."""
    return _SENTINEL_RE.findall(text)


__all__ = [
    "NumericToken",
    "NumericManifest",
    "TokenKind",
    "extract_tokens",
    "encode_sentinels",
    "decode_sentinels",
    "filter_real_missing",
    "find_stray_sentinels",
]
