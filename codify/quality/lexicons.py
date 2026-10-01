"""Word lists for legibility scoring, keyed by language. Legibility asks whether the
letter forms survived, which closed-class words answer, and those are a property of
the language rather than the script: 206 shipped configs are Latin, spanning English,
French, Spanish and Bahasa, and one list cannot serve them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

__all__ = [
    "ARABIC_WORDS",
    "ENGLISH_WORDS",
    "BAHASA_WORDS",
    "WordList",
    "languages_with_lists",
    "register",
    "words_for",
]


@dataclass(frozen=True)
class WordList:
    """Closed-class and legal-content words for one language. `function_words` separate prose
    from lists; `lexicon_words` separate a clean list from damaged text, since a terse
    document still names the instruments it is about.
    """

    language: str
    function_words: frozenset[str] = field(default_factory=frozenset)
    lexicon_words: frozenset[str] = field(default_factory=frozenset)
    # Letters of this language's script. Tokenising on them rather than any alphabetic
    # run keeps a bilingual page scored on the language being judged; the default
    # accepts any letter, right for a Latin-script language on a Latin-script page.
    letter_re: re.Pattern[str] | None = None
    # Floors for `verdict_from_rates`, measured per language and not transferable:
    # clean Arabic prose runs 91 to 176 per 1,000 where English runs 213 to 433. A
    # None lexicon floor means the content-noun tie-break is unusable here, so a
    # low-prose document reports no verdict rather than a wrong one.
    prose_floor: float | None = None
    lexicon_floor: float | None = None

    def __post_init__(self) -> None:
        # A single-character entry cannot evidence surviving letter forms:
        # letter-spaced text is nothing but bare glyphs, and one such entry
        # scores that noise as prose.
        singles = {w for w in self.function_words | self.lexicon_words if len(w) < 2}
        if singles:
            raise ValueError(
                f"{self.language}: single-character word list entries {sorted(singles)}"
            )

    def is_letter(self, ch: str) -> bool:
        return ch.isalpha() if self.letter_re is None else self.letter_re.match(ch) is not None


# Latin letters including the accented range, so a Cyrillic or Arabic run on a
# bilingual page is a separator rather than a token that can only deflate.
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")

_REGISTRY: dict[str, WordList] = {}


def register(words: WordList) -> WordList:
    _REGISTRY[words.language] = words
    return words


# Arabic, moved verbatim from the script pack so its calibration still holds.
ARABIC_WORDS = register(
    WordList(
        language="ara",
        letter_re=re.compile(r"[\u0621-\u064a\u066e-\u06d3]"),
        # Calibrated on tests/fixtures/legibility/marked.jsonl: clean prose
        # measured 91 to 176 per 1,000.
        prose_floor=80.0,
        lexicon_floor=12.0,
        function_words=frozenset(
            {
                "من",
                "في",
                "على",
                "إلى",
                "عن",
                "أن",
                "أو",
                "لا",
                "ما",
                "كل",
                "بين",
                "بعد",
                "عند",
                "غير",
                "قد",
                "كان",
                "يكون",
                "هذا",
                "هذه",
                "التي",
                "الذي",
                "له",
                "لها",
                "به",
                "مع",
                "وفي",
                "ومن",
                "وعلى",
            }
        ),
        lexicon_words=frozenset(
            {
                "قانون",
                "مادة",
                "المادة",
                "قرار",
                "مرسوم",
                "نظام",
                "لائحة",
                "أحكام",
                "مجلس",
                "الوزراء",
                "وزير",
                "الرئيس",
                "السلطة",
                "رقم",
                "لسنة",
                "بشأن",
                "تعديل",
            }
        ),
    )
)

# Bahasa (ind). Particles and auxiliaries that survive only if the letters did,
# plus the nouns any instrument names about itself.
BAHASA_WORDS = register(
    WordList(
        language="ind",
        letter_re=_LATIN_RE,
        # Measured over 61 documents: function words p5 149.5, median 181.6;
        # lexicon median 56.2, with 5 of 64 under the floor.
        prose_floor=80.0,
        lexicon_floor=12.0,
        function_words=frozenset(
            {
                "yang",
                "dan",
                "dengan",
                "untuk",
                "dari",
                "pada",
                "dalam",
                "atau",
                "oleh",
                "ini",
                "itu",
                "tidak",
                "adalah",
                "akan",
                "dapat",
                "telah",
                "sebagai",
                "kepada",
                "atas",
                "secara",
                "tersebut",
                "bahwa",
                "jika",
                "karena",
                "serta",
                "juga",
                "agar",
                "maka",
            }
        ),
        lexicon_words=frozenset(
            {
                "undang",
                "peraturan",
                "pasal",
                "ayat",
                "menteri",
                "pemerintah",
                "negara",
                "hukum",
                "ketentuan",
                "keputusan",
                "presiden",
                "daerah",
                "izin",
                "usaha",
                "kawasan",
                "hutan",
                "wajib",
                "dimaksud",
            }
        ),
    )
)

# English.
ENGLISH_WORDS = register(
    WordList(
        language="eng",
        letter_re=_LATIN_RE,
        # Built from document frequency over held statutes rather than guessed: they
        # write "provisions", not "provision".
        prose_floor=80.0,
        lexicon_floor=12.0,
        function_words=frozenset(
            {
                "the",
                "of",
                "and",
                "to",
                "in",
                "that",
                "is",
                "for",
                "be",
                "or",
                "as",
                "by",
                "with",
                "any",
                "this",
                "which",
                "shall",
                "not",
                "on",
                "from",
                "at",
                "an",
                "may",
                "such",
                "other",
                "if",
                "under",
            }
        ),
        lexicon_words=frozenset(
            {
                "act",
                "section",
                "provisions",
                "regulations",
                "rules",
                "public",
                "government",
                "republic",
                "approval",
                "effect",
                "hereby",
                "amended",
                "authorized",
                "provided",
                "necessary",
                "general",
                "national",
                "established",
                "authority",
                "order",
            }
        ),
    )
)


def extend(words: WordList | None, extra: Iterable[str]) -> WordList | None:
    """A copy of `words` carrying the jurisdiction's own legal vocabulary, which
    a language list cannot hold because those terms name one jurisdiction."""
    terms = frozenset(e for e in extra if len(e) > 1)
    if words is None or not terms:
        return words
    return replace(words, lexicon_words=words.lexicon_words | terms)


def words_for(language: str | None) -> WordList | None:
    """The list a language selects, or None when none is registered.

    None means the text cannot be scored, which is the honest state for a
    language nobody has written a list for; it is not a score of zero."""
    return _REGISTRY.get((language or "").strip().lower())


def languages_with_lists() -> frozenset[str]:
    return frozenset(_REGISTRY)
