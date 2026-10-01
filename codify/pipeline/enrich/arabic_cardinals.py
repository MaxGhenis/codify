"""Arabic cardinal-number words: parser + amount-in-words consistency check.

Statutory money amounts are usually written twice, numeral and words:
``(50,000) خمسون ألف دينار زرزوري``. OCR corrupts the numeral far more often
than the words (a misread digit is silent; a misread word is gibberish), so
when the two readings disagree the words are the stronger signal and the
disagreement is worth a warning. ``parse_cardinal_words`` reads the words;
``find_money_word_mismatches`` pairs each parenthesised numeral with the
adjacent amount-in-words (currency-anchored, both orders) and reports pairs
that disagree. Detector only: nothing is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from codify.lang import normalise_digits
from codify.pipeline.enrich.arabic_normalise import fold_arabic_for_match

# Vocabulary is keyed on fold_arabic_for_match output (hamza-alef and
# ta-marbuta folded), so ألف and مائة arrive as الف and مائه.
_UNITS: dict[str, int] = {
    "واحد": 1,
    "واحده": 1,
    "احد": 1,
    "احدي": 1,
    "اثنان": 2,
    "اثنين": 2,
    "اثنتان": 2,
    "اثنتين": 2,
    "اثنا": 2,
    "اثني": 2,
    "ثلاثه": 3,
    "ثلاث": 3,
    "اربعه": 4,
    "اربع": 4,
    "خمسه": 5,
    "خمس": 5,
    "سته": 6,
    "ست": 6,
    "سبعه": 7,
    "سبع": 7,
    "ثمانيه": 8,
    "ثماني": 8,
    "ثمان": 8,
    "تسعه": 9,
    "تسع": 9,
}
_TENS: dict[str, int] = {
    "عشرون": 20,
    "عشرين": 20,
    "ثلاثون": 30,
    "ثلاثين": 30,
    "اربعون": 40,
    "اربعين": 40,
    "خمسون": 50,
    "خمسين": 50,
    "ستون": 60,
    "ستين": 60,
    "سبعون": 70,
    "سبعين": 70,
    "ثمانون": 80,
    "ثمانين": 80,
    "تسعون": 90,
    "تسعين": 90,
}
_TEN_WORD = frozenset({"عشر", "عشره"})  # compound-teen second word
_HUNDRED_WORD = frozenset({"مائه", "مئه", "مايه"})
_TWO_HUNDRED = frozenset({"مائتان", "مائتين", "مئتان", "مئتين"})
# Scale words with an inherent multiplier (dual forms carry their own 2).
_SCALES: dict[str, tuple[int, int]] = {
    "الف": (1_000, 1),
    "الفا": (1_000, 1),
    "الاف": (1_000, 1),
    "الفان": (1_000, 2),
    "الفين": (1_000, 2),
    "مليون": (1_000_000, 1),
    "مليونا": (1_000_000, 1),
    "ملايين": (1_000_000, 1),
    "مليونان": (1_000_000, 2),
    "مليونين": (1_000_000, 2),
}


def _token_values(tokens: list[str]) -> list[tuple[str, int]] | None:
    """Classify each folded token as (kind, value); None if any is unknown.
    Kinds: unit, ten, teen10, hundred, scale:<mult>."""
    out: list[tuple[str, int]] = []
    for tok in tokens:
        if tok in _UNITS:
            out.append(("unit", _UNITS[tok]))
        elif tok in _TENS:
            out.append(("ten", _TENS[tok]))
        elif tok in _TEN_WORD:
            out.append(("teen10", 10))
        elif tok in _HUNDRED_WORD:
            out.append(("hundred", 100))
        elif tok in _TWO_HUNDRED:
            out.append(("hundred", 200))
        elif tok in _SCALES:
            scale, mult = _SCALES[tok]
            out.append((f"scale:{mult}", scale))
        else:
            # Fused hundreds: ثلاثمائه / خمسمئه (both hundred spellings).
            for suffix in ("مائه", "مئه"):
                if tok.endswith(suffix) and tok[: -len(suffix)] in _UNITS:
                    out.append(("hundred", _UNITS[tok[: -len(suffix)]] * 100))
                    break
            else:
                return None
    return out


def parse_cardinal_words(words: str) -> int | None:
    """Parse an Arabic cardinal-words phrase to an int (1 to 999,999,999).

    Accepts orthographic variants (hamza forms, ta-marbuta, و-conjoined
    tokens), compound teens (``احد عشر``), fused hundreds (``خمسمائة``)
    and dual scales (``الفان``). Returns None when any token falls outside
    the number vocabulary or the total leaves the supported range."""
    folded = fold_arabic_for_match(words)
    tokens = [t.lstrip("و") for t in folded.split() if t.lstrip("و")]
    if not tokens:
        return None
    classified = _token_values(tokens)
    if classified is None:
        return None
    total = 0
    group = 0
    i = 0
    while i < len(classified):
        kind, value = classified[i]
        if kind == "unit":
            if i + 1 < len(classified) and classified[i + 1][0] == "teen10":
                group += 10 + value
                i += 2
                continue
            group += value
        elif kind == "ten":
            group += value
        elif kind == "teen10":
            group += value  # bare عشرة = 10
        elif kind == "hundred":
            group += value
        else:  # scale
            mult = int(kind.split(":", 1)[1])
            total += (group or mult) * value
            group = 0
        i += 1
    result = total + group
    if not 1 <= result <= 999_999_999:
        return None
    return result


# Folded spellings (see fold_arabic_for_match): ليرة is stored as ليره.
_CURRENCY = frozenset(
    {"دينار", "دينارا", "دنانير", "درهم", "درهما", "دراهم", "ريال", "ريالا", "ريالات"}
    | {"دولار", "دولارا", "دولارات", "جنيه", "جنيها", "جنيهات", "ليره", "ليرات"}
    | {"يورو", "فرنك", "فرنكا", "فرنكات", "شيكل", "شيقل", "شواقل"}
)

_PAREN_NUMERAL_RE = re.compile(r"\(\s*(?P<numeral>[0-9٠-٩۰-۹][0-9٠-٩۰-۹,.،٬\s]*)\s*\)")

# How far around the numeral we look for the amount-in-words.
_WINDOW_TOKENS = 8


@dataclass(frozen=True)
class MoneyWordMismatch:
    """A parenthesised numeral disagreeing with its adjacent words-amount."""

    numeral_text: str
    numeral_value: int
    words_text: str
    words_value: int
    offset: int


def _numeral_value(raw: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", normalise_digits(raw))
    if not digits:
        return None
    return int(digits)


def _words_after(text: str, start: int) -> tuple[str, int | None]:
    """Amount-in-words directly after a numeral: tokens up to a currency
    word, all within the window. ("(50,000) خمسون ألف دينار")."""
    tail = text[start:]
    tokens = tail.split()[: _WINDOW_TOKENS + 1]
    for n, tok in enumerate(tokens):
        if fold_arabic_for_match(tok) in _CURRENCY:
            phrase = " ".join(tokens[:n])
            return phrase, parse_cardinal_words(phrase) if phrase else None
    return "", None


def _words_before(text: str, end: int) -> tuple[str, int | None]:
    """Amount-in-words directly before a numeral, currency word adjacent
    to the paren: ("خمسون ألف دينار زرزوري (50,000)"). Walks back over the
    currency (plus at most one adjective like وطني), then collects the
    trailing run of number-vocabulary tokens."""
    head_tokens = text[:end].split()[-(_WINDOW_TOKENS + 2) :]
    k = len(head_tokens)
    seen_currency = False
    for back in range(1, min(3, k + 1)):
        if fold_arabic_for_match(head_tokens[k - back]) in _CURRENCY:
            seen_currency = True
            k -= back
            break
    if not seen_currency:
        return "", None
    run: list[str] = []
    for tok in reversed(head_tokens[:k]):
        if _token_values([fold_arabic_for_match(tok).lstrip("و")]) is None:
            break
        run.insert(0, tok)
    phrase = " ".join(run)
    return phrase, parse_cardinal_words(phrase) if phrase else None


def find_money_word_mismatches(text: str) -> list[MoneyWordMismatch]:
    """All numeral/words money pairs in ``text`` whose readings disagree.

    Currency-anchored: a parenthesised numeral only pairs with words when a
    currency word sits in the same neighbourhood, so cross-reference numerals
    ("المادة (5)") never enter. Pairs whose words fail to parse are skipped
    (unknown vocabulary is not evidence of a mismatch)."""
    out: list[MoneyWordMismatch] = []
    for m in _PAREN_NUMERAL_RE.finditer(text):
        numeral = _numeral_value(m.group("numeral"))
        if numeral is None:
            continue
        words, value = _words_after(text, m.end())
        if value is None:
            words, value = _words_before(text, m.start())
        if value is None:
            continue
        # "(50) ألف دينار" is multiplier notation (numeral × scale), not a
        # second reading of the amount; a words phrase opening on a bare
        # scale word never disagrees with its numeral.
        first_word = fold_arabic_for_match(words).split()[0].lstrip("و") if words.strip() else ""
        if first_word in _SCALES:
            continue
        if value != numeral:
            out.append(
                MoneyWordMismatch(
                    numeral_text=m.group("numeral").strip(),
                    numeral_value=numeral,
                    words_text=words,
                    words_value=value,
                    offset=m.start(),
                )
            )
    return out


__all__ = [
    "MoneyWordMismatch",
    "find_money_word_mismatches",
    "parse_cardinal_words",
]
