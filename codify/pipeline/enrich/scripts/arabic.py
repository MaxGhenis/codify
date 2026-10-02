"""Arabic-script data for the structurer.

All of it was inline in `anchors.py` and `validator.py`, running on every document
whatever the jurisdiction. Grouped so a second script means a sibling module rather
than another edit to the general passes. The normalisation, cardinal-word and
digit-confusion helpers keep their own modules, having callers outside the structurer.
"""

from __future__ import annotations

import re

from codify.pipeline.enrich.arabic_normalise import AR_ORDINAL_TO_INT
from codify.pipeline.enrich.scripts import ARABIC_SCRIPT, ScriptPack, register

# Arabic-block letters, excluding diacritics, digits and punctuation, so a
# joiner bridge spans م↔ا↔د↔ة and not `ة (`.
_LETTER_RE = re.compile(r"[ء-يٮ-ۓ]")

# Tatweel and the zero-width joiners. OCR inserts these mid-word and the
# pre-structurer normaliser strips them, so the class exists for callers that
# hand a raw literal straight to a compiled regex.
_JOINER_OPT = r"[ـ‌‍]*"

# Words that make the text before a marker prose rather than a header:
# "بموجب المادة 5" cites article 5, it does not open it.
_PROSE_PRECURSORS: tuple[str, ...] = (
    "في",
    "من",
    "إلى",
    "عن",
    "بموجب",
    "بمقتضى",
    "وفق",
    "حسب",
    "بحسب",
    "أحكام",
    "نص",
    "نصت",
    "حكم",
    "تنص",
    "بنص",
    # Nouns that introduce a reference to an annex/table rather than head one
    # ("بند الملحق الأول" cites the annex's first item in prose).
    "بند",
    "بنود",
    "مرفق",
    # Recital/penalty citation forms observed spawning phantom articles in
    # the v5 review: "ولاسيما المادة (٥٥) منه", "بقوة المادة (41) من
    # القانون الأساسي", "كل من يخالف المادة (10)".
    "ولاسيما",
    "لاسيما",
    "سيما",
    "بقوة",
    "يخالف",
    "بأحكام",
    # Verbs and nouns introducing a cross-reference to an existing article. Amending
    # instruments use the verb forms at line start, and without them the cited article
    # becomes a phantom anchor whose TOC-dedup then deletes the genuine host of that
    # number, which is how a measured instrument emitted 1,2,4,5,3.
    "تعديل",
    "تعديلات",
    "تعدل",
    "يعدل",
    "عدلت",
    "إلغاء",
    "تلغى",
    "يلغى",
    "ألغيت",
    "إضافة",
    "تضاف",
    "يضاف",
    "أضيفت",
    "استبدال",
    "تستبدل",
    "يستبدل",
    "استبدلت",
    "تطبيق",
    "تنفيذ",
    "بعد",
    "قبل",
    "هذه",
    "هذا",
    "تلك",
    "ذلك",
    # Preamble citation precursors. Decree-laws may cite the authorising Basic Law
    # article as "استناداً لأحكام … لا سيما المادة (55)", and without these the citation was
    # promoted to a structural Article 55 anchor that then owned the preamble body.
    "استناداً",
    "استنادا",
    "بناءً",
    "بناء",
    "سيما",
    "اطلاع",
    "اطلاعنا",
)

# Marks a citation only when adjacent on one line. "عقوبة مخالفة المادة 10" is
# a penalty heading citing article 10, but "…ارتكب مخالفة" ends a sentence and
# the next line may open a real article.
_SAMELINE_PRECURSORS: tuple[str, ...] = ("مخالفة",)

# A list numbered in ordinal words rather than digits.
_ORDINAL_LIST_MARKERS: tuple[tuple[str, str], ...] = (
    ("أولاً", "1"),
    ("أولا", "1"),
    ("ثانياً", "2"),
    ("ثانيا", "2"),
    ("ثالثاً", "3"),
    ("ثالثا", "3"),
    ("رابعاً", "4"),
    ("رابعا", "4"),
    ("خامساً", "5"),
    ("خامسا", "5"),
    ("سادساً", "6"),
    ("سادسا", "6"),
    ("سابعاً", "7"),
    ("سابعا", "7"),
    ("ثامناً", "8"),
    ("ثامنا", "8"),
    ("تاسعاً", "9"),
    ("تاسعا", "9"),
    ("عاشراً", "10"),
    ("عاشرا", "10"),
)

# The alphabetic enumerator sequence as legal lists use it. هـ and ه render the
# same letter, so the alias folds one onto the other.
_LETTER_ORDER: tuple[str, ...] = tuple("أبجدهوزحطيكلمنسعفصقرشتثخذضظغ")
_LETTER_ALIAS = {"ه": "هـ"}

# Accepts Arabic-Indic and extended Arabic-Indic digits alongside ASCII.
_DIGIT_RE = re.compile(r"^\(?([\d٠-٩۰-۹]+)\)?$")

# Abjad letter numbering (annex أ/ب/ج) folds to its ordinal rank so eIds stay
# ASCII. Hamza-carrier forms of alif fold to bare alif before ranking.
_ABJAD_RANK = {c: i + 1 for i, c in enumerate("ابجدهوزحطيكلمنسعفصقرشتثخذضظغ")}
_ABJAD_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})


def abjad_letter_class() -> str:
    """Every letter `abjad_index_to_latin` folds, for callers that must detect
    an eId the fold will change (the normalise-eids selector)."""
    return "".join(sorted(set(_ABJAD_RANK) | set("أإآ")))


def abjad_index_to_latin(letter: str) -> str | None:
    """Fold an abjad-letter index to a stable ASCII token: أ→a … ي→z for the
    first 26 letters, l27/l28 for ظ/غ so they stay distinct from numeric bis
    indices 27/28. None when it is not a single abjad letter, so callers can
    flag the drift rather than ship a non-ASCII eId."""
    rank = _ABJAD_RANK.get(letter.translate(_ABJAD_FOLD))
    if rank is None:
        return None
    return chr(96 + rank) if rank <= 26 else f"l{rank}"


# Doubled ta-marbuta and hamza are OCR artefacts, Arabic geminating with a shadda
# diacritic rather than by repeating the character. Detect only, since correcting
# Arabic morphology without an Arabic-aware layer risks worse damage than the noise.
_GARBLE_PATTERNS = (re.compile(r"ةة"), re.compile(r"ءء"))

# The plural article noun, which governs a whole citation list.
_PLURAL_MARKER_RE = re.compile(r"المواد")

ARABIC = register(
    ScriptPack(
        script=ARABIC_SCRIPT,
        letter_re=_LETTER_RE,
        joiner_opt=_JOINER_OPT,
        ordinal_to_int=AR_ORDINAL_TO_INT,
        letter_order=_LETTER_ORDER,
        letter_alias=_LETTER_ALIAS,
        digit_re=_DIGIT_RE,
        prose_precursors=_PROSE_PRECURSORS,
        sameline_precursors=_SAMELINE_PRECURSORS,
        ordinal_list_markers=_ORDINAL_LIST_MARKERS,
        abjad_rank=_ABJAD_RANK,
        abjad_fold=_ABJAD_FOLD,
        garble_patterns=_GARBLE_PATTERNS,
        plural_marker_re=_PLURAL_MARKER_RE,
    )
)
