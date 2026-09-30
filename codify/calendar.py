"""Local-calendar → Gregorian conversion for FRBR URIs.

FRBR URIs always use Gregorian years. Non-Gregorian jurisdictions store
the local date as an FRBRalias; this module converts them.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from functools import lru_cache
from typing import Any, Literal, NamedTuple, cast

import structlog

from .jurisdictions import CalendarConversion, load_config, try_load_config
from .lang import normalise_digits, word_bounded

logger = structlog.get_logger()


class CalendarConversionError(ValueError):
    pass


#: Which grid a month is on: the Gregorian one, or the calendar's own, which the
#: config may declare to be Gregorian. A caller naming neither means Gregorian.
MonthGrid = Literal["gregorian", "local"]


def to_gregorian_year(
    local_year: str | int,
    country_code: str,
    month: int | None = None,
    day: int | None = None,
    *,
    month_grid: MonthGrid = "gregorian",
) -> int:
    """Convert a local-calendar year to Gregorian. A named jurisdiction with no
    config raises: reading a Hijri year as Gregorian is a plausible wrong date."""
    if not country_code:
        # Nothing to resolve; an absent config raises rather than defaulting.
        return _coerce_int(local_year)
    cfg = load_config(country_code)
    if cfg.frbr is None or cfg.frbr.calendar_conversion is None:
        return _coerce_int(local_year)

    return _apply_rule(
        local_year, cfg.frbr.calendar_conversion, month=month, day=day, month_grid=month_grid
    )


#: Where a local year meets 1 January on its own grid: the month straddling it,
#: its last day surely before, its first day surely after. Poush; Tahsas.
_LOCAL_TURN = {"bikram_samvat": (9, 15, 18), "ethiopian": (4, 21, 23)}
#: The same on the Gregorian grid: the month the local new year falls in, the
#: last day surely before it, the first day surely on or after.
_GREGORIAN_TURN = {"bikram_samvat": (4, 12, 14), "ethiopian": (9, 10, 12)}
#: How many months each calendar's own grid holds.
_LOCAL_MONTHS = {"bikram_samvat": 12, "ethiopian": 13}


def _in_the_earlier_gregorian_year(
    rule: CalendarConversion, month: int | None, day: int | None, month_grid: MonthGrid
) -> bool | None:
    """Whether the date falls in the earlier of the two Gregorian years its
    local year straddles, or None where the month, or the day, cannot say."""
    # Off the grid, a month or a day says nothing, as no month says nothing.
    if month is None or day is not None and not 1 <= day <= 31:
        return None
    # A month of the calendar's own grid is Gregorian where the config says so.
    if month_grid == "gregorian" or rule.month_day_is_gregorian:
        turn = _GREGORIAN_TURN.get(rule.kind)
        if turn is None or not 1 <= month <= 12:
            return None
        turn_month, before, after = turn
        turn_month = rule.new_year_month or turn_month
        if rule.new_year_day:
            before, after = rule.new_year_day - 1, rule.new_year_day
        if month != turn_month:
            return month > turn_month
        # In the new-year month, the days before it close the earlier year.
        if day is None or before < day < after:
            return None
        return day >= after
    turn = _LOCAL_TURN.get(rule.kind)
    if turn is None or not 1 <= month <= _LOCAL_MONTHS[rule.kind]:
        return None
    turn_month, before, after = turn
    if month != turn_month:
        return month < turn_month
    if day is None or before < day < after:
        return None
    return day <= before


def _coerce_int(v: str | int) -> int:
    if isinstance(v, int):
        return v
    # Handle Japanese era names like "令和4" or "Reiwa 4"
    cleaned = re.sub(r"[^\d]", "", str(v).strip())
    if not cleaned:
        raise CalendarConversionError(f"cannot parse year {v!r}")
    return int(cleaned)


_SIGNED_INT_RE = re.compile(r"^[+-]?\d+$")


def _apply_rule(
    local_year: str | int,
    rule: CalendarConversion,
    *,
    month: int | None,
    day: int | None = None,
    month_grid: MonthGrid = "gregorian",
) -> int:
    # Every calendar here begins at one. Parse a numeric string before the check,
    # or "0" and "-5" pass a guard the integers fail. A non-numeric value is an
    # era name and belongs to the branch that reads it.
    numeric = str(local_year).strip()
    if _SIGNED_INT_RE.match(numeric) and int(numeric) < 1:
        raise CalendarConversionError(f"{local_year!r} is not a year")
    earlier = _in_the_earlier_gregorian_year(rule, month, day, month_grid)
    converted = _convert(local_year, rule, earlier)
    # Off the Gregorian grid a month cannot be read against the new-year month.
    if month is None or not 1 <= month <= 12 or day is not None and not 1 <= day <= 31:
        return converted
    if rule.new_year_reform_year is None:
        return converted
    # Shifted here and nowhere else, so no caller can disagree or shift twice.
    on_gregorian_grid = month_grid == "gregorian" or rule.month_day_is_gregorian
    return converted + reform_shift(rule, _coerce_int(local_year), month, on_gregorian_grid)


def _convert(local_year: str | int, rule: CalendarConversion, earlier: bool | None) -> int:
    """The year by the rule's kind, month-blind but for the kinds whose own arm
    reads which side of the Gregorian new year the date fell."""
    kind = rule.kind
    if kind == "epoch_offset":
        if rule.epoch_year is None:
            raise CalendarConversionError("epoch_offset requires epoch_year")
        year_only = _leading_year(str(local_year))
        if year_only is None or year_only < 1:
            raise CalendarConversionError(f"no year in {local_year!r}")
        return year_only + rule.epoch_year

    if kind == "bikram_samvat":
        offset = rule.offset if rule.offset is not None else 57
        year = _coerce_int(local_year)
        return year - offset if earlier else year - offset + 1

    if kind == "buddhist":
        epoch = rule.epoch_year if rule.epoch_year is not None else -543
        return _coerce_int(local_year) + epoch

    if kind == "era_table":
        return _convert_era_table(local_year, rule)

    if kind == "hijri_lunar":
        return _convert_hijri_lunar(_coerce_int(local_year))

    if kind == "hijri_solar":
        # Solar Hijri ≈ Gregorian - 621 (approximation; varies by month)
        return _coerce_int(local_year) + 621

    if kind == "ethiopian":
        # 7-8 years behind: EC + 7 from the new year in September, EC + 8 after
        # the Gregorian one.
        year = _coerce_int(local_year)
        return year + 8 if earlier is False else year + 7

    raise CalendarConversionError(f"unknown calendar conversion kind: {kind}")


#: Every spelling of a calendar the prompt, the configs or a caller may use, to
#: the one name the rest of this module reads. One set, so no alias bypasses.
_CALENDAR_ALIASES = {
    **dict.fromkeys(("hijri", "hijri_lunar", "lunar_hijri", "islamic"), "lunar_hijri"),
    **dict.fromkeys(("hijri_solar", "solar_hijri", "persian", "jalali"), "solar_hijri"),
    **dict.fromkeys(("buddhist", "buddhist_era", "thai"), "buddhist_era"),
    **dict.fromkeys(("ethiopian", "ethiopic"), "ethiopian"),
    **dict.fromkeys(("minguo", "roc", "taiwan"), "minguo"),
    "japanese_era": "japanese_era",
}


def calendar_family(label: str | None) -> str:
    """The one name for a calendar however it was spelt; an unknown spelling is
    itself, normalised, so it matches nothing it should not."""
    named = normalise_calendar(label)
    return _CALENDAR_ALIASES.get(named, named)


def year_from_calendar(year: str | int, calendar: str, country: str = "") -> int | None:
    """Convert a year + calendar to Gregorian. Returns None if the calendar
    isn't supported, or if it needs a jurisdiction's era table and no country
    was given to find one. Uses Muharram 1 / Farvardin 1 / Meskerem 1 / etc. for
    the conversion since FRBR URIs only need the year."""
    cal = calendar_family(calendar)
    if cal == "minguo":
        # Before the coercion, which concatenated every digit of a native date and
        # returned early on 民國元年, where 元 is the first year.
        year_only = _leading_year(str(year))
        if year_only is None or year_only < 1:
            return None
        return year_only + 1911
    if cal == "japanese_era":
        # Before the int coercion: this arm's input is "Reiwa 5", not a number,
        # and the era table belongs to the jurisdiction rather than this module.
        if not country or str(year).strip().lstrip("-").isdigit():
            # A bare number names no era, and Showa 5 is not Reiwa 5. Without a
            # country there is no table to read either.
            return None
        # to_gregorian_year coerces a bare number when a jurisdiction declares
        # no rule, which would read "Reiwa 5" on a Gregorian country as 5.
        cfg = try_load_config(country)
        rule = cfg.frbr.calendar_conversion if cfg and cfg.frbr else None
        # Any rule is not enough: a fixed offset would read an era year as its own.
        if rule is None or rule.kind != "era_table":
            return None
        try:
            return to_gregorian_year(year, country)
        except CalendarConversionError:
            return None
    try:
        n = _coerce_int(year)
    except CalendarConversionError:
        return None
    # Every calendar here counts from one; zero and below are not years.
    if n < 1:
        return None
    if cal in ("", "gregorian"):
        return n
    if cal == "lunar_hijri":
        # hijridate uses Umm al-Qura (accurate but only covers 1343-1500 AH);
        # fall back to convertdate's tabular Islamic for older years.
        try:
            from hijridate import Hijri

            return Hijri(n, 1, 1).to_gregorian().year
        except (OverflowError, ImportError):
            from convertdate import islamic  # type: ignore[import-untyped]

            return cast(int, islamic.to_gregorian(n, 1, 1)[0])
    if cal == "solar_hijri":
        from convertdate import persian

        return cast(int, persian.to_gregorian(n, 1, 1)[0])
    if cal == "ethiopian":
        # Ethiopian year begins on Meskerem 1 (≈ 11 Sept Gregorian); the
        # overlap year is EC + 7 before that date and EC + 8 after, so
        # for FRBR year purposes the predominant Gregorian year is EC + 8.
        return n + 8
    if cal in ("coptic",):
        from convertdate import coptic

        return cast(int, coptic.to_gregorian(n, 1, 1)[0])
    if cal in ("hebrew",):
        from convertdate import hebrew

        return cast(int, hebrew.to_gregorian(n, 7, 1)[0])  # Tishrei 1, civil new year
    if cal == "buddhist_era":
        return n - 543
    return None


# Conversion kinds whose `_apply_rule` arm already branches on the month.
_MONTH_SENSITIVE_KINDS = frozenset({"bikram_samvat", "ethiopian"})

# A blank line, which ends the paragraph a cue introduces.
_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n")

# Between the words of a cue: whitespace crossing at most one line break. `\s+`
# swallows a blank line, putting the break inside the cue where nothing sees it.
_CUE_GAP = r"(?:[^\S\n]+\n?[^\S\n]*|[^\S\n]*\n[^\S\n]*)"

# How much text after a cue may hold the date it introduces; beyond it the date
# belongs to the next paragraph.
_DATE_WINDOW_CHARS = 200


class _LocalDatePatterns(NamedTuple):
    cue: re.Pattern[str]
    date: re.Pattern[str]
    month_index: dict[str, int]


def compile_local_date_patterns(rule: CalendarConversion) -> _LocalDatePatterns | None:
    """Cue and date patterns for a calendar that names its months, else None."""
    # Only the year converts here, so a calendar with a month grid of its own
    # cannot have a date composed from it.
    if not rule.month_names or not rule.date_cues or not rule.month_day_is_gregorian:
        return None
    # Folded, and the patterns case-blind with it: a Latin name in capitals is
    # the same month, and twelve names stay twelve.
    index = {name.casefold(): i for i, name in enumerate(rule.month_names, start=1) if name}
    # Each literal bounded where it is a Latin word: "May" is not "Mayor".
    months = "|".join(word_bounded(m) for m in sorted(index, key=lambda m: (-len(m), m)))
    particles = "|".join(
        word_bounded(p) for p in sorted(rule.year_particles, key=lambda p: (-len(p), p)) if p
    )
    # The particle carries the separator that follows it, so the pattern holds
    # one whitespace run rather than two around an optional group.
    optional_particle = rf"(?:(?:{particles})\s*)?" if particles else ""
    # A cue declared with single spaces meets a signature block's line breaks.

    # Longest first, or a shorter cue wins at the same offset and spends the
    # reach on the rest of the longer one.
    ordered = sorted({c for c in rule.date_cues if c.strip()}, key=lambda c: (-len(c), c))
    cues = "|".join(_CUE_GAP.join(map(word_bounded, c.split())) for c in ordered)
    return _LocalDatePatterns(
        re.compile(cues, re.IGNORECASE),
        re.compile(
            rf"(?<![0-9])(?P<day>[0-9]{{1,2}})\s*(?P<month>{months})\s*"
            rf"{optional_particle}(?P<year>[0-9]{{3,4}})(?![0-9])",
            re.IGNORECASE,
        ),
        index,
    )


class _GrammarKey(NamedTuple):
    """The fields the patterns are built from. `eras` is absent: it holds dicts,
    which no cache key may carry."""

    kind: str
    month_names: tuple[str, ...]
    date_cues: tuple[str, ...]
    year_particles: tuple[str, ...]
    month_day_is_gregorian: bool


@lru_cache(maxsize=32)
def _cached_patterns(key: _GrammarKey) -> _LocalDatePatterns | None:
    return compile_local_date_patterns(
        CalendarConversion(
            kind=key.kind,  # type: ignore[arg-type]
            month_names=list(key.month_names),
            date_cues=list(key.date_cues),
            year_particles=list(key.year_particles),
            month_day_is_gregorian=key.month_day_is_gregorian,
        )
    )


def _rule_and_patterns(
    country: str,
) -> tuple[CalendarConversion, _LocalDatePatterns] | None:
    """The rule and its compiled patterns, or None. Keyed on the rule's fields:
    a country key outlives `try_load_config.cache_clear()`."""
    cfg = try_load_config(country) if country else None
    rule = cfg.frbr.calendar_conversion if cfg is not None and cfg.frbr is not None else None
    if rule is None:
        return None
    # Before the key is built: a rule declaring no grammar need not be hashable
    # to say so.
    if not rule.month_names or not rule.date_cues or not rule.month_day_is_gregorian:
        return None
    patterns = _cached_patterns(_grammar_key(rule))
    return None if patterns is None else (rule, patterns)


def _grammar_key(rule: CalendarConversion) -> _GrammarKey:
    return _GrammarKey(
        rule.kind,
        tuple(rule.month_names),
        tuple(rule.date_cues),
        tuple(rule.year_particles),
        rule.month_day_is_gregorian,
    )


def declares_local_date_grammar(country: str) -> bool:
    """Whether this jurisdiction names the months a dated line is written in.
    Read before decoding a source, so declaring none costs nothing."""
    return _rule_and_patterns(country) is not None


def _declared_rule(country: str) -> CalendarConversion | None:
    """The conversion rule the jurisdiction declares, or None: naming a calendar
    without one leaves only the generic conversion to apply."""
    cfg = try_load_config(country) if country else None
    return cfg.frbr.calendar_conversion if cfg is not None and cfg.frbr is not None else None


def declares_this_calendar(label: str, country: str) -> bool:
    """Whether `label` names the calendar this jurisdiction declares, so its own
    conversion rule applies rather than the generic one for that calendar."""
    cfg = try_load_config(country) if country else None
    if cfg is None:
        return False
    # Through the one alias set, so a spelling the prompt emits and one the
    # config uses name the same calendar.
    return calendar_family(cfg.calendar) == calendar_family(label)


def canonical_year_and_date(metadata: dict[str, Any]) -> tuple[str, str]:
    """The year and date fields as every match downstream expects them: trimmed
    and in ASCII digits. A model pads both fields and decorates the year."""
    fields = (str(metadata.get("year") or ""), str(metadata.get("date") or ""))
    return (normalise_digits(fields[0]).strip(), normalise_digits(fields[1]).strip())


#: A date field, whole: a year run of up to four digits, a month, a day.
_DATE_FIELD = re.compile(r"([0-9]{1,4})-([0-9]{2})-([0-9]{2})")


class DateParts(NamedTuple):
    """What a date field states: its year run as written, since a local year is
    not a Gregorian one, and a month and day within the widest grid."""

    year: str
    month: int
    day: int


def stated_date_parts(raw: str) -> DateParts | None:
    """The one reading of a date field, or None where the field is not wholly
    a date: every consumer of a year run, a month or a day reads this."""
    found = _DATE_FIELD.fullmatch(normalise_digits(raw).strip())
    if found is None:
        return None
    month, day = int(found.group(2)), int(found.group(3))
    if not (1 <= month <= 13 and 1 <= day <= 31):
        return None
    return DateParts(found.group(1), month, day)


def month_day_stating_this_year(metadata: dict[str, Any], token: str) -> tuple[int, int] | None:
    """The month and day of a metadata date whose year run is `token`, or None.
    A year that began mid-year needs them to settle, on every year path."""
    parts = stated_date_parts(canonical_year_and_date(metadata)[1])
    # On the year the token states, not the string carrying it: a model writes
    # the era beside the number.
    if parts is None or parts.year != sole_year_token(normalise_digits(token)):
        return None
    return (parts.month, parts.day)


def labelled_year_as_gregorian(
    token: str, label: str, country: str, month_day: tuple[int, int] | None = None
) -> int | None:
    """A local year in Gregorian: the jurisdiction's rule where the label names
    its calendar, else the generic conversion. The month settles a mid-year one."""
    # Never for an era-named calendar: a bare number there is part of a year,
    # not one, and the generic path refuses it on purpose.
    if (
        normalise_calendar(label) not in ERA_NAMED_CALENDARS
        and declares_this_calendar(label, country)
        and _declared_rule(country) is not None
    ):
        month, day = month_day if month_day is not None else (None, None)
        try:
            return to_gregorian_year(token, country, month=month, day=day, month_grid="local")
        except CalendarConversionError:
            return None
    return year_from_calendar(token, label, country)


def local_date_from_text(text: str, country: str) -> date | None:
    """The Gregorian date a source states its document was made on, or None.
    Cue-anchored: an unintroduced dated line is usually one the document amends."""
    found_rule = _rule_and_patterns(country)
    if found_rule is None:
        return None
    rule, patterns = found_rule
    # One line-ending convention before any pattern runs: a carriage return is
    # a break, and two of them a paragraph, as with newlines.
    folded = normalise_digits(text.replace("\r\n", "\n").replace("\r", "\n"))
    return _first_stated_date(folded, patterns, rule, country)


def _first_stated_date(
    folded: str,
    patterns: _LocalDatePatterns,
    rule: CalendarConversion,
    country: str,
) -> date | None:
    """The first date a declared cue introduces, or None. Separate so a caller
    can bound it without a jurisdiction to load."""
    # One pass over each, both in order: a fresh search per cue is quadratic.
    # Cue ends, not starts, or a long cue spends the window on itself.
    cues = [m.end() for m in patterns.cue.finditer(folded)]
    if not cues:
        return None
    nearest = 0
    for candidate in patterns.date.finditer(folded, cues[0]):
        while nearest + 1 < len(cues) and cues[nearest + 1] <= candidate.start():
            nearest += 1
        if candidate.start() - cues[nearest] >= _DATE_WINDOW_CHARS:
            continue
        # A cue introduces what follows it in its own paragraph, the date's own
        # parts included: wrapped lines count, a blank line ends the reach.
        if _PARAGRAPH_BREAK.search(folded, cues[nearest], candidate.end()):
            continue
        # Validated inside the walk, or a syntactic non-date (31 April) hides a
        # valid later cue.
        stated = _compose(candidate, patterns, rule, country)
        if stated is not None:
            return stated
    return None


def reform_shift(
    rule: CalendarConversion, local_year: int, month: int, on_gregorian_grid: bool
) -> int:
    """1 where a year that began mid-year puts this month in the next Gregorian
    year, else 0. A kind whose own arm reads the month is shifted already."""
    reform = rule.new_year_reform_year
    # The new-year month is Gregorian-side too, so a month of another grid
    # cannot be compared with it.
    if reform is None or rule.kind in _MONTH_SENSITIVE_KINDS or not on_gregorian_grid:
        return 0
    return 1 if local_year < reform and month < (rule.new_year_month or 1) else 0


def _compose(
    found: re.Match[str],
    patterns: _LocalDatePatterns,
    rule: CalendarConversion,
    country: str,
) -> date | None:
    """The Gregorian date a matched line states, or None when it states none."""
    month = patterns.month_index.get(found.group("month").casefold())
    if month is None:
        # Case-blind matching reaches spellings the folded index does not hold.
        return None
    local_year = int(found.group("year"))
    try:
        gregorian_year = _apply_rule(
            local_year, rule, month=month, day=int(found.group("day")), month_grid="gregorian"
        )
    except CalendarConversionError as exc:
        # A misconfigured rule otherwise reads as "this document states no
        # date", the same answer a whole corpus would give.
        logger.warning("local_date_conversion_failed", country=country, error=str(exc)[:160])
        return None
    try:
        return date(gregorian_year, month, int(found.group("day")))
    except ValueError:
        logger.warning("local_date_out_of_range", country=country, raw=found.group(0)[:40])
        return None


def _convert_hijri_lunar(hijri_year: int) -> int:
    """Lunar Hijri → Gregorian. Kept for back-compat with `_apply_rule`."""
    result = year_from_calendar(hijri_year, "hijri")
    if result is None:  # only when hijridate fails to import
        return round(hijri_year * 0.970229 + 621.5643)
    return result


def _era_start_year(era: dict[str, Any]) -> int:
    """Extract the Gregorian start year from an era entry."""
    if "start_gregorian" in era:
        return int(era["start_gregorian"])
    start = era.get("start", "")
    if start:
        return int(str(start)[:4])
    return 0


# Calendars whose year means nothing without its era name, so a conversion that
# fails has no number to fall back on: 6 is not a year, it is part of one.
ERA_NAMED_CALENDARS = frozenset({"japanese_era"})


_KANJI_DIGITS = {
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def kanji_number(text: str) -> int | None:
    """A kanji numeral below 100, the range an era year occupies. 六 is 6, 十 is
    10, 二十三 is 23. None for anything that is not one, malformed sequences of
    accepted characters included: 十十 and 二三十 are not numbers."""
    if not text or any(c not in _KANJI_DIGITS and c != "十" for c in text):
        return None
    if text.count("十") > 1:
        return None
    if "十" not in text:
        # A bare numeral is one digit; 二三 is not twenty-three.
        return _KANJI_DIGITS[text] if len(text) == 1 else None
    tens, _, units = text.partition("十")
    if len(tens) > 1 or len(units) > 1:
        return None
    high = _KANJI_DIGITS[tens] if tens else 1
    low = _KANJI_DIGITS[units] if units else 0
    return high * 10 + low


# The year in a native date: 元 for the first year, else the digits before a year
# marker, else the first run.
_YEAR_RUN_RE = re.compile(r"元\s*年|(\d+)\s*(?:年|年度)|(\d+)")


def normalise_calendar(value: str | None) -> str:
    """The calendar name as every reader should see it. Four callers lowercased
    it and three did not strip, so a padded value took a different branch in
    each."""
    return (value or "").strip().lower()


def _leading_year(text: str) -> int | None:
    m = _YEAR_RUN_RE.search(text)
    if m is None:
        return None
    # A run preceded by a minus is not a year: "-5" read the 5 and dropped the sign.
    if m.start() and text[m.start() - 1] == "-":
        return None
    digits = m.group(1) or m.group(2)
    return int(digits) if digits else 1


def sole_year_token(raw: str) -> str:
    """The one whole 3-4 digit run in a string, or "". Shared so the URI year and
    the stored year cannot disagree about what a raw year holds."""
    runs = re.findall(r"(?<!\d)\d{3,4}(?!\d)", raw)
    return runs[0] if len(runs) == 1 else ""


def title_year_token(title: str, number: object = "") -> str:
    """The title's one whole year run, or "" when the same digits are the
    document number: "Act No. 2010" numbered 2010 names no year."""
    token = sole_year_token(title)
    if not reads_as_a_gregorian_year(token) or token == str(number or "").strip():
        return ""
    return token


def reads_as_a_gregorian_year(value: str | int) -> bool:
    """A four-digit year in a range no era reaches. Japanese eras count to about
    64, so a number this size is a Gregorian year the model mislabelled, and
    refusing it would lose a year that is plainly right."""
    text = str(value).strip()
    return text.isdigit() and len(text) == 4 and 1000 <= int(text) <= 2999


def _folded(text: str) -> str:
    """Case- and diacritic-insensitive, so the prompt's "Showa" reaches a table
    spelling it "Shōwa"."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(c for c in stripped if not unicodedata.combining(c)).casefold()


def _composed(text: str) -> str:
    """NFC, so a decomposed input has the same length as the prefix matched
    against it: slicing a decomposed "Shōwa" by 5 leaves "a 64"."""
    return unicodedata.normalize("NFC", text)


def _convert_era_table(local_year: str | int, rule: CalendarConversion) -> int:
    """Japanese/Chinese era table → Gregorian."""
    if not rule.eras:
        raise CalendarConversionError("era_table requires eras list")

    year_str = _composed(str(local_year).strip())

    # Try matching era name in the input (e.g., "令和4", "Reiwa 4")
    for era in rule.eras:
        era_name = era.get("name", "")
        era_abbrev = era.get("abbrev", "")
        start = _era_start_year(era)

        for prefix in [_composed(era_name), _composed(era_abbrev)]:
            if prefix and _folded(year_str).startswith(_folded(prefix)):
                num_part = year_str[len(prefix) :].strip()
                if num_part:
                    # The leading run only. Stripping every non-digit read
                    # "令和6年法律第1号" as era year 61, four decades out.
                    m = re.match(r"\d+|元|[〇一二三四五六七八九十]+", num_part)
                    if m is None:
                        raise CalendarConversionError(f"no era year in {year_str!r}")
                    numeral = m.group()
                    if numeral == "元":
                        n: int | None = 1
                    elif numeral.isdigit():
                        n = int(numeral)
                    else:
                        n = kanji_number(numeral)
                    if n is None or n < 1:
                        raise CalendarConversionError(f"no era year in {year_str!r}")
                    converted = start + n - 1
                    # An era year past its successor's first year is not that era's:
                    # Heisei 32 would be 2020, three years after Heisei ended. The
                    # boundary year itself stays valid, since Showa 64 and Heisei 1
                    # are both 1989 under the predominant-year rule.
                    successors = [
                        _era_start_year(e) for e in rule.eras if _era_start_year(e) > start
                    ]
                    if successors and converted > min(successors):
                        raise CalendarConversionError(f"{year_str!r} is past the end of that era")
                    return converted
                # The era alone names no year: "Reiwa" is not Reiwa 1.
                raise CalendarConversionError(f"no era year in {year_str!r}")

    # A bare number is an era-local year in the current era, which is the only
    # era it can be. A name this table does not carry is not: assuming the latest
    # era there turned 昭和64 into 2082, so refuse rather than answer confidently.
    if not year_str.lstrip("-").isdigit():
        raise CalendarConversionError(f"no era in this jurisdiction's table matches {year_str!r}")
    latest = max(rule.eras, key=_era_start_year)
    return _era_start_year(latest) + _coerce_int(local_year) - 1
