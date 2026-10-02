"""Tests for calendar conversion."""

from pathlib import Path

import pytest

from codify.calendar import (
    CalendarConversionError,
    _apply_rule,
    reform_shift,
    to_gregorian_year,
    year_from_calendar,
)
from codify.jurisdictions import CalendarConversion, JurisdictionConfigError


@pytest.fixture(autouse=True)
def calendar_configs(tmp_path, monkeypatch):
    """Exercise every conversion rule without private jurisdiction profiles."""
    from tests.config_fixtures import isolated_configs

    rules = {
        "sa": {"kind": "hijri_lunar"},
        "ir": {"kind": "hijri_solar"},
        "th": {"kind": "buddhist"},
        "tw": {"kind": "epoch_offset", "epoch_year": 1911},
        "xnp": {"kind": "bikram_samvat"},
        "jp": {
            "kind": "era_table",
            "eras": [
                {"name": "Reiwa", "abbrev": "令和", "start": "2019-05-01"},
                {"name": "Heisei", "abbrev": "平成", "start": "1989-01-08"},
                {"name": "Shōwa", "abbrev": "昭和", "start": "1926-12-25"},
                {"name": "Taishō", "abbrev": "大正", "start": "1912-07-30"},
                {"name": "Meiji", "abbrev": "明治", "start": "1868-01-01"},
            ],
        },
    }
    configs = {
        code: {
            "calendar": {"jp": "japanese_era", "tw": "minguo"}.get(code, "gregorian"),
            "frbr": {"country_code": code, "calendar_conversion": rule},
        }
        for code, rule in rules.items()
    }
    with isolated_configs(monkeypatch, tmp_path / "data" / "jurisdictions", configs):
        yield


class TestHijriLunar:
    def test_1443(self):
        assert to_gregorian_year(1443, "sa") == 2021

    def test_1446(self):
        assert to_gregorian_year(1446, "sa") == 2024

    def test_string_input(self):
        assert to_gregorian_year("1443", "sa") == 2021


class TestSolarHijri:
    def test_1401(self):
        # Solar Hijri 1401 ≈ 2022
        result = to_gregorian_year(1401, "ir")
        assert 2022 <= result <= 2023

    def test_1403(self):
        result = to_gregorian_year(1403, "ir")
        assert 2024 <= result <= 2025


class TestBuddhistEra:
    def test_2565(self):
        assert to_gregorian_year(2565, "th") == 2022

    def test_2569(self):
        assert to_gregorian_year(2569, "th") == 2026


class TestEraTable:
    def test_reiwa_6(self):
        # Reiwa 6 = 2024 (Reiwa started 2019)
        result = to_gregorian_year(6, "jp")
        assert result == 2024

    def test_reiwa_1(self):
        result = to_gregorian_year(1, "jp")
        assert result == 2019


class TestGregorianPassthrough:
    def test_a_gregorian_jurisdiction(self):
        assert to_gregorian_year(2025, "gb") == 2025

    def test_an_unknown_country_raises(self):
        """Passthrough is for an unnamed jurisdiction, not an unwritten one."""
        with pytest.raises(JurisdictionConfigError):
            to_gregorian_year(2025, "zz")
        assert to_gregorian_year(2025, "") == 2025

    def test_string(self):
        assert to_gregorian_year("2025", "gb") == 2025


class TestEdgeCases:
    def test_empty_string_raises(self):
        with pytest.raises(CalendarConversionError):
            to_gregorian_year("", "gb")

    def test_non_numeric_raises(self):
        with pytest.raises(CalendarConversionError):
            to_gregorian_year("abc", "gb")


class TestImperialErasAreTwoSystems:
    """`imperial_era` covered Japanese eras and ROC/Minguo, which convert
    differently, so a year filed under it could not be converted at all."""

    def test_a_minguo_year_is_the_roc_year_plus_1911(self) -> None:
        assert year_from_calendar(1, "minguo") == 1912
        assert year_from_calendar(114, "minguo") == 2025
        assert year_from_calendar("38", "roc") == 1949

    def test_a_japanese_era_year_converts_through_the_jurisdiction_s_table(
        self,
    ) -> None:
        """The eras live in the jurisdiction's config, in both scripts, so this
        module carries no table of its own to drift from them."""
        assert year_from_calendar("Reiwa 5", "japanese_era", "jp") == 2023
        assert year_from_calendar("令和5", "japanese_era", "jp") == 2023
        assert year_from_calendar("平成31", "japanese_era", "jp") == 2019
        assert year_from_calendar("Shōwa 64", "japanese_era", "jp") == 1989
        assert year_from_calendar("Meiji 1", "japanese_era", "jp") == 1868

    def test_an_era_year_carrying_a_number_after_it_reads_only_the_era_year(
        self,
    ) -> None:
        """Stripping every non-digit read 令和6年法律第1号 as era year 61."""
        assert year_from_calendar("令和6年法律第1号", "japanese_era", "jp") == 2024

    def test_an_era_the_table_does_not_carry_is_refused(self) -> None:
        """Assuming the latest era turned 昭和64 into 2082, so an unknown name
        has no answer; returning one would be a wrong date, not a missing one."""
        assert year_from_calendar("Genna 5", "japanese_era", "jp") is None

    def test_without_a_country_there_is_no_table_to_read(self) -> None:
        assert year_from_calendar("Reiwa 5", "japanese_era") is None

    def test_the_conflated_token_is_refused(self) -> None:
        """The old value named neither system; nothing may convert under it."""
        import pydantic

        from codify.jurisdictions import JurisdictionConfig, LegalEra

        assert year_from_calendar(114, "imperial_era", "tw") is None
        with pytest.raises(pydantic.ValidationError):
            JurisdictionConfig(country_code="xx", name="X", calendar="imperial_era")
        with pytest.raises(pydantic.ValidationError):
            LegalEra(id="x", calendar="imperial_era")

    def test_the_two_jurisdictions_declare_the_system_they_use(self) -> None:
        from codify.jurisdictions import load_config

        assert load_config("jp").calendar == "japanese_era"
        assert load_config("tw").calendar == "minguo"


class TestAnUnconvertedEraYearIsAudible:
    """A year that cannot be converted yields nothing rather than a number that
    looks like a year, and says so: silence here is a wrong date nobody sees."""

    def test_an_unknown_era_warns_and_names_it(self) -> None:
        import structlog
        from structlog.testing import capture_logs

        from codify.pipeline.stages import resolve_year

        with capture_logs() as captured:
            year = resolve_year({"year": "Genna 6", "calendar": "japanese_era"}, "jp")
        structlog.reset_defaults()

        assert year == ""
        warned = [r for r in captured if r.get("event") == "year_calendar_unconverted"]
        assert len(warned) == 1, captured
        assert warned[0]["raw"] == "Genna 6"
        assert warned[0]["calendar"] == "japanese_era"
        assert warned[0]["country"] == "jp"

    def test_a_converted_era_year_says_nothing(self) -> None:
        import structlog
        from structlog.testing import capture_logs

        from codify.pipeline.stages import resolve_year

        with capture_logs() as captured:
            year = resolve_year({"year": "Reiwa 6", "calendar": "japanese_era"}, "jp")
        structlog.reset_defaults()

        assert year == "2024"
        assert [r for r in captured if r.get("event") == "year_calendar_unconverted"] == []


class TestTheStoredYearMatchesTheUri:
    """Ingest resolves the URI year and the stored year from the same metadata
    by two different functions. A calendar one can convert and the other cannot
    files the document under one number and records another."""

    def test_both_paths_agree_on_an_era_year(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        for raw in ("Reiwa 6", "令和6", "平成31", "Shōwa 64"):
            metadata = {"year": raw, "calendar": "japanese_era"}
            uri_year = resolve_year(metadata, "jp")
            stored = gregorian_year(metadata, "jp")
            assert stored is not None, raw
            assert uri_year == str(stored), raw

    def test_both_paths_agree_when_the_era_is_unknown(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        metadata = {"year": "Genna 6", "calendar": "japanese_era"}
        assert resolve_year(metadata, "jp") == ""
        assert gregorian_year(metadata, "jp") is None


class TestTheThreeThingsTheEraTableMustNotLose:
    """Each of these worked once and was lost when the table moved into config."""

    def test_gannen_is_year_one(self) -> None:
        """元 marks an era's first year in place of the digit."""
        assert year_from_calendar("令和元年", "japanese_era", "jp") == 2019
        assert year_from_calendar("令和元", "japanese_era", "jp") == 2019

    def test_a_bare_number_under_an_era_calendar_is_refused(self) -> None:
        """The jurisdiction profile calls a Minguo-style bare year meaningless
        here: Showa 5 and Reiwa 5 are different years, so there is no answer."""
        assert year_from_calendar("5", "japanese_era", "jp") is None
        assert year_from_calendar(5, "japanese_era", "jp") is None

    def test_a_bare_number_still_reads_as_the_current_era_for_the_jurisdiction(
        self,
    ) -> None:
        """to_gregorian_year is asked about a jurisdiction, not a loose string,
        so a number there is a current-era year. That contract is unchanged."""
        from codify.calendar import to_gregorian_year

        assert to_gregorian_year(6, "jp") == 2024
        assert to_gregorian_year(1, "jp") == 2019

    def test_an_era_name_matches_whatever_its_case(self) -> None:
        assert year_from_calendar("reiwa 6", "japanese_era", "jp") == 2024
        assert year_from_calendar("REIWA 6", "japanese_era", "jp") == 2024


class TestARefusalMustReachBothPaths:
    """Declining to convert is only worth anything if the number that was
    declined does not then get filed as the year one line later."""

    def test_the_prompt_s_spelling_reaches_a_table_that_uses_macrons(self) -> None:
        """The prompt names "Showa"; the jurisdiction table spells it "Shōwa"."""
        assert year_from_calendar("Showa 64", "japanese_era", "jp") == 1989
        assert year_from_calendar("Taisho 1", "japanese_era", "jp") == 1912
        assert year_from_calendar("Shōwa 64", "japanese_era", "jp") == 1989

    def test_a_bare_era_number_yields_no_year_on_either_path(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        for raw in ("6", "64"):
            metadata = {"year": raw, "calendar": "japanese_era"}
            assert resolve_year(metadata, "jp") == "", raw
            assert gregorian_year(metadata, "jp") is None, raw

    def test_a_four_digit_year_is_kept_though_the_calendar_says_era(self) -> None:
        """No era counts to four digits, so this is a Gregorian year mislabelled,
        and refusing it would lose a year that is plainly right."""
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        metadata = {"year": "2024", "calendar": "japanese_era"}
        assert resolve_year(metadata, "jp") == "2024"
        assert gregorian_year(metadata, "jp") == 2024


class TestAnEraCalendarNeedsThatJurisdictionSTable:
    def test_an_era_year_on_a_jurisdiction_with_no_table_has_no_answer(self) -> None:
        """to_gregorian_year coerces a bare number where no rule is declared,
        which would have read "Reiwa 5" on a Gregorian jurisdiction as 5."""
        assert year_from_calendar("Reiwa 5", "japanese_era", "gb") is None
        assert year_from_calendar("Reiwa 5", "japanese_era", "jp") == 2023


class TestTheDigitFallbackReadsWholeNumbers:
    def test_a_longer_number_is_not_read_as_a_shorter_one(self) -> None:
        """An unanchored three-to-four digit run matched 1202 inside 12024."""
        from codify.pipeline.stages import resolve_year

        assert resolve_year({"year": "x12024", "calendar": "hindu"}, "jp") == ""
        assert resolve_year({"year": "x2024y", "calendar": "hindu"}, "jp") == "2024"


class TestTheEraTableIsTheRightJurisdictionS:
    def test_a_different_calendar_s_rule_does_not_serve_an_era_year(self) -> None:
        """tw carries a fixed 1911 offset, which read "Reiwa 5" as 1916."""
        assert year_from_calendar("Reiwa 5", "japanese_era", "tw") is None
        assert year_from_calendar(114, "minguo") == 2025

    def test_a_decomposed_era_name_matches(self) -> None:
        """Matching folds diacritics but slices by the raw prefix length, so a
        decomposed "Shōwa" left "a 64" behind."""
        import unicodedata

        for form in ("NFC", "NFD", "NFKD"):
            raw = unicodedata.normalize(form, "Shōwa 64")
            assert year_from_calendar(raw, "japanese_era", "jp") == 1989, form


class TestOneRuleForTheYearToken:
    def test_the_uri_and_the_stored_year_read_a_raw_year_the_same_way(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        for raw, calendar in [
            ("x2024y", "hindu"),
            ("x12024", "hindu"),
            ("2024", "japanese_era"),
            ("6", "japanese_era"),
            ("Reiwa 6", "japanese_era"),
        ]:
            metadata = {"year": raw, "calendar": calendar}
            uri = resolve_year(metadata, "jp")
            stored = gregorian_year(metadata, "jp")
            assert uri == (str(stored) if stored is not None else ""), (raw, calendar)


class TestOnlyAWholeYearIsAYear:
    def test_a_five_digit_token_is_not_a_year(self) -> None:
        """It reached the URI as a five-digit year, because an all-digit raw
        year skipped the token rule entirely."""
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        for raw in ("12024", "99", "1"):
            metadata = {"year": raw}
            assert resolve_year(metadata, "xz") == "", raw
            assert gregorian_year(metadata, "xz") is None, raw

    def test_a_three_or_four_digit_year_still_passes(self) -> None:
        """A URI segment is four digits, so a three-digit year is carried padded
        and stored as the number it is; the canonical form reads the same."""
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        for raw in ("1443", "2024"):
            metadata = {"year": raw}
            assert resolve_year(metadata, "xz") == raw, raw
            assert gregorian_year(metadata, "xz") == int(raw), raw
        for raw in ("622", "0622"):
            assert resolve_year({"year": raw}, "xz") == "0622", raw
            assert gregorian_year({"year": raw}, "xz") == 622, raw


class TestOfficialTextNumbersItsEras:
    """The profile records that official text uses kanji numerals, and the
    prompt asks for the era name in the source script, so 令和六年 must convert."""

    def test_a_kanji_era_year_converts(self) -> None:
        assert year_from_calendar("令和六年", "japanese_era", "jp") == 2024
        assert year_from_calendar("平成十年", "japanese_era", "jp") == 1998
        assert year_from_calendar("昭和二十三年", "japanese_era", "jp") == 1948

    def test_kanji_numbers_below_a_hundred(self) -> None:
        from codify.calendar import kanji_number

        assert kanji_number("六") == 6
        assert kanji_number("十") == 10
        assert kanji_number("二十三") == 23
        assert kanji_number("九十九") == 99
        assert kanji_number("x") is None


class TestAPatchFalseReplayStoresWhatItStored:
    """A patch-false arm exists to reproduce the old result exactly. Applying
    the new refusal there would rewrite a year an old run already stored."""

    def test_the_legacy_arm_keeps_the_old_coercion(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year

        metadata = {"year": "6", "calendar": "japanese_era"}
        assert gregorian_year(metadata, "", legacy=True) == 6
        assert gregorian_year(metadata, "jp") is None


class TestABareEraNamesNoYear:
    def test_an_era_without_a_number_is_refused(self) -> None:
        """ "Reiwa" alone is not Reiwa 1; returning the era's start year would
        date a document to a year its text never states."""
        assert year_from_calendar("令和", "japanese_era", "jp") is None
        assert year_from_calendar("Reiwa", "japanese_era", "jp") is None
        assert year_from_calendar("令和元年", "japanese_era", "jp") == 2019


class TestFrbrDateCalendarIsAlwaysGregorian:
    def test_no_config_declares_another_calendar_for_its_uris(self) -> None:
        """ADR 001: the URI year is Gregorian everywhere. `calendar` records what
        the jurisdiction uses; `frbr.date_calendar` records what its URIs use."""
        import json

        from codify.jurisdictions import JURISDICTIONS_DIR

        root = JURISDICTIONS_DIR
        offenders = {}
        configs = sorted(root.glob("*/config.json"))
        assert configs, "packaged jurisdiction configs must be present"
        for config in configs:
            frbr = json.loads(config.read_text(encoding="utf-8")).get("frbr") or {}
            declared = frbr.get("date_calendar")
            if declared and declared != "gregorian":
                offenders[config.parent.name] = declared
        assert offenders == {}, offenders


class TestAnEraYearStaysInsideItsEra:
    """An era year past its successor's first year is not that era's, and the
    arithmetic alone would answer with a plausible wrong year rather than refuse."""

    def test_a_year_past_the_era_is_refused(self) -> None:
        assert year_from_calendar("平成32", "japanese_era", "jp") is None
        assert year_from_calendar("昭和65", "japanese_era", "jp") is None
        assert year_from_calendar("明治46", "japanese_era", "jp") is None

    def test_a_year_below_one_is_refused(self) -> None:
        assert year_from_calendar("Reiwa 0", "japanese_era", "jp") is None

    def test_the_transition_year_is_still_valid_on_both_sides(self) -> None:
        """Showa 64 and Heisei 1 are both 1989 under the predominant-year rule."""
        assert year_from_calendar("昭和64", "japanese_era", "jp") == 1989
        assert year_from_calendar("平成元年", "japanese_era", "jp") == 1989
        assert year_from_calendar("平成31", "japanese_era", "jp") == 2019
        assert year_from_calendar("令和元年", "japanese_era", "jp") == 2019


class TestAMinguoDateIsNotOneLongNumber:
    """Both resolvers pass the whole date when the model gave no year, and
    coercing that concatenated every digit: 中華民國114年6月9日 became 11469."""

    def test_a_full_native_date_reads_only_its_year(self) -> None:
        assert year_from_calendar("中華民國114年6月9日", "minguo") == 2025
        assert year_from_calendar("114年6月9日", "minguo") == 2025

    def test_a_bare_minguo_year_is_unchanged(self) -> None:
        assert year_from_calendar("114", "minguo") == 2025
        assert year_from_calendar(1, "minguo") == 1912


class TestTheRetiredCalendarTokenMapsByCountry:
    """A legacy config carrying the retired value names a real system per
    country. Falling through to gregorian would silently lose it."""

    def test_a_legacy_config_keeps_its_system(self, tmp_path: object) -> None:
        import json
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from fix_jurisdiction_configs import fix_config

        for code, expected in (("jp", "japanese_era"), ("tw", "minguo")):
            d = Path(str(tmp_path)) / code
            d.mkdir()
            config = d / "config.json"
            config.write_text(
                json.dumps({"code": code, "name": "X", "calendar": "imperial_era"}),
                encoding="utf-8",
            )
            fix_config(config)
            assert json.loads(config.read_text())["calendar"] == expected, code

    def test_another_country_carrying_it_is_loud(self, tmp_path: object) -> None:
        import json
        import sys

        import pytest

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from fix_jurisdiction_configs import fix_config

        d = Path(str(tmp_path)) / "zz"
        d.mkdir()
        config = d / "config.json"
        config.write_text(
            json.dumps({"code": "zz", "name": "X", "calendar": "imperial_era"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="names no system"):
            fix_config(config)


class TestMinguoYearOne:
    """民國元年 is the canonical way to write 1912, and 元 marks a first year in
    both calendars this module converts. Handling it in one and not the other
    left the founding year unreadable."""

    def test_gannen_is_year_one_on_both_resolvers(self) -> None:
        from codify.calendar import to_gregorian_year

        for raw in ("民國元年", "元年", "中華民國元年"):
            assert year_from_calendar(raw, "minguo") == 1912, raw
            assert to_gregorian_year(raw, "tw") == 1912, raw

    def test_a_numbered_minguo_year_is_unchanged(self) -> None:
        from codify.calendar import to_gregorian_year

        for raw, want in (
            ("民國36年", 1947),
            ("中華民國114年6月9日", 2025),
            ("114", 2025),
        ):
            assert year_from_calendar(raw, "minguo") == want, raw
            assert to_gregorian_year(raw, "tw") == want, raw


class TestTheCalendarNameIsReadTheSameWayEverywhere:
    """Four callers lowercased the calendar and three did not strip, so a padded
    value took a different branch in each."""

    def test_a_padded_value_converts_on_every_path(self) -> None:
        from codify.pipeline.enrich.metadata import gregorian_year
        from codify.pipeline.stages import resolve_year

        metadata = {"year": "114", "calendar": " MINGUO "}
        assert year_from_calendar("114", "minguo ") == 2025
        assert resolve_year(metadata, "tw") == "2025"
        assert gregorian_year(metadata, "tw") == 2025


class TestTheConfigFixerWalksTheRealTree:
    """A path guessed from the script's own location pointed inside the package,
    where no config tree exists, so the documented entry point walked nothing."""

    def test_main_reaches_configs_and_maps_the_retired_token(self, tmp_path: object) -> None:
        import json
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import fix_jurisdiction_configs as fixer

        tree = Path(str(tmp_path)) / "jurisdictions"
        (tree / "jp").mkdir(parents=True)
        config = tree / "jp" / "config.json"
        config.write_text(
            json.dumps({"code": "jp", "name": "X", "calendar": "imperial_era"}),
            encoding="utf-8",
        )
        original = fixer.JURISDICTIONS_DIR
        try:
            fixer.JURISDICTIONS_DIR = tree
            fixer.main()
        finally:
            fixer.JURISDICTIONS_DIR = original
        assert json.loads(config.read_text())["calendar"] == "japanese_era"

    def test_the_resolver_is_the_package_s_own(self) -> None:
        import sys

        from codify.jurisdictions import JURISDICTIONS_DIR

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import importlib

        import fix_jurisdiction_configs as fixer

        importlib.reload(fixer)

        assert fixer.JURISDICTIONS_DIR == JURISDICTIONS_DIR
        assert fixer.JURISDICTIONS_DIR.exists()


class TestAMalformedKanjiNumeralIsNotANumber:
    """The characters being accepted ones does not make the sequence a numeral,
    and the digit table raised KeyError rather than declining."""

    def test_a_malformed_sequence_returns_nothing(self) -> None:
        from codify.calendar import kanji_number

        for raw in ("十十", "二三十", "十二十", "二三"):
            assert kanji_number(raw) is None, raw

    def test_it_does_not_leak_out_of_the_converters(self) -> None:
        from codify.calendar import to_gregorian_year

        assert year_from_calendar("令和十十年", "japanese_era", "jp") is None
        with pytest.raises(CalendarConversionError):
            to_gregorian_year("令和十十年", "jp")

    def test_a_well_formed_numeral_is_unchanged(self) -> None:
        from codify.calendar import kanji_number

        assert kanji_number("六") == 6
        assert kanji_number("十") == 10
        assert kanji_number("二十三") == 23
        assert kanji_number("六十六") == 66


class TestYearZeroIsNotAYear:
    """Minguo and Juche begin at one, as does every calendar here, so zero is
    not an off-by-one to absorb: it mapped to the epoch year itself."""

    def test_zero_and_below_are_refused_on_both_resolvers(self) -> None:
        from codify.calendar import to_gregorian_year

        for raw in (0, -5, "0", "-5"):
            assert year_from_calendar(raw, "minguo") is None, raw
            with pytest.raises(CalendarConversionError):
                to_gregorian_year(raw, "tw")

    def test_other_calendars_refuse_zero_too(self) -> None:
        assert year_from_calendar(0, "buddhist_era") is None
        assert year_from_calendar(0, "lunar_hijri") is None
        assert year_from_calendar(0, "ethiopian") is None

    def test_a_real_year_still_converts(self) -> None:
        from codify.calendar import to_gregorian_year

        assert year_from_calendar(114, "minguo") == 2025
        assert to_gregorian_year(114, "tw") == 2025


class TestAStringYearIsGuardedLikeAnInteger:
    """The non-positive guard tested the type before the value, so "0" passed
    where 0 raised: seven of the nine rule-carrying jurisdictions accepted it."""

    def test_every_jurisdiction_with_a_rule_refuses_a_non_positive_string(self) -> None:
        import json

        from codify.calendar import to_gregorian_year
        from codify.jurisdictions import JURISDICTIONS_DIR

        # Resolve configuration paths independently of pytest's working directory.
        codes = [
            path.parent.name
            for path in sorted(JURISDICTIONS_DIR.glob("*/config.json"))
            if (json.loads(path.read_text()).get("frbr") or {}).get("calendar_conversion")
        ]
        assert codes, "no jurisdiction declares a conversion rule"
        for code in codes:
            for raw in (0, -5, "0", "-5", " 0 ", "+0"):
                with pytest.raises(CalendarConversionError):
                    to_gregorian_year(raw, code)

    def test_a_real_year_still_converts_on_each_kind(self) -> None:
        from codify.calendar import to_gregorian_year

        assert to_gregorian_year(2565, "th") == 2022
        assert to_gregorian_year("令和6", "jp") == 2024
        assert to_gregorian_year(114, "tw") == 2025
        assert to_gregorian_year("1443", "sa") == 2021


class TestAMonthIsReadOnItsOwnGrid:
    """Anchored on the calendars themselves: 1 Baisakh 2080 was 14 April 2023 and
    1 Meskerem 2016 was 12 September 2023, so each year straddles two Gregorian."""

    @staticmethod
    def _gregorian_grid(kind: str, **fields: int) -> CalendarConversion:
        return CalendarConversion(
            kind=kind,
            month_day_is_gregorian=True,
            month_names=[f"m{i}" for i in range(12)],
            **fields,
        )

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "expected"),
        [
            # BS 2080 ran 14 April 2023 to 12 April 2024.
            ("bikram_samvat", "2080", 6, 2023),
            ("bikram_samvat", "2080", 12, 2023),
            ("bikram_samvat", "2080", 1, 2024),
            ("bikram_samvat", "2080", 3, 2024),
            # EC 2016 ran 12 September 2023 to 10 September 2024.
            ("ethiopian", "2016", 10, 2023),
            ("ethiopian", "2016", 12, 2023),
            ("ethiopian", "2016", 1, 2024),
            ("ethiopian", "2016", 8, 2024),
        ],
    )
    def test_a_gregorian_month_lands_in_the_year_the_calendar_says(
        self, kind: str, local_year: str, month: int, expected: int
    ) -> None:
        assert _apply_rule(local_year, self._gregorian_grid(kind), month=month) == expected

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "day", "expected"),
        [
            # 14 April 2023 opened BS 2080; 12 April 2024 closed it.
            ("bikram_samvat", "2080", 4, 14, 2023),
            ("bikram_samvat", "2080", 4, 30, 2023),
            ("bikram_samvat", "2080", 4, 12, 2024),
            ("bikram_samvat", "2080", 4, 1, 2024),
            # 12 September 2023 opened EC 2016; 10 September 2024 closed it.
            ("ethiopian", "2016", 9, 12, 2023),
            ("ethiopian", "2016", 9, 30, 2023),
            ("ethiopian", "2016", 9, 10, 2024),
            ("ethiopian", "2016", 9, 1, 2024),
        ],
    )
    def test_the_new_year_month_is_settled_by_its_day(
        self, kind: str, local_year: str, month: int, day: int, expected: int
    ) -> None:
        assert _apply_rule(local_year, self._gregorian_grid(kind), month=month, day=day) == expected

    def test_a_declared_new_year_day_settles_the_whole_month(self) -> None:
        """A jurisdiction naming the day removes the band: on 13 April, which the
        default band leaves open, the declared day decides."""
        rule = self._gregorian_grid("bikram_samvat", new_year_month=4, new_year_day=13)
        assert _apply_rule("2080", rule, month=4, day=13) == 2023
        assert _apply_rule("2080", rule, month=4, day=12) == 2024

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "day"),
        [("bikram_samvat", "2080", 4, 13), ("ethiopian", "2016", 9, 11)],
        ids=["the day the new year moves over", "the day the new year moves over, ethiopian"],
    )
    def test_the_new_year_month_without_a_certain_day_settles_nothing(
        self, kind: str, local_year: str, month: int, day: int
    ) -> None:
        rule = self._gregorian_grid(kind)
        blind = _apply_rule(local_year, rule, month=None)
        assert _apply_rule(local_year, rule, month=month, day=day) == blind
        assert _apply_rule(local_year, rule, month=month, day=None) == blind

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "day", "expected"),
        [
            # 1 Poush 2080 was 16 December 2023; 30 Poush 2080 was 14 January 2024.
            ("bikram_samvat", "2080", 9, 1, 2023),
            ("bikram_samvat", "2080", 9, 15, 2023),
            ("bikram_samvat", "2080", 9, 18, 2024),
            ("bikram_samvat", "2080", 9, 30, 2024),
            ("bikram_samvat", "2080", 8, None, 2023),
            ("bikram_samvat", "2080", 1, None, 2023),
            ("bikram_samvat", "2080", 10, None, 2024),
            ("bikram_samvat", "2080", 12, None, 2024),
            # 1 Tahsas 2016 was 11 December 2023; 30 Tahsas 2016 was 9 January 2024.
            ("ethiopian", "2016", 4, 1, 2023),
            ("ethiopian", "2016", 4, 21, 2023),
            ("ethiopian", "2016", 4, 23, 2024),
            ("ethiopian", "2016", 4, 30, 2024),
            ("ethiopian", "2016", 1, None, 2023),
            ("ethiopian", "2016", 3, None, 2023),
            ("ethiopian", "2016", 5, None, 2024),
            ("ethiopian", "2016", 13, None, 2024),
        ],
    )
    def test_a_local_month_lands_in_the_year_the_calendar_says(
        self, kind: str, local_year: str, month: int, day: int | None, expected: int
    ) -> None:
        rule = CalendarConversion(kind=kind)
        assert _apply_rule(local_year, rule, month=month, day=day, month_grid="local") == expected

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "day"),
        [
            ("bikram_samvat", "2080", 9, None),
            ("bikram_samvat", "2080", 9, 16),
            ("ethiopian", "2016", 4, None),
            ("ethiopian", "2016", 4, 22),
        ],
    )
    def test_the_month_straddling_january_without_a_certain_day_settles_nothing(
        self, kind: str, local_year: str, month: int, day: int | None
    ) -> None:
        rule = CalendarConversion(kind=kind)
        assert _apply_rule(
            local_year, rule, month=month, day=day, month_grid="local"
        ) == _apply_rule(local_year, rule, month=None)

    @pytest.mark.parametrize(
        ("declared_gregorian", "month_grid", "expected"),
        [
            # The calendar's own grid, undeclared: January is Baisakh, 2023.
            (False, "local", 2023),
            # Named Gregorian, whatever the config declares: January, 2024.
            (False, "gregorian", 2024),
            (True, "gregorian", 2024),
            # A declared Gregorian grid makes the calendar's own months Gregorian.
            (True, "local", 2024),
        ],
    )
    def test_the_month_is_read_on_the_grid_the_caller_names(
        self, declared_gregorian: bool, month_grid: str, expected: int
    ) -> None:
        """Which grid month 1 of BS 2080 is on decides the year, and the
        keyword names it."""
        rule = (
            self._gregorian_grid("bikram_samvat")
            if declared_gregorian
            else CalendarConversion(kind="bikram_samvat")
        )
        assert _apply_rule("2080", rule, month=1, day=20, month_grid=month_grid) == expected

    def test_a_bare_public_call_reads_the_month_as_gregorian(self) -> None:
        """The public argument keeps the meaning it shipped with: a month named
        by a caller that says nothing about the grid is a Gregorian month."""
        assert _apply_rule("2080", CalendarConversion(kind="bikram_samvat"), month=1) == 2024
        assert to_gregorian_year("2080", "xnp", month=1) == 2024

    @pytest.mark.parametrize(
        ("kind", "local_year", "month", "day", "month_grid"),
        [
            # Off the Gregorian grid: month 0, month 13, day 0, day 99.
            ("bikram_samvat", "2080", 0, 10, "gregorian"),
            ("bikram_samvat", "2080", 13, 10, "gregorian"),
            ("bikram_samvat", "2080", 4, 0, "gregorian"),
            ("bikram_samvat", "2080", 4, 99, "gregorian"),
            ("ethiopian", "2016", 0, 10, "gregorian"),
            ("ethiopian", "2016", 13, 10, "gregorian"),
            ("ethiopian", "2016", 9, 0, "gregorian"),
            ("ethiopian", "2016", 9, 40, "gregorian"),
            # Off the calendar's own grid: month 0, one past its last, day 0, day 40.
            ("bikram_samvat", "2080", 0, 10, "local"),
            ("bikram_samvat", "2080", 13, 10, "local"),
            ("bikram_samvat", "2080", 9, 0, "local"),
            ("bikram_samvat", "2080", 9, 40, "local"),
            ("ethiopian", "2016", 0, 10, "local"),
            ("ethiopian", "2016", 14, 10, "local"),
            ("ethiopian", "2016", 4, 0, "local"),
            ("ethiopian", "2016", 4, 40, "local"),
        ],
    )
    def test_a_value_off_the_grid_settles_nothing(
        self, kind: str, local_year: str, month: int, day: int, month_grid: str
    ) -> None:
        """A month or day no grid holds cannot say which side of the new year the
        date fell; the year converts as it does with no month at all."""
        rule = (
            self._gregorian_grid(kind)
            if month_grid == "gregorian"
            else CalendarConversion(kind=kind)
        )
        blind = _apply_rule(local_year, rule, month=None)
        assert _apply_rule(local_year, rule, month=month, day=day, month_grid=month_grid) == blind

    @pytest.mark.parametrize(
        ("month", "day"),
        [(0, 10), (13, 10), (2, 0), (2, 99)],
        ids=["month 0", "month 13", "day 0", "day 99"],
    )
    def test_a_value_off_the_grid_takes_no_reform_shift(self, month: int, day: int) -> None:
        """The reform shift reads the month against the new-year month; a value
        off the grid cannot be read against it, so a pre-reform year stays."""
        rule = self._gregorian_grid("buddhist", new_year_month=4, new_year_reform_year=2484)
        assert _apply_rule("2478", rule, month=month, day=day) == _apply_rule(
            "2478", rule, month=None
        )
        assert (
            _apply_rule("2478", rule, month=2, day=10) == _apply_rule("2478", rule, month=None) + 1
        )

    def test_a_thirteenth_month_is_no_month_of_the_gregorian_grid(self) -> None:
        rule = self._gregorian_grid("bikram_samvat")
        assert _apply_rule("2080", rule, month=13, day=1) == _apply_rule("2080", rule, month=None)
        rule = CalendarConversion(kind="ethiopian")
        assert _apply_rule("2016", rule, month=13, month_grid="local") == 2024

    def test_a_month_of_a_grid_no_table_describes_settles_nothing(self) -> None:
        """A local month on a grid nothing describes cannot meet the reform's
        Gregorian-side threshold; the same month named Gregorian can."""
        rule = CalendarConversion(kind="buddhist", new_year_month=4, new_year_reform_year=2484)
        blind = _apply_rule("2478", rule, month=None)
        assert _apply_rule("2478", rule, month=2, month_grid="local") == blind
        assert _apply_rule("2478", rule, month=2, month_grid="gregorian") == blind + 1
        assert reform_shift(rule, 2478, 2, on_gregorian_grid=False) == 0


class TestEveryAliasReachesTheDeclaredRule:
    """The prompt and the generic converter each accept spellings of a calendar
    the config names another way; one alias set folds them all, so no spelling
    a model emits can bypass a declared rule."""

    ALIASES = [
        ("lunar_hijri", "hijri_lunar", ["hijri", "lunar_hijri", "hijri_lunar", "islamic"]),
        ("solar_hijri", "hijri_solar", ["hijri_solar", "solar_hijri", "persian", "jalali"]),
        ("buddhist_era", "buddhist", ["buddhist", "buddhist_era", "thai"]),
        ("ethiopian", "ethiopian", ["ethiopian", "ethiopic"]),
        ("minguo", "epoch_offset", ["minguo", "roc", "taiwan"]),
        ("japanese_era", "era_table", ["japanese_era"]),
    ]

    @pytest.mark.parametrize(("declared", "kind", "labels"), ALIASES, ids=[a[0] for a in ALIASES])
    def test_each_label_names_the_declared_calendar(
        self, declared: str, kind: str, labels: list[str], tmp_path, monkeypatch
    ) -> None:
        from codify.calendar import declares_this_calendar
        from tests.config_fixtures import isolated_configs

        rule: dict[str, object] = {"kind": kind}
        if kind == "epoch_offset":
            rule["epoch_year"] = 1911
        if kind == "era_table":
            rule["eras"] = [{"name": "Reiwa", "abbrev": "令和", "start": "2019-05-01"}]
        configs = {
            "xq": {
                "calendar": declared,
                "frbr": {"country_code": "xq", "calendar_conversion": rule},
            }
        }
        with isolated_configs(monkeypatch, tmp_path / "j", configs):
            for label in labels:
                assert declares_this_calendar(label, "xq"), label
                assert declares_this_calendar(f" {label.upper()} ", "xq"), label
            assert not declares_this_calendar("gregorian", "xq")

    def test_the_generic_converter_reads_the_same_aliases(self) -> None:
        assert year_from_calendar(2566, "thai") == year_from_calendar(2566, "buddhist_era") == 2023
        assert year_from_calendar(114, "roc") == year_from_calendar(114, "minguo") == 2025
        assert year_from_calendar(1445, "islamic") == year_from_calendar(1445, "lunar_hijri")
        assert year_from_calendar(1403, "jalali") == year_from_calendar(1403, "solar_hijri")
