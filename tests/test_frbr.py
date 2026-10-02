"""Canonical FRBR builder."""

from __future__ import annotations

from datetime import date

import pytest

from codify.agents.tools.frbr import construct_frbr_uri
from codify.frbr import (
    UncitableFrbrUri,
    build_frbr_expression_uri,
    build_frbr_work_uri,
    default_token,
    expression_uri_date,
    law_number_token,
    parse_source_ref,
    series_number,
    token_country,
)
from codify.jurisdictions import try_load_config


def test_work_uri_from_config_pattern() -> None:
    try_load_config.cache_clear()
    # AL patterns are canonical (doctype-keyed) post-W4.
    assert build_frbr_work_uri("al", "ligj", 2020, "162") == "/akn/al/act/ligj/2020/162"
    assert build_frbr_work_uri("al", "vendim", 2021, "285") == "/akn/al/act/vendim/2021/285"


def test_work_uri_fallback_for_unconfigured() -> None:
    # No config / no pattern → canonical default (matches adapter output).
    assert build_frbr_work_uri("zz", "foo", 2024, "1") == "/akn/zz/act/foo/2024/1"


def test_work_uri_act_doctype_has_no_subtype() -> None:
    # A plain "act" (or empty) doctype collapses to /akn/{cc}/act/{year}/{number}
    # with no doubled `act/act`. Load-bearing for the demo seed's base-law shape.
    assert build_frbr_work_uri("zz", "act", 2024, "1") == "/akn/zz/act/2024/1"
    assert build_frbr_work_uri("zz", "", 2024, "1") == "/akn/zz/act/2024/1"


def test_work_uri_yearless_draft_gets_the_unknown_date_placeholder() -> None:
    # Was: omit the year segment. Cobalt refuses both `/akn/ua/act//draft-abc`
    # and `/akn/ua/act/draft-abc`, so omitting it swapped one unparseable
    # identity for another. AKN wants the segment present.
    assert build_frbr_work_uri("ua", "act", "", "draft-abc") == "/akn/ua/act/0001/draft-abc"


def test_work_uri_pattern_needing_extra_field_falls_back() -> None:
    # AL's `kod` pattern needs {subtype}; the builder supplies only year/number,
    # so it falls back to the canonical default (and logs the under-served field).
    try_load_config.cache_clear()
    assert build_frbr_work_uri("al", "kod", 2017, "7905") == "/akn/al/act/kod/2017/7905"


def test_work_uri_unknown_doctype_for_configured_jurisdiction() -> None:
    # Config exists but has no pattern for this doctype → canonical default.
    try_load_config.cache_clear()
    assert build_frbr_work_uri("al", "udhezim", 2022, "3") == "/akn/al/act/udhezim/2022/3"


def test_construct_frbr_uri_agent_tool_delegates() -> None:
    # The agent tool is a thin delegate; it must emit the canonical `act/` shape.
    try_load_config.cache_clear()
    assert construct_frbr_uri("al", "vendim", "2021", "285") == "/akn/al/act/vendim/2021/285"
    assert construct_frbr_uri("ke", "act", "2023", "47") == "/akn/ke/act/2023/47"


def test_migration_canonical_shape_matches_builder() -> None:
    # The 0035 migration hard-codes /akn/al/act/{doctype}/{year}/{number} in SQL.
    # For the doctypes that actually exist in the live AL data (ligj, vendim) the
    # builder must produce the identical shape, or a re-ingest forks the Law.
    try_load_config.cache_clear()
    for doctype in ("ligj", "vendim"):
        migration_shape = f"/akn/al/act/{doctype}/2020/162"
        assert build_frbr_work_uri("al", doctype, 2020, "162") == migration_shape


def test_work_uri_is_idempotent_inputs() -> None:
    # Same inputs → same URI (so a re-ingest never forks the Law).
    a = build_frbr_work_uri("al", "vendim", "2021", "285")
    b = build_frbr_work_uri("al", "vendim", 2021, "285")
    assert a == b == "/akn/al/act/vendim/2021/285"


def test_expression_uri_composition() -> None:
    work = "/akn/al/act/ligj/2020/162"
    assert build_frbr_expression_uri(work, "sqi", date(2020, 12, 23)) == f"{work}/sqi@2020-12-23"
    assert build_frbr_expression_uri(work, "sqi", "2026-06-17") == f"{work}/sqi@2026-06-17"
    # A full ISO timestamp is truncated to the date (the [:10] slice).
    ts = "2020-12-23T09:00:00Z"
    assert build_frbr_expression_uri(work, "sqi", ts) == f"{work}/sqi@2020-12-23"
    # Defensive: no date → no @ suffix.
    assert build_frbr_expression_uri(work, "sqi", None) == f"{work}/sqi"
    # An empty string is no date either. It is falsy but not None, so an `is
    # None` test alone emitted a trailing bare `@` and minted an expression URI
    # no row could match.
    assert build_frbr_expression_uri(work, "sqi", "") == f"{work}/sqi"


def test_work_uri_string_none_year_becomes_the_placeholder():
    # str(None) leaking from upstream metadata must not mint /None/ segments;
    # it means "year unresolved", which is what the placeholder says.
    assert build_frbr_work_uri("ua", "act", "None", "x1") == "/akn/ua/act/0001/x1"


def test_work_uri_null_and_none_year_sentinels() -> None:
    # str(None) leaks, JSON "null" and blank all mean "year unresolved", and all
    # resolve to the placeholder rather than to a missing segment.
    assert build_frbr_work_uri("xz", "act", "None", "7") == "/akn/xz/act/0001/7"
    assert build_frbr_work_uri("xz", "act", "null", "7") == "/akn/xz/act/0001/7"
    assert build_frbr_work_uri("xz", "act", "  ", "7") == "/akn/xz/act/0001/7"


def test_configured_template_never_formats_sentinel_year() -> None:
    # xz declares uri_patterns for act; the sentinel must resolve before
    # the template branch or it mints /null/ segments there.
    try_load_config.cache_clear()
    for sentinel in ("None", "null", "", None):
        uri = build_frbr_work_uri("xz", "act", sentinel, "7")  # type: ignore[arg-type]
        assert "null" not in uri.lower() and "none" not in uri.lower()
        assert "//" not in uri.removeprefix("/")


def test_builder_refuses_a_uri_with_an_empty_segment() -> None:
    # The last gate before an identity is stored. Returning `/akn/xz/act/2009/`
    # names a law by a hole: nobody can cite it and Cobalt cannot parse it, so
    # the ingest fails here rather than the defect reaching a unique index.
    with pytest.raises(UncitableFrbrUri) as exc:
        build_frbr_work_uri("xz", "act", "2009", "")
    # The inputs are what was wrong, so the error names them, not just the result.
    assert "country='xz'" in str(exc.value)

    with pytest.raises(UncitableFrbrUri):
        build_frbr_work_uri("", "act", "2009", "7")


def test_number_token_rejects_an_office_routing_code() -> None:
    # The defect this closes: a model asked for a number returned the whole
    # parenthetical, and `( /11/11 ر.م.و/ إ.هـ)` was formatted into an FRBR path.
    assert law_number_token("ر.م.و.إ.هـ") == ""
    assert law_number_token("11") == "11"
    assert law_number_token("5A") == "5A"


def test_expression_uri_date_reads_what_build_wrote():
    work = "/akn/xz/act/1999/7"
    assert expression_uri_date(build_frbr_expression_uri(work, "ara", "1999-06-08")) == "1999-06-08"
    # A component tail and a manifestation format both sit after the date, so
    # neither may be mistaken for part of it.
    assert expression_uri_date(f"{work}/ara@1999-06-08/!schedule_1") == "1999-06-08"
    assert expression_uri_date(f"{work}/ara@1999-06-08.akn") == "1999-06-08"


def test_expression_uri_date_is_none_when_the_uri_names_none():
    # What `build_frbr_expression_uri(..., None)` mints, and what 158 stored rows
    # carry. Not a date of "today"; the URI simply does not name one.
    assert expression_uri_date("/akn/xz/act/1999/7/ara") is None
    assert expression_uri_date("/akn/xz/act/1999/7") is None
    assert expression_uri_date(None) is None
    assert expression_uri_date("") is None


def test_expression_uri_date_refuses_a_shape_it_cannot_read_correctly():
    """A reader that returns the wrong date is worse than one that returns none.
    `@2026-08-021` matched a prefix and handed back `2026-08-02`, which rebuilds a
    URI the row does not hold and forks the version the caller means to replace."""
    work = "/akn/xz/act/1999/7"
    assert expression_uri_date(f"{work}/eng@2026-08-021") is None
    assert expression_uri_date(f"{work}/eng@2026-13-45") is None
    assert expression_uri_date(f"{work}/eng@2026-02-30") is None


def test_expression_uri_date_takes_the_first_segment_when_a_uri_carries_two():
    """Not a shape our writers emit, but a URI is a stored string. Pinned so the
    answer is a decision rather than an accident of the regex."""
    assert expression_uri_date("/akn/xz/act/1999/7/ara@1999-06-08/eng@2026-08-02") == "1999-06-08"


class TestSeriesNumber:
    """A court case number is `{number}/{series}/{year}`. Discarding it whole
    cost judgments their citable identity."""

    def test_a_case_number_yields_the_number_it_starts_with(self) -> None:
        assert series_number("45/PUL-IX/2011") == "45"
        assert series_number("132/PUL-XXII/2024") == "132"

    def test_arabic_indic_digits_are_folded_first(self) -> None:
        assert series_number("٤٥/PUL-IX/٢٠١١") == "45"

    def test_a_roman_first_segment_is_not_a_number(self) -> None:
        """`I/DEW/2001` is a citation shape, not a path segment: the URI
        number space is numeric, so this still falls to the draft address."""
        assert series_number("I/DEW/2001") == ""

    def test_a_compound_law_number_is_left_to_the_guard(self) -> None:
        """`4/2016` is a whole law number, not a series citation, and
        `law_number_token` already accepts it."""
        assert series_number("4/2016") == ""
        assert law_number_token("4/2016") == "4/2016"

    def test_an_office_routing_code_is_still_refused(self) -> None:
        """The shape this narrowing must not admit."""
        assert series_number("2008-04-13/11/11") == ""
        assert series_number("ر.م.و.إ.هـ") == ""


class TestPublisherTokens:
    def test_the_default_token_follows_the_frbr_country_not_the_code(self) -> None:
        """A jurisdiction publishing under another country's URIs takes its tokens."""
        assert default_token("gb", "act") == "ukpga"
        assert default_token("gb-eng", "act") == "ukpga"
        assert default_token("gb-eng", "si") == "uksi"
        assert default_token("xq", "act") == "act"

    def test_a_token_names_the_country_that_publishes_it(self) -> None:
        assert token_country("asp") == "gb"
        assert token_country("uksi") == "gb"
        assert token_country("eur") is None
        assert token_country("act") is None


@pytest.mark.parametrize("suffix", ["2024/32", "0001/32", "2024/32/main#art_2", "si/2024/32"])
def test_akn_country_segment_is_not_an_external_publisher_prefix(suffix: str) -> None:
    uri = f"/akn/xl/act/{suffix}"
    work, _, eid = uri.partition("#")
    assert parse_source_ref(uri) == (work.removesuffix("/main"), eid or None)


@pytest.mark.parametrize(
    "uri,expected",
    [
        (
            "https://www.legislation.gov.uk/id/ukpga/2024/32",
            ("/akn/gb/act/ukpga/2024/32", None),
        ),
        (
            "https://www.legislation.gov.uk/id/ukpga/2024/32/section/2",
            ("/akn/gb/act/ukpga/2024/32", "section-2"),
        ),
        ("/european/regulation/2024/0032", ("/akn/eu/act/reg/2024/32", None)),
        ("/european/directive/2024/0032", ("/akn/eu/act/dir/2024/32", None)),
        ("/european/unknown/2024/0032", None),
        ("https://www.legislation.gov.uk/id/unknown/2024/32", None),
        ("https://unsupported.example/id/act/2024/32", None),
    ],
)
def test_external_source_dispatch_remains_unchanged(uri: str, expected: object) -> None:
    assert parse_source_ref(uri) == expected


@pytest.fixture(autouse=True)
def route_configs(tmp_path, monkeypatch):
    from tests.config_fixtures import isolated_configs

    configs = {
        "gb-eng": {"frbr": {"country_code": "gb"}},
        "eac": {"frbr": {"country_code": "aa-eac"}},
        "eu": {
            "frbr": {
                "country_code": "eu",
                "uri_patterns": {
                    "regulation": "/akn/eu/act/reg/{year}/{number}",
                    "directive": "/akn/eu/act/dir/{year}/{number}",
                },
            }
        },
    }
    with isolated_configs(monkeypatch, tmp_path / "data" / "jurisdictions", configs):
        yield
