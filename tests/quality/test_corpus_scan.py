"""The corpus measurement itself. A harness that flatters the scanner is worse
than none. Each test names the wrong number it prevents."""

from __future__ import annotations

from pathlib import Path

import pytest

from codify.jurisdictions import JurisdictionConfigError, load_config
from codify.quality.corpus_scan import (
    aggregate,
    is_derived_export,
    marker_form,
    scan_corpus,
    scan_text,
)

_TWO_ARTICLES = "مادة (1)\nنص الأولى.\n\nمادة (2)\nنص الثانية.\n"

# Bidi controls as RTL extraction leaves them; unnormalised this finds nothing.
_BIDI_WRAPPED = "‫مادة ‪(1)‬‬\nنص.\n\n‫مادة ‪(2)‬‬\nنص.\n"


def _scan(text: str, doctype: str = "act"):
    return scan_text(text, config=load_config("xz"), country="xz", doctype=doctype)


class TestNormalisation:
    def test_bidi_wrapped_markers_are_still_counted(self) -> None:
        """Without normalisation a Bidi-wrapped marker is never counted."""
        assert _scan(_BIDI_WRAPPED).basic_units == 2

    def test_a_plain_document_counts_its_articles(self) -> None:
        assert _scan(_TWO_ARTICLES).basic_units == 2


class TestMarkerCensus:
    def test_a_refused_form_is_counted_as_present_and_unclaimed(self) -> None:
        """A refused marker must still reach the denominator, or it is invisible."""
        scan = _scan("المادة السابقة تنطبق على هذه الحالة.\n")
        present, claimed = scan.marker_forms["word_or_prose"]
        assert present == 1
        assert claimed == 0

    def test_the_dash_and_unbalanced_paren_forms_are_claimed(self) -> None:
        """Both shapes were refused by the separator class."""
        for text, form in (
            ("مادة - 1\nنص.\n\nمادة - 2\nنص.\n", "dash"),
            ("مادة )1\nنص.\n\nمادة )2\nنص.\n", "unbalanced_close"),
        ):
            assert _scan(text).marker_forms[form] == [2, 2], form

    def test_a_claimed_marker_is_counted_against_its_own_form(self) -> None:
        present, claimed = _scan(_TWO_ARTICLES).marker_forms["balanced_parens"]
        assert (present, claimed) == (2, 2)

    def test_one_anchor_cannot_claim_two_census_lines(self) -> None:
        """A keyword line with no number cannot anchor, so it must not be
        credited by its neighbour."""
        scan = _scan("مادة\nمادة (1)\nنص.\n")
        assert scan.marker_forms["nothing_follows"] == [1, 0]
        assert scan.marker_forms["balanced_parens"] == [1, 1]

    def test_forms_are_named_by_what_follows_the_keyword(self) -> None:
        assert marker_form("(1)") == "balanced_parens"
        assert marker_form(")1") == "unbalanced_close"
        assert marker_form(")1(") == "mirrored_parens"
        assert marker_form("- 1") == "dash"
        assert marker_form("— ٢٧") == "dash"
        assert marker_form("1") == "bare"
        assert marker_form("") == "nothing_follows"
        assert marker_form("جديدة") == "word_or_prose"


class TestGroupingRecall:
    def test_a_grouping_keyword_that_produced_no_anchor_is_visible(self) -> None:
        """A dropped chapter must read as present-and-unfound, not as absent."""
        text = "الفصل الثاني\n\n" + _TWO_ARTICLES
        present, found = _scan(text).container_recall["chapter"]
        assert present == 1
        # Whether the scan keeps this one is the anchors module's business; that
        # it is visible as present is this module's.
        assert found <= present

    def test_recall_cannot_exceed_the_lines_it_was_measured_against(self) -> None:
        """The numerator counted every anchor of the kind while the denominator
        counted line-initial keywords: an off-column chapter gave 200%."""
        text = "الفصل الأول\n\nمادة (1)\nنص.\n\nنص العمود    الفصل الثاني\n\nمادة (2)\nنص.\n"
        present, found = _scan(text).container_recall["chapter"]
        assert found <= present

    def test_subdivisions_are_not_scored_as_grouping_levels(self) -> None:
        """Subdivision markers are mid-line, so a line-initial count of them
        yields a rate above 100%."""
        assert "paragraph" not in _scan(_TWO_ARTICLES).container_recall

    def test_a_grouping_keyword_the_class_never_declares_is_reported(self) -> None:
        """`xz/decree` declares no chapter, so silence would read as perfect recall."""
        scan = scan_text(
            "الفصل الأول\n\n" + _TWO_ARTICLES,
            config=load_config("xz"),
            country="xz",
            doctype="decree",
        )
        assert scan.container_recall == {}
        assert scan.undeclared_grouping_lines["chapter"] == 1


class TestProvenance:
    def test_our_own_exports_are_recognised(self) -> None:
        """Scoring our own exports scores the pipeline against itself."""
        assert is_derived_export(Path("Law 5 of 1999.ar.txt"))
        assert is_derived_export(Path("Law 5 of 1999.he.txt"))
        assert not is_derived_export(Path("Law 5 of 1999.txt"))
        assert not is_derived_export(Path("قانون رقم 5 لسنة 1999.txt"))

    def test_derived_exports_are_excluded_but_counted(self, tmp_path: Path) -> None:
        """A silently shrunken corpus is worse than no rate, so the exclusion
        has to appear in the summary."""
        (tmp_path / "law.txt").write_text(_TWO_ARTICLES, encoding="utf-8")
        (tmp_path / "law.AR.txt").write_text(_TWO_ARTICLES, encoding="utf-8")
        sweep = scan_corpus(tmp_path, jurisdiction="xz")
        assert len(sweep.scans) == 1
        assert aggregate(sweep)["documents_excluded_as_our_own_export"] == 1
        assert len(scan_corpus(tmp_path, jurisdiction="xz", include_derived=True).scans) == 2

    def test_an_unreadable_file_is_counted_not_scored(self, tmp_path: Path) -> None:
        """A wrong-codec file scans to zero anchors, which reads as a
        structuring failure rather than a file we could not read."""
        (tmp_path / "law.txt").write_text(_TWO_ARTICLES, encoding="utf-8")
        (tmp_path / "cp1256.txt").write_bytes("مادة (1)\nنص.\n".encode("cp1256"))
        sweep = scan_corpus(tmp_path, jurisdiction="xz")
        assert len(sweep.scans) == 1
        assert aggregate(sweep)["documents_unreadable"] == 1

    def test_an_unknown_jurisdiction_raises(self, tmp_path: Path) -> None:
        """Every document would score zero against English builtins, which reads
        as a corpus-wide collapse rather than a typo."""
        (tmp_path / "law.txt").write_text(_TWO_ARTICLES, encoding="utf-8")
        with pytest.raises(JurisdictionConfigError, match="no jurisdiction config"):
            scan_corpus(tmp_path, jurisdiction="zz-not-a-country")


class TestCensusVocabulary:
    def test_every_census_alias_is_grouped_under_the_kind_the_scan_resolves(self) -> None:
        """`gb` declares `SCHEDULE` as `schedule` while the pattern answers
        `hcontainer`; counting by the config would show a level as
        present-and-unfound when it was found under another name."""
        from codify.pipeline.enrich.anchors import _kind_from_match, cached_regex, keyword_aliases
        from codify.quality.corpus_scan import _census_aliases

        for country, doctype in (("xz", "act"), ("gb", "act"), ("al", "ligj"), ("ua", "law")):
            regex = cached_regex(country, doctype)
            grouped = _census_aliases(keyword_aliases(load_config(country), doctype), regex)
            for kind, terms in grouped.items():
                for term in terms:
                    match = regex.search(f"\n{term} 5\n")
                    resolved = _kind_from_match(match) if match else None
                    assert resolved in (None, kind), (country, term, kind, resolved)


class TestDoctypeClassification:
    def test_the_year_in_the_filename_selects_the_document_class(self, tmp_path: Path) -> None:
        """The doctype picks the hierarchy recall is scored against, so a 1999
        law classified as a modern decree-law is scored against the wrong levels."""
        from codify.quality.corpus_scan import scan_file

        cfg = load_config("xz")
        for name, expected in (
            ("قانون رقم 5 لسنة 1999", "act"),
            ("قرار بقانون رقم 3 لسنة 2016", "decree_law"),
        ):
            path = tmp_path / f"{name}.txt"
            path.write_text(_TWO_ARTICLES, encoding="utf-8")
            assert scan_file(path, config=cfg, country="xz").doctype == expected

    def test_a_filename_with_no_year_still_classifies(self, tmp_path: Path) -> None:
        """A hash-named or scanner-named file must not raise; it falls to the
        jurisdiction default, which is why the summary carries the doctype."""
        from codify.quality.corpus_scan import scan_file

        path = tmp_path / "a3f9c2e1.txt"
        path.write_text(_TWO_ARTICLES, encoding="utf-8")
        assert scan_file(path, config=load_config("xz"), country="xz").doctype


class TestAggregate:
    def test_the_honest_rate_counts_markers_no_alias_matches(self) -> None:
        """An alias census cannot see a mangled keyword, so a rate over it alone
        flatters the scanner. `منها` is a real word and stays unresolved."""
        text = "مادة (1)\nنص.\n\nمنها (20)\nنص.\n\nمادة (40)\nنص.\n"
        summary = aggregate([_scan(text)])
        assert summary["mangled_marker_lines"] == 1
        assert summary["honest_marker_lines"] == summary["marker_lines"] + 1
        assert summary["honest_capture_rate"] < summary["capture_rate"]

    def test_a_corpus_with_no_markers_reports_no_rate(self) -> None:
        """None, never 1.0: a rate over nothing is not a perfect score."""
        summary = aggregate([_scan("نص بلا أي علامات هيكلية.\n")])
        assert summary["capture_rate"] is None
        assert summary["honest_capture_rate"] is None

    def test_totals_add_across_documents(self) -> None:
        summary = aggregate([_scan(_TWO_ARTICLES), _scan(_TWO_ARTICLES)])
        assert summary["documents"] == 2
        assert summary["basic_units"] == 4
        assert summary["marker_forms"]["balanced_parens"] == [4, 4]


class TestUnmeasurableDocuments:
    def test_a_class_with_no_provision_level_is_not_a_miss(self) -> None:
        """`ohada/avis` and the `sadc/*` classes declare no article, section or
        rule. Scoring them as zero-article reported a miss they cannot commit."""
        scan = scan_text(
            "Article 1\nBody.\n", config=load_config("ohada"), country="ohada", doctype="avis"
        )
        assert scan.measurable is False
        summary = aggregate([scan])
        assert summary["documents"] == 1
        assert summary["documents_measurable"] == 0
        assert summary["documents_without_basic_unit"] == 0

    def test_a_class_with_no_grouping_level_cannot_lose_a_container(self) -> None:
        """`xz/decree` declares article and paragraph only, so counting it as a
        document that lost its chapters is counting an impossibility."""
        scan = scan_text(_TWO_ARTICLES, config=load_config("xz"), country="xz", doctype="decree")
        assert scan.grouping_declared is False
        assert aggregate([scan])["documents_with_units_and_no_container"] == 0


class TestPassAccounting:
    """Pass accounting distinguishes a no-op pass from one that never ran."""

    def test_a_pass_that_ran_and_changed_nothing_reports_zero(self) -> None:
        """Absent means did-not-run. Folding the two together would read a
        never-executed pass as evidence that it earns its place."""
        fires = _scan(_TWO_ARTICLES).fires
        assert fires["drop_toc_duplicates"] == 0
        assert "drop_uk_prose_containers" not in fires

    def test_a_producing_pass_counts_the_anchors_it_added(self) -> None:
        assert _scan(_TWO_ARTICLES).fires["regex"] == 2

    def test_a_deleting_pass_counts_negative(self) -> None:
        """`_drop_toc_duplicates` removes the TOC twin of a body article; a
        signless count would read a deletion as a contribution."""
        text = "مادة (1)\nمادة (2)\n\nمادة (1)\nنص الأولى.\n\nمادة (2)\nنص الثانية.\n"
        assert _scan(text).fires["drop_toc_duplicates"] < 0

    def test_a_declaring_pass_counts_the_spans_it_raised(self) -> None:
        """They touch no anchor, so an anchor delta would call them all dead."""
        scan = _scan("مادة (1)\nنص.\n\nمنها (3)\nنص.\n")
        assert scan.fires["declare_unclaimed_markers"] == 1
        assert scan.fires["declare_toc_without_body"] == 0

    def test_the_two_declaring_passes_under_one_call_are_told_apart(self) -> None:
        """`_declare_numbering_gaps` also calls `_declare_out_of_order`, so a
        span delta over the pair credits both to whichever name is written
        first. Over the corpus that was 1,275 gaps where 494 are gaps and 781
        are out-of-order."""
        scan = _scan("مادة (1)\nنص.\n\nمادة (3)\nنص.\n\nمادة (2)\nنص.\n")
        assert scan.fires["declare_out_of_order"] > 0
        assert "declare_numbering_gaps" in scan.fires

    def test_two_passes_sharing_a_span_kind_are_told_apart(self) -> None:
        """`duplicate_number` was one number covering TOC dedup and eId
        collision; `orphan_text` covered orphan drop and compilation wrapper."""
        text = "مادة (1)\nمادة (2)\n\nمادة (1)\nنص.\n\nمادة (2)\nنص.\n"
        assert _scan(text).spans_by_pass["drop_toc_duplicates"] > 0

    def test_every_pass_that_raised_a_span_has_a_fire_count(self) -> None:
        """The general form of the `declare_out_of_order` defect: a pass can run,
        raise spans and never appear in `fires`, where the contract reads it as
        never-executed and a dead-pass sweep reads that as grounds for deleting it."""
        out_of_order = "مادة (1)\nنص.\n\nمادة (3)\nنص.\n\nمادة (2)\nنص.\n"
        for text in (_TWO_ARTICLES, _BIDI_WRAPPED, out_of_order):
            scan = _scan(text)
            assert set(scan.spans_by_pass) <= set(scan.fires), text

    def test_the_rollup_reports_documents_not_just_totals(self) -> None:
        """A pass firing 900 times in one document is not a pass that earns its
        place across a corpus."""
        summary = aggregate([_scan(_TWO_ARTICLES), _scan(_TWO_ARTICLES)])
        net, did_something, ran = summary["passes"]["regex"]
        assert (net, did_something, ran) == (4, 2, 2)

    def test_a_pass_offered_rarely_is_not_a_dead_pass(self) -> None:
        """Without the third number a conditional pass offered once and finding
        nothing is byte-identical to one offered everywhere and never firing,
        and only the second is grounds for deleting it."""
        summary = aggregate([_scan(_TWO_ARTICLES), _scan(_TWO_ARTICLES), _scan("نص بلا علامات.\n")])
        assert summary["passes"]["drop_toc_duplicates"] == [0, 0, 3]
        assert summary["passes"]["numbered_heading_fallback"] == [0, 0, 1]

    def test_the_uk_lane_does_not_credit_its_provisions_to_the_regex(self) -> None:
        """`raw` is extended in place on the UK path, so a count taken after it
        reports every UK-provision anchor and both UK drops as regex work."""
        text = "PART 1\nGeneral\n\n1 Interpretation\n(1) In this Act...\n"
        scan = scan_text(text, config=load_config("gb"), country="gb", doctype="act")
        assert scan.fires["regex"] == scan.by_pass["regex"] == 1
        assert scan.fires["scan_uk_provisions"] == scan.by_pass["uk_provisions"] == 2
