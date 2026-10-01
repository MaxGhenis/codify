"""The long title reaching `<longTitle>`, and the short name resolving."""

from __future__ import annotations

import pytest

from codify.pipeline.enrich import titles
from codify.pipeline.enrich.anchors import StructuralAnchor
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.scaffold import scaffold_from_anchors
from codify.pipeline.enrich.titles import (
    declared_short_title,
    derived_short_title,
    markup_short_title,
    resolve_short_title,
)
from codify.storage.repository import _extract_long_title

# The real shape: number line, then the long title, then the enacting formula.
PREFACE = """ATLANTEAN ACT No. 11306

An Act Converting the Poseidonis National High School into an Independent National
High School and Appropriating Funds Therefor"""


def _section(number: str) -> StructuralAnchor:
    return StructuralAnchor(
        kind="section",
        keyword="SECTION",
        number=number,
        char_offset=0,
        line=1,
        matched_text=f"SECTION {number}",
        depth=0,
        akn_eid=f"sec_{number}",
    )


def _scaffold(country: str, preface: str = PREFACE) -> str:
    scaffold, _ = scaffold_from_anchors([_section("1")], preface=preface, country=country)
    return scaffold


class TestLongTitle:
    def test_declared_lead_in_is_marked_up(self) -> None:
        assert "LONGTITLE An Act Converting" in _scaffold("xa")

    def test_a_jurisdiction_declaring_none_is_untouched(self) -> None:
        """`xy` declares no lead-in, so its scaffold must be byte-identical."""
        assert "LONGTITLE" not in _scaffold("xy")
        assert _scaffold("xy") == _scaffold("")

    def test_the_number_line_is_not_the_long_title(self) -> None:
        scaffold = _scaffold("xa")
        assert "LONGTITLE ATLANTEAN ACT" not in scaffold

    def test_only_the_first_match_is_marked(self) -> None:
        """A second `An Act ...` line cites a different statute."""
        preface = PREFACE + "\n\nAn Act Amending Atlantean Act No. 9999"
        assert _scaffold("xa", preface).count("LONGTITLE") == 1

    def test_the_lead_in_need_not_open_its_paragraph(self) -> None:
        """Whether a blank line precedes the title is a source line-break
        artefact; requiring one made the feature a silent no-op."""
        preface = "ATLANTEAN ACT No. 11306\nAn Act Converting the School."
        assert "LONGTITLE An Act Converting the School." in _scaffold("xa", preface)

    def test_furniture_after_the_title_is_not_swallowed(self) -> None:
        preface = "An Act Converting the School.\nApproved: June 2019"
        emitted = _scaffold("xa", preface)
        assert "LONGTITLE An Act Converting the School." in emitted
        assert "Approved: June 2019" in emitted
        assert "LONGTITLE An Act Converting the School. Approved" not in emitted

    def test_it_survives_the_round_trip_to_akn(self) -> None:
        """The whole point: the title reaches the column via existing machinery."""
        akn = parse_to_akn(_scaffold("xa"), "xa", doctype="act", date="2019", number="11306")
        assert _extract_long_title(akn) == (
            "An Act Converting the Poseidonis National High School into an Independent "
            "National High School and Appropriating Funds Therefor"
        )

    def test_no_long_title_element_without_the_declaration(self) -> None:
        akn = parse_to_akn(_scaffold("xy"), "xy", doctype="act", date="2019", number="1")
        assert _extract_long_title(akn) is None


class TestDeclaredShortTitle:
    def test_the_law_names_itself(self) -> None:
        text = 'This Act shall be known as the "Poseidonis Education Act".'
        assert declared_short_title(text, "xa") == "Poseidonis Education Act"

    def test_may_be_cited_as(self) -> None:
        assert (
            declared_short_title("This Act may be cited as the Police Act.", "xa") == "Police Act"
        )

    def test_a_stray_closing_glyph_is_dropped(self) -> None:
        """Sources spell the closing quote several ways; one reached the corpus."""
        text = 'This Act shall be known as the "Coral Day Act of 2012″.'
        assert declared_short_title(text, "xa") == "Coral Day Act of 2012"

    def test_a_full_stop_inside_the_quotes_is_dropped(self) -> None:
        """A seventh of the clauses in one corpus close the sentence inside the quote."""
        text = 'This Act shall be known as the "Ward Care Act of 2012."'
        assert declared_short_title(text, "xa") == "Ward Care Act of 2012"

    def test_a_letter_stuck_to_the_year_is_dropped(self) -> None:
        text = 'This Act shall be known as the "State Enterprise Governance Act of 2011l".'
        assert declared_short_title(text, "xa") == "State Enterprise Governance Act of 2011"

    def test_an_apostrophe_does_not_truncate_an_unquoted_name(self) -> None:
        """Scanning a set of closers cut this to "People"."""
        text = "This Act shall be known as the Citizens' Small Business Act. SEC. 3 Something."
        assert declared_short_title(text, "xa") == "Citizens' Small Business Act"

    def test_a_quoted_name_still_ends_at_its_own_closer(self) -> None:
        text = (
            'This Act shall be known as the "ATLANTRADE." The Corporation shall exist twenty years.'
        )
        assert declared_short_title(text, "xa") == "ATLANTRADE"

    def test_an_over_long_name_is_rejected_not_truncated(self) -> None:
        """The capture window used to equal the length cap, so an over-long name
        came back cut mid-word and was written as if it were the name."""
        text = (
            "This Act shall be known as the Act Providing for the Modernization, "
            "Standardization and Regulation of the Procurement Activities of the "
            "Government of Atlantis and for Other Purposes Whatsoever Herein."
        )
        assert declared_short_title(text, "xa") is None

    def test_an_internal_identifier_keeps_its_letter(self) -> None:
        """The stray-character cleanup is for a terminal OCR artefact; an
        internal one is a legal identifier."""
        text = 'This Act shall be known as the "Section 123A Amendment Act".'
        assert declared_short_title(text, "xa") == "Section 123A Amendment Act"

    def test_a_hijri_year_sheds_its_stray_letter(self) -> None:
        text = 'This Act shall be known as the "Deepwater Act of 1443h".'
        assert declared_short_title(text, "xa") == "Deepwater Act of 1443"

    def test_the_code_names_itself_too(self) -> None:
        assert declared_short_title('This Code shall be known as the "Civil Code".', "xa") == (
            "Civil Code"
        )

    def test_a_school_the_act_creates_is_not_the_act(self) -> None:
        """The same phrasing names what a law creates; a sixth of one corpus's
        stored short names were a school, campus or office before it was anchored."""
        text = "The school shall be known as the Poseidonis Memorial National High School."
        assert declared_short_title(text, "xa") is None

    def test_a_campus_introduced_by_it_is_not_the_act(self) -> None:
        text = "...in the Province of Meropis. It shall be known as the Meropis-Cirene Campus."
        assert declared_short_title(text, "xa") is None

    def test_an_office_the_act_creates_is_not_the_act(self) -> None:
        text = "The head of this institution shall be known as the President of the Institute."
        assert declared_short_title(text, "xa") is None

    def test_lead_ins_without_subjects_resolve_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anchored or not at all: a fallback would leave the defect reachable
        by omitting one config key."""
        config = titles.load_config("xa")
        assert config.structuring is not None
        monkeypatch.setattr(config.structuring, "citation_subjects", [])
        assert declared_short_title('This Act shall be known as the "Police Act".', "xa") is None

    def test_an_act_that_names_nothing_gets_nothing(self) -> None:
        assert declared_short_title("Section 1. There is hereby created a school.", "xa") is None

    def test_a_jurisdiction_declaring_no_lead_in_never_matches(self) -> None:
        assert declared_short_title("This Act may be cited as the Police Act.", "xy") is None


class TestDerivedShortTitle:
    def test_modern_titles_carry_the_number_after_the_form(self) -> None:
        title = (
            "COMMISSION DELEGATED DIRECTIVE 2014/7/HU of 18 October 2013 amending, for "
            "the purposes of adapting to technical progress, Annex IV to Directive 2011/65/HU"
        )
        assert derived_short_title(title, "xu") == "COMMISSION DELEGATED DIRECTIVE 2014/7/HU"

    def test_older_titles_carry_it_in_a_trailing_parenthetical(self) -> None:
        title = (
            "COUNCIL DIRECTIVE of 25 July 1977 on the education of the children of "
            "migrant workers (77/486/HUC)"
        )
        assert derived_short_title(title, "xu") == "COUNCIL DIRECTIVE 77/486/HUC"

    def test_an_ordinal_prefix_is_kept(self) -> None:
        title = "FIRST COMMISSION DIRECTIVE of 13 November 1979 laying down methods (79/1067/HUC)"
        assert derived_short_title(title, "xu") == "FIRST COMMISSION DIRECTIVE 79/1067/HUC"

    def test_a_jurisdiction_without_the_rule_derives_nothing(self) -> None:
        assert derived_short_title("COUNCIL DIRECTIVE 93/13/HUC of 5 April 1993", "xa") is None


class TestResolutionOrder:
    def test_a_declared_name_beats_a_derived_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No jurisdiction declares both today, so give one both and check."""
        monkeypatch.setattr(titles, "_designation_rule", lambda country: "eu_instrument")
        assert (
            resolve_short_title(
                title="COUNCIL DIRECTIVE 93/13/HUC of 5 April 1993",
                body_text='This Act may be cited as the "Unfair Terms Act".',
                country="xa",
            )
            == "Unfair Terms Act"
        )

    def test_the_designation_is_used_when_nothing_is_declared(self) -> None:
        assert (
            resolve_short_title(
                title="COUNCIL DIRECTIVE 93/13/HUC of 5 April 1993",
                body_text="Article 1. This Directive applies to unfair terms.",
                country="xu",
            )
            == "COUNCIL DIRECTIVE 93/13/HUC"
        )

    def test_neither_source_leaves_it_null(self) -> None:
        assert resolve_short_title(title="An Act", body_text="Nothing here.", country="xa") is None


class TestASourceThatStatesItsOwnName:
    """AKN title extraction and refusal rules."""

    def test_the_markup_name_is_read(self) -> None:
        doc = "<akomaNtoso><preface><docTitle>Finance Act 1958</docTitle></preface></akomaNtoso>"
        assert markup_short_title(doc) == "Finance Act 1958"

    def test_a_namespace_prefix_does_not_hide_it(self) -> None:
        doc = (
            '<akn:akomaNtoso xmlns:akn="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
            "<akn:preface><akn:docTitle>Northern Ireland Act 1962</akn:docTitle>"
            "</akn:preface></akn:akomaNtoso>"
        )
        assert markup_short_title(doc) == "Northern Ireland Act 1962"

    def test_a_real_title_carrying_markup(self) -> None:
        from pathlib import Path

        sample = Path(__file__).resolve().parents[4] / "data" / "samples" / "akn"
        sample = sample / "uk_uksi_1990_1304.akn.xml"
        if not sample.exists():
            pytest.skip("sample missing")
        name = markup_short_title(sample.read_text())

        assert name == "The European Communities (Designation) (No. 2) Order 1990"
        assert "<" not in (name or "")

    def test_unparseable_markup_is_none(self) -> None:
        assert markup_short_title("docTitle but not xml") is None

    def test_nothing_stated_is_none(self) -> None:
        assert markup_short_title("<akomaNtoso><p>An Act to do things.</p></akomaNtoso>") is None
        assert markup_short_title("") is None

    def test_a_doctype_entity_cannot_name_the_law(self) -> None:
        doc = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE akomaNtoso [<!ENTITY x "Injected Name Act 1999">]>'
            "<akomaNtoso><preface><docTitle>&x;</docTitle></preface></akomaNtoso>"
        )
        assert markup_short_title(doc) is None

    def test_a_status_marker_is_not_part_of_the_name(self) -> None:
        for marker in ("(revoked)", "(repealed)", "(revoked 1.1.1999)", "(repealed 31.3.2004)"):
            doc = f"<akomaNtoso><docTitle>Finance Act 1958 {marker}</docTitle></akomaNtoso>"
            assert markup_short_title(doc) == "Finance Act 1958", marker

    def test_a_parenthetical_belonging_to_the_title_is_kept(self) -> None:
        doc = "<akomaNtoso><docTitle>Senedd and Elections (Wales) Act 2020</docTitle></akomaNtoso>"
        assert markup_short_title(doc) == "Senedd and Elections (Wales) Act 2020"

    def test_a_work_uri_is_not_a_name(self) -> None:
        for uri in ("ukpga/1958/56", "nisi/2004/3078"):
            doc = f"<akomaNtoso><docTitle>{uri}</docTitle></akomaNtoso>"
            assert markup_short_title(doc) is None

    def test_a_number_in_a_real_title_is_not_a_uri(self) -> None:
        doc = "<akomaNtoso><docTitle>Regulation (EU) 2019/1020</docTitle></akomaNtoso>"
        assert markup_short_title(doc) == "Regulation (EU) 2019/1020"

    def test_an_annex_does_not_name_the_act(self) -> None:
        doc = (
            "<akomaNtoso><act>"
            "<attachment><doc><preface><docTitle>Schedule 1</docTitle></preface></doc></attachment>"
            "<preface><docTitle>Finance Act 1958</docTitle></preface>"
            "</act></akomaNtoso>"
        )
        assert markup_short_title(doc) == "Finance Act 1958"

    def test_a_title_stated_only_by_an_annex_names_nothing(self) -> None:
        doc = (
            "<akomaNtoso><act><attachment><doc><preface>"
            "<docTitle>Schedule 1</docTitle>"
            "</preface></doc></attachment></act></akomaNtoso>"
        )
        assert markup_short_title(doc) is None

    def test_a_declared_name_still_wins(self) -> None:
        assert (
            resolve_short_title(
                title="Directive (EU) 2016/943",
                body_text="nothing declared here",
                country="xu",
                akn_xml="<akomaNtoso><docTitle>Something Else</docTitle></akomaNtoso>",
            )
            == "Directive (EU) 2016/943"
        )

    def test_a_jurisdiction_with_no_rule_gets_the_markup(self) -> None:
        assert (
            resolve_short_title(
                title="An Act to consolidate certain enactments",
                body_text="body",
                country="xa",
                akn_xml="<akomaNtoso><docTitle>Finance Act 1958</docTitle></akomaNtoso>",
            )
            == "Finance Act 1958"
        )
