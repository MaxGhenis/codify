"""Unit tests for codify.pipeline.enrich.eid, informal reference → canonical eId."""

from __future__ import annotations

import pytest

from codify.jurisdictions import JurisdictionConfigError
from codify.pipeline.enrich.eid import resolve_informal_reference

# --- resolve_informal_reference -------------------------------------------------


class TestResolveCommonLaw:
    """gb.act is common-law: section > subsection > paragraph > subparagraph."""

    def test_section_only(self):
        assert resolve_informal_reference("section 5", "gb") == "sec_5"

    def test_section_subsection(self):
        assert resolve_informal_reference("section 5(2)", "gb") == "sec_5__subsec_2"

    def test_section_subsection_paragraph(self):
        assert resolve_informal_reference("section 5(2)(a)", "gb") == "sec_5__subsec_2__para_a"

    def test_section_subsection_paragraph_subparagraph(self):
        assert (
            resolve_informal_reference("section 5(2)(a)(i)", "gb")
            == "sec_5__subsec_2__para_a__subpara_i"
        )

    def test_lowercase_alias_s(self):
        assert resolve_informal_reference("s. 5(2)", "gb") == "sec_5__subsec_2"

    def test_s_no_period(self):
        assert resolve_informal_reference("s 5", "gb") == "sec_5"

    def test_section_suffixed_number(self):
        # Inserted sections often use 5A, 5B, etc.
        assert resolve_informal_reference("section 5A", "gb") == "sec_5a"

    def test_whitespace_tolerance(self):
        assert (
            resolve_informal_reference("  section 5 ( 2 ) ( a ) ", "gb")
            == "sec_5__subsec_2__para_a"
        )

    def test_a_level_the_hierarchy_omits(self):
        # nz.act declares no `chapter`, so there is nothing to resolve it to.
        assert resolve_informal_reference("chapter 3", "nz") is None

    def test_part(self):
        assert resolve_informal_reference("Part 2", "gb") == "part_2"

    def test_too_many_subdivisions_trims_gracefully(self):
        # gb has 3 subdivision levels (subsec, para, subpara). 4+ should
        # still resolve up to level 3.
        result = resolve_informal_reference("section 5(2)(a)(i)(x)", "gb")
        assert result == "sec_5__subsec_2__para_a__subpara_i"


class TestResolveFrance:
    """fr.loi is civil-law: article is the basic unit."""

    def test_article(self):
        assert resolve_informal_reference("Article 5", "fr") == "art_5"

    def test_article_abbreviated(self):
        assert resolve_informal_reference("Art. 5", "fr") == "art_5"

    def test_article_with_parens(self):
        # Falls back to hierarchical synthetic: fr has Article > Alinéa
        # (in civil law alineas are numbered, not lettered, so (2) is
        # interpreted as the first subdivision level).
        result = resolve_informal_reference("Article 5(2)", "fr")
        # fr.loi's first subdivision in the hierarchy is alinea → al
        assert result == "art_5__al_2"


class TestResolveEdgeCases:
    def test_empty_string(self):
        assert resolve_informal_reference("", "gb") is None

    def test_gibberish(self):
        assert resolve_informal_reference("this is not a reference", "gb") is None

    def test_a_missing_jurisdiction_raises(self):
        """Distinct from the test below: an unknown unit in a known jurisdiction
        is a reference this jurisdiction does not use, and still answers None."""
        with pytest.raises(JurisdictionConfigError, match="zz"):
            resolve_informal_reference("section 5", "zz")

    def test_unknown_unit(self):
        # "canon" is not in gb.act hierarchy
        assert resolve_informal_reference("canon 5", "gb") is None


# --- Cross-script + inflected reference resolution ---------------------------


class TestCyrillicReferences:
    def test_ukrainian_article_nominative(self):
        assert resolve_informal_reference("Стаття 5", "ua", "act") == "art_5"

    def test_ukrainian_article_lowercase(self):
        assert resolve_informal_reference("стаття 5", "ua", "act") == "art_5"

    def test_ukrainian_article_genitive(self):
        # "статті" is the genitive, the form that appears in "of article 5".
        assert resolve_informal_reference("статті 7", "ua", "act") == "art_7"

    def test_ukrainian_chapter_resolves_to_chp(self):
        # Bluebell hard-codes chapter → chp; config must match.
        assert resolve_informal_reference("Глава 3", "ua", "act") == "chp_3"

    def test_ukrainian_subsection_inflected(self):
        # "частиною" instrumental, "by/with part N"
        assert resolve_informal_reference("частиною 2", "ua", "act") == "subsec_2"

    def test_ukrainian_nested_reference(self):
        # Civil-law nested form: article 5(2)(a)
        eid = resolve_informal_reference("Стаття 5(2)(a)", "ua", "act")
        assert eid == "art_5__subsec_2__para_a"


class TestArabicReferences:
    def test_arabic_article_bare(self):
        # Indefinite "مادة 5"
        assert resolve_informal_reference("مادة 5", "xz", "act") == "art_5"

    def test_arabic_article_definite(self):
        # Definite "المادة 5", the form that almost always appears in citations
        assert resolve_informal_reference("المادة 5", "xz", "act") == "art_5"

    def test_arabic_article_arabic_indic_digit(self):
        # Arabic-Indic numeral
        assert resolve_informal_reference("المادة ٥", "xz", "act") == "art_5"

    def test_arabic_article_multi_digit(self):
        assert resolve_informal_reference("المادة ١٢", "xz", "act") == "art_12"

    def test_arabic_chapter_resolves(self):
        # باب maps to part in this doctype
        assert resolve_informal_reference("الباب 1", "xz", "act") == "part_1"


class TestRegressionEnglish:
    """Make sure adding Cyrillic + Arabic didn't break existing Latin behaviour."""

    def test_section_5(self):
        assert resolve_informal_reference("section 5", "gb", "act") == "sec_5"

    def test_nested_sections(self):
        assert (
            resolve_informal_reference("section 5(2)(a)", "gb", "act") == "sec_5__subsec_2__para_a"
        )


class TestAbbreviationAuthority:
    """The element supplies the prefix; a config override wins."""

    def test_the_canonical_abbreviation_applies_without_a_config_override(self):
        # fr's config said chap, which no Bluebell output ever carried.
        assert resolve_informal_reference("chapter 3", "fr") == "chp_3"

    def test_a_uk_statutory_instrument_keeps_its_publishers_prefix(self):
        # SIs map "Regulation" onto AKN section; legislation.gov.uk numbers
        # them reg_N.
        assert resolve_informal_reference("regulation 4", "gb", "si") == "reg_4"
        assert resolve_informal_reference("section 4", "gb", "act") == "sec_4"


def test_no_document_class_gives_two_units_the_same_prefix() -> None:
    """Two local terms resolving to one prefix make their references
    indistinguishable, and the second silently wins the reverse lookup."""
    import json

    from codify.akn.eid import eid_abbrev
    from codify.jurisdictions import JURISDICTIONS_DIR

    # Known collisions inherited from before the canonical table; this test
    # exists to stop the list growing.
    allowed = {
        "by/act",
        "by/code",
        "ch/bundesgesetz",
        "ci/loi",
        "co/act",
        "co/decreto",
        "es/constitucion",
        "it/costituzione",
        "ml/act",
        "mu/civil_code",
        "ne/loi",
        "pl/act",
        "th/act",
        "th/constitution",
        "us/bill",
        "us/cfr_title",
        "us/public_law",
        "us/usc_title",
        "va/canon_law_codex",
    }
    found = set()
    root = JURISDICTIONS_DIR
    configs = sorted(root.glob("*/config.json"))
    assert configs, "packaged jurisdiction configs must be present"
    for cfg in configs:
        doc = json.loads(cfg.read_text())
        for name, dc in (doc.get("document_classes") or {}).items():
            prefixes: dict[str, int] = {}
            for entry in dc.get("hierarchy") or []:
                element = entry.get("akn_element")
                if not element:
                    continue
                key = entry.get("eid_abbrev") or eid_abbrev(element)
                prefixes[key] = prefixes.get(key, 0) + 1
            if any(count > 1 for count in prefixes.values()):
                found.add(f"{cfg.parent.name}/{name}")
    assert found <= allowed, sorted(found - allowed)


def test_every_resolvable_prefix_is_one_the_scanner_can_emit() -> None:
    """The bug this guards: for months the scanner minted `list_a` while configs
    resolved references to `point_a`, and nothing failed. A reference resolving
    to a prefix no document carries is dropped silently."""
    import json

    from codify.jurisdictions import JURISDICTIONS_DIR
    from codify.pipeline.enrich.anchors import _abbrev_map_for

    root = JURISDICTIONS_DIR
    divergent = []
    configs = sorted(root.glob("*/config.json"))
    assert configs, "packaged jurisdiction configs must be present"
    for cfg in configs:
        code = cfg.parent.name
        for doctype, dc in (json.loads(cfg.read_text()).get("document_classes") or {}).items():
            emits = _abbrev_map_for(code, doctype)
            for entry in dc.get("hierarchy") or []:
                element = entry.get("akn_element")
                # Only unambiguous elements: where two local terms share one
                # element the scanner cannot tell them apart by design.
                if not element or element not in emits:
                    continue
                resolves_to = entry.get("eid_abbrev") or emits[element]
                if resolves_to != emits[element]:
                    divergent.append(
                        f"{code}/{doctype}: {element} {resolves_to} != {emits[element]}"
                    )
    assert not divergent, divergent
