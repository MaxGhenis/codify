"""Classifying a stored law from its own metadata. The source text is gone for
the historic corpus, so the inputs are the title, the year and the AKN preamble."""

from __future__ import annotations

from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.stages import preamble_from_akn, reclassify_stored_law

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


# Called per test, not at import: a tree without the config must skip, and a
# module-level call raises during collection where no hook can convert it.
def xz_config() -> JurisdictionConfig:
    return load_config("xz")


# The enabling-article formula a decree-law recites. Priority 80, above the title rule.
_DECREE_PREAMBLE = "استناداً لأحكام النظام الأساسي ولا سيما المادة الحادية والستون منه"


def _akn(preamble: str = "") -> str:
    block = f"<preamble><p>{preamble}</p></preamble>" if preamble else ""
    return (
        f'<akomaNtoso xmlns="{NS}"><act><meta/>{block}'
        '<body><article eId="art_1"><num>1</num>'
        "<content><p>نص.</p></content></article></body></act></akomaNtoso>"
    )


class TestSignals:
    def test_an_old_law_classifies_from_its_title(self) -> None:
        doctype, signal = reclassify_stored_law(
            xz_config(), title="قانون ضريبة الحيوانات رقم 38 لسنة 1944", year=1944, akn_xml=_akn()
        )
        assert (doctype, signal) == ("act", "title_and_year")

    def test_a_decree_law_title_classifies_without_any_preamble(self) -> None:
        doctype, _ = reclassify_stored_law(
            xz_config(), title="قرار بقانون رقم 3 لسنة 2016", year=2016, akn_xml=_akn()
        )
        assert doctype == "decree_law"

    def test_the_preamble_overrides_a_title_that_reads_as_an_ordinary_law(self) -> None:
        """The rule that earns the preamble extractor: a decree-law
        titled `قانون…` reads as `act` on its title alone."""
        title = "قانون الشركات التجارية رقم (7) لسنة 2012م"
        bare, _ = reclassify_stored_law(xz_config(), title=title, year=2012, akn_xml=_akn())
        withp, signal = reclassify_stored_law(
            xz_config(), title=title, year=2012, akn_xml=_akn(_DECREE_PREAMBLE)
        )
        assert bare == "act"
        assert (withp, signal) == ("decree_law", "preamble")

    def test_the_signal_says_title_when_the_preamble_changed_nothing(self) -> None:
        """Otherwise every law with any preamble would look preamble-decided."""
        _, signal = reclassify_stored_law(
            xz_config(), title="قانون رقم 5 لسنة 1944", year=1944, akn_xml=_akn("نص تمهيدي عادي")
        )
        assert signal == "title_and_year"


class TestRefusals:
    def test_a_law_with_no_title_is_left_alone(self) -> None:
        """Returning the jurisdiction default here would write the same silent
        fallback this reclassification exists to remove."""
        assert reclassify_stored_law(xz_config(), title="", year=1944, akn_xml=_akn()) == (
            None,
            "no_title",
        )

    def test_an_unknown_jurisdiction_is_left_alone(self) -> None:
        """Its own signal: a run reporting 400 unclassifiable rows has to be
        able to say whether that is 400 missing configs or 400 untitled laws."""
        assert reclassify_stored_law(None, title="قانون", year=1944, akn_xml=_akn()) == (
            None,
            "no_config",
        )

    def test_a_jurisdiction_with_no_rules_classifies_nothing(self) -> None:
        """Its default is the value being replaced, so falling back to it would
        report a rewrite that decided nothing. `gb` declares no rules."""
        assert reclassify_stored_law(
            load_config("gb"), title="Widget Act 2015", year=2015, akn_xml=_akn()
        ) == (None, "no_rule_matched")

    def test_unparseable_akn_is_left_alone(self) -> None:
        """Otherwise it lands in `title_and_year` and reads as evidence the
        preamble rarely helps, when it means the preamble was never read."""
        assert reclassify_stored_law(
            xz_config(), title="قانون رقم 5 لسنة 1944", year=1944, akn_xml="<akomaNtoso><unclosed>"
        ) == (None, "akn_unparseable")


class TestPreambleExtraction:
    def test_unparseable_akn_is_none_not_empty(self) -> None:
        """Distinct from a document that simply has no preamble."""
        assert preamble_from_akn("<akomaNtoso><unclosed>") is None

    def test_an_absent_preamble_is_empty(self) -> None:
        assert preamble_from_akn(_akn()) == ""

    def test_the_preamble_is_capped(self) -> None:
        """`resolve_doctype` reads a prefix of the source; a whole statute would
        make every preamble rule scan the entire document."""
        assert len(preamble_from_akn(_akn("ن" * 5000))) == 2000
