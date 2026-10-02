"""Document class from a jurisdiction's classification rules.

The pipeline once ignored `structuring.classification_rules` and took
`default_document_class` instead. Doctype feeds the FRBR work URI, so a
defaulted class puts a law at the wrong address.
"""

from __future__ import annotations

from datetime import date

import pytest

from codify import jurisdictions as jurisdictions_module
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline import stages as stages_module
from codify.pipeline.stages import resolve_descriptors, resolve_doctype

# Classification rules inline, so these tests need no corpus config.
_CLASSIFIER = {
    "code": "xx",
    "name": "Fixture",
    "languages": ["ara"],
    "tradition": ["civil_law"],
    "authoritative_language": "ara",
    "default_document_class": "qarar_bi_qanun",
    "structuring": {
        "classification_rules": [
            {
                "signal": "title_regex",
                "pattern": "^القانون الأساسي",
                "target_document_class": "basic_law",
                "priority": 100,
            },
            {
                "signal": "title_regex",
                "pattern": "^Basic Law",
                "pattern_flags": "i",
                "target_document_class": "basic_law",
                "priority": 100,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?قرار بقانون",
                "target_document_class": "qarar_bi_qanun",
                "priority": 90,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?مرسوم بقانون",
                "target_document_class": "qarar_bi_qanun",
                "priority": 89,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?قرار\\s+(?:رئيس\\s+(?:مجلس\\s+)?الوزراء|مجلس\\s+الوزراء)",
                "target_document_class": "qarar_majlis_wuzara",
                "priority": 88,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?قرار\\s+(?:ال)?(?:وزير|وزاري)",
                "target_document_class": "qarar_wazir",
                "priority": 86,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?مرسوم",
                "target_document_class": "marsoum",
                "priority": 85,
            },
            {
                "signal": "title_regex",
                "pattern": (
                    "^(?:ال)?قرار\\s+(?:ال)?(?:رئيس|رئاسي)(?!\\s+(?:مجلس|الوزراء|(?:ال)?سلطة(?!\\s"
                    "+الاتحادية)|(?:ال)?ه"
                    "يئة|(?:ال)?بلدية|(?:ال)?لجنة|(?:ال)?ديوان|(?:ال)?دائرة|(?:ال)?محكمة|(?:ال)?مؤ"
                    "سسة|(?:ال)?جامعة))"
                ),
                "target_document_class": "qarar_rais",
                "priority": 84,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?نظام",
                "target_document_class": "nizam",
                "priority": 82,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?لائحة",
                "target_document_class": "laihat",
                "priority": 81,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?تعليمات",
                "target_document_class": "taalimat",
                "priority": 79,
            },
            {
                "signal": "title_regex",
                "pattern": "^أمر",
                "target_document_class": "amr",
                "priority": 72,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?قرار",
                "target_document_class": "qarar",
                "priority": 71,
            },
            {
                "signal": "preamble_match",
                "pattern": "المادة الحادية والستون",
                "target_document_class": "qarar_bi_qanun",
                "priority": 80,
            },
            {
                "signal": "preamble_match",
                "pattern": "مرسوم بقانون",
                "target_document_class": "qarar_bi_qanun",
                "priority": 70,
            },
            {
                "signal": "date_range",
                "date_from": "1961-07-15",
                "target_document_class": "qarar_bi_qanun",
                "priority": 50,
            },
            {
                "signal": "title_regex",
                "pattern": "^(?:ال)?قانون(?!\\s+الأساسي)",
                "target_document_class": "qanun",
                "priority": 65,
            },
            {
                "signal": "date_range",
                "date_until": "1961-07-14",
                "target_document_class": "qanun",
                "priority": 40,
            },
        ]
    },
    "document_classes": {
        "basic_law": {
            "label": "Basic Law (القانون الأساسي)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qanun": {
            "label": "Law (قانون)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qarar_bi_qanun": {
            "label": "Decree-Law (قرار بقانون)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "marsoum": {
            "label": "Presidential Decree (مرسوم رئاسي)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "laihat": {
            "label": "By-law (لائحة)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qarar_majlis_wuzara": {
            "label": "Cabinet Decision (قرار مجلس الوزراء)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qarar_wazir": {
            "label": "Ministerial Decision (قرار وزير)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qarar_rais": {
            "label": "Presidential Decision (قرار رئيس)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "qarar": {
            "label": "Decision (قرار)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "nizam": {
            "label": "Regulation (نظام)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "taalimat": {
            "label": "Instructions (تعليمات)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
        "amr": {
            "label": "Ordinance (أمر)",
            "akn_element": "act",
            "bluebell_compatible": True,
            "basic_unit": "article",
        },
    },
}
_CONFIG = JurisdictionConfig.model_validate(_CLASSIFIER)


@pytest.fixture(autouse=True)
def _fixture_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # resolve_descriptors takes a code and reloads the config by it.
    real = jurisdictions_module.load_config
    stub = lambda c: _CONFIG if c == "xx" else real(c)  # noqa: E731
    monkeypatch.setattr(jurisdictions_module, "load_config", stub)
    monkeypatch.setattr(stages_module, "load_config", stub, raising=False)


def _cfg() -> JurisdictionConfig:
    return _CONFIG


def test_a_title_rule_wins_over_the_default() -> None:
    assert (
        _cfg().classify_document_class(title="قرار بقانون رقم (18) لسنة 2016") == "qarar_bi_qanun"
    )


def test_no_match_returns_none_so_the_caller_defaults() -> None:
    """None is "no rule matched", not a class. An unmatched document is
    unclassified rather than misclassified."""
    assert _cfg().classify_document_class(title="Untitled") is None


def test_the_pipeline_falls_back_to_the_jurisdiction_default() -> None:
    descriptors = resolve_descriptors(
        {"title": "Untitled", "number": "1"},
        jurisdiction_code="xx",
        source_bytes=b"",
        fallback_stem="x",
    )
    assert descriptors.doctype == _cfg().default_document_class


def test_a_jurisdiction_with_no_rules_always_defaults() -> None:
    descriptors = resolve_descriptors(
        {"title": "Some Act", "number": "1"},
        jurisdiction_code="gb",
        source_bytes=b"",
        fallback_stem="x",
    )
    cfg = load_config("gb")
    assert cfg is not None
    assert descriptors.doctype == cfg.default_document_class


def test_an_unknown_jurisdiction_falls_back_to_act() -> None:
    assert resolve_doctype(None, title="x", raw_date="", year="2000", source="") == "act"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("قانون رقم (9) لسنة 1987 بشأن البيئة", "qanun"),
        ("قانون التجارة رقم (2) لسنة 1974م", "qanun"),
        ("القانون المدني رقم (4) لسنة 1968", "qanun"),
        ("القانون الأساسي المعدل لسنة 1962", "basic_law"),
        ("قرار بقانون رقم (18) لسنة 2016", "qarar_bi_qanun"),
    ],
)
def test_arabic_titles_classify_to_their_own_class(title: str, expected: str) -> None:
    """A qanun names its subject before the number, and the Civil Code opens
    القانون المدني, so a rule anchored on "قانون رقم" matched neither."""
    assert _cfg().classify_document_class(title=title) == expected


def test_the_basic_law_is_not_swept_up_by_the_general_qanun_rule() -> None:
    """It carries its own class and its own URI pattern."""
    assert _cfg().classify_document_class(title="القانون الأساسي لسنة 1962") == "basic_law"


_HAMZA_ABOVE = "ٔ"  # combining mark; NFC composes ا+◌ٔ → أ, ي+◌ٔ → ئ


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # الأساسي written decomposed (bare alef + combining hamza), how OCR
        # stored the Basic Law, which the precomposed rule pattern missed.
        (f"القانون الا{_HAMZA_ABOVE}ساسي المعدل لسنة 1962", "basic_law"),
        # رئيس decomposed (ي + combining hamza); the prime-minister form must
        # still reach the cabinet rule, not fall through to generic qarar.
        (f"قرار ري{_HAMZA_ABOVE}يس مجلس الوزراء رقم 11 لسنة 2008", "qarar_majlis_wuzara"),
        (f"قرار ري{_HAMZA_ABOVE}يس دولة زرزورة رقم 7 لسنة 2018", "qarar_rais"),
    ],
)
def test_decomposed_arabic_hamza_is_matched_via_nfc(title: str, expected: str) -> None:
    """Arabic corpora commonly store hamza decomposed. Rules match on NFC so the
    byte form of the title does not decide its class."""
    assert _HAMZA_ABOVE in title  # guard: the fixture really is decomposed
    assert _cfg().classify_document_class(title=title) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("قرار مجلس الوزراء رقم (5) لسنة 2015", "qarar_majlis_wuzara"),
        ("قرار رئيس الوزراء رقم 4 لسنة 2010", "qarar_majlis_wuzara"),
        ("قرار وزير العدل رقم (3) لسنة 2019", "qarar_wazir"),
        ("قرار وزاري رقم 2 لسنة 2020", "qarar_wazir"),
        ("القرار الوزاري رقم 2 لسنة 2020", "qarar_wazir"),
        ("قرار رئيس دولة زرزورة رقم (7) لسنة 2018", "qarar_rais"),
        ("قرار رئاسي رقم 9 لسنة 2017", "qarar_rais"),
        ("قرار رقم (259) لسنة 2009", "qarar"),
        ("نظام رقم (4) لسنة 2011", "nizam"),
        ("النظام المالي رقم 1 لسنة 2005", "nizam"),
        ("لائحة تنفيذية رقم 2 لسنة 2012", "laihat"),
        ("تعليمات رقم (1) لسنة 2016", "taalimat"),
        ("مرسوم رقم (9) لسنة 2014", "marsoum"),
        ("أمر بشأن الأراضي لسنة 1936", "amr"),
    ],
)
def test_subordinate_instruments_get_their_own_class(title: str, expected: str) -> None:
    """The bulk ingest collapsed nine instrument types into three; each
    subordinate form now carries its own class and FRBR subtype segment."""
    assert _cfg().classify_document_class(title=title) == expected


def test_a_cabinet_decision_by_the_prime_minister_is_not_read_as_presidential() -> None:
    """رئيس مجلس الوزراء is the head of cabinet, not the President; the cabinet
    rule ranks above the presidential rule and its lookahead excludes the PM."""
    assert (
        _cfg().classify_document_class(title="قرار رئيس مجلس الوزراء رقم 11 لسنة 2008")
        == "qarar_majlis_wuzara"
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # رئيس of a named authority/body is that body's own decision, not the head
        # of state; it falls through the presidential rule to the generic decision.
        ("قرار رئيس سلطة الواحات رقم (33) لسنة 2013", "qarar"),
        ("قرار رئيس سلطة الموانئ رقم (55) لسنة 2010", "qarar"),
        ("قرار رئيس هيئة القوافل رقم 2 لسنة 2014", "qarar"),
        ("قرار رئيس لجنة التنظيم رقم 8 لسنة 2016", "qarar"),
        # definite ال- spellings of the body name must be excluded too.
        ("قرار رئيس المحكمة العليا رقم 3 لسنة 2015", "qarar"),
        ("قرار رئيس البلدية رقم 2 لسنة 2018", "qarar"),
        # genuine head-of-state forms keep matching the presidential rule.
        ("قرار رئيس دولة زرزورة رقم (7) لسنة 2018", "qarar_rais"),
        ("قرار رئيس السلطة الاتحادية الزرزورية رقم 3 لسنة 2015", "qarar_rais"),
        ("قرار الرئيس رقم 5 لسنة 2016", "qarar_rais"),
        ("قرار رئاسي رقم 9 لسنة 2017", "qarar_rais"),
    ],
)
def test_head_of_an_authority_is_not_read_as_presidential(title: str, expected: str) -> None:
    """رئيس سلطة/هيئة/لجنة X is the head of a sub-authority; only رئيس دولة /
    رئيس السلطة الاتحادية and the adjective قرار رئاسي are the President."""
    assert _cfg().classify_document_class(title=title) == expected


def test_a_decree_by_law_outranks_the_plain_decree_rule() -> None:
    """مرسوم بقانون carries legislative force and must not be read as an
    executive decree by the plain ^مرسوم rule."""
    assert _cfg().classify_document_class(title="مرسوم بقانون رقم 3 لسنة 2016") == "qarar_bi_qanun"


def test_a_decision_that_cites_its_parent_decree_law_stays_a_decision() -> None:
    """A subordinate decision routinely cites its enabling decree-law in the
    preamble. The document's own title must decide over that parent citation,
    so the generic decision rule ranks above the مرسوم بقانون preamble rule."""
    assert (
        _cfg().classify_document_class(
            title="قرار رقم (259) لسنة 2009",
            preamble="صدر بموجب مرسوم بقانون رقم 5 لسنة 2008",
        )
        == "qarar"
    )


@pytest.mark.parametrize(
    "title",
    [
        "لائحة تنفيذية لقرار بقانون رقم 3 لسنة 2016",
        "لائحة تنفيذية لقرار مجلس الوزراء رقم 5 لسنة 2015",
    ],
)
def test_a_title_that_only_references_another_instrument_is_not_read_as_one(title: str) -> None:
    """Every instrument title rule is anchored, so an executive by-law whose
    title cites a parent decree-law or cabinet decision is not swept into that
    parent's class; its own leading لائحة decides."""
    assert _cfg().classify_document_class(title=title) == "laihat"


def test_the_definite_article_form_of_marsoum_still_classifies() -> None:
    """المرسوم carries the definite article; the rule matches it like its
    siblings (نظام, لائحة)."""
    assert _cfg().classify_document_class(title="المرسوم رقم (9) لسنة 2014") == "marsoum"


def test_a_preamble_rule_matches_when_the_title_does_not() -> None:
    assert (
        _cfg().classify_document_class(title="Untitled", preamble="صدر بموجب مرسوم بقانون")
        == "qarar_bi_qanun"
    )


def test_a_date_rule_places_a_document_either_side_of_the_cutover() -> None:
    """A dated rule splits the classes either side of its bound."""
    assert _cfg().classify_document_class(title="x", doc_date=date(1961, 7, 16)) == "qarar_bi_qanun"
    assert _cfg().classify_document_class(title="x", doc_date=date(1961, 7, 14)) == "qanun"


def test_a_date_rule_is_skipped_when_no_date_is_known() -> None:
    """A rule bounded by dates cannot match a document with no date; treating
    it as satisfied would classify by regime on no evidence."""
    assert _cfg().classify_document_class(title="Untitled", doc_date=None) is None


def test_a_bare_year_does_not_decide_a_bound_that_falls_inside_it() -> None:
    """The fixture's cutover is mid-year, so a document known only by year could
    sit either side; 1 January would classify all of it by the earlier rule."""
    for raw_date, year, expected in (
        ("", "1961", _cfg().default_document_class),
        ("", "1950", "qanun"),
        ("", "2016", "qarar_bi_qanun"),
    ):
        assert resolve_doctype(_cfg(), title="x", raw_date=raw_date, year=year, source="") == (
            expected
        ), year


def test_priority_order_decides_when_two_rules_match() -> None:
    """The decree-law title rule sits above the date rules, so a pre-cutover
    decree-law is not reclassified by its date."""
    assert (
        _cfg().classify_document_class(
            title="قرار بقانون رقم 8 لسنة 1955", doc_date=date(1955, 1, 1)
        )
        == "qarar_bi_qanun"
    )


def test_an_uncompilable_pattern_is_skipped_not_treated_as_a_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule nobody can compile must not classify every document that reaches
    it, and must not take the ingest down either."""
    cfg = _cfg().model_copy(deep=True)
    assert cfg.structuring is not None
    broken = cfg.structuring.classification_rules[0].model_copy(
        update={"pattern": "([unclosed", "priority": 999, "signal": "title_regex"}
    )
    cfg.structuring.classification_rules.insert(0, broken)
    assert cfg.classify_document_class(title="قرار بقانون رقم (18) لسنة 2016") == "qarar_bi_qanun"


def test_the_source_preamble_reaches_the_rules_through_the_pipeline() -> None:
    descriptors = resolve_descriptors(
        {"title": "Untitled", "number": "1"},
        jurisdiction_code="xx",
        source_bytes="صدر بموجب مرسوم بقانون رقم كذا".encode(),
        fallback_stem="x",
    )
    assert descriptors.doctype == "qarar_bi_qanun"


def test_office_routing_code_never_reaches_the_frbr_path() -> None:
    """The seam that produced the defect.

    A model asked for a law number returned the whole parenthetical it sat in,
    `( /11/11 ر.م.و/ إ.هـ)`, and `resolve_descriptors` passed it through to the
    FRBR path verbatim. This fails if the guard is removed from that seam, which
    is the point: the unit test on `law_number_token` alone cannot see whether
    anyone calls it."""
    from codify.frbr import build_frbr_work_uri
    from codify.pipeline.stages import resolve_descriptors

    desc = resolve_descriptors(
        {"title": "قرار رئيس مجلس الوزراء", "number": "11/11 ر.م.و/ إ.هـ", "date": "2008-04-13"},
        jurisdiction_code="xx",
        source_bytes=b"body text",
        fallback_stem="stem",
    )
    assert desc.number.isascii()
    assert desc.number.startswith("draft-")  # fell through to the content address
    assert build_frbr_work_uri("xx", desc.doctype, desc.year, desc.number).isascii()


def test_a_usable_number_is_kept_verbatim() -> None:
    # The guard must not eat real numbers on its way to rejecting bad ones.
    from codify.pipeline.stages import resolve_descriptors

    desc = resolve_descriptors(
        {"title": "قرار", "number": "259", "date": "2009-01-01"},
        jurisdiction_code="xx",
        source_bytes=b"body text",
        fallback_stem="stem",
    )
    assert desc.number == "259"


def test_the_caller_selects_the_document_class() -> None:
    """The structurer used to override it, which parsed a civil-law code with
    the act hierarchy and collided its chapter numbers."""
    desc = resolve_descriptors(
        {"title": "Some Act", "number": "1"},
        jurisdiction_code="xa",
        source_bytes=b"",
        fallback_stem="x",
        requested_doctype="code",
    )
    assert desc.doctype == "code"


def test_an_undeclared_class_is_refused_not_ignored() -> None:
    with pytest.raises(ValueError, match="declares no document class"):
        resolve_descriptors(
            {"title": "Some Act", "number": "1"},
            jurisdiction_code="xa",
            source_bytes=b"",
            fallback_stem="x",
            requested_doctype="statute",
        )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Civil Code of Atlantis", "code"),
        ("The Consolidated Penal Code", "code"),
        ("The 1987 Constitution of the Commonwealth of Atlantis", "constitution"),
        ("Harbour Dues Act of 2018", "act"),
    ],
)
def test_codes_classify_without_the_caller(title: str, expected: str) -> None:
    desc = resolve_descriptors(
        {"title": title, "number": "1"},
        jurisdiction_code="xa",
        source_bytes=b"",
        fallback_stem="x",
    )
    assert desc.doctype == expected


def test_an_omitted_doctype_leaves_the_rules_to_classify() -> None:
    """The ingest APIs used to default the field to "act", which reached
    `requested` as a real value and stopped every rule from running."""
    desc = resolve_descriptors(
        {"title": "Civil Code of Atlantis", "number": "386"},
        jurisdiction_code="xa",
        source_bytes=b"",
        fallback_stem="x",
        requested_doctype=None,
    )
    assert desc.doctype == "code"


@pytest.mark.parametrize(
    "title",
    [
        "An Act Amending the Civil Code of Atlantis",
        "An Act Further Amending the Consolidated Penal Code",
        "An Act Amending Article 315 of the Consolidated Penal Code",
    ],
)
def test_an_act_that_amends_a_code_is_not_one(title: str) -> None:
    """These are section-based acts. Routing them through the
    article-based hierarchy would structure them wrongly."""
    cfg = load_config("xa")
    assert cfg is not None
    assert cfg.classify_document_class(title=title) is None


class TestNumberSurvivesAModelMiss:
    """The number had one source where the year has three. A model that answers
    with the full case number, or with nothing, cost the document its identity."""

    @staticmethod
    def _descriptors(metadata: dict[str, object], title: str) -> object:
        return resolve_descriptors(
            {**metadata, "title": title},
            jurisdiction_code="xl",
            source_bytes="d0d0d0d0d0d0d0d0",
            fallback_stem="scan",
        )

    def test_a_full_case_number_is_reduced_rather_than_discarded(self) -> None:
        desc = self._descriptors(
            {"number": "45/PUL-IX/2011", "year": "2011"}, "Putusan Nomor 45/PUL-IX/2011"
        )
        assert desc.number == "45"

    def test_the_title_answers_when_the_model_does_not(self) -> None:
        desc = self._descriptors({"number": "", "year": "2011"}, "Putusan Nomor 45/PUL-IX/2011")
        assert desc.number == "45"

    def test_a_law_states_its_own_number_before_the_one_it_amends(self) -> None:
        desc = self._descriptors(
            {"number": "", "year": "2024"},
            "Undang-Undang Nomor 32 Tahun 2024 tentang Perubahan atas "
            "Undang-Undang Nomor 8 Tahun 1991 tentang Konservasi",
        )
        assert desc.number == "32"

    def test_a_number_nowhere_to_be_found_still_falls_to_the_content_address(self) -> None:
        desc = self._descriptors({"number": "", "year": "2011"}, "An untitled scan")
        assert desc.number.startswith("draft-")

    def test_trailing_punctuation_does_not_cost_the_document_its_number(self) -> None:
        desc = self._descriptors({"number": "", "year": "2011"}, "Putusan Nomor 45/PUL-IX/2011.")
        assert desc.number == "45"


class TestTheTitleFallbackDeclines:
    """A draft URI is inert; a wrong number is not. `save_document` gets-or-creates
    on the work URI, so a number belonging to another act files this document
    under that one."""

    @staticmethod
    def _number(metadata: dict[str, object], title: str) -> str:
        return resolve_descriptors(
            {**metadata, "title": title},
            jurisdiction_code="xl",
            source_bytes="d0d0d0d0d0d0d0d0",
            fallback_stem="scan",
        ).number

    def test_a_title_the_model_flagged_as_an_amendment_is_not_read(self) -> None:
        got = self._number(
            {"number": "", "year": "2024", "is_amendment": True},
            "Perubahan Atas Undang-Undang Nomor 8 Tahun 1991 tentang Konservasi",
        )
        assert got.startswith("draft-")

    def test_a_number_after_a_reference_cue_belongs_to_the_other_act(self) -> None:
        """The model is told to leave `number` empty for amendment bills, so the
        titles that reach this fallback are the ones most likely to name another
        act's number first."""
        for title in (
            "Perubahan Atas Undang-Undang Nomor 8 Tahun 1991 tentang Konservasi",
            "An Act to amend the Companies Act, No. 7 of 2007",
            "Peraturan tentang pelaksanaan Undang-Undang Nomor 6 Tahun 1983",
        ):
            assert self._number({"number": "", "year": "2024"}, title).startswith("draft-"), title

    def test_a_filename_is_not_evidence_of_a_law_s_number(self) -> None:
        """`title` falls back to the filename stem when extraction fails, and the
        metadata prompt refuses the filename for the title for the same reason."""
        desc = resolve_descriptors(
            {"number": "", "year": "2024"},
            jurisdiction_code="xl",
            source_bytes="d0d0d0d0d0d0d0d0",
            fallback_stem="batch no 12",
        )
        assert desc.number.startswith("draft-")

    def test_the_act_s_own_number_is_still_taken_when_it_leads(self) -> None:
        got = self._number(
            {"number": "", "year": "2024"},
            "Undang-Undang Nomor 32 Tahun 2024 tentang Perubahan atas "
            "Undang-Undang Nomor 8 Tahun 1991",
        )
        assert got == "32"
