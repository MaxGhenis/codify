"""The structural checks. Each test names the wrong number it prevents: a defect
that reads as a pass, or a deliberate transformation that reads as a defect."""

from __future__ import annotations

from codify.jurisdictions import JurisdictionConfig, LegalEra, load_config
from codify.quality.structural_scan import CHECKS, era_of, scan_version

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _doc(body: str, meta: str = "", attachments: str = "") -> str:
    return (
        f'<akomaNtoso xmlns="{NS}"><act>'
        f"<meta>{meta}</meta><body>{body}</body>{attachments}</act></akomaNtoso>"
    )


def _article(num: str, eid: str, extra: str = "") -> str:
    return f'<article eId="{eid}"><num>{num}</num><content><p>text</p></content>{extra}</article>'


def _point(num: str, eid: str) -> str:
    return f'<point eId="{eid}"><num>{num}</num><content><p>p</p></content></point>'


def _schedule(inner: str) -> str:
    return (
        '<attachments><attachment eId="att_1"><doc name="schedule"><mainBody>'
        f"{inner}</mainBody></doc></attachment></attachments>"
    )


_BLOB = '<section eId="sec_1"><content><p>the whole act as prose</p></content></section>'
# A `<section>` is itself a provision; this is a body with none at all.
_NO_PROVISIONS = "<p>loose prose with no provision element</p>"
_TRANSLATION = "translation_structure_preserved"


def _find(findings: list, check: str):  # type: ignore[no-untyped-def]
    return next(f for f in findings if f.check == check)


def _scan(xml: str, **kw):  # type: ignore[no-untyped-def]
    return scan_version(xml, config=load_config("xz"), doctype="act", **kw)


def test_every_check_reports_on_every_version() -> None:
    """A missing row is indistinguishable from a passing one once counted."""
    assert [f.check for f in _scan(_doc(_article("1", "art_1")))] == list(CHECKS)


class TestParseability:
    def test_no_stored_akn_is_not_a_parse_failure(self) -> None:
        """A row with no document is not a malformed document."""
        findings = _scan("")
        assert all(f.failed is None for f in findings)
        assert _find(findings, "akn_parseable").detail["reason"] == "no_akn_stored"

    def test_malformed_akn_fails_once_and_abstains_on_the_rest(self) -> None:
        findings = _scan("<akomaNtoso><act><body>")
        assert _find(findings, "akn_parseable").failed is True
        assert all(f.failed is None for f in findings[1:])


class TestBodyUnits:
    def test_a_verbatim_prose_blob_fails(self) -> None:
        """The no-anchor fallback emits a `<section>`, which a jurisdiction may declare, so
        testing any declared element passed the defect."""
        finding = _find(_scan(_doc(_BLOB)), "body_units_present")
        assert finding.failed is True
        assert finding.detail["basic_unit"] == "article"
        assert finding.detail["count"] == 0

    def test_a_schedule_article_does_not_satisfy_the_body(self) -> None:
        """Walking the whole tree let a prose body with an annexed form pass, and
        annexing forms is ordinary."""
        xml = _doc(_BLOB, attachments=_schedule(_article("1", "att_1__art_1")))
        finding = _find(_scan(xml), "body_units_present")
        assert finding.failed is True
        assert finding.detail["count"] == 0
        assert finding.detail["attachment_provisions"] == 1

    def test_an_act_with_articles_passes(self) -> None:
        assert _find(_scan(_doc(_article("1", "art_1"))), "body_units_present").failed is False

    def test_no_declared_basic_unit_cannot_be_measured(self) -> None:
        bare = JurisdictionConfig(code="zz", name="Z", tradition=["civil_law"], languages=["eng"])
        findings = scan_version(_doc(_article("1", "art_1")), config=bare, doctype="act")
        assert _find(findings, "body_units_present").failed is None


class TestEidUniqueness:
    def test_a_duplicate_provision_eid_fails(self) -> None:
        xml = _doc(_article("1", "art_1") + _article("1", "art_1"))
        finding = _find(_scan(xml), "eid_unique")
        assert finding.failed is True
        assert finding.detail["duplicate_count"] == 1

    def test_metadata_references_are_not_provisions(self) -> None:
        """`<meta>` carries eId-bearing TLC entries."""
        meta = '<references><TLCOrganization eId="art_1" href="/x"/></references>'
        assert _find(_scan(_doc(_article("1", "art_1"), meta=meta)), "eid_unique").failed is False

    def test_an_unlifted_quoted_amendment_is_not_a_duplicate(self) -> None:
        """The lift that renames `embeddedStructure` logs and skips, so an amending
        act quoting article 1 of its target reported a false duplicate."""
        quote = (
            '<mod eId="art_2__mod_1"><embeddedStructure eId="art_2__mod_1__qstr_1">'
            + _article("1", "art_1")
            + "</embeddedStructure></mod>"
        )
        xml = _doc(_article("1", "art_1") + _article("2", "art_2", extra=quote))
        assert _find(_scan(xml), "eid_unique").failed is False

    def test_a_document_with_no_provisions_cannot_be_checked(self) -> None:
        """The most-broken versions would otherwise inflate the passed bucket."""
        finding = _find(_scan(_doc(_NO_PROVISIONS)), "eid_unique")
        assert finding.failed is None
        assert finding.detail["reason"] == "no_provisions"


class TestNumbering:
    def test_two_articles_sharing_a_number_fail_at_the_basic_unit(self) -> None:
        """`_assign_eids` appends `_2` on a collision, so the suffix is the defect.
        Keying order on the whole eId made the pair uncomparable and hid it."""
        xml = _doc(_article("1", "art_1") + _article("1", "art_1_2"))
        finding = _find(_scan(xml), "numbering_monotonic_basic_unit")
        assert finding.failed is True
        assert finding.detail["breaks_by_kind"] == {"article": 1}

    def test_a_bis_article_is_not_a_defect(self) -> None:
        """Insertions are normal, so comparing `<num>` digits read every inserted
        article as a break."""
        xml = _doc(_article("5", "art_5") + _article("5 مكرر", "art_5bis") + _article("6", "art_6"))
        assert _find(_scan(xml), "numbering_monotonic_basic_unit").failed is False

    def test_a_repeated_point_does_not_count_against_the_article_rate(self) -> None:
        """One combined verdict let list-item noise decide the article rate."""
        points = _point("1", "art_1__point_1") + _point("1", "art_1__point_1_2")
        xml = _doc(_article("1", "art_1", extra=points) + _article("2", "art_2"))
        assert _find(_scan(xml), "numbering_monotonic_children").failed is True
        assert _find(_scan(xml), "numbering_monotonic_basic_unit").failed is False

    def test_word_numbered_articles_abstain_rather_than_borrow_a_verdict(self) -> None:
        """The oldest acts spell ordinals (`المادة الأولى`), and one
        document-wide counter let a nested point list earn the article verdict."""
        articles = _article("الأولى", "art_alula") + _article("الثانية", "art_althania")
        points = _point("1", "art_x__point_1") + _point("2", "art_x__point_2")
        xml = _doc(articles + f'<hcontainer eId="art_x">{points}</hcontainer>')
        basic = _find(_scan(xml), "numbering_monotonic_basic_unit")
        assert basic.failed is None
        assert basic.detail["reason"] == "numbering_unparseable"
        assert _find(_scan(xml), "numbering_monotonic_children").failed is False

    def test_ascending_articles_pass(self) -> None:
        xml = _doc(_article("1", "art_1") + _article("2", "art_2"))
        assert _find(_scan(xml), "numbering_monotonic_basic_unit").failed is False


class TestTranslations:
    def test_transliterated_eids_are_not_a_structural_loss(self) -> None:
        """Translations renumber eIds by design, so comparing identifiers reports
        an undamaged corpus as damaged."""
        source = _doc(_article("١", "art_١") + _article("٢", "art_٢"))
        translated = _doc(_article("1", "art_1") + _article("2", "art_2"))
        assert _find(_scan(source, translations={"heb": translated}), _TRANSLATION).failed is False

    def test_a_translation_that_lost_every_article_fails(self) -> None:
        source = _doc(_article("١", "art_١") + _article("٢", "art_٢"))
        assert _find(_scan(source, translations={"eng": _doc(_BLOB)}), _TRANSLATION).failed is True

    def test_a_loss_is_not_cancelled_by_a_split_in_the_same_container(self) -> None:
        """Netting let a dropped article cancel a split one, which is the standard
        translation failure pair, so losses were understated."""
        source = _doc(_article("1", "art_1") + _article("2", "art_2") + _article("3", "art_3"))
        lost_and_split = _doc(
            _article("1", "art_1") + _article("3", "art_3") + _article("3", "art_3_bis")
        )
        finding = _find(_scan(source, translations={"eng": lost_and_split}), _TRANSLATION)
        assert finding.failed is True

    def test_a_loss_is_not_cancelled_by_a_split_in_another_container(self) -> None:
        source = _doc(
            '<chapter eId="chp_1">'
            + _article("1", "chp_1__art_1")
            + _article("2", "chp_1__art_2")
            + "</chapter>"
            '<chapter eId="chp_2">' + _article("3", "chp_2__art_3") + "</chapter>"
        )
        translated = _doc(
            '<chapter eId="chp_1">' + _article("1", "chp_1__art_1") + "</chapter>"
            '<chapter eId="chp_2">'
            + _article("3", "chp_2__art_3")
            + _article("4", "chp_2__art_4")
            + "</chapter>"
        )
        assert _find(_scan(source, translations={"eng": translated}), _TRANSLATION).failed is True

    def test_a_split_alone_is_not_a_loss(self) -> None:
        """A translation may split a provision the source ran together."""
        source = _doc(_article("1", "art_1"))
        split = _doc(_article("1", "art_1") + _article("2", "art_2"))
        finding = _find(_scan(source, translations={"eng": split}), _TRANSLATION)
        assert finding.failed is False
        assert finding.detail["split"] == {"eng": 1}

    def test_dropping_a_footnote_is_not_a_lost_provision(self) -> None:
        """`authorialNote` carries an eId, so counting it as a provision damned
        every translation that omits footnotes."""
        note = '<authorialNote eId="fn_1"><p>note</p></authorialNote>'
        source = _doc(_article("1", "art_1", extra=note))
        without = _doc(_article("1", "art_1"))
        assert _find(_scan(source, translations={"eng": without}), _TRANSLATION).failed is False

    def test_a_dropped_wrapper_around_an_unchanged_child_is_not_a_loss(self) -> None:
        """Excluding wrappers from the sequence is not enough while one can still
        be a parent: `(point, hcontainer)` against `(point, article)` reads as a
        deletion, and Bluebell re-derives wrappers per parse."""
        point = '<point eId="{eid}"><num>1</num><content><p>a</p></content></point>'
        wrapped = _article(
            "1",
            "art_1",
            extra=f'<hcontainer eId="art_1__hc_1">{point.format(eid="p1")}</hcontainer>',
        )
        flat = _article("1", "art_1", extra=point.format(eid="art_1__point_1"))
        finding = _find(
            _scan(_doc(wrapped), translations={"eng": _doc(flat)}),
            "translation_structure_preserved",
        )
        assert finding.failed is False

    def test_a_lost_named_hcontainer_is_a_loss(self) -> None:
        """`<hcontainer>` carries law as well as scaffolding: a config may
        declare `mukrrar`, the bis article. Generic-by-tag hid a dropped one."""
        mukrrar = (
            '<hcontainer name="mukrrar" eId="art_1bis"><num>1 مكرر</num>'
            "<content><p>inserted</p></content></hcontainer>"
        )
        source = _doc(_article("1", "art_1") + mukrrar)
        dropped = _doc(_article("1", "art_1"))
        finding = _find(_scan(source, translations={"eng": dropped}), _TRANSLATION)
        assert finding.failed is True
        assert finding.detail["diverged"]["eng"] == {"mukrrar": 1}

    def test_bluebells_own_wrapper_is_still_scaffolding(self) -> None:
        """Every wrapper in the corpus carries this default name, so it must stay
        generic or the wrapper-noise fix regresses."""
        wrapped = _doc(
            '<hcontainer name="hcontainer" eId="hc_1">' + _article("1", "art_1") + "</hcontainer>"
        )
        flat = _doc(_article("1", "art_1"))
        assert _find(_scan(wrapped, translations={"eng": flat}), _TRANSLATION).failed is False

    def test_no_translations_is_not_a_pass(self) -> None:
        assert _find(_scan(_doc(_article("1", "art_1"))), _TRANSLATION).failed is None

    def test_a_source_with_no_provisions_cannot_be_compared(self) -> None:
        finding = _find(
            _scan(_doc(_NO_PROVISIONS), translations={"eng": _doc(_NO_PROVISIONS)}),
            _TRANSLATION,
        )
        assert finding.failed is None
        assert finding.detail["reason"] == "source_has_no_provisions"


class TestChecksNeedingSourceText:
    def test_both_abstain_without_the_source(self) -> None:
        """Retaining the source is what lets these two run at all."""
        findings = _scan(_doc(_article("1", "art_1")))
        for check in ("toc_entries_have_bodies", "anchor_coverage"):
            finding = _find(findings, check)
            assert finding.failed is None
            assert finding.detail["reason"] == "no_source_text"

    def test_the_source_lets_coverage_return_a_verdict(self) -> None:
        """Abstaining on every version is the state this replaces."""
        source = "مادة (1)\nنص الأولى.\n\nمادة (2)\nنص الثانية.\n"
        body = _article("1", "art_1") + _article("2", "art_2")
        finding = _find(_scan(_doc(body), source_text=source), "anchor_coverage")
        assert finding.failed is False
        assert finding.detail["expected"] == 2


class TestCoverageCanFail:
    """The ratio alone cannot: its denominator is built from the keywords the
    scan matched, so a keyword the scan could not read leaves both sides."""

    # `منها` is a real word, so no alias recovers it, but the line is
    # marker-shaped and the scan declares it unclaimed. A recovery pass that
    # later claims it fails these tests rather than hollowing them out.
    _MANGLED = "منها (1)\nنص الأولى.\n\nمنها (2)\nنص الثانية.\n"
    _MIXED = "مادة (1)\nنص الأولى.\n\nمنها (2)\nنص الثانية.\n"
    _PROSE_CITE = "مادة (1)\nنص.\n\nوفقاً لأحكام المادة (7) من هذا القانون.\n\nمادة (2)\nنص.\n"

    def test_markers_the_scan_could_not_read_fail_the_check(self) -> None:
        """Every marker mangled, so the ratio has no denominator to fail on."""
        finding = _find(_scan(_doc(_BLOB), source_text=self._MANGLED), "anchor_coverage")
        assert finding.failed is True
        assert finding.detail["unclaimed_markers"] == 2

    def test_one_unread_marker_among_read_ones_fails(self) -> None:
        """The shape the corpus presents and the ratio hides: the markers it did
        read give 1.0 while a marker is lost."""
        body = _article("1", "art_1") + _article("2", "art_2")
        finding = _find(_scan(_doc(body), source_text=self._MIXED), "anchor_coverage")
        assert finding.failed is True
        assert finding.detail["ratio"] >= 1.0
        assert finding.detail["unclaimed_markers"] == 1

    def test_a_cross_reference_in_prose_is_not_an_unclaimed_marker(self) -> None:
        """`unmatched_marker` also covers markers the scan deliberately refused.
        Refused prose citations must not count as missing structural markers."""
        body = _article("1", "art_1") + _article("2", "art_2")
        finding = _find(_scan(_doc(body), source_text=self._PROSE_CITE), "anchor_coverage")
        assert finding.failed is False
        assert finding.detail["unclaimed_markers"] == 0

    def test_a_source_presenting_no_markers_still_abstains(self) -> None:
        """A one-page decree with no markers has nothing to be measured against,
        and must not be scored as though it lost them."""
        finding = _find(
            _scan(_doc(_BLOB), source_text="نص بلا أي علامات هيكلية على الإطلاق.\n"),
            "anchor_coverage",
        )
        assert finding.failed is None
        assert finding.detail["reason"] == "no_markers_in_source"

    def test_every_verdict_carries_the_sets_it_was_formed_from(self) -> None:
        """A bare boolean cannot be checked against the source."""
        detail = _find(_scan(_doc(_BLOB), source_text=self._MANGLED), "anchor_coverage").detail
        assert {"captured", "expected", "missing", "unclaimed_markers"} <= set(detail)

    def test_the_stored_source_is_normalised_before_it_is_scanned(self) -> None:
        """Versions retain the raw extraction. Scanning it unnormalised finds none
        of the markers the structurer saw, so every document would read as lost."""
        bidi = "\u202bمادة \u202a(1)\u202c\u202c\nنص.\n\n\u202bمادة \u202a(2)\u202c\u202c\nنص.\n"
        body = _article("1", "art_1") + _article("2", "art_2")
        finding = _find(_scan(_doc(body), source_text=bidi), "anchor_coverage")
        assert finding.failed is False
        assert finding.detail["captured"] == 2


class TestEras:
    def test_a_year_lands_in_its_declared_era(self) -> None:
        config = load_config("xz")
        assert era_of(config, 1936) == "protectorate"
        assert era_of(config, 1979) == "assembly"
        assert era_of(config, 2020) == "decree_law"

    def test_a_transition_year_belongs_to_the_earlier_era(self) -> None:
        """Eras carry exact dates; most works carry only a year."""
        config = load_config("xz")
        assert era_of(config, 1961) == "protectorate"
        assert era_of(config, 1962) == "assembly"

    def test_the_three_no_era_reasons_stay_apart(self) -> None:
        """No config loaded, declares no eras, and year missing were one bucket,
        so its abstention count meant something different from row to row."""
        assert era_of(None, 1998) == "no_config"
        assert era_of(load_config("gb"), 1998) == "no_eras_declared"
        assert era_of(load_config("xz"), None) == "unknown"

    def test_a_year_outside_every_era_names_the_config_gap(self) -> None:
        config = JurisdictionConfig(
            code="zz",
            name="Z",
            tradition=["civil_law"],
            languages=["eng"],
            legal_eras=[LegalEra(id="modern", **{"from": "2000-01-01"})],
        )
        assert era_of(config, 1900) == "outside_declared_eras"
