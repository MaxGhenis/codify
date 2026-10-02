"""Audit extensions + repair loop tests for Phase 1."""

from __future__ import annotations

from typing import Any

import pytest

from codify.translate.anchors import SourceUnit
from codify.translate.audit import audit_translation
from codify.translate.numeric_extract import extract_tokens
from codify.translate.repair import (
    MAX_REPAIR_ITERATIONS,
    RepairAttempt,
    repair_provision,
)


class TestAuditPhaseOneExtensions:
    def test_backward_compatible_when_no_extras(self):
        """Old-shape call (no keyword args) still returns the pre-existing
        fields plus zero-valued Phase 1 extras."""
        out = audit_translation("source", "translated", {})
        assert "content_coverage" in out
        assert out["sentinels_expected"] == 0
        assert out["sentinels_missing"] == 0
        assert out["numeric_slot_recall"] == 1.0
        assert out["num_alien_script_hits"] == 0
        assert out["header_bleed_hits"] == {}

    def test_numeric_slot_recall_computed(self):
        out = audit_translation(
            "source",
            "translated",
            {},
            sentinels_expected=10,
            sentinels_missing=2,
        )
        assert out["numeric_slot_recall"] == 0.8
        assert out["sentinel_preservation"] == 0.8

    def test_zero_expected_is_perfect_recall(self):
        out = audit_translation(
            "source",
            "translated",
            {},
            sentinels_expected=0,
            sentinels_missing=0,
        )
        assert out["numeric_slot_recall"] == 1.0

    def test_missing_by_eid_persists(self):
        out = audit_translation(
            "src",
            "tgt",
            {},
            sentinels_expected=5,
            sentinels_missing=2,
            sentinels_missing_by_eid={"art_5": ["N003", "N004"]},
        )
        assert out["sentinels_missing_by_eid"] == {"art_5": ["N003", "N004"]}

    def test_alien_script_detected_in_num_when_target_eng(self):
        """A translated AKN with Arabic letters inside <num> is contamination
        for target eng."""
        translated_xml = (
            "<akn:akn xmlns:akn='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<akn:article><akn:num>مادة 5</akn:num><akn:p>text</akn:p></akn:article>"
            "</akn:akn>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
        )
        assert out["num_alien_script_hits"] == 1

    def test_alien_script_ignored_for_ara_target(self):
        translated_xml = (
            "<akn:akn xmlns:akn='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<akn:article><akn:num>مادة 5</akn:num><akn:p>text</akn:p></akn:article>"
            "</akn:akn>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="ara")
        assert out["num_alien_script_hits"] == 0

    def test_alien_script_clean_ascii_num_is_fine(self):
        translated_xml = (
            "<akn:akn xmlns:akn='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<akn:article><akn:num>5</akn:num><akn:p>text</akn:p></akn:article>"
            "</akn:akn>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="eng")
        assert out["num_alien_script_hits"] == 0

    def test_header_bleed_regex_detected(self):
        """If a running-header pattern still fires against the translated
        text, the audit catches it."""
        translated = "Some prose here.\nAl-Waqa'i Al-Zerzuriyya\nMore prose after the header."
        out = audit_translation(
            "source",
            translated,
            {},
            header_patterns=[r"Al\s*-?\s*Waqa'?i\s+Al\s*-?\s*Zerzuriyya"],
        )
        assert out["header_bleed_hits"]
        assert len(out["header_bleed_hits"][r"Al\s*-?\s*Waqa'?i\s+Al\s*-?\s*Zerzuriyya"]) == 1

    def test_header_bleed_clean_output_no_hits(self):
        translated = "Fully translated substantive text with no header residue."
        out = audit_translation(
            "source",
            translated,
            {},
            header_patterns=[r"Al\s*-?\s*Waqa'?i\s+Al\s*-?\s*Zerzuriyya"],
        )
        assert out["header_bleed_hits"] == {}

    def test_bad_regex_pattern_survives_without_crashing(self):
        translated = "text"
        out = audit_translation(
            "src",
            translated,
            {},
            header_patterns=["[unclosed"],
        )
        # No hit reported for the bad pattern; audit doesn't crash.
        assert out["header_bleed_hits"] == {}


# --- v3.8 audit extensions ---------------------------------------------


_AKN_HEADER = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _akn_wrap(body_inner: str) -> str:
    """Minimal AKN skeleton with a `<body>` populated by ``body_inner``."""
    return f'<akomaNtoso xmlns="{_AKN_HEADER}"><act><body>{body_inner}</body></act></akomaNtoso>'


class TestAuditV38Extensions:
    """Audit per-provision sentinels, monetary recall and heading scripts."""

    def test_stray_sentinels_by_eid_persists(self):
        """Analogue of the existing sentinels-missing-by-eid test.
        Ensures the raise helper in the persist step can name eids."""
        out = audit_translation(
            "src",
            "tgt",
            {},
            stray_sentinels=1,
            stray_sentinels_by_eid={"art_5": ["⟨N003⟩"]},
        )
        assert out["stray_sentinels_by_eid"] == {"art_5": ["⟨N003⟩"]}
        assert out["stray_sentinels"] == 1

    def test_backward_compatible_when_stray_by_eid_absent(self):
        """An older caller that only knows about the `stray_sentinels`
        scalar still gets a clean audit; the new field defaults to
        empty dict."""
        out = audit_translation("src", "tgt", {}, stray_sentinels=0)
        assert out["stray_sentinels_by_eid"] == {}

    def test_heading_alien_hits_flags_arabic_heading(self):
        """Untranslated source-script headings are reported by provision."""
        translated_xml = _akn_wrap(
            "<article eId='art_47'>"
            "<num>47</num><heading>صندوق العدسات الزرقاء</heading>"
            "<content><p>Translated body text.</p></content>"
            "</article>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="eng")
        assert out["heading_alien_script_hits"] == 1
        # Copilot #7: per-eid map lets a downstream repair loop target the
        # offending provision rather than just knowing the count.
        assert out["heading_alien_script_hits_by_eid"] == {"art_47": 1}
        # Body-`<p>` is clean English; the body counter stays at zero,
        # so the two checks don't confuse an operator diagnosing which
        # element type carries the residue.
        assert out["body_alien_script_hits"] == 0

    def test_heading_alien_flags_mixed_script_heading(self):
        """The configured threshold catches partly translated headings."""
        # Word-ish alien block big enough to clear 20% of non-whitespace
        # chars: `Retention of ملكية والحقوق rights` is roughly 30%
        # Arabic letters.
        translated_xml = _akn_wrap(
            "<article eId='art_x'>"
            "<num>X</num>"
            "<heading>Retention of ملكية والحقوق rights</heading>"
            "<content><p>Body prose.</p></content>"
            "</article>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="eng")
        assert out["heading_alien_script_hits"] == 1
        assert out["heading_alien_script_hits_by_eid"] == {"art_x": 1}

    def test_heading_alien_ignores_ara_target(self):
        """When the target is Arabic, Arabic headings are the intended
        shape; not a defect."""
        translated_xml = _akn_wrap(
            "<article eId='art_1'><num>1</num><heading>صندوق العدسات الزرقاء</heading>"
            "<content><p>محتوى</p></content></article>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="ara")
        assert out["heading_alien_script_hits"] == 0

    def test_heading_alien_clean_translation_passes(self):
        translated_xml = _akn_wrap(
            "<article eId='art_1'>"
            "<num>1</num>"
            "<heading>Ownership of upper and lower storeys</heading>"
            "<content>"
            "<p>The upper and lower storeys of a building shall be owned separately.</p>"
            "</content>"
            "</article>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="eng")
        assert out["heading_alien_script_hits"] == 0

    def test_heading_alien_unparseable_flagged_distinctly(self):
        """`heading_alien_script_hits` is zero on unparseable input, but
        the companion `_unparseable` flag lets the operator distinguish
        drift from a clean pass; mirrors the body-`<p>` variant."""
        out = audit_translation("src", "not valid xml", {}, target_language="eng")
        # Both body and heading checks share the same unparseable signal.
        assert out["heading_alien_script_hits"] == 0
        assert out["heading_alien_script_unparseable"] is True

    def test_money_missing_flags_dropped_penalty(self):
        """A synthetic deposit with no amount fails the monetary recall check."""
        translated_xml = _akn_wrap(
            "<article eId='art_61'>"
            "<num>61</num>"
            "<content>"
            "<p eId='art_61__content__p_1'>The fictional observatory collects a deposit "
            "denominated in Dinars.</p>"
            "</content></article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={"art_61": ["2,400"]},
        )
        assert out["money_missing_by_eid"] == {"art_61": ["2,400"]}
        assert out["money_duplicated_by_eid"] == {}

    def test_money_missing_distinct_amounts_not_covered_by_repeat(self):
        """Copilot #2: source `2,400` + `3,000`, target `2,400` twice.
        Aggregate counts match, but a distinct amount is dropped; the
        multiset compare catches it where a count compare would not."""
        translated_xml = _akn_wrap(
            "<article eId='art_5'>"
            "<num>5</num>"
            "<content>"
            "<p eId='art_5__content__p_1'>Fines of 2,400 and 2,400 dinars apply.</p>"
            "</content></article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={"art_5": ["2,400", "3,000"]},
        )
        assert out["money_missing_by_eid"] == {"art_5": ["3,000"]}
        assert out["money_duplicated_by_eid"] == {"art_5": ["2,400"]}

    def test_money_duplicated_flags_double_emission(self):
        """Repeated numeric output must not count as one source amount."""
        translated_xml = _akn_wrap(
            "<article eId='art_8'>"
            "<num>8</num>"
            "<content>"
            "<p eId='art_8__content__p_1'>A fine of (68,000) 68,000 Zerzuran Dinars.</p>"
            "</content></article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={"art_8": ["68,000"]},
        )
        assert out["money_duplicated_by_eid"] == {"art_8": ["68,000"]}
        assert out["money_missing_by_eid"] == {}

    def test_money_clean_source_target_match(self):
        """A translated body containing the expected surface once matches
        source; no missing, no duplicated."""
        translated_xml = _akn_wrap(
            "<article eId='art_10'>"
            "<num>10</num>"
            "<content>"
            "<p eId='art_10__content__p_1'>A fine of 2,400 Zerzuran Dinars applies.</p>"
            "</content></article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={"art_10": ["2,400"]},
        )
        assert out["money_missing_by_eid"] == {}
        assert out["money_duplicated_by_eid"] == {}

    def test_money_direct_p_eid_attributes_to_owner(self):
        """Copilot #3: a direct `<p eId='art_1__p_1'>` (no intro/content
        scope) must attribute to `art_1`, not to the raw `<p>` eid. The
        `_owning_eid` strip must handle the `__p_N` suffix directly."""
        translated_xml = _akn_wrap(
            "<article eId='art_1'>"
            "<num>1</num>"
            "<p eId='art_1__p_1'>A fine of 2,400 Zerzuran Dinars applies.</p>"
            "</article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={"art_1": ["2,400"]},
        )
        assert out["money_missing_by_eid"] == {}

    def test_money_source_zero_skips_check(self):
        """If the source manifest carried no money tokens for the eid,
        the check does not fire; no false positives on articles that
        legitimately have no penalty."""
        translated_xml = _akn_wrap(
            "<article eId='art_2'>"
            "<num>2</num>"
            "<content><p eId='art_2__content__p_1'>Definitions apply.</p></content>"
            "</article>"
        )
        out = audit_translation(
            "src",
            translated_xml,
            {},
            target_language="eng",
            source_money_surfaces_by_eid={},
        )
        assert out["money_missing_by_eid"] == {}
        assert out["money_duplicated_by_eid"] == {}

    def test_money_check_disabled_when_kwarg_absent(self):
        """Without `source_money_surfaces_by_eid`, the counters default to
        empty; old callers stay green."""
        translated_xml = _akn_wrap(
            "<article eId='art_1'>"
            "<num>1</num>"
            "<content><p>A fine of 2,400 Zerzuran Dinars applies.</p></content>"
            "</article>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="eng")
        assert out["money_missing_by_eid"] == {}
        assert out["money_duplicated_by_eid"] == {}


# --- Repair loop --------------------------------------------------------


class _RepairFixesEverythingLLM:
    """Stub that returns the sentinel-encoded source verbatim as the
    repaired output; all sentinels appear correctly on decode."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        import json

        # Parse the JSON payload out of the prompt (comes after "Provision:")
        start = prompt.find("Provision:\n")
        assert start >= 0
        payload = json.loads(prompt[start + len("Provision:\n") :])
        return schema.model_validate(
            {
                "eid": payload["eid"],
                "heading": None,
                "lines": [payload["body_source_with_sentinels"]],
            }
        )


class _RepairStillDropsLLM:
    """Stub that returns the prior translation unchanged; repair doesn't
    fix anything. Used to prove the giving-up-after-N path."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        import json

        start = prompt.find("Provision:\n")
        assert start >= 0
        payload = json.loads(prompt[start + len("Provision:\n") :])
        return schema.model_validate(
            {
                "eid": payload["eid"],
                "heading": None,
                "lines": payload["prior_translation"]["lines"],
            }
        )


def _unit(eid: str, body: str) -> SourceUnit:
    return SourceUnit(
        akn_eid=eid,
        kind="article",
        akn_type="article",
        number="1",
        depth=1,
        heading=None,
        body_text=body,
    )


class TestRepairProvision:
    async def test_repair_restores_missing_sentinels(self):
        """When the stub returns the encoded source, all sentinels
        round-trip; remaining_missing is empty."""
        unit = _unit("art_5", "See Article 5 within 30 days.")
        manifest = extract_tokens(unit.body_text)
        # Fake prior translation where both sentinels were dropped
        prior_lines = ["See Article  within  days."]
        attempt = await repair_provision(
            unit,
            manifest,
            prior_lines=prior_lines,
            prior_heading=None,
            target_language="Identity",
            llm=_RepairFixesEverythingLLM(),
            missing_sentinels=[t.sentinel_id for t in manifest.tokens],
            stray_sentinels=[],
        )
        assert isinstance(attempt, RepairAttempt)
        assert attempt.eid == "art_5"
        assert attempt.remaining_missing == []

    async def test_repair_does_not_fix_reports_remaining(self):
        unit = _unit("art_5", "See Article 5.")
        manifest = extract_tokens(unit.body_text)
        prior_lines = ["See Article ."]  # sentinel dropped
        attempt = await repair_provision(
            unit,
            manifest,
            prior_lines=prior_lines,
            prior_heading=None,
            target_language="Identity",
            llm=_RepairStillDropsLLM(),
            missing_sentinels=[manifest.tokens[0].sentinel_id],
            stray_sentinels=[],
        )
        # Repair failed to restore; remaining_missing reports the failure.
        assert attempt.remaining_missing == [manifest.tokens[0].sentinel_id]

    def test_max_iterations_constant_sensible(self):
        assert MAX_REPAIR_ITERATIONS >= 1
        assert MAX_REPAIR_ITERATIONS <= 3


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


class TestMissingContainerTitleAudit:
    """Audit surfaces headingless containers by eid so
    the persist gate can block delivery."""

    _NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

    def test_headingless_chapter_reported(self):
        translated_xml = (
            f"<akn xmlns='{self._NS}'>"
            "<chapter eId='chp_5'><num>5</num>"
            "<article eId='chp_5__art_1'><num>1</num><p>text</p></article>"
            "</chapter></akn>"
        )
        out = audit_translation("src", translated_xml, {}, translated_akn_xml=translated_xml)
        assert out["missing_container_titles_by_eid"] == {"chp_5": "chapter"}

    def test_titled_chapter_clean(self):
        translated_xml = (
            f"<akn xmlns='{self._NS}'>"
            "<chapter eId='chp_5'><num>5</num><heading>General provisions</heading>"
            "<article eId='chp_5__art_1'><num>1</num><p>text</p></article>"
            "</chapter></akn>"
        )
        out = audit_translation("src", translated_xml, {}, translated_akn_xml=translated_xml)
        assert out["missing_container_titles_by_eid"] == {}
        assert out["missing_container_titles_unparseable"] is False

    def test_unparseable_xml_reports_empty_and_flags_unparseable(self):
        """Unparseable XML yields an empty by-eid dict AND the unparseable
        flag, so the persist gate knows the empty dict is untrusted."""
        out = audit_translation("src", "not xml at all", {})
        assert out["missing_container_titles_by_eid"] == {}
        assert out["missing_container_titles_unparseable"] is True

    def test_translated_positional_arg_used_when_no_explicit_xml(self):
        """The fallback branch: `translated` itself is parseable AKN and no
        translated_akn_xml is passed."""
        translated_xml = (
            f"<akn xmlns='{self._NS}'>"
            "<chapter eId='chp_2'><num>2</num>"
            "<article eId='chp_2__art_1'><num>1</num><p>text</p></article>"
            "</chapter></akn>"
        )
        out = audit_translation("src", translated_xml, {})
        assert out["missing_container_titles_by_eid"] == {"chp_2": "chapter"}


class TestUntranslatedMarkerCounter:
    """The delivery gate reads untranslated_marker_hits; this pins the
    producer side so a key rename cannot silently disable the safety net."""

    def test_marker_in_target_counts(self) -> None:
        from codify.translate.audit import audit_translation

        audit = audit_translation("نص المصدر", "text with [UNTRANSLATED] inside", {})
        assert audit["untranslated_marker_hits"] >= 1

    def test_marker_in_xml_counts_and_none_is_safe(self) -> None:
        from codify.translate.audit import audit_translation

        with_xml = audit_translation("نص", "clean", {}, translated_akn_xml="<p>[UNTRANSLATED]</p>")
        assert with_xml["untranslated_marker_hits"] == 1
        without = audit_translation("نص", "clean", {}, translated_akn_xml=None)
        assert without["untranslated_marker_hits"] == 0


_UNTRANSLATED_BODY = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act><body><article eId="art_35"><num>35</num>
    <paragraph eId="art_35__point_1"><num>1-</num><content>
      <p>كفالة حسن التنفيذ: تكون بنسبة ١٠٪ من قيمة العقد.</p>
    </content></paragraph>
  </article></body></act>
</akomaNtoso>"""


def test_alien_patterns_resolve_from_a_language_name() -> None:
    """Runs carry "English", not "eng"; the missed lookup read as a clean zero."""
    from codify.translate.audit import alien_patterns_for

    assert alien_patterns_for("English")
    assert alien_patterns_for("eng")
    assert alien_patterns_for("Hebrew")
    assert not alien_patterns_for("Klingon")


def test_untranslated_body_paragraph_is_counted_for_a_named_target() -> None:
    audit = audit_translation(
        "source",
        "target",
        {},
        target_language="English",
        translated_akn_xml=_UNTRANSLATED_BODY,
    )
    assert audit["body_alien_script_hits"] == 1
    assert "art_35__point_1" in audit["body_alien_script_hits_by_eid"]
    assert audit["body_alien_script_unparseable"] is False


def test_unknown_target_language_reports_not_checked_not_clean() -> None:
    audit = audit_translation(
        "source",
        "target",
        {},
        target_language="Klingon",
        translated_akn_xml=_UNTRANSLATED_BODY,
    )
    assert audit["body_alien_script_unparseable"] is True


def test_letter_spaced_runs_are_counted_per_eid() -> None:
    """Repair letter-spaced output without introducing lexical changes."""
    from codify.translate.audit import count_letter_spaced_runs

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_6'><paragraph eId='art_6__point_6'>"
        "<content><p>When r e s o r t i n g to Direct Purchase, the Procuring "
        "Entity shall prepare an estimate.</p></content>"
        "</paragraph></article></body></act></akomaNtoso>"
    )
    result = count_letter_spaced_runs(xml)
    assert result is not None
    hits, by_eid = result
    assert hits == 1
    assert by_eid == {"art_6__point_6": ["r e s o r t i n g"]}


def test_ordinary_prose_and_short_sequences_do_not_trip_the_check() -> None:
    from codify.translate.audit import count_letter_spaced_runs

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_1'><content>"
        "<p>The Minister shall issue a decision within 30 days, and a b c "
        "is a short sequence.</p>"
        "</content></article></body></act></akomaNtoso>"
    )
    result = count_letter_spaced_runs(xml)
    assert result == (0, {})


def test_letter_spacing_check_reports_unparseable_rather_than_clean() -> None:
    from codify.translate.audit import count_letter_spaced_runs

    assert count_letter_spaced_runs("<not xml") is None


def test_spaced_digits_are_not_letter_spacing() -> None:
    """A table row of single digits is legitimate; only letters are prose."""
    from codify.translate.audit import count_letter_spaced_runs

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_1'><content>"
        "<p>1 2 3 4 5 6</p></content></article></body></act></akomaNtoso>"
    )
    assert count_letter_spaced_runs(xml) == (0, {})


def test_a_spelled_out_initialism_is_not_letter_spacing() -> None:
    """UNRWA and UNESCO appear spelled out in this corpus, and the gate is a
    hard fail, so a false positive would wedge the law."""
    from codify.translate.audit import count_letter_spaced_runs

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_1'><content>"
        "<p>The U N R W A convention and the U N E S C O charter apply.</p>"
        "</content></article></body></act></akomaNtoso>"
    )
    assert count_letter_spaced_runs(xml) == (0, {})


def test_audit_translation_surfaces_the_letter_spacing_counters() -> None:
    """The gate keys off these exact names, so the wiring needs its own guard."""
    from codify.translate.audit import audit_translation

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_6'><content>"
        "<p>When r e s o r t i n g to Direct Purchase.</p>"
        "</content></article></body></act></akomaNtoso>"
    )
    result = audit_translation("مصدر", "When resorting", {}, translated_akn_xml=xml)
    assert result["letter_spaced_runs"] == 1
    assert result["letter_spaced_runs_by_eid"] == {"art_6": ["r e s o r t i n g"]}
    assert result["letter_spaced_unparseable"] is False


def test_a_document_with_no_body_is_not_reported_clean() -> None:
    """A fragment or a missing namespace sweeps nothing, which is the same
    "did not run" state as unparseable rather than a clean zero."""
    from codify.translate.audit import count_letter_spaced_runs

    fragment = "<article eId='art_1'><p>r e s o r t i n g</p></article>"
    assert count_letter_spaced_runs(fragment) is None


def test_a_long_all_caps_run_is_a_corruption_not_an_initialism() -> None:
    """The exemption must not swallow the defect it sits next to: a corrupted
    word in an upper-case heading is still letter spacing."""
    from codify.translate.audit import count_letter_spaced_runs

    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        "<act><body><article eId='art_1'><content>"
        "<p>N E G O T I A T I O N S shall be conducted.</p>"
        "</content></article></body></act></akomaNtoso>"
    )
    hits, by_eid = count_letter_spaced_runs(xml)
    assert hits == 1
    assert by_eid == {"art_1": ["N E G O T I A T I O N S"]}
