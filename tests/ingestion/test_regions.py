"""Block shapes measured on the two named cases: types, ordinates, markers."""

from __future__ import annotations

import re

from codify.pipeline.enrich.ocr import OcrBlock, PageDimensions
from codify.pipeline.enrich.regions import (
    DEFAULT_FOOTNOTE_MARKERS,
    Region,
    RegionVocabulary,
    classify_page,
    vocabulary_for,
)

# Modern-era closing phrase; the older eras use their own.
ARABIC_VOCAB = RegionVocabulary(closing_phrases=("صدر بمدينة",))
A4 = PageDimensions(dpi=87, width=720, height=1018)


def _b(kind: str, y0: int, y1: int, content: str = "") -> OcrBlock:
    return OcrBlock(
        type=kind,
        content=content,
        top_left_x=0,
        top_left_y=y0,
        bottom_right_x=670,
        bottom_right_y=y1,
    )


def _kinds(regions: list[Region]) -> list[str]:
    return [r.kind for r in regions]


class TestAttestation:
    """The endpoint types every attestation line `text`, never `signature`."""

    def _page(self) -> list[OcrBlock]:
        return [
            _b("header", 170, 187, "الوقائع الزرزورية"),
            _b("title", 551, 572, "# **مادة (٧٩)**"),
            _b("text", 579, 634, "على جميع الجهات المختصة، كل فيما يخصه، تنفيذ أحكام هذه اللائحة"),
            _b("text", 653, 678, "صدر بمدينة زرزورة بتاريخ : ١٢ / ٤ / ٢٠٠٤ ميلادية"),
            _b("text", 683, 706, "الموافق : ٢٢ / صفر / ١٤٢٥ هجرية"),
            _b("text", 722, 747, "**سالم بن رشيد (أبو فارس)**"),
            _b("text", 752, 774, "**رئيس مجلس الوزراء**"),
            _b("footer", 825, 842, "-١٩٨-"),
        ]

    def test_the_final_article_keeps_its_own_body(self) -> None:
        regions = classify_page(self._page(), A4, page_number=32, vocab=ARABIC_VOCAB)
        assert _kinds(regions) == [
            "furniture",
            "body",
            "body",
            "conclusions",
            "conclusions",
            "conclusions",
            "conclusions",
            "furniture",
        ]

    def test_the_closing_phrase_is_what_starts_it(self) -> None:
        """Phrase plus position carry it, because the type never fires."""
        first = [r for r in classify_page(self._page(), A4, page_number=32, vocab=ARABIC_VOCAB)][3]
        assert "phrase_closing" in first.signals
        assert "after_last_heading" in first.signals
        assert "type_signature" not in first.signals

    def test_the_signatory_lines_continue_it_without_a_phrase(self) -> None:
        """Name and role carry no phrase; reading order binds them."""
        regions = classify_page(self._page(), A4, page_number=32, vocab=ARABIC_VOCAB)
        assert regions[5].detail == "continues the attestation"
        assert regions[6].detail == "continues the attestation"

    def test_without_the_jurisdiction_phrase_nothing_is_lifted(self) -> None:
        """No declared phrase means no text lifted. One signal never decides."""
        regions = classify_page(self._page(), A4, page_number=32, vocab=RegionVocabulary())
        assert "conclusions" not in _kinds(regions)


class TestFootnotes:
    """16 of 16 sit at or below 0.82 of page height and open with a superscript."""

    PAGE_H = PageDimensions(dpi=87, width=720, height=1023)

    def test_a_reference_in_the_band_with_a_marker_is_a_footnote(self) -> None:
        blocks = [
            _b("text", 300, 500, "المادة (١) تعاريف"),
            _b(
                "references", 881, 899, "¹ عدلت بموجب المادة (٧) من القرار بقانون رقم (٧) لسنة ٢٠١٠"
            ),
            _b("references", 901, 920, "² أضيفت بموجب المادة (١٥) من القرار بقانون رقم (٧)"),
            _b("footer", 950, 970, "-١٢-"),
        ]
        assert _kinds(classify_page(blocks, self.PAGE_H, page_number=7)) == [
            "body",
            "footnote",
            "footnote",
            "furniture",
        ]

    def test_the_older_parenthesised_marker_is_accepted_too(self) -> None:
        """46 of 48 older footnotes open `(١)`. One form only drops an era."""
        block = _b("references", 900, 921, "(١) مفسوخ بموجب الأمر العسكري")
        [region] = classify_page([block], self.PAGE_H, page_number=3)
        assert region.kind == "footnote"
        assert "marker_footnote" in region.signals

    def test_a_reference_high_on_the_page_without_a_marker_is_flagged(self) -> None:
        """Stays in the body and says so, rather than trusting the type alone."""
        block = _b("references", 200, 260, "انظر القانون رقم ٥ لسنة ١٩٩٩")
        [region] = classify_page([block], self.PAGE_H, page_number=4)
        assert region.kind == "body"
        assert region.flagged
        assert "band" in region.detail


class TestTheInverseError:
    def test_a_heading_the_engine_called_furniture_is_flagged_not_dropped(self) -> None:
        """The failure the furniture work left undetectable."""
        block = _b("header", 170, 190, "# **مادة (٧٧)**")
        [region] = classify_page([block], A4, page_number=5)
        assert region.kind == "body"
        assert region.flagged

    def test_an_ordinary_running_head_is_furniture_and_says_nothing(self) -> None:
        block = _b("header", 170, 187, "الوقائع الزرزورية")
        [region] = classify_page([block], A4, page_number=5)
        assert (region.kind, region.flagged) == ("furniture", False)


class TestTheDocumentedVocabulary:
    def test_every_documented_type_is_mapped(self) -> None:
        """Absence from one sample is not absence from the vocabulary."""
        documented = [
            "text",
            "title",
            "list",
            "table",
            "image",
            "equation",
            "caption",
            "code",
            "references",
            "aside_text",
            "header",
            "footer",
            "signature",
        ]
        blocks = [_b(t, 300, 340, "body") for t in documented]
        regions = classify_page(blocks, A4, page_number=1)
        unmapped = [r.block_type for r in regions if "unmapped" in r.detail]
        assert unmapped == []

    def test_a_marginal_label_becomes_an_aside_rather_than_a_definition(self) -> None:
        """The `تعاريف` label that contaminated the military orders: a narrow
        block in the margin, where band and type agree."""
        block = OcrBlock(
            type="aside_text",
            content="تعاريف",
            top_left_x=590,
            top_left_y=300,
            bottom_right_x=700,
            bottom_right_y=330,
        )
        [region] = classify_page([block], A4, page_number=2)
        assert region.kind == "aside"

    def test_a_dimensionless_aside_type_is_read_as_body_flagged(self) -> None:
        """No geometry to corroborate the type, so an aside_text block is read as
        body and flagged rather than removed on the type alone."""
        block = OcrBlock(
            type="aside_text",
            content="تعاريف",
            top_left_x=590,
            top_left_y=300,
            bottom_right_x=700,
            bottom_right_y=330,
        )
        [region] = classify_page([block], None, page_number=2)
        assert region.kind == "body"
        assert region.flagged

    def test_a_full_width_aside_type_is_read_as_body_flagged(self) -> None:
        """Band and type must agree: an `aside_text` block spanning the text
        measure is a mis-typed provision, read as body and flagged."""
        block = _b("aside_text", 300, 360, "This spans the whole column like a provision.")
        [region] = classify_page([block], A4, page_number=2)
        assert region.kind == "body"
        assert region.flagged

    def test_an_unknown_future_type_is_flagged_rather_than_assumed_body(self) -> None:
        block = _b("chart", 300, 400, "something new")
        [region] = classify_page([block], A4, page_number=2)
        assert region.kind == "unknown"
        assert region.flagged


class TestGeometryIsOptional:
    def test_a_page_with_no_dimensions_still_classifies_on_type_and_marker(self) -> None:
        """Losing the band costs one signal, not the whole page."""
        blocks = [_b("references", 900, 921, "¹ عدلت بموجب المادة (٧)")]
        [region] = classify_page(blocks, None, page_number=6)
        assert region.kind == "footnote"
        assert "band_bottom" not in region.signals

    def test_default_geometry_is_not_read_as_the_page_top(self) -> None:
        """Absent geometry defaults to 0 and must not read as the page top."""
        block = OcrBlock(type="references", content="no marker here")
        [region] = classify_page([block], A4, page_number=6)
        assert "band_bottom" not in region.signals
        assert region.flagged


def test_signals_are_recorded_even_when_they_did_not_decide() -> None:
    """A reviewer has to see what agreed, not only what won."""
    block = _b("references", 900, 921, "¹ عدلت بموجب المادة (٧)")
    [region] = classify_page([block], PageDimensions(dpi=87, width=720, height=1023), page_number=1)
    assert set(region.signals) >= {"type_references", "band_bottom", "marker_footnote"}


def test_the_marker_patterns_are_anchored_at_the_start() -> None:
    """A citation mid-sentence is not a footnote marker."""
    from codify.pipeline.enrich.regions import DEFAULT_FOOTNOTE_MARKERS

    assert not any(re.search(p, "نص عادي (١) داخل الجملة") for p in DEFAULT_FOOTNOTE_MARKERS)


class TestEraVocabulary:
    """`صدر بمدينة` is modern-era; applying it to older text would be a guess."""

    def _config(self):
        from codify.jurisdictions import load_config

        return load_config("xz")

    def test_a_modern_era_gets_the_declared_closing_phrase(self) -> None:
        assert "صدر بمدينة" in vocabulary_for(self._config(), "assembly").closing_phrases

    def test_an_older_era_gets_none_until_one_is_measured(self) -> None:
        assert vocabulary_for(self._config(), "sultanate").closing_phrases == ()

    def test_an_unresolved_era_gets_none_rather_than_the_modern_set(self) -> None:
        for sentinel in ("unknown", "no_eras_declared", "outside_declared_eras"):
            assert vocabulary_for(self._config(), sentinel).closing_phrases == ()

    def test_the_documented_marker_forms_survive_a_config_that_adds_its_own(self) -> None:
        from codify.jurisdictions import JurisdictionConfig

        cfg = JurisdictionConfig.model_validate(
            {
                "code": "zz",
                "name": "Test",
                "tradition": ["civil_law"],
                "languages": ["eng"],
                "footnote_markers": [r"^\*+"],
            }
        )
        markers = vocabulary_for(cfg, "unknown").footnote_markers
        assert len(markers) == len(DEFAULT_FOOTNOTE_MARKERS) + 1
        assert any(p.search("¹ note") for p in markers)
        assert any(p.search("** note") for p in markers)

    def test_no_config_is_no_phrases_rather_than_a_crash(self) -> None:
        assert vocabulary_for(None, "assembly") == RegionVocabulary()


def test_the_band_and_a_marker_decide_without_the_type() -> None:
    """The measured shape: a footnote the endpoint typed `text`, low on the page.
    Without this the band constant is never load-bearing."""
    block = _b("text", 900, 921, "¹ عدلت بموجب المادة (٧)")
    [region] = classify_page([block], PageDimensions(dpi=87, width=720, height=1023), page_number=4)
    assert region.kind == "footnote"
    assert set(region.signals) >= {"band_bottom", "marker_footnote"}


def test_the_same_block_high_on_the_page_is_body() -> None:
    """One signal is not a decision, so the band has to be doing real work."""
    block = _b("text", 200, 240, "¹ عدلت بموجب المادة (٧)")
    [region] = classify_page([block], PageDimensions(dpi=87, width=720, height=1023), page_number=4)
    assert region.kind == "body"
