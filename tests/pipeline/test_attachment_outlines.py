"""Outline anchoring for attachment content that numbers itself without keywords."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codify.jurisdictions import AttachmentCaption, load_config
from codify.pipeline.enrich.anchors import (
    _rank_map_for,
    build_anchor_regex,
    scan_anchors_with_ambiguity,
)

# Independently authored toy observatory instructions, not a legal excerpt.
DOC = """CHAPTER I
SAMPLE INSTRUCTIONS

Article 1
The toy observatory inventories:
1. blue lenses;
2. copper dials.

ANNEX IV
SAMPLE INVENTORY

CHAPTER I
EQUIPMENT

A. Lenses
Keep each sample lens in a blue box.

B. Dials
1. A sample dial has two painted hands.
2. The sample register records:
1) the box colour;
2) the shelf number.

CHAPTER II
STORAGE

A. Shelves
1. Store the sample box on a dry shelf.
"""


@pytest.fixture(autouse=True)
def synthetic_outline_config(monkeypatch):
    """Supply toy grammars without a dependency on private jurisdiction data."""
    from codify import jurisdictions
    from codify.pipeline.enrich import anchors

    original = jurisdictions.load_config
    raw = original("xa").model_dump()
    # The toy classes below replace xa's; its routing rules would name missing ones.
    raw["structuring"]["classification_rules"] = []

    def level(term, element, kind, numbering, **extra):
        return dict(
            local_term=term,
            akn_element=element,
            level=kind,
            numbering=numbering,
            **extra,
        )

    chapter = level("Chapter", "chapter", "higher", "roman")
    article = level("Article", "article", "basic", "arabic_continuous")
    paragraph = level(
        "Paragraph",
        "paragraph",
        "subdivision",
        "arabic_parenthesized",
        marker_form="parenthesized_arabic",
    )
    point = level("Point", "point", "subdivision", "alpha_lower_period")
    body = [chapter, article, paragraph, point]

    def document(hierarchy, basic="article"):
        return dict(
            label="Synthetic instrument",
            akn_element="act",
            basic_unit=basic,
            hierarchy=hierarchy,
        )

    raw["document_classes"] = {
        "act": document(body),
        "regulation": document(body),
        "constitution": document([chapter, article, paragraph]),
        "summary": document(
            [level("Paragraph", "paragraph", "basic", "arabic_period")], "paragraph"
        ),
        "judgment": document(
            [
                level("Section", "section", "higher", "arabic"),
                level("Paragraph", "paragraph", "basic", "arabic_period"),
            ],
            "paragraph",
        ),
    }
    raw["attachments"] = [
        dict(caption="EXPLANATION", normative=False),
        dict(
            caption="ANNEX",
            normative=True,
            hierarchy=[
                level(
                    "Chapter",
                    "chapter",
                    "grouping",
                    "roman",
                    marker_form="upper_roman_period",
                ),
                level(
                    "Letter",
                    "section",
                    "higher",
                    "alpha_upper",
                    marker_form="upper_letter_period",
                ),
                level(
                    "Number",
                    "paragraph",
                    "basic",
                    "arabic_period",
                    marker_form="arabic_period",
                ),
                level(
                    "Point",
                    "point",
                    "subdivision",
                    "arabic_closing_paren",
                    marker_form="arabic_closing_paren",
                ),
            ],
        ),
    ]
    config = jurisdictions.JurisdictionConfig.model_validate(raw)

    def load(country):
        if country == "xa":
            return config
        if country == "xb":
            return original("xa")
        return original(country)

    monkeypatch.setattr(jurisdictions, "load_config", load)
    monkeypatch.setattr(anchors, "load_config", load)
    monkeypatch.setitem(globals(), "load_config", load)
    caches = [value for value in vars(anchors).values() if hasattr(value, "cache_clear")]
    for value in caches:
        value.cache_clear()
    yield
    for value in caches:
        value.cache_clear()


def _scan(text: str, doctype: str):
    config = load_config("xa")
    return scan_anchors_with_ambiguity(
        text, build_anchor_regex(config, doctype), country="xa", doctype=doctype
    )


def test_annex_outline_levels_anchor_and_nest() -> None:
    eids = [
        a.akn_eid for a in _scan(DOC, "regulation").anchors if (a.akn_eid or "").startswith("sch")
    ]
    assert "schedule_1__chp_I__sec_A" in eids
    assert "schedule_1__chp_I__sec_B__para_1" in eids
    assert "schedule_1__chp_I__sec_B__para_2__point_1" in eids
    assert "schedule_1__chp_II__sec_A__para_1" in eids


def test_body_list_items_are_not_drawn_into_the_outline() -> None:
    # "1. penyusunan;" sits in the body under Article 1 and is ordinary text.
    body = [
        a.akn_eid
        for a in _scan(DOC, "regulation").anchors
        if not (a.akn_eid or "").startswith("sch")
    ]
    assert body == ["chp_I", "chp_I__art_1"]


def test_outline_scan_reports_its_own_fire_count() -> None:
    scan = _scan(DOC, "regulation")
    assert scan.fires["scan_attachment_outlines"] > 0
    assert {a.source_pass for a in scan.anchors if a.kind in {"section", "point"}} == {
        "attachment_outline"
    }


def test_attachment_only_level_gets_a_rank_between_its_neighbours() -> None:
    # `regulation` declares no section in its body, so without this an annex
    # section ranks below point and cannot scope the paragraphs under it.
    ranks = _rank_map_for("xa", "regulation")
    assert ranks is not None
    assert ranks["article"] < ranks["section"] < ranks["paragraph"]


def test_jurisdictions_without_an_annex_hierarchy_are_unchanged() -> None:
    # No annex level to splice in, so every rank stays a whole number: the
    # fractional ranks above are the only thing that inserts between two.
    ranks = _rank_map_for("xb", "act")
    assert ranks is not None
    assert sorted(ranks.values()) == [float(i) for i in range(len(ranks))]


def test_an_attachment_declares_no_hierarchy_by_default() -> None:
    assert AttachmentCaption(caption="EXPLANATION").hierarchy == []


def test_caption_level_inside_an_attachment_still_needs_captions() -> None:
    with pytest.raises(ValidationError, match="declares no captions"):
        AttachmentCaption(
            caption="ANNEX",
            hierarchy=[
                {
                    "local_term": "Chapter",
                    "akn_element": "chapter",
                    "level": "grouping",
                    "marker_form": "caption",
                }
            ],
        )


# A caption in the declared colon form that opens a body table, not an annex.
# `_drop_embedded_schedule_captions` rejects it on numbering continuity.
EMBEDDED_CAPTION = """CHAPTER I
KETENTUAN UMUM

Article 1
Ketentuan pertama.

ANNEX: Daftar Istilah

A. Istilah umum
1. Pertama.
2. Kedua.

Article 2
Ketentuan kedua.
1. Rincian.

Article 3
Ketentuan ketiga.
"""

# EXPLANATION declares no hierarchy, and its header block names ANNEX.
EXPLANATION_NAMING_ANNEX = """CHAPTER I
KETENTUAN UMUM

Article 1
Dalam.

EXPLANATION
ATAS
SYNTHETIC OBSERVATORY NOTES
ABOUT SAMPLE STORAGE AND THE ANNEX

I. UMUM
A. Latar belakang.
1. Butir pertama.
"""


def test_a_body_table_caption_does_not_open_an_outline_span() -> None:
    # Scanning past a rejected caption would pull every later article inside it.
    scan = _scan(EMBEDDED_CAPTION, "regulation")
    assert [a.akn_eid for a in scan.anchors] == [
        "chp_I",
        "chp_I__art_1",
        "chp_I__art_2",
        "chp_I__art_3",
    ]
    assert "scan_attachment_outlines" not in scan.fires


def test_an_attachment_takes_only_its_own_declared_grammar() -> None:
    # "ANNEX" appears in the Penjelasan's header block; it must not lend it
    # the Lampiran outline levels.
    eids = [a.akn_eid for a in _scan(EXPLANATION_NAMING_ANNEX, "regulation").anchors]
    assert eids == ["chp_I", "chp_I__art_1", "schedule_1"]


def test_consecutive_attachment_only_levels_never_tie() -> None:
    # Equal ranks stop one level scoping the next, which is the whole point.
    for doctype in ("regulation", "act", "summary", "judgment", "constitution"):
        ranks = _rank_map_for("xa", doctype)
        assert ranks is not None
        assert len(set(ranks.values())) == len(ranks), doctype
        # The schedule sentinel outranks everything declared.
        assert min(ranks.values()) > -1.0, doctype


def test_an_attachment_level_sits_directly_above_its_successor() -> None:
    regulation = _rank_map_for("xa", "regulation")
    assert regulation is not None
    assert regulation["article"] < regulation["section"] < regulation["paragraph"]
    act = _rank_map_for("xa", "act")
    assert act is not None
    assert act["article"] < act["section"] < act["paragraph"]


# The annex's own CHAPTER, in a class whose body hierarchy has no chapter level.
KEYWORDED_ANNEX_LEVEL = """Paragraph 1
Isi pertama.

ANNEX I
PEDOMAN

CHAPTER I
PENDAHULUAN

A. Latar Belakang
1. Uraian.
"""


def test_a_keyworded_annex_level_anchors_even_when_the_body_lacks_it() -> None:
    # `summary` declares only paragraph, so the body regex carries no
    # chapter keyword and the annex's CHAPTER would otherwise be lost.
    eids = [a.akn_eid for a in _scan(KEYWORDED_ANNEX_LEVEL, "summary").anchors]
    assert "schedule_1__chp_I" in eids
    assert "schedule_1__chp_I__sec_A__para_1" in eids


def test_a_keyworded_annex_level_is_not_anchored_twice() -> None:
    # `regulation` does declare chapter, so the body regex already claimed CHAPTER I.
    chapters = [
        a.akn_eid for a in _scan(KEYWORDED_ANNEX_LEVEL, "regulation").anchors if a.kind == "chapter"
    ]
    assert chapters == ["schedule_1__chp_I"]


def test_an_attachment_hierarchy_obeys_the_level_contract() -> None:
    from codify.jurisdictions import JurisdictionConfig

    base = {
        "code": "zz",
        "name": "T",
        "type": "national",
        "tradition": ["civil_law"],
        "calendar": "gregorian",
        "languages": ["eng"],
        "authoritative_language": "eng",
        "default_document_class": "act",
        "document_classes": {
            "act": {
                "label": "A",
                "akn_element": "act",
                "basic_unit": "article",
                "hierarchy": [{"local_term": "Art", "akn_element": "article", "level": "basic"}],
            }
        },
    }
    with pytest.raises(ValidationError, match="attachment 'ANNEX'"):
        JurisdictionConfig.model_validate(
            base
            | {
                "attachments": [
                    {
                        "caption": "ANNEX",
                        "hierarchy": [
                            {
                                "local_term": "One",
                                "akn_element": "paragraph",
                                "level": "basic",
                            },
                            {
                                "local_term": "Two",
                                "akn_element": "point",
                                "level": "basic",
                            },
                        ],
                    }
                ]
            }
        )


# One annex heading over two numbered runs, which is how a Lampiran lays out
# two tables under a single caption. There is no container to separate them.
RESTARTED_ANNEX_LIST = """CHAPTER I
KETENTUAN UMUM

Article 1
Dalam.

ANNEX I
PEDOMAN

CHAPTER I
PENDAHULUAN

A. Pengertian
1. Istilah pertama.
2. Istilah kedua.
1. Istilah ketiga.
2. Istilah keempat.
"""


def test_a_restarted_annex_list_does_not_block_structuring() -> None:
    # Two "1." siblings would not round-trip, and duplicate_number is a
    # blocking invariant, so the whole document used to fail.
    scan = _scan(RESTARTED_ANNEX_LIST, "regulation")
    blocking = [a for a in scan.ambiguity if a.kind == "duplicate_number" and not a.resolved]
    assert blocking == []


def test_the_restarted_run_falls_back_to_body_text() -> None:
    eids = [a.akn_eid for a in _scan(RESTARTED_ANNEX_LIST, "regulation").anchors]
    # The first run anchors; the repeat is presentation and is left to body-fill.
    assert "schedule_1__chp_I__sec_A__para_1" in eids
    assert "schedule_1__chp_I__sec_A__para_1_2" not in eids
    assert sum(1 for e in eids if e.endswith("__para_1")) == 1


def test_a_dropped_restarted_container_takes_its_subtree_with_it() -> None:
    # Exercised directly: the pipeline dedups this shape upstream, but the eId
    # walk must not re-parent a dropped section's children onto the chapter.
    from codify.pipeline.enrich.anchors import (
        StructuralAnchor,
        _assign_eids,
        _rank_map_for,
    )

    def anchor(kind: str, num: str, off: int) -> StructuralAnchor:
        return StructuralAnchor(
            kind=kind,
            keyword=num,
            number=num,
            char_offset=off,
            line=off,
            matched_text=num,
            source_pass="attachment_outline",
        )

    anchors = [
        anchor("chapter", "I", 0),
        anchor("section", "A", 10),
        anchor("paragraph", "1", 20),
        anchor("section", "A", 30),  # the run restarts
        anchor("paragraph", "1", 40),
    ]
    spans: list = []
    out = _assign_eids(anchors, _rank_map_for("xa", "regulation"), spans=spans)
    assert [a.akn_eid for a in out] == ["chp_I", "chp_I__sec_A", "chp_I__sec_A__para_1"]
    # Both the repeated section and its child are recorded, not silently lost.
    restarted = [s for s in spans if (s.detail or {}).get("reads_as") == "restarted_annex_list"]
    assert [s.detail["kind"] for s in restarted] == ["section", "paragraph"]


def _synthetic_outline(name: str) -> str:
    excerpt = (Path(__file__).parent / "fixtures" / name).read_text()
    return "CHAPTER I\nKETENTUAN UMUM\nArticle 1\nKetentuan berlaku.\n\nANNEX IV\n" + excerpt


def test_synthetic_roman_outline_preserves_both_letter_sequences() -> None:
    scan = _scan(_synthetic_outline("ministerial_annex_roman.txt"), "regulation")
    eids = {a.akn_eid for a in scan.anchors}
    assert {
        "schedule_1__chp_I__sec_A",
        "schedule_1__chp_I__sec_B",
        "schedule_1__chp_I__sec_C",
        "schedule_1__chp_II__sec_A__para_1",
        "schedule_1__chp_II__sec_B",
        "schedule_1__chp_III",
    } <= eids
    assert "schedule_1__sec_I" not in eids


def test_synthetic_h_to_i_outline_remains_lettered() -> None:
    scan = _scan(_synthetic_outline("ministerial_annex_letters.txt"), "regulation")
    eids = {a.akn_eid for a in scan.anchors}
    assert {"schedule_1__sec_H", "schedule_1__sec_I"} <= eids
    assert "schedule_1__chp_I" not in eids


def test_letter_i_cannot_borrow_roman_evidence_after_another_i() -> None:
    text = _synthetic_outline("ministerial_annex_letters.txt")
    text += "\n" + (Path(__file__).parent / "fixtures" / "ministerial_annex_roman.txt").read_text()
    eids = {a.akn_eid for a in _scan(text, "regulation").anchors}
    assert "schedule_1__sec_I" in eids
    assert "schedule_1__chp_I__sec_A" in eids


def test_isolated_i_without_a_roman_sequence_remains_ambiguous() -> None:
    text = _synthetic_outline("ministerial_annex_roman.txt").split("II. METHOD")[0]
    eids = {a.akn_eid for a in _scan(text, "regulation").anchors}
    assert "schedule_1__sec_I" in eids
    assert "schedule_1__chp_I" not in eids


def test_synthetic_letter_i_does_not_borrow_a_later_ii() -> None:
    text = _synthetic_outline("ministerial_annex_letters.txt")
    roman = (Path(__file__).parent / "fixtures" / "ministerial_annex_roman.txt").read_text()
    text += "\nII. METHOD" + roman.split("II. METHOD", 1)[1]
    eids = {a.akn_eid for a in _scan(text, "regulation").anchors}
    assert "schedule_1__sec_I" in eids
    assert "schedule_1__chp_I" not in eids


def test_roman_sequence_cannot_cross_a_keyworded_chapter() -> None:
    text = _synthetic_outline("ministerial_annex_roman.txt").replace(
        "II. METHOD", "CHAPTER VII\nMETODE\nII. METHOD"
    )
    eids = {a.akn_eid for a in _scan(text, "regulation").anchors}
    assert "schedule_1__sec_I" in eids
    assert "schedule_1__chp_I" not in eids
