"""Unit tests for the structural-anchor probe (Phase A)."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest

from codify import jurisdictions as jurisdictions_module
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.enrich import anchors as anchors_module
from codify.pipeline.enrich.anchors import (
    StructuralAnchor,
    _roman_or_digit,
    _sequence_survivor,
    anchor_summary,
    build_anchor_regex,
    markers_outside_boundary,
    scan_anchors,
    scan_anchors_with_ambiguity,
)

# Arabic statute hierarchy, inline so these tests need no corpus config.
_QANUN = {
    "code": "xx",
    "name": "Fixture",
    "languages": ["ara"],
    "tradition": ["civil_law"],
    "authoritative_language": "ara",
    "default_document_class": "qanun",
    "structuring": {},
    "amendments": {
        "style": "textual_civil_law",
        "trigger_phrases": [
            "تعدل المادة",
            "يعدل نص المادة",
            "المادة التالية",
            "المادة الجديدة التالية",
            "بالمادة التالية",
            "المواد التالية",
            "بالمواد التالية",
            "النص التالي",
            "بالنص التالي",
            "بما يلي",
            "كما يلي",
            "الآتي",
        ],
    },
    "document_classes": {
        "qanun": {
            "label": "Statute",
            "akn_element": "act",
            "basic_unit": "article",
            "bluebell_compatible": True,
            "hierarchy": [
                {
                    "local_term": "باب",
                    "akn_element": "part",
                    "level": "higher",
                    "bluebell_keyword": "PART",
                    "numbering": "arabic_ordinal",
                },
                {
                    "local_term": "فصل",
                    "akn_element": "chapter",
                    "level": "higher",
                    "bluebell_keyword": "CHAPTER",
                    "numbering": "arabic_ordinal",
                },
                {
                    "local_term": "فرع",
                    "akn_element": "section",
                    "level": "higher",
                    "bluebell_keyword": "SECTION",
                    "numbering": "arabic_ordinal",
                },
                {
                    "local_term": "قسم",
                    "akn_element": "subdivision",
                    "level": "higher",
                    "bluebell_keyword": "SUBDIVISION",
                    "numbering": "arabic_ordinal",
                },
                {
                    "local_term": "مادة",
                    "akn_element": "article",
                    "level": "basic",
                    "bluebell_keyword": "ARTICLE",
                    "numbering": "arabic_continuous",
                },
                {
                    "local_term": "فقرة",
                    "akn_element": "paragraph",
                    "level": "subdivision",
                    "bluebell_keyword": "PARAGRAPH",
                    "numbering": "positional_implicit",
                },
                {
                    "local_term": "بند",
                    "akn_element": "point",
                    "level": "subdivision",
                    "bluebell_keyword": "POINT",
                    "numbering": "arabic_or_alpha",
                },
            ],
        }
    },
}
# The tests ask for this hierarchy under both names.
_QANUN["document_classes"]["act"] = _QANUN["document_classes"]["qanun"]
_QANUN_CONFIG = JurisdictionConfig.model_validate(_QANUN)


def _clear_config_caches() -> None:
    # Only this module's own caches: `ordinal_word_folds` is imported, does not
    # depend on the patched loader, and rereads every config when cleared.
    for fn in vars(anchors_module).values():
        if hasattr(fn, "cache_clear") and getattr(fn, "__module__", "") == anchors_module.__name__:
            fn.cache_clear()


@pytest.fixture(autouse=True)
def _fixture_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Both modules: some helpers import load_config inside the call, so the
    # name bound here is not the one they reach.
    real = jurisdictions_module.load_config
    stub = lambda c: _QANUN_CONFIG if c == "xx" else real(c)  # noqa: E731
    monkeypatch.setattr(jurisdictions_module, "load_config", stub)
    monkeypatch.setattr(anchors_module, "load_config", stub)
    monkeypatch.setitem(globals(), "load_config", stub)
    # Those helpers are lru_cached, so a warm entry outlives the patch.
    _clear_config_caches()
    yield
    # monkeypatch restores the loader; the caches would keep fixture entries.
    _clear_config_caches()


# ── Albanian Ligj ───────────────────────────────────────────────────────────


ALBANIAN_SAMPLE = """\
Pjesa I
Dispozita të përgjithshme

Kreu I — Hyrje

Neni 1
Objekti i ligjit
Ky ligj ka për qëllim të rregullojë veprimtarinë e organizatave jofitimprurëse.

Neni 2
Përkufizimet
Në kuptim të këtij ligji termat e mëposhtëm kanë këto kuptime:
  (a) "organizatë jofitimprurëse" — sipas nenit 4;
  (b) "anëtar" — personi fizik që ushtron të drejtat.

Kreu II — Themelimi

Neni 3
Autoriteti përgjegjës
Ministria përgjegjëse për regjistrimin është Ministria e Drejtësisë.

Neni 4
Procedura e regjistrimit
Procedura e regjistrimit ndjek hapat e mëposhtëm.
"""


def test_albanian_anchors_pick_up_neni_kreu_pjesa() -> None:
    config = load_config("al")
    assert config is not None, "Albanian config must be present for this test."
    regex = build_anchor_regex(config, "act")

    anchors = scan_anchors(ALBANIAN_SAMPLE, regex)

    summary = anchor_summary(anchors)
    assert summary == {"part": 1, "chapter": 2, "article": 4}, summary

    # Numbers should be preserved from <num> position in the source.
    article_numbers = [a.number for a in anchors if a.kind == "article"]
    assert article_numbers == ["1", "2", "3", "4"]

    chapter_numbers = [a.number for a in anchors if a.kind == "chapter"]
    assert chapter_numbers == ["I", "II"]


# ── English Act (regression) ────────────────────────────────────────────────


ENGLISH_SAMPLE = """\
PART I
Preliminary

CHAPTER 1 — Definitions and scope

Section 1. Short title.
This Act may be cited as the Test Act.

Section 2. Interpretation.
In this Act, unless the context otherwise requires —
  (a) "Minister" means the Minister responsible;
  (b) "officer" means an authorised officer.

PART II
Administration

Section 3. Establishment.
There is hereby established a Board.
"""


def test_english_anchors_still_match_under_fallback_regex() -> None:
    # Pass `config=None` so the builder falls back to the built-in alias
    # union. This guards the no-config path.
    regex = build_anchor_regex(None, "act")
    anchors = scan_anchors(ENGLISH_SAMPLE, regex)

    summary = anchor_summary(anchors)
    assert summary.get("part") == 2
    assert summary.get("chapter") == 1
    assert summary.get("section") == 3


# ── UK keyword-less sections ────────────────────────────────────────────────


UK_SAMPLE = """\
PART 1 Preliminary

1 Overview
(1) This Act makes provision about data protection.
(2) It supplements the UK GDPR.

2 Terms relating to processing
(1) In this Act, "data" means information —
  (a) point one;
  (b) point two.

13A Special category data
(1) This section applies to special category data.
"""


def test_uk_bare_number_sections_and_subsections() -> None:
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(UK_SAMPLE, regex, country="gb")

    summary = anchor_summary(anchors)
    assert summary.get("part") == 1
    assert summary.get("section") == 3  # "1", "2", "13A", no "Section" keyword
    assert summary.get("subsection") == 4
    assert [a.number for a in anchors if a.kind == "section"] == ["1", "2", "13A"]
    # "(a)"/"(b)" are points, not subsections, must not be picked up as subsections.
    assert not any(a.kind == "subsection" and a.number in {"a", "b"} for a in anchors)


def test_uk_scan_is_gated_to_uk_jurisdictions() -> None:
    """Without a UK country the keyword-less sections/subsections stay invisible,
    so no other jurisdiction is affected."""
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(UK_SAMPLE, regex)  # no country

    summary = anchor_summary(anchors)
    assert summary.get("section") is None
    assert summary.get("subsection") is None
    assert summary.get("part") == 1  # the keyworded PART still matches


UK_TOC_SAMPLE = """\
Arrangement of Sections

1 Overview
2 Terms relating to processing
3 Special category data

PART 1 Preliminary

1 Overview
(1) This Act makes provision about data protection.
(2) It supplements the UK GDPR.

2 Terms relating to processing
(1) In this Act, "data" means information.

3 Special category data
(1) This section applies to special category data.
"""


def test_uk_arrangement_of_sections_toc_is_dropped() -> None:
    """The contents block repeats section headings; only the body sections (the
    ones followed by their subsections) should survive."""
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(UK_TOC_SAMPLE, regex, country="gb")

    sections = [a for a in anchors if a.kind == "section"]
    assert [a.number for a in sections] == ["1", "2", "3"]  # not 6
    # The surviving sections are the body ones (after "PART 1"), each with a subsection.
    part_offset = next(a.char_offset for a in anchors if a.kind == "part")
    assert all(a.char_offset > part_offset for a in sections)


def test_uk_section_ignores_dates_and_sentence_lines() -> None:
    text = (
        "PART 1 Preliminary\n\n"
        "1 Overview\n(1) This Act makes provision.\n\n"
        "1 January 2018 is the appointed day for this purpose.\n\n"
        "5 Things are needed for this purpose to be achieved here.\n"
    )
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(text, regex, country="gb")

    sections = [a for a in anchors if a.kind == "section"]
    assert [a.number for a in sections] == ["1"]  # date + trailing-period line skipped


def test_uk_section_ignores_year_led_prose_lines() -> None:
    """A prose line starting with a bare year must not become section 2015."""
    text = (
        "PART 1 Preliminary\n\n"
        "1 Overview\n(1) This Act makes provision.\n\n"
        "2015 The Money Laundering Regulations were made under this power\n"
    )
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(text, regex, country="gb")

    assert [a.number for a in anchors if a.kind == "section"] == ["1"]


def test_uk_in_prose_part_and_chapter_citations_dropped() -> None:
    """Mid-sentence and sentence-line "Part N"/"Chapter N" are citations, not
    structure; real line-anchored unpunctuated headings survive."""
    text = (
        "PART 1 Introduction\n\n"
        "1 Overview\n"
        "(1) This is made under Part 2 of these Regulations.\n"
        "(2) Part 3 makes provision about supervision.\n\n"
        "PART 2 Money laundering\n\n"
        "2 Interpretation\n(1) In these Regulations, terms have their meaning.\n"
    )
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(text, regex, country="gb")

    assert [a.number for a in anchors if a.kind == "part"] == ["1", "2"]


# ── TOC skip ────────────────────────────────────────────────────────────────


def test_anchors_respect_toc_end() -> None:
    text = (
        "Section 1. TOC entry one.\n"
        "Section 2. TOC entry two.\n\n"
        "Real body starts here.\n\n"
        "Section 1. Real title.\nBody."
    )
    regex = build_anchor_regex(None, "act")
    # Skip past the fake TOC.
    anchors = scan_anchors(text, regex, toc_end=60)
    assert all(a.char_offset >= 60 for a in anchors)
    assert len(anchors) == 1


# ── Regex shape ────────────────────────────────────────────────────────────


def test_builder_uses_named_groups_for_every_kind() -> None:
    config = load_config("al")
    regex = build_anchor_regex(config, "act")
    # The compiled pattern should contain one named group per AKN element.
    expected_kinds = {"part", "title", "chapter", "section", "article", "paragraph"}
    assert isinstance(regex, re.Pattern)
    for kind in expected_kinds:
        assert f"(?P<k_{kind}>" in regex.pattern, f"missing group for kind {kind!r}"


# ── Parenthesised abbreviations in local_term ───────────────────────────────


BG_SAMPLE = """\
Глава 1
Общи положения

Чл. 1. (1) Този кодекс урежда трудовите отношения.
(2) Други разпоредби.

Чл. 2. Работодателят е длъжен да...

Чл. 3. Работникът има право...

Член 4. Допълнителни положения.

Глава 2
Трудов договор

Чл. 5. Трудовият договор се сключва писмено.
"""


def test_bulgarian_abbreviated_chlen_matches() -> None:
    """Bulgaria's config has ``local_term: 'Член (Чл.)'``, the in-text
    article anchor is most commonly ``Чл.``. The probe must catch both
    forms because they refer to the same hierarchy entry."""
    config = load_config("bg")
    assert config is not None
    regex = build_anchor_regex(config, "code")
    anchors = scan_anchors(BG_SAMPLE, regex)
    summary = anchor_summary(anchors)
    # Four "Чл." occurrences + one full "Член" + two "Глава" occurrences.
    assert summary.get("article") == 5, summary
    assert summary.get("chapter") == 2, summary


def test_split_term_with_parens_drops_english_gloss() -> None:
    """Georgian and Greek configs store the English gloss inside parens
    (``'მუხლი (Article)'``). The probe must keep only the Georgian/Greek
    head, the English gloss would never appear in raw text."""
    from codify.pipeline.enrich.anchors import _split_term_with_parens

    assert _split_term_with_parens("მუხლი (Article)") == ["მუხლი"]
    assert _split_term_with_parens("Παράρτημα (Annex/Schedule)") == ["Παράρτημα"]
    assert _split_term_with_parens("Член (Чл.)") == ["Член", "Чл."]
    assert _split_term_with_parens("Articolul") == ["Articolul"]
    # Mixed forms with both transliteration and English, keep only the
    # head (Arabic), drop the slashed pieces.
    assert _split_term_with_parens("باب (Bab/Part)") == ["باب"]


ABBREVIATED_ACT_SAMPLE = """\
CHAPTER I
GENERAL PROVISIONS

SECTION 1. Short Title. - This Act shall be known as the Test Act.

SEC. 2. Declaration of Policy. - The State shall do the thing.

SEC. 3. Definition of Terms. - As used in this Act:

SEC. 4. Coverage. - This Act shall apply to all agencies.
"""


ABBREVIATED_CODE_SAMPLE = """\
BOOK I
PERSONS

TITLE I
CIVIL PERSONALITY

ARTICLE 37. Juridical capacity is inherent in every natural person.

ART. 38. Minority restricts capacity to act.

ART. 39. The following circumstances modify capacity to act.
"""


def test_abbreviated_section_matches() -> None:
    """Some publishers write the first marker in full and every marker after it
    as ``SEC.``, so a config carrying only ``Section`` scans one section out of
    twenty. Both forms are the same hierarchy entry."""
    config = load_config("xa")
    assert config is not None
    regex = build_anchor_regex(config, "act")
    summary = anchor_summary(scan_anchors(ABBREVIATED_ACT_SAMPLE, regex))
    # One full "SECTION" + three "SEC." + one "CHAPTER".
    assert summary.get("section") == 4, summary
    assert summary.get("chapter") == 1, summary


# The shape that reorders: a heading, and a later section whose body cites it
# by number.
CITED_HEADING_SAMPLE = (
    "SEC. 12. Possession of Equipment. — The penalty of imprisonment shall be "
    "imposed upon any person who shall possess any equipment.\n\n"
    "SEC. 14. Possession During Parties. — The maximum penalty provided for in "
    "Section 12 of this Act shall be imposed upon any person, who shall possess "
    "any equipment during a social gathering.\n\n"
    "SEC. 15. Use of Dangerous Drugs. — A person apprehended shall be subject to "
    "the provisions of Section 15 of this Act.\n"
)


def test_a_citation_does_not_displace_the_heading_it_names() -> None:
    """Under `relaxed` a marker may follow any whitespace, so "provided for in
    Section 12 of this Act" scans as a heading. The dedup then prefers it:
    `section` is not a container, so the sequence survivor never runs and
    selection falls to keep-last, and a citation is always later than the
    heading it duplicates. The heading is dropped and the survivor carries an
    offset inside section 14, so a long act stored every section and still
    read `14, 12 … 32, 17 … 44, 5`.

    The relaxed boundary is the control: it returns ``['14', '12', '15']``."""
    relaxed = load_config("xa")
    assert relaxed is not None and relaxed.structuring is not None
    anchored = relaxed.model_copy(
        update={
            "structuring": relaxed.structuring.model_copy(
                update={"marker_boundary": "line_anchored"}
            )
        }
    )

    def numbers(config: JurisdictionConfig) -> list[str]:
        anchors = scan_anchors(CITED_HEADING_SAMPLE, build_anchor_regex(config, "act"))
        return [a.number for a in anchors if a.kind == "section"]

    assert numbers(relaxed) == ["14", "12", "15"]
    assert numbers(anchored) == ["12", "14", "15"]


def test_abbreviated_article_matches() -> None:
    """The same split in the civil-law codes, where the basic unit is the
    article and ``ART.`` carries the run."""
    config = load_config("xa")
    assert config is not None
    regex = build_anchor_regex(config, "code")
    summary = anchor_summary(scan_anchors(ABBREVIATED_CODE_SAMPLE, regex))
    # One full "ARTICLE" + two "ART.".
    assert summary.get("article") == 3, summary


# ── Arabic-Indic + depth + akn_eid ──


ARABIC_SAMPLE = """\
الفصل الأول
أحكام عامة

المادة ١
يسمى هذا القانون "قانون التجارة".

المادة ٢
أحكام تمهيدية.

المادة ٣
يطبق هذا القانون على جميع المعاملات.
"""


def test_arabic_indic_numerals_matched() -> None:
    """Older Arabic-script docs use Arabic-Indic numerals (١٢٣) in anchor
    positions. The probe must match these the same as Western Arabic."""
    config = _QANUN_CONFIG
    assert config is not None
    regex = build_anchor_regex(config, "qanun")
    anchors = scan_anchors(ARABIC_SAMPLE, regex)
    summary = anchor_summary(anchors)
    # Three مادة anchors with Arabic-Indic numbers + one الفصل.
    assert summary.get("article", 0) >= 3, summary


# مـادة with U+0640 TATWEEL between م and ا. Visually identical to مادة
# but the anchor regex sees a different byte sequence. Some Arabic gazettes
# use this variant on most article headers; without normalisation those articles never enter
# the AKN. The pre-structurer path calls ``normalise_rtl_extract`` which
# strips joiners, so the two variants become equivalent before scanning.
ARABIC_TATWEEL_SAMPLE = """\
الفصل الأول
أحكام عامة

مـادة (١)
تعريفات.

مـادة (٢)
أحكام تمهيدية.

مادة (٣)
يطبق هذا القانون على جميع المعاملات.
"""


def test_arabic_tatweel_variant_matched_after_normalisation() -> None:
    """``مـادة`` (with TATWEEL) and ``مادة`` (without) both anchor as
    articles once the pre-structurer strips joiner characters."""
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    config = _QANUN_CONFIG
    assert config is not None
    regex = build_anchor_regex(config, "qanun")
    normalised = normalise_rtl_extract(ARABIC_TATWEEL_SAMPLE)
    # Joiner is stripped so the mixed variants unify.
    assert "ـ" not in normalised
    anchors = scan_anchors(normalised, regex)
    summary = anchor_summary(anchors)
    # All three مـادة/مادة headers captured plus one الفصل.
    assert summary.get("article", 0) == 3, summary


def test_arabic_zwnj_and_zwj_joiners_stripped() -> None:
    """Zero-width joiners appear from word-processor round-trips and are
    invisible on the page. The normaliser drops U+200C and U+200D so the
    marker literal `مادة` matches whether or not joiners were inserted."""
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    variants = ["م‌ادة", "م‍ادة", "مـ‌ادة"]  # ZWNJ, ZWJ, tatweel+ZWNJ
    for form in variants:
        assert normalise_rtl_extract(form) == "مادة", form


def test_normalise_rtl_extract_strips_stray_paren_before_header() -> None:
    """RTL reorder prefixes real headers with a lone `)` on their own line
    (`)مادة 3`), defeating the anchor boundary. The normaliser strips it so the
    header is recognised, preserving the articles that follow it."""
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    text = "مادة (2)\nنص.\n)مادة (3)\nنص آخر.\n"
    normalised = normalise_rtl_extract(text)
    regex = build_anchor_regex(_QANUN_CONFIG, "qanun")
    eids = [a.akn_eid for a in scan_anchors(normalised, regex) if a.kind == "article"]
    assert eids == ["art_2", "art_3"]


def test_stray_paren_strip_only_targets_numbered_headers() -> None:
    """The strip needs a numbered مادة header after the `)`: an in-prose citation
    `(المادة 5)` and a `)` before a non-numbered word are left alone."""
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    text = "(المادة 5) من القانون.\n)أحكام عامة\n"
    assert normalise_rtl_extract(text) == text


def test_stray_paren_strip_stays_on_one_line() -> None:
    """A line ending in a lone `)` is not stripped just because the next line
    opens a numbered header: the strip is a same-line artifact only."""
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    text = "نهاية البند )\nمادة 3\nنص."
    assert normalise_rtl_extract(text) == text


def test_anchor_regex_matches_arabic_variants_without_prior_normalisation() -> None:
    """Belt-and-braces: even when a caller hands the regex a raw literal
    that still carries TATWEEL/ZWNJ/ZWJ (e.g. the coverage validator
    counting marker occurrences pre-normalisation), the anchor regex must
    match. `alias_regex_fragment` injects an optional-joiner class between
    adjacent Arabic letters at compile time."""
    config = _QANUN_CONFIG
    assert config is not None
    regex = build_anchor_regex(config, "qanun")
    # Each variant on its own line, no prior normalisation.
    samples = [
        "\nمادة (١)\n",  # bare
        "\nمـادة (٢)\n",  # tatweel
        "\nم‌ادة (٣)\n",  # ZWNJ
        "\nم‍ادة (٤)\n",  # ZWJ
        "\nمـ‌ادة (٥)\n",  # tatweel + ZWNJ
    ]
    for sample in samples:
        anchors = scan_anchors(sample, regex)
        assert anchors, f"no anchor for {sample!r}"
        assert any(a.kind == "article" for a in anchors), f"no article kind for {sample!r}"


def test_alias_regex_fragment_leaves_non_arabic_intact() -> None:
    """A Latin-script alias is unchanged (no joiner class inserted)."""
    from codify.pipeline.enrich.anchors import alias_regex_fragment

    assert alias_regex_fragment("Article") == re.escape("Article")
    assert alias_regex_fragment("Section") == re.escape("Section")
    # Regex-special chars in a non-Arabic alias are still escaped.
    assert alias_regex_fragment("Art.") == re.escape("Art.")


# ── Coverage-delta gate ───────────────────────────────────────────────────


def test_expected_markers_count_structural_position_only() -> None:
    """The counter mirrors ``build_anchor_regex``'s column-boundary shape:
    a marker at line start (or preceded by a tab / column gap) counts, an
    in-prose reference on the same line as surrounding words does not. If
    prose refs counted, the coverage-delta gate would fire false-positive
    on any law that quotes its own articles in bodies."""
    from codify.pipeline.enrich.anchors import anchor_coverage

    config = _QANUN_CONFIG
    assert config is not None
    text = (
        "مادة (١)\n"
        "تعريفات.\n"
        "مادة (٢)\n"
        "أحكام أولية.\n"
        "في المادة (١) من هذا القانون\n"  # in-prose, must NOT count
        "مـادة (٣)\n"  # tatweel variant, counts after regex tolerance
        "أحكام."
    )
    # Three structural headings; the in-prose "في المادة" is excluded.
    assert len(anchor_coverage(text, [], config, "qanun", "article").expected) == 3


def test_anchor_coverage_is_none_when_the_source_claims_no_markers() -> None:
    """None, not 1.0: a short order and a document whose markers went
    unrecognised both land here, and a perfect score would hide the second.
    Both still pass the gate."""
    from codify.pipeline.enrich.anchors import anchor_coverage

    config = _QANUN_CONFIG
    assert config is not None
    cov = anchor_coverage("just prose", [], config, "qanun", kind="article")
    assert cov.ratio is None
    assert not cov.captured
    assert not cov.expected


def test_anchor_coverage_flags_undershoot() -> None:
    """Simulate DL 8/2014 pre-fix: many marker occurrences, few captured."""
    from codify.pipeline.enrich.anchors import (
        StructuralAnchor,
        anchor_coverage,
    )

    config = _QANUN_CONFIG
    assert config is not None
    text = "\n".join(f"مادة ({i})\ntext" for i in range(1, 11))  # 10 markers
    # Pretend only 2 got captured.
    fake_anchors = [
        StructuralAnchor(
            kind="article",
            keyword="مادة",
            number=str(i),
            char_offset=0,
            line=i,
            matched_text=f"مادة ({i})",
        )
        for i in (1, 2)
    ]
    cov = anchor_coverage(text, fake_anchors, config, "qanun", kind="article")
    ratio, captured, expected = cov.ratio, len(cov.captured), len(cov.expected)
    assert expected == 10
    assert captured == 2
    assert ratio == 0.2


def test_assigned_depths_and_eids() -> None:
    """``scan_anchors`` populates ``depth`` / ``parent_eid`` / ``akn_eid``
    deterministically; the scaffolder relies on these for stable join keys."""
    config = load_config("al")
    assert config is not None
    regex = build_anchor_regex(config, "ligj")
    anchors = scan_anchors(ALBANIAN_SAMPLE, regex)
    # PART → CHAPTER → ARTICLE nests three deep.
    by_kind = {a.kind: a for a in anchors if a.kind in {"part", "chapter", "article"}}
    assert by_kind["part"].depth == 0
    assert by_kind["chapter"].depth == 1
    assert by_kind["chapter"].parent_eid == by_kind["part"].akn_eid
    article = next(a for a in anchors if a.kind == "article")
    assert article.depth == 2
    # akn_eid is non-empty and follows the parent__child convention.
    assert article.akn_eid
    assert article.parent_eid is not None
    assert article.akn_eid.startswith(article.parent_eid + "__")


def test_chapter_repeats_under_different_parts_keep_distinct_eids() -> None:
    """Two CHAPTER 1 anchors under different PARTs survive the TOC dedup
    (different ancestor chains) and get unique eIds via the parent-prefix
    join, the scaffolder's join key has to be unique."""
    config = load_config("al")
    assert config is not None
    regex = build_anchor_regex(config, "ligj")
    text = "Pjesa I\nKreu 1\nNeni 1\nx\nPjesa II\nKreu 1\nNeni 2\ny\n"
    anchors = scan_anchors(text, regex)
    chapters = [a for a in anchors if a.kind == "chapter"]
    assert len(chapters) == 2
    assert chapters[0].akn_eid != chapters[1].akn_eid


# ── TOC dedup ────────────────────────────────────────────────────────────────


GR_TOC_SAMPLE = """\
ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ

Άρθρο 1
Σκοπός

Άρθρο 2
Πεδίο εφαρμογής

Άρθρο 3
Ορισμοί

[the actual body of the law follows below, roughly 4x as much text]

ΚΕΦΑΛΑΙΟ Α'
Γενικές διατάξεις

Άρθρο 1
Σκοπός

Σκοπός του παρόντος νόμου είναι η ρύθμιση των θεμάτων που αφορούν την υγεία.
Οι διατάξεις του εφαρμόζονται σε όλα τα ιδρύματα. Συνεχίζονται οι κανονισμοί.
Καθιερώνονται νέοι μηχανισμοί ελέγχου. Διασφαλίζεται η ποιότητα των υπηρεσιών.

Άρθρο 2
Πεδίο εφαρμογής

Ο νόμος εφαρμόζεται σε δημόσια και ιδιωτικά νοσοκομεία.
Καλύπτονται όλες οι μονάδες πρωτοβάθμιας φροντίδας υγείας.
Περιλαμβάνονται τα διαγνωστικά κέντρα και τα φαρμακεία.

Άρθρο 3
Ορισμοί

Για τους σκοπούς του παρόντος νόμου, οι ακόλουθοι όροι ορίζονται ως εξής.
Νοσοκομείο σημαίνει κάθε εγκατάσταση παροχής υπηρεσιών υγείας.
Ασθενής σημαίνει κάθε πρόσωπο που λαμβάνει ιατρικές υπηρεσίες.
"""


def test_toc_duplicates_dropped() -> None:
    """When the same (kind, number) appears in the TOC region AND later
    in the body, the TOC occurrence is dropped and only the body anchor
    survives, eId disambiguation should not fire."""
    config = load_config("gr")
    assert config is not None
    regex = build_anchor_regex(config, "act")
    anchors = scan_anchors(GR_TOC_SAMPLE, regex)
    articles = [a for a in anchors if a.kind == "article"]
    # Three distinct articles, each appearing exactly once after dedup.
    assert len(articles) == 3, [a.akn_eid for a in articles]
    # Bare eIds, no _2 / _3 disambiguation suffix after the slash.
    eids = sorted(a.akn_eid for a in articles)
    assert eids == ["art_1", "art_2", "art_3"], eids
    # The surviving Article 1 is the body one, not the TOC one (TOC was
    # earlier in the document).
    art_1 = next(a for a in articles if a.akn_eid == "art_1")
    assert art_1.char_offset > 100, art_1.char_offset


def test_anchors_inside_quotes_are_dropped() -> None:
    """Amendment text quoting another law's article must not yield a
    host-document anchor."""
    config = load_config("ua")
    assert config is not None
    regex = build_anchor_regex(config, "act")
    text = (
        "Стаття 5\nОсновний текст статті.\n\n"
        "Стаття 6\nПрикінцеві положення.\n"
        '«У статті 18 Закону України "Про X" викласти в такій редакції».\n'
    )
    anchors = scan_anchors(text, regex)
    nums = sorted(a.number for a in anchors if a.kind == "article")
    # Articles 5 and 6 are real; the quoted "статті 18" is masked.
    assert nums == ["5", "6"], nums


# ── Embedded amendment-replacement articles ──────────────────────────────────


def _qanun_regex():
    config = _QANUN_CONFIG
    assert config is not None
    return build_anchor_regex(config, "qanun")


def test_embedded_amendment_article_marked_not_dropped() -> None:
    """An amendment that repeals an article and supplies its replacement text
    ("…substitute the following article:" + `مادة ١٥١`) marks the embedded
    replacement header with `quoted_amendment=True` so the scaffolder wraps
    it in a Bluebell QUOTE block. The article count for the amending
    instrument only includes the host anchors, since `_assign_eids` skips
    quoted anchors."""
    text = (
        "المادة ١\nيسمى هذا القانون قانون الجمارك المعدل.\n\n"
        "المادة ٢\nأحكام تمهيدية.\n\n"
        "المادة ٣\nتلغى المادة ١٥١ من القانون الأصلي ويستعاض عنها بالمادة التالية :--\n\n"
        "المادة ١٥١\nالنص الجديد للمادة المستبدلة.\n\n"
        "المادة ٤\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    articles = [a for a in anchors if a.kind == "article"]
    nums = [a.number for a in articles]
    assert nums == ["١", "٢", "٣", "١٥١", "٤"], nums
    # The embedded article is marked, the host articles are not.
    quoted = {a.number for a in articles if a.quoted_amendment}
    assert quoted == {"١٥١"}, quoted
    # And its akn_eid is empty (never went through _assign_eids), so it
    # will not surface as a peer <article>.
    embedded = next(a for a in articles if a.number == "١٥١")
    assert embedded.akn_eid == ""


def test_genuine_numbering_gap_not_dropped() -> None:
    """A real non-contiguous article (no amendment lead-in) is kept, the cue
    is required, not just the forward jump."""
    text = "المادة ١\nنص أول.\n\nالمادة ٢\nنص ثان.\n\nالمادة ٥\nنص خامس بعد فجوة مشروعة.\n"
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢", "٥"], nums


def test_tab_preceded_marginal_article_detected() -> None:
    """Old scans put the rubric in the left margin and the article
    marker after a tab; the tab is a column boundary so the marker is detected,
    while a single-space-preceded in-prose reference (المادة ١٢) is not."""
    text = (
        "اسم القانون\t\tالمادة ١ يطلق على هذا القانون اسم قانون تنظيم المدن.\n\n"
        "تعديل المادة ١٢\tالمادة ٢ تعدل الفقرة الثانية من المادة ١٢ من القانون الأصلي.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢"], nums


def test_space_run_column_article_detected() -> None:
    """Some old scans separate the trailing-margin rubric from the next
    article marker by a run of spaces rather than a tab; a 3+ space gap is a
    column boundary, while the single-space reference (المادة ٤١) is not."""
    text = (
        "سن المندوب السامي ما يلي :-\n\n"
        "المادة ١ يطلق على هذا القانون اسم قانون تنظيم المدن.          اسم القانون\n"
        "تعديل المادة ٤١ من القانون الأصلي          المادة ٢ تعدل المادة الحادية والأربعين.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢"], nums


def test_contiguous_added_article_kept_despite_cue() -> None:
    """A lead-in that introduces a genuinely contiguous new article (no forward
    jump) is kept, the jump is required, not just the cue."""
    text = (
        "المادة ١\nنص أول.\n\n"
        "المادة ٢\nنص ثان.\n\n"
        "المادة ٣\nتضاف المادة التالية :--\n\n"
        "المادة ٤\nالمادة الجديدة المضافة.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢", "٣", "٤"], nums


def test_embedded_article_with_words_between_pointer_and_colon() -> None:
    """Real lead-ins put words between the "the following" pointer and the colon
    ("…the following new article as article 152:"); the embedded header is
    marked as quoted amendment rather than promoted to a peer article."""
    text = (
        "المادة ١\nأحكام.\n\n"
        "المادة ٢\nأحكام.\n\n"
        "المادة ٣\nيعدل القانون الأصلي باضافة المادة الجديدة التالية اليه كمادة ١٥٢ :--\n\n"
        "المادة ١٥٢\nالنص الجديد للمادة المستبدلة.\n\n"
        "المادة ٤\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    articles = [a for a in anchors if a.kind == "article"]
    quoted = {a.number for a in articles if a.quoted_amendment}
    assert quoted == {"١٥٢"}, quoted


def test_taadel_almaadda_marked() -> None:
    """`تعدل المادة` is a declared amendment trigger phrase. Before the config
    surface existed, an amending act of this shape emitted `1, 2, 4, 5, 3, 6, ...`
    because article 3 (the amended article) was promoted. With the phrase in the
    catalogue the embedded article is marked and the host sequence stays monotonic."""
    text = (
        "المادة ١\nنص أول.\n\n"
        "المادة ٢\nنص ثان.\n\n"
        "المادة ٣\nتعدل المادة ٩ من قانون العقوبات لتصبح على النحو التالي :--\n\n"
        "المادة ٩\nالنص الجديد.\n\n"
        "المادة ٤\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    articles = [a for a in anchors if a.kind == "article"]
    quoted = {a.number for a in articles if a.quoted_amendment}
    assert quoted == {"٩"}, quoted
    host_nums = [a.number for a in articles if not a.quoted_amendment]
    assert host_nums == ["١", "٢", "٣", "٤"]


def test_backward_jump_amendment_marked() -> None:
    """An amending act's shape: host sequence 1, 2, 4, 5 then article 5
    quotes an amendment to a *lower-numbered* article. `تعدل المادة ٣`
    puts number 3 back into the stream. The old forward-only guard
    couldn't catch this; the new != host_max + 1 rule does."""
    text = (
        "المادة ١\nنص أول.\n\n"
        "المادة ٢\nنص ثان.\n\n"
        "المادة ٤\nنص رابع.\n\n"
        "المادة ٥\nتعدل المادة ٣ من قانون العقوبات لتصبح على النحو التالي :--\n\n"
        "المادة ٣\nالنص الجديد.\n\n"
        "المادة ٦\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    articles = [a for a in anchors if a.kind == "article"]
    quoted = {a.number for a in articles if a.quoted_amendment}
    assert quoted == {"٣"}, quoted
    host_nums = [a.number for a in articles if not a.quoted_amendment]
    assert host_nums == ["١", "٢", "٤", "٥", "٦"]


def test_nested_child_anchors_marked_with_embedded_article() -> None:
    """A quoted embedded article that contains paragraphs/points must
    mark those nested child anchors too, else they leak past END QUOTE
    as peers of the host chapter."""
    text = (
        "المادة ١\nنص أول.\n\n"
        "المادة ٢\nتعدل المادة ١٥١ من القانون الأصلي لتصبح كما يلي :--\n\n"
        "المادة ١٥١\nالنص الجديد.\n"
        "١- الفقرة الأولى من النص الجديد.\n"
        "٢- الفقرة الثانية من النص الجديد.\n\n"
        "المادة ٣\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    quoted_kinds = {(a.kind, a.number) for a in anchors if a.quoted_amendment}
    # The embedded article and its two nested paragraphs are all marked.
    assert ("article", "١٥١") in quoted_kinds
    # Any paragraph-rank anchors that scan produces from the nested content
    # are marked too; the exact kind depends on the config, but they must
    # not leak as unmarked peers.
    non_article_quoted = [k for (k, _) in quoted_kinds if k != "article"]
    unquoted_non_article = [
        a
        for a in anchors
        if not a.quoted_amendment and a.kind not in ("article", "chapter", "part", "section")
    ]
    # Nested children either aren't detected at all (fine) or are marked (fine).
    # What's forbidden: unmarked child anchors between the embedded article
    # and article 3.
    art151_idx = next(i for i, a in enumerate(anchors) if a.number == "١٥١")
    art3_idx = next(i for i, a in enumerate(anchors) if a.number == "٣" and a.kind == "article")
    between = anchors[art151_idx + 1 : art3_idx]
    for a in between:
        assert a.quoted_amendment, (
            f"anchor {a.kind}={a.number} between quoted article and next host is unmarked; "
            f"would leak past END QUOTE"
        )
    _ = non_article_quoted, unquoted_non_article  # inspection only, not asserted


def test_plural_embedded_articles_marked() -> None:
    """A plural lead-in ("substitute the following articles:") followed by two
    embedded replacement headers marks both; the host sequence stays intact."""
    text = (
        "المادة ١\nنص أول.\n\n"
        "المادة ٢\nنص ثان.\n\n"
        "المادة ٣\nتلغى المواد ١٥١ و١٥٢ ويستعاض عنها بالمواد التالية :--\n\n"
        "المادة ١٥١\nالنص الجديد الأول.\n\n"
        "المادة ١٥٢\nالنص الجديد الثاني.\n\n"
        "المادة ٤\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    articles = [a for a in anchors if a.kind == "article"]
    quoted = {a.number for a in articles if a.quoted_amendment}
    assert quoted == {"١٥١", "١٥٢"}, quoted
    # Host sequence numbers get proper eids; quoted ones do not.
    host_nums = [a.number for a in articles if not a.quoted_amendment]
    assert host_nums == ["١", "٢", "٣", "٤"]


# ── Lone compilation-container artifact ──────────────────────────────────────


def test_lone_compilation_container_dropped() -> None:
    """A scanned ordinance whose only grouping anchor is a high-numbered باب
    (compilation chapter, e.g. "الباب ٤٢") drops that wrapper; articles surface
    at top level."""
    text = "الباب ٤٢\n\nالمادة ١\nنص.\n\nالمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex())
    assert all(a.kind != "part" for a in anchors), [a.kind for a in anchors]
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢"], nums


def test_lone_first_chapter_kept() -> None:
    """A genuine sole "Chapter 1" (sequence start) is not mistaken for a
    compilation wrapper."""
    text = "الفصل ١\nأحكام عامة\n\nالمادة ١\nنص.\n\nالمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex())
    assert any(a.kind == "chapter" for a in anchors), [a.kind for a in anchors]


def test_lone_roman_part_kept() -> None:
    """A lone "PART I" (Roman one) is kept, only purely-numeric numbers >= 2
    are treated as compilation references."""
    config = load_config("al")
    assert config is not None
    regex = build_anchor_regex(config, "ligj")
    text = "Pjesa I\nNeni 1\nx\n\nNeni 2\ny\n"
    anchors = scan_anchors(text, regex)
    assert any(a.kind == "part" for a in anchors), [a.kind for a in anchors]


def test_two_real_containers_not_dropped() -> None:
    """The compilation-container filter is a no-op when the act has genuine
    internal structure (more than one grouping anchor)."""
    text = "الفصل ١\nأحكام\n\nالمادة ١\nنص.\n\nالفصل ٢\nأحكام\n\nالمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex())
    assert len([a for a in anchors if a.kind == "chapter"]) == 2


def test_distant_leadin_colon_does_not_drop() -> None:
    """The lead-in colon must sit close to the embedded header; a "the
    following:" colon far (>120 chars) above a forward-jump article is too
    ambiguous to drop, keep it rather than over-reach."""
    filler = "حشو " * 40  # > 120 chars between the colon and the next header
    text = (
        "المادة ١\nأ.\n\n"
        "المادة ٢\nب.\n\n"
        f"المادة ٣\nيستعاض عنها بالمادة التالية :\n{filler}\n\n"
        "المادة ١٥١\nنص.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert "١٥١" in nums, nums


def test_arabic_chapter_ordinal_words_detected() -> None:
    """Arabic legal codes use ordinal WORDS for chapter/part numbers
    ('الفصل الأول' = Chapter One), not digits. The anchor scan must
    recognise the ordinal-word forms as numbers."""
    text = (
        "الفصل الأول\nتعاريف وأحكام عامة\n\n"
        "مادة ١\nنص.\n\n"
        "الفصل الثاني\nنطاق التطبيق\n\n"
        "مادة ٢\nنص.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    chapter_nums = [a.number for a in anchors if a.kind == "chapter"]
    assert chapter_nums == ["الأول", "الثاني"], chapter_nums


def test_arabic_chapter_compound_ordinal_detected() -> None:
    """Compound ordinals like 'الحادي عشر' (eleventh) must match too."""
    text = "الفصل العاشر\nنص.\n\nمادة ١\nنص.\n\nالفصل الحادي عشر\nنص.\n\nمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex())
    chapter_nums = [a.number for a in anchors if a.kind == "chapter"]
    assert chapter_nums == ["العاشر", "الحادي عشر"], chapter_nums


def test_inline_mid_line_article_detected_after_ocr() -> None:
    """OCR/vision-extracted old PDFs lose the column layout, the
    article marker ends up mid-line ('اسم المرسوم المادة ١ يطلق...'). The
    relaxed boundary lets the scan find it, while the prose-precursor filter
    keeps in-prose references ('في المادة ٥') out."""
    text = (
        "اسم المرسوم المادة ١ يطلق على هذا المرسوم اسم مرسوم تعيين.\n"
        "تعيين المادة ٢ يُعين مدير الأشغال العمومية مراقباً للمناجم.\n"
        "وفقا لأحكام في المادة ٥ من قانون أصلي.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    # ١ and ٢ are real anchors; ٥ is a "في" cross-reference and must drop.
    assert "١" in nums and "٢" in nums, nums
    assert "٥" not in nums, nums


def test_numbered_heading_pattern_fallback() -> None:
    """Early-20th-century regulations use 'HEADING\\nN - ...' with
    no مادة keyword. The heading-pattern fallback fires when the primary
    scan returns nothing, producing synthetic article anchors."""
    text = (
        "نظام تنظيم المدن (اعداد الخرائط)\n\n"
        "تعاريف\n١ - في هذا النظام تعني الكلمات التالية.\n\n"
        "اعداد\n٢ - تعد الخارطة على اساس النقاط الحكومية.\n\n"
        "مضمون الخارطة\n٣ - تحدد الخارطة الحدود والإرتدادات.\n\n"
        "خطوط المناسيب\n٤ - تبين الخارطة خطوط المناسيب.\n"
    )
    anchors = scan_anchors(text, _qanun_regex())
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["١", "٢", "٣", "٤"], nums


def test_heading_pattern_fallback_requires_min_hits() -> None:
    """A single 'HEADING\\nN - ...' occurrence is not enough, could be a
    bullet point in a one-clause decree. Fire only when ≥3 hits exist."""
    text = "عنوان وحيد\n١ - بند واحد فقط.\n\nنص حر بدون ترقيم منظم.\n"
    anchors = scan_anchors(text, _qanun_regex())
    assert anchors == [], anchors


def test_ua_declension_references_not_promoted() -> None:
    """Ukrainian marks a heading in the nominative ("Стаття 5.") and a
    cross-reference in a declined case ("статті 5", "статтею 5"). Only the
    heading is an anchor; declined mentions are prose."""
    regex = build_anchor_regex(load_config("ua"), "act")
    text = (
        "Стаття 5. Загальні положення\n"
        "1. Відповідно до статті 5 цього Кодексу та згідно зі статтею 30 "
        "особа має право, визначене частиною шостою статті 30 цього Кодексу.\n"
        "Стаття 6. Інше\n"
    )
    anchors = scan_anchors(text, regex)
    arts = [a for a in anchors if a.kind == "article"]
    assert [a.number for a in arts] == ["5", "6"]


def test_ua_heading_forms_drop_declensions_keep_abbreviations() -> None:
    """Heading anchors keep the primary form + capitalised abbreviations but
    drop lowercase declensions; a lowercase *primary* form (частина) stays.
    Reference resolution keeps every form (tested via _split_term_with_parens)."""
    from codify.pipeline.enrich.anchors import _heading_forms, _split_term_with_parens

    assert _heading_forms("Член (Чл.)") == ["Член", "Чл."]
    assert _heading_forms("Стаття (статті, статтю, статтею)") == ["Стаття"]
    assert _heading_forms("частина (частини, частину)") == ["частина"]
    # The full expansion (declensions included) is what reference resolution sees.
    assert _split_term_with_parens("Стаття (статті, статтю)") == ["Стаття", "статті", "статтю"]


def test_ua_cyrillic_roman_section_numerals() -> None:
    """OCR of "РОЗДІЛ І" uses Cyrillic І (U+0406); it must parse as a section
    with the numeral normalised to Latin "I" for an ASCII eId."""
    regex = build_anchor_regex(load_config("ua"), "act")
    anchors = scan_anchors("РОЗДІЛ І\nЗагальні положення\nСтаття 1. Мета\n", regex, country="ua")
    secs = [a for a in anchors if a.kind == "section"]
    assert [a.number for a in secs] == ["I"]


def test_ua_chapters_restart_per_section_all_survive() -> None:
    """Ukraine nests Глава (chapter) under Розділ (section), inverting AKN's
    default rank. Chapters that restart numbering in each section must nest
    under their own section, not collapse to global chapter numbers."""
    regex = build_anchor_regex(load_config("ua"), "act")
    doc = (
        "РОЗДІЛ I\nГЛАВА 1\nСтаття 1. A\nтекст.\nГЛАВА 2\nСтаття 2. B\nтекст.\n"
        "РОЗДІЛ II\nГЛАВА 1\nСтаття 3. C\nтекст.\nГЛАВА 2\nСтаття 4. D\nтекст.\n"
    )
    anchors = scan_anchors(doc, regex, country="ua", doctype="act")
    chapters = [a for a in anchors if a.kind == "chapter"]
    assert len(chapters) == 4
    # Each chapter nests under its section, not a global chapter scope.
    assert {a.parent_eid for a in chapters} == {"sec_I", "sec_II"}


# ── Container heading capture ────────────────────────────────────────────────


class TestContainerHeadingCapture:
    """Container anchors capture the descriptive title
    from the marker line or the next non-blank line, so the scaffolder can
    emit `<heading>` instead of a bare "Chapter 5" marker."""

    def test_arabic_next_line_title_captured(self) -> None:
        text = (
            "الفصل الأول\nأحكام عامة\n\n"
            "المادة ١\nنص المادة.\n\n"
            "الفصل الثاني\nالعقوبات\n\n"
            "المادة ٢\nنص المادة.\n"
        )
        anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
        chapters = [a for a in anchors if a.kind == "chapter"]
        assert [c.heading for c in chapters] == ["أحكام عامة", "العقوبات"]

    def test_same_line_title_captured(self) -> None:
        config = load_config("al")
        assert config is not None
        regex = build_anchor_regex(config, "act")
        anchors = scan_anchors(ALBANIAN_SAMPLE, regex)
        chapters = [a for a in anchors if a.kind == "chapter"]
        assert [c.heading for c in chapters] == ["Hyrje", "Themelimi"]

    def test_no_title_when_next_line_is_another_anchor(self) -> None:
        """A chapter immediately followed by an article marker has no title
        in source; heading stays None so the validator surfaces it."""
        text = "الفصل الأول\nالمادة ١\nنص المادة.\n"
        anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
        chapter = next(a for a in anchors if a.kind == "chapter")
        assert chapter.heading is None

    def test_prose_sentence_not_captured_as_title(self) -> None:
        """A sentence-shaped next line (terminal punctuation) is body
        content, not a title."""
        text = "الفصل الأول\nيهدف هذا القانون الى تنظيم قطاع الانشاءات في زرزورة.\nالمادة ١\nنص.\n"
        anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
        chapter = next(a for a in anchors if a.kind == "chapter")
        assert chapter.heading is None

    def test_overlong_line_not_captured(self) -> None:
        long_line = "كلمة " * 30
        text = f"الفصل الأول\n{long_line}\nالمادة ١\nنص.\n"
        anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
        chapter = next(a for a in anchors if a.kind == "chapter")
        assert chapter.heading is None

    def test_articles_do_not_capture_headings(self) -> None:
        """Only container kinds get the lookahead; article headings ride
        the LLM body-fill path."""
        text = "المادة ١\nتعاريف\nفي هذا القانون تكون للكلمات المعاني المخصصة.\n"
        anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
        article = next(a for a in anchors if a.kind == "article")
        assert article.heading is None


def test_heading_capture_when_number_recurs_in_title() -> None:
    """Regression pin: the container number recurring inside the title
    ("KREU I - PARIMET E PERGJITHSHME" contains I) must not truncate the
    capture. Guards against any future marker-end logic that searches for
    the number inside the line."""
    config = load_config("al")
    assert config is not None
    regex = build_anchor_regex(config, "act")
    text = "Kreu I — PARIMET E PERGJITHSHME\n\nNeni 1\nKy ligj rregullon parimet.\n"
    anchors = scan_anchors(text, regex)
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading == "PARIMET E PERGJITHSHME"


def test_arabic_heading_capture_with_ordinal_recurrence() -> None:
    """Regression pin, RTL twin of the recurrence case: the ordinal stem
    recurs inside a title word (`الأولية` contains `الأول`); the full
    title must survive any marker-end search logic."""
    text = "الفصل الأول: الأحكام الأولية\nالمادة ١\nنص المادة.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading == "الأحكام الأولية"


def test_toc_body_ordinal_digit_twins_dedup_to_one_chapter() -> None:
    """The `_normalise_number` docstring contract: a TOC `الفصل الأول` and
    body `الفصل ١` fold to the same chp_1 and dedup collapses them."""
    text = "الفصل الأول\nالفصل ١\nأحكام عامة\nالمادة ١\nنص المادة.\nالمادة ٢\nنص المادة.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapters = [a for a in anchors if a.kind == "chapter"]
    assert len(chapters) == 1
    assert chapters[0].akn_eid == "chp_1"


def test_dropped_anchor_marker_line_not_captured_as_title() -> None:
    """A marker-shaped next line whose anchor an upstream pass dropped
    (TOC dedup, orphan drop) must not become the chapter title. Exercises
    the regex re-test directly: the offsets list deliberately omits the
    marker line, simulating the upstream drop."""
    from codify.pipeline.enrich.anchors import StructuralAnchor, _heading_for

    text = "الفصل الأول\nالمادة ٥\n\nالمادة ١\nنص.\n"
    chapter = StructuralAnchor(
        kind="chapter",
        keyword="الفصل",
        number="الأول",
        char_offset=0,
        line=1,
        matched_text="الفصل الأول",
    )
    # Offsets carry only the chapter and the LAST article; the `المادة ٥`
    # line at offset 12 is "dropped" and must be caught by the regex guard.
    offsets = [0, text.index("المادة ١")]
    heading = _heading_for(text, chapter, 0, offsets, _qanun_regex())
    assert heading is None


def test_soft_wrapped_prose_not_captured_as_title() -> None:
    """A wrapped body sentence (no terminal punctuation on its first line,
    prose continuing on the next) is not a title."""
    text = (
        "الفصل الأول\n"
        "تسري احكام هذا الفصل على جميع العقود المبرمة قبل نفاذ\n"
        "هذا القانون وبعده.\n"
        "المادة ١\nنص المادة.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading is None


def test_ua_rozdil_section_captures_heading() -> None:
    """Ukraine maps РОЗДІЛ to kind `section` as the top grouping level;
    section anchors are capture-eligible when the document has articles."""
    regex = build_anchor_regex(load_config("ua"), "act")
    text = "РОЗДІЛ І\nЗагальні положення\n\nСтаття 1. Мета\nтекст.\n"
    anchors = scan_anchors(text, regex, country="ua", doctype="act")
    section = next(a for a in anchors if a.kind == "section")
    assert section.heading == "Загальні положення"


def test_gb_basic_unit_sections_do_not_capture_headings() -> None:
    """gb documents have no articles, so their sections stay out of the
    capture set; UK section headings ride the keyword/body-fill path."""
    regex = build_anchor_regex(load_config("gb"), "act")
    anchors = scan_anchors(UK_SAMPLE, regex, country="gb")
    assert all(a.heading is None for a in anchors if a.kind == "section")


def test_line_split_marker_does_not_capture_number_as_heading() -> None:
    """An OCR marker split across lines (`الفصل\nالأول`) must not leak the
    number line in as the title; the real next-line title is captured.
    Asserted unconditionally: this test is the sole pin of the text-
    coordinate (match_end) logic, and a vacuous pass would silently
    vacate that coverage."""
    text = "الفصل\nالأول\n\nأحكام عامة\n\nمادة ١\nنص.\n\nمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapters = [a for a in anchors if a.kind == "chapter"]
    assert chapters, "the inline hierarchy stopped matching the line-split marker"
    assert chapters[0].heading == "أحكام عامة"


def test_ordinal_21_plus_markers_scan_as_anchors() -> None:
    """The scanner pattern derives from the canonical ordinal table, so
    compounds past 20 (`الفصل الحادي والعشرون`) become anchors. Two
    chapters, so the lone-compilation-container drop stays out of the way."""
    text = (
        "الفصل العشرون\nأحكام عامة\n\n"
        "المادة ١\nنص المادة.\n\n"
        "الفصل الحادي والعشرون\nأحكام ختامية\n\n"
        "المادة ٢\nنص المادة.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    nums = [a.akn_eid for a in anchors if a.kind == "chapter"]
    assert nums == ["chp_20", "chp_21"], nums


def test_inline_prose_leadin_with_colon_not_captured() -> None:
    """A colon-terminated lead-in after the marker is prose, not a title;
    the trailing colon must survive stripping so the guard can fire. The
    next line is a short BODY line (not a marker), so this also pins the
    rejected-inline short-circuit: a rejected lead-in must not promote
    the following line to title."""
    text = "الفصل الأول: تسري الأحكام التالية:\nتعاريف عامة\n\nالمادة ١\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading is None


def test_nextline_colon_leadin_not_captured() -> None:
    """The next-line path keeps trailing punctuation too: a bare marker
    followed by a colon-terminated lead-in line captures nothing."""
    text = "الفصل الأول\nتسري الأحكام التالية:\n\nالمادة ١\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading is None


def test_enumerated_item_not_captured_as_title() -> None:
    """An enumerated body item on the next line ((أ) ...) is content,
    not a title."""
    text = "الفصل الأول\n(أ) تعريفات\n\nالمادة ١\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading is None


def test_num_prefix_strip_when_match_stops_before_number() -> None:
    """Regex variants whose group(0) stops before the number: the leading
    number token is stripped only when a separator follows, so a title
    merely starting with the same string is left whole."""
    from codify.pipeline.enrich.anchors import StructuralAnchor, _heading_for

    regex = _qanun_regex()
    stopped_before_num = StructuralAnchor(
        kind="chapter",
        keyword="CHAPTER",
        number="2",
        char_offset=0,
        line=1,
        matched_text="CHAPTER",
    )
    heading = _heading_for(
        "CHAPTER 2 - Definitions\nARTICLE 1\nBody.\n", stopped_before_num, 0, [0], regex
    )
    assert heading == "Definitions"
    heading = _heading_for(
        "CHAPTER 2nd Schedule provisions\nARTICLE 1\nBody.\n", stopped_before_num, 0, [0], regex
    )
    assert heading != "nd Schedule provisions"


def test_lone_wrapper_drop_ignores_sub_kinds() -> None:
    """The lone-compilation-wrapper drop keeps its narrow vocabulary: a
    part accompanied by a subpart is still a lone wrapper candidate, and
    a lone subpart is ordinary structure, never dropped."""
    from codify.pipeline.enrich.anchors import StructuralAnchor, _drop_lone_compilation_container

    def _mk(kind: str, number: str, offset: int) -> StructuralAnchor:
        return StructuralAnchor(
            kind=kind,
            keyword=kind.upper(),
            number=number,
            char_offset=offset,
            line=1,
            matched_text=f"{kind.upper()} {number}",
        )

    # part(42) + subpart(1): subpart does not count, part is lone and dropped.
    dropped = _drop_lone_compilation_container(
        [_mk("part", "42", 0), _mk("subpart", "1", 10), _mk("article", "1", 20)]
    )
    assert [a.kind for a in dropped] == ["subpart", "article"]
    # Lone subpart(5): not a wrapper kind, nothing dropped.
    kept = _drop_lone_compilation_container([_mk("subpart", "5", 0), _mk("article", "1", 10)])
    assert [a.kind for a in kept] == ["subpart", "article"]


def test_two_column_line_marker_not_captured_as_title() -> None:
    """Old two-column layouts put the next unit's marker on the same
    line after a tab run; the inline remainder is that marker, never the
    container's title."""
    text = "الفصل الأول\t\tالمادة ١ يطلق على هذا القانون اسم قانون تنظيم المدن.\nالمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading is None


# ── Phantom anchors from amendment lead-ins (v38-I-01 / I-02 root cause) ────


def test_amendment_verb_leadin_citation_not_promoted_to_anchor() -> None:
    """An amending act's shape: host articles in order, lead-ins citing the
    amended article with the VERB form (`تعدل المادة (3) من القانون
    الأصلي`). The citation must not become a phantom anchor, or TOC-dedup
    deletes the genuine host of that number and scrambles the sequence."""
    text = (
        "مادة (1)\nيسمى هذا القرار بقانون قرار مكافحة الفساد المعدل.\n\n"
        "مادة (2)\nيستبدل مصطلح المجلس بمصطلح الجمعية.\n\n"
        "مادة (3)\nيعدل نص المادة (1) من القانون الأصلي، ليصبح على النحو التالي:\n"
        "يكون للكلمات والعبارات الواردة المعاني المخصصة لها أدناه.\n\n"
        "مادة (4)\nيعدل نص المادة (2) من القانون الأصلي، ليصبح على النحو التالي:\n"
        "يخضع لأحكام هذا القرار بقانون رئيس الدولة.\n\n"
        "مادة (5)\nتعدل المادة (3) من القانون الأصلي، لتصبح على النحو الآتي:\n"
        "تنشأ بمقتضى أحكام هذا القرار بقانون هيئة مكافحة الفساد.\n\n"
        "مادة (6)\nأحكام ختامية.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["1", "2", "3", "4", "5", "6"], nums


def test_curly_quote_desync_does_not_eat_following_marker() -> None:
    """OCR mixes ASCII and curly quotes; an odd curly count used to leave
    the quote mask stuck open across the next marker (an article
    vanished). Toggle depth resets at a blank line."""
    text = (
        "مادة (12)\nنص يحتوي على اقتباس ”مشوه بدون إغلاق\n\n"
        "مادة (13)\nنص المادة الثالثة عشرة.\n\n"
        "مادة (14)\nنص المادة الرابعة عشرة.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["12", "13", "14"], nums


def test_markdown_fence_stripped_from_captured_heading() -> None:
    """A next-line title carrying an OCR markdown fence (`### تعاريف`)
    captures the bare title."""
    text = "الفصل الأول\n### تعاريف\n\nالمادة ١\nنص المادة.\n\nالمادة ٢\nنص.\n"
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    chapter = next(a for a in anchors if a.kind == "chapter")
    assert chapter.heading == "تعاريف"


def test_curly_quote_desync_reset_tolerates_crlf() -> None:
    """The blank-line reset fires on CRLF blank lines too."""
    text = (
        "مادة (12)\r\nنص يحتوي على اقتباس ”مشوه بدون إغلاق\r\n\r\n"
        "مادة (13)\r\nنص المادة الثالثة عشرة.\r\n\r\n"
        "مادة (14)\r\nنص المادة الرابعة عشرة.\r\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["12", "13", "14"], nums


def test_curly_quote_reset_tolerates_whitespace_only_separator() -> None:
    """A separator line carrying OCR space residue still resets the mask."""
    text = (
        "مادة (12)\nنص يحتوي على اقتباس ”مشوه بدون إغلاق\n   \n"
        "مادة (13)\nنص المادة.\n\n"
        "مادة (14)\nنص المادة.\n"
    )
    anchors = scan_anchors(text, _qanun_regex(), country="xx", doctype="qanun")
    nums = [a.number for a in anchors if a.kind == "article"]
    assert nums == ["12", "13", "14"], nums


# ── Bis articles (مكرر) + Arabic schedule aliases ───────────────────────────


class TestBisArticles:
    """AC 1/2005 shape: inserted articles numbered `المادة (6) مكرر`.
    The suffix folds into the num (eid `art_6bis`) instead of leaking into
    body text and colliding with the base article."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_bis_article_absorbed_and_folded(self) -> None:
        text = "المادة (6)\nنص أصلي.\nالمادة (6) مكرر\nنص مضاف.\nالمادة (7)\nنص تال.\n"
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_6", "art_6bis", "art_7"]
        assert anchors[1].number == "6 مكرر"

    def test_bis_with_index_and_feminine_form(self) -> None:
        text = "المادة (9) مكرر (1)\nنص.\nالمادة (10) مكررة\nنص آخر.\n"
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_9bis1", "art_10bis"]

    def test_base_and_bis_never_collide(self) -> None:
        text = "المادة (6)\nنص.\nالمادة (6) مكرر\nنص.\nالمادة (6) مكرر (2)\nنص.\n"
        eids = [a.akn_eid for a in self._scan(text)]
        assert len(eids) == len(set(eids)) == 3

    def test_letter_indexed_bis_stays_distinct(self) -> None:
        # Inheritance Law 1944 shape: two bis articles indexed by abjad letter.
        # The letter must survive into a distinct ASCII eid, not fold both to
        # `15bis` and collapse into one element.
        text = "المادة (15) مكررة (أ)\nنص.\nالمادة (15) مكررة (ب)\nنص آخر.\n"
        anchors = self._scan(text)
        eids = [a.akn_eid for a in anchors]
        assert eids == ["art_15bisa", "art_15bisb"]
        assert all(e.isascii() for e in eids)

    def test_letter_indexed_bis_survives_to_shipped_akn(self) -> None:
        # The eid that ships is regenerated by Bluebell, not the scanner; the
        # letter fold must hold across the scanner→scaffold→parse round-trip.
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            BodyBlock,
            BodyFillResponse,
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )

        text = "المادة (15) مكررة (أ)\nنص.\nالمادة (15) مكررة (ب)\nنص آخر.\n"
        anchors = self._scan(text)
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(
            scaffold,
            e2a,
            BodyFillResponse(bodies=[BodyBlock(eid=a.akn_eid, lines=["نص."]) for a in anchors]),
        )
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2020", number="1", language="ara"
        )
        for eid in ("art_15bisa", "art_15bisb"):
            assert f'eId="{eid}"' in xml

    def test_letter_bis_does_not_collide_with_numeric_bis(self) -> None:
        text = "المادة (15) مكرر (1)\nنص.\nالمادة (15) مكررة (أ)\nنص.\n"
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == ["art_15bis1", "art_15bisa"]

    def test_unparenthesised_letter_is_not_a_bis_index(self) -> None:
        # A bare abjad letter must not be read as an index; only "( letter )"
        # counts, else the conjunction و ("and") folds into a phantom index.
        text = "المادة (15) مكررة و ما بعدها\nنص.\n"
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == ["art_15bis"]

    def test_non_abjad_letter_index_stays_ascii(self) -> None:
        # An in-range but non-abjad glyph (ة) can't index a bis; the eid must
        # still be ASCII rather than shipping the raw glyph.
        text = "المادة (15) مكرر (ة)\nنص.\n"
        eids = [a.akn_eid for a in self._scan(text)]
        assert all(e.isascii() for e in eids)


class TestArabicScheduleAnchors:
    """PP-Amendment / AC 18/2016 shape: trailing annex material must anchor
    (and window) instead of being swallowed into the last article."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_trailing_annex_captured_top_level(self) -> None:
        text = "مادة (1)\nنص المادة.\nمادة (2)\nنص آخر.\nالملحق رقم (1)\nجدول الرواتب والدرجات.\n"
        anchors = self._scan(text)
        annex = [a for a in anchors if a.kind == "schedule"]
        assert len(annex) == 1
        assert annex[0].akn_eid == "schedule_1"
        assert annex[0].depth == 0

    def test_prose_annex_reference_not_anchored(self) -> None:
        text = "مادة (1)\nيحال إلى الملحق رقم (1) من هذا القانون.\nمادة (2)\nنص.\n"
        anchors = self._scan(text)
        assert [a.kind for a in anchors] == ["article", "article"]

    def test_builtin_schedule_aliases_survive_config_hierarchy(self) -> None:
        # The fixture declares no schedule entry; the builtin aliases must be
        # unioned in regardless.
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        assert rx.search("\nAnnex 1\n") is not None

    def test_windows_include_schedule(self) -> None:
        from codify.pipeline.enrich.anchors import windows_from_anchors

        text = "مادة (1)\nنص.\nمادة (2)\nنص.\nالملحق رقم (1)\nمحتوى الملحق.\n"
        anchors = self._scan(text)
        windows = windows_from_anchors(text, anchors)
        windowed_eids = {a.akn_eid for w in windows for a in w.anchors}
        assert "schedule_1" in windowed_eids


class TestScheduleAndBisEndToEnd:
    """Scan → scaffold → Bluebell round-trips: the silent failure modes these
    changes fix (indented SCHEDULE dropped; non-ASCII bis eids) regress
    invisibly without a parse-level assertion."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_trailing_annex_round_trips_to_attachment(self) -> None:
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            BodyBlock,
            BodyFillResponse,
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )

        text = "مادة (1)\nنص المادة.\nمادة (2)\nنص آخر.\nالملحق رقم (1)\nجدول الرواتب والدرجات.\n"
        anchors = self._scan(text)
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(
            scaffold,
            e2a,
            BodyFillResponse(
                bodies=[BodyBlock(eid=a.akn_eid, lines=["محتوى تجريبي."]) for a in anchors]
            ),
        )
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2020", number="1", language="ara"
        )
        assert "<attachment" in xml
        assert 'eId="att_1"' in xml
        assert "محتوى تجريبي" in xml

    def test_bis_eids_agree_between_scanner_and_parse(self) -> None:
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            BodyBlock,
            BodyFillResponse,
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )

        text = "المادة (6)\nنص.\nالمادة (6) مكرر\nنص.\nالمادة (6) مكرر (2)\nنص.\n"
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_6", "art_6bis", "art_6bis2"]
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(
            scaffold,
            e2a,
            BodyFillResponse(bodies=[BodyBlock(eid=a.akn_eid, lines=["نص."]) for a in anchors]),
        )
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2020", number="1", language="ara"
        )
        for eid in ("art_6", "art_6bis", "art_6bis2"):
            assert f'eId="{eid}"' in xml

    def test_mid_document_table_caption_not_promoted(self) -> None:
        # A جدول caption inside the body is an embedded table, not an annex;
        # promoting it would pull every later article inside the attachment.
        text = "مادة (1)\nنص.\nالجدول رقم (1)\nبيانات الجدول.\nمادة (2)\nنص.\nمادة (3)\nنص.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2", "art_3"]

    def test_prose_bis_reference_not_anchored(self) -> None:
        text = (
            "مادة (1)\nكما ورد في المادة (12) مكرر من القانون الأساسي يلتزم الجميع.\n"
            "مادة (2)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2"]


class TestAnnexWithOwnArticles:
    """An annex carrying its own restarted article numbering keeps both the
    schedule anchor and its content; the digit-repair prior must not cross
    the annex boundary and 'repair' the restart."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_annex_articles_nest_under_schedule(self) -> None:
        text = "مادة (1)\nنص.\nمادة (2)\nنص.\nالملحق رقم (1)\nمادة (1)\nنص بند الملحق الأول.\n"
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == ["art_1", "art_2", "schedule_1", "schedule_1__art_1"]

    def test_annex_articles_round_trip_into_attachment(self) -> None:
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            BodyBlock,
            BodyFillResponse,
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )

        text = "مادة (1)\nنص.\nمادة (2)\nنص.\nالملحق رقم (1)\nمادة (1)\nنص بند الملحق الأول.\n"
        anchors = self._scan(text)
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(
            scaffold,
            e2a,
            BodyFillResponse(bodies=[BodyBlock(eid=a.akn_eid, lines=["نص."]) for a in anchors]),
        )
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2020", number="1", language="ara"
        )
        assert 'eId="att_1__art_1"' in xml

    def test_digit_repair_stops_at_annex_boundary(self) -> None:
        # Main-body OCR repair (4,5,1,7 → 4,5,6,7) still fires, while the
        # annex's restart at 1 is left alone.
        text = (
            "مادة (4)\nنص.\nمادة (5)\nنص.\nمادة (1)\nنص.\nمادة (7)\nنص.\n"
            "الملحق رقم (1)\nمادة (1)\nنص الملحق.\n"
        )
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == ["art_4", "art_5", "art_6", "art_7", "schedule_1", "schedule_1__art_1"]

    def test_arabic_indic_bis_display_stays_arabic(self) -> None:
        anchors = self._scan("المادة (٩) مكرر (١)\nنص.\n")
        assert anchors[0].number == "٩ مكرر (١)"
        assert anchors[0].akn_eid == "art_9bis1"


class TestWrappedCitationPhantoms:
    """OCR can wrap recital or penalty sentences mid-citation, stranding the
    precursor at the end of the previous line; the marker must not anchor."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_wrapped_walasima_recital(self) -> None:
        # A wrapped recital citation must not swallow the following article.
        text = (
            "بعد الاطلاع على قانون المرصد رقم (٤١) لسنة ٢٠٤٢ ولاسيما\n"
            "المادة (٥٥) منه، وعلى ما عرضه وزير العدل.\nمادة (٢)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_2"]

    def test_wrapped_biquwwa_enactment(self) -> None:
        # Civil Code: fabricated Article 41 from the Basic Law citation.
        text = (
            "وبعد أن أصبح القانون مصدراً بقوة\n"
            "المادة (41) من القانون الأساسي لسنة 2003، صدر القانون التالي:\n"
            "مادة (1)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_1"]

    def test_wrapped_ahkam_penalty_clause(self) -> None:
        # Env 7/1999: 11 phantom duplicates under the penalties part.
        text = (
            "مادة (60)\nيعاقب كل من يخالف أحكام\n"
            "المادة (10) من هذا القانون بغرامة مالية.\nمادة (61)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_60", "art_61"]

    def test_terminated_previous_line_still_anchors(self) -> None:
        text = "مادة (1)\nنص المادة الأولى.\nالمادة (2)\nنص المادة الثانية.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2"]


class TestMarkerBoundaryPolicy:
    """Bahasa writes `Pasal 78` identically as citation and as header,
    so position on the line is the only thing telling them apart."""

    def _scan(self, text: str, country: str, **kw: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(load_config(country), "act", **kw)
        return scan_anchors(text, rx, country=country)

    # A regency regulation's recital, which nested the document under a
    # phantom art_343: "based on the provision of Article 343 of the Minister's
    # Regulation…".
    RECITAL = (
        "Menimbang : a. bahwa berdasarkan ketentuan Pasal 343 Peraturan "
        "Menteri Dalam Negeri Nomor 12 Tahun 2031 tentang Tata Cara "
        "Perencanaan;\n\nPasal I\nKetentuan diubah.\n"
    )

    def test_mid_line_citation_is_not_an_anchor(self) -> None:
        assert [a.akn_eid for a in self._scan(self.RECITAL, "xl")] == ["art_I"]

    def test_relaxed_still_admits_a_mid_line_marker(self) -> None:
        # Pinned so the two policies cannot silently converge.
        text = "اسم المرسوم المادة ١ يطلق على هذا المرسوم اسم مرسوم تعيين.\n"
        assert [a.akn_eid for a in self._scan(text, "xx")] == ["art_1"]

    def test_centred_marker_indented_past_the_line_start_window_anchors(self) -> None:
        # More leading space than `^\s{0,8}` allows, so the newline branch claims it.
        text = "Bab I\nKetentuan Umum\n" + " " * 40 + "Pasal 1\nIsi pasal.\n"
        assert "chp_I__art_1" in [a.akn_eid for a in self._scan(text, "xl")]

    def test_column_gap_still_anchors(self) -> None:
        text = "Ketentuan Umum   Pasal 1\nIsi pasal.\n"
        assert [a.akn_eid for a in self._scan(text, "xl")] == ["art_1"]

    def test_wrapped_citation_stays_suppressed_under_line_anchored(self) -> None:
        # Pins the newline inside the match, which `_is_prose_reference` needs.
        text = (
            "بعد الاطلاع على قانون المرصد رقم (٤١) لسنة ٢٠٤٢ ولاسيما\n"
            "المادة (٥٥) منه، وعلى ما عرضه وزير العدل.\nمادة (٢)\nنص.\n"
        )
        anchors = self._scan(text, "xx", boundary="line_anchored")
        assert [a.akn_eid for a in anchors] == ["art_2"]

    def test_a_precursor_spelled_without_its_hamza_still_suppresses(self) -> None:
        # Scans write أحكام as احكام, and an unfolded list misses every one.
        text = "مع مراعاة احكام\nالمادة (١٢) من هذا القانون.\nمادة (٢)\nنص.\n"
        anchors = self._scan(text, "xx", boundary="line_anchored")
        assert [a.akn_eid for a in anchors] == ["art_2"]

    def test_lost_line_layout_is_not_read_as_a_document_without_markers(self) -> None:
        # The denominator is line-anchored too, so a flattened source measures
        # zero expected markers and would skip the coverage gate entirely.
        flat = re.sub(r"\n+", " ", self.RECITAL)
        assert not self._scan(flat, "xl")
        assert markers_outside_boundary(flat, load_config("xl"), "act", "article") > 0

    def test_wrapped_citation_tail_is_not_a_heading(self) -> None:
        """A citation list wrapping onto its own line starts that line, so
        the boundary policy admits it. The previous line's last word is the only
        thing that says it is a citation, which is what precursors read.

        Left unfiltered on a base statute this outranked the real `Pasal 18`
        heading in the TOC-twin drop, orphaning its `Ayat (1)`."""
        text = (
            "Pasal 17\n Ayat (1)\n Sebagai acuan pokok, harus diperhatikan juga\n"
            "Pasal 11, Pasal 14, Pasal 16, Pasal 17, dan\n"
            "Pasal 18.\n Ayat (2)\n Cukup jelas.\n"
        )
        nums = [a.number for a in self._scan(text, "xl") if a.kind == "article"]
        assert nums == ["17"], nums


class TestRomanHostAmendments:
    """A Bahasa amending instrument's own body is `Pasal I` and
    `Pasal II`, so an Arabic-numbered article between them is the text being
    inserted into another law, not a provision of this one."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(load_config("xl"), "act")
        return scan_anchors(text, rx, country="xl")

    PERBUP = (
        "Pasal I\nDi antara Pasal 2 dan Pasal 3 disisipkan 1 pasal, yakni:\n"
        "Pasal 2A\nRKPD diubah.\n"
        "Pasal II\nPeraturan ini mulai berlaku pada tanggal diundangkan.\n"
    )

    def test_inserted_article_is_quoted_not_a_peer(self) -> None:
        anchors = self._scan(self.PERBUP)
        quoted = [a.number for a in anchors if a.quoted_amendment]
        real = [a.number for a in anchors if a.kind == "article" and not a.quoted_amendment]
        assert quoted == ["2A"], quoted
        assert real == ["I", "II"], real

    def test_material_after_the_last_roman_host_is_not_quoted(self) -> None:
        """`Pasal II` closes the amendment payload. An annex after it belongs to
        the amending instrument, so quoting it would drop it from the scaffold."""
        anchors = self._scan(self.PERBUP + "Pasal 5\nLampiran milik Peraturan ini.\n")
        assert [a.number for a in anchors if a.quoted_amendment] == ["2A"]

    def test_a_single_roman_article_is_not_a_host(self) -> None:
        """A base statute whose `Pasal 1` was OCR'd as `Pasal I` would
        otherwise quote every article after it. Real scans do exactly this."""
        text = "Pasal I\nKetentuan umum.\nPasal 2\nIsi.\nPasal 3\nIsi.\n"
        assert not [a for a in self._scan(text) if a.quoted_amendment]


class TestDeclaredAttachmentCaption:
    """A Bahasa statute can restate every article number in its
    Penjelasan. Scanned as body it lands in the closing chapter, so a law shows
    a second copy of every article, each reading `Cukup jelas`."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(load_config("xl"), "act")
        return scan_anchors(text, rx, country="xl")

    # Two chapters, as a real statute has: the elucidation trails the last of
    # them, which is what keeps its numbering out of the body's chain.
    STATUTE = (
        "Bab I\nKETENTUAN UMUM\n\nPasal 1\nDalam Undang-Undang ini yang dimaksud.\n\n"
        "Pasal 2\nPembangunan ketenagakerjaan.\n\n"
        "Bab II\nKETENTUAN PENUTUP\n\nPasal 3\nMulai berlaku.\n\n"
        + " "
        * 36
        + "PENJELASAN\nATAS\nUNDANG-UNDANG NOMOR 13\n\nUMUM\n\nPasal 1\nCukup jelas.\n\n"
        "Pasal 2\nCukup jelas.\n\nPasal 3\nCukup jelas.\n"
    )

    def test_elucidation_lands_in_an_attachment(self) -> None:
        eids = [a.akn_eid for a in self._scan(self.STATUTE) if a.kind == "article"]
        body = [e for e in eids if not e.startswith("schedule")]
        elucidation = [e for e in eids if e.startswith("schedule")]
        # The complete body list, not a shape check: a loose assertion hid the
        # loss of `chp_II__art_3` to its elucidation twin.
        assert body == ["chp_I__art_1", "chp_I__art_2", "chp_II__art_3"], body
        assert elucidation == ["schedule_1__art_1", "schedule_1__art_2", "schedule_1__art_3"]

    def test_a_centred_caption_is_still_found(self) -> None:
        """A statute may indent it 36 spaces; the inferred Arabic form keeps a
        tight margin, so only the declared branch is permissive."""
        sched = [a for a in self._scan(self.STATUTE) if a.kind == "schedule"]
        assert sched and sched[0].heading == "PENJELASAN", sched

    def test_promotion_needs_the_declaration(self) -> None:
        """Declaring the caption is what promotes it."""
        from codify.pipeline.enrich.anchors import _annex_caption_re

        assert "PENJELASAN" in _annex_caption_re("xl").pattern
        assert "PENJELASAN" not in _annex_caption_re("xx").pattern

    def test_lampiran_promotes_too(self) -> None:
        """Ministerial regulations carry a Lampiran and no Penjelasan."""
        from codify.pipeline.enrich.anchors import _annex_caption_re

        assert _annex_caption_re("xl").search("\nLAMPIRAN\n")

    def test_a_declared_caption_may_carry_its_own_number(self) -> None:
        """Bahasa attachments are numbered, so requiring the caption to end
        its line matched PENJELASAN and missed every LAMPIRAN I."""
        from codify.pipeline.enrich.anchors import _annex_caption_re

        rx = _annex_caption_re("xl")
        m = rx.search("\n       LAMPIRAN I\n")
        assert m and (m.group("declared") + m.group("decnum")).strip() == "LAMPIRAN I"
        assert rx.search("\nLAMPIRAN II\n")
        # A number is not licence to swallow a sentence that opens with it.
        assert not rx.search("\nLAMPIRAN I MERUPAKAN BAGIAN TIDAK TERPISAHKAN\n")


class TestUnnumberedAnnex:
    """The الملحق: colon heading form is recognised as an annex."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_colon_annex_anchors_with_heading(self) -> None:
        text = (
            "مادة (18)\nنص.\nمادة (19)\nنص أخير.\n"
            "الملحق: سلم الدرجات والرواتب والعلاوات\nالدرجة أ1 راتب أساسي.\n"
        )
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_18", "art_19", "schedule_1"]
        assert anchors[-1].heading == "سلم الدرجات والرواتب والعلاوات"

    def test_colon_annex_round_trips_to_attachment(self) -> None:
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            BodyBlock,
            BodyFillResponse,
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )

        text = "مادة (1)\nنص.\nالملحق: سلم الرواتب\nمحتوى الجدول.\n"
        anchors = self._scan(text)
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(
            scaffold,
            e2a,
            BodyFillResponse(bodies=[BodyBlock(eid=a.akn_eid, lines=["نص."]) for a in anchors]),
        )
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2016", number="18", language="ara"
        )
        assert "<attachment" in xml

    def test_mid_document_colon_caption_dropped(self) -> None:
        text = "مادة (1)\nنص.\nجدول: بيانات تفصيلية\nصف أول.\nمادة (2)\nنص.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2"]

    def test_prose_colon_mention_not_anchored(self) -> None:
        text = "مادة (1)\nكما هو مبين في الجدول: أدناه ضمن هذه المادة.\nمادة (2)\nنص.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2"]


class TestWave4ReviewEdges:
    def _scan(self, text: str, toc_end: int = 0) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx", toc_end=toc_end)

    def test_unterminated_body_line_before_genuine_article_still_anchors(self) -> None:
        # OCR often drops the terminal period; a genuine heading after an
        # unterminated body line (no precursor at its tail) must anchor.
        text = "مادة (1)\nنص المادة الأولى بدون علامة ختام\nمادة (2)\nنص المادة الثانية.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_2"]

    def test_annex_heading_at_end_of_file_not_truncated(self) -> None:
        text = "مادة (1)\nنص.\nالملحق: سلم الرواتب"
        anchors = self._scan(text)
        assert anchors[-1].kind == "schedule"
        assert anchors[-1].heading == "سلم الرواتب"

    def test_annex_inside_toc_region_ignored(self) -> None:
        text = "الملحق: سلم الرواتب\nمادة (1)\nنص.\nالملحق: سلم الرواتب\nمحتوى.\n"
        anchors = self._scan(text, toc_end=10)
        assert [a.kind for a in anchors] == ["article", "schedule"]

    def test_definitions_line_with_colon_not_promoted(self) -> None:
        # A definitions line for the term الجدول must not become an annex
        # that swallows the whole body (code-review demonstrated shape).
        text = "الجدول: يقصد به الجدول المرفق بهذا القانون.\nمادة (1)\nنص.\nمادة (2)\nنص.\n"
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_1", "art_2"]

    def test_leading_colon_annex_before_any_article_ignored(self) -> None:
        # An undetected TOC/preface entry has no preceding body anchor and
        # must never own the document as attachment content.
        text = "الملحق: سلم الرواتب\nمادة (1)\nنص.\nمادة (2)\nنص.\n"
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_1", "art_2"]


class TestCoverageGateDenominator:
    """The expected count must speak the
    scanner's language (distinct provisions, prose references excluded) or
    improved suppression reads as lost coverage."""

    def test_toc_and_references_do_not_inflate_expected(self) -> None:
        from codify.pipeline.enrich.anchors import anchor_coverage

        cfg = _QANUN_CONFIG
        # TOC listing both articles, two real articles, and a wrapped
        # line-start reference to a THIRD number (المادة (55) of another
        # law). Without the prose filter that reference would count as a
        # distinct expected provision and the ratio would read 2/3.
        text = (
            "محتويات التشريع:\n"
            "مادة (1) تعاريف\n"
            "مادة (2) أحكام عامة\n"
            "مادة (1)\nنص المادة الأولى وفق قانون التحكيم ولاسيما\n"
            "المادة (55) منه المعمول به في هذا الشأن.\n"
            "مادة (2)\nنص المادة الثانية.\n"
        )
        rx = build_anchor_regex(cfg, "act")
        anchors = scan_anchors(text, rx, country="xx")
        cov = anchor_coverage(text, anchors, cfg, "act", "article")
        ratio, captured, expected = cov.ratio, len(cov.captured), len(cov.expected)
        assert expected == 2
        assert captured == 2
        assert ratio == 1.0

    def test_missed_variant_still_fails_the_gate(self) -> None:
        from codify.pipeline.enrich.anchors import anchor_coverage

        cfg = _QANUN_CONFIG
        # Three distinct provisions at structural position; if the scanner
        # captured only two of them, the gate must still see three expected.
        text = "مادة (1)\nنص.\nمادة (2)\nنص.\nمادة (3)\nنص.\n"
        assert len(anchor_coverage(text, [], cfg, "act", "article").expected) == 3


class TestWave41Residuals:
    """v6 rerun residuals, each diagnosed from the fresh OCR artifacts."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_penalty_heading_citation_not_anchored(self) -> None:
        # Env 7/1999: penalty articles are headed "عقوبة مخالفة المادة 10";
        # the cited article must not become a phantom sibling.
        text = (
            "مادة (61)\nعقوبة مخالفة المادة 10\n"
            "يعاقب كل من يخالف أحكام المادة (10) من هذا القانون بغرامة.\n"
            "مادة (62)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_61", "art_62"]

    def test_misread_duplicate_repaired_before_dedup(self) -> None:
        # ArbReg 39/2004: a second "مادة (١)" that is really article 6 sits
        # between 5 and 7; repair must fix it before dedup can collapse it
        # with the genuine article 1.
        text = (
            "مادة (١)\nتعاريف.\nمادة (٢)\nنص.\nمادة (٣)\nنص.\nمادة (٤)\nنص.\n"
            "مادة (٥)\nنص.\nمادة (١)\nنص المادة السادسة.\nمادة (٧)\nنص.\n"
        )
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == [f"art_{n}" for n in (1, 2, 3, 4, 5, 6, 7)], eids

    def test_bare_keyword_annex_with_next_line_heading(self) -> None:
        # AC 18/2016 body form: "الملحق" alone on a line, heading beneath.
        text = "مادة (18)\nنص.\nمادة (19)\nنص.\n\nالملحق\n\nسلم الدرجات والرواتب\nالدرجة أ1.\n"
        anchors = self._scan(text)
        assert anchors[-1].kind == "schedule"
        assert anchors[-1].heading == "سلم الدرجات والرواتب"

    def test_bare_keyword_in_prose_not_promoted(self) -> None:
        # A sentence ending with the word الملحق must not anchor: the
        # bare-keyword form requires the keyword alone on its line.
        text = "مادة (1)\nيصدر النظام وفق ما ورد في الملحق\nالمشار إليه أعلاه.\nمادة (2)\nنص.\n"
        assert [a.kind for a in self._scan(text)] == ["article", "article"]

    def test_bare_annex_heading_not_duplicated_in_body(self) -> None:
        from codify.pipeline.enrich.bluebell import parse_to_akn
        from codify.pipeline.enrich.scaffold import (
            assemble_filled_scaffold,
            scaffold_from_anchors,
        )
        from codify.pipeline.enrich.verbatim import fill_bodies_verbatim

        text = "مادة (1)\nنص.\n\nالملحق\n   \nسلم الدرجات والرواتب\nالدرجة أ1 راتب أساسي.\n"
        anchors = self._scan(text)
        assert anchors[-1].heading == "سلم الدرجات والرواتب"
        # Verbatim path: the heading line must not repeat as body text.
        resp = fill_bodies_verbatim(text, list(anchors))
        annex_block = next(b for b in resp.bodies if b.eid == "schedule_1")
        assert all("سلم الدرجات" not in ln for ln in annex_block.lines)
        # Assemble path: same guarantee through the scaffold.
        scaffold, e2a = scaffold_from_anchors(anchors)
        filled = assemble_filled_scaffold(scaffold, e2a, resp)
        xml = parse_to_akn(
            filled, country="xx", doctype="act", date="2016", number="18", language="ara"
        )
        # The heading appears in <heading> and the attachment's FRBRalias,
        # but never as a body paragraph.
        from lxml import etree

        root = etree.fromstring(xml.encode("utf-8"))
        ns = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
        body_ps = ["".join(p.itertext()) for p in root.iter(f"{{{ns}}}p")]
        assert all("سلم الدرجات" not in t for t in body_ps), body_ps

    def test_sentence_ending_mukhalafa_does_not_swallow_next_header(self) -> None:
        # "…ارتكب مخالفة" ends a sentence; the genuine header after it
        # (even across a blank line) must still anchor.
        for gap in ("\n", "\n\n"):
            text = f"مادة (10)\nيعد من ارتكب ذلك قد ارتكب مخالفة{gap}مادة (11)\nنص المادة.\n"
            eids = [a.akn_eid for a in self._scan(text)]
            assert eids == ["art_10", "art_11"], (gap, eids)

    def test_penalty_heading_same_line_still_suppressed(self) -> None:
        text = "مادة (61)\nعقوبة مخالفة المادة 10\nنص العقوبة.\nمادة (62)\nنص.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_61", "art_62"]

    def test_wrapped_trailing_jadwal_not_promoted(self) -> None:
        # Prose wrapping a trailing الجدول onto its own line inside the last
        # article must not become a phantom annex.
        text = "مادة (1)\nنص.\nمادة (2)\nتستوفى الرسوم المبينة في\nالجدول\nالمرفق طيه.\n"
        assert [a.kind for a in self._scan(text)] == ["article", "article"]

    def test_blank_line_set_off_bare_annex_still_anchors(self) -> None:
        text = "مادة (1)\nنص.\nمادة (2)\nنص أخير.\n\nالملحق\n\nسلم الدرجات والرواتب\nمحتوى.\n"
        anchors = self._scan(text)
        assert anchors[-1].kind == "schedule"
        assert anchors[-1].heading == "سلم الدرجات والرواتب"


class TestBlankLinePrecursorRelease:
    """A prose precursor that ends its own paragraph (blank line before the
    marker) must not suppress the next article header."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_dhalika_before_blank_line_releases_marker(self) -> None:
        # Civil Code: art 812 vanished because art 811 ends "…بغير ذلك".
        text = (
            "مادة (811)\n"
            "لا يجوز للوكيل أن يبيع ماله لموكله ما لم يوجد اتفاق يقضي بغير ذلك\n"
            "\n"
            "مادة (812)\n\nيكون الشراء للوكيل.\n\nمادة (813)\nنص.\n"
        )
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == ["art_811", "art_812", "art_813"], eids

    def test_single_soft_wrap_still_suppressed(self) -> None:
        text = (
            "بعد الاطلاع على قانون المرصد رقم (٤١) لسنة ٢٠٤٢ ولاسيما\n"
            "المادة (٥٥) منه، صدر القانون التالي.\nمادة (٢)\nنص.\n"
        )
        assert [a.akn_eid for a in self._scan(text)] == ["art_2"]


class TestLetteredAnnex:
    """Abjad-lettered annexes ("الملحق (أ)") anchor as schedules with an
    ASCII rank-derived eid."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_lettered_annex_anchors_as_schedule(self) -> None:
        text = (
            "مادة (9)\nنص.\nمادة (10)\nنص أخير.\n\n"
            "الملحق (أ)\n\n"
            "يعدل الملحق (أ) من النظام الأصلي، ليصبح على النحو الآتي:\n"
            "1. الأسقف المالية للجهات المشترية.\n"
        )
        anchors = self._scan(text)
        annex = anchors[-1]
        assert annex.kind == "schedule"
        assert annex.number == "أ"
        assert annex.akn_eid == "schedule_1"
        # The prose mention inside the annex must not anchor a second time.
        assert sum(1 for a in anchors if a.kind == "schedule") == 1


class TestMisreadArticleKeywordRecovery:
    """A keyword OCR mangled (مادة → إدارة) leaves a numeric gap; the bare
    "word (N)" line carrying exactly the missing number is the lost marker."""

    def _scan(self, text: str) -> list[StructuralAnchor]:
        rx = build_anchor_regex(_QANUN_CONFIG, "act")
        return scan_anchors(text, rx, country="xx")

    def test_misread_markers_recovered_and_toc_twins_dropped(self) -> None:
        # AC 1/2005: body arts 16/17 read as "إدارة"; their TOC twins then
        # survived dedup and absorbed the TOC tail.
        text = (
            "مادة (١٥) محاكمة الأعضاء\n"
            "مادة (١٦) إقرارات الذمة المالية\n"
            "مادة (١٧) الاشتباه بوجود جريمة\n"
            "مادة (١٨) تقديم المعلومات\n"
            "مادة (١٥)\nنص المادة الخامسة عشرة هنا.\n"
            "إدارة (١٦)\nإقرارات الذمة المالية\nنص المادة السادسة عشرة هنا.\n"
            "إدارة (١٧)\nالاشتباه بوجود جريمة\nنص المادة السابعة عشرة هنا.\n"
            "مادة (١٨)\nنص المادة الثامنة عشرة هنا.\n"
        )
        anchors = self._scan(text)
        assert [a.akn_eid for a in anchors] == ["art_15", "art_16", "art_17", "art_18"]
        # Survivors are the body markers, not the TOC entries (an anchor's
        # match starts on the preceding newline, so line is one lower).
        assert all(a.line >= 4 for a in anchors), [(a.akn_eid, a.line) for a in anchors]

    def test_gap_without_bare_marker_lines_left_alone(self) -> None:
        text = "مادة (1)\nنص المادة الأولى.\nمادة (4)\nنص المادة الرابعة.\n"
        assert [a.akn_eid for a in self._scan(text)] == ["art_1", "art_4"]


class TestPlainContainerNesting:
    """Baseline Arabic nesting, with no recovery pass involved.

    Kept when `recover_container_twin` went: it never exercised that pass and
    still passes without it. What it pins is that ordinary parts and articles
    nest in document order, which is the thing a future recovery pass would
    break first."""

    def _scan(self, text: str):
        from codify.pipeline.enrich.anchors import cached_regex, scan_anchors

        return scan_anchors(text, cached_regex("xx", "act"), country="xx", doctype="act")

    def test_a_lone_container_whose_articles_never_recur_is_left_alone(self) -> None:
        text = (
            "الباب الأول تعاريف\n"
            "مادة (1)\nنص التعريف هنا.\nمادة (2)\nنص آخر هنا.\n"
            "الباب الثاني العقوبات\n"
            "مادة (3)\nنص العقوبة هنا.\n"
            "الباب الثالث أحكام ختامية\n"
            "مادة (4)\nنص الحكم الختامي هنا.\n"
        )
        eids = [a.akn_eid for a in self._scan(text)]
        assert eids == [
            "part_1",
            "part_1__art_1",
            "part_1__art_2",
            "part_2",
            "part_2__art_3",
            "part_3",
            "part_3__art_4",
        ], eids


class TestRecoveryPassGuards:
    """Review-driven negatives for the orphan-children recovery pass."""

    def _scan(self, text: str):
        from codify.pipeline.enrich.anchors import cached_regex, scan_anchors

        return scan_anchors(text, cached_regex("xx", "act"), country="xx", doctype="act")

    def test_gap_wider_than_cap_not_bridged(self) -> None:
        body = (
            "مادة (1)\nنص.\n"
            + "بند (2)\nسطر.\nبند (3)\nسطر.\nبند (4)\nسطر.\nبند (5)\nسطر.\nبند (6)\nسطر.\n"
            + "مادة (7)\nنص.\n"
        )
        eids = [a.akn_eid for a in self._scan(body) if a.kind == "article"]
        assert eids == ["art_1", "art_7"]


def test_a_measured_full_score_is_distinguishable_from_an_unmeasurable_one() -> None:
    """The whole point of the tri-state ratio: everything found and nothing to
    find must not read the same in the bundle."""
    from codify.pipeline.enrich.anchors import anchor_coverage, cached_regex, scan_anchors

    config = _QANUN_CONFIG
    assert config is not None
    regex = cached_regex("xx", "qanun")

    complete = "مادة ١\nنص.\n\nمادة ٢\nنص.\n"
    cov = anchor_coverage(
        complete,
        scan_anchors(complete, regex, country="xx", doctype="qanun"),
        config,
        "qanun",
        "article",
    )
    assert cov.ratio == 1.0

    # No مادة keyword anywhere: nothing to measure against.
    unmeasurable = "اعفاء من دفع رسوم ١ - يعفى من دفع الرسوم .\nبدء سريان ٢ - يسري .\n"
    cov = anchor_coverage(
        unmeasurable,
        scan_anchors(unmeasurable, regex, country="xx", doctype="qanun"),
        config,
        "qanun",
        "article",
    )
    assert cov.ratio is None


def test_uk_amendment_quoted_block_is_not_anchored() -> None:
    """Straight-quote amendment blocks in UK Acts must not produce host anchors.

    PDF extraction from legislation.gov.uk preserves straight " rather than
    curly " for substitution blocks, so _quote_mask alone does not suppress
    them.  _uk_amendment_mask detects the em-dash → line-start " pattern and
    masks the block; the host section's own later subsections must survive.
    """
    from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity

    text = (
        "3 Substitution of Articles 5 and 6\n"
        "(1) Articles 5 and 6 are revoked.\n"
        "(2) After Article 4 insert—\n"
        "\n"
        '"5 Officeholder duties\n'
        "(1) An officeholder must notify creditors.\n"
        "(2) The notice must include the information specified.\n"
        "(3) The officeholder may apply to the court.\n"
        "(4) The court may extend the period.\n"
        '(5) Nothing in this Article limits any duty."\n'
        "\n"
        "(3) Article 8 is amended as follows.\n"
        "(4) In paragraph (1), omit “only”.\n"
        "\n"
        "4 Minor amendments\n"
        "(1) Schedule 1 is amended as follows.\n"
        '(2) In paragraph 3, for "ten" substitute "twenty".\n'
        "(3) In paragraph 4, omit sub-paragraph (b).\n"
        "(4) After paragraph 6 insert the paragraph in Schedule 2.\n"
    )
    regex = cached_regex("gb", "act")
    scan = scan_anchors_with_ambiguity(text, regex, country="gb", doctype="act")

    blocking = [s for s in scan.ambiguity if s.blocking]
    assert blocking == [], [s.detail for s in blocking]

    # The quoted article 5 with subsections (1)-(5) must be invisible.
    sections = {a.number for a in scan.anchors if a.kind == "section"}
    assert sections == {"3", "4"}, f"phantom sections leaked: {sections}"

    # Section 3's own (3) and (4) must survive, they follow the quoted block.
    sec3_subsecs = {
        a.number for a in scan.anchors if a.kind == "subsection" and a.parent_eid == "sec_3"
    }
    assert "3" in sec3_subsecs, "section 3's own (3) was dropped"
    assert "4" in sec3_subsecs, "section 3's own (4) was dropped"

    # Inline short quotes in section 4 must not suppress its subsections.
    sec4_subsecs = {
        a.number for a in scan.anchors if a.kind == "subsection" and a.parent_eid == "sec_4"
    }
    assert sec4_subsecs == {"1", "2", "3", "4"}


def test_the_uk_lane_labels_its_two_scanners_separately() -> None:
    """`scan_anchors` stamps "regex" twice with the UK scan extended in between,
    so only `_stamp`'s already-named guard keeps the second call from
    relabelling every UK anchor. A wrong answer in the regex-only case still
    reads "regex", so this is the case that can catch it."""
    anchors = scan_anchors(UK_SAMPLE, build_anchor_regex(load_config("gb"), "act"), country="gb")
    passes = {a.source_pass for a in anchors}
    assert passes == {"regex", "uk_provisions"}
    assert all(a.source_pass == "uk_provisions" for a in anchors if a.kind == "subsection")


def test_a_fallback_scan_labels_every_anchor_it_produced() -> None:
    """Both fallbacks replace the anchor list wholesale, and the ordinal one only
    fires when the heading one returned nothing."""
    from codify.pipeline.enrich.anchors import cached_regex

    regex = cached_regex("xx", "qanun")

    # Same shape as test_numbered_heading_pattern_fallback; the fallback needs
    # three hits before it fires.
    headings = (
        "تعاريف\n١ - في هذا النظام تعني الكلمات التالية.\n\n"
        "اعداد\n٢ - تعد الخارطة على اساس النقاط الحكومية.\n\n"
        "مضمون الخارطة\n٣ - تحدد الخارطة الحدود والإرتدادات.\n"
    )
    anchors = scan_anchors(headings, regex, country="xx", doctype="qanun")
    assert anchors and all(a.source_pass == "numbered_heading_fallback" for a in anchors)

    ordinals = "أولاً: نص أول .\nثانياً: نص ثان .\nثالثاً: نص ثالث .\n"
    anchors = scan_anchors(ordinals, regex, country="xx", doctype="qanun")
    assert anchors and all(a.source_pass == "ordinal_list_fallback" for a in anchors)


class TestMarkerNumberSeparator:
    """What may sit between a keyword and its number, and what may follow it.
    Each synthetic shape below exercises a previously unmatched marker."""

    def _articles(self, text: str) -> list[str]:
        from codify.pipeline.enrich.anchors import cached_regex

        regex = cached_regex("xx", "qanun")
        anchors = scan_anchors(text, regex, country="xx", doctype="qanun")
        return [a.number or "" for a in anchors if a.kind == "article"]

    def test_an_em_dash_before_the_number_is_a_separator(self) -> None:
        """The oldest shape in the corpus: 287 markers, none claimed."""
        assert self._articles("مادة— ٢٧\nنص.\n\nمادة— ٢٨\nنص.\n") == ["٢٧", "٢٨"]

    def test_an_unbalanced_closing_paren_is_a_separator(self) -> None:
        """RTL extraction drops the opener: 269 markers, none claimed."""
        assert self._articles("مادة )1\nنص.\n\nالمادة )2\nنص.\n") == ["1", "2"]

    def test_a_rubric_fused_to_the_number_still_anchors(self) -> None:
        """A digit followed by an Arabic letter is not a word boundary, which
        hid the old column layout: 348 articles over 87 documents."""
        text = "مادة  — ٢تفسير اصطلاح\nنص.\n\nمادة  — ٣صلاحية المندوب\nنص.\n"
        assert self._articles(text) == ["٢", "٣"]

    def test_a_number_is_still_never_split(self) -> None:
        """The guarantee the word boundary was really protecting."""
        assert self._articles("مادة 12\nنص.\n") == ["12"]
        assert self._articles("مادة ١٢٣\nنص.\n") == ["١٢٣"]

    def test_the_coverage_denominator_uses_the_same_separator(self) -> None:
        """The gate carried its own copy; drift there is a false pass."""
        from codify.pipeline.enrich.anchors import _marker_numbers

        text = "مادة )1\nنص.\n\nمادة— 2\nنص.\n"
        assert _marker_numbers(text, _QANUN_CONFIG, "qanun", "article") == {"1", "2"}

    def test_the_keyword_survives_a_non_space_separator(self) -> None:
        """Splitting on whitespace yielded the keyword `مادة-5`."""
        from codify.pipeline.enrich.anchors import cached_regex

        regex = cached_regex("xx", "qanun")
        anchors = scan_anchors("مادة-5\nنص.\n", regex, country="xx", doctype="qanun")
        assert [a.keyword for a in anchors] == ["مادة"]


class TestMangledKeywordRecovery:
    """Keywords OCR corrupted past the alias set. 2,340 of the 2,691 unclaimed
    synthetic marker lines; each test names what stays refused."""

    def _scan(self, text: str) -> list[tuple[str, str]]:
        from codify.pipeline.enrich.anchors import cached_regex

        regex = cached_regex("xx", "qanun")
        return [
            (a.kind, a.number or "")
            for a in scan_anchors(text, regex, country="xx", doctype="qanun")
        ]

    def test_a_dropped_tail_still_names_its_level(self) -> None:
        """`الماد` and `ماد` are the two commonest manglings, 950 lines between them."""
        text = "مادة (1)\nنص.\n\nالماد (2)\nنص.\n\nماد (3)\nنص.\n"
        assert self._scan(text) == [("article", "1"), ("article", "2"), ("article", "3")]

    def test_a_substituted_letter_still_names_its_level(self) -> None:
        text = "مادة (1)\nنص.\n\nالمادت (2)\nنص.\n\nماتة (3)\nنص.\n"
        assert self._scan(text) == [("article", "1"), ("article", "2"), ("article", "3")]

    def test_an_ordinary_word_is_not_a_marker(self) -> None:
        """`منها` reads as a marker line but is prose. It must stay refused, or
        the pass invents a provision boundary."""
        assert self._scan("مادة (1)\nنص.\n\nمنها (2)\nنص.\n") == [("article", "1")]

    def test_a_longer_word_is_not_a_shorter_keyword(self) -> None:
        """`Party` is `Part` plus a letter. Admitting an insertion read an
        ordinary English noun as a container keyword in every Latin-script
        jurisdiction; OCR truncates and substitutes, it does not insert."""
        from codify.pipeline.enrich.anchors import _kind_for_mangled_keyword, keyword_aliases

        aliases = keyword_aliases(load_config("gb"), "act")
        for word in ("Party", "Parties", "Partly", "Pat"):
            assert _kind_for_mangled_keyword(word, aliases) is None, word

    def test_the_definite_article_alone_names_nothing(self) -> None:
        """`ال` prefixes every Arabic alias, so it cannot choose between them.
        The builtin schedule aliases put `الملحق` in every jurisdiction, so a
        length floor alone would let it resolve uniquely outside Arabic."""
        assert self._scan("مادة (1)\nنص.\n\nال (2)\nنص.\n") == [("article", "1")]

    def test_a_short_prefix_of_a_long_alias_names_nothing(self) -> None:
        """Two letters of `Subparagraph` or `Chapter` is not evidence; OCR drops
        a keyword's tail, it does not abbreviate it."""
        from codify.pipeline.enrich.anchors import _kind_for_mangled_keyword, keyword_aliases

        for country, doctype in (("gb", "act"), ("eu", "directive"), ("ua", "law")):
            aliases = keyword_aliases(load_config(country), doctype)
            for token in ("ال", "Ch", "Se", "Sub", "Sec", "Chap"):
                assert _kind_for_mangled_keyword(token, aliases) is None, (country, token)

    def test_a_line_carrying_more_than_a_marker_is_left_alone(self) -> None:
        """The whole-line shape is what makes a mangled keyword safe to read."""
        assert self._scan("مادة (1)\nنص.\n\nالماد (2) وما بعدها من أحكام\nنص.\n") == [
            ("article", "1")
        ]

    def test_a_recovered_marker_names_the_pass_that_found_it(self) -> None:
        """Counts use this field; an unattributed anchor is invisible."""
        from codify.pipeline.enrich.anchors import cached_regex

        anchors = scan_anchors(
            "مادة (1)\nنص.\n\nالماد (2)\nنص.\n",
            cached_regex("xx", "qanun"),
            country="xx",
            doctype="qanun",
        )
        recovered = [a for a in anchors if a.keyword == "الماد"]
        assert [a.source_pass for a in recovered] == ["recover_misread_marker"]


class TestNumberEndBoundary:
    """`_NUM_PATTERN` matches Roman numerals and a lone Latin letter, so relaxing
    the trailing boundary for Arabic reached every Latin-script jurisdiction."""

    def _scan(self, country: str, doctype: str, text: str) -> list[tuple[str, str]]:
        from codify.pipeline.enrich.anchors import cached_regex

        regex = cached_regex(country, doctype)
        return [
            (a.kind, a.number or "")
            for a in scan_anchors(
                "x\n\n" + text + "\nbody.\n", regex, country=country, doctype=doctype
            )
        ]

    @pytest.mark.parametrize(
        ("country", "doctype", "text"),
        [
            ("gb", "act", "Chapter Definitions"),
            ("gb", "act", "Part Miscellaneous"),
            ("gb", "act", "Section Interpretation"),
            ("gb", "act", "SCHEDULE\nRegulations revoked"),
            ("al", "ligj", "KREU Vendimet e gjykates"),
            ("ua", "law", "Стаття Хімічні речовини"),
        ],
    )
    def test_a_word_after_a_keyword_is_not_its_number(
        self, country: str, doctype: str, text: str
    ) -> None:
        """`Chapter Definitions` anchored as chapter `D`, stealing the first
        letter of the rubric and reparenting everything below it."""
        assert self._scan(country, doctype, text) == []

    def test_a_hyphen_inside_a_word_is_not_a_separator(self) -> None:
        """`Part-time workers` anchored as part `t`."""
        assert self._scan("gb", "act", "Part-time workers") == []

    @pytest.mark.parametrize(
        ("country", "doctype", "text", "expected"),
        [
            ("gb", "act", "PART II", ("part", "II")),
            ("al", "ligj", "Neni 3", ("article", "3")),
            ("xx", "qanun", "مادة  — ٢تفسير اصطلاح", ("article", "٢")),
            ("xx", "qanun", "مادة )1", ("article", "1")),
        ],
    )
    def test_a_real_marker_still_anchors(
        self, country: str, doctype: str, text: str, expected: tuple[str, str]
    ) -> None:
        assert expected in self._scan(country, doctype, text)

    def test_a_prose_line_does_not_enter_the_coverage_denominator(self) -> None:
        """`_marker_numbers` is case-insensitive where the scanner is not, so it
        harvested `i` from `section introduces` and failed a clean document."""
        from codify.pipeline.enrich.anchors import _marker_numbers

        text = "Section 1\nbody\nsection introduces the concept\nSection 2\nbody\n"
        assert _marker_numbers(text, load_config("gb"), "act", "section") == {"1", "2"}


class TestRecoveryRespectsTheScansRefusals:
    """The mangled-keyword pass is a second sweep of the text, so it has to
    honour the guards the first one applied or it overrules them."""

    def _scan(self, text: str) -> list[tuple[str, str, str]]:
        from codify.pipeline.enrich.anchors import cached_regex

        return [
            (a.kind, a.number or "", a.source_pass or "")
            for a in scan_anchors(text, cached_regex("xx", "qanun"), country="xx", doctype="qanun")
        ]

    def test_a_quoted_replacement_marker_stays_refused(self) -> None:
        """A marker inside amendment quotation is not a host provision;
        reclaiming it would give the host an article it does not have."""
        text = "مادة (1)\nنص.\n\nيستبدل النص التالي: « مادة (7)\nنص مقتبس.\n »\n\nمادة (2)\nنص.\n"
        assert self._scan(text) == [("article", "1", "regex"), ("article", "2", "regex")]

    def test_an_unclosed_low_nine_quote_does_not_swallow_the_rest(self) -> None:
        """Nothing in the close set can close a low-9 quote: it pairs with a
        curly quote, not a guillemet. One stray „ from source noise therefore
        masked every marker to the end of the file, and each was dropped as
        quoted text with no gate tripping. On one long code that was 61% of the
        document."""
        text = "مادة (1)\nنص.\n\nسطر فيه ضجيج „ هنا\n\nمادة (2)\nنص.\n\nمادة (3)\nنص.\n"
        assert self._scan(text) == [
            ("article", "1", "regex"),
            ("article", "2", "regex"),
            ("article", "3", "regex"),
        ]

    def test_an_unclosed_guillemet_does_not_swallow_the_rest(self) -> None:
        """Guillemets pair in principle and not in scraped text. The same
        Corporation Code carries a lone « that masked a further 40,701
        characters after the low-9 one was bounded."""
        text = "مادة (1)\nنص.\n\nسطر فيه ضجيج « هنا\n\nمادة (2)\nنص.\n\nمادة (3)\nنص.\n"
        assert self._scan(text) == [
            ("article", "1", "regex"),
            ("article", "2", "regex"),
            ("article", "3", "regex"),
        ]

    def test_a_stray_opener_inside_a_real_quotation_does_not_free_it(self) -> None:
        """Pairing every opener on one stack lets a » pop a „, so the genuine
        « is the one reported unclosed and the markers it holds are released as
        host articles. Each opener pairs only against the closer it takes."""
        text = (
            "مادة (1)\nنص.\n\nيستبدل النص التالي: « مادة (7)\nنص فيه „ ضجيج.\n\n"
            "بقية النص المقتبس.\n »\n\nمادة (2)\nنص.\n"
        )
        assert self._scan(text) == [("article", "1", "regex"), ("article", "2", "regex")]

    def test_a_cover_listing_is_not_reclaimed_as_a_body_marker(self) -> None:
        """Everything before the cover listing ends is a listing, so a number
        appearing only there must not become a body anchor."""
        from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity

        text = "الفهرس\nمادة (1)\nمادة (2)\nمادة (3)\n\nالمتن\n\nمادة (1)\nنص.\n\nمادة (2)\nنص.\n"
        result = scan_anchors_with_ambiguity(
            text,
            cached_regex("xx", "qanun"),
            country="xx",
            doctype="qanun",
            toc_end=text.index("المتن"),
        )
        assert [a.number for a in result.anchors] == ["1", "2"]


def test_anchor_regex_reproducible_across_hashseeds():
    # The compiled pattern must be identical across processes regardless of
    # PYTHONHASHSEED, equal-length aliases (e.g. PART/Part) previously tied on
    # set iteration order (#658).
    import os
    import subprocess
    import sys

    code = (
        "from codify.pipeline.enrich.anchors import cached_regex;"
        "print(cached_regex('xa', 'act').pattern)"
    )

    def _pattern(seed: str) -> str:
        env = {key: os.environ[key] for key in ("PATH", "PYTHONPATH", "HOME") if key in os.environ}
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        return out.stdout.strip()

    patterns = {_pattern(seed) for seed in ("0", "1", "12345", "99991")}
    assert len(patterns) == 1, "anchor regex varies across PYTHONHASHSEED"


class TestDeclaredOrdinalWords:
    """Bahasa Bagian are numbered with ordinal words, which the number
    pattern did not match: hundreds of Bagian lines yielded a handful of `part`
    anchors, so repeated `Paragraf 1`s had no distinct parent to scope them."""

    def _regex(self) -> re.Pattern[str]:
        cfg = load_config("xl")
        assert cfg is not None
        return build_anchor_regex(cfg, "permen", boundary=cfg.structuring.marker_boundary)

    def test_an_ordinal_word_anchors_and_folds_to_its_integer(self) -> None:
        """Both halves matter: matching, and folding so the eId is `part_1`
        rather than `part_kesatu`, which AKN's ASCII rule forbids anyway."""
        text = (
            "BAB I\nUMUM\n\nBagian Kesatu\nRuang\n\nPasal 1\nIsi.\n\n"
            "Bagian Kesebelas\nLain\n\nPasal 2\nIsi.\n"
        )
        parts = [
            a
            for a in scan_anchors(text, self._regex(), country="xl", doctype="permen")
            if a.kind == "part"
        ]
        assert [a.akn_eid for a in parts] == ["chp_I__part_1", "chp_I__part_11"]

    def test_repeated_divisions_are_scoped_by_their_own_part(self) -> None:
        """The failure this exists to prevent. With no `part` anchor the 29
        separate `Paragraf 1`s in one regulation collapsed onto each other,
        because they had no distinct parent to scope them."""
        text = (
            "BAB I\nUMUM\n\nBagian Kesatu\nRuang\n\nParagraf 1\nSatu\n\nPasal 1\nIsi.\n\n"
            "Bagian Kedua\nLingkup\n\nParagraf 1\nSatu\n\nPasal 2\nIsi.\n"
        )
        from codify.pipeline.enrich.anchors import scan_anchors_with_ambiguity

        scan = scan_anchors_with_ambiguity(text, self._regex(), country="xl", doctype="permen")
        divisions = [a.akn_eid for a in scan.anchors if a.kind == "division"]
        assert divisions == ["chp_I__part_1__dvs_1", "chp_I__part_2__dvs_1"]
        assert not [s for s in scan.ambiguity if s.blocking]

    def test_lowercase_prose_does_not_anchor(self) -> None:
        """`Bagian ketiga terendah dari lereng` is prose. Case is what parts it
        from the heading `Bagian Ketiga`, and the scanner is case-sensitive."""
        text = "BAB I\nUMUM\n\nPasal 1\nBagian ketiga terendah dari lereng.\n"
        anchors = scan_anchors(text, self._regex(), country="xl", doctype="permen")
        assert not [a for a in anchors if a.kind == "part"]

    @pytest.mark.parametrize("country", [None, "ua", "al", "it"])
    def test_a_jurisdiction_declaring_none_compiles_the_base_pattern(
        self, country: str | None
    ) -> None:
        """The words are declared per jurisdiction, so a jurisdiction that
        declares none must compile exactly what it compiled before."""
        cfg = _QANUN_CONFIG if country is None else load_config(country)
        assert cfg is not None
        rx = build_anchor_regex(cfg, cfg.default_document_class or "act")
        from codify.pipeline.enrich.anchors import _num_pattern_with

        assert _num_pattern_with({}) in rx.pattern
        assert "Kesatu" not in rx.pattern

    def test_the_fold_does_not_depend_on_what_was_compiled_first(self) -> None:
        """The fold was populated as a side effect of compiling a regex, so the
        same `<num>` derived two different eIds depending on process history.
        A repair worker that had not compiled the Bahasa pattern renamed
        every Bagian off its ingest-time eId and rewrote the references."""
        from codify.pipeline.enrich.anchors import _normalise_number

        cold = _normalise_number("Kesatu")
        cfg = load_config("xl")
        assert cfg is not None
        build_anchor_regex(cfg, "permen", boundary=cfg.structuring.marker_boundary)
        assert cold == _normalise_number("Kesatu") == "1"

    def test_no_two_words_fold_to_the_same_number(self) -> None:
        """Two words on one integer collide, and the TOC dedup then silently
        drops one heading and everything scoped under it."""
        cfg = load_config("xl")
        assert cfg is not None
        values = list(cfg.structuring.ordinal_words.values())
        assert len(values) == len(set(values))

    def test_a_compound_ordinal_may_wrap_between_its_words(self) -> None:
        """A centred heading wraps, and a literal space in the alternation fell
        back to the shorter word: `Bagian Kedua\nBelas` numbered itself 2 and
        collided with a real Bagian Kedua elsewhere in the document."""
        text = "BAB I\nU\n\nBagian Kedua\nBelas\nRuang\n\nPasal 1\nIsi.\n"
        parts = [
            a
            for a in scan_anchors(text, self._regex(), country="xl", doctype="permen")
            if a.kind == "part"
        ]
        assert [a.akn_eid for a in parts] == ["chp_I__part_12"]


class TestAmendmentItemScope:
    """An omnibus restates dozens of statutes, each numbering from Pasal 1.
    With nothing between them in the ancestor chain the dedup read the repeats
    as TOC twins and collapsed them, orphaning their Ayat to pile and collide.
    `Angka` is the item that introduces one amended statute's article."""

    OMNIBUS = (
        "Bab I\nKETENTUAN UMUM\n\n"
        "Pasal 31\nAngka 1\nPasal 6\nAyat (1)\nKetentuan pertama.\n\n"
        "Angka 2\nPasal 19\nAyat (1)\nKetentuan kedua.\n\n"
        "Pasal 32\nAngka 1\nPasal 6\nAyat (1)\nKetentuan ketiga.\n"
    )

    def _scan(self, text: str, country: str = "xl", doctype: str = "act") -> list[StructuralAnchor]:
        rx = build_anchor_regex(load_config(country), doctype)
        return scan_anchors(text, rx, country=country, doctype=doctype)

    def test_an_item_parents_the_article_it_introduces(self) -> None:
        eids = {a.akn_eid for a in self._scan(self.OMNIBUS)}
        assert "chp_I__art_31__indent_1__art_6" in eids, sorted(eids)
        assert "chp_I__art_31__indent_1__art_6__para_1" in eids, sorted(eids)

    def test_two_statutes_carrying_the_same_article_both_survive(self) -> None:
        """The regression this exists to prevent. Both hosts amend a Pasal 6;
        one used to be dropped as the other's twin, and its Angka reparented
        onto the previous host, colliding there."""
        sixes = [a.akn_eid for a in self._scan(self.OMNIBUS) if a.number == "6"]
        assert sixes == [
            "chp_I__art_31__indent_1__art_6",
            "chp_I__art_32__indent_1__art_6",
        ], sixes

    def test_an_ordinary_item_list_keeps_its_declared_depth(self) -> None:
        """`Angka` is the deepest subdivision everywhere except in front of an
        article, so a plain numbered list must not become a container."""
        text = (
            "Bab I\nUMUM\n\nPasal 1\nSyarat:\nHuruf a\nSatu.\nAngka 1\nRincian.\nAngka 2\nLain.\n"
        )
        items = [a for a in self._scan(text) if a.kind == "indent"]
        assert items and not any(a.amendment_item for a in items)
        assert all(a.akn_eid.startswith("chp_I__art_1__point_a__indent_") for a in items), [
            a.akn_eid for a in items
        ]

    def test_the_host_article_after_a_list_is_not_swallowed(self) -> None:
        """Anchor adjacency is not enough. The last item of an ordinary list
        and the host's next article are consecutive anchors too, so adjacency
        alone nested `art_2` under the list and reparented the rest of the
        document. The article has to follow the marker in the source."""
        text = (
            "Bab I\nUMUM\n\nPasal 1\nSyarat:\nHuruf a\nSatu.\nAngka 1\nRincian.\n"
            "Angka 2\nLain.\n\nPasal 2\nKetentuan berikutnya.\n\nPasal 3\nMulai berlaku.\n"
        )
        anchors = self._scan(text)
        assert not any(a.amendment_item for a in anchors)
        arts = [a.akn_eid for a in anchors if a.kind == "article"]
        assert arts == ["chp_I__art_1", "chp_I__art_2", "chp_I__art_3"], arts

    def test_a_jurisdiction_declaring_no_item_kind_is_unchanged(self) -> None:
        cfg = _QANUN_CONFIG
        assert cfg is not None and cfg.amendments is not None
        assert not cfg.amendments.amendment_item_kind

    def test_perppu_declares_the_same_hierarchy_as_an_act(self) -> None:
        """The drafting manual applies one hierarchy to every instrument
        type. `perppu` omitted Paragraf and Angka, so the scoping had nothing
        to work with on a Perppu that is itself an omnibus."""
        cfg = load_config("xl")
        assert cfg is not None
        act = [h.akn_element for h in cfg.get_document_class("act").hierarchy]
        perppu = [h.akn_element for h in cfg.get_document_class("perppu").hierarchy]
        assert perppu == [e for e in act if e != "book"], perppu


class TestMarkerTolerances:
    """Scanned Bahasa sources lose the space between a marker keyword and
    its number (`Pasal24`, `Angka22`), over half of all unmatched markers.
    Declared per jurisdiction, so every other compiled pattern is untouched."""

    def _scan(self, text: str, country: str = "xl") -> list[StructuralAnchor]:
        rx = build_anchor_regex(load_config(country), "act")
        return scan_anchors(text, rx, country=country, doctype="act")

    def test_a_marker_that_lost_its_separator_still_anchors(self) -> None:
        got = [(a.kind, a.number) for a in self._scan("Bab I\nU\n\nPasal24\nIsi.\n")]
        assert ("article", "24") in got, got

    def test_a_word_that_merely_opens_with_the_keyword_does_not(self) -> None:
        """`Pasalnya` and `Angkatan` are ordinary words. Losing the separator
        requirement must not turn every such word into a marker."""
        for text in (
            "Bab I\nU\n\nPasal 1\nPasalnya diubah.\n",
            "Bab I\nU\n\nPasal 1\nAngkatan bersenjata.\n",
        ):
            nums = [a.number for a in self._scan(text) if a.kind in ("article", "indent")]
            assert nums == ["1"], (text, nums)

    def test_the_number_must_be_the_whole_line_without_a_separator(self) -> None:
        """Matching a leading digit and ignoring the rest would invent
        `article 2` from `Pasal24 dan seterusnya`, and a wrong number collides
        with a real article and displaces it."""
        anchors = self._scan("Bab I\nU\n\nPasal 1\nPasal24 dan seterusnya.\n")
        got = [a.number for a in anchors if a.kind == "article"]
        assert got == ["1"], got

    def test_a_keyword_the_scan_misread_still_anchors(self) -> None:
        """`Pasal` reads `Pasai`, and `Angka` reads `Ang)<a` where the ligature
        breaks. The number is undamaged in this class, so nothing is guessed."""
        got = [(a.kind, a.number) for a in self._scan("Bab I\nU\n\nPasai 5\nIsi.\n")]
        assert ("article", "5") in got, got
        got = [
            (a.kind, a.number) for a in self._scan("Bab I\nU\n\nPasal 1\nAng)<a 22\nPasal 6\nX.\n")
        ]
        assert ("indent", "22") in got, got

    def test_glyph_tolerance_does_not_admit_prose(self) -> None:
        anchors = self._scan("Bab I\nU\n\nPasal 1\nPasai. Ini prosa biasa.\n")
        assert [a.number for a in anchors if a.kind == "article"] == ["1"]

    def test_a_misread_digit_folds_to_the_number_it_was(self) -> None:
        """`l`, `L` and `I` read for 1 and `O` for 0: `Pasal l1` is article 11,
        `Pasal L27` is 127. The eid has to carry the true number, or the
        article collides with whatever really holds the damaged one."""
        for text, eid in (
            ("Pasal l1\nIsi.\n", "art_11"),
            ("Pasal L27\nIsi.\n", "art_127"),
            ("Pasal l0O\nIsi.\n", "art_100"),
        ):
            got = [a.akn_eid for a in self._scan(text) if a.kind == "article"]
            assert got == [eid], (text, got)

    def test_a_roman_article_stays_roman(self) -> None:
        """`Pasal I` and `Pasal II` are the two-Pasal amending structure, not a
        misread 1. A run with no real digit in it is left alone, which is what
        keeps the two apart."""
        for text, eid in (("Pasal I\nX.\n", "art_I"), ("Pasal II\nX.\n", "art_II")):
            got = [a.akn_eid for a in self._scan(text) if a.kind == "article"]
            assert got == [eid], (text, got)

    def test_a_letter_suffixed_article_keeps_its_suffix(self) -> None:
        got = [a.akn_eid for a in self._scan("Pasal 14A\nX.\n") if a.kind == "article"]
        assert got == ["art_14A"], got

    def test_a_number_the_scan_split_is_rejoined(self) -> None:
        """`Pasal 2 1` is article 21. Taking the leading group alone reads it as
        article 2, which then displaces the real article 2 and orphans
        everything the true article 21 held."""
        for text, eid in (("Pasal 2 1\nCukup jelas.\n", "art_21"), ("Pasal 1 12\nX.\n", "art_112")):
            got = [a.akn_eid for a in self._scan(text) if a.kind == "article"]
            assert got == [eid], (text, got)

    def test_a_citation_with_a_tail_still_reads_as_its_own_article(self) -> None:
        """The rejoin only fires when the groups run to the end of the line, so
        `Pasal 5 ayat (2) huruf a` keeps reading as article 5 rather than
        swallowing the subdivision numbers into one."""
        got = [
            a.akn_eid for a in self._scan("Pasal 5 ayat (2) huruf a\nX.\n") if a.kind == "article"
        ]
        assert got == ["art_5"], got

    def test_a_jurisdiction_declaring_no_tolerance_is_unchanged(self) -> None:
        cfg = _QANUN_CONFIG
        assert cfg is not None and not cfg.structuring.marker_tolerances
        assert (
            "(?P<sep>" not in build_anchor_regex(cfg, cfg.default_document_class or "act").pattern
        )


class TestMarkerToleranceScoping:
    """A tolerance must reach exactly as far as its declaration. These pin the
    two ways it silently did not."""

    def test_the_glyph_fold_does_not_leak_into_other_jurisdictions(self) -> None:
        """`_normalise_number` is country-agnostic and feeds repair eId
        derivation as well as the scan. Folding inside it turned a clean
        letter-suffixed `5I`, `5O` or `14T` into 51, 50 and 147 everywhere,
        changing eIds for jurisdictions that declare no tolerance at all."""
        from codify.pipeline.enrich.anchors import _normalise_number

        for num in ("5I", "5O", "14T", "15A"):
            assert _normalise_number(num) == num, num

    def test_a_split_number_stays_digit_only_without_the_glyph_tolerance(self) -> None:
        """Declaring `split_number` alone must not admit glyph damage: two
        tolerances with one blast radius cannot be reasoned about."""
        cfg = load_config("xl")
        assert cfg is not None
        original = cfg.structuring.marker_tolerances
        try:
            cfg.structuring.marker_tolerances = ["split_number"]
            rx = build_anchor_regex(cfg, "act")
            # `Pasal L` still matches through the base grammar's lone-letter
            # alternative, which predates this. What must not happen is the
            # split branch joining `L 2` into one glyph-damaged number.
            glyphed = rx.search("\nPasal L 2\nX.\n")
            assert glyphed is not None and glyphed.group("num") == "L", glyphed
            split = rx.search("\nPasal 2 1\nX.\n")
            assert split is not None and split.group("num") == "2 1", split
        finally:
            cfg.structuring.marker_tolerances = original

    def test_the_coverage_denominator_sees_recovered_markers(self) -> None:
        """`_marker_numbers` is the denominator the coverage gate divides by. A
        recovered marker that raises `captured` without raising `expected`
        inflates coverage past the gate and hides genuinely missing ones."""
        from codify.pipeline.enrich.anchors import _marker_numbers

        cfg = load_config("xl")
        assert cfg is not None
        assert "24" in _marker_numbers("Pasal24\nIsi.\n", cfg, "act", "article")


# ── Omnibus restatement: which duplicate survives ────────────────────────────


def _xl_regex():
    config = load_config("xl")
    assert config is not None
    return build_anchor_regex(config, "uu")


def _babs(*numbers: str) -> str:
    """A document whose chapters carry the given numbers in the given order."""
    return "".join(
        f"BAB {num}\nKETENTUAN {i + 1}\n\nPasal {i + 1}\nIsi bab {i + 1}.\n\n"
        for i, num in enumerate(numbers)
    )


def _kept_chapter(text: str, number: str) -> int:
    anchors = scan_anchors(text, _xl_regex(), country="xl", doctype="uu")
    chapters = [a for a in anchors if a.kind == "chapter"]
    kept = [a for a in chapters if a.number == number]
    assert len(kept) == 1, [a.number for a in chapters]
    return kept[0].char_offset


def test_restated_chapter_keeps_the_one_in_sequence() -> None:
    """An omnibus restates whole statutes, so a repeated BAB is real text rather than
    a contents twin, and keep-last drops the chapter that belongs in the run."""
    text = _babs("IV", "V", "VI", "VII", "V")
    # The survivor sits between IV and VI, not at the trailing restatement.
    assert _kept_chapter(text, "V") < text.index("BAB VI")


def test_adjacent_duplicate_still_keeps_the_last() -> None:
    """A marker and its own heading line land as consecutive siblings, which is the
    contents shape keep-last already reads correctly."""
    text = _babs("I", "II", "II", "IV")
    # Without the adjacency guard the first II scores better and wrongly survives.
    assert _kept_chapter(text, "II") > text.index("KETENTUAN 2")


def test_doubling_back_run_falls_back_to_keep_last() -> None:
    """A run that reverses tells us nothing about which occurrence is the body one,
    so the sequence rule must stand aside rather than guess."""
    text = _babs("I", "II", "III", "VII", "II", "V")
    # The first II reads as in-sequence, but the run doubles back at the second.
    assert _kept_chapter(text, "II") > text.index("BAB VII")


def test_equally_well_placed_duplicates_keep_the_last() -> None:
    """A restated run fits its neighbours exactly as well as the original, so a tie
    is not evidence and the incumbent rule holds."""
    text = _babs("I", "II", "IV", "I", "II", "IV")
    assert _kept_chapter(text, "II") > text.index("BAB IV")


def test_sequence_selected_drop_is_not_reported_as_a_contents_twin() -> None:
    """`corpus_scan` aggregates the span reason, so a restatement counted as a
    contents twin would be evidence for the premise this pass disproves."""
    scan = scan_anchors_with_ambiguity(
        _babs("IV", "V", "VI", "VII", "V"), _xl_regex(), country="xl", doctype="uu"
    )
    spans = [s for s in scan.ambiguity if s.emitted_by == "drop_toc_duplicates"]
    assert [s.detail["reads_as"] for s in spans] == ["sequence_superseded"], [
        s.detail for s in spans
    ]


def _articles_for(numbers: list[int]) -> list[StructuralAnchor]:
    return [
        StructuralAnchor(
            kind="article",
            keyword="Pasal",
            number=str(n),
            char_offset=i * 100,
            line=i,
            matched_text=f"Pasal {n}",
        )
        for i, n in enumerate(numbers)
    ]


def test_restated_article_reads_by_its_neighbours_wherever_it_sits() -> None:
    """An article's number runs the body and the elucidation restates it, but with a
    readable subtree the neighbours place the repeat in both."""
    anchors = _articles_for([4, 5, 6, 7, 5])
    siblings = list(range(5))
    positions = [1, 4]  # the two Pasal 5
    assert _sequence_survivor(anchors, siblings, positions, False) == 1
    # Inside a Penjelasan it needs the subtree to read; here there is none to damage.
    assert _sequence_survivor(anchors, siblings, positions, True) == 1


def test_penjelasan_boundary_holds_through_the_scan() -> None:
    """The body/elucidation split has to survive the real pass order: the schedule
    anchor is recovered and then re-dropped when it merely captions a table."""
    restated = [4, 5, 6, 7, 5]
    body = "".join(f"Pasal {n}\nIsi pasal.\n\n" for n in restated)
    # A Penjelasan restarts its own numbering, which is what keeps the caption.
    text = (
        body + "PENJELASAN\n\nATAS\n\n" + "".join(f"Pasal {n}\nCukup jelas.\n\n" for n in restated)
    )
    split = text.index("PENJELASAN")
    anchors = scan_anchors(text, _xl_regex(), country="xl", doctype="uu")
    assert any(a.kind == "schedule" for a in anchors), [a.kind for a in anchors]
    fives = [a.char_offset for a in anchors if a.kind == "article" and a.number == "5"]
    assert len(fives) == 2, [(a.kind, a.number) for a in anchors]
    in_body, in_schedule = sorted(fives)
    # Body: the occurrence sitting between 4 and 6 wins.
    assert in_body < text.index("Pasal 6")
    # Elucidation: with a readable subtree the run places it there too, so the
    # survivor sits between its own 4 and 6 rather than trailing Pasal 7.
    assert in_schedule < text.index("Pasal 7", split)


def test_specimen_legislation_in_a_drafting_manual_is_not_the_document_s_own() -> None:
    """A drafting-manual statute illustrates its rules by displaying specimen
    instruments; their markers are shown, not enacted."""
    text = (
        "Pasal 1\nKetentuan umum.\n\n"
        "2. Judul memuat keterangan mengenai jenis dan nomor.\n"
        "Contoh 1:\nUNDANG-UNDANG REPUBLIK LANGKASUKA\nNOMOR 5 TAHUN 2014\n"
        "Pasal 3\nDihapus.\n"
        "3. Penomoran ditulis dengan angka Arab.\n\n"
        "Pasal 2\nKetentuan penutup.\n"
    )
    anchors = scan_anchors(text, _xl_regex(), country="xl", doctype="uu")
    arts = [a for a in anchors if a.kind == "article"]
    # The specimen's Pasal 3 is marked quoted, so it takes no eId of its own.
    assert [a.number for a in arts if not a.quoted_amendment] == ["1", "2"]
    assert [a.number for a in arts if a.quoted_amendment] == ["3"]


def test_a_worked_example_is_not_a_specimen() -> None:
    """The same lead-in introduces arithmetic examples in an elucidation, which
    display no legislation; suppressing under those silences the host."""
    text = (
        "Pasal 1\nKetentuan umum.\n\n"
        "Contoh 1:\nDalam hal Pengusaha Kena Pajak A melakukan penyerahan\n"
        "Barang Kena Pajak pada tanggal 1 Juli.\n"
        "Pasal 2\nKetentuan penutup.\n"
    )
    anchors = scan_anchors(text, _xl_regex(), country="xl", doctype="uu")
    arts = [a for a in anchors if a.kind == "article"]
    assert [a.number for a in arts if not a.quoted_amendment] == ["1", "2"]


def test_glyph_damaged_numbers_compare_as_their_digits() -> None:
    """A scan reads zero as O and one as l or I; the sequence has to see through
    that, while a clean Roman numeral still parses as Roman."""
    assert _roman_or_digit("2O2") == 202
    assert _roman_or_digit("2OO") == 200
    assert _roman_or_digit("10I") == 101
    assert _roman_or_digit("II") == 2
    assert _roman_or_digit("XL") == 40


def test_a_damaged_subtree_blocks_selection_inside_an_elucidation() -> None:
    """Choosing between restated articles reparents everything beneath them, so
    an unreadable child marker means keep-last stands."""
    clean = _articles_for([4, 5, 6, 7, 5])
    siblings, positions = list(range(5)), [1, 4]
    assert _sequence_survivor(clean, siblings, positions, True) == 1

    damaged = list(clean)
    damaged.insert(
        2,
        StructuralAnchor(
            kind="paragraph",
            keyword="Ayat",
            number="(21",
            char_offset=150,
            line=2,
            matched_text="Ayat (21",
        ),
    )
    assert _sequence_survivor(damaged, [0, 1, 3, 4, 5], [1, 5], True) is None


class TestAmendingActSummary:
    """A Bahasa amending act declares `two_pasal_roman`: its own articles
    are Roman, and the Arabic ones between them replace another law's text. The
    scanner has always marked them; the summary used to count them anyway, and
    the validator then read the AKN as 3 articles short of a document that was
    never supposed to exist."""

    TEXT = (
        "UNDANG-UNDANG REPUBLIK LANGKASUKA\n"
        "NOMOR 32 TAHUN 2024\n"
        "TENTANG PERUBAHAN ATAS UNDANG-UNDANG NOMOR 8 TAHUN 1991\n\n"
        "Pasal I\n\n"
        "Beberapa ketentuan dalam Undang-Undang Nomor 8 Tahun 1991 diubah "
        "sebagai berikut:\n\n"
        "Pasal 1\n"
        "Dalam Undang-Undang ini yang dimaksud dengan konservasi adalah pengelolaan.\n\n"
        "Pasal 2\n"
        "Ketentuan mengenai kawasan suaka alam diatur dengan Peraturan Pemerintah.\n\n"
        "Pasal 3\n"
        "Setiap orang dilarang melakukan kegiatan yang bertentangan dengan ketentuan ini.\n\n"
        "Pasal II\n\n"
        "Undang-Undang ini mulai berlaku pada tanggal diundangkan.\n"
    )

    def _scan(self) -> list:
        from codify.jurisdictions import load_config
        from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

        return scan_anchors(
            self.TEXT, build_anchor_regex(load_config("xl"), "act"), country="xl", doctype="act"
        )

    def test_the_quoted_articles_are_marked(self) -> None:
        anchors = self._scan()
        assert [a.number for a in anchors if a.quoted_amendment] == ["1", "2", "3"]
        assert [a.number for a in anchors if not a.quoted_amendment] == ["I", "II"]

    def test_the_summary_reports_the_act_s_own_two_articles(self) -> None:
        from codify.pipeline.enrich.anchors import anchor_summary

        assert anchor_summary(self._scan()) == {"article": 2}


def test_uk_amendment_block_does_not_borrow_the_next_block_s_closing_quote() -> None:
    """An unterminated block must be skipped, not run on to the next one.

    OCR drops a closing quote often enough to plan for. Searching on without a
    bound finds the *following* block's quote and masks every genuine provision
    in between, which is silent: the section simply is not there. Under-masking
    surfaces as a duplicate_number instead, so skipping is the safer failure.
    """
    from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity

    text = (
        "3 First amendment\n"
        "(1) After Article 4 insert—\n"
        "\n"
        '"5 Quoted article\n'
        "(1) A quoted subsection.\n"  # closing quote dropped by OCR
        "\n"
        "4 A GENUINE SECTION\n"
        "(1) A genuine subsection of the host Act.\n"
        "\n"
        "5 Second amendment\n"
        "(1) After Article 9 insert—\n"
        "\n"
        '"10 Another quoted article\n'
        '(1) Another quoted subsection."\n'
        "\n"
        "6 Final section\n"
        "(1) A final subsection.\n"
    )
    regex = cached_regex("gb", "act")
    scan = scan_anchors_with_ambiguity(text, regex, country="gb", doctype="act")

    sections = {a.number for a in scan.anchors if a.kind == "section"}
    # 4 and 6 are the host Act's own; losing them is the failure this guards.
    assert {"4", "6"} <= sections, f"genuine sections swallowed: {sections}"
    # The well-formed second block is still masked.
    assert "10" not in sections, f"phantom section leaked: {sections}"


def test_quoted_amendment_articles_leave_the_coverage_ratio_alone() -> None:
    """A quoted amendment must not move coverage in either direction.

    Before this, `anchor_coverage` counted quoted anchors toward `captured`
    while `anchor_summary` excluded them, so an amending act read as better
    covered than it was. Excluding them from the numerator alone would have
    been the same error mirrored: their markers are in the source, so the
    denominator sees them too.
    """
    from dataclasses import replace

    from codify.pipeline.enrich.anchors import anchor_coverage, cached_regex, scan_anchors

    config = _QANUN_CONFIG
    assert config is not None
    text = "مادة ١\nنص.\n\nمادة ٢\nنص.\n\nمادة ٣\nنص.\n"
    anchors = scan_anchors(text, cached_regex("xx", "qanun"), country="xx", doctype="qanun")
    assert len(anchors) == 3

    # Article 3 stands in for one the scanner marked as borrowed structure,
    # whether from a quote span or a replacement lead-in.
    quoted = [replace(a, quoted_amendment=a.number == "٣") for a in anchors]
    assert anchor_coverage(text, quoted, config, "qanun", "article").ratio == 1.0
    assert anchor_coverage(text, anchors, config, "qanun", "article").ratio == 1.0


def test_a_host_number_the_quote_repeats_is_still_expected() -> None:
    """Dropping by number would hide the omission this gate exists to catch.

    A host article and a quoted one can carry the same number. If the scan
    captures only the quoted occurrence and the denominator loses the number,
    coverage reads 1.0 over a host provision that was never anchored. Excluding
    by position keeps the host's own occurrence in ``expected``, so the miss
    still shows.
    """
    from dataclasses import replace

    from codify.pipeline.enrich.anchors import anchor_coverage, cached_regex, scan_anchors

    config = _QANUN_CONFIG
    assert config is not None
    text = "مادة ١\nنص.\n\nمادة ٢\nنص.\n\nمادة ١\nنص مقتبس.\n"
    anchors = scan_anchors(text, cached_regex("xx", "qanun"), country="xx", doctype="qanun")
    assert len(anchors) == 3

    # The scan reached only the quoted occurrence of ١; the host one was missed.
    quoted_only = [replace(a, quoted_amendment=True) for a in anchors[2:]]
    cov = anchor_coverage(text, [anchors[1], *quoted_only], config, "qanun", "article")
    assert "1" in cov.expected, "the host's own occurrence must survive"
    assert cov.ratio is not None and cov.ratio < 1.0


class TestConfigLookupSeams:
    """Two mechanisms let a test of the config guard pass without exercising it."""

    def test_the_guard_is_reachable_by_a_module_level_patch(self) -> None:
        """Three helpers imported `load_config` inside the call, so patching the
        module name never reached them and a test asserting the raise passed
        vacuously."""
        from unittest.mock import patch

        import codify.pipeline.enrich.anchors as anchors

        for helper, args in (
            (anchors._aliases_for, ("gb", "act")),
            (anchors._prose_filter_for, ("gb",)),
            (anchors._sameline_filter_for, ("gb",)),
            (anchors.cached_regex, ("gb", "act")),
        ):
            helper.cache_clear()
            with (
                patch.object(anchors, "load_config", side_effect=RuntimeError("patched")),
                pytest.raises(RuntimeError, match="patched"),
            ):
                helper(*args)
            helper.cache_clear()

    def test_a_warm_cache_answers_after_the_config_would_refuse(self) -> None:
        """Pinned because it is surprising, not because it is wrong: these helpers
        are `lru_cache`d by jurisdiction, so a country scanned before its config
        went missing keeps its entry. A test that warms the cache and then expects
        the raise is testing nothing."""
        from unittest.mock import patch

        import codify.pipeline.enrich.anchors as anchors
        from codify.jurisdictions import JurisdictionConfigError

        anchors._aliases_for.cache_clear()
        warm = anchors._aliases_for("gb", "act")
        with patch.object(anchors, "load_config", side_effect=JurisdictionConfigError("gone")):
            assert anchors._aliases_for("gb", "act") == warm
        anchors._aliases_for.cache_clear()


class TestAmendmentLeadIn:
    """What decides whether a quoted amendment leaves the coverage sets."""

    def test_a_lead_in_may_end_on_its_pointer_instead_of_a_colon(self) -> None:
        # An insertion clause names the new article and stops: no colon follows.
        import re

        from codify.pipeline.enrich.anchors import _leadin_before

        trigger = re.compile("الآتي")
        text = "مادة (2)\nتضاف مادة جديدة برقم (47 مكرر) نصها كالآتي\nالمادة (47) مكرر\n"
        cand = text.index("المادة (47)")
        assert _leadin_before(text, 0, cand, trigger, "47 مكرر") == "singular"

    def test_prose_ending_on_a_pointer_does_not_swallow_the_next_heading(self) -> None:
        """Ordinary prose ends on "as follows" constantly. Without the colon,
        only a clause naming the article it introduces reads as a lead-in."""
        import re

        from codify.pipeline.enrich.anchors import _leadin_before

        trigger = re.compile("الآتي")
        text = "المادة ١٢\nيلتزم الموظف بتنفيذ واجباته على النحو الآتي\nالمادة ١\n"
        cand = text.rindex("المادة ١")
        assert _leadin_before(text, 0, cand, trigger, "١") is None

    def test_a_pointer_far_from_the_marker_is_not_a_lead_in(self) -> None:
        # Distance is the whole of the precision once the colon is gone.
        import re

        from codify.pipeline.enrich.anchors import _leadin_before

        trigger = re.compile("الآتي")
        text = (
            "على النحو الآتي\n"
            + "نص عادي يملأ السطر ويبعد المؤشر عن العنوان. " * 3
            + "\nالمادة (5)\n"
        )
        cand = text.index("المادة (5)")
        assert _leadin_before(text, 0, cand, trigger, "5") is None

    def test_an_offset_in_trailing_space_keys_its_own_marker_line(self) -> None:
        """The anchor offset can sit before the newline when a line ends on a
        space, where a plain line-start keys the line above and every
        position-keyed exclusion misses."""
        from codify.pipeline.enrich.anchors import _line_start, _marker_line_start

        text = "لتصبح على النحو الآتي: \nالمادة (36)\n"
        offset = text.index(" \n")
        assert _line_start(text, offset) == 0
        assert _marker_line_start(text, offset) == text.index("المادة (36)")


class TestBisArticlesStayHostStructure:
    """A bis article is permanent structure, not a quoted amendment."""

    def test_a_host_bis_article_after_its_base_is_not_a_quoted_amendment(self) -> None:
        from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

        config = _QANUN_CONFIG
        text = (
            "المادة ٥\nتحدد الشروط كما يلي\n"
            "المادة ٥ مكرر\nنص المادة الخامسة مكرر، وهي مادة أصيلة.\n"
            "المادة ٦\nنص سادس.\n"
        )
        anchors = scan_anchors(
            text, build_anchor_regex(config, "qanun"), country="xx", doctype="qanun"
        )
        arts = [a for a in anchors if a.kind == "article"]
        assert [a.akn_eid for a in arts] == ["art_5", "art_5bis", "art_6"]
        assert not any(a.quoted_amendment for a in arts)


class TestQuotedAmendmentLeavesBothCoverageSets:
    """Pins the two mechanisms through `anchor_coverage` itself.

    Each is unit-tested above, but a unit test on either passes while the
    coverage sets stay wrong, which is how the original defect survived. This
    asserts the outcome the gate reads.
    """

    SOURCE = (
        "أصدرنا القانون الآتي:\n"
        "مادة (1)\n"
        "تعدل المواد: 36، 48 من القانون الأساسي، لتصبح على النحو الآتي: \n"
        " المادة (36)\nنص المادة السادسة والثلاثين.\n"
        " المادة (48)\nنص المادة الثامنة والأربعين.\n"
        "مادة (2)\n"
        "تضاف مادة جديدة برقم (47 مكرر) نصها كالآتي\n"
        " المادة (47) مكرر\nنص المادة الجديدة.\n"
        "مادة (3)\nيعمل بهذا القانون من تاريخ نشره.\n"
    )

    def _coverage(self):
        from codify.pipeline.enrich.anchors import (
            anchor_coverage,
            build_anchor_regex,
            scan_anchors,
        )

        config = _QANUN_CONFIG
        anchors = scan_anchors(
            self.SOURCE,
            build_anchor_regex(config, "qanun"),
            country="xx",
            doctype="qanun",
        )
        return anchor_coverage(self.SOURCE, anchors, config, "qanun", "article")

    def test_the_amending_act_is_scored_on_its_own_three_articles(self) -> None:
        coverage = self._coverage()
        # The other law's 36, 48 and 47bis are quoted, so they belong to
        # neither set: counted in one alone, the act reads as half-structured.
        assert sorted(coverage.expected) == ["1", "2", "3"]
        assert sorted(coverage.captured) == ["1", "2", "3"]
        assert coverage.ratio == 1.0


def test_two_grouping_terms_at_one_depth_keep_both_levels() -> None:
    """A second grouping term must be distinguished from the first, not deleted.

    Mapping it onto a child element drops every one of its anchors when no
    parent of that element is open, so the level disappears and the collision
    it was declared to resolve goes with it.
    """
    from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

    text = (
        "الباب الأول\nأحكام عامة\n\n"
        "قسم ١\nنص القسم الأول.\n\n"
        "قسم ٢\nنص القسم الثاني.\n\n"
        "مادة ١\nنص المادة.\n"
    )
    anchors = scan_anchors(
        text, build_anchor_regex(_QANUN_CONFIG, "qanun"), country="xx", doctype="qanun"
    )
    kinds = [a.kind for a in anchors]
    assert kinds.count("subdivision") == 2, kinds
    eids = [a.akn_eid for a in anchors if a.kind == "subdivision"]
    assert len(set(eids)) == 2, eids
