"""Jurisdiction configuration, machine-readable profiles for the pipeline."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import unicodedata
from datetime import date
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal, NamedTuple, cast

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from codify.data_paths import data_dir

# Atlas models live in a sibling module. The three container models are used by
# JurisdictionConfig's annotations; the rest are re-exports so
# `from codify.jurisdictions import <name>` keeps resolving.
from .jurisdictions_atlas import (
    BindingRegime,
    BindingRegimeKind,  # noqa: F401  (re-export)
    DirectEffectLevel,  # noqa: F401  (re-export)
    MembershipStatus,  # noqa: F401  (re-export)
    ReportingCadence,  # noqa: F401  (re-export)
    ResidualLaw,
    SupranationalMembership,
    SuspensionPeriod,  # noqa: F401  (re-export)
    canonicalise_body,  # noqa: F401  (re-export)
)

logger = structlog.get_logger()

# _STRICT for models the pipeline reads at inference time; _LOOSE for
# documentation-carrier models that aggregate jurisdiction metadata.
_STRICT = ConfigDict(extra="forbid", populate_by_name=True)
_LOOSE = ConfigDict(extra="allow", populate_by_name=True)


JURISDICTIONS_DIR = data_dir(__file__, "jurisdictions")


Tradition = Literal[
    "civil_law",
    "common_law",
    "islamic",
    "customary",
    "hindu",
    "confucian",
    "jewish",
    "socialist",
    "sui_generis",
]
Calendar = Literal[
    "gregorian",
    "lunar_hijri",
    "solar_hijri",
    "ethiopian",
    # Not one "imperial era": these convert differently. Minguo is year + 1911;
    # a Japanese era year needs the era name, since Showa 5 is not Reiwa 5.
    "japanese_era",
    "minguo",
    "buddhist_era",
    "dual",
]
HierarchyLevel = Literal["higher", "basic", "subdivision", "grouping", "presentational"]
# How a level is recognised in the text when it carries no hierarchy keyword.
# `caption`: a closed vocabulary of heading lines. `bracketed_decimal`: "[1.1]".
MarkerForm = Literal[
    "caption",
    "bracketed_decimal",
    # Outline markers, for annex content that numbers itself without a keyword:
    # "A. Latar Belakang", "1. RKUPH", "2) Penilaian".
    "upper_letter_period",
    "upper_roman_period",
    "lower_letter_period",
    "arabic_period",
    "arabic_closing_paren",
    "lower_letter_closing_paren",
    # The operative subdivision of an article, printed as "(1) " with no
    # keyword of its own.
    "parenthesized_arabic",
]
ValidationStatus = Literal["production", "draft", "stub"]
JurisdictionType = Literal["national", "supranational", "subnational", "crown_dependency"]

# AKN 3.0 hierarchical element names, restricted to those actually used
# across the corpus. A few less-common but valid ones (preamble, recital)
# are structural wrappers, not hierarchy members, so they're not here.
AknElement = Literal[
    "book",
    "part",
    "title",
    "subtitle",
    "tome",
    "chapter",
    "subchapter",
    "section",
    "subsection",
    "division",
    "subdivision",
    "subpart",
    "article",
    "paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "point",
    "indent",
    "alinea",
    "rule",
    "subrule",
    "item",
    "attachment",
    "hcontainer",
    "proviso",
]

# Bluebell keyword set sourced from the akn-ontology skill + corpus usage.
BluebellKeyword = Literal[
    "BOOK",
    "PART",
    "TITLE",
    "SUBTITLE",
    "TOME",
    "CHAPTER",
    "SUBCHAPTER",
    "SECTION",
    "SUBSECTION",
    "DIVISION",
    "SUBDIVISION",
    "SUBPART",
    "ARTICLE",
    "PARAGRAPH",
    "SUBPARAGRAPH",
    "CLAUSE",
    "SUBCLAUSE",
    "POINT",
    "INDENT",
    "ALINEA",
    "RULE",
    "SUBRULE",
    "PROVISO",
    "TRANSITIONAL",
    "CROSSHEADING",
    "PART_HEADING",
]

# AKN 3.0 TLC classes (all 10, per the ontology).
TlcClass = Literal[
    "TLCPerson",
    "TLCOrganization",
    "TLCRole",
    "TLCObject",
    "TLCConcept",
    "TLCEvent",
    "TLCLocation",
    "TLCProcess",
    "TLCTerm",
    "TLCReference",
]


class HierarchyEntry(BaseModel):
    model_config = _LOOSE

    local_term: str
    akn_element: AknElement
    level: HierarchyLevel
    bluebell_keyword: BluebellKeyword | None = None
    numbering: str | None = None
    # Overrides the element's canonical eId prefix. Set it only where the element
    # cannot imply one (hcontainer: sch, xhd; two local terms on one element) or
    # the publisher's own AKN differs (UK SIs number sections reg_N).
    eid_abbrev: str = ""
    note: str = ""
    # Unified multilingual store for local_term_ara / local_script_rus / etc.
    # Keys are ISO 639-3 language codes or "script:<iso-15924>".
    local_terms: dict[str, str] | None = None
    akn_name: str | None = None
    # Genres whose levels carry no keyword: a judgment writes "DUDUK PERKARA"
    # and "[1.1]", never its hierarchy term. Unset keeps keyword matching.
    marker_form: MarkerForm | None = None
    # The closed caption vocabulary a `caption` level answers to.
    captions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _captions_match_marker_form(self) -> "HierarchyEntry":
        # A caption level with no vocabulary matches nothing, and the keyword
        # path is already off, so the level would vanish with no error.
        if self.marker_form == "caption" and not [c for c in self.captions if c.strip()]:
            raise ValueError(
                f"hierarchy level {self.local_term!r} is caption-marked but declares no captions"
            )
        if self.marker_form != "caption" and self.captions:
            raise ValueError(
                f"hierarchy level {self.local_term!r} declares captions "
                "without marker_form='caption'"
            )
        return self


def _check_hierarchy_levels(where: str, hierarchy: list[HierarchyEntry]) -> None:
    """One basic level, and no higher level after a basic or subdivision one.

    Only `higher`, `basic` and `subdivision` order the sequence; `grouping` and
    `presentational` are orthogonal and may appear anywhere."""
    basic = [h for h in hierarchy if h.level == "basic"]
    if len(basic) > 1:
        raise ValueError(
            f"{where}: {len(basic)} hierarchy entries with "
            f"level=basic (expected at most 1): {[h.local_term for h in basic]}"
        )
    seen_basic = seen_subdivision = False
    for h in hierarchy:
        if h.level == "higher" and (seen_basic or seen_subdivision):
            raise ValueError(
                f"{where}: higher-level entry '{h.local_term}' "
                f"appears after a basic/subdivision entry"
            )
        seen_basic = seen_basic or h.level == "basic"
        seen_subdivision = seen_subdivision or h.level == "subdivision"


class HContainer(BaseModel):
    model_config = _LOOSE

    local_term: str
    name: str
    description: str = ""
    bluebell_proxy: BluebellKeyword = "SUBSECTION"
    requires_postprocessing: bool = True
    note: str = ""
    numbering: str | None = None
    label: str | None = None
    akn_element: AknElement | None = None
    level: str | None = None
    position: str | None = None
    bluebell_keyword: BluebellKeyword | None = None


class DocumentClass(BaseModel):
    model_config = _LOOSE

    label: str
    # Compact display form (e.g. "VKM" for al/vendim). Falls back to the
    # title-cased doctype when unset. Single source of truth for the UI + PDF.
    short_label: str | None = None
    akn_element: str = "act"
    bluebell_compatible: bool = True
    basic_unit: str = "section"
    # A matching year prefix in the printed serial is separate from the URI number.
    number_has_year_prefix: bool = False
    # "title_identity": the URI number is always the title digest, never a stated
    # serial, for classes whose serials repeat across issuers.
    number_source: Literal["stated", "title_identity"] = "stated"
    hierarchy: list[HierarchyEntry] = Field(default_factory=list)
    hcontainers: list[HContainer] = Field(default_factory=list, alias="hcontainers")
    direct_effect: bool | None = None
    note: str = ""
    # Another class of this jurisdiction whose fields this one lays over. One level.
    extends: str | None = None
    # Where this class sits in the legal hierarchy, for ranking a name search:
    # higher outranks lower on a close title match. Unset means no weighting, so
    # a jurisdiction whose hierarchy nobody has transcribed keeps pure relevance
    # order.
    #   4 constitutional  3 primary legislation  2 executive decree
    #   1 subordinate regulation or decision     0 implementing instructions
    search_rank: int | None = None
    akn_subtype: str | None = None
    local_label: str | None = None
    prompt_variant: str | None = None
    frbr_subtype: str | None = None
    # Per-class authoritative language, falling back to the jurisdiction's when
    # None. Vatican needs it: Latin for canon law, Italian for the City State.
    authoritative_language: str | None = None


#: The Gregorian month each kind's new year falls in when the config names none.
_DEFAULT_NEW_YEAR_MONTH = {"bikram_samvat": 4, "ethiopian": 9}


class CalendarConversion(BaseModel):
    """Deterministic conversion rule for non-Gregorian local calendars.

    kind selects which optional fields apply:

    - epoch_offset  single-epoch calendars. Gregorian = local_year + epoch_year,
                    where epoch_year is an integer offset (ROC 1911, Juche 1911,
                    Buddhist -543).
    - era_table     named era segments with boundaries.
    - hijri_lunar   Islamic Hijri, ~354 days. Needs codify.calendar, not an offset.
    - hijri_solar   Iranian Hijri, Gregorian year length.
    - ethiopian     ~7-8 year offset, 13-month year.
    - buddhist      Thai/Lao, epoch_year -543.
    - bikram_samvat Nepali, 57-year offset, month-sensitive (Chaitra straddles the
                    mid-April new year).
    """

    model_config = _STRICT

    kind: Literal[
        "epoch_offset",
        "era_table",
        "hijri_lunar",
        "hijri_solar",
        "ethiopian",
        "buddhist",
        "bikram_samvat",
    ]
    # Integer added to the local year to give the Gregorian year (Buddhist -543,
    # ROC and Juche 1911, since their year 1 is 1912 CE). None for calendars
    # without a simple offset: hijri, ethiopian, era_table, bikram_samvat.
    epoch_year: int | None = None
    # Era table: [{"name": "Reiwa", "start": "2019-05-01"}] for Japanese-style.
    eras: list[dict[str, Any]] | None = None
    # Month-sensitive conversions (Bikram Samvat New Year mid-April).
    new_year_month: int | None = None
    new_year_day: int | None = None
    # Offset variant for month-sensitive: subtract `offset` from local year
    # before the new-year day, otherwise `offset + 1`.
    offset: int | None = None
    # Month names in the local calendar's own order, index 0 = month 1. Present
    # only where a dated line is read off the source text.
    month_names: list[str] = Field(default_factory=list)
    # Whether the months and days coincide with the Gregorian ones, so only the
    # year converts. False for any calendar with a grid of its own.
    month_day_is_gregorian: bool = False
    # Phrases that introduce the date a document was made. A dated line elsewhere
    # in the text is some other instrument's.
    date_cues: list[str] = Field(default_factory=list)
    # Words marking a year as belonging to this calendar ("B.E.", "A.H.").
    year_particles: list[str] = Field(default_factory=list)
    # The local year began in this month before `new_year_reform_year`, so an
    # earlier date carries the previous year's number and the offset is short.
    new_year_reform_year: int | None = None
    note: str = ""

    @model_validator(mode="after")
    def _date_grammar_is_usable(self) -> "CalendarConversion":
        """A blank month renumbers the calendar and a blank cue compiles to a
        pattern matching the start of every document."""
        # On the folded form, since the lookup folds its key.
        names = [n.strip().casefold() for n in self.month_names]
        if self.month_names and (not all(names) or len(set(names)) != len(names)):
            raise ValueError("month_names must be non-blank and distinct")
        if self.date_cues and not any(c.strip() for c in self.date_cues):
            raise ValueError("date_cues must carry at least one non-blank cue")
        if self.month_day_is_gregorian and self.month_names and len(self.month_names) != 12:
            # Named at all, the months of a Gregorian grid are twelve; any other
            # count renumbers them. Unnamed, the grid serves the metadata path.
            raise ValueError("month_names on a Gregorian grid must be exactly 12")
        if self.new_year_month is not None and not 1 <= self.new_year_month <= 12:
            raise ValueError("new_year_month must be a month, 1 to 12")
        if self.new_year_day is not None:
            # Against the month it will be read in: a day that month never has
            # would file every date of the month on one side.
            month = self.new_year_month or _DEFAULT_NEW_YEAR_MONTH.get(self.kind, 1)
            longest = max(calendar.monthrange(year, month)[1] for year in (2023, 2024))
            if not 1 <= self.new_year_day <= longest:
                raise ValueError(f"new_year_day must be a day of month {month}, 1 to {longest}")
        if self.new_year_reform_year is not None and self.new_year_reform_year < 1:
            # Every calendar here counts from one.
            raise ValueError("new_year_reform_year must be a year")
        if self.new_year_reform_year is not None and not self.new_year_month:
            # Without one the shift can never fire, so the field reads as set
            # and does nothing.
            raise ValueError("new_year_reform_year needs new_year_month")
        if self.kind == "era_table" and (self.month_names or self.date_cues):
            # The grammar captures a bare number, which an era table reads as a
            # year of its latest era.
            raise ValueError(
                "an era_table conversion cannot carry a date grammar (month_names, date_cues)"
            )
        return self


#: Hex characters of the digest a title-derived segment carries; six (2^24)
#: collided on ordinary titles.
SLUG_DIGEST_CHARS = 12


class TitleIdentity(BaseModel):
    """How a jurisdiction that numbers nothing derives an identity from a title.
    Every field is a literal `codify.frbr.identity_from_title` reads."""

    model_config = _STRICT

    # Document-kind words opening a short title, longest first.
    strip_prefixes: list[str] = Field(default_factory=list)
    # Words marking the year that follows as the instrument's own.
    year_particles: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _literals_are_not_blank(self) -> "TitleIdentity":
        """A blank particle matches everywhere, so any run of digits would read
        as the year and any parenthetical as an edition."""
        for name in (
            "strip_prefixes",
            "year_particles",
            "edition_markers",
            "consolidation_markers",
        ):
            if any(not v.strip() for v in getattr(self, name)):
                raise ValueError(f"{name} must not carry a blank entry")
        return self

    # Words inside a parenthetical naming this instrument's edition, spelling
    # variants included; the first is canonical and is what a slug carries.
    edition_markers: list[str] = Field(default_factory=list)
    # Parenthetical words marking a re-publication; an edition number after one
    # is the edition folded in, not this document's.
    consolidation_markers: list[str] = Field(default_factory=list)


class FrbrConfig(BaseModel):
    model_config = _LOOSE

    country_code: str
    # Declared where instruments carry no number and the title is the identity.
    title_identity: TitleIdentity | None = None
    uri_patterns: dict[str, str] = Field(default_factory=dict)
    date_calendar: Calendar = "gregorian"
    date_calendar_note: str | None = None
    # Structured conversion rule. Populated for non-Gregorian
    # jurisdictions (NP, TW, KP, JP, IR, ET, TH). FRBR URIs always use
    # Gregorian (ADR 001), this drives the conversion utility.
    calendar_conversion: CalendarConversion | None = None
    note: str | None = None
    notes: list[str] | str | None = None
    number_format: str | dict[str, str] | None = None
    number_note: str | None = None
    number_extraction: str | dict[str, Any] | None = None
    sample_uris: list[Any] | dict[str, str] | None = None

    @model_validator(mode="after")
    def _grammars_suit_the_conversion(self) -> "FrbrConfig":
        """A title grammar captures a bare number; an era table reads one as a
        year of its latest era, so the two cannot be declared together."""
        rule = self.calendar_conversion
        if self.title_identity is not None and rule is not None and rule.kind == "era_table":
            raise ValueError("an era_table conversion cannot carry a title_identity grammar")
        return self


class NumberingConfig(BaseModel):
    model_config = _LOOSE

    act_citation: str | None = None
    act_citation_alt: str | None = None
    insertion_style: str | None = None
    note: str | None = None
    insertion_example: str | None = None
    insertion_note: str | None = None
    article_numbering: str | None = None
    # Per-instrument citation map, absorbs si_citation / decree_citation /
    # loi_citation / law_citation / ordonnance_citation etc. without a
    # separate field per jurisdiction.
    citations: dict[str, str] = Field(default_factory=dict)
    # Whether an instrument number identifies a work on its own. False almost
    # everywhere: most jurisdictions restart numbering each year, so a citation
    # that omits its date names one of many. Declaring it true lets a year-less
    # citation resolve on the number, and is a claim about the whole series
    # rather than about the corpus held, which may hold only one of the
    # collisions and so cannot disprove it.
    numbers_unique_across_years: bool = False


class EnactingFormula(BaseModel):
    model_config = _LOOSE

    text: str | None = None
    position: str = "preamble"
    era: str = ""
    from_date: date | None = Field(default=None, alias="from")
    to_date: date | None = Field(default=None, alias="to")
    note: str = ""
    document_class: str | list[str] | None = None
    language: str | None = None
    transliteration: str | None = None
    instrument_type: str | None = None
    applies_to: list[Any] | str | None = None
    verified: bool = False
    enacting_body_tlc: str | None = None

    @field_validator("from_date", "to_date", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        # Legacy configs store "" instead of null; coerce so date parsing
        # doesn't fail on empty strings.
        if v == "":
            return None
        return v


class AmendmentConfig(BaseModel):
    model_config = _LOOSE

    style: str | None = None
    phrasing: str | None = None
    cross_reference_format: str | None = None
    # Lead-in phrases marking an embedded amendment article: a number following
    # one of these plus a colon is quoted content, not a new top-level article.
    trigger_phrases: list[str] = Field(default_factory=list)
    # Body shape of an amending instrument. `two_pasal_roman`: the host's own
    # articles are Roman, so an Arabic-numbered one after them is quoted text.
    amending_body_structure: str | None = None
    # Hierarchy element that numbers one amendment made by an amending article
    # (Indonesia's `Angka`). What follows it belongs to the amended statute, so
    # it must not compete in the host's numbering. Empty leaves the pass inert.
    amendment_item_kind: str | None = None


class TLCEntry(BaseModel):
    model_config = _STRICT

    # NCName-compatible, permitting camelCase and hyphenated forms both seen in
    # the corpus. min_length=1 keeps the empty alias from raising an opaque
    # pattern-mismatch.
    eId: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9-]*$", min_length=1)
    tlc_class: TlcClass = Field(alias="class")
    href: str
    showAs: str
    note: str = ""


class ClassificationRule(BaseModel):
    """Deterministic rule for assigning a document class from the incoming header or
    metadata. Tried in descending priority; the first match wins. No match falls back to
    `default_document_class` and to whatever heuristics `StructuringConfig.prompt_additions`
    still carries.

    signal          what `pattern` is tested against
      date_range      nothing; the match is `date_from`/`date_until` alone, for laws
                      post-dating a regime change (PS `qarar_bi_qanun` from 2007-06-14)
      preamble_match  the preamble or enacting-formula text
      issuer_role     the enacting body's role ("Pope", "Council of Ministers", "OHR")
      gazette_series  the official gazette series the document was published in
      title_regex     the document title ("Basic Law:", "Motu Proprio")
    """

    model_config = _STRICT

    signal: Literal["date_range", "preamble_match", "issuer_role", "gazette_series", "title_regex"]
    pattern: str = ""
    pattern_flags: str = ""  # e.g. "i" for case-insensitive
    target_document_class: str
    priority: int = 0
    date_from: date | None = None
    date_until: date | None = None
    note: str = ""


class PlaceholderStatus(StrEnum):
    """The AKN `@status` a placeholder unit should carry; named so the generated
    contract keeps a stable alias rather than a numbered `Status`."""

    INCOMPLETE = "incomplete"
    REMOVED = "removed"
    EDITORIAL = "editorial"


class PlaceholderMarker(BaseModel):
    """A whole-string pattern a publisher writes where content is missing or
    struck, and the AKN status the unit should carry for it."""

    model_config = _STRICT

    pattern: str
    status: PlaceholderStatus = PlaceholderStatus.INCOMPLETE


class AttachmentCaption(BaseModel):
    """A caption line opening an attachment. AKN models a schedule and a
    mandatory companion alike, so legal force is declared per caption."""

    model_config = _LOOSE

    # Centred in the source, so the match tolerates a wide indent.
    caption: str
    # False for an explanatory memorandum: no norm, and no legal basis.
    normative: bool = True
    # Levels the attachment's own content uses: an annex can be a lettered
    # outline rather than the article hierarchy of the body it is attached to.
    # Empty keeps today's behaviour: the body's levels, or none.
    hierarchy: list[HierarchyEntry] = Field(default_factory=list)
    # True where the caption opens a longer title on the same line ("SCHEDULE of
    # fees…"), so the caption need only begin the line. The line is the heading.
    prefix: bool = False


class StructuringConfig(BaseModel):
    model_config = _LOOSE

    prompt_variant: str | None = None
    prompt_additions: str | None = None
    # Deterministic classification rules, evaluated by descending priority
    # before the AI prompt. prompt_additions keeps tacit style hints only.
    classification_rules: list[ClassificationRule] = Field(default_factory=list)
    # `line_anchored` where the drafting standard puts the marker on its own
    # line and the sources keep it, so a mid-sentence citation cannot anchor.
    marker_boundary: Literal["relaxed", "line_anchored"] = "relaxed"
    # Words ending the previous line when a citation wraps onto the next, which
    # a boundary rule cannot see. Empty falls back to the script pack's list.
    prose_precursors: list[str] = Field(default_factory=list)
    # Precursors that mark a citation only when adjacent on the same line.
    sameline_precursors: list[str] = Field(default_factory=list)
    # Ordinal words used as container numbers ("Bagian Kesatu" is part 1).
    # Declared in the source's own casing; matching is case-sensitive.
    ordinal_words: dict[str, int] = Field(default_factory=dict)
    # Words after a marker's number that mark an inserted unit ("5 bis"), mapped to the eId form.
    # One shared table, like `ordinal_words`: never blank, a number, or a bare Latin letter.
    insertion_suffixes: dict[str, str] = Field(default_factory=dict)
    # Words that, following a marker's number on its own line, make the line a
    # citation list rather than a provision ("Article 5 to Article 9 apply").
    citation_successors: list[str] = Field(default_factory=list)

    @field_validator("insertion_suffixes")
    @classmethod
    def _suffixes_are_distinctive_words(cls, value: dict[str, str]) -> dict[str, str]:
        for word, form in value.items():
            bare_latin = len(word) == 1 and word.isascii()
            # Letters and marks only: a decoration such as `**` or `-` is never a word.
            lettered = all(unicodedata.category(ch)[0] in "LM" for ch in word)
            if not word or not lettered or bare_latin:
                raise ValueError(f"insertion suffix {word!r} is not a distinctive word")
            if not re.fullmatch(r"[a-z]+", form):
                raise ValueError(f"insertion suffix {word!r} maps to {form!r}, not an eId form")
        return value

    @field_validator("citation_successors", "prose_precursors", "sameline_precursors")
    @classmethod
    def _cue_words_are_not_blank(cls, value: list[str]) -> list[str]:
        if any(not word.strip() or word.strip() != word for word in value):
            raise ValueError("a cue word cannot be blank or carry surrounding whitespace")
        return value

    # Words that introduce a citation's number ("Pasal 41 ayat (3)"). A line
    # ending in one of these makes the bracketed number on the next line part
    # of the citation, not a structural marker opening a new provision.
    reference_nouns: list[str] = Field(default_factory=list)
    # True where the drafting standard puts every article inside a container, so
    # an article before the first one is a citation in the opening material.
    # Indonesian laws cite the constitution that way in `Mengingat`.
    body_opens_with_container: bool = False
    example_markers: list[str] = Field(
        default_factory=list,
        description=(
            "Lead-ins that introduce specimen legislation in a drafting manual "
            "(`Contoh`), whose markers are displayed rather than enacted."
        ),
    )
    # OCR damage the sources carry, tolerated for this jurisdiction only and
    # byte-identical elsewhere. `missing_separator` is "Pasal24".
    marker_tolerances: list[
        Literal["missing_separator", "keyword_glyph", "digit_glyph", "split_number"]
    ] = Field(default_factory=list)
    long_title_lead_ins: list[str] = Field(
        default_factory=list,
        description=(
            "Openings of the document's own long title (`An Act`), marked up as "
            "`<longTitle>` so the official title comes from the text rather than "
            "from a filename."
        ),
    )
    citation_lead_ins: list[str] = Field(
        default_factory=list,
        description=(
            "Phrases introducing a law's own short name (`may be cited as`), the "
            "common-law short-title device."
        ),
    )
    citation_articles: list[str] = Field(
        default_factory=list,
        description=(
            "Articles that may sit between the clause and the name (`the`). "
            "Language data: most languages have none, and no article is assumed."
        ),
    )
    citation_subjects: list[str] = Field(
        default_factory=list,
        description=(
            "How an instrument refers to itself when it declares its short name "
            "(`This Act`). Required alongside `citation_lead_ins`: the same phrasing "
            "names the schools and offices a law creates, so an unanchored clause "
            "takes the school's name for the law's."
        ),
    )
    designation_rule: Literal["eu_instrument"] | None = Field(
        default=None,
        description=(
            "Derives a short name from the title's own grammar where the tradition "
            "declares no short title. EU acts are cited by form and number."
        ),
    )


class DisplayConfig(BaseModel):
    model_config = _LOOSE

    heading_type: str = "marginal_note"
    heading_case: str = "title_case"
    script: str = "latin"
    rtl: bool = False
    font_preference: str = "serif"
    # The pair a publisher prints around amended words; brackets are only the
    # commonest. A list because a tuple crosses the schema boundary as unknown.
    amendment_markers: list[str] = Field(
        default_factory=lambda: ["[", "]"], min_length=2, max_length=2
    )


class ValidationConfig(BaseModel):
    model_config = _LOOSE

    status: ValidationStatus = "draft"
    documents_examined: int = 0
    bluebell_tested: bool = False
    unresolved_ambiguities: int = 0
    note: str = ""
    documents_examined_note: str | None = None
    ambiguities: list[str] = Field(default_factory=list)
    # Some configs use a URL string, others a bool flag ("does laws.africa cover this?").
    laws_africa_reference: str | bool | None = None


class IdentificationConfig(BaseModel):
    model_config = _LOOSE

    system: str = ""
    pattern: str = ""
    eli_base: str = ""


# Each kind selects a concrete Acquirer; new values only when one ships.
SourceAdapterKind = Literal[
    "akn_native",  # Native AKN3 export. legislation.gov.uk, laws.africa.
    "eurlex_cellar",  # EUR-Lex Cellar (AKN4EU).
    "html_portal",  # HTML portal, pages + linked PDFs.
    "pdf_gazette",  # Date-keyed PDF gazette (qbz.gov.al, JORF).
    "paragraf_propisi",  # Paragraf Lex slug-keyed PDFs (rs).
    "sparql",  # SPARQL endpoint (non-Cellar).
    "bulk_xml_archive",  # Bulk XML archives (USLM, Indigo dumps).
]


# SourceAdapter kind → tier rank. Tier 1 native AKN; Tier 2 structured /
# bespoke; Tier 3 unstructured PDFs. Used by JurisdictionConfig.effective_tier.
_ADAPTER_TIER: dict[SourceAdapterKind, Literal[1, 2, 3]] = {
    "akn_native": 1,
    "eurlex_cellar": 1,
    "html_portal": 2,
    "sparql": 2,
    "bulk_xml_archive": 2,
    "paragraf_propisi": 3,
    "pdf_gazette": 3,
}


AknSourceFormat = Literal[
    "akn_xml",
    "akn_xml_v2",
    "bespoke_xml",
    "structured_html",
    "html",
    "pdf",
    "pdf_scanned",
    "docx",
    "rtf",
    "json",
    "csv",
    "rdf",
]

AknSourceProtocol = Literal[
    "http",
    "rest",
    "odata",
    "sru",
    "oai_pmh",
    "soap",
    "sparql",
    "bulk_dump",
]

AknSourceAuthScheme = Literal[
    "none",
    "token",
    "bearer",
    "api_key",
    "operator_code",
    "basic",
    "soap_session",
    "email_registration",
]


# Format to fetchability tier, AKN-native down to binary. Drives
# `effective_tier` so the atlas reflects what is fetchable here, not what is
# nominally configured.
_FORMAT_TIER: dict[str, Literal[1, 2, 3]] = {
    "akn_xml": 1,
    "akn_xml_v2": 1,
    "bespoke_xml": 2,
    "structured_html": 2,
    "json": 2,
    "csv": 2,
    "rdf": 2,
    "html": 3,
    "pdf": 3,
    "pdf_scanned": 3,
    "docx": 3,
    "rtf": 3,
}


class AknSource(BaseModel):
    """One upstream that yields documents convertible to AKN.

    Distinct from `SourceAdapter`, the acquisition substrate keyed by adapter kind:
    this is a format-and-protocol pairing declared in `config.json#akn_sources`.
    Both coexist during the schema migration and `effective_tier` walks both; legacy
    configs without `format`/`protocol` default to PDF over HTTP.
    """

    model_config = _STRICT

    name: str
    url_template: str
    format: AknSourceFormat = "pdf"
    protocol: AknSourceProtocol = "http"
    coverage: Literal["complete", "partial", "stub", "unknown"] = "unknown"
    coverage_note: str | None = None
    schema_id: str | None = None
    schema_url: str | None = None
    license: str | None = None
    auth_scheme: AknSourceAuthScheme = "none"
    auth_env_var: str = ""
    # Token-gated source: skip in `effective_tier` when the env var is unset
    # unless `fallback_format` declares a usable downgrade.
    requires_token: bool = False
    fallback_format: AknSourceFormat | None = None
    # Free-text, these configs were hand-written before the structured shape
    # landed and frequently carry narrative provenance.
    description: str = ""


class DiscoveryConfig(BaseModel):
    """Config-driven discovery for adapters that support it. The *basket
    definition* (which acts to find) lives here as per-doctype query strings;
    the portal mechanics (request shape, facets) stay in the adapter."""

    model_config = _STRICT

    queries: dict[str, str] = Field(default_factory=dict)
    max_total: int = Field(default=5000, gt=0)
    output: str | None = None  # basket path, relative to repo root


class SourceAdapter(BaseModel):
    """One upstream portal for this jurisdiction; multiple permitted.

    url_template placeholders: {country} {doctype} {year} {number} {frbr_uri}. Auth
    tokens resolve from the env var named by auth_env_var, never stored here.
    """

    model_config = _STRICT

    kind: SourceAdapterKind
    name: str
    base_url: str | None = None
    url_template: str | None = None
    auth_scheme: Literal["none", "token", "bearer", "oauth"] = "none"
    auth_env_var: str = ""
    # Names a registered structured parser (codify.pipeline.parsers) that
    # converts this adapter's primary body deterministically; None -> the
    # universal structuring lane.
    parser: str | None = None
    rate_limit_per_minute: int | None = None
    robots_respect: bool = True
    licence: str | None = None
    notes: str = ""
    discovery: DiscoveryConfig | None = None

    @model_validator(mode="after")
    def _auth_env_var_required_when_secured(self) -> "SourceAdapter":
        if self.auth_scheme != "none" and not self.auth_env_var:
            raise ValueError(
                f"SourceAdapter {self.name!r}: auth_scheme={self.auth_scheme!r} "
                f"requires auth_env_var (the env var name holding the secret)"
            )
        return self


class LegalEra(BaseModel):
    """A period whose documents share a drafting convention and authority.

    Structural quality tracks era more than jurisdiction, so it is reported per era.
    ``from_``/``to`` are inclusive ISO dates, either may be open, and the field is
    spelled ``from`` in the config files.
    """

    model_config = _LOOSE

    id: str
    label: str = ""
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    # The calendar this era's instruments are dated in, where it differs from the
    # jurisdiction's current one. None means it does not differ. Same closed set
    # as the jurisdiction's own field, so the two cannot drift apart.
    calendar: Calendar | None = None
    enacting_authority: str = ""
    document_class: str = ""
    # Phrases opening the attestation that closes a document of this era, so the
    # promulgation and signature stop being read as the last article's text.
    # Era-scoped because the wording follows the enacting authority.
    closing_phrases: list[str] = Field(default_factory=list)
    # Regex for how a footnote marker opens in this era. The modern set uses
    # superscripts and the older ones a parenthesised numeral; declaring only
    # one form silently drops the other era.
    footnote_markers: list[str] = Field(default_factory=list)
    # How this era's offices are named in provision text. Review candidates, not
    # verdicts: a match says the office is named, not that the provision grants it
    # a power, since the same name appears in signature blocks and definitions.
    # Opt-in per era, because an era ending is not its authority being abolished.
    authority_patterns: list[str] = Field(default_factory=list)

    def year_range(self) -> tuple[int | None, int | None]:
        """The era's bounds as years, for matching a work whose date is a bare year."""

        def year(value: str | None) -> int | None:
            return int(value[:4]) if value and value[:4].isdigit() else None

        return year(self.from_), year(self.to)


class JurisdictionConfig(BaseModel):
    model_config = _LOOSE

    code: str
    name: str
    name_local: str | None = None
    name_en: str | None = None
    # ISO-639-3 language code → localised jurisdiction name. Used by the
    # PDF exporter to render the cover in the target language without a
    # hardcoded Python map. Mirror of `Law.title_translations`.
    name_translations: dict[str, str] = Field(default_factory=dict)
    type: JurisdictionType = "national"
    tradition: list[Tradition]

    @field_validator("tradition")
    @classmethod
    def tradition_not_empty(cls, v: list[Tradition]) -> list[Tradition]:
        if not v:
            raise ValueError("tradition list must not be empty")
        return v

    calendar: Calendar = "gregorian"
    languages: list[str]
    authoritative_language: str | None = None

    document_classes: dict[str, DocumentClass] = Field(default_factory=dict)
    default_document_class: str = "act"

    frbr: FrbrConfig | None = None
    numbering: NumberingConfig | None = None
    enacting_formulae: list[EnactingFormula] = Field(default_factory=list)
    # Evidence the source carries its own formula, so no fallback is injected.
    # Never a cut point: a phrase here may open the instrument, not close it.
    enacting_formula_markers: list[str] = Field(default_factory=list)
    # Phrases declaring the instrument adopts another body's numbered text, which
    # every count over the document must read as an attachment, not its own structure.
    adoption_markers: list[str] = Field(default_factory=list)
    # Phrases the recitals precede, after which pre-body text is page bleed.
    opening_material_terminators: list[str] = Field(default_factory=list)
    # Captions opening an attachment, whose numbering restarts the body's.
    attachments: list[AttachmentCaption] = Field(default_factory=list)
    # Whole-string markers for missing or struck content, beyond the platform default.
    placeholders: list[PlaceholderMarker] = Field(default_factory=list)
    amendments: AmendmentConfig | None = None
    core_tlcs: list[TLCEntry] = Field(default_factory=list)
    structuring: StructuringConfig | None = None
    supranational_memberships: list[SupranationalMembership] = Field(default_factory=list)
    # Non-membership binding regimes (treaty frameworks, bilateral regimes,
    # constitutional annexes). Populated in migration for BA, PW, MH, VA
    # etc., where the existing "membership" modelling was semantically wrong.
    binding_regimes: list[BindingRegime] = Field(default_factory=list)
    # Foreign corpora still residually applicable here: pre-independence or
    # occupying-power law left in force, VA for suppletive Italian law, and so on.
    # Enables amendment/consolidation agents to traverse layered systems.
    residual_legal_systems: list[ResidualLaw] = Field(default_factory=list)
    # Drafting periods, for reporting quality per era rather than per corpus.
    # Empty means one undivided population. Distinct from `CalendarRule.eras`,
    # which is a regnal year table.
    legal_eras: list[LegalEra] = Field(default_factory=list)
    display: DisplayConfig | None = None
    validation: ValidationConfig | None = None
    # Running header and footer text that leaks into article bodies during OCR.
    # Stripped from `<p>` before translation so the LLM never sees it. Literal
    # regex strings, so a profile author adds patterns without touching Python.
    ocr_header_patterns: list[str] = Field(default_factory=list)

    # Furniture that shares a line with real content, so whole-line disposal would
    # take the content with it: a portal credit run, a stamped URL. Narrower than
    # `ocr_header_patterns`, which also matches the real title on page one.
    furniture_inline_patterns: list[str] = Field(default_factory=list)

    # Legal vocabulary particular to this jurisdiction, folded into the language's
    # word list when a page is scored. A term naming one jurisdiction is that
    # jurisdiction's data, not the language's.
    lexicon_words: list[str] = Field(default_factory=list)

    # Furniture whose whole line is disposable: mastheads, page numbers, stamps.
    # Narrower than `ocr_header_patterns`, which also carries the running title
    # and so matches the real title on page one. Substring scrubbing is safe for
    # both sets, whole-line disposal only for this one.
    furniture_line_patterns: list[str] = Field(default_factory=list)

    # The same two vocabularies for a jurisdiction that declares no eras, which
    # is 259 of the 261 configs. An era's own lists win where it declares them.
    closing_phrases: list[str] = Field(default_factory=list)
    footnote_markers: list[str] = Field(default_factory=list)

    # Minimum ratio of captured anchors over marker literals in the source.
    # Ingest fails below it, so a silently-partial AKN cannot ship. 0.80
    # tolerates in-prose cross-references without over-firing. 0.0 disables.
    min_anchor_coverage: float = Field(default=0.80, ge=0.0, le=1.0)

    # Share of a document's contents listing that may lack a body before the
    # source is read as truncated. Off by default: the separating value was
    # measured on the corpora that declare it, and a partial corpus cannot
    # evidence a threshold for a series it has never seen. 0.0 disables.
    source_truncation_floor: float = Field(default=0.0, ge=0.0, le=1.0)

    # Read by codify.acquisition.
    source_adapters: list[SourceAdapter] = Field(default_factory=list)
    # Legacy / hand-curated list of upstreams. Coexists with `source_adapters`
    # during the schema-migration window; `effective_tier` walks both.
    akn_sources: list[AknSource] = Field(default_factory=list)

    # Supranational-only fields
    members: list[str] = Field(default_factory=list)
    akn_profile: str | None = None
    identification: IdentificationConfig | None = None

    # Free-text note at the jurisdiction level for cross-cutting guidance.
    note: str = ""

    # Promoted from registry.json, which is derived rather than maintained.
    # Ladder: 1 AKN native, 2 XML-structured, 3 HTML anchors, 4 PDF-only,
    # 5 no online statute base.
    tier: Literal[1, 2, 3, 4, 5] | None = None
    status: ValidationStatus | None = None
    coverage_status: Literal["in_scope", "subsumed", "no_corpus", "deferred"] = "in_scope"
    subsumed_by: str | None = None

    # Fictional jurisdictions authored as synthetic test material (the AKN-spine
    # scan factory). Discovered and validated like any other config, but kept out
    # of the derived registry and the world atlas, which are real-only.
    synthetic: bool = False

    # Ships in the open core. Defaults False, so a jurisdiction joins the open
    # set by a deliberate edit: excluded by default rather than included and
    # simplified, because a simplified config is still a tell about what was
    # studied.
    public_reference: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_tier(self) -> Literal[1, 2, 3, 4, 5] | None:
        """Tier actually achievable in the current environment.

        Walks both `source_adapters` and the legacy `akn_sources`, taking the best
        (lowest) tier. Token-gated entries with an unset env var are skipped unless the
        source declares a `fallback_format`, whose tier contributes instead.
        """
        best: int | None = None

        for sa in self.source_adapters:
            if sa.auth_scheme != "none" and not os.environ.get(sa.auth_env_var, ""):
                continue
            rank = _ADAPTER_TIER.get(sa.kind)
            if rank is None:
                continue
            if best is None or rank < best:
                best = rank

        for src in self.akn_sources:
            # Token-gated source: env var must be set, else fall back if declared.
            gated = src.requires_token or (
                src.auth_scheme not in ("none",) and src.auth_env_var != ""
            )
            usable_format: str | None = src.format
            if gated and not os.environ.get(src.auth_env_var, ""):
                usable_format = src.fallback_format
            if usable_format is None:
                continue
            rank = _FORMAT_TIER.get(usable_format)
            if rank is None:
                continue
            if best is None or rank < best:
                best = rank

        if best is None:
            return self.tier
        return cast("Literal[1, 2, 3]", best)

    @model_validator(mode="after")
    def _subsumed_by_consistency(self) -> "JurisdictionConfig":
        if self.coverage_status == "subsumed" and not self.subsumed_by:
            raise ValueError(
                f"jurisdiction {self.code!r} has coverage_status='subsumed' but no subsumed_by code"
            )
        if self.coverage_status != "subsumed" and self.subsumed_by:
            raise ValueError(
                f"jurisdiction {self.code!r} has subsumed_by={self.subsumed_by!r} "
                f"but coverage_status={self.coverage_status!r}; subsumed_by is "
                "only meaningful when coverage_status == 'subsumed'"
            )
        return self

    @model_validator(mode="after")
    def _resolve_document_class_extends(self) -> "JurisdictionConfig":
        # A class that extends another is the parent with the child's fields laid over.
        parents = {name: dc.extends for name, dc in self.document_classes.items()}
        for name, dc in self.document_classes.items():
            if dc.extends is None:
                continue
            parent = self.document_classes.get(dc.extends)
            if parent is None or parents[dc.extends] is not None:
                raise ValueError(
                    f"document class '{name}' extends '{dc.extends}', which is "
                    "missing or itself extends another class"
                )
            overlay = {k: getattr(dc, k) for k in dc.model_fields_set if k != "extends"}
            self.document_classes[name] = parent.model_copy(update=overlay)
        return self

    @model_validator(mode="after")
    def _title_numbered_classes_have_a_grammar(self) -> "JurisdictionConfig":
        # Without the grammar the forced number would be the content address.
        if self.frbr is not None and self.frbr.title_identity is not None:
            return self
        for name, dc in self.document_classes.items():
            if dc.number_source == "title_identity":
                raise ValueError(
                    f"document class '{name}' sets number_source='title_identity' "
                    "but frbr.title_identity is not declared"
                )
        return self

    @model_validator(mode="after")
    def _default_document_class_exists(self) -> "JurisdictionConfig":
        if self.document_classes and self.default_document_class not in self.document_classes:
            raise ValueError(
                f"default_document_class '{self.default_document_class}' "
                f"not in document_classes keys {sorted(self.document_classes.keys())}"
            )
        return self

    @model_validator(mode="after")
    def _national_cannot_claim_direct_effect(self) -> "JurisdictionConfig":
        if self.type == "national":
            for dc_name, dc in self.document_classes.items():
                if dc.direct_effect is True:
                    raise ValueError(
                        f"document class '{dc_name}': national jurisdictions cannot "
                        f"set direct_effect=true (reserved for supranational bodies)"
                    )
        return self

    @model_validator(mode="after")
    def _classification_rule_targets_exist(self) -> "JurisdictionConfig":
        if not self.structuring or not self.structuring.classification_rules:
            return self
        for rule in self.structuring.classification_rules:
            if rule.target_document_class not in self.document_classes:
                raise ValueError(
                    f"classification_rule target '{rule.target_document_class}' "
                    f"not in document_classes {sorted(self.document_classes.keys())}"
                )
        return self

    @model_validator(mode="after")
    def _discovery_queries_are_doctypes(self) -> "JurisdictionConfig":
        # A discovery query keyed on a doctype the jurisdiction doesn't declare
        # is a typo/rename drift; catch it at config load, not at discover() time.
        if not self.document_classes:
            return self
        for sa in self.source_adapters:
            if sa.discovery is None:
                continue
            unknown = set(sa.discovery.queries) - set(self.document_classes)
            if unknown:
                raise ValueError(
                    f"source_adapter {sa.name!r} discovery queries {sorted(unknown)} "
                    f"not in document_classes {sorted(self.document_classes.keys())}"
                )
        return self

    @model_validator(mode="after")
    def _residual_source_jurisdictions_exist(self) -> "JurisdictionConfig":
        # Existence check, not a load: loading would recurse into config
        # resolution. An unresolved JURISDICTIONS_DIR (test harnesses, alternate
        # working directories) skips the check rather than failing every entry.
        if not self.residual_legal_systems or not JURISDICTIONS_DIR.exists():
            return self
        for r in self.residual_legal_systems:
            path = JURISDICTIONS_DIR / r.source_jurisdiction / "config.json"
            if not path.exists():
                raise ValueError(
                    f"residual_legal_systems: source_jurisdiction "
                    f"'{r.source_jurisdiction}' has no config at {path}"
                )
        return self

    @model_validator(mode="after")
    def _hierarchy_levels_coherent(self) -> "JurisdictionConfig":
        for dc_name, dc in self.document_classes.items():
            _check_hierarchy_levels(f"document class '{dc_name}'", dc.hierarchy)
        # An attachment declares its own levels and needs the same contract:
        # its ranks are merged into the containment order like any other.
        for att in self.attachments:
            _check_hierarchy_levels(f"attachment '{att.caption}'", att.hierarchy)
        return self

    def classify_document_class(
        self,
        *,
        title: str = "",
        preamble: str = "",
        issuer_role: str = "",
        gazette_series: str = "",
        doc_date: date | None = None,
    ) -> str | None:
        """First matching classification rule's target class, or ``None``.

        ``None`` means no rule matched and the caller falls back to
        ``default_document_class``: an unmatched document is unclassified, not
        misclassified.

        A signal the caller has no text for cannot match, so a rule keyed on it is
        skipped rather than treated as satisfied. Date bounds apply to every signal, and
        a rule with bounds but no date to test against is skipped for the same reason.
        """
        # OCR'd Arabic titles often store hamza decomposed (bare alef + U+0654)
        # where the rule patterns use the precomposed letter; canonically equal,
        # byte-different, so match on NFC. 61% of the PS corpus is affected,
        # including the Basic Law and every presidential decision.
        title = unicodedata.normalize("NFC", title)
        preamble = unicodedata.normalize("NFC", preamble)
        issuer_role = unicodedata.normalize("NFC", issuer_role)
        gazette_series = unicodedata.normalize("NFC", gazette_series)
        rules = self.structuring.classification_rules if self.structuring else []
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            if rule.date_from or rule.date_until:
                if doc_date is None:
                    continue
                if rule.date_from and doc_date < rule.date_from:
                    continue
                if rule.date_until and doc_date > rule.date_until:
                    continue
            if rule.signal == "date_range":
                if rule.date_from or rule.date_until:
                    return rule.target_document_class
                continue
            haystack = {
                "title_regex": title,
                "preamble_match": preamble,
                "issuer_role": issuer_role,
                "gazette_series": gazette_series,
            }.get(rule.signal, "")
            if not haystack or not rule.pattern:
                continue
            flags = re.IGNORECASE if "i" in rule.pattern_flags.lower() else 0
            try:
                matched = re.search(rule.pattern, haystack, flags) is not None
            except re.error:
                # A rule nobody can compile must not silently classify every
                # document that reaches it.
                logger.warning(
                    "classification_rule_pattern_invalid",
                    country=self.code,
                    signal=rule.signal,
                    pattern=rule.pattern[:80],
                )
                continue
            if matched:
                return rule.target_document_class
        return None

    def get_document_class(self, doctype: str) -> DocumentClass | None:
        explicit = self.document_classes.get(doctype)
        if explicit is not None:
            return explicit
        return self.document_classes.get(self.default_document_class)


# --- Loader ---


class ResolvedConfig(BaseModel):
    """Which jurisdiction config a run used, and where it came from.

    Recorded so a bundle can be read back and its provenance established. The
    digest is what distinguishes two runs of the same code against a config that
    changed between them, which the code alone cannot.
    """

    code: str
    path: str | None = None
    sha256: str | None = None

    @property
    def found(self) -> bool:
        return self.path is not None


def resolve_config(country_code: str) -> ResolvedConfig:
    """Identify the config for `country_code` without loading it.

    Absence is a `found` of False, not a raise: the callers report which
    jurisdiction was missing and where they looked, which a traceback buries.
    """
    code = country_code.strip().lower()
    if "/" in code or ".." in code:
        return ResolvedConfig(code=code)
    path = JURISDICTIONS_DIR / code / "config.json"
    if not path.exists():
        return ResolvedConfig(code=code)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Recorded relative to the tree root, not absolutely: a bundle travels, and
    # an absolute path names the machine that produced it rather than the config.
    try:
        where = str(path.relative_to(JURISDICTIONS_DIR.parents[1]))
    except ValueError:
        where = str(path)
    return ResolvedConfig(code=code, path=where, sha256=digest)


# Named in every refusal: the fix is always to write this file.
CONFIG_HELP = (
    "write {path}: `code`, `name`, `tradition` and `languages` are required, "
    "everything else defaults; see docs/jurisdictions/adding-a-jurisdiction.md"
)


def log_config_absent(country_code: str, degraded: str) -> None:
    """For a caller that probes and carries on. `degraded` says what the reader
    loses, in their words, not the field's."""
    logger.warning(
        "jurisdiction_config_absent",
        jurisdiction_code=country_code,
        degraded=degraded,
        fix=CONFIG_HELP.format(path=JURISDICTIONS_DIR / country_code / "config.json"),
    )


class JurisdictionConfigError(LookupError):
    """A caller named a jurisdiction and no config answered. A `LookupError`
    because 78 broad `except ValueError` sites would default past it."""


class JurisdictionDataMissing(JurisdictionConfigError):
    """The data is not in this tree at all, as opposed to being present and
    wrong. Only absence is a legitimate skip; a config that ships and
    contradicts itself is a defect wherever it is read."""


# A config that is absent or does not validate. A caller tolerating its own
# failures re-raises these first: a thinner result is not a fallback.
CONFIG_FAULTS: tuple[type[Exception], ...] = (JurisdictionConfigError, ValidationError)


def load_config(country_code: str) -> JurisdictionConfig:
    """The jurisdiction's config, or a fault.

    Absence raises: a caller receiving nothing produced a thinner result.
    """
    config = try_load_config(country_code)
    if config is not None:
        return config
    # A code no tree could ship is malformed input, not absent data.
    if "/" in country_code or ".." in country_code:
        raise JurisdictionConfigError(f"not a jurisdiction code: {country_code!r}")
    if not JURISDICTIONS_DIR.exists():
        raise JurisdictionDataMissing(
            f"no jurisdiction data at {JURISDICTIONS_DIR}: the package was installed "
            "without it, so every config lookup would answer with defaults; "
            + CONFIG_HELP.format(path=JURISDICTIONS_DIR / country_code / "config.json")
        )
    raise JurisdictionDataMissing(
        f"no jurisdiction config for {country_code!r} in {JURISDICTIONS_DIR}; "
        + CONFIG_HELP.format(path=JURISDICTIONS_DIR / country_code / "config.json")
    )


@lru_cache(maxsize=64)
def try_load_config(country_code: str) -> JurisdictionConfig | None:
    """The config, or None if the code is malformed or names no config file.

    For a caller asking whether one exists; anything needing a config wants
    `load_config`. Clear cache with ``try_load_config.cache_clear()``.
    """
    if "/" in country_code or ".." in country_code:
        return None
    path = JURISDICTIONS_DIR / country_code / "config.json"
    if not path.exists():
        return None
    return JurisdictionConfig.model_validate_json(path.read_text())


class FrbrCountry(NamedTuple):
    """The country segment a jurisdiction's work URIs carry, and its provenance.

    `resolved` is False when no config supplied the segment and it was guessed
    from the jurisdiction code. A caller minting an identity reads it to record
    the segment as guessed rather than authoritative.
    """

    segment: str
    resolved: bool


def placeholder_statuses_for(work_uri: str | None) -> tuple[tuple[str, str], ...]:
    """`(pattern, status)` for the platform placeholder marker and the jurisdiction's
    own, keyed by the work URI's country like the attachment rule, so ingest and
    every later pass agree. The platform marker stands for untranscribed content."""
    from codify.akn.frbr import country_of

    country = country_of(work_uri) if work_uri else None
    return _placeholder_entries(try_load_config(country) if country else None)


def placeholder_statuses_for_code(jurisdiction_code: str) -> tuple[tuple[str, str], ...]:
    """As `placeholder_statuses_for`, from a jurisdiction code at ingest, before the
    document carries a work URI the row could be keyed by."""
    return _placeholder_entries(try_load_config(jurisdiction_code.strip().lower()))


def _placeholder_entries(config: JurisdictionConfig | None) -> tuple[tuple[str, str], ...]:
    from codify.quality.sentinels import PLACEHOLDER_PATTERN

    own = tuple((m.pattern, str(m.status)) for m in config.placeholders) if config else ()
    return ((PLACEHOLDER_PATTERN, str(PlaceholderStatus.INCOMPLETE)), *own)


def placeholder_statuses_by_code() -> dict[str, tuple[tuple[str, str], ...]]:
    """Each jurisdiction that declares its own markers, with its full entry list,
    for a sweep not scoped to one jurisdiction. Reads each config once."""
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for entry in sorted(JURISDICTIONS_DIR.glob("*/config.json")):
        config = try_load_config(entry.parent.name)
        if config is not None and config.placeholders:
            out[entry.parent.name] = _placeholder_entries(config)
    return out


def placeholder_markers_for(work_uri: str | None) -> tuple[str, ...]:
    """The patterns alone, for the row-level exclusion rule."""
    return tuple(pattern for pattern, _ in placeholder_statuses_for(work_uri))


def resolve_frbr_country(jurisdiction_code: str) -> FrbrCountry:
    """The country segment for this jurisdiction, with whether a config set it.

    Not always the jurisdiction code: `gb-eng` publishes under `/akn/gb/`, `eac`
    under `/akn/aa-eac/`. A code with no config, or a config with no FRBR block,
    falls back to the code itself and reports `resolved=False`.
    """
    code = jurisdiction_code.strip().lower()
    cfg = try_load_config(code)
    if cfg is None or cfg.frbr is None:
        return FrbrCountry(code, resolved=False)
    return FrbrCountry(cfg.frbr.country_code.strip().lower(), resolved=True)


def frbr_country(jurisdiction_code: str) -> str:
    """The country segment this jurisdiction's work URIs carry.

    A missing config guesses the code as its own country and warns, since a
    guessed segment becomes a law's citation identity. Read `resolve_frbr_country`
    where the caller must branch on whether the segment was guessed.
    """
    country = resolve_frbr_country(jurisdiction_code)
    if not country.resolved:
        logger.warning("frbr_country_guessed_from_code", jurisdiction=jurisdiction_code)
    return country.segment


def work_uri_is_homed(work_uri: str, jurisdiction_code: str) -> bool:
    """Whether a work URI's country names this jurisdiction.

    An empty default, not `country_of`'s `"xx"`: `xx` is a real code here, so
    the default would let a country-less URI match it.
    """
    from codify.akn.frbr import country_of

    return country_of(work_uri, default="").strip().lower() == frbr_country(jurisdiction_code)


@lru_cache(maxsize=32)
def load_short_titles(country_code: str) -> dict[str, str]:
    """{normalised short title: work URI} from short_titles.json, {} if absent."""
    if "/" in country_code or ".." in country_code:
        return {}
    path = JURISDICTIONS_DIR / country_code / "short_titles.json"
    if not path.exists():
        return {}
    loaded: dict[str, str] = json.loads(path.read_text())
    return loaded


def authoritative_language(jurisdiction: str) -> str:
    """The jurisdiction's authoritative ISO 639-3 language from config, the
    single source for adapter fallbacks. Raises if the config is missing or
    declares no language (fail loud rather than ingest under a wrong language)."""
    cfg = load_config(jurisdiction)
    lang = cfg.authoritative_language or (cfg.languages[0] if cfg.languages else None)
    if lang is None:
        raise ValueError(f"jurisdiction {jurisdiction!r} declares no languages")
    return lang


def act_short_label(jurisdiction: str, doctype: str) -> str:
    """Compact doctype label from config (e.g. al/vendim → "VKM").
    Falls back to the title-cased doctype when unconfigured."""
    cfg = try_load_config(jurisdiction)
    if cfg is not None:
        dc = cfg.document_classes.get(doctype)
        if dc is not None and dc.short_label:
            return dc.short_label
        if dc is None:
            # Doctype unknown to the config, a typo/drift, not a missing label.
            logger.warning("doctype_not_in_config", jurisdiction=jurisdiction, doctype=doctype)
    return doctype.title() if doctype else "Act"


def act_short_code(frbr_work_uri: str, doctype: str) -> str:
    """`/akn/al/act/vendim/2021/285` + "vendim" → "VKM 285/2021". The
    jurisdiction comes from the FRBR; number/year are its last two segments."""
    from codify.akn.frbr import country_of

    segs = [s for s in (frbr_work_uri or "").split("/") if s]
    label = act_short_label(country_of(frbr_work_uri), doctype)
    # A real act work URI ends in …/{year}/{number} with a numeric year; anything
    # else is malformed, return the bare label rather than a garbage "X al/akn".
    if len(segs) < 2 or not segs[-2].isdigit():
        logger.warning("act_short_code_malformed_frbr", frbr_work_uri=frbr_work_uri)
        return label
    year, number = segs[-2], segs[-1]
    return f"{label} {number}/{year}".strip()


@lru_cache(maxsize=1)
def ordinal_word_folds() -> dict[str, str]:
    """Every declared ordinal word mapped to its integer, across all jurisdictions. The
    words are language-distinctive, so one table serves callers deriving an eId
    without knowing the country.

    Read straight from JSON: a sweep of 257 configs would evict `load_config`'s
    64-entry cache.
    """
    out: dict[str, str] = {}
    for entry in sorted(JURISDICTIONS_DIR.glob("*/config.json")):
        try:
            raw = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("ordinal_words_unreadable", config=str(entry))
            continue
        words = (raw.get("structuring") or {}).get("ordinal_words") or {}
        for word, value in words.items():
            out[word] = str(value)
            out[word.upper()] = str(value)
    return out


@lru_cache(maxsize=1)
def insertion_suffix_folds() -> dict[str, str]:
    """Every declared insertion suffix mapped to its ASCII form, across all
    jurisdictions; one table, like `ordinal_word_folds`, for eId derivation."""
    out: dict[str, str] = {}
    for entry in sorted(JURISDICTIONS_DIR.glob("*/config.json")):
        try:
            raw = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("insertion_suffixes_unreadable", config=str(entry))
            continue
        words = (raw.get("structuring") or {}).get("insertion_suffixes") or {}
        out.update({str(word): str(value) for word, value in words.items()})
    return out


def fold_inserted_suffix(num: str) -> str | None:
    """`"5 bis"` as `"5bis"` when the number ends in a declared suffix, else None."""
    value = num.strip()
    folds = insertion_suffix_folds()
    # Longest word first: the suffix is matched at the end, never inferred.
    for word in sorted(folds, key=lambda w: (-len(w), w)):
        base = value[: -len(word)].rstrip() if value.endswith(word) else ""
        if base:
            # A slashed base keys the way the parser writes it.
            return re.sub(r"(?<=\d)/(?=\d)", "-", base) + folds[word]
    return None


@lru_cache(maxsize=32)
def _all_declared_ranks() -> dict[tuple[str, str], int]:
    """Every declared rank across all jurisdictions, keyed `(code, doctype)`.

    Read straight from JSON like `ordinal_word_folds`, so a 257-config sweep does not
    evict `load_config`'s cache.
    """
    out: dict[tuple[str, str], int] = {}
    for entry in sorted(JURISDICTIONS_DIR.glob("*/config.json")):
        try:
            raw = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("search_ranks_unreadable", config=str(entry))
            continue
        for doctype, cls in (raw.get("document_classes") or {}).items():
            rank = (cls or {}).get("search_rank")
            if rank is not None:
                out[(entry.parent.name, doctype)] = int(rank)
    return out


def doctype_search_ranks(country_codes: list[str] | None) -> dict[tuple[str, str], int]:
    """Hierarchy rank per `(jurisdiction_code, doctype)` for the scope given.

    Empty when no class in scope declares one, which is most of the corpus, leaving
    pure relevance order rather than an invented one.

    Keyed by jurisdiction as well as doctype because doctype names are
    jurisdiction-local vocabularies that collide: 96 names are defined by more than
    one config, `qanun` among them, meaning a statute in one jurisdiction and a
    provincial instrument in another. `None` scope means the whole corpus, which is
    what an unscoped search passes, and is not the same as an empty list.
    """
    every = _all_declared_ranks()
    if country_codes is None:
        return every
    wanted = set(country_codes)
    return {key: rank for key, rank in every.items() if key[0] in wanted}


def load_registry() -> list[dict[str, Any]]:
    """Every jurisdiction this tree carries. Absence raises: an empty registry
    reads as a deployment with no jurisdictions rather than a missing file."""
    path = JURISDICTIONS_DIR / "registry.json"
    if not path.exists():
        raise JurisdictionDataMissing(f"no jurisdiction registry at {path}")
    entries: list[dict[str, Any]] = json.loads(path.read_text()).get("jurisdictions", [])
    return entries


@lru_cache(maxsize=64)
def load_profile(country_code: str) -> str | None:
    """Prose from `data/jurisdictions/<code>/profile.md`, or None."""
    if "/" in country_code or ".." in country_code:
        return None
    path = JURISDICTIONS_DIR / country_code / "profile.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
