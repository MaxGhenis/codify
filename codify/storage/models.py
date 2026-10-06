"""DB row models. Distinct from `codify.akn` domain models."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any, Literal

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import REAL as PG_REAL
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def uuid_pk() -> Any:
    return Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True),
    )


def ts_now() -> Any:
    return Field(
        default_factory=utcnow,
        sa_column=Column(TIMESTAMP(timezone=True), nullable=False),
    )


class Jurisdiction(SQLModel, table=True):
    __tablename__ = "jurisdictions"

    id: uuid.UUID = uuid_pk()
    code: str = Field(sa_column=Column(Text, nullable=False, unique=True))
    name: str = Field(sa_column=Column(Text, nullable=False))
    family: str | None = Field(default=None, sa_column=Column(Text))
    # Alembic migrations are the authoritative DDL.
    calendar: str = Field(default="gregorian", sa_column=Column(Text, nullable=False))
    languages: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(Text), nullable=False)
    )
    extra: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    created_at: datetime = ts_now()
    updated_at: datetime = ts_now()


class SourceDocument(SQLModel, table=True):
    """Provenance for an uploaded source file, content-addressed by sha256
    (the same digest that names its MinIO object). One row per distinct file."""

    __tablename__ = "source_documents"

    sha256: str = Field(sa_column=Column(Text, primary_key=True))
    original_filename: str = Field(sa_column=Column(Text, nullable=False))
    byte_size: int = Field(sa_column=Column(BigInteger, nullable=False))
    page_count: int | None = Field(default=None, sa_column=Column(Integer))
    object_key: str = Field(sa_column=Column(Text, nullable=False))
    jurisdiction_code: str | None = Field(default=None, sa_column=Column(Text))
    # PDF info-dict provenance, populated when source is a PDF (migration 0043).
    pdf_title: str | None = Field(default=None, sa_column=Column(Text))
    pdf_author: str | None = Field(default=None, sa_column=Column(Text))
    pdf_producer: str | None = Field(default=None, sa_column=Column(Text))
    pdf_creation_date: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
    first_ingested_at: datetime = ts_now()


class Law(SQLModel, table=True):
    __tablename__ = "laws"

    id: uuid.UUID = uuid_pk()
    jurisdiction_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("jurisdictions.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    title: str = Field(sa_column=Column(Text, nullable=False))
    # The name the law gives itself, or the designation its title implies. Null
    # for most acts; `title` is the fallback.
    short_title: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Translated titles keyed by ISO-639-3 language; the source-language title
    # stays in `title`.
    title_translations: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    # Tokenised title, short title and every translation, each in its own
    # language. Derived by `title_tokens_for`; both title write paths refresh it.
    title_tokens: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    title_search_pipeline_version: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    doctype: str = Field(sa_column=Column(Text, nullable=False))
    # 'draft' | 'enacted' (migration 0060). Drafts come from uploads whose
    # work URI carries a source-hash slug; registry syncs may stamp this.
    status: str = Field(default="enacted", sa_column=Column(Text, nullable=False))
    year: int | None = Field(default=None, sa_column=Column(Integer))
    number: str | None = Field(default=None, sa_column=Column(Text))
    frbr_work_uri: str = Field(sa_column=Column(Text, nullable=False, unique=True))
    # Gazette metadata (name/issue/year/page/date) when the source carries it.
    # none_as_null so an absent one is SQL NULL, not JSONB 'null', which the
    # fill-if-null conditional update can find.
    gazette: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB(none_as_null=True), nullable=True)
    )
    created_at: datetime = ts_now()
    updated_at: datetime = ts_now()


class Version(SQLModel, table=True):
    __tablename__ = "versions"

    id: uuid.UUID = uuid_pk()
    law_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("laws.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    expression_uri: str = Field(sa_column=Column(Text, nullable=False, unique=True))
    language: str = Field(sa_column=Column(Text, nullable=False))
    expression_date: date = Field(sa_column=Column(Date, nullable=False))
    # Content hash of the uploaded source file (== source_documents.sha256);
    # null for acquisition-adapter and pre-provenance versions.
    source_sha256: str | None = Field(
        default=None,
        sa_column=Column(
            Text,
            ForeignKey("source_documents.sha256", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    parent_version_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    akn_xml: str = Field(sa_column=Column(Text, nullable=False))
    ingested_at: datetime = ts_now()
    embedded_at: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
    acquis_chapter: int | None = Field(default=None, sa_column=Column(SmallInteger, nullable=True))
    # How the version's text was obtained: text_extraction (PDF text layer),
    # vision_ocr, or mixed. ocr_model names the vision model when any page was
    # OCR'd. NULL on versions ingested before provenance capture until backfilled.
    ocr_source: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    ocr_model: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # sha256 of the parent version's akn_xml at translation time: a
    # translation is stale when the parent no longer hashes to it. Not
    # source_sha256, which is the uploaded file.
    source_akn_sha256: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # The span of a multi-act source this version was cut from (migration 0023).
    source_span_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("source_spans.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Deterministic-repair audit trail. NULL = original ingest; non-NULL =
    # the structurer post-passes were applied to the AKN tree after ingest.
    repaired_at: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
    repair_attribution: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Stamped by the repair path alone and only moves forward, where the
    # columns above answer "who wrote last" and any backfill overwrites them.
    reviewed_at: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
    # The human who approved the repair that stamped reviewed_at (0113).
    reviewed_by: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Phase 5 composite quality grade: A|B|C|ungraded (nullable, legacy
    # pre-Phase-5 rows and non-translation versions leave this NULL).
    translation_quality_grade: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Per-provision delivery findings behind the grade (grade-and-annotate),
    # mirroring structural_quality: {eid, category, severity, detail} rows for
    # the manifest, opt-in export annotations and the reader drill-down.
    translation_quality: list[dict[str, Any]] | None = Field(default=None, sa_column=Column(JSONB))
    # Structural verdict from validate_akn: clean|warning|blocking|ungraded,
    # NULL until backfilled. structural_quality holds the findings behind it.
    structural_quality_grade: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    structural_quality: list[dict[str, Any]] | None = Field(default=None, sa_column=Column(JSONB))
    # Invisible in the AKN, so while set the version grades blocking.
    # `none_as_null` or the ORM writes JSON null, which `IS NULL` misses.
    structure_halt: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB(none_as_null=True))
    )
    # The run that wrote this row, so a replay can tell its own write from an
    # earlier one. NULL is pre-feature, never "unknown run".
    ingest_run_id: uuid.UUID | None = Field(default=None, sa_column=Column(PG_UUID(as_uuid=True)))
    # Function words per 1,000 tokens in the source. NULL when it could not be
    # judged, which is not the same as scoring zero.
    source_legibility: float | None = Field(default=None, sa_column=Column(PG_REAL, nullable=True))
    created_at: datetime = ts_now()


class VersionSourceText(SQLModel, table=True):
    """The extracted text a version was structured from, kept so the structural
    checks that read the source can run long after the ingest run is collected."""

    __tablename__ = "version_source_texts"

    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    text: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = ts_now()


class SourceSpan(SQLModel, table=True):
    """One region of a multi-act source in one generation of its split. Cuts are a
    page, a marker line and an offset into that page's text, so a re-read can find
    them again; a re-split retires the generation and writes the next."""

    __tablename__ = "source_spans"
    __table_args__ = (
        UniqueConstraint(
            "source_sha256", "generation", "ordinal", name="source_spans_generation_ordinal_key"
        ),
    )

    id: uuid.UUID = uuid_pk()
    source_sha256: str = Field(
        sa_column=Column(
            Text, ForeignKey("source_documents.sha256", ondelete="CASCADE"), nullable=False
        )
    )
    generation: int = Field(sa_column=Column(Integer, nullable=False))
    ordinal: int = Field(sa_column=Column(Integer, nullable=False))
    kind: str = Field(sa_column=Column(Text, nullable=False))
    first_page: int = Field(sa_column=Column(Integer, nullable=False))
    last_page: int = Field(sa_column=Column(Integer, nullable=False))
    start_marker: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    start_offset: int = Field(sa_column=Column(Integer, nullable=False))
    # Exclusive, in the last page's text; NULL runs to the page's end.
    end_offset: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    issue: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    heading: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    act_key: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    signals: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(Text), nullable=False, server_default="{}")
    )
    reason: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    created_at: datetime = ts_now()
    retired_at: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )


class SourceSpanPage(SQLModel, table=True):
    """A page read a span draws on; a shared boundary page belongs to two spans."""

    __tablename__ = "source_span_pages"

    span_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("source_spans.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    page_read_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("page_reads.id", ondelete="RESTRICT"),
            primary_key=True,
        )
    )


class PageRead(SQLModel, table=True):
    """What the engines made of one page of a scan, kept for the version's life."""

    __tablename__ = "page_reads"
    __table_args__ = (UniqueConstraint("run_id", "page_number", name="page_reads_run_page_unique"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # Nulled, not cascaded: the janitor collects runs at 14 days.
    run_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
        ),
    )
    # NULL until the persist step stamps it, and left NULL on a read that never
    # became a version's text.
    version_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    page_number: int = Field(sa_column=Column(Integer, nullable=False))
    engine: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    model: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    # NULL on any route that did not rasterise the page.
    dpi: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    text: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    # What the other engine made of the same page. Never authoritative.
    rival_text: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    divergence: float | None = Field(default=None, sa_column=Column(PG_REAL, nullable=True))
    # NULL means no engine looked, not that the page has no furniture.
    layout: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = ts_now()


class Section(SQLModel, table=True):
    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("version_id", "akn_eid"),)

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    parent_section_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    akn_eid: str = Field(sa_column=Column(Text, nullable=False))
    # Work-scoped id (AKN 3.0 Naming Convention). Stable across renumbering;
    # equal to akn_eid when no renumbering has occurred.
    akn_wid: str = Field(sa_column=Column(Text, nullable=False))
    akn_type: str = Field(sa_column=Column(Text, nullable=False))
    title: str | None = Field(default=None, sa_column=Column(Text))
    position: int = Field(sa_column=Column(Integer, nullable=False))
    created_at: datetime = ts_now()
    updated_at: datetime = ts_now()


class Provision(SQLModel, table=True):
    __tablename__ = "provisions"
    __table_args__ = (UniqueConstraint("version_id", "akn_eid"),)

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    section_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    akn_eid: str = Field(sa_column=Column(Text, nullable=False))
    # Work-scoped id (AKN 3.0 Naming Convention). Stable across renumbering;
    # equal to akn_eid when no renumbering has occurred.
    akn_wid: str = Field(sa_column=Column(Text, nullable=False))
    akn_type: str = Field(sa_column=Column(Text, nullable=False))
    text: str = Field(sa_column=Column(Text, nullable=False))
    position: int = Field(sa_column=Column(Integer, nullable=False))
    # False for attachment content a jurisdiction declares carries no norm:
    # an explanatory apparatus interprets but cannot be a legal basis.
    normative: bool = Field(default=True, sa_column=Column(Boolean, nullable=False))
    # True when the unit carries no law by its shape (marker, table row, digits);
    # NULL means not yet derived. Independent of `normative`, which is declared.
    excluded_from_pool: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    # Which shape excluded it (placeholder, table, digits), so a wrong exclusion is attributable.
    exclusion_reason: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Lexical-arm index input: `text` run through `codify.search.tokenise` for
    # this expression's language. NULL means the row predates the tokeniser or
    # a backfill has not reached it yet.
    search_tokens: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    search_pipeline_version: int | None = Field(
        default=None, sa_column=Column(SmallInteger, nullable=True)
    )
    created_at: datetime = ts_now()
    updated_at: datetime = ts_now()


EdgeClass = Literal[
    "freetext_reference",
    "mod_textual",
    "mod_meaning",
    "mod_scope",
    "mod_force",
    "mod_efficacy",
]


class CrossReference(SQLModel, table=True):
    __tablename__ = "cross_references"

    id: uuid.UUID = uuid_pk()
    source_provision_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("provisions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    target_provision_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("provisions.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    # Law-level resolution for bare work-URI refs (no provision fragment);
    # `target_uri` stays as provenance when this is stamped.
    target_law_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("laws.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Container-level resolution (multi-provision units live in `sections`).
    target_section_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("sections.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    target_uri: str | None = Field(default=None, sa_column=Column(Text))
    # Which resolver policy stamped this row. Null means never resolved; below
    # RESOLVER_VERSION means stamped by a policy since superseded, and the
    # resolver re-examines it rather than trusting a verdict it would not
    # reach again.
    resolver_version: int | None = Field(default=None, sa_column=Column(SmallInteger))
    # 'href' (the publisher wrote this link), 'registry' (an upstream register
    # resolved it) or 'text' (one of our passes read it out of prose). Only the
    # last needs the citing text to bear it out.
    resolution_origin: str | None = Field(default=None, sa_column=Column(Text))
    ref_type: str = Field(sa_column=Column(Text, nullable=False))
    # Coarse graph bucket: `freetext_reference` for a citation with no known
    # effect, `mod_<category>` for the five AKN modification groups. The
    # operation subtype lives in `ref_type`.
    edge_class: EdgeClass = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = ts_now()


class AnnexRow(SQLModel, table=True):
    """One table row and the context its text omits. Not a provision, but
    keyed on version and eId like any citable unit."""

    __tablename__ = "annex_rows"

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    # The <tr> eId. A row is what an amending act replaces, so it is the unit
    # an <mod> targets.
    akn_eid: str = Field(sa_column=Column(Text, nullable=False))
    table_eid: str = Field(sa_column=Column(Text, nullable=False))
    row_index: int = Field(sa_column=Column(Integer, nullable=False))
    # [{"column": int, "label": str, "text": str}], in logical column order.
    cells: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, server_default="[]")
    )
    # Ancestor path, outermost first.
    lineage: list[str] = Field(default_factory=list, sa_column=Column(ARRAY(Text), nullable=False))
    # How the context was found, so coverage reports per mechanism.
    mechanisms: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(Text), nullable=False)
    )
    resolved_text: str = Field(sa_column=Column(Text, nullable=False))
    search_tokens: str | None = Field(default=None, sa_column=Column(Text))
    # Which tokeniser wrote them, so a bump can select the stale rows rather
    # than leaving them unmatchable with no way to find them.
    search_pipeline_version: int | None = Field(
        default=None, sa_column=Column(SmallInteger, nullable=True)
    )
    created_at: datetime = ts_now()


class GoodsCodeReference(SQLModel, table=True):
    """One goods code a version states, keyed on version and eId: most sit in
    a table cell, and a row is not a provision."""

    __tablename__ = "goods_code_references"

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    akn_eid: str = Field(sa_column=Column(Text, nullable=False))
    system: str = Field(sa_column=Column(Text, nullable=False))
    # Digits only, so the spaced, dotted and solid forms all match one query.
    code: str = Field(sa_column=Column(Text, nullable=False))
    surface: str = Field(sa_column=Column(Text, nullable=False))
    # "ex" before a code restricts it to part of the heading, which changes
    # what the measure covers.
    partial: bool = Field(sa_column=Column(Boolean, nullable=False, server_default="false"))
    # `column` is a typed table column, `prose` an anchored mention. A mention
    # in a recital is not the same evidence as a listing.
    source: str = Field(sa_column=Column(Text, nullable=False))
    # The band the row sits under. A schedule grouped by country states the
    # country once, so without this the code cannot say who it binds.
    lineage: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(Text), nullable=False, server_default="{}")
    )
    created_at: datetime = ts_now()


AknCategory = Literal["textual", "meaning", "scope", "force", "efficacy"]

AknAction = Literal[
    "repeal",
    "substitution",
    "insertion",
    "replacement",
    "renumbering",
    "split",
    "join",
    "variation",
    "termModification",
    "authenticInterpretation",
    "exceptionOfScope",
    "extensionOfScope",
    "entryIntoForce",
    "endOfEnactment",
    "postponementOfEntryIntoForce",
    "prorogationOfForce",
    "reEnactment",
    "unconstitutionality",
    "entryIntoEfficacy",
    "endOfEfficacy",
    "inapplication",
    "retroactivity",
    "extraefficacy",
    "postponementOfEfficacy",
    "prorogationOfEfficacy",
]

EventType = Literal["generation", "amendment", "repeal"]


class AmendmentEffect(SQLModel, table=True):
    """One `<textualMod>` parsed from a document's `<meta><analysis>`. `akn_category`
    is which of the five AKN modification groups it belongs to, `akn_action` the
    exact `type` attribute from the OASIS enum, and `quoted` preserves the
    unbounded `<old>`/`<new>`/`<previous>` blocks one mod can carry.
    """

    __tablename__ = "amendment_effects"

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    source_akn_wid: str = Field(sa_column=Column(Text, nullable=False))
    target_frbr_uri: str = Field(sa_column=Column(Text, nullable=False))
    target_akn_wid: str | None = Field(default=None, sa_column=Column(Text))
    akn_category: AknCategory = Field(sa_column=Column(Text, nullable=False))
    akn_action: AknAction = Field(sa_column=Column(Text, nullable=False))
    quoted: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    authority_uri: str | None = Field(default=None, sa_column=Column(Text))
    mod_eid_ref: str | None = Field(default=None, sa_column=Column(Text))
    # When the effect takes effect, and whether the publisher has applied it.
    # Both nullable because the publisher's own record is incomplete: an effect
    # with no date is reported as undated, never placed on the timeline by guess.
    in_force_date: date | None = Field(default=None, sa_column=Column(Date))
    applied: bool | None = Field(default=None, sa_column=Column(Boolean))
    #: The publisher's own id for this effect, where a feed supplied one. The
    #: key a re-read updates against; null for effects a document declared.
    publisher_effect_id: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class RegistryWork(SQLModel, table=True):
    """A row of an official document registry (works we may not hold)."""

    __tablename__ = "registry_works"
    __table_args__ = (UniqueConstraint("jurisdiction_id", "external_id"),)

    id: uuid.UUID = uuid_pk()
    jurisdiction_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("jurisdictions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    external_id: str = Field(sa_column=Column(Text, nullable=False))
    ref: str = Field(sa_column=Column(Text, nullable=False))
    title: str = Field(sa_column=Column(Text, nullable=False))
    doc_types: str | None = Field(default=None, sa_column=Column(Text))
    status: int | None = Field(default=None, sa_column=Column(SmallInteger))
    law_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True), ForeignKey("laws.id", ondelete="SET NULL"), nullable=True
        ),
    )
    updated_at: datetime = ts_now()


class LifecycleEventRow(SQLModel, table=True):
    """One `<eventRef>` from a document's `<meta><lifecycle>`: the chronological
    journal of generation, amendment and repeal, complementing `amendment_effects`
    which carries the per-modification detail.
    """

    __tablename__ = "lifecycle_events"

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    event_date: date = Field(sa_column=Column(Date, nullable=False))
    event_type: EventType = Field(sa_column=Column(Text, nullable=False))
    source_uri: str | None = Field(default=None, sa_column=Column(Text))
    refers_uri: str | None = Field(default=None, sa_column=Column(Text))
    originating_uri: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class ProvisionEmbedding(SQLModel, table=True):
    """Partitioned by `jurisdiction_id`, one HNSW per partition; `version_id` is
    the in-scan filter. Both are the provision's, copied so the index can see them."""

    __tablename__ = "provision_embeddings"
    __table_args__ = (UniqueConstraint("provision_id", "model_id", "jurisdiction_id"),)

    id: uuid.UUID = uuid_pk()
    provision_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("provisions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    embedding: list[float] = Field(sa_column=Column(HALFVEC(768), nullable=False))
    model_id: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = ts_now()
    version_id: uuid.UUID = Field(sa_column=Column(PG_UUID(as_uuid=True), nullable=False))
    jurisdiction_id: uuid.UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("jurisdictions.id"), primary_key=True)
    )


class VersionUnitEmbedding(SQLModel, table=True):
    """Embedding cache keyed on (version_id, akn_eid)."""

    __tablename__ = "version_unit_embeddings"
    __table_args__ = (UniqueConstraint("version_id", "akn_eid", "model_id"),)

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    akn_eid: str = Field(sa_column=Column(Text, nullable=False))
    embedding: list[float] = Field(sa_column=Column(HALFVEC(768), nullable=False))
    model_id: str = Field(sa_column=Column(Text, nullable=False))
    # Hash of the text embedded; a unit whose folded text changed misses the cache.
    text_sha256: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = ts_now()


class FindingRow(SQLModel, table=True):
    __tablename__ = "findings"

    id: uuid.UUID = uuid_pk()
    lens_run_id: uuid.UUID = Field(sa_column=Column(PG_UUID(as_uuid=True), nullable=False))
    lens_name: str = Field(sa_column=Column(Text, nullable=False))
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    provision_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("provisions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    provision_eid: str = Field(sa_column=Column(Text, nullable=False))
    severity: str = Field(sa_column=Column(Text, nullable=False))
    confidence: float = Field(sa_column=Column(PG_REAL, nullable=False))
    rationale: str = Field(sa_column=Column(Text, nullable=False))
    recommendation: str | None = Field(default=None, sa_column=Column(Text))
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    review_status: str = Field(default="open", sa_column=Column(Text, nullable=False))
    reviewed_by: str | None = Field(default=None, sa_column=Column(Text))
    reviewed_at: datetime | None = Field(
        default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=True)
    )
    decision_note: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class FindingNoteRow(SQLModel, table=True):
    """Per-finding concordance Note. Prompt version in PK so spec bumps don't serve stale."""

    __tablename__ = "finding_notes"

    finding_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("findings.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    # VARCHAR(16) to match migration 0047, tightens schema/model alignment.
    prompt_version: str = Field(sa_column=Column(String(16), primary_key=True, nullable=False))
    body: str = Field(sa_column=Column(Text, nullable=False))
    generated_at: datetime = ts_now()


class ExaminationRow(SQLModel, table=True):
    __tablename__ = "examinations"

    id: uuid.UUID = uuid_pk()
    finding_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("findings.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    decision: str = Field(sa_column=Column(Text, nullable=False))
    actor: str = Field(sa_column=Column(Text, nullable=False))
    at: datetime = ts_now()
    note: str | None = Field(default=None, sa_column=Column(Text))
    prior_state: str | None = Field(default=None, sa_column=Column(Text))
    run_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(PG_UUID(as_uuid=True), nullable=True)
    )
    created_at: datetime = ts_now()


class PageReadDisputeRow(SQLModel, table=True):
    """A reviewer's verdict on one region of one page read: the OCR trace's label
    sink. Append-only (UPDATE blocked at the DB), and the only place a dispute is
    recorded. It never writes back to provisions: corrections flow through the
    repair loop, this is a gold-set label against the read attempt."""

    __tablename__ = "page_read_disputes"

    id: uuid.UUID = uuid_pk()
    # The read *attempt*, not the version: a re-read is a new page_reads row.
    page_read_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("page_reads.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    # The layout.blocks index the dispute points at, or NULL for a whole-page verdict.
    block_index: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    # confirmed | disputed | corrected, enforced by a CHECK on the column.
    verdict: str = Field(sa_column=Column(Text, nullable=False))
    # The gold-set label payload: what the region should have said. NULL unless corrected.
    corrected_text: str | None = Field(default=None, sa_column=Column(Text))
    note: str | None = Field(default=None, sa_column=Column(Text))
    actor: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = ts_now()


class RepairProposalRow(SQLModel, table=True):
    """One agent-proposed repair plan and its lifecycle. Low/medium risk rows
    record what a run applied (visible for review); high-risk rows queue
    pending a human decision, and only `persist_approved_repair` may turn a
    pending row into a persisted change."""

    __tablename__ = "repair_proposals"

    id: uuid.UUID = uuid_pk()
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    # Provenance, not identity: the janitor collects runs at 14 days.
    run_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
        ),
    )
    eid: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    check_name: str = Field(sa_column=Column(Text, nullable=False))
    # The full finding dict: apply-time gates need target_finding to verify
    # the repair still clears it.
    finding: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    ops: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
    reasoning: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    # low | medium | high, CHECK-enforced.
    risk_class: str = Field(sa_column=Column(Text, nullable=False))
    # pending | approved | rejected | applied | superseded, CHECK-enforced;
    # a high-risk row cannot be applied undecided (table CHECK).
    status: str = Field(sa_column=Column(Text, nullable=False))
    # The SandboxReport dump: reasons, structure changes, before/after deltas.
    audit: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    # Staleness pin: sha of versions.akn_xml when the proposal was recorded.
    # An other-writers fence only; apply re-runs the full gates regardless.
    akn_sha256: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    source_sha256: str = Field(
        default="", sa_column=Column(Text, nullable=False, server_default="")
    )
    model: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    prompt_version: str = Field(
        default="", sa_column=Column(Text, nullable=False, server_default="")
    )
    decided_by: str | None = Field(default=None, sa_column=Column(Text))
    decided_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))
    decision_note: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class SchemeMatchRow(SQLModel, table=True):
    __tablename__ = "scheme_matches"

    id: uuid.UUID = uuid_pk()
    lens_run_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("lens_runs.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    lens_name: str = Field(sa_column=Column(Text, nullable=False))
    scheme_id: str = Field(sa_column=Column(Text, nullable=False))
    confidence: float = Field(sa_column=Column(PG_REAL, nullable=False))
    findings: list[str] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
    rationale: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = ts_now()


class LensRunRow(SQLModel, table=True):
    __tablename__ = "lens_runs"

    id: uuid.UUID = uuid_pk()
    lens_name: str = Field(sa_column=Column(Text, nullable=False))
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    started_at: datetime = ts_now()
    completed_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))
    prompt_version: str | None = Field(default=None, sa_column=Column(Text))
    summary_text: str | None = Field(default=None, sa_column=Column(Text))
    finding_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    corruption_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    drafting_style_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    scheme_match_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    status: str = Field(default="running", sa_column=Column(Text, nullable=False))
    # Free-text per-run warning surfaced to the operator (e.g. partial-warm gap).
    # Status stays 'succeeded'; warnings is the non-fatal signal channel.
    warnings: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class FeedbackRow(SQLModel, table=True):
    __tablename__ = "feedback"

    id: uuid.UUID = uuid_pk()
    provision_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("provisions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    # Historical FK to the retired assessments table; kept as a plain UUID so
    # pre-retirement feedback rows still carry their audit trail.
    assessment_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    finding_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("findings.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    langfuse_trace_id: str | None = Field(default=None, sa_column=Column(Text))
    kind: str = Field(sa_column=Column(Text, nullable=False))
    value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    comment: str | None = Field(default=None, sa_column=Column(Text))
    user_id: str = Field(sa_column=Column(Text, nullable=False))
    org: str | None = Field(default=None, sa_column=Column(Text))
    jurisdiction: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()


class EventRow(SQLModel, table=True):
    """Append-only application event. Trigger blocks UPDATE/DELETE at depth 0."""

    __tablename__ = "events"

    id: uuid.UUID = uuid_pk()
    actor_id: str = Field(sa_column=Column(Text, nullable=False))
    entity_type: str = Field(sa_column=Column(Text, nullable=False))
    entity_id: uuid.UUID = Field(sa_column=Column(PG_UUID(as_uuid=True), nullable=False))
    event_type: str = Field(sa_column=Column(Text, nullable=False))
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    created_at: datetime = ts_now()


class Acquisition(SQLModel, table=True):
    """One upstream fetch. Unique on (jurisdiction, url, sha256) for re-fetch dedupe."""

    __tablename__ = "acquisitions"
    __table_args__ = (
        UniqueConstraint(
            "jurisdiction_code",
            "source_url",
            "content_sha256",
            name="acquisitions_jurisdiction_url_sha_key",
        ),
    )

    id: uuid.UUID = uuid_pk()
    jurisdiction_code: str = Field(sa_column=Column(Text, nullable=False))
    frbr_work_uri: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    source_url: str = Field(sa_column=Column(Text, nullable=False))
    etag: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    last_modified: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    content_sha256: str = Field(sa_column=Column(Text, nullable=False))
    licence: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    fetched_at: datetime = ts_now()


class TranslationRun(SQLModel, table=True):
    """One translation job, a source Version rendered into a target language. The
    translated act persists as a sibling Version referenced by
    `target_version_id`; this row carries the job lifecycle and notes.
    """

    __tablename__ = "translation_runs"

    id: uuid.UUID = uuid_pk()
    source_version_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            # CASCADE, not RESTRICT: supersede drops the old source version and
            # its translation-attempt audit rows die with it (see 0086).
            ForeignKey("versions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    target_language: str = Field(sa_column=Column(Text, nullable=False))
    target_version_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Lifecycle: 'running' on insert; 'succeeded' / 'failed' on finalisation.
    status: str = Field(default="running", sa_column=Column(Text, nullable=False))
    started_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))
    completed_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))
    progress_pct: int | None = Field(default=None, sa_column=Column(Integer))
    latest_eid: str | None = Field(default=None, sa_column=Column(Text))
    error: str | None = Field(default=None, sa_column=Column(Text))
    # Phase-1 translation notes (terminology + style sheet) and the
    # phase-3 deterministic consistency audit report.
    notes: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    audit: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = ts_now()


class RunRow(SQLModel, table=True):
    """Orchestration record for every durable job. `id` doubles as the DBOS workflow
    id and, where a domain result table exists, as that row's id. Status is the
    single vocabulary: queued|running|succeeded|failed|cancelled.
    """

    __tablename__ = "runs"

    id: uuid.UUID = uuid_pk()
    kind: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default="queued", sa_column=Column(Text, nullable=False))
    actor_id: str | None = Field(default=None, sa_column=Column(Text))
    # Jurisdiction cannot stand in: two workspaces can hold one jurisdiction,
    # and a run is work product. NULL fails closed, visible to its own actor
    # and platform admins only.
    org_id: str | None = Field(default=None, sa_column=Column(Text))
    jurisdiction_code: str | None = Field(default=None, sa_column=Column(Text))
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    tallies: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    parent_run_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE")),
    )
    error: str | None = Field(default=None, sa_column=Column(Text))
    # MinIO object key of the uploaded source (ingest kinds), survives for
    # retry/replay; the tempfile-deleted-in-finally pattern is dead.
    object_key: str | None = Field(default=None, sa_column=Column(Text))
    # Tag plus digest of the image that finished this run, so the code behind
    # an artifact is readable without forensics. Null where none was supplied.
    image_ref: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = ts_now()
    started_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))
    completed_at: datetime | None = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True)))


class RunArtifactRow(SQLModel, table=True):
    """Intermediate stage output, steps pass these row ids, never payloads."""

    __tablename__ = "run_artifacts"

    id: uuid.UUID = uuid_pk()
    run_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
        )
    )
    stage: str = Field(sa_column=Column(Text, nullable=False))
    kind: str = Field(sa_column=Column(Text, nullable=False))
    content_text: str | None = Field(default=None, sa_column=Column(Text))
    content_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = ts_now()


__all__ = [
    "Acquisition",
    "AmendmentEffect",
    "CrossReference",
    "EventRow",
    "ExaminationRow",
    "FeedbackRow",
    "FindingNoteRow",
    "FindingRow",
    "Jurisdiction",
    "Law",
    "LensRunRow",
    "LifecycleEventRow",
    "PageRead",
    "PageReadDisputeRow",
    "Provision",
    "ProvisionEmbedding",
    "RegistryWork",
    "RepairProposalRow",
    "RunArtifactRow",
    "RunRow",
    "SchemeMatchRow",
    "Section",
    "SourceDocument",
    "TranslationRun",
    "Version",
    "VersionSourceText",
    "VersionUnitEmbedding",
    "ts_now",
    "utcnow",
    "uuid_pk",
]
