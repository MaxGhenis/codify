"""Ingestion progress events. Async generators yield these as work progresses."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from codify.akn import Document


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PageExtracted(_EventBase):
    kind: Literal["page_extracted"] = "page_extracted"
    page: int
    method: Literal["text", "ocr"]
    text_len: int
    # Why the text layer was not trusted, for a page that went to OCR. Empty
    # when it was trusted, so a reader can tell a scan from a rejected extract.
    divert_reason: str = ""
    # The vision route's finish reason, so a content-filter refusal is not read
    # as a blank page. Empty where no vision call was made.
    finish_reason: str = ""
    # Why the page's content is missing from the read (`unreadable_reason`);
    # empty for a page read whole or a blank leaf.
    unreadable: str = ""


class MetadataExtracted(_EventBase):
    kind: Literal["metadata_extracted"] = "metadata_extracted"
    metadata: dict[str, Any]


class AnchorsDetected(_EventBase):
    """Pre-flight regex scan of raw text, before the structuring LLM runs."""

    kind: Literal["anchors_detected"] = "anchors_detected"
    summary: dict[str, int]
    total: int


class StructureProgress(_EventBase):
    """Per-window body-fill progress during the structuring stage."""

    kind: Literal["structure_progress"] = "structure_progress"
    fraction: float
    label: str


class Structured(_EventBase):
    kind: Literal["structured"] = "structured"
    bluebell_len: int


class Parsed(_EventBase):
    kind: Literal["parsed"] = "parsed"
    akn_xml_len: int


class Enriched(_EventBase):
    kind: Literal["enriched"] = "enriched"
    pass_name: Literal[
        "cobalt",
        "akn_meta",
        "placeholders",
        "amendments",
        "enacting",
        "hcontainers",
        "references",
        "conclusions",
        "notes",
        "asides",
        "inline_markup",
        "validator",
    ]


class ValidationIssued(_EventBase):
    kind: Literal["validation_issued"] = "validation_issued"
    issue: dict[str, Any]


class Complete(_EventBase):
    kind: Literal["complete"] = "complete"
    document: Document
    akn_xml: str


class Failed(_EventBase):
    kind: Literal["failed"] = "failed"
    stage: str
    error: str


IngestionEvent: TypeAlias = Annotated[
    PageExtracted
    | MetadataExtracted
    | AnchorsDetected
    | StructureProgress
    | Structured
    | Parsed
    | Enriched
    | ValidationIssued
    | Complete
    | Failed,
    Field(discriminator="kind"),
]


__all__ = [
    "AnchorsDetected",
    "Complete",
    "Enriched",
    "Failed",
    "IngestionEvent",
    "MetadataExtracted",
    "PageExtracted",
    "Parsed",
    "StructureProgress",
    "Structured",
    "ValidationIssued",
]
