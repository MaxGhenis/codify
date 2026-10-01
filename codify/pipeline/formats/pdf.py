"""PDF ingestion, orchestrates Extract → Metadata → Structure → Parse → Enrich → Type."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from langfuse import get_client

from codify.akn.io import parse_akn
from codify.core.tracing import (
    attach_trace_attribution,
    get_current_actor,
    get_current_release,
    get_current_session,
    langfuse_trace_context,
)
from codify.jurisdictions import CONFIG_FAULTS, try_load_config
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.cover_reconciliation import extract_cover_article_numbers
from codify.pipeline.enrich.metadata import calendar_hint, extract_metadata
from codify.pipeline.enrich.ocr import (
    PageResult,
    combine_page_texts_with_spans,
    extract_text_from_pdf,
    furniture_inline_patterns,
    unreadable_pages,
    unreadable_reason,
)
from codify.pipeline.enrich.region_text import combine_text_for_structure
from codify.pipeline.enrich.regions import (
    Region,
    classify_layouts,
    vocabulary_for_jurisdiction,
)
from codify.pipeline.enrich.structure import ScanTrace, text_to_bluebell_scaffolded
from codify.pipeline.enrich.validator import validate_akn
from codify.pipeline.events import (
    AnchorsDetected,
    Complete,
    Enriched,
    Failed,
    IngestionEvent,
    MetadataExtracted,
    PageExtracted,
    Parsed,
    Structured,
    StructureProgress,
    ValidationIssued,
)
from codify.pipeline.stages import resolve_descriptors, run_enrich_passes
from codify.quality.invariants import halt_finding

if TYPE_CHECKING:
    from codify.core.llm import LLMClient

logger = structlog.get_logger()


class _ScanCollector:
    """Harvests the scan-side evidence the validator grades on (orphaned drops
    and the container-coverage probe) while still forwarding each trace to a
    caller-supplied on_scan. Mirrors ingest_step_structure's on_scan so the
    format/CLI lane grades by the same rubric as the durable ingest lane."""

    def __init__(self, forward: Callable[[ScanTrace], None] | None) -> None:
        self._forward = forward
        self.orphaned_drops: list[dict[str, Any]] = []
        self.container: dict[str, Any] | None = None
        self.halts: list[dict[str, Any]] = []
        self.body_fill: dict[str, Any] | None = None

    def __call__(self, trace: ScanTrace) -> None:
        self.orphaned_drops.extend(
            dict(s.detail, start=s.start)
            for s in trace.ambiguity
            if s.emitted_by == "declare_orphaned_drops"
        )
        self.container = trace.container
        self.halts.extend(asdict(h) for h in trace.halts)
        if trace.body_fill is not None:
            self.body_fill = asdict(trace.body_fill)
        if self._forward is not None:
            self._forward(trace)


async def ingest(
    source: Path | str,
    jurisdiction_code: str,
    *,
    llm: LLMClient,
    model: str | None = None,
    ocr_model: str | None = None,
    on_pages: Callable[[list[PageResult]], None] | None = None,
    on_scan: Callable[[ScanTrace], None] | None = None,
) -> AsyncIterator[IngestionEvent]:
    """PDF → AKN. Caller supplies an LLMClient; defensive on every stage.

    ``model`` overrides the LLMClient's default for the body-fill call;
    ``ocr_model`` overrides it for the vision OCR call. None routes to the
    client's configured default.

    ``on_pages`` and ``on_scan`` hand a caller the extracted pages and the scan
    result. The event stream carries counts; diagnosing an ingest needs the
    inputs they were computed from.
    """
    pdf_path = Path(source)
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="ingest_document.pdf",
        trace_context=langfuse_trace_context(),
        input={"jurisdiction": jurisdiction_code, "source": str(pdf_path)},
        metadata={"jurisdiction": jurisdiction_code, "pipeline": "ingest"},
    ) as root:
        attach_trace_attribution(
            langfuse,
            user_id=get_current_actor(),
            session_id=get_current_session(),
            release=get_current_release(),
            tags=[f"jurisdiction:{jurisdiction_code}", "pipeline:ingest"],
        )
        # Stage 1, Extract
        with langfuse.start_as_current_observation(as_type="span", name="extract") as span:
            try:
                pdf_bytes = pdf_path.read_bytes()
                pages = await extract_text_from_pdf(
                    pdf_bytes,
                    client=llm,
                    country=jurisdiction_code,
                    ocr_model=ocr_model,
                )
            except Exception as exc:  # noqa: BLE001
                span.update(level="ERROR", status_message=str(exc))
                yield Failed(stage="extract", error=f"{type(exc).__name__}: {exc}")
                return
        for page in pages:
            yield PageExtracted(
                page=page.page_number,
                method="ocr" if page.method != "text_extraction" else "text",
                text_len=len(page.text),
                divert_reason=page.divert_reason,
                finish_reason=page.finish_reason,
                unreadable=unreadable_reason(page),
            )
        if on_pages:
            on_pages(pages)

        async for event in _ingest_pages(
            pages,
            jurisdiction_code,
            llm=llm,
            source_bytes=pdf_bytes,
            source_name=pdf_path.name,
            fallback_stem=pdf_path.stem,
            model=model,
            on_scan=on_scan,
            langfuse=langfuse,
            root=root,
        ):
            yield event


async def ingest_text(
    text: str,
    jurisdiction_code: str,
    *,
    llm: LLMClient,
    name: str = "source",
    model: str | None = None,
    on_scan: Callable[[ScanTrace], None] | None = None,
) -> AsyncIterator[IngestionEvent]:
    """Already-extracted text → AKN, on the same stages a PDF runs after OCR.

    Much of a legacy corpus arrives as text, and sharing the stages is what
    makes a text-sourced document comparable to a scanned one.
    """
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="ingest_document.text",
        trace_context=langfuse_trace_context(),
        input={"jurisdiction": jurisdiction_code, "source": name},
        metadata={"jurisdiction": jurisdiction_code, "pipeline": "ingest"},
    ) as root:
        page = PageResult(
            page_number=1,
            text=text,
            method="text_extraction",
            furniture=furniture_inline_patterns(jurisdiction_code),
        )
        async for event in _ingest_pages(
            [page],
            jurisdiction_code,
            llm=llm,
            source_bytes=text.encode("utf-8"),
            source_name=name,
            fallback_stem=name,
            model=model,
            on_scan=on_scan,
            langfuse=langfuse,
            root=root,
        ):
            yield event


def _regions_from_pages(
    pages: list[PageResult], jurisdiction_code: str, year: str
) -> dict[int, list[Region]]:
    """Classify what the layout engine saw, for the passes that consume it."""
    return classify_layouts(
        {page.page_number: page.layout for page in pages if page.layout is not None},
        vocab=vocabulary_for_jurisdiction(jurisdiction_code, year),
    )


async def _ingest_pages(
    pages: list[PageResult],
    jurisdiction_code: str,
    *,
    llm: LLMClient,
    source_bytes: bytes,
    source_name: str,
    fallback_stem: str,
    model: str | None,
    on_scan: Callable[[ScanTrace], None] | None,
    langfuse: Any,
    root: Any,
) -> AsyncIterator[IngestionEvent]:
    """Metadata through typed document: every stage after text extraction."""
    raw_text, page_spans = combine_page_texts_with_spans(pages)

    # Stage 2, Metadata
    with langfuse.start_as_current_observation(as_type="span", name="metadata") as span:
        try:
            metadata = await extract_metadata(
                raw_text,
                llm,
                filename=source_name,
                calendar_hint_text=calendar_hint(try_load_config(jurisdiction_code)),
            )
        except Exception as exc:  # noqa: BLE001
            span.update(level="ERROR", status_message=str(exc))
            yield Failed(stage="metadata", error=f"{type(exc).__name__}: {exc}")
            return
    yield MetadataExtracted(metadata=metadata)

    desc = resolve_descriptors(
        metadata,
        jurisdiction_code=jurisdiction_code,
        source_bytes=source_bytes,
        fallback_stem=fallback_stem,
        classification_text=raw_text,
    )

    # Regions gate the structurer: classify the layout blocks now (era vocab needs
    # desc.year) so header/footer never become body and footnotes relocate to a
    # liftable position. `raw_text` stays the full superset for metadata (the
    # gazette number lives in a footer) and the Stage-6 validator; only the
    # structurer sees the filtered text. Reused at Stage 5. Empty on the
    # born-digital / vision route, where structure_text falls back to raw_text.
    try:
        regions = _regions_from_pages(pages, jurisdiction_code, desc.year)
    except CONFIG_FAULTS as exc:
        yield Failed(stage="regions", error=f"{type(exc).__name__}: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("regions_unavailable", error=str(exc))
        regions = {}
    structure_text = combine_text_for_structure(raw_text, regions, page_spans)

    # Stage 3, Structure. Bridge the structurer's callbacks through a queue
    # so the anchor outline streams immediately and each body-fill window
    # reports progress, instead of one event after the whole stage.
    events_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _on_anchors(summary: dict[str, int], total: int) -> None:
        events_q.put_nowait(("anchors", (summary, total)))

    def _on_progress(fraction: float, label: str) -> None:
        events_q.put_nowait(("progress", (fraction, label)))

    # Collect the scan-side evidence the validator grades on (orphaned drops,
    # container coverage), mirroring ingest_step_structure, so this lane grades
    # by the same rubric as production instead of a weaker subset. The caller's
    # on_scan is still invoked (the CLI appends every trace).
    scan = _ScanCollector(on_scan)

    anchor_summary_seen: dict[str, int] | None = None
    with langfuse.start_as_current_observation(as_type="span", name="structure") as span:
        task = asyncio.create_task(
            text_to_bluebell_scaffolded(
                structure_text,
                client=llm,
                country=jurisdiction_code,
                doctype=desc.doctype,
                on_anchors=_on_anchors,
                on_progress=_on_progress,
                on_scan=scan,
                model=model,
                # The tool for diagnosing a halt must not refuse what the durable
                # lane lands, or it hands back an exception instead of a bundle.
                halt_policy="land",
            )
        )
        while not task.done() or not events_q.empty():
            try:
                kind, payload = await asyncio.wait_for(events_q.get(), timeout=0.1)
            except TimeoutError:
                continue
            if kind == "anchors":
                summary, total = payload
                anchor_summary_seen = summary
                yield AnchorsDetected(summary=summary, total=total)
            else:
                fraction, label = payload
                yield StructureProgress(fraction=fraction, label=label)
        try:
            bluebell_text = await task
        except Exception as exc:  # noqa: BLE001
            span.update(level="ERROR", status_message=str(exc))
            yield Failed(stage="structure", error=f"{type(exc).__name__}: {exc}")
            return
    yield Structured(bluebell_len=len(bluebell_text))

    # Stage 4, Parse Bluebell → AKN XML
    with langfuse.start_as_current_observation(as_type="span", name="parse") as span:
        try:
            akn_xml = parse_to_akn(
                bluebell_text,
                country=jurisdiction_code,
                doctype=desc.doctype,
                date=desc.raw_date or desc.year,
                number=desc.number,
                language=desc.language,
            )
        except Exception as exc:  # noqa: BLE001
            span.update(level="ERROR", status_message=str(exc))
            yield Failed(stage="parse", error=f"{type(exc).__name__}: {exc}")
            return
    yield Parsed(akn_xml_len=len(akn_xml))

    # Stage 5, Enrich (shared pass sequence in codify.pipeline.stages).
    # `regions` was computed before Stage 3 and is reused here.
    passes_done: list[str] = []
    with langfuse.start_as_current_observation(as_type="span", name="enrich"):
        akn_xml = await run_enrich_passes(
            akn_xml,
            llm=llm,
            jurisdiction_code=jurisdiction_code,
            desc=desc,
            on_pass=passes_done.append,
            regions=regions,
        )
    for pass_name in passes_done:
        yield Enriched(pass_name=pass_name)  # type: ignore[arg-type]

    # Stage 6, Validate (issues are advisory; never fatal here).
    with langfuse.start_as_current_observation(as_type="span", name="validate"):
        # Emitted as a finding, or the bundle reads clean for a document the
        # durable lane grades blocking.
        for halt in scan.halts:
            yield ValidationIssued(issue=halt_finding(halt))
        try:
            for issue in validate_akn(
                akn_xml,
                expected_anchor_summary=anchor_summary_seen,
                expected_cover_article_numbers=extract_cover_article_numbers(raw_text),
                source_text=raw_text,
                orphaned_drops=scan.orphaned_drops or None,
                container_coverage=scan.container or None,
                provenance="extracted",
                unreadable_pages=unreadable_pages(pages) or None,
                body_fill=scan.body_fill,
            ):
                yield ValidationIssued(issue=issue)
            yield Enriched(pass_name="validator")  # noqa: S106
        except Exception as exc:  # noqa: BLE001
            logger.warning("validator_failed", error=str(exc))

    # Stage 7, Typed Document via codify.akn.io.parse_akn
    with langfuse.start_as_current_observation(as_type="span", name="type") as span:
        try:
            document = parse_akn(akn_xml)
        except Exception as exc:  # noqa: BLE001
            span.update(level="ERROR", status_message=str(exc))
            yield Failed(stage="type", error=f"{type(exc).__name__}: {exc}")
            return

    root.update(output={"frbr": document.frbr_work_uri, "akn_xml_len": len(akn_xml)})
    yield Complete(document=document, akn_xml=akn_xml)


__all__ = ["ingest", "ingest_text"]
