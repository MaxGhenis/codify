"""Read what the pipeline did, one document at a time or across a whole corpus.

    codify ingest-one gazette.pdf --jurisdiction xa --out bundle/

runs one document through the pipeline and writes a bundle: page images, every
anchor with its producing pass, the coverage measurement with both number sets,
the pre-body-fill scaffold, the final AKN and the validator findings. Every
artifact is captured from the run, not recomputed. It needs a chat model behind
an OpenAI-compatible endpoint (`LITELLM_BASE_URL` and `LITELLM_API_KEY`, read
from the environment or the nearest `.env`) and writes no database.

    codify scan-corpus ~/corpus --jurisdiction xa

measures the anchor scan over raw sources before anything is ingested, which is
the only way to count a marker the scanner never claimed. It needs no model and
no database.

    codify segment bundle/

re-reads a bundle's stored text and page spans and rewrites its
`segmentation.json` under the jurisdiction's current config: which acts the
source holds, and why any region was held. It needs no model and no database.

The index-* subcommands build the acquisition indexes a bulk import reads.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog

from codify.core.llm import create_llm_client
from codify.jurisdictions import JURISDICTIONS_DIR, load_config, resolve_config
from codify.pipeline.enrich.ocr import (
    PageResult,
    PageSpan,
    combine_page_texts_with_spans,
    unreadable_pages,
)
from codify.pipeline.enrich.structure import ScanTrace
from codify.pipeline.events import (
    Complete,
    Enriched,
    Failed,
    MetadataExtracted,
    ValidationIssued,
)
from codify.pipeline.formats.pdf import ingest, ingest_text
from codify.pipeline.segment import Segmentation, segment
from codify.quality.structural_quality_grade import structural_quality_grade

logger = structlog.get_logger()

# Readable for a scanned Arabic gazette page. OCR renders at its own DPI.
_PAGE_DPI = 200


def _render_pages(pdf_bytes: bytes, out: Path) -> list[str]:
    from pdf2image import convert_from_bytes

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for number, image in enumerate(convert_from_bytes(pdf_bytes, dpi=_PAGE_DPI), start=1):
        name = f"page-{number:03d}.png"
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        (out / name).write_bytes(buf.getvalue())
        written.append(name)
    return written


def _foundry_ocr_url() -> str | None:
    """The direct Azure Foundry Mistral OCR endpoint, derived as the API derives it.

    Mistral OCR bypasses the proxy, and the env var carries a LiteLLM path
    suffix, so the root is taken from the endpoint and the Mistral path appended.
    """
    from urllib.parse import urlparse

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if not endpoint or not os.environ.get("AZURE_OPENAI_API_KEY"):
        return None
    u = urlparse(endpoint)
    return f"{u.scheme}://{u.netloc}/providers/mistral/azure/ocr"


def _anchor_row(anchor: Any) -> dict[str, Any]:
    row = asdict(anchor)
    # char_offset locates the full text in source.txt; a preview identifies it.
    row["matched_text"] = row["matched_text"][:120]
    return row


def _coverage_json(
    trace: ScanTrace | None, *, pages: int | None = None, unreadable: dict[int, str] | None = None
) -> dict[str, Any] | None:
    """Coverage as the bundle reports it. `ratio` is the lowest of the anchor,
    body-fill and page ratios measured, so no one of them reads clean alone."""
    cov = trace.coverage if trace else None
    fill = trace.body_fill if trace else None
    container = trace.container if trace else None
    if cov is None and container is None and fill is None and not pages:
        return None
    out: dict[str, Any] = {}
    ratios: list[float] = []
    if cov is not None:
        anchor_ratio = None if cov.ratio is None else round(cov.ratio, 4)
        if anchor_ratio is not None:
            ratios.append(anchor_ratio)
        out.update(
            {
                "kind": cov.kind,
                "ratio": anchor_ratio,
                "anchor_ratio": anchor_ratio,
                "captured": sorted(cov.captured),
                "expected": sorted(cov.expected),
                "missing": sorted(cov.missing),
                # Without this a document refused on the masked count writes an
                # artifact that reads as clean.
                "masked": cov.masked,
                "unclosed": cov.unclosed,
            }
        )
    if container is not None:
        # Heading-vs-container probe: how many grouping headings survived as
        # containers. `7 headings -> 0 containers` is the flattening signal.
        out["container"] = container
    if fill is not None:
        fill_ratio = None if fill.ratio is None else round(fill.ratio, 4)
        if fill_ratio is not None:
            ratios.append(fill_ratio)
        out["body_fill"] = {**asdict(fill), "model_filled": fill.model_filled, "ratio": fill_ratio}
    if pages:
        lost = sorted(unreadable or {})
        page_ratio = round((pages - len(lost)) / pages, 4)
        ratios.append(page_ratio)
        out["pages"] = {"total": pages, "unreadable": lost, "ratio": page_ratio}
    out["ratio"] = min(ratios) if ratios else None
    return out


_BUNDLE_FILES = (
    "manifest.json",
    "source.txt",
    "anchors.jsonl",
    "coverage.json",
    "ambiguity.jsonl",
    "scaffold.bluebell",
    "final.akn.xml",
    "validator.json",
    "events.jsonl",
    "page_spans.json",
    "segmentation.json",
)


def _clear_bundle(out: Path) -> None:
    """Remove this command's own artifacts from a reused directory.

    A run that fails early writes fewer files, and the remainder would read as
    belonging to it. Only files this command writes are touched.
    """
    for name in _BUNDLE_FILES:
        (out / name).unlink(missing_ok=True)
    pages = out / "pages"
    if pages.is_dir():
        for page in pages.glob("page-*.png"):
            page.unlink()


async def _run(args: argparse.Namespace) -> int:
    source = Path(args.source)

    # Before any work, and before the output directory is touched. Without a
    # config the pipeline does not stop: it falls back to defaults and produces
    # a plausible bundle from a jurisdiction it knows nothing about, which reads
    # as a result rather than as the absence of one. On the open core, where
    # five of 257 configs ship, that is the default experience for the rest.
    config = resolve_config(args.jurisdiction)
    if not config.found:
        print(
            f"no config for jurisdiction {config.code!r}: looked in "
            f"{JURISDICTIONS_DIR / config.code / 'config.json'}",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _clear_bundle(out)
    # Much of a legacy corpus arrives as text, not scans. Both lanes share the
    # stages after extraction.
    is_text = source.suffix.lower() in {".txt", ".md"}
    raw_bytes = source.read_bytes()

    llm = create_llm_client(
        base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
        # Matches the gateway's own default (docker-compose.yaml): `docker
        # compose` reads .env without exporting it to a later `uv run`.
        api_key=(
            os.environ.get("LITELLM_API_KEY")
            or os.environ.get("LITELLM_MASTER_KEY")
            or "sk-codify-dev"
        ),
        model=args.model,
        azure_foundry_ocr_url=_foundry_ocr_url(),
        azure_foundry_ocr_key=os.environ.get("AZURE_OPENAI_API_KEY") or None,
        azure_foundry_ocr_rpm=int(os.environ.get("AZURE_FOUNDRY_OCR_RPM") or 40),
        # Retried on any content-filter refusal (pages, metadata, body-fill); unset,
        # a refused page stays unread.
        fallback_model=args.fallback_model or None,
        # No run context here, so proxy attribution would carry nothing; direct
        # mode also lets the URL name any OpenAI-compatible provider.
        telemetry_mode="direct",
    )

    pages: list[PageResult] = []
    traces: list[ScanTrace] = []
    findings: list[dict[str, Any]] = []
    events: list[str] = []
    akn_xml = ""
    metadata: dict[str, Any] = {}
    failure: dict[str, str] | None = None
    # Unset until the validator reports finishing; no findings is not clean otherwise.
    validated = False
    started = time.monotonic()

    stream = (
        ingest_text(
            raw_bytes.decode("utf-8"),
            config.code,
            llm=llm,
            name=source.stem,
            model=args.model,
            on_scan=traces.append,
        )
        if is_text
        else ingest(
            source,
            config.code,
            llm=llm,
            model=args.model,
            ocr_model=args.ocr_model or None,
            on_pages=pages.extend,
            on_scan=traces.append,
        )
    )
    try:
        async for event in stream:
            events.append(event.model_dump_json())
            if isinstance(event, ValidationIssued):
                findings.append(event.issue)
            elif isinstance(event, MetadataExtracted):
                metadata = event.metadata
            elif isinstance(event, Enriched) and event.pass_name == "validator":  # noqa: S105
                validated = True
            elif isinstance(event, Complete):
                akn_xml = event.akn_xml
            elif isinstance(event, Failed):
                failure = {"stage": event.stage, "error": event.error}
            if not args.quiet:
                print(f"  {type(event).__name__}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001, an unhandled stage is what to record
        failure = {"stage": "raised", "error": f"{type(exc).__name__}: {exc}"}
        logger.warning("ingest_one_raised", error=str(exc)[:300])

    elapsed = time.monotonic() - started

    # One trace per run. No scaffold means the gate raised or no anchors were
    # found, which is itself the answer.
    trace = traces[0] if traces else None
    source_text, spans = (
        (raw_bytes.decode("utf-8"), []) if is_text else combine_page_texts_with_spans(pages)
    )
    # Untranslated newlines: span offsets index these exact characters.
    (out / "source.txt").write_text(source_text, encoding="utf-8", newline="")
    page_spans = _page_spans_json(spans, pages)
    (out / "page_spans.json").write_text(
        json.dumps(page_spans, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    segmentation_failed = None
    try:
        _write_segmentation(out, source_text, page_spans, config.code)
    except Exception as exc:  # noqa: BLE001, a segmenter fault must not void the spend
        segmentation_failed = f"{type(exc).__name__}: {exc}"
        logger.warning("segmentation_failed", error=str(exc)[:300])
    (out / "events.jsonl").write_text("".join(f"{line}\n" for line in events), encoding="utf-8")
    (out / "validator.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False))

    anchors = trace.anchors if trace else ()
    (out / "anchors.jsonl").write_text(
        "".join(f"{json.dumps(_anchor_row(a), ensure_ascii=False)}\n" for a in anchors),
        encoding="utf-8",
    )
    unreadable = {} if is_text else unreadable_pages(pages)
    coverage = _coverage_json(trace, pages=None if is_text else len(pages), unreadable=unreadable)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False))
    ambiguity = trace.ambiguity if trace else ()
    (out / "ambiguity.jsonl").write_text(
        "".join(f"{json.dumps(asdict(s), ensure_ascii=False)}\n" for s in ambiguity),
        encoding="utf-8",
    )
    if trace and trace.scaffold is not None:
        (out / "scaffold.bluebell").write_text(trace.scaffold, encoding="utf-8")
    if akn_xml:
        (out / "final.akn.xml").write_text(akn_xml, encoding="utf-8")

    page_files: list[str] = []
    if not is_text:
        try:
            page_files = _render_pages(raw_bytes, out / "pages")
        except Exception as exc:  # noqa: BLE001, poppler missing must not void the bundle
            logger.warning("page_render_failed", error=str(exc)[:200])

    # A refusal is known, not unmeasured, so it outranks a validator that did not finish.
    halted = any(f.get("check") == "structure_halted" for f in findings)
    manifest = {
        "source": str(source),
        "source_bytes": len(raw_bytes),
        "lane": "text" if is_text else "pdf",
        # The resolved code, not the argument: `--jurisdiction GB` resolves the
        # `gb` config here, and passing the raw code on would have the pipeline
        # look up `GB`, find nothing on a case-sensitive filesystem, and fall
        # back to defaults, which is the failure this whole change prevents.
        "jurisdiction": config.code,
        # What the model read off the document, so `codify load` needs no flags.
        "metadata": {k: metadata.get(k) for k in ("title", "doctype", "year", "number", "date")},
        # The config this run actually resolved, by path and digest. The code
        # alone cannot distinguish two runs against a config that changed
        # between them, and cannot show that one was found at all.
        "jurisdiction_config": config.model_dump(),
        "model": args.model,
        "ocr_model": None if is_text else (args.ocr_model or None),
        "fallback_model": args.fallback_model or None,
        "elapsed_seconds": round(elapsed, 1),
        "pages": None if is_text else len(page_files),
        # Which pages did not have their text layer trusted, and why. The reason
        # is what distinguishes a scan from a text layer we rejected, and a
        # bundle taken later cannot recover it from the process log.
        "pages_diverted_to_ocr": (
            None if is_text else {p.page_number: p.divert_reason for p in pages if p.divert_reason}
        ),
        # Pages nothing read twice. The whole divert mechanism is inert for these,
        # so a bundle with too little text cannot be blamed on the structurer.
        "pages_read_once": None if is_text else sum(1 for p in pages if not p.rival_available),
        # Pages that came back with no text, and why. Their content is not in the
        # AKN; the validator grades the document blocking on them.
        "pages_unreadable": None if is_text else unreadable,
        "anchors": len(anchors),
        "anchors_by_pass": _by_pass(anchors),
        "scaffold_built": bool(trace and trace.scaffold is not None),
        "fallback": trace.fallback if trace else None,
        "coverage": coverage,
        "ambiguity": len(ambiguity),
        "ambiguity_by_kind": _by_kind(ambiguity),
        "ambiguity_blocking": sum(1 for s in ambiguity if s.blocking),
        "validator_findings": len(findings),
        # The grade a store would record for this bundle, so a run that finished
        # with blocking findings cannot read as a clean one.
        "grade": asdict(structural_quality_grade(findings, degraded=not (validated or halted))),
        "akn_bytes": len(akn_xml),
        # Set when the segmenter raised, so a missing segmentation.json is explained.
        "segmentation_failed": segmentation_failed,
        "failed": failure,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 1 if failure else 0


def _page_spans_json(spans: list[PageSpan], pages: list[PageResult]) -> dict[str, Any]:
    """Each page's place in source.txt and the furniture lifted off it, in the
    shape the platform's page_texts artifact stores."""
    return {
        "pages": [span.model_dump() for span in spans],
        "furniture": [
            {"page": page.page_number, "header": page.header, "footer": page.footer}
            for page in pages
            if page.header or page.footer
        ],
    }


def _furniture(rows: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(row["page"]): "\n".join(p for p in (row.get("header"), row.get("footer")) if p)
        for row in rows
    }


def _segmentation_json(result: Segmentation) -> dict[str, Any]:
    """Offsets index source.txt; texts are left out, since source.txt holds them."""

    def without_text(item: Any) -> dict[str, Any]:
        return {k: v for k, v in asdict(item).items() if k != "text"}

    return {
        "outcome": result.outcome,
        "front_matter": list(result.front_matter) if result.front_matter else None,
        "segments": [without_text(s) for s in result.segments],
        "held": [without_text(h) for h in result.held],
        "reconciliation": [{**asdict(r), "reads": r.describe()} for r in result.reconciliation],
    }


def _write_segmentation(
    out: Path, source_text: str, page_spans: dict[str, Any], jurisdiction: str
) -> dict[str, Any]:
    result = segment(
        source_text,
        [PageSpan.model_validate(s) for s in page_spans.get("pages") or []],
        config=load_config(jurisdiction),
        furniture=_furniture(page_spans.get("furniture") or []),
    )
    payload = {
        # Which config decided, so a re-run under an edited one is told apart.
        "jurisdiction_config": resolve_config(jurisdiction).model_dump(),
        **_segmentation_json(result),
    }
    (out / "segmentation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _run_segment(args: argparse.Namespace) -> int:
    """Re-run the segmenter over a bundle's stored text, with no model."""
    bundle = Path(args.bundle)
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        source_text = (bundle / "source.txt").read_bytes().decode("utf-8")
        page_spans = json.loads((bundle / "page_spans.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"not a readable bundle with page spans: {bundle}: {exc}", file=sys.stderr)
        return 2
    code = args.jurisdiction or manifest.get("jurisdiction") or ""
    if not resolve_config(code).found:
        print(f"no config for jurisdiction {code!r}", file=sys.stderr)
        return 2
    payload = _write_segmentation(bundle, source_text, page_spans, resolve_config(code).code)
    print(f"{payload['outcome']}: {len(payload['segments'])} acts, {len(payload['held'])} held")
    for row in payload["reconciliation"]:
        print(f"  {row['reads']}")
    return 0


def _by_kind(spans: tuple[Any, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.kind] = counts.get(span.kind, 0) + 1
    return dict(sorted(counts.items()))


def _by_pass(anchors: tuple[Any, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in anchors:
        counts[a.source_pass or "unattributed"] = counts.get(a.source_pass or "unattributed", 0) + 1
    return counts


def _run_index_datadump(args: argparse.Namespace) -> int:
    """Map every CELEX in a bulk FORMEX archive to the member holding its body."""
    from codify.acquisition.adapters.eu.datadump import build_index
    from codify.acquisition.index import index_path

    archive = Path(args.archive)
    if not archive.exists():
        print(f"archive not found: {archive}", file=sys.stderr)
        return 2

    def progress(scanned: int, total: int, indexed: int) -> None:
        print(f"  scanned {scanned}/{total}, indexed {indexed}", file=sys.stderr)

    payload = build_index(
        archive,
        language=args.language or None,
        max_files=args.max_files,
        on_progress=None if args.quiet else progress,
    )
    out = Path(args.out or index_path("eu"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    stats = payload["stats"]
    print(f"{out}: {stats}", file=sys.stderr)
    return 0


def _run_index_ee_archive(args: argparse.Namespace) -> int:
    """Build index of active statutes from Estonian Riigi Teataja bulk XML zip."""
    from codify.acquisition.adapters.ee.datadump import build_ee_index
    from codify.acquisition.index import index_path

    archive = Path(args.archive)
    if not archive.exists():
        print(f"archive not found: {archive}", file=sys.stderr)
        return 2

    def progress(scanned: int, total: int, indexed: int) -> None:
        print(f"  scanned {scanned}/{total}, indexed {indexed}", file=sys.stderr)

    payload = build_ee_index(
        archive,
        principal_only=args.principal_only,
        max_files=args.max_files,
        on_progress=None if args.quiet else progress,
    )
    out = Path(args.out or index_path("ee"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = payload["stats"]
    print(f"Wrote index to {out}: {stats}", file=sys.stderr)
    return 0


def _run_index_legislation_gov_uk(args: argparse.Namespace) -> int:
    """Walk the publisher's feeds for every type token the jurisdiction declares."""
    from codify.acquisition import selected_adapter
    from codify.acquisition.adapters.legislation_gov_uk import build_index, write_index
    from codify.acquisition.index import index_path
    from codify.acquisition.politeness import make_polite_client
    from codify.frbr import token_family
    from codify.jurisdictions import load_config

    # One code shape, so `GB` and `gb` load one config and name one index directory.
    cfg = load_config(args.jurisdiction.strip().lower())
    tokens = (
        [t.strip() for t in args.types.split(",") if t.strip()]
        if args.types
        else [n for n in cfg.document_classes if token_family(n)]
    )
    adapter = selected_adapter(cfg.code, kind="akn_native")

    def progress(token: str, year: int, got: int, expected: int) -> None:
        print(f"  {token}/{year}: {got}/{expected}", file=sys.stderr)

    out = Path(args.out or index_path(cfg.code))
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists():
        print(f"{out}: resuming the build already there", file=sys.stderr)

    async def run() -> dict[str, Any]:
        client = make_polite_client(adapter)
        try:
            # Written after every token-year, so a killed build resumes here.
            return await build_index(
                client,
                tokens,
                years=(args.year_from, args.year_to),
                on_progress=None if args.quiet else progress,
                checkpoint=out,
                resume=args.resume,
            )
        finally:
            await client.aclose()

    payload = asyncio.run(run())
    write_index(out, payload)
    stats = payload["stats"]
    print(
        f"{out}: {stats['items']} items over {len(tokens)} types, "
        f"{len(stats['short_years'])} short years",
        file=sys.stderr,
    )
    return 0


def _run_scan_corpus(args: argparse.Namespace) -> int:
    """Measure the anchor scan over a directory of raw sources."""
    from codify.quality.corpus_scan import aggregate, scan_corpus

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    sweep = scan_corpus(root, jurisdiction=args.jurisdiction, include_derived=args.include_derived)
    scans = sweep.scans
    if not scans:
        # Print nothing: the documented workflow redirects stdout over a pinned
        # baseline, and a redirect keeps the file whatever the exit code says.
        print(
            f"no scannable sources under {root} (unreadable {sweep.unreadable}, "
            f"no text layer {sweep.no_text_layer}, derived exports {sweep.excluded_derived})",
            file=sys.stderr,
        )
        return 1
    if args.per_document:
        rows = "\n".join(json.dumps(asdict(s), ensure_ascii=False) for s in scans)
        Path(args.per_document).write_text(rows + "\n", encoding="utf-8")
    measured = aggregate(sweep)
    summary = {
        "corpus": {
            "jurisdiction": args.jurisdiction,
            "root": str(root),
            "include_derived": args.include_derived,
        },
        **measured,
    }
    # Set when containment failed. The measurement still prints: it is the thing
    # the caller asked for, and withholding it turns "you may not conclude a pass
    # is dead" into "the scan produced nothing".
    containment_failed = False
    if args.containment:
        from codify.quality.regression_corpus import check_containment, doc_id_of

        # Characters observed, not filenames: a stub at the right path is not
        # the document. Passes measured, so one the manifest omits is named
        # rather than certified by silence.
        read: dict[str, int] = {}
        for scan in scans:
            if doc_id := doc_id_of(scan.path):
                read[doc_id] = read.get(doc_id, 0) + scan.chars
        # Only the passes this sweep measured at zero: those are the ones a
        # deletion would be argued from, and the ones the manifest must cover.
        zero_firing = [
            name
            for name, row in (measured.get("passes") or {}).items()
            if isinstance(row, list) and row and row[0] == 0
        ]
        report = check_containment(read, zero_firing_passes=zero_firing)
        summary["containment"] = {
            "evidenced": {k: list(v) for k, v in report.evidenced.items()},
            "unevidenced": report.unevidenced,
            "unevidenceable": report.unevidenceable,
            "undocumented": report.undocumented,
            "deleted": report.deleted,
            "unanswered": list(report.unanswered),
            "incoherent": list(report.incoherent),
        }
        for defect in report.incoherent:
            print(f"MANIFEST {defect}", file=sys.stderr)
        for name in report.unanswered:
            print(
                f"UNANSWERED {name}: this sweep measured it and the manifest says "
                "nothing about it, so its firing count is not evidence either way.",
                file=sys.stderr,
            )
        for name, why in sorted(report.unevidenced.items()):
            print(
                f"UNEVIDENCED {name}: motivating document not read ({why}). "
                "A zero firing count for this pass means nothing here.",
                file=sys.stderr,
            )
        for name, why in sorted(report.unevidenceable.items()):
            print(
                f"UNREACHABLE {name}: no sweep of this corpus can judge it ({why}).",
                file=sys.stderr,
            )
        for name, why in sorted(report.undocumented.items()):
            print(f"NO-DOCUMENT {name}: {why}", file=sys.stderr)
        containment_failed = not report.ok
    if args.check:
        # Compare the measurement, not the `corpus` block: its root is an
        # absolute path and differs on every machine that runs this.
        regenerate = (
            f"Regenerate with: codify scan-corpus {root} "
            f"--jurisdiction {args.jurisdiction} > {args.check}"
        )
        try:
            baseline = json.loads(Path(args.check).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"UNREADABLE baseline {args.check}: {exc}", file=sys.stderr)
            print(regenerate, file=sys.stderr)
            return 1
        if not isinstance(baseline, dict):
            print(f"UNREADABLE baseline {args.check}: not an object", file=sys.stderr)
            print(regenerate, file=sys.stderr)
            return 1
        # `containment` is a property of the sweep, not of the scanner, so it
        # is compared no more than the machine-local root is.
        ignored = ("corpus", "containment")
        expected = {k: v for k, v in baseline.items() if k not in ignored}
        measured = {k: v for k, v in measured.items() if k not in ignored}
        if expected != measured:
            drifted = sorted(
                k for k in set(expected) | set(measured) if expected.get(k) != measured.get(k)
            )
            print(f"DRIFT in {args.check}: {', '.join(drifted)}", file=sys.stderr)
            for key in drifted:
                print(
                    f"  {key}: expected {expected.get(key)!r}, measured {measured.get(key)!r}",
                    file=sys.stderr,
                )
            print(regenerate, file=sys.stderr)
            return 1
        print(f"{args.check} matches ({measured['documents']} documents).")
        return 1 if containment_failed else 0
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if containment_failed else 0


def _load_env() -> None:
    """Read the nearest `.env` the way the API does, without overriding a real
    environment variable. The CLI reads credentials from `os.environ`, so without
    this the OCR key is absent and `_layout_pass` fails silently: the rival read
    never happens and a corrupt text layer is trusted."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
    # Tracing is optional; the SDK warns on every run it is absent.
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        logging.getLogger("langfuse").setLevel(logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(
        prog="codify",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan-corpus", help="measure the anchor scan over a directory of raw sources"
    )
    scan.add_argument("root", help="directory searched recursively for .txt and .pdf sources")
    scan.add_argument("--jurisdiction", required=True, help="jurisdiction code, e.g. xa")
    scan.add_argument("--per-document", help="write one JSON row per document to this path")
    scan.add_argument(
        "--check",
        metavar="BASELINE",
        help="compare against a pinned baseline; exit 1 naming every drifted key",
    )
    scan.add_argument(
        "--containment",
        action="store_true",
        help=(
            "report which passes this sweep read the motivating document for; "
            "exit 1 when one was missed, so a zero firing count is not mistaken "
            "for evidence the pass is dead"
        ),
    )
    scan.add_argument(
        "--include-derived",
        action="store_true",
        help="also score our own exports (`<stem>.ar/.en/.he`), excluded by default",
    )
    scan.set_defaults(func=_run_scan_corpus)

    one = sub.add_parser("ingest-one", help="run one document through the pipeline into a bundle")
    one.add_argument("source", help="path to the source PDF, or a .txt of already-extracted text")
    one.add_argument("--jurisdiction", required=True, help="jurisdiction code, e.g. xa")
    one.add_argument("--out", required=True, help="bundle directory to write")
    one.add_argument(
        "--model",
        default=os.environ.get("LITELLM_MODEL", "gemini-3.7-flash"),
        help="body-fill model (default: %(default)s); the manifest records which model ran",
    )
    one.add_argument(
        "--ocr-model",
        default=os.environ.get("LITELLM_OCR_MODEL", ""),
        help="transcription engine; empty is the vision route on the body-fill model",
    )
    one.add_argument(
        "--fallback-model",
        default=os.environ.get("LITELLM_CONTENT_FILTER_FALLBACK_MODEL", ""),
        help="model retried on any content-filter refusal (pages, metadata, body-fill); "
        "empty disables",
    )
    one.add_argument("--quiet", action="store_true", help="suppress per-event progress")
    one.set_defaults(func=lambda a: asyncio.run(_run(a)))

    seg = sub.add_parser(
        "segment",
        help="re-run the segmenter over a bundle's stored text into segmentation.json",
    )
    seg.add_argument("bundle", help="bundle directory written by ingest-one")
    seg.add_argument(
        "--jurisdiction",
        default="",
        help="config to segment under (default: the bundle's)",
    )
    seg.set_defaults(func=_run_segment)

    idx = sub.add_parser(
        "index-datadump", help="map CELEX to archive member for a bulk FORMEX archive"
    )
    idx.add_argument("archive", help="path to the bulk FORMEX zip")
    idx.add_argument(
        "--out",
        default=None,
        help="index JSON to write (default: <ACQUISITION_INDEX_DIR>/eu/index.json)",
    )
    idx.add_argument(
        "--language",
        default="",
        help="ISO 639 code of the archive; read from its filename when omitted",
    )
    idx.add_argument("--max-files", type=int, default=None, help="stop after N members")
    idx.add_argument("--quiet", action="store_true", help="suppress progress")
    idx.set_defaults(func=_run_index_datadump)

    ee_idx = sub.add_parser(
        "index-ee-archive",
        help="index the statutes in force in a Riigi Teataja bulk XML zip",
    )
    ee_idx.add_argument("archive", help="path to the Riigi Teataja XML zip (e.g. xml.2026.zip)")
    ee_idx.add_argument(
        "--out",
        default=None,
        help="index JSON to write (default: <ACQUISITION_INDEX_DIR>/ee/index.json)",
    )
    ee_idx.add_argument(
        "--principal-only", action="store_true", help="only index principal acts and codes"
    )
    ee_idx.add_argument("--max-files", type=int, default=None, help="stop after N members")
    ee_idx.add_argument("--quiet", action="store_true", help="suppress progress")
    ee_idx.set_defaults(func=_run_index_ee_archive)

    leg = sub.add_parser(
        "index-legislation-gov-uk",
        help="enumerate every item the publisher's feeds list, per type and year",
    )
    leg.add_argument("--jurisdiction", required=True, help="jurisdiction code carrying the adapter")
    leg.add_argument(
        "--out",
        default=None,
        help="index JSON to write (default: <ACQUISITION_INDEX_DIR>/<jurisdiction>/index.json)",
    )
    leg.add_argument("--types", default="", help="comma-separated type tokens; all when omitted")
    leg.add_argument("--year-from", type=int, default=None, help="first year to list")
    leg.add_argument("--year-to", type=int, default=None, help="last year to list")
    leg.add_argument("--quiet", action="store_true", help="suppress per-year progress")
    leg.add_argument(
        "--resume",
        action="store_true",
        help="continue the build at --out (same types and years); otherwise --out is overwritten",
    )
    leg.set_defaults(func=_run_index_legislation_gov_uk)

    from codify.cli_store import register as _register_store

    _register_store(sub)

    args = parser.parse_args(argv)
    # Pipeline logs go to stderr so the manifest on stdout stays machine-readable.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
