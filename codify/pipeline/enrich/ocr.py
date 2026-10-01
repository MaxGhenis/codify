"""PDF text extraction with parallel vision OCR."""

from __future__ import annotations

import asyncio
import inspect
import io
import re
import threading
import unicodedata
import weakref
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pypdf import PdfReader, PdfWriter

from codify.core.llm import LLMClient, VisionText, content_filtered
from codify.core.tracing import error_fields
from codify.pipeline.enrich.scripts import ScriptPack
from codify.quality.legibility import lexicon_rate
from codify.quality.lexicons import ARABIC_WORDS
from codify.quality.page_read import BLANK_INK, PageVerdict, divergence, ink_ratio, score_page

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
OCR_SYSTEM_PROMPT = (PROMPTS_DIR / "ocr_system.txt").read_text()

MIN_TEXT_LENGTH = 50
# Sized to Mistral OCR's 40 RPM budget. Bursts fire above steady-state, then
# LiteLLM's 429 backoff levels flow to the target rate. Sole-tenant on the
# endpoint means burst budget is ours; lower on shared deployments.
MAX_CONCURRENT = 35
# Pages rasterised per render call. The per-page path re-parsed the whole PDF
# for every page, which is quadratic in document size; a window amortises the
# parse without holding a whole gazette's images in memory at once.
VISION_RENDER_WINDOW = 8
# Pixels one render window may produce in total, summed across its pages:
# pdf2image decodes the whole window at once, so a per-page cap peaks at eight
# times itself. Neither declared size nor dpi is bounded by the upload cap, and
# Pillow's bomb guard sits at ~89 Mpx per image. Eight A4 pages at 300dpi is
# ~70 Mpx and ~1 GB resident.
MAX_WINDOW_PIXELS = 70_000_000
# A page whose /MediaBox is missing: poppler defaults to US Letter.
LETTER_AREA_SQ_IN = 8.5 * 11.0
# A guard, not a tuning: no diverted page measured so far carries a text layer
# over 79 characters, so this has never bound.
ANCHOR_CHAR_LIMIT = 4000
# Azure AI Foundry's Mistral OCR endpoint rejects documents >30 pages with
# `document_parser_too_many_pages` (code 3730). 25 leaves headroom in case the
# limit tightens; PDFs at or under this cap go in one call.
MISTRAL_OCR_MAX_PAGES_PER_CALL = 25
# The engine asked for layout, whatever reads the text. Only this endpoint
# separates furniture from body and returns typed regions with geometry.
LAYOUT_MODEL = "mistral-ocr-4-0"
# Content nouns per 1,000 tokens, above which a particle-less page reads as a
# list rather than a ruin. Higher than the document-level floor, an index page
# carrying names densely in few tokens. Over 2,537 documents it spares 358 pages
# with intact vocabulary and keeps the 643 whose vocabulary is gone.
GATE_LEXICON_FLOOR = 30.0

# The OCR endpoint writes a figure into the markdown as `![img-0.jpeg](img-0.jpeg)`,
# and on a gazette that lands above the law title, where a masthead once did. The
# figure is not lost: the block keeps its id and geometry on the page layout.
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

BLANK_HALLUCINATIONS = [
    "this page is blank",
    "the provided image is blank",
    "there is no text to extract",
    "the image does not contain any text",
    "no text content",
    "blank page",
]


# This gate is Arabic-only by construction: it detects a CMap fault that
# scrambles Arabic letter forms. Other scripts fail differently and are left
# to the normal path.


def _looks_garbled(text: str) -> bool:
    """True for an Arabic-dominant page scrambled by a bad font CMap: no particle
    density, or doubled-alef and mid-word alef-maqsura runs. Such pages OCR rather
    than trust the text layer. Non-Arabic is never flagged.
    """
    text = unicodedata.normalize("NFC", text)
    # Only the base Arabic block; presentation-form PDFs (U+FB50+) aren't seen
    # in this corpus and would read as non-Arabic, i.e. left to the normal path.
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    # Guard on chars, not words: garble runs together into few tokens. Share is
    # over non-whitespace, or a spaced two-column extract of the same garble
    # falls under the floor while the tighter extract diverts.
    nonspace = sum(1 for c in text if not c.isspace())
    if arabic < 60 or arabic < 0.4 * max(nonspace, 1):
        return False
    words = text.split()
    n = max(len(words), 1)
    double_alef = text.count("اا") / n
    # Particle density is paired with vocabulary rate. Requiring definite-
    # article density too misses corrupt text that preserves ``ال`` while
    # destroying function words; vocabulary also avoids diverting clean lists.
    common = sum(1 for w in words if w in ARABIC_WORDS.function_words) / n
    lexicon = lexicon_rate(text, ARABIC_WORDS)
    no_prose_and_no_vocabulary = common < 0.05 and (lexicon is None or lexicon < GATE_LEXICON_FLOOR)
    # A bad CMap injects alef-maqsura (ى) mid-word while leaving article density
    # intact. Clean Arabic carries ى only word-finally, so a mid-word rate above
    # a hair is a garble tell (clean ~0, corruption ~0.1).
    midword_maqsura = len(re.findall(r"ى(?=[؀-ۿ])", text)) / arabic
    return double_alef > 0.3 or midword_maqsura > 0.02 or no_prose_and_no_vocabulary


# Al-Muqtafi PDFs watermark every page, so a scanned law's text layer holds only
# the library credit. Strip the stamp so the page falls below MIN_TEXT_LENGTH
# and OCRs.
@lru_cache(maxsize=64)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def furniture_inline_patterns(country: str) -> tuple[str, ...]:
    """Furniture sharing a line with content, which the jurisdiction declares."""
    if not country:
        return ()
    from codify.jurisdictions import try_load_config

    config = try_load_config(country)
    return tuple(config.furniture_inline_patterns) if config else ()


def furniture_line_patterns(country: str) -> Sequence[str]:
    """Whole-line furniture patterns the jurisdiction declares, or none."""
    if not country:
        return ()
    from codify.jurisdictions import try_load_config

    config = try_load_config(country)
    return tuple(config.furniture_line_patterns) if config else ()


def _strip_furniture_lines(text: str, patterns: Sequence[str]) -> str:
    """Drop whole lines that are source furniture. The jurisdiction declares the
    patterns; a jurisdiction that declares none loses no lines."""
    rx = _compiled(tuple(patterns))
    return "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not any(r.search(ln) for r in rx)
    )


def _scrub_furniture_inline(text: str, patterns: Sequence[str]) -> str:
    """Remove furniture that shares a line with real content, which whole-line
    disposal would take the content with."""
    for rx in _compiled(tuple(patterns)):
        text = rx.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _is_presentation_form_heavy(text: str) -> bool:
    """True when a text layer is dominated by Arabic Presentation Forms
    (U+FB50-FEFF), which extract as isolated glyphs, often reversed, with article
    numbers detached from `مادة`. The page renders fine, so OCR recovers clean
    standard Arabic.
    """
    nonspace = [c for c in text if not c.isspace()]
    if len(nonspace) < 40:
        return False
    pf = sum(1 for c in nonspace if 0xFB50 <= ord(c) <= 0xFEFF)
    return pf / len(nonspace) > 0.3


def _clean_ocr_output(text: str) -> str:
    """Strip blank-page hallucinations and image placeholders from OCR output."""
    stripped = _IMAGE_REF_RE.sub("", text).strip()
    if len(stripped) < 200:
        lower = stripped.lower()
        for pattern in BLANK_HALLUCINATIONS:
            if pattern in lower:
                return ""
    return stripped


async def _call(fn: Callable[..., object] | None, *args: object) -> None:
    """Call a callback, awaiting if it's async."""
    if fn is None:
        return
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


class OcrBlock(BaseModel):
    """One typed region as the OCR endpoint returned it.

    Geometry is four named ordinates in the source page's units, not a `bbox`
    list, and not the dpi we render at. Scale by `PageDimensions` to draw it.
    """

    type: str
    content: str = ""
    top_left_x: int = 0
    top_left_y: int = 0
    bottom_right_x: int = 0
    bottom_right_y: int = 0
    table_id: str | None = None
    image_id: str | None = None


class PageDimensions(BaseModel):
    """The source page's own units. Observed dpi 87 where we render at 300."""

    dpi: int
    width: int
    height: int


def crop_box_for_block(
    block: OcrBlock, dims: PageDimensions, *, render_dpi: int
) -> tuple[int, int, int, int]:
    """PIL crop box `(left, top, right, bottom)` for a block on a rendered page.

    Block ordinates are in `dims.dpi` units, so scale them onto `render_dpi`. Raises
    on geometry naming no real region, before the expensive render rather than after.
    """
    if dims.dpi <= 0 or render_dpi <= 0:
        raise ValueError("block geometry unavailable: non-positive dpi")
    scale = render_dpi / dims.dpi
    left = round(block.top_left_x * scale)
    top = round(block.top_left_y * scale)
    right = round(block.bottom_right_x * scale)
    bottom = round(block.bottom_right_y * scale)
    if right <= left or bottom <= top:
        raise ValueError("block geometry unavailable: box has no positive area")
    return (left, top, right, bottom)


class WordConfidence(BaseModel):
    """`start_index` is an offset into the markdown, so doubt maps onto text."""

    text: str
    confidence: float
    start_index: int


class PageLayout(BaseModel):
    """What a layout-capable engine saw on the page, as it reported it.

    Absent rather than empty when no engine looked. Strict on unknown keys, unlike
    `OcrBlock`: only our own dumps are read back, and a drifted key would silently
    drop every block.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str
    model: str = ""
    header: str = ""
    footer: str = ""
    blocks: list[OcrBlock] = Field(default_factory=list)
    dimensions: PageDimensions | None = None
    page_confidence: float | None = None
    min_word_confidence: float | None = None
    words: list[WordConfidence] = Field(default_factory=list)
    hyperlinks: list[str] = Field(default_factory=list)
    # Response fields with no published shape and none observed yet: `tables` and
    # `images` were both empty on the page this was pinned against.
    raw: dict[str, Any] = Field(default_factory=dict)


class PageResult(BaseModel):
    page_number: int
    text: str
    method: str
    # Why the text layer was not trusted; empty when it was. On the result
    # rather than the log so a later bundle still names the reason.
    divert_reason: str = ""
    # The engine that actually produced this text, which is not always the one
    # asked for: a rate-limited OCR deployment falls back to the vision route.
    # Empty for a trusted text layer, which had no engine.
    model: str = ""
    # Header and footer the endpoint lifted out of `text`. Retained, not dropped:
    # the footer carries the gazette issue number and a misread heading survives.
    header: str = ""
    footer: str = ""
    # Furniture patterns this jurisdiction declares, set where the country is
    # known so every page-level helper scrubs without being handed them again.
    furniture: tuple[str, ...] = ()
    # What a layout-capable engine saw. None means nobody looked, which is not
    # the same as a page with no furniture on it.
    layout: PageLayout | None = Field(default=None, exclude=True)
    # False until two engines each produced text for this page. Settled at the
    # end of extraction, because promotion moves a read between the two fields.
    rival_available: bool = False
    # The other engine's transcription of the same page, kept but never
    # authoritative, and `divergence` is how far apart the two read.
    rival_text: str = ""
    divergence: float | None = None
    # Dark fraction of the rendered page, from `ink_ratio`, computed only where
    # the render exists (the vision route). None on the Mistral and text lanes,
    # which rasterise nothing; the verdict abstains on ink there by design.
    ink: float | None = None
    # The vision route's finish reason. A content-filter refusal returns empty
    # text, which without this reads the same as a blank leaf.
    finish_reason: str = ""


# The classifier reads `blocks` and `dimensions`; the rest is bulk (`words`
# alone is ~70% of a gazette page) and `page_reads.layout` already holds it.
_LAYOUT_DUMP_EXCLUDE = frozenset(
    {"raw", "words", "hyperlinks", "page_confidence", "min_word_confidence"}
)


def layout_to_json(pages: Sequence[PageResult]) -> dict[str, Any]:
    """Per-page layout for a run artifact. One serialisation, both ingest lanes.

    Absent page != empty page: `layout` is None on the vision route.
    """
    return {
        str(page.page_number): page.layout.model_dump(
            exclude=set(_LAYOUT_DUMP_EXCLUDE), mode="json"
        )
        for page in pages
        if page.layout is not None
    }


def layout_from_json(raw: dict[str, Any] | None) -> dict[int, PageLayout]:
    """Inverse of `layout_to_json`. The list branch reads the legacy four-key
    shape, for DBOS runs replaying against an artifact the old code wrote."""
    # Anything else raises: a skipped page reads downstream as an empty one.
    return {
        int(page): (
            _layout_from_legacy_blocks(payload)
            if isinstance(payload, list)
            else PageLayout.model_validate(payload)
        )
        for page, payload in (raw or {}).items()
    }


def _layout_from_legacy_blocks(raw: list[Any]) -> PageLayout:
    """Read the legacy block-list shape, where dimensions occupy a pseudo-block."""
    dims = next((r for r in raw if isinstance(r, dict) and r.get("type") == "_dimensions"), None)
    return PageLayout(
        engine="",
        blocks=[
            OcrBlock(
                type=str(r.get("type") or ""),
                content=str(r.get("content") or ""),
                top_left_y=int(r.get("y0") or 0),
                bottom_right_y=int(r.get("y1") or 0),
            )
            for r in raw
            if isinstance(r, dict) and r.get("type") != "_dimensions"
        ],
        # Only height was recorded, and only height is read.
        dimensions=PageDimensions(dpi=0, width=0, height=int(dims["height"])) if dims else None,
    )


def build_document_context(
    pdf_bytes: bytes,
    title: str = "",
    country: str = "",
) -> str:
    """Fast profile from pypdf partial text + upload metadata."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    snippets = []
    for page in reader.pages[:3]:
        text = (page.extract_text() or "").strip()[:200]
        if text:
            snippets.append(text)

    parts = []
    if title:
        parts.append(f"Document: {title}")
    if country:
        parts.append(f"Jurisdiction: {country.upper()}")
    parts.append(f"Pages: {len(reader.pages)}")
    if snippets:
        parts.append(f"Partial content: {' | '.join(snippets)}")
    return "\n".join(parts)


async def _mistral_ocr_pdf_chunked(
    *,
    client: LLMClient,
    pdf_bytes: bytes,
    reader: PdfReader,
    total_pages: int,
    model: str,
    on_progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    """Send a PDF to Mistral OCR, splitting into ≤MISTRAL_OCR_MAX_PAGES_PER_CALL
    chunks when needed. Rewrites each chunk's page ``index`` back to the
    document's absolute page number so downstream reassembly is transparent.
    Chunks run concurrently; ordering is restored by page index."""
    if total_pages <= MISTRAL_OCR_MAX_PAGES_PER_CALL:
        return await client.ocr(pdf=pdf_bytes, model=model)

    chunk_bytes: list[tuple[int, bytes]] = []
    for start in range(0, total_pages, MISTRAL_OCR_MAX_PAGES_PER_CALL):
        end = min(start + MISTRAL_OCR_MAX_PAGES_PER_CALL, total_pages)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        chunk_bytes.append((start, buf.getvalue()))

    await _call(
        on_progress,
        f"Mistral OCR split into {len(chunk_bytes)} chunks "
        f"of ≤{MISTRAL_OCR_MAX_PAGES_PER_CALL} pages",
    )

    # Bound concurrent chunks so a large PDF (e.g. 900pg → 36 chunks) can't
    # burst past the Azure Foundry per-minute quota that MAX_CONCURRENT
    # already sizes the per-page fallback against.
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _one(start: int, data: bytes) -> list[dict[str, Any]]:
        async with sem:
            pages = await client.ocr(pdf=data, model=model)
        for p in pages:
            idx = p.get("index")
            if idx is not None:
                p["index"] = int(idx) + start
        return pages

    results = await asyncio.gather(*(_one(s, d) for s, d in chunk_bytes))
    merged: list[dict[str, Any]] = []
    for chunk_pages in results:
        merged.extend(chunk_pages)
    # `or 0`, not a get default: a present-but-null index raises in int() and the
    # caller's except swallows it into a silent engine downgrade.
    merged.sort(key=lambda p: int(p.get("index") or 0))
    return merged


OCR_ONLY_MODEL_PREFIXES = ("mistral-ocr", "mistral-document")
"""Deployments that answer on the OCR endpoint and nowhere else."""


_VISION_GATES: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _vision_gate() -> asyncio.Semaphore:
    """One vision budget per event loop, not per document.

    Per loop rather than per process, because DBOS uses a second loop when launched
    from sync code and would take a second full budget against the same quota. A
    semaphore inside `extract_text_from_pdf` bounds one document, so the real ceiling
    was `ingest_queue_concurrency` times `MAX_CONCURRENT`.
    """
    loop = asyncio.get_running_loop()
    gate = _VISION_GATES.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(MAX_CONCURRENT)
        _VISION_GATES[loop] = gate
    return gate


def _render_windows(pages: list[int]) -> list[list[int]]:
    """Group diverted pages into windows bounded by page span, not by count.

    Chunking by count let a sparse set (pages 1, 50, 900) render the whole span
    between its ends, which on a 16 GB host is an OOM.
    """
    windows: list[list[int]] = []
    for page in pages:
        if windows and page - windows[-1][0] < VISION_RENDER_WINDOW:
            windows[-1].append(page)
        else:
            windows.append([page])
    return windows


# Each render holds a whole window's bitmaps (~1 GB at the ceiling) and the API
# container is also the DBOS worker, so this times MAX_WINDOW_PIXELS is the real
# transient budget: two concurrent windows against a 6g limit.
MAX_CONCURRENT_RENDERS = 2
_RENDER_GATES: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = weakref.WeakKeyDictionary()
# The real bound, and it must be threading rather than asyncio: DBOS runs
# workflows on its own loop (api/runs/resources.py), so a per-loop semaphore
# hands each loop a full budget and the process gets two of them.
_RENDER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RENDERS)


def _render_gate() -> asyncio.Semaphore:
    """Keeps renders from piling into the thread pool. One per loop, as
    `_vision_gate` does; `_RENDER_SLOTS` is what actually bounds the process."""
    loop = asyncio.get_running_loop()
    gate = _RENDER_GATES.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)
        _RENDER_GATES[loop] = gate
    return gate


def _dpi_for_pages(pdf_bytes: bytes, first: int, last: int, dpi: int) -> int:
    """`dpi`, lowered so the whole window fits under the pixel ceiling.

    Per window: pdf2image reads the window's raw output before decoding, so eight
    pages each under a per-page cap peak at eight times it. Clamps rather than
    refuses, an oversized page usually being a legitimate plan. An unreadable box
    falls through to the requested dpi and lets the renderer complain.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages[first - 1 : last]
    except Exception:  # noqa: BLE001 - see the docstring: advisory, not a validator
        return dpi
    total_area = 0.0
    for page in pages:
        try:
            box = page.mediabox
            # abs(): an inverted box ([0 0 -w h]) is legal and poppler normalises
            # it, so a signed area would read as "no geometry" and skip the clamp.
            area = abs(float(box.width)) * abs(float(box.height)) / (72.0 * 72.0)
        except Exception:  # noqa: BLE001 - a page with no /MediaBox renders at the default
            area = LETTER_AREA_SQ_IN
        total_area += area
    if total_area <= 0:
        return dpi
    applied = min(dpi, int((MAX_WINDOW_PIXELS / total_area) ** 0.5))
    if applied < dpi:
        logger.warning(
            "render_dpi_clamped",
            requested=dpi,
            applied=applied,
            first=first,
            last=last,
            window_sq_in=round(total_area, 1),
        )
    # No floor: past ~12,650pt square a floor binds and pixel count grows
    # quadratically again, which is the bomb this stops. An unreadably large
    # page renders badly, which beats a dead container.
    return max(applied, 1)


async def _render_window(pdf_bytes: bytes, first: int, last: int, dpi: int) -> dict[int, bytes]:
    """Rasterise pages `first..last` in one parse, off the event loop.

    Poppler failures propagate: a page that will not render fails the run."""

    effective_dpi = _dpi_for_pages(pdf_bytes, first, last, dpi)

    def _work() -> dict[int, bytes]:
        from pdf2image import convert_from_bytes

        with _RENDER_SLOTS:
            out: dict[int, bytes] = {}
            for offset, image in enumerate(
                convert_from_bytes(pdf_bytes, first_page=first, last_page=last, dpi=effective_dpi)
            ):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                out[first + offset] = buf.getvalue()
            return out

    try:
        async with _render_gate():
            return await asyncio.to_thread(_work)
    except ImportError as exc:
        # Only the absent package. Poppler errors derive from bare Exception,
        # and swallowing them records an unrenderable page as read.
        logger.error("pdf2image_missing", first=first, last=last, error=str(exc)[:300])
        return {}


async def render_block(
    pdf_bytes: bytes,
    page_no: int,
    block: OcrBlock,
    dims: PageDimensions,
    *,
    base_dpi: int = 300,
    zoom: float = 1.0,
) -> bytes:
    """PNG of a single block, re-rendered at `base_dpi * zoom` then cropped.

    The cheap re-read persisted geometry enables: four blocks at 4x rather than a
    whole page. `zoom` re-rasterises for genuine detail, not an upscale.
    """
    render_dpi = round(base_dpi * zoom)
    # The clamp may lower this, and the crop box scales by dpi, so ask what was
    # actually rendered. Reading the requested dpi displaces the crop on any
    # page large enough to clamp: an A3 gazette at 600 lands 1.3x off.
    effective_dpi = _dpi_for_pages(pdf_bytes, page_no, page_no, render_dpi)
    box = crop_box_for_block(block, dims, render_dpi=effective_dpi)
    pages = await _render_window(pdf_bytes, page_no, page_no, render_dpi)
    png = pages.get(page_no)
    if png is None:
        raise ValueError(f"page {page_no} did not render")

    def _crop() -> bytes:
        from PIL import Image

        with Image.open(io.BytesIO(png)) as image:
            # A box overshooting the page (dpi is approximate) black-pads rather
            # than clamps; a thin margin on a re-read crop is harmless.
            buf = io.BytesIO()
            image.crop(box).save(buf, format="PNG")
            return buf.getvalue()

    return await asyncio.to_thread(_crop)


def _is_ocr_only_model(model: str | None) -> TypeGuard[str]:
    return model is not None and model.startswith(OCR_ONLY_MODEL_PREFIXES)


def resolve_ocr_model(configured: str | None) -> str | None:
    """Vision by default. Config may override, loudly."""
    if not configured:
        return None
    logger.warning(
        "ocr_engine_overridden",
        model=configured,
        default="vision",
    )
    return configured


def _vision_model(ocr_model: str | None) -> str | None:
    """The model for the per-page chat vision route, or None for the default.

    A Mistral OCR deployment answers on the OCR endpoint only, so asking the chat
    gateway for it is a guaranteed 400 on every page.
    """
    return None if _is_ocr_only_model(ocr_model) else ocr_model


class _LayoutPassFailed(RuntimeError):
    """The layout engine returned nothing. Caught inside the pipeline, where the
    pages then record that nothing read them a second time."""


async def _layout_pass(
    client: LLMClient,
    pdf_bytes: bytes,
    reader: PdfReader,
    total_pages: int,
) -> tuple[dict[int, PageLayout], dict[int, str]]:
    """Layout and a rival transcription for every page, best effort.

    Transcription fails the run; layout does not. A failure here leaves layout None,
    which reads as nobody looked rather than as a page with no furniture.
    """
    try:
        pages = await _mistral_ocr_pdf_chunked(
            client=client,
            pdf_bytes=pdf_bytes,
            reader=reader,
            total_pages=total_pages,
            model=LAYOUT_MODEL,
            on_progress=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "layout_pass_failed",
            **error_fields(exc),
            model=LAYOUT_MODEL,
            pages=total_pages,
        )
        raise _LayoutPassFailed from exc
    layouts: dict[int, PageLayout] = {}
    rivals: dict[int, str] = {}
    for p in pages:
        try:
            page_num = int(p["index"]) + 1
        except (KeyError, TypeError, ValueError):
            continue
        header = _strip_control_chars(p.get("header") or "").strip()
        footer = _strip_control_chars(p.get("footer") or "").strip()
        rivals[page_num] = _clean_ocr_output(p.get("markdown") or "")
        try:
            layouts[page_num] = _layout_from_page(p, LAYOUT_MODEL, header, footer)
        except ValidationError as exc:
            # One malformed ancillary field must not cost the transcription. No
            # layout for this page reads as nobody looked, which is the truth.
            # Field paths, not `str(exc)`: pydantic quotes the input.
            logger.warning(
                "layout_page_unparsed",
                page=page_num,
                **error_fields(exc),
                fields=sorted({".".join(str(part) for part in e["loc"]) for e in exc.errors()}),
            )
    logger.info("layout_pass_done", pages=len(layouts), expected=total_pages)
    return layouts, rivals


# Token-Jaccard divergence above which an embedded text layer is distrusted and
# the page diverts to vision OCR. Injected same-script glyphs pass every other
# trust check but disagree with a clean vision read. Clean layers sit ~0.3, and
# every sampled page above ~0.55 was corrupt. Unlike `_DIVERGENCE_HIGH`, this
# switches which read wins.
_TEXT_LAYER_MAX_DIVERGENCE = 0.5

# Reasons that fault the layer's content rather than its quantity. A layer
# diverted for one of these does not anchor the read that replaces it.
_DISTRUSTED_LAYER_REASONS = frozenset({"text_layer_divergent", "letter_spaced"})


def _text_layer_untrusted(divergence_score: float | None, threshold: float) -> bool:
    """A born-digital layer disagreeing with the rival beyond ``threshold`` is
    untrusted. ``None`` (no rival to compare) never faults the layer: absence of
    a second read is not evidence against the first."""
    return divergence_score is not None and divergence_score > threshold


def _layer_split_words(text: str, rival: str) -> set[str]:
    """Words the layer spells as loose letters and the rival spells whole. Some
    born-digital layers encode a tracked-out label as literal spaces, which a
    divergence score cannot see: the damage is a few words on a page of hundreds.
    Returned rather than counted so the divert can record its evidence."""
    from codify.quality.page_read import split_words_the_rival_keeps

    if not rival.strip():
        return set()
    # Cleaned on both sides: the damaged layer usually double-spaces, which reads
    # as nothing until the padding collapses.
    return split_words_the_rival_keeps(clean_page_text(text), clean_page_text(rival))


def _divert_decision(
    text: str, layer_diverges: bool, split_words: set[str], furniture: Sequence[str] = ()
) -> tuple[bool, str]:
    """Whether a page diverts to OCR, and the reason if it does.

    One ordered evaluation of the five predicates, so the gate and its reason cannot
    disagree. Evaluated separately, a predicate could fire the gate while the label
    fell through to "unknown".
    """
    if len(_strip_furniture_lines(text, furniture)) < MIN_TEXT_LENGTH:
        return True, "too_short"
    if _looks_garbled(text):
        return True, "garbled"
    if _is_presentation_form_heavy(text):
        return True, "presentation_forms"
    if layer_diverges:
        return True, "text_layer_divergent"
    if split_words:
        return True, "letter_spaced"
    return False, ""


def _vision_anchor(divert_reason: str, fallback_text: str) -> str:
    """The text hint the vision re-read is anchored on. A layer diverted for what
    its text says is distrusted, so it does not anchor its own replacement: the
    damage that got it pulled must not nudge the fresh read. Other diverts (too
    short, garbled) anchor on what little text was there."""
    return "" if divert_reason in _DISTRUSTED_LAYER_REASONS else fallback_text


def _settle_rival_available(pages: list[PageResult]) -> list[PageResult]:
    """Two engines each produced text for this page. Read from the texts and
    read last, because promotion moves an empty read between the two fields."""
    return [
        p.model_copy(update={"rival_available": bool(p.text.strip() and p.rival_text.strip())})
        for p in pages
    ]


def _promote_rival_when_empty(page: PageResult) -> PageResult:
    """Keep the layout-pass transcription for a page the authoritative read left empty.

        A RECITATION refusal on famous legal text returns nothing (`finish_reason`
    `content_filter`), so a scanned page
        reads as blank though the layout pass already transcribed it. A measured engine
        A/B put vision ahead wherever both engines read, so this fires only on an empty
        read, never to override a real one.

        Recorded rather than hidden: the empty vision read moves into `rival_text`, so
        the `page_reads` row still shows the primary engine returned nothing.
    """
    if (page.text or "").strip() or not (page.rival_text or "").strip():
        return page
    if (
        page.ink is not None
        and page.ink < BLANK_INK
        and page.divert_reason not in _DISTRUSTED_LAYER_REASONS
    ):
        # A genuinely blank leaf: the empty read was right and a rival emitting
        # body text is noise. Same cutoff as `diverted_to_nothing`. A distrusted
        # layer is exempt, having carried text over the length floor.
        return page
    logger.warning(
        "ocr_rival_promoted",
        page=page.page_number,
        rival_chars=len(page.rival_text.strip()),
        divert_reason=page.divert_reason,
    )
    return page.model_copy(
        update={
            "text": page.rival_text,
            # The empty read that fired promotion is now the recorded rival, so
            # the trace keeps "the primary returned nothing".
            "rival_text": page.text,
            "method": "mistral_ocr",
            # Promotion fires only on a vision-route page, whose rival is always
            # the layout pass, so LAYOUT_MODEL is the true source.
            "model": LAYOUT_MODEL,
            "divergence": None,
        }
    )


def _has_visible_text_overlay(page: Any) -> bool:
    """Free-text and form appearances are absent from pypdf's text layer."""
    if not hasattr(page, "get"):
        return False
    for ref in page.get("/Annots", []) or []:
        annotation = ref.get_object() if hasattr(ref, "get_object") else ref
        if not isinstance(annotation, Mapping):
            continue
        if int(annotation.get("/F", 0)) & (2 | 32):
            continue
        subtype = annotation.get("/Subtype")
        if subtype == "/FreeText" and annotation.get("/Contents"):
            return True
        if subtype in {"/Widget", "/Stamp"} and annotation.get("/AP"):
            return True
    return False


async def extract_text_from_pdf(
    pdf_bytes: bytes,
    client: LLMClient | None = None,
    on_progress: Callable[[str], None] | None = None,
    title: str = "",
    country: str = "",
    ocr_model: str | None = None,
    ocr_dpi: int = 300,
    text_layer_max_divergence: float = _TEXT_LAYER_MAX_DIVERGENCE,
) -> list[PageResult]:
    """Extract text from a PDF, OCRing scanned pages in parallel via the vision model.

    ``ocr_model`` overrides the client default; ``ocr_dpi`` is the render resolution,
    higher recovering more from faded scans at more pixels and cost.
    ``text_layer_max_divergence`` diverts a born-digital page when its embedded text
    disagrees with the rival read beyond that token-Jaccard threshold.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    overlay_pages = {
        i + 1 for i, page in enumerate(reader.pages) if _has_visible_text_overlay(page)
    }
    doc_context = build_document_context(pdf_bytes, title, country)
    # Source furniture is declared per jurisdiction, not compiled in: a portal's
    # masthead is data about that source, and it leaves with the config.
    furniture_lines = furniture_line_patterns(country)
    furniture_inline = furniture_inline_patterns(country)

    # Fast path: Mistral OCR reads the whole PDF in one call, keeping cross-page
    # context and layout. Needs a client configured for direct Azure Foundry.
    salvaged_layouts: dict[int, PageLayout] = {}
    salvaged_rivals: dict[int, str] = {}
    authoritative_pages: dict[int, PageResult] = {}
    mistral_complete = False
    if client and _is_ocr_only_model(ocr_model):
        await _call(on_progress, f"Mistral OCR reading all {total_pages} pages...")
        try:
            pages = await _mistral_ocr_pdf_chunked(
                client=client,
                pdf_bytes=pdf_bytes,
                reader=reader,
                total_pages=total_pages,
                model=ocr_model,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            # Loud, because the fallback silently swaps the OCR engine: whatever
            # comes back is no longer the model the caller asked for, and the
            # run still reports success.
            logger.error(
                "mistral_ocr_failed_falling_back_to_vision",
                **error_fields(exc),
                requested_model=ocr_model,
                pages=total_pages,
            )
            pages = []
        if pages:
            results: list[PageResult] = []
            all_furniture: list[int] = []
            for p in pages:
                # Any index that will not place the page, None or otherwise:
                # loud here, and the page-set check below turns it into a re-read.
                try:
                    page_num = int(p["index"]) + 1
                except (KeyError, TypeError, ValueError):
                    logger.error(
                        "ocr_page_missing_index",
                        index=repr(p.get("index"))[:40],
                        markdown_chars=len(p.get("markdown") or ""),
                        has_furniture=bool(p.get("header") or p.get("footer")),
                    )
                    continue
                text = _clean_ocr_output(p.get("markdown") or "")
                # Same control-char scrub the body gets: these strings land in a
                # jsonb column, which rejects NUL outright.
                header = _strip_control_chars(p.get("header") or "").strip()
                footer = _strip_control_chars(p.get("footer") or "").strip()
                if not text and (header or footer):
                    all_furniture.append(page_num)

                results.append(
                    PageResult(
                        furniture=furniture_inline,
                        page_number=page_num,
                        text=text,
                        method="mistral_ocr",
                        model=ocr_model or "",
                        header=header,
                        footer=footer,
                        layout=_layout_from_page(p, ocr_model or "", header, footer),
                    )
                )
            results.sort(key=lambda r: r.page_number)
            # Compare the page SET, not the count: a duplicated index masks a
            # missing one when the totals happen to agree, and that commits one
            # page's text twice while another page of law is gone.
            expected_pages = set(range(1, total_pages + 1))
            numbered = [r.page_number for r in results]
            got = set(numbered)
            mistral_complete = got == expected_pages and len(results) == total_pages
            if mistral_complete:
                if all_furniture:
                    # One event per document, as low_text_pages does: a gazette
                    # scan would otherwise emit hundreds of identical warnings.
                    logger.warning("ocr_page_all_furniture", pages=list(all_furniture))
                logger.info(
                    "mistral_ocr_full_pdf_done",
                    total_pages=total_pages,
                    returned=len(results),
                )
                if not overlay_pages:
                    return results
                authoritative_pages = {
                    r.page_number: r for r in results if r.page_number not in overlay_pages
                }
            # Reuse layout and rival text for overlay reads or incomplete-response fallback.
            salvaged_layouts = {r.page_number: r.layout for r in results if r.layout}
            salvaged_rivals = {r.page_number: r.text for r in results if r.text}
            if not mistral_complete:
                logger.error(
                    "mistral_ocr_page_set_mismatch",
                    expected=total_pages,
                    returned=len(results),
                    missing=sorted(expected_pages - got),
                    extra=sorted(got - expected_pages),
                    duplicated=sorted({n for n in got if numbered.count(n) > 1}),
                )

    # Layout for every page, from the one engine that separates it. Runs even
    # when the text layer is trusted: geometry and furniture are what the trace
    # viewer and the region classifier consume, and a born-digital page has both.
    layouts: dict[int, PageLayout] = salvaged_layouts
    rivals: dict[int, str] = salvaged_rivals
    if client and not layouts and not mistral_complete:
        try:
            layouts, rivals = await _layout_pass(client, pdf_bytes, reader, total_pages)
        except _LayoutPassFailed:
            # Already logged with its cause. The pages carry the consequence.
            layouts, rivals = {}, {}

    # Separate pages into text-extractable and needs-OCR
    text_results: dict[int, PageResult] = dict(authoritative_pages)
    ocr_page_nums: list[int] = []
    # Raw text kept per diverted page, so a page is never dropped if OCR can't run.
    ocr_fallback_text: dict[int, str] = {}
    divert_reasons: dict[int, str] = {}

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        if page_num in authoritative_pages:
            continue
        text = (page.extract_text() or "").strip()
        rival = rivals.get(page_num, "")
        div = divergence(text, rival)
        layer_diverges = _text_layer_untrusted(div, text_layer_max_divergence)
        split_words = _layer_split_words(text, rival)

        diverted, reason = _divert_decision(text, layer_diverges, split_words, furniture_lines)
        if page_num in overlay_pages:
            diverted, reason = True, "visible_annotation"
        if not diverted:
            result = PageResult(
                furniture=furniture_inline,
                page_number=page_num,
                text=text,
                method="text_extraction",
                layout=layouts.get(page_num),
                rival_text=rival,
                divergence=div,
            )
            text_results[page_num] = result
            await _call(on_progress, f"Page {page_num}/{total_pages}: text extracted")
            logger.info("page_text_extracted", page=page_num, total=total_pages, chars=len(text))
        else:
            logger.info(
                "page_diverted_to_ocr",
                page=page_num,
                total=total_pages,
                reason=reason,
                # The evidence, so a divert can be audited after the fact.
                split_words=sorted(split_words)[:5] if split_words else None,
            )
            divert_reasons[page_num] = reason
            ocr_page_nums.append(page_num)
            ocr_fallback_text[page_num] = text

    # Parallel OCR for scanned pages, in render windows so the PDF is parsed
    # once per window rather than once per page.
    if ocr_page_nums and client:
        await _call(on_progress, f"OCR {len(ocr_page_nums)} scanned pages in parallel...")
        gate = _vision_gate()

        async def ocr_page(page_num: int, image_bytes: bytes | None) -> PageResult:
            async with gate:
                anchor = _vision_anchor(
                    divert_reasons.get(page_num, ""), ocr_fallback_text.get(page_num, "")
                )
                logger.info(
                    "page_vision_ocr_start",
                    page=page_num,
                    total=total_pages,
                    anchor_chars=len(anchor.strip()),
                )
                text = await _ocr_page_with_context(
                    pdf_bytes,
                    page_num,
                    total_pages,
                    client,
                    doc_context,
                    _vision_model(ocr_model),
                    ocr_dpi,
                    image_bytes=image_bytes,
                    render_if_missing=False,
                    page_text_layer=anchor,
                )
                await _call(on_progress, f"Page {page_num}/{total_pages}: OCR complete")
                finish_reason = getattr(text, "finish_reason", "")
                logger.info(
                    "page_vision_ocr_done", page=page_num, chars=len(text), finish=finish_reason
                )
                return PageResult(
                    furniture=furniture_inline,
                    page_number=page_num,
                    text=text,
                    method="vision_ocr",
                    divert_reason=divert_reasons.get(page_num, ""),
                    model=_vision_model(ocr_model) or str(getattr(client, "model", "")),
                    layout=layouts.get(page_num),
                    rival_text=rivals.get(page_num, ""),
                    divergence=divergence(text, rivals.get(page_num, "")),
                    ink=ink_ratio(image_bytes) if image_bytes is not None else None,
                    finish_reason=finish_reason,
                )

        for window in _render_windows(ocr_page_nums):
            images = await _render_window(pdf_bytes, window[0], window[-1], ocr_dpi)
            # No `return_exceptions`: until a degraded page is represented on the
            # run and in `ocr_source`, a page that will not read fails the run,
            # which is loud and recoverable, rather than shipping a law short of
            # its source.
            for result in await asyncio.gather(*[ocr_page(n, images.get(n)) for n in window]):
                text_results[result.page_number] = result

    # OCR couldn't run (no client): keep each diverted page's raw text rather
    # than dropping the page from the output.
    for page_num in ocr_page_nums:
        if page_num not in text_results:
            logger.warning("ocr_unavailable_kept_raw_text", page=page_num)
            text_results[page_num] = PageResult(
                furniture=furniture_inline,
                page_number=page_num,
                text=ocr_fallback_text.get(page_num, ""),
                method="text_extraction",
                layout=layouts.get(page_num),
                # Diverted, then OCR could not run. The reason still stands and
                # the raw text is a fallback, not a decision to trust it.
                divert_reason=divert_reasons.get(page_num, ""),
            )

    ordered_pages = [text_results[i] for i in sorted(text_results.keys())]

    # Coverage backstop: promote the Mistral layout-pass transcription for any
    # page the authoritative read left empty (see `_promote_rival_when_empty`).
    ordered_pages = _settle_rival_available([_promote_rival_when_empty(p) for p in ordered_pages])

    return ordered_pages


async def _ocr_page_with_context(
    pdf_bytes: bytes,
    page_number: int,
    total_pages: int,
    client: LLMClient,
    doc_context: str,
    ocr_model: str | None = None,
    ocr_dpi: int = 300,
    image_bytes: bytes | None = None,
    render_if_missing: bool = True,
    page_text_layer: str = "",
) -> str:
    """OCR one page, anchored on the text layer the divert gate rejected.

    `image_bytes` is the page the caller's render window already rasterised;
    rendering again would re-parse the whole PDF per page. A bare image lets the
    model invent text where the scan is ambiguous, and a partial transcription to
    compare against does not.
    """
    # The fan-out has already tried a window covering this page. Rendering
    # again per page restores the parse-per-page this exists to remove, and a
    # missing poppler would do it once per page of the document.
    if image_bytes is None and render_if_missing:
        rendered = await _render_window(pdf_bytes, page_number, page_number, ocr_dpi)
        image_bytes = rendered.get(page_number)
    if image_bytes is None:
        image_bytes = _extract_first_image(pdf_bytes, page_number)
    if not image_bytes:
        # Returning "" here made an unrenderable page a successful empty one:
        # `combine_page_texts` drops it and the run reports success, which is
        # the silent loss this module keeps having to relearn.
        raise RuntimeError(f"page {page_number} could not be rendered for OCR")

    # Mistral OCR path is handled by the full-PDF fast path in
    # extract_text_from_pdf. This per-page fallback only runs for the chat
    # vision route (Gemini and friends).
    anchor = (page_text_layer or "").strip()[:ANCHOR_CHAR_LIMIT]
    anchor_block = (
        f"\n\nText layer extracted from this page. It was rejected as unreliable "
        f"and may be partial or garbled, so transcribe the image and use this "
        f"only to disambiguate:\n{anchor}"
        if anchor
        else ""
    )
    prompt = f"""Context:
{doc_context}

This is page {page_number} of {total_pages}. Extract all text from this page.{anchor_block}"""

    text = await client.vision(
        prompt=prompt,
        images=[image_bytes],
        system=OCR_SYSTEM_PROMPT,
        model=ocr_model,
    )
    return VisionText(_clean_ocr_output(text), getattr(text, "finish_reason", ""))


def _extract_first_image(pdf_bytes: bytes, page_number: int) -> bytes | None:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page = reader.pages[page_number - 1]
        for image_obj in page.images:
            img = image_obj.image
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return None
    except Exception:
        return None


# XML 1.0 forbids C0 control chars except tab/newline/CR; vision OCR
# occasionally emits NUL or other control bytes that crash the lxml parse.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control_chars(text: str) -> str:
    """Remove XML-illegal control characters from extracted/OCR'd text."""
    return _CONTROL_CHARS_RE.sub("", text)


# OCR models emit markdown the gazette never carried: heading prefixes, bare
# marker lines, LaTeX footnote superscripts. Left in they reach the AKN preface
# verbatim and become placeholder garbage in translation. Patterns never match
# across a page join, so both combiners apply them per page and agree byte for
# byte.
_MD_HEADING_PREFIX_RE = re.compile(r"^[\t ]*#{1,6}[\t ]*", re.M)
_MD_BARE_MARKER_LINE_RE = re.compile(r"^[#*\-=\t ]{1,10}$\n?", re.M)
_LATEX_SUPERSCRIPT_RE = re.compile(r"\$\^?\{([^{}$\n]{0,12})\}\$")


def _strip_markdown_furniture(text: str) -> str:
    text = _MD_HEADING_PREFIX_RE.sub("", text)
    text = _LATEX_SUPERSCRIPT_RE.sub("", text)
    return _MD_BARE_MARKER_LINE_RE.sub("", text)


def clean_page_text(text: str, furniture: Sequence[str] = ()) -> str:
    """Canonical per-page cleaning, shared by both combiners so spans match."""
    return _strip_markdown_furniture(_scrub_furniture_inline(_strip_control_chars(text), furniture))


def low_text_pages(pages: list[PageResult], threshold: int = 40) -> list[int]:
    """Pages whose cleaned OCR text is empty or near-empty.

    A gazette page carrying an annex table often OCRs to nothing, and the loss is
    invisible downstream because the combined text simply lacks the material.
    Callers warn so the gap is one log search away.
    """
    return [
        p.page_number
        for p in pages
        if len(clean_page_text(p.text or "", p.furniture).strip()) < threshold
    ]


def page_verdict(page: PageResult, *, pack: ScriptPack | None = None) -> PageVerdict:
    """The deterministic quality verdict for one page, from what the read left on it.

    One scorer call site, so provenance, combined text and persisted metrics agree on
    which page is degraded. `pack` is the jurisdiction's declared script, so drift is
    measured against what the page should be: detecting it from the output would
    score a wholly Latin hallucination on an Arabic page as pure. Detection is the
    fallback only where no script is declared.
    """
    from codify.pipeline.enrich.scripts import detect_pack

    text = clean_page_text(page.text, page.furniture)
    return score_page(
        text,
        ink=page.ink,
        rival_text=page.rival_text,
        divergence_score=page.divergence,
        pack=pack if pack is not None else detect_pack(text),
    )


DEGRADED_MARKER_RE = re.compile(r"^⟦page (\d+) unreadable⟧$", re.MULTILINE)


# The editorial remark the structurer leaves where an unreadable page stood.
UNREADABLE_REMARK = "[Page {page} of the source could not be read]"
UNREADABLE_REMARK_RE = re.compile(r"\[Page (\d+) of the source could not be read\]")


def degraded_page_marker(page_number: int) -> str:
    """The line that stands in for a page that read to nothing. Visible in the
    combined text where a silent drop hid the loss; the structurer turns it into
    an editorial remark in the AKN."""
    return f"⟦page {page_number} unreadable⟧"


def diverted_to_nothing(page: PageResult) -> bool:
    """A page routed to OCR that came back empty and had ink to read: a failed
    read, not a blank leaf. Ink is absent on the text lane, where an empty divert
    is taken at face value. The same `BLANK_INK` cutoff the scorer uses, so a page
    is blank in one place and blank in all."""
    if not page.divert_reason or clean_page_text(page.text, page.furniture).strip():
        return False
    if page.divert_reason in _DISTRUSTED_LAYER_REASONS:
        # The page had a text layer over the length floor, so an empty read is a
        # failed read however little ink the render found.
        return True
    return page.ink is None or page.ink >= BLANK_INK


def unreadable_reason(page: PageResult) -> str:
    """Why a page with content on it came back with no text, or "" if it did not:
    the refusing finish reason, else `empty_read` for an inked divert read empty."""
    if clean_page_text(page.text, page.furniture).strip():
        return ""
    if content_filtered(page.finish_reason):
        return page.finish_reason
    return "empty_read" if diverted_to_nothing(page) else ""


def unreadable_pages(pages: Sequence[PageResult]) -> dict[int, str]:
    """Page number to `unreadable_reason`, for every page that has one."""
    return {p.page_number: r for p in pages if (r := unreadable_reason(p))}


def _page_body_for_combine(page: PageResult) -> str | None:
    """One page's contribution to the combined text, or None to omit it.

    An unreadable page leaves a marker, so the loss the coverage gate could not
    see is at least on the page. A genuinely blank page is omitted.
    """
    cleaned = clean_page_text(page.text, page.furniture)
    if cleaned.strip():
        return cleaned
    if unreadable_reason(page):
        return degraded_page_marker(page.page_number)
    return None


_PAGE_SEP = "\n\n"


class PageSpan(BaseModel):
    """One page's `[start, end)` in the combined text."""

    page: int
    method: str
    start: int
    end: int


def combine_page_texts_with_spans(pages: list[PageResult]) -> tuple[str, list[PageSpan]]:
    """The combined text plus where each page landed in it.

    Text and offsets come from one assembly so they cannot drift apart."""
    low = low_text_pages(pages)
    if low:
        logger.warning("ocr_page_low_text", pages=low, page_count=len(pages))
    bodies: list[str] = []
    spans: list[PageSpan] = []
    pos = 0
    for page in pages:
        body = _page_body_for_combine(page)
        if body is None:
            continue
        if bodies:
            pos += len(_PAGE_SEP)
        spans.append(
            PageSpan(page=page.page_number, method=page.method, start=pos, end=pos + len(body))
        )
        pos += len(body)
        bodies.append(body)
    return _PAGE_SEP.join(bodies), spans


def combine_page_texts(pages: list[PageResult]) -> str:
    return combine_page_texts_with_spans(pages)[0]


def _scrub_block(block: OcrBlock) -> OcrBlock:
    """Block content lands in a jsonb column, which rejects NUL outright."""
    block.content = _strip_control_chars(block.content)
    return block


def _scrub_word(word: WordConfidence) -> WordConfidence:
    """Same reason as the blocks: this text is persisted, not just logged."""
    word.text = _strip_control_chars(word.text)
    return word


def _layout_from_page(page: dict[str, Any], model: str, header: str, footer: str) -> PageLayout:
    """Build the layout record from one raw OCR page object.

    Typed fields are the ones observed on a live response. `tables` and `images` were
    empty there, so they stay in `raw` rather than given a shape nobody has seen.
    """
    scores = page.get("confidence_scores") or {}
    dims = page.get("dimensions") or {}
    known = {
        "markdown",
        "header",
        "footer",
        "blocks",
        "dimensions",
        "confidence_scores",
        "hyperlinks",
        "index",
    }
    return PageLayout(
        engine="mistral_ocr",
        model=model,
        header=header,
        footer=footer,
        blocks=[
            _scrub_block(OcrBlock.model_validate(b))
            for b in (page.get("blocks") or [])
            if isinstance(b, dict)
        ],
        dimensions=PageDimensions.model_validate(dims) if dims else None,
        page_confidence=scores.get("average_page_confidence_score"),
        min_word_confidence=scores.get("minimum_page_confidence_score"),
        words=[
            _scrub_word(WordConfidence.model_validate(w))
            for w in (scores.get("word_confidence_scores") or [])
            if isinstance(w, dict)
        ],
        hyperlinks=[
            _strip_control_chars(h) for h in (page.get("hyperlinks") or []) if isinstance(h, str)
        ],
        # In memory only, and excluded from the persisted dump: the endpoint may
        # put anything here, including strings Postgres would refuse.
        raw={k: v for k, v in page.items() if k not in known},
    )
