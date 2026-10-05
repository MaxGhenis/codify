"""Where each region of a segmented source starts and ends, as a page and an offset
into that page's text, so a split survives being stored and the source re-read.

A cut's offset indexes the page's text as extraction combined it. After a re-read
the text moves, so a cut is found again by its marker line, the offset only
breaking ties; a marker no longer on its page is lost, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from codify.pipeline.enrich.ocr import PageSpan
from codify.pipeline.segment import PAGE_SEPARATOR, Segmentation

SpanKind = Literal["act", "held", "skipped", "front_matter"]

# A marker is the cut's first line, bounded so a run-on line stays a key.
_MARKER_CHARS = 200
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpanCut:
    """One region to record: where it starts and ends, as a page and an offset
    into that page's text, and the marker line a re-read finds the start by."""

    kind: SpanKind
    first_page: int
    last_page: int
    start_offset: int
    end_offset: int | None
    start_marker: str = ""
    issue: str = ""
    heading: str = ""
    act_key: str = ""
    signals: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""


def _page(spans: Sequence[PageSpan], offset: int, *, end: bool) -> PageSpan:
    """The page an offset falls on; one in a separator belongs to the page after a
    start and the page before an end."""
    ordered = sorted(spans, key=lambda s: s.start)
    if end:
        return next((s for s in reversed(ordered) if s.start < offset), ordered[0])
    return next((s for s in ordered if offset < s.end or s.start >= offset), ordered[-1])


def _marker(text: str, start: int) -> str:
    line = next((x for x in text[start:].splitlines() if x.strip()), "")
    return _SPACE_RE.sub(" ", line).strip()[:_MARKER_CHARS]


def cuts_from_segmentation(
    result: Segmentation, text: str, spans: Sequence[PageSpan], *, issue: str = ""
) -> list[SpanCut]:
    """The regions of `result` in reading order, tiling `text`: front matter, acts
    and held regions, each placed on the pages `spans` lay out."""
    if not spans:
        raise ValueError("a source with no page spans cannot be cut by page")
    regions: list[tuple[int, int, SpanKind, str, str, tuple[str, ...], str]] = []
    if result.front_matter:
        regions.append((*result.front_matter, "front_matter", "", "", (), ""))
    regions += [(s.start, s.end, "act", s.heading, s.key, s.signals, "") for s in result.segments]
    regions += [(h.start, h.end, "held", "", "", (), h.reason) for h in result.held]
    cuts: list[SpanCut] = []
    for start, end, kind, heading, key, signals, reason in sorted(regions):
        first = _page(spans, start, end=False)
        last = _page(spans, end, end=True)
        cuts.append(
            SpanCut(
                kind=kind,
                first_page=first.page,
                last_page=last.page,
                start_offset=max(0, start - first.start),
                end_offset=None if end >= last.end else end - last.start,
                # The heading names the cut even where the cut takes the page's furniture.
                start_marker=_marker(heading, 0) if heading else _marker(text, start),
                issue=issue,
                heading=heading,
                act_key=key,
                signals=signals,
                reason=reason,
            )
        )
    return cuts


def _normal(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip().casefold()


def locate_cut(page_text: str, marker: str, offset: int) -> int | None:
    """Where the cut `marker` names now falls in `page_text`, nearest `offset`; None
    when the page no longer carries it. Matched with spacing and case folded."""
    if not marker.strip():
        return offset if offset <= len(page_text) else None
    want = _normal(marker)
    found: list[int] = []
    at = 0
    for line in page_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if _normal(stripped).startswith(want):
            found.append(at + len(line) - len(stripped))
        at += len(line)
    if not found:
        return None
    # A cut at the page's start took the furniture above its heading: it stays there.
    return 0 if offset == 0 else min(found, key=lambda x: abs(x - offset))


def span_text(
    pages: Mapping[int, str], first_page: int, last_page: int, start: int, end: int | None
) -> str:
    """A region's text from per-page texts: its pages joined as extraction joins
    them, trimmed at the cuts. A page with no text is skipped, as extraction skips it."""
    parts: list[str] = []
    for page in range(first_page, last_page + 1):
        body = pages.get(page)
        if not body:
            continue
        lo = start if page == first_page else 0
        hi = end if page == last_page and end is not None else len(body)
        parts.append(body[lo:hi])
    return PAGE_SEPARATOR.join(parts)
