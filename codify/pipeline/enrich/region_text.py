"""Region-aware structure text: what the structurer should see.

`combine_page_texts` gives every consumer the full page markdown, and metadata
plus the validator want that superset. The structurer must not weave furniture
into provisions, so this filters the combined text: `header` and `footer` lines
are dropped and `footnote` lines are relocated to the foot as standalone
paragraphs the body-fill emits separably, so the footnote lift catches them
deterministically. Regions decide which lines move; the kept text stays the
authoritative markdown. `aside`, `body` and `unknown` lines are never touched.

A region acts only on the page it was observed on. A heading that straddles a
page break is emitted twice, once as a catchword at the foot of the page and
again as the real heading overleaf; matching the catchword document-wide deleted
both, and with them whole chapters and articles.

One text-level filter serves both ingest lanes (the in-process PDF path and the
durable API structure step), so they cannot drift. Lives apart from `ocr.py`
because it depends on `regions`, which depends on `ocr`; that avoids the cycle.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from codify.pipeline.enrich.ocr import PageSpan
from codify.pipeline.enrich.regions import OVERLAP_FLOOR, Region, overlap, words

# A one-token region ("٣" in a footer) would spuriously match a short body line,
# so only whole multi-word blocks decide a drop or relocation. This also keeps a
# bare inline marker inside a provision from ever matching a footnote block.
_MIN_REGION_WORDS = 2


def _matches(line: str, region_text: str) -> bool:
    """The line IS this block, not merely contained in a longer one. Requiring
    overlap both ways stops a short body line whose words are a subset of a
    running footer from being dropped; that biases towards leaving furniture in,
    which the footnote lift and the aside pass then recover."""
    return (
        overlap(line, region_text) >= OVERLAP_FLOOR and overlap(region_text, line) >= OVERLAP_FLOOR
    )


def _acted(regions: dict[int, list[Region]]) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Furniture and footnote block texts worth acting on, keyed by their page."""
    drop: dict[int, list[str]] = {}
    tail: dict[int, list[str]] = {}
    for page in regions.values():
        for region in page:
            if len(words(region.text)) < _MIN_REGION_WORDS:
                continue
            if region.kind == "furniture":
                drop.setdefault(region.page_number, []).append(region.text)
            elif region.kind == "footnote":
                tail.setdefault(region.page_number, []).append(region.text)
    return drop, tail


def _blocks(by_page: dict[int, list[str]], page: int | None) -> list[str]:
    """This page's blocks; every page's when the line cannot be placed, which is
    the pre-span behaviour and the only reading available without page spans."""
    if page is None:
        return [block for blocks in by_page.values() for block in blocks]
    return by_page.get(page, [])


def _page_of(starts: list[int], pages: list[int], offset: int) -> int | None:
    """The page whose span holds this offset; the nearest page at the edges."""
    if not starts:
        return None
    return pages[max(bisect_right(starts, offset) - 1, 0)]


def combine_text_for_structure(
    text: str,
    regions: dict[int, list[Region]],
    page_spans: Sequence[PageSpan] | None = None,
) -> str:
    """Filter already-combined page text for the structurer: drop header/footer
    lines, relocate footnote lines to the foot. Byte-identical to the input when
    no page carries regions (the born-digital or vision route).

    ``page_spans`` place each line on its page so a region acts only there.
    Without them every region acts on the whole document."""
    drop, tail = _acted(regions)
    # Unreadable-page markers pass through: the structurer lifts them out of the
    # provisions and places each as an editorial remark.
    if not drop and not tail:
        return text

    ordered = sorted(page_spans or (), key=lambda span: span.start)
    starts = [span.start for span in ordered]
    pages = [span.page for span in ordered]

    kept: list[str] = []
    relocated: list[str] = []
    # Offsets index the text as given, which is what the spans were built over.
    offset = 0
    for line in text.split("\n"):
        start, offset = offset, offset + len(line) + 1
        if len(words(line)) < _MIN_REGION_WORDS:
            kept.append(line)  # too short to attribute; never dropped
            continue
        page = _page_of(starts, pages, start)
        if any(_matches(line, block) for block in _blocks(drop, page)):
            continue  # header/footer furniture
        if any(_matches(line, block) for block in _blocks(tail, page)):
            relocated.append(line.strip())  # footnote to the foot
            continue
        kept.append(line)

    body = "\n".join(kept).rstrip()
    if not relocated:
        return body
    # Each footnote as its own blank-delimited paragraph past the last body line,
    # so the trailing window emits it as a standalone <p> the lift can move.
    return body + "\n\n" + "\n\n".join(relocated)
