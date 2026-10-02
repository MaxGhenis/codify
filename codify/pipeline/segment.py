"""Tell apart the acts one source holds, before any later stage assumes it holds one.

Deterministic, and every rule comes from jurisdiction config. A boundary is a declared
act heading that one independent signal agrees with: a closing phrase before it, a
numbering restart after it, a contents entry naming it on its page, or a page start.
Known non-boundaries veto a heading: an adopted or attached text, a repeat of the open
act's own heading, a quotation, a missing enacting formula. Where the evidence
disagrees the region is held for review, never folded into a neighbour.

A bound volume of gazette issues is read a page at a time by `segment_volume`, which
applies the same rule to issue headings and then segments each issue into acts.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from codify.jurisdictions import JurisdictionConfig, SegmentationConfig, heading_line_pattern
from codify.lang import normalise_digits
from codify.pipeline.enrich.adoption import _pattern as _adoption_pattern
from codify.pipeline.enrich.anchors import (
    _kind_from_match,
    _normalise_number,
    build_anchor_regex,
)
from codify.pipeline.enrich.closing import _phrase_pattern
from codify.pipeline.enrich.ocr import PageSpan

Outcome = Literal["single", "decided", "abstained"]
Status = Literal[
    "opening",
    "matched",
    "corroborated",
    "uncorroborated",
    "vetoed",
    "heading_only",
    "contents_only",
    "page_mismatch",
    "label_mismatch",
]

# How far before a heading an adoption or attachment marker claims it.
VETO_WINDOW = 400
# A restart needs the sequence it leaves to have run this far.
RESTART_FROM = 3
# Lines at a page's top and bottom searched for a printed page number.
_EDGE_LINES = 3
# Separates pages when a volume's pages are joined into one issue text.
PAGE_SEPARATOR = "\n\n"

_LINE_END_NUMBER_RE = re.compile(r"(?m)(\d+)[ \t]*$")
# Longer than this between an entry's heading and its page, and it is body text.
_ENTRY_MAX_CHARS = 1000
_NOT_WORD_RE = re.compile(r"[\W_]+")
_QUOTE_OPENERS = frozenset("\u201c\u00ab\u2039\u300c\u300e")
_QUOTE_CLOSERS = frozenset("\u201d\u00bb\u203a\u300d\u300f")


@dataclass(frozen=True)
class Segment:
    """One act's `[start, end)` in the text it was cut from."""

    start: int
    end: int
    first_page: int | None
    last_page: int | None
    heading: str
    key: str
    signals: tuple[str, ...]
    text: str = field(repr=False)


@dataclass(frozen=True)
class HeldSpan:
    """A region the evidence could not settle, kept whole for review."""

    start: int
    end: int
    first_page: int | None
    last_page: int | None
    reason: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class ReconciliationRow:
    """One contents entry or one heading, and what became of it."""

    source: Literal["contents", "heading"]
    label: str
    key: str
    status: Status
    printed_page: int | None = None
    pdf_page: int | None = None
    offset: int | None = None
    signals: tuple[str, ...] = ()
    veto: str = ""

    def describe(self) -> str:
        """The row in plain words, for a run report."""
        where = f"PDF page {self.pdf_page}" if self.pdf_page is not None else "no page found"
        listed = f" (listed at page {self.printed_page})" if self.printed_page is not None else ""
        reason = {
            "opening": "opens the source",
            "matched": "agrees with the contents",
            "corroborated": "agreed by " + ", ".join(self.signals),
            "uncorroborated": "no signal agrees it opens an act",
            "vetoed": f"not an act boundary: {self.veto}",
            "heading_only": "found in the text but not in the contents"
            + ("" if self.signals else ", and no signal agrees: read as a citation"),
            "contents_only": "listed in the contents but no heading found",
            "page_mismatch": "listed in the contents on a different page",
            "label_mismatch": "a different act stands where the contents places this one",
        }[self.status]
        return f"{self.source} {self.label!r}{listed}, {where}: {reason}"


@dataclass(frozen=True)
class Segmentation:
    outcome: Outcome
    segments: tuple[Segment, ...]
    held: tuple[HeldSpan, ...] = ()
    reconciliation: tuple[ReconciliationRow, ...] = ()
    # Masthead and contents ahead of the first act: neither an act nor in doubt.
    front_matter: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Heading:
    start: int
    end: int
    # Where the pattern's match ends: a contents entry's page follows it.
    match_end: int
    label: str
    key: str
    page: int | None


@dataclass(frozen=True)
class _Entry:
    label: str
    key: str
    offset: int
    printed_page: int | None
    pdf_page: int | None


class _Pages:
    """Offsets to pages, and pages to the numbers printed on them."""

    def __init__(
        self,
        text: str,
        spans: Sequence[PageSpan],
        furniture: Mapping[int, str],
        printed: re.Pattern[str] | None,
    ) -> None:
        self.spans = sorted(spans, key=lambda s: s.start)
        self.starts = [s.start for s in self.spans]
        self.printed: dict[int, int] = {}
        for span in self.spans:
            number = _printed_number(
                text[span.start : span.end], furniture.get(span.page, ""), printed
            )
            if number is not None:
                self.printed[span.page] = number
        self.by_printed: dict[int, int] = {}
        for page, number in sorted(self.printed.items()):
            self.by_printed.setdefault(number, page)
        self.pattern = printed

    def page_at(self, offset: int) -> PageSpan | None:
        index = bisect_right(self.starts, offset) - 1
        if index < 0:
            return None
        span = self.spans[index]
        return span if offset <= span.end else None

    def start_of(self, page: int) -> int | None:
        return next((s.start for s in self.spans if s.page == page), None)

    def pdf_page(self, listed: int) -> int | None:
        """The PDF page a contents reference names: printed numbers where declared."""
        if self.pattern is None:
            return listed if any(s.page == listed for s in self.spans) else None
        return self.by_printed.get(listed)


def _printed_number(page_text: str, furniture: str, pattern: re.Pattern[str] | None) -> int | None:
    """The page number the publisher printed: furniture first, then the page's edges."""
    if pattern is None:
        return None
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    edges = [*furniture.splitlines(), *lines[:_EDGE_LINES], *lines[-_EDGE_LINES:]]
    for line in edges:
        match = pattern.search(line.strip())
        # Alternatives may each carry a group; the one that took part holds the number.
        number = next((g for g in match.groups() if g), "") if match else ""
        if normalise_digits(number).isdigit():
            return int(normalise_digits(number))
    return None


def _key(match: re.Match[str]) -> str:
    """What identifies the act a heading names: its named groups, else the whole match."""
    # Space inside a named group is extraction noise ("5 0-19"), never a distinction.
    named = [re.sub(r"\s+", "", v) for v in match.groupdict().values() if v]
    raw = " ".join(named) if named else match.group(0)
    return _NOT_WORD_RE.sub(" ", normalise_digits(raw).casefold()).strip()


def _heading_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    # One pattern each: two may both name a `number` group.
    return [heading_line_pattern(p) for p in patterns]


def _headings(text: str, patterns: Sequence[re.Pattern[str]], pages: _Pages) -> list[_Heading]:
    found: dict[int, _Heading] = {}
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = text.rfind("\n", 0, match.start()) + 1
            if start in found:
                continue
            end = text.find("\n", match.end())
            end = len(text) if end < 0 else end
            page = pages.page_at(start)
            found[start] = _Heading(
                start=start,
                end=end,
                match_end=match.end(),
                label=text[start:end].strip(),
                key=_key(match),
                page=page.page if page else None,
            )
    return [found[k] for k in sorted(found)]


def _contents(
    text: str,
    rules: SegmentationConfig,
    headings: Sequence[_Heading],
    markers: Sequence[_Marker],
    pages: _Pages,
) -> tuple[list[tuple[int, int]], list[_Entry]]:
    """Contents listings and their entries. A listing ends at the first heading a
    numbered provision follows, the body's own, or where the earliest page it names
    begins. An entry whose page cannot be found stays in the listing, unplaced."""
    words = [w for w in rules.contents_keywords if w.strip()]
    if not words:
        return [], []
    keyword = re.compile(
        r"(?im)^[ \t]*(?:" + "|".join(map(re.escape, words)) + r")[ \t]*[:.]?[ \t]*$"
    )
    blocks: list[tuple[int, int]] = []
    entries: list[_Entry] = []
    for found in keyword.finditer(text):
        if _within(found.start(), blocks):
            continue
        end = len(text)
        # The entry being read, and the last heading inside it.
        current: _Heading | None = None
        last: _Heading | None = None
        for heading in [*(h for h in headings if h.start > found.end()), None]:
            stop = end if heading is None else min(heading.start, end)
            if last is not None and any(last.match_end <= m.offset < stop for m in markers):
                end = last.start
                break
            if current is not None:
                page = _reference(text, current.match_end, stop, pages)
                near = stop - current.match_end <= _ENTRY_MAX_CHARS
                if page is None and heading is not None and heading.start < end and near:
                    # A citation wrapped onto its own line, inside the entry.
                    last = heading
                    continue
                if page is not None:
                    entries.append(_Entry(current.label, current.key, current.start, *page))
                    begins = pages.start_of(page[1])
                    if begins is not None and begins > found.end():
                        end = min(end, begins)
            if heading is None or heading.start >= end:
                break
            current = last = heading
        blocks.append((found.start(), end))
    return blocks, entries


def _reference(text: str, start: int, stop: int, pages: _Pages) -> tuple[int, int] | None:
    """The printed and PDF page an entry names: its last line-end number that is a
    page of this source."""
    stop = min(stop, start + _ENTRY_MAX_CHARS)
    for number in reversed(list(_LINE_END_NUMBER_RE.finditer(text, start, stop))):
        printed = int(normalise_digits(number.group(1)))
        pdf_page = pages.pdf_page(printed)
        if pdf_page is not None:
            return printed, pdf_page
    return None


@dataclass(frozen=True)
class _Marker:
    offset: int
    number: str


def _markers(text: str, config: JurisdictionConfig, doctype: str) -> list[_Marker]:
    """Basic-unit markers in reading order, for numbering restarts."""
    from codify.pipeline.enrich.structure import basic_unit_kind

    kind = basic_unit_kind(config, doctype)
    if kind is None:
        return []
    regex = build_anchor_regex(config, doctype)
    markers: list[_Marker] = []
    for match in regex.finditer(text):
        at = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        if _kind_from_match(match) != kind or _quoted(text, at):
            continue
        markers.append(_Marker(at, _normalise_number(match.groupdict().get("num"))))
    return markers


def _closing_regex(config: JurisdictionConfig) -> re.Pattern[str] | None:
    phrases = [
        *config.closing_phrases,
        *(p for era in config.legal_eras for p in era.closing_phrases),
    ]
    words = [p for p in phrases if p.strip()]
    return re.compile("|".join(map(_phrase_pattern, words))) if words else None


def _phrases_regex(phrases: Iterable[str]) -> re.Pattern[str] | None:
    words = [p for p in phrases if p.strip()]
    if not words:
        return None
    return re.compile("|".join(r"\s+".join(map(re.escape, p.split())) for p in words))


def _caption_regex(config: JurisdictionConfig) -> re.Pattern[str] | None:
    """Caption lines as the attachment scan reads them: a prefix caption opens its
    line, any other is the whole line, numbered or followed by a colon at most."""
    declared = [a for a in config.attachments if a.caption.strip()]
    exact = "|".join(re.escape(a.caption) for a in declared if not a.prefix)
    opener = "|".join(re.escape(a.caption) for a in declared if a.prefix)
    alts = [
        *(
            [rf"(?:{exact})(?:[ \t]+(?:[IVXLCDM]+|\d+))?[ \t]*(?:[:：][ \t]*(?=\S)|\r?$)"]
            if exact
            else []
        ),
        *([rf"(?:{opener})(?!\w)"] if opener else []),
    ]
    return re.compile(r"(?m)^[ \t]*(?:" + "|".join(alts) + ")") if alts else None


class _Rules:
    """The jurisdiction's vocabularies, compiled once per call."""

    def __init__(self, config: JurisdictionConfig) -> None:
        self.closing = _closing_regex(config)
        self.adoption = _adoption_pattern(tuple(config.adoption_markers))
        self.caption = _caption_regex(config)
        self.enacting = _phrases_regex(config.enacting_formula_markers)


def _page_start(text: str, heading: _Heading, pages: _Pages) -> bool:
    span = pages.page_at(heading.start)
    return span is not None and _bare(text[span.start : heading.start], pages.pattern)


def _bare(above: str, printed: re.Pattern[str] | None) -> bool:
    """Only blank lines, rules or the printed page number stand above a heading."""
    for line in above.splitlines():
        line = line.strip()
        if not line or not any(ch.isalnum() for ch in line):
            continue
        if printed is not None and printed.search(line):
            continue
        return False
    return True


def _quoted(text: str, at: int) -> bool:
    """A quotation opened shortly before `at` and not yet closed. Local, so one stray
    mark cannot claim pages; straight quotes are punctuation, as elsewhere."""
    for ch in reversed(text[max(0, at - VETO_WINDOW) : at]):
        if ch in _QUOTE_CLOSERS:
            return False
        if ch in _QUOTE_OPENERS:
            return True
    return False


def _within(offset: int, blocks: Sequence[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in blocks)


def _number(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def segment(
    text: str,
    spans: Sequence[PageSpan],
    *,
    config: JurisdictionConfig,
    furniture: Mapping[int, str] | None = None,
    doctype: str | None = None,
) -> Segmentation:
    """Split `text` into the acts it holds, or say why it cannot.

    `spans` place each page in `text`, and `furniture` carries each page's header
    and footer where extraction lifted them out. Offsets in the result index `text`.
    """
    rules = config.segmentation
    if rules is None or not rules.act_heading_patterns:
        return _single(text, spans)
    printed = re.compile(rules.printed_page_pattern) if rules.printed_page_pattern else None
    pages = _Pages(text, spans, furniture or {}, printed)
    headings = _headings(text, _heading_patterns(rules.act_heading_patterns), pages)
    markers = _markers(text, config, doctype or config.default_document_class)
    blocks, entries = _contents(text, rules, headings, markers, pages)
    candidates = [h for h in headings if not _within(h.start, blocks)]
    if not candidates and not entries:
        return _single(text, spans)
    return _decide(text, pages, candidates, entries, markers, _Rules(config))


def _decide(
    text: str,
    pages: _Pages,
    candidates: list[_Heading],
    entries: list[_Entry],
    markers: list[_Marker],
    rules: _Rules,
) -> Segmentation:
    # Where a contents listing exists, only a heading it lists can open the source.
    listed = {e.key for e in entries}
    first = next((c for c in candidates if not entries or c.key in listed), None)
    opening = first if first and not any(m.offset < first.start for m in markers) else None
    rows: list[ReconciliationRow] = []
    decided: list[tuple[_Heading, tuple[str, ...]]] = []
    # Where the evidence disagreed, and why; None places a doubt nowhere, so everywhere.
    doubts: list[tuple[int | None, str]] = []
    unmatched = list(entries)
    # An entry another heading names is that heading's, never a mismatch for this one.
    named = {c.key for c in candidates}
    open_heading: _Heading | None = None
    previous = 0
    for index, heading in enumerate(candidates):
        following = candidates[index + 1].start if index + 1 < len(candidates) else len(text)
        is_opening = heading is opening
        veto = "" if is_opening else _veto(text, heading, open_heading, markers, rules)
        if veto:
            rows.append(_row(heading, "vetoed", veto=veto))
            continue
        signals = (
            ()
            if is_opening
            else _signals(text, heading, previous, following, markers, pages, rules)
        )
        status: Status
        if entries:
            status, entry = _against_contents(heading, entries, unmatched, named)
            if entry is not None:
                unmatched.remove(entry)
                rows.append(_entry_row(entry, status))
            if status == "matched":
                signals = (*signals, "contents")
        else:
            status = "opening" if is_opening else "corroborated" if signals else "uncorroborated"
        row = _row(heading, status, signals)
        rows.append(row)
        if status in ("matched", "corroborated", "opening"):
            if not is_opening:
                decided.append((heading, signals))
            open_heading, previous = heading, heading.start
        elif status != "heading_only" or signals:
            doubts.append((heading.start, row.describe()))
        if is_opening:
            open_heading, previous = heading, heading.start
    for entry in unmatched:
        row = _entry_row(entry, "contents_only")
        rows.append(row)
        begins = pages.start_of(entry.pdf_page) if entry.pdf_page is not None else None
        doubts.append((begins, row.describe()))
    if not decided and not doubts:
        return _single(text, pages.spans, tuple(rows))
    return _assemble(text, pages, opening, decided, doubts, tuple(rows))


def _veto(
    text: str,
    heading: _Heading,
    open_heading: _Heading | None,
    markers: list[_Marker],
    rules: _Rules,
) -> str:
    if _quoted(text, heading.start):
        return "inside a quotation"
    if open_heading is not None and heading.key == open_heading.key:
        return "repeats the open act's own heading"
    window = text[max(0, heading.start - VETO_WINDOW) : heading.start]
    if rules.adoption is not None and rules.adoption.search(window):
        return "follows a declaration adopting another text"
    if rules.caption is not None and rules.caption.search(window):
        return "follows an attachment caption"
    if rules.enacting is not None and open_heading is not None:
        opened = _opening_region(text, open_heading.start, markers)
        here = _opening_region(text, heading.start, markers)
        if rules.enacting.search(opened) and not rules.enacting.search(here):
            return "carries no enacting formula where the act before it does"
    return ""


def _opening_region(text: str, start: int, markers: list[_Marker]) -> str:
    """From a heading to its first numbered provision: where a formula would stand."""
    first = next((m.offset for m in markers if m.offset > start), len(text))
    return text[start:first]


def _signals(
    text: str,
    heading: _Heading,
    previous: int,
    following: int,
    markers: list[_Marker],
    pages: _Pages,
    rules: _Rules,
) -> tuple[str, ...]:
    signals: list[str] = []
    before = [m for m in markers if previous <= m.offset < heading.start]
    if rules.closing is not None:
        lo = before[-1].offset if before else previous
        for match in rules.closing.finditer(text, lo, heading.start):
            if not _quoted(text, match.start()):
                signals.append("closing")
                break
    after = next((m for m in markers if heading.start <= m.offset < following), None)
    reached = [n for m in before if (n := _number(m.number)) is not None]
    if after is not None and after.number == "1" and reached and max(reached) >= RESTART_FROM:
        signals.append("restart")
    if _page_start(text, heading, pages):
        signals.append("page_start")
    return tuple(signals)


def _against_contents(
    heading: _Heading, entries: list[_Entry], unmatched: list[_Entry], named: set[str]
) -> tuple[Status, _Entry | None]:
    same = next((e for e in unmatched if e.key == heading.key), None)
    if same is not None:
        return ("matched" if same.pdf_page == heading.page else "page_mismatch"), same
    if any(e.key == heading.key for e in entries):
        # Listed once and already claimed: a second heading for one entry.
        return "heading_only", None
    placed = next(
        (
            e
            for e in unmatched
            if e.pdf_page is not None and e.pdf_page == heading.page and e.key not in named
        ),
        None,
    )
    if placed is not None:
        return "label_mismatch", placed
    return "heading_only", None


def _row(
    heading: _Heading, status: Status, signals: tuple[str, ...] = (), veto: str = ""
) -> ReconciliationRow:
    return ReconciliationRow(
        source="heading",
        label=heading.label,
        key=heading.key,
        status=status,
        pdf_page=heading.page,
        offset=heading.start,
        signals=signals,
        veto=veto,
    )


def _entry_row(entry: _Entry, status: Status) -> ReconciliationRow:
    return ReconciliationRow(
        source="contents",
        label=entry.label,
        key=entry.key,
        status=status,
        printed_page=entry.printed_page,
        pdf_page=entry.pdf_page,
        offset=entry.offset,
    )


def _assemble(
    text: str,
    pages: _Pages,
    opening: _Heading | None,
    decided: list[tuple[_Heading, tuple[str, ...]]],
    doubts: list[tuple[int | None, str]],
    rows: tuple[ReconciliationRow, ...],
) -> Segmentation:
    first = opening.start if opening is not None else 0
    cuts: list[tuple[int, _Heading | None, tuple[str, ...]]] = [(first, opening, ())]
    # An act opening a page takes the page's furniture with it.
    cuts += [(_cut(pages, h, signals), h, signals) for h, signals in decided]
    segments: list[Segment] = []
    held: list[HeldSpan] = []
    # A doubt ahead of the opening is not front matter: hold that region instead.
    early = [why for at, why in doubts if at is not None and at < first]
    if early:
        first_page, last_page = _page_range(pages, 0, first)
        held.append(HeldSpan(0, first, first_page, last_page, "; ".join(early), text[:first]))
    for index, (start, heading, signals) in enumerate(cuts):
        end = cuts[index + 1][0] if index + 1 < len(cuts) else len(text)
        first_page, last_page = _page_range(pages, start, end)
        here = [why for at, why in doubts if at is None or start <= at < end]
        if here:
            held.append(
                HeldSpan(start, end, first_page, last_page, "; ".join(here), text[start:end])
            )
            continue
        segments.append(
            Segment(
                start=start,
                end=end,
                first_page=first_page,
                last_page=last_page,
                heading=heading.label if heading else "",
                key=heading.key if heading else "",
                signals=signals,
                text=text[start:end],
            )
        )
    outcome: Outcome = "abstained" if held else "decided"
    front = (0, first) if first and not early else None
    return Segmentation(outcome, tuple(segments), tuple(held), rows, front)


def _cut(pages: _Pages, heading: _Heading, signals: tuple[str, ...]) -> int:
    span = pages.page_at(heading.start) if "page_start" in signals else None
    return span.start if span is not None else heading.start


def _page_range(pages: _Pages, start: int, end: int) -> tuple[int | None, int | None]:
    """First and last page a region touches; a page separator belongs to the page before."""
    first = bisect_right(pages.starts, start) - 1
    last = bisect_right(pages.starts, max(start, end - 1)) - 1
    if first < 0 or last < 0:
        return None, None
    return pages.spans[first].page, pages.spans[last].page


def _single(
    text: str, spans: Sequence[PageSpan], rows: tuple[ReconciliationRow, ...] = ()
) -> Segmentation:
    """One act: the source unchanged, whatever headings were read and set aside."""
    ordered = sorted(spans, key=lambda s: s.start)
    return Segmentation(
        outcome="single",
        segments=(
            Segment(
                start=0,
                end=len(text),
                first_page=ordered[0].page if ordered else None,
                last_page=ordered[-1].page if ordered else None,
                heading="",
                key="",
                signals=(),
                text=text,
            ),
        ),
        reconciliation=rows,
    )


@dataclass(frozen=True)
class SourcePage:
    """One page of a volume as extraction gives it."""

    page: int
    text: str
    furniture: str = ""


@dataclass(frozen=True)
class Issue:
    """One gazette issue cut from a volume, and the acts it holds."""

    heading: str
    key: str
    first_page: int
    last_page: int
    status: Status
    signals: tuple[str, ...]
    segmentation: Segmentation


@dataclass
class _OpenIssue:
    heading: str = ""
    key: str = ""
    status: Status = "opening"
    signals: tuple[str, ...] = ()
    pages: list[SourcePage] = field(default_factory=list)
    doubts: list[str] = field(default_factory=list)


def segment_volume(
    pages: Iterable[SourcePage],
    *,
    config: JurisdictionConfig,
    doctype: str | None = None,
) -> Iterator[Issue]:
    """Issues in reading order, each segmented into acts, holding one issue's pages
    at a time. An issue boundary follows `segment`'s rule: a declared issue heading
    and one agreeing signal, a page start, a closing phrase on the page before, or
    printed page numbers starting again."""
    rules = config.segmentation
    patterns = _heading_patterns(rules.issue_heading_patterns) if rules else []
    printed = (
        re.compile(rules.printed_page_pattern) if rules and rules.printed_page_pattern else None
    )
    closing = _closing_regex(config)
    issue = _OpenIssue()
    waiting: tuple[SourcePage, re.Match[str]] | None = None
    before = ""
    for page in pages:
        if waiting is not None:
            issue = yield from _settle(issue, *waiting, page, printed, closing, config, doctype)
            waiting = None
        match = _issue_heading(page.text, patterns, issue.key if issue.pages else None, before)
        before = page.text
        if match is not None and issue.pages:
            # Decided once the next page is read: a restart may first show there.
            waiting = (page, match)
            continue
        if match is not None:
            issue.heading, issue.key = _line(page.text, match), _key(match)
        issue.pages.append(page)
    if waiting is not None:
        issue = yield from _settle(issue, *waiting, None, printed, closing, config, doctype)
    if issue.pages:
        yield _close(issue, config, doctype)


def _settle(
    issue: _OpenIssue,
    page: SourcePage,
    match: re.Match[str],
    after: SourcePage | None,
    printed: re.Pattern[str] | None,
    closing: re.Pattern[str] | None,
    config: JurisdictionConfig,
    doctype: str | None,
) -> Generator[Issue, None, _OpenIssue]:
    """Close the open issue at `page` if its heading is agreed, else absorb it."""
    heading, key = _line(page.text, match), _key(match)
    issue_pages = issue.pages
    signals: list[str] = []
    if _bare(page.text[: page.text.rfind("\n", 0, match.start()) + 1], printed):
        signals.append("page_start")
    if _closes(issue_pages[-1].text, closing, config, doctype):
        signals.append("closing")
    seen = [
        n for p in issue_pages if (n := _printed_number(p.text, p.furniture, printed)) is not None
    ]
    fresh = [
        n
        for p in (page, after)
        if p is not None and (n := _printed_number(p.text, p.furniture, printed)) is not None
    ]
    if seen and fresh and fresh[0] < max(seen):
        signals.append("restart")
    if signals:
        yield _close(issue, config, doctype)
        issue = _OpenIssue(heading=heading, key=key, status="corroborated", signals=tuple(signals))
    else:
        issue.doubts.append(f"issue heading {heading!r} on PDF page {page.page}: no signal agrees")
    issue.pages.append(page)
    return issue


def _close(issue: _OpenIssue, config: JurisdictionConfig, doctype: str | None) -> Issue:
    bodies: list[str] = []
    spans: list[PageSpan] = []
    position = 0
    for page in issue.pages:
        if bodies:
            position += len(PAGE_SEPARATOR)
        spans.append(
            PageSpan(page=page.page, method="", start=position, end=position + len(page.text))
        )
        position += len(page.text)
        bodies.append(page.text)
    text = PAGE_SEPARATOR.join(bodies)
    furniture = {p.page: p.furniture for p in issue.pages if p.furniture}
    if issue.doubts:
        first, last = issue.pages[0].page, issue.pages[-1].page
        whole = HeldSpan(0, len(text), first, last, "; ".join(issue.doubts), text)
        result = Segmentation("abstained", (), (whole,))
    else:
        result = segment(text, spans, config=config, furniture=furniture, doctype=doctype)
    return Issue(
        heading=issue.heading,
        key=issue.key,
        first_page=issue.pages[0].page,
        last_page=issue.pages[-1].page,
        status="uncorroborated" if issue.doubts else issue.status,
        signals=issue.signals,
        segmentation=result,
    )


def _issue_heading(
    text: str, patterns: Sequence[re.Pattern[str]], open_key: str | None, before: str
) -> re.Match[str] | None:
    """The page's first issue heading that is neither the open issue's running head
    nor quoted, a quotation opened on the page before included."""
    carried = before[-VETO_WINDOW:] + PAGE_SEPARATOR
    joined = carried + text
    found = sorted((m for p in patterns for m in p.finditer(text)), key=lambda m: m.start())
    return next(
        (m for m in found if _key(m) != open_key and not _quoted(joined, len(carried) + m.start())),
        None,
    )


def _closes(
    text: str, closing: re.Pattern[str] | None, config: JurisdictionConfig, doctype: str | None
) -> bool:
    """An unquoted closing after the page's last numbered provision, as for acts."""
    if closing is None:
        return False
    markers = _markers(text, config, doctype or config.default_document_class)
    lo = markers[-1].offset if markers else 0
    return any(not _quoted(text, m.start()) for m in closing.finditer(text, lo))


def _line(text: str, match: re.Match[str]) -> str:
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    return text[start : len(text) if end < 0 else end].strip()
