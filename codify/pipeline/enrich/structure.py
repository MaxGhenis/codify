"""Raw text → Bluebell via anchor scan → scaffold → per-window body-fill."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import structlog

from codify.core.llm import LLMClient
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.pipeline.enrich.anchors import (
    AnchorCoverage,
    StructuralAnchor,
    Window,
    anchor_coverage,
    anchor_summary,
    cached_regex,
    markers_outside_boundary,
    scan_anchors_with_ambiguity,
    windows_from_anchors,
)
from codify.pipeline.enrich.arabic_normalise import JOINER_STRIP_TABLE
from codify.pipeline.enrich.container_coverage import container_coverage_probe
from codify.pipeline.enrich.kinds import CONTAINER_KINDS, kind_to_kw
from codify.pipeline.enrich.ocr import DEGRADED_MARKER_RE
from codify.pipeline.enrich.scaffold import (
    BodyBlock,
    BodyFillResponse,
    _join_soft_wraps,
    assemble_filled_scaffold,
    scaffold_from_anchors,
)
from codify.pipeline.enrich.table_fidelity import preserve_source_tables
from codify.pipeline.enrich.tables import has_table, nest_pipe_tables
from codify.pipeline.enrich.verbatim import fill_bodies_verbatim
from codify.quality.invariants import AmbiguitySpan

logger = structlog.get_logger()

MAX_CONCURRENT_SECTIONS = 30

_EMPTY_BLOCK = BodyBlock(eid="", lines=[])


@dataclass(frozen=True)
class StructureHalt:
    """A gate that would have refused the document, recorded instead of raised, so
    the law is readable and repair can reach it. The run carries an error finding, so
    the version grades blocking."""

    gate: str
    kind: str
    spans: int
    detail: str
    eid: str = ""
    first_offset: int | None = None
    captured: int | None = None
    expected: int | None = None
    ratio: float | None = None


@dataclass(frozen=True)
class BodyFillTrace:
    """What body-fill achieved against what the scaffold asked of it. `expected`
    counts units with body text in the source; `verbatim` were copied from it
    unstructured after the model failed them, and `empty` got nothing."""

    windows: int
    calls: int
    calls_failed: int
    expected: int
    verbatim: tuple[str, ...] = ()
    empty: tuple[str, ...] = ()

    @property
    def model_filled(self) -> int:
        return self.expected - len(self.verbatim) - len(self.empty)

    @property
    def ratio(self) -> float | None:
        """Share of expected bodies the model wrote; None when none were expected."""
        return self.model_filled / self.expected if self.expected else None


@dataclass(frozen=True)
class ScanTrace:
    """The deterministic half of a structuring run: the scan fixes the document's shape
    before any model call, so an ingest that produced nothing is explained here rather
    than in the output. ``scaffold`` is None when none was built, and ``fallback``
    then names the lane taken.
    """

    anchors: tuple[StructuralAnchor, ...]
    coverage: AnchorCoverage | None
    scaffold: str | None = None
    fallback: str | None = None
    ambiguity: tuple[AmbiguitySpan, ...] = ()  # present whether or not the gate fired
    # Gates that would have refused this document under `halt_policy="fail"`.
    halts: tuple[StructureHalt, ...] = ()
    # Guarded heading-vs-container probe: {present, found, grouping_declared}. The
    # ingest probe turns a below-floor ratio into a container_coverage warning.
    container: dict[str, int | bool] | None = None
    # None where body-fill never ran: no scaffold, or nothing to fill.
    body_fill: BodyFillTrace | None = None


PROMPTS_DIR = Path(__file__).parent / "prompts"
BODY_FILL_PROMPT = (PROMPTS_DIR / "structure_body_fill.txt").read_text()
# The prompt closes by saying the rules above are the only ones in force and that
# nothing follows the source tag. Additions go before that block, not after it.
_BODY_FILL_CLOSING = "Untrusted input\n"


def _body_fill_system(*additions: str) -> str:
    """The base rules, then any per-run additions, then the closing block."""
    rules, marker, closing = BODY_FILL_PROMPT.strip().partition(_BODY_FILL_CLOSING)
    if not marker:
        raise ValueError(f"{_BODY_FILL_CLOSING!r} missing from structure_body_fill.txt")
    parts = [rules.strip(), *(a for a in additions if a), marker + closing.strip()]
    return "\n\n".join(parts)


# RTL extraction leaves the visual `مادة N` in two logical-byte shapes the anchor
# regex misses: pdftotext's Bidi-controlled parens (`مادة ( )N` after strip), and
# the LLM extractors' `Nمادة`. The regex wants marker-then-digit, so either shape
# yields zero article anchors.
_BIDI_CONTROLS = str.maketrans("", "", "‎‏‪‫‬‭‮⁦⁧⁨⁩")
_EMPTY_PARENS_DIGIT_RE = re.compile(r"\(\s*\)\s*(\d+)")
# Rewrite `Nمادة` → `مادة N`; digits may be Latin, Arabic-Indic or Persian/Urdu.
# `[ \t]*` not `\s*`: `\s*` swallows a newline and welds the previous line's page
# number to the article number (`30\nمادة139` → `مادة 30139`), which cost one
# civil code ~65 articles.
_DIGIT_PREFIX_MADDA_RE = re.compile(r"([0-9٠-٩۰-۹]+)[ \t]*مادة")

# RTL reorder prefixes headers with a stray `)` on its own line (`)مادة 3`),
# defeating the regex's column boundary. Stripped before a line-start numbered
# article keyword; on three measured instruments it cost 21, nine and seven
# articles.
_STRAY_PAREN_MADDA_RE = re.compile(
    r"(?m)^([^\S\n]{0,8})\)[^\S\n]{0,4}(?=(?:ال)?مادة(?:[^\S\n]|\()*[0-9٠-٩۰-۹])"
)


# Balanced markdown bold only, mirroring translate/write.py's sanitiser:
# an unpaired `**` may be literal source content and is preserved.
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def normalise_rtl_extract(text: str) -> str:
    """Strip Bidi controls and Arabic joiners, and reorder digit/marker pairs RTL
    extraction leaves in the wrong logical order. Balanced markdown bold fences are
    unwrapped: `**` glued to a keyword (`# **مادة (٤٨)**`) defeats the anchor regex's
    column boundary. Unpaired `**` is left alone, being possibly literal source.
    """
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = text.translate(_BIDI_CONTROLS)
    text = text.translate(JOINER_STRIP_TABLE)
    text = _EMPTY_PARENS_DIGIT_RE.sub(r"(\1)", text)
    text = _DIGIT_PREFIX_MADDA_RE.sub(r"مادة \1", text)
    return _STRAY_PAREN_MADDA_RE.sub(r"\1", text)


def _expects_body(kind: str, basic: str | None = None) -> bool:
    # Only leaf units are flagged empty. `section` stays excluded even though the
    # canonical CONTAINER_KINDS omits it, unless the doctype numbers its provisions
    # as sections: then a section left empty is a dropped provision.
    return kind == basic or kind not in (CONTAINER_KINDS | {"section"})


def _anchors_with_body_source(text: str, anchors: list[StructuralAnchor]) -> set[str]:
    """Anchors with own body text, excluding captured headings and child spans."""
    ordered = sorted((a for a in anchors if not a.quoted_amendment), key=lambda a: a.char_offset)
    out: set[str] = set()
    for i, anchor in enumerate(ordered):
        end = ordered[i + 1].char_offset if i + 1 < len(ordered) else len(text)
        marker, _, body = text[anchor.char_offset : end].lstrip().partition("\n")
        body = " ".join(body.split())
        heading = " ".join((anchor.heading or "").split())
        if heading and heading not in " ".join(marker.split()):
            if body == heading:
                body = ""
            elif body.startswith(heading + " "):
                body = body[len(heading) :].strip()
        if body:
            out.add(anchor.akn_eid)
    return out


def _build_jurisdiction_context(config: JurisdictionConfig, doctype: str) -> str:
    lines = [f"Jurisdiction: {config.name} ({config.code.upper()})"]
    doc_class = config.get_document_class(doctype)
    if doc_class:
        lines.append(f"Basic unit: {doc_class.basic_unit}")
        if doc_class.hierarchy:
            hier_desc = " > ".join(
                f"{h.bluebell_keyword or h.akn_element.upper()} ({h.local_term})"
                for h in doc_class.hierarchy
            )
            lines.append(f"Hierarchy: {hier_desc}")
        if doc_class.hcontainers:
            for hc in doc_class.hcontainers:
                lines.append(f"Special element: {hc.local_term} → use {hc.bluebell_proxy} keyword")
    if config.structuring and config.structuring.prompt_additions:
        lines.append(config.structuring.prompt_additions)
    return "\n".join(lines)


def _scaffold_for_window(
    window_anchors: tuple[StructuralAnchor, ...],
    eid_to_anchor: dict[str, StructuralAnchor],
) -> str:
    """Mini-scaffold: each window anchor preceded by its ancestor headers."""
    seen: set[str] = set()
    ordered: list[StructuralAnchor] = []
    for anchor in window_anchors:
        if anchor.quoted_amendment:
            # Quoted-amendment anchors have no top-level eid and the outer
            # scaffolder emits them inside a QUOTE block. As window-fill targets
            # the LLM would see an eid="" header and drop the body.
            continue
        chain: list[StructuralAnchor] = []
        eid: str | None = anchor.parent_eid
        while eid is not None and eid in eid_to_anchor and eid not in seen:
            chain.append(eid_to_anchor[eid])
            eid = eid_to_anchor[eid].parent_eid
        for ancestor in reversed(chain):
            if ancestor.akn_eid not in seen:
                seen.add(ancestor.akn_eid)
                ordered.append(ancestor)
        if anchor.akn_eid not in seen:
            seen.add(anchor.akn_eid)
            ordered.append(anchor)

    lines = ["BODY"]
    for anchor in ordered:
        keyword = kind_to_kw(anchor.kind)
        indent = "  " * (anchor.depth + 1)
        number = anchor.number or ""
        lines.append(f"  eid={anchor.akn_eid}")
        lines.append(f"{indent}{keyword} {number}".rstrip())
    return "\n".join(lines)


# Appended per body-fill window that carries a fenced table row. A window with no
# table has no use for it.
TABLE_ROWS_RULE = (
    "This source contains a markdown table. A line whose content is "
    "`| cell | cell |` is body content: copy it into `lines` verbatim and in "
    "order, the `|---|---|` separator row included, and never reflow it into "
    "sentences, merge its rows, or drop it. A schedule of rates read as prose "
    "loses which value belongs to which heading."
)


def _unreadable_remark(page: int) -> str:
    return f"{{{{*[Page {page} of the source could not be read]}}}}"


def _lift_unreadable_markers(text: str) -> tuple[str, list[tuple[int, int]]]:
    """The text without unreadable-page marker lines, and (offset, page) for each,
    the offset indexing the returned text. Kept out of what the model fills."""
    kept: list[str] = []
    found: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        m = DEGRADED_MARKER_RE.match(line)
        if m:
            found.append((pos, int(m.group(1))))
            continue
        kept.append(line)
        pos += len(line) + 1
    return "\n".join(kept), found


def _with_remarks_inline(text: str, markers: list[tuple[int, int]]) -> str:
    """Put each marker back where it stood, as a remark paragraph of its own."""
    for offset, page in sorted(markers, reverse=True):
        at = min(offset, len(text))
        text = f"{text[:at]}\n\n{_unreadable_remark(page)}\n\n{text[at:]}"
    return text


def _place_unreadable_remarks(
    markers: list[tuple[int, int]],
    anchors: list[StructuralAnchor],
    by_eid: dict[str, BodyBlock],
    basic: str | None,
) -> None:
    """Append each marker's remark to the provision whose body the lost page
    continued, or the first provision when the page precedes them all."""
    hosts = sorted(
        (a for a in anchors if not a.quoted_amendment and _expects_body(a.kind, basic)),
        key=lambda a: a.char_offset,
    )
    if not hosts:
        return
    for offset, page in markers:
        before = [a for a in hosts if a.char_offset <= offset]
        host = before[-1] if before else hosts[0]
        block = by_eid.get(host.akn_eid) or BodyBlock(eid=host.akn_eid)
        remark = _unreadable_remark(page)
        lines = [*block.lines, remark] if before else [remark, *block.lines]
        by_eid[host.akn_eid] = block.model_copy(update={"lines": lines})


def _verbatim_single_section(text: str) -> str:
    """Wrap anchorless document text in one SECTION so the body survives as a
    provision instead of being discarded. Used only when no structural anchors
    are detected (e.g. short orders with prose but no `مادة N` markers)."""
    lines = ["BODY", "  SECTION 1"]
    # This path builds its own body, so it nests tables itself or loses them:
    # an anchorless document is exactly the kind that is mostly a table.
    for para in nest_pipe_tables(_join_soft_wraps(text.splitlines())):
        lines.append(f"    {para}" if para else "")
    return "\n".join(lines) + "\n"


class SourceTruncationError(RuntimeError):
    """The source declares units it does not contain, so it is not the whole document.

    Distinct from `AnchorCoverageError`, which measures how well the scanner read the
    text it was given. Both counts here come from the same text, which is why coverage
    cannot see this: truncation shrinks numerator and denominator together.
    """

    def __init__(self, *, kind: str, listed: int, bodied: int, ratio: float, threshold: float):
        super().__init__(
            f"source truncated: {bodied} of {listed} {kind} units pair a contents "
            f"line with a body, so {ratio:.0%} are listed and never bodied, against "
            f"a {threshold:.0%} floor. Check the source file is complete."
        )
        self.kind = kind
        self.listed = listed
        self.bodied = bodied
        self.ratio = ratio
        self.threshold = threshold


class AnchorCoverageError(RuntimeError):
    """Scanner captured too few structural anchors against the source's marker count.
    Raised before the structurer LLM sees a partial document, and carries the observed
    ratio and counts so the ingest error record names the failure without further
    logging.
    """

    def __init__(
        self,
        *,
        kind: str,
        captured: int,
        expected: int,
        ratio: float,
        threshold: float,
        masked: int = 0,
        unclosed: int = 0,
    ) -> None:
        # A ratio cannot show this one: the denominator is built through the
        # same mask, so the hidden markers leave both sides and it reads 1.0.
        super().__init__(
            f"{masked} {kind} marker(s) hidden by {unclosed} unclosed quote(s); "
            f"the source text carries a quote nothing closes, so those "
            f"provisions were never anchored and coverage cannot show it."
            if masked
            else f"anchor coverage {ratio:.0%} below floor {threshold:.0%}: "
            f"scanner captured {captured} {kind} anchors but the source "
            f"text contains {expected} structural marker occurrences. "
            f"Structurer halted; investigate marker-literal variants before "
            f"retrying."
        )
        self.kind = kind
        self.captured = captured
        self.expected = expected
        self.ratio = ratio
        self.masked = masked
        self.unclosed = unclosed
        self.threshold = threshold


class BodyFillError(RuntimeError):
    """Body-fill failed and the model wrote no body for the document."""

    def __init__(self, *, windows: int, detail: str | None = None) -> None:
        super().__init__(
            detail
            or f"body fill produced no text for any of {windows} window(s); "
            "the scaffold is correct and the document has no body"
        )
        self.windows = windows


class AnchorInvariantError(RuntimeError):
    """The anchor set breaks a contract that holds whatever produced it.

    Raised before the scaffold, on unresolved spans only, carrying them so the
    ingest error names what went wrong."""

    def __init__(self, *, spans: list[AmbiguitySpan]) -> None:
        by_kind: Counter[str] = Counter(s.kind for s in spans)
        summary = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
        first = spans[0]
        super().__init__(
            f"anchor set breaks {len(spans)} invariant(s): {summary}. "
            f"First at offset {first.start} ({first.emitted_by}): {first.detail}. "
            f"Structurer halted; the anchors would not round-trip."
        )
        self.spans = spans
        self.by_kind = dict(by_kind)


# The top-level provision carrying prose. `paragraph`/`point` sit below as
# list-item children, so `hierarchy[-1]` would pick a child whose absence is
# normal. Ordered by preference, so a hierarchy with both `article` and `section`
# picks `article`.
_BASIC_UNIT_KINDS: tuple[str, ...] = ("article", "section", "rule")


def basic_unit_kind(config: JurisdictionConfig | None, doctype: str) -> str | None:
    """The AKN element carrying the doctype's numbered provisions: first hierarchy entry
    whose ``akn_element`` is in ``_BASIC_UNIT_KINDS``, so the gate compares real
    provisions rather than list-item children. None when the config has no hierarchy or
    no basic-unit kind, and the caller then skips the coverage gate.
    """
    if config is None:
        return None
    doc_class = config.get_document_class(doctype)
    if doc_class is None or not doc_class.hierarchy:
        return None
    # A class whose provisions are marked without a keyword names its own citable
    # unit: an MK judgment is cited by paragraph, and `section` would gate against
    # its four captions. Narrowed to marker-form classes.
    declared = doc_class.basic_unit
    if declared and any(h.akn_element == declared and h.marker_form for h in doc_class.hierarchy):
        return declared
    for candidate in _BASIC_UNIT_KINDS:
        if any(h.akn_element == candidate for h in doc_class.hierarchy):
            return candidate
    return None


async def text_to_bluebell_scaffolded(
    text: str,
    *,
    client: LLMClient,
    country: str,
    doctype: str = "act",
    on_progress: Callable[[float, str], None] | None = None,
    on_anchors: Callable[[dict[str, int], int], None] | None = None,
    on_scan: Callable[[ScanTrace], None] | None = None,
    on_stream: Callable[[str], None] | None = None,
    model: str | None = None,
    halt_policy: Literal["fail", "land"] = "fail",
) -> str:
    """scan then scaffold then per-window chat_schema body-fill then assemble.

    Under ``halt_policy="land"`` a refusing gate records a `StructureHalt` and
    continues, a truncated source included: the copy held is the best one there is,
    and a blocking version names the gap where a failed run leaves nothing to find.
    Nothing measured, and stranded markers, still raise."""
    text = normalise_rtl_extract(text)
    text, unreadable = _lift_unreadable_markers(text)
    config = load_config(country)
    regex = cached_regex(country, doctype)
    scan = scan_anchors_with_ambiguity(text, regex, country=country, doctype=doctype)
    anchors = scan.anchors
    # Source offsets throughout: the trace's readers locate the source with them.
    trace_anchors = anchors
    # The signature ends the body: the span to the first attachment leaves the
    # text every gate below reads and returns as the conclusions.
    from codify.pipeline.enrich.closing import bound_body_at_closing, closing_phrases_for

    bound = bound_body_at_closing(text, anchors, closing_phrases_for(country), country=country)
    text, anchors = bound.text, bound.anchors

    if on_anchors:
        # The total counts the same population the summary does, or the trace
        # reports a document with more anchors than kinds to put them in.
        own = [a for a in anchors if not a.quoted_amendment]
        on_anchors(anchor_summary(own), len(own))

    coverage: AnchorCoverage | None = None
    # Guarded heading-vs-container probe from this same scan; the ingest probe
    # turns a below-floor ratio into a container_coverage warning. Independent of
    # the coverage gate, so measured once here and carried on every trace.
    container = container_coverage_probe(
        text, replace(scan, anchors=anchors), config, country, doctype
    )

    halts: list[StructureHalt] = []

    def _trace(
        scaffold: str | None = None,
        fallback: str | None = None,
        body_fill: BodyFillTrace | None = None,
    ) -> None:
        if on_scan:
            on_scan(
                ScanTrace(
                    anchors=tuple(trace_anchors),
                    coverage=coverage,
                    scaffold=scaffold,
                    fallback=fallback,
                    ambiguity=tuple(scan.ambiguity),
                    halts=tuple(halts),
                    container=container,
                    body_fill=body_fill,
                )
            )

    # Source-integrity gate, ahead of coverage because it answers a different
    # question: not how well the text was read, but whether it is the whole
    # document. It refuses under `land` as well, since a truncated source that
    # lands is missing law recorded as present.
    # `source_truncation_floor = 0.0` disables it, which is every jurisdiction that has
    # not measured its own separating value.
    tail_floor = config.source_truncation_floor if config is not None else 0.0
    tail = next(
        (sp for sp in scan.ambiguity if sp.kind == "untwinned_tail"),
        None,
    )
    if tail is not None and tail_floor > 0:
        listed = int(tail.detail.get("present", 0))
        bodied = int(tail.detail.get("twinned", 0))
        absent = listed - bodied
        share = absent / listed if listed else 0.0
        if share > tail_floor:
            logger.warning(
                "source_truncation_suspected",
                country=country,
                doctype=doctype,
                kind=tail.detail.get("kind"),
                listed=listed,
                bodied=bodied,
                share=round(share, 3),
                threshold=tail_floor,
            )
            detail = (
                f"{bodied} of {listed} {tail.detail.get('kind')} units pair a "
                f"contents line with a body, so {share:.0%} are listed and never "
                f"bodied, against a {tail_floor:.0%} floor"
            )
            if halt_policy == "land":
                # Landed rather than refused: the source is the best copy held, and a
                # blocking version names the gap where a failed run leaves nothing to
                # find. Recovery is a longer copy of the file, not a re-run of this one.
                halts.append(
                    StructureHalt(
                        gate="source_truncated",
                        kind=str(tail.detail.get("kind") or ""),
                        spans=int(tail.detail.get("untwinned_count") or 0),
                        detail=detail,
                        eid=tail.eid,
                        first_offset=tail.start,
                        captured=bodied,
                        expected=listed,
                        ratio=share,
                    )
                )
            else:
                _trace(fallback="source_truncation")
                raise SourceTruncationError(
                    kind=str(tail.detail.get("kind") or ""),
                    listed=listed,
                    bodied=bodied,
                    ratio=share,
                    threshold=tail_floor,
                )

    # Coverage-delta gate: anchor count for the basic-unit kind against marker
    # occurrences in the source. Below-floor fails loud, so a silent drop, as a
    # tatweel-variant `مـادة` once caused, cannot ship as a partial AKN.
    # `min_anchor_coverage = 0.0` disables it; no markers measures None, not 1.0.
    threshold = config.min_anchor_coverage if config is not None else 0.0
    kind = basic_unit_kind(config, doctype)
    if threshold > 0 and kind is not None:
        coverage = anchor_coverage(text, anchors, config, doctype, kind, with_masked=True)
        ratio = coverage.ratio
        captured, expected = len(coverage.captured), len(coverage.expected)
        if ratio is not None and ratio < threshold:
            logger.warning(
                "anchor_coverage_below_floor",
                country=country,
                doctype=doctype,
                kind=kind,
                captured=captured,
                expected=expected,
                ratio=round(ratio, 3),
                threshold=threshold,
            )
            # Markers found and none recognised: the text is not this document,
            # so it fails whatever the policy. Truncation is a separate check.
            if halt_policy == "land" and ratio > 0:
                halts.append(
                    StructureHalt(
                        gate="coverage_below_floor",
                        kind=kind,
                        spans=expected - captured,
                        detail=(
                            f"captured {captured} of {expected} {kind} markers, "
                            f"{ratio:.0%} against a {threshold:.0%} floor"
                        ),
                        captured=captured,
                        expected=expected,
                        ratio=ratio,
                    )
                )
            else:
                _trace(fallback="coverage_gate")
                raise AnchorCoverageError(
                    kind=kind,
                    captured=captured,
                    expected=expected,
                    ratio=ratio,
                    threshold=threshold,
                )
        # The ratio cannot see this: its denominator is built through the same
        # mask, so a document hiding half its subdivisions still reads 1.0.
        if coverage.masked:
            logger.warning(
                "anchor_coverage_markers_masked",
                country=country,
                doctype=doctype,
                kind=kind,
                masked=coverage.masked,
                unclosed=coverage.unclosed,
            )
            if halt_policy == "land":
                halts.append(
                    StructureHalt(
                        gate="markers_masked",
                        kind=kind,
                        spans=coverage.masked,
                        detail=(
                            f"{coverage.masked} marker(s) hidden by "
                            f"{coverage.unclosed} unclosed quote(s)"
                        ),
                    )
                )
            else:
                _trace(fallback="coverage_gate")
                raise AnchorCoverageError(
                    kind=kind,
                    captured=captured,
                    expected=expected,
                    ratio=ratio if ratio is not None else 0.0,
                    threshold=threshold,
                    masked=coverage.masked,
                    unclosed=coverage.unclosed,
                )
        # An empty denominator usually means no markers of this kind. Under
        # `line_anchored` it can instead mean the source lost its line layout,
        # which that pattern cannot tell apart, so ask the relaxed boundary first.
        stranded = markers_outside_boundary(text, config, doctype, kind) if expected == 0 else 0
        if stranded:
            logger.warning(
                "anchor_boundary_policy_void",
                country=country,
                doctype=doctype,
                kind=kind,
                stranded=stranded,
            )
            _trace(fallback="coverage_gate")
            raise AnchorCoverageError(
                kind=kind,
                captured=0,
                expected=stranded,
                ratio=0.0,
                threshold=threshold,
            )
        if expected == 0 and len(text) > 5000:
            # A non-trivial document with no markers of the basic kind is usually
            # a doctype mismatch (regulation against an act hierarchy), which would
            # otherwise no-op the gate silently.
            logger.info(
                "anchor_coverage_gate_skipped",
                country=country,
                doctype=doctype,
                kind=kind,
                reason="zero_marker_occurrences",
                text_length=len(text),
            )
    elif threshold > 0 and kind is None:
        logger.info(
            "anchor_coverage_gate_skipped",
            country=country,
            doctype=doctype,
            reason="no_hierarchy_for_doctype",
        )

    if not anchors:
        if text.strip():
            logger.warning(
                "scaffold_no_anchors", country=country, doctype=doctype, fallback="verbatim_section"
            )
            _trace(fallback="verbatim_section")
            return _verbatim_single_section(_with_remarks_inline(text, unreadable))
        logger.warning("scaffold_no_anchors", country=country, doctype=doctype, fallback="empty")
        _trace(fallback="empty")
        return "PREFACE\n\nBODY\n"

    # After the anchorless fallback: a document with no anchors loses structure,
    # not law, so verbatim beats refusal.
    blocking = [s for s in scan.ambiguity if s.blocking]
    if blocking:
        logger.warning(
            "anchor_invariants_broken",
            country=country,
            doctype=doctype,
            spans=len(blocking),
            kinds=sorted({s.kind for s in blocking}),
            detail=[s.detail for s in blocking[:5]],
        )
        # Landing rests on `_assign_eids` having suffixed each colliding eId,
        # true of `duplicate_number` and asserted of no other kind.
        first = blocking[0]
        if halt_policy == "land" and all(s.kind == "duplicate_number" for s in blocking):
            halts.append(
                StructureHalt(
                    gate="duplicate_anchor",
                    kind=first.kind,
                    spans=len(blocking),
                    detail=f"{first.emitted_by}: {first.detail}",
                    eid=first.eid,
                    first_offset=first.start,
                )
            )
        else:
            _trace(fallback="invariant_gate")
            raise AnchorInvariantError(spans=blocking)

    from codify.pipeline.enrich.enacting import split_opening_material

    preface, preamble = split_opening_material(text[: min(a.char_offset for a in anchors)], country)
    scaffold, eid_to_anchor = scaffold_from_anchors(
        anchors, preface=preface, preamble=preamble, country=country, conclusions=bound.conclusions
    )
    windows = windows_from_anchors(text, anchors)
    if not windows:
        # Containers only: nothing to fill, and the skeleton ships as the document.
        logger.warning("body_fill_skipped", reason="no_basic_unit_anchors", anchors=len(anchors))
        _trace(scaffold=scaffold)
        return scaffold

    # One trace per run, after body-fill, carrying what it achieved; emitted on
    # the way out of a failure too, so the scaffold is never lost with it.
    fill_trace: list[BodyFillTrace] = []
    try:
        base_additions: list[str] = []
        if config is not None:
            base_additions.append(_build_jurisdiction_context(config, doctype))

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SECTIONS)
        completed = 0
        failed = 0
        calls = 0
        total = len(windows)

        def nonlocal_failed() -> None:
            nonlocal failed
            failed += 1

        async def fill_window(window: Window, label: str) -> BodyFillResponse:
            nonlocal completed, calls
            calls += 1
            async with semaphore:
                window_scaffold = _scaffold_for_window(window.anchors, eid_to_anchor)
                # Tagged, not concatenated: the source is OCR of a document the
                # pipeline did not write, and a line in it shaped like an
                # instruction reads as one when it arrives bare. The tag carries a
                # digest of the slice, so a source containing a closing tag cannot
                # end the span early and nothing has to be stripped from it:
                # censoring a provision would be the worse failure.
                body = window.text[window.body_start : window.body_end]
                tag = f"source-{hashlib.sha256(body.encode()).hexdigest()[:12]}"
                user_prompt = (
                    f"<scaffold>\n{window_scaffold}\n</scaffold>\n\n<{tag}>\n{body}\n</{tag}>"
                )
                # Per window, and only where the nester would build a table.
                additions = list(base_additions)
                if has_table(body.splitlines()):
                    additions.append(TABLE_ROWS_RULE)
                window_system = _body_fill_system(*additions)
                try:
                    response = await client.chat_schema(
                        user_prompt,
                        BodyFillResponse,
                        system=window_system,
                        model=model,
                    )
                except Exception as exc:
                    nonlocal_failed()
                    logger.warning(
                        "body_fill_failed",
                        chunk=label,
                        anchors=len(window.anchors),
                        error=f"{type(exc).__name__}: {str(exc)[:160]}",
                    )
                    response = BodyFillResponse(bodies=[])
                owned = {a.akn_eid for a in window.anchors if not a.quoted_amendment}
                rejected = [block.eid for block in response.bodies if block.eid not in owned]
                if rejected:
                    logger.warning("body_fill_unowned_targets", chunk=label, eids=rejected)
                response = BodyFillResponse(
                    bodies=[block for block in response.bodies if block.eid in owned]
                )
            completed += 1
            if on_progress:
                on_progress(min(completed / total, 1.0), f"window {completed}/{total}")
            if on_stream:
                on_stream(response.model_dump_json())
            return response

        # eid → body; a non-empty body always wins so recovery refills overwrite drops.
        by_eid: dict[str, BodyBlock] = {}

        def absorb(responses: list[BodyFillResponse]) -> None:
            for r in responses:
                for block in r.bodies:
                    if block.lines or block.eid not in by_eid:
                        by_eid[block.eid] = block

        absorb(await asyncio.gather(*[fill_window(w, f"w{i}") for i, w in enumerate(windows)]))

        # An overflowed window drops every body (LengthFinishReasonError) and a
        # succeeded one can still omit some; both surface as empty anchors that have
        # source text. Bare-heading anchors are legitimately empty and left alone; the
        # rest re-fill in progressively smaller windows.
        has_body_source = _anchors_with_body_source(text, anchors)
        basic = basic_unit_kind(config, doctype)
        with_body = [
            a.akn_eid
            for a in anchors
            if _expects_body(a.kind, basic) and a.akn_eid in has_body_source
        ]
        for max_per in (2, 1):
            targets = {
                a.akn_eid
                for a in anchors
                if _expects_body(a.kind, basic)
                and a.akn_eid in has_body_source
                and not by_eid.get(a.akn_eid, _EMPTY_BLOCK).lines
            }
            if not targets:
                break
            recovery = [
                w
                for w in windows_from_anchors(
                    text, anchors, max_per_window=max_per, min_per_window=1, target_size=10**9
                )
                if any(a.akn_eid in targets for a in w.anchors)
            ]
            if not recovery:
                break
            logger.info(
                "body_fill_recovery", empty=len(targets), windows=len(recovery), max_per=max_per
            )
            absorb(
                await asyncio.gather(
                    *[fill_window(w, f"r{max_per}-{i}") for i, w in enumerate(recovery)]
                )
            )

        # Some windows overflow the output-token budget because the model loops on the
        # content rather than by size, so even a single-anchor window fails. Splice the
        # raw source span in: an imperfect body beats a dropped one.
        still_empty = {
            a.akn_eid
            for a in anchors
            if _expects_body(a.kind, basic)
            and a.akn_eid in has_body_source
            and not by_eid.get(a.akn_eid, _EMPTY_BLOCK).lines
        }
        fallback: list[BodyBlock] = []
        if still_empty:
            try:
                fallback = [
                    b
                    for b in fill_bodies_verbatim(text, anchors).bodies
                    if b.eid in still_empty and b.lines
                ]
            except Exception as exc:  # noqa: BLE001, never let the fallback break ingest
                logger.warning("body_fill_verbatim_fallback_failed", error=str(exc)[:160])
                fallback = []
            if fallback:
                logger.info("body_fill_verbatim_fallback", count=len(fallback))
                absorb([BodyFillResponse(bodies=fallback)])
        verbatim = {b.eid for b in fallback}
        stats = BodyFillTrace(
            windows=total,
            calls=calls,
            calls_failed=failed,
            expected=len(with_body),
            verbatim=tuple(e for e in with_body if e in verbatim),
            empty=tuple(e for e in with_body if not by_eid.get(e, _EMPTY_BLOCK).lines),
        )
        fill_trace.append(stats)

        if failed and failed == calls and not any(b.lines for b in by_eid.values()):
            # Every call failed, recovery included, and nothing recovered it: a
            # correct skeleton with no law in it, refused under either policy.
            raise BodyFillError(windows=total)
        if failed and stats.expected and not stats.model_filled:
            # The model could not be reached for any body. Empty answers stay
            # findings (verbatim or empty bodies), not a halt.
            logger.warning(
                "body_fill_no_model_output", calls=calls, failed=failed, expected=stats.expected
            )
            detail = (
                f"the model wrote no body ({failed} of {calls} body-fill calls failed); "
                f"{len(stats.verbatim)} of {stats.expected} bodies are source text "
                f"copied in unstructured and {len(stats.empty)} are empty"
            )
            if halt_policy != "land":
                raise BodyFillError(windows=total, detail=detail)
            halts.append(
                StructureHalt(
                    gate="body_fill_failed",
                    kind=basic or "",
                    spans=stats.expected,
                    detail=detail,
                    captured=0,
                    expected=stats.expected,
                    ratio=0.0,
                )
            )

        literal_eids = preserve_source_tables(text, anchors, by_eid)
        _place_unreadable_remarks(unreadable, anchors, by_eid, basic)
        return assemble_filled_scaffold(
            scaffold,
            eid_to_anchor,
            BodyFillResponse(bodies=list(by_eid.values())),
            literal_body_eids=literal_eids,
        )
    finally:
        _trace(scaffold=scaffold, body_fill=fill_trace[0] if fill_trace else None)


__all__ = [
    "AnchorCoverageError",
    "AnchorInvariantError",
    "BodyFillError",
    "BODY_FILL_PROMPT",
    "BodyFillTrace",
    "ScanTrace",
    "basic_unit_kind",
    "normalise_rtl_extract",
    "text_to_bluebell_scaffolded",
]
