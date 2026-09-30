# ruff: noqa: E501
"""Deterministic post-enrichment AKN validation.

Checks identity, references, definitions, numbering, hierarchy and common
enrichment artefacts. Results are returned as issue dictionaries for persistence.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.akn._schema import parse_xml
from codify.akn.vocabulary import (
    QUOTED_AMENDMENT_ANCESTORS,
    count_provisions_by_kind,
    local_name,
    provision_text,
)
from codify.frbr import UNKNOWN_YEAR, is_citable_work_uri
from codify.lang import normalise_digits
from codify.pipeline.enrich.akn_meta import FORMULA_PROVENANCE
from codify.pipeline.enrich.anchors import arabic_article_headers
from codify.pipeline.enrich.arabic_cardinals import find_money_word_mismatches
from codify.pipeline.enrich.arabic_normalise import (
    fold_arabic_for_match,
    heading_restates_number,
)
from codify.pipeline.enrich.kinds import CONTAINER_KINDS
from codify.pipeline.enrich.scripts.arabic import ARABIC
from codify.quality.invariants import classify_gap, missing_between
from codify.quality.lexicons import ARABIC_WORDS

logger = structlog.get_logger()


def validate_akn(
    akn_xml: str,
    *,
    huge_tree: bool = False,
    expected_anchor_summary: dict[str, int] | None = None,
    expected_cover_article_numbers: list[int] | None = None,
    source_text: str | None = None,
    closing_phrases: list[str] | None = None,
    page_yield: list[tuple[int, float]] | None = None,
    recoverable_pages: set[int] | None = None,
    orphaned_drops: list[dict[str, Any]] | None = None,
    container_coverage: dict[str, Any] | None = None,
    provenance: Literal["native", "extracted"] | None = None,
    unreadable_pages: dict[int, str] | None = None,
    body_fill: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run all checks, returning issue dicts. The optional arguments are evidence only the
    scan or extractor can see, so each enables a check that cannot be derived from the
    AKN alone:

    ``expected_anchor_summary`` kind->count from the pre-flight regex probe; a shortfall
    signals dropped structural units. ``expected_cover_article_numbers`` the article
    numbers the cover page carried; a gap over +/-1 means whole-article drops the
    sequence repair cannot see, delegated to ``cover_reconciliation``. ``source_text``
    the raw structuring text; an enacting formula absent from it is fabricated
    boilerplate. ``closing_phrases`` the jurisdiction's attestation vocabulary, for
    promulgation lines left in the last body provision. ``page_yield`` (page_number,
    chars_per_ink) per source page, naming pages the primary engine read empty.
    ``recoverable_pages`` the subset of those whose content the AKN carries; the two
    together decide whether an empty page blocks delivery or only warns, and a page
    absent from the subset stays blocking. ``orphaned_drops`` text an anchor drop left
    with no provision to hold it. ``container_coverage`` the guarded heading-vs-container
    probe ({present, found}); below-floor means the structurer flattened a grouping level.
    ``provenance`` is where the structure came from: ``native`` AKN a publisher
    structured, whose dates and numbering are the publisher's own, or text the pipeline
    ``extracted`` (OCR, vision, a text layer), where a numbering gap is likely lost
    content. None keeps the checks provenance-neutral. ``unreadable_pages`` page number
    to reason for pages that came back without text. ``body_fill`` the structurer's
    `BodyFillTrace` as a dict, naming bodies the model never wrote.
    """
    root = parse_xml(akn_xml, huge_tree=huge_tree)
    issues: list[dict[str, Any]] = []

    issues.extend(_check_eid_uniqueness(root))
    issues.extend(_check_eid_unusable(root))
    issues.extend(_check_ref_resolution(root))
    issues.extend(_check_definition_completeness(root))
    issues.extend(_check_section_numbering(root))
    issues.extend(_check_hierarchy_coherence(root))
    issues.extend(_check_uncitable_work_uri(root))
    if expected_anchor_summary:
        issues.extend(_check_anchor_count_mismatch(root, expected_anchor_summary))
    if expected_cover_article_numbers is not None:
        issues.extend(_check_cover_body_reconciliation(root, expected_cover_article_numbers))
    bis_scopes = collapsed_bis_scopes(root)
    issues.extend(_check_collapsed_bis_article(bis_scopes))
    issues.extend(
        _check_number_set_continuity(
            root,
            suppress=bis_scopes,
            extracted=provenance == "extracted",
            unreadable=sorted(unreadable_pages or {}),
        )
    )
    if provenance == "extracted":
        issues.extend(_check_leading_gap(root))
    issues.extend(_check_seam_duplication(root))
    issues.extend(_check_body_artefacts(root))
    issues.extend(_check_markup_as_text(root))
    issues.extend(_check_orphan_articles(root))
    issues.extend(_check_empty_body(root))
    issues.extend(_check_structureless_body(root))
    issues.extend(_check_ocr_garble(root))
    issues.extend(_check_empty_articles(root))
    issues.extend(_check_swallowed_enumerators(root))
    issues.extend(_check_undetected_amendment_shape(root))
    issues.extend(check_missing_container_titles(root))
    issues.extend(_check_formula_integrity(root))
    issues.extend(_check_money_words_mismatch(root))
    issues.extend(_check_identity_consistency(root, native=provenance == "native"))
    if source_text is not None:
        issues.extend(_check_fabricated_formula(root, source_text))
        issues.extend(_check_header_coverage(root, source_text))
    if closing_phrases:
        issues.extend(_check_displaced_terminal_material(root, closing_phrases))
    if page_yield:
        issues.extend(_check_page_yield(page_yield, recoverable_pages))
    if orphaned_drops:
        issues.extend(_check_orphaned_drops(orphaned_drops))
    if container_coverage:
        issues.extend(_check_container_coverage(container_coverage))
    if unreadable_pages:
        issues.extend(_check_unreadable_pages(unreadable_pages))
    if body_fill:
        issues.extend(_check_body_fill(body_fill))

    if issues:
        logger.info("validation_issues_found", count=len(issues))

    return issues


_NUMBERED_KINDS = ("article", "section", "paragraph", "point", "subsection")


def _check_number_set_continuity(
    root: etree._Element,
    *,
    suppress: dict[str, set[str]] | None = None,
    extracted: bool = False,
    unreadable: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Catch structural drops the count-only validator misses. For each continuously
    numbered kind, flag duplicates (two eIds sharing a num, the same unit emitted twice
    in adjacent chunks) and numeric gaps (9 -> 20). Gaps fire on arabic-only numbers, to
    avoid false positives on roman or jurisdiction schemes like "22/1".

    ``suppress`` maps a parent eId to folded letters whose duplication a more specific
    finding already explains, so the agent is not routed to a cosmetic renumber. An
    unrelated duplicate under the same parent stays intact.

    ``extracted`` warns on every gap, since text read off a page loses provisions
    with the page. ``unreadable`` pages keep a gap from being called a repeal.
    """
    suppress = suppress or {}
    out: list[dict[str, Any]] = []
    for kind in _NUMBERED_KINDS:
        entries: list[tuple[str, str, str]] = []  # (eid, num, scope)
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if el.tag.split("}", 1)[-1] != kind:
                continue
            eid = el.get("eId")
            if not eid:
                continue
            # Direct child <num> only, nested children's nums belong to them.
            num_el = next(
                (c for c in el if isinstance(c.tag, str) and c.tag.split("}", 1)[-1] == "num"),
                None,
            )
            if num_el is None:
                continue
            num_text = "".join(num_el.itertext()).strip().rstrip(".,;:")
            if not num_text:
                continue
            # Fold non-ASCII digits so `(۳)` and `(٣)` collide under one key: both
            # render as "3" and mean the same point, and without the fold OCR
            # script-mixing hides real duplicates from this check.
            num_key = normalise_digits(num_text)
            # Scope by the nearest eId-bearing ancestor: a subsection may sit
            # under an unlabelled content/hcontainer wrapper, and using the bare
            # parent would collapse every section's "(1)" to document level.
            scope = next(
                (a.get("eId") for a in el.iterancestors() if a.get("eId")),
                "",
            )
            entries.append((eid, num_key, scope))
        if not entries:
            continue

        # 1. Duplicates. Numbering only has to be unique among siblings, so
        # scope by parent eId for every kind, section V in two chapters is fine.
        by_scope_num: dict[tuple[str, str], list[str]] = {}
        for eid, num, scope in entries:
            by_scope_num.setdefault((scope, num), []).append(eid)
        for (scope, num), eids in by_scope_num.items():
            if len(eids) > 1:
                letter = fold_arabic_for_match(re.sub(r"[()\[\].,؛:\s]", "", num))
                if kind == "point" and letter in suppress.get(scope, set()):
                    continue
                where = f" within parent {scope!r}" if scope else " at document level"
                out.append(
                    {
                        "check": "duplicate_number",
                        "severity": "warning",
                        "kind": kind,
                        "number": num,
                        "scope": scope or None,
                        "eids": eids,
                        "message": (
                            f"{kind} number {num!r} appears on {len(eids)} distinct eIds"
                            f"{where} ({', '.join(eids)})."
                        ),
                    }
                )

        # 2. Numeric gaps, scoped per parent: a document-wide sequence would read
        # the jump between one article's points and the next as a gap and host the
        # repair on the wrong provision. Only the contiguous arabic-numbered head
        # per scope, since mixed schemes (22/1, 22/2) break continuity legitimately.
        by_scope: dict[str, list[tuple[str, str]]] = {}
        for eid, num, scope in entries:
            by_scope.setdefault(scope, []).append((eid, num))
        for _scope, scoped in by_scope.items():
            arabic_nums: list[int] = []
            num_to_eid: dict[int, str] = {}
            for eid, num in scoped:
                if num.isdigit():
                    arabic_nums.append(int(num))
                    num_to_eid.setdefault(int(num), eid)
                else:
                    # Non-arabic, stop walking; ranges beyond are unsafe.
                    break
            seen: set[int] = set()
            deduped: list[int] = []
            for n in arabic_nums:
                if n not in seen:
                    seen.add(n)
                    deduped.append(n)
            if len(deduped) >= 2:
                _, total, max_run, lo, hi = missing_between(deduped)
                if total:
                    likely = classify_gap(total, max_run, lo, hi)
                    out.extend(
                        _gap_run_findings(
                            root,
                            kind,
                            sorted(seen),
                            num_to_eid,
                            likely=likely,
                            extracted=extracted,
                            unreadable=unreadable or [],
                        )
                    )
    return out


# One finding per gap run keeps emission bounded on pathological sequences.
_MAX_GAP_RUNS = 10


def _gap_acknowledged(root: etree._Element, eid: str) -> bool:
    """A gap whose host provision carries an editorial-gap note is a confirmed repeal, not
    re-fired. Only the host's own note counts: a note whose nearest numbered ancestor is
    a nested provision belongs to that provision's gap, so an article's note must not
    clear its section's, or the reverse.
    """
    hits = root.xpath(".//*[@eId=$e]", e=eid)
    if not hits:
        return False
    host = hits[0]
    for note in host.iter(f"{{{AKN_NS}}}authorialNote"):
        if note.get("marker") != "editorial-gap":
            continue
        owner = next(
            (
                a
                for a in note.iterancestors()
                if isinstance(a.tag, str) and a.tag.split("}", 1)[-1] in _NUMBERED_KINDS
            ),
            None,
        )
        if owner is host:
            return True
    return False


def _gap_run_findings(
    root: etree._Element,
    kind: str,
    numbers: list[int],
    num_to_eid: dict[int, str],
    *,
    likely: str,
    extracted: bool = False,
    unreadable: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Per-run gap findings, hosted on the provision numbered before the run so
    a repair (an editorial annotation) has a target and can clear the finding."""
    if unreadable:
        # A page nobody read can hold the missing numbers, so no repeal is inferred.
        likely = "unreadable_page"
    runs: list[tuple[int, int, str | None]] = []  # (first_missing, last_missing, host_eid)
    for a, b in zip(numbers, numbers[1:], strict=False):
        if b - a > 1:
            runs.append((a + 1, b - 1, num_to_eid.get(a)))
    label = (
        "a structuring drop"
        if likely == "defect"
        else "legitimate repeals"
        if likely == "repeal"
        else f"content on unreadable page(s) {', '.join(map(str, unreadable or []))}"
        if likely == "unreadable_page"
        else "a drop or repeals"
    )
    out: list[dict[str, Any]] = []
    unacknowledged = [
        (first, last, host)
        for first, last, host in runs
        if not (host and _gap_acknowledged(root, host))
    ]
    for first, last, host in unacknowledged[:_MAX_GAP_RUNS]:
        missing = list(range(first, min(last, first + 49) + 1))
        finding: dict[str, Any] = {
            "check": "number_gap",
            # A contiguous run of holes is almost always a structuring drop;
            # scattered singletons in a dense sequence are almost always
            # repeals, except in extracted text or beside an unreadable page.
            "severity": (
                "warning" if extracted or likely in {"defect", "unreadable_page"} else "info"
            ),
            "likely": likely,
            "kind": kind,
            "missing": missing,
            "range": [first, last],
            "message": (
                f"{kind} numbers {missing} are absent between {first - 1} and "
                f"{last + 1} (likely {label})."
            ),
        }
        if host:
            finding["eid"] = host
        out.append(finding)
    return out


def _check_leading_gap(root: etree._Element) -> list[dict[str, Any]]:
    """Body numbering that starts above 1, for the basic-unit kinds numbered through
    the document: in extracted text the opening provisions went with a lost page or
    a misread heading. Quoted amendments carry the amended act's numbers, so skipped.
    """
    body = _document_body(root)
    if body is None:
        return []
    out: list[dict[str, Any]] = []
    for kind in ("article", "section"):
        first: tuple[etree._Element, str] | None = None
        for el in body.iter(f"{{{AKN_NS}}}{kind}"):
            num_el = el.find(f"{{{AKN_NS}}}num")
            if num_el is None or _under_quoted_amendment(el):
                continue
            num = normalise_digits("".join(num_el.itertext()).strip().rstrip(".,;:"))
            if num:
                first = (el, num)
                break
        if first is None or not first[1].isdigit() or int(first[1]) <= 1:
            continue
        el, num = first
        eid = el.get("eId") or ""
        if eid and _gap_acknowledged(root, eid):
            continue
        last = int(num) - 1
        finding: dict[str, Any] = {
            "check": "number_gap",
            "severity": "warning",
            "likely": "leading",
            "kind": kind,
            "missing": list(range(1, min(last, 50) + 1)),
            "range": [1, last],
            "message": (
                f"{kind} numbering starts at {num}, so {kind}s 1 to {last} are absent "
                "from the start of the body (likely lost with an opening page)."
            ),
        }
        if eid:
            finding["eid"] = eid
        out.append(finding)
    return out


# Abjad first letter, folded (أ/إ/آ collide); a point run restarting here is the collapse signal.
_FOLDED_ABJAD_FIRST = fold_arabic_for_match("أ")
_FOLDED_BIS = fold_arabic_for_match("مكرر")  # substring of مكررة too


def _has_bis_marker(article: etree._Element) -> bool:
    """The article's own num/heading names it a bis (مكرر / bis) provision."""
    text = " ".join(
        "".join(el.itertext())
        for tag in ("num", "heading")
        if (el := article.find(f"{{{AKN_NS}}}{tag}")) is not None
    )
    return _FOLDED_BIS in fold_arabic_for_match(text) or "bis" in text.lower()


def _direct_child_point_nums(article: etree._Element) -> list[tuple[str, str]]:
    """(eid, folded bare letter/number) per direct-child <point>, in order. Bare means
    digits normalised and parens, dots, whitespace stripped with orthography folded, so
    `(أ)`, `( أ )` and `أ.` all read as the abjad first letter.
    """
    out: list[tuple[str, str]] = []
    for pt in article:
        if not isinstance(pt.tag, str) or pt.tag.split("}", 1)[-1] != "point":
            continue
        eid = pt.get("eId")
        if not eid:
            continue
        num_el = next(
            (c for c in pt if isinstance(c.tag, str) and c.tag.split("}", 1)[-1] == "num"),
            None,
        )
        if num_el is None:
            continue
        raw = normalise_digits("".join(num_el.itertext()))
        bare = fold_arabic_for_match(re.sub(r"[()\[\].,؛:\s]", "", raw))
        if bare:
            out.append((eid, bare))
    return out


def collapsed_bis_scopes(root: etree._Element) -> dict[str, set[str]]:
    """Map each collapsed-bis article eId to the folded letters its collapse duplicates
    (the leading run, up to the second أ), so the duplicate-number check suppresses
    exactly those. Requires a bis (مكرر) marker, so an ordinary flattened lettered run
    keeps its actionable duplicate_number. Arabic abjad only.
    """
    scopes: dict[str, set[str]] = {}
    for article in root.iter(f"{{{AKN_NS}}}article"):
        eid = article.get("eId")
        if not eid or not _has_bis_marker(article):
            continue
        letters = [bare for _, bare in _direct_child_point_nums(article) if bare.isalpha()]
        first_positions = [i for i, letter in enumerate(letters) if letter == _FOLDED_ABJAD_FIRST]
        if len(first_positions) < 2:
            continue
        # Suppress only the leading run (up to the second أ); later duplicates stay visible.
        scopes[eid] = set(letters[: first_positions[1]])
    return scopes


def _check_collapsed_bis_article(scopes: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Flag an article merging two consecutive bis (مكرر) articles. A single article letters
    its sub-points once; a restart such as (أ), (ب), (أ), (ب) means two separately
    lettered provisions were collapsed into one, interleaving their sub-structure.
    Suppressing the generic duplicate_number stops a cosmetic renumber the repair agent
    cannot action; splitting the merged article belongs to the structurer, not here.

    Scope: only the canonical direct-child (أ)-restart on an article marked bis. A second
    run itself mis-lettered, or points wrapped rather than direct children, is missed.
    """
    out: list[dict[str, Any]] = []
    for eid in sorted(scopes):
        out.append(
            {
                "check": "collapsed_bis_article",
                "severity": "error",
                "eid": eid,
                "message": (
                    f"article {eid!r} carries two sub-point runs each restarting at the "
                    "abjad first letter; two consecutive bis (مكرر) articles were merged "
                    "into one element and should be split."
                ),
            }
        )
    return out


# Seam signature: opening span >= _MIN chars repeats after a <= _MAX-char digit-bearing gap.
_MIN_SEAM_SPAN = 40
_MAX_SEAM_GAP = 8


def _seam_repeat_span(text: str) -> int | None:
    """Return the length of the dropped prefix if ``text`` restarts after a
    page break (verbatim opening span + short digit-bearing gap + the span
    again), else None."""
    stripped = re.sub(r"\s+", " ", text).strip()
    n = len(stripped)
    if n < 2 * _MIN_SEAM_SPAN:
        return None
    for j in range(_MIN_SEAM_SPAN, min(n, 800)):
        k = 0
        while j + k < n and stripped[k] == stripped[j + k]:
            k += 1
        if k < _MIN_SEAM_SPAN:
            continue
        gap = stripped[k:j]
        if not (1 <= len(gap) <= _MAX_SEAM_GAP):
            continue
        if any(c.isalpha() for c in gap) or not any(c.isdigit() for c in gap):
            continue
        return j
    return None


def _check_seam_duplication(root: etree._Element) -> list[dict[str, Any]]:
    """Flag an article opening with a page-break restart, where body-fill re-emitted its
    opening span and truncated the first copy, which otherwise ships graded clean.

    Narrow by design: only a restart at the body's first prose char, within ~800 chars,
    with a short digit-bearing gap. Mid-body or gapless repeats are missed. Detect only:
    the auto-delete this replaced removed text that was not a duplicate.
    """
    out: list[dict[str, Any]] = []
    for article in root.iter(f"{{{AKN_NS}}}article"):
        eid = article.get("eId")
        if not eid:
            continue
        # Body text only: the article's own num/heading would misalign the first copy.
        body = "".join(
            "".join(child.itertext())
            for child in article
            if isinstance(child.tag, str) and child.tag.split("}", 1)[-1] not in ("num", "heading")
        )
        if _seam_repeat_span(body) is not None:
            out.append(
                {
                    "check": "seam_duplication",
                    "severity": "warning",
                    "eid": eid,
                    "message": (
                        f"article {eid!r} repeats its opening span verbatim after a short "
                        "gap; a page-break restart the body-fill emitted twice."
                    ),
                }
            )
    return out


# `/akn/ps/act/2005/1/!main`, the component an FRBRthis carries and an
# FRBRuri does not.
_COMPONENT_TAIL = re.compile(r"/![^/]*$")


def _check_uncitable_work_uri(root: etree._Element) -> list[dict[str, Any]]:
    """A work URI nobody can cite. The number falls back to a content address
    when neither the model nor the title yields one, which is a legitimate way
    to finish an ingest and a bad way to leave a law: it cannot be cited,
    resolved, or repaired by the operations that restate identity."""
    work = root.find(f".//{{{AKN_NS}}}identification/{{{AKN_NS}}}FRBRWork")
    if work is None:
        return []
    # FRBRuri is the bare work URI; FRBRthis appends a component (`/!main`),
    # which would shift the year/number window and fault every document.
    uri_el = work.find(f"{{{AKN_NS}}}FRBRuri")
    if uri_el is None:
        uri_el = work.find(f"{{{AKN_NS}}}FRBRthis")
    uri = _COMPONENT_TAIL.sub("", (uri_el.get("value") or "") if uri_el is not None else "")
    if not uri or is_citable_work_uri(uri):
        return []
    return [
        {
            "check": "uncitable_work_uri",
            "severity": "warning",
            "uri": uri,
            "message": (
                f"Work URI {uri!r} carries no citable year and number, so the "
                f"document cannot be cited or resolved by its identity."
            ),
        }
    ]


def _check_anchor_count_mismatch(
    root: etree._Element,
    expected: dict[str, int],
) -> list[dict[str, Any]]:
    """Compare pre-flight anchor counts against AKN element counts. Tolerance is asymmetric:
    undershooting is an error (sections likely dropped), overshooting a warning (the LLM
    may have split a section, or the regex over-counted a TOC that escaped the
    heuristic). Only top-level structural kinds; preface, preamble and conclusions carry
    no anchor signature.
    """
    out: list[dict[str, Any]] = []
    for kind, expected_n in expected.items():
        if expected_n <= 0:
            continue
        # Bluebell parses SCHEDULE into <attachment><doc name="schedule">;
        # no element ever carries the localname "schedule", and counting any
        # attachment would let an unrelated one mask a dropped annex.
        actual_n = 0
        for el in root.iter():
            if not isinstance(el.tag, str) or _under_quoted_amendment(el):
                continue
            localname = el.tag.split("}", 1)[-1]
            if kind == "schedule":
                if localname == "doc" and el.get("name") == "schedule":
                    actual_n += 1
            elif localname == kind:
                actual_n += 1
        if actual_n < expected_n:
            out.append(
                {
                    "check": "anchor_count_mismatch",
                    "severity": "error",
                    "kind": kind,
                    "expected": expected_n,
                    "got": actual_n,
                    "message": (
                        f"Pre-flight scan found {expected_n} '{kind}' anchors in "
                        f"the raw text but the assembled AKN contains only "
                        f"{actual_n}. Structural units the scan saw are missing "
                        f"from the assembled document."
                    ),
                }
            )
        elif actual_n > expected_n * 2:
            out.append(
                {
                    "check": "anchor_count_mismatch",
                    "severity": "warning",
                    "kind": kind,
                    "expected": expected_n,
                    "got": actual_n,
                    "message": (
                        f"AKN contains {actual_n} '{kind}' elements but "
                        f"pre-flight regex saw only {expected_n} in raw text. "
                        f"Possible over-splitting or false-positive in the "
                        f"anchor probe."
                    ),
                }
            )
    return out


# --- Check 1: eId uniqueness ------------------------------------------------


def _check_eid_uniqueness(root: etree._Element) -> list[dict[str, Any]]:
    eids: Counter[str] = Counter()
    for el in root.iter():
        eid = el.get("eId")
        if eid:
            eids[eid] += 1

    return [
        {
            "check": "eid_uniqueness",
            "severity": "error",
            "eid": eid,
            "message": f"Duplicate eId '{eid}' appears {count} times",
        }
        for eid, count in eids.items()
        if count > 1
    ]


# An eId too deep or too long cannot be cited. The ceilings clear real hierarchy
# widely, so only a flat list structured as a chain of children trips them.
EID_MAX_DEPTH = 12
EID_MAX_LENGTH = 255


def _check_eid_unusable(root: etree._Element) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for el in root.iter():
        eid = el.get("eId")
        if not eid:
            continue
        depth = eid.count("__") + 1
        if depth <= EID_MAX_DEPTH and len(eid) <= EID_MAX_LENGTH:
            continue
        out.append(
            {
                "check": "eid_unusable",
                "severity": "error",
                # Full, not truncated: the validation panel uses this as the
                # reader's jump target, and a clipped eId addresses nothing.
                "eid": eid,
                "depth": depth,
                "length": len(eid),
                "message": (
                    f"eId is {'too deep' if depth > EID_MAX_DEPTH else 'too long'}: "
                    f"{depth} levels (ceiling {EID_MAX_DEPTH}), "
                    f"{len(eid)} characters (ceiling {EID_MAX_LENGTH})"
                    + (
                        "; a flat list has most likely structured as a chain of children"
                        if depth > EID_MAX_DEPTH
                        else ""
                    )
                ),
            }
        )
    # One finding per document, not per node: a chain trips the ceiling once per
    # link, and 400 copies of the same defect buries every other finding.
    return out[:1]


# --- Check 2: Cross-reference resolution ------------------------------------


def _check_ref_resolution(root: etree._Element) -> list[dict[str, Any]]:
    all_eids = {el.get("eId") for el in root.iter() if el.get("eId")}
    issues: list[dict[str, Any]] = []

    for ref in root.iter(f"{{{AKN_NS}}}ref"):
        href = ref.get("href", "")
        if not href.startswith("#"):
            continue  # external ref, skip
        target = href[1:]
        if target and target not in all_eids:
            issues.append(
                {
                    "check": "ref_resolution",
                    "severity": "warning",
                    "href": href,
                    "message": f"Internal ref '{href}' has no matching eId in the document",
                }
            )

    return issues


# --- Check 3: Definition completeness ---------------------------------------


def _check_definition_completeness(root: etree._Element) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    # Collect TLCTerm eIds from <references>
    tlc_eids = set()
    for tlc in root.iter(f"{{{AKN_NS}}}TLCTerm"):
        eid = tlc.get("eId")
        if eid:
            tlc_eids.add(f"#{eid}")

    # Every <term refersTo> should have a matching TLCTerm
    for term in root.iter(f"{{{AKN_NS}}}term"):
        rt = term.get("refersTo", "")
        if not rt:
            issues.append(
                {
                    "check": "definition_completeness",
                    "severity": "warning",
                    "message": "<term> missing required refersTo attribute",
                }
            )
            continue
        if rt not in tlc_eids:
            issues.append(
                {
                    "check": "definition_completeness",
                    "severity": "warning",
                    "refersTo": rt,
                    "message": f"<term refersTo='{rt}'> has no matching <TLCTerm> in <references>",
                }
            )

    return issues


# --- Check 5: Hierarchy coherence -------------------------------------------

# Grouping containers and the generic hcontainer legitimately hold provisions
# directly (UK part, EU chapter, UA subchapter), so those are fine. The check still
# catches provisions under a basic unit at the wrong level, or under an
# inline/content wrapper.
_PERMISSIVE_PARENTS = {
    f"{{{AKN_NS}}}{tag}"
    for tag in (
        "hcontainer",
        "part",
        "title",
        "book",
        "chapter",
        "subchapter",
        "section",
        "division",
        "subdivision",
    )
}
_REQUIRES_PARENT = {
    f"{{{AKN_NS}}}subsection": {f"{{{AKN_NS}}}section", f"{{{AKN_NS}}}article"}
    | _PERMISSIVE_PARENTS,
    f"{{{AKN_NS}}}paragraph": {
        f"{{{AKN_NS}}}section",
        f"{{{AKN_NS}}}article",
        f"{{{AKN_NS}}}subsection",
    }
    | _PERMISSIVE_PARENTS,
    f"{{{AKN_NS}}}subparagraph": {f"{{{AKN_NS}}}paragraph", f"{{{AKN_NS}}}subsection"}
    | _PERMISSIVE_PARENTS,
}

# Quoted amendment fragments carry their own (often partial) hierarchy; it is
# not the document's structure, so it is exempt from the coherence check.
_QUOTED_CONTAINERS = (f"{{{AKN_NS}}}quotedStructure", f"{{{AKN_NS}}}embeddedStructure")


# --- Section numbering continuity --------------------------------------------
# Separate from _check_number_set_continuity because this reads the leading digits
# of an alphanumeric label (2A -> 2), so a 1, 2A, 10 sequence still reports the
# 2->10 gap that the digit-set check stops scanning at.
_NUM_RE = re.compile(r"(\d+)")


def _check_section_numbering(root: etree._Element) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    body = root.find(".//akn:body", NS)
    if body is None:
        return issues

    # Check direct section/article children of body and of each part/chapter
    containers = [body]
    for tag in ("part", "chapter"):
        containers.extend(body.iter(f"{{{AKN_NS}}}{tag}"))

    for container in containers:
        # Bucket by element type: sections and articles number independently at body level.
        nums_by_tag: dict[str, list[tuple[int, str]]] = {}
        for child in container:
            tag = child.tag.split("}")[-1]
            if tag not in ("section", "article"):
                continue
            num_el = child.find("akn:num", NS)
            if num_el is None or not num_el.text:
                continue
            m = _NUM_RE.search(num_el.text)
            if m:
                nums_by_tag.setdefault(tag, []).append((int(m.group(1)), child.get("eId", "")))

        # Check for gaps within each bucket (inserted sections like 5A are OK).
        for tag, nums in nums_by_tag.items():
            for i in range(1, len(nums)):
                prev_num, prev_eid = nums[i - 1]
                curr_num, curr_eid = nums[i]
                gap = curr_num - prev_num
                if gap > 2:
                    container_eid = container.get("eId", container.tag.split("}")[-1])
                    issues.append(
                        {
                            "check": "section_numbering",
                            "severity": "info",
                            "container": container_eid,
                            "message": (
                                f"<{tag}> numbering gap in <{container_eid}>: "
                                f"{prev_eid} (num {prev_num}) → {curr_eid} (num {curr_num}), gap of {gap}"
                            ),
                        }
                    )

    return issues


def _check_hierarchy_coherence(root: etree._Element) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for child_tag, valid_parents in _REQUIRES_PARENT.items():
        for el in root.iter(child_tag):
            if any(a.tag in _QUOTED_CONTAINERS for a in el.iterancestors()):
                continue
            parent = el.getparent()
            if parent is not None and parent.tag not in valid_parents:
                child_name = child_tag.split("}")[-1]
                parent_name = parent.tag.split("}")[-1]
                eid = el.get("eId", "")
                issues.append(
                    {
                        "check": "hierarchy_coherence",
                        "severity": "warning",
                        "eid": eid,
                        "message": f"<{child_name}> (eId={eid}) has unexpected parent <{parent_name}>",
                    }
                )

    return issues


# --- Check 6: LLM body-fill artefacts in body text ---------------------------

# LLM meta-output bleeding into body text: markdown fences, the JSON-fragment tails
# `]}]` and `}]}`, and stop-token failures like "as valid JSON". None appears in
# legitimate prose. One occurrence anywhere in a <p> is enough, so a trailing
# artefact does not slip past a tail-only check.
_LEAKAGE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"`{3,}"),
    re.compile(r"\]\s*\}\s*\]"),
    re.compile(r"\}\s*\]\s*\}"),
    re.compile(r"\bas valid (?:single[- ]line )?JSON\b", re.IGNORECASE),
    re.compile(r"\b(?:more precisely|to be precise),?\s+", re.IGNORECASE),
)


# Names a drafter never writes as a whole provision. `article`, `section`, `chapter`,
# `paragraph` and `clause` are absent on purpose: "Article 2" is a heading.
_STRUCTURAL_ELEMENT_NAMES = (
    "hcontainer",
    "blockList",
    "blockContainer",
    "crossHeading",
    "embeddedStructure",
    "authorialNote",
    "subFlow",
    "longTitle",
    "amendmentList",
    "portionBody",
    "tblock",
    "quotedStructure",
    "quotedText",
    "listIntroduction",
    "listWrapUp",
    "mainBody",
    "amendmentBody",
    "debateBody",
    "judgmentBody",
    "debateSection",
    "componentRef",
    "documentRef",
    "eventRef",
    "temporalGroup",
    "wrapUp",
)

# Whole text, not a match within it. One separator per repetition: two would let a
# space belong to either side, and a failing long repeat explores every partition.
_MARKUP_AS_TEXT = re.compile(
    r"^(?:(?:" + "|".join(_STRUCTURAL_ELEMENT_NAMES) + r")(?:\s+\d+)?\s*)+$",
    re.IGNORECASE,
)


def _strip_format_marks(text: str) -> str:
    """RTL OCR sprinkles invisible marks and one anywhere defeats an anchored match.
    Stripped by category so a mark nobody listed still counts, then trimmed again
    because removing a leading mark leaves the space that was behind it."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf").strip()


def _check_markup_as_text(root: etree._Element) -> list[dict[str, Any]]:
    """A paragraph whose whole text is markup vocabulary: the model emitting its own
    scaffolding as content. Matched whole rather than searched, so a heading naming
    an article is not taken for one. Per paragraph rather than per provision, since
    grouping paragraphs into provisions is the storage mapper's job, not this one's.
    """
    issues: list[dict[str, Any]] = []
    for p in root.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext()).strip()
        if not text or not _MARKUP_AS_TEXT.match(_strip_format_marks(text)):
            continue
        ancestor = p.getparent()
        while ancestor is not None and not ancestor.get("eId"):
            ancestor = ancestor.getparent()
        anchor_eid = ancestor.get("eId") if ancestor is not None else ""
        shown = text if len(text) <= 60 else f"{text[:60]}…"
        issues.append(
            {
                "check": "markup_as_text",
                "severity": "error",
                "eid": anchor_eid,
                "message": (
                    f"A paragraph reads {shown!r}, which names parts of a document "
                    f"rather than saying anything. Whatever it held was replaced by "
                    f"the words used to describe its shape."
                ),
            }
        )
    return issues


def _check_body_artefacts(root: etree._Element) -> list[dict[str, Any]]:
    """Detect LLM meta-output leaked into body text: JSON fence fragments (``]}]``, ``}]}``),
    markdown fences, and explanatory tails inside <p>. Duplicate and gap checks operate
    on numbering rather than text, so nothing else surfaces this. One match anywhere in
    the paragraph is enough, none of these fragments meaning anything in legal prose.
    """
    issues: list[dict[str, Any]] = []
    for p in root.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext()).strip()
        if not text:
            continue
        match = next((m for pat in _LEAKAGE_PATTERNS if (m := pat.search(text))), None)
        if match is None:
            continue
        ancestor = p.getparent()
        while ancestor is not None and not ancestor.get("eId"):
            ancestor = ancestor.getparent()
        anchor_eid = ancestor.get("eId") if ancestor is not None else ""
        snippet = text[max(0, match.start() - 20) : match.end() + 20]
        issues.append(
            {
                "check": "body_artefact",
                "severity": "error",
                "eid": anchor_eid,
                "message": (
                    f"Body text under {anchor_eid!r} contains LLM-output "
                    f"artefact near {snippet!r}. Likely body-fill stop-token failure."
                ),
            }
        )
    return issues


# --- Check 7: Orphan articles at document root ------------------------------


# A contiguous leading run of root articles is the faithful "general provisions
# before Chapter One" shape, and the rank-stack structurer can only orphan a leading
# prefix. Below this fraction grades warning; interspersed articles or a larger
# fraction (Customs Code: 99 beside 9+9) stay error. Reparenting would mis-place
# genuine general provisions and, for a dropped heading, fabricate a container.
_ORPHAN_LEADING_RATIO = 0.20


def _check_orphan_articles(root: etree._Element) -> list[dict[str, Any]]:
    """When a document declares sections or chapters, articles must live inside them rather
    than beside the containers at <body> root.

    A small contiguous leading run of root articles (general provisions before the first
    container) is faithful and grades ``warning``. Interspersed root articles, or a large
    fraction of all articles at root (the Customs Code failure: 99 articles beside 9
    sections and 9 chapters), stay ``error``. Fires only when containers and root
    articles coexist, since a document with none legitimately has articles at root.
    """
    body = next(iter(root.iter(f"{{{AKN_NS}}}body")), None)
    if body is None:
        return []
    direct_articles: list[etree._Element] = []
    article_positions: list[int] = []
    first_container_pos: int | None = None
    for pos, child in enumerate(body):
        if not isinstance(child.tag, str):
            continue
        tag = child.tag.split("}", 1)[-1]
        if tag == "article":
            direct_articles.append(child)
            article_positions.append(pos)
        elif tag in ("section", "chapter", "part", "title", "book"):
            if first_container_pos is None:
                first_container_pos = pos
    if not (direct_articles and first_container_pos is not None):
        return []
    # Scope the denominator to this body's own articles: a quoted amendment
    # borrows another act's structure, and counting it would shrink the fraction
    # and pass off a malformed body as a small leading run.
    total_articles = sum(
        1
        for a in body.iter(f"{{{AKN_NS}}}article")
        if not any(anc.tag in _QUOTED_CONTAINERS for anc in a.iterancestors())
    )
    ratio = len(direct_articles) / total_articles if total_articles else 1.0
    leading = all(pos < first_container_pos for pos in article_positions)
    faithful = leading and ratio <= _ORPHAN_LEADING_RATIO
    sample = [a.get("eId", "") for a in direct_articles[:5]]
    if faithful:
        message = (
            f"{len(direct_articles)} <article> elements form a leading run before "
            f"the first container (general-provisions shape); nesting is worth a "
            f"review but the text is present. Sample eIds: {sample}."
        )
    else:
        shape = "interspersed with" if not leading else "a large fraction beside"
        message = (
            f"{len(direct_articles)} of {total_articles} <article> elements sit "
            f"directly under <body>, {shape} container elements "
            f"(section/chapter/part). Likely structurer container-assembly "
            f"failure, articles should be inside the containers. Sample eIds: {sample}."
        )
    return [
        {
            "check": "orphan_articles",
            "severity": "warning" if faithful else "error",
            "count": len(direct_articles),
            # Complete, not a sample: the repair scope gate reads this to decide
            # what a plan may touch, and a truncated list refuses the tail of a
            # legitimate fix. The message keeps its short sample for humans.
            "eids": [a.get("eId", "") for a in direct_articles],
            "message": message,
        }
    ]


# Measured over 16,629 versions: 274 one-provision bodies sit under 5,000 chars
# and are legitimately flat; 10 sit above 10,000 and all should have structure.
_STRUCTURELESS_BODY_CHARS = 10_000
# A stub body carries a heading or a stray line, never a provision's worth of text.
_EMPTY_BODY_CHARS = 200


def _own_prose_length(body: etree._Element) -> int:
    """Body prose the act is responsible for, whitespace collapsed. Quoted amendments are
    subtracted because `count_provisions_by_kind` already skips their provisions:
    counting their text but not their structure reads a correctly-structured amending act
    as one long provision.
    """

    def squashed(el: etree._Element) -> int:
        return len(" ".join(provision_text(el).split()))

    quoted = sum(
        squashed(el)
        for el in body.iter()
        if isinstance(el.tag, str)
        and local_name(el.tag) in QUOTED_AMENDMENT_ANCESTORS
        # Outermost quote only, else a nested one is subtracted twice.
        and not any(local_name(a.tag) in QUOTED_AMENDMENT_ANCESTORS for a in el.iterancestors())
    )
    return squashed(body) - quoted


# A body the caller referred out to an annex is a line or two, not a document.
# Above this the body is substantial in its own right and is judged as one.
_REFERRING_BODY_CHARS = 1_000

# `doc` and `judgment` roots name their body differently; a document carrying none
# of them has no body at all.
_BODY_TAGS = ("body", "mainBody", "judgmentBody")


def _document_body(root: etree._Element) -> etree._Element | None:
    """The document's own body, whichever of the three names it carries. An
    attachment has a `mainBody` of its own, so one nested in an annex is skipped."""
    for tag in _BODY_TAGS:
        for el in root.iter(f"{{{AKN_NS}}}{tag}"):
            if not any(local_name(a.tag) == "attachment" for a in el.iterancestors()):
                return el
    return None


def _attachments_carry_content(root: etree._Element) -> bool:
    """Whether an annex holds anything. Presence of the element is not enough: an
    empty one would exempt a document that holds nothing anywhere."""
    return any(
        any(count_provisions_by_kind(att).values()) or _own_prose_length(att) > _EMPTY_BODY_CHARS
        for att in root.iter(f"{{{AKN_NS}}}attachment")
    )


def _check_empty_body(root: etree._Element) -> list[dict[str, Any]]:
    """A body holding no provisions and no prose. `structureless_body` cannot see it:
    that check requires substantial text, because it describes text that survived
    without structure. Here nothing survived, so it is an error rather than a warning.
    """
    body = _document_body(root)
    # No body element at all is the same defect, not an exemption from it.
    if body is not None and any(count_provisions_by_kind(body).values()):
        return []
    chars = _own_prose_length(body) if body is not None else 0
    # An annex is a sibling of the body, so a schedule-bearing act whose body is one
    # referring line still carries its content. Annexes are inlined here rather than
    # referenced, so that shape is ordinary.
    if chars <= _REFERRING_BODY_CHARS and _attachments_carry_content(root):
        return []
    if chars > _EMPTY_BODY_CHARS:
        return []
    return [
        {
            "check": "empty_body",
            "severity": "error",
            "chars": chars,
            "message": (
                "The document has no content: it holds nothing divided into articles, "
                f"sections or paragraphs, and only {chars:,} characters of text. Beyond "
                "the title and identifying details there is nothing to read, cite or "
                "compare."
            ),
        }
    ]


def _check_structureless_body(root: etree._Element) -> list[dict[str, Any]]:
    """A long body holding one provision has no structure, and so contradicts
    nothing that the other checks look for. Without this it grades clean.

    Warning, not error: the text all survives, it just cannot be navigated.
    """
    body = _document_body(root)
    if body is None:
        return []
    provisions = sum(count_provisions_by_kind(body).values())
    if provisions > 1:
        return []
    chars = _own_prose_length(body)
    # The higher floor is for the one-provision case, where a short document
    # legitimately has a single article. A provisionless body is unstructured at any
    # length, unless an annex carries the content and this is the line referring to it.
    referred_out = chars <= _REFERRING_BODY_CHARS and _attachments_carry_content(root)
    floor = _EMPTY_BODY_CHARS if provisions == 0 and not referred_out else _STRUCTURELESS_BODY_CHARS
    if chars <= floor:
        return []
    return [
        {
            "check": "structureless_body",
            "severity": "warning",
            "chars": chars,
            # Says only what the AKN shows. Why the text was never divided up, and
            # whether any of it is missing, are not knowable from here.
            "message": (
                f"The whole document sits in a single block of {chars:,} characters, "
                f"with no chapters, articles or sections inside it. Until it is "
                f"divided up, the document cannot be navigated, cited by provision, "
                f"or compared against another text."
            ),
        }
    ]


# --- Check 8: OCR garble in Arabic body text --------------------------------

# Extraction artefacts body-fill passes through unchanged. Repeated ``ة``
# (ta-marbuta, U+0629) is the signature on modern PS Arabic (``الآتيةة``): the letter
# never doubles legitimately, gemination using shadda rather than repetition.
# Detect-only, since fixing Arabic morphology without an Arabic-aware layer risks
# worse damage than the noise.
_OCR_GARBLE_PATTERNS = ARABIC.garble_patterns


def _check_ocr_garble(root: etree._Element) -> list[dict[str, Any]]:
    """Detect OCR mojibake the body-fill LLM passed through unchanged, such as the
    duplicated ta-marbuta the modern-Arabic re-ingests leak from the PDF extraction
    layer. Surfaced as candidates for human review; the fix belongs in extraction or
    body-fill normalisation.
    """
    issues: list[dict[str, Any]] = []
    by_anchor: dict[str, int] = {}
    samples: dict[str, str] = {}
    for p in root.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext())
        if not text:
            continue
        for pat in _OCR_GARBLE_PATTERNS:
            for m in pat.finditer(text):
                anc = p.getparent()
                while anc is not None and not anc.get("eId"):
                    anc = anc.getparent()
                eid = anc.get("eId") if anc is not None else ""
                by_anchor[eid] = by_anchor.get(eid, 0) + 1
                if eid not in samples:
                    samples[eid] = text[max(0, m.start() - 15) : m.end() + 15]
    for eid, n in by_anchor.items():
        issues.append(
            {
                "check": "ocr_garble",
                "severity": "warning",
                "eid": eid,
                "count": n,
                "message": (
                    f"{n} OCR-garble pattern(s) under {eid!r} (e.g. "
                    f"{samples[eid]!r}). Likely PDF text-extraction mojibake."
                ),
            }
        )
    return issues


# --- Check 9: Empty articles -------------------------------------------------


def _check_empty_articles(root: etree._Element) -> list[dict[str, Any]]:
    """Articles carrying only a `<num>` and perhaps a `<heading>`, with no `<content>`,
    `<intro>` or descendant `<p>` text. PS Criminal Code 1936 had three appended after
    art_391 that the structurer recovered as fragments but could not place. A genuinely
    repealed article usually carries a short repeal notice, so empty and headless is
    almost always a pipeline drop rather than source intent.
    """
    issues: list[dict[str, Any]] = []
    for art in root.iter(f"{{{AKN_NS}}}article"):
        eid = art.get("eId", "")
        # Concatenate all descendant text content except <num> and <heading>.
        body_chars = 0
        for desc in art.iter():
            if not isinstance(desc.tag, str):
                continue
            local = desc.tag.split("}", 1)[-1]
            if local in ("num", "heading", "article"):
                continue
            if desc.text:
                body_chars += len(desc.text.strip())
        if body_chars == 0:
            issues.append(
                {
                    "check": "empty_article",
                    "severity": "warning",
                    "eid": eid,
                    "message": (
                        f"Article {eid!r} has a <num> but no body content. "
                        f"Likely a structurer fragment-recovery drop, the "
                        f"source text the LLM saw didn't land under this anchor."
                    ),
                }
            )
    return issues


# --- Check 10: Swallowed enumerators inside sibling body text ---------------

# The Arabic alphabetic-enumerator sequence as it appears in legal lists.
# Each entry is paired with its canonical position (1-based) so we can detect
# gaps in either Arabic-letter, Latin-letter, or Roman-numeral sequences.
_ARABIC_LETTER_ORDER = ARABIC.letter_order
# Each item also has standalone forms used as enumerators:
#   هـ ↔ ه, both render as "haa"; the legal-style is usually هـ with kasf
_ARABIC_LETTER_ALIAS = ARABIC.letter_alias


def _alpha_ar_index(token: str) -> int | None:
    """Position (1-based) of an Arabic letter token in the legal-enum order."""
    tok = _ARABIC_LETTER_ALIAS.get(token, token).strip("()")
    # Strip trailing ـ kashida and leading separators
    bare = tok.rstrip("ـ").strip()
    for i, letter in enumerate(_ARABIC_LETTER_ORDER):
        if bare == letter or bare == f"{letter}ـ":
            return i + 1
    return None


_LATIN_LETTER_RE = re.compile(r"^\(?([A-Za-z])\)?$")
_ARABIC_DIGIT_RE = ARABIC.digit_re


def _check_swallowed_enumerators(root: etree._Element) -> list[dict[str, Any]]:
    """A sibling enumerator absent as a structural element but present as literal text
    inside the preceding sibling's body (the art_٣٨٠ pattern).

    Signature: the parent's direct `<point>` children form an incomplete same-family
    sequence (``(أ)(ب)(ج)(د)(و)``, missing ``(هـ)``) and one child's body contains the
    missing enumerator literally. Body-fill emitted the structure with an item
    swallowed; the deterministic structurer never saw inline enumerators it could nest.
    """
    issues: list[dict[str, Any]] = []
    for parent in root.iter():
        if not isinstance(parent.tag, str):
            continue
        children = [c for c in parent if isinstance(c.tag, str) and c.tag.split("}")[-1] == "point"]
        if len(children) < 3:
            continue
        # Collect (num, index, child), only same-family-Arabic-letter for now.
        positions: list[tuple[int, etree._Element]] = []
        for c in children:
            num_el = next(
                (cc for cc in c if isinstance(cc.tag, str) and cc.tag.split("}")[-1] == "num"),
                None,
            )
            if num_el is None:
                continue
            num_text = "".join(num_el.itertext()).strip()
            idx = _alpha_ar_index(num_text)
            if idx is None:
                # mixed family, skip parent
                positions = []
                break
            positions.append((idx, c))
        if len(positions) < 3:
            continue
        # Look for gaps (consecutive indices should be sequential)
        indices = [p[0] for p in positions]
        if indices != sorted(indices):
            # out-of-order, different problem, leave alone
            continue
        for i in range(len(indices) - 1):
            gap_size = indices[i + 1] - indices[i]
            if gap_size <= 1:
                continue
            # Look for each missing letter in the previous child's body text
            preceding = positions[i][1]
            preceding_text = "".join(preceding.itertext())
            for missing_idx in range(indices[i] + 1, indices[i + 1]):
                missing_letter = _ARABIC_LETTER_ORDER[missing_idx - 1]
                # Look for "(letter)" or "(letterـ)" inline in the preceding text.
                pat = re.compile(rf"\(\s*{missing_letter}ـ?\s*\)")
                match = pat.search(preceding_text)
                if match:
                    parent_eid = parent.get("eId", "")
                    preceding_eid = preceding.get("eId", "")
                    issues.append(
                        {
                            "check": "swallowed_enumerator",
                            "severity": "warning",
                            "eid": preceding_eid,
                            "missing": f"({missing_letter})",
                            # The surface as it appears in the body, what a
                            # Split's marker search must be given, since the
                            # canonical form may not occur literally.
                            "matched": match.group(0),
                            "parent": parent_eid,
                            "message": (
                                f"Enumerator '({missing_letter})' is absent from {parent_eid!r}'s "
                                f"point sequence but appears inline inside {preceding_eid!r}'s "
                                f"body text. Body-fill LLM swallowed the next sibling into the "
                                f"previous point's prose."
                            ),
                        }
                    )
    return issues


def _check_cover_body_reconciliation(
    root: etree._Element, expected_cover_article_numbers: list[int]
) -> list[dict[str, Any]]:
    """Compare the cover TOC's article set size against the emitted body.
    Delegates the divergence classification to ``cover_reconciliation`` so
    the tolerance policy lives with the extractor."""
    from codify.pipeline.enrich.cover_reconciliation import reconcile_cover_vs_body

    body_articles = len(root.findall(f".//{{{AKN_NS}}}article"))
    issue = reconcile_cover_vs_body(expected_cover_article_numbers, body_articles)
    return [issue] if issue else []


# Widest single jump the amendment detector's plural budget allows. A larger
# forward jump on top-level article numbers without an amendment cue means either
# an undetected quoted amendment or a lost anchor range.
_MAX_STEP_WITHOUT_CUE = 8


_QUOTED_AMENDMENT_ANCESTORS = QUOTED_AMENDMENT_ANCESTORS


def _under_quoted_amendment(el: etree._Element) -> bool:
    """True when ``el`` sits inside a mod or quoted-structure wrapper. Used
    to exclude embedded amendment articles from top-level continuity checks."""
    for ancestor in el.iterancestors():
        if not isinstance(ancestor.tag, str):
            continue
        localname = ancestor.tag.split("}", 1)[-1]
        if localname in _QUOTED_AMENDMENT_ANCESTORS:
            return True
    return False


def _check_undetected_amendment_shape(root: etree._Element) -> list[dict[str, Any]]:
    """Warn when a container's direct article children go strictly non-monotonic or
    overshoot ``_MAX_STEP_WITHOUT_CUE``, which the amendment detector misses when its
    trigger-phrase catalogue is incomplete (the measured `1, 2, 4, 5, 3, 6` shape).

    Grouped by direct parent, so an act whose numbering restarts per chapter is not
    flagged. Articles inside a `<mod>` or `<quotedStructure>` wrapper are skipped, so
    correctly-marked amendments stay quiet.
    """
    seen: dict[etree._Element, list[tuple[str, int]]] = {}
    for art in root.iter(f"{{{AKN_NS}}}article"):
        if _under_quoted_amendment(art):
            continue
        num_el = art.find(f"{{{AKN_NS}}}num")
        if num_el is None or not num_el.text:
            continue
        folded = normalise_digits(num_el.text).strip()
        if not folded.isdigit():
            continue
        parent = art.getparent()
        if parent is None:
            continue
        seen.setdefault(parent, []).append((art.get("eId") or "", int(folded)))

    issues: list[dict[str, Any]] = []
    for group in seen.values():
        for i in range(1, len(group)):
            prev_eid, prev = group[i - 1]
            eid, current = group[i]
            if current <= prev:
                kind, reason = "non_monotonic", f"went non-monotonic at {eid} ({prev} to {current})"
            elif current - prev > _MAX_STEP_WITHOUT_CUE:
                kind, reason = "step_over_ceiling", f"jumped from {prev} to {current} at {eid}"
            else:
                continue
            issues.append(
                {
                    "check": "undetected_amendment_shape",
                    "severity": "warning",
                    "shape": kind,
                    "eid": eid,
                    "previous_eid": prev_eid,
                    "previous_number": prev,
                    "number": current,
                    "message": (
                        f"article numbering {reason}; possible undetected amendment "
                        "quoted-structure or lost anchor range"
                    ),
                }
            )
    return issues


# Containers whose presentation leads with a descriptive title, shared with the
# capture pass. `section` joins only when the document also carries articles: it is
# a grouping level there (as `РОЗДІЛ` is), but in article-less doctypes (gb acts,
# SIs) it is the basic unit, whose heading rides body-fill and may be absent.
_TITLED_CONTAINER_KINDS = tuple(sorted(CONTAINER_KINDS))


def check_missing_container_titles(root: etree._Element) -> list[dict[str, Any]]:
    """Warn when a container has no descriptive `<heading>`, which the reader shows as a
    bare marker ("Chapter 5") that reads as a structural gap. The source may genuinely
    lack the title, so this routes the call to an operator rather than shipping silently.
    """
    kinds = _TITLED_CONTAINER_KINDS
    if root.find(f".//{{{AKN_NS}}}article") is not None:
        kinds = kinds + ("section",)
    issues: list[dict[str, Any]] = []
    for localname in kinds:
        for el in root.iter(f"{{{AKN_NS}}}{localname}"):
            heading = el.find(f"{{{AKN_NS}}}heading")
            heading_text = "".join(heading.itertext()).strip() if heading is not None else ""
            num_el = el.find(f"{{{AKN_NS}}}num")
            num = (num_el.text or "").strip() if num_el is not None else ""
            # A heading repeating its own num ("Chapter One / One") is the
            # bare-ordinal shape: no title, but it looks deliberate, so it
            # gets its own finding rather than hiding among absent headings.
            if heading_text and heading_restates_number(heading_text, num):
                issues.append(
                    {
                        "check": "container_heading_restates_num",
                        "severity": "warning",
                        "eid": el.get("eId") or "",
                        "container": localname,
                        "message": (
                            f"{localname} {num or '<no num>'} heading "
                            f"{heading_text!r} only restates its own number"
                        ),
                    }
                )
                continue
            if heading_text:
                continue
            issues.append(
                {
                    "check": "missing_container_title",
                    "severity": "warning",
                    "eid": el.get("eId") or "",
                    "container": localname,
                    "message": (
                        f"{localname} {num or '<no num>'} has no descriptive heading; "
                        "the reader shows a bare marker"
                    ),
                }
            )
    return issues


# --- Check: fabricated enacting formula --------------------------------------


def _check_fabricated_formula(root: etree._Element, source_text: str) -> list[dict[str, Any]]:
    """Flag enacting formulae absent from the raw structuring text, meaning fabricated
    enacting language. Formulae with refersTo="#codify" are the era-gated config fallback
    injected only when the source carries none, and are skipped on that provenance. Both
    sides are digit-normalised and orthography-folded, so faithful transcription that
    normalised alef or diacritics is not flagged.
    """
    folded_source = fold_arabic_for_match(normalise_digits(source_text))
    issues: list[dict[str, Any]] = []
    for fml in root.iter(f"{{{AKN_NS}}}formula"):
        if fml.get("name") != "enactingFormula":
            continue
        if fml.get("refersTo") == FORMULA_PROVENANCE:
            continue
        raw = re.sub(r"\s+", " ", "".join(fml.itertext())).strip()
        text = fold_arabic_for_match(normalise_digits(raw))
        if not text or text in folded_source:
            continue
        issues.append(
            {
                "check": "fabricated_formula",
                "severity": "error",
                "eid": fml.get("eId") or "",
                "message": (
                    f"Enacting formula {raw[:80]!r} does not appear in the source "
                    "text; fabricated enacting language the document never carried"
                ),
            }
        )
    return issues


# Below this many source headers a short order carries too few to trust; above
# this quoted share the markers live in an amendment's QUOTE blocks, not its own
# body, so the source header count is not the article count.
_HEADER_COVERAGE_MIN = 6
_HEADER_COVERAGE_QUOTED_SHARE = 0.4
_HEADER_COVERAGE_FLOOR = 0.9


def _check_header_coverage(root: etree._Element, source_text: str) -> list[dict[str, Any]]:
    """Source claims more article headers than the document carries. Detection only.

    The ingest coverage gate derives its expected count from the same scan it grades, so
    a systemic drop suppresses both and ships at ~100%. This counts headers independently
    (digit-agnostic, quote-masked) and flags the shortfall the gate cannot see. Non-Arabic
    sources carry no `مادة` and abstain.
    """
    # version_source_texts stores the raw extract; the structurer normalises it
    # internally, so mirror that before counting or born-digital headers read as
    # absent.
    from codify.pipeline.enrich.structure import normalise_rtl_extract

    clean, quoted = arabic_article_headers(normalise_rtl_extract(source_text))
    total = clean + quoted
    if total < _HEADER_COVERAGE_MIN or quoted / total >= _HEADER_COVERAGE_QUOTED_SHARE:
        return []
    articles = sum(1 for _ in root.iter(f"{{{AKN_NS}}}article"))
    if articles >= _HEADER_COVERAGE_FLOOR * clean:
        return []
    return [
        {
            "check": "header_coverage",
            "severity": "warning",
            "message": (
                f"source carries {clean} article headers but the document has "
                f"{articles}; {clean - articles} appear dropped in structuring"
            ),
        }
    ]


# A real page yields ~30-50k chars_per_ink; below this floor it had ink and produced
# no text, meaning a failed OCR read. accept_ratio is not used: a diverted page still
# yields text (43% of the corpus sits below 0.9 accept), whereas a zero-yield page is
# rare (~2.4%) and real.
_PAGE_YIELD_FLOOR = 500.0
# Fraction of the rival's content tokens that must appear in the AKN body for an
# empty-read page to count as recoverable. Calibrated on ps/act/1863 pages 69 and 71,
# which measured 0.66 and 0.72, so 0.5 clears them with margin while excluding a page
# whose overlap is only shared boilerplate.
_PAGE_YIELD_CONTAINMENT_MIN = 0.5
_AR_DIACRITICS = re.compile(r"[ً-ْـ]")


def _content_tokens(s: str) -> set[str]:
    """Distinctive word tokens: NFKC-folded (OCR emits Arabic presentation forms),
    diacritics and tatweel stripped, function words and single characters dropped,
    so overlap reflects content rather than shared legal boilerplate."""
    s = _AR_DIACRITICS.sub("", unicodedata.normalize("NFKC", s)).lower()
    return {
        w for w in re.findall(r"\w+", s) if len(w) >= 2 and w not in ARABIC_WORDS.function_words
    }


def recoverable_empty_pages(akn_xml: str, page_rival_texts: dict[int, str]) -> set[int]:
    """Empty-read pages whose content the AKN actually carries. Recoverable requires the
    rival's text for that page to be present in the AKN body, not merely that a rival
    read something: a legacy version whose refused page never reached the AKN must stay
    blocking. Parse failure or no rivals returns empty, so the default is to block.
    """
    if not page_rival_texts:
        return set()
    try:
        root = parse_xml(akn_xml)
    except etree.XMLSyntaxError:
        return set()
    akn_tokens = _content_tokens(" ".join(root.itertext()))
    recoverable = set()
    for page, rival in page_rival_texts.items():
        rt = _content_tokens(rival)
        if rt and len(rt & akn_tokens) / len(rt) >= _PAGE_YIELD_CONTAINMENT_MIN:
            recoverable.add(page)
    return recoverable


def _page_yield_finding(pages: list[int], total: int, severity: str, tail: str) -> dict[str, Any]:
    plural = "s" if len(pages) > 1 else ""
    shown = ", ".join(str(p) for p in pages[:8]) + (", …" if len(pages) > 8 else "")
    return {
        "check": "page_yield",
        "severity": severity,
        "message": (
            f"{len(pages)} of {total} source pages the primary engine read empty "
            f"(failed OCR at page{plural} {shown}); {tail}"
        ),
    }


def _check_page_yield(
    page_yield: list[tuple[int, float]],
    recoverable_pages: set[int] | None = None,
) -> list[dict[str, Any]]:
    """A source page the primary engine read empty: blocking when the content is lost,
    warning when the AKN carries it (see ``recoverable_empty_pages``).

    ``page_yield`` is (page_number, chars_per_ink) so the message names the real page
    rather than a list position, blank pages being absent and row order guaranteed only
    by the caller's ORDER BY. ``recoverable_pages`` is the subset the caller verified
    against the AKN; a page absent from it stays blocking.
    """
    recoverable = recoverable_pages or set()
    failed = [pg for pg, cpi in page_yield if cpi < _PAGE_YIELD_FLOOR]
    if not failed:
        return []
    lost = [pg for pg in failed if pg not in recoverable]
    recovered = [pg for pg in failed if pg in recoverable]
    issues: list[dict[str, Any]] = []
    if lost:
        # Blocking: the content is gone. The delivery gate (which ships
        # clean/warning) must exclude it.
        issues.append(
            _page_yield_finding(
                lost,
                len(page_yield),
                "error",
                "neither engine read them, so the document is likely missing that content",
            )
        )
    if recovered:
        issues.append(
            _page_yield_finding(
                recovered,
                len(page_yield),
                "warning",
                "the AKN carries their content from the other engine, so this is recoverable",
            )
        )
    return issues


def _check_orphaned_drops(orphaned_drops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """An anchor pass removed a marker and left the text it governed behind. Only the
    scanner can see it: once the AKN exists the text has been absorbed by whatever
    preceded the drop, and a preamble holding definitions is indistinguishable from a
    long preamble. Warning, since every character survives, but provisions have left the
    body so it must not read as clean.
    """
    if not orphaned_drops:
        return []
    worst = max(orphaned_drops, key=lambda d: int(d.get("orphaned_chars") or 0))
    total = sum(int(d.get("orphaned_chars") or 0) for d in orphaned_drops)
    passes = sorted({str(d.get("dropped_by") or "unknown") for d in orphaned_drops})
    # The scanner reports no ratio when the drops left it no undisturbed gap to
    # measure against, which is the case where they took nearly everything.
    ratio = worst.get("ratio")
    against = (
        f"{ratio}x this document's median anchor gap"
        if ratio is not None
        else "with no undisturbed anchor gap left to compare against"
    )
    return [
        {
            "check": "orphaned_drops",
            "severity": "warning",
            "count": len(orphaned_drops),
            "message": (
                f"{len(orphaned_drops)} anchor drop(s) left {total:,} characters without a "
                f"provision to hold them (worst {int(worst.get('orphaned_chars') or 0):,} "
                f"characters, {against}, dropped by {worst.get('dropped_by')}); that text "
                f"falls to whatever precedes the drop, which for the first anchor is the "
                f"preamble. Passes involved: {', '.join(passes)}."
            ),
        }
    ]


def _check_unreadable_pages(pages: dict[int, str]) -> list[dict[str, Any]]:
    """Source pages that came back with no text: a content-filter refusal or an inked
    page read empty. Whatever they held is missing from the AKN, so this blocks."""
    ordered = sorted(pages)
    shown = ", ".join(f"{p} ({pages[p]})" for p in ordered[:8])
    more = ", …" if len(ordered) > 8 else ""
    return [
        {
            "check": "unreadable_page",
            "severity": "error",
            "pages": ordered,
            "reasons": {str(p): pages[p] for p in ordered},
            "message": (
                f"{len(ordered)} source page(s) came back with no text: {shown}{more}; "
                "their content is missing from the document."
            ),
        }
    ]


def _check_body_fill(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Provisions whose body the model never wrote. Empty ones lost their text, so
    error; source text copied in unstructured keeps every word, so warning."""
    empty = list(trace.get("empty") or [])
    verbatim = list(trace.get("verbatim") or [])
    expected = int(trace.get("expected") or 0)
    failed = int(trace.get("calls_failed") or 0)
    calls = int(trace.get("calls") or 0)
    issues: list[dict[str, Any]] = []
    if empty:
        issues.append(
            {
                "check": "body_fill_incomplete",
                "severity": "error",
                "eids": empty,
                "count": len(empty),
                "message": (
                    f"{len(empty)} of {expected} provisions with source text have no body "
                    f"({failed} of {calls} body-fill calls failed); first {empty[0]!r}."
                ),
            }
        )
    if verbatim:
        issues.append(
            {
                "check": "body_fill_verbatim",
                "severity": "warning",
                "eids": verbatim,
                "count": len(verbatim),
                "message": (
                    f"{len(verbatim)} of {expected} provision bodies are source text copied "
                    f"in unstructured after the model did not fill them ({failed} of {calls} "
                    f"body-fill calls failed); first {verbatim[0]!r}."
                ),
            }
        )
    return issues


def _check_container_coverage(probe: dict[str, Any]) -> list[dict[str, Any]]:
    """Source grouping headings (Bab/Fasl) that mostly produced no container, meaning the
    structurer flattened a grouping level, as the PS cabinet-decision config gap did
    before its marker was added.

    Only the scan can see this, so it arrives as evidence. Warning, not error: the
    detector is deliberately config-independent and retains false positives, and blocking
    on a heuristic would exclude readable laws. Guards live in ``container_coverage_probe``.
    """
    from codify.pipeline.enrich.container_coverage import container_coverage_below_floor

    if not container_coverage_below_floor(probe):
        return []
    present = int(probe.get("present") or 0)
    found = int(probe.get("found") or 0)
    return [
        {
            "check": "container_coverage",
            "severity": "warning",
            "count": present - found,
            "message": (
                f"{present} grouping heading(s) in the source produced only {found} "
                f"container(s); {present - found} were flattened into the article run. "
                "The structurer did not recognise this grouping level, so the reader "
                "loses the document's chapter/part structure."
            ),
        }
    ]


_ARABIC_LETTERS = re.compile(r"[؀-ۿ]")
_HEBREW_LETTERS = re.compile(r"[֐-׿]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")

_SCRIPT_BY_LANGUAGE = {"ara": _ARABIC_LETTERS, "heb": _HEBREW_LETTERS}


def _check_formula_integrity(root: etree._Element) -> list[dict[str, Any]]:
    """Flag enacting formulae that are truncated or script-mismatched. Runs without source
    text, unlike the containment check: a formula ending in a literal ellipsis is
    truncated whatever its provenance, and one whose dominant script disagrees with the
    expression's FRBRlanguage is untranslated or misplaced delivery text.
    """
    lang_el = root.find(f".//{{{AKN_NS}}}FRBRlanguage")
    language = (lang_el.get("language") or "") if lang_el is not None else ""
    expected = _SCRIPT_BY_LANGUAGE.get(language, _LATIN_LETTERS if language else None)
    issues: list[dict[str, Any]] = []
    for fml in root.iter(f"{{{AKN_NS}}}formula"):
        if fml.get("name") != "enactingFormula":
            continue
        raw = re.sub(r"\s+", " ", "".join(fml.itertext())).strip()
        eid = fml.get("eId") or ""
        if raw.endswith(("...", "…")):
            issues.append(
                {
                    "check": "formula_truncated",
                    "severity": "error",
                    "eid": eid,
                    "message": f"Enacting formula is truncated (ends in ellipsis): {raw[-60:]!r}",
                }
            )
        if expected is not None and len(raw) > 10:
            counts = {
                "expected": len(expected.findall(raw)),
                "total": len(_ARABIC_LETTERS.findall(raw))
                + len(_HEBREW_LETTERS.findall(raw))
                + len(_LATIN_LETTERS.findall(raw)),
            }
            if counts["total"] > 10 and counts["expected"] < counts["total"] / 2:
                issues.append(
                    {
                        "check": "formula_script_mismatch",
                        "severity": "error",
                        "eid": eid,
                        "message": (
                            f"Enacting formula script does not match expression language "
                            f"{language!r}: {raw[:80]!r}"
                        ),
                    }
                )
    return issues


def _check_money_words_mismatch(root: etree._Element) -> list[dict[str, Any]]:
    """Flag money amounts whose parenthesised numeral disagrees with the adjacent
    amount-in-words ("(500,000) خمسون ألف دينار": the words say 50,000). Words survive OCR
    far better than digits, so a disagreement almost always means a corrupted numeral.
    Detector only; both readings ride in the finding.
    """
    issues: list[dict[str, Any]] = []
    for p in root.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext())
        if not text or "(" not in text:
            continue
        mismatches = find_money_word_mismatches(text)
        if not mismatches:
            continue
        anc: etree._Element | None = p
        while anc is not None and not anc.get("eId"):
            anc = anc.getparent()
        eid = anc.get("eId") if anc is not None else ""
        for mm in mismatches:
            issues.append(
                {
                    "check": "money_words_mismatch",
                    "severity": "warning",
                    "eid": eid,
                    "numeral": mm.numeral_text,
                    "numeral_value": mm.numeral_value,
                    "words": mm.words_text,
                    "words_value": mm.words_value,
                    "message": (
                        f"Numeral ({mm.numeral_text}) reads {mm.numeral_value:,} but the "
                        f"adjacent words ({mm.words_text}) read {mm.words_value:,} under "
                        f"{eid!r}; the numeral is likely OCR-corrupted."
                    ),
                }
            )
    return issues


# --- Identity self-consistency ----------------------------------------------

# A Hijri year that reached a Gregorian slot unconverted. 1300-1500 AH spans
# 1882-2076 CE, so nothing in that band is a plausible Gregorian enactment year
# here; 1431 (2010 CE) prompted the check.
_HIJRI_YEAR_RANGE = range(1300, 1501)

# Below this, a year is a mis-parse rather than a date. The oldest instrument
# any configured jurisdiction carries is 19th century.
_EARLIEST_PLAUSIBLE_YEAR = 1800

# The year segment is the second-to-last, ahead of the number. Cobalt writes
# FRBRthis with a trailing `/!main` component, so anchoring on the end of the
# string finds nothing on the shape the pipeline actually emits.
_URI_YEAR = re.compile(r"/(\d{4})/[^/]+(?:/![^/]+)?$")


def _check_identity_consistency(
    root: etree._Element, *, native: bool = False
) -> list[dict[str, Any]]:
    """Does the document agree with itself about when it was made? Every field compared is
    one already stored, so a disagreement is arithmetic rather than extraction: a year in
    the future, an unconverted Hijri year, or a date contradicting the year segment of
    the document's own FRBR URI. Each names a law nobody can cite correctly.

    ``native`` skips the two plausibility checks: a publisher's own date is not a
    mis-parse, and an old statute book holds instruments from before 1800.
    """
    work = root.find(f".//{{{AKN_NS}}}identification/{{{AKN_NS}}}FRBRWork")
    if work is None:
        return []
    date_el = work.find(f"{{{AKN_NS}}}FRBRdate")
    # FRBRuri is the bare work URI, FRBRthis the same path plus a component suffix,
    # so prefer the one needing no unpicking. Not `or`: a childless lxml element is
    # falsy, so a present but empty FRBRuri would fall through to FRBRthis.
    uri_el = work.find(f"{{{AKN_NS}}}FRBRuri")
    if uri_el is None:
        uri_el = work.find(f"{{{AKN_NS}}}FRBRthis")
    stated = (date_el.get("date") or "") if date_el is not None else ""
    uri = (uri_el.get("value") or "") if uri_el is not None else ""

    issues: list[dict[str, Any]] = []
    year: int | None = None
    if stated[:4].isdigit():
        year = int(stated[:4])
    # The unknown-date placeholder says the year never resolved, which the URI
    # already reports. Reading it as a date would call every year-less document
    # implausibly old and grade it blocking.
    if stated[:4] == UNKNOWN_YEAR:
        year = None

    if year is not None:
        this_year = datetime.now(UTC).year
        if year > this_year:
            issues.append(
                {
                    "check": "identity_year_in_future",
                    "severity": "error",
                    "message": f"FRBRdate year {year} is later than the current year {this_year}",
                }
            )
        elif not native and year in _HIJRI_YEAR_RANGE:
            issues.append(
                {
                    "check": "identity_year_unconverted_hijri",
                    "severity": "error",
                    "message": (
                        f"FRBRdate year {year} reads as a Hijri year that was never "
                        "converted to the Gregorian calendar"
                    ),
                }
            )
        elif not native and year < _EARLIEST_PLAUSIBLE_YEAR:
            issues.append(
                {
                    "check": "identity_year_implausible",
                    "severity": "error",
                    "message": f"FRBRdate year {year} predates any instrument this corpus holds",
                }
            )

    # The placeholder is exempt on the URI side too: a document whose year never
    # resolved gets `/0000/` by design, and once akn_meta writes a real date the two
    # disagree by construction, grading every such document blocking.
    m = _URI_YEAR.search(uri)
    if m and m.group(1) == UNKNOWN_YEAR:
        m = None
    if m and year is not None and int(m.group(1)) != year:
        issues.append(
            {
                "check": "identity_year_uri_mismatch",
                "severity": "error",
                "message": (
                    f"FRBR work URI names year {m.group(1)} but FRBRdate says {year}; "
                    "the document is citable two ways and neither is authoritative"
                ),
            }
        )

    return issues


def _check_displaced_terminal_material(
    root: etree._Element, closing_phrases: list[str]
) -> list[dict[str, Any]]:
    """Attestation lines the conclusions pass left inside the last provision. Fires on the
    shape `emit_conclusions` silently declines: a closing phrase plus short attestation
    lines trailing the final body container. The repair is `move_to_conclusions` on the
    named element.
    """
    from codify.pipeline.enrich.conclusions import find_displaced_attestation

    container, start = find_displaced_attestation(root, closing_phrases)
    if container is None or start is None:
        return []
    host = container if container.get("eId") else None
    if host is None:
        host = next((a for a in container.iterancestors() if a.get("eId")), None)
    if host is None:
        return []
    trailing = container.findall(f"{{{AKN_NS}}}p")
    count = len(trailing) - trailing.index(start)
    return [
        {
            "check": "displaced_terminal_material",
            "severity": "warning",
            "eid": host.get("eId"),
            "count": count,
            "message": (
                f"{count} attestation line(s) (promulgation/signature) sit inside "
                f"{host.get('eId')!r} instead of <conclusions>."
            ),
        }
    ]
