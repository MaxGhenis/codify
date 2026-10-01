"""What a scan records about itself.

An ingest that finds nothing still reports success, because the coverage gate's
denominator comes from the same keywords the scanner failed to find and a
document with no keywords therefore has no expected set to fall short of. The
scan trace is what makes that visible, so these tests pin the three things a
reader needs from it: which pass produced each anchor, which numbers coverage
compared, and that a trace is emitted on the paths that produce no scaffold.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from codify.jurisdictions import load_config
from codify.pipeline.enrich.anchors import anchor_coverage, cached_regex, scan_anchors
from codify.pipeline.enrich.structure import ScanTrace, text_to_bluebell_scaffolded

# A descriptive rubric, then an Arabic-Indic numeral, then a dash, all on one
# line and with no مادة keyword. The regex anchor pass finds nothing in it,
# which is what the scan trace has to make visible.
_HEADING_FUSED_NUMBER = """سلطة الموانئ الأطلسية
نظام بشأن تسجيل الأدوات الميكانيكية
اعفاء من دفع رسوم ١ - يعفى من دفع الرسوم تسجيل الأدوات بموجب هذا النظام .
بدء سريان ٢ - يسري هذا النظام اعتبارا من تاريخ نشره في الجريدة .
الاسم ٣ - يطلق على هذا النظام اسم "نظام تسجيل الأدوات الميكانيكية" .
"""

_KEYWORDED = """مادة ١
نص المادة الأولى.

مادة ٢
نص المادة الثانية.
"""


class _NeverCalled:
    """The structurer must not reach a model on the no-anchor path."""

    async def chat_schema(self, *a: object, **k: object) -> object:
        raise AssertionError("body fill ran on a document with no anchors")


class _EmptyBodies:
    """Fills nothing; these tests are about the scan, not the prose."""

    async def chat_schema(self, *a: object, **k: object) -> object:
        from codify.pipeline.enrich.scaffold import BodyFillResponse

        return BodyFillResponse(bodies=[])


def _anchors(text: str) -> list[object]:
    return scan_anchors(text, cached_regex("xz", "act"), country="xz", doctype="act")


def test_an_anchor_names_the_pass_that_produced_it() -> None:
    anchors = _anchors(_KEYWORDED)
    assert anchors and all(a.source_pass == "regex" for a in anchors)  # type: ignore[attr-defined]


def test_coverage_reports_both_number_sets() -> None:
    anchors = _anchors(_KEYWORDED)
    cov = anchor_coverage(_KEYWORDED, anchors, load_config("xz"), "act", "article")
    assert cov.captured == frozenset({"1", "2"})
    assert cov.expected == frozenset({"1", "2"})
    assert cov.missing == frozenset()
    assert cov.ratio == 1.0


def test_a_document_with_no_markers_reports_no_measurement() -> None:
    """Why a zero-anchor ingest still succeeds: with no expected numbers there
    is no denominator, so the gate cannot fire on the worst case. The ratio is
    None rather than 1.0 so the bundle does not read as perfect coverage."""
    anchors = _anchors(_HEADING_FUSED_NUMBER)
    cov = anchor_coverage(_HEADING_FUSED_NUMBER, anchors, load_config("xz"), "act", "article")
    assert cov.captured == frozenset()
    assert cov.expected == frozenset()
    assert cov.ratio is None


@pytest.mark.asyncio
async def test_the_no_anchor_path_still_emits_a_trace() -> None:
    """The trace is the only record of a run that produced nothing, so it has to
    survive the early return the verbatim fallback takes."""
    seen: list[ScanTrace] = []
    out = await text_to_bluebell_scaffolded(
        _HEADING_FUSED_NUMBER,
        client=_NeverCalled(),  # type: ignore[arg-type]
        country="xz",
        doctype="act",
        on_scan=seen.append,
    )
    assert len(seen) == 1
    trace = seen[0]
    assert trace.anchors == ()
    assert trace.scaffold is None
    assert trace.fallback == "verbatim_section"
    assert trace.coverage is not None and trace.coverage.ratio is None
    assert "BODY" in out


def test_promoting_an_orphan_child_keeps_its_provenance() -> None:
    """`_drop_orphan_children` retypes a child anchor to article. Rebuilding it
    field by field dropped whichever fields were added later."""
    from dataclasses import replace

    from codify.pipeline.enrich.anchors import (
        StructuralAnchor,
        _drop_orphan_children,
        _rank_map_for,
    )

    children = [
        replace(
            StructuralAnchor(
                kind="point",
                keyword="البند",
                number=str(n),
                char_offset=n * 100,
                line=n,
                matched_text=f"بند {n}",
            ),
            source_pass="regex",
            quoted_amendment=True,
        )
        for n in range(1, 6)
    ]
    out = _drop_orphan_children(children, _rank_map_for("xz", "act"))
    promoted = [a for a in out if a.kind == "article"]
    assert promoted
    assert all(a.source_pass == "regex" and a.quoted_amendment for a in promoted)


@pytest.mark.asyncio
async def test_the_gate_traces_the_missing_numbers_before_it_raises(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`missing` is the field the bundle exists to surface, and the gate path is
    the only one where it is non-empty. Trace after the raise and the operator
    gets counts in an exception and a null coverage file."""
    from codify.pipeline.enrich import structure as structure_mod
    from codify.pipeline.enrich.anchors import AnchorCoverage

    monkeypatch.setattr(
        structure_mod,
        "anchor_coverage",
        lambda text, anchors, config, doctype, kind, **_: AnchorCoverage(
            kind=kind,
            ratio=0.05,
            captured=frozenset({"1"}),
            expected=frozenset(str(n) for n in range(1, 21)),
        ),
    )
    seen: list[ScanTrace] = []
    with pytest.raises(structure_mod.AnchorCoverageError):
        await text_to_bluebell_scaffolded(
            _KEYWORDED,
            client=_NeverCalled(),  # type: ignore[arg-type]
            country="xz",
            doctype="act",
            on_scan=seen.append,
        )
    assert len(seen) == 1
    assert seen[0].fallback == "coverage_gate"
    assert seen[0].scaffold is None
    assert seen[0].coverage is not None
    assert len(seen[0].coverage.missing) == 19


_COLLIDING = """مادة ١
نص المادة الأولى.

مادة ٢
نص المادة الثانية.

مادة ٢
نص ثان يحمل نفس الرقم.
"""


@pytest.mark.asyncio
async def test_a_surviving_number_collision_stops_the_run(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A collision papered over with a suffix means one of two provisions is
    wrong. Both upstream passes are disabled so the gate is what is tested: dedup
    collapses the twin, digit repair renumbers it away."""
    from codify.pipeline.enrich import anchors as anchors_mod
    from codify.pipeline.enrich import structure as structure_mod

    monkeypatch.setattr(
        anchors_mod, "_drop_toc_duplicates", lambda anchors, rank_map=None, spans=None: anchors
    )
    monkeypatch.setattr(anchors_mod, "_repair_article_digit_ocr", lambda anchors: anchors)
    seen: list[ScanTrace] = []
    with pytest.raises(structure_mod.AnchorInvariantError) as exc:
        await text_to_bluebell_scaffolded(
            _COLLIDING,
            client=_NeverCalled(),  # type: ignore[arg-type]
            country="xz",
            doctype="act",
            on_scan=seen.append,
        )
    assert exc.value.by_kind == {"duplicate_number": 1}
    # Traced before the raise, or the operator gets an exception and no bundle.
    assert len(seen) == 1
    assert seen[0].fallback == "invariant_gate"
    assert seen[0].scaffold is None
    assert seen[0].ambiguity


@pytest.mark.asyncio
async def test_a_resolved_toc_twin_does_not_stop_the_run(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Dedup left on: the twin collapses, the choice is recorded, the run
    proceeds. Blocking here refuses every document with a table of contents."""
    from codify.pipeline.enrich import anchors as anchors_mod

    monkeypatch.setattr(anchors_mod, "_repair_article_digit_ocr", lambda anchors: anchors)
    seen: list[ScanTrace] = []
    out = await text_to_bluebell_scaffolded(
        _COLLIDING,
        client=_EmptyBodies(),  # type: ignore[arg-type]
        country="xz",
        doctype="act",
        on_scan=seen.append,
    )
    assert out
    duplicates = [s for s in seen[0].ambiguity if s.kind == "duplicate_number"]
    assert duplicates, "the collapsed twin should still be recorded"
    assert all(s.resolved for s in duplicates)
    assert not [s for s in seen[0].ambiguity if s.blocking]


@pytest.mark.asyncio
async def test_an_empty_source_traces_the_empty_fallback() -> None:
    """The only exit where the trace is the sole evidence the scan ran."""
    seen: list[ScanTrace] = []
    await text_to_bluebell_scaffolded(
        "   \n  ",
        client=_NeverCalled(),  # type: ignore[arg-type]
        country="xz",
        doctype="act",
        on_scan=seen.append,
    )
    assert len(seen) == 1
    assert seen[0].fallback == "empty"
    assert seen[0].anchors == ()


@pytest.mark.asyncio
async def test_the_scaffold_path_traces_the_scaffold_with_eids_assigned() -> None:
    """The happy path, and the only exit carrying anchors. `anchors.jsonl` is
    worthless without eIds, so the snapshot must be taken after they are set."""
    from codify.pipeline.enrich.scaffold import BodyBlock

    class _Filling:
        async def chat_schema(self, prompt, schema, system=None, model=None):  # type: ignore[no-untyped-def]
            return schema(bodies=[BodyBlock(eid="art_1", lines=["نص."])])

    seen: list[ScanTrace] = []
    await text_to_bluebell_scaffolded(
        _KEYWORDED,
        client=_Filling(),  # type: ignore[arg-type]
        country="xz",
        doctype="act",
        on_scan=seen.append,
    )
    assert len(seen) == 1
    trace = seen[0]
    assert trace.scaffold is not None
    assert trace.fallback is None
    assert [a.akn_eid for a in trace.anchors] == ["art_1", "art_2"]
    assert all(a.source_pass == "regex" for a in trace.anchors)


# `منها` ("of them") is prose, so no alias resolves it and it stays declared.
# A recoverable mangling like `المادم` is claimed instead, not declared.
_MANGLED = """مادة (1)
نص.

منها (20)
نص مفقود.

مادة (40)
نص.
"""


def test_a_marker_the_regex_never_matched_is_declared() -> None:
    """A marker-shaped line the scan loop never sees is still declared, so the
    rejections recorded there cannot be the whole picture."""
    from codify.pipeline.enrich.anchors import scan_anchors_with_ambiguity

    r = scan_anchors_with_ambiguity(
        _MANGLED, cached_regex("xz", "act"), country="xz", doctype="act"
    )
    unclaimed = [s for s in r.ambiguity if s.detail.get("reason") == "marker_shaped_line_unclaimed"]
    assert [s.detail["number"] for s in unclaimed] == ["20"]


def test_a_recovered_marker_is_not_reported_as_unclaimed() -> None:
    """`_recover_misread_article_markers` bridges gaps of three or fewer, and a
    marker it recovered is not lost."""
    from codify.pipeline.enrich.anchors import scan_anchors_with_ambiguity

    text = "مادة (1)\nنص.\n\nالمادم (2)\nنص.\n\nمادة (3)\nنص.\n"
    r = scan_anchors_with_ambiguity(text, cached_regex("xz", "act"), country="xz", doctype="act")
    assert [a.akn_eid for a in r.anchors] == ["art_1", "art_2", "art_3"]
    assert not [s for s in r.ambiguity if s.detail.get("reason") == "marker_shaped_line_unclaimed"]


def test_a_dropped_compilation_wrapper_is_declared() -> None:
    """A lone container numbered 2 or higher is read as a compilation wrapper
    and deleted. Silent deletion is the loss `AmbiguitySpan` exists to prevent."""
    from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity

    result = scan_anchors_with_ambiguity(
        "مقدمة\n\nالفصل الثاني\n\nمادة (1)\nنص.\n",
        cached_regex("xz", "act"),
        country="xz",
        doctype="act",
    )
    assert [a.kind for a in result.anchors] == ["article"]
    dropped = [s for s in result.ambiguity if s.emitted_by == "drop_lone_compilation_container"]
    assert [s.detail["reads_as"] for s in dropped] == ["compilation_wrapper"]
    assert all(s.resolved and not s.blocking for s in dropped)


# The gates below are the only places the pipeline throws away a document it
# could write. Two record and continue; two keep failing.

from codify.quality.invariants import AmbiguitySpan  # noqa: E402


def _with_blocking_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force one unresolved duplicate. The scanner repairs a synthetic repeat
    before the gate sees it, so the shape has to be injected to be tested."""
    from codify.pipeline.enrich import structure as mod

    real = mod.scan_anchors_with_ambiguity

    def _scan(*a: object, **k: object) -> object:
        scan = real(*a, **k)  # type: ignore[arg-type]
        span = AmbiguitySpan(
            kind="duplicate_number",
            start=42,
            end=50,
            eid="art_2",
            emitted_by="assign_eids",
            detail={"num": "2"},
        )
        return type(scan)(anchors=scan.anchors, ambiguity=(*scan.ambiguity, span))

    monkeypatch.setattr(mod, "scan_anchors_with_ambiguity", _scan)


async def test_a_blocking_span_lands_the_document_under_the_landing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The colliding eIds are already suffixed, so the document round-trips. What
    the scanner cannot say is which reading the source meant, and repair cannot be
    asked that until the version exists."""
    _with_blocking_span(monkeypatch)
    traces: list[ScanTrace] = []
    out = await text_to_bluebell_scaffolded(
        _KEYWORDED,
        client=_EmptyBodies(),
        country="xz",
        doctype="act",
        on_scan=traces.append,
        halt_policy="land",
    )
    assert out.strip()
    halts = [h for t in traces for h in t.halts]
    assert [h.gate for h in halts] == ["duplicate_anchor"]
    assert halts[0].spans == 1
    assert halts[0].first_offset == 42


async def test_the_default_policy_still_refuses_a_blocking_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Landing is opt-in: nothing that does not ask for it changes behaviour."""
    from codify.pipeline.enrich.structure import AnchorInvariantError

    _with_blocking_span(monkeypatch)
    with pytest.raises(AnchorInvariantError):
        await text_to_bluebell_scaffolded(
            _KEYWORDED, client=_EmptyBodies(), country="xz", doctype="act"
        )


def _coverage(monkeypatch: pytest.MonkeyPatch, *, ratio: float | None, expected: int) -> None:
    """Pin what the gate measures. The floor is a property of the config, so the
    only way to test each side of it is to fix the measurement."""
    from codify.pipeline.enrich import structure as mod
    from codify.pipeline.enrich.anchors import AnchorCoverage

    captured = frozenset(str(i) for i in range(int((ratio or 0) * expected)))
    monkeypatch.setattr(
        mod,
        "anchor_coverage",
        lambda *a, **k: AnchorCoverage(
            kind="article",
            ratio=ratio,
            captured=captured,
            expected=frozenset(str(i) for i in range(expected)),
        ),
    )


async def test_a_thin_document_lands_with_the_coverage_shortfall_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half a law read is still law. It lands graded blocking rather than vanishing."""
    _coverage(monkeypatch, ratio=0.5, expected=10)
    traces: list[ScanTrace] = []
    out = await text_to_bluebell_scaffolded(
        _KEYWORDED,
        client=_EmptyBodies(),
        country="xz",
        doctype="act",
        on_scan=traces.append,
        halt_policy="land",
    )
    assert out.strip()
    halts = [h for t in traces for h in t.halts]
    assert [h.gate for h in halts] == ["coverage_below_floor"]
    assert (halts[0].captured, halts[0].expected, halts[0].ratio) == (5, 10, 0.5)


async def test_nothing_measured_still_fails_under_the_landing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ratio 0.0 is not a thin document, it is a document nothing was read from.
    A truncated source landing as a version records missing law as present."""
    from codify.pipeline.enrich.structure import AnchorCoverageError

    _coverage(monkeypatch, ratio=0.0, expected=10)
    with pytest.raises(AnchorCoverageError):
        await text_to_bluebell_scaffolded(
            _KEYWORDED,
            client=_EmptyBodies(),
            country="xz",
            doctype="act",
            halt_policy="land",
        )


async def test_markers_outside_the_boundary_still_fail_under_the_landing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty denominator with markers the relaxed boundary can see means the
    source lost the line layout the policy assumes, so the text is not the document."""
    from codify.pipeline.enrich import structure as mod
    from codify.pipeline.enrich.structure import AnchorCoverageError

    _coverage(monkeypatch, ratio=None, expected=0)
    monkeypatch.setattr(mod, "markers_outside_boundary", lambda *a, **k: 7)
    with pytest.raises(AnchorCoverageError):
        await text_to_bluebell_scaffolded(
            _KEYWORDED,
            client=_EmptyBodies(),
            country="xz",
            doctype="act",
            halt_policy="land",
        )


_PARTS_ONLY = """PART I — PRELIMINARY

Short title
This Act may be cited as the Example Act.

PART II — THE AUTHORITY

Establishment
There is established an Authority.
"""


async def test_a_scan_with_no_basic_unit_says_the_body_was_never_filled() -> None:
    """Unnumbered sections anchor their parts and nothing below; the skeleton
    then ships as the document, which must not read as a clean run."""
    with capture_logs() as logs:
        out = await text_to_bluebell_scaffolded(_PARTS_ONLY, client=_NeverCalled(), country="xa")  # type: ignore[arg-type]
    skipped = [r for r in logs if r["event"] == "body_fill_skipped"]
    assert skipped and skipped[0]["reason"] == "no_basic_unit_anchors"
    assert "PART I" in out and "This Act may be cited" not in out
