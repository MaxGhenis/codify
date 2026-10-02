"""Two acts in one source, each numbering from 1, fold onto the second.

Keep-last is right for a contents listing above a body, and stays. What changes is
that a fold whose dropped run closes one instrument and opens another is declared
blocking, so the run halts instead of landing one act as the whole source.

Fictional jurisdiction throughout: the shape is the subject, not any corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest

import codify.jurisdictions as jurisdictions
from codify.pipeline.enrich import structure as structure_mod
from codify.pipeline.enrich.anchors import (
    _declare_act_boundary_suspected,
    cached_regex,
    scan_anchors_with_ambiguity,
)
from codify.quality.invariants import AmbiguitySpan, halt_finding
from tests.config_fixtures import isolated_configs

# A code no other test configures, so per-country caches cannot carry another shape.
COUNTRY = "xf"
DOCTYPE = "act"
CLOSING = "given under the Seal of the Assembly"
HEADING = r"ACT No\. (?P<number>\d+) OF \d{4}"


def _config(**overrides: object) -> dict[str, object]:
    """The fictional `xa` configuration, under its own code, with the overrides."""
    base = json.loads((jurisdictions.JURISDICTIONS_DIR / "xa" / "config.json").read_text())
    base.update(code=COUNTRY, closing_phrases=[CLOSING], **overrides)
    return base


@pytest.fixture
def armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Closing phrases and an act heading declared, as a gazette jurisdiction would."""
    configs = {COUNTRY: _config(segmentation={"act_heading_patterns": [HEADING]})}
    with isolated_configs(monkeypatch, tmp_path / "jurisdictions", configs):
        yield


@pytest.fixture
def unarmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The same closing phrases and no heading declared: the detector has no evidence."""
    with isolated_configs(monkeypatch, tmp_path / "jurisdictions", {COUNTRY: _config()}):
        yield


def _act(number: int, title: str, sections: int) -> list[str]:
    lines = [f"ACT No. {number} OF 2020", title, "", "BE IT ENACTED by the Assembly:", ""]
    for i in range(1, sections + 1):
        lines += [
            f"Section {i}. Duty {i}",
            "The keeper of every harbour shall levy the dues set out here.",
            "",
        ]
    # Mid-line: a phrase opening its line would end the body before the fold.
    lines += [f"This Act was {CLOSING} at Port Town.", "Speaker of the Assembly", ""]
    return lines


def _two_acts() -> str:
    """The reproduction's shape: two consecutive instruments, sections 1-4 each."""
    return "\n".join(_act(3, "THE HARBOUR DUES ACT", 4) + _act(4, "THE LIGHTHOUSE ACT", 4))


def _scan(text: str):  # type: ignore[no-untyped-def]
    return scan_anchors_with_ambiguity(
        text, cached_regex(COUNTRY, DOCTYPE), country=COUNTRY, doctype=DOCTYPE
    )


def _suspected(text: str) -> list[AmbiguitySpan]:
    return [s for s in _scan(text).ambiguity if s.kind == "act_boundary_suspected"]


@pytest.mark.usefixtures("armed")
def test_two_acts_numbered_from_one_declare_the_boundary() -> None:
    text = _two_acts()
    found = _suspected(text)
    assert len(found) == 1, [s.detail for s in _scan(text).ambiguity]
    span = found[0]
    assert span.blocking
    # The run the fold orphaned: the first act's last section to the second's first.
    assert text[span.start :].lstrip().startswith("Section 4. Duty 4")
    second = text.index("ACT No. 4 OF 2020")
    assert span.detail["heading_at"] == second
    assert span.detail["heading"] == "ACT No. 4 OF 2020"
    assert text[span.detail["closing_at"] :].startswith(CLOSING)
    assert span.end > second
    assert text[span.end :].lstrip().startswith("Section 1. Duty 1")
    assert span.detail["number"] == "4"


def test_the_fold_outcome_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declaring is all the detector does: keep-last still keeps the second act."""
    text = _two_acts()
    seen = {}
    for name, segmentation in (("armed", {"act_heading_patterns": [HEADING]}), ("unarmed", None)):
        fields = _config(segmentation=segmentation) if segmentation else _config()
        with isolated_configs(monkeypatch, tmp_path / name, {COUNTRY: fields}):
            scan = _scan(text)
            seen[name] = (
                [(a.kind, a.number, a.char_offset, a.akn_eid) for a in scan.anchors],
                sorted(s.kind for s in scan.ambiguity),
            )
    second = text.index("ACT No. 4 OF 2020")
    assert [n for _, n, _, _ in seen["armed"][0]] == ["1", "2", "3", "4"]
    assert all(offset > second for _, _, offset, _ in seen["armed"][0])
    assert seen["armed"][0] == seen["unarmed"][0]
    assert "act_boundary_suspected" not in seen["unarmed"][1]
    assert seen["armed"][1].count("act_boundary_suspected") == 1


@pytest.mark.usefixtures("armed")
def test_a_contents_listing_folds_without_a_finding() -> None:
    """Keep-last's own case: the listing above the body is not an act boundary."""
    lines = ["ACT No. 3 OF 2020", "THE HARBOUR DUES ACT", "", "CONTENTS", ""]
    lines += [f"Section {i}. Duty {i}" for i in range(1, 5)]
    lines += [""] + _act(3, "THE HARBOUR DUES ACT", 4)[2:]
    scan = _scan("\n".join(lines))
    assert any(s.kind == "duplicate_number" for s in scan.ambiguity)
    assert not [s for s in scan.ambiguity if s.kind == "act_boundary_suspected"]


@pytest.mark.usefixtures("armed")
def test_a_closing_with_no_heading_after_it_declares_nothing() -> None:
    """A repeated run under one title, signed between: restated, not a second act."""
    text = _two_acts().replace("ACT No. 4 OF 2020", "SCHEDULE OF DUES")
    assert not _suspected(text)


@pytest.mark.usefixtures("armed")
def test_a_heading_with_no_closing_before_it_declares_nothing() -> None:
    """An unsigned run may be a contents listing that names the act: no closing,
    no evidence the first instrument ended."""
    text = _two_acts().replace(f"This Act was {CLOSING} at Port Town.\n", "", 1)
    assert text.count(CLOSING) == 1
    assert not _suspected(text)


@pytest.mark.usefixtures("armed")
def test_a_heading_before_the_closing_declares_nothing() -> None:
    """Order matters: the second act opens after the first one closes."""
    text = _two_acts()
    first_close = text.index(f"This Act was {CLOSING}")
    second = text.index("ACT No. 4 OF 2020")
    # Swap the first act's closing line out of its run and put it after the heading.
    closing_line = f"This Act was {CLOSING} at Port Town.\n"
    text = text[:first_close] + text[first_close + len(closing_line) : second]
    text += (
        "ACT No. 4 OF 2020\n" + closing_line + _two_acts()[second + len("ACT No. 4 OF 2020\n") :]
    )
    assert text.count("ACT No. 4 OF 2020") == 1
    assert not _suspected(text)


@pytest.mark.usefixtures("armed")
def test_a_quoted_heading_declares_nothing() -> None:
    """A heading inside a quotation is text being cited, not an act opening.
    Curly quotes: straight ones are ordinary punctuation to the quote reader."""
    text = _two_acts().replace(
        "ACT No. 4 OF 2020\nTHE LIGHTHOUSE ACT",
        "The schedule reads: \u201c\nACT No. 4 OF 2020\nTHE LIGHTHOUSE ACT\u201d",
    )
    assert not _suspected(text)


@pytest.mark.usefixtures("armed")
def test_a_quoted_closing_declares_nothing() -> None:
    """The first act recites a closing it does not itself carry."""
    signed = f"This Act was {CLOSING} at Port Town."
    first = _two_acts().index(signed)
    text = _two_acts()
    text = (
        text[:first]
        + f"It recites \u201c{CLOSING}\u201d from the charter."
        + text[first + len(signed) :]
    )
    assert text.count(signed) == 1
    assert not _suspected(text)


@pytest.mark.usefixtures("unarmed")
def test_no_declared_heading_declares_nothing() -> None:
    """Silence, not a finding: the closing alone cannot say a second act began."""
    assert not _suspected(_two_acts())


@pytest.mark.usefixtures("armed")
def test_only_a_fold_run_is_read() -> None:
    """Another pass's drop is not keep-last folding, so its run is not this finding's."""
    text = _two_acts()
    second = text.index("ACT No. 4 OF 2020")
    start = text.index("Section 4. Duty 4")

    def run(emitted_by: str) -> list[AmbiguitySpan]:
        spans = [
            AmbiguitySpan(
                kind="duplicate_number",
                start=start,
                end=start + 9,
                emitted_by=emitted_by,
                resolved=True,
                detail={"number": "4"},
            )
        ]
        _declare_act_boundary_suspected(text, [], spans, COUNTRY)
        return [s for s in spans if s.kind == "act_boundary_suspected"]

    assert [s.detail["heading_at"] for s in run("drop_toc_duplicates")] == [second]
    assert run("drop_preamble_citation_articles") == []


@pytest.mark.usefixtures("armed")
def test_a_heading_past_the_run_declares_nothing() -> None:
    """The run ends at the next marker; a heading beyond it belongs to a later run."""
    text = _two_acts()
    start = text.index("Section 4. Duty 4")
    closing = text.index(CLOSING)
    spans = [
        AmbiguitySpan(
            kind="duplicate_number",
            start=start,
            end=start + 9,
            emitted_by="drop_toc_duplicates",
            resolved=True,
            detail={"number": "4"},
        ),
        # Another drop between the closing and the heading ends the run there.
        AmbiguitySpan(
            kind="duplicate_number",
            start=closing + len(CLOSING),
            end=closing + len(CLOSING) + 1,
            emitted_by="drop_orphan_children",
            resolved=True,
        ),
    ]
    _declare_act_boundary_suspected(text, [], spans, COUNTRY)
    assert not [s for s in spans if s.kind == "act_boundary_suspected"]


class _NeverCalledLLMClient:
    """The gate sits ahead of body-fill, so any model call is a failure."""

    async def chat(self, *a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("the structurer must stop before body-fill")

    async def chat_stream(self, *a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("the structurer must stop before body-fill")

    async def chat_schema(self, *a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("the structurer must stop before body-fill")


@pytest.mark.asyncio
@pytest.mark.usefixtures("armed")
async def test_the_structurer_refuses_under_the_failing_policy() -> None:
    with pytest.raises(structure_mod.AnchorInvariantError) as excinfo:
        await structure_mod.text_to_bluebell_scaffolded(
            _two_acts(),
            client=_NeverCalledLLMClient(),  # type: ignore[arg-type]
            country=COUNTRY,
            doctype=DOCTYPE,
        )
    assert excinfo.value.by_kind == {"act_boundary_suspected": 1}


class _EmptyBodies:
    """Fills nothing: the halt is recorded before body-fill, and lands regardless."""

    async def chat_schema(self, *a: object, **k: object) -> object:
        from codify.pipeline.enrich.scaffold import BodyFillResponse

        return BodyFillResponse(bodies=[])


@pytest.mark.asyncio
@pytest.mark.usefixtures("armed")
async def test_the_landing_policy_records_a_blocking_halt() -> None:
    """Landed with a halt the run reports as an error, never as a clean success."""
    traces: list[structure_mod.ScanTrace] = []
    await structure_mod.text_to_bluebell_scaffolded(
        _two_acts(),
        client=_EmptyBodies(),  # type: ignore[arg-type]
        country=COUNTRY,
        doctype=DOCTYPE,
        on_scan=traces.append,
        halt_policy="land",
    )
    assert traces, "the scan trace never fired"
    halts = traces[-1].halts
    assert [h.gate for h in halts] == ["act_boundary_suspected"]
    finding = halt_finding(asdict(halts[0]))
    assert finding["severity"] == "error"
    assert finding["gate"] == "act_boundary_suspected"
    assert "splitting the source" in finding["message"]
