"""The in-process lane reports what it lost: a page the model refused reaches
the events, the validator and the AKN, and a config fault stops the run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from codify.jurisdictions import JurisdictionConfigError
from codify.pipeline.enrich.ocr import PageResult
from codify.pipeline.enrich.scaffold import BodyBlock, BodyFillResponse
from codify.pipeline.events import Complete, Failed, PageExtracted, ValidationIssued
from codify.pipeline.formats import pdf as pdf_mod

_PAGE_ONE = """\
PART I
PRELIMINARY

Section 1
This Act may be cited as the Harbour Widgets Act.

Section 2
In this Act "widget" means a mechanical device registered under this Act.
"""


class _BodyFillOnly:
    """Fills every eId it is asked for; every other call fails, as offline."""

    _EID_RE = re.compile(r"eid=(\S+)")

    async def chat_schema(self, prompt: str, schema: Any, system: Any = None, model: Any = None):
        if schema is not BodyFillResponse:
            raise RuntimeError("offline")
        eids = self._EID_RE.findall(prompt)
        return BodyFillResponse(bodies=[BodyBlock(eid=e, lines=[f"Body of {e}."]) for e in eids])

    def __getattr__(self, name: str) -> Any:
        async def _offline(*_a: object, **_k: object) -> Any:
            raise RuntimeError("offline")

        return _offline


def _pages() -> list[PageResult]:
    return [
        PageResult(page_number=1, text=_PAGE_ONE, method="text_extraction"),
        PageResult(
            page_number=2,
            text="",
            method="vision_ocr",
            divert_reason="too_short",
            finish_reason="content_filter: RECITATION",
        ),
    ]


async def _events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    async def _extract(*_a: object, **_k: object) -> list[PageResult]:
        return _pages()

    monkeypatch.setattr(pdf_mod, "extract_text_from_pdf", _extract)
    source = tmp_path / "act.pdf"
    source.write_bytes(b"%PDF-fake")
    return [e async for e in pdf_mod.ingest(source, "xa", llm=_BodyFillOnly())]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_refused_page_reaches_the_events_the_findings_and_the_akn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = await _events(monkeypatch, tmp_path)

    extracted = [
        (e.page, e.finish_reason, e.unreadable) for e in events if isinstance(e, PageExtracted)
    ]
    assert extracted == [
        (1, "", ""),
        (2, "content_filter: RECITATION", "content_filter: RECITATION"),
    ]

    unreadable = [
        (e.issue["severity"], e.issue["pages"])
        for e in events
        if isinstance(e, ValidationIssued) and e.issue.get("check") == "unreadable_page"
    ]
    assert unreadable == [("error", [2])]

    done = [e for e in events if isinstance(e, Complete)]
    assert done, [type(e).__name__ for e in events]
    assert "[Page 2 of the source could not be read]" in done[0].akn_xml


@pytest.mark.asyncio
async def test_the_lane_validates_as_extracted_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    real = pdf_mod.validate_akn

    def _spy(xml: str, **kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return real(xml, **kwargs)

    monkeypatch.setattr(pdf_mod, "validate_akn", _spy)
    await _events(monkeypatch, tmp_path)
    assert (seen["provenance"], seen["unreadable_pages"], seen["body_fill"]["expected"]) == (
        "extracted",
        {2: "content_filter: RECITATION"},
        2,
    )


@pytest.mark.asyncio
async def test_a_config_fault_in_the_region_pass_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _broken(*_a: object, **_k: object) -> Any:
        raise JurisdictionConfigError("no config")

    monkeypatch.setattr(pdf_mod, "_regions_from_pages", _broken)
    events = await _events(monkeypatch, tmp_path)
    assert [(e.stage, e.error) for e in events if isinstance(e, Failed)] == [
        ("regions", "JurisdictionConfigError: no config")
    ]
    assert not [e for e in events if isinstance(e, Complete)]


@pytest.mark.asyncio
async def test_an_inked_page_read_empty_says_so_on_its_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No finish reason to go on: the event still tells it from a blank leaf."""

    async def _extract(*_a: object, **_k: object) -> list[PageResult]:
        return [
            PageResult(page_number=1, text=_PAGE_ONE, method="text_extraction"),
            PageResult(
                page_number=2, text="", method="vision_ocr", divert_reason="too_short", ink=0.3
            ),
            PageResult(
                page_number=3, text="", method="vision_ocr", divert_reason="too_short", ink=0.001
            ),
        ]

    monkeypatch.setattr(pdf_mod, "extract_text_from_pdf", _extract)
    source = tmp_path / "act.pdf"
    source.write_bytes(b"%PDF-fake")
    events = [e async for e in pdf_mod.ingest(source, "xa", llm=_BodyFillOnly())]  # type: ignore[arg-type]
    page_events = [e for e in events if isinstance(e, PageExtracted)]
    assert [(e.page, e.unreadable) for e in page_events] == [(1, ""), (2, "empty_read"), (3, "")]
    assert '"unreadable":"empty_read"' in page_events[1].model_dump_json()
