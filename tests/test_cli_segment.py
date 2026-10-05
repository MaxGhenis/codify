"""The bundle carries what the segmenter read, and `codify segment` re-reads it.

A fictional issue holding three acts, with printed page numbers lifted into
furniture, runs through `ingest-one` and then through `codify segment`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from codify import cli
from codify.pipeline.enrich.ocr import PageResult
from tests.config_fixtures import isolated_configs
from tests.pipeline.test_segment import COUNTRY, SEGMENTATION, _config, _three_act_issue

# Printed numbers only in furniture: a bundle that drops furniture loses the contents.
RULES = {**SEGMENTATION, "printed_page_pattern": r"^Page (\d+)$"}


@pytest.fixture(autouse=True)
def _restore_structlog() -> Iterator[None]:
    """`main` repoints structlog at a stderr pytest later closes."""
    saved = structlog.get_config().copy()
    yield
    structlog.configure(**saved)


@pytest.fixture
def jurisdiction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    with isolated_configs(
        monkeypatch, tmp_path / "jurisdictions", {COUNTRY: _config(segmentation=RULES)}
    ):
        yield COUNTRY


def _pages() -> list[PageResult]:
    issue = _three_act_issue()
    for lines in issue[1:]:
        del lines[:2]
    return [
        PageResult(
            page_number=number,
            text="\n".join(lines),
            method="text_layer",
            footer=f"Page {number}" if number > 1 else "",
        )
        for number, lines in enumerate(issue, start=1)
    ]


def _fake_ingest(pages: list[PageResult]) -> Callable[..., AsyncIterator[Any]]:
    async def ingest(
        *_a: object, on_pages: Callable[[list[PageResult]], None], **_k: object
    ) -> AsyncIterator[Any]:
        on_pages(pages)
        return
        yield

    return ingest


async def _ingest_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: str) -> Path:
    monkeypatch.setattr(cli, "create_llm_client", lambda **_k: object())
    monkeypatch.setattr(cli, "ingest", _fake_ingest(_pages()))
    source = tmp_path / "issue.pdf"
    source.write_bytes(b"not rendered")
    out = tmp_path / "bundle"
    args = argparse.Namespace(
        source=str(source),
        jurisdiction=code,
        out=str(out),
        model="m",
        ocr_model="",
        fallback_model="",
        quiet=True,
    )
    await cli._run(args)
    return out


async def test_ingest_one_writes_the_spans_and_furniture_the_segmenter_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jurisdiction: str
) -> None:
    out = await _ingest_one(monkeypatch, tmp_path, jurisdiction)
    spans = json.loads((out / "page_spans.json").read_text())
    source = (out / "source.txt").read_text()
    assert [(p["page"], source[p["start"] : p["end"]][:8]) for p in spans["pages"]] == [
        (1, "THE ATLA"),
        (2, "ACT No. "),
        (3, "ACT No. "),
    ]
    assert spans["furniture"] == [
        {"page": 2, "header": "", "footer": "Page 2"},
        {"page": 3, "header": "", "footer": "Page 3"},
    ]
    result = json.loads((out / "segmentation.json").read_text())
    assert result["outcome"] == "decided"
    assert [s["key"] for s in result["segments"]] == ["act 3", "act 4", "act 5"]
    assert [
        (r["key"], r["status"], r["printed_page"])
        for r in result["reconciliation"]
        if r["source"] == "contents"
    ] == [("act 3", "matched", 2), ("act 4", "matched", 2), ("act 5", "matched", 3)]
    assert all("text" not in s for s in result["segments"])
    assert json.loads((out / "manifest.json").read_text())["segmentation_failed"] is None


async def test_segment_reproduces_the_bundle_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jurisdiction: str
) -> None:
    out = await _ingest_one(monkeypatch, tmp_path, jurisdiction)
    written = (out / "segmentation.json").read_bytes()
    (out / "segmentation.json").unlink()
    assert cli.main(["segment", str(out)]) == 0
    assert (out / "segmentation.json").read_bytes() == written
    assert cli.main(["segment", str(out)]) == 0
    assert (out / "segmentation.json").read_bytes() == written


async def test_segment_reads_under_the_config_as_it_now_stands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jurisdiction: str
) -> None:
    out = await _ingest_one(monkeypatch, tmp_path, jurisdiction)
    with isolated_configs(monkeypatch, tmp_path / "edited", {COUNTRY: _config(segmentation=None)}):
        assert cli.main(["segment", str(out)]) == 0
    result = json.loads((out / "segmentation.json").read_text())
    assert (result["outcome"], len(result["segments"])) == ("single", 1)


def test_segment_refuses_a_bundle_without_page_spans(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"jurisdiction": "xa"}))
    (tmp_path / "source.txt").write_text("text")
    assert cli.main(["segment", str(tmp_path)]) == 2
    assert "page spans" in capsys.readouterr().err
    assert not (tmp_path / "segmentation.json").exists()
