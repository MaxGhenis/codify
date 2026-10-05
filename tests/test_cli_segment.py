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
from tests.pipeline.test_segment import (
    COUNTRY,
    SEGMENTATION,
    _act,
    _config,
    _signed,
    _three_act_issue,
)

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


class _Offline:
    def __getattr__(self, name: str) -> object:
        async def _fail(*_a: object, **_k: object) -> object:
            raise ConnectionError("offline")

        return _fail


async def _ingest_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: bytes) -> Path:
    monkeypatch.setattr(cli, "create_llm_client", lambda **_k: _Offline())
    source = tmp_path / "acts.txt"
    source.write_bytes(body)
    out = tmp_path / "bundle"
    args = argparse.Namespace(
        source=str(source),
        jurisdiction=COUNTRY,
        out=str(out),
        model="m",
        ocr_model="",
        fallback_model="",
        quiet=True,
    )
    await cli._run(args)
    return out


async def test_a_crlf_text_source_segments_the_same_on_a_re_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jurisdiction: str
) -> None:
    """Offsets index source.txt as written; a newline translated on reading moves them."""
    lines = [*_act(3, "THE HARBOUR DUES ACT", 4), *_signed(), *_act(4, "THE PILOTS ACT", 3)]
    out = await _ingest_text(monkeypatch, tmp_path, "\r\n".join(lines).encode())
    written = (out / "segmentation.json").read_bytes()
    assert json.loads(written)["outcome"] == "decided"
    spans = json.loads((out / "page_spans.json").read_text())
    assert spans == {"page_count": 0, "pages": [], "furniture": []}
    assert cli.main(["segment", str(out)]) == 0
    assert (out / "segmentation.json").read_bytes() == written


async def test_a_segmenter_fault_leaves_the_bundle_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jurisdiction: str
) -> None:
    def _raise(*_a: object, **_k: object) -> object:
        raise RuntimeError("segmenter fault")

    monkeypatch.setattr(cli, "segment", _raise)
    out = await _ingest_one(monkeypatch, tmp_path, jurisdiction)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["segmentation_failed"] == "RuntimeError: segmenter fault"
    assert not (out / "segmentation.json").exists()
    assert (out / "page_spans.json").exists() and (out / "source.txt").exists()


_SPAN = {"page": 1, "method": "text_layer"}


@pytest.mark.parametrize(
    "spans",
    [
        [],
        {"page_count": 1, "pages": [{"page": 1}], "furniture": []},
        {"page_count": 1, "pages": [], "furniture": [{"header": "x"}]},
        {"page_count": 1, "pages": [{**_SPAN, "start": 3, "end": 1}], "furniture": []},
        {"page_count": 1, "pages": [{**_SPAN, "start": 0, "end": 9}], "furniture": []},
        {"page_count": 1, "pages": [{**_SPAN, "start": -1, "end": 2}], "furniture": []},
        {
            "page_count": 2,
            "pages": [{**_SPAN, "start": 0, "end": 3}, {**_SPAN, "page": 2, "start": 2, "end": 4}],
            "furniture": [],
        },
        {"page_count": 1, "pages": [], "furniture": [{"page": float("inf"), "header": "x"}]},
        {"page_count": 1, "furniture": []},
        {"page_count": 1, "pages": [], "furniture": None},
        {"page_count": 1, "pages": [{**_SPAN, "page": "1", "start": 0, "end": 2}], "furniture": []},
        {"page_count": 1, "pages": [], "furniture": [{"page": 1.5, "header": "x"}]},
        {"page_count": 1, "pages": [], "furniture": [{"page": True, "header": "x"}]},
        {"page_count": 1, "pages": [], "furniture": [{"page": 1, "header": 7}]},
        {"pages": [], "furniture": []},
        {"page_count": -1, "pages": [], "furniture": []},
        {"page_count": 1, "pages": [{**_SPAN, "page": 2, "start": 0, "end": 2}], "furniture": []},
        {"page_count": 2, "pages": [], "furniture": [{"page": 0, "header": "x"}]},
        {
            "page_count": 2,
            "pages": [{**_SPAN, "start": 0, "end": 1}, {**_SPAN, "start": 2, "end": 3}],
            "furniture": [],
        },
        {"page_count": 1, "pages": [{**_SPAN, "start": 1, "end": 1}], "furniture": []},
        {
            "page_count": 2,
            "pages": [{**_SPAN, "page": 2, "start": 0, "end": 1}, {**_SPAN, "start": 2, "end": 3}],
            "furniture": [],
        },
        {
            "page_count": 1,
            "pages": [],
            "furniture": [{"page": 1, "header": "a"}, {"page": 1, "header": "b"}],
        },
        {"page_count": 1, "pages": [{**_SPAN, "start": 1, "end": 4}], "furniture": []},
        {"page_count": 1, "pages": [{**_SPAN, "start": 0, "end": 3}], "furniture": []},
        {
            "page_count": 2,
            "pages": [{**_SPAN, "start": 0, "end": 1}, {**_SPAN, "page": 2, "start": 2, "end": 4}],
            "furniture": [],
        },
    ],
    ids=[
        "not-an-object",
        "span-missing-offsets",
        "furniture-missing-page",
        "span-reversed",
        "span-past-the-text",
        "span-before-the-text",
        "spans-overlapping",
        "furniture-page-infinite",
        "pages-missing",
        "furniture-not-a-list",
        "span-page-a-string",
        "furniture-page-fractional",
        "furniture-page-boolean",
        "furniture-header-not-text",
        "page-count-missing",
        "page-count-negative",
        "span-page-past-the-count",
        "furniture-page-zero",
        "span-page-repeated",
        "span-empty",
        "span-pages-backwards",
        "furniture-page-repeated",
        "span-starting-late",
        "span-ending-short",
        "spans-with-a-gap",
    ],
)
def test_segment_refuses_malformed_page_spans(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spans: object
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"jurisdiction": "xa"}))
    (tmp_path / "source.txt").write_text("text")
    (tmp_path / "page_spans.json").write_text(json.dumps(spans))
    assert cli.main(["segment", str(tmp_path)]) == 2
    assert "not a readable bundle" in capsys.readouterr().err


def test_segment_refuses_a_manifest_that_is_not_an_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "manifest.json").write_text("[]")
    (tmp_path / "source.txt").write_text("text")
    (tmp_path / "page_spans.json").write_text(
        json.dumps({"page_count": 0, "pages": [], "furniture": []})
    )
    assert cli.main(["segment", str(tmp_path)]) == 2
    assert "not a readable bundle" in capsys.readouterr().err
