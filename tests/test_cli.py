"""The bundle writer's own helpers.

`ingest-one` exists so a reader is not misled about what an ingest did, which
makes a wrong artifact worse here than a missing one. These cover the pure
helpers; the pipeline itself is covered in tests/parse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from codify.cli import _by_pass, _clear_bundle, _coverage_json, _foundry_ocr_url
from codify.core.llm import LiteLLMClient
from codify.pipeline.enrich.anchors import AnchorCoverage, StructuralAnchor
from codify.pipeline.enrich.structure import ScanTrace


def _trace(coverage: AnchorCoverage | None) -> ScanTrace:
    return ScanTrace(anchors=(), coverage=coverage, scaffold=None, fallback=None)


def test_an_unmeasurable_ratio_serialises_rather_than_raising() -> None:
    """The zero-anchor document is what the bundle is for, and it is the one
    whose ratio is None. Rounding it would kill the write after the model spend."""
    cov = AnchorCoverage(kind="article", ratio=None, captured=frozenset(), expected=frozenset())
    assert _coverage_json(_trace(cov)) == {
        "kind": "article",
        "ratio": None,
        "anchor_ratio": None,
        "captured": [],
        "expected": [],
        "missing": [],
        "masked": 0,
        "unclosed": 0,
    }


def test_a_measured_ratio_reports_the_numbers_it_missed() -> None:
    cov = AnchorCoverage(
        kind="article",
        ratio=0.5,
        captured=frozenset({"1", "2"}),
        expected=frozenset({"1", "2", "3", "4"}),
    )
    out = _coverage_json(_trace(cov))
    assert out is not None
    assert out["ratio"] == 0.5
    assert out["missing"] == ["3", "4"]


def test_a_gate_that_did_not_run_reports_no_coverage_block() -> None:
    assert _coverage_json(_trace(None)) is None


def test_a_reused_directory_keeps_nothing_from_the_previous_run(tmp_path: Path) -> None:
    """A shorter or failed second run would otherwise present the first run's
    scaffold, AKN and page images as its own."""
    pages = tmp_path / "pages"
    pages.mkdir()
    written = [
        tmp_path / "manifest.json",
        tmp_path / "final.akn.xml",
        tmp_path / "scaffold.bluebell",
        tmp_path / "coverage.json",
        pages / "page-001.png",
        pages / "page-002.png",
    ]
    for f in written:
        f.write_text("stale")
    _clear_bundle(tmp_path)
    assert not [f for f in written if f.exists()]


def test_every_file_the_bundle_writes_is_also_cleared(tmp_path: Path) -> None:
    """A name the writer adds but the tuple omits survives a re-run."""
    from codify.cli import _BUNDLE_FILES

    for name in _BUNDLE_FILES:
        (tmp_path / name).write_text("stale")
    _clear_bundle(tmp_path)
    assert not [n for n in _BUNDLE_FILES if (tmp_path / n).exists()]
    assert "ambiguity.jsonl" in _BUNDLE_FILES


def test_clearing_a_bundle_leaves_everything_else_alone(tmp_path: Path) -> None:
    """The output directory is the operator's choice, so a broader delete would
    take their own files with it."""
    keep = tmp_path / "notes.md"
    keep.write_text("mine")
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "reference.jpg").write_text("mine")
    _clear_bundle(tmp_path)
    assert keep.read_text() == "mine"
    assert (tmp_path / "pages" / "reference.jpg").exists()


def test_the_foundry_url_drops_the_gateway_path_suffix(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The endpoint env var carries a LiteLLM path; keeping it would send OCR to
    a 404 and every page would fall back, reading as an OCR-quality problem."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.services.ai.azure.com/api/p/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert _foundry_ocr_url() == "https://x.services.ai.azure.com/providers/mistral/azure/ocr"


def test_no_foundry_url_without_both_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert _foundry_ocr_url() is None
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert _foundry_ocr_url() is None


def test_an_anchor_with_no_recorded_pass_reads_as_unattributed() -> None:
    """`anchors_by_pass` is where an attribution regression becomes visible."""
    anchor = StructuralAnchor(
        kind="article", keyword="مادة", number="1", char_offset=0, line=1, matched_text="مادة 1"
    )
    assert _by_pass((anchor,)) == {"unattributed": 1}
    from dataclasses import replace

    assert _by_pass((replace(anchor, source_pass="regex"),)) == {"regex": 1}


@pytest.fixture
def _restore_structlog():
    """structlog's configuration is global and `main` repoints it at a
    `sys.stderr` pytest later closes."""
    import structlog

    saved = structlog.get_config().copy()
    yield
    structlog.configure(**saved)


def test_each_subcommand_routes_to_its_own_handler(
    monkeypatch, tmp_path: Path, _restore_structlog: None
) -> None:
    """The dispatch table is per-parser defaults: a subcommand added without
    `set_defaults` parses cleanly and dies on `args.func`."""
    from codify import cli

    fired: list[str] = []

    async def _fake_run(args: object) -> int:
        fired.append("ingest-one")
        return 0

    # `_run` stays a coroutine so the real `asyncio.run` is used; patching that
    # globally leaks into every async test that runs after this one.
    monkeypatch.setattr(cli, "_run", _fake_run)
    monkeypatch.setattr(cli, "_run_scan_corpus", lambda a: fired.append("scan-corpus") or 0)

    cli.main(["scan-corpus", str(tmp_path), "--jurisdiction", "ps"])
    cli.main(["ingest-one", "x.txt", "--jurisdiction", "ps", "--out", str(tmp_path)])
    assert fired == ["scan-corpus", "ingest-one"]


def test_a_mistargeted_root_exits_non_zero_without_printing(
    capsys, tmp_path: Path, _restore_structlog: None
) -> None:
    """A redirect keeps the file whatever the exit code says."""
    from codify.cli import main

    assert main(["scan-corpus", str(tmp_path), "--jurisdiction", "ps"]) == 1
    assert capsys.readouterr().out == ""
    assert main(["scan-corpus", str(tmp_path / "nope.txt"), "--jurisdiction", "ps"]) == 2


def test_an_unscannable_corpus_says_why_on_stderr(
    capsys, tmp_path: Path, _restore_structlog: None
) -> None:
    from pypdf import PdfWriter

    from codify.cli import main

    blank = PdfWriter()
    blank.add_blank_page(width=10, height=10)
    blank.write(tmp_path / "scan.pdf")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "law.en.txt").write_text("Article 1", encoding="utf-8")
    assert main(["scan-corpus", str(tmp_path), "--jurisdiction", "xa"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unreadable 1, no text layer 1, derived exports 1" in captured.err


def test_the_structuring_pass_asks_for_the_counts_the_bundle_writes() -> None:
    """The artifact reads whatever that pass measured, so a field can be wired
    end to end and still always zero if the pass never asks for it. That is
    what it did: the gate saw two hidden markers where the bundle wrote none.

    Asserted on the call rather than through the pass, which needs a model.
    """
    import inspect

    from codify.jurisdictions import load_config
    from codify.pipeline.enrich import structure as structure_mod
    from codify.pipeline.enrich.anchors import (
        anchor_coverage,
        build_anchor_regex,
        scan_anchors,
    )

    source = inspect.getsource(structure_mod)
    assert "anchor_coverage(text, anchors, config, doctype, kind, with_masked=True)" in source

    config = load_config("xl")
    body = "".join(f"Pasal {i}\n(1) Ketentuan nomor {i}.\n" for i in range(1, 21))
    text = body + "Pasal 21\n(1) Satu dengan “kutip hilang.\n(2) Dua.\n(3) Tiga.\n"
    anchors = scan_anchors(text, build_anchor_regex(config, "act"), country="xl", doctype="act")

    # Off by default, so nothing pays for a count it does not read.
    assert anchor_coverage(text, anchors, config, "act", "article").masked == 0

    coverage = anchor_coverage(text, anchors, config, "act", "article", with_masked=True)
    written = _coverage_json(_trace(coverage))
    assert written is not None
    assert written["masked"] == 2, written
    assert written["unclosed"] == 1, written


async def test_the_client_is_built_off_the_proxy(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Proxy attribution rides the request body as `metadata`, which a provider's
    own OpenAI-compatible endpoint rejects with a 400; the CLI has no run to
    attribute anyway."""
    from codify import cli

    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "create_llm_client", _capture)
    source = tmp_path / "act.txt"
    source.write_text("1. Short title.\n")
    args = argparse.Namespace(
        source=str(source),
        jurisdiction="xa",
        out=str(tmp_path / "bundle"),
        model="m",
        ocr_model="",
        fallback_model="",
        quiet=True,
    )
    with pytest.raises(_Stop):
        await cli._run(args)
    assert seen["telemetry_mode"] == "direct"
    # The mode is only worth pinning because of what it drops from the request.
    direct = LiteLLMClient(base_url="http://x/v1", api_key="k", model="m", telemetry_mode="direct")
    assert direct._body() == {}


def test_help_keeps_the_docstring_examples_on_their_own_lines(capsys) -> None:  # type: ignore[no-untyped-def]
    from codify.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "\n    codify ingest-one gazette.pdf --jurisdiction xa --out bundle/\n" in out
    assert "\n    codify scan-corpus ~/corpus --jurisdiction xa\n" in out


async def test_the_fallback_model_reaches_the_client(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Without it a content-filter refusal has nowhere to go, however the client
    compares finish reasons."""
    from codify import cli

    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "create_llm_client", _capture)
    source = tmp_path / "act.txt"
    source.write_text("1. Short title.\n")
    for flag, expected in (("other-model", "other-model"), ("", None)):
        args = argparse.Namespace(
            source=str(source),
            jurisdiction="xa",
            out=str(tmp_path / "bundle"),
            model="m",
            ocr_model="",
            fallback_model=flag,
            quiet=True,
        )
        with pytest.raises(_Stop):
            await cli._run(args)
        assert seen["fallback_model"] == expected


def test_the_fallback_flag_defaults_to_the_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from codify import cli

    captured: list[argparse.Namespace] = []
    monkeypatch.setenv("LITELLM_CONTENT_FILTER_FALLBACK_MODEL", "env-fallback")
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    monkeypatch.setattr(cli.asyncio, "run", lambda coro: coro.close() or 0)
    monkeypatch.setattr(cli, "_run", lambda a: captured.append(a) or _noop())
    cli.main(["ingest-one", "x.txt", "--jurisdiction", "xa", "--out", "b"])
    assert captured[0].fallback_model == "env-fallback"


async def _noop() -> int:
    return 0


def test_coverage_reads_the_lowest_of_what_was_measured() -> None:
    """Anchors all found reads 1.0; a body-fill that wrote nothing and a page that
    read empty must pull the headline down with them."""
    from codify.pipeline.enrich.structure import BodyFillTrace

    cov = AnchorCoverage(
        kind="section", ratio=1.0, captured=frozenset({"1", "2"}), expected=frozenset({"1", "2"})
    )
    fill = BodyFillTrace(
        windows=1, calls=3, calls_failed=3, expected=2, verbatim=("s1",), empty=("s2",)
    )
    trace = ScanTrace(anchors=(), coverage=cov, body_fill=fill)
    out = _coverage_json(trace, pages=4, unreadable={3: "empty_read"})
    assert out is not None
    assert (out["ratio"], out["anchor_ratio"], out["body_fill"]["ratio"], out["pages"]) == (
        0.0,
        1.0,
        0.0,
        {"total": 4, "unreadable": [3], "ratio": 0.75},
    )


def test_an_unreadable_page_alone_lowers_coverage() -> None:
    cov = AnchorCoverage(kind="section", ratio=1.0, captured=frozenset(), expected=frozenset())
    out = _coverage_json(_trace(cov), pages=4, unreadable={3: "empty_read"})
    assert out is not None and out["ratio"] == 0.75


async def test_an_unanswered_ingest_reads_blocking(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The bundle must not read as a result when no model call succeeded: the
    grade blocks and coverage is zero, though every anchor was found."""
    import json

    from codify import cli

    class _Unreachable:
        def __getattr__(self, name: str) -> object:
            async def _fail(*_a: object, **_k: object) -> object:
                raise ConnectionError("unreachable")

            return _fail

    monkeypatch.setattr(cli, "create_llm_client", lambda **_k: _Unreachable())
    source = tmp_path / "act.txt"
    source.write_text(
        "PART I\nPRELIMINARY\n\nSection 1\nThis Act may be cited as the Widgets Act.\n\n"
        "Section 2\nIn this Act a widget is a device.\n"
    )
    out = tmp_path / "bundle"
    args = argparse.Namespace(
        source=str(source),
        jurisdiction="xa",
        out=str(out),
        model="m",
        ocr_model="",
        fallback_model="",
        quiet=True,
    )
    await cli._run(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["grade"]["grade"],
        manifest["coverage"]["anchor_ratio"],
        manifest["coverage"]["ratio"],
    ) == ("blocking", 1.0, 0.0)
    assert "body_fill_failed" in manifest["grade"]["reason"], manifest["grade"]


def test_measured_pages_alone_make_a_coverage_block() -> None:
    """An anchorless scan measures nothing else, and its lost page must still show."""
    out = _coverage_json(None, pages=2, unreadable={2: "empty_read"})
    assert out == {"pages": {"total": 2, "unreadable": [2], "ratio": 0.5}, "ratio": 0.5}


class _EchoOrFail:
    """Fills bodies, or with ``fail`` raises on every call, as an unreachable model."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def chat_schema(self, prompt, schema, system=None, model=None):  # type: ignore[no-untyped-def]
        from codify.pipeline.enrich.scaffold import BodyBlock, BodyFillResponse

        if self.fail or schema is not BodyFillResponse:
            raise RuntimeError("offline")
        eids = __import__("re").findall(r"eid=(\S+)", prompt)
        return BodyFillResponse(bodies=[BodyBlock(eid=e, lines=["Body."]) for e in eids])

    def __getattr__(self, name: str) -> object:
        async def _fail(*_a: object, **_k: object) -> object:
            raise RuntimeError("offline")

        return _fail


async def _bundle_grade(monkeypatch, tmp_path: Path, *, fail: bool, crash: bool) -> str:  # type: ignore[no-untyped-def]
    import json

    from codify import cli
    from codify.pipeline.formats import pdf as pdf_mod

    def _crash(*_a: object, **_k: object) -> list[dict[str, object]]:
        raise RuntimeError("validator crashed")

    if crash:
        monkeypatch.setattr(pdf_mod, "validate_akn", _crash)
    monkeypatch.setattr(cli, "create_llm_client", lambda **_k: _EchoOrFail(fail))
    source = tmp_path / "act.txt"
    source.write_text("PART I\nPRELIMINARY\n\nSection 1\nThis Act may be cited.\n")
    out = tmp_path / "bundle"
    args = argparse.Namespace(
        source=str(source),
        jurisdiction="xa",
        out=str(out),
        model="m",
        ocr_model="",
        fallback_model="",
        quiet=True,
    )
    await cli._run(args)
    return str(json.loads((out / "manifest.json").read_text())["grade"]["grade"])


@pytest.mark.parametrize(
    "fail,crash,expected",
    [(False, True, "ungraded"), (True, True, "blocking"), (False, False, "warning")],
    ids=["validator-crashed", "halt-outranks-crash", "validator-finished"],
)
async def test_the_grade_knows_whether_the_validator_finished(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path: Path, fail: bool, crash: bool, expected: str
) -> None:
    """No findings because nothing was checked must not read as clean, and a
    refusal is known even when the validator did not finish."""
    assert await _bundle_grade(monkeypatch, tmp_path, fail=fail, crash=crash) == expected
