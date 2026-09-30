"""A jurisdiction the pipeline does not know must stop it, not be papered over.

Without a config the pipeline does not fail: it falls back to defaults and emits
a plausible bundle for a jurisdiction it knows nothing about. That is the worst
shape for an external user, because nothing prompts them to look. It matters more
after the open-core split, where five of 257 configs ship and every other code a
user might name lands on this path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

import pytest

from codify.cli import _run
from codify.jurisdictions import (
    frbr_country,
    resolve_config,
    resolve_frbr_country,
    try_load_config,
)


def _args(source: pathlib.Path, out: pathlib.Path, jurisdiction: str) -> argparse.Namespace:
    return argparse.Namespace(
        source=str(source),
        out=str(out),
        jurisdiction=jurisdiction,
        model="unused",
        ocr_model=None,
        fallback_model=None,
        quiet=True,
    )


def test_a_missing_config_is_refused_before_any_work(tmp_path: pathlib.Path) -> None:
    """Refused early: no LLM client is built, no output directory is touched, and
    the exit code is non-zero so a script cannot read it as success."""
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    out = tmp_path / "bundle"

    code = asyncio.run(_run(_args(source, out, "zz")))

    assert code != 0
    assert not out.exists(), "a refused run must not leave a bundle behind"


def test_a_traversal_code_is_refused_the_same_way(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    assert asyncio.run(_run(_args(source, tmp_path / "b", "../../etc"))) != 0


def test_the_resolver_reports_where_it_looked() -> None:
    """The code alone cannot say whether a config was found, nor which one: two
    runs of `gb` against a config that changed between them are indistinguishable
    without the digest."""
    found = resolve_config("gb")
    assert found.found
    assert found.path == "data/jurisdictions/gb/config.json"
    assert len(found.sha256 or "") == 64

    missing = resolve_config("zz")
    assert not missing.found
    assert missing.path is None and missing.sha256 is None


def test_the_digest_tracks_the_file_rather_than_the_code(
    tmp_path: pathlib.Path,
) -> None:
    """What makes the manifest field provenance rather than decoration."""
    import codify.jurisdictions as j

    original = j.JURISDICTIONS_DIR
    try:
        j.JURISDICTIONS_DIR = tmp_path
        (tmp_path / "xx").mkdir()
        config = tmp_path / "xx" / "config.json"
        config.write_text(json.dumps({"code": "xx"}), encoding="utf-8")
        first = resolve_config("xx").sha256
        config.write_text(json.dumps({"code": "xx", "name": "Changed"}), encoding="utf-8")
        assert resolve_config("xx").sha256 != first
    finally:
        j.JURISDICTIONS_DIR = original


def test_frbr_country_maps_through_the_config() -> None:
    """`gb-eng` homes under `gb`, `eac` under `aa-eac`. The segment comes from
    the config, and a config supplied it, so `resolved` is True."""
    assert resolve_frbr_country("gb-eng") == ("gb", True)
    assert resolve_frbr_country("eac") == ("aa-eac", True)


def test_frbr_country_of_a_missing_config_is_flagged_as_guessed() -> None:
    """No config: the segment is guessed from the code, and `resolved` is False
    so a caller minting a citation identity can record it as guessed. This is the
    default path for every code an external user names after the open-core split."""
    guessed = resolve_frbr_country("zz")
    assert guessed == ("zz", False)
    assert frbr_country("zz") == "zz"


def test_the_builder_homes_a_default_doctype_under_the_config_country() -> None:
    """A doctype with no configured pattern falls to the canonical default, which
    must still map `gb-eng` to `gb`; otherwise it mints `/akn/gb-eng/...`, which
    the homing check and the ingest guard reject."""
    from codify.frbr import build_frbr_work_uri
    from codify.jurisdictions import work_uri_is_homed

    try_load_config.cache_clear()
    uri = build_frbr_work_uri("gb-eng", "unmapped_doctype", 2024, "1")
    assert uri == "/akn/gb/act/unmapped_doctype/2024/1"
    assert work_uri_is_homed(uri, "gb-eng")


@pytest.mark.parametrize("code", ["", "   "])
def test_an_empty_jurisdiction_is_refused(tmp_path: pathlib.Path, code: str) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    assert asyncio.run(_run(_args(source, tmp_path / "b", code))) != 0


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No gateway. The tests below are about config resolution, and a real client
    would make them depend on a reachable model and on the network."""
    import codify.cli as cli

    class _NoModel:
        """Constructs, then refuses on use. Raising at construction instead would
        abort `_run` before it reached the pipeline, and these tests are about
        what happens once it does."""

        def __getattr__(self, name: str) -> object:
            async def _refuse(*_args: object, **_kwargs: object) -> object:
                raise RuntimeError(f"no LLM in this test ({name})")

            return _refuse

    monkeypatch.setattr(cli, "create_llm_client", lambda *a, **k: _NoModel())


def test_a_known_jurisdiction_under_the_same_args_is_not_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. Without it the refusal tests above would pass just as well if
    `_run` were failing for some reason having nothing to do with the config: the
    run gets past the guard, writes its bundle, and records which config it used,
    even though there is no model behind it."""
    _offline(monkeypatch)
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    out = tmp_path / "bundle"

    assert asyncio.run(_run(_args(source, out, "gb"))) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["jurisdiction_config"]["code"] == "gb"
    assert manifest["jurisdiction_config"]["sha256"]


def test_the_resolved_code_is_what_the_pipeline_receives(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--jurisdiction GB` resolves the `gb` config. Passing the raw code on would
    have the pipeline look up `GB`, find nothing on a case-sensitive filesystem,
    and fall back to defaults, reinstating the silent fallback. macOS cannot show
    this: its filesystem is case-insensitive, so the assertion is on the code the
    run records rather than on a path lookup."""
    _offline(monkeypatch)
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    out = tmp_path / "bundle"

    assert asyncio.run(_run(_args(source, out, "  GB  "))) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["jurisdiction"] == "gb"
    assert manifest["jurisdiction_config"]["code"] == "gb"


@pytest.fixture(autouse=True)
def route_configs(tmp_path, monkeypatch):
    from tests.config_fixtures import isolated_configs

    configs = {
        "gb-eng": {"frbr": {"country_code": "gb"}},
        "eac": {"frbr": {"country_code": "aa-eac"}},
        "eu": {
            "frbr": {
                "country_code": "eu",
                "uri_patterns": {
                    "regulation": "/akn/eu/act/reg/{year}/{number}",
                    "directive": "/akn/eu/act/dir/{year}/{number}",
                },
            }
        },
    }
    with isolated_configs(monkeypatch, tmp_path / "data" / "jurisdictions", configs):
        yield


def test_the_manifest_carries_what_the_model_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`codify load` files the law under the manifest's metadata, so the fields the
    pipeline read must reach it, and only those."""
    import codify.cli as cli
    from codify.pipeline.events import Failed, MetadataExtracted

    read = {"title": "The Act", "year": "1992", "number": "7", "date": "1992-03-04", "chapter": "x"}

    async def fake_ingest(*_: object, **__: object):  # type: ignore[no-untyped-def]
        yield MetadataExtracted(metadata=read)
        yield Failed(stage="structure", error="stopped after metadata")

    monkeypatch.setattr(cli, "ingest_text", fake_ingest)
    monkeypatch.setattr(cli, "create_llm_client", lambda *a, **k: object())
    source = tmp_path / "doc.txt"
    source.write_text("Article 1\n(1) Text.\n", encoding="utf-8")
    out = tmp_path / "bundle"

    asyncio.run(_run(_args(source, out, "gb")))
    manifest = json.loads((out / "manifest.json").read_text())

    assert manifest["metadata"] == {
        "title": "The Act",
        "doctype": None,
        "year": "1992",
        "number": "7",
        "date": "1992-03-04",
    }
