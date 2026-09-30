"""A config that is absent or does not validate raises where it is read.

`load_config` raises on absence so no caller builds a thinner result from
nothing; these are the readers that used to catch that and return a default.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codify import jurisdictions
from codify.jurisdictions import JurisdictionConfigError, try_load_config
from codify.pipeline.enrich import anchors, titles
from codify.repair import dossier


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A jurisdictions tree holding one config that does not validate, `zz`."""
    (tmp_path / "zz").mkdir()
    (tmp_path / "zz" / "config.json").write_text(json.dumps({"code": "zz"}))
    monkeypatch.setattr(jurisdictions, "JURISDICTIONS_DIR", tmp_path)
    caches = (try_load_config, anchors._example_markers_for, anchors._trigger_phrases_for)
    for cached in caches:
        cached.cache_clear()
    yield tmp_path
    for cached in caches:
        cached.cache_clear()


_AKN = (
    '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act><meta/>'
    '<body><section eId="sec_1"><num>1</num><content><p>t</p></content></section>'
    "</body></act></akomaNtoso>"
)

_READERS: list[tuple[str, Callable[[str], Any]]] = [
    ("marginal_note_kind", lambda c: anchors._marginal_note_kind(c, "act")),
    ("declared_marker_entries", lambda c: anchors._declared_marker_entries(c, "act")),
    ("attachment_hierarchies", anchors._attachment_hierarchies),
    ("attachment_outlines", lambda c: anchors._scan_attachment_outlines("", [], 0, c)),
    ("basic_unit_line_re", anchors._basic_unit_line_re),
    ("example_markers", anchors._example_markers_for),
    ("trigger_phrases", anchors._trigger_phrases_for),
    ("rank_map", lambda c: anchors._rank_map_for(c, "act")),
    ("abbrev_map", lambda c: anchors._abbrev_map_for(c, "act")),
    ("short_title", lambda c: titles.declared_short_title("known as the Test Act", c)),
    ("designation_rule", titles._designation_rule),
    ("long_title_lead_ins", titles.long_title_lead_ins),
    ("closing_phrases", lambda c: dossier._closing_phrases(c, "2001")),
]


@pytest.mark.parametrize("name,read", _READERS, ids=[n for n, _ in _READERS])
def test_an_absent_config_raises(data_dir: Path, name: str, read: Callable[[str], Any]) -> None:
    with pytest.raises(JurisdictionConfigError):
        read("qq")


@pytest.mark.parametrize("name,read", _READERS, ids=[n for n, _ in _READERS])
def test_a_config_that_does_not_validate_raises(
    data_dir: Path, name: str, read: Callable[[str], Any]
) -> None:
    with pytest.raises(ValidationError):
        read("zz")


def test_a_blank_code_is_still_no_jurisdiction(data_dir: Path) -> None:
    """Callers pass "" for a document with no jurisdiction; that is not a fault."""
    assert [read("") for _, read in _READERS[:9]] == [None, (), (), [], None, (), (), None, {}]


def test_the_region_vocabulary_fault_fails_the_enrich_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the vocabulary faults: every pass after it has a config to read."""
    import asyncio

    from codify.pipeline import stages
    from codify.pipeline.stages import Descriptors, run_enrich_passes

    def _broken(*_a: object, **_k: object) -> Any:
        raise JurisdictionConfigError("no config")

    monkeypatch.setattr(stages, "vocabulary_for_jurisdiction", _broken)
    desc = Descriptors(
        title="t", raw_date="2015", number="9", year="2015", language="eng", doctype="act"
    )
    with pytest.raises(JurisdictionConfigError, match="no config"):
        asyncio.run(
            run_enrich_passes(
                _AKN,
                llm=object(),  # type: ignore[arg-type]
                jurisdiction_code="xa",
                desc=desc,
                skip_external_refs=True,
            )
        )


def test_the_dossier_regions_fault_is_raised(data_dir: Path) -> None:
    from codify.pipeline.enrich.ocr import PageLayout

    class _Read:
        page_number = 1
        layout = PageLayout(engine="test").model_dump()

    with pytest.raises(JurisdictionConfigError):
        dossier._flagged_regions([_Read()], country="qq", year="2001")  # type: ignore[list-item]


@pytest.mark.parametrize("seam", ["emit_references", "emit_inline_markup"])
def test_a_config_fault_inside_an_enrich_pass_propagates(
    monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    """Every other pass failure is logged and skipped; a config fault is not."""
    import asyncio

    from codify.pipeline import stages
    from codify.pipeline.stages import Descriptors, run_enrich_passes

    def _sync(*_a: object, **_k: object) -> str:
        raise JurisdictionConfigError("no config")

    async def _async(*_a: object, **_k: object) -> str:
        raise JurisdictionConfigError("no config")

    monkeypatch.setattr(stages, seam, _async if seam == "emit_inline_markup" else _sync)
    desc = Descriptors(
        title="t", raw_date="2015", number="9", year="2015", language="eng", doctype="act"
    )
    with pytest.raises(JurisdictionConfigError):
        asyncio.run(
            run_enrich_passes(
                _AKN,
                llm=object(),  # type: ignore[arg-type]
                jurisdiction_code="xa",
                desc=desc,
                skip_external_refs=True,
            )
        )
