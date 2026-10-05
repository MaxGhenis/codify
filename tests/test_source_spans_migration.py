"""Migration 0023 sits on the single chain and touches `versions` without a long lock."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "0023_source_spans"


def test_source_spans_is_the_head_after_label_provisions() -> None:
    scripts = ScriptDirectory.from_config(Config(str(_ROOT / "alembic.ini")))
    assert scripts.get_heads() == [_REVISION]
    rev = scripts.get_revision(_REVISION)
    assert rev is not None and rev.down_revision == "0022_label_provisions_kind"


def test_versions_is_altered_under_a_short_lock_and_indexed_concurrently() -> None:
    source = (_ROOT / "codify" / "migrations" / "versions" / f"{_REVISION}.py").read_text()
    assert "SET lock_timeout = '5s'" in source
    assert "NOT VALID" in source and "VALIDATE CONSTRAINT" in source
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX}" in source
