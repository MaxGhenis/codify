"""The label_provisions migration widens runs_kind_check and nothing else."""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection

_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "0022_label_provisions_kind"


def _migration() -> ModuleType:
    path = _ROOT / f"codify/migrations/versions/{_REVISION}.py"
    spec = importlib.util.spec_from_file_location("label_provisions_kind", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_is_the_single_head_after_static_site_export() -> None:
    scripts = ScriptDirectory.from_config(Config(str(_ROOT / "alembic.ini")))
    assert scripts.get_heads() == [_REVISION]
    rev = scripts.get_revision(_REVISION)
    assert rev is not None and rev.down_revision == "0021_static_site_export_kind"


def test_adds_only_the_new_run_kind() -> None:
    assert _migration()._RUN_KINDS == ("label_provisions",)


@pytest.fixture
def conn() -> Iterator[Connection]:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    from codify.testing import postgres_url

    engine = create_engine(make_url(postgres_url()).set(drivername="postgresql+psycopg"))
    try:
        with engine.connect() as connection, connection.begin():
            schema = "ssekind_" + uuid.uuid4().hex
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}", public')
            connection.exec_driver_sql(
                "CREATE TABLE runs (id integer PRIMARY KEY, kind text NOT NULL, "
                "CONSTRAINT runs_kind_check CHECK (kind IN ('retire_law', 'other_kind')))"
            )
            connection.exec_driver_sql("INSERT INTO runs VALUES (1, 'other_kind')")
            yield connection
            connection.rollback()
    finally:
        engine.dispose()


def _invoke(connection: Connection, function: str) -> None:
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with Operations.context(MigrationContext.configure(connection)):
        getattr(_migration(), function)()


def _definition(connection: Connection) -> str:
    return str(
        connection.exec_driver_sql(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'runs_kind_check' "
            "AND conrelid = 'runs'::regclass"
        ).scalar_one()
    )


@pytest.mark.integration
def test_upgrade_admits_kind_idempotently_and_downgrade_restores(conn: Connection) -> None:
    before = _definition(conn)
    _invoke(conn, "upgrade")
    after = _definition(conn)
    assert "label_provisions" in after and "other_kind" in after
    conn.exec_driver_sql("INSERT INTO runs VALUES (2, 'label_provisions')")
    conn.exec_driver_sql("DELETE FROM runs WHERE id = 2")
    _invoke(conn, "upgrade")
    assert _definition(conn) == after
    _invoke(conn, "downgrade")
    assert _definition(conn) == before
    assert conn.exec_driver_sql("SELECT kind FROM runs").scalar_one() == "other_kind"


@pytest.mark.integration
def test_populated_downgrade_refuses(conn: Connection) -> None:
    _invoke(conn, "upgrade")
    conn.exec_driver_sql("INSERT INTO runs VALUES (2, 'label_provisions')")
    before = _definition(conn)
    with pytest.raises(RuntimeError, match="retains .* evidence"):
        _invoke(conn, "downgrade")
    assert _definition(conn) == before
