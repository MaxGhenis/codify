"""Alembic environment for the core corpus schema.

Independent of the product chain in ``apps/api``: it runs on its own version
table (``alembic_version_core``) so the open-source core can build and evolve
the corpus schema on its own, in the same database as the product or standalone.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from logging.config import fileConfig
from typing import Literal

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from sqlalchemy.engine import make_url

_IncludeNameType = Literal[
    "schema",
    "table",
    "column",
    "index",
    "unique_constraint",
    "foreign_key_constraint",
    "check_constraint",
]
_ParentNames = MutableMapping[
    Literal["schema_name", "table_name", "schema_qualified_table_name"], str | None
]

# Own version table: the product chain owns the default `alembic_version`.
_VERSION_TABLE = "alembic_version_core"

# Migrations fail fast instead of hanging behind long-running queries that hold
# AccessShare on shared tables; the deploy operator drains and retries.
_LOCK_TIMEOUT = "30s"

# The tables this chain owns. Used to scope autogenerate for future core
# migrations so a shared database's product tables are never proposed for drop.
CORE_TABLES = frozenset(
    {
        "jurisdictions",
        "source_documents",
        "laws",
        "versions",
        "version_source_texts",
        "page_reads",
        "source_spans",
        "source_span_pages",
        "sections",
        "provisions",
        "search_terms",
        "cross_references",
        "goods_code_references",
        "annex_rows",
        "amendment_effects",
        "registry_works",
        "law_links",
        "lifecycle_events",
        "provision_embeddings",
        "version_unit_embeddings",
        "findings",
        "structural_findings",
        "finding_notes",
        "examinations",
        "page_read_disputes",
        "repair_proposals",
        "adjudications",
        "scheme_matches",
        "lens_runs",
        "lens_run_directives",
        "feedback",
        "events",
        "acquisitions",
        "translation_runs",
        "runs",
        "run_artifacts",
        "anticorruption_factors",
        "anticorruption_phenomena",
        "anticorruption_schemes",
    }
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Hand-written migrations; no autogenerate target metadata today.
target_metadata = None


def _resolve_url() -> str:
    raw = os.environ.get("POSTGRES_URL")
    if not raw:
        if os.environ.get("ENVIRONMENT") in {None, "", "localhost", "development"}:
            raw = "postgresql://codify:codify@localhost:5432/codify"
        else:
            raise RuntimeError(
                "POSTGRES_URL must be set when ENVIRONMENT is not localhost/development"
            )
    url = make_url(raw)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


def _include_name(name: str | None, type_: _IncludeNameType, parent_names: _ParentNames) -> bool:
    # Only relevant once a core migration uses autogenerate: keep the shared
    # database's product tables out of the diff.
    if type_ == "table":
        return name in CORE_TABLES
    return True


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=_VERSION_TABLE,
            include_name=_include_name,
        )
        with context.begin_transaction():
            connection.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT}'"))
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError(
        "The core migration chain needs a live database connection: the baseline "
        "builds or adopts based on the existing schema, so --sql offline mode is "
        "not supported."
    )
run_migrations_online()
