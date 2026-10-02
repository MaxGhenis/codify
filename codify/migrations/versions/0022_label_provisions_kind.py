"""Admit the label_provisions run kind on the canonical chain.

Widens runs_kind_check only; existing kinds and rows are preserved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022_label_provisions_kind"
down_revision: str | Sequence[str] | None = "0021_static_site_export_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RUN_KINDS = ("label_provisions",)
_CONSTRAINTS = {
    "runs": ("runs_kind_check", _RUN_KINDS),
}


def _parse_kinds(expression: str) -> list[str]:
    # Accept only PostgreSQL's deparse of a kind IN literal-list check. Parsing
    # literals from an arbitrary expression could silently remove another guard.
    literal = r"'[a-z][a-z0-9_.]*'"
    if any(not re.fullmatch(literal, token) for token in re.findall(r"'[^']*'", expression)):
        raise RuntimeError("Incompatible kind constraint literal")
    compact = re.sub(r"::(?:text\[\]|text|character varying)", "", expression)
    compact = re.sub(r"[\s()]", "", compact)
    if not re.fullmatch(rf"kind=ANYARRAY\[{literal}(?:,{literal})*\]", compact):
        raise RuntimeError("Incompatible kind constraint expression")
    kinds = re.findall(literal, compact)
    return list(dict.fromkeys(kind[1:-1] for kind in kinds))


def _current_kinds(table: str, constraint: str) -> list[str]:
    row = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT contype, convalidated, connoinherit, pg_get_expr(conbin, conrelid) "
                "FROM pg_constraint WHERE conrelid = to_regclass(:table) AND conname = :name"
            ),
            {"table": table, "name": constraint},
        )
        .one_or_none()
    )
    if row is None or tuple(row[:3]) != ("c", True, False):
        raise RuntimeError(f"Missing or incompatible {table}.{constraint}")
    return _parse_kinds(row[3])


def _locked_definitions() -> dict[str, list[str]]:
    # Prevent inserts and competing DDL between validation and the swap.
    op.execute("LOCK TABLE runs IN ACCESS EXCLUSIVE MODE")
    return {table: _current_kinds(table, name) for table, (name, _) in _CONSTRAINTS.items()}


def _replace(table: str, constraint: str, kinds: list[str]) -> None:
    if not kinds:
        raise RuntimeError(f"Refusing empty kind vocabulary for {table}")
    op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
    values = ", ".join(f"'{kind}'" for kind in kinds)
    op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} CHECK (kind IN ({values}))")


def upgrade() -> None:
    definitions = _locked_definitions()
    for table, (constraint, added) in _CONSTRAINTS.items():
        kinds = definitions[table]
        missing = [kind for kind in added if kind not in kinds]
        if missing:
            _replace(table, constraint, [*kinds, *missing])


def downgrade() -> None:
    definitions = _locked_definitions()
    for table, (_, added) in _CONSTRAINTS.items():
        for kind in added:
            present = (
                op.get_bind()
                .execute(
                    sa.text(f"SELECT EXISTS (SELECT 1 FROM {table} WHERE kind = :kind)"),
                    {"kind": kind},
                )
                .scalar_one()
            )
            if present:
                raise RuntimeError(f"Cannot downgrade: {table} retains {kind} evidence")
    for table, (constraint, added) in _CONSTRAINTS.items():
        kinds = definitions[table]
        retained = [kind for kind in kinds if kind not in added]
        if retained != kinds:
            _replace(table, constraint, retained)
