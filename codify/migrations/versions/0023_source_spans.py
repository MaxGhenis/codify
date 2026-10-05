"""Add source_spans: the acts one uploaded source holds, as page ranges over it.

A span tiles part of its source: an act, a region held for review, a skipped
non-act, or front matter. Each cut is kept as a page, a marker line and an offset,
so a later re-read can find it again. Generations are append-only: a re-split
retires the live one and writes the next. `source_span_pages` links a span to
the page reads it draws on, and `versions.source_span_id` names a child's span.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0023_source_spans"
down_revision: str | None = "0022_label_provisions_kind"
branch_labels: str | None = None
depends_on: str | None = None

_FK = "versions_source_span_id_fkey"
_INDEX = "versions_source_span_idx"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS source_spans (
            id uuid PRIMARY KEY,
            source_sha256 text NOT NULL
                REFERENCES source_documents(sha256) ON DELETE CASCADE,
            generation integer NOT NULL,
            ordinal integer NOT NULL,
            kind text NOT NULL,
            first_page integer NOT NULL,
            last_page integer NOT NULL,
            start_marker text NOT NULL DEFAULT '',
            start_offset integer NOT NULL,
            end_offset integer,
            issue text NOT NULL DEFAULT '',
            heading text NOT NULL DEFAULT '',
            act_key text NOT NULL DEFAULT '',
            signals text[] NOT NULL DEFAULT '{}',
            reason text NOT NULL DEFAULT '',
            created_at timestamptz NOT NULL DEFAULT now(),
            retired_at timestamptz,
            CONSTRAINT source_spans_generation_ordinal_key
                UNIQUE (source_sha256, generation, ordinal),
            CONSTRAINT source_spans_kind_check
                CHECK (kind IN ('act', 'held', 'skipped', 'front_matter')),
            CONSTRAINT source_spans_pages_check
                CHECK (first_page >= 1 AND last_page >= first_page),
            CONSTRAINT source_spans_offsets_check
                CHECK (start_offset >= 0 AND (end_offset IS NULL OR end_offset >= 0))
        )
        """
    )
    # The live generation is what every reader asks for.
    op.execute(
        "CREATE INDEX IF NOT EXISTS source_spans_live_idx "
        "ON source_spans (source_sha256, ordinal) WHERE retired_at IS NULL"
    )
    # Restrict, not cascade: a read a span still draws on must not vanish quietly.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS source_span_pages (
            span_id uuid NOT NULL REFERENCES source_spans(id) ON DELETE CASCADE,
            page_read_id uuid NOT NULL REFERENCES page_reads(id) ON DELETE RESTRICT,
            PRIMARY KEY (span_id, page_read_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS source_span_pages_read_idx ON source_span_pages (page_read_id)"
    )
    # `versions` is live and large: wait briefly for its lock, never queue reads behind it.
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE versions ADD COLUMN IF NOT EXISTS source_span_id uuid")
    op.execute(
        f"""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{_FK}') THEN
                ALTER TABLE versions ADD CONSTRAINT {_FK} FOREIGN KEY (source_span_id)
                    REFERENCES source_spans(id) ON DELETE SET NULL NOT VALID;
            END IF;
        END $$
        """
    )
    # The column is all NULL, so validation scans without blocking writes.
    with op.get_context().autocommit_block():
        op.execute(f"ALTER TABLE versions VALIDATE CONSTRAINT {_FK}")
        # A cancelled concurrent build leaves an unusable index under the name.
        if _index_valid() is False:
            op.execute(f"DROP INDEX CONCURRENTLY {_INDEX}")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX} "
            "ON versions (source_span_id) WHERE source_span_id IS NOT NULL"
        )


def _index_valid() -> bool | None:
    """Whether the index exists and is usable; None when it does not exist."""
    row = (
        op.get_bind()
        .execute(
            sa.text("SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass(:n)"),
            {"n": _INDEX},
        )
        .one_or_none()
    )
    return None if row is None else bool(row[0])


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX}")
    op.execute("SET lock_timeout = '5s'")
    op.execute(f"ALTER TABLE versions DROP CONSTRAINT IF EXISTS {_FK}")
    op.execute("ALTER TABLE versions DROP COLUMN IF EXISTS source_span_id")
    op.execute("DROP TABLE IF EXISTS source_span_pages")
    op.execute("DROP TABLE IF EXISTS source_spans")
