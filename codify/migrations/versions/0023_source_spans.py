"""Add source_spans: the acts one uploaded source holds, as page ranges over it.

A span tiles part of its source: an act, a region held for review, a skipped
non-act, or front matter. Each cut is kept as a page, a marker line and an offset,
so a later re-read can find it again. Generations are append-only: a re-split
retires the live one and writes the next. `source_span_pages` links a span to
the page reads it draws on, and `versions.source_span_id` names a child's span.
"""

from __future__ import annotations

from alembic import op

revision: str = "0023_source_spans"
down_revision: str | None = "0022_label_provisions_kind"
branch_labels: str | None = None
depends_on: str | None = None


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
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS source_span_pages (
            span_id uuid NOT NULL REFERENCES source_spans(id) ON DELETE CASCADE,
            page_read_id uuid NOT NULL REFERENCES page_reads(id) ON DELETE CASCADE,
            PRIMARY KEY (span_id, page_read_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS source_span_pages_read_idx ON source_span_pages (page_read_id)"
    )
    op.execute(
        "ALTER TABLE versions ADD COLUMN IF NOT EXISTS source_span_id uuid "
        "REFERENCES source_spans(id) ON DELETE SET NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS versions_source_span_idx "
        "ON versions (source_span_id) WHERE source_span_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS versions_source_span_idx")
    op.execute("ALTER TABLE versions DROP COLUMN IF EXISTS source_span_id")
    op.execute("DROP TABLE IF EXISTS source_span_pages")
    op.execute("DROP TABLE IF EXISTS source_spans")
