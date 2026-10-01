# Test scopes

Install development dependencies with `uv sync --group dev --extra migrations`.
The default GitHub test job runs:

```sh
uv run pytest tests -m "not integration and not live_llm" -q
```

This checks deterministic code without running the database or model-provider
suites. Tests that need jurisdiction configurations not shipped here skip
explicitly; a pass does not validate those configurations.

Database tests need a disposable PostgreSQL instance with pgvector and
pg_textsearch. Some tests commit or recreate data, so never point them at a
shared or production database. The `integration` CI job runs them.

1. `docker compose up -d --wait postgres` builds and starts one. `POSTGRES_URL`
   defaults to that container, `postgresql://codify:codify@localhost:5432/codify`.
   `CODIFY_PG_PORT` moves the host port; if you set it, set `POSTGRES_URL` to
   the same port.
2. `uv run alembic -c alembic.ini upgrade head` applies the migrations.
3. `REQUIRE_DB=1 uv run pytest tests -m "integration and not live_llm"` runs
   the integration tests; `REQUIRE_DB` makes an unreachable database a failure
   rather than a skip.

`live_llm` tests are separate and require configured providers; they can incur
charges. Neither offline tests nor schema checks certify the accuracy of a
real legal corpus.
