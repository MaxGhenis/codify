# Test markers

Two markers gate tests that need external services. Combine with pytest's `-m`
expression syntax to opt in or out.

| Marker        | Requires                                       | CI behaviour                                                                 |
| ------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `integration` | Postgres (migrated to head; see `docs/offline-suite.md`) | run by the `integration` CI job |
| `live_llm`    | A LiteLLM gateway (`LITELLM_BASE_URL`, exported) | never in CI; can incur charges                                             |

Common selectors, from the repository root:

```bash
# What CI runs (pure unit; no DB, no LLM):
uv run pytest tests -m "not integration and not live_llm"

# With the compose Postgres up and migrated; no live LLM:
REQUIRE_DB=1 uv run pytest tests -m "not live_llm"

# Live LLM. pytest reads the environment, not .env; the test clients speak the
# gateway's dialect, so a provider's own endpoint is not enough here:
LITELLM_BASE_URL=http://localhost:4000/v1 LITELLM_API_KEY=... uv run pytest tests -m "live_llm"
```

`codify.testing.postgres_url()` and `codify.testing.ollama_host()` are the
shared helpers for fixtures that talk to those services. Use them; don't
re-roll the env-var lookup per file.
