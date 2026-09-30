# Working with documents

[Back to the README](../README.md)

Commands below run from a source checkout after the [quick start](../README.md#quick-start).
Model-assisted operations require a configured endpoint and may incur charges.

## Inspect an ingestion

The generated `bundle/` directory contains:

| File                | Contents                                                      |
| ------------------- | ------------------------------------------------------------- |
| `pages/`            | Page images rendered for inspection after the run             |
| `source.txt`        | Extracted text transcription                                  |
| `anchors.jsonl`     | Structural markers detected during the scanning pass          |
| `ambiguity.jsonl`   | Ambiguous structural spans found during scanning              |
| `coverage.json`     | Anchor, body-fill and page coverage; `ratio` is the lowest    |
| `scaffold.bluebell` | Structural skeleton generated prior to filling provision text |
| `final.akn.xml`     | Generated Akoma Ntoso 3.0 document                            |
| `validator.json`    | Structural validation findings                                |
| `manifest.json`     | Run metadata, unreadable pages and the structural grade       |
| `events.jsonl`      | Pipeline event history, including progress and failures       |

Jurisdiction rules identify the document's structure. The model fills the text
within the detected sections and articles.

**Inspection order.** Inspect `anchors.jsonl` before `final.akn.xml`. The model only
populates text inside anchored basic units (e.g., sections, articles). If a pass
identifies higher-level containers but no basic units, it logs `body_fill_skipped` and
exports only the skeleton. If no anchors are found, it logs `scaffold_no_anchors` and
preserves the source text unparsed.

**Dry runs.** To inspect detected anchors across `.txt` and text-layer `.pdf` files
without making model calls or requiring a database, run the anchor scan directly:

```bash
uv run codify scan-corpus ./documents --jurisdiction xa
```

The summary is JSON on stdout; diagnostics such as an empty sweep go to stderr, and
`--per-document` writes one row per scanned document to a path of your choosing.

**Log messages.**

- `layout_pass_failed ... OcrNotConfigured`: the optional secondary OCR engine (Azure AI
  Foundry) is not configured. The pipeline reads the scan once via the model's vision
  endpoint rather than twice, and the run otherwise proceeds.
- `page_diverted_to_ocr`: the page's extracted text layer was rejected (the event's
  `reason` names the test it failed: too short, garbled, letter-spaced, presentation
  forms, divergent from the scan, or a visible annotation) and the page was rasterised
  for OCR.

## Store, search, compare

To search a bundle, load it into PostgreSQL and embed its provisions. Docker Compose
builds a local database with pgvector and pg_textsearch. Use a dedicated local
database: the commands below apply migrations and write records.

Before using `--embed` or `search`, configure an OpenAI-compatible embeddings endpoint. `EMBEDDING_BASE_URL`,
`EMBEDDING_API_KEY` and `EMBEDDING_MODEL` select it. The URL and key fall back to
`LITELLM_BASE_URL` and `LITELLM_API_KEY`; the default embedding model is
`gemini-embedding-2`, not the chat model. Choose a model your endpoint supports.

```bash
uv sync --extra migrations
docker compose up -d --wait postgres
uv run --extra migrations alembic -c alembic.ini upgrade head
uv run codify load bundle/ --embed
uv run codify search "right of access" --jurisdiction xa
```

`load` reads `final.akn.xml` and `manifest.json` from a bundle and writes the law, its
version and its provisions; the jurisdiction and the title come from the manifest, so a
bundle needs no flags. A bare `.akn.xml` has no manifest and needs `--jurisdiction` and
`--title`. `--embed` also embeds the provisions, which `search` needs. Loading the same
file twice returns the version it already holds.
`search` runs the hybrid retrieval (lexical and vector, fused) and prints one JSON match
per line, best first, with the provision's eId, score and text; it exits 1 when nothing
matched.

```bash
uv run codify compare a.akn.xml b.akn.xml --out report.json
uv run codify compare "$REFERENCE_VERSION_ID" "$DOMESTIC_VERSION_ID"
```

`compare` aligns the second document's provisions against the first, article by
article, through the chat model, and writes a report with a summary (aligned, partial,
gap) and every alignment. Either side is an AKN file or a stored version id.
Set the version-ID variables above to your stored UUIDs. Comparison needs both
chat and embeddings endpoints; cost depends on the documents and model.

`POSTGRES_URL` names the database. Unset, it is the compose one while `ENVIRONMENT` is
unset, `localhost` or `development`; under any other value an unset URL is an error.

## Configuration

The CLI loads environment variables from a `.env` file in the working directory or parent
directories; explicitly exported shell variables take precedence. Note that `pytest` and
`alembic` do not read `.env` files automatically and require variables to be exported in
your environment. See [`.env.example`](../.env.example) for the default setup.

| Variable                                | Purpose                                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `LITELLM_BASE_URL`                      | Base URL for the OpenAI-compatible chat endpoint                                        |
| `LITELLM_API_KEY`                       | API key for the chat endpoint                                                           |
| `LITELLM_MODEL`                         | Default model used for body fill (overridden by `--model`)                              |
| `LITELLM_CONTENT_FILTER_FALLBACK_MODEL` | Model retried when the first refuses a page on a content filter (`--fallback-model`)    |
| `POSTGRES_URL`                          | Postgres connection string for migrations and tests (defaults to local Compose service) |
| `AZURE_OPENAI_ENDPOINT`                 | Endpoint for the optional secondary Azure AI Foundry OCR engine                         |
| `AZURE_OPENAI_API_KEY`                  | API key for the optional Azure OCR engine                                               |
| `LANGFUSE_PUBLIC_KEY`                   | Public key for optional Langfuse tracing                                                |
| `LANGFUSE_SECRET_KEY`                   | Secret key for optional Langfuse tracing                                                |

## Jurisdictions

Configurations are included in distributed package wheels if their `config.json` sets
either `synthetic` or `public_reference` to true. This filtering is enforced by
`codify/open_wheel.py` at build time to prevent the bundled set from drifting out of sync.

- **Synthetic jurisdictions**: contain mock legislation formatted to real-world
  legislative structures, allowing test suites to assert against known-good parses
  without distributing copyrighted corpora.
- **Public reference configurations**: cover jurisdictions that publish their legal texts
  openly.

A source checkout reads the repository's `data/` directory, wherever the command is
run from; an installed wheel reads the data bundled inside the package. To supply a custom dataset, set
`CODIFY_DATA_ROOT` to an absolute path containing `jurisdictions/` and `frameworks/`
directories before starting Python. Relative paths are rejected, and custom data roots
completely replace bundled data.

To add a jurisdiction, see [adding a jurisdiction](jurisdictions/adding-a-jurisdiction.md).
