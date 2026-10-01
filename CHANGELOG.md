# Changelog

Dates are cut dates. A version is released when its tag exists; the first tag is
the launch tag.

## Unreleased

Breaks, in that the structurer's output changes for documents it already read:

1. A declared closing phrase also matches where the source breaks the line
   between two of its words (one line break, any spaces or tabs), so the
   attestation after it leaves the body. Conclusion lifting, signatory grouping
   and the page-region signal read phrases the same way. A blank phrase no
   longer matches.
2. Losses are reported instead of reading as success, so a run that finished
   clean before can now grade `warning` or `blocking`, or fail:
   - A page the model refused on a content filter, or an inked page read empty,
     is recorded: `PageResult.finish_reason` and `PageExtracted.finish_reason`,
     an `unreadable_page` error finding, `pages_unreadable` in the bundle
     manifest, and an editorial remark in the AKN where the page stood. The
     structurer now receives the unreadable-page marker and places the remark
     itself; `combine_text_for_structure` no longer strips it.
   - Body-fill records what it achieved (`ScanTrace.body_fill`). Bodies copied
     from the source after the model failed them are a `body_fill_verbatim`
     warning; bodies left empty a `body_fill_incomplete` error. When the model
     wrote no body, whether its calls raised or came back empty, the structurer
     raises `BodyFillError`, or under `halt_policy="land"` records a
     `body_fill_failed` halt. The scan trace is emitted once, after body-fill.
   - `coverage.json` `ratio` is the lowest of the anchor, body-fill and page
     ratios; the anchor figure moves to `anchor_ratio`. The manifest carries the
     structural grade.
   - A jurisdiction config that is absent or does not validate raises where it
     is read. Anchor-scan helpers, title helpers, the region vocabulary, the
     enrich passes and the repair dossier used to catch it and return defaults.
   - `validate_akn(provenance="extracted")`, which the PDF and text lanes pass,
     grades every numbering gap `warning` and adds a leading-gap `number_gap`
     when the first article or section is numbered above 1. A gap beside an
     unreadable page is no longer called a repeal.
3. `validate_akn(provenance="native")`, which the lanes reading publisher XML
   (native AKN, FORMEX and the other publisher formats) pass, skips
   `identity_year_implausible` and `identity_year_unconverted_hijri`: a
   publisher's own date is not a misparse.
   Callers that do not pass `provenance` see the previous checks.

New, additive:

- `ingest-one --fallback-model`, defaulting to
  `LITELLM_CONTENT_FILTER_FALLBACK_MODEL`: the model retried on any
  content-filter refusal, so page reads, metadata and body-fill can all run on
  it.
- `validate_akn` takes `provenance`, `unreadable_pages` and `body_fill`;
  `codify.jurisdictions.CONFIG_FAULTS`; `codify.core.llm.content_filtered`.
- `document_classes.<class>.number_source` in the jurisdiction config:
  `"stated"` (the default, unchanged behaviour) or `"title_identity"`, which
  numbers the class's work URI from its title digest (`t-...`) and ignores any
  extracted number, for classes whose serials repeat across issuers and so
  collide. Declaring it requires `frbr.title_identity`.

## 0.5.0 (2026-09-20)

One structuring change, so the minor moves as `VERSIONING.md` prescribes while the
major is zero; the rest is additive.

Breaks, in that the structurer's output changes for documents it already read:

1. The anchor scan reads marginal-note layouts: where a jurisdiction declares
   `display.heading_type: marginal_note`, a bare basic-unit number under a
   heading line (the shape a scanned gazette transcribes to) is a section, with
   that line as its heading. The coverage denominator counts the same markers, so
   a scan that lost them reads below 1.0 instead of measuring nothing. The
   bundled `xa` scan now structures its twenty sections rather than six parts.

New:

- A web app over the server, `apps/web`: laws, reader, search, jurisdictions and
  ingest with a live run stream; types generated from `contract/openapi.json`.
- Two reader routes on the server, `/versions/{version_id}/document` and
  `/laws/{law_id}/versions`; the run stream carries a `stored` event with the new
  version and law ids; search matches carry law and version context.
- Notebooks under `docs/notebooks/`: Codify 101, 201, 301 and 301a.

Repository: the publish action reads current metadata; Dependabot version
updates with a release cooldown; pinned actions advanced; pypdf 6.18.1.

## 0.4.0 (2026-09-18)

The first published release. Everything after the bundle: a store, a server, a
contract and an MCP surface, over the library that was already there. No
structuring behaviour changes. Under `VERSIONING.md` the minor moves while the
major is zero.

New commands:

- `codify load bundle/ --embed` writes a bundle, or a bare AKN file, into
  Postgres and embeds its provisions; `codify search` runs the hybrid retrieval
  over them; `codify compare` aligns two documents or two stored versions.
- `codify serve` is a thin HTTP server over the same library calls: ten read and
  run routes with in-memory runs and an SSE stream, `--openapi` prints the
  schema, and `contract/openapi.json` is the committed copy a drift test guards.
- `codify mcp` exposes seven read tools to an MCP client over stdio or
  streamable HTTP, behind the `mcp` extra.

Library, additive:

- `storage.get_version` and `list_versions` take `with_akn=False` for a
  metadata read that leaves the body unloaded; `get_version_akn_length`.
- Every JSON route answers with a Pydantic model, `codify.serve.schemas`; the run
  stream is server-sent events.
- The EU fetcher resolves every hop, redirects included, through the address
  guard.

Repository: security policy, code of conduct, contributor guide, issue and pull
request templates, and the boundary decision naming structure-preserving
translation as core.

## 0.3.0 (2026-09-18)

Body grammar the jurisdiction config declares, and three behaviour changes that
apply without a declaration. Under `VERSIONING.md` the minor moves while the
major is zero.

Breaks, in that the structurer's output changes for documents it already read:

1. A doctype whose numbered provision is the `section` element now refills a
   section the model left empty, in narrower windows and then verbatim, as it
   always did for articles. 0.2.0 excluded `section` from that recovery, so a
   dropped section shipped empty.
2. A quotation no longer runs over a provision heading: a span nothing closes
   ends at the blank line before the next marker, and a marker whose number
   wraps onto the next line bounds a reversed quotation. 0.2.0 let a stray
   curly quote in a definitions block mask the provisions after it.
3. A slashed insertion after a keyword marker (`7/1`) is read as its own
   number, keyed as the parser's `sec_7-1`; 0.2.0 read it as `7` and dropped
   one of the pair as a twin. Keyword-less marker forms (`7.`) are unchanged.
4. Where a jurisdiction declares `closing_phrases`, the scan ends the body at the
   first line opening with one: markers between it and the first attachment
   caption are dropped and counted (`tail_excluded`), the span is emitted verbatim
   as `CONCLUSIONS`, and an attachment caption before it titles a body table
   rather than opening an attachment. 0.2.0 fed the span to the body-fill and
   lifted the attestation out of the last provision afterwards.

New config, additive:

- `structuring.insertion_suffixes`: words that follow a number to mark an
  inserted unit, each mapped to the ASCII form its eId carries (`5 bis`). The
  scan reads the suffix, on the same line or the next, and no longer files the
  unit as a twin of its base.
- `structuring.citation_successors`: words that, following a marker's number on
  its own line, make the line a citation run rather than a provision.
- `structuring.prose_precursors` in a script that runs words together (Thai,
  Lao, Khmer, Myanmar blocks) may follow a letter directly.
- `attachments[].prefix`: a caption that opens a longer title on the same line.

## 0.2.0 (2026-09-17)

Thirteen breaks, so the minor moves, as `VERSIONING.md` prescribes while the
major is zero. Items 1 to 11 are in year resolution, enumerated by running
every year-resolution grid cell and edge row through the 0.1.0 and 0.2.0
callers (`resolve_year`, `gregorian_year`, `resolve_descriptors`) and grouping
every differing cell; each of those lines names the input shape that triggers
it, and a stored row moves under them only where a jurisdiction with a
non-Gregorian calendar has a law stored, a stored law has a year below 1000,
or a work URI carries a one- to three-digit year segment. Items 12 and 13 are
storage changes with no input shape: migration 0020 copies every existing
`provision_embeddings` row into the partitioned table.

1. A date field the model left unlabelled is read as written; 0.1.0 converted
   its year through the jurisdiction's calendar, filing a Gregorian `1968-09-09`
   under 1425 in a Buddhist-Era jurisdiction. Converted now only where a
   declared title grammar shows the year is local.
2. `to_gregorian_year(..., month=)` read a Gregorian month of a Bikram Samvat
   year the wrong way round, filing April to December under the later year.
   Corrected against the calendar (1 Baisakh 2080 was 14 April 2023); the new
   `month_grid` keyword names the grid a month is on, defaulting to Gregorian.
3. A three-digit year is carried as a four-digit segment and stored as its
   number: `622` files under `/0622/` where 0.1.0 minted a URI
   `is_citable_work_uri` refused, and a converted year below 1000 is carried
   the same way instead of dropped.
4. The sentinels `0000` and `0001` in a year field are no year; 0.1.0 returned
   them, and `0000` crashed the document-class match.
5. A five-digit year run (`12024`, in a year or date field) is no year; 0.1.0
   converted it to `11481` and crashed the class match.
6. A date field not shaped as a date (`2024-01-01junk`, a three-digit day,
   `unknown`) states nothing: no year run and no date. 0.1.0 split it on a
   dash and converted what came first, or passed it through as the work date.
   A date-shaped value with a day the month does not hold (`2024-02-31`,
   `2511-04-31`) keeps its year run and emits no date; 0.1.0 emitted it.
7. A year field in native digits (`๒๕๑๑`) is returned in ASCII by
   `resolve_year`; 0.1.0 returned the native digits while storing `2511`.
8. A date labelled with a calendar the jurisdiction does not declare is read
   as written; 0.1.0 retried its year through the jurisdiction's own rule.
9. A work date is emitted only where its year is the URI year; 0.1.0 emitted
   `2511-09-09` beside a URI year of 2020 and `unknown` beside any year.
10. The stored year is the URI year wherever the URI has one; 0.1.0 stored
    `None` where the date field held no date while the URI took the title's
    year, and stored a raw local year where the URI took the converted one.
11. A labelled year converts by the jurisdiction's declared rule; 0.1.0 used
    the generic table for that calendar and ignored a declared epoch.
12. `provision_embeddings` is partitioned by `jurisdiction_id` and carries
    `version_id` (migration 0020): the primary key is `(id, jurisdiction_id)`,
    the unique key `(provision_id, model_id, jurisdiction_id)`, and each
    partition holds its own HNSW index, built from
    `docs/runbooks/vector-index-build.md` on a populated database. A scoped
    search reads one partition's index instead of scanning every embedding in
    scope; the dense arm's `hnsw.iterative_scan` is `relaxed_order`.
13. `upsert_embedding` reads the provision's version and jurisdiction itself.
    No partition is created at run time: a jurisdiction created after 0020
    writes to the DEFAULT partition, which has its own index, until a
    migration promotes it with `codify.storage.partitions.promote_sql`.

Three more groups of differing cells are reachable only through config fields
0.1.0 could not load, so they are capability rather than breaks: a dated line
the source states is read where a date grammar is declared; a labelled local
date converts with its month and the new-year reform where a Gregorian grid
and a reform year are declared; and such a date's month and day carry into the
work date.

## 0.1.0 (2026-09-09)

First version of the standalone library, before any published release or benchmark run.

- Standalone Python library and CLI with selected public-reference and synthetic
  jurisdiction configurations. Missing configurations raise
  `JurisdictionDataMissing`; `try_load_config` supports optional lookup.
- Acquisition, AKN structuring and validation, retrieval, comparison, translation
  and optional PostgreSQL storage are included.
- Commercial lens implementations and catalogues are excluded. Generic
  scheme-match reads accept an optional `remediation_templates` mapping; callers
  requiring enrichment must provide it explicitly.
- Test fixtures and documentation have been adapted for a standalone checkout.
  Database and provider tests remain separate from the default offline CI.

Benchmark numbers from before the split are not results for this version. The
synthetic retrieval evaluator can be run separately against a disposable
PostgreSQL database; it does not measure accuracy on a real statute book.
