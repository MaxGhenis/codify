# Changelog

Dates are cut dates. A version is released when its tag exists; the first tag is
the launch tag.

## Unreleased

New, additive:

- Migration `0023_source_spans` and `codify.storage.spans`: a multi-act
  source's split is stored as spans over its pages, one generation at a time.
  Each span is an act, a held region, a skipped non-act or front matter; its
  cuts are a page, an offset into that page's text and a marker line.
  `write_span_generation` retires the live generation and writes the next,
  deleting nothing. `versions.source_span_id` names a child version's span, and
  `get_page_reads`, `get_page_read` and `count_disputes_by_page_read` read a
  child through the page reads its span links (`source_span_pages`), with the
  source's own page numbers. `codify.pipeline.span_cuts` turns a `Segmentation`
  into cuts, rebuilds a span's text from page texts, and finds a cut again after
  a re-read by its marker (`None` when the page no longer carries it).
  `locate_generation` finds a whole split again, each end where the next
  region now starts, or `None` when a start is lost or the starts run
  backwards. A child's repair evidence is trimmed to its span. A page read a span links cannot be deleted (`RESTRICT`):
  a host's cleanup must spare linked reads. The `versions` column is added under
  a 5-second lock timeout, its foreign key validated and its index built without
  blocking writes.
  `segmentation.skip_heading_patterns` names instruments that are not acts
  (notices, appointments): their segments are cut as skipped, kept and listed.
- `codify.pipeline.segment`: splits a source holding several acts, such as a
  gazette issue, without a model call. `segment` returns a `Segmentation`:
  `single` (the text unchanged), `decided` (one segment per act) or
  `abstained` (what the evidence could not settle is held whole, with a
  reconciliation table in plain words). A boundary is a declared act heading
  that a closing phrase, a numbering restart, a contents entry on its page or
  a page start agrees with. An adopted text, an attachment, a repeat of the
  open act's heading, a quotation or a missing enacting formula vetoes one. A
  heading no signal agrees with is read as a citation where it stands, named
  in the reconciliation table, and neither splits nor holds. `segment_volume`
  reads a bound volume a page at a time, cuts it into issues by the same rule
  and segments each issue. Configured by the jurisdiction's `segmentation`
  block: `act_heading_patterns`, `issue_heading_patterns`, `contents_keywords`
  and `printed_page_pattern`, each pattern refused at load if it does not
  compile or can match empty text.

Fixed:

- `segment`: where several headings name one listed act, the contents entry
  goes to the one on its listed page, so a prose mention elsewhere no longer
  makes the source abstain.

## 0.6.0 (2026-10-02)

Eight breaks, so the minor moves, as `VERSIONING.md` prescribes while the major
is zero; the rest is additive.

Breaks, in that the structurer's output changes for documents it already read:

1. A declared closing phrase also matches where the source breaks the line
   between two of its words (one line break, any spaces or tabs), so the
   attestation after it leaves the body. Conclusion lifting, signatory grouping
   and the page-region signal read phrases the same way. A blank phrase no
   longer matches.
2. Losses are reported instead of reading as success, so a run that finished
   clean before can now grade `warning` or `blocking`, or fail:
   - A page the model refused on a content filter (partial text included), or
     an inked page read empty, is recorded: `PageResult.finish_reason`,
     `PageExtracted.finish_reason` and `PageExtracted.unreadable`, an
     `unreadable_page` error finding, `pages_unreadable` in the bundle
     manifest, and an editorial remark in the AKN where the page stood. The
     structurer now receives the unreadable-page marker and places the remark
     itself; `combine_text_for_structure` no longer strips it.
   - Body-fill records what it achieved (`ScanTrace.body_fill`). Bodies copied
     from the source after the model failed them, or restored because the model
     changed a table, are a `body_fill_verbatim` warning; bodies left empty a
     `body_fill_incomplete` error. When every call failed and the model wrote
     no body, the structurer raises `BodyFillError`, or under
     `halt_policy="land"` records a `body_fill_failed` halt. The scan trace is emitted once, after body-fill.
   - `coverage.json` `ratio` is the lowest of the anchor, body-fill and page
     ratios; the anchor figure moves to `anchor_ratio`. The manifest carries the
     structural grade.
   - A jurisdiction config that is absent or does not validate raises where it
     is read. Anchor-scan helpers, title helpers, the region vocabulary, the
     enrich passes and the repair dossier used to catch it and return defaults.
     A config file that cannot be read or decoded raises
     `JurisdictionConfigError`. The PDF and text lanes read the config first and
     emit `Failed(stage="config")` before any extraction or model call.
   - `validate_akn(provenance="extracted")`, which the PDF and text lanes pass,
     grades every numbering gap `warning` and adds a leading-gap `number_gap`
     when the first article or section is numbered above 1. A gap after a
     provision carrying an unreadable-page remark is no longer called a repeal.
     Numbering checks now cover `rule` units as well as articles and sections.
3. `validate_akn(provenance="native")`, which the lanes reading publisher XML
   (native AKN, FORMEX and the other publisher formats) pass, skips
   `identity_year_implausible` and `identity_year_unconverted_hijri`: a
   publisher's own date is not a misparse.
   Callers that do not pass `provenance` see the previous checks.
4. A quotation that opens on prose and that nothing closes now ends before the
   next keyword-led container or basic-unit heading, blank line or not. A
   dropped closer on a quoted name in a preamble no longer masks the first
   chapter and article below it. A quotation opening on a heading still masks
   to the blank line, since an amendment may quote several provisions.
5. A content-filter finish reason that carries a suffix
   (`content_filter: RECITATION`) counts as a block on page reads and schema
   calls, so the fallback model is tried. Such a page used to read as empty and
   such a call to fail validation.
6. Series-number citations ("Act No. 5") are wrapped only where the
   jurisdiction config declares them in `numbering.series_citations`; nothing
   is keyed on a country code. A config that relied on the built-in list must
   declare its series, and `Blg.` is no longer a default connector.
7. The amount checks recognise dirhams, riyals, liras and francs, with their
   Arabic and Hebrew forms, beside the currencies they read before, so the
   amount-in-words warning and the numeric extractor report amounts they
   skipped.
8. The six real jurisdiction configs 0.5.0 shipped are revised: compound
   heading terms split into a term and its aliases, enacting formulae scoped by
   document class, diacritics restored, URI patterns kept only for declared
   classes, and keys no model declares removed. A heading written in an alias
   now anchors, so output for those jurisdictions can change.

New, additive:

- Configs for 96 more jurisdictions across every region and legal tradition,
  taking the real jurisdictions shipped from 6 to 102. The registry is
  regenerated, and a test checks it against the configs on disk.

- `segmentation.act_heading_patterns` in the jurisdiction config: line-start
  regexes for a line opening an act. Where a jurisdiction declares them and
  its closing phrases, a duplicate-number fold whose dropped run holds a
  closing phrase and then an act heading raises an `act_boundary_suspected`
  span, which is blocking: the structurer refuses, or under
  `halt_policy="land"` records an `act_boundary_suspected` halt. The source
  reads as two acts numbered from 1, and keep-last kept only the later one.
  The fold itself is unchanged.
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
- `numbering.series_citations` and `numbering.series_citation_connectors` in
  the jurisdiction config (break 6).
- A same-document anchor with no stored row of its own resolves to the nearest
  stored row: the element's own text row, its first stored part, or the row
  enclosing it, never a part of quoted text or a placeholder. `ResolveStats`
  counts `resolved_container` and `resolved_enclosing`; `nearest_stored_row`
  and `element_index` are public. Unresolved rows carry no resolver version, so
  the next pass re-examines them.
- The EUR-Lex HTML lane reads older acts whose markup leaves the opening
  paragraph unclosed; they used to fail with no article blocks.
- Migration 0021 adds `static_site_export` to the run kinds; the downgrade
  refuses while rows of that kind exist.

Repository: tests and examples that used jurisdictions the repository does not
ship run on the fictional ones; CI runs the integration suite against Postgres;
a shorter README with usage and interface guides; simplified notebooks; a
corpus ownership design; Dependabot skips TypeScript and `@types/node` major
bumps. Dependencies: pydantic-ai 2, with the repair agent keeping its early
end; openai 3; google-genai 2.

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
