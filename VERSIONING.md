# Versioning

What this package promises as a library, a command and a schema, and what breaks
each promise. It publishes as `codify-core` and imports as `codify`; the version
below is the distribution's.

## The public surface

- **Modules.** Everything under `codify.` not starting with an underscore. Which
  of them downstream applications depend on is
  `codify/contract/product_surface.txt`, checked by a test rather than counted
  here by hand. One underscore module, `codify.akn._schema`, is imported across
  it too: debt to repay by promoting or replacing it, not surface. Breaks:
  removing or renaming a symbol, changing a parameter's name, order or meaning,
  inserting one ahead of an existing parameter, widening a return type.
- **The command.** `codify`, and the output its subcommands write:
  `ingest-one`, `scan-corpus`, `load`, `search`, `compare`, `serve`, `mcp`,
  `index-datadump`, `index-ee-archive` and `index-legislation-gov-uk`.
  Breaks: removing a subcommand or a flag, changing a flag's meaning or the
  shape of the output.
- **Jurisdiction config.** `JurisdictionConfig`, `ResolvedConfig` and the
  `config.json` files the package carries. Breaks: removing a field, making an
  optional field required, reinterpreting a field, removing a shipped
  jurisdiction.
- **Events.** `ProgressEvent` and the domain events written through
  `codify.storage`. Breaks: removing a field, renaming an event, changing when
  one fires.
- **Database schema.** The core migration chain, `alembic_version_core`, and
  every object it owns: tables, columns, defaults, indexes, triggers and
  functions. Breaks: dropping or renaming any of them, narrowing a type,
  dropping a default, or adding a constraint, which can reject writes that were
  allowed even when every existing row passes.

Anything purely additive is not a break; a parameter added at the end or made
keyword-only counts only if it is optional.

## Version numbers

Major for any break above, including output that changes while signatures do
not. Minor for new capability or data. Patch for the rest.

## 0.x

The current version is `0.6.0`. While the major is zero the minor acts as the
major: pin `>=0.6,<0.7`. The step from `0.1.0` to `0.2.0` carried thirteen
breaks, the step to `0.3.0` four, the step to `0.5.0` one and the step to
`0.6.0` eight, each enumerated in the changelog; `0.4.0` added commands and
carried no break.
