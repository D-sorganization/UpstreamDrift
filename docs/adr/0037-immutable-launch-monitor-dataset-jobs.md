# ADR 0037: Immutable Launch-Monitor Dataset Jobs

- Status: Accepted
- Date: 2026-08-20
- Decision Makers: UpstreamDrift Maintainers
- Related Issues: [#8806](https://github.com/D-sorganization/UpstreamDrift/issues/8806), [Tools #4226](https://github.com/D-sorganization/Tools/issues/4226)

## Context

The inline analysis API deliberately caps requests at 20,000 records. The
private launch-monitor authority currently contains 261,666 rows, so copying
the corpus into an HTTP request would defeat that boundary, increase memory and
network exposure, and discard the authority's immutable repository, manifest,
source-file, and qualification lineage.

A client-supplied filesystem path is not an acceptable replacement. It would
permit path traversal, disclose server layout, and make authorization depend on
untrusted request text. An exact Git commit alone is also insufficient because
a checkout can contain modified data files.

## Decision

Add contract `launch-monitor-dataset-job/1.0.0` under the existing Launch
Monitor Analytics router. The server administrator maps opaque aliases to
absolute authorized checkout roots. Requests name only an alias plus the exact
`owner/repository`, 40-character commit, corpus-manifest SHA-256, deterministic
Parquet-content SHA-256, and expected row count. Dataset and manifest paths are
fixed by the service; the request cannot supply a path, URL, SQL expression, or
inline records.

Before an operation reads observations, the service verifies:

1. the alias is server-authorized and resolves without symlink traversal;
2. the checkout's `origin` and `HEAD` exactly match the request;
3. the fixed corpus manifest bytes and sorted Parquet path-plus-byte digest
   match the request;
4. manifest source totals and the physical Parquet row count both match the
   expected row count; and
5. the committed qualification manifest binds the corpus manifest, acquisition
   manifest, and source-summary hashes used for backing joins.

The initial allowlist contains source summary, metric summary, and Pearson
correlation. It contains no arbitrary query language. Numeric outputs are
aggregates only, groups with fewer than ten complete rows are suppressed,
results are limited to 5,000 items, and pages are limited to 200 items. Source
summaries expose content-addressed source provenance and counts but never shot
records, source filenames, paths, or raw repository URLs. Sources below the
ten-row disclosure floor are suppressed. Jobs use a process-local registry
with at most 64 retained entries and two worker threads by default. The service
has an idempotent close/context
manager contract, and the router lifespan joins workers during application
shutdown. Capacity exhaustion returns a retryable structured HTTP 429 with a
`Retry-After` header rather than falling through as an internal error.

Unavailable authorities, dependencies, hash mismatches, and unsupported
operations produce structured, data-free states. Unexpected failures do not
return exception text or server paths. Existing inline v1/v2 and source-backed
strokes-gained contracts remain unchanged.

The production route registry classifies this router as protected and injects
the global bearer-token authentication dependency. Authentication may be
disabled only at the existing explicit local-development boundary; the local
server is not a production deployment. Root aliases are authorization policy
for data locations, not a replacement for caller authentication.

## Alternatives Considered

1. Raise the inline limit above 261,666: rejected because it moves restricted
   rows through the public API surface and duplicates the private authority.
2. Accept an absolute path in each request: rejected because authorization and
   containment must be server policy, not client input.
3. Trust only `git rev-parse HEAD`: rejected because dirty or replaced working
   files can differ from that commit.
4. Return paged shot rows: rejected because issue #8806 requires analysis by
   reference, not a private-data export channel.
5. Add arbitrary SQL: rejected because bounded operations and output privacy
   cannot be enforced safely over caller-authored expressions.

## Consequences

- Positive: the full authority can be analyzed without inline transfer.
- Positive: every job is bound to repository, commit, manifest, content, and
  row-count identity before use.
- Positive: source/backing joins retain traceability without exposing private
  observation values.
- Positive: clients can distinguish unavailable data from failed execution.
- Negative: the current job registry is process-local and results disappear on
  restart; clients can safely resubmit the immutable reference.
- Negative: hashing all Parquet bytes adds bounded startup work to each job.
- Negative: the operation allowlist must grow intentionally as new aggregate
  analyses are qualified.

## Validation

- A temporary synthetic 261,666-row Parquet authority exercises the complete
  verification and correlation path without committing private rows.
- Tests alter commit, manifest hash, content hash, and row count independently
  and require structured fail-closed states.
- API tests reject inline records and client paths, assert OpenAPI response
  models, verify capacity exhaustion, exercise lifespan cleanup, and verify the
  checked-in JSON Schema matches the Python authority.
- Security, architecture, Ruff, format, mypy, schema, and focused/broad tests
  are required before publication.
