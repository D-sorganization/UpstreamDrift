# ADR 0034: Launch Monitor Analysis Contract V2

- Status: Accepted
- Date: 2026-08-19
- Decision Makers: UpstreamDrift Maintainers
- Supersedes: The v1 wire result as the preferred cross-repository contract
- Retains: Contract v1 as a compatibility surface

## Context

The canonical shot schema in ADR 0031 makes imported values comparable and
traceable, but the v1 flexible-analysis response carries only selected units, a
dataset fingerprint, and numerical diagnostics. A consumer cannot determine
the source authority, backing rows, transformations, identity trust, exclusions,
or the exact scope of a vendor/model claim from that response alone. Tools,
UpstreamDrift clients, and private campaign scripts therefore risk inventing
different metadata wrappers around the same statistics.

## Decision

UpstreamDrift owns the canonical Python and HTTP analysis contract. Contract
`2.0.0` is a strict Pydantic model and a generated JSON Schema. FastAPI uses the
same response model in OpenAPI. The envelope retains a complete v1 numerical
payload when analysis is available and adds:

1. canonical and display units;
2. commit-addressed dataset authority, content-addressed source references,
   versioned transformations, exact input-record hashes, source row IDs, and a
   declared source join or explicit unlinked reason for every backing record;
3. per-variable missing/non-numeric counts and per-analysis exclusions;
4. structured result availability instead of ambiguous null values;
5. uncertainty and multiplicity methods with assumptions;
6. explicitly declared player-identity trust and evidence;
7. vendor/model/software/measurement-status provenance and optional analytical
   model provenance; and
8. claim flags that cannot imply device emulation, certification, or causality
   by default.

The older `/analyze` route and `FlexibleAnalysisResult.to_dict()` remain contract
`1.0.0`. New consumers use `/v2/analyze`. A v2 result can be adapted to v1 only
when it contains an available embedded numerical result.

Player grouping fails closed unless the caller supplies a trusted identity
level, the exact identifier column, and evidence. Session, club, row order,
filename, and file layout are never identity evidence. Insufficient complete
rows and rank-deficient OLS designs are explicit unavailable results. Invalid
requests and unsafe aggregation remain errors.

ADR 0036 strengthens this boundary: `PlayerIdentityV2` rejects forbidden
pseudo-identifiers during model validation even when a user attests them.
Session boundaries and observation order use separate evidence contracts and
cannot upgrade a field into player identity.

Commit identifiers are full 40-character lowercase hexadecimal SHAs. The safe
explicit identity level is `explicit_user_attested`; a session label is not an
identity level. All canonical metrics and retained numeric source fields remain
selectable, but only registry metrics receive registry-authoritative units.
Non-canonical units must be declared by the source context and are labeled
`source_declared`; absent declarations remain `unknown` rather than receiving a
silently authoritative unit.

## Consequences

- Python, API, desktop, React, and campaign clients can validate the same
  machine-readable result.
- Restricted values need not be copied into an exported result: exact record
  hashes and content-addressed source references retain the audit trail.
- Consumers must intentionally declare identity and source trust rather than
  receiving inferred metadata.
- The v2 envelope is larger because exact backing-record references are retained.
- Breaking changes require a new major contract version; generated-schema drift
  is a test failure.

## Validation

- Domain tests cover units, lineage, source hashes, missingness/exclusions,
  availability, uncertainty, identity enforcement, vendor provenance, and v1
  compatibility.
- API tests cover capabilities, schema publication, OpenAPI registration, and a
  traceable v2 analysis response.
- The checked-in schema must equal `contract_v2_json_schema()`.
