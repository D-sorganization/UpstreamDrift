# ADR 0038: Canonical Launch Monitor Player Covariation Contract

- Status: Accepted
- Date: 2026-08-20
- Decision Makers: UpstreamDrift Maintainers
- Related Issues/PRs: #8807
- Builds On: ADR 0034, ADR 0036

## Context

A correlation over pooled launch-monitor shots mixes variation within a player
with differences between players. Those effects can differ in magnitude or
even direction. A population statement also requires an explicit method for
combining player estimates, handling sparse or constant groups, and reporting
heterogeneity. Generic v1/v2 analysis remains useful, but it does not define
that player-specific scientific boundary.

The result must remain auditable without copying restricted source values. It
also must never infer a player from a session, club, source file, row position,
or other collection structure.

## Decision

UpstreamDrift owns the versioned
`launch-monitor-player-covariation/1.0.0` contract. A selected-pair result
reports distinct pooled, player-centered, between-player, and per-player
descriptive estimates. Population synthesis uses inverse-variance Fisher-z
fixed effects and DerSimonian-Laird Fisher-z random effects, with Q,
tau-squared, and I-squared heterogeneity diagnostics. Confidence intervals and
their assumptions are named in the result.

Every analysis requires a trusted `PlayerIdentityV2` whose identifier column
matches the request. ADR 0036 forbidden pseudo-identities remain invalid even
when attested. Pairwise missingness, non-numeric/non-finite values, blank player
labels, small groups, constant variables, and insufficient eligible players
are reported through explicit counts or typed unavailable states.

The result reuses v2 units, dataset authority, source references, vendor/model
provenance, and source-joinable backing-record hashes. Unknown or retained
source fields may be selected, but their units remain `unknown` unless the
source explicitly declares them; selection does not make a unit canonical.

A bounded deterministic all-pairs scan is exploratory. It reports unavailable
pairs, direction consistency, and multiplicity warnings, and ranks available
pairs by absolute random-effects correlation. The contract makes no causal,
device-emulation, certification, or universal population claim.

## Alternatives Considered

1. Keep the calculation only in a UI client. Rejected because PyQt, web, and
   downstream research clients would drift in identity and inference rules.
2. Report pooled correlation as the player relationship. Rejected because
   aggregation reversal can make that interpretation materially wrong.
3. Drop ineligible players or pairs silently. Rejected because absence of an
   estimate is part of the scientific result and must be machine-readable.

## Consequences

- Python, HTTP, JSON Schema, generated clients, and fixtures share one authority.
- Consumers can distinguish within-player behavior, between-player composition,
  and a heterogeneous population synthesis.
- Player labels appear in selected-pair results and must be handled under the
  source dataset's privacy and usage controls.
- Exploratory rankings require confirmatory testing on held-out data; they are
  not automatic swing diagnoses or improvement claims.

## Validation

- Golden domain fixtures exercise aggregation reversal, fixed/random synthesis,
  heterogeneity, unavailable groups, missingness, units, and source-linked rows.
- API tests cover schema/OpenAPI publication, identity failures, pair results,
  exploratory scans, and capability discovery.
- Schema generation is idempotent, and Ruff, mypy, Bandit, architecture-budget,
  API, and focused domain gates run before protected publication.
