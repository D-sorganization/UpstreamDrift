# ADR 0039: Attested Launch Monitor Longitudinal Sessions

- Status: Accepted
- Date: 2026-08-20
- Decision Makers: UpstreamDrift Maintainers
- Related Issues/PRs: #8808
- Builds On: ADR 0034, ADR 0036, ADR 0038

## Context

Shot-level regressions over time treat repeated shots as independent evidence
and let high-volume sessions dominate. Session names, filenames, row position,
and shot IDs also cannot establish a player, session boundary, or chronological
order. A canonical longitudinal result therefore needs an explicit evidence
boundary and a session-level inference unit before PyQt, React, or research
consumers can describe change over time.

Longitudinal association is not proof of player improvement. Equipment,
environment, club selection, intervention timing, and other unmeasured factors
can create or obscure a directional pattern.

## Decision

UpstreamDrift owns contract
`launch-monitor-longitudinal-session/1.0.0`. Analysis requires trusted,
separate player identity, session identity, and numeric or timestamp order
evidence. It fails closed rather than deriving any of those fields from data
layout.

Shots are first reduced to one equal-weight player/session/stratum aggregate.
Per-player slopes then use one equal-weight mean per ordered session. The
pooled descriptive estimate uses player-fixed-effects OLS with declared
numeric confounders and categorical strata. Uncertainty uses a player-clustered
sandwich covariance, CR1 finite-cluster correction, and a Student-t reference
with player-cluster degrees of freedom.

Fewer than the declared player-cluster minimum, too few ordered sessions,
rank deficiency, degenerate clustered variance, nonconstant within-session
order, missing fields, and no complete finite observations produce typed
unavailable states. Missingness and complete row-level backing hashes remain
available even when an estimate is not.

Every result labels its primary unit `player_session_stratum`, its association
scope `descriptive_directional`, and both shot-level inference and causal
improvement false. The request's preferred metric direction is retained as
context; it does not turn a slope into an improvement claim.

## Alternatives Considered

1. Fit a slope to every shot. Rejected because it creates pseudo-replication
   and volume weighting by session.
2. Infer sessions or order from filenames or source rows. Rejected because the
   collection layout is not identity or chronological evidence.
3. Label an adjusted positive slope as improvement. Rejected because measured
   and unmeasured confounding remains and the design is observational.
4. Return a zero estimate when uncertainty is not estimable. Rejected because
   statistical unavailability is evidence that consumers must preserve.

## Consequences

- Python, HTTP, JSON Schema, generated clients, and golden fixtures share one
  versioned authority.
- Repeating shots within a session does not change the session count or pooled
  point estimate.
- A small allowed cluster count is a minimum computability boundary, not a
  claim of high-powered generalization; consumers must retain the reported
  interval, cluster count, and limitations.
- Consumers may describe observed directional association but must not claim
  coaching efficacy, causal improvement, or device/model certification.

## Validation

- A content-addressed 36-shot synthetic fixture covers four players and twelve
  ordered sessions, with exact source and row backing.
- Tests cover shot duplication, trusted-evidence failures, nonconstant order,
  incomplete/non-finite rows, blank identities, insufficient clusters,
  explicit strata/confounders, strict schemas, and API publication.
- Schema generation is deterministic; Ruff, architecture budgets, focused and
  broader launch-monitor tests, generated-client freshness, and protected CI
  run before merge.
