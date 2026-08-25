# ADR 0036: Launch Monitor Player, Session, and Order Identity Boundaries

- Status: Accepted
- Date: 2026-08-20
- Decision Makers: UpstreamDrift Maintainers
- Related Issues/PRs: #8805
- Amends: ADR 0034

## Context

The v2 contract required evidence before player grouping, but an attested
`session_id` could still be declared as the player identifier. Session, club,
source, filename, and row position describe collection structure, not a person.
Accepting one as player identity would silently merge or split people and make
within-player, population, and longitudinal estimates invalid.

Longitudinal work also needs session boundaries and observation order. Those
facts are useful evidence, but neither establishes player identity. They need
their own typed records so clients cannot overload `PlayerIdentityV2`.

## Decision

`PlayerIdentityV2` rejects the normalized session, club, source, file/filename,
row-order, and source-row field names regardless of trust level or attestation.
This is an invalid request and therefore a model-validation error (`422` over
FastAPI), rather than a statistical unavailable state.

`AnalysisContextV2` and `LaunchMonitorAnalysisResultV2` carry two additive
records:

- `SessionIdentityV2` declares the session identifier column, trust level, and
  evidence supporting the session boundaries.
- `OrderEvidenceV2` declares the order column, timestamp/ordinal/source-sequence
  semantics, unit, trust level, and evidence supporting the ordering.

A declared session or order record must be complete. The default
`not_provided` state carries no associated fields. Missing or untrusted evidence
does not invalidate an analysis that does not use it. A future operation that
requires session or order evidence must return a structured unavailable result
rather than infer it from source layout or row position.

The v1 request and result remain unchanged. The v2 additions have defaults, so
existing valid v2 requests remain accepted.

## Consequences

- User attestation cannot convert collection metadata into player identity.
- Upcoming longitudinal operations can state session and time-scale evidence
  without conflating either with a person.
- Clients receive the same identity records in the response that they supplied
  in the context, preserving the audit trail.
- A malformed identity declaration fails before statistical computation.

## Validation

- Domain tests reject every forbidden pseudo-identity, including an explicitly
  user-attested `session_id`.
- Domain tests require complete session and order evidence.
- API tests verify `422` errors for forbidden player identifiers and successful
  round trips for separate session/order evidence.
- Generated JSON Schema and OpenAPI-derived client types are deterministic and
  checked for drift.
