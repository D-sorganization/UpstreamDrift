# ADR 0040: Data-Free Launch-Monitor Conformance Bundle

- **Status:** Accepted
- **Date:** 2026-08-21
- **Decision owners:** Launch-monitor analytics maintainers
- **Validation:** `tests/unit/launch_monitor/test_conformance_bundle.py`

## Context

UpstreamDrift owns several versioned launch-monitor result contracts, but a
consumer previously had to assemble independent fixtures to verify analysis
v2, player covariation, attested longitudinal sessions, source-backed strokes
gained, and the distance/target proxy. That makes cross-repository drift easy
to miss. Publishing private or observed shot rows as a shared fixture is not an
acceptable solution.

## Decision

Publish `launch-monitor-analytics-conformance/1.0.0` as the canonical Python
and JSON Schema authority for one deterministic consumer bundle. It contains
exactly one available and one structured-unavailable synthetic case for each
of the five result families.

Each case retains its complete versioned result envelope and a uniform wrapper
with:

- canonical or source-declared units and their authority;
- conservative claims, including `causal_inference=false`;
- separate player, session, and order evidence;
- content-addressed synthetic source references and source-joinable backing
  record hashes;
- explicit exclusion counts; and
- a canonical scenario SHA-256.

The bundle carries a second canonical SHA-256 over all bundle content except
that self-referential hash field. Validation fails closed after any mutation.
Generation is deterministic and the checked-in schema and golden JSON must
equal the Python authority.

Golden snapshot serialization quantizes finite floating-point values to eight
significant digits before validation and hashing. This boundary removes
platform- and dependency-specific least-significant tails from linear-algebra
results while retaining substantially more precision than any source-declared
launch-monitor measurement. The analytics result contracts and computations
are not quantized or otherwise changed.

## Data Boundary

The bundle is classified
`synthetic_contract_fixture_no_private_rows`. It embeds no input `records`
array and no observed or private corpus row. Some underlying result contracts
retain derived synthetic per-row outputs or opaque backing hashes because
those fields are part of their public wire contract; they are not measurements
from a player or vendor export.

Synthetic vendor labels remain vendor-comparable provenance examples only.
They do not claim device emulation, certification, or reverse-engineered vendor
physics.

## Consequences

- Python, OpenAPI/JSON Schema, React, PyQt, and external consumers can validate
  one content-addressed interoperability artifact.
- Available and unavailable paths cannot silently diverge between consumers.
- Identity and chronological evidence remain explicit rather than inferred.
- Unknown variable units remain unknown unless a source explicitly declares
  them; a source declaration is not promoted to canonical authority.
- This fixture adds no endpoint and changes no analytics behavior.
