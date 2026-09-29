# ADR 0031: Canonical Launch Monitor Shot Schema

- Status: Accepted
- Date: 2026-08-04
- Decision Makers: UpstreamDrift Maintainers
- Related Issues/PRs: [#8342](https://github.com/D-sorganization/UpstreamDrift/issues/8342)

## Context

Launch-monitor exports differ by vendor, device, software version, locale,
selected report, and subscription entitlement. Several fields also differ in
meaning: a vendor may measure a value, estimate it, derive it from other fields,
or omit it. A narrow dataclass containing only shared fields would discard the
vendor-specific variables that make interdependency research valuable. A raw
table alone would make cross-vendor analysis unsafe because units and naming
would remain ambiguous.

The research questions are observational. Correlation, regularized regression,
feature importance, and shallow neural networks can reveal predictive structure,
but cannot identify causal impact physics without an experimental design.
Identity-derived variables also create deterministic leakage. For example,
smash factor contains ball speed by definition.

## Decision

Use a two-layer, wide shot table:

1. Canonical columns use stable names and units from `METRICS` in
   `src/shared/python/launch_monitor/schema.py`. Speeds are m/s, distances are
   metres, angles are radians, spin is rad/s, and time is seconds.
2. Every original field is retained as `source::<exact header>`. The import
   manifest records the original header, source unit, unit evidence, SHA-256,
   source row, profile, and warnings.

Reported canonical values carry a `status::<metric>` column. The initial status
vocabulary is `reported`, `measured`, `estimated`, `derived`, and `unknown`.
`reported` is deliberately conservative when an export does not reveal whether
the vendor measured or estimated the value.

Vendor adapters are header-fingerprint profiles layered over a generic mapping
workflow. They are not rigid parsers that claim all historical releases use one
layout. Unknown fields survive import. Users can override the profile, mapping,
unit, sign multiplier, and measurement status before committing an import.

The project file stores original imported sessions and a durable treatment audit
log. Treatment produces a separate analysis view; it does not overwrite raw
sessions. Matched-shot cross-monitor comparisons are distinct from unmatched,
descriptive comparisons.

## Alternatives Considered

1. One dataclass per vendor: rejected because it fragments aggregation and
   duplicates analysis logic.
2. Only a lowest-common-denominator schema: rejected because it discards rare
   club-delivery, putting, and vendor-quality fields.
3. Store only raw exports and map at analysis time: rejected because unit and
   convention errors would be repeated throughout the analysis stack.
4. Treat all reported values as measured: rejected because several devices and
   configurations estimate some metrics, and export files do not always expose
   that distinction.

## Consequences

- Positive: imports are auditable, extensible, loss-minimizing, and safe to
  aggregate across sessions.
- Positive: headless analysis does not depend on PyQt6; MLP support imports
  scikit-learn lazily.
- Positive: deterministic identities and derivations can be marked and excluded
  from predictors.
- Negative: project tables are wider than a lowest-common-denominator table.
- Negative: a new or localized export can require mapping review even when its
  vendor is recognized.
- Follow-up: add verified profiles when maintainers receive legally shareable,
  versioned sample exports; do not infer private protocols.

## Validation

- Vendor-shaped fixtures cover TrackMan, Foresight, FlightScope, Garmin,
  SkyTrak, and Uneekor header families.
- Unit, provenance, raw-field preservation, project persistence, filtering,
  derivation status, statistics, predictive models, monitor comparison,
  dispersion, trends, and PyQt6 embed behavior have focused tests.
- The launcher manifest and feature-parity registry include the workbench.
- Ruff, format, file/module size, focused tests, and offscreen GUI rendering are
  required before merge.
