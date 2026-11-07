---
issue: 14427
type: feat
summary: "Event-relative time normalisation (101-point resampling) between named swing events"
---

Added 101-point resampling (`0%` to `100%` in `1%` increments, `n=101` points) between ordered swing events (e.g. `address -> top -> impact`) in `biomechanics.metric_definitions.event_time_normalisation`.
Declared `event_relative_101` normalization kind and `EventTimeNormalisationDescriptor` schema in `schema.py` and exposed in `metric_definitions`.
Full TDD test suite verifying analytic signals, monotonicity, event ordering, and boundary conditions.
