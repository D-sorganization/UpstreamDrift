---
issue: 11898
summary: "Instrument native Moco solve progress and solver-stage identity without changing model, request, physics, or solver policy"
---

The native Moco runner now atomically preserves its latest progress fields and
an ordered request/source-bound stage journal, including the current-input
audit duration and monotonic timestamps. Immediately before study.solve(), it
records actual study-solver getter outcomes after the existing single-thread
setting while preserving missing and error states. No model, request, physics,
or solver policy changes; progress is diagnostic evidence only and does not
qualify any production row.

The event recorder is factored into a typed helper to keep the single-attempt
solver within the architecture function budget without changing event fields,
ordering, timing origins, or solver policy.
