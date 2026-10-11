# F08 Moco Progress Instrumentation

The native Moco runner now atomically preserves its latest progress fields and
an ordered request/source-bound stage journal, including the current-input
audit duration and monotonic timestamps. Immediately before study.solve(), it
records actual study-solver getter outcomes after the existing single-thread
setting while preserving missing and error states. No model, request, physics,
or solver policy changes; progress is diagnostic evidence only and does not
qualify any production row.
