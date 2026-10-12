---
issue: 11792
summary: "Add a strict analytic muscle replay input contract."
dl_state: "in_progress"
next_step: "Build a separately pinned replay worker with complete execution provenance; validate force, activation, trajectory, and native replay behavior before any qualification claim."
branch: "feat/analytic-replay-input-12392"
---

The input-contract slice for child issue #12392 adds a bounded standard-library
JSON decoder for an explicitly synthetic, one-coordinate analytic muscle replay
fixture. It preserves raw excitation and activation-floor fields separately
and validates clock, units, bounds, IDs, and derived initial path length. It
introduces no execution worker, force law, native SDK call, or qualification
result. The calculation inventory stays empty; all registered model/variant
rows remain unqualified.

Validation: 32 focused tests pass with one existing warning; Ruff and the
canonical diff-scoped mypy check pass. See
`docs/development/feedback_controls/F07_ANALYTIC_MUSCLE_REPLAY_INPUT_TURNOVER.md`.
