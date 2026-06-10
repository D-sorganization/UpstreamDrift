# ADR-0029: Inverse Swing Optimization Core

- Status: Proposed
- Date: 2026-06-10
- Decision Makers: Codex agent, physics maintainers
- Related Issues/PRs: #7220

## Context

`SwingBallFlightPipeline` is forward-only: it maps a swing state through impact
and ball flight. Issue #7220 needs the inverse core: given target carry, height,
and lateral curve, search for swing parameters that make the existing forward
pipeline produce that target. This first PR intentionally excludes GUI target
mode so the physics API and diagnostics can be reviewed independently.

## Decision

Add `src/shared/python/physics/swing_optimizer.py` with a `SwingOptimizer` that
uses SciPy `minimize(..., method="SLSQP")` over four bounded parameters:
clubhead speed, loft, attack angle, and face-to-path. The optimizer composes
`SwingBallFlightPipeline` by default and accepts an injected forward pipeline for
focused tests.

The objective is a weighted squared error over available target dimensions:
carry error is always included, and max-height / lateral errors are included
when present on `FlightTarget`. Driver and 7-iron `ClubPreset` helpers define
conservative bounds and initial guesses. Runtime is bounded by both max
iterations and a monotonic-clock timeout. Forward evaluations are cached by the
rounded optimizer parameter tuple.

## Alternatives Considered

1. Grid search. Simple and deterministic, but it scales poorly once GUI target
   mode adds extra variables or tighter tolerances.
2. Bayesian optimization. Useful for expensive simulators, but heavyweight for
   the current Rust-backed flight kernel and not needed for the first API.
3. Differentiable or gradient-based engine coupling. Attractive long-term, but
   it would cross engine boundaries and exceed this bounded first PR.

## Consequences

- Positive: The inverse core reuses the existing forward pipeline and reports
  convergence, timeout, evaluation count, residual errors, and unreachable-target
  diagnostics without changing GUI behavior.
- Negative: The first implementation optimizes the current simplified swing
  parameterization, so it does not claim full biomechanical inverse dynamics.
- Follow-ups: Add GUI target mode and controller-level offscreen tests before
  treating #7220 as fully accepted.

## Validation

Focused unit tests cover driver and 7-iron roundtrips against an injected
forward pipeline, unreachable-target diagnostics, and timeout behavior. CI
should also run ruff, spec path checks, version checks, file-size budget, and
diff/substance checks for the PR.
