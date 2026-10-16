# F-UD-018

**Two carry distances per flight: last below-ground sample vs interpolated landing**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/physics/ball_trajectory_analysis.py:21`, `src/shared/python/physics/ball_simulator.py:94`, `rust_core/upstream-physics/src/ball_flight.rs:224`, `src/shared/python/physics/ball_enhanced_simulator.py:221`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/physics/ball_trajectory_analysis.py:21-28; src/shared/python/physics/ball_simulator.py:94-111;
  rust_core/upstream-physics/src/ball_flight.rs:224; src/shared/python/physics/ball_enhanced_simulator.py:221-223
- Evidence: `calculate_carry_distance` returns `hypot(x, y)` of `trajectory[-1]`, which Rust terminates at the first sample with
  `z <= 0` (and the enhanced simulator appends the point with `position[2] < 0` before breaking); `simulate_landing`
  interpolates to z = 0.
- Failure scenario: at dt = 0.01 s and ~30 m/s horizontal landing speed, `analyze_trajectory()["carry_distance"]` is up to
  0.3 m long and `simulate_landing()` is not; `landing_angle`/`flight_time` use the same below-ground sample. The shot-pattern
  studies in `docs/research/shot_pattern_analysis` quote metre-level dispersion statistics from one of these.
- Why a newer engineer misses this: both numbers are "where the ball came down"; the discrepancy is below the eye's resolution
  on a plot and above the tolerance of a parity test.
- Proposed change: make `TrajectoryAnalysisMixin` interpolate the final segment to z = 0 (or clip the last point) and route
  `simulate_landing` through the same helper; Rust `terminate` at `t > 0.05` also fails a downhill chip launched at z = 0 with
  negative launch angle, so guard on `vz < 0` instead of a time constant.
- Verification: `abs(carry_distance - norm(simulate_landing()[:2])) < 1e-9`.
- Effort: S

## Draft Pull Request

**Title:** `fix(physics): interpolate landing to z=0 so carry has one definition`

### Why This Matters (The Pitch)

A single flight has two carry distances. calculate_carry_distance returns the hypot of the last trajectory sample, which the integrator emits at the first sample with z at or below zero, while simulate_landing interpolates to z = 0. At dt = 0.01 s and about 30 m/s horizontal landing speed, analyze_trajectory()['carry_distance'] can be up to 0.3 m long, and landing_angle and flight_time use the same below-ground sample. The shot-pattern studies in docs/research/shot_pattern_analysis quote metre-level dispersion statistics from one of these numbers. The difference is invisible on a plot and larger than a parity test's tolerance. The fix is to interpolate the final segment inside the analysis mixin and route simulate_landing through the same helper; as a bonus, the Rust termination guard on a time constant should become a vz < 0 check so downhill chips with negative launch angles work. Size S, and researchers get one answer.

### Proposed PR Body

## Problem

`ball_trajectory_analysis.py:21-28` returns `hypot(x, y)` of `trajectory[-1]`, which Rust (`ball_flight.rs:224`) and the enhanced simulator (`ball_enhanced_simulator.py:221-223`) end at the first `z <= 0` sample, while `simulate_landing` (`ball_simulator.py:94-111`) interpolates to z = 0. Carry can be up to 0.3 m long at dt = 0.01 s and ~30 m/s horizontal speed.

## Change

- [ ] Make `TrajectoryAnalysisMixin` interpolate the final segment to z = 0 (or clip the last point) and route `simulate_landing` through the same helper
- [ ] Replace the Rust `t > 0.05` termination guard with a `vz < 0` condition

## Verification

`abs(carry_distance - norm(simulate_landing()[:2])) < 1e-9`.

## Risk and Rollback

Low to medium: reported carry and landing angle shift by up to the stated 0.3 m. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-018

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
