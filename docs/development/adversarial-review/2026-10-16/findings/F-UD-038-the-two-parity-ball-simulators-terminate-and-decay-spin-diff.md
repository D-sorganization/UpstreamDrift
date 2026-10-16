# F-UD-038

**The two "parity" ball simulators terminate and decay spin differently**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | numerics | high       | S      | proposed |

**Locations:** `rust_core/upstream-physics/src/ball_flight.rs:224`, `rust_core/upstream-physics/src/ball_flight.rs:218`, `src/shared/python/physics/ball_enhanced_simulator.py:221`, `src/shared/python/physics/ball_enhanced_simulator.py:231`

## Finding

- Severity: Low - Category: numerics
- Confidence: high
- Location: rust_core/upstream-physics/src/ball_flight.rs:224; rust_core/upstream-physics/src/ball_flight.rs:218;
  src/shared/python/physics/ball_enhanced_simulator.py:221-223; src/shared/python/physics/ball_enhanced_simulator.py:231-232
- Evidence: Rust: `terminate = t > 0.05 && z <= 0`, spin in the RK4 state (`domega = -k*omega`); Python enhanced: break when
  `position[2] < 0 and t > 0` after appending the point, spin decayed by operator splitting after the step.
- Failure scenario: `test_driver_trajectory_matches_public_simulator` matches carry/apex to 1e-12 only because both consume
  the same Rust trajectory; the Python RK4 path used for Monte-Carlo (`monte_carlo_simulation`) is a different integrator whose
  parity is never asserted. Splitting spin decay out of the RK4 stages is O(dt) accurate, not O(dt⁴).
- Why a newer engineer misses this: both are "RK4 with the same forces".
- Proposed change: put spin in the Python state vector too (7-state), share the termination predicate, and add a Python-vs-Rust
  trajectory parity test with `rtol=1e-9`.
- Verification: that test.
- Effort: S

## Draft Pull Request

**Title:** `fix(physics): match Python ball simulator to Rust on spin and landing`

### Why This Matters (The Pitch)

The Python RK4 ball simulator used for Monte Carlo is a different integrator from the Rust one, and nothing asserts they agree. Rust keeps spin in the RK4 state and terminates when t > 0.05 and z <= 0; Python decays spin by operator splitting after the step (O(dt) accurate, not O(dt^4)) and breaks after appending a below-ground point. The existing parity test passes only because both sides consume the same Rust trajectory. Adding spin to the Python state vector, sharing the termination predicate and adding an rtol=1e-9 parity test closes the gap.

### Proposed PR Body

## Problem

`rust_core/upstream-physics/src/ball_flight.rs:218,224` vs `src/shared/python/physics/ball_enhanced_simulator.py:221-223,231-232`: different termination predicate and spin-decay treatment; `monte_carlo_simulation` uses the Python path.

## Change

- [ ] Put spin in the Python state vector (7-state)
- [ ] Share the termination predicate
- [ ] Add a Python-vs-Rust trajectory parity test with `rtol=1e-9`

## Verification

The Python-vs-Rust trajectory parity test.

## Risk and Rollback

Low to medium: Monte Carlo outputs will shift slightly. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-038

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
