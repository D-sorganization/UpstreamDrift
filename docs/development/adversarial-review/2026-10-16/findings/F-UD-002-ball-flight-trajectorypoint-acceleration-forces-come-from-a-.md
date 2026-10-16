# F-UD-002

**Ball-flight `TrajectoryPoint.acceleration`/`forces` come from a different aero model than the trajectory**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | M      | proposed |

**Locations:** `src/shared/python/physics/ball_simulator.py:145`, `src/shared/python/physics/ball_simulator.py:260`, `src/shared/python/physics/ball_simulator.py:298`, `src/shared/python/physics/ball_simulator.py:343`, `src/shared/python/physics/ball_properties.py:43`, `rust_core/upstream-physics/src/aerodynamics.rs:300`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/shared/python/physics/ball_simulator.py:145-154; src/shared/python/physics/ball_simulator.py:260-261;
  src/shared/python/physics/ball_simulator.py:298; src/shared/python/physics/ball_simulator.py:343;
  src/shared/python/physics/ball_properties.py:43-49; rust_core/upstream-physics/src/aerodynamics.rs:300-313;
  rust_core/upstream-physics/src/ball_flight.rs:209-219
- Evidence:
  ```python
  # ball_simulator.py:148-152 (Rust path)
  # ball.cd0 is the spin polynomial's constant term, a different
  # model; passing it under-predicted drag by ~16 %.
  drag_coefficient=float(GOLF_BALL_DRAG_COEFFICIENT),
  # ball_simulator.py:260-261 (post-processing the SAME trajectory)
  forces = self._calculate_forces(vel, launch)
  acc = (forces["gravity"] + forces["drag"] + forces["magnus"]) / self.ball.mass
  # ball_simulator.py:298, 343
  omega = launch.spin_rate * 2 * np.pi / 60          # constant, never decays
  cd = self.ball.cd0 + s_ratio * (self.ball.cd1 + s_ratio * self.ball.cd2)
  ```
  Rust integrates `cd_dimpled_sphere(Re, 0.25)` with `domega/dt = -spin_decay_rate*omega`.
- Failure scenario: a driver at 70 m/s, 3000 rpm gives Python Cd ≈ 0.234 vs Rust Cd ≈ 0.254 at launch (≈8 %), and the gap grows
  along the flight because Python keeps launch spin while Rust decays it. Every `TrajectoryPoint.acceleration` and `forces`
  dict exported via `flight_trajectory_export.py` and the ball-flight API is therefore inconsistent with the `position`/`velocity`
  columns next to it — `np.gradient(velocity, t)` will not reproduce `acceleration`. Anyone fitting drag from exported forces
  recovers the wrong model.
- Why a newer engineer misses this: the two models live in different languages and the Rust path only returns `(t, x, v)`;
  "recompute forces in Python" looks like harmless post-processing. The author even noticed the 16 % discrepancy for the Cd
  input, fixed it on the way _in_, and left the way _out_ untouched.
- Proposed change: either (a) have `simulate_ball_trajectory_py` return the per-step `AeroForces` (the Rust closure already
  computes them) and store those, or (b) derive `acceleration` by differentiating the Rust velocities and drop the `forces`
  dict from the Rust-backed path. Also make `rust_kernel.py:7-8` and `ball_simulator.py:75-77` stop promising a Python/Numba
  fallback that `ball_simulator.py:128-131` raises `RuntimeError` on.
- Verification: `np.testing.assert_allclose(np.gradient(vel, t, axis=1), acc, rtol=1e-3)` on a driver trajectory; a test that
  the integrated drag force at t=0 equals `0.5*rho*A*cd_dimpled_sphere(Re, 0.25)*v²`.
- Effort: M

## Draft Pull Request

**Title:** `fix(physics): derive exported accel/forces from the integrated model`

### Why This Matters (The Pitch)

Every exported TrajectoryPoint carries a position and velocity from the Rust integrator and an acceleration and forces dict from a different model. The Rust path integrates the Reynolds-crisis drag with decaying spin; the Python post-processing recomputes forces with the spin-polynomial drag and a launch spin that never decays. For a driver at 70 m/s and 3000 rpm, Python Cd is about 0.234 against Rust's 0.254 at launch, roughly 8 %, and the gap widens along the flight. So np.gradient(velocity, t) will not reproduce the acceleration sitting in the same row, and anyone fitting drag from flight_trajectory_export.py output or the ball-flight API recovers the wrong model. The author spotted a 16 % Cd mismatch on the way in and fixed it there, then left the way out alone. Fixing it makes the exported physics self-consistent for researchers, API consumers and the shot-pattern studies, and lets the module stop promising a Python/Numba fallback that the simulator raises on.

### Proposed PR Body

## Problem

`src/shared/python/physics/ball_simulator.py:260-261` recomputes `forces` and `acc` in Python for a trajectory integrated in Rust (`rust_core/upstream-physics/src/aerodynamics.rs`, `ball_flight.rs`). Python uses the `ball_properties.py:43-49` spin polynomial and a constant `omega` (`:298`, `:343`); Rust uses `cd_dimpled_sphere(Re, 0.25)` with `domega/dt = -spin_decay_rate*omega`. At 70 m/s and 3000 rpm that is Cd ≈ 0.234 vs ≈ 0.254 at launch.

## Change

- [ ] Either (a) have `simulate_ball_trajectory_py` return the per-step `AeroForces` the Rust closure already computes and store those, or (b) derive `acceleration` by differentiating the Rust velocities and drop `forces` from the Rust-backed path
- [ ] Remove the Python/Numba fallback promises at `rust_kernel.py:7-8` and `ball_simulator.py:75-77` that `ball_simulator.py:128-131` contradicts by raising `RuntimeError`

## Verification

`np.testing.assert_allclose(np.gradient(vel, t, axis=1), acc, rtol=1e-3)` on a driver trajectory, plus a test that the integrated drag force at t=0 equals `0.5*rho*A*cd_dimpled_sphere(Re, 0.25)*v²`.

## Risk and Rollback

Medium: consumers of the exported `forces` dict will see changed values (that is the fix). Rollback is a revert; the trajectory positions and velocities are unchanged.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-002

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
