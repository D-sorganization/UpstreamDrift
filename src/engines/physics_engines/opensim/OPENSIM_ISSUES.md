# OpenSim Implementation Issues

Companion file to [`OPENSIM_PARITY_SPEC.md`](OPENSIM_PARITY_SPEC.md). Each
section is a draft of a GitHub issue ready to be filed when this branch
lands. Issues are sized for one focused PR each (≤ 1200 LOC including
tests).

> **Honest framing:** OpenSim is **the most-greenfield** of the five
> engines in the cross-engine parity matrix. The MVP delivers a
> joint-torque-actuated humanoid that matches the Simscape contracts.
> Muscle-level work is explicitly post-MVP.

Sequencing DAG:

```
Issue 1 (model) ──► Issue 2 (coord-map) ──► Issue 3 (fk) ──► Issue 4 (sim)
                                                                  │
                                                  ┌───────────────┼───────────────┐
                                                  ▼               ▼               ▼
                                          Issue 5 (synth)  Issue 6 (fit)   Issue 7 (viz)
                                                  │               │               │
                                                  └───────────────┼───────────────┘
                                                                  ▼
                                                       Issue 8 (equivalence)
```

---

## Issue 1 — `OPENSIM-MODEL-AUTHORING`

**Title:** OpenSim: author golf-humanoid `.osim` from Rajagopal2015 base

**Labels:** `engine:opensim`, `area:model`, `priority:high`, `mvp`

### Context

The OpenSim engine has no humanoid model today. Per
[OPENSIM_PARITY_SPEC.md §3](OPENSIM_PARITY_SPEC.md#3-body-model), we
adapt `Rajagopal2015.osim` from the `shared/models/opensim/opensim-models`
submodule by stripping muscles, adding `CoordinateActuator`s for every
DOF, attaching the club rigidly to the right hand, and rescaling
segments to match the shared anthropometric YAML.

### Deliverable

- `scripts/build_humanoid_osim.py` — deterministic generator script.
- `src/engines/physics_engines/opensim/models/golf_humanoid.osim`
  (committed, generated artifact).
- `src/engines/physics_engines/opensim/models/golf_humanoid_actuators.xml`.
- `src/engines/physics_engines/opensim/models/README.md` — provenance,
  license, regeneration command.
- `src/engines/physics_engines/opensim/python/tests/test_model_loads.py`.
- License-review note recorded in `models/README.md` confirming
  redistribution rights for the chosen base model (see Risks §7.2 in
  the parity spec).

### Acceptance criteria

- `python3 scripts/build_humanoid_osim.py` is deterministic — running
  it twice produces byte-identical `.osim` output.
- `osim.Model("…/golf_humanoid.osim").initSystem()` succeeds with no
  warnings.
- `model.getCoordinateSet().getSize() == 23`.
- Every coordinate has exactly one matching `CoordinateActuator`.
- A `Body` named `Club` exists with a `WeldJoint` parent
  `right_hand` → child `Club`.
- `tests/test_model_loads.py` passes in < 5 s.
- License-review note signed off by the project owner.

### Estimated size

~800 LOC (script + tests + generated XML).

### Dependencies

None. **Root issue** — every other issue blocks on this one.

---

## Issue 2 — `OPENSIM-COORD-MAP`

**Title:** OpenSim ↔ Simscape coordinate-convention mapping helper

**Labels:** `engine:opensim`, `area:math`, `priority:high`, `mvp`

### Context

OpenSim and Simscape disagree on:

- Frame orientation (Y-up vs Z-up).
- Quaternion ordering (`[x,y,z,w]` vs `[w,x,y,z]`).
- Joint-angle sign conventions on a coordinate-by-coordinate basis.

We need a table-driven mapping module so all engines speak the same
canonical convention at the cross-engine boundary.

### Deliverable

- `python/opensim_golf/coordinate_map.py` exposing:
  - `to_simscape(q_opensim) -> q_simscape`
  - `from_simscape(q_simscape) -> q_opensim`
  - `quat_eigen_to_canonical(q_xyzw) -> q_wxyz` (and inverse)
  - `frame_y_up_to_z_up(v) -> v` (and inverse).
- `python/tests/test_coordinate_map.py`.

### Acceptance criteria

- `to_simscape ∘ from_simscape == identity` within 1e-12 absolute
  tolerance for 100 random poses.
- Golden test on the at-address pose: hard-coded expected Simscape
  pose → assertion passes.
- Module is pure-Python (no `import opensim`) so tests run on every CI
  without the OpenSim wheel.

### Estimated size

~250 LOC.

### Dependencies

Issue 1 (uses the model's coordinate ordering).

---

## Issue 3 — `OPENSIM-FK`

**Title:** OpenSim forward-kinematics: extract grip + clubhead from state

**Labels:** `engine:opensim`, `area:kinematics`, `priority:high`, `mvp`

### Context

Every cross-engine simulator must produce `grip` and `clubhead`
trajectories in the canonical world frame. Per
[OPENSIM_PARITY_SPEC.md §2.2](OPENSIM_PARITY_SPEC.md#22-forward-sim-wrapper-simulate_with_coefficientspy)
this is the boundary between OpenSim's internal state and the canonical
`SimOutput`.

### Deliverable

- `python/opensim_golf/fk.py` with:
  - `compute_grip(model, state) -> (pos, quat)`
  - `compute_clubhead(model, state) -> (pos, quat)`
  - `compute_skeleton_fk(model, states) -> dict` (vectorised).
- `python/tests/test_grip_fk_matches_simscape.py`.

### Acceptance criteria

- Grip RMSE vs Simscape `compute_skeleton_fk.m` ≤ 5 mm at three poses
  (address, top-of-backswing, impact).
- Clubhead RMSE ≤ 5 mm at the same three poses.
- Vectorised path ≥ 10× faster than a per-step Python loop.
- All quaternions returned in canonical `[w,x,y,z]` ordering.

### Estimated size

~400 LOC.

### Dependencies

Issues 1 + 2.

---

## Issue 4 — `OPENSIM-SIMULATE`

**Title:** OpenSim `simulate_with_coefficients` + polynomial torque controller

**Labels:** `engine:opensim`, `area:simulation`, `priority:high`, `mvp`

### Context

The contractual heart of the engine. Per the cross-engine spec §2.2,
every engine ships `simulate_with_coefficients(theta, options,
initial_pose) -> SimOutput`. For OpenSim this means: a polynomial-torque
controller subclassing `osim.Controller`, an integrator wrapper around
`osim.Manager`, and packaging into the canonical `SimOutput`.

### Deliverable

- `python/opensim_golf/controller.py` — `PolynomialTorqueController`
  with `set_theta`, `get_theta`, `tau_at(t, j)`.
- `python/opensim_golf/simulate_with_coefficients.py` — top-level
  wrapper.
- `python/opensim_golf/default_options.py` — `SimOptions`,
  `SynthOptions`, `FitOptions` dataclasses.
- Refactor `opensim_golf/core.py` to import canonical `SimOutput` instead
  of its bespoke `SimulationResult`.
- `python/tests/test_simulate_with_coefficients.py`.

### Acceptance criteria

- `simulate_with_coefficients(theta_zeros, default_options)` returns a
  `SimOutput` with `np.allclose(out.q, q_initial)` over the entire grid.
- `out` is a canonical `SimOutput` (correct fields + dtypes + shapes).
- Wall-clock for a 1.0 s sim ≤ 10 s on a developer laptop (warm).
- `solver_status == "success"` for nominal inputs.
- Unit-step torque on a single joint produces the analytically
  expected linear-ramp velocity (within integrator tolerance).

### Estimated size

~900 LOC.

### Dependencies

Issues 1 + 2 + 3.

---

## Issue 5 — `OPENSIM-SYNTH-ORACLE`

**Title:** OpenSim TDD oracle `synthesize_target_from_coefficients`

**Labels:** `engine:opensim`, `area:tdd`, `priority:medium`, `mvp`

### Context

Per the cross-engine spec §2.7, every engine ships
`synthesize_target_from_coefficients(theta) -> ClubTarget`. This is the
linchpin of the recovery test: given a known `theta_truth`, fitting
should recover it.

### Deliverable

- `python/opensim_golf/synthesize_target_from_coefficients.py`.
- `python/tests/test_synthesize_oracle.py`.

### Acceptance criteria

- Output is a valid `ClubTarget` with all schema fields populated.
- `source.format == "synthetic"` and `source.subject_id == "opensim"`
  (encoding engine identity via the existing `SourceProvenance` fields
  in `src/shared/python/motion_matching/club_target.py` — no schema
  change required).
- `theta_truth` is persisted by the synthesizer (e.g. returned
  alongside the `ClubTarget`, or hashed into `source.sha256`); recovery
  tests pass it explicitly to the fitter rather than reading it back
  off `source`. Adding a dedicated `theta_truth` field to
  `SourceProvenance` is a cross-engine schema migration and is out of
  scope for this MVP issue.
- Reproducible: same `theta` → byte-identical `ClubTarget` (within
  floating-point rounding) on two consecutive runs.

### Estimated size

~300 LOC.

### Dependencies

Issue 4.

---

## Issue 6 — `OPENSIM-FIT`

**Title:** OpenSim `fit_swing_opensim` motion-matching driver

**Labels:** `engine:opensim`, `area:optimization`, `priority:high`, `mvp`

### Context

The user-facing motion-matching entry point for OpenSim. MVP optimizer
is `scipy.optimize.minimize(method="L-BFGS-B")` with finite-difference
gradients. Imports `compute_cost` from
`shared/python/motion_matching/cost.py` — **does not** write its own
cost function (cross-engine spec §2.3).

### Deliverable

- `python/opensim_golf/fit_swing_opensim.py`.
- `python/tests/test_fit_recovers_truth.py`.

### Acceptance criteria

- Recovery test "synthesize → fit → `np.allclose(theta_fit,
theta_truth, atol=1e-2)`" converges in < 60 s for a 1.0 s
  synthesised target on a developer laptop.
- `FitResult.solver_status == "success"`.
- Final cost < `1e-3` for noise-free synthesis-recovery cases.
- Returned `FitResult` matches the canonical schema (cross-engine
  spec §2.4).

### Estimated size

~600 LOC.

### Dependencies

Issues 4 + 5.

---

## Issue 7 — `OPENSIM-VIZ`

**Title:** OpenSim three canonical visualisation figures

**Labels:** `engine:opensim`, `area:visualization`, `priority:medium`, `mvp`

### Context

Per the cross-engine spec §2.5, every engine renders the three
canonical figures: trajectory overlay, error timecourse, fit-quality
card. Reuse `shared/python/motion_matching/plot_*.py` where possible;
only the 3D viewer is engine-specific.

### Deliverable

- `python/opensim_golf/visualize.py` exposing:
  - `plot_trajectory_overlay(target, sim_out) -> Figure`
  - `plot_error_timecourse(target, sim_out) -> Figure`
  - `plot_fit_quality_card(fit_result) -> Figure`.
- `python/tests/test_visualize_smoke.py`.
- Optional: thin wrapper `viz_opensim_native.py` invoking
  `opensim.Visualizer` for interactive local use.

### Acceptance criteria

- All three figures render headlessly under matplotlib Agg.
- Snapshot test against committed reference PNGs (5% pixel tolerance).
- No matplotlib warnings emitted under pytest.
- Runtime of the smoke-test suite < 5 s.

### Estimated size

~500 LOC.

### Dependencies

Issues 4 + 6.

---

## Issue 8 — `OPENSIM-EQUIVALENCE`

**Title:** Cross-engine equivalence test: OpenSim vs Simscape oracle

**Labels:** `engine:opensim`, `area:cross-engine`, `priority:high`, `mvp`

### Context

The gating test for cross-engine parity. Per the cross-engine spec §2.2,
every engine must round-trip a fixed `theta` to within 5 mm grip-position
RMSE vs the Simscape reference at three poses (address, top-of-backswing,
impact).

### Deliverable

- `tests/cross_engine/test_opensim_vs_simscape.py`.
- A small reference fixture (`.npz` or `.json`) capturing the Simscape
  ground-truth grip + clubhead trajectories so this test runs on CI
  without requiring MATLAB.

### Acceptance criteria

- Grip RMSE ≤ 5 mm at all three poses.
- Clubhead RMSE ≤ 5 mm at all three poses.
- Test marked `@pytest.mark.slow` and `@pytest.mark.requires_opensim`.
- Test runtime < 3 minutes warm.

### Estimated size

~250 LOC.

### Dependencies

All of Issues 1–6.

---

## Post-MVP follow-ups (not part of this wave)

These are **explicitly out of scope** for the MVP but are recorded so
they aren't lost:

- `OPENSIM-MUSCLES-RESTORE` — re-enable Rajagopal2015's 80-muscle force
  set; runs Computed Muscle Control (CMC) on a known-good kinematics
  trajectory. Blocked on body-marker mocap availability.
- `OPENSIM-FIT-MULTISTART` — multistart wrapper around `fit_swing_opensim`
  using the XML-print-and-reload pattern for multiprocessing.
- `OPENSIM-CLUB-ATTACHMENT-FLEX` — replace the rigid `WeldJoint` with a
  `BushingForce` to model grip compliance. Research-only.
- `OPENSIM-VIZ-NATIVE` — interactive 3D viewer using
  `opensim.Visualizer` for local debugging (not CI).
- `OPENSIM-PERFORMANCE` — replace per-step Python controller callback
  with a pre-compiled `osim.PrescribedController` for 2–3× speedup.

---

_Issue drafts landed 2026-05-06 alongside the OpenSim parity spec._
