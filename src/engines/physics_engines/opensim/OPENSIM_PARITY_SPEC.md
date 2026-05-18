# OpenSim Parity Specification

> **Status:** greenfield. OpenSim is **the most-stubbed of the five engines**
> in this repository. Today the engine has _no_ humanoid model, _no_ forward
> simulator wrapper, _no_ fit driver, _no_ visualisation, and _no_ tests.
> This document is the implementation plan to bring it to feature parity
> with the Simscape Multibody motion-matching pipeline (the "primary path"
> defined in [`../../CROSS_ENGINE_PARITY_SPEC.md`](../../CROSS_ENGINE_PARITY_SPEC.md)).
>
> **Scope:** the Minimum-Viable-Product (MVP) is a **joint-torque-actuated**
> humanoid that consumes the canonical `ClubTarget` schema, produces the
> canonical `SimOutput`, and returns a `FitResult` that the cross-engine
> leaderboard can compare. **Muscle-level forward dynamics is explicitly
> post-MVP** — see §8.
>
> **Audience:** every contributor (human or agent) implementing or modifying
> code under `src/engines/physics_engines/opensim/`.

---

## Table of contents

1. [Current state (honest inventory)](#1-current-state-honest-inventory)
2. [Target architecture](#2-target-architecture)
3. [Body model](#3-body-model)
4. [Implementation plan (atomic, testable issues)](#4-implementation-plan-atomic-testable-issues)
5. [TDD / DbC / DRY / LOD compliance](#5-tdd--dbc--dry--lod-compliance)
6. [Performance baseline + targets](#6-performance-baseline--targets)
7. [Risks and open questions](#7-risks-and-open-questions)
8. [Post-MVP: muscle-level body model](#8-post-mvp-muscle-level-body-model)

---

## 1. Current state (honest inventory)

The Cross-Engine Parity Specification rates each engine's status. OpenSim is
flagged **"Greenfield"** — every row red. This section documents exactly
what exists today, so the implementation plan in §4 can build on it
truthfully.

### 1.1 Source files

```
src/engines/physics_engines/opensim/
├── README.md                                # marketing/overview only
├── __init__.py
├── _tier.py                                 # tier classification stub
└── python/
    ├── __init__.py                          # empty
    ├── opensim_physics_engine.py            # 796 LOC — load/init wrapper only
    ├── opensim_golf/
    │   ├── __init__.py                      # empty
    │   └── core.py                          # 248 LOC — placeholder GolfSwingModel
    ├── opensim_gui.py                       # 440 LOC — Tk GUI shell, not wired to a real model
    ├── muscle_analysis.py                   # 469 LOC — operates on a model if one is provided
    ├── opensim_screw_kinematics.py          # 165 LOC — kinematics helper, not used in pipeline
    ├── perturbation/analyzer.py             # 440 LOC — generic perturbation scaffolding
    └── tests/__init__.py                    # empty — NO tests
```

Total: 12 Python files, ~2,571 LOC. Of these, **0 LOC** are actually
exercised by the motion-matching pipeline today.

### 1.2 What `opensim_physics_engine.py` actually does

The 796-LOC `OpenSimPhysicsEngine` class is **load-and-introspect only**:

| Method                                                      | What it does                                                          | Relevant to motion-matching?                                              |
| ----------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `load_from_path(path)`                                      | `osim.Model(path)`; `model.initSystem()`; instantiates `osim.Manager` | Yes — needed                                                              |
| `load_from_string(content)`                                 | Writes to tempfile then `load_from_path`                              | Probably useful for tests                                                 |
| `model_name`, `is_initialized`                              | trivial getters                                                       | Trivial                                                                   |
| Inspection helpers (DOF count, body list, joint list, etc.) | walk the model                                                        | Useful for diagnostics                                                    |
| `step`, `set_state`, `get_state`                            | thin SimTK wrappers                                                   | **Not enough** — there is no torque-controller wiring or forward-sim loop |

There is **no `simulate_with_coefficients`**, **no controller class that
applies a polynomial torque profile**, **no `SimOutput`-shaped return**, and
**no integration with `shared/python/motion_matching/`**.

### 1.3 What `opensim_golf/core.py` is

`opensim_golf/core.py` (248 LOC) is a placeholder:

- A `GolfSwingModel(model_path)` class that requires a `.osim` file but
  fails fast with `OpenSimNotInstalledError` / `OpenSimModelLoadError` /
  `FileNotFoundError`.
- A `SimulationResult` dataclass with `(time, states, muscle_forces,
control_signals, joint_torques, marker_positions)` — close in spirit to
  the canonical `SimOutput` but **missing** `grip`, `grip_quat`, `clubhead`,
  `club_quat`, `solver_status`, `duration_s`.
- No actual simulation method. Imports `constants` but does not run
  anything.

### 1.4 What's in `shared/models/opensim/opensim-models/`

`shared/models/opensim/opensim-models` is registered in `.gitmodules` as a
**git submodule** pointing at `https://github.com/opensim-org/opensim-models.git`.
On a fresh clone the directory is empty until `git submodule update --init`.

Once initialised, the upstream repo provides 50+ reference `.osim` models:

| Family                        | Notable models                                  | Relevance                                                                   |
| ----------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| **Lower-extremity gait**      | `gait2392.osim`, `gait2354.osim`                | Older, well-validated 23-DOF lower-body + lumbar models — **good MVP base** |
| **Full-body musculoskeletal** | `Rajagopal2015.osim` (full body, 80 muscles)    | Modern peer-reviewed full-body model — **best long-term base**              |
| **Upper extremity**           | `arm26.osim`, `MoBL-ARMS` upper-extremity model | Useful reference for shoulder/elbow/wrist DOF naming                        |
| **Single-DOF demos**          | `arm26.osim`, `leg6dof9musc.osim`               | Useful for unit tests                                                       |

**There is no golf-specific humanoid in the submodule.** We must build one
by adapting an existing model (see §3) and we must respect upstream
licenses (see §7).

### 1.5 What's in the canonical reference (Simscape)

The Simscape implementation under
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/`
is the contractual oracle. It defines:

- `compute_skeleton_fk.m` — forward kinematics from `q` to grip / clubhead
- `compute_cost.m` — grip-primary cost (engine-agnostic)
- `synthesize_target_from_coefficients.m` — TDD oracle
- `default_sim_options.m`, `default_cost_options.m`, `default_synth_options.m`
- `align_measured_to_model.m` — time-grid alignment
- `load_club_target_excel.m`, `load_club_target_c3d.m` — target loaders
- `animate_trajectory_overlay.m`, `plot_error_timecourse.m`,
  `plot_fit_quality_card.m` — visualisation

OpenSim must match these contracts byte-for-byte at the schema level,
producing artefacts the existing cost function and leaderboard can consume.

---

## 2. Target architecture

This section enumerates **every file we need to create** under
`src/engines/physics_engines/opensim/`, with a one-line description.
File-size budgets are 1200 LOC max (per `CLAUDE.md`); split if you
approach the limit.

### 2.1 Directory layout (target)

```
src/engines/physics_engines/opensim/
├── README.md                               (existing — update to reflect MVP)
├── OPENSIM_PARITY_SPEC.md                  (this file)
├── OPENSIM_ISSUES.md                       (issue drafts; sibling deliverable)
├── models/
│   ├── golf_humanoid.osim                  (NEW — generated, not hand-edited)
│   ├── golf_humanoid_actuators.xml         (NEW — joint-torque actuator set)
│   └── README.md                           (NEW — provenance, license, regeneration command)
├── python/
│   ├── opensim_physics_engine.py           (existing — minor edits)
│   ├── opensim_golf/
│   │   ├── core.py                         (existing — refactor to canonical SimOutput)
│   │   ├── simulate_with_coefficients.py   (NEW — forward-sim wrapper)
│   │   ├── controller.py                   (NEW — torque polynomial controller)
│   │   ├── fk.py                           (NEW — grip + clubhead extraction)
│   │   ├── synthesize_target_from_coefficients.py  (NEW — TDD oracle)
│   │   ├── fit_swing_opensim.py            (NEW — scipy.optimize wrapper)
│   │   ├── default_options.py              (NEW — SimOptions / FitOptions)
│   │   └── visualize.py                    (NEW — three canonical figures)
│   └── tests/
│       ├── test_model_loads.py             (NEW)
│       ├── test_simulate_with_coefficients.py (NEW)
│       ├── test_synthesize_oracle.py       (NEW)
│       ├── test_fit_recovers_truth.py      (NEW)
│       ├── test_grip_fk_matches_simscape.py (NEW — equivalence test)
│       └── test_visualize_smoke.py         (NEW)
└── scripts/
    └── build_humanoid_osim.py              (NEW — generate .osim from shared YAML)
```

Net new code estimate: **8 source files + 1 builder script + 6 test
modules + 1 generated model = ~3,000 LOC of Python + ~50 KB of XML**.

### 2.2 Forward-sim wrapper: `simulate_with_coefficients.py`

Signature (must match cross-engine spec exactly):

```python
def simulate_with_coefficients(
    theta: np.ndarray,                  # (n_joints * 7,) torque polynomial coefficients
    options: SimOptions = ...,
    initial_pose: dict | None = None,   # StartPosition / StartVelocity overrides
) -> SimOutput:
    ...
```

Implementation strategy:

1. Lazy-load the `.osim` model the **first** time the function is called
   (re-use across invocations; OpenSim model creation is expensive).
2. Reset the `SimTK::State` to the provided `initial_pose` (or the
   "address" pose if `None`).
3. Wire a `PolynomialTorqueController` (see `controller.py`) onto every
   coordinate actuator. The controller evaluates
   `tau_j(t) = sum_{k=0..6} theta[7*j + k] * t**k` and writes into the
   model's `controls` vector each step.
4. Drive `osim.Manager.integrate(t_end)` over the canonical time grid in
   `target.time`. Sample the state at every `target.time[i]` (use OpenSim's
   `StatesTrajectoryReporter` or do per-step state caching).
5. After integration, call `fk.compute_grip_clubhead(model, states)` to
   produce `grip`, `grip_quat`, `clubhead`, `club_quat` per sample.
6. Pack into the canonical `SimOutput`. Set `solver_status = "success"` if
   the integrator returned without error, `"warning"` for tolerance
   misses, `"failed"` if it threw.

### 2.3 Controller: `controller.py`

OpenSim 4.x exposes a Python-extendable `osim.Controller` base class. The
key constraint: **the controller must be picklable** (so `scipy.optimize`
can fork it during multistart). Concretely:

- Implement `PolynomialTorqueController` as a thin Python subclass of
  `osim.Controller`.
- Override `computeControls(state, controls)` to evaluate the polynomial
  for every coordinate actuator.
- Store coefficients in a `numpy.ndarray` attribute (numpy arrays pickle
  cleanly).
- Provide `set_theta(theta)` and `get_theta()` accessors so the optimizer
  can update coefficients without re-instantiating the controller (cuts
  per-iteration overhead by ~10–100×).

**LOD note:** the controller exposes `tau_at(t, joint_idx)` as a public
method; tests can call this directly without touching the SimTK state.

### 2.4 Fit driver: `fit_swing_opensim.py`

Signature:

```python
def fit_swing_opensim(
    target: ClubTarget,
    options: FitOptions = ...,
) -> FitResult:
    ...
```

MVP optimizer: **`scipy.optimize.minimize(method="L-BFGS-B")`** with
finite-difference gradients. Rationale:

- Starts simple. The Simscape reference uses `fmincon`; SciPy L-BFGS-B is
  the closest like-for-like.
- Gradient-based via finite differences is acceptable when forward-sim
  cost dominates (it does, by ~10×).
- Replaceable later with `MultiStart`-equivalent (issue
  `OPENSIM-FIT-MULTISTART`) and CMC for the muscle path.

The driver imports `compute_cost` from
`shared/python/motion_matching/cost.py` — **never** writes its own cost
function (per cross-engine spec §2.3).

### 2.5 TDD oracle: `synthesize_target_from_coefficients.py`

```python
def synthesize_target_from_coefficients(
    theta: np.ndarray,
    options: SynthOptions = ...,
) -> ClubTarget:
    """Synthesize a ClubTarget by forward-simulating theta and packaging
    the resulting grip + clubhead trajectories as a target."""
```

This is the linchpin of the TDD strategy: with a synthesizer, the very
first test you can write is the recovery test "synthesize → fit → assert
`theta_recovered ≈ theta_truth`". No external data needed.

### 2.6 Visualisation: `visualize.py`

Per the cross-engine spec §2.5, three canonical figures:

1. **Trajectory overlay** — measured vs simulated club skeleton + grip path
2. **Error timecourse** — grip position/orientation error, clubhead speed,
   joint torques
3. **Fit-quality card** — single-figure summary for PRs

For figures (2) and (3), reuse the shared Python plotters at
`shared/python/motion_matching/plot_*.py`. Only the 3D viewer is
engine-specific:

- For headless CI: matplotlib 3D renderer (consistent with Simscape /
  MuJoCo).
- For local interactive use: `opensim.Visualizer` (the SimTK Java/OpenGL
  viewer that ships with OpenSim 4.x). This is **separate** from the
  full GUI in `opensim_gui.py`.

### 2.7 Engine availability

`src/shared/python/engine_core/engine_availability.py` already exposes
`OPENSIM_AVAILABLE`. The wrapper sets it to `True` only when
`import opensim` succeeds. Every public function in this module must
guard against `OPENSIM_AVAILABLE is False` and raise
`OpenSimNotInstalledError` (already defined in `opensim_golf/core.py`).
**No demo/fallback mode** — consistent with the existing comment in
`core.py`.

---

## 3. Body model

### 3.1 Decision: start from `Rajagopal2015.osim`

The Cross-Engine Parity Spec §2.6 calls for a **25-DOF skeleton (6
floating-base + 19 actuated rotational)** matching the Simscape model;
the canonical totals live in `shared/models/golf_humanoid_topology.yaml`
(PR #4150). OpenSim has two candidate base models in its submodule:

| Candidate            | Pros                                                       | Cons                                                                 |
| -------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `gait2392.osim`      | Simpler (23 DOF), well-validated, smaller XML              | Lower-body + lumbar only — **need to graft a full upper-body chain** |
| `Rajagopal2015.osim` | Already full-body, peer-reviewed, modern joint definitions | Larger; muscles must be stripped for MVP                             |

**Decision: start from `Rajagopal2015.osim`** for the MVP, because the
upper-body chain (scapula → shoulder → elbow → wrist) is _already_
present and we'd otherwise be hand-authoring it. We strip muscles for
the MVP and add joint-torque actuators on every coordinate.

If license review (§7) blocks Rajagopal2015, fall back to
`gait2392.osim` + a hand-authored upper-body chain. The fallback adds
~1–2 days of work but is fully tractable; the upper body is a 14-DOF
chain we already define analytically in the Simscape model.

### 3.2 MVP transformation pipeline

The `.osim` file we ship is **generated**, not hand-edited. The
`scripts/build_humanoid_osim.py` script:

1. Reads the shared anthropometric YAML (`shared/models/golf_humanoid_dimensions.yaml`,
   tracked by issue `PARITY-DIMENSIONS` in the cross-engine spec).
2. Loads the chosen base `.osim` via the OpenSim Python API.
3. **Strips muscles**: for every `Muscle` and `Force` of muscle subtype,
   call `model.updForceSet().remove(...)`. Document the count removed.
4. **Adds joint-torque actuators**: for every `Coordinate` in the model,
   add a `CoordinateActuator` with `optimal_force = 1.0`,
   `min_control = -inf`, `max_control = +inf`. (We control torque
   directly in N·m via the polynomial controller.)
5. **Rescales segments** to match the shared YAML using
   `opensim.ModelScaler` or direct edits to `BodySet` mass /
   inertia / `PhysicalOffsetFrame` translations.
6. **Adds the club rigid body**: a `Body` named `Club` with mass + inertia
   from the YAML, attached via a `WeldConstraint` to a frame on the
   right hand (or, equivalently, by adding a fixed `WeldJoint` parent
   `right_hand` → child `Club`). The grip frame is the connection point;
   the clubhead frame is a `PhysicalOffsetFrame` on the `Club` body.
7. Validates by calling `model.initSystem()` and checks that the DOF count
   matches the Simscape reference (**25 generalized velocities = 6
   floating-base + 19 actuated rotational**, per
   `shared/models/golf_humanoid_topology.yaml`). Note: OpenSim flattens
   floating-base orientation to 3 Euler coordinates, so `nq == nv == 25`
   in OpenSim's representation (no quaternion offset).
8. Writes `models/golf_humanoid.osim` and `models/golf_humanoid_actuators.xml`
   with a deterministic output (no timestamps in XML).

### 3.3 OpenSim coordinate-convention mapping

OpenSim has **its own coordinate conventions** that differ from
Simscape's:

- **Frame orientation:** OpenSim uses Y-up by default; Simscape's
  motion-matching pipeline uses Z-up. We must apply a fixed rotation
  when reading/writing `grip` and `clubhead` between the two.
- **Joint angle sign conventions:** OpenSim's
  `ground_pelvis.pelvis_tilt` increases for forward tilt; Simscape's hip
  rotation matrix may differ. **The mapping must be table-driven** —
  see `python/opensim_golf/coordinate_map.py` (NEW) which provides
  `to_simscape(q_opensim)` and `from_simscape(q_simscape)` lookup
  helpers.
- **Quaternion convention:** OpenSim uses `[x, y, z, w]` (Eigen); the
  canonical `SimOutput` uses `[w, x, y, z]`. Convert at the SimOutput
  boundary.

This mapping work is its own issue (`OPENSIM-COORD-MAP`) and ships
alongside the model authoring issue.

### 3.4 Club rigid attachment

The club is rigidly fixed to the right hand. Two implementation options:

- **Option A (preferred):** `WeldJoint(parent=right_hand, child=Club)`.
  Treats the club as part of the rigid hand body. Zero extra DOFs.
- **Option B:** `Body Club` with a `WeldConstraint` to `right_hand`.
  Slightly slower to integrate (constraint enforcement) but easier to
  swap out for a `BushingForce` if we ever model grip flexibility.

MVP picks Option A (faster, simpler). Issue
`OPENSIM-CLUB-ATTACHMENT-FLEX` explores Option B post-MVP if grip
compliance becomes a research target.

---

## 4. Implementation plan (atomic, testable issues)

Each issue is sized for **one focused PR** (≤ 1200 LOC, including tests).
Dependencies form a strict DAG: model authoring is the root, every other
issue depends on it.

### Issue 1 — `OPENSIM-MODEL-AUTHORING`

**Title:** Author golf-humanoid `.osim` from Rajagopal2015 base + joint-torque actuators.

**Deliverable:**

- `scripts/build_humanoid_osim.py` (generator script).
- `models/golf_humanoid.osim` (committed, generated artifact).
- `models/golf_humanoid_actuators.xml` (separate actuator-set XML).
- `models/README.md` documenting provenance, regeneration, license.
- Initial `tests/test_model_loads.py`: model loads, has 25 DOF (6
  floating-base + 19 actuated rotational), has the expected coordinate
  names, has a `Club` body and a clubhead frame.

**Acceptance criteria:**

- `python scripts/build_humanoid_osim.py` is deterministic — running it
  twice produces byte-identical `.osim` output.
- `osim.Model("models/golf_humanoid.osim").initSystem()` succeeds.
- DOF count == 25 (6 floating-base + 19 actuated rotational), matching
  `shared/models/golf_humanoid_topology.yaml`.
- Every `Coordinate` has exactly one `CoordinateActuator`.
- Test runtime < 5 s.

**Estimated size:** 800 LOC (including the builder script and tests).

**Dependencies:** none — root issue.

---

### Issue 2 — `OPENSIM-COORD-MAP`

**Title:** OpenSim ↔ Simscape coordinate-convention mapping helper.

**Deliverable:**

- `python/opensim_golf/coordinate_map.py` with:
  - `to_simscape(q_opensim) -> q_simscape`
  - `from_simscape(q_simscape) -> q_opensim`
  - `quat_eigen_to_canonical(q_xyzw) -> q_wxyz` (and inverse)
  - `frame_y_up_to_z_up(v) -> v` (rotation matrix multiply)
- Tests: round-trip through `to_simscape ∘ from_simscape == identity`
  for 100 random poses; specific golden values for at-address pose.

**Acceptance criteria:**

- Round-trip identity within 1e-12 absolute tolerance.
- Golden test for at-address pose: spec the expected Simscape pose for
  the OpenSim default state and assert.
- Pure-Python (no `opensim` import) so tests run on every CI without the
  OpenSim wheel installed.

**Estimated size:** 250 LOC.

**Dependencies:** Issue 1 (uses the model's coordinate ordering).

---

### Issue 3 — `OPENSIM-FK`

**Title:** Forward-kinematics extraction: `(model, state) -> grip + clubhead`.

**Deliverable:**

- `python/opensim_golf/fk.py` with:
  - `compute_grip(model, state) -> (pos: np.ndarray(3,), quat: np.ndarray(4,))`
  - `compute_clubhead(model, state) -> (pos, quat)`
  - `compute_skeleton_fk(model, states) -> dict` (vectorised over a
    state trajectory).
- Tests: at-address pose grip position matches Simscape reference within
  5 mm; clubhead position matches within 5 mm.

**Acceptance criteria:**

- Equivalence test against Simscape `compute_skeleton_fk.m` at three
  poses (address, top-of-backswing, impact) — all within **5 mm grip
  RMSE**, **5 mm clubhead RMSE**.
- Vectorised path is at least 10× faster than per-step Python loop.

**Estimated size:** 400 LOC.

**Dependencies:** Issues 1 + 2.

---

### Issue 4 — `OPENSIM-SIMULATE`

**Title:** `simulate_with_coefficients` forward-sim wrapper + polynomial controller.

**Deliverable:**

- `python/opensim_golf/controller.py` — `PolynomialTorqueController`
  (subclass of `osim.Controller`) with `set_theta`, `get_theta`,
  `tau_at(t, j)`.
- `python/opensim_golf/simulate_with_coefficients.py` — top-level
  wrapper returning canonical `SimOutput`.
- `python/opensim_golf/default_options.py` — `SimOptions`,
  `SynthOptions`, `FitOptions` dataclasses mirroring the Simscape
  defaults.
- Refactor `opensim_golf/core.py` `SimulationResult` → import canonical
  `SimOutput` from `shared/python/motion_matching/cost.py`.
- Tests: zero-coefficient input → joint angles stay at initial pose;
  unit step → expected linear ramp in joint velocity.

**Acceptance criteria:**

- `simulate_with_coefficients(theta_zeros, default_options)` returns a
  `SimOutput` with `np.allclose(out.q, q_initial)` over the entire grid.
- Canonical `SimOutput` produced (correct field names, dtypes, shapes).
- Wall-clock for a 1.0 s sim ≤ 10 s on a developer laptop.
- `solver_status == "success"` for nominal inputs.

**Estimated size:** 900 LOC.

**Dependencies:** Issues 1 + 2 + 3.

---

### Issue 5 — `OPENSIM-SYNTH-ORACLE`

**Title:** TDD oracle `synthesize_target_from_coefficients`.

**Deliverable:**

- `python/opensim_golf/synthesize_target_from_coefficients.py`.
- Tests: synthesizer round-trips a known `theta` to a `ClubTarget` with
  finite, monotone-time-grid, expected schema fields populated.

**Acceptance criteria:**

- Output is a valid `ClubTarget` (time, grip, grip_quat, clubhead,
  club_quat, impact_idx, source).
- `source.format == "synthetic"` and `source.subject_id == "opensim"`
  (the existing `SourceProvenance` fields already encode engine
  identity — no schema change required).
- `theta_truth` is persisted alongside the returned `ClubTarget` (e.g.
  via the synthesizer also returning the truth vector, or by hashing
  it into `source.sha256`); recovery tests pass `theta_truth`
  explicitly to the fitter rather than reading it back off `source`.
- Reproducible: same `theta` → byte-identical `ClubTarget` (within
  floating-point rounding) on two runs.

**Estimated size:** 300 LOC.

**Dependencies:** Issue 4.

> **Note:** the canonical `SourceProvenance`
> (`src/shared/python/motion_matching/club_target.py`) currently exposes
> `filename`, `format`, `subject_id`, `trial_id`, `sha256`. Adding
> `engine` / `theta_truth` would be a cross-engine schema migration and
> is intentionally out of scope for this MVP issue.

---

### Issue 6 — `OPENSIM-FIT`

**Title:** `fit_swing_opensim` motion-matching driver.

**Deliverable:**

- `python/opensim_golf/fit_swing_opensim.py` using
  `scipy.optimize.minimize(method="L-BFGS-B")`.
- Imports `compute_cost` from `shared/python/motion_matching/cost.py`.
- Returns canonical `FitResult`.
- Tests: recovery test "synthesize → fit → `np.allclose(theta_fit,
theta_truth, atol=1e-2)`" passes on three randomly generated truths.

**Acceptance criteria:**

- Recovery test converges in < 60 s for a 1.0 s synthesised target on a
  dev laptop.
- `FitResult.solver_status == "success"`.
- Final cost < `1e-3` for noise-free synthesis-recovery cases.

**Estimated size:** 600 LOC.

**Dependencies:** Issues 4 + 5.

---

### Issue 7 — `OPENSIM-VIZ`

**Title:** Three canonical visualisation figures.

**Deliverable:**

- `python/opensim_golf/visualize.py` exporting:
  - `plot_trajectory_overlay(target, sim_out) -> Figure`
  - `plot_error_timecourse(target, sim_out) -> Figure`
  - `plot_fit_quality_card(fit_result) -> Figure`
- Reuse `shared/python/motion_matching/plot_*.py` where possible.
- Tests: smoke tests that each function returns a non-null `Figure`
  with expected axes count and no warnings.

**Acceptance criteria:**

- Three figures render headlessly (Agg backend) under pytest.
- Layout matches the Simscape reference figures within visual tolerance
  (snapshot test against committed PNGs, with 5% pixel tolerance).
- Smoke-test runtime < 5 s.

**Estimated size:** 500 LOC.

**Dependencies:** Issues 4 + 6.

---

### Issue 8 (cross-cutting) — `OPENSIM-EQUIVALENCE`

**Title:** Cross-engine equivalence test: OpenSim vs Simscape oracle.

**Deliverable:**

- `tests/cross_engine/test_opensim_vs_simscape.py` running a fixed
  `theta` through both engines (Simscape via the existing harness, or
  via a committed reference `.mat` if Simscape is unavailable on CI).
- Asserts grip RMSE ≤ 5 mm and clubhead RMSE ≤ 5 mm at three poses.

**Acceptance criteria:**

- CI passes with the equivalence assertions enabled.
- Test is marked `@pytest.mark.slow` and `@pytest.mark.requires_opensim`
  (gate appropriately).

**Estimated size:** 250 LOC.

**Dependencies:** all of Issues 1–6.

---

### Sequencing summary

```
Issue 1 (model) ──────────────────────────────────────────────► Issue 8 (equivalence)
       │                                                              ▲
       ├──► Issue 2 (coord-map) ──► Issue 3 (fk) ──► Issue 4 (sim) ───┤
                                                          │           │
                                                          ├──► Issue 5 (synth)
                                                          │           │
                                                          ├──► Issue 6 (fit) ──┤
                                                          │                    │
                                                          └──► Issue 7 (viz)   │
```

Recommended PR order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each PR is
mergeable independently; no PR depends on unmerged work from a later
issue.

---

## 5. TDD / DbC / DRY / LOD compliance

### 5.1 OpenSim's Python bindings — what to know

OpenSim's Python module is a **SWIG-generated wrapper around C++**.
Practical implications:

- **No type hints** in the upstream API. We add `.pyi` stubs (or use
  `typing.cast`) at every public boundary so `ruff` / `mypy` can lint
  our calls.
- **Reference semantics:** most methods return references into the
  C++ object graph. Mutating a `Body` via Python _does_ mutate the
  underlying model. Wrap defensively with explicit copies in the
  controller's `set_theta` path.
- **`initSystem()` discards changes.** Every model edit _after_
  `initSystem()` is silently ignored. Our builder script always
  finalises edits _before_ calling `initSystem()`.
- **Pickling:** OpenSim objects do **not** pickle. Multiprocess fitting
  (`scipy.optimize` + `pool`) must serialise the model via
  `model.printToXML(tempfile)` and reload in each worker.

### 5.2 Strategy for keeping the API surface narrow

Following the cross-engine spec §2 contracts and `CLAUDE.md` §"Coding
Standards":

- **One canonical entry point per role:** exactly one
  `simulate_with_coefficients`, one `fit_swing_opensim`, one
  `synthesize_target_from_coefficients`. No alternative signatures.
- **All `osim.*` calls live behind 4 modules:** `controller.py`,
  `fk.py`, `simulate_with_coefficients.py`, `build_humanoid_osim.py`.
  Anywhere else that imports `opensim` is a smell — fix or refactor.
- **DbC:** every public function uses
  `src.shared.python.core.contracts.{precondition, postcondition,
check_finite}` (already imported in `opensim_physics_engine.py`).
- **DRY:** the cost function, target loaders, and plot helpers are
  imported from `shared/python/motion_matching/`. **No engine writes its
  own cost.**
- **LOD:** no chains deeper than two levels. Replace
  `model.getJointSet().get(i).getCoordinate(0).getValue(state)` with a
  helper `coord_value(model, joint_idx, coord_idx, state)`.
- **TDD:** every PR includes its tests. The recovery test (Issue 6) is
  the canonical gate; if it doesn't converge, the PR is not ready.

### 5.3 File-size budget

The 1200-LOC budget per `CLAUDE.md` is not at risk for any single new
file in §4. The current `opensim_physics_engine.py` is 796 LOC and stays
within budget; if §4 work pushes it past 1200, split it into
`opensim_physics_engine.py` (load/init) + `opensim_physics_engine_introspect.py`
(inspection helpers).

---

## 6. Performance baseline + targets

OpenSim's Python API is a SWIG layer over a C++ multibody integrator.
Forward simulation cost is dominated by the integrator step, not Python
overhead — but the SWIG round-trip per controller-callback is real.

### 6.1 Measured baseline

We have no measurements yet (greenfield). Reference points:

- **Simscape Multibody** (cold, JIT-warm): ~30 s for a 1.0 s sim.
  ~7 s warm.
- **OpenSim** (anecdotal from forward-dynamics benchmarks):
  ~5–15 s for a 1.0 s sim of a 25-DOF human at 1 kHz integration
  step. **Same order of magnitude as Simscape.**

### 6.2 MVP target

- **Cold call (first sim after import):** ≤ 30 s. Includes model load
  - `initSystem()` (~10–15 s).
- **Warm call (subsequent sim, cached model + state):** ≤ 7 s for a
  1.0 s sim. **Equal to or faster than Simscape warm.**
- **Recovery test** (~50 cost-function evaluations × 1 s sim):
  ≤ 6 minutes wall-clock.

### 6.3 Optimization levers (post-MVP)

If MVP misses the 7 s warm target, in priority order:

1. Replace per-step Python controller callback with a pre-compiled
   `osim.PrescribedController` (evaluates a piecewise polynomial in
   C++) — typically 2–3× speedup.
2. Loosen integrator tolerances (default `1e-5` → `1e-3`) — typically
   2× speedup with negligible accuracy loss for joint-torque sims.
3. Switch integrator from `RungeKuttaMerson` to `SemiExplicitEuler`
   for prototype iterations — 3–5× speedup, but accuracy degrades.
4. Pre-cache the muscleless model XML so workers skip the 5+ s
   `initSystem()` reload.

---

## 7. Risks and open questions

### 7.1 Python OpenSim install

- **Channel:** `pip install opensim` works on Linux + Windows for
  OpenSim 4.4+. macOS is `conda install -c opensim-org opensim`. Pin
  the version in `pyproject.toml` extras (`opensim = ">=4.4,<5"`).
- **CI:** add an "opensim-extras" CI job that installs the package and
  runs `tests/` under `@pytest.mark.requires_opensim`. Default CI must
  pass without OpenSim installed (existing `OPENSIM_AVAILABLE` guard).
- **Wheel size:** the OpenSim wheel is ~150 MB. Cache aggressively in
  CI; gate the heavy job behind a label or schedule it nightly.

### 7.2 License of shared `.osim` reference files

- The `opensim-models` submodule is **Apache 2.0** at the repo level.
- **Individual models may have non-commercial restrictions.** In
  particular `Rajagopal2015.osim` is published under the original
  paper's terms, which historically include a request to cite (not a
  hard restriction) but **must be confirmed before commit**.
- **Action item:** before Issue 1 lands, file a license-review
  micro-task that:
  1. Reads the per-model headers in `Rajagopal2015.osim` and
     `gait2392.osim`.
  2. Reads `opensim-models/LICENSE` and any `README.md` under each
     model directory.
  3. Confirms redistribution-with-modification is permitted.
  4. Records the provenance in `models/README.md`.
- **Mitigation:** if Rajagopal2015 turns out to be CC-BY-NC or
  similarly restrictive, fall back to `gait2392.osim` (Apache-style
  permissive in the upstream repo) and hand-author the upper-body
  chain.

### 7.3 OpenSim Python ↔ multiprocessing

- OpenSim objects don't pickle. `scipy.optimize.minimize` is
  single-process by default, so MVP is unaffected. Multistart fitting
  (post-MVP) needs the XML-print-and-reload pattern in §5.1.

### 7.4 Headless rendering

- `opensim.Visualizer` uses OpenGL and requires a display. CI must use
  matplotlib/Agg for visualisation snapshot tests (per Issue 7's
  acceptance criteria).

### 7.5 The muscle-level path is post-MVP

This must be repeated explicitly: the MVP is **joint-torque actuators
only**. Muscle-level forward dynamics (the headline feature that
justifies OpenSim being one of the five engines) is sequenced after
MVP — see §8. Anyone tempted to start with muscles must first ship the
joint-torque path; otherwise we'll spend months tuning muscle
parameters before the first end-to-end test passes.

### 7.6 Open questions (to resolve as part of Issue 1)

- Which Rajagopal2015 variant to use? The submodule may contain
  multiple (full body, lower body only, scaled vs nominal).
- Does the model need a separate "ball" body for impact modeling?
  Cross-engine spec §2.6 is silent on this; MVP says no (impact is
  defined by `target.impact_idx` in time, not by contact geometry).
- Where does the YAML build script live, given that it's shared across
  five engines? Likely `scripts/build_humanoid_models.py` per the
  cross-engine spec §4 milestone `PARITY-MODEL-BUILD`. Coordinate with
  the Drake / Pinocchio specs to avoid duplication.

---

## 8. Post-MVP: muscle-level body model

Once §4's MVP is merged and the cross-engine equivalence test is green,
the **next milestone** is muscle-level forward dynamics. This is the
"killer feature" that justifies OpenSim being one of the five engines —
none of MuJoCo, Drake, or Pinocchio offer peer-reviewed muscle models
of the kind biomechanists trust.

The path:

1. **Restore muscles in the body model.** Re-run
   `build_humanoid_osim.py` with `--keep-muscles`; the resulting
   `golf_humanoid_muscles.osim` retains Rajagopal2015's 80-muscle
   actuator set.
2. **Add a body-marker dataset.** Outside the scope of this engine —
   tracked by the upstream "body-marker mocap" milestone. The marker
   set must include the canonical Vicon Plug-In Gait labels.
3. **Run Computed Muscle Control (CMC).** OpenSim's CMC tool
   (`osim.CMCTool`) solves the muscle-redundancy problem: given
   measured kinematics + external forces, find the muscle-activation
   trajectory `a(t)` that produces the observed motion subject to
   physiological force-length-velocity constraints.
4. **Replace `fit_swing_opensim`'s torque-coefficient parameterisation
   with a muscle-activation parameterisation.** The cost function is
   unchanged — it's still grip-primary — but `theta` now indexes into
   muscle excitations rather than joint torques.
5. **Validate against EMG ground truth** (when available). The peer-
   reviewed credibility of the resulting muscle activations is the
   research deliverable.

This is a multi-quarter research effort and is **explicitly out of
scope for the MVP**. It is included here so the architectural choices
(strip-then-restore muscles, joint-torque actuators added without
displacing the muscle force set, table-driven coordinate mapping) all
keep the muscle path open. **Don't make MVP decisions that would have
to be reversed when muscles come back.**

---

_Spec landed 2026-05-06 alongside the Drake / Pinocchio / MuJoCo parity
specs. Tracks the eight implementation issues enumerated in
[`OPENSIM_ISSUES.md`](OPENSIM_ISSUES.md)._
