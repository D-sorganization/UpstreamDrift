# Cross-Engine Parity Specification

The 3D Golf Model project simulates and motion-matches a golf swing on five
physics engines. Today only the **Simscape Multibody** implementation is at
production grade (see [Simscape_Multibody_Models/3D_Golf_Model/PROJECT_SPEC.md](Simscape_Multibody_Models/3D_Golf_Model/PROJECT_SPEC.md)).
This document defines the **contracts every engine must implement** so that
all five produce comparable results on the same measured swing, share the
same loaders + cost function + visualisation, and are interchangeable behind
a single `fit_swing(target, engine, options)` call.

> **Audience:** every contributor (human or agent) implementing or modifying
> a physics-engine wrapper under `src/engines/physics_engines/`.
>
> **Status:** spec landed; per-engine implementations tracked in the issues
> linked at the bottom of each section.

---

## 1. The five engines and their roles

| Engine                 | Path                                                   | License     | Strengths                                                                                             | Why we have it                                                       |
| ---------------------- | ------------------------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Simscape Multibody** | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/` | MATLAB Home | Primary path. Full Stateflow torque polynomial, integrated Simulink toolchain                         | Reference implementation; ground-truth oracle for every other engine |
| **MuJoCo**             | `src/engines/physics_engines/mujoco/`                  | Apache-2.0  | Speed (orders of magnitude faster than Simscape), MJCF model authoring, GPU-friendly contact dynamics | Production training datasets + real-time inference                   |
| **Drake**              | `src/engines/physics_engines/drake/`                   | BSD-3       | Rigorous multibody dynamics, automatic-differentiation gradients, mathematical-program optimization   | Gradient-based fitting; verifiable optimization                      |
| **Pinocchio**          | `src/engines/physics_engines/pinocchio/`               | BSD-2       | Fastest articulated-body algorithms in robotics; analytical Jacobians and Hessians                    | Online inference, embedded deployment, real-time analysis            |
| **OpenSim**            | `src/engines/physics_engines/opensim/`                 | Apache-2.0  | Biomechanics-grade muscle models, peer-reviewed model library, clinical credibility                   | Muscle-level analysis (post-MVP); body-marker IK                     |

Each engine consumes the **same** target struct + cost function + visualisation
contracts. The differences are in the forward simulator and (eventually) the
optimizer.

---

## 2. Parity contracts (the things every engine must implement)

### 2.1 Target schema (immutable across engines)

Every engine reads a `ClubTarget` struct (Python) / `target` struct (MATLAB)
**unchanged** from the canonical schema in
[3D_Golf_Model/matlab/motion_matching/shared/CLUB_IK_SPEC.md](Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CLUB_IK_SPEC.md):

```python
@dataclass(frozen=True)
class ClubTarget:
    time:       np.ndarray   # (N,)   simulation timegrid in seconds
    grip:       np.ndarray   # (N,3)  PRIMARY anchor — mid-hands position (m)
    grip_quat:  np.ndarray   # (N,4)  PRIMARY anchor — mid-hands orientation [w,x,y,z]
    clubhead:   np.ndarray   # (N,3)  secondary
    club_quat:  np.ndarray   # (N,4)  secondary
    impact_idx: int
    events:     dict | None  # A/T/I/F samples + CHS_mph
    source:     SourceProvenance
```

**Engine-specific loaders are forbidden.** Use the canonical Python
loader in `shared/python/motion_matching/load_club_target.py` (which mirrors
the MATLAB `load_club_target_excel.m`). Issue #PARITY-LOADERS tracks the
promotion of this code to a shared package every engine imports from.

### 2.2 Forward-sim wrapper

Every engine ships a function with this signature:

```python
def simulate_with_coefficients(
    theta: np.ndarray,            # (n_joints * 7,) polynomial torque coefficients
    options: SimOptions = ...,    # engine-agnostic options struct
    initial_pose: dict | None = None,   # *StartPosition* / *StartVelocity* overrides
) -> SimOut:
    ...
```

Returning the canonical `SimOut`:

```python
@dataclass(frozen=True)
class SimOut:
    time:       np.ndarray   # (N,)
    q:          np.ndarray   # (N, n_joints)   joint angles (rad)
    qd:         np.ndarray   # (N, n_joints)   angular velocities (rad/s)
    qdd:        np.ndarray   # (N, n_joints)   angular accelerations
    tau:        np.ndarray   # (N, n_joints)   joint torques (N·m)
    grip:       np.ndarray   # (N,3)  PRIMARY world position
    grip_quat:  np.ndarray   # (N,4)  PRIMARY world orientation
    clubhead:   np.ndarray   # (N,3)  secondary world position
    club_quat:  np.ndarray   # (N,4)  secondary world orientation
    solver_status: str       # "success" | "warning" | "failed"
    duration_s: float        # wall-clock for instrumentation
```

Equivalence test: every engine must round-trip a fixed `theta` to within
**5 mm grip-position RMSE vs the Simscape reference** at three test poses
(impact, top-of-backswing, address). Tracked by issue
#PARITY-EQUIVALENCE-TEST.

Engine-specific target-synthesis adapters that wrap `simulate_with_coefficients`
must preserve this pose control. MuJoCo exposes `initial_pose` as a keyword-only
argument on `synthesize_target_from_coefficients` and passes it through unchanged
to the forward-sim wrapper.

### 2.3 Cost function (already shared)

The Python cost function at `shared/python/motion_matching/cost.py` mirrors
the MATLAB `compute_cost.m` numerically. It is engine-agnostic — every
engine's fit driver imports it directly. The cost is grip-primary; see
[3D_Golf_Model/matlab/motion_matching/shared/COST_FUNCTION_SPEC.md](Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/COST_FUNCTION_SPEC.md).
**No engine writes its own cost function.**

### 2.4 Fit driver

Every engine ships at least one of:

```python
def fit_swing_<engine>(
    target: ClubTarget,
    options: FitOptions = ...,
) -> FitResult:
    ...
```

Where `FitResult` matches the canonical schema in
[3D_Golf_Model/matlab/motion_matching/shared/CODING_STANDARDS.md](Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CODING_STANDARDS.md). The optimizer is
engine-specific (scipy.optimize.minimize, Drake MathematicalProgram,
Pinocchio + nlopt, OpenSim StaticOptimization, etc.) but the inputs and
outputs are identical.

### 2.5 Visualisation

Three views per [VISUALIZATION_SPEC.md](Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/VISUALIZATION_SPEC.md):

1. **Trajectory overlay** — measured vs simulated club skeleton + grip path
2. **Error timecourse** — grip position/orientation error, clubhead speed, joint torques
3. **Fit quality card** — single-figure summary for PRs

Engines provide a thin renderer that consumes the canonical `FitResult` +
`ClubTarget` and emits these three figures. Renderers should reuse the
shared Python plotters (`shared/python/motion_matching/plot_*.py`); only
engine-specific 3D viewers (Drake Visualizer, MuJoCo Viewer, OpenSim's GUI,
Meshcat for Pinocchio) need engine-bespoke code.

### 2.6 Body model (humanoid + club)

Every engine has **a single canonical full-body humanoid model** with the
club rigidly attached at the grip via a 6-DOF locked joint (or the
engine-native equivalent). Anatomical conventions:

- Skeleton parity with the Simscape model: **25 generalized velocities
  (6 floating-base + 19 actuated rotational DOFs)** distributed across the
  Hip(6) → Spine(2) → Torso(1) → Scapula(2)+(2) → Shoulder(3)+(3) →
  Elbow(1)+(1) → Wrist(2)+(2) → Hand(rigid) + Club(rigid) chain.
  - **q vs v note:** for floating-base systems the configuration vector `q`
    has **7 elements per floating base** (3 position + 4 quaternion) while
    the velocity vector `v` has **6** (3 linear + 3 angular). Therefore the
    canonical totals are **26 q-positions and 25 v-velocities**
    (= 7 + 19 vs 6 + 19). Engines that flatten the quaternion to a 3-DOF
    Euler triple (e.g. Drake's `RollPitchYawFloatingJoint`) report 25 for
    both `q` and `v`.
  - **Source of truth:** `shared/models/golf_humanoid_topology.yaml`
    (PR #4150 — PARITY-DIMENSIONS). All five engine spec docs derive their
    DOF counts from this file; any discrepancy is a bug in the engine
    spec, not in the topology YAML.
- Segment lengths come from the **same model-workspace constants** the
  Simscape model uses, exposed via a shared YAML at
  `shared/models/golf_humanoid_dimensions.yaml` (issue #PARITY-DIMENSIONS).
- Inertia parameters from the same anthropometric tables (Dempster +
  de Leva); shared YAML at `shared/models/golf_humanoid_inertia.yaml`.

Engine-native files (URDF/MJCF/.osim) are **generated** from the shared YAMLs
by `scripts/build_humanoid_models.py` so they stay in sync. Hand-edited
engine files are forbidden.

### 2.7 Test suite (TDD enforced)

Per [CODING_STANDARDS.md](Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CODING_STANDARDS.md):

- **TDD**: every public function has a test in the same PR, written first.
  No "tests in a follow-up" PRs.
- **DbC**: preconditions on the `arguments` block (or pydantic / dataclass
  validators in Python); postconditions as `assert` at the end of every
  function.
- **DRY**: no duplicated logic blocks > 5 lines across engines. Promote to
  `shared/python/motion_matching/`.
- **LOD**: no method chains deeper than 2 levels.
- **TDD oracle**: every engine implements
  `synthesize_target_from_coefficients(theta) -> ClubTarget` so that the
  recovery test "synthesize → fit → check theta_recovered ≈ theta_truth"
  is the first test you can write.

### 2.8 Cross-engine leaderboard

Every engine's fit driver writes its `FitResult` to
`results/<trial>/<engine>.json` and the leaderboard helper at
`shared/python/motion_matching/leaderboard.py` aggregates them into a
comparison table. Issue #PARITY-LEADERBOARD wires this up.

### 2.9 Adapter conformance merge gate

Every adapter that touches engine I/O must pass the CC-7 conformance harness
before merge:

```bash
python -m pytest tests/integration/cross_engine -q
```

The harness extends
`src/shared/python/engine_core/cross_engine_validator.py` and runs five
canonical-v2 checks against each registered adapter or stub:

1. `round_trip_state_remap` verifies
   `from_canonical(to_canonical(q)) == q` to `1e-9` across rigid and
   quaternion DOFs.
2. `forward_kinematics_reference_pose` compares FK against known address,
   top-of-backswing, and impact reference poses using the 5 mm end-effector
   tolerance.
3. `inverse_forward_dynamics_consistency` checks that inverse-dynamics torques
   reproduce the requested acceleration through forward dynamics.
4. `post_export_mass_properties` checks post-export mass, CoM, and inertia
   against canonical CC-3 properties.
5. `differential_cross_engine_reference` compares perturbed outputs against the
   selected reference engine within tolerance.

Capability-aware skips are allowed only when the adapter's declared capability
set does not include the tested surface, for example a rigid-only engine
skipping muscle-specific checks. Numerical differences caused by legitimate
solver or contact-model behavior must be recorded in
`tests/integration/cross_engine/divergence_registry.yaml` with engine pair,
check, metric, tolerance, and rationale. An unregistered divergence is a hard
failure.

---

## 3. Per-engine target architecture (summary)

Detailed implementation plans live in the per-engine spec docs:

- [`physics_engines/mujoco/MUJOCO_PARITY_SPEC.md`](physics_engines/mujoco/MUJOCO_PARITY_SPEC.md)
- [`physics_engines/drake/DRAKE_PARITY_SPEC.md`](physics_engines/drake/DRAKE_PARITY_SPEC.md)
- [`physics_engines/pinocchio/PINOCCHIO_PARITY_SPEC.md`](physics_engines/pinocchio/PINOCCHIO_PARITY_SPEC.md)
- [`physics_engines/opensim/OPENSIM_PARITY_SPEC.md`](physics_engines/opensim/OPENSIM_PARITY_SPEC.md)

The summary table:

| Engine    | Humanoid model               | `simulate_with_coefficients` | `fit_swing_*`                        | Visualisation                 | Tests      | Status                |
| --------- | ---------------------------- | ---------------------------- | ------------------------------------ | ----------------------------- | ---------- | --------------------- |
| Simscape  | ✅ `.slx`                    | ✅                           | ✅ fmincon/multistart/hybrid         | ✅ MATLAB plots               | ✅ 17/17   | **Production**        |
| MuJoCo    | ✅ MJCF (3 variants + myo\*) | 🟡 inline (needs decoupling) | 🟡 inline                            | ✅ Mujoco viewer + custom GUI | 🟡 partial | **Closest to parity** |
| Pinocchio | ✅ URDF (`golfer.urdf`)      | 🟡 inside `motion_training/` | 🟡 `torque_fitting.py`               | 🟡 Meshcat stubs              | 🟡 partial | **Second**            |
| Drake     | 🔴 missing                   | 🔴 missing                   | 🟡 `motion_optimization.py` skeleton | 🟡 stubs                      | 🔴 none    | **Big gap**           |
| OpenSim   | 🔴 missing                   | 🔴 missing                   | 🔴 missing                           | 🔴 missing                    | 🔴 none    | **Greenfield**        |

---

## 4. Cross-cutting milestones

| ID                     | Title                               | Description                                                                                                                                                                                          |
| ---------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PARITY-DIMENSIONS**  | Shared anthropometric YAML          | Pull segment lengths + masses + inertias from the Simscape model workspace into `shared/models/golf_humanoid_dimensions.yaml` and `…_inertia.yaml`. Single source of truth.                          |
| **PARITY-MODEL-BUILD** | `build_humanoid_models.py`          | Generate engine-native URDF/MJCF/.osim from the shared YAML. Run as a CI step so model files never drift.                                                                                            |
| **PARITY-LOADERS**     | Promote shared loaders to top level | Move/refactor `shared/python/motion_matching/cost.py` and add `load_club_target.py`, `align_to_simulation_grid.py`, `synthesize_target_from_coefficients.py` so every engine imports from one place. |
| **PARITY-EQUIVALENCE** | Cross-engine equivalence test       | Fixed `theta` → 5 engines → grip RMSE vs Simscape ≤ 5 mm at three poses. CI gate.                                                                                                                    |
| **PARITY-LEADERBOARD** | Cross-engine leaderboard            | Run all 5 engines on every test trial; emit `LEADERBOARD.md`.                                                                                                                                        |
| **PARITY-DOCS**        | Cross-engine docs currency          | Extend the docs-currency lint to every engine subtree.                                                                                                                                               |

---

## 5. Why this matters

> "We want full humanoid models implemented and we need club traces and
> comparisons to the desired trajectories in here as well." — project owner

Specifically:

1. **MuJoCo** gives us 100–1000× faster simulations than Simscape. Critical
   for population-scale optimization (multistart, evolutionary, RL) and
   real-time inference.
2. **Drake** gives us auto-differentiation through the dynamics, which
   enables gradient-based fitting that converges in seconds rather than
   minutes.
3. **Pinocchio** gives us analytical Jacobians/Hessians that make
   second-order optimizers (Newton, Levenberg-Marquardt) tractable. Useful
   for high-precision fits at the end of the pipeline.
4. **OpenSim** gives us muscle-level forward dynamics. Once body-marker
   data is available, we can solve for muscle activations rather than just
   joint torques — the next research frontier for this project.
5. **Five-way agreement** is the strongest evidence we have the right
   answer. If all five engines, with independent solvers and (semi-)
   independent body model implementations, converge to the same `theta`
   for a given target, the result is robust.

---

## 6. Working agreements

Same as PROJECT_SPEC.md plus:

- **No engine-specific cost or loader code.** If you find yourself needing
  it, you've found a missing abstraction in `shared/python/motion_matching/`
  — file an issue and lift the abstraction first.
- **Engine assets are generated**, not hand-edited. If you need to
  customize an engine model, edit the shared YAML and regenerate.
- **Every engine PR includes the equivalence test result** in its body.
- **Cross-engine PRs come in stages**: model first, simulator second, fit
  driver third, visualisation last. Don't conflate them.

---

_Last updated 2026-05-06 alongside the cross-engine parity issue wave._
