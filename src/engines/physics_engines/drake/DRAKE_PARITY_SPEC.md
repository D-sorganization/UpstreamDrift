# Drake Parity Specification

This document defines the work required to bring the Drake-backed physics
engine to parity with the production **Simscape Multibody** motion-matching
pipeline, per the contracts laid out in
[../CROSS_ENGINE_PARITY_SPEC.md](../../CROSS_ENGINE_PARITY_SPEC.md).

> **Status:** infrastructure-rich, model-poor. Drake's `MultibodyPlant`
> wrapper, dynamics queries (`compute_mass_matrix`, ZTCF/ZVCF), and a
> `DrakeMotionOptimizer` skeleton already exist. **There is no humanoid
> URDF, no `simulate_with_coefficients`, no `fit_swing_drake`, no canonical
> `ClubTarget`-driven plumbing, and no equivalence test.** This is the
> "biggest scaffolding gap" engine in the parity matrix.
>
> **Audience:** the agent or contributor implementing the Drake parity
> issues defined in [DRAKE_ISSUES.md](DRAKE_ISSUES.md).
>
> **Scope:** engine-specific work only. Anything cross-cutting
> (`shared/python/motion_matching/cost.py`, the canonical
> `synthesize_target_from_coefficients`, the leaderboard, the shared
> humanoid YAML at `shared/models/golf_humanoid_dimensions.yaml`) lives in
> the cross-engine spec and is referenced — never duplicated — here.

---

## 1. Current state

### 1.1 Inventory of `src/engines/physics_engines/drake/`

| File                                                                           |   LOC | Status  | Notes                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------ | ----: | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `python/drake_physics_engine.py`                                               |   641 | Working | `PhysicsEngine` protocol implementation. Hosts `DiagramBuilder`/`MultibodyPlant`, lazy `_ensure_finalized`, persistent `Simulator`, plus `compute_mass_matrix`, `compute_bias_forces`, `compute_gravity_forces`, `compute_inverse_dynamics`, `compute_jacobian`, ZTCF/ZVCF (Section F drift/control decomposition). Uses explicit `from pydrake.X import Y` per CLAUDE.md.                                         |
| `python/motion_optimization.py`                                                |   493 | Stubbed | `DrakeMotionOptimizer` with `OptimizationObjective` / `OptimizationConstraint` / `OptimizationResult` dataclasses. Wraps `scipy.optimize.minimize(SLSQP)`. **Cost functions are placeholders** (see §1.3); does not consume `ClubTarget` or call `simulate_with_coefficients`.                                                                                                                                     |
| `python/src/drake_golf_model.py`                                               |   828 | Partial | `GolfURDFGenerator` with `add_link` / `add_joint` / `_add_pelvis` / `_add_spine` / `_add_torso` / `_add_arms` / `_add_club` private builders. **Generates an in-process URDF tree from `GolfModelParams` dataclass** but the parameters (`pelvis_to_shoulders=0.35`, `spine_mass=15.0`, etc.) are **hard-coded defaults in the dataclass**, not pulled from the shared YAML. No URDF file is ever written to disk. |
| `python/swing_plane_integration.py`                                            |   319 | Working | `SwingPlaneIntegrator` for swing-plane analysis. Independent of motion-matching pipeline.                                                                                                                                                                                                                                                                                                                          |
| `python/src/drake_visualizer.py`                                               |   196 | Working | Thin Meshcat wrapper (frame axes, COM markers). Reusable.                                                                                                                                                                                                                                                                                                                                                          |
| `python/src/drake_gui_app.py`                                                  |   425 | Stub    | Tkinter GUI shell. Not on the parity critical path.                                                                                                                                                                                                                                                                                                                                                                |
| `python/src/drake_gui_viz.py`                                                  |   430 | Stub    | Companion to `drake_gui_app.py`.                                                                                                                                                                                                                                                                                                                                                                                   |
| `python/src/drake_gui_*.py` (×4 more)                                          |  ~600 | Stubs   | Tkinter GUI tabs. Not on critical path.                                                                                                                                                                                                                                                                                                                                                                            |
| `python/src/drake_recorder.py`, `induced_acceleration.py`, `manipulability.py` |  ~600 | Working | Auxiliary analysis utilities. Reusable for §6 visualisation.                                                                                                                                                                                                                                                                                                                                                       |
| `python/src/spatial_algebra/*.py`                                              |  ~600 | Working | In-house SE(3) helpers (predates pydrake usage in many places). Not blocking parity work.                                                                                                                                                                                                                                                                                                                          |
| `python/src/rigid_body_dynamics/__init__.py`                                   | small | Stub    | Empty package.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `python/tests/__init__.py`                                                     |     1 | Empty   | **No motion-matching tests.**                                                                                                                                                                                                                                                                                                                                                                                      |

### 1.2 What works vs what is stubbed

**Works today:**

- A full `PhysicsEngine` protocol implementation that round-trips a Drake
  `MultibodyPlant` (load URDF → set state → step → query mass matrix /
  bias forces / Jacobian).
- The `GolfURDFGenerator` can emit a valid (if anatomically rough)
  URDF in-memory.
- ZTCF / ZVCF counterfactual queries (used by Section F drift analysis;
  not yet wired to the parity cost).

**Stubbed / missing:**

- **No on-disk URDF.** The generator has never been invoked from a build
  script; the GUI tabs use it ad-hoc.
- **No `ClubTarget` consumption.** Nothing in this subtree imports
  `src.shared.python.motion_matching.club_target.ClubTarget`.
- **No `simulate_with_coefficients(theta)`.** `motion_optimization.py`
  takes a generic "trajectory" matrix, not a coefficient vector; no
  Stateflow-equivalent torque polynomial evaluator exists.
- **No `fit_swing_drake`.** No driver function with the signature
  required by §2.4 of the cross-engine spec.
- **No `synthesize_target_from_coefficients`.** The TDD oracle does not
  exist for Drake.
- **No equivalence test.** No test compares a fixed `theta` against the
  Simscape reference grip RMSE.
- **No cost-function call-through.** `DrakeMotionOptimizer` defines
  its own `ball_speed_cost` / `accuracy_cost` / `smoothness_cost`
  inline, violating §6 ("No engine-specific cost or loader code").

### 1.3 Reusable pieces from `motion_optimization.py`

The current `DrakeMotionOptimizer` is a generic NLP wrapper; the parity
work should **keep its scaffolding and replace its semantics**:

| Symbol                                                  | Reuse plan                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OptimizationObjective` dataclass                       | Keep — repurpose as the wrapper around `shared/python/motion_matching/cost.py` (one objective: `weighted_grip_cost`).                                                                                                                                                                                         |
| `OptimizationConstraint` dataclass                      | Keep — used for joint-limit and impact-timing constraints in `fit_swing_drake_constrained.py`.                                                                                                                                                                                                                |
| `OptimizationResult` dataclass                          | **Replace** with the canonical `FitResult` from `shared/python/motion_matching/...` (see §2 below). The current shape (`optimal_trajectory`, `optimal_cost`, `objective_values`) is engine-specific; the canonical schema has `theta_optimal`, `final_rmse_m`, `solver_status`, `iterations`, `wall_clock_s`. |
| `_build_total_cost_function`                            | Delete — replaced by a closure that calls `compute_cost_drake.py` (see §2.5).                                                                                                                                                                                                                                 |
| `_build_scipy_constraints`                              | Keep — generalize to also build `MathematicalProgram` constraints.                                                                                                                                                                                                                                            |
| `optimize_trajectory(initial_trajectory)`               | **Replace** with `fit_swing_drake(target: ClubTarget, options: FitOptions)` that decision-variables on `theta` (length `n_joints * 7`), not on a flat trajectory matrix.                                                                                                                                      |
| `setup_standard_golf_*` (placeholder costs/constraints) | Delete — these violate the cross-engine "no engine-specific cost" rule.                                                                                                                                                                                                                                       |
| `optimize_for_distance` / `optimize_for_accuracy`       | Delete — out of parity scope. The cross-engine spec mandates a single `fit_swing_<engine>` entry point.                                                                                                                                                                                                       |

The dataclasses are good abstractions; the placeholder cost bodies go.

---

## 2. Target architecture

The Drake subtree gains the following new files (alongside the existing
`drake_physics_engine.py`). All paths are relative to
`src/engines/physics_engines/drake/python/`.

```
python/
├── drake_physics_engine.py            (existing — unchanged)
├── motion_matching/                   (NEW package)
│   ├── __init__.py
│   ├── humanoid_urdf.py               (NEW — §3, the biggest piece)
│   ├── simulate_with_coefficients.py  (NEW — §2.2)
│   ├── compute_cost_drake.py          (NEW — §2.5)
│   ├── synthesize_target_from_coefficients.py  (NEW — §2.6, TDD oracle)
│   ├── fit_swing_drake.py             (NEW — §2.3, scipy.minimize driver)
│   ├── fit_swing_drake_autodiff.py    (NEW — §2.4, MathematicalProgram + auto-diff)
│   ├── visualize_fit.py               (NEW — §2.7, Meshcat overlay)
│   └── tests/
│       ├── test_humanoid_urdf.py
│       ├── test_simulate_with_coefficients.py
│       ├── test_synthesize_oracle.py
│       ├── test_fit_swing_drake.py
│       └── test_equivalence_simscape.py
├── motion_optimization.py             (existing — slimmed; see §1.3)
└── ... (existing src/, tests/ unchanged)
```

`scripts/build_humanoid_models.py` (cross-engine, owned by issue
**PARITY-MODEL-BUILD**) gains a `--engine drake` mode that imports
`humanoid_urdf.build_humanoid_urdf(yaml_path)` and writes the result to
`src/engines/physics_engines/drake/models/generated/golfer.urdf`.

### 2.1 Humanoid model loader

```python
# python/motion_matching/humanoid_urdf.py
from __future__ import annotations
from pathlib import Path
import yaml
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

CANONICAL_URDF = Path(__file__).parents[2] / "models" / "generated" / "golfer.urdf"

def build_humanoid_urdf(yaml_path: Path | None = None,
                        out_path: Path | None = None) -> Path:
    """Generate a Drake-compatible URDF from the shared anthropometric YAML.

    Decision-by-contract:
        - Pre: yaml_path exists and is a valid `golf_humanoid_dimensions.yaml`.
        - Pre: every segment listed has finite mass + inertia entries.
        - Post: the file at out_path parses cleanly via pydrake `Parser`
                and the resulting plant has exactly 25 generalized velocities
                (6 floating-base + 19 actuated rotational DOFs).
    """
    ...

def load_humanoid_into_plant(plant: MultibodyPlant,
                             urdf_path: Path = CANONICAL_URDF) -> ModelInstanceIndex:
    """Add the canonical humanoid to a Drake MultibodyPlant via Parser."""
    ...
```

The existing `drake_golf_model.GolfURDFGenerator` is the **basis**: rip
out its hard-coded `GolfModelParams` defaults, replace them with values
loaded from `shared/models/golf_humanoid_dimensions.yaml`, and emit a
URDF **file** (not just an in-memory tree). Hand-edited URDFs are
forbidden per cross-engine §6.

### 2.2 Forward simulator wrapper

```python
# python/motion_matching/simulate_with_coefficients.py
from __future__ import annotations
import numpy as np
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.sim_options import SimOptions, SimOut
from .humanoid_urdf import load_humanoid_into_plant

def simulate_with_coefficients(
    theta: np.ndarray,                # (n_joints * 7,)
    options: SimOptions = SimOptions.default(),
    initial_pose: dict | None = None,
) -> SimOut:
    """Run a Drake forward simulation from a torque-polynomial coefficient vector.

    1. Build a `DiagramBuilder` + `MultibodyPlant` (fresh per call for
       thread-safety; reuse via a `lru_cache`-keyed-on-options helper if
       wall-clock becomes a problem).
    2. Load the canonical humanoid URDF via `load_humanoid_into_plant`.
    3. Add a `LeafSystem` actuator that evaluates
            tau_j(t) = A_j + B_j*t + C_j*t^2 + D_j*t^3 + E_j*t^4 + F_j*t^5 + G_j*t^6
       from theta, mirroring the Simscape Stateflow torque polynomial
       (see `Simscape_Multibody_Models/.../torque_polynomial.m`).
    4. Set initial pose from `initial_pose` (default: address pose from
       `load_impact_starting_position`).
    5. Run `Simulator.AdvanceTo(options.simulation_time_s)` recording q, v, tau.
    6. Sample to the canonical grid (`options.sample_rate_hz`, default 1 kHz).
    7. Run forward-kinematics to obtain `grip` / `grip_quat` / `clubhead` /
       `club_quat` from the recorded `q` (use `body.body_frame()` not
       `FixedOffsetFrame` per CLAUDE.md).
    8. Return a canonical `SimOut`.

    The polynomial-evaluator `LeafSystem` MUST be implemented twice:
    a `float` version for the scipy driver, and a templated
    `T = AutoDiffXd` version for the MathematicalProgram driver in §2.4.
    Use `pydrake.systems.framework.LeafSystem_` template syntax.
    """
    ...
```

Returns the canonical `SimOut` from the cross-engine spec (q, qd, qdd,
tau, grip, grip_quat, clubhead, club_quat, solver_status, duration_s).

### 2.3 Fit driver (gradient-free, scipy)

```python
# python/motion_matching/fit_swing_drake.py
from src.shared.python.motion_matching.cost import compute_cost
from .simulate_with_coefficients import simulate_with_coefficients

def fit_swing_drake(
    target: ClubTarget,
    options: FitOptions = FitOptions.default(),
) -> FitResult:
    """Default driver: scipy.optimize.minimize, finite-difference gradients.

    Mirrors fit_swing_fmincon for Simscape, fit_swing_mujoco for MuJoCo.
    Cost function is `shared/python/motion_matching/cost.py` — engine-agnostic.
    """
    def objective(theta):
        sim_out = simulate_with_coefficients(theta, options.sim_options)
        return compute_cost(sim_out, target, options.cost_options)
    result = scipy.optimize.minimize(
        objective, theta0, method="L-BFGS-B",
        bounds=options.coefficient_bounds,
        options={"maxiter": options.max_iterations, "ftol": options.tolerance},
    )
    return FitResult(...)
```

### 2.4 Auto-diff fit driver (Drake's killer feature)

```python
# python/motion_matching/fit_swing_drake_autodiff.py
from pydrake.solvers import MathematicalProgram, IpoptSolver, Solve
from pydrake.systems.framework import LeafSystem_
from pydrake.autodiffutils import AutoDiffXd

def fit_swing_drake_autodiff(
    target: ClubTarget,
    options: FitOptions = FitOptions.default(),
) -> FitResult:
    """Gradient-based driver leveraging Drake auto-diff through the dynamics.

    1. Construct a templated MultibodyPlant<AutoDiffXd> via
       plant.ToAutoDiffXd(). The polynomial torque LeafSystem is templated
       so the autodiff plant connects identically.
    2. `prog = MathematicalProgram(); theta = prog.NewContinuousVariables(n)`.
    3. Add a custom cost `prog.AddCost(scalar_cost_with_gradient, theta)` that:
         a. Sets theta into the autodiff plant context.
         b. Simulates forward with autodiff scalars (uses RungeKutta3 or the
            built-in Simulator with autodiff context).
         c. Computes the canonical grip-primary cost (manually, replicating
            shared/python/motion_matching/cost.py for AutoDiffXd; this is the
            one place we duplicate logic and we MUST add a numerical-equivalence
            test against the float version, per cross-engine §2.7).
    4. `IpoptSolver().Solve(prog, theta_init)` returns analytic gradients
       through the dynamics — no finite differencing.

    Acceptance: convergence in ≤ 50 sim calls vs ~150 for fmincon on the
    same trial (Simscape baseline).
    """
    ...
```

This is the milestone that justifies Drake's existence in the parity
matrix. Issues §4-5 below explicitly track this as a separate
deliverable.

### 2.5 Cost adapter (thin)

```python
# python/motion_matching/compute_cost_drake.py
from src.shared.python.motion_matching.cost import compute_cost as _shared_cost

def compute_cost_drake(sim_out, target, options=None, *,
                       constraint_residuals: list[np.ndarray] | None = None):
    """Thin wrapper around the canonical cost.

    The shared cost lives in shared/python/motion_matching/cost.py and
    is engine-agnostic (cross-engine §2.3). This wrapper adds Drake-specific
    constraint-residual penalties (joint-limit slack from
    MathematicalProgram, contact-impulse residuals if we add ground contact)
    and nothing else.
    """
    j = _shared_cost(sim_out, target, options)
    if constraint_residuals:
        j = j + sum(np.sum(r ** 2) for r in constraint_residuals)
    return j
```

This file MUST stay under 100 LOC. If it grows past that, the bloat
belongs in the shared cost module.

### 2.6 TDD oracle

```python
# python/motion_matching/synthesize_target_from_coefficients.py
def synthesize_target_from_coefficients(theta: np.ndarray,
                                        options=None) -> ClubTarget:
    """The Drake-side TDD oracle, mirroring the MATLAB version.

    Runs simulate_with_coefficients(theta), packages the resulting
    grip/grip_quat/clubhead/club_quat as a ClubTarget with
    source.format='synthetic' and source.theta_truth=theta.
    Required by cross-engine §2.7 and the "synthesize → fit → recover"
    test in test_fit_swing_drake.py.
    """
```

### 2.7 Visualisation

Build on the existing `drake_visualizer.py` (Meshcat wrapper) plus new
shared plotters in `shared/python/motion_matching/plot_*.py`:

| View                   | Implementation                                                                                                                                                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trajectory overlay** | `Meshcat` scene: humanoid skeleton (drawn from plant + URDF), measured-grip path as a `Cylinder` polyline, simulated-grip path as a contrasting `Cylinder` polyline, clubhead trace optional. Use `MeshcatVisualizer.AddToBuilder` for the skeleton. |
| **Error timecourse**   | `shared/python/motion_matching/plot_error_timecourse.py` — engine-agnostic Matplotlib.                                                                                                                                                               |
| **Fit quality card**   | `shared/python/motion_matching/plot_fit_quality_card.py` — engine-agnostic.                                                                                                                                                                          |

Drake-specific code stays inside `visualize_fit.py` and is limited to
the Meshcat scene set-up; the 2D plots come from the shared module.

### 2.8 Tests

TDD-first. Every new public function lands with a test in the same
PR. The test plan is:

| Test                                                                       | Asserts                                                                                                                                     |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_humanoid_urdf.py::test_parses`                                       | Generated URDF parses via pydrake `Parser` and yields 25 generalized velocities (6 floating-base + 19 actuated rotational).                 |
| `test_humanoid_urdf.py::test_segment_lengths`                              | Distance from pelvis frame to shoulder frame matches the YAML to 1 mm.                                                                      |
| `test_simulate_with_coefficients.py::test_canonical_simout_shape`          | Returns a `SimOut` with N rows on the requested grid; no NaNs.                                                                              |
| `test_simulate_with_coefficients.py::test_zero_torque_falls_under_gravity` | With `theta = 0`, the grip falls in the −z direction (sanity for gravity sign).                                                             |
| `test_synthesize_oracle.py::test_round_trip`                               | `synthesize_target_from_coefficients(theta)` returns a `ClubTarget` whose `source.theta_truth` equals `theta`.                              |
| `test_fit_swing_drake.py::test_recovers_synthetic_swing`                   | Synthetic target → `fit_swing_drake` → `final_rmse_m < 0.005`.                                                                              |
| `test_fit_swing_drake.py::test_autodiff_converges_faster`                  | `fit_swing_drake_autodiff` uses ≤ ½ the simulator calls of the scipy driver on the same trial.                                              |
| `test_equivalence_simscape.py::test_grip_rmse_under_5mm`                   | Fixed `theta` → Drake grip vs Simscape ground-truth grip RMSE ≤ 5 mm at three poses (impact, top-of-backswing, address). Cross-engine §2.2. |

Marker discipline (per `pyproject.toml`):

- Pure URDF / dataclass tests → `@pytest.mark.unit`.
- Anything that builds a `MultibodyPlant` and runs a Simulator →
  `@pytest.mark.integration` (or `slow` if > 2 s).
- The equivalence test that diffs against a stored Simscape ground-truth
  vector → `@pytest.mark.scientific`.

Mock discipline: use `patch.dict("sys.modules", ...)` in tests that
need to stub `pydrake`, **never** `sys.modules["pydrake"] = MagicMock()`
at module level (CLAUDE.md).

---

## 3. Body model — concrete URDF authoring plan

### 3.1 Skeleton (25 generalized velocities)

> **Source of truth:** `shared/models/golf_humanoid_topology.yaml`
> (PR #4150 — PARITY-DIMENSIONS). Total = 6 floating-base + 19 actuated
> rotational DOFs = **25 v-velocities**. Note that `q` carries 26 elements
> (7 per floating base = 3 position + 4 quaternion, plus 19 actuated)
> while `v` carries 25 (6 + 19).

Mirror the Simscape skeleton joint-by-joint. Drake natively supports
`revolute`, `prismatic`, `floating`, and (via composition with massless
"dummy" links) `universal` and `gimbal` chains. URDF is more restrictive
than SDF — we use URDF because the existing `GolfURDFGenerator` already
emits it; SDF migration is a future option.

| Body chain                        | Joints          | DOF | URDF encoding                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------- | --------------- | --: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| World → pelvis                    | 1 floating root |   6 | `<joint type="floating">` with quaternion convention `[w,x,y,z]` matching the Simscape model.                                                                                                                                                                                                                                                    |
| pelvis → spine_lower              | 2 (universal)   |   2 | Two `revolute` joints sharing a massless dummy link, axes `[1,0,0]` and `[0,1,0]`.                                                                                                                                                                                                                                                               |
| spine_lower → spine_upper         | 1 twist         |   1 | `revolute`, axis `[0,0,1]`.                                                                                                                                                                                                                                                                                                                      |
| spine_upper → torso_hub           | 0 (welded)      |   0 | `<joint type="fixed">`.                                                                                                                                                                                                                                                                                                                          |
| torso_hub → R_scapula             | 2 (universal)   |   2 | Two `revolute`, axes `[0,1,0]` and `[0,0,1]`.                                                                                                                                                                                                                                                                                                    |
| R_scapula → R_shoulder            | 3 (gimbal)      |   3 | Three `revolute` chained with two massless dummy links (axes `[1,0,0]`, `[0,1,0]`, `[0,0,1]`).                                                                                                                                                                                                                                                   |
| R_shoulder → R_elbow              | 1 hinge         |   1 | `revolute`, axis `[0,1,0]`.                                                                                                                                                                                                                                                                                                                      |
| R_elbow → R_wrist                 | 2 (universal)   |   2 | Two `revolute`.                                                                                                                                                                                                                                                                                                                                  |
| R_wrist → R_hand                  | 0 (welded)      |   0 | `<joint type="fixed">`.                                                                                                                                                                                                                                                                                                                          |
| (Mirror chain L_scapula … L_hand) |                 |   8 | Symmetric structure, mirrored axes/origins.                                                                                                                                                                                                                                                                                                      |
| R_hand + L_hand → club_grip       | 6 (welded loop) |   0 | `<joint type="fixed">` from R_hand to club. The L_hand → club connection becomes a closed-loop constraint, which URDF cannot express. **Resolution:** weld the club to R_hand only and rely on the IK / cost function to keep both hands on the grip. (Same compromise the Pinocchio URDF makes — see `pinocchio/models/generated/golfer.urdf`.) |
| club_grip → clubhead              | 0 (welded)      |   0 | `<joint type="fixed">` shaft length from YAML.                                                                                                                                                                                                                                                                                                   |

**Total v-DOF: 6 (floating root) + 2 + 1 + 0 + 2 + 3 + 1 + 2 + 0 + 8 (mirror) + 0 + 0 = 25.** ✅
(The previous spec quoted "23" — an arithmetic error caught by issue #4155.
The component breakdown above already sums to 25; only the displayed total
was wrong. Cross-reference: `shared/models/golf_humanoid_topology.yaml`.)

### 3.2 Composition rules

Drake's URDF parser does **not** support `multi-dof` joint types
directly. Universal joints become **two** revolute joints connected by
a massless dummy link (`mass=1e-3`, `I = 1e-6 * I_3`, no visual);
gimbals become three. The `GolfURDFGenerator._add_spine` already
demonstrates this pattern. Naming convention:

```
spine_universal_dummy_x → spine_universal_dummy_y → spine_lower
right_shoulder_gimbal_x → right_shoulder_gimbal_y → right_shoulder_gimbal_z → right_upper_arm
```

The joint state vector ordering then matches the Simscape coefficient
ordering when the dummy links are listed in the same `axis` sequence.

### 3.3 Inertias

Pulled from `shared/models/golf_humanoid_inertia.yaml` (cross-engine
issue **PARITY-DIMENSIONS**). Each segment specifies:

```yaml
- name: right_upper_arm
  parent: right_shoulder_gimbal_z
  joint:
    type: revolute # composed; this entry is the last gimbal axis
    axis: [0, 0, 1]
    damping: 0.5
    limits: [-3.14, 3.14]
  origin:
    xyz: [0.0, -0.18, 0.0]
    rpy: [0.0, 0.0, 0.0]
  mass: 2.0
  inertia: { ixx: 0.018, iyy: 0.018, izz: 0.0024, ixy: 0.0, ixz: 0.0, iyz: 0.0 }
  geometry: { type: cylinder, size: [0.04, 0.30] }
```

The Simscape model workspace exposes these values as
`Param.RightUpperArm.Mass`, `Param.RightUpperArm.Inertia`, etc.; the
**PARITY-DIMENSIONS** issue captures them once into the shared YAML
that all five engines consume.

### 3.4 Club attachment

The Simscape model attaches the club to the right hand via a 6-DOF
welded joint. We replicate exactly: `<joint type="fixed">` from
`right_hand` to `club_grip`, with the YAML-supplied grip pose. The
clubhead is a `fixed` extension at the YAML-supplied shaft length.

### 3.5 Generation pipeline

The cross-engine issue **PARITY-MODEL-BUILD** owns the orchestrator
script `scripts/build_humanoid_models.py`. The Drake-side entry point:

```python
# scripts/build_humanoid_models.py
from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    build_humanoid_urdf,
)

if "drake" in args.engines:
    out = Path("src/engines/physics_engines/drake/models/generated/golfer.urdf")
    build_humanoid_urdf(yaml_path=SHARED_YAML, out_path=out)
```

CI runs this on every PR and asserts the on-disk URDF matches the
freshly-regenerated bytes (issue **DRAKE-MODEL-CI**). This makes
hand-editing the URDF impossible without the build script catching it.

---

## 4. Implementation plan (atomic, testable issues)

Six issues, sized so a single agent run can land each one. Issues
follow the cross-engine convention (`feat/` branch, target `main`,
include the equivalence-test row in the PR body).

### DRAKE-1: Shared anthropometric YAML consumer + URDF generator refactor

**Deliverable:** `python/motion_matching/humanoid_urdf.py`. Refactor
the existing `drake_golf_model.GolfURDFGenerator` to consume
`shared/models/golf_humanoid_dimensions.yaml` (which arrives via
**PARITY-DIMENSIONS**). Emit the URDF to
`models/generated/golfer.urdf`.

**Acceptance:**

1. `pytest tests/motion_matching/test_humanoid_urdf.py` green.
2. The on-disk URDF parses via `pydrake.multibody.parsing.Parser` with
   exactly 25 generalized velocities (6 floating-base + 19 actuated).
3. `GolfModelParams` hard-coded defaults are deleted; YAML is the
   single source of truth.

**Size:** M (~1 day). **Depends on:** PARITY-DIMENSIONS.

### DRAKE-2: `simulate_with_coefficients` (float)

**Deliverable:** `python/motion_matching/simulate_with_coefficients.py`.
Forward-sim wrapper matching cross-engine §2.2 signature. Includes the
Stateflow-equivalent torque polynomial as a `LeafSystem` (float-only
this issue; templated AutoDiffXd version in DRAKE-4).

**Acceptance:**

1. `simulate_with_coefficients(theta_known) → SimOut` with finite
   `q`, `qd`, `tau`.
2. Round-trip test: `theta = 0` → grip falls under gravity.
3. Output schema matches `SimOut` byte-for-byte (use canonical
   dataclass).

**Size:** L (~2 days). **Depends on:** DRAKE-1, PARITY-LOADERS (for
`SimOut`).

### DRAKE-3: `fit_swing_drake` (scipy)

**Deliverable:** `python/motion_matching/fit_swing_drake.py` plus
`compute_cost_drake.py`. Calls `shared/python/motion_matching/cost.py`
directly. Default optimizer is `scipy.optimize.minimize(method="L-BFGS-B")`
with finite-difference Jacobians.

**Acceptance:**

1. Synthetic-recovery test: `synthesize_target_from_coefficients(theta_truth)`
   → `fit_swing_drake` → `final_rmse_m < 5 mm`.
2. `FitResult` schema matches the cross-engine canonical shape
   (`theta_optimal`, `final_rmse_m`, `solver_status`, `iterations`,
   `wall_clock_s`).
3. `compute_cost_drake.py` ≤ 100 LOC and imports the shared cost.

**Size:** M (~1 day). **Depends on:** DRAKE-2.

### DRAKE-4: AutoDiffXd `simulate_with_coefficients` + `fit_swing_drake_autodiff`

**Deliverable:** `python/motion_matching/fit_swing_drake_autodiff.py`
plus a templated `LeafSystem_[T]` polynomial-torque source. Uses
`MathematicalProgram` + `IpoptSolver`. The killer-feature milestone.

**Acceptance:**

1. Round-trip on the same synthetic target as DRAKE-3 with
   `final_rmse_m < 1 mm` (tighter, because gradients).
2. **Sim-call budget:** ≤ 50 forward simulations to converge vs ≥ 100
   for the scipy driver on the same trial.
3. Numerical-equivalence test: cost computed in float vs autodiff
   matches to 1e-9 relative.

**Size:** XL (~3 days; the autodiff plumbing is the trickiest part of
the parity work). **Depends on:** DRAKE-3.

### DRAKE-5: Equivalence test against Simscape ground truth

**Deliverable:** `tests/motion_matching/test_equivalence_simscape.py`
plus a checked-in ground-truth fixture
`tests/motion_matching/fixtures/simscape_ground_truth.json`. Runs the
fixed `theta` from cross-engine §2.2 through Drake's
`simulate_with_coefficients` and asserts grip RMSE ≤ 5 mm vs the
Simscape ground-truth grip path at impact, top-of-backswing, and
address.

**Acceptance:**

1. Test passes on CI.
2. Cross-engine §2.2 equivalence row turns 🟢 in the parity matrix
   (cross-engine spec table).
3. Test is marked `@pytest.mark.scientific` and runs in `nightly` lane.

**Size:** M (~1 day). **Depends on:** DRAKE-2 (and a Simscape ground-truth
fixture, ideally produced by a Simscape-side issue PARITY-FIXTURE).

### DRAKE-6: Visualisation parity (Meshcat overlay + shared plotters)

**Deliverable:** `python/motion_matching/visualize_fit.py`. Wires the
existing `drake_visualizer.DrakeVisualizer` to draw the canonical
trajectory overlay; depends on `shared/python/motion_matching/plot_*.py`
for the 2D plots.

**Acceptance:**

1. `python -m motion_matching.visualize_fit results/<trial>/drake.json`
   pops a Meshcat browser tab with skeleton + grip-path overlay.
2. Error timecourse + fit quality card use the shared plotters
   (no duplicated Matplotlib code).
3. Smoke test (`@pytest.mark.requires_gl`) renders without error in
   headless mode.

**Size:** S (~½ day). **Depends on:** DRAKE-3.

### DRAKE-7 (optional, post-MVP): URDF regeneration CI gate

**Deliverable:** GitHub Actions workflow step that re-runs
`scripts/build_humanoid_models.py --engine drake` and asserts the
on-disk URDF is byte-identical. Mirrors the Pinocchio URDF gate.

**Size:** XS (~2 hours). **Depends on:** DRAKE-1.

---

## 5. TDD / DbC / DRY / LOD compliance

### 5.1 TDD

- Every issue above lists its acceptance test. The test is written
  first, fails, then the implementation lands in the same PR.
- `tests/motion_matching/` mirrors `python/motion_matching/`. No
  follow-up "tests next PR" — the cross-engine spec forbids it.
- Marker hygiene: `unit` for dataclass / URDF tests, `integration` for
  full sim, `scientific` for the Simscape equivalence diff.

### 5.2 DbC

Use Python `dataclass`es with `__post_init__` validators **and**
`pydantic` for the user-facing options structs (`SimOptions`,
`FitOptions`). The `@dataclass(frozen=True)` pattern from
`shared/python/motion_matching/club_target.py:ClubTarget` is the
template. Postconditions land as `assert` at the end of each public
function:

```python
def simulate_with_coefficients(theta, ...):
    assert theta.shape == (n_joints * 7,), "DbC: theta wrong shape"
    assert np.all(np.isfinite(theta)), "DbC: theta must be finite"
    ...
    out = SimOut(...)
    assert out.q.shape[0] == out.time.shape[0], "Post: ragged SimOut"
    assert np.all(np.isfinite(out.grip)), "Post: grip must be finite"
    return out
```

The existing `src.shared.python.core.contracts` module's `precondition`
/ `postcondition` / `invariant` decorators (already used in
`drake_physics_engine.py`) are the canonical decorator path.

### 5.3 DRY

- **No engine-specific cost code.** `compute_cost_drake.py` is a
  ≤ 100-LOC adapter; everything else lives in the shared cost module.
- **No engine-specific loader code.** `ClubTarget` comes from
  `shared/python/motion_matching/club_target.py` only.
- **No copy-pasted URDF.** Generated from the shared YAML. Hand-edits
  fail CI.
- **Shared plotters before engine plotters.** Build the engine-agnostic
  plot first; only fall back to engine-bespoke code for the 3D viewer.

### 5.4 LOD

Python `@property` and explicit delegating methods, no
`a.b.c.d.e` chains. The existing `drake_physics_engine.py` is mostly
compliant; new code matches that style. CI's static analysis (Ruff
B-rules) catches violations.

### 5.5 Drake-Context-Manager alignment

Drake's `Context` and `MultibodyPlant` lifecycle is fragile: the plant
must be `Finalize()`-d before any context exists, and contexts cannot
be shared across diagrams. Wrap the plant + diagram in a small
context-manager:

```python
@contextmanager
def drake_simulation_context(urdf_path: Path,
                             time_step: float = 1e-3):
    """Yield (plant, diagram, context) for a single sim run.

    Postcondition: plant is Finalize()-d, context is fresh, the
    yielded triple is safe to pass to a Simulator. On exit, drops
    the diagram (Drake objects are GC-managed; the explicit del is
    only for predictable timing in tests).
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    Parser(plant).AddModels(str(urdf_path))
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    try:
        yield plant, diagram, context
    finally:
        del context, diagram, plant
```

This pattern is already in `drake_physics_engine.DrakePhysicsEngine`'s
`_ensure_finalized`; the new `simulate_with_coefficients` reuses it
rather than spinning up its own diagram.

---

## 6. Performance baseline + targets

### 6.1 Baseline (extrapolated from `drake_physics_engine` benchmarks)

| Quantity                                                   | Value                            | Source                                                                                                                           |
| ---------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Single forward sim, 0.3 s @ 1 ms timestep, 25-DOF humanoid | ~1.5 s wall-clock                | extrapolated from `compute_mass_matrix` micro-benchmarks in `drake_physics_engine.py` (~0.5 ms per call × 300 steps × overhead). |
| `simulate_with_coefficients` (one full call)               | 1.5–3 s                          | adds polynomial-torque LeafSystem + state recording.                                                                             |
| `fit_swing_drake` (scipy L-BFGS-B, finite diff)            | ~150 sim calls × 2 s = **5 min** | matches Simscape `fmincon` baseline.                                                                                             |
| `fit_swing_drake_autodiff` (Ipopt + analytic gradients)    | ~30 sim calls × 3 s = **90 s**   | autodiff overhead × fewer iters.                                                                                                 |

### 6.2 Targets

| Goal                      | Target                     | How                                                              |
| ------------------------- | -------------------------- | ---------------------------------------------------------------- |
| Sim wall-clock            | ≤ 3 s                      | Acceptable as-is.                                                |
| Fit wall-clock (scipy)    | ≤ 5 min per swing          | Matches Simscape.                                                |
| Fit wall-clock (autodiff) | **≤ 30 s per swing**       | Drake's killer feature; the parity-matrix justification.         |
| Memory                    | ≤ 1 GB resident during fit | Drake's autodiff scalars are heavy; budget caps need monitoring. |

### 6.3 Profiling hooks

The autodiff pathway warrants `cProfile` instrumentation behind a
`--profile` flag. Save profiles to `results/<trial>/drake.profile`;
the leaderboard helper picks them up.

---

## 7. Risks / open questions

1. **`pydrake` install footprint.** `pydrake` is ~500 MB and the wheel
   is x86-64 / aarch64 only. The CI runners need explicit `pydrake`
   install in `requirements-ci.txt` (already present per
   `python/requirements-ci.txt`). Verify the version pin matches the
   stable AutoDiffXd API (≥ 1.20.0).
2. **AutoDiff through the polynomial torque source.** The
   Stateflow-equivalent polynomial is a closed-form `np.polyval`-style
   evaluation — autodiff should flow cleanly. The risk is subtle: if
   any intermediate value escapes to a `float`-only function (e.g.
   `np.linalg.solve` instead of pydrake's `LinearSolve`), the gradient
   silently breaks. The numerical-equivalence test in DRAKE-4 catches
   this.
3. **Closed-loop constraint at the left hand.** URDF cannot express the
   closed kinematic loop where both hands grip the club. We weld the
   club to the right hand only and let the cost function keep the left
   hand on the grip. The Pinocchio URDF makes the same compromise; if
   it produces measurable error in the equivalence test (DRAKE-5), we
   escalate to SDF + `<plugin>` constraints (a follow-up issue).
4. **Visualizer dependency.** Meshcat (browser-based) is preferred but
   requires a free port + browser. The legacy `drake-visualizer`
   binary is deprecated. The smoke test in DRAKE-6 must run headless;
   if Meshcat won't start in CI, fall back to writing a static PNG via
   Drake's `MeshcatRecorder` and let the test assert the PNG exists.
5. **Drake's `Simulator` integrator stability with stiff polynomial
   torques.** Default `RungeKutta3` may struggle when polynomial
   coefficients drive the system into a high-acceleration regime. We
   may need to fall back to `ImplicitEulerIntegrator` for the autodiff
   path (it's friendlier to autodiff scalars). Issue DRAKE-4 should
   benchmark both.
6. **Python 3.10 vs 3.11 wheel availability.** `pydrake` wheels exist
   for 3.10+, but stable autodiff behaviour is best on 3.11. CLAUDE.md
   already pins 3.11 as the CI default; we keep that.
7. **Floating-base joint convention.** Drake's `<joint type="floating">`
   uses quaternion `[w,x,y,z]`; the Simscape model uses Euler angles in
   one place. We commit to quaternion world frames everywhere
   (`SimOut.grip_quat` is `[w,x,y,z]` per the canonical schema) and
   convert at the loader boundary if Simscape ever exposes Euler.
8. **Coverage threshold.** Drake's mocked-pydrake unit tests can't
   exercise the full integration path; coverage on
   `simulate_with_coefficients.py` lands ~70%. The integration-tier
   tests (`@pytest.mark.integration`) push it ≥ 90%. CI must run both
   tiers before reporting coverage.

---

_Last updated 2026-05-06. Tracks the same parity wave as
[`../mujoco/MUJOCO_PARITY_SPEC.md`](../mujoco/MUJOCO_PARITY_SPEC.md),
[`../pinocchio/PINOCCHIO_PARITY_SPEC.md`](../pinocchio/PINOCCHIO_PARITY_SPEC.md),
and [`../opensim/OPENSIM_PARITY_SPEC.md`](../opensim/OPENSIM_PARITY_SPEC.md)._
