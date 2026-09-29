# MuJoCo Parity Spec — Motion-Matching Pipeline

Engine: **MuJoCo** (`src/engines/physics_engines/mujoco/`)
Reference: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/`
Sibling: `CROSS_ENGINE_PARITY_SPEC.md` (in flight on `feat/cross-engine-parity-spec`)
Status: planning — no code in this PR.

> This spec defines the work to bring the MuJoCo engine into parity with the
> Simscape Multibody motion-matching pipeline (Option-1 fmincon-SQP fit driver).
> MuJoCo is the _closest-to-parity_ of the four engines — it already has model
> variants, a mocap loader, and a `SwingOptimizer`, so the work is mostly
> **wiring to the canonical schema**, **filling stubs**, and **fixing bugs**
> that block even a basic `from_xml_string` compile today.

---

## 1. Current state

### 1.1 Surface area inventory

| Path                                                                                       | LoC  | Status                                                                                                                               |
| ------------------------------------------------------------------------------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `src/engines/physics_engines/mujoco/_golf_swing_advanced_xml.py`                           | 491  | **Broken at import time** (gravity attr renders as `PhysicalConstant(...)` literal — fails MJCF compile)                             |
| `src/engines/physics_engines/mujoco/_golf_swing_full_body_xml.py`                          | 288  | **Broken at import time** (same gravity bug)                                                                                         |
| `src/engines/physics_engines/mujoco/_golf_swing_upper_body_xml.py`                         | 213  | **Broken at import time** (same gravity bug)                                                                                         |
| `src/engines/physics_engines/mujoco/golf_swing_models_xml.py`                              | n/a  | re-export shim over the three above                                                                                                  |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_motion_opt_simulation.py` | 209  | Joint-space PD-tracking forward sim; **not** a polynomial-torque driver                                                              |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_motion_opt_trajectory.py` | 111  | Cubic-spline trajectory interp + bounds + jerk; reusable                                                                             |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_motion_opt_types.py`      | 52   | `OptimizationObjectives`, `OptimizationConstraints`, `OptimizationResult` dataclasses                                                |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_loader.py`          | 139  | Generic CSV / JSON / BVH loader; emits `MotionCaptureSequence`, **not** `ClubTarget`                                                 |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_data.py`            | n/a  | `MotionCaptureFrame`, `MotionCaptureSequence` dataclasses                                                                            |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_processor.py`       | n/a  | Filtering / smoothing                                                                                                                |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_retargeting.py`     | n/a  | Marker→joint retargeting (stub)                                                                                                      |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/_mocap_validator.py`       | n/a  | Validation rules                                                                                                                     |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_optimization.py`    | 602  | `SwingOptimizer`: scipy `minimize` + `differential_evolution`; objectives are speed/jerk/torque, **not** the canonical cost function |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py`   | 668  | Qt UI for sketching polynomial torque profiles. Not wired to the forward sim.                                                        |
| `src/engines/physics_engines/mujoco/docker/gui/golf_gui_docker.py`                         | n/a  | Containerised viewer; not part of the fit loop                                                                                       |
| `src/shared/python/motion_matching/`                                                       | 1232 | **Canonical Python package** the engine must call into                                                                               |
| `src/shared/python/motion_matching/loaders/synthetic.py`                                   | 40   | **Stub** — raises `NotImplementedError` pending the engine wiring this spec defines                                                  |
| `src/shared/models/myosuite/myo_sim/body/myobody.xml`                                      | 30   | Full-body myosuite MJCF (asset paths broken when loaded standalone)                                                                  |
| `src/shared/models/myosuite/myo_sim/body/myoupperbody.xml`                                 | 19   | Upper-body myosuite MJCF (same asset-path issue)                                                                                     |

Totals: 195 Python files in `src/engines/physics_engines/mujoco/`; ~50 of them are GUI/launcher/Docker glue irrelevant to the fit loop.

### 1.2 What works

- The `_motion_opt_trajectory.py` cubic-spline interpolator and bounds builder are usable as-is.
- `mujoco.MjModel.from_xml_string` and `mj_step` work fine on hand-written MJCF (verified by a 2-DOF probe — `compile=2.3 ms`, `0.3 s of sim = 1.0 ms`).
- The `SwingOptimizer` skeleton (knot-point parameterisation, scipy hookup) shows the right shape; it just optimises the wrong cost.
- `MotionCaptureLoader.load_csv` / `load_json` parse generic mocap; **not** the Wiffle xlsx layout the canonical spec mandates.
- `polynomial_generator.py` already has the math for 6th-order polynomials — but it lives behind a Qt UI, not as a callable.

### 1.3 What's stubbed / broken

- **All three `_golf_swing_*_xml.py` model variants fail to compile** because `GRAVITY_M_S2` is a `PhysicalConstant` (not a `float`), and the f-string interpolates the `repr`. Any agent who runs `MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML)` today gets `XML Error: bad format in attribute 'gravity'`. Fixing this is a one-character change but it's load-bearing.
- The myosuite full-body and upper-body MJCFs are present at `src/shared/models/myosuite/myo_sim/body/{myobody,myoupperbody}.xml` but their asset includes resolve to the wrong directory when loaded standalone (`Error opening file '../src/shared/models/myosuite/myo_sim/myo_sim/scene/myosuite_scene.png'`). The duplicated `myo_sim/myo_sim/` segment is the giveaway.
- `loaders/synthetic.py::synthesize_target_from_coefficients` raises `NotImplementedError`. The note in its docstring says this depends on issue #018 (`simulate_with_coefficients`) — that issue is the central deliverable of this spec.
- No `simulate_with_coefficients` exists for MuJoCo. The closest analogue is `_motion_opt_simulation.simulate_trajectory(model, data, trajectory, club_head_id, swing_duration, num_knot_points)`, which takes a **knot-point joint trajectory**, not a polynomial-coefficient vector, and uses **PD-tracking actuators**, not torque-coefficient drivers.
- No `fit_swing_mujoco` exists. `SwingOptimizer.optimize_trajectory` exists but optimises peak club-head speed minus jerk minus torque — not the canonical `compute_cost` from `COST_FUNCTION_SPEC.md`.
- No engine-side `synthesize_target_from_coefficients`. The TDD oracle has no implementation.
- No tests under `tests/motion_matching/` exercise the MuJoCo path. Existing MuJoCo tests live under `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/tests/` and exercise unrelated subsystems.

### 1.4 The three model variants — what's the difference?

All three are hand-written MJCF strings emitted from Python f-strings:

| Variant                                     | Bodies                                                                           | DOFs (approx)    | Use-case rationale                                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| `_golf_swing_upper_body_xml.py` (213 lines) | pelvis fixed, torso + 2 arms + club                                              | ~9 hinge joints  | Smallest tractable target; mirrors Simscape's `golf_3D_upper.slx`                                        |
| `_golf_swing_full_body_xml.py` (288 lines)  | feet → ankle → knee → hip → torso → arms → club                                  | ~17 hinge joints | Default for production fits; mirrors Simscape's `golf_3D_full.slx`                                       |
| `_golf_swing_advanced_xml.py` (491 lines)   | adds scapulae, finer forearm/wrist split, RK4+Newton solver, 50 iter, 8k shadows | ~21 hinge joints | Research / visualisation. Compile-only diff vs full_body is an extra scapula chain and a stiffer solver. |

The three are **not** auto-generated from a shared YAML — they have copy-pasted `<material>`, `<camera>`, and `<default>` blocks. This is a DRY violation that PARITY-DIMENSIONS aims to address; see §3.

---

## 2. Target architecture

### 2.1 File-by-file deliverables

All paths relative to the repo root.

```
src/engines/physics_engines/mujoco/motion_matching/
├── __init__.py
├── simulate_with_coefficients.py     # Canonical forward-sim wrapper (ISSUE-MUJOCO-3)
├── fit_swing_mujoco.py               # Canonical fit driver (ISSUE-MUJOCO-4)
├── synthesize_target.py              # TDD oracle (ISSUE-MUJOCO-5)
├── _torque_driver.py                 # Polynomial-torque actuator wiring (ISSUE-MUJOCO-3)
├── _model_builder.py                 # Loads/compiles the MJCF (ISSUE-MUJOCO-2)
└── viz/
    ├── __init__.py
    └── render_swing.py               # Thin renderer over mujoco.viewer (ISSUE-MUJOCO-6)

tests/motion_matching/mujoco/
├── __init__.py
├── conftest.py
├── test_simulate_with_coefficients.py
├── test_fit_swing_mujoco.py
├── test_synthesize_target.py
├── test_torque_driver.py
└── test_model_builder.py
```

The package lives under `src/engines/physics_engines/mujoco/motion_matching/` (not `python/mujoco_humanoid_golf/`) to keep the parity work isolated from the legacy launcher. The legacy package is left untouched; it can be migrated separately.

### 2.2 Canonical signatures

```python
# simulate_with_coefficients.py
from src.shared.python.motion_matching.club_target import AlignOptions
from src.engines.physics_engines.mujoco.motion_matching._motion_opt_types import SimOutput

def simulate_with_coefficients(
    theta: np.ndarray,            # shape (n_joints * 7,)
    opts: SimOptions,             # MuJoCo-specific sim options (timestep, T, model variant)
) -> SimOutput:
    """Run the MuJoCo forward model with polynomial torque coefficients.

    Output (matches CROSS_ENGINE_PARITY_SPEC §2.2):
        SimOutput(
            time:      (N,)   float64,
            grip:      (N,3)  float64,    # mid-hands position, world frame, metres
            grip_quat: (N,4)  float64,    # [w,x,y,z]
            clubhead:  (N,3)  float64,
            club_quat: (N,4)  float64,
            tau:       (N,J)  float64,    # joint torques
            omega:     (N,J)  float64,    # joint angular velocities
        )
    """
```

`SimOutput` mirrors the Python `compute_cost` `SimOutput` dataclass already
in `src/shared/python/motion_matching/cost.py` — re-export it from the engine
package; do **not** redefine.

```python
# fit_swing_mujoco.py
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import CostOptions

def fit_swing_mujoco(
    target: ClubTarget,
    opts: FitOptions,             # holds CostOptions, SimOptions, scipy minimizer kwargs
) -> FitResult:
    """Fit polynomial torque coefficients θ to a measured club trajectory.

    Returns FitResult mirroring §1 of CODING_STANDARDS.md provenance block:
      coefficients, final_rmse_m, final_total_work_J, solver,
      solver_options, target_hash, git_commit, mujoco_version,
      duration_s, timestamp_utc.
    """
```

```python
# synthesize_target.py
def synthesize_target_from_coefficients(
    theta: np.ndarray,
    opts: AlignOptions,
) -> ClubTarget:
    """TDD oracle: run the model with known θ, emit a canonical ClubTarget.

    This function REPLACES the NotImplementedError stub at
    src/shared/python/motion_matching/loaders/synthetic.py — but does so by
    importing this engine's simulate_with_coefficients. The shared loader
    becomes a thin dispatcher keyed on opts.engine.
    """
```

### 2.3 Polynomial-torque driver — Stateflow analogue in MuJoCo

In Simscape, a Stateflow chart evaluates `τ_j(t) = Σ_{k=0..6} θ[j,k] · t^k` per joint at each major step. In MuJoCo, the equivalent is the **`mjcb_control` callback**:

```python
# _torque_driver.py
import mujoco
import numpy as np

class PolynomialTorqueDriver:
    """Per-joint 6th-order polynomial torque, applied via mj_step's control callback.

    θ has shape (n_joints, 7); θ[j,k] is the coefficient of t^k for joint j.
    """
    def __init__(self, model, theta: np.ndarray, t0: float = 0.0):
        if theta.shape != (model.nu, 7):
            raise ValueError(f"theta must be ({model.nu}, 7); got {theta.shape}")
        self._theta = theta.astype(np.float64).copy()
        self._t0 = t0

    def install(self, model, data):
        """Register self as mjcb_control for this (model, data)."""
        def cb(_m, d):
            t = d.time - self._t0
            powers = np.array([t**k for k in range(7)])    # (7,)
            d.ctrl[:] = self._theta @ powers               # (n_joints,)
        mujoco.set_mjcb_control(cb)

    def uninstall(self):
        mujoco.set_mjcb_control(None)
```

Each MJCF must declare `<actuator><motor joint="..."/></actuator>` for every
controlled joint so `data.ctrl` directly maps to torque. The full-body model
needs ~17 motors. The callback is reset between `simulate_with_coefficients`
calls; this is critical for thread-safety (`mjcb_control` is process-global).

**Concurrency note:** parallel fits must use `multiprocessing` (not threads)
because `mjcb_control` is global. `fit_swing_mujoco` does not parallelise on
its own; the dataset-sweep harness in `src/shared/python/motion_matching/dataset/sweep.py`
is responsible for the process pool.

---

## 3. Body model

### 3.1 Recommendation: **use the existing `_golf_swing_full_body_xml.py` after the gravity fix**, with a deprecation path to the YAML-driven build

**Why not myosuite?** `myo_sim/body/myobody.xml` is gorgeous — full musculotendon
units, ~50 actuated joints — but (a) musculotendon dynamics ≠ the polynomial
torque driver the cost function is built around, (b) the asset includes are
broken in this layout (mesh and texture paths resolve to `../src/shared/models/myosuite/myo_sim/myo_sim/...`),
and (c) it's parametrically far from Simscape's `golf_3D_full.slx`, which would
make cross-engine numerical comparison meaningless.

**Why not generate from PARITY-DIMENSIONS YAML now?** The YAML doesn't exist
yet (it's a sibling spec). When it lands, `_model_builder.py` should prefer it
and emit the MJCF deterministically. Until then, fix the existing XML.

**Recommendation:**

1. Fix the gravity bug in all three `_golf_swing_*_xml.py` files: cast
   `GRAVITY_M_S2` (and the other `PhysicalConstant`s) to `float()` at module
   load. One-line change. Add an import-time test that compiles each variant.
2. Promote `_golf_swing_full_body_xml.py` as the **default** for the parity
   work — it has feet → torso → arms → club at roughly Simscape parity (~17
   DOF).
3. Add `<actuator><motor joint="…" gear="1"/></actuator>` blocks for every
   joint that the polynomial driver controls. This is the largest concrete
   MJCF change (~30 lines added per variant).
4. Defer `_model_builder.py::from_yaml(path)` to a later PR (PARITY-DIMENSIONS
   dependency); ship a `from_string(name: Literal["upper", "full", "advanced"])`
   for now.

### 3.2 Club attachment

The Simscape club is attached at the grip via a 6-DOF locked joint with
`mass = 0.205 kg`, `inertia = diag(2e-5, 2e-3, 2e-3) kg·m²` for a driver.

In MJCF, the equivalent is:

```xml
<body name="club_grip" pos="0 0 0">
  <inertial pos="0 0 0.5" mass="0.205"
            diaginertia="2e-3 2e-3 2e-5"/>
  <geom name="grip" type="capsule" fromto="0 0 0 0 0 0.15"
        size="0.015" material="grip_mat"/>
  <body name="club_head" pos="0 0 1.07">
    <inertial pos="0 0 0" mass="0.190" diaginertia="2e-4 2e-4 1e-4"/>
    <geom name="head" type="box" size="0.05 0.025 0.04" material="head_mat"/>
  </body>
</body>
```

attached **without a joint** to the right hand. The grip is the rigid contact
in the same sense as the Simscape `weldjoint` between the right hand and the
butt of the club. (See CLUB_IK_SPEC.md §"Why grip-primary?" — the grip is the
PRIMARY anchor.)

Mass & inertia values **must** come from
`src/shared/python/motion_matching/club_configurations.py` (or the equivalent
constants module). Do **not** hard-code per-variant.

---

## 4. Implementation plan

Six issues, sized for one PR each. Issue bodies live in `MUJOCO_ISSUES.md`.

| #              | Title                                                      | Size | Depends on                     |
| -------------- | ---------------------------------------------------------- | ---- | ------------------------------ |
| ISSUE-MUJOCO-1 | Fix gravity-constant bug in MJCF generators                | XS   | —                              |
| ISSUE-MUJOCO-2 | `_model_builder.py` — load + cache compiled MJCF           | S    | ISSUE-MUJOCO-1                 |
| ISSUE-MUJOCO-3 | `simulate_with_coefficients.py` + polynomial-torque driver | M    | ISSUE-MUJOCO-2                 |
| ISSUE-MUJOCO-4 | `fit_swing_mujoco.py` — canonical fit driver               | M    | ISSUE-MUJOCO-3, PARITY-COST    |
| ISSUE-MUJOCO-5 | `synthesize_target_from_coefficients` engine impl          | S    | ISSUE-MUJOCO-3, PARITY-LOADERS |
| ISSUE-MUJOCO-6 | `viz/render_swing.py` — thin renderer                      | S    | ISSUE-MUJOCO-3                 |

Total: ~1 XS, 3 S, 2 M. Roughly 2 weeks of single-developer work assuming
the cross-engine and dimensions specs land in parallel.

Acceptance gates per issue are listed in `MUJOCO_ISSUES.md`. The unifying
acceptance criterion for the spec as a whole is:

> Given a synthetic `ClubTarget` produced by `synthesize_target_from_coefficients(θ_truth)`,
> `fit_swing_mujoco` recovers `θ_truth` to within `‖θ_fit − θ_truth‖∞ < 1e-3`
> and `final_rmse_m < 1e-3` in under 0.5 s wall-clock on a single core.

---

## 5. TDD / DbC / DRY / LOD compliance

### 5.1 Replicating MATLAB's `arguments` block

MATLAB:

```matlab
function result = fit_swing_fmincon(target, opts)
    arguments
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time","butt","clubhead","club_quat","impact_idx"])}
        opts (1,1) struct = default_option1_options()
    end
```

Python:

```python
from dataclasses import dataclass
from src.shared.python.core.contracts.decorators import precondition, postcondition
from src.shared.python.motion_matching.club_target import ClubTarget

@dataclass(frozen=True)
class FitOptions:
    cost: CostOptions = field(default_factory=CostOptions)
    sim:  SimOptions  = field(default_factory=SimOptions)
    minimizer: MinimizerOptions = field(default_factory=MinimizerOptions)
    rng_seed: int = 0

@precondition(lambda target, opts: isinstance(target, ClubTarget),
              "target must be a ClubTarget (CLUB_IK_SPEC.md schema)")
@postcondition(lambda result: result.final_rmse_m >= 0.0,
               "RMSE must be non-negative")
@postcondition(lambda result: np.isfinite(result.final_rmse_m),
               "RMSE must be finite")
def fit_swing_mujoco(target: ClubTarget, opts: FitOptions) -> FitResult:
    ...
```

`ClubTarget.__post_init__` already runs the schema/validation checks at
construction (`src/shared/python/motion_matching/club_target.py:75–77`), so
`isinstance(target, ClubTarget)` covers the MATLAB `mustHaveFields` check —
no need to re-implement it. **DRY.**

`pydantic` is _not_ required here; `dataclasses(frozen=True)` plus the
`precondition` / `postcondition` decorators from
`src.shared.python.core.contracts.decorators` already ship and are used
throughout `motion_matching/`. Adding pydantic would be a new dep with no
benefit.

### 5.2 Sharing with `src/shared/python/motion_matching/`

| Component                                        | Lives in shared/             | Engine package re-exports                                           |
| ------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------- |
| `ClubTarget`, `AlignOptions`, `SourceProvenance` | yes (`club_target.py`)       | import — never redefine                                             |
| `CostOptions`, `compute_cost`, `SimOutput`       | yes (`cost.py`)              | import                                                              |
| `compute_total_work`                             | yes (`cost.py`)              | import                                                              |
| `quaternion_geodesic_angles`                     | yes (`_geodesic.py`)         | import                                                              |
| Excel / C3D loaders                              | yes (`loaders/`)             | engine **never** loads files; consumes `ClubTarget`                 |
| Synthetic loader dispatch                        | yes (`loaders/synthetic.py`) | engine **registers** its `synthesize_target_from_coefficients` impl |
| `simulate_with_coefficients`                     | **engine-specific**          | lives at `src/engines/physics_engines/mujoco/motion_matching/`      |
| Polynomial driver                                | **engine-specific**          | same                                                                |
| `fit_swing_*` driver                             | **engine-specific**          | same                                                                |

The dispatcher pattern: `loaders/synthetic.py` becomes:

```python
def synthesize_target_from_coefficients(theta, opts):
    backend = opts.engine  # Literal["simscape","mujoco","drake","pinocchio","opensim"]
    if backend == "mujoco":
        from src.engines.physics_engines.mujoco.motion_matching.synthesize_target \
            import synthesize_target_from_coefficients as impl
        return impl(theta, opts)
    ...
```

This avoids hard-importing every engine at package load (mujoco, drake, etc.
are heavy deps the GUI shouldn't pull in).

### 5.3 LOD enforcement

The 2-level cap in `CLAUDE.md` and `CODING_STANDARDS.md` rules out chains
like `result.solver_options.minimizer.scipy_kwargs["maxiter"]`. Add a
delegating accessor on `FitOptions`:

```python
@property
def maxiter(self) -> int:
    return self.minimizer.scipy_kwargs.get("maxiter", 200)
```

Same pattern in `simulate_with_coefficients` for `model.opt.timestep` etc.
(MuJoCo's `mjModel` object naturally chains; wrap critical reads in
properties on `SimOptions` / `_model_builder.CompiledModel`).

### 5.4 TDD harness

Each public function gets a test in the same PR. Tests under
`tests/motion_matching/mujoco/`:

```python
@pytest.mark.unit
def test_simulate_with_coefficients_zero_torque_falls_under_gravity():
    theta = np.zeros((NJ, 7))
    out = simulate_with_coefficients(theta.flatten(), SimOptions())
    assert out.grip[-1, 2] < out.grip[0, 2]   # gravity pulls grip down

@pytest.mark.unit
def test_synthesize_then_fit_recovers_theta():
    rng = np.random.default_rng(0)
    theta_truth = rng.uniform(-1.0, 1.0, size=(NJ, 7))
    target = synthesize_target_from_coefficients(theta_truth.flatten(),
                                                  AlignOptions())
    result = fit_swing_mujoco(target, FitOptions())
    assert result.final_rmse_m < 1e-3
    assert np.linalg.norm(result.coefficients - theta_truth.flatten(),
                          ord=np.inf) < 1e-2
```

The synth-then-fit test is the **single most important** acceptance gate; if
it fails the optimizer is broken.

---

## 6. Performance baseline + targets

### 6.1 Measured today

A 2-DOF hand-written MJCF probe (closest thing that compiles given the
gravity bug):

```
nv=2, dt=2.0 ms, 0.3 s = 150 mj_step → 1.0 ms wall-clock
compile (MjModel.from_xml_string) → 2.3 ms
```

That's `~6.7 µs/step`. Extrapolated linearly to 17-DOF full-body model
(the cost is super-linear in `nv` for `factor_M` but at this scale dominated
by `mj_step` constant factors), expect **~2–5 ms per 0.3 s swing** for
forward sim plus ~10 ms model-compile (one-shot per fit).

The legacy `motion_optimization.SwingOptimizer` adds a PD-tracking inner
loop (kp=100, kd=20) and computes Jacobians per step for objective
evaluation; in profiling on a comparable model this ran ~30 ms per swing.
With ~200 fmincon iterations that's **~6 s per fit, warm**.

> **No measurement of the full pipeline today** because (a) the MJCF
> generators are broken and (b) `synthesize_target_from_coefficients` is a
> stub. The 6 s number is from logs in PR #4045 / #4067 against a hand-fixed
> XML. Treat as ±2× until ISSUE-MUJOCO-1 lands and we can re-measure.

### 6.2 Targets

| Metric                           | Today (estimated) | Target      | How                                                                                                                                                                                                                                            |
| -------------------------------- | ----------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Forward-sim wall-clock per swing | 2–5 ms            | < 5 ms      | Hold the line; no actuator network needed                                                                                                                                                                                                      |
| Cost evaluation per swing        | ~1 ms             | < 1 ms      | Pure NumPy in `compute_cost`                                                                                                                                                                                                                   |
| One fit (200 iter SQP)           | ~6 s              | **< 0.5 s** | (1) cache `MjModel` across iterations, (2) reuse `MjData` (don't realloc), (3) batch-evaluate finite-difference gradients across `n_proc` workers, (4) optionally swap fmincon-SQP for L-BFGS-B which has better step quality at low dimension |
| 100-swing dataset sweep          | not yet possible  | < 60 s      | `multiprocessing.Pool(n=8)` over a process-pool that holds compiled `MjModel` per worker                                                                                                                                                       |

**Why < 0.5 s is realistic:** Simscape's 7 s/swing baseline is dominated by
Simulink's compile-once-per-call overhead. MuJoCo's overhead is ~10 ms compile
and ~3 ms sim per swing. Even with 200 SQP iterations × (1 sim + 7·F-D for
gradient) = 1600 sims at 3 ms each = 4.8 s — to hit < 0.5 s we **must** use
analytic gradients via `mjd_transitionFD` (MuJoCo's built-in finite-difference
helper) + a smoothed, low-iter-count solver, or warm-start from a CVAE prior
(see `src/shared/python/motion_matching/inverse/cvae.py`).

### 6.3 MuJoCo features that hit those targets

- **`mjd_transitionFD`** — MuJoCo's internal finite-difference routine for
  state-transition Jacobians; replaces scipy's per-iteration FD pass.
- **`mujoco.MjData(model)` reuse** — never re-allocate; just `mj_resetData`.
- **`mjcb_control` global callback** — already discussed; keeps the inner
  loop in C.
- **`compile_mjs`** is _not_ relevant here — that's MuJoCo's MJS scenegraph
  authoring API, not a JIT.
- **GPU offload** — MuJoCo MJX (the JAX port) gives ~100× speedup on parallel
  rollouts but compiling MJX is non-trivial and its body-model coverage is
  thinner than the C kernel. **Defer**; revisit if ISSUE-MUJOCO-4 misses
  the < 0.5 s target by > 2×.

---

## 7. Risks / open questions

### 7.1 Risks

1. **Model retopology may be required.** The Simscape full-body model has
   joint axes specified in body frame with non-orthogonal rotation orders
   (e.g. shoulder is a 3-DOF gimbal with specific Euler-angle order). Naive
   port to MJCF hinge stacks may give different quaternions for the same
   joint values, breaking grip-quat parity. Mitigation: PARITY-DIMENSIONS
   YAML must specify joint axes in a frame-independent form, and the
   MJCF builder must apply the same convention.

2. **`mjcb_control` is process-global.** Any parallel fit needs
   `multiprocessing`, not threading. Caller-facing helpers must document
   this. There is no per-`MjData` callback in the public C API.

3. **The polynomial torque driver is an open-loop input** — there's no
   feedback at all. Joint limits aren't enforced unless explicitly added
   via MJCF `<joint limited="true" range="..."/>` and the optimizer's
   bounds. Without limits, large-θ fits can drive joints past plausible
   ranges. Mitigation: keep the same `[lb, ub]` bounds from
   `build_coefficient_bounds` (Simscape side) and add a soft-penalty term
   in `compute_cost`'s regularizer for joint-limit violation.

4. **Quaternion sign flip in the grip-quat target.** MuJoCo emits
   continuous quaternions from `mj_step`, but the source xlsx has 3×3
   rotation matrices that may have ambiguous sign at the first frame.
   `compute_cost` already uses `2·acos(|q1·q2|)` (`_geodesic.py`) — verify
   this is also true on the engine-emitted side. Add a regression test.

5. **The legacy `SwingOptimizer` will rot.** Once `fit_swing_mujoco` lands,
   the GUI launchers in `golf_suite_launcher.py` /
   `humanoid_launcher.py` still call the old optimizer. They should
   migrate but that's out of scope for this spec; flag as
   `MUJOCO-FOLLOWUP`.

### 7.2 Open questions

- **Should the polynomial-torque driver be 6th-order, 7-coefficient, or
  something else?** Simscape uses 7-coefficient. MuJoCo has no constraint
  on this. Recommendation: match Simscape exactly so the same θ vector
  cross-validates across engines.
- **Time horizon T.** Simscape runs 0.3 s. The MJCF model's natural settling
  time depends on damping/armature in the joint defaults. Confirm that 0.3 s
  is enough on the MuJoCo side or document the deviation.
- **MJX path.** Should the spec mandate an MJX-compatible MJCF? MJX has
  a stricter subset (no general meshes, limited solver options). Going
  MJX-first would constrain the model more than necessary today. Recommend
  **defer**.

### 7.3 License / dependency concerns

- **MuJoCo:** Apache-2.0 (since 2.1.5). No concern.
- **myosuite assets:** Apache-2.0. The asset path issue (§1.3) is mechanical,
  not legal. This spec recommends _not_ using myosuite for the parity work
  (§3.1) so the issue is moot for now.
- **scipy.optimize:** BSD-3. `differential_evolution` exists; SQP via
  `scipy.optimize.minimize(method="SLSQP")`. No new deps.
- **pydantic:** explicitly **not** required (§5.1).
- **mujoco-py (the legacy 2.x bindings):** **avoid**. The current package
  uses `mujoco` (the 3.x bindings) which ships its own pip wheel with no
  binary licence concerns.

---

## Appendix A — Cross-references

- Cross-engine spec: `src/engines/CROSS_ENGINE_PARITY_SPEC.md` (in flight on
  `feat/cross-engine-parity-spec`)
- Cost spec: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/COST_FUNCTION_SPEC.md`
- Club-IK spec: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CLUB_IK_SPEC.md`
- Coding standards: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CODING_STANDARDS.md`
- Simscape reference fit driver: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option1_direct_optimization/fit_swing_fmincon.m`
- Sibling parity spec branches: `feat/drake-parity-spec`, `feat/pinocchio-parity-spec`, `feat/opensim-parity-spec`
