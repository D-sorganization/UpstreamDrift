# EPIC: `bioptim` Optimal-Control Backend for UpstreamDrift

**Status:** Proposed — 2026-09-07 · **Tracking issue:** #9762
**Prerequisite fixes:** #9755 (inertials), #9756 (FD transcription), #9757 (MAP sentinel), #9758 (identifiability gate), #9759 (CI coverage), #9760 (backend registry + ADR), #9761 (upstream bioptim PR)
**Decision:** GO, narrowly scoped. Adopt `pyomeca/bioptim` as an opt-in optimal-control
transcription layer, driven by UpstreamDrift's _own_ CasADi dynamics through bioptim's
custom-model protocol. **Do not** adopt biorbd, bioviz, or bioptim's `PinocchioModel`.
**Owner:** Dieter · **Agents:** any (Jules/Claude Code/Codex) — follow the lease protocol in `CLAUDE.md`.

---

## 0. Executive Summary (Read This if Nothing Else)

### What `bioptim` Is

A CasADi-based optimal-control-problem (OCP) framework from the S2M lab (Université de
Montréal). Direct multiple shooting and direct collocation, IPOPT / FATROP / ACADOS
interfaces, a large penalty library (track markers, minimize torque/qddot, COM, phase
transitions), multiphase problems, time-invariant parameters as decision variables
(simultaneous state + parameter estimation), moving-horizon estimation, multi-start, and
stochastic OCP. MIT licence. Latest tag `Release_3.4.0` (2025-11-25); `master` is
`3.5.0-dev` with commits through 2026-07-26. ~120 stars, 77 open issues, 5–6 regular
contributors. Breaking API changes in every recent minor release.

### Why It Is Worth Integrating

UpstreamDrift already has three swing optimizers and two estimators, and every one of
them has a gap that bioptim closes for free:

| Existing module                                                 | Gap                                                                                                                                                                                                                                                                             | What bioptim gives                                                                                                                                 |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optimization/casadi_backend.py`                                | Not a true transcription: velocities/accelerations are central finite differences on a coarse node grid; dynamics are never enforced _between_ nodes. Uses **placeholder inertials** (`_LINK_MASS = 1.0`, `_LINK_INERTIA = 1e-2`), so torque limits are physically meaningless. | Real multiple-shooting / collocation with RK4/IRK/collocation ODE solvers; the OCP is correct by construction.                                     |
| `estimation/map_estimator.py`                                   | `scipy.optimize.least_squares` with a `NON_FINITE_RESIDUAL_SENTINEL = 1e12` hack — a symptom of finite-difference Jacobians through an unstable forward model.                                                                                                                  | Exact AD gradients/Hessians from CasADi; sparsity exploited by IPOPT; `ParameterList` makes segment lengths/masses first-class decision variables. |
| `estimation/moving_horizon.py`                                  | Hand-rolled window management over the same scipy solver.                                                                                                                                                                                                                       | `RecedingHorizonOptimization` with warm start, tested upstream.                                                                                    |
| `optimization/crocoddyl_backend.py`                             | Single-phase DDP; dual-libpinocchio hazard; platform-uneven wheels.                                                                                                                                                                                                             | Keep it. DDP is complementary (fast, unconstrained-ish). bioptim is for constrained, multiphase, tracking, and estimation problems.                |
| markerless video → `motion_pipeline.contracts.KeypointSequence` | No optimal-estimation path from keypoints to dynamically consistent (q, qdot, tau).                                                                                                                                                                                             | `ObjectiveFcn.Lagrange.TRACK_MARKERS` + torque-driven dynamics = the standard "optimal estimation" workflow, with a worked upstream example.       |

### The Blocker, and How It Was Cleared

bioptim, biorbd, and CasADi-enabled Pinocchio are **conda-forge only**. UpstreamDrift is
pip-first (`pyproject.toml` is the single source of truth; `environment.yml` is a pip
wrapper; Docker installs `pin` from PyPI). Verified today:

- PyPI `pin==4.1.0` **does not ship** `pinocchio.casadi` → bioptim's `PinocchioModel` is
  unusable here, and mixing conda-forge pinocchio with the `pin` wheel is the same
  dual-`libpinocchio` crash documented in `crocoddyl_backend.py`.
- bioptim's core imports `biorbd_casadi` in exactly **two non-model places**, both trivial:
  a type hint in `models/protocols/holonomic_biomodel.py:3` and a version string in
  `optimization/optimal_control_program.py:341`. With a 6-line stub module standing in for
  `biorbd_casadi`, `import bioptim` succeeds and the upstream `custom_model` example
  **solves to convergence in 0.5 s** with `casadi==3.6.7` from PyPI (IPOPT+MUMPS bundled
  in the casadi wheel). No conda anywhere.

So the integration vehicle is: **bioptim (git-pinned) + PyPI casadi + a UD-owned CasADi
symbolic swing model implementing bioptim's `StateDynamics` protocol.** Pinocchio stays
the _numeric validation oracle_ (as it already is for `build_symbolic_rnea`), not a
symbolic dependency.

### Non-Goals

- No biorbd `.bioMod` files, no bioviz, no pyorerun.
- No muscle-driven dynamics (bioptim muscle models are biorbd-only).
- No replacement of `swing_optimizer.py` (scipy flagship) or the crocoddyl backend in this
  epic; consolidation is a _later_ decision informed by Phase 2 parity numbers.
- No GUI work. Headless API + tests + one example only.

---

## 1. Architecture

```
src/shared/python/optimization/
├── casadi_backend.py            (existing; symbolic RNEA kept, transcription superseded in Phase 6)
├── crocoddyl_backend.py         (existing; untouched)
└── ocp/                         ← NEW subpackage (all bioptim code lives here)
    ├── __init__.py              lazy exports; never imports bioptim at module import time
    ├── _compat.py               bioptim availability probe + biorbd_casadi shim installer
    ├── symbolic_model.py        SymbolicSwingModel: CasADi Functions (rnea, crba, fd, fk, markers)
    │                            built from GolferModel/ClubModel with REAL inertials
    ├── bioptim_model.py         SwingBioModel(StateDynamics): the bioptim protocol adapter
    ├── swing_ocp.py             build_max_speed_ocp(...) → OptimalControlProgram
    ├── tracking_ocp.py          build_tracking_ocp(keypoints, ...) → OCP (optimal estimation)
    ├── parameter_ocp.py         add_parameter_block(ocp, specs) — simultaneous estimation
    ├── mhe.py                   receding-horizon wrapper matching estimation.moving_horizon API
    ├── result.py                OcpSolution → CasadiSwingResult / MapEstimatorResult adapters
    └── tests/                   colocated tests, all marked `requires_bioptim`
```

**Dependency direction (enforced by an import-linter contract, see Phase 0):**
`ocp/` may import from `optimization/`, `estimation/`, `motion_pipeline.contracts`,
`simulation_backends.provenance`. Nothing outside `ocp/` may import `bioptim` directly.

**Extras in `pyproject.toml`:**

```toml
optimal-control = ["casadi>=3.6.0,<3.7.0"]   # tighten ceiling: master uses casadi.MX_eye, removed in 3.8
bioptim = [
  "upstream-drift[optimal-control]",
  "bioptim @ git+https://github.com/pyomeca/bioptim.git@<SHA>",   # pin a SHA, never a branch
  "matplotlib>=3.8",
]
```

`tkinter` is imported at module scope by `bioptim/gui/plot.py`; document that Debian/Ubuntu
CI images need `python3-tk` (add to `Dockerfile.heavy_test` and the optional-stack workflow).

---

## 2. Phases and Tickets

Each ticket lists: **Goal · Files · Steps · Acceptance · Pitfalls · Size**. Tickets within a
phase are independent unless a `depends:` line says otherwise. Sizes: S ≤ 200 LOC, M ≤ 600,
L > 600 (incl. tests).

### Phase 0 — Guardrails and Spike (Must Merge Before Anything Else)

#### 0.1 Pin, Extra, Shim, Smoke Test — Size S

- **Goal:** `pip install -e '.[bioptim]'` works on a clean Ubuntu venv; `import bioptim`
  succeeds without conda.
- **Files:** `pyproject.toml`, `src/shared/python/optimization/ocp/_compat.py`,
  `src/shared/python/optimization/ocp/__init__.py`, `tests/conftest.py` (marker),
  `Makefile` (`make sync-deps` regenerates `environment.yml`).
- **Steps:**
  1. Choose the SHA: start from `Release_3.4.0` (verified free of `MX_eye`, so casadi 3.6–3.8 work). If master features are needed later, re-pin deliberately.
  2. `_compat.py`:
     ```python
     def bioptim_available() -> bool          # find_spec("bioptim") is not None, mock-tolerant like casadi_available()
     def require_bioptim():                    # installs the biorbd_casadi shim into sys.modules IF biorbd_casadi
                                               # is not importable, THEN imports bioptim; raises BioptimNotAvailableError with install hint
     ```
     Shim = a `types.ModuleType("biorbd_casadi")` with `__version__ = "1.12.0"` and a
     `__getattr__` returning a permissive dummy class. Comment _why_ (the two upstream call
     sites) and link ticket 0.4.
  3. Add pytest marker `requires_bioptim` to `[tool.pytest.ini_options].markers` mirroring `requires_casadi`.
  4. Smoke test `ocp/tests/test_compat.py`: import succeeds; solving bioptim's own
     `toy_examples/custom_model` pendulum reaches `sol.status == 0` and `q[-1] ≈ 3.14`.
- **Acceptance:** CI job (0.2) green; `pytest -m requires_bioptim` passes locally in < 60 s.
- **Pitfalls:** set `MPLBACKEND=Agg` in tests; never call `sol.graphs()`/`sol.animate()`.
  bioptim prints IPOPT output — pass `Solver.IPOPT(show_online_optim=False)` and set
  `solver.set_print_level(0)` in tests.

#### 0.2 CI Job — Size S

- **Goal:** bioptim tests run in `ci-optional-stack.yml` alongside the crocoddyl probe.
- **Steps:** new matrix leg `bioptim`; `apt-get install python3-tk`; `pip install -e '.[bioptim]'`;
  `pytest -m requires_bioptim --timeout=600`. Same fail-soft summary pattern as the
  Pinocchio leg (report "unavailable" rather than fail if install itself fails).
- **Acceptance:** job appears in the step summary; 0.1 smoke test runs in it.

#### 0.3 Import-Linter Contract — Size S

- **Goal:** prevent `bioptim` leaking into the rest of `src/`.
- **Files:** wherever the repo's existing import-linter / architecture tests live (`tests/architecture/`).
- **Acceptance:** a test fails if any module outside `optimization/ocp/` imports `bioptim`.

#### 0.4 Upstream PR to `pyomeca/bioptim` — Size S (External, #9761)

- **Goal:** make `biorbd_casadi` optional upstream so the shim can be deleted.
- **Steps:** (a) `holonomic_biomodel.py`: guard the import with `TYPE_CHECKING`; (b)
  `optimal_control_program.py`: `try: import biorbd_casadi ... except ImportError: version = None`;
  (c) `models/biorbd/biorbd_model.py` `check_version` only when importable; (d) lazy-import
  `tkinter` in `gui/plot.py`. Add a CI leg upstream that runs the custom_model test without biorbd.
- **Acceptance:** PR opened and linked here. Not a merge blocker for this epic.

---

### Phase 1 — Symbolic Swing Model With Real Inertials

#### 1.1 `SymbolicSwingModel` — Size M

- **Goal:** one place that turns `GolferModel` + `ClubModel` into CasADi `Function`s with
  the **same inertials the URDF bridge emits**, replacing the `_LINK_MASS=1.0` placeholders.
- **Files:** `ocp/symbolic_model.py`; read `optimization/model_provider.py`,
  `motion_pipeline/model_bridge.py`, `optimization/casadi_backend.py` (reuse
  `build_symbolic_rnea` structure — do not copy-paste; refactor it to accept per-link
  `(mass, com, inertia)` and call it from both places).
- **Functions to expose (all `casadi.Function`, SX):**
  - `rnea(q, v, a) -> tau`
  - `mass_matrix(q) -> M` via CRBA-by-RNEA: column _i_ of M = `rnea(q, 0, e_i) - rnea(q, 0, 0)`
  - `nonlinear_effects(q, v) -> h` = `rnea(q, v, 0)`
  - `forward_dynamics(q, v, tau) -> qddot` = `casadi.solve(M, tau - h)` (7 DOF — dense solve is fine)
  - `fk(q) -> {joint_name: 4x4}`, `clubhead_position(q)`, `clubhead_velocity(q, v)` (jacobian-times-v, symbolic)
  - `markers(q) -> 3 x n_markers` where the marker set = joint centres + clubhead + any
    virtual markers needed to map `motion_pipeline.contracts.Keypoint` names. Store `marker_names: tuple[str, ...]`.
- **Parameterisation:** constructor takes an optional `parameters: dict[str, SX]` (segment
  lengths, masses). When present, geometry/inertials are built from those symbols so
  Phase 4 can hand them to bioptim's `ParameterList`. When absent, use numeric values.
- **Acceptance (tests, `requires_pinocchio` + `requires_casadi`):**
  - `rnea` matches `pin.rnea` on the bridge URDF to 1e-9 at 50 random (q,v,a).
  - `mass_matrix` matches `pin.crba` (upper-triangular fill) to 1e-9.
  - `forward_dynamics` matches `pin.aba` to 1e-8.
  - `markers` match Pinocchio frame placements for the same names.
  - The **old** `casadi_backend.build_symbolic_rnea` test still passes (refactor is behaviour-preserving when given the old placeholder inertials).
- **Pitfalls:** `ca.solve` on SX works; if you switch to MX for speed, use `ca.solve(M, rhs, "symbolicqr")`. Keep everything SX unless profiling says otherwise — bioptim custom models default to MX; see 1.2 for the conversion.

#### 1.2 `SwingBioModel` — `bioptim` Protocol Adapter — Size M, Depends: 1.1

- **Goal:** implement `bioptim.StateDynamics` so bioptim can drive `SymbolicSwingModel`.
- **Files:** `ocp/bioptim_model.py`. Template: `bioptim/examples/toy_examples/custom_model/custom_package/my_model.py` and the 3.4.0 migration notes (properties `name`, `name_dofs`, `state_configuration_functions`, `control_configuration_functions`, `algebraic_configuration_functions`, `extra_configuration_functions` are **required** since 3.4.0).
- **Required surface (minimum):**
  - `name`, `name_dofs` (= `optimization._swing_kinematics.JOINTS`), `nb_q/nb_qdot/nb_tau`
  - `state_configuration_functions -> [States.Q, States.QDOT]`, `control_configuration_functions -> [Controls.TAU]`
  - `dynamics(time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp) -> DynamicsEvaluation(dxdt=vertcat(qdot, fd(q,qdot,tau,params)), defects=None)`
  - `forward_dynamics()`, `inverse_dynamics()` returning `casadi.Function` with the
    signature bioptim's penalties call: `(q, qdot, tau, parameters)` — check
    `bioptim/limits/penalty.py` for exact call sites of `controller.model.markers()(controller.q, controller.parameters.cx)` etc.
  - `markers()`, `marker_names`, `nb_markers`, `marker_index(name)`, `markers_velocities()`
    (needed by `TRACK_MARKERS`, `MINIMIZE_MARKERS_VELOCITY`)
  - `center_of_mass()`, `center_of_mass_velocity()` (for `MINIMIZE_COM_*`)
  - `tau_max()` returning per-joint bounds from `GolferModel` torque limits
  - `bounds_from_ranges()` if used by `BoundsList("q", ...)` helpers — otherwise construct `Bounds` explicitly from `_swing_constraints.py` joint ranges
  - `copy()`, `serialize()` (bioptim pickles solutions; return `(SwingBioModel, {"golfer": ..., "club": ...})`)
- **Acceptance:** a torque-driven OCP "swing from address pose to any pose minimising ∫τ²"
  builds and converges with `OdeSolver.RK4()`, `n_shooting=20`, `final_time=1.0`.
  Second test with `OdeSolver.COLLOCATION(polynomial_degree=3)` also converges.
- **Pitfalls:** bioptim asserts on free symbolic variables inside `Function`s — every
  `Function` must list _all_ its inputs, including the (possibly empty) `parameters` SX.
  `PhaseDynamics.SHARED_DURING_THE_PHASE` is the fast path; use it.

---

### Phase 2 — Max-Clubhead-Speed Swing OCP (Parity Target: `casadi_backend`)

#### 2.1 `build_max_speed_ocp` — Size M, Depends: 1.2

- **Goal:** same problem `casadi_backend.solve_swing_casadi` solves, as a real OCP.
- **Objective:** Mayer `-‖clubhead_velocity(q_T, v_T)‖²` (custom objective via
  `ObjectiveFcn.Mayer.CUSTOM`) + Lagrange `MINIMIZE_CONTROL("tau", weight=effort_w)` +
  smooth injury surrogate terms ported from `optimization/smooth_costs.py` as
  `ObjectiveFcn.Lagrange.CUSTOM`. Constraints: joint ranges as bounds, `tau` bounds from
  `tau_max`, address pose as initial-state bound, optional impact-plane constraint at the last node.
- **Two phases (optional flag):** backswing / downswing with a `PhaseTransitionFcn.CONTINUOUS`
  and free phase durations (`time_min/time_max` in `DynamicsOptions`). Start single-phase; add the flag only once single-phase parity is proven.
- **Return:** `CasadiSwingResult`-shaped object (`success, x, fun, message, iterations`) via
  `ocp/result.py` so `swing_bridge.SwingOptimizationBridge` can select the backend by name
  without new plumbing. Decision-vector layout must match `[angles.flatten(), velocities.flatten()]` on the node grid (resample from bioptim's time grid if `n_shooting != n_nodes`).
- **Acceptance:**
  - Converges from the same initial guess as `casadi_backend`; terminal clubhead speed ≥ the
    `casadi_backend` value on the benchmark golfer (record numbers in
    `docs/estimation/bioptim_parity.md`).
  - `swing_bridge` selects `backend="bioptim"` and produces a `SwingOptimizationResult`.
  - Provenance: attach `ProvenanceStamp` with bioptim SHA + casadi version (bioptim exposes `sol.bioptim_version_used`).
- **Pitfalls:** scale variables (`VariableScalingList`) — IPOPT stalls if τ is O(100) and q is O(1). Warm-start from the scipy flagship solution via `InitialGuessList(..., InterpolationType.EACH_FRAME)`.

#### 2.2 Benchmark + Parity Doc — Size S, Depends: 2.1

- **Files:** `benchmarks/` (existing pytest-benchmark pattern), `docs/estimation/bioptim_parity.md`.
- **Acceptance:** table of {scipy, casadi_backend, crocoddyl, bioptim-RK4, bioptim-collocation} × {clubhead speed, wall time, iterations, max dynamics defect}. The dynamics-defect column is the whole point: it will show `casadi_backend`'s finite-difference kinematics violate the ODE between nodes.

---

### Phase 3 — Keypoint Tracking OCP (Optimal Estimation)

#### 3.1 `build_tracking_ocp` — Size L, Depends: 1.2

- **Goal:** from a `motion_pipeline.contracts.KeypointSequence` (or `MarkerTrajectory`),
  recover dynamically consistent `(q, qdot, tau)` by tracking markers with torque-driven dynamics.
- **Inputs:** `KeypointSequence`, `GolferModel`, `ClubModel`, a `dict[keypoint_name -> marker_name]`
  (put the default map in `ocp/keypoint_map.py`; unmapped keypoints are ignored with a `structlog` warning), `weights: TrackingWeights` dataclass (marker weight, τ weight, qddot weight, per-marker confidence gating).
- **Formulation:** single phase, `n_shooting = n_frames - 1`, `final_time` from frame
  timestamps; `ObjectiveFcn.Lagrange.TRACK_MARKERS(target=..., weight=w_m, node=Node.ALL)`
  with `target` shaped `(3, n_markers, n_frames)` and NaN-confidence frames masked by
  per-node weight (`ObjectiveWeight` supports per-node arrays since 3.3.1); plus
  `MINIMIZE_CONTROL("tau")` and `MINIMIZE_STATE("qdot", derivative=True)` regularisers.
  Upstream template: `bioptim/examples/biomechanics/.../optimal_estimation` (search for `TRACK_MARKERS` in examples).
- **Initial guess:** run the existing kinematic fit (whatever `motion_pipeline` already does
  to get q from keypoints) and pass it frame-by-frame; without it IPOPT will wander.
- **Output:** `TrackingResult` (q, qdot, tau on the frame grid, per-marker RMS residual,
  solver status, provenance). Adapter to `estimation.map_estimator.MapEstimatorResult` in `ocp/result.py`.
- **Acceptance (synthetic round-trip, `requires_bioptim`):** generate ground truth with
  `estimation/synthetic_ground_truth.py`, project to markers, add 5 mm Gaussian noise +
  10 % dropped frames; recover q within 1° RMS and τ within 10 % RMS of truth. Compare
  against `map_estimator` on the same fixture and record in the parity doc.
- **Pitfalls:** unit/frame conventions — keypoints from video are usually camera frame,
  metres or pixels; convert to the rig's world frame before building `target`. Time
  stamps may be non-uniform; bioptim assumes uniform `dt` per phase — resample keypoints
  first (`signal_toolkit` has resamplers).

---

### Phase 4 — Simultaneous State + Parameter Estimation

#### 4.1 `add_parameter_block` — Size M, Depends: 1.1 (Parameterised Model), 3.1

- **Goal:** segment lengths / masses as `bioptim.Parameter`s so the tracking OCP estimates them jointly.
- **Mapping:** `estimation.map_estimator.SharedParameterSpec(name, initial, kind, lower, upper, prior, prior_scale, locked)` → `ParameterList.add(name, function=<setter on SwingBioModel>, size=1, scaling=...)` + `ParameterObjectiveList` quadratic prior `(p - prior)² / prior_scale²`. `locked` → not added (numeric).
- **Model side:** `SymbolicSwingModel(parameters={...})` builds geometry from the parameter SX; `SwingBioModel.parameters` exposes the stacked SX bioptim threads through `dynamics(..., parameters, ...)`.
- **Identifiability gate:** before solving, call `estimation/identifiability.py` on the
  marker Jacobian w.r.t. parameters at the initial guess; refuse (or auto-lock) parameters
  below the threshold and say so in the result. This is the check that stops the optimizer
  from "explaining" a marker offset with a 20 cm forearm.
- **Acceptance:** synthetic fixture with true forearm length perturbed 8 %; recovered
  length within 1 cm; q/τ accuracy as in 3.1; locked parameters unchanged bit-for-bit.
- **Pitfalls:** parameters enter every node's dynamics → Hessian density rises; use
  `Solver.IPOPT(...)` with `set_hessian_approximation("limited-memory")` for > 5 parameters and note it in the doc.

---

### Phase 5 — Moving-Horizon Variant

#### 5.1 `ocp/mhe.py` — Size M, Depends: 3.1

- **Goal:** drop-in alternative to `estimation.moving_horizon.MovingHorizonEstimator` using bioptim's `RecedingHorizonOptimization` / `MovingHorizonEstimator` classes.
- **Contract:** accept `MovingHorizonOptions` (window_size, step_size, latency_budget_ms) and emit `MovingHorizonResult` so the realtime consumers don't change. Latency check: if a window solve exceeds `latency_budget_ms`, return the previous window's tail and flag `degraded=True` (same semantics as the existing estimator — read it first).
- **Acceptance:** on the 3.1 synthetic fixture, windowed solution within 2° of the batch tracking solution; median window latency reported in the benchmark.
- **Pitfalls:** bioptim's MHE rebuilds nothing between windows only if `ocp` is constructed once and `update_function` mutates targets in place — do not rebuild the OCP per window (that was the 30 s/window trap in early bioptim MHE examples).

---

### Phase 6 — Consolidation and Documentation

#### 6.1 ADR — Size S, Depends: 2.2

- `docs/adr/00XX-bioptim-ocp-backend.md`: decision, alternatives (own `casadi.Opti` transcription ≈ 300 LOC; conda-forge stack; stay with scipy), consequences, the SHA-pin policy, and the explicit statement that Pinocchio is the numeric oracle and biorbd is out of scope.

#### 6.2 Route `casadi_backend` Transcription Through `ocp` — Size M, Depends: 2.2

- Keep `build_symbolic_rnea` (now inertial-correct) as the shared symbolic kernel; make
  `solve_swing_casadi` delegate to `build_max_speed_ocp` when bioptim is available and emit a
  `DeprecationWarning` on the finite-difference path. Remove the FD path one release later.
  **Do not** delete anything in this epic — deprecate only.

#### 6.3 Design-Manual + Governance — Size S

- Per `CLAUDE.md`: update `manuals/upstreamdrift` QMD calculation registry with the OCP
  formulations (objective, constraints, transcription), update `SPEC.md`/`AGENT_HANDOFF.md`,
  and run `python3 -m scripts.check_design_manual_governance`.

#### 6.4 Example — Size S

- `src/shared/python/optimization/examples/bioptim_tracking_example.py`: load a bundled
  compact swing dataset (`tests/test_load_compact_swing_dataset.py` shows how), run 3.1, print RMS. No plots.

---

## 3. Risks and Mitigations

| Risk                                                                | Likelihood | Mitigation                                                                                                                                                                                                     |
| ------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bioptim breaking API changes (every minor release has had them)     | High       | SHA pin; upgrade only via a ticket that re-runs Phase 0–3 tests; `ocp/` is the only place that knows bioptim's API.                                                                                            |
| Academic bus factor (2–3 core devs)                                 | Medium     | Everything of value (symbolic model, formulations) lives in UD code; bioptim is replaceable by a `casadi.Opti` rewrite of `swing_ocp.py`/`tracking_ocp.py` (~2 days) if abandoned.                             |
| Shim breaks when upstream touches biorbd imports                    | Medium     | 0.4 upstream PR removes the need; `test_compat.py` catches it on re-pin.                                                                                                                                       |
| `tkinter`/matplotlib in headless CI                                 | Low        | `python3-tk` + `MPLBACKEND=Agg`; never call plotting APIs.                                                                                                                                                     |
| casadi version drift (`MX_eye` removed in 3.8; master needs < 3.7)  | Medium     | ceiling in the extra; 0.1 records which casadi versions the pinned SHA was tested with.                                                                                                                        |
| Optimizer sprawl (now 5 backends)                                   | High       | 6.2 deprecation; ADR states which backend owns which problem class: scipy = quick/legacy, crocoddyl = DDP, bioptim = constrained/tracking/estimation.                                                          |
| Fixed-base 7-DOF model has no ground contact / two-hand closed loop | Known      | Same simplification the existing backends make; holonomic constraints in bioptim are biorbd-only, so a closed-loop grip would need a custom constraint via `ConstraintFcn.CUSTOM` — out of scope, note in ADR. |

---

## 4. Verification Commands (Copy Into PR Descriptions)

```bash
pip install -e '.[dev,pinocchio,bioptim]'
sudo apt-get install -y python3-tk                      # CI images
MPLBACKEND=Agg pytest -m "requires_bioptim" -q --timeout=600
pytest tests/architecture -q                            # import-linter contract (0.3)
pytest src/shared/python/optimization -m "requires_casadi and requires_pinocchio" -q   # 1.1 oracle tests
python -m scripts.check_design_manual_governance         # 6.3
ruff check . && black --check . && mypy src/shared/python/optimization/ocp
```

---

## 5. Evidence Log (What Was Actually Verified On 2026-09-07)

- `pyomeca/bioptim@master` (2026-07-26, `__version__ == "3.5.0"`): `pyproject.toml` declares
  `dependencies = []` and a `conda_only` extra (`biorbd>=1.12`, `pinocchio`, `pyqt`, `pyqtgraph`, `python-graphviz`). Not on PyPI (404 for `bioptim`, `biorbd`, `biorbd_casadi`).
- Third-party imports across `bioptim/` (excluding examples/tests): `acados_template`
  (lazy), `biorbd_casadi`, `casadi`, `matplotlib`, `numpy`, `packaging`, `scipy`, `tkinter`. No pyqt/graphviz at import time.
- Hard `biorbd_casadi` uses outside `models/biorbd/`: `models/protocols/holonomic_biomodel.py:3` (type hint), `optimization/optimal_control_program.py:4,341` (version string), `models/biorbd/biorbd_model.py:23` (`check_version`, needs a parseable `__version__`).
- `models/pinocchio/pinocchio_model.py` requires `pinocchio.casadi`; PyPI `pin==4.1.0` wheel lacks it (`ModuleNotFoundError`), and its `requires_dist` has no casadi extra.
- With a stub `biorbd_casadi` and `casadi==3.6.7` (PyPI): `import bioptim` in 0.7 s;
  `toy_examples/custom_model` pendulum swing-up, `n_shooting=30`, IPOPT status 0, cost 134.0, 0.5 s wall.
- `casadi==3.8.0` fails on `from casadi import MX_eye` in `bioptim/limits/constraints.py` (master); `Release_3.4.0` has no `MX_eye` usages.
- Custom-model protocol (from `my_model.py` and the 3.4.0 migration guide): required
  properties `name`, `name_dofs`, `state_configuration_functions`, `control_configuration_functions`, `algebraic_configuration_functions`, `extra_configuration_functions`; method `dynamics(...) -> DynamicsEvaluation`; penalties call `model.markers()(q, parameters)`, `model.markers_velocities()`, `model.center_of_mass()` etc.
- UpstreamDrift seams: `optimization/casadi_backend.py` (`_LINK_MASS = 1.0`, `_LINK_INERTIA = 1e-2`, FD velocity consistency), `optimization/_swing_kinematics.JOINTS` (7 DOF), `optimization/model_provider.build_swing_rig/build_pinocchio_model`, `estimation/map_estimator.py` (scipy `least_squares`, `NON_FINITE_RESIDUAL_SENTINEL`), `estimation/moving_horizon.py`, `estimation/identifiability.py`, `motion_pipeline/contracts.py` (`KeypointSequence`, `MarkerTrajectory`), pytest markers `requires_casadi`/`requires_pinocchio`/`requires_crocoddyl`, `ci-optional-stack.yml` crocoddyl probe pattern.
