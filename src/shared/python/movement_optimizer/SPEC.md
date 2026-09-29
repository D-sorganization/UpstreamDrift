# SPEC.md -- Movement-Optimizer Repository Specification

## 1. Identity

| Field            | Value                                                   |
| ---------------- | ------------------------------------------------------- |
| Repository Name  | `Movement-Optimizer`                                    |
| GitHub URL       | `https://github.com/D-sorganization/Movement-Optimizer` |
| Owner            | D-sorganization                                         |
| Primary Language | Python 3.10+                                            |
| License          | MIT                                                     |
| Package Name     | `movement-optimizer`                                    |
| Current Version  | `1.0.0`                                                 |
| Spec Version     | `1.0.15`                                                |
| Last Spec Update | 2026-06-17                                              |

## 2. Purpose

Movement-Optimizer is a biomechanics trajectory optimizer for barbell exercises. It models the body as a sagittal-plane planar chain, computes trajectories with Lagrangian inverse dynamics, and exposes both a GUI workflow and a headless CLI for batch optimisation.

## 3. Scope

### 2026-06-17 Update

- Motion-tab slider/text controls and scroll-panel construction now live in
  `movement_optimizer.gui.motion_controls`, keeping the tab modules under the
  enforced source-size budget without changing the Swingset or Chain Dynamics
  interaction contract.

### 2026-06-16 Update

- Swingset policy search now precomputes cyclic control matrices with vectorized
  NumPy trigonometry before the necessarily sequential state rollout, reducing
  optimizer callback overhead without changing policy values.
- Chain Dynamics now uses torque-based bend stiffness/damping over rod-link
  inertia, validates single-link gravity against a slender pendulum, and
  initializes kick velocity as a tip-weighted profile instead of a mid-chain
  sine wave.
- Swingset and Chain Dynamics analysis tabs include default-on autoplay
  checkboxes so optimized or simulated motion starts playing as soon as results
  are ready.
- Analysis-tab playback now has explicit responsiveness contracts: switching
  from a playing barbell exercise into Swingset Model or Chain Dynamics stops
  the barbell animation timer before the shared playback controls retarget.
- Swingset and chain overlays cache rollout-wide force fields, avoiding
  repeated finite-difference and torque recomputation during every animation
  frame.
- Numeric sliders no longer emit continuous drag-time refreshes, and the
  Swingset optimizer action is styled as the primary command.
- The Swingset and Chain Dynamics tabs now expose per-element animation
  visibility: a "Show in animation" checklist toggles each MotionCanvas
  layer (grid/chain/rider/markers/forces) independently, on top of the
  existing force-vector filters.
- Each tab splits into Animation and Plots sub-tabs so the analysis plots
  get a roomy dedicated area; a "Show plot legends" control and a
  toggleable, top-strip-reserving policy-trace legend keep legends from
  obscuring the plotted curves.

### In Scope

- Sagittal-plane movement optimisation for barbell exercises
- Body and dynamics modelling
- Trajectory optimisation and result persistence
- GUI visualization and comparison tooling
- Export helpers for plots and animation artifacts
- Optional Rust acceleration in `rust_core/`

### Out Of Scope

- Full 3D biomechanics simulation
- Networked services or remote orchestration
- Non-barbell exercise domains unless they fit the current factory model

## 4. Architecture

### Package Layout

```text
src/movement_optimizer/
├── __main__.py          # GUI entrypoint for `python -m movement_optimizer`
├── cli.py               # Headless batch CLI
├── backend.py           # Physics backend interface
├── config.py            # Runtime configuration and state paths
├── constants.py         # Physical constants and tuning values
├── comparison.py        # Trial comparison helpers
├── export.py            # CSV/PNG/PDF/GIF export helpers
├── persistence.py       # JSON save/load for sessions and results
├── rendering.py         # Matplotlib rendering helpers
├── spine_loads.py       # Spine load analysis
├── strength.py          # Torque and load-capacity helpers
├── models/              # Body model and Lagrangian dynamics
├── exercises/           # Exercise configuration factories
├── trajectory/          # Optimizer, cache, result, tuning types
└── gui/                 # PyQt6 windows, tabs, widgets, and dialogs
rust_core/                # Optional PyO3/maturin hot path accelerator
tests/                   # Pytest suite
```

### Key Boundaries

- `models/` owns body geometry, exercise configs, and analytical dynamics.
- `trajectory/` owns optimisation orchestration, cache handling, and result types.
- `exercises/` owns exercise-specific configuration factories.
- `gui/` owns all PyQt6 presentation and interaction code.
- `cli.py` owns the headless batch interface and JSON output shaping.
- `__main__.py` owns the GUI startup path.
- Sidebar and playback GUI widgets expose facade methods for state changes,
  signal binding, and summary values so main-window mixins do not traverse into
  child widget internals.

## 5. Entry Points

- `movement-optimizer` console script maps to `movement_optimizer.__main__:main`.
- `python -m movement_optimizer` launches the GUI.
- `python -m movement_optimizer.cli` runs headless optimisation.
- `run.py` and the platform launch scripts are convenience wrappers around the package entrypoints.

## 6. Runtime Contract

- The default body model is a 3-link planar sagittal chain.
- `models/` provides the primary squat, full squat, deadlift, and bench press configuration factories alongside the body and dynamics types.
- `exercises/` provides supplemental factories for clean, jerk, snatch, gait, and sit-to-stand flows.
- Optimisation uses multi-start search and SciPy-based solvers.
- GUI state is stored locally and does not require external services.
- Optional Rust acceleration is an implementation detail, not a hard dependency.
  When the compiled `rust_core` extension is absent the dynamics fall back to an
  equivalent NumPy path, so results are identical and only performance differs.

## 7. Data And Configuration

### Inputs

- Body parameters such as mass, height, and segment multipliers
- Barbell mass and exercise-specific configuration values
- Optional runtime state directory via `MOVEMENT_OPTIMIZER_STATE_DIR`

### Outputs

- Optimisation summaries and detailed JSON results
- Matplotlib figures and exported plots
- GIF/PNG/PDF artifacts
- Persisted session state

## 8. Testing And CI

### Test Strategy

- `pytest` is the canonical test framework.
- Tests live in `tests/` and use shared fixtures from `tests/conftest.py`.
- Unit tests should cover model, trajectory, GUI helper, and export behavior.
- Property-based tests use Hypothesis where parameter-space coverage matters.

### Canonical Commands

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=movement_optimizer --cov-report=term-missing
ruff check src/ tests/
ruff format src/ tests/
mypy --ignore-missing-imports src/movement_optimizer/
```

### Quality Expectations

- Public APIs must be type-hinted.
- Preconditions should be checked early with `ValueError` or `TypeError`.
- `src/` code should use logging rather than `print`.
- Tests should remain deterministic and avoid network access.

## 9. Change Log

| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-16 | 1.0.14  | Added per-element animation layer toggles (grid/chain/rider/markers/forces) to the Swingset and Chain Dynamics tabs via a shared `_MotionViewMixin`, split each tab into Animation/Plots sub-tabs, and made plot/policy-trace legends toggleable so they no longer obscure the plotted data. `MotionAnalysisPanel.set_legends_visible`/`has_legends` encapsulate legend control (LoD).                        |
| 2026-06-15 | 1.0.12  | Lifted the legacy `scipy<1.16` ceiling after verifying current SciPy imports `CubicSpline` cleanly, and added a dependency-contract regression so the stale cap cannot return silently.                                                                                                                                                                                                                       |
| 2026-05-16 | 1.0.11  | Isolated nightly workflow installs into a dedicated .nightly-venv virtual environment with PIP_NO_CACHE_DIR=1 to avoid shared runner cache corruption that caused ImportError: cannot import name '\_spropack' from scipy.sparse.linalg.\_propack (#462).                                                                                                                                                     |
| 2026-04-22 | 1.0.10  | Added GUI sidebar/playback facade methods and routed main-window mixins through them to reduce deep object traversal in animation, comparison, cancellation, and signal binding code (#272).                                                                                                                                                                                                                  |
| 2026-04-16 | 1.0.9   | Extracted spline-building responsibility from `TrajectoryOptimizer` into `optimizer_spline.py` (`build_splines`, `eval_trajectory`); extracted `_compute_bench_bar_cost` private helper from `_compute_cost`; exported new functions from `trajectory/__init__.py`; added 14 characterization/unit tests in `test_issue_247_split_optimizer.py` (#247).                                                       |
| 2026-04-14 | 1.0.7   | Split `gui/widgets.py` (489 LOC) into three focused modules (`labelled_slider.py`, `parameter_sidebar.py`, `playback_controls.py`) and decomposed `models/lagrangian_dynamics.py` (463 LOC) by extracting `LagrangianKinematicsMixin` into `lagrangian_kinematics.py` and balance helpers into `lagrangian_balance.py`. Each resulting module is ≤300 LOC; `widgets.py` becomes a thin re-export shim (#218). |
| 2026-04-14 | 1.0.6   | Added NaN/infinite input validation to `HillTorqueModel` constructor and key methods (`torque_angle_factor`, `torque_velocity_factor`, `available_torque`). All seven numeric constructor parameters are now checked with `math.isfinite`; NaN or infinite values raise `ValueError` immediately rather than propagating silently (#236).                                                                     |
| 2026-04-11 | 1.0.5   | Split `tests/test_trajectory.py` (678 LOC) into three focused modules — `test_trajectory_generation.py`, `test_trajectory_optimization.py`, and `test_trajectory_validation.py` — and promoted the `squat_optimizer` / `full_squat_optimizer` fixtures to `conftest.py` for shared reuse (#211).                                                                                                              |
| 2026-04-11 | 1.0.4   | Decomposed `TrajectoryOptimizer.optimize()` and `_package_results()` into thin orchestrators backed by focused helpers (`_optimize_single_start`, `_optimize_parallel_starts`, `_collect_future_results`, `_finalize_parallel_results`, `_evaluate_solution`, `_validate_solution`, `_build_result_object`) to satisfy the Function Size target (#214).                                                       |
| 2026-04-11 | 1.0.3   | Added a stable public API to `ProgressTracker` (`cost_history`, `iteration_count`, `elapsed()`, `lock()`) and refactored `TrajectoryOptimizer` to stop reaching into its private attributes, eliminating a cluster of Law-of-Demeter violations in the optimiser engine.                                                                                                                                      |
| 2026-04-10 | 1.0.2   | Replaced the last `print()` call in `src/` with direct stdout JSON emission in the CLI summary path and updated the CLI regression test to preserve the headless output contract without violating the no-print rule.                                                                                                                                                                                         |
| 2026-04-09 | 1.0.1   | Added a shared provider-pack manifest, validator, regression tests, and launcher icon asset so Movement-Optimizer can publish a launcher-compatible utility pack without embedding UpstreamDrift-specific path logic.                                                                                                                                                                                         |
| 2026-04-06 | 1.0.0   | Initial repository specification aligned to the current package layout, entrypoints, and test contract.                                                                                                                                                                                                                                                                                                       |
