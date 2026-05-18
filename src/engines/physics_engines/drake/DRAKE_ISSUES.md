# Drake Parity — Implementation Issue Drafts

Issues to file once `DRAKE_PARITY_SPEC.md` lands. Each entry is ready to
drop into `gh issue create` body verbatim. Labels follow the
established repo convention (`engine:drake`, `motion-matching`,
`parity`, `tdd`).

All issues target branch `main` (per GAAI fleet policy as of
2026-05-01) and follow CLAUDE.md guard-rails:
explicit `from pydrake.X import Y`, `body.body_frame()` not
`FixedOffsetFrame`, and `patch.dict("sys.modules", ...)` for any
pydrake-mocking unit tests.

---

## DRAKE-1: Generate the canonical humanoid URDF from shared YAML

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:M`, `area:body-model`

**Depends on:** PARITY-DIMENSIONS (cross-engine: shared
`golf_humanoid_dimensions.yaml`).

### Body

The Drake parity work starts with the URDF — it is the single biggest
gap in the engine and a prerequisite for every downstream issue.
Today's `drake_golf_model.GolfURDFGenerator` builds an in-memory
`xml.etree` tree from a hard-coded `GolfModelParams` dataclass and is
never serialised to disk. We need a regenerable on-disk URDF driven
by the shared anthropometric YAML.

**Deliverables:**

- New module `python/motion_matching/humanoid_urdf.py` exposing
  `build_humanoid_urdf(yaml_path, out_path) → Path` and
  `load_humanoid_into_plant(plant, urdf_path) → ModelInstanceIndex`.
- Refactor `GolfURDFGenerator` to read every dimension from the YAML,
  delete the hard-coded `GolfModelParams` defaults.
- Generated URDF lives at `models/generated/golfer.urdf`. Hand-edits
  forbidden (CI gate is DRAKE-7).
- `scripts/build_humanoid_models.py --engine drake` calls into this
  module (PARITY-MODEL-BUILD wires the orchestrator).

**Acceptance:**

- `pytest tests/motion_matching/test_humanoid_urdf.py` green.
- The URDF parses cleanly via `pydrake.multibody.parsing.Parser` and
  the resulting plant reports exactly **23 generalized velocities**.
- Pelvis-to-shoulder Euclidean distance in the parsed plant matches
  the YAML to 1 mm.
- `GolfModelParams` no longer carries hard-coded defaults; YAML is the
  sole source of truth.

---

## DRAKE-2: `simulate_with_coefficients` (float pathway)

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:L`, `area:simulator`

**Depends on:** DRAKE-1, PARITY-LOADERS (canonical `SimOut`
dataclass).

### Body

Implement the canonical forward-sim wrapper required by
cross-engine §2.2. Drake takes a coefficient vector of length
`n_joints * 7`, assembles a Stateflow-equivalent torque polynomial
`tau_j(t) = A + Bt + Ct² + Dt³ + Et⁴ + Ft⁵ + Gt⁶`, and runs a fixed-
duration simulation. This issue lands the **float-only** version; the
templated `AutoDiffXd` version is DRAKE-4.

**Deliverables:**

- `python/motion_matching/simulate_with_coefficients.py` with the
  canonical signature
  `simulate_with_coefficients(theta, options, initial_pose) → SimOut`.
- A `LeafSystem` actuator that evaluates the per-joint polynomial.
  Subclass `pydrake.systems.framework.LeafSystem` (NOT the templated
  `LeafSystem_[T]` yet — that's DRAKE-4).
- Forward kinematics post-step to extract `grip` / `grip_quat` /
  `clubhead` / `club_quat` using `body.body_frame()` per CLAUDE.md.
- Sample to the canonical 1 kHz grid via `SimOptions.sample_rate_hz`.

**Acceptance:**

- Returns a fully-populated `SimOut` (no NaNs, finite
  `q`/`qd`/`tau`/`grip`).
- With `theta = 0`, the grip falls in the −z world direction
  (gravity sanity check).
- Output schema matches the cross-engine canonical `SimOut`
  byte-for-byte (use the dataclass from
  `shared/python/motion_matching/...`).

---

## DRAKE-3: `fit_swing_drake` scipy driver + `compute_cost_drake` adapter

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:M`, `area:fit-driver`

**Depends on:** DRAKE-2.

### Body

The default fit driver: `scipy.optimize.minimize(method="L-BFGS-B")`
with finite-difference Jacobians, calling
`shared/python/motion_matching/cost.py` directly via a thin Drake
adapter. Mirrors `fit_swing_fmincon` (Simscape) and `fit_swing_mujoco`
(MuJoCo).

**Deliverables:**

- `python/motion_matching/fit_swing_drake.py` exposing
  `fit_swing_drake(target, options) → FitResult`.
- `python/motion_matching/compute_cost_drake.py` (≤ 100 LOC) that
  imports the canonical cost from
  `src.shared.python.motion_matching.cost` and adds Drake-specific
  constraint-residual penalties only. Slim by design — anything else
  belongs in the shared module.
- Slim `motion_optimization.py` per spec §1.3: keep
  `OptimizationObjective` / `OptimizationConstraint` dataclasses,
  delete the placeholder cost bodies, delete
  `optimize_for_distance` / `optimize_for_accuracy`.
- Tests: synthetic-recovery (`final_rmse_m < 5 mm`) and
  cost-contract (Drake adapter agrees with shared cost numerically).

**Acceptance:**

- `synthesize_target_from_coefficients(theta_truth) →
fit_swing_drake → final_rmse_m < 5 mm`.
- `FitResult` schema matches cross-engine canonical (`theta_optimal`,
  `final_rmse_m`, `solver_status`, `iterations`, `wall_clock_s`).
- `compute_cost_drake.py` ≤ 100 LOC.

---

## DRAKE-4: `fit_swing_drake_autodiff` (the killer feature)

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:XL`, `area:fit-driver`, `feature:autodiff`

**Depends on:** DRAKE-3.

### Body

The reason Drake is in the parity matrix at all: gradient-based
fitting via `MathematicalProgram` + `IpoptSolver`, with analytic
gradients flowing through the dynamics via `AutoDiffXd`. This issue
lands the milestone that justifies the engine's place alongside
MuJoCo and Pinocchio.

**Deliverables:**

- Templated `LeafSystem_[T]` polynomial-torque source (so the
  autodiff plant connects identically to the float plant).
- `python/motion_matching/fit_swing_drake_autodiff.py` exposing
  `fit_swing_drake_autodiff(target, options) → FitResult`.
- `plant.ToAutoDiffXd()` plumbing + an autodiff-friendly cost
  (manual replication of the float cost using `pydrake.math` /
  `AutoDiffXd` arithmetic). Numerical-equivalence test guards
  against silent gradient breakage.
- Integrator selection: benchmark `RungeKutta3` vs
  `ImplicitEulerIntegrator` for autodiff stability under stiff
  polynomial torques.

**Acceptance:**

- Same synthetic target as DRAKE-3 → `final_rmse_m < 1 mm` (tighter
  because gradients).
- **Sim-call budget:** ≤ 50 forward simulations to converge vs
  ≥ 100 for DRAKE-3's scipy driver on the same trial.
- Numerical equivalence: float-cost and autodiff-cost agree to
  1e-9 relative on a fixed `(theta, target)` pair.
- Wall-clock target ≤ 30 s per fit (spec §6).

---

## DRAKE-5: Equivalence test against Simscape ground truth

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:M`, `area:tests`, `tier:scientific`

**Depends on:** DRAKE-2 (and a Simscape-side ground-truth fixture
issue PARITY-FIXTURE if one isn't already merged).

### Body

The cross-engine §2.2 equivalence requirement: a fixed `theta` driven
through Drake's `simulate_with_coefficients` must produce a grip
trajectory within 5 mm RMSE of the Simscape reference at three
canonical poses (impact, top-of-backswing, address).

**Deliverables:**

- `tests/motion_matching/test_equivalence_simscape.py` marked
  `@pytest.mark.scientific`.
- `tests/motion_matching/fixtures/simscape_ground_truth.json` (the
  canonical Simscape reference — produced once by the Simscape
  pipeline and checked into the repo as a versioned fixture).
- A pytest helper that parses the fixture, runs the Drake forward
  sim with the same `theta`, and computes per-pose grip RMSE.

**Acceptance:**

- Test passes on CI's `nightly` lane.
- Cross-engine parity-matrix row for Drake turns 🟢 in
  `CROSS_ENGINE_PARITY_SPEC.md` § 3 (the spec edit is part of this PR).
- Failure mode is informative: report per-pose RMSE in the assert
  message so future regressions are easy to triage.

---

## DRAKE-6: Visualisation parity (Meshcat overlay + shared 2D plotters)

**Labels:** `engine:drake`, `motion-matching`, `parity`, `tdd`,
`size:S`, `area:visualisation`

**Depends on:** DRAKE-3.

### Body

Cross-engine §2.5 mandates three views per fit: trajectory overlay,
error timecourse, fit-quality card. The 2D plots come from
`shared/python/motion_matching/plot_*.py` (engine-agnostic); only the
3D overlay is Drake-specific (Meshcat).

**Deliverables:**

- `python/motion_matching/visualize_fit.py` consuming a
  `FitResult` + `ClubTarget` and rendering in Meshcat using the
  existing `drake_visualizer.DrakeVisualizer` helper.
- Headless smoke test marked `@pytest.mark.requires_gl` that
  asserts the Meshcat URL is reachable (or, if Meshcat fails in CI,
  that a fallback static PNG was produced).
- CLI entry point `python -m
src.engines.physics_engines.drake.python.motion_matching.visualize_fit
results/<trial>/drake.json`.

**Acceptance:**

- Meshcat overlay shows the humanoid skeleton, measured-grip path,
  and simulated-grip path on the same scene.
- Error timecourse + fit-quality card use the shared plotters
  (zero duplicated Matplotlib code).
- Smoke test green in headless CI.

---

## DRAKE-7: URDF regeneration CI gate

**Labels:** `engine:drake`, `parity`, `ci`, `size:XS`,
`area:body-model`

**Depends on:** DRAKE-1.

### Body

A `pre-merge` CI step that re-runs
`scripts/build_humanoid_models.py --engine drake` on every PR and
asserts the on-disk URDF is byte-identical. Mirrors the URDF-gate
already in place for Pinocchio. Closes the loop on "hand-edited engine
files are forbidden" (cross-engine §6).

**Deliverables:**

- New job in `.github/workflows/ci-standard.yml` (or a dedicated
  `ci-engine-models.yml`) that runs the build script with `--check`.
- Build-script support for `--check`: regenerate to a tmpfile, diff
  against the on-disk URDF, exit non-zero on mismatch.

**Acceptance:**

- A deliberately edited URDF causes the gate to fail with a clear
  error message pointing the contributor to the YAML.
- A clean checkout passes the gate in ≤ 30 s.
- Local-run instructions live in `models/generated/README.md`.

---

_Last updated 2026-05-06. Tracks the same parity wave as the
sister engines' issue lists. PRs reference this file in their body and
target branch `main`._
