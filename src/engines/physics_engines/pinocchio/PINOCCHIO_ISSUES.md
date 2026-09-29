# Pinocchio Parity — Issue Drafts

Six implementation issues to take Pinocchio from "IK-only motion training
package" to "full peer of the Simscape motion-matching pipeline". Body
text below is meant to be copy-pasted directly into GitHub issue bodies.

Companion to [PINOCCHIO_PARITY_SPEC.md](./PINOCCHIO_PARITY_SPEC.md). Read
that first for context.

Convention: issue bodies are second-person to the agent that will
implement them. Each issue declares its own acceptance tests and is small
enough to land in one PR (~< 800 LOC of diff including tests).

Labels used throughout:
`engine:pinocchio`, `parity`, `spec:cross-engine`, `area:simulation`,
`area:optimization`, `area:loaders`, `area:visualization`, `area:tests`,
`size:S`, `size:M`, `size:L`, `priority:P0/P1/P2`.

---

## Issue 1 — PIN-MODEL-GRIP-FRAME

**Title:** `feat(pinocchio): add mid-hands frame + floating base to golfer.urdf`

**Labels:** `engine:pinocchio`, `parity`, `area:model`, `size:S`, `priority:P0`

**Depends on:** none

**Body:**

The current `models/generated/golfer.urdf` welds the club to `hand_left`
via a fixed joint. Cross-engine spec §2.6 mandates the club is welded to
the **mid-hands** frame (geometric mean of left and right hand tip
frames in the address pose), and the pelvis is a floating base (6 DOF).

### Tasks

1. Add a `mid_hands` virtual link as a child of `thorax3` (or a more
   appropriate ancestor — see audit note). Origin = midpoint of
   `hand_left_tip` and `hand_right_tip` frames in the address pose.
2. Replace `hand_left_to_club_shaft` with `mid_hands_to_club_shaft`
   (`type="fixed"`).
3. Insert a `floating_base` joint between the URDF root and `pelvis`:
   `<joint type="floating">`. Pinocchio loads this as 7 generalised
   coordinates (qx, qy, qz, qw, x, y, z) and 6 velocity DOFs.
4. Update `golfer_ik.urdf` to also have the floating base — its IK
   role still needs a free-floating root.
5. Add a header comment block to **both** URDFs declaring scope:
   - `golfer.urdf`: forward-simulation with club welded.
   - `golfer_ik.urdf`: body-only, club tracked externally.
6. Spot-check by loading both URDFs into Pinocchio and asserting:
   - `model.nq == 30` (29 + quaternion w) for `golfer.urdf` with club.
   - `pin.getFrameId('mid_hands')` returns a valid frame.

### Acceptance criteria

- [ ] Both URDFs load without error via `pin.buildModelFromUrdf`.
- [ ] `mid_hands` frame exists and is positioned ≤ 1 cm from the
      geometric mean of the two hand-tip frames in the address pose
      (`q = q0`).
- [ ] Floating base present; verified with
      `model.joints[1].shortname() == 'JointModelFreeFlyer'`.
- [ ] Header comment block in both URDFs explains scope.
- [ ] Test `tests/unit/test_urdf_invariants.py` passes
      (TDD-first: written before the URDF edits).

### Size

S — ~150 LOC test + 30 LOC URDF diff + 10 LOC docs.

---

## Issue 2 — PIN-SIMULATE

**Title:** `feat(pinocchio): simulate_with_coefficients (RK4 + ABA forward simulator)`

**Labels:** `engine:pinocchio`, `parity`, `area:simulation`, `size:M`, `priority:P0`

**Depends on:** PIN-MODEL-GRIP-FRAME (#1)

**Body:**

Implement the canonical engine-agnostic forward-sim wrapper per
cross-engine spec §2.2. New file:

`src/engines/physics_engines/pinocchio/python/pinocchio_golf/simulate_with_coefficients.py`

### Implementation outline

```python
def simulate_with_coefficients(
    theta: np.ndarray,                    # (n_joints * 7,)
    options: SimOptions = ...,
    initial_pose: dict | None = None,
) -> SimOut:
    ...
```

1. Lazily build and cache `pin.Model` from `golfer.urdf` at module level.
2. Reshape `theta` to `(n_joints, 7)`.
3. Set `q0`, `qd0` from `initial_pose` or zeros.
4. RK4 integrate from 0 to `options.t_final` at `options.dt`:
   - At each stage, compute per-joint torque
     `tau_j(t) = sum_k a_jk * t^k` for k=0..6.
   - Compute `qdd = pin.aba(model, data, q, qd, tau)`.
5. Re-walk the trajectory once with `pin.computeForwardKinematics` to
   extract `grip` (mid-hands frame) and `clubhead` (`club_head` link)
   poses.
6. Pack into the canonical `SimOut`.

### Pinocchio gotchas to honour

- **Do NOT call `pin.computeTotalEnergy`.** Use
  `computeKineticEnergy` + `computePotentialEnergy`.
- `pin.Data` is **not thread-safe**. Cache per-thread.
- For RK4 stages, hold `data` immutable per stage; copy `q, qd` between
  stages.

### Tests (TDD-first)

`tests/unit/test_simulate_with_coefficients.py`:

- [ ] Zero-torque + zero-velocity → trajectory drifts only via gravity.
- [ ] Energy conservation in zero-gravity, zero-damping mode: drift
      ≤ 1% over 1.0 s.
- [ ] Returned `SimOut` has correct shapes and finite values.
- [ ] Determinism: same theta twice → identical trajectories.

### Acceptance criteria

- [ ] All four tests above pass.
- [ ] Wall-clock per call ≤ 100 ms for 1.0 s @ 1 kHz sim on a single
      modern CPU core (post-warmup, theta = zeros).
- [ ] Function signature matches cross-engine §2.2 exactly.

### Size

M — ~600 LOC implementation + 400 LOC tests.

---

## Issue 3 — PIN-TDD-ORACLE

**Title:** `test(pinocchio): synthesize_target_from_coefficients + recovery harness`

**Labels:** `engine:pinocchio`, `parity`, `area:tests`, `size:S`, `priority:P0`

**Depends on:** PIN-SIMULATE (#2)

**Body:**

Lay the TDD oracle so the optimiser issue (PIN-FIT-DRIVER #4) has its
acceptance test ready before its implementation lands.

### Tasks

1. New file `pinocchio_golf/synthesize_target_from_coefficients.py`:
   ```python
   def synthesize_target_from_coefficients(
       theta: np.ndarray, options: SimOptions = ...
   ) -> ClubTarget:
       sim_out = simulate_with_coefficients(theta, options)
       return ClubTarget(
           time=sim_out.time, grip=sim_out.grip, grip_quat=sim_out.grip_quat,
           clubhead=sim_out.clubhead, club_quat=sim_out.club_quat,
           impact_idx=int(np.argmax(np.linalg.norm(np.diff(sim_out.clubhead, axis=0), axis=1))),
           events=None, source=SourceProvenance(loader='synthetic'),
       )
   ```
2. New file `tests/unit/test_synthesize_target.py`:
   - Round-trip determinism.
   - Shape contracts for ClubTarget.
3. New file `tests/unit/test_fit_swing_pinocchio_recovery.py`:
   - Skip-marked initially (`pytest.skip("optimiser not implemented")`).
   - Once #4 lands, the skip is removed.
   - Asserts `‖θ_recovered − θ_truth‖∞ < 1e-3` on noise-free synth.

### Acceptance criteria

- [ ] Synthesize round-trip test passes.
- [ ] Recovery test exists, marked `@pytest.mark.skip` with link to #4.

### Size

S — ~150 LOC code + 200 LOC tests.

---

## Issue 4 — PIN-FIT-DRIVER

**Title:** `feat(pinocchio): fit_swing_pinocchio with analytical Jacobians + LM optimizer`

**Labels:** `engine:pinocchio`, `parity`, `area:optimization`, `size:L`, `priority:P0`, `killer-feature`

**Depends on:** PIN-SIMULATE (#2), PIN-TDD-ORACLE (#3), PARITY-LOADERS (cross-engine)

**Body:**

**This is the issue that lands the Pinocchio killer feature.**

Pinocchio is one of two engines in the fleet (the other is Drake) that
ships analytical first- and second-order derivatives of articulated-body
dynamics. MuJoCo and Simscape rely on finite differences inside their
optimisers, costing one extra forward sim per parameter per iteration.
With 7 polynomial coefficients × 23 joints = 161 parameters, that's 161
sims per gradient step in Simscape — versus 1 forward sim + 1 derivative
pass (≈ 3× one sim) in Pinocchio.

The 5-second wall-clock fit target is the quantitative win.

### Implementation outline

New file `pinocchio_golf/fit_swing_pinocchio.py`:

```python
def fit_swing_pinocchio(
    target: ClubTarget,
    options: FitOptions = default_fit_options(),
) -> FitResult:
    ...
```

1. Build initial guess `theta0`:
   - If `options.theta0` provided, use it.
   - Else: run `dual_hand_ik_solver` (reusing motion_training package)
     on `target.grip` + waypoint approximation of `target.clubhead`. Get
     `q(t)`, finite-diff for `qd, qdd`, compute
     `tau(t) = pin.rnea(q, qd, qdd)`, polyfit per joint.
   - Fallback path that does NOT depend on Pink (issue PIN-IK-FALLBACK).
2. Build LM optimiser via
   `scipy.optimize.least_squares(method='lm', x0=theta0, jac=analytical_jac)`.
3. `analytical_jac(theta)` chains:
   - `pin.computeABADerivatives` (`∂qdd/∂{q, qd, tau}`).
   - RK4 stage chain rule for `∂{q, qd}_n+1 / ∂theta`.
   - `pin.computeJointJacobians` to map `∂q → ∂grip`.
4. Cost is the shared `compute_cost` function from
   `shared/python/motion_matching/cost.py` — **do not write a Pinocchio
   cost function**.
5. Pack into the canonical `FitResult`.

### Tests

- [ ] Recovery test (`test_fit_swing_pinocchio_recovery.py` — un-skip).
- [ ] Smoke test on a real Rob Neal trial:
      `final_rmse_m < 0.05` and wall-clock `< 5 s`.
- [ ] Determinism on a fixed seed.
- [ ] LM convergence: outer iterations < 50 on the recovery problem.

### Acceptance criteria

- [ ] All tests pass.
- [ ] **5-second wall-clock fit** demonstrated in the smoke test on
      CI's standard runner (single core).
- [ ] PR description includes a benchmark table:
      Simscape fmincon vs Pinocchio LM on the same trial.

### Size

L — ~1000 LOC code + 600 LOC tests. Largest issue in this set.

---

## Issue 5 — PIN-LOADER-ADAPTER

**Title:** `feat(pinocchio): ClubTarget adapter on club_trajectory_parser + Rob Neal *.mat reader`

**Labels:** `engine:pinocchio`, `parity`, `area:loaders`, `size:S`, `priority:P1`

**Depends on:** none directly; consumes types from PARITY-LOADERS once landed

**Body:**

The `motion_training/club_trajectory_parser.py` already parses Excel
trajectories into a Pinocchio-internal `ClubTrajectory` dataclass. The
canonical cross-engine schema is `ClubTarget` (see
[CLUB_IK_SPEC.md](../../Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CLUB_IK_SPEC.md)).
We need an adapter and a Rob Neal `*.mat` reader.

### Tasks

1. Add `ClubTrajectory.to_club_target() -> ClubTarget`. Quaternion
   conversion via `scipy.spatial.transform.Rotation.from_matrix(...).as_quat(scalar_first=True)`.
2. Add a Rob Neal loader. Two-pass implementation:
   - Pass A: `scripts/inspect_swing_dataset_mat.py` (one-shot tool) to dump
     the field schema of `*.mat` and `*_targetKinematics.mat` and
     commit the output as `tests/fixtures/swing_dataset_schema.json`.
   - Pass B: `motion_training/swing_dataset_loader.py` reading
     `_targetKinematics.mat` directly into `ClubTarget`.
3. Add tests `tests/unit/test_load_club_target.py` covering both Excel
   and Rob Neal paths.

### Cross-cutting note

Once `shared/python/motion_matching/loaders/` exists (issue
PARITY-LOADERS-ROBNEAL), **promote** the Rob Neal loader from this
location to the shared package and replace this engine's import with
`from shared.python.motion_matching.loaders.club_swing_dataset import load`.

### Acceptance criteria

- [ ] `to_club_target()` round-trips through a sample Rob Neal trial
      with byte-identical (within float tolerance) output to a hand-checked
      fixture.
- [ ] Rob Neal loader handles all 8 trial files in
      `data/club_swing_dataset/{TW,GW}_{ProV1,wiffle}*.mat`.
- [ ] Tests pass.

### Size

S — ~250 LOC code + 250 LOC tests + 50 LOC inspection script.

---

## Issue 6 — PIN-VIZ-AND-LEADERBOARD

**Title:** `feat(pinocchio): visualize_fit + leaderboard JSON writer`

**Labels:** `engine:pinocchio`, `parity`, `area:visualization`, `size:S`, `priority:P1`

**Depends on:** PIN-FIT-DRIVER (#4)

**Body:**

Wire the per-engine visualisation entry point and emit the
JSON file the cross-engine leaderboard consumes.

### Tasks

1. New file `pinocchio_golf/visualize_fit.py`:
   ```python
   def visualize_fit(target: ClubTarget, result: FitResult, *,
                    out_dir: Path | None = None,
                    interactive: bool = False) -> dict[str, Path]:
       """Emit the three canonical figures + Meshcat overlay."""
   ```
2. Reuse the shared plotters once they exist:
   - `shared/python/motion_matching/plot_error_timecourse.py`
   - `shared/python/motion_matching/plot_fit_quality_card.py`
3. Engine-bespoke 3D viewer: extend
   `motion_training/motion_visualizer.py` with a `MotionVisualizer.overlay(target, result)`
   method that renders both measured and simulated club skeletons in
   different colours.
4. After `fit_swing_pinocchio` completes, write
   `results/<trial>/pinocchio.json` with the canonical `FitResult`
   serialisation. Schema from CODING_STANDARDS.md.

### Tests

- [ ] `tests/unit/test_visualize_fit_smoke.py` — runs end-to-end on
      a fixture and asserts files exist.
- [ ] `tests/unit/test_leaderboard_writer.py` — round-trip JSON.

### Acceptance criteria

- [ ] `visualize_fit` produces three figures + a Meshcat URL string.
- [ ] `pinocchio.json` validates against the canonical FitResult schema.
- [ ] PARITY-LEADERBOARD picks up the Pinocchio entry without code
      changes on the leaderboard side.

### Size

S — ~300 LOC code + 200 LOC tests.

---

## Housekeeping issues (filed but not in the critical six)

These should be opened alongside the six above so they live in the
backlog, but are not gating cross-engine parity.

### PIN-RENAME-TORQUE-UTIL

`refactor(pinocchio): rename torque_fitting.py to poly_torque_util.py`

`pinocchio_golf/torque_fitting.py` is a 1D polyfit utility, not the
swing-fit driver. Rename to `poly_torque_util.py`. Add deprecation shim.
Add unit test. Labels: `engine:pinocchio`, `area:cleanup`, `size:S`.

### PIN-DOF-AUDIT

`test(pinocchio): kinematic equivalence audit vs Simscape at 3 poses`

Verify the URDF joint chain produces the same end-effector frames as
Simscape at address, top-of-backswing, and impact poses. Drives any
necessary corrections to forearm DOF split. Labels: `engine:pinocchio`,
`area:model`, `area:tests`, `size:M`.

### PIN-DOC-URDF-SCOPE

`docs(pinocchio): URDF header comments declaring scope`

Add explanatory comment blocks to `golfer.urdf` (forward-sim with welded
club) and `golfer_ik.urdf` (body-only, external club tracking). Labels:
`engine:pinocchio`, `area:docs`, `size:S`.

### PIN-IK-FALLBACK

`feat(pinocchio): differential IK fallback that does not depend on Pink`

Implement a pure-Pinocchio damped-pseudo-inverse IK using
`pin.computeJointJacobians` for the seed-trajectory step in
`fit_swing_pinocchio`. Removes Pink as a single-point-of-failure.
Labels: `engine:pinocchio`, `area:optimization`, `size:M`.

### PIN-DEPS-AUDIT

`chore(pinocchio): pin Pinocchio version supporting computeABADerivatives`

Confirm `pyproject.toml` requires Pinocchio ≥ 2.6 so analytical
ABA derivatives are available. Required for PIN-FIT-DRIVER. Labels:
`engine:pinocchio`, `area:deps`, `size:S`.

### PIN-LOD-CLEANUP

`refactor(pinocchio): break method chains > 2 deep across the package`

Audit and fix LOD violations like
`pipeline.ik_solver.model.nq` (training_pipeline.py:138) by adding
delegating methods. Labels: `engine:pinocchio`, `area:cleanup`, `size:M`.

---

## Summary table

| #   | ID                      | Size | Priority | Depends                |
| --- | ----------------------- | ---- | -------- | ---------------------- |
| 1   | PIN-MODEL-GRIP-FRAME    | S    | P0       | —                      |
| 2   | PIN-SIMULATE            | M    | P0       | #1                     |
| 3   | PIN-TDD-ORACLE          | S    | P0       | #2                     |
| 4   | PIN-FIT-DRIVER          | L    | P0       | #2, #3, PARITY-LOADERS |
| 5   | PIN-LOADER-ADAPTER      | S    | P1       | —                      |
| 6   | PIN-VIZ-AND-LEADERBOARD | S    | P1       | #4                     |
| h   | PIN-RENAME-TORQUE-UTIL  | S    | P2       | —                      |
| h   | PIN-DOF-AUDIT           | M    | P1       | #1                     |
| h   | PIN-DOC-URDF-SCOPE      | S    | P2       | —                      |
| h   | PIN-IK-FALLBACK         | M    | P2       | #2                     |
| h   | PIN-DEPS-AUDIT          | S    | P0       | — (do first!)          |
| h   | PIN-LOD-CLEANUP         | M    | P2       | —                      |

`h` = housekeeping (not in the critical six but worth filing).

The natural delivery order is:
**PIN-DEPS-AUDIT → #1 → #2 → #3 → #5 (in parallel) → #4 → #6.**

_Last updated 2026-05-06; tracks PR `feat/pinocchio-parity-spec`._
