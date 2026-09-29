# Pinocchio Parity Specification

> **Audience:** anyone implementing or modifying the Pinocchio engine wrapper
> under `src/engines/physics_engines/pinocchio/`.
>
> **Parent doc:** [CROSS_ENGINE_PARITY_SPEC.md](../../CROSS_ENGINE_PARITY_SPEC.md).
> Read that first. This file is the Pinocchio-specific addendum.
>
> **Status:** spec; implementation tracked in
> [PINOCCHIO_ISSUES.md](./PINOCCHIO_ISSUES.md). Existing surface area is
> substantial but **parallel to** rather than _aligned with_ the Simscape
> motion-matching pipeline. The plan is **decouple-and-rewire**, not
> ground-up rewrite.
>
> **Pinocchio-specific gotcha:** `pinocchio.computeTotalEnergy` does NOT
> exist in the Python bindings we depend on. Use
> `computeKineticEnergy` + `computePotentialEnergy` separately and sum.
> This rule lives in the repo `CLAUDE.md` and applies everywhere here.

---

## 1. Current state

### 1.1 Inventory of relevant files

```
src/engines/physics_engines/pinocchio/
├── models/generated/
│   ├── golfer.urdf            (1038 lines, body + club, fixed joint at hand_left)
│   └── golfer_ik.urdf         (1031 lines, body only — no club)
├── data/club_swing_dataset/
│   ├── GW_ProV1.mat / GW_ProV1_targetKinematics.mat
│   ├── GW_wiffle.mat / GW_wiffle_targetKinematics.mat
│   ├── TW_ProV1.mat / TW_ProV1_targetKinematics.mat
│   ├── TW_wiffle.mat / TW_wiffle_targetKinematics.mat
│   ├── ClubDataGUI_v2.m       (MATLAB import GUI; legacy)
│   └── Old Revs/              (deprecated; do not consume)
├── python/
│   ├── motion_training/       (THE motion-matching surface today)
│   │   ├── club_trajectory_parser.py     (468 lines, Excel/Mat parser)
│   │   ├── dual_hand_ik_solver.py        (560 lines, Pink-based dual-hand IK)
│   │   ├── motion_visualizer.py          (719 lines, Meshcat + matplotlib)
│   │   ├── trajectory_exporter.py        (429 lines, MuJoCo/Drake/STO export)
│   │   ├── training_pipeline.py          (380 lines, top-level orchestration)
│   │   └── tests/__init__.py             (no actual tests yet)
│   ├── pinocchio_golf/        (the legacy GUI + analysis surface)
│   │   ├── gui.py             (PySide GUI for interactive playback)
│   │   ├── torque_fitting.py  (83 lines — see §1.4 below)
│   │   ├── induced_acceleration.py
│   │   ├── manipulability.py
│   │   ├── pinocchio_recorder.py
│   │   └── ... (mixins, ui/)
│   ├── dtack/                 (back-end abstraction over pinocchio/mujoco/pink)
│   │   ├── backends/{pinocchio,mujoco,pink}_backend.py
│   │   ├── ik/pink_solver.py
│   │   ├── sim/dynamics.py
│   │   ├── viz/{meshcat,geppetto,swing_dataset}_viewer.py
│   │   └── utils/{matlab_importer,urdf_exporter,mjcf_exporter}.py
│   ├── pinocchio_physics_engine.py
│   ├── pinocchio_screw_kinematics.py
│   └── swing_plane_integration.py
└── tests/{integration,validation}/__init__.py    (placeholders only)
```

91 Python files in total; the _motion-matching-relevant_ set is the seven
above, plus `dtack/sim/dynamics.py` and `pinocchio_physics_engine.py` which
host the lower-level forward-dynamics primitives we'll wrap.

### 1.2 `golfer.urdf` vs `golfer_ik.urdf` — the difference

Both files have **identical 23-DOF internal body chains** rooted at `pelvis`
(Pinocchio-URDF count; this is +4 over the canonical 19-actuated Simscape
chain because the existing URDF carries decorative `hand→fingers` joints
and a 2-DOF elbow/forearm split — see §3.1 and §3.2 below, and the
canonical totals in `shared/models/golf_humanoid_topology.yaml`).
The diff is exclusively in the right-hand subtree at line ~585:

| URDF             | Right after `hand_left` link                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `golfer.urdf`    | `hand_left_to_club_shaft` (fixed) → `club_shaft` link → `club_shaft_to_club_head` (fixed) → `club_head` link → only then `hand_left_to_fingers_left` |
| `golfer_ik.urdf` | `hand_left_to_fingers_left` directly — no club at all                                                                                                |

The two are otherwise byte-for-byte identical. Implication:

- `golfer.urdf` is the **forward-simulation model** (club rigidly welded to
  the lead hand; matches the Simscape "club locked to mid-hands frame"
  assumption).
- `golfer_ik.urdf` is the **IK-target model** for `dual_hand_ik_solver.py`,
  which solves _both hands as end-effectors_ tracking external grip frames
  on a free-floating club. The club is intentionally absent so its position
  does not constrain IK redundantly.

**Important divergence from the parity contract:** the club is welded to
`hand_left` only, not to a `mid_hands` virtual frame between the two hands.
That mismatches §2.6 of the cross-engine spec, which requires the club's
6-DOF lock to be at the mid-hands grip frame. Issue **PIN-MODEL-GRIP-FRAME**
addresses this.

### 1.3 The `motion_training/` package — what it actually does

`MotionTrainingPipeline.run()` (training_pipeline.py:110-174) executes
**inverse-kinematics motion reconstruction**, NOT torque optimization:

1. Parse Excel/Mat club trajectory (`club_trajectory_parser.py`) →
   `ClubTrajectory` dataclass with mid-hands position+rotation per frame.
2. Initialise `DualHandIKSolver` (Pink + Pinocchio) on `golfer_ik.urdf`.
3. Per-frame: solve joint angles `q(t)` such that left and right hand
   end-effectors coincide with offset positions on the grip.
4. Save `q_trajectory`, plot, optionally Meshcat playback, optionally
   export joint trajectories to MuJoCo/Drake/STO formats.

**Verdict — is this surface-equivalent to `fit_swing_*`?**

**No.** `motion_training` is the Pinocchio counterpart to
_`option3_inverse_nn` data preparation_ — it produces ground-truth body
poses from measured club motion. It does **not** fit polynomial torque
coefficients. There is no cost function, no optimiser, no
`SimOut/FitResult` round-trip. The cross-engine `fit_swing_pinocchio`
function does not yet exist.

That said, `motion_training` is **highly leverageable**:

- The IK solver gives us a high-quality `theta0` starting joint trajectory.
- The club-trajectory parser already speaks the same kinematic language
  (mid-hands position + rotation) as the canonical `ClubTarget`.
- The Meshcat viewer + matplotlib plotting fulfil VISUALIZATION_SPEC.md §1
  (trajectory overlay) and §2 (error timecourse) with minor adapter work.

So the work is **adapt**, not **rewrite**. Issues **PIN-FIT-DRIVER** and
**PIN-LOADER-ADAPTER** capture this.

### 1.4 `torque_fitting.py` — what it actually is

83 lines in `pinocchio_golf/torque_fitting.py`. It is **neither the cost
function nor the swing optimizer**. It is a **1D least-squares polynomial
regressor** with this signature:

```python
def fit_torque_poly(t: ArrayLike, tau: ArrayLike, degree: int = 6) -> NDArray:
    return np.polyfit(t, tau, degree)
```

It takes a _known_ per-joint torque time series and fits an N-th degree
polynomial to it (numpy `polyfit`). Useful as a **post-hoc smoother** on a
measured/computed torque signal, but _unrelated_ to motion-matching.

**Plan:** keep the file (it has utility), but **rename and document its
scope** so no one mistakes it for the fitter. The new
`fit_swing_pinocchio.py` (issue PIN-FIT-DRIVER) is the actual swing-fit
optimiser. Renaming tracked under PIN-RENAME-TORQUE-UTIL.

### 1.5 Rob Neal `*.mat` files — relation to `ClubTarget`

The `data/club_swing_dataset/` directory has **paired files**:

| Pair                          | Contents (inferred from filename + ClubDataGUI_v2.m)              |
| ----------------------------- | ----------------------------------------------------------------- |
| `<name>.mat`                  | Raw club-marker positions/rotations at native MoCap rate          |
| `<name>_targetKinematics.mat` | Resampled, smoothed target kinematics for downstream optimisation |

Naming convention: `{TW,GW}_{ProV1,wiffle}` = subjects Tiger Woods (TW) /
Generic Walker (GW), balls ProV1 / wiffle. Four trials per category.

**Mapping to canonical `ClubTarget`:**

| ClubTarget field | Rob Neal source                                                                   |
| ---------------- | --------------------------------------------------------------------------------- |
| `time`           | derived from MoCap sample rate (240 Hz default) + `_targetKinematics` time vector |
| `grip`           | grip-marker XYZ from `_targetKinematics.mat` (already in metres after parser)     |
| `grip_quat`      | grip-frame rotation matrix → quaternion (parser handles 3×3 → quat)               |
| `clubhead`       | club-face-marker XYZ from `_targetKinematics.mat`                                 |
| `club_quat`      | club-face rotation → quaternion                                                   |
| `impact_idx`     | `events.impact` from `SwingEventMarkers` (parser extracts from sheet metadata)    |
| `events`         | `{address, top, impact, finish, club_head_speed}` from event markers              |
| `source`         | new: `SourceProvenance(loader='club_swing', file=<name>.mat, sheet=<name>)`       |

**Coverage gap:** the canonical Python loader at
`shared/python/motion_matching/load_club_target.py` (issue PARITY-LOADERS)
**does not yet** speak the `_targetKinematics.mat` format. Two paths:

1. **Short term:** keep `club_trajectory_parser.py` as the Pinocchio
   adapter; add a `to_club_target()` method that emits the canonical
   schema. (Issue PIN-LOADER-ADAPTER.)
2. **Promotion:** lift the parser into
   `shared/python/motion_matching/loaders/club_swing_dataset.py` so MuJoCo, Drake,
   and OpenSim get the loader for free. (Issue PARITY-LOADERS-ROBNEAL,
   filed against the cross-engine tracker.)

We do (1) first; (2) is a second-pass DRY action once shared/ exists.

---

## 2. Target architecture

The end state is one cross-engine-shaped surface. New file layout:

```
src/engines/physics_engines/pinocchio/python/
├── pinocchio_golf/
│   ├── simulate_with_coefficients.py     [NEW — PIN-SIMULATE]
│   ├── fit_swing_pinocchio.py            [NEW — PIN-FIT-DRIVER]
│   ├── synthesize_target_from_coefficients.py [NEW — PIN-TDD-ORACLE]
│   ├── visualize_fit.py                  [NEW — PIN-VIZ thin wrapper]
│   ├── poly_torque_util.py               [RENAMED from torque_fitting.py]
│   └── (existing GUI stuff — untouched)
├── motion_training/
│   ├── club_trajectory_parser.py         [REFACTOR + add to_club_target]
│   └── (existing IK solver — kept; provides q0 seed)
└── tests/
    ├── unit/
    │   ├── test_simulate_with_coefficients.py     [TDD-FIRST]
    │   ├── test_fit_swing_pinocchio_recovery.py   [TDD-FIRST oracle test]
    │   ├── test_synthesize_target.py              [TDD-FIRST]
    │   └── test_load_club_target.py
    └── integration/
        └── test_equivalence_vs_simscape.py        [PARITY-EQUIVALENCE gate]
```

### 2.1 URDF verification & finalisation (issue **PIN-MODEL-GRIP-FRAME**)

Required changes to `golfer.urdf`:

1. Insert a new virtual link `mid_hands` between `thorax3` and the club,
   positioned at the geometric mean of `hand_left` and `hand_right` tip
   frames in the address pose.
2. Replace `hand_left_to_club_shaft` (fixed, anchored on `hand_left`) with
   `mid_hands_to_club_shaft` (fixed, anchored on the new virtual link).
3. The 6-DOF lock is realised as URDF `<joint type="fixed">`; Pinocchio
   loads this as a frame, not a joint, which is the correct semantic.
4. Add a `floating_base` (`<joint type="floating">`) at the pelvis so the
   model has 6 base v-DOFs + 23 internal = **29 generalised velocities
   (`nv`)** and **30 configuration coordinates (`nq`)** — Pinocchio's
   floating base contributes 7 q (3 position + 4 quaternion) and 6 v
   (3 linear + 3 angular). The canonical Simscape chain per
   `shared/models/golf_humanoid_topology.yaml` is 6 + 19 = 25 v; the +4
   extra here are decorative finger joints and the elbow/forearm split
   noted in §3.1.

Acceptance: `pinocchio.buildModelFromUrdf` returns `model.nq == 30`
(equivalently `model.nv == 29`) and `pin.getFrameId('mid_hands')` is
valid.

**Decision on `golfer_ik.urdf`:** retain unchanged for now. It serves the
post-MVP body-marker IK pipeline (§3.4 below).

### 2.2 `simulate_with_coefficients.py` (issue **PIN-SIMULATE**)

Wraps Pinocchio's RNEA + a custom integrator that applies the polynomial
torque driver. Signature is fixed by §2.2 of the cross-engine spec.

```python
def simulate_with_coefficients(
    theta: np.ndarray,           # (n_joints * 7,)
    options: SimOptions = ...,
    initial_pose: dict | None = None,
) -> SimOut:
    """
    Forward-simulate the golfer + club system using polynomial torques.

    Internals:
      1. Build pinocchio.Model from golfer.urdf (cached at module level).
      2. Reshape theta -> (n_joints, 7)  [polynomial degree 6 by spec].
      3. Set initial state q0, qd0 from `initial_pose` or default.
      4. RK4 integrate from t=0 to t=options.t_final at options.dt:
         - At each step, evaluate tau_j(t) = sum_k a_jk * t^k
         - Compute qdd via pin.aba(model, data, q, qd, tau)
         - q, qd <- RK4 step
      5. After integration, re-walk the trajectory once with
         pin.computeForwardKinematics to extract grip + clubhead frames.
      6. Pack into SimOut.
    """
```

**Why analytical Jacobians matter here, but not yet:** for _forward_
simulation we don't need them. They become the killer feature in §2.3.

**Pinocchio-specific implementation notes:**

- Use `pin.aba` (Articulated Body Algorithm), O(n), not `pin.crba` + solve.
- Cache `pin.Data` per worker thread (not safe across threads).
- For energy diagnostics in `SimOut`, sum
  `pin.computeKineticEnergy` + `pin.computePotentialEnergy`. **Do not call
  `pin.computeTotalEnergy`** — see CLAUDE.md.

**Performance target:** < 100 ms per forward sim of a 1.0 s swing at 1 kHz
sample rate, single-threaded. Pinocchio's C++ ABA is fast enough that the
Python call overhead dominates; consider batching `aba` calls if profiling
shows a bottleneck.

### 2.3 `fit_swing_pinocchio.py` (issue **PIN-FIT-DRIVER**)

The fit driver. Signature fixed by §2.4 of the cross-engine spec.

```python
def fit_swing_pinocchio(
    target: ClubTarget,
    options: FitOptions = ...,
) -> FitResult:
    ...
```

**The Pinocchio killer feature: second-order optimisation.**

Where MuJoCo and Simscape rely on gradient-free or finite-difference
gradients into `fmincon`, Pinocchio gives us **analytical first- and
second-order derivatives** of the dynamics:

- `pin.computeJointJacobians` — `∂grip/∂q` analytically.
- `pin.computeRNEADerivatives` — `∂tau/∂q`, `∂tau/∂qd`, `∂tau/∂qdd`.
- `pin.computeABADerivatives` — `∂qdd/∂q`, `∂qdd/∂qd`, `∂qdd/∂tau`.
- Chain rule through the integrator (RK4 stages each get their own
  derivative pass) yields `∂SimOut/∂theta` exactly, no finite differences.

This unlocks:

1. **Levenberg-Marquardt** as the default optimiser (`scipy.optimize.least_squares`
   with `method="lm"` and a user-supplied Jacobian). LM is the right tool
   when the cost is a sum of squared residuals (which it is — see
   COST_FUNCTION_SPEC.md).
2. **Trust-region Newton** as the precision finisher (`method="trf"` with
   Hessian via Gauss-Newton approximation, or `scipy.optimize.minimize`
   with `method="trust-ncg"` and an analytical Hessian).

Initial guess strategy:

1. If `options.theta0` provided, use it.
2. Else: run `motion_training/dual_hand_ik_solver.py` on the target's
   grip+clubhead waypoints to get `q(t)`. Differentiate twice for `qd`,
   `qdd`. Compute `tau(t) = pin.rnea(q, qd, qdd)`. Polyfit
   `tau_j(t)` per joint to get `theta0`. **This is the role
   `motion_training/` plays in the new pipeline** — IK seed, not
   primary fitter.

Acceptance:

- Recovery test (synthesize → fit → recover) on a held-out theta with
  noise σ=0 must hit `‖θ_recovered − θ_truth‖∞ < 1e-3`.
- Wall-clock target: **< 5 s per swing fit** end-to-end on a single
  modern CPU core.

### 2.4 `synthesize_target_from_coefficients.py` (issue **PIN-TDD-ORACLE**)

```python
def synthesize_target_from_coefficients(
    theta: np.ndarray,
    options: SimOptions = ...,
) -> ClubTarget:
    """
    Forward-sim with `simulate_with_coefficients`, repackage SimOut into a
    ClubTarget. Used by tests to construct ground-truth (theta, target)
    pairs for the recovery test.
    """
```

Trivial wrapper around `simulate_with_coefficients` + repackaging. Lives
here so the recovery test imports it directly without circular deps.

### 2.5 Refactor `torque_fitting.py` (issue **PIN-RENAME-TORQUE-UTIL**)

1. Move `pinocchio_golf/torque_fitting.py` →
   `pinocchio_golf/poly_torque_util.py`.
2. Rename module-level `fit_torque_poly` → `fit_polynomial_to_signal` to
   make scope explicit.
3. Add a deprecation shim at the old path that warns + re-exports.
4. Add tests at `tests/unit/test_poly_torque_util.py` (synthetic
   sin(t) → polyfit → eval roundtrip).

This is **not** "consume `ClubTarget` and emit `FitResult`" — that role
belongs to `fit_swing_pinocchio.py`. The brief asked us to verify, and the
finding is that `torque_fitting.py` is mis-scoped to be that thing.

### 2.6 Visualisation (issue **PIN-VIZ**)

Existing `motion_training/motion_visualizer.py` already does Meshcat 3D
playback + matplotlib joint/error plots. The new `visualize_fit.py` is a
**thin adapter** that:

1. Accepts `(target: ClubTarget, result: FitResult) -> None`.
2. Calls into `MotionVisualizer` for the Meshcat overlay (red = measured,
   blue = simulated).
3. Calls into `shared/python/motion_matching/plot_error_timecourse.py`
   (issue PARITY-LOADERS) for the 2D error/torque/CHS plot.
4. Calls into `plot_fit_quality_card.py` for the single-figure summary.

No new visualisation code unless the shared plotter is missing a hook.

### 2.7 Tests (TDD-first; issue **PIN-TESTS**)

Six tests, written **before** the implementation in the same PR:

1. `test_synthesize_target.py` — round-trip
   `theta → SimOut → ClubTarget → SimOut'` is identity within ε.
2. `test_simulate_with_coefficients_basics.py` — zero-torque trajectory
   stays at q0; gravity-only trajectory pendulum-falls correctly.
3. `test_simulate_with_coefficients_energy.py` — kinetic+potential energy
   conservation at zero damping ≤ 1% drift over 1 s. **Note: must call
   `computeKineticEnergy + computePotentialEnergy` separately.**
4. `test_fit_swing_pinocchio_recovery.py` — synthesize at known
   `theta_truth` → fit → assert recovered ≈ truth.
5. `test_fit_swing_pinocchio_smoke.py` — fit a real Rob Neal trial,
   assert `final_rmse_m < 0.05` (5 cm) and wall-clock `< 5 s`.
6. `test_equivalence_vs_simscape.py` — fixed `theta` → run Pinocchio sim
   → grip RMSE vs cached Simscape reference output ≤ 5 mm at three poses.
   This test is the **PARITY-EQUIVALENCE** gate.

---

## 3. Body model

### 3.1 Recommendation: **retain the existing URDFs**, regenerate only when shared YAML lands

The 23-DOF internal chain in `golfer.urdf` is a **superset** of the
canonical 19-actuated Simscape kinematic breakdown: Hip(6) + Spine via
lumbar1/2/3 (each 2 DOF intermediate + revolute = 6) + Torso/thorax1-3 (3)

- Scapula (2 each) + Shoulder-gimbal (3 each) + Elbow (1 each) + Wrist
  intermediate+revolute (2 each) + fingers (1 each, decorative). Total 23
  internal revolute DOFs + 6 base = **29 v-velocities** (and 30 q-positions
  because the floating base contributes 7 q vs 6 v). The +4 over the
  canonical 25 v-velocity Simscape count come from the two decorative
  finger joints and the elbow/forearm 2-DOF split (vs Simscape's lumped
  1 DOF). The canonical count lives in
  `shared/models/golf_humanoid_topology.yaml` (PR #4150).

When `shared/models/golf_humanoid_dimensions.yaml` lands (issue
PARITY-DIMENSIONS), wire `scripts/build_humanoid_models.py` to regenerate
both URDFs from that YAML. Until then, the existing files are
authoritative for Pinocchio.

### 3.2 Joint mapping to the Simscape 25-DOF (6 floating + 19 actuated) chain

| Simscape joint name                                | Pinocchio joint(s)                                                                                  | Notes                                                                                                         |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Pelvis_RX/RY/RZ/TX/TY/TZ`                         | floating-base joint                                                                                 | Add per §2.1                                                                                                  |
| `Hip_FlexExt`, `Hip_AbAd`, `Hip_IntExt` (per side) | `pelvis_to_<side>_thigh_gimbal_z` + `..._gimbal_y` + `..._to_<side>_thigh`                          | 3-DOF gimbal split as expected                                                                                |
| `Knee_FlexExt` (per side)                          | `<side>_thigh_to_<side>_shank`                                                                      | 1 DOF                                                                                                         |
| `Ankle_PlantarDorsiflex`, `Ankle_InvEv` (per side) | `<side>_shank_to_<side>_foot_intermediate` + `<side>_foot_intermediate_to_<side>_foot`              | 2 DOF                                                                                                         |
| `Lumbar1/2/3_FlexExt`, `Lumbar*_Lateral`           | `pelvis_to_lumbar1_intermediate` + `lumbar1_intermediate_to_lumbar1`, repeat ×3                     | 2 DOF per lumbar segment                                                                                      |
| `Thorax1/2/3_<axis>`                               | `lumbar3_to_thorax1`, `thorax1_to_thorax2`, `thorax2_to_thorax3`                                    | 1 DOF each (matches Simscape)                                                                                 |
| `Scapula_<side>_X/Y`                               | `thorax3_to_scapula_<side>_intermediate` + `scapula_<side>_intermediate_to_scapula_<side>`          | 2 DOF per side                                                                                                |
| `Shoulder_<side>_X/Y/Z`                            | `scapula_<side>_to_upper_arm_<side>_gimbal_z` + `..._gimbal_y` + `..._to_upper_arm_<side>`          | 3 DOF gimbal                                                                                                  |
| `Elbow_<side>_FlexExt`                             | `upper_arm_<side>_to_forearm_<side>_intermediate` + `forearm_<side>_intermediate_to_forearm_<side>` | NOTE Pinocchio uses 2 DOFs here for prono-supination + elbow flexion; Simscape lumps as 1 — see PIN-DOF-AUDIT |
| `Wrist_<side>_FlexExt`, `Wrist_<side>_RadUlnar`    | `forearm_<side>_to_hand_<side>_intermediate` + `hand_<side>_intermediate_to_hand_<side>`            | 2 DOF                                                                                                         |
| (decorative, no Simscape twin)                     | `hand_<side>_to_fingers_<side>`                                                                     | locked or weakly constrained at q=0                                                                           |

The forearm-elbow split is the only known mismatch. Issue
**PIN-DOF-AUDIT** validates the kinematic equivalence by FK at three test
poses (address, top, impact).

### 3.3 Club attachment

Per §2.1: weld via fixed joint `mid_hands_to_club_shaft` rooted at the new
`mid_hands` virtual link. Club itself is two rigid bodies (`club_shaft`,
`club_head`) connected by a fixed joint, masses + inertias from the
existing URDF.

For force/torque accounting at the grip, expose
`pin.getFrameJacobian(model, data, frame_mid_hands, pin.ReferenceFrame.LOCAL)`
which gives the 6×n grip Jacobian — the workhorse of the LM optimiser.

### 3.4 `golfer_ik.urdf` — current vs post-MVP

`golfer_ik.urdf` is **currently in active use** by
`motion_training/dual_hand_ik_solver.py`. It is also the right model for
the post-MVP body-marker IK pipeline (when we have body MoCap data, not
just club traces). Keep it as a sibling artifact; document its scope in
the file header. Issue **PIN-DOC-URDF-SCOPE** lands the comment block.

---

## 4. Implementation plan (six atomic issues)

Detailed bodies in [PINOCCHIO_ISSUES.md](./PINOCCHIO_ISSUES.md).

| #   | ID                      | Title                                                                                            | Size | Depends                    |
| --- | ----------------------- | ------------------------------------------------------------------------------------------------ | ---- | -------------------------- |
| 1   | PIN-MODEL-GRIP-FRAME    | Add mid-hands frame + floating base to `golfer.urdf`; document scope of both URDFs               | S    | —                          |
| 2   | PIN-SIMULATE            | `simulate_with_coefficients.py` with RK4 + ABA forward simulator                                 | M    | #1                         |
| 3   | PIN-TDD-ORACLE          | `synthesize_target_from_coefficients.py` + recovery test infra                                   | S    | #2                         |
| 4   | PIN-FIT-DRIVER          | `fit_swing_pinocchio.py` — LM + analytical Jacobians; **the killer-feature issue**               | L    | #2, #3, PARITY-LOADERS     |
| 5   | PIN-LOADER-ADAPTER      | `club_trajectory_parser.to_club_target()` + Rob Neal `_targetKinematics.mat` reader              | S    | PARITY-LOADERS (read-only) |
| 6   | PIN-VIZ-AND-LEADERBOARD | Wire `visualize_fit.py` + emit `results/<trial>/pinocchio.json` for the cross-engine leaderboard | S    | #4                         |

Additional housekeeping issues (not counted toward the six):

- **PIN-RENAME-TORQUE-UTIL** — rename `torque_fitting.py` → `poly_torque_util.py`.
- **PIN-DOF-AUDIT** — validate joint-mapping equivalence vs Simscape at 3 poses.
- **PIN-DOC-URDF-SCOPE** — add header comments to both URDFs.

### 4.1 Killer-feature framing for issue PIN-FIT-DRIVER

The body of issue #4 makes this explicit:

> Pinocchio is one of two engines in the fleet that ships analytical
> derivatives of articulated-body dynamics (the other is Drake). MuJoCo and
> Simscape rely on finite differences inside `fmincon`, costing one extra
> forward simulation per parameter per iteration. With ~7 coefficients ×
> ~19 actuated joints (canonical) ≈ ~140 parameters, that's >100 sims per gradient evaluation in
> Simscape — vs 1 sim + 1 derivative pass (≈ 3× one sim) in Pinocchio.
>
> The acceptance criterion that exercises this is the **5-second wall-clock
> fit**. Hit it, and we have a clear quantitative win to advertise.

---

## 5. TDD / DbC / DRY / LOD compliance

### 5.1 TDD conversion plan

Current state: zero substantive tests in `motion_training/tests/` or
`tests/{integration,validation}/`.

Plan, per cross-engine §2.7:

1. For every new file in §2 (PIN-SIMULATE, PIN-TDD-ORACLE, PIN-FIT-DRIVER,
   PIN-LOADER-ADAPTER, PIN-VIZ), the test is in the **same PR**, written
   first. PRs without tests are blocked at review.
2. Existing `dual_hand_ik_solver.py` and `club_trajectory_parser.py` get
   regression tests in the PR that adds `to_club_target()` (PIN-LOADER-ADAPTER).
   Lift the simplest IK convergence assertion + Excel-parse golden fixture
   into `tests/unit/`.
3. The recovery test (`test_fit_swing_pinocchio_recovery.py`) is the
   **first test for the optimiser** and exists before the optimiser does —
   it serves as the executable definition of "the fitter works".

### 5.2 DbC

Every public function in §2 declares preconditions in dataclass /
pydantic validators (Python equivalent of MATLAB `arguments` blocks):

```python
@dataclass(frozen=True)
class SimOptions:
    t_final: float = 1.0
    dt: float = 1e-3
    integrator: Literal["rk4", "semi_implicit"] = "rk4"

    def __post_init__(self) -> None:
        if not (self.t_final > 0):
            raise ValueError("t_final must be positive")
        if not (0 < self.dt <= self.t_final):
            raise ValueError("dt must be in (0, t_final]")
```

Postconditions as `assert` blocks at function exit; on by default in CI,
disabled with `python -O` for production deployments where appropriate.

### 5.3 DRY plan — what gets pushed to `shared/python/motion_matching/`

| Lift target                                     | Currently lives at                               | Promote to                                                                                   |
| ----------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `ClubTarget`, `SimOut`, `FitResult` dataclasses | nowhere — to be authored                         | `shared/python/motion_matching/types.py` (issue PARITY-LOADERS)                              |
| `compute_cost(sim_out, target, opts)`           | `compute_cost.m` MATLAB only                     | `shared/python/motion_matching/cost.py` (issue PARITY-LOADERS)                               |
| Polynomial torque evaluation `tau(t; theta)`    | inline in §2.2 plan                              | `shared/python/motion_matching/poly_torque.py`                                               |
| Rob Neal `*.mat` reader                         | `motion_training/club_trajectory_parser.py`      | `shared/python/motion_matching/loaders/club_swing_dataset.py` (issue PARITY-LOADERS-ROBNEAL) |
| Error timecourse / fit quality plots            | `motion_training/motion_visualizer.py` (partial) | `shared/python/motion_matching/plot_*.py`                                                    |

The Pinocchio engine then **imports** all of the above; engine-bespoke
code is exclusively the forward-sim wrapper, the LM driver, and the
Meshcat 3D viewer.

### 5.4 LOD

Forbid method chains > 2 deep. Concretely:

- `pin.computeForwardKinematics(model, data, q); data.oMi[frame_id].translation` — **OK** (1 chain).
- `pipeline.ik_solver.model.nq` (training_pipeline.py:138) — **VIOLATION** (3 chains).
  Fix: add `MotionTrainingPipeline.dof()` delegating method.
- Audit all of the existing 91 files for similar violations as part of
  PIN-LOD-CLEANUP (housekeeping issue, not blocking).

---

## 6. Performance baseline + targets

### 6.1 Baseline (estimated, no current measurements)

Pinocchio benchmarks on similar 30-DOF humanoids report:

- ABA forward dynamics: ~5 µs per call (C++).
- Forward kinematics + Jacobians: ~15 µs per call.
- Python binding overhead: ~50–100 µs per call.

For a 1.0 s swing at 1 kHz with RK4 (4 stages × 1000 steps = 4000 ABA
calls): **0.2–0.5 s** wall-clock per forward simulation. Headroom over
the 100 ms target is concerning — we will likely need to:

1. Drop sample rate to 500 Hz (still 10× the swing's dominant
   frequency content). 4× speedup.
2. Use semi-implicit Euler instead of RK4 where stability allows.
3. Batch multiple ABA calls into a single Pinocchio data context.

If these don't get us to 100 ms, escalate to a Cython/pybind shim around
the integrator inner loop.

### 6.2 Targets (acceptance criteria)

| Metric                                      | Target                      | Issue                           |
| ------------------------------------------- | --------------------------- | ------------------------------- |
| Single forward sim wall-clock               | < 100 ms                    | PIN-SIMULATE                    |
| Single swing fit (LM, analytical Jacobians) | **< 5 s**                   | PIN-FIT-DRIVER                  |
| Recovery error on noise-free synthetic      | `‖θ_rec − θ_truth‖∞ < 1e-3` | PIN-TDD-ORACLE / PIN-FIT-DRIVER |
| Real-trial RMSE (Rob Neal)                  | grip < 5 cm                 | PIN-FIT-DRIVER                  |
| Equivalence vs Simscape on fixed θ          | grip RMSE ≤ 5 mm at 3 poses | PARITY-EQUIVALENCE              |

### 6.3 Why these are realistic

- 5 s for a full fit is conservative given LM typically converges in
  10–30 outer iterations × 3× one-sim derivative cost ≈ 30–90 sim-
  equivalents. At 100 ms/sim that's 3–9 s; the target is the optimistic
  end of that band.
- 5 mm equivalence is achievable: both engines integrate the same ODE
  with consistent inertias; the only error sources are integrator
  truncation and `pin.aba` vs Simscape's solver tolerance.

---

## 7. Risks / open questions

### 7.1 `motion_training/` — refactor or rewrite?

**Verdict: refactor.** The IK solver and club-trajectory parser are
solid. The orchestration (training*pipeline.py) is misnamed but the
pipeline graph is correct for the \_IK-seed* role it will play in the new
architecture. Specifically:

- `MotionTrainingPipeline.run()` becomes the implementation of the
  "compute theta0 from measured trajectory" step inside
  `fit_swing_pinocchio` (§2.3 above).
- The `save_trajectory` / `save_plots` / `_visualize` branches stay
  optional; under `fit_swing_pinocchio` they're disabled.
- Renaming the package `motion_training` → `motion_seeding` (or similar)
  is a nice-to-have, **not in the critical path**. Defer to a follow-up
  housekeeping issue.

The single risky bit is `dual_hand_ik_solver.py`'s use of Pink. Pink is
a separate dependency; if it breaks, we have a single point of failure
for the seed. Mitigation: implement a fallback "differential IK from
target grip frame" path inside `fit_swing_pinocchio` that does NOT need
Pink — uses just `pin.computeJointJacobians` + damped pseudo-inverse.
Issue PIN-IK-FALLBACK (housekeeping).

### 7.2 Rob Neal `*.mat` — shared loader scope

Open question: what's the right shared loader API? Options:

1. **One loader per format**, dispatched by file extension. Cleanest
   contract; one new loader per format added.
2. **Polymorphic `load_club_target(path) -> ClubTarget`** that sniffs the
   format. Tidiest caller; harder to evolve.

Recommendation: (1), starting with `loaders/club_swing_dataset.py`,
`loaders/excel.py`, `loaders/c3d.py`, with a top-level `load_club_target`
that delegates by extension. Track under PARITY-LOADERS-ROBNEAL.

The risk is that the `_targetKinematics.mat` schema is undocumented
outside `ClubDataGUI_v2.m`. Mitigation: write a one-shot
`scripts/inspect_swing_dataset_mat.py` that loads each `*.mat` and prints
field names + shapes; commit the output as a fixture.

### 7.3 Forearm DOF mismatch (PIN-DOF-AUDIT)

`golfer.urdf` splits forearm motion into two revolutes
(`forearm_<side>_intermediate_to_forearm_<side>` plus the
`upper_arm_*_to_forearm_*_intermediate` predecessor). Simscape combines
elbow flex/ext and forearm pronosupination differently. Until the
PIN-DOF-AUDIT issue verifies kinematic equivalence at three test poses,
the 5 mm equivalence-test target carries hidden risk. If audit fails, we
either:

(a) regenerate `golfer.urdf` from the shared YAML once it lands (issue
PARITY-MODEL-BUILD), or
(b) lock the redundant DOF at q=0 in the Pinocchio model.

### 7.4 `computeTotalEnergy` trap

Restating the project rule: **NEVER call `pin.computeTotalEnergy`.** The
Python binding does not expose it on the Pinocchio version pinned in
`pyproject.toml`; calls fail at runtime, not import time, which is the
worst possible failure mode. Energy diagnostics use:

```python
KE = pin.computeKineticEnergy(model, data, q, qd)
PE = pin.computePotentialEnergy(model, data, q)
total = KE + PE
```

The energy-conservation test (PIN-TESTS #3) is the canary — if it
disappears or gets disabled, this rule is at risk of regression.

### 7.5 Pinocchio Python version pin

The `pyproject.toml` may pin a Pinocchio version older than
`pin.computeABADerivatives`, which lands in 2.6+. Confirm before sizing
PIN-FIT-DRIVER. Issue PIN-DEPS-AUDIT (housekeeping) does the version
sweep.

---

## 8. Out of scope (for this spec; tracked elsewhere)

- Body-marker IK pipeline (uses `golfer_ik.urdf`; landing post-MVP).
- Coppelia bridge — `pinocchio_golf/coppelia_bridge.py` is a separate
  concern.
- The legacy GUI (`pinocchio_golf/gui.py`) is preserved as-is; making it
  consume the new `FitResult` is a follow-up.
- Real-time inference / embedded deployment (the eventual reason
  Pinocchio is in the fleet, but a phase-3 deliverable).
- Muscle-driven dynamics (OpenSim's territory).

---

_Last updated 2026-05-06; tracks PR `feat/pinocchio-parity-spec`._
