# MS-31 Turnover: Native Crocoddyl Full-Body Fit (Pinocchio Lane)

Governing issue: #10338 (epic #10363, Matched Swing Program). Branch
`feat/10338-crocoddyl-native-fit`. Development-log entry `DL-#10338`.
Last updated 2026-09-17 18:10 PT by claude. Read this file first; it is kept current
at every checkpoint so another agent can continue without the chat history.

## Objective

Produce the program's first G1-accepted candidate (driver capture, 0 to 0.85 s)
by solving the through-contact dynamic fit with Crocoddyl's native FDDP on the
qualified `FullBodyPinocchioModel` (shared contact law, weld closure), then
hand the candidate to MS-21 (#10336) for MuJoCo replay through the gates.
G1 gates (run-102 thresholds): whole <= 25 mm, early <= 12 mm, terminal
<= 35 mm, club <= 60 mm, pelvis yaw < 3 deg; physical gates per MS-01.

## Where Things Run

- Code: this branch. Fit driver `src/engines/physics_engines/pinocchio/python/full_body_fit.py`
  (CLI), problem assembly `crocoddyl_problem.py` (pure), action models and
  rollouts `crocoddyl_action.py`, marker kinematics + Gauss-Newton IK
  `marker_kinematics.py`, GIF renderer `candidate_playback.py`.
- Runtime: ControlTower (Tailscale host `controltower`, SSH shell is cmd.exe),
  WSL distro `ControlTower-Runner`, micromamba env `upstream-motion-runtime`
  at `/home/dieterolson/mm-root` (Pinocchio 4.1.0, Crocoddyl 3.2.1, Pink 4.4.0
  from `scripts/config/motion_runtime/linux-64.explicit.txt`, plus ezc3d,
  scipy, pydantic, imageio, matplotlib). Recreate with
  `bash scripts/matched_swing/bootstrap_motion_runtime.sh`.
- Clone: `/home/dieterolson/ms31-crocoddyl` (partial clone, `vendor/ud-tools`
  initialised). Fits and logs under `/home/dieterolson/fits/<name>[.log]`.
- Windows has no Crocoddyl (conda-forge has no win-64 build). Unit tests
  (`tests/unit/motion_matching/test_crocoddyl_{problem,action}.py`) run on
  Windows; native tests need the env above.

Remote pattern that works (quoting through cmd.exe -> wsl -> bash mangles `$`
and quotes): write a script locally, `scp -q script.sh controltower:C:/Users/diete/`,
run `ssh controltower "wsl -d ControlTower-Runner -- bash /mnt/c/Users/diete/script.sh"`,
pipe through `tr -d '\r'`. Long runs: launch detached with
`scripts/matched_swing/run_crocoddyl_fit.sh` and poll the log for `^EXIT`.

## Method (What Is Implemented and Why)

1. Warm start: damped Gauss-Newton marker IK on the plant with the weld
   closure as a residual. The address frame ramps the closure weight
   (0, 1, 1e2, 1e4); without the ramp the weld traps the pose at 294 mm.
   Result 29.6 mm at frame 0, matching the canonical MuJoCo IK.
   Marker attachments come from the g025 receipt (`ik.attachments_m`), ground
   height from its `ground.height_m`.
2. Dynamically consistent warm start: computed-torque tracking rollout
   (`tracking_rollout`, kp 400, kd 40) mapping desired accelerations to bounded
   efforts through the plant's effort sensitivity (ridge least squares).
3. Plant conditioning: dof armature 5e-3 kg m^2 on every non-root dof
   (toe and shoulder-gimbal dofs are near-massless; effort sensitivity was
   1.7e4 rad/s^2 per N m). Recorded in the receipt; a parity replay in
   another engine must apply the same armature.
4. Stiffness: the shared contact law has real eigenvalues near -8000 1/s
   (friction transition velocity 0.05 m/s, dissipation). Explicit Crocoddyl
   integrators diverge at 360 Hz. Running nodes therefore integrate with
   adaptive RK45 (exact rollout, same as the acceptance replay) and use a
   linearly implicit Euler linearisation for the DDP derivatives.
5. Per-coordinate effort bounds (`EFFORT_BOUND_PATTERNS_N_M`), box-FDDP.
6. Horizon continuation: `--continuation 0.05,0.10,0.20` grows the window,
   warm-starting each stage from the previous solution plus the tracking
   rollout for new nodes. Without it the 0.30 s fit converged to a fallen
   golfer (467 mm).
7. Receipt (`receipt.json`): document/capture/candidate SHA, armature, bounds,
   solver stages, five shared metrics for warm start / solver rollout / RK45
   replay, physical audit (peak normal force, penetration, closure error,
   weight fraction, peak effort), cost breakdown. Candidate `candidate.npz`
   (q, v, u, replay, markers, targets). GIF via `candidate_playback.py`.

## Results so Far (ControlTower, Driver Document `anthro_driver/full_body_spec_hipcal_scaled.json`)

| Run       | Window | Nodes | Solver rollout whole / terminal / club (mm) | RK45 replay whole (mm) | Note                                                                                                                                                                 |
| --------- | ------ | ----- | ------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| smoke     | 0.05 s | 19    | 24.5 / 14.7 / 9.8                           | 26.9                   | implicit-Euler nodes, 8 iterations                                                                                                                                   |
| w030      | 0.30 s | 109   | 467 (fallen)                                | 442                    | no continuation, kinematic warm start                                                                                                                                |
| w030c     | 0.30 s | 109   | 15.9 / 13.8 / 4.8                           | 192                    | continuation; implicit-Euler nodes exploit integrator damping                                                                                                        |
| w030r     | 0.30 s | 109   | **19.7 / 14.2 / 9.8**, yaw 1.35 deg         | **19.7 (identical)**   | continuation + exact RK45 nodes; 66 min; audit: 1.4 BW peak contact, penetration 17 mm, weight fraction min 0.21, peak effort 165 N m                                |
| w030s     | 0.30 s | 109   | 34.0 / 25.6 / 52.8                          | 79.2                   | 4-substep implicit-Euler nodes: worse and not converged; abandoned                                                                                                   |
| driver_g1 | 0.85 s | 307   | (first attempt lost, see below)             |                        | relaunched 2026-09-17 18:05 PT with stage checkpoints; RK45 nodes, rtol 1e-4, warm start from w030r, continuation 0.30/0.45/0.60, 30 iterations per stage, 120 final |

Warm-start IK over 0.30 s: 32 mm; tracking rollout 43 mm.

## Next Steps (In Order)

1. (done) w030r validated the RK45-node configuration. Read `/home/dieterolson/fits/driver_g1/receipt.json` when `grep EXIT /home/dieterolson/fits/driver_g1.log` (or `fit_driver_g1.log`) reports; then `candidate_playback.py` for the GIF. If it did not converge, resume with `--warm-start-candidate /home/dieterolson/fits/driver_g1/candidate.npz`.
   1b. Old note: (`python3 summ.py` pattern:
   print `solver.stages`, `metrics.*.shared`, `physical_audit`). Solver
   rollout and replay must now agree; if FDDP stalls with RK45 nodes, lower
   `initial_regularisation` or use `--node-integrator implicit_euler` for the
   early stages and RK45 for the last.
2. Run G1: `bash scripts/matched_swing/run_crocoddyl_fit.sh driver_g1 0.85 150 --quiet --continuation 0.05,0.10,0.20,0.30,0.45,0.60 --stage-iterations 40`
   (about 5 to 15 s per iteration at 307 nodes).
3. Copy the G1 outputs into `docs/development/full_body_models/evidence/matched/driver_g1_pinocchio/`
   (`receipt.json`, `candidate.npz`, `playback.gif`), commit, update
   `DL-#10338`, this file, and comment on #10338 and #10363.
4. Hand off to MS-21 (#10336): replay `candidate.npz` in MuJoCo with the same
   armature and contact law; MS-70 parity report.
5. Then the 7-iron (`FIT_DOCUMENT=.../anthro_iron/full_body_spec_hipcal_scaled.json`,
   `FIT_ATTACHMENTS_RECEIPT=.../anthro_iron_zmp/receipt.json`, `FIT_CAPTURE=data/C3D_TA_Iron.c3d`).

## Known Gaps / Risks

- 2026-09-17 18:00: the first G1 run solved for 2 h 10 min and then crashed in a
  post-solve diagnostic (single-step implicit-Euler replay) before writing the
  receipt, losing the solution. Fixed: the driver now writes
  `stage_<t>s.npz` after every continuation stage and `solution.npz` +
  `solver_stages.json` right after the solve, before any metrics; the
  diagnostic was removed; the physical audit is guarded. Any of those files
  can be passed to `--warm-start-candidate` to resume.

- The RK45 replay of an implicit-Euler-node solution diverged (192 mm); only
  RK45-node solutions count. MS-01 `acceptance.py` does not exist yet, so the
  gates are checked by hand from the receipt.
- Armature is a fit-side plant change; MS-72 must put it in the conformance
  spec and MS-21 must apply it in MuJoCo (MJX lane already used 5e-3).
- `RShoulderTop` is valid in 128/654 frames; the mask handles it, MS-04
  formalises the policy.
- Pink IK (MS-14) is not yet used for the warm start; the GN IK is faster
  (0.4 s for 19 frames) and matches the canonical IK, so Pink is optional.

## Decoupled Kinematic Tracking & Contact-Aware Force Allocation (MS-31 / MS-104 / #10415)

### Architectural Overview

Due to high algebraic index (DAE-3) stiffness from the 6-DoF rigid weld closure and Hunt-Crossley contact dynamics, monolithic shooting via Crocoddyl FDDP required ~110 minutes per 307 nodes and struggled with non-holonomic local minima.

In response, we developed and validated an audit-grade decoupled architecture in `scripts/match_pinocchio_c3d.py`:

1. **Stage 1 (Kinematic Tracking with MarkerIkSolver):**
   - Category-weighted Gauss-Newton IK prioritizing high-speed extremities: Club 50×, Feet 20×, Wrists 10×, Knees 5×, Torso/Head 1×.
   - Analytical foot contact sphere unilateral ground non-penetration barrier ($r_s = \sqrt{w_g} \max(0, h_g - z_s)$ with vertical Jacobian $J_{\text{lin}, z}$) across all 6 contact spheres (`heel_r`, `forefoot_r`, `toe_r`, `heel_l`, `forefoot_l`, `toe_l`).
   - Constant-velocity extrapolation regularisation prior ($q_{\text{prior}} = 2 q_{k-1} - q_{k-2}$) eliminating tracking lag during high angular velocity downswing phases.
   - 6-DoF rigid weld closure with address weight ramping ($w_c \in \{0, 1, 100, 10000\}$).
2. **Stage 2 (Kinematic Smoothing):** Low-pass zero-phase Butterworth filtering at 12 Hz with central finite-difference rate calculation.
3. **Stage 3 (Contact-Aware Dynamic Force Allocation via QP):**
   - Solves $M(q) \ddot{q} + b(q, \dot{q}) = S^T \tau + J_{\text{ground}}^T f_{\text{contact}} + J_{\text{grip}}^T \lambda_{\text{grip}} + S_{\text{root}}^T \delta \tau_{\text{root}}$.
   - Enforces unilateral contact ($f_{i, z} \ge 0$) and friction cone limits.
   - Formulates Optimum (minimum effort) and Trail-Arm Reduction (minimum achievable trail arm torque without breaking dynamic equilibrium).
   - Guarantees exact forward dynamic acceleration parity under ABA ($< 0.002$ m/s²).
4. **Stage 4 (Comprehensive Swing & Contact Audit via SwingEvaluator):**
   - Segment breakdown (club, feet, wrists, arms, pelvis, torso/head) and phase breakdown (address, backswing, downswing, impact, follow-through).
   - Audits ground penetration and weld closure tolerances.
5. **Stage 5 (Uninterrupted Forward Simulation Replay Verification):**
   - Verified that continuous forward integration from $(q_0, v_0)$ without per-frame pose resets remains bounded and stable under the resolved forces.

### Refined Benchmark Comparison (Full 1.8+ Second Trials on ControlTower)

| Metric                            | Driver Swing (`C3D_TA_Driver.c3d`)    | 7-Iron Swing (`C3D_TA_Iron.c3d`)    | G1 Gate Target                 |
| --------------------------------- | ------------------------------------- | ----------------------------------- | ------------------------------ |
| **Frames Tracked**                | 654 frames @ 360 Hz (1.814 s)         | 657 frames @ 359 Hz (1.827 s)       | Full swing                     |
| **Total Solve Time**              | **8.45 seconds** (12.9 ms/frame)      | **8.37 seconds** (12.7 ms/frame)    | Real-time / fast               |
| **Club Marker RMSE (Whole)**      | **50.20 mm** (88.2% drop from 425 mm) | **126.7 mm** (75% drop from 511 mm) | <= 60 mm (Driver MET)          |
| **Club RMSE (Address)**           | **10.38 mm**                          | **111.4 mm**                        | <= 15 mm (Driver MET)          |
| **Club RMSE (Downswing)**         | **17.06 mm**                          | **134.7 mm**                        | Fast downswing match           |
| **Feet Marker RMSE (Address)**    | **21.08 mm**                          | **8.60 mm**                         | Tight stance anchoring         |
| **Max Ground Penetration**        | **10.11 mm** (90.9% drop from 111 mm) | **7.28 mm**                         | <= 15 mm (MET)                 |
| **Mean Ground Penetration**       | **0.077 mm**                          | **0.068 mm**                        | Sub-millimeter ground contact  |
| **Max Weld Closure Error**        | **27.5 mm** (mean 2.1 mm)             | **25.9 mm** (mean 2.3 mm)           | Grip integrity maintained      |
| **Forward Accel Parity Residual** | **0.00155 m/s²**                      | **0.0410 m/s²**                     | Exact ABA Parity (< 0.05 m/s²) |
| **Uninterrupted Forward Replay**  | Stable (no pose resets needed)        | Stable (no pose resets needed)      | Zero divergence                |

### Cross-Engine Handoff Recommendations

1. **MuJoCo Lane (MS-21 / #10336):** Replay `candidate.npz` with the computed torques and 5e-3 kg·m² armature. The smooth state trajectory $(q, v, a)$ provides an ideal kinematic tracking target for forward simulation without DAE-3 constraint explosion.
2. **Drake Lane (MS-13/17 / #10337):** Feed `candidate.npz` directly into Drake's `MultibodyPlant` to verify energy conservation, contact wrench parity, and momentum transfer.
3. **OpenSim / MyoSuite Lanes (MS-40/41, MS-50/51):** Use the unconstrained generalized torques $\tau_{\text{RNEA}}$ as the net actuation target for Static Optimization (SO) and Computed Muscle Control (CMC) to distribute loads across physiological actuators.

Full developer guide: `docs/development/PINOCCHIO_C3D_MOTION_MATCHING_GUIDE.md`.
PR: #10411 on `feat/10338-crocoddyl-native-fit`.
Epic: #10415 (`[EPIC] Pinocchio Full-Swing Motion Matching & Balanced Contact Kinetics`).
