# MS-31 Turnover: Native Crocoddyl Full-Body Fit (Pinocchio Lane)

Governing issue: #10338 (epic #10363, Matched Swing Program). Branch
`feat/10338-crocoddyl-native-fit`. Development-log entry `DL-#10338`.
Last updated 2026-09-17 15:55 PT by claude. Read this file first; it is kept current
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

| Run       | Window | Nodes | Solver rollout whole / terminal / club (mm) | RK45 replay whole (mm) | Note                                                                                                                                  |
| --------- | ------ | ----- | ------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| smoke     | 0.05 s | 19    | 24.5 / 14.7 / 9.8                           | 26.9                   | implicit-Euler nodes, 8 iterations                                                                                                    |
| w030      | 0.30 s | 109   | 467 (fallen)                                | 442                    | no continuation, kinematic warm start                                                                                                 |
| w030c     | 0.30 s | 109   | 15.9 / 13.8 / 4.8                           | 192                    | continuation; implicit-Euler nodes exploit integrator damping                                                                         |
| w030r     | 0.30 s | 109   | **19.7 / 14.2 / 9.8**, yaw 1.35 deg         | **19.7 (identical)**   | continuation + exact RK45 nodes; 66 min; audit: 1.4 BW peak contact, penetration 17 mm, weight fraction min 0.21, peak effort 165 N m |
| w030s     | 0.30 s | 109   | pending                                     | pending                | continuation + 4-substep implicit-Euler nodes (speed candidate)                                                                       |
| driver_g1 | 0.85 s | 307   | pending                                     | pending                | RK45 nodes, rtol 1e-4, warm start from w030r candidate, continuation 0.30/0.45/0.60, launched 2026-09-17 15:55 PT                     |

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

- The RK45 replay of an implicit-Euler-node solution diverged (192 mm); only
  RK45-node solutions count. MS-01 `acceptance.py` does not exist yet, so the
  gates are checked by hand from the receipt.
- Armature is a fit-side plant change; MS-72 must put it in the conformance
  spec and MS-21 must apply it in MuJoCo (MJX lane already used 5e-3).
- `RShoulderTop` is valid in 128/654 frames; the mask handles it, MS-04
  formalises the policy.
- Pink IK (MS-14) is not yet used for the warm start; the GN IK is faster
  (0.4 s for 19 frames) and matches the canonical IK, so Pink is optional.

## Parallel Lane

OpenSim/Moco phase A (MS-42, #10341) runs on ControlTower in
`/home/dieterolson/opensim-10003` by a separate agent (worktree
`_wt_claude_10341`, branch `feat/10341-moco-g1`); 0.10 and 0.30 s rungs
converged, 0.60 s was solving at last update. Its receipts land under
`docs/development/opensim_tour_matching/evidence/os7_moco_g1/`.
