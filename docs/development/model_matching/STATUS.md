# Model-Matching Status Report

MATCH-1 (#14443), part of epic #14447. This page records the best current fit
per physics engine against the private tour-average driver capture (capture-A:
34 markers, 360 Hz, 654 frames). The table is generated from
`model_matching_status.json` beside this page; edit the ledger and run
`python3 scripts/generate_model_matching_status.py --write`.

## Summary

- **No fit is physically accepted.** The best whole-body match (MuJoCo,
  50.9 mm marker RMS) is REJECTED at the G3 acceptance horizon.
- **The newest fits are OS9 (2026-10-25) and the MSK club model
  (2026-10-28, unverified).** Neither is a dynamic whole-body fit.
- The best OpenSim marker fit (OS9, 32.5 mm) is kinematic IK only and fails
  its plausibility gate.
- The committed Drake G1 receipt reports 0.0 mm and PASSED, but it scored the
  warm-start markers rather than the Drake rollout (#11567). Do not cite it as
  a fit.
- MyoSuite and Pinocchio have receipts, but only kinematic replays or fits with
  body-marker errors of 134-186 mm.
- Nothing here is ready for public use.

## Per-Engine Status

<!-- generated:model-matching-status -->
As of 2026-11-04. Public ready: no.

| Engine | Model | Approach | Key Parameters | Marker RMS (mm) | Joint-Angle RMS (deg) | Date | Verdict | Blocker | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| mujoco | full-body-v1 (44 coordinates), de Leva 1996 scaled, hip-calibrated, hands welded to club | Marker IK (26.1 mm) then forward dynamics with KKT feedback tracking | 12 Hz tracking cutoff; address calibration 5.6 mm; peak joint torque 537 N*m | 50.9 | 2.4 | 2026-10-10 | REJECTED (G3): closure residual 62.2 mm > 5 mm, weight fraction 0.04-2.82 outside 0.2-3.0, contact forces and replay artifacts missing | #13177 refit on the v2 model; #13191 grip coupling | Private execution receipt (fit s32-fw-drv, capture sha256 cbcb4d84...a0971d); joint RMS is FD vs its own IK reference, not ground truth |
| opensim | Rajagopal skeleton golf_humanoid, tour-scaled, address-registered markers (OS9) | Kinematic marker IK only; no club-hand coupling, no muscles | Address calibration 19.1 mm; max frame RMS 49.6 mm; humerus scales 1.46/1.24 | 32.5 | not measured | 2026-10-25 | Kinematics only; plausibility FAILS (trunk 30.2 deg vs 30 deg bound, implausible humerus scale) | #13991 anatomical plausibility; #14429 OpenSim classify CI hang. The OpenSim G1 dynamic fit (evidence/matched/driver_g1_opensim, 2026-09-20) is worse: 245 mm, solver hit max iterations | docs/development/opensim_tour_matching/evidence/os9_address_registration/receipt.json; humerus scales from docs/development/opensim_tour_matching/EPIC_10003.md and docs/development/HANDOFF.md |
| opensim-msk-club | golf_humanoid Rajagopal skeleton with club, no muscles | Club-tracking kinematics (msk_club_tracking.track_swing), turn_feet mode | Lead grip error <= 1.8 mm; trail grip gap up to 17.8 mm | 62.0 (UNVERIFIED) | not measured | 2026-10-28 | Kinematics only; trail-hand gap open | #13191 grip coupling; #13177 refit | Render notes of 2026-10-28 only; landmark RMS 20 mm at address rising to 62-90 mm; no receipt re-checked (UNVERIFIED) |
| pinocchio | Full-body Pinocchio model with decoupled club (#10415) | Kinematic marker fit; separate G1 forward rollout REJECTED (2757 mm whole-marker RMSE) | Club segment 50.2 mm, address club 10.4 mm, arms 191.6 mm, torso/head 180.7 mm; 654 frames | 133.5 | not measured | 2026-09-18 | Kinematics only; body markers far off (only the club segment is close) | No current whole-body Pinocchio refit scoped | evidence/matched/driver_full_pinocchio/receipt.json and evidence/matched/driver_g1_pinocchio/receipt.json |
| drake | Drake full-body trajectory optimization (MS-30, #10337); v2 golfer load and mass-parity only (#14028) | IPOPT trajectory optimization warm-started from the Crocoddyl G1 candidate | G1 horizon; 100 iterations; knot control | not measured | not measured | 2026-09-20 | No credible fit; committed receipt is invalid | Re-run and re-score the Drake rollout after #11567; the stale PASSED receipt should be relabelled | evidence/matched/driver_g1_drake/receipt.json reports 0.0 mm on every metric and PASSED; not credible because the fit scored warm-start markers, not the Drake rollout (#11567) |
| myosuite | MyoSuite replay of the G1 candidate | Kinematic replay only; dynamics not run | Parity limit 15 mm; early 62.9 mm, terminal 297.9 mm | 185.7 | not measured | 2026-09-21 | REJECTED: parity failed, no dynamics | MyoSuite dynamics stage never run; model port not maintained | evidence/matched/driver_g1_myosuite/receipt.json (parity marker RMS 195.2 mm, failed) |
<!-- end-generated:model-matching-status -->

## How to Read the Table

- **Marker RMS** is the whole-swing mean marker error in millimetres.
- **Joint-angle RMS** for MuJoCo is forward dynamics against its own IK
  reference, not against ground-truth joint angles; none of the fits has a
  ground-truth joint-angle comparison.
- **UNVERIFIED** marks values taken from documentation or render notes that
  were not re-checked against an execution receipt for this report.
- **not measured** means the metric was never computed for that fit.
- Feedback-tracked forward dynamics is not open-loop replay; an animation or
  a low marker RMS does not establish physical acceptance.
