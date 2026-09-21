# OpenSim Matching Handoff

Updated 2026-09-13 (claude, lease on #10003). Epic
[#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003), plan
[EPIC_10003.md](EPIC_10003.md). Branch `docs/10003-opensim-matching-epic`,
worktree `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003`,
PR not created. Sibling full-body epic #10062
([design](../full_body_models/EPIC_FULL_BODY_CONTACT.md)) shares the capture
contract and the calibration algorithm from this lane.

## State in One Paragraph

OS-0 through OS-6 are complete. The OpenSim tour-average swing matching program
has delivered an end-to-end repeatable workflow: baseline environment audit (OS-0),
canonical TRC/marker contract (OS-1), golf coordinate unlocking and club calibration (OS-2b),
segment scaling and 654-frame IK feasibility baseline (OS-3b), constraint-aware dynamic
Moco tracking with zero-feedback forward replay (OS-4), global degree-six polynomial
effort profiles across all 39 actuators with forward simulation replay (OS-5), and
unified CLI router, headless visualization, and clean-machine reproduction handoff (OS-6).
OpenSim matching lane (#10003) is fully closed and ready for transition to multi-engine
contact modeling (#10062).

## What Exists (All Test-First)

- Shared capture contract `src/shared/python/motion_matching/tour_capture_contract.py`:
  frozen identity of `data/C3D_TA_Driver.c3d` (SHA256 545405cc…, 360 Hz,
  654 frames, metres, Y-up), 38 labels grouped by segment (head, trunk,
  pelvis, arms, legs, club, unassigned), validated loader (invalid = negative
  residual or nonfinite), `tracked_labels()` = 34 labels. Tests:
  `tests/opensim/test_tour_capture_contract.py` (5).
- OpenSim package `src/engines/physics_engines/opensim/python/tour_matching/`:
  `marker_map.py` (34 labels → golf_humanoid bodies; head markers on `torso`
  because the model has no head body, knees on femur, ankles on tibia, toes on
  calcn, club markers on `Club`), `trc.py` (TRC writer/reader; missing markers
  as explicit `NaN` cells because OpenSim trims trailing empty cells and then
  rejects the row), `marker_set.py` (safe parse, `attach_marker_set`,
  `unlock_coordinates`, `locked_coordinates`, `write_model`),
  `marker_calibration.py` (alternating placement/IK with injected FK and IK,
  Kabsch reuse). Tests: `test_tour_marker_map.py` (4), `test_trc_export.py`
  (3), `test_marker_set_authoring.py` (3), `test_marker_calibration.py` (2).
- Drivers under this directory: `os0_runtime_audit.py` (prior session),
  `os1_export_trc.py` (local; writes `evidence/tour_average_tracked.trc`,
  SHA256 eac6c880…, receipt `evidence/os1_trc_receipt.json`),
  `os2_runtime_qualification.py` and `os3_calibrate_ik.py` (ControlTower).

## Measured Results

| Run             | Evidence                            | Result                                                                                                                                                  |
| --------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS-1 TRC export | `evidence/os1_trc_receipt.json`     | 654 frames, 34 labels, 21,534 valid points, roundtrip 5e-7 m; `RShoulderTop` has only 128 valid samples                                                 |
| OS-2 runtime    | `evidence/os2_runtime_receipt.json` | OpenSim 4.6 (2026-06-22) with Moco, IK, Scale; model loads with 23 bodies, 39 coordinates, 2 constraints, 0 markers; TRC read by OpenSim: 654 rows × 34 |
| OS-3 locked     | `evidence/os3_stride20/`            | RMS 0.332 → 0.203 → 0.199 m; arm and lumbar coordinates never moved                                                                                     |
| OS-3 unlocked   | `evidence/os3_unlocked_stride20/`   | RMS 0.198 → 0.065 → 0.087 → 0.078 m over four alternations, stride 20 (33 frames)                                                                       |
| OS-3b full IK   | `evidence/os3b_scale_ik/`           | Full 654-frame IK on scaled model: club marker RMS 2.0 cm, early marker RMS 4.2 cm, 3D animated GIF overlay                                             |
| OS-4 MocoTrack  | `evidence/os4_moco_tracking/`       | 15 IPOPT iterations to Solve_Succeeded (obj: 8.508e-2); zero-feedback forward simulation replay to t=0.10s in 9.2 ms                                    |
| OS-5 Sextic Fit | `evidence/os5_polynomial_profile/`  | Degree-6 profile across all 39 actuators ($R^2 > 0.99$, max error < 0.011 N\*m); continuous forward replay via Manager to t=0.10s in 7.16 ms (5 steps)  |
| OS-6 Handoff    | `evidence/os6_handoff/`             | Unified CLI router (7 subcommands), deterministic run hashing, 3D overlay / error / effort PNGs, and reproduction_receipt.json                          |

## Next Steps

OpenSim matching epic #10003 is complete through OS-6.
Transition to multi-engine contact model program #10062 (Pinocchio #10065, MuJoCo #10066, Drake #10067 parity).

- OS-2b: [COMPLETE] golf variant in the builder (unlock 17 coordinates, widen clamps,
  club length from the capture's club markers), regenerated model, tests merged in #10073.
- OS-3b: [COMPLETE] segment scaling from first-frame marker pairs; keep-best iteration;
  full 654-frame IK; per-frame and per-marker RMS report; overlay animation (PR #10075).
- OS-4: [COMPLETE] MocoTrack pilot on the scaled model with the calibrated MarkerSet
  over tracking horizon, coordinate actuators only, receipts (PR #10078).
- OS-5: [COMPLETE] global degree-six effort profile fit across all 39 actuators with
  continuous zero-feedback forward simulation replay via opensim.Manager (PR #10081).
- OS-6: [COMPLETE] Repeatability, visualization, and handoff (CLI integration, 3D overlays,
  error timecourses, clean-machine reproduction receipt).

## Known Limits

Three head markers share the torso body (no head body), exactly as the
native Hub limitation; the capture has no force plates, so ground reaction
is inferred, never measured; the right-hand-only club weld leaves the left
hand free, unlike the native closed loop. None of the OpenSim results implies
Pinocchio, MuJoCo, Drake or Simscape equivalence.
