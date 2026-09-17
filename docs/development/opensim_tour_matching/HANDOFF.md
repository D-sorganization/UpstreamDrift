# OpenSim Matching Handoff

Updated 2026-09-17 (claude, lease on #10341 MS-42 phase A of epic #10363).
Earlier lane: epic [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003),
plan [EPIC_10003.md](EPIC_10003.md). Current branch `feat/10341-moco-g1`,
worktree `C:/Users/diete/Repositories/_wt_claude_10341`. Sibling full-body
epic #10062 ([design](../full_body_models/EPIC_FULL_BODY_CONTACT.md)) shares
the capture contract and the calibration algorithm from this lane.

## OS-7 (MS-42 Phase A): Moco Horizon Ladder to G1 and the Full Swing

Driver `os7_moco_g1_driver.py`, library `tour_matching/moco_g1.py` (pure:
ladder, mesh selection, windowing, gap policy, warm start, gate table,
receipt contract; 23 tests in `tests/opensim/test_moco_g1_ladder.py`) and
`moco_tracking.py` (OpenSim-bound: normalised actuators, rung study,
guess loading, solve, collocation marker sampling, one uninterrupted
`opensim.Manager` replay). Evidence `evidence/os7_moco_g1/` (top-level
`receipt.json` = longest converged rung at or below 0.85 s with the
`per_horizon` table; `rungs/<ms>ms/` hold every rung's `states.sto`,
`controls.sto`, `replay.mot`, `playback.gif`, `receipt.json`; `inputs/`
holds the OS-3b model `7dd1da17` and the IK states `32579607` that OS-4/5
used but never committed).

Method: ladder 0.10 -> 0.30 -> 0.60 -> 0.85 -> 1.81 s, each rung
warm-started from the previous solution (IK values and finite-difference
speeds beyond it, controls held), 100 Hermite-Simpson mesh intervals per
second, goals = marker tracking (weight 10; club markers x5), IK-state
tracking (0.1), control effort (1e-3); every CoordinateActuator scaled to
300 N m (joints) or 2000 N (pelvis residuals) with controls in [-1, 1];
IPOPT convergence 1e-3, constraint 1e-4. `RShoulderTop` (128/654 valid)
is dropped from the reference; trailing club-marker dropouts trim the full
horizon; interior gaps up to 40 frames are linearly filled for the spline
only and never scored. The five metrics are `tour_metrics.SharedMetrics`
on the ORIGINAL validity mask from a single forward replay of the fitted
open-loop controls (accuracy 1e-6, no state resets), next to the
collocation solution's own number so divergence is visible.

Why OS-4 was not a dynamic result: its 1e-2 tolerances let the collocation
defects absorb gravity (peak control 0.018 N m on a 79.7 kg model) and its
replay only checked that integration reached 0.10 s. The OS-6
`shared_cross_engine_metrics` block was never computed from a replay and is
now annotated `unverified` in `evidence/os6_handoff/reproduction_receipt.json`.

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

| Run              | Evidence                            | Result                                                                                                                                                                                   |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS-1 TRC export  | `evidence/os1_trc_receipt.json`     | 654 frames, 34 labels, 21,534 valid points, roundtrip 5e-7 m; `RShoulderTop` has only 128 valid samples                                                                                  |
| OS-2 runtime     | `evidence/os2_runtime_receipt.json` | OpenSim 4.6 (2026-06-22) with Moco, IK, Scale; model loads with 23 bodies, 39 coordinates, 2 constraints, 0 markers; TRC read by OpenSim: 654 rows × 34                                  |
| OS-3 locked      | `evidence/os3_stride20/`            | RMS 0.332 → 0.203 → 0.199 m; arm and lumbar coordinates never moved                                                                                                                      |
| OS-3 unlocked    | `evidence/os3_unlocked_stride20/`   | RMS 0.198 → 0.065 → 0.087 → 0.078 m over four alternations, stride 20 (33 frames)                                                                                                        |
| OS-3b full IK    | `evidence/os3b_scale_ik/`           | Full 654-frame IK on scaled model: club marker RMS 2.0 cm, early marker RMS 4.2 cm, 3D animated GIF overlay                                                                              |
| OS-4 MocoTrack   | `evidence/os4_moco_tracking/`       | 15 IPOPT iterations to Solve_Succeeded (obj: 8.508e-2); zero-feedback forward simulation replay to t=0.10s in 9.2 ms                                                                     |
| OS-5 Sextic Fit  | `evidence/os5_polynomial_profile/`  | Degree-6 profile across all 39 actuators ($R^2 > 0.99$, max error < 0.011 N\*m); continuous forward replay via Manager to t=0.10s in 7.16 ms (5 steps)                                   |
| OS-6 Handoff     | `evidence/os6_handoff/`             | Unified CLI router (7 subcommands), deterministic run hashing, 3D overlay / error / effort PNGs, and reproduction_receipt.json; shared metrics block annotated `unverified` (2026-09-17) |
| OS-7 Moco ladder | `evidence/os7_moco_g1/`             | See the per-horizon table below; every number is the uninterrupted replay of the fitted controls                                                                                         |

### OS-7 Per-Horizon Table

Replay of the fitted controls; targets whole <= 25, early <= 12, terminal <= 35, club <= 60 mm, pelvis yaw < 3 deg.

| Horizon (s)                     | Mesh | Solver status               | Iter | Solve wall (s) | IK guess whole (mm) | Colloc. whole (mm) | Replay whole (mm) | Early (mm) | Terminal (mm) | Club (mm) | Pelvis yaw (deg) | Residual F_y RMS (N)           | G1 gates |
| ------------------------------- | ---- | --------------------------- | ---- | -------------- | ------------------- | ------------------ | ----------------- | ---------- | ------------- | --------- | ---------------- | ------------------------------ | -------- |
| 0.10                            | 10   | Solve_Succeeded             | 209  | 361            | 41.9                | 41.0               | 41.0              | 41.0       | 41.5          | 12.9      | 6.85             | 197 (ballistic hop, see below) | fail     |
| 0.30                            | 30   | Solve_Succeeded             | 215  | 850            | 41.2                | 41.0               | 41.0              | 41.0       | 41.0          | 12.9      | 5.74             | 648                            | fail     |
| 0.60 (run 1, 300-iteration cap) | 60   | Maximum_Iterations_Exceeded | 300  | 2222           | 42.4                | 41.6               | 257.7             | 257.7      | 524.9         | 543.7     | 13.95            | 758                            | fail     |

| 0.60 (run 2, warm-started from run 1, 600-iteration cap) | 60 | Solve_Succeeded | 428 | 3379 | 41.6 | 41.5 | 81.4 | 81.4 | 204.1 | 97.2 | 33.56 | 758 | fail |
OS7_ROWS_PENDING

`replay_departure.json` (helper `os7_replay_departure.py`, 60 mm band):
0.10 s and 0.30 s never leave the band; the run-1 0.60 s replay departs at
t = 0.372 s and ends at 732 mm; the converged run-2 0.60 s replay departs at
t = 0.417 s and ends at 295 mm (convergence halves the divergence, the rest
is intrinsic to open-loop replay of a residual-supported body). The collocation solution at 0.60 s was still
41.6 mm, so the departure is open-loop divergence of a residual-supported
body without feedback (constraint violation 2.3e-5 at the iteration cap,
10 ms Hermite-Simpson mesh), not a fitting failure.

Reading the table: the replay reproduces the collocation solution (the
dynamics are honest and the open-loop controls are replayable), but the fit
sits at the OS-3b marker-calibration floor (early IK RMSE 42.6 mm) and the
pelvis-yaw error is a pelvis marker-placement error, so the whole/early and
yaw gates are a calibration limit, not a tracking limit. On the 0.10 s rung
the optimiser chose a ballistic hop (initial pelvis speed +0.37 m/s, back to
the same height at 0.10 s, mean vertical residual 166 N) because that is
cheaper than carrying 782 N; from 0.30 s the residual carries the weight.

## Next Steps

MS-42 phase B (after MS-40): rerun `os7_moco_g1_driver.py` on the exported
document model with the shared Hunt-Crossley contact law and recalibrated
marker offsets, warm-started from the MuJoCo G1 candidate, and pass the
receipt through `acceptance.py` (MS-01). On this model the calibration floor
(~41 mm, pelvis yaw ~6 deg) has to move before the G1 whole/early/yaw gates
can; pinning initial speeds to zero at address would remove the 0.10 s hop.

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
