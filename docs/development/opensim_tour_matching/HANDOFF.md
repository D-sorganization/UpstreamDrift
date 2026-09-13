# OpenSim Matching Handoff

Updated 2026-09-13 (claude, lease on #10003). Epic
[#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003), plan
[EPIC_10003.md](EPIC_10003.md). Branch `docs/10003-opensim-matching-epic`,
worktree `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003`,
PR not created. Sibling full-body epic #10062
([design](../full_body_models/EPIC_FULL_BODY_CONTACT.md)) shares the capture
contract and the calibration algorithm from this lane.

## State in One Paragraph

OS-0, OS-1 and an OS-2 runtime preflight are done; OS-3 (marker placement
plus inverse kinematics) runs end to end on ControlTower and reaches a
full-body marker RMS of 6.5 cm on a 33-frame subsample of the tour swing with
an unscaled generic model. That is kinematic feasibility, not a match and not
dynamics. OS-4 (Moco tracking) and OS-5 (global sextic efforts) have not
started. Everything below runs without OpenSim except the two ControlTower
drivers, which use the isolated venv described under Runtime.

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

Defects found, in priority order for the next agent:

1. The packaged `golf_humanoid.osim` locks 21 coordinates (all arm, lumbar,
   subtalar and toe coordinates) inherited from the gait base. The OS-3 driver
   unlocks the 17 arm and lumbar coordinates in a tracking variant written
   beside the outputs (`base_unlocked.osim`, `golf_humanoid_tour_markers.osim`).
   The builder `scripts/build_humanoid_osim.py` should do this permanently for
   a golf variant, with a test, and the packaged model regenerated.
2. Coordinate clamps bind: `arm_flex_r` spans exactly ±90°, `lumbar_rotation`
   reaches −90°. Ranges must be widened for a golf swing and recorded.
3. The model is unscaled: segment lengths are the Rajagopal defaults; the
   club body has placeholder length and inertia. Scaling from first-frame
   marker pairs (ScaleTool or the anthropometrics pipeline) is OS-3b.
4. The alternation oscillates (0.065 then 0.087). Keep the best iteration, or
   damp the placement update; report per-marker RMS per iteration.
5. Only 33 frames were used; the full 654-frame IK and its per-frame RMS
   plot are still to be produced.

## Runtime

ControlTower WSL distro `ControlTower-Runner`, venv
`/home/dieterolson/opensim-10003` (Python 3.12.3; `pip install opensim`
gives 4.6 with Moco; also numpy, scipy, ezc3d, defusedxml, pytest). The
Pinocchio venv is untouched. A runtime bundle of the needed source files is
unpacked at `/home/dieterolson/opensim-10003-runtime` (namespace layout, no
package `__init__` beyond `tour_matching`). Launch through
`systemd-run --user` because processes spawned from an SSH session die at
logout; example (from the launch scripts used here):

```bash
RT=/home/dieterolson/opensim-10003-runtime; W=/mnt/c/Users/diete/opensim-10003
systemd-run --user --unit=opensim-os3 --collect --quiet -p WorkingDirectory=$RT -E PYTHONPATH=$RT \
  bash -c "/home/dieterolson/opensim-10003/bin/python docs/development/opensim_tour_matching/os3_calibrate_ik.py \
  --osim $RT/golf_humanoid.osim --trc $RT/docs/development/opensim_tour_matching/evidence/tour_average_tracked.trc \
  --output $W/os3-<name> --stride 20 --iterations 4 > $W/os3-<name>.log 2>&1"
```

Rebuild the bundle from the repository files listed in
`os3_calibrate_ik.py` imports whenever those files change; record the driver
SHA256 in the receipt (the driver does this for its inputs already).

## Next Bounded Tasks (Lower-Agent Ready)

Copy-ready prompt: [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

- OS-2b: [COMPLETE] golf variant in the builder (unlock 17 coordinates, widen clamps,
  club length from the capture's club markers), regenerated model, tests merged in #10073.
- OS-3b: [COMPLETE] segment scaling from first-frame marker pairs; keep-best iteration;
  full 654-frame IK; per-frame and per-marker RMS report; overlay animation (PR #10075).
- OS-4: MocoTrack pilot on the scaled model with the calibrated MarkerSet
  over a 0.85 s prefix, coordinate actuators only, receipts.
- OS-5: global degree-six effort profile fit with uninterrupted replay,
  compared on the five shared metrics (whole, early, terminal, club, yaw).

## Known Limits

Three head markers share the torso body (no head body), exactly as the
native Hub limitation; the capture has no force plates, so ground reaction
is inferred, never measured; the right-hand-only club weld leaves the left
hand free, unlike the native closed loop. None of the OpenSim results implies
Pinocchio, MuJoCo, Drake or Simscape equivalence.
