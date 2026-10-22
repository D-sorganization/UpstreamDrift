# Snatch Same-Input Bundle: First and Second Pull (LIFT-5, #11745)

`snatch_bundle.npz` packages a 0.2 s window of the **snatch pull, centred on
the knee pass** as a same-input `InputBundle`. It is built by the back-squat
pipeline (see `../back_squat/README.md`) with the recipe in
`src/shared/python/lifting/lift_recipes.py`. The turnover, catch and recovery
are in the planar solution, but not in the window.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer solves the `snatch`
   exercise: an 80 kg, 1.78 m lifter, a 100 kg bar, 2.0 s and 151 samples.
   Its configuration is the timed via schedule of Tools #5653 (Tools issue
   #5652). Via poses sit at the parity standard's phase starts: knee pass at
   0.30, power position at 0.45, catch at 0.55 and recovery at 0.70. The catch
   uses the standard's 18.70° trunk and −87.8° knee. The bar model is
   phase-correct (`snatch_bar_path`): the bar hangs one arm length below the
   shoulder through the pulls and sits one arm length above it from the catch
   on. A smoothstep blends the two across the turnover.
2. **Hand-held bar.** `lift_hands_to_full_body` moves the left hand to the
   standard's snatch grip width (0.9 m, parity standard 2.1.0) and solves the
   arms. From the straight-armed seed, the arm solve at this grip settles in a
   hyperextended-elbow minimum: the bar is 139 mm off at 0.3 m. The hang search
   therefore retries the bracket's lower end from a −0.8 rad bent-elbow seed,
   which reaches 0.3 m within 2.3 mm. The model's arms then hang the bar at
   most 0.475 m below the shoulder midpoint. The optimizer's shoulder-to-bar
   offset is scaled by 0.689.
3. **Window.** A 0.2 s MuJoCo window centred on the knee pass (0.567 s)
   starts at 0.467 s: 200 steps at 1 ms.

## Measured Numbers

| Quantity                                   | Before (Single Via)     | After (Timed Schedule)                    |
| ------------------------------------------ | ----------------------- | ----------------------------------------- |
| Shoulder height, setup → 0.93 s            | 0.876 → 0.653 m (falls) | 0.876 → 1.373 m                           |
| Shoulder over the pull (0.1-0.45)          | falls                   | rises 0.390 m monotonically, peak 1.376 m |
| Optimizer bar height at the setup          | 1.565 m (overhead)      | 0.187 m (hanging)                         |
| Catch trunk lean (standard 18.70°)         | 59.3°                   | 18.70° (tolerance 3.87°)                  |
| Optimizer peak vertical bar velocity, pull | n/a                     | 1.21 m/s at 0.653 s                       |
| Model bar height (world), setup → end      | n/a                     | 0.462 → 1.884 m                           |
| Window bar rise, mean vertical velocity    | n/a                     | 0.123 m, 0.61 m/s                         |
| Window bar drift forward / lateral         | n/a                     | −13.7 mm / −0.3 mm                        |
| Bundle size                                | n/a                     | 231,194 bytes                             |

The IK residuals below are the maximum over all 151 frames, in metres,
rounded up to 0.1 mm. They are asserted in
`tests/unit/lifting/test_pull_bundles.py`:

| Landmark   | Max Residual (m) |
| ---------- | ---------------- |
| ankle_l    | 0.0473           |
| ankle_r    | 0.0502           |
| knee_l     | 0.0455           |
| knee_r     | 0.0431           |
| hip_l      | 0.0471           |
| hip_r      | 0.0473           |
| shoulder_l | 0.0198           |
| shoulder_r | 0.0112           |
| bar        | 0.0402           |

The grip closure is at most 1.75e-4 m, and the lowest foot sphere is 2.07 mm
into the ground. These maxima are set after the window:

| Part of the Lift           | Worst Residuals                 | Feet Into Ground |
| -------------------------- | ------------------------------- | ---------------- |
| Pull (0-0.45)              | knee_r 11.2 mm, bar 0.23 mm     | 0.69 mm          |
| Window (0.467-0.667 s)     | knee_r 8.1 mm, bar 0.06 mm      | 0.69 mm          |
| Turnover (0.50)            | bar 40.1 mm, shoulder_l 19.7 mm | 0.69 mm          |
| Catch and hold (0.53-0.77) | legs 43-50 mm                   | up to 2.07 mm    |

The pull and window rows match the deadlift. The full-body model cannot
follow the turnover or reach the planar deep-squat catch. A strict xfail
records this against the deadlift bounds; those bounds are not loosened.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` uses the back-squat protocol and fixed tolerances:

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 6.2e-14                 | 2.9e-12                          | 2.0e-14                               | 9.5e-15                        |
| Pinocchio | 4.7e-14                 | 2.1e-12                          | 2.2e-14                               | 9.9e-15                        |
| OpenSim   | 1.6e-15                 | 1.1e-12                          | 9.9e-15                               | 1.7e-15                        |

Every engine meets both L2 bounds (1e-3 m frames, 1e-6 rad coordinates). They
hold over the full open-loop window and over the re-anchored segments.

## Current Records (Whole-Row Lift Body, #13037)

The sections above were measured on the earlier lift body and stay for
history. The bundle and `bar_path_parity.json` are now regenerated on the AN-1
lift body built from the whole de Leva Table 4 rows (#13102) with the default
10 deg tibial torsion (#12814), with the same Tools #5653 code (`b7f36b35`).
That code reproduces the earlier committed bundle on the earlier body to
1.3e-13 rad. The recorded `tools_pin` is unchanged.

IK residuals (maximum over all 151 frames, mm): ankle_l 42.45, ankle_r 45.00,
knee_l 37.16, knee_r 34.28, hip_l 40.64, hip_r 40.66, shoulder_l 8.97,
shoulder_r 15.09, bar 19.47. Grip closure at most 2.35e-4 m, lowest foot
sphere 1.88 mm into the ground, bundle 231,579 bytes. Only `shoulder_r` and
the closure exceed their regression snapshots; they are re-derived with the
same rounding (0.0151 m and 2.4e-4 m; #13041, issuecomment-6243170509).

| Part of the Lift           | Worst Residuals                 | Feet Into Ground |
| -------------------------- | ------------------------------- | ---------------- |
| Pull (0-0.45)              | knee_r 11.3 mm, bar 0.23 mm     | 0.69 mm          |
| Window (0.467-0.667 s)     | knee_r 8.1 mm, bar 0.06 mm      | 0.69 mm          |
| Turnover (0.50)            | bar 19.5 mm, shoulder_r 15.1 mm | 0.93 mm          |
| Catch and hold (0.53-0.77) | legs 34-45 mm                   | up to 1.88 mm    |

The strict xfail against the deadlift bounds still fails, so it stays.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 7.3e-14                 | 3.3e-12                          | 3.0e-14                               | 2.2e-15                        |
| Pinocchio | 5.5e-14                 | 2.5e-12                          | 3.2e-14                               | 3.2e-15                        |
| OpenSim   | 2.9e-15                 | 4.1e-13                          | 9.2e-15                               | 1.2e-15                        |

The run stays at eight RK4 substeps: |z| 2.37 (amplification 0.53), inside
the 2.785 limit.

## Barbell Bundle and OpenSim IK/ID (OSL-3, #13041)

`snatch_barbell_bundle.npz` (SHA-256 `24c7e8e4c140e2e3a52047851fd460e07e871dc17573e96414b6a493e9d4dacc`)
packages the snatch pull window with the LIFT-4 barbell body and load,
built by `scripts/generate_barbell_squat_evidence.py --lift snatch --part bundle`
from `snatch_bundle.npz`.

### Bar-Hold Metrics (Provenance)

The barbell hold re-tracks the snatch pull standing pose with the bar carried by the hands:
- **Bar deviation:** 162.03 mm (0.16203 m) against the reference trajectory.
- **Left hand force:** peak magnitude 1073.52 N (mean [−13.62, −17.79, 635.96] N),
  peak torque magnitude 59.00 Nm.
- **Right hand force:** peak magnitude 1414.29 N (mean [96.76, −13.31, 734.45] N),
  peak torque magnitude 59.00 Nm.

### OpenSim Inverse Kinematics and Inverse Dynamics Receipt

The OpenSim IK/ID receipt (`docs/development/lifting/evidence/opensim_ik_id/receipts.json`)
is measured with real OpenSim 4.6 and the RajagopalLaiUhlrich2023 model over 41 frames
(stride 5):
- **Marker kinematics:** RMS error 0.0089 m (0.89 cm, within 0.0200 m threshold),
  maximum marker error 0.0579 m (exceeds 0.0400 m threshold due to the bar marker).
  The kinematics fails the best-practice marker gate (`within_best_practice = False`).
- **Segment angles:** sagittal segment-angle RMS error 0.0095 rad (0.54°, well within
  the 0.0349 rad / 2.0° threshold).
- **Arm reach:** Left hand reach miss 7.68 µm (7.6825e-6 m), right hand reach miss
  7.57 µm (7.5725e-6 m), well within the 1.0 mm reach bound.
- **Coordinate range violations:** `pro_sup_l` (18 frames, min −0.00037 rad, range
  [0.0, 2.09] rad) and `pro_sup_r` (18 frames, min −0.00038 rad, range [0.0, 2.09] rad).
- **Hicks residual gate:** peak external force 3102.39 N; force threshold 155.12 N
  (5% of peak external force); moment threshold 27.10 Nm (1% of COM height × peak external force).
  The residual gate fails for both forces and moments:
  - Pelvis force: peak $t_x$ 463.57 N (RMS 188.08 N), peak $t_y$ 290.28 N (RMS 61.66 N),
    peak $t_z$ 148.30 N (RMS 62.19 N); max force 463.57 N vs 155.12 N threshold (`forces_within = False`).
  - Pelvis moment: peak tilt 202.60 Nm (RMS 85.61 Nm), peak list 89.87 Nm (RMS 33.33 Nm),
    peak rotation 92.32 Nm (RMS 40.71 Nm); max moment 202.60 Nm vs 27.10 Nm threshold (`moments_within = False`).
- The residual gate failures are strictly xfailed in `tests/unit/lifting/test_opensim_lift_ik_id.py`
  with measured numbers; gates are not loosened.

## Limitations

- **Pull window only.** The turnover and catch frames of the full-body IK are
  not credible (see above). Fixing this needs a full-body catch that the
  model's ankle and arm ranges can reach.
- **Hand-held bar in plain bundle.** In `snatch_bundle.npz` the bar is held
  on the club `Grip` frame with no barbell body or load (LIFT-4, #11744);
  `snatch_barbell_bundle.npz` provides the barbell body and load and is
  registered in `SAME_INPUT_LIFT_BUNDLES`.
- **OpenSim residual gate.** OpenSim inverse dynamics fails the Hicks residual
  gate (forces 463.57 N vs 155.12 N, moments 202.60 Nm vs 27.10 Nm).
  Results are software receipts, not physical acceptance.
- **Scaled bar offset.** The bar offset is scaled to the model's arm hang
  (0.689).
- **Provenance pin.** The bundle and `bar_path_parity.json` are regenerated
  on the vendor pin bump to the Tools #5653 merge commit (`a4595051`); the
  recorded `tools_pin` is now that commit. Against the bundle generated on the
  PR head (`b7f36b35`) the efforts differ by at most 5.2e-10 and the reference
  coordinates by 2.1e-13 rad.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift snatch
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift snatch
python3 scripts/generate_barbell_squat_evidence.py --lift snatch --part bundle --source-commit <sha>
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
