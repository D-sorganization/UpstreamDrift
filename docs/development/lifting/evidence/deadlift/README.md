# Deadlift Same-Input Bundle (LIFT-5, #11745)

`deadlift_bundle.npz` packages a conventional deadlift as a same-input
`InputBundle`, built by the back-squat pipeline (see
`../back_squat/README.md`) with the recipe in
`src/shared/python/lifting/lift_recipes.py`.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer solves the `deadlift`
   exercise (floor setup, lockout): 80 kg, 1.78 m lifter, 100 kg bar, 2.0 s,
   151 samples. The bar uses the optimizer's `deadlift` bar model: one arm
   length (0.689 m) straight below the shoulder.
2. **Hand-held bar.** `lift_hands_to_full_body`
   (`src/shared/python/lifting/bundles.py`) moves the left hand to the
   standard's deadlift grip width (0.45 m, parity standard 2.1.0) and solves
   the arms with the bar long axis held medio-lateral. The model's arms hang
   the bar at most 0.551 m below the shoulder midpoint (measured by bisection
   on the standing and the setup pose), so the optimizer's shoulder-to-bar
   offset is scaled by 0.799 before it is retargeted onto the model.
3. **Window.** A 0.2 s MuJoCo tracking window centred on the knee pass (first
   sample with the bar at the optimizer's knee height, 0.533 s) starts at
   0.427 s: 200 steps at 1 ms.

## Measured Numbers

| Quantity                                          | Value              |
| ------------------------------------------------- | ------------------ |
| Optimizer bar height above the ankle, setup → end | 0.201 → 0.680 m    |
| Optimizer peak vertical bar velocity              | 0.56 m/s at 0.12 s |
| Model bar height (world), setup → lockout         | 0.456 → 0.935 m    |
| Window bar rise, mean vertical velocity           | 0.066 m, 0.33 m/s  |
| Window bar drift forward / lateral                | 4.3 mm / −1.2 mm   |
| Bundle size                                       | 213,783 bytes      |

IK residuals (maximum over all 151 frames, metres, rounded up to 0.1 mm;
asserted in `tests/unit/lifting/test_pull_bundles.py`): ankle_l 0.0080,
ankle_r 0.0040, knee_l 0.0227, knee_r 0.0100, hip_l 0.0149, hip_r 0.0144,
shoulder_l 0.0096, shoulder_r 0.0078, bar 0.0040. Grip closure at most
1.7e-7 m, lowest foot sphere 0.69 mm into the ground.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` (measured at commit `37aaf459`) uses the back-squat
protocol and the same fixed tolerances.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 1.1e-8                  | 2.0e-4                           | 4.4e-5                                | 8.4e-9                         |
| Pinocchio | 1.1e-8                  | 2.0e-4                           | 4.3e-5                                | 8.3e-9                         |
| OpenSim   | 1.1e-8                  | 2.1e-4                           | 4.2e-5                                | 8.2e-9                         |

The bar and every spec frame agree far inside the 1 mm L2 frame bound. The
1e-6 rad coordinate bound is not met: nine of the ten re-anchored segments
end within 6.2e-10 rad, but the segment from 140 to 160 ms ends about
4.4e-5 rad from MuJoCo, and the full window drifts to 2.0e-4 rad (worst
final coordinate `subtalar_angle_r`). This is the same foot-sphere signature
as the back squat. The coordinate gate is a strict xfail in
`tests/unit/lifting/test_pull_bar_path_parity.py`; the bound is not
loosened.

## Current Records (Whole-Row Lift Body, #13037)

The sections above were measured on earlier bodies and stay for history. The
bundle is now built on the AN-1 lift body regenerated on the whole de Leva
Table 4 rows (#13102) with the default 10 deg tibial torsion (#12814), and
every number below is read from the committed bundle and
`bar_path_parity.json` (source commit `858f4c03`).

IK residuals (maximum over all 151 frames, mm): ankle_l 3.78, ankle_r 3.11,
knee_l 3.44, knee_r 10.96, hip_l 8.30, hip_r 5.90, shoulder_l 4.11,
shoulder_r 4.58, bar 2.65. Grip closure at most 8.9e-5 m, lowest foot sphere
0.69 mm into the ground, bundle 231,068 bytes. The bounds in
`tests/unit/lifting/test_pull_bundles.py` are regression snapshots of the
committed bundle, so they are re-derived with their original rounding (up to
0.1 mm; closure to two significant figures): both ankles, knee_l, knee_r,
both shoulders and the closure (7.7e-5 to 9.0e-5 m) (#13041, issuecomment-6243170509).

The bundle now records 16 RK4 substeps. At the earlier eight-substep
hand-held policy its stiffest foot mode (3.4e4 1/s) gave |z| = 4.24, outside
the RK4 real stability limit 2.785 (amplification 6.5), and the foot
coordinates drifted 9.9e-5 rad between engines. The bundle builder now
raises the substep count from the bundle's own `rk4_stability` record until
|z| is inside the limit (`stiffness.stable_substeps`); at 16 substeps |z| is
2.49 (amplification 0.63). Open-loop MuJoCo replays at 16, 32 and 64
substeps differ by at most 6.3e-9 rad.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 3.8e-15                 | 1.7e-13                          | 8.2e-15                               | 6.8e-16                        |
| Pinocchio | 1.6e-14                 | 9.1e-14                          | 5.4e-15                               | 1.4e-15                        |
| OpenSim   | 1.1e-15                 | 3.1e-14                          | 4.6e-15                               | 7.0e-16                        |

The coordinate gate now passes, and its strict xfail in
`tests/unit/lifting/test_pull_bar_path_parity.py` is removed (source commit
`33abf79c`).

## Barbell Bundle and OpenSim IK/ID (OSL-3, #13041)

`deadlift_barbell_bundle.npz` (SHA-256 `a8483a3771b844f7ec652e839fd64337d6f806b82d26f0fcc7ba8bf3a19bac40`)
packages the conventional deadlift with the LIFT-4 barbell body and load,
built by `scripts/generate_barbell_squat_evidence.py --lift deadlift --part bundle`
from `deadlift_bundle.npz`.

### Bar-Hold Metrics (Provenance)

The barbell hold re-tracks the standing pose with the bar carried by the hands:
- **Bar deviation:** 35.95 mm (0.03595 m) against the reference trajectory.
- **Left hand force:** peak magnitude 614.05 N (mean [34.07, 1.11, 456.98] N),
  peak torque magnitude 4.71 Nm.
- **Right hand force:** peak magnitude 618.50 N (mean [42.98, 1.61, 459.50] N),
  peak torque magnitude 4.71 Nm.

### OpenSim Inverse Kinematics and Inverse Dynamics Receipt

The OpenSim IK/ID receipt (`docs/development/lifting/evidence/opensim_ik_id/receipts.json`)
is measured with real OpenSim 4.6 and the RajagopalLaiUhlrich2023 model over 41 frames
(stride 5):
- **Marker kinematics:** RMS error 0.0112 m (1.12 cm, within the 0.0200 m threshold),
  maximum marker error 0.0390 m (within the 0.0400 m threshold). The kinematics
  satisfies the best-practice marker gate (`within_best_practice = True`).
- **Segment angles:** sagittal segment-angle RMS error 0.0158 rad (0.91°, well within
  the 0.0349 rad / 2.0° threshold).
- **Arm reach:** Left hand reach miss 7.11 µm (7.1098e-6 m), right hand reach miss
  6.91 µm (6.9120e-6 m), well within the 1.0 mm reach bound.
- **Coordinate range violations:** `pro_sup_l` (40 frames, min −0.00065 rad, range
  [0.0, 2.09] rad) and `pro_sup_r` (40 frames, min −0.00057 rad, range [0.0, 2.09] rad).
- **Hicks residual gate:** peak external force 2218.39 N; force threshold 110.92 N
  (5% of peak external force); moment threshold 18.28 Nm (1% of COM height × peak external force).
  The residual gate fails for both forces and moments:
  - Pelvis force: peak $t_x$ 166.99 N (RMS 26.76 N), peak $t_y$ 288.23 N (RMS 45.19 N),
    peak $t_z$ 5.51 N (RMS 1.25 N); max force 288.23 N vs 110.92 N threshold (`forces_within = False`).
  - Pelvis moment: peak tilt 147.88 Nm (RMS 26.54 Nm), peak list 9.88 Nm (RMS 2.80 Nm),
    peak rotation 2.77 Nm (RMS 1.84 Nm); max moment 147.88 Nm vs 18.28 Nm threshold (`moments_within = False`).
- The residual gate failures are strictly xfailed in `tests/unit/lifting/test_opensim_lift_ik_id.py`
  with measured numbers; gates are not loosened.

## Literature Comparison (Evidence, Not Tuning Targets)

Indicative ranges, quoted from the cited sources as recalled for this record
(moderate confidence; check the papers before quantitative use):

- **Bar start height.** Standard 20 kg plates (0.45 m diameter) put the bar
  centre about 0.225 m above the floor. The model's bar starts 0.456 m above
  the world ground, because the model's arms hang the bar about 0.14 m
  shorter than the optimizer's arm: this bundle is closer to a block pull
  from knee height than a floor pull.
- **Bar path.** In conventional deadlifts the bar moves close to vertically,
  within a few centimetres forward/back (Escamilla et al., 2000,
  _Med Sci Sports Exerc_ 32(7):1265-1275). The optimizer path moves 0.06 m
  forward overall; inside the window the model bar moves 4.3 mm forward.
- **Bar velocity.** Mean concentric bar velocities fall from about 0.6-0.8 m/s
  at light loads to about 0.15-0.25 m/s near one repetition maximum
  (load-velocity studies summarised by Garhammer, 1993,
  _J Strength Cond Res_ 7(2):76-89, for power output; deadlift-specific
  load-velocity profiles report similar ranges). A 100 kg bar for an 80 kg
  lifter is submaximal for a trained lifter; the window's 0.33 m/s mean is
  inside this range, but the optimizer's velocity peaks at 0.12 s rather
  than near the knee, which a minimum-effort solve need not reproduce.
- **Phase timing.** Heavy deadlifts take about 1.5-3 s from lift-off to
  lockout. The 2.0 s solve with knee pass at 0.53 s is inside that range.

## Limitations

- Hand-held bar: in `deadlift_bundle.npz` the bar is the club's `Grip` frame
  with the left hand moved to the grip width; `deadlift_barbell_bundle.npz`
  adds the LIFT-4 barbell body and load.
- Bar offset scaled to the model's arm hang (0.799), so the setup is high
  (see above). The optimizer's own bar model is unchanged.
- Golf full-body spec (asymmetric arm chain) as for the back squat.
- OpenSim inverse dynamics fails the Hicks residual gate (forces 288.23 N vs
  110.92 N, moments 147.88 Nm vs 18.28 Nm). Results are software receipts,
  not physical acceptance.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift deadlift
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift deadlift
python3 scripts/generate_barbell_squat_evidence.py --lift deadlift --part bundle --source-commit <sha>
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
