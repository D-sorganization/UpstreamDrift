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
0.69 mm into the ground, bundle 230,973 bytes. Against the unchanged bounds
in `tests/unit/lifting/test_pull_bundles.py`, both ankles, knee_l, knee_r,
both shoulders and the closure (7.7e-5 m) now fail; no bound is widened.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 2.8e-8                  | 9.5e-5                           | 4.4e-5                                | 1.2e-8                         |
| Pinocchio | 2.9e-8                  | 9.9e-5                           | 4.5e-5                                | 1.3e-8                         |
| OpenSim   | 7.2e-9                  | 4.6e-5                           | 6.3e-6                                | 1.7e-9                         |

The worst segment is again 120 to 140 ms and the worst final coordinate
`subtalar_angle_r` (`subtalar_angle_l` in OpenSim); the other nine segments
end within 8.2e-13 rad. The coordinate gate stays a strict xfail. The
bundle's own `rk4_stability` record shows why the foot coordinates drift:
at the hand-held eight-substep policy its stiffest foot mode
(3.4e4 1/s) gives |z| = 4.24, outside the RK4 real stability limit 2.785
(amplification 6.5). The pull bundles do not use the loaded 64-substep
policy; changing that is a follow-up, not part of this regeneration.

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

- Hand-held bar: the bar is the club's `Grip` frame with the left hand moved
  to the grip width; there is no barbell body, load or plate contact in the
  dynamics (LIFT-4, #11744).
- Bar offset scaled to the model's arm hang (0.799), so the setup is high
  (see above). The optimizer's own bar model is unchanged.
- Golf full-body spec (asymmetric arm chain) as for the back squat.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift deadlift
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift deadlift
```
