# Clean-and-Jerk Same-Input Bundle: First Pull (LIFT-5, #11745)

`clean_and_jerk_bundle.npz` packages the **first pull of the clean** as a
same-input `InputBundle`, built by the back-squat pipeline (see
`../back_squat/README.md`) with the recipe in
`src/shared/python/lifting/lift_recipes.py`. The rack, the jerk and every
overhead phase are not in this bundle.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer solves the `clean`
   exercise: 80 kg, 1.78 m lifter, 100 kg bar, 1.2 s, 151 samples. Its own
   clean bar model keeps the bar at the shoulders (front rack) for the whole
   lift, which is wrong during the pull, so the recipe uses the optimizer's
   `deadlift` bar model instead: one arm length (0.689 m) below the shoulder.
2. **Hand-held bar.** `lift_hands_to_full_body` moves the left hand to the
   standard's clean-and-jerk grip width (0.6 m, parity standard 2.1.0) and
   solves the arms with the bar long axis medio-lateral. The model's arms hang
   the bar at most 0.502 m below the shoulder midpoint at this wider grip, so
   the optimizer's shoulder-to-bar offset is scaled by 0.729.
3. **Window.** A 0.2 s MuJoCo window centred on the knee pass (0.240 s)
   starts at 0.136 s: 200 steps at 1 ms.

## Measured Numbers

| Quantity                                          | Value              |
| ------------------------------------------------- | ------------------ |
| Optimizer bar height above the ankle, setup → end | 0.201 → 0.658 m    |
| Optimizer peak vertical bar velocity              | 1.29 m/s at 0.06 s |
| Model bar height (world), setup → end             | 0.505 → 0.957 m    |
| Window bar rise, mean vertical velocity           | 0.150 m, 0.75 m/s  |
| Window bar drift forward / lateral                | 18.4 mm / −6.0 mm  |
| Bundle size                                       | 214,502 bytes      |

IK residuals (maximum over all 151 frames, metres, rounded up to 0.1 mm;
asserted in `tests/unit/lifting/test_pull_bundles.py`): ankle_l 0.0078,
ankle_r 0.0037, knee_l 0.0227, knee_r 0.0101, hip_l 0.0094, hip_r 0.0139,
shoulder_l 0.0087, shoulder_r 0.0078, bar 0.0045. Grip closure at most
6.5e-7 m, lowest foot sphere 0.69 mm into the ground.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` (measured at commit `37aaf459`), back-squat protocol
and fixed tolerances:

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 1.2e-12                 | 3.4e-10                          | 1.3e-12                               | 3.9e-15                        |
| Pinocchio | 4.5e-13                 | 1.2e-10                          | 3.0e-13                               | 2.8e-14                        |
| OpenSim   | 7.2e-13                 | 2.0e-10                          | 6.1e-14                               | 3.0e-15                        |

Every engine meets both L2 bounds (1e-3 m frames, 1e-6 rad coordinates) over
the full open-loop window and the re-anchored segments; no foot-contact
event occurs in this window.

## Current Records (Whole-Row Lift Body, #13037)

The sections above were measured on earlier bodies and stay for history. The
bundle is now built on the AN-1 lift body regenerated on the whole de Leva
Table 4 rows (#13102) with the default 10 deg tibial torsion (#12814), and
every number below is read from the committed bundle and
`bar_path_parity.json` (source commit `858f4c03`).

IK residuals (maximum over all 151 frames, mm): ankle_l 3.78, ankle_r 2.83,
knee_l 3.43, knee_r 10.98, hip_l 8.32, hip_r 5.92, shoulder_l 4.10,
shoulder_r 3.06, bar 1.15. Grip closure at most 3.3e-5 m, lowest foot sphere
0.69 mm into the ground, bundle 231,160 bytes. Against the unchanged bounds
in `tests/unit/lifting/test_pull_bundles.py`, both ankles, knee_l, knee_r,
both shoulders and the closure (2.8e-5 m) now fail; no bound is widened.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 1.1e-14                 | 5.1e-13                          | 1.2e-14                               | 1.3e-15                        |
| Pinocchio | 2.0e-14                 | 4.1e-13                          | 1.2e-14                               | 2.6e-15                        |
| OpenSim   | 1.7e-15                 | 8.4e-14                          | 5.0e-15                               | 8.1e-16                        |

Every engine meets both L2 bounds. RK4 stability: |z| = 2.24, amplification
0.44.

## Literature Comparison (Evidence, Not Tuning Targets)

Indicative ranges, quoted from the cited sources as recalled for this record
(moderate confidence; check the papers before quantitative use):

- **First-pull timing and velocity.** In competition cleans the first pull
  (lift-off to knee height) takes about 0.4-0.6 s, and the bar reaches about
  0.9-1.2 m/s near the knee before the second pull accelerates it to a peak
  of about 1.6-2.0 m/s (Garhammer, 1993, _J Strength Cond Res_ 7(2):76-89;
  Kipp, Harris and Sabick, 2011, _J Strength Cond Res_ 25(5):1229-1234).
  The optimizer reaches the knee at 0.24 s with its peak velocity of
  1.29 m/s at 0.06 s, then slows: its first pull is about twice as fast as
  the literature range and front-loaded, with no second-pull acceleration.
  The window's 0.75 m/s mean is inside the first-pull range.
- **Bar path.** Weightlifters bring the bar back towards the body during
  the first pull (a few centimetres; Garhammer, 1993). The model bar moves
  18 mm forward through the window instead.
- **Bar start height.** As for the deadlift, the shorter model arm hang lifts
  the setup bar to 0.505 m above the ground against about 0.225 m with
  standard plates.

## Limitations

- First pull only. The optimizer's clean and jerk configurations cannot be
  used for the later phases: the clean keeps the bar on the shoulders, and
  the `jerk` solve did not converge (measured 1.5 s solve, dip trunk 48.7
  degrees). The rack, jerk and overhead phases need a Tools configuration
  with a pull-to-rack bar model.
- Hand-held bar on the club `Grip` frame; no barbell body or load (LIFT-4,
  #11744). Golf full-body spec as for the back squat.
- Bar offset scaled to the model's arm hang (0.729).

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift clean_and_jerk
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift clean_and_jerk
```
