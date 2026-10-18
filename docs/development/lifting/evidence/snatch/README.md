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

| Quantity                                       | Before (Single Via)     | After (Timed Schedule)                    |
| ---------------------------------------------- | ----------------------- | ----------------------------------------- |
| Shoulder height, setup → 0.93 s                | 0.876 → 0.653 m (falls) | 0.876 → 1.373 m                           |
| Shoulder over the pull (0.1-0.45)              | falls                   | rises 0.390 m monotonically, peak 1.376 m |
| Optimizer bar height at the setup              | 1.565 m (overhead)      | 0.187 m (hanging)                         |
| Catch trunk lean (standard 18.70°)             | 59.3°                   | 18.70° (tolerance 3.87°)                  |
| Optimizer peak vertical bar velocity, pull     | n/a                     | 1.21 m/s at 0.653 s                       |
| Model bar height (world), setup → end          | n/a                     | 0.462 → 1.884 m                           |
| Window bar rise, mean vertical velocity        | n/a                     | 0.123 m, 0.61 m/s                         |
| Window bar drift forward / lateral             | n/a                     | −13.7 mm / −0.3 mm                        |
| Bundle size                                    | n/a                     | 231,194 bytes                             |

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

| Part of the Lift                      | Worst Residuals                          | Feet Into Ground |
| ------------------------------------- | ---------------------------------------- | ---------------- |
| Pull (0-0.45)                         | knee_r 11.2 mm, bar 0.23 mm              | 0.69 mm          |
| Window (0.467-0.667 s)                | knee_r 8.1 mm, bar 0.06 mm               | 0.69 mm          |
| Turnover (0.50)                       | bar 40.1 mm, shoulder_l 19.7 mm          | 0.69 mm          |
| Catch and hold (0.53-0.77)            | legs 43-50 mm                            | up to 2.07 mm    |

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

## Limitations

- **Pull window only.** The turnover and catch frames of the full-body IK are
  not credible (see above). Fixing this needs a full-body catch that the
  model's ankle and arm ranges can reach.
- **No OpenSim IK/ID.** The bundle is not registered in
  `SAME_INPUT_LIFT_BUNDLES`. The OpenSim IK/ID needs a LIFT-4 barbell record,
  and this bundle has none; the deadlift and clean bundles are unregistered
  for the same reason. The measured result of `opensim_lift_ik_id` on this
  bundle is "IK/ID setup failed: bundle carries no barbell record".
- **Hand-held bar.** The bar is held on the club `Grip` frame with no barbell
  body or load (LIFT-4, #11744). The full-body spec is the golf spec, as for
  the back squat.
- **Scaled bar offset.** The bar offset is scaled to the model's arm hang
  (0.689).
- **Provenance pin.** The bundle was generated against the Tools PR #5653
  code checked out in `vendor/ud-tools`, while the recorded `tools_pin` is the
  committed gitlink. Regenerate it after the pin bump to the Tools merge
  commit.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift snatch
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift snatch
```
