# Clean-and-Jerk Same-Input Bundle: The Whole Lift (LIFT-5, #11745)

`clean_and_jerk_bundle.npz` packages the **whole clean and jerk** as a
same-input `InputBundle`: floor setup, first and second pull, turnover,
front-squat catch, recovery to a standing front rack, jerk dip and drive,
power-jerk catch overhead and recovery to a standing lockout. It is built by
the back-squat pipeline (see `../back_squat/README.md`) with the recipe in
`src/shared/python/lifting/lift_recipes.py`.

The bundle records the whole-lift solution and its whole-lift IK. Its 0.2 s
MuJoCo window, the part replayed in the engines, is centred on the knee pass
of the first pull. **The window does not cover the rack, the jerk or the
overhead phases**; those are in the recorded solution and IK only.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer solves the
   `clean_and_jerk` exercise (`make_clean_and_jerk_config`, Tools #5661):
   80 kg, 1.78 m lifter, 100 kg bar, 4.0 s, 151 samples, 12 waypoints. A
   `ViaSchedule` pins one via pose at each of the parity standard's eight
   phase starts (setup 0, first pull 0.08, second pull 0.2, rack catch 0.3,
   rack recovery 0.4, jerk dip-drive 0.55, overhead catch 0.7, overhead
   recovery 0.85) and at three events inside them (full extension 0.25, dip
   bottom 0.6, drive extension 0.65). The rack catch is the standard's catch
   pose (knee −97.2°, trunk 18.31°) with the shank solved so the centre of
   mass is over the inner base of support. The standard's symmetric stance
   makes the jerk a power jerk; it is caught at the jerk dip depth (knee
   −52°) with the lockout trunk lean (−0.05°).
2. **Phase-correct bar.** `src/shared/python/lifting/phase_bar_paths.py`
   evaluates the Tools `clean_and_jerk_bar_path`: the bar hangs one arm
   length below the shoulder through the pulls, sits at the shoulder (front
   rack) from the rack catch to the drive extension and is one arm length
   above it from the overhead catch on, with smoothstep blends across the
   turnover (0.25→0.3) and the drive (0.65→0.7).
3. **Hand-held bar.** `lift_hands_to_full_body` puts the hands at the
   standard's clean-and-jerk grip width (0.6 m) with the bar long axis
   medio-lateral. The arms hang the bar at most 0.562 m below the shoulder
   midpoint, so the optimizer's shoulder-to-bar offset is scaled by 0.816.
4. **Window.** A 0.2 s window centred on the knee pass starts at 0.693 s:
   200 steps at 1 ms. `bundle_stability` selects 16 RK4 substeps
   (|z| = 1.88, amplification 0.30).

## Before and After

Before is the previous bundle (the first pull of the `clean` exercise,
1.2 s, `deadlift` bar model, the whole-row lift body, source `858f4c03`).
After is this bundle (source `5aac442d`, Tools pin `0c585b39`).

| Quantity                                   | Before                                     | After                               |
| ------------------------------------------ | ------------------------------------------ | ----------------------------------- |
| Lift covered by the solution               | first pull to front rack                   | floor to overhead lockout           |
| Planar bar at the setup                    | 0.201 m (hand-held model)                  | 0.201 m (hang)                      |
| Front-squat catch                          | none (knee never past 90.5° setup flexion) | knee −97.2°, trunk 18.31°           |
| Bar overhead                               | never                                      | from 0.7 (1.978 m), lockout 2.068 m |
| Model bar height (world), setup → end      | 0.432 → 0.884 m                            | 0.432 → 2.044 m                     |
| Peak pull bar velocity (planar)            | 1.29 m/s at 0.06 s                         | 1.72 m/s at 0.85 s                  |
| Window start, rise, mean vertical velocity | 0.136 s, 0.155 m, 0.78 m/s                 | 0.693 s, 0.250 m, 1.25 m/s          |
| Window bar drift forward / lateral         | −30.5 / 0.3 mm                             | −42.3 / −1.4 mm                     |
| RK4 substeps (bundle_stability)            | 8 (max abs z 2.24)                         | 16 (max abs z 1.88)                 |
| Bundle size                                | 231,153 bytes                              | 232,229 bytes                       |

In the new planar solution the shoulder rises monotonically from 0.890 to
1.376 m through the pulls (0.08–0.25), drops into the catch, dips from 1.380
to 1.291 m in the jerk and drives back to 1.376 m. The setup hold sags the
bar 11 mm before it leaves the floor.

## IK Residuals

Whole-lift maxima over all 151 frames (mm). The thresholds in
`tests/unit/lifting/test_pull_bundles.py` are these measured whole-lift
maxima rounded up to 0.1 mm, the convention of every pull bundle; they are a
regression snapshot of this bundle, not a physical gate. The physical gate
is the deadlift bound, which this bundle fails (strict xfail
`test_clean_and_jerk_whole_lift_ik_meets_the_deadlift_bounds`, below).

| Residual                         | Before       | After       |
| -------------------------------- | ------------ | ----------- |
| ankle_l / ankle_r                | 3.78 / 2.83  | 71.6 / 74.4 |
| knee_l / knee_r                  | 3.43 / 10.98 | 69.8 / 66.4 |
| hip_l / hip_r                    | 8.32 / 5.92  | 56.5 / 57.6 |
| shoulder_l / shoulder_r          | 4.10 / 3.06  | 27.3 / 29.7 |
| bar                              | 1.15         | 56.5        |
| grip closure                     | 0.033        | 0.380       |
| lowest foot sphere (into ground) | 0.69         | 2.55        |

Per phase (maximum over the phase's frames, mm):

| Phase (fraction)                | Frames | Worst legs / shoulders      | Bar  | Feet |
| ------------------------------- | ------ | --------------------------- | ---- | ---- |
| setup (0–0.08)                  | 12     | knee_r 11.1                 | 1.15 | 0.68 |
| first pull (0.08–0.2)           | 18     | knee_r 10.9                 | 0.04 | 0.69 |
| second pull, turnover (0.2–0.3) | 15     | legs 31–40                  | 39.1 | 1.78 |
| rack catch (0.3–0.4)            | 15     | legs 57–74                  | 56.5 | 2.55 |
| rack recovery (0.4–0.55)        | 23     | legs 49–58                  | 55.8 | 2.2  |
| jerk dip-drive (0.55–0.7)       | 22     | legs ≤ 6.7, shoulders 27–29 | 56.1 | 0.69 |
| overhead catch (0.7–0.85)       | 23     | legs 19–29                  | 3.9  | 1.62 |
| overhead recovery (0.85–1)      | 23     | hip_l 8.0                   | 0.78 | 0.87 |
| window (0.693–0.893 s)          | 8      | knee_r 8.1                  | 0.34 | 0.69 |

The pull, the window and the overhead recovery are deadlift class. The
full-body model cannot follow the front rack: the planar rack bar sits at
the shoulder joints, which the hands cannot reach at the 0.6 m grip (bar
39–57 mm off from the turnover to the drive), and the planar deep front squat
cannot be reached with the feet flat (legs 31–74 mm, feet up to 2.55 mm into
the ground). The overhead catch at the jerk dip depth leaves the legs 19–29
mm off.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` (measured at commit `5aac442d`, the window above),
back-squat protocol and fixed tolerances:

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 3.0e-15                 | 2.2e-12                          | 1.6e-14                               | 2.2e-15                        |
| Pinocchio | 4.3e-14                 | 1.4e-12                          | 1.0e-14                               | 2.5e-15                        |
| OpenSim   | 2.7e-15                 | 9.6e-13                          | 8.8e-15                               | 1.0e-15                        |

Every engine meets both L2 bounds (1e-3 m frames, 1e-6 rad coordinates) over
the full open-loop window and the re-anchored segments. Before (first-pull
bundle): Drake 1.1e-14 m / 5.1e-13 rad, Pinocchio 2.0e-14 m / 4.1e-13 rad,
OpenSim 1.7e-15 m / 8.4e-14 rad, MuJoCo bit exact. The parity covers the
window only, not the rack, jerk or overhead phases.

## Literature Comparison (Evidence, Not Tuning Targets)

Indicative ranges, quoted from the cited sources as recalled for this record
(moderate confidence; check the papers before quantitative use):

- **Pull.** In competition cleans the bar peaks at about 1.6–2.0 m/s in the
  second pull (Garhammer, 1993, _J Strength Cond Res_ 7(2):76-89; Kipp,
  Harris and Sabick, 2011, _J Strength Cond Res_ 25(5):1229-1234). The
  planar peak is 1.72 m/s at 0.85 s, in the second pull, inside that range.
- **Lift duration.** A competition clean and jerk takes several seconds with
  a pause in the rack; the 4.0 s duration and the standard's phase fractions
  are inputs, not fitted.
- **Jerk drive.** The bar leaves the shoulders at about 1.5–2 m/s in the
  jerk (Garhammer, 1993). The planar bar moves one arm length relative to
  the shoulder across the 0.65→0.7 blend, peaking at 4.64 m/s at 2.69 s:
  the blend is a bar-placement model, not jerk dynamics.

## Barbell Bundle and OpenSim IK/ID (OSL-3, #13041)

`clean_and_jerk_barbell_bundle.npz` (SHA-256 `0b5221ff7cc3b7128be609428a9681e7f0cc4398f38a7c8e0c35163d0beb45aa`)
packages the clean and jerk pull window with the LIFT-4 barbell body and load,
built by `scripts/generate_barbell_squat_evidence.py --lift clean_and_jerk --part bundle`
from `clean_and_jerk_bundle.npz`.

### Bar-Hold Metrics (Provenance)

The barbell hold re-tracks the clean standing pose with the bar carried by the hands:

- **Bar deviation:** 116.02 mm (0.11602 m) against the reference trajectory.
- **Left hand force:** peak magnitude 1364.72 N (mean [25.12, −0.93, 845.98] N),
  peak torque magnitude 7.38 Nm.
- **Right hand force:** peak magnitude 1387.04 N (mean [22.45, −1.11, 851.21] N),
  peak torque magnitude 7.38 Nm.

### OpenSim Inverse Kinematics and Inverse Dynamics Receipt

The OpenSim IK/ID receipt (`docs/development/lifting/evidence/opensim_ik_id/receipts.json`)
is measured with real OpenSim 4.6 and the RajagopalLaiUhlrich2023 model over 41 frames
(stride 5):

- **Marker kinematics:** RMS error 0.0198 m (1.98 cm, within 0.0200 m threshold),
  maximum marker error 0.1612 m (exceeds 0.0400 m threshold due to the bar marker).
  The kinematics fails the best-practice marker gate (`within_best_practice = False`).
- **Segment angles:** sagittal segment-angle RMS error 0.0220 rad (1.26°, well within
  the 0.0349 rad / 2.0° threshold).
- **Arm reach:** Left hand reach miss 6.43 µm (6.4341e-6 m), right hand reach miss
  6.02 µm (6.0187e-6 m), well within the 1.0 mm reach bound.
- **Coordinate range violations:** `pro_sup_l` (40 frames, min −0.00227 rad, range
  [0.0, 2.09] rad) and `pro_sup_r` (40 frames, min −0.00209 rad, range [0.0, 2.09] rad).
- **Hicks residual gate:** peak external force 4462.06 N; force threshold 223.10 N
  (5% of peak external force); moment threshold 38.77 Nm (1% of COM height × peak external force).
  The residual gate fails for both forces and moments:
  - Pelvis force: peak $t_x$ 458.52 N (RMS 92.66 N), peak $t_y$ 257.66 N (RMS 63.79 N),
    peak $t_z$ 17.99 N (RMS 4.94 N); max force 458.52 N vs 223.10 N threshold (`forces_within = False`).
  - Pelvis moment: peak tilt 259.03 Nm (RMS 80.19 Nm), peak list 7.92 Nm (RMS 4.89 Nm),
    peak rotation 21.71 Nm (RMS 6.01 Nm); max moment 259.03 Nm vs 38.77 Nm threshold (`moments_within = False`).
- The residual gate failures are strictly xfailed in `tests/unit/lifting/test_opensim_lift_ik_id.py`
  with measured numbers; gates are not loosened.

## Limitations

- The engine window covers the knee pass of the first pull only. The rack,
  the jerk and the overhead phases are in the recorded solution and IK, not
  in the replayed window.
- Rack and catch IK fail the deadlift bounds (strict xfail above). A forward
  front-rack bar offset in the Tools planar model would let the hands reach
  the rack; it is not attempted here.
- In `clean_and_jerk_bundle.npz` the bar is held on the club `Grip` frame;
  `clean_and_jerk_barbell_bundle.npz` adds the LIFT-4 barbell body and load
  and is registered in `SAME_INPUT_LIFT_BUNDLES`.
- OpenSim inverse dynamics fails the Hicks residual gate (forces 458.52 N vs
  223.10 N, moments 259.03 Nm vs 38.77 Nm). Results are software receipts,
  not physical acceptance.
- The optimizer omits the Coriolis terms (max |qd| 6.26 rad/s, above its
  2 rad/s slow-movement assumption), so its joint torques are indicative.
- Bar offset scaled to the model's arm hang (0.816).

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift clean_and_jerk
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift clean_and_jerk
python3 scripts/generate_barbell_squat_evidence.py --lift clean_and_jerk --part bundle --source-commit <sha>
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
