# Head Gaze Stabilisation Reference

Modelling reference for OSV-3 (#11729, epic #11726). Code:
`src/shared/python/motion_matching/gaze.py` (pure functions),
`src/shared/python/motion_matching/pipeline/gaze_residual.py` (matching
residual and receipt block), `src/shared/python/model_appearance/ball.py`
(the single ball-at-address function, shared with GCV-13 #11719). A LaTeX copy
of the equations is in `head_gaze_reference.tex`.

## Scope and Status

- Implemented: eye point, gaze error, gaze schedule, head stability metrics,
  neck inverse kinematics inside the range of motion, and an optional soft gaze
  residual in the matching inverse kinematics (`--gaze-weight`, default 0).
  The gaze-regularised reports and clips use `REPORTING_GAZE_WEIGHT = 0.1`
  (`motion_matching/gaze_sweep.py`), selected from a two-capture sweep (see
  Gaze Weight Selection).
- Not implemented here: neck PD or torque tracking of the schedule in the
  forward-dynamics engines, the MyoSuite neck actuator, OpenSim `joint_Head`
  coordinates, and the GCV-12 visual head channel for `golf_humanoid.osim`.
  Iron, MyoSuite, OpenSim, Drake and Pinocchio receipts are not produced yet.
- The gaze-regularised head is a soft prior. It is never measured data.
  Default receipts stay marker-faithful.

## Frames and Units

SI units, Z up. The head frame (`Head`, a frame site of the full-body
document) has its origin at the neck joint, +x anterior, +y left, +z up. The
neck joint chain is `Rx(NeckInputX) Ry(NeckInputY) Rz(NeckInputZ)` in the
torso (`Torso`) frame, so with +x anterior `NeckInputX` is lateral bending,
`NeckInputY` is flexion and extension and `NeckInputZ` is axial rotation
(checked on the capture-A driver trajectory: the torso x axis points
anteriorly). Neck ranges are +-45, +-60 and +-80 degrees
(`range_of_motion.py`).

## Equations

- Eye point: `e = R_head * o_eye + t_head`. The offset is
  `o_eye = (0.09, 0.0, 0.128)` m: 0.2429 m head length (de Leva male, the
  value in the documents) minus an assumed 0.115 m vertex-to-eye distance
  (adult-male survey order of magnitude, an assumption, not a fitted number),
  0.09 m anterior (the forehead marker is at x = 0.10 m).
- Gaze direction: `g = R_head * a`, where `a` is the head gaze axis.
- Gaze error: `theta_gaze = angle(g, b - e)` with `b` the ball at address.
- Ball at address (one function, `ball_position_at_address`): the face-centre
  contact point plus one ball radius (0.021335 m) along the unit face normal,
  then lowered to rest on the ground (`z = z_ground + r`).
- Gaze schedule: the target direction is `(b - e)/|b - e|` until
  `t_impact + t_hold` (0.03 s), then a great-circle blend to the target-line
  direction `x_t` over `T_rel` (0.35 s, configurable) with the minimum-jerk
  fraction `s(u) = 10u^3 - 15u^4 + 6u^5`, `u = (t - t_impact - t_hold)/T_rel`.
  The blend is C2 at both ends. Equivalent to the target point `e + d x_t` at
  eye height because `d` cancels in the direction. Antiparallel directions are
  rejected.
- Impact time: the lowest clubhead point within 0.05 s of the peak clubhead
  speed (`detect_impact_index`). Never a fixed time. The trajectory is smoothed
  at the reference cutoff first so marker jitter does not move the peak.
- Neck IK: bounded least squares `parent_R * R_xyz(q) * a = direction` inside
  the ROM with a small prior toward the previous or zero pose (the roll about
  the axis is free). Out-of-range requests are clamped and reported
  (`clamped`, `residual_deg`).
- Soft residual: the head axis target row `sqrt(w) (R_head a - d_k)` joins the
  per-frame axis targets of the matching inverse kinematics. The marker rows
  keep their weights. A marker-faithful pass plans the schedule once (ball,
  impact, eye positions); the same targets are reused by the consistency
  re-solve.

## Calibrated Gaze Axis

The head +x axis is not the line of sight of a real golfer: the eyes look below
the head's forward axis at address. On the capture-A driver the nominal +x axis
is 37 degrees from the line of sight to the ball at address, so the issue's
literal definition gave a 37 degree error floor and made the soft residual
fight the markers (trunk marker error 100 mm at weight 10 in a first attempt,
recorded as a failed experiment). The head gaze axis `a` is therefore
calibrated at address as the head-frame direction of the address line of sight
to the ball (the subject looks at the ball at address by definition).
`theta_gaze` is then the change of that gaze through impact. The receipt also
records the nominal +x axis error (`address_to_impact_nominal_axis`).

## Receipt Block

`receipt.json` carries `head_gaze` with: `gaze_weight`, `regularised`, the
plan (ball, impact index and time, target line, hold and release times, eye
offset, gaze axis), `address_to_impact` (eye translation range per axis in mm,
head yaw, pitch and roll range in degrees, `theta_gaze` max and RMS), the
nominal-axis error, a neck IK schedule summary and the neck coordinate ranges.

## Results

Capture-A driver, MuJoCo native model, anthropometric driver document,
inverse kinematics reference (`q_ref`), address to impact. Reproduce with
`python -m src.shared.python.motion_matching.pipeline.cli --spec
docs/development/full_body_models/full_body_spec_anthro_driver.json
--skip-hip-calibration --static-seeds --capture driver --gaze-weight W --out
RUN`.

| Gaze weight | Marker RMS (mm) | theta_gaze RMS (deg) | theta_gaze max (deg) | Eye range x / y / z (mm) | Head yaw / pitch / roll range (deg) |
|---|---|---|---|---|---|
| 0 | 28.0 | 21.26 | 39.10 | 56 / 123 / 31 | 47 / 9 / 72 |
| 0.3 | 30.5 | 7.63 | 20.46 | 60 / 128 / 23 | 32 / 8 / 31 |
| 1 | 32.7 | 1.78 | 6.50 | 57 / 105 / 26 | 27 / 4 / 19 |
| 3 | 33.4 | 0.77 | 1.86 | 57 / 109 / 26 | 29 / 3 / 18 |
| 10 | 33.8 | 0.80 | 2.14 | 55 / 99 / 23 | 29 / 3 / 17 |
| 30 | 33.9 | 0.96 | 2.29 | 57 / 93 / 24 | 29 / 3 / 17 |
| 100 | 33.8 | 0.97 | 2.19 | 53 / 96 / 29 | 28 / 3 / 17 |

This first exploratory sweep picked 10 for reporting by eye (the numbers are
flat above 1). OSV-3b replaces that choice with the rule-based selection below. The
marker error rises from 28.0 to 33.8 mm: the soft gaze costs marker fidelity,
which is why the default stays 0. Eye translation barely changes (the
stillness comes from the torso solution; the head is not pinned). The
neck solution stays inside the ROM (bounded IK coordinates). Method: the
marker-faithful run supplies the calibration (scaled document and marker
attachments); each weight re-solves the trajectory inverse kinematics and the
consistency step on it (`evidence/head_gaze/sweep_driver.json`). The w=0 and w=10
head metrics at address-to-impact use the same plan definitions. Open: iron,
MyoSuite, OpenSim, Drake and Pinocchio receipts; forward-dynamics neck
tracking.

## Gaze Weight Selection

OSV-3b selects the reporting weight by a rule, not by eye
(`motion_matching/gaze_sweep.py`):

- Feasible: marker RMS at most 10 % above the weight-0 run of the same capture
  and OSV-10 face-fit RMS at most 5 degrees.
- Knee: the point of the feasible Pareto front (marker RMS, `theta_gaze` RMS)
  farthest below the chord joining the front's two ends, in normalised
  coordinates.
- Default: the smallest per-capture knee among the weights that are feasible
  in every capture. Postcondition: the result is feasible in every capture.

Method: `scripts.sweep_gaze_weight` runs the shared pipeline up to the
trajectory inverse kinematics with the club-face fixture flags (face weight 3,
the fixture run's segment scales: driver femur/tibia 1.0/1.0, iron 0.97/1.0)
and reports on the reference trajectory `q_ref`, address to impact. These
flags differ from the exploratory table above, so its absolute numbers differ.
The table lists marker RMS, face-fit RMS, `theta_gaze` RMS and max, eye
translation range, head yaw/pitch/roll range and the feasibility flag.

| Capture | w | Marker RMS (mm) | Face RMS (deg) | theta_gaze RMS / max (deg) | Eye range x / y / z (mm) | Yaw / pitch / roll (deg) | Feasible |
|---|---|---|---|---|---|---|---|
| capture-A driver | 0 | 32.6 | 0.76 | 29.81 / 79.82 | 93 / 242 / 100 | 105 / 50 / 91 | yes |
| capture-A driver | 0.1 | 33.2 | 0.82 | 8.54 / 20.11 | 44 / 104 / 24 | 23 / 9 / 37 | yes |
| capture-A driver | 0.2 | 34.7 | 0.75 | 6.45 / 15.77 | 48 / 104 / 23 | 22 / 8 / 30 | yes |
| capture-A driver | 0.3 | 35.4 | 0.84 | 5.45 / 14.13 | 48 / 107 / 31 | 24 / 7 / 25 | yes |
| capture-A driver | 0.5 | 35.6 | 0.86 | 4.33 / 12.68 | 48 / 112 / 25 | 24 / 6 / 22 | yes |
| capture-A driver | 1 | 37.2 | 0.76 | 2.65 / 9.40 | 50 / 116 / 26 | 27 / 5 / 19 | no |
| capture-A driver | 2 | 37.7 | 0.91 | 1.40 / 4.79 | 49 / 107 / 26 | 27 / 4 / 18 | no |
| capture-A driver | 3 | 37.6 | 0.82 | 1.24 / 4.70 | 46 / 127 / 25 | 27 / 4 / 18 | no |
| capture-A driver | 5 | 38.4 | 0.87 | 0.90 / 3.39 | 49 / 126 / 27 | 28 / 4 / 18 | no |
| capture-B iron | 0 | 30.8 | 0.58 | 18.55 / 32.63 | 60 / 125 / 38 | 45 / 8 / 66 | yes |
| capture-B iron | 0.1 | 32.1 | 0.66 | 6.12 / 11.09 | 54 / 82 / 37 | 19 / 9 / 16 | yes |
| capture-B iron | 0.2 | 33.3 | 0.64 | 4.10 / 8.02 | 53 / 76 / 36 | 24 / 7 / 11 | yes |
| capture-B iron | 0.3 | 33.9 | 0.59 | 3.05 / 6.23 | 51 / 77 / 36 | 28 / 5 / 9 | no |
| capture-B iron | 0.5 | 34.8 | 0.62 | 1.96 / 4.12 | 51 / 78 / 36 | 30 / 4 / 7 | no |
| capture-B iron | 1 | 37.0 | 0.76 | 0.94 / 1.95 | 55 / 81 / 36 | 32 / 2 / 6 | no |
| capture-B iron | 2 | 37.4 | 0.65 | 0.50 / 0.95 | 50 / 70 / 36 | 30 / 2 / 4 | no |

Result: the knee is 0.1 in both captures and the common feasible set is
{0, 0.1, 0.2}, so `REPORTING_GAZE_WEIGHT = 0.1`. At 0.1 the address-to-impact
gaze error RMS falls from 29.8 to 8.5 degrees (driver) and from 18.6 to 6.1
degrees (iron), for a marker RMS cost of 0.6 and 1.3 mm. The marker-faithful
head swings widely (driver yaw range 105 degrees, eye y range 242 mm); the
smallest weight already removes most of that motion. Iron w=0.3 misses the
marker limit by 0.01 mm (33.89 against 33.88 mm); the rule is applied as
written, with no tolerance widened. The face fit stays below 1 degree at every
weight. Qualified receipts keep `--gaze-weight 0`. A regression test
(`tests/scripts/test_head_gaze_sweep_scripts.py`) recomputes the selection
from the committed evidence. Evidence:
`evidence/head_gaze/gaze_weight_sweep.json` and `gaze_weight_sweep.png`.
Reproduce (one run per capture and weight, then the summary):

```
python3 -m scripts.sweep_gaze_weight --capture driver --weight W --scales 1.0 1.0 --out SWEEP/driver_W
python3 -m scripts.sweep_gaze_weight --capture iron --weight W --scales 0.97 1.0 --out SWEEP/iron_W
python3 -m scripts.summarize_gaze_sweep SWEEP docs/development/full_body_models/evidence/head_gaze/gaze_weight_sweep.json --plot docs/development/full_body_models/evidence/head_gaze/gaze_weight_sweep.png
```

Limits of the selection: two captures, MuJoCo matching inverse kinematics
only, and a rule whose 10 % marker tolerance and 5 degree face cap are policy
choices, not physiological limits.

Published tour head-motion ranges are not tabulated here yet. They must be
cited and recorded as evidence, not tuned targets; this is open.

## Limitations

- The face normal is the clubhead impact path direction expressed in the
  clubhead frame (the model's clubhead frame carries an arbitrary roll about
  the shaft), a square-face assumption. The face centre is taken at the
  clubhead frame origin, an error of a few centimetres in the ball position.
- The target line is the horizontal face normal at address.
- The eye offset is an assumed anthropometric midpoint.
- The gaze-regularised result raises the marker error (see the table); the
  default stays marker-faithful.
- The neck prior is minimal-motion; there is no head-neck dynamics.
