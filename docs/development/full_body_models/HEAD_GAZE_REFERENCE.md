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
| ----------- | --------------- | -------------------- | -------------------- | ------------------------ | ----------------------------------- |
| 0           | 28.0            | 21.26                | 39.10                | 56 / 123 / 31            | 47 / 9 / 72                         |
| 0.3         | 30.5            | 7.63                 | 20.46                | 60 / 128 / 23            | 32 / 8 / 31                         |
| 1           | 32.7            | 1.78                 | 6.50                 | 57 / 105 / 26            | 27 / 4 / 19                         |
| 3           | 33.4            | 0.77                 | 1.86                 | 57 / 109 / 26            | 29 / 3 / 18                         |
| 10          | 33.8            | 0.80                 | 2.14                 | 55 / 99 / 23             | 29 / 3 / 17                         |
| 30          | 33.9            | 0.96                 | 2.29                 | 57 / 93 / 24             | 29 / 3 / 17                         |
| 100         | 33.8            | 0.97                 | 2.19                 | 53 / 96 / 29             | 28 / 3 / 17                         |

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

| Capture          | w   | Marker RMS (mm) | Face RMS (deg) | theta_gaze RMS / max (deg) | Eye range x / y / z (mm) | Yaw / pitch / roll (deg) | Feasible |
| ---------------- | --- | --------------- | -------------- | -------------------------- | ------------------------ | ------------------------ | -------- |
| capture-A driver | 0   | 32.6            | 0.76           | 20.23 / 37.06              | 48 / 113 / 25            | 49 / 11 / 75             | yes      |
| capture-A driver | 0.1 | 33.2            | 0.82           | 8.54 / 20.11               | 44 / 104 / 24            | 23 / 9 / 37              | yes      |
| capture-A driver | 0.2 | 34.7            | 0.75           | 6.45 / 15.77               | 48 / 104 / 23            | 22 / 8 / 30              | yes      |
| capture-A driver | 0.3 | 35.4            | 0.84           | 5.45 / 14.13               | 48 / 107 / 31            | 24 / 7 / 25              | yes      |
| capture-A driver | 0.5 | 35.6            | 0.86           | 4.33 / 12.68               | 48 / 112 / 25            | 24 / 6 / 22              | yes      |
| capture-A driver | 1   | 37.2            | 0.76           | 2.65 / 9.40                | 50 / 116 / 26            | 27 / 5 / 19              | no       |
| capture-A driver | 2   | 37.7            | 0.91           | 1.40 / 4.79                | 49 / 107 / 26            | 27 / 4 / 18              | no       |
| capture-A driver | 3   | 37.6            | 0.82           | 1.24 / 4.70                | 46 / 127 / 25            | 27 / 4 / 18              | no       |
| capture-A driver | 5   | 38.4            | 0.87           | 0.90 / 3.39                | 49 / 126 / 27            | 28 / 4 / 18              | no       |
| capture-A driver | 10  | 38.4            | 0.83           | 0.54 / 1.43                | 47 / 112 / 28            | 28 / 4 / 18              | no       |
| capture-B iron   | 0   | 30.8            | 0.58           | 18.55 / 32.63              | 60 / 125 / 38            | 45 / 8 / 66              | yes      |
| capture-B iron   | 0.1 | 32.1            | 0.66           | 6.12 / 11.09               | 54 / 82 / 37             | 19 / 9 / 16              | yes      |
| capture-B iron   | 0.2 | 33.3            | 0.64           | 4.10 / 8.02                | 53 / 76 / 36             | 24 / 7 / 11              | yes      |
| capture-B iron   | 0.3 | 33.9            | 0.59           | 3.05 / 6.23                | 51 / 77 / 36             | 28 / 5 / 9               | no       |
| capture-B iron   | 0.5 | 34.8            | 0.62           | 1.96 / 4.12                | 51 / 78 / 36             | 30 / 4 / 7               | no       |
| capture-B iron   | 1   | 37.0            | 0.76           | 0.94 / 1.95                | 55 / 81 / 36             | 32 / 2 / 6               | no       |
| capture-B iron   | 2   | 37.4            | 0.65           | 0.50 / 0.95                | 50 / 70 / 36             | 30 / 2 / 4               | no       |
| capture-B iron   | 3   | 37.1            | 0.65           | 0.48 / 0.98                | 50 / 72 / 36             | 32 / 2 / 4               | no       |
| capture-B iron   | 5   | 37.7            | 0.83           | 0.56 / 1.10                | 53 / 75 / 35             | 33 / 2 / 4               | no       |

Result: the knee is 0.1 in both captures and the common feasible set is
{0, 0.1, 0.2}, so `REPORTING_GAZE_WEIGHT = 0.1`. At 0.1 the address-to-impact
gaze error RMS falls from 20.2 to 8.5 degrees (driver) and from 18.6 to 6.1
degrees (iron), for a marker RMS cost of 0.6 and 1.3 mm. The marker-faithful
head rolls and turns widely (yaw / roll range 49 / 75 degrees driver, 45 / 66
degrees iron); the smallest weight already halves that motion. Iron w=0.3
misses the marker limit by 0.01 mm (33.893 against a 33.885 mm limit); the
rule is applied as
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

## Published Head-Motion Evidence

Recorded as evidence for comparison, not as tuned targets. No source found so
far reports eye-point translation in millimetres for tour players over a full
swing, so the eye-translation ranges reported here have no published
counterpart yet.

| Source                                                                                                         | Population and method                                            | Head result                                                                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zhang, Liu and Liu (2026), _Front. Sports Act. Living_ 8:1867187, doi:10.3389/fspor.2026.1867187               | 12 female professionals, Qualisys 250 Hz, driver, 5-iron, 7-iron | Head rotation at the top of the backswing −33.6 ± 9.1° (driver) and −25.4 ± 11.1° (7-iron); forward tilt at the top 67.0 ± 8.1° (driver) and 73.9 ± 7.9° (7-iron). The head-angle reference frame is not stated in the article. |
| Batbayar, Tserenchimed and Kim (2019), _Proc. Inst. Mech. Eng. H_ 233(5):554–561, doi:10.1177/0954411919838643 | Kinematic determinants of performance (as cited by Zhang et al.) | Elite golfers change head forward tilt and rotation significantly less than amateurs (qualitative; not yet read in full).                                                                                                       |

The published rotation is head rotation at the top, about 25–34°, while the
model's numbers are yaw ranges from address to impact. The two are not the same
quantity, so the comparison is indicative only. It does show that tour heads
turn substantially with the torso: they do not counter-rotate to keep the face
square to the ball. That matters for the gaze-schedule neck below, which
saturates the neck's yaw range when it is asked to hold the gaze axis on the
ball.

## Forward-Dynamics Neck Tracking

`pipeline/gaze_tracking.py` gives the forward-dynamics replay a gaze-schedule
neck. The replay's computed-torque tracker (`dynamics.controller`) is the neck
PD/torque controller; it tracks whatever neck reference `q_track` carries.

- `--fd-neck ik` (default): the neck follows the inverse-kinematics reference,
  so qualified receipts are unchanged.
- `--fd-neck gaze`: after the feasibility filters, each frame's three neck
  coordinates are re-solved so the head gaze axis points along the schedule
  (ball until impact + 0.03 s, then the minimum-jerk release over 0.35 s). The
  solve is a bounded least squares inside the neck range of motion and uses the
  model's own forward kinematics, so it assumes no joint convention. A weak
  prior (weight 0.01) toward the reference neck angles fixes the roll about the
  gaze axis. The eye point turns with the head, so the schedule is evaluated
  twice, the second time from the solved neck. The neck columns are then
  low-passed at the tracking cutoff and clipped back into the range. The ZMP
  reference keeps the IK neck. Head markers do not drive this neck; it is a
  modelled behaviour, never measured data.

The receipt block `dynamics.head_gaze` reports, for the tracked reference and
for the replay, the angle between the head gaze axis and the scheduled
direction in four windows (address to impact, hold, release, after release),
plus the address-to-impact head stability metrics. An empty window reports
`null`, never zero. Both captures, MuJoCo replay, canonical commands
(`CANONICAL_RUN.md` §2A) plus the flag shown:

| Capture | Neck        | Replay schedule error RMS (deg): address→impact / hold / release / after | Replay eye range x/y/z (mm) | Replay head yaw/pitch/roll range (deg) | Neck frames clamped | FD marker RMS (mm): whole / address→impact / head |
| ------- | ----------- | ------------------------------------------------------------------------ | --------------------------- | -------------------------------------- | ------------------- | ------------------------------------------------- |
| driver  | IK, w = 0   | 20.3 / 25.2 / 27.8 / 16.0                                                | 109 / 121 / 33              | 45 / 9 / 75                            | —                   | 77.0 / 32.3 / 87.0                                |
| driver  | IK, w = 0.1 | 10.6 / 10.8 / 10.6 / 12.7                                                | 110 / 133 / 32              | 31 / 10 / 42                           | —                   | 72.4 / 31.8 / 88.2                                |
| driver  | gaze, w = 0 | 1.8 / 5.0 / 15.4 / 9.3                                                   | 138 / 207 / 48              | 32 / 12 / 30                           | 213 of 654          | 94.7 / 49.6 / 170.4                               |
| 7-iron  | IK, w = 0   | 19.0 / 24.8 / 30.2 / 34.6                                                | 98 / 98 / 31                | 42 / 12 / 69                           | —                   | 71.2 / 33.3 / 86.5                                |
| 7-iron  | IK, w = 0.1 | 13.5 / 16.6 / 22.5 / 39.4                                                | 102 / 84 / 33               | 23 / 15 / 47                           | —                   | 88.5 / 32.2 / 107.8                               |
| 7-iron  | gaze, w = 0 | 1.2 / 3.7 / 6.3 / 15.1                                                   | 111 / 133 / 47              | 22 / 5 / 10                            | 114 of 654          | 88.0 / 41.6 / 144.8                               |

Findings:

- **The replay neck tracks its reference exactly.** In the driver gaze run the
  three neck coordinates follow `q_track` within 0.2° (RMS 0.01–0.02°). The
  tracked reference meets the schedule within 0.3° (driver) and 0.1° (7-iron)
  before impact.
- **What remains in the replay is torso error, not neck lag.** The neck
  reference is open loop: it is solved on the reference torso, so the replay's
  own torso error after impact passes straight into the head's world
  orientation (release 15.4° driver, 6.3° 7-iron). Closing that loop needs a
  gaze controller that re-solves the neck from the simulated torso at each
  control step. That is not built yet.
- **The neck range saturates.** Holding the gaze axis on the ball through the
  backswing drives `NeckInputZ` to its ±80° bounds on 213 (driver) and 114
  (7-iron) frames. The reference neck speed reaches 974°/s. Together with the
  published head rotation of about 25–34° at the top, this suggests that a
  strict eyes-on-ball schedule is stricter than tour golfers are.
- **Marker cost.** Head-marker RMS doubles (87 to 170 mm driver, 87 to 145 mm
  7-iron) because the markers no longer drive the neck. Other segments move by
  up to ±20 mm between configurations, in both directions; the w = 0.1 driver
  replay is better than w = 0 on the trunk and arms. The replay is that
  sensitive to small reference changes (see #12042), so these differences are
  not attributed to the neck.
- **The ZMP filter must run first.** The 7-iron `--zmp-filter` re-poses the
  body. A gaze neck applied before it left the tracked reference 13.3° off the
  schedule, with a solve residual of only 0.004°, so the neck step now runs
  after the filters. The same filter explains why the 7-iron w = 0.1 tracked
  reference (13.6°) is worse than its IK reference (7.9°).

Evidence: `evidence/head_gaze/fd_neck_tracking.json`. Runs were made on
ControlTower (`ud-sim`) at `b298115bb0` (IK rows) and `4a83b6b7cf` (gaze rows;
the step order changed only for `--fd-neck gaze`). Reproduce with:

```
python3 -m src.shared.python.motion_matching.pipeline.cli --spec docs/development/full_body_models/full_body_spec_anthro_driver.json --capture driver --static-seeds [--gaze-weight 0.1 | --fd-neck gaze] --out RUN
python3 -m src.shared.python.motion_matching.pipeline.cli --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json --capture iron --static-seeds --zmp-filter [--gaze-weight 0.1 | --fd-neck gaze] --out RUN
python3 -m scripts.summarize_fd_neck_tracking docs/development/full_body_models/evidence/head_gaze/fd_neck_tracking.json RUN...
python3 -m scripts.render_head_gaze_clips RUN_IK RUN_GAZE OUT/driver_fd_ikneck_vs_gazeneck --trajectory replay --labels "FD replay, IK neck" "FD replay, gaze neck"
```

Clips of the forward-dynamics replay (IK neck on the left, gaze neck on the
right; 1920x1080, 60 fps, 1x, 0.5x and impact 0.25x) are on host brick under
`~/Videos/Parity Audit/golfer_realism/head_gaze/osv3c/`.

## Head-Gaze Clips

Both clip scripts replay the sweep run's inverse-kinematics reference `q_ref`
for w = 0 and w = 0.1 and pair them side by side (gaze off | gaze on), 1920x1080
at 60 fps, at 1x, 0.5x and an impact-centred 0.25x window of 0.4 s:

- `scripts.render_head_gaze_clips`: MuJoCo, head-forward axis and line of
  sight drawn as capsules.
- `scripts.render_head_gaze_engines`: the Drake, Pinocchio, OpenSim and
  MyoSuite native viewers, face-on and down-the-line, through
  `native_viewer_export` with the same two glyphs as arrows (head-forward
  0.55 m, red; eye to ball at address, green). Each run's same-input bundle
  (closed-loop MuJoCo replay of `q_ref`) is built once and reused while it is
  newer than its inputs.

The clips are kinematic replays, not forward dynamics. Checking their frames
exposed two overlay defects, fixed with this work: MuJoCo draws
`mjGEOM_ARROW` at half the `mjv_connector` length, so MuJoCo and MyoSuite
arrows were half their scale (`force_glyphs.ARROW_RENDER_LENGTH_FRACTION`);
Drake's MeshCat cylinder and cone run along +z while the shared renderer poses
+y shapes, so Drake shafts lay across their segment (`DrakeMeshcatSink`).

```
python3 -m scripts.render_head_gaze_engines SWEEP/driver_0 OUT/driver --label driver_gaze0
python3 -m scripts.render_head_gaze_engines SWEEP/driver_0.1 OUT/driver --label driver_gazeon
python3 -m scripts.render_head_gaze_engines --pair OUT/driver/driver_gaze0 OUT/driver/driver_gazeon OUT/driver/paired --capture driver
```

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
- The forward-dynamics gaze neck is open loop (solved on the reference torso),
  replays in MuJoCo only, and saturates the neck yaw range; see the section
  above. MyoSuite and OpenSim neck actuation, and the receipts of the other
  engines, are not covered yet.
