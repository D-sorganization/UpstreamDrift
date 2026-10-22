# Back-Squat Same-Input Bundle (LIFT-5, #11745)

`back_squat_bundle.npz` is the first lift trajectory packaged as a same-input
`InputBundle` (`src/shared/python/motion_matching/same_input/bundle.py`), so
every engine can replay identical efforts from an identical start state.

## How It Is Built

1. **Planar solution.** The Tools Movement Optimizer
   (`movement_optimizer`, consumed read-only from `vendor/ud-tools/src`) solves
   the `full_squat` exercise (stand, bottom, stand) as a sagittal
   shank/thigh/trunk chain: 80 kg, 1.78 m lifter, 100 kg bar, 3.0 s, 151
   samples.
2. **Retargeting.** `src/shared/python/lifting/sagittal_retarget.py` rotates
   each segment of the full-body model's solved standing pose rigidly about
   the lateral axis until its sagittal angle equals the optimizer's. Ankles
   stay planted; knees, hips, shoulders and the bar landmark follow, keeping
   the model's own segment lengths and left/right offsets.
3. **Full-body IK.** `src/shared/python/lifting/bundles.py` tracks the nine
   landmarks (ankles, knees, hips, shoulders, bar) with the shared MuJoCo
   marker IK (`FullBodyMarkerKinematics`), four foot spheres held flat on the
   ground, arm coordinates held at the seed pose.
4. **Bundle.** A 0.2 s window centred on the squat bottom (lowest bar sample)
   is tracked in MuJoCo by the pipeline computed-torque controller held per
   step (`generate_reference_bundle`), giving 200 steps at 1 ms.

## Provenance in the Manifest

`InputBundle.load(path).provenance` holds the source commit, Tools pin,
optimizer version and parameters, the full planar solution
(`sagittal_solution`), the IK summary (per-landmark maximum residuals,
closure error, foot-sphere heights, options, standing pose), the IK bar path
over the whole lift, the window's bar path (`window_bar_path_m`), the seed
pose record and its SHA-256, and the model limitations.

## Measured IK Residuals

Maximum over all 151 frames, metres, rounded up to 0.1 mm (also asserted by
`tests/unit/lifting/test_lift_bundles.py`):

| Landmark   | Max Residual |
| ---------- | ------------ |
| ankle_l    | 0.0083       |
| ankle_r    | 0.0052       |
| knee_l     | 0.0270       |
| knee_r     | 0.0099       |
| hip_l      | 0.0149       |
| hip_r      | 0.0138       |
| shoulder_l | 0.0097       |
| shoulder_r | 0.0097       |
| bar        | 0.0044       |

Grip-closure error stays below 1.5e-6 m and the lowest foot sphere penetrates
at most 0.54 mm. The largest residuals come from the full-body spec's own
left/right asymmetry (segment lengths, calibrated hip centres), which a
symmetric planar chain cannot reproduce.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` records every engine replaying the whole bundle open
loop (same start state, same efforts, shared ZOH-RK4 integrator,
`src/shared/python/lifting/bar_path_parity.py`) and placing the bar landmark
with its own forward kinematics. Deviations are measured against the MuJoCo
bar path stored in the bundle (`window_bar_path_m`). The tolerances were fixed
before measuring: the epic #11605 level-L2 bounds (1e-3 m for frames,
1e-6 rad for coordinates) for the full window and for a replay re-anchored
to the reference every 20 steps, and 1e-9 m for the 20-step CI replay.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 4.9e-6                  | 4.4e-3                           | 1.8e-4                                | 2.0e-7                         |
| Pinocchio | 5.0e-6                  | 4.5e-3                           | 1.9e-4                                | 2.1e-7                         |
| OpenSim   | 4.9e-6                  | 4.4e-3                           | 2.0e-4                                | 2.2e-7                         |

The bar path, and every spec frame, agrees across all four engines well
inside the 1 mm L2 frame bound. The coordinate bound is not met, and the
measured cause is local rather than steady exponential growth: in the
segmented replay, five of the ten 20-step segments end at round-off
(below 1e-14 rad) in every engine, while the segment from 80 to 100 ms, ending at
the squat bottom, ends 1.8e-4 to 2.0e-4 rad from MuJoCo in
`subtalar_angle_r` in all three other engines; the following segments decay
from 2.3e-7 rad back to round-off. The same 80 ms onset drives the full-window
drift (the bar path agrees to within 1e-15 m for the first 80 ms). A
non-smooth contact event in the foot spheres is consistent with this but is
not yet confirmed. Both coordinate gates are recorded as strict xfails in
`tests/unit/lifting/test_bar_path_parity.py` with these numbers; the bounds
are not loosened. The CI test replays the first 20 steps in each importable
engine and asserts the bar path to 1e-9 m.

Regenerate (one to two minutes per engine on an unloaded host):

```bash
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py
```

## Limitations

- The only spec the same-input harness replays in MuJoCo, Drake and
  Pinocchio is the golf full-body spec, so the lifter keeps its arms and
  dual-grip club; the arms are held fixed, which keeps the grip closure exact.
- There is no barbell body or load in the full-body dynamics. The bar
  landmark is the spec's `Hub` frame (shoulder-girdle centre), matching the
  optimizer's back-squat bar at the shoulder joint. The barbell and hand
  contact arrive with LIFT-4 (#11744).
- Literature comparison of bar path, bar velocity and phase timing is not yet
  recorded: TODO(#11745) add cited reference values as evidence, not tuning
  targets.

## Lift Body (OSL-14, #13057)

The bundles now replay the AN-1 lift body
(`docs/development/lifting/evidence/lift_body/lift_body_spec.json`, from
`src/shared/python/lifting/lift_body.py`), not the golf spec. The limitation
text above, which describes the golf spec, is kept for the bundle provenance.
The lift body has:

- a lumbar joint at L5/S1;
- de Leva hips 0.1851 m apart;
- cited ranges, solved as bounded (TRF) IK.

The grip-width shift also moves the left wrist joint. Every coordinate of the
squat bundle stays inside its cited range.

The LIFT-4 barbell re-track (`back_squat_barbell_bundle.npz`) still leaves its
source (#13072). The controller is now built on the rigid-bar spec
(`barbell_runs.rigid_bar_spec`), which removed the fast divergence. Before that,
the bar-weight feedforward left out the bar's load on the floating base. The
SpineInputY error went from 0.99 deg to 0.0002 deg at step 3, and from
9.76 deg to 0.0001 deg at step 10.

The loaded body still pivots forward on its toes. With the bar in the hands,
the source kinematics put the combined centre of mass 0.15-0.17 m ahead of the
forefoot spheres, while the body alone sits 0.039 m inside them. By step 200:

- pelvis pitch is 11.8 deg off the source;
- the shoulder landmarks are 13.2 cm off;
- the LIFT-5 bounds are crossed from step 58 (ankle) onwards.

The deviation per landmark is recorded in the provenance under
`retrack_landmark_deviation_m`. The bound test is a strict xfail in
`tests/unit/lifting/test_barbell_evidence.py`. See `lift_models.tex`, #13072
subsection.

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py
```

Generation takes a few minutes (most of it the 200-step MuJoCo tracking run).

## Back-Bar Geometry (#12347)

The committed bundle above still carries the golf club grip (hands 0.08 m
apart in front of the body), which is why a barbell attached there tips the
lifter forward (#12345). `scripts/generate_lift_bundle.py --geometry back_bar`
builds the bundle on the standard's back-squat geometry instead
(`src/shared/python/lifting/back_bar.py`, every value read from the parity
standard through `lift_spec`):

- the left hand is moved to the standard's grip width (`grip.width_m`) along
  the club body, with its mass, and the bar centre (`Grip` frame) stays midway
  between the palms;
- the bar sits on a torso shelf: a fixed offset in the `Hub` frame fitted so the
  bar is at the `start` via-pose bar position when the lifter stands (mid-foot
  origin, scaled to the lifter height);
- the scapulae stay at the seed pose and the arm chain is solved with the bar
  long axis medio-lateral (`tests/unit/lifting/test_back_bar_geometry.py`).

Measured on the standing start pose: bar centre 3.4 mm from the standard
position, palms 0.59997 m apart (standard 0.6 m), bar axis 0.05 degrees from
medio-lateral; the committed bundle's golf grip is 0.72 m, 0.076 m and 86
degrees off. A full scratch regeneration takes about 45 s, but is not
committed: it changes the bundle spec hash recorded in the bar-path, kinetics
and barbell evidence (#12340, #12341, #12345) and exceeds two recorded IK
thresholds (`hip_r` 0.0153 m against 0.0138 m, closure 1.9e-6 m against
1.5e-6 m), so it lands with that evidence in the next slice.

## Loaded Foot Stability (OSL-16, #13104)

The committed bundles are the bar-balanced source (#13093,
`--load-balance`) tracked at 64 RK4 substeps with a 0.25 ms effort hold
(`barbell_runs.LOADED_TRACKING_POLICY`). At the original eight substeps the
stiffest loaded foot-contact mode (1.3e5 1/s) left the RK4 stability region,
and the right foot points ran 0.23 m from the IK. The sweep is in
`loaded_foot_tracking.json` (`scripts/measure_loaded_foot_tracking.py`), and
`lift_models.tex` (OSL-16 subsection) holds the full account. The earlier
sections above describe the eight-substep records. They stay for history; the
records now read:

- **Feet:** foot points within 1.81/1.79 mm of the IK, against the ankle
  bounds 3.6/5.7 mm.
- **Bar-path parity:** over the full window the engines agree within 2.2e-9
  rad and the bar within 4.3e-12 m. The coordinate xfails are now plain
  tests.
- **Re-track:** every landmark is within its LIFT-5 bound. Ankles 0.97,
  knees 1.21, hips 1.39, shoulders 1.83 and bar 2.03 mm. The bar is bounded
  by the tightest body-landmark bound (3.6 mm), not the IK's 0.1 mm
  constraint precision.
- **Hold:** the hold starts in contact equilibrium (`barbell_runs.hold_start`,
  #11744), with the CoP under the combined CoM. Body plus bar is carried over the
  whole 1.5 s: ground 1.000, hands 1.000, left share 0.500, deflection 0.49 mm.
  The old start sat 6.7 cm out of balance and toppled backwards.
- **Size:** loaded bundles carry their hold efforts, up to 600,000 bytes.

## Whole-Row Lift Body (#13037)

Every record in this directory is regenerated on the AN-1 lift body built from
the whole de Leva Table 4 rows (#13102) with the default 10 deg tibial torsion
(#12814). The numbers in the previous section were measured on the earlier
body. The list below was measured on the whole-row body before the OSL-17
rack and is kept for history; the current records are in the last section.

- **Source IK:** max residuals ankle 4.99/4.43, knee 3.62/11.82, hip
  8.11/9.32, shoulder 3.65/3.04 and bar 0.01 mm (left/right); closure 9.0e-8
  m; lowest sphere 0.69 mm; balance offset 0.207 m to 1.3e-4 m in five
  corrections; 463,674 bytes. The regression-snapshot bounds in
  `tests/unit/lifting/test_lift_bundles.py` for `ankle_r` (3.6 to 4.5 mm),
  `hip_r` (9.2 to 9.4 mm) and the closure (6.4e-8 to 9.0e-8 m) are re-derived
  with their original rounding (#13041, issuecomment-6243170509; see
  `lift_models.tex`, "Lift Body on the Whole de Leva Rows").
- **Feet:** foot points within 1.77/1.75 mm of the IK at 64 substeps and a
  0.25 ms hold; 64 against 128 substeps moves no coordinate by more than
  1.6e-8 rad.
- **Bar-path parity:** full window within 6.3e-10 rad and the bar within
  7.5e-12 m.
- **Re-track:** ankles 0.99/1.00, knees 1.19/1.22, hips 1.39/1.38,
  shoulders 1.85/1.55 and bar 1.87 mm; 496,399 bytes. Bushing parity largest
  difference 6.2e-7 (body acceleration, scale 1.0e3).
- **Hold:** from `barbell_runs.hold_start`, ground 1.000, hands 1.000, left
  share 0.500, peak deflection 0.49 mm.

## High-Bar Rack and Settled Start (OSL-17, #13131)

The bar sits on the cited high-bar rack: one shaft radius (14 mm) behind the
lift body's C7 landmark (`BackTop`), along the trunk's posterior axis. The
source is Glassbrook et al. 2019 (J Strength Cond Res 33(S1):S1-S18), citing
Wretenberg et al. 1996. It is defined in
`src/shared/python/lifting/back_rack.py`. It replaces the parity standard's
start-pose bar position, which lies above the shoulders.

Both bundles start after a 0.1 s lead-in (100 steps, recorded as
`provenance["lead_in"]` and `provenance["barbell"]["lead_in"]`), so state 0
is settled on its contact. The source bundle's state-0 vertical force is
2224.2 N, steady over the first 20 states; before, it was 1346.6 N rising to
1771.4 N. The OpenSim squat now passes the Hicks residual gate on all 41
states. The numbers and the failed experiments are in
`docs/research/lift_models/lift_models.tex` (OSL-17). Regenerate with:

```bash
python3 scripts/generate_lift_bundle.py --lift squat --load-balance --lead-in 0.1
python3 scripts/generate_barbell_squat_evidence.py --part bundle --source-commit <sha>
python3 scripts/generate_barbell_squat_evidence.py --part parity --source-commit <sha>
```

The barbell hold starts from the balanced IK standing pose
(`barbell_runs.hold_pose`, recorded as `provenance["ik"]["q_balanced_standing"]`):
at the pre-balance `q_standing` the body-plus-bar centre of mass lies 18.6 cm
behind the centre of pressure and no contact equilibrium exists.

Open regressions on the rack bundle, recorded as strict xfails with the bounds
unchanged (#13131):

- Full-window cross-engine coordinates: 6.0e-5 rad (Drake), 2.0e-5 rad
  (Pinocchio) and 1.4e-5 rad (OpenSim) against 1e-6 rad; before, 2e-10 rad.
  The bar stays within 8.0e-7 m and every 20-step segment within 4.8e-13 rad.
- Re-track bar deviation: 6.79 mm against the 3.6 mm LIFT-5 bound; before,
  2.03 mm.
- OpenSim arm reach: the Rajagopal arm ranges stop the hands 4.63 cm (left)
  and 6.03 cm (right) from the high-bar grips against the 1 mm reach bound,
  and the IK carries the arm coordinates up to 1.21e-3 rad past their ranges.
- Body-alone centre of mass: 11.9 cm forward, ahead of the forefoot spheres
  (10.4 cm); body plus bar is over the feet.

All numbers here are on the lift body before the de Leva whole-row fix
(#13102).

## High-Bar Rack on the Whole-Row Lift Body (#13041)

Every record in this directory is now the OSL-17 rack regenerated on the
whole-row lift body (#13037), with the commands above and the default thread
count (the OSL-17 bundle was built without `OMP_NUM_THREADS=1`; the committed
policy run reproduces the bundle).

- **Source IK:** max residuals ankle 4.70/4.89, knee 8.32/14.80, hip
  12.48/6.91, shoulder 8.22/4.66 and bar 0.05 mm (left/right); closure 2.81e-7
  m; lowest sphere 0.69 mm; balance offset 0.097 m to 6.9e-6 m in four
  corrections; 519,569 bytes; 64 RK4 substeps, |z| 2.21. Only `ankle_r`
  exceeds its snapshot (rack on the earlier body 3.53 mm); it is re-derived
  from 4.5 to 4.9 mm with the same rounding.
- **Bar-path parity:** the bar within 3.7e-5 m (Pinocchio) and every 20-step
  segment within 7.0e-13 rad; the full window ends 8.5e-4 (Drake), 2.7e-3
  (Pinocchio) and 8.6e-5 rad (OpenSim) from MuJoCo on `LWInputY`.
- **Barbell parity:** largest engine difference 5.5e-7 (body acceleration).

The OSL-17 strict xfails, re-measured with the bounds unchanged; every one
still fails:

| Regression                               | Rack, Earlier Body | Rack, Whole-Row Body |
| ---------------------------------------- | ------------------ | -------------------- |
| Full-window coordinates, Drake (rad)     | 6.0e-5             | 8.5e-4               |
| Re-track bar deviation (mm, bound 3.6)   | 6.79               | 6.45                 |
| Body-alone CoM / forefoot (cm)           | 11.9 / 10.4        | 11.65 / 10.02        |
| OpenSim reach miss L / R (cm, bound 0.1) | 4.63 / 6.03        | 3.81 / 5.69          |
| Arm range overshoot (rad)                | 1.21e-3            | 8.0e-4               |

- **Full-window coordinates.** The open-loop dynamics under the recorded
  efforts have an unstable left-wrist mode: the largest real eigenvalue of the
  state Jacobian is on `LWInputY`, 116-132 1/s (101-115 1/s on the earlier
  rack body; before the rack 64-70 1/s on `SpineInputX`). Engine round-off of
  about 1e-14 grows by exp(lambda x 0.2 s), 1e9 to 1e11, over the window,
  which matches every measured case. The run is inside the RK4 limit, so this
  is a physical instability of the open-loop posture, not an integration error.
- **OpenSim reach.** This miss and the pre-rack whole-row elbow-limit miss
  (8.3/21.5 mm) have one cause: the folded Rajagopal arm cannot bring the hand
  within about 0.20 m of its shoulder landmark in these directions
  (shoulder-to-grip distance plus reach miss is 0.198-0.213 m in all four
  cases), while the grips sit 0.146-0.190 m from the shoulders.
- **New failure, OpenSim marker gate.** Marker RMS is 2.45 cm against the
  2 cm gate (maximum 3.81 cm, inside 4 cm); it was 1.94 cm. The foot markers
  rise (forefoot 2.4 to 3.8 cm, heel 1.5 to 2.3 cm, ankle 1.0 to 1.6 cm): the
  10 deg tibial torsion (#12814) turns the bundle feet out from 8.8 to 17.9 deg,
  and the torsion-free Rajagopal foot cannot follow. The gate is a
  best-practice threshold and is not loosened, so the marker-gate tests are
  strict xfails tracked by OSL-18 (#13157). The Hicks gate passes (146.6 N,
  18.5 N m) and stays a passing test.
- **Follow-ups.** The arm reach is tracked by OSL-19 (#13158). The open-loop
  left-wrist mode that blocks full-window parity is tracked by OSL-20 (#13159).

## High-Bar Rack Grip-Width Sweep and Wrist-Mode Resolution (OSL-19, #13158; OSL-20, #13159)

The high-bar rack placement was derived from Glassbrook et al. 2019 (doi:10.1519/JSC.0000000000001836)
and Wretenberg et al. 1996, which specify bar placement across the top of the trapezius slightly
inferior to the C7 vertebra. Neither paper specifies a barbell grip width. The parity standard
permits barbell grip widths in the range [0.45, 0.90] m. The standard default of 0.60 m was
selected for an anterior/sagittal bar position, not a posterior high-bar rack.

### Reachable Workspace and Joint-Limit Bottleneck of the Rajagopal Arm (OSL-19)

In the Rajagopal musculoskeletal model, the upper arm (humerus) length is $L_1 = 0.282$ m and the
forearm (radius/ulna) length is $L_2 = 0.250$ m. The elbow flexion coordinate `elbow_flex` is
kinematically bounded at $150^\circ$ ($2.618$ rad).

When the elbow is flexed to its maximum ($150^\circ$), the minimum distance from the glenohumeral
joint center to the hand center of mass in planar 2D geometry is:
$$d_{\min} = \sqrt{L_1^2 + L_2^2 - 2 L_1 L_2 \cos(30^\circ)} = 0.2059\text{ m}$$
The maximum extended reach is $0.6334$ m.

At the standard grip width $W = 0.60$ m, the shoulder-to-grip distance is $0.1700$ m (left) and
$0.1675$ m (right). Both sit strictly inside the unreachable interior sphere ($< 0.2059$ m),
missing reach by $38.1$ mm (left) and $56.9$ mm (right).

#### Geometry Discrepancies and Landmark Definitions

Distances depend on the precise anatomical landmarks and endpoints evaluated:

- **Bundle shoulder landmark to bundle grip point:** At $W = 0.75$ m (frame 0), the Euclidean distance from the bundle `shoulder_l`/`shoulder_r` landmark to the corresponding bundle grip point is **$0.2198$ m** (left) and **$0.2033$ m** (right).
- **OpenSim joint centre (`LSJC`/`RSJC`) to bundle grip point:** The distance from the OpenSim glenohumeral joint centre to the bundle grip point is **$0.2232$ m** (left) and **$0.2052$ m** (right).
- **OpenSim joint centre (`LSJC`/`RSJC`) to OpenSim hand CoM:** The distance from the OpenSim glenohumeral joint centre to the model's hand centre of mass is **$0.2232$ m** (left) and **$0.2059$ m** (right).
- **Hand CoM to grip point reach error:** Hand CoM reaches the bundle grip point within **$0.0024$ mm** on the left, but halts **$1.51$ mm** ($1.5063$ mm) short on the right.

The physical binding constraint halting the right arm $1.51$ mm from the grip is **not** the 2D planar $d_{\min} = 0.2059$ m. Instead, the right arm is simultaneously pinned at **three 3D joint limits**:

1. `arm_flex_r` = $90.00^\circ$ ($1.570796$ rad, upper bound),
2. `arm_rot_r` = $-90.00^\circ$ ($-1.570796$ rad, lower bound),
3. `elbow_flex_r` = $150.00^\circ$ ($2.618000$ rad, upper bound).

#### Measured Coordinate Range Overshoot

The true measured coordinate overshoot in OpenSim IK across the 41-frame squat motion reaches a maximum of **$4.18\times 10^{-5}$ rad** across 5 coordinates:

- `arm_rot_r`: min $-1.570838$ rad vs bound $-1.570796$ rad (overshoot **$4.18\times 10^{-5}$ rad**, active on 6 frames),
- `arm_flex_r`: max $1.570834$ rad vs bound $1.570796$ rad (overshoot **$3.79\times 10^{-5}$ rad**, active on 6 frames),
- `pro_sup_r`: min $-3.25\times 10^{-5}$ rad vs bound $0.0$ rad (overshoot **$3.25\times 10^{-5}$ rad**, active on 6 frames),
- `pro_sup_l`: min $-2.97\times 10^{-5}$ rad vs bound $0.0$ rad (overshoot **$2.97\times 10^{-5}$ rad**, active on 6 frames),
- `elbow_flex_r`: max $2.618001$ rad vs bound $2.618000$ rad (overshoot **$8.79\times 10^{-7}$ rad**, active on 2 frames).

Because right-hand reach error ($1.51$ mm $> 1.0$ mm) and range overshoot ($4.18\times 10^{-5}$ rad $> 0.0$ rad) remain non-zero, the corresponding tests are retained as strict xfails with measured numbers, and `Refs #13158` is maintained.

### Open-Loop Wrist Instability and Parity Resolution (OSL-20)

At $W = 0.60$ m, solving IK for a grip behind the neck forced the lifter's wrists against their joint
limits (`LWInputX` pinned at limit $25.0^\circ$). This constraint coupling excited an unstable open-loop
wrist eigenvalue ($\lambda \approx 116\text{--}132\,\text{s}^{-1}$ on `LWInputY`), which amplified numerical
round-off across the 0.2 s window ($8.5\times 10^{-4}$ rad in Drake, $2.7\times 10^{-3}$ rad in Pinocchio),
causing the full-window coordinate parity gate to fail.

### Grip-Width Sweep and Selection Rule

#### Pre-Declared Selection Rule

We evaluate grip widths $W \in [0.45, 0.90]$ m against an explicit pre-declared rule:
Select the narrowest width in $[0.45, 0.90]$ m that satisfies all five physical gates:

1. Hand-to-bar reach miss $\le 1.0$ mm bilateral,
2. Zero coordinate range overshoot ($= 0.0$ rad),
3. Wrists clear of joint limits by $> 10^{-3}$ rad,
4. No unstable open-loop wrist mode ($\lambda \le 75\text{ s}^{-1}$),
5. Cross-engine Level-L2 coordinate parity $\le 1.0\times 10^{-6}$ rad.

If no width in $[0.45, 0.90]$ m satisfies all five gates, select the width that minimizes right-arm reach miss while strictly satisfying the wrist limit clearance, wrist mode elimination, and cross-engine parity gates, leaving a strict xfail with honest measured numbers for any open criteria.

#### Sweep Results Across Candidate Widths

Measured with the full pipeline at each width (squat bundle, barbell bundle,
OpenSim reach + IK at stride 20, MuJoCo open-loop state Jacobian at frame 0,
full-window bar-path replay in Drake, Pinocchio and OpenSim). The machine-readable
record is [`grip_width_sweep.json`](grip_width_sweep.json). Overshoot is the
largest IK excursion past a model coordinate range; "at bound" lists the reach
solve's arm coordinates within $10^{-3}$ rad of a range limit; parity is the
full-window maximum coordinate error against the MuJoCo reference.

| $W$ (m)             | Reach Miss L / R (mm) | Max Overshoot (rad)      | Reach Coordinates at Bound                | Wrists `LWX`, `LWY`, `RWX`, `RWY` ($^\circ$) | Max Real $\lambda$ ($\text{s}^{-1}$), Mode | Parity Drake / Pinocchio / OpenSim (rad)                            |
| :------------------ | :-------------------- | :----------------------- | :---------------------------------------- | :------------------------------------------- | :----------------------------------------- | :------------------------------------------------------------------ |
| **0.75 (selected)** | **0.0024 / 1.51**     | **$4.18\times 10^{-5}$** | `arm_flex_r`, `arm_rot_r`, `elbow_flex_r` | 11.9, 41.6, -2.8, -20.9                      | **59.37 (`SpineInputX`)**                  | **$6.0\times 10^{-9}$ / $2.4\times 10^{-9}$ / $1.6\times 10^{-9}$** |
| 0.80                | 29.06 / 3.78          | $1.03\times 10^{-3}$     | `arm_flex_l/r`, `arm_rot_l/r`             | 24.5, 47.2, -39.9, 33.4                      | 158.40 (`LWInputY`)                        | 2.10 / 2.02 / 0.20 (diverges)                                       |
| 0.82                | 10.69 / 6.30          | $5.70\times 10^{-4}$     | `arm_flex_l/r`, `arm_rot_l/r`             | 24.5, -54.2, -40.0, 14.0                     | 167.45 (`RWInputX`)                        | 15.5 / 15.5 / 4.13 (diverges)                                       |
| 0.85                | 0.0021 / 9.46         | $3.25\times 10^{-4}$     | `arm_rot_l`, `arm_flex_r`, `arm_rot_r`    | 18.9, 21.4, -22.9, -15.5                     | 66.98 (`RSInputX`)                         | $2.4\times 10^{-8}$ / $3.7\times 10^{-8}$ / $2.1\times 10^{-8}$     |
| 0.88                | 0.0020 / 11.77        | $4.22\times 10^{-4}$     | `arm_flex_r`, `arm_rot_r`                 | 21.8, 18.0, -27.6, -14.5                     | 70.59 (`RSInputX`)                         | $1.2\times 10^{-8}$ / $8.8\times 10^{-8}$ / $4.0\times 10^{-8}$     |
| 0.90                | 0.0037 / 13.27        | $4.88\times 10^{-4}$     | `arm_flex_r`, `arm_rot_r`                 | 23.7, 15.6, -30.4, -13.8                     | 73.03 (`RSInputX`)                         | $6.3\times 10^{-9}$ / $7.4\times 10^{-8}$ / $3.1\times 10^{-8}$     |

#### Selection Evaluation

Applying the pre-declared rule:

1. No width meets reach $\le 1.0$ mm and zero overshoot together: the right
   arm reaches its `arm_flex_r` and `arm_rot_r` limits at every width.
2. $W = 0.80$ and $0.82$ m put the wrists at their limits (`LWInputX`
   $24.5^\circ$ of $25^\circ$, `RWInputX` $-40^\circ$), excite an unstable
   wrist mode ($\lambda = 158$ and $167\,\text{s}^{-1}$) and diverge in the
   full-window replay, so they fail the wrist and parity gates.
3. $W = 0.85$--$0.90$ m pass the stability and parity gates, but the right-hand
   miss grows with width (9.46, 11.77, 13.27 mm) and the overshoot is about ten
   times that at 0.75 m.
4. By the fallback rule (minimise right reach miss while meeting the wrist,
   stability and parity gates), **$W = 0.75$ m** is selected: right miss
   1.51 mm, overshoot $4.18\times 10^{-5}$ rad, $\lambda = 59.37\,\text{s}^{-1}$
   on `SpineInputX`, parity at or below $6.0\times 10^{-9}$ rad.

An earlier draft of this section and of PR #13344 contained rows for
0.80--0.90 m that were not measured; the table above replaces them with the
measured sweep.

Reproduce: for each $W$, run `scripts/generate_lift_bundle.py --lift squat
--load-balance --lead-in 0.1 --grip-width W --out <bundle>`, build the barbell
bundle with `build_barbell_bundle`, then `reaching_lift_build` and
`run_inverse_kinematics` at stride 20, `stiffness._state_jacobian` on the
MuJoCo plant at frame 0, and `bar_path_parity.replay_bar_path` per engine.

### Model Accommodation Statement

The selected barbell grip width of $0.75$ m is an explicit model accommodation to the Rajagopal musculoskeletal arm range limits, not a measured human high-bar athletic grip. In the Rajagopal model, the rigid thoracic trunk and fixed revolute joint limits severely restrict the posterior upper-extremity reachable workspace.
