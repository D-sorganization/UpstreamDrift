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
body; the records now read:

- **Source IK:** max residuals ankle 4.99/4.43, knee 3.62/11.82, hip
  8.11/9.32, shoulder 3.65/3.04 and bar 0.01 mm (left/right); closure 9.0e-8
  m; lowest sphere 0.69 mm; balance offset 0.207 m to 1.3e-4 m in five
  corrections; 463,674 bytes. `ankle_r` (bound 3.6 mm), `hip_r` (9.2 mm) and
  the closure (6.4e-8 m) exceed their unchanged bounds in
  `tests/unit/lifting/test_lift_bundles.py`. The bounds are not widened (see
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
