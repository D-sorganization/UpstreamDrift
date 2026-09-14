# Full-Body Models Handoff (Epic #10062)

Updated 2026-09-13 (claude; leases on #10063 and #10064). Branch
`docs/10003-opensim-matching-epic`, worktree
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003`, PR not
created. Design: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md).
Copy-ready prompt for the next agent: [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

## Done (Test-First, No Engine Required)

FB-1 (#10063) and FB-2 (#10064) are implemented and their child issues can be
closed by the PR that carries this branch.

- `src/shared/python/motion_matching/full_body_spec.py`: `BodySpec`,
  `JointSpec`, `LowerLimbExtension`, `ContactSpec`, `ContactSphere`,
  `MarkerAttachment`; `derive_full_body_spec`, `validate_full_body_spec`,
  `upper_body_slice`, `canonical_sha256`, `save/load_full_body_spec`,
  `pelvis_alignment` (Kabsch reuse). The validator enforces that the embedded
  upper-body slice is byte-identical to the qualified native spec, that every
  body is reachable from world through exactly one joint (reusing
  `order_native_tree`), SPD inertias, positive masses, the shared contact
  parameter set, tracked capture labels only, and provenance. Tests:
  `tests/unit/motion_matching/test_full_body_spec.py` (14).
- `src/shared/python/motion_matching/contact_law.py`: the shared law
  (`f_n = k d (1 + c d_dot)` clipped at zero; regularised Coulomb friction with
  static/dynamic/viscous terms and `tanh` transition), `ContactParameters`,
  `GroundPlane`, `ContactSample`, `sphere_ground_contact`,
  `random_contact_states`, `contact_parity_report`. Tests:
  `tests/unit/motion_matching/test_contact_law.py` (5) with closed-form values.
- Committed document `full_body_spec_v1.json` (canonical SHA256
  `06272a18eadda9fb89b2bca2c1948f7c57963ebb8e635193bf10947943c67a5f`), built
  by `build_full_body_spec.py` with `build_receipt.json`: 41 coordinates (the
  27 qualified upper-body coordinates first, then hip flexion/adduction/
  rotation, knee, ankle, subtalar and MTP per side), 24 bodies (14 upper plus
  femur, tibia, talus, calcn, toes per side from Rajagopal 2016 via the
  packaged golf_humanoid.osim), 23 joints, four foot contact spheres, and 34
  marker attachments (25 with the native lane's qualified offsets, 9 lower-limb
  and `RShoulderTop` attachments with offsets left `null` for FB-4).
- Pelvis alignment between the native `Hip` frame and the Rajagopal pelvis
  frame by rigid registration of the four waist markers: RMS residual 5.8 mm,
  transform recorded in the receipt.

## Simplifications Recorded in the Document

Patella bodies and patellofemoral coupler constraints dropped; the walker
knee reduced to a hinge about its primary axis (coupled spline translations
ignored); OpenSim body inertias interpreted as about the mass centre in the
body frame; contact stiffness, dissipation, friction and sphere placements
are placeholders sized from the calcaneus geometry; ground height is
uncalibrated (`calibrated: false`) until FB-4 sets it from toe markers in the
native world frame. The hip primitive sequence Rx, Ry, Rz maps onto OpenSim's
Z, X, Y flexion/adduction/rotation order through a recorded permutation; FB-3
forward-kinematics parity tests against OpenSim decide whether that mapping
and the Euler convention (SimTK body-fixed XYZ) are right, and must fail
loudly if not.

## FB-3-M Completed (#10066)

- `src/engines/physics_engines/mujoco/python/full_body_mjcf.py`: `export_full_body_mjcf`
  converts `full-body-v1` documents to MJCF XML preserving all 41 scalar joints
  (27 upper-body joints + 14 lower-limb joints), upper-body kinematic transforms
  and aggregated inertias, 16 frame sites, 2 weld closure sites, and adds 4
  calcaneus contact sphere geoms and sites (`contact_heel_r`, `contact_forefoot_r`,
  `contact_heel_l`, `contact_forefoot_l`). Stock MuJoCo contact solver is disabled
  (`flag contact="disable"`).
- `src/engines/physics_engines/mujoco/python/full_body_model.py`: `NativeMujocoFullBodyModel`
  provides full-body kinematics, `evaluate_contact_samples`, and `accelerations`.
  Applies the shared FB-2 Hunt-Crossley and regularized Coulomb contact law
  (`sphere_ground_contact`) via spatial wrenches in `data.xfrc_applied` and
  generalized force projection `jac_pos.T @ f_contact`. Resolves the dual-grip
  closure constraint via an explicit rigid solve.
- `tests/unit/motion_matching/test_full_body_mujoco.py`: 5 live simulation tests
  verifying compilation ($nq=41, nv=41$), upper-body slice inertia and FK parity
  to 1e-12, frame poses parity against the qualified native model, and exact 0.0 N
  contact force parity on the shared FB-2 harness.

## FB-3-P Completed (#10065)

FB-3-P is implemented and verified on ControlTower with Pinocchio 4.1.0:

- `src/engines/physics_engines/pinocchio/python/native_model.py`: extended with
  `FullBodyPinocchioModel` and `build_full_body_pinocchio_model` without touching
  the qualified `NativePinocchioModel`.
- Foot contact spheres added as operational frames to the Pinocchio model before
  `createData()`.
- Shared contact law evaluated via `contact_forces(...)` using
  `sphere_ground_contact` from `src.shared.python.motion_matching.contact_law`.
- Contact forces converted to generalized joint torques via spatial Jacobian
  projection $\tau_{\text{contact}} = \sum J^T F_{\text{contact}}$ and combined with
  actuator efforts into `pin.constraintDynamics(...)` retaining the weld loop
  closure constraint solver.
- Gate results:
  - Gate (a): Upper-body slice reproduces qualified model mass matrix and FK to
    $0.0$ max difference (< $10^{-12}$) across 20 random states.
  - Gate (b): Full-body FK matches spec frames to $0.0$ max difference (< $10^{-12}$)
    across 20 random states.
  - Gate (c): Contact forces at analytic states match the FB-2 reference with $0.0$
    difference in normal force, friction force, and penetration depth.
  - Gate (d): Closure residuals unchanged on the upper-body chain to $0.0$ max
    difference (< $10^{-12}$) across 20 random positions and velocities.
- Evidence archived under `docs/development/full_body_models/evidence/fb3_pinocchio/`:
  - Driver: `verify_pinocchio_full_body.py` (auto-dispatches to ControlTower if
    local Pinocchio is absent).
  - Receipt: `receipt.json`.
- Tests: `tests/unit/motion_matching/test_pinocchio_full_body_builder.py` (6 passed
  on ControlTower in 0.46s; clean skip on Windows).

## FB-3-D Completed (#10067)

FB-3-D is implemented and verified on ControlTower with Drake 1.57.0:

- `src/engines/physics_engines/drake/python/full_body_urdf.py`: exports `full-body-v1`
  documents to URDF XML preserving all 41 scalar joints (27 upper-body joints + 14
  lower-limb joints), physical solid inertias, 16 upper-body frames, 2 weld closure
  frames, and 4 foot contact sphere links (`contact_heel_r`, `contact_forefoot_r`,
  `contact_heel_l`, `contact_forefoot_l`). Sidecar metadata emitted with `schema_version: 1`,
  `requires_sidecar: True`, and `representation: native-full-body-urdf-v1`.
- `src/engines/physics_engines/drake/python/full_body_model.py`: `FullBodyDrakeModel`
  provides full-body kinematics, `contact_forces`, and `accelerations`.
  Applies the shared FB-2 Hunt-Crossley and regularized Coulomb contact law
  (`sphere_ground_contact`) via spatial contact wrenches and generalized force
  projection $J_{\text{trans}}^T F_{\text{contact}}$ into the rigid continuous
  constrained KKT solve (`solve_weld_acceleration`) enforcing the 6-DOF dual-grip weld
  closure without numerical relaxation.
- Gate results:
  - Gate (a): Upper-body slice reproduces qualified model mass matrix and FK to
    $0.0$ max difference (< $10^{-12}$) across 20 random states, with exact link mass
    and inertia parity.
  - Gate (b): Full-body FK matches spec frames to $0.0$ max difference (< $10^{-12}$)
    across 20 random states.
  - Gate (c): Contact forces at analytic states match the FB-2 reference with $0.0$
    difference in normal force, friction force, and penetration depth across all 4 spheres.
  - Gate (d): Closure residuals unchanged on the upper-body chain to $0.0$ max position
    difference and $2.22 \times 10^{-16}$ max velocity difference (< $10^{-12}$) across
    20 random positions and velocities.
  - Accelerations: 41 coordinates, all finite under combined contact forces and weld closure.
- Evidence archived under `docs/development/full_body_models/evidence/fb3_drake/`:
  - Driver: `verify_drake_full_body.py` (auto-dispatches to ControlTower if local Drake is absent).
  - Receipt: `receipt.json` (`059ed50e48b65e0d84e3d7c7cfc7bfd0a99262470e28387e5f9b8a5ee0447001`).
- Tests: `tests/unit/motion_matching/test_full_body_drake.py` (6 passed on ControlTower
  in 0.98s; 1 passed, 5 cleanly skipped on Windows without pydrake).

## Visual Skeleton Layer Completed (Step 1 of the Visuals Plan)

- `src/shared/python/motion_matching/visual_skeleton.py`: one engine-agnostic
  visual description derived from any native or full-body spec (capsule per
  body, COM and frame spheres, ground plane opposite gravity with a
  `calibrated` flag, world-segment mapping for viewers); 5 unit tests.
- `src/engines/physics_engines/mujoco/python/visual_layer.py` and
  `export_full_body_mjcf(..., visual=True)`: massless non-colliding visual
  geoms, floor, lights, camera; physics proven identical to the plain export
  (masses, inertias, xpos, qacc) in `tests/unit/motion_matching/test_mujoco_visual_layer.py`.
- Evidence `evidence/visual_layer/` (default pose PNG, returned81 kinematic
  playback GIF, MJCF, receipt). Steps 3 to 6 (viewer tile, cross-engine replay
  videos, OpenSim real horizon, FB-3 re-checks) are specified for lower-level
  agents in [VISUALS_HANDOFF.md](VISUALS_HANDOFF.md).
- Capsule radius policy (user review 2026-09-13: club and segments looked
  uniformly thick): radius of a uniform-density cylinder,
  `sqrt(m / (pi * 1500 kg/m^3 * L))`, clamped to 6 to 50 mm, and one capsule per
  child joint plus one to the farthest centre of mass. Club shaft 17 mm,
  forearms 36 to 40 mm, torso and thighs 50 mm (`address_segments.json`).
- Address-pose evidence at the returned81 candidate's q0 (the Simscape
  qualified start, legs at zero): `address_{front,side,top}.png` with frame-0
  capture markers (black) and native marker predictions (blue) overlaid
  (`render_address_views.py`, frame-0 valid marker RMS 2.9e-13 m), the
  labelled joint-to-joint Simscape skeleton `simscape_skeleton_address.png`
  with `address_posture.json` (`render_simscape_skeleton.py`), and
  `render_playback.py` for the default pose and playback GIF. No MATLAB is
  runnable on DeskComputer (`C:/Program Files/MATLAB/R2025b` has no
  `bin/matlab.exe`), so the Simscape skeleton is drawn from the ported
  geometry whose frame parity against R2025b is receipted at 2.1e-13 m
  (`native_frame_parity_r2025b.json`).
- Posture facts the user should decide on (model and start pose, not
  rendering): Simscape geometry inputs UpperArmLength 14.5 in (0.368 m) and
  LowerArmLength 12 in (0.305 m elbow to wrist); hub to shoulder 0.254 m;
  the hip, torso and spine joints coincide, so the pelvis-to-rod bend of
  -16.9 deg (SpineInputX) at q0 is what the frame-0 marker fit needed; the
  shoulders sit 0.09 m (L) and 0.19 m (R) below the hub at q0 (scapula
  coordinates 30/-40 deg and 35/38 deg). A straight spine or higher
  scapulae at address means refitting q0 (or re-calibrating attachments)
  with those coordinates constrained, then re-qualifying.

## Ground Support Program (User Direction 2026-09-13, in Progress)

User direction: when the legs are shown, the golfer must be carried by the
ground through modelled contact, keep dynamic balance, and the legs must
match the c3d leg markers under full physics in MuJoCo, Drake and Pinocchio.
Gates: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md) section
"GS Ground Support". Branch feat/10062-visual-skeleton-layer, PR #10087.

Shared contracts (all with unit tests under `tests/unit/motion_matching`):

- `src/shared/python/motion_matching/ground_support.py`: capture to native
  world `(x, y, z) -> (x, -z, y)` (capture frame 0 is native t=0 exactly),
  ground height from the lowest toe markers minus a standoff, support report
  (weight fraction, centre of pressure, convex support polygon).
- `hip_calibration.py`: functional hip centres by sphere fit of the knee
  markers in the pelvis frame (sd 2 mm over 649 frames), anatomical pelvis
  axes (right = hip line, up = the Hip frame's +z, forward = up x right),
  rewrite of a document's hip joints. `segment_scaling.py`: length scaling
  of named bodies. `marker_calibration.calibrate_marker_offsets` gained an
  anatomical prior (`prior_offsets`, `prior_weight`).

MuJoCo lane:

- `full_body_markers.py`: marker FK on spec frames and bodies (lower-limb
  offsets mapped through `adapter.body_frames`, because MJCF bodies sit at
  the joint follower frame), projected Levenberg-Marquardt pose IK with grip
  closure, one-sided ground penalty, stance pins, planted-sphere anchors
  (`plant_stance`), CoM-over-support rows, joint bounds, locks, prior
  trajectory, body poses for the shared calibration.
- `full_body_simulation.py`: RK4 over the adapter's closure-constrained
  accelerations with an unactuated root; affine dynamics `a = A tau + b`
  from one KKT solve; least-norm inverse dynamics (closure removes six
  directions, singular values below 1e-2 dropped); computed-torque hold and
  tracking controllers with per-joint natural frequencies, optional CoM
  balance and root regulation; feet preload; support record. Standing hold
  on the balanced address posture: weight fraction 1.000, CoM drift 1.5 mm
  over 2 s.

Specification defects found and fixed:

- v1 had the hip and knee permutation matrices swapped in
  `build_full_body_spec.py` (hip flexion turned about the femur's long axis,
  the knee about the femur's y axis). `full_body_spec_v2.json`
  (`build_receipt_v2.json`) fixes it; `test_lower_limb_axes.py` guards it.
  v1 and every FB-3 receipt built on it are superseded for lower-limb use.
- The v1/v2 pelvis alignment (from the OS-3 pelvis offsets, a 0.2 m RMS fit)
  put the hips mirrored and about 0.15 m off; the driver relocates them to
  the functional centres (`full_body_spec_hipcal.json`).
- Heel and metatarsal spheres alone leave the address centre of mass 3 to
  6 cm ahead of the support polygon (the model tips forward); the driver
  adds toe spheres at calcaneus x = 0.23 m and raises the placeholder
  contact stiffness from 5e4 to 2e5 N/m (`full_body_spec_hipcal_scaled.json`).

Evidence `evidence/ground_support/` (`run_ground_support.py`, `receipt.json`,
`ik_trajectory.npz`, `dynamics_record.npz`, playback GIFs and frame
montages), spec SHA in the receipt:

- Address (GS-2): whole 3.1 mm, legs 5.4/5.2 mm, arms 0.5/0.4 mm, pelvis
  4.3 mm; hip flexion 44/59 deg, hip rotation -17/+7 deg; feet flat.
- Leg calibration: 23 to 16 mm on 109 frames; femur scale 0.97 chosen from a
  {0.94, 0.97, 1.00}^2 grid; after scaling 20 to 13 mm.
- Reference (GS-3): full-capture IK 28.8 mm whole (legs 12/14 mm, arms 31/26,
  pelvis 44, head 47, club 10) against an upper-body-only floor of 26 mm;
  smoothed 12 Hz plus consistency re-solve 28.5 mm, stance spheres within
  2.4 mm of the plane, planted-sphere drift 2.1 mm, closure 0.9 mm.
- Dynamics (GS-4, open): computed-torque tracking, unactuated root, joints
  to 0.0035 rad. To 1.0 s (address and backswing): root error max 14 mm,
  marker RMS 22 mm, weight fraction 0.36 to 1.70. Downswing and impact
  diverge: root error 21 mm at 1.1 s, 65 at 1.2, 151 at 1.3, 313 mm at
  1.75 s; whole-run marker RMS 174 mm, inside-polygon 85 %, peak torque
  4375 N m. Root regulation through the legs and CoM balance terms did not
  help (receipted in the session, not adopted). Joint bounds are the
  Rajagopal ranges widened 2x because the hip zero twist is not calibrated.

## Anthropometry and Posture Review (User Direction 2026-09-14)

The user asked why the torso looks arched at address and how the body
dimensions were determined. Answer in
[evidence/anthropometry/REVIEW.md](evidence/anthropometry/REVIEW.md) with
`receipt.json` (`review_anthropometry.py`): the golfer's markers show a hip
hinge with a modest trunk lean, no spinal arch; the model's bend is a 7 deg
extension plus a 21 deg side bend at its only trunk joint, which sits at the
base of the neck 0.515 m above the hips, with the hub 0.14 m above the
shoulder centre so the clavicle links point 21 and 47 deg downward. Upper
arms are 33 % and forearms 15 % too long, the trunk-plus-head 70 % too heavy
(model 108 kg versus about 78 kg). Shared modules added: `anthropometry.py`
(de Leva table, transcribed, verify against the paper before qualification),
`posture_metrics.py`, `anthropometric_candidate.py` (de Leva lengths,
masses and inertias on the current topology; unqualified). The candidate
run (`ground_support/candidate_anthro/`) shows scaling alone makes the fit
worse (48 mm), so the trunk topology must change; section 8 of the review
is the ready-to-file child issue. The driver `run_ground_support.py` gained
`--anthropometric`, `--recalibrate-upper`, `--out` and posture metrics in
its receipt.

## AN-1 Iteration 1: Anthropometric Native Geometry (#10099, in Progress)

Findings and receipts in [evidence/anthropometry/REVIEW.md](evidence/anthropometry/REVIEW.md)
section 9. Code (all with unit tests, 15 in the touched files):

- `src/shared/python/motion_matching/anthropometric_geometry.py`:
  `build_upper_body(native, stature_m, mass_kg, trunk_scale, arm_scale,
shoulder_scale)`; 27 native coordinate names, native body/joint/frame
  names, club and closure verbatim; pelvis, trunk to the shoulder centre,
  hub at the shoulders, scapula `Rx` elevation + `Rz` protraction (the
  native `Ry` was a pure spin), upper arms forward at zero pose so the
  shoulder gimbal stays away from its singularity, one-sided elbows;
  `COORDINATE_RANGES_DEG` and `ADDRESS_SEED_DEG` travel in the document.
- `docs/development/full_body_models/build_anthropometric_spec.py` writes
  `full_body_spec_anthro_<club>.json` (+ `build_receipt_anthro_<club>.json`) with
  the Rajagopal legs on a fixed pelvis alignment, toe spheres, 2e5 N/m.
  Canonical scales: trunk 1.15, arm 1.10, shoulder 1.00 at 1.71 m, 78 kg.
- `marker_calibration.static_marker_offsets` (static-trial placement);
  `full_body_markers.solve_pose(marker_weights=...)` and
  `solve_trajectory(restarts, restart_threshold_m)`.
- Driver `run_ground_support.py`: `--static-seeds` (neutral address with
  locked scapulae and a bounded spine, offsets from the first 24 frames),
  document bounds and seed for anthropometric documents, head markers at
  weight 0.1 in every IK (`HEAD_MARKER_WEIGHT`), four restarts above 30 mm.
  These last two change the baseline numbers of every rerun: report
  body-only and all-marker RMS side by side.
- `evidence/anthropometry/scan_geometry.py` ranks (trunk, arm, shoulder)
  scales by the decimated-swing body-marker RMS with static offsets and no
  calibration (`scan_geometry_receipt.json`).

State against the #10099 acceptance (full driver run, REVIEW.md 9.1,
`evidence/ground_support/anthro_driver/receipt.json`): address 2.1 mm with
spine bend 0.3 deg forward, 6.3 deg lateral and clavicle links within 5 deg
of horizontal (met); full-capture IK 26.2 mm on the non-head markers, equal
to the qualified geometry's 26.2 mm, 40.0 mm with the deweighted head (no
neck; 104 mm); tracking to 1.0 s root error 5 mm (qualified 14 mm),
whole-run root RMS 49 mm (qualified 172 mm), weight fraction still reaches
0 (GS-4 open). Solver additions for this: `continuous_branches` (Euler
branch and 2 pi continuity before smoothing), per-coordinate
`prior_weights` (0.1 on the three collinear spins), restart margin 3 mm.
Renders: `evidence/visual_layer/*_driver.*` and `*_iron.*` (scripts take `--spec
--trajectory --suffix`). Unqualified until Simscape carries the geometry.

### Neck and Address Arms (User Direction 2026-09-14)

Head body on a three-axis neck (`NeckInputX/Y/Z`, 30 upper coordinates;
Simscape has none, accepted), head markers at full weight; address elbows
bounded in the static trial with weak elbow-pit axis rows
(`solve_pose(axis_targets=...)`) and restarts in the address fit; lane
settings shared by driver and scan (`configure_lane`). Receipt
`evidence/ground_support/anthro_driver/receipt.json`: full-capture IK 30.2 mm
over all markers (head 28 mm), address 11.4 mm neutral, left elbow -25 deg,
right -2 deg, backswing root error 7 mm. The left elbow pit still faces
outward: forcing it inward breaks the swing fit (REVIEW.md 9.2 lists the
four attempts); the wrist frame copied from the native document is the
suspect. Renders `evidence/visual_layer/*_driver.*` and `*_iron.*` refreshed.

### Wrist Axis, Setup Position, Parity, Centre of Mass (2026-09-14)

Wrist cock axis rolled onto the elbow axis (`WRIST_ROLL`), lead scapula
retraction and elbow windows in every address fit, centre-of-mass overlay
(`visual_layer.add_com_markers`, driver playback and address views,
`centre_of_mass` in the address receipt), cross-engine setup parity
(`evidence/setup_parity/`: Pinocchio 9e-16 m, Drake 6.5e-06 m on the
same document; poses translate by coordinate name). Receipt (then anthro_v1, now anthro_driver):
IK 30.4 mm all markers, address 8.3 mm, elbows -7.5/-4.7 deg,
backswing root 5 mm, CoM inside the polygon at address: True.
Open: the left elbow pit faces outward (REVIEW.md 9.3).

### Clubs, Two Captures, Torso Visuals, Ranges of Motion (2026-09-14)

`club_models.py` (driver, 7-iron; `apply_club`), documents
`full_body_spec_anthro_driver.json` and `full_body_spec_anthro_iron7.json`
(builder `--club`), the 7-iron capture registered in the contract
(`--capture iron`), `visual_hints` for torso ellipsoids, clavicles, shaft and
club heads, `range_of_motion.py` with receipt flags, address balance rows.
Receipts `evidence/ground_support/anthro_driver/` and `anthro_iron/`:
IK 24.8 / 34.1 mm, address 7.1 / 15.5 mm, CoM
inside the polygon in both; setup parity `evidence/setup_parity/receipt_*.json`.
REVIEW.md section 10.

### Anatomical Wrist, Visual Realism, Launcher Tool, Epic #10113 (2026-09-14)

Wrist Rz flexion with a 25 deg neutral-grip offset (driver IK 24.0 mm,
7-iron 24.1 mm; wrists flagged, not bounded: bounding collapses the fit
because the left humerus roll is wrong, MM-2/MM-5); thicker legs, slimmer
torso, hidden marker spheres; `club_models.from_database`; the "Motion
Matching" launcher tile (`src/tools/motion_matching`, tests under
`tests/tools/motion_matching`). Epic #10113 tracks MM-1 to MM-10; REVIEW.md
section 11 carries the table.

## Next

- Epic #10113 in order: MM-2 grip roll calibration from the address so the
  wrist ranges can be imposed (#10104, with MM-5 #10107 the left pit);
  MM-6 hip zero twist so the leg ranges apply unwidened (#10108); MM-3
  meshes from BunkerShot3D solids and catalogue clubs (#10105); MM-7
  downswing dynamics (#10109); MM-10 dynamics replay parity (#10112);
  MM-8 Simscape parity (#10110); MM-9 de Leva verification (#10111).
  Keep one static-trial round; never `--recalibrate-upper`.
- GS-4: the downswing needs a dynamically consistent reference or the FB-5
  contact-aware shooting fit; candidates in order: stance timing from the
  reference contact forces instead of marker heights, a hip zero-twist
  calibration so the Rajagopal ranges apply unwidened, then the two-window
  fit on the full body with contact. Every attempt keeps its receipt.
- GS-5: rebuild FB-3-P and FB-3-D on v2 with the hip-calibrated, scaled
  document and rerun same-input replay parity.
- Cross-engine same-input replay and comparison across MuJoCo FB-3-M (#10066), Pinocchio FB-3-P (#10065), and Drake FB-3-D (#10067).
- FB-4 ground height and marker calibration, and FB-5 fitting per Epic #10062.
