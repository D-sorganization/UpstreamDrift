# Anthropometry and Posture Review of the Full-Body Golf Model

Date 2026-09-14. Numbers come from `receipt.json` beside this file
(`review_anthropometry.py`), from the ground-support receipt in
`../ground_support/`, and from the native geometry document. The golfer's
stature and mass were never measured; everything below is inferred.

## 1. How the Current Dimensions Were Determined

| Part                                                  | Source of the dimension                                                                                                                                            | Source of mass and inertia                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Pelvis to neck block (`LowerTorso`, `UpperTorsoBase`) | Simscape `GolfSwing3D_Kinetic` rigid transforms (6 in, 1.2 in offsets); the root "Hip" joint sits 0.515 m above the functional hip centres                         | Simscape solids: 20 kg + 4 kg                        |
| Neck-base rod (`COMRod`, spine joint to hub)          | Simscape constant 9.6 in (0.249 m)                                                                                                                                 | 16 kg "UpperTorsoTop" + 5 kg head + 2.1 kg neck      |
| Hub to shoulder links (`HubtoLS/RS`)                  | Simscape constant 10 in each (0.254 m)                                                                                                                             | 10 kg each                                           |
| Upper arm, forearm                                    | Simscape workspace lengths chosen by the native lane's geometry sweep (`geometry_in` 14.5 in and 12 in, from a 13.5 to 15.5 by 11 to 13 grid judged by marker fit) | Simscape solids 2.27 kg, 1.13 kg + 1.13 kg           |
| Club and hands                                        | Simscape club with two 0.55 kg hand solids                                                                                                                         | 1.48 kg                                              |
| Legs                                                  | Rajagopal 2016 generic (1.70 m, 75 kg subject); femur scaled 0.97 by the pinned IK grid; hips relocated to functional centres                                      | Rajagopal masses (thigh 9.8, shank 3.9, foot 1.7 kg) |
| Marker attachments                                    | 25 upper-body offsets calibrated by the native lane on the first 0.85 s and frozen; 8 leg offsets calibrated here with an anatomical prior                         | not applicable                                       |

Nothing in this chain used the golfer's own anthropometry. The upper-body
lengths are Simscape constants plus a two-parameter sweep; the masses are
the Simscape model's default solids.

## 2. What the Capture Says About the Golfer

Marker-to-marker distances (mean over the 654 frames) and derived lengths:

| Quantity                                                 | Value           | Note                                       |
| -------------------------------------------------------- | --------------- | ------------------------------------------ |
| Knee to ankle marker, left/right                         | 0.424 / 0.422 m | shank close to 0.42 m                      |
| Functional hip radius to knee marker, right/left         | 0.403 / 0.383 m | femur about 0.40 / 0.38 m                  |
| Elbow to wrist marker, left/right                        | 0.277 / 0.281 m | forearm about 0.26 to 0.28 m               |
| Shoulder top to shoulder top                             | 0.329 m         | biacromial breadth about 0.36 to 0.40 m    |
| Waist left to waist right                                | 0.344 m         |                                            |
| Head top max height                                      | 1.697 m         | golfer never fully upright in this capture |
| Stature from de Leva proportions (thigh, shank, forearm) | 1.70 m          | head-top bound gives at least 1.72 m       |

Working estimate: stature 1.71 m; mass unknown, 78 kg assumed (BMI 26.7,
typical for a tour athlete) until the user supplies it.

## 3. Model Versus Anthropometry

Lengths (model at zero pose; de Leva scaled to 1.71 m):

| Segment                                      | Model           | de Leva                          | Capture                  | Verdict                                                   |
| -------------------------------------------- | --------------- | -------------------------------- | ------------------------ | --------------------------------------------------------- |
| Upper arm (shoulder to elbow)                | 0.368 m         | 0.277 m                          | 0.33 to 0.41 marker span | 25 to 33 % too long                                       |
| Forearm (elbow to wrist)                     | 0.305 m         | 0.264 m                          | 0.28 m marker span       | 15 % too long                                             |
| Shoulder to shoulder at zero (two hub links) | 0.508 m         | 0.39 m biacromial                | 0.33 m marker span       | 30 % too wide; fitted pose folds the links down to 0.40 m |
| Hips to shoulder centre (fitted address)     | 0.60 m          | about 0.50 m                     |                          | 20 % too long                                             |
| Hips to root ("spine") joint                 | 0.515 m         | trunk 0.52 m (hips to cervicale) |                          | the only trunk joint sits at the base of the neck         |
| Hub above the shoulder centre (address)      | 0.14 m          | 0 (hub is the shoulder centre)   |                          | links must point 21 and 47 deg downward                   |
| Femur                                        | 0.393 / 0.381 m | 0.415 m                          | 0.40 / 0.38 m            | fine                                                      |
| Tibia                                        | 0.435 / 0.431 m | 0.426 m                          | 0.42 m                   | fine                                                      |

Masses (kg):

| Group                                     | Model | de Leva at 78 kg            | Verdict        |
| ----------------------------------------- | ----- | --------------------------- | -------------- |
| Trunk and head (incl. 20 kg of hub links) | 67.0  | 39.4 (trunk 33.9, head 5.4) | 70 % too heavy |
| Arms                                      | 9.1   | 7.7                         | fine           |
| Legs                                      | 30.7  | 30.9                        | fine           |
| Club and hands                            | 1.48  | 0.96 hands + club           | fine           |
| Total                                     | 108.3 | 78                          | 39 % too heavy |

The trunk inertias in the native document are spheres and short rods placed
at solid origins; none follows a radius-of-gyration table. The 20 kg carried
by the two hub links is the shoulder girdle mass, placed 0.25 m out from the
neck base, which inflates the trunk inertia about the spine.

## 4. The Posture Question (Straight Torso at Address)

Model-free, from the markers at address: pelvis plane tilted 18.6 deg forward
and 2.5 deg to the left; trunk line from the waist centroid to the upper back
marker 28 deg forward and 6 deg to the right; upper back rolled 7.7 deg (left
shoulder higher, the usual secondary tilt). The golfer is hinged at the hips
with a modest forward and side lean of the trunk; nothing in the markers
suggests a rounded or arched spine.

Model, at the fitted address: the hips-to-root block is 23.9 deg forward,
the neck-base rod 16.7 deg forward and 18.4 deg to the right, so the single
"spine" joint carries a 7 deg extension and a 21 deg side bend. At the top of
the backswing the side bend reverses to 24 deg the other way. The rendering
shows exactly this: one straight 0.5 m block, a kink at the neck base, then
the rod and two hub links pointing 21 and 47 deg downward to the shoulders
(67 deg at the top). The "depressed scapula" and the "arched" look are the
same defect: the hub is 0.14 m above the shoulder centre and 0.15 m too far
from the hips, so the fit lowers the scapula joints and leans the neck rod
to put the shoulder markers where they were measured.

Conclusion: the bend is not the golfer's and not the fit's fault; the fixed
Simscape trunk dimensions force it. A straight, neutral torso at address is
only reachable by changing the geometry.

## 5. Recommendations (Ordered, Each With a Test)

1. Subject anthropometry as data: a `subject` block in the document
   (stature, mass, source, date) and every segment length, mass, centre of
   mass and inertia derived from de Leva (`anthropometry.py`) unless measured.
   Test: rebuilding the document from the block reproduces it.
2. Anthropometric upper-body topology (native geometry v2, unqualified until
   Simscape parity): pelvis body from the hip centres to L5/S1 (0.11 m),
   lumbar-thoracic trunk to the shoulder centre (0.40 m), hub at the shoulder
   centre with 0.195 m clavicle links, head and neck from the cervicale
   (0.24 m), upper arm 0.277 m, forearm 0.264 m, hand plus grip as today.
   Masses at (1.71 m, 78 kg): pelvis 8.7, middle trunk 12.7, upper trunk
   12.5, head 5.4, upper arm 2.1, forearm 1.26, hand 0.48, thigh 11.0, shank
   3.4, foot 1.07 kg; inertias from the radii of gyration. Tests:
   `test_lower_limb_axes`-style axis checks for every joint, mass total
   within 1 % of the subject mass, each inertia tensor SPD and equal to
   `m (k L)^2` up to the stated tolerance.
3. Recalibrate all 33 marker attachments on the new geometry (planted feet,
   anatomical prior) and adopt posture acceptance numbers: address spine
   bend within 10 deg forward and 10 deg lateral, clavicle links within
   15 deg of horizontal, whole-swing IK RMS at or below the 26 mm
   upper-body-only floor measured today.
4. Simscape: update the model workspace lengths and rigid transforms to the
   same numbers and re-run the R2025b frame parity script so the upper body
   is qualified again; until then the new document is a candidate and every
   receipt says so. MATLAB is not installed on DeskComputer.
5. Engines: MuJoCo, Drake and Pinocchio consume the document through their
   builders, so the same change propagates; rerun the FB-3 parity receipts.
6. Inertia validation after the dynamics gate: peak vertical ground reaction
   between 1.2 and 1.6 body weights for a driver swing, joint torque ranges
   from the literature, and club head speed, all receipted.

## 6. Immediate Step Taken Here

`review_anthropometry.py` and `receipt.json` (this folder); the shared
modules `anthropometry.py` (de Leva table, transcribed, to be checked line by
line against the paper before qualification) and `posture_metrics.py`
(tilt and spine bend split by plane) with unit tests; the Simscape skeleton
plot now draws the club through the exported clubhead frame (the earlier
picture mapped the clubhead through the wrong body frame).

## 7. Experiment: Scaling Alone Does Not Work

`anthropometric_candidate.py` builds an unqualified candidate on the current
topology: upper arms and forearms scaled to de Leva, hub links to half the
biacromial breadth, every segment's mass and inertia from de Leva at 1.71 m
and 78 kg (total 78 kg instead of 108 kg), and all 33 marker attachments
recalibrated (`ground_support/candidate_anthro/receipt.json`, driver flags
`--anthropometric 1.71 78 --recalibrate-upper`).

| Metric                                    | Qualified geometry | Scaled candidate |
| ----------------------------------------- | ------------------ | ---------------- |
| Address marker RMS                        | 3.1 mm             | 48.3 mm          |
| Full-capture IK RMS                       | 28.8 mm            | 48.4 mm          |
| Hips to shoulder centre at address        | 0.595 m            | 0.510 m          |
| Clavicle links below horizontal (L / R)   | 21 / 47 deg        | 42 / 81 deg      |
| Spine bend at address (forward / lateral) | -7 / 21 deg        | 61 / 85 deg      |
| Tracking to 1.0 s, root error max         | 14 mm              | 43 mm            |

Shortening the arms and the hub links without moving the hub down to the
shoulder centre leaves the shoulders unable to reach the shoulder markers,
so the fit folds the neck-base joint and the links even further. The masses
and inertias part of the candidate is sound and is kept; the geometry part
proves the point of section 4: the trunk topology itself (root joint at the
base of the neck, hub 0.14 m above the shoulders) has to change, which is a
new native geometry with its own Simscape parity, not a scaling.

## 8. Next Child Issue (Ready to File)

"Anthropometric native geometry v2": a test-first builder that writes the
upper-body chain from a `subject` block (stature, mass) with explicit
frames: pelvis (hip centres to L5/S1), trunk to the shoulder centre, hub at
the shoulder centre with clavicle links, head from the cervicale, arms and
hands, club and grip closure exactly as today, all masses and inertias from
`anthropometry.py`. Acceptance: joint-axis tests for every joint, mass total
within 1 %, zero-pose lengths equal to the subject table, and on the tour
capture with recalibrated attachments an address spine bend within 10 deg
in both planes, clavicle links within 15 deg of horizontal, whole-swing IK
at or below 26 mm, followed by the Simscape update and R2025b parity.

## 9. AN-1 Iteration 1: Anthropometric Geometry Built and Measured (#10099)

`anthropometric_geometry.build_upper_body(native, stature_m, mass_kg,
trunk_scale, arm_scale, shoulder_scale)` writes the anatomical chain with
the 27 native coordinate names, the native body, joint and frame names, and
the club, hands and closure copied verbatim (`test_anthropometric_geometry.py`,
5 tests: names and closure preserved, zero-pose lengths equal the subject
table, masses and SPD inertias, joint axes in MuJoCo, ranges and seed).
`build_anthropometric_spec.py` assembles `full_body_spec_anthro_v1.json`
with the Rajagopal legs on a fixed pelvis alignment, toe spheres and the
2e5 N/m contact. Findings, each with a receipt:

1. **Static-trial marker placement is required.** Anatomical seed offsets
   were 35 to 140 mm off the golfer's markers (waist markers sit about
   0.14 m above the functional hip centres, not 0.08 m; head and back
   markers 0.1 m higher than seeded), and the swing-wide calibration with a
   40-frame prior toward wrong seeds moved offsets into anatomically
   impossible places while the IK bent the spine to compensate (address
   spine bend 24 deg forward, links 40 to 60 deg down). The driver now
   fits a neutral address (scapulae locked at zero, spine within 10 deg
   forward and 5 deg lateral) and places every marker from the first 24
   address frames (`marker_calibration.static_marker_offsets`, driver flag
   `--static-seeds`); the right shoulder-top marker is absent until frame
   526 and keeps its mirrored seed. Address RMS with those offsets is
   1 mm and the address posture is neutral by construction.
2. **The scapula joint had a null direction.** Its second primitive turned
   about the clavicle link's own axis (a pure spin of the shoulder,
   duplicating the shoulder gimbal); every IK used it to depress the links.
   It is now `Rz` (protraction about the vertical); the coordinate names
   are unchanged.
3. **The shoulder gimbal must not start with hanging arms.** With the upper
   arm along -z at zero pose the middle rotation sits near its 90 deg
   singularity through the whole swing and the trajectory IK diverged
   (75 to 200 mm). The upper arm now points forward at zero pose
   (`ARM_FORWARD`), the elbow range is one-sided (-150 to 5 deg, flexion
   lifts the wrist), shoulder, forearm and wrist angles stay unbounded
   because they wrap during a swing; a hands-forward `address_seed_deg`
   and `coordinate_ranges_deg` travel in the document.
4. **The chain has no neck.** The head turns about 90 deg relative to the
   thorax during the swing; its three markers cannot be fitted by a rigid
   head and were bending the trunk. They now carry weight 0.1 in every IK
   (`solve_pose(marker_weights=...)`, `HEAD_MARKER_WEIGHT`) and are reported
   separately; whole-swing numbers below are given with and without them.
5. **Restarts.** `solve_trajectory(restarts, restart_threshold_m)` re-solves
   a frame above 30 mm from four perturbed starts and keeps the best; the
   downswing and finish went from 70 to 200 mm to 46 to 56 mm.

Subject-specific lengths (`scan_geometry.py`, `scan_geometry_receipt.json`):
offsets from the static trial, no further calibration, decimated-swing IK
RMS of the 30 non-head markers ranks (trunk, arm, shoulder) scale factors on
the de Leva lengths at 1.71 m, 78 kg.

| trunk | arm | shoulder | body RMS | all RMS | address bend fwd / lat | links L / R |
| ----- | --- | -------- | -------- | ------- | ---------------------- | ----------- |
| 1.25  | 1.0 | 1.0      | 24.6 mm  | 39.9 mm | -5.6 / 13.0 deg        | -19 / 19    |
| 1.05  | 1.1 | 1.0      | 24.8 mm  | 39.3 mm | 11.1 / 4.1 deg         | -7 / 7      |
| 1.15  | 1.1 | 1.0      | 28.9 mm  | 43.8 mm | 9.3 / 6.9 deg          | -3 / 3      |
| 1.15  | 1.0 | 1.0      | 30.7 mm  | 49.0 mm | -2.5 / 9.9 deg         | -20 / 20    |
| 1.35  | 1.0 | 1.0      | 39.5 mm  | 54.4 mm | -2.6 / 15.4 deg        | -26 / 26    |

The basin is flat (24.6 to 29 mm across trunk 1.05 to 1.25, arm 1.0 to
1.2, shoulder 0.85 to 1.15; the IK noise is a few mm), so the capture
identifies the trunk as 5 to 25 % longer than the de Leva mean for the
stature and does not resolve the arms and shoulder width further. The
canonical document uses trunk 1.15, arm 1.10, shoulder 1.00: the only
scanned point whose address meets the posture acceptance (bend 9.3 deg
forward, 6.9 deg lateral, links within 3 deg of horizontal).

Against the acceptance in section 8 and #10099: posture met; whole-swing
body-marker RMS 28.9 mm (24.6 at the flat-basin best) against the
qualified geometry's 26.2 mm body-only (28.8 mm with its head at 47 mm);
with the deweighted head included 43.8 mm, which is not a like-for-like
number. The remaining error is not in the lengths: arms 36 to 52 mm and
trunk 28 mm persist across the basin, so the next lever is structural
(thoracic flexibility or a scapulothoracic model, a neck joint for the
head), each a coordinate-set change that must be carried into Simscape.
The de Leva masses and inertias are in the document; nothing here is
qualified until the Simscape model carries the same numbers.

### 9.1 Full Driver Run on the Canonical Document

`ground_support/anthro_v1/receipt.json` (`run_ground_support.py --spec
full_body_spec_anthro_v1.json --skip-hip-calibration --static-seeds --out
anthro_v1`; static placements kept fixed, legs calibrated). Two more
solver changes were needed and are receipted in the same run: the shoulder
gimbals are kept on the Euler branch nearest the previous frame and every
coordinate unwrapped (`continuous_branches`; the same poses, no 2 pi or
branch jumps into the 12 Hz smoother), and the three spins that turn about
one line with a straight elbow (shoulder Z, forearm, wrist Y) carry a
0.1 prior toward the previous frame (`prior_weights`). A restart replaces a
warm start only when it is 3 mm better.

| Metric                                   | Qualified geometry (receipt.json) | Anthropometric (anthro_v1)   |
| ---------------------------------------- | --------------------------------- | ---------------------------- |
| Address marker RMS                       | 3.1 mm                            | 2.1 mm                       |
| Address spine bend forward / lateral     | -7 / 21 deg                       | 0.3 / 6.3 deg                |
| Clavicle links below horizontal L / R    | 21 / 47 deg                       | -5 / 5 deg                   |
| Full-capture IK, all markers             | 28.8 mm (head at full weight)     | 40.0 mm (head at weight 0.1) |
| Full-capture IK, non-head markers        | 26.2 mm                           | 26.2 mm                      |
| Head markers                             | 47 mm                             | 104 mm (deweighted, no neck) |
| Legs / pelvis / club                     | 12, 14 / 44 / 10 mm               | 12, 15 / 19 / 19 mm          |
| Arms L / R                               | 30 / 26 mm                        | 42 / 30 mm                   |
| Reference closure max                    | 1.3 mm                            | 2.7 mm                       |
| Tracking to 1.0 s, root error max        | 14 mm                             | 5 mm                         |
| Tracking to 1.0 s, marker RMS            | 22 mm                             | 25 mm                        |
| Whole-run tracking root RMS / marker RMS | 172 mm / 174 mm                   | 49 mm / 89 mm                |
| Inside support polygon (stance frames)   | 85 %                              | 79 %                         |
| Weight fraction range                    | 0.36 to 1.70 (to 1.0 s)           | 0.0 to 3.2 (to 1.0 s)        |
| Peak joint torque                        | 4375 N m                          | 4453 N m                     |
| Total mass                               | 108 kg                            | 78 kg                        |

The neutral address, the human masses and the like-for-like body-marker
fit are in hand; the downswing tracking is better than on the qualified
geometry but the weight fraction still leaves the ground (GS-4 stays open).
Address renders: `visual_layer/address_{front,side,top}_anthro.png` and
`simscape_skeleton_address_anthro.png` (frame 0 of the IK trajectory).

### 9.2 Neck Joint and Address Arms (User Direction 2026-09-14)

The user asked for a neck (tilt, nod, turn; the Simscape model has none and
that is accepted) and for anatomically plausible arms at address: less
flexion in the left arm and elbow pits facing up or a little toward the
other arm.

- **Neck.** `anthropometric_geometry` adds a head body on a three-axis neck
  at the cervicale (`NeckInputX/Y/Z`, 30 upper-body coordinates, ranges
  45/60/80 deg); head markers attach to the head. With the neck the head
  markers carry full weight again and fit at 28 mm over the swing
  (104 mm deweighted before).
- **Address arms.** The neutral (static-trial) fit bounds the elbows
  (left -25 to 5 deg, right -35 to 5 deg) and every address fit carries a
  weak axis row (`solve_pose(axis_targets=...)`) pulling each upper arm's
  pit axis toward up plus 0.4 times the direction to the other shoulder;
  the address fit gained perturbed restarts (6 per leg seed) because the
  address is a multi-minimum problem for the arms and pelvis.
- **Result** (`ground_support/anthro_v1/receipt.json`): full-capture IK
  30.2 mm over all 34 markers (head 28, trunk 34, pelvis 22, arms 49/39,
  legs 13/16, club 21), address 11.4 mm with spine bend 6.7 deg forward /
  5.8 deg lateral and links within 14 deg, left elbow -25 deg (at its
  bound), right -2 deg, tracking to 1.0 s root error 7 mm, whole-run root
  RMS 53 mm, 83 % of stance frames inside the support polygon. The right
  elbow pit faces up and inward (0.29 up, 0.56 inward); the left pit still
  faces outward and downward (-0.46 up, -0.88 inward).
- **What did not work, all receipted in the session:** a dominant pit
  weight (5.0), humeri seeded turned inward, a second static-trial round
  with the placed markers, and shoulder joints 0.10 m below the cervicale
  each turned the left pit inward at address but put both elbows at the
  hyperextension bound and broke the swing fit (68 to 81 mm; the second
  round is a degenerate fixed point, address 3 mm and swing 81 mm).
  Whatever forces the left pit inward makes the left forearm want to bend
  backwards, so the left humerus-forearm-club chain prefers the pit
  outward: the suspect is the roll of the wrist frame copied verbatim from
  the native document (its forearm axes are not this document's), which
  ties the club's cock axis to the forearm in a way that only a pit-out
  humerus can satisfy. Next: derive the wrist base rotation from this
  document's forearm frame (cock axis perpendicular to the pit plane) with
  a test, then rerun the pit rows at address.

### 9.3 Wrist Axis, Setup Position, Cross-Engine Parity, Centre of Mass (User Direction 2026-09-14)

- **Wrist cock axis.** The wrist base copied from the native document had its
  cock axis (`Rx`) parallel to the elbow-pit direction, so radial deviation
  moved the hand sideways. `WRIST_ROLL` turns the copied base a quarter turn
  about the forearm: the cock axis is now parallel to the elbow axis and a
  positive cock lifts the hand toward the elbow pit, on both sides, with the
  forearm unrotated (`test_joint_axes_are_anatomical_in_mujoco`). The
  hand-to-club relation (child transform and closure weld) is unchanged.
- **Setup constraints** (driver, anthropometric documents only): lead
  scapula may retract up to 20 deg in the static trial (right scapula and
  both elevations locked), left elbow -12 to 5 deg, right elbow -15 to
  -3 deg (5 to 10 deg nominal) in every address fit, pit rows as before.
- **Result** (`ground_support/anthro_v1/receipt.json`): full-capture IK
  30.4 mm over all 34 markers (head 26, arms 46/44,
  legs 16/17, club 16), address 8.3 mm, spine
  bend -7.8 deg forward / 6.7 deg lateral, links
  -2.7/1.6 deg, left elbow -7.5 deg, right elbow -4.7 deg,
  lead scapula retraction 2.5 deg, backswing root error 5 mm,
  whole-run root RMS 72 mm, 75 % of stance frames inside the
  support polygon. Elbow pits at address: right 0.11 / 0.31 (up, inward), left
  -0.47 up, -0.82 inward. The left pit still faces outward; with the corrected wrist,
  dominant pit rows toward up-inward, inward-only and inward-with-lift were
  each tried again and each turned the left pit inward at address while the
  swing fit rose to 47 to 86 mm (traces in the session log). The left
  humerus roll the swing demands is opposite to the pit-inward address; the
  next lever is a two-axis elbow (flexion plus carrying angle) or a check of
  the left forearm/hand roll in the club closure, both test-first.
- **Centre of mass.** `visual_layer.whole_body_com` and `add_com_markers`
  overlay the body-plus-club centre of mass (red) and its ground projection
  (yellow) on the playback GIFs and the address views; the driver records
  `centre_of_mass` at address: 0.912 m above the ground, inside the
  support polygon: True, 0.038 m from the polygon centroid.
- **Same pose in every engine.** `evidence/setup_parity/verify_setup_parity.py`
  evaluates the fitted address (frame 0, by coordinate name) in MuJoCo here
  and in Drake 1.57.0 and Pinocchio 4.1.0 on ControlTower from the
  same document: Pinocchio 9e-16 m / 5e-08 rad, Drake 6.5e-06 m /
  7.3e-06 rad (URDF text precision) over 17 frames; gate 1e-5. The
  translation rule between models is the document's `coordinate_order`:
  a pose is a mapping from coordinate name to value and every adapter's
  `frame_poses` takes that mapping; nothing is indexed by position.
- **Mass and dimensions.** The document carries 79.4 kg (de Leva
  segments at 78 kg plus the native club and hand solids) at 1.71 m with
  trunk 1.15, arm 1.10 and shoulder 1.00 scale factors from the swing;
  an average male golfer is 1.75 to 1.80 m and 80 to 85 kg, so this
  subject is short and light of that mean and the table is a subject
  estimate, not a population default; the stature and mass are explicit
  inputs of `build_anthropometric_spec.py`.

## 10. Clubs, Captures, Torso Visuals, Ranges of Motion, Balance (User Direction 2026-09-14)

- **Clubs.** `club_models.py` holds typical retail values: driver 45.5 in,
  198 g head, 65 g graphite shaft, 50 g grip (313 g); 7-iron 37 in, 268 g
  head, 110 g steel shaft, 50 g grip (428 g); head inertia from radii of
  gyration, shaft and grip as rods. `apply_club` rewrites the club body of a
  document and moves the hands along the shaft so the wrist sits
  `length - 32 mm` from the head, keeping the wrist base and the closure
  weld (tested on both clubs: hand relation unchanged to 1e-9 m). Documents:
  `full_body_spec_anthro_driver.json`, `full_body_spec_anthro_iron7.json`
  (builder `--club`), each carrying a `club` block and total mass
  79.4 / 79.5 kg body plus club.
- **Captures.** The 7-iron file `data/C3D_TA_Iron.c3d` (359 Hz, 657 frames,
  the same 34 tracked labels plus three extras) is registered next to the
  driver in `tour_capture_contract.TOUR_CAPTURES`; the driver takes
  `--capture driver|iron`.
- **Visuals.** Documents carry `visual_hints`: three torso ellipsoids
  (pelvis, abdomen, thorax at 19/17/21 % of stature wide, 13/13/14 % deep),
  45 mm clavicle capsules, a 6.5 mm shaft, an ellipsoid driver head and a box
  iron blade; the visual skeleton and the MuJoCo layer render them; nothing
  in the dynamics changes.
- **Ranges of motion.** `range_of_motion.py` holds `HUMAN_RANGES_DEG` (hinge
  and spine/neck/scapula coordinates in the document's conventions plus the
  Rajagopal leg ranges) and `violations`; documents carry the full table, the
  IK bounds the spine, neck, scapulae and elbows, and every receipt flags
  excursions of the IK reference and of the simulation
  (`range_of_motion_flags`). The wrist cock and the forearm/wrist spins are
  flagged, not bounded: they absorb slop of the hand-club chain, and bounding
  them broke the fit (receipted: driver IK 77 mm with them bounded).
- **Balance.** Address fits pull the body-plus-club centre of mass over the
  centroid of the contact spheres (`ADDRESS_BALANCE_WEIGHT`), moving it from
  38 mm to 0.011 m (driver) and 0.01 m (iron) off the
  centroid, inside the polygon in both.

| Receipt (`ground_support/<run>/receipt.json`) | Driver (`anthro_driver`) | 7-iron (`anthro_iron`) |
| --------------------------------------------- | ------------------------ | ---------------------- |
| Full-capture IK, all markers                  | 24.8 mm                  | 34.1 mm                |
| Address marker RMS                            | 7.1 mm                   | 15.5 mm                |
| Address spine bend forward / lateral          | 8.9 / 5.7 deg            | 9.4 / 8.0 deg          |
| Elbows left / right                           | 0.4 / -5.0 deg           | 0.0 / -2.4 deg         |
| Backswing tracking root error max             | 6 mm                     | 5 mm                   |
| CoM inside the polygon at address             | True                     | True                   |
| Setup parity Drake / Pinocchio                | 5e-06 / 3e-16 m          | 4e-06 / 6e-16 m        |

Open: the left elbow pit still faces outward (section 9.3); the 7-iron
downswing tracking (GS-4) is open as for the driver.

## 11. Anatomical Wrist, Visual Realism, Launcher Tool, Epic #10113 (User Direction 2026-09-14)

Tracked as epic #10113 (children #10103 to #10112).

- **Wrist (MM-2, #10104).** The second wrist primitive is now flexion /
  extension about the palm normal (`Rz`) instead of the native spin about
  the club (redundant with forearm pronation); a neutral-grip ulnar offset
  of 25 deg between the wrist base and the hand makes the coordinates read
  zero at a neutral grip (`GRIP_ULNAR_OFFSET_DEG`, tested). With this wrist
  and the wrists still flagged rather than bounded, both captures fit
  better: driver full-capture IK 24.0 mm (24.8 before), 7-iron 24.1 mm
  (34.1 before); centre of mass 8 mm and 3 mm from the polygon centroid.
  Bounding the wrists and forearms to human ranges in the IK collapses the
  fit (driver 65 mm, 7-iron 68 mm, address 23 mm; receipted in the session)
  and the unbounded wrist cock reads +51 to +66 deg at address and up to
  +159 deg in the swing, which is not a human deviation but the left
  humerus rolled so that the elbow pit faces down and outward (section 9.3,
  MM-5 #10107). The roll of the grip about the club (copied from the native
  hand frame) must be calibrated from the address before the wrist ranges
  can be imposed; until then the wrists and forearms are flagged only
  (`IK_UNBOUNDED`) and every receipt lists the excursions.
- **Visual realism (MM-1, #10103).** Thigh and calf capsules at 72 and 55 mm,
  torso ellipsoids slimmer (17/15.5/18.5 % of stature wide, 11 to 12 %
  deep) and overlapping so the trunk reads as one shape, clavicle capsules
  40 mm, centre-of-mass and frame spheres moved to a hidden render group.
- **Club specs from the club database (MM-3, #10105, part).**
  `club_models.from_database(club_id)` builds a `ClubSpec` from
  `ClubDatabase` (lengths, head, shaft and grip masses, head MOI) so the
  repository has one source of club numbers; head meshes from the
  BunkerShot3D solids and catalogue clubs remain in #10105.
- **Launcher (MM-4, #10106).** `src/tools/motion_matching` (PyQt6 form:
  capture, club, stature, mass, scale factors; runs the document builder
  and the ground-support driver, streams the log, shows the receipt
  summary and playback GIFs) registered as the "Motion Matching" tile
  (category motion_matching, `models.yaml`, launcher manifest, feature
  parity as a web gap on #10106) with unit tests for the command and
  summary layer.

| Receipt (`ground_support/<run>/receipt.json`) | Driver          | 7-iron         |
| --------------------------------------------- | --------------- | -------------- |
| Full-capture IK, all markers                  | 24.0 mm         | 24.1 mm        |
| Address marker RMS                            | 8.7 mm          | 6.9 mm         |
| Address spine bend forward / lateral          | 8.9 / 6.3 deg   | 9.9 / 6.9 deg  |
| Elbows left / right                           | 1.6 / -13.8 deg | 3.3 / -5.3 deg |
| Backswing tracking root error max             | 4 mm            | 5 mm           |
| Whole-run tracking root RMS                   | 25 mm           | 33 mm          |
| Stance frames inside the support polygon      | 84 %            | 89 %           |
| CoM offset from the polygon centroid          | 8 mm            | 3 mm           |

## 12. Elbow Pits From the Markers, Grip Roll, Club Mesh Epic (2026-09-14)

- **What the markers say.** `posture_metrics.elbow_pit_direction` reads the
  direction the forearm folds toward from the shoulder, elbow and wrist
  markers. Driver address: lead pit up -0.35, inward (toward the trail
  side) +0.49, forward -0.38; trail pit up +0.11, inward +0.58; both pits
  turn upward through the backswing (0.9 at the top). The lead arm's marker
  chord angle at address is 48 deg (about 20 to 25 deg true flexion), the
  trail arm 29 deg. The earlier fixed target (up plus a little inward) was
  therefore wrong for the lead arm at address, which is why forcing it broke
  every fit (9.2, 9.3).
- **Data-driven pits.** The driver now pulls each upper arm's pit axis toward
  the marker-derived direction, averaged over the static frames for the
  address fits and per frame through the trajectory
  (`solve_trajectory(axis_targets_per_frame=...)`), with the elbow windows
  widened to what the markers show (left -35 to 5 deg, right -30 to -3).
  Receipts with the wrists flagged: driver IK 26.0 mm, 7-iron
  24.1 mm; address 12.9 / 5.2 mm; backswing root
  5 / 4 mm; pits at address left -0.09 / +0.61,
  right +0.08 / +1.00 (up, inward).
- **Wrist ranges.** Imposing the human wrist and forearm ranges in the IK
  with these pits gave driver 49 mm and 7-iron 42 mm, with forearm pronation
  pinned at +90 deg and the lead cock at its +25 deg radial limit at address
  (arms 63 to 74 mm, club 71 mm). The one free parameter left in the
  hand-club chain is the roll of the hands about the shaft, copied from the
  native hand frames; `GRIP_ROLL_DEG` (builder `--grip-roll`) makes it
  explicit and `scan_grip_roll.py` calibrates it from the driver swing by
  the sum of wrist and forearm excursions beyond the human ranges with the
  wrists unbounded: roll 0 deg is best with a total excursion of 224 deg (lead cock 122 deg beyond its range), +45 deg 277, -45 deg 333, -90 deg 303, +90 deg 451, so no roll brings the wrists within human ranges and the roll is not the lever. The ranges are imposed only once the
  calibrated roll keeps the fit; until then the driver flags the wrists and
  forearms (`IK_UNBOUNDED`) and every receipt lists the excursions. Ranges
  never enter the equations of motion.
- **Club meshes.** Organised as epic #10120 (CM-1 to CM-6) after a survey of
  the three repositories: Tools `rate_of_closure/club` already builds
  parametric heads and writes STL, UpstreamDrift `bunkershot3d/geometry`
  lofts wedges with mass properties, Tools_Private `glass_models` is a
  gmsh vessel mesher for FEA, and Gasification_Model has no mesh code; the
  epic builds one shared club mesh library, a Club Mesh Studio tool, a
  versioned head library and mesh heads in the three engines' visuals.

## 13. Closure Weld Fitted From the Address (MM-2, 2026-09-14)

User direction: the MuJoCo, Drake, Pinocchio and OpenSim full-body models
are the showpiece and may depart from the block-limited Simscape model
where that makes them better; Simscape remains the cross-validation lane.

`closure_fit.fit_closure_placement` rewrites the dual-grip weld's club-side
placement so the weld holds exactly at a fitted address in which the
closure is released, the trail wrist is held at anatomical values
(ulnar -10 deg, flexion 0, pronation 30 deg; `TRAIL_WRIST_ADDRESS_DEG`)
and the lead wrist is bounded to human ranges, with the elbow pits from
the markers (driver flag `--fit-closure`, needs `--static-seeds`). The weld
stays a weld, so the engines and their parity are untouched; only the
numbers in `closure.placement_b` change and the document records them
under `closure_fit`.

Driver capture, closure fitted, wrists and forearms bounded in the IK:
open-chain address fit 43.7 mm (hands held on the grip point, weld orientation free, trail wrist locked at ulnar -10 / flexion 0 / pronation 30 deg, lead wrist bounded), weld turned 65.6 deg and moved 1.2 mm, address with the fitted weld and bounded wrists 41.1 mm, full-capture IK 67.8 mm (`anthro_driver_fit/receipt.json`); the lead cock sits at its +25 deg radial limit at address.

Verdict: fitting the weld from the address does not make the human wrist ranges reachable either; with the pits fixed by the markers the lead wrist still needs +50 to +65 deg of cock at address and 110 deg of travel through the swing (the unbounded receipts), about twice a human radial-ulnar range. The C3D carries no hand markers, so the wrist axes are observed only through the club: the remaining hypothesis is that the cock coordinate is absorbing motion that belongs to flexion/extension and pronation because the wrist base axes are still rolled relative to the golfer's hand, and the test for it is to fit the hand frame from the club orientation at three swing phases (address, top, impact) and solve the constant hand-to-club rotation that minimises the wrist excursions jointly, rather than the shaft roll alone. The driver keeps the wrists flagged (driver 26.0 mm, 7-iron 24.1 mm) and `--fit-closure` stays available as an experiment with its receipt.

## 14. Hand-to-Club Rotation Fitted From the Matches, Wrists Bounded (MM-2, 2026-09-14)

The capture has no hand markers, so the golfer's hand frame is observed only
through the club. The wrist chain of the anthropometric documents is
`Rz(pronation) @ base @ Rx(cock) @ Rz(flexion)` from the proximal forearm to
the hand body, and one constant in it was never observed: the rotation of the
hand body on the wrist follower frame (the wrist joint's
`child_to_follower`, copied from the native document with the quarter-turn
and the neutral-grip turn applied). When that constant is wrong the cock
coordinate absorbs motion that belongs to flexion and pronation, which is what
sections 12 and 13 measured: 110 deg of lead cock travel and pronation pinned
at 90 deg, with neither the grip roll (one axis) nor the closure weld able to
fix it.

`grip_fit.py` fits the constant. From an unbounded match it reconstructs the
forearm-to-hand rotation per frame (MuJoCo forward kinematics of the run's
document), decomposes it exactly into the three coordinates
(`wrist_angles`; the chain is an intrinsic Y-X-Z Euler sequence) and searches
the follower-side rotation under which the excursions beyond the human ranges
are smallest over every frame of every run given (`fit_grip_rotation`:
15 deg grid over extrinsic x-y-z angles, Nelder-Mead refinement, root mean
square excess as the cost). `evidence/anthropometry/fit_grip_rotation.py`
runs it over the driver and 7-iron matches together
(`fit_grip_rotation_receipt.json`, inputs: identity-rotation documents
matched with `--free-wrists`, driver IK 26.0 mm, 7-iron 24.1 mm):

| Hand  | Rotation (x, y, z deg) | RMS excess before -> after | Max excess after (pronation, cock, flexion) |
| ----- | ---------------------- | -------------------------- | ------------------------------------------- |
| lead  | -89.7, 46.7, 0.0       | 40.6 -> 3.9 deg            | 2.7, 16.3, 0.2 deg                          |
| trail | -34.2, 46.0, 62.0      | 17.7 -> 1.2 deg            | 0.0, 19.6, 0.0 deg                          |

One rotation per hand serves both captures (driver lead 3.9 / trail 1.5 deg
RMS, 7-iron 3.9 / 0.8 deg). The residual is the lead cock reaching 41 deg
radial at the top against the 25 deg human limit; everything else is inside.
The fitted values are the builder defaults (`GRIP_ROTATION_DEG`, flags
`--lead-grip-rotation` / `--trail-grip-rotation`), recorded in
`subject.grip_rotation_deg`, and the driver bounds the wrists and forearms
to the human ranges by default for any document that carries a fitted
rotation (`--free-wrists` releases them again, `--bound-wrists` forces them
for an unfitted document). Nothing enters the equations of motion: the wrist
joints keep their two free coordinates; only the hand's resting orientation
on them changed.

Receipts with the wrists bounded (`ground_support/anthro_driver`,
`anthro_iron`): driver address 5.1 mm, full-capture IK 27.3 mm (was 26.0 mm
unbounded, 67.8 mm with the closure fit); 7-iron address 4.4 mm, IK 28.6 mm
(was 24.1 mm unbounded); no wrist or forearm flags on either; backswing
dynamics root 6.2 / 6.0 mm, markers 18.5 / 21.6 mm; support-polygon fraction
0.82 / 0.88. Wrist angles at address, driver: lead pronation 30, cock -31,
flexion 16 deg; trail 9, -30, 11 deg. Refitting the rotation on the bounded
runs returns the identity (`fit_grip_rotation_residual_receipt.json`).
Cross-engine setup parity at the new addresses: Drake 5.9e-6 / 5.4e-6 m,
Pinocchio 7e-16 / 8e-16 m (`setup_parity/receipt_anthro_*.json`).

Caveats. An unbounded match can sit on the far Euler branch of the wrist
chain (cock beyond 90 deg with pronation and flexion turned by 180 deg), a
geometrically identical pose that the range flags report as a 155 deg
excursion; the fit compares rotation matrices, not angles, and is immune,
but unbounded receipts of fitted documents must be read with that in mind.
The 7-iron's left arm rose from 40 to 48 mm under the bounds, the price of
the 25 deg radial limit; MM-2 leaves the human ranges as they are rather
than widening them to the golfer.

## 15. Downswing Dynamics: Compliant Sole, Tracked Reference, Zero-Moment Point (MM-7, 2026-09-14)

GS-4 left the forward-dynamics replay diverging after the top: with the
driver match of section 14 the root error was 21 mm at 1.2 s, 40 at 1.4,
164 at 1.75 s, the body airborne twice (weight fraction 0 at 1.31 and from
1.43 s) and the peak joint torque 2523 N m. `evidence/ground_support/
downswing_experiment.py` replays only the dynamics stage of a finished run
with one setting changed at a time and writes `<run>/downswing_<name>.json`;
`--free-wrists` runs are not needed since the reference is the run's own.

### What Was Ruled Out

- **Reference jerk.** The consistency re-solve steps 20 mm in one frame
  where markers drop out at impact (root acceleration 492 m/s^2, 218 000
  deg/s^2 in the joints). A second zero-phase low-pass of the tracked
  reference at 12 or 8 Hz removes the impact torque spike (2523 to 811 N m)
  but not the divergence (212 / 188 mm). Dropping the acceleration
  feedforward: 156 mm.
- **Friction creep.** The regularised Coulomb law (transition velocity
  50 mm/s) was suspected because the pelvis yaw lags the reference by 11 to
  13 deg through the transition. Transition velocities of 10, 5 and 2 mm/s
  (time steps down to 0.1 ms) leave the timeline unchanged (7/13/24/24/38
  mm at 1.0/1.1/1.2/1.3/1.4 s); friction coefficients 1.5, 2.0 and 3.0
  (spiked shoes) change nothing before impact either. The feet were not
  creeping: the lead foot's forefoot and toe spheres sit 1 to 2 mm above the
  plane from 1.0 s (the sim stands on its heels), so a root pitch error
  below one degree unloads a foot, and an unloaded foot slides freely
  (lead toe 35 mm by 1.2 s, 121 mm by 1.35 s) whatever the coefficient.
- **Root regulation through the legs.** Gains (100, 20) with legs at
  10 rad/s: 147 mm; (400, 40) and (900, 60): 1.0 m and 0.9 m with
  weight fractions of 100 to 200 (the planted-coupling pseudo-inverse
  amplifies once a foot is off the plane).

### What the Reference Demands

`full_body_simulation.reference_zmp` computes, from the whole-body momentum
rates of the tracked reference, the ground reaction and the zero-moment
point this model would need, against the hull of the spheres on the plane
at each frame. For the driver reference: vertical reaction 0.61 BW with
0.69 BW horizontal at 1.15 s (ratio 1.12, above any shoe), 1.69 BW with
0.82 BW horizontal at 1.3 s; the zero-moment point leaves the support
polygon on 77 % of the frames between 1.0 and 1.5 s, by
tens of centimetres. No controller can realise those frames on unilateral
feet. The tour-average capture is a composite of many swings, and averaged
kinematics are not dynamically consistent for any one body; the model's
de Leva mass distribution differs from the golfers' as well. The receipts
now carry this diagnostic (`dynamics.reference_zmp`).

### What Works: A Compliant Sole and a Band-Limited Tracked Reference

Contact stiffness (N/m), dissipation (s/m) and the tracked reference's
low-pass were varied on the driver run (columns: root error at 1.2 / 1.4 s
and at the end, maximum, weight fraction range, peak torque):

| name               | k (kN/m) | d   | mu  | v_t   | low-pass Hz | ff  | legs | root reg      | 1.2 s | 1.4 s | end  | max  | wf min | wf max | torque N m |
| ------------------ | -------- | --- | --- | ----- | ----------- | --- | ---- | ------------- | ----- | ----- | ---- | ---- | ------ | ------ | ---------- |
| base               | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 40    | 164  | 178  | 0.0    | 5.59   | 2523       |
| ff0                | 200      | 1   | 0.9 | 0.05  | none        | 0.0 | 30   | off           | 27    | 29    | 144  | 156  | 0.0    | 5.36   | 1541       |
| k100               | 100      | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 23    | 32    | 124  | 128  | 0.0    | 6.15   | 2777       |
| k25_d2             | 25       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 44    | 50    | 155  | 169  | 0.11   | 4.99   | 2615       |
| k40                | 40       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 11    | 39    | 90   | 93   | 0.0    | 5.43   | 2701       |
| k50                | 50       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 42    | 56   | 57   | 0.11   | 5.56   | 2684       |
| k50_d05            | 50       | 0.5 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 47    | 81   | 82   | 0.0    | 5.21   | 2642       |
| k50_d2             | 50       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 36    | 38   | 45   | 0.28   | 6.21   | 2693       |
| k50_d2_lp12        | 50       | 2.0 | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 38    | 33   | 43   | 0.38   | 2.37   | 476        |
| k50_d2_lp12_dt05   | 50       | 2.0 | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 38    | 33   | 43   | 0.38   | 2.37   | 473        |
| k50_d3             | 50       | 3.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 34    | 37   | 48   | 0.0    | 7.01   | 2766       |
| k50_dt05           | 50       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 42    | 56   | 57   | 0.1    | 5.56   | 2375       |
| k50_lp12           | 50       | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 41    | 51   | 53   | 0.08   | 2.73   | 508        |
| k50_mu15           | 50       | 1   | 1.5 | 0.05  | none        | 1.0 | 30   | off           | 23    | 47    | 85   | 87   | 0.04   | 6.15   | 3395       |
| k60                | 60       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 26    | 38    | 46   | 52   | 0.08   | 5.48   | 2649       |
| k60_d2             | 60       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 26    | 36    | 43   | 54   | 0.12   | 6.23   | 2685       |
| lp12               | 200      | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 22    | 26    | 183  | 212  | 0.0    | 2.98   | 811        |
| lp12_ff0           | 200      | 1   | 0.9 | 0.05  | 12.0        | 0.0 | 30   | off           | 25    | 37    | 169  | 195  | 0.0    | 3.1    | 828        |
| lp12_legs10        | 200      | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 10.0 | off           | 20    | 73    | 47   | 78   | 0.0    | 4.6    | 758        |
| lp12_vt010_dt05    | 200      | 1   | 0.9 | 0.01  | 12.0        | 1.0 | 30   | off           | 24    | 24    | 176  | 207  | 0.0    | 2.91   | 729        |
| lp8                | 200      | 1   | 0.9 | 0.05  | 8.0         | 1.0 | 30   | off           | 22    | 51    | 151  | 188  | 0.0    | 3.08   | 783        |
| mu15               | 200      | 1   | 1.5 | 0.05  | none        | 1.0 | 30   | off           | 23    | 22    | 135  | 154  | 0.0    | 6.05   | 3393       |
| mu15_lp12          | 200      | 1   | 1.5 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 17    | 183  | 217  | 0.0    | 3.33   | 1059       |
| mu20               | 200      | 1   | 2.0 | 0.05  | none        | 1.0 | 30   | off           | 24    | 15    | 126  | 145  | 0.0    | 5.59   | 3887       |
| mu30               | 200      | 1   | 3.0 | 0.05  | none        | 1.0 | 30   | off           | 31    | 18    | 107  | 129  | 0.0    | 5.12   | 4751       |
| rootreg_legs10     | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (100.0, 20.0) | 24    | 33    | 138  | 147  | 0.0    | 9.38   | 3691       |
| rr400_legs10       | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (400.0, 40.0) | 48    | 575   | 981  | 1032 | 0.0    | 196.77 | 71607      |
| rr400_legs10_nobal | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (400.0, 40.0) | 176   | 589   | 1101 | 1859 | 0.0    | 131.43 | 26948      |
| rr900_legs8_mu15   | 200      | 1   | 1.5 | 0.05  | none        | 1.0 | 8.0  | (900.0, 60.0) | 66    | 423   | 585  | 908  | 0.0    | 112.76 | 43585      |
| vt002_dt01         | 200      | 1   | 0.9 | 0.002 | none        | 1.0 | 30   | off           | 24    | 37    | 84   | 84   | 0.0    | 5.28   | 2653       |
| vt005_dt025        | 200      | 1   | 0.9 | 0.005 | none        | 1.0 | 30   | off           | 24    | 38    | 148  | 157  | 0.0    | 5.26   | 2565       |
| vt010_dt05         | 200      | 1   | 0.9 | 0.01  | none        | 1.0 | 30   | off           | 23    | 39    | 151  | 159  | 0.0    | 5.16   | 2329       |

Softening the sole from 200 to 50 kN/m (a 78 kg golfer sinks 16 mm at
rest instead of 4; shoe midsole plus turf) lets the body give the few
centimetres the composite reference demands instead of unloading a foot:
root error 57 mm maximum, never airborne. Dissipation 2 s/m damps the
impact rebound (45 mm), and the 12 Hz low-pass of the tracked reference
takes the peak torque from 2693 to 476 N m without costing tracking
(43 mm). Halving the time step reproduces the same numbers, so the result
is not an integration artefact. 25 kN/m sinks too far (169 mm), 100 kN/m
still hops (128 mm).

Adopted: `build_anthropometric_spec.py` writes 5.0e4 N/m and 2 s/m into the
anthropometric documents (`CONTACT_STIFFNESS_N_M`,
`CONTACT_DISSIPATION_S_M`; the qualified native v2 keeps 2.0e5, and the
driver's `add_toe_spheres` raises only native documents), and the driver
tracks the re-solved reference through one more 12 Hz zero-phase low-pass
(`TRACKING_CUTOFF_HZ`). Full-pipeline receipts: driver dynamics root error
38 mm, whole-run marker RMS 74.6 mm (was 77.3),
weight fraction 0.38 to 2.37, peak torque 476 N m; 7-iron root error
31 (to 1.5 s; 133 at 1.75 s in the follow-through) mm, marker RMS 112.3 mm, weight fraction 0.29 to 3.65, peak
torque 835 N m. The softer sole costs a little backswing sway
(driver root error to 1.0 s 20 mm, was 7 mm). Setup parity is
untouched (the contact law is in the shared document): Drake 5.9e-06 m
driver, 5.4e-06 m 7-iron.

### Cart-Table Dynamics Filter (MM-7B Experiment, Not Adopted)

`dynamics_filter.py` (model-free: `project_inside`, `cart_table_shift`)
and the driver's `--zmp-filter` implement the Kagami-style correction:
project the reference zero-moment point into the per-frame support polygon
shrunk by 2 cm, find the smallest smooth centre-of-mass shift whose
cart-table effect closes the gap, re-solve the IK against the markers with
that centre-of-mass path as per-frame rows (`solve_pose(com_target=...)`,
`solve_trajectory(com_targets_per_frame=...)`), low-pass, repeat. On the
driver (`ground_support/anthro_driver_zmp/receipt.json`) three passes take
the zero-moment point outside the polygon from 77 % to 62 %, 57 % and 57 %
of the downswing frames while the centre-of-mass shift grows to 112, 160
and 278 mm, the reference's marker error rises from 27 to 38, 55 and 98 mm,
and the replay is worse than without the filter (root error 129 mm at
1.4 s, 497 mm at 1.75 s; 7-iron 337 mm at 1.4 s, `anthro_iron_zmp/receipt.json`). Verdict: the single-mass cart-table relation
cannot describe this reference. Its zero-moment-point excursions are driven
by the angular momentum of the arms and club (the term the cart table
ignores), so the centre of mass would have to move tens of centimetres to
compensate, which the markers forbid. A consistent motion needs the
upper-body swing itself to change, not the centre-of-mass path. The flag
stays available as an experiment with its receipt.

### What Remains, Next Step MM-7B

The reference is still not dynamically consistent for this model (zero-moment
point outside on 77 % of the downswing frames), so the replay follows it to
within 3 to 4 cm at the pelvis rather than exactly. The next step is a
whole-body formulation rather than a centre-of-mass filter: the FB-5
contact-aware shooting fit (optimise the tracked reference of every
coordinate under the replay's own contact dynamics so that the marker error
of the replayed motion is minimised, with the zero-moment point as a
penalty), or an equivalent trajectory optimisation over the angular
momentum of the arm-club system. `reference_zmp` and the downswing
experiment harness provide the metrics; the 38 mm root error of the
compliant-sole replay is the baseline to beat.

## 16. Contact-Aware Shooting Fit (FB-5, MM-7B, 2026-09-14)

The replay is the plant. `run_ground_support.py --shooting-fit N`
(`shooting_fit`, `replay`) iterates: replay the tracked reference with the
computed-torque controller on the compliant sole, measure the pelvis drift
of the replay from the reference pelvis path, move the pinned pelvis
command against that drift (iterative learning,
`command <- command - 0.7 (replayed - reference)` on the two horizontal
slides and the three root rotations; the vertical slide stays free so the
planted feet set it), re-solve the joints against the markers with the
command pinned (`solve_trajectory(locked_per_frame=...)`, consistency prior,
stance planted, human bounds), low-pass at 12 Hz, replay again. The
objective is the marker error of the replayed motion; the best iteration's
reference is what the dynamics stage then tracks, and every iteration is in
the receipt (`dynamics.shooting_fit`).

A plain fixed point (pin the pelvis where the replay drifted and re-solve)
was tried first and is anti-corrective: driver replays 74.6, 73.9, 90, 199,
266 mm over four iterations, 7-iron 112, 110, 151, 256, 335 mm. The
iterative-learning form:

| Capture            | Iteration | Replay markers | Root max | Root at 1.4 s | Weight fraction min | ZMP outside 1.0 to 1.5 s |
| ------------------ | --------- | -------------- | -------- | ------------- | ------------------- | ------------------------ |
| driver (gain 0.7)  | 0         | 74.6 mm        | 43 mm    | 38 mm         | 0.38                | 77 %                     |
| driver (gain 0.7)  | 1         | 103.9 mm       | 66 mm    | 39 mm         | 0.35                | 83 %                     |
| driver (gain 0.7)  | 2         | 126.6 mm       | 71 mm    | 62 mm         | 0.32                | 82 %                     |
| driver (gain 0.7)  | 3         | 130.3 mm       | 91 mm    | 72 mm         | 0.29                | 86 %                     |
| driver (gain 0.7)  | 4         | 134.8 mm       | 120 mm   | 68 mm         | 0.31                | 81 %                     |
| 7-iron (gain 0.7)  | 0         | 112.3 mm       | 165 mm   | 31 mm         | 0.29                | 76 %                     |
| 7-iron (gain 0.7)  | 1         | 128.8 mm       | 112 mm   | 27 mm         | 0.36                | 79 %                     |
| 7-iron (gain 0.7)  | 2         | 146.1 mm       | 138 mm   | 36 mm         | 0.32                | 81 %                     |
| 7-iron (gain 0.7)  | 3         | 174.5 mm       | 152 mm   | 38 mm         | 0.20                | 84 %                     |
| 7-iron (gain 0.7)  | 4         | 192.9 mm       | 136 mm   | 31 mm         | 0.06                | 87 %                     |
| driver (gain 0.25) | 0         | 74.6 mm        | 43 mm    | 38 mm         | 0.38                | 77 %                     |
| driver (gain 0.25) | 1         | 81.3 mm        | 43 mm    | 36 mm         | 0.37                | 81 %                     |
| driver (gain 0.25) | 2         | 87.4 mm        | 47 mm    | 36 mm         | 0.36                | 81 %                     |
| driver (gain 0.25) | 3         | 92.6 mm        | 48 mm    | 38 mm         | 0.36                | 80 %                     |
| driver (gain 0.25) | 4         | 96.9 mm        | 51 mm    | 40 mm         | 0.35                | 78 %                     |
| driver (gain 0.25) | 5         | 100.6 mm       | 54 mm    | 43 mm         | 0.35                | 79 %                     |
| driver (gain 0.25) | 6         | 103.9 mm       | 56 mm    | 45 mm         | 0.35                | 80 %                     |

Verdict: every gain diverges monotonically from iteration 0 on both captures (gain 0.7: driver 74.6 to 134.8 mm, 7-iron 112.3 to 192.9 mm; gain 0.25: driver 74.6 to 103.9 mm over six passes), so the best reference is always the unmodified one and the dynamics stage keeps it. The attribution receipted with the soft-sole replay explains why: substituting only the replayed pelvis rotation into the reference already gives 72.7 of the 74.6 mm replay error (translation alone 35.3 mm, the joints with the reference pelvis 34.6 mm); the pelvis yaw lags the reference by 15 deg through the downswing and 34 deg in the follow-through, and that lag is what the ground can supply in yaw moment for this reference, not something a displaced pelvis command can pre-compensate (a command displaced against the drift costs marker fit in the reference immediately and the replay does not recover it). Friction on the compliant sole is not the lever either (transition velocity 10 mm/s: identical; coefficient 1.5: yaw 14.9 to 9.1 deg at 1.3 s but 78.6 mm overall). The shooting fit therefore needs the whole-body swing as the decision variable with the contact dynamics as constraints: a trajectory optimisation with a differentiable simulator (the JaxSim differentiable backend, epic #6647, or MuJoCo MJX) over the arm-club angular momentum and pelvis rotation, where the composite reference is a soft target and the ground yaw-moment capacity is respected. The iterative-learning harness, `reference_zmp` and the downswing experiment harness stay as the evaluation tools; `--shooting-fit` remains available with its receipts (`anthro_driver_shoot`, `anthro_iron_shoot`, `anthro_driver_shoot_g025`).

## 17. Differentiable Trajectory Optimisation With MuJoCo MJX (FB-5, MM-7B, 2026-09-14)

The whole tracked reference is now optimised through the contact dynamics
with gradients. `export_mjx_package.py` writes a finished run as a
self-contained package (the MJCF with its grip site weld stiffened, the
tracked reference, the capture markers and validity, the marker attachments
in MJCF body frames, spheres, contact law, ground, controller gains);
`mjx_trajectory_optimisation.py` runs in an isolated environment (JAX
0.11, MuJoCo 3.13 with MJX; recipe in HANDOFF) and is a JAX port of the
replay: the shared Hunt-Crossley plus regularised-Coulomb sphere law applied
as world wrenches at the calcanei, the dual-grip weld as a stiff
spring-damper wrench (MJX's constraint solver is an iterative loop JAX
cannot reverse-differentiate, so every equality is removed from the model),
the root-free computed-torque controller, semi-implicit Euler at six
substeps per capture frame, a `lax.scan` over frames with checkpointing that
emits the replayed marker positions. Decision variables: knot values every
40 ms of a correction added to the 38 actuated coordinates (47 knots,
1786 parameters); cost: mean squared replayed marker error over valid
markers up to 1.65 s (the soft-weld plant departs from the rigid-weld one in
the last follow-through) plus a small regulariser; Adam on the gradient
through the whole rollout (about 52 s per iteration on CPU after a
697 s compile).

Port and pitfalls (all receipted in `anthro_driver/mjx_*`): the
near-massless hand standoff explodes on a spring weld without a 5e-3
kg m^2 armature floor on every dof; the previous substep's applied torque
sits in `qfrc_smooth` and must be cleared before the controller's forward
pass; a vector norm at zero tangential speed and the NaN of invalid markers
under a masked `where` both poison the gradient. With those fixed the MJX
replay of the unmodified reference is 41.0 mm to 1.65 s against the
shared simulator's 74.6 mm over the whole swing, with the same downswing
profile. A learning rate of 2e-3 rad per knot makes the objective wander
(41 to 107 mm between iterations; `mjx_opt_lr2e-3_log.txt`); 5e-4 descends
monotonically.

| Iteration (lr 5e-4) | MJX replay markers to 1.65 s |
| ------------------- | ---------------------------- | --------------------- | ------------------------ | -------------------------- |
| Iteration           | MJX replay to 1.45 s         | shared plant to 1.5 s | shared plant whole swing | pelvis yaw lag 1.3 / 1.4 s |
| ---                 | ---                          | ---                   | ---                      | ---                        |
| 0 (unmodified)      | 41.0 mm                      | 56.1 mm               | 74.6 mm                  | 14.9 / 12.3 deg            |
| 2                   | 34.2 mm                      | 48.0 mm               | 78.1 mm                  | 12.2 / 8.2 deg             |
| 4                   | 32.0 mm                      | 42.4 mm               | 79.6 mm                  | 10.6 / 5.4 deg             |
| 6                   | 30.5 mm                      | 42.8 mm               | 83.3 mm                  | 10.3 / 4.3 deg             |
| 8                   | 29.5 mm                      | 44.6 mm               | 195.5 mm                 | 10.4 / 4.0 deg             |
| 10                  | 28.8 mm                      | 43.7 mm               | 192.4 mm                 | 10.6 / 3.8 deg             |
| 12                  | 28.4 mm                      | 42.1 mm               | 190.0 mm                 | 10.3 / 3.4 deg             |
| 14                  | 27.9 mm                      | 41.6 mm               | 184.9 mm                 | 9.9 / 3.1 deg              |
| 16                  | 27.5 mm                      | 40.2 mm               | 182.3 mm                 | 9.4 / 2.6 deg              |

Validation in the shared-law simulator (`downswing_experiment.py
--reference`, the real plant with the rigid weld and RK4): every snapshot replayed in the shared-law simulator (`downswing_mjx_h145_iter*.json`), columns above.

Verdict: the differentiable optimisation works and transfers within its cost window. The replayed marker error to 1.5 s falls from 56.1 to 40.2 mm (28 %) and the pelvis yaw lag at 1.4 s from 12.3 to 2.6 deg with corrections below half a degree at the knots, which is the first method in this lane to move the pelvis yaw. It does not yet transfer to the whole swing: the follow-through after 1.5 s is uncosted (the soft-weld MJX plant departs from the rigid-weld one there, 187 and 322 mm at 1.7 and 1.8 s even at 12 substeps) and from iteration 8 the shared plant collapses after impact (whole-swing 182 to 195 mm), while iterations 4 to 6 keep it near the baseline (79.6 and 83.3 mm against 74.6). The transferable choice today is therefore iteration 4 to 6 for a whole-swing replay or iteration 16 for the swing to 1.5 s. Next: give the MJX plant a faithful follow-through. A five-times stiffer, more damped weld leaves the late window numerically identical (`mjx_diag_w1e6.txt`), so the weld is not the cause; the divergence after 1.6 s (small pelvis error, large marker error) has to be located joint by joint against the shared plant's late replay before a full-horizon solve, then the 7-iron follows. A joint-by-joint comparison of the two plants' replays of the unmodified reference (`mjx_diagnose.npz` against `downswing_soft_base.npz`) shows every joint tracked equally well; the difference is the pelvis rotation: MJX lags 7.6 to 8.7 deg through the downswing where the shared plant lags 14.9 to 15.1, and 50.6 deg in the follow-through (1.65 to 1.82 s) where the shared plant lags 34.4. The two plants differ in their ground yaw-moment response (contact evaluated at the substep start with semi-implicit Euler against RK4), which is where the next agent should look. The receipts (`mjx_optimisation_receipt.json`, `mjx_opt_*_log.txt`, the snapshot replays) and the two scripts are the harness for it.
