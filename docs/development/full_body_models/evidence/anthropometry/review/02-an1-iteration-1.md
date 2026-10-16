# Anthropometry Review AN-1 Iteration 1

Section 9 of the [anthropometry review](../REVIEW.md). Moved verbatim.

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
