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
