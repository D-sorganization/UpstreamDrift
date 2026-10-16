# Anthropometry Review of Clubs, Wrist, Closure, and Grip

Sections 10 to 14 of the [anthropometry review](../REVIEW.md). Moved verbatim.

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
