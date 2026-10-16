# Handoff Archive: AN-1 Anthropometric Geometry

Anthropometry review pointer and AN-1 iteration log (2026-09-14). Moved verbatim from the [full-body handoff](../HANDOFF.md).

## Anthropometry and Posture Review (User Direction 2026-09-14)

The user asked why the torso looks arched at address and how the body
dimensions were determined. Answer in
[evidence/anthropometry/REVIEW.md](../evidence/anthropometry/REVIEW.md) with
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

Findings and receipts in [evidence/anthropometry/REVIEW.md](../evidence/anthropometry/REVIEW.md)
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

### Marker-Driven Elbow Pits, Grip Roll, Club Mesh Epic (2026-09-14)

Elbow pits follow the markers per frame (`posture_metrics.elbow_pit_direction`,
`Lane.pit_targets_per_frame`); `GRIP_ROLL_DEG` / `--grip-roll` exposes the
hand roll about the shaft and `evidence/anthropometry/scan_grip_roll.py`
calibrates it (roll 0 deg is best with a total excursion of 224 deg (lead cock 122 deg beyond its range), +45 deg 277, -45 deg 333, -90 deg 303, +90 deg 451, so no roll brings the wrists within human ranges and the roll is not the lever). Receipts (wrists flagged): driver IK
26.0 mm, 7-iron 24.1 mm. Bounding the wrists still costs the
fit until the roll is calibrated (REVIEW.md 12, MM-2 #10104). Club meshes
are epic #10120.

### Closure Fitted From the Address, Showpiece Direction (2026-09-14)

User direction: MuJoCo, Drake, Pinocchio and OpenSim full-body models are
the showpiece and may improve beyond the block-limited Simscape model;
Simscape stays the cross-validation lane. `closure_fit.py` and the driver's
`--fit-closure` fit the two-hand weld from the address with anatomical
wrists. Driver result: open-chain address fit 43.7 mm (hands held on the grip point, weld orientation free, trail wrist locked at ulnar -10 / flexion 0 / pronation 30 deg, lead wrist bounded), weld turned 65.6 deg and moved 1.2 mm, address with the fitted weld and bounded wrists 41.1 mm, full-capture IK 67.8 mm (`anthro_driver_fit/receipt.json`); the lead cock sits at its +25 deg radial limit at address. Verdict: fitting the weld from the address does not make the human wrist ranges reachable either; with the pits fixed by the markers the lead wrist still needs +50 to +65 deg of cock at address and 110 deg of travel through the swing (the unbounded receipts), about twice a human radial-ulnar range. The C3D carries no hand markers, so the wrist axes are observed only through the club: the remaining hypothesis is that the cock coordinate is absorbing motion that belongs to flexion/extension and pronation because the wrist base axes are still rolled relative to the golfer's hand, and the test for it is to fit the hand frame from the club orientation at three swing phases (address, top, impact) and solve the constant hand-to-club rotation that minimises the wrist excursions jointly, rather than the shaft roll alone. The driver keeps the wrists flagged (driver 26.0 mm, 7-iron 24.1 mm) and `--fit-closure` stays available as an experiment with its receipt. Details REVIEW.md 13.

### Hand-to-Club Rotation Fitted, Wrists Bounded (MM-2, 2026-09-14)

`src/shared/python/motion_matching/grip_fit.py` fits the constant rotation
of each hand on its wrist from the matched swings (the one unobserved
constant of the hand-club chain; the C3D has no hand markers). Fitted over
the driver and 7-iron together: lead (-89.7, 46.7, 0.0) deg, trail
(-34.2, 46.0, 62.0) deg, wrist excursions beyond the human ranges 40.6 ->
3.9 deg RMS (lead) and 17.7 -> 1.2 deg (trail). These are the builder
defaults (`GRIP_ROTATION_DEG`) and the driver now bounds the wrists and
forearms by default for fitted documents. Receipts: driver address 5.1 mm,
IK 27.3 mm; 7-iron 4.4 / 28.6 mm; no wrist flags; parity Drake 6e-6 m,
Pinocchio 1e-15 m. Details REVIEW.md 14; derivation
`evidence/anthropometry/fit_grip_rotation_receipt.json`.

### Downswing Dynamics: Compliant Sole, Tracked Reference, ZMP (MM-7, 2026-09-14)

`evidence/ground_support/downswing_experiment.py` replays the dynamics
stage of a finished run with one setting changed (32 receipts under
`anthro_driver/downswing_*.json`). Reference jerk, friction creep, friction
coefficients and root regulation were ruled out; the feet were being
unloaded by sub-degree root tilt on a 200 kN/m sole. Adopted: anthropometric
documents carry a 50 kN/m, 2 s/m sole (`build_anthropometric_spec.py`) and
the driver tracks the re-solved reference through a 12 Hz low-pass
(`TRACKING_CUTOFF_HZ`). Driver: root error 38 mm through impact
(was 178), never airborne, peak torque 476 N m (was 2523), whole-run
marker RMS 74.6 mm; 7-iron root 31 (to 1.5 s; 133 at 1.75 s in the follow-through) mm, marker RMS
112.3 mm. `full_body_simulation.reference_zmp` (in every receipt as
`dynamics.reference_zmp`) shows the composite reference's zero-moment point
outside the feet on 77 % of the downswing frames: the remaining
3 to 4 cm is the reference, not the controller. Details REVIEW.md 15.

### Cart-Table Dynamics Filter Tried (MM-7B, Not Adopted, 2026-09-14)

`src/shared/python/motion_matching/dynamics_filter.py`, centre-of-mass rows
in the marker solver (`com_target`, `com_targets_per_frame`) and the driver's
`--zmp-filter` implement the cart-table zero-moment-point correction. On the
driver it moves the centre of mass 112 to 278 mm, raises the reference's
marker error from 27 to 98 mm and makes the replay worse (497 mm); the
excursions come from the arm-club angular momentum the cart table ignores.
Kept as an experiment (`anthro_driver_zmp/receipt.json`); MM-7b now means a
whole-body shooting fit (FB-5). Details REVIEW.md 15.

### Contact-Aware Shooting Fit (FB-5, MM-7B, 2026-09-14)

`run_ground_support.py --shooting-fit N` replays the tracked reference,
moves the pinned pelvis command against the replay's drift (iterative
learning) and re-solves the joints against the markers
(`solve_trajectory(locked_per_frame=...)`); the best replay is kept.
On both captures every gain diverges from iteration 0 (driver 74.6 to 134.8 mm at gain 0.7, to 103.9 mm at gain 0.25; 7-iron 112.3 to 192.9 mm), so the dynamics stage keeps the unmodified reference. The replay error is pelvis yaw lag (72.7 of 74.6 mm), a ground yaw-moment limit for this composite reference; the next form is a differentiable-simulator trajectory optimisation (JaxSim #6647 or MJX). Details REVIEW.md 16.

### Differentiable Trajectory Optimisation With MJX (FB-5, MM-7B, 2026-09-14)

`evidence/ground_support/export_mjx_package.py --run <run>` then, in the
MJX environment, `mjx_trajectory_optimisation.py --run <run> --iterations N
--learning-rate 5e-4` (`--diagnose` replays only, `--init` warm-starts,
`--horizon` sets the cost window). Environment recipe (Windows, CPU):
`python -m venv ~/.venv-mjx && ~/.venv-mjx/Scripts/pip install "jax[cpu]"
mujoco-mjx defusedxml numpy scipy` (JAX 0.11.1, MuJoCo 3.13, MJX 3.13; the
main environment keeps MuJoCo 3.3.4). Validate any optimised reference in
the shared-law plant with `downswing_experiment.py --run <run> --reference
<npz>`. Driver result: to 1.5 s the shared-plant replay drops from 56.1 to 40.2 mm (pelvis yaw lag at 1.4 s 12.3 to 2.6 deg); the uncosted follow-through collapses from iteration 8, so iterations 4 to 6 are the whole-swing choice (79.6 mm against 74.6). The MJX plant's own follow-through diverges after 1.6 s (not the weld: a five-times stiffer one is identical), and the joint-by-joint comparison shows it is the pelvis yaw response of the contact (joints track equally; MJX lags 50.6 deg in the follow-through, the shared plant 34.4, and less than the shared plant through the downswing), so the contact integration is the next thing to reconcile before a full-horizon solve. Details REVIEW.md 17.

### Completion Plan (Handoff Epic #10162, 2026-09-14)

The remaining work is organised as epic #10162 with tiered, self-contained
children (each carries files, TDD steps, DbC/LoD/DRY constraints, commands,
acceptance criteria and receipts): expert HO-5 (#10159, MJX plant
reconciliation, full horizon, 7-iron); moderate HO-4 (#10158, tile stages),
HO-8 (#10108, hip zero twist), HO-10 (#10112, dynamics replay parity);
cheap HO-1 (#10155, pipeline package under src, first), HO-2 (#10156,
receipt schema), HO-3 (#10157, MJX environment and JAX-gated tests), HO-6
(#10160, 7-iron receipts), HO-7 (#10161, design-decision record), HO-9
(#10111, de Leva verification); blocked #10110 (Simscape lane). Order:
HO-1, then HO-2 and HO-3, HO-7 and HO-9 any time, HO-4/HO-8/HO-10 after
HO-1, HO-6 after HO-5. Definition of done is on the epic.

**HO-0 (#10186, expert) comes before all of them (2026-09-15).** The 22
commits after PR #10087 (`11cccded9` to `ee418af35`) are on this branch
only, and `main` took a parallel lane on the same program (FB-4 #10089,
FB-5 #10092, FB-6 #10094, viewer tile #10090, fixes #10164/#10165): a
merge conflicts in 22 files (dry run 2026-09-15). HO-0 lands the branch on
`main` with one merge commit resolved by the per-file rules in the issue
(branch side for the 20 visual-layer files, main side for SPEC.md and the
divergence inventory, hand-merge of this file and the development log,
`static_marker_offsets` ported into the shared `marker_calibration`
module), records a "Two Implementations, One Program" table (main's
`full_body_ik.py`/`full_body_forward_dynamics.py`/`cross_engine_replay.py`
against this lane's `full_body_markers.py`/`full_body_simulation.py`/
`verify_setup_parity.py`), and repoints the epic and children at `main`.
Until HO-0 merges, child branches must start from this branch and will
not merge; nobody should start HO-1 before HO-0 is done.

### Two Implementations Coexist (Read This First)

Two implementations of the full-body program currently coexist in the repository: the FB-4/5/6 lane landed on main (#10089, #10092, #10094) and the Ground Support lane landed in this PR (#10186, #10113, #10162). Until unified in HO-1 (#10155), see the [Two Implementations, One Program](two-implementations-module-map.md#two-implementations-one-program) reconciliation table below for canonical assignments and receipts.

### How to Continue (Read This First)

1. Run the pipeline from the launcher tile "Motion Matching" (`python -m src.tools.motion_matching`) or headlessly via `python docs/development/full_body_models/evidence/ground_support/run_ground_support.py --spec <doc> ...`. The launcher tile provides a tabbed interface:
   - **Matching Tab**: Full pipeline execution with granular stage controls (free/bound wrists, fit closure, cart-table ZMP filter, whole-body shooting fit iterations and gain, cutoff frequency).
   - **Downswing Experiment Tab**: Run parameter sweep and variant comparison experiments against reference fits with custom torque/gain overrides.
   - **MJX Tab**: Export MJX optimization packages and validate optimized trajectory references against the shared contact-law plant.
     Headless pipeline runs: `python docs/development/full_body_models/build_anthropometric_spec.py ... --club driver|iron7`
     then `python docs/development/full_body_models/evidence/ground_support/run_ground_support.py --spec <doc> --skip-hip-calibration --static-seeds [--free-wrists] --capture driver|iron --out <run>` (wrists bounded by default for fitted documents; refit the hand rotation with `evidence/anthropometry/fit_grip_rotation.py --run <free-wrist run> ...`).
2. Read the receipt (`<run>/receipt.json`): `address.calibrated`, `ik`
   (full-capture IK, `range_of_motion_flags`, `attachments_m`), `dynamics`
   (`root_error_timeline_m`, `weight_fraction`, `reference_zmp`,
   `backswing_to_1s`); playback GIFs. Dynamics-only variants:
   `evidence/ground_support/downswing_experiment.py --run <run> --name <n> [...]`.
3. Cross-engine: `evidence/setup_parity/verify_setup_parity.py --run <run>`
   (Drake and Pinocchio on ControlTower over SSH; poses by coordinate name).
4. Never rerun with `--recalibrate-upper`; keep one static-trial round;
   ranges act on the matching only (`range_of_motion.py`).
5. Epics: #10162 (handoff, HO-0 landing #10186 first, then HO-1 to HO-10, tiered), #10113 (MM-1 to MM-10)
   and #10120 (CM-1 to CM-6) hold every open item with acceptance
   criteria; update DL-#10062 and this handoff in every implementation
   commit.
