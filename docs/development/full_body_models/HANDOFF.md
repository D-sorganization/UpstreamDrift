# Full-Body Models Handoff (Epic #10062)

## Global Polynomial Step Prerequisite (#10265)

The name-safe effort mapping and native RK4 step preserve the global degree-six
control contract and all six unactuated root coordinates. Derivatives apply
the chain rule through every RK4 stage and substep. Qualification here concerns
local numerical derivatives; it does not establish a fitted physical swing.
Independent acceptance replay must explicitly retain the optimization ground:
the historical forward runner currently auto-calibrates a near-zero plane.

## Qualified Solver Integration (#10254)

Resume with [Worker Turnover](../../plans/qualified_motion_integration/TURNOVER.md)
and [Pink Pipeline Packets](../../plans/qualified_motion_integration/PINK_TURNOVER.md).
Storage recovery is complete and native runtime probes pass. Adapter/derivative
PRs remain under review; production Pink selection and full-body Crocoddyl
qualification are not complete. The turnover records separate implementation,
CI and scientific gates.

The [integration plan](../../plans/qualified_motion_integration/README.md)
defines coordinate, closure, underactuation, timing and evidence boundaries.
First slice #10255 corrects the missing contact-force chain rule in full-body
Pinocchio derivatives. Independent real-engine differences reproduce failure
before the change and pass afterward; contact kinks remain explicitly nonsmooth.
No fit receipts were regenerated. The #10250 refresh is merged; #10271 receipt integrity and #10162's
physical acceptance remain prerequisites. #10256 and #10257 are separately
owned viewer and Pink adapter slices, not completed product integration.

## Optional Viewer Adapter Status (#10256, #10254)

MeshCat and Gepetto adapters now own persistent native visualizers, validate
configuration dimensions and preserve separate visual/collision models. Scene
cleanup is scoped to each adapter. Real MeshCat dispatch is exercised in the
opt-in heavy integration test; live Gepetto qualification remains pending.
These wrappers are prerequisites for replay integration, not evidence that the
full-body product already routes through them or that any model fit is valid.

## Reproducible Numerical Runtime (#10262)

Use the optional runtime manifest/lock and checker documented in
`docs/engines/pinocchio.md`. Real Crocoddyl descent and Pink hard-equality /
infeasible-QP probes pass in the consistent conda-forge stack. Preserve the
receipt's explicit source-freshness status; a runtime pass is not model or
physical acceptance. Linux lock requires x86-64-v3. Viewer qualification
remains separate.

## Pink Adapter Status (#10257, #10254)

The low-level Pink adapters now share validated state/time handling, retain
collision geometry, forward hard constraints and limits, refresh cached
kinematics and propagate infeasibility. Four real-native tests cover equality,
infeasibility, sequential state refresh and free-flyer `nq != nv` behavior.
Mixed unit/native collection isolates the real tests from unit mocks and
checks their executed results. This does not implement the full-body marker,
stance or weld tasks; those must use the qualified #10260 closure derivative
and explicit physical-time contract before production selection.

Finite weld correction #10260 differentiates the six-component pose error away
from closure with the correct SE(3) log Jacobian. Trajectory acceleration
linearization retains the separate constraint velocity Jacobian. Eleven real
closure regression tests and the 11 contact derivative tests pass on Pinocchio
3.8 and 4.1; no motion-fit evidence or acceptance thresholds changed.
The principal-log branch at rotation pi is rejected for derivative evaluation;
trajectory finite differences also reject steps that could cross that cut.

Updated 2026-09-16 (claude; HO-9 #10111). Branch
`feat/10111-de-leva-table`. Design: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md).
Design Decisions: [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).
Copy-ready prompt for the next agent: [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

## Completed Slices (Archived)

Completed slices (FB-3-M/P/D, visual skeleton layer, launcher tile, FB-4, FB-5, FB-6, Visuals step 4) are archived verbatim in [FULL_BODY_MODELS_HANDOFF_ARCHIVE_2026.md](../archive/FULL_BODY_MODELS_HANDOFF_ARCHIVE_2026.md).

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

### Dated AN-1 Findings (Archived)

The dated 2026-09-14 AN-1 findings (neck and arms through MJX trajectory optimisation) are archived verbatim in [FULL_BODY_MODELS_HANDOFF_ARCHIVE_2026.md](../archive/FULL_BODY_MODELS_HANDOFF_ARCHIVE_2026.md).

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

Two implementations of the full-body program currently coexist in the repository: the FB-4/5/6 lane landed on main (#10089, #10092, #10094) and the Ground Support lane landed in this PR (#10186, #10113, #10162). Until unified in HO-1 (#10155), see the [Two Implementations, One Program](#two-implementations-one-program) reconciliation table below for canonical assignments and receipts.

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

## Two Implementations, One Program

The following table records the canonical module assignments to be executed in HO-1 (#10155):

| Concern                       | On `main` (FB-4/5/6 lane)                                                                                                   | On the branch (ground-support lane)                                                                                                                  | Canonical Choice for HO-1                                                                                                                                 | Validating Receipt / Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Marker IK                     | `src/shared/python/motion_matching/full_body_ik.py` (240), `src/engines/physics_engines/mujoco/python/full_body_ik.py` (92) | `src/engines/physics_engines/mujoco/python/full_body_markers.py` (775; static seeds, planted stance, axis rows, CoM rows, `locked_per_frame`)        | **Consolidated** (`full_body_markers.py` retired to deprecation shim; shared IK in `full_body_ik.py`, MuJoCo adapter in `mujoco/python/full_body_ik.py`)  | Validated by tour capture receipts regenerated on the mirrored left hip (#11737), the flexion-negative right knee (#12057) and the hip rewrite with the spec's own pelvis alignment (#12109), the driver on its measured club cluster (#12221), then the driver receipts regenerated for #12221 with the per-capture grip rotation fitted and the club cluster held fixed (`--fix-club-markers`), and the 7-iron receipts regenerated the same way for #12279: driver address 10.95 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/receipt.json#address.calibrated.marker_rms_m`), IK 30.28 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/receipt.json#ik.marker_rms_m`); 7-iron address 8.38 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json#address.calibrated.marker_rms_m`), IK 28.93 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json#ik.marker_rms_m`). The pre-HO-8 driver IK 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`) used widened leg bounds and carries IK range-of-motion flags; history only (`CANONICAL_RUN.md` §2C). Primary uncalibrated baselines: driver address 50.94 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#address.calibrated.marker_rms_m`), IK 46.46 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#ik.marker_rms_m`); 7-iron address 40.43 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json#address.calibrated.marker_rms_m`), IK 44.52 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json#ik.marker_rms_m`); its earlier 264.8 mm trajectory-IK failure came from the face residual fitting the driver head-triad geometry that the 7-iron document carries, now gated by the triad-shape check (#12030, `CANONICAL_RUN.md` §2B). Pre-OSV-6 baselines: IK 52.27 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_pre_osv6/receipt.json#ik.marker_rms_m`) and 71.98 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_pre_osv6/receipt.json#ik.marker_rms_m`). Carries static trial placement, planted stance, and anatomical joint limits. |
| Forward dynamics and tracking | `src/shared/python/motion_matching/full_body_forward_dynamics.py` (785; zero-feedback, polynomial control, contact audit)   | `src/engines/physics_engines/mujoco/python/full_body_simulation.py` (692; computed torque with root free, acceleration feedforward, `reference_zmp`) | **Consolidated** (`full_body_simulation.py` retired to deprecation shim; forward dynamics and tracking consolidated into `full_body_forward_dynamics.py`) | Validated by whole-run marker RMS 82.00 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/receipt.json#dynamics.marker_rms_m`) and 82.81 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json#dynamics.marker_rms_m`) with the GCV-20 ball impact, weld projection and impact-split reference ZMP (#11767, #12117), on top of the pelvis-alignment fix (#12109; 75.4 and 59.6 mm without GCV-20; 77.0 and 71.2 mm after the right-knee fix #12057, 61.6 and 47.0 mm before it with the trail knee pinned at its 10 deg bound), pre-HO-8 74.6 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#dynamics.marker_rms_m`), 89.1 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_pre_osv6/receipt.json#dynamics.marker_rms_m`), 116.0 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_pre_osv6/receipt.json#dynamics.marker_rms_m`), and 144.0 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json#dynamics.marker_rms_m`) with unactuated floating root, compliant sole, and computed-torque tracking.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| MJCF and native model         | `src/engines/physics_engines/mujoco/python/full_body_mjcf.py` (172), `native_model.py` (226, on both)                       | `src/engines/physics_engines/mujoco/python/full_body_model.py` (211; anthropometric documents, toe spheres, closure sites)                           | **Ground Support Lane** (`full_body_model.py` to be consolidated into `full_body_mjcf.py` / `native_model.py`)                                            | Carries anthropometric documents, toe contact spheres, and dual-grip closure sites required for ground support.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Cross-engine replay           | `src/shared/python/motion_matching/cross_engine_replay.py` (366), `derivative_resolution.py` (193)                          | `docs/development/full_body_models/evidence/setup_parity/verify_setup_parity.py` (poses by coordinate name over SSH)                                 | **Main Lane** (`cross_engine_replay.py`, `derivative_resolution.py`)                                                                                      | Validated by `evidence/fb6_parity/parity_report.json` and step-size convergence analysis across MuJoCo, Pinocchio, and Drake; replaces the SSH script in #10112.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Contact law                   | `src/shared/python/motion_matching/contact_law.py`                                                                          | same module (imported by the branch's simulation)                                                                                                    | **Identical**                                                                                                                                             | Same shared module used across both implementations; no conflict.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Evidence                      | `evidence/fb4_calibration`, `fb5_matching`, `fb6_parity`, `viewer`                                                          | `evidence/ground_support`, `anthropometry`, `setup_parity`, `visual_layer`                                                                           | **Both Retained**                                                                                                                                         | Both sets of evidence are retained in the tree under `docs/development/full_body_models/evidence/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Status and Next Steps

- **HO-0 (#10186)**: Landed on `main` reconciling ground-support and FB-4/5/6 lanes.
- **HO-1 (#10155)**: Landed on `main` in PR #10218 (`afea5e0a8`). Ground-support pipeline package established under `src/shared/python/motion_matching/pipeline/` with CLI wrapper in `docs/development/full_body_models/evidence/ground_support/run_ground_support.py`.
- **HO-2 (#10156)**: Implemented Pydantic V2 receipt schema and validator (`src/shared/python/motion_matching/pipeline/receipt_schema.py`), unit tests validating all committed receipts and rejection paths (`test_receipt_schema.py`), generated markdown documentation (`docs/development/full_body_models/RECEIPTS.md`), and freshness test (`test_receipts_markdown_freshness.py`).
- **HO-7 (#10161)**: Consolidated seventeen sections of findings from `evidence/anthropometry/REVIEW.md` into authoritative design decision record `docs/development/full_body_models/DESIGN_DECISIONS.md`. Established structured parser, validator with DbC contracts, and full test suite `tests/unit/motion_matching/pipeline/test_design_decisions.py`. Merged to `main` in PR #10235 (`27b271fe2`).
- **HO-4 (#10158)**: Exposed all pipeline stages in the Motion Matching launcher tile (`src/tools/motion_matching/gui.py`, `pipeline.py`). Tabbed interface across Matching (with granular Stages options for free/bound wrists, fit closure, ZMP filter, shooting fit, cutoff frequency), Downswing experiment, and MJX tabs, with asynchronous `RunWorker` and strict LoD. Headless PyQt6 and pipeline test suites in `tests/tools/motion_matching/`. Added `tools.motion_matching` to `src/config/feature_parity.json` (tracking gap #10106) and regenerated `docs/development/feature_parity_matrix.md`. Merged to `main` in PR #10236 (`db4fe88c4`).
- **HO-9 (#10111, MM-9)**: Verified de Leva (1996) male segment table line-by-line against Table 4 of the published paper. Pinning unit test in `tests/unit/motion_matching/test_de_leva_table.py` asserts each segment against literal paper values (mass %, CoM %, and principal radii %). Corrected shank `com_fraction` from 0.4459 to 0.4395 and radii from (0.255, 0.249, 0.103) to (0.251, 0.246, 0.102) in `src/shared/python/motion_matching/anthropometry.py`, eliminating bony-landmark carryover from Zatsiorsky-Seluyanov 1985. Committed verification receipt `docs/development/full_body_models/evidence/anthropometry/de_leva_verification.json`.
- **HO-3 (#10157)**: Delivered cross-platform MJX virtualenv setup scripts (`scripts/setup_mjx_env.ps1`, `scripts/setup_mjx_env.sh`) reading pinned versions from `scripts/config/mjx_env_pins.json`, refactored differentiable trajectory optimization into pure testable functions with `SiteState` and `WeldGains` parameter tuples complying with architecture parameter budgets, added `requires_jax` marker to `pyproject.toml`, and implemented 6 unit tests in `tests/unit/motion_matching/test_mjx_optimisation.py` covering knot basis, tail freezing mask, contact force parity (< 1e-6 N), weld wrench equilibrium, initial preload, and finite gradients on truncated 12-frame horizon. Merged to `main` in PR #10228 (`09346c9dd`).
- **HO-11 (#10250)**: Rebuilt anthropometric documents (`full_body_spec_anthro_driver.json`, `..._iron7.json`) and ground-support receipts with canonical `de_leva_table_sha256` and `anthropometry` blocks after HO-9 de Leva shank fix. Added document freshness test suite `tests/unit/motion_matching/test_document_freshness.py` with 9 unit/contract tests enforcing byte-level agreement against `DE_LEVA_MALE`. Updated Pydantic receipt schema (`receipt_schema.py`), pipeline receipt builder (`receipt.py`), and regenerated `RECEIPTS.md`. Fixed local engine imports in `pipeline/lane.py` and `pipeline/address.py`. Re-executed driver and 7-iron captures, validating sub-millimeter shift (< 1e-4 m) between unedited and rebuilt models. Merged to `main` in PR #10261 (`81ea27bfb`).
- **HO-8 (#10108, MM-6)**: Delivered hip coordinate zero-twist functional calibration (`src/shared/python/motion_matching/hip_calibration.py`) using knee medial markers or shank-thigh ankle plane normal fallback, post-multiplying `parent_to_base` by $R_z(\theta)$ to rotate coordinate zero without translating hip centers. Set `BOUND_WIDENING: float = 1.0` in `src/shared/python/motion_matching/pipeline/constants.py`. Regenerated receipts for `anthro_driver`, `anthro_driver_shoot`, `anthro_iron`, and `anthro_iron_shoot`, verifying 0 lower-limb `range_of_motion_flags` on the IK reference for both captures. Re-verified cross-engine setup parity across MuJoCo, Drake, and Pinocchio.
- **Review 2026-09-16 (expert agent)**: receipts on `main` unchanged through
  the landing on canonical calibrated runs (driver address 5.1 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#address.calibrated.marker_rms_m`), IK 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`), dynamics 74.6 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#dynamics.marker_rms_m`)). Two gaps found and filed: **HO-11 (#10250)** the anthropometric
  documents and every receipt were stale since HO-9 changed the de Leva
  shank row; **HO-12 (#10251)** `run_ground_support.py` was
  779 lines on `main` against HO-1's acceptance of under 400.
- **MS-03 (#10324)**: Reconciled headline tour numbers with primary receipts on `main`. Formally designated canonical calibrated reference runs (`anthro_driver_shoot_g025`, `anthro_iron_zmp`) vs primary uncalibrated baselines (`anthro_driver`, `anthro_iron`), produced `CANONICAL_RUN.md` and `bisect_receipt.json` attributing the 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`) to 52.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_pre_osv6/receipt.json#ik.marker_rms_m`) shift to omission of `--static-seeds` during HO-8 regeneration, and added document freshness test suite `tests/unit/motion_matching/test_handoff_numbers_match_receipts.py`.
- **MS-11 (#10330)**: Finished HO-1 IK and forward dynamics consolidation into shared modules. Retired `full_body_markers.py` (813 -> 44 lines) and `full_body_simulation.py` (709 -> 55 lines) to backward-compatible deprecation shims, eliminating 1,423 duplicate lines across engine files while preserving zero MuJoCo imports in `src/shared/python/motion_matching/`. Consolidated ground-support IK into `BaseFullBodyIK` in `src/shared/python/motion_matching/full_body_ik.py` and `FullBodyMarkerKinematics` in `src/engines/physics_engines/mujoco/python/full_body_ik.py`. Consolidated computed torque, reference ZMP, foot preloading, and RK4 rollout into `src/shared/python/motion_matching/full_body_forward_dynamics.py`. Added comprehensive consolidation test suite `tests/unit/motion_matching/test_full_body_consolidation.py` verifying import boundaries, deprecation warnings, and strict numerical parity.
- **Doc hotspot rule**: `DEVELOPMENT_LOG.md` (`DL-#10062`) and `HANDOFF.md` "Status and Next Steps" are edited by every child, so they conflict whenever another child merges first. Resolve once, as the last step before merge: `git merge origin/main`, take `main`'s copy of both files (`git checkout --theirs`), re-apply only your own lines (your DL "Last verified"/"Next step", your one status bullet), commit, push. Never push repeated "Merge branch 'main'" commits with unresolved hunks. PR title prefix must match the diff (`test:`/`docs:`/`refactor:` when nothing under `src/` changes) or phantom-guard fails. `equivalence (3.11)` and `Trivy Container Scan` are not required and fail on `main` already; name them as pre-existing in the PR body, do not fix them in a child PR. Run `python scripts/ci/check_architecture_budget.py` before pushing (8-parameter and function-size budgets on changed files).
- **Next**: Land HO-12 PR (#10251), then proceed to HO-10 (#10160), HO-5 (#10159), and HO-6 (#10121) in epic #10162.
