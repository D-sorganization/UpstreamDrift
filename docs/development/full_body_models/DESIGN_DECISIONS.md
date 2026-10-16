# Design Decisions for the Full-Body Showpiece

Authoritative record of architectural and biomechanical decisions for the full-body golf model showpiece (epic [#10162](https://github.com/D-sorganization/UpstreamDrift/issues/10162), task HO-7 [#10161](https://github.com/D-sorganization/UpstreamDrift/issues/10161)). This document consolidates findings across sections 1 to 17 of [REVIEW.md](evidence/anthropometry/REVIEW.md), preserving the decisions, rationales, evidence receipts, and rejected alternatives for ongoing development.

---

## 1. Anthropometric Geometry From de Leva

### What

Replace Simscape default geometric constants and masses with subject-specific segment dimensions, masses, centres of mass, and inertia tensors derived from de Leva (1996) body-segment parameter proportions via `src/shared/python/motion_matching/anthropometry.py` and `build_anthropometric_spec.py`. For the tour capture subject (stature 1.71 m, mass 78 kg), segment dimensions use trunk scale 1.15, arm scale 1.10, and shoulder scale 1.00, reducing total body mass from 108.3 kg to 78.0 kg (79.4 kg including club and hands).

### Why

The original Simscape model placed the root "spine" joint 0.515 m above functional hip centres (at the base of the neck), gave the hub 0.508 m clavicle links pointing 21° to 47° downward, and assigned 67.0 kg to the trunk and head (including 20 kg in rigid shoulder links). This geometric distortion forced unnatural spine bends (-7° forward, +21° lateral) and depressed scapulae at address. de Leva scaling establishes anatomically grounded limb lengths, centers of mass, positive-definite inertia tensors, and neutral address posture.

### Evidence Receipt

- [`evidence/anthropometry/receipt.json`](evidence/anthropometry/receipt.json)
- [`evidence/anthropometry/scan_geometry_receipt.json`](evidence/anthropometry/scan_geometry_receipt.json)
- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Scaling upper arm, forearm, and hub lengths on the legacy Simscape trunk topology without relocating the root joint and hub down to shoulder level ([`evidence/ground_support/candidate_anthro/receipt.json`](evidence/ground_support/candidate_anthro/receipt.json)). Shortening the limbs without lowering the hub prevented shoulders from reaching marker targets, severely worsening address marker RMS from 3.1 mm to 48.3 mm and increasing spine bend to 61° forward / 85° lateral.

### REVIEW.md Sections

- [REVIEW.md Section 1: How the Current Dimensions Were Determined](evidence/anthropometry/REVIEW.md#1-how-the-current-dimensions-were-determined)
- [REVIEW.md Section 3: Model Versus Anthropometry](evidence/anthropometry/REVIEW.md#3-model-versus-anthropometry)
- [REVIEW.md Section 5: Recommendations](evidence/anthropometry/REVIEW.md#5-recommendations-ordered-each-with-a-test)
- [REVIEW.md Section 7: Experiment: Scaling Alone Does Not Work](evidence/anthropometry/REVIEW.md#7-experiment-scaling-alone-does-not-work)
- [REVIEW.md Section 9: AN-1 Iteration 1](evidence/anthropometry/REVIEW.md#9-an-1-iteration-1-anthropometric-geometry-built-and-measured-10099)

---

## 2. Arms Forward at Zero Pose

### What

Define the zero reference pose of the upper arms pointing forward along the $+X$ axis (`ARM_FORWARD` convention) rather than hanging vertically along $-Z$. Restrict the elbow hinge coordinate range to a one-sided window ($-150^\circ$ to $+5^\circ$, where negative flexion lifts the wrist toward the shoulder).

### Why

With upper arms hanging downward at zero pose along $-Z$, the middle pitch rotation of the spherical shoulder gimbal sits directly at its $90^\circ$ gimbal singularity throughout the address and backswing. This caused numerical solver divergence in the trajectory inverse kinematics, blowing up marker tracking error to 75–200 mm. Pointing arms forward moves the singularity completely outside the functional workspace of the golf swing.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Hanging-arm zero pose (`ARM_DOWN`), which diverged due to gimbal lock across multiple swing phases.

### REVIEW.md Sections

- [REVIEW.md Section 9: AN-1 Iteration 1](evidence/anthropometry/REVIEW.md#9-an-1-iteration-1-anthropometric-geometry-built-and-measured-10099)

---

## 3. Scapula Rz

### What

Formulate the second rotational degree of freedom of the clavicle/scapula joint as protraction/retraction about the local vertical axis (`Rz`) rather than a roll rotation about the clavicle link axis.

### Why

In the original model formulation, the second scapula primitive was an axial spin around the clavicle link itself, creating a redundant null space with the shoulder gimbal. The optimization solver exploited this null space to artificially depress the clavicle links downward to satisfy marker positions, producing an unnatural hunched appearance. Reorienting the axis to `Rz` provides independent physiological protraction and retraction without humeral coupling.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Axial clavicle roll primitives, which created a null motion direction and drove unconstrained clavicle depression.

### REVIEW.md Sections

- [REVIEW.md Section 9: AN-1 Iteration 1](evidence/anthropometry/REVIEW.md#9-an-1-iteration-1-anthropometric-geometry-built-and-measured-10099)

---

## 4. One Static-Trial Round

### What

Calibrate subject marker attachments from a single neutral static trial round across the first 24 capture frames (`marker_calibration.static_marker_offsets`, `--static-seeds`), locking scapulae at zero and constraining spine posture within $10^\circ$ forward and $5^\circ$ lateral bend. Freeze all 34 marker attachment offsets for all subsequent trajectory inverse kinematics and dynamics optimizations.

### Why

Unconstrained swing-wide marker calibration with generic anatomical priors deformed marker offsets into physically impossible positions (e.g. waist markers moving 140 mm away from hip centers) while the IK solver bent the spine $24^\circ$ forward to compensate. One round of static trial fitting yields an address marker RMS of ~1.0 mm and guarantees neutral posture by construction.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Running a second static-trial round using placed markers as seeds, or running unconstrained dynamic marker calibration across the entire swing (`--recalibrate-upper`). A second static-trial round formed a degenerate fixed point with excellent static fit (3 mm) but disastrous swing tracking (81 mm error).

### REVIEW.md Sections

- [REVIEW.md Section 9: AN-1 Iteration 1](evidence/anthropometry/REVIEW.md#9-an-1-iteration-1-anthropometric-geometry-built-and-measured-10099)
- [REVIEW.md Section 9.2: Neck Joint and Address Arms](evidence/anthropometry/REVIEW.md#92-neck-joint-and-address-arms-user-direction-2026-09-14)

---

## 5. Marker-Driven Elbow Pits

### What

Direct upper-arm elbow-pit orientation using vector directions computed directly from the shoulder, elbow, and wrist markers (`posture_metrics.elbow_pit_direction`), averaged across static address frames and tracked dynamically per frame (`solve_trajectory(axis_targets_per_frame=...)`).

### Why

An arbitrary anatomical heuristic (forcing both elbow pits to face upwards and inwards towards each other at address) directly contradicted the capture data: the lead arm marker chord showed $48^\circ$ flexion with the pit pointing downward and outward (-0.35 up, +0.49 inward, -0.38 forward). Forcing the heuristic caused severe hyperextension against bounds and degraded swing tracking error from 24 mm to 68–81 mm. Deriving pit vectors directly from markers resolved the conflict cleanly.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Fixed heuristic elbow pit targets (up and inward), inward humerus rotational seeding, and dominant pit objective weights (weight 5.0), all of which forced elbow hyperextension and broke full-swing kinematic tracking.

### REVIEW.md Sections

- [REVIEW.md Section 9.2: Neck Joint and Address Arms](evidence/anthropometry/REVIEW.md#92-neck-joint-and-address-arms-user-direction-2026-09-14)
- [REVIEW.md Section 9.3: Wrist Axis, Setup Position](evidence/anthropometry/REVIEW.md#93-wrist-axis-setup-position-cross-engine-parity-centre-of-mass-user-direction-2026-09-14)
- [REVIEW.md Section 12: Elbow Pits from the Markers](evidence/anthropometry/REVIEW.md#12-elbow-pits-from-the-markers-grip-roll-club-mesh-epic-2026-09-14)

---

## 6. Anatomical Wrist Axes and Neutral-Grip Turn

### What

Redefine the wrist kinematic joint sequence so radial/ulnar deviation (cocking, `Rx`) is aligned parallel to the elbow flexion axis (lifting the hand toward the elbow pit) and flexion/extension (`Rz`) acts about the palm normal. Apply a $25^\circ$ neutral-grip ulnar offset (`GRIP_ULNAR_OFFSET_DEG`) between wrist base and hand bodies so coordinates read zero at standard address grip.

### Why

The inherited native wrist joint copied from Simscape placed the cock axis perpendicular to the forearm flexion plane, making radial deviation translate the hand sideways rather than upward, while the second wrist primitive duplicated forearm pronation/supination. The corrected anatomical sequence restored true radial/ulnar cocking and flexion/extension.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)
- [`evidence/ground_support/anthro_iron/receipt.json`](evidence/ground_support/anthro_iron/receipt.json)

### What Was Tried and Rejected

Legacy wrist frames without ulnar offset, which forced large non-zero address angles and coupled wrist cocking into forearm spin.

### REVIEW.md Sections

- [REVIEW.md Section 9.3: Wrist Axis, Setup Position](evidence/anthropometry/REVIEW.md#93-wrist-axis-setup-position-cross-engine-parity-centre-of-mass-user-direction-2026-09-14)
- [REVIEW.md Section 11: Anatomical Wrist, Visual Realism](evidence/anthropometry/REVIEW.md#11-anatomical-wrist-visual-realism-launcher-tool-epic-10113-user-direction-2026-09-14)

---

## 7. Fitted Hand-to-Club Rotation (`GRIP_ROTATION_DEG`)

### What

Calibrate a fixed 3D spatial rotation between the wrist follower frame and the hand body (`subject.grip_rotation_deg`, `GRIP_ROTATION_DEG`), solved globally via Nelder-Mead optimization (`evidence/anthropometry/fit_grip_rotation.py`) to minimize total joint range excursions beyond human limits across all swing frames for both driver and 7-iron captures simultaneously. Fitted values: lead hand $[-89.7^\circ, 46.7^\circ, 0.0^\circ]$, trail hand $[-34.2^\circ, 46.0^\circ, 62.0^\circ]$.

### Why

Because the C3D dataset contains no markers on the hands, the wrist orientation is observed only through the club shaft. Misalignment in the resting hand-to-club attachment forced the optimizer to dump unmodelled rotation into wrist cocking (producing $+110^\circ$ of unphysical cock travel and pegging forearm pronation at $+90^\circ$). Calibrating `GRIP_ROTATION_DEG` reduced RMS excursion from $40.6^\circ$ down to $3.9^\circ$ on the lead hand and $17.7^\circ$ down to $1.2^\circ$ on the trail hand.

### Evidence Receipt

- [`evidence/anthropometry/fit_grip_rotation_receipt.json`](evidence/anthropometry/fit_grip_rotation_receipt.json)
- [`evidence/anthropometry/fit_grip_rotation_residual_receipt.json`](evidence/anthropometry/fit_grip_rotation_residual_receipt.json)
- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### Per-Capture Refit on the Fixed Clusters (#12221)

The values above remain the documents' values. With the club clusters held fixed, one rotation per capture was refitted from free-wrist IK runs, minimising the RMS excess of pronation, cock and flexion beyond `coordinate_ranges_deg` over the whole swing plus the last 30 pre-impact frames (weight 10) and the first 30 frames (weight 20). Totals (extrinsic x-y-z, degrees): driver lead $[-36.864, 57.315, 70.926]$, trail $[9.769, 52.918, 83.914]$; 7-iron lead $[-48.993, 15.681, 29.676]$, trail $[37.035, 49.131, 121.189]$. With the fixture flags and `--fix-club-markers`, the tracked impact speed was $-1.9\%$ on both clubs, the impact-face gaps were $0.56^\circ$ (driver) and $0.62^\circ$ (7-iron), and no wrist sat on its bound at address. The refit is valid only with `--fix-club-markers`: with the cluster free the static trial re-absorbs the grip into the club triad and moves the capture's own face. It is applied to the driver; the 7-iron is deferred. `full_body_spec_anthro_driver.json` carries the fitted rotation (`scripts/set_spec_grip_rotation.py`), and the calibrated driver commands add `--fix-club-markers`. Regenerated driver address / IK / dynamics (mm): calibrated 11.0 / 30.3 / 82.0, nominal 50.9 / 46.5 / 71.9; the Drake driver dynamics worsened (77.8 to 86.3). The driver club-face fixture was regenerated (impact speed -2.5 %, was -5.1 %; impact face 16.88 / 14.96 deg). The 7-iron refit meets GCV-20 and OSV-10 on the tracked swing, but its same-input export gives -5.0 %, -5.1 ms and an 8.71 deg impact face, which would turn eight passing face tests red, so its document and receipts are unchanged and its fit receipt stays committed. Under #12279, the 1 ms ZOH export hold was found to slide the 7-iron feet by 43 mm with 24.0 mm root error at 0.7 s and -0.147 rad `mtp_angle_r` error at 0.8 s; the 0.125 ms hold reduces foot slip to 3.3 mm (bounded within 10.0 mm), root error at 0.7 s to 8.2 mm, `mtp_angle_r` error at 0.8 s to -0.001 rad, 7-iron impact speed from -5.0% to -2.5%, and impact face from 8.71 deg to 5.56 deg (driver speed from -2.5% to -1.2%, timing -4.4 ms to -3.5 ms, face 16.88 deg to 16.91 deg; see volume 2 [decision 20](DESIGN_DECISIONS_2.md#20-same-input-effort-hold-is-a-bundle-parameter-12279)). The replay is sensitive at the 3 to 15 mm level to tiny parameter changes (see `CANONICAL_RUN.md` §2A). Making `--fix-club-markers` the calibrated default remains an owner decision.

- [`evidence/anthropometry/fit_grip_rotation_receipt_driver.json`](evidence/anthropometry/fit_grip_rotation_receipt_driver.json)
- [`evidence/anthropometry/fit_grip_rotation_receipt_iron7.json`](evidence/anthropometry/fit_grip_rotation_receipt_iron7.json)
- Calculation record: "Per-Capture Grip Rotation on the Fixed Clusters (Issue #12221)" in `docs/research/simscape_matching_reference/simscape_matching_reference.tex`

### What Was Tried and Rejected

One-dimensional grip roll rotation scans (`scan_grip_roll.py`, tested from $-90^\circ$ to $+90^\circ$) and adjusting the closure weld placement from address (`closure_fit.fit_closure_placement`). Neither 1D roll nor closure weld adjustment could eliminate the multidimensional wrist coordinate excursion.

### REVIEW.md Sections

- [REVIEW.md Section 12: Elbow Pits from the Markers](evidence/anthropometry/REVIEW.md#12-elbow-pits-from-the-markers-grip-roll-club-mesh-epic-2026-09-14)
- [REVIEW.md Section 13: Closure Weld Fitted from the Address](evidence/anthropometry/REVIEW.md#13-closure-weld-fitted-from-the-address-mm-2-2026-09-14)
- [REVIEW.md Section 14: Hand-to-Club Rotation Fitted from the Matches](evidence/anthropometry/REVIEW.md#14-hand-to-club-rotation-fitted-from-the-matches-wrists-bounded-mm-2-2026-09-14)

---

## 8. Human Ranges in the Matching Only, Wrists Bounded by Default

### What

Enforce anatomical joint range limits (`range_of_motion.py`, `HUMAN_RANGES_DEG`) strictly inside the kinematic matching and inverse kinematics optimization layers, bounding spine, neck, scapulae, elbows, and wrists by default for documents with fitted grip rotations. Do not embed joint hard-stops into the forward dynamics equations of motion.

### Why

Joint limits represent active muscular restraint and ligamentous elasticity rather than rigid mechanical stops. Imposing rigid joint limits in forward dynamics causes integrator chatter, stiff differential equations, and numerical failure during impact transients. Imposing bounds in the IK layer produces kinematically compliant reference trajectories while keeping the dynamical equations smooth and clean.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)
- [`evidence/ground_support/anthro_iron/receipt.json`](evidence/ground_support/anthro_iron/receipt.json)

### What Was Tried and Rejected

Enforcing wrist bounds prior to calibrating `GRIP_ROTATION_DEG`, which destroyed marker tracking (driver IK error rose from 24 mm to 65–77 mm). Bounding is only valid once the hand frame rotation has been calibrated.

### REVIEW.md Sections

- [REVIEW.md Section 10: Ranges of Motion](evidence/anthropometry/REVIEW.md#10-clubs-captures-torso-visuals-ranges-of-motion-balance-user-direction-2026-09-14)
- [REVIEW.md Section 14: Hand-to-Club Rotation Fitted from the Matches](evidence/anthropometry/REVIEW.md#14-hand-to-club-rotation-fitted-from-the-matches-wrists-bounded-mm-2-2026-09-14)

---

## 9. Clubs From `club_models`

### What

Define club specifications (lengths, head mass, shaft mass, grip mass, center of mass, and inertia tensors) via the unified `src/shared/python/motion_matching/club_models.py` module, linking directly to the shared `ClubDatabase` (`club_models.from_database(club_id)`).

### Why

Consolidates physical club properties into a single shared source of truth across driver and 7-iron models, eliminating divergent ad-hoc masses across simulation scripts, CAD models, and unit tests.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)
- [`evidence/ground_support/anthro_iron/receipt.json`](evidence/ground_support/anthro_iron/receipt.json)

### What Was Tried and Rejected

Hardcoding club masses and inertia values separately in individual document generator scripts and simulation fixtures.

### REVIEW.md Sections

- [REVIEW.md Section 10: Clubs, Captures](evidence/anthropometry/REVIEW.md#10-clubs-captures-torso-visuals-ranges-of-motion-balance-user-direction-2026-09-14)
- [REVIEW.md Section 11: Club Specs from the Club Database](evidence/anthropometry/REVIEW.md#11-anatomical-wrist-visual-realism-launcher-tool-epic-10113-user-direction-2026-09-14)

---

## 10. Compliant 50 kN/m Sole

### What

Set ground contact sphere stiffness to $5.0 \times 10^4\text{ N/m}$ ($50\text{ kN/m}$) and dissipation to $2.0\text{ s/m}$ in full-body specifications (`CONTACT_STIFFNESS_N_M`, `CONTACT_DISSIPATION_S_M`).

### Why

At the previous stiff setting ($200\text{ kN/m}$), static deflection was only 4 mm. The large angular momentum demands of the golf downswing tilted the pelvis slightly ($< 1^\circ$ pitch error), completely lifting the forefoot and toe spheres 1–2 mm off the ground. Once unloaded, Coulomb friction collapsed to zero, causing the feet to skate 120 mm across the turf and rendering the golfer airborne. Softening to $50\text{ kN/m}$ allows realistic turf/shoe compliance (16 mm sinkage), keeping all contact spheres firmly planted, eliminating lift-off, and reducing downswing root error from 164 mm to 38 mm.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/downswing_k50_d2.json`](evidence/ground_support/anthro_driver/downswing_k50_d2.json)
- [`evidence/ground_support/anthro_driver/downswing_k50_d2_lp12.json`](evidence/ground_support/anthro_driver/downswing_k50_d2_lp12.json)
- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Overly soft soles ($25\text{ kN/m}$, which sank excessively, causing 169 mm root error), stiff soles ($100\text{ kN/m}$, which continued hopping with 128 mm error), and increasing friction coefficients up to $\mu=3.0$ (ineffective because unloaded contact spheres slide regardless of $\mu$).

### REVIEW.md Sections

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW_DYNAMICS.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

---

## 11. 12 Hz Tracked Reference

### What

Filter the kinematic joint reference trajectory through a 12 Hz zero-phase low-pass filter (`TRACKING_CUTOFF_HZ = 12.0`) before computing torque tracking commands in forward dynamics.

### Why

Kinematic inverse kinematics re-solving introduces high-frequency frame-to-frame noise (particularly near impact where markers drop out), producing artificial root accelerations up to $492\text{ m/s}^2$ and unphysical joint torque spikes exceeding $2500\text{ N}\cdot\text{m}$. A 12 Hz low-pass filter completely eliminates impact jerk spikes, reducing peak joint torques from $2693\text{ N}\cdot\text{m}$ down to realistic values ($476\text{ N}\cdot\text{m}$ for driver, $835\text{ N}\cdot\text{m}$ for 7-iron) without degrading tracking fidelity.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/downswing_k50_d2_lp12.json`](evidence/ground_support/anthro_driver/downswing_k50_d2_lp12.json)
- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)

### What Was Tried and Rejected

Tracking raw unfiltered kinematic trajectories (caused severe impact torque spikes $>2500\text{ N}\cdot\text{m}$) and 8 Hz filtering (oversmoothed transition dynamics, increasing tracking error to 188 mm).

### GCV-20 Amendment (#11767): Release-Preserving Pre-Contact Cutoff and Weld-Consistent Track

The post-contact half keeps 12 Hz. The pre-contact half (the filter is split at the capture impact) now uses the lowest of 12, 15, 18, 20, 25, 30 Hz that keeps the unfiltered reference's speed-peak time within one sample and its last whole pre-contact segment speed within 1 % (`pipeline.release_cutoff`); both captures select 25 Hz (driver reference at 12 Hz: peak 13.1 ms before impact, 46.40 m/s; at 25 Hz: 2.0 ms, 50.62 m/s; unfiltered 2.0 ms, 50.46 m/s). The filtered track is then projected onto the grip weld over the trail-arm coordinates (`pipeline.weld_projection`), because per-coordinate filtering opens it by up to 16.1 mm.

| Club | Tracked reference | Peak vs capture (ms) | Impact speed vs capture | Face gap at impact (deg) |
| --- | --- | --- | --- | --- |
| Driver | 12 Hz (before) | -20.0 | -20.1 % | - |
| Driver | 25 Hz | -10.2 | -12.5 % | 1.8 |
| Driver | 25 Hz + weld (after) | -4.1 | -8.0 % | 1.7 |
| 7-iron | 12 Hz (before) | -11.6 | -9.8 % | - |
| 7-iron | 25 Hz | -4.8 | -6.1 % | 0.99 |
| 7-iron | 25 Hz + weld (after) | -3.7 | -5.4 % | 1.36 |

Measured by re-exporting the same-input bundle from the committed runs with one change at a time. Timing now meets the 5 ms acceptance; impact speed does not (3 % limit). Remaining causes: ground yaw slip of the unactuated root (decision 13; an actuated-root diagnostic gives -1.4 ms/-4.9 % and -0.1 ms/-4.0 %) and residual tracking loss. Failed experiments (12 Hz + weld fails the 7-iron face gate at 7.98 deg; a ball-passage release metric picked the straddling segment; leg root regulation, contact parameters, wrench-QP, no controller split) and reproduction commands are in the calculation reference `docs/research/simscape_matching_reference/simscape_matching_reference.tex` (GCV-20 second pass). Fixtures are not regenerated until acceptance and the OSV-10 gate pass.

A full pipeline re-solve with the final code (`5f8ee41fe5`, both changes plus the 7-iron zero-moment-point filter) confirms the re-export: driver peak -3.7 ms and impact speed 47.94 against 51.75 m/s (-7.4 %); 7-iron peak -4.2 ms and 39.36 against 41.05 m/s (-4.1 %); face at impact 2.8 deg and 4.6 deg from the capture. Timing passes and speed does not, so `test_clubhead_speed_peaks_with_the_capture_and_matches_it_at_impact` is a strict expected failure with these numbers (tolerances unchanged) and the committed fixtures stay; the re-solved fixtures would also fail `test_impact_frame_puts_the_clubhead_at_the_ball` (the detected frame is 0.34 m from the ball for the driver and 0.18 m for the 7-iron, limit 0.15 m). Closing the gap needs the ground yaw-moment model of decision 13, outside GCV-20.

### REVIEW.md Sections

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW_DYNAMICS.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

---

## 12. Reference Zero-Moment-Point Diagnostic

### What

Implement model-based Zero-Moment-Point (ZMP) diagnostics (`full_body_simulation.reference_zmp`, `dynamics.reference_zmp`), evaluating whether the momentum rate of the tracked kinematic reference can physically be supported by the contact polygon given ground reaction limits.

### Why

Reveals that tour-average marker trajectories (an ensemble average of many distinct swings) are inherently dynamically inconsistent with unilateral contact mechanics: between 1.0 s and 1.5 s, the reference ZMP leaves the support polygon on 77% of downswing frames. Computing this metric explains why no feedback controller can achieve zero root drift without adjusting the underlying reference motion.

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/receipt.json`](evidence/ground_support/anthro_driver/receipt.json)
- [`evidence/ground_support/anthro_iron/receipt.json`](evidence/ground_support/anthro_iron/receipt.json)

### What Was Tried and Rejected

Attempting to force root stabilization through aggressive PD control gains on the legs ($K_p=900, K_d=60$), which destabilized the simulation and amplified ground reaction spikes to unphysical levels ($>100$ body weights).

### REVIEW.md Sections

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW_DYNAMICS.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

---

## 13. Rejected: Grip-Roll Scan, Closure Fit From the Address, Cart-Table Filter, Fixed-Point and Iterative-Learning Shooting Fits

### What

A consolidated record of five rejected optimization strategies:

1. **1D Grip-Roll Scan**: Searching 1D roll angles about the shaft.
2. **Address-Based Closure Fit**: Re-solving the dual-grip weld transform from address pose.
3. **Cart-Table ZMP Filter**: Filtering center-of-mass trajectory using a simplified cart-table inverted pendulum model.
4. **Fixed-Point Shooting Fit**: Pinning pelvis commands to drifted replay coordinates.
5. **Iterative-Learning Shooting Fit**: Moving pelvis feedforward commands against replay drift with fixed gain.

### Why

- **1D Grip-Roll Scan & Address Closure Fit**: Hand orientation requires a full 3D rotation tensor (`GRIP_ROTATION_DEG`), not a 1D rotation or open-chain address re-solve.
- **Cart-Table ZMP Filter**: The golf swing is dominated by upper-body and club angular momentum rates, which the single-mass cart-table approximation omits. Forcing cart-table compensation shifted the center of mass by 278 mm and worsened marker error to 98 mm.
- **Fixed-Point & Iterative-Learning Shooting Fits**: Diverged monotonically from iteration 0 across all tested gains ($0.7$ and $0.25$) on both driver (error grew from 74.6 mm to 134.8 mm) and 7-iron (112.3 mm to 192.9 mm). Pelvis drift is governed by ground yaw-moment capacity, which cannot be pre-compensated by shifting pelvis position commands without losing marker fit.

### Evidence Receipt

- [`evidence/anthropometry/scan_grip_roll_receipt.json`](evidence/anthropometry/scan_grip_roll_receipt.json)
- [`evidence/ground_support/anthro_driver_fit/receipt.json`](evidence/ground_support/anthro_driver_fit/receipt.json)
- [`evidence/ground_support/anthro_driver_zmp/receipt.json`](evidence/ground_support/anthro_driver_zmp/receipt.json)
- [`evidence/ground_support/anthro_driver_shoot/receipt.json`](evidence/ground_support/anthro_driver_shoot/receipt.json)
- [`evidence/ground_support/anthro_driver_shoot_g025/receipt.json`](evidence/ground_support/anthro_driver_shoot_g025/receipt.json)
- [`evidence/ground_support/anthro_iron_shoot/receipt.json`](evidence/ground_support/anthro_iron_shoot/receipt.json)
- [`evidence/ground_support/anthro_iron_zmp/receipt.json`](evidence/ground_support/anthro_iron_zmp/receipt.json)

### What Was Tried and Rejected

Documented above for each approach with concrete failure receipts.

### REVIEW.md Sections

- [REVIEW.md Section 12: Grip Roll](evidence/anthropometry/REVIEW.md#12-elbow-pits-from-the-markers-grip-roll-club-mesh-epic-2026-09-14)
- [REVIEW.md Section 13: Closure Weld Fitted from the Address](evidence/anthropometry/REVIEW.md#13-closure-weld-fitted-from-the-address-mm-2-2026-09-14)
- [REVIEW.md Section 15: Cart-Table Dynamics Filter](evidence/anthropometry/REVIEW_DYNAMICS.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)
- [REVIEW.md Section 16: Contact-Aware Shooting Fit](evidence/anthropometry/REVIEW_DYNAMICS.md#16-contact-aware-shooting-fit-fb-5-mm-7b-2026-09-14)

---

## 14. MJX Differentiable Optimisation (Windowed)

### What

Formulate whole-body trajectory optimization using MuJoCo MJX in JAX (`mjx_trajectory_optimisation.py`), computing reverse-mode automatic differentiation gradients through contact dynamics to optimize coordinate spline knot corrections over a cost window (up to 1.45–1.65 s).

### Why

Enables end-to-end gradient descent through forward dynamics and contact mechanics. Reduced driver marker error to 1.5 s from 56.1 mm to 40.2 mm (a 28% improvement) and reduced downswing pelvis yaw lag at 1.4 s from $12.3^\circ$ to $2.6^\circ$ with small joint adjustments ($< 0.5^\circ$ at knots).

### Evidence Receipt

- [`evidence/ground_support/anthro_driver/mjx_optimisation_receipt.json`](evidence/ground_support/anthro_driver/mjx_optimisation_receipt.json)
- [`evidence/ground_support/anthro_driver/downswing_mjx_h145_iter4.json`](evidence/ground_support/anthro_driver/downswing_mjx_h145_iter4.json)
- [`evidence/ground_support/anthro_driver/downswing_mjx_h145_iter16.json`](evidence/ground_support/anthro_driver/downswing_mjx_h145_iter16.json)

### What Was Tried and Rejected

Large optimization learning rates ($2 \times 10^{-3}\text{ rad/knot}$), which caused objective instability; unwindowed whole-swing optimization without follow-through contact reconciliation, where late-stage post-impact divergence degraded full-swing performance from iteration 8 onwards. Windowed iterations 4 to 6 are selected for whole-swing replay transfer.

### REVIEW.md Sections

- [REVIEW.md Section 17: Differentiable Trajectory Optimisation with MuJoCo MJX](evidence/anthropometry/REVIEW_DYNAMICS.md#17-differentiable-trajectory-optimisation-with-mujoco-mjx-fb-5-mm-7b-2026-09-14)

---

## 15. Address Foot Progression (OSV-4, #11730)

### What

Foot yaw has no coordinate of its own: it is pelvis yaw plus `hip_rotation_*` (plus a little hip-flexion and knee coupling). `src/shared/python/motion_matching/foot_progression.py` defines the one toe-out angle every report quotes: the signed angle about the up axis from golfer-forward `f = x_t x up` (toward the ball) to the foot long axis, positive = toes turned out (lead foot toward the target, trail foot away), reported per foot. A left-handed golfer mirrors `f` and swaps which foot leads. The marker long axis is the ground projection of `AnkleOut` to the `ToeIn`/`ToeOut` midpoint; a model foot uses calcn to toes. The capture value is the median over the first static window before takeaway (wrist markers stay within 0.03 m of their opening pose). The issue text writes `f = up x x_t`; with the lead foot on the target side that points behind the golfer in both the capture world (toes toward -X, target -Y) and the spec world (toes toward +X, target +Y), so the code uses `x_t x up` ("toward the ball").

`python3 -m scripts.foot_progression_report` prints the capture values. `--foot-progression {off,capture,default,style}` on the ground-support pipeline (`pipeline/cli.py`) makes the choice explicit: `style` is the default since R3 (#12683; see below), `off` keeps the legacy multi-start feet, `capture` seeds and refits each foot to its measured toe-out (a foot whose markers are missing or unstable gets the 20 degree default, flagged `is_default`), and `default` forces 20 degrees per foot. The receipt's `address.foot_progression` block holds, per foot, target, model angle, error, the capture measurement (headline, raw binding definition, forefoot cross-check, window frames, spread) and a decimated model time series with the finish value.

**How It Works**

1. The shared address seed (`address_feet.seed_document_feet`) solves `hip_rotation_*` against forward kinematics at pelvis yaw 0 and writes them into the document's `address_seed_deg`, so every engine that reads the seed (MuJoCo, Drake, Pinocchio, MyoSuite) starts from the same feet. OpenSim starts from `tour_matching/address.py` (`toe_out_deg=`), solved against OpenSim forward kinematics because its hip rotation is mirrored.
2. After the multi-start address fit, `refine_foot_progression` moves `hip_rotation_*` by the measured sensitivity (central difference, sign-safe) and re-solves with a prior weight of 1000 on the two hip-rotation coordinates. The prior is soft: the markers can overrule it; the stance spheres stay pinned flat by the solver. It converges to 0.5 degrees (the issue asks for 2).
3. The per-engine multi-start `hip_rotation` seeds of -20/0/+20 are not applied when the option is on.
4. The swing keeps the calibrated address feet (#12222). Every trajectory solve plants its stance spheres where they first touch (`plant_stance`); with the option on, `address_feet.planted_stance` also sets `anchor_stance_at_init`, so the plant starts at the solve's start pose instead of the first frame's own fit. This applies to the swing IK (which starts at the calibrated address), the consistency re-solve, the shooting refit and the ZMP filter, so each pass keeps the feet of its input's first frame. Without it, each pass re-fitted frame 0 and the 7-iron trail toe-out drifted −0.88 → −1.40 → −2.23 degrees across the three passes (target −0.37). With the option off, the anchors still come from the first frame's fit, so `off` receipts are unchanged.

### Why

**Findings On the Strange Feet**

- **The spec's left hip rotation axis was not mirrored.** `+hip_rotation_r` turns the right foot in (9.6 degrees for 10), `+hip_rotation_l` turned the left foot out (10.4 degrees for 10); OpenSim turns both in. The equal-sign multi-start seeds (`ADDRESS_SEEDS_DEG`, +/-20) therefore splay the feet by up to 40 degrees from each other, and since calibrated foot-marker offsets make yaw a null mode of the marker fit, the lowest-RMS seed decides the feet. OSV-6 (#11737) mirrors it like OpenSim (`docs/research/hip_axis_mirroring/`); seeding stays sign-safe.
- **The stock toe marker seeds carry a 12.5 degree yaw bias.** `RToeIn`/`RToeOut` (and the left mirror) on `calcn` stagger `ToeIn` 0.02 m ahead of `ToeOut` over a 0.09 m span. In the C3D the `ToeIn`-`ToeOut` line is perpendicular to the foot axis within a few degrees (forefoot-line and malleolus-corrected ankle-toe estimates agree within about 4 degrees), so a marker fit against the stock seeds turns the model foot by that bias. The left and right seeds are exact mirrors (a unit test pins that), so this is a bias, not a mirroring error. `square_forefoot_seeds` gives both toe markers the mean forward offset; it applies only with `--foot-progression` on.
- **`AnkleOut` is the lateral malleolus**, about 0.04 m outside the heel line, so the unmodified ankle-toe axis reads about 13 degrees toe-in on a straight foot. The headline capture angle moves the heel proxy medially by 0.04 m and the raw binding value is kept alongside for traceability.

**Measured Capture Values (Degrees of Toe-Out at Address)**

| Capture | Lead (left) | Trail (right) |
| --- | --- | --- |
| Tour driver | 16.4 | 4.0 |
| Tour 7-iron | 15.0 | -0.4 |

The tour players are not at 20 degrees per foot (the trail foot is close to square), so 20 degrees is only the fallback for unreliable markers. The owner's capture O value is not measured here: its C3D is private and was not present on the implementing host; run `python3 -m scripts.foot_progression_report` with `CAPTURE_DATA_DIR` set. Until then it takes the flagged 20 degree default.

### What Was Tried and Rejected

**Measured outcome on the tour driver capture (MuJoCo, full pipeline address stage).** Legacy seed: lead 41.7, trail -2.8 degrees (capture 16.4 and 4.0). With `--foot-progression capture`: lead 37.0, trail 4.05 degrees. The trail foot reaches the capture; the lead foot does not, because `hip_rotation_l` reaches its -40 degree limit while the marker fit's leg posture (hip adduction -14.6 degrees) contributes the rest of the yaw (sensitivity about 0.85 degrees of foot yaw per degree of hip rotation, so about -64 degrees would be needed). The lead-foot target is therefore NOT met in the full fit; the shared FK seed alone reaches both feet within 2 degrees (unit-tested). An address solve that also trades hip adduction against stance-width markers is the open follow-up (still 15-32 degrees off after OSV-6).

While a foot is in stance, its yaw is held at the address value (step 4). Once it leaves stance, foot yaw is not constrained: it follows the matched solution and contact (the balance work in #11667 handles slip). The time series and the finish value are reported, not controlled. In the follow-through, the lead foot turns to about 82 degrees and the trail foot to between −133 and −158 degrees.

**Club-Face Fixtures (#12222).** `scripts/regenerate_club_face_fixtures.py` runs with `--foot-progression capture`. It commits each run's hip-calibrated, scaled document as `tests/fixtures/club_face/spec_<club>.json`, because the calibrated hip frames set the foot yaw: the raw document's FK of the same pose turns the lead foot 29–30 degrees and lifts it 10–11 cm off the floor. Fixture frame 0 against the capture (MuJoCo FK): driver 16.64 / 3.57 against 16.37 / 4.03 degrees, 7-iron 15.43 / −0.96 against 15.00 / −0.37 degrees. `tests/unit/motion_matching/test_club_face_fixture_feet.py` holds them to 2 degrees and the feet to 5 cm.

**Trail-Foot Style Target (R3, #12683).** The owner chose decision B on #12678 (2026-10-15): about 20 degrees of trail toe-out at address, as a style target recorded as owner-set, not measured. `foot_progression.TRAIL_TOE_OUT_STYLE_DEG` holds it. It is defined from `DEFAULT_TOE_OUT_DEG`, so there is one literal. `resolve_foot_target` applies it:

- In `style` mode, the trail foot gets the style target and the lead foot resolves exactly as in `capture` mode.
- In `capture` mode, a reliable measurement wins for both feet. This is bit-for-bit the previous resolver.
- Each receipt foot entry carries `target_source` (`capture`, `default` or `style`).
- `--foot-progression` now defaults to `style` (`address_feet.CANONICAL_FOOT_PROGRESSION`), so the canonical commands turn the trail foot out.

Measured calibrated address, trail toe-out, before (`off`) / after (`style`):

| Capture | Before (`off`) | After (`style`) | Note |
| --- | --- | --- | --- |
| capture-A driver | -2.93 deg | **19.80 deg** | Address RMS 10.95 → 10.54 mm |
| capture-B 7-iron | -9.81 deg | **13.53 deg** | `hip_rotation_r` on its -40 deg limit; address RMS 8.42 → 12.19 mm |

The 7-iron therefore misses the target by 6.5 degrees. The hip range was not widened. Reaching 20 degrees there needs an owner decision: widen the range, accept the shortfall, or take the yaw from elsewhere in the leg chain, re-measured after R4's knee flex. The lead foot matches the `capture`-mode lead within 0.25 degrees on both captures. The evidence is in [`evidence/foot_progression/trail_toe_out_12678/README.md`](evidence/foot_progression/trail_toe_out_12678/README.md). The upstreamdrift design manual has no address-stage chapter or registered calculation, so this section remains the calculation-level reference.

### Evidence Receipt

- [`evidence/foot_progression/capture_report.json`](evidence/foot_progression/capture_report.json)
- [`evidence/foot_progression/trail_toe_out_12678/address_toe_out_summary.json`](evidence/foot_progression/trail_toe_out_12678/address_toe_out_summary.json)
- Tests: `tests/unit/motion_matching/test_foot_progression.py`, `tests/unit/motion_matching/pipeline/test_address_feet.py`, `tests/opensim/test_golf_address.py`, `tests/unit/motion_matching/test_club_face_fixture_feet.py`, `tests/unit/motion_matching/test_full_body_markers.py::test_stance_anchored_at_the_start_pose_holds_the_first_frame`.

Section 16 on: [volume 2](DESIGN_DECISIONS_2.md).
