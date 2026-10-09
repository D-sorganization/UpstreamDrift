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

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

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

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

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

- [REVIEW.md Section 15: Downswing Dynamics: Compliant Sole](evidence/anthropometry/REVIEW.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)

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
- [REVIEW.md Section 15: Cart-Table Dynamics Filter](evidence/anthropometry/REVIEW.md#15-downswing-dynamics-compliant-sole-tracked-reference-zero-moment-point-mm-7-2026-09-14)
- [REVIEW.md Section 16: Contact-Aware Shooting Fit](evidence/anthropometry/REVIEW.md#16-contact-aware-shooting-fit-fb-5-mm-7b-2026-09-14)

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

- [REVIEW.md Section 17: Differentiable Trajectory Optimisation with MuJoCo MJX](evidence/anthropometry/REVIEW.md#17-differentiable-trajectory-optimisation-with-mujoco-mjx-fb-5-mm-7b-2026-09-14)

---

## 15. Address Foot Progression (OSV-4, #11730)

### What

Foot yaw has no coordinate of its own: it is pelvis yaw plus `hip_rotation_*` (plus a little hip-flexion and knee coupling). `src/shared/python/motion_matching/foot_progression.py` defines the one toe-out angle every report quotes: the signed angle about the up axis from golfer-forward `f = x_t x up` (toward the ball) to the foot long axis, positive = toes turned out (lead foot toward the target, trail foot away), reported per foot. A left-handed golfer mirrors `f` and swaps which foot leads. The marker long axis is the ground projection of `AnkleOut` to the `ToeIn`/`ToeOut` midpoint; a model foot uses calcn to toes. The capture value is the median over the first static window before takeaway (wrist markers stay within 0.03 m of their opening pose). The issue text writes `f = up x x_t`; with the lead foot on the target side that points behind the golfer in both the capture world (toes toward -X, target -Y) and the spec world (toes toward +X, target +Y), so the code uses `x_t x up` ("toward the ball").

`python3 -m scripts.foot_progression_report` prints the capture values. `--foot-progression {off,capture,default}` on the ground-support pipeline (`pipeline/cli.py`) makes the choice explicit: `off` (default) keeps every existing receipt reproducible, `capture` seeds and refits each foot to its measured toe-out (a foot whose markers are missing or unstable gets the 20 degree default, flagged `is_default`), and `default` forces 20 degrees per foot. The receipt's `address.foot_progression` block holds, per foot, target, model angle, error, the capture measurement (headline, raw binding definition, forefoot cross-check, window frames, spread) and a decimated model time series with the finish value.

**How It Works**

1. The shared address seed (`address_feet.seed_document_feet`) solves `hip_rotation_*` against forward kinematics at pelvis yaw 0 and writes them into the document's `address_seed_deg`, so every engine that reads the seed (MuJoCo, Drake, Pinocchio, MyoSuite) starts from the same feet. OpenSim starts from `tour_matching/address.py` (`toe_out_deg=`), solved against OpenSim forward kinematics because its hip rotation is mirrored.
2. After the multi-start address fit, `refine_foot_progression` moves `hip_rotation_*` by the measured sensitivity (central difference, sign-safe) and re-solves with a prior weight of 1000 on the two hip-rotation coordinates. The prior is soft: the markers can overrule it; the stance spheres stay pinned flat by the solver. It converges to 0.5 degrees (the issue asks for 2).
3. The per-engine multi-start `hip_rotation` seeds of -20/0/+20 are not applied when the option is on.

### Why

**Findings On the Strange Feet**

- **The spec's left hip rotation axis is not mirrored.** `+hip_rotation_r` turns the right foot in (9.6 degrees for 10), `+hip_rotation_l` turns the left foot out (10.4 degrees for 10); OpenSim turns both in. The equal-sign multi-start seeds (`ADDRESS_SEEDS_DEG`, +/-20) therefore splay the feet by up to 40 degrees from each other, and since calibrated foot-marker offsets make yaw a null mode of the marker fit, the lowest-RMS seed decides the feet. The coordinate meaning is left as is (qualified receipts depend on it); seeding is sign-safe by solving against forward kinematics, and a unit test pins the finding.
- **The stock toe marker seeds carry a 12.5 degree yaw bias.** `RToeIn`/`RToeOut` (and the left mirror) on `calcn` stagger `ToeIn` 0.02 m ahead of `ToeOut` over a 0.09 m span. In the C3D the `ToeIn`-`ToeOut` line is perpendicular to the foot axis within a few degrees (forefoot-line and malleolus-corrected ankle-toe estimates agree within about 4 degrees), so a marker fit against the stock seeds turns the model foot by that bias. The left and right seeds are exact mirrors (a unit test pins that), so this is a bias, not a mirroring error. `square_forefoot_seeds` gives both toe markers the mean forward offset; it applies only with `--foot-progression` on.
- **`AnkleOut` is the lateral malleolus**, about 0.04 m outside the heel line, so the unmodified ankle-toe axis reads about 13 degrees toe-in on a straight foot. The headline capture angle moves the heel proxy medially by 0.04 m and the raw binding value is kept alongside for traceability.

**Measured Capture Values (Degrees of Toe-Out at Address)**

| Capture | Lead (left) | Trail (right) |
| --- | --- | --- |
| Tour driver | 16.4 | 4.0 |
| Tour 7-iron | 15.0 | -0.4 |

The tour players are not at 20 degrees per foot (the trail foot is close to square), so 20 degrees is only the fallback for unreliable markers. The owner's capture O value is not measured here: its C3D is private and was not present on the implementing host; run `python3 -m scripts.foot_progression_report` with `CAPTURE_DATA_DIR` set. Until then it takes the flagged 20 degree default.

### What Was Tried and Rejected

**Measured outcome on the tour driver capture (MuJoCo, full pipeline address stage).** Legacy seed: lead 41.7, trail -2.8 degrees (capture 16.4 and 4.0). With `--foot-progression capture`: lead 37.0, trail 4.05 degrees. The trail foot reaches the capture; the lead foot does not, because `hip_rotation_l` reaches its -40 degree limit while the marker fit's leg posture (hip adduction -14.6 degrees) contributes the rest of the yaw (sensitivity about 0.85 degrees of foot yaw per degree of hip rotation, so about -64 degrees would be needed). The lead-foot target is therefore NOT met in the full fit; the shared FK seed alone reaches both feet within 2 degrees (unit-tested). An address solve that also trades hip adduction against stance-width markers is the open follow-up.

Foot yaw through the swing is not constrained; it follows the matched solution and contact (the balance work in #11667 handles slip). The time series and the finish value are reported, not controlled.

### Evidence Receipt

- [`evidence/foot_progression/capture_report.json`](evidence/foot_progression/capture_report.json)
- Tests: `tests/unit/motion_matching/test_foot_progression.py`, `tests/unit/motion_matching/pipeline/test_address_feet.py`, `tests/opensim/test_golf_address.py`.

## 16. Compliant Bushing Grip Model (OSV-7 Phase 1)

### What

An engine-agnostic grip interface (`src/shared/python/grip_contact/`) and an OpenSim `grip_model="bushing"` option (`full_body_osim.export_full_body_osim`, `full_body_grip_topology.py`, `grip_bushing_sim.py`). `weld` stays the default and its output is byte-identical to before; `contact` raises `NotImplementedError` (phase 2, #11739).

Model:

- Club: a free body (six coordinates, `ClubFree*`) with unchanged mass and inertia. The left hand solids move to a new `LGrip` hand body and the right hand solid to the right standoff body; every solid keeps its mass and inertia, so total mass is conserved.
- One `BushingForce` per hand between a hand-body frame and a club frame. Both frames come from the spec's existing grip closure and club geometry, with no invented offsets. Grip-frame x is the shaft axis; y and z are across the grip. The right hand frame is tied to the left hand body at the weld-model location (one hand defines the club, see Input Kinematics).
- Wrench: $\mathbf{F} = -K_t\,\boldsymbol{\delta} - C_t\,\dot{\boldsymbol{\delta}}$ and $\mathbf{M} = -K_r\,\boldsymbol{\theta} - C_r\,\dot{\boldsymbol{\theta}}$ per axis in the hand frame, converted to force and couple ON THE CLUB at the grip point, $\boldsymbol{\tau} = \mathbf{M} - (\mathbf{p}-\mathbf{o})\times\mathbf{F}$.
- Stiffness defaults (`default_bushing()`): $K_t = 10^6$ N/m, $K_r = 1600$ N m/rad per axis. Unfitted engineering defaults for a stiff surrogate of the weld. Only the existence, authors and venue of the three cited papers (Serina et al. 1997, Wu et al. 2004, Komi et al. 2008) were confirmed by search; no number is sourced from them, and an earlier 0.5 to 3 N/mm pulp figure and a FingerTPS reference were withdrawn.
- Damping is designed, not tuned (`grip_contact/damping.py`). See Damping Design.
- Internal force (`decompose_hand_forces`): net $F_L + F_R$; internal $(F_L - F_R)/2$, split along the inter-hand line into a squeeze part and a transverse part. A transverse internal force is a force pair carrying a couple $M = F_t d$ with $d = 0.076$ m.
- Wrench split: `analyze_grip(split_method="bushing")` in `biomechanics/grip_wrench.py`; `allocate_min_norm` is the labelled weld proxy because the weld split is indeterminate.

### Why

A rigid weld cannot report per-hand forces. A finite stiffness makes the split determinate while keeping the club dynamics essentially those of the weld.

### Review Findings (PR #11765) and What Changed

The first run reported 3 to 6.6 kN per hand against 0.5 to 1.8 kN net, ringing at 30 to 40 Hz, and 3 to 6 mm deflection. The diagnosis:

1. **Internal force is a couple, not a squeeze.** Of the 6.0 kN internal peak, 6.0 kN was transverse (a force pair carrying up to 484 N m) and only 0.46 kN was axial squeeze. It is the club couple divided by the 76 mm hand spacing.
2. **Input kinematics, not the bushing, caused the load.** The OpenSim IK candidate (`anthro_driver_opensim/candidate.npz`) has a marker RMS of 262 mm (the MuJoCo canonical IK is 34 to 52 mm). Its model wrist markers reach 95 m/s while the measured wrist markers peak at 9.7 m/s (p95 7.1 to 8.1). It contains 15 frames with steps of 1 to 3 rad in one 2.8 ms frame (median step 0.001 rad), at 0.947, 1.231, 1.258, 1.436, 1.544 s and others; several are persistent IK solution-branch switches, and ten coordinates also carry 2 pi branch flips. The two-hand loop is open by 134 mm and 51 degrees at address and 180 mm and up to 180 degrees later, so the right arm chain cannot be prescribed. The 25 Hz zero-phase filter then spreads each step backwards in time as a precursor load.
3. **Damping was a per-axis guess.** The lowest modes are club pendulum modes at 23.7 and 24.7 Hz (stiffness $2k_r + k_t d^2/2$ of the force pair, mass 0.8 m from the hands). The old coefficients left them at modal $\zeta = 0.21$ (nominal 0.3), which is the ringing seen.

Fixes (all in the shared pipeline, none by tuning the bushing):

- `grip_contact.trajectory_conditioning`: `unwrap_angular` (lossless), `detect_ik_outliers` plus `condition_trajectory` (PCHIP over frames where at least two angles leave a running median by more than 0.35 rad; 7 frames repaired), and `first_discontinuity_time` (0.95 s). Persistent branch switches cannot be interpolated, so the motion is cut before the first one, before filtering. The full 1.8 s is retained as a separate receipt and is reported as failing.
- Closure consistency: one hand defines the club. The right bushing frame is fixed on the left hand body at the weld location, so the two bushings cannot be stretched against each other by inconsistent hand kinematics; the right arm chain of the IK is not used. `input_kinematics_report` records the residual, hand speeds and measured wrist speeds in the receipt.
- Damping Design: see below.

### Damping Design

Small motion of the rigid club about its centre of mass, state $(u, \theta)$. Bushing $i$ at $r_i$ with frame $R_i$ sees $A_i x$, $A_i = [[I, -[r_i]_\times],[0, I]]$, so $K = \sum_i A_i^T \mathrm{diag}(R_i K_b R_i^T) A_i$ and $C$ likewise with the coefficients. Mass and inertia come from the spec club solids (0.313 kg, centre of mass 0.256 m from the head, hands excluded). For the undamped modes $\phi_j$ the modal ratio $\zeta_j = \phi_j^T C \phi_j / (2 \omega_j \phi_j^T M \phi_j)$ is linear in the six per-axis coefficients, so setting all six to the requested $\zeta$ is a 6 by 6 non-negative linear solve (`design_damping`). The requested value is $\zeta = 0.7$ (about 5 % overshoot, fastest settling without sustained ringing); nothing else is chosen. Result: $c_t = (554, 209, 212)$ N s/m, $c_r = (0.81, 28.6, 27.3)$ N m s/rad, all six modes at $\zeta = 0.700$ (frequencies 23.7, 24.7, 402, 460, 931, 946 Hz). `test_free_vibration_decays_at_the_stated_ratio` integrates the linear model and recovers the logarithmic-decrement ratio within 3 %.

### Evidence Receipt

- [`evidence/grip_kinetics/receipt.json`](evidence/grip_kinetics/receipt.json), [`driver_bushing_series.npz`](evidence/grip_kinetics/driver_bushing_series.npz): valid window (0 to 0.94 s).
- [`receipt_full_window_conditioned.json`](evidence/grip_kinetics/receipt_full_window_conditioned.json), [`driver_bushing_series_full_window.npz`](evidence/grip_kinetics/driver_bushing_series_full_window.npz): full 1.8 s with the conditioned input.
- [`receipt_before_fix.json`](evidence/grip_kinetics/receipt_before_fix.json): first run, kept for the before and after comparison.
- Reproduce: `PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_grip_kinetics.py --accuracy 1e-3 --sensitivity 0.1 10`
- Clips and plots (outside the repository): `~/Videos/Parity Audit/golfer_realism/grip_kinetics/` (the `_full_window` files show the failing window).

Before and after (default stiffness):

| Quantity | First run (to 1.3 s) | Valid window (0 to 0.94 s) | Full 1.8 s, conditioned |
| --- | --- | --- | --- |
| Peak per-hand force L / R (N) | 5466 / 6626 | 142 / 162 | 21058 / 27981 |
| Peak net force (N) | 1830 | 28 | 22501 |
| Peak internal force (N) | 6038 | 151 | 23761 |
| Squeeze part / couple (N, N m) | 460 / 484 | 1.3 / 12.1 | 5861 / 1850 |
| Max deflection L / R (mm) | 5.5 / 6.7 | 0.14 / 0.16 | 19.9 / 26.8 |
| Max rotation (deg) | 8.9 | 0.22 | 37 |
| Modal damping ratio (lowest modes) | 0.21 | 0.70 | 0.70 |
| Static hold error | 4.4e-5 | 1.1e-5 | n/a |

Stiffness sensitivity on the valid window: $\times 0.1$ gives 84 / 98 N, 0.84 / 0.98 mm and 1.3 degrees; $\times 10$ gives 159 / 180 N, 0.016 / 0.018 mm and 0.02 degrees. Left share of the hand force: 46.4 % at peak and 46.1 % median with the bushing, against 48.4 % and 48.1 % for the minimum-norm weld proxy.

Acceptance bounds (never loosened):

- Deflection at most 3 mm and 2 degrees (owner-reviewable): met in the valid window; the full-window test is a strict xfail.
- Internal force at most 500 N (superseded by the squeeze and couple-consistency checks of section 17): a 38 N m couple at 76 mm, well above the 12 N m seen in the valid window and consistent with the review statement that per-hand grip forces near impact are a few hundred newtons. Met in the valid window; the full-window test is a strict xfail.

### What Was Tried and Rejected

- Hunt-Crossley sphere-sphere and sphere-cylinder contact: no force produced; open meshes fail. Deferred to phase 2.
- Closing the weld loop with `assemble`: the IK leaves the loop open, so assembly produced an 84 kN preload.
- Per-axis damping $2\zeta\sqrt{km}$ with the club mass: leaves the pendulum modes underdamped (see Review Findings).
- Spike repair plus filtering of the whole 1.8 s: the filter spreads the persistent branch switches, and peak forces rose to 21 to 28 kN. Replaced by cutting before the first switch.
- Applying the Drake or MuJoCo dynamics and IK records to the OpenSim model: their coordinate conventions differ (z-up, 41 or 44 differently ordered columns) and the closure residual was 1.2 m, so no cross-engine record could be substituted.

### Limitations

- (Superseded by section 17, which drives the full window from the closure-consistent fits.) With the IK candidate, the kinetics are valid only for the 0 to 0.94 s window (backswing and transition). The downswing, impact and finish are not covered because the committed OpenSim IK candidate is not a qualified motion. The needed work is a closure-consistent, marker-qualified IK (marker RMS near 30 to 50 mm) in the OpenSim coordinate convention, then a re-run of the receipt; until then no impact-phase hand-force number should be presented.
- The right hand arm chain of the input is ignored (open loop); the split is a property of this approximation.
- Stiffness defaults and the deflection and internal-force bounds are owner-reviewable engineering values; the three citations were confirmed to exist but support no number.
- Explicit stiff integration of the full 1.8 s takes about 35 minutes at accuracy 1e-3.
- The bushing amplification of the couple (12 to 13 % at default stiffness) is a modelling result, not a validated one; whether `K_r` should change is an owner decision and nothing is tuned here.
- Software correctness only. Scientific qualification stays in the design-manual governance pathway.
- Out of scope for phase 1: the contact model, MuJoCo, Drake and Pinocchio parity, and the `golf_humanoid.osim` builder.

## 17. Impact-Phase Bushing Grip From the Closure-Consistent Fits (OSV-7)

### What

The section 16 bushing simulation is re-driven over the full 0 to 1.8 s window from the OSV-10 fitted swings (`tests/fixtures/club_face/swing_q_{driver,iron7}.npz`: the 1 kHz same-input reference of the shared ground-support fit, every second sample, IK marker RMS 33.6 mm for the driver and 31.6 mm for the iron, replayed identically in every engine). The columns are mapped by coordinate name onto the committed `full_body_spec_anthro_<club>.json` with `grip_contact.load_coordinate_swing` and `map_coordinates`. The mapping is the identity here but checked: a missing, duplicated or unused coordinate raises `ValueError`. The coordinates are used as committed, unfiltered and unrepaired (`condition_trajectory` finds nothing to repair; the 25 Hz filter changes peak forces by under 4 % and opens the loop slightly). The model, stiffness, damping (zeta = 0.70) and bounds are unchanged.

### Input Kinematics (Reported First)

| | Driver (capture A) | 7-iron (capture B) |
| --- | --- | --- |
| Hand-loop closure, max over swing | 5.0e-5 mm, 8e-6 deg | 1.7e-4 mm, 2e-5 deg |
| Model left grip-point speed, peak | 9.63 m/s at 1.244 s | 9.16 m/s at 1.262 s |
| Model right closure-frame speed, peak | 8.47 m/s at 1.238 s | 8.75 m/s at 1.296 s |
| Measured left / right wrist-marker speed, peak | 9.73 / 9.58 m/s | 9.30 / 9.01 m/s |

The loop is closed to float32 precision and hand speeds match the wrist markers within 1 to 12 %, so the input is credible for kinetics.

### Integrator

The integrator is OpenSim `Manager` Runge-Kutta-Merson (explicit, error-controlled) at accuracy 1e-3, sampled at the 2 ms fixture times, taking 1.5 to 7 minutes per swing. Convergence of the peaks against RK-Merson at 1e-5 and implicit CPodes at 1e-4: driver internal force 509.8 / 509.8 / 511.2 N, iron 524.1 / 524.1 / 526.3 N; deflections agree within 0.5 %.

### Results (Hand Acting on the Club)

| Quantity | Driver | 7-iron |
| --- | --- | --- |
| Impact time (shared `club_face.impact_frame`) | 1.326 s | 1.336 s |
| Lead (L) / trail (R) force at impact | 506 / 530 N | 517 / 552 N |
| Lead share at impact | 48.9 % | 48.4 % |
| Net force at impact (peak) | 332 N (439 N at 1.296 s) | 360 N (457 N) |
| Internal force at impact (full-window peak) | 491 N (510 N at 1.322 s) | 503 N (524 N at 1.332 s) |
| Squeeze part of the internal force at impact | 2.2 N | 3.9 N |
| Force-pair couple / equivalent couple at midpoint, at impact | 39.4 / 71.2 N m | 40.4 / 71.7 N m |
| Peak equivalent couple at midpoint | 99.6 N m at 1.314 s | 98.6 N m at 1.324 s |
| Free torque per hand at impact | 18.6 N m | 17.0 N m |
| Peak per-hand force L / R | 515 / 564 N | 533 / 584 N |
| Max deflection | 0.56 mm, 0.84 deg | 0.58 mm, 0.84 deg |
| Window 0 to 1.30 s: peak internal force | 364 N | 293 N |

Acceptance against the unchanged bounds:

- Deflection (3 mm, 2 deg): met over the full window for both clubs. `test_full_swing_deflection_within_limits` passes.
- Internal force: the flat 500 N bound is replaced by the two physics checks below (owner decision on PR #11774). Both pass over 0 to 1.8 s for both clubs, so `test_full_swing_internal_force_is_physical` is a plain test, no longer a strict xfail.

| Internal-force check, 0 to 1.8 s | Driver | 7-iron |
| --- | --- | --- |
| Peak squeeze (bound 50 N) | 3.3 N | 3.9 N |
| Peak transverse internal force | 509.8 N at 1.322 s | 524.1 N at 1.332 s |
| Couple/d prediction at that sample | 509.8 N | 524.1 N |
| Max relative error, checked samples (bound 5 %) | 6e-14 | 2e-13 |
| Samples above the 2 N m noise floor | 81 % | 79 % |
| Hand-moment peak, realised club motion | 99.6 N m at 1.314 s | 98.6 N m at 1.324 s |
| Hand-moment peak, club welded to the prescribed hand | 89.1 N m at 1.312 s | 87.0 N m at 1.322 s |

### Sanity Check Against Published Magnitudes

- Nesbit (2005), J Sports Sci Med 4(4):499-519, Table 3 (85 golfers): the golfer-club linear force at impact averages 397.5 N (range 300 to 490 N). The model's net force at impact, 332 and 360 N, and its peaks, 439 and 457 N, lie inside that range.
- Grober (2020), arXiv:2006.11778, section VIII, quotes MacKenzie's instrumented-grip data for one golfer in the last frame before impact: F = 456 N, moment of force 55.8 N m, couple -59.1 N m. Grober notes that a 50 N m couple at a 1/6 m hand spacing needs about 300 N per hand. The model's 71 N m at impact (peak 99.6 N m) is the same order but larger, and its 80.3 mm spacing needs a larger internal force. This is an open item on #11739, not a test: one golfer is not a bound, and the model's couple includes the bushing amplification below.

### Why

Why the flat internal-force bound was replaced: of the internal force at impact, 99.9 % is transverse: a force pair carrying the club couple, with a squeeze of only 2 to 4 N. The flat 500 N bound (section 16, set for a 38 N m backswing couple) was mis-specified: it could not tell a real couple-carrying pair from the "fighting hands" artefact of inconsistent kinematics; both raise `|F_int|`. At impact the published couple of about 60 N m alone implies roughly 750 N at 80.3 mm if the free torques carried nothing (the measured spacing is 80.32 mm, not the 76 mm quoted in section 16).

The replacement lives in the shared `grip_contact.couple_check` module and uses only `decompose_hand_forces`:

1. **Squeeze.** The axial internal force along the inter-hand line `u` (positive in compression) stays within 50 N over 0 to 1.8 s. The club needs no squeeze, so a large one means the hands are pulled against each other.
2. **Couple consistency.** With `P` the hand midpoint, `d = |p_R - p_L|` and the hand-acting-on-the-club convention, Newton-Euler of the club gives the moment the hands must apply about `P`: `M_hands,P = I w' + w x (I w) + (c - P) x m (a_c - g)`. The net hand force has no moment about the midpoint, so the contact-force moment is carried by the internal pair alone: `M_contact = M_hands,P - tau_L - tau_R = -d u x F_int`. The predicted transverse pair is `|M_contact,perp| / d` (the component normal to `u`), and it must equal the measured `|F_int,perp|` within 5 %. Samples with `|M_contact,perp|` below 2 N m (2 % of the swing peak, about the address and backswing level) are skipped, because the relative error of a vanishing couple is meaningless; 79 to 81 % of the samples are checked, including the peak.

Scope of the couple check. It uses the engine's realised accelerations (`realizeAcceleration`, not finite differences). So it is a Newton-Euler closure that agrees to round-off: it proves that the extracted per-hand wrenches, frames, signs and spec inertia account for the club's motion and that the transverse pair is couple-carrying. It does not prove that the club follows the input swing; for that, the same function on the club welded to the prescribed lead hand gives the moment the swing demands: 89.1 N m (driver) and 87.0 N m (iron) against the realised 99.6 and 98.6 N m. The bushing amplifies the couple dynamically by 12 to 13 %; with stiffness x10 the driver peak falls to 90.7 N m (internal 530 N, squeeze 1.4 N). The internal force barely changes with stiffness; it is set by the couple and `d`.

### What Was Tried and Rejected

- The flat 500 N internal-force bound (see Why).
- The OpenSim IK candidate as input: 134 mm loop gap, 95 m/s hands.
- Finite-difference club accelerations: 2 ms samples alias the 400 to 950 Hz bushing modes (errors up to 96 %).

### Evidence Receipt

- [`evidence/grip_kinetics/receipt_full_swing_driver.json`](evidence/grip_kinetics/receipt_full_swing_driver.json) and [`receipt_full_swing_iron7.json`](evidence/grip_kinetics/receipt_full_swing_iron7.json), with [`driver_bushing_series_full_swing.npz`](evidence/grip_kinetics/driver_bushing_series_full_swing.npz) and [`iron7_bushing_series_full_swing.npz`](evidence/grip_kinetics/iron7_bushing_series_full_swing.npz).
- Reproduce: `CAPTURE_DATA_DIR=... MPLBACKEND=Agg PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_full_swing_grip_kinetics.py --club driver --convergence` (and `--club iron7`).
- Plots and 0.5x hands close-ups (outside the repository): `~/Videos/Parity Audit/golfer_realism/grip_kinetics/full_swing/`.

### Limitations

- Both bushings' hand frames sit on the left hand body (section 16); with a closed loop this is kinematically the same as prescribing the right arm. `K_r` is equal per hand, so the 50/50 free-torque split is by construction, not a measurement.
- The couple and internal force oscillate at about 15 Hz from 1.25 to 1.40 s. This is in the fitted wrist kinematics (it survives the 25 Hz filter) and is not validated against measured club angular acceleration.
- The fitted swing has no ball. The impact metrics are those of the club passing through the ball position, not of the collision.
- Software correctness only. Scientific qualification stays in the design-manual governance pathway. MuJoCo, Drake and Pinocchio bushing parity and the contact model remain open (#11739).
