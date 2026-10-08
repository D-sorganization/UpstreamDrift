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
- One `BushingForce` per hand between a hand-body frame and a club frame. Both frames come from the spec's existing grip closure and club geometry, with no invented offsets. Grip-frame x is the shaft axis; y and z are across the grip.
- Wrench: $\mathbf{F} = -K_t\,\boldsymbol{\delta} - C_t\,\dot{\boldsymbol{\delta}}$ and $\mathbf{M} = -K_r\,\boldsymbol{\theta} - C_r\,\dot{\boldsymbol{\theta}}$ per axis, in the hand frame. The record is converted to force and couple ON THE CLUB at the grip point, $\boldsymbol{\tau} = \mathbf{M} - (\mathbf{p}-\mathbf{o})\times\mathbf{F}$, because OpenSim reports the moment about the club body origin.
- Defaults (`default_bushing()`): $K_t = 10^6$ N/m, $K_r = 1600$ N m/rad per axis, damping ratio $\zeta = 0.3$. These are engineering defaults with cited scale brackets (fingertip pulp and Komi grip-force sources in the module docstring), never fitted. The citations are from the literature record and were not re-verified against full texts.
- Wrench split: `analyze_grip(split_method="bushing")` in `biomechanics/grip_wrench.py`. The weld split is indeterminate, so `allocate_min_norm` (zero free torque, minimum norm, equal axial split) is the labelled weld proxy.

### Why

A rigid weld cannot report per-hand forces. A finite stiffness makes the split determinate and measurable while keeping the club dynamics essentially those of the weld.

### Evidence Receipt

- [`evidence/grip_kinetics/receipt.json`](evidence/grip_kinetics/receipt.json) and [`evidence/grip_kinetics/driver_bushing_series.npz`](evidence/grip_kinetics/driver_bushing_series.npz)
- Reproduce: `PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_grip_kinetics.py --accuracy 1e-3 --t-end 1.3 --sensitivity 0.1 10`
- Clip and plot (outside the repository): `~/Videos/Parity Audit/golfer_realism/grip_kinetics/`
- Tests: `tests/unit/grip_contact/`, `tests/unit/motion_matching/test_full_body_osim_grip_models.py`, `tests/unit/motion_matching/test_grip_bushing_opensim.py`

Results (driver candidate, 25 Hz zero-phase prefilter, window 0 to 1.3 s):

- Static hold: the sum of hand forces is 3.0693 N against a club weight of 3.0695 N, a relative error of $4.4\times10^{-5}$ (limit 1%). Moment balance about the grip point holds.
- $F = K\delta$ per axis holds to numerical precision (`probe_deflection`).
- Driven swing, default stiffness: to 0.9 s the maximum deflection is 0.25 mm and 0.34 deg. Over the full window to 1.3 s it is 5.5 mm / 6.7 mm (left / right) and 8.9 deg, so the owner limits of 3 mm and 2 deg are NOT met (peak 5.5 / 6.6 kN per hand).
- Sensitivity of the window to 1.3 s: stiffness $\times 0.1$ gives 14 / 18 mm and 30 deg; $\times 10$ gives 0.45 / 0.49 mm and 0.66 deg.
- Left share of the hand force: 45.2% at the peak and 45.8% median (loaded samples) with the bushing, against 47.7% / 48.0% for the minimum-norm weld proxy. The difference is up to 7.7 kN at peak (rms 4.4 kN), dominated by the opposing, squeeze-like force pair.
- The integrator accuracy was checked: 1e-3 and 1e-4 agree to better than 0.1% on a 0.3 s window.

### What Was Tried and Rejected

- Hunt-Crossley sphere-sphere and sphere-cylinder contact: no force produced; open meshes fail. Deferred to phase 2.
- Closing the weld loop with `assemble`: the IK candidate leaves the loop open by 134 mm and 50 deg at address, so assembly produced an 84 kN preload. The right bushing frame is therefore tied to the left hand body, which is where the weld model places it.
- Translational-lever damping for the rotational modes: the pitch mode kept $\zeta \approx 0.01$ and the static hold did not settle. Rotational damping is now set per axis, $c_r = 2\zeta\sqrt{k_r I_\mathrm{ref}}$.

### Limitations

- The default parameters are unfitted engineering defaults; the unmet deflection criterion is a finding about the candidate motion and the defaults, not a tuned result. The candidate has a one-frame hand-velocity spike near 0.95 s (10.9 m/s against 1.3 m/s either side) and hand speeds above 17 m/s at 1.3 s, so the late window is a stress test of the motion, not a calibrated swing.
- The right hand is tied to the left hand body (open IK loop), so the per-hand split is a property of this approximation.
- The window ends at 1.3 s: explicit stiff integration of the full 1.8 s trajectory exceeded the 50 minute time box. This is a follow-up.
- Software correctness only. Scientific qualification stays in the design-manual governance pathway.
- Out of scope for phase 1: the contact model, MuJoCo, Drake and Pinocchio parity, and the `golf_humanoid.osim` builder.
