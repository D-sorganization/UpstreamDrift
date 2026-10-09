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

| Capture     | Lead (left) | Trail (right) |
| ----------- | ----------- | ------------- |
| Tour driver | 16.4        | 4.0           |
| Tour 7-iron | 15.0        | -0.4          |

The tour players are not at 20 degrees per foot (the trail foot is close to square), so 20 degrees is only the fallback for unreliable markers. The owner's capture O value is not measured here: its C3D is private and was not present on the implementing host; run `python3 -m scripts.foot_progression_report` with `CAPTURE_DATA_DIR` set. Until then it takes the flagged 20 degree default.

### What Was Tried and Rejected

**Measured outcome on the tour driver capture (MuJoCo, full pipeline address stage).** Legacy seed: lead 41.7, trail -2.8 degrees (capture 16.4 and 4.0). With `--foot-progression capture`: lead 37.0, trail 4.05 degrees. The trail foot reaches the capture; the lead foot does not, because `hip_rotation_l` reaches its -40 degree limit while the marker fit's leg posture (hip adduction -14.6 degrees) contributes the rest of the yaw (sensitivity about 0.85 degrees of foot yaw per degree of hip rotation, so about -64 degrees would be needed). The lead-foot target is therefore NOT met in the full fit; the shared FK seed alone reaches both feet within 2 degrees (unit-tested). An address solve that also trades hip adduction against stance-width markers is the open follow-up.

Foot yaw through the swing is not constrained; it follows the matched solution and contact (the balance work in #11667 handles slip). The time series and the finish value are reported, not controlled.

### Evidence Receipt

- [`evidence/foot_progression/capture_report.json`](evidence/foot_progression/capture_report.json)
- Tests: `tests/unit/motion_matching/test_foot_progression.py`, `tests/unit/motion_matching/pipeline/test_address_feet.py`, `tests/opensim/test_golf_address.py`.

---

## 16. MyoSuite Neck Retarget: NeckInputY to `neck_flexion` (OSV-3, #11729)

### What

The anthro document neck joint is `Rx(NeckInputX) Ry(NeckInputY) Rz(NeckInputZ)` in the COMRod frame (`full_body_spec_anthro_driver.json`, joint `GolfSwing3D_Kinetic/Neck Joint`), and the head forward axis (towards `HeadFront`, offset `[0.1, 0, 0.16]` m on `Head`) is +x. So `NeckInputX` is lateral bending, `NeckInputY` is flexion/extension (head pitch) and `NeckInputZ` is axial rotation. The pinned myo_sim head chain (`head/assets/myohead_simple_chain.xml` at `33f3ded9`) has only `neck_rotation` (hinge about `[0.2, 1, 0]`) and `neck_flexion` (hinge about head z), and no lateral bending.

`src/engines/physics_engines/myosuite/python/coordinate_map_anthro.json` now maps:

| Source       | Target                                  | Sign |
| ------------ | --------------------------------------- | ---- |
| `NeckInputY` | `neck_flexion`                          | -1   |
| `NeckInputZ` | `neck_rotation`                         | +1   |
| `NeckInputX` | none (in `omitted_source`, with a note) | n/a  |

**Sign derivation (forward kinematics, MuJoCo).** Head forward-axis pitch `p = asin(f . up)` (positive = above the horizontal), central difference at `+/-0.2` rad from the neutral pose, all other coordinates zero:

- Native plant (`get_plant("mujoco", spec)`, `f` = `Head[0.1, 0, 0.16]` minus `Head[0, 0, 0.16]`, up = +z): `dp/dNeckInputY = -11.459 deg` per 0.2 rad (head pitches down). `NeckInputX` and `NeckInputZ` give 0. Analytically `Ry(b) x = (cos b, 0, -sin b)`, so the pitch change is `-b`.
- MyoSuite (`myobody_simpleupper.xml` at the pin, `f` = head body +x, up = -gravity): `dp/dneck_flexion = +11.459 deg` per 0.2 rad (head pitches up); `neck_rotation` gives 0.002 deg. The head +x axis is anterior: its cosine with the right foot calcn-to-toes direction is 0.99992.

Opposite pitch sensitivities of equal magnitude give `neck_flexion = -NeckInputY`. The magnitudes agree to 2e-7 deg, so the mapping is one-to-one on pitch at the neutral pose. Away from neutral the two chains differ: the hinge axes and joint centres are not co-located, and MyoSuite applies rotation before flexion.

### Why

The previous map sent `NeckInputX` (lateral bending) to `neck_flexion`, so a sideways head tilt was replayed as a nod. It also carried a secondary `NeckInputY -> neck_rotation` entry with weight 0.5. `retarget_frame` assigns rather than accumulates, so that entry overwrote the `NeckInputZ` contribution, and `neck_rotation` was effectively `0.5 * NeckInputY` (unit test `test_neck_input_z_alone_drives_neck_rotation` was red on the old map). Removing it makes `NeckInputZ` the only `neck_rotation` driver.

### What Was Tried and Rejected

- **Keeping `NeckInputX -> neck_flexion`** (the previous map). `NeckInputX` is lateral bending, so this replayed a sideways head tilt as a nod.
- **Keeping the 0.5-weighted `NeckInputY -> neck_rotation` entry.** It overwrote `NeckInputZ`, the only axial-rotation input.

### Limitations

- MyoSuite cannot represent neck lateral bending. `NeckInputX` is dropped, not absorbed into another coordinate.
- Other secondary (weighted) entries in the map have the same overwrite problem (for example `RScapInputY` 0.5 over `RSInputY` on `arm_flex_r`). That is outside this decision and tracked as a follow-up.
- Earlier MyoSuite replay receipts (`evidence/matched/driver_g1_myosuite/receipt.json`, 11 mapped coordinates, historical map) are historical and were not regenerated. They neither record nor depend on a hash of the map file.
- The map remains diagnostic (`qualification.diagnostic_only`). This is a kinematic sign/axis correction, not dynamics or gaze qualification for #11729.

### Reproduction

```bash
git clone https://github.com/MyoHub/myo_sim.git /path/to/myo_sim
git -C /path/to/myo_sim checkout 33f3ded946f55adbdcf963c99999587aadaf975f
MUJOCO_GL=egl python3 -m scripts.myosuite_neck_flexion_sign --myo-sim /path/to/myo_sim
python3 -m pytest tests/unit/engines/myosuite/test_retarget.py -q
```

### Evidence Receipt

- [`evidence/myosuite_neck/neck_flexion_sign.json`](evidence/myosuite_neck/neck_flexion_sign.json)
- Tests: `tests/unit/engines/myosuite/test_retarget.py` (`test_neck_input_y_drives_neck_flexion_with_fk_sign`, `test_neck_input_x_is_unmapped_and_documented`, `test_neck_input_z_alone_drives_neck_rotation`, `test_fixture_map_partitions_source_coordinates`).

---

## 17. MyoSuite Retarget Map Is One-to-One: Weighted Secondary Entries Dropped (#11729)

### What

`load_retarget_map` (`src/engines/physics_engines/myosuite/python/retarget.py`) now rejects any map in which two sources share a target, a source appears twice, or a sign is zero, and raises `ValueError`. The six weighted secondary entries in `coordinate_map_anthro.json` were removed, and their sources were added to `omitted_source` with notes:

| Source                                | Former entry       | Why it has no MyoSuite target                                                                                                                                                                            |
| ------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HipInputX`, `HipInputY`, `HipInputZ` | 0.25 x `lumbar_*`  | They give the `LowerTorso` orientation relative to `world` (the pelvis root), not a lumbar angle. Adding them to the lumbar joints moves the trunk relative to the pelvis that the capture did not move. |
| `RScapInputY`, `LScapInputY`          | 0.5 x `arm_flex_*` | The scapula joint is `Rx(ScapInputX) Rz(ScapInputY)`, so `ScapInputY` is a rotation about z (protraction/retraction). `arm_flex` follows `SInputY`, a y-axis rotation, so the axes do not match.         |
| `RScapInputX`                         | 0.5 x `arm_add_r`  | `myobody_simpleupper` has no scapular DOF. The left side never had this entry, so the absorption was asymmetric.                                                                                         |

`LScapInputX` was already omitted. Its note used to say it was "absorbed via `arm_add_l`", but no such entry existed, so the note was corrected.

`retarget_frame`, `project_to_source` and `interpolate_unmapped` keep their one-to-one assignment semantics. `project_to_source` is now an exact inverse on mapped sources, because the loader guarantees a one-to-one map with non-zero signs.

### Why

`retarget_frame` assigns `out[target] = sign * q[source]` for each entry in order. Every secondary entry came after its primary, so on the shipped map it overwrote the primary. `SpineInputX`, `SpineInputY`, `TorsoInput`, `RSInputX`, `RSInputY` and `LSInputY` had no effect, and their targets were replayed as 0.25 x pelvis angle or 0.5 x scapula angle. Section 16 removed the same defect for `NeckInputY -> neck_rotation`.

Each mapped target is now driven only by its axis-matched primary. Inputs that MyoSuite cannot represent are listed as omitted rather than silently blended in, and the loader check stops the same overwrite from coming back.

### What Was Tried and Rejected

- **Accumulating weighted contributions** (`out[target] += weight * q[source]`). This would keep the intent of the old entries. But none of the weights (0.25, 0.5) was derived from kinematics, and the Hip and Scap axes do not correspond to the targets they were added to (see the table). Accumulation would also make `project_to_source` under-determined.
- **Reordering the entries so the primary is written last.** This hides the overwrite without removing it, and it silently discards the secondary input.

### Limitations

- MyoSuite replay still has no shoulder-girdle motion, and the pelvis root orientation is not driven. `_qpos_from_retarget` reads `pelvis_r*`, which is not in the target list, so the free-joint quaternion stays at identity. Absorbing scapula motion into the shoulder needs a forward-kinematics fit of the scapula and shoulder chain against the glenohumeral joint. It should not be added back as a fixed weight.
- Earlier MyoSuite replay receipts were produced with the overwriting map and are historical. They were not regenerated.
- The map remains diagnostic (`qualification.diagnostic_only`).

### Neck Torque Capacities

`src/shared/python/myofullbody/neck.py` `CAPACITY_NM` labelled `NeckInputX` as flexion/extension (30 N m) and `NeckInputY` as lateral bending (36 N m). Section 16 established that the anthro neck joint is `Rx(X) Ry(Y) Rz(Z)` with the head forward axis on +x, so X is lateral bending and Y is flexion. The capacities were swapped to `NeckInputX` 36 N m (lateral bending) and `NeckInputY` 30 N m (flexion, the smaller of flexion and extension). `NeckInputZ` stays at 15 N m. The MyoFullBody receipts under `evidence/myofullbody/` record the old `capacity_nm` and are historical until they are regenerated.

### Reproduction

```bash
python3 -m pytest tests/unit/engines/myosuite/test_retarget.py tests/unit/engines/myofullbody/test_myofullbody_neck.py -q
```

### Evidence Receipt

- LaTeX reference: [`myosuite_retarget_map.tex`](../../research/myosuite_retarget_map/myosuite_retarget_map.tex). The MyoFullBody neck capacities are also in [`myofullbody_swing.tex`](../../research/myofullbody_swing/myofullbody_swing.tex).
- Tests: `tests/unit/engines/myosuite/test_retarget.py` (`test_primary_source_alone_drives_its_target`, `test_fixture_map_targets_are_one_to_one`, `test_secondary_sources_are_omitted_and_documented`, `test_fixture_map_round_trips_mapped_sources`, `test_loader_rejects_two_sources_on_one_target`, `test_loader_rejects_duplicate_source`, `test_loader_rejects_zero_sign`); `tests/unit/engines/myofullbody/test_myofullbody_neck.py` (`test_capacities_follow_anthro_neck_axes`). All of them failed before the fix.
