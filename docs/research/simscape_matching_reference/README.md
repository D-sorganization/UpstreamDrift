# Simscape Matching Reference

Start with the [maintained LaTeX reference](simscape_matching_reference.tex).
It lays out the modeling process, coordinate frames, SI/degree conventions,
calibration, equations, assumptions, numerical methods, fixed acceptance gates,
failed experiments and reproduction. The [refinement record](MATCHING_REFINEMENT.md)
preserves earlier reviews; the [current handoff](../../development/HANDOFF.md)
records outstanding delivery and scientific work.

## Captures and Driving Modes

Capture A is the tour-average reference. Capture O is the owner's GEARS capture.
They are different data sources; avatar appearance cannot establish playing
ability. Camera view, calibration, topology and driving mode must be identified
before comparing motion. New shareable matches use the Human ellipsoid model
with capture-specific geometry; legacy cylinder outputs remain historical.

The delivered tour/owner clips use inverse-kinematics poses and are labeled
**IK / DYNAMICS UNQUALIFIED**. They are not torque-driven independent replay.
The Desktop selection is `Best_Human_Matches_20261002_HeadTracked`, with two
camera views per capture, fully decoded 800 by 600, 30 fps H.264 clips and a share ZIP.
No rejected dynamic experiment replaces that selection.

## Modeling Process

1. Validate capture identity, units, marker/sensor frames, handedness and timing.
2. Fit longitudinal Human geometry and apply recorded subject mass/inertia
   assumptions. Artistic transverse proportions are not anatomical validation.
3. Match body positions, calibrated foot orientations and measured head
   orientation with explicit keyed initial poses and shared SO(3) residuals.
4. Check native FK, joint mapping, pose validity, residuals and observability.
5. Configure native starts, rates, gravity, ground/contact geometry and reference
   tables. Verify the actual assembled initial state and physical interfaces.
6. Qualify torque/feedback experiments against unchanged force and pose gates;
   retain failures. Remove prescribed drives and prove independent replay before
   labeling a full swing as forward-dynamics qualified.
7. Render and fully decode both views, keeping model/capture hashes, geometry,
   sampling, numerical metrics and driving mode with each shareable output.

The engine's [reference pointer](../../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/REFERENCE.md)
links to this single calculation-level source. The current model's two-axis
neck, incomplete observability and subject inertia assumptions remain limitations.
Flat-sole calibration is not a measured contact-force record.

## Current Qualification

The configured Human upper-body learner preserves caller-owned workspace,
initial-state configuration and model lifecycle, and returns the best torque
profile actually simulated. Its 19 native contract tests and two short tour/owner
probes passed; the earlier combined 311 native/legacy regression also passed.
Those short feedback-assisted results do not qualify the full Human swing.

The leg command now supports a constant 12-vector or a sampled 12-by-N torque
table on the leg-reference grid. Type, finite-value and shape contracts reject
invalid input. Shared bracketing supplies interpolation and endpoint clamping.
The nine new tests plus existing balance regression passed 16 tests with zero failures/incomplete in R2025b.
The neck radian table uses the same `LegReferenceTime` and must have N columns.

| Human Experiment                   | Maximum Pelvis Motion | Maximum Rotation | Status                              |
| ---------------------------------- | --------------------: | ---------------: | ----------------------------------- |
| Tour Original Balance-On Hold      |             20.041 mm |        5.582 deg | Rejects Both Pose Gates             |
| Owner Original Balance-On Hold     |             14.818 mm |        4.408 deg | Rejects Both Pose Gates             |
| Tour Stiffness ×4 / Damping ×2     |              6.128 mm |        1.751 deg | Also Rejects Peak Force 2.263 BW    |
| Tour Ramped Empirical Feedforward  |             11.012 mm |        2.773 deg | Force Gates Pass; Pose Gates Reject |
| Owner Ramped Empirical Feedforward |              7.256 mm |        2.171 deg | Force Gates Pass; Pose Gates Reject |

Pose limits remain 5 mm and 1 degree, with separately registered support and force
limits. Saved baseline net effort is 45.954/56.019 N m RMS, substantially below
servo-only 226.583/236.384 N m because balance offsets the servo target. Hip logs
are generalized XYZ command taps; knee/ankle logs are native actuator sensors.
The late windows still move, so their means are not established static gravity
torques. The tour ramp retains the original 1.981 BW peak. Owner profile results
completed separately at 19:16:13Z: 7.256 mm / 2.171 degrees. All force gates pass; both fixed pose gates reject.

**Full Human forward dynamics remains unqualified.** Feedback and a prescribed
neck remain in these experiments. Scientific gates #11156, #11160 and #11173
stay open. Process exit0, green tests, optimizer success and animation quality
cannot substitute for physical acceptance.

## Evidence and Reproduction

Each anonymous JSON record below contains its scope, actual metrics, hashes,
receipts and limits. Raw private captures, pose/torque traces, absolute private
paths and execution caches remain outside Git. Exact probe names, inputs,
runner parameters and mathematical details are in the LaTeX reference.

Use MATLAB R2025b explicitly and the serialized `tools/run_matlab_locked.ps1`
runner. Its `-Script` is an actual saved script path, with a unique private
`-Log`; natural-exit receipts distinguish success from watchdog termination.
Run `test_gs3dx_leg_feedforward_profile` and `test_gs3dx_fit_balance` for the
sampled-torque seam. Test receipts describe their actual native model and scope.

- [Balance Enabled Hold Comparison Review 20261002](balance_enabled_hold_comparison_review_20261002.json)
- [Bilateral Assembled Contact Review 20261002](bilateral_assembled_contact_review_20261002.json)
- [Bilateral Stance Geometry Review 20261002](bilateral_stance_geometry_review_20261002.json)
- [Capture Runtime Comparison 20261002](capture_runtime_comparison_20261002.json)
- [Configured Human Learning Review 20261002](configured_human_learning_review_20261002.json)
- [Contact Capture Audit 20261001](contact_capture_audit_20261001.json)
- [Desktop Head Tracking Delivery 20261002](desktop_head_tracking_delivery_20261002.json)
- [Fresh Video Verification 20261002](fresh_video_verification_20261002.json)
- [Grip Candidate Comparison 20261001](grip_candidate_comparison_20261001.json)
- [Head And Quiet Review 20261002](head_and_quiet_review_20261002.json)
- [Head Track Audit 20261002](head_track_audit_20261002.json)
- [Head Weight Followup Review 20261002](head_weight_followup_review_20261002.json)
- [Human Ankle Gain Interface Review 20261002](human_ankle_gain_interface_review_20261002.json)
- [Human Default Policy Review 20261002](human_default_policy_review_20261002.json)
- [Leg Feedforward Profile Review 20261002](leg_feedforward_profile_review_20261002.json)
- [Leg Orientation Contract Review 20261002](leg_orientation_contract_review_20261002.json)
- [Main Integration Tests 20261002](main_integration_tests_20261002.json)
- [Main Video Verification 20261002](main_video_verification_20261002.json)
- [Matching Comparison 20261001](matching_comparison_20261001.json)
- [Native Configured Learning Integration Tests 20261002](native_configured_learning_integration_tests_20261002.json)
- [Native Head And Quiet Integration Tests 20261002](native_head_and_quiet_integration_tests_20261002.json)
- [Native Helper Integration Tests 20261002](native_helper_integration_tests_20261002.json)
- [Native Helper Review 20261002](native_helper_review_20261002.json)
- [Native Human Policy Integration Tests 20261002](native_human_policy_integration_tests_20261002.json)
- [Native Leg Contact Integration Tests 20261002](native_leg_contact_integration_tests_20261002.json)
- [Owner Constant Hold Review 20261002](owner_constant_hold_review_20261002.json)
- [Saved Leg Net Actuation Review 20261002](saved_leg_net_actuation_review_20261002.json)
- [Saved Leg Servo Diagnosis Review 20261002](saved_leg_servo_diagnosis_review_20261002.json)
- [Saved Native Hold Diagnosis Review 20261002](saved_native_hold_diagnosis_review_20261002.json)
- [Seeded Head And Contact Review 20261002](seeded_head_and_contact_review_20261002.json)
- [Selected Contact Geometry Review 20261002](selected_contact_geometry_review_20261002.json)
- [Tour Constant Hold Review 20261002](tour_constant_hold_review_20261002.json)
- [Tour Ramped Leg Feedforward Review 20261002](tour_ramped_leg_feedforward_review_20261002.json)
- [Video Verification 20261001](video_verification_20261001.json)

## Reference Governance and Verification

This is a separate research and experiment reference. The sole editable
engineering design-manual source remains [manuals/upstreamdrift](../../../manuals/upstreamdrift)
QMD; generated manual artifacts must not be edited directly. The repository's
[modeling-documentation policy](../../../AGENTS.md) requires calculation-level
references with substantive modeling changes, including failed experiments.

The same standalone `.tex` remains open in Codex's editor. Its current PDF is
unverified because the built-in compiler reports `Unable to find standard
directories for platform`. Do not interpret structural checks as PDF review.

```sh
python3 -m pytest -q docs/research/simscape_matching_reference/test_simscape_matching_reference.py
python3 -m scripts.check_design_manual_governance
```

These checks cover structure/privacy and manual policy respectively. The manual
calculation inventory, rendering review, physical qualification and protected
delivery each need their own evidence. Prior published head 78918cde passed its
reported checks; new source changes require fresh protected checks.

The owner-session video companion is tracked by [epic #11268](https://github.com/D-sorganization/UpstreamDrift/issues/11268). Its original-byte acquisition, grading, frozen camera/pairing protocol, markerless comparison and error budget are incorporated in the maintained LaTeX reference; no video pairing or comparison result is claimed yet.

- [Owner Ramped Leg Feedforward Review](owner_ramped_leg_feedforward_review_20261002.json)

## Startup Transient Audit

A native read-only audit of the saved ramped runs completed at 19:37:01Z. Both
captures first breach the 1-degree gate during the 50–200 ms torque ramp, then
return closer to the original pose by one second. Signed vertical displacement
changes direction; sustained downward sag and a unique anatomical pitch direction
are not established. Whole-run gates remain rejected. The next controlled native
experiment changes only the ramp to 0–50 ms, retaining all force/posture limits.
It is a prospective test, with no outcome claimed here.

- [Ramped Hold Posture Review](ramped_hold_posture_review_20261002.json)

## Supported Posture Checkpoint

The controlled 20–50 ms empirical leg-torque ramp passed all five unchanged one-second hold gates in separate native runs: tour 3.176 mm / 0.836 degrees, peak 1.981 BW (19:54:39Z); owner 2.321 mm / 0.699 degrees, peak 1.730 BW (20:01:28Z). The preceding 0–50 ms tour ramp passed posture but rejected peak 2.000907 BW; rounding cannot change that verdict. Gains, geometry, contacts and model binaries were retained. This is supported posture with feedback and prescribed neck, not identified static gravity or independent full-swing replay.

| Capture | Maximum Pelvis Motion | Rotation Change | Peak Force | Registered One-Second Screen |
| ------- | --------------------- | --------------- | ---------- | ---------------------------- |
| Tour    | 3.176 mm              | 0.836 deg       | 1.981 BW   | All Five Gates Pass          |
| Owner   | 2.321 mm              | 0.699 deg       | 1.730 BW   | All Five Gates Pass          |

- [Controlled Early Ramp Review](tour_early_leg_feedforward_review_20261002.json)
- [Contact-Guarded Hold Review](guarded_leg_feedforward_hold_review_20261002.json)

## Selected Motion and Video Acquisition Checkpoint

The owner-approved public album source recorded by main #11283 is accessible from DeskComputer. Its observed Download all ZIP contains six source-export MP4s that exactly match the previously staged streams by size and SHA-256; raw-camera provenance remains unverified. The pinned OpenPose BODY_25 network loaded and completed synthetic inference at 20:33:36Z; this is runtime evidence, not owner detection accuracy. The selected head-weight-0.1 references have 55 tour / 46 owner samples over 1.8 / 1.5 seconds. All 101 frames passed the existing named native-target and quiet-reference mapper at 20:37:51Z, with no model loaded. A subsequent native audit at 20:42:55Z verified ankle FK against both configured Human models to numerical precision across all frames. All six streams fully decoded with FFmpeg and 9,539 increasing presentation timestamps. A controlled local owner trial confirmed a gap-to-measured transition near the largest root step. Stronger rotational smoothing reduces leg-coordinate jumps but worsens root discontinuity and mean marker RMS, so it is not promoted. Separate gap-weight trials reduce the transition root step to 23.664 / 17.292 mm, with interpolation provenance retained; neither is promoted before full-chain and uncertainty review. No stance correction, velocity reference, full-motion dynamics, video pairing or camera comparison is qualified by those audits.

[Neutral provenance evidence](selected_motion_video_checkpoint_20261002.json) retains native receipts and explicit qualification boundaries.

## Selective Missing-Position Contracts

The full owner trial with global missing-position weight 1 is rejected: measured-marker mean/peak RMS increased from 16.754/36.987 mm to 18.791/49.148 mm despite reducing the transition root step. Optional selective missing-position scales now preserve measured targets, offset masks and head/foot terms; default metadata remains unchanged. Native R2025b RED retained eight passing seed tests and exposed ten new failures; GREEN passed ten new and 35 existing tests. Actual-model comparison confirms exact complete-output parity for omitted/empty overrides against the previous solver at source frames 1, 241 and 249. The full 46-frame selective lower-body trial reduces the transition root step from 53.926 to 17.350 mm and the maximum leg step from 38.380 to 34.090 degrees while retaining measured-marker mean/peak RMS at 16.753/36.975 mm. The body-fit screen passes, but remaining 34-degree leg discontinuity, observability, ROM and both-view visual review preclude promotion. Full forward-dynamics qualification remains open.

## Native Branch and Range Diagnostics

A native no-model audit separately measures true adjacent SO(3) rotations and checks the existing implementation ROM table. The largest tour leg-coordinate step is 22.312 degrees at source frames 517–529, with a 24.581-degree left-hip rotation. The selective owner step is 34.090 degrees at 241–249, with a 26.710-degree left-hip rotation (30.025 degrees for the selected owner output). These changes are not solely coordinate wrapping; their agreement with measured swing motion and observation gaps remains under review. The current table flags 12 tour and 13 owner rows, including both owner elbow signs across all 46 samples. This is an implementation-range diagnostic, not a clinical diagnosis: hips/free pelvis lack absolute ranges and wrists lack anatomical neutrals. Native clean joint-centre geometry and offset calibration must be checked before revising conventions or penalties. No candidate is promoted.

## Native Unoffset Geometry and Marker Observability

Native unoffset geometry confirms owner address elbow angles of 27.80/40.40 degrees with LE negative/RE positive, while tour address has 47.69/39.06 degrees with LE positive/RE negative. Unsigned native flexion agrees with the absolute primitive at all reviewed stages; this does not justify globally reversing the ROM signs. Generic joint-centre proxy offsets remain assumptions rather than subject-specific anatomical calibration. The tour largest-step interval has measured thigh-direction changes of 1.568/3.302 degrees and a 9.022-degree pelvis-cluster rotation; axial thigh twist remains unobserved. The corresponding owner interval has a missing-to-measured transition, so an observed pelvis rotation cannot be asserted there. Address-relative pelvis-cluster alignment averages 9.831 degrees for tour and 11.494 degrees for selected owner (peaks 35.017/38.666 degrees); selective owner gap weights scarcely change this. Next review native closed-loop seed families and an optional measured pelvis-orientation constraint, preserving missing masks and unchanged body/head/foot gates. No new video is promoted.

## Owner Seed Feasibility and Rejected Address Fits

Six native owner seed families close the grip loop while preserving the original root, lower-body and neck values to numerical precision. Four families satisfy both elbow rows; one seed has zero flagged rows in the current implementation table at address. Those are feasibility checks, not match acceptance. Re-fitting address with the original offsets gives marker RMS 30.745/40.683 mm for the two bilateral seeds versus 6.061 mm for the original control. The mild seed retains zero flagged rows but degrades marker and head fit; both alternatives are rejected for promotion. Next investigate branch-consistent offset calibration with a frozen calibration/held-out split, and measured pelvis-orientation constraints. No global sign reversal or threshold relaxation is justified.
