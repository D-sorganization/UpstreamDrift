# Simscape Matching Reference

Start with the [maintained LaTeX reference](simscape_matching_reference.tex).
It lays out the modeling process, coordinate frames, SI/degree conventions,
calibration, equations, assumptions, numerical methods, fixed acceptance gates,
failed experiments and reproduction. The [refinement record](MATCHING_REFINEMENT.md)
preserves earlier reviews; the [current handoff](../../development/HANDOFF.md)
records outstanding delivery and scientific work.

## Current Review Status

| Deliverable            | Current Evidence                                                                             | Remaining Qualification                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Owner Session Videos   | Six Originals Verified Locally; Visible Marker Placement Reviewed                            | Exact GEARS Trial Pairing, Camera and Anatomical Offsets                         |
| Desktop MP4s           | Eight Refined Human Clips; 104 Comparison Clips Independently Hash/Decode Checked            | IK Only; Dynamics and Contact Unqualified                                        |
| Native Saved States    | 344 Position Poses and 48 Position/Velocity States Freshly Rechecked With Natural Exit Zero  | Whole-Capture Continuous Path and Acceleration                                   |
| C2 Reference Prototype | All Dense Candidates Remain Rejected; Original Fitted Poses Also Require Temporal Regularity | Controlled Data Refit, Full-Path Fit/Regularity, Raw-Clock Coverage and Dynamics |
| Moving Dynamics        | Fixed Supported Holds Have Their Own Recorded Gates                                          | All 35 Input Torques, Contact and Independent Full-Swing Replay                  |
| LaTeX Reference        | Source Updated in the Existing Editor                                                        | PDF Compiler Reports Missing Platform Standard Directories                       |
| Protected Delivery     | Draft PR #11256 With Explicit Scientific Limits                                              | Required CI and Scientific Qualification Remain Open                             |

Use the [aligned comparison checkpoint](aligned_refined_checkpoint_20261003.json) and [tangent/C2 checkpoint](tangent_c2_checkpoint_20261003.json) for the latest scopes and exit dispositions. Dated experiments below retain their original acceptance limits.

## Captures and Driving Modes

Capture A is the tour-average reference. Capture O is the owner's GEARS capture.
They are different data sources; avatar appearance cannot establish playing
ability. Camera view, calibration, topology and driving mode must be identified
before comparing motion. New shareable matches use the Human ellipsoid model
with capture-specific geometry; legacy cylinder outputs remain historical.

The delivered tour/owner clips use inverse-kinematics poses and are labeled
**IK / DYNAMICS UNQUALIFIED**. They are not torque-driven independent replay.
The current torso-refined Human selection is `Simscape_Matches_20261002_Scapula_Refined_1080p`, with eight 1920 by 1080, 30 fps H.264 clips: both captures, both views, clean and marker modes. The earlier 800 by 600 head-tracked package remains a historical comparison. The 104-video collection covers 13 construction/model variants and includes partial historical topologies; its target scores are not all directly comparable. No rejected continuity or dynamic experiment replaces the refined Human files.

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

## Private Pelvis Parity and Concurrent Candidate Review

The private optional pelvis-orientation implementation passed exact configured-Human output parity on three native samples: baseline equals omission equals explicit zero; diagnostic targets at zero weight preserve joint coordinates and marker RMS, and missing pelvis error remains NaN. The native process exited naturally at 23:47:11Z. Positive-weight observations are limited probes; full tour/owner trials are running and no candidate is promoted. Four concurrent scapula/head-weight candidates exceed the existing mean-marker-RMS degradation screen; they do not replace selected Desktop IK videos. A separate four-cell owner seed/offset calibration comparison is queued, preserving geometry/head calibration and freezing address-only offsets before same-source tracking. The existing LaTeX editor source contains these findings; five document/privacy checks pass, while built-in PDF compilation remains unavailable with Unable to find standard directories for platform. Full forward dynamics, all-actuator/prescribed-neck torque recovery and fresh independent replay remain required. Epics #11161 and #11268 and peer draft #11172 remain open dependencies.

The full sampled private pelvis trials completed naturally at 2026-10-03T00:04:02Z: 55 tour samples and two owner trials of 46 samples, all native loop statuses one, with unchanged model/solver hashes and no model save or simulation. Tour marker mean/peak RMS improves to 13.447/38.387 mm; owner pelvis-only gives 16.543/36.396 mm and pelvis plus selective missing lower targets gives 16.544/36.392 mm. Same-target native control pelvis FK establishes post-address means 10.005 to 7.074 degrees (tour) and 12.081 to 9.755 degrees (owner), with 9.743 degrees for the combined owner trial. Tour head peak worsens 18.918 to 28.375 degrees; owner foot peak also worsens slightly. No candidate is promoted. Continuity/ROM/both-view/anatomical and independent dynamics reviews remain required. The four-cell owner seed/offset trial is now running. See pelvis_orientation_full_sampled_checkpoint_20261003.json and the maintained LaTeX reference.

The full private thirteen-trial collection is now located and hash/header-verified, with 361-391 frames per trial, declared 240 Hz/metre units and 35 point labels. The selected capture-O export has 367 frames/38 reference-layout labels; actual source-trial metadata and manifest SHA identify trial 12. Inputs and selected export are unchanged. No populated header event labels or EVENT parameter groups were found. Club type for every trial, event/camera correspondence, paired physical timing and a qualified multi-trial envelope remain open. Reuse shared resolver/event/comparison authorities; do not run the private single-export script indiscriminately because it overwrites the selected export and manifest.

The four-cell owner seed/offset comparison exited naturally at 2026-10-03T00:17:56Z. Original/original reproduces selected control metrics exactly. Marker mean/peak RMS is 16.754/36.987 mm original/original, 39.008/257.266 mild/original, 18.201/35.645 original/recalibrated and 25.640/38.811 mild/recalibrated. Both address-only offset calibrations are frozen before tracking the same 46 samples, and new offsets are seed-specific. Every alternative fails the existing mean-marker degradation screen and retains signed implementation-ROM flags. None is promoted. Offset recalibration alone does not resolve the problem under this tested configuration; no causal or clinical-anatomy claim follows. Private model/solver hashes remain unchanged, no model saved, no simulation. Continue pelvis continuity/ROM review, independent anatomy/observability and full dynamics rather than applying a global sign flip or relaxing thresholds.

Native pelvis continuity/ROM review exited naturally at 00:25:08Z: marker screens pass but tour maximum root step increases to 32.495 mm and head peak worsens; owner pelvis-only root transition is 54.714 mm, while selective-gap-only and combined are 17.350/17.585 mm. Combined owner leg max step 36.417 deg worsens relative to selective-gap-only 34.090 deg. No candidate is promoted. A separate hash-preserving source/export audit verifies 69 filled scalar coordinates in 23 marker frames, with three sampled IK frames affected. Raw finite coordinates remain exactly unchanged; 216 missing scalar coordinates remain unfilled. Residuals and camera flags in ezc3d meta_points do not distinguish filled observations; points fourth row is homogeneous XYZ1, not residual. Current RMS uses exported-data gap flags and must not be claimed as independent raw-measurement-only accuracy until original validity masks are propagated. Both original files and selected Desktop clips remain preserved.

Local acquisition update (2026-10-03 UTC): the complete six-video source export is now also on the operator local machine; archive and all MP4 hashes verified. Indoor surface-marker placements were reviewed, with exact labels, anatomical offsets, trial pairing and physical clock still unverified. Private source-mask TDD passed 13 baseline and nine new tests after RED; actual owner importer default data/callback parity and masked callbacks passed. No production integration, candidate promotion or full forward-dynamics qualification. The existing LaTeX editor remains open; compilation is unverified because the built-in compiler cannot find platform standard directories. See `local_video_marker_placement_checkpoint_20261003.json` and the reference acquisition subsection.

Source observation update (2026-10-03 UTC): optional logical masks now enter the canonical converter before importer callbacks. Native production integration passed 13 existing and nine new contract tests and exact tour/owner default data, callback, joint-centre and head-target parity. Full owner refit on the same 627 source-available targets changed mean RMS 16.677 to 16.658 mm and peak 36.987 to 37.050 mm; largest root increment increased 53.926 to 55.295 mm and 13 signed implementation-ROM rows remain flagged. No candidate/video promotion or dynamics qualification. See `source_mask_full_owner_checkpoint_20261003.json` and the maintained LaTeX reference. Existing unmasked caches/media cannot be relabelled as source-mask-qualified.

Native neck recovery checkpoint (2026-10-03 UTC): the owner one-second supported hold recovered Rx/Ry computed primitive actuation torques through two verified N*m PS–Simulink logging converters. Sample RMS/peak values are 5.950843/9.530841 and 3.574840/5.157831 N*m (1,483 finite samples per primitive). All five original hold gates still pass, with the same 2.321210 mm / 0.699395 degree pelvis motion as the preserved control. Sensing-only logs lacked these channels; connected outputs establish recovery. Neck motion remains prescribed and upper/balance feedback remains active. Four temporary blocks were discarded without saving the model. This is not full-swing torque recovery or independent replay. See `owner_neck_supported_hold_checkpoint_20261003.json` and the maintained LaTeX reference.

Tour neck recovery repetition (2026-10-03 UTC): natural R2025b completion at 02:04:26Z recovered Rx/Ry computed actuation torques (sample RMS/peak 3.827226/7.366211 and 2.061589/3.959804 N\*m; 1,587 finite samples per primitive). All five unchanged supported-hold gates pass; full-precision pelvis displacement/rotation match the preserved control exactly (3.175803 mm / 0.836223 degrees). Both captures now verify the connected prescribed-neck logging route. Neck motion and upper/balance feedback remain present; full-swing recovery and independent replay remain unqualified. See `tour_neck_supported_hold_checkpoint_20261003.json` and the maintained LaTeX reference.

Human moving-reference binding update (2026-10-03 UTC): the upper-body reference helper now accepts an explicit native joint-variable table and resolves block-path/primitive keys through the shared key authority. Human never reuses Fit numbered IDs; legacy explicit Fit retains its unbound interface. Seven contract tests pass. Production native integration completed naturally at 02:21:45Z for all 55 tour and 46 owner selected samples: 21 upper actuator-coordinate references at 30 Hz match the preserved numerical filter/rate/start calculations exactly after verified native block/ID/unit binding. Model hash unchanged, no simulation or save. Moving-start/contact reconciliation and full-swing torque recovery/replay remain open. See `upper_reference_binding_checkpoint_20261003.json` and the maintained LaTeX reference.

Closed-chain reference update (2026-10-03 UTC): independent upper-angle filtering returns native status -1 for all 55 tour and 46 owner requested poses (model constraints satisfied, some targets missed). Native-adjusted poses pass a complete target recheck. The explicit `filter_reference=false` conversion preserves checked sample geometry; eight contract tests and production native roundtrips pass for all selected and adjusted A/O poses. Dependent-arm trials and their adverse changes remain unpromoted. Tangent velocities, between-sample interpolation, moving contact initialization and complete torque coverage still need verification; saved holds have neck/leg datasets but no upper torque buses. See `reference_filter_closure_checkpoint_20261003.json` and the maintained LaTeX reference. No dynamics or new best-video qualification is claimed.

Native moving-reference audit (2026-10-03 UTC): all 101 original sampled poses accept zero native velocity. All 101 differentiated moving requests and all 99 complete-coordinate chart midpoint requests return status -1, missing some complete native targets. A separate nonzero root-rotation/frame-output control verifies follower-resolved spherical velocity. The native results do not qualify an accepted continuous curve, unchanged returned moving poses, measured player velocities or full dynamics. Preserve these adverse outcomes and require constraint-aware trajectory construction, derivative consistency and native initialization before complete torque recovery/replay. See `reference_velocity_midpoint_checkpoint_20261003.json` and the maintained LaTeX calculations. No model save, simulation or new best-video promotion occurred.

The maintained reference now records the PR11351 refined torso preview tradeoffs, local six-video acquisition, source observation qualification, and the independent saved tangent-state recheck. See [native tangent-state recheck](native_tangent_state_recheck_20261003.json). These sampled native states and IK media do not qualify a continuous trajectory or full forward dynamics.

The exporter accepts a logical observation mask with a caller-bound source contract. Native RED/GREEN, actual owner two-frame hash/forwarding/cache checks and 39 production contracts are recorded in [observation export contract checkpoint](observation_export_contract_checkpoint_20261003.json). The maintained LaTeX documents hash, label, clock and cache semantics. Physical measurement accuracy and full-swing dynamics remain unverified; existing Desktop previews retain their original qualification.

## Full Native Pose Continuity and Local Acquisition Review (2026-10-03)

`continuity_experiment_checkpoint_20261003.json` records private full-pose rotational continuity experiments and their natural-exit receipts. All 16 pure helper tests pass after genuine RED isolation. Zero-weight whole-output parity passes on eight owner frames and 53 tour frames. All 305 poses across the sparse owner and dense owner/tour trials pass fresh native position rechecking. This is sampled closure evidence, not continuous velocity, contact or forward-dynamics acceptance.

The owner dense window improves mean target RMS 28.739→28.131 mm and maximum interior wrist increment 24.327→20.356 degrees, while retaining an adverse approximately 40 mm root increment. The tour window improves RMS 26.396→26.154 mm but worsens interior wrist/root increments (7.850→10.231 degrees; 2.180→2.757 mm). Seed boundaries are reported separately. The prototype is private and unpromoted; production solver code, model bytes and existing Desktop MP4s are unchanged by these experiments.

The local download and marker-placement review requested by the owner are complete: six originals, 88,738,421 bytes, verified archive and additional original-name copies. The supplemental private acquisition-review receipt is deterministic, SHA-256 `691b97b68ce58412e400fd552d849f46a5bd2c14f4f442afa72ee87bd5fb345d`. COV-1/#11269 remains open for additional catalog/software acceptance, ffprobe JSON, camera-original provenance and owner recollection. Exact video-trial pairing, camera calibration and anatomical marker offsets remain unqualified. Source media and filenames stay private.

The standalone LaTeX reference now distinguishes historical and newer torso-refined Desktop candidates, corrects mask tensor indexing and defines source-verified balance command units and net torque subtraction. The compiler still reports `Unable to find standard directories for platform`; the latest PDF is unverified. Full moving references, all 35 actuators including the neck, contact acceptance and fresh independent A/O replay remain open. PR #11256 stays draft; no protected merge or full-goal completion is claimed.

Root-availability review (2026-10-03 UTC): saved-array analysis reproduces the legacy owner-window root-step maxima and associates the largest approximately 40 mm increment with restoration of seven derived lower-body targets. Waist-marker gaps can propagate through the pelvis-axis joint-centre estimator despite visible raw leg markers. This is association, not causation. The current refined preview uses a different head-axis, back-marker, spine/scapula and gap-weight objective; the legacy-window maximum does not establish its root error. The private translation-option TDD completed naturally with eight RED failures and eight GREEN passes, no incomplete tests and no model/simulation. The aligned refined port is now natively evaluated but remains private and unpromoted. See root_availability_checkpoint_20261003.json and the maintained LaTeX calculations. Full moving references, contact, all 35 actuators, independent replay and PDF qualification remain open.

Original-fit regularity review (2026-10-03 UTC): prescribing right-wrist Ry and freeing right-forearm Rz is an IK chart experiment, not a physical-model or actuator change. The projected-support curve accepts A 1,278/1,297 and O 702/721; replacing only that wrist curve with the original-anchor-only curve accepts A 1,284/1,297 and O 698/721. Both natural-zero audits reject full-path promotion. Rates on accepted subsets exclude failed samples and adjacent pairs and cannot establish global improvement. The projected-support wrist reference changes up to 13.43/41.52 degrees from its original-anchor curve, despite small independent-coordinate deformation. Original 30 Hz fitted anchor poses already have closed-arm scalar steps of 44.26/117.16 degrees and shoulder SO(3) steps of 38.14/50.16 degrees. Freezing anchors was a controlled experiment, not a future-fit constraint. A private refit window audit is running with baseline/whole-output zero parity, full-pose regularity weights, original address-prefix priors, separate boundary/interior metrics and fresh native pose checks. No refit results or new candidate promotion are claimed. The six Google Photos source videos remain locally downloaded, hash/decode verified and reviewed for visible marker placement; exact GEARS-trial pairing and anatomical offsets remain unresolved. See the maintained LaTeX and aggregate tangent_c2_checkpoint_20261003.json; PDF compilation still fails with the platform standard-directories error.

Closed-arm regularity and foot review (2026-10-03 UTC): six additional assembled support poses retain all 101 original anchors and 11 prior projections. Native dense acceptance improves to A 1,295/1,297 and O 721/721; remaining two tour failures are midpoints. The returned velocities nevertheless reach right-wrist peaks of 113,025 and 169,742 degrees/s, and the owner has a 99.72-degree wrapped scalar step and 115.22-degree shoulder SO(3) step across consecutive accepted samples. Both curves remain rejected for moving-reference/dynamics promotion; pointwise compatibility is insufficient. A fresh native styled-foot audit shows forefoot-minus-rearfoot direction aligned with sole forward axis (dot above 0.9937), with small address yaw errors. Later right-foot horizontal reversals occur when projection lengths are small; their 3D marker-to-visual angles remain about 24 degrees for tour and 11 degrees for owner despite near-180-degree projected yaw. Preserve the documented ankle-marker-height/flat-sole calibration assumptions; no blind shoe flip is justified. Initial World-Frame uniqueness probe exit-one is retained alongside corrected natural-zero audits. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. Alternate coordinate-target allocation is a private unqualified hypothesis; no physical model/actuator authority change is claimed.

Denser C2/fit review (2026-10-03 UTC): evaluating the SAME candidate curves at 2,018 source and midpoint samples reproduces all 1,010 source samples exactly, but native acceptance is A 1,294/1,297 and O 718/721: three new midpoint assembly failures per capture. Original source samples and all 101 anchors still pass; full-path acceptance is rejected. The new failure diagnostic finds six nearby assembled states with unchanged root translation and all 11 controls passing. A separate native dense fit audit exits naturally zero with original anchor point/RMS parity: mean/peak derived target RMS is 16.98/38.02 mm for A and 21.06/59.60 mm for O; mean/peak directly measured three-back-marker RMS is 26.77/54.66 mm and 38.41/88.12 mm respectively. Valid derived target coverage is 7..14 per frame, with gaps excluded. These mean per-frame RMS scores use different sampling from sparse previews and are not direct model/golfer rankings. No curve or video is promoted to full-path or forward-dynamics acceptance. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX.

Tangent/C2 review (2026-10-03 UTC): fresh native checking passes all 48 saved q/v states; original timeout 124 and tour shutdown 125 remain recorded. Private C2 TDD passes 19 tests. Closed-arm guess correction eliminates all 11 owner anchor mismatches while preserving independent targets bitwise; 6 tour and 5 owner assembly failures remain. Nearby assembled states pass fresh position gates, but a same-target closest-seed retry recovers none of the 11 failures (six controls pass). A controlled projected C2 curve adds eleven upper-limb support poses, preserves original 55/46 fitted anchors, root motion, lower body and unaffected curves, and naturally exits zero with ALL 649 tour and 361 owner source-rate q/v/fresh-state checks accepted. Maximum scalar/spherical target deformation is 0.112/0.173 degrees for A and 0.338/0.343 degrees for O; dense raw-marker RMS is not evaluated. Three axis-contract tests pass after real RED/GREEN correction. The candidate remains unpromoted pending between-state closure, acceleration, full raw-clock coverage, all 35 input torques including neck, contact and independent full-swing forward replay. Parent verifies 104 Desktop video hashes and 5,252 decoded 1080p frames; comparison topologies differ and Human ellipsoids remain the new matching standard. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. PDF compilation remains unavailable due to missing platform standard directories.

Aligned refined-objective review (2026-10-03 UTC): actual sparse refined previews have zero consecutive source-frame transitions; largest root increments are 17.827 mm (A) and 30.166 mm (O) over 1/30 s. Address-prefix controlled eight-pose windows preserve the spine prior. Private parameter TDD has eight RED failures/eight GREEN passes. The original 344-pose fit wrapper returned 125 after completion; a fresh independent saved-pose recheck exited naturally with zero and verified all 344 poses plus exact whole-output zero parity. Small root improvements accompany worse wrist increments; no candidate is promoted. Native counts distinguish 48 position variables, 43 velocity variables, 37 floating IK parameters and 35 requested control axes. See aligned_refined_checkpoint_20261003.json and the maintained LaTeX. Full continuous references, contact, all 35 input torques including the neck, independent replay and PDF qualification remain open.
