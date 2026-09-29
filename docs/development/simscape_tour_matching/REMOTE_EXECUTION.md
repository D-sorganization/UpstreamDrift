# Remote MATLAB Execution and Task Transfer

## Separate Instrumentation Checkout

All-channel instrumentation was qualified in C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-actuator-audit, isolated from active fitting. The saved revolute reference now emits ActuatorTorque for the five previously unsensed coordinates. Five native tests pass; the refined 0.3 s replay has exactly unchanged time/q/qd/qdd/markers and all 27 channels match their polynomials. All final process receipts are explicit zero. Archive actuator-instrumentation-01-bundle.zip has SHA256 b5ae16b4d4901d4d12765c64196efe38d051a19ca1702838f7c323b37525b025 on both machines, with 36 verified file digests. See the handoff and bundle receipt for before/after model hashes.

To migrate an older isolated reference, load Kinetically_Driven_Revolute_Joint, call enable_revolute_actuator_log(model_name), then save/close that reference. Run qualified_candidate_replay with a fresh directory containing the completed fit report, raw final_native_replay.mat and driver_marker_payload.json. audit_saved_golf_candidate defaults to requiring 27 channels; pass a third argument 22 only for historical raw fits. The archived fresh-sensor-sample demonstrates the 27-channel audit; candidate-replay retains the original 22-channel fit data and the independent instrumented replay separately.

The active 0.4 s fit and queued cold validation still use the old source at 59e944b44. Preserve that checkout until both finish; the isolated instrumentation processes are terminal.

## Active 0.4-Second Quadratic Run

Runtime 59e944b44; run prefix-400ms-quadratic-01 uses the completed refined 0.3 s candidate, 81 controls, --basis quadratic --duration 0.4 --finite-difference-step 0.0001 --max-nfev 8. Fitting SSH 42924 / PowerShell 43780 / Python 42772,36996 / R2025b MATLAB 41308 were confirmed live. Cold replay/audit is already queued in SSH 71640 / PowerShell 39208. Launch scripts and argument JSON are in the run directory. Do not restart or update runtime while either process is active. Native exit receipts now require an actual integer; null and blank values cannot qualify as success. AGENT_HANDOFF.md is the current-state entry point.

## Completed Refinement and Receipt Correction

prefix-300ms-linear-refine-02 and its independent cold replay/audit are terminal. Final RMS 1.695470 mm, exact cold marker parity, 22 sensed channels qualified. Cold exit receipt is zero; fitting exit receipt is empty and must be treated as unknown. The same empty fitting receipt occurs in the earlier 0.1/0.2/0.3 s archives. Corrected metadata preserves all archive hashes and cold evidence. Never coerce an empty string to integer zero. Future launchers use Start-Process -Wait -PassThru and reject null ExitCode; future gates reject empty/noninteger receipts.

Archive on both machines: prefix-300ms-linear-refine-02-bundle.zip, SHA256 ddead3176156d826d16f20c248e0a98208bfb501c6bc909ce83bd9ec5ce30aa7; 128 file digests verified. Runtime remains 31595aba9. Draft PR #9948 is open. Next use the completed report as --transfer-report for fresh 0.4 s quadratic fitting after syncing the published code. AGENT_HANDOFF.md has current details; older live-process sections below are historical.

## Queued Validation for the Active Refinement

The live fitting session is 99257. Dependent cold replay/audit is already queued in session 47374 (remote PowerShell 31188), using prefix-300ms-linear-refine-02/cold-replay-launch.ps1. It checks successful fit completion, completed JSON and exact runtime 31595aba9 before launching explicit R2025b. Do not duplicate the replay or update the runtime. Inspect cold-process-id.txt and cold-exit-code.txt once created. Last observation: 430 fit evaluations, no cold result. AGENT_HANDOFF.md is the current recovery entry point.

## Active 0.3-Second Refinement

DeskComputer runtime is exact published 31595aba9. Active run C:/Users/diete/SimscapeTour9921/prefix-300ms-linear-refine-02 uses --duration 0.3 --basis linear --finite-difference-step 0.0001 --max-nfev 10 and same-horizon --transfer-report from prefix-300ms-linear-01. Python process 32420 was confirmed live; SSH session 99257 retains process completion. See launch-arguments.json and fit.err/log; exit-code.txt is created after completion. Do not restart on an observation timeout or change the runtime during the experiment. Collect the report/checkpoints, then run independent cold replay and frame-aware audit. Current AGENT_HANDOFF.md takes precedence over historical run sections below.

## 0.3-Second Run and Force-Frame Audit

The 0.3 s fit, independent cold replay and 22-channel actuator audit all exited zero. No native process remains live. Fit/cold source is exact d344b726d; runtime is now 78b0feef4 after advancing the completed checkout for the frame-aware audit. Preserve the three untracked subsystem backups and all stashes. Final RMS is 2.056357 mm; cold marker difference is zero. Force audit error is 1.137e-13 N and torque error zero; five coordinates remain unlogged. Optimizer convergence is false.

Recovery archive: prefix-300ms-linear-01-bundle.zip under DeskComputer C:/Users/diete/SimscapeTour9921 and local C:/Users/diete/Repositories/simscape-tour-checkpoints. SHA256: 94c3062b34d4292ea44fcfba78696a606e1e2e315be4079a2b557f50d277cdd3. All 61 file digests and ZIP integrity were verified; see native_evidence/prefix_300ms_linear_bundle_receipt.json. Actual commands were first_prefix_fit.py --repo <runtime> --run-dir <fresh-run> --initial-state <qualified-state.json> --duration 0.3 --basis linear --transfer-report <completed-0.2-report.json> --max-nfev 8, followed in fresh explicit R2025b processes by qualified_candidate_replay(repo,run_dir) and audit_saved_golf_candidate(repo,run_dir). Copy the capture payload into each fresh run directory before launching.

AGENT_HANDOFF.md is the current-state entry point; lower sections retain historical experiments and must not override it. Next expose and test a smaller finite-difference step for a same-horizon candidate refinement. Checkpoint resume validates the exact capture, state, attachments and basis identity; never overwrite earlier run directories. Save every five evaluations as already implemented, retain the SSH process until exit, cold-replay each selected result and archive at each completed milestone.

## Completed Linear-Torque 0.2-Second Run

DeskComputer C:/Users/diete/SimscapeTour9921/prefix-200ms-linear-01 completed with exit zero. Its cold replay and actuator audit also exited zero; no fitting process remains live. Final marker RMS is 0.859647 mm, p95 1.873274 mm, maximum 3.575221 mm. Cold marker difference is exactly zero, and all sixteen logged actuator profiles match their polynomials exactly. Eleven coordinates remain unlogged. The optimizer reached its evaluation budget and is not marked converged.

Runtime a58c866ca still carries the two owned fitting patches from 6875566d5; audit code is f91920f0f in scratch. Preserve the active source patches before checkout. The local and remote prefix-200ms-linear-01-bundle.zip archives both hash to 32a491e2a14631c950ae4633edbad860fe917c90954cfd6839fa1e7f046bdfba; all 104 file hashes were checked. See prefix_200ms_linear_bundle_receipt.json. The next run is a fresh 0.3 s linear prefix using this run's completed report as --transfer-report. The parameter transfer passes the current bounds.

Checkout recovery preserved prior patches in linear-source-precheckout and the named stash recorded in AGENT_HANDOFF.md. A scoped git stash returned nonzero because the shared path is ignored, despite creating a stash and staging files. Inspect status after such a failure; do not assume cleanup completed. Existing subsystem backups and older stashes remain intact.

## Completed Qualified-State 0.1-Second Run

DeskComputer C:/Users/diete/SimscapeTour9921/prefix-100ms-01 completed with Python/MATLAB exit zero. The independent cold replay also exited zero and matches every final marker coordinate exactly. No fit or replay remains live. Runtime is ba18f2e7b plus tour_fit_state.py from 7f4ab9d5f; the exact runner is archived in the run directory. Results are 0.356491 mm RMS, 0.827594 mm p95 and 1.591758 mm maximum across 37 samples and 25 markers. The optimizer hit its evaluation limit and is not marked converged.

The full run archive is C:/Users/diete/SimscapeTour9921/prefix-100ms-01-bundle.zip, copied locally to C:/Users/diete/Repositories/simscape-tour-checkpoints/prefix-100ms-01-bundle.zip. SHA256 e673a2a9db8772c3718ae8cefaa049b224a5830f3599cae43ad7b5a910c1fa1a matches; all 77 file hashes were verified. See prefix_100ms_bundle_receipt.json and CHECKPOINTS.md. Keep the existing remote subsystem backups and stashes. The next experiment must use a fresh directory and explicit torque-candidate transfer to the longer horizon.

## Qualified Initial Velocity and Session Targets

SELF fixes the nonzero initial-velocity regression with configure_capture_velocity_targets.m. Load the model, keep its returned onCleanup guard alive for the entire fit, and clear it when the fit ends. It enables the schema's independent velocity targets, leaves dependent rates to the native constraints, releases Fast Restart at session boundaries, and restores original mask selectors and primitive flags on normal completion or error. It does not save the source model. The sole forward wrapper and default simulation options remain unchanged.

Two R2025b TestCase methods pass, covering cold/warm/warm/cold replays and exception cleanup. The saved 50 ms marker replay now has initial_state_verified=true and status=initial-state-qualified: q error 7.247e-13, qd error 3.986e-14 and initial marker error 6.686e-13 m. Geometry remains (14.5,12) inches, with single-initial-frame offsets; the seven-frame velocity residual is 4.225 mm/s. This qualifies initialization, not anatomical geometry or a complete swing.

The rejected per-call SimulationInput mask-override prototype achieved the requested rates but failed restoration and emitted non-tunable Fast Restart warnings. Its archive is C:/Users/diete/Repositories/simscape-tour-checkpoints/transient-velocity-policy.zip, copied to C:/Users/diete/SimscapeTour9921/transient-velocity-policy.zip. The prototype helper was removed from both current source trees. Do not recreate a per-call independent_velocity_targets option; use the session guard. The first_prefix_fit.py reproduction runner now creates the guard and clears it in finally, then closes its private model without saving.

Qualified evidence is native_evidence/initial_velocity_seed_qualified_r2025b.json and capture_initial_velocity_session_green_r2025b.json. Earlier unqualified reports remain failure evidence. Completed run: C:/Users/diete/SimscapeTour9921/velocity-seed-02, with raw MAT, capture payload, helper, probe, tests and log in velocity-seed-02-bundle.zip. Both local and remote archive hashes match 798e83a1f5916e9004d29ec9d0fef9e08c9266f2c8ccd7a5d0aa41f96af081ed; qualified_velocity_bundle_receipt.json records paths and per-file hashes. MATLAB exited zero; no native process remains live. Runtime is 81a8086d9 with the new helper copied into shared (ignored until tracked by this implementation); the default/forward/capture-option files were restored to published content. Preserve existing subsystem backups and stashes.

Next extend the fitting runner to consume the qualified q/qd, geometry and fixed offsets and to accept a longer horizon (start with 0–0.1 s). The existing runner still constructs default geometry and zero initial rates; --checkpoint alone only supplies torque parameters. Record the initial-state/attachment identity and joint order in resume checks before allowing changed-state runs. Keep the earlier verified torque checkpoint intact, fit each prefix from the same t0 state, and progressively allow continuous nonconstant polynomial efforts. Full-swing qualification remains open. R2025b is the only required release.

## Initial-Velocity Mapping and Failing Regression

SELF adds golf_marker_velocity_map.m. It resolves native velocity identities by block path/primitive, uses explicit m/s and rad/s units, and maps 21 independent rates into all 27 joint rates plus fixed-marker world velocities. Frame angular velocities use follower axes and frame linear velocities use world axes. The native TestCase first failed on the missing function, then passed a hip-angular column comparison against centered native pose differencing and exact independent-rate identity. The dependent-coordinate list is now shared in golf_kinematic_schema.json; fit_golf_pose_seed uses it and its R2025b regression also passes.

The best geometry seed's first seven capture frames (0–1/60 s) give target marker-velocity RMS 0.0206622 m/s. A damped least-squares tangent fit (damping 0.001; rank 21) reduces residual RMS to 0.00422497 m/s. Offsets remain calibrated only at t0 and geometry remains provisional. IMPORTANT: its forward initialization FAILED acceptance. q and initial markers match within about 7e-13, but maximum qd mismatch is 0.0819077 rad/s. TorsoInput requests 0.002033743 rad/s but starts at zero; LWInputY requests -0.081907693 rad/s but starts at zero; RWInputY shifts from -0.024958902 to 0.056948791 rad/s. Other requested rates match. Do not classify the report's status=replayed as accepted.

Inspection of the source SLX confirms Torso Revolute Joint RzVelocityTargetPriority=None and both wrist Universal Joint RyVelocityTargetPriority=None. Their initial velocity value bindings are correct; the targets are disabled. Next add a per-simulation initialization policy through simulate_with_coefficients/SimulationInput: explicitly enable the schema's independent velocity targets and leave dependent targets unconstrained. Use mask priority selectors (their initialization callbacks are already repaired and tested), verify any direct hip-joint Specify/priority settings, and test restoration of source defaults. The failing nonzero-qdot replay is the red regression; do not weaken its 1e-8 acceptance threshold or overwrite source model defaults to hide the issue.

Evidence: marker_velocity_red_r2025b.json, marker_velocity_green_r2025b.json, pose_dependency_green_r2025b.json, initial_velocity_seed_unqualified_r2025b.json and initial_velocity_mismatch_audit.json. Reproduction: native_evidence/reproduction/initial_velocity_seed.m, with a fresh output directory containing driver_marker_payload.json from the saved bundle. Failed raw replay is C:/Users/diete/SimscapeTour9921/velocity-seed-01/initial_velocity_seed.mat, copied locally to C:/Users/diete/Repositories/simscape-tour-checkpoints/initial_velocity_seed_unqualified.mat. SHA256 matches on both machines: 824355037eb045a631e7635121c8496f65f635b31b44eee2c0bcf8941ea4fee5. The failed MATLAB process exited 1; the two qualification tests exited 0. No native process remains live.

Runtime is 317cc5d8d with copied schema and fit_golf_pose_seed changes in its shared directory; golf_marker_velocity_map.m is in remote scratch. Preserve those owned patches before checkout. R2025b remains mandatory; the full torque-driven swing remains open. Once initial velocity is verified, refit and extend prefixes with continuous polynomial efforts from the same t0 state.

## Changed-Geometry Forward Replay Verified

SELF records successful R2025b forward replay of the (14.5,12)-inch geometry seed for 0–0.05 s with zero polynomial efforts and Fast Restart off. Initial q agrees with the fitted pose within 7.247e-13; initial native-snapshot positions agree within 6.378e-13 m. Fifteen logged frame positions agree with the changed-geometry snapshot within 2.625e-13 m at 0, 0.025 and 0.05 s. Hip has no qualified direct position source and is omitted from that logged-frame comparison; its zero entry in the error array is not an independent Hip check. Native logged arm lengths are 0.3683 m per upper arm and 0.1524 m per forearm half, proving the requested geometry reaches the forward model. MATLAB exited zero, original workspace objects were restored and no model file was saved.

Evidence: native_evidence/geometry_candidate_replay_r2025b.json; reproduction: native_evidence/reproduction/geometry_candidate_replay.m, called with repository and output directory. Raw replay is C:/Users/diete/SimscapeTour9921/geometry_candidate_replay.mat. Runtime is 317cc5d8d; no native process remains live. This validates geometry/state application only, not anatomical identification or a torque-fitted swing. The earlier default-geometry torque checkpoint remains intact. Next estimate native constraint-consistent initial velocities and compare fixed-attachment fits over longer prefixes, retaining the geometry-identification limitations. R2025b remains mandatory.

## Bounded Starting-Pose Geometry Sweep

SELF adds fit_golf_pose_seed.m, extracting the existing pose objective into a reusable native helper with finite frame/coordinate contracts and strict final target/constraint acceptance. The R2025b TestCase first failed on the missing function, then passed recovery of a known translated native pose; the final test discovers its versioned fixture from the repository, without environment configuration. Evidence: pose_fit_red_r2025b.json and pose_fit_green_r2025b.json.

Nine fixed arm-length pairs were fitted at the initial pose using the same nine surface proxies and weak pose prior. All nine returned native status 1 and optimizer exitflag 3. Default (12,14) inches gave Euclidean proxy RMS 54.471 mm; best tested (14.5,12) gave 49.343 mm. This is a coarse single-pose seed, not anatomical calibration: the upper-arm optimum is at the tested boundary, surface proxies differ from joint centers, and no changed-length forward replay has yet been qualified. Keep the verified default-geometry torque checkpoint intact. Next constrain attachments/anatomy using multiple frames, estimate consistent initial velocities, then compare candidate geometry in the authoritative forward dynamics and grow polynomial-torque prefixes.

Runtime is 5309fb140 with fit_golf_pose_seed.m loaded from C:/Users/diete/SimscapeTour9921. The report's source_revision=9a381f018 denotes the unchanged physics base; geometry_sweep_bundle_receipt.json records the actual runtime and source hashes. Completed run: C:/Users/diete/SimscapeTour9921/geometry-sweep-02; MATLAB exited zero and no run remains live. Each candidate saved a separate numbered checkpoint. The earlier root-directory sweep rejected malformed target-array input before fitting; preserve it as failure evidence, not as geometry results. The corrected sweep stops on programming errors and rejects only explicit native constraint failures. Original workspace objects were restored and the model files remain unchanged.

A verified continuation bundle is stored at C:/Users/diete/SimscapeTour9921/geometry-sweep-02-bundle.zip and C:/Users/diete/Repositories/simscape-tour-checkpoints/geometry-sweep-02-bundle.zip; SHA256 ca61cfbe43a0800f04bb53190a2b03da25584213ddf37975141f306172ece64a. It includes input pose/targets, helper, test, sweep script, logs and all nine checkpoints. Numerical result: native_evidence/native_geometry_seed_sweep_r2025b.json. Reproduction: native_evidence/reproduction/native_geometry_seed_sweep.m, called with repository and a fresh run directory containing the two input JSON files from the bundle. R2025b remains the only required release.

## Native Arm-Length Bindings Verified

SELF records a successful R2025b native geometry probe (MATLAB exit zero). Defaults are UpperArmLength=12 in and LowerArmLength=14 in. Changing LeftUpperArmLength and RightUpperArmLength from 12 to 13 leaves all six measured arm-segment distances unchanged. Changing the shared UpperArmLength to 13 and LowerArmLength to 15 increases each upper arm by 0.0254 m and each forearm half by 0.0127 m. All three KinematicsSolver cases return status 1 with all independent targets satisfied. These are geometry/pose checks, not new forward-dynamics fits. Original model-workspace objects are restored; no model is saved. Native runtime remains at 9a381f018, and no probe process remains live.

Evidence: native_evidence/native_geometry_probe_r2025b.json; executable probe: native_evidence/reproduction/native_geometry_probe.m. The first snapshot took 38.51 s and subsequent changed-geometry snapshots about 9.9 s, making a small outer geometry search practical. Use the active shared parameters in inches, rebuild KinematicsSolver on each changed geometry, and apply identical overrides to the authoritative forward replay. Do not optimize the inactive aliases. Workspace values may be Simulink.Parameter objects; read their Value for numerical work and preserve their originals when restoring.

The surface-distance audit (native_evidence/arm_surface_distance_audit.json) covers all 654 frames: left/right elbow-to-wrist median distances are 0.2781/0.2817 m; these are surface-marker distances, not joint-center lengths. Right shoulder-back to elbow differs substantially from left shoulder-top to elbow, so do not treat those labels as symmetric joint centers. Next perform bounded geometry/attachment calibration with explicit anatomical priors and estimate a constraint-consistent initial velocity, then refit longer prefixes. Full-swing qualification remains outstanding.

## Independent Candidate Replay

The saved 25-marker, 0�0.05 s candidate passed a fresh R2025b process on DeskComputer: maximum coordinate difference from the saved fitted prediction is exactly 0 m; initial q differs from its seed by at most 1.4094098665928811e-9. Fast Restart was off; MATLAB exited zero. This verifies repeatability of this candidate only, not full-swing fidelity or physiology. Native source is 9a381f018c2fe96b1b36c23a3c7d8c5aaeb3f74e. Runtime checkout now points to that commit; its prior prefix-fit patch is preserved in a named stash, and the three .slx.r2025a backups remain untouched. No native fit/replay process remains live.

Evidence: native_evidence/cold_candidate_replay_r2025b.json. Reproduction: native_evidence/reproduction/cold_candidate_replay.m and cold_candidate_input.json (fixed coefficients, state, attachments, clock and expected marker array). The original raw replay is C:/Users/diete/SimscapeTour9921/cold_candidate_replay.mat. The first probe stopped before simulation on the unloaded-model precondition; the saved probe explicitly loads the model before constructing KinematicsSolver. Next calibrate active lengths and consistent initial velocity, then extend prefixes from t0 with continuous polynomial efforts. R2025b remains the only required release.

## Latest Saved Result and Checkpoints

SELF records the first completed native torque fit: 25 fixed body-marker attachments over 0–0.05 s, 0.176671 mm RMS / 0.280906 mm p95 / 0.594141 mm max, after 283 native calls including baseline and final checks. The optimizer hit max_nfev=10 (281 calls counted in its stage), so accepted_numerically is false. Initial native q agrees with the waist-constrained seed within 1.410e-9. Default lengths and zero initial velocities remain provisional. No full swing or physiological effort qualification is claimed. The fit process exited zero; no fit/observer process remains live.

Read [Incremental Checkpoints and Replay](CHECKPOINTS.md) from the repository root (or CHECKPOINTS.md beside REMOTE_EXECUTION.md). It gives exact paths, commands, limitations and ordered next actions. Source, inputs, exact original script, enhanced replay script and SHA256 manifest are preserved in first-prefix-50ms-20260910.zip on both machines: local C:/Users/diete/Repositories/simscape-tour-checkpoints/first-prefix-20260910 and remote C:/Users/diete/SimscapeTour9921/checkpoints/first-prefix-25. Receipt: native_evidence/first_prefix_bundle_receipt.json. The immutable evaluation-00283 snapshot in those directories contains the best warm-start parameters and complete history. These are candidate checkpoints, not serialized SciPy internals.

The tested tour_checkpoints.py helper rejects partial reports, preserves earlier snapshots and checks capture hash, marker order and effort scales for warm starts. The enhanced reproduction script automatically snapshots its existing five-evaluation reports and accepts --checkpoint. Three checkpoint tests, 23 prefix/capture tests, scoped Ruff/mypy and two R2025b native marker-adapter tests pass. The user explicitly requested incremental saving and breadcrumbs; retain this behavior for every subsequent run. Next validate a cold replay of the candidate, calibrate active UpperArmLength/LowerArmLength bindings (inches) and initial velocity, then grow the prefix with continuous polynomial inputs. Both native upper arms use UpperArmLength, not the left/right aliases. The full epic stays active.

## Current Native Fit Experiment

The two R2025b `test_simulate_golf_markers` tests pass after the missing-function red run. The adapter calls only `simulate_with_coefficients`, projects fixed offsets at native state timestamps, then interpolates Cartesian marker positions onto the requested clock. Invalid simulations, joint states, frame solves or coverage fail rather than becoming optimizer penalties. The shared Python prefix fitter now accepts a positive finite relative finite-difference step; twenty tests pass, including a quantized-oracle regression that stalls with the default step and converges with a resolved perturbation. Scoped Ruff and mypy pass.

DeskComputer's isolated `C:/Users/diete/SimscapeTour9921/python-r2025b` environment uses Python 3.12 and the engine installed from R2025b's local extern/engines/python directory. The original global R2024b engine remains untouched. The engine probe confirms MATLAB root C:/Program Files/MATLAB/R2025b. The waist-constrained pose seed in `native_pose_seed_waist.json` adds a Hip target from the four waist markers to the previous eight proxies: 54.478 mm RMS Euclidean proxy error, max 80.819 mm, hip proxy error 0.4 mm, native KinematicsSolver status 1 (all targets and physical constraints satisfied). This still uses provisional surface proxies; the initial forward replay now passes the q and fixed-offset marker assertions.

The first 22-marker experiment stopped on a forearm perturbation because canonical interpolated q at 1/120 s violated the closed chain (KinematicsSolver -1, only 9/27 targets satisfied). The corrected adapter extracts joint states on the original raw HipPositionX log clock, projects native poses, then interpolates marker positions. The stronger R2025b native test uses a 360 Hz output grid and 0.2 Nm forearm torque; both tests pass. The retry has 25 markers including all three head markers and has passed the previously failing perturbation. Baseline initial q matches the waist seed within 1.410e-9, and first-frame markers match their fixed-offset calibration within 1e-8 m. The completed experiment is `first_prefix_fit.py` in remote scratch, launched through the isolated Python environment with --repo pointing to the runtime checkout and --run-dir pointing to scratch. It fits 50 ms, 25 active markers, fixed offsets calibrated from the first frame, zero initial joint velocities and constant degree-six Bernstein controls. Parameters p in [0,2] map to effort=scale\*(p-1), with exploratory scales 1500 N for root translations and 200 Nm for angular joints; these are search bounds, not physiological qualification. Relative finite-difference step is 0.001, max_nfev is 10. Existing prefix_fit performs the optimization; MATLAB is the forward oracle. Report `first_prefix_fit.json` updates every five evaluations, logs are first_prefix_fit.log/.err. Inspect the existing process before restarting. Current files are copied patches over runtime commit 0d8b34004; do not checkout or modify its loaded source during the experiment.

The 22-marker selection is an early fitting stage, not full-capture coverage. RShoulderTop is missing initially; Marker_0 has unknown meaning; Uname\*36/37/38 are redundant waist derivatives; eight leg markers lack modeled legs. Contrary to an earlier working assumption, the native model DOES have a rigid Head/Neck body fixed to the Hub's rigid chain (SLX system_7475, Head216-Neck214-RigidTransform217-COMRod628-Hub7493). Include the three head markers as Hub-fixed attachments in the expanded objective; independent neck motion cannot be represented by the current 27-DOF topology. Do not classify head markers as absent geometry. Back markers are assigned to Hub's upper-torso body, shoulder skin markers to scapula bodies, upper-arm markers to shoulder bodies, wrist markers to forearms and club clusters to the rigid club. All attachment choices remain provisional pending multiframe geometry calibration.

Next: inspect the first native fit result, retain all source scripts and numerical evidence, then expand the marker set and time prefix. No full torque-driven tour swing has been qualified. R2025b remains mandatory.

## Current R2025b Marker-Frame Qualification

Working branch: `feat/9921-simscape-tour-matching`; implementation commit SELF (resolve with git rev-parse HEAD); PR not created. SELF extends the native frame schema to 16 frames and add intrinsic XYZ world-from-body rotations. Twelve orientations agree with native logged rotation matrices. `HipGlobalPosition` is the fixed hip-joint BASE position, not the moving hip follower: SLX system_7475 wires sensor 6266 through converter 6271 to that log. The Hip frame intentionally has no direct position source; its test uses measured sliding coordinates resolved through the model's PlaneTilt. Do not attach waist markers to HipGlobalPosition.

`simulate_with_coefficients` now explicitly disables Fast Restart when requested; its six workspace regressions pass in R2025b. A native zero/2/zero Nm probe reduced subsequent 20 ms runs from 55.88 s to 2.87/2.80 s and reproduced the first zero-torque joint trajectory exactly. Its successful result report does not imply clean process shutdown; the four-run cold-replay probe with explicit cleanup now succeeds with MATLAB exit zero, exact warm/cold joint trajectory equality and FastRestart off on the fourth run. Evidence: native_evidence/fast_restart_cold_probe.json.

Current validation: all seven R2025b tests pass: four marker-projection tests, two intrinsic XYZ tests and the expanded native frame replay (16 positions and 12 logged orientations at three times). Evidence: native_evidence/rotation_frames_green.json; MATLAB exits zero. The projector uses fixed body-local offsets in metres and rejects invalid body indices, attachment counts and improper rotation matrices. Red evidence was recorded before implementation. No capture marker calibration or torque optimization has completed.

Resume on DeskComputer using R2025b explicitly. DeskComputer runtime is now detached at validated commit `0d8b3400421c8c71807356eba4d9b7cdb273c33f`. The copied validation patches were preserved in a named stash before checkout; the three historical .slx.r2025a backups remain untracked and untouched. Local implementation is published and clean; this documentation checkpoint is SELF. Scratch is `C:/Users/diete/SimscapeTour9921`. The native frame and cold-replay probes are complete. Inspect JSON status and MATLAB exit separately when repeating them. Next calibrate fixed marker attachments and initial velocity before optimizing the first torque prefix through the sole simulation wrapper. Local design-manual governance passes but release remains blocked by the existing missing calculation inventory. The central development-log validator reports pre-existing duplicate/missing fields in other entries; DL-#9921 adds no reported errors. R2025b remains the sole required release. Repository Management policy PR #1633 is merged.

SELF adds build_golf_kinematics.m and golf_kinematic_schema.json: 27 coordinate identities resolved by block path/primitive and 15 qualified sensor frames. A fresh R2025b TestCase replay passes for all frames at three timestamps; independent saved-run parity is within 2.054e-13 m. Clubhead must use Transform Sensor10/F, not the adjacent sensor (127 mm offset). Kinematics solves take roughly 1.5-6 ms after construction. This enables pose calibration only; contacts, joint limits and torque dynamics require the authoritative forward replay. The provisional first-frame seed completed: eight surface proxies improve from 1.298 m to 69.635 mm RMS Euclidean distance (40.204 mm per-coordinate RMS), max proxy distance 106.547 mm. The optimizer ended with flag 2 and KinematicsSolver flag -1 (physical constraints satisfied, some requested joint targets missed); use the returned q, not optimizer x. A fresh R2025b 20 ms forward replay of returned q succeeds, reproducing joint components within 1.037e-12 and all frame positions within 9.722e-13 m. Evidence: native_pose_seed_r2025b.json, native_pose_replay_r2025b.json and initial_pose_seed.png. This is an initial-pose seed only; full marker attachments, geometry, initial velocities and continuous torque fitting remain required. Reference artifacts are at C:/Users/diete/Repositories/reference-fit-artifacts-9914; their surface-proxy fits are seeds, not native full-dynamics evidence.

SELF adds capture_fit_sim_options(duration_s), which builds on default_sim_options and keeps both killswitch values at one for the complete requested horizon. Three R2025b native tests pass after a recorded red run, including a 1.81 s wrapper replay across the old one-second cutoff and preservation of source workspace defaults. This is the required starting configuration for capture fitting; add fixed geometry and initial-state overrides to its input_overrides struct. The R2025b KinematicsSolver probe now succeeds on the exact model (73.15 s construction); include both src/model and genpath(src/functions), as in the normal runtime setup. Its joint-position/velocity table is recorded in native_evidence/kinematics_solver_probe_r2025b.json. Next use this solver to qualify body-frame outputs and initialize the capture pose; final acceptance remains a torque-driven forward replay. Full marker calibration and torque fitting remain outstanding.

R2025b actuation audit (SELF): the loaded GolfSwing3D_Kinetic model contains 13 physical joint blocks exposing 27 coordinates, all InputTorque/ComputedMotion. The stale readable main-model snapshot contains a kinematic RE definition absent from the loaded SLX; use the loaded model as authority. Sixteen scalar ActuatorTorque logs exist (scapula, shoulders, wrists and spine); the remaining 11 efforts need qualification. Default KillswitchStepTime is 1 s, shorter than the 1.814 s driver capture, so the fitting configuration must keep actuation enabled through the complete target. Evidence: native_evidence/actuation_audit_r2025b.json. The 2 Nm LSInputX probe succeeded in R2025b: all 21 actuator samples equal 2 Nm over 20 ms, with an explicit 2 s killswitch override (49.67 s wall time). Evidence: native_evidence/torque_probe_r2025b.json; raw replay remains in remote scratch torque_probe_sim_out.mat. This verifies one polynomial-to-actuator path, not all 27 efforts or a fitted swing. Next: qualify the remaining efforts, preserve actuation for the full horizon, and register native body markers and initial pose against the capture.

R2025b qualification: all 22 native tests pass on DeskComputer in MATLAB 25.2.0.3177638 (R2025b) Update 5. The actual 20 ms forward simulation succeeds with finite joint states and club/grip positions (95.89 s). Source revision: `4191ca3aca126216ea02e26e485ff72908f3ad2b`. Evidence: `native_evidence/native_r2025b_suite.json` and `native_evidence/baseline_r2025b.json`. Actuator-torque extraction and full-swing fitting remain unqualified.

## Required MATLAB Release

MATLAB R2025b is the required execution, model-save and validation release for the Simscape golf model and tour-average matching epic #9921. The user has the complete required licensed feature set in R2025b. R2026a is not a requirement; do not select it from PATH or use its successful probes as R2025b acceptance evidence. On DeskComputer and ControlTower launch `C:/Program Files/MATLAB/R2025b/bin/matlab.exe` explicitly. Preserve historical reports with their actual release; run acceptance checks in R2025b. The full forward-dynamics matching goal remains active with this constraint.

## Verified Connectivity

On 2026-09-09, SSH over Tailscale succeeded from OGLaptop to the Windows
`deskcomputer` and `controltower` aliases using the existing SSH configuration,
batch authentication and strict checking of the existing host keys. No firewall,
SSH-server or credential configuration was changed. The offline Linux
ControlTower monitoring entry is a different peer; use the Windows alias.

Required executable on both Windows machines:

- `C:/Program Files/MATLAB/R2025b/bin/matlab.exe`

PATH pointed to R2026a at discovery; use the explicit R2025b executable. No MATLAB process was active at discovery. The local
OGLaptop directories contain libraries but no runnable MATLAB installation.

## Current Execution State

Published runtime checkout: `4191ca3aca126216ea02e26e485ff72908f3ad2b`.
Only three preserved `.slx.r2025a` backups remain visible as untracked files.
Use R2025b for all new execution; R2026a results below are historical.

Primary execution host: DeskComputer. Fallback: ControlTower.
Current execution source:
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime`.
Initialization repair checkpoint (SELF, #9927): all three referenced joint files
are saved natively in R2025b Update 5. Gimbal targets now use their instance
parameters; embedded mask callbacks implement None/Low/High for all position and
velocity priorities. The callbacks have no added MATLAB-path dependency.
The native selector tests cover 36 settings and independent instances, migrations
are idempotent, and all 22 tests pass together in R2026a. `baseline_priorities`
completed the actual model run with finite q/qd and club/grip at 0.02 s, taking
39.18 s. Tau remains unavailable through the current canonical extraction.
See the checked-in `initialization_model_manifest.json` and native reports.
Block names and physical connections are unchanged. Universal/revolute default
target-enable flags now reflect their None defaults; instance callbacks apply
the selected priorities when the main model initializes.

The earlier backward-export experiment is not delivered: MATLAB warns that
backward Simscape export is unsupported. Original `.slx.r2025a` backups and the
exploratory export remain in the isolated remote workspace/scratch for evidence.
The delivered files use native R2025b saves and were exercised in R2026a.

Earlier checkpoint history follows; its pending actions are superseded by the
verified initialization repair above.
Current published base is `2d3903bfd066df4ddf94dee845833fa82d024197`; SELF adds
direct joint extraction in copied source files. `joint_sensor_green.json` has
five passing tests; `baseline_joint.json` records an actual 20 ms run with finite
q/qd and club/grip positions, but missing tau. `initial_motion_audit.json`
extracts all 27 q/qd/qdd channels from the earlier raw baseline without a new
simulation. Use direct joint buses: the consolidated RScap angular acceleration
X is incorrectly a three-component signal, while its direct sensor is scalar.

Issue #9927 is the next blocking model repair. `shoulder_parameter_audit.json`
and source inspection show that `Kinetically_Driven_Gimbal_Joint.slx` ignores
instance initial targets/priorities: its primitive uses LSStartPosition and
LSStartVelocity for both shoulders and hard-codes High priorities. The actual
initial shoulder angles are identical. Sibling joint priority controls also
need qualification. No SLX edit has been applied yet. Native model geometry,
closure and actuation must be retained when repairing this interface.
This new isolated checkout starts at published `d1c0ab391` and receives the SELF
repair that avoids parameter-file discovery when joint order is explicit.
Thirteen native tests pass with only its shared matching directory added to the
MATLAB path (`native_tests_isolated.json`); the preceding clean-path failure is
retained as `native_tests_published.json`. Check `git rev-parse HEAD` and status
before resuming. The preliminary checkout described below is retained evidence,
not the source for new fitting runs.
Remote scratch location: `%USERPROFILE%/SimscapeTour9921` (under the `diete` user).
A hidden R2026a batch process completed `runtime_probe.m` with exit code 0,
writing `runtime_probe.json` and `runtime_probe.log` there. MATLAB version is
`26.1.0.3203278 (R2026a)`; license availability checks for Simulink, Simscape,
Simscape Multibody and Optimization Toolbox all returned 1. The copied report
is `deskcomputer_runtime_probe.json` beside this document.

Keep the SSH connection alive and use `Start-Process -Wait -PassThru` with
`-WindowStyle Hidden`, MATLAB `-wait -batch`, and explicit stdout/stderr/log
files. The first detached probe produced no result after SSH exited; the
waited invocation completed successfully. A started process is not evidence
of a successful MATLAB run.

An isolated remote worktree now exists at
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-9921`, detached
at the receiving clone's `e4336293492b5a4ad6948c8a9949c1531ace4c81`. This is an
exploratory baseline checkout, not the published foundation branch. The main
SLX SHA-256 matches the local source exactly:
`daca9a90ad0ab819c7d61641ed594b8f230f8658f2a5f4ce34026db44b52ddc9`.
`model_audit.m` completed with exit code 0 and status `loaded`. The workspace is
stored in the model file; preload/postload/init callbacks are empty. Defaults
include `ModelingMode=3`, `KinematicMode=3`, `LocalDampeningEnable=0` and
`DampeningGlobalGain=10`. These selectors do not replace checking primitive
actuation. Its full JSON and log are in the remote scratch directory.
`baseline_probe.m` completed a 20 ms run in 97.53 s. Its requested coefficients
were not qualified: native regression subsequently proved that the wrapper's
global overrides were shadowed by the model workspace. The local wrapper now
targets the model workspace explicitly for coefficients and input overrides.
Raw-output retention is opt-in and bypasses the cache. These changes and
`shared/tests/test_simulation_input_workspace.m` have been copied into the remote
worktree; preserve those uncommitted edits before switching its revision.

`workspace_tests_red3.json` records two expected failures (coefficient 5 produced
1; requested raw evidence missing) and one pass. `workspace_clock_tests.json`
records five passing native regressions, including geometry overrides and cache
bypass. Eight `resample_logged_signal` tests also pass after observed missing-
function failures. Compact reports are copied into `native_evidence` here.
`baseline_corrected.m` completed with exit 0 in
35.21 s, retaining `baseline_corrected_sim_out.mat`. Club/grip arrays are finite;
joint q/qd/tau are missing in canonical extraction. Raw `CombinedSignalBus` is
available along with `tout`, `xout` and `simlog`. Qualifying its signal names and
actual timestamps exposed sample-index interpolation in the original extractor.
The repaired extractor preserves timeseries clocks, requires matching `tout` for
array logs, and leaves uncovered samples NaN. `baseline_clock.m` completed with
exit 0 against the actual model; inspect its JSON/log/MAT in remote scratch.
Native parameter discovery yields
27 polynomial channels (189 coefficients), including three translation forces.
Keep force and torque units distinct. `unit_audit.json` proves native joint angle,
angular velocity and acceleration converters use degrees, degrees/s and
degrees/s^2. Canonical q/qd/qdd claim radians: explicit conversion is required
when adding the nested signal map. Several revolute buses omit actuator torque;
do not substitute reaction torque or coefficient evaluation as measured torque.

Source worktree on OGLaptop:
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour`.
Branch: `feat/9921-simscape-tour-matching`.
Saved foundation commit: `62c7f845ee5fc4ef88e3b37a2948099841a3d3d4`.
Epic #9921; foundation #9924. The foundation has 17 passing focused tests,
scoped Ruff/mypy, file-size and governance checks. All commit hooks passed.
The foundation and remote-handoff commit `9dd2ff95f` are published on origin.
Normal commit and push hooks passed; no hooks were bypassed. Native interface
work is tracked by #9925; SELF is its local implementation commit. The remote
preliminary worktree still has copied source edits on detached `e43362934`.
Subsequent MATLAB scripts must use the current `...-runtime` source root above;
earlier exploratory scripts intentionally retain their original source paths.
Do not edit tracked files concurrently with pre-push checks.

## Safe Resume Procedure

1. Read `AGENT_HANDOFF.md`, this document and the sibling `README.md`. Read the
   receiving repository's own AGENTS/CLAUDE instructions before edits.
2. Check epic/foundation claims and register a unique session through central
   `Repository_Management/scripts/agent_communicate`. Inspect the inbox before
   editing. Reference-model #9914 is separately owned; preserve its worktree.
3. Inspect existing repositories and MATLAB processes. Create an isolated
   worktree from the published task branch, not a dirty existing checkout.
   Verify its commit and initialize the pinned Tools submodule if Python is used.
4. Inspect the runtime probe JSON/log. Confirm Simulink, Simscape, Simscape
   Multibody and the intended optimizer are available, and record the release.
5. Locate the model at
   `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx`.
   Inspect the live workspace, callbacks, referenced joints, actuation modes,
   initialization and output logging before the first short baseline simulation.
6. Preserve model/data hashes, resolved initial state, joint geometry/inertias,
   solver settings, parameter order and actual output clocks. Use the existing
   `simulate_with_coefficients` forward authority, extending its native marker
   extraction as needed; do not introduce a second simulation implementation.
7. Save each experiment to a fresh run directory. Record process ID, exact
   command, status and artifacts here. Copy compact results back over SCP.
   Run optimizers beside MATLAB so Tailscale only carries orchestration and
   artifacts, rather than one network transaction per optimizer evaluation.

## Remaining Model-Matching Work

- Qualify full-body marker attachments and the initial geometric residual floor.
- Calibrate constant lengths, inertias and feasible initial state with bounds.
- Integrate the tested prefix optimizer with actual MATLAB marker outputs;
  add durable checkpoint/resume, regularization and physical constraints.
- Progress from the initial short interval through the complete available motion.
- Independently replay without motion prescriptions or window state resets,
  tighten solver tolerances, and record marker/phase errors and constraint limits.

Native initialization and a cold-replayed 0.1 s torque fit are recorded. Anatomical
geometry identification, full-swing torque profiles and full-swing validation remain outstanding. Do not mark the
epic complete until that evidence exists.
