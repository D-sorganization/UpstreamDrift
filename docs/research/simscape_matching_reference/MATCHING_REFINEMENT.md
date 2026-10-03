# Simscape Motion Matching Refinement Requirements and Evidence Status

## Current Verified Status (October 2, 2026)

The selected shareable clips remain tour 13.188/40.121 mm and owner 16.643/36.865 mm mean/max frame position RMS. Both use Human ellipsoids and IK; full forward dynamics remains unqualified. The controlled common-source/runtime tour comparison produced identical poses with the old and current model, so the newer model binary alone does not explain the changed fit.

Two new pure helpers passed 18 native initialization-mapping tests and 8 marker-cluster tests, with zero failed/incomplete tests. Parent review added actual failing regressions for matrix-shaped poses and a scale-dependent degeneracy cutoff before fixes passed. These counts are separate from the earlier 187-test integration suite. The first native state-target-expression check failed (original loop status 1, mapped -1, scalar discrepancy 16.320 degrees). A compiled diagram update refreshed stale masked start values; a fresh KinematicsSolver then reproduced all 22 joint frames for both captures, with maximum translation error 4.44e-16 m and rotation-matrix error below 9.49e-15. The run exited naturally with code zero at 10:03:13Z. No simulation ran and no model was saved; state-target priorities, controller references and physical initialization remain unqualified.

HeadTop/HeadFront/HeadSide tracks are finite and nondegenerate in all 654 tour and 367 owner frames. Pair-distance variation reaches 6.761% for the owner; cluster axes require body-frame calibration and do not establish anatomical orientation. A conservative optional head-motion candidate is under development and has not replaced selected clips. See `native_helper_review_20261002.json` and `head_track_audit_20261002.json` in the research reference directory. The latest LaTeX source remains uncompiled: the built-in compiler reports `Unable to find standard directories for platform`.

## Historical Experiments and Requirements

## Verified Full Human Exports

Both full sampled swings now use GS3DX_Human ellipsoids with calibrated foot orientation and freshly calibrated marker offsets. Native solves exited naturally with code zero at 2026-10-02T05:15:54Z (A) and 05:23:22Z (O). Position RMS excludes orientation residuals.

| Capture  | Fit mean / max frame RMS (mm) | Human mean / max frame RMS (mm) | Measured foot samples |
| -------- | ----------------------------- | ------------------------------- | --------------------- |
| A, tour  | 12.746 / 40.209               | 13.188 / 40.121                 | 55 per foot           |
| O, owner | 26.558 / 116.576              | 16.767 / 37.626                 | 45 of 46 per foot     |

The owner fit improves both position metrics. The tour fit has a small mean position tradeoff while resolving the reversed-foot ambiguity. Maximum left/right foot angular errors are 12.83/19.82 degrees for A and 19.55/20.38 degrees for O; these diagnostics do not establish anatomical or contact qualification.

Four 800 by 600 H.264 MP4s were completely decoded at 30 fps, with 55 tour and 46 owner frames. Native caption-corrected rerenders exited naturally at 05:28:11Z and 05:31:30Z. The Desktop shareable ZIP contains both views, sanitized provenance, hashes and a same-capture/same-frame comparison; no raw captures or pose caches. Original Fit clips remain preserved.

Owner wrist offsets remain approximately 55/62 mm and the club-target offset 106 mm. Native parameter and grip-contract tests pass; the functional-grip fit worsened held-out position errors and was rejected. These are IK clips; forward dynamics remains unqualified.

## Overview and Refinement Scope

This document specifies the refined motion-matching and modeling requirements governing Simscape golf swing evaluations, extending the active matching program (2026-10-01).

- **Visual Style and Model Policy**: Future shareable matching exports must use the refined Human ellipsoid golfer geometry, conforming to the tour-average reference visual standard. Segment lengths, coordinate reference frames, joint mappings, and physical parameter definitions must be explicitly tracked and validated when changing visual appearance or model topology.
- **Export Boundary and Qualification**: All current `capture-A` and `capture-O` video exports are kinematic inverse kinematics (IK) visualizations, not forward-dynamics-driven swings. Changes in visual geometry or rendering shaders do not qualify physical dynamics or contact stability.
- **Future Ellipsoid Export Policy**: Export of refined Human ellipsoid visualizations requires calibrated foot roll/pitch/yaw angles, private source hash / cache management, and independent recording of 14 target RMS errors and orientation residuals.
- **Observability Limits**: Head and neck orientation is not yet constrained by this solver and requires future orientation constraints; the model must not be characterized as anatomically complete.
- **Privacy and Provenance**: Capture aliases `capture-A` (public tour reference) and `capture-O` (owner optical capture) protect private identities. Public records publish only neutral aliases, cryptographic hashes, and aggregated metrics; raw capture binaries, absolute private paths, and local caches remain outside git.

## Capture Specifications and Avatar Interpretation Limits

- **Capture A (Tour Reference)**: Sampled at 360 Hz across 654 frames. Used as the tour reference capture.
- **Capture O (Owner Swing)**: Optical motion capture sampled at 240 Hz across 367 frames.
- **Evaluation Principle**: Golfer playing ability or technique cannot and must not be inferred from avatar appearance or kinematic fitting distortions. Avatar distortions may result from marker protocol mismatches, segment scaling assumptions, marker dropouts, or uncalibrated sensor offsets, and cannot establish golfer skill.

## Verified Baseline Desktop IK Exports

Four Desktop Fit IK H.264 30 fps video deliverables represent the current verified baseline:

- **Capture A Baseline**:
  - Sample coverage: 55 frames.
  - Mean measured-target RMS: 12.745748 mm.
  - Maximum measured-target RMS: 40.208544 mm.
  - Diagnostic worst sampled target: `trailElbow` at 96.517 mm (frame 517).
  - Native process termination: Natural exit 0.
- **Capture O Baseline**:
  - Sample coverage: 46 frames.
  - Mean measured-target RMS: 26.558095 mm.
  - Maximum measured-target RMS: 116.576104 mm.
  - Diagnostic worst sampled target: `clubhead` at 375.358 mm (frame 89).
  - Native process termination: Natural exit 0.
- **Baseline Integrity**: These four MP4 files are verified historical baselines generated using the legacy GS3DX_Fit cylinder model. They are preserved as reference benchmarks; refined Human clips are available separately; the original files remain preserved.

## Candidate Evaluations and Backward-Pass Rejection

- **Owner Backward-Pass Candidate**:
  - Configuration: Evaluated using identical model identity and marker offset set.
  - Mean measured-target RMS: 32.185137 mm.
  - Maximum measured-target RMS: 46.052114 mm.
  - Process receipt: Natural exit 0 (receipt timestamp: `03:57:32.790355Z`).
  - Decision: **REJECTED**. While the backward pass reduced peak tracking error from 116.58 mm to 46.05 mm, the mean tracking error worsened from 26.56 mm to 32.19 mm. Under this experiment's selection rule, candidates that degrade mean tracking fidelity cannot be selected merely for reduced peak residuals.
- **Marker Offset Norm Distribution**:
  - Fitted marker offset norms on `capture-O` cluster heavily in the 70–100 mm range, whereas `capture-A` offsets are substantially smaller.
  - Scientific assessment: This offset distribution is hypothesized to compensate for capture protocol definitions, marker centroid shifts, and skeletal geometry discrepancies, rather than serving as evidence of physical provenance or golfer motion.

## Coordinate Topology and Joint Role Resolution

- **Independent Fit Coordinates (37 vs 39 vs 33)**:
  - In the Human model topology, actual joint roles yield **37 independent coordinates** during fitting with a closed grip kinematic loop.
  - Earlier failures referencing 33 coordinates reported a false 39-coordinate count due to hardcoded indexing errors and mismatched coordinate indexing tables. Clarifying that the actual Human kinematic topology possesses 37 independent degrees of freedom resolves this structural index bug.
- **Native 3-Frame Fit Probe**:
  - Frame sample: `[1, 13, 25]`.
  - Position RMS: 13.296 mm, 12.536 mm, 12.374 mm (all finite and well-conditioned).
  - Left foot orientation residual: 165.0 deg, 167.0 deg, 167.2 deg against calibrated shoe orientation matrix $R$.
  - Process receipt: Natural exit 0 (receipt timestamp: `04:54:21.730288Z`).
  - Physical finding: This experiment demonstrates a large orientation ambiguity in this position-only fit.

## Foot Orientation Formulation and Calibration Trials

- **Calibrated Foot Orientation Methodology**:
  - Uses shared calibrated foot triads and SVD address mean orientation:
    $$R = F_f F_0^T R_z(\text{yaw})$$
  - Sole flatness assumption: Assuming a flat sole on the ground at address (frame 1) is a mathematical alignment assumption, **not** dynamic normal force or contact evidence.
  - Fail-closed contract: If frame 1 is absent or corrupt, the alignment routine fails closed.
- **Foot Orientation Weighting Trial**:
  - Tested with orientation objective weight 0.1, reusing existing Fit marker offsets.
  - Position RMS degraded significantly: 51.263 mm, 43.243 mm, 39.399 mm.
  - Left foot orientation error improved: 36.505 deg, 17.058 deg, 13.764 deg.
  - Process receipt: Natural exit 0 (receipt timestamp: `04:57:13.758186Z`).
  - Decision: **REJECTED**. The resulting position error degradation is unacceptable for production export.
- **Active Calibration Status**:
  - Fresh Human calibration on frames [1,13,25] completed with natural exit 0 at 2026-10-02T05:01:00.641030Z: position RMS 1.188/1.195/1.208 mm and both foot angular errors below 0.7 deg. These are calibration frames, not held-out validation. Both full Human exports completed and were fully decoded.
  - Full $\text{SO}(3)$ 18-component chordal formulation with weight 0.1 remains exploratory and is not certified (it excludes position RMS, and the gap foot metric returns `NaN`).

## Human Ellipsoid Visual and Geometric Adapter

- **Adapter Verification**: 10 native MATLAB adapter tests pass with natural exit 0.
- **Geometric Scaling**: The adapter scales longitudinal segment geometry to match subject skeletal proportions.
- **Fixed Geometry**: Artistic widths, head geometry, and shoe dimensions remain fixed to preserve reference styling.
- **Mass and Inertia Invariance**: Baseline segment masses and moments of inertia are **NOT** scaled to the owner's measured 104.3 kg body mass. Consequently, forward dynamic simulation using these visual assets remains unqualified.

## Stance Stability and Open Scientific Gates

- **Original Worker Commit Review**: Reviewed commits `144e81188` through `f2ca443a0`; subsequent original-worker documentation and queued experiments remain separate.
- **1-Second Stance Simulation Failures**:
  - Resampled, quiet, frozen address, and still upper-body tests all fail physical stance stability.
  - Ground reaction normal force decays to zero after 0.5 s.
  - At 1.0 s, body tilt reaches extreme angles: 101 deg, 108 deg, 114 deg, and 124 deg.
  - Pelvis drops far below the ground contact plane.
  - Historical process `EXIT 0` was an unhardened execution receipt and did not constitute physical qualification.
- **Queued Damping Experiment**: The unsaved damping-times-ten test completed and failed: initial right-heel force 15,513 N, force zero by 0.60 s, tilt 117 deg and pelvis z -1.83 m at 1.0 s. It must not be repeated alone. The original worker subsequently queued a composed pelvis-level pose check stopping at 0.02 s; no outcome is claimed. This test must not be duplicated, cancelled, prematurely saved, or claimed as successful. Ongoing original worker processes must be preserved.
- **Scientific Gates**:
  - Gate #11156 (lead elbow and wrist tracking): **RED**.
  - Gate #11160 (clubhead kinematics and impact face alignment): **RED**.
  - Gate #11173 (open-loop full-swing forward dynamics): **RED**.
  - No full-swing open-loop forward dynamics qualification exists.

## Software Test Inventory and Verification Boundaries

- **Pure Mathematics and Architecture Tests**: 102 tests PASS, 0 skipped, following the integration of full orientation helpers.
- **Strict Validator Tests**: All five additional strict validator tests pass within the 102-test focused suite (0 incomplete), including missing frame-1 calibration and degenerate address mean rejection.
- **LaTeX Reference Compilation**: The earlier 16-page research reference compiled and was visually reviewed. Latest full-export source additions remain uncompiled because the built-in editor compiler reports a platform directory error. The canonical engineering design manual remains governed under `manuals/upstreamdrift` QMD.
- **Active Program Status**: Full refined ellipsoid exports for Capture A and Capture O are verified; the overall epic still requires continued tour and owner physical dynamics matching, and formal review through protected branch delivery. The goal remains unfinished.

## Subject Physics and Rejected Functional-Grip Candidate

The native subject-physics adapter/catalog passed 18 tests with zero incomplete tests (natural exit 0, 2026-10-02T06:03:51Z). It applies a declared mass and whitelisted fitted lengths in memory, preserves joint expressions and parameter metadata, and audits equipment through the existing native inertia catalog. Parameter consistency is not a dynamics qualification. The selected Human videos still use the baseline physical parameters because mass does not drive IK.

The sphere solver passed 15 tests and the controlled grip contract passed 12 tests, both with zero incomplete tests. A functional-grip owner candidate completed naturally with code zero at 06:15:53Z. Its 24 held-out samples worsened mean/max frame RMS from 26.011/37.626 to 27.969/41.278 mm; full-swing values worsened from 16.767/37.626 to 17.948/41.278 mm. It was rejected. Its smaller lead-wrist offset alone is insufficient for selection. Capture lengths were estimated from the complete recording; held-out status concerns grip/offset calibration only. See `grip_candidate_comparison_20261001.json`.

## Fresh Contact Geometry and Capture Metadata

A read-only native forward-kinematics audit of the selected owner's first pose, with fitted Human geometry and declared 104.3 kg mass, exited naturally with code zero at 06:42:14Z. Against the existing model ground plane, the ten contact-sphere clearances ranged from -64.399 to -48.619 mm (15.780 mm spread). This geometric penetration is not a simulated force measurement. Simulation initial targets were not updated, no dynamics was run and no model was saved. The pelvis frame's local up vector tilted 57.889 degrees; that frame-relative quantity does not establish an anatomical torso angle or a causal instability mechanism. Existing World Frame selection matches the cached IK. Ground placement and a consistent gravity/contact initialization must be verified separately.

Both actual C3D files specify metres and contain no EVENT parameter group. Under ezc3d 1.7.2, A has 717 negative-residual samples, all already nonfinite in XYZ; O has no negative residuals but 1,173 nonfinite XYZ samples. These counts cover all marker tracks. The fourth point row is homogeneous XYZ1, not residual metadata. Neither peak-speed timing nor address-line crossing establishes measured ball contact. Importer units/missingness contracts and export dependency/runtime provenance are under review; no new export identity is yet accepted.

## Import/Export Hardening Checks

Capture-unit and missingness conversion: 13 native tests pass. Runtime fingerprint: 17 native tests pass. Export contracts plus live runtime query: 12 native tests pass. All have zero incomplete tests; actual runtime values are recorded in the handoff. Fresh full A/O exports subsequently completed; current selected outputs and model/runtime identities are recorded in the current status and canonical handoff. The inventory records direct components and selected sources, not all transitive packages or external mesh assets.

### Reviewed Native Helper Checkpoint

The combined suite passed 213 MATLAB R2025b checks with zero failed or
incomplete tests; the serialized process exited naturally with code zero at
10:27:31Z. The 18 mapping and 8 cluster tests are included in that total.
See `native_helper_integration_tests_20261002.json` in the research reference
directory. These are software/parameter checks, with no physical acceptance.

The unfinished head prototype was withheld after review found insufficient
gap/coverage validation and an unproven baseline-preservation claim. Its
source, tests and partial execution evidence are retained privately for
continuation; the baseline IK source was restored before the combined check.
No head-constrained candidate or new selected video is claimed. Protected CI
still requires a fresh run after regenerating the monolith register. The
leaderboard runner's missing local action remains unexplained: its checkout
log already records sparse-checkout disable, so an additional cleanup patch
was not accepted on the proposed explanation alone.
