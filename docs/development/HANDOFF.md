# Scapula and Quiet Torso Matching Handoff - #11329

- Branch: `feat/simscape-scapula-protraction-20261002`; commit SELF; PR #11351; development entry `DL-#11329`.
- Scope: requested scapula protraction, quieter native spine/torso matching, direct back-marker residuals, complete measured-marker overlays, neutral Model Swing labels and undistorted 1080P exports. Independent dynamics remains separate and unqualified.
- Validation: MATLAB R2025b Update 5; 50 focused native tests passed. Entire disabled-feature output equals legacy output; default/explicit scapula parity checked. Mirrored native joint-role schemas cover 13 moving-scapula variants; Human has physical forward-direction perturbation evidence. Published Human full fits exited naturally with code 0 and equal the successful native quiet-spine trial exactly. Full trajectories were not fitted for every variant.
- Selected profiles: back markers 0.5, head-axis weights 0.2/0.16, posture 0.04, smoothing 0.025, base gap 0.1, ROM 0.025, feet 0.1, scapula prior 7.5, native spine bend 15 and twist 45 degrees relative to the original address seed, spine posture/smoothing 0.08 and soft hinge 0.35. Explicit lower-body gap weights 1 retain interpolated priors; displayed raw markers remain missing. Hard bands release after top. Full 3D head error around 43 degrees remains a limitation of the selected two-coordinate-neck guide.
- Results: previous spine bending excursions 47.57/60.65 and 39.34/49.38 degrees reduce to 15.44/16.53 and 17.49/15.17. Mean/peak 14-target RMS is 16.57/32.84 and 19.81/37.78 mm; raw back-marker mean RMS is 26.73/38.23 mm. These are native kinematic results, not clinical joint measurements. Segment dimensions and original position offsets are preserved.
- Delivery: all eight corrected 1080P/30-fps MP4 exports passed native checks with process code 0; all 404 frames decoded and both views were reviewed. Desktop file hashes and ZIP CRC/membership were verified. Previous Desktop versions were rejected for excessive spine motion and pixel stretching. Do not use their earlier decode receipts as acceptance of the corrected videos. New marked/clean files use Model Swing 1/2 titles, all measured channels, cyan back/waist points and identical camera bounds.
- Source checks: pinned Ruff 0.15.17 passed format/lint. The existing LaTeX source was updated in place; built-in compilation failed with `Unable to find standard directories for platform`.
- Dependency: branch base includes 31 parent commits from draft #11256, including shared-capture #11172. Review this refinement's own commits separately. Do not merge the unqualified parent scientific program through this dependent draft.
- Next step: Desktop previews are complete. Source PR #11351 remains draft pending parent acceptance; no protected-main merge or CI success is claimed. Main documentation conflicts were resolved preserving concurrent records; tested Simscape sources and models were unchanged.

- Variant refinement: `target_scope=auto` selects available native targets, masks unavailable foot diagnostics and validates keyed seeds against actual native coordinates. Baseline/Slim/Quat complete sampled A/O trajectories have passed. Contact, Golfer and Fit have also completed both swings; remaining stages are running. The rigid-foot FullBody stage keeps its topology and needs native acceptance of the new closed-leg/upper-body adapter. An independent all-frame native replay is queued. Do not interpret static schema coverage as complete variant matching acceptance.
- Current checks: 76 native contracts pass, including original Human warm-start contracts; exact disabled-feature Human output parity passed before the final grounded-stage adapter. The CI retry ran 20,336 passing tests and found a stale generated monolith register; regeneration passes five focused tests. Full CI success and protected-main integration remain pending.

---

# OpenSim Muscle Lines of Action — #11301 (FTO-16)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11301`
- Branch: `feat/issue-11301-fto16-opensim-muscle-lines`; commit: SELF; PR: see the PR for this branch (`Closes #11301`, `Refs #11285`)
- Governing issue: #11301 (epic #11285, ADR-0052); DL entry `DL-#11285`
- Completed: `OpenSimForceTorqueSource.muscle_wrenches(state)` plus `include_muscles=True` constructor flag; `sample` includes them. `MUSCLE` wrenches at the first and last force-carrying `getPointForceDirections` points (force = tendon force x ground direction, rotated to Z-up with the FTO-15 rotation; torque None); labels `muscle:<name>:origin|insertion`. `tests/unit/engines/opensim/test_opensim_muscle_wrenches.py` (10 tests: hanging-block equilibrium, equal and opposite ends, passive force at zero activation, no muscles, disabled, via point, postcondition). Docs: OPENSIM_INTEGRATION.md; cross-reference in `get_muscle_forces` docstring.
- Review fixes (Codex P1): the path is read with `getPath()` and a `GeometryPath.safeDownCast` (non-point paths are omitted, not a `bad_cast`); every `PointForceDirection` is released after use (the array only stores pointers). Tests: FunctionBasedPath muscle, RSS growth over 40k samples.
- Decisions: a leading/trailing path point on the same body as its neighbour has zero direction in OpenSim, so the effective ends are the first/last points with a nonzero direction (never a zero wrench). Negative or non-finite tendon force raises AssertionError (issue postcondition). Only end attachments are drawn; the full polyline is a follow-up.
- Validation: `python3 -m pytest tests/unit/engines/opensim/test_opensim_muscle_wrenches.py tests/unit/engines/opensim/test_opensim_force_torque.py` passes on Linux with opensim 4.6; ruff clean.
- Next steps: FTO-17 playback; open follow-up issue for the full muscle path polyline.

# Pinocchio Force/Torque Provider — #11298 (FTO-13)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11298`
- Branch: `feat/issue-11298-fto13-pinocchio-provider`; commit: SELF; PR: see the PR for this branch (`Closes #11298`, `Refs #11285`)
- Governing issue: #11298 (parent epic #11285, ADR-0052 section 4)
- Completed:
  - `src/engines/physics_engines/pinocchio/python/pinocchio_force_torque.py`: `PinocchioForceTorqueSource` with its own `pin.Data`; RNEA with the actual acceleration, world-frame `JOINT_REACTION` at the joint origin, `JOINT_ACTUATOR` for RX/RY/RZ/RUB/RevoluteUnaligned/Spherical joints (free-flyer omitted), `CONTACT` pass-through of `ContactSample`, axial loads via `axial_force_from_proximal_reaction`.
  - `pinocchio_physics_engine.py`: `get_force_torque_frame`, `get_segment_axial_loads`, `get_applied_torques`, `set_contact_samples` (`compute_contact_forces` sums them; zeros when none), `force_visualization=FULL`.
  - `CROSS_ENGINE_PARITY_SPEC.md` section 2.5.0: Pinocchio force-overlay channel row.
- Key decisions: FTO-2 (#11287, `conversions.py`) and FTO-11 (`segment_axes.py`) are not on main, so the world-frame conversion, torque wrench and segment axes are minimal private helpers in the new module. Axis rule: one child joint gives that joint origin, a leaf gives the body COM, a branching body or a zero-length axis is reported unavailable (None), never guessed. Wrench `body` is the BODY frame attached to the joint; the joint name appears only in labels (`reaction:<joint>`, `actuator:<joint>`, `contact:<body>`). Contacts are passed as a mapping of body (frame) name to `ContactSample`; unknown bodies are omitted. The engine recomputes acceleration with ABA at the sampled (q, v, tau) because `self.a` goes stale; ABA excludes external contact forces. Replace the private helpers when FTO-2/FTO-11 land.
- Validation: `ruff check`/`ruff format --check` clean on changed files; `pytest tests/unit/engines/pinocchio/test_pinocchio_force_torque.py tests/engines/physics_engines/test_pinocchio_engine.py`: all pass.
- Next steps: FTO-14 (Pinocchio GUI) consumes the provider; FTO-21 parity; swap private helpers for FTO-2/FTO-11 modules.

# Drake Force/Torque Provider — #11296 (FTO-11)

- Repository: `D-sorganization/UpstreamDrift`; branch `feat/issue-11296-fto11-drake-provider`; commit: SELF; PR: see the FTO-11 PR.
- Governing issue: #11296 (epic #11285, ADR-0052; development log `DL-#11285`).
- Completed: `drake_force_torque.py` (`DrakeForceTorqueSource`: joint reaction, net actuation, point and hydroelastic contact, opt-in gravity, axial loads); `force_overlay/segment_axes.py`; engine wiring (`get_force_torque_frame`, `get_segment_axial_loads`, `force_visualization=FULL`, hydroelastic-aware `compute_contact_forces`).
- Key decisions: Drake's reaction port is expressed in the child joint frame, so it is rotated to world; unavailable actuation is listed in `source.unavailable_labels` because `ForceTorqueFrame` has no legend; FTO-2 converters (`joint_torque_wrench`, `SegmentAxis`, `axial_loads_from_reactions`) are reused; `compute_contact_forces` now returns the force on non-world bodies (+m\*g at rest).
- Validation: `python3 -m pytest tests/engines/drake/test_drake_force_torque.py tests/unit/force_overlay` (40 passed with the capability test); ruff check/format clean on changed files.
- Known limits: discrete-time plants read zero reactions before the first step; point contact on a box face yields one unstable point, so the point test uses a sphere.
- Next steps: FTO-12 Drake GUI; FTO-21 parity.

# Simscape Output Force Channels — #11304 (FTO-19)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11304`
- Branch: `feat/issue-11304-fto19-simscape-output-force-channels`; commit: SELF; PR: see the PR for this branch (`Refs #11304`, `Refs #11285`)
- Governing issue: #11304 (epic #11285, ADR-0052); DL entry `DL-#11285`
- Completed (Python side, no MATLAB): `SimscapeOutput.force_columns` (optional, validated) and `to_force_series()`; `logsout_to_simscape_output` reads an optional `forces` key; the CSV loader was split so `force_series_from_columns` is the single core (DRY); `tests/engines/simscape/test_output_force_columns.py`; parity spec section 3.1 note. Review fixes: live output wrenches carry source `simscape_output` (CSV keeps `simscape_csv`, via `wrench_source`); non-mapping `forces` raises `SimscapeSimulationError`.
- Not done (needs a Windows R2025b host, not faked): channel audit of `GolfSwing3D_Kinetic.slx` / GS3DX logsout, `extract_sim_out.m` emitting `forces`, one-candidate evidence run, trimmed fixture from real output. Issue #11304 stays open for these.
- Validation: `python3 -m pytest tests/engines/simscape tests/unit/engines/simscape/test_force_channels.py -n auto --timeout=60` passes; ruff, mypy, file-size, error-handling clean.
- Next steps: on the R2025b host run the audit, extend `extract_sim_out.m`, record release string, model SHA and channel count.

# OpenSim Force/Torque Provider — #11300 (FTO-15)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11300`
- Branch: `feat/issue-11300-fto15-opensim-provider`; commit: SELF; PR: see the PR for this branch (`Closes #11300`, `Refs #11285`)
- Governing issue: #11300 (epic #11285, ADR-0052); DL entry `DL-#11285`
- Completed: `src/engines/physics_engines/opensim/python/opensim_force_torque.py` (`OpenSimForceTorqueSource`, `R_ZUP_FROM_OPENSIM_GROUND`); engine `get_force_torque_frame`, `get_segment_axial_loads`, `compute_contact_forces`; `contact_forces` and `force_visualization` PARTIAL; `tests/unit/engines/opensim/test_opensim_force_torque.py` (25 tests); capability test updated; "Force Overlay Channels" section in `docs/development/OPENSIM_INTEGRATION.md`.
- Decisions: FTO-2 (#11287) conversions landed on main while this PR was open but FTO-11 segment axes did not, so the world conversion, torque shift (reuses `motion_matching.force_torque.transform_wrench`) and segment axes are private helpers; swapping to `force_overlay.conversions` is a follow-up. Review fixes: the up axis is read from model gravity (Z-up models are not rotated), `step()` keeps the manager state, actuator labels use the actuator name, capabilities are PARTIAL. `sample` realizes the state to Acceleration itself because the Python bindings cannot read the stage. The existing `coord_map._R_YUP_TO_ZUP` is an axis swap with det -1 (a reflection), so a new proper rotation is defined instead; fixing the old constant is a separate follow-up. Record torque of HuntCrossley/Smooth forces is about the body origin (verified by the offset-mass-centre test). Wrench labels: `reaction:<joint>`, `actuator:<joint>.<coordinate>`, `contact:<force>`; body is the base body name.
- Known: `OpenSimPhysicsEngine.set_state` calls `opensim.Vector(n)` with one argument, which the 4.6 bindings reject (pre-existing, untouched). Programmatic `CustomJoint` construction segfaults the 4.6 bindings, so that test loads an XML model.
- Validation: `python3 -m pytest tests/unit/engines/opensim tests/unit/engines/test_mujoco_opensim_capabilities_7050.py` passes (104) on a Linux host with opensim 4.6; ruff check/format clean on changed files.
- Next steps: FTO-16 muscles; FTO-17 playback; FTO-21 parity; replace private helpers with FTO-2/FTO-11 modules.

# Force Conversions — #11287 (FTO-2)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11287`
- Branch: `feat/fto-11287-shared-conversions`; commit: SELF; PR: #11287
- Governing issue: #11287 (parent epic #11285, design authority ADR-0052 §1 and `force_torque_overlay_epic.md`)
- Objective: [FTO-2] Shared force conversions: joint torque to moment vector, local to world, reactions to tension/compression.
- Completed:
  - `src/shared/python/force_overlay/conversions.py`:
    - `joint_torque_wrench`: 1D revolute or multi-axis gimbal moments with unit-norm axis validation within 1e-6 (never silently normalized).
    - `world_wrench_from_local`: SO(3) orthonormal validation within 1e-9; routes all rotations (both full wrenches and single-half wrenches) through `transform_wrench` (DRY).
    - `move_wrench_point`: moment-arm adjustment `tau_B = tau_A + (p_A - p_B) x F`; validates `force_n is not None` when moving to a new point (raises ValueError if force_n is None since torque_nm is unknown and an OverlayWrench cannot have both halves None).
    - `SegmentAxis`: frozen dataclass defining segment endpoints, rejecting coincident proximal/distal points.
    - `axial_loads_from_reactions`: maps `JOINT_REACTION` wrenches to `axial_force_from_proximal_reaction` yielding `AxialLoadFrame` (tension positive, compression negative, missing reactions None).
    - `frame_with_axial_loads`: returns a copy of `ForceTorqueFrame` with `axial_loads` populated.
  - Re-exported functions from `src/shared/python/force_overlay/__init__.py`.
  - Added user guide paragraph in `docs/user_guide/body_part_viz/force_colors.md` ("Producing loads from reaction wrenches").
  - 13 unit tests in `tests/unit/force_overlay/test_conversions.py` covering all contract branches, red-first TDD, and synthetic two-link chain agreement.
- Next steps: Wave B child issues: FTO-3 (#11288) glyph builder and FTO-24 (#11309) video camera projection.

# Simscape Force Loader — #11303 (FTO-18)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11303`
- Branch: `feat/issue-11303-fto18-simscape-loader`; commit: SELF; PR: see PR for #11303
- Governing issue: #11303 (epic #11285, ADR-0052); DL entry `DL-#11285`
- Completed: `src/engines/simscape/force_channels.py` (channel table, `load_simscape_force_series`), `SimscapeAdapter.load_force_series`, `tests/unit/engines/simscape/test_force_channels.py`, parity spec section 3.1.
- Decisions: FTO-2 (#11287) is not on main, so rotation (`R @ v`) and CSV column reading are private helpers here; swap to `world_wrench_from_local` when FTO-2 lands. Axial loads (step 4) deferred to FTO-2's `axial_loads_from_reactions`.
- Review fixes: adapter forwards `rotation_tol`; actuator axes are per joint per `calculateJointPowerWork.m::getActuatorTorques` (Torso has no axis, omitted; LF/RF Z column is not in the committed trial so it is reported missing).
- Findings: the committed trial logs R only orthonormal to ~6e-3, so the 1e-6 default rejects it; real-data tests pass `rotation_tol=1e-2` explicitly. The only local/global pairs (MP couple/hand) satisfy world = R^T @ local, so the joint `R @ v` convention is unconfirmed on real data. Needs owner/MATLAB review.
- Validation: `python3 -m pytest tests/unit/engines/simscape/test_force_channels.py tests/unit/force_overlay -n auto --timeout=60` passes; ruff check/format, file-size, error-handling ratchet clean.
- Next steps: FTO-2 integration; FTO-19/20/21; resolve the R tolerance and convention questions.

# Force and Torque Overlay Contract — #11286 (FTO-1)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11286`
- Branch: `feat/fto-11286-force-torque-overlay-contract`; commit: SELF; PR: #11286
- Governing issue: #11286 (parent epic #11285, design authority ADR-0052 §1 and `force_torque_overlay_epic.md`)
- Objective: [FTO-1] Force/torque overlay contract: ForceTorqueFrame, wire schema and shared fixtures.
- Completed:
  - `src/shared/python/force_overlay/__init__.py`: explicit `__all__`, headless import guard.
  - `src/shared/python/force_overlay/contracts.py`: `WrenchKind` (7 categorical values), `OverlayWrench` (frozen dataclass with optional halves, DbC validation, `to_spatial_wrench`, `to_dict`/`from_dict`), `ForceTorqueFrame` (frozen dataclass, `by_kind`, `axial_loads` temporal alignment within 1e-12, `to_dict`/`from_dict`), `ForceTorqueProvider` protocol, `read_force_torque_frame`.
  - `src/shared/python/force_overlay/series.py`: `ForceTorqueSeries` (strictly increasing times, single engine, `frame_at` with linear interpolation and gap-bounding, pickle-free NPZ serialization with boolean masks, dict round-trip).
  - Promoted `validate_vec3` to public in `src/shared/python/motion_matching/force_torque.py` with `_validate_vec3` backward-compatible alias.
  - Wire schema `schemas/force-torque-frame-v1.json` (JSON Schema Draft 2020-12) and 7 conformance cases in `schemas/force-torque-frame-examples.json`.
  - 25 unit tests in `tests/unit/force_overlay/` (contracts, series, schema fixtures, headless import purity).
- Validation:
  - `python3 -m ruff check src/shared/python/force_overlay/ tests/unit/force_overlay/ src/shared/python/motion_matching/force_torque.py`: 0 violations.
  - `python3 -m ruff format --check src/shared/python/force_overlay/ tests/unit/force_overlay/ src/shared/python/motion_matching/force_torque.py`: 0 diffs.
  - `python3 -m pytest tests/unit/force_overlay -n auto --timeout=60`: 25 passed.
  - `python3 scripts/ci/check_file_size_budget.py`: OK.
  - `python3 scripts/ci/check_error_handling_ratchet.py`: OK.
- Next steps: Wave B child issues: FTO-2 (#11287) shared conversions and FTO-3 (#11288) glyph builder.

# Capture-O Video Companion Planning — #11268

- Repository: `D-sorganization/UpstreamDrift`; branch `claude/elegant-tesla-f2heae`; commit SELF; PR: see the planning PR for this branch.
- Objective: plan (not execute) markerless reconstruction of the owner's capture-session video and its comparison with `capture-O` and the matched models.
- Completed: epic #11268 and children #11269–#11279 (COV-1 to COV-11) with TDD/DbC/LoD/DRY contracts, hosts and dependencies; `docs/development/capture-o-video/procedure.md`; development-log entry `DL-#11268`.
- Key decisions: neutral ids only (`capture-O`, `cov-NN`, `subject-O`); media stays in `$CAPTURE_DATA_DIR/capture-O-video/`; the album link is public by owner decision (https://photos.app.goo.gl/XU322J42Rg8mev2aA); comparison protocol COV-3 is `tier:strong` and must be frozen before results are inspected.
- Validation: document title check, Ruff and the development-log validator on the changed files (see the PR body).
- Blockers: album download needs a fleet machine (cloud proxy returns 403); PR #11172 (registry, export, comparison) unmerged; COV-10 waits on #11165 and an R2025b host.
- Next steps: 1) fleet agent downloads the album and runs COV-1 #11269; 2) COV-2 #11270; 3) frontier/owner decision COV-3 #11271.

# Simscape Matching Review and Continuation Handoff

## Earlier Verified Full Human Exports

Both full sampled swings now use GS3DX_Human ellipsoids with calibrated foot orientation and freshly calibrated marker offsets. Native solves exited naturally with code zero at 2026-10-02T05:15:54Z (A) and 05:23:22Z (O). Position RMS excludes orientation residuals.

| Capture  | Fit mean / max frame RMS (mm) | Human mean / max frame RMS (mm) | Measured foot samples |
| -------- | ----------------------------- | ------------------------------- | --------------------- |
| A, tour  | 12.746 / 40.209               | 13.188 / 40.121                 | 55 per foot           |
| O, owner | 26.558 / 116.576              | 16.767 / 37.626                 | 45 of 46 per foot     |

The owner fit improves both position metrics. The tour fit has a small mean position tradeoff while resolving the reversed-foot ambiguity. Maximum left/right foot angular errors are 12.83/19.82 degrees for A and 19.55/20.38 degrees for O; these diagnostics do not establish anatomical or contact qualification.

Four 800 by 600 H.264 MP4s were completely decoded at 30 fps, with 55 tour and 46 owner frames. Native caption-corrected rerenders exited naturally at 05:28:11Z and 05:31:30Z. The Desktop shareable ZIP contains both views, sanitized provenance, hashes and a same-capture/same-frame comparison; no raw captures or pose caches. Original Fit clips remain preserved.

Owner wrist offsets remain approximately 55/62 mm and the club-target offset 106 mm. Native parameter and grip-contract tests pass; the functional-grip fit worsened held-out position errors and was rejected. These are IK clips; forward dynamics remains unqualified.

## Current Main Model Comparison and Selected Delivery

The current model (`91997471`) full tour export exited naturally at 07:49:26Z, but mean/max RMS 13.366/41.385 mm is worse than the selected `9a26ee80` tour result. The current owner export exited naturally at 08:39:18Z and improves mean/max RMS to 16.643/36.865 mm. Both current-model clips per capture fully decoded. Owner foot orientation remains measured in 45/46 samples: left mean/max 4.517/19.556 degrees and right 9.421/20.299 degrees. Small mixed orientation changes remain explicit; left max and right mean are slightly worse than prior. This selection prioritizes position tracking with calibrated feet, not anatomical or physical acceptance.

Latest selected Desktop delivery: `Simscape_Matches_20261001/Best_Human_Matches_20261002.zip`, SHA256 `a97db61e9fb43dbae4961b24727526554ce68abe081f60c79146b7b29b302cb3`. It includes the earlier model's tour clips and current model's owner clips, separately identified hashes/runtime/provenance, all four fully decoded MP4s, no raw captures or private pose caches. Prior packages are retained.

Selected and current runtime versions differ (NumPy 2.4.4/ezc3d 1.7.2 versus 2.2.6/1.6.3). Read-only canonical SI XYZ/validity/ordered-label digests match exactly for both captures under those runtimes. This narrows the importer concern; it does not prove equivalent numerical solver behavior or identify why the model fit changed. The controlled earlier-model run on current source/runtime produced joint arrays identical to the current-model tour run (maximum difference zero); both give 13.366/41.385 mm. The first wrapper exited one after solving because it read an absent return-report field; the corrected JSON-provenance check reused the verified pose cache and exited naturally at 08:59:42Z. Recorded solve-source hashes match the earlier selected execution, whose poses differ, but full transitive/runtime equivalence is not established. The model binary alone does not explain this difference under common tested conditions. Continue cross-execution reproducibility and physical initialization review. No saved-model geometry is overwritten to force prior fit.

## Native Mapping and Head-Cluster Review

The selected shareable clips remain tour 13.188/40.121 mm and owner 16.643/36.865 mm mean/max frame position RMS. Both use Human ellipsoids and IK; full forward dynamics remains unqualified. The controlled common-source/runtime tour comparison produced identical poses with the old and current model, so the newer model binary alone does not explain the changed fit.

Two new pure helpers passed 18 native initialization-mapping tests and 8 marker-cluster tests, with zero failed/incomplete tests. Parent review added actual failing regressions for matrix-shaped poses and a scale-dependent degeneracy cutoff before fixes passed. These counts are separate from the earlier 187-test integration suite. The first native state-target-expression check failed (original loop status 1, mapped -1, scalar discrepancy 16.320 degrees). A compiled diagram update refreshed stale masked start values; a fresh KinematicsSolver then reproduced all 22 joint frames for both captures, with maximum translation error 4.44e-16 m and rotation-matrix error below 9.49e-15. The run exited naturally with code zero at 10:03:13Z. No simulation ran and no model was saved; state-target priorities, controller references and physical initialization remain unqualified.

HeadTop/HeadFront/HeadSide tracks are finite and nondegenerate in all 654 tour and 367 owner frames. Pair-distance variation reaches 6.761% for the owner; cluster axes require body-frame calibration and do not establish anatomical orientation. A conservative optional head-motion candidate is under development and has not replaced selected clips. See `native_helper_review_20261002.json` and `head_track_audit_20261002.json` in the research reference directory. The latest LaTeX source remains uncompiled: the built-in compiler reports `Unable to find standard directories for platform`.

## Identity and Branch Status

- **Repository**: D-sorganization/UpstreamDrift
- **Active Branch**: `feat/simscape-matching-review-main-20261002`; commit `SELF` contains this continuation handoff.
- **Reviewed Baseline Source**: `144e81188dd7bb106f81d89b7a7330dd20cce511` on `feat/simscape-gs3dx-exploratory` (unpublished baseline commit).
- **Original Worker Commit Turnover**: Commits `144e81188` through `f2ca443a0` (documentation-only continuation) reviewed and preserved.
- **Draft PR and Governing Issues**: Original draft PR #11179; reviewed continuation draft PR [#11256](https://github.com/D-sorganization/UpstreamDrift/pull/11256); governing issues #10950, #10979, #11156, #11160, #11161, #11173.
- **Coordination**: Session `simscape-20261001-codex`, governing #11173; check live lease/presence before expansion or handoff. Mailbox evidence is incomplete; do not infer absence of peers.
- **Development Log**: Existing `DL-#10950`, updated in place.
- **Historical Context**: Prior turnovers remain recoverable via `git show 144e81188:docs/development/HANDOFF.md`.
- **Documentation Authority**:
  - The editable standalone research reference is [simscape_matching_reference.tex](../research/simscape_matching_reference/simscape_matching_reference.tex) (earlier 16-page revision compiled and visually reviewed; latest full-export additions remain uncompiled because the built-in compiler reports a platform directory error).
  - The canonical engineering design manual remains governed under `manuals/upstreamdrift` QMD.

## Current State and Operational Scope

Review work proceeds within this isolated worktree while preserving the original agent worktree and unpublished branch states.
The parent owns LaTeX, export pipelines, source code, development log (DL), AGENTS, and SPEC definitions; delegates handle bounded drafting and runner hardening.

- **Current Goal**: Verified full Human ellipsoid exports for Capture A and Capture O, continued tour and owner physical matching, and formal review through protected branch delivery. The goal remains actively in progress and **unfinished**.
- **Kinematic vs Dynamics Distinction**: All current video exports are kinematic inverse kinematics (IK) visualizations, not forward-dynamics-driven swings. No forward dynamics qualification is claimed.
- **Impact Acceptance**: Club acceptance strictly requires at most 2.0 degrees of face error at explicit `phases.contact`. Missing contact fails qualification. Peak head speed is not ball contact.
- **Receipt Integrity**: The serialized MATLAB runner returns nonzero for failed script status, timeouts, or forced termination. Read `<log>.receipt.json`: historical process exit 0 without a hardened receipt does not establish native success.

## Verified Baseline Desktop IK Exports

Desktop delivery folder: `Simscape_Matches_20261001` on ControlTower. Four Desktop Fit IK H.264 30 fps clips represent the verified baseline:

- **Capture A Baseline (Tour Reference, 360 Hz, 654 frames)**:
  - Sample coverage: 55 frames.
  - Mean measured-target RMS: 12.745748 mm.
  - Maximum measured-target RMS: 40.208544 mm.
  - Diagnostic worst sampled target: `trailElbow` at 96.517 mm (frame 517).
  - Process receipt: Natural exit 0.
- **Capture O Baseline (Owner optical, 240 Hz, 367 frames)**:
  - Sample coverage: 46 frames.
  - Mean measured-target RMS: 26.558095 mm.
  - Maximum measured-target RMS: 116.576104 mm.
  - Diagnostic worst sampled target: `clubhead` at 375.358 mm (frame 89).
  - Process receipt: Natural exit 0.
  - Skill assessment rule: Golfer ability cannot and must not be inferred from avatar distortions.
- **Baseline Policy**: These four clips are preserved historical baselines using the legacy GS3DX_Fit cylinder model; they are not refined ellipsoid clips yet.

## Candidate Evaluations and Backward-Pass Rejection

- **Owner Backward Pass**:
  - Tested on identical model identity and marker offset sets.
  - Mean measured-target RMS: 32.185137 mm.
  - Maximum measured-target RMS: 46.052114 mm.
  - Process receipt: Natural exit 0 (receipt timestamp: `03:57:32.790355Z`).
  - Outcome: **REJECTED**. The backward pass improved peak residual (46.05 mm vs 116.58 mm) but substantially worsened mean tracking (32.19 mm vs 26.56 mm).
- **Marker Offset Distributions**:
  - Owner fitted marker offset norms cluster in the 70–100 mm range, whereas tour reference offsets are substantially smaller.
  - Hypothesis: Offset magnitudes indicate capture definition differences and geometric compensation, not provenance or swing technique evidence.

## Kinematic Topology and Joint Role Resolution

- **Independent Fit Coordinates (37 vs 39 vs 33)**:
  - The Human model kinematic topology possesses **37 independent coordinates** during fitting with a closed grip loop.
  - The earlier hardcoded 33-coordinate failure incorrectly reported a 39-coordinate count due to mismatched coordinate indexing in the legacy solver. Clarifying that actual joint roles yield 37 independent degrees of freedom resolves the structural index bug.
- **Native 3-Frame Fit Probe**:
  - Frame sample: `[1, 13, 25]`.
  - Position RMS: 13.296 mm, 12.536 mm, 12.374 mm (finite; numerical conditioning was not assessed).
  - Left foot orientation residual: 165.0 deg, 167.0 deg, 167.2 deg against calibrated shoe orientation $R$.
  - Process receipt: Natural exit 0 (receipt timestamp: `04:54:21.730288Z`).
  - Core finding: Minimizing positional marker residuals alone does not identify or constrain foot orientation.

## Foot Orientation Formulation and Calibration Trials

- **Calibrated Foot Orientation Methodology**:
  - Evaluated using shared calibrated foot triads and SVD address mean orientation:
    $$R = F_f F_0^T R_z(\text{yaw})$$
  - The flat sole condition at address (frame 1) is a geometric modeling assumption, **not** dynamic ground reaction force evidence. If frame 1 is missing, the routine fails closed.
- **Foot Orientation Weighting Trial**:
  - Weight 0.1 applied to foot orientation residual, reusing existing Fit offsets.
  - Position RMS degraded: 51.263 mm, 43.243 mm, 39.399 mm.
  - Left foot orientation error improved: 36.505 deg, 17.058 deg, 13.764 deg.
  - Process receipt: Natural exit 0 (receipt timestamp: `04:57:13.758186Z`).
  - Outcome: **REJECTED**. Position degradation was too severe for production substitution.
- **Active Probe Status**:
  - Fresh Human offset calibration at frames [1, 13, 25] completed with natural exit 0 (2026-10-02T05:01:00.641030Z). Position RMS is 1.188/1.195/1.208 mm and all foot angular errors are below 0.7 deg. These same frames were used for calibration; this is not held-out or full-swing validation.
  - Full $\text{SO}(3)$ 18-component chordal formulation with weight 0.1 is exploratory and not certified (excludes position RMS; gap foot metric returns `NaN`).

## Model Adapters and Visual Scaling Limits

- **Human Ellipsoid Adapter**:
  - 10 native MATLAB adapter tests pass with natural exit 0.
  - Scales capture longitudinal segment geometry to match subject proportions.
  - Artistic widths, head geometry, and shoe dimensions remain fixed.
  - Baseline segment mass and inertia properties are **NOT** scaled to the owner's 104.3 kg body mass.
  - Forward dynamics remain unqualified.
- **Future Ellipsoid Export Policy**:
  - Future shareable matches must use the stylish Human ellipsoid model.
  - Calibrated foot roll/pitch/yaw must be constrained and their residuals reported.
  - Public capture aliases and hashes provide provenance; private raw captures and pose caches remain uncommitted.
  - Independently record 14 target RMS errors and orientation residuals.
  - Head and neck orientation is not yet constrained by this solver and require future orientation constraints; do not describe the model as anatomically complete.

## Physical Stance Tests and Forward Dynamics Status

- **1-Second Stance Simulation Failures**:
  - 1-second resampled, quiet, frozen address, and still upper-body tests all fail physical stance stability.
  - Ground reaction normal force decays to zero after 0.5 s.
  - At 1.0 s, the reported frame-tilt metric reaches 101 deg, 108 deg, 114 deg, and 124 deg.
  - The pelvis drops far below the floor plane.
  - Historical process `EXIT 0` was an unhardened execution receipt, not physical stance qualification.
- **Queued Damping Experiment**:
  - The unsaved damping-times-ten test completed and failed: initial right-heel force 15,513 N, force zero by 0.60 s, tilt 117 deg and pelvis z -1.83 m at 1.0 s. It must not be repeated alone. The original worker subsequently queued a composed pelvis-level pose check stopping at 0.02 s; no outcome is claimed.
  - Do not duplicate, cancel, save, or claim success for the original worker's pending composed-pose check. The original worker state must be preserved; verify its live process and receipt before attributing any new result.

## Open Scientific Gates

- **Gate #11156 (Lead Elbow and Wrist Tracking)**: **RED**. 20-degree elbow limit and 15-mm tracking penalty remain active.
- **Gate #11160 (Clubhead Kinematics and Contact-Face Error)**: **RED**. Head RMS/max, shaft maximum, speed, and contact-face limits remain frozen.
- **Gate #11173 (Open-Loop Full-Swing Forward Dynamics)**: **RED**. Contact spikes, intervals with zero measured normal force, and competing balance torques prevent qualification. Zero force alone does not establish that the entire body is airborne.
- **Owner matching** remains unqualified in Simscape. Capture registry delivery is a separate workstream; external MuJoCo ZMP diagnostics do not establish Simscape feasibility.

## Software Test Inventory and Verification Boundaries

- **Pure Mathematics and Architecture Tests**: 102 tests PASS, 0 skipped, following initial full orientation helper implementation.
- **Strict Validator Tests**: The five additional validator tests now pass within the 102-test suite, including missing address calibration and ambiguous orientation rejection.
- **LaTeX Reference Compilation**: Earlier 16-page revision compiled and visually reviewed. Latest full-export source additions remain uncompiled: built-in compiler reports Unable to find standard directories for platform.
- **Overall Goal Status**: Full refined A/O exports, continued tour and owner physical matching, and protected delivery remain unfinished.

## Reproduction and Operational Constraints

- Set `GS3DX_CAPTURE_ID` to `capture-A` or `capture-O`.
- Explicitly define `GS3DX_OUTPUT_DIR`, `CAPTURE_REGISTRY_REPO`, and `MATLAB_PYTHON_EXE`.
- Execute via `tools/run_matlab_locked.ps1` using MATLAB R2025b (`C:/Program Files/MATLAB/R2025b/bin/matlab.exe`).
- Do not kill live processes or duplicate queued stance runs.
- Public records must use only neutral aliases (`capture-A`, `capture-O`) and cryptographic hashes.

## Next Actions

1. Verify importer and export-provenance contracts, then evaluate further wrist/club/head refinements against held-out samples.
2. Review position and orientation diagnostics and decoded ellipsoid videos before selecting replacements.
3. Preserve original worker tasks; review the pending composed pelvis-level check without duplicating it.
4. Preserve the verified Human clips and enforce ellipsoid rendering for future accepted improvements; report position and orientation separately.
5. Review the isolated branch through protected delivery without premature qualification claims.

## Subject Physics and Rejected Functional-Grip Candidate

The native subject-physics adapter/catalog passed 18 tests with zero incomplete tests (natural exit 0, 2026-10-02T06:03:51Z). It applies a declared mass and whitelisted fitted lengths in memory, preserves joint expressions and parameter metadata, and audits equipment through the existing native inertia catalog. Parameter consistency is not a dynamics qualification. The selected Human videos still use the baseline physical parameters because mass does not drive IK.

The sphere solver passed 15 tests and the controlled grip contract passed 12 tests, both with zero incomplete tests. A functional-grip owner candidate completed naturally with code zero at 06:15:53Z. Its 24 held-out samples worsened mean/max frame RMS from 26.011/37.626 to 27.969/41.278 mm; full-swing values worsened from 16.767/37.626 to 17.948/41.278 mm. It was rejected. Its smaller lead-wrist offset alone is insufficient for selection. Capture lengths were estimated from the complete recording; held-out status concerns grip/offset calibration only. See `grip_candidate_comparison_20261001.json`.

## Fresh Contact Geometry and Capture Metadata

A read-only native forward-kinematics audit of the selected owner's first pose, with fitted Human geometry and declared 104.3 kg mass, exited naturally with code zero at 06:42:14Z. Against the existing model ground plane, the ten contact-sphere clearances ranged from -64.399 to -48.619 mm (15.780 mm spread). This geometric penetration is not a simulated force measurement. Simulation initial targets were not updated, no dynamics was run and no model was saved. The pelvis frame's local up vector tilted 57.889 degrees; that frame-relative quantity does not establish an anatomical torso angle or a causal instability mechanism. Existing World Frame selection matches the cached IK. Ground placement and a consistent gravity/contact initialization must be verified separately.

Both actual C3D files specify metres and contain no EVENT parameter group. Under ezc3d 1.7.2, A has 717 negative-residual samples, all already nonfinite in XYZ; O has no negative residuals but 1,173 nonfinite XYZ samples. These counts cover all marker tracks. The fourth point row is homogeneous XYZ1, not residual metadata. Neither peak-speed timing nor address-line crossing establishes measured ball contact. Importer units/missingness contracts and recorded dependency/runtime provenance passed focused tests and fresh full exports; current-model comparisons are recorded above.

## Importer and Runtime Contract Verification

Thirteen pure importer tests, seventeen pure runtime-fingerprint tests, and twelve exporter precondition tests pass with zero incomplete tests. The live runtime is MATLAB R2025b Update 5, Simulink/Simscape/Multibody 25.2, Python 3.12.10, NumPy 2.4.4 and ezc3d 1.7.2. Required recorded source hashes fail closed; optional resolver absence is explicit. Direct runtime components are recorded, not every transitive package or external STL asset. Fresh A/O exports exited naturally and all four clips fully decoded; current-model owner selection and prior-model tour selection are recorded above.

## Current-Main Integration Boundary

This continuation is based on `cee0a65e0`. Current-main model promotion, compiled home-budget diagnostics and anatomical mesh work (#11207, #11208, #11209) are preserved. The current-main Human model SHA256 begins `91997471`; existing verified Desktop matches use the separate `9a26ee80` source. Their successful receipts do not qualify the new integration. The integrated source passed 187 native MATLAB tests (zero incomplete) and 44 focused Python tests. Both preserved-model full matches exited naturally; the selected owner improves position tracking and the prior-model tour remains selected, as recorded above. Forward dynamics remains unqualified.

## Historical Player Capture Continuation Preserved From Main

The following separate continuation state was present on current main and is retained without treating its validation as evidence for Simscape matching.

# Active Necromatcher Native Fit Delivery — #11240

Current branch `feat/necromatcher-native-fit-11235` (PR #11240) retargeted to `main` following merge of workspace #11239. It adds source-bound native trajectory fitting with preserved Hermite splines, native video export, research refit controls, ground placement, and effort bindings. All 88 fitting/spline/IK tests and 142 workspace unit tests pass locally.

- Issue: #11235; parent #11232
- PR: #11240 (retargeted to `main`)
- Branch: `feat/necromatcher-native-fit-11235`
- Validation: 88 fitting tests, 142 workspace tests pass; Ruff lint/format clean.

# Active Necromatcher Workspace Delivery — #11239 (Merged)

native adapter and shared source-frame archive reader. Sixteen UI tests, desktop
recall and worker failure tests pass. Web type checking and scoped ESLint pass.
Native import/overlay and actual Hogan/Tiger web review are implemented. URL
navigation now hides stale results, verifies player/swing/capture ownership and
supports retry after frame loading errors. Further form tests, final parity and
fitted-model handoffs remain in progress. CI cycle 2 refreshes canonical launcher
context/atlas views after their freshness checks failed. Library draft PR #11237
depends on capture PR #11231. The capture
PR has an unrelated inherited title-case failure at `.jules/bolt.md:208`; do not
mark it accepted or change unrelated work under this delivery.

Owner priority is the integrated historical-player workspace (#11232), with library #11233, tile/review #11234 and real fitting/downstream qualification #11235. See [Necromatcher Turnover](necromatcher-turnover.md) for contracts, TDD evidence, real capture imports and current PR state. Tiger #11226 and Hogan #11229 remain open.

# Historical Player Capture Handoff

## Active: Tiger 2000 and Ben Hogan

- Repository/worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-historical-capture`.
- Branch: `feat/historical-player-capture-11226`; implementation commit: ab2c869813a2b2be2f14b512ee34647694614377; handoff refresh: SELF.
- PR: https://github.com/D-sorganization/UpstreamDrift/pull/11231; open, CI pending. Epics: #11226 (Tiger), #11229 (Hogan); shared runner: #11230.
- Objective: complete both source-grounded historical reconstructions and make future players repeatable.
- Implemented: bounded PyAV streaming, existing SourceAsset/FrameIdentity contracts,
  existing MediaPipe estimator, normalized image XY/visibility/missingness, exact
  container PTS, lossless decoded PNGs, source/frame/observations/model/code hashes.
- Source authority: `docs/development/historical_capture/source-catalog.json`.
- Procedure: `docs/development/historical-capture-procedure.md`.
- Media/results: `C:/Users/diete/Downloads/historical-capture/` (outside Git).
- Tiger download: requested t_J6Vik3Tss, complete 1080p50 video, separate audio,
  metadata/description, and losslessly remuxed video with audio. Uploader claims
  2000; upload date is 2022-06-10 and does not verify the recording year.
- Final Hogan run: 110–135 presentation seconds, 750 frames, 739 detections,
  MediaPipe 1.0.1; missing detections preserved. Tiger final run: 2000 frames, 1994 detections (110-150 s).
- Earlier MediaPipe 0.10.32 pilot receipts are historical evidence, not current
  runner qualification. Final runs use content-based asset/frame IDs.
- TDD: missing module failed collection, then absent export function failed the
  streaming test; invalid local URI and decimal/Fraction boundary tests failed
  before correction. Last focused suite: 35 passed; latest capture suite: 12 passed.
- Checks: repository-wide Ruff lint passed; format check passed (8263 files);
  tracked file-size budget passed; procedure title check passed; design-manual
  governance verified existing `blocked-inventory-required` release state.
- Commands: `python3 -m pytest tests/unit/shadow_tracker/test_historical_capture.py tests/unit/shadow_tracker/test_footage_workflows.py tests/unit/shadow_tracker/test_fail_closed_fitting.py -q -o addopts=''`; `python3 -m ruff check .`; `python3 -m ruff format --check .`; `python3 scripts/ci/check_file_size_budget.py`.
- Whole Shadow Tracker suite passed: 364 tests in 36.28 s. Scoped mypy passed. Existing import deprecation warnings
  remain. No native-engine, scientific, or website acceptance is claimed.
- Ownership: this worktree started clean from fb1ac44949 on origin/main. Original
  checkout/other worktrees and their user-owned files were preserved. Partial
  failed output directories have no receipt and are not valid capture results.

## Ordered Continuation

1. Final Tiger/Hogan receipts collected; inspect dense source-bound landmark
   overlays and split each window at every cut/identity change. Contact-sheet
   inspection shows foreground body tracking, with errors/low-confidence joints.
2. Higher-resolution Hogan source processed: 899 frames, 892 detections; select clean continuous swings,
   and review P1–P10 checkpoint and impact intervals.
3. Monitor published shared-runner PR #11231 using the
   repository ci-watch-and-fix skill. Keep parent epics open.
4. Resolve recording/event lineage, playback scale and usage permissions; fit
   cameras and subject anthropometry with declared priors/uncertainty.
5. Fit dense constrained kinematics, compare native MuJoCo/Drake/Pinocchio FK,
   then independently qualify uninterrupted forward replay where supported.
6. Evaluate held-out film lineages and integrate eligible comparison artifacts
   into UpstreamDrift/AffineDrift. Do not claim completion until epic evidence exists.

# Historical Capture Continuation

Current authority: [Root Agent Handoff](../../AGENT_HANDOFF.md).
Shared runner #11230, Tiger #11226 and Hogan #11229: streaming extraction is implemented; reconstruction acceptance remains open.

# Current Handoff — Qualify OpenSim Native Dual-Club Dynamics and Replay (#11095)

- Branch: `feat/mmr-10o-opensim-dual-club-11095`
- Pull request: Refs #11095 (partial: fail-closed conversion; native qualification still requires opensim bindings on a pinned host/native CI lane).
- Done: OpenSim qualification schema plus **fail-closed conversion** after review audit (placeholder receipts, gates that always qualified, invented marker metrics):
  - `OpenSimQualificationReceipt` records `missing_evidence` and a resolvable `remedy`; status gated on every recorded check.
  - Unavailable `opensim` runtime ⇒ `UNAVAILABLE` even with a replay payload; unknown native test counts and absent `is_fresh_simulation`/`actuation_applied` flags are missing evidence (never assumed satisfied); missing rollout/marker data and non-finite values reject; derivative mismatch (`dq/dt` vs `v`) rejects.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command).
- Tests: 17 unit tests passed (`tests/unit/engines/opensim/test_opensim_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN); Ruff check and format clean.
- Limitation: no native OpenSim execution exists anywhere in this evidence; real qualification requires the opensim bindings on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine opensim`.

# Current Handoff — Qualify MyoSuite Native Dual-Club Dynamics and Replay (#11096)

- Branch: `feat/mmr-10m-myosuite-dual-club-11096`
- Pull request: Refs #11096 (partial: fail-closed conversion; native qualification still requires myosuite/MuJoCo on a pinned host/native CI lane).
- Done: MyoSuite qualification schema plus **fail-closed conversion** after review audit (placeholder receipts, gates that always qualified, invented marker metrics):
  - `MyoSuiteQualificationReceipt` records `missing_evidence` and a resolvable `remedy`; status gated on every recorded check.
  - Unavailable `myosuite`/MuJoCo runtime ⇒ `UNAVAILABLE` even with a replay payload; unknown native test counts and absent `is_fresh_simulation`/`actuation_applied` flags are missing evidence (never assumed satisfied); missing rollout/marker data, non-finite values, and unnormalized root quaternions reject; derivative mismatch (`dq/dt` vs `v`) rejects.
  - Removed SPEC-claimed but never-enforced tolerances and fabricated values (scaled early/terminal/clubhead RMS, hardcoded `pelvis_yaw_error_pct`); only `whole_rms_m` computed from recorded `markers_m`/`target_m` is emitted.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command); README model hashes demoted to regeneration targets.
- Tests: 19 unit tests passed (`tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN); Ruff check and format clean.
- Limitation: no native MyoSuite execution exists anywhere in this evidence; real qualification requires the myosuite/MuJoCo stack on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine myosuite`.

# Current Handoff — Qualify Drake Native Dual-Club Dynamics and Replay (#11094)

- Branch: `feat/mmr-10d-drake-dual-club-11094`
- Pull request: Refs #11094 (partial: fail-closed conversion; native qualification still requires pydrake on a pinned host/native CI lane).
- Done: [MMR-10D] Drake qualification contracts plus **fail-closed conversion** after review audit (placeholder receipts `c0ffee`/`deadbeef`/`cafebabe`, gates that always qualified, invented marker metrics):
  - `DrakeQualificationReceipt` now records `missing_evidence` and a resolvable `remedy`; `DrakeQualificationStatus` (`QUALIFIED`, `REJECTED`, `UNAVAILABLE`) is gated on every recorded check.
  - Unavailable `pydrake` runtime ⇒ `UNAVAILABLE` even when a replay payload is supplied; unknown native test counts are treated as missing evidence (never assumed nonzero); absent `is_fresh_simulation`/`actuation_applied` flags are unverified (fail-closed), not assumed fresh.
  - Missing `native_state`/`time_s` rollout or `markers_m`/`target_m` marker data is recorded as missing evidence and blocks qualification; derivative (`dq/dt` vs `v`) mismatch and non-finite state/energy values reject.
  - Removed synthesized marker metrics (scaled early/terminal/clubhead RMS, hardcoded `pelvis_yaw_error_pct`); only `whole_rms_m` computed from recorded observations is emitted.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command). Nightly lane receipt remains honest `status: fail` (0 executed tests, engine unavailable).
  - Added `"drake"` to `ENGINE_LANES` in `scripts/ci/run_native_engine_lane.py` and updated `scripts/ci/run_native_engine_lane.sh`.
- Tests: 36 focused tests passed (16 `tests/unit/engines/drake/test_drake_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN; 20 lane/freshness tests). Ruff check and format clean on changed files.
- Limitation: no native Drake execution exists anywhere in this evidence; real qualification requires pydrake on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine drake`.
- Next step: merge drivers follow; do not treat UNAVAILABLE receipts as engine qualification.

# Current Handoff — Consolidate Bolt Micro-Optimisation PRs (#11112, #11128, #11129)

# Current Handoff — Restore the High-Severity UI Npm Audit Gate (#11184)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `C:/Users/diete/Repositories/Worktrees/luna-upstream11184-20260930`
- Branch: `fix/main-npm-audit-11184`
- Commit: `SELF` (publication metadata update; implementation commit `d3b3a27bea36070c2db3a06e6e7be727a30e9667`)
- Pull request: [#11187](https://github.com/D-sorganization/UpstreamDrift/pull/11187), draft with `agent:codex` label; branch `fix/main-npm-audit-11184`.
- Governing issue: #11184 — restore the UI `npm audit --audit-level=high` gate using only compatible patched transitive resolutions.
- Done: added `ui/src/test/dependencySecurityContract.test.ts`; moved only `brace-expansion` 5.0.9 → 5.0.12 and `undici` 8.10.0 → 8.11.2 in `ui/package-lock.json`; added one SPEC row and an active development-log entry. No manifest, override, audit-policy, or moderate-advisory changes.
- RED evidence: baseline `npm ci` completed with 2 HIGH and 2 MODERATE advisories; baseline `npm audit --audit-level=high` exited 1. The regression contract failed on old lock versions 5.0.9 and 8.10.0.
- GREEN evidence: final `npm ci` passed; `npm audit --audit-level=high` passed with 2 MODERATE findings left (`@humanfs/node` 0.16.7 and nested `fflate` 0.6.10); `npm ls brace-expansion undici --all` showed only 5.0.12 and 8.11.2 on the affected paths. Contract: 2 passed; lint and type-check passed; all UI tests passed (99 files, 936 tests); build passed. Vitest emitted jsdom `scrollTo` notices; build emitted a large-chunk warning.
- Documentation checks: SPEC changelog validation and fleet hook passed. The repository development-log validator still exits 1 on pre-existing duplicate IDs, portfolio WIP/active-entry ceilings, and file-size ceiling; it reports no DL-#11184 finding.
- Compatibility evidence: registry metadata confirms the published patch releases and parent ranges `minimatch@10.2.5` → `^5.0.5`, `jsdom@30.0.1` → `^8.9.0`. The selected versions stay within those ranges.
- Coordination: fresh Repository_Management inbox was complete with no conflicts or new messages since 2026-09-29. Renewed the existing `codex-luna-upstream11184-20260930` presence, preserving its issue, branch and goal and adding `ui/src/test` and `docs/development`; presence expires at 13:04 UTC. Scoped REST lookup found no pre-existing PR for this branch. Authenticated `git ls-remote` confirmed `origin/main` remained exactly `aeb2edbca47c8b91a504fe199ed77c2e377fa6d5` before publication.
- Publication: root reviewed and accepted the bounded source, lockfile and test diff plus RED→GREEN evidence. Implementation commit `d3b3a27bea36070c2db3a06e6e7be727a30e9667` and metadata commit `5d2f3257ed2342bd2c6064aebed6a99a9564700e` passed normal pre-commit hooks; both branch pushes passed normal pre-push hooks. Draft PR #11187 is open with `Closes #11184` in its body and `agent:codex` label. SPEC uses actual PR key #11187. Remote refs were verified after publication: topic branch at `5d2f3257ed2342bd2c6064aebed6a99a9564700e`, main still at `aeb2edbca47c8b91a504fe199ed77c2e377fa6d5`. Two MODERATE audit findings remain (`@humanfs/node@0.16.7` and nested `fflate@0.6.10`). Root alone decides readiness and merge.
- Worktree state: clean after publication metadata; no merge, release, cleanup, or unrelated changes were performed. The primary checkout’s pre-existing untracked paths and other worktrees remain untouched.
- Next step: root decides whether draft PR #11187 is ready for review/merge. Do not mark ready, merge, release, or clean up as part of this handoff.

---

# Previous Handoff — Consolidate Bolt Micro-Optimisation PRs (#11112, #11128, #11129)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/w-ud-bolt-cons`
- Branch: `claude/ud-bolt-consolidated-0929` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11138; supersedes #11112, #11128 and #11129 (originals left open, closed citing the consolidator's squash SHA once it lands).
- Done: took only the `src/` numeric rewrites from three open Bolt PRs, dropping each PR's own SPEC row (this branch writes one row), and the `.jules/bolt.md` note carried by #11128.
  - #11112 (`bot/bolt-norm-einsum-residual`): row/column `np.linalg.norm(..., axis=N)` becomes `sqrt(np.einsum(...))` across `bunkershot3d/ball/qualification_fit.py` (column norm, `axis=0`) and 8 `motion_matching` sites (`club_calibration`, `club_only/acceptance`, `club_only/control_replay`, `dynamics_filter`, `execution/downswing`, `hip_calibration`, `multi_shooting_fit`, `pipeline/reference`). Carried unchanged, including the new identity test.
  - #11128 (`bolt-opt-pink-tasks-vdot-...`): small fixed-size vector norms in `engines/physics_engines/pinocchio/python/pink_tasks.py` (weld rotation/translation residuals, marker error) become `math.sqrt(np.vdot(x, x))`. Carried the numeric rewrite but dropped the inline `# ⚡ Bolt: ...` rationale comments to match this repo's default no-comments-unless-non-obvious-WHY convention (no prior Bolt PR merged into this repo left such comments in source).
  - #11129 (`bolt-optimize-physics-validation-norm-...`): `physics_validation.py` Jacobian error norm becomes `math.sqrt(np.vdot(diff.ravel(), diff.ravel()))`. Carried the rewrite; fixed the PR's `import math` placement (it was appended after `import numpy as np`, violating isort stdlib-before-third-party grouping) and dropped its inline Bolt comment for the same reason as #11128.
- Verified equivalence: `sqrt(einsum("ij,ij->i"))` / `("ij,ij->j")` / `("...i,...i->...")` and `sqrt(vdot(x,x))` are algebraically identical to `np.linalg.norm` for real arrays at every axis/keepdims/1-D/empty combination used; `tests/unit/motion_matching/test_row_norm_einsum_identity.py` (carried from #11112) checks this directly, including NaN/Inf and empty-array edges. `np.vdot` conjugates its first argument, which is a no-op for the real `float64` arrays at every #11128/#11129 call site.
- Tests: `python -m pytest -q -o addopts="" -p no:cacheprovider --tools-mode vendored --timeout=0` over the test files covering every changed module (qualification_fit, club_calibration, club_only/acceptance+control_replay, dynamics_filter, hip_calibration, multi_shooting_fit, pipeline/reference, pink_tasks, physics_validation, plus the new identity test): 226 passed, 2 pre-existing failures unrelated to this change (`test_club_observation_contracts.py::test_public_input_validation_survives_python_dash_o` and the equivalent in `test_club_plausibility_acceptance.py` — both fail identically on `origin/main` with `ModuleNotFoundError: No module named 'bunkershot3d'` from a `python -O` subprocess whose `sys.path` doesn't include the repo root; unrelated to any …
- Next step: CI green on the refreshed head, then squash-merge this PR and close #11112/#11128/#11129 citing its squash SHA.

---

# Past Handoff — Anti-Phantom-Merge Path Extraction for Scripts, Workflows, and Parentheticals (#11124)

- Repository: D-sorganization/UpstreamDrift
- Branch: `fix/phantom-guard-scripts-paths-11124`
- Commit: `SELF`
- Pull request: closes #11124 (merged as #11131, squash `657d95287e`).
- Done:
  - Extended `ISSUE_PATH_PATTERN` in `scripts/ci/check_phantom_guard_paths.py` with boundary lookbehind `(?<![\w/.-])` and recognized repository prefixes `scripts/` and `.github/workflows/` (and `.github/`).
  - Added trailing punctuation trimming from extracted paths.
  - Implemented `_drop_parenthetical_prose_fragments` to filter comma-separated fragments inside parentheses that lack recognized code/configuration file extensions.
  - Added unit tests for scripts/workflows path extraction and parenthetical prose dropping, plus regression test reproducing issue #10965 / PR #11113.
- Tests: TDD red-to-green workflow followed; 27 unit tests pass in `tests/scripts/test_check_phantom_guard_paths.py`. Ruff, Black, LOD, and Architecture Budget clean.
- Resolution: merged ahead of the consolidator (keep-both rewrite of this section at #11138 refresh); superseded by the merge itself.

# Current Handoff — Govern Screenshot Schema, Capture Metadata, and Qualified Assets (#9191)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/comp-b3-screenshot-governance-9191`
- Issue: #9191
- Commit: `SELF`
- Pull request: Closes #9191.
- Done:
  - Updated screenshot schemas (`docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json` and `docs/api/contracts/upstreamdrift-companion-v1.schema.json`) with exact capture metadata (`capture_environment`, `source_commit`, pixels/viewport, and summary screenshot counts).
  - Implemented `scripts/companion_screenshots.py` single authority for screenshot verification, PNG IHDR byte extraction, SHA-256 digest validation, registry loading/validation, and headless deterministic asset generation with Matplotlib Agg backend.
  - Generated 6 representative deterministic PNG visual assets in `docs/screenshots/` (pendulum simulator desktop light/dark + mobile dark, project map, rate of closure, tour matching viewer).
  - Registered all 74 visible programs in `scripts/config/companion_screenshots.v1.json` (6 captured records, 70 pending records with explicit reasons).
  - Wired screenshot registry and inventory into `scripts/companion_catalog.py` and consumer publication bundles.
- Tests:
  - 165 passed across `tests/companion` and `tests/unit/scripts/test_verify_companion_screenshots.py`.
  - Architecture budget OK; DRY duplication gate OK; hardcoded style ratchet OK; ruff check and format clean; prettier clean.
- Next step: Create PR and release lease.

# Current Handoff — Implement Simscape Continuous-Replay Qualification Harness (#11107)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-07-simscape-continuous-replay-11107`
- Commit: `SELF`
- Pull request: addresses #11107 (bounded implementation).
- Done:
  - Implemented `src/shared/python/motion_matching/simscape_replay_harness.py`:
    - Full-rate, unbroken continuous-replay trajectory contract checking (`validate_continuous_replay_inputs`) with fail-closed enforcement for missing terminal samples, non-finite states, duplicated/non-monotonic timestamps, non-R2025b releases, hidden motion prescription, candidate hash mismatches, and state resets.
    - Added `compute_per_marker_channels` and `compute_phase_channels` generating per-marker and per-phase RMS error distributions over full native coordinates.
    - Integrated canonical metrics into `acceptance.evaluate()` without modifying frozen thresholds (`qualify_simscape_continuous_replay`).
    - Added structured qualification receipt schema (`simscape-continuous-replay-receipt/1`) with lossless serialization (`save_replay_qualification_receipt`, `load_replay_qualification_receipt`).
    - Implemented `load_continuous_replay_trajectory` to parse both explicit `q`/`v` and composite `native_state` arrays.
  - Added `-ContinuousQualification` switch to `scripts/matlab/run_simscape_candidate.ps1`.
  - Review fixes (Codex P1/P2 on this PR): duration gate requires start at t=0 and the elapsed horizon span; acceptance metrics delegate to canonical `compute_replay_five_metrics()` (0.60 s early window, canonical club-cluster labels); pelvis yaw measured from `WaistLeft`/`WaistRight`; unmeasured normal force/penetration/closure disclosed as unavailable instead of fabricated passes; receipt loader preserves null terminal channels; `load_replay_evidence_inputs()` derives the recomputed NPZ digest (fail-closed), candidate `q0`/`qd0`, and receipt-declared control identity; `replay_returned102_r2025b.m` records the actuation/reset identity (run-102 receipt extended additively, measured values unchanged).
  - Evaluated existing native candidate replay evidence (`two_window_fit_9967_102`) producing `simscape_replay_qualification.json`, confirming honest rejection at the terminal phase (~40.3 mm > 35.0 mm G1 ceiling); canonical metrics cross-check the committed native receipt (early 9.995 mm, club 8.391 mm, yaw 0.595 %, terminal 40.30 mm).
  - Implemented 22-test suite in `tests/unit/motion_matching/test_simscape_continuous_replay_harness.py` (11 fail-closed + 11 review-fix regressions, red-first documented in the PR body).
- Validation:
  - 22/22 tests passing in `test_simscape_continuous_replay_harness.py`.
  - Full `tests/unit/motion_matching` run: 1901 passed, 11 failed — all pre-existing on the pristine baseline (`c3d_reader.load_c3d` absent, `bunkershot3d` absent, stability-matrix precondition); not regressions of this change.
  - Lint/format/type checks clean on touched modules: `ruff check`, `ruff format`, `mypy`.
  - Section 12 changelog entry updated in `SPEC.md` and verified with `check_spec_changelog_duplicates.py`.
- Next step: CI green, PR review by owner.

# Current Handoff — Ship the Historical-Video Evidence Review Workflow (MMR-12, #11098)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-12-historical-video-review-11098` (baseline `origin/main` 7b74fb68c6)
- Pull request: closes #11098.
- Done:
  - Preserved multi-shot frame isolation and revision lineage in `DefaultShadowTrackerService`: frames from multiple distinct shots sharing local `frame_id`s (e.g. `frame-000000`) are stored without collision under scoped `(shot_id, frame_id)` namespace indexing and retrieved deterministically.
  - Manual mask updates invalidate downstream fit results deterministically while tracking revision lineage (`parent_revision_id`, `producer_id`, `correction_note`).
  - Added video file ingestion via `import_video`: container PTS extraction, constant and variable-frame-rate (VFR) containers, affine timing mappings (slow motion), and shot boundary cut definitions.
  - Bounded decode limits and prompt cancellation: passes `DecodeLimits` with active cancellation token callbacks to abort decode loops during ingestion rather than only upon completion.
  - Fail-closed error handling: invalid media and missing codecs raise descriptive errors while leaving active session state intact and recoverable.
  - Installed PyQt review journey: `ShadowTrackerWidget` and `ShadowTrackerReviewModel` support video import toolbar action, keyboard scrubbing (`Key_Left`, `Key_Right`, `Key_Home`, `Key_End`, `Key_W` for worst-frame jump), shortcut bundle persistence (`Ctrl+S`, `Ctrl+O`, `Ctrl+I`), viewport rendering with clock authority metadata, and honest refusal reporting for unverified automated fitting backends.
  - Review fixes (Codex P1s on this PR): `import_video` stages the full decoded batch and validates scope ownership, single source asset, and revision-id uniqueness before any session mutation, so a failing repeat import leaves the reviewed session exactly intact; without an evidenced timing mapping imported frames keep `physical_time_s=None` with the canonical unknown-time reason (PTS authority stays in `timing_mode`/`clock_evidence`, never transplanted into fabricated provenance); the viewport renders unknown physical time as `Physical Time: unknown (<reason>)` via `format_clock_evidence_text` instead of formatting `None` (pre-fix `TypeError` aborted the paint event on GUI-default imports).
- Tests:
  - `tests/unit/shadow_tracker/test_service.py`: 16 passed (incl. repeat-import atomicity against scope collision / foreign asset, unknown-time provenance preserved in export, VFR/slow-motion timing, cancellation during decode, corrupt media recovery, bundle save/reload lineage).
  - `tests/tools/shadow_tracker/test_shadow_tracker_gui.py`: 7 passed (incl. unknown-time viewport rendering and clock-evidence text formatting, installed PyQt review journey, keyboard navigation, worst frame jump, mask correction dirty tracking, honest fit refusal).
  - Entire `tests/unit/shadow_tracker/` and `tests/tools/shadow_tracker/` suites: 330 collected, all passing.
  - Ruff check/format and mypy clean on touched modules; architecture budget and spec changelog duplicate check: PASS.
- Next step: CI green, then ready and arm the PR; release agent lease.

# Current Handoff — Repair Reduced-Model and Club-Only Product Claims (MMR-11 #11097)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-11-reduced-model-claims-11097`
- Commit: `SELF`
- Pull request: Refs #11097 (partial: fail-closed gates shipped; not claiming "Closes" — see Open below).
- Done (all code-level, unit-test-verified):
  - Disqualified four historical unverified driven-triple pendulum receipts (`driver_amateur`, `driver_elite`, `iron_amateur`, `iron_elite`) at projection time in coverage; driver/iron matrix cells report rejection truthfully. Receipt/evidence files on disk were NOT rewritten.
  - Enforced complete geometry, inertia, control, q0, and v0 hashes and non-positive out-of-plane residual rejection (`out_of_plane_rmse_m <= 0.0`) on promoted baseline packages, failing integrity checks fail-closed. Committed packages were NOT regenerated from raw artifacts in this PR.
  - Early planar floor rejection in `validate_and_project_target` gates the CURRENT target's butt/clubhead distances to the selected plane against a DbC-validated `max_marker_rmse_m` (finite, strictly positive; NaN ceilings fail closed instead of silently disabling the gate), rejecting before expensive optimization.
  - Required fresh continuous replay (`has_continuous_replay`) and labeled inferred body posture (`body_motion_disclaimer`) in club-only UI result views, preventing unqualified views from displaying verified status.
  - Matrix qualification reports cannot claim all-complete while unresolved cells remain (`assert_matrix_not_all_complete_with_unresolved`); the ~80 unresolved cells are still unresolved and tracked.
- Open (MMR-11 acceptance criteria NOT satisfied by this PR): raw-to-package reproduction/regeneration of reported baselines and residuals, Board-selected required club-only cells, and any native qualification run.
- Tests: TDD red-to-green for both Codex findings (current-target planar-floor gating; ceiling DbC validation) plus all fail-closed gate tests. Scoped runs at `46b804bc69`: 18 passed (`tests/unit/motion_matching/test_fit_options_dbc.py`, `tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py`) and 86 passed (`tests/unit/tour_baselines/test_coverage_matrix.py`, `test_qualification.py`, `test_baseline_packages.py`, `tests/unit/motion_matching/test_club_matrix_qualification.py`, `test_club_ui_integration.py`). `ruff check` clean on changed files. Full suite, mypy and CI gates were not run by this slice.
- Next step: raw-to-package regeneration and Board cell-selection follow-ups for #11097.

# Current Handoff — Publish a Best-Candidate Viewer With Honest Residuals (#11102)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-16-best-candidate-viewer-11102`
- Issue: #11102
- Pull request: refs #11102 (mergeability and PR body owned by the main lane; desktop-slice scope below)
- Done:
  - Enforced raw observation immutability on `ReplayData` by setting numpy array flags to `writeable = False` on `time_s`, `coordinates`, `model_markers_m`, `target_markers_m`, and `valid_mask`.
  - Added dataclasses `MarkerResidual`, `FrameResidual`, and `ResidualSummary` with `mean_rms_m` property.
  - Implemented `compute_residual_summary` with per-frame RMS, per-marker errors, phase boundaries (Address, Top, Impact, Finish), and 3D residual vectors `(model - target)`.
  - Implemented `export_board_ready_still` generating high-DPI stills with observed dots, model skeleton, residual vector lines, and comprehensive metadata banner (Candidate SHA, Engine, Drive Mode, Frame, RMS, Verdict).
  - Extended `MatchedSwingFilter` with `drive_mode` and `profile`. Implemented `extract_drive_mode` and `rank_candidates(rows, capture, drive_mode)` sorting comparable candidates in strictly ascending RMS error order while preserving disqualified/rejected verdicts honestly.
  - Wired `rank_candidates` into the browser's real list-build path (`_apply_filters`) so the auto-selected first row is the best comparable candidate (ascending `whole_marker_rmse_m`); rejected rows remain visible with their verdicts.
  - Fixed the viewer frame RMS (`_evaluate_single_frame_residual`, `viewer_frame`, `get_per_engine_rms`) to pool per-marker 3D distances (`sqrt(mean(sum(valid_diff**2, axis=-1)))`) matching the canonical `tour_metrics.compute_shared_metrics`, so residual summaries, physics scores and captions agree with the ledger.
  - Frames with zero valid markers claim no worst marker (`FrameResidual.valid_markers`) and are excluded from global-worst selection and `mean_rms_m`, so the Worst Residual jump never lands on unobserved placeholder data.
  - `TourMatchingViewerWidget.load_file` now accepts optional receipt provenance (`candidate_hash`, `engine_name`, `drive_mode`, `is_accepted`, `rejection_reason`); the browser's `_on_open_tour_matching_viewer` forwards the selected `LedgerRow`'s hash, engine, drive mode and verdict so captions match the selected receipt and rejected candidates show their failure banner.
  - Extended `TourMatchingViewerWidget` with properties `current_frame`, `current_rendered_frame_index`, `current_frame_time_s`, `current_rms_error`, `physics_score`, `candidate_hash`, `drive_mode`, `title_caption`, and `capabilities_caption`.
  - Implemented `select_worst_residual()` jumping directly to the exact discrete frame with the highest marker residual error.
  - Implemented camera view presets (`perspective`, `front`, `side`, `top`, `isometric`) and appearance presets (`default`, `high_contrast`, `residual_vectors`, `dots_and_mesh`) preserving physics scores and numerical receipts.
  - Implemented `launch_native_backend` and graceful `_show_recovery_message` handling missing engines without crashes.
- Tests (scoped, `/tmp/ud-venv-11132`, offscreen PyQt6 + mujoco): red outcomes for all four review fixes recorded pre-fix; post-fix `pytest tests/unit/tools/test_matched_swing_browser_best_candidate.py tests/unit/tools/test_tour_matching_viewer_residuals.py` → 21 passed and the targeted viewer suites (core, playback, combo, forces, native_button, adapter) → 29 passed; two combo pins that predated the review fixes updated (old flattened-RMS value and removed `#d9534f` hex literal, see 70762eb7fe). Exit code 0 on every run.
- Not done (remains open on #11102 — the desktop PyQt seam above is fixed, but the issue's acceptance does not stop there):
  - The same journey is not exercised on the supported web/API surfaces (installable-PyQt and web/API parity, accessibility keyboard review and missing-engine recovery on real hosts are unreviewed).
  - Human visual review of captions/stills and acceptance checkboxes in #11102 are unchecked; no claim is made that "all acceptance criteria" pass.
- Next step: main-lane rebase/CI and frontier review of PR #11132.

---

# Past Handoff — Unify Per-Package Coverage Gates on the Exclusion Budget (#10965)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `/tmp/ud-wt-10965` (baseline `origin/main` 64962e8d6)
- Branch: `bot/issue-10965-coverage-gate-authority`
- Commit: `SELF`
- Pull request: closes #10965.
- Done: `scripts/check_coverage_gates.py` reads the 6 gates directly from `scripts/config/mypy_exclusion_budget.json` (single DRY gate authority), matches coverage-report files by repository-relative path prefix using `summary.num_statements`/`covered_lines`, and `--strict` now fails (exit 1) on a gate matching zero files; without `--strict` it warns instead. `check_mypy_exclusion_budget.py` treats an expired gate `ratchet_on` as a non-fatal warning (gates are CI-enforced) and makes `ratchet_to` optional. The `tests` CI job emits `--cov-report=json:coverage.json` and a new step runs `python3 scripts/check_coverage_gates.py --report coverage.json --strict` after the threshold enforcer.
- Measured floors (min_coverage rounded DOWN to 0.1, ratchet_on 2027-01-01, local targeted pytest with coverage, optional-engine backends not installed): api-routes 84.3% (4406/5221), data-io 50.1% (1450/2891), execution-checkpointing 22.4% (567/2521), deployment 48.4% (660/1362), optimization 49.9% (1405/2815), engine-adapters 7.1% (3687/51393).
- Tests: TDD red (2 new unmatched/strict tests failed with old exit-2 semantics) then green; `tests/unit/scripts/test_check_coverage_gates.py` + `test_check_mypy_exclusion_budget.py`: 29 passed. Checker smoke against a real coverage.json fixture: strict exit 1 on unmatched gates, lax exit 0 with WARNING line.
- Next step: CI green, then ready and arm the PR.

---

# Past Handoff — Fleet Critic Pass 2026-09-25

## Identity

- Repository: D-sorganization/UpstreamDrift
- Branch: `staff/fleet-critic-task-653b92-v2`
- Baseline commit: `28b37bd47` (origin/main)
- Implementation commit: SELF
- Pull request: not created (supersedes #10942)
- Governing task: Fleet Critic scheduled pass (bi-weekly, 1st & 3rd Friday)
- Session: `fleet-critic-task-653b92`

## Objective and Status

- Objective: Produce the 2026-09-25 scheduled Fleet Critic review for
  UpstreamDrift, covering the last 30 days of changes (NM-09–NM-12,
  TB-08–TB-10, Bolt optimizations), and commit the artifact under
  `docs/critiques/2026-09-25/`.
- Status: Complete — critique committed, draft PR to be opened. Re-landed as a
  superseding PR because #10942's branch conflicted with main.
- Completed:
  1. Reviewed recent commits; focused on neural-motion matrix builder,
     tour-baseline qualification, benchmark runner, and Bolt journal.
  2. Produced `docs/critiques/README.md` (index), `summary.md`, and
     `weaknesses.md` (6 findings: 3 High, 2 Medium, 1 Low).
  3. Added this handoff section.
- Remaining: Push branch and open draft PR.

## Key Findings

Three High-severity weaknesses in the neural-motion checkpoint pipeline:

1. `matrix/builder.py:_make_evidence()` — synthetic three-seed evidence.
2. `matrix/builder.py:_build_card_for_model()` — model-ID-derived hashes.
3. `benchmark/runner.py:run_model_comparative_benchmark()` — self-referential
   speedup baseline.

Two Medium: hardcoded economics in model cards; refinement sensitivity proxy.
One Low: Bolt speedup claims without benchmark fixtures.

## Files Changed

- `docs/critiques/README.md` — new critique index
- `docs/critiques/2026-09-25/summary.md` — executive summary
- `docs/critiques/2026-09-25/weaknesses.md` — 6-finding weakness catalog
- `docs/development/HANDOFF.md` — this section (inserted, not a replacement)
- `docs/index.md` — catalog row for the new `docs/critiques/` directory (doc-catalog gate)
- `SPEC.md` — one change-log row for this pass

## Validation

- No source code changed; linting/tests not required.
- `python scripts/ci/check_spec_changelog_duplicates.py` — passed.

## Blockers and Risks

- None. Read-only critique pass; no source modifications.

## Next Steps

1. Commit these files.
2. Push `staff/fleet-critic-task-653b92-v2` and open a draft PR targeting `main`.
3. High-severity findings 1–3 should be forwarded to the next Board meeting
   as candidates for the consensus priority list.
4. Author/owner to decide remediation order; suggested tracking via one
   consolidated issue covering all three High findings.

## Change Log

- SELF — Fleet Critic scheduled pass: 6 scientific weaknesses in neural-motion
  checkpoint matrix and benchmark runner.

---

# Past Handoff — Consolidated Bolt Row-Norm Micro-Optimisations (#11073, #11074, #11076)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-bolt-einsum`
- Branch: `claude/ud-bolt-einsum-consolidated` (baseline `origin/main` 0206717f)
- Commit: `SELF`
- Pull request: #11081 (https://github.com/D-sorganization/UpstreamDrift/pull/11081). It supersedes the Bolt PRs #11073, #11074 and #11076, which conflicted on `SPEC.md` and carried `n/a` change-log rows.
- Done: took only the `src/` changes from the three Bolt PRs. Row-wise `np.linalg.norm(..., axis=1)` becomes `np.sqrt(np.einsum("ij,ij->i", d, d))` in `contact_identification`, `multi_shooting_fit` and `tour_baselines/calibration`, and max-norm reductions take one square root after `np.max` in `contact_mode_qualifier` and `calibration`. The unmeasured inline speed claims were removed.
- Measured (NumPy 2.2.6, this workstation): 4.8x on 20000x3 rows and 1.8x on 657x3 rows; results match `np.linalg.norm` to 1e-15.
- Excluded: #11077. On top of the same edits it reverts the `nightly-cross-engine.yml` hardening, deletes cross-engine validation tests and raises the architecture budget.
- Validation: `test_contact_identification`, `test_contact_mode_qualifier_pf04`, `test_multi_shooting_fit` and `test_tour_calibration`: 54 passed. Architecture budget OK. Divergence inventory regenerated (all four files are UD-only).
- Next step: CI green, then ready and arm the PR; close the three Bolt PRs as superseded after merge.

---

# Past Handoff — Run the MJX L-BFGS Arm Toward Convergence (#11071)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11071-lbfgs`
- Branch: `claude/ud-11071-lbfgs-converge` (baseline `origin/main` d4286fbf)
- Commit: `SELF`
- Pull request: #11075 (https://github.com/D-sorganization/UpstreamDrift/pull/11075); closes #11071 (follow-up to #11058, epic #11006).
- Done: re-ran only `mjx-lbfgs` at `--mjx-iterations 60` with the MJX interpreter (Python 3.12.10, MuJoCo 3.13.0, DeskComputer, commit ec249edf, whose `src/` and `scripts/` match main d4286fbf). `none` and `shooting` do not read `--mjx-iterations`, and only `knot_gradient_optimiser.py` changed since their #11058 runs, so their cases hold a `reuse.json` pointer to `../mjx_benchmark` instead of copies; the ledger therefore does not count them twice. `load_row` follows the pointer and rejects a dangling one (`tests/unit/motion_matching/test_benchmark_mjx_knot_solvers_script.py`).
- Budget choice: the #11058 L-BFGS runs took about 1.6-2.8 ks for 10 iterations. 60 iterations took 6217 s (driver) and 5097 s (iron), 70 evaluations each.
- Result (replay RMS through the shared simulator, driver / iron): 40.8 / 42.6 mm, down from 47.2 / 52.9 mm at 10 iterations; shooting stays 84.5 / 88.6 mm. Downswing weight fraction min 0.31 / 0.29, so the iron did **not** lose contact (it rose from 0.17). Both runs stopped on `max_iterations`.
- Trend: the MJX-plant RMS in the optimiser history went 37.6 → 28.2 mm (driver) and 34.5 → 27.3 mm (iron) between evaluations 10 and 69, about 0.05 mm per evaluation at the end. The shared-simulator replay stays 12-15 mm above the MJX plant, so more iterations mostly shrink the plant number, not the scored one.
- Decision: keep `none` as the default (the promotion rule needs a `converged` stop). 60 iterations (about 1.5 h per capture) is the practical budget reached.
- Evidence: `docs/development/full_body_models/evidence/mjx_benchmark_lbfgs60/` (receipts, `provenance.json`, `rows.json`, `REPORT.md`); ledger 133 receipts.
- Next step: none; merged as PR #11075.

---

# Past Handoff — MJX Knot Optimiser Head-to-Head Benchmark and Promotion Decision (#11058)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11058-benchmark`
- Branch: `claude/ud-11058-benchmark` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11072. Closes #11058 (epic #11006 P5).
- Done:
  - `knot_gradient_optimiser.lbfgs_minimise` runs SciPy L-BFGS-B on the same JAX gradient as Adam; `--mjx-method {adam,lbfgs}`.
  - Every MJX reference is rescored through the shared `FullBodySimulator` (`trajectory_optimiser.shared_simulator_replay`, via `pipeline.dynamics.score_reference`). A stage that writes no reference is an error.
  - `weight_fraction_report` gives receipts a per-phase weight fraction.
  - `solver_benchmark.py` scores the receipts and makes the promotion decision; `scripts/benchmark_mjx_knot_solvers.py` runs the benchmark or re-renders it (`--render-only`).
- Evidence: the receipts, `rows.json` and `REPORT.md` are in `docs/development/full_body_models/evidence/mjx_benchmark`. The runs used the MJX interpreter (Python 3.12.10, MuJoCo 3.13.0) on DeskComputer at commit `fee18106`, with shooting at 8 passes and MJX at 10 iterations, one run at a time.
- Result (replay RMS, driver / iron): none 84.5 / 88.6 mm; shooting 84.5 / 88.6 mm (never beats its start); mjx-adam 62.4 / 55.7 mm, but it **loses iron downswing contact** (weight fraction min 0.00); mjx-lbfgs 47.2 / 52.9 mm (weight fraction min 0.33 / 0.17). No arm reaches G1 (25 mm). IPOPT is unavailable: no cyipopt or pydrake binding.
- Decision: keep `none` as the default. Both MJX methods stopped on the iteration budget, not on convergence. Follow-up #11071 runs L-BFGS to convergence.
- Deviation: the issue's `least_squares` FD arm was not run, because one Jacobian means 47 knots × 38 actuated coordinates = 1786 shared replays. L-BFGS-B on the exact gradient replaces it; the report says so.
- Tests: `tests/unit/motion_matching/` (solver benchmark, trajectory optimiser selection, knot gradient optimiser, MJX optimisation, pipeline) pass in Python312 and the MJX environment.
- Next step: none; merged as PR #11072.

---

# Past Handoff — Force-Plate Stitching Tests Canonical Import and Overlay Cleanup (#11034)

- Repository: D-sorganization/UpstreamDrift
- Branch: `fix/force-plate-test-module-identity-11034` (baseline `origin/main`)
- Pull request: closes #11034.
- Problem: In `unit-test-gate` under `pytest -n auto`, `test_force_plate_stitching.py` intermittently failed with `AttributeError: module 'shared.python.sidekick.lab.bio.force_plate_stitching' has no attribute 'CombinedForcePlateProcessor'` due to in-test manipulation of `sys.modules` and overlay reinstallation resolving against stale parent module attributes.
- Fix:
  1. `test_force_plate_stitching.py`: import canonical `CombinedForcePlateProcessor` directly via `from src.shared.python.sidekick.lab.bio.force_plate_stitching import CombinedForcePlateProcessor`.
  2. `sidekick_extension_overlay.py`: when uninstalling loaded extension modules, cleanly `delattr` the attribute from the parent module in `sys.modules` if present to prevent attribute pollution.
- Tests: `pytest tests/unit/sidekick/lab/bio/test_force_plate_stitching.py tests/unit/launcher/test_sidekick_extension_overlay.py` and `pytest -n 2 ...` pass (13 passed).
- Next step: none; merged as PR #11069.

---

# Past Handoff — Shooting Fit No Longer Crashes on an Unimported `fs` (#11059)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11059-shooting`
- Branch: `claude/ud-11059-shooting-fs` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11060 (draft → ready). Closes #11059.
- Problem: `pipeline/dynamics.py` imports `full_body_forward_dynamics as fs` only under `TYPE_CHECKING`; `build_tracking_controller`, `replay` and `zmp_filter` import it locally, but `shooting_fit` did not, so `--shooting-fit N` raised `NameError: name 'fs' is not defined` before its first pass.
- Fix: the same local import in `shooting_fit`.
- Tests: `test_shooting_fit_runs_a_pass` runs one pass with stubbed replay and marker errors; it fails on `main` with the production `NameError` and passes with the fix. `tests/unit/motion_matching/pipeline/`: 77 passed, 10 skipped.
- Acceptance: the driver run `--static-seeds --shooting-fit 8` that crashed on `main` completes (rc 0, 1123 s). Pass 0 replays at 84.5 mm, matching the canonical `anthro_driver_seeds` receipt. Passes 1-8 score 108.6, 121.3, 122.0, 114.5, 118.8, 111.1, 96.3 and 95.9 mm, so the fit keeps pass 0 (`best_iteration` 0) and the final dynamics replay is 84.5 mm.
- Next step: CI green, mark ready and arm the PR.

---

# Past Handoff — Opt-In MJX Knot Trajectory Optimiser Stage in the Matching Pipeline (#11051)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11051-pipeline`
- Branch: `claude/ud-11051-mjx-pipeline` (baseline `origin/main`)
- Commit: merged to `main` as PR #11057
- Pull request: #11057 (https://github.com/D-sorganization/UpstreamDrift/pull/11057). Closes #11051 (epic #11006 package P4). Executed by agy (Gemini 3.8 Flash), reviewed and reworked by the orchestrator.
- Built: `pipeline/trajectory_optimiser.py` (`TRAJECTORY_OPTIMISERS = {none, mjx-knots}`, `run_trajectory_optimiser`) and the CLI flags `--trajectory-optimiser` (default `none`) and `--mjx-iterations` (default 40, `DEFAULT_MJX_ITERATIONS`). With `mjx-knots` the stage exports the MJX package, runs the #11049 knot optimiser, and writes `mjx_optimised_reference.npz` and `mjx_optimisation_receipt.json`; `receipt.json` gains a `trajectory_optimiser` summary. The receipt builder is shared with the #11046 evidence CLI (`optimisation_receipt`).
- Decision: when JAX or `mujoco.mjx` is missing, selecting `mjx-knots` raises `DependencyUnavailableError` naming the module and `--trajectory-optimiser none`. The epic's "falls back cleanly" is read as a clear error, not a silent fallback, because the default stays `none` and a silent skip would hide that the requested stage never ran.
- Tests: the six prototype tests in `test_mjx_optimisation.py` loaded private copies from the evidence script that #11046 replaced; they duplicated `test_knot_gradient_optimiser`, `test_jax_contact` and `test_mjx_tracking_plant`, so they were replaced by a constant-parity test and a toy-package stage test. `test_trajectory_optimiser_selection.py` covers parser defaults and rejections, the `none` no-op, both missing-dependency errors and the receipt key. 29 passed in `~/.venv-mjx` (MuJoCo 3.13); pipeline tests 76 passed, 10 skipped in default Python.
- Acceptance: `--trajectory-optimiser mjx-knots --mjx-iterations 3` on the static-seeds run: port check 0.06516 m, best 0.05029 m (47 knots, 38 actuated coordinates, float32); the evidence CLI on a copy of the same package gives an identical history. Default run on branch vs `main`: identical file set, `receipt.json` differs only in `elapsed_s` and has no `trajectory_optimiser` key.
- Next step: none; merged as PR #11057.

---

# Past Handoff — Every `mj_fullM` Call Routed Through One MuJoCo 3.13-Safe Helper (#11055)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11055-fullm`
- Branch: `claude/ud-11055-fullm-helper` (baseline `origin/main`)
- Commit: merged to `main` as PR #11056
- Pull request: #11056; entry DL-#11055. Closes #11055. Executed by agy (Gemini 3.8 Flash), reviewed by the orchestrator.
- Problem: `pyproject.toml` allows `mujoco>=3.6,<4`, but MuJoCo 3.13 removed `MjData.qM` and changed `mj_fullM(model, dst, data.qM)` to `mj_fullM(model, data, dst)`; measured in `~/.venv-mjx` (3.13.0) the old form raises `AttributeError`, which stopped the matching pipeline there.
- Built: `src/shared/python/engine_core/mujoco_compat.py::full_mass_matrix(mj, model, data, dst=None)` picks the signature at run time and imports no `mujoco` at module level; every `src` call site (mujoco engine, myosuite engine, motion matching, simulation backends, physics validation), including the two existing hand-written dual paths, now calls it. `src/engines/physics_engines/mujoco/docker/` keeps its own pinned MuJoCo and is untouched.
- Orchestrator changes on review: the hinge test checks the analytic inertia (2/5 m r^2 = 0.004 kg m^2) instead of repeating the helper's branch, and the package-level re-export was dropped.
- Placement: the helper lives in `engine_core`, not `simulation_backends` as #11055 proposed; importing anything under `simulation_backends` runs its package `__init__` (measured 8.98 s, 1671 modules), while `engine_core` costs 0.19 s (158 modules, the `src.shared.python` baseline), and the helper is imported by 19 modules.
- DRY ratchet: shortening the four `mj_fullM` call sites in `counterfactuals.py` made the ZTCF/ZVCF blocks hash-identical, so the state-load, forward, mass-matrix and solve sequence is one `_forward_acceleration` helper; ZTCF and ZVCF outputs on a two-link model match `main` exactly.
- Changed-file gates surfaced existing debt in touched files, fixed rather than excepted: `advanced_kinematics.solve_inverse_kinematics` (111 > 100 lines) has its damped-least-squares and nullspace step extracted to `_dls_step`, same arithmetic; `sim_widget.set_axial_color_scale` calls `_render_once()` (what the mixin's `render()` does) because mypy resolves `self.render()` to `QWidget.render`, which needs an argument; the divergence inventory records `mujoco_compat.py` as `ud-only`.
- Tests: `tests/unit/engine_core/test_mujoco_compat.py` (analytic hinge, preallocated `dst`, both branches with fakes, three preconditions) plus `test_native_force_equations.py`: 12 passed on MuJoCo 3.13. The 200 test files that reference a touched module, in default Python (MuJoCo 3.4): 1967 passed; the 31 failures are 19 that fail identically on `main` and 12 that pass when run alone (batch-order pollution).
- Next step: none; merged as PR #11056.

---

# Past Handoff — MJX Knot Optimiser Core Moved Into `src` (#11049)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11049-optimiser`
- Branch: `claude/ud-11049-mjx-optimiser` (baseline `origin/main`)
- Commit: merged to `main` as PR #11054
- Pull request: #11054; entry DL-#11049. Closes #11049. Package P3 of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed and refactored by the orchestrator.
- Built: `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py` holds the package loader (`load_mjx_package`, strips equality blocks and floors armature), `KnotOptimisationSettings` (frozen, validated), `optimise_reference` and `diagnose_reference`, sharing one `_prepare` problem builder. The module writes no files and does not change JAX config; the result records the dtype it ran in.
- The evidence CLI `mjx_trajectory_optimisation.py` is now a thin wrapper (362 → 268 lines); every flag, output file and receipt key is kept, and `main()` still runs float32.
- Measured parity on the regenerated `anthro_driver_seeds` package (`~/.venv-mjx`, float32, `--iterations 3`), pristine #11046 CLI vs this CLI: replay marker RMS and total cost identical at every iteration (65.16494, 52.73439, 54.64461, 50.29464 mm; 0.0 relative difference); 47 knots, 38 actuated coordinates.
- Tests: `tests/unit/engines/mujoco/test_mjx_knot_optimiser.py` (17 tests: settings validation, loader, missing file, missing root coordinate, `init_delta` shape, iteration callback, diagnose) and the evidence CLI test share `tests/unit/engines/mujoco/mjx_toy_package.py`; 29 passed in `~/.venv-mjx`, skip without JAX; the CLI test restores `jax_enable_x64` so later x64 tests are unaffected.
- Next step: none; merged as PR #11054.

---

# Past Handoff — MJX Evidence Prototype Rewired Onto the Tested `src` Plant (#11046)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11046-rewire`
- Branch: `claude/ud-11046-mjx-rewire` (baseline `origin/main`)
- Commit: merged to `main` as PR #11048
- Pull request: #11048; entry DL-#11046. Closes #11046. Package 2c of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `evidence/ground_support/mjx_trajectory_optimisation.py` is a thin CLI (648 → 362 lines) over `mjx_tracking_plant` (`TrackingPlantSpec.from_package`, `build_tracking_plant`, `reference_derivatives`, `substep_tables`), `jax_contact.WeldGains` and `knot_gradient_optimiser` (`knot_grid`, `knot_basis`, `horizon_knot_mask`, `AdamSettings`, `adam_minimise`). Every flag, output file and receipt key is kept.
- Orchestrator changes on review: the root vertical coordinate is looked up by name (`TranslationInputZ`) instead of a heuristic, and the new CLI test used a coordinate name the documents do not have.
- Measured parity on the regenerated `anthro_driver_seeds` package (`~/.venv-mjx`, float32), old prototype vs rewired: `--iterations 0` port check 65.16457 vs 65.16494 mm (5.6e-6 relative); `--iterations 3` best replay 50.249 vs 50.295 mm (9.2e-4 relative; float32 reordering differences grow per Adam step: 5.6e-6, 1.6e-5, 7.2e-5, 9.2e-4).
- Tests: `tests/unit/engines/mujoco/test_mjx_evidence_cli.py` (toy package, `--iterations 2`) passes in `~/.venv-mjx` and skips without JAX; ruff and format clean.
- Next step: CI green, mark ready and arm the PR.

---

# Past Handoff — PR-Scoped Tests That All Skip No Longer Fall Back to Whole-`src` Coverage (#11052)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11052-ci`
- Branch: `claude/ud-11052-ci-exit5` (baseline `origin/main`)
- Commit: merged to `main` as PR #11053
- Pull request: #11053; entry DL-#11052. Closes #11052. Workflow-only change, so it ships alone.
- Cause: in `ci-standard.yml` (Run Core Test Suite), a PR-scoped pytest run that collects nothing (exit 5, e.g. a module-level `importorskip("jax")`) fell through to the dependency-light lane with whole-`src` coverage and the 75 % floor. That lane covers about 12 % of `src`, so it failed every such PR (#11048: 2562 passed, "Total coverage: 11.73%").
- Built: when exit 5 happens and no `src`/dependency coverage target changed, the step now reports "Core test suite NOT EXECUTED" (warning plus step summary, the #8771 wording), sets `core_suite_executed=false` and `coverage_generated=false`, and exits 0. With changed source targets, the existing fallback is unchanged.
- Tests: `tests/ci/test_ci_infrastructure.py::test_all_skipped_selection_without_source_changes_is_not_executed` (red before, green after); `tests/ci/` 201 passed, 1 skipped (`--tools-mode vendored`).
- Next step: none; merged as PR #11053.

---

# Past Handoff — Canonical Calibrated Runs Regenerated on Current Code (#11044)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11044-canonical`
- Branch: `claude/ud-11044-canonical` (baseline `origin/main`)
- Commit: merged to `main` as PR #11050
- Pull request: #11050 — merged to `main` as PR #11050
- Finding: the canonical calibrated runs (driver 5.1 / 27.3 / 74.6 mm, 7-iron 4.4 / 28.6 / 144.0 mm) were produced with `BOUND_WIDENING = 2.0`, and both receipts record IK range-of-motion flags (10 and 9 coordinates). Measured bisect on `main` @ `3a11b5c07`, driver `--static-seeds`: current 7.9 / 34.1 / 84.5 mm; pre-HO-11 base spec identical; `BOUND_WIDENING = 2.0` 6.3 / 28.1 / 53.8 mm with 10 flags; hip zero-twist off 6.0 / 36.0 / 82.0 mm; all three reverted 4.9 / 25.7 / 55.1 mm. 7-iron `--static-seeds --zmp-filter`: 6.6 / 31.6 / 88.6 mm, and 4.1 / 23.5 / 53.4 mm with 9 flags at 2.0.
- Built: new canonical receipts `evidence/ground_support/anthro_driver_seeds` and `anthro_iron_seeds_zmp` (receipt plus final scaled spec, added to the provenance-chain test), `bisect_11044_receipt.json` generated from the run receipts, `CANONICAL_RUN.md` §2A/§2C/§3 revised, citations updated in `matched_swing_program/README.md`, `full_body_models/HANDOFF.md` and a correction note in `plans/tour_baselines/final_acceptance_report.md`; ledger and status section regenerated.
- Gate consequence: G1 (whole-swing IK <= 30 mm, 0 RoM violations) is not met by any MuJoCo receipt; the README gate row and the generator's MuJoCo status now say so. `verdicts_2026-09.json` is a dated record and was not edited.
- Next step: none; merged as PR #11050.

---

# Past Handoff — Ball-Flight Parity Fixture Export Is Opt-In (#11008)

---

# Implementation Handoff - Drift Wizard Sidekick Product Knowledge Pack (#10943)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-parity-fixture-optin`
- Branch: `fix/ball-flight-parity-fixture-opt-in` (kept current by merging `origin/main`)
- Implementation commit: `SELF`
- Pull request: #11012 — merged to `main` as PR #11012
- Governing issue: #11008 (DL-#11008); SPEC row keyed by PR #11012

## Objective and Status

- Objective: stop `test_export_reference_vectors` from rewriting the committed golden
  `tests/parity_fixtures/ball_flight/default_trajectory.json` on every Rust-enabled run.
- Done: vector construction is in `build_default_trajectory_vectors()`, the schema is in
  `assert_vector_schema()`, and writing goes through `write_vectors()` (LF, round-trip
  checked). The export test now writes to `tmp_path` and asserts that the committed bytes
  are unchanged. `test_regenerate_committed_fixture` rewrites the committed file only when
  `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1` (exactly `1`).
- New `TestParityFixtureContract` (not Rust-gated, marked `unit`): checks the opt-in switch,
  the committed fixture's schema, the fixture path pinned in
  `src/config/capability_migration.json`, and that malformed vectors are rejected.
  `pytestmark` became a class-level `@requires_rust` so the contract tests run in every lane.
- The fixture bytes and the path are unchanged, so the sha256 pin stays valid.

## Validation

- `python -m pytest tests/parity/test_ball_flight_parity.py -o addopts=""` (Python 3.12, Rust
  kernel available, main merged through #11045): 19 passed, 1 skipped (opt-in regen).
  `git status` stays clean after the run.
- With `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1 ... -k regenerate`: the fixture is rewritten (reverted).
- `ruff check` / `ruff format --check`, the suite-marker ratchet and the SPEC changelog check pass.

## Blockers and Risks

- The unit-test gate intermittently errors in `test_force_plate_stitching.py` (#11034, not
  this change); #11042 added diagnostics, the fix is pending.
- The committed golden is stale against the current model: #11004 (merged) did not regenerate
  it, so it still records carry 187 m against 243 m from the current model. No test compares
  values against it. Regenerating it and updating the sha256/size pin is a follow-up.

## Next Steps

1. Keep the branch current with `origin/main` until auto-merge lands it.
2. After merge, file or pick up the fixture regeneration follow-up.

---

# Past Handoff — Replays Start on the Dual-Grip Weld (#11043)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11043-dls`
- Branch: `claude/ud-11043-dls` (baseline `origin/main`)
- Commit: merged to `main` as PR #11047
- Pull request: #11047; entry DL-#11043. Closes #11043.
- Root cause: the KKT solve enforces the weld only at acceleration level (`J a = -J̇ q̇`), so it conserves the relative weld velocity it starts with. `replay` seeded `v0` with the one-sided finite difference of the smoothed reference, which violated the weld by 187 mm/s and 32.1 deg/s on the `--fit-closure` run (17.6 mm/s on `--static-seeds`). The grip opened linearly, gave the weld Jacobian nonzero root columns, and a controller direction crossed `MIN_SINGULAR_VALUE` at t = 0.333 s: 9.5 kN·m on `RScapInputY` in one step.
- Built: `motion_matching/weld_manifold.py::project_onto_weld` (mass-weighted projection, the weld's own inelastic impulse), `FullBodySimulator.consistent_velocity`, with `_mass_and_weld` extracted from `affine_dynamics`. `pipeline/dynamics.replay` and `execution/downswing` now start from the projected velocity.
- Measured with the full pipeline on this branch (driver, DeskComputer, Python 3.12): `--static-seeds --fit-closure` 861.2 → 83.0 mm dynamics RMS, weight fraction max 50.0 → 3.72, peak torque 29 968 → 1 124 N·m, support 0.50 → 0.98, backswing root 493 → 8 mm. `--static-seeds` control 83.0 → 84.5 mm, weight fraction 3.44 → 3.76, peak torque 719 → 912 N·m, support 0.95 → 0.96.
- Tests: `tests/unit/motion_matching/test_weld_manifold.py` (6), plus `test_full_body_simulation.py`, `pipeline/test_dynamics.py` and the downswing tests: all pass. Reverting the `replay` line fails `test_replay_starts_on_the_weld`.
- Not done: the committed evidence receipts were not regenerated; the pipeline still has no fail-closed guard on a diverged replay (the issue's fallback acceptance).
- Pre-push mypy caught an un-annotated `Array` alias in `weld_manifold.py`; it is now `TypeAlias`.
- Next step: none; merged as PR #11047.

---

# Past Handoff — Force-Plate Fixture Reports Module Identity on the Intermittent Failure (#11034)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11034-fp`
- Branch: `claude/ud-11034-force-plate-flake` (baseline `origin/main`)
- Commit: merged to `main` as PR #11042
- Pull request: #11042; entry DL-#11034. Refs #11034; the issue stays open for the root cause.
- Built: `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py` fails with `_module_identity_report` (resolved file and spec, module names, `sys.meta_path`, related `sys.modules` entries) instead of a bare `AttributeError` when the import resolves to a module without `CombinedForcePlateProcessor`.
- Not reproduced locally (Windows, Python 3.12, CI `PYTHONPATH`): the file alone, after `test_sidekick_extension_overlay.py`, and pairwise after every test that rewires `sys.meta_path` or `shared.*`. Two re-runs of the Linux `unit-test-gate` on this PR passed, and `main` has been green since `c2e6fe878`.
- Suspect: the overlay test's synthetic `force_plate_stitching` (canonical name, no class) combined with `SharedImportAliasFinder._find_canonical_spec` re-inserting itself at `meta_path[0]` and `_CanonicalAliasLoader` writing six alias spellings.
- Next step: none; merged as PR #11042.

---

# Past Handoff — MJX Tracking Plant and Differentiable Rollout (#11039)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11039-plant`
- Branch: `claude/ud-11039-mjx-plant` (baseline `origin/main`)
- Commit: merged to `main` as PR #11040
- Pull request: #11040; entry DL-#11039. Package 2b of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/engines/physics_engines/mujoco/python/motion_matching/mjx_tracking_plant.py` (imports JAX and MJX; no `__init__` imports it): `TrackingPlantSpec` (explicit `root_vertical_index`, `validate_against_model` refuses equality constraints and out-of-range ids), computed-torque control over the reference, sphere-ground and grip-weld wrenches from `jax_contact`, substep/frame/rollout, and `initial_state` that seats the model at static penetration under `model.opt.gravity`.
- Orchestrator changes on review: `from_package` now requires `substeps` and `root_vertical_index` (the heuristics were removed) and raises unconditionally when the package declares a grip closure without `weld_gains`; `build_tracking_plant` works on a copy, so the caller's `model.opt.timestep` is untouched; `rollout` reuses `rollout_diagnostic`.
- Location: the engine package, not `src/shared/python/motion_matching/`, because the shared tree may not import `mujoco` (`test_import_graph_no_mujoco_in_shared_motion_matching`); the JAX contact laws it uses stay shared.
- Measured in `~/.venv-mjx` on the toy model: computed-torque residual 7.1e-15, marker RMS 0.28 mm over 20 frames without contact, rollout gradient against central differences 2.2e-9 relative.
- The evidence prototype is still untouched; package 3 benchmarks the plant on an exported `mjx_package.npz` and decides promotion.
- Next step: none; merged as PR #11040.

---

# Past Handoff — Differentiable JAX Contact Law and Grip Weld (#11037)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11037-contact`
- Branch: `claude/ud-11037-jax-contact` (baseline `origin/main`)
- Commit: merged to `main` as PR #11038
- Pull request: #11038; entry DL-#11037. Package 2a of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/jax_contact.py` (imports JAX; no `__init__` imports it): `sphere_ground_contact_jax` over the shared `ContactParameters`/`GroundPlane` with an explicit tangential-speed floor, and `weld_wrench_jax` with caller-supplied `WeldGains`.
- Measured parity against `contact_law.sphere_ground_contact` on 2000 seeded states: normal 0.0 N, friction 1.0e-11 N maximum difference (float64).
- The tests run only where JAX is installed: `~/.venv-mjx` from `scripts/setup_mjx_env.ps1` now exists on DeskComputer (jax 0.11.1, mujoco 3.13.0); the default env and CI skip them.
- Next step: none; merged as PR #11038.

---

# Past Handoff — JAX-Free Knot Basis and Adam Driver for the MJX Solver (#11032)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11032-knot`
- Branch: `claude/ud-11032-knot-adam` (baseline `origin/main`)
- Commit: merged to `main` as PR #11035
- Pull request: #11035; entry DL-#11032. Package 1 of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/knot_gradient_optimiser.py` (numpy only): `knot_grid` (integer knot count, so no untouched trailing knot as the prototype's float `arange` could give), `knot_basis` (vectorised hat functions, refuses untouched knots), `horizon_knot_mask`, keyword-only `AdamSettings`, and `adam_minimise` over an array namespace `xp` with best-by-objective tracking and non-finite stops.
- Orchestrator rewrite on review: `AdamSettings` was a hand-parsed `*args` initialiser and is now a keyword-only frozen dataclass; `adam_minimise` copied nothing, so freezing `best_x` could freeze the caller's `x0` (now copied, with a test).
- The evidence prototype is untouched; package 2 (MJX plant adapter, needs `scripts/setup_mjx_env`) will reuse this driver.
- Next step: none; merged as PR #11035.

---

# Past Handoff — White-Jerk RTS Kinematic Smoother With Uncertainty (#11029)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11029-rts`
- Branch: `claude/ud-11029-rts-smoother` (baseline `origin/main`)
- Commit: merged to `main` as PR #11031
- Pull request: #11031; entry DL-#11029. Package 5 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/estimation/kinematic_smoother.py` (exact white-jerk discretisation, vectorised Kalman filter plus RTS pass per coordinate, NaN = missing frame, log marginal likelihood, per-coordinate ML noise fit seeded from finite differences) and `smooth_reference_bayesian` in `motion_matching/pipeline/reference.py`. `smooth_reference` is unchanged and pinned bit-for-bit against `butter(4)` + `filtfilt`.
- Orchestrator hardening on review: the initial mean and covariance must be given together and in shape, infinities are refused (only NaN means missing), and result and noise arrays are read-only.
- Validation: see DL-#11029. The 18 local failures in `tests/unit/motion_matching` (event alignment, Rust parity/bench, surrogate training and others) fail identically on clean `origin/main`.
- Next step: none; merged as PR #11031.

---

# Past Handoff — Fitted Sparse Residual for the Physics-Structured Surrogate (#11024)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11024-residual`
- Branch: `claude/ud-11024-sparse-residual` (baseline `origin/main`)
- Commit: merged to `main` as PR #11025
- Pull request: #11025; entry DL-#11024. Package 2 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/neural_motion/surrogates/sparse_residual.py` (numpy-only STLSQ, `CandidateLibrary`, `SparseResidualFit`); `PhysicsStructuredSurrogate` takes an optional fitted residual and refuses without one. The hand-chosen `0.02*tanh(0.5*prior)` residual is deleted, not kept as a fallback.
- Changed existing test: only `test_forward_surrogates_nm07.py::test_physics_structured_surrogate_prior_and_residual`, which asserted the invented constant; it now asserts the refusal and still checks the prior's shape.
- The STLSQ loop is its own helper `_stlsq` so `fit_sparse_residual` stays inside the 100-line architecture budget.
- CI gates fixed on replay: `setflags(write=False)` instead of the three-level `.flags.writeable` chain (lod-quality-gate), and the fail-closed refusal in `residual_correction` names its tracking issue on the same line (stub-introduction guard).
- Validation: see DL-#11024. Known local failures that also fail on clean `origin/main`: `test_artifact_identity_dbc`, `test_nm01_dbc_optimize` (1 each) and `test_swing_surrogate_training` (4).
- Next step: none; merged as PR #11025.

---

# Past Handoff — Parameter Covariance on the Shared Least-Squares Fits (#11021)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11021-covariance`
- Branch: `claude/ud-11021-fit-covariance` (baseline `origin/main`)
- Commit: merged to `main` as PR #11023
- Pull request: #11023; entry DL-#11021. Child of MS-101 #10375 (Board RM#1793). Executed by agy (Gemini 3.8 Flash), reviewed line by line; the orchestrator replaced the duplicated call-site guard with one tested helper (`fitted_uncertainty_or_none`).
- Built: `src/shared/python/estimation/fit_uncertainty.py`; prefix and multiple-shooting fits report parameter uncertainty.
- Validation: see DL-#11021. `residual_regularization.Array` is now a declared `TypeAlias`: the pre-push mypy env has no numpy, and the bare alias failed as `Array?` in any change to `multi_shooting_fit.py`.
- CI `repo-structure-gates` flagged `least_squares_parameter_uncertainty` at 182 lines (budget 100). It is now split into single-purpose helpers (`_validated_jacobian`, `_requested_indices`, `_free_mask`, `_rank_and_condition`, `_free_covariance`, `_marginal_statistics`) with unchanged behaviour: the 81 estimation tests and the wiring tests pass unmodified, and `check_architecture_budget.py` is OK.
- Next step: none; merged as PR #11023.

---

# Past Handoff — Canonical Swing Event Detector (#11014)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11014-events`
- Branch: `claude/ud-11014-swing-events` (baseline `origin/main`)
- Commit: merged to `main` as PR #11020
- Pull request: #11020; entry DL-#11014. Package 1 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/analysis/swing_events.py` (one detector plus `peak_speed_index`); four call sites delegate to it; parity and contract tests added.
- Validation: see DL-#11014 (374 passed; plus 188 passed across reconstruct, statistical-analysis, advanced-analysis and swing-comparison consumers).
- Next step: none; merged as PR #11020. Package 2 is #11024.

---

# Past Handoff — Retract the NM-09/NM-12 DIAGNOSTIC Receipt Claims (#10960)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10960-receipts`
- Branch: `claude/ud-10960-diagnostic-receipts` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11017 (merged); entry DL-#10960. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: the committed NM-09 checkpoint-matrix receipt lists no `qualified_native` models (the three move to `unqualified`) and its limitation says native ODE replay is not verified; the NM-12 turnover receipt promotes no models (the three move to `unmeasured_models`), `end_to_end_verification.status` is `not_verified` and `all_issues_completed` is false. New `tests/unit/neural_motion/test_diagnostic_receipts_10960.py` fails on the old receipts: a DIAGNOSTIC receipt may not qualify or promote anything, and no nested status except `validation.outcome` may read `passed`.
- Validation: `pytest tests/unit/neural_motion/ --deselect test_artifact_audit.py` -> 155 passed (the artifact audit times out locally on a 9.1 GB parquet on disk; CI runs it).
- Next step: CI green, mark ready, arm; then close #10960 with evidence (all code slices already on main).

---

# Past Handoff — Rust Kernel in the Linux Unit-Test Gate (#9411)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-gate-rust`
- Branch: `claude/ud-unit-gate-rust-kernel` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10997 (draft), workflow change shipping alone; entry DL-#9411.
- Built: `ci-standard.yml` `unit-test-gate` installs the pinned Rust toolchain, caches the Cargo registry, maturin-builds only `rust_core/upstream-physics` into the unit-gate venv and runs the fail-closed `import_built_rust_wheels.py upstream_physics` probe before pytest; timeout 25 -> 35 min. The 12 BunkerShot workbench/GUI IDs leave `unit_gate_quarantine.json` (67 -> 55 on main after #11010) because `ball_simulator` raises without the kernel.
- Validation: `pytest tests/ci/test_unit_gate_rust_kernel.py tests/ci/test_ci_infrastructure.py tests/unit/repo_hygiene/test_hygiene_guards_run_in_ci.py` -> 111 passed, 1 skipped (new file fails 3/3 against the old workflow); `tests/ci/test_unit_gate_quarantine_contract.py` -> 11 passed; the 12 retired tests pass locally with the kernel installed. Linux proof is this PR's unit-test-gate run.
- First Linux run (0f01454cf): the kernel built and imported, and none of the 12 retired IDs failed. Six tests that the kernel un-skipped failed. Root `Cargo.lock` clutter came from this workflow and is fixed (the untracked lock is removed after the build). Rust vs enhanced carry and TrackMan windows = #11000, fixed in #11004. The degrees-regression test is updated in #11004. The two #9243 uncertainty claims = #11003 (tier:strong).
- Unblocked: #11004 and #11010 (#11003) merged; replayed onto main d716c698f.
- Next step: unit-test-gate green on the replay, then mark ready and arm via `automerge_guard.py`.

---

# Past Handoff — Restate the #9243 BunkerShot Uncertainty Claims (#11003)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11003`
- Branch: `claude/ud-9243-claims-restated` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11010 (draft); entry DL-#11003. Unblocks UD#10997 (Rust kernel in the unit gate).
- Decision (tier:strong, owner delegated the judgement 2026-09-26): both failures are stale claims, not model defects. (1) The window-band fixture aimed at a fixed 2 m target; the corrected F0 model's grid carries are 0.05-0.66 m (nominal 0.690 m), so the window was empty on all three grids. The fixture now targets the nominal shot's own carry and asserts a non-empty window; the band measures (0.0, 1.141, 2.282) and the not-decorative claim holds. (2) The accelerated-mass share is 0.676 < `DOMINANCE_SHARE` 0.75: it dominates but does not swamp. The test now states that; the threshold is unchanged.
- Validation: `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py` 30 passed with upstream-physics 2.1.3 built from source, and 30 passed with the user-site 2.1.0 wheel.
- Known: an absolute nominal carry of 0.69 m is short for a greenside splash shot; it sits inside the named uncalibrated transfer-efficiency gap (TestHonestyBoundary), not this PR.
- Next step: CI green, mark ready, arm, then replay #10997.

# Past Handoff — Rust Ball-Flight Kernel Base Cd (#11000)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-rust-drag`
- Branch: `claude/ud-rust-drag-base-cd` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11004 (draft); entry DL-#11000. Found by UD#10997 (Rust kernel in the unit gate), which is blocked on this PR and on #11003.
- Built: `ball_simulator.py` passes `GOLF_BALL_DRAG_COEFFICIENT` (0.25) instead of `ball.cd0` (0.21) as the kernel's Reynolds-curve base Cd. New kernel-independent boundary test with a fake `upstream_physics`. `test_raw_degrees_would_collapse_carry` became `test_raw_degrees_are_refused_before_they_collapse_carry` (the launch contract now raises). The exported parity fixture `default_trajectory.json` is left untouched: `src/config/capability_migration.json` pins its byte hash (`test_green_fixture_byte_hashes_unchanged`).
- Validation: see DL-#11000. Local kernel = upstream-physics 2.1.3 built with maturin into a scratch venv; the user-site 2.1.0 wheel on this box is stale and gives different numbers.
- Known: `tests/parity/test_ball_flight_parity.py::test_export_reference_vectors` rewrites the committed, hash-pinned fixture on every run where the kernel is present (pre-existing; follow-up task filed). Restore it with `git checkout -- tests/parity_fixtures` after local runs.
- Next step: CI green, mark ready, arm, then rebase #10997 and fix its untracked root `Cargo.lock` (root-clutter gate).

# Past Handoff — #9411 Unit-Gate Quarantine Burn-Down (Slice 2) 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9411-probe`
- Branch: `claude/ud-9411-unquarantine-passing`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10990 (draft)
- Governing issue: #9411 (DL-#9411)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: retire unit-gate quarantine IDs whose tests pass for the right reason.
- Status: 54 IDs retired (155 → 101). The second Linux run re-quarantined the order-dependent
  `test_install_prompt` worker test and replaced a module `reload` in `test_synthesize_target` (it
  broke the pinocchio_golf facade identity test) with an AST check. The first push retired 90, but Linux CI failed 24 of
  them (GL-less mujoco, missing trimesh, bunker GUI carry, force-plate ownership guard) and the
  child-copy contract flagged the `ai/adapters` edits. `ai/adapters` and the gear-effect sign flip
  were reverted to main (the flip contradicted `test_impact_physics_value_assertions`: toe impact
  gives draw spin, so the two `test_impact_model` gear tests need a convention decision, not a
  sign change); their 11 dependent IDs and the 24 Linux failures are re-quarantined.
- Slices (agy on DeskComputer and OG Laptop, reviewed and trimmed by Claude):
  - validation-contracts: shoulder FK origin (the gear-effect sign flip was reverted).
  - signal-hygiene: launcher test drift.
  - ai-gui-adapters: lazy API routers; theme router prefix (the `ai` adapter shadow edits were
    reverted: Tools child copies may only be deleted, never edited).
  - data-fitting split: `data_fitting.py` 1081 → 479 LOC by removing classes duplicated in
    `_data_fitting_models.py` / `_data_fitting_solvers.py`. The agy `theta_optimal` alias was
    dropped; `test_init_dempster` asserts `coefficients` again (a #4273 blind rename).
  - mujoco-viewer split: coordinator duplicates removed. The agy `sys.modules` mock lookup was
    replaced by patching `_mujoco_viewer_backend` in `test_model_explorer_temp_file.py`.
  - sidekick-drift subset: Simscape C3D embed adapter clears its widget on cleanup; test-only
    retargets. Shadow deletions, the `ai` panel growth and the bootstrap path flip were held
    back for a dedicated #9406 slice.
- Reverted: edits to Tools-owned shadow files (`signal_toolkit` guards, `model_generation`
  gravity and positive-mass checks, `sidekick/data_processing`, `ai/gui/session_manager` and
  `assistant_panel`, `humanoid_character_builder`). Each grew the file's drift from vendor; the
  fixes belong in Tools. 11 IDs that only passed because of them are quarantined again.
- Rejected: deleting `tests/unit/test_safe_eval.py`. Its five failures are real (Tools#5360,
  fixed in Tools#5361); the IDs stay quarantined until the vendor pin moves.

## Validation

- Quarantine rerun after the revert: `pytest -n 6 -o addopts="" --tools-mode vendored <101
candidate IDs>` → 90 passed, 11 failed (re-quarantined).
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 65 IDs in 10 clusters;
  `tests/ci/test_unit_gate_quarantine_contract.py`: 11 passed.
- Data-fitting tests (4 files): 72 passed. Viewer, embed-adapter, launcher and sidekick tests:
  234 passed, 4 failed (all 4 still quarantined).
- `ruff check` / `ruff format --check` on changed files: clean. 29 local repo-structure steps pass.

## Blockers and Risks

- Remaining 65 IDs: inertia API (#6995 design, 10), `safe_eval` (Tools#5360, 5), sidekick
  shadow retirement (#9406), signal_toolkit limits/core, `test_level` (`-O` contract level
  design), size budgets for Tools-owned files, Simscape `ezc3d`.
- Order-dependent failures in the remaining set come from `sys.modules` pollution, not the DbC
  level: `tests/unit/sidekick/standalone/test_session_store.py` makes later tests load the Tools
  copies (gravity 9.81, no guards). The #9406 shadow retirement removes that test's target.
- `check_architecture_budget.py` fails on unmodified main (`validate_inertia_tensor`, 101 lines).
- Rebase onto main after #10988: HANDOFF conflicts (take this file), DL/SPEC keep both rows.

## Next Steps

1. After #10988 merges, rebase, open the draft PR, add the SPEC row keyed by its number.
2. Get `quality-gate` green; mark ready; arm through `scripts/automerge_guard.py`.
3. Comment on #9411 with the remaining clusters and owners.

---

# Past Handoff — Bump `vendor/ud-tools` to Tools Main `3678409fc` (#9411)

- **Branch:** `claude/ud-9411-vendor-bump-safe-eval`; PR #10995, pairs with Tools#5364 (`UD-PAIR`).
- **Change:** gitlink 95ed6b478 → 3678409fc; `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, `reconciliation.py` and its test aligned; converged child copies in `src/shared/python/` on canonical Tools; retired 10 quarantined tests in `scripts/config/unit_gate_quarantine.json` (ratchet 155 -> 145 node IDs); divergence inventory and agent context regenerated.
- **Validation:** companion (143 passed), reconciliation (3 passed), 10 un-quarantined tests passed, child-copy contract (20 passed), quarantine ratchet (145 IDs in 10 clusters), divergence inventory and `agent_context check` clean.
- **Next:** verify pre-push checks, push, run CI Standard, verify quality gate, and arm auto-merge.

---

# Past Handoff — Consolidated Bolt Norm Micro-Optimizations (#10983, #10984)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-bolt-consol`
- Branch: `claude/ud-bolt-consolidated-20260926` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10993 (draft); supersedes #10983 and #10984 (both conflicted on their SPEC rows)
- Issue: none (Bolt performance PRs). No material development-log change — two behaviour-preserving single-expression rewrites.
- Built: `fit_pipeline._quality` computes landmark distances as `sqrt(einsum)` (#10984); `shot._rotation_increment` computes the angular-velocity norm once with `math.sqrt(np.vdot)` and reuses it for the axis (#10983). Inline comments shortened to the 88-column limit; the Bolt journal entry is kept with its date corrected to 2026-09-26.
- Validation: `pytest tests/motion_capture/test_reference_fit_pipeline.py tests/motion_capture/test_reference_fit_preview.py tests/bunkershot3d/solvers/test_shot.py` -> 45 passed; ruff check/format clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`, then close #10983 and #10984 as superseded.

---

# Past Handoff — Main Red on Bandit B314 in the Coverage Gate Checker (#10989)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10989`
- Branch: `fix/10989-coverage-gates-defusedxml` (baseline `84ac37579`)
- Commit: `SELF`
- Pull request: #10994 (draft)
- Issue: #10989 (fleet-main-health: CI Standard red on main); development-log entry DL-#10965
- Cause: #10988 landed `scripts/check_coverage_gates.py` parsing Cobertura XML with `xml.etree.ElementTree.parse`; the push-lane full-tree `bandit -ll -ii` flags B314 and fails `security-scans`, which fails `quality-gate`.
- Fix: parse with `defusedxml.ElementTree` (a core dependency, the convention in `scripts/config/coverage_enforcer.py`); `Element` imported under `TYPE_CHECKING` for the annotation. New test `test_xml_entity_expansion_is_rejected` was red on stdlib ET and is green now.
- Validation: `pytest tests/unit/scripts/test_check_coverage_gates.py` -> 13 passed (new test carries `@pytest.mark.unit` for the suite-marker ratchet); `bandit -ll -ii scripts/check_coverage_gates.py` clean; ruff and mypy clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`; #10989 closes itself when main's next CI Standard run succeeds.

---

# Implementation Handoff — #9406 Sidekick Shadow Retirement 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9406-shadow`
- Branch: `claude/ud-9406-sidekick-shadow-retire`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10991 (draft)
- Governing issue: #9406 (DL-#9406)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: stop UpstreamDrift shipping stale copies of Tools-owned `sidekick` modules, so the
  quarantined tests that demand their absence pass.
- Status: agy (Gemini 3.8 Flash) slice on DeskComputer, reviewed and trimmed by Claude.
  - Deleted, now resolved from `vendor/ud-tools/src/shared/python/sidekick/`: `standalone/`
    (runner, window, session_store, preferences, onboarding), `persistence/`, `__main__.py`,
    `ui/tools_sidebar/default_tabs.py`.
  - `embedded_tool_bootstrap._bootstrap_python_paths`: vendored or explicit Tools paths now
    precede UpstreamDrift's own, as the bootstrap tests require.
  - `sidekick.spec`: Windows icon is the pinned `vendor/ud-tools/assets/tools_icon_hq.ico`.
  - Tests retargeted to the vendor API: pickle is no longer auto-detected (`test_data_io`), C3D
    export needs `C3D_ALLOW_ANY_EXPORT_PATH` (`test_c3d_reader`), and `test_cli` patches the
    launcher factory under every import prefix.
  - 14 quarantine IDs retired (155 → 141).
- Dropped from the agy output: an `architecture_budget.json` exception (the violation predates
  this branch) and an `ai/gui/assistant_panel.py` vendor copy (the backward-compat identity tests
  still fail because vendor imports through the `shared.python` prefix).

## Validation

- `pytest -n 6 -o addopts="" --tools-mode vendored <33 sidekick/packaging/bootstrap IDs>`:
  14 passed (all retired); the rest stay quarantined.
- `UNIT_GATE_QUARANTINE=1 pytest -n 6 --tools-mode vendored tests/unit/sidekick tests/unit/launcher
tests/launchers tests/unit/packaging tests/unit/repo_hygiene tests/integration/sidekick
tests/ui/c3d_viewer`: 67 failed / 2686 passed here vs 65 / 2687 on `b8c27a7d2`; the only new
  ID (`test_embed_adapter::test_create_main_widget_returns_qwidget`) passes alone twice
  (xdist order). The #9406 slice also fixed one main failure.
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 141 IDs.

## Blockers and Risks

- The same launcher/launchers failures appear on main in this Windows environment.
- #9411 (quarantine burn-down) edits the same ledger; whichever lands second takes the union
  of removals.

## Next Steps

1. Open the draft PR, add the SPEC row keyed by its number, get `quality-gate` green.
2. Mark ready and arm through `scripts/automerge_guard.py`.

## Motion Matching Review Packet

Owner-requested review on `docs/motion-matching-board-review`, PR #11083. See
[Board Packet](2026-09-28-motion-matching-board-review.md): 18 draft issue bodies,
main and GS3DX branch evidence, metrics and licensing budgets, and Shadow Tracker
integration/qualification gaps. No implementation issues claimed or closed.
375 focused tests and full Ruff lint/format passed. Board approval and native
qualification are separate next steps; preserve the active #10979 work.

## Current Coordination Limits

Issue leases succeeded. The presence inbox reported incomplete board evidence
(page limit and malformed comments); absence of messages is not evidence that
the repository is unoccupied. This owned isolated worktree preserves all others.
The original root handoff already exceeded its 150-line guideline; unrelated
active sections were preserved. No full-repository pytest/coverage run was made;
364 scoped tests and all configured pre-push checks (including mypy, Bandit and
core/DbC/utils tests) passed. A guessed SPEC test path was absent; no SPEC-test
pass is claimed. Required commit and design-manual governance hooks passed.

## Publication Refresh

Merged origin/main 51a0c1bfa4 into the owned branch without conflicts, retaining
both SPEC rows. Post-merge focused Shadow Tracker tests are being verified;
source/model receipts remain unchanged. Pre-push checks must pass on the merge.

## Additional Hogan Source

Owner-requested DJDYMjmvFwg was downloaded with yt-dlp, including audio and
metadata: 10 Minutes of Ben Hogan (Every Angle Ever Recorded), Sonic Titan Golf,
10:23, 1920x1080 at 60 presentation fps. SHA-256 is recorded in source-catalog.json.
Original archive cadence, individual recording dates and film overlap are unknown;
this compilation must not be treated as synchronized multiview or independent
held-out footage. Local media: Downloads/historical-capture/ben_hogan/.

The new Hogan compilation window 253-267 presentation seconds was processed:
839 frames, 769 detections and 70 explicit missing detections. Receipt and hashes
are committed; source-bound frames and observations remain outside Git. This
window is unreviewed and may cross cuts; 60 presentation fps does not establish
original film timing or independent multiview.

The first post-merge push was stopped because documentation changed while the
security hook was running (no security issues were identified). Finish the
current documentation commit and retry from a clean worktree.

## Main Refresh

Commit `SELF` merges current main at `80dd2f7be`, retaining both SPEC changelog entries. No exploratory GS3DX source or model files changed upstream from the previously tested integration base. The 187 native integration tests and 44 focused Python checks therefore retain their stated scope. Tour matching on the current-main Human model exited naturally with code zero at 07:49:26Z, but mean/max frame position RMS increased to 13.366/41.385 mm, so it has not replaced the earlier selected tour clips. Owner matching and same-runtime/model delta diagnosis remain pending. Draft continuation PR: https://github.com/D-sorganization/UpstreamDrift/pull/11256.

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

### Head and Quiet-Reference Review Checkpoint

The reviewed head residual shares the normalized SO(3) chordal helper with
feet and validates complete head sequences and masks before model setup.
The combined MATLAB R2025b suite passed **267 checks**, zero failed or
incomplete, with natural exit zero at **12:34:19Z**. Two added quiet-reference
guards first failed (19 passed, two failed), then passed in the combined
suite: integer right-ankle inputs preserve other leg fractions, and malformed
neck-unit metadata raises the contract error. Neck prescribed profiles use
**radians**; native initial neck targets use **degrees**. Zero feedforward
does not establish gravity compensation or equilibrium.

The corrected four-frame A/O native comparison retains personalized geometry
and preserves original joint/position outputs exactly when head tracking is
disabled. An earlier private probe lost seven owner geometry values across
an unsaved close/reload boundary and is excluded from selection evidence.
Both tour head candidates failed screening. Owner weight 0.03 passed only
the coarse screen, with left-foot maximum worsening 1.9971 degrees.

The completed **46-frame owner** cold-start, fixed-offset comparison at weight
0.03 improves mean/max head error from 75.517/129.297 to 27.456/54.163 degrees,
and mean/max position RMS from 38.524/81.119 to 26.612/37.606 mm. All native
statuses are one and the process exited naturally at **12:33:07Z**. It does
not replace the selected 16.643/36.865 mm owner clip. Fixed-offset solving
skips calibration's fitted starting pose; keyed warm-start review is ongoing.

Current owner FK contact-sphere clearances are **-64.462 to -48.262 mm**, a
16.199 mm spread, under the validated stored native ground transform. That
audit exited naturally at **12:06:25Z**, without simulation or model save.
The ground transform is parameter-derived, not a KinematicsSolver output.
The assembled 20 ms diagnostic exited naturally at **12:38:27Z**. Scalar
joints at t0 match requested targets (zero translation error, maximum scalar
rotation error 5.43e-9 degrees); all five spherical rotation matrices also match (maximum matrix error 1.34e-10, native postprocess natural exit zero at 12:47:09Z). The contact clearance mismatch survives actual assembly. Initial
left/right normal force is 32.205/24.407 kN, falling to 106.684/0 N at 20 ms;
maximum pelvis displacement is 38.370 mm. This uses constant references,
stored primitive flags, upper tracking enabled and balance correction disabled.
It does not qualify equilibrium or open-loop full-swing dynamics. No force
magnitude or collapse prediction is accepted from FK alone.

See [Sanitized Review Evidence](../research/simscape_matching_reference/head_and_quiet_review_20261002.json) and
[Combined Native Test Inventory](../research/simscape_matching_reference/native_head_and_quiet_integration_tests_20261002.json).
Desktop selections are unchanged. All physical gates remain open. Latest
LaTeX compilation still fails with `Unable to find standard directories for platform`;
the prior PDF does not validate the new sections. Protected CI and review remain required.

### Keyed Seed and Contact Checkpoint

The keyed initial-pose contract and integration passed **275 native MATLAB
R2025b software checks**, with zero failed or incomplete tests and natural
exit zero at **13:08:09Z**. The full output structure, with explicit Human
selection, matched the frozen pre-extension solver at three sampled frames
for each capture. The full seeded comparison completed naturally at
**13:26:50Z**, all 55 tour and 46 owner native statuses one.

At head weight 0.03, tour mean/peak position RMS is **13.349/42.116 mm**,
head post-address mean/peak **27.331/50.315 degrees**; owner position RMS is
**16.596/36.746 mm**, head **27.667/54.064 degrees**. Both missed the
prospective 30% head-improvement criterion. Tour peak position RMS also
exceeds the selected Desktop clip by 1.995 mm. **Neither is selected.**
Further weights 0.06 and 0.10 are a separate prospective experiment with
unchanged acceptance limits against the same-source seeded baseline and
selected clips. Current native loop success does not establish physical
acceptance or anatomical calibration.

The tangent-plane contact diagnostic completed naturally at **13:30:55Z**.
It changes only the ground normal translation by **-64.462 mm**, retaining
the actual assembled pose. Native clearances become **0 to 16.199 mm**.
Initial left/right normal forces are **0/0 N**, final 20 ms forces
**588.495/0 N**, maximum pelvis displacement **1.505 mm**. Upper tracking
remains enabled, balance correction disabled and upper feedforward zero;
this is a unilateral minimum-touch diagnostic, not bilateral equilibrium
or open-loop replay. Both-foot support, capture-consistent geometry/pose,
COM/balance references and gravity-support torques remain required.

See `seeded_head_and_contact_review_20261002.json` in the standalone research
reference directory for sanitized aggregates, direct source inventory and
actual receipts. The updated LaTeX source remains in the same editor; its
built-in compiler still reports `Unable to find standard directories for platform`.
No new rendered PDF is claimed. CI Standard run **37010261028** on published
79d3b97 failed MyPy/core tests; exact-log review is active, protected review
remains required. The Human-default migration is a separate policy change
under source review, not covered by explicit-Human parity. Physical gates
#11156, #11160 and #11173 remain open.

### Human Default and Reviewed Head-Tracked Delivery

New direct whole-body IK calls default to **GS3DX_Human**. Historical Fit
builders, neck-injection harnesses and reproduction examples explicitly
select Fit. This intentional policy change leaves solver mathematics
unchanged. Native RED exposed the wrong default and Human-only seed
rejection; native GREEN passed both policy tests at **13:58:47Z**. The full
reviewed suite passed **277 software/parameter checks**, zero failed or
incomplete, with natural exit zero at **14:07:07Z**. These do not qualify
physical replay or anatomical calibration.

The prospective **head weight 0.10** screen passes both captures. Mean/peak
body RMS is **13.540/40.368 mm tour**, **16.754/36.987 mm owner**. Mean/peak
post-address head error is **10.671/18.918 degrees tour**, **11.318/22.169
degrees owner**; mean head error improves 69.8%/68.8% against their seeded
weight-zero baselines. Each foot remains within the recorded peak limits.
Tour 0.06 fails the peak body-RMS criterion against the prior Desktop clip.

Both-view rendering completed naturally at **14:02:59Z** after preserving
the first attempt's failed provenance-write receipt. All four H.264 MP4s
fully decoded at 800x600, 30 fps, with 55 tour and 46 owner frames. Every
sampled frame was reviewed in ordered contact sheets for both views; no
obvious projected limb/head flips or scene clipping were observed at that
scale. The new selections are saved on the **local user Desktop** in
`Best_Human_Matches_20261002_HeadTracked`, with a matching ZIP and sanitized
provenance/verification. ZIP SHA-256:
`48c34d60a50610879f9f9f12ee90526c9fe091d45c72dc1d1c7744cb93089108`.
Earlier packages are preserved. These are IK videos; the floor is decorative,
head targets cluster-relative, and contact/balance/forward dynamics remain
unqualified. The videos cannot establish player skill.

See `head_weight_followup_review_20261002.json`,
`human_default_policy_review_20261002.json`,
`native_human_policy_integration_tests_20261002.json` and
`desktop_head_tracking_delivery_20261002.json` in the research reference.
The same LaTeX source/editor is updated; built-in compilation remains
unverified with the platform-directory error. CI run 37010261028 failed
dispatch-context coverage/base-ref checks and a whole-repo MyPy baseline;
all 2,563 executed core tests passed. Source and workflow-context review
does not establish passing protected CI. Current main reconciliation,
fresh checks and protected review remain required. Bilateral contact,
capture-consistent balance/gravity support and full native dynamics remain
active requirements; physical gates are open.

### Leg Orientation Contract and Selected Contact Geometry

The analytical leg IK previously accepted a 180-degree orientation mismatch because
its skew residual vanished. Native R2025b TDD reproduced false success (RED: one
pass, five failures, zero incomplete), then passed all six contract tests after
reusing the shared SO(3) chordal residual with a 12x6 Jacobian and separate final
position/orientation bounds. The existing welded-foot native FK test, reachable
IK and trajectory tests also pass. The combined suite passed **287 native
software/parameter checks**, zero failed/incomplete, natural exit zero at
**2026-10-02T14:57:32Z**. This does not qualify Human ankle-to-foot-solid frame
correspondence, anatomical limits, or independent forward dynamics.

Fresh native FK evaluated BOTH promoted head0.10 addresses without simulation
or model save. Per-foot lowest-contact heights differ **29.081 mm tour** and
**15.097 mm owner**; within-foot spreads are below 1.5 mm. With the ground normal
fixed, one plane translation cannot remove this two-foot discrepancy. The bounded
native correction passes both captures: each minimum clearance 0.250 mm, sole spreads below 1.5 mm, foot XY/orientation retained, leg rotations at most 8.731 degrees tour / 4.229 degrees owner. Root/upper coordinates, passive midfoot and fitted geometry are unchanged. The accepted address candidates are staged for assembled-state/contact diagnostics;
equilibrium, gravity torques and forward replay remain open. The selected Desktop
MP4s are unchanged IK visualizations. See the standalone LaTeX reference and
`leg_orientation_contract_review_20261002.json`,
`selected_contact_geometry_review_20261002.json`, and
`native_leg_contact_integration_tests_20261002.json` for equations and receipts.

Latest LaTeX source is maintained in the same editor. Built-in compilation still
fails with `Unable to find standard directories for platform`; no new PDF is
claimed. Changelog duplicates for PR #11256 were consolidated. Subsequent CI
code-quality failed a GitHub fetch because of runner certificate verification;
no certificate validation was disabled and protected review remains required.

### Bilateral Address and Assembled Gravity Diagnostics

Both selected Human address candidates pass their prospectively fixed native
geometry gates. Each foot minimum clearance is 0.250 mm against one plane;
horizontal foot-solid position/orientation, root/trunk/upper coordinates,
passive midfoot coordinates and fitted geometry are retained. Maximum leg
rotation changes are 8.731 degrees tour and 4.229 degrees owner. These are
address corrections, not a new measured whole-swing fit or Desktop promotion.

Separate 20 ms R2025b simulations verify the actual assembled scalar and
spherical pose plus all ten contact clearances. Both feet develop support:
at 20 ms, tour left/right normal force is 443.969/407.322 N, owner
502.848/407.399 N. Maximum pelvis displacement is 1.513/1.482 mm. Both
native processes exited naturally with zero status; geometry/physics were
reapplied before logged-pose FK and the Human binary was not saved. The
tour body mass is model-default 80 kg, owner 104.3 kg; total mechanism
masses include unchanged equipment. Zero initial force reflects 0.25 mm
clearance. Endpoint support does not establish standing equilibrium.

A separate one-second constant-reference hold is registered before results:
each foot >=0.05 BW and summed normal force 0.8-1.2 BW over 0.5-1 s;
whole-run sum peak <=2 BW, pelvis displacement <=5 mm and rotation change
<=1 degree. It retains upper PD tracking, prescribed neck and zero upper
feedforward, with balance correction off. Even a passing hold is not
independent open-loop replay. Actual gravity and native mass define BW.

Read `bilateral_stance_geometry_review_20261002.json` and
`bilateral_assembled_contact_review_20261002.json` alongside the same
standalone LaTeX reference. Its abstract now identifies the latest head0.10
Desktop selections; historical force trials are explicitly attributed and
zero support is not treated as proof of airborne geometry. PDF compilation
remains unavailable. Full-swing physical gates and protected review stay open.

The tour one-second hold is rejected: all three force screens pass, but pelvis displacement 246.153 mm and rotation 14.989 degrees violate the fixed 5 mm / 1 degree bounds. Native process exits naturally with zero at 15:18:18Z; successful execution does not establish physical success. Owner hold and reference/torque/COM diagnosis remain separate active work. See `tour_constant_hold_review_20261002.json`.

The owner one-second hold also rejects the fixed pose limits: force screens
pass but pelvis displacement is 118.109 mm and rotation change 7.866 degrees.
Native exit is naturally zero at 15:23:43Z; final left/right forces are
531.659/499.520 N. Both captures require controller/reference/COM diagnosis;
neither hold qualifies standing stability or independent open-loop motion.
Read `owner_constant_hold_review_20261002.json`. Gates remain unchanged.

Saved native endpoint diagnosis completed naturally in R2025b at 15:54:34Z.
COM horizontal displacement is 230.802 mm (tour) and 110.766 mm (owner);
terminal projected COM lies outside the contact-point hull by 167.030 and
13.899 mm. Native ankle rotations change 12.485/12.476 degrees (tour L/R)
and 4.069/4.525 degrees (owner L/R). This is endpoint motion, not continuous
contact-slip measurement or a causal diagnosis. Initial forces are zero;
there is no initial active support hull. Native workspace and hold scripts
confirm zero leg feedforward and unchanged servo gains. A preliminary probe
rejected its incorrect rigid five-sphere constellation assumption; the
corrected probe measures ankle followers directly and preserves midfoot
articulation. No new simulation or model save occurred. See
`saved_native_hold_diagnosis_review_20261002.json` and the updated LaTeX.
Next: verify Human ankle FK/gain compatibility before a same-stance
balance-enabled hold with the unchanged registered force/drift gates.

Native Human ankle/gain interface checks passed for both exact fitted address
stances (R2025b natural exit zero at16:03:31Z). Native/analytical Jacobian
differences are below2.3e-13 m/degree and gain differences below7.5e-9 degree/m.
Same-stance balance-on holds retain all original gains, zero feedforward,
prescribed neck and fixed force/drift gates. Tour pelvis displacement/rotation
is 20.041 mm / 5.582 degrees;
owner is 14.818 mm / 4.408 degrees.
Hold screens: tour REJECT, owner REJECT.
Improvement is not physical acceptance. See
`human_ankle_gain_interface_review_20261002.json` and
`balance_enabled_hold_comparison_review_20261002.json`. The historical upper-only API hardcoded FitTrack and overwrote starts;
that restriction is resolved by the configured-model implementation and native
lifecycle probes below. Those probes do not qualify full forward dynamics.

### Configured Human Upper-Body Learning and Native Lifecycle Evidence

`gs3dx_track_learn` accepts an explicit already-loaded model with
`initialization="configured"`. This mode preserves the caller workspace and
initial-state configuration, bypasses legacy drive/TrackStart replacements,
and leaves the caller model loaded. Historical FitTrack behavior remains
available through default legacy initialization. References, gains,
feedforward, timing, filter support and enabled tracking are checked before
native dynamics. The returned feedforward is the best profile actually
simulated, with explicit upper-only qualification and first-reference-sample
state provenance. The original Q-filter learning law is retained.

Native R2025b TDD progressed from **1 pass, 13 failed, 13 incomplete** on the
old API to **19 passed, zero failed/incomplete** on the strengthened contracts.
MATLAB marks assertion-aborted RED tests both failed and incomplete. The
valid configured Human probes completed two 0.1 s constant-reference
iterations: tour angle RMS **0.104749 -> 0.021832 degrees**, PD RMS
**5.092613 -> 1.456887 N m**; owner **0.152201 -> 0.022887 degrees**,
**8.803216 -> 1.482412 N m**. Both preserved all **837 workspace values**,
including **620 independently snapshotted parameter values**, caller dirty
state and initial upper angles/rates within the registered 1e-5 bounds.
Neither saved the model binary. Parent mask velocity priority controls the
primitive target; native tracing explains why a child-only override failed.
Three unsuccessful probes and their natural-exit receipts remain recorded.

These are short upper-learning integration checks with balance/feedback,
zero leg feedforward and prescribed neck. They do not qualify full-swing
learning or independent forward dynamics. The physical hold gates still
reject both captures (tour 20.041 mm / 5.582 degrees; owner 14.818 mm / 4.408
degrees, against 5 mm / 1 degree). Separate actual servo and balance torques
and verify settling before treating any empirical leg offset as gravity
feedforward; a rejected transient is not a unique static solution.

See `configured_human_learning_review_20261002.json` and the updated LaTeX
equations/contracts/reproduction account. The combined native regression suite completed naturally at 17:26:58Z:
**311 passed, zero failed/incomplete**, including the five legacy FitTrack
checks. Its 0.3 s historical replay remained within original limits
(rounded angle RMS 0.28 degrees, worst joint 0.49 degrees, PD 16.7 N m).
See `native_configured_learning_integration_tests_20261002.json`; this
short legacy replay does not qualify the configured Human full swing. Built-in LaTeX compilation remains
unavailable: `Unable to find standard directories for platform`. PDF review,
current-head protected CI and full-body replay remain open.

### Saved Leg Servo Effort and Late-Window Motion

Read-only native R2025b analysis completed naturally at **17:47:04Z**.
The 2 ms diagnostic grid reconstructs baseline leg servo feedback from the
saved balance-enabled holds and actual gains, excluding balance correction
and total actuation. Initial reference closure is below 1e-6 degrees.
Whole-run baseline servo RMS is **226.583 N m tour**, **236.384 N m owner**.
The 0.5-1.0 s analysis window still moves: pelvis maximum displacement and
rotation relative to its first window sample are **10.439 mm / 3.147 degrees
tour**, **5.893 mm / 1.283 degrees owner**. Largest per-axis leg-rate RMS is
**7.691 / 7.275 degrees/s**. This is not verified static equilibrium; no
mean torque is promoted to gravity feedforward and the fixed hold gates
remain rejected. Large baseline effort cannot alone establish harmful
cancellation because balance deliberately shifts the servo target.

The reader loaded its own unchanged Human model only for saved block-path
resolution and closed it without saving. No new dynamics/FK ran. Two failed
path-resolution attempts are retained. Raw traces remain private; see
`saved_leg_servo_diagnosis_review_20261002.json` and the same updated LaTeX
source. Inspect nested joint datasets/controller signals to recover actual
net actuation and balance terms before a bounded support-control change.
Current PDF remains unverified because the built-in compiler is unavailable.
CI Standard run37042119516 executed published3187eccd0 and exposed an owned
duplicate #11256 SPEC row; it is consolidated and the duplicate/version
checks pass locally. Fresh current-head protected checks remain required.

### Generalized Leg Effort and Sampled Feedforward

Read-only native reconstruction confirms net leg actuator effort of **45.954 N m RMS
tour** and **56.019 N m RMS owner**, compared with servo-only 226.583/236.384 N m.
Hip logs are generalized XYZ command taps before the follower-frame virtual-work
map; knee/ankle channels are native actuator sensors. Balance corrections offset
much of the baseline servo effort. These moving windows are not static gravity
identification; offsetting terms do not alone prove harmful cancellation.

The leg command now accepts a finite real 12-vector or a 12-by-reference-samples
feedforward table, using the shared time interpolation. Native TDD exposed the
missing interface (one pass/eight failures/four incomplete); the implementation
and existing balance replay then passed **16 tests, zero failed/incomplete** at
18:51:03Z. Constant behavior and the original legacy COM bound are retained.

The configured Human tour profile, zero through 50 ms and ramped to the empirical
seed at 200 ms, completed naturally at 19:06:38Z. It reduces pelvis motion to
**11.012 mm / 2.773 degrees** and passes all three fixed force gates (peak 1.981
bodyweights), but still rejects both fixed pose gates (5 mm/1 degree). The
gain-four/damping-two tour experiment is also rejected: 6.128 mm/1.751 degrees
and peak 2.263 bodyweights. No model binary was saved. Owner profile execution
is separate and not claimed complete here. Prescribed neck and feedback remain;
full independent Human replay is unqualified.

The neck input uses `[LegReferenceTime(:), NeckReference.']`; its radian table
must match the leg grid. The failed initial profile setup and corrected actual
model execution are retained. See the three new aggregate evidence files and
the same maintained LaTeX reference. PDF verification remains blocked by the
built-in compiler's platform-directory error. Earlier head 78918cde passed all
reported checks; these new changes require their own protected checks.

### Owner Session Video Integration — #11268

The owner requested incorporating the capture-session video companion into this active Simscape goal. Reviewed parent #11161, epic #11268, children #11269–#11279, acquisition and frozen protocol. Planning PR #11280 is on main. Registry/export/comparison draft #11172 remains open and conflicting; reuse its authorities. Acquire originals privately, grade clips, freeze camera/landmark/pairing/timing choices before evaluation, compare against the 13-swing envelope with abstention, then compare markerless/Necromatcher and matched Simscape projections at supported L0–L3 levels. Media, locator and per-frame results remain private; public comparison summaries need owner approval and normalized aggregates. No session video observations or pairing verified here yet. LaTeX includes this scope and actual separate owner ramp outcome. Full forward-dynamics qualification remains open.

### Ramped Hold Startup Transient — Native Audit

The reviewed read-only audit exited naturally at 19:37:01Z without loading a model or simulating. First 1-degree crossings are 75.288/86.346 ms (tour/owner), and 5 mm crossings 167.171/161.093 ms, during the delayed empirical ramp. The one-second pose is much closer to the initial pose; both signed vertical ranges include upward and downward motion. This contradicts an unsupported sustained-sag diagnosis. Rotation-vector axes are initial pelvis coordinates, not anatomical pitch. Whole-run maxima remain rejected. The prospective intervention changes only the same bounded torque seed to a 0–50 ms ramp; no result is claimed yet. Its first setup attempt stopped before simulation due to an inherited zero-at-50-ms assertion; the failed script/receipt are preserved and the corrected caller validates the new endpoint. The active serial native trial must be resumed through its existing process handle, never duplicated.

### Supported Posture Checkpoint — Both Captures

The controlled 20–50 ms empirical leg-torque ramp passed all five unchanged one-second hold gates in separate native runs: tour 3.176 mm / 0.836 degrees, peak 1.981 BW (19:54:39Z); owner 2.321 mm / 0.699 degrees, peak 1.730 BW (20:01:28Z). The preceding 0–50 ms tour ramp passed posture but rejected peak 2.000907 BW; rounding cannot change that verdict. Gains, geometry, contacts and model binaries were retained. This is supported posture with feedback and prescribed neck, not identified static gravity or independent full-swing replay.

Next verify physical timing and named solver-ID/unit mappings for the selected head-tracked IK poses, then prepare moving references through existing upper/leg authorities. Do not infer duration from video FPS, silently carry initial stance offsets, invent looser tracking gates, or omit prescribed-neck controls from independent replay. The public album source is located and six served streams are privately acquired; untouched-camera provenance remains unresolved. OpenPose BODY_25 pinned weights now resolve and its registered DNN completed synthetic inference; alternative backends remain unavailable or unconfigured. No video inference or pairing is claimed. Exact pinned Tools 3678409fc51024150ab28970b72e3b468935f345 is now initialized in this worktree for normal public imports.

### Selected Motion and Owner Video Acquisition

The owner-approved public album source recorded by main #11283 is accessible from DeskComputer. Its observed Download all ZIP contains six source-export MP4s that exactly match the previously staged streams by size and SHA-256; raw-camera provenance remains unverified. The pinned OpenPose BODY_25 network loaded and completed synthetic inference at 20:33:36Z; this is runtime evidence, not owner detection accuracy. The selected head-weight-0.1 references have 55 tour / 46 owner samples over 1.8 / 1.5 seconds. All 101 frames passed the existing named native-target and quiet-reference mapper at 20:37:51Z, with no model loaded. A subsequent native audit at 20:42:55Z verified ankle FK against both configured Human models to numerical precision across all frames. All six streams fully decoded with FFmpeg and 9,539 increasing presentation timestamps. A controlled local owner trial confirmed a gap-to-measured transition near the largest root step. Stronger rotational smoothing reduces leg-coordinate jumps but worsens root discontinuity and mean marker RMS, so it is not promoted. Separate gap-weight trials reduce the transition root step to 23.664 / 17.292 mm, with interpolation provenance retained; neither is promoted before full-chain and uncertainty review. No stance correction, velocity reference, full-motion dynamics, video pairing or camera comparison is qualified by those audits.

Next review selected-coordinate continuity and explicit accepted-stance correction before developing motion references. Grade the private streams and freeze camera, pairing, timing and held-out comparison protocols before reporting agreement. Draft #11172 retains ownership of shared capture services; #11268 supplements #11161 rather than replacing marker evidence.

### Selective Missing-Position Contracts

The full owner trial with global missing-position weight 1 is rejected: measured-marker mean/peak RMS increased from 16.754/36.987 mm to 18.791/49.148 mm despite reducing the transition root step. Optional selective missing-position scales now preserve measured targets, offset masks and head/foot terms; default metadata remains unchanged. Native R2025b RED retained eight passing seed tests and exposed ten new failures; GREEN passed ten new and 35 existing tests. Actual-model comparison confirms exact complete-output parity for omitted/empty overrides against the previous solver at source frames 1, 241 and 249. The full 46-frame selective lower-body trial reduces the transition root step from 53.926 to 17.350 mm and the maximum leg step from 38.380 to 34.090 degrees while retaining measured-marker mean/peak RMS at 16.753/36.975 mm. The body-fit screen passes, but remaining 34-degree leg discontinuity, observability, ROM and both-view visual review preclude promotion. Full forward-dynamics qualification remains open.

### Native Branch and Range Diagnostics

A native no-model audit separately measures true adjacent SO(3) rotations and checks the existing implementation ROM table. The largest tour leg-coordinate step is 22.312 degrees at source frames 517–529, with a 24.581-degree left-hip rotation. The selective owner step is 34.090 degrees at 241–249, with a 26.710-degree left-hip rotation (30.025 degrees for the selected owner output). These changes are not solely coordinate wrapping; their agreement with measured swing motion and observation gaps remains under review. The current table flags 12 tour and 13 owner rows, including both owner elbow signs across all 46 samples. This is an implementation-range diagnostic, not a clinical diagnosis: hips/free pelvis lack absolute ranges and wrists lack anatomical neutrals. Native clean joint-centre geometry and offset calibration must be checked before revising conventions or penalties. No candidate is promoted.

### Native Unoffset Geometry and Marker Observability

Native unoffset geometry confirms owner address elbow angles of 27.80/40.40 degrees with LE negative/RE positive, while tour address has 47.69/39.06 degrees with LE positive/RE negative. Unsigned native flexion agrees with the absolute primitive at all reviewed stages; this does not justify globally reversing the ROM signs. Generic joint-centre proxy offsets remain assumptions rather than subject-specific anatomical calibration. The tour largest-step interval has measured thigh-direction changes of 1.568/3.302 degrees and a 9.022-degree pelvis-cluster rotation; axial thigh twist remains unobserved. The corresponding owner interval has a missing-to-measured transition, so an observed pelvis rotation cannot be asserted there. Address-relative pelvis-cluster alignment averages 9.831 degrees for tour and 11.494 degrees for selected owner (peaks 35.017/38.666 degrees); selective owner gap weights scarcely change this. Next review native closed-loop seed families and an optional measured pelvis-orientation constraint, preserving missing masks and unchanged body/head/foot gates. No new video is promoted.

### Owner Seed Feasibility and Rejected Address Fits

Six native owner seed families close the grip loop while preserving the original root, lower-body and neck values to numerical precision. Four families satisfy both elbow rows; one seed has zero flagged rows in the current implementation table at address. Those are feasibility checks, not match acceptance. Re-fitting address with the original offsets gives marker RMS 30.745/40.683 mm for the two bilateral seeds versus 6.061 mm for the original control. The mild seed retains zero flagged rows but degrades marker and head fit; both alternatives are rejected for promotion. Next investigate branch-consistent offset calibration with a frozen calibration/held-out split, and measured pelvis-orientation constraints. No global sign reversal or threshold relaxation is justified.
