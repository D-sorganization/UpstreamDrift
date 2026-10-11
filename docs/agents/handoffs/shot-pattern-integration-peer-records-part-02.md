# Active: Visible Head and Neck (GCV-12) - #11718

- Branch `claude/gcv-12-visible-head`; epic #11706. The anthropometric specs already carry a `Head` body on a three-axis neck at the cervicale; only the native Simscape spec (v1/v2) has none, so the head rides the `Head` body (follows the fitted neck) and falls back to the torso for v1/v2 in MuJoCo only.
- Code: `model_appearance/head.py` (procedural head, face, ears, neck, hair or cap), `mujoco/python/head_visual.py`, native viewer `backends/_head.py`; schema fields `head` and `body_model`. Visual only: no mass, inertia or DOF change (identity tests). Gaze channel `head.orientation_override` plus `drive_visual_head` for OSV-3 #11729.
- Blend balls are capped at 1.1x the adjoining limb radius (test); `*hubto*` bodies are a small `shoulder` part, not torso-sized pads.
- Not done: `body_model: meshes` (rejected until CMB-6 #11657), web `GolferModel.tsx` head. Owner renders in `~/Videos/Parity Audit/forces_and_impact/visuals/head/`.

# Active: Head Gaze Stabilisation - #11729

- Branch `claude/osv-3-head-gaze`; epic #11726. `motion_matching/gaze.py` (eye point, gaze error, schedule, metrics, neck IK), `pipeline/gaze_residual.py` (soft residual, receipt `head_gaze`), `model_appearance/ball.py` (the one ball-at-address function, reuse in GCV-13). `--gaze-weight` defaults to 0 (marker-faithful).
- Reference: `docs/development/full_body_models/HEAD_GAZE_REFERENCE.md` (+ `.tex`). Capture-A driver, weight 10: theta_gaze RMS 21.3 to 0.8 deg, marker RMS 28.0 to 33.8 mm. Gaze axis is calibrated at address (nominal +x is 37 deg off).
- Open: iron and other engines, neck PD in forward dynamics, MyoSuite neck map audit (X is lateral bending, Y flexion; map not changed), published tour head ranges.

# Active: Compliant Bushing Grip Model - #11739

- Branch `claude/osv-7-bushing-grip` (phase 1, Refs #11739 #11726): `src/shared/python/grip_contact/` interface, OpenSim `grip_model="bushing"` (`weld` default unchanged, `contact` raises), `split_method="bushing"`.
- Valid window 0 to 0.94 s with designed damping (zeta 0.7): per-hand peak 142/162 N, internal 151 N (bound 500), 0.16 mm and 0.22 deg. Beyond 0.95 s the committed OpenSim IK candidate is unusable (marker RMS 262 mm, branch switches); the full-window test is a strict xfail.
- Next: phase 2 contact model, other engine parity, full 1.8 s run, `golf_humanoid.osim` builder.
- Next: qualified closure-consistent OpenSim IK input, then re-run the receipt; phase 2 contact model, other engine parity, `golf_humanoid.osim` builder.

# Active: Impact Parameters Panel - GCV-17 (#11723)

- Branch `claude/gcv-17-impact-panel`, stacked on GCV-16 (PR #11769, adapters); epic #11706. Shared card model `impact_parameters/panel_model.py` feeds `GET /api/analysis/impact-parameters`, the PyQt6 dock `src/tools/impact_parameters_panel/` (tile `impact_parameters`) and web `ImpactParametersPanel.tsx`. Parity entry `analysis.impact_parameters`.
- Run series come from `run.simulation_data["clubhead_series"]`, an engine `get_clubhead_series()`, or a MuJoCo run with a `clubhead` body; otherwise the card is unavailable with a reason. Matched-swing ledger ids are not simulation run ids, so that view shows unavailable until candidates carry a club series.
- Next: wire engine runs to record `clubhead_series`; video HUD stamp (GCV-14); Impact Explorer prefill is owned by Tools (#9546).

# Active: ClubheadSeries Engine Adapters - GCV-16 (#11722)

- Branch `claude/gcv-16-clubhead-adapters`; epic #11706. Package `src/shared/python/impact_parameters/adapters/`: one `ClubFaceSpec` (face centre and axes in the club body frame) and one `rigid_body_series` kernel; MuJoCo/MyoSuite, Drake, Pinocchio, OpenSim and Simscape adapters only supply their own FK.
- Impact time: `adapters/impact_time.py::select_impact_index` is the single integration point for the OSV-8/OSV-10 closest-approach rule (not on main yet). `ClubFaceSpec` default centre is the club body origin until GCV-11 face geometry lands.
- Next: GCV-17 impact panel consumes these adapters.

# Active: Centroidal Feasibility Filter V2 — #11669

- Branch `feat/centroidal-filter-v2-11669` (PR #11703); epic #11667. Next: Balance-3 contact-consistent inverse dynamics.

# Active: High-FPS Video Frame Schedule — GCV-14, Epic #11706

- Branch `claude/gcv-14-frame-schedule` (PR #11734): half- and full-speed video variants. Next: impact-time detection from clubhead kinematics.

# Active: Shared Grip Wrench Core — #11713

- Branch `claude/gcv-7-grip-wrench`; epic #11706. Module `src/shared/python/biomechanics/grip_wrench.py` (hand-on-club wrench, midpoint net force and couple, contact-moment/free-torque split, per-hand MOF, club-local frame, `split_method`).
- Simscape fixtures (`tests/unit/engines/simscape/test_force_channels.py`) have only total hand force, LH MOF and midpoint couple, no per-hand forces, so the Simscape cross-check is deferred to GCV-9 (#11715).
- Next: engine adapters populate `ContactReaction.grip_wrench` through `to_contact_reaction_wrench`.

# Active: Finish Feasibility Balance - #11667

- Balance-1 (#11668), branch `feat/finish-feasibility-metrics-11668`: `pipeline/finish_feasibility.py` adds ZMP-inside fraction, friction-cone utilisation, foot slide and yaw pivot, pelvis yaw error and vertical force range to `dynamics.finish_feasibility` (reference and simulation). `finish_feasibility_cli` annotates saved runs.
- Baseline: `evidence/ground_support/finish_feasibility_baseline.json` (driver reference ZMP inside 0.29 over 1.0-1.5 s, iron 0.55). The committed canonical receipts were not edited: their trajectories are not committed and a rerun does not reproduce their spec hash.
- Next: Balance-2 (#11669) centroidal feasibility filter v2; a linearised joint-space QP prototype has not yet reduced the outside fraction.

# Active: Address Foot Progression — #11730 (OSV-4, Epic #11726)

- `motion_matching/foot_progression.py` defines and measures per-foot toe-out; `pipeline/address_feet.py` seeds and refits it; `--foot-progression {off,capture,default}` is opt-in (default `off`, so existing receipts stay valid).
- Findings: the spec's left `hip_rotation` axis is not mirrored (OpenSim's is), and the stock toe marker seeds carry a 12.5 degree yaw bias; see `docs/development/full_body_models/DESIGN_DECISIONS.md` section 15.
- Tour captures measure 15 to 16 degrees lead and 0 to 4 degrees trail, not 20. Owner capture O is private and was not on the implementing host: run `python3 -m scripts.foot_progression_report` with `CAPTURE_DATA_DIR` set.
- Driver result: legacy 41.7/-2.8 vs option 37.0/4.05 (lead/trail, capture 16.4/4.0); lead is pinned by the hip_rotation_l limit. Next: let the address solve trade hip adduction against stance width; regenerate qualified receipts with the option only on request; Drake, Pinocchio and MyoSuite stills need their own renderers.

# Active: MyoFullBody Muscle-Driven Swing, Epic #11642

- Branch `feat/myofullbody-muscle-swing-11642`; children #11643 to #11647 (MFB-6 #11648 is blocked and not started). Code: `src/shared/python/myofullbody/`; scripts `fetch_myofullbody.py`, `run_myofullbody_swing.py`, `render_myofullbody_swing.py`; reference `docs/research/myofullbody_swing/myofullbody_swing.tex`.
- Assets are fetched to `~/.cache/upstreamdrift/myofullbody` (pinned `myo_sim` commit, sha256 per file); never commit them. Arm muscles are MoBL-derived (non-commercial); owner ruled the app non-commercial.
- Hybrid: spec inverse dynamics gives the efforts, MyoFullBody gives only moment arms and force capacity. Receipts are fail-closed (`NOT_QUALIFIED` when reserves exceed 10 % of effort RMS); the current status is `NOT_QUALIFIED` for driver and iron.
- MFB-7 (#11689): share IK, a bounded torque neck (not a muscle) and reserve attribution landed. Driver reserve over effort: arms 65 %, legs 32 %, trunk 53 %, neck 0 %; iron is within 3 points. Still `NOT_QUALIFIED`. Arms have a structural floor of about 55 % (no independent scapula or clavicle joints); trunk and legs are capacity-limited (unlimited capacity: 13-15 %).
- Next: add independent shoulder-girdle joints for the arms, a justified strength model for trunk and legs, an impact-window treatment, and a thoracic muscle model.

# Active: MOSAIC Model-Aware Matching — #11532

- Branch `feat/mosaic-model-aware-matching`; audits closed epic #11421 (change fragment `changes/11532-mosaic-model-aware-multi-trial-matching.md`). Design: `docs/plans/EPIC_MOSAIC_MODEL_AWARE_MATCHING.md`.
- Implemented (`src/shared/python/estimation/mosaic/`): vectorized planar inertial regressor fixture (`inertial.py`, `planar_chain.py`); B-spline kinematic basis; variable-projection inner solve; Levenberg-Marquardt outer solve with continuation; inertial observability subspace and IK init; TVLQR local policy and replay gate; torque template; subject-fit pipeline.
- Docs: methods reference `docs/research/model_aware_matching/model_aware_matching.tex` (+README, structural test); DIME section corrections and restored tabular opening in `docs/research/simscape_matching_reference/simscape_matching_reference.tex`.
- Validation: 36 mosaic tests and 10 doc tests pass; ruff clean; both LaTeX references compile with pdflatex.
- Limits: planar fixture only; no human capture qualified; no engine regressor provider yet.
- Next: MOSAIC-01/02/13 (#11533, #11534, #11545): benchmark freeze, Pinocchio regressor provider, DIME defect remediation (#11547–#11554).

# Active: OpenCap to OpenSim Integration — #11400

- Branch `feat/opencap-import-11409` at SELF; PR #11409. Slice covers #11409 (OpenCap: Import Session Action in PyQt6 and React/Tauri).
- Implemented: `OpenCapImportAction` and `OpenCapImportDialog` in PyQt6 (`src/engines/physics_engines/opensim/python/opencap_import_action.py`), `MainWidget` OpenCap session loading in `opensim_gui.py`, `load_opencap_session` in `OpenSimPhysicsEngine`, `OpenCapImportModal` in React/Tauri (`ui/src/components/opencap/OpenCapImportModal.tsx`), FastAPI routes in `src/api/routes/opencap.py`, and `inspect_opencap_session` in `opencap_session.py`.
- Validation: 21 Python tests passing in `tests/ui/engines/opensim/test_opencap_import_action.py`, `tests/unit/api/test_routes_opencap.py`, and `tests/unit/motion_pipeline/sources/test_opencap_session.py`; 5 Vitest tests passing in `OpenCapImportModal.test.tsx`; 39 feature parity tests passing; ruff lint/format clean.

# Active: Necromatcher Native Fit Delivery — #11240

- Current implementation: `feat/necromatcher-native-fit-11235` (PR #11240) retargeted to `main` following merge of workspace #11239.
- Implemented: source-bound native trajectory fitting with preserved Hermite splines, native video export, research refit controls, ground placement, and effort bindings.
- Validation: 88 fitting tests, 142 workspace unit tests pass; ruff clean; fail-closed validation active.

# Necromatcher Workspace Handoff — #11239 (Merged)

- Current workspace implementation: `feat/necromatcher-workspace-11234` in `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher-workspace`, baseline commit `1b0099a405`. Both hosts create players/swings, import immutable versions, review original images/landmarks and export packages. Real Hogan and Tiger web imagery is verified; Hogan frame navigation reaches frame 750 at source PTS 134.967 s. 31 library/API/native/launcher tests and 10 inventory tests pass; scoped mypy passes eight production files. Further web form tests and fitting remain active.
- Dependency PR #11237 (library) is published as a draft over #11231. CI #11231 has an unrelated `.jules/bolt.md:208` title-case failure inherited from main; capture unit/structure/code checks are green. No CI bypass or unrelated code edits.
- Workspace draft PR [#11239](https://github.com/D-sorganization/UpstreamDrift/pull/11239) is published over #11237 and attached to this chat. Cleanup LoD fix `d762816466` passed native tests, global LoD and push hooks; its LoD CI gate passed. Navigation tests reproduced stale assets on URL recall and endless frame loading after failure; scoped results and retry fix both, and reject a player/swing URL mismatch. Sixteen UI tests pass. CI cycle 2 addresses stale launcher context/atlas views using canonical generators and explicit boundary review. Continue form verification, CI tracking and actual fitting under #11235; keep the full goal active.

- Owner priority/goal: Necromatcher epic #11232; persistent library #11233; Tiger #11226 and Hogan #11229 remain open.
- Owned worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher`, branch `feat/necromatcher-library-11233`; depends on capture foundation PR #11231. Library code is published in draft PR #11237; tile/review work continues separately under #11234.
- Current contracts, TDD evidence, CI status and continuation: [Necromatcher Turnover](../../../docs/development/necromatcher-turnover.md).
- Implemented in progress: immutable source/model/control versions, restart recall, checked capture archives, portable exports and local HTTP API. 43 workspace/API tests passed; scoped library mypy passed; real capture imports verified after reopening. Malformed-profile and export-mutation failures are covered by the passing suite.
- Next: finish library acceptance/import real captures, publish focused PR, add player tiles and web/desktop review, integrate real fitting and downstream simulation/impact/analysis. Do not certify fixed-output coordinator artifacts or uncalibrated source time.

# Historical Player Capture Handoff

## Active: Grip Wrench Extraction in Every Engine (GCV-8 #11714, Epic #11706)

Per-hand `WrenchKind.GRIP` frames, wrench exerted by the hand ON THE CLUB (ADR-0052):

- MuJoCo and MyoSuite scene: `mujoco/python/grip_efc.py` reads the weld `efc_force` rows (`efc_type`, `efc_id`) and maps them to a club wrench with the club-point Jacobians; `MujocoForceTorqueSource` emits it when `grip_weld_l`/`grip_weld_r` exist.
- Full-body MuJoCo, MyoSuite spec and Drake: `solve_with_multipliers` (qacc bitwise unchanged) and `grip_analysis`; the closing (right) hand is the multiplier, the holding (left) hand is club Newton-Euler, see `biomechanics/grip_extraction.py`.
- Pinocchio: `PinocchioForceTorqueSource.grip_from_allocation` gives the net with `split_method="allocation"`, no per-hand values. OpenSim: `grip_analysis()` is unavailable with a reason.
- Open: web/video overlays are GCV-10 (#11716, `analysis.grip_wrench` is a gap). The pinned myo_sim assets are absent on some hosts, so the golfer-scene test skips there.

## Active: Realistic Club Meshes in Every Engine Visual Layer (#11717, #11727)

- One adapter, `src/shared/python/model_appearance/club_head_mesh.py`, builds the head from the Tools parametric builder, with committed STLs in `assets/club_heads/` as fallback. `club_assembly.py` adds the shaft and grip. Never edit `vendor/ud-tools` for this.
- Wired: MuJoCo (appearance and plain visual layers), OpenSim (`club_visuals.py`, `full_body_osim.py`, `tour_matching/club_geometry.py`), Drake and Pinocchio MeshCat exports, MyoSuite via the plain visual layer, and the native OpenSim viewer.
- Open: the web `ClubHead` component (#11717) and the ball visual (#11719) are tracked by gap entry `render.club_head_and_ball`. The head centre of mass is not moved to the mesh centre.
- Saved OpenSim models use bare mesh names; loaders call `club_visuals.register_geometry_path()`.

## Active: Same-Input Cross-Engine Dynamics Parity (Epic #11605)

Same spec, same initial state, same joint torques: every engine must reproduce
the MuJoCo motion. Shared code is `src/shared/python/motion_matching/same_input/`
(`VectorPlant`, closure projection, `StepPolicy` RK4 ZOH integrator with 8
substeps, `InputBundle` `same-input-bundle/v1`, scoring). CLI:
`python -m scripts.same_input_bundle {export,replay,closed-loop} --engine
{mujoco,drake,pinocchio,opensim,myosuite}`. Full-swing results (driver and
7-iron): closed loop at most 7e-10 rad for Drake/Pinocchio, 5e-12 rad for OpenSim, and
bit-identical for MyoSuite. The 50 ms segmented open loop is at most 1.1e-9 rad. The full-horizon open
loop is limited by the swing's physical instability (about 40 1/s), so it stays
within bound to about 0.5 s. Reference: `docs/research/same_input_parity/`.
Do not reintroduce MJCF `fullinertia` (MuJoCo's eigen-solve is inaccurate;
the exporter emits `diaginertia` + `quat`). Open: P-8 Simscape (#11613) on the
MATLAB R2025b host.

## Active: OpenSim Musculoskeletal Swing (#11617, Epic #11605)

- Branch `feat/opensim-musculoskeletal-swing`: `musculoskeletal_*.py` in `src/engines/physics_engines/opensim/python/`, runner `scripts/run_musculoskeletal_swing.py`, receipt under `docs/development/full_body_models/evidence/musculoskeletal/`.
- Static Optimization only, leg muscles only, estimated GRF; valid for the backswing (0-1.08 s). Moco did not converge. Superseded inputs, see phase 2.
- Phase 2 (branch `feat/opensim-msk-spec-driven`, stacked on phase 1): same-input bundles drive a spec-skeleton model with 80 grafted leg muscles, exact contact, per-frame bounded-LSQ static optimisation (`musculoskeletal_pipeline_v2.py`, `scripts/run_musculoskeletal_spec_swing.py`, receipts `receipt_v2_*.json`, notes `docs/research/musculoskeletal_swing/README.md`). ID matches bundle efforts (median 0.01 N m); leg reserve RMS about 35 N m; arms/trunk are torque stand-ins; not qualified. Next: arm/trunk muscle graft from an officially downloaded model (needs approval), left hip rotation capacity, independent open-loop replay.

## Simscape Matching Review (2026-10-02)

GS3DX manual inventory follow-through (2026-10-03 UTC): canonical calculation-registry blocker `UP-D1-gs3dx-whole-body-matching-inventory` records the matching, frames/units, observation/calibration, C2/rate, actuation/contact/replay and media-provenance calculations still needing governed QMD coverage. The separate editable research LaTeX is linked evidence, not an approved canonical manual. The registry remains empty and release-blocked; no publication exemption or generated-artifact edit is introduced.

Original-fit regularity review (2026-10-03 UTC): the separate LaTeX reference and `tangent_c2_checkpoint_20261003.json` contain the equations, source hashes, actual receipts and rejected experiments. Full-clock rate fitting has an independent 2,042-pose audit; adding low/moderate acceleration regularization produces another 2,042 poses. The acceleration fitter saves its results but hangs at shutdown (watchdog 125); a fresh independent native process rechecks all new poses and exits naturally with process/wrapper 0 at 17:47:42 UTC. Stronger smoothing reduces the owner's maximum wrist step from 13.24 to 9.53 degrees, but increases root motion from 6.54 to 9.39 mm and worsens marker error. The tour also trades better shoulder continuity for worse wrist motion. No new best video is promoted. These sampled position checks and finite-window derivative proxies do not qualify continuous motion, contact or independent dynamics.

The current 14-position-target objective includes the clubhead, not the grip centroid. Raw club triads provide an orientation observation with address fixture alignment, while changing relative head/grip orientation prevents an unsupported rigid observation assumption. Ten native input-contract tests verify scalar validation and mask isolation without model loading. Completed club-orientation window evaluation tests baseline, zero, and two positive weights across 536 window poses (four trials each, 75 tour poses and 59 owner poses per trial), establishing exact whole-output parity at zero weight. The original window fitter saves results and hangs at shutdown (watchdog 125); a fresh independent native audit rechecks all 536 new poses and exits naturally with process/wrapper 0 at 18:18:38 UTC. Mean club orientation error improves from 10.30 to 5.92 degrees for Tour (capture_A) and 19.20 to 10.73 degrees for Owner (capture_O), while position target error shifts from 27.242 to 27.248 mm (Tour) and 37.989 to 38.014 mm (Owner), with boundary transition tradeoffs from sparse dt 1/30 s into dense steps.

A subsequent full source-clock club comparison evaluates 3,063 trial poses across both captures (three trials each, with 654 tour poses at 360 Hz and 367 owner poses at 240 Hz per trial), producing 2,042 new poses from positive weights (0.025 and 0.075) and retaining 1,021 rate-control poses without refitting. Observed club triads are available at 618 tour and 345 owner samples; missing frames remain excluded rather than counted as agreement. The original fitting process saved all poses before hanging at shutdown (receipt 125 at 19:13:08 UTC); a separate fresh native R2025b audit verified all 3,063 saved poses, club rotation matrices, observation masks, metrics, and source hashes with maximum physical joint residual <= 9.93e-16, exiting naturally with process/wrapper 0 at 19:15:18 UTC. For private video previews at 30 fps (55 tour display frames, 46 owner display frames), owner weight 0.025 and tour weight 0.075 are selected as a documented tradeoff: tour mean club error improves from 11.859 to 5.476 degrees with derived 14-target position RMS shifting from 16.887 to 16.944 mm and near-unchanged wrist and root steps, while owner weight 0.025 improves mean club error from 21.143 to 19.287 degrees, reduces derived position RMS from 20.800 to 20.762 mm, and reduces maximum wrist step from 13.240 to 13.060 degrees as well as shoulder geodesic and root displacement. The stronger owner weight 0.075 worsens maximum wrist step to 17.536 degrees (shoulder step to 12.534 degrees, root displacement to 7.006 mm) and is not selected for default preview.

Eight Human ellipsoid H.264 MP4 previews are delivered on both Desktops in Simscape_Matches_20261003_Club_Refined_1080p, with a verified ZIP. Native render-to-pose parity, independent full decoding of 404 frames, 1080p/30fps/PTS checks, hashes and sampled visual review pass; the render process exits naturally with code 0 at 19:32:57 UTC. These previews remain inverse-kinematic (IK) visualizations, not forward dynamics (FD). Native actuator discovery on the nominal 80 kg model identifies 25 joint blocks (19 actuated spanning 35 active actuator axes, 6 passive/unactuated blocks) and 99 logged leaf series in an independent 0.02 s pilot with verified neck torque (+/-0.25 N*m) exiting naturally with code 0 at 18:48:51 UTC, but leaves expose state and acceleration quantities rather than measured actuator torque. Native peer and input-graph audits across 267 connection ports and 27 driving converters document native input parameter units in aggregate: 21 scalar converters with Unit 1 (7 revolute axes, 14 universal axes across 7 body joints), 2 neck converters with N*m (adapted torque drive), and 4 spherical vector converters with N\*m (FollowerFrame 3D torque vectors across 4 spherical joints). Input Unit 1 applies no scaling and does not by itself imply a torque-unit error; physical output units are inferred from the destination. This is a parameter inventory, not physical torque measurement. Moving swing, contact, and feedback-off replay across all 35 axes remain outstanding without speculative all-35 measurement or anatomical truth.

The controller XYZ chart remains a separate concern: the owner's left-shoulder peak condition number rises from 21.2 to 54.2 after rate smoothing. Native joint inventory identifies 33 input-torque components plus two prescribed neck axes; only the bounded two-neck pilot has measured primitive-actuator evidence. The complete 35-axis moving/contact/open-loop goal remains open. Six owner session videos are locally downloaded and decoded; exact capture-system trial pairing and anatomical calibration remain unresolved. Existing Human ellipsoid Desktop previews remain kinematic fits. Canonical manual governance still reports two QMD sources, zero registered calculations and an owned inventory blocker. The existing LaTeX editor continues to report its standard-directories platform error; no compiled-PDF claim is made.

Closed-arm regularity and foot review (2026-10-03 UTC): six additional assembled support poses retain all 101 original anchors and 11 prior projections. Native dense acceptance improves to A 1,295/1,297 and O 721/721; remaining two tour failures are midpoints. The returned velocities nevertheless reach right-wrist peaks of 113,025 and 169,742 degrees/s, and the owner has a 99.72-degree wrapped scalar step and 115.22-degree shoulder SO(3) step across consecutive accepted samples. Both curves remain rejected for moving-reference/dynamics promotion; pointwise compatibility is insufficient. A fresh native styled-foot audit shows forefoot-minus-rearfoot direction aligned with sole forward axis (dot above 0.9937), with small address yaw errors. Later right-foot horizontal reversals occur when projection lengths are small; their 3D marker-to-visual angles remain about 24 degrees for tour and 11 degrees for owner despite near-180-degree projected yaw. Preserve the documented ankle-marker-height/flat-sole calibration assumptions; no blind shoe flip is justified. Initial World-Frame uniqueness probe exit-one is retained alongside corrected natural-zero audits. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. Alternate coordinate-target allocation is a private unqualified hypothesis; no physical model/actuator authority change is claimed.

Denser C2/fit review (2026-10-03 UTC): evaluating the SAME candidate curves at 2,018 source and midpoint samples reproduces all 1,010 source samples exactly, but native acceptance is A 1,294/1,297 and O 718/721: three new midpoint assembly failures per capture. Original source samples and all 101 anchors still pass; full-path acceptance is rejected. The new failure diagnostic finds six nearby assembled states with unchanged root translation and all 11 controls passing. A separate native dense fit audit exits naturally zero with original anchor point/RMS parity: mean/peak derived target RMS is 16.98/38.02 mm for A and 21.06/59.60 mm for O; mean/peak directly measured three-back-marker RMS is 26.77/54.66 mm and 38.41/88.12 mm respectively. Valid derived target coverage is 7..14 per frame, with gaps excluded. These mean per-frame RMS scores use different sampling from sparse previews and are not direct model/golfer rankings. No curve or video is promoted to full-path or forward-dynamics acceptance. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX.

Tangent/C2 review (2026-10-03 UTC): fresh native checking passes all 48 saved q/v states; original timeout 124 and tour shutdown 125 remain recorded. Private C2 TDD passes 19 tests. Closed-arm guess correction eliminates all 11 owner anchor mismatches while preserving independent targets bitwise; 6 tour and 5 owner assembly failures remain. Nearby assembled states pass fresh position gates, but a same-target closest-seed retry recovers none of the 11 failures (six controls pass). A controlled projected C2 curve adds eleven upper-limb support poses, preserves original 55/46 fitted anchors, root motion, lower body and unaffected curves, and naturally exits zero with ALL 649 tour and 361 owner source-rate q/v/fresh-state checks accepted. Maximum scalar/spherical target deformation is 0.112/0.173 degrees for A and 0.338/0.343 degrees for O; dense raw-marker RMS is not evaluated. Three axis-contract tests pass after real RED/GREEN correction. The candidate remains unpromoted pending between-state closure, acceleration, full raw-clock coverage, all 35 input torques including neck, contact and independent full-swing forward replay. Parent verifies 104 Desktop video hashes and 5,252 decoded 1080p frames; comparison topologies differ and Human ellipsoids remain the new matching standard. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. PDF compilation remains unavailable due to missing platform standard directories.

Aligned refined-objective review (2026-10-03 UTC): actual sparse refined previews have zero consecutive source-frame transitions; largest root increments are 17.827 mm (A) and 30.166 mm (O) over 1/30 s. Address-prefix controlled eight-pose windows preserve the spine prior. Private parameter TDD has eight RED failures/eight GREEN passes. The original 344-pose fit wrapper returned 125 after completion; a fresh independent saved-pose recheck exited naturally with zero and verified all 344 poses plus exact whole-output zero parity. Small root improvements accompany worse wrist increments; no candidate is promoted. Native counts distinguish 48 position variables, 43 velocity variables, 37 floating IK parameters and 35 requested control axes. See aligned_refined_checkpoint_20261003.json and the maintained LaTeX. Full continuous references, contact, all 35 input torques including the neck, independent replay and PDF qualification remain open.

- **2026-10-03 04:27 UTC observation export update**: The actual owner two-frame hash/forwarding/cache probe exited naturally with success; 39 production contracts pass. The optional source observation contract is caller-bound and declares physical measurement unverified. Six Google Photos originals are verified locally. Refined peer PR11351 previews remain IK candidates. Dense source-seeded reconstruction preserves all anchors but rejects owner frames 253--256 and contains large adjacent rotations. Continuous fitting and full 35-actuator forward dynamics/replay remain open. See the maintained LaTeX and observation_export_contract_checkpoint_20261003.json.

Current reviewed evidence: **267 native checks pass** (12:34:19Z), zero failed/incomplete. Corrected A/O no-head and zero-weight parity is exact on four frames each. The 46-frame owner head candidate improves against its cold-start baseline but remains worse in mean positional RMS than the selected clip; no Desktop replacement. Owner contact clearances remain -64.462 to -48.262 mm; assembled pose verified under the diagnostic settings; stance unqualified. See the current handoff and sanitized head/quiet review evidence.

Active review branch: `feat/simscape-matching-review-main-20261002`; draft PR [#11256](https://github.com/D-sorganization/UpstreamDrift/pull/11256), reviewed continuation commit `SELF`. Source and runtime qualification remain in progress; no merge, complete goal, or physical acceptance is claimed. Reviewed original worker commits `144e81188` through `f2ca443a0` (documentation-only continuation); unpublished original `144e81188` baseline.
See [Current Handoff](../../../docs/development/HANDOFF.md), [Modeling Reference](../../../docs/research/simscape_matching_reference/README.md), and [Refinement Requirements](../../../docs/research/simscape_matching_reference/MATCHING_REFINEMENT.md).

- **Captures and Avatars**: Capture A is the tour reference (360 Hz, 654 frames); Capture O is the owner's optical capture (240 Hz, 367 frames). Golfer skill or athletic ability cannot and must not be inferred from avatar appearance or kinematic fitting distortions.
- **Verified Baseline Desktop Exports**: Four Desktop Fit IK H.264 30 fps clips represent the verified baseline (natural exit 0 both; Capture A mean 12.745748 mm / max 40.208544 mm across 55 frames; Capture O mean 26.558095 mm / max 116.576104 mm across 46 frames; diagnostic worst sampled: Capture O clubhead 375.358 mm at frame 89, Capture A trailElbow 96.517 mm at frame 517). These are legacy GS3DX_Fit IK clips; refined Human clips now exist separately, preserving these baselines.
- **Candidate Rejection**: Owner backward pass (same identity + offsets: mean 32.185137 mm / max 46.052114 mm, receipt `03:57:32.790355Z`, natural exit 0) was rejected because improved peak error does not justify worsened mean error. Fitted marker offset norms cluster 70–100 mm on owner vs tour mostly smaller (hypothesized capture definitions / geometry compensation, not provenance proof).
- **Coordinate Topology (37 vs 39 vs 33)**: Native Human fit has 37 independent coordinates with grip closed; actual joint roles yield 37 independent coordinates (the earlier hardcoded 33 failure reported 39 because of mismatched indices; this distinction matters). Native 3-frame [1, 13, 25] probe: position RMS 13.296 / 12.536 / 12.374 mm, but left foot orientation residual 165.0 / 167.0 / 167.2 deg vs calibrated shoe $R$ (receipt `04:54:21.730288Z`, natural exit 0), proving position does not identify foot orientation.
- **Foot Orientation Calibration**: Shared calibrated foot triads and SVD address mean $R = F_f F_0^T R_z(\text{yaw})$; flat sole at frame 1 is an assumption, not dynamic force evidence (missing frame 1 fails closed). Foot orientation weight 0.1 trial (position 51.263 / 43.243 / 39.399 mm, left foot 36.505 / 17.058 / 13.764 deg, receipt `04:57:13.758186Z`, natural exit 0) was rejected due to degraded positions. Fresh Human calibration probe on the same three frames completed with natural exit 0 at 05:01:00.641030Z: position RMS 1.188/1.195/1.208 mm, both foot angular errors below 0.7 deg. Calibration frames do not qualify full-swing generalization. Full $\text{SO}(3)$ 18-component chordal weight 0.1 is exploratory, not certified (excludes position RMS, gap foot metric NaN).
- **Test Inventory and Visual Adapters**: Pure tests: 102 PASS, 0 skip after final strict validators. Human ellipsoid adapter: 10 native tests pass (natural exit 0); scales longitudinal geometry, artistic width/head/shoe fixed, body mass/inertia remains baseline (not owner 104.3 kg), dynamics unqualified. Future ellipsoid Human-only export policy requires calibrated foot roll/pitch/yaw, private source hash/cache, and independently recorded 14 target RMS and orientation metrics. Selected clips do not constrain head/neck orientation. The optional head candidate is software-tested and compared natively below; anatomy and candidate selection remain unqualified.
- **Stance Stability and Open Gates**: 1 sec resampled / quiet / frozen address / still upper body tests all fail stance (force zero after 0.5s, 1s tilt 101/108/114/124 deg, pelvis far below floor). Historical EXIT 0 was not natural receipt qualification. The damping-times-ten experiment failed (initial 15,513 N heel force; 1s tilt117 deg, pelvis z -1.83m). Original worker f2ca443a0 queued a composed pelvis-level 0.02s check; preserve it, do not duplicate/cancel/save/claim success. Gates #11156, #11160, #11173 remain RED; no full swing open-loop qualification.
- **Reference Documentation**: Earlier 16-page LaTeX revision compiled and visually reviewed; latest full-export additions remain uncompiled (built-in compiler platform directory error); standalone separate canonical QMD manual policy. Current goal includes full refined A/O exports + continued tour and owner physical matching + protected delivery, unfinished.

- **Full Human Exports**: A mean/max frame RMS 13.188/40.121 mm; O 16.767/37.626 mm. Native full solves natural exit 0; all four H.264 30-fps clips completely decoded. Desktop shareable ZIP contains sanitized provenance and same-frame comparisons, excluding raw captures/caches. A has a small mean position tradeoff; O improves both metrics. Native sphere/grip/physics tests pass (15/12/18, zero incomplete); the functional-grip candidate worsened 24 held-out samples and was rejected; physical gates remain red. Latest LaTeX full-export additions remain uncompiled due to editor platform directory error.
- **Fresh Contact Geometry / Import Contract**: Read-only native owner address FK exited naturally at 06:42:14Z; current-plane contact clearances -64.399 to -48.619 mm. No dynamics or simulation initial-target update. Both C3D files specify metres and have no EVENT annotations; timing remains proxy-only. Importer and runtime/dependency provenance hardening is under review.

Main integration passed 187 native MATLAB tests and 44 focused Python tests against model hash `919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f`. The verified Desktop clips retain the separately identified `9a26ee80` model. Current-main anatomical meshes, compiled-budget diagnostics and promoted model files are preserved.

Latest selected package: `Best_Human_Matches_20261002.zip` on Desktop. Tour retains model `9a26ee80` (13.188/40.121 mm); current-model `91997471` owner improves to 16.643/36.865 mm with small mixed foot-orientation changes. Both current-model native exports exited naturally; all four selected clips fully decoded. Forty-two runner tests and five reference checks pass after the function-budget refactor; current protected CI remains pending. See canonical handoff for receipts and exact limits.

### Native Target Binding and Head-Cluster Continuation

The selected shareable clips remain tour 13.188/40.121 mm and owner 16.643/36.865 mm mean/max frame position RMS. Both use Human ellipsoids and IK; full forward dynamics remains unqualified. The controlled common-source/runtime tour comparison produced identical poses with the old and current model, so the newer model binary alone does not explain the changed fit.

Two new pure helpers passed 18 native initialization-mapping tests and 8 marker-cluster tests, with zero failed/incomplete tests. Parent review added actual failing regressions for matrix-shaped poses and a scale-dependent degeneracy cutoff before fixes passed. These counts are separate from the earlier 187-test integration suite. The first native state-target-expression check failed (original loop status 1, mapped -1, scalar discrepancy 16.320 degrees). A compiled diagram update refreshed stale masked start values; a fresh KinematicsSolver then reproduced all 22 joint frames for both captures, with maximum translation error 4.44e-16 m and rotation-matrix error below 9.49e-15. The run exited naturally with code zero at 10:03:13Z. No simulation ran and no model was saved; state-target priorities, controller references and physical initialization remain unqualified.

HeadTop/HeadFront/HeadSide tracks are finite and nondegenerate in all 654 tour and 367 owner frames. Pair-distance variation reaches 6.761% for the owner; cluster axes require body-frame calibration and do not establish anatomical orientation. A conservative optional head-motion candidate is under development and has not replaced selected clips. See `native_helper_review_20261002.json` and `head_track_audit_20261002.json` in the research reference directory. The latest LaTeX source remains uncompiled: the built-in compiler reports `Unable to find standard directories for platform`.

## Active: Tiger 2000 and Ben Hogan

- Branch/worktree: `feat/historical-player-capture-11226`, `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-historical-capture`; commit SELF; PR [#11231](https://github.com/D-sorganization/UpstreamDrift/pull/11231), main merged at 7b98249f0e89f9ba452e40d887f3fa38855a9273; CI pending.
- Tracking: shared runner #11230, Tiger #11226, Hogan #11229. Full reconstruction remains active.
- Continuation and exact checks: [Detailed State](../../../docs/development/HANDOFF.md); [Procedure](../../../docs/development/historical-capture-procedure.md).
- Implemented: source-bound streaming image observations, rational container PTS, missingness, lossless frames and source/frame/model/code hashes.
- Real Results: Tiger 2000/1994 detections; Hogan practice 750/739; higher-resolution Hogan 899/892. MediaPipe 1.0.1; no qualified 3D motion or dynamics.
- Validation: 364 Shadow Tracker tests passed; repository-wide Ruff lint/format and file-size budget passed; scoped mypy passed. TDD red/green evidence is documented.
- Source media/results: `C:/Users/diete/Downloads/historical-capture/`; Tiger video plus audio downloaded. [Source Catalog](../../../docs/development/historical_capture/source-catalog.json).
- Additional Hogan source: `DJDYMjmvFwg.mp4`, 10:23, 1080p60 with audio; cataloged; 253-267 s extracted (839/769 detections), timing/lineage unverified.
- Next: publish focused PR, inspect dense overlays, split continuous swings, bind P1-P10 checkpoints, calibrate cameras/time/body, qualify native parity/replay, then integrate eligible site artifacts.
- Constraints: physical time, event/year lineage and rights remain unverified; missing club landmarks; contact sheets are preliminary review. Keep epics open.
- Ownership: original checkout and other worktrees preserved. Presence inbox unavailable (board page limit/malformed evidence); checked issue leases succeeded.
