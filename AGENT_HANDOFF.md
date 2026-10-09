# Active: Shot Pattern Analysis — Resume Checkpoint

- Branch `codex/shot-pattern-analysis`; checkout `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`. Read [Turnover](docs/research/shot_pattern_analysis/TURNOVER.md), [Results](docs/research/shot_pattern_analysis/README.md), and [Issue Register](docs/research/shot_pattern_analysis/issues/README.md) before resuming. This checkpoint preserves work for the user's upcoming connection loss.
- All 24 cells / 720,000 synthetic shots finished. Driver tee and 7-iron/PW approach V2 scoring complete; flight CSVs unchanged, all manifest hashes independently verified. Exact frozen source/native archive and rejected V1 scoring archive are tracked. Commit `1601ac294c` preserves CSV bytes across Git checkouts.
- Corrected central impact, geometric face–loft/lean sensitivity, full launcher tile, custom CLI/GUI, covariance/common-range controls, 72-row statistics and 1080p shareable graphics implemented. Astra independently recomputed statistics (max discrepancy 2.84e-14). No universal curvature benefit; carry and longitudinal spread affect scoring.
- Gemini 3.8 via `agy` filed 18 issues: SPA-001–014 #11840–11853; SPA-015–018 #11862–11865. Personal GitHub identity expressly authorized for this task. All issues stay open until merged acceptance evidence; no PR/push yet.
- Latest fast tool/UI suite: 126 passed. Five-module mypy and focused provenance tests passed; full pre-PR gates remain incomplete. Exact resume commands belong in Turnover. Canonical pre-PR runner is Repository_Management/scripts/pre_pr.py; an initial run exposed duplicate module names during diff mypy. Do not publish as fully validated until resolved or explicitly bounded.
- Limits: illustrative geometric inputs, no measured golfer populations or qualified off-center gear effect. Canonical engineering manual remains blocked-inventory-required. Standalone research LaTeX compilation unverified because the compiler could not download its uncached Tectonic bundle. Preserve unrelated handoff entries below.

# Active: MeshCat Camera Framing - NV-9 #11697

- Branch `claude/nv-9-meshcat-framing-11697`; epic #11673. MeshCat kept the 75 deg three.js default FOV; `MeshcatPage` now sets the shared `golf_view_presets.VIEWER_FOV_Y_RAD` (0.7 rad, OpenSim's value) on entry and raises if the page has no viewer camera. New `golf_view_presets.framing`: `projected_extent` and `fit_distance_m` (15 % margin) with unit tests.
- Open (engine host): verify the Drake/Pinocchio renders at 720p, feed the per-swing body bounding box from engine FK into `fit_distance_m` per view, and judge glyph legibility.

# Active: Feedback Controls Planning — #11784

Documentation branch `docs/feedback-controls-11784`; commit `SELF`. Read [Design](docs/development/feedback_controls/DESIGN.md), [Issue Dependencies](docs/development/feedback_controls/IMPLEMENTATION_PLAN.md) and [Turnover](docs/development/feedback_controls/TURNOVER.md). Goal: all-model six-engine parity culminating in muscle-driven OpenSim and independent excitation replay. Planning only; no new model/video is qualified. Next: F01 inventory/gate freeze, coordinate MOSAIC #11532 and parity #11605.

# Active: Impact-Phase Bushing Grip, OSV-7 #11739

- Branch `claude/osv-7-impact-grip`. The OpenSim bushing grip is driven over 0 to 1.8 s by the OSV-10 fits (`tests/fixtures/club_face/swing_q_*.npz`), mapped by name with `grip_contact.load_coordinate_swing`. Loop closure is below 0.001 mm and hand speeds are within 12 % of the measured wrist markers.
- Deflection is 0.56 / 0.58 mm and 0.84 deg, inside the bounds. Owner decision on PR #11774: the flat 500 N internal-force bound is replaced by `grip_contact.couple_check` (squeeze at most 50 N; transverse pair equals the couple/d Newton-Euler prediction within 5 %, 2 N m noise floor). Both pass for driver and iron (squeeze 3.3 / 3.9 N), so the full-window test is no longer an xfail. The club-welded-to-hand demand is 89 / 87 N m against the realised 100 / 99 N m (bushing amplification). Grip-frame spacing is 80.3 mm, not 76. See DESIGN_DECISIONS.md section 17.
- Next: MuJoCo (soft weld) and Drake (`LinearBushingRollPitchYaw`) bushing parity on the same fixtures. The plan is on #11739.

# Active: Ground Reaction Design Manual Slice - GCV-18 #11724

- Branch `claude/gcv-18-grf-design-manual`. Provisional QMD chapter `manuals/upstreamdrift/chapters/10-ground-reaction.qmd` (GCV-1 equations, conventions, unavailable values, symbols, tests, limitations) and registry blocker `UP-D1-ground-reaction-breakdown-inventory`; the registry stays `blocked-inventory-required` with no calculations. User manual §12.4/§12.7 now state what `grf_metrics.py`/`stability_metrics.py` do not compute and give the correct CoP and free-moment equations.
- Open in #11724: grip wrench (GCV-7) and impact parameters (GCV-15) chapters, ADR-0052, force-overlay and native-export user guides, shared-infrastructure and C4 entries.

# Active: Club Force and Torque Overlays - GCV-10 #11716

- Branch `claude/gcv-10-grip-overlays`; epic #11706. Per-hand, net-at-midpoint and couple glyphs (groups `grip_per_hand`, `grip_net`, `grip_couple`, `grip_mof`) carry the `split_method` label; unavailable quantities are listed as unavailable, never zero (`force_overlay/grip_frame.py`, frame metadata `grip_unavailable_labels`).
- Plots: `biomechanics/grip_plot_model.py` (series), `GET /analysis/grip-wrench`, web `GripWrenchCharts.tsx`, PyQt tile `grip_wrench_plots` (`src/tools/grip_wrench_plots/`). `analysis.grip_wrench` parity gap closed.
- Native export: `--grip` adds the overlay (Drake uses its own KKT multiplier; other viewers show the MuJoCo plant at that engine's pose) and writes `<swing>_<engine>_grip_wrench.json`; `--views hands_closeup --no-grid` tracks the grip midpoint per frame (`view_lookats`, `WorkerJob.lookats`).
- Videos: `~/Videos/Parity Audit/forces_and_impact/club_forces/` (MyoSuite/MuJoCo arena, 1080p60, 1x/0.5x, clean 0.25x impact clip, plot PNG). Follow-up: Drake render did not finish under load (14 frames in about 12 minutes, then a Playwright EPIPE crash); rerun when the host is idle.
- Open: `hands_closeup` azimuth is judged by eye only; Pinocchio gives a net allocation only and OpenSim is unavailable.

# Active: Lift Pack Parity Baseline - LIFT-1 #11741, Epic #11740

- Branch `claude/lift-1-pack-parity-baseline` (PR #11771). Audit package `src/shared/python/lifting/pack_audit/`; run `python3 scripts/lifting/run_pack_parity_baseline.py`; results in `docs/development/lifting/PACK_PARITY_BASELINE.md` and `pack_parity_baseline.json`.
- Finding: same-q FK, feet and total mass agree across the four packs; grips, start poses, limits, phases, bench mass and contacts do not. Nine new pack issues filed (MuJoCo_Models#427-#428, OpenSim_Models#414-#416, Drake_Models#390-#391, Pinocchio_Models#449-#451).
- Next: LIFT-2 shared exercise spec.

# Active: Shared Ground Reaction Core — GCV-1, Epic #11706

- Branch `claude/gcv-1-ground-reaction` (PR #11733): per-foot and net GRF, CoP, free moment and moment about the CoM in `src/shared/python/biomechanics/ground_reaction.py`. Next: GCV-2 (#11708) engine wiring.

# Active: Force Arrow Scaling — GCV-4, Epic #11706

- Branch `claude/gcv-4-arrow-scaling` (PR #11738): body-weight and peak arrow scale modes; overlay route takes a `ForceOverlayQuery` dependency. Next: GCV-5 (#11711) plots, API and web.

# Active: Club-Face Orientation Residual, OSV-10 #11759

- Branch `claude/osv-10-face-roll-refit`, stacked on PR #11752 (OSV-8). The shared marker IK adds `sqrt(w)(R(q)a - R_capture a)` on the `Clubhead` frame from the capture head triad (`motion_matching/club_face_target.py`, `FACE_ORIENTATION_WEIGHT = 3`, `--face-weight 0` = marker-only). It applies in the full-capture IK, the consistency re-solve and the ZMP/shooting re-solves through the existing `axis_targets` path.
- Fixtures: `tests/fixtures/club_face/*` are regenerated only by `python3 -m scripts.regenerate_club_face_fixtures --work RUNS --capture {driver,iron}` (about 40 min each); `provenance.json` holds the hashes, calibrated triad offsets and before/after face events.
- Result (model vs capture, each at its own sub-sample impact `club_face.ball_passage`): the driver impact is 11.3 vs 8.8 deg (was 29.5) and the iron 7.6 vs 8.4 (was 17.9); top and address are within 1.1 deg. Marker RMS: driver IK +2.5 %, iron flat; dynamics improved. The strict xfail is replaced by `test_face_tracks_the_capture_at_address_top_and_impact` (5 deg, every engine's FK). MyoSuite was not installed locally.
- Open: the model's peak clubhead speed falls about 24 ms before the ball (the capture's falls about 3 ms before it). Reference: `docs/research/simscape_matching_reference/simscape_matching_reference.tex` (OSV-10 section).

# Active: Clubface Roll at Address, OSV-8 #11755

- On PR #11752. The matched hand-club chain leaves the club roll to fitted wrist constants, so club `+x` was open 30.7 deg (driver) and 44.8 deg (7-iron) at address in every engine. One shared constant, `ADDRESS_SQUARE_FACE_ROLL_DEG` in `model_appearance/club_assembly.py`, now rolls the head about the shaft (and defines `clubface_vector`); `club.face_roll_deg` in a spec overrides it.
- Tests: `tests/unit/model_appearance/test_clubface_square_at_address.py` (each engine's own FK at the captured address pose, `tests/fixtures/club_face/address_poses.json`). OpenSim STLs and `provenance.json` regenerated.
- Impact: `club_face.impact_frame` (shared `detect_impact_index`, closest-approach fallback, raises if not at the ball). True impact face is +30 deg (driver, t=1.327) and +18 deg (iron, t=1.337) open, while the capture head triad is about +2 deg: the IK/matched trajectory under-rotates the club through release (grows from 4 deg at the top), so a constant roll cannot fix it; needs an IK refit with head-triad weight (test `test_face_is_square_at_impact` is a strict xfail). `tour_matching` club models are unrolled.

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
- Current contracts, TDD evidence, CI status and continuation: [Necromatcher Turnover](docs/development/necromatcher-turnover.md).
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
See [Current Handoff](docs/development/HANDOFF.md), [Modeling Reference](docs/research/simscape_matching_reference/README.md), and [Refinement Requirements](docs/research/simscape_matching_reference/MATCHING_REFINEMENT.md).

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
- Continuation and exact checks: [Detailed State](docs/development/HANDOFF.md); [Procedure](docs/development/historical-capture-procedure.md).
- Implemented: source-bound streaming image observations, rational container PTS, missingness, lossless frames and source/frame/model/code hashes.
- Real Results: Tiger 2000/1994 detections; Hogan practice 750/739; higher-resolution Hogan 899/892. MediaPipe 1.0.1; no qualified 3D motion or dynamics.
- Validation: 364 Shadow Tracker tests passed; repository-wide Ruff lint/format and file-size budget passed; scoped mypy passed. TDD red/green evidence is documented.
- Source media/results: `C:/Users/diete/Downloads/historical-capture/`; Tiger video plus audio downloaded. [Source Catalog](docs/development/historical_capture/source-catalog.json).
- Additional Hogan source: `DJDYMjmvFwg.mp4`, 10:23, 1080p60 with audio; cataloged; 253-267 s extracted (839/769 detections), timing/lineage unverified.
- Next: publish focused PR, inspect dense overlays, split continuous swings, bind P1-P10 checkpoints, calibrate cameras/time/body, qualify native parity/replay, then integrate eligible site artifacts.
- Constraints: physical time, event/year lineage and rights remain unverified; missing club landmarks; contact sheets are preliminary review. Keep epics open.
- Ownership: original checkout and other worktrees preserved. Presence inbox unavailable (board page limit/malformed evidence); checked issue leases succeeded.

# Motion-Matching Handoff

## Active: Fix the PreconditionError Exception-Identity Split at the Shared Contracts Seam (#11175)

- Branch: `bot/contracts-exception-identity`; PR #11175; owner `fleet-orchestrator` (agent: claude).
- `_contracts_exceptions.py` redefined the DbC exception classes in parallel with the public `src.shared.python.contracts` module, so `@precondition` failures raised through `contracts` could not be caught as the shard-imported `PreconditionError` — `tests/unit/motion_matching/test_stability_matrix.py::TestStabilityMatrix::test_get_canonical_test_invalid` stayed red on main after the #11171 restore (`27632bd195`).
- Fix: the shard re-exports the SAME class objects; `ContractEvaluationError` stays shard-local (with message validation). Both import paths remain usable and referentially identical, asserted in the test.
- Verification: red on main HEAD with the extended identity test; 27 passed in the stability-matrix module, 95 passed in shard contract suites; ruff check + format clean on changed files.
- Next: merge #11175 (squash) when CI green; see `DL-#11175`.

## Active: Repair Reduced-Model and Club-Only Product Claims (MMR-11 #11097)

- Branch: `feat/mmr-11-reduced-model-claims-11097`; PR #11130 (Refs, not Closes — partial slice); worktree `/tmp/wtk/mmr-11-reduced-model-claims-11097`; last code commit `46b804bc69`.
- Shipped (code-level fail-closed gates, unit-test verified): driven-triple receipts disqualified at projection time (evidence files untouched, packages NOT regenerated); promoted-package hash + `out_of_plane_rmse_m <= 0.0` integrity gates; club-only fresh continuous replay and labeled inferred posture; matrix all-complete claims fail closed while unresolved cells remain; pendulum planar-floor early rejection gated on the CURRENT target's plane distances with a DbC-validated finite-positive `max_marker_rmse_m`.
- Verification (scoped, at `46b804bc69`): 18 passed (`tests/unit/motion_matching/test_fit_options_dbc.py` + pendulum provider tests), 86 passed (tour_baselines/matrix/UI scoped files); `ruff check` clean on changed files. Full suite/mypy/CI were not run by this slice.
- Open per MMR-11 acceptance: raw-to-package reproduction/regeneration, Board-selected required club-only cells, native qualification runs. See `docs/development/HANDOFF.md` and `DL-#11097`.

## MMR-16 Best-Candidate Viewer Review Fixes (#11102)

Current slice (desktop PyQt): PR [#11132](https://github.com/D-sorganization/UpstreamDrift/pull/11132), branch `feat/mmr-16-best-candidate-viewer-11102`, review fixes at `e86a4d4f1c`, docs at HEAD.

- `rank_candidates` is wired into the matched-swing browser's real list-build path (`_apply_filters`): the auto-selected first row is the best comparable candidate by ascending `whole_marker_rmse_m`; rejected rows stay visible with their verdicts (Codex P1: zero production callers).
- Viewer RMS (`_evaluate_single_frame_residual`, `viewer_frame`, `get_per_engine_rms`) pools per-marker 3D distances (`sqrt(mean(sum(valid_diff**2, axis=-1)))`) matching canonical `tour_metrics.compute_shared_metrics`, so residual summaries, physics scores and captions agree with the ledger's `whole_marker_rmse_m` (Codex P1).
- Frames with zero valid markers claim no worst marker (`FrameResidual.valid_markers`) and are excluded from global-worst selection and `mean_rms_m`; the Worst Residual jump never lands on unobserved placeholder data (Codex P2).
- `TourMatchingViewerWidget.load_file` accepts optional receipt provenance (`candidate_hash`, `engine_name`, `drive_mode`, `is_accepted`, `rejection_reason`), and the browser's `_on_open_tour_matching_viewer` forwards the selected `LedgerRow`'s hash, engine, drive mode and verdict, so captions match the selected receipt and rejected candidates show their failure banner (Codex P1).
- Validation: scoped pytest with `/tmp/ud-venv-11132` (PyQt6 + mujoco, offscreen) — red outcomes recorded pre-fix for every finding; then `tests/unit/tools/test_matched_swing_browser_best_candidate.py tests/unit/tools/test_tour_matching_viewer_residuals.py` 21 passed, targeted viewer suites 29 passed; `ruff check` / `ruff format --check` clean on changed files. Two stale combo pins updated (flattened-RMS value, removed `#d9534f` literal from 70762eb7fe).
- Honest remainder on #11102: web/API surface parity, accessibility and native visual review, and the remaining acceptance checkboxes are NOT exercised by this slice; no "all acceptance criteria" claim is made.
- Next: main-lane rebase/CI of PR #11132 and frontier review.

## Motion Matching Board Review — 2026-09-28

- Branch: `docs/motion-matching-board-review`; owner-requested documentation review, PR #11083.
- Packet: `docs/development/2026-09-28-motion-matching-board-review.md`; 18 draft issue bodies, native evidence matrix, recent GS3DX branch review, anatomical/Home-budget options and historical-video roadmap.
- Main reviewed: `94ade65293`; GS3DX PR #10963 reviewed at `752a94fdd9444f98b5e6f9a39b6e1638ffdb269e`. Active #10979 remains with its owner; no implementation claim or dispatch.
- Validation: 375 focused tests passed; full Ruff lint/format and packet title case passed. No new native fit or MATLAB qualification campaign.
- Next: Board dispositions via packet prompt, deduplicate against existing programs and product-review R05/R06/R12, then claim bounded approved implementation slices.

## Product Review for Expert Panel — 2026-09-28

- Branch: `docs/product-review-20260928`; documentation-only owner request; PR #11080.
- Report: `docs/development/2026-09-28-product-review-board-proposals.md` — 12 prioritized issue briefs, source permalinks, executable counterexamples, dependencies and RunnerDashboard panel prompt.
- Reviewed UpstreamDrift `599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309` and Tools `3678409fc51024150ab28970b72e3b468935f345`; implementation and vendor pin unchanged.
- Validation: 108 focused tests passed; report title case and full Ruff lint passed. Native engine, physical and human usability qualification were not performed.
- Publication: GitHub CLI restored by selecting the valid stored account; live issue/PR reconciliation completed. Implementation issues are proposed only; no lease or claim on another agent's work.
- Next: review R01–R12, deduplicate against current issues/PRs (closed #10960 and open critic PR #10977), and approve bounded implementation slices. No automatic merge, dispatch or scientific approval.

Deferred external validation: six Board plans live in `docs/development/planning/`.
Software remains active; no physical evidence is supplied. See the current
`docs/development/HANDOFF.md` for #10783 enforcement, prior #9546 closure and publication gates.

## Active: Bump `vendor/ud-tools` to Tools Main With K0 + K3a (#10944)

Branch `agy/issue-10944`; PR for #10944 (prereq for #10943).
Advances `vendor/ud-tools` gitlink to `95ed6b47857e9a47211ab1973d02b28beae718bc` (Tools#5348 K0 knowledge-pack engine, Tools#5350 K3a Sidekick Wizards). Synchronizes child copy of `src/shared/python/ai/` (`knowledge/`, `wizards.py`, base adapter, panel tools, assistant panel, RAG deprecation). Pins in `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, and `reconciliation.py` aligned. Divergence inventory regenerated.
Focused verification: `pytest tests/companion/test_companion_catalog.py tests/unit/repo_hygiene/test_tools_child_copy_contract.py tests/unit/ai/test_knowledge_and_wizards.py tests/config/impact_acceptance/test_impact_acceptance_matrix.py tests/unit/tour_baselines/test_reconciliation.py -q -n 0 --no-cov`.

## Markerless Mocap Program (#9063)

- Tools #4706 owns capture and contract schemas; UpstreamDrift #9069 (folded into #9422) owns app orchestration and makes no physical-lab qualification claim.
- ADR-0041 Amendment 1 (#9630, #9619): consumer-side self-calibration fitters live in `src/motion_capture/reconstruct/`; Tools keeps vendor-neutral reference geometry.
- Real-data acceptance is deferred to `docs/development/planning/DV-9619.md`; the rig soak to `DV-9613.md`.

## TB-12: Publish Baseline Guide, Agent Runbooks and End-to-End Acceptance (#10597) [MERGED] / Epic #10584 [CLOSED]

Branch `feat/tb12-baseline-guide-acceptance-10597`; PR [#10814](https://github.com/D-sorganization/UpstreamDrift/pull/10814) merged to `main` (`fbebf5c47`) on 2026-09-24. Parent Epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) (CLOSED); governing issue [#10597](https://github.com/D-sorganization/UpstreamDrift/issues/10597).

- Deliverables:
  - User Guide: `docs/plans/tour_baselines/baseline_guide.md` (Title-Case headings, launcher navigation, Driver/Iron physics, 4-tier model hierarchy, visual semantics, physical 3D RMSE formulas, cryptographic provenance).
  - Agent Runbook: `docs/plans/tour_baselines/agent_runbook.md` (clean-environment reproduction, submodule pin checks, MATLAB R2025b requirements for Simscape, tamper verification).
  - Final Acceptance Report: `docs/plans/tour_baselines/final_acceptance_report.md` (coverage matrix, software integration sign-off vs. ongoing full-body physical qualification under #10363, #10378, #10430, #10440).
  - End-to-End Acceptance Tests: `tests/acceptance/test_tour_baselines_journey.py` (8 acceptance tests covering roster completeness, detail inspectability, where-this-came-from provenance, headless replay, safe cloning, model comparisons, evidence audits, and reproduction commands).
- Scientific Status & Boundaries:
  - Planar double pendulum baselines (`driven_double_pendulum`) qualify within tolerance ($< 15$ mm 3D marker RMSE) with valid cryptographic packages.
  - Planar upper-body golfer models reject due to out-of-plane planar projection residuals ($> 110$ mm normal residual vs 55 mm ceiling).
  - Full-body and spatial models (Simscape, Pinocchio, Drake, OpenSim, MyoSuite) remain governed by their dedicated program issues (#10363, #10378, #10430, #10440) and fail-closed software contracts; no physical qualification or fake convergence is claimed.
- Epic Closure: Concludes all 13 child work packages under Epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) (TB-00 through TB-12). All child packages are merged and closed.
- Next Handoff: Full-body native model qualifications under governing programs (#10363, #10378, #10430, #10440).
- Focused verification: `pytest tests/acceptance/test_tour_baselines_journey.py tests/unit/motion_matching/test_tour_baselines_presenter.py -q -n 0 --no-cov`.

## NM-12: Publish Model Cards, Reproduction Commands and Final Turnover (#10627)

Branch `feat/nm12-model-cards-repro-turnover-10627`; PR against `main`. Parent epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603); governing issue [#10627](https://github.com/D-sorganization/UpstreamDrift/issues/10627).

- Implements `neural_motion.turnover`: publication-ready `ModelReproductionCard` schema (`neural-model-reproduction-card/1.0.0`), `build_reproduction_catalog`, `save_reproduction_catalog`, `load_reproduction_card`, `generate_reproduction_commands`, and `verify_end_to_end_flow`.
- Comprehensive 20-model catalog covering all registered golf models in `list_golf_models()` without omissions or fabricated dynamics.
- Clean-environment CLI reproduction commands for 5 lifecycle phases: `generate`, `train`, `evaluate`, `infer`, `replay`.
- Promotion verdicts partitioned into `PROMOTED` (planar/constrained with favorable 12.3x speedups), `RESEARCH_ONLY` (kinematic reconstruction without torque supervision), `BLOCKED_PREREQUISITE` (uninstalled full-body runtimes with fail-closed honesty), and `REFERENCE_ONLY` (standard catalog URDFs).
- 5-step automated user flow verification (`DATASET_REGISTRATION` -> `TRAINING_RUN` -> `CHECKPOINT_SELECTION` -> `OBSERVED_MOTION_MATCHING` -> `PHYSICAL_REPLAY`).
- Evidence receipt: `docs/plans/neural_motion_matching/evidence/nm12_model_cards_turnover_receipt.json`.
- Plan report: `docs/plans/neural_motion_matching/model_cards_reproduction_turnover.md`.
- Concludes and completes Epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603) (all NM-00 through NM-12 deliverables fulfilled).
- Focused verification: `python -m pytest tests/unit/neural_motion/test_turnover_nm12.py -q -n 0 --no-cov --timeout=60` (16 passed in 7.6s).

## NM-11: Integrate Model-Specific Training and Inference With Existing Tools (#10626)

Branch `feat/nm11-training-inference-tools-10626`; PR against `main`. Parent epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603); governing issue [#10626](https://github.com/D-sorganization/UpstreamDrift/issues/10626).

- Multi-runner framework dispatch in `src/shared/python/training/runtime/runner_registry.py` registering both classical and neural runners (`neural_motion`).
- `NeuralMotionRunner` (`src/shared/python/training/runtime/adapters/neural_motion.py`) conforming to `TrainingJobRunner`, emitting `ProgressSink` events, respecting cooperative cancellation via `CancelToken`, and generating verifiable `ModelCheckpointCard` outputs.
- Portable training job packaging (`src/shared/python/training/portable.py`): `export_job_package` and `import_job_package` with SHA-256 manifest verification and zip-slip directory traversal defenses.
- GUI training controller integration (`src/tools/training_controller/`): exposed view models `ModelTopologyItem` and `DatasetSchemaItem` dynamically queryable via `TrainingDashboardController`.
- Motion Matching GUI integration (`src/tools/motion_matching/gui.py`): added `Neural Motion Matching` controls (Mode: Classical Only / Neural Preview / Neural Verified, model selection, classical fallback checkbox) and results inspection badges (`neural_status_badge`, `metric_neural_confidence`, `metric_time_breakdown`).
- Focused verification: `pytest tests/unit/training/test_neural_motion_runner_nm11.py tests/unit/training/test_portable_packaging_nm11.py tests/unit/training/test_view_model_nm11.py tests/tools/motion_matching/test_motion_matching_gui.py -q -n 0 --no-cov`.

## NM-10: Benchmark Accepted-Match Speed, Data Efficiency and Break-Even (#10625)

Branch `feat/nm10-benchmark-speed-10625`; PR against `main`. Parent epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603); governing issue [#10625](https://github.com/D-sorganization/UpstreamDrift/issues/10625).

- Implements `neural_motion.benchmark`: comparative 5-method benchmarking (`cold_solver`, `retrieval_solver`, `existing_neural`, `forward_surrogate_polish`, `learned_proposal_polish`).
- Latency decomposition accounts for preprocessing, native initialization, proposal inference, rejected attempts, native polish, verification replay, and I/O.
- Acceptance rate calculation strictly includes rejected attempts in denominator ($A_{\text{rate}} = \frac{N_{\text{acc}}}{N_{\text{acc}} + N_{\text{rej}}}$).
- Truthful break-even calculation: if savings $\le 0$, explicitly reports `has_break_even=False` with no queries.
- Frozen promotion gates enforce $\ge 2\times$ median speedup, non-worse p95 latency, and non-worse accepted quality rate; failing models marked as `RESEARCH_ONLY`.
- Data efficiency trajectory confirms active acquisition superiority (1.48x sample multiplier over random).
- Evidence receipt: `docs/plans/neural_motion_matching/evidence/nm10_benchmark_speed_efficiency_receipt.json` (labeled `DIAGNOSTIC` under #10960 / #11146; historical unmeasured baselines).
- Markdown report: `docs/plans/neural_motion_matching/benchmark_accepted_speed.md`.
- Focused verification: `python -m pytest tests/unit/neural_motion/test_benchmark_nm10.py -q -n 0 --no-cov --timeout=60` (13 passed in 0.58s).

## NM-09: Train and Qualify a Checkpoint for Every Physical Model (#10624)

Branch `feat/nm09-checkpoint-matrix-10624`; PR against `main`. Parent epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603); governing issue [#10624](https://github.com/D-sorganization/UpstreamDrift/issues/10624).

- Implements `neural_motion.matrix`: `NeuralCheckpointMatrix` and `ModelCheckpointCard` evaluating all 20 models across the #10585 roster.
- Models with variable dimensions ($n_q, n_v, n_u, n_c$) strictly match registered `GolfModelIdentity`.
- Precondition `assert_model_checkpoint_compatible` enforces that no model can accidentally load another's card.
- Native ODE forward replay verified on reduced planar mechanisms (`driven_double_pendulum`, `driven_triple_pendulum`) and constrained upper body (`constrained_upper_body_golfer`, loop constraint violation $< 10^{-4}$).
- Reconstruction models provide kinematic proposals without fabricated torques.
- Missing runtimes (Simscape/MATLAB R2025b, Drake, OpenSim, MyoSuite) fail closed with explicit named blockers.
- Deterministic cryptographic hash chain (`dataset_hash` -> `split_hash` -> `weight_digest` -> `checkpoint_hash` -> `matrix_digest`).
- Focused verification: `python -m pytest tests/unit/neural_motion/test_checkpoint_matrix_nm09.py -q -n 0 --no-cov --timeout=60`.

## Main Red Fix: Docs-Consistency Cross-Repo Paths (#10743)

Branch `fix/10743-docs-consistency-cross-repo` teaches `scripts/check_agent_docs_consistency.py` that a backticked path directly after a sibling repo name (`_SIBLING_REPOS`) is cross-repo. Durable upstream fix (full URL in the synced block) belongs to Repository_Management #1689.

## Hip-Calibrated Receipt Provenance (#10271)

Branch `fix/10271-hipcal-provenance`; PR [#10722](https://github.com/D-sorganization/UpstreamDrift/pull/10722). Restores receipt provenance chain (`receipt-provenance-chain/1`) on `anthro_driver`/`anthro_iron` (software-contract; native regen deferred on low disk).

## GolfSwingVisualizer MATLAB Consolidation (#9225)

PR [#10715](https://github.com/D-sorganization/UpstreamDrift/pull/10715) (open; implementation commit `f33b40c9c`); branch
`bot/issue-9225-golfviz-consolidation` off `origin/main` @ `901b2de5e`;
worktree `C:/Users/diete/Repositories/UpstreamDrift-worktrees/local-9225`;
governing issue [#9225](https://github.com/D-sorganization/UpstreamDrift/issues/9225)
(source:assessment P2, DRY PP1); DL-#9225.
The four per-tree `GolfSwingVisualizer.m` copies (1180–1183 lines each; 323
shared 8-line blocks between the worst pair) are deleted and replaced by one
fleet-shared package class at
`src/engines/Simscape_Multibody_Models/shared/+golfviz/GolfSwingVisualizer.m`
(the 2D-variant superset, restoring the `rng(1)` reproducible ground texture the
3D copies had silently lost). Both `launch_gui.m` launchers add the shared
directory to the MATLAB path and fail loudly if it is missing; all four call
sites use `golfviz.GolfSwingVisualizer(...)`. Verified headless in MATLAB
R2025b: package resolution from a bare `addpath`, class parse, DbC
precondition, and both launchers' path setups. Honest gap: the DRY duplication
ratchet is Python-scoped and does not fingerprint `.m` files, so no baseline
drop is claimed (follow-up). Full GUI launch/render is not exercised (headless
GUI rule).
Next step: Confirm `quality-gate` green on the PR; squash auto-merge lands.

## Tour Baselines: Exact-Capture Matching & Fail-Closed Qualification (#10829, #10830, #10831, #10799, #10800, #10844, #10849) [MERGED]

PRs [#10837](https://github.com/D-sorganization/UpstreamDrift/pull/10837), [#10841](https://github.com/D-sorganization/UpstreamDrift/pull/10841), [#10844](https://github.com/D-sorganization/UpstreamDrift/pull/10844), and [#10853](https://github.com/D-sorganization/UpstreamDrift/pull/10853) merged to `main` on 2026-09-24.

- Exact capture match enforced across `TourBaselinesPresenter` and baseline package generation; unmatching session captures fail closed with `has_exact_capture_match=False` rather than silently falling back to unmatching session captures.
- Fail-closed legacy evidence: eliminated auto-migration from `IndependentBaselineQualifier.qualify()`; explicit migration sets `UNVERIFIED` and `has_native_replay=False` until native evidence is regenerated.
- Dynamics inertia digest & cache refresh: `compute_pendulum_inertia_hash` digests actual cached physical properties consumed during rollout integration; `create_calibrated_double_pendulum_dynamics` passes calibrated lengths to constructor before parameter caching; `refresh_cache()` synchronizes runtime parameter mutations (#10849).
- Surrogate validation quaternion norm optimization: vectorized norm calculation via `np.einsum` (#10844).
- Focused verification: `pytest tests/unit/tour_baselines/ tests/unit/motion_matching/test_tour_baselines_presenter.py -q -n 0 --no-cov`.

## Required Before Continuing

- Read `AGENTS.md`, `CLAUDE.md`, and `docs/development/DEVELOPMENT_LOG.md`.
- Epic #10584 and all child packages TB-00 through TB-12 are closed. Full-body models remain under governing programs (#10363, #10378, #10430, #10440).
- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only the `manuals/upstreamdrift` QMD source and run `python3 -m scripts.check_design_manual_governance` for governed changes.
- Update this handoff, the development-log entry, and exactly one `SPEC.md` change-log row for every substantive PR.

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

Simscape #11256 continuation: keyed seed software275 GREEN; full seeded A/O head0.03 candidates miss the 30% screen and remain unselected. Minimum-touch ground registration removes initial penetration but bilateral support remains unqualified. See the current calculation-level research reference and sanitized seeded/contact receipts; Human-default migration and protected CI repair remain active.

Simscape #11256 latest: Human-default native TDD2 GREEN and combined277 GREEN; head0.10 candidates pass prospective body/head/foot screens and all four H264 views are reviewed, fully decoded and saved on the local user Desktop in Best_Human_Matches_20261002_HeadTracked.zip. Earlier selections preserved. Standalone LaTeX updated but compiler unavailable. See current research reference/evidence; protected CI/current-main reconciliation and bilateral-contact/gravity-support/full forward replay remain active.

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

Simscape #11256: both controlled Human 20–50 ms startup ramps pass all five fixed one-second hold gates: tour 3.176 mm / 0.836 deg, owner 2.321 mm / 0.699 deg. Native receipts at 19:54:39Z / 20:01:28Z. Earlier 0–50 ms tour rejected peak 2.000907 BW. Current feedback/prescribed-neck scope remains explicit; no full-swing or independent forward qualification. Actual equations, failed receipts and aggregate evidence are in the maintained LaTeX reference. Selected-reference metadata and full-motion handoff are next; private video epic #11268 is incorporated. Prior b21836863 checks passed; new changes need their own protected checks.

Simscape #11256: the native reader and independent logger trace establish actual net leg effort of 45.954/56.019 N m RMS. Sampled feedforward TDD and balance regression passed 16 tests with zero failures/incomplete. The actual Human tour ramp improves motion to 11.012 mm/2.773 degrees; all three force gates pass, but both fixed pose gates reject. The gain-four diagnostic also rejects peak force. Owner execution is separate; no new best video or independent replay is claimed. Source, the maintained LaTeX reference and aggregate evidence agree. Earlier head 78918cde CI passed; new-head checks, PDF review and full physical qualification remain open.

Local acquisition update (2026-10-03 UTC): the complete six-video source export is now also on the operator local machine; archive and all MP4 hashes verified. Indoor surface-marker placements were reviewed, with exact labels, anatomical offsets, trial pairing and physical clock still unverified. Private source-mask TDD passed 13 baseline and nine new tests after RED; actual owner importer default data/callback parity and masked callbacks passed. No production integration, candidate promotion or full forward-dynamics qualification. The existing LaTeX editor remains open; compilation is unverified because the built-in compiler cannot find platform standard directories. See `local_video_marker_placement_checkpoint_20261003.json` and the reference acquisition subsection.

Source observation update (2026-10-03 UTC): optional logical masks now enter the canonical converter before importer callbacks. Native production integration passed 13 existing and nine new contract tests and exact tour/owner default data, callback, joint-centre and head-target parity. Full owner refit on the same 627 source-available targets changed mean RMS 16.677 to 16.658 mm and peak 36.987 to 37.050 mm; largest root increment increased 53.926 to 55.295 mm and 13 signed implementation-ROM rows remain flagged. No candidate/video promotion or dynamics qualification. See `source_mask_full_owner_checkpoint_20261003.json` and the maintained LaTeX reference. Existing unmasked caches/media cannot be relabelled as source-mask-qualified.

Native neck recovery checkpoint (2026-10-03 UTC): the owner one-second supported hold recovered Rx/Ry computed primitive actuation torques through two verified N*m PS–Simulink logging converters. Sample RMS/peak values are 5.950843/9.530841 and 3.574840/5.157831 N*m (1,483 finite samples per primitive). All five original hold gates still pass, with the same 2.321210 mm / 0.699395 degree pelvis motion as the preserved control. Sensing-only logs lacked these channels; connected outputs establish recovery. Neck motion remains prescribed and upper/balance feedback remains active. Four temporary blocks were discarded without saving the model. This is not full-swing torque recovery or independent replay. See `owner_neck_supported_hold_checkpoint_20261003.json` and the maintained LaTeX reference.

Tour neck recovery repetition (2026-10-03 UTC): natural R2025b completion at 02:04:26Z recovered Rx/Ry computed actuation torques (sample RMS/peak 3.827226/7.366211 and 2.061589/3.959804 N\*m; 1,587 finite samples per primitive). All five unchanged supported-hold gates pass; full-precision pelvis displacement/rotation match the preserved control exactly (3.175803 mm / 0.836223 degrees). Both captures now verify the connected prescribed-neck logging route. Neck motion and upper/balance feedback remain present; full-swing recovery and independent replay remain unqualified. See `tour_neck_supported_hold_checkpoint_20261003.json` and the maintained LaTeX reference.

Human moving-reference binding update (2026-10-03 UTC): the upper-body reference helper now accepts an explicit native joint-variable table and resolves block-path/primitive keys through the shared key authority. Human never reuses Fit numbered IDs; legacy explicit Fit retains its unbound interface. Seven contract tests pass. Production native integration completed naturally at 02:21:45Z for all 55 tour and 46 owner selected samples: 21 upper actuator-coordinate references at 30 Hz match the preserved numerical filter/rate/start calculations exactly after verified native block/ID/unit binding. Model hash unchanged, no simulation or save. Moving-start/contact reconciliation and full-swing torque recovery/replay remain open. See `upper_reference_binding_checkpoint_20261003.json` and the maintained LaTeX reference.

Closed-chain reference update (2026-10-03 UTC): independent upper-angle filtering returns native status -1 for all 55 tour and 46 owner requested poses (model constraints satisfied, some targets missed). Native-adjusted poses pass a complete target recheck. The explicit `filter_reference=false` conversion preserves checked sample geometry; eight contract tests and production native roundtrips pass for all selected and adjusted A/O poses. Dependent-arm trials and their adverse changes remain unpromoted. Tangent velocities, between-sample interpolation, moving contact initialization and complete torque coverage still need verification; saved holds have neck/leg datasets but no upper torque buses. See `reference_filter_closure_checkpoint_20261003.json` and the maintained LaTeX reference. No dynamics or new best-video qualification is claimed.

Native moving-reference audit (2026-10-03 UTC): all 101 original sampled poses accept zero native velocity. All 101 differentiated moving requests and all 99 complete-coordinate chart midpoint requests return status -1, missing some complete native targets. A separate nonzero root-rotation/frame-output control verifies follower-resolved spherical velocity. The native results do not qualify an accepted continuous curve, unchanged returned moving poses, measured player velocities or full dynamics. Preserve these adverse outcomes and require constraint-aware trajectory construction, derivative consistency and native initialization before complete torque recovery/replay. See `reference_velocity_midpoint_checkpoint_20261003.json` and the maintained LaTeX calculations. No model save, simulation or new best-video promotion occurred.

Root-availability review (2026-10-03 UTC): saved-array analysis reproduces the legacy owner-window root-step maxima and associates the largest approximately 40 mm increment with restoration of seven derived lower-body targets. Waist-marker gaps can propagate through the pelvis-axis joint-centre estimator despite visible raw leg markers. This is association, not causation. The current refined preview uses a different head-axis, back-marker, spine/scapula and gap-weight objective; the legacy-window maximum does not establish its root error. The private translation-option TDD completed naturally with eight RED failures and eight GREEN passes, no incomplete tests and no model/simulation. The aligned refined port remains source-only and unpromoted. See root_availability_checkpoint_20261003.json and the maintained LaTeX calculations. Full moving references, contact, all 35 actuators, independent replay and PDF qualification remain open.

### Selected Club Continuity Update (2026-10-03)

PR #11256 merged at `7db86ff173653a235ea63bb4af347af0a8a5ab83`. The eight verified Human ellipsoid MP4s remain on both Desktops in `Simscape_Matches_20261003_Club_Refined_1080p`; these are IK reconstructions. The source reference now presents this current selection before historical experiments.

The selected club fits were tested at all anchors and midpoints using independent native position, velocity and fresh 91-state checks. Sequential seeding accepted 1,306/1,307 tour states and 732/733 owner states. A targeted diagnostic recovered the owner's rejected original anchor from its saved closed-arm seed; the previous accepted pose reproduced an alternate branch. A complete anchor-seeded rerun exited naturally at 20:31:23 UTC and accepted all 733 owner states (367 anchors and 366 midpoints), while the tour still rejected midpoint 998 at 1.3847222 s. All original anchors matched within 1.73e-15 physical residual. The peak accepted dependent angular rates remain approximately 1,639 and 4,118 deg/s; sampled-state acceptance does not qualify between-sample branch continuity, dependent acceleration, anatomy, contact or forward dynamics.

The private production integration candidate passed 17 native pre-model contract cases (parent RED, candidate GREEN, 51 recorded checks) at 20:20:58 UTC. Numerical whole-output parity and positive-fit validation remain separate pending gates. Original failed runner receipts are preserved. A custom vector measurement block built successfully, but the eight-axis instrumented simulation hit the Home license's 1,000-nonvirtual-block limit. No all-35 measured torque or moving feedback-off replay claim follows from these tests. Original physical model files were not saved.

See the separate editable research reference and the extended aggregate `tangent_c2_checkpoint_20261003.json` for scope and provenance. The built-in LaTeX compiler still fails with `Unable to find standard directories for platform`; PDF compilation/page review are unverified. Canonical calculation inventory and governed manual release remain blocked; this update grants neither a release exemption nor scientific approval.

### Impact Parameters Package (GCV-15, 2026-10-07)

`src/shared/python/impact_parameters/` extracts speed, attack angle, club path, face angle, face-to-path, dynamic and spin loft, swing plane and low point relative to an explicit `TargetFrame` (default recorded: Z-up, target -Y, ADR-0041). Definitions are in the package docstring for design-manual transfer via GCV-18. Tools delivery and D-plane are reached only through the fail-closed `tools_gateway.py` and agree with the UD definitions within 0.01 deg. Open: launch direction has no Tools provider; toe/high needs GCV-11 face geometry and GCV-13 ball; smash factor needs an impact model with calibration status; `rate_of_closure` `delivery_at` is not called directly.
