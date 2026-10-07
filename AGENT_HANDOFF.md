# Active: MyoFullBody Muscle-Driven Swing, Epic #11642

- Branch `feat/myofullbody-muscle-swing-11642`; children #11643 to #11647 (MFB-6 #11648 is blocked and not started). Code: `src/shared/python/myofullbody/`; scripts `fetch_myofullbody.py`, `run_myofullbody_swing.py`, `render_myofullbody_swing.py`; reference `docs/research/musculoskeletal_swing/myofullbody_swing.tex`.
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


## Active: Native Per-Engine Viewer Export (Epic #11673)

- Landed (PR 1, NV-1..3): `src/shared/python/golf_view_presets/` (face-on, down-the-line, overhead, oblique; MuJoCo, Drake MeshCat, MeshCat (Rx(-90 deg)) and simbody adapters), the `mjv_initGeom` slot fix in `force_glyphs.py` (#11675, subprocess segfault regression test), and `force_overlay/bundle_provider.py` + `mujoco/python/overlay_source.py` (bundle efforts to joint-torque arcs, shared contact law to per-foot GRF at the CoP, weight at the CoM).
- Landed (PR 2, NV-4..6): `src/tools/native_viewer_export/` (CLI `python -m src.tools.native_viewer_export`; Drake and Pinocchio MeshCat via Playwright, OpenSim simbody under xvfb only with 2D projected glyphs, MyoSuite via EGL), the 2x2 Pillow compositor, and a fix to `DrakeMeshcatSink` cone arguments (`MeshcatCone(height, a, b)`). Pitfalls: MeshCat camera = translate `/Cameras/default` is NOT the look-at; set `viewer.controls.target` via `MeshcatPage.look_at`; use the full Playwright Chromium (much faster than the headless shell); run the Drake URDF export in a child (`shared.python` aliases need `src` first on `PYTHONPATH`).
- Open: NV-7 Gepetto (#11680) and NV-8 Simscape stay open.
- Camera convention: Z-up, golfer faces -X, target line -Y; MuJoCo azimuth is the heading of the view direction.

## Active: Same-Input Cross-Engine Dynamics Parity (Epic #11605)
