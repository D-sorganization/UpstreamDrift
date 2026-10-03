# Active: OpenCap to OpenSim Integration — #11400

- Branch `claude/opencap-upstreamdrift-integration-e7ev4n` at SELF; PR #11410. Slice covers #11401 (ADR-0053), #11402 (LaiUhlrich2022 marker vocabulary), #11403 (`load_opencap_session`: model, SI kinematics, subject).
- Validation, constraints and the next step: `DL-#11400` in `docs/development/DEVELOPMENT_LOG.md`. Blocked children: #11404 waits on #11169, #11405 on #9627; #11408 needs a physical reference capture.

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

## Simscape Matching Review (2026-10-02)

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
