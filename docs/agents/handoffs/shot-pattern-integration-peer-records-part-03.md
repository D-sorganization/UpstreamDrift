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

## F05d Native Manifold Control Candidate (#11932; 2026-10-09)

The contact-free floating-root/two-hinge Crocoddyl BoxFDDP action uses the
actual native Euler MuJoCo step and $n_x=17$, $n_{dx}=16$ manifold state.
It checks candidate plans against a nonlinear native rollout and verified
zero-torque fallback, then shares the direct-torque executor for post-limit
ZOH export and independent complete-native-state replay. The supported
MuJoCo 3.8/Crocoddyl 3.2.1 paired receipt binds two starts, common T01
preflight, every accepted/timed-out call, cost, wall time and replay hashes.
The reachable reference is generated by native steps. BoxFDDP accepts
3/8 synthetic steps at the 0.2 s cooperative budget; SciPy SLSQP accepts
0/8. All other steps use verified fallback. Neither method completes a
fully accepted four-step run, although BoxFDDP has lower realized native
cost in both starts. This does not prove
a hard 10 ms deadline or golfer/contact/muscle/capture qualification.
Read provisional manual chapter 26 and
`docs/development/feedback_controls/F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md`.

Parent F05 #11789, F09/F10 and manual publication remain open.
PR #11944 merged into the F05c feature base at
`c1e76ce91db11b257604d9ddbd227268d2e27db1`; F05c PR #11922 is
still open on F05b, issue #11932 is still open, and this has not reached
`main`.

## F02 Native Manifold Feedback Child (#11946)

F02 now has an actual MuJoCo floating-root feedback runner using the F05c
tangent derivative admission and F06 complete-state torque replay; see
provisional manual chapter 27 and
`docs/development/feedback_controls/F02_NATIVE_MANIFOLD_FEEDBACK_TURNOVER.md`.
The source-hashed static receipt shows 0.1261 rad final hip error versus
0.4 rad frozen nominal over 12 steps. A second test consumes a varying
native-replayed torque trajectory, derives per-step native tangent Jacobians
and MOSAIC gains, then compares held-out frozen-feedforward and total-
feedback replays at 0.07508 versus 0.02460 rad final hip error. Their
post-limit input digests differ, while each fresh native replay reproduces
the complete integration-state trajectory exactly. F03 still lacks a native
9/8/2 optimized trajectory export; the teacher is admissible but is not
captured or optimized. This child is stacked on F05c #11922 and must remain
unarmed until its prerequisite reaches `main`; F02 #11786 and full
production/capture qualification remain open.

### F09b Native Observation Scoring (#11837)

`src/engines/feedback_observation_qualification.py` scores supplied native
marker positions against exact measured observations using F01's registry,
F09a's position-only sampler, and the existing acceptance evaluator. A case
requires a validated T01 `ExperimentReplayBundle`; identity includes its
actual complete initial-state payload SHA-256, model-identity digest, and
capability-declaration digest. Input values/grid and executed policy are
cross-checked against F01 evidence before scoring. Reports preserve every
required comparison row and all six required engines; scores do not promote
unqualified rows or missing evidence.

The focused synthetic suite covers all-six denominator retention, measured
clock scoring, gate/configuration identity, stale-input rejection, and full
initial-state payload binding. It does not establish native replay,
cross-engine parity, muscle physiology, or private-capture acceptance. Resume
from `docs/development/feedback_controls/F09B-OBSERVATION-QUALIFICATION.md`;
rerun the focused command recorded there with the pinned Tools submodule.

## Native Pinocchio Replay Handoff (#11900)

Native RED f273e45f2f precedes implementation; twelve actual Pinocchio4.1 tests pass including saturated torque reproduction and fixture refinement. See docs/development/feedback_controls/F06_PINOCCHIO_NATIVE_TURNOVER.md and canonical chapter25 for equations, commands, failed setups, resource ownership and remaining full-model/capture gates. Shared native admission changes require actual Drake/MuJoCo regression evidence before publication. This child does not close F06 or the full matching epic.

## F09e Pinocchio Native Marker FK (#11914)

Branch `feat/11914-f09e-pinocchio-marker-fk` is stacked on F09d #11913 and
merges the exact native replay provider head from #11906. Actual Pinocchio 4.1
`nq=8`, `nv=7` marker FK passes with independent nonzero floating-base and
frame-offset transforms. The public `PinocchioPhysicsEngine.get_link_transforms`
frame index was made compatible with Pinocchio 4.1. Native receipts retain
separate inventory and adapter identities, complete q/v, and unknown reset
count; marker results remain unqualified. See
`docs/development/feedback_controls/F09E-NATIVE-MARKER-FK.md` and chapter30.

## F09f MyoSuite Native Excitation Replay (#11918)

Branch `feat/11918-f09f-myosuite-native-replay` adds a distinct MyoSuite
muscle-excitation consumer to the F09 native execution path. It writes exact
post-mapping normalized excitation directly to native `data.ctrl`; it never
calls Gym `step` or treats `[-1, 1]` actions as muscle excitation. MyoSuite
3.0.0/MuJoCo 3.6.0 public elbow fixture passes in the owned Python 3.12 runtime.
The built-in fixture does not bind required production driver/iron models;
those rows and the six-engine denominator remain unqualified. See
`docs/development/feedback_controls/F09F-MYOSUITE-NATIVE-EXCITATION.md` and
canonical chapter31. The generic legacy four-value Gym step issue remains
separate because this provider uses native plant stepping only.
