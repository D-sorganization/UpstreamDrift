# Necromatcher Handoff

## Authored Ground Placement

The shared author_ground_placement operation creates separate source-linked fits
using canonical native foot placement, a compensating camera translation and
all-frame rigid/pixel conservation checks. It preserves original solver evidence
in the immutable source version and records fresh source/runtime fingerprints.
Placement lineage is revalidated during import, recall and export. Thirteen new
cases pass; the combined suite is 136 passed and one unavailable Drake skip.
See [Ground Placement Procedure](docs/development/necromatcher-ground-placement.md).
Next: apply committed code to actual Hogan/Tiger versions, inspect residual
whole-track penetration, then closure/control recovery and downstream consumers.

## Authored Forward Replay

`workspace.replay_authored_profile` connects exact bound effort profiles to the
existing full-body RK4 simulator. It preserves saved initial poses and explicit
operator rates, rejects nonzero root polynomials, and repeats integration with a
fresh model at a finer step. Canonical Trace metadata preserves ordered SI units,
source frames and parent hashes. Numerical agreement and grip gaps are separate;
scientific and source-clock qualification remain false. See
[Replay Procedure](docs/development/necromatcher-replay.md). Immutable replay library storage, recall, export and shared local API
transport now revalidate source-bound traces and all parents. Next: explicit world-ground/camera placement revision
(the actual initial poses penetrate the assumed ground by 0.850/0.495 m), then
closure, controls and
unit-aware impact/whole-analysis consumers. The full goal remains active.

## Compiled Native Resources and Authored Controls

NativeFitBinding now shares one hash-checked compiled plant between projection
review and refitting. Named compiled joints independently establish ordered m/rad
units; missing capability and declaration mismatch fail. Authored controls must
match exact model/fit hashes, order and N/N\*m units before native effort mapping.
Detached sample metadata cannot change retained compiled model identity.
96 focused checks pass; one unavailable real-Drake check skips. See
[Native Resource Procedure](docs/development/necromatcher-native-resources.md). Historical driving controls, independent
replay and downstream/site integration remain open; no scientific acceptance is
claimed. Actual Hogan/Tiger resource checks from source 54d8f6fb13 verify
750/210 saved frames, three compiled translation and 41 rotation coordinates,
and thirteen projected attachments at start/middle/end. The native-resource
receipt preserves exact identities; motion qualification remains rejected research.

## Unit-Preserving Authored Controls

The new effort-profile/2 format binds exact model and research-fit hashes and
preserves ordered m/rad coordinates with N/N\*m generalized efforts. Import,
recall and export reject incompatible units and legacy torque profiles for known
translation coordinates. Bounded evaluation rejects extrapolation and overflow;
coefficients use immutable backing bytes. The existing native/web import route
accepts the format. See [Authored Effort Procedure](docs/development/necromatcher-effort-profiles.md).
54 focused profile/library/fit/refit checks and two-module mypy pass.
A separate red-first canonical handoff regression now preserves mixed units,
exact asset hashes, draft status and unqualified controls; 31 effort/handoff
checks pass after schema registration.
Actual historical driving controls and independent
replay remain open; source timing and scientific acceptance remain unqualified.

## Native and Web Research Refit Controls

Necromatcher now submits, polls and cancels immutable research refits through
one shared NativeRefitSession over the canonical matching executor. Admission
permits one active run per host; terminal views reopen canonical manifests.
Requests are saved before execution, with execution_started distinguishing queued
identity from actual worker start. Native dialogs submit off the Qt thread;
web controls retain job IDs in URLs, reject stale source completions and expose
explicit coordinate scales, sampling, priors and budgets. Saved nonterminal runs
without an owned handle retain their recorded status with an explicit unverified
execution message; absence of a handle does not prove worker termination.

Ninety focused Python checks pass, including five new controller/API/native checks
and a real clean worker rejected
for absent model assumptions. Nineteen web page/form controls pass; TypeScript,
scoped ESLint and three changed source modules pass type validation. Continue
anatomical bounds/closed motion,
mixed effort units, independent replay and downstream handoffs. The full goal
and #11235 remain open; no scientific acceptance or green CI is claimed.

## Source-Stamped Native Research Jobs

Canonical matching jobs now execute immutable native warm-start refits in a clean
interpreter with source/runtime/input fingerprints captured at execution. Explicit
work outcomes preserve rejection after successful computation. Cancellation and
publication share a commit gate, and dense/held-out errors retain source confidence.
Seventy-three fitting/job/storage/API/native review checks passed at implementation
commit `5038b0748bec967931ab76a07593aa1af14ca2d8`. Its actual v4 jobs saved
750 Hogan frames and 210 Tiger frames with launch/worker source and runtime
identities. Held-out RMS is 7.635 px for Hogan and 11.861 px for Tiger; Hogan
improved slightly versus v2 while Tiger worsened. Both computations succeeded
with rejected qualification, evaluation-budget exhaustion and large grip gaps.
The portable [V4 Run Receipt](docs/development/historical_capture/native-refit-job-receipt-v4.json)
records exact identities and metrics. A follow-up reproduces and repairs clean
worker launches without inherited PYTHONPATH using the shared core environment
builder extracted from Capture Rig; 84 focused checks pass without inherited
PYTHONPATH. UI/API job submission, anatomical bounds,
closed motion, effort units and qualified downstream replay still require work.

## Native and Web Projection Review

The saved 750-frame Hogan and 210-frame Tiger fits now have native/web model
projection review using exact source frames and verified native XML bindings.
A real Windows Qt DLL-order failure was reproduced and fixed with a reusable
clean-interpreter worker; a fresh Qt-parent regression guards it. Twenty-five
fit/storage/API/GUI/overlay checks pass. Camera and physical time remain
unqualified; continue fit jobs, closed motion, effort units and downstream replay.
Native PR #11240 exhausted its three CI remediation cycles; the exact prior
head's remaining shallow-checkout failure is recorded in
[Native CI Report](docs/ci-failures/11235-20261001.md).

## Persistent Fit Storage

Source-bound kinematic research fit storage and recall now use the existing
immutable library and portable swing export. Real Hogan/Tiger v2 versions retain
750/210 exact source frames, model/capture hashes, native coordinate samples,
original generic model definitions and rejection evidence. Sixteen new storage
regressions cover stale bindings, cross-swing inputs and malformed claims.
An additional API regression verifies import, exact source-frame recall and stale
parent rejection. Web labeling and import distinguish research fits from authored
controls; fourteen page/form tests pass. Continue with
native fit overlays/job execution, qualified closed motion and downstream replay.
See [Native Fitting Turnover](docs/development/necromatcher-native-fit.md).

## Native Image Fitting Progress

The shared historical-fit package now fits actual native geometry to image
observations and retains Hermite coefficients for full-source evaluation.
Nine fitting/evidence tests pass, including independent spline-chain derivative
checks and reduced native evaluation counts. Full-body held-out RMS improves to
7.694 px for Hogan and 11.666 px for Tiger. Both solves remain evaluation-limited;
maximum grip separation is 0.126/0.257 m and Tiger violates declared ROM.
No dynamics acceptance is inferred. Launcher logos, migration classifications,
companion inventories and generated context views are repaired. Continue with
source-stamped fit jobs, bounded/closed motion, mixed force/torque units, stored
fit versions, independent replay and downstream handoffs. See the native fitting
turnover procedure.

Active native-fit worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher-native-fit`,
branch `feat/necromatcher-native-fit-11235`. Actual MuJoCo closure probing failed
before the marker-independent repair; five new regressions now pass, and broader
native checks pass 12 with one Drake skip. The generic driver specimen loads 44
coordinates; its nonzero grip separation is not historical fit acceptance.
Continue the full goal under #11235 using [Native Fitting Turnover](docs/development/necromatcher-native-fit.md).

Workspace CI uses three remediation cycles; remaining runner Rust/Clippy setup
failure is recorded in [Workspace CI Report](docs/ci-failures/11234-20261001.md).
Five native GUI/overlay tests pass, including owner-thread completion and Qt
responsiveness; the supported GUI heuristic annotation does not increase its
baseline. Generated API types are refreshed, and web requests reuse them.
Native closure TDD work now exists on `feat/necromatcher-native-fit-11235` in the
separate owned native-fit worktree; no historical fit is accepted yet.

## Active: Integrated Historical Player Workspace

- Current workspace implementation: `feat/necromatcher-workspace-11234` in `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher-workspace`, baseline commit `1b0099a405`. Both hosts create players/swings, import immutable versions, review original images/landmarks and export packages. Real Hogan and Tiger web imagery is verified; Hogan frame navigation reaches frame 750 at source PTS 134.967 s. 31 library/API/native/launcher tests and 10 inventory tests pass; scoped mypy passes eight production files. Further web form tests and fitting remain active.
- Dependency PR #11237 (library) is published as a draft over #11231. CI #11231 has an unrelated `.jules/bolt.md:208` title-case failure inherited from main; capture unit/structure/code checks are green. No CI bypass or unrelated code edits.
- Workspace draft PR [#11239](https://github.com/D-sorganization/UpstreamDrift/pull/11239) is published over #11237 and attached to this chat. Cleanup LoD fix `d762816466` passed native tests, global LoD and push hooks; its LoD CI gate passed. Navigation tests reproduced stale assets on URL recall and endless frame loading after failure; scoped results and retry fix both, and reject a player/swing URL mismatch. Sixteen UI tests pass. CI cycle 2 addresses stale launcher context/atlas views using canonical generators and explicit boundary review. Continue form verification, CI tracking and actual fitting under #11235; keep the full goal active.

- Owner priority/goal: Necromatcher epic #11232; persistent library #11233; Tiger #11226 and Hogan #11229 remain open.
- Owned worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher`, branch `feat/necromatcher-library-11233`; depends on capture foundation PR #11231. Library code is published in draft PR #11237; tile/review work continues separately under #11234.
- Current contracts, TDD evidence, CI status and continuation: [Necromatcher Turnover](docs/development/necromatcher-turnover.md).
- Implemented in progress: immutable source/model/control versions, restart recall, checked capture archives, portable exports and local HTTP API. 43 workspace/API tests passed; scoped library mypy passed; real capture imports verified after reopening. Malformed-profile and export-mutation failures are covered by the passing suite.
- Next: finish library acceptance/import real captures, publish focused PR, add player tiles and web/desktop review, integrate real fitting and downstream simulation/impact/analysis. Do not certify fixed-output coordinator artifacts or uncalibrated source time.

# Historical Player Capture Handoff

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
