# Motion-Matching Handoff

Deferred external validation: six Board plans live in `docs/development/planning/`.
Software remains active; no physical evidence is supplied. See the current
`docs/development/HANDOFF.md` for scope, prior #9546 closure and publication gates.

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
- Evidence receipt: `docs/plans/neural_motion_matching/evidence/nm10_benchmark_speed_efficiency_receipt.json`.
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

## Tour Baselines TB-06: Constrained Upper-Body Golfer (#10591)

Branch `feat/10591-upper-body-capture`; parent epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584); program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363). Lease holder: `codex` (session `01a0ca1f-c09b-78a1-8878-37402a7d4eec`).

- PR [#10732](https://github.com/D-sorganization/UpstreamDrift/pull/10732) merged as `234b8dc146c5b2c24d80cf9eb4d09d37e68c1b72`. It establishes `upper_body_replay`: constrained native replay, separately recorded seven actuator torques and four constraint reactions, exact terminal source frame, native kinematic points, explicit planar marker attachments, rigid capture-frame embedding, and standard physical marker metrics only on an identical body-target clock.
- The evaluator does not synthesize 3D attachments, infer capture transforms, interpolate source clocks, or claim acceptance. A capture-specific fit must calibrate fixed geometry, feasible `q0`/`v0`, marker attachments, and bounded torque controls before it can produce Driver or Iron evidence.
- PR [#10733](https://github.com/D-sorganization/UpstreamDrift/pull/10733) merged as `91fff0cc582d5a906bfb0530d7bda2d8c67aaf6c`. PR [#10735](https://github.com/D-sorganization/UpstreamDrift/pull/10735) merged as `930e1edcbd1ec60d7deb83907694662e8496feb7`, superseding stale/conflicted #10645 with native closed-loop fitting and continuous-control replay. Its controls, residual policy, and finite bounded-solver tolerances are shared through `motion_matching.bernstein_controls`.
- `upper_body_capture.py` now retains native C3D source-clock metadata and produces one fixed rigid capture plane from declared markers. On the six shoulder, elbow, and wrist markers, the real Driver and Iron traces yield 110.4 mm and 112.7 mm plane RMSE respectively. This is a diagnostic, not a flattened projection or qualification result.
- PR [#10740](https://github.com/D-sorganization/UpstreamDrift/pull/10740) merged as `56ad6a8dcd13221699ec14442b13c1df6fc5729a`. Its source-clock, planarity lower-bound, and campaign-receipt contracts retain reproducible Driver and Iron receipts under `docs/plans/tour_baselines/evidence/`: 110.4 mm and 112.7 mm irreducible normal RMSE, respectively, above the profile 55 mm 3D marker ceiling. Both outcomes are rejected and indexed in `reports/matched_swing_ledger.json`.
- Source boundaries are `physics_golfer.py`, `golfer_constraints.py`, `simulation_golfer.py`, and `model_registry.py`. Do not edit `vendor/ud-tools`.
- Focused validation: `python3 -m pytest tests/unit/motion_matching/test_bernstein_controls.py tests/unit/engines/physics_engines/pendulum/test_golfer_fit.py tests/unit/pendulum_simulator/test_upper_body_replay.py -q -n 0 --no-cov --timeout=60` (32 passed; manufactured optimizer regression 9.2 s); scoped Ruff, format, and `python3 scripts/ci/check_dry_duplication_gate.py` pass.
- Campaign conclusion: do not calibrate planar marker attachments, q0/v0, or torque controls against these captures because their measured normal residual alone exceeds the full 3D acceptance ceiling. A spatial upper-body topology would require a separately scoped follow-up; this planar topology retains both rejected outcomes.

## Tour Baselines TB-07: Reconcile Existing Reference and Full-Body Results (#10592) [MERGED]

PR [#10730](https://github.com/D-sorganization/UpstreamDrift/pull/10730) merged as `22b7fd7a534ddfbc0b1db943e6443fdc9ac7efb3`. The TB-04 Driver and Iron receipts remain `DISQUALIFIED`; coverage and the unified ledger report them as rejected rather than qualified.

## Tour Baselines TB-08: Bounded Fit Campaigns With Checkpoints (#10593)

Branch `feat/tb08-bounded-fit-campaigns-10593`; parent epic [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584); program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363). Lease holder: `antigravity` (session `f826c3ee-41ca-461c-87f2-1ab5695f3736`).

- Implemented `src/shared/python/tour_baselines/campaign.py`:
  - `CampaignJobSpec`: parameter bounds, basis/order, holdout window, seeds, wall-time limits, reproduction command, content hashes.
  - `rank_candidates`: Pareto / feasibility ordering where feasible candidates always beat lower-error infeasible candidates, which are retained as rejected evidence.
  - `FitCampaignService`: atomic immutable checkpoints, cryptographic hash checking on resume with `IncompatibleResumeError`, cancellation/timeout diagnostics without candidate promotion.
  - `GeneralizationDisclaimer`: explicit notice that within-capture holdout is not population generalization.
  - `compute_clock_scores`: exact full-rate clock scores matching stored metrics without resampling distortion.
  - Pilot benchmark runner for measuring per-evaluation runtime and freezing campaign budgets.
- Verified by unit test matrix in `tests/unit/tour_baselines/test_fit_campaign.py` (7 passed) and full tour baseline suite (64 passed).

## Required Before Continuing

- Read `AGENTS.md`, `CLAUDE.md`, and `docs/development/DEVELOPMENT_LOG.md`.
- Keep #10591 open until a merged PR demonstrably meets its acceptance criteria. A replay foundation or a rejected candidate is not qualification.
- Manual governance: UP-D0 (#9066) and UP-D1 (#9067) remain release blockers. Edit only the `manuals/upstreamdrift` QMD source and run `python3 -m scripts.check_design_manual_governance` for governed changes.
- Update this handoff, the development-log entry, and exactly one `SPEC.md` change-log row for every substantive PR.
