# Deferred External Validation Planning — 2026-09-22

- Worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-validation-planning`.
  Branch `docs/deferred-validation-planning`; implementation `cb127ad02`; PR #10741 is open.
  Central Repository_Management #1687; governing impact epic #9700.
- Six repo-owned plans preserve unavailable impact/acoustic/perceptual studies,
  cohort/generalization validation, observable-accuracy reference measurements,
  everyday-reference calibration, three-camera hardware soak and Impact Explorer
  predictive-accuracy evidence. Catalog entries retain original issue snapshots,
  missing resources, acceptance boundaries and Board/reactivation gates.
- #9700/#10375/#10382/#9619/#9613 stay open for executable software and available
  data. #9546 was already closed through #10446 at 66527df069df; this migration
  preserves that state and makes no new completion claim. No engine, public
  physics API, scientific evidence, calibration or provider pin changes here.
- #10363/#10380 should reference DV-10375/DV-10382; use the same experiment
  records rather than duplicate cohorts. Existing industrial/product/experiment
  governance remains authoritative; a deferred requirement still blocks its
  original physical, perceptual or release claim.
- Validation: unchanged source bodies verified against the staged snapshot;
  strict six-plan catalog, manual governance, SPEC and normal commit/push hooks
  pass. Hosted checks are pending. The
  checkout is sparse to conserve C: capacity; no full local suite is claimed.
- Next: pass normal hooks and hosted checks, merge, compare published plan bytes,
  then post immutable scope links and audit receipts. Preserve the existing
  closed state of #9546 and keep the five open mixed sources open.

# Current Matching Continuation Handoff

## One User Config Root + QSettings Namespace (#8907, DL-#8907) — 2026-09-22

- **Repo / worktree:** D-sorganization/UpstreamDrift ·
  `C:\Users\diete\Repositories\_wt_ud_8907` · branch
  `fix/8907-settings-root` · commit `SELF` · PR not created at commit time
  (see PR list for the branch). Agent `claude`, session `fleet-remediation-j`.
- **Done:** `src/shared/python/data_io/user_config_root.py` owns the root
  (`user_config_dir`, `user_config_path`) and `migrate_legacy_user_dirs`
  (copy launcher-owned names only, never clobber, marker file, retry on
  failure). `launcher_constants._get_config_dir()` runs it after the `.kiro`
  block. `src/launchers/launcher_settings_store.py` owns the canonical
  QSettings pair, legacy aliasing and `persist_window_geometry()` (event
  filter saves on hide). All listed writers and the diagnostics log viewer
  now use the helpers.
- **Not done (by design):** `~/.upstreamdrift/mcp_servers.json` is a
  Tools-owned contract read by vendored `ai/mcp` modules; moving it needs a
  Tools change first. Other `~/.golf_modeling_suite` users outside the issue
  list (chat sessions, subjects, sidekick runs, auth credentials, C3D viewer
  styles) are untouched.
- **Validation:** `QT_QPA_PLATFORM=offscreen python -m pytest
tests/unit/data_io/test_user_config_root.py
tests/launchers/test_launcher_settings_store.py -q` (all pass).
- **Next:** relocate the MCP config in Tools; extend "Restore Defaults" to
  clear the canonical QSettings store.

## Fix CI Standard 'Deleted Python Test Files' Check False-Positives in Shallow Checkouts (#10751) — 2026-09-23

- **Repo / Worktree:** D-sorganization/UpstreamDrift · `C:\Users\diete\Repositories\_worktrees\UpstreamDrift-10751` · branch `fix/10751-deleted-tests-shallow-checkout` · agent `antigravity`
- **Problem:** In CI Standard (`.github/workflows/ci-standard.yml`), the `tests` job evaluated deleted test files by comparing `diff_base` directly against `HEAD`. When `merge_base` resolution failed in shallow checkouts (due to default `fetch-depth: 1` and `--depth=10`), `diff_base` fell back to `origin/${{ github.base_ref }}` (tip of `main`). Any new test files added to `main` after the PR branched were falsely reported as deleted by the PR, blocking unrelated dependabot and feature PRs.
- **Fix:**
  1. Updated `actions/checkout` in `ci-standard.yml` `tests` job to use `fetch-depth: 0`, and updated base ref fetch without shallow depth restriction.
  2. Created `scripts/ci/check_deleted_test_files.py` providing a reusable, DbC-decorated, LoD-compliant detection CLI and Python API with merge-base resolution and fallback-to-base support.
  3. Invoked `scripts/ci/check_deleted_test_files.py` directly from `ci-standard.yml`, avoiding duplication between bash diffing and python helper.
  4. Added comprehensive test suite `tests/scripts/test_check_deleted_test_files.py` (10 tests) including an explicit regression test reproducing the scenario of test files added to `main` after a PR branch is cut, plus fallback-to-base test coverage.
  5. Updated `tests/ci/test_ci_infrastructure.py` asserting `fetch-depth: 0` and helper script invocation.
- **Validation:**
  - `pytest tests/scripts/test_check_deleted_test_files.py -v` (10 passed).
  - `pytest tests/ci/test_ci_infrastructure.py` (85 passed, 1 skipped).
  - `ruff check`, `black --check`, and `run_mypy.py` pass cleanly with zero errors.
- **Next:** Await owner review for workflow change on PR #10762.

## Tests In-Place JSON Mutation Fix (#10750) — 2026-09-23

- **Repo / Worktree:** D-sorganization/UpstreamDrift · `C:\Users\diete\Repositories\_worktrees\UpstreamDrift-10750` · branch `fix/10750-tests-in-place-json-mutation` · agent `antigravity`
- **Problem:** Running unit tests rewrote committed repository JSON files in place:
  1. `test_fast_match_evidence_fixture_roundtrip` directly mutated `docs/plans/club_only_matching/evidence/club_fast_matching.json` with machine-specific timing profiles.
  2. `HumanoidLauncher.__init__` unconditionally called `self.config_manager.save(self.config)` upon instantiation, dirtying `src/engines/physics_engines/mujoco/docker/src/simulation_config.json` with host-specific paths during test runs.
- **Fix:**
  1. Export `save_fast_match_evidence(result, evidence_dir=None)` in `fast_matching.py` with DbC contracts; default retains canonical committed evidence directory; `test_fast_match_evidence_fixture_roundtrip` writes to `tmp_path`, and `test_fast_match_evidence_does_not_mutate_committed_file` asserts committed bytes remain untouched.
  2. Update `HumanoidLauncher.__init__(self, config_path=None, save_on_init=False)` to accept custom `config_path` (and optional `SIMULATION_CONFIG_PATH` env var) and avoid mutating on-disk config during initialization unless `save_on_init=True`. `save_config()` remains wired to explicit user save and simulation launch.
  3. Mark `TestHumanoidLauncher` with `@pytest.mark.unit` to satisfy the suite-marker ratchet, and add regression tests asserting committed `simulation_config.json` is not mutated on instantiation and custom `config_path` routing is respected.
  4. Regenerate `divergence_inventory.v1.json` and shrink `suite_marker_baseline.json`.
- **Validation:**
  - `pytest tests/unit/motion_matching/test_club_fast_matching.py` (13 passed, 0 dirty files).
  - `pytest tests/unit/test_gui_coverage.py::TestHumanoidLauncher` (4 passed, 0 dirty files).
  - `pytest tests/unit/engines/physics_engines/mujoco/test_humanoid_launchers.py tests/unit/launcher/test_launch_mode_fixes.py` (15 passed).
  - `ruff check` and `ruff format` pass cleanly.
- **Next:** Open PR, verify CI, auto-merge.

## Fleet Remediation — Ledger Freshness Test Pollution (2026-09-22)

- **Branch:** `fix/ledger-freshness-after-10733` · **agent:** `claude` (fleet-remediation)
- **Problem:** `test_driver_and_iron_qualification_receipts` rewrote the committed
  TB-04 receipts in `docs/plans/tour_baselines/evidence/` in place; after #10733
  the regenerated bytes differ, so `tests/unit/motion_matching/test_ledger.py::test_ledger_freshness`
  failed on main (91fff0cc5) whenever it ran after the receipt test.
- **Fix:** `save_qualification_receipts(repo_root, evidence_dir=None)`; the test writes to `tmp_path`.
  Committed receipts are unchanged.
- **Validation:** receipt test then ledger test in one process, `-p no:randomly`: 11 passed
  (unpatched main: 1 failed, reproduced).
- **Next:** none after merge.

## Tools Session Bridge CameraCapabilities (#9604, DL-#9422) — 2026-09-22

- **Repo / worktree:** D-sorganization/UpstreamDrift ·
  `C:\Users\diete\Repositories\_wt_ud_9604` · branch
  `fix/9604-camera-capabilities` · commit `SELF` · PR not created at commit
  time (see PR list for the branch). Agent `claude`, session
  `fleet-remediation-i`.
- **Done:** `src/motion_capture/rig/tools_bridge.py` gains `CameraRecord` and
  `map_camera_records(manifest, plan)`: per-camera Tools `CameraIdentity` +
  `CameraCapabilities` (vendor pin `a9ed0e7c`). The export reuses the same
  identity builder, `_ready_tools()` probe guard and `_clock_kind()` lookup.
- **Validation:** `python -I tests/fixtures/mocap_session_export/run_checks.py`
  (14 passed); `python -m pytest tests/motion_capture/rig/test_tools_session_export.py -q --no-cov`.
- **Next:** once merged, #9604 acceptance is complete; the C3D upload consumer
  (#8865) is the next slice on DL-#9422.

## Simulation Polling Client Side (#8941, DL-#8941) — 2026-09-22

- **Repo / worktree:** D-sorganization/UpstreamDrift ·
  `C:\Users\diete\Repositories\_wt_ud_8941ui` · branch
  `fix/8941-client-polling` · commit `SELF` · PR #10748 (predicted at commit
  time; see PR list for the branch). Agent `claude`, session
  `fleet-remediation-o`.
- **Done:** `ui/src/hooks/usePolling.ts` (interval > 0 contract, single-flight
  ticks, paused while disabled or the tab is hidden, cleared on unmount) and
  `ui/src/hooks/useIncrementalSeries.ts` (monotonic `since` cursor via
  `advanceCursor`, new epoch on regression, end-aligned append, `maxPoints`
  window). `AnalysisPanel` fetches
  `/api/analysis/statistics?collect=true&limit=&since=` once per tick through
  `ui/src/api/analysisStatistics.ts`; `apiFetchWithHeaders` exposes the cursor
  header. `ForceOverlayPanel` uses `usePolling` at 500 ms (was 200 ms).
  Server: `collect` query flag (default false) on `/analysis/statistics`.
- **Validation:** in `ui/`: `npm ci`, `npx tsc -b`, `npm run lint`,
  `npx vitest run` (98 files / 925 tests passed), `npm run build`. Server:
  `pytest tests/unit/api/test_analysis_statistics_window.py tests/unit/api/test_routes_analysis_tools.py tests/api/test_generated_ui_api_types.py tests/scripts/test_monolith_register.py --no-cov`
  (44 passed; `analysis_tools.py` kept at 800 LOC so it stays off the monolith register); ruff,
  architecture budget and error-handling ratchet pass.
- **Next:** remaining #8941 items — `ActuatorPanel`/`SimulationToolbar`
  1000 ms loops onto `usePolling`; force/analysis frames over `/ws/simulate`
  (#8936/#8940).

## Analysis Statistics Server Side (#8941, DL-#8941) — 2026-09-22

- **Repo / worktree:** D-sorganization/UpstreamDrift ·
  `C:\Users\diete\Repositories\_wt_ud_8941` · branch
  `fix/8941-analysis-stats-server` · commit `SELF` · PR not created at commit
  time (see PR list for `fix/8941-analysis-stats-server`). Agent `claude`,
  session `fleet-remediation-f`.
- **Done:** `src/api/routes/analysis_tools.py` — history is
  `deque(maxlen=500)` plus monotonic `_metric_sample_total`; one aggregation
  helper `_compute_statistics` runs in `anyio.to_thread.run_sync` over a tuple
  snapshot taken on the loop; `GET /analysis/statistics?since=&limit=`
  (`since>=0`, `1<=limit<=500`, else 422) trims only `time_series`; summaries
  and the default body are unchanged; cursor returned in the
  `X-Analysis-Next-Since` header so the Pydantic model and generated UI types
  are untouched. JSON export now serialises `list(history)`.
- **Validation:** `python -m pytest tests/unit/api/test_analysis_statistics_window.py tests/unit/api/test_routes_analysis_tools.py tests/api/test_generated_ui_api_types.py tests/api/test_phase3_api.py -q --no-cov` (87 passed).
- **Next:** client-side #8941 items — merge metrics+statistics calls, use
  `since`, delete the 200 ms / 500 ms polling loops, publish frames on
  `/ws/simulate`. Do not touch `ui/` from this branch.

## Hip-Calibrated Receipt Provenance (#10271) — 2026-09-22

- **Goal:** Restore end-to-end provenance for `anthro_driver` / `anthro_iron`
  ground-support receipts (child of #10254).
- **Branch / worktree:** `fix/10271-hipcal-provenance` at
  `C:\Users\diete\Repositories\Worktrees\UpstreamDrift-10271-provenance`
- **Lease:** `cursor-10271-202609220928` (agent `local`)
- **Done:** `receipt_provenance.py` chain validator (`receipt-provenance-chain/1`);
  TDD suite; producer emits `spec_canonical_sha256`; CI gate on both baselines;
  receipts re-anchored to current base + scaled digests after verifying scaled
  docs already carry current de Leva/shank; intermediate hipcal not fabricated;
  physical metrics unchanged.
- **Disk blocker:** C: ≈ 1.5–2 GB free — full native MuJoCo re-execution deferred;
  provenance re-anchor is software-contract only (not a new physical run).
- **Validate:**
  ```powershell
  $env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; $env:PYTHONPATH='.'
  python -m pytest tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py -q -n 0 --no-cov --timeout=60
  python -O -c "from src.shared.python.motion_matching.pipeline.receipt_provenance import *"
  ```
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10722 (ready-for-review)
- **Next:** Confirm CI green on #10722; after merge, regen natively when disk recovers.

## GolfSwingVisualizer MATLAB Consolidation (#9225)

- PR [#10715](https://github.com/D-sorganization/UpstreamDrift/pull/10715)
  (open); implementation commit `f33b40c9c`.
- Worktree: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/local-9225`,
  branch `bot/issue-9225-golfviz-consolidation` (off `origin/main` @ `901b2de5e`),
  DL-#9225. Governing issue
  [#9225](https://github.com/D-sorganization/UpstreamDrift/issues/9225)
  (source:assessment P2, DRY PP1).
- Delivered: the four per-tree `GolfSwingVisualizer.m` copies (1180–1183 lines
  each) deleted; one fleet-shared package class at
  `src/engines/Simscape_Multibody_Models/shared/+golfviz/GolfSwingVisualizer.m`
  (2D-variant superset: the reproducible-ground-texture `rng(1)` seeding the 3D
  copies had silently lost). Both `launch_gui.m` launchers add the shared
  directory to the MATLAB path and fail loudly if it is missing; all four call
  sites use `golfviz.GolfSwingVisualizer(BASEQ, ZTCFQ, DELTAQ)`.
- Validation: MATLAB R2025b `-batch` headless — package resolves from a bare
  `addpath`, class parses (75 methods), constructor DbC precondition
  `GolfSwingVisualizer:InvalidInput` fires, and both launchers' relative path
  setups resolve the shared package with no shadow copy. Full GUI launch/render
  is not exercised (headless GUI rule); the diff is behaviour-preserving except
  for the restored `rng` seeding on the 3D trees.
- Honest gaps: the `check_dry_duplication_gate.py` ratchet is Python-scoped
  (`src/**/*.py`) and does not fingerprint `.m` files, so no ratchet baseline
  drops; extending it to MATLAB is a follow-up. `quality-gate` is expected
  green (no Python/CI surface changed).
- Next: confirm `quality-gate` green on the PR; squash auto-merge lands; release
  the lease.

## MS-104 Full-Swing Qualification Matrix (#10378)

- PR [#10707](https://github.com/D-sorganization/UpstreamDrift/pull/10707)
  (open); branch `feat/ms104-full-swing-qualification` off origin/main.
- Delivered: `src/shared/python/motion_matching/full_swing_qualification.py`
  evaluates 36 required full-body flagship cells (6 engines × driver/iron ×
  G1/G2/G3) against the matched-swing ledger plus explicit evidence links
  (MS-100 acceptance, MS-72 conformance, native replay, numerical
  convergence, provenance hashes). Fail-closed `release_status`; reduced
  27-DOF Simscape oracle is partial only and cannot fill a flagship cell.
  Named per-engine owner blockers retained (MS-21/30/42/53/107/111; folded
  MS-109/110/112 → #10378).
- Evidence: `docs/plans/matched_swing/evidence/ms104_full_swing_qualification.json`
  (`release_status=blocked`, `incomplete_required_count=36`).
- Validation: `python -m pytest tests/unit/motion_matching/test_full_swing_qualification.py -q -n 0 --no-cov`
  GREEN (15 passed); ruff clean on touched files. CI fix: regenerated
  `docs/shared_tools/divergence_inventory.v1.json` against pinned
  `vendor/ud-tools` (includes tools-only `launch_monitor/gspro_connect.py`).
  Merged `origin/main` through #10709 (NM-06), #10718/#10720 (CO-10), and #10721
  (succession handoff); `tests/unit/motion_matching/jobs/test_matching_jobs.py`
  retained; SPEC keeps one row each for #10707, #10709, #10718, #10720, #10721
  (deduped duplicate #10721 key after main merge).
- Named blockers: no invented six-engine native pass; every incomplete cell
  names its owner issue; software-contract fixtures only.
- Next: Confirm CI green on #10707 tip after SPEC duplicate-key repair; do not
  start MS-106 or steal NM-07 #10622 from this worktree.

## Succession — Motion Matching (2026-09-22)

- **Goal:** Finish Antigravity-started motion matching (club-only CO + neural NM)
  with TDD/DbC/LoD/DRY + fleet rules; professional long-term quality. Software
  contracts ≠ native Fit/G1/G3 success; do not close epics on docs/GUI alone.
- **Session:** `b27ccab3-1128-492c-a0fb-001367ea3aa8` · **agent:** `local`
- **Durable copy:** agent store
  `…/b27ccab3-1128-492c-a0fb-001367ea3aa8/files/MOTION_MATCHING_HANDOFF.md`
- **Landed:** CO-00..CO-10 · NM-00..NM-06
  - CO: #10667, #10670, #10675, #10678, #10680, #10681, #10687, #10700, #10703,
    #10711, **#10718** / **#10720** (CO-10 `abd35e66b`)
  - NM: #10668, #10672, #10679, #10686, #10698, #10701, **#10709** (merged
    `08bcec302`)
- **In flight:** CO-09 review-gap harden
  [#10717](https://github.com/D-sorganization/UpstreamDrift/pull/10717)
  (`fix/10613-co09-review-gaps`) rematched onto post-CO-10 main.
- **In flight / Landed:** NM-07 [#10622](https://github.com/D-sorganization/UpstreamDrift/issues/10622) (in flight by `local` session `76a4b2bb-f31a-4077-a500-f97c2a7c1541`).
- **Remaining:** NM-08..NM-12 (#10623–#10627). Epics
  [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602) /
  [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603) stay
  open until physical gaps (desk-native receipts) close.
- **Ordered next:** (1) land #10717 squash auto-merge (2) do not touch NM-07
  #10622 (3) renew leases via Repository_Management `post_agent_lease`
- **Worktrees:**
  - CO-09 review-gap: `C:\Users\diete\Repositories\Worktrees\UpstreamDrift-10613-co09-local`
  - CO-10 (merged): `…\UpstreamDrift-10614-co10-push` — leave idle
  - Duplicate CO-10 (closed #10719): `…\UpstreamDrift-10614-co10` — leave idle
  - NM-06 (merged): `…\UpstreamDrift-10621-nm06` — do not touch
  - Session root MS-60: `…\UpstreamDrift-10347-ms60` (clean; unrelated)
  - CO-08 worktree removed after merge (#10703)
- **Validation:**
  ```powershell
  $env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'
  python -m pytest tests/unit/motion_matching/test_club_ui_integration.py tests/unit/tools/test_tour_matching_viewer_core.py::test_club_only_compare_view_separates_observed_and_inferred tests/unit/workspace/test_results_browser.py -q -n 0 --no-cov --timeout=120
  ```
- **Risks:** prior session shell flaky; push ref-lock races; SPEC §12 keep-both
  rows on rebase; never edit `vendor/ud-tools`.

## Analysis Tab Debounce and Memoization (#8932) [IN REVIEW]

- Repo/worktree: `C:\Users\diete\Repositories\_wt_ud_8932`, branch
  `fix/8932-analysis-debounce` from `234b8dc14`; commit `SELF`; PR [#10734](https://github.com/D-sorganization/UpstreamDrift/pull/10734) open, auto-merge armed.
- Delivered: `src/shared/python/dashboard/_analysis_refresh.py`
  (`DebouncedRefresh` 150 ms single-shot QTimer, `BoundedResultCache` LRU of 8,
  `analysis_cache_key` with DbC checks and a blake2b signal digest). Spectrogram
  and Wavelet spinboxes are debounced and their transforms memoized;
  SwingPlaneTab creates its axes once and swaps artists; `draw_idle()` on these
  three tabs. Spectrogram/Wavelet share a `_SignalTransformTab` base (required by
  `check_dry_duplication_gate.py`). DL entry `DL-#8932`.
- Validation: `$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; python -m pytest tests/unit/shared_python/test_analysis_tab_refresh.py tests/unit/shared_python/test_advanced_analysis_features.py tests/unit/shared_python/test_dashboard_advanced_analysis.py -q -o addopts=""` (27 passed); ruff check/format clean.
- Remaining on #8932: repo-wide blocking `canvas.draw()` sweep and axes reuse in
  `plot_engine/pyqt6_widget.py`; the PR uses `Refs #8932`.
- Next: land the PR; then convert `plot_engine/pyqt6_widget.py` to axes reuse.

## CO-10 Publish Reproduction Guide and Final Club-Only Turnover (#10614) [MERGED]

- Baseline merged via PR [#10718](https://github.com/D-sorganization/UpstreamDrift/pull/10718)
  (squash `2f6e119ef`). Survivor follow-up PR
  [#10720](https://github.com/D-sorganization/UpstreamDrift/pull/10720) merged
  (`abd35e66b`). Duplicate #10719 stays closed.
- Delivered: `club_only/reproduction.py` freezes exact saved-job commands,
  trial/model roster, raw-source provenance, assumptions, candidate selection,
  clean-environment portable replay (MS-105 jobs), and evidence-linked matrix
  reconciliation with executable next-step prompts. Operator guide
  `docs/plans/club_only_matching/REPRODUCTION_GUIDE.md`; evidence
  `docs/plans/club_only_matching/evidence/club_reproduction_turnover.json`;
  schema `club-only-reproduction/1.0.0`.
- Validation: `python -m pytest tests/unit/motion_matching/test_club_reproduction_turnover.py -q -n 0 --no-cov --timeout=60` (13 passed); architecture budget GREEN after helper split. Saved-job `fast_preview_match` uses `build_club_only_result_view`; clean export passes `asset_paths={}`.
- Limitations: software-contract turnover only; `native_g1_pass` false;
  `epic_closure_allowed` false; epic #10602 stays open; no G3/neural inheritance.
- Next: N/A — merged; finish CO-09 review-gap #10717; on DeskComputer schedule
  native Fit/G1 for the first unresolved full_body_drake×TW_wiffle matrix cell
  and attach a receipt under docs/plans/club_only_matching/evidence/.

## CO-09 Integrate Club-Only Matching Into Existing UI and Results (#10613) [MERGED]

- Merged via PR [#10711](https://github.com/D-sorganization/UpstreamDrift/pull/10711).
- Delivered: `club_only/ui_integration.py` binds workbook identity + fast
  matching into FitSwingProvider/pipeline/ledger/`ResultsBrowser` without a
  parallel solver. Preview vs verified display statuses stay honest;
  observed/inferred legend and trial clock are required; cancel/resume hooks
  reuse CO-07 checkpoints; ledger lane is `club_only` with named native
  blockers. Motion Matching GUI adds a Club-Only tab; ResultsBrowser indexes
  `club_only_ui_result` JSON. Schema `club-only-ui-integration/1.0.0`; evidence
  `docs/plans/club_only_matching/evidence/club_ui_integration.json`.
- Validation: `python -m pytest tests/unit/motion_matching/test_club_ui_integration.py tests/unit/tools/test_tour_matching_viewer_core.py::test_club_only_compare_view_separates_observed_and_inferred tests/unit/workspace/test_results_browser.py -q -n 0 --no-cov --timeout=120` GREEN after review-gap harden (workbook loader, receipt-hashed ledger, default JSON index, viewer compare). GUI tab test expects 4 tabs including Club-Only.
- Limitations: software-contract UI only; `native_g1_pass` false; blockers
  `native_g1_qualification_requires_desk_native_receipt`,
  `software_contract_ui_integration_is_not_native_evidence`. Predicted club
  frame series remains unavailable until a continuous trajectory package is
  emitted (reason via `tour_matching_viewer.core.club_only_compare_from_ui_result`).
- Next: Confirm CI green on review-gap PR [#10717](https://github.com/D-sorganization/UpstreamDrift/pull/10717) and squash-merge; CO-10 already on main via #10718/#10720.
- 2026-09-22 — Review-gap autofix: full native workbook clock on load; cooperative
  cancel saves checkpoint for resume with frozen session preset; ledger rows
  append to `reports/matched_swing_ledger.json` with repo-relative receipt paths.
- 2026-09-22 — Bugfix: clear club-only checkpoint after successful finish; ledger
  dedup replaces rows for the same receipt path when bytes change.
- Rematch (SELF): merge origin/main after CO-10 #10720; keep review-gap harden unique vs turnover.
- 2026-09-22 — LOD fix: `ClubOnlyUiSession.preset_name()` delegates preset wire name so GUI cancel/resume avoids `session.preset.value` chains.

## MS-105 Reliable Matching Jobs, Recovery and Portable Results (#10379) [MERGED]

- Merged via PR [#10704](https://github.com/D-sorganization/UpstreamDrift/pull/10704).
- Delivered: `src/shared/python/motion_matching/jobs/` — atomic run
  manifests/checkpoints, fault classification, portable packages, both-shell
  progress DTOs; PF-08 budgets with `guarantee=false`.
- Evidence: `docs/plans/matched_swing/evidence/ms105_jobs_recovery.json`.
- Next: N/A — merged; continue CO-09 on main.

## Realtime Pub/Sub Wiring #8869 Handoff

- Workspace: `C:/Users/diete/Repositories/agent-worktrees/pr-10655-local`.
- Branch: `fix/8869-realtime-pubsub-decision`; PR [#10655](https://github.com/D-sorganization/UpstreamDrift/pull/10655) (open). Governing
  issue #8869 (folds in #8868, #8942A). Seam #9406: `realtime` is `split
pending` — UD keeps this facade.
- Entry DL-#8869. **Decision: WIRE, not delete.** Explicit `transport="ws"` /
  `REALTIME_TRANSPORT=ws` routes to `WSPubSub`; other transports raise
  `ValueError` instead of silent file fallback. Renamed colliding
  `register_channel` → `register_channel_hint`; deleted dead `file_pubsub.py`.
- Next action: green CI after post-#10703 merge, squash merge, teardown worktree.

## CO-08 Qualify Club-Only Matrix and Plausibility Tradeoffs (#10612) [MERGED]

- Merged via PR [#10703](https://github.com/D-sorganization/UpstreamDrift/pull/10703) on `main` (`17a0ee033`).
- Schema `club-matrix-qualification/1.0.0`; evidence
  `docs/plans/club_only_matching/evidence/club_matrix_qualification.json`.
- Limitations: software-contract scoring only; no native Fit/G1 claim.
- Next: N/A — merged; continue CO-09 on main.

## GUI Thread-Blocking Migration #8880 [MERGED]

- Merged to main via PR [#10656](https://github.com/D-sorganization/UpstreamDrift/pull/10656).
- Migrated `bunker_shot_gui`, `ball_flight_gui`, and `swing_flight_pipeline`
  onto `src/tools/async_action.py`; added GUI thread-blocking ratchet.
- Next: N/A — merged; remaining un-migrated tools tracked by the ratchet.

## CO-07 Optimize Fast Matching and Expose Candidate Diversity (#10611) [MERGED]

- Merged to main via PR [#10700](https://github.com/D-sorganization/UpstreamDrift/pull/10700)
  (SHA `f9ece7f6a` on this worktree base).
- Delivered: `club_only/fast_matching.py` with fast-preview vs verified-fit
  budgets, immutable target/model/profile cache keys, checkpoint/resume identity,
  cold vs retrieval vs reduced-to-full starts, feasibility-first pruning and
  bounded Pareto diversity, optional empty neural proposal slot, and stage
  profiling including verification time. Schema `club-fast-matching/1.0.0`;
  evidence `docs/plans/club_only_matching/evidence/club_fast_matching.json`.
- Limitations: software-contract scoring only; no native Fit/G1 claim.
- Next: N/A — merged; continue CO-08/CO-09 on main.

## CO-06 Recover Feasible Controls and Independently Replay (#10610) [MERGED]

## Issue #8905: Empty Filter State and Search Debounce

- Branch: cursor/fix-issue-8905-empty-filter-debounce-7996
- PR: not created (pending)
- Governing issue: #8905
- Objective: Fix blank void when model grid filter results are empty; debounce
  search input to avoid costly grid teardown/rebuild on every keystroke.
- Files changed:
  - `src/launchers/launcher_layout_manager.py`: Added `_show_empty_state`,
    `_handle_empty_state_link`, `on_clear_filters` callback, and
    `_empty_state_label` state. Empty state shows centered QLabel with query
    info and "Clear filters" link when active filters produce zero results.
  - `src/launchers/_launcher_top_bar_ui.py`: Added 150ms debounce timer for
    search input via `_search_debounce_timer`, `_on_search_text_changed`, and
    `_apply_debounced_search`.
  - `src/launchers/upstream_drift_launcher.py`: Added `on_clear_filters`
    callback wiring to layout manager and `_on_empty_state_clear_filters` handler.
  - `tests/launchers/test_launcher_layout_manager.py`: Added 5 unit tests for
    empty state visibility, content, removal, and callback behavior.
  - `tests/launchers/test_launcher_ui_setup.py`: Added 3 unit tests for
    debounce timer setup and behavior.
- Validation:
  - `python3 -m pytest tests/launchers/test_launcher_layout_manager.py -v` — 40 passed
  - `python3 -m pytest tests/launchers/test_launcher_ui_setup.py -v -k "debounce or search_text"` — 3 passed
  - `python3 -m ruff check <files>` — all checks passed
  - `python3 -m ruff format --check <files>` — already formatted
- Blockers: None
- Next: Commit, push, and create PR.

## CO-04 Match Club-Only Motion With Double and Triple Pendulums (#10608)

- Merged to main via PR [#10687](https://github.com/D-sorganization/UpstreamDrift/pull/10687)
  (squash merge SHA `f191dd09d`).
- Delivered: `club_only/control_replay.py` recovers minimum-effort controls
  (reuses `ContactForceAllocator` for floating-base plants; reduced software
  plants copy RNEA onto actuated channels), separates net torque / actuated /
  passive / ground / grip / root slack, saves continuous control packages, and
  independently open-loop replays from q0/v0 with no measured-state resets.
  Root slack, infeasible contact, unknown impact, and measured resets reject
  with actionable reasons; kinematic preview stays separate from torque replay.
  Schema `club-control-replay/1.0.0`; evidence
  `docs/plans/club_only_matching/evidence/club_control_replay.json`.
- Validation: `python -m pytest tests/unit/motion_matching/test_club_control_replay.py -q`
  GREEN (12 passed).
- Limitations: software-contract / unit-inertia plant only; native G1 remains
  blocked (`native_g1_qualification_requires_desk_native_receipt`,
  `software_contract_replay_is_not_native_evidence`). No invented native pass.
- Validation addendum: split allocate/replay helpers under function-line budget;
  wrap require() preds in bool() for mypy.
- CI fix: `native_g1_gates.validate_native_g1_claim_contract` shared by
  `control_replay.py` and `replay_package.py` (DRY fingerprint d169d28ac9c7);
  regenerate monolith register + divergence inventory for new club_only modules.
- Merged `origin/main` through MS-51 MyoSuite golfer scene landings and CO-06 #10687.
- Next: N/A — merged; continue CO-07/CO-08 on main.

## CO-04 Match Club-Only Motion With Double and Triple Pendulums (#10608) [MERGED]

- Merged to main via PR [#10680](https://github.com/D-sorganization/UpstreamDrift/pull/10680).
- Delivered: hub-variant IDs + external-work accounting; separate in-plane vs
  original 3D errors; CO-03 seed mapping; cold vs retrieval best-feasible
  retention; first-frame-before-integrate scoring; eight-cell double/triple ×
  four-trial matrix with replay packages and named native blockers (no invented
  G1 pass). Evidence:
  docs/plans/club_only_matching/evidence/club_pendulum_match.json.
- Limitations: software-contract / synthetic fixtures only; TB-05 native triple
  qualification and desk native G1 remain open blockers.

## CO-05 Plausible Upper-Body and Full-Body Candidates (#10609) [MERGED]

- Merged to main via PR [#10681](https://github.com/D-sorganization/UpstreamDrift/pull/10681).
- Delivered: explicit topology maps, local null-space proposals + closure
  reprojection, roster×trial matrix with separated score lanes; schema
  `club-body-candidates/1.0.0`; evidence
  `docs/plans/club_only_matching/evidence/club_body_candidates.json`.
- Limitations: synthetic fixtures for software contracts only; no native G1
  acceptance; missing-runtime cells remain unqualified; kinematic preview
  pending CO-06 replay.

## MS-14 Pinocchio MatchingPlant Full Lane #10333 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10333-ms14`.
- Branch: `feat/10333-pinocchio-matching-plant`; PR [#10684](https://github.com/D-sorganization/UpstreamDrift/pull/10684) open with squash auto-merge armed. Governing issue #10333
  (MS-14, epic #10363). Session `cursor-10333-ms14`.
- Entry DL-#10333. Delivered: PinocchioMatchingPlant derivatives +
  `create_constrained_ik` (Pink), fitter `resolve_fit_native_plant` bridge,
  ConstrainedIkReceipt `closure_residual_m` budget, unit contracts in
  `test_pinocchio_plant.py`, honest blocked receipts under
  `docs/development/full_body_models/evidence/ground_support/anthro_driver_{pinocchio,pink}/`.
- Validation: `pytest tests/unit/motion_matching/pipeline/test_pinocchio_plant.py -q -n 0 --no-cov`
  (contracts green; native tests skip without real Pinocchio); ruff + architecture budget clean.
  CI hygiene (PR #10684): `python -m scripts.shared_tools.divergence_inventory --write`;
  `python -m src.shared.python.motion_matching ledger --write`; unit tests for divergence
  inventory, ledger freshness, and ground-support receipt scan (MS-14 lane schema excluded).
- Limitation: Windows host has no Pinocchio/Pink; receipts are `blocked` /
  `accepted=false` / `native_claims=false`. Do not treat as G1 or green weld closure.
- Rebased: merged origin/main through CO-06 #10687 (`f191dd09d`); kept MS-14 +
  CO-06 HANDOFF/DL rows; regenerated matched_swing status README + divergence
  inventory; blocked native G1 receipts unchanged.
- Next: Confirm quality-gate green + squash auto-merge of PR #10684; ControlTower
  `upstream-motion-runtime` for native pink receipts when scheduled (MS-107 owns G1).

## MS-51 MyoSuite Golfer Scene #10344 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10344-ms51`.
- Branch: `fix/10344-ms51-myosuite-repair`; PR [#10685](https://github.com/D-sorganization/UpstreamDrift/pull/10685) **merged** to main. Governing issue #10344
  (MS-51, epic #10363). Soft dep: merged MS-102 #10376 / PR #10677.
- Entry DL-#10344. Delivered: pinned myo_sim documentation + bootstrap scripts;
  `golfer_scene.generate_golfer_scene` (myobody_simpleupper + club_models + dual
  grip welds + four foot contact markers); coordinate map with explicit omissions;
  inventory `myosuite/driver`+`iron` status `ready` with generated hashes; native
  MuJoCo load/step tests; docs/engines/myosuite.md.
- Validation: `pytest tests/unit/engines/myosuite/test_golfer_scene.py -q -n 0 --no-cov`;
  `python scripts/ci/check_lod.py src/engines/physics_engines/myosuite/python/golfer_scene.py`;
  architecture budget clean on `golfer_scene.py` (generate_golfer_scene ≤100 lines).
- CI repair (SELF): LoD helper for path basename lowercasing; receipt payload
  extracted from generate_golfer_scene; unit tests use defusedxml.ElementTree.
- Honest limits: no G1 success; `parity_budget_qualified=false`; partial map is
  diagnostic only; muscle params are upstream myo_sim (not golf-calibrated);
  contact spheres are contype=0 markers for the shared law.
- Next: not applicable — landed on main via #10685; MS-14 rematch only.

## MS-61 Simscape Topology + Full-Marker Terminal #10348 [MERGED]

- Workspace: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10348-ms61.
- Branch: feat/issue-10348-ms61-simscape-topology; PR [#10676](https://github.com/D-sorganization/UpstreamDrift/pull/10676) **merged** to main. Governing issue #10348 (MS-61, epic #10363). Session b27ccab3-1128-492c-a0fb-001367ea3aa8.
- Entry DL-#10348. Delivered: fail-closed topology classification (reduced_27_no_neck), dual terminal disclosure (full_marker_terminal.py + fit_metrics), acceptance hooks, run-103 blocked native_gate.json + R2025b runtime/parity receipts, runner Fit fail-closed stub.
- Rebased: merged origin/main (includes MS-102 #10677); kept fail-closed topology + blocked native_gate — no invented G1 pass.
- CI fix: regenerated docs/development/matched_swing_program/README.md via python scripts/generate_matched_swing_status.py --write after ledger grew to 103 receipts (Simscape 40).
- Validation: pytest tests/docs/test_matched_swing_status_freshness.py::test_matched_swing_status_section_is_fresh GREEN; prior focused topology suite still authoritative for MS-61 behavior.
- Limitations: full-marker terminal still ~40.3 mm from run-102 source; neck/full-body model work owned by MS-104 (#10378); no invented native G1 pass.
- Next action: Continue under MS-104 (#10378) / next matched-swing dispatch — do not invent native G1 pass.

## MS-102 Engine and Model Inventory #10376 Handoff

- Workspace: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10376-ms102.
- Branch: eat/ms102-engine-model-inventory; PR #10677 **merged** to main. Governing issue #10376
  (MS-102, epic #10363).
- Entry DL-#10376. Delivered: src/config/engine_model_inventory.json (authority-
  derived ledger), src/engines/model_inventory.py (load/reconcile/qualify), unit
  tests, structural receipts under docs/development/matched_swing_program/evidence/ms102/.
- Six flagship engines Ã— driver/iron packaged; JaxSim/putting_green reconciled;
  MyoSuite flagship status
  epair â†’ #10344 (MS-51); Simscape R2025b required.
- Next: native SDK receipts on supported hosts via MS-103 preflight (not claimed by
  structural receipts).

## PF-06 Feasible Force Null Spaces and Torque-Distribution Tradeoffs (#10436)

- Worktree: `Worktrees/UpstreamDrift-10504-pf06-rebase`, branch
  `feat/issue-10436-pf06-feasible-force-nullspace`, DL-#10436.
- Changes:
  - `force_nullspace.py`: Scaled SVD and column-pivoted QR null space
    representations with dynamic rank and contact mode reporting
    (`ForceNullSpace.from_balance`). Added `NullSpaceAnalysis` and
    `validate_null_space` checking condition numbers and residuals
    ($A N = 0$, $A x_p = b$). Implemented `redistribute_trajectory`
    penalizing physical rates and ensuring strict basis sign-change
    invariance across frames. Formulated `explore_torque_tradeoffs`
    generating Pareto alternatives (`baseline_minimum_effort`,
    `conservative_default`, `trail_arm_reduced_50`,
    `trail_arm_reduced_80`, `hard_zero_trail`, `relaxed_minimum_trail`,
    `grip_squeeze_minimized`, `ground_load_regularized`) with per-joint
    torque/power, lead/trail effort, ground COP, grip wrench, and
    explicit SI units. Exported reproducible Pareto tables to JSON and
    CSV. Selected conservative default with mechanical rationale
    (reserve torque margins; no unfounded metabolic/injury claims).
  - `test_force_nullspace.py` and `test_force_nullspace_pf06.py`: unit
    coverage for null-space and tradeoff acceptance criteria.
- Reproduction:
  `pytest tests/unit/motion_matching/test_force_nullspace.py tests/unit/motion_matching/test_force_nullspace_pf06.py -v`.
- Status: rebased onto `origin/main` (includes MS-62); SPEC Â§12 `#10504`
  present; focused PF-06 suites 29 passed; architecture budget and DRY
  duplication gates clean locally.
- Next: Confirm CI green after force-with-lease push; merge closes #10436.

## MS-60 Simscape Run Management #10347 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10347-ms60`.
- Branch: `fix/issue-10347-ms60-run-management`; PR #10669 **merged**. Governing issue #10347 (MS-60, epic #10363).
- Entry DL-#10347 (shipped). Delivered: fail-closed R2025b run manifest, returned-replayâ†’MatchedSwingCandidate converter, `run_simscape_candidate.ps1`, committed run-102 candidate/manifest/playback GIF.
- Next action: DeskComputer second-person replay under 30 minutes when MS-61 Fit is scheduled.

## MS-62 Simscape Coordinate Slice #10349 Handoff

- Workspace: `C:/Users/diete/Repositories/agent-worktrees/issue-10349-local`.
- Branch: `fix/issue-10349-ms-62-local`; PR #10665 open. Governing issue #10349 (MS-62, epic #10363).
- Entry DL-#10349. Delivered: `coordinate_slice.py` (JSON map, kinematic projection, boundary wrenches, CLI), `align_measured_to_model.m` geometry-document workspace overrides, evidence under `evidence/matched/driver_g1_simscape_slice/`.
- Validation: `python -m pytest tests/unit/motion_matching/test_coordinate_slice.py -q` (9 passed); `python scripts/ci/check_architecture_budget.py` clean; `ruff check` clean on touched Python.
- Limitations: kinematic slice only; legacy source `u` is 38-DOF actuated so sliced candidate is kinematic profile; Simscape R2025b native replay and boundary-load acceptance unqualified.
- Next action: drive PR #10665 CI green; Simscape R2025b native replay on DeskComputer.

## MS-52 MyoSuite Kinematic Replay #10345 Handoff

- Workspace: `C:/Users/diete/Repositories/agent-worktrees/issue-10345-local`.
- Branch: `fix/issue-10345-ms-52-local`; PR #10666. Governing issue #10345 (epic #10363).
- Delivered: `retarget.py`, `replay.py`, `golfer_scene.py`, `coordinate_map_anthro.json`, `viz/render_replay.py`; unit/native tests; evidence `evidence/matched/driver_g1_myosuite/{candidate.npz,receipt.json,playback.gif}`.
- Validation: `pytest tests/unit/engines/myosuite/test_retarget.py tests/myosuite/test_replay_native.py -q`; cross-engine registration; ledger/status refresh for 101 receipts.
- Honest limit: placeholder MyoBody MJCF yields diagnostic marker parity only; 15 mm gate deferred to MS-51 scene.
- Next action: merge PR #10666 after rebase CI green; teardown worktree.

## ADR-0046 G2 Re-Point Workbenches at Canonical Layer (#9349)

- Worktree: `_worktrees/UpstreamDrift-pr-10178`, branch `conductor/issue-9349`, PR #10178.
- Status: Stage 2 (G2) closed. Module retirement previously landed under #9348.
  - Both launch-monitor tiles state they are surfaces of "the same analytics engine" in `src/config/models.yaml` (desktop) and `src/config/launcher_manifest.json` (web).
  - Pinned by `tests/config/launcher_manifest/test_launch_monitor_tiles_share_one_engine.py`.
  - ADR-0046 records G2 as landed; capability atlas regenerated.
- Verification: 4 unit tests pass in `test_launch_monitor_tiles_share_one_engine.py`.

## Video Analyzer Real GUI #8883 Handoff

- Workspace: `C:/Users/diete/Repositories/_worktrees/UpstreamDrift-issue-8883`.
- Branch: `fix/8883-video-analyzer-gui`; base `origin/main` @ `4651b8793`;
  implementation SELF; PR #10651. Governing issue #8883
  ("Video Analyzer tile opens a bare 'GUI placeholder' label ... the
  sibling-repo fallback is a dead end"); related #8854 (closed — confirms
  the sibling `src/video_analyzer/` path never exists in this checkout)
  and #10512 (prior "provider required" diagnostic framing, now
  superseded for this tile).
- Entry DL-#8883. Delivered a real, minimal GUI instead of the stopgap
  status-honesty fix:
  - `src/tools/video_analyzer/analyzer.py`: added `_pose_frames_from_video`
    (lazy MediaPipe pose estimation via the
    `src.shared.python.pose_estimation` registry, converted to
    `PoseFrame`/`Landmark`) and `SwingAnalyzer.analyze_video(video_path)`,
    which wires the already-tested head-stability math to a real video
    file. Raises `FileNotFoundError`/`RuntimeError` (with the registry's
    install hint) rather than failing silently.
  - `src/tools/video_analyzer/gui.py`: replaced the static
    `QLabel("Video Analyzer (GUI placeholder)")` with `MainWidget`
    (choose-video button, Analyze button, report pane, status line),
    following the `simulation_backends_launcher` convention — synchronous
    testable core (`set_video_path`, `run_analysis`) plus an async
    entry point (`run_analysis_async`) via `src.tools.async_action` so
    MediaPipe decoding a whole video does not freeze the GUI thread. A
    failed analysis renders an honest error in the status/report panes.
  - `src/launchers/external_tools_adapter.py`: `get_video_analyzer_dockable_ui`
    no longer tries importing `video_analyzer.launch_pyqt6` from the
    (nonexistent) sibling repo; it returns the real window from
    `src.tools.video_analyzer.gui` directly. `_import_video_analyzer` and
    its `_wrap_external_widget`/Tools-repo dependency are gone for this
    tile.
  - `src/launchers/task_launch_truthfulness.py`: `video_analyzer`'s audit
    entry changed from `PROVIDER_REQUIRED` to `PRODUCTION_SOLVER` — it no
    longer needs an external Tools provider; MediaPipe is an optional
    runtime dependency handled by an in-UI error, not a launch blocker.
  - `src/config/models.yaml`: `video_analyzer`'s `status: "ready"` left
    unchanged — it is now honestly ready (real GUI wired to tested math).
  - Tests updated/added: `tests/unit/test_video_analyzer_pipeline.py`
    (new — `analyze_video` success/error paths, MediaPipe mocked),
    `tests/ui/tools/video_analyzer/` (new — headless `MainWidget` tests,
    conftest mirrors `simulation_backends`'), `tests/launchers/test_simulation_guis.py`
    and `tests/launchers/test_task_launch_truthfulness.py` (rewrote the
    two tests that asserted the old provider-required/placeholder
    contract for this tile), `scripts/config/suite_marker_baseline.json`
    (renamed nodeid).
  - SPEC.md Section 12 row added for `#8883`.
- Validation: `ruff check`/`ruff format --check` clean on all changed
  files; `mypy` on the four changed `src/` files reports 0 new errors
  (12 pre-existing errors surface transitively from unrelated imported
  modules — `document_reader.py`, `keypoint_offsets.py`,
  `_shot_tracer_gui.py`, `rtmpose_onnx_estimator.py` — none in the diff);
  `pytest -q tests/unit/test_video_analyzer_pipeline.py
tests/unit/test_video_analyzer_math.py tests/ui/tools/video_analyzer
tests/launchers/test_simulation_guis.py
tests/launchers/test_task_launch_truthfulness.py
tests/config/feature_parity` → all passing (required
  `git submodule update --init vendor/ud-tools` first; it was
  uninitialized in the fresh worktree).
- Limitations (declared, not hidden): MediaPipe itself is not exercised
  end-to-end by these tests (mocked at the registry seam) — no CI runner
  here is asserted to have MediaPipe installed, so a real video was never
  decoded during this work; the head-stability score is the only metric
  wired up (spine tilt / hip turn / shoulder turn on `PostureMetrics`
  stay at their defaults, matching the issue's scope).
- 2026-09-22 remediation (merge of `origin/main`): the DRY gate flagged
  `_apply_theme_best_effort` duplicated between this GUI and
  `simulation_backends_launcher/gui.py` (fingerprint `bec8b6584288`); both
  now call the shared `src/tools/window_theme.apply_theme_best_effort`
  (tests: `tests/tools/test_window_theme.py`).
- Next action: open PR `Closes #8883`, push, poll CI, merge with
  `--auto --squash` once green, then tear down this worktree/branch.

## Bunker Contact Regimes #9544 Handoff

- Workspace: `C:/Users/diete/Repositories/_issue_worktrees/UpstreamDrift-conductor-issue-9544`.
- Branch: `conductor/issue-9544`; base `b455aa4fd` (origin/main); implementation
  SELF; PR #10455. Governing issue #9544 (epic #9541). Pinned Tools
  `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1` (gitlink unchanged; the issue's
  audit pin `3d93bb2câ€¦` predates the current gitlink).
- Entry DL-#9544. Delivered: `bunkershot3d.ball.regimes` (four regimes,
  refusal of every non-splash launch), `RotationMode`/`RotationCoupling` on
  `solvers/shot.py` with wrench conventions, `vandv.conservation`
  support-impulse and driver-work ledger, workbench classification before
  carry, `docs/bunkershot3d/contact-regimes.md` (F2 requirement record).
- Validation: `python3 -m pytest tests/bunkershot3d/ball/test_contact_regimes_9544.py tests/bunkershot3d/solvers/test_rotation_coupling_9544.py`
  (29 passed); `python3 -m pytest tests/bunkershot3d tests/unit/tools/bunker_shot_gui -n auto`
  passes except three pre-existing failures reproduced with the classifier
  bypassed (pyvista `n_faces` in `test_shot_scene_render_vtk.py`; two #9243
  band/budget assertions). `ruff check` and `ruff format --check` clean.
- Limitations (declared, not hidden): F0 does not measure the face-ball sand
  cushion, so regimes are geometric conventions; no measured validation
  exists for any regime; coupled mode ships the boundary and an oracle
  coupling, not a calibrated shaft model; F2/3-D contact remains unbuilt.
- Next action: open PR `Closes #9544`; completion comment must cite merge SHA,
  pin, test results and the limitations above.

## Bunker Transfer Qualification #9543 Handoff

- Workspace: `C:/Users/diete/Repositories/_issue_worktrees/UpstreamDrift-conductor-issue-9543`.
- Branch: `conductor/issue-9543`; base `49d94788a856d4aeeec095ebb2e3625fc724879c`
  (audit baseline in the issue: `1f69a51fce997932f04a6ad1dd95bf4d065ba971`);
  implementation SELF; PR #10457. Pinned Tools tree
  `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`, materialised read-only into the
  worktree's empty `vendor/ud-tools` for the test run; not edited.
- Entry DL-#9543. Objective: the software half of the sand-to-ball transfer
  measurement-to-prediction program (intake â†’ calibration â†’ held-out
  validation â†’ versioned evidence), with the launch verdict floor lifted only
  inside a qualified regime.
- Files: `src/bunkershot3d/ball/qualification.py` (contract, registered matrix
  and protocol, tolerances, evidence object, #9239 disposition),
  `src/bunkershot3d/ball/rig_capability.py` (what the three-camera rig can
  and cannot measure), `src/bunkershot3d/ball/qualification_fit.py` (fit,
  holdout, `qualify`, report), `src/bunkershot3d/ball/splash.py` (qualification-aware
  `launch_verdict`, `compute_ball_launch_from_splash`, provenance),
  `src/bunkershot3d/vandv/validation.py` (`measured_record` on
  `ValidationComparison`), `src/bunkershot3d/sand/provenance.py` and
  `vandv/measurement_intake.py` (`CALIBRATED` basis at rank 1),
  `tests/bunkershot3d/ball/test_transfer_qualification.py`,
  `docs/bunkershot3d/transfer-qualification.md`, SPEC.md row and section.
- Key decisions: no new ledger specs (every spec must be named by a level
  step); launch references are `MeasurementRecord`s against three explicit keys
  in `BALL_LAUNCH_REFERENCE_SPECS`; intake is programmatic because a stroke
  carries the solver's verdict; `spin_lever_arm_fraction` is never fitted;
  the launch model's own status is `WITHIN` inside a qualified regime and the
  solver's verdict still combines via `worst_of`.
- Validation: `python3 -m pytest tests/bunkershot3d/ball tests/bunkershot3d/vandv tests/bunkershot3d/sand tests/bunkershot3d/study tests/bunkershot3d/test_public_api_8608.py -q -n 4 --timeout=120`
  â†’ 1018 passed; scoped `ruff check`, `ruff format --check`, `mypy`,
  `scripts/ci/check_lod.py src/bunkershot3d/ball`, file-size, architecture and
  error-handling ratchet checks pass. Not run: the full suite and pre-push
  `pytest-unit` hook (CI's job).
- Limitations: no measured strokes exist, so no regime is qualified and every
  shipped launch verdict stays `BEYOND_VALIDATION`; the fitted-parameter
  uncertainty is not yet entered into the #9243 ranking budget; `BunkerShotState`
  does not carry a qualification (pass it to `compute_ball_launch_from_splash`).
- Next: open the PR referencing #9543 without `Closes`; the issue stays open
  until strokes measured under `MEASUREMENT_PROTOCOL` produce a qualification.

## Tools MocapSession Consumer Slice (#9422)

Working directory: `C:/Users/diete/Repositories/_issue_worktrees/UpstreamDrift-conductor-issue-9422`.
Branch: `conductor/issue-9422`. Commit: SELF. PR: #10466.
Development-log entry: DL-#9422. Governing issue: #9422 (readiness P6, M-track
consumer of Tools #4706); objective: route UpstreamDrift capture sessions
through the one canonical Tools `MocapSession` contract instead of probing it
as absent.

Completed: `src/motion_capture/rig/tools_bridge.py` probes the pinned Tools
family (`shared.python.sidekick.lab.mocap`), pins `mocap-session/1.0.0`
(another version is `incompatible`), and `export_session_manifest` /
`export_to_bundle` build a `MocapSessionManifest` through the Tools builders
and `dumps_canonical`. `capture` and `record` write `mocap_session.json` beside
`session_manifest.json` and record the outcome under `tools_schema.export`;
`record --consent-recorded` is the recorded-consent term the Tools policy
requires for retained raw video, otherwise the export is `rejected` as data.
Docs: `docs/motion_capture/capture_rig.md` (Tools Schema Bridge), SPEC row.

Validation: `python3 tests/fixtures/mocap_session_export/run_checks.py` â€” 8
passed (Tools family resolved first, like the capture-rig worker checks);
`python3 -m pytest tests/motion_capture/rig/test_tools_session_export.py
tests/motion_capture/rig/test_recorder_bridge_cli.py
tests/motion_capture/rig/test_bundle_record.py
tests/unit/repo_hygiene/test_vendored_tools_fallback.py` â€” 40 passed. Known,
pre-existing on this workstation: `python3 -I` subprocess checks
(`test_reference_calibration_worker.py` and the new integration test) fail
with `No module named 'numpy'` because numpy lives in the user site; CI has
it in the system site. Pre-existing and unrelated:
`tests/architecture/test_markerless_mocap_authority.py::test_spec_and_handoff_point_to_the_current_program`
fails on `AGENT_HANDOFF.md` length (993 > 150).

Constraints: the root test process keeps `sidekick.lab.mocap` unresolvable
by design (`src/__init__.py`, UD's own Sidekick cached first); the ready path
is therefore checked in a Tools-first process, not by building a hybrid
package. D-track (Tools #4707; D3 #4717 open) and prerequisites #8865/#8866/
#8867 stay open â€” #9422 is not closed by this slice.

Next steps: (1) open the PR for this branch; (2) route the C3D upload path
(#8865) through the same pinned contract; (3) surface `tools_schema.export`
in the capture-rig UI as a disabled-reason, not a hidden failure.

## PF-04 Qualify Contact Modes and Native Pinocchio Force Feasibility (#10434)

- Branch: `feat/issue-10434-pf04-qualify-contact-modes-pinocchio-forces`, PR #10499 (auto-merge armed), lease `antigravity-ud-10434`, DL-#10434.
- Changes:
  - `contact_mode_qualifier.py`: Infer heel/toe support modes (`FLAT`, `HEEL_ONLY`, `TOE_ONLY`, `FLIGHT`) and whole-body support states with clearance and velocity hysteresis; support mode ambiguity metric; COP and 2D convex hull support polygon containment under arbitrary surface normal n_hat; slip speed thresholding and friction cone saturation ratio; constitutive Hunt-Crossley compliance comparison (`sphere_ground_contact`) vs inverse dynamics force allocation; separate linear force (N) and moment (N\*m) residual budgets; unphysical load rejection (> 5000 N, > 300 Nm); and mass/geometry/friction sensitivity reporting.
- Reproduction: `pytest tests/unit/motion_matching/test_contact_mode_qualifier_pf04.py`.
- Next: Land PR #10499 via CI and proceed to PF-05.

## MS-16 MuJoCo Native IK and MJ_Inverse Tracking (#10366)

- Worktree: `Worktrees/UpstreamDrift-10660-land`, PR branch `fix/issue-10366-ms-16-mujoco-native-tools-marker-ik-on-m-cursor-composer-local`, lease `claim:local` session `local-10660-land-ci`, DL-#10366, PR #10660.
- Tip: on latest main; finite-bounds gate for `minimize.least_squares`; weld-aware `mj_inverse` audit; `ShootingFitConfig` keeps architecture budget; `_persist_dynamics_artifacts` keeps `_simulate_and_receipt` under function-lines; refreshed `reports/matched_swing_ledger.json` (101 receipts) after PF-06 merge so `test_ledger_freshness` passes.
- Reproduction: `pytest tests/unit/motion_matching/test_mujoco_ik_minimize.py tests/unit/motion_matching/test_mujoco_mj_inverse.py tests/unit/motion_matching/test_ledger.py::test_ledger_freshness -q`
- Next: Confirm unit-test-gate / quality-gate green; squash auto-merge lands; close duplicate #10662.

## PF-03 Enforce Contact, Actuator and Root Constraints in Force Allocation (#10433)

- Branch: `feat/issue-10433-pf03-contact-actuator-root-constraints`, PR #10498 (auto-merge armed), lease `antigravity-ud-10433`, DL-#10433.
- Changes:
  - `src/shared/python/motion_matching/contact_force_allocator.py`:
    - Replaced penalty-augmented least squares and unconstrained post-projection with constrained QP inverse dynamics.
    - Added `FeasibilityStatus` enum and `AllocationObjective.HARD_ZERO_TRAIL`.
    - Enforced 8-faceted polyhedral friction pyramid and non-negative normal force (f_n >= 0) along arbitrary surface normals.
    - Enforced contact separation mask (f_s = 0 for separated contacts).
    - Enforced strict actuator bounds without post-projection, using isolated actuator slack to detect and report violations without bounds breaching.
    - Isolated diagnostic root slack delta_tau_root so phantom root forces never produce false physical success.
    - Added `verify_torque_and_rate_bounds` for discrete trajectory limits and rate verification.
  - `tests/unit/motion_matching/test_contact_force_allocator_pf03.py`: Comprehensive test suite for all 10 acceptance scenarios.
- Validation: 12 unit tests pass 100% across PF-03 and legacy suites; ruff clean; black clean; mypy strict clean; bandit clean.
- Next: Land PR #10498 via CI and proceed to PF-04.

## PF-10 Connect Qualified Matching Strategies to Engine Feature Contracts (#10440)

- Worktree: primary, branch `feat/issue-10440-pf10-matching-strategies-contracts`, lease `antigravity-ud-10440`, DL-#10440.
- Changes:
  - `matching_strategy.py`: Implemented versioned strategy schema (`STRATEGY_SCHEMA_VERSION = "matched-strategy-v1"`), stage-separated qualification matrix tracking all 6 stages (`model_available`, `kinematic_fit`, `force_feasible`, `replay_accepted`, `runtime_budget_met`, `muscle_qualified`), strategy presets, controller specifications, and contact reaction history containers.
  - `CandidateStrategyPackage`: Bound candidate trajectories with strategy metadata, accelerations, and contact reactions. Enforced fail-closed name-permuted coordinate remapping and lossless .npz serialization without pickle.
  - `StrategyComparisonService`: Cross-strategy comparison service reporting torque profiles, kinematics/closure errors, and capability auditing invalidating supported status on missing SDKs.
  - Test suites: 8 unit tests in `test_matching_strategy.py` verifying stage ordering, acceptance invariants, contract serialization, .npz roundtrip, name-permuted remapping, comparison service, capability invalidation, and 6-engine / dual-club contract coverage.
- Reproduction: `pytest tests/unit/motion_matching/test_matching_strategy.py -v`.
- Next: PR auto-merge, complete lease on #10440, claim next issue.
  > > > > > > > origin/main

## Coupled Grip, Shaft, and Ground Rollup Handoff Checkpoint (#8684) â€” 2026-09-11

- Worktree: C:/Users/diete/Repositories/\_issue_worktrees/UpstreamDrift-conductor-issue-8684.
- Branch: conductor/issue-8684; checkpoint SELF; PR #9998. Parent #8668.
- Objective: close parent research issue #8684 by rolling up the executed child tiers (#8685 grip discretization, #8797 friction/events, #8715 shaft, #8723 ground) against its four registered questions. No new solver, atlas, tolerance, or golden file; the required design was delivered by the children.
- Changed: `COMPREHENSIVE_RESEARCH_PROGRAM.md` (child summaries plus the four-question rollup and promotion boundary), four new tier rows in `MODEL_COMPLETION_FALSIFICATION_MATRIX.md`, DL-#8684, and the re-pinned `release_manifest.json`, `CHECKSUMS.sha256`, `claim_evidence_manifest.json` via `qualify_open_release write`.
- Validation: `qualify_open_release validate`, `claim_evidence_integrity validate`, `tests/research/test_proximal_distal_release_bundle.py`, `tests/unit/research/test_proximal_distal_claim_evidence_integrity.py`, `test_proximal_distal_terminology_contract.py`, doc size budget, and the development-log validator; outcomes recorded in the PR body.
- Open: stateful friction, calibrated equipment, unilateral foot contact, uncertainty crossed with shaft/ground, the frozen #9306 smoke; human promotion stays blocked on #8556. Peer handoffs below are preserved.

## PF-09 Replace Synthetic Force Adapters With Native Bridges (#10439)

- Worktree: primary, branch `feat/issue-10439-pf09-native-force-bridges`, lease `antigravity-ud-10439`, DL-#10439.
- Changes:
  - `multi_engine_torque_allocator.py`: Extended `BaseEngineForceAdapter` protocol with `model_hash`, `coordinate_order`, `contact_names`, and `compute_mass_and_bias`. Quarantined `_AnalyticalMultibodyBase` as `SyntheticMultibodyFixture` requiring explicit `allow_synthetic=True`. Enforced fail-closed `RuntimeError` in `create_engine_force_adapter` for unbridged native engines (Drake, OpenSim, Simscape) in production mode. Corrected `MujocoForceAdapter` to compute raw unconstrained dynamics M a + bias eliminating `qfrc_inverse` passive/constraint force double counting, added input validation (`_checked_vector`) and state refresh (`_prepare_state`) before mutation, and verified exact acceleration parity.
  - `force_adapter.py` & `native_model.py` (Pinocchio): Implemented `PinocchioForceAdapter` implementing the shared interface with fresh constraint kinematics refresh (`_refresh_constraint_data` and `closure_force_jacobian`).
  - `allocate_swing_torques.py`: Extended CLI choices to include `pinocchio` and added `--allow-synthetic` flag to gate quarantined fixtures.
  - Test suites: Added `test_native_force_equations.py` (MuJoCo raw equation checks), `test_force_adapter.py` & `test_force_mapping.py` (Pinocchio integration), and `test_force_bridges_pf09.py` (quarantine enforcement, protocol conformance, CLI synthetic gate).
- Reproduction: `pytest tests/unit/motion_matching/test_native_force_equations.py tests/unit/motion_matching/test_multi_engine_torque_allocator.py tests/unit/motion_matching/test_force_bridges_pf09.py -m "requires_mujoco or unit" -v`.
- Next: PR auto-merge, release lease on #10439, claim next issue in sequence (#10440: PF-10).

## Club-Only Observation Contracts CO-01 #10605 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10605-co01`.
- Branch: `fix/issue-10605-co01-club-observation`; PR [#10670](https://github.com/D-sorganization/UpstreamDrift/pull/10670) merged. Governing issue #10605 (CO-01, epic #10602).
- Entry DL-#10605 shipped. Delivered: `ClubObservation`, `club_calibration.py`, legacy adapters, four-trial fixture pack.
- Next action: superseded by CO-02 #10606.

## Club-Only Pendulum Matching CO-04 #10608 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10608-co04`.
- Branch: `feat/issue-10608-co04-pendulum-club-match`; PR [#10680](https://github.com/D-sorganization/UpstreamDrift/pull/10680) open (squash auto-merge). Governing issue #10608 (CO-04, epic #10602). Prerequisites CO-02 #10675 and CO-03 #10678 merged on main. Head SELF.
- Entry DL-#10608. Delivered: `hub_accounting` (fixed-pivot vs prescribed moving-hub IDs + external work), `match_errors` (separate in-plane/3D RMSE), `pendulum_match` (double/triple fit, first-frame before step, cold vs retrieval retention, reconstruction reject), `match_matrix`/`replay_package` (eight-cell software matrix with named native blockers), evidence `club_pendulum_match.json`. Fit lives under pendulum engines package; helpers split for architecture budget; hub track assignment typed for mypy.
- Validation: `python -m pytest tests/unit/motion_matching/test_club_pendulum_match.py -q -n 0 --no-cov --timeout=120` (10 passed); `ruff check` clean on touched Python; architecture budget + dependency direction OK.
- Limitations: software-contract fits on synthetic CO-01 fixtures only; native G1 remains blocked with named gates; not CO-08 scientific qualification.
- Next action: confirm CI green + squash merge of #10680; then dispatch CO-05 #10609 (do not start in this PR).

## Club-Only Starting Guesses CO-03 #10607 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10607-co03`.
- Branch: `feat/10607-co03-retrieval-constrained-ik`; PR [#10678](https://github.com/D-sorganization/UpstreamDrift/pull/10678) merged. Governing issue #10607 (CO-03, epic #10602).
- Entry DL-#10607 shipped. Delivered: hand geometry, retrieval, constrained IK, seed cache, evidence.
- Next action: superseded by CO-04 #10608.

## Club-Only Plausibility Priors CO-02 #10606 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-co02-10606`.
- Branch: `feat/co02-golf-plausibility-priors`; PR [#10675](https://github.com/D-sorganization/UpstreamDrift/pull/10675) merged. Governing issue #10606 (CO-02, epic #10602).
- Entry DL-#10606 shipped. Delivered: priors, profiles, ambiguity, club-only acceptance.
- Next action: superseded by CO-03 #10607.

## Neural Forward Surrogates and Physics-Structured Alternatives NM-07 #10622 Handoff [IN PROGRESS]

- Workspace: `C:/Users/diete/Repositories/UpstreamDrift`.
- Branch: `feat/nm07-forward-surrogates-10622`; PR #10622 in flight. Governing issue #10622 (NM-07, epic #10603). Entry DL-#10622 in_progress.
- Delivered:
  - `src/shared/python/neural_motion/surrogates/` (`neural-surrogate-comparison/1.0.0`): `SurrogateCandidateKind`, `SurrogateComparisonConfig`, `SurrogateAblationResult`, `SurrogateComparisonReport`.
  - `PhysicsStructuredSurrogate`: analytical rigid polynomial prior + bounded residual dynamics with trust-region and contact-boundary awareness.
  - `compare_surrogates_and_alternatives`: comparative benchmark evaluating unconstrained forward surrogate inversion (flagged for adversarial exploitation risk), hybrid polish, physics-structured residual dynamics, masked proposal, and diffusion fallback (rejected due to >500 ms latency and sample inefficiency).
  - `src/shared/python/motion_matching/surrogate/validate.py`: real-clock timegrid resampling (`resample_to_timegrid`), sign-invariant SO(3) geodesic distance (`quaternion_geodesic_error_rad`), trust-region verification, directional derivative / cosine-similarity gradient fidelity check, and contact-boundary failure rejection.
  - `src/shared/python/motion_matching/surrogate/nm07_comparison.py`: discovery module.
  - Evidence: `docs/plans/neural_motion_matching/evidence/nm07_forward_surrogates_receipt.json`; schema `neural-surrogate-comparison-receipt/1.0.0`.
  - Documentation: `docs/plans/neural_motion_matching/forward_surrogates.md`.
- Validation:
  ```powershell
  python -m pytest tests/unit/motion_matching/test_forward_surrogates_nm07.py tests/unit/neural_motion/test_forward_surrogates_nm07.py tests/unit/neural_motion/test_surrogate_nm07_discovery.py tests/unit/motion_matching/test_surrogate_validate.py tests/unit/motion_matching/test_hybrid.py -q -n 0 --no-cov --timeout=60
  ```
  All tests passed; architecture budget OK; ruff check & format OK; spec changelog duplicates check OK.
- Limitations: Software contracts and synthetic fixtures only; no fabricated native acceleration or training success claims; contact boundaries fail closed for smooth rigid models without hybrid collision handling.
- Next action: Hand off to NM-08 (#10623): Closed-Loop Tracking vs Native Tracking Baseline under epic #10603.

## Neural Masked Proposals NM-06 #10621 Handoff [MERGED]

- Workspace: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10621-nm06.
- Branch: feat/10621-nm06-masked-proposals; PR [#10709](https://github.com/D-sorganization/UpstreamDrift/pull/10709)
  **merged** (`08bcec302`; tip `05f712281`). Governing issue #10621 (NM-06,
  epic #10603). Entry DL-#10621 shipped.
- Delivered: `src/shared/python/neural_motion/proposals/` (`neural-masked-proposals/1.0.0`) binding TaskDimensions.u_dim / MaskedTrajectoryTask, masked conditioning, selection + mixture ablation, observation/regularization training (not coeff MSE alone), hybrid fail-closed polish_fn refine, strict checkpoint mismatch rejection. Inverse reuse: masked_proposal.py, proposal_shared.py, proposal_training.py, collapse.py, basis_time.py; regressor_training stays on TrainingConfig + epoch helpers (not flat kwargs).
- Gate repairs: merge origin/main (CO-09 #10711); extract `motion_matching.inverse.proposal_shared` for DRY; regenerate divergence inventory with `--repo-root .` for NM-06 ud-only paths; keep architecture budgets + optional-torch unit-lane path; do not raise DRY baseline max.
- Validation: NM-06 pytest green with torch; software-contract fixtures only.
- Next action: N/A — merged. Check NM-07 #10622 `claim:antigravity` before any start; do not steal.

## Neural Dynamics Baselines NM-05 #10620 Handoff

- PR [#10701](https://github.com/D-sorganization/UpstreamDrift/pull/10701) **merged**.
  Governing issue #10620 (NM-05, epic #10603). Entry DL-#10620 shipped.
- Delivered: `src/shared/python/neural_motion/baselines/` (`neural-dynamics-baselines/1.0.0`).
- Next action: superseded by NM-06 dispatch.

## Neural Teacher Episodes NM-04 #10619 Handoff

- PR [#10698](https://github.com/D-sorganization/UpstreamDrift/pull/10698) **merged**. Governing issue #10619 (NM-04, epic #10603). Entry DL-#10619 shipped.
- Delivered: `src/shared/python/neural_motion/teachers/`; schemas `neural-teacher-episodes/1.0.0` and `neural-acquisition-log/1.0.0`.
- Next action: superseded by NM-05 dispatch.

## Neural Episode Storage NM-03 #10618 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-local-10618`.
- Branch: `feat/10618-nm03-episode-storage`; PR [#10686](https://github.com/D-sorganization/UpstreamDrift/pull/10686) **merged** (`800703cb`). Governing issue #10618 (NM-03, epic #10603). Entry DL-#10618 shipped.
- Delivered: `src/shared/python/neural_motion/episodes/` (`neural-episode-store/1.0.0`); episode_storage.md + receipt.
- Limitations: synthetic software-contract tests only.
- Next action: superseded by NM-04 dispatch.

## Neural Dataset Labels NM-02 #10617 Handoff

- Workspace: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10617-nm02` (sole NM-02 worktree).
- Branch: `fix/10617-nm02-native-dataset-labels`; PR [#10679](https://github.com/D-sorganization/UpstreamDrift/pull/10679) **merged** (`4bb10daa0`). Governing issue #10617 (NM-02, epic #10603). Entry DL-#10617 shipped.
- Delivered: channel evidence (no zero-as-measurement), native vs interval accelerations, requested/applied controls, `ModelDoFLayout`, restore `StateError`, residual helper, first-wave mock + ODE adapters and receipts.
- CI unblock (SELF): split `_finalize_channels` under architecture function-lines budget; fix `MockPhysicsEngine.set_control` mypy; DRY helpers `_first_step_native_residual`, `_residual_norm`, `_single_sample_dynamics_config` (fingerprint `2430854b10cc`); SPEC §12 row keyed `#10679`; regenerate divergence inventory for `channel_finalize` / `sim_buffers` / `sim_recording` after core.py split.
- Main sync (SELF): after CI Standard green on `4f8de75bf`, PR went DIRTY; merged `origin/main` and kept NM-02 inventory totals (`ud-only` 1415) plus the three generator-split modules.
- Validation: adapters.py DRY scan clean for `2430854b10cc`; NM-02 unit tests green locally; `test_committed_inventory_is_current_when_vendor_present` green after inventory refresh; prior tip CI Standard SUCCESS.
- Limitations: software + ODE residual only; no training/speed claims; other engines deferred to NM-09.
- Next action: superseded by NM-03 dispatch.

## Club-Only and Neural Matching Planning (2026-09-20)

- **Club-Only Epic:** [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602); CO-00..CO-10 baseline shipped (#10667–#10718); survivor follow-up PR [#10720](https://github.com/D-sorganization/UpstreamDrift/pull/10720) rematching unique runnable-command fixes.
- **Neural Epic:** [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603); NM-00..NM-06 shipped (#10668–#10709); next candidate NM-07 #10622 only after claim check (`claim:antigravity` — do not steal). Remaining NM-08..NM-12 (#10623–#10627).
- **Read:** [Shared Review](../plans/club_neural_review/REVIEW.md); [Club-Only Turnover](../plans/club_only_matching/TURNOVER.md); [Neural Turnover](../plans/neural_motion_matching/TURNOVER.md); [NM-00 Artifact Audit](../plans/neural_motion_matching/artifact_audit.md); [NM-01 Learning Freeze](../plans/neural_motion_matching/learning_freeze.md); [NM-02 Dataset Labels](../plans/neural_motion_matching/dataset_labels.md); [NM-03 Episode Storage](../plans/neural_motion_matching/episode_storage.md); [NM-04 Teacher Episodes](../plans/neural_motion_matching/teacher_episodes.md); [NM-05 Dynamics Baselines](../plans/neural_motion_matching/dynamics_baselines.md); [NM-06 Masked Proposals](../plans/neural_motion_matching/masked_proposals.md).
- **CO-10 state:** baseline MERGED via [#10718](https://github.com/D-sorganization/UpstreamDrift/pull/10718) (`2f6e119ef`); survivor [#10720](https://github.com/D-sorganization/UpstreamDrift/pull/10720) rematch in flight. Duplicate #10719 stays closed.
- **NM-06 state:** MERGED via PR [#10709](https://github.com/D-sorganization/UpstreamDrift/pull/10709) (`08bcec302`); DL-#10621 — do not touch.
- **Next:** Land rematched #10720; then check NM-07 claim before any neural work.

## BunkerShot3D Product Acceptance Matrix (Epic #9541)

Working directory:
`C:/Users/diete/Repositories/_issue_worktrees/UpstreamDrift-conductor-issue-9541`.
Branch: `conductor/issue-9541`. Base HEAD `9af408974f1bb1a738461875edafe5defee2260a`;
implementation SELF; PR #10459.
Development-log entry: DL-#9541. Tools pin `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`
(the worktree's `vendor/ud-tools` was empty and was materialized read-only from
the main checkout's submodule objects for testing; nothing under `vendor/` is
edited).

Added `src/config/bunkershot3d_qualification.json`, the epic's product
acceptance matrix, in the #9539 readiness-ledger schema so
`src/config/industrial_readiness_loader.py` is reused without change. All
sixteen checklist children are open with owner, dependency order and a
narrow RED/GREEN plan; eight acceptance criteria carry their blockers;
`release_status` is `blocked`. The gate in
`tests/config/bunkershot3d_qualification/` additionally binds `release_status`
to the live `shipped_register()` (zero measurements) and
`credibility_assessment()` (validation 0 of 4), so the matrix cannot be greened
by editing JSON. `CLAUDE.md` documents the update-on-land rule.

Validation: `py -3.12 -m pytest tests/config/bunkershot3d_qualification tests/config/industrial_readiness -p no:randomly` â†’ 37 passed; `ruff check` and `ruff format --check` clean on the new test.
Not done and not claimed: no physics, calibration, GUI or API change; no
measurement, rendered evidence or independent review. The tool remains an
exploratory simulator.

Next: U1 (#9286) and U2 (#9542) first, per the matrix's dependency order; each
landing PR records its merge SHA and proving test in its entry.

---

Updated 2026-09-18. Governing epic #10363; documentation review branch
`docs/matching-agent-continuation`; commit SELF; PR https://github.com/D-sorganization/UpstreamDrift/pull/10393.
Review workspace: `C:/Users/diete/Repositories/_codex_worktrees/upstream-matching-handoff`.
Development-log entry: `DL-#10363`.

OpenSim golf-model improvement epic #10394 now has nine children (#10395â€“#10403).

## Impact Zone Epic #9546: Readiness Index (2026-09-18)

Branch `conductor/issue-9546`, commit SELF, PR #10446. Worktree
`_issue_worktrees/UpstreamDrift-conductor-issue-9546`; `vendor/ud-tools`
materialised read-only from the main checkout at pin `62e8cdbf9`.

- `src/config/impact_zone_readiness.json` + `docs/operations/impact-zone-readiness-index.md`
  reconcile I1-I4, #9484 and #9349 against `5347cba0f` under the existing
  loader contract (`I<n>`/`R<n>` keys admitted; generator renders one index
  per ledger). `release_status: blocked`.
- `tests/shared_contracts/test_impact_interval_provider.py` consumes Tools
  #5088/#5079 (I1/I2) through the vendored solver on the audit probe. RED at
  the audit pin `3d93bb2c` (names absent, `ceil` budget and
  `unilateral_release = max(0, residual)` present); GREEN at `62e8cdbf9`.
- Validation: `pytest tests/config/industrial_readiness tests/scripts/test_declared_route_producers.py`
  45 passed; `pytest tests/shared_contracts/test_impact_interval_provider.py --tools-mode=vendored`
  3 passed (Windows, Python 3.13); `generate_industrial_readiness_index --check` OK;
  `check_spec_changelog_duplicates` OK; development-log validator OK.
- Open: I3 (#9549) needs the Tools #4946 run-record seam before anything
  else moves; I4 (#9550) waits on I3; Tools half of #9349 (Impact Explorer
  tab) is not re-pointed. Next: bump the pin when #4946 lands and mark
  I1/I2/I3 in the ledger with merge SHAs, tests and acceptance evidence.

OpenSim golf-model improvement epic #10394 now has nine children (#10395–#10403).
Start with [the golf-model assignment](opensim_tour_matching/GOLF_MODEL_AGENT_PROMPT.md)
and [detailed epic](opensim_tour_matching/EPIC_GOLF_MODEL.md) for the missing club,
arm scaling, address alignment and muscle/tendon extension work.

The current Pinocchio/OpenSim native jobs were running at the 01:45 UTC
snapshot. No solver was launched, stopped or accepted by this review. Read
[the bounded agent prompt](matched_swing_program/AGENT_CONTINUATION_PROMPT.md),
[Pinocchio turnover](matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md)
and [OpenSim handoff](opensim_tour_matching/HANDOFF.md). They contain exact
source identities, live job/output paths, saved checkpoints, recovery evidence,
failed gates, test results and escalation rules. Pinocchio native source is
not the merged main scaffold; OpenSim local work contains owner changes.
Preserve them. Review validation: 14 Pinocchio pure tests and 27 OpenSim ladder
tests passed; native acceptance was not rerun. Source hashes match deployed
files after line-ending normalization. Recovery archive is a selected subset.

Next action: inspect the existing native jobs and retrieve newly completed
receipts using the lane handoff before considering another fit.

The historical Simscape continuation below remains useful for that lane;
it is not the latest Pinocchio/OpenSim state.

## PF-02 Calibrate and Smooth Full-Swing Pinocchio Kinematics With Exact Grip Compatibility (#10432)

- Branch: `feat/issue-10432-pf02-pinocchio-kinematics-grip-calibration`, PR #10497 (auto-merge armed), lease `antigravity-ud-10432`, DL-#10432.
- Changes:
  - `marker_kinematics.py`: Added `SolveDiagnostics` with projected gradient norm, cost decrease, active bounds count, and convergence metrics. Added `solve_frame_multi_start` with unconstrained geometric floor estimation. Added `refine_overlapping_window` with triangular window blending and temporal continuity.
  - `kinematic_smoothing.py`: Implemented zero-phase Butterworth smoothing with analytical/numerical derivative compatibility (q_dot â‰ˆ v and v_dot â‰ˆ a), `BoundarySpikeAudit` for boundary jerk and acceleration jump detection, and `audit_cutoff_sensitivity`.
  - Addressed backlog items:
    - MM-2 (#10104): Physiological wrist range of motion compliance producing 0 violations.
    - MM-5 (#10107): Left elbow pit up-and-inward address verification (`dot(pit, up) > 0.3` and `dot(pit, inward) > 0.3`).
    - Separate calibration provenance for driver and 7-iron enforced fail-closed.
- Reproduction: `pytest tests/unit/motion_matching/test_pinocchio_kinematics_calibration.py`.
- Next: Land PR #10497 with auto-merge enabled, proceed to PF-03.

## MS-107 Qualify Crocoddyl Full-Body Fit & Analytic Pelvis Yaw (#10381)

- Branch: `feat/10381-crocoddyl-pelvis-yaw-g1`, PR #10599 (auto-merge armed), DL-#10381.
- Changes:
  - `src/engines/physics_engines/pinocchio/python/crocoddyl_problem.py`:
    - Added non-negative `pelvis_yaw: float = 0.0` to `FitWeights` with contract validation.
    - Added `waist_indices` property on `MarkerTargets` resolving `WaistLeft` and `WaistRight` marker indices or `(-1, -1)`.
  - `src/engines/physics_engines/pinocchio/python/crocoddyl_action.py`:
    - Integrated `compute_pelvis_yaw_residual_and_derivative` into `_NodeCost` (`__init__`, `value`, `gradient_hessian`).
    - Evaluates 2-component unit vector difference residual $r_{yaw} = w_{yaw} \cdot (\hat{u} - \hat{u}_{tgt})$.
    - Adds configuration gradient contribution $J_{yaw}^T r_{yaw}$ and Gauss-Newton Hessian contribution $J_{yaw}^T J_{yaw}$.
    - Passed waist indices from `targets.waist_indices` into `ImplicitEulerAction` and `TerminalAction`.
  - `src/engines/physics_engines/pinocchio/python/full_body_fit.py`:
    - Added `--pelvis-yaw-weight` CLI argument forwarding to `FitWeights`.
    - Updated `cost_breakdown` to calculate and record `"pelvis_yaw"` in execution receipts.
  - `tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py`:
    - 4 unit tests verifying waist marker indexing, zero residual/gradient when aligned, central-difference gradient match, positive semi-definite Gauss-Newton Hessian, and no-op behavior when weight is zero or markers missing.
  - `SPEC.md` & `docs/development/DEVELOPMENT_LOG.md`:
    - Synchronized specification and active development log entry.
- Reproduction: `python -m pytest tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py tests/unit/motion_matching/test_crocoddyl_action.py tests/unit/motion_matching/test_crocoddyl_problem.py -v`.
- Next: Land PR #10599 with auto-merge enabled.

## PF-01 Freeze Fast-Matching Evidence, Schemas and Negative Acceptance Fixtures (#10431)

- Branch: `feat/issue-10431-pf01-fast-matching-evidence-schemas-fixtures`, PR #10495 (merged into main), lease `antigravity-ud-10431`, DL-#10431.
- Summary:
  - Preserved existing rejected driver/iron fast-matching receipts and candidate artifacts (`evidence/matched/`).
  - Extended `MatchedSwingCandidate` schema, `CandidateAuxiliary` (`root_forces`, `contact_modes`, `grip_wrench`), `CandidateMetadata` (`solver_status`, `handedness`, `name_maps`), checksum calculation, and NPZ conversion logic in `candidate_convert.py`.
  - Truthfully renamed `AllocationObjective.MINIMUM_TRAIL_ARM` with backwards-compatible `trail_zero` parsing, added `HARD_ZERO_TRAIL` mode enforcing exact zero trail arm torques, enforced actuator bounds clipping post-solve, and evaluated Coulomb friction cone ratios in `ContactForceAllocator`.
  - Updated `SwingEvaluator` to avoid fabricating impact phases without declared `t_events`, audit closure translation and rotation separately (`ClosureAudit`), and return `NaN` RMSE for empty marker populations.
  - Added 7 negative acceptance fixtures in `test_acceptance.py` and corresponding evaluators in `acceptance.py`: (1) friction cone violation, (2) torque bound overwrite, (3) missing root histories, (4) 44 vs 41 coordinate dimension mismatch, (5) missing club coverage / empty population, (6) truncated horizon duration, and (7) synthetic engine false qualification.
- Validation: 46 unit tests pass (`pytest tests/unit/motion_matching/test_candidate.py tests/unit/motion_matching/test_contact_force_allocator.py tests/unit/motion_matching/test_swing_evaluator.py tests/unit/motion_matching/test_acceptance.py`), ruff clean, black clean.
- Next: Landed in main via PR #10495.

## ORG-24 Reconcile, Audit, and Freeze Feature Preservation Across All Historical Boundaries (#10533)

- Branch: `feat/issue-10533-org24-feature-preservation-audit`, lease `antigravity-ud-10533`, DL-#10533.
- Changes:
  - `src/shared/python/workspace/feature_preservation_audit.py`: Implemented `FeaturePreservationAuditor`, `AuditReport`, `AuditSectionResult`, `AuditStatus`, and `AuditFailureError`.
  - Reconciled all 159 baseline capabilities from ORG-01 with zero unaccounted or silently removed entries.
  - Verified 13 golden preservation fixtures byte-for-byte against recorded SHA-256 and size baselines.
  - Verified acyclic transitive resolution for legacy aliases (`starting_pose_matcher` -> `motion_target_preview`, `putting_green_gui` -> `putting_green`).
  - Audited 5 core workspaces (`simulation`, `analysis`, `capture`, `putting`, `training` + `governance`) ensuring non-empty capability allocations and strict contract enforcement.
  - Verified honest engine qualifications and external dependencies (#10351, #10353).
  - Published final frozen feature preservation audit disposition report at `docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md`.
  - `src/shared/python/workspace/__init__.py`: Exported public auditor and audit types.
  - `tests/integration/test_feature_preservation_audit.py`: Comprehensive test suite with 4 RED failure cases and 7 GREEN acceptance cases.
- Reproduction: `python -m pytest tests/integration/test_feature_preservation_audit.py --timeout=60`.
- Next: Open PR with auto-merge, complete lease on #10533, and proceed to close Epic #10508 when all child PRs are merged.

## ORG-23 Validate and Accept Every Enabled Recommended Task Journey Across Shipped Surfaces (#10532)

- Branch: `feat/issue-10532-org23-installed-workspace-journeys`, lease `antigravity-ud-10532`, DL-#10532.
- Changes:
  - `src/shared/python/workspace/installed_journeys.py`: Implemented `InstalledWorkspaceJourneysCoordinator`, `JourneyExecutionResult`, `FitJourneyResult`, `ComparisonJourneyResult`, `OptimizationJourneyResult`, `EstimationComparisonJourneyResult`, and `UtilityNavigationResult`.
  - Implemented 6 recommended task journeys across shipped surfaces:
    1. Optical/video import -> inspection -> save/reopen
    2. Model/pose -> supported fit -> replay/export
    3. Shot -> named flight comparison -> reopen without path re-entry
    4. Optimization/training -> result in project store
    5. Bounded estimation -> cross-engine comparison & injury indicators
    6. Global help/assistant navigation and sidekick dispatch
  - Enforced failure recovery: cancellation/restart preserves source media, missing external dependencies fail actionably without state corruption, corrupted/unsupported artifact schemas are rejected without output pollution.
  - Decoupled `CaptureRigWidget.open_in_inspect_targets()` from StepRail action set to preserve contract and fix test suites.
  - `src/shared/python/workspace/__init__.py`: Exported journey coordinator and result types.
  - `tests/integration/test_installed_workspace_journeys.py`: Full RED and GREEN integration test suite (9 tests).
- Reproduction: `python -m pytest tests/integration/test_installed_workspace_journeys.py --timeout=60`.
- Next: Auto-merge PR #10580, release lease on #10532.

## ORG-22 Reconcile and Document Intentionally Excluded, Research-Only, and Incomplete Workflows (#10530)

- Branch: `feat/issue-10530-org22-research-lifecycle`, lease `antigravity-ud-10530`, DL-#10530.
- Changes:
  - `src/config/research_capability_lifecycle.py`: Implemented `ResearchCapabilityLifecycleManager`, `IncompleteCapabilityRecord`, `CLINotInteractiveGUIError`, and `audit_research_and_excluded_capabilities`.
  - Audited all packages in `src/tools/` against launcher tiles and `src/config/registry_exclusions.yaml`.
  - Enforced fail-closed rule preventing CLI-only or headless capabilities from being claimed as GUI tiles (`CLINotInteractiveGUIError`).
  - Standardized CLI entry points for retained headless research tools (`python -m src.tools.model_converter`, `contraction`, `drift_control`, `sg_optimizer`).
  - Added structured incomplete capability tracking linking owners, issues, useful access, inputs/outputs, missing acceptance, and next actionable steps.
  - Honestly documented SG optimizer Phase 3 PyQt6 UI follow-up tied to #6272 without fake GUIs or claims of abandonment.
  - `src/config/__init__.py`: Exported lifecycle management symbols.
  - `src/config/capability_migration.py`: Added `load()` classmethod.
  - `tests/config/test_research_capability_lifecycle.py`: 8 comprehensive acceptance tests covering all RED and GREEN criteria.
- Reproduction: `python -m pytest tests/config/test_research_capability_lifecycle.py --timeout=60`.
- Next: Open PR with auto-merge, release lease, proceed to ORG-23 (#10532).

## ORG-19 Unify Sidekick, Setup, Help, and Library as Global Utilities (#10528)

- Branch: `feat/issue-10528-org19-global-utilities`, lease `antigravity-ud-10528`, DL-#10528.
- Changes:
  - `src/shared/python/workspace/global_utilities.py`: Implemented `GlobalWorkspaceUtilitiesCoordinator`, `AssistantContextSnapshot`, `WorkspaceContextualHelp`, `OnboardingPreferences`, `PlatformExecutionEnvironment`, `NativeActionUnavailableError`, and `UnknownUtilityError`.
  - Assistant context updates cleanly on workspace switch without creating duplicate sessions or stale run references.
  - Canonical alias resolution maps legacy utility IDs (`legacy_assistant`, `setup_wizard`, `library_browser`, `help_center`) to canonical utilities.
  - Sticky onboarding preferences persist across session reload and migration.
  - Focus restoration returns focus cleanly to calling widgets upon overlay dismissal.
  - Browser platform environment refuses native-only controls fail-closed with `NativeActionUnavailableError`.
  - Assistant conversation history persists across workspace navigation without deletion.
  - `src/shared/python/workspace/__init__.py`: Exported public coordinator and utility data structures.
  - `tests/launchers/test_global_workspace_utilities.py`: Full RED and GREEN unit test suite (7/7 tests passing with `pytestmark = pytest.mark.unit`).
- Reproduction: `python -m pytest tests/launchers/test_global_workspace_utilities.py --timeout=60`.
- Next: Open PR with auto-merge, release lease, proceed to next issue in backlog.

## ORG-16 Consolidate Optimization and Training Launchers (#10525)

- Branch: `feat/issue-10525-org16-optimization-training`, lease `antigravity-ud-10525`, DL-#10525.
- Changes:
  - `src/shared/python/workspace/optimization_training_workspace.py`: Implemented `OptimizationTrainingWorkspaceCoordinator`, `OptimizationJobConfig`, `TrainingJobConfig`, `OptimizationObjective`, `OptimizationResult`, `WorkspaceJob`, `WorkspaceJobKind`, `WorkspaceJobState`, and diagnostic error types (`InvalidOptimizationConfigError`, `IncompatibleBackendError`, `ModelCompatibilityError`, `DuplicateJobSubmissionError`).
  - Bounded job form over the public optimizer and training controller authority: validates objectives, constraints, and model compatibility prior to dispatch.
  - Fail-closed handling for unsupported/uninstalled backends (e.g. Crocoddyl/Drake transcription remains honestly disabled).
  - Enforced lifecycle state integrity (cancel/pause/resume; cancelled jobs cannot be mislabeled complete).
  - Deduplicated identical submissions using configuration digests without creating redundant jobs.
  - Linked dataset selection with provenance directly to durable project sessions in `SessionProjectStore` (survives store save and reopen).
  - `src/shared/python/workspace/__init__.py`: Exported public coordinator and configuration data structures.
  - `tests/integration/test_optimization_training_workspace.py`: Full RED and GREEN regression test suite (9/9 tests passing).
- Reproduction: `python3 -m pytest tests/integration/test_optimization_training_workspace.py --timeout=60`.
- Next: Open PR with auto-merge, release lease, proceed to next issue in backlog.

## ORG-15 Compose Terrain, Putting, Scene, Bunker, and Simulator Delivery Modes (#10524)

- Branch: `feat/issue-10524-org15-scene-delivery-modes`, lease `local`, DL-#10524.
- Changes:
  - `src/shared/python/workspace/shot_course_workspace.py`: Implemented `ShotCourseWorkspaceCoordinator`, `ShotCourseMode`, `TerrainConfig`, `ShotCourseRun`, `PuttingFixture`, `BunkerFidelityTier`, `BunkerRunRecord`, `SimulatorDeliveryRequest`, and `ActionAvailability`.
  - Enforced explicit model assumptions: scene view is visual inspection only; bunker preserves F0-F3 fidelity tiers; putting conforms to rolling/ground contracts; terrain mutation increments revision and invalidates prior runs; simulator delivery verifies destination capabilities and produces explicit submission receipts.
  - `src/shared/python/workspace/__init__.py`: Exported coordinator and domain value objects.
  - `tests/integration/test_shot_course_workspace.py`: 7 RED/GREEN integration tests covering incompatible flight-to-ground transition rejection, terrain mutation invalidation of dependent runs, scene view rejection of computed shots, unsupported simulator destination disabling, putting fixture round-trip, bunker fidelity export round-trip, and simulator network failure and cancellation flows.
- Reproduction: `pytest tests/integration/test_shot_course_workspace.py --timeout=60`.
- Next: PR auto-merge, complete lease on #10524, pick next issue in Epic #10508.

## ORG-14 Trajectory Viewers Handoff (#10523)

- Branch: `feat/issue-10523-org14-trajectory-viewers`, lease `antigravity-ud-10523`, DL-#10523.
- Changes:
  - `trajectory_handoff.py`: Implemented `ShotTrajectoryHandoffCoordinator` connecting swing-state extraction, end-to-end impact/flight simulation, wire export, artifact registration, and specialized viewers per ADR-0047.
  - Wire contract: `swing_sim.ball_flight_trajectory/1` with immutable SI sample positions and timestamps surviving interchange without mutating original retained samples.
  - Honest engine sourcing: Routes `manual` and `mujoco` (via `MuJoCoSwingStateProvider`) engines to `SwingBallFlightPipeline`. Refuses unsupported engines (`drake`, `pinocchio`) fail-closed with `UnsupportedEngineSourceError`; refuses arbitrary unvalidated full-body runs fail-closed with `ExtractionAdapterError`; refuses mismatched frames and invalid hashes fail-closed with `FrameUnitMismatchError` and `InvalidTrajectoryHashError`.
  - Results Workspace actions: Extended `WorkspaceActionType` with `COMPARE_FLIGHT_MODELS` and `OPEN_IN_IMPACT_EXPLORER` and diagnostic availability logic in `ResultsWorkspaceCoordinator`.
  - Session context & rollback: Carries environmental and launch conditions across active session context; provides atomic transaction staging and rollback.
- Reproduction: `python -m pytest tests/integration/test_shot_trajectory_handoff.py -v`.
- Next: PR auto-merge and release lease.

## [ORG-12] Connect Subject, Club, Model, Pose, Fit, and Dynamics Stages (#10522)

- Worktree / Branch: `feat/issue-10522-org12-model-match-handoff`, lease `antigravity-ud-10522`, DL-#10522.
- Changes:
  - `src/shared/python/workspace/model_match_handoff.py`:
    - Implemented `SubjectSpec`, `ClubSpec`, `ModelSpec`, `InitialPoseSpec` task adapters binding configuration to `SessionProjectStore`.
    - Implemented route separation with `MatchingRoute` (`TOUR_MATCHING` vs `GENERAL_MOTION_PIPELINE`) and `resolve_matching_route`. Tour route explicitly rejects 2D optical observations / arbitrary video schemas and requires qualified tour captures (driver/iron/c3d).
    - Implemented honest physics engine qualification across all 6 engines (`mujoco`, `drake`, `pinocchio`, `opensim`, `myosuite`, `simscape`) via `EngineQualification`, `get_engine_qualification`, `list_available_backends`, and `is_backend_available`.
    - Implemented `FitJobRequest` validating subject/club identity matches, initial pose frame support, engine availability, dynamic capability, and route compatibility before execution.
    - Implemented `FitJobResult` enforcing the contract invariant that kinematic outputs cannot be marked as dynamic qualified.
    - Implemented `ModelMatchHandoffCoordinator`:
      - Updates session model and marks downstream state as invalidated.
      - Executes discrete `FitStage.KINEMATICS` and `FitStage.DYNAMICS` stages, creating versioned artifact references and recording `RunMetadata` in `SessionProjectStore`.
      - Handles job cancellation with diagnostics, leaving prior completed runs preserved.
      - Implemented `reopen_run` returning action descriptor pointing to the existing Results/Replay seam (`results_browser`).
  - `src/shared/python/workspace/__init__.py`: Exported all new model match handoff types and functions.
  - `tests/integration/test_model_match_handoff.py`: 3 integration tests covering all RED preconditions (subject/club/frame mismatch, unavailable engine, kinematic vs dynamic qualification, unsupported capture routing) and GREEN execution (fit stages, invalidation on model change, cancellation preserving prior runs, reopen descriptor).
- Reproduction: `python -m pytest tests/integration/test_model_match_handoff.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10522, notify parent orchestrator.

## [ORG-11] Move Tour Matching Execution Out of Documentation Without Changing Results (#10520)

- Worktree / Branch: `feat/issue-10520-org11-motion-matching-packaging`, lease `antigravity-ud-10520`, DL-#10520.
- Changes:
  - `src/shared/python/motion_matching/execution/`: Created new execution package outside documentation tree:
    - `assets.py`: Reference asset resolution (`get_native_geometry_spec`, `get_opensim_model`, `get_candidate_geometry_spec`, `get_capture_c3d`, `resolve_output_root`) with explicit environment variable overrides and clear actionable `FileNotFoundError` messages.
    - `spec_builder.py`: Packaged anthropometric candidate spec builder with self-contained OpenSim parsing, leg extension, and CLI entry point.
    - `downswing.py`: Packaged downswing tracking experiment runner and CLI entry point.
    - `mjx_export.py`: Packaged MuJoCo MJX package exporter and CLI entry point.
    - `driver.py`: Packaged ground support driver entry point delegating to `pipeline.cli`.
    - `__init__.py`: Public package exports.
  - `docs/development/full_body_models/`: Converted legacy scripts into thin compatibility wrappers that emit `DeprecationWarning` pointing to the packaged modules while preserving CLI argument schemas and exit codes:
    - `build_anthropometric_spec.py`
    - `evidence/ground_support/run_ground_support.py`
    - `evidence/ground_support/downswing_experiment.py`
    - `evidence/ground_support/export_mjx_package.py`
  - `src/tools/motion_matching/pipeline.py`: Pointed `BUILDER`, `DRIVER_SCRIPT`, `DOWNSWING_SCRIPT`, and `EXPORT_MJX_SCRIPT` to packaged execution scripts; extended `MatchRequest` with `output_root` and asset override fields; updated `document_path`, `output_dir`, and `build_command` to write outside docs/package.
  - `tests/integration/test_installed_motion_matching.py`: New regression suite covering wheel execution without docs tree, entry point schema/cancellation/defaults parity, and deterministic numeric parity across packaged service and wrapper.
- Reproduction: `python -m pytest tests/integration/test_installed_motion_matching.py tests/tools/motion_matching/test_pipeline.py tests/tools/motion_matching/test_motion_matching_gui.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10520, notify parent orchestrator.

## [ORG-10] Connect Capture Rig, Optical Import, Pose Inspection, and Model Calibration Workspaces (#10519)

- Worktree / Branch: `feat/issue-10519-org10-capture-inspection-handoff`, lease `antigravity-ud-10519`, DL-#10519.
- Changes:
  - `src/shared/python/workspace/capture_inspection_handoff.py`: Added `CaptureInspectionHandoff`, `TrimSpec`, `CropSpec`, `PreparedVideoInspection`, `ObservationSet2D`, `OpticalMarkerTarget`, `EstimatorType`, `FreeMoCapJobAdapter`, and `JobStatus`.
  - Enforced that trim/crop/offset survive handoffs with explicit time conversions.
  - Maintained MediaPipe and OpenPose as explicit estimator choices with separate observation sets, confidence scores, and source pixels.
  - FreeMoCap input/output directory validation before subprocess spawn; cancellation leaves source files untouched with preserved HMR2/AGPL license isolation.
  - C3D and optical imports keep missing samples masked (NaN); reject incompatible spatial units and frames; reject pretending 2-D coordinates are metric 3-D.
  - Auto-registered target observations in `SessionProjectStore` preserving annotations, calibration, and club metadata without manual path re-entry.
  - `src/tools/capture_rig/gui.py` & `journey_actions.py`: Added "Open in Inspect Targets" action.
  - `src/shared/python/workspace/__init__.py`: Exported all new primitives.
  - `tests/integration/test_capture_target_handoff.py`: 6 comprehensive integration tests covering all RED and GREEN criteria.
- Reproduction: `python -m pytest tests/integration/test_capture_target_handoff.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10519, notify parent orchestrator.

## [ORG-21] Generate Accurate Atlas, Help, Parity, and Completion Records (#10531)

- Worktree / Branch: `feat/issue-10531-org21-accurate-atlas-parity`, lease `2026-09-20-org21-atlas`, DL-#10531.
- Changes:
  - `src/config/industrial_readiness.json`: Reconciled item U3 (#8820, PR #9995) from open to merged with merge SHA `8ef1bec803de292e44724cdf1f96d3ebf52bf2f2`, verified implementation paths (`src/shared/python/data_io/export.py`, `src/shared/python/data_io/provenance.py`, `src/shared/python/dashboard/_recorder_playback.py`), and test evidence (`tests/unit/test_dashboard_export_provenance.py`).
  - `docs/operations/industrial-readiness-index.md`: Regenerated from updated industrial readiness ledger.
  - `src/tools/training_controller/README.md`: Updated status to accurately reflect shipped PyQt6 GUI reality (`MainWindow`, `MainWidget`, `gui.py`, `_embed_adapter.py`, `__main__.py`) and removed obsolete draft branch notes.
  - `tests/scripts/test_workspace_documentation_freshness.py`: Added 8 comprehensive regression tests covering workspace membership drift, undocumented/dangling aliases, broken links, stale generated views, shell-only parity vs compute-complete separation, deterministic generators, training controller README accuracy, and industrial readiness U3 reconciliation.
  - `scripts/check_agent_docs_consistency.py`: Exempt markdown headings (such as `### The Rules`) and centrally managed notices from the duplicate paragraph check to prevent false positives when fleet-managed sections (`fleet-guard`, `development-logs`) share standard subheadings.
- Reproduction: `py -3.12 -m pytest tests/scripts/test_workspace_documentation_freshness.py tests/scripts/test_capability_atlas.py tests/config/industrial_readiness/ -v --timeout=60` and `python scripts/check_agent_docs_consistency.py`.
- Next: Open PR, arm auto-merge (`--auto --squash`), release lease on #10531, notify parent orchestrator.

## [ORG-10] Connect Capture Rig, Optical Import, Pose Inspection, and Model Calibration Workspaces (#10519)

- Worktree / Branch: `feat/issue-10519-org10-capture-inspection-handoff`, lease `antigravity-ud-10519`, DL-#10519.
- Changes:
  - `src/shared/python/workspace/capture_inspection_handoff.py`: Added `CaptureInspectionHandoff`, `TrimSpec`, `CropSpec`, `PreparedVideoInspection`, `ObservationSet2D`, `OpticalMarkerTarget`, `EstimatorType`, `FreeMoCapJobAdapter`, and `JobStatus`.
  - Enforced that trim/crop/offset survive handoffs with explicit time conversions.
  - Maintained MediaPipe and OpenPose as explicit estimator choices with separate observation sets, confidence scores, and source pixels.
  - FreeMoCap input/output directory validation before subprocess spawn; cancellation leaves source files untouched with preserved HMR2/AGPL license isolation.
  - C3D and optical imports keep missing samples masked (NaN); reject incompatible spatial units and frames; reject pretending 2-D coordinates are metric 3-D.
  - Auto-registered target observations in `SessionProjectStore` preserving annotations, calibration, and club metadata without manual path re-entry.
  - `src/tools/capture_rig/gui.py` & `journey_actions.py`: Added "Open in Inspect Targets" action.
  - `src/shared/python/workspace/__init__.py`: Exported all new primitives.
  - `tests/integration/test_capture_target_handoff.py`: 6 comprehensive integration tests covering all RED and GREEN criteria.
- Reproduction: `python -m pytest tests/integration/test_capture_target_handoff.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10519, notify parent orchestrator.

## [ORG-09] Guided Workflow Transitions Across Unified Workspaces (#10518)

- Worktree / Branch: `feat/issue-10518-org09-workflow-transitions`, lease `antigravity-ud-10518`, DL-#10518.
- Changes:
  - `src/shared/python/workspace/workflow_coordinator.py`: Added `WorkflowCoordinator`, `WorkflowStepId` (7-step canonical pipeline), `WorkflowMode`, `StepStatus`, `StepProjection`, and `WorkflowProjection`.
  - Step transitions enforce cryptographic hash verification and disk existence, engine requirements (single-view vs 3-D physics), and strict contract distinction preventing dynamics from inheriting purely kinematic passes.
  - Added cancellation, retry attempt tracking, and later-stage entry from imported artifacts (`entry_from_artifacts`).
  - Added pure state projection with `.to_dict()` and `.get_step()` for Qt (`WorkflowStripWidget`) and React/Tauri (`WorkflowStrip.tsx`) parity.
  - `src/shared/python/workspace/__init__.py`: Exported all workflow coordinator primitives.
  - `tests/unit/workspace/test_workflow_transitions.py`: 8 comprehensive unit tests covering all RED and GREEN criteria.
- Reproduction: `python -m pytest tests/unit/workspace/test_workflow_transitions.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10518, notify parent orchestrator.

## [ORG-08] Unified Artifact and Project Context Handoff Between Workspaces (#10517)

- Worktree / Branch: `feat/issue-10517-org08-workspace-handoff`, lease `antigravity-ud-10517`, DL-#10517.
- Changes:
  - `src/shared/python/workspace/artifact_handoff.py`: Added `ArtifactKind`, `ArtifactReference`, `WorkspaceHandoff`, cryptographic hash verification (`compute_file_sha256`), supported frames and schemas validation, and named adapter registry (`register_artifact_adapter`, `convert_artifact`) with recorded provenance.
  - `src/shared/python/workspace/project_store.py`: Added `RunMetadata`, extended `ProjectMetadata` with `runs`, `active_run_id`, and `extra_fields` migration preservation. Extended `SessionProjectStore` with `register_run`, `load_run`, `list_runs`, `set_active_run`, `get_active_run`, `clone_run`, `check_run_artifacts`, `export_handoff`, and `import_handoff`. Enforced Design-by-Contract boundary checks (cross-session subject mismatch, frame/schema validity, artifact presence and hash verification before writes, atomic write resilience).
  - `src/shared/python/workspace/__init__.py`: Exported all new workspace handoff types and functions.
  - `tests/unit/workspace/test_artifact_handoff.py`: 12 focused unit tests covering all RED and GREEN acceptance criteria.
- Reproduction: `python -m pytest tests/unit/workspace/test_artifact_handoff.py tests/unit/workspace/test_project_store.py tests/unit/workspace/test_results_browser.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease on #10517, notify parent orchestrator.

## ORG-20 Consume Provider Ownership Decisions and Verify Runtime Import Authority (#10529)

- Branch: `feat/issue-10529-org20-provider-ownership`, lease held by `local`, DL-#10529.
- Changes:
  - `src/shared/python/config/tools_vendor_authority.py`: Implemented `assert_runtime_provenance_parity` enforcing identical implementation roots between test (pytest) and packaged app runtime contexts with fail-closed `ProviderUnavailableError`. Implemented `verify_provider_provenance` asserting module paths resolve within canonical provider roots. Implemented `inspect_provider_authority` handling pinned gitlinks, clean installed wheel distributions (`ud-tools`), and probe import failures without silent fallback.
  - `tests/integration/test_installed_provider_authority.py`: 8 integration tests covering all RED and GREEN cases:
    1. `test_pytest_and_packaged_app_resolving_different_roots_fail_provenance`: Divergent roots between pytest and packaged app fail provenance.
    2. `test_wrong_pin_produces_blocked_state`: Pin mismatch returns available=False with stale pin message without falling back.
    3. `test_missing_wheel_and_vendor_produces_blocked_state`: Missing gitlink and wheel produces blocked state.
    4. `test_provider_import_failure_produces_blocked_state`: Provider probe failure surfaces blocked state.
    5. `test_sidekick_public_seam_runs_through_intended_authority`: Sidekick adapter runs through intended Tools authority.
    6. `test_movement_optimizer_public_seam_runs_through_intended_authority`: Movement Optimizer delegates to `tools_movement_optimizer`.
    7. `test_pendulum_public_seam_runs_through_intended_authority`: Pendulum public seam delegates through intended Tools authority.
    8. `test_old_supported_imports_delegate_correctly`: `upstream_drift_tools` cleanly delegates to `sidekick` with formal deprecation warning.
- Reproduction: `pytest tests/integration/test_installed_provider_authority.py --timeout=60`.
- Next: Push branch, open PR with auto-merge, update issue, release lease.

## ORG-07 Group Engine Dashboards, Exercise Variants, and Repository Shortcuts (#10514)

- Worktree / Branch: `feat/issue-10514-org07-model-variant-grouping`, lease `antigravity-ud-10514`, DL-#10514.
- Changes:
  - `model_variant_grouping.py`: Implemented `ModelVariant`, `LogicalModelIdentity`, `LogicalModelChoice`, and `ModelGroupingProjection` projecting 28 exercise variants across 4 providers into 7 logical choices without dropping underlying engine assets. Implemented `resolve_shortcut` resolving legacy IDs and presets.
  - `model_pack_manifest.py`, `model_registry.py`: Preserved `exercise` and `preset_params` fields across serialization and registry loading.
  - `launcher_model_handlers.py`: Enhanced `SharedRepoHandler` with `get_missing_checkout_diagnostic` to emit explicit actionable diagnostics on missing sibling checkouts. Prevented `sit_to_stand` fallback to `gait` in `BiomechExerciseHandler`.
  - `exercise_dashboard.py`: Dynamicized exercise title in error dialogs/widgets instead of hardcoding "Gait".
  - `models.yaml`: Moved `*_models_shared` to Models/Integrations access; annotated engine dashboards as advanced modes; tagged exercise shortcuts and movement optimizer task mapping.
- Reproduction: `pytest tests/config/test_model_variant_grouping.py tests/config/test_tile_paths_resolve.py tests/unit/config/test_model_pack_manifest.py tests/launchers/test_launcher_model_handlers.py --timeout=60`.
- Next: Open PR, arm auto-merge (`--auto --squash`), release lease, report to parent orchestrator.

## ORG-18 Surface Cross-Engine Comparison and Injury Indicators in Dedicated Workspaces (#10527)

- Branch: `feat/issue-10527-org18-comparison-indicator-workspace`, lease held by `local`, DL-#10527.
- Changes:
  - `src/shared/python/workspace/comparison_indicator_workspace.py`: Implemented `ComparisonIndicatorWorkspaceCoordinator`, `ComparisonRunArtifact`, `ModelFidelityLevel`, `CrossEngineComparisonAdapter`, `BiomechanicalLoadChannels`, `InjuryIndicatorAdapter`, and fail-closed compatibility validators (`validate_run_compatibility`, `IncompatibleArtifactError`).
  - Implemented model fidelity level invariance ensuring disparate fidelity tiers (`stub_pendulum`, `simplified_kinetics`, `qualified_full_body`) are never erroneously cross-compared.
  - Adapted biomechanical load channels to `InjuryRiskScorer` without mock fallbacks and stamped all outputs with mandatory non-clinical disclaimers.
  - Surfaced canonical comparison and injury indicator capabilities across `results_and_compare` and `exercise_analysis` workspace shells.
  - `src/shared/python/workspace/__init__.py`: Exported all new coordinator and adapter symbols.
  - `tests/integration/test_comparison_indicator_workspace.py`: 10 integration tests validating capability exposure across workspace shells, multi-run comparison, run compatibility validation (units, coordinates, timebase dt, channels, finite values), model fidelity level invariance, injury indicator execution, missing load channel fail-closed validation, and non-clinical disclaimer enforcement.
- Reproduction: `pytest tests/integration/test_comparison_indicator_workspace.py --timeout=60`.
- Next: Open PR, arm auto-merge, update issue.

## ORG-06 Apply the Same Workspace Navigation to React and Tauri (#10516)

- Worktree: main checkout, branch `feat/issue-10516-org06-react-workspace-navigation`, lease `antigravity-ud-10516`, DL-#10516.
- Changes:
  - `workspaceNavigation.ts`: Authoritative shared catalog definitions for the five primary workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`).
  - `capabilityAdapter.ts`: `resolveWorkspaceToolAction` handles in-app web routes, desktop-window launches under Tauri, and actionable explanations with web alternatives for browser users.
  - `WorkspaceNavigation.tsx`: Accessible `WorkspaceSidebar` with visible focus and focus recovery to `#main-content`, `WorkspaceBreadcrumb`, and `WorkspaceView` listing workspace member tools.
  - `WorkspacePage.tsx`: Integrates `WorkspaceShell` with `WorkspaceSidebar` and `WorkspaceView` for bookmarkable task URLs (`/workspaces/:slug`).
  - `LauncherDashboard.tsx`: Added task workspaces bar linking directly to each workspace destination.
  - `routeTitles.ts`: Centralized page titles for all `/workspaces/:slug` routes.
- Reproduction: `npm run test:run -- src/components/layout/WorkspaceNavigation.test.tsx` (from `ui/`).
- Next: Open PR referencing Fixes #10516, enable auto-merge, release lease.

## ORG-05 Build Task-Oriented Desktop Navigation Over Existing Embedded Tools (#10515)

- Worktree: main checkout, branch `feat/issue-10515-org05-desktop-navigation`, lease `antigravity-ud-10515`, DL-#10515.
- Changes:
  - `workspace_navigation.py`: Defined 5 primary task workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`); implemented `ALIAS_MAP` canonical migration utilities (`migrate_model_order`, `migrate_favorites`, `migrate_saved_layout`); implemented single-instance tool reuse policy (`find_existing_tool_tab`, `focus_or_open_tool_tab`); guarded dirty tab close (`close_tool_tab_guarded`); created accessible return-to-workspace breadcrumb bar (`WorkspaceBreadcrumbBar`); implemented discoverability status explanations (`explain_tool_status`).
  - `launcher_layout_manager.py`: Integrated `migrate_saved_layout` in `load_layout()` preserving user tile scaling, view mode, and dock state; updated `get_filtered_order()` to support workspace filters.
  - `_launcher_navigation_ui.py`: Updated `_build_sidebar_filter_buttons()` with workspace destinations; added workspace routing to `_on_sidebar_routed()`; added `navigate_to_workspace()`.
  - `launcher_ui_setup.py`: Enforced single-instance tool reuse in `dock_widget_as_tab()` before adding duplicate tabs.
  - `test_workspace_navigation.py`: 22 comprehensive unit tests covering all requirements.
- Reproduction: `pytest tests/launchers/test_workspace_navigation.py tests/launchers/test_workspace_tabs.py tests/launchers/test_launcher_layout_manager.py tests/launchers/test_launcher_ui_setup.py -v`.
- Next: Open PR referencing Fixes #10515, enable auto-merge, release lease.

## MS-82 Motion Matching Tile: Visual Playback, Standardized Metrics, and Navigation Handoff (#10355)

- Branch: `feat/10355-motion-matching-tile-playback`, lease `antigravity-10355`, DL-#10355.
- Changes:
  - `src/tools/motion_matching/gui.py`: Added visual playback support using `QMovie` for `ik_playback.gif` and `tracking_playback.gif` with safe cleanup on widget close; added display of five standardized headline metrics (`full_capture_ik_rms_mm`, `address_marker_rms_mm`, `backswing_root_error_max_mm`, `whole_run_root_rms_mm`, `inside_support_polygon_fraction`) and color-coded acceptance badge; populated physics backend combo from plant registry; added navigation action buttons "Open in Results Browser" and "Open in Viewer".
  - `src/tools/motion_matching/pipeline.py`: Added `available_engines()` querying plant registry and fallback engines; added `extract_five_metrics_and_acceptance(summary)` for standardized metric extraction and qualification verdict determination.
  - `src/tools/matched_swing_browser/model.py`: Resolved circular dependency between `matched_swing_browser` and `workspace` by moving `ResultFilter` import to `TYPE_CHECKING` and lazy runtime usage in `to_result_filter()`.
  - `src/config/feature_parity.json` & `docs/development/feature_parity_matrix.md`: Upgraded `tools.motion_matching` from `gap` to `parity`, closing #10106 capability gap.
  - `tests/tools/motion_matching/test_motion_matching_gui.py`: Unit tests for engine selection from plant registry, results pane movies, metrics, acceptance badge, navigation buttons, and step failures.
- Reproduction: `pytest tests/tools/motion_matching/ tests/config/feature_parity/ -v`.
- Next: Open PR, arm auto-merge, verify merge, release lease.

## ORG-17 Replace Canonical Estimation Shell With Bounded Estimator Coordinator (#10526)

- Branch: `feat/issue-10526-org17-estimation-workflow`, lease `antigravity-org17`, DL-#10526.
- Changes:
  - `src/shared/python/workspace/estimation_workspace.py`: Implemented `EstimationWorkspaceCoordinator`, `EstimationRunConfig`, `EstimationRunResult`, `ParameterPriorConfig`, and `IdentifiabilityGateConfig`.
  - Wires `solve_single_trial_map`, `IdentifiabilityGateOptions`, parameter priors and bounds, and spline trajectory evaluation into an application service.
  - Provides fail-closed validation for non-finite cost/trajectories and ill-conditioned systems, with complete provenance persistence round-trips.
  - `src/shared/python/workspace/__init__.py`: Exported coordinator and dataclasses.
  - `tests/integration/test_estimation_workspace.py`: 8 integration tests validating capability availability, synthetic parameter recovery, prior regularization, identifiability gating, non-finite cost gating, parameter bounds enforcement, and run provenance serialization roundtrips.
- Reproduction: `pytest tests/integration/test_estimation_workspace.py --timeout=60`.
- Next: PR auto-merge and release lease.

## ORG-03 Replace Misleading Launches With Real Tasks or Explicit Nonlaunchable States (#10512)

- Branch: `feat/issue-10512-org03-launch-truthfulness`, lease `issue-10512`, DL-#10512.
- Changes:
  - `src/launchers/task_launch_truthfulness.py`: Authoritative launch disposition audit table (`LaunchDisposition`) covering production solvers, prototype demos, library-only algorithms, parametric CLIs, provider-required tools, and service previews. Implemented pre-flight parameter validation contract for FreeMoCap sidecar runner (`launch_freemocap_with_validation`), rejecting zero-arg headless launches and dialog cancellations without spawning subprocesses.
  - `src/launchers/launcher_model_handlers.py`: Guarded `SpecialAppHandler` to reject direct launches of library-only components (`swing_optimizer`, `injury_analysis`) and parametric CLIs (`motion_capture`), exposing actionable `status_message()` with tracking issue references and remediation. Marked `GolfSimulationSuiteHandler` as prototype demo with truthful status disclosure.
  - `src/launchers/external_tools_adapter.py`: Removed blank placeholder `VideoAnalyzerWindow` fallback from `_import_video_analyzer()`, allowing missing provider / import errors to surface through the shared `_UnavailableToolWindow` diagnostic. Superseded for `video_analyzer` by #8883 (2026-09-21): the "provider required" framing no longer applies since the tile is now self-contained — see the `#8883` section below.
  - `src/launchers/launcher_process_manager.py`: Added `launch_script_with_args()` and safe argument forwarding in `launch_script()`. Hardened `_assign_to_job()` on Windows to safely handle non-integer or mock process PIDs without crashing.
  - `src/config/launcher_manifest.json` & `src/config/models.yaml`: Updated metadata and status chips for `golf_simulation_suite` (`prototype`), `swing_optimizer` (`experimental`/library), `injury_analysis` (`experimental`/library), removing qualified engine claims.
  - `ui/public/capability-atlas/`: Regenerated `graph.json` and `index.html` via `python -m scripts.generate_capability_atlas`.
  - `tests/launchers/test_task_launch_truthfulness.py`: TDD acceptance suite verifying all RED/GREEN cases: problematic tiles cannot claim ready/gui_ready; missing video analyzer yields diagnostic window; FreeMoCap zero args/cancellation performs no spawn and valid args pass through unchanged; simulator prototype is marked demo-only.
- Reproduction: `pytest tests/launchers/test_task_launch_truthfulness.py tests/launchers/test_simulation_guis.py tests/launchers/test_launcher_process_manager.py tests/scripts/test_capability_atlas.py --timeout=60`.
- Next: Validate against rebased main, enable auto-merge on #10536, release lease on #10512, claim #10513 (ORG-04).

## ORG-02 Separate Capability Identity, Maturity, Availability, and Qualification (#10511)

- Branch: `feat/issue-10511-org02-capability-state-contract`, lease `antigravity-ud-10511`, DL-#10511.
- Changes:
  - `capability_state.py`: Implemented orthogonal typed models: `SurfaceAvailability` (with DbC invariant demanding non-empty reason and remediation when unavailable), `CapabilityAvailability` (desktop, web, api, cli), and `CapabilityQualification` (with status, is_qualified, receipt, failure reasons). Implemented `adapt_engine_matrix_qualification` consuming #10351 engine matrix contract (non-engine tools are exempt). Implemented `RuntimeProbeKey` and `RuntimeProbeCache` for lazy cached probing tied to provider pin / runtime identity. Established `CANONICAL_TILE_DISPLAY_NAMES` and `resolve_canonical_display_name`.
  - `launcher_manifest_loader.py`: Extended `LauncherTile` with `maturity`, `availability`, and `qualification` fields. Implemented `_build_default_availability` deriving surface states from provider and web contracts. Updated `LauncherTile.to_dict()` to serialize orthogonal fields while maintaining backward compatibility for legacy `status`. Enforced that unqualified engines never claim `ready` or `stable`.
  - `models.yaml`: Synchronized canonical display names for `matlab_suite` ("Matlab Models") and `golf_simulation_suite` ("Golf Simulation Suite").
- Reproduction: `pytest tests/config/launcher_manifest/test_capability_state_contract.py tests/config/launcher_manifest/ --timeout=60` and `pytest tests/config/test_launcher_registry_parity.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease.

## ORG-01 Baseline Every Capability and Preserve Tile, Layout, and Artifact Identity (#10510)

- Worktree: main checkout, branch `feat/issue-10510-org01-capability-baseline`, lease `antigravity-ud-10510`, DL-#10510.
- Changes:
  - `src/config/capability_migration.py`: Machine-checkable capability migration data model and validation engine (`CapabilityMigrationInventory`, `MigrationEntry`, `FixtureEntry`). Enforces uniqueness, acyclic alias resolution, primary workspace exclusivity, provider availability decoupling, saved layout validation, and fixture hash verification.
  - `src/config/capability_migration.json`: Complete canonical baseline inventory cataloging all 104 launcher tiles/models (61 base desktop models + 29 provider models + 14 web catalog tiles), 45 feature parity contracts, 9 excluded tool packages, and 15 golden preservation fixtures with SHA-256 integrity hashes.
  - `scripts/generate_capability_baseline.py`: Markdown generator and freshness validator for `docs/development/ORG01_CAPABILITY_BASELINE.md`.
  - `docs/development/ORG01_CAPABILITY_BASELINE.md`: Architecture decision document recording baseline inventory, workspace domains, alias resolution graph, and preservation of ADR-0047 viewer identity and provider seams.
  - `tests/config/test_capability_migration_coverage.py`: 18 RED and GREEN regression tests covering unclassified local/provider/manifest/CLI detection, alias cycle detection, duplicate identity prevention, missing provenance failure, legacy alias resolution, and fixture hash preservation.
  - `scripts/capability_atlas/render.py` & `docs/architecture/CAPABILITY_ATLAS.md`: Linked ORG-01 baseline from the generated capability atlas.
- Reproduction: `pytest tests/config/test_capability_migration_coverage.py --timeout=60` and `python -m scripts.generate_capability_baseline --check`.
- Next: PR auto-merge, complete lease on #10510, proceed to downstream issues in Epic #10508.

## ORG-13 Results Workspace Handoff (#10521)

- Branch: `feat/issue-10521-org13-results-workspace-handoff`, lease `antigravity-ud-10521`, DL-#10521.
- Changes:
  - `results_workspace.py`: Implemented `ResultsWorkspaceCoordinator`, `ResultArtifactItem`, `ResultCategory`, `WorkspaceActionType`, `ActionAvailability`, `ComparisonResult`, `MissingAssetDiagnosticError`, `UnitMismatchDiagnosticError`, and `HandoffDispatchPayload`.
  - Consumed public contract of #10353 (`MatchedSwingBrowserModel` / `MatchedSwingFilter`) and canonical `ResultsBrowser` indexing, without creating a competing browser.
  - Enforced selected run isolation in tool handoffs, preventing fallback to global active sessions.
  - Implemented artifact categorization (source data, processed recipes, kinematic replay, dynamic run, measurements, flight trajectory, qualification verdict).
  - Validated missing assets and unit mismatches during run comparison (never guessing similarly named files or silently comparing mismatched units).
  - Verified provenance retention (#8820) across export and reimport round-trips.
- Reproduction: `python3 -m pytest tests/integration/test_results_workspace_handoff.py tests/tools/matched_swing_browser/test_model.py --timeout=60`.
- Next: PR auto-merge and release lease.

## Docker Audit Repair (#10472)

The Docker image now pins Tornado 6.5.8, matching the generated runtime and
development locks and the declared runtime floor. This resolves the in-image
pip-audit findings GHSA-wwv5-g3v4-889x and GHSA-8423-8fgw-73vq (plus the third
Tornado 6.5.7 advisory) without an audit waiver. CI must confirm the complete
Docker build and dependency-artifact regeneration before merge.

## ORG-04 Validate Every Browser, Tauri, and Native Launch Destination (#10513)

- Branch: `feat/issue-10513-org04-launch-destinations`, lease `antigravity-ud-10513`, DL-#10513.
- Changes:
  - `ui/src/routes.ts`: Declared canonical `KNOWN_APP_ROUTES` array and `isKnownAppRoute(route: string): boolean` helper.
  - `ui/src/api/webLaunch.ts`: Updated `resolveTileLaunchAction` to reject unknown/unregistered routes with `'Unavailable'` badge and reason instead of silently navigating to 404.
  - `ui/src/api/launcherReachability.ts`: Implemented `evaluateTileReachability` and `generateReachabilityMatrix` covering browser and Tauri contexts.
  - `ui/src/api/launcherReachability.test.tsx`: Comprehensive Vitest suite for reachability matrix and unknown route rejection.
  - `src/config/launcher_manifest_loader.py`: Sanitized `web_route` for Movement Optimizer in `_build_provider_tile`, deriving `native-window` cleanly.
  - `src/shared/python/movement_optimizer/model_pack.yaml`: Removed unmapped `web_route: "/tools/movement-optimizer"`.
  - `tests/config/launcher_manifest/test_parity.py`: Added checks for loaded tiles in `test_route_mode_routes_exist_in_react_router`, and added `test_every_tile_destination_resolves_authoritatively`.
- Reproduction: `npm run test:run` in `ui/` (874 tests pass); `pytest tests/config/launcher_manifest/ tests/config/test_launcher_registry_parity.py` (95 tests pass).
- Next: Land PR, arm auto-merge, release lease.

## MV-06 Expose Real Forces, Torques, and Explicit Counterfactual Semantics (#10482)

- Worktree: `UpstreamDrift-10482-forces`, branch `feat/10482-forces-torques-counterfactual`, lease `antigravity-10482-mv06`, DL-#10482.
- Changes:
  - `force_torque.py`: `SpatialWrench`, `transform_wrench` with moment arm calculation, `compute_center_of_pressure` with strict threshold semantics (Fz <= 5.0 N returns None), and `ContactReaction`.
  - `counterfactual.py`: `AccelerationDecomposition` (gravity, drift, control, ZTCF, ZVCF), `CounterfactualStrategy`, `CounterfactualFork`, and `create_counterfactual_rollout` with cryptographic baseline immutability check.
  - `candidate_session.py`: Added `get_wrench_at`, `get_center_of_pressure`, `get_joint_torques_at`, `get_closure_residual_at`, and `create_counterfactual_fork`.
  - `requests.py`, `analysis.py`, `simulation_service.py`: `GET /analysis/candidate/forces` and `POST /analysis/candidate/counterfactual` endpoints (failing closed with 409 when no session is loaded).
  - `force_inspection.py`, `gui.py`: `ForceInspectionWidget` synchronized with physical playback time embedded in Tour Matching Viewer.
- Reproduction: `pytest tests/unit/motion_matching/test_force_torque.py tests/unit/motion_matching/test_counterfactual.py tests/unit/motion_matching/test_candidate_session_forces.py tests/unit/api/test_candidate_session_analysis_routes.py tests/unit/tools/test_tour_matching_viewer_forces.py`.
- Next: PR auto-merge; epic #10476 completion!

## MV-05 Manage MeshCat and Gepetto Launch Lifecycle and URDF Loading (#10481)

- Completed in PR #10501 (merged). Managed subprocess lifecycle in `ViewerProcessManager`, active port polling, existing listener detection, native viewer backends (`open_in_native_viewer`), `ModelBundle.extract_to`, CLI and GUI integration.

## MV-04 Reuse Shared Physical-Time Playback Across Viewers (#10480)

- Completed in PR #10496 (merged). PhysicalTimePlayback, quaternion SLERP, Euclidean LERP, playback adapters matrix, Tour Matching Viewer transport controls.

## MV-03 Bind Saved Candidates to Viewer & Analysis Sessions (#10479)

- Completed in PR #10492 (merged). CandidateSession ingestion, WSL host probe, multi-candidate replay overlay with ENGINE_COLORS, rejected fit banner, GIF export.

## MV-02 Anatomical Visual Assets and Skin Toggling (#10478)

- Completed in PR #10488 (merged). Added visual skin modes, visible diagnostic fallback (magenta), cadence pacing, and verified physics immutability under visual skin toggles.

## MV-01 Qualify Shared URDF Bundles and Numeric Precision (#10477)

- # Completed in PR #10485 (merged). 17g float serialization, ModelBundleManifest, ModelBundle zip export/import, Drake export integration, Pinocchio parity verified.

## MS-21 MuJoCo Replay Continuation (#10336)

Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-codex-10336`.
Branch: `feat/10336-mujoco-candidate-replay`. Base HEAD:
`5347cba0f4378cd72a6e8afea9fb27c8bfe5db75`. Implementation `94ccb1825`;
PR #10448 open. User authorized commit, push, PR, CI repair and merge.
Development-log entry: DL-#10336. Ownership was checked free and leased as
`codex-mujoco-replay-10336-20260918` before editing. The original checkout's
Pinocchio changes and untracked evidence were preserved.

Implemented name-mapped armature and saved-control replay through the existing
MuJoCo rigid-grip/shared-contact plant, exact G1 slicing and fail-closed checks.
See [Reproduction and Results](../../evidence/matched/driver_full_mujoco_replay/README.md)
for commands, candidate identity and receipt. Historical source evidence is
unchanged. Diagnostic armature 0.005 kgÂ·mÂ² follows turnover guidance but cannot
be certified as source-identical: the merged analytic producer never applies
armature and its receipt omits it, contact configuration and interpolation.
Missing root history and independent source dynamics also block qualification.

Validation: 40 focused replay/acceptance/schema/full-body/contact tests pass;
scoped Ruff and mypy pass; agent-context and architecture/file-size budgets pass.
PR CI found CVE-2026-63374 and CVE-2026-64847 in the inherited AnyIO 4.12.1
pins. SELF updates both runtime/development locks to the reported fixed 4.14.2;
the audit remains enforced. SELF refreshes the matched-swing ledger for the new
receipt and repairs the existing cross-engine gravity fixture (#4249): MuJoCo
and Drake now consume the same collision-free canonical URDF and right-hand
anchor rather than unlike demo plants. The unchanged 5 mm gate passes with
real MuJoCo 3.8.0 / Drake 1.51.1 on Linux (15 passed, 2 unavailable-engine skips),
including new analytic free-fall checks. This fixture does not qualify contact
or bilateral grip closure. Replay plus ledger tests: 28 passed. The fixture uses
defusedxml for both reading and serialization to satisfy the XML security gate.
Continue PR CI on the branch updated with main's shadow-tracker merge.
Commit/push hooks pass, including security and bounded unit checks. All 21
replay tests also pass with CI-pinned MuJoCo 3.8.0 in an isolated environment.
The central development-log validator flags inherited missing verifying SHAs,
a missing PR field in DL-#9967, and the existing portfolio WIP excess. These
unrelated entries are not repaired in this diff. The repo-local validator
script is absent, so the central Repository_Management copy was used.
Next: review this focused diff, then obtain a provenance-complete source
candidate and independent native replay. Do not close #10336 or promote G2/G3
from these diagnostic artifacts. Earlier lane handoffs follow unchanged.

---

## Shadow Tracker #10233 Handoff

- Workspace: `C:/Users/diete/Repositories/UpstreamDrift-10233-pr`.
- Branch: `fix/shadow-tracker-10233-pr`; HEAD/base:
  `5347cba0f` (origin/main); implementation `c76d6f02c`; handoff update SELF; PR #10450 open.
- CI repair SELF: upgrade AnyIO locks 4.12.1 -> 4.14.2 for
  CVE-2026-63374 and CVE-2026-64847; no audit waiver or gate weakening.
- Entry DL-#10233: provider implementation and tests complete; renderer and
  unrelated original-checkout work preserved.
- Contracts, RED/GREEN commands, compatibility and limitations:
  [Shadow Tracker turnover](../plans/shadow_tracker/TURNOVER_CURRENT.md).
- Next action: validate PR #10450 checks, repair any actionable CI failures,
  then merge through normal branch protection.

## OpenSim Epic #10394 Status (OG-01 Through OG-09 Completed)

- Governing epic #10394 / #10363. All 9 child tasks completed and qualified:
  - OG-01 (#10404, merged): Baseline model geometry and structural qualification audit.
  - OG-02 (#10407, merged): Consistent OpenSim segment scaling module and acromion proxy reconstruction.
  - OG-03 (#10405, merged): Parameterized visual club geometry and grip offset frames.
  - OG-04 (#10408, merged): Pure 3D rigid capture registration with Kabsch alignment and camera presets.
  - OG-05 (#10409, merged): Two-handed address pose calibration, grip closure verification, and tolerance profile.
  - OG-06 (#10410, merged): Full-swing dynamic tracking qualification and multi-stage ladder progression.
  - OG-07 (#10412, merged): Versioned OpenSim model variants, actuation profiles, and adapter boundaries.
  - OG-08 (#10413, merged): Muscle and tendon extension qualification, moment arm derivative check, and static tendon equilibrium.
  - OG-09 (#10403, merged): Native viewer package packaging, timeline scrubbing, visual layer toggles, and release evidence export.

---

# Native Multi-Engine Matching Handoff

## Matched Swing Program: Pinocchio Lane (MS-31, #10338)

Updated 2026-09-18 (local/claude). Full 654-frame Driver (`C3D_TA_Driver.c3d`, 1.814 s) and 657-frame 7-Iron (`C3D_TA_Iron.c3d`, 1.827 s) swings successfully matched in ~8.3 s using the decoupled kinematic tracking + analytic inverse dynamics pipeline (`scripts/match_pinocchio_c3d.py`).
Delivered PR #10411 on `feat/10338-crocoddyl-native-fit`. Both Optimum (minimum 2-norm, 3.9 ms) and Trail-Side Zero (tau_trail == 0, 168-215 ms, 436-616 N grip force transfer) torque solutions computed with exact forward dynamics acceleration parity.
Current turnover: `docs/development/matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md` and `docs/development/PINOCCHIO_C3D_MOTION_MATCHING_GUIDE.md`. Program epic #10363.

## Current Status

Run102 is the latest committed native fit found in the 2026-09-16 review.
It is a rejected0â€“0.85 s prefix: MATLAB terminal RMS40.301 mm exceeds35 mm.
Overall20.267 mm, early9.995 mm, club8.389 mm and yaw0.610% pass their gates.
The optimizer exhausted its physical evaluation budget and returned a fallback;
accepted=false and optimizer_converged=false. No full-swing acceptance exists.

[Current Completion Handoff and Agent Prompt](simscape_tour_matching/COMPLETION_HANDOFF_20260916.md)
is the authoritative next-work plan. It supersedes RUN101_REVIEW_AND_TURNOVER.md
for task ordering. Source reviewed:2f0460d25; fetched main:d2aafa43c.

Priorities: recover clean-checkout native fitting providers currently available
only through historical/frozen runtime sources; coordinate finite-weld derivative
fix issue10260 / PR10263; establish articulated terminal feasibility; run the first
bounded0.90 s fit; produce reproducible manifests and synchronized motion reports.
PR10263 is not yet a qualified merged dependency. Do not duplicate its owner.

Run101 refinement supports4.52 micrometer marker agreement with refined R2025b
and46.9 nanometer Pinocchio self-refinement change. Run102 marker agreement at
baseline settings is0.0605 mm maximum. These are prefix-specific measurements,
not full-horizon rate/effort qualification. Keep geometry seed and force-frame fixes.

Historical evidence lives under simscape_tour_matching/native_evidence. Remote
hosts are DeskComputer (explicit R2025b) and ControlTower (Pinocchio WSL).
Remote live jobs were not inspected in this review; check before launching.
Epic9921 / issue9967 are closed despite the incomplete full goal; reconcile tracking.
This review launches no simulation or fitting job.

## Representation Qualification

Run58 is terminal (handle2605 exited0). Local-chart DOP853 on native spherical
Pinocchio PASSES all existing parity gates over the run19 0.85-second fixture:
marker maximum2.62982e-8 m; native q maximum1.28599e-7; native rate
maximum1.15849e-5; closure pose3.74372e-11 and rate9.40426e-11.
Elapsed37.6895 s/9972 RHS calls versus7.2147 s for the tighter scalar reference.
This is a same-input representation result, NOT a C3D fit or full R2025b parity.

Run59 is TERMINAL, original handle97788 exited0: same setup with smaller maximum
step. It FAILS native velocity parity (0.000587270); marker maximum1.15187e-7 m,
native q6.06676e-7, closure pose6.60249e-11/rate1.61521e-10. Elapsed61.2725 s.
Therefore58's isolated pass is not robust convergence or representation acceptance.
Those representation runs are terminal. The pointwise acceleration/history audit
found no sampled history dependence or substantial actuator-route mismatch:
maximum routed native acceleration difference2.18034e-6 rad/sÂ² at a reference
539073.49 rad/sÂ²; effort roundtrip1.25056e-12. Sampled unscaled inertia condition
is about6.7e7 in both representations. These results support investigating
conditioning/trajectory sensitivity, not declaring a physical-model mismatch.
They do not establish global convergence. Detailed receipt:
simscape_tour_matching/native_evidence/manifold_acceleration_10043_19.
Remote runtime /home/dieterolson/native-manifold-10043-18; driver
/mnt/c/Users/diete/compare_native_manifold_replay_9967_59.py; output
/mnt/c/Users/diete/native-manifold-replay-9967-59. Horizon0.85, methoddop853,
rtol1e-12, atol1e-14, max_step1/1440, max_evaluations100000. Scalar reference
rtol1e-12, atol1e-14, max_step0.000125. Poll the existing handle/process before
launching another run. All root runs through59 are terminal; none should restart.

## Transition Sensitivity and Regularized Fitting

Run60 completes six original-state replays. Repeated baseline results are exactly
identical. A late LSInputX B6 perturbation of1e-6 Nm changes native rates by
0.00750466 rad/s near0.786111 s but markers by only0.5386 micrometres. The
1e-4 Nm trials produce about0.765 rad/s and50.8 micrometres. Central derivative
estimates retain amplitude dependence; these are measured sensitivity evidence,
not a waiver of parity gates. Exact inputs, executed source and raw trajectories
are archived in native_evidence/perturbation_9967_60.

Shared prefix and multiple-shooting fitters now accept checked analytic penalty
Jacobians. NativeEffortPenalty reuses the native actuator/frame provider and
seven-point quadrature to evaluate exact mean-square total degree-six primitive
efforts. Twelve new tests passed after RED; combined fitter/effort/control regression77 tests
passes, as do Ruff and four-module mypy. Immutable ControlTower runtime61 passes
96 focused/real Pinocchio tests; source archives/hashes are preserved in
native_evidence/regularized_fit_runtime_9967_61. Existing runner behavior remains
the default with zero penalty. Trial61 is terminal0 in75.40 s: whole RMS29.813 mm, early10.707 mm,
terminal75.846 mm, club23.930 mm, versus baseline30.791/10.667/99.989/56.557 mm.
All three primal sensitivity gates pass. Max_nfev3 was reached; no numerical
acceptance or convergence is claimed. Native effort cost barely changes
(1.55060 to1.55026), so improvement cannot be attributed to the penalty alone.
Exact evidence and convenient candidate: native_evidence/regularized_fit_9967_61.

Run62 is terminal0 after468.965 s: whole28.6353 mm, early10.8248 mm,
terminal66.7134 mm, club31.0615 mm. It reaches max_nfev20 with four active
correction bounds; accepted/converged both false. Returned canonical candidate
7467c5d82817251858255bdf0e560a712f0d20a366ee0094c364a6c14481e1d5.
Native evidence folder regularized_fit_9967_62 contains the terminal receipt,
measured replay arrays and marker-comparison.png (root visually inspected).
Run63 is terminal1 after77.672 s and three forward evaluations. The B4/B5/B6
trial fails the unchanged sensitivity-primal marker agreement gate on its third
candidate, hash e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d.
There is no returned accepted candidate. First two Jacobians passed. Diagnostic64
PASSES at that exact candidate before its600 s wall budget: augmented solve
409.086 s/1,222,535 evaluations, primal marker discrepancy8.9147861e-9 m against
the unchanged1e-7 gate. Total launch466.239 s; diagnostic call/save419.190 s.
The augmented settings were rtol1e-12/atol1e-14/max_step0.000125. The provider's
independent primal remained rtol1e-11/atol1e-13; max_step applies to both. This
supports an accuracy-dependent63 failure, not a derivative correctness or C3D
acceptance claim. Cost is far above the default sensitivity calls. All jobs through69 are terminal; diagnostic70 is the next authorized experiment.
Runtime65 adds the checked call-budget API and passes36 tests including nine
real Pinocchio tests. Diagnostic65 changes only max_step to0.000125 relative
to63, retaining augmented tolerances1e-10/1e-12: marker difference3.88459e-8 m,
82931 sensitivity calls,27.0993 s (versus64's409.086 s). Gates are unchanged.
Audit66 checks saved64 LSInputX B6 column41 against six original-state replays:
relative errors9.11e-5/4.94e-4/9.53e-6 for1e-5/1e-4/1e-3 Nm, all below the
existing1e-3 threshold. This verifies one direction, not the entire Jacobian.
Raw sources/inputs/replays are preserved in sensitivity_9967_65 and derivative_9967_66.

Runner now accepts --max-step for both forward and sensitivity paths and
--max-sensitivity-evaluations; defaults preserve prior behavior. Five new parser
checks went RED/GREEN;13 runner tests pass. Runtime68 qualifies32 tests including nine real Pinocchio tests, plus runner
--help. Fit68 is TERMINAL1 after425.444 s, former PID2842183 / handle54657;
output /mnt/c/Users/diete/native-regularized-fit-9967-68. Authorized fit68 retains original
baseline19, restarts returned62, uses B4/B5/B6,max_nfev10,max_step0.000125,
sensitivity budget100000, and unchanged weight0.01/scales/amplitude/bounds.
Eight Jacobian checks pass, then the ninth candidate fails unchanged marker
agreement (not its evaluation budget). Candidate canonical hash:
c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039.
The last unqualified checkpoint reports28.138 mm whole/65.565 mm terminal;
it is not a returned solution. Exact evidence: native_evidence/regularized_fit_9967_68.
All previous numerical runtimes remain immutable.

Manufactured diagnostic67 confirms inactive sensitivity-column padding changes
physical integration error under the existing combined error norm. This suggests
an architecture improvement, not a proof of native model error. Keep it as a
regression case for opt-in grouped DOP853 error control, now implemented through
shared integrate_forward/integrate_sensitivities and the native provider/runner.
The maximum of SciPy's existing physical and sensitivity block norms controls
steps; default behavior is unchanged. The protected SciPy hook requires runtime
qualification. Initial-step selection remains unchanged.57 focused tests pass
after RED/GREEN, plus Ruff; no global accuracy guarantee is inferred.
Diagnostic70 is TERMINAL1 on the exact failed68 candidate, max_step0.000125,
rtol1e-10/atol1e-12,100000 sensitivity calls, separate_error_control=True.
Runtime70 passes49 focused tests including nine real Pinocchio cases. Agreement
fails narrowly at1.02308673e-7 m, t0.85, WaistRBack axis1; limit remains1e-7.
Call35.969 s, launch40.103 s. Diagnostic71 is TERMINAL1: tighter augmented
rtol3e-11/atol3e-13 yields1.02308684e-7 m, effectively unchanged, in34.025 s.
Diagnostic72 is TERMINAL0: max_step0.0000625 with original70 tolerances yields
2.506897942e-8 m agreement,164339 calls/53.916 s sensitivity (71.554 s call/save).
Marker samples and state/marker sensitivity Jacobians are archived; primal q/qd
trajectories were not saved. This qualifies one candidate, not all derivatives.
Fit73 is TERMINAL0 after989.414 s, accepted/converged false, max_nfev10.
All ten Jacobian checks pass; final whole28.10547 mm, terminal65.39791 mm,
early10.86040 mm and club30.04069 mm. Settings: original19 baseline, B456,
grouped control, max_step0.0000625,200000 sensitivity calls.
Runtime73 qualifies62 tests plus runner help. Former PID2891462/handle55242
are terminal. Output /mnt/c/Users/diete/native-regularized-fit-9967-73. A separate
post-return replay saves actual q/qd (not qdd), markers and a rejected overlay.
Final candidate canonical SHA256:
786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a. Four changed source modules pass mypy.
Marker agreement failures now report discrepancy, time, marker and axis.

The shared native motion sequence API is implemented and publicly exported.
It converts complete batches with immutable time/model/frame/branch metadata
and original per-sample references, reusing NativeJointStateAdapter. Root passes
101 related tests including12 new strict atomic file-I/O tests. Public
save_native_motion/load_native_motion preserve the validated sequence plus an
optional distinct raw-model hash. UI and complete dynamic frame transport remain;
Five new consumer checks pass, including actual MuJoCo all16 frames at three
samples; Drake/Pinocchio consumer tests skip locally for missing runtimes.
These are conversion checks, not alternate MuJoCo/Drake dynamics qualification. See dedicated
REPRESENTATION_HANDOFF for runnable usage and the remaining ordered plan.

## MuJoCo Spherical Export and Fitting Conditioning

The opt-in native_spherical_mjcf exporter now reuses canonical MJCF geometry and
shared group inventory to replace three XYZ triples with ball joints. Real local
MuJoCo3.3.4 compiles nq30/nv27; five tests verify unchanged inertias/closure/scalars
and all body/site poses across three manufactured states. This is kinematics
qualification, not a dynamics adapter or stock mj_step acceptance. Next gates:
tangent conventions, dual effort/convective acceleration, reused rigid-closure
solve, then same-input trajectories. Dedicated turnover:
../development/mujoco_native_matching/MUJOCO_MANIFOLD_HANDOFF.md.

Offline audit74 uses saved72 Jacobians, existing valid masks, amplitude10 and
terminal weight10. Marker-only23100-by-81 condition number is7.452e7; column
normalization leaves3.767e7. Penalty rows/bounds are excluded. This supports
checking correlated sensitivities, not claiming a proven cause of stagnation.
Fit73 ended with only modest matching improvement. Inspect
actual/predicted reduction and a mixed-control directional derivative before
more compute. See native_evidence/sensitivity_condition_9967_74.

## Fixed-Attachment Feasibility

Audit69 uses existing rigid-body relaxation on all654 frames. Three head markers
are attached to Hub; no independent native head joint exists. Fixed-offset lower
bounds are17.393 mm whole swing/24.279 mm terminal, even with independent body
poses and no dynamics/connectivity. Separating head as a hypothetical independent
six-DOF body lowers this to4.471/9.177 mm, but changes the model. No marker/model
changes were made. Current torque errors exceed the rigidity floor, so both
optimization and physical approximation matter. Preserve original25/35 mm gates;
never silently omit head markers. See native_evidence/rigidity_9967_69.

## Reproducible Compute Limits

`integrate_forward` and `integrate_sensitivities` now accept optional positive
integer max_evaluations, checked before excess derivative/linearization calls.
`replay_marker_sensitivities` exposes max_sensitivity_evaluations for the augmented
solve only; its independent primal replay remains separate. DefaultsNone retain
existing behavior. Seven new tests and native forwarding checks went RED/GREEN;
27 related tests, Ruff and pinned mypy pass. This source update is NOT in immutable
runtime61; qualify a new runtime before using it remotely. Budgets are not a
numerical accuracy or wall-time guarantee. Exhaustion raises without partial data.

## Why Earlier Attempts Stalled

- Static pose fitting and feedback tracking did not produce an open-loop sextic
  motion. Run45 feedback whole RMS35.2 mm becomes346.5 mm in time-only replay,
  and655.5 mm after global-sixth-order compression. Do not promote the initializer.
- An actual native gimbal singularity stopped run41. An independent inverse-angle
  branch bug caused run49 to apply a different native effort map. Branch metadata
  is now explicit and tested; true singular native actuator inverses remain rejected.
- Native rate errors are amplified near the shoulder chart. Run52 physical
  angular error0.000208 rad/s became native rate error0.00465 rad/s. Tight scalar
  reference54 and manifold tolerance refinement led to passed58. Gates were not
  relaxed. Run54/55 second-level budgets were insufficient even for their selected
  maximum step; those failures are not proof of solver nonconvergence.
- The Python manifold implementation remains slower than scalar Pinocchio. Keep
  scalar native dynamics available as the fitting baseline while qualifying
  alternate representations. Do not block fitting on all-engine alternate builders.

## Two-Window Preflight and Direct Chart Derivatives

Stage1 diagnostic75 is TERMINAL0. Window0 starts at original q0/qd0 and window1
at the saved uninterrupted73 sample216 at0.6 s. Against uninterrupted73, maximum
marker distance is5.369e-12 m, native q4.422e-11 and qd8.804e-9. First endpoint
minus saved node is9.77e-14. Full pose/rate closure and the42-dimensional q/v
node chart pass. Zero retraction shifts physical state by4.75e-12. Exact evidence:
native_evidence/two_window_preflight_9967_75. This is fixture qualification, not
an optimized segmented swing or a derivative pass.

MultipleShootingOptions now accepts window_jacobian_state_coordinates="node"
(default "physical"). Node mode takes theta followed by direct chart columns,
retaining physical endpoint rows and the negative next-node transform derivative
in continuity. No pseudoinverse extension is needed. Eleven new nonlinear tests
went RED/GREEN; root passes44 combined shooting/node tests and source mypy/Ruff.

Diagnostic76 is TERMINAL (former PID2923511/handle25121), on frozen runtime73. It checks
81 control columns in window0 and81+42 in window1, then at most16 signed
perturbation trials for control, node-position, node-velocity and mixed directions.
Marker, q, qd and scaled endpoint/continuity checks retain1e-3 derivative gates
with explicit weak-block reporting. Process exit0 is not scientific acceptance:
status is failed_derivative_gates. LS B6 h1e-4 and mixed h1e-6 pass all blocks;
smaller steps fail some resolved checks. Both node directions pass resolved blocks
but near-zero cross-continuity derivatives fail relative checks (analytic norms
about5e-14/2e-17 versus central estimates about1e-10/1e-11). Establish physical
absolute-error floors/structural-zero treatment before declaring full qualification;
do not widen C3D or replay gates. Call225.112 s, launch250.611 s, all16 trials
complete. No optimizer or other numerical job remains active. Evidence:
native_evidence/two_window_derivatives_9967_76.

## Derivative Resolution Floors (Audit 77, Terminal)

Local analysis of the archived run76 arrays already resolves the two failure
classes. (a) Node-direction cross-continuity blocks: the zero-retraction
Jacobian equals diag(scales)\*basis to1.6e-10 with scaled orthonormality defect
3.1e-15, so the velocity response of the pure-position singular direction and
the position response of the Dq null direction are structurally zero up to
that defect; the archived retracted nodes match the linear prediction to
1.2e-14 (h1e-5) and1.3e-12 (h1e-4), so the central estimates (1.7e-10/1.5e-11)
are retraction roundoff divided by2h, not derivative error. (b) LSInputX h1e-5
and mixed h1e-7 absolute errors are consistent with replay noise divided by
step (endpoint q about4e-10 to2e-9 per replay); the archived perturbed
coefficients reproduce the intended Bernstein increments to4.8e-10 (control)
and4.3e-7 (mixed h1e-7) relative, and effort evaluation error is at most
2.3e-6 relative to h, far below the observed4e-3 failures.

New shared providers: `motion_matching/derivative_resolution.py`
(central_difference_floor, classify_derivative_block with verdicts passed /
structural_zero / unresolved_at_step / failed, qualify_direction and the
orthonormality cross_block_norm_bound) and
`pinocchio/python/native_node_chart.py` (closure/Jacobian/basis/retraction
wrapper reused by drivers). Twenty new unit tests pass with Ruff and mypy.
A pass now additionally requires the measured floor to be below the gate times
the analytic norm; agreement inside noise is reported unresolved, never passed.

Audit77 is TERMINAL0 (202.6 s launch,26 of32 budgeted window replays) on
runtime77 (frozen73 file set refreshed from f96766c8e plus five new files;122
qualification tests). Measured per-block replay error by tolerance variation at
unchanged max_step: markers8.2e-10 m, endpoint q9.3e-10, qd1.6e-8. Loose and
working replays coincide while both differ from tight by that amount: the
integration is max-step limited and the non-reproducible part is step-sequence
roundoff/constraint-solver noise, so tolerance alone does not shrink it. Base
replays reproduce run76's primal arrays exactly. Every factor-one failure sits
at the smallest step with absolute error1.02xâ€“1.14x the single-pair floor;
`reclassify.py` applies the tested `floor_safety_factor=2` to the same archived
arrays and every direction/block then has a resolved pass (LSInputX h1e-4;
mixed h1e-6 and h1e-5) or an orthonormality-verified structural zero, with no
unexplained failure. Gates are unchanged. Evidence and turnover:
native_evidence/two_window_floor_9967_77 (raw ZIP SHA2563d5768fcâ€¦).

MultipleShootingOptions now also accepts shared_boundary_policy="once"
(default "both"), which observes the capture sample shared by adjacent windows
only in the earlier window so marker rows, Jacobian rows, segmented RMS and
equality offsets match an uninterrupted single-window objective at zero
defect. Four RED/GREEN tests;46 combined shooting tests, mypy and Ruff pass.

## Two-Window Direct-Node SLSQP Trials 78â€“81 (Terminal, Historical)

Runs 78â€“81 explored direct-node SLSQP optimizations restarting from run 73/79/80 (see `native_evidence/two_window_fit_9967_78` through `81`). All terminated at bounds or iteration limits with continuity defects remaining between 3.99e-4 and 1.61e-3, leaving candidates rejected against the 25/35 mm gates. Continuations only reshaped the last 0.1 s; error growth 0.4â€“0.75 s remained unchanged from run 73. Archived rotation-chart audits confirm Jacobian conditioning issues. See native evidence for archived raw ZIPs.

The archived run73 sampled rotation-chart audit gives peak condition2.680/17.886/
2.889 for hip/left/right shoulder. These samples do not bound between-sample
extrema or explain the full7.45e7 trajectory-control Jacobian condition by
coordinate maps alone. See native_evidence/returned73_chart_audit.

## Ordered Next Work

Use [Next Agent Execution Plan](simscape_tour_matching/NEXT_AGENT_CONVERGENCE_EXECUTION.md)
for the existing-provider, two-window derivative preflight now that fit73 is terminal.
It specifies integrated nodes, retraction/continuity chain-rule checks and bounded
SLSQP acceptance without recreating the solver or accepting state-reset motion.

1. Decide, one receipt each, on rebalancing terminal/early weighting (early RMS
   and pelvis yaw regress under the100x terminal weight), relaxing the Â±0.05 node
   box, or extending the horizon beyond0.85 s with additional integrated nodes on
   actual capture samples (same-input replay check first). Continue from exact
   returned81 with the run81 driver pattern; report early-motion and pelvis-yaw
   regressions alongside any terminal gain; compare on uninterrupted metrics.

2. Qualify the variant over further representative trajectories and against
   MATLAB R2025b before claiming full equivalence. Qualify tangent derivatives
   before using the manifold variant in an optimizer. Preserve existing scalar API.
3. Resume trajectory/control co-optimization using native constrained forward
   dynamics. Reuse native profile, sensitivity, marker and shooting providers.
   Start from integrated states, permit trajectory changes, and optimize a global
   degree-six native effort basis. Do not repeat tight static-node boxes or simply
   compress an arbitrary feedback torque trace. Existing seven-DOF bioptim tracking
   is a different physical model and cannot silently replace this native model.
4. Require uninterrupted original-state replay through1.813888889 s, all654 capture
   frames and valid observations, closure, effort audit and independent R2025b
   validation before accepting the final swing. Preserve per-marker/time residuals,
   overlay animation, coefficient/time-basis metadata and exact input hashes.
5. Implement/qualify remaining engine variants sequentially using canonical
   pose_interchange providers, without inferring dynamics parity from pose roundtrips.

## Engine and Representation Handoffs

- [Pinocchio Manifold](simscape_tour_matching/PINOCCHIO_MANIFOLD_HANDOFF.md): native
  spherical variant nq30/nv27, exact fixed transforms/inertia/weld reused. DOP853
  integrates local tangent coordinates and carries physical endpoints unchanged;
  no quaternion projection or target-state reset. Runtime18 passes96 tests with
  one optional MuJoCo skip; root independently verified generic and real Pin tests.
  Remote receipts qualify the numerical runtime bundle, not full GUI/application
  deployment; retain the documented package boundaries when reproducing.
- [Canonical Representations](simscape_tour_matching/REPRESENTATION_HANDOFF.md):
  angle/quaternion/rate/convective-acceleration/effort conversions and fixed-frame
  transport. Preserve units, named frames, branch/winding and model identity.
  Moving-frame acceleration transport and #8867 convention consolidation remain open;
  #8867 is a Cluster C residual in the
  [adversarial review remediation ledger](adversarial_review_remediation_9410.md)
  (epic #9410, 34 of 61 children landed on `main` at `db4fe88c4`).
- [MuJoCo](mujoco_native_matching/HANDOFF.md) and
  [Drake](drake_native_matching/HANDOFF.md): native adapters and narrower receipts
  exist; alternate quaternion builders/full trajectory equivalence remain open.
  Drake's projected accelerations are not derivative-consistent trajectories.
- OpenSim epic #10003: plan/handoff branch docs/10003-opensim-matching-epic,
  documented plan commit1a68091b6. Implementation/runtime qualification is not
  accepted. Inspect its branch and requested planning check-in before proceeding.
- Local branch triage #9162 (branch conductor/issue-9162, DL-#9162): the
  disposition ledger and bundle-first deletion runbook live in
  [branch_triage_9162.md](branch_triage_9162.md). The runbook has not been
  executed; no local branch has been deleted. Nothing in it touches matching.

The documented development-log validator path is absent in this checkout; no
validator pass is claimed. Normal configured commit/push hooks still apply.

## Evidence and Reproduction

Evidence root: simscape_tour_matching/native_evidence. Preserve raw ZIP archives. Original native model SHA256 `b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`. Run19 canonical candidate SHA256 `b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f` (rejected 0.85 s fit).

ControlTower: ssh alias controltower; WSL ControlTower-Runner. Raw run receipts identify exact archived source and inputs. Never overwrite runs.

[Convergence Review](simscape_tour_matching/CONVERGENCE_REVIEW_20260912.md) gives strategy and delegation gates. [Historical Handoff](HANDOFF_HISTORY_20260912.md) preserves earlier matching history. Update this concise handoff and DEVELOPMENT_LOG with each commit.

## Change Log

- 2026-09-22T15:20:00Z — Consolidated the four drifted `GolfSwingVisualizer.m`
  copies into one fleet-shared `+golfviz` package class; both launchers wire the
  shared path; PR [#10715](https://github.com/D-sorganization/UpstreamDrift/pull/10715)
  open (DL-#9225, issue #9225). Commit f33b40c9c.

- 2026-09-22T16:30:00Z — Rematch CO-10 survivor #10720 onto trunk after #10718 baseline; keep runnable saved-job + architecture split; do not reopen #10719; do not touch NM-06. Commit SELF.
- 2026-09-22T16:15:00Z — Succession handoff PR #10721: CO-10 #10718 and NM-06 #10709 merged; do not steal NM-07 #10622. Commit SELF.
- 2026-09-22T15:00:00Z — NM-06 #10709: rematch onto origin/main after CO-09/#10650 landed; keep proposal_shared DRY helpers. Commit SELF.
- 2026-09-22T14:35:07Z — CI remediation for #10650: joint_panel unit tests now
  construct a module-scoped offscreen QApplication (autouse qapp fixture,
  mirroring starting_pose_matcher/test_joint_slider_panel.py) - constructing
  JointPanel without a QApplication aborts the process (Qt qFatal,
  Windows exit 9 / 0xC0000409), so the rewritten real-Qt assertions never ran.
  Focused check: QT_QPA_PLATFORM=offscreen python -m pytest
  tests/unit/tools/pose_studio/test_joint_panel.py -q (13 passed). Commit SELF.
- 2026-09-22T15:20:00Z — NM-06 #10709: regenerate divergence inventory for NM-06 paths on rematched tip; keep proposal_shared DRY. Commit SELF.
- 2026-09-22T14:20:00Z — NM-06 #10709: extract proposal_shared helpers to clear DRY duplication gate (no baseline raise). Commit SELF.
- 2026-09-22T14:00:00Z — NM-06 #10709: restore regressor TrainingConfig + epoch helpers under architecture budgets (prefer LoD split over net-zero gate bypass). Commit SELF.
- 2026-09-22T13:45:00Z — NM-06 #10709: rematch main; lazy torch inverse exports; architecture + unit-gate fixes. Commit SELF.
- 2026-09-22T12:40:00Z — Rematch #10684 onto origin/main after CO-06 #10687 merge (`f191dd09d`); kept MS-14 + CO-06 HANDOFF/DL rows; regenerated matched_swing README + divergence inventory; blocked G1 honesty preserved. Commit SELF.
- 2026-09-22T07:15:00Z — Rematch #10684 onto origin/main after MS-51 #10685 merge; kept MS-14 + MS-51 DL/HANDOFF rows; regenerated matched_swing status README. Commit SELF.
- 2026-09-22T03:50:00Z — Fix unit-test-gate on #10684: divergence inventory, ledger (103→106 receipts), MS-14 lane receipt excluded from ground-support schema scan. Commit SELF.
- 2026-09-21T21:10:00Z — Refresh matched_swing ledger (101 receipts) for #10660 unit-test-gate freshness. Commit SELF.
- 2026-09-21T20:42:00Z — Fix architecture budget on #10660: ShootingFitConfig and dynamics artifact helper. Commit SELF.
- 2026-09-21T20:25:00Z — Restore finite-bounds gate for minimize.least_squares on #10660; tip includes main MS-52. Commit SELF.
- 2026-09-21T10:12:34Z — CI remediation for #10663: replace BLE001 noqa catch-alls in preflight capacity checks with concrete exception tuples. Commit SELF.
- 2026-09-21T10:11:03Z — CI remediation for #10660: FakePlant accepts `ik_backend`; refreshed `reports/matched_swing_ledger.json` to 101 receipts. Commit SELF.
- 2026-09-21T10:40:00Z — Fix unit-test-gate on #10660: regenerate matched_swing status (99→100) and stop hardcoding receipt count in browser model test. Commit SELF.
